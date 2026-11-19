# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Complete decode convolution adapter timing after the strict component gate.

Only parent-owned hardware opt-in executes TTNN. Mutable carry is warmed into a
fixed-input steady state before tracing; changing-input correctness is separate.
"""
import argparse
import ast
import hashlib
import importlib.util
import json
import os
import statistics
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

COMPONENT = Path(__file__).with_name("v6_decode_conv_mixed_transport_probe.py")
COMPONENT_SHA = "4f1e2a409a889c40f2f15d0edbe8492f505049058f3152d9f7becab61f59d44f"
E = Path(__file__).resolve().parents[1] / "doc/fused_decoder/endurance"
CASES = "actual3271,actual227,actual738,actual5799,random,zeros,finite,tiny"


def load_component():
    assert hashlib.sha256(COMPONENT.read_bytes()).hexdigest() == COMPONENT_SHA
    spec = importlib.util.spec_from_file_location("frozen_decode_conv_component", COMPONENT)
    component = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(component)
    return component


def unobserved_native(component, t):
    """Remove only diagnostic accumulator retention from the extracted oracle."""
    _, native_sha = component.native_function(None)
    tree = ast.parse(component.BASE.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "FusedDecoder")
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_linear_attention")
    start = next(
        i
        for i, n in enumerate(method.body)
        if isinstance(n, ast.Assign) and any(isinstance(x, ast.Name) and x.id == "history" for x in n.targets)
    )
    end = next(
        i
        for i, n in enumerate(method.body[start:], start)
        if isinstance(n, ast.Expr)
        and isinstance(n.value, ast.Call)
        and ast.unparse(n.value.func) == "ttnn.slice"
        and any(k.arg == "output_tensor" for k in n.value.keywords)
    )
    block = method.body[start : end + 1]
    assert hashlib.sha256(ast.dump(ast.Module(block, type_ignores=[])).encode()).hexdigest() == native_sha
    fn = ast.parse("def convolution(self,qkv,state):\n s=1\n return conv,(q,k,v)\n").body[0]
    fn.body = fn.body[:1] + block + fn.body[-1:]
    module = ast.fix_missing_locations(ast.Module([fn], type_ignores=[]))
    scope = {"ttnn": t}
    exec(compile(module, str(component.BASE) + ":unobserved_conv", "exec"), scope)
    return scope["convolution"], native_sha


def check_gate(path, component, fixture_names):
    gate = json.loads(path.read_text())
    assert gate["source_sha256"] == COMPONENT_SHA
    assert gate["status"] == "component_exact" and gate["device_closed"] is True
    assert set(gate["fixture_names"]) == set(fixture_names)
    assert len(gate["cases"]) == len(fixture_names)
    for row in gate["cases"]:
        assert row["output"]["bits_different"] == row["carry"]["bits_different"] == 0
        assert len(row["partial"]) == 4 and all(p["bits_different"] == 0 for p in row["partial"])
    return dict(path=str(path), sha256=component.sha(path), status=gate["status"])


def run(args):
    assert not args.output.exists()
    assert args.warmup >= 4 and args.replays >= 100 and args.samples >= 3
    component = load_component()
    weights, fixtures, result = component.prepare(args)
    _, native_sha = unobserved_native(component, None)
    result.update(
        source_sha256=component.sha(__file__),
        component_sha256=COMPONENT_SHA,
        native_conv_ast_sha256=native_sha,
        status="cpu_setup_pass",
        timing=[],
        dynamic_sequence=[],
        gate_required=str(args.component_gate),
        measured_native_instrumentation="none; original block AST hash identical",
        measured_candidate_debug=False,
        sequence_contract="Synthetic nonchronological saved/random QKV sequence with one evolving carry; same inputs for both paths.",
        trace_state_contract="Fixed QKV; at least four eager warmups replace all three carry rows before capture; steady output/carry checked before and after every variant.",
    )
    args.output.with_suffix(".source.py.snapshot").write_bytes(Path(__file__).read_bytes())

    def save():
        args.output.write_text(json.dumps(result, indent=2) + "\n")

    save()
    if not args.run_device:
        assert "ttnn" not in sys.modules
        print("DECODE_CONV_TIMING_CPU_SETUP_PASS", result["source_sha256"])
        return
    result["component_gate"] = check_gate(args.component_gate, component, result["fixture_names"])
    for variable in ("TT_METAL_WATCHER", "TT_METAL_DEVICE_PROFILER"):
        assert os.environ.get(variable) in (None, "", "0"), variable
    from ttnn.tools import trace_allocation_tracker

    import ttnn as t
    from models.autoports.qwen_qwen3_6_35b_a3b.tests.runtime_audit import device_only

    mesh = None
    try:
        mesh = t.open_mesh_device(
            t.MeshShape(1, 1), physical_device_ids=[0], l1_small_size=32768, trace_region_size=args.trace_region_size
        )

        def upload(x, dtype=t.bfloat16, layout=t.TILE_LAYOUT, memory=t.DRAM_MEMORY_CONFIG):
            return t.from_torch(x.contiguous(), device=mesh, dtype=dtype, layout=layout, memory_config=memory)

        native, _ = unobserved_native(component, t)
        narrow = t.UnaryWithParam(t.UnaryOpType.TYPECAST, float(t.float32.value), float(t.bfloat16.value))
        widen = t.UnaryWithParam(t.UnaryOpType.TYPECAST, float(t.bfloat16.value), float(t.float32.value))
        adapters = {}
        for layer, weight in weights.items():
            taps = [upload(weight[:, 0, j].float().reshape(1, 1, 8192), t.float32) for j in range(4)]
            adapters[layer] = (
                SimpleNamespace(
                    taps_fp32=taps, conv_post_chain=[narrow, widen, t.UnaryWithParam(t.UnaryOpType.SILU), narrow]
                ),
                component.DecodeConv(t, mesh, taps, args.max_cores),
            )

        def adapter(variant, layer, qkv, carry):
            obj, candidate = adapters[layer]
            if variant == "native":
                return native(obj, qkv, {"conv": carry})
            output, partial = candidate(qkv, carry, debug=False)
            assert partial is None
            return output, (output[:, :, :2048], output[:, :, 2048:4096], output[:, :, 4096:])

        def read(output, carry):
            return [
                t.to_torch(output[0]).clone(),
                *[t.to_torch(x).clone() for x in output[1]],
                t.to_torch(carry).clone(),
            ]

        def exact(reference, observed):
            checks = [component.compare(a, b) for a, b in zip(reference, observed)]
            assert len(reference) == len(observed) == 5
            assert all(c["bits_different"] == 0 for c in checks), checks
            return checks

        fixture = next(f for f in fixtures if f["name"] == args.timing_case)
        layer = fixture["layer"]
        sequence = [f for f in fixtures if f["layer"] == layer and not f["name"].startswith(("finite", "tiny"))]
        sequence_inputs = [upload(f["qkv"], memory=t.L1_MEMORY_CONFIG) for f in sequence]
        oldcarry = upload(fixture["carry"], layout=t.ROW_MAJOR_LAYOUT)
        newcarry = upload(fixture["carry"], layout=t.ROW_MAJOR_LAYOUT)
        for iteration in range(3):
            for f, qkv in zip(sequence, sequence_inputs):
                with device_only():
                    old = adapter("native", layer, qkv, oldcarry)
                    new = adapter("fused", layer, qkv, newcarry)
                checks = exact(read(old, oldcarry), read(new, newcarry))
                result["dynamic_sequence"].append(dict(iteration=iteration, input=f["name"], checks=checks))
                del old, new
        del oldcarry, newcarry, sequence_inputs
        save()
        qkv = upload(fixture["qkv"], memory=t.L1_MEMORY_CONFIG)
        reference_carry = upload(fixture["carry"], layout=t.ROW_MAJOR_LAYOUT)
        for _ in range(args.warmup):
            with device_only():
                reference_output = adapter("native", layer, qkv, reference_carry)
        reference = read(reference_output, reference_carry)
        del reference_output, reference_carry
        variants = ("native", "fused") if args.order == "native-first" else ("fused", "native")
        for variant in variants:
            carry = upload(fixture["carry"], layout=t.ROW_MAJOR_LAYOUT)
            for _ in range(args.warmup):
                with device_only():
                    output = adapter(variant, layer, qkv, carry)
                t.synchronize_device(mesh)
            warmed_check = exact(reference, read(output, carry))
            del output
            eager_samples = []
            for _ in range(args.samples):
                t.synchronize_device(mesh)
                start = time.perf_counter()
                for _ in range(args.eager_calls):
                    with device_only():
                        output = adapter(variant, layer, qkv, carry)
                t.synchronize_device(mesh)
                eager_samples.append(1000 * (time.perf_counter() - start) / args.eager_calls)
            eager_check = exact(reference, read(output, carry))
            del output
            trace_id, capturing = None, False
            try:
                trace_id = t.begin_trace_capture(mesh, cq_id=0)
                capturing = True
                with device_only():
                    output = adapter(variant, layer, qkv, carry)
                t.end_trace_capture(mesh, trace_id, cq_id=0)
                capturing = False
                for tensor in (output[0], *output[1]):
                    trace_allocation_tracker.acknowledge_corruptible(tensor)
                for _ in range(args.warmup):
                    t.execute_trace(mesh, trace_id, cq_id=0, blocking=True)
                first = read(output, carry)
                first_check = exact(reference, first)
                samples = []
                for _ in range(args.samples):
                    t.synchronize_device(mesh)
                    start = time.perf_counter()
                    for _ in range(args.replays):
                        t.execute_trace(mesh, trace_id, cq_id=0, blocking=False)
                    t.synchronize_device(mesh)
                    samples.append(1000 * (time.perf_counter() - start) / args.replays)
                last = read(output, carry)
                row = dict(
                    variant=variant,
                    warmed_ms=eager_samples,
                    warmed_median_ms=statistics.median(eager_samples),
                    traced_ms=samples,
                    traced_median_ms=statistics.median(samples),
                    replays_per_sample=args.replays,
                    warmed_check=warmed_check,
                    eager_check=eager_check,
                    first_check=first_check,
                    last_check=exact(reference, last),
                    deterministic=exact(first, last),
                    output_memory=str(output[0].memory_config()),
                    carry_memory=str(carry.memory_config()),
                )
                result["timing"].append(row)
                save()
                print(json.dumps(row), flush=True)
            finally:
                if trace_id is not None:
                    if capturing:
                        t.end_trace_capture(mesh, trace_id, cq_id=0)
                    t.synchronize_device(mesh)
                    t.release_trace(mesh, trace_id)
                del output, carry
        by_name = {row["variant"]: row for row in result["timing"]}
        result.update(
            status="completed_microbenchmark",
            improved=by_name["fused"]["traced_median_ms"] < by_name["native"]["traced_median_ms"],
        )
    except Exception:
        result.update(status="error", error=traceback.format_exc())
        raise
    finally:
        if mesh is not None:
            t.close_mesh_device(mesh)
            result["device_closed"] = True
        save()


def main():
    component = load_component()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--run-device", action="store_true")
    p.add_argument("--component-gate", type=Path, default=E / "v6_decode_conv_mixed_transport_all_device.json")
    p.add_argument("--layer", type=int, default=34)
    p.add_argument("--max-cores", type=int, default=64)
    p.add_argument("--timing-case", default="actual3271")
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--samples", type=int, default=3)
    p.add_argument("--eager-calls", type=int, default=20)
    p.add_argument("--replays", type=int, default=100)
    p.add_argument("--trace-region-size", type=int, default=32 * 1024 * 1024)
    p.add_argument("--order", choices=("native-first", "fused-first"), default="native-first")
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface"))
        / "hub/models--Qwen--Qwen3.6-35B-A3B/snapshots"
        / component.REVISION,
    )
    args = p.parse_args()
    args.cases = CASES
    assert args.eager_calls > 0
    run(args)


if __name__ == "__main__":
    main()
