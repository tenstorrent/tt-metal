# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Complete chunk64 adapter: frozen native triangular loop versus fused recurrence.

Reuses the frozen canonical operand/oracle harness, the complete standalone
8cda graph, and the measured recurrence component. Only the triangular loop is
replaced in a new subclass. Device use and timing require --run-device.
"""
import argparse
import ast
import copy
import hashlib
import importlib.machinery
import importlib.util
import json
import os
import statistics
import sys
import time
import traceback
from pathlib import Path

from models.autoports.qwen_qwen3_6_35b_a3b.tests import v6_triangular_recurrence_fusion_probe as component
from models.autoports.qwen_qwen3_6_35b_a3b.tests.frozen_source_compatibility import raw_sha256, verify_source

COMPONENT_SHA = "8229068839e3e06750353eb66fe720c45445e197b981d982dba1328ab5de4853"
OUTPUT_HARNESS_SHA = "4c17aca9c01f3e19ed60b99a5f4880bf7a83053713778316c19cba781737b5d6"


def patched_state_ast():
    assert component.sha(component.BASE) == component.BASE_SHA
    tree = ast.parse(component.BASE.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Chunk64StateGraph")
    original = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__call__")
    method = copy.deepcopy(original)
    old = []
    replacement = ast.parse('a = self.triangular_recurrence(a)\nif observer:\n observer(prefix + "A_row63",a)\n').body

    class Replace(ast.NodeTransformer):
        def visit_For(self, node):
            if ast.unparse(node.target) == "i" and ast.unparse(node.iter) == "range(1, 64)":
                old.append(copy.deepcopy(node))
                return copy.deepcopy(replacement)
            return self.generic_visit(node)

    method = Replace().visit(method)
    assert len(old) == 1
    restored = copy.deepcopy(method)

    class Restore(ast.NodeTransformer):
        def visit_Assign(self, node):
            if ast.dump(node) == ast.dump(replacement[0]):
                return copy.deepcopy(old[0])
            return node

        def visit_If(self, node):
            if ast.dump(node) == ast.dump(replacement[1]):
                return None
            return self.generic_visit(node)

    restored = Restore().visit(restored)
    assert ast.dump(restored) == ast.dump(original)
    module = ast.fix_missing_locations(ast.Module([method], type_ignores=[]))
    proof = dict(
        restoration_exact=True,
        original_state_call_ast_sha256=hashlib.sha256(ast.dump(original).encode()).hexdigest(),
        patched_state_call_ast_sha256=hashlib.sha256(ast.dump(method).encode()).hexdigest(),
        scope="Only63row loop replaced; observer retains A_row63. Native A_row1/2/31 covered separately, not recomputed in candidate runtime.",
    )
    return module, proof


def load_graphs(t, mesh, upload):
    loader = importlib.machinery.SourceFileLoader("_v6_triangular_frozen_graph", str(component.BASE))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    patched, _ = patched_state_ast()
    scope = dict(vars(module))
    exec(compile(patched, str(__file__) + ":state_call", "exec"), scope)

    class Candidate(module.Chunk64OutputGraph):
        state_call = scope["__call__"]

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.triangular_recurrence = component.TriangularRecurrence(self.t, self.mesh)

    baseline = module.Chunk64OutputGraph(t, mesh, upload, variant="sfpu_fma", cumsum="compensated")
    candidate = Candidate(t, mesh, upload, variant="sfpu_fma", cumsum="compensated")
    assert Candidate.__call__ is module.Chunk64OutputGraph.__call__
    return baseline, candidate


def bits(a, b):
    import torch

    assert a.shape == b.shape and a.dtype == b.dtype
    dtype = torch.int32 if a.dtype == torch.float32 else torch.int16
    return int((a.contiguous().view(dtype) != b.contiguous().view(dtype)).sum())


def timing(t, mesh, graph, inputs, expected, args):
    from models.autoports.qwen_qwen3_6_35b_a3b.tests.runtime_audit import device_only

    for _ in range(args.warmup):
        with device_only():
            warm = graph(**inputs)
    t.synchronize_device(mesh)
    samples = []
    for _ in range(args.samples):
        t.synchronize_device(mesh)
        start = time.perf_counter()
        for _ in range(args.iterations):
            with device_only():
                out = graph(**inputs)
        t.synchronize_device(mesh)
        samples.append((time.perf_counter() - start) * 1000 / args.iterations)
    eager_bits = [bits(x, t.to_torch(y)) for x, y in zip(expected, out)]
    assert eager_bits == [0, 0, 0]
    result = dict(warmed_ms=samples, warmed_median_ms=statistics.median(samples), warmed_last_bits=eager_bits)
    if args.trace_replays:
        tid = None
        try:
            with device_only():
                tid = t.begin_trace_capture(mesh, cq_id=0)
                captured = graph(**inputs)
                t.end_trace_capture(mesh, tid, cq_id=0)
            t.synchronize_device(mesh)
            t.execute_trace(mesh, tid, cq_id=0, blocking=True)
            first = [t.to_torch(x).clone() for x in captured]
            first_bits = [bits(x, y) for x, y in zip(expected, first)]
            assert first_bits == [0, 0, 0]
            samples = []
            for _ in range(args.samples):
                t.synchronize_device(mesh)
                start = time.perf_counter()
                for _ in range(args.trace_replays):
                    t.execute_trace(mesh, tid, cq_id=0, blocking=False)
                t.synchronize_device(mesh)
                samples.append((time.perf_counter() - start) * 1000 / args.trace_replays)
            last = [t.to_torch(x).clone() for x in captured]
            last_bits = [bits(x, y) for x, y in zip(expected, last)]
            deterministic = [bits(x, y) for x, y in zip(first, last)]
            assert last_bits == deterministic == [0, 0, 0]
            result.update(
                traced_ms=samples,
                traced_median_ms=statistics.median(samples),
                first_bits=first_bits,
                last_bits=last_bits,
                first_last_bits=deterministic,
            )
        finally:
            if tid is not None:
                t.release_trace(mesh, tid)
    return result


def run(args):
    assert not args.output.exists()
    assert verify_source(component.__file__, COMPONENT_SHA)
    _, proof = patched_state_ast()
    import torch

    from models.autoports.qwen_qwen3_6_35b_a3b.tests import v6_chunk64_output_probe as harness

    assert verify_source(harness.__file__, OUTPUT_HARNESS_SHA)
    torch.set_num_threads(4)
    # Existing CPU harness is unmodified; it recreates canonical real prefix operands.
    result, payload = harness.cpu_setup(args)
    result.update(
        status="cpu_setup_pass",
        source_sha256=component.sha(__file__),
        base_sha256=component.BASE_SHA,
        component_sha256=raw_sha256(component.__file__),
        component_historical_sha256=COMPONENT_SHA,
        output_harness_sha256=raw_sha256(harness.__file__),
        output_harness_historical_sha256=OUTPUT_HARNESS_SHA,
        ast_proof=proof,
        classification="Complete same-operand chunk64 adapter correctness/timing diagnostic; no decoder acceptance claim",
        device_executed=args.run_device,
        device_closed=False,
        adapter_cases=[],
        timing={},
        oracle_assisted=False,
        acceptance_eligible=False,
    )
    args.output.with_suffix(".source.py.snapshot").write_bytes(Path(__file__).read_bytes())
    args.output.with_suffix(".state.py.snapshot").write_text(ast.unparse(patched_state_ast()[0]) + "\n")
    payload_path = args.output.with_suffix(".cpu.pt")
    torch.save(payload, payload_path)
    result.update(cpu_payload=str(payload_path), cpu_payload_sha256=component.sha(payload_path))

    def save():
        args.output.write_text(json.dumps(result, indent=2) + "\n")

    save()
    if not args.run_device:
        assert "ttnn" not in sys.modules
        print("TRIANGULAR_FULL_OUTPUT_CPU_PASS", component.sha(__file__))
        return
    for variable in ("TT_METAL_WATCHER", "TT_METAL_DEVICE_PROFILER"):
        assert os.environ.get(variable) in (None, "", "0"), variable
    import ttnn as t
    from models.autoports.qwen_qwen3_6_35b_a3b.tests.runtime_audit import device_only

    evidence = component.E / "v6_triangular_recurrence_fusion_all_device.json"
    checked = json.loads(evidence.read_text())
    assert checked["status"] == "component_exact" and checked["device_closed"]
    assert len(checked["cases"]) == 48 and all(row["bits_different"] == 0 for row in checked["cases"])
    result["component_device_evidence"] = dict(path=str(evidence), sha256=component.sha(evidence))
    mesh = None
    try:
        mesh = t.open_mesh_device(
            t.MeshShape(1, 1),
            physical_device_ids=[0],
            l1_small_size=32768,
            trace_region_size=args.trace_region_size if args.trace_replays else 0,
        )
        upload = lambda x: t.from_torch(
            x.contiguous(), device=mesh, dtype=t.float32, layout=t.TILE_LAYOUT, memory_config=t.DRAM_MEMORY_CONFIG
        )
        native, candidate = load_graphs(t, mesh, upload)
        observed = {}
        for n in args.lengths:
            item = payload["cases"][n]
            inputs = {key: upload(value) for key, value in item["operands"].items()}
            inputs.update(query=upload(item["query"]), initial=upload(payload["initial"]))
            outputs = {}
            intermediate = {}
            for label, graph in [("native", native), ("candidate", candidate)]:
                handles = {}

                # Observer records device handles only; no host calculation in guarded runtime.
                def observe(name, tensor):
                    handles[name] = tensor

                with device_only():
                    out = graph(**inputs, observer=observe)
                t.synchronize_device(mesh)
                outputs[label] = [t.to_torch(x).clone() for x in out]
                intermediate[label] = {key: t.to_torch(value).clone() for key, value in handles.items()}
                handles.clear()
            a, b = outputs["native"], outputs["candidate"]
            comparisons = {name: bits(x, y) for name, x, y in zip(("output_bf16", "state_fp32", "output_fp32"), a, b)}
            hf = [item["canonical_output"], item["canonical"], item["canonical_output_fp32"]]
            hf_comparisons = {
                name: bits(x, y) for name, x, y in zip(("output_bf16", "state_fp32", "output_fp32"), hf, b)
            }
            common = set(intermediate["native"]) & set(intermediate["candidate"])
            stages = {key: bits(intermediate["native"][key], intermediate["candidate"][key]) for key in sorted(common)}
            row = dict(
                length=n,
                vs_native_bits=comparisons,
                vs_hf_bits=hf_comparisons,
                common_intermediate_bits=stages,
                omitted_candidate_observers=sorted(set(intermediate["native"]) - common),
            )
            result["adapter_cases"].append(row)
            observed[n] = dict(outputs=outputs, intermediates=intermediate)
            save()
            print(json.dumps(row), flush=True)
            assert all(v == 0 for v in comparisons.values()) and all(v == 0 for v in stages.values()), row
            # The established graph acceptance on these canonical operands is bit exact.
            assert all(v == 0 for v in hf_comparisons.values()), row
            if args.timing and n == args.timing_length:
                for label, graph in [("native", native), ("candidate", candidate)]:
                    result["timing"][label] = timing(t, mesh, graph, inputs, a, args)
                    save()
        path = args.output.with_suffix(".device.pt")
        torch.save(observed, path)
        result.update(
            status="complete_adapter_exact", device_tensors=str(path), device_tensors_sha256=component.sha(path)
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
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--run-device", action="store_true")
    p.add_argument("--lengths", type=int, nargs="+", default=[128])
    p.add_argument("--initial-state", choices=["zero", "prefix64"], default="zero")
    p.add_argument("--variant", default="sfpu_fma", choices=["sfpu_fma"])
    p.add_argument("--sum-order", default="canonical", choices=["canonical"])
    p.add_argument("--cumsum", default="compensated", choices=["compensated"])
    p.add_argument("--timing", action="store_true")
    p.add_argument("--timing-length", type=int, default=128)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--iterations", type=int, default=3)
    p.add_argument("--samples", type=int, default=3)
    p.add_argument("--trace-replays", type=int, default=0)
    p.add_argument("--trace-region-size", type=int, default=67108864)
    args = p.parse_args()
    assert not args.timing or args.timing_length in args.lengths
    run(args)


if __name__ == "__main__":
    main()
