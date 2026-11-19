# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Diagnostic only: persistent BF16 RM convolution cache on frozen selected v4.

Actual allocation method and actual convolution block, including stable-buffer copy.
No TTNN import/device activity in --oracle-only mode. Production is never patched.
"""

import argparse
import ast
import copy
import hashlib
import json
import os
import statistics
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

import torch

from models.autoports.qwen_qwen3_6_35b_a3b.tests.conv_first_tap_mixed_probe import compare
from models.autoports.qwen_qwen3_6_35b_a3b.tests.v4_conv_adapter_probe import DOC, REVISION, digest, load_weights

BASE_HASH = "b1633d51621b4c45d9514a66fad21faa57ad562185502342c3d87bd5ebc2701c"


def conv_function(path, ttnn):
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef))
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_linear_attention")
    start = next(
        i
        for i, n in enumerate(fn.body)
        if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "history" for t in n.targets)
    )

    def is_cache_write(n):
        if not isinstance(n, ast.Expr) or not isinstance(n.value, ast.Call):
            return False
        call = n.value
        return isinstance(call.func, ast.Attribute) and (
            call.func.attr == "copy"
            or (call.func.attr == "slice" and any(k.arg == "output_tensor" for k in call.keywords))
        )

    end = next(i for i, n in enumerate(fn.body[start:], start) if is_cache_write(n))
    extracted = ast.parse("def convolution(self, qkv, state):\n    s = qkv.shape[1]\n    return conv\n").body[0]
    extracted.body = extracted.body[:1] + copy.deepcopy(fn.body[start : end + 1]) + extracted.body[-1:]
    module = ast.Module(body=[extracted], type_ignores=[])
    ast.fix_missing_locations(module)
    scope = {"ttnn": ttnn}
    exec(compile(module, str(path) + ":convolution", "exec"), scope)
    return scope["convolution"], hashlib.sha256(ast.dump(module).encode()).hexdigest()


def allocation_function(path, ttnn):
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef))
    fn = copy.deepcopy(next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "new_recurrent_state"))
    module = ast.Module(body=[fn], type_ignores=[])
    ast.fix_missing_locations(module)
    scope = {"ttnn": ttnn, "DRAM": None if ttnn is None else ttnn.DRAM_MEMORY_CONFIG}
    exec(compile(module, str(path) + ":new_recurrent_state", "exec"), scope)
    return scope["new_recurrent_state"], hashlib.sha256(ast.dump(module).encode()).hexdigest()


def tensor_digest(x):
    return hashlib.sha256(x.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def run(args):
    torch.set_num_threads(4)
    assert args.overwrite or not args.output.exists(), f"Preserve evidence: {args.output}"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    paths = {
        "baseline": DOC / "v4_persistent_rm_cache_baseline.py.snapshot",
        "persistent_rm": DOC / "v4_persistent_rm_cache.py.snapshot",
        "persistent_rm_slice": DOC / "v4_persistent_rm_cache_slice.py.snapshot",
    }
    assert digest(paths["baseline"]) == BASE_HASH
    trees = {k: ast.parse(p.read_text()) for k, p in paths.items()}
    methods = {}
    for k, tree in trees.items():
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef))
        methods[k] = {n.name: ast.dump(n) for n in cls.body if isinstance(n, ast.FunctionDef)}
    assert methods["baseline"].keys() == methods["persistent_rm"].keys()
    assert {n for n in methods["baseline"] if methods["baseline"][n] != methods["persistent_rm"][n]} == {
        "new_recurrent_state",
        "_linear_attention",
    }
    old = paths["baseline"].read_text()
    expected = (
        old.replace(
            '"conv": zeros((1, 3, 8192), ttnn.bfloat16, ttnn.TILE_LAYOUT)',
            '"conv": zeros((1, 3, 8192), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT)',
        )
        .replace('ttnn.to_layout(state["conv"], ttnn.ROW_MAJOR_LAYOUT)', 'state["conv"]')
        .replace(
            "carry = ttnn.to_layout(history_rm[:, s : s + 3, :], ttnn.TILE_LAYOUT)",
            "carry = history_rm[:, s : s + 3, :]",
        )
    )
    assert expected == paths["persistent_rm"].read_text(), "Only the three specified layout edits are allowed"
    expected_slice = expected.replace(
        '        carry = history_rm[:, s : s + 3, :]\n        ttnn.copy(carry, state["conv"])',
        '        ttnn.slice(history_rm, (0, s, 0), (1, s + 3, 8192), output_tensor=state["conv"])',
    )
    assert expected_slice == paths["persistent_rm_slice"].read_text()
    weights = load_weights(args.checkpoint)
    generator = torch.Generator().manual_seed(args.seed)

    def random_rows(n):
        return (3 * torch.randn((1, n, 8192), generator=generator)).bfloat16()

    cases = [(f"s{s}", random_rows(s), random_rows(3)) for s in args.lengths]
    all_codes = torch.arange(65536, dtype=torch.int32).to(torch.uint16).view(torch.bfloat16)
    finite = all_codes[torch.isfinite(all_codes)]
    history = torch.cat([finite, torch.zeros(256, dtype=torch.bfloat16)]).reshape(1, 8, 8192)
    cases.append(("all_finite_bf16_history", history[:, 3:].contiguous(), history[:, :3].contiguous()))
    schedules = {
        "prefill128_decode3_continuation33_decode1": [128, 1, 1, 1, 33, 1],
        "prefill33_decode2_continuation129_decode1": [33, 1, 1, 129, 1],
    }
    chains = {name: [random_rows(s) for s in lengths] for name, lengths in schedules.items()}
    result = dict(
        classification="diagnostic_only_not_acceptance",
        status="cpu_setup_pass",
        source_sha256=digest(Path(__file__)),
        snapshots={
            k: dict(
                path=str(p),
                sha256=digest(p),
                convolution_ast_sha256=conv_function(p, None)[1],
                allocation_ast_sha256=allocation_function(p, None)[1],
            )
            for k, p in paths.items()
        },
        dependencies={
            str(Path(__file__).with_name(n)): digest(Path(__file__).with_name(n))
            for n in ("conv_first_tap_mixed_probe.py", "v4_conv_adapter_probe.py")
        },
        checkpoint=str(args.checkpoint),
        checkpoint_revision=REVISION,
        weight_sha256=tensor_digest(weights["linear_attn.conv1d.weight"]),
        seed=args.seed,
        input_description="Real layer0 checkpoint taps; seeded synthetic BF16 projected rows and exhaustive finite BF16 history. Not claimed captured model activations.",
        scope="Actual new_recurrent_state plus AST-extracted convolution through carry copy. Chains test convolution cache chronology, not full attention/decoder acceptance.",
        timing_scope="Complete convolution adapters including input/history concat, all tap slices/conversions/arithmetic/activation, QKV output slices, carry slice and stable-cache copy. Allocation/reset/readback are setup outside trace, as in production.",
        expected_cache_payload_bytes={"baseline_tile": 32 * 8192 * 2, "persistent_rm": 3 * 8192 * 2},
        expected_cache_payload_saving_bytes=29 * 8192 * 2,
        cases=[],
        chains=[],
        timing=[],
        device_exact=None,
    )
    result["cpu_cases"] = [
        dict(name=name, rows=x.shape[1], input_sha256=tensor_digest(x), seed_cache_sha256=tensor_digest(seed))
        for name, x, seed in cases
    ]
    result["cpu_chain_mapping"] = {}
    for name, xs in chains.items():
        cache = torch.zeros((1, 3, 8192), dtype=torch.bfloat16)
        full = cache.clone()
        entries = []
        for x in xs:
            cache = torch.cat([cache, x], dim=1)[:, -3:].contiguous()
            full = torch.cat([full, x], dim=1)
            check = compare(full[:, -3:], cache)
            assert check["exact"]
            entries.append(dict(rows=x.shape[1], input_sha256=tensor_digest(x), carry_mapping=check))
        result["cpu_chain_mapping"][name] = entries

    def save():
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")

    if args.oracle_only:
        save()
        print("CPU_SETUP_PASS", len(cases), "cases", len(chains), "chains")
        return
    for name in (
        "TT_METAL_WATCHER",
        "TT_METAL_DEVICE_PROFILER",
        "TT_METAL_TRACE_ALLOC_TRACKING",
        "TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE",
    ):
        assert os.environ.get(name) in (None, "", "0"), f"Clean timing requires {name} disabled"
    import ttnn
    from models.autoports.qwen_qwen3_6_35b_a3b.tests.provenance import record
    from models.autoports.qwen_qwen3_6_35b_a3b.tests.runtime_audit import device_only

    record(args.output)
    mesh = None
    try:
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), physical_device_ids=[0], trace_region_size=32 * 1024 * 1024)

        def upload(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
            return ttnn.from_torch(
                x.contiguous(), device=mesh, dtype=dtype, layout=layout, memory_config=ttnn.DRAM_MEMORY_CONFIG
            )

        narrow = ttnn.UnaryWithParam(ttnn.UnaryOpType.TYPECAST, float(ttnn.float32.value), float(ttnn.bfloat16.value))
        widen = ttnn.UnaryWithParam(ttnn.UnaryOpType.TYPECAST, float(ttnn.bfloat16.value), float(ttnn.float32.value))
        obj = SimpleNamespace(
            mesh_device=mesh,
            conv_post_chain=[narrow, widen, ttnn.UnaryWithParam(ttnn.UnaryOpType.SILU), narrow],
            taps_fp32=[
                upload(weights["linear_attn.conv1d.weight"][:, 0, j].reshape(1, 1, 8192), ttnn.float32)
                for j in range(4)
            ],
        )
        functions = {k: conv_function(p, ttnn)[0] for k, p in paths.items()}
        allocations = {k: allocation_function(p, ttnn)[0] for k, p in paths.items()}
        layouts = {v: ttnn.TILE_LAYOUT if v == "baseline" else ttnn.ROW_MAJOR_LAYOUT for v in paths}

        def identity(state):
            return (id(state["conv"]), state["conv"].buffer_address())

        def assert_state(state, variant, original):
            assert identity(state) == original, "Persistent cache object/address changed"
            assert tuple(state["conv"].shape) == (1, 3, 8192)
            assert state["conv"].dtype == ttnn.bfloat16 and state["conv"].layout == layouts[variant]

        def fresh(variant):
            state = allocations[variant](obj)
            assert_state(state, variant, identity(state))
            assert compare(torch.zeros((1, 3, 8192), dtype=torch.bfloat16), ttnn.to_torch(state["conv"]))["exact"]
            assert state["recurrent"].dtype == ttnn.float32 and state["recurrent"].layout == ttnn.TILE_LAYOUT
            assert compare(torch.zeros((1, 32, 128, 128)), ttnn.to_torch(state["recurrent"]))["exact"]
            return state

        result["status"] = "running"
        save()
        for name, host_x, host_seed in cases:
            dx = upload(host_x)
            states = {v: fresh(v) for v in paths}
            ids = {v: identity(s) for v, s in states.items()}
            saved = {}
            for v in paths:
                ttnn.copy(upload(host_seed, layout=layouts[v]), states[v]["conv"])
                saved[v] = {k: ttnn.clone(t) for k, t in states[v].items()}
                assert compare(host_seed, ttnn.to_torch(saved[v]["conv"]))["exact"]
            outputs, intermediates = {}, {}
            row = dict(name=name, rows=host_x.shape[1], carry={}, reset={}, cache={})
            for v in paths:
                seen = []
                original_addcmul, original_chain = ttnn.addcmul, ttnn.unary_chain

                def addcmul(a, b, c, **kw):
                    seen.append(b)
                    return original_addcmul(a, b, c, **kw)

                def chain(value, ops, **kw):
                    if ops is obj.conv_post_chain:
                        seen.append(value)
                    return original_chain(value, ops, **kw)

                try:
                    ttnn.addcmul, ttnn.unary_chain = addcmul, chain
                    with device_only():
                        out = functions[v](obj, dx, states[v])
                finally:
                    ttnn.addcmul, ttnn.unary_chain = original_addcmul, original_chain
                assert len(seen) == 4
                outputs[v] = ttnn.to_torch(out)
                intermediates[v] = [ttnn.to_torch(t) for t in seen]
                expected_carry = torch.cat([host_seed, host_x], dim=1)[:, -3:]
                row["carry"][v] = compare(expected_carry, ttnn.to_torch(states[v]["conv"]))
                assert_state(states[v], v, ids[v])
                row["cache"][v] = dict(
                    layout=str(states[v]["conv"].layout),
                    padded_shape=list(states[v]["conv"].padded_shape),
                    payload_bytes=int(states[v]["conv"].padded_shape[-2]) * 8192 * 2,
                    stable_object_and_address=True,
                )
                for k in saved[v]:
                    ttnn.copy(saved[v][k], states[v][k])
                assert_state(states[v], v, ids[v])
                reset_check = compare(host_seed, ttnn.to_torch(states[v]["conv"]))
                with device_only():
                    again = functions[v](obj, dx, states[v])
                row["reset"][v] = dict(cache=reset_check, output=compare(outputs[v], ttnn.to_torch(again)))
                assert compare(torch.zeros((1, 32, 128, 128)), ttnn.to_torch(states[v]["recurrent"]))["exact"]
                del seen, out, again
            row["output"] = {v: compare(outputs["baseline"], outputs[v]) for v in paths if v != "baseline"}
            row["intermediates"] = {
                v: [compare(a, b) for a, b in zip(intermediates["baseline"], intermediates[v])]
                for v in paths
                if v != "baseline"
            }
            row["intermediate_names"] = ["tap1_fp32", "tap2_fp32", "tap3_fp32", "preactivation_fp32"]
            row["exact"] = (
                all(c["exact"] for c in row["output"].values())
                and all(c["exact"] for checks in row["intermediates"].values() for c in checks)
                and all(c["exact"] for c in row["carry"].values())
                and all(c["cache"]["exact"] and c["output"]["exact"] for c in row["reset"].values())
            )
            result["cases"].append(row)
            save()
            if not row["exact"] or name == "all_finite_bf16_history":
                row["timing_skipped"] = (
                    "Different complete adapter" if not row["exact"] else "Exhaustive case is correctness only"
                )
                continue
            stable_output, stable_carry = None, None
            for v in paths:
                trace, capturing, out = None, False, None
                try:
                    for _ in range(max(4, args.warmup)):
                        with device_only():
                            out = functions[v](obj, dx, states[v])
                    ttnn.synchronize_device(mesh)
                    if v == "baseline":
                        stable_output, stable_carry = ttnn.to_torch(out), ttnn.to_torch(states[v]["conv"])
                    trace = ttnn.begin_trace_capture(mesh, cq_id=0)
                    capturing = True
                    with device_only():
                        out = functions[v](obj, dx, states[v])
                    ttnn.end_trace_capture(mesh, trace, cq_id=0)
                    capturing = False
                    for _ in range(args.warmup):
                        ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
                    first = compare(stable_output, ttnn.to_torch(out))
                    samples = []
                    for _ in range(args.samples):
                        ttnn.synchronize_device(mesh)
                        start = time.perf_counter()
                        for _ in range(args.replays):
                            ttnn.execute_trace(mesh, trace, cq_id=0, blocking=False)
                        ttnn.synchronize_device(mesh)
                        samples.append(1000 * (time.perf_counter() - start) / args.replays)
                    last = compare(stable_output, ttnn.to_torch(out))
                    carry = compare(stable_carry, ttnn.to_torch(states[v]["conv"]))
                    assert_state(states[v], v, ids[v])
                    timing = dict(
                        name=name,
                        variant=v,
                        median_ms=statistics.median(samples),
                        samples_ms=samples,
                        replays_per_sample=args.replays,
                        first_output=first,
                        last_output=last,
                        carry=carry,
                        stable_object_and_address=True,
                    )
                    result["timing"].append(timing)
                    assert first["exact"] and last["exact"] and carry["exact"]
                finally:
                    if trace is not None:
                        if capturing:
                            ttnn.end_trace_capture(mesh, trace, cq_id=0)
                        ttnn.synchronize_device(mesh)
                        ttnn.release_trace(mesh, trace)
                    out = None
                    save()
            print("CASE_COMPLETE", name, flush=True)
            del states, saved, dx

        for name, host_rows in chains.items():
            dxs = [upload(x) for x in host_rows]
            row = dict(name=name, lengths=[x.shape[1] for x in host_rows], steps=[], replay=[], postprefill_reset=[])
            result["chains"].append(row)
            states = {v: fresh(v) for v in paths}
            ids = {v: identity(s) for v, s in states.items()}
            saved = {v: {k: ttnn.clone(t) for k, t in state.items()} for v, state in states.items()}
            host_cache = torch.zeros((1, 3, 8192), dtype=torch.bfloat16)
            expected_outputs = []
            postprefill = {}
            for step, (hx, dx) in enumerate(zip(host_rows, dxs)):
                host_cache = torch.cat([host_cache, hx], dim=1)[:, -3:].contiguous()
                outputs = {}
                checks = {}
                for v in paths:
                    with device_only():
                        out = functions[v](obj, dx, states[v])
                    outputs[v] = ttnn.to_torch(out)
                    checks[v] = compare(host_cache, ttnn.to_torch(states[v]["conv"]))
                    assert_state(states[v], v, ids[v])
                    if step == 0:
                        postprefill[v] = {k: ttnn.clone(t) for k, t in states[v].items()}
                expected_outputs.append(outputs["baseline"])
                row["steps"].append(
                    dict(
                        step=step,
                        output={v: compare(outputs["baseline"], outputs[v]) for v in paths if v != "baseline"},
                        carry=checks,
                    )
                )
            # Match the acceptance harness's restore-after-prefill boundary as well.
            for v in paths:
                for k in postprefill[v]:
                    ttnn.copy(postprefill[v][k], states[v][k])
                assert compare(host_rows[0][:, -3:], ttnn.to_torch(states[v]["conv"]))["exact"]
                checks = []
                for expected_output, dx in zip(expected_outputs[1:], dxs[1:]):
                    with device_only():
                        out = functions[v](obj, dx, states[v])
                    checks.append(compare(expected_output, ttnn.to_torch(out)))
                    assert_state(states[v], v, ids[v])
                carry = compare(host_cache, ttnn.to_torch(states[v]["conv"]))
                row["postprefill_reset"].append(dict(variant=v, outputs=checks, final_carry=carry))
                save()
                assert all(c["exact"] for c in checks) and carry["exact"]
            # Reuse one captured graph after restoring the same stable cache buffer.
            # The trace contains the entire chain, no host oracle/upload/reset calls.
            for v in paths:
                trace, capturing, retained_outputs = None, False, None
                try:
                    for k in saved[v]:
                        ttnn.copy(saved[v][k], states[v][k])
                    trace = ttnn.begin_trace_capture(mesh, cq_id=0)
                    capturing = True
                    with device_only():
                        retained_outputs = [functions[v](obj, dx, states[v]) for dx in dxs]
                    ttnn.end_trace_capture(mesh, trace, cq_id=0)
                    capturing = False
                    for repeat in range(2):
                        for k in saved[v]:
                            ttnn.copy(saved[v][k], states[v][k])
                        assert compare(
                            torch.zeros((1, 3, 8192), dtype=torch.bfloat16), ttnn.to_torch(states[v]["conv"])
                        )["exact"]
                        ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
                        checks = [compare(a, ttnn.to_torch(b)) for a, b in zip(expected_outputs, retained_outputs)]
                        carry = compare(host_cache, ttnn.to_torch(states[v]["conv"]))
                        assert_state(states[v], v, ids[v])
                        row["replay"].append(
                            dict(
                                variant=v,
                                repeat=repeat,
                                outputs=checks,
                                final_carry=carry,
                                stable_object_and_address=True,
                            )
                        )
                        save()
                        assert all(c["exact"] for c in checks) and carry["exact"]
                finally:
                    if trace is not None:
                        if capturing:
                            ttnn.end_trace_capture(mesh, trace, cq_id=0)
                        ttnn.synchronize_device(mesh)
                        ttnn.release_trace(mesh, trace)
                    retained_outputs = None
            row["exact"] = all(
                all(c["exact"] for c in r["output"].values()) and all(c["exact"] for c in r["carry"].values())
                for r in row["steps"]
            )
            save()
            print("CHAIN_COMPLETE", name, flush=True)
        result["device_exact"] = all(c["exact"] for c in result["cases"] + result["chains"])
        result["status"] = "exact" if result["device_exact"] else "different"
        assert all(digest(paths[k]) == r["sha256"] for k, r in result["snapshots"].items())
    except Exception:
        result.update(status="failed", error=traceback.format_exc())
        raise
    finally:
        save()
        if mesh is not None:
            ttnn.close_mesh_device(mesh)
    print(result["status"], args.output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface"))
        / "hub/models--Qwen--Qwen3.6-35B-A3B/snapshots"
        / REVISION,
    )
    parser.add_argument("--lengths", type=int, nargs="+", default=[1, 33, 128, 129])
    parser.add_argument("--seed", type=int, default=154)
    parser.add_argument("--warmup", type=int, default=4)
    parser.add_argument("--samples", type=int, default=15)
    parser.add_argument("--replays", type=int, default=10)
    parser.add_argument("--oracle-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--output", type=Path, default=DOC / "v4_persistent_rm_cache_probe.json")
    args = parser.parse_args()
    assert min(args.lengths) > 0 and min(args.warmup, args.samples, args.replays) > 0
    run(args)


if __name__ == "__main__":
    main()
