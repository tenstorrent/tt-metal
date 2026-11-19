# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Product+contraction versus selected product plus accepted direct sum.

Parent-only serialized hardware run. Full FP32 operand and reduction semantics
remain; no production installation. Both scaled-Q and plain-K contractions,
both output layouts, changed-input traces, four rank-distinct inputs.
"""
import argparse
import hashlib
import json
import os
import sys
import time
import traceback
from pathlib import Path

import torch


def fixtures(root):
    cases, provenance = [], []
    for pos in (128, 8157):
        path = root / f"doc/fused_decoder/endurance_precise_probe.json.pos{pos}.pt"
        saved = torch.load(path, map_location="cpu", weights_only=False)["tt_values"]
        provenance.append(dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        for rounded in (False, True):
            state = saved["delta_state_input"]
            if rounded:
                state = state.bfloat16().float()
            for key in ("q", "k"):
                col = saved["delta_" + key].permute(0, 2, 3, 1).contiguous()
                cases.append((f"real_pos{pos}_round{rounded}_{key}", state.contiguous(), col))
    gen = torch.Generator().manual_seed(2026111603)
    cases.append(
        ("random_rank_distinct", torch.randn(1, 32, 128, 128, generator=gen), torch.randn(1, 16, 128, 1, generator=gen))
    )
    cancellation = torch.zeros(1, 32, 128, 128)
    cancellation[:, :, 0::16] = 2**24
    cancellation[:, :, 1::16] = 1
    cancellation[:, :, 2::16] = -(2**24)
    cancellation[:, :, 3::16] = 1
    cancellation *= torch.arange(1, 33).reshape(1, 32, 1, 1)
    cases.append(("cancellation_rank_distinct", cancellation, torch.ones(1, 16, 128, 1)))
    signed = torch.zeros_like(cancellation)
    signed[:, 0::2] = -0.0
    cases.append(("signed_zero", signed, torch.ones(1, 16, 128, 1)))
    return cases, provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("acceptance", "timing", "cpu"), default="acceptance")
    parser.add_argument("--state-memory", choices=("l1", "dram"), default="dram")
    parser.add_argument("--column-memory", choices=("l1", "dram"), default="dram")
    parser.add_argument("--iterations", type=int, default=100)
    args = parser.parse_args()
    torch.set_num_threads(4)
    root = Path(__file__).resolve().parents[1]
    record = dict(
        passed=False,
        command=sys.argv,
        mode=args.mode,
        state_memory=args.state_memory,
        column_memory=args.column_memory,
        checks=[],
        timings={},
        scope="component only, no full-model acceptance",
        source_sha256={
            str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (
                Path(__file__),
                root / "tt/product_recurrent_sum_probe.py",
                root / "tt/direct_recurrent_sum_probe.py",
            )
        },
    )
    mesh, traces = None, {}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        cases, record["fixture_provenance"] = fixtures(root)
        assert all(torch.isfinite(a).all() and torch.isfinite(b).all() for _, a, b in cases)
        record["fixture_names"] = [name for name, _, _ in cases]
        if args.mode == "cpu":
            record.update(passed=True, hardware_verified=False)
            return
        import ttnn as t
        from models.autoports.qwen_qwen3_6_35b_a3b.tt.direct_recurrent_sum_probe import DirectRecurrentSum
        from models.autoports.qwen_qwen3_6_35b_a3b.tt.product_recurrent_sum_probe import ProductRecurrentSum

        assert os.environ.get("TT_METAL_TRACE_ALLOC_TRACKING") == ("1" if args.mode == "acceptance" else "0")
        assert os.environ.get("TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE") == "0"
        if args.mode == "timing":
            assert not os.environ.get("TT_METAL_WATCHER") and not os.environ.get("TT_METAL_DEVICE_PROFILER")
        t.set_fabric_config(t.FabricConfig.FABRIC_1D)
        mesh = t.open_mesh_device(t.MeshShape(1, 4), trace_region_size=200000000)
        memory = {"l1": t.L1_MEMORY_CONFIG, "dram": t.DRAM_MEMORY_CONFIG}

        def upload(x, mem=None):
            kw = dict(dtype=t.float32, layout=t.TILE_LAYOUT, mesh_mapper=t.ShardTensorToMesh(mesh, dim=1))
            if mem is not None:
                kw.update(device=mesh, memory_config=memory[mem])
            return t.from_torch(x, **kw)

        def host(x):
            return [t.to_torch(rank).contiguous() for rank in t.get_device_tensors(x)]

        def compare(name, value, expected):
            actual = host(value)
            assert len(actual) == len(expected) == 4
            for rank, (a, b) in enumerate(zip(actual, expected)):
                assert a.shape == b.shape and a.dtype == b.dtype == torch.float32
                count = int((a.view(torch.int32) != b.view(torch.int32)).sum())
                record["checks"].append(
                    dict(name=name, rank=rank, differences=count, max_abs=float((a - b).abs().max()))
                )
                assert count == 0, (name, rank, count)

        state = upload(cases[0][1], args.state_memory)
        column = upload(cases[0][2], args.column_memory)
        base, engine = DirectRecurrentSum(t, mesh), ProductRecurrentSum(t, mesh)

        def old(scaled, terminal):
            product = t.multiply(
                t.reshape(state, (1, 4, 2, 128, 128)),
                t.reshape(column, (1, 4, 1, 128, 1)),
                input_tensor_b_activations=[t.UnaryWithParam(t.UnaryOpType.MUL_UNARY_SFPU, 128**-0.5)]
                if scaled
                else [],
                memory_config=t.L1_MEMORY_CONFIG,
            )
            return base(t.reshape(product, (1, 8, 128, 128)), terminal_rows=terminal)

        functions = {}
        for scaled in (False, True):
            for terminal in (False, True):
                suffix = f"scale{int(scaled)}_terminal{int(terminal)}"
                functions["old_" + suffix] = lambda s=scaled, r=terminal: old(s, r)
                functions["new_" + suffix] = lambda s=scaled, r=terminal: engine(
                    state, column, scale_q=s, terminal_rows=r
                )
        outputs = {}
        for name, fn in functions.items():
            value = fn()
            outputs[name] = t.empty_like(value)
            t.copy(value, outputs[name])
            host(outputs[name])
        t.synchronize_device(mesh)
        for name, fn in functions.items():
            trace = t.begin_trace_capture(mesh, cq_id=0)
            try:
                t.copy(fn(), outputs[name])
            finally:
                t.end_trace_capture(mesh, trace, cq_id=0)
                traces[name] = trace
        entries = mesh.num_program_cache_entries()
        for fixture, a, b in cases:
            t.copy_host_to_device_tensor(upload(a), state)
            t.copy_host_to_device_tensor(upload(b), column)
            for scaled in (False, True):
                for terminal in (False, True):
                    suffix = f"scale{int(scaled)}_terminal{int(terminal)}"
                    expected = host(old(scaled, terminal))
                    compare(fixture + "/eager_" + suffix, functions["new_" + suffix](), expected)
                    for prefix in ("old_", "new_"):
                        name = prefix + suffix
                        t.execute_trace(mesh, traces[name], cq_id=0, blocking=False)
                        compare(fixture + "/trace_" + name, outputs[name], expected)
        assert mesh.num_program_cache_entries() == entries
        t.copy_host_to_device_tensor(upload(cases[0][1]), state)
        t.copy_host_to_device_tensor(upload(cases[0][2]), column)
        record["timing_fixture"] = cases[0][0]
        for name, trace in traces.items():
            for _ in range(3):
                t.execute_trace(mesh, trace, cq_id=0, blocking=False)
            t.synchronize_device(mesh)
            start = time.perf_counter()
            for _ in range(args.iterations):
                t.execute_trace(mesh, trace, cq_id=0, blocking=False)
            t.synchronize_device(mesh)
            record["timings"][name] = (time.perf_counter() - start) * 1000 / args.iterations
        record.update(
            passed=True, hardware_verified=True, timing_eligible=args.mode == "timing", program_cache_entries=entries
        )
    except Exception:
        record["error"] = traceback.format_exc()
        raise
    finally:
        args.output.write_text(json.dumps(record, indent=2) + "\n")
        if mesh is not None:
            import ttnn as t

            for trace in traces.values():
                t.release_trace(mesh, trace)
            t.close_mesh_device(mesh)
        print(
            json.dumps(dict(passed=record["passed"], checks=len(record["checks"]), timings=record["timings"])),
            flush=True,
        )


if __name__ == "__main__":
    main()
