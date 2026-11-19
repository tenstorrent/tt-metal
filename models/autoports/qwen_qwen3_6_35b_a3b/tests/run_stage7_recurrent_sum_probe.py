# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Serialized exact native-layout contraction probe; never installs production code.

Real-derived products use saved full-head recurrence operands, split into four
rank-distinct head groups. Device comparisons use the current selected TT
reduction as oracle, not the CPU reconstruction. --mode timing must run without
watcher/profiler/allocation tracking; acceptance uses tracking with no cache skip.
"""
import argparse
import hashlib
import json
import os
import sys
import time
import traceback
from pathlib import Path
from types import MethodType, SimpleNamespace

import torch


def ordered(products):
    chunks = products.reshape(1, 32, 8, 16, 128)
    partial = chunks[:, :, :, 0].clone()
    for row in range(1, 16):
        partial = partial + chunks[:, :, :, row]
    total = partial[:, :, 0].clone()
    for group in range(1, 8):
        total = total + partial[:, :, group]
    return total.unsqueeze(2)


def fixtures(root, positions):
    results, provenance = [], []
    for pos in positions:
        path = root / f"doc/fused_decoder/endurance_precise_probe.json.pos{pos}.pt"
        values = torch.load(path, map_location="cpu", weights_only=False)["tt_values"]
        provenance.append(dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        q, k = [values[f"delta_{key}"].repeat_interleave(2, dim=2).permute(0, 2, 3, 1) for key in ("q", "k")]
        v = values["delta_v"].float().permute(0, 2, 1, 3)
        for rounded in (False, True):
            state = values["delta_state_input"]
            if rounded:
                state = state.bfloat16().float()
            decay = state * values["delta_g"].exp().reshape(1, 32, 1, 1)
            product = decay * k
            results.append((f"real_pos{pos}_round{rounded}_memory", product))
            delta = (v - ordered(product)) * values["delta_beta"].reshape(1, 32, 1, 1)
            updated = decay + k * delta
            results.append((f"real_pos{pos}_round{rounded}_output", updated * (q * (128**-0.5))))
    rng = torch.Generator().manual_seed(20261116)
    results.append(("rank_distinct_random", torch.randn(1, 32, 128, 128, generator=rng)))
    cancellation = torch.zeros(1, 32, 128, 128)
    # Starting from zero and tree/whole-axis reassociation are visible here.
    cancellation[:, :, 0::16] = 2**24
    cancellation[:, :, 1::16] = 1
    cancellation[:, :, 2::16] = -(2**24)
    cancellation[:, :, 3::16] = 1
    cancellation *= torch.arange(1, 33).reshape(1, 32, 1, 1)
    results.append(("cancellation_rank_distinct", cancellation))
    signed = torch.zeros_like(cancellation)
    signed[:, 0::2] = -0.0
    results.append(("signed_zero", signed))
    return [(name, value.contiguous()) for name, value in results], provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("acceptance", "timing", "cpu"), default="acceptance")
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--positions", type=int, nargs="+", default=(128, 8157))
    args = parser.parse_args()
    torch.set_num_threads(4)
    root = Path(__file__).resolve().parents[1]
    record = dict(
        passed=False,
        command=sys.argv,
        mode=args.mode,
        checks=[],
        timings={},
        scope="Contraction component only; full recurrence/layer34/model acceptance still required",
        source_sha256={
            str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (Path(__file__), root / "tt/direct_recurrent_sum_probe.py", root / "tt/multichip_decoder.py")
        },
        environment={
            k: os.environ.get(k)
            for k in (
                "TT_METAL_TRACE_ALLOC_TRACKING",
                "TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE",
                "TT_METAL_WATCHER",
                "TT_METAL_DEVICE_PROFILER",
            )
        },
    )
    mesh, traces = None, {}
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(record, indent=2) + "\n")

    try:
        cases, record["fixture_provenance"] = fixtures(root, args.positions)
        record["cpu_fixtures"] = [
            dict(
                name=name,
                finite=bool(torch.isfinite(x).all()),
                reassociated_difference=int(
                    (ordered(x).squeeze(2).view(torch.int32) != x.sum(2).view(torch.int32)).sum()
                ),
            )
            for name, x in cases
        ]
        if args.mode == "cpu":
            record["passed"] = all(x["finite"] for x in record["cpu_fixtures"])
            record["hardware_verified"] = False
            return
        import ttnn as t
        from models.autoports.qwen_qwen3_6_35b_a3b.tt.direct_recurrent_sum_probe import DirectRecurrentSum
        from models.autoports.qwen_qwen3_6_35b_a3b.tt.multichip_decoder import LinearTP4Mixin

        assert os.environ.get("TT_METAL_TRACE_ALLOC_TRACKING") == ("1" if args.mode == "acceptance" else "0")
        assert os.environ.get("TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE") == "0"
        if args.mode == "timing":
            assert not os.environ.get("TT_METAL_WATCHER") and not os.environ.get("TT_METAL_DEVICE_PROFILER")
        t.set_fabric_config(t.FabricConfig.FABRIC_1D)
        mesh = t.open_mesh_device(t.MeshShape(1, 4), trace_region_size=200000000)
        config = t.init_device_compute_kernel_config(
            mesh.arch(),
            math_fidelity=t.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )
        dec = SimpleNamespace(compute=config)
        old_sum = MethodType(LinearTP4Mixin._recurrent_sum, dec)
        candidate = DirectRecurrentSum(t, mesh)

        def upload(x, device=True):
            kw = dict(dtype=t.float32, layout=t.TILE_LAYOUT, mesh_mapper=t.ShardTensorToMesh(mesh, dim=1))
            if device:
                kw.update(device=mesh, memory_config=t.L1_MEMORY_CONFIG)
            return t.from_torch(x, **kw)

        def host(value):
            return [t.to_torch(rank).contiguous() for rank in t.get_device_tensors(value)]

        def compare(name, actual, expected):
            actual_ranks = host(actual)
            assert len(actual_ranks) == len(expected) == 4
            for rank, (a, b) in enumerate(zip(actual_ranks, expected)):
                assert a.shape == b.shape and a.dtype == b.dtype == torch.float32
                count = int((a.view(torch.int32) != b.view(torch.int32)).sum())
                record["checks"].append(
                    dict(name=name, rank=rank, differences=count, max_abs=float((a - b).abs().max()))
                )
                assert count == 0, (name, rank, count)

        products = upload(cases[0][1])
        functions = dict(
            old_head=lambda: old_sum(products),
            new_head=lambda: candidate(products),
            old_terminal=lambda: t.permute(old_sum(products), (0, 2, 1, 3)),
            new_terminal=lambda: candidate(products, terminal_rows=True),
        )
        outputs = {}
        for name, fn in functions.items():
            value = fn()
            outputs[name] = t.empty_like(value)
            t.copy(value, outputs[name])
            host(outputs[name])  # Compile every readback layout before first capture.
        t.synchronize_device(mesh)
        for name, fn in functions.items():
            trace = t.begin_trace_capture(mesh, cq_id=0)
            try:
                t.copy(fn(), outputs[name])
            finally:
                t.end_trace_capture(mesh, trace, cq_id=0)
                traces[name] = trace
        cache_entries = mesh.num_program_cache_entries()
        for name, value in cases:
            staging = upload(value, device=False)
            t.copy_host_to_device_tensor(staging, products)
            expected = host(old_sum(products))
            expected_terminal = [x.permute(0, 2, 1, 3).contiguous() for x in expected]
            compare(name + "/eager_head", candidate(products), expected)
            compare(name + "/eager_terminal", candidate(products, terminal_rows=True), expected_terminal)
            for variant, trace in traces.items():
                t.execute_trace(mesh, trace, cq_id=0, blocking=False)
                compare(
                    name + "/trace_" + variant,
                    outputs[variant],
                    expected_terminal if "terminal" in variant else expected,
                )
        assert mesh.num_program_cache_entries() == cache_entries
        t.copy_host_to_device_tensor(upload(cases[0][1], device=False), products)
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
            passed=True,
            hardware_verified=True,
            program_cache_entries=cache_entries,
            timing_eligible=args.mode == "timing",
            replay_count=len(traces) * (len(cases) + args.iterations + 3),
        )
    except Exception:
        record["error"] = traceback.format_exc()
        raise
    finally:
        save()
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
