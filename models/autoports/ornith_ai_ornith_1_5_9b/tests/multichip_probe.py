# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Serial real-weight optimized baseline / TP4 contract and timing runner."""

import argparse
import hashlib
import json
import os
import statistics
import time
from pathlib import Path

import torch

import ttnn
from models.autoports.ornith_ai_ornith_1_5_9b.tests.run_decoder import ROOT, config, pcc, weights
from models.autoports.ornith_ai_ornith_1_5_9b.tests.validate_optimized import buffers, clear, device_only
from models.autoports.ornith_ai_ornith_1_5_9b.tt.multichip_decoder import MultichipDecoder
from models.autoports.ornith_ai_ornith_1_5_9b.tt.optimized_decoder import OptimizedDecoder


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", action="store_true")
    ap.add_argument("--optimization")
    ap.add_argument("--movement-control", action="store_true")
    ap.add_argument("--chunk-size", type=int, default=None)
    ap.add_argument("--trace", action="store_true")
    ap.add_argument("--profile", action="store_true")
    ap.add_argument(
        "--candidate",
        choices=[
            "default",
            "separate",
            "gather",
            "scatter",
            "both",
            "direct",
            "hybrid",
            "native",
            "budget96",
            "budget64",
            "native_unfused",
            "prefill_l1",
            "available_links",
            "two_link_reduce",
        ],
        default="default",
    )
    ap.add_argument("--benchmark", action="store_true")
    ap.add_argument("--prefill-warmup", type=int, default=0)
    ap.add_argument("--prefill-repeats", type=int, default=5)
    ap.add_argument("--positive-control", action="store_true")
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--readers", type=int)
    ap.add_argument("--cores", type=int)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--lengths", default="32")
    ap.add_argument("--layout", default="sharded")
    ap.add_argument("--output", required=True)
    ap.add_argument("--reference", required=True)
    args = ap.parse_args()
    assert args.prefill_warmup >= 0 and args.prefill_repeats > 0
    requested_benchmark = args.benchmark
    # Long replay timing loops can overflow profiler buffers. Profiler windows
    # record their own synchronized host durations; latency ranking is unprofiled.
    if args.profile:
        args.benchmark = False
    torch.set_num_threads(8)
    torch.manual_seed(73)
    c, sd = config(), weights(args.layer)
    baseline = args.baseline
    count = 1 if baseline else 4
    if not baseline:
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(
        ttnn.MeshShape(1, count), physical_device_ids=[0] if baseline else [], trace_region_size=32 * 1024 * 1024
    )
    records, saved, profile_times = [], {}, {}
    expected = {} if baseline else torch.load(args.reference, weights_only=True)

    def upload(x, integer=False, residual=False):
        mapper = (
            ttnn.ShardTensorToMesh(mesh, dim=3)
            if residual and args.layout == "sharded" and not baseline
            else ttnn.ReplicateTensorToMesh(mesh)
        )
        return ttnn.from_torch(
            x.contiguous(),
            device=mesh,
            dtype=ttnn.int32 if integer else ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT if integer else ttnn.TILE_LAYOUT,
            mesh_mapper=mapper,
        )

    def download(x, shard_dim=None):
        if baseline:
            return ttnn.to_torch(x)
        if shard_dim is None:
            return ttnn.to_torch(ttnn.get_device_tensors(x)[0])
        return ttnn.to_torch(x, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=shard_dim))

    def check(name, x, residual=None, state_dim=None):
        value = download(
            x,
            state_dim if state_dim is not None else (3 if args.layout == "sharded" and residual is not None else None),
        )
        saved[name] = value
        if not baseline:
            row = {"label": name, "pcc": pcc(expected[name], value)}
            if residual is not None:
                row["contribution_pcc"] = pcc(expected[name] - residual, value - residual)
            if residual is not None and args.layout == "replicated":
                ranks = [ttnn.to_torch(v) for v in ttnn.get_device_tensors(x)]
                row["all_ranks_pcc"] = min(pcc(expected[name], v) for v in ranks)
                row["all_ranks_contribution_pcc"] = min(pcc(expected[name] - residual, v - residual) for v in ranks)
            print(json.dumps(row), flush=True)
            assert min(v for k, v in row.items() if "pcc" in k) >= 0.995, row
            records.append(row)

    try:
        decoder_cls = MultichipDecoder
        if args.candidate != "default":
            from models.autoports.ornith_ai_ornith_1_5_9b.tests.multichip_topologies import (
                AvailableLinksDecoder,
                Budget64PrefillDecoder,
                BudgetPrefillDecoder,
                DirectReduceDecoder,
                FusedBothDecoder,
                FusedGatherDecoder,
                FusedScatterDecoder,
                HybridDecoder,
                L1PrefillDecoder,
                NativeResidualDecoder,
                NativeUnfusedDecoder,
                TwoLinkReduceDecoder,
            )

            decoder_cls = {
                "separate": MultichipDecoder,
                "gather": FusedGatherDecoder,
                "scatter": FusedScatterDecoder,
                "both": FusedBothDecoder,
                "direct": DirectReduceDecoder,
                "hybrid": HybridDecoder,
                "native": NativeResidualDecoder,
                "budget96": BudgetPrefillDecoder,
                "budget64": Budget64PrefillDecoder,
                "native_unfused": NativeUnfusedDecoder,
                "prefill_l1": L1PrefillDecoder,
                "available_links": AvailableLinksDecoder,
                "two_link_reduce": TwoLinkReduceDecoder,
            }[args.candidate]
        if args.optimization:
            from models.autoports.ornith_ai_ornith_1_5_9b.tests.multichip_optimization import candidate

            decoder_cls = candidate(args.optimization)
        layer = (OptimizedDecoder if baseline else decoder_cls).from_state_dict(
            sd,
            hf_config=c,
            layer_idx=args.layer,
            mesh_device=mesh,
            **({"chunk_size": args.chunk_size} if args.chunk_size is not None else {}),
            **(
                {}
                if baseline
                else {
                    "residual_layout": args.layout,
                    **({"movement_optimized": False} if args.movement_control else {}),
                    "readers": args.readers,
                    "cores": args.cores,
                    **(
                        {"collective": "separate", "fused_prefill": False, "native_local": False}
                        if args.candidate != "default"
                        else {}
                    ),
                }
            ),
        )
        lengths = list(map(int, args.lengths.split(",")))
        capacity = ((max(lengths) + max(64, args.steps) + 31) // 32) * 32
        state = layer.allocate_state(context=capacity)
        table = upload(torch.randperm(capacity // 32, dtype=torch.int32)[None], True)
        for n in lengths:
            clear(state)
            x = (torch.randn(1, 1, n, 4096) * 0.1).bfloat16()
            tx = upload(x, residual=True)
            with device_only():
                out = layer.prefill_forward(tx, state=state, page_table=table)
            check(f"prefill_{n}", out, x)
            prefill_ms = []
            if args.profile:
                from tracy import signpost

                clear(state)
                tmp = layer.prefill_forward(tx, state=state, page_table=table)
                ttnn.synchronize_device(mesh)
                del tmp
                clear(state)
                signpost("PERF_PREFILL")
                begin = time.perf_counter()
                tmp = layer.prefill_forward(tx, state=state, page_table=table)
                ttnn.synchronize_device(mesh)
                profile_times["prefill_host_ms"] = (time.perf_counter() - begin) * 1000
                signpost("PERF_PREFILL_END")
                del tmp
            if args.benchmark:
                for repeat in range(args.prefill_warmup + args.prefill_repeats):
                    clear(state)
                    ttnn.synchronize_device(mesh)
                    begin = time.perf_counter()
                    tmp = layer.prefill_forward(tx, state=state, page_table=table)
                    ttnn.synchronize_device(mesh)
                    elapsed_ms = (time.perf_counter() - begin) * 1000
                    if repeat >= args.prefill_warmup:
                        prefill_ms.append(elapsed_ms)
                    del tmp
            snap = [(v, ttnn.clone(v)) for v in buffers(state)]
            td = upload(torch.zeros(1, 1, 1, 4096, dtype=torch.bfloat16), residual=True)
            pos = upload(torch.tensor([n], dtype=torch.int32), True)

            def restore(snap=snap):
                for dst, src in snap:
                    ttnn.copy(src, dst)

            def forward(td=td, pos=pos):
                with device_only():
                    return layer.decode_forward(td, state=state, page_table=table, current_position=pos)

            trace = None
            decode_ms = []
            if args.trace:
                from ttnn.tools import trace_allocation_tracker

                for _ in range(2):
                    tmp = forward()
                    del tmp
                    restore()
                ttnn.synchronize_device(mesh)
                trace = ttnn.begin_trace_capture(mesh, cq_id=0)
                traced = forward()
                ttnn.end_trace_capture(mesh, trace, cq_id=0)
                # The sole capture output is overwritten by this same trace and
                # read before another replay. No other live trace shares its storage.
                trace_allocation_tracker.acknowledge_corruptible(traced)
                if args.positive_control:
                    unsafe = ttnn.empty((1, 1, 32, 4096), device=mesh, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
                    rejected = False
                    try:
                        ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
                    except RuntimeError as error:
                        rejected = "unsafe" in str(error).lower() or "allocation" in str(error).lower()
                        print("POSITIVE_CONTROL_REJECTED", str(error)[:240], flush=True)
                    assert rejected, "tracker failed to reject unsafe allocation"
                    del unsafe
                    records.append({"label": "trace_allocation_positive_control", "passed": True})
                if args.profile:
                    ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
                    signpost("PERF_DECODE")
                    begin = time.perf_counter()
                    ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
                    profile_times["decode_host_ms"] = (time.perf_counter() - begin) * 1000
                    signpost("PERF_DECODE_END")
                if args.benchmark:
                    for repeat in range(7):
                        ttnn.synchronize_device(mesh)
                        begin = time.perf_counter()
                        for _ in range(50):
                            ttnn.execute_trace(mesh, trace, cq_id=0, blocking=False)
                        ttnn.synchronize_device(mesh)
                        decode_ms.append((time.perf_counter() - begin) * 1000 / 50)
                restore()
            if args.benchmark:
                print("CANDIDATE_TIMING", n, prefill_ms, decode_ms, flush=True)
            for step in range(args.steps):
                d = (torch.randn(1, 1, 1, 4096) * 0.1).bfloat16()
                hd = ttnn.from_torch(
                    d,
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    mesh_mapper=(
                        ttnn.ShardTensorToMesh(mesh, dim=3)
                        if not baseline and args.layout == "sharded"
                        else ttnn.ReplicateTensorToMesh(mesh)
                    ),
                )
                hp = ttnn.from_torch(
                    torch.tensor([n + step], dtype=torch.int32), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT
                )
                ttnn.copy_host_to_device_tensor(hd, td)
                ttnn.copy_host_to_device_tensor(hp, pos)
                if trace is not None:
                    ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
                    out = traced
                else:
                    out = forward()
                check(f"decode_{n}_{step}", out, d)
            records.append(
                {
                    "label": f"timing_{n}",
                    "prefill_ms": statistics.median(prefill_ms) if prefill_ms else None,
                    "decode_ms": statistics.median(decode_ms) if decode_ms else None,
                    "prefill_samples_ms": prefill_ms,
                    "decode_samples_ms": decode_ms,
                }
            )
            if layer.kind == "full_attention":
                check(f"key_{n}", state.key, state_dim=1)
                check(f"value_{n}", state.value, state_dim=1)
            else:
                check(f"recurrent_{n}", state.recurrent, state_dim=1)
                # Conv rank ordering is Q,K,V per chip; reconstruct global Q,K,V.
                value = download(state.conv, 3)
                if not baseline:
                    pieces = value.split(2048, dim=-1)
                    value = torch.cat(
                        [
                            torch.cat([p[..., lo:hi] for p in pieces], -1)
                            for lo, hi in [(0, 512), (512, 1024), (1024, 2048)]
                        ],
                        -1,
                    )
                saved[f"conv_{n}"] = value
                if not baseline:
                    score = pcc(expected[f"conv_{n}"], value)
                    assert score >= 0.995, score
                    records.append({"label": f"conv_{n}", "pcc": score})
            if args.refresh and trace is not None:
                for repeat in range(3):
                    clear(state)
                    d = (torch.randn(1, 1, 1, 4096) * 0.1).bfloat16()
                    hd = ttnn.from_torch(
                        d,
                        dtype=ttnn.bfloat16,
                        layout=ttnn.TILE_LAYOUT,
                        mesh_mapper=(
                            ttnn.ShardTensorToMesh(mesh, dim=3)
                            if not baseline and args.layout == "sharded"
                            else ttnn.ReplicateTensorToMesh(mesh)
                        ),
                    )
                    hp = ttnn.from_torch(
                        torch.tensor([0], dtype=torch.int32), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT
                    )
                    ht = ttnn.from_torch(
                        torch.randperm(capacity // 32, dtype=torch.int32)[None],
                        dtype=ttnn.int32,
                        layout=ttnn.ROW_MAJOR_LAYOUT,
                    )
                    ttnn.copy_host_to_device_tensor(hd, td)
                    ttnn.copy_host_to_device_tensor(hp, pos)
                    ttnn.copy_host_to_device_tensor(ht, table)
                    ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
                    check(f"refresh_{n}_{repeat}", traced, d)
                records.append(
                    {"label": f"trace_refresh_{n}", "trace_id": str(trace), "captures": 1, "refresh_replays": 3}
                )
            if trace is not None:
                ttnn.release_trace(mesh, trace)
        if baseline:
            torch.save(saved, args.reference)
        Path(args.output).write_text(
            json.dumps(
                {
                    "baseline": baseline,
                    "optimization": args.optimization,
                    "movement_control": args.movement_control,
                    "requested_chunk_size": args.chunk_size,
                    "chunk_size": layer.chunk_size,
                    "command": __import__("sys").argv,
                    "candidate_sha256": (
                        hashlib.sha256((ROOT / "tests/multichip_optimization.py").read_bytes()).hexdigest()
                        if args.optimization
                        else None
                    ),
                    "state_shapes": {
                        name: list(getattr(state, name).shape)
                        for name in ("conv", "recurrent", "key", "value")
                        if getattr(state, name) is not None
                    },
                    "runtime_fallback_guard": "torch dispatch and host transfer APIs forbidden during prefill/decode",
                    "baseline_sha256": hashlib.sha256((ROOT / "tt/optimized_decoder.py").read_bytes()).hexdigest(),
                    "layout": args.layout,
                    "layer": args.layer,
                    "records": records,
                    "trace": args.trace,
                    "candidate": args.candidate,
                    "profile": args.profile,
                    "profile_window_times": profile_times,
                    "requested_benchmark": requested_benchmark,
                    "prefill_warmup": args.prefill_warmup,
                    "prefill_repeats": args.prefill_repeats,
                    "steps": args.steps,
                    "benchmark": args.benchmark,
                    "tracking": os.getenv("TT_METAL_TRACE_ALLOC_TRACKING", "0"),
                    "skip_program_cache_tracking": os.getenv("TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE", "0"),
                    "regime": "serial",
                    "devices": list(range(count)),
                    "readers": args.readers,
                    "cores": args.cores,
                    "decode_geometry": None if baseline else layer.decode_geometry,
                    "ccl_links": None if baseline else layer.ccl.num_links,
                    **(
                        {"collective": "separate", "fused_prefill": False, "native_local": False}
                        if args.candidate != "default"
                        else {}
                    ),
                    "source_sha256": hashlib.sha256((ROOT / "tt/multichip_decoder.py").read_bytes()).hexdigest(),
                },
                indent=2,
            )
            + "\n"
        )
        print("PROBE_PASS", args.output, flush=True)
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
