# SPDX-License-Identifier: Apache-2.0
"""Serialized real-weight TP4 MoE comparison, outside any live serving run.

Readbacks and synchronization here are explicit standalone component controls;
they are not instrumentation for the Stage 11 live phase collector.
"""

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

import torch
from transformers import AutoConfig

import ttnn
from models.autoports.qwen_qwen3_8_flash_next.tests import benchmark_batch_moe_candidate as candidate
from models.autoports.qwen_qwen3_8_flash_next.tests.metrics import pcc_metrics
from models.autoports.qwen_qwen3_8_flash_next.tt import precision
from models.autoports.qwen_qwen3_8_flash_next.tt.checkpoint import CheckpointState
from models.autoports.qwen_qwen3_8_flash_next.tt.multichip_decoder import MeshOps, ParallelMoE


def host_output(value):
    return torch.cat([ttnn.to_torch(shard) for shard in ttnn.get_device_tensors(value)], dim=-1)


def kernel_identity(config):
    return {
        key: str(getattr(config, key))
        for key in ("math_fidelity", "math_approx_mode", "fp32_dest_acc_en", "packer_l1_acc", "dst_full_sync_en")
    }


def routes(model, value, packed):
    """Read actual TT router IDs outside captures, preserving shape-dependent math."""
    batch = value.shape[1]
    rows = (
        [candidate.logical_reshape(value, (1, 1, batch, 2560))]
        if packed
        else [value[:, lane : lane + 1, :, :] for lane in range(batch)]
    )
    result = []
    for row in rows:
        logits = model.ops.linear(row, model.router, compute_kernel_config=model.router_compute)[..., :512]
        values, ids = ttnn.topk(
            ttnn.pad(logits, [(0, 0), (0, 0), (0, 0), (0, 512)], float("-inf")),
            k=32,
            dim=-1,
            sorted=True,
        )
        result.append(ttnn.to_torch(ttnn.get_device_tensors(ids)[0]).reshape(-1, 32)[:, :10].long())
        del logits, values, ids
    return torch.cat(result, dim=0)


def measured(mesh, call, repeats, warmup):
    trace = None
    output = call()
    ttnn.synchronize_device(mesh)
    eager = host_output(output)
    del output
    try:
        trace = ttnn.begin_trace_capture(mesh, cq_id=0)
        output = call()
        ttnn.end_trace_capture(mesh, trace, cq_id=0)
        ttnn.synchronize_device(mesh)
        for _ in range(warmup):
            ttnn.execute_trace(mesh, trace, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh)
        seconds = []
        for _ in range(repeats):
            begin = time.perf_counter()
            ttnn.execute_trace(mesh, trace, cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh)
            seconds.append(time.perf_counter() - begin)
        replay = host_output(output)
        return (
            eager,
            replay,
            {
                "seconds": seconds,
                "median_seconds": statistics.median(seconds),
                "min_seconds": min(seconds),
                "max_seconds": max(seconds),
                "warmup_replays": warmup,
                "measured_replays": repeats,
                "boundary": "host execute_trace through existing standalone synchronize_device",
            },
        )
    finally:
        if trace is not None:
            ttnn.release_trace(mesh, trace)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--trace-bytes", type=int, default=268435456)
    args = parser.parse_args()
    if args.repeats < 1 or args.warmup < 1:
        parser.error("positive repeats and warmup are required")
    if any(
        os.environ.get(key, "0") not in ("", "0")
        for key in ("TT_METAL_DEVICE_PROFILER", "TT_METAL_WATCHER", "TT_METAL_WATCHER_DUMP_ALL")
    ):
        raise RuntimeError("Use this unprofiled component probe without watcher or profiler")
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(8)
    policy = precision.for_layer(precision.load_policy(), args.layer)
    cfg = AutoConfig.from_pretrained(args.checkpoint, local_files_only=True).text_config
    state = CheckpointState(args.checkpoint, f"model.language_model.layers.{args.layer}.")
    report = {
        "scope": "one real-weight MoE component; not full-model or serving qualification",
        "argv": sys.argv,
        "checkpoint": str(args.checkpoint.resolve()),
        "policy": policy,
        "layer": args.layer,
        "mesh": [1, 4],
        "results": [],
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "source_sha256": {},
        "workspace": {
            "packed_capacity_bf16_bytes_per_chip": 128 * 128 * 2560 * 2,
            "packed_output_bfp8_bytes_per_chip": 128 * 4 * 80 * 1088,
            "existing_singleton_capacity_bf16_bytes_per_chip": 128 * 32 * 2560 * 2,
            "note": "temporary packed workspace reuses existing128-row implementation; full-stack memory not proven",
        },
    }
    for path in (
        Path(__file__),
        Path(candidate.__file__),
        precision.ROOT / "tt/multichip_decoder.py",
        Path(precision.__file__),
    ):
        report["source_sha256"][str(path.resolve())] = hashlib.sha256(path.read_bytes()).hexdigest()
        (args.output / (path.stem + ".py.txt")).write_bytes(path.read_bytes())
    (args.output / "identity.json").write_text(json.dumps(report, indent=2) + "\n")
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=args.trace_bytes)
    try:
        ops = MeshOps(mesh)
        model = ParallelMoE(state, cfg, ops)
        for name in ("shared_gu", "shared_down"):
            precision.cast_attr(model, name, policy["weight_groups"]["shared_expert"])
        model.shared_compute = precision.compute(mesh, policy["compute_fidelities"]["shared_expert"])
        model.router_compute = precision.compute(mesh, policy["compute_fidelities"]["router"])
        model.routed_compute = ttnn.init_device_compute_kernel_config(
            mesh.arch(),
            math_fidelity=getattr(ttnn.MathFidelity, policy["compute_fidelities"]["routed_expert_gate_up"]),
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
            dst_full_sync_en=False,
        )
        ops.precision_ccl = policy["ccl_dtype"]
        report["actual_precision"] = {
            "router": str(model.router.dtype),
            "shared_gu": str(model.shared_gu.dtype),
            "shared_down": str(model.shared_down.dtype),
            "gate_dtypes": sorted({str(x.dtype) for x in model.gates}),
            "up_dtypes": sorted({str(x.dtype) for x in model.ups}),
            "down_dtypes": sorted({str(x.dtype) for x in model.downs}),
            "router_compute": kernel_identity(model.router_compute),
            "shared_compute": kernel_identity(model.shared_compute),
            "routed_compute": kernel_identity(model.routed_compute),
            "ccl": ops.precision_ccl,
        }
        rng = torch.Generator().manual_seed(20261016)
        independent = torch.randn((1, 32, 1, 2560), generator=rng).bfloat16()
        patterns = {
            "b1_unchanged": independent[:, :1].clone(),
            "b32_independent": independent,
            "b32_repeated": independent[:, :1].expand(1, 32, 1, 2560).contiguous(),
            "b32_one_nonzero": torch.cat((independent[:, :1], torch.zeros_like(independent[:, 1:])), dim=1),
        }
        for name, host in patterns.items():
            device = ops.upload(host)
            baseline_routes = routes(model, device, packed=False)
            packed_routes = routes(model, device, packed=host.shape[1] > 1)
            baseline, baseline_replay, before = measured(
                mesh,
                lambda: ParallelMoE.__call__(model, device),
                args.repeats,
                args.warmup,
            )
            packed, packed_replay, after = measured(
                mesh,
                lambda: candidate.packed_batch_decode(model, device),
                args.repeats,
                args.warmup,
            )
            record = {
                "name": name,
                "input_shape": list(host.shape),
                "eager_comparison": pcc_metrics(baseline, packed),
                "baseline_eager_vs_replay": pcc_metrics(baseline, baseline_replay),
                "candidate_eager_vs_replay": pcc_metrics(packed, packed_replay),
                "replay_comparison": pcc_metrics(baseline_replay, packed_replay),
                "route_order_equal": torch.equal(baseline_routes, packed_routes),
                "route_set_rows_equal": int(
                    (baseline_routes.sort(-1).values == packed_routes.sort(-1).values).all(-1).sum()
                ),
                "baseline_distinct_experts": int(baseline_routes.unique().numel()),
                "baseline": before,
                "candidate": after,
                "median_speedup": before["median_seconds"] / after["median_seconds"],
            }
            report["results"].append(record)
            torch.save(
                {
                    "input": host,
                    "baseline": baseline,
                    "candidate": packed,
                    "baseline_replay": baseline_replay,
                    "candidate_replay": packed_replay,
                    "baseline_routes": baseline_routes,
                    "candidate_routes": packed_routes,
                },
                args.output / f"{name}.pt",
            )
            (args.output / "results.json").write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps(record), flush=True)
            del device
        # These component gates do not assert model accuracy or serving readiness.
        assert all(r["replay_comparison"]["pcc"] >= 0.995 for r in report["results"])
        assert report["results"][0]["replay_comparison"]["exact_equal"], "B1 path must remain identical"
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
