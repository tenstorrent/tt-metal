# SPDX-License-Identifier: Apache-2.0
"""Real-weight S256 MoE: two128 groups versus one256, exclusive TP4 only.

No production patch or native B1 serving change. Standalone timing completion
uses explicit synchronization; no profiler is enabled. Release each capture
before diagnostics or the next variant's allocations.
"""

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import time
import traceback
from pathlib import Path

os.environ.setdefault("TT_METAL_TRACE_ALLOC_TRACKING", "1")
os.environ.setdefault("TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE", "0")

import torch
from transformers import AutoConfig
from ttnn.tools import trace_allocation_tracker

import ttnn
from models.autoports.qwen_qwen3_8_flash_next.tests import benchmark_prefill_moe_group_candidate as candidate
from models.autoports.qwen_qwen3_8_flash_next.tests.benchmark_batch_moe_probe import host_output, kernel_identity
from models.autoports.qwen_qwen3_8_flash_next.tests.metrics import pcc_metrics
from models.autoports.qwen_qwen3_8_flash_next.tt import precision
from models.autoports.qwen_qwen3_8_flash_next.tt.checkpoint import CheckpointState
from models.autoports.qwen_qwen3_8_flash_next.tt.multichip_decoder import MeshOps, ParallelMoE

ROOT = Path(__file__).resolve().parents[1]


def read_routes(model, value, group):
    rows, scores = [], []
    for start in range(0, value.shape[2], group):
        x = value[:, :, start : min(start + group, value.shape[2]), :]
        logits = model.ops.linear(x, model.router, compute_kernel_config=model.router_compute)[..., :512]
        top, ids = ttnn.topk(
            ttnn.pad(logits, [(0, 0), (0, 0), (0, 0), (0, 512)], float("-inf")),
            k=32,
            dim=-1,
            sorted=True,
        )
        score = ttnn.softmax(
            ttnn.add(top, model.mask), dim=-1, numeric_stable=True, compute_kernel_config=model.ops.compute
        )
        rows.append(ttnn.to_torch(ttnn.get_device_tensors(ids)[0]).reshape(-1, 32)[:, :10].long())
        scores.append(ttnn.to_torch(ttnn.get_device_tensors(score)[0]).reshape(-1, 32)[:, :10])
    return torch.cat(rows), torch.cat(scores)


def measured(mesh, call, repeats, warmup):
    eager_tensor = call()
    eager = host_output(eager_tensor)
    del eager_tensor
    trace = None
    output = None
    try:
        trace = ttnn.begin_trace_capture(mesh, cq_id=0)
        output = call()
        ttnn.end_trace_capture(mesh, trace, cq_id=0)
        for _ in range(warmup):
            ttnn.execute_trace(mesh, trace, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh)
        seconds = []
        for _ in range(repeats):
            tick = time.perf_counter()
            ttnn.execute_trace(mesh, trace, cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh)
            seconds.append(time.perf_counter() - tick)
    finally:
        if trace is not None:
            ttnn.release_trace(mesh, trace)
    replay = host_output(output)
    return (
        eager,
        replay,
        dict(
            seconds=seconds,
            median_seconds=statistics.median(seconds),
            warmup_replays=warmup,
            measured_replays=repeats,
            trace_released_before_diagnostics=True,
        ),
    )


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
        parser.error("Positive warmup/repeats required")
    for name in ("TT_METAL_DEVICE_PROFILER", "TT_METAL_WATCHER", "TT_METAL_WATCHER_DUMP_ALL"):
        if os.environ.get(name, "0") not in ("", "0"):
            raise RuntimeError(f"Disable {name} for this standalone unprofiled probe")
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(8)
    policy = precision.for_layer(precision.load_policy(), args.layer)
    report = dict(
        status="preparing",
        args={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        scope="isolated real-weight prefill MoE only; native B1/production unchanged",
        commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        policy=policy,
        context=policy["max_context"],
        results=[],
        source_sha256={},
        timing_boundary="host execute_trace through standalone completion synchronize_device",
        workspace_per_chip={
            "capacity128_bf16": 128 * 128 * 2560 * 2,
            "capacity256_bf16": 128 * 256 * 2560 * 2,
            "output128_bfp8": 128 * 4 * 80 * 1088,
            "output256_bfp8": 128 * 8 * 80 * 1088,
            "dispatch256_cb": 5120 + 256 * 192 + 64,
            "compact256_cb": 2176 + 256 * 320 + 64,
        },
    )

    def save():
        (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")

    for path in (Path(__file__), Path(candidate.__file__), ROOT / "tt/multichip_decoder.py", Path(precision.__file__)):
        report["source_sha256"][str(path.resolve())] = hashlib.sha256(path.read_bytes()).hexdigest()
        (args.output / (path.name + ".txt")).write_bytes(path.read_bytes())
    save()
    mesh = None
    try:
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=args.trace_bytes)
        assert ttnn._ttnn.operations.trace.trace_allocation_tracking_enabled()
        assert trace_allocation_tracker.TRACE_ALLOC_TRACKING
        assert os.environ["TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE"] == "0"
        report["tracking"] = dict(native=True, python=True, skip_program_cache=False)
        cfg = AutoConfig.from_pretrained(args.checkpoint, local_files_only=True).text_config
        state = CheckpointState(args.checkpoint, f"model.language_model.layers.{args.layer}.")
        ops = MeshOps(mesh)
        owner = ParallelMoE(state, cfg, ops)
        for name in ("shared_gu", "shared_down"):
            precision.cast_attr(owner, name, policy["weight_groups"]["shared_expert"])
        owner.shared_compute = precision.compute(mesh, policy["compute_fidelities"]["shared_expert"])
        owner.router_compute = precision.compute(mesh, policy["compute_fidelities"]["router"])
        owner.routed_compute = ttnn.init_device_compute_kernel_config(
            mesh.arch(),
            math_fidelity=getattr(ttnn.MathFidelity, policy["compute_fidelities"]["routed_expert_gate_up"]),
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
            dst_full_sync_en=False,
        )
        ops.precision_ccl = policy["ccl_dtype"]
        view = candidate.make_candidate(owner)
        assert all(
            getattr(owner, name) is getattr(view, name)
            for name in ("gates", "ups", "downs", "router", "shared_gu", "shared_down", "routed_compute")
        )
        report["actual_precision"] = dict(
            router=str(owner.router.dtype),
            shared_gu=str(owner.shared_gu.dtype),
            shared_down=str(owner.shared_down.dtype),
            gates=sorted({str(t.dtype) for t in owner.gates}),
            ups=sorted({str(t.dtype) for t in owner.ups}),
            downs=sorted({str(t.dtype) for t in owner.downs}),
            routed_compute=kernel_identity(owner.routed_compute),
            router_compute=kernel_identity(owner.router_compute),
            shared_compute=kernel_identity(owner.shared_compute),
            shared_weight_objects=True,
        )
        rng = torch.Generator().manual_seed(20261016)
        independent = torch.randn(1, 1, 256, 2560, generator=rng).bfloat16()
        patterns = {
            "s128_control": independent[:, :, :128].contiguous(),
            "s256_independent": independent,
            "s256_repeated_capacity256": independent[:, :, :1].expand(1, 1, 256, 2560).contiguous(),
        }
        for name, host in patterns.items():
            # Match the B32 owner's B1 serial-prefill view policy at S256.
            ops.large_prefill = host.shape[2] >= 256
            ops.l1_prefill = True
            device = ops.upload(host)
            old_ids, old_scores = read_routes(owner, device, 128)
            new_ids, new_scores = read_routes(view, device, 256)
            # Warm both shapes before either capture. No live capture survives
            # measured(), including when later diagnostic reads allocate.
            for call in (lambda: owner(device), lambda: candidate.forward(view, device)):
                warm = call()
                ttnn.synchronize_device(mesh)
                del warm
            baseline, baseline_replay, before = measured(mesh, lambda: owner(device), args.repeats, args.warmup)
            changed, changed_replay, after = measured(
                mesh, lambda: candidate.forward(view, device), args.repeats, args.warmup
            )
            counts = torch.bincount(new_ids[new_scores != 0], minlength=512)
            record = dict(
                name=name,
                input_shape=list(host.shape),
                baseline=before,
                candidate=after,
                median_speedup=before["median_seconds"] / after["median_seconds"],
                route_order_equal=torch.equal(old_ids, new_ids),
                route_scores_equal=torch.equal(old_scores, new_scores),
                max_tokens_per_expert=int(counts.max()),
                active_experts=int((counts > 0).sum()),
                eager_comparison=pcc_metrics(baseline, changed),
                replay_comparison=pcc_metrics(baseline_replay, changed_replay),
                baseline_eager_replay=pcc_metrics(baseline, baseline_replay),
                candidate_eager_replay=pcc_metrics(changed, changed_replay),
            )
            raw = args.output / f"{name}.pt"
            torch.save(
                dict(
                    input=host,
                    baseline=baseline,
                    candidate=changed,
                    baseline_replay=baseline_replay,
                    candidate_replay=changed_replay,
                    baseline_route_ids=old_ids,
                    candidate_route_ids=new_ids,
                    baseline_route_scores=old_scores,
                    candidate_route_scores=new_scores,
                    counts=counts,
                ),
                raw,
            )
            record["raw"] = dict(path=str(raw), sha256=hashlib.sha256(raw.read_bytes()).hexdigest())
            report["results"].append(record)
            save()
            print(json.dumps(record), flush=True)
            assert torch.isfinite(changed_replay).all()
            assert record["baseline_eager_replay"]["exact_equal"]
            assert record["candidate_eager_replay"]["exact_equal"]
            if name == "s128_control":
                assert record["replay_comparison"]["exact_equal"]
            if name == "s256_repeated_capacity256":
                assert record["max_tokens_per_expert"] == 256
            del device
        report["status"] = "measurements_complete_review_numerics"
        save()
    except Exception:
        report["status"] = "failed"
        report["error"] = traceback.format_exc()
        save()
        raise
    finally:
        if mesh is not None:
            ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
