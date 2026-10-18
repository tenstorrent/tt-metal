# SPDX-License-Identifier: Apache-2.0
"""Isolated real MoE S1024: two512 versus one1024 groups; TP4 exclusive."""

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
from models.autoports.qwen_qwen3_8_flash_next.tests import benchmark_prefill_moe1024_candidate as candidate
from models.autoports.qwen_qwen3_8_flash_next.tests.benchmark_batch_moe_probe import host_output, kernel_identity
from models.autoports.qwen_qwen3_8_flash_next.tests.benchmark_prefill_moe_group_probe import read_routes
from models.autoports.qwen_qwen3_8_flash_next.tests.metrics import pcc_metrics
from models.autoports.qwen_qwen3_8_flash_next.tests.multichip_capacity_probe import view as memory_view
from models.autoports.qwen_qwen3_8_flash_next.tt import precision
from models.autoports.qwen_qwen3_8_flash_next.tt.checkpoint import CheckpointState
from models.autoports.qwen_qwen3_8_flash_next.tt.multichip_decoder import MeshOps, ParallelMoE

ROOT = Path(__file__).resolve().parents[1]


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
        captured_memory = dict(dram=memory_view(mesh), l1=memory_view(mesh, ttnn.BufferType.L1))
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
            captured_memory=captured_memory,
            released_memory=dict(dram=memory_view(mesh), l1=memory_view(mesh, ttnn.BufferType.L1)),
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--trace-bytes", type=int, default=268435456)
    args = parser.parse_args()
    if args.repeats < 1 or args.warmup < 1:
        parser.error("Positive warmup and repeat counts required")
    assert all(
        os.environ.get(n, "0") in ("", "0")
        for n in ("TT_METAL_DEVICE_PROFILER", "TT_METAL_WATCHER", "TT_METAL_WATCHER_DUMP_ALL")
    )
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(8)
    policy = precision.for_layer(precision.load_policy(), args.layer)
    report = dict(
        status="preparing",
        scope="isolated MoE, no production/nativeB1 changes or full-model speed claim",
        checkpoint=str(args.checkpoint.resolve()),
        layer=args.layer,
        policy=policy,
        context=policy["max_context"],
        results=[],
        commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        timing_boundary="warmed standalone execute_trace through synchronize_device completion",
        workspace_per_chip={
            str(n): dict(
                dispatch_bf16=128 * n * 2560 * 2,
                down_bfp8=128 * (n // 32) * 80 * 1088,
                compact_l1_bfp8=10 * (n // 32) * 80 * 1088,
                dispatch_cb=5120 + n * 192 + 64,
                compact_cb=2176 + n * 320 + 64,
            )
            for n in (512, 1024)
        },
    )
    sources = [
        Path(__file__),
        Path(candidate.__file__),
        Path(candidate.base.__file__),
        Path(candidate.base.group256_candidate.__file__),
        ROOT / "tests/benchmark_prefill_moe_group_probe.py",
        ROOT / "tt/multichip_decoder.py",
        Path(precision.__file__),
    ]
    report["source_sha256"] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    for path in sources:
        (args.output / (path.name + ".txt")).write_bytes(path.read_bytes())

    def save():
        (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")

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
        owner = ParallelMoE(
            CheckpointState(args.checkpoint, f"model.language_model.layers.{args.layer}."), cfg, MeshOps(mesh)
        )
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
        owner.ops.precision_ccl = policy["ccl_dtype"]
        view = candidate.make_candidate(owner)
        assert not hasattr(owner, "prefill_group_size")
        assert all(
            getattr(owner, n) is getattr(view, n)
            for n in ("router", "gates", "ups", "downs", "shared_gu", "shared_down", "routed_compute")
        )
        report["actual_precision"] = dict(
            router=str(owner.router.dtype),
            shared_gu=str(owner.shared_gu.dtype),
            shared_down=str(owner.shared_down.dtype),
            gates=sorted({str(t.dtype) for t in owner.gates}),
            ups=sorted({str(t.dtype) for t in owner.ups}),
            downs=sorted({str(t.dtype) for t in owner.downs}),
            routed_compute=kernel_identity(owner.routed_compute),
            shared_compute=kernel_identity(owner.shared_compute),
            router_compute=kernel_identity(owner.router_compute),
            shared_weights=True,
        )
        rng = torch.Generator().manual_seed(20261016512)
        independent = torch.randn(1, 1, 1024, 2560, generator=rng).bfloat16()
        patterns = dict(
            s512_control=independent[:, :, :512].contiguous(),
            s1024_independent=independent,
            s1024_repeated_capacity1024=independent[:, :, :1].expand(1, 1, 1024, 2560).contiguous(),
        )
        for name, host in patterns.items():
            owner.ops.large_prefill = host.shape[2] >= 256
            owner.ops.l1_prefill = True
            device = owner.ops.upload(host)
            route = {n: read_routes(view, device, n) for n in (512, 1024)}
            calls = {
                512: lambda: candidate.forward512(view, device),
                1024: lambda: candidate.forward1024(view, device),
            }
            for call in calls.values():
                output = call()
                ttnn.synchronize_device(mesh)
                del output
            raw, timings = {}, {}
            for group, call in calls.items():
                eager, replay, timing = measured(mesh, call, args.repeats, args.warmup)
                raw[group] = dict(eager=eager, replay=replay, route_ids=route[group][0], route_scores=route[group][1])
                timings[str(group)] = timing
                assert torch.equal(eager, replay), (name, group, "eager/replay")
                assert torch.isfinite(replay).all()
            ids, scores = route[1024]
            counts = torch.bincount(ids[scores != 0], minlength=512)
            record = dict(
                name=name,
                input_shape=list(host.shape),
                timings=timings,
                comparison512_to1024=pcc_metrics(raw[512]["replay"], raw[1024]["replay"]),
                routes_equal=torch.equal(route[512][0], route[1024][0]),
                route_scores_equal=torch.equal(route[512][1], route[1024][1]),
                max_tokens_per_expert=int(counts.max()),
                active_experts=int((counts > 0).sum()),
                speedup512_to1024=timings["512"]["median_seconds"] / timings["1024"]["median_seconds"],
            )
            path = args.output / (name + ".pt")
            torch.save(dict(input=host, variants=raw, counts=counts), path)
            record["raw"] = dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            report["results"].append(record)
            save()
            print(json.dumps(record), flush=True)
            assert record["comparison512_to1024"]["exact_equal"]
            assert record["routes_equal"] and record["route_scores_equal"]
            if name == "s1024_repeated_capacity1024":
                assert int(counts.max()) == 1024
            del device
        report["status"] = "measurements_complete_review_numerics"
        save()
    except BaseException:
        report.update(status="failed", error=traceback.format_exc())
        save()
        raise
    finally:
        if mesh is not None:
            ttnn.close_mesh_device(mesh)
            report["mesh_closed"] = True
            save()


if __name__ == "__main__":
    main()
