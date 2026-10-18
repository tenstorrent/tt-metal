# SPDX-License-Identifier: Apache-2.0
"""Standalone TEST ONLY B8 MoE padding-route probe; root owns device execution.

One real layer's selected weights, unchanged BF4 routed arithmetic and native
capacity128. No model/public cache state is created. Output guard rows and
finite poisoned inactive inputs are checked. Diagnostics/canary copies occur
in both measured graphs; measurements are component controls, not serving data.
"""

import argparse
import gc
import hashlib
import os
import statistics
import subprocess
import time
import traceback
from pathlib import Path

import torch
from transformers import AutoConfig
from ttnn.tools import trace_allocation_tracker

import ttnn
from models.autoports.qwen_qwen3_8_flash_next.tests import benchmark_decode_padding_route_candidate as candidate
from models.autoports.qwen_qwen3_8_flash_next.tests import benchmark_prefill_view_probe as common
from models.autoports.qwen_qwen3_8_flash_next.tests.benchmark_batch_moe_probe import kernel_identity
from models.autoports.qwen_qwen3_8_flash_next.tt import precision
from models.autoports.qwen_qwen3_8_flash_next.tt.checkpoint import CheckpointState
from models.autoports.qwen_qwen3_8_flash_next.tt.multichip_decoder import MeshOps, ParallelMoE


def exact(a, b):
    return torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))


def allocate(ops):
    return {
        "unmasked_scores": ops.upload(torch.zeros(1, 1, 8, 32)),
        "scores": ops.upload(torch.zeros(1, 1, 8, 32)),
        "ids": ops.upload(torch.zeros(1, 1, 8, 32, dtype=torch.int32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT),
        "counts": ops.upload(torch.zeros(128, dtype=torch.int32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT),
        "guarded_output": ops.upload(torch.full((1, 10, 1, 640), 0.125), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT),
    }


def run(args, report, save):
    mesh = None
    try:
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=args.trace_bytes)
        ops = MeshOps(mesh)
        cfg = AutoConfig.from_pretrained(args.checkpoint, local_files_only=True).text_config
        selected = precision.load_policy()
        policy = precision.for_layer(selected, args.layer)
        if selected["config_id"] != "combined_head4_gdn4_lofi" or cfg.max_position_embeddings != 262144:
            raise ValueError("Selected model/context/precision required")
        owner = ParallelMoE(CheckpointState(args.checkpoint, f"model.language_model.layers.{args.layer}."), cfg, ops)
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
        report["policy"] = selected
        report["actual_precision"] = dict(
            router=str(owner.router.dtype),
            shared_gu=str(owner.shared_gu.dtype),
            shared_down=str(owner.shared_down.dtype),
            gate=sorted({str(t.dtype) for t in owner.gates}),
            up=sorted({str(t.dtype) for t in owner.ups}),
            down=sorted({str(t.dtype) for t in owner.downs}),
            routed=kernel_identity(owner.routed_compute),
            shared=kernel_identity(owner.shared_compute),
            router_compute=kernel_identity(owner.router_compute),
            ccl=ops.precision_ccl,
        )
        rng = torch.Generator().manual_seed(81107)
        base = (torch.randn(1, 8, 1, 2560, generator=rng) * 0.5).bfloat16()
        cases = []
        # Repeated activity with fresh input signs tests dynamic replay and stale masks.
        for index, active in enumerate((5, 8, 6, 7, 5, 8)):
            host = base.clone() * (-1 if index % 2 else 1)
            host[:, active:] = 31.0 if index % 2 else -29.0
            mask = torch.zeros(1, 1, 8, 32, dtype=torch.bfloat16)
            mask[:, :, :active] = 1
            cases.append(dict(active=active, input=host, mask=mask))
        torch.save(cases, args.output / "inputs.pt")
        device = ops.upload(base)
        active_device = ops.upload(cases[0]["mask"])
        buffers = allocate(ops)
        # Every persistent snapshot is allocated before either trace begins.
        snapshots = {variant: [allocate(ops) for _ in cases] for variant in ("baseline", "masked")}
        # Warm destination override paths before a trace can pin any allocation.
        for rows in snapshots.values():
            for row in rows:
                for name, value in buffers.items():
                    ttnn.copy(value, row[name])
        ttnn.synchronize_device(mesh)
        report["geometry"] = {name: common.tensor_identity(value) for name, value in buffers.items()}
        report["geometry"]["active_mask"] = common.tensor_identity(active_device)
        report["results"] = []
        raw = {}
        for variant in ("baseline", "masked"):
            model, source = candidate.make(owner, active_device, buffers, masked=variant == "masked")
            (args.output / f"{variant}_group.py.txt").write_text(source)

            def forward():
                result = ParallelMoE.__call__(model, device)
                if tuple(result.shape) != (1, 8, 1, 640):
                    raise ValueError(f"Unexpected output geometry {result.shape}")
                ttnn.experimental.slice_write(
                    ttnn.to_layout(result, ttnn.ROW_MAJOR_LAYOUT),
                    buffers["guarded_output"],
                    [0, 1, 0, 0],
                    [1, 9, 1, 640],
                    [1, 1, 1, 1],
                )

            def refresh(case):
                common.host_copy(case["input"], device, mesh)
                common.host_copy(case["mask"], active_device, mesh)

            trace = None
            timings = []
            try:
                refresh(cases[0])
                forward()
                ttnn.synchronize_device(mesh)
                trace = ttnn.begin_trace_capture(mesh, cq_id=0)
                forward()
                ttnn.end_trace_capture(mesh, trace, cq_id=0)
                for index, case in enumerate(cases):
                    refresh(case)
                    ttnn.synchronize_device(mesh)
                    for _ in range(args.warmup):
                        ttnn.execute_trace(mesh, trace, cq_id=0, blocking=False)
                    ttnn.synchronize_device(mesh)
                    seconds = []
                    for _ in range(args.repeats):
                        begin = time.perf_counter()
                        ttnn.execute_trace(mesh, trace, cq_id=0, blocking=False)
                        ttnn.synchronize_device(mesh)
                        seconds.append(time.perf_counter() - begin)
                    timings.append(
                        dict(active=case["active"], seconds=seconds, median_seconds=statistics.median(seconds))
                    )
                    for name, tensor in buffers.items():
                        ttnn.copy(tensor, snapshots[variant][index][name])
                    ttnn.synchronize_device(mesh)
                ttnn.release_trace(mesh, trace)
                trace = None
            finally:
                if trace is not None:
                    ttnn.release_trace(mesh, trace)
            # No diagnostics or new persistent device allocations while a trace is live.
            raw[variant] = [
                {key: common.read_shards(value) for key, value in row.items()} for row in snapshots[variant]
            ]
            report[variant + "_timings"] = timings
            if not all(exact(value, cases[-1]["input"]) for value in common.read_shards(device)):
                raise AssertionError("MoE mutated its input")
            save()
        torch.save(raw, args.output / "raw.pt")
        report["raw_sha256"] = common.sha(args.output / "raw.pt")
        for index, case in enumerate(cases):
            active = case["active"]
            a, b = raw["baseline"][index], raw["masked"][index]
            chips = []
            for chip in range(4):
                ids = a["ids"][chip].reshape(8, 32).long()
                original = a["unmasked_scores"][chip].reshape(8, 32)
                gated = b["scores"][chip].reshape(8, 32)
                expected_before = candidate.route_counts(ids.tolist(), original.tolist(), chip)
                expected_after = candidate.route_counts(ids.tolist(), original.tolist(), chip, active)
                before = a["counts"][chip].reshape(-1).long().tolist()
                after = b["counts"][chip].reshape(-1).long().tolist()
                record = dict(
                    chip=chip,
                    route_ids_exact=exact(a["ids"][chip], b["ids"][chip]),
                    unmasked_scores_exact=exact(a["unmasked_scores"][chip], b["unmasked_scores"][chip]),
                    active_scores_exact=exact(original[:active], gated[:active]),
                    inactive_scores_zero=bool(torch.all(gated[active:] == 0)),
                    active_output_exact=exact(
                        a["guarded_output"][chip][:, 1 : 1 + active], b["guarded_output"][chip][:, 1 : 1 + active]
                    ),
                    active_output_finite=bool(torch.isfinite(b["guarded_output"][chip][:, 1 : 1 + active]).all()),
                    guard_canaries=all(
                        bool(torch.all(row["guarded_output"][chip][:, [0, 9]] == 0.125)) for row in (a, b)
                    ),
                    counts_match=before == expected_before and after == expected_after,
                    original_counts=before,
                    masked_counts=after,
                    original_unique_experts=sum(v > 0 for v in before),
                    active_unique_experts=sum(v > 0 for v in after),
                    padding_exclusive_experts=sum(x > 0 and y == 0 for x, y in zip(before, after)),
                    removed_routes=sum(before) - sum(after),
                )
                chips.append(record)
                flags = (
                    "route_ids_exact",
                    "unmasked_scores_exact",
                    "active_scores_exact",
                    "inactive_scores_zero",
                    "active_output_exact",
                    "active_output_finite",
                    "guard_canaries",
                    "counts_match",
                )
                if not all(record[name] for name in flags):
                    raise AssertionError({name: record[name] for name in flags})
            report["results"].append(
                dict(
                    index=index,
                    active=active,
                    chips=chips,
                    median_speedup=report["baseline_timings"][index]["median_seconds"]
                    / report["masked_timings"][index]["median_seconds"],
                )
            )
            save()
        report["status"] = "passed_exact_component"
    finally:
        if mesh is not None:
            ttnn.close_mesh_device(mesh)
            report["mesh_closed"] = True
        save()
        gc.collect()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--checkpoint", type=Path, default=Path(f"/home/vkovacevic/qwen38-flash/hf-local/{common.REVISION}")
    )
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--trace-bytes", type=int, default=268435456)
    args = parser.parse_args()
    if args.repeats < 2 or args.warmup < 1:
        parser.error("At least two repeats and one warmup required")
    if (
        os.environ.get("TT_METAL_TRACE_ALLOC_TRACKING") != "1"
        or os.environ.get("TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE", "0") != "0"
    ):
        raise RuntimeError("Full trace allocation tracking required")
    if (
        not ttnn._ttnn.operations.trace.trace_allocation_tracking_enabled()
        or not trace_allocation_tracker.TRACE_ALLOC_TRACKING
    ):
        raise RuntimeError("Native and Python trace tracking required")
    if any(os.environ.get(k, "0") not in ("", "0") for k in ("TT_METAL_DEVICE_PROFILER", "TT_METAL_WATCHER")):
        raise RuntimeError("Standalone unprofiled probe required")
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(8)
    paths = [
        Path(__file__),
        Path(candidate.__file__),
        precision.ROOT / "tt/multichip_decoder.py",
        Path(precision.__file__),
        Path(common.__file__),
    ]
    report = dict(
        status="running",
        source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        layer=args.layer,
        checkpoint=str(args.checkpoint),
        batch=8,
        mesh=[1, 4],
        sources={str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths},
        trace_tracking=dict(native=True, python=True, skip_program_cache=False, exemptions=[]),
        scope="One real-weight MoE including original scatter-reduce; no decoder/public state/full-model claim",
        timing="Warmed trace launch through standalone synchronization; identical diagnostic and output guard copies included; refresh/snapshots/readback excluded",
        poison="Finite BF16 +31/-29 inactive rows; no claim of NaN router support",
        acceptance="Exactness only; root reviews measured benefit before any adoption; no predicted speedup",
    )
    for path in paths:
        (args.output / (path.name + ".txt")).write_bytes(path.read_bytes())

    def save():
        common.dump(args.output / "summary.json", report)

    save()
    try:
        run(args, report, save)
    except BaseException:
        report["status"] = "failed"
        report["error"] = traceback.format_exc()
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
