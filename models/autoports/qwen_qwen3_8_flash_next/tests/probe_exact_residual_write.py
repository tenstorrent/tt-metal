# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""TP4 local2560 exact residual writer control; no production monkey-patching."""

import argparse
import gc
import hashlib
import json
import os
import statistics
import time
import traceback
from pathlib import Path

import torch
from ttnn.tools import trace_allocation_tracker as tracker

import ttnn
from models.autoports.qwen_qwen3_8_flash_next.tests.probe_linear_contract import runtime_audit
from models.autoports.qwen_qwen3_8_flash_next.tt.exact_residual_write import exact_residual_write
from models.autoports.qwen_qwen3_8_flash_next.tt.multichip_decoder import MeshLoader, MeshResidual


def fixtures(count):
    """Independent rank operands, random and BF16 midpoint/cancellation stress."""
    families = []
    for stress in (False, True):
        ranks = []
        for rank in range(4):
            rng = torch.Generator().manual_seed(620 + 1000 * count + 31 * rank)
            shape = (1, 1, count, 2560)
            block = torch.randn(shape, generator=rng).bfloat16()
            injection = torch.randn((1, 1, count, 1), generator=rng).bfloat16()
            residual = torch.randn(shape, generator=rng).bfloat16()
            if stress:
                values = torch.tensor(
                    [1, 1.0078125, 1.9921875, -1, -1.0078125, -1.9921875, 0.00390625, -0.00390625],
                    dtype=torch.bfloat16,
                ).roll(rank)
                block = values.repeat(count * 320).reshape(shape)
                factors = torch.tensor([1.5, -1.5, 1.0078125, -1.0078125], dtype=torch.bfloat16).roll(rank)
                injection = factors.repeat((count + 3) // 4)[:count].reshape(1, 1, count, 1)
                rounded_update = (block.float() * injection.float()).bfloat16()
                residual = (-rounded_update.float() + (rank + 1) * 0.001953125).bfloat16()
            ranks.append((residual, block, injection))
        assert all(not torch.equal(ranks[0][1], ranks[r][1]) for r in (1, 2, 3))
        families.append(ranks)
    return families


def read_ranks(value):
    return [ttnn.to_torch(t) for t in ttnn.get_device_tensors(value)]


def exact(actual, expected):
    assert len(actual) == len(expected) == 4
    result = [
        {
            "rank": rank,
            "exact": torch.equal(a.view(torch.int16), b.view(torch.int16)),
            "max_abs": float((a.float() - b.float()).abs().max()),
        }
        for rank, (a, b) in enumerate(zip(actual, expected, strict=True))
    ]
    assert all(item["exact"] for item in result), result
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--clean-timing", action="store_true")
    parser.add_argument("--repeats", type=int, default=64)
    parser.add_argument("--rounds", type=int, default=3)
    args = parser.parse_args()
    assert args.repeats > 0 and args.rounds > 0
    assert bool(tracker.TRACE_ALLOC_TRACKING) != args.clean_timing
    assert not args.clean_timing or not os.environ.get("TT_METAL_WATCHER")
    torch.set_num_threads(8)
    root = Path(__file__).resolve().parents[1]
    sources = [
        Path(__file__),
        root / "tt/exact_residual_write.py",
        root / "tt/multichip_decoder.py",
        root / "tt/fused_decoder.py",
    ]
    sources += sorted((root / "tt/kernels").glob("exact_residual_write_*.cpp"))
    report = {
        "passed": False,
        "rows": [],
        "mesh": [1, 4],
        "scope": "Synthetic actual-shape component control, not full-model acceptance or speedup",
        "tracking": bool(tracker.TRACE_ALLOC_TRACKING),
        "clean_timing": args.clean_timing,
        "watcher": os.environ.get("TT_METAL_WATCHER"),
        "sources_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        "boundary": "Preloaded persistent inputs; nonblocking replay loop, one synchronization at window end; no per-replay read/copy/sync",
        "preserved_math": "BF16(FP32(residual) + FP32(BF16(FP32(block)*FP32(injection))))",
        "equality": "BF16 storage bits, including signed zero",
    }
    mesh, trace = None, None
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def phase(label):
        report["last_phase"] = label
        print(f"EXACT_WRITER_PHASE {label}", flush=True)

    try:
        phase("open_mesh")
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=0)
        loader = MeshLoader({}, mesh)
        owner = MeshResidual.__new__(MeshResidual)
        for count in (1, 3, 17, 32):
            phase(f"q{count}:setup_and_baseline_eager")
            hosts = fixtures(count)
            inputs = [[loader.shards([rank[i] for rank in family], dim=3) for i in range(3)] for family in hosts]
            expected = []
            for values in inputs:
                with runtime_audit():
                    output = owner.write(*values)
                expected.append(read_ranks(output))
                del output
            row = {"count": count, "eager": [], "traces": {}}
            report["rows"].append(row)
            for index, values in enumerate(inputs):
                phase(f"q{count}:native_eager:family{index}")
                with runtime_audit():
                    output = exact_residual_write(*values)
                row["eager"].append(
                    {
                        "family": ("random", "rounding_cancellation")[index],
                        "ranks": exact(read_ranks(output), expected[index]),
                    }
                )
                del output
            # All immutable input sets and persistent replay inputs predate capture.
            live = [ttnn.empty_like(t) for t in inputs[0]]
            for name, function in (("eight_op_baseline", owner.write), ("native_local2560", exact_residual_write)):
                phase(f"q{count}:{name}:warm")
                for src, dst in zip(inputs[0], live, strict=True):
                    ttnn.copy(src, dst)
                with runtime_audit():
                    warm = function(*live)
                del warm
                gc.collect()
                phase(f"q{count}:{name}:capture")
                trace = ttnn.begin_trace_capture(mesh, cq_id=0)
                try:
                    with runtime_audit():
                        output = function(*live)
                finally:
                    ttnn.end_trace_capture(mesh, trace, cq_id=0)
                item = {
                    "changed_input_replays": [],
                    "windows_ms_per_replay": [],
                    "runtime_audit": "passed_eager_warm_capture",
                }
                row["traces"][name] = item
                if tracker.TRACE_ALLOC_TRACKING:
                    phase(f"q{count}:{name}:allocation_guard_positive_control")
                    # This trace owns and overwrites its sole retained capture output.
                    tracker.acknowledge_corruptible(output)
                    sentinel = ttnn.empty_like(live[0])
                    try:
                        try:
                            ttnn.execute_trace(mesh, trace, cq_id=0, blocking=False)
                        except RuntimeError as error:
                            assert "still alive before trace replay" in str(error), str(error)
                            item["allocation_guard_positive_control"] = {"rejected": True, "error": str(error)}
                        else:
                            raise AssertionError("Allocation guard did not reject retained post-capture sentinel")
                    finally:
                        ttnn.deallocate(sentinel)
                        del sentinel
                phase(f"q{count}:{name}:changed_input_replay")
                for repeat in range(12):
                    family = repeat % 2
                    for src, dst in zip(inputs[family], live, strict=True):
                        ttnn.copy(src, dst)
                    ttnn.execute_trace(mesh, trace, cq_id=0, blocking=False)
                    ttnn.synchronize_device(mesh)
                    item["changed_input_replays"].append(
                        {"family": family, "ranks": exact(read_ranks(output), expected[family])}
                    )
                # Keep the last stress family resident; correctness checked after every window.
                phase(f"q{count}:{name}:timing")
                for _ in range(args.rounds):
                    ttnn.synchronize_device(mesh)
                    start = time.perf_counter()
                    for _ in range(args.repeats):
                        ttnn.execute_trace(mesh, trace, cq_id=0, blocking=False)
                    ttnn.synchronize_device(mesh)
                    item["windows_ms_per_replay"].append((time.perf_counter() - start) * 1000 / args.repeats)
                    exact(read_ranks(output), expected[1])
                item["median_ms"] = statistics.median(item["windows_ms_per_replay"])
                item["repeats_per_window"] = args.repeats
                ttnn.release_trace(mesh, trace)
                trace = None
                del output
            row["speedup"] = (
                row["traces"]["eight_op_baseline"]["median_ms"] / row["traces"]["native_local2560"]["median_ms"]
            )
            print(json.dumps({"count": count, "speedup": row["speedup"]}), flush=True)
            del live, inputs
        report["passed"] = True
        phase("passed")
    except BaseException:
        report["error"] = traceback.format_exc()
        raise
    finally:
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        if mesh is not None:
            if trace is not None:
                ttnn.release_trace(mesh, trace)
            ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
