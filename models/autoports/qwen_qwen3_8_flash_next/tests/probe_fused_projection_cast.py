# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Test folding BF16 weight widening into the accurate binary SFPU unpack."""

import argparse
import json
import statistics
import time
from pathlib import Path

import torch
from ttnn.tools import trace_allocation_tracker

import ttnn
from models.autoports.qwen_qwen3_8_flash_next.tests.probe_components import load_weights
from models.autoports.qwen_qwen3_8_flash_next.tt.decoder_components import WeightLoader
from models.autoports.qwen_qwen3_8_flash_next.tt.fused_decoder import FusedPreciseProjection
from models.autoports.qwen_qwen3_8_flash_next.tt.precise_gdn_projection import PreciseGDNProjection


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--snapshot", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    assert trace_allocation_tracker.TRACE_ALLOC_TRACKING
    torch.set_num_threads(8)
    torch.manual_seed(2891)
    weights = {}
    for name in ("linear_attn.in_proj_qkv.weight", "linear_attn.out_proj.weight"):
        weights.update(load_weights(a.snapshot, 0, name))
    device = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0)
    report = dict(rows=[], passed=False)
    trace = None
    try:
        loader = WeightLoader(weights, device)
        for name in weights:
            original = PreciseGDNProjection(loader, name)
            reference = FusedPreciseProjection(original, loader.compute, 64, balanced_tree="qkv" in name)
            candidate = FusedPreciseProjection.__new__(FusedPreciseProjection)
            candidate.__dict__.update(reference.__dict__)
            candidate.widen_in_unpack = True
            direct = FusedPreciseProjection(
                original,
                loader.compute,
                64,
                balanced_tree="qkv" in name,
                widen_in_unpack=True,
                host_weight=weights[name],
            )
            for n in (1, 2, 7, 31, 32):
                x = loader.tensor((torch.randn(1, 1, n, original.input_width) * 0.02).bfloat16())
                expected = ttnn.to_torch(reference(x))
                actual = ttnn.to_torch(candidate(x))
                direct_output = ttnn.to_torch(direct(x))
                row = dict(
                    projection=name,
                    tokens=n,
                    exact=torch.equal(expected, actual),
                    direct_exact=torch.equal(expected, direct_output),
                    max_abs=float((expected.float() - actual.float()).abs().max()),
                    times={},
                )
                for label, fn in [("explicit_cast", reference), ("unpack_widen", candidate), ("direct_unpack", direct)]:
                    trace = ttnn.begin_trace_capture(device, cq_id=0)
                    output = fn(x)
                    ttnn.end_trace_capture(device, trace, cq_id=0)
                    trace_allocation_tracker.acknowledge_corruptible(output)
                    samples = []
                    for _ in range(10):
                        start = time.perf_counter()
                        ttnn.execute_trace(device, trace, cq_id=0, blocking=True)
                        samples.append((time.perf_counter() - start) * 1000)
                    replay = ttnn.to_torch(output)
                    assert torch.equal(replay, expected if label == "explicit_cast" else actual)
                    row["times"][label] = statistics.median(samples[2:])
                    ttnn.release_trace(device, trace)
                    trace = None
                report["rows"].append(row)
                print(row, flush=True)
        report["passed"] = all(r["exact"] and r["direct_exact"] for r in report["rows"])
    finally:
        if trace is not None:
            ttnn.release_trace(device, trace)
        ttnn.close_mesh_device(device)
        a.output.write_text(json.dumps(report, indent=2) + "\n")
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
