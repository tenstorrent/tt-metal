# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Math utilization of vsa_sdpa from a tracy ops CSV + the run's geometry.

Utilization = computed attention FLOPs / (device kernel time x grid peak), per device, using
tt-perf-report's Blackhole peak model (4096 FLOP/cycle x 1.35 GHz, / 2 for HiFi2). FLOPs count the
math the block mask requires -- QK + PV over each row's listed 64-token blocks (exempt-query rows
are fully dense) -- so wasted work (pad tiles, re-reads, stalls) lowers utilization, as it should.

Usage: python vsa_util_report.py <ops.csv> <duration_s> [sparsity=0.9] [placement=identity]
"""

from __future__ import annotations

import csv as csv_module
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[4]))

import torch

from models.tt_dit.pipelines.minimax_h3.vsa_geometry import build_vsa_geometry
from models.tt_dit.models.transformers.minimax_h3.vsa_stages_minimax_h3 import compute_topk
from models.tt_dit.tests.models.minimax_h3.test_performance_minimax_h3 import _packed_sizes

TFLOPS_PER_CORE_HIFI2 = 4096 * 1.35 / 1000 / 2  # tt-perf-report's Blackhole model
HEADS_PER_DEVICE = 14
SP = 8


def vsa_flops_per_device(duration_s: float, sparsity: float, placement: str) -> dict[int, float]:
    """Computed FLOPs of one vsa_sdpa call per SP shard (TP shards are identical)."""
    sizes = _packed_sizes(duration_s)
    grid = (sizes["latent_frames"], sizes["grid_h"], sizes["grid_w"])
    geometry = build_vsa_geometry((sizes["num_text"], 0, sizes["num_audio"]), grid, sp_factor=SP, placement=placement)
    k = compute_topk(sparsity, int(geometry.is_candidate.sum()))
    n_exempt = int(geometry.is_exempt.sum())
    n_real = int((geometry.valid_counts > 0).sum())
    rows_per_shard = geometry.tiles_per_shard

    flops = {}
    per_visit = 2 * (2 * 64 * 64 * 128)  # QK + PV for one (row, block) pair
    row_exempt = geometry.is_exempt.reshape(SP, rows_per_shard)
    for shard in range(SP):
        dense_rows = int(row_exempt[shard].sum())
        sparse_rows = rows_per_shard - dense_rows
        visits = dense_rows * n_real + sparse_rows * (n_exempt + k)
        flops[shard] = visits * HEADS_PER_DEVICE * per_visit
    return flops


def op_times_per_device(csv_path: str, op_code: str = "VsaSdpaOperation") -> dict[int, list[float]]:
    times = defaultdict(list)
    with open(csv_path) as f:
        for row in csv_module.DictReader(f):
            if row["OP CODE"].strip() == op_code:
                times[int(row["DEVICE ID"])].append(float(row["DEVICE KERNEL DURATION [ns]"]))
    return times


def main() -> None:
    csv_path, duration = sys.argv[1], float(sys.argv[2])
    sparsity = float(sys.argv[3]) if len(sys.argv) > 3 else 0.9
    placement = sys.argv[4] if len(sys.argv) > 4 else "identity"

    times = op_times_per_device(csv_path)
    if not times:
        print("no VsaSdpaOperation rows in the CSV")
        return
    flops = vsa_flops_per_device(duration, sparsity, placement)

    # tracy device ids don't map 1:1 to SP shards; report the time spread and bound utilization
    # with the worst device against the heaviest shard's FLOPs and the mean against the mean.
    per_dev_ns = {d: sum(v) for d, v in times.items()}
    worst_ns = max(per_dev_ns.values())
    mean_ns = sum(per_dev_ns.values()) / len(per_dev_ns)
    peak = TFLOPS_PER_CORE_HIFI2 * 130 * 1e12  # full 13x10 grid
    f_max, f_mean = max(flops.values()), sum(flops.values()) / len(flops)

    print(f"vsa_sdpa @ {duration:g}s (sparsity {sparsity}, {placement}):")
    print(f"  per-device op time: max {worst_ns/1e6:8.3f} ms, mean {mean_ns/1e6:8.3f} ms over {len(per_dev_ns)} devices")
    print(f"  computed FLOPs/device: max {f_max/1e12:.4f} TF, mean {f_mean/1e12:.4f} TF")
    print(f"  utilization: worst-dev {100*f_max/(worst_ns*1e-9*peak):6.2f} %   mean {100*f_mean/(mean_ns*1e-9*peak):6.2f} %")
    print(f"  60% target time (heaviest shard): {f_max/(0.6*peak)*1e3:8.3f} ms")


if __name__ == "__main__":
    main()
