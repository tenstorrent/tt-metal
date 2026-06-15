# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
Scope control for compute perf counters: down-sample cores per op-grid.

Compute counters are SPMD, so a handful of cores represent an op's compute.
Sampling shrinks the per-core x per-op x per-counter row explosion that drives
post-process cost and OOM at mesh scale.

What survives sampling: the per-core utilization distribution stats
(FPU/SFPU/MATH Util Min/Median/Max/Avg %), because they are ratios computed
per core and then aggregated. What does NOT survive: grid-summed metrics that
divide a Σ-over-cores count by the full grid size ("Avg ... on full grid (%)",
avg_*_count) -- those undercount by grid/k and must not be read when sampling.

Two invariants make the sample trustworthy:
  * Per-op-grid: cores are drawn from each op's own recorded grid, never a
    fixed device-wide mask -- a fixed mask silently drops ops whose grid
    excludes the masked cores.
  * Deterministic: the same op grid always yields the same cores (evenly
    spaced over the sorted grid, no RNG), so two runs report the same numbers.

This is compute-only. NoC/bandwidth counters are an across-cores aggregate and
must NOT be sampled -- per-core NoC load is non-uniform, so sampling undercounts
bandwidth. Keep NoC capture full and aggregate per op (Phase 3).
"""


def _evenly_spaced_indices(n, k):
    """k indices spread across range(n), endpoints included, deterministic."""
    if k >= n:
        return list(range(n))
    if k == 1:
        return [0]
    step = (n - 1) / (k - 1)
    return sorted({round(i * step) for i in range(k)})


def sample_cores_per_op(perf_counter_df, k, op_keys=("run_host_id", "trace_id_count")):
    """Keep only k cores per op, drawn evenly from each op's own grid.

    Returns a filtered copy of perf_counter_df. k <= 0 or k >= an op's grid
    size leaves that op untouched.
    """
    if k is None or k <= 0:
        return perf_counter_df

    keep_parts = []
    for _op, op_df in perf_counter_df.groupby(list(op_keys), sort=True):
        cores = sorted({(int(cx), int(cy)) for cx, cy in zip(op_df["core_x"], op_df["core_y"])})
        if k >= len(cores):
            keep_parts.append(op_df)
            continue
        chosen = {cores[i] for i in _evenly_spaced_indices(len(cores), k)}
        mask = [(int(cx), int(cy)) in chosen for cx, cy in zip(op_df["core_x"], op_df["core_y"])]
        keep_parts.append(op_df[mask])

    import pandas as pd

    return pd.concat(keep_parts) if keep_parts else perf_counter_df
