# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Is a model worth device time? Cheap host-side screens before preparing caches and running A/Bs.

``imbalance`` measures how unevenly the ORIGINAL layout loads the chips on captured traffic: utilization =
mean chip tiles / busiest chip tiles (1.0 = even). Low utilization means there is imbalance to remove.
Reference points on 8x4 Blackhole Galaxy prefill: GLM-5.3 0.42 -> -4.1% measured; Kimi-K3 0.72 -> -0.9%.

``modeled_gain`` scores presets on a sequence's routing (busiest-chip tiles summed over chunks and layers).

``expected_gain`` turns a modeled gain into a rough on-device range: modeled x routed-expert share of layer
time (from an adapter's stage profile, if available) x a scaling factor. The scaling factor (how much routed
expert time actually follows busiest-chip tokens) is unknown for a new model; the range uses [0.1, 1].
"""

from __future__ import annotations

import numpy as np

from .spec import MoeSpec


def _chip_tiles(ids_layer_chunk: np.ndarray, owners: np.ndarray, spec: MoeSpec) -> np.ndarray:
    cnt = np.bincount(ids_layer_chunk.ravel().astype(np.int64), minlength=spec.num_experts)
    tiles = (cnt + 31) // 32
    chips = np.zeros((spec.groups, spec.rows))
    np.add.at(chips, (np.arange(spec.num_experts) // spec.group_size, owners), tiles)
    return chips


def peak_tiles(ids: np.ndarray, spec: MoeSpec, owners_per_layer) -> tuple[float, float]:
    """(sum of busiest-chip tiles, sum of mean-chip tiles) over all layers and chunks of ids [L, T, K]."""
    L, T, _ = ids.shape
    pk = mn = 0.0
    for li in range(L):
        for c in range(T // spec.chunk_tokens):
            chips = _chip_tiles(
                ids[li, c * spec.chunk_tokens : (c + 1) * spec.chunk_tokens], owners_per_layer[li], spec
            )
            pk += chips.max()
            mn += chips.mean()
    return pk, mn


def imbalance(ids: np.ndarray, spec: MoeSpec) -> dict:
    owners = [spec.identity_owners()] * ids.shape[0]
    pk, mn = peak_tiles(ids, spec, owners)
    u = mn / pk
    return dict(utilization=round(u, 4), busiest_over_mean=round(1 / u, 3), experts_per_chip=spec.experts_per_chip)


def modeled_gain(ids: np.ndarray, spec: MoeSpec, plans: dict) -> dict:
    """plans: name -> per-layer owners. Returns peak-tile change vs the original layout, in %."""
    base, _ = peak_tiles(ids, spec, [spec.identity_owners()] * ids.shape[0])
    return {name: round(100 * (peak_tiles(ids, spec, owners)[0] / base - 1), 3) for name, owners in plans.items()}


def expected_gain(modeled_pct: float, expert_share: float | None = None) -> dict:
    share = 1.0 if expert_share is None else expert_share
    lo, hi = modeled_pct * share * 0.1, modeled_pct * share
    return dict(
        modeled_pct=modeled_pct,
        routed_expert_share=expert_share,
        expected_pct_range=[round(min(lo, hi), 2), round(max(lo, hi), 2)],
    )


def verdict(
    utilization: float, modeled_pct: float, min_modeled_pct: float = -3.0, max_utilization: float = 0.8
) -> dict:
    go = utilization <= max_utilization and modeled_pct <= min_modeled_pct
    why = []
    if utilization > max_utilization:
        why.append(f"original layout already even (utilization {utilization:.2f} > {max_utilization})")
    if modeled_pct > min_modeled_pct:
        why.append(f"modeled gain {modeled_pct:.1f}% smaller than {min_modeled_pct}%")
    return dict(go=go, reasons=why or ["imbalance and modeled gain large enough for an on-device A/B"])
