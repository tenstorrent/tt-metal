# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Load-balanced placement of routed experts INSIDE each dispatch group (V41_MOE_PLACE, default off).

Every MoE kernel (dispatch table, offset_cumsum, the combine reader's ``offset = group * experts_per_group + row *
experts_per_chip``) assumes the default column-major layout: global expert e lives on mesh (row (e % G_size) // E,
col e // G_size) in local slot e % E. Rather than teach the kernels a new layout, a placement RELABELS experts: expert e
gets the id e' of the slot it should occupy, so every kernel still sees the default layout.

- the gate's weight rows and correction bias are permuted (row e' <- expert e): each logit is the same dot product in
  any column, so the top-k picks the same experts and emits e' (an exact fp32 score tie at the top-6 boundary is the
  only way the order could differ; the byte-identity gate is the arbiter);
- the routed-expert weights for slot e' are expert e's, moved as whole per-device host shards (the bytes do not change).

Only moves INSIDE a dispatch group (mesh column) are allowed: each chip's reduce sums its group's top-k slots and the TP
reduce-scatter adds the groups, so a token's per-group sum (and its order) is unchanged. Moving an expert across groups
would change that grouping and is rejected.

V41_MOE_PLACE=<json>: {"layers": {"<L>": {"placement": {"<e>": [group, chip], ...}}}} (route_balance.py output);
V41_MOE_PLACE_LAYERS=1,2,... optionally limits it to those layers.
"""

from __future__ import annotations

import json
import os
import re
from functools import lru_cache
from typing import Optional

from loguru import logger


@lru_cache(maxsize=None)
def _load(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def layer_of(cache_name_prefix: Optional[str]) -> Optional[int]:
    m = re.match(r"layer_(\d+)\.", cache_name_prefix or "")
    return int(m.group(1)) if m else None


def new_ids_for(
    cache_name_prefix: Optional[str], num_routed_experts: int, mesh_rows: int, mesh_cols: int
) -> Optional[list[int]]:
    """new_id_of[e] for this layer (a permutation that keeps every expert in its dispatch group), or None."""
    path = os.environ.get("V41_MOE_PLACE")
    layer = layer_of(cache_name_prefix)
    if not path or layer is None:
        return None
    only = os.environ.get("V41_MOE_PLACE_LAYERS")
    if only and str(layer) not in {s.strip() for s in only.split(",") if s.strip()}:
        return None
    rec = _load(path).get("layers", {}).get(str(layer))
    if not rec:
        return None
    n, rows, cols = num_routed_experts, mesh_rows, mesh_cols
    per_chip = n // (rows * cols)
    per_group = rows * per_chip
    assert per_chip * rows * cols == n, (n, rows, cols)
    slots: dict[tuple[int, int], list[int]] = {}
    for e_str, (g, c) in rec["placement"].items():
        e = int(e_str)
        assert e // per_group == int(g), f"layer {layer}: expert {e} moved out of its dispatch group ({g})"
        slots.setdefault((int(g), int(c)), []).append(e)
    new_id_of = [-1] * n
    for (g, c), experts in slots.items():
        assert len(experts) == per_chip, f"layer {layer}: chip ({g}, {c}) gets {len(experts)} experts, not {per_chip}"
        for i, e in enumerate(sorted(experts)):
            new_id_of[e] = (g * rows + c) * per_chip + i
    assert sorted(new_id_of) == list(range(n)), f"layer {layer}: placement is not a permutation"
    moved = sum(1 for e in range(n) if new_id_of[e] != e)
    logger.info(f"[moe place] layer {layer}: {moved} of {n} experts relabelled inside their dispatch groups ({path})")
    return new_id_of


def inverse(new_id_of: list[int]) -> list[int]:
    inv = [0] * len(new_id_of)
    for e, e2 in enumerate(new_id_of):
        inv[e2] = e
    return inv


def permute_expert_shards(host_tensors: list, new_id_of: list[int], mesh_rows: int, mesh_cols: int, mesh_shape) -> list:
    """host_tensors[i] = the multi-device HOST tensor of local slot i (device at mesh (row, col) holds expert
    (col * rows + row) * per_chip + i). Returns the per-slot host tensors with expert e's shard moved to slot
    new_id_of[e]. Shards are whole host buffers: no value changes."""
    import ttnn

    per_chip = len(host_tensors)
    shards = [ttnn.get_device_tensors(t) for t in host_tensors]  # row-major mesh coordinates
    ndev = mesh_rows * mesh_cols
    assert all(len(s) == ndev for s in shards), [len(s) for s in shards]
    out = [[None] * ndev for _ in range(per_chip)]
    for e, e2 in enumerate(new_id_of):
        d_old, i_old = divmod(e, per_chip)  # device index in the column-major expert numbering
        d_new, i_new = divmod(e2, per_chip)
        g_old, r_old = divmod(d_old, mesh_rows)
        g_new, r_new = divmod(d_new, mesh_rows)
        assert g_old == g_new
        out[i_new][r_new * mesh_cols + g_new] = shards[i_old][r_old * mesh_cols + g_old]
    assert all(x is not None for row in out for x in row)
    return [ttnn.from_host_shards(row, mesh_shape) for row in out]
