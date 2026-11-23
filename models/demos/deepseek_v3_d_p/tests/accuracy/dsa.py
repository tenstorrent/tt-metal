# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Sparse attention (DSA) accuracy against a reference: what a different key selection costs.

Keys (latent, rope part, index keys) and the attention output are continuous products, scored by
``windowed.error_metrics`` (absolute and relative). Selection recall and misses come from ``topk``. This module adds
the attention weight a selection difference moves, given the reference attention weights over each selection
(computing them is the model's). These are diagnostics: no bar is defined on them.
"""

from __future__ import annotations

import torch


def selection_masses(
    want_weights: torch.Tensor, got_weights: torch.Tensor, want_sel: torch.Tensor, got_sel: torch.Tensor
) -> dict:
    """One query row: ``want_sel`` the reference selection (valid keys, best first by the reference indexer),
    ``got_sel`` the device's (valid, distinct keys); ``want_weights`` / ``got_weights`` ``[heads, |sel|]`` the
    reference attention softmax over each selection. Per head (``[heads]``):

    * ``missed_mass``: weight, of the softmax over ``want_sel``, on the keys ``got_sel`` misses;
    * ``bottom_mass``: weight, of the same softmax, on as many of ``want_sel``'s lowest-ranked keys (what a pure
      near-tie miss would drop);
    * ``intruder_mass``: weight, of the softmax over ``got_sel``, on its keys ``want_sel`` lacks;

    and the counts ``missed``, ``intruders`` and ``missed_key_max``, the largest weight of one missed key in one
    head (0 with no miss)."""
    missed = ~torch.isin(want_sel, got_sel)
    intruder = ~torch.isin(got_sel, want_sel)
    n = int(missed.sum())
    bottom = torch.zeros_like(missed)
    if n:
        bottom[len(want_sel) - n :] = True
    return {
        "missed": n,
        "intruders": int(intruder.sum()),
        "missed_mass": want_weights[:, missed].sum(-1),
        "bottom_mass": want_weights[:, bottom].sum(-1),
        "intruder_mass": got_weights[:, intruder].sum(-1),
        "missed_key_max": float(want_weights[:, missed].max()) if n else 0.0,
    }


def error_split(want: torch.Tensor, want_on_got_sel: torch.Tensor, got: torch.Tensor) -> dict[str, float]:
    """Split a device output's error ``got − want`` into the selection part ``want_on_got_sel − want`` (the
    reference attention over the device's selection) and the rest ``got − want_on_got_sel``: their cosine (0 when a
    part is zero, e.g. identical selections) and each part's share of the squared error (the shares plus the cross
    term ``2 cos |sel| |rest| / |total|²`` sum to one)."""
    sel, rest = want_on_got_sel - want, got - want_on_got_sel
    total = float((got - want).square().sum())
    norms = float(sel.norm() * rest.norm())
    return {
        "selection_rest_cos": float(sel @ rest) / norms if norms else 0.0,
        "selection_share": float(sel.square().sum()) / total,
        "rest_share": float(rest.square().sum()) / total,
    }
