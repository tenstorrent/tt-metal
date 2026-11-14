# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Lossless speculative rejection sampling for DFlash2 (port of z-lab DFlash _rejection_sample)."""

import torch

from models.demos.muse_glimmer_30b.tt.sampling import draw


def _prob_of(probs_row, ids_row, token):
    return float(probs_row[ids_row == token].sum())


def rejection_sample(drafts, q_probs, q_ids, target_probs, target_ids, generator):
    """drafts: L ints drawn from q row i. q_probs/q_ids [L, kq]; target_probs/target_ids [L+1, k].
    Returns (accepted, bonus): the number of accepted drafts and the token drawn after them."""
    length = len(drafts)
    u = torch.rand(length, generator=generator)
    accepted = 0
    for i in range(length):
        p_i = _prob_of(target_probs[i], target_ids[i], drafts[i])
        q_i = _prob_of(q_probs[i], q_ids[i], drafts[i])
        if not float(u[i]) * q_i < p_i:
            break
        accepted += 1
    if accepted == length:
        return accepted, draw(target_probs[length], target_ids[length], generator)
    q_on_target = (
        q_probs[accepted].unsqueeze(0) * (q_ids[accepted].unsqueeze(0) == target_ids[accepted].unsqueeze(-1))
    ).sum(-1)
    residual = torch.clamp(target_probs[accepted] - q_on_target, min=0)
    total = residual.sum()
    row = residual / total if total > 0 else target_probs[accepted]
    return accepted, draw(row, target_ids[accepted], generator)
