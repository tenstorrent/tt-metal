# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""M7 plan item 2: DFlash2's candidate-selector walk vectorized over users (tt/dflash2.py select_path, per user).

select_path scores every edge a step can take (predecessor = the anchor at position 0, else one of the previous
position's candidates) and then walks greedily, first maximum winning ties. Here the same scores are formed for U users
at once with one batched matmul, and the walk is K argmax/gather steps over [U, n]: no Python loop over users or
candidates.

  unary [U, K, n] fp32, cand [U, K, n] int64, hp [U, K, rank] fp32, anchors [U] int64,
  pred_cb / succ_cb [V, rank] fp32  ->  drafts [U, K] int64

hp = final_h @ W_hp (W_hp = candidate_selector.hidden_projection.weight.T, [H, rank]); hidden_projection() does that
for U * K rows in one matmul.
"""

import torch


def hidden_projection(final_h, w_hp):
    """[U, K, H] or [N, H] final hidden rows (bf16 or fp32) x W_hp [H, rank] -> fp32 [..., rank]"""
    return torch.matmul(final_h.float(), w_hp.float())


def select_paths(unary, cand, hp, anchors, pred_cb, succ_cb):
    """Greedy selector paths for U users (select_path's walk, batched). Returns drafts [U, K] int64."""
    U, K, n = cand.shape
    preds = torch.cat([anchors.view(U, 1, 1).expand(U, 1, n), cand[:, :-1]], dim=1)  # [U, K, n]
    context = pred_cb[preds] * hp[:, :, None, :]  # [U, K, n, rank]
    edges = torch.matmul(context, succ_cb[cand].transpose(-1, -2))  # [U, K, n (pred), n (cand)]
    score = unary[:, :, None, :] + edges
    users = torch.arange(U)
    i = torch.zeros(U, dtype=torch.int64)  # position 0: every predecessor row is the anchor; select_path reads row 0
    drafts = torch.empty(U, K, dtype=torch.int64)
    for pos in range(K):
        row = score[users, pos, i]  # [U, n]
        i = torch.argmax(row, dim=-1)  # first maximum (torch.argmax returns the first maximal index)
        drafts[:, pos] = cand[users, pos, i]
    return drafts


def unkey(keys):
    """the D-program's top-16 order keys (d_topk.hpp) back to their bf16 values, as fp32: an int64 tensor of keys in,
    a float32 tensor of the same shape out (vectorized r64/d/d_alpha0*.py unkey)"""
    k = keys & 0xFFFF
    bits = torch.where((k & 0x8000) != 0, k ^ 0x8000, k ^ 0xFFFF)
    bits = torch.where((bits & 0x8000) != 0, bits - 0x10000, bits).to(torch.int16)
    return bits.view(torch.bfloat16).float()


def candidates(out_words, out_offset, rows, users, block):
    """One D launch's selector inputs from chip 0's out tensor words (the hub-merged global top-16).

    out_words: int64 tensor of the out tensor's uint32 words (& 0xFFFFFFFF); out_offset: d_model's OUT_WORDS; rows: R;
    users: the launch's live user indices u (rows u * block .. u * block + block - 1); the draft rows are a user's rows
    1 .. block - 1 (row 0 is the anchor). Returns (unary [U, K, 16] fp32, cand [U, K, 16] int64), K = block - 1."""
    w = out_words.reshape(-1)[out_offset:].reshape(rows, 16, 2)
    idx = torch.as_tensor(users, dtype=torch.int64)[:, None] * block + torch.arange(1, block)[None, :]  # [U, K]
    sel = w[idx]  # [U, K, 16, 2]
    return unkey(sel[..., 0]), sel[..., 1]
