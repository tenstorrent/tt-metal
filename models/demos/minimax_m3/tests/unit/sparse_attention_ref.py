# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import torch
from ttnn.operations.transformer_golden import BLK_KV


def sparse_attention_ref_msa_sampled_tokens(
    q, k, v, indices, scale, sample_tokens, *, blk_kv=BLK_KV, causal=False, chunk_start_idx=0
):
    """Memory-light MSA reference for selected query tokens. `causal=True` adds the token-level causal
    mask (a selected key may be a future token only inside the diagonal block; mask those out)."""
    B, H, _, d = q.shape
    n_kv, v_dim = k.shape[1], v.shape[-1]
    assert B == 1 and H % n_kv == 0
    G = H // n_kv

    qf, kf, vf = q.float(), k.float(), v.float()
    outs = []
    for s in sample_tokens:
        out = torch.zeros(H, v_dim, dtype=torch.float32)
        p = s + chunk_start_idx
        for g in range(n_kv):
            row = indices[0, g, s]
            valid_blocks = row[row >= 0].long()
            if valid_blocks.numel() == 0:
                continue
            h0, h1 = g * G, (g + 1) * G
            k_sel = torch.cat([kf[0, g, int(blk) * blk_kv : (int(blk) + 1) * blk_kv] for blk in valid_blocks], dim=0)
            v_sel = torch.cat([vf[0, g, int(blk) * blk_kv : (int(blk) + 1) * blk_kv] for blk in valid_blocks], dim=0)
            scores = torch.matmul(qf[0, h0:h1, s] * scale, k_sel.transpose(0, 1))
            if causal:
                key_ids = torch.cat([torch.arange(int(blk) * blk_kv, (int(blk) + 1) * blk_kv) for blk in valid_blocks])
                scores = scores.masked_fill(key_ids.view(1, -1) > p, float("-inf"))
            attn = scores.softmax(dim=-1, dtype=torch.float32)
            out[h0:h1] = torch.matmul(attn, v_sel)
        outs.append(out)
    return torch.stack(outs, dim=1).unsqueeze(0)  # [1,H,len(sample_tokens),v_dim]
