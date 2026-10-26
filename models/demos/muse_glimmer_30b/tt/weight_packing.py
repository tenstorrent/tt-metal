# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Pure-PyTorch tensor-parallel weight packing for Muse-Glimmer (no ttnn import).

All packed tensors are "in x out" (already transposed from HF Linear [out, in]) and laid out so that
ShardTensorToMesh(dim=-1) (or dim=-2 for row-parallel weights) hands rank i its block.
"""

import torch


def kv_head_for_rank(rank: int, tp: int, n_q_heads: int = 32, n_kv_heads: int = 2) -> int:
    """KV head owned by `rank` when KV heads are replicated (n_kv_heads < tp)."""
    return (rank * (n_q_heads // tp)) * n_kv_heads // n_q_heads


def pack_qkvg(q_w, k_w, v_w, g_w, tp: int, head_dim: int = 128) -> torch.Tensor:
    """Pack q/k/v/gate HF weights [out, in] into [H, tp * W_local], rank-major [Q_i | K_i | V_i | G_i] blocks.

    W_local = (2*ql + 2*kvl) * head_dim. If n_kv < tp, K/V are replicated (rank i gets kv head
    kv_head_for_rank(i)); otherwise rank i gets the contiguous chunk of n_kv // tp heads.
    """
    n_q = q_w.shape[0] // head_dim
    n_kv = k_w.shape[0] // head_dim
    assert n_q % tp == 0 and g_w.shape == q_w.shape and v_w.shape == k_w.shape
    ql = n_q // tp
    blocks = []
    for i in range(tp):
        rows = slice(i * ql * head_dim, (i + 1) * ql * head_dim)
        if n_kv < tp:
            kh = kv_head_for_rank(i, tp, n_q, n_kv)
            kv_rows = slice(kh * head_dim, (kh + 1) * head_dim)
        else:
            kvl = n_kv // tp
            kv_rows = slice(i * kvl * head_dim, (i + 1) * kvl * head_dim)
        blocks.append(torch.cat([q_w[rows].T, k_w[kv_rows].T, v_w[kv_rows].T, g_w[rows].T], dim=-1))
    return torch.cat(blocks, dim=-1).contiguous()


def pack_o(o_w, tp: int) -> torch.Tensor:
    """o_w [H, nq*hd] -> o_w.T [nq*hd, H]. Row-parallel: shard on dim -2, rank i gets rows
    [i*ql*hd:(i+1)*ql*hd] (matching the head split of pack_qkvg). Partial outputs need all_reduce."""
    assert o_w.shape[1] % tp == 0
    return o_w.T.contiguous()


def _pad_cols(t, to):
    if to is None or to == t.shape[-1]:
        return t
    assert to > t.shape[-1]
    return torch.nn.functional.pad(t, (0, to - t.shape[-1]))


def pack_gate_up(gate_w, up_w, tp: int, pad_to=None) -> torch.Tensor:
    """gate/up [I, H] -> [H, tp * 2*pad_to]; rank block = [gate_i.T | up_i.T], each zero-padded to pad_to
    (default I_local, i.e. no padding). Zero padding is exact: silu(0) * 0 = 0."""
    inter = gate_w.shape[0]
    assert inter % tp == 0 and up_w.shape == gate_w.shape
    il = inter // tp
    blocks = [
        torch.cat(
            [_pad_cols(gate_w[i * il : (i + 1) * il].T, pad_to), _pad_cols(up_w[i * il : (i + 1) * il].T, pad_to)],
            dim=-1,
        )
        for i in range(tp)
    ]
    return torch.cat(blocks, dim=-1).contiguous()


def pack_column_parallel(w, tp: int, pad_to=None) -> torch.Tensor:
    """w [N, H] (HF Linear) -> [H, tp * pad_to]; rank block i = w[i*N/tp:(i+1)*N/tp].T zero-padded to pad_to columns."""
    n = w.shape[0]
    assert n % tp == 0
    nl = n // tp
    return torch.cat([_pad_cols(w[i * nl : (i + 1) * nl].T, pad_to) for i in range(tp)], dim=-1).contiguous()


def pack_down(down_w, tp: int, pad_to=None) -> torch.Tensor:
    """down_w [H, I] -> [tp * pad_to, I-padded rows]. Row-parallel: rank i gets rows
    [i*pad_to:(i+1)*pad_to] = its I_local rows followed by zero rows (default pad_to = I_local, no padding)."""
    assert down_w.shape[1] % tp == 0
    t = down_w.T
    if pad_to is None:
        return t.contiguous()
    il = t.shape[0] // tp
    assert pad_to >= il
    blocks = []
    for i in range(tp):
        blk = t[i * il : (i + 1) * il]
        blocks.append(torch.nn.functional.pad(blk, (0, 0, 0, pad_to - il)))
    return torch.cat(blocks, dim=0).contiguous()


def pack_lm_head(lm_w, padded_vocab: int) -> torch.Tensor:
    """lm_w [V, H] -> [H, padded_vocab], zero-padded columns."""
    vocab, hidden = lm_w.shape
    assert padded_vocab >= vocab
    out = torch.zeros(hidden, padded_vocab, dtype=lm_w.dtype)
    out[:, :vocab] = lm_w.T
    return out


def unit_offset(w) -> torch.Tensor:
    """Centered-norm weight -> plain-norm weight: (1 + w) computed in fp32, cast to bf16."""
    return (1 + w.float()).to(torch.bfloat16)
