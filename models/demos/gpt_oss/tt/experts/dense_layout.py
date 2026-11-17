# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host layout of the batched dense decode MoE weights (experts/weights.py load_dense_decode_expert_weights) and the
core split of its decode programs (experts/decode.py DenseDecodePlan).

Per device (TP shard d of tp), every expert's shard side by side, each expert padded from il = I / tp to il_pad (whole
tiles) intermediate columns with zero weights and zero biases:
  gate / up [H, E * il_pad]: expert e in columns [e * il_pad, (e + 1) * il_pad), its shard's il columns first;
  down [E * il_pad, H]: expert e in rows [e * il_pad, (e + 1) * il_pad), its shard's il rows first;
  gate / up bias [tokens, E * il_pad] (one identical row per token), down bias [E, H] (real on device 0 only).
gpt-oss-20b at TP 4: il = 720 (22.5 tiles), il_pad = 736 (23 tiles), so every expert is whole tiles and the routed
prefill reads each expert's tiles through a page table (stream_pages.py dense_page_table).

A padded column contributes exactly zero: gate = up = 0 there (zero weights, zero bias), and the gpt-oss SwiGLU
(clamp(u, -l, l) + 1) * g * sigmoid(alpha g) is 0 at g = 0; its down rows are zero as well. The real columns of each
16-column bfloat4_b exponent group are the same as in the unpadded layout (every expert starts at a multiple of 16).

Pure torch, no ttnn: host tests (tests/unit/test_dense_layout.py) load this file directly.
"""

import torch

TILE = 32


def padded_expert_width(inter_local):
    """Intermediate columns per expert per device in the dense layout: inter_local rounded up to whole tiles."""
    return -(-inter_local // TILE) * TILE


def dram_shard_tiles(n_tiles, banks, workers_per_bank=1):
    """Shard width (tiles) of a weight [K, n_tiles * 32] DRAM width-sharded over `banks` banks for the DRAM-sharded
    matmul with workers_per_bank readers per bank: that matmul needs shard width == workers_per_bank *
    ceil(n_tiles / (banks * workers_per_bank)) (matmul_multicore_reuse_mcast_dram_sharded_program_factory.cpp). The
    last shard may be partial; every shard keeps this width as its row stride."""
    w = workers_per_bank
    return w * -(-n_tiles // (banks * w))


def _check(tp, il, il_pad):
    assert il_pad % TILE == 0 and il_pad >= il, (il, il_pad)
    assert tp >= 1


def gate_up_columns(w, tp, il_pad):
    """w [E, H, I] (gate or up, checkpoint channel order) -> [1, 1, H, tp * E * il_pad]: TP shard d (columns
    [d * il, (d + 1) * il) of every expert) of every expert side by side, each expert zero padded to il_pad columns,
    shards in device order (the column-parallel mesh mapper splits the last dim into tp parts)."""
    E, H, I = w.shape
    il = I // tp
    _check(tp, il, il_pad)
    out = torch.zeros(H, tp, E, il_pad, dtype=w.dtype)
    out[..., :il] = w.reshape(E, H, tp, il).permute(1, 2, 0, 3)
    return out.reshape(1, 1, H, tp * E * il_pad)


def down_rows(w, tp, il_pad):
    """w [E, I, H] (down) -> [1, 1, tp * E * il_pad, H]: TP shard d (rows [d * il, (d + 1) * il)) of every expert
    stacked, each expert zero padded to il_pad rows, shards in device order (row-parallel mesh mapper)."""
    E, I, H = w.shape
    il = I // tp
    _check(tp, il, il_pad)
    out = torch.zeros(tp, E, il_pad, H, dtype=w.dtype)
    out[:, :, :il] = w.reshape(E, tp, il, H).permute(1, 0, 2, 3)
    return out.reshape(1, 1, tp * E * il_pad, H)


def bias_rows(b, tp, il_pad, tokens):
    """b [E, I] (gate or up bias) -> [1, 1, tokens, tp * E * il_pad]: the gate_up_columns order, zero on the padded
    columns, one identical row per token."""
    E, I = b.shape
    il = I // tp
    _check(tp, il, il_pad)
    out = torch.zeros(tp, E, il_pad, dtype=b.dtype)
    out[..., :il] = b.reshape(E, tp, il).permute(1, 0, 2)
    return out.reshape(1, 1, 1, -1).expand(1, 1, tokens, tp * E * il_pad).contiguous()


def down_bias_rows(b, tp):
    """b [E, H] -> [1, 1, E, tp * H]: device 0's part b, the others zero (row-parallel: the TP sum adds it once)."""
    E, H = b.shape
    db = b.reshape(1, 1, E, H)
    return torch.cat([db] + [torch.zeros_like(db)] * (tp - 1), dim=-1)


def expand_matrix(num_experts, il_pad):
    """[1, 1, E, E * il_pad]: 1 where column c belongs to expert e = c // il_pad (padded columns included; their
    activation is 0)."""
    expand = torch.zeros(1, 1, num_experts, num_experts * il_pad)
    for e in range(num_experts):
        expand[0, 0, e, e * il_pad : (e + 1) * il_pad] = 1
    return expand


def dense_core_split(hidden, il_pad, num_experts, act_cores, hidden_cores, eltwise_cores, grid_cores):
    """Tiles per core of the batched dense decode programs (DenseDecodePlan): the [32, E * il_pad] activation
    width-sharded over act_cores (gate / up matmul outputs, down matmul input), the hidden dim over hidden_cores (the
    gate / up input, the down output) and the activation over eltwise_cores for the unfused elementwise steps. Every
    split must be exact; returns dict(act=..., hidden=..., eltwise=..., act_tiles=..., hidden_tiles=...)."""
    assert hidden % TILE == 0 and il_pad % TILE == 0, (hidden, il_pad)
    act_tiles, hidden_tiles = num_experts * il_pad // TILE, hidden // TILE
    for name, tiles, cores in (
        ("activation", act_tiles, act_cores),
        ("hidden", hidden_tiles, hidden_cores),
        ("eltwise activation", act_tiles, eltwise_cores),
    ):
        assert tiles % cores == 0, f"{name}: {tiles} tiles do not split over {cores} cores"
        assert cores <= grid_cores, f"{name}: {cores} cores > the {grid_cores}-core grid"
    return dict(
        act=act_tiles // act_cores,
        hidden=hidden_tiles // hidden_cores,
        eltwise=act_tiles // eltwise_cores,
        act_tiles=act_tiles,
        hidden_tiles=hidden_tiles,
    )
