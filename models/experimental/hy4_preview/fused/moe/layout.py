# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""DRAM layout of a chip's share of the routed experts for the streamed decode MoE (fused/moe/program.py), the host
side of the row-tile buffers, and the golden of the chip's partial output.

What a chip holds of an expert is a ``Share``: ``P`` pairs of gate / up tile columns (32 intermediate channels each)
and whether every expert or only some. Tensor parallel over N chips (TP N) is P = 64 / N for all experts; expert
parallel (EP) is P = 64 for the chip's own experts.

Both weight streams are stored per DRAM bank (one WIDTH_SHARDED shard of BFP4 tiles per bank) and expert-major, so a
picked expert is a byte offset ``id * stride`` in every bank and every bank holds the same number of bytes of every
expert: the 8 banks are loaded equally whatever the union of picks is.

gate | up, [6144, 2 * 32 P] per expert. A unit is one pair p (its gate tile column and its up tile column: 192 K
tiles each). A pair is cut into S K ranges ("pieces", S * P a multiple of 8) so that an expert has a whole number
``pb`` of pieces per bank: piece g = p * S + q lives in bank g % 8 as item t = g // 8 of the expert. Inside a piece the
tiles are K-major, [gate tile, up tile] per K tile, which is custom_mm's in1 order for two output tiles.
    TP 32: P 2, S 4, one piece (48 K tiles, 55 KB) per bank       TP 8: P 8, S 1, one pair (221 KB) per bank
    EP:    P 64, S 1, 8 pairs per bank
down, [32 P, 6144] per expert: bank b holds the output tile columns [24 b, 24 b + 24), each as its P K tiles
(column-major, custom_mm's read_transposed order), so a down core owning ``cols`` adjacent columns reads one
contiguous block per expert.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

import ttnn

TILE = 32
BANKS = 8
BFP4_TILE_BYTES = 576
HIDDEN = 6144
INTERMEDIATE = 2048
K_TILES = HIDDEN // TILE  # 192
COLS_PER_BANK = K_TILES // BANKS  # 24 output tile columns of the down projection per bank
ROWS = 8
MAX_SLOTS = 80  # union slots of one program run: 64 routed (8 rows x 8 picks) and up to 16 plain ones
MAX_UNITS = 640  # slots x P
META_IN_WORDS = 128 + ROWS * MAX_SLOTS  # uint16 words of the routing result [U, ids | w]


@dataclass(frozen=True)
class Share:
    """A chip's share of every expert it holds: P pairs of gate / up tile columns, each cut into S K ranges."""

    P: int
    S: int = 1
    name: str = ""

    def __post_init__(self):
        assert (self.P * self.S) % BANKS == 0 and BANKS % self.S == 0, (self.P, self.S)

    @property
    def pb(self):
        """gate | up pieces of an expert per bank."""
        return self.P * self.S // BANKS

    @property
    def piece_k(self):
        return K_TILES // self.S

    @property
    def piece_bytes(self):
        return self.piece_k * 2 * BFP4_TILE_BYTES

    @property
    def gate_up_stride(self):
        """Bytes of one expert's gate | up in a bank."""
        return self.pb * self.piece_bytes

    @property
    def down_stride(self):
        """Bytes of one expert's down in a bank."""
        return COLS_PER_BANK * self.P * BFP4_TILE_BYTES

    @property
    def expert_bytes(self):
        """Bytes of one expert on the chip (both streams, all banks)."""
        return BANKS * (self.gate_up_stride + self.down_stride)

    @property
    def channels(self):
        return TILE * self.P


def tp_share(n: int) -> Share:
    """1 / n of every expert, cut on the intermediate channels."""
    P = INTERMEDIATE // TILE // n
    assert P * n * TILE == INTERMEDIATE and P >= 1
    return Share(P=P, S=max(1, BANKS // P), name=f"tp{n}")


def ep_share() -> Share:
    """Whole experts."""
    return Share(P=INTERMEDIATE // TILE, S=1, name="ep")


# ----------------------------------------------------------------------------------------------------------------
# Weight streams


def gate_up_stream(gate, up, share: Share):
    """gate, up: [E, 32 P, 6144] (the chip's intermediate channels of gate_up_proj's two halves) -> [rows, 8 * 32],
    bank b's shard a column of tiles ordered expert -> item t -> K tile -> (gate, up)."""
    E, I, H = gate.shape
    P, S, pb, kp = share.P, share.S, share.pb, share.piece_k
    assert I == TILE * P and H == HIDDEN and up.shape == gate.shape

    def tiles(w):  # [E, I, H] -> [E, K, P, 32 (k), 32 (n)]: tile (k, p) of w.T
        return w.reshape(E, P, TILE, K_TILES, TILE).permute(0, 3, 1, 4, 2)

    t = torch.stack((tiles(gate), tiles(up)), dim=3)  # [E, K, P, 2, 32, 32]
    t = t.reshape(E, S, kp, P, 2, TILE, TILE).permute(0, 3, 1, 2, 4, 5, 6)  # [E, P, S, kp, 2, 32, 32]
    t = t.reshape(E, pb, BANKS, kp, 2, TILE, TILE).permute(2, 0, 1, 3, 4, 5, 6)  # [bank, E, t, kp, 2, 32, 32]
    return t.reshape(BANKS, -1, TILE).permute(1, 0, 2).reshape(-1, BANKS * TILE)


def down_stream(down, share: Share):
    """down: [E, 6144, 32 P] (the chip's intermediate channels of down_proj) -> [rows, 8 * 32], bank b's shard a
    column of tiles ordered expert -> output tile column (24 per bank) -> K tile."""
    E, H, I = down.shape
    P = share.P
    assert I == TILE * P and H == HIDDEN
    t = down.reshape(E, K_TILES, TILE, P, TILE).permute(0, 1, 3, 4, 2)  # [E, col, k, 32 (k), 32 (n)]: tiles of down.T
    t = t.reshape(E, BANKS, COLS_PER_BANK, P, TILE, TILE).permute(1, 0, 2, 3, 4, 5)
    return t.reshape(BANKS, -1, TILE).permute(1, 0, 2).reshape(-1, BANKS * TILE)


def two_terms(w):
    """w [E, a, b] (tiles on the last two dims of w.transpose(1, 2)) -> [2 E, a, b]: entry 2 e is w[e] (stored as
    q1 = BFP4(w)), entry 2 e + 1 the residual w - q1 (stored as q2 = BFP4(w - q1)). q1 + q2 has the accuracy of BFP8
    and each term is exact for a LoFi multiply."""
    q1 = bfp4_round(w.transpose(1, 2)).transpose(1, 2)
    return torch.stack((w, w - q1), dim=1).reshape(2 * w.shape[0], *w.shape[1:])


def two_terms_value(w):
    """What the device computes with for two_terms(w): q1 + q2, [E, a, b]."""
    q1 = bfp4_round(w.transpose(1, 2)).transpose(1, 2)
    return q1 + bfp4_round((w - q1).transpose(1, 2)).transpose(1, 2)


def stream_memory_config(rows: int):
    """One shard per DRAM bank, [rows, 32] each."""
    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.DRAM,
        ttnn.ShardSpec(
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(BANKS - 1, 0))}),
            (rows, TILE),
            ttnn.ShardOrientation.ROW_MAJOR,
        ),
    )


def to_device_stream(device, stream, cache_file_name=None, host_round=False):
    """A [rows, 256] stream (or None to load ``cache_file_name``) -> the BFP4 DRAM tensor, one shard per bank.

    Created directly with the sharded memory config (the default, and the cache), ttnn rounds float32 to BF16
    before it packs BFP4 (ttnn/core/tensor/py_to_tt_tensor.cpp compute_host_dtype), so the tensor holds
    BFP4(BF16(w)), not bfp4_round(w): the two differ in under 1 % of the entries of a random float32 weight, by one
    BFP4 step, which is 3 % rms (fused/fidelity/GUIDE.md section 3). host_round=True packs on the host first (one
    rounding, what bfp4_round and fused/moe/prepare.py's files hold) and moves the result to the device."""
    rows = stream.shape[0] if stream is not None else None
    if host_round:
        assert cache_file_name is None
        host = ttnn.from_torch(stream[None, None], dtype=ttnn.bfloat4_b, layout=ttnn.TILE_LAYOUT)
        return ttnn.to_device(host, device, memory_config=stream_memory_config(rows))
    if cache_file_name is None:
        return ttnn.from_torch(
            stream[None, None],
            dtype=ttnn.bfloat4_b,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=stream_memory_config(rows),
        )
    return ttnn.as_tensor(
        stream[None, None] if stream is not None else None,
        dtype=ttnn.bfloat4_b,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=stream_memory_config(rows) if rows is not None else None,
        cache_file_name=cache_file_name,
    )


def bfp4_round(w):
    """w [..., 32 a, 32 b] rounded as the device stores it in BFP4 tiles (the tiles of the last two dims)."""
    lead = w.shape[:-2]
    t = ttnn.from_torch(
        w.reshape(1, -1, *w.shape[-2:]).float().contiguous(), dtype=ttnn.bfloat4_b, layout=ttnn.TILE_LAYOUT
    )
    return ttnn.to_torch(t).float().reshape(*lead, *w.shape[-2:])


# ----------------------------------------------------------------------------------------------------------------
# Row tiles: [ROWS, n] BF16 as n / 32 tiles of [ROWS, 32], each face 0 (columns 0..15) then face 1, rows in order


def to_row_tiles(x):
    """[ROWS, n] -> [n / 32, 2 * 16 * ROWS]."""
    rows, n = x.shape
    return x.reshape(rows, n // TILE, 2, 16).permute(1, 2, 0, 3).reshape(n // TILE, 2 * 16 * rows).contiguous()


def from_row_tiles(t, rows=ROWS):
    """[n / 32, 2 * 16 * ROWS] -> [ROWS, n]."""
    n_tiles = t.shape[0]
    return t.reshape(n_tiles, 2, rows, 16).permute(2, 0, 1, 3).reshape(rows, n_tiles * TILE)


def meta_words(ids, weights, n_plain=0):
    """The routing result as the hub takes it: uint16 [1, 768] = [U, ids (local expert ids), 0.., n_plain at word
    126 | BF16 bits of w[ROWS][80]], weights [ROWS, U]. The last n_plain slots are plain SwiGLU (no clamp)."""
    U = len(ids)
    assert U <= MAX_SLOTS and tuple(weights.shape) == (ROWS, U), (U, weights.shape)
    head = torch.zeros(128, dtype=torch.int32)
    head[0] = U
    head[1 : 1 + U] = torch.as_tensor(ids, dtype=torch.int32)
    head[126] = n_plain
    w = torch.zeros(ROWS, MAX_SLOTS)
    w[:, :U] = weights
    bits = w.to(torch.bfloat16).view(torch.int16).to(torch.int32) & 0xFFFF
    return torch.cat((head, bits.flatten())).reshape(1, META_IN_WORDS)


# ----------------------------------------------------------------------------------------------------------------
# Golden


def swiglu(gate, up, limit=10.0):
    """The routed expert's activation (reference/moe.py Hy4MoE.swiglu)."""
    return torch.nn.functional.silu(gate.clamp(max=limit)) * up.clamp(min=-limit, max=limit)


def partial_golden(x, ids, weights, gate, up, down, limit=10.0, n_plain=0):
    """The chip's partial output [ROWS, 6144]: sum over the union slots i of
    w[:, i] * swiglu(x @ gate[ids[i]].T, x @ up[ids[i]].T) @ down[ids[i]].T, with the chip's share of the weights
    (gate, up: [E, I_c, 6144]; down: [E, 6144, I_c]). Summed over the chips of a tensor-parallel split, or over the
    chips' experts, this is reference/moe.py Hy4MoE.routed. The last n_plain slots are plain SwiGLU (the shared
    expert, the dense FFN: reference/moe.py Hy4FFN)."""
    out = torch.zeros(x.shape[0], down.shape[1], dtype=torch.float64)
    x = x.double()
    for i, e in enumerate(ids):
        lim = limit if i < len(ids) - n_plain else float("inf")
        act = swiglu(x @ gate[e].double().T, x @ up[e].double().T, lim) * weights[:, i : i + 1].double()
        out += act @ down[e].double().T
    return out.float()


def route_golden(x, router_weight, router_bias, top_k=8, scale=2.827):
    """reference/moe.py Hy4MoE.route on its own: (scores [T, E], indices [T, top_k] best first, weights [T, top_k])."""
    scores = torch.nn.functional.linear(x.float(), router_weight.float()).sigmoid()
    indices = torch.topk(scores + router_bias.float(), k=top_k, dim=-1, sorted=True)[1]
    weights = scores.gather(1, indices)
    weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)
    return scores, indices, weights * scale


def union_of(indices, weights, expert_base=0, expert_count=None):
    """The union list (ascending ids, numbered from expert_base) and [T, U] weights of per-row picks, as the hub
    builds them, restricted to the experts [expert_base, expert_base + expert_count)."""
    T, K = indices.shape
    hi = float("inf") if expert_count is None else expert_base + expert_count
    ids = sorted({int(e) for e in indices.flatten().tolist() if expert_base <= e < hi})
    slot = {e: i for i, e in enumerate(ids)}
    w = torch.zeros(T, len(ids))
    for r in range(T):
        for j in range(K):
            e = int(indices[r, j])
            if e in slot:
                w[r, slot[e]] = weights[r, j]
    return [e - expert_base for e in ids], w
