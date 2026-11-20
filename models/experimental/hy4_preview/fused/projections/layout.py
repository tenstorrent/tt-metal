# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The split of Hy4-preview's attention projections over the 32 chips of a Galaxy for the decode megakernel, the
weight streams a chip reads, and the goldens of a chip's share (pure torch; ttnn only to round weights as the device
stores them).

The split (chip c of 32; "tile column" = 32 output channels; T = [8 rows, 32] row tile)
---------------------------------------------------------------------------------------
Everything that is fed by the broadcast input x is split on its OUTPUT: every chip has all of x after X1, so a chip
computes whole output columns, nothing is summed across chips and each chip reads 1/32 of the weight.

    q_a            6144 -> 2048    chip c owns columns [64 c, 64 c + 64)                      2 tile columns
    gate           6144 -> 16384   chip c owns the 512 columns of heads 2 c, 2 c + 1          16 tile columns
    "pool"         the small weights, one tile column per chip (POOL): kv_a's 18 columns (16 pass, 2 rope) on
                   chips 0..17 and, on full-indexer layers, the index key's 4 on chips 18..21 and the 32 head
                   weights on chip 22

Everything per head is split by HEAD (chip c owns heads 2 c and 2 c + 1), so that q_b's output, kv_b on both sides,
the gate and o_proj's input all meet on the chip that owns the head:

    q_b            2048 -> 64 x 256   output split: chip c reads the 512 columns of its heads   16 tile columns
    kv_b, key side 192 -> 512 per head (absorption: q_abs = q_nope W_k)                          own 2 heads
    kv_b, value    512 -> 256 per head (attn = ctx W_v^T)                                        own 2 heads
    o_proj         64 x 256 -> 6144   INPUT split: chip c reads the 512 rows of its heads and makes a partial of
                                      all 6144 outputs, summed over the chips

What crosses chips (the payloads are counted in ``payloads``):

    X1   root -> all       x [8, 6144] BF16 and the 8 input-norm scalars
    Xa   all-gather        q_a's 2 tiles + the lane sums of its squares + the pool tile of every chip: afterwards
                           every chip holds q_a [8, 2048] whole (q_b's input; the norm scalar), the 576-wide latent
                           row (for the chip that appends it) and the raw index key / head weights
    X3   all-gather        the absorbed queries of the chip's 2 heads for the 8 rows
    X4   merge             must leave ctx of heads 2 c, 2 c + 1 on chip c
    X5   reduce-to-root    the o_proj partials [8, 6144], a SUM (the MoE's X6 has the same shape)

Norms are deferred. The input norm's gamma is folded into the columns of every x-fed weight and its per-row
1 / RMS multiplies the outputs (``scale``). q_a's norm weight is folded into q_b's rows and its per-row
rsqrt(mean(q_a^2) + eps) multiplies the queries, from the lane sums gathered in Xa.

Frames: the device frame of tt/frames.py (RoPE dims interleaved), so a chip's absorbed queries are what the sparse
attention expects (fused/sparse_mla/op.py "q").

Weight streams. A program's weights are one DRAM tensor, one WIDTH_SHARDED shard per bank; a bank's shard is the
blocks of the units placed on that bank's cores back to back. A unit's block is its [K, ct] tiles K-major (per K
tile the ct output tiles: custom_mm's in1 order), once per term.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

import torch

from models.experimental.hy4_preview.tt import frames

TILE = 32
ROWS = 8
BANKS = 8
CHIPS = 32
HIDDEN = 6144
Q_LORA = 2048
KV_LORA = 512
ROPE = 64
HEADS = 64
NOPE = 192
QK = NOPE + ROPE
V = 256
LATENT = KV_LORA + ROPE
IDX_HEADS = 32
IDX_DIM = 128
HEADS_PER_CHIP = HEADS // CHIPS
ROW_TILE_BYTES = 2 * TILE * ROWS
TILE_BYTES = {"bfp8": 1088, "bfp4": 576, "bf16": 2048}
NORM_EPS = 1e-6

# finishing modes of a unit (kernels/proj_sfpu.hpp)
FIN_NONE, FIN_SCALE, FIN_SIGMOID, FIN_STAT, FIN_NORM, FIN_NORM_ROPE, FIN_GATE, FIN_ROUND = range(8)
FIN_SCALE_ROPE, FIN_CONST = 8, 9


def fin_has_const(fin):
    return FIN_SCALE <= fin <= FIN_NORM_ROPE or fin == FIN_SCALE_ROPE


# meta page: where the hub's parameter words sit (kernels/proj_common.hpp)
PARAM_SCALE_WORD = 128  # uint16 word of scale[0] (fp32, byte 256)
PARAM_LANES_WORD = 144  # uint16 word of lanes[0][0] (fp32, byte 288)
META_IN_WORDS = 768


# ----------------------------------------------------------------------------------------------------------------
# Who owns what


def heads_of(chip):
    return [HEADS_PER_CHIP * chip + j for j in range(HEADS_PER_CHIP)]


def pool(full: bool):
    """The small weights' tile columns in chip order: (name, tile column)."""
    cols = [("kv", t) for t in range(LATENT // TILE)]
    if full:
        cols += [("key", t) for t in range(IDX_DIM // TILE)] + [("head_w", 0)]
    return cols


def pool_of(chip, full: bool):
    p = pool(full)
    return p[chip] if chip < len(p) else None


def tiles_read(chip, full: bool, fused_index_query: bool = True):
    """Weight tiles (32 x 32) chip `chip` reads per layer, by projection, one term."""
    kt = HIDDEN // TILE
    t = {
        "q_a": kt * 2,
        "pool": kt if pool_of(chip, full) else 0,
        "gate": kt * (HEADS_PER_CHIP * V // TILE),
        "q_b": (Q_LORA // TILE) * (HEADS_PER_CHIP * QK // TILE),
        "kv_b key side": HEADS_PER_CHIP * (NOPE // TILE) * (KV_LORA // TILE),
        "kv_b value side": HEADS_PER_CHIP * (KV_LORA // TILE) * (V // TILE),
        "o_proj": (HEADS_PER_CHIP * V // TILE) * kt,
    }
    if full:
        # one index head per chip: from x through the fused weight (rides in Xa) or from q_a (needs a gather of
        # its own before the scan)
        t["index query"] = (kt if fused_index_query else Q_LORA // TILE) * (IDX_DIM // TILE)
    return t


def payloads(full: bool):
    """Bytes of each cross-chip step of the projections: (sent by a chip, landing on a chip)."""
    xa = 2 * ROW_TILE_BYTES + 2 * ROW_TILE_BYTES + ROW_TILE_BYTES  # q_a, its lane sums (FP32 tile), the pool tile
    if full:
        xa += (IDX_DIM // TILE) * ROW_TILE_BYTES
    x3 = HEADS_PER_CHIP * (LATENT // TILE) * ROW_TILE_BYTES
    return {
        "X1 broadcast x": (HIDDEN // TILE * ROW_TILE_BYTES + 4 * ROWS, HIDDEN // TILE * ROW_TILE_BYTES + 4 * ROWS),
        "Xa gather q_a + pool": (xa, CHIPS * xa),
        "X3 gather absorbed queries": (x3, CHIPS * x3),
        "X4 ctx of the own heads (in)": (0, HEADS_PER_CHIP * (KV_LORA // TILE) * ROW_TILE_BYTES),
        "X5 sum of o_proj partials": (HIDDEN // TILE * ROW_TILE_BYTES, 0),
    }


# ----------------------------------------------------------------------------------------------------------------
# Weights: checkpoint -> device frame, norms folded, as [K, N] right operands


@dataclass
class LayerWeights:
    """One layer's projections as right operands ``in @ wt`` ([in, out], fp32), device frame, norm weights folded:
    the input norm's gamma into the rows of every x-fed weight, q_a's norm weight into q_b's rows."""

    q_a: torch.Tensor  # [6144, 2048]
    kv_a: torch.Tensor  # [6144, 576]   (pass 512 | rope 64 interleaved)
    gate: torch.Tensor  # [6144, 16384] head-major
    q_b: torch.Tensor  # [2048, 64 * 256] per head (nope 192 | rope 64 interleaved)
    w_k: torch.Tensor  # [64, 192, 512]
    w_v: torch.Tensor  # [64, 512, 256]
    o: torch.Tensor  # [16384, 6144]
    kv_norm: torch.Tensor  # [512] kv_a's norm weight (applied where the latent row is finished)
    key: torch.Tensor | None = None  # [6144, 128]  (rope 64 interleaved | pass 64), before the LayerNorm
    head_w: torch.Tensor | None = None  # [6144, 32], both scales folded
    idx_q: torch.Tensor | None = None  # [2048, 32 * 128] per head (rope | pass)
    key_norm_w: torch.Tensor | None = None  # [128] the key LayerNorm's weight, in the key's column order
    key_norm_b: torch.Tensor | None = None  # [128] its bias (both applied where the key row is finished)

    @property
    def full(self):
        return self.key is not None

    def fused_index_query(self):
        """[6144, 32 * 128]: x -> the index queries without q_a's per-row norm scalar (a positive factor per row,
        which no top-k sees)."""
        return self.q_a.double().matmul(self.idx_q.double()).float()


def layer_weights(attention: dict, gamma_in=None, indexer: dict | None = None, cfg=None, idx_cfg=None):
    """attention: the nine tensors by checkpoint name ([out, in]); gamma_in: the layer's input norm weight [6144]
    (None: ones); indexer: the five indexer tensors on a full layer."""
    from models.experimental.hy4_preview.reference.attention import AttentionConfig
    from models.experimental.hy4_preview.reference.indexer import IndexerConfig

    cfg = cfg or AttentionConfig()
    w = {k: v.float() for k, v in frames.attention_weights(attention, cfg).items()}
    g = torch.ones(cfg.hidden_size) if gamma_in is None else gamma_in.float()
    gq = w["q_a_layernorm"]
    kv_b = w["kv_b_proj"].reshape(cfg.num_heads, cfg.kv_b_head_dim, cfg.kv_lora_rank)
    out = LayerWeights(
        q_a=(w["q_a_proj"] * g).t().contiguous(),
        kv_a=(w["kv_a_proj_with_mqa"] * g).t().contiguous(),
        gate=(w["linear_gate"] * g).t().contiguous(),
        q_b=(w["q_b_proj"] * gq).t().contiguous(),
        w_k=kv_b[:, : cfg.qk_nope_head_dim].contiguous(),
        w_v=kv_b[:, cfg.qk_nope_head_dim :].transpose(1, 2).contiguous(),
        o=w["o_proj"].t().contiguous(),
        kv_norm=w["kv_a_layernorm"],
    )
    if indexer is not None:
        idx_cfg = idx_cfg or IndexerConfig()
        iw = {k: v.float() for k, v in frames.indexer_weights(indexer, idx_cfg).items()}
        out.key = (iw["wk.weight"] * g).t().contiguous()
        out.head_w = (iw["weights_proj.weight"] * g * idx_cfg.weight_scale).t().contiguous()
        out.idx_q = (iw["wq_b.weight"] * gq).t().contiguous()
        out.key_norm_w, out.key_norm_b = iw["k_norm.weight"].contiguous(), iw["k_norm.bias"].contiguous()
    return out


def chip_weights(lw: LayerWeights, chip: int, fused_index_query: bool = True):
    """Chip `chip`'s share as right operands [K, N]."""
    h0 = HEADS_PER_CHIP * chip
    hs = heads_of(chip)
    out = {
        "q_a": lw.q_a[:, 64 * chip : 64 * chip + 64],
        "gate": lw.gate[:, V * h0 : V * (h0 + HEADS_PER_CHIP)],
        "q_b": [lw.q_b[:, QK * h : QK * (h + 1)] for h in hs],
        "w_k": [lw.w_k[h] for h in hs],
        "w_v": [lw.w_v[h] for h in hs],
        "o": lw.o[V * h0 : V * (h0 + HEADS_PER_CHIP)],
    }
    p = pool_of(chip, lw.full)
    if p is not None:
        src = {"kv": lw.kv_a, "key": lw.key, "head_w": lw.head_w}[p[0]]
        out["pool"] = src[:, TILE * p[1] : TILE * (p[1] + 1)]
    if lw.full:
        cols = slice(IDX_DIM * chip, IDX_DIM * (chip + 1))
        out["idx_q"] = lw.fused_index_query()[:, cols] if fused_index_query else lw.idx_q[:, cols]
    return out


# ----------------------------------------------------------------------------------------------------------------
# Weight formats


def stored(wt, fmt, terms=1):
    """wt [32 a, 32 b] as the device computes with it: BFP8 / BFP4 rounding of its tiles, or the sum of two BFP4
    terms (fused/moe/layout.py two_terms)."""
    import ttnn

    dtype = {"bfp8": ttnn.bfloat8_b, "bfp4": ttnn.bfloat4_b, "bf16": ttnn.bfloat16}[fmt]

    def rnd(m):
        t = ttnn.from_torch(m.reshape(1, 1, *m.shape).float().contiguous(), dtype=dtype, layout=ttnn.TILE_LAYOUT)
        return ttnn.to_torch(t).float().reshape(m.shape)

    q1 = rnd(wt)
    return q1 if terms == 1 else q1 + rnd(wt - q1)


def term_list(wt, fmt, terms=1, pre_rounded=False):
    """The matrices to store, one per term: [wt] or [wt, wt - stored(wt)]. pre_rounded: every term as the value
    its own rounding gives (stored(term)), which the stream tensor then holds unchanged."""
    out = [wt] if terms == 1 else [wt, wt - stored(wt, fmt, 1)]
    return [stored(m, fmt, 1) for m in out] if pre_rounded else out


def stored_stream(wt, fmt, terms=1, pre_rounded=False):
    """wt as a program's weight stream holds it. The stream is a sharded tensor, and ttnn rounds float32 to BF16
    before it packs a block format into one (ttnn/core/tensor/py_to_tt_tensor.cpp compute_host_dtype; read back
    from the board in fused/fidelity/tests/test_stream_reference.py), so every term is BFP(BF16(term)), not
    stored(term): 10 % of the BFP8 and 0.7 % of the BFP4 entries of a float32 weight differ by one step, and the
    second of two BFP4 terms no longer completes the first (3.4 % from the weight instead of 0.9 %; fused/fidelity/
    GUIDE.md section 3). With pre_rounded terms the stream holds stored(wt, fmt, terms)."""
    return sum(stored(m.to(torch.bfloat16).float(), fmt, 1) for m in term_list(wt, fmt, terms, pre_rounded))


def block_tiles(wt, ct):
    """wt [32 K, 32 ct] -> [K * ct, 32, 32] tiles, K-major."""
    k = wt.shape[0] // TILE
    assert wt.shape == (TILE * k, TILE * ct), (tuple(wt.shape), ct)
    return wt.reshape(k, TILE, ct, TILE).permute(0, 2, 1, 3).reshape(k * ct, TILE, TILE)


def bank_streams(blocks_per_bank):
    """blocks_per_bank[b] = list of [n, 32, 32] tile blocks -> ([rows, 8 * 32] stream, per bank the tile offset of
    each block). Banks are padded to the longest."""
    assert len(blocks_per_bank) == BANKS
    offsets, cols = [], []
    for blocks in blocks_per_bank:
        off, n = [], 0
        for blk in blocks:
            off.append(n)
            n += blk.shape[0]
        offsets.append(off)
        cols.append(torch.cat(blocks, dim=0) if blocks else torch.zeros(0, TILE, TILE))
    n_max = max(1, max(c.shape[0] for c in cols))
    cols = [torch.cat((c, torch.zeros(n_max - c.shape[0], TILE, TILE)), dim=0) for c in cols]
    stream = torch.stack(cols, dim=1).reshape(n_max, BANKS, TILE, TILE).permute(0, 2, 1, 3)
    return stream.reshape(n_max * TILE, BANKS * TILE).contiguous(), offsets


# ----------------------------------------------------------------------------------------------------------------
# Row tiles and parameter words


def to_row_tiles(x):
    """[ROWS, n] -> [n / 32, 2 * 16 * ROWS]: tile t = columns [32 t, 32 t + 32) as two faces of [ROWS, 16]."""
    rows, n = x.shape
    assert rows in (ROWS, 2 * ROWS) and n % TILE == 0, tuple(x.shape)  # (16 rows: a program built with rows=16)
    return x.reshape(rows, n // TILE, 2, 16).permute(1, 2, 0, 3).reshape(n // TILE, 2 * 16 * rows).contiguous()


def from_row_tiles(t):
    n_tiles, rows = t.shape[0], t.shape[1] // TILE
    return t.reshape(n_tiles, 2, rows, 16).permute(2, 0, 1, 3).reshape(rows, n_tiles * TILE)


def rope_tile(table):
    """An interleaved cos or sin table [ROWS, 64] (frames.rope_tables) -> [ROWS, 32], laid out for the SFPU: element
    (r, 16 t + 2 l + h) is the value of pair 16 t + 8 h + l, so that the register (tile t of the rope dims, face h)
    finds its 8 pairs on its 8 lanes (kernels/proj_sfpu.hpp)."""
    rows = table.shape[0]
    assert rows in (ROWS, 2 * ROWS) and table.shape[1] == ROPE, tuple(table.shape)
    per_pair = table[:, 0::2]  # [ROWS, 32], both dims of a pair carry the same value
    return per_pair.reshape(rows, 2, 2, 8).permute(0, 1, 3, 2).reshape(rows, TILE).contiguous()


def lane_sums(v):
    """[ROWS, n] -> [ROWS, 8]: lane l = sum of v^2 over the columns with (column % 16) // 2 == l, the partial sums
    a unit leaves in its statistic tile (their sum over the lanes is the row's sum of squares)."""
    return v.double().pow(2).reshape(v.shape[0], -1, 8, 2).sum(dim=(1, 3)).float()


def stat_tile_lanes(tile):
    """A unit's FP32 statistic tile [ROWS, 32] (from_row_tiles) -> its lane sums [ROWS, 8] (the even columns of the
    first face)."""
    return tile[:, 0:16:2]


def f32_words(values):
    """fp32 values -> their uint16 words, little endian."""
    out = []
    for v in values:
        b = struct.unpack("<I", struct.pack("<f", float(v)))[0]
        out += [b & 0xFFFF, b >> 16]
    return out


def meta_words(scale=None, lanes=None, n_plain=0, rows=None):
    """The hub's parameter page (uint16 [1, 768]): [U = 1, id 0 | ... | scale[ROWS] fp32 at byte 256 | lanes
    [ROWS][8] fp32 at byte 288]. scale: the per-row multiplier of the finishing modes (default 1); lanes: the lane
    sums of q_a's squares, summed over the chips (FIN_NORM / FIN_NORM_ROPE)."""
    words = torch.zeros(META_IN_WORDS, dtype=torch.int32)
    words[0] = 1
    words[126] = n_plain
    rows = ROWS if rows is None else rows  # (16: the lane sums follow the 16 scales, kernels/proj_common.hpp)
    lanes_word = PARAM_SCALE_WORD + 2 * rows
    assert rows != ROWS or lanes_word == PARAM_LANES_WORD
    scale = torch.ones(rows) if scale is None else scale
    assert scale.numel() == rows and (lanes is None or lanes.shape == (rows, 8)), (rows, tuple(scale.shape))
    words[PARAM_SCALE_WORD : PARAM_SCALE_WORD + 2 * rows] = torch.tensor(f32_words(scale.tolist()))
    if lanes is not None:
        words[lanes_word : lanes_word + 2 * rows * 8] = torch.tensor(f32_words(lanes.flatten().tolist()))
    return words.reshape(1, META_IN_WORDS)


def round7(x):
    """x rounded to 7 significant bits (to nearest, ties away from zero in magnitude): what a LoFi custom_mm keeps
    of its left operand without loss."""
    b = x.float().contiguous().view(torch.int32)
    return ((b + 0x10000) & ~0x1FFFF).view(torch.float32)


def trunc_bits(x, bits):
    """x truncated toward zero to `bits` significant bits."""
    b = x.float().contiguous().view(torch.int32)
    return (b & ~((1 << (24 - bits)) - 1)).view(torch.float32)


# ----------------------------------------------------------------------------------------------------------------
# Goldens of a chip's share (fp32 references; `w` is chip_weights(...), stored() applied by the caller if wanted)


def hadamard_sign(dim=IDX_DIM):
    """The +-1 pattern of frames.hadamard and its (BF16-rounded) magnitude."""
    h = frames.hadamard(dim)
    return torch.sign(h), float(h[0, 0])


def pre_golden(y, scale, w, cos=None, sin=None):
    """y [8, 6144] (the collapsed input, not normalised), scale [8] (1 / RMS): the chip's q_a columns [8, 64], their
    lane sums [8, 8], its pool tile [8, 32] or None, its gate [8, 512] (sigmoid), and on a full layer its index
    head's query [8, 128]: "idx_q" before RoPE, with cos / sin (the interleaved tables, [8, 64]) also "idx_q_dev",
    the query in the cache's frame (RoPE on the leading 64 dims, Hadamard basis) without q_a's norm scalar."""
    s = scale.double()[:, None]
    yd = y.double()
    q_a = (yd @ w["q_a"].double()) * s
    out = {
        "q_a": q_a.float(),
        "lanes": lane_sums(q_a),
        "gate": torch.sigmoid((yd @ w["gate"].double()) * s).float(),
    }
    if "pool" in w:
        out["pool"] = ((yd @ w["pool"].double()) * s).float()
    if "idx_q" in w and w["idx_q"].shape[0] == HIDDEN:
        q = (yd @ w["idx_q"].double()) * s
        out["idx_q"] = q.float()
        if cos is not None:
            roped = torch.cat((interleaved_rope(q[:, :ROPE], cos.double(), sin.double()), q[:, ROPE:]), dim=1)
            out["idx_q_dev"] = (roped @ frames.hadamard(IDX_DIM).double()).float()
    return out


def norm_scalar(lanes, n=Q_LORA, eps=NORM_EPS):
    """Lane sums [8, 8] of a row's squares, summed over the chips -> rsqrt(mean + eps) [8]."""
    return torch.rsqrt(lanes.double().sum(dim=1) / n + eps).float()


def interleaved_rope(v, cos, sin):
    """RoPE on interleaved pairs: v [..., 64], cos / sin the interleaved tables (tt/frames.py rope_tables)."""
    return v * cos + (v @ frames.rope_swap(v.shape[-1]).to(v.dtype)) * sin


def query_golden(q_a, lanes, cos, sin, w):
    """q_a [8, 2048] whole (not normalised), lanes [8, 8] its lane sums, cos / sin [8, 64] interleaved tables: the
    absorbed queries of the chip's heads, [heads, 8, 576] (q_nope W_k | q_rope), device frame, not scaled by the
    softmax scale; and the intermediate q [heads, 8, 256]."""
    s = norm_scalar(lanes).double()[:, None]
    q_lat, qs = [], []
    for wq, wk in zip(w["q_b"], w["w_k"]):
        q = (q_a.double() @ wq.double()) * s
        rope = interleaved_rope(q[:, NOPE:], cos.double(), sin.double())
        qs.append(torch.cat((q[:, :NOPE], rope), dim=1).float())
        q_lat.append(torch.cat((q[:, :NOPE] @ wk.double(), rope), dim=1).float())
    return torch.stack(q_lat), torch.stack(qs)


def post_golden(ctx, gate, w):
    """ctx [heads, 8, 512] (the merged, normalised attention output of the chip's heads in latent space), gate
    [8, 512] (sigmoid, the chip's heads): the chip's partial of the sublayer output [8, 6144] and the gated head
    outputs [8, 512]."""
    attn = torch.cat([c.double() @ wv.double() for c, wv in zip(ctx, w["w_v"])], dim=1)
    gated = attn * gate.double()
    return (gated @ w["o"].double()).float(), gated.float()


def sparse_mla_pages(q_lat_by_chip):
    """The 32 chips' absorbed queries (chip c: [2, 8, 576]) -> the pages the sparse attention reads
    (fused/sparse_mla/op.py "q"): [users 8, groups 8, 8 heads, 576], page (u, g) = heads 8 g .. 8 g + 7 of user u."""
    q = torch.cat(list(q_lat_by_chip), dim=0)  # [64 heads, 8 users, 576]
    return q.permute(1, 0, 2).reshape(ROWS, HEADS // 8, 8, LATENT).contiguous()


def reference_row(attention_ref, x_norm, cos_hf, sin_hf):
    """reference/attention.py on 8 independent rows: (q_resid [8, 2048], k_pass, k_rope (HF frame), q [64, 8, 256]
    (HF frame), gate [8, 16384])."""
    with torch.no_grad():
        q_resid, k_pass, k_rope = attention_ref.latent(x_norm, cos_hf, sin_hf)
        q = attention_ref.queries(q_resid, cos_hf, sin_hf)
        gate = attention_ref.gate(x_norm)
    return q_resid, k_pass, k_rope, q, gate
