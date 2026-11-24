# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Hy4-preview prefill routed MoE on the shared expert layout ``tp8g4`` (#65884, phase 1): design and host reference.

Layout (decode's ``split="tp8g4"``, PR #66982 ``fused/moe/prepare.py``; the copy of the expert weights both engines
are to share). Physical 8x4 Galaxy, chip ``c = 4 * row + col``. A mesh column ``g`` is a GROUP that owns the routed
experts ``[64 g, 64 g + 64)``; each of its 8 chips (mesh row ``s``) holds the SLICE ``[256 s, 256 s + 256)`` of the
2048 intermediate channels of every one of those 64 experts: gate and up ``[64, 256, 6144]``, down ``[64, 6144, 256]``
(``prepare.chip_group`` / ``chip_weights``: group = column, slice = row). 170 MB of BFP4 per chip per layer, the same
bytes as prefill's EP32 today (8 whole experts per chip, the same column owns the same 64 experts).

Prefill dataflow at chunk S = 4096, per MoE layer (residual per chip ``[S/8, 6144/4]``: tokens over axis 0, hidden
over axis 1). Each token picks 8 experts, which fall in 1 to 4 groups.

  0. router (unchanged): ``scores, top_i, top_w`` per token. From them the host or a small device op builds, per
     group g, the PLAN (``plan_group``): the pairs (token, expert of g) in expert-major order, each expert's rows
     padded to a tile (32); ``counts[64]``, ``offsets[64]``, a gather index ``rows[R_g]`` (token id, or a zero row
     for padding), the pair weights ``w[R_g]`` and, for the combine, ``slot_rows[K_g, S]`` (the buffer row of a
     token's k-th pick in g, or none; ``K_g`` = most picks one token puts in g, 8 at worst).
  1. input: every chip of group g needs the FULL 6144 hidden of every token routed to g (any of the 4096). The
     residual arrives as ``[S/8, 1536]``; the shared expert's 2D-TP MLP already all-gathers it over axis 0 to
     ``[S, 1536]`` (its step 1); one more all-gather over axis 1 gives ``[S, 6144]`` on every chip. (Compacted:
     only the ``T_g`` tokens with a pick in g, about 90 % of the chunk: little to gain.)
  2. local gather: ``X_g = x[rows]`` ``[R_g, 6144]``, expert-major (no fabric; the same on all 8 chips of g).
  3. experts, per chip (g, s): for each of its 64 experts e with ``M_e`` rows: ``gu = X_e @ [Wg_e,s | Wu_e,s]``
     ``[M_e, 512]``, ``act = silu(min(g, 10)) * clamp(u, -10, 10)`` ``[M_e, 256]`` (the clamp is elementwise in the
     intermediate channel, so slicing it is exact), ``y = (w * act) @ Wd_e,s`` ``[M_e, 6144]``, a partial sum over
     the group's 8 slices. Every expert share is read once per chunk.
  4. local combine: ``P_g,s[t] = sum_k y[slot_rows[k, t]]`` ``[S, 6144]``, token-major (no fabric).
  5. reduce: the 32 partials sum to the routed output. Reduce-scatter over axis 1 on the hidden dim (sums the 4
     groups) -> ``[S, 1536]`` partial over axis 0; ADD the shared expert's own partial here (its step 5 output,
     ``[S, 1536]`` partial over axis 0, before its reduce-scatter); then ONE reduce-scatter over axis 0 on the tokens
     -> ``[S/8, 1536]`` = routed + shared, the residual's layout. The shared expert's reduce-scatter and gather are
     the routed path's own: the shared expert costs no collective of its own.

Bytes per chip per chunk (computed by ``hop_bytes`` from the 4K LongBench golden's fp32 routing, layers 1 / 2 / 4;
bf16 activations; bytes a chip receives over the fabric, = bytes it sends for these collectives; EP32 dispatch /
combine are the busiest chip's remote pairs x 12 KB):

  hop                                      tp8g4                          EP32 today
  AG axis 1 [S/8,1536] -> [S/8,6144]       -                              4.7 MB
  AG axis 0 [S/8,1536] -> [S,1536]         11.0 MB (the shared expert's)  11.0 MB (shared expert's own)
  AG axis 1 [S,1536]   -> [S,6144]         37.7 MB                        -
  dispatch (pairs to the expert's chip)    -                              16.1 / 17.0 / 23.5 MB (mean 11.0)
  combine (pairs back)                     -                              16.1 / 17.0 / 23.5 MB (mean 11.0)
  RS axis 1 (hidden)                       37.7 MB ([S,6144] partial)     4.7 MB ([S/8,6144])
  RS axis 0 (tokens)                       11.0 MB (+ shared partial)     11.0 MB (shared expert's own)
  total, busiest chip                      97.5 MB                        63.7 / 65.5 / 78.5 MB

tp8g4's are routing-independent; EP32's grow with hot experts (busiest chip 1509 / 1609 / 2137 pairs, mean 1024).
Local DRAM traffic is larger in tp8g4: the gather writes ``R_g`` rows (8.4 K to 10.1 K x 12 KB) and the expert
output is ``[R_g, 6144]`` on every chip of the group (8x EP32's per chip, the price of slicing the intermediate:
each slice produces a full-width partial). Weight reads are the same (170 MB/chip). Load balance: within a group the
8 chips do identical work; across groups the busiest has 1.05-1.11x the mean pairs (EP32's busiest chip: 1.5-2.1x).
Worst case, every token puts all 8 picks in one group: ``R_g`` = 32 K rows on that group's 8 chips (4x the mean),
``K_g`` = 8 (measured K_g 6-8 on the golden: a few tokens put 7-8 picks in one group).

Shared expert: decode's files hold it only on group 3 (``prepare.GROUP_SHARED``, entries 64 / 65, two BFP4 terms,
zeros elsewhere). For a 4096-token chunk that puts ALL shared-expert compute on column 3 (4096 x 6144 x 256 x 3 MACs
per chip, ~50 % of a group's routed MACs) with the other 24 chips idle for it. Prefill keeps its own 2D-TP
``TtMlp2D`` (BFP8, 1/32 per chip, 1.25 MB/chip/layer) and adds its pre-RS partial before the axis-0 reduce-scatter
above; it does not read entries 64 / 65.

What a prefill-rate op over this layout must do that decode's does not: M rows per expert (``M_e`` ~ 128 mean,
thousands at worst) with each expert share streamed once per chunk; decode's program has 8 rows (``ROWS = 8``) and
re-reads the shares on every pass.

This module is the host reference: the split computed per chip exactly as above (``chip_partial``), its sum
(``tp8g4_routed``), the per-group plans, the per-hop bytes, and BFP4 weights as the device stores them
(``bfp4_slices``, ttnn's own BFP4 round trip on the host; ``fused/moe/layout.bfp4_round``). Import-light apart from
``bfp4_slices`` (ttnn, host only: run it with ``TT_METAL_MOCK_CLUSTER_DESC_PATH`` set).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

GROUPS, SLICES = 4, 8  # mesh columns (groups) x mesh rows (slices)
E, TOPK, HIDDEN, INTER = 256, 8, 6144, 2048
EPG = E // GROUPS  # 64 experts per group
I_SLICE = INTER // SLICES  # 256 intermediate channels per chip
TILE = 32
LIMIT = 10.0


def group_slice(chip: int) -> tuple[int, int]:
    """(group, slice) of chip ``c = 4 row + col``: its mesh column and mesh row (``prepare.chip_group('tp8g4')``)."""
    return chip % GROUPS, chip // GROUPS


def slice_weights(gate_up: torch.Tensor, down: torch.Tensor, group: int, part: int):
    """The chip's share, views of the checkpoint tensors: gate, up ``[64, 256, 6144]``, down ``[64, 6144, 256]``.
    ``gate_up`` ``[256, 4096, 6144]`` (gate rows first), ``down`` ``[256, 6144, 2048]``, as stored."""
    e = slice(group * EPG, (group + 1) * EPG)
    inter = gate_up.shape[1] // 2
    c0, c1 = part * inter // SLICES, (part + 1) * inter // SLICES
    return gate_up[e, c0:c1], gate_up[e, inter + c0 : inter + c1], down[e, :, c0:c1]


def clamped_swiglu(g, u, limit=LIMIT):
    return F.silu(g.clamp(max=limit)) * u.clamp(min=-limit, max=limit)


# ---------------------------------------------------------------------------------------------------------- plan
@dataclass
class GroupPlan:
    """What the chips of one group need from the router's picks (identical on its 8 chips)."""

    group: int
    counts: torch.Tensor  # [64] pairs per local expert
    offsets: torch.Tensor  # [64] first buffer row of each expert's region (tile-aligned)
    rows: torch.Tensor  # [R] token id of each buffer row, -1 = padding
    weights: torch.Tensor  # [R] the pair's routing weight (x 2.827), 0 on padding
    slot_rows: torch.Tensor  # [K, S] buffer row of token t's k-th pick in the group, -1 = none
    tokens: torch.Tensor  # [T_g] the tokens with at least one pick in the group
    pair_rows: torch.Tensor  # [S, k] buffer row of (token, top-k slot) if that pick is in the group, else -1

    @property
    def R(self) -> int:
        return int(self.rows.numel())

    @property
    def pairs(self) -> int:
        return int(self.counts.sum())

    @property
    def K(self) -> int:
        return int(self.slot_rows.shape[0])


def plan_group(top_i: torch.Tensor, top_w: torch.Tensor, group: int, pad: int = TILE) -> GroupPlan:
    S, k = top_i.shape
    base = group * EPG
    flat = top_i.flatten().long()
    sel = ((flat >= base) & (flat < base + EPG)).nonzero().flatten()
    local = flat[sel] - base
    order = torch.argsort(local * S + sel // k, stable=True)  # expert-major, tokens ascending inside an expert
    sel, local = sel[order], local[order]
    counts = torch.bincount(local, minlength=EPG)
    padded = (counts + pad - 1) // pad * pad
    offsets = torch.cumsum(padded, 0) - padded
    R = int(padded.sum())
    rows = torch.full((R,), -1, dtype=torch.long)
    weights = torch.zeros(R, dtype=top_w.dtype)
    rank = torch.arange(sel.numel()) - (torch.cumsum(counts, 0) - counts)[local]
    pos = offsets[local] + rank
    tok = sel // k
    rows[pos] = tok
    weights[pos] = top_w.flatten()[sel]
    per_token = torch.bincount(tok, minlength=S)
    K = max(int(per_token.max()), 1)
    slot_rows = torch.full((K, S), -1, dtype=torch.long)
    # k-th pick of each token in this group (in buffer order)
    tok_order = torch.argsort(tok * R + pos, stable=True)
    t_sorted, p_sorted = tok[tok_order], pos[tok_order]
    first = torch.cumsum(per_token, 0) - per_token
    slot = torch.arange(t_sorted.numel()) - first[t_sorted]
    slot_rows[slot, t_sorted] = p_sorted
    pair_rows = torch.full((S * k,), -1, dtype=torch.long)
    pair_rows[sel] = pos
    return GroupPlan(
        group, counts, offsets, rows, weights, slot_rows, (per_token > 0).nonzero().flatten(), pair_rows.view(S, k)
    )


def plan(top_i, top_w, pad: int = TILE) -> list[GroupPlan]:
    return [plan_group(top_i, top_w, g, pad) for g in range(GROUPS)]


# ------------------------------------------------------------------------------------------------------- compute
def chip_partial(x, p: GroupPlan, gate, up, down, *, limit=LIMIT, dtype=torch.float32):
    """Chip (p.group, slice)'s partial ``[S, 6144]`` as the device computes it: gather the group's rows, the 64
    sliced experts on their regions, the routing weight on the activation, then the token-major combine."""
    S = x.shape[0]
    x = x.to(dtype)
    X = torch.zeros(p.R, x.shape[1], dtype=dtype)
    valid = p.rows >= 0
    X[valid] = x[p.rows[valid]]
    Y = torch.zeros(p.R + 1, down.shape[1], dtype=dtype)  # row R: the zero row the combine reads for "no pick"
    for j in range(EPG):
        m = int(p.counts[j])
        if m == 0:
            continue
        a = int(p.offsets[j])
        xs = X[a : a + m]
        act = clamped_swiglu(xs @ gate[j].to(dtype).T, xs @ up[j].to(dtype).T, limit)
        Y[a : a + m] = (act * p.weights[a : a + m, None].to(dtype)) @ down[j].to(dtype).T
    out = torch.zeros(S, down.shape[1], dtype=dtype)
    for k in range(p.K):
        idx = p.slot_rows[k].clone()
        idx[idx < 0] = p.R
        out += Y[idx]
    return out


def tp8g4_routed(x, top_i, top_w, gate_up, down, *, limit=LIMIT, dtype=torch.float32, weights_fn=None, plans=None):
    """The routed output ``[S, 6144]`` as the sum of the 32 chips' partials: over each group's 8 slices (the device's
    reduce over axis 0), then over the 4 groups (axis 1). ``weights_fn(gate, up, down) -> (gate, up, down)`` maps a
    chip's share before use (e.g. ``bfp4_slices``)."""
    plans = plans or plan(top_i, top_w)
    out = torch.zeros(x.shape[0], down.shape[1], dtype=dtype)
    for g in range(GROUPS):
        group_sum = torch.zeros_like(out)
        for s in range(SLICES):
            w = slice_weights(gate_up, down, g, s)
            if weights_fn is not None:
                w = weights_fn(*w)
            group_sum += chip_partial(x, plans[g], *w, limit=limit, dtype=dtype)
        out += group_sum
    return out


# ------------------------------------------------------------------------------------------------- weight formats
def _round_tiles(w, fn):
    """``fn`` on the tiles of ``w.T`` per expert (the matmul operand the device stores: [K, N] tiles)."""
    return fn(w.float().transpose(1, 2).contiguous()).transpose(1, 2)


def bfp4_slices(gate, up, down):
    """A chip's share as the device holds it in BFP4 (ttnn's host conversion, one rounding of the BF16 weights)."""
    from models.experimental.hy4_preview.fused.moe import layout as L

    return tuple(_round_tiles(w, L.bfp4_round) for w in (gate, up, down))


def bfp8_slices(gate, up, down):
    import ttnn

    def rnd(t):
        tt = ttnn.from_torch(t.reshape(1, -1, *t.shape[-2:]), dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT)
        return ttnn.to_torch(tt).float().reshape(t.shape)

    return tuple(_round_tiles(w, rnd) for w in (gate, up, down))


def two_term_bfp4_slices(gate, up, down):
    """Decode's two-term BFP4 (its shared expert's form, ``layout.two_terms_value``): q1 + BFP4(w - q1)."""
    from models.experimental.hy4_preview.fused.moe import layout as L

    return tuple(L.two_terms_value(w.float()) for w in (gate, up, down))


# ---------------------------------------------------------------------------------------------------------- bytes
MB = 1e6


def hop_bytes(plans: list[GroupPlan], top_i: torch.Tensor, act_bytes: int = 2, rows=8, cols=4) -> dict:
    """Per chip per chunk, bytes received over the fabric by each hop (= sent, for these collectives), tp8g4 against
    EP32 (``init_helpers``: chip (r, c) holds experts 64 c + 8 r .. + 7; dispatch along axis 0 inside a column).
    Max over chips where the hop depends on the routing."""
    S = top_i.shape[0]
    h_loc = HIDDEN // cols
    t_loc = S // rows
    ag0 = (rows - 1) * t_loc * h_loc * act_bytes  # [S/8, 1536] -> [S, 1536] over axis 0
    out = {}
    out["tp8g4"] = {
        "AG axis 0 [S/8,1536]->[S,1536] (shared expert's)": ag0,
        "AG axis 1 [S,1536]->[S,6144]": (cols - 1) * S * h_loc * act_bytes,
        "RS axis 1 hidden [S,6144]->[S,1536]": (cols - 1) * S * h_loc * act_bytes,
        "RS axis 0 tokens [S,1536]->[S/8,1536] (+ shared partial)": ag0,
    }
    # EP32: pairs a chip receives in dispatch = pairs to its 8 experts; of them the ones from its own row stay local.
    flat = top_i.long()
    tok_row = (torch.arange(S) // t_loc)[:, None].expand_as(flat)
    owner_row = (flat % EPG) // 8
    recv = torch.zeros(rows, cols)
    remote = torch.zeros(rows, cols)
    for r in range(rows):
        for c in range(cols):
            mine = (flat // EPG == c) & (owner_row == r)
            recv[r, c] = mine.sum()
            remote[r, c] = (mine & (tok_row != r)).sum()
    row_b = HIDDEN * act_bytes
    out["ep32"] = {
        "AG axis 1 [S/8,1536]->[S/8,6144]": (cols - 1) * t_loc * h_loc * act_bytes,
        "AG axis 0 (shared expert)": ag0,
        "dispatch (remote pairs, max chip)": float(remote.max()) * row_b,
        "combine (remote pairs back, max chip)": float(remote.max()) * row_b,
        "RS axis 1 hidden [S/8,6144]->[S/8,1536]": (cols - 1) * t_loc * h_loc * act_bytes,
        "RS axis 0 (shared expert)": ag0,
    }
    out["ep32_pairs_per_chip"] = {"mean": float(recv.mean()), "max": float(recv.max()), "min": float(recv.min())}
    out["ep32_remote_mean_bytes"] = float(remote.mean()) * row_b
    out["tp8g4_rows_per_group"] = [p.R for p in plans]
    out["tp8g4_pairs_per_group"] = [p.pairs for p in plans]
    out["tp8g4_tokens_per_group"] = [int(p.tokens.numel()) for p in plans]
    out["tp8g4_K_per_group"] = [p.K for p in plans]
    for name in ("tp8g4", "ep32"):
        out[name + "_total"] = sum(v for k, v in out[name].items())
    return out


# ------------------------------------------------------------------------------------------------- golden inputs
def golden_moe_inputs(ckpt, streams_attn: torch.Tensor, layer: int, eps: float, hc_eps: float, hc_mag: float):
    """The FFN input ``hn`` [S, 6144] fp32 of ``layer`` from the reference iHC streams after its attention
    (``streams_L{layer}_attn.pt``) and the iHC ``post`` [S, 4]: ``reference/hy4/model.py`` ``_ffn``'s first step."""
    p = f"model.layers.{layer}."
    fn = ckpt.get(p + "hc_mlp_layer.hc_pre.hc_fn", torch.float32)
    base = ckpt.get(p + "hc_mlp_layer.hc_pre.hc_base", torch.float32)
    scale = ckpt.get(p + "hc_mlp_layer.hc_pre.hc_scale", torch.float32)
    norm_w = ckpt.get(p + "post_attention_layernorm.weight", torch.float32)
    x4 = streams_attn.float()
    hc = x4.shape[1]
    flat = x4.flatten(1)
    mixes = (flat @ fn.T) * torch.rsqrt(flat.square().mean(-1, keepdim=True) + eps)
    pre_l, post_l = mixes.split(hc, dim=-1)
    pre = torch.sigmoid(pre_l * scale[0] + base[:hc]) + hc_eps
    post = hc_mag * torch.sigmoid(post_l * scale[1] + base[hc:]) + hc_eps
    h = (pre.unsqueeze(-1) * x4).sum(1)
    hn = h * torch.rsqrt(h.square().mean(-1, keepdim=True) + eps) * norm_w
    return hn, post
