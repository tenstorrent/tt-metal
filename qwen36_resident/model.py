# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Resident Qwen3.6-27B decode on QB2: builds the one-program-per-chip decode step and its torch reference.

A decode step runs `layers` decoder layers (layer l is a full-attention layer when attn_interval > 0 and
(l + 1) % attn_interval == 0, otherwise a Gated DeltaNet layer) and optionally the lm_head, TP-sharded over
a 1 x N mesh. Per chip:
  - 32 streamer cores (4 per DRAM bank) hold a replica of x, run the rmsnorms, stream their column slices
    of every weight through the weight ring and run the projections, conv1d + silu and the MLP;
  - 12 GDN head cores (one per local value head) run the delta-rule recurrence;
  - the attention leader, the tail core and 30 chunk workers run the gated attention over the KV cache;
  - one hub core all-reduces the partials over fabric and multicasts the slots back.
Recurrent state, conv history and KV cache live in DRAM and are updated in place every step; the host
writes the token state (position, x0 = embedding, rope tables) before a step and reads x / logits after.

Weights are given per kind with `copies` distinct layers each ("gdn", "attn": mixer weights of the k-th
layer of that kind use copy k % copies; "mlp", "out": per layer l, copy l % copies). Norm weights are
folded into the following projection. The torch reference applies the same (dequantized) weights and
bf16 rounding points.
"""

import math
import os
import weakref

import numpy as np
import torch
from loguru import logger

import ttnn
from qwen36_resident import PKG_DIR
from qwen36_resident.common import (
    EPS,
    HIDDEN,
    INTER,
    L1_WEIGHT_BUDGET,
    PER_BANK,
    TILE,
    TILE_BYTES,
    Deferred,
    bank_core_cols,
    block_geometry,
    device_zeros,
    f32_bits,
    gate_up_core_cols,
    materialize,
    quantize,
    rect_cover,
    split,
    streamer_cores,
)

KDIR = f"{PKG_DIR}/kernels/"
# GDN
NK, NV, DK, DV, CONV_K = 16, 48, 128, 128, 4
# attention
NQ, NKV, HD, ROT = 24, 4, 256, 64
ROPE_THETA = 10_000_000.0
PROJ_DT, OUT_DT, GU_DT, DOWN_DT, HEAD_DT = (
    ttnn.bfloat8_b,
    ttnn.bfloat8_b,
    ttnn.bfloat4_b,
    ttnn.bfloat8_b,
    ttnn.bfloat8_b,
)
# attention workers (the last one the tail core); more workers split long contexts finer (and change the summation
# order of the partials). 40 still fits a rectangle of the P300 grid next to the streamers.
WORKERS = int(os.environ.get("RESIDENT_ATTN_WORKERS", "32"))
FANIN = 8
# chunk workers take tiles round robin over all banks (attn_common.hpp chunk_of)
ATTN_RR = os.environ.get("RESIDENT_ATTN_RR", "0") == "1"
# chunk workers read their KV tiles over both NOCs (attn_common.hpp two_nocs; the same reads and arithmetic, so
# the same numbers): 24.1 -> 20.5 ms TPOT at 262K, the KV stream is NOC-bound on one NOC
ATTN_2NOC = os.environ.get("RESIDENT_ATTN_2NOC", "1") == "1"
KV_BLOCK = 8  # KV position tiles per streamed block of an attention worker
(
    SEM_SLOTS,
    SEM_ACT,
    SEM_GATHER,
    SEM_FLAG,
    SEM_HEADS,
    SEM_ROWS,
    SEM_Q,
    SEM_m,
    SEM_M,
    SEM_PART,
    SEM_ADDR,
    SEM_LOCAL,
    SEM_GROUP,
    SEM_RADDR,  # row groups: the mixers' row buffer address, leader -> streamers
) = range(14)
HEAD_FP32_CBS = (5, 6, 7, 15, 30, 31)
# the chunked verify's head cores (gdn_chunk_compute.cpp): fp32 CBs unpacked straight to DEST, and every CB
CHUNK_FP32_CBS = (5, 6, 11, 16, 25, 28)
# (GDN of a verify of R > 2 rows in the chunked / WY form, gdn_chunk_*.cpp; 0: the rows one after another)
GDN_CHUNK = os.environ.get("RESIDENT_GDN_CHUNK", "1") == "1"
# (attention of a verify of R > 2 rows in packed passes of several rows, attn_common.hpp; 0: a pass per row)
ATTN_PACKED = os.environ.get("RESIDENT_ATTN_PACKED", "1") == "1"
# (row groups: attention tail cores taking the active users alternately, attn_common.hpp kTails; 1 = one tail. Two
# tails halve the tail's per-user work but take a chunk worker; while the per-user loop is bound by the M round trip
# (not the tail) one tail is faster at long contexts: default 1)
ATTN_TAILS = int(os.environ.get("RESIDENT_ATTN_TAILS", "1"))
HUB_LINKS = int(os.environ.get("RESIDENT_HUB_LINKS", "2"))  # fabric links per neighbour the hubs' all-reduce uses
# (debug, row groups: the attention leader's per-iteration timeline, attn_leader_writer.cpp mark_it)
ATTN_TS = os.environ.get("RESIDENT_ATTN_TS", "0") == "1"
# (debug: the GDN head cores' per-step timeline, gdn_head_compute.cpp mark)
GDN_TS = os.environ.get("RESIDENT_GDN_TS", "0") == "1"
# row groups: size the weight ring from the L1 actually left on the streamers (up to the one-group ring) instead of
# the per-group reservation estimate (act_bytes), which overstates what a second group takes
RING_AUTO = os.environ.get("RESIDENT_RING_AUTO", "1") == "1"
RING_MARGIN = 64 * 1024  # L1 left free below the lowest L1 buffer (later small allocations, e.g. global semaphores)
RING_MIN = 96 * 1024  # a smaller weight ring is refused (row groups), not run slowly
HUB_MC_NOCS = int(os.environ.get("RESIDENT_HUB_MC_NOCS", "2"))  # NOCs the hub multicasts x over (mlp_hub.cpp mc_nocs)
DM_OS_KERNELS = ("streamer_reader.cpp", "streamer_writer.cpp")  # (verify rows: -Os, see kernel())
TOK_PREV = 14  # token state word: the previous verify step's accepted drafts (chunked GDN; none: 0xFFFFFFFF)
TOK_X0 = 64
TOK_SLOT = 4  # token state word: the GDN state slot of row 0 (verify)
TOK_SLOT_ROWS = 15  # the same for a verify of R > 2 rows (words 1 .. R hold their positions)
TOK_ACTIVE, TOK_PRESERVE = 9, 10  # rows mode token state words (kernels/active_rows.hpp)
TOP_K = 32  # sampling::kTopK: the candidates of on-device sampling, the largest top-k it serves
KV_ROW = (HD // TILE) * TILE_BYTES[ttnn.bfloat8_b]


def bf16(t):
    return t.bfloat16().float()


# Live decode models, for the persistent-L1 layout check of traces captured beside them (see l1_layout)
_LIVE_MODELS = weakref.WeakSet()


def l1_layout():
    """the persistent L1 buffers of every live ResidentModel, as a sorted tuple of addresses. A trace replays the CB
    and intermediate addresses its programs got at capture, which were placed around the L1 buffers live then and
    are not checked again: a decode model built after a trace (e.g. PPPrefill's) with another L1 layout can be
    overwritten by the trace's replays (seen 2026-10-30: a B = 8 rows model built after the PP traces were captured
    beside a B = 1 model decoded garbage, and vice versa). Compare it with the layout at capture."""
    return tuple(sorted(a for m in list(_LIVE_MODELS) for a in m.l1_addrs))


def l1_low_water(mesh):
    """the lowest allocated L1 address on the device in the allocator's view (every L1 buffer, the decode models',
    drafters', samplers', ...; an offset from the allocator's base), or None when nothing is allocated"""
    lows = [
        int(b["address"]) for b in ttnn.get_memory_view(mesh, ttnn.BufferType.L1).block_table if b["allocated"] == "yes"
    ]
    return min(lows) if lows else None


def cb_bytes_on(cbs, cores):
    """the CB bytes (no backing tensor, 64 B rounded) the descriptors take on the busiest of `cores` ((x, y) pairs)"""
    want = set(cores)
    per_core = dict.fromkeys(want, 0)
    for cb in cbs:
        if cb.has_buffer():
            continue
        size = -(-cb.total_size // 64) * 64
        for r in cb.core_ranges.ranges():
            for x in range(r.start.x, r.end.x + 1):
                for y in range(r.start.y, r.end.y + 1):
                    if (x, y) in want:
                        per_core[(x, y)] += size
    return max(per_core.values(), default=0)


def cb_extent(cbs):
    """the static CB region the CB descriptors take on their busiest core: bytes above the L1 allocator's base (CBs
    without a backing tensor, each rounded up to 64 B, an upper bound of the allocator's alignment)"""
    per_core = {}
    for cb in cbs:
        if cb.has_buffer():
            continue
        size = -(-cb.total_size // 64) * 64
        for r in cb.core_ranges.ranges():
            for x in range(r.start.x, r.end.x + 1):
                for y in range(r.start.y, r.end.y + 1):
                    per_core[(x, y)] = per_core.get((x, y), 0) + size
    return max(per_core.values(), default=0)


class TraceL1Guard:
    """what a trace's replays rely on: the L1 its programs' CBs and intermediates got at capture still holds no buffer.
    check() raises otherwise: a buffer allocated there since is overwritten by every replay (a silent garbage decode).
    With cb_top (bytes above the allocator's base: the traced programs' static CB region, cb_extent) that region is
    the bound; without it (programs we don't describe, e.g. ttnn ops) the lowest buffer live at capture is, which is
    conservative. The allocator's view covers every L1 buffer but costs a memory-view walk, so a decode trace checks
    the decode models' registry every replay and the allocator every `every` replays."""

    def __init__(self, mesh, what, every=64, cb_top=None):
        self.mesh, self.what, self.every, self.n, self.cb_top = mesh, what, every, 0, cb_top
        if cb_top is not None:
            self.low = cb_top
            self.reg = ttnn.get_allocator_base_address(mesh, ttnn.BufferType.L1) + cb_top
        else:
            self.low = l1_low_water(mesh)
            self.reg = min(l1_layout(), default=None)

    def check(self, full=False):
        reg = min(l1_layout(), default=None)
        bad = reg is not None and (self.reg is None or reg < self.reg)
        self.n += 1
        if full or self.n % self.every == 0:
            low = l1_low_water(self.mesh)
            bad |= low is not None and (self.low is None or low < self.low)
        if bad:
            where = (
                "in its programs' CB region" if self.cb_top is not None else "below the L1 buffers live at its capture"
            )
            raise RuntimeError(
                f"{self.what} trace: an L1 buffer now lies {where}, so its replays overwrite it; capture the trace "
                "again after allocating it (model.TraceL1Guard)"
            )


def rmsnorm(x, w=None):
    y = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + EPS)
    return y if w is None else y * w


def is_attn(l, interval):
    return interval > 0 and (l + 1) % interval == 0


class Dims:
    def __init__(self, n, banks, vocab=0):
        self.n, self.banks = n, banks
        step = TILE * banks
        pad = lambda c: (c + step - 1) // step * step
        # GDN projection row: q | k | v (conv) | z | a | b
        self.nk, self.nv = NK // n, NV // n
        self.gq, self.gv = self.nk * DK, self.nv * DV
        self.conv_ch = 2 * self.gq + self.gv
        self.z0 = self.conv_ch
        self.a0 = self.z0 + self.gv
        self.b0 = self.a0 + self.nv
        self.g_cols = pad(self.b0 + self.nv)
        self.conv_tiles = self.conv_ch // TILE
        # attention projection row: q heads | gates | k | v
        self.h = NQ // n
        assert NKV // n == 1, "one KV head per chip"
        self.Dt = HD // TILE
        self.aq = self.h * HD
        self.gate0, self.k0 = self.aq, 2 * self.aq
        self.v0 = self.k0 + HD
        self.a_cols = pad(self.v0 + HD)
        assert self.aq == self.gv, "the out-projections of both mixers take the same K"
        self.Ot = self.gv // TILE
        self.ic = INTER // n
        self.It = self.ic // TILE
        self.Ht = HIDDEN // TILE
        self.vocab = pad(vocab // n) if vocab else 0  # lm_head columns per chip (padded)
        self.vocab_chip = vocab // n  # of which real vocab entries


def rope_tables(pos):
    inv = 1.0 / ROPE_THETA ** (torch.arange(0, ROT, 2, dtype=torch.float64) / ROT)
    ang = pos * inv
    return torch.cos(ang).float(), torch.sin(ang).float()


def rope(t, cos, sin):
    h = ROT // 2
    t = t.clone()
    a, b = t[..., :h].clone(), t[..., h:ROT].clone()
    t[..., :h] = a * cos - b * sin
    t[..., h:ROT] = b * cos + a * sin
    return t


def quantize_rows(t, dtype=ttnn.bfloat8_b):
    """bf8 of each row on its own (bf8 shares exponents within 16 elements of a row)."""
    rows = t.reshape(-1, t.shape[-1])
    padded = torch.zeros((rows.shape[0] + TILE - 1) // TILE * TILE, rows.shape[1])
    padded[: rows.shape[0]] = rows
    return quantize(padded, dtype)[: rows.shape[0]].reshape(t.shape)


class State:
    """Decode state (torch): GDN recurrent states / conv histories per GDN copy, KV caches per attention
    copy (positions < pos, bf8), shared by the device model (as its initial DRAM contents) and the
    reference (updated in place by torch_step).
    kv=False: no host KV images (K, V None): the device model allocates its KV caches on device, zero-filled
    (what a served model does: at max_pos 256K the fp32 images would be ~34 GB of host memory)."""

    def __init__(self, d, gdn_copies, attn_copies, pos, max_pos, seed=1, zero=False, kv=True):
        g = torch.Generator().manual_seed(seed)
        rn = (lambda *s: torch.zeros(*s)) if zero else (lambda *s: torch.randn(*s, generator=g))
        self.pos, self.max_pos = pos, max_pos
        # [copy][chip] fp32 [nv, DK, DV]
        self.gdn = [[0.05 * rn(d.nv, DK, DV) for _ in range(d.n)] for _ in range(gdn_copies)]
        # [copy][chip] [3, conv_ch] inputs at pos - 3, pos - 2, pos - 1 (bf16)
        self.hist = [[bf16(rn(CONV_K - 1, d.conv_ch)) for _ in range(d.n)] for _ in range(gdn_copies)]
        # [copy][chip] bf8 [max_pos, HD], rows >= pos zero
        self.K, self.V = ([], []) if kv else (None, None)
        for _ in range(attn_copies if kv else 0):
            ks, vs = [], []
            for _ in range(d.n):
                k, v = quantize_rows(rn(max_pos, HD)), quantize_rows(rn(max_pos, HD))
                k[pos:], v[pos:] = 0, 0
                ks.append(k)
                vs.append(v)
            self.K.append(ks)
            self.V.append(vs)


def random_weights(d, gdn_copies, attn_copies, mlp_copies, seed=0):
    g = torch.Generator().manual_seed(seed)
    rn = lambda *s: torch.randn(*s, generator=g)
    w = dict(gdn=[], attn=[], mlp=[], out=[])
    for _ in range(gdn_copies):
        chips = []
        for _ in range(d.n):
            Wq = torch.zeros(HIDDEN, d.g_cols)
            Wq[:, : d.b0 + d.nv] = rn(HIDDEN, d.b0 + d.nv) / math.sqrt(HIDDEN)
            chips.append(
                dict(
                    Wq=quantize(Wq, PROJ_DT),
                    taps=bf16(0.5 * rn(d.conv_ch, CONV_K)),
                    dt=0.5 * rn(d.nv),
                    neg_a=-torch.exp(torch.rand(d.nv, generator=g) * 2 - 1),
                )
            )
        w["gdn"].append(dict(chips=chips, norm_w=bf16(1 + 0.1 * rn(DV))))
    for _ in range(attn_copies):
        chips = []
        for _ in range(d.n):
            Wq = torch.zeros(HIDDEN, d.a_cols)
            Wq[:, : d.v0 + HD] = rn(HIDDEN, d.v0 + HD) / math.sqrt(HIDDEN)
            chips.append(dict(Wq=quantize(Wq, PROJ_DT)))
        w["attn"].append(dict(chips=chips, wq=bf16(1 + 0.1 * rn(HD)), wk=bf16(1 + 0.1 * rn(HD))))
    for _ in range(mlp_copies):
        w["out"].append([quantize(rn(d.gv, HIDDEN) * (0.5 / math.sqrt(d.gv)), OUT_DT) for _ in range(d.n)])
        w["mlp"].append(
            [
                dict(
                    G=quantize(rn(HIDDEN, d.ic) / math.sqrt(HIDDEN), GU_DT),
                    U=quantize(rn(HIDDEN, d.ic) / math.sqrt(HIDDEN), GU_DT),
                    D=quantize(rn(d.ic, HIDDEN) * (0.5 / math.sqrt(d.ic)), DOWN_DT),
                )
                for _ in range(d.n)
            ]
        )
    if d.vocab:
        w["head"] = [quantize(rn(HIDDEN, d.vocab) / math.sqrt(HIDDEN), HEAD_DT) for _ in range(d.n)]
    return w


def torch_step(x0, d, w, st, layers, interval, lm_head=False, attn_out=None, xs=None):
    """One decode step at st.pos (fp32 reference, bf16 at the device's rounding points); updates st in place
    (states, histories, KV rows at pos) and returns x (and the per-chip logits with lm_head). xs (a list) gets
    x after every layer."""
    pos = st.pos
    cos, sin = rope_tables(pos)
    x = x0.clone()
    g_i = a_i = 0
    for l in range(layers):
        h = bf16(rmsnorm(x))
        parts = []
        if is_attn(l, interval):
            c = a_i % len(w["attn"])
            wa = w["attn"][c]
            for chip, cw in enumerate(wa["chips"]):
                y = bf16(h @ cw["Wq"])
                q = y[: d.aq].view(d.h, HD)
                gate = y[d.gate0 : d.k0].view(d.h, HD)
                k, v = y[d.k0 : d.v0], y[d.v0 : d.v0 + HD]
                qn = rope(rmsnorm(q, wa["wq"] / math.sqrt(HD)), cos, sin)
                kn = bf16(rope(rmsnorm(k, wa["wk"]), cos, sin))
                K = torch.cat([st.K[c][chip][:pos], kn[None]])
                V = torch.cat([st.V[c][chip][:pos], v[None]])
                p = torch.softmax(qn @ K.T, dim=-1)
                o = bf16((p @ V) * torch.sigmoid(gate)).flatten()
                if attn_out is not None:
                    attn_out.append((l, chip, o))
                st.K[c][chip][pos] = quantize_rows(kn[None])[0]
                st.V[c][chip][pos] = quantize_rows(v[None])[0]
                parts.append(o @ w["out"][l % len(w["out"])][chip] + (x if chip == 0 else 0))
            a_i += 1
        else:
            c = g_i % len(w["gdn"])
            wg = w["gdn"][c]
            scale = DK**-0.5
            for chip, cw in enumerate(wg["chips"]):
                y = bf16(h @ cw["Wq"])
                qkv = y[: d.conv_ch]
                window = torch.cat([st.hist[c][chip], qkv[None]], 0)  # oldest first
                st.hist[c][chip] = window[1:]
                conv = bf16(torch.nn.functional.silu((window * cw["taps"].T).sum(0)))
                q, k, v = conv[: d.gq], conv[d.gq : 2 * d.gq], conv[2 * d.gq : d.conv_ch]
                z, a, b = y[d.z0 : d.a0], y[d.a0 : d.b0], y[d.b0 : d.b0 + d.nv]
                outs = []
                for hh in range(d.nv):
                    kh = hh // (d.nv // d.nk)
                    qn = q[kh * DK : (kh + 1) * DK]
                    qn = qn * torch.rsqrt(qn.pow(2).sum() + 1e-6) * scale
                    kn = k[kh * DK : (kh + 1) * DK]
                    kn = kn * torch.rsqrt(kn.pow(2).sum() + 1e-6)
                    decay = torch.exp(cw["neg_a"][hh] * torch.nn.functional.softplus(a[hh] + cw["dt"][hh]))
                    beta = torch.sigmoid(b[hh])
                    S = st.gdn[c][chip][hh]
                    delta = beta * (v[hh * DV : (hh + 1) * DV] - (decay * kn) @ S)
                    S = decay * S + torch.outer(kn, delta)
                    st.gdn[c][chip][hh] = S
                    on = rmsnorm(qn @ S, wg["norm_w"])
                    outs.append(bf16(on * torch.nn.functional.silu(z[hh * DV : (hh + 1) * DV])))
                o = torch.cat(outs)
                if attn_out is not None:
                    attn_out.append((l, chip, o))
                parts.append(o @ w["out"][l % len(w["out"])][chip] + (x if chip == 0 else 0))
            g_i += 1
        x = sum(parts)
        h = bf16(rmsnorm(x))
        m = w["mlp"][l % len(w["mlp"])]
        x = sum(
            bf16(torch.nn.functional.silu(h @ cm["G"]) * (h @ cm["U"])) @ cm["D"] + (x if chip == 0 else 0)
            for chip, cm in enumerate(m)
        )
        if xs is not None:
            xs.append(x)
    st.pos += 1
    if lm_head:
        h = bf16(rmsnorm(x))
        return x, [h @ wh for wh in w["head"]]
    return x


class BankLayout:
    """interleaved_bank_layout (bench_resident_mlp) as one gather: the tile order is computed once per
    (Kt, core_cols, sb) and applied to every weight of that shape."""

    def __init__(self, Kt, core_cols, sb):
        self.Kt, self.sb = Kt, sb
        nkb = Kt // sb
        self.width = PER_BANK * max(len(c) for cols in core_cols for c in cols)
        self.banks = len(core_cols)
        rows, cols_ = [], []
        for cols in core_cols:
            src_k = torch.full((Kt * self.width,), -1, dtype=torch.long)
            src_c = torch.full((Kt * self.width,), -1, dtype=torch.long)
            for j, cj in enumerate(cols):
                for t, col in enumerate(cj):
                    for kb in range(nkb):
                        slot = (t * nkb + kb) * PER_BANK + j
                        lin = slot * sb + torch.arange(sb)
                        src_k[lin] = kb * sb + torch.arange(sb)
                        src_c[lin] = col
            rows.append(src_k)
            cols_.append(src_c)
        self.src_k, self.src_c = torch.stack(rows), torch.stack(cols_)  # [banks, Kt * width]

    def __call__(self, w):
        """w [K, N] -> [K, banks * width] (see interleaved_bank_layout)"""
        K, N = w.shape
        Kt, Nt = K // TILE, N // TILE
        tiles = w.reshape(Kt, TILE, Nt, TILE).permute(0, 2, 1, 3)
        tiles = torch.cat([tiles, torch.zeros(Kt, 1, TILE, TILE, dtype=w.dtype)], dim=1)  # column Nt = zeros
        k = self.src_k.clamp(min=0)
        c = torch.where(self.src_c < 0, torch.full_like(self.src_c, Nt), self.src_c)
        g = tiles[k, c]  # [banks, Kt * width, 32, 32]
        g = g.reshape(self.banks, Kt, self.width, TILE, TILE).permute(1, 3, 0, 2, 4)
        return g.reshape(K, self.banks * self.width * TILE)


def worker_rect(mesh, used, count):
    """The first rectangle of `count` free cores (row-major search, widest first)."""
    g = mesh.compute_with_storage_grid_size()
    for wd in range(min(count, g.x), 0, -1):
        if count % wd:
            continue
        hgt = count // wd
        for y0 in range(g.y - hgt + 1):
            for x0 in range(g.x - wd + 1):
                cells = [(x, y) for y in range(y0, y0 + hgt) for x in range(x0, x0 + wd)]
                if not any(c in used for c in cells):
                    return [ttnn.CoreCoord(x, y) for x, y in cells]
    raise RuntimeError(f"no free rectangle of {count} cores")


def chunk_of(w, tp, banks, chunks=None):
    """Mirror of attn_common.hpp chunk_of: (bank, j, J, rows) of chunk worker w of `chunks` (default WORKERS - 1;
    round robin: (None, w, chunks, tiles))."""
    W = chunks or WORKERS - 1
    if ATTN_RR:
        return None, w, W, (tp - w + W - 1) // W if tp > w else 0
    live = min(tp, banks)
    if live == 0:
        return 0, 0, 1, 0
    b, j = w % live, w // live
    J = (W - b + live - 1) // live
    rows = (tp - b + banks - 1) // banks
    return b, j, J, ((rows - j + J - 1) // J if rows > j else 0)


def tile_bytes_rows_replicated(v):
    """bf16 32x32 tile (TILE layout bytes) whose 32 rows all equal v [32]."""
    t = torch.tensor(v, dtype=torch.float32).bfloat16().view(torch.int16).numpy().astype(np.uint16)
    face_rows = [np.tile(t[:16], (16, 1)), np.tile(t[16:], (16, 1))]
    return np.concatenate([face_rows[0], face_rows[1], face_rows[0], face_rows[1]]).reshape(-1)


def rope_compact(n):
    """[n, ROT] uint16: for positions 0..n-1 the bf16 bits of rope_tables' cos (ROT / 2) then sin (ROT / 2), the
    rope table the feeding hub expands (mlp_hub.cpp expand_rope; 128 B per position)"""
    inv = 1.0 / ROPE_THETA ** (torch.arange(0, ROT, 2, dtype=torch.float64) / ROT)
    ang = torch.arange(n, dtype=torch.float64)[:, None] * inv
    cos, sin = torch.cos(ang).float(), torch.sin(ang).float()
    return torch.cat([cos, sin], dim=1).bfloat16().view(torch.int16).numpy().astype(np.uint16)


def rope_expand(c):
    """[..., ROT] rope_compact rows -> [..., 3, 1024] uint16: the cos, sin and -sin tiles, as
    tile_bytes_rows_replicated lays them out (-sin: the sign bits flipped, as bf16(-x) == -bf16(x))"""
    h = ROT // 2
    out = []
    for v in (c[..., :h], c[..., h:], c[..., h:] ^ np.uint16(0x8000)):
        lo = np.repeat(v[..., None, :16], 16, axis=-2).reshape(*v.shape[:-1], -1)
        hi = np.repeat(v[..., None, 16:], 16, axis=-2).reshape(*v.shape[:-1], -1)
        out.append(np.concatenate([lo, hi, lo, hi], axis=-1))
    return np.stack(out, axis=-2)


def rope_words(n):
    """[n, 3, 1024] uint16: for positions 0..n-1 the cos, sin and -sin tiles of rope_tables, as
    tile_bytes_rows_replicated lays them out"""
    return rope_expand(rope_compact(n))


def tiny_order(mat):
    """[B, N] -> the elements of its batch x 32 tiles in device order (per tile: face 0 = columns 0..15 of
    every row, then face 1)"""
    B, N = mat.shape
    return mat.reshape(B, N // TILE, 2, TILE // 2).permute(1, 2, 0, 3).reshape(-1)


def tiny_unorder(flat, B):
    """inverse of tiny_order"""
    N = flat.numel() // B
    return flat.reshape(N // TILE, 2, B, TILE // 2).permute(2, 0, 1, 3).reshape(B, N)


def tiny_index(cols, u, B):
    """device-order index of column `cols` of row u in batch x 32 tiles"""
    t, e = cols // TILE, cols % TILE
    return t * TILE * B + (e // 16) * 16 * B + u * 16 + e % 16


def bf16_bits(t):
    return t.flatten().bfloat16().view(torch.int16).numpy().astype(np.uint16)


class ResidentModel:
    def __init__(
        self,
        mesh,
        d,
        w,
        st,
        layers,
        interval,
        lm_head=False,
        timeline=False,
        packet_bytes=4096,
        embed=None,
        verify=False,
        fc=None,
        tok=None,
        feed_peer=None,
        taps=None,
        kv_pool_blocks=None,
        block_tables=None,
        kv_block_size=64,
        rows=False,
        groups=1,
    ):
        """st: the decode state (State), or one State per user for a batch of len(st) users. More than 8 users
        run as sub-batches of 8, one launch each (the launches share the weights and buffers, each has its
        own decode state). embed (the embedding table on device, one row per page; batch 1 with the lm_head):
        each step also writes the next step's token state from its own token (step(None)), drawn on device
        with the parameters of set_sampling (greedy until set).
        verify: st is one user's State and a step runs two rows of it, at positions p and p + 1 (a token
        and its draft); see token_state for continuing from either row. fc (the MTP draft layer): per chip
        its K slice [2 HIDDEN / n, HIDDEN] of the fc that makes x0 from the token state's two vectors per row
        (see streamer_common.hpp). tok: the token state tensor (default: a new one, see tok_tensor).
        verify with embed: the hub feeds the MTP draft loop (mlp_hub.cpp kinds 1 and 2), writing the next
        token state of feed_peer's model (the draft model for the main one and back).
        verify = R (an int, 4 or 8): a step runs R rows of the user at p .. p + R - 1 (DFlash2: a token and R - 1
        drafts); GDN states in R slots, the conv history a ring of single-row slots (streamer_common.hpp), the next
        step continues from row a at ring + 1 + a, slot (slot + a) % R (token_state).
        taps: [(layer, tensor)] (up to 5): after each step the tensor (bf16 TILE, [R, HIDDEN / n] per chip, rows padded
        to 32) holds this chip's columns of every row's x after decoder layer `layer` (the DFlash2 drafter's context
        taps, mlp_hub.cpp).
        kv_pool_blocks (paged KV, attn_paged.hpp): the KV caches are one pool per K / V of this many blocks of
        kv_block_size positions, shared by every user and sub-batch, instead of a contiguous max_pos cache per user;
        block_tables: per user the physical block of each logical block (default: user u takes blocks
        u * blocks_per_user ...), see set_page_table. The users' States are placed into the pool through them.
        rows (multi-user rows mode, kernels/active_rows.hpp): the rows of a launch are independent users, each run
        only when its active bit is set (token_state active=), with the conv history in canonical order; a user's
        arithmetic does not depend on its neighbours. Batch > 1, no verify. With embed, the hub of each launch feeds
        each active row its own token (mlp_hub.cpp rows mode; per-row sampling parameters, set_sampling rows=).
        groups (rows mode, kernels/active_rows.hpp): row groups of `batch` rows that one launch carries, the weights
        streamed once for all of them (custom_mm takes at most 8 rows, so more users per launch come as groups).
        groups > 1: exactly 8 * groups users (st), one launch; rows mode, paged KV and the hub feed, no fc / verify /
        taps; 1, 2 or 4. Anything else raises rather than running as more launches.
        verify rows (DB1, verify = 8 with rows and groups > 1): row group g is user g's verify of 8 rows (st: one State
        per group, block_tables one per group); per user its own positions, ring step, GDN slot and prev
        (token_state_vrows); the GDN head core g of each value head runs group g's rows (the verify chain stays on one
        core); taps allowed (group g's rows land in rows 8 g .. of the tap tiles)."""
        self.mesh, self.d, self.layers, self.interval, self.lm_head = mesh, d, layers, interval, lm_head
        R = 0 if not verify else (2 if verify is True else int(verify))
        assert R in (0, 2, 4, 8), "verify runs 2, 4 or 8 rows"
        self.verify, self.R = bool(verify), R
        self.gdn_chunk = R > 2 and GDN_CHUNK
        self.taps = list(taps or [])
        assert len(self.taps) <= 5
        self.rows = rows
        # verify rows (DB1): a row group per user, its 8 rows a verify of that user (see the docstring)
        self.vrows = bool(verify) and rows and groups > 1
        assert not rows or not verify or self.vrows, "rows mode: independent users (verify: row groups, vrows)"
        if groups not in (1, 2, 4):
            raise NotImplementedError(f"row groups per launch: groups={groups} (1, 2 or 4)")
        if groups > 1 and not (
            rows
            and kv_pool_blocks is not None
            and embed is not None
            and fc is None
            and ((not verify and not taps) or (self.vrows and R == 8))
        ):
            raise ValueError(
                "row groups: rows mode with paged KV and the hub feed (embed), no fc; verify (8 rows per group, taps) "
                "or neither verify nor taps"
            )
        if self.vrows and self.gdn_chunk:
            logger.info("verify rows: the sequential GDN chain (gdn_chunk has no row groups yet)")
            self.gdn_chunk = False
        self.groups = groups
        # (row groups: two attention tail cores, attn_common.hpp kTails; the chunk workers are the others)
        self.attn_tails = ATTN_TAILS if rows and groups > 1 and not verify else 1
        assert self.attn_tails in (1, 2)
        # with row groups the rows' positions live past the token state's x0 / rope (active_rows.hpp tok_pos_off);
        # one group keeps them in the head's words 1 .. batch
        self.tok_pos_off = 0
        self.fc = fc is not None
        self.sample = embed is not None and not verify  # the hub draws the fed token (sampling.hpp)
        if self.vrows:
            assert len(st) == groups, "verify rows: one State per row group"
            st = [s for s in st for _ in range(R)]  # (user g's R slots start as copies of its state)
        elif verify:
            assert isinstance(st, State)
            st = [st] * R
        sts_all = list(st) if isinstance(st, (list, tuple)) else [st]
        if groups > 1:
            assert len(sts_all) == 8 * groups, f"row groups: 8 users per group, one launch ({len(sts_all)} users)"
            n_sub = 1
        else:
            n_sub = -(-len(sts_all) // 8)
        assert len(sts_all) % n_sub == 0
        sts = sts_all[: len(sts_all) // n_sub]  # the users of one launch
        B = self.batch = len(sts) // groups  # rows per group (the tiles' rows)
        self.launch_rows = len(sts)  # users per launch: groups x batch
        self.n_sub, self.users = n_sub, len(sts_all)
        assert B in (1, 2, 4, 8), "custom_mm takes 1, 2, 4 or 8 rows"
        assert not rows or B > 1, "rows mode: a batch of users"
        assert len({s.max_pos for s in sts_all}) == 1
        n, banks = d.n, d.banks
        self.n_attn = sum(is_attn(l, interval) for l in range(layers))
        self.n_gdn = layers - self.n_attn
        gdn_copies, attn_copies = max(len(w["gdn"]), 1), max(len(w["attn"]), 1)
        mlp_copies = len(w["mlp"])
        assert len(w["out"]) == mlp_copies
        if verify:
            # row 1 continues each layer's state from row 0 of the same step: no state is reused within a step
            assert gdn_copies == self.n_gdn or not self.n_gdn, "verify: one GDN copy (state) per GDN layer"
            assert attn_copies == self.n_attn or not self.n_attn, "verify: one KV cache per attention layer"

        # ---- cores
        core_groups, used = streamer_cores(mesh)
        cores = [c for grp in core_groups for c in grp]
        S = len(cores)
        self.S = S
        grid = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in cores])
        xs, ys = [c.x for c in cores], [c.y for c in cores]
        x0r, x1r, y0r, y1r = min(xs), max(xs), min(ys), max(ys)
        hub = next(ttnn.CoreCoord(x, y) for x in range(x0r, x1r + 1) for y in range(y0r, y1r + 1) if (x, y) not in used)
        used.add((hub.x, hub.y))
        workers = worker_rect(mesh, used, WORKERS)
        used |= {(c.x, c.y) for c in workers}
        cgrid = mesh.compute_with_storage_grid_size()
        free = [ttnn.CoreCoord(x, y) for y in range(cgrid.y) for x in range(cgrid.x) if (x, y) not in used]
        # GDN head cores: G per value head (head-major: core hh * G + g), core g of a head takes users
        # g * B / G .. (g + 1) * B / G
        # (verify rows: G = groups, core g of a head runs group g's rows, all of one user: gdn_head_reader.cpp)
        G = (
            groups
            if self.vrows
            else 1
            if verify
            else int(os.environ.get("RESIDENT_HEAD_GROUPS", "0"))
            or max(g for g in range(1, B + 1) if B % g == 0 and 1 + d.nv * g <= len(free))
        )
        assert B % G == 0 and 1 + d.nv * G <= len(free), (B, G, len(free))
        self.head_groups = G
        leader, heads = free[0], free[1 : 1 + d.nv * G]
        tail_core = workers[-1]
        tail_cores = workers[len(workers) - self.attn_tails :]  # (kTails: the second tail is the worker before)
        assert all((WORKERS - 1 - t) % FANIN for t in range(self.attn_tails)), "a tail core is a group head"
        C = lambda cs: ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in cs])
        hub_grid, lead_grid, tail_grid, head_grid = C([hub]), C([leader]), C(tail_cores), C(heads)
        work_grid = ttnn.CoreRangeSet([ttnn.CoreRange(workers[0], workers[-1])])
        chunk_grid = work_grid.subtract(tail_grid)
        prep_grid = lead_grid.merge(tail_grid)
        attn_grid = work_grid.merge(lead_grid)
        mixer_grid = head_grid.merge(prep_grid)
        all_grid = grid.merge(hub_grid).merge(attn_grid).merge(head_grid)
        nohub_grid = grid.merge(attn_grid).merge(head_grid)
        phys = lambda c: mesh.worker_core_from_logical_core(c)
        sphys = [phys(c) for c in cores]
        hub_p, lead_p, tail_p = phys(hub), phys(leader), phys(tail_core)
        tail2_p = [phys(c) for c in tail_cores[:-1]]  # (kTails: the second tail core)
        head_p = [phys(c) for c in heads]
        wphys = [phys(c) for c in workers]

        tiny, full = ttnn.Tile([B, TILE]), ttnn.Tile([TILE, TILE])
        rep, per_chip = ttnn.ReplicateTensorToMesh(mesh), ttnn.ShardTensorToMesh(mesh, dim=0)
        dram = ttnn.DRAM_MEMORY_CONFIG
        dram_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(banks - 1, 0))})

        # Every setup tensor is synchronized before the next is created: tilize ops queued behind each other
        # while L1 buffers are still being allocated were seen to hang the device.
        def settled(t):
            ttnn.synchronize_device(mesh)
            return t

        self.l1_addrs = []  # persistent L1 buffers (see l1_layout)

        def l1(t, grid_, shard, dtype=ttnn.bfloat16, tile=tiny, mapper=rep, layout=ttnn.TILE_LAYOUT):
            out = settled(_l1(t, grid_, shard, dtype, tile, mapper, layout))
            self.l1_addrs.append(out.buffer_address())
            return out

        def _l1(t, grid_, shard, dtype=ttnn.bfloat16, tile=tiny, mapper=rep, layout=ttnn.TILE_LAYOUT):
            return ttnn.from_torch(
                t,
                dtype=dtype,
                layout=layout,
                device=mesh,
                tile=tile if layout == ttnn.TILE_LAYOUT else None,
                mesh_mapper=mapper,
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                    ttnn.BufferType.L1,
                    ttnn.ShardSpec(grid_, shard, ttnn.ShardOrientation.ROW_MAJOR),
                ),
            )

        def dram_t(t, dtype, layout=ttnn.TILE_LAYOUT, mapper=per_chip):
            return settled(
                ttnn.from_torch(t, dtype=dtype, layout=layout, device=mesh, memory_config=dram, mesh_mapper=mapper)
            )

        # ---- column assignment (the remainders of the projections and gate|up on different cores)
        qg_bank, qa_bank = d.g_cols // TILE // banks, d.a_cols // TILE // banks
        ht_bank, it_bank = d.Ht // banks, d.It // banks
        v_bank = d.vocab // TILE // banks if d.vocab else 0
        nqg_split, nqa_split = split(qg_bank, PER_BANK), split(qa_bank, PER_BANK)
        nd_split = split(ht_bank, PER_BANK)
        g_extra = qg_bank % PER_BANK
        ng_split = split(it_bank, PER_BANK, g_extra)
        nv_split = split(v_bank, PER_BANK) if v_bank else [(0, 0)] * PER_BANK
        mx = lambda sp: max(c for c, _ in sp)
        nq_max, nd_max, ng_max = max(mx(nqg_split), mx(nqa_split)), mx(nd_split), mx(ng_split)
        nv_max = max(mx(nv_split), 1)
        assert nq_max <= TILE and ng_max <= TILE
        # column block of the elementwise phases: the whole 32x32 views (32 / B tiles each) holding the most
        # projection or gate columns of a core
        blk_tiles = (TILE // B) * -(-max(nq_max, ng_max) * B // TILE)
        self._blk = blk_tiles

        # ---- DRAM weights: per entry, copies stacked along K (a fixed per-bank stride)
        # (fc: the GDN projection entry carries the fc, columns split as the out-projection's)
        entries = [
            (2 * d.Ht // n, PROJ_DT, HIDDEN) if self.fc else (d.Ht, PROJ_DT, d.g_cols),
            (d.Ht, PROJ_DT, d.a_cols),
            (d.Ot, OUT_DT, HIDDEN),
            (d.Ht, GU_DT, 2 * d.ic),
            (d.It, DOWN_DT, HIDDEN),
            (d.Ht, HEAD_DT, d.vocab),
        ]
        geo = [block_geometry(Kt, dt) for Kt, dt, _ in entries]
        cols_of = [
            bank_core_cols(d.Ht if self.fc else d.g_cols // TILE, banks),
            bank_core_cols(d.a_cols // TILE, banks),
            bank_core_cols(d.Ht, banks),
            gate_up_core_cols(d.It, banks, g_extra),
            bank_core_cols(d.Ht, banks),
            bank_core_cols(d.vocab // TILE, banks) if d.vocab else None,
        ]
        # per entry, per copy: a function giving the copy's per-chip matrices, called as the copy is placed (the
        # weights may be Deferred, read from the checkpoint only then: weights.load(lazy=True))
        mat = materialize
        gate_up = lambda cm: mat(cm["GU"]) if "GU" in cm else torch.cat([mat(cm["G"]), mat(cm["U"])], dim=1)
        per_entry_mats = [
            [lambda: list(fc)] if self.fc else [lambda wg=wg: [mat(c["Wq"]) for c in wg["chips"]] for wg in w["gdn"]],
            [lambda wa=wa: [mat(c["Wq"]) for c in wa["chips"]] for wa in w["attn"]],
            [lambda o=o: [mat(x) for x in o] for o in w["out"]],
            [lambda m=m: [gate_up(cm) for cm in m] for m in w["mlp"]],
            [lambda m=m: [mat(cm["D"]) for cm in m] for m in w["mlp"]],
            [lambda: [mat(h) for h in w["head"]]] if lm_head else [],
        ]
        # one DRAM tensor per copy, allocated back to back so the copies sit at a fixed per-bank stride
        self._keep = []
        self.w_entries = []  # (first copy's tensor or None, stride, copies)
        for (Kt, dt, _), (sb, _, _, _), cc, mats in zip(entries, geo, cols_of, per_entry_mats):
            if not mats:
                self.w_entries.append((None, 0, 1))
                continue
            layout = BankLayout(Kt, cc, sb)
            mc = ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                ttnn.BufferType.DRAM,
                ttnn.ShardSpec(dram_grid, [Kt * TILE, layout.width * TILE], ttnn.ShardOrientation.ROW_MAJOR),
            )
            copies = []
            for m in mats:
                t = ttnn.from_torch(
                    torch.stack([layout(mm) for mm in m()]).unsqueeze(1),
                    dtype=dt,
                    layout=ttnn.TILE_LAYOUT,
                    device=mesh,
                    memory_config=mc,
                    mesh_mapper=per_chip,
                )
                copies.append(settled(t))
            addrs = [t.buffer_address() for t in copies]
            stride = (addrs[1] - addrs[0]) & 0xFFFFFFFF if len(addrs) > 1 else 0
            assert all(((b - a) & 0xFFFFFFFF) == stride for a, b in zip(addrs, addrs[1:])), (
                "weight copies are not at a fixed DRAM stride",
                addrs[:4],
            )
            self._keep += copies
            self.w_entries.append((copies[0], stride, len(copies)))
        Deferred.release()

        lcm = math.lcm(*TILE_BYTES.values())

        # the activation buffers grow with the batch: the weight ring gives up their extra L1 (streamer CBs,
        # and the L1 tensors that take the same addresses on every core: mixer rows and output, logits)
        # (row groups: every per-row buffer once per group; the conv taps are shared, counted per group anyway)
        def act_bytes(b, gr=1):
            blk = (TILE // b) * -(-max(nq_max, ng_max) * b // TILE)
            cbs_ = 64 * b * gr * (2 * d.Ht + d.It + 2 * nd_max + 13 * blk) + (18432 if b > 1 else 0)
            return cbs_ + b * gr * 2 * (d.gv + nv_max * TILE + max(d.g_cols, d.a_cols))

        # (each further sub-batch has its own logits, an L1 tensor)
        extra = (n_sub - 1) * B * 2 * nv_max * TILE if lm_head else 0
        # verify: two more conv history outputs and the row selections (R > 2 rows: 9 row selections; verify rows: the
        # 3 shared S_j and 6 H_j,b per group, streamer_common.hpp)
        n_shift = 3 + 6 * groups if self.vrows else 9
        extra += (2 * blk_tiles * 64 * B + 3 * 2048) if R == 2 else (n_shift * 2048 if verify else 0)
        ring_bytes = ((L1_WEIGHT_BUDGET - act_bytes(B, groups) + act_bytes(1) - extra) // lcm) * lcm
        # (row groups: the one-group ring, the most the L1 measurement below may grow the ring to)
        ring_one = ((L1_WEIGHT_BUDGET - act_bytes(B) + act_bytes(1) - extra) // lcm) * lcm
        self.ring_bytes = ring_bytes

        # ---- streamer tensors
        # (row groups: a shard holds every group's rows, tile row gr = group gr: o block gr at gr * o_tiles tiles)
        LR = self.launch_rows
        # (row groups: o shares the streamers' activation CB instead, streamer_common.hpp kOinInAct)
        self.o_in_t = l1(torch.zeros(S * LR, d.gv), grid, [LR, d.gv]) if groups == 1 else None
        o_in_addr = self.o_in_t.buffer_address() if self.o_in_t is not None else 0
        self.logits_t = l1(torch.zeros(S * LR, nv_max * TILE), grid, [LR, nv_max * TILE]) if lm_head else None
        blk = blk_tiles * TILE * B
        # conv taps per (streamer, GDN copy): [tap][32 x 1x32 tiles], only the core's q|k|v columns
        self._S, self._gdn_copies, self._attn_copies = S, gdn_copies, attn_copies
        self._nqg_split, self._qg_bank = nqg_split, qg_bank
        # (every user's row holds the same taps)
        taps = torch.zeros(n, S * gdn_copies, CONV_K * blk)
        for chip in range(n):
            for i in range(S):
                bank, j = divmod(i, PER_BANK)
                cnt, first = nqg_split[j]
                for c, wg in enumerate(w["gdn"]):
                    row = i * gdn_copies + c
                    tp_ = wg["chips"][chip]["taps"]
                    for t in range(cnt):
                        g = bank * qg_bank + first + t
                        if g >= d.conv_tiles:
                            continue
                        cols = slice(g * TILE, (g + 1) * TILE)
                        e = t * TILE + torch.arange(TILE)
                        for tap in range(CONV_K):
                            for u in range(B):
                                taps[chip, row, tap * blk + tiny_index(e, u, B)] = tp_[cols, tap]
        taps_t = dram_t(taps.reshape(n * S * gdn_copies, -1), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT)
        self._sts = sts
        hist = self._hist_host(sts)
        self.hist_t = dram_t(hist, ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT)
        # host copies of the recurrent state's initial contents (reset() restores them; not the KV caches, see reset)
        self._initial = [(ttnn.from_torch(hist, dtype=ttnn.bfloat16, mesh_mapper=per_chip), self.hist_t)]

        # ---- mixer rows (GDN heads, attention leader and tail core)
        row_w = max(d.g_cols, d.a_cols)
        # (row groups, E4.2: a CB on the mixer cores instead, kernels/active_rows.hpp kCbRows)
        rows_t = l1(torch.zeros((len(heads) + 2) * LR, row_w), mixer_grid, [LR, row_w]) if groups == 1 else None
        rows_addr = rows_t.buffer_address() if rows_t is not None else 0

        # ---- GDN heads: state [copy][head][16] fp32 tiles (interleaved: a state's tiles spread over the
        # banks), norm weight [copy][4] row-0 tiles
        norm = torch.zeros(n, gdn_copies * 4 * TILE, TILE)
        for chip in range(n):
            for c in range(gdn_copies):
                for vt in range(4):
                    if w["gdn"]:
                        norm[chip, (c * 4 + vt) * TILE] = w["gdn"][c]["norm_w"][vt * TILE : (vt + 1) * TILE]
        state = self._state_host(sts)
        self.state_t = dram_t(state, ttnn.float32)
        self._initial.append(
            (ttnn.from_torch(state, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, mesh_mapper=per_chip), self.state_t)
        )
        norm_t = dram_t(norm.reshape(-1, TILE), ttnn.bfloat16)

        # ---- attention: q / M buffers, norm weights, KV caches (position tile t -> bank t % banks)
        Dt, heads_a = d.Dt, d.h
        q_t = l1(torch.zeros(WORKERS * TILE, Dt * TILE), work_grid, [TILE, Dt * TILE], tile=full)
        M_t = l1(torch.zeros((WORKERS + 1) * TILE, TILE), attn_grid, [TILE, TILE], tile=full)
        # attention of a verify of R > 2 rows: passes of P rows (P * heads <= 32, P dividing R; attn_common.hpp)
        self.attn_group = (
            max(p for p in range(1, R + 1) if R % p == 0 and p * d.h <= TILE) if R > 2 and ATTN_PACKED else 1
        )
        packed = self.attn_group > 1
        attn_users = B // self.attn_group
        slot_words = 512 if packed and self.attn_group * d.h > 16 else 128  # a worker's row-max slot (bf16)
        m_slots_t = l1(
            torch.zeros(1, WORKERS * slot_words), lead_grid, [1, WORKERS * slot_words], layout=ttnn.ROW_MAJOR_LAYOUT
        )
        one_t = l1(torch.ones(WORKERS * TILE, TILE), work_grid, [TILE, TILE], tile=full)
        # (one tile per core of prep_grid: the leader and the tail core(s))
        mean_t = l1(torch.full(((1 + self.attn_tails) * TILE, TILE), 1.0 / HD), prep_grid, [TILE, TILE], tile=full)
        rows32 = lambda v: v[None].expand(TILE, -1)
        qw = torch.cat([rows32(wa["wq"] / math.sqrt(HD)) for wa in w["attn"]] or [torch.zeros(TILE, HD)], dim=1)
        kw = torch.cat([rows32(wa["wk"]) for wa in w["attn"]] or [torch.zeros(TILE, HD)], dim=1)
        # as [copies * Dt] tiles in order: a (TILE, copies*HD) row-major tile order is copy-major
        qw_t = dram_t(qw.reshape(TILE, -1), ttnn.bfloat16, mapper=rep)
        kw_t = dram_t(kw.reshape(TILE, -1), ttnn.bfloat16, mapper=rep)
        max_tp = sts[0].max_pos // TILE
        rpb = (max_tp + banks - 1) // banks
        self.rpb = rpb
        kv_stride = rpb * KV_ROW

        # (the rows of a verify step of R > 2 rows share one user's KV cache: one user's, not R)
        kv_users = 1 if R > 2 else B
        self._kv_mc = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.DRAM,
            ttnn.ShardSpec(dram_grid, [kv_users * attn_copies * rpb * TILE, HD], ttnn.ShardOrientation.ROW_MAJOR),
        )

        def cache(which, sts_=sts):
            if sts_[0].K is None:
                # no host images: zero-filled on device (stale rows are only ever read masked, but must be finite)
                return device_zeros(
                    mesh, [banks * kv_users * attn_copies * rpb * TILE, HD], ttnn.bfloat8_b, self._kv_mc
                )
            return ttnn.from_torch(
                self._kv_host(sts_[:kv_users], which),
                dtype=ttnn.bfloat8_b,
                layout=ttnn.TILE_LAYOUT,
                device=mesh,
                memory_config=self._kv_mc,
                mesh_mapper=per_chip,
            )

        self.kv_paged = kv_pool_blocks is not None
        if self.kv_paged:  # the pools stand in for the caches, addressed through the page lists (no stride)
            kv_stride = 0
            if self.vrows:  # (a table per user; every row of the user reads and writes through it)
                bpu = -(-sts_all[0].max_pos // kv_block_size)
                ut = block_tables or [range(g * bpu, (g + 1) * bpu) for g in range(groups)]
                assert len(ut) == groups, "verify rows: one block table per user (row group)"
                self._kv_pool_setup(sts_all, kv_pool_blocks, kv_block_size, [ut[r // R] for r in range(LR)])
            else:
                self._kv_pool_setup(sts_all[:1] if verify else sts_all, kv_pool_blocks, kv_block_size, block_tables)
            if packed:  # the tail core's page list holds up to two tail tiles
                self.kv_max_n = max(self.kv_max_n, 2)
        else:
            self.K_t, self.V_t = settled(cache("K")), settled(cache("V"))
        # a chunk worker streams its KV tiles in blocks of kv_block (L1 independent of the context length)
        nck = WORKERS - self.attn_tails
        kv_block = min(KV_BLOCK, max(chunk_of(wi, max_tp, banks, nck)[3] for wi in range(nck)) or 1)
        kv_block = max(kv_block, 2 if packed else 1)  # (the tail core scores up to two tail tiles at once)
        nodes = list(range(WORKERS))
        groups_ = [
            nodes[i : i + FANIN] for i in range(0, WORKERS, FANIN)
        ]  # (children: consecutive, attn_worker_reader)
        slots_per_node = max(FANIN - 1, len(groups_))

        # ---- token state: ring step, positions, x0, rope tables of the positions
        # (row groups: group gr's x0 at TOK_X0 + gr * Ht tiles, row r's rope at rope_off + r * 3 tiles, and the rows'
        # positions at tok_pos_off, 64 B aligned, one word per row)
        self.rope_off = TOK_X0 + (2 if self.fc else 1) * d.Ht * 64 * LR
        self.tok_words = (self.rope_off + 3 * 2048 * LR) // 4
        if groups > 1:
            self.tok_pos_off = -(-(self.rope_off + 3 * 2048 * LR) // 64) * 64
            self.tok_words = (self.tok_pos_off + -(-4 * LR // 64) * 64) // 4
        # verify rows: per row its user's ring step, GDN slot and prev at tok_ver_off + 128 * k (k = 0, 1, 2; a word per
        # row, active_rows.hpp)
        self.tok_ver_off = 0
        if self.vrows:
            self.tok_ver_off = self.tok_words * 4
            self.tok_words += 3 * 32
        if tok is None:
            tok = dram_t(torch.zeros(1, self.tok_words, dtype=torch.int32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, rep)
        assert tok.shape[-1] == self.tok_words
        self.tok_t = tok

        # ---- hub
        # the hub's slots are a hub CB; it dumps the last layer's x here (for the host)
        # (row groups: one page per group)
        self.x_t = dram_t(torch.zeros(n * groups, B * HIDDEN), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT)
        self.ccl_sems = [ttnn.create_global_semaphore(mesh, hub_grid, 0) for _ in range(2)]  # per slot parity
        # the hubs' all-reduce over RESIDENT_HUB_LINKS fabric links per neighbour where that many trained, else fewer
        # (a cabling or mesh descriptor with one link per neighbour takes the one-link path)
        self.hub_links = 1
        if n > 1 and HUB_LINKS > 1:
            node = lambda c: mesh.get_fabric_node_id(ttnn.MeshCoordinate(0, c))  # noqa: E731
            trained = min(
                len(ttnn.get_forwarding_link_indices(node(c), node(c + s)))
                for c in range(n)
                for s in (1, -1)
                if 0 <= c + s < n
            )
            self.hub_links = max(1, min(HUB_LINKS, trained))
        logger.info(f"resident hub all-reduce: {self.hub_links} fabric link(s) per neighbour")
        # per streamer (order key, vocab index) of its largest logit, written by the streamer writers
        # (sample: its top-k list, sampling.hpp)
        rec_words = S * LR * (2 * TOP_K if self.sample else 4)  # (sampling: B = 1, or rows mode)
        # (row groups: the records live in the hub's parity-0 slots instead, mlp_hub.cpp; this L1 tensor would take its
        # range on every core: 262 KB at 32 sampled rows)
        self.argmax_t = (
            l1(
                torch.zeros(1, rec_words, dtype=torch.int32),
                hub_grid,
                [1, rec_words],
                ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
            )
            if lm_head and groups == 1
            else None
        )
        # feed: the hubs' candidate slots, their semaphore and launch counters, the rope table (a compact page per
        # position, rope_compact; the hub expands it into the token state's tiles) and the (token, position) out
        # word
        self.feed = embed is not None
        if self.feed:
            assert (
                lm_head and (B == 1 or verify or rows) and (n_sub == 1 or rows)
            ), "the hub feeds batch 1, verify or rows"
            assert (feed_peer is not None) == (verify and not rows)  # (verify rows: the rows-mode feed)
            self.embed, self.feed_peer = embed, feed_peer
            fed_rows = LR if rows else 2 if verify else 1  # tokens per launch (mlp_hub.cpp kCandWords)
            self._cand_words = 2 * TOP_K * fed_rows if self.sample else max(4, 2 * fed_rows)  # per chip
            cand_words = n * self._cand_words
            self.cand_t = l1(
                torch.zeros(1, cand_words, dtype=torch.int32),
                hub_grid,
                [1, cand_words],
                ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
            )
            # candidates, launches (row groups: then each parity's CCL count at the last launch's end, mlp_hub.cpp)
            # (and each parity's slot-free count of the pipelined sub-rounds)
            self.feed_sems = [ttnn.create_global_semaphore(mesh, hub_grid, 0) for _ in range(2 if groups == 1 else 6)]
            rope = rope_compact(sts[0].max_pos).view(np.int32)
            self.rope_t = dram_t(torch.from_numpy(rope.copy()), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, rep)
            # (rows mode: per launch, 16 B of out and 16 parameter words per row)
            out_words, par_words = (4 * LR, 16 * LR) if rows else (16, 16)
            self._feed_io = lambda: (
                dram_t(torch.zeros(1, out_words, dtype=torch.int32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, rep),
                dram_t(torch.zeros(1, par_words, dtype=torch.int32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, rep),
            )
            self.out_t, self.params_t = self._feed_io()

        # ---- optional timelines
        def ts_buf(grid_, count, words):
            return ttnn.from_torch(
                torch.zeros(count, words, dtype=torch.int32),
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=mesh,
                mesh_mapper=rep,
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                    ttnn.BufferType.L1,
                    ttnn.ShardSpec(grid_, [1, words], ttnn.ShardOrientation.ROW_MAJOR),
                ),
            )

        self.ts_t = ts_buf(grid, S, layers * 32) if timeline else None
        ts_addr = self.ts_t.buffer_address() if timeline else 0
        # the hub's: per round (per sub-round with row groups) 8 words (mlp_hub.cpp kHubTs)
        self.hub_ts_t = ts_buf(hub_grid, 1, (2 * layers + int(self.fc)) * self.groups * 8) if timeline else None
        self.l1_addrs += [ts_addr, self.hub_ts_t.buffer_address()] if timeline else []
        # (debug: the attention leader's timeline per (layer, user) iteration, row groups)
        self.attn_ts_t = ts_buf(lead_grid, 1, max(1, self.n_attn) * LR * 8) if ATTN_TS and groups > 1 else None
        self.gdn_ts_t = ts_buf(head_grid, len(heads), 64 * 16) if GDN_TS and not self.gdn_chunk else None
        # (and the head readers' / writers' per-layer words: [0] rows in, [1] inputs pushed, [2] first o, [3] output out)
        self.gdn_io_ts_t = ts_buf(head_grid, len(heads), max(1, self.n_gdn) * 4) if self.gdn_ts_t is not None else None
        if self.gdn_ts_t is not None:
            self.l1_addrs += [self.gdn_ts_t.buffer_address(), self.gdn_io_ts_t.buffer_address()]
        if self.attn_ts_t is not None:
            self.l1_addrs.append(self.attn_ts_t.buffer_address())

        # ---- per sub-batch decode state (sub-batch 0's is above)
        subs = [
            dict(
                hist=self.hist_t,
                state=self.state_t,
                K=self.K_t,
                V=self.V_t,
                tok=self.tok_t,
                x=self.x_t,
                argmax=self.argmax_t,
                logits=self.logits_t,
                out=getattr(self, "out_t", None),
                params=getattr(self, "params_t", None),
            )
        ]
        # paged KV: a page table per launch (rows = its users), apart from the token state (set_page_table)
        pt_zeros = lambda: dram_t(
            torch.zeros(LR, self.pt_cols, dtype=torch.int32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, rep
        )
        if self.kv_paged:
            subs[0]["pt"] = pt_zeros()
        for k in range(1, n_sub):
            sk = sts_all[k * B : (k + 1) * B]
            hist_k, state_k = self._hist_host(sk), self._state_host(sk)
            sub = dict(
                hist=dram_t(hist_k, ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT),
                state=dram_t(state_k, ttnn.float32),
                K=self.K_t if self.kv_paged else settled(cache("K", sk)),
                V=self.V_t if self.kv_paged else settled(cache("V", sk)),
                pt=pt_zeros() if self.kv_paged else None,
                tok=dram_t(torch.zeros(1, self.tok_words, dtype=torch.int32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, rep),
                x=dram_t(torch.zeros(n, B * HIDDEN), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT),
                argmax=(
                    l1(
                        torch.zeros(1, rec_words, dtype=torch.int32),
                        hub_grid,
                        [1, rec_words],
                        ttnn.uint32,
                        layout=ttnn.ROW_MAJOR_LAYOUT,
                    )
                    if lm_head
                    else None
                ),
                logits=l1(torch.zeros(S * B, nv_max * TILE), grid, [B, nv_max * TILE]) if lm_head else None,
            )
            if self.feed:
                sub["out"], sub["params"] = self._feed_io()
            self._initial.append((ttnn.from_torch(hist_k, dtype=ttnn.bfloat16, mesh_mapper=per_chip), sub["hist"]))
            self._initial.append(
                (
                    ttnn.from_torch(state_k, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, mesh_mapper=per_chip),
                    sub["state"],
                )
            )
            subs.append(sub)

        def build(sub):
            """the CBs and the per-chip programs of one launch over sub-batch `sub`'s decode state"""
            nonlocal ring_bytes  # (row groups: sized from the L1 left, below)
            # ---- CBs
            full_page = TILE * TILE * 2
            f32 = (ttnn.float32, 4096)
            kv_page = TILE_BYTES[ttnn.bfloat8_b]

            def cb(idx, grid_, pages, dt=ttnn.bfloat16, page=full_page, tiny_tile=False):
                fd = ttnn.CBFormatDescriptor(buffer_index=idx, data_format=dt, page_size=page)
                if tiny_tile:
                    fd = ttnn.CBFormatDescriptor(
                        buffer_index=idx, data_format=dt, page_size=page, tile=ttnn.TileDescriptor(B, TILE)
                    )
                return ttnn.CBDescriptor(total_size=pages * page, core_ranges=grid_, format_descriptors=[fd])

            def tiny_cb(idx, pages, grid_=grid):
                return cb(idx, grid_, pages, page=64 * B, tiny_tile=True)

            def aliased(tiny_idx, full_idx, tiles):
                return ttnn.CBDescriptor(
                    total_size=tiles * 64 * B,
                    core_ranges=grid,
                    format_descriptors=[
                        ttnn.CBFormatDescriptor(
                            buffer_index=tiny_idx,
                            data_format=ttnn.bfloat16,
                            page_size=64 * B,
                            tile=ttnn.TileDescriptor(B, TILE),
                        ),
                        ttnn.CBFormatDescriptor(buffer_index=full_idx, data_format=ttnn.bfloat16, page_size=full_page),
                    ],
                )

            def aliased_n(tiny_idxs, full_idxs, tiles):
                """one memory of `tiles` batch x 32 tiles under several CB indices (tiny and full-tile views)"""
                fds = [
                    ttnn.CBFormatDescriptor(
                        buffer_index=i, data_format=ttnn.bfloat16, page_size=64 * B, tile=ttnn.TileDescriptor(B, TILE)
                    )
                    for i in tiny_idxs
                ] + [
                    ttnn.CBFormatDescriptor(buffer_index=i, data_format=ttnn.bfloat16, page_size=full_page)
                    for i in full_idxs
                ]
                return ttnn.CBDescriptor(total_size=tiles * 64 * B, core_ranges=grid, format_descriptors=fds)

            T = ttnn.cb_descriptor_from_sharded_tensor

            def ring_cb(size):
                return ttnn.CBDescriptor(
                    total_size=size,
                    core_ranges=grid,
                    format_descriptors=[
                        ttnn.CBFormatDescriptor(buffer_index=1 + e, data_format=dt, page_size=TILE_BYTES[dt])
                        for e, (_, dt, _) in enumerate(entries)
                    ],
                )

            cbs = [
                ring_cb(max(ring_bytes, lcm)),  # (row groups: resized from the L1 left below; the estimate may be < 0)
                cb(7, grid, 1, ttnn.uint32, 16),
                *(
                    [cb(8, grid, d.It, page=64 * B, tiny_tile=True)]
                    if groups == 1
                    else [  # (row groups: one memory for the activation (8) and o (21), streamer_common.hpp)
                        ttnn.CBDescriptor(
                            total_size=groups * max(d.It, d.Ot) * 64 * B,
                            core_ranges=grid,
                            format_descriptors=[
                                ttnn.CBFormatDescriptor(
                                    buffer_index=i,
                                    data_format=ttnn.bfloat16,
                                    page_size=64 * B,
                                    tile=ttnn.TileDescriptor(B, TILE),
                                )
                                for i in (8, 21)
                            ],
                        )
                    ]
                ),
                tiny_cb(9, groups * nd_max),
                tiny_cb(10, groups * nd_max),
                *(
                    [aliased(11, 12, d.Ht), aliased(13, 14, d.Ht)]
                    if groups == 1
                    # row groups (E4.1, streamer_common.hpp kHInPlace): h is written over x, one memory for both
                    else [aliased_n((11, 13), (12, 14), groups * d.Ht), tiny_cb(39, groups * nd_max)]
                ),
                aliased(15, 27, groups * blk_tiles),
                aliased(26, 28, groups * blk_tiles),
                aliased(16, 29, groups * blk_tiles),
                aliased(17, 22, groups * blk_tiles),
                aliased(18, 23, groups * blk_tiles),
                aliased(19, 24, CONV_K * blk_tiles),
                aliased(20, 25, groups * (CONV_K - 1) * blk_tiles),
                *([T(21, self.o_in_t)] if groups == 1 else []),
                cb(30, grid, groups * (3 if R == 2 or (self.rows and not self.vrows) else 1) * blk_tiles * B // TILE),
            ]
            if self.verify:
                cbs.append(cb(36, grid, 3 if R == 2 else n_shift))
            # hub: slots (2 parities x chips) and their sum x (32x32 views), fabric packet headers (2 parities x
            # packets x 2 directions) and a word
            hub_headers = 4 * -(-HIDDEN * 2 * B // packet_bytes)
            x_views = HIDDEN * B // (TILE * TILE)
            cbs.append(cb(0, hub_grid, 2 * n * x_views))
            cbs.append(cb(2, hub_grid, x_views))
            # (the feed: its candidates in fabric packets, mlp_hub.cpp cand_packets)
            cand_packets = -(-4 * self._cand_words // packet_bytes) if self.feed else 0
            free_headers = 4 if groups > 1 else 0  # (row groups: the slot-free signals, mlp_hub.cpp)
            cbs.append(cb(1, hub_grid, 1, page=(hub_headers + 2 * cand_packets + free_headers) * 160 + 16))
            if groups > 1:
                # the hub compute's sub-round count; the streamers' active-group mask (reader -> compute); the mixers'
                # row buffer (E4.2, kernels/active_rows.hpp kCbRows)
                cbs += [cb(3, hub_grid, 1, ttnn.uint32, 16), cb(37, grid, 1, ttnn.uint32, 16)]
                # (not on the tail core: it reads its k / v rows from the leader's)
                cbs.append(cb(40, head_grid.merge(lead_grid), groups * (row_w // TILE), page=64 * B, tiny_tile=True))
            if B > 1:
                cbs += [cb(32, grid, 1), cb(33, grid, 1), cb(34, grid, 2, *f32), cb(35, grid, 1, *f32), cb(38, grid, 1)]
            if lm_head:
                cbs.append(T(31, sub["logits"]))
            # GDN heads (same CB set as the per-layer benchmark)
            bf, f4 = (ttnn.bfloat16, 2048), (ttnn.float32, 4096)
            head_cb_spec = {
                0: (bf, 4),
                1: (bf, 4),
                2: (bf, 4),
                3: (bf, 4),
                4: (bf, 4),
                5: (f4, 16),
                6: (f4, 1),
                7: (f4, 1),
                8: (f4, 1),
                9: (f4, 1),
                10: (f4, 4),
                11: (f4, 4),
                12: (f4, 1),
                13: (f4, 4),
                14: (f4, 4),
                15: (f4, 1),
                16: (f4, 1),
                17: (f4, 16),
                18: (f4, 4),
                19: (f4, 1),
                20: (f4, 4),
                21: (f4, 4),
                23: (f4, 16),
                24: (f4, 16),
                25: (f4, 4),
                26: (f4, 4),
                27: (f4, 4),
                28: (bf, 4),
                30: (f4, 1),
                31: (f4, 1),
            }
            if self.gdn_chunk:
                head_cb_spec = {
                    0: (bf, 4),
                    1: (bf, 4),
                    2: (bf, 4),
                    3: (bf, 4),
                    4: (bf, 4),
                    5: (f4, 16),
                    6: (f4, 5),
                    7: (f4, 5),
                    8: (f4, 2),
                    9: ((ttnn.bfloat16, 4 * 64 * B), 1),
                    11: (f4, 1),
                    12: (f4, 1),
                    13: (f4, 4),
                    14: (f4, 4),
                    15: (bf, 5),
                    16: (f4, 1),
                    17: (f4, 16),
                    18: (f4, 16),
                    19: (f4, 4),
                    20: (f4, 4),
                    21: (f4, 4),
                    22: (f4, 2),
                    23: (f4, 4),
                    24: (f4, 4),
                    25: (f4, 2),
                    26: (bf, 2),
                    27: (f4, 4),
                    28: (f4, 2),
                    # (the single tiles' FIFO: 14 pushes per layer, so a layer's tiles never wrap: the unpacker
                    # addresses a tile at an index from the read pointer without wrapping)
                    29: (f4, 14),
                    30: (bf, 4),
                    31: (f4, 10),
                }
            if groups > 1 and not self.gdn_chunk:
                # row groups: the state CBs hold two states, so the reader fetches the next user's state (DRAM read +
                # local copy) while compute works on this one
                head_cb_spec = {**head_cb_spec, 5: (f4, 32), 17: (f4, 32)}
                # and (gdn_head_compute.cpp kBatched) the batched results for the reader (decay, beta, decay * k),
                # compute's copy of the k norms, the per-user row-0 tiles (q / k norms, decay * k), the gathered o
                head_cb_spec.update(
                    {
                        32: (f4, 1),
                        33: (f4, 1),
                        34: (f4, 4),
                        35: (f4, 4),
                        36: (f4, 4),
                        37: (f4, 4),
                        38: (f4, 4),
                        39: (f4, 4),
                    }
                )
                # (the reader runs a user ahead: two users' v; the writer's write-back of a state overlaps the next
                # user's update: two states)
                head_cb_spec.update({2: (bf, 8), 24: (f4, 32)})
            for idx, ((dt, page), cnt) in head_cb_spec.items():
                cbs.append(cb(idx, head_grid, cnt, dt, page))
            if not self.gdn_chunk:
                cbs.append(cb(22, head_grid, 1, page=groups * 4 * 64 * B))  # the stage (a block per group)
            if self.rows:
                cbs.append(cb(29, head_grid, 1, ttnn.uint32, 64))  # active_rows.hpp: the core's active users
            # attention workers (chunk workers and the tail core)
            cbs += [
                T(0, q_t),
                T(3, M_t),
                T(4, one_t),
                cb(1, work_grid, 2 * kv_block * Dt, ttnn.bfloat8_b, kv_page),
                cb(2, work_grid, 2 * kv_block * Dt, ttnn.bfloat8_b, kv_page),
                cb(5, work_grid, kv_block, *f32),
                cb(6, work_grid, kv_block),
                cb(9, chunk_grid, 2 * Dt, *f32),
                cb(15, chunk_grid, 2, *f32),
                cb(24, chunk_grid, 2, *f32),
                cb(26, chunk_grid, 1, *f32),
                cb(7, work_grid, 1),
                cb(8, work_grid, Dt + 1),
                # (row groups: up to 32 users' words, the -inf rows at 256 and the positions at 512)
                # (rows mode: + the group heads' per-user child masks, attn_common.hpp kChildMaskOff)
                cb(10, work_grid, 1, ttnn.uint32, 1088 if packed or groups > 1 else (640 if self.rows else 256)),
                *(
                    [cb(23, attn_grid, slots_per_node * (Dt + 1))]
                    # (verify rows: unpipelined, kLeadPipelined off — the workers write the one set of slots)
                    if groups == 1 or self.vrows
                    # (row groups: the leader's slots hold two iterations, attn_common.hpp kLeadPipelined)
                    else [
                        cb(23, work_grid, slots_per_node * (Dt + 1)),
                        cb(23, lead_grid, 2 * slots_per_node * (Dt + 1)),
                    ]
                ),
                cb(25, attn_grid, Dt + 1),
                # leader and tail core
                cb(13, prep_grid, Dt),
                cb(14, prep_grid, 2),
                # (row groups: + the tail core's per-row rope tiles, attn_common.hpp kTailBatched)
                cb(
                    17,
                    prep_grid,
                    # (verify rows: per pass its q rope, per user its tail's k rope, attn_common.hpp kRopePages)
                    3 * LR + ((3 * attn_users + 3) * groups if packed else 0)
                    # (row groups: the leader's q-pass rope tiles / the tail core's per-row rope tiles)
                    + (max(3, 3 * -(-LR // (TILE // heads_a))) if groups > 1 and not self.vrows else 0),
                ),
                T(29, mean_t),
                # leader
                cb(2, lead_grid, 1, page=groups * heads_a * Dt * 64 * B),
                cb(9, lead_grid, Dt if groups == 1 else 2 * Dt),  # (row groups: q passes, attn_common.hpp)
                cb(10, lead_grid, 2 * Dt),
                cb(15, lead_grid, Dt),
                cb(24, lead_grid, 2 * Dt),
                cb(26, lead_grid, 2 * Dt),
                cb(27, lead_grid, Dt if groups == 1 else 2 * Dt),
                cb(30, lead_grid, 1),
                *([cb(7, lead_grid, 1, ttnn.uint32, 64)] if self.rows else []),  # the token state head
                *([cb(41, lead_grid, Dt)] if groups > 1 else []),  # (row groups: the staged q, attn_common.hpp)
                # (row groups: two passes' sigmoid(gate), attn_common.hpp kGatePass)
                *([cb(42, lead_grid, 2 * Dt)] if groups > 1 else []),
                # tail core
                cb(11, tail_grid, Dt),
                cb(12, tail_grid, 4 * groups if packed else 2 * LR),
                cb(16, tail_grid, Dt),
                cb(18, tail_grid, Dt),
                cb(19, tail_grid, (2 if packed else 1) * Dt),
                cb(20, tail_grid, (2 if packed else 1) * Dt),
                cb(21, tail_grid, Dt),
                cb(22, tail_grid, 2 * attn_users * groups if packed else LR),
                cb(28, tail_grid, (4 if packed else 2) * Dt, ttnn.bfloat8_b, kv_page),
                cb(31, tail_grid, (4 if packed else 2) * Dt, ttnn.bfloat8_b, kv_page),
            ]
            if self.kv_paged:
                # page lists of the workers' tiles, then a page-table window (attn_paged.hpp cb_pages)
                cbs.append(cb(30, work_grid, 1, ttnn.uint32, -(-(4 * LR * self.kv_max_n + 2048 + 64) // 64) * 64))
            if groups > 1 and RING_AUTO:
                # row groups: the ring takes the streamers' L1 left between their other CBs and the lowest L1 buffer
                # (every L1 buffer takes the same offset on every core), less a margin; never more than the one-group
                # ring, never less than the estimate
                room = (l1_low_water(mesh) or 0) - cb_bytes_on(cbs[1:], [(c.x, c.y) for c in cores]) - RING_MARGIN
                ring_bytes = max(ring_bytes, min(ring_one, (room // lcm) * lcm))
                if ring_bytes < RING_MIN:
                    raise RuntimeError(
                        f"row groups={groups}: the streamers' L1 leaves a {ring_bytes} B weight ring (< {RING_MIN} B)"
                    )
                cbs[0] = ring_cb(ring_bytes)
                self.ring_bytes = ring_bytes
                logger.info(f"row groups: weight ring {ring_bytes} B (L1 room {room} B, one-group ring {ring_one} B)")
            # the hub needs free semaphore ids for its fabric connections (2 per connection, 2 or 4 connections): the
            # head-group one only on the heads, and on the hub only the ones mlp_hub.cpp owns
            hub_sems = (SEM_GATHER, SEM_FLAG, SEM_ADDR)
            sems = [
                ttnn.SemaphoreDescriptor(id=i, core_ranges=all_grid if i in hub_sems else nohub_grid, initial_value=0)
                for i in range(SEM_GROUP)
            ]
            sems.append(ttnn.SemaphoreDescriptor(id=SEM_GROUP, core_ranges=head_grid, initial_value=0))
            if groups > 1:
                sems.append(
                    ttnn.SemaphoreDescriptor(
                        id=SEM_RADDR, core_ranges=grid.merge(prep_grid).merge(head_grid), initial_value=0
                    )
                )

            # every kernel: multi-user rows mode (active_rows.hpp)
            # (row_tiles: batch x 32 tiles per row group in the mixers' row buffer, gdn_head_reader.cpp)
            common_named = [
                ("rows_mode", int(self.rows)),
                ("gdn_layers", self.n_gdn),
                ("groups", self.groups),
                ("row_tiles", row_w // TILE),
                ("o_tiles", d.gv // TILE),  # the streamers' o_in tiles per row group (gdn_head_writer.cpp)
                ("tok_pos_off", self.tok_pos_off),  # the rows' positions with row groups (active_rows.hpp)
                ("tok_ver_off", self.tok_ver_off),  # verify rows: per-row ring / slot / prev words (active_rows.hpp)
                ("sem_raddr", SEM_RADDR),
                ("attn_tails", self.attn_tails),  # (attn_common.hpp kTails)
            ]

            def kernel(src, core_ranges, ct, rt, config, named=()):
                # (verify rows: the streamer's data-movement kernels built for size, see streamer_common.hpp — the
                # streamer core's kernels sit at the kernel config buffer's limit)
                defines = [("RESIDENT_DM_OS", "1")] if self.vrows and src in DM_OS_KERNELS else []
                return ttnn.KernelDescriptor(
                    kernel_source=KDIR + src,
                    source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                    core_ranges=core_ranges,
                    compile_time_args=ct,
                    named_compile_time_args=list(named) + common_named,
                    defines=defines,
                    runtime_args=rt,
                    config=config,
                )

            dm = lambda proc, noc: ttnn.DataMovementConfigDescriptor(processor=proc, noc=noc)
            NCRISC, BRISC = ttnn.DataMovementProcessor.RISCV_1, ttnn.DataMovementProcessor.RISCV_0
            NOC0, NOC1 = ttnn.NOC.RISCV_0_default, ttnn.NOC.RISCV_1_default

            def compute_cfg(fidelity, fp32, full_sync, unpack=None):
                c = ttnn.ComputeConfigDescriptor()
                c.math_fidelity = fidelity
                c.fp32_dest_acc_en = fp32
                c.dst_full_sync_en = full_sync
                if unpack is not None:
                    c.unpack_to_dest_mode = unpack
                return c

            streamer_compute = compute_cfg(ttnn.MathFidelity.HiFi4, True, True)
            head_unpack = type(ttnn.ComputeConfigDescriptor().unpack_to_dest_mode)([ttnn.UnpackToDestMode.Default] * 64)
            for i in CHUNK_FP32_CBS if self.gdn_chunk else HEAD_FP32_CBS:
                head_unpack[i] = ttnn.UnpackToDestMode.UnpackToDestFp32
            head_compute = compute_cfg(ttnn.MathFidelity.HiFi4, True, False, head_unpack)
            attn_compute = compute_cfg(ttnn.MathFidelity.HiFi4, True, True)
            work_compute = compute_cfg(ttnn.MathFidelity.HiFi2, True, False)

            entry_rt = []
            attn_ct = [self.n_attn, attn_copies, WORKERS, S, kv_block, heads_a, Dt, ROT // TILE]
            attn_ct += [
                SEM_ROWS,
                SEM_Q,
                SEM_m,
                SEM_M,
                SEM_PART,
                SEM_HEADS,
                f32_bits(EPS),
                SEM_ADDR,
                slots_per_node,
                banks,
                SEM_LOCAL,
                B,
                int(self.verify),
                int(ATTN_RR),
                int(ATTN_2NOC),
                self.attn_group,
            ]
            # attention worker reader / writer: paged KV (attn_paged.hpp)
            pg = self.kv_paged
            kv_named = [
                ("kv_paged", int(pg)),
                ("kv_tpb", self.kv_tpb if pg else 1),
                ("kv_max_n", self.kv_max_n if pg else 1),
                ("kv_copy_pages", self.kv_copy_pages if pg else 0),
                ("kv_pt_row_bytes", 4 * self.pt_cols if pg else 64),
            ]
            group = d.nv // d.nk
            kv_addrs = [sub["K"].buffer_address(), sub["V"].buffer_address()]
            tok_addr = sub["tok"].buffer_address()
            # GDN row runs per streamer: (first tile of its block, tiles, first head core, head cores), and the
            # row writes each head core gets per layer
            regions = []  # (first tile, end tile, first head core, head cores) of the GDN projection row
            for kh in range(d.nk):
                dests = (kh * group * G, group * G)
                regions += [(kh * 4, kh * 4 + 4, *dests), (d.gq // TILE + kh * 4, d.gq // TILE + kh * 4 + 4, *dests)]
            for hh in range(d.nv):
                dests = (hh * G, G)
                regions += [(2 * d.gq // TILE + hh * 4, 2 * d.gq // TILE + hh * 4 + 4, *dests)]
                regions += [(d.z0 // TILE + hh * 4, d.z0 // TILE + hh * 4 + 4, *dests)]
            regions.append((d.a0 // TILE, (d.b0 + d.nv - 1) // TILE + 1, 0, d.nv * G))
            runs, row_writes = [], [0] * len(heads)
            for i in range(S):
                bank, j = divmod(i, PER_BANK)
                nqg, qgf = nqg_split[j]
                b0 = bank * qg_bank + qgf
                mine = []
                for r0, r1, df, dn in regions:
                    lo, hi = max(r0, b0), min(r1, b0 + nqg)
                    if lo < hi:
                        mine.append([lo - b0, hi - lo, df, dn])
                        for hc in range(df, df + dn):
                            row_writes[hc] += 1
                runs.append(mine)
            mesh_pd = ttnn.MeshProgramDescriptor()
            for chip in range(n):
                coord = ttnn.MeshCoordinate(0, chip)
                ct = [
                    layers,
                    interval,
                    n,
                    S,
                    d.Ht,
                    d.It,
                    chip,
                    ring_bytes,
                    f32_bits(EPS),
                    f32_bits(1 / math.sqrt(HIDDEN)),
                ]
                ct += [SEM_SLOTS, SEM_ACT, SEM_GATHER, SEM_HEADS, SEM_ROWS, ng_max, nd_max, nq_max, nv_max]
                ct += [d.conv_tiles, d.Ot, d.nv * G, int(lm_head)]
                for (Kt, _, _), (sb, pages, page, block) in zip(entries, geo):
                    ct += [Kt, sb, pages, page, block]
                ct += [
                    SEM_LOCAL,
                    B,
                    SEM_ADDR,
                    d.nv,
                    blk_tiles,
                    SEM_FLAG,
                    int(self.verify),
                    int(self.fc),
                    int(self.sample),
                ]
                rr, wr, cr = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
                peers = [v for p in sphys for v in (p.x, p.y)]
                head_xy = [v for p in head_p for v in (p.x, p.y)]
                for i, c in enumerate(cores):
                    bank, j = divmod(i, PER_BANK)
                    nqg, qgf = nqg_split[j]
                    nqa, qaf = nqa_split[j]
                    nd, dfirst = nd_split[j]
                    ng, gfirst = ng_split[j]
                    nvv, vfirst = nv_split[j]
                    v0 = (bank * v_bank + vfirst) * TILE
                    counts = [nd if self.fc else nqg, nqa, nd, 2 * ng, nd, nvv]
                    ent = []
                    for cnt, (t, stride, copies) in zip(counts, self.w_entries):
                        ent += [cnt if t is not None else 0, t.buffer_address() if t is not None else 0, stride, copies]
                    rr[c.x][c.y] = (
                        [bank, i & 0x3, j, PER_BANK]
                        + ent
                        + [
                            taps_t.buffer_address(),
                            sub["hist"].buffer_address(),
                            i * gdn_copies,
                            gdn_copies,
                            tok_addr,
                            ts_addr,
                            max(0, min(nqg, d.conv_tiles - (bank * qg_bank + qgf))),
                        ]
                    )
                    n_conv = max(0, min(nqg, d.conv_tiles - (bank * qg_bank + qgf)))
                    wr[c.x][c.y] = (
                        [ng, bank * it_bank + gfirst, nd, bank * ht_bank + dfirst, hub_p.x, hub_p.y]
                        + [int(i == 0), 0]
                        + [nqg, bank * qg_bank + qgf, nqa, bank * qa_bank + qaf, rows_addr, n_conv]
                        + [sub["hist"].buffer_address(), i * gdn_copies, gdn_copies, tok_addr]
                        + head_xy
                        + [lead_p.x, lead_p.y, tail_p.x, tail_p.y]
                        + [v for p in tail2_p for v in (p.x, p.y)]
                        + peers
                        + [ts_addr]
                        + [(sub["argmax"].buffer_address() if groups == 1 else 1) if lm_head else 0]
                        + [max(0, min(nvv * TILE, d.vocab_chip - v0)), chip * d.vocab_chip + v0, i]
                        + [len(runs[i])]
                        + [v for r in runs[i] for v in r]
                    )
                    cr[c.x][c.y] = [ng, nd, bank * ht_bank + dfirst, nqg, nqa, n_conv, nvv, ts_addr]

                hub_args = [
                    sub["x"].buffer_address(),
                    *[ttnn.get_global_semaphore_address(s) for s in self.ccl_sems],
                    0,
                ] + rect_cover(mesh, cores)
                rounds = 2 * layers + int(self.fc)
                hub_ct = [n, chip, HIDDEN * 2 * B, packet_bytes, rounds, S, SEM_GATHER, SEM_SLOTS]
                hub_ct += [SEM_FLAG, int(lm_head), 1, 1, SEM_ADDR]
                hub_ct += [int(self.feed), sts[0].max_pos, self.tok_words * 4, self.rope_off, 3 * 2048]
                feed_kind = (2 if self.fc else 1) if self.feed and verify and not self.rows else 0
                hub_ct += [feed_kind, TOK_X0 + (1 if self.fc else 2) * d.Ht * 64 * B, HIDDEN * 2, int(self.sample)]
                # taps: the hub round holding x after the tap's decoder layer
                rounds_of = [2 * l + 1 + int(self.fc) for l, _ in self.taps]
                hub_ct += [len(self.taps)] + (rounds_of + [0xFFFFFFFF] * 5)[:5] + [d.Ht]
                hub_ct += [self.hub_links, HUB_MC_NOCS]
                if self.feed:
                    hub_args += [
                        sub["argmax"].buffer_address() if groups == 1 else 0,
                        self.cand_t.buffer_address(),
                        *[ttnn.get_global_semaphore_address(s) for s in self.feed_sems],
                        tok_addr,
                        self.embed.buffer_address(),
                        self.rope_t.buffer_address(),
                        sub["out"].buffer_address(),
                        sub["params"].buffer_address(),
                    ]
                    if feed_kind:
                        hub_args.append(self.feed_peer.buffer_address())
                hub_args += [t.buffer_address() for _, t in self.taps]
                hub_args.append(self.hub_ts_t.buffer_address() if self.hub_ts_t is not None else 0)
                hub_rt = ttnn.RuntimeArgs()
                hub_rt[hub.x][hub.y] = hub_args

                # GDN heads
                hr, hw = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
                gdn_io_ts = self.gdn_io_ts_t.buffer_address() if getattr(self, "gdn_io_ts_t", None) is not None else 0
                for hc, c in enumerate(heads):
                    hh, gi = divmod(hc, G)
                    nu = B // G
                    # (the core's rows: u0 .. u0 + nu of every group; verify rows: all of group gi, u0 = gi, nu = B)
                    u0_arg, nu_arg = (gi, B) if self.vrows else (gi * nu, nu)
                    kh = hh // group
                    a_col, b_col = d.a0 + hh, d.b0 + hh
                    gates = []
                    for wg in w["gdn"] or [None]:
                        cw = wg["chips"][chip] if wg else None
                        gates += [
                            f32_bits(float(cw["dt"][hh])) if cw else 0,
                            f32_bits(float(cw["neg_a"][hh])) if cw else 0,
                        ]
                    hr[c.x][c.y] = (
                        [
                            rows_addr,
                            sub["state"].buffer_address(),
                            norm_t.buffer_address(),
                            kh * 4,
                            d.gq // TILE + kh * 4,
                            2 * d.gq // TILE + hh * 4,
                            d.z0 // TILE + hh * 4,
                            a_col // TILE,
                            a_col % TILE,
                            b_col // TILE,
                            b_col % TILE,
                            hh,
                        ]
                        + gates
                        + [gdn_io_ts, u0_arg, nu_arg, row_writes[hc], tok_addr]
                    )
                    col = head_p[hh * G]
                    hw[c.x][c.y] = (
                        [sub["state"].buffer_address(), o_in_addr, hh]
                        + peers
                        + [gdn_io_ts, u0_arg, nu_arg, G, col.x, col.y, tok_addr]
                    )
                head_ct = [4, 4, self.n_gdn, gdn_copies, S]
                # (runtime arg 0: the debug per-step timeline, 0 = off)
                head_compute_rt = ttnn.RuntimeArgs()
                for c in heads:
                    head_compute_rt[c.x][c.y] = [self.gdn_ts_t.buffer_address() if self.gdn_ts_t is not None else 0]
                head_compute_ct = [d.nk, d.nv, 4, 4, group, f32_bits(DK**-0.5), f32_bits(1e-6), f32_bits(1e-6)]
                head_compute_ct += [f32_bits(1.0 / DV), self.n_gdn * (1 if self.gdn_chunk else B // G), R]

                # attention cores
                parent = {}
                for g_i, grp in enumerate(groups_):
                    parent[grp[0]] = (lead_p, g_i)
                    for k, w_i in enumerate(grp[1:]):
                        parent[w_i] = (wphys[grp[0]], k)
                kids = {grp[0]: grp[1:] for grp in groups_}
                ar, aw, ac = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
                for w_i, c in enumerate(workers):
                    is_tail = w_i >= WORKERS - self.attn_tails
                    role = 2 + (WORKERS - 1 - w_i) if is_tail else 1  # (kTails: 2 the last worker, 3 the one before)
                    ch = kids.get(w_i, [])
                    pp, slot = parent[w_i]
                    ar[c.x][c.y] = (
                        [role, w_i]
                        + kv_addrs
                        + [kv_stride, len(ch), 0, tok_addr, self.rope_off]
                        + ([rows_addr, kw_t.buffer_address()] if is_tail else [])
                        + ([sub["pt"].buffer_address()] if self.kv_paged else [])
                        + ([lead_p.x, lead_p.y] if is_tail and groups > 1 else [])  # (the leader's row buffer)
                    )
                    aw[c.x][c.y] = (
                        [role, w_i, lead_p.x, lead_p.y, m_slots_t.buffer_address(), pp.x, pp.y, slot, len(ch)]
                        + [v for k in ch for v in (wphys[k].x, wphys[k].y)]
                        + [0]
                        + (kv_addrs + [kv_stride, tok_addr] if is_tail else [])
                        + ([v for k in range(WORKERS - 1) for v in (wphys[k].x, wphys[k].y)] if is_tail else [])
                    )
                    ac[c.x][c.y] = [role, len(ch)]
                lr, lw, lc = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
                lr[leader.x][leader.y] = [rows_addr, len(groups_), 0, tok_addr, self.rope_off]
                lr[leader.x][leader.y].append(qw_t.buffer_address())
                w0p, w1p = wphys[0], wphys[-1]
                lw[leader.x][leader.y] = (
                    [q_t.buffer_address(), M_t.buffer_address(), m_slots_t.buffer_address(), WORKERS]
                    + [w0p.x, w0p.y, w1p.x, w1p.y, o_in_addr]
                    + peers
                    + [len(groups_)]
                    + [v for grp in groups_ for v in (wphys[grp[0]].x, wphys[grp[0]].y)]
                    + [self.attn_ts_t.buffer_address() if self.attn_ts_t is not None else 0, tok_addr]
                    + [v for p in tail2_p for v in (p.x, p.y)]  # (kTails: the second tail core)
                )
                lc[leader.x][leader.y] = [len(groups_)]

                kernels = [
                    kernel("streamer_reader.cpp", grid, ct, rr, dm(NCRISC, NOC0)),
                    kernel("streamer_writer.cpp", grid, ct, wr, dm(BRISC, NOC1)),
                    kernel("streamer_compute.cpp", grid, ct, cr, streamer_compute),
                    kernel("mlp_hub.cpp", hub_grid, hub_ct, hub_rt, dm(BRISC, NOC0)),
                    kernel(
                        "hub_compute.cpp",
                        hub_grid,
                        [n, x_views, rounds],  # (row groups: rounds x the active groups, from the hub)
                        ttnn.RuntimeArgs(),
                        streamer_compute,
                    ),
                ]
                if self.n_gdn:
                    kernels += [
                        kernel(
                            "gdn_chunk_reader.cpp" if self.gdn_chunk else "gdn_head_reader.cpp",
                            head_grid,
                            head_ct + [SEM_ROWS, d.nv, SEM_LOCAL, B, int(self.verify), SEM_ADDR],
                            hr,
                            # (row groups: the head writers' state write-backs (8 x 64 KB per core and layer at 32
                            # users) on NOC1, the readers on NOC0 — on NOC0 the write-backs ran at half the rate)
                            dm(NCRISC, NOC0 if groups > 1 and not self.gdn_chunk else NOC1),
                        ),
                        kernel(
                            "gdn_chunk_writer.cpp" if self.gdn_chunk else "gdn_head_writer.cpp",
                            head_grid,
                            head_ct + [SEM_HEADS, d.nv, SEM_LOCAL, B, SEM_GROUP, int(self.verify), SEM_ADDR],
                            hw,
                            dm(BRISC, NOC1 if groups > 1 and not self.gdn_chunk else NOC0),
                        ),
                        kernel(
                            "gdn_chunk_compute.cpp" if self.gdn_chunk else "gdn_head_compute.cpp",
                            head_grid,
                            head_compute_ct,
                            head_compute_rt,
                            head_compute,
                        ),
                    ]
                if self.n_attn:
                    kernels += [
                        kernel("attn_worker_reader.cpp", work_grid, attn_ct, ar, dm(NCRISC, NOC1), kv_named),
                        kernel("attn_worker_writer.cpp", work_grid, attn_ct, aw, dm(BRISC, NOC0), kv_named),
                        kernel("attn_worker_compute.cpp", chunk_grid, attn_ct, ac, work_compute),
                        kernel("attn_worker_compute.cpp", tail_grid, attn_ct, ac, attn_compute),
                        kernel("attn_leader_reader.cpp", lead_grid, attn_ct, lr, dm(NCRISC, NOC1)),
                        kernel("attn_leader_writer.cpp", lead_grid, attn_ct, lw, dm(BRISC, NOC0)),
                        kernel("attn_leader_compute.cpp", lead_grid, attn_ct, lc, attn_compute),
                    ]
                program = ttnn.ProgramDescriptor(kernels=kernels, semaphores=sems, cbs=cbs)
                self._cb_top = max(getattr(self, "_cb_top", 0), cb_extent(cbs))  # (TraceL1Guard)
                args = program.kernels[3].runtime_args[hub.x][hub.y]
                for link in range(self.hub_links):  # (mlp_hub.cpp links: a (fwd, bwd) set per link)
                    for nb in (chip + 1, chip - 1):
                        if 0 <= nb < n:
                            me = mesh.get_fabric_node_id(coord)
                            args.append(1)
                            args.extend(
                                ttnn.setup_fabric_connection(
                                    me, mesh.get_fabric_node_id(ttnn.MeshCoordinate(0, nb)), link, program, hub
                                )
                            )
                        else:
                            args.append(0)
                mesh_pd[ttnn.MeshCoordinateRange(coord, coord)] = program
            io = (
                [sub["x"]]
                + ([self.o_in_t] if self.o_in_t is not None else [])
                + ([rows_t] if rows_t is not None else [])
                + [q_t, M_t, m_slots_t, one_t, mean_t]
            )
            io += [taps_t, sub["hist"], sub["state"], norm_t, qw_t, kw_t, sub["K"], sub["V"], sub["tok"]]
            io += self._keep
            if lm_head:
                io += [sub["logits"]] + ([sub["argmax"]] if sub["argmax"] is not None else [])
            if self.kv_paged:
                io.append(sub["pt"])
            if self.feed:
                io += [self.cand_t, self.rope_t, sub["out"], sub["params"], self.embed]
                if self.feed_peer is not None:
                    io.append(self.feed_peer)
            if timeline:
                io += [self.ts_t, self.hub_ts_t]
            io += [t for _, t in self.taps]
            return mesh_pd, io

        built = [build(sub) for sub in subs]
        self.subs = subs
        self.mesh_pds = [m for m, _ in built]
        self.ios = [io for _, io in built]
        self.mesh_pd, self.io = built[0]
        self.nv_split, self.v_bank = nv_split, v_bank
        if self.kv_paged:
            self.set_page_table(self.block_tables)
        sems = self.ccl_sems + (self.feed_sems if self.feed else [])
        self.l1_addrs += [ttnn.get_global_semaphore_address(g) for g in sems]
        _LIVE_MODELS.add(self)

    @staticmethod
    def tok_tensor(mesh, d, batch, fc=False):
        """a zero token state tensor for a model of `batch` rows (fc: the MTP draft model)"""
        words = (TOK_X0 + (2 if fc else 1) * d.Ht * 64 * batch + 3 * 2048 * batch) // 4
        return ttnn.from_torch(
            torch.zeros(1, words, dtype=torch.int32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )

    def token_state(self, x0, pos, ring=None, slot=0, words=None, prev=None, active=None, preserve=False):
        """host tensor of the token state: x0 [HIDDEN] at pos, or per user x0 [users, HIDDEN] at positions pos
        (a list); ring is the conv ring step (default: the position, batch 1). With sub-batches, one host
        tensor per sub-batch. Verify: rows [t, draft] at [p, p + 1]; after a step at ring k and slot s the
        next one continues from row 0 at ring k + 1, slot s, or from row 1 at ring k + 2, slot s ^ 1.
        Verify of R > 2 rows: rows [t, d_1 .. d_R-1] at [p .. p + R - 1]; the next step continues from row a
        (a drafts accepted) at ring k + 1 + a, slot (s + a) % R; with the chunked GDN (gdn_chunk) it gets prev = a
        instead of a slot (None: the state is the committed one, the first step after a reset or handoff).
        Rows mode: active (per user, default all) says which users run the step (an inactive user's x0 and
        position are written as 0); preserve: the inactive users hold live state (a masked launch beside them, e.g.
        a new user's first token), see kernels/active_rows.hpp. A launch with no active user changes no slot's
        state only with preserve set (without it every row's conv history is rewritten)."""
        if self.n_sub > 1:
            x0 = x0.reshape(self.users, HIDDEN)
            B = self.batch
            act = [True] * self.users if active is None else list(active)
            return [
                self._token_state(
                    x0[k * B : (k + 1) * B],
                    pos[k * B : (k + 1) * B],
                    ring,
                    active=act[k * B : (k + 1) * B],
                    preserve=preserve,
                )
                for k in range(self.n_sub)
            ]
        if self.gdn_chunk:
            words = {**(words or {}), TOK_PREV: 0xFFFFFFFF if prev is None else int(prev)}
        return self._token_state(x0, pos, ring, slot, words, active, preserve)

    def token_state_rows(self, rows, preserve=False):
        """rows mode: the host token state of one launch from its active rows, rows = {row: (x0 [HIDDEN], pos)}
        (the launch's other rows inactive); preserve: see token_state"""
        assert self.rows and len(rows) > 0
        LR = self.launch_rows
        x0, pos = torch.zeros(LR, HIDDEN), [0] * LR
        for u, (v, p) in rows.items():
            x0[u], pos[u] = v.reshape(-1).float(), int(p)
        return self._token_state(x0, pos, None, active=[u in rows for u in range(LR)], preserve=preserve)

    def token_state_vrows(self, x0, pos, ring, slot, prev=None, active=None):
        """verify rows: the host token state of a step of every user's R rows. Per user g (row group g): x0 [R, HIDDEN]
        (x0: [groups, R, HIDDEN]), its row 0's position pos[g] (row r at pos[g] + r), its conv ring step ring[g] and
        GDN slot slot[g] (after a step at (k, s) with a drafts accepted the next continues at k + 1 + a, (s + a) % R),
        prev[g] the previous step's accepted drafts (None: none, the committed state; the chunked GDN's word, kept
        for it), active[g] (default all): an inactive user's rows do not run."""
        assert self.vrows
        U, R, LR = self.groups, self.R, self.launch_rows
        act = [True] * U if active is None else [bool(a) for a in active]
        prev = [None] * U if prev is None else list(prev)
        x0 = x0.reshape(LR, HIDDEN)
        rows_pos = [int(pos[g]) + r for g in range(U) for r in range(R)]
        words, base = {}, self.tok_ver_off // 4
        for row in range(LR):
            g = row // R
            words[base + row] = int(ring[g])
            words[base + 32 + row] = int(slot[g]) % R
            words[base + 64 + row] = 0xFFFFFFFF if prev[g] is None else int(prev[g])
        return self._token_state(x0, rows_pos, None, extra=words, active=[act[r // R] for r in range(LR)])

    def out_tensor(self, launch=0):
        """the device out tensor of a launch's fed tokens (rows mode: 4 words per row, see fed_token)"""
        return self.subs[launch]["out"]

    def row_order(self):
        """rows mode: the order to fill a launch's rows in. Core g of a GDN value head runs rows g B / G .. (g + 1) B / G
        one after another, so the i-th user goes to row (i % G) B / G + i // G: consecutive users on different head
        cores (G = 2: 0, 4, 1, 5, ...). A user's tokens are the same in any row; only the step time changes.
        Row groups: group 0's rows first, in that order, then group 1's (row gr * batch + r)."""
        B, G = self.batch, self.head_groups
        return [gr * B + (i % G) * (B // G) + i // G for gr in range(self.groups) for i in range(B)]

    def clear_slot(self, s):
        """rows mode: zero slot s's GDN states and its row of the conv histories (a request's state before its first
        token, what a prefill of no tokens hands off), every other slot untouched; its KV needs nothing (a step reads
        positions below its own only). On device: the prefill handoff kernel copies a zero buffer into the slot's
        state pages, hist_scatter writes zeros into its history row."""
        assert self.rows
        k, r = divmod(s, self.launch_rows)  # (row groups: row r = gr * batch + u of the launch)
        gr, u = divmod(r, self.batch)
        sub, d, mesh, n, B, LR = self.subs[k], self.d, self.mesh, self.d.n, self.batch, self.launch_rows
        pdir = f"{PKG_DIR}/prefill/kernels/"
        rows, cols, _ = self.hist_index()
        K, G = len(rows), self._gdn_copies
        core = ttnn.CoreCoord(0, 0)
        one_core = ttnn.CoreRangeSet([ttnn.CoreRange(core, core)])
        if not hasattr(self, "_clear"):
            dram, rep = ttnn.DRAM_MEMORY_CONFIG, ttnn.ReplicateTensorToMesh(mesh)
            self._clear = dict(
                sem=ttnn.create_global_semaphore(mesh, one_core, 0),
                state=ttnn.from_torch(
                    torch.zeros(16 * TILE, TILE),
                    dtype=ttnn.float32,
                    layout=ttnn.TILE_LAYOUT,
                    device=mesh,
                    memory_config=dram,
                    mesh_mapper=rep,
                ),
                hist=ttnn.from_torch(
                    torch.zeros(G * (CONV_K - 1) * K, TILE),
                    dtype=ttnn.bfloat16,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    device=mesh,
                    memory_config=dram,
                    mesh_mapper=rep,
                ),
            )
        zs, zh, sem = self._clear["state"], self._clear["hist"], self._clear["sem"]
        # GDN states: per (copy, head) the slot's 16 pages, chip-local kind-0 jobs (kernels/handoff.cpp)
        c, h = torch.meshgrid(torch.arange(G), torch.arange(d.nv), indexing="ij")
        jobs = torch.zeros(n, G * d.nv, 16, dtype=torch.int64)
        jobs[:, :, 1], jobs[:, :, 3], jobs[:, :, 5] = zs.buffer_address(), 4096, sub["state"].buffer_address()
        jobs[:, :, 6], jobs[:, :, 7] = (((c * d.nv + h) * LR + r) * 16).reshape(-1), 16
        jobs[:, :, 0] = torch.arange(n)[:, None]
        table = (jobs & 0xFFFFFFFF).numpy().astype(np.uint32).view(np.int32)
        jobs_t = ttnn.from_torch(
            torch.from_numpy(table.reshape(n * G * d.nv, 16).copy()),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        cb = lambda idx, total, page: ttnn.CBDescriptor(
            total_size=total,
            core_ranges=one_core,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=idx, data_format=ttnn.bfloat16, page_size=page)],
        )
        state_pd = ttnn.MeshProgramDescriptor()
        for chip in range(n):
            rt = ttnn.RuntimeArgs()
            # (then no fabric connections: every job is chip-local)
            rt[core.x][core.y] = [
                jobs_t.buffer_address(),
                G * d.nv,
                chip,
                ttnn.get_global_semaphore_address(sem),
                0,
            ] + [
                0,
                0,
            ]
            kern = ttnn.KernelDescriptor(
                kernel_source=pdir + "handoff.cpp",
                source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                core_ranges=one_core,
                compile_time_args=[],
                runtime_args=rt,
                config=ttnn.DataMovementConfigDescriptor(
                    processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.RISCV_0_default
                ),
            )
            coord = ttnn.MeshCoordinate(0, chip)
            state_pd[ttnn.MeshCoordinateRange(coord, coord)] = ttnn.ProgramDescriptor(
                kernels=[kern], semaphores=[], cbs=[cb(0, 8 * 4096, 4096), cb(1, 64 * 64, 64)]
            )
        ttnn.generic_op([jobs_t, zs, sub["state"]], state_pd)
        # conv history: zeros into row u of every (streamer, copy) row (prefill/kernels/hist_scatter.cpp)
        tiles = (cols[::TILE] // TILE).tolist()
        rt = ttnn.RuntimeArgs()
        # (row groups: group gr's history rows follow the S x copies rows of the groups before it)
        pairs = [v for i in range(K) for v in (int(rows[i]) + gr * self._S, tiles[i])]
        rt[core.x][core.y] = [zh.buffer_address(), sub["hist"].buffer_address(), 0, G] + pairs + [u]
        hist_kern = ttnn.KernelDescriptor(
            kernel_source=pdir + "hist_scatter.cpp",
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=one_core,
            compile_time_args=[K, G, B, CONV_K - 1, self._blk * TILE * B, CONV_K - 1, 1, 0],
            runtime_args=rt,
            config=ttnn.DataMovementConfigDescriptor(
                processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.RISCV_0_default
            ),
        )
        hist_pd = ttnn.MeshProgramDescriptor()
        hist_pd[
            ttnn.MeshCoordinateRange(ttnn.MeshCoordinate(0, 0), ttnn.MeshCoordinate(0, n - 1))
        ] = ttnn.ProgramDescriptor(kernels=[hist_kern], semaphores=[], cbs=[cb(0, K * 64, 64)])
        ttnn.generic_op([zh, sub["hist"]], hist_pd)
        ttnn.synchronize_device(mesh)
        ttnn.deallocate(jobs_t)

    # ---- slot snapshots (Strata S1 / S3): a slot's GDN states and conv history <-> device staging tensors
    def _page_copy(self, copies):
        """device DRAM -> DRAM page copies on every chip (prefill/kernels/handoff.cpp kind-0 jobs, chip-local):
        copies = [(src tensor, src page, dst tensor, dst page, pages[, page bytes])], pages of 4 KB (fp32 tiles) by
        default; a KV pool page (one position tile of K or V, 8 bf8 tiles) is KV_ROW B"""
        n, mesh = self.d.n, self.mesh
        core = ttnn.CoreCoord(0, 0)
        one_core = ttnn.CoreRangeSet([ttnn.CoreRange(core, core)])
        jobs = torch.zeros(n, len(copies), 16, dtype=torch.int64)
        for j, (src, sp, dst, dp, cnt, *pb) in enumerate(copies):
            jobs[:, j, 1], jobs[:, j, 2], jobs[:, j, 3] = src.buffer_address(), sp, pb[0] if pb else 4096
            jobs[:, j, 5], jobs[:, j, 6], jobs[:, j, 7] = dst.buffer_address(), dp, cnt
        jobs[:, :, 0] = torch.arange(n)[:, None]
        table = (jobs & 0xFFFFFFFF).numpy().astype(np.uint32).view(np.int32)
        jobs_t = ttnn.from_torch(
            torch.from_numpy(table.reshape(n * len(copies), 16).copy()),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        cb = lambda idx, total, page: ttnn.CBDescriptor(  # noqa: E731
            total_size=total,
            core_ranges=one_core,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=idx, data_format=ttnn.bfloat16, page_size=page)],
        )
        pd = ttnn.MeshProgramDescriptor()
        for chip in range(n):
            rt = ttnn.RuntimeArgs()
            rt[core.x][core.y] = [jobs_t.buffer_address(), len(copies), chip, self._snap_sem_addr(), 0, 0, 0]
            kern = ttnn.KernelDescriptor(
                kernel_source=f"{PKG_DIR}/prefill/kernels/handoff.cpp",
                source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                core_ranges=one_core,
                compile_time_args=[],
                runtime_args=rt,
                config=ttnn.DataMovementConfigDescriptor(
                    processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.RISCV_0_default
                ),
            )
            coord = ttnn.MeshCoordinate(0, chip)
            pd[ttnn.MeshCoordinateRange(coord, coord)] = ttnn.ProgramDescriptor(
                kernels=[kern], semaphores=[], cbs=[cb(0, 8 * 4096, 4096), cb(1, 64 * 64, 64)]
            )
        ttnn.generic_op([jobs_t] + list({id(t): t for c in copies for t in (c[0], c[2])}.values()), pd)
        ttnn.synchronize_device(mesh)
        ttnn.deallocate(jobs_t)

    def _snap_sem_addr(self):
        if not hasattr(self, "_snap_sem"):
            core = ttnn.CoreCoord(0, 0)
            self._snap_sem = ttnn.create_global_semaphore(self.mesh, ttnn.CoreRangeSet([ttnn.CoreRange(core, core)]), 0)
        return ttnn.get_global_semaphore_address(self._snap_sem)

    def _hist_row(self, s, compact, gather):
        """slot s's row of every conv history <-> the compact tensor (hist_scatter.cpp one target row; gather: history
        -> compact, else compact -> history)"""
        k, r = divmod(s, self.launch_rows)
        gr, u = divmod(r, self.batch)
        sub, n, B = self.subs[k], self.d.n, self.batch
        rows, cols, _ = self.hist_index()
        K, G = len(rows), self._gdn_copies
        core = ttnn.CoreCoord(0, 0)
        one_core = ttnn.CoreRangeSet([ttnn.CoreRange(core, core)])
        tiles = (cols[::TILE] // TILE).tolist()
        rt = ttnn.RuntimeArgs()
        pairs = [v for i in range(K) for v in (int(rows[i]) + gr * self._S, tiles[i])]
        rt[core.x][core.y] = [compact.buffer_address(), sub["hist"].buffer_address(), 0, G] + pairs + [u]
        kern = ttnn.KernelDescriptor(
            kernel_source=f"{PKG_DIR}/prefill/kernels/hist_scatter.cpp",
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=one_core,
            compile_time_args=[K, G, B, CONV_K - 1, self._blk * TILE * B, CONV_K - 1, 1, int(gather)],
            runtime_args=rt,
            config=ttnn.DataMovementConfigDescriptor(
                processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.RISCV_0_default
            ),
        )
        cbd = ttnn.CBDescriptor(
            total_size=K * 64 + 64 * B,
            core_ranges=one_core,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=0, data_format=ttnn.bfloat16, page_size=64)],
        )
        pd = ttnn.MeshProgramDescriptor()
        pd[ttnn.MeshCoordinateRange(ttnn.MeshCoordinate(0, 0), ttnn.MeshCoordinate(0, n - 1))] = ttnn.ProgramDescriptor(
            kernels=[kern], semaphores=[], cbs=[cbd]
        )
        ttnn.generic_op([compact, sub["hist"]], pd)
        ttnn.synchronize_device(self.mesh)

    def _kv_page_runs(self, table, n_tiles):
        """(pool page, tile, run length) runs of the first n_tiles position tiles through a block table (paged KV;
        a block's tiles are consecutive pages)"""
        tpb, runs = self.kv_tpb, []
        for t0 in range(0, n_tiles, tpb):
            runs.append((int(table[t0 // tpb]) * tpb, t0, min(tpb, n_tiles - t0)))
        return runs

    def _kv_copies(self, table, n_tiles, staging, save):
        """page copies between the pools' pages of the first n_tiles tiles (every attention copy, K then V) and the
        staging tensor (page (kv * copies + c) * n_tiles + t)"""
        out, C = [], self._attn_copies
        for kv, pool in enumerate((self.K_t, self.V_t)):
            for c in range(C):
                for page, t, cnt in self._kv_page_runs(table, n_tiles):
                    pp, sp = c * self.kv_copy_pages + page, (kv * C + c) * n_tiles + t
                    out.append((pool, pp, staging, sp, cnt, KV_ROW) if save else (staging, sp, pool, pp, cnt, KV_ROW))
        return out

    def snapshot_slot(self, s, n_tokens=0):
        """rows mode: slot s's recurrent state as device staging tensors {state: its GDN states (per chip copies x
        heads x 16 fp32 tiles), hist: its conv history row (per chip copies x 3 x K compact 64 B pages)} and, with
        n_tokens (paged KV), {kv: its K / V pages of positions < n_tokens (per chip 2 x attention copies x tiles pages
        of KV_ROW B)} — restore_slot puts them into any slot (a prefix cache entry / a swapped-out user, Strata
        S1 / S3)"""
        assert self.rows
        d, mesh, G = self.d, self.mesh, self._gdn_copies
        k, r = divmod(s, self.launch_rows)
        K = len(self.hist_index()[0])
        shard = ttnn.ShardTensorToMesh(mesh, dim=0)
        state = ttnn.from_torch(
            torch.zeros(d.n * G * d.nv * 16 * TILE, TILE),
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=shard,
        )
        hist = ttnn.from_torch(
            torch.zeros(d.n * G * (CONV_K - 1) * K, TILE),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=shard,
        )
        LR, sub = self.launch_rows, self.subs[k]
        self._page_copy(
            [
                (sub["state"], ((c * d.nv + h) * LR + r) * 16, state, (c * d.nv + h) * 16, 16)
                for c in range(G)
                for h in range(d.nv)
            ]
        )
        self._hist_row(s, hist, gather=True)
        snap = dict(state=state, hist=hist, n_tokens=0)
        if n_tokens:
            assert self.kv_paged
            n_tiles = -(-n_tokens // TILE)
            kv = ttnn.from_torch(
                torch.zeros(d.n * 2 * self._attn_copies * n_tiles, KV_ROW // 4, dtype=torch.int32),
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=mesh,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=shard,
            )
            self._page_copy(self._kv_copies(self.block_tables[s], n_tiles, kv, save=True))
            snap.update(kv=kv, n_tokens=n_tokens)
        return snap

    def restore_slot(self, s, snap):
        """rows mode: a snapshot_slot result into slot s (its GDN states and conv history row, and the snapshot's KV
        pages into slot s's own blocks — set its block table first; other slots untouched)"""
        assert self.rows
        d, G = self.d, self._gdn_copies
        k, r = divmod(s, self.launch_rows)
        LR, sub = self.launch_rows, self.subs[k]
        self._page_copy(
            [
                (snap["state"], (c * d.nv + h) * 16, sub["state"], ((c * d.nv + h) * LR + r) * 16, 16)
                for c in range(G)
                for h in range(d.nv)
            ]
        )
        self._hist_row(s, snap["hist"], gather=False)
        if snap.get("n_tokens"):
            n_tiles = -(-snap["n_tokens"] // TILE)
            self._page_copy(self._kv_copies(self.block_tables[s], n_tiles, snap["kv"], save=False))

    SNAP_TENSORS = ("state", "hist", "kv")

    def snapshot_to_host(self, snap):
        """a snapshot's staging tensors to host memory (ttnn host tensors, one shard per chip), the device copies freed
        — the host tier of a prefix cache entry / a swapped-out user (Strata S1 / S3)"""
        out = {"n_tokens": snap.get("n_tokens", 0)}
        for k in self.SNAP_TENSORS:
            if k in snap:
                out[k] = ttnn.from_device(snap[k])
                ttnn.deallocate(snap[k])
        return out

    def snapshot_from_host(self, host):
        """a snapshot_to_host result back into device staging tensors (for restore_slot)"""
        snap = {"n_tokens": host.get("n_tokens", 0)}
        for k in self.SNAP_TENSORS:
            if k in host:
                snap[k] = ttnn.to_device(host[k], self.mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return snap

    def _token_state(self, x0, pos, ring, slot=0, extra=None, active=None, preserve=False):
        B = self.launch_rows
        x0 = x0.reshape(B, -1)
        pos = [pos] if B == 1 and not isinstance(pos, (list, tuple)) else list(pos)
        assert len(pos) == B and (ring is not None or B == 1 or self.rows)
        ring = (0 if self.rows else pos[0]) if ring is None else ring
        if self.rows:
            active = [True] * B if active is None else [bool(a) for a in active]
            assert len(active) == B
            x0 = torch.where(torch.tensor(active)[:, None], x0.float(), torch.zeros(()))
            pos = [p if a else 0 for p, a in zip(pos, active)]
            extra = dict(extra or {})
            extra[TOK_ACTIVE] = sum(1 << u for u, a in enumerate(active) if a)
            extra[TOK_PRESERVE] = int(preserve)
            ring = 0
        else:
            assert active is None and not preserve, "active / preserve: rows mode"
        words = np.zeros(self.tok_words * 2, dtype=np.uint16)
        # (row groups: the positions in their region, the head's words 1 .. batch unused)
        for i, v in enumerate([ring] + (pos if self.groups == 1 else [])):
            words[2 * i], words[2 * i + 1] = v & 0xFFFF, v >> 16
        if self.groups > 1:
            p0 = self.tok_pos_off // 2
            for i, v in enumerate(pos):
                words[p0 + 2 * i], words[p0 + 2 * i + 1] = v & 0xFFFF, v >> 16
        if self.verify:
            words[2 * (TOK_SLOT if self.R == 2 else TOK_SLOT_ROWS)] = slot
        for i, v in (extra or {}).items():  # further head words (the MTP feed's, see mlp_hub.cpp)
            words[2 * i], words[2 * i + 1] = v & 0xFFFF, v >> 16
        b = self.batch  # (row groups: x0 of group gr after group gr - 1's, each in its batch x 32 tiles)
        x0_dev = torch.cat([tiny_order(x0[g * b : (g + 1) * b]) for g in range(self.groups)])
        words[TOK_X0 // 2 : TOK_X0 // 2 + x0.numel()] = bf16_bits(x0_dev)
        r0 = self.rope_off // 2
        if not hasattr(self, "_rope"):
            # every position's compact entry, built once (expanding one costs microseconds)
            self._rope = rope_compact(self._sts[0].max_pos)
        words[r0 : r0 + 3 * 1024 * B] = rope_expand(self._rope[pos]).reshape(-1)
        t = torch.from_numpy(words.view(np.int32).copy()).reshape(1, -1)
        return ttnn.from_torch(
            t, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh)
        )

    def hist_index(self):
        """(streamer row, element in the row's 32 x 32 slot, conv column of the chip's q | k | v columns) of
        every conv history element; the same on every chip (batch 1 layout)"""
        d, S = self.d, self._S
        if not hasattr(self, "_hist_map"):
            # (streamer, block tile) -> conv tile of the chip's q | k | v columns, for the conv tiles only
            rows, tiles, convs = [], [], []
            for i in range(S):
                bank, j = divmod(i, PER_BANK)
                cnt, first = self._nqg_split[j]
                for t in range(cnt):
                    g = bank * self._qg_bank + first + t
                    if g < d.conv_tiles:
                        rows.append(i)
                        tiles.append(t)
                        convs.append(g)
            e = torch.arange(TILE)
            self._hist_map = (
                torch.tensor(rows),
                (torch.tensor(tiles)[:, None] * TILE + e).reshape(-1),
                (torch.tensor(convs)[:, None] * TILE + e).reshape(-1),
            )
        return self._hist_map

    def _hist_host(self, sts, ring=None):
        """conv histories of the users' states as the hist tensor: per (streamer, GDN copy) row, 3 slots of
        32 batch x 32 tiles; the ring slot of age a (0 = oldest) at ring step s is (s * uses + a) % 3 (see
        conv_step in streamer_common.hpp; ring: by default the position at batch 1, else 0)"""
        sts = sts if isinstance(sts, (list, tuple)) else [sts]
        if self.vrows and len(sts) == self.launch_rows:
            # verify rows: group g's ring (user g's histories at its ring step, ring: per user or one for all) after
            # the S x copies rows of the groups before it
            b, n = self.batch, self.d.n
            rings = list(ring) if isinstance(ring, (list, tuple)) else [ring or 0] * self.groups
            per = [
                self._hist_host_rows(sts[g * b], rings[g]).reshape(n, self._S * self._gdn_copies, -1)
                for g in range(self.groups)
            ]
            return torch.cat(per, dim=1).reshape(n * self.groups * self._S * self._gdn_copies, -1)
        if self.R > 2:
            return self._hist_host_rows(sts[0], ring or 0)
        if self.groups > 1 and len(sts) == self.launch_rows:
            # row groups: group gr's rows (its batch x 32 tiles) after the S x copies rows of the groups before it
            b, n = self.batch, self.d.n
            per = [
                self._hist_host(sts[g * b : (g + 1) * b], ring).reshape(n, self._S * self._gdn_copies, -1)
                for g in range(self.groups)
            ]
            return torch.cat(per, dim=1).reshape(n * self.groups * self._S * self._gdn_copies, -1)
        B = len(sts)
        if ring is None:
            ring = sts[0].pos if B == 1 else 0
        d, n, S, copies, blk = self.d, self.d.n, self._S, self._gdn_copies, self._blk * TILE * B
        rows, cols, src = self.hist_index()
        hist = torch.zeros(n, S * copies, (4 if self.verify else CONV_K - 1) * blk)
        rep_rows = rows.repeat_interleave(TILE)
        if self.verify:
            # pair slots (see streamer_common.hpp): T(ring + j) = (h_j, h_j+1), row 1 of the third unused
            for c in range(min(len(sts[0].hist), copies)):
                h = torch.stack(sts[0].hist[c])  # [n, 3, conv_ch]
                for j in range(CONV_K - 1):
                    for u in range(2):
                        if j + u < CONV_K - 1:
                            at = tiny_index(cols, u, B)
                            hist[:, rep_rows * copies + c, (ring + j) % 4 * blk + at] = h[:, j + u, src]
            return hist.reshape(n * S * copies, -1)
        for u, st in enumerate(sts):
            at = tiny_index(cols, u, B)
            for c in range(min(len(st.hist), copies)):
                uses = (self.n_gdn - c + copies - 1) // copies
                h = torch.stack(st.hist[c])  # [n, 3, conv_ch]
                for age in range(CONV_K - 1):
                    slot = (ring * uses + age) % 3
                    hist[:, rep_rows * copies + c, slot * blk + at] = h[:, age, src]
        return hist.reshape(n * S * copies, -1)

    def _hist_host_rows(self, st, ring):
        """verify of R > 2 rows: st's conv histories as a ring of 16 single-row slots, 16 / R blocks of R-row tiles
        (streamer_common.hpp): history row j (the input at pos - 3 + j) in slot (ring + j) % 16, row s % R of
        block s // R"""
        n, S, copies, R = self.d.n, self._S, self._gdn_copies, self.R
        blk = self._blk * TILE * R
        rows, cols, src = self.hist_index()
        hist = torch.zeros(n, S * copies, 16 // R * blk)
        rep_rows = rows.repeat_interleave(TILE)
        for c in range(min(len(st.hist), copies)):
            h = torch.stack(st.hist[c])  # [n, 3, conv_ch]
            for j in range(CONV_K - 1):
                slot = (ring + j) % 16
                hist[:, rep_rows * copies + c, (slot // R) * blk + tiny_index(cols, slot % R, R)] = h[:, j, src]
        return hist.reshape(n * S * copies, -1)

    def _state_host(self, sts):
        """GDN states of the users: per (copy, head, user) 16 fp32 tiles, interleaved over the banks"""
        sts = sts if isinstance(sts, (list, tuple)) else [sts]
        B = len(sts)
        d, n, copies = self.d, self.d.n, self._gdn_copies
        state = torch.zeros(n, copies, d.nv, B, 4, TILE, 4, TILE)
        for u, st in enumerate(sts):
            for c in range(min(len(st.gdn), copies)):
                for chip in range(n):
                    state[chip, c, :, u] = st.gdn[c][chip].reshape(d.nv, 4, TILE, 4, TILE)
        return state.permute(0, 1, 2, 3, 4, 6, 5, 7).reshape(n * copies * d.nv * B * 16 * TILE, TILE)

    def _kv_host(self, sts, which):
        """KV caches of the users: per bank users, then copies, then rows; position tile t -> bank t % banks,
        row t // banks"""
        sts = sts if isinstance(sts, (list, tuple)) else [sts]
        B = len(sts)
        n, banks, copies, rpb = self.d.n, self.d.banks, self._attn_copies, self.rpb
        t = torch.zeros(n, banks, B, copies, rpb * TILE, HD)
        for u, st in enumerate(sts):
            caches = st.K if which == "K" else st.V
            for c in range(min(len(caches), copies)):
                for chip in range(n):
                    src = caches[c][chip]
                    pts = src.shape[0] // TILE
                    tiles = src[: pts * TILE].reshape(pts, TILE, HD)
                    for b in range(banks):
                        sel = tiles[b::banks]
                        t[chip, b, u, c, : sel.shape[0] * TILE] = sel.reshape(-1, HD)
        return t.reshape(n * banks * B * copies * rpb * TILE, HD)

    # ---- paged KV (attn_paged.hpp)
    def _kv_pool_setup(self, sts, blocks, block_size, tables):
        """the K / V pools of `blocks` blocks of block_size positions, the page-table geometry and the users'
        block tables (sts: the users owning KV; their States' caches are placed through the tables)"""
        d, banks = self.d, self.d.banks
        assert block_size % TILE == 0
        self.kv_tpb = block_size // TILE
        self.kv_blocks, self.kv_block_size = blocks, block_size
        self.kv_copy_pages = blocks * self.kv_tpb
        self._kv_pool_rows = -(-self._attn_copies * self.kv_copy_pages // banks)  # pages per bank
        max_pos = sts[0].max_pos
        self.blocks_per_user = -(-max_pos // block_size)
        self.pt_cols = -(-self.blocks_per_user // 16) * 16  # whole 64 B DRAM reads per row
        # page list entries per user: a chunk's tiles grow with the position once every bank holds some
        max_tp = max_pos // TILE
        tps = list(range(min(max_tp, 2 * banks) + 1)) + [max_tp]
        nck = WORKERS - self.attn_tails
        self.kv_max_n = max(1, max(chunk_of(w, tp, banks, nck)[3] for w in range(nck) for tp in tps))
        if tables is None:
            bpu = self.blocks_per_user
            assert blocks >= len(sts) * bpu, (blocks, len(sts), bpu)
            tables = [range(u * bpu, (u + 1) * bpu) for u in range(len(sts))]
        self.block_tables = [[int(b) for b in t] for t in tables]
        dram_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(banks - 1, 0))})
        self._pool_mc = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.DRAM,
            ttnn.ShardSpec(dram_grid, [self._kv_pool_rows * TILE, HD], ttnn.ShardOrientation.ROW_MAJOR),
        )
        pools = []
        for which in ("K", "V"):
            if sts[0].K is None:  # no host images (State kv=False): zero-filled on device
                shape = [banks * self._kv_pool_rows * TILE, HD]
                pools.append(device_zeros(self.mesh, shape, ttnn.bfloat8_b, self._pool_mc))
            else:
                pools.append(
                    ttnn.from_torch(
                        self._kv_pool_host(sts, which),
                        dtype=ttnn.bfloat8_b,
                        layout=ttnn.TILE_LAYOUT,
                        device=self.mesh,
                        memory_config=self._pool_mc,
                        mesh_mapper=ttnn.ShardTensorToMesh(self.mesh, dim=0),
                    )
                )
            ttnn.synchronize_device(self.mesh)
        # (not restored by reset(): positions at or past a step's are only read masked; stale rows are finite)
        self.K_t, self.V_t = pools

    def _kv_pool_host(self, sts, which, tables=None):
        """the pool image of the users' K or V caches: position tile t of user u's copy c at page
        c * kv_copy_pages + tables[u][t // kv_tpb] * kv_tpb + t % kv_tpb (bank page % banks, row page // banks);
        pages no user owns are zero"""
        tables = self.block_tables if tables is None else tables
        n, banks, copies, tpb = self.d.n, self.d.banks, self._attn_copies, self.kv_tpb
        img = torch.zeros(n, banks, self._kv_pool_rows, TILE, HD)
        for u, st in enumerate(sts):
            caches = st.K if which == "K" else st.V
            tbl = torch.as_tensor(tables[u], dtype=torch.long)
            for c in range(min(len(caches), copies)):
                for chip in range(n):
                    src = caches[c][chip]
                    pts = min(src.shape[0] // TILE, len(tbl) * tpb)
                    assert not src[pts * TILE :].any(), "KV rows past the user's block table"
                    t = torch.arange(pts)
                    page = c * self.kv_copy_pages + tbl[t // tpb] * tpb + t % tpb
                    img[chip, page % banks, page // banks] = src[: pts * TILE].reshape(pts, TILE, HD)
        return img.reshape(-1, HD)

    def page_table_host(self, tables):
        """host tensors (one per sub-batch) of the page tables: tables[u] lists the physical blocks of user u's
        logical blocks 0, 1, ... (blocks past a user's last position are never read; missing rows / entries 0)"""
        B, out = self.launch_rows, []
        for k in range(self.n_sub):
            pt = torch.zeros(B, self.pt_cols, dtype=torch.int32)
            for r, tbl in enumerate(tables[k * B : (k + 1) * B]):
                tbl = torch.as_tensor(list(tbl), dtype=torch.int64)
                assert len(tbl) <= self.pt_cols and bool(((tbl >= 0) & (tbl < self.kv_blocks)).all())
                pt[r, : len(tbl)] = tbl.to(torch.int32)
            out.append(
                ttnn.from_torch(
                    pt,
                    dtype=ttnn.uint32,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh),
                )
            )
        return out

    def set_page_table(self, tables):
        """write the launches' page tables (paged KV), and only them: a step reads them at its start"""
        assert self.kv_paged
        self.block_tables = [[int(b) for b in t] for t in tables]
        for host, sub in zip(self.page_table_host(self.block_tables), self.subs):
            ttnn.copy_host_to_device_tensor(host, sub["pt"])

    # ---- one user's state (rows mode): whole-tensor host round trips, for tests and tools
    def _slot_index(self, s):
        """(launch, row, element indices of the row in a history row, KV pages relative to copy 0) of slot s
        (row groups: the row of the launch, gr * batch + u; the elements are row u's of group gr's history rows)"""
        k, r = divmod(s, self.launch_rows)
        u = r % self.batch
        blk = self._blk * TILE
        e = torch.arange(blk)
        slots = 4 if self.verify else CONV_K - 1
        hist_el = torch.cat([j * blk * self.batch + tiny_index(e, u, self.batch) for j in range(slots)])
        tbl = torch.as_tensor(self.block_tables[s], dtype=torch.long)
        pages = (tbl[:, None] * self.kv_tpb + torch.arange(self.kv_tpb)).reshape(-1)
        return k, r, hist_el, pages

    def _pool_rows(self, t):
        """the pool tensor t as [n, copies, pages per copy, TILE, HD] (host)"""
        d = self.d
        host = ttnn.to_torch(t, mesh_composer=ttnn.ConcatMeshToTensor(self.mesh, dim=0)).float()
        host = host.reshape(d.n, d.banks, self._kv_pool_rows, TILE, HD)
        page = torch.arange(self._attn_copies * self.kv_copy_pages)
        return host[:, page % d.banks, page // d.banks].reshape(d.n, self._attn_copies, self.kv_copy_pages, TILE, HD)

    def read_slot(self, s):
        """slot s's state as stored: GDN states [n, copies, heads, 16 * 1024] fp32, conv histories (its row's
        elements of the q|k|v tiles, the ones the conv reads; a whole-view write-back also fills the views' other
        tiles, which nothing reads), K / V [n, copies, its pages, TILE, HD] (paged pool)"""
        assert self.rows and self.kv_paged
        k, r, _, pages = self._slot_index(s)
        sub, d, B, G = self.subs[k], self.d, self.batch, self._gdn_copies
        gr, u = divmod(r, B)
        cat = lambda t: ttnn.to_torch(t, mesh_composer=ttnn.ConcatMeshToTensor(self.mesh, dim=0)).float()
        state = cat(sub["state"]).reshape(d.n, G, d.nv, self.launch_rows, -1)[:, :, :, r]
        hrows, hcols, _ = self.hist_index()  # per conv element (streamer, element of a batch-1 slot)
        blk = self._blk * TILE * B
        el = torch.cat([j * blk + tiny_index(hcols, u, B) for j in range(CONV_K - 1)])
        rows = hrows.repeat_interleave(TILE).repeat(CONV_K - 1)
        per_chip = cat(sub["hist"]).reshape(d.n, self.groups, self._S * G, -1)[:, gr]
        hist = torch.stack([per_chip[:, rows * G + c, el] for c in range(G)], dim=1)
        return dict(
            state=state,
            hist=hist,
            K=self._pool_rows(self.K_t)[:, :, pages],
            V=self._pool_rows(self.V_t)[:, :, pages],
        )

    def write_slot(self, s, st):
        """write State st as slot s's state (GDN states, conv histories at canonical order, KV through the slot's
        block table), every other slot's left bit-identical (read-modify-write of the whole tensors)"""
        assert self.rows and self.kv_paged
        k, r, hist_el, pages = self._slot_index(s)
        sub, d, B, LR = self.subs[k], self.d, self.batch, self.launch_rows
        gr = r // B
        per_chip = ttnn.ShardTensorToMesh(self.mesh, dim=0)
        cat = lambda t: ttnn.to_torch(t, mesh_composer=ttnn.ConcatMeshToTensor(self.mesh, dim=0))
        state = cat(sub["state"]).float().reshape(d.n, self._gdn_copies, d.nv, LR, -1)
        state[:, :, :, r] = self._state_host([st]).reshape(d.n, self._gdn_copies, d.nv, 1, -1)[:, :, :, 0]
        # (row groups: group gr's history rows; hist_el are row u's elements of a group's batch x 32 tiles)
        hist = cat(sub["hist"]).float()
        rows_g = hist.reshape(d.n, self.groups, -1, hist.shape[-1])[:, gr]
        rows_g[:, :, hist_el] = self._hist_host([st] * B, ring=0).reshape(d.n, -1, hist.shape[-1])[:, :, hist_el]
        writes = [
            (
                ttnn.from_torch(
                    state.reshape(-1, TILE), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, mesh_mapper=per_chip
                ),
                sub["state"],
            ),
            (ttnn.from_torch(hist, dtype=ttnn.bfloat16, mesh_mapper=per_chip), sub["hist"]),
        ]
        n, banks = d.n, d.banks
        for which, dev in (("K", self.K_t), ("V", self.V_t)):
            pool = cat(dev).float().reshape(n, banks, self._kv_pool_rows, TILE, HD)
            new = self._kv_pool_host([st], which, tables=[self.block_tables[s]]).reshape(
                n, banks, self._kv_pool_rows, TILE, HD
            )
            for c in range(self._attn_copies):
                pg = c * self.kv_copy_pages + pages
                pool[:, pg % banks, pg // banks] = new[:, pg % banks, pg // banks]
            writes.append(
                (
                    ttnn.from_torch(
                        pool.reshape(-1, HD), dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, mesh_mapper=per_chip
                    ),
                    dev,
                )
            )
        for host, dev in writes:
            ttnn.copy_host_to_device_tensor(host, dev)
        ttnn.synchronize_device(self.mesh)

    def load_hist(self, st, ring=None):
        """write st's conv histories (at st.pos; verify of R > 2 rows: at ring step `ring`) as the decode's"""
        host = ttnn.from_torch(
            self._hist_host(st, ring), dtype=ttnn.bfloat16, mesh_mapper=ttnn.ShardTensorToMesh(self.mesh, dim=0)
        )
        ttnn.copy_host_to_device_tensor(host, self.hist_t)

    def load_state(self, st):
        """write st (GDN states, conv histories at st.pos, KV caches) as the decode state"""
        per_chip = ttnn.ShardTensorToMesh(self.mesh, dim=0)
        hosts = [
            (ttnn.from_torch(self._hist_host(st), dtype=ttnn.bfloat16, mesh_mapper=per_chip), self.hist_t),
            (
                ttnn.from_torch(
                    self._state_host(st), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, mesh_mapper=per_chip
                ),
                self.state_t,
            ),
        ]
        for which, dev in (("K", self.K_t), ("V", self.V_t)):
            # (paged: the whole pool, st's caches through user 0's block table)
            kv = self._kv_pool_host([st], which) if self.kv_paged else self._kv_host(st, which)
            hosts.append(
                (ttnn.from_torch(kv, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, mesh_mapper=per_chip), dev)
            )
        for host, dev in hosts:
            ttnn.copy_host_to_device_tensor(host, dev)
        ttnn.synchronize_device(self.mesh)

    def reset(self, state=True):
        """restore the recurrent decode state (GDN states, conv histories) to its initial contents; state=False
        only resets the feeding hubs' counters, for a caller that overwrites every state it reads.
        The KV caches are not restored: a step at position p reads the complete position tiles below p's tile
        (chunk workers, attn_common.hpp chunk_of) and p's own tile with every column past p masked to -inf
        (the tail core's cb_mask), so rows at positions >= p only ever get weight exp(-inf) = 0. Rows below the
        state's initial position are never written by a step, and a prefill handoff rewrites the prompt's. Stale
        rows must only be finite (0 * inf would poison the masked sums): device-allocated caches start zeroed and
        every later write is a computed K / V row."""
        if state:
            for host, dev in self._initial:
                ttnn.copy_host_to_device_tensor(host, dev)
            ttnn.synchronize_device(self.mesh)
        if self.feed:
            # a feeding hub counts monotonically from its launch counter; zero them all while idle
            for s in self.ccl_sems + self.feed_sems:
                ttnn.reset_global_semaphore_value(s, 0)
            ttnn.synchronize_device(self.mesh)

    def capture_trace(self):
        """record the step's launches into a trace (the device needs a trace region): step() then replays it,
        which skips the host's per-launch work (~0.5 ms); the token state is written before each replay.
        Rows mode: one trace per launch, so a step replays only the launches it runs; each launch first runs once
        eagerly (a trace records only compiled programs) on an idle token state: no row active, preserve set, which
        changes no slot's state. The launches' token states are left idle."""
        if self.rows:
            LR = self.launch_rows
            idle = self._token_state(torch.zeros(LR, HIDDEN), [0] * LR, None, active=[False] * LR, preserve=True)
            for k in range(self.n_sub):
                ttnn.copy_host_to_device_tensor(idle, self.subs[k]["tok"])
                ttnn.generic_op(self.ios[k], self.mesh_pds[k])
            ttnn.synchronize_device(self.mesh)
            if getattr(self, "feed", False):
                self.reset(state=False)  # the warm-up counted on the feeding hubs' semaphores
        groups = [[k] for k in range(self.n_sub)] if self.rows else [list(range(self.n_sub))]
        self.trace_ids = []
        for ks in groups:
            tid = ttnn.begin_trace_capture(self.mesh, cq_id=0)
            for k in ks:
                ttnn.generic_op(self.ios[k], self.mesh_pds[k])
            ttnn.end_trace_capture(self.mesh, tid, cq_id=0)
            self.trace_ids.append(tid)
        self.trace_id = self.trace_ids[0]
        ttnn.synchronize_device(self.mesh)
        self._trace_guard = TraceL1Guard(self.mesh, "decode", cb_top=self._cb_top)

    def release_trace(self):
        for tid in getattr(self, "trace_ids", None) or []:
            ttnn.release_trace(self.mesh, tid)
        self.trace_ids, self.trace_id = [], None

    def step(self, tok_host, launches=None):
        """tok_host None (feed): the token state the previous step wrote; else one host token state per launch
        (an entry None: keep that launch's). launches (rows mode): the launches to run (default all), each with
        at least one active user"""
        ks = list(range(self.n_sub)) if launches is None else list(launches)
        assert self.rows or launches is None, "a subset of the launches: rows mode"
        if tok_host is not None:
            for k, t in enumerate(tok_host if isinstance(tok_host, list) else [tok_host]):
                if t is not None:
                    ttnn.copy_host_to_device_tensor(t, self.subs[k]["tok"])
        if getattr(self, "trace_id", None) is not None:
            self._trace_guard.check()
            for tid in [self.trace_ids[k] for k in ks] if self.rows else self.trace_ids:
                ttnn.execute_trace(self.mesh, tid, cq_id=0, blocking=False)
        else:
            for k in ks:
                ttnn.generic_op(self.ios[k], self.mesh_pds[k])

    def x(self):
        """x after the step: [HIDDEN], or [users, HIDDEN]"""
        xs = []
        for sub in self.subs:
            got = ttnn.to_torch(sub["x"], mesh_composer=ttnn.ConcatMeshToTensor(self.mesh, dim=0)).float()
            xs += [tiny_unorder(got[g], self.batch) for g in range(self.groups)]  # (chip 0's page per group)
        x = torch.cat(xs)
        return x[0] if self.users == 1 else x

    def mixer_output(self):
        """the last layer's mixer output as received by streamer 0 of each chip: [n, gv], or [n, batch, gv]
        (the last sub-batch's users: the launches share the buffer)"""
        if self.o_in_t is None:
            raise NotImplementedError("row groups: o lives in the streamers' activation CB")
        o = ttnn.to_torch(self.o_in_t, mesh_composer=ttnn.ConcatMeshToTensor(self.mesh, dim=0)).float()
        o = o.reshape(self.d.n, self.S, self.launch_rows, -1)[:, 0]
        return o[:, 0] if self.launch_rows == 1 else o

    @staticmethod
    def sampling_words(temperature=0.0, top_k=1, top_p=1.0, seed=0):
        """the hub's sampling parameter words (sampling.hpp): temperature 0 or top_k 1 is greedy, top_k <= TOP_K
        (< 1: TOP_K); seed None picks one (a request without a seed, per row in rows mode)"""
        top_k = TOP_K if top_k < 1 else min(int(top_k), TOP_K)
        words = np.zeros(16, dtype=np.uint32)
        words[0] = min(round(65536 / temperature), 2**31 - 1) if temperature > 0 else 0
        words[1] = top_k
        words[2] = min(max(round(top_p * 65536), 1), 65536)
        words[3] = (int(np.random.randint(0, 2**32, dtype=np.uint64)) if seed is None else int(seed)) & 0xFFFFFFFF
        return words

    def set_sampling(self, temperature=0.0, top_k=1, top_p=1.0, seed=None, rows=None):
        """the parameters of the fed token's draw (sample), read by every step until set again; seed None picks
        one (see sampling_words). Rows mode: rows = per user (temperature, top_k, top_p, seed) (default: the
        arguments for every user), written to every launch"""
        assert self.sample
        if seed is None:
            seed = int(np.random.randint(0, 2**32, dtype=np.uint64))
        per = rows if rows is not None else [(temperature, top_k, top_p, seed)] * (self.users if self.rows else 1)
        assert len(per) == (self.users if self.rows else 1)
        for k, sub in enumerate(self.subs if self.rows else self.subs[:1]):
            chunk = per[k * self.launch_rows : (k + 1) * self.launch_rows] if self.rows else per
            words = np.concatenate([self.sampling_words(*p) for p in chunk])
            t = ttnn.from_torch(
                torch.from_numpy(words.view(np.int32).copy()).reshape(1, -1),
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh),
            )
            ttnn.copy_host_to_device_tensor(t, sub["params"])

    def fed_token(self):
        """(token, position) the last step fed into the next one (feed); rows mode: per user (token, position,
        active) of its launch's last step"""
        if self.rows:
            out = []
            for sub in self.subs:
                o = ttnn.to_torch(ttnn.get_device_tensors(sub["out"])[0]).reshape(self.launch_rows, 4)
                out += [(int(r[0]), int(r[1]), bool(r[2])) for r in o]
            return out
        out = ttnn.to_torch(ttnn.get_device_tensors(self.out_t)[0]).reshape(-1)
        return int(out[0]), int(out[1])

    def argmax(self):
        """vocab index of the largest logit (the first one on ties) from the streamers' candidates; a list
        per user for batch > 1"""
        if self.groups > 1:
            raise NotImplementedError(
                "row groups: the streamers' records live in the hub's slots (use fed_token / logits)"
            )
        out = []
        for sub in self.subs:
            raw = ttnn.to_torch(sub["argmax"], mesh_composer=ttnn.ConcatMeshToTensor(self.mesh, dim=0))
            LR = self.launch_rows
            if self.sample:  # each streamer's list starts with its largest (key, index); lists [row][streamer]
                c = raw.reshape(-1, LR, self.S, 2, TOP_K)[..., 0].transpose(1, 2)
            else:
                c = raw.reshape(-1, self.S, LR, 4)[..., :2]
            c = c.to(torch.int64) & 0xFFFFFFFF
            for u in range(LR):
                cu = c[:, :, u].reshape(-1, 2)
                best = cu[:, 0].max()
                out.append(int(cu[cu[:, 0] == best, 1].min()))
        return out[0] if self.users == 1 else out

    def logits(self, launches=None):
        """per-chip logits [n, vocab_chip] in vocab order, or [users, n, vocab_chip]; launches: only those launches'
        users ([8 * len(launches), n, vocab_chip], launch order as given)"""
        B = self.launch_rows
        if not hasattr(self, "_vocab_src"):
            # vocab column -> (streamer, column in its slice); one gather instead of a loop per streamer
            src_i = torch.full((self.d.vocab,), -1, dtype=torch.long)
            src_w = torch.zeros(self.d.vocab, dtype=torch.long)
            for i in range(self.S):
                bank, j = divmod(i, PER_BANK)
                cnt, first = self.nv_split[j]
                c0 = (bank * self.v_bank + first) * TILE
                src_i[c0 : c0 + cnt * TILE] = i
                src_w[c0 : c0 + cnt * TILE] = torch.arange(cnt * TILE)
            self._vocab_src = (src_i.clamp(min=0), src_w, src_i < 0)
        src_i, src_w, unset = self._vocab_src
        outs = []
        for sub in self.subs if launches is None else [self.subs[k] for k in launches]:
            raw = ttnn.to_torch(sub["logits"], mesh_composer=ttnn.ConcatMeshToTensor(self.mesh, dim=0)).float()
            raw = raw.reshape(self.d.n, self.S, B, -1)
            out = raw[:, src_i, :, src_w].permute(2, 1, 0)  # [vocab, n, B] -> [B, n, vocab]
            out[:, :, unset] = 0
            outs.append(out)
        out = torch.cat(outs)
        return out[0] if self.users == 1 and launches is None else out
