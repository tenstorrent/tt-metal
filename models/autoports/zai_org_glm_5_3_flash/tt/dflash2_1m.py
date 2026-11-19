# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Stage 17: the DFlash2 drafter at 1M context (doc/ttnn/17-mtp-spec-decode/dflash2_intrace.md §10). Host side:

1. SIZING (`size_drafter`, `sizing_table`): the drafter's per-token state is its OWN K / V cache (tt/dflash2_kind.py:
   the taps are consumed by fc -> hidden_norm -> k / v projections at write time and never stored), L layers x 2 x nkv
   heads x hd bf16 = 32 KiB per token for the Apache-2.0 drafter (8 layers, 8 kv heads x 128). At TP8 it is already
   head-sharded (one kv head per chip = 4 KiB per token per chip, no replication); a sequence shard over the same 8
   chips holds the same 4 KiB per token per chip. The attention is DENSE (config.json: every layer `full_attention`,
   `is_causal: false`, no sliding window; tt/dflash2_ref.py: one softmax over the whole context + the block) -- the
   per-step cost at 1M is the K / V read of the whole context per chip (4 GiB bf16 / 2.1 GiB bfp8 per step).
2. The owner map of a sequence-sharded drafter cache (`DrafterShardMap`: the target's unit rule, unit = 128 tokens,
   unit u on chip u mod C, local position (p // 128 // C) 128 + p % 128 -- the same numbers as tt/dsa_shard.py
   ShardMap on the DSA shard branch; the drafter's page is 32 tokens = 4 pages per unit) and the split-K attention
   references: `attention_partial` (per owner: fp32 (m, l, O) over its local rows for ALL heads), `merge_partials`
   (the online-softmax merge in a fixed owner order: bitwise replicas), `split_k_attention` vs `dense_attention`.
3. The LMCache unit blob of the drafter (`UnitFormat`, `pack_units`, `unpack_units`): chip- and form-independent on
   disk: per unit [L, 2 (k, v), nkv, 128 tokens, hd] bf16 (bfp8 cache values are bf16-exact), head-major, so a unit
   written from a head-sharded cache (every chip holds its head's rows) restores into a sequence-sharded one (the
   owner holds every head's rows) and the other way round.
4. `bfp8_rows` / `bfp8_dequant`: the host mirror of tt-metal's bf16 -> bfp8_b row encoder (bfp_fast_pack.cpp
   pack_row: shared exponent = the row max over 16 values, truncating shift, round-to-nearest-even to 7 bits,
   saturation; zero / denormal -> 0) used by kernels/spec/df2_bfp8.hpp's RISC encoder (GLM53_DF2_KV_DTYPE=bfp8).
"""

from __future__ import annotations

import math

import torch

from models.autoports.zai_org_glm_5_3_flash.tt.dflash2_ref import Dims

BF16, FP32 = torch.bfloat16, torch.float32
UNIT = 128  # tokens per shard unit (the target's pooled-key page = 2 vLLM blocks of 64)
PAGE = 32  # tokens per drafter cache page (the kind's block = one tile row)
PAGES_PER_UNIT = UNIT // PAGE
LMCACHE_CHUNK = 256  # tokens per LMCache chunk (2 units)
BYTES_PER_VALUE = dict(bf16=2.0, bfp8=1088 / 1024)  # bfp8_b: 1,088-B tile per 1,024 values
CONTEXTS = (65536, 131072, 1048576)
GIB = 2**30
MIB = 2**20


# ====================================================================== 1. sizing
def size_drafter(cfg: dict, *, tp: int = 8, ctx: int = 1048576, kv_dtype: str = "bf16", shared_pt: bool = True) -> dict:
    """bytes of the drafter's per-user state per chip at context `ctx` on `tp` chips, by cache form, plus the per-step
    attention cost (host arithmetic from the config; nothing measured here)"""
    d = Dims(cfg)
    bpv = BYTES_PER_VALUE[kv_dtype]
    rows = ctx + PAGE  # positions up to c + 31 are written (the kind's cap rounding)
    kv_tok = d.L * 2 * d.nkv * d.hd * bpv  # the drafter's K + V of one token, every head, every layer
    taps_tok = d.n_taps * d.H * 2  # the taps of one token (bf16) IF they were stored (they are not: consumed)
    nblk = -(-rows // PAGE)
    pt = 32 * nblk * 4 * (1 if shared_pt else d.L)  # [32, nblk] int32 page table(s)
    rope = 2 * d.hd * 2 * rows  # cos / sin tables, bf16, replicated
    forms = dict(
        head=kv_tok * rows / tp,  # today's TP layout: kv heads split over the chips (nkv % tp == 0)
        seq=kv_tok * rows / tp,  # unit u on chip u mod tp, every head on the owner
        replicated=kv_tok * rows,  # never built: every chip the whole cache
        taps_replicated=taps_tok * rows,  # the brief's fear: per-token hidden taps on every chip
    )
    # the attention per step: every chip reads its K / V of the whole context (dense attention), 8 rows x its heads
    att_bytes = kv_tok * rows / tp
    flops = 8 * d.nh * rows * d.hd * 4 / tp  # QK^T and PV, 8 block rows, per chip
    gbps = (400e9, 480e9)
    return dict(
        tp=tp, ctx=ctx, kv_dtype=kv_dtype, kv_bytes_per_token=kv_tok, kv_bytes_per_token_per_chip=kv_tok / tp,
        taps_bytes_per_token=taps_tok, nblk=nblk, page_table_bytes=pt, rope_tables_bytes=rope,
        forms=forms, attention_read_bytes_per_step=att_bytes,
        attention_ms_per_step=tuple(att_bytes / g * 1e3 for g in gbps[::-1]), attention_gflop_per_step_per_chip=flops / 1e9,
        lmcache_bytes_per_user=kv_tok * ctx,  # the host / NVMe tier holds every head of every token once
    )  # fmt: skip


def prefill_ctx_flops_per_token(cfg: dict, tap_mode: str = "mean") -> float:
    """the drafter's context path per prompt token (fc + the k / v projections of every layer), FLOPs"""
    d = Dims(cfg)
    fc_in = d.n_taps * d.H * (4 if tap_mode == "fold" else 1)
    return 2 * fc_in * d.H + d.L * 2 * (2 * d.H * d.nkv * d.hd)


def sizing_table(cfg: dict, *, tp: int = 8, contexts=CONTEXTS) -> str:
    """the markdown table of §9 (per chip, GiB / MiB; the target's sharded 1M pool for scale)"""
    d = Dims(cfg)
    out = []
    out.append(f"drafter: {d.L} layers, {d.nkv} kv heads x {d.hd}, {d.nh} q heads, H {d.H}, {d.n_taps} taps; TP{tp}")
    s0 = size_drafter(cfg, tp=tp, ctx=contexts[0])
    out.append(
        f"per token: K/V {s0['kv_bytes_per_token'] / 1024:.0f} KiB bf16 (every head, every layer) = "
        f"{s0['kv_bytes_per_token_per_chip'] / 1024:.0f} KiB per chip head-sharded OR sequence-sharded; "
        f"{size_drafter(cfg, tp=tp, ctx=contexts[0], kv_dtype='bfp8')['kv_bytes_per_token_per_chip'] / 1024:.3f} KiB bfp8; "
        f"taps if stored {s0['taps_bytes_per_token'] / 1024:.0f} KiB (never stored); cos/sin 512 B replicated"
    )
    hdr = "| context | head-sharded bf16 (today) | sequence-sharded bf16 | head/seq bfp8 | replicated bf16 (never built) | taps replicated (the fear) | cos/sin tables | page table | attention read / step | attention ms / step (480-400 GB/s) bf16 / bfp8 | LMCache bytes / user bf16 / bfp8 |"
    out.append(hdr)
    out.append("|" + "---|" * 11)
    for c in contexts:
        s, s8 = size_drafter(cfg, tp=tp, ctx=c), size_drafter(cfg, tp=tp, ctx=c, kv_dtype="bfp8")
        f = s["forms"]
        g = lambda b: f"{b / GIB:.2f} GiB" if b >= GIB else f"{b / MIB:.0f} MiB"  # noqa: E731
        out.append(
            f"| {c:,} | {g(f['head'])} | {g(f['seq'])} | {g(s8['forms']['head'])} | {g(f['replicated'])} | "
            f"{g(f['taps_replicated'])} | {g(s['rope_tables_bytes'])} | {g(s['page_table_bytes'])} | {g(s['attention_read_bytes_per_step'])} | "
            f"{s['attention_ms_per_step'][0]:.1f}-{s['attention_ms_per_step'][1]:.1f} / {s8['attention_ms_per_step'][0]:.1f}-{s8['attention_ms_per_step'][1]:.1f} | "
            f"{g(s['lmcache_bytes_per_user'])} / {g(s8['lmcache_bytes_per_user'])} |"
        )
    return "\n".join(out)


# ====================================================================== 2. the owner map and the split-K references
class DrafterShardMap:
    """positions / pages of one user's drafter cache -> (owner chip, local index) for C chips: the target's unit rule
    (served_1m_memory_plan.md: unit = 128 tokens, unit u on chip u mod C, interleaved so every length balances)"""

    def __init__(self, C: int):
        assert C >= 1, C
        self.C = int(C)

    def owner(self, p: int) -> int:
        return (p // UNIT) % self.C

    def local_pos(self, p: int) -> int:
        return (p // UNIT // self.C) * UNIT + p % UNIT

    def local_cap(self, cap: int) -> int:
        """local positions one slot of `cap` global positions needs on EVERY chip (the same on all: one program)"""
        units = -(-cap // UNIT)
        return -(-units // self.C) * UNIT

    def n_local_rows(self, p: int, r: int) -> int:
        """rows of chip r among positions 0..p (inclusive)"""
        n = p + 1
        fu, rem = divmod(n, UNIT)
        full = max(0, (fu - r + self.C - 1) // self.C)
        return full * UNIT + (rem if fu % self.C == r else 0)

    def local_bound(self, cur: int, r: int) -> int:
        """the paged SDPA's inclusive bound on chip r for the global inclusive bound `cur` (-1: no local row)"""
        return self.n_local_rows(cur, r) - 1

    def page_owner(self, b: int) -> int:  # drafter page b (32 tokens) = unit b // 4
        return (b // PAGES_PER_UNIT) % self.C

    def local_page(self, b: int) -> int:
        return (b // PAGES_PER_UNIT // self.C) * PAGES_PER_UNIT + b % PAGES_PER_UNIT

    def global_page(self, r: int, lb: int) -> int:
        return ((lb // PAGES_PER_UNIT) * self.C + r) * PAGES_PER_UNIT + lb % PAGES_PER_UNIT

    def local_rows(self, L: int, r: int) -> torch.Tensor:
        """the global positions < L that chip r owns, in local order (local index i -> global position)"""
        out = []
        units = -(-L // UNIT)
        for u in range(r, units, self.C):
            n = min(UNIT, L - u * UNIT)
            out.append(torch.arange(u * UNIT, u * UNIT + n))
        return torch.cat(out) if out else torch.zeros(0, dtype=torch.long)

    def local_pages_of(self, pos: torch.Tensor, r: int, scratch: int) -> torch.Tensor:
        """the kind's 32 write positions (global, int) -> chip r's local positions; rows another chip owns point at
        the scratch rows (local_cap + j): the cache-write kernel then lands them on the scratch page, never read"""
        pos = pos.to(torch.int64)
        out = torch.empty_like(pos)
        for j, p in enumerate(pos.tolist()):
            out[j] = self.local_pos(p) if (p >= 0 and self.owner(p) == r) else scratch + j
        return out.to(torch.int32)


def attention_partial(q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, scale: float):
    """q [R, nh, hd], K / V [S, nkv, hd] (fp32 math; GQA: head h uses kv head h // (nh / nkv)) -> the fp32 partial
    (m [R, nh], l [R, nh], O [R, nh, hd]) of the online softmax over these S rows; S == 0 -> (-inf, 0, 0)"""
    R, nh, hd = q.shape
    nkv = K.shape[1]
    g = nh // nkv
    if K.shape[0] == 0:
        return torch.full((R, nh), -float("inf")), torch.zeros(R, nh), torch.zeros(R, nh, hd)
    qf = q.float().reshape(R, nkv, g, hd)
    s = torch.einsum("rkgd,skd->rkgs", qf, K.float()) * scale  # [R, nkv, g, S]
    m = s.amax(-1)
    p = torch.exp(s - m[..., None])
    l = p.sum(-1)
    O = torch.einsum("rkgs,skd->rkgd", p, V.float())
    return m.reshape(R, nh), l.reshape(R, nh), O.reshape(R, nh, hd)


def merge_partials(parts: list) -> torch.Tensor:
    """[(m, l, O)] in a FIXED order (chip 0 .. C-1: every chip computes the same bits) -> o = sum a_c O_c / sum a_c
    l_c with a_c = exp(m_c - M), M = max m_c (the in-chip partial-tree math of kernels/mla/dsa_common.hpp)"""
    M = torch.stack([p[0] for p in parts]).amax(0)
    l, O = torch.zeros_like(parts[0][1]), torch.zeros_like(parts[0][2])
    for m_c, l_c, O_c in parts:
        a = torch.exp(m_c - M)  # exp(-inf) = 0 for a chip without rows
        a = torch.where(torch.isfinite(m_c), a, torch.zeros_like(a))
        l = l + a * l_c
        O = O + a[..., None] * O_c
    return O / l[..., None]


def dense_attention(q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, scale: float) -> torch.Tensor:
    m, l, O = attention_partial(q, K, V, scale)
    return O / l[..., None]


def split_k_attention(q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, scale: float, sm: DrafterShardMap):
    """the sequence-sharded step: chip r attends ITS rows (the owner map) with every head, the partials merged in chip
    order; K / V [S, nkv, hd] in global position order -> o [R, nh, hd] (== dense_attention up to fp32 rounding)"""
    S = K.shape[0]
    parts = []
    for r in range(sm.C):
        idx = sm.local_rows(S, r)
        parts.append(attention_partial(q, K[idx], V[idx], scale))
    return merge_partials(parts)


def seq_form_bytes_per_layer_step(cfg: dict, tp: int = 8, rows: int = 8) -> dict:
    """fabric bytes per chip per layer per step of the split-K form (the target's AG / RS structure on a 2x4 mesh:
    shard_tp8_design.md §3): AG = every chip's q piece [rows, nh / tp, hd] bf16 to every chip (round 1 the column
    peer, round 2 the row line); RS = the partial [rows, nh, hd] fp32 + (m, l) to the head owners (round 1 the other
    row's half to the column peer, round 2 the row peers' nh / tp slices)"""
    d = Dims(cfg)
    piece = rows * (d.nh // tp) * d.hd * 2
    ag_in = piece + 3 * 2 * piece if tp == 8 else piece * (tp - 1)
    partial = rows * d.nh * d.hd * 4 + 2 * rows * d.nh * 4
    rs_out = partial // 2 + 3 * (partial // tp) if tp == 8 else partial * (tp - 1) // tp
    return dict(q_piece=piece, ag_in=ag_in, partial=partial, rs_out=rs_out, rs_in=rs_out)


# ====================================================================== 3. the LMCache unit blob
class UnitFormat:
    """one unit of the drafter's context on disk / the host tier: [L, 2 (k, v), nkv, UNIT, hd] bf16, head-major:
    chip-independent (no owner, no head split in the format), dtype-independent (bfp8 values are bf16-exact)"""

    def __init__(self, cfg: dict):
        self.d = d = Dims(cfg)
        self.shape = (d.L, 2, d.nkv, UNIT, d.hd)
        self.bytes = int(math.prod(self.shape)) * 2
        self.chunk_bytes = self.bytes * (LMCACHE_CHUNK // UNIT)

    def describe(self) -> str:
        return (f"unit {UNIT} tokens = {PAGES_PER_UNIT} drafter pages: [L {self.d.L}, k|v, nkv {self.d.nkv}, {UNIT}, "
                f"hd {self.d.hd}] bf16 = {self.bytes / MIB:.2f} MiB; LMCache chunk ({LMCACHE_CHUNK}) = {self.chunk_bytes / MIB:.2f} MiB")  # fmt: skip


def pack_units(caches: list, u0: int, n_units: int, sm: DrafterShardMap | None) -> torch.Tensor:
    """caches = [(K_chip_rows, V_chip_rows)] per LAYER, where K_chip_rows = [per-chip tensors [S_local, nkv_l, hd]]
    in local position order (head-sharded: nkv_l = nkv / tp, chip r holds heads [r nkv_l, (r + 1) nkv_l) of every
    position; sequence-sharded (sm given): nkv_l = nkv, chip r holds its owned positions) -> the blob
    [n_units, L, 2, nkv, UNIT, hd] bf16 of units u0 .. u0 + n_units - 1"""
    L = len(caches)
    tp = len(caches[0][0])
    nkv_l, hd = caches[0][0][0].shape[1], caches[0][0][0].shape[2]
    nkv = nkv_l if sm is not None else nkv_l * tp
    out = torch.zeros(n_units, L, 2, nkv, UNIT, hd, dtype=BF16)
    for i in range(n_units):
        u = u0 + i
        for li, (Kr, Vr) in enumerate(caches):
            for src, kv in ((Kr, 0), (Vr, 1)):
                if sm is None:  # head-sharded: rows u UNIT .. of every chip, heads side by side
                    for r in range(tp):
                        rows = src[r][u * UNIT : (u + 1) * UNIT]
                        out[i, li, kv, r * nkv_l : (r + 1) * nkv_l] = rows.to(BF16).permute(1, 0, 2)
                else:  # sequence-sharded: the owner's local rows of the unit
                    r = u % sm.C
                    lu = u // sm.C
                    rows = src[r][lu * UNIT : (lu + 1) * UNIT]
                    out[i, li, kv] = rows.to(BF16).permute(1, 0, 2)
    return out


def unpack_units(blob: torch.Tensor, u0: int, tp: int, sm: DrafterShardMap | None) -> list:
    """the inverse of pack_units: the blob -> per layer per (k, v) per chip a list of (local_row0, rows [UNIT, nkv_l,
    hd]) writes (the caller lands them: ops.write on the host backend, paged_fill_cache on ttnn)"""
    n_units, L, _, nkv, _, hd = blob.shape
    writes = [[[[] for _ in range(tp)] for _ in range(2)] for _ in range(L)]
    nkv_l = nkv if sm is not None else nkv // tp
    for i in range(n_units):
        u = u0 + i
        for li in range(L):
            for kv in range(2):
                if sm is None:
                    for r in range(tp):
                        rows = blob[i, li, kv, r * nkv_l : (r + 1) * nkv_l].permute(1, 0, 2).contiguous()
                        writes[li][kv][r].append((u * UNIT, rows))
                else:
                    r, lu = u % sm.C, u // sm.C
                    writes[li][kv][r].append((lu * UNIT, blob[i, li, kv].permute(1, 0, 2).contiguous()))
    return writes


# ====================================================================== 4. the bfp8 row encoder (host mirror)
def bfp8_rows(x: torch.Tensor) -> tuple:
    """x [..., 16 k] bf16 -> (shared exponents [..., k] uint8, codes [..., 16 k] uint8: sign << 7 | 7-bit mantissa),
    bitwise tt-metal's host packer (bfp_fast_pack.cpp pack_row, is_exp_a = false) per 16-value group"""
    b = x.to(BF16).contiguous().view(torch.int16).to(torch.int64) & 0xFFFF
    shp = b.shape[:-1]
    b = b.reshape(*shp, -1, 16)
    e = (b >> 7) & 0xFF
    shared = e.amax(-1, keepdim=True)
    dsh = (shared - e).clamp(max=31)
    mant = ((1 << 23) | ((b & 0x7F) << 16)) >> dsh
    SHIFT, MANT_W = 17, 7
    rmask, tie, maxv = (1 << SHIFT) - 1, 1 << (SHIFT - 1), (1 << MANT_W) - 1
    rv = mant & rmask
    q = mant >> SHIFT
    q = q + ((rv > tie) | ((rv == tie) & (q & 1 == 1))).to(torch.int64)
    q = q.clamp(max=maxv)
    q = torch.where(e == 0, torch.zeros_like(q), q)
    sign = torch.where(q == 0, torch.zeros_like(q), b >> 15)
    codes = (sign << MANT_W) | q
    return shared.squeeze(-1).to(torch.uint8), codes.reshape(*shp, -1).to(torch.uint8)


def bfp8_dequant(shared: torch.Tensor, codes: torch.Tensor) -> torch.Tensor:
    """(shared [..., k], codes [..., 16 k]) -> the fp32 values the unpacker produces: sign (-1)^s x q x 2^(e - 127 - 6)"""
    e = shared.to(torch.int64).unsqueeze(-1).expand(*shared.shape, 16).reshape(*shared.shape[:-1], -1)
    c = codes.to(torch.int64)
    q, s = (c & 0x7F).float(), (c >> 7).float()
    v = torch.ldexp(q, (e - 127 - 6).float()) * (1 - 2 * s)
    return v


def bfp8_like(x: torch.Tensor) -> torch.Tensor:
    """x -> the values after a bf16 -> bfp8_b round trip along 16-value groups of the last dim (fp32)"""
    return bfp8_dequant(*bfp8_rows(x)).reshape(x.shape)


def bfp8_tile_bytes(x: torch.Tensor) -> torch.Tensor:
    """x [32, 32] bf16 -> the 1,088 bytes of its bfp8_b tile (64 exponent bytes: face f row r at f 16 + r; then 4 x
    256 mantissa bytes: face f row r col c at 64 + f 256 + r 16 + c): the layout kernels/spec/df2_bfp8.hpp patches"""
    assert tuple(x.shape) == (32, 32), x.shape
    out = torch.zeros(1088, dtype=torch.uint8)
    for f, (r0, c0) in enumerate(((0, 0), (0, 16), (16, 0), (16, 16))):
        face = x[r0 : r0 + 16, c0 : c0 + 16]
        sh, codes = bfp8_rows(face)
        out[f * 16 : (f + 1) * 16] = sh.reshape(16)
        out[64 + f * 256 : 64 + (f + 1) * 256] = codes.reshape(256)
    return out


if __name__ == "__main__":
    import argparse
    import json

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="/home/alex/glm53/work-17-d0/dflash2/drafter/config.json")
    ap.add_argument("--tp", type=int, default=8)
    a = ap.parse_args()
    cfg = json.load(open(a.config))
    print(sizing_table(cfg, tp=a.tp))
    f = prefill_ctx_flops_per_token(cfg)
    print(f"prefill ctx: {f / 1e6:.0f} MFLOP per token -> 1M: {f * 1048576 / 1e12:.0f} TFLOP total, "
          f"{f * 1048576 / a.tp / 1e12:.1f} TFLOP per chip")  # fmt: skip
    print("seq-form fabric bytes per chip per layer per step:", seq_form_bytes_per_layer_step(cfg, a.tp))
    print(UnitFormat(cfg).describe())


# ====================================================================== 5. the blob on a DFlash2Kind (host backend / ttnn)
def _cache_tensor(kind, li: int, kv: int):
    """(the cache handle, per_head): ops mode = the paged [nblk, nkv_c, 32, hd] tensor (a per-chip list on the torch
    backend); mk mode = the per-kv-head list of [nblk, 1, 32, hd] tensors"""
    c = kind.L[li]["Kc" if kv == 0 else "Vc"]
    mk = kind.layer_mode == "mk"
    return (c if mk else c["t"]), mk


def kind_cache_rows(kind, li: int, kv: int, chip: int) -> torch.Tensor:
    """chip's rows of layer li's K (kv = 0) / V cache as [positions (local order under the shard), nkv_c, hd] (host)"""
    t, mk = _cache_tensor(kind, li, kv)
    if (
        mk
    ):  # per-head [nblk, 1, 32, hd], or (769d40a2fd, the device) ONE [nkv_c nblk, 1, 32, hd]: head g's blocks at [g nblk, ..)
        hs = [kind.ops.host(h, chip) for h in t]
        nkv_c = kind.S.nkv_c
        if len(hs) == 1 and nkv_c > 1 and hs[0].shape[0] == nkv_c * kind.nblk:
            host = hs[0].reshape(nkv_c, kind.nblk, PAGE, -1).permute(1, 0, 2, 3)
        else:
            host = torch.cat(hs, 1)
    else:
        host = kind.ops.host(t, chip)
    nblk, nkv_c, _, hd = host.shape
    return host.permute(0, 2, 1, 3).reshape(nblk * PAGE, nkv_c, hd)


def kind_write_rows(kind, li: int, kv: int, chip: int, row0: int, rows: torch.Tensor) -> None:
    """rows [n PAGE, nkv_c, hd] into chip's cache of layer li at local rows row0 .. (whole pages: row0 % 32 == 0)"""
    from models.autoports.zai_org_glm_5_3_flash.tt.dflash2_kind import TorchOps

    assert row0 % PAGE == 0 and rows.shape[0] % PAGE == 0, (row0, rows.shape)
    n = rows.shape[0] // PAGE
    t, mk = _cache_tensor(kind, li, kv)
    ops = kind.ops
    if ops.kv_dtype == "bfp8":
        rows = bfp8_like(rows.to(BF16)).to(BF16)
    pages = rows.to(BF16).reshape(n, PAGE, -1, rows.shape[-1]).permute(0, 2, 1, 3)  # [n, nkv_c, 32, hd]
    if isinstance(ops, TorchOps):
        if mk:
            for h, th in enumerate(t):
                th[chip][row0 // PAGE : row0 // PAGE + n] = pages[:, h : h + 1].to(th[chip].dtype)
        else:
            t[chip][row0 // PAGE : row0 // PAGE + n] = pages.to(t[chip].dtype)
        return
    # ttnn: paged_fill_cache with a per-chip input (the other chips write their own rows or the scratch page)
    ttnn = ops.ttnn
    nblk = kind.nblk  # per head (a merged mk cache holds nkv_c x nblk blocks)
    blocks = [row0 // PAGE + i for i in range(n)]
    tabs, parts = [], []
    for r in range(ops.tp):
        if r == chip:
            tabs.append(torch.tensor([blocks], dtype=torch.int32))
            parts.append(pages.permute(1, 0, 2, 3).reshape(1, pages.shape[1], n * PAGE, -1))
        else:  # the scratch page (never read): block nblk - 1, zeros
            tabs.append(torch.tensor([[nblk - 1] * n], dtype=torch.int32))
            parts.append(torch.zeros(1, pages.shape[1], n * PAGE, pages.shape[-1], dtype=BF16))
    table = ops.shards(tabs, torch.int32, "rm")
    dt = ttnn.bfloat8_b if ops.kv_dtype == "bfp8" else ttnn.bfloat16
    if mk:
        merged = len(t) == 1 and pages.shape[1] > 1  # ONE tensor: head h's blocks at [h nblk, (h + 1) nblk)
        for h in range(pages.shape[1]):
            th = t[0] if merged else t[h]
            tab_h = table
            if merged:
                tab_h = ops.shards([tb + h * nblk for tb in tabs], torch.int32, "rm")
            stage = ops.shards([p[:, h : h + 1] for p in parts], BF16, "tile", tt_dtype=dt)
            ttnn.experimental.paged_fill_cache(th, stage, tab_h, batch_idx=0)
    else:
        stage = ops.shards(parts, BF16, "tile", tt_dtype=dt)
        ttnn.experimental.paged_fill_cache(t, stage, table, batch_idx=0)
    ttnn.synchronize_device(ops.mesh)


def export_units(kind, u0: int, n_units: int) -> torch.Tensor:
    """units u0 .. u0 + n_units - 1 of the kind's context -> the blob [n_units, L, 2, nkv, UNIT, hd] bf16 (head-major,
    owner-free): the head form reads every chip's head slice, the sequence form the owner's rows"""
    tp = kind.ops.tp
    caches = []
    for li in range(len(kind.L)):
        caches.append(
            ([kind_cache_rows(kind, li, 0, r) for r in range(tp)], [kind_cache_rows(kind, li, 1, r) for r in range(tp)])
        )
    return pack_units(caches, u0, n_units, kind.sm)


def import_units(kind, blob: torch.Tensor, u0: int) -> None:
    """the inverse: the blob's units land at u0 .. in the kind's caches (either form, either dtype)"""
    for li, per_kv in enumerate(unpack_units(blob, u0, kind.ops.tp, kind.sm)):
        for kv, per_chip in enumerate(per_kv):
            for r, writes in enumerate(per_chip):
                for r0, rows in writes:
                    kind_write_rows(kind, li, kv, r, r0, rows)


class DrafterPagedView:
    """the plugin-facing page handle of one drafter cache (one layer, K or V): read_pages / write_pages over GLOBAL
    drafter page ids (32 tokens each) with pages [n, nkv (every head), PAGE, hd] bf16 -- the vllm-tt-plugin
    kv_connector TTPagedCacheHandle protocol (read_pages(ids) / write_pages(ids, pages)); chip-independent: the head
    form gathers every chip's head slice, the sequence form reads the owner (page b -> chip (b // 4) % C, local page)"""

    def __init__(self, kind, li: int, kv: int):
        self.kind, self.li, self.kv = kind, li, kv
        d = kind.d
        self.shape = (
            kind.nblk if kind.sm is None else kind.sm.local_cap(kind.cap) // PAGE * kind.sm.C,
            d.nkv,
            PAGE,
            d.hd,
        )
        self.dtype = torch.bfloat16

    @property
    def num_pages(self) -> int:
        return self.shape[0]

    def read_pages(self, ids: list) -> torch.Tensor:
        kind, tp = self.kind, self.kind.ops.tp
        out = torch.zeros(len(ids), *self.shape[1:], dtype=self.dtype)
        rows = [kind_cache_rows(kind, self.li, self.kv, r) for r in range(tp)]
        nkv_l = rows[0].shape[1]
        for i, b in enumerate(ids):
            if kind.sm is None:
                for r in range(tp):
                    out[i, r * nkv_l : (r + 1) * nkv_l] = rows[r][b * PAGE : (b + 1) * PAGE].permute(1, 0, 2)
            else:
                r, lb = kind.sm.page_owner(b), kind.sm.local_page(b)
                out[i] = rows[r][lb * PAGE : (lb + 1) * PAGE].permute(1, 0, 2)
        return out

    def write_pages(self, ids: list, pages: torch.Tensor) -> None:
        kind, tp = self.kind, self.kind.ops.tp
        pages = pages.reshape(len(ids), *self.shape[1:])
        nkv_l = kind.d.nkv // tp
        for i, b in enumerate(ids):
            if kind.sm is None:
                for r in range(tp):
                    kind_write_rows(
                        kind, self.li, self.kv, r, b * PAGE, pages[i, r * nkv_l : (r + 1) * nkv_l].permute(1, 0, 2)
                    )
            else:
                r, lb = kind.sm.page_owner(b), kind.sm.local_page(b)
                kind_write_rows(kind, self.li, self.kv, r, lb * PAGE, pages[i].permute(1, 0, 2))


def connector_views(kind) -> dict:
    """{"drafter.layers.{li}.{k|v}": DrafterPagedView}: the drafter's caches for the KV connector, beside the
    target's kv_cache_connector_views (stack_model.py); the plugin needs its per-layer spec list extended by these
    model-owned caches (page = 32 tokens, 4 per unit, 8 per LMCache chunk)"""
    return {
        f"drafter.layers.{li}.{n}": DrafterPagedView(kind, li, kv)
        for li in range(len(kind.L))
        for kv, n in ((0, "k"), (1, "v"))
    }
