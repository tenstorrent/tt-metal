# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Stage 17: the DFlash2 block drafter INSIDE the decode trace (the SpecMTP drafter kind "dflash2";
doc/ttnn/17-mtp-spec-decode/dflash2_intrace.md). The Apache-2.0 drafter for GLM-5.3-Flash ONLY.

One speculative step (one trace replay, no host work) with this drafter:
  verify (w8, taps)  the main model's verify at the lone user's plain width; on the 9 tap layers the completed residual
                     of the verify rows is copied (bf16) into the drafter's tap slots (SpecMTP._verify: the verify runs
                     are split at the tap layers, 5 more launches; production = an mhc_post output, no launch)
  ACCEPT             SpecCtl: n, the bonus into token row 0, the record (n, c)
  PREP               kernels/spec/df2_ctl.cpp DF2_PREP (one core): from the record and token row 0 -> the block's
                     token ids page [bonus, mask x K], the positions of this step's rows (c + j: the context K / V
                     write), of the block rows (c' + j, c' = c + n + 1) and the block's attention bound (c' + K)
  ctx                the tap slots (all wv rows; rows past n are garbage that later writes overwrite before any read)
                     -> fc (the mean over the 4 mHC streams folded into fc) -> hidden_norm -> per layer the fused
                     q|k|v projection, k_norm, RoPE -> the drafter's paged K / V cache at positions c + j
  block              [bonus, mask x K] embedding rows -> 8 layers (grouped 2-tap dynamic conv, GQA attention over the
                     cache up to position c' + K = the context AND the block (non-causal), SwiGLU) -> final norm ->
                     lm_head (this chip's vocab slice) -> top-16 per chip -> global top-16 (all-gather + top-k)
  select             hidden_projection, the predecessor / successor codebook gathers, the bilinear scores (stock ops)
                     -> DF2_SELECT (one core): the greedy walk from the bonus -> token rows 1..K, the record words 16..
The drafter's K / V cache is the user's own paged cache (block 32, every page-table row = the user's blocks); the block
rows' K / V are written at c' .. c' + K and overwritten by the next step's context write, so no rollback is needed.

Layout (TP over the mesh, the converter's rules, tests/s17/dflash2_convert.py): column split fc (the fused taps; the
output all-gathered), q|k|v (whole GQA groups per chip), up|gate, lm_head (per-chip top-16); row split o / down (fp32
partials all-reduced, one bf16 rounding); replicated conv kernel projections, norms, conv base rows, the group
expansion, the codebooks and the selector projection; the embedding table column-split (one all-gather per step).
Every matmul is a 7a streaming-matmul program (tt/mm_stream.py, the weight in the mk-stream-v1 per-core bank layout,
bfloat8_b), one generic_op launch each, no DRAM-sharded DSLinear.

Two backends run the SAME op sequence (DFlash2Kind is written against the `Ops` interface): `TtnnOps` on a mesh
(the device / the mock) and `TorchOps` on the host (per-chip torch tensors, the collectives as sums / concats): the
host test (tests/s17/test_dflash2_kind_host.py) compares the torch run with tt/dflash2_ref.py."""

from __future__ import annotations

import collections
import math
import os

import torch

from models.autoports.zai_org_glm_5_3_flash.tt.dflash2_ref import Dims, rope_cos_sin, rotate_half

TILE = 32
KDIR = os.path.join("models", "autoports", "zai_org_glm_5_3_flash", "tt", "kernels")
BF16, FP32 = torch.bfloat16, torch.float32
DF2_REV = 4  # kernels/spec/df2_ctl.cpp revision (a define: a long-lived process caches binaries by path + defines); 2: + the rows' cos / sin tiles (rung 3)
DF2_CONV_REV = 1  # kernels/spec/df2_conv_*.cpp revision
DF2_CACHE_REV = 2  # kernels/spec/df2_cache.cpp revision; 2: the bfp8 cache-tile path (DF2_KV_BFP8)
TAP_LAYERS = (5, 9, 14, 19, 24, 28, 33, 38, 42)
CONV_MODES = ("ops", "fused")  # DFlash2Kind._conv: the 10-launch stock-op chain / the rung-2 fused program (1 launch)
# rung 3 M1 (tt/dflash2_mm.py): "stream" = every matmul a 7a mm_stream program (up to 128 cores); "ds" = DS
# reader-phase programs (16-24 reader cores, kernels/mk/reader_*.cpp) with the same-input matmuls grouped into one
# launch (ctx: the 8 qkv projections; head: lm_head + hproj); a matmul whose reader L1 exceeds the budget stays stream
MM_MODES = ("stream", "ds")
# rung 3 M2 (tt/dflash2_mk.py): "ops" = the per-op chain (rung 2 / M1); "mk" = the drafter's layers and context as
# MkLayer-style megakernel programs (owners + DS readers + fabric lanes), the paged SDPA per kv head, the head
# / selector chain as before
LAYER_MODES = ("ops", "mk")
# the head's top-K over this chip's logits: "ttnn" = ttnn.topk on the [32, V] fp32 tensor (every row); "kernel" = the
# exact fp32 top-K of the draft rows 1 .. K as two one-kernel launches (kernels/spec/df2_topk.cpp: 32 scan cores +
# one merge core; the 2026-10-23 silicon breakdown: the head chain was 13.1 of the block's 17.5 ms at TP2)
TOPK_MODES = ("ttnn", "kernel")
TOPK_CORES = 32
DF2_TOPK_REV = 1


def fp32_key(bits: torch.Tensor) -> torch.Tensor:
    """the kernel's order-preserving int32 key of fp32 bit patterns (int32 view): bits >= 0 -> bits, else bits ^ 0x7FFFFFFF"""
    neg = bits < 0
    return torch.where(neg, bits ^ 0x7FFFFFFF, bits)


def topk_rows_host(logits: torch.Tensor, K: int, topk: int, ncores: int = TOPK_CORES) -> tuple:
    """kernels/spec/df2_topk.cpp on the host: logits [32, V] fp32 -> (vals [32, 32] fp32, ids [32, 32] int64): rows
    1 .. K hold the top-k (descending, ties to the lower column; -FLT_MAX / 0 elsewhere). The two-stage selection is
    exact, so this equals torch.topk's values on those rows; the ids agree up to the order of exactly equal values."""
    V = int(logits.shape[-1])
    keys = fp32_key(logits.float().contiguous().view(torch.int32))
    vals = torch.full((TILE, TILE), -3.4028234663852886e38)
    ids = torch.zeros(TILE, TILE, dtype=torch.int64)
    tiles = V // TILE
    for r in range(1, K + 1):
        cand = []  # per core: its top-k (key, col) with the kernel's tie rule (stable sort by key desc, col asc)
        for i in range(ncores):
            c0, c1 = i * tiles // ncores * TILE, (i + 1) * tiles // ncores * TILE
            kr = keys[r, c0:c1]
            order = torch.argsort(-kr.long() * V + torch.arange(c1 - c0)) if False else None
            # stable: sort by (-key, col)
            kk = kr.long()
            idx = torch.arange(c0, c1)
            o = torch.argsort(kk * (-1), stable=True)[:topk]
            cand += [(int(kk[j]), int(idx[j])) for j in o]
        cand.sort(key=lambda e: (-e[0], e[1]))
        for q, (k_, col) in enumerate(cand[:topk]):
            vals[r, q] = torch.tensor(k_ ^ 0x7FFFFFFF if k_ < 0 else k_, dtype=torch.int32).view(torch.float32)
            ids[r, q] = col
    return vals, ids


# the 1M form (tt/dflash2_1m.py, dflash2_intrace.md §10): GLM53_DF2_KV_DTYPE = the K / V cache tiles' dtype (bfp8: the
# cache-write kernels re-encode the rows, kernels/spec/df2_bfp8.hpp; half the bytes and half the per-step read);
# GLM53_DFLASH2_SHARD=1 = the sequence-sharded cache (chip r holds EVERY head's K / V of the positions it owns, the
# target's 128-token unit rule) with the split-K attention merged across the chips (host backend; the device
# attention partial is the rung-3 M3 item)
KV_DTYPES = ("bf16", "bfp8")


def default_kv_dtype() -> str:
    """GLM53_DF2_KV_DTYPE, else bfp8 on the sequence-sharded line (GLM53_DSA_SHARD=1: the 1M serve, coordinator
    2026-10-23 10:01:59Z) and bf16 elsewhere"""
    return os.environ.get("GLM53_DF2_KV_DTYPE") or ("bfp8" if os.environ.get("GLM53_DSA_SHARD", "0") == "1" else "bf16")


class _MM:
    """a matmul launch unit on the torch backend (one weight: [K, N] per chip): raw(x) = the arithmetic, __call__ = raw
    + the launch count; the ttnn units (mm_stream / tt.dflash2_mm.DsMM) have the same surface"""

    def __init__(self, ops, parts: list, out, name: str):
        self.ops, self.W, self.out, self.name = ops, [_bfp8_like(p).float() for p in parts], out, name
        self.n = int(parts[0].shape[1])
        self.sm = None

    def raw(self, x):
        return [(xr.float() @ Wr).to(self.out) for xr, Wr in zip(x, self.W)]

    def __call__(self, x):
        self.ops._tick(1, "ds_mm" if self.ops.mm_mode == "ds" else "stream_mm")
        return self.raw(x)


class _MMGroup:
    """several torch units of the same input: one launch in "ds" mode (the ReaderPhases program), N in "stream\" """

    def __init__(self, ops, mms: list, name: str):
        self.ops, self.mms, self.name = ops, mms, name
        self.sm = None

    def __call__(self, x):
        if self.ops.mm_mode == "ds":
            self.ops._tick(1, "ds_group")
            return [m.raw(x) for m in self.mms]
        return [m(x) for m in self.mms]


def _divisors(n: int) -> list:
    return [d for d in range(1, n + 1) if n % d == 0]


def _bfp8_like(w: torch.Tensor) -> torch.Tensor:
    """host emulation hook (the torch backend rounds the matmul weights to bf16; ttnn packs bfloat8_b, proved to cost
    no acceptance: dflash2_bfp8_accept.json)"""
    return w.to(BF16)


def conv_consts(group: int) -> dict:
    """the rung-2 fused conv program's 0 / 1 selection tiles (kernels/spec/df2_conv_common.hpp), fp32 host tensors:
    E [32, 32, 32] (E[e][j, h] = 1 iff j == e + h // G: the cf tile columns e .. of a column tile -> its channels),
    J [4, 32, 32] (J[k][r, j] = 1 iff j == k: J[k] @ B = row k of B on every row), Sh [32, 32] (Sh[r, r-1] = 1:
    x_prev = Sh @ x, row 0 has no predecessor); gpt = groups per column tile (32 / G), the kernels' GPT"""
    assert 32 % group == 0, group
    gpt = TILE // group
    E = torch.zeros(TILE, TILE, TILE)
    h = torch.arange(TILE)
    for e in range(TILE):
        j = e + h // group
        ok = j < TILE
        E[e, j[ok], h[ok]] = 1
    J = torch.zeros(4, TILE, TILE)
    for k in range(4):
        J[k, :, k] = 1
    Sh = torch.zeros(TILE, TILE)
    Sh[torch.arange(1, TILE), torch.arange(TILE - 1)] = 1
    return dict(E=E, J=J, Sh=Sh, gpt=gpt, group=group)


def conv_fused_host(x: torch.Tensor, cf: torch.Tensor, base4: torch.Tensor, side: int, K: dict) -> torch.Tensor:
    """the fused conv program's arithmetic on the host, tile by tile with the kernels' index math: x [32, H] bf16, cf
    [32, 4 NG] bf16, base4 [32, H] bf16 (rows 0..3 = base[side, tap] at k = 2 side + tap) -> [32, H] bf16 (fp32 math)"""
    xf, cff, bf = x.float(), cf.float(), base4.float()
    H = xf.shape[1]
    ng, gpt = H // K["group"], K["gpt"]
    E, J, Sh = K["E"], K["J"], K["Sh"]
    out = torch.empty(TILE, H)
    for c in range(H // TILE):
        xt, bt = xf[:, c * TILE : (c + 1) * TILE], bf[:, c * TILE : (c + 1) * TILE]
        term = []
        for t in range(2):
            q0 = side * 2 * ng + t * ng + c * gpt
            cft = cff[:, (q0 // TILE) * TILE : (q0 // TILE + 1) * TILE]
            coef = cft @ E[q0 % TILE] + J[2 * side + t] @ bt
            term.append(coef * (Sh @ xt if t == 1 else xt))
        out[:, c * TILE : (c + 1) * TILE] = term[1] + term[0]
    return out.to(BF16)


# ====================================================================== the host layout of the drafter (both backends)
class DFlash2Shards:
    """the drafter's weights as per-chip host shards in the kind's layout ([in, out] for every matmul)"""

    def __init__(self, cfg: dict, w: dict, tp: int, *, tap_mode: str = "fold", ksplit: bool = False,
                 kv_all: bool = False):  # fmt: skip
        d = self.d = Dims(cfg)
        assert d.nkv % tp == 0 and d.V % (tp * TILE) == 0 and d.I % (tp * TILE) == 0 and d.H % (tp * TILE) == 0, (
            tp,
            d.nkv,
            d.V,
            d.I,
            d.H,
        )
        self.tp, self.tap_mode = tp, tap_mode
        self.nh_l, self.nkv_l, self.V_l, self.I_l, self.H_l = d.nh // tp, d.nkv // tp, d.V // tp, d.I // tp, d.H // tp
        # kv_all (the sequence-sharded cache): every chip projects EVERY k / v head (its own q heads): the cache of
        # chip r holds all heads of the positions it owns; nkv_c = the kv heads per chip's cache
        self.kv_all = kv_all
        self.nkv_c = d.nkv if kv_all else self.nkv_l
        H, hd = d.H, d.hd
        t = lambda m: m.to(BF16).t().contiguous()  # noqa: E731  HF [out, in] -> [in, out]
        col = lambda m, r, n: m[r * n : (r + 1) * n]  # noqa: E731  output-column shard r of width n (HF rows)
        # fc: the 9 taps; "fold": the 4-stream mean folded in (input = the raw 4-stream residual of every tap, W/4 per
        # stream: exact in bfloat8_b); "mean": the mean taken by ops before fc (bitwise the prototype's taps)
        fc = w["fc.weight"]  # [H, n_taps H]
        if tap_mode == "fold":
            fc = (fc.float().reshape(H, d.n_taps, 1, H).repeat(1, 1, 4, 1) * 0.25).reshape(H, d.n_taps * 4 * H).to(BF16)
        self.fc_in = int(fc.shape[1])
        self.fc = [t(col(fc, r, self.H_l)) for r in range(tp)]
        if ksplit:  # rung 3: chip r holds K rows [r kk, (r + 1) kk) of the full [fc_in, H] (the partials all-reduced)
            fcT, kk = t(fc), self.fc_in // tp
            assert kk % TILE == 0, (self.fc_in, tp)
            self.fc_k = [fcT[r * kk : (r + 1) * kk].contiguous() for r in range(tp)]
            del fcT
        self.lm_head = [t(col(w["lm_head.weight"], r, self.V_l)) for r in range(tp)]
        self.layers = []
        for i in range(d.L):
            p, a = f"layers.{i}.", f"layers.{i}.self_attn."
            kvc = lambda m, r: m if kv_all else col(m, r, self.nkv_l * hd)  # noqa: E731
            qkv = [t(torch.cat([col(w[a + "q_proj.weight"], r, self.nh_l * hd), kvc(w[a + "k_proj.weight"], r),
                                kvc(w[a + "v_proj.weight"], r)], 0)) for r in range(tp)]  # fmt: skip
            o = [t(w[a + "o_proj.weight"][:, r * self.nh_l * hd : (r + 1) * self.nh_l * hd]) for r in range(tp)]
            ug = [
                t(
                    torch.cat(
                        [
                            col(w[p + "mlp.up_proj.weight"], r, self.I_l),
                            col(w[p + "mlp.gate_proj.weight"], r, self.I_l),
                        ],
                        0,
                    )
                )
                for r in range(tp)
            ]  # fmt: skip  (up | gate: ttnn.swiglu = first * silu(second))
            down = [t(w[p + "mlp.down_proj.weight"][:, r * self.I_l : (r + 1) * self.I_l]) for r in range(tp)]
            self.layers.append(dict(
                qkv=qkv, o=o, ug=ug, down=down,
                kp_attn=t(w[p + "attention_conv.kernel_projection.weight"]), kp_mlp=t(w[p + "mlp_conv.kernel_projection.weight"]),
                base_attn=w[p + "attention_conv.base_kernel"].float().reshape(4, H), base_mlp=w[p + "mlp_conv.base_kernel"].float().reshape(4, H),
                base4_attn=self._base4(w[p + "attention_conv.base_kernel"]), base4_mlp=self._base4(w[p + "mlp_conv.base_kernel"]),
                in_ln=w[p + "input_layernorm.weight"].to(BF16), post_ln=w[p + "post_attention_layernorm.weight"].to(BF16),
                q_norm=w[a + "q_norm.weight"].to(BF16), k_norm=w[a + "k_norm.weight"].to(BF16)))  # fmt: skip
        self.hidden_norm, self.norm = w["hidden_norm.weight"].to(BF16), w["norm.weight"].to(BF16)
        E = torch.zeros(d.ng, H)  # group -> channel expansion (exact 0 / 1)
        E[torch.arange(H) // d.group, torch.arange(H)] = 1
        self.E = E.to(BF16)
        Sh = torch.zeros(TILE, TILE)  # x_prev = Sh x (row 0 has no predecessor)
        Sh[torch.arange(1, TILE), torch.arange(TILE - 1)] = 1
        self.Sh = Sh
        emb = w["embed_tokens.weight"].to(BF16).clone()
        emb[d.mask_id] = w["mask_embedding"].to(BF16)  # the mask token's row IS the learned mask embedding
        self.embed = [emb[:, r * self.H_l : (r + 1) * self.H_l].contiguous() for r in range(tp)]
        self.pred_cb = w["candidate_selector.predecessor_codebook"].to(BF16)
        self.succ_cb = w["candidate_selector.successor_codebook"].to(BF16)
        self.hproj = t(w["candidate_selector.hidden_projection.weight"])
        self.vocab_off = [r * self.V_l for r in range(tp)]

    @staticmethod
    def _base4(base: torch.Tensor) -> torch.Tensor:
        """base_kernel [2 sides, 2 taps, H] -> the fused conv's [32, H] bf16 base tile row: row 2 side + tap = base[side,
        tap] (exact: the values are bf16), rows 4..31 zero (J[k] @ B[c] broadcasts row k)"""
        H = int(base.shape[-1])
        b = torch.zeros(TILE, H, dtype=BF16, device=base.device)  # (meta weights stay meta: the image restore path)
        b[:4] = base.to(BF16).reshape(4, H)
        return b

    # the host tensors a served process never reads after the uploads (11.3 GiB RSS at TP8 for the real drafter:
    # the per-chip shard lists ARE the weights again, bf16); release() drops them once the kind has uploaded
    # everything, keeping the small rows (norms, conv bases) the host emulation / probes read and the byte counts
    BIG = ("fc", "fc_k", "lm_head", "embed", "pred_cb", "succ_cb", "hproj", "E")
    BIG_LAYER = ("qkv", "o", "ug", "down", "kp_attn", "kp_mlp")
    released = False

    def release(self) -> None:
        """drop the uploaded matmul / table host tensors (the byte counts are kept as computed before the drop)"""
        if self.released:
            return
        self._bytes, self._read = self.bytes_per_chip(), self.read_bytes_per_step()
        for k in self.BIG:
            if hasattr(self, k):
                setattr(self, k, None)
        for L in self.layers:
            for k in self.BIG_LAYER:
                L[k] = None
        # the kept small rows are zero-copy VIEWS of the safetensors mmap (every one pins the whole 6.2 GB mapping
        # as file-backed RSS): own copies instead
        clone = lambda t: t.clone() if isinstance(t, torch.Tensor) else t  # noqa: E731
        for k in ("hidden_norm", "norm", "Sh"):
            setattr(self, k, clone(getattr(self, k)))
        for L in self.layers:
            for k in list(L):
                L[k] = clone(L[k])
        self.released = True

    def bytes_per_chip(self) -> dict:
        """DRAM bytes per chip of the resident weights (bfloat8_b matmuls: 1.0625 B / value; bf16 tables / norms)"""
        if self.released:
            return dict(self._bytes)
        b8 = lambda m: int(m.numel() * 17 // 16)  # noqa: E731
        b16 = lambda m: int(m.numel() * 2)  # noqa: E731
        mm = b8(self.fc[0]) + b8(self.lm_head[0]) + b8(self.hproj)
        for L in self.layers:
            mm += sum(b8(L[k][0]) for k in ("qkv", "o", "ug", "down")) + b8(L["kp_attn"]) + b8(L["kp_mlp"])
        tables = b16(self.embed[0]) + b16(self.pred_cb) + b16(self.succ_cb) + b16(self.E)
        return dict(matmul_bfp8=mm, tables_bf16=tables, total=mm + tables)

    def read_bytes_per_step(self) -> int:
        """weight bytes one step streams per chip (every matmul once; the conv expansion 16x; tables by row)"""
        if self.released:
            return self._read
        b8 = lambda m: int(m.numel() * 17 // 16)  # noqa: E731
        n = b8(self.fc[0]) + b8(self.lm_head[0]) + b8(self.hproj) + 16 * 2 * int(self.E.numel() * 2)
        for L in self.layers:
            n += (
                2 * b8(L["qkv"][0]) + sum(b8(L[k][0]) for k in ("o", "ug", "down")) + b8(L["kp_attn"]) + b8(L["kp_mlp"])
            )
        return n


# ====================================================================== the op interface
class Ops:
    """the op set the kind is written against (ttnn on a mesh / torch on the host); tensors are opaque handles"""

    tp: int


class TorchOps(Ops):
    """host backend: a mesh tensor = a list of per-chip torch tensors (replicated: equal copies); shapes mimic ttnn's
    4-D activations [1, 1, 32, W]; the sharded memory configs are no-ops; the kernels are torch code"""

    def __init__(self, tp: int, *, mm: str | None = None, layer: str | None = None, kv_dtype: str | None = None,
                 shard: bool | None = None):  # fmt: skip
        self.tp = tp
        self.launches = 0
        self.census = collections.Counter()  # launches per op name (the rung-3 sizing: dflash2_intrace.md)
        self.mm_mode = mm or os.environ.get("GLM53_DF2_MM", "stream")
        assert self.mm_mode in MM_MODES, self.mm_mode
        self.layer_mode = layer or os.environ.get("GLM53_DF2_LAYER", "ops")
        assert self.layer_mode in LAYER_MODES, self.layer_mode
        self.kv_dtype = kv_dtype or default_kv_dtype()
        assert self.kv_dtype in KV_DTYPES, self.kv_dtype
        self.shard = (os.environ.get("GLM53_DFLASH2_SHARD", "0") == "1") if shard is None else bool(shard)

    def _tick(self, n: int = 1, name: str | None = None):
        import sys

        self.launches += n
        self.census[name or sys._getframe(1).f_code.co_name] += n

    # tensors
    def shards(self, parts: list, dtype=BF16, layout="tile", tt_dtype=None):
        return [p.to(dtype).clone() for p in parts]

    def rep(self, t: torch.Tensor, dtype=BF16, layout="tile", tt_dtype=None):
        return [t.to(dtype).clone() for _ in range(self.tp)]

    def zeros(self, shape, dtype=BF16, layout="tile", tt_dtype=None):
        return [torch.zeros(*shape, dtype=dtype) for _ in range(self.tp)]

    def host(self, t, chip: int = 0) -> torch.Tensor:
        return t[chip].clone()

    def write(self, t, host: torch.Tensor):
        for x in t:
            x.copy_(host.to(x.dtype).reshape(x.shape))

    def write_shards(self, t, hosts: list):  # a different host value per chip
        for x, h in zip(t, hosts):
            x.copy_(h.to(x.dtype).reshape(x.shape))

    def cache_zeros(self, shape):  # a K / V cache (bf16 here; bfp8 values are bf16-exact: cache_write rounds)
        return self.zeros(shape)

    # matmuls
    def stream_mm(self, parts: list, *, out=BF16, name: str = ""):
        return _MM(self, parts, out, name)

    def mm(self, parts: list, *, out=BF16, name: str = ""):
        """a matmul launch unit (the mode decides the program kind on ttnn; one arithmetic here)"""
        return _MM(self, parts, out, name)

    def mm_group(self, mms: list, *, name: str = ""):
        """units of the same input -> g(x) = [m(x) for m]: one launch in "ds" mode (one ReaderPhases program)"""
        return _MMGroup(self, mms, name)

    def linear_bias(self, x, Wr, bias, out=FP32):
        self._tick(1)
        return [(xr.float() @ Wr[0].float() + b.float()).to(out) for xr, b in zip(x, bias)]

    # elementwise / layout
    def _map(self, f, *ts):
        import sys

        self._tick(1, sys._getframe(1).f_code.co_name)
        return [f(*parts) for parts in zip(*ts)]

    def typecast(self, x, dtype):
        return self._map(lambda a: a.to(dtype), x)

    def add(self, x, y):
        return self._map(lambda a, b: (a.float() + b.float()).to(a.dtype), x, y)

    def mul(self, x, y):
        return self._map(lambda a, b: (a.float() * b.float()).to(a.dtype), x, y)

    def rms(self, x, w, eps):
        def f(a, ww):
            af = a.float()
            return (af * torch.rsqrt(af.pow(2).mean(-1, keepdim=True) + eps) * ww.float().reshape(-1)).to(a.dtype)

        return self._map(f, x, w)

    def swiglu(self, x):  # [up | gate] -> up * silu(gate)
        def f(a):
            h = a.shape[-1] // 2
            return (a[..., :h].float() * torch.nn.functional.silu(a[..., h:].float())).to(a.dtype)

        return self._map(f, x)

    def slice(self, x, dim, a, b):
        return self._map(lambda t: t.narrow(dim, a, b - a).clone(), x)

    def concat(self, xs, dim):
        self._tick(1)
        return [torch.cat(parts, dim) for parts in zip(*xs)]

    def shift_rows(self, x, Sh):  # x_prev = Sh x on the 32 rows (fp32)
        return self._map(lambda a, s: (s.float() @ a.float()).to(a.dtype), x, Sh)

    def pad_rows(self, x, rows):  # the verify width's wv rows -> the slot's 32 rows (zeros past wv)
        if int(x[0].shape[-2]) == rows:
            return x
        self._tick(1)
        return [torch.nn.functional.pad(t, (0, 0, 0, rows - t.shape[-2])) for t in x]

    def reshape(self, x, shape):  # metadata only
        return [t.reshape(shape) for t in x]

    def conv_fused(
        self, x, cf, base4, side: int, K: dict, res=None
    ):  # the rung-2 program (kernels/spec/df2_conv_*.cpp)
        if res is not None:  # DF2_CONV_RES: x = the fp32 partial -> bf16 (the packer's rounding), out = res + conv(x)
            self._tick(1, "conv_fused_res")
            return [(conv_fused_host(xr.to(BF16).reshape(TILE, -1), c.reshape(TILE, -1), b.reshape(TILE, -1), side, K).float()
                     + r.float().reshape(TILE, -1)).to(BF16).reshape(r.shape) for xr, c, b, r in zip(x, cf, base4, res)]  # fmt: skip
        self._tick(1)
        return [conv_fused_host(xr.reshape(TILE, -1), cr.reshape(TILE, -1), br.reshape(TILE, -1), side, K).reshape(xr.shape)
                for xr, cr, br in zip(x, cf, base4)]  # fmt: skip

    # collectives
    def all_gather(self, x, dim):
        self._tick(1)
        full = torch.cat(x, dim)
        return [full.clone() for _ in range(self.tp)]

    def all_reduce(self, x):
        self._tick(1)
        s = sum(t.float() for t in x)
        return [s.to(x[0].dtype) for _ in range(self.tp)]

    # tables
    def embedding(self, ids, table):  # ids [1, n] / [n, m] u32 -> [1, n, W] / [n, m, W]
        return self._map(lambda i, tb: tb[i.long()], ids, table)

    # attention pieces (shapes as ttnn's: q [1, 32, nh, hd] etc.)
    def create_heads(self, qkv, nh, nkv, hd):
        self._tick(1)
        q = [t.reshape(32, -1)[:, : nh * hd].reshape(1, 32, nh, hd).clone() for t in qkv]
        k = [t.reshape(32, -1)[:, nh * hd : (nh + nkv) * hd].reshape(1, 32, nkv, hd).clone() for t in qkv]
        v = [t.reshape(32, -1)[:, (nh + nkv) * hd :].reshape(1, 32, nkv, hd).clone() for t in qkv]
        return q, k, v

    def head_norm(self, x, w, eps):  # the sharded <-> interleaved reshards around rms_norm (3 launches on ttnn)
        self._tick(2)
        return self.rms(x, w, eps)

    def rot_mats(self, posu, cos_tab, sin_tab):  # positions [1, 32] u32 -> cos / sin [1, 32, 1, hd] bf16
        self._tick(6)
        cos = [c[p.reshape(-1).long()].reshape(1, 32, 1, -1) for p, c in zip(posu, cos_tab)]
        sin = [s[p.reshape(-1).long()].reshape(1, 32, 1, -1) for p, s in zip(posu, sin_tab)]
        return cos, sin

    def rope(self, x, cos, sin):  # HF rotate-half, fp32 math, bf16 out
        def f(a, c, s):
            af = a.float()
            return (af * c.float() + rotate_half(af) * s.float()).to(a.dtype)

        return self._map(f, x, cos, sin)

    def cache_update(
        self, cache, kh, pos
    ):  # cache [blocks, nkv, 32, hd] (dense pages), kh [1, 32, nkv, hd], pos [32] i32
        self._tick(1)
        for c, k, p in zip(cache["t"], kh, pos):
            for j in range(32):
                pj = int(p[j])
                if 0 <= pj < c.shape[0] * TILE:
                    c[pj // TILE, :, pj % TILE, :] = k[0, j].to(c.dtype)
        return cache

    def cache_write(self, Kc, Vc, kh, vh, pos):
        """the 32 K and V rows at positions pos (the device: ONE df2_cache.cpp launch; here two row writes). race=True
        emulates paged_update_cache's 32-users-on-shared-pages read-modify-write race (the last row of every
        destination tile survives): the Stage 17 u1 silicon signature, for the anchors host test"""
        self._tick(1)
        q8 = None
        if self.kv_dtype == "bfp8":  # the kernel re-encodes the rows as bfp8_b (16-value groups along hd)
            from models.autoports.zai_org_glm_5_3_flash.tt.dflash2_1m import bfp8_like

            q8 = lambda row: bfp8_like(row.to(BF16))  # noqa: E731
        if not getattr(self, "race", False):
            for cache, src in ((Kc, kh), (Vc, vh)):
                for c, k, p in zip(cache["t"], src, pos):
                    for j in range(32):
                        pj = int(p[j])
                        if 0 <= pj < c.shape[0] * TILE:
                            row = k[0, j]
                            c[pj // TILE, :, pj % TILE, :] = (q8(row) if q8 else row).to(c.dtype)
            return
        for cache, src in ((Kc, kh), (Vc, vh)):
            for c, k, p in zip(cache["t"], src, pos):
                last = {}
                for j in range(32):
                    pj = int(p[j])
                    if 0 <= pj < c.shape[0] * TILE:
                        last[pj // TILE] = j  # the tile's last writer wins; the other rows keep the stale tile
                for blk, j in last.items():
                    pj = int(p[j])
                    c[blk, :, pj % TILE, :] = k[0, j].to(c.dtype)

    def sdpa_split(self, q, Kc, Vc, cur_l, scale, sm, nh_l: int):
        """the sequence-sharded attention (tt/dflash2_1m.py): q [1, 32, nh_l, hd] per chip -> all-gathered over the
        chips; chip r attends ITS local rows <= cur_l[r][row] (every kv head) and emits the fp32 (m, l, O) partial;
        the partials merged in chip order (bitwise replicas) -> every chip's own q heads [1, 32, nh_l, hd] bf16"""
        from models.autoports.zai_org_glm_5_3_flash.tt.dflash2_1m import attention_partial, merge_partials

        self._tick(1, "all_gather")
        q_all = torch.cat(q, 2)  # [1, 32, nh, hd]
        nh, hd = q_all.shape[2], q_all.shape[3]
        parts = []  # per chip: per row (m, l, O)
        for K, V, cu in zip(Kc["t"], Vc["t"], cur_l):
            nkv = K.shape[1]
            Kf = K.float().permute(0, 2, 1, 3).reshape(-1, nkv, hd)  # local position order
            Vf = V.float().permute(0, 2, 1, 3).reshape(-1, nkv, hd)
            rows = []
            for j in range(32):
                n = int(cu[j]) + 1
                rows.append(attention_partial(q_all[0, j : j + 1], Kf[: max(n, 0)], Vf[: max(n, 0)], scale))
            parts.append(rows)
        self._tick(1, "sdpa_partial")
        self._tick(1, "merge_partials")
        o = torch.stack([merge_partials([parts[r][j] for r in range(self.tp)])[0] for j in range(32)])  # [32, nh, hd]
        o = o.reshape(1, 32, nh, hd)
        return [o[:, :, r * nh_l : (r + 1) * nh_l].to(q[0].dtype).clone() for r in range(self.tp)]

    def sdpa(self, q, Kc, Vc, cur, scale):  # q [1, 32, nh, hd] -> [1, 32, nh, hd]; rows attend positions <= cur[row]
        self._tick(1)
        out = []
        for qr, K, V, cu in zip(q, Kc["t"], Vc["t"], cur):
            nh, nkv = qr.shape[2], K.shape[1]
            g = nh // nkv
            Kf = K.float().permute(1, 0, 2, 3).reshape(nkv, -1, K.shape[3])  # [nkv, S, hd]
            Vf = V.float().permute(1, 0, 2, 3).reshape(nkv, -1, V.shape[3])
            o = torch.zeros(1, 32, nh, qr.shape[3])
            for j in range(32):
                n = int(cu[j]) + 1
                if n <= 0:
                    continue
                qf = qr[0, j].float().reshape(nkv, g, -1)
                s = torch.einsum("kgd,ksd->kgs", qf, Kf[:, :n]) * scale
                o[0, j] = torch.einsum("kgs,ksd->kgd", torch.softmax(s, -1), Vf[:, :n]).reshape(nh, -1)
            out.append(o.to(qr.dtype))
        return out

    def concat_heads(self, a, nh):  # [1, 32, nh, hd] -> [1, 1, 32, nh hd] (+ the reshard launch)
        self._tick(1)
        return self._map(lambda t: t[:, :, :nh].reshape(1, 1, 32, -1).clone(), a)

    # top-k / selector
    def topk(self, x, k):
        self._tick(1)
        v, i = zip(*(torch.topk(t.float(), k, dim=-1, largest=True, sorted=True) for t in x))
        return list(v), [j.to(torch.int64) for j in i]

    def add_offset(self, ids, off):  # per-chip vocab offsets (typecast + add on ttnn)
        self._tick(2)
        return [i + o for i, o in zip(ids, off)]

    def topk_rows(self, logits, K: int, k: int):
        """the kernel's top-k of rows 1 .. K (kernels/spec/df2_topk.cpp): (vals [1,1,32,32] fp32, ids [1,1,32,32] int64);
        2 launches on ttnn"""
        self._tick(2, "topk_rows")
        out = [topk_rows_host(t.reshape(TILE, -1), K, k) for t in logits]
        return [v.reshape(1, 1, TILE, TILE) for v, _ in out], [i.reshape(1, 1, TILE, TILE) for _, i in out]

    def gather(self, x, dim, idx):
        return self._map(lambda t, i: torch.gather(t, dim, i.long()), x, idx)

    def to_rm(self, x):  # layout change (ttnn: to_layout)
        self._tick(1)
        return x

    def prev_ids(self, bonus16, cand_rm):  # [bonus x 16 (row 0, unused), bonus x 16, cand rows 1..30]
        self._tick(2)
        return [torch.cat([b, b, c[1:31]], 0) for b, c in zip(bonus16, cand_rm)]

    def selector_scores(
        self, hp, pred, succ, unary
    ):  # hp [32, rank] fp32, pred / succ [32, 16, rank] bf16 -> [32, 16, 16]
        self._tick(5)  # (typecast, reshape, mul, transpose + matmul, add)
        out = []
        for h, p, s, u in zip(hp, pred, succ, unary):
            sc = torch.einsum("lpr,lcr->lpc", p.float() * h.float().reshape(32, 1, -1), s.float())
            out.append(sc + u.float().reshape(32, 1, -1))
        return out

    # the kernels (torch code here)
    def prep(self, st: "DFlash2State", K: int, mask_id: int, rope=None):
        self._tick(1)
        for r in range(self.tp):
            rec, tin = st.rec[r].reshape(-1), st.tin[r]
            n, c = int(rec[0]), int(rec[1])
            c1 = c + n + 1
            bonus = int(tin[0, 0])
            ids = torch.zeros(1, 32, dtype=torch.int64)
            ids[0, 0], ids[0, 1 : K + 1] = bonus, mask_id
            st.ids[r].copy_(ids)
            st.bonus16[r].fill_(bonus)
            j = torch.arange(32)
            st.bpos[r].copy_(c1 + j)
            st.bposu[r].copy_((c1 + j).reshape(1, 32))
            st.cpos[r].copy_(c + j)
            st.cposu[r].copy_((c + j).reshape(1, 32))
            st.cur[r].fill_(c1 + K)
            if (
                st.rank is not None
            ):  # the sequence-sharded cache: the rows' local positions, the local bound (df2_ctl.cpp DF2_SHARD)
                from models.autoports.zai_org_glm_5_3_flash.tt.dflash2_1m import DrafterShardMap

                sm = DrafterShardMap(self.tp)
                st.cpos_l[r].copy_(sm.local_pages_of((c + j).clamp(min=0), r, st.local_cap))
                st.bpos_l[r].copy_(sm.local_pages_of((c1 + j).clamp(min=0), r, st.local_cap))
                st.cur_l[r].fill_(sm.local_bound(max(c1 + K, -1), r))
            if rope is not None:  # rung 3: the rows' cos / sin tiles (the device PREP writes them the same way)
                cos_tab, sin_tab, cosb, sinb, cosc, sinc, cap, cur_g = rope
                cur_g[r].fill_(c1 + K)
                for base, (co, si) in ((c1, (cosb, sinb)), (c, (cosc, sinc))):
                    p = (base + j).clamp(0, cap - 1)
                    co[r].copy_(cos_tab[r][p].reshape(co[r].shape))
                    si[r].copy_(sin_tab[r][p].reshape(si[r].shape))

    def mk_build(self, kind):
        """rung 3: the host emulation of the layer / ctx programs (tt/dflash2_mk.Df2MkTorch)"""
        from models.autoports.zai_org_glm_5_3_flash.tt.dflash2_mk import Df2MkTorch

        return Df2MkTorch(self, kind.d, kind.S, nblk=kind.nblk)

    def select_walk(self, st: "DFlash2State", scores, cand_rm, K: int):
        self._tick(1)
        for r in range(self.tp):
            prev, toks = 0, []
            for s in range(1, K + 1):
                j = int(torch.argmax(scores[r][s, prev, :16]))
                toks.append(int(cand_rm[r][s, j]))
                prev = j
            for s, tok in enumerate(toks, 1):
                st.tin[r][s, 0] = tok
                st.rec[r].reshape(-1)[16 + s - 1] = tok


class TtnnOps(Ops):
    """device backend on a mesh (one chip = a 1x1 mesh): DRAM-interleaved intermediates; the sharded chain of the
    fused decode-attention ops (create_heads, rotary_embedding_hf, paged_update_cache, paged sdpa decode,
    concat_heads) on 32 cores; the 7a streaming matmul per weight; the two control kernels of kernels/spec/df2_ctl.cpp
    """

    def __init__(self, mesh, *, kb: int = 8, log=print):
        import ttnn

        self.ttnn, self.mesh, self.kb, self.log = ttnn, mesh, kb, log
        self.tp = int(mesh.get_num_devices())
        r, c = int(mesh.shape[0]), int(mesh.shape[1])
        self.axes = [1, 0] if (r > 1 and c > 1) else [None]
        self.DRAM, self.L1 = ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG
        self.dt = {BF16: ttnn.bfloat16, FP32: ttnn.float32, torch.int32: ttnn.int32, torch.uint32: ttnn.uint32,
                   torch.int64: ttnn.uint32}  # fmt: skip
        self.ckc = ttnn.init_device_compute_kernel_config(mesh.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
                                                          math_approx_mode=False, fp32_dest_acc_en=True,
                                                          packer_l1_acc=False)  # fmt: skip
        self.hifi2 = ttnn.init_device_compute_kernel_config(mesh.arch(), math_fidelity=ttnn.MathFidelity.HiFi2,
                                                            math_approx_mode=False, fp32_dest_acc_en=True,
                                                            packer_l1_acc=False)  # fmt: skip
        grid = mesh.compute_with_storage_grid_size()
        self.ncores = int(grid.x) * int(grid.y)
        self.grid = grid
        # the 32 batch cores of the attention chain: ONE rectangle from (0, 0) (nlp_concat_heads_decode treats a
        # multi-range grid as a sub-core-grid run and needs sub_core_grids then; tt_transformers' num_to_corerange(32))
        assert int(grid.x) >= 8 and int(grid.y) >= 4, (grid.x, grid.y)
        self.b32 = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))])
        self.rep_mapper = ttnn.ReplicateTensorToMesh(mesh) if self.tp > 1 else None
        self.launches = 0
        self.census = collections.Counter()  # launches per op name (the mock census by op: dflash2_intrace.md rung 3)
        self.ctl_core = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))])
        # rung 3 M1 knobs (dflash2_intrace.md §8; every one bitwise the rung-2 kind's arithmetic per output tile):
        #   GLM53_DF2_MM=stream|ds       the matmul program kind (tt/dflash2_mm.py)
        #   GLM53_DF2_MM_CORES=<n>       mm_stream programs on at most n cores (default: the grid; fewer cores = fewer
        #                                per-core runtime-arg writes per launch; the per-core ring must fit L1)
        #   GLM53_DF2_CONV_CORES=<n>     the fused conv program on at most n cores (the column tiles balanced)
        self.mm_mode = os.environ.get("GLM53_DF2_MM", "stream")
        assert self.mm_mode in MM_MODES, self.mm_mode
        self.mm_cores = int(os.environ.get("GLM53_DF2_MM_CORES", "0")) or self.ncores
        self.conv_cores = int(os.environ.get("GLM53_DF2_CONV_CORES", "0")) or self.ncores
        self.mm_plans = {}  # name -> the unit's geometry (the mock rows)
        #   GLM53_DF2_HEADS_SHARDED=1    the attention output stays on the batch cores: the paged SDPA decode writes
        #                                its output height-sharded (nkv_l == 1 only: the op refuses a sharded output
        #                                for GQA, i.e. TP2) and nlp_concat_heads_decode's width-sharded L1 output is
        #                                read in place by the o matmul (TensorAccessor): concat_heads 3 -> 1 launch
        self.heads_sharded = os.environ.get("GLM53_DF2_HEADS_SHARDED", "0") == "1"
        #   GLM53_DF2_LAYER=ops|mk       rung 3 M2: the layers / ctx as megakernel programs (tt/dflash2_mk.py)
        self.layer_mode = os.environ.get("GLM53_DF2_LAYER", "ops")
        assert self.layer_mode in LAYER_MODES, self.layer_mode
        self.merged_cache = True  # mk: one K / V tensor per layer (tt/dflash2_mk.py)
        #   GLM53_DF2_KV_DTYPE=bf16|bfp8  the K / V cache tiles (tt/dflash2_1m.py; kernels/spec/df2_bfp8.hpp); unset:
        #                                bfp8 on the sequence-sharded line (GLM53_DSA_SHARD=1), else bf16
        #   GLM53_DFLASH2_SHARD=1        the sequence-sharded cache (PREP's local pages, the owner-aware writes; the
        #                                split-K attention partial on device = the rung-3 M3 item: sdpa_split raises)
        self.kv_dtype = default_kv_dtype()
        assert self.kv_dtype in KV_DTYPES, self.kv_dtype
        self.shard = os.environ.get("GLM53_DFLASH2_SHARD", "0") == "1"

    def _tick(self, n: int = 1, name: str | None = None):
        import sys

        self.launches += n
        self.census[name or sys._getframe(1).f_code.co_name] += n

    # ------------------------------------------------------------ tensors
    def _mesh(self, hosts: list):
        ttnn = self.ttnn
        if self.tp == 1:
            return ttnn.to_device(hosts[0], self.mesh, memory_config=self.DRAM)
        return ttnn.to_device(ttnn.from_host_shards(hosts, self.mesh.shape), self.mesh, memory_config=self.DRAM)

    def shards(self, parts: list, dtype=BF16, layout="tile", tt_dtype=None):
        ttnn = self.ttnn
        lay = ttnn.TILE_LAYOUT if layout == "tile" else ttnn.ROW_MAJOR_LAYOUT
        dt = tt_dtype or self.dt[dtype]
        hosts = [
            ttnn.from_torch(p.to(dtype if dtype in (BF16, FP32) else p.dtype), dtype=dt, layout=lay) for p in parts
        ]
        return self._mesh(hosts)

    def rep(self, t: torch.Tensor, dtype=BF16, layout="tile", tt_dtype=None):
        return self.shards([t] * self.tp, dtype, layout, tt_dtype)

    def zeros(self, shape, dtype=BF16, layout="tile", tt_dtype=None):
        return self.rep(
            torch.zeros(*shape, dtype=dtype if dtype in (BF16, FP32) else torch.int32), dtype, layout, tt_dtype
        )

    def host(self, t, chip: int = 0) -> torch.Tensor:
        ttnn = self.ttnn
        return ttnn.to_torch(ttnn.get_device_tensors(t)[chip] if self.tp > 1 else t)

    def write(self, t, host: torch.Tensor):
        ttnn = self.ttnn
        h = ttnn.from_torch(host.reshape(tuple(t.shape)).to({ttnn.bfloat16: BF16, ttnn.float32: FP32}.get(t.dtype, torch.int32)),
                            dtype=t.dtype, layout=t.layout, mesh_mapper=self.rep_mapper)  # fmt: skip
        ttnn.copy_host_to_device_tensor(h, t)

    def write_shards(self, t, hosts: list):  # a different host value per chip (the rank page, the local positions)
        ttnn = self.ttnn
        dt = {ttnn.bfloat16: BF16, ttnn.float32: FP32}.get(t.dtype, torch.int32)
        hs = [ttnn.from_torch(h.reshape(tuple(t.shape)).to(dt), dtype=t.dtype, layout=t.layout) for h in hosts]
        h = hs[0] if self.tp == 1 else ttnn.from_host_shards(hs, self.mesh.shape)
        ttnn.copy_host_to_device_tensor(h, t)

    def cache_zeros(self, shape):
        """a K / V cache allocated and zeroed ON the device (no host staging: a 1M cache is 256 MiB per layer per
        chip), in the cache dtype (GLM53_DF2_KV_DTYPE)"""
        ttnn = self.ttnn
        dt = ttnn.bfloat8_b if self.kv_dtype == "bfp8" else ttnn.bfloat16
        return ttnn.zeros(shape, dtype=dt, layout=ttnn.TILE_LAYOUT, device=self.mesh, memory_config=self.DRAM)

    # ------------------------------------------------------------ matmuls (7a streaming)
    def stream_mm(self, parts: list, *, out=BF16, name: str = ""):
        from models.autoports.zai_org_glm_5_3_flash.tt.mm_stream import BF8, StreamMatmul, stream_layout, stream_tiles

        ttnn = self.ttnn
        from models.autoports.zai_org_glm_5_3_flash.tt.mm_stream import TILE_BYTES, _tiles_per_txn

        k, n = int(parts[0].shape[0]), int(parts[0].shape[1])
        nt = n // TILE
        kt = k // TILE
        kb = max(d for d in _divisors(kt) if d <= self.kb)
        # cores: the most <= the cap that divide the column tiles (the cap = the grid unless GLM53_DF2_MM_CORES), raised
        # to the next divisor while the per-core L1 (the 3-deep weight ring, the activation block, the fp32 partials,
        # the output tiles) exceeds the budget; the arithmetic per output tile does not depend on the core count
        from models.autoports.zai_org_glm_5_3_flash.tt.dflash2_mm import L1_BUDGET

        l1 = (
            lambda c: 3 * kb * (nt // c) * TILE_BYTES[BF8]
            + 2 * kb * 2048
            + (nt // c) * (4096 + TILE_BYTES[self.dt[out]])
        )  # noqa: E731
        divs = [d for d in _divisors(nt) if d <= self.ncores]
        cores = (
            max(d for d in divs if d <= min(self.mm_cores, self.ncores))
            if any(d <= self.mm_cores for d in divs)
            else divs[0]
        )
        while l1(cores) > L1_BUDGET and any(d > cores for d in divs):
            cores = min(d for d in divs if d > cores)
        sm = StreamMatmul.__new__(StreamMatmul)
        sm.device, sm.lay = self.mesh, stream_layout(k, n, cores)
        sm.k, sm.n, sm.kt, sm.nt, sm.ncores = k, n, sm.lay["kt"], sm.lay["nt"], cores
        sm.kb, sm.nb = kb, sm.kt // kb
        sm.w_dtype, sm.out_dtype, sm.fidelity = BF8, self.dt[out], ttnn.MathFidelity.HiFi2
        self.mm_plans[name] = dict(kind="stream", cores=cores, nt=sm.nt, kb=kb, l1=l1(cores))
        sm.tb = TILE_BYTES[BF8]
        sm.tpt = _tiles_per_txn(kb * sm.nt, sm.tb)
        sm.subw = max(s for s in (4, 2, 1) if sm.nt % s == 0)
        hosts = [ttnn.from_torch(stream_tiles(p.float(), sm.lay).reshape(1, 1, -1, TILE), dtype=BF8, layout=ttnn.TILE_LAYOUT)
                 for p in parts]  # fmt: skip
        sm.weight = self._mesh(hosts)
        allc = [ttnn.CoreCoord(x, y) for y in range(self.grid.y) for x in range(self.grid.x)]
        sm.core_list = allc[:cores]
        sm.cores = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in sm.core_list])
        sm.calls = 0
        ops = self

        class _Unit:
            name = ""

            def raw(self, x):
                return sm(x, out_memory_config=ops.DRAM)

            def __call__(self, x):
                ops._tick(1, "stream_mm")
                return self.raw(x)

        mm = _Unit()
        mm.sm, mm.n, mm.name = sm, n, name
        return mm

    def mm(self, parts: list, *, out=BF16, name: str = ""):
        """a matmul launch unit: "ds" = a DsMM (tt/dflash2_mm.py) when its reader L1 fits the budget, else (and in
        "stream" mode) an mm_stream program"""
        if self.mm_mode == "ds":
            from models.autoports.zai_org_glm_5_3_flash.tt.dflash2_mm import L1_BUDGET, DsMM, ds_plan

            plan = ds_plan(int(parts[0].shape[1]), out, kb=self.kb)
            if plan["fits"]:
                u = DsMM(self, parts, out=out, name=name, kb=self.kb)
                self.mm_plans[name] = dict(kind="ds", **plan)
                return u
            self.log(f"dflash2 mm {name}: DS reader L1 {plan['l1']} > {L1_BUDGET}: mm_stream")
        return self.stream_mm(parts, out=out, name=name)

    def mm_group(self, mms: list, *, name: str = ""):
        """units of the same input: ONE ReaderPhases program when every unit is a DsMM, else the units one by one"""
        from models.autoports.zai_org_glm_5_3_flash.tt.dflash2_mm import DsGroup, DsMM

        if self.mm_mode == "ds" and all(isinstance(m, DsMM) for m in mms):
            g = DsGroup(self, mms, name=name)
            self.mm_plans[name] = dict(kind="ds_group", phases=[m.name for m in mms], l1=g.l1)
            return g
        if self.mm_mode == "ds":
            self.log(f"dflash2 mm group {name}: not every unit is a DS program: {len(mms)} launches")

        class _Seq:
            sm = None

            def __call__(self_, x):
                return [m(x) for m in mms]

        g = _Seq()
        g.name, g.mms = name, mms
        return g

    def _unpad(self, y, n: int):
        """a DS output [1, 1, 32, n_pad] -> [.., n] (one slice launch when the weight was padded: the TP8 lm_head)"""
        if int(y.shape[-1]) == n:
            return y
        self._tick(1, "slice_pad")
        return self.ttnn.slice(y, [0, 0, 0, 0], [1, 1, TILE, n], memory_config=self.DRAM)

    def linear_bias(self, x, Wr, bias, out=FP32):
        self._tick(1)
        return self.ttnn.linear(
            x, Wr, bias=bias, dtype=self.dt[out], compute_kernel_config=self.ckc, memory_config=self.DRAM
        )

    # ------------------------------------------------------------ elementwise / layout
    def typecast(self, x, dtype):
        self._tick(1)
        return self.ttnn.typecast(x, self.dt[dtype], memory_config=self.DRAM)

    def add(self, x, y):
        self._tick(1)
        return self.ttnn.add(x, y, memory_config=self.DRAM)

    def mul(self, x, y):
        self._tick(1)
        return self.ttnn.multiply(x, y, memory_config=self.DRAM)

    def rms(self, x, w, eps):
        self._tick(1)
        return self.ttnn.rms_norm(x, epsilon=eps, weight=w, compute_kernel_config=self.ckc, memory_config=self.DRAM)

    def swiglu(self, x):
        self._tick(1)
        return self.ttnn.swiglu(x, memory_config=self.DRAM)

    def slice(self, x, dim, a, b):
        self._tick(1)
        s, e = [0] * len(x.shape), list(x.shape)
        s[dim], e[dim] = a, b
        return self.ttnn.slice(x, s, e, memory_config=self.DRAM)

    def concat(self, xs, dim):
        self._tick(1)
        return self.ttnn.concat(xs, dim=dim, memory_config=self.DRAM)

    def shift_rows(self, x, Sh):
        self._tick(1)
        return self.ttnn.matmul(Sh, x, compute_kernel_config=self.ckc, memory_config=self.DRAM)

    def reshape(self, x, shape):
        return self.ttnn.reshape(x, shape)

    def pad_rows(self, x, rows):
        r = int(x.shape[-2])
        if r == rows:
            return x
        self._tick(1)
        return self.ttnn.pad(x, [(0, 0), (0, 0), (0, rows - r), (0, 0)], 0.0, memory_config=self.DRAM)

    # ------------------------------------------------------------ collectives (stock; the in-program fabric phase is the production item)
    def all_gather(self, x, dim):
        for ax in self.axes:
            self._tick(1)
            x = self.ttnn.all_gather(x, dim=dim, cluster_axis=ax, memory_config=self.DRAM) if self.tp > 1 else x
        return x

    def all_reduce(self, x):
        for ax in self.axes[::-1]:
            self._tick(1)
            x = self.ttnn.all_reduce(x, cluster_axis=ax, memory_config=self.DRAM) if self.tp > 1 else x
        return x

    # ------------------------------------------------------------ tables
    def embedding(self, ids, table):
        self._tick(1)
        return self.ttnn.embedding(ids, table, layout=self.ttnn.TILE_LAYOUT, memory_config=self.DRAM)

    # ------------------------------------------------------------ the attention chain (tt_transformers' decode pattern)
    def _hs(self, w: int):
        ttnn = self.ttnn
        return ttnn.create_sharded_memory_config(shape=(TILE, w), core_grid=self.b32, strategy=ttnn.ShardStrategy.HEIGHT,
                                                 orientation=ttnn.ShardOrientation.ROW_MAJOR,
                                                 use_height_and_width_as_shard_shape=True)  # fmt: skip

    def create_heads(self, qkv, nh, nkv, hd):
        self._tick(1)
        return self.ttnn.experimental.nlp_create_qkv_heads_decode(qkv, num_heads=nh, num_kv_heads=nkv,
                                                                   memory_config=self._hs(hd))  # fmt: skip

    def head_norm(self, x, w, eps):
        ttnn = self.ttnn
        self._tick(3)
        mc = x.memory_config()
        y = ttnn.to_memory_config(x, self.L1, dtype=x.dtype)
        y = ttnn.rms_norm(y, epsilon=eps, weight=w, compute_kernel_config=self.ckc)
        return ttnn.to_memory_config(y, mc, dtype=y.dtype)

    def rot_mats(self, posu, cos_tab, sin_tab):
        ttnn = self.ttnn
        self._tick(6)
        out = []
        for tab in (cos_tab, sin_tab):
            e = ttnn.embedding(posu, tab, layout=ttnn.TILE_LAYOUT, memory_config=self.DRAM)  # [1, 32, hd]
            e = ttnn.transpose(ttnn.unsqueeze_to_4D(e), 1, 2)  # [1, 32, 1 (32), hd]
            out.append(ttnn.interleaved_to_sharded(e, self._hs(int(e.shape[-1]))))
        return out

    def rope(self, x, cos, sin):
        self._tick(1)
        return self.ttnn.experimental.rotary_embedding_hf(x, cos, sin, is_decode_mode=True)

    def cache_update(self, cache, kh, pos):
        self._tick(1)
        self.ttnn.experimental.paged_update_cache(cache["t"], kh, update_idxs_tensor=pos, page_table=cache["pt"])
        return cache

    def sdpa_split(self, q, Kc, Vc, cur_l, scale, sm, nh_l: int):
        raise NotImplementedError(
            "GLM53_DFLASH2_SHARD=1 on device: the split-K attention needs an attention program that emits the fp32 "
            "(m, l, O) partial of this chip's local rows for every head (the rung-3 M3 attention phase + the "
            "fabric AG / RS lanes of tt/tp_dsa_shard.py): not built -- the stock paged SDPA decode normalises its "
            "output and exposes no log-sum-exp (dflash2_intrace.md §10)"
        )

    def sdpa(self, q, Kc, Vc, cur, scale):
        self._tick(1)
        ttnn = self.ttnn
        # GLM53_DF2_HEADS_SHARDED: the output on the batch cores (height-sharded, as create_heads' q): not for GQA
        mem = self._hs(int(q.shape[-1])) if self.heads_sharded and int(Kc["t"].shape[1]) == 1 else self.DRAM
        return ttnn.transformer.paged_scaled_dot_product_attention_decode(
            q, Kc["t"], Vc["t"], page_table_tensor=Kc["pt"], cur_pos_tensor=cur, scale=scale,
            compute_kernel_config=self.hifi2, memory_config=mem)  # fmt: skip

    def concat_heads(self, a, nh):
        ttnn = self.ttnn
        if self.heads_sharded:  # the width-sharded L1 output is read in place by the o matmul's movers
            if a.is_sharded():
                self._tick(1)
                s = a
            else:
                self._tick(2)
                s = ttnn.to_memory_config(a, self._hs(int(a.shape[-1])))
            return ttnn.experimental.nlp_concat_heads_decode(s, num_heads=nh)
        self._tick(3)
        s = ttnn.to_memory_config(a, self._hs(int(a.shape[-1])))
        c = ttnn.experimental.nlp_concat_heads_decode(s, num_heads=nh)
        return ttnn.to_memory_config(c, self.DRAM)

    # ------------------------------------------------------------ top-k / selector
    def topk(self, x, k):
        self._tick(1)
        v, i = self.ttnn.topk(x, k=k, dim=-1, largest=True, sorted=True, memory_config=self.DRAM)
        return v, i

    def topk_rows(self, logits, K: int, k: int):
        """kernels/spec/df2_topk.cpp: logits [1, 1, 32, V] fp32 TILE -> (vals [1,1,32,32] fp32 TILE, ids [1,1,32,32] uint32
        TILE) of rows 1 .. K: SCAN on TOPK_CORES cores (column-tile ranges) + MERGE on one core (the persistent partials
        / vals / ids tensors of _topk_bufs, allocated before any capture)"""
        ttnn = self.ttnn
        from models.autoports.zai_org_glm_5_3_flash.tt.rt_guard import checked

        V = int(logits.shape[-1])
        tiles = V // TILE
        key = (V, K, k)
        bufs = getattr(self, "_topk_bufs", {})
        if key not in bufs:
            nc = TOPK_CORES
            bufs[key] = dict(
                partials=self.zeros((nc, K * k * 2), torch.int64, "rm"),
                vals=self.zeros((1, 1, TILE, TILE), FP32),
                ids=self.zeros((1, 1, TILE, TILE), torch.int64),
            )
            self._topk_bufs = bufs
        b = bufs[key]
        nc = TOPK_CORES
        allc = [ttnn.CoreCoord(x_, y_) for y_ in range(self.grid.y) for x_ in range(self.grid.x)][:nc]
        crs = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in allc])
        batch = 8
        ct = [K, k, nc, tiles, batch]
        src = ttnn.KernelDescriptor.SourceType.FILE_PATH
        dm0 = ttnn.DataMovementConfigDescriptor(
            processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.RISCV_0_default
        )
        page = K * k * 8
        # SCAN
        rt = ttnn.RuntimeArgs()
        for i, c in enumerate(allc):
            t0, t1 = i * tiles // nc, (i + 1) * tiles // nc
            rt[c.x][c.y] = [logits.buffer_address(), b["partials"].buffer_address(), t0, t1 - t0, i]
        kd = ttnn.KernelDescriptor(kernel_source=os.path.join(KDIR, "spec", "df2_topk.cpp"), source_type=src, core_ranges=crs,
                                   compile_time_args=ct, runtime_args=rt, defines=[("TOPK_MODE", "1"), ("DF2_TOPK_REV", str(DF2_TOPK_REV))],
                                   config=dm0)  # fmt: skip
        size = batch * 4096 + K * k * 8 + 64
        cb = ttnn.CBDescriptor(total_size=size, core_ranges=crs, format_descriptors=[
            ttnn.CBFormatDescriptor(buffer_index=0, data_format=ttnn.uint32, page_size=size)])  # fmt: skip
        self._tick(1, "topk_scan")
        ttnn.generic_op([logits, b["partials"]], checked(ttnn.ProgramDescriptor(kernels=[kd], semaphores=[], cbs=[cb])))
        # MERGE
        rt2 = ttnn.RuntimeArgs()
        rt2[0][0] = [b["partials"].buffer_address(), b["vals"].buffer_address(), b["ids"].buffer_address()]
        kd2 = ttnn.KernelDescriptor(kernel_source=os.path.join(KDIR, "spec", "df2_topk.cpp"), source_type=src, core_ranges=self.ctl_core,
                                    compile_time_args=ct, runtime_args=rt2, defines=[("TOPK_MODE", "2"), ("DF2_TOPK_REV", str(DF2_TOPK_REV))],
                                    config=dm0)  # fmt: skip
        size2 = nc * page + K * k * 8 + 2 * 4096 + 128
        cb2 = ttnn.CBDescriptor(total_size=size2, core_ranges=self.ctl_core, format_descriptors=[
            ttnn.CBFormatDescriptor(buffer_index=0, data_format=ttnn.uint32, page_size=size2)])  # fmt: skip
        self._tick(1, "topk_merge")
        ttnn.generic_op(
            [b["partials"], b["vals"], b["ids"]],
            checked(ttnn.ProgramDescriptor(kernels=[kd2], semaphores=[], cbs=[cb2])),
        )
        return b["vals"], b["ids"]

    def add_offset(self, ids, off):
        self._tick(2)
        return self.ttnn.add(
            self.ttnn.typecast(ids, self.ttnn.uint32, memory_config=self.DRAM), off, memory_config=self.DRAM
        )

    def gather(self, x, dim, idx):
        self._tick(1)
        return self.ttnn.gather(x, dim, idx, memory_config=self.DRAM)

    def to_rm(self, x):
        self._tick(1)
        return self.ttnn.to_layout(x, self.ttnn.ROW_MAJOR_LAYOUT)

    def prev_ids(self, bonus16, cand_rm):
        self._tick(2)
        ttnn = self.ttnn
        body = ttnn.slice(cand_rm, [1, 0], [31, int(cand_rm.shape[-1])], memory_config=self.DRAM)
        return ttnn.concat([bonus16, bonus16, body], dim=0, memory_config=self.DRAM)

    def selector_scores(self, hp, pred, succ, unary):
        ttnn = self.ttnn
        self._tick(5)
        hp3 = ttnn.reshape(hp, (32, 1, int(hp.shape[-1])))
        ph = ttnn.multiply(ttnn.typecast(pred, ttnn.float32, memory_config=self.DRAM), hp3, memory_config=self.DRAM)
        st = ttnn.transpose(ttnn.typecast(succ, ttnn.float32, memory_config=self.DRAM), -2, -1)
        sc = ttnn.matmul(ph, st, compute_kernel_config=self.ckc, memory_config=self.DRAM)  # [32, 16, 16]
        return ttnn.add(sc, ttnn.reshape(unary, (32, 1, int(unary.shape[-1]))), memory_config=self.DRAM)

    # ------------------------------------------------------------ the control kernels
    def _ctl(self, mode: int, K: int, mask_id: int, args: list, io: list, hd: int = 0):
        ttnn = self.ttnn
        from models.autoports.zai_org_glm_5_3_flash.tt.rt_guard import checked

        rt = ttnn.RuntimeArgs()
        rt[0][0] = [int(a) for a in args] + [0] * (24 - len(args))
        kd = ttnn.KernelDescriptor(
            kernel_source=os.path.join(KDIR, "spec", "df2_ctl.cpp"), source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=self.ctl_core, compile_time_args=[K, mask_id, hd], runtime_args=rt,
            defines=[("DF2_MODE", str(mode)), ("DF2_REV", str(DF2_REV))] + ([("DF2_SHARD", str(self.tp))] if self.shard else []),
            config=ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.RISCV_0_default))  # fmt: skip
        cb = ttnn.CBDescriptor(total_size=21056, core_ranges=self.ctl_core, format_descriptors=[
            ttnn.CBFormatDescriptor(buffer_index=0, data_format=ttnn.uint32, page_size=21056)])  # fmt: skip
        self._tick(1)
        ttnn.generic_op(io, checked(ttnn.ProgramDescriptor(kernels=[kd], semaphores=[], cbs=[cb])))

    def prep(self, st: "DFlash2State", K: int, mask_id: int, rope=None):
        pages = [st.tin, st.rec, st.ids, st.bonus16, st.bpos, st.bposu, st.cpos, st.cposu, st.cur]
        args = [p.buffer_address() for p in pages]
        hd = 0
        if rope is not None:  # rung 3: + the rows' cos / sin tiles from the tables (args 9..14, cap at 15)
            cos_tab, sin_tab, cosb, sinb, cosc, sinc, cap, cur_g = rope
            tiles = [cos_tab, sin_tab, cosb, sinb, cosc, sinc]
            args += [t.buffer_address() for t in tiles] + [int(cap), cur_g.buffer_address(), int(cur_g.shape[-1])]
            pages = pages + tiles + [cur_g]
            hd = int(cosb.shape[-1])
        if self.shard:  # the sequence-sharded cache: args 17..21 (kernels/spec/df2_ctl.cpp DF2_SHARD)
            args += [0] * (17 - len(args))
            sp = [st.rank, st.cpos_l, st.bpos_l, st.cur_l]
            args += [p.buffer_address() for p in sp] + [int(st.local_cap)]
            pages = pages + sp
        self._ctl(1, K, mask_id, args, pages, hd=hd)

    def mk_build(self, kind):
        """rung 3: the layer / ctx megakernel programs on this mesh (tt/dflash2_mk.Df2Mk)"""
        from models.autoports.zai_org_glm_5_3_flash.tt.dflash2_mk import Df2Mk

        return Df2Mk(self, kind.d, kind.S, nblk=kind.nblk, log=self.log)

    def select_walk(self, st: "DFlash2State", scores, cand_rm, K: int):
        io = [scores, cand_rm, st.tin, st.rec]
        self._ctl(2, K, 0, [p.buffer_address() for p in io], io)

    # ------------------------------------------------------------ the K / V cache write (kernels/spec/df2_cache.cpp)
    def cache_write(self, Kc, Vc, kh, vh, pos):
        """kh / vh [1, 32, nkv (32 padded), hd] bf16 height-sharded on the 32 batch cores (create_heads / RoPE outputs),
        pos [32] int32 -> the rows into the paged caches Kc / Vc at pos[j]: ONE one-core launch, the 32 rows written
        sequentially (paged_update_cache with 32 users on the same pages races its read-modify-writes: df2_cache.cpp)"""
        ttnn = self.ttnn
        from models.autoports.zai_org_glm_5_3_flash.tt.rt_guard import checked

        nblk, nkv, _, hd = (int(v) for v in Kc["t"].shape)
        ht = hd // TILE
        assert tuple(kh.shape)[1] == TILE and kh.is_sharded() and vh.is_sharded(), (kh.shape, kh.memory_config())
        cores = [ttnn.CoreCoord(x, y) for y in range(4) for x in range(8)]  # the b32 rectangle, row-major shard order
        nocs = [self.mesh.worker_core_from_logical_core(c) for c in cores]
        pt_bytes = nblk * 4
        args = [Kc["t"].buffer_address(), Vc["t"].buffer_address(), Kc["pt"].buffer_address(), pos.buffer_address(),
                kh.buffer_address(), vh.buffer_address(), pt_bytes] + [int(v) for c in nocs for v in (c.x, c.y)]  # fmt: skip
        rt = ttnn.RuntimeArgs()
        rt[0][0] = args
        kd = ttnn.KernelDescriptor(
            kernel_source=os.path.join(KDIR, "spec", "df2_cache.cpp"), source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=self.ctl_core, compile_time_args=[nkv, ht, TILE], runtime_args=rt,
            defines=[("DF2_CACHE_REV", str(DF2_CACHE_REV))] + ([("DF2_KV_BFP8", "1")] if self.kv_dtype == "bfp8" else []),
            config=ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.RISCV_0_default))  # fmt: skip
        size = 128 + ((pt_bytes + 63) & ~63) + (2 * TILE + 1) * ht * 2048 + 64
        cb = ttnn.CBDescriptor(total_size=size, core_ranges=self.ctl_core, format_descriptors=[
            ttnn.CBFormatDescriptor(buffer_index=0, data_format=ttnn.uint32, page_size=size)])  # fmt: skip
        self._tick(1)
        ttnn.generic_op(
            [Kc["t"], Vc["t"], Kc["pt"], pos, kh, vh],
            checked(ttnn.ProgramDescriptor(kernels=[kd], semaphores=[], cbs=[cb])),
        )

    # ------------------------------------------------------------ the rung-2 fused conv program (kernels/spec/df2_conv_*.cpp)
    def conv_fused(self, x, cf, base4, side: int, K: dict, res=None):
        """x [1,1,32,H] bf16, cf [1,1,32,4 NG] bf16, base4 [1,1,32,H] bf16 (rows 0..3 = base[side, tap]) -> [1,1,32,H]
        bf16: ONE generic_op over min(H / 32, grid) cores (column tiles balanced; the index math in df2_conv_common.hpp).
        K: the device constant tiles (DFlash2Kind: E [32 tiles], J [4], Sh [1]) + gpt / group.
        res [1,1,32,H] bf16 (DF2_CONV_RES, rung 3 M2 prep): x is the fp32 all-reduced partial; the program rounds it
        to bf16 (a pack / unpack pair) and returns res + conv(bf16(x)): the typecast + add launches folded in"""
        ttnn = self.ttnn
        from models.autoports.zai_org_glm_5_3_flash.tt.rt_guard import checked

        H = int(x.shape[-1])
        fold = res is not None
        assert (x.dtype == ttnn.float32) == fold, (x.dtype, fold)
        ncols = H // TILE
        ng, gpt = H // K["group"], K["gpt"]
        assert int(cf.shape[-1]) == 4 * ng and tuple(base4.shape) == (1, 1, TILE, H), (cf.shape, base4.shape, H)
        out = ttnn.empty((1, 1, TILE, H), ttnn.bfloat16, ttnn.TILE_LAYOUT, self.mesh, self.DRAM)
        cores = min(ncols, self.ncores, self.conv_cores)  # (GLM53_DF2_CONV_CORES: fewer, longer per-core column runs)
        allc = [ttnn.CoreCoord(x_, y_) for y_ in range(self.grid.y) for x_ in range(self.grid.x)][:cores]
        crs = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in allc])
        acc = lambda t: list(ttnn.TensorAccessorArgs(t).get_compile_time_args())  # noqa: E731
        E, J, Sh = K["E_t"], K["J_t"], K["Sh_t"]
        addrs = [t.buffer_address() for t in (x, cf, E, J, base4, Sh, out)]
        rt_r, rt_w, rt_c = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        for i, c in enumerate(allc):
            c0 = i * ncols // cores
            n = (i + 1) * ncols // cores - c0
            rt_r[c.x][c.y] = addrs + [c0, n, side] + ([res.buffer_address()] if fold else [])
            rt_w[c.x][c.y] = [addrs[6], c0, n]
            rt_c[c.x][c.y] = [n]
        src = ttnn.KernelDescriptor.SourceType.FILE_PATH
        defs = [("DF2_CONV_REV", str(DF2_CONV_REV))] + ([("DF2_CONV_RES", "1")] if fold else [])
        kd = lambda name, ct, rt, cfg: ttnn.KernelDescriptor(  # noqa: E731
            kernel_source=os.path.join(KDIR, "spec", f"df2_conv_{name}.cpp"), source_type=src, core_ranges=crs,
            compile_time_args=ct, runtime_args=rt, defines=defs, config=cfg)  # fmt: skip
        cc = ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
                                          fp32_dest_acc_en=True)  # fmt: skip
        if fold:  # the fp32 partial's tiles unpack exactly into the fp32 DST
            modes = [ttnn.UnpackToDestMode.Default] * 64
            modes[8] = ttnn.UnpackToDestMode.UnpackToDestFp32  # CB_X32 (df2_conv_common.hpp)
            cc.unpack_to_dest_mode = ttnn._ttnn.program_descriptor.VectorUnpackToDestMode(modes)
        kernels = [
            kd("reader", [gpt, ng] + acc(x) + acc(cf) + acc(E) + acc(J) + acc(base4) + acc(Sh) + (acc(res) if fold else []), rt_r,
               ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.RISCV_0_default)),
            kd("writer", acc(out), rt_w,
               ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_1, noc=ttnn.NOC.RISCV_1_default)),
            kd("compute", [], rt_c, cc),
        ]  # fmt: skip
        page = 2048  # bf16 tile
        cb = lambda i, n: ttnn.CBDescriptor(total_size=page * n, core_ranges=crs, format_descriptors=[  # noqa: E731
            ttnn.CBFormatDescriptor(buffer_index=i, data_format=ttnn.bfloat16, page_size=page)])  # fmt: skip
        # CB_X 0, CB_CF 1, CB_E 2, CB_J 3, CB_B 4, CB_SH 5, CB_ZERO 6, CB_OUT 7 (df2_conv_common.hpp)
        cbs = [cb(0, 2), cb(1, 4), cb(2, 4), cb(3, 2), cb(4, 2), cb(5, 1), cb(6, 1), cb(7, 2)]
        io = [x, cf, E, J, base4, Sh, out]
        if fold:  # CB_X32 (fp32, 4-KiB pages), CB_RES (bf16)
            cbs.append(ttnn.CBDescriptor(total_size=4096 * 2, core_ranges=crs, format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=8, data_format=ttnn.float32, page_size=4096)]))  # fmt: skip
            cbs.append(cb(9, 2))
            io.append(res)
        self._tick(1, "conv_fused_res" if fold else "conv_fused")
        ttnn.generic_op(io, checked(ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs)))
        return out


# ====================================================================== device state of one speculating user
class DFlash2State:
    """the pages the step reads / writes (DRAM, replicated): the token page `tin` [32, 64] u32 and the record `rec`
    [1, 64] u32 are SpecMTP's (shared); the rest are the drafter's own"""

    def __init__(self, ops: Ops, tin, rec):
        i32 = torch.int32
        self.tin, self.rec = tin, rec
        self.ids = ops.zeros((1, 32), torch.int64, "rm")  # u32: the block's token ids (row 0 bonus, 1..K mask)
        self.bonus16 = ops.zeros((1, 16), torch.int64, "rm")  # u32: bonus x 16 (the selector's row-1 predecessors)
        self.bpos = ops.zeros((32,), i32, "rm")  # the block rows' positions c' + j (the cache write / cur_pos source)
        self.bposu = ops.zeros((1, 32), torch.int64, "rm")  # the same as u32 (the RoPE row gather)
        self.cpos = ops.zeros((32,), i32, "rm")  # this step's verify rows' positions c + j (the context write)
        self.cposu = ops.zeros((1, 32), torch.int64, "rm")
        self.cur = ops.zeros((32,), i32, "rm")  # the block's attention bound c' + K (every row)
        # the sequence-sharded cache (DFlash2Kind.shard, add_shard): the rows' LOCAL positions on this chip (rows
        # another chip owns -> the scratch rows local_cap + j), the local bound, the chip's rank page
        self.cpos_l = self.bpos_l = self.cur_l = self.rank = None
        self.local_cap = 0

    def add_shard(self, ops: Ops, local_cap: int):
        i32 = torch.int32
        self.cpos_l, self.bpos_l, self.cur_l = (ops.zeros((32,), i32, "rm") for _ in range(3))
        self.rank = ops.shards([torch.full((1, 32), r, dtype=i32) for r in range(ops.tp)], i32, "rm")
        self.local_cap = int(local_cap)


class DFlash2Kind:
    """the drafter kind: build once (weights resident), then ctx_program() / block_program() per step (the calls the
    trace captures), prepare() once before the first capture (eager compile)"""

    def __init__(self, ops: Ops, cfg: dict, w: dict, *, K: int, cap: int, tin, rec, tap_mode: str = "fold",
                 conv: str = "fused", keep_last: bool = True, conv_res: bool | None = None, layer: str | None = None,
                 log=print):  # fmt: skip
        d = self.d = Dims(cfg)
        assert 1 <= K <= d.K, (K, d.K)
        self.ops, self.K, self.log, self.tap_mode = ops, K, log, tap_mode
        self.cap = -(-(cap + TILE) // TILE) * TILE  # positions up to c + 31 are written (garbage rows)
        tp = ops.tp
        # the 1M form (tt/dflash2_1m.py): the cache dtype and the sequence shard (the ops layer mode; the mk programs
        # take the local pages the same way once the attention partial exists)
        self.kv_dtype = getattr(ops, "kv_dtype", "bf16")
        self.shard = bool(getattr(ops, "shard", False))
        self.sm = None
        self.cap_cache = self.cap  # positions the cache holds (local + the scratch page under the shard)
        if self.shard:
            from models.autoports.zai_org_glm_5_3_flash.tt.dflash2_1m import DrafterShardMap

            self.sm = DrafterShardMap(tp)
            self.cap_cache = self.sm.local_cap(self.cap) + TILE
        self.nblk = self.cap_cache // TILE
        self.pt_shared = None  # ONE fixed page table for every layer (ops mode): [32, nblk] int32
        # rung 3 (GLM53_DF2_LAYER=mk, tt/dflash2_mk.py): the layers / ctx as megakernel programs; the matmul units of
        # the fabric phases (o, down, fc) are bf16 (the lanes pack bf16), fc is K-split per chip, the K / V caches
        # are per kv head (the SDPA layout trick), PREP writes the rows' cos / sin tiles
        self.layer_mode = layer or getattr(ops, "layer_mode", None) or os.environ.get("GLM53_DF2_LAYER", "ops")
        assert self.layer_mode in LAYER_MODES, self.layer_mode
        mk = self.layer_mode == "mk"
        assert not (self.shard and mk), "GLM53_DFLASH2_SHARD=1 runs the ops layer mode (the mk heads phase: later)"
        self.S = S = DFlash2Shards(cfg, w, tp, tap_mode=tap_mode, ksplit=mk, kv_all=self.shard)
        # "ops": the stock-op conv (10 launches per call); "fused": the rung-2 program (1 launch; the SAME fp32
        # arithmetic: proved on the host backend by tests/s17/test_dflash2_kind_host.py, on device by Stage 17's unit)
        self.conv_mode = conv
        assert conv in CONV_MODES, conv
        # rung 3 M2 prep (GLM53_DF2_CONV_RES=1, fused mode only): the two "finish" convs of a layer take the fp32
        # all-reduced partial and add the residual inside the program (kernels/spec DF2_CONV_RES): -32 launches / step
        self.conv_res = conv == "fused" and (
            conv_res if conv_res is not None else os.environ.get("GLM53_DF2_CONV_RES", "0") == "1"
        )
        nh_l, nkv_l, nkv_c, hd = S.nh_l, S.nkv_l, S.nkv_c, d.hd
        self.scale = 1.0 / math.sqrt(hd)
        row = lambda v: ops.rep(v.reshape(1, 1, 1, -1))  # noqa: E731
        # matmuls (launch units: mm_stream programs, or DS reader-phase programs in GLM53_DF2_MM=ds with the
        # same-input units grouped into one launch) and the replicated small tensors
        self.fc = ops.mm(S.fc_k if mk else S.fc, name="fc")
        self.lm_head = ops.mm(S.lm_head, out=FP32, name="lm_head")
        self.hproj = ops.mm([S.hproj] * tp, out=FP32, name="hproj")
        self.head_mm = ops.mm_group([self.lm_head, self.hproj], name="head")  # both of the final norm's output
        self.hidden_norm, self.norm = row(S.hidden_norm), row(S.norm)
        self.E = ops.rep(S.E)
        self.Sh = ops.rep(S.Sh.reshape(1, 1, TILE, TILE), FP32)
        # the fused conv's selection tiles (bf16, exact 0 / 1): E [32 tiles], J [4 tiles], Sh [1 tile]
        self.KC = KC = conv_consts(d.group)
        KC["E_t"] = ops.rep(KC["E"].reshape(1, 1, TILE * TILE, TILE))
        KC["J_t"] = ops.rep(KC["J"].reshape(1, 1, 4 * TILE, TILE))
        KC["Sh_t"] = ops.rep(KC["Sh"].reshape(1, 1, TILE, TILE))
        self.L = []
        for i, Ly in enumerate(S.layers):
            self.L.append(dict(
                qkv=ops.mm(Ly["qkv"], name=f"qkv{i}"), o=ops.mm(Ly["o"], out=BF16 if mk else FP32, name=f"o{i}"),
                ug=ops.mm(Ly["ug"], name=f"ug{i}"), down=ops.mm(Ly["down"], out=BF16 if mk else FP32, name=f"down{i}"),
                kp_attn=ops.mm([Ly["kp_attn"]] * tp, name=f"kpa{i}"), kp_mlp=ops.mm([Ly["kp_mlp"]] * tp, name=f"kpm{i}"),
                base_attn=[ops.rep(Ly["base_attn"][j].reshape(1, 1, 1, -1), FP32) for j in range(4)],
                base_mlp=[ops.rep(Ly["base_mlp"][j].reshape(1, 1, 1, -1), FP32) for j in range(4)],
                base4_attn=ops.rep(Ly["base4_attn"].reshape(1, 1, TILE, -1)), base4_mlp=ops.rep(Ly["base4_mlp"].reshape(1, 1, TILE, -1)),
                in_ln=row(Ly["in_ln"]), post_ln=row(Ly["post_ln"]), q_norm=row(Ly["q_norm"]), k_norm=row(Ly["k_norm"]),
                # the drafter's paged K / V cache of this layer: [blocks, nkv_l, 32, hd] bf16; every page-table row =
                # the user's blocks in order (one speculating user: a fixed table)
                Kc=self._cache(nkv_c, hd, True), Vc=self._cache(nkv_c, hd, False)))  # fmt: skip
            if not mk:
                self.L[-1]["Vc"]["pt"] = self.L[-1]["Kc"]["pt"]
            log(f"dflash2 kind: layer {i} resident")
        # the ctx program's 8 qkv projections take the same input (hidden_norm's output): one launch in "ds" mode
        self.ctx_mm = ops.mm_group([Ly["qkv"] for Ly in self.L], name="ctx_qkv")
        # the matmul launch units in program order (the breakdown job replays them alone on dummy inputs of the
        # real inputs' specs, recorded by mm_probe = {} while the programs run): [(name, unit)]
        self.mm_units = dict(
            ctx=[("fc", self.fc), ("ctx_qkv", self.ctx_mm)],
            block=[(f"{n}{i}", Ly[k]) for i, Ly in enumerate(self.L)
                   for n, k in (("kpa", "kp_attn"), ("qkv", "qkv"), ("o", "o"), ("kpm", "kp_mlp"), ("ug", "ug"), ("down", "down"))]
            + [("head", self.head_mm)],
        )  # fmt: skip
        self.mm_probe = None  # dict: unit name -> the input's TensorSpec, filled by _mm (device only)
        self.embed = ops.shards(S.embed, BF16, "rm")  # [V, H / tp] per chip (row-major: ttnn.embedding)
        self.pred_cb, self.succ_cb = ops.rep(S.pred_cb, BF16, "rm"), ops.rep(S.succ_cb, BF16, "rm")
        # the per-chip vocab offsets added to the top-k ids ([32, 16] from ttnn.topk; the kernel's [32, 32] tile)
        self.topk_mode = os.environ.get("GLM53_DF2_TOPK", "ttnn")
        assert self.topk_mode in TOPK_MODES, self.topk_mode
        vo_w = TILE if self.topk_mode == "kernel" else d.top_k
        self.vocab_off = ops.shards(
            [torch.full((1, 1, TILE, vo_w), o, dtype=torch.int32) for o in S.vocab_off], torch.int64
        )
        cos, sin = rope_cos_sin(torch.arange(self.cap), d.theta, hd)
        self.cos_tab, self.sin_tab = ops.rep(cos.to(BF16), BF16, "rm"), ops.rep(sin.to(BF16), BF16, "rm")
        # rung 3: the rows' cos / sin tiles (cosb, sinb: the block rows' positions; cosc, sinc: this step's rows'),
        # written by PREP (seed_ctx writes cosc / sinc from the host); the layer / ctx programs
        self.rope_tiles = [ops.zeros((1, 1, TILE, hd)) for _ in range(4)] if mk else None
        # rung 3: the attention bound as a [1, G] page (the SDPA's batch rows = the kv groups' q heads)
        self.cur_g = ops.zeros((1, nh_l), torch.int32, "rm") if mk else None  # (one SDPA launch: batch = q head)
        self.mk = ops.mk_build(self) if mk else None
        # the tap slots: the verify's completed residual on the tap layers (bf16); "fold": the raw 4 streams [wv, 4H]
        wv = TILE
        self.tap_w = 4 * d.H if tap_mode == "fold" else d.H
        self.taps = [ops.zeros((1, 1, wv, self.tap_w)) for _ in range(d.n_taps)]
        self.quarter = ops.rep(torch.full((1, 1, 1, 1), 0.25), FP32)  # tap mode "mean": the stream mean's 1/4
        self.st = DFlash2State(ops, tin, rec)
        if self.shard:
            self.st.add_shard(ops, self.cap_cache - TILE)
        self.counters = dict(ctx=0, block=0)
        # the block's outputs a reader wants after a replay (u1: the final hidden, the candidates, the top-k values):
        # PERSISTENT buffers allocated here, BEFORE any capture, and copied into at the end of block_program (3
        # launches; GLM53_DF2_KEEP_LAST=0 / keep_last=False drops them). A capture-time tensor kept on the kind
        # (the first version held the op outputs themselves) is refused by the trace-allocation guard of every later
        # checked replay ('device buffer(s) still alive before trace replay': Stage 17 u2 / u3, 2026-10-22).
        self.keep_last = keep_last
        self.last = dict(hid=ops.zeros((1, 1, TILE, d.H)), cand=ops.zeros((TILE, d.top_k), torch.int64, "rm"),
                         unary=ops.zeros((1, 1, TILE, d.top_k), FP32)) if keep_last else None  # fmt: skip
        # stage probes (tests/s17/test_dflash2_anchors_host.py): None = off; a dict collects name -> tensor (ctx
        # stages: a list per name, one per call). On a device the handles are CAPTURE-TIME tensors: usable with
        # unchecked replays only (the guard refuses them) -- a diagnosis mode, never the served construction.
        self.probe = None
        # every weight is on the device (or the mock) now: the host shard lists (11.3 GiB at TP8) go, unless a
        # diagnosis wants them (GLM53_DF2_KEEP_HOST=1); the torch backend's "device" tensors ARE these tensors: kept
        if isinstance(ops, TtnnOps) and os.environ.get("GLM53_DF2_KEEP_HOST", "0") != "1":
            S.release()
            import gc

            gc.collect()
            log(
                f"dflash2 host shards released: resident {S.bytes_per_chip()['total'] / 2**20:.0f} MiB / chip on device"
            )

    def _p(self, name: str, t, ctx: bool = False):
        if self.probe is not None:
            if ctx:
                self.probe.setdefault(name, []).append(t)
            else:
                self.probe[name] = t
        return t

    def _mm(self, name: str, unit, x):
        """one matmul launch unit on x (the probe records the input's spec for the breakdown job's dummies)"""
        if self.mm_probe is not None and name not in self.mm_probe:
            self.mm_probe[name] = x.spec
        return unit(x)

    def _cache(self, nkv_l: int, hd: int, is_k: bool):
        """a layer's K or V cache: mk = one [blocks, 1, 32, hd] bf16 tensor per kv head (the SDPA layout trick, the
        owners' row writes); ops = the [blocks, nkv_l, 32, hd] paged cache + the fixed page table (K; V shares it)"""
        ops = self.ops
        if self.layer_mode == "mk":
            if getattr(
                ops, "merged_cache", False
            ):  # the device: ONE tensor, kv head g's blocks at [g nblk, (g + 1) nblk)
                return [ops.cache_zeros((nkv_l * self.nblk, 1, TILE, hd))]
            return [ops.cache_zeros((self.nblk, 1, TILE, hd)) for _ in range(nkv_l)]
        if is_k and self.pt_shared is None:
            self.pt_shared = self._pt()
        return dict(t=ops.cache_zeros((self.nblk, nkv_l, TILE, hd)), pt=self.pt_shared if is_k else None)

    def _pt(self):
        ops = self.ops
        pt = torch.arange(self.nblk, dtype=torch.int32).reshape(1, -1).repeat(TILE, 1)
        return ops.shards([pt] * ops.tp, torch.int32, "rm")

    # ------------------------------------------------------------ the taps (called by SpecMTP's verify on the tap layers)
    def tap(self, slot: int, res_flat):
        """res_flat: the verify width's flat residual [1, 1, wv, 4 x 4096] fp32 (the 4 mHC streams side by side) ->
        the slot (bf16; "mean": the stream mean by ops, 9 launches)"""
        ops, d = self.ops, self.d
        if self.tap_mode == "fold":
            src = res_flat
        else:
            parts = [ops.slice(res_flat, 3, s * d.H, (s + 1) * d.H) for s in range(4)]
            # the 1/4 scalar is a PERSISTENT device constant (self.quarter): tap() runs inside the captured step, where a
            # host upload is not replayed and its freed memory is reused by later ops (garbage on replay)
            src = ops.mul(ops.add(ops.add(parts[0], parts[1]), ops.add(parts[2], parts[3])), self.quarter)
        self._copy(ops.pad_rows(ops.typecast(src, BF16), TILE), self.taps[slot])

    def _copy(self, src, dst):
        ops = self.ops
        if isinstance(ops, TorchOps):
            for s, t in zip(src, dst):
                t.copy_(s.reshape(t.shape).to(t.dtype))
        else:
            ops._tick(1, "tap_copy")
            ops.ttnn.copy(src, dst)

    # ------------------------------------------------------------ programs
    def _conv(self, x, cf, side: int, base: list, base4=None):
        """x [1, 1, 32, H] bf16, cf = kernel_projection(..) [1, 1, 32, 4 ng] bf16 -> bf16 (fp32 math as the
        reference: coef_t = base_t + E-expanded delta_t; out = coef_0 x + coef_1 x_prev) -- 10 launches ("ops") or
        the one fused program ("fused": base4 = the layer's [1,1,32,H] base tile row)"""
        ops, d = self.ops, self.d
        if self.conv_mode == "fused":
            return ops.conv_fused(x, cf, base4, side, self.KC)
        return self._conv_ops(x, cf, side, base)

    def _conv_finish(self, x32, cf, base: list, base4, res):
        """the "finish" conv (side 1) of the fp32 all-reduced partial x32, added to the residual: res + conv(bf16(x32))
        -- the fold (one program, DF2_CONV_RES) or typecast + conv + add (the rung-2 chain, 3 launches in fused mode)"""
        ops = self.ops
        if self.conv_res:
            return ops.conv_fused(x32, cf, base4, 1, self.KC, res)
        return ops.add(res, self._conv(ops.typecast(x32, BF16), cf, 1, base, base4))

    def _conv_ops(self, x, cf, side: int, base: list):
        ops, d = self.ops, self.d
        xf = ops.typecast(x, FP32)
        xp = ops.shift_rows(xf, self.Sh)
        out = None
        for t, src in ((0, xf), (1, xp)):
            o = side * 2 * d.ng + t * d.ng
            coef = ops.linear_bias(ops.slice(cf, 3, o, o + d.ng), self.E, base[side * 2 + t], FP32)
            term = ops.mul(coef, src)
            out = term if out is None else ops.add(out, term)
        return ops.typecast(out, BF16)

    def _kv_write(self, Ly, kh, vh, posi, cos, sin):
        ops, d = self.ops, self.d
        k = ops.rope(ops.head_norm(kh, Ly["k_norm"], d.eps), cos, sin)
        ops.cache_write(Ly["Kc"], Ly["Vc"], k, vh, posi)  # one launch; NOT paged_update_cache (df2_cache.cpp)
        return k

    def _wpos(self, which: str):
        """the cache-write position page: global positions, or this chip's LOCAL pages under the sequence shard"""
        st = self.st
        if which == "ctx":
            return st.cpos_l if self.shard else st.cpos
        return st.bpos_l if self.shard else st.bpos

    def _attend(self, q, Ly):
        ops, st = self.ops, self.st
        if self.shard:
            return ops.sdpa_split(q, Ly["Kc"], Ly["Vc"], st.cur_l, self.scale, self.sm, self.S.nh_l)
        return ops.sdpa(q, Ly["Kc"], Ly["Vc"], st.cur, self.scale)

    def ctx_program(self):
        """this step's verify rows' taps -> the context K / V of every layer at positions c + j"""
        ops, d, S, st = self.ops, self.d, self.S, self.st
        x = ops.concat(self.taps, 3) if d.n_taps > 1 else self.taps[0]
        if self.mk is not None:  # rung 3: ONE program (fc K-split -> all-reduce -> hidden_norm -> 8 x qkv -> 8 x heads)
            cosc, sinc = self.rope_tiles[2], self.rope_tiles[3]
            self.mk.ctx(x, self.fc, self.L, cosc, sinc, st.cpos, self.KC, koff_tiles=S.fc_in // ops.tp // TILE)
            self.counters["ctx"] += 1
            return
        nrm = self._p(
            "ctx.nrm", ops.rms(ops.all_gather(self._mm("fc", self.fc, x), 3), self.hidden_norm, d.eps), ctx=True
        )
        cos, sin = ops.rot_mats(st.cposu, self.cos_tab, self.sin_tab)
        qkvs = self._mm("ctx_qkv", self.ctx_mm, nrm)  # the 8 layers' q|k|v of the same rows (one launch in "ds")
        for i, Ly in enumerate(self.L):
            q, k, v = ops.create_heads(qkvs[i], S.nh_l, S.nkv_c, d.hd)  # (q unused: one fused program)
            self._p(f"ctx.L{i}.v", v, ctx=True)
            self._p(f"ctx.L{i}.k", self._kv_write(Ly, k, v, self._wpos("ctx"), cos, sin), ctx=True)
        self.counters["ctx"] += 1

    def prep_program(self):
        """after SpecCtl ACCEPT / BOOT: the step's pages from the record (kernels/spec/df2_ctl.cpp DF2_PREP)"""
        rope = None if self.mk is None else (self.cos_tab, self.sin_tab, *self.rope_tiles, self.cap, self.cur_g)
        self.ops.prep(self.st, self.K, self.d.mask_id, rope=rope)

    def step(self):
        """the drafter's part of a speculative step: PREP -> ctx (this step's verify rows) -> block"""
        self.prep_program()
        self.ctx_program()
        self.block_program()

    def boot(self):
        """the drafter's part of the bootstrap (after a plain step / prefill, the context already seeded): no ctx"""
        self.prep_program()
        self.block_program()

    def block_program(self):
        """the block [bonus, mask x K] -> drafts into token rows 1..K (one trace section; PREP ran before)"""
        ops, d, S, st, K = self.ops, self.d, self.S, self.S and self.st, self.K
        e = ops.all_gather(ops.embedding(st.ids, self.embed), 2)  # [1, 32, H]
        res = self._p("emb", ops.reshape(e, (1, 1, TILE, d.H)))  # the residual: every layer adds to it in place
        if self.mk is not None:  # rung 3: the layers as the megakernel programs + the SDPA per layer
            return self._block_head(self._block_mk(res), st, K)
        cos, sin = ops.rot_mats(st.bposu, self.cos_tab, self.sin_tab)
        for i, Ly in enumerate(self.L):
            P = lambda n, t: self._p(f"L{i}.{n}", t)  # noqa: E731
            M = lambda n, unit, t: self._mm(f"{n}{i}", unit, t)  # noqa: E731
            x = P("x_ln", ops.rms(res, Ly["in_ln"], d.eps))
            cf = P("cf", M("kpa", Ly["kp_attn"], x))
            x = P("conv_a", self._conv(x, cf, 0, Ly["base_attn"], Ly["base4_attn"]))
            q, k, v = ops.create_heads(M("qkv", Ly["qkv"], x), S.nh_l, S.nkv_c, d.hd)
            P("v", v)
            q = P("q", ops.rope(ops.head_norm(q, Ly["q_norm"], d.eps), cos, sin))
            P("k", self._kv_write(Ly, k, v, self._wpos("block"), cos, sin))
            a = P("att", self._attend(q, Ly))
            o32 = ops.all_reduce(M("o", Ly["o"], ops.concat_heads(a, S.nh_l)))  # fp32 partials summed
            res = P("res", self._conv_finish(o32, cf, Ly["base_attn"], Ly["base4_attn"], res))
            x = P("x_post", ops.rms(res, Ly["post_ln"], d.eps))
            cf2 = P("cf2", M("kpm", Ly["kp_mlp"], x))
            x = P("conv_m", self._conv(x, cf2, 0, Ly["base_mlp"], Ly["base4_mlp"]))
            y32 = ops.all_reduce(M("down", Ly["down"], ops.swiglu(M("ug", Ly["ug"], x))))
            # the FFN's finish conv + the residual add (the next layer's input / the final norm's): one fold
            res = P("res2", self._conv_finish(y32, cf2, Ly["base_mlp"], Ly["base4_mlp"], res))
        return self._block_head(self._p("hid", ops.rms(res, self.norm, d.eps)), st, K)

    def _block_mk(self, e):
        """rung 3 (tt/dflash2_mk.py): A_0 on the embedding rows, then per layer the SDPA (per kv head) and the merged
        [B_i + A_{i+1}] program; the last program ends with the final norm -> hid"""
        st, mk, L = self.st, self.mk, self.L
        cosb, sinb = self.rope_tiles[0], self.rope_tiles[1]
        mk.layer_a0(L[0], e, cosb, sinb, st.bpos, self.KC)
        res_in = e
        for i, Ly in enumerate(L):
            att = mk.attention(Ly, self.cur_g, self.scale)
            mk.layer_ba(i, Ly, L[i + 1] if i + 1 < len(L) else None, att, res_in, cosb, sinb, st.bpos, self.KC)
            res_in = mk.res
        return self._p("hid", mk.hid)

    def _block_head(self, hid, st, K):
        """the final hidden -> lm_head / top-16 / the selector walk -> token rows 1..K (the stock-op chain), in three
        parts (the breakdown job's pieces): _head_mm, _head_topk, _head_sel"""
        logits, hp = self._head_mm(hid)
        vals, ids = self._head_topk(logits)
        return self._head_sel(vals, ids, hp, st, K, hid)

    def _head_mm(self, hid):
        """lm_head (this chip's vocab slice) and the selector's hidden projection of the same rows (one launch in "ds")"""
        return self._mm("head", self.head_mm, hid)

    def _head_topk(self, logits):
        """the per-chip top-16: ttnn.topk over the [32, V] fp32 logits, or (GLM53_DF2_TOPK=kernel) the exact top-16 of
        the draft rows 1 .. K by the two-launch kernel -> vals / ids [1,1,32,16 | 32]"""
        if self.topk_mode == "kernel":
            return self.ops.topk_rows(logits, self.K, self.d.top_k)
        return self.ops.topk(logits, self.d.top_k)

    def _head_sel(self, vals, ids, hp, st, K, hid=None):
        """the global top-16 (exact: every global entry is in its slice's top-16) -> the selector -> the walk"""
        ops, d = self.ops, self.d
        vals_all = ops.all_gather(vals, 3)
        ids_all = ops.all_gather(ops.add_offset(ids, self.vocab_off), 3)
        unary, j = ops.topk(vals_all, d.top_k)
        cand = ops.gather(ids_all, 3, j)  # [1, 1, 32, 16] u32: global vocab ids
        cand_rm = ops.reshape(ops.to_rm(cand), (TILE, d.top_k))
        prev = ops.prev_ids(st.bonus16, cand_rm)
        self._p("hp", hp)  # [1, 1, 32, rank] fp32
        scores = ops.selector_scores(
            ops.reshape(hp, (TILE, d.rank)),
            ops.embedding(prev, self.pred_cb),
            ops.embedding(cand_rm, self.succ_cb),
            ops.reshape(unary, (TILE, d.top_k)),
        )  # fmt: skip  [32, 16, 16] fp32
        self._p("scores", scores)
        ops.select_walk(st, scores, cand_rm, K)
        self.counters["block"] += 1
        if self.keep_last:  # into the persistent buffers (no capture-time tensor survives this function)
            for name, src in (("hid", hid), ("cand", cand_rm), ("unary", unary)):
                if src is not None:
                    self._copy(src, self.last[name])

    # ------------------------------------------------------------ host seeding (the pair unit: the prompt's context)
    def seed_ctx(self, taps_mean: torch.Tensor, p0: int):
        """host-side context seeding for positions p0 .. (the prefill's taps; mean over the streams already taken):
        rows of 32 through the ctx program with the tap slots and the position pages written by the host (NOT part of
        a step; the pair unit's prompt context, as the prototype's ctx trace)"""
        ops, d, st = self.ops, self.d, self.st
        n = int(taps_mean.shape[0])
        # rung 3: the ctx program's all-reduce carries rows 0..7 only (K + 1 <= 8 valid rows per step), so the seeding
        # goes 8 rows per call; a chunk's garbage rows 8..31 land on the next chunk's positions and are rewritten
        step = 8 if self.mk is not None else TILE
        for r0 in range(0, n, step):
            rows = taps_mean[r0 : r0 + step]
            m = int(rows.shape[0])
            for s in range(d.n_taps):
                slot = torch.zeros(TILE, self.tap_w)
                v = rows[:, s].float()
                slot[:m] = v.repeat(1, 4) if self.tap_mode == "fold" else v
                ops.write(self.taps[s], slot.reshape(1, 1, TILE, -1))
            pos = torch.arange(p0 + r0, p0 + r0 + TILE, dtype=torch.int32)
            if m < TILE:  # rows past the prompt: positions that the next steps overwrite before any read
                pos[m:] = p0 + r0 + m + torch.arange(TILE - m)
            ops.write(st.cpos, pos)
            ops.write(st.cposu, pos.reshape(1, TILE))
            if self.shard:  # the rows' local positions per chip (PREP's DF2_SHARD arithmetic, from the host)
                ops.write_shards(st.cpos_l, [self.sm.local_pages_of(pos, r, st.local_cap) for r in range(ops.tp)])
            if self.mk is not None:  # the rows' cos / sin tiles PREP would have written
                cos, sin = rope_cos_sin(pos.clamp(0, self.cap - 1), d.theta, d.hd)
                ops.write(self.rope_tiles[2], cos.to(BF16).reshape(1, 1, TILE, -1))
                ops.write(self.rope_tiles[3], sin.to(BF16).reshape(1, 1, TILE, -1))
            self.ctx_program()


# the Apache-2.0 drafter (canada-quant/GLM-5.3-Flash-DFlash2-G: config.json, model.safetensors, mask_embedding.pt):
# the hosts' local copies, the first that exists unless GLM53_DF2_DIR names one
DF2_DIRS = ("/home/alex/glm53/scratch-big/17/dflash2/drafter", "/home/alex/glm53/work-17-d0/dflash2/drafter")


def build_dflash2_kind(mesh, *, K: int, cap: int, tin, rec, log=print) -> DFlash2Kind:
    """SpecMTP's builder: GLM53_DF2_DIR (the Apache-2.0 drafter's directory: config.json, model.safetensors,
    mask_embedding.pt), GLM53_DF2_WEIGHTS=safetensors|random (random: the real config with random values -- the mocks
    / censuses), GLM53_DF2_TAP_MODE=fold|mean, GLM53_DF2_CONV=fused|ops (the rung-2 conv program / the stock chain),
    GLM53_DF2_KEEP_LAST=1|0 (the persistent hid / cand / unary copies a reader takes after a replay: 3 launches);
    rung 3 M1 (TtnnOps): GLM53_DF2_MM=stream|ds, GLM53_DF2_MM_CORES, GLM53_DF2_CONV_CORES (tt/dflash2_mm.py); the
    folds GLM53_DF2_CONV_RES=1 (the finish convs add the residual: -32 launches) and GLM53_DF2_HEADS_SHARDED=1 (the
    attention output stays on the batch cores: -16 at TP8, -8 at TP2); the 1M form (tt/dflash2_1m.py):
    GLM53_DF2_KV_DTYPE=bf16|bfp8 (the cache tiles), GLM53_DFLASH2_SHARD=1 (the sequence-sharded cache)"""
    from models.autoports.zai_org_glm_5_3_flash.tt import dflash2_ref as DR

    d = os.environ.get("GLM53_DF2_DIR") or next((p for p in DF2_DIRS if os.path.isdir(p)), DF2_DIRS[0])
    cfg = DR.hf_config(os.path.join(d, "config.json"))
    # GLM53_DF2_WEIGHTS: "safetensors" (the served default: the Apache-2.0 drafter's bf16 weights read on the host and
    # uploaded as ordinary DRAM from_torch uploads -- next to a restored image they are real uploads, exactly as the
    # MTP layer's tile-cache uploads are; an image BUILT with the drafter records them), "meta" (the restore of such an
    # image: shapes from the safetensors header, no host read; tp_image replays every meta upload from the image),
    # "random" (the real config with random values: mocks / censuses)
    # SERVED DEFAULTS (coordinator 2026-10-23 ~01:40Z): the silicon-proven rung-2 path -- GLM53_DF2_LAYER=ops,
    # GLM53_DF2_MM=stream, GLM53_DF2_CONV=fused, no GLM53_DF2_CONV_RES / HEADS_SHARDED folds (identity 16 / 16, forced
    # rollbacks + DSA state bitwise at TP2 on 43ba510172; +27 ms / step at TP2, 380 launches). The rung-3 megakernel
    # path (GLM53_DF2_LAYER=mk, 48 launches) computes garbage on silicon as of this note and stays behind its knob
    # until its numerics pass; the DS matmuls / folds (M1) are selectable the same way. Every knob below reads its
    # default from this function, so a served launch sets only GLM53_SPEC_KIND=dflash2 GLM53_SPEC_K=K (K=4 today).
    mode = os.environ.get("GLM53_DF2_WEIGHTS", "safetensors")
    if mode == "random":
        w = DR.random_weights(cfg, seed=0)
    elif mode == "meta":
        w = DR.meta_weights(d)
    else:
        assert mode == "safetensors", mode
        w = DR.load_weights(d)
    log(f"dflash2 drafter: {d} weights={mode} K={K} cap={cap}")
    ops = TtnnOps(mesh, log=log)
    return DFlash2Kind(ops, cfg, w, K=K, cap=cap, tin=tin, rec=rec, tap_mode=os.environ.get("GLM53_DF2_TAP_MODE", "fold"),
                       conv=os.environ.get("GLM53_DF2_CONV", "fused"), layer=os.environ.get("GLM53_DF2_LAYER", "ops"),
                       keep_last=os.environ.get("GLM53_DF2_KEEP_LAST", "1") == "1", log=log)  # fmt: skip
