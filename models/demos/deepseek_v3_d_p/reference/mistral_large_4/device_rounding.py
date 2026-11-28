# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""The CPU reference's weights rounded as the device stores them, to separate weight-format error from compute error.

A bfloat8_b / bfloat4_b tensor shares one 8-bit exponent per 16 consecutive values of a tile face row, i.e. 16
consecutive values along the last dim of the tensor as passed to ``ttnn.from_torch``. Which 16 values share an exponent
therefore depends on the orientation the device module hands to from_torch:

    "nk"  the checkpoint's [out, in] as stored, read with transpose_b: groups run along the contraction (in)
          (attention wq / wkv_b / wo, the dense MLP and shared expert, the LM head)
    "kn"  transposed to [in, out] first: groups run along the output dim
          (the routed experts, tt/mistral_large_4/experts.py; the indexer's wq_b)

``round_like_device`` returns the round trip in the checkpoint orientation. impl "ttnn" is literally
``ttnn.to_torch(ttnn.from_torch(w_oriented, dtype, TILE_LAYOUT))`` on the host (no device, but ttnn's host conversion
reads the HAL, so it needs a host with Tenstorrent hardware or a simulator). impl "torch" is a vectorized emulation of
tt_metal/impl/data_format/blockfloat_common.cpp convert_u32_to_bfp (max exponent per group, significand aligned to it,
round to nearest even at the format's width, saturate, exponent-zero inputs flushed) and its decode; the Rounder checks
it bit-exact against ttnn on the first tensor of every tag.
"""

from __future__ import annotations

import math
import time

import torch

BFP_BITS = {"bfloat8_b": 7, "bfloat4_b": 3, "bfloat2_b": 1}  # magnitude bits per value
NO_ROUNDING = ("bfloat16", "float32")  # the reference already holds its weights in bf16
GROUP = 16
BLOCK = 1 << 22  # values per emulation block (bounds the int32 temporaries to ~16 MB each)


def _round_groups(v: torch.Tensor, bits: int) -> torch.Tensor:
    """v: fp32 [A, 16, B], groups along dim 1 -> the BFP round trip, fp32."""
    u = v.view(torch.int32)
    exp = (u >> 23) & 0xFF
    shared = exp.amax(dim=1, keepdim=True)
    man = ((u & 0x7FFFFF) | 0x800000) >> (shared - exp).clamp_(max=31)
    shift = 24 - bits
    rest = man & ((1 << shift) - 1)
    q = man >> shift
    tie = 1 << (shift - 1)
    q += ((rest > tie) | ((rest == tie) & ((q & 1) == 1))).to(torch.int32)
    q.clamp_(max=(1 << bits) - 1).masked_fill_(exp == 0, 0)
    # value = q * 2^(shared - 127 - (bits - 1)): q has at most `bits` significant bits, exact in fp32 and bf16
    val = torch.ldexp(q.float(), (shared - (127 + bits - 1)).float())
    return torch.where(u < 0, -val, val)


def bfp_round(x: torch.Tensor, bits: int, dim: int = -1) -> torch.Tensor:
    """Round x as a BFP_B format with `bits` magnitude bits, one shared exponent per 16 consecutive values along
    `dim` (the from_torch tensor's last dim). Returns fp32 of x's shape."""
    x = x.float().contiguous()
    d = dim % x.dim()
    n = x.shape[d]
    assert n % GROUP == 0, f"dim {d} of {tuple(x.shape)} is not a multiple of {GROUP} (tile padding not emulated)"
    pre = math.prod(x.shape[:d]) * (n // GROUP)
    suf = math.prod(x.shape[d + 1 :])
    v = x.view(pre, GROUP, suf)
    out = torch.empty_like(v)
    step = max(1, BLOCK // (GROUP * suf))
    for a in range(0, pre, step):
        out[a : a + step] = _round_groups(v[a : a + step], bits)
    return out.view(x.shape)


def ttnn_round_trip(x: torch.Tensor, dtype_name: str) -> torch.Tensor:
    """ttnn's own host conversion of x (as passed to from_torch) to dtype_name and back, fp32 of x's shape."""
    import ttnn

    t = ttnn.from_torch(x.float().contiguous(), dtype=getattr(ttnn, dtype_name), layout=ttnn.TILE_LAYOUT)
    return ttnn.to_torch(t).float().reshape(x.shape)


def round_like_device(w: torch.Tensor, dtype_name: str, layout: str, impl: str = "ttnn") -> torch.Tensor:
    """w in checkpoint orientation ([out, in], or any shape for "nk") -> its device rounding, same orientation, fp32."""
    if dtype_name in NO_ROUNDING:
        return w.float()
    assert dtype_name in BFP_BITS, dtype_name
    assert layout in ("nk", "kn"), layout
    if layout == "kn":
        assert w.dim() == 2, f"kn layout needs a 2-D weight, got {tuple(w.shape)}"
    if impl == "ttnn":
        return ttnn_round_trip(w.t(), dtype_name).t() if layout == "kn" else ttnn_round_trip(w, dtype_name)
    assert impl == "torch", impl
    # "kn": the groups of w.t()'s last dim are runs of 16 along w's dim 0; rounded in place, no transposed copy
    return bfp_round(w, BFP_BITS[dtype_name], dim=0 if layout == "kn" else -1)


class Rounder:
    """round_like_device with a cross-check and timing. impl "torch" is compared bit-exact with ttnn on the first tensor
    of each tag (a layer), and raises on any difference."""

    def __init__(self, impl: str = "ttnn", check: bool = True):
        self.impl, self.check = impl, check
        self.checked: set = set()
        self.calls, self.seconds, self.checks = 0, 0.0, 0

    def __call__(self, w: torch.Tensor, dtype_name: str, layout: str, tag=None) -> torch.Tensor:
        if dtype_name in NO_ROUNDING:
            return w.float()
        t0 = time.time()
        r = round_like_device(w, dtype_name, layout, self.impl)
        self.calls += 1
        self.seconds += time.time() - t0
        if self.impl == "torch" and self.check and (tag, dtype_name, layout) not in self.checked:
            self.checked.add((tag, dtype_name, layout))
            ref = round_like_device(w, dtype_name, layout, "ttnn")
            if not torch.equal(r, ref):
                bad = (r != ref).sum().item()
                raise RuntimeError(f"torch BFP emulation != ttnn for {tag} {dtype_name} {layout}: {bad} values differ")
            self.checks += 1
        return r

    def stats(self) -> dict:
        return {"impl": self.impl, "calls": self.calls, "seconds": round(self.seconds, 1), "ttnn_checks": self.checks}


# ---------------------------------------------------------------------------------------- the routed experts
class _RoundedLayer:
    """A LayerExperts (reference/mistral_large_4/weights.py) whose weight() returns the device rounding of each
    expert matrix. Device layout (experts.py TtExperts.__init__): expert_weight(e, m).to(bf16).t() -> [in, out], i.e.
    "kn". A rounded matrix is kept (bf16, exact) while the store's dequant budget lasts, else rounded on each use."""

    def __init__(self, le, layer: int, dtype_name: str, rounder: Rounder):
        self.le, self.layer, self.dtype_name, self.rounder = le, layer, dtype_name, rounder
        self.w = le.w  # experts_forward reads the shapes from here
        self.cache = le.__dict__.setdefault("_rounded", {})

    def weight(self, e: int, m: str, dtype: torch.dtype, out: torch.Tensor | None = None) -> torch.Tensor:
        from models.demos.deepseek_v3_d_p.reference.mistral_large_4.weights import dequant

        key = (e, m, self.dtype_name)
        w = self.cache.get(key)
        if w is None:
            le = self.le
            w = self.rounder(dequant(le.w[(e, m)], le.s[(e, m)], torch.float32), self.dtype_name, "kn", self.layer)
            if le.store is not None and le.store.reserve(2 * w.numel()):
                w = self.cache[key] = w.to(torch.bfloat16)
        return w.to(dtype) if out is None else out.copy_(w)


class RoundedExperts:
    """ExpertStore stand-in for Reference.experts: get(i, next_layer) as the store, with the experts rounded."""

    def __init__(self, store, dtype_name: str, rounder: Rounder):
        self.store, self.dtype_name, self.rounder = store, dtype_name, rounder

    def get(self, i: int, next_layer: int | None = None):
        return _RoundedLayer(self.store.get(i, next_layer=next_layer), i, self.dtype_name, self.rounder)


# ---------------------------------------------------------------------------------------- the dense weights
def dense_plan(w: dict, get) -> dict[str, tuple[str, str]]:
    """One reference layer's dense weights (Reference.w[i]) -> {key: (device dtype, layout)}, from the device modules
    with the settings ``get`` (tt/mistral_large_4/settings.py settings.get) in effect. Weights the device keeps in
    bf16 / fp32 are left out."""
    owns_kv = "wkv_a" in w
    shared_dt = get("ATTN_SHARED_KV_WEIGHTS")
    plan = {
        # attention.py: wq bf8 on a KV-owning layer, else ATTN_SHARED_KV_WEIGHTS; wo always bf8; all [N, K]
        "wq": ("bfloat8_b" if owns_kv else shared_dt, "nk"),
        "wo": ("bfloat8_b", "nk"),
        # the absorbed wkv_b halves [heads, nope | v, lora]: groups along lora, the checkpoint's last dim
        "wkv_b": (get("ATTN_WKV_B_DTYPE") if owns_kv else shared_dt, "nk"),
    }
    if owns_kv:
        plan["wkv_a"] = (get("ATTN_WKV_A_DTYPE"), "nk")
    if "idx_wq_b" in w:
        # indexer.py: wq_b and wk transposed to [H, out] before from_torch; weights_proj stays bf16
        plan["idx_wq_b"] = ("bfloat8_b", "kn")
        plan["idx_wk"] = (get("INDEXER_WK_DTYPE"), "kn")
    if "gate" in w:
        plan["gate"] = (get("ROUTER_WEIGHT_DTYPE"), "nk")  # router.py: [E, H], transpose_b
    for p in ("", "shared_"):  # mlp.py: the dense MLP and the shared expert, bf8, [N, K]
        for m in ("w1", "w2", "w3"):
            if p + m in w:
                plan[p + m] = ("bfloat8_b", "nk")
    return {k: v for k, v in plan.items() if v[0] not in NO_ROUNDING}


def round_dense(ref, rounder: Rounder, get) -> dict[str, int]:
    """Round every dense weight of a Reference in place (wkv_b in place, so its w_uk / w_uv views follow), and the LM
    head (model.py: bf8, [V, H], transpose_b). Returns {dtype/layout: tensors rounded}."""
    done: dict[str, int] = {}
    for i, w in ref.w.items():
        for key, (dt, layout) in dense_plan(w, get).items():
            t = w[key]
            t.copy_(rounder(t, dt, layout, tag=f"L{i}").view(t.shape))
            done[f"{dt}/{layout}"] = done.get(f"{dt}/{layout}", 0) + 1
    head = ref.lm_head
    head.copy_(rounder(head, "bfloat8_b", "nk", tag="lm_head"))
    done["bfloat8_b/nk"] = done.get("bfloat8_b/nk", 0) + 1
    return done
