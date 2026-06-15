# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Chunked-prefill KERNEL sweep (NO-PATCH committed-base op) for the 12 pi05 projections.

Canonical methodology of deep-work/unified_matmul_3stage_kerneltime.md:
  * Metric = DEVICE KERNEL DURATION (tt-perf-report col 20) via extract_perf.py
    (EXTRACT_MODE=mmsweep METRIC=KERNEL; MMSWEEP_OP=native|mmd).
  * One (stage, proj, config) per tracy subprocess -> one signpost region per repeat.
  * N_ITERS forwards per region; N_REPEAT repeat regions for the min-of-5 frozen-min.
  * PCC >= threshold vs torch via assert_pcc BEFORE timing (PCC-fail -> skip timing).
  * Residency evidence: buffer_address of the L1-resident weight stable across chunks.

Configs (selected by CFG env, one per subprocess):
  native        ttnn.linear, full M one op (op-code MatmulDeviceOperation)
  mmd_full      MatmulDecodeLinearNoMSplit force_partial=False, T=M, resident weight
  mmd_partial   MatmulDecodeLinearNoMSplit force_partial=True,  T=M, resident weight
  native_T<T>   ttnn.linear called M/T times (weight re-streamed), summed KERNEL
  mmd_full_T<T> mmd full, M/T calls reusing resident weight
  mmd_partial_T<T> mmd partial, M/T calls reusing resident weight

ONLY committed-base op kwargs are passed (partial_width_sharded / dtype /
compute_kernel_config); _matmul_decode_kwargs returns {} on the committed op (no
stream_k token in the docstring). NOFIT/INVALID is recorded when a factory rejects
a shape (planner assert/RuntimeError or device op error) -- NO silent fallback.

Run (one config at a time):
  CFG=mmd_partial ONLY_STAGE=VLM ONLY_PROJ=gate python -m tracy -p -r -v \
    --op-support-count 200000 \
    -m 'pytest profile_chunked_prefill_nopatch_stages.py::test_profile -x -s'
"""
from __future__ import annotations

import math
import os
import sys

import torch
import ttnn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _lib.pcc_utils import assert_pcc, compute_pcc  # noqa: E402
from _lib.matmul_decode_linear_no_m_split import (  # noqa: E402
    MatmulDecodeLinearNoMSplit,
)

try:
    from tracy import signpost
except Exception:
    def signpost(*a, **k):
        pass

TT_METAL_COMMIT = "e4500c1fae97c103b16fc24fc7010b852992a9e6"
SEED = 1234
BF8 = ttnn.bfloat8_b
BF16 = ttnn.bfloat16

N_ITERS = int(os.environ.get("N_ITERS", "5"))
N_REPEAT = int(os.environ.get("N_REPEAT", "5"))   # min-of-5 frozen-min
CFG = os.environ.get("CFG", "native")
ONLY_STAGE = os.environ.get("ONLY_STAGE", "")
ONLY_PROJ = os.environ.get("ONLY_PROJ", "")
PCC_THRESHOLD = float(os.environ.get("PCC_THRESHOLD", "0.99"))

# Matched compute kernel config (HiFi4 + fp32 DST), used by BOTH native and mmd legs
# so the comparison is apples-to-apples (same precision the frozen table used).
_CKC = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4,
    fp32_dest_acc_en=True,
    packer_l1_acc=False,
    math_approx_mode=False,
)

# (stage, M, [(proj, K, N, weight_dtype, pad_n, pad_k)]) -- the 12 golden projections.
STAGES = {
    "SigLIP": (256, [
        ("qkv", 1152, 4608, BF16, None, None),
        ("o", 1536, 1152, BF16, None, None),
        ("fc1", 1152, 4304, BF8, 4320, None),
        ("fc2", 4304, 1152, BF8, None, 4320),
    ]),
    "VLM": (288, [
        ("qkv", 2048, 2560, BF16, None, None),
        ("o", 2048, 2048, BF16, None, None),
        ("gate", 2048, 16384, BF16, None, None),
        ("up", 2048, 16384, BF16, None, None),
        ("down", 16384, 2048, BF8, None, None),
    ]),
    "DENOISE": (64, [
        ("gate", 1024, 4096, BF16, None, None),
        ("up", 1024, 4096, BF16, None, None),
        ("down", 4096, 1024, BF8, None, None),
    ]),
}

# Valid chunked-T per stage (skip non-dividing).
VALID_T = {"SigLIP": (32, 64), "VLM": (32, 96), "DENOISE": (32,)}


def _weight(K, N):
    torch.manual_seed(SEED)
    return (torch.randn(K, N) * 0.02).to(torch.bfloat16)


def _act(M, K):
    torch.manual_seed(SEED + 1)
    return (torch.randn(M, K) * 0.5).to(torch.bfloat16)


class _NativeLinear:
    """Full-M single ttnn.linear (MatmulDeviceOperation)."""

    def __init__(self, dev, w_KN, *, weight_dtype, pad_n, pad_k, role):
        self.device = dev
        self.role = role
        self.K_orig, self.N_orig = int(w_KN.shape[0]), int(w_KN.shape[1])
        w = w_KN
        if pad_n and pad_n > self.N_orig:
            w = torch.nn.functional.pad(w, (0, pad_n - self.N_orig))
        if pad_k and pad_k > self.K_orig:
            w = torch.nn.functional.pad(w, (0, 0, 0, pad_k - self.K_orig))
        self.K, self.N = int(w.shape[0]), int(w.shape[1])
        self.weight = ttnn.from_torch(w, dtype=weight_dtype, layout=ttnn.TILE_LAYOUT,
                                      device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    def __init__(self, dev, w_KN, *, weight_dtype, pad_n, pad_k, role, t_rows=None):
        self.device = dev
        self.role = role
        self.t_rows = int(t_rows) if t_rows else None
        self.K_orig, self.N_orig = int(w_KN.shape[0]), int(w_KN.shape[1])
        w = w_KN
        if pad_n and pad_n > self.N_orig:
            w = torch.nn.functional.pad(w, (0, pad_n - self.N_orig))
        if pad_k and pad_k > self.K_orig:
            w = torch.nn.functional.pad(w, (0, 0, 0, pad_k - self.K_orig))
        self.K, self.N = int(w.shape[0]), int(w.shape[1])
        self.weight = ttnn.from_torch(w, dtype=weight_dtype, layout=ttnn.TILE_LAYOUT,
                                      device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    def _linear_one(self, x_dev):
        if x_dev.shape[-1] != self.K:
            x_dev = ttnn.pad(x_dev, [(0, 0)] * (len(x_dev.shape) - 1)
                             + [(0, self.K - x_dev.shape[-1])], 0.0)
        y = ttnn.linear(x_dev, self.weight, dtype=BF16,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG,
                        compute_kernel_config=_CKC)
        yshape = list(y.shape)
        if yshape[-1] != self.N_orig:
            start = [0] * len(yshape)
            end = yshape[:-1] + [self.N_orig]
            y = ttnn.slice(y, start, end)
        return y

    def __call__(self, x_dev):
        # Full-M single op, or T-row chunked (weight re-streamed each chunk: native
        # ttnn.linear folds the DRAM weight read into each call -> no residency).
        orig = list(x_dev.shape)
        K_in = orig[-1]
        M = orig[0] if len(orig) == 2 else math.prod(orig[:-1])
        if self.t_rows is None or M <= self.t_rows:
            return self._linear_one(x_dev)
        x2d = ttnn.reshape(x_dev, [M, K_in]) if len(orig) != 2 else x_dev
        T = self.t_rows
        n = math.ceil(M / T)
        outs = []
        for c in range(n):
            r0, r1 = c * T, min((c + 1) * T, M)
            x_c = ttnn.slice(x2d, [r0, 0], [r1, K_in])
            outs.append(self._linear_one(x_c))
            ttnn.deallocate(x_c)
        if len(orig) != 2 and x2d is not x_dev:
            ttnn.deallocate(x2d)
        y = outs[0] if len(outs) == 1 else ttnn.concat(outs, dim=0)
        if len(outs) > 1:
            for t in outs:
                ttnn.deallocate(t)
        if len(orig) >= 3:
            y = ttnn.reshape(y, list(orig[:-1]) + [self.N_orig])
        return y


def _build(dev, stage, M, proj, K, N, wd, pn, pk):
    """Return (callable, tag, kind) for CFG, or raise to mark NOFIT/INVALID."""
    role = f"{stage.lower()}_{proj}"
    w = _weight(K, N)
    # parse chunked T from CFG suffix
    t_rows = None
    base = CFG
    if "_T" in CFG:
        base, tstr = CFG.rsplit("_T", 1)
        t_rows = int(tstr)
    if base == "native":
        return _NativeLinear(dev, w, weight_dtype=wd, pad_n=pn, pad_k=pk,
                             role=role, t_rows=t_rows), CFG
    force_partial = base.endswith("partial")
    # MatmulDecodeLinearNoMSplit: m_rows=M (full sequence), t_rows=T for chunked.
    mmd = MatmulDecodeLinearNoMSplit(
        dev, w, bias=None, out_dtype=BF16, weight_dtype=wd,
        role=role, force_partial=force_partial, pad_n=pn, pad_k=pk,
        m_rows=M, t_rows=t_rows, fp32_dest_acc=True, resident_weight=True,
        full_seq_mode=(t_rows is None))
    return mmd, CFG


def _torch_ref(w_KN, x_MK, N_orig):
    return (x_MK.to(torch.float32) @ w_KN.to(torch.float32))[:, :N_orig]


def _run(dev, stage, M, proj, K, N, wd, pn, pk):
    role = f"{stage.lower()}_{proj}"
    print(f"\n==== CFG={CFG} {stage}.{proj} M={M} K={K} N={N} dtype={wd} ====", flush=True)
    w = _weight(K, N)
    x = _act(M, K)
    ref = _torch_ref(w, x, N)

    try:
        mod, tag = _build(dev, stage, M, proj, K, N, wd, pn, pk)
    except Exception as e:
        print(f"RESULT {stage}.{proj} CFG={CFG} : NOFIT-BUILD :: {type(e).__name__}: {e}",
              flush=True)
        return

    if hasattr(mod, "describe"):
        try:
            d = mod.describe()
            print(f"  plan: mode={d.get('mode')} n_chunks={d.get('n_chunks')} "
                  f"G={d.get('k_split_G')} npc={getattr(mod,'npc','?')} "
                  f"a_cores={d.get('a_cores')} t_knob={d.get('t_knob_active')} "
                  f"t_rows={d.get('t_rows')} m_tiles={d.get('m_tiles')}", flush=True)
        except Exception:
            pass

    def _mk_x():
        return ttnn.from_torch(x.reshape(1, M, K), dtype=BF16,
                               layout=ttnn.TILE_LAYOUT, device=dev)

    # ---- PCC gate (before any timing). Fresh input per forward (wrapper consumes it). ----
    try:
        x_dev = _mk_x()
        y = mod(x_dev)
        y_t = ttnn.to_torch(y).reshape(M, -1)[:, :N].to(torch.float32)
        ttnn.deallocate(y)
    except Exception as e:
        print(f"RESULT {stage}.{proj} CFG={CFG} : INVALID-RUN :: {type(e).__name__}: {e}",
              flush=True)
        return
    res = compute_pcc(y_t, ref)
    pcc = res[0][0]
    print(f"  PCC={pcc:.6f} (thr={PCC_THRESHOLD})", flush=True)
    if not (pcc >= PCC_THRESHOLD):
        ttnn.deallocate(x_dev)
        print(f"RESULT {stage}.{proj} CFG={CFG} : PCC-FAIL pcc={pcc:.6f}", flush=True)
        return

    def _snap():
        out = []
        cache = getattr(mod, "_b_l1_cache", None)
        if not cache:
            return out
        for t in list(cache.values()):
            try:
                if t.is_allocated():
                    out.append(t.buffer_address())
            except Exception:
                pass
        return sorted(out)

    # ---- residency evidence (mmd resident-weight buffer_address stable) ----
    addrs_fwd1 = _snap()
    if hasattr(mod, "_b_l1_cache"):
        print(f"  RESIDENT buffer_addrs after fwd1: {addrs_fwd1}", flush=True)

    # ---- timing: warm-up OUTSIDE signpost, then N_REPEAT regions of N_ITERS forwards ----
    # Fresh input per forward (the wrapper consumes/aliases its input), matching the
    # canonical profile_unified_mmsweep_stages harness.
    xw = _mk_x(); ttnn.deallocate(mod(xw)); ttnn.deallocate(xw)  # warm-up OUTSIDE signpost
    ttnn.synchronize_device(dev)
    addrs_before = _snap()
    for rep in range(N_REPEAT):
        tag = f"MMD_{stage}_{proj}_r{rep}"
        signpost(header=tag)
        for _ in range(N_ITERS):
            xi = _mk_x()
            yy = mod(xi)
            ttnn.deallocate(yy)
            ttnn.deallocate(xi)
        ttnn.synchronize_device(dev)
    addrs_after = _snap()
    if addrs_before:
        stable = addrs_before == addrs_after and len(addrs_before) > 0
        print(f"  RESIDENT stable across reps: {stable} "
              f"(before={addrs_before} after={addrs_after})", flush=True)
    print(f"RESULT {stage}.{proj} CFG={CFG} : TIMED pcc={pcc:.6f}", flush=True)


def test_profile(dev):
    for stage, (M, projs) in STAGES.items():
        if ONLY_STAGE and stage != ONLY_STAGE:
            continue
        for proj, K, N, wd, pn, pk in projs:
            if ONLY_PROJ and proj != ONLY_PROJ:
                continue
            _run(dev, stage, M, proj, K, N, wd, pn, pk)
