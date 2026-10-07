# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Round 3 eltwise binary (#58818 review): bit dumps of the caller kernels taken this round, PR kernels against main's (the
opt-in define removed by dump_kedit.sh; save then compare, as test_eb_dump_kedit). Every bf16 pattern enters each op where
its input space allows:

- layer_norm / rms_norm pre_all_gather (layernorm_pre_allgather.cpp, _2d.cpp): x one-hot rows (each row one pattern, so the
  row sums show x*x and x exactly), with a residual (x + res per element), and dense rows with and without the specials;
- fused_recurrent_gated_delta_rule: q = k = 0, so the final state is the bcast-scalar product S * exp(g), S every pattern;
  plus random q, k, v on a finite S;
- rotary_embedding_hf: x every pattern (64 heads, shifted), cos and sin the special and normal set;
- indexer_score_dsa: k one-hot rows (each key one pattern), q all ones, so relu(q.k) is the pattern; the gates w are the
  special and normal set (the column-broadcast multiply);
- scale_mask_softmax with a padded last tile (the dest-reuse add of the padding mask): rows of patterns."""
import os
import time

import numpy as np
import pytest
import torch
import ttnn

from eb_dump_lib import PATS, b16_set, b16_small, bf16_from_bits, cls_bits, f32_of_bf16, kernel_variants, out_bits, stage

STAGE = os.environ.get("EB_DUMP_STAGE", "save")


@pytest.fixture(scope="module")
def device():
    dev = ttnn.CreateDevice(device_id=0, l1_small_size=24576)
    yield dev
    kernel_variants(f" {STAGE}")
    ttnn.close_device(dev)


def _dev(x_bits_or_f32, shape, device, dtype=ttnn.bfloat16):
    if dtype == ttnn.float32:
        t = torch.from_numpy(np.ascontiguousarray(x_bits_or_f32, dtype=np.float32)).reshape(shape)
    else:
        t = bf16_from_bits(np.asarray(x_bits_or_f32).reshape(-1)).reshape(shape)
    return ttnn.from_torch(t, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)


def _ckc(device, fid, fp32):
    return ttnn.init_device_compute_kernel_config(
        device.arch(), math_fidelity=getattr(ttnn.MathFidelity, fid), math_approx_mode=False, fp32_dest_acc_en=fp32, packer_l1_acc=False
    )


def _bits_all(res):
    if isinstance(res, (tuple, list)):
        return np.concatenate([out_bits(t).astype(np.uint32).reshape(-1) for t in res if t is not None])
    return out_bits(res)


def _guard(tag, fn):
    try:
        fn()
    except Exception as e:  # noqa: BLE001
        print(f"\nDUMP {tag}: NOT RUN ({type(e).__name__}: {str(e).splitlines()[0][:300] if str(e) else ''})", flush=True)


FINITE = PATS[cls_bits(PATS) <= 2]
CKC = [("HiFi4", True), ("HiFi4", False), ("HiFi2", True), ("LoFi", False)]


def _ln_data(arr, res):
    """x and residual (bf16 bits) as (rows, W)."""
    B = b16_set()
    nb = B.size
    if arr.startswith("onehot"):
        W = int(arr[6:])
        reps = 4 if res else 1
        R = 65536 * reps
        r = np.arange(R)
        col = (r * 7) % W
        x = np.zeros((R, W), dtype=np.uint16)
        x[r, col] = PATS[r % 65536]
        rr = np.zeros((R, W), dtype=np.uint16)
        if res:
            rr[r, col] = B[(r + (r // 65536) * 131) % nb]
        return x, rr
    src = FINITE if arr == "densefin" else PATS
    W = 32
    n = -(-src.size // (32 * W)) * 32 * W
    x = np.resize(src, n).reshape(-1, W)
    rr = B[np.arange(x.size) % nb].reshape(x.shape) if res else np.zeros_like(x)
    return x, rr


LN = [(op, arr, res) for op in ("layer", "rms") for arr in ("onehot32", "onehot256", "densefin", "denseall") for res in (False, True)]


@pytest.mark.parametrize("op, arr, res", LN, ids=[f"{o}-{a}-{'res' if r else 'nores'}" for o, a, r in LN])
def test_ln_pre(device, op, arr, res):
    t0 = time.time()
    x, rr = _ln_data(arr, res)
    shape = (1, 1) + x.shape
    tx = _dev(x, shape, device)
    tr = _dev(rr, shape, device) if res else None
    fn = ttnn.layer_norm_pre_all_gather if op == "layer" else ttnn.rms_norm_pre_all_gather
    for fid, fp32 in CKC:
        tag = f"lnpre_{op}_{arr}_{'res' if res else 'nores'}_{fid}_{'d32' if fp32 else 'd16'}"

        def run():
            kw = dict(dtype=ttnn.bfloat16, compute_kernel_config=_ckc(device, fid, fp32), memory_config=ttnn.DRAM_MEMORY_CONFIG)
            if res:
                kw["residual_input_tensor"] = tr
            stage(tag, out_bits(fn(tx, **kw)), extra=f"(input {shape}, {time.time() - t0:.1f} s)")

        _guard(tag, run)


@pytest.mark.parametrize("arr", ["onehot256", "densefin", "denseall"])
def test_ln_pre_2d(device, arr):
    """rms_norm_pre_all_gather with use_2d_core_grid=True (layernorm_pre_allgather_2d.cpp): one-hot rows of 8 tiles, and dense
    rows of 64 tiles (the 2D grid splits the width over up to 8 cores; one tile of width leaves no worker cores)."""
    t0 = time.time()
    x, _ = _ln_data(arr, False)
    shape = (1, 1) + x.shape if arr.startswith("onehot") else (1, 1, x.size // 2048, 2048)
    tx = _dev(x, shape, device)
    for fid, fp32 in CKC:
        tag = f"lnpre2d_{arr}_{shape[2]}x{shape[3]}_{fid}_{'d32' if fp32 else 'd16'}"
        _guard(tag, lambda: stage(tag, out_bits(ttnn.rms_norm_pre_all_gather(
            tx, dtype=ttnn.bfloat16, compute_kernel_config=_ckc(device, fid, fp32), memory_config=ttnn.DRAM_MEMORY_CONFIG,
            use_2d_core_grid=True)), extra=f"({time.time() - t0:.1f} s)"))


def test_frgdn(device):
    """S [1, 32, 128, 128] fp32 holds every bf16 pattern 8 times (shifted per copy); q = k = 0 makes the final state
    S * exp(g) per head; four g sets cover the special and normal set. Then random q, k, v, beta on the finite patterns."""
    t0 = time.time()
    H, K, V = 32, 128, 128
    B = b16_set()
    bf = f32_of_bf16(B)
    s = np.concatenate([np.roll(f32_of_bf16(PATS), 4099 * c) for c in range(8)]).reshape(1, H, K, V)
    with np.errstate(all="ignore"):
        gvals = np.log(np.abs(bf.astype(np.float64))).astype(np.float32)
    gsets = [gvals[(np.arange(H) * 17 + j) % gvals.size] for j in range(4)]
    gsets.append(np.array([-np.inf, np.inf, np.nan, 0.0] * 8, dtype=np.float32))
    z = np.zeros((1, 1, H, K), np.float32)
    ts = _dev(s, (1, H, K, V), device, ttnn.float32)
    for fid in ("HiFi4", "HiFi2", "LoFi"):
        for gi, g in enumerate(gsets):
            tag = f"frgdn_qk0_g{gi}_{fid}"

            def run():
                q = _dev(z, (1, 1, H, K), device, ttnn.float32)
                tv = _dev(np.ones((1, 1, H, V), np.float32), (1, 1, H, V), device, ttnn.float32)
                tg = _dev(g, (1, 1, H), device, ttnn.float32)
                tb = _dev(np.full((1, 1, H), 0.5, np.float32), (1, 1, H), device, ttnn.float32)
                res = ttnn.transformer.fused_recurrent_gated_delta_rule(
                    q, q, tv, tg, tb, scale=K**-0.5, initial_state=ts, output_final_state=True,
                    compute_kernel_config=_ckc(device, fid, True))
                stage(tag, _bits_all(res), extra=f"({time.time() - t0:.1f} s)")

            _guard(tag, run)
        tag = f"frgdn_random_{fid}"

        def run_r():
            rng = np.random.default_rng(5)
            sf = np.resize(f32_of_bf16(FINITE), H * K * V).astype(np.float32)
            sf = np.where(np.abs(sf) > 1e4, np.float32(0.5), sf).reshape(1, H, K, V)
            q = _dev(rng.standard_normal((1, 1, H, K)).astype(np.float32) * 0.1, (1, 1, H, K), device, ttnn.float32)
            k = _dev(rng.standard_normal((1, 1, H, K)).astype(np.float32) * 0.1, (1, 1, H, K), device, ttnn.float32)
            v = _dev(rng.standard_normal((1, 1, H, V)).astype(np.float32), (1, 1, H, V), device, ttnn.float32)
            tg = _dev(-np.abs(rng.standard_normal((1, 1, H))).astype(np.float32), (1, 1, H), device, ttnn.float32)
            tb = _dev(rng.uniform(0, 1, (1, 1, H)).astype(np.float32), (1, 1, H), device, ttnn.float32)
            res = ttnn.transformer.fused_recurrent_gated_delta_rule(
                q, k, v, tg, tb, scale=K**-0.5, initial_state=_dev(sf, (1, H, K, V), device, ttnn.float32),
                output_final_state=True, compute_kernel_config=_ckc(device, fid, True))
            stage(tag, _bits_all(res), extra=f"({time.time() - t0:.1f} s)")

        _guard(tag, run_r)


def test_rope_hf(device):
    """x [1, 64, 1024, 64]: head h holds every pattern rolled by 1031 h; cos and sin [1, 1, 1024, 64] the special and
    normal set (two arrangements)."""
    t0 = time.time()
    Hh, S, D = 64, 1024, 64
    B = b16_set()
    x = np.stack([np.roll(PATS, 1031 * h) for h in range(Hh)]).reshape(1, Hh, S, D)
    i = np.arange(S * D)
    cos = B[i % B.size].reshape(1, 1, S, D)
    sin = B[(i * 7 + 3) % B.size].reshape(1, 1, S, D)
    tx, tc, tsn = _dev(x, x.shape, device), _dev(cos, cos.shape, device), _dev(sin, sin.shape, device)
    for fid, fp32 in CKC:
        tag = f"rope_hf_{fid}_{'d32' if fp32 else 'd16'}"
        _guard(tag, lambda: stage(tag, out_bits(ttnn.experimental.rotary_embedding_hf(
            tx, tc, tsn, is_decode_mode=False, compute_kernel_config=_ckc(device, fid, fp32))), extra=f"({time.time() - t0:.1f} s)"))


@pytest.mark.parametrize("heads", [1, 8])
def test_indexer(device, heads):
    """k [1, 1, 65536, 128] one-hot rows (key t holds pattern t), q [1, heads, 64, 128] all ones: q.k = the pattern; gates
    w [1, 1, 64, heads] from the special and normal set; causal start past the last key, so no key is masked."""
    t0 = time.time()
    T, Dm, Sq = 65536, 128, 64
    B = b16_set()
    k = np.zeros((T, Dm), np.uint16)
    k[np.arange(T), np.arange(T) % Dm] = PATS
    q = np.full((1, heads, Sq, Dm), 0x3F80, np.uint16)
    tk, tq = _dev(k, (1, 1, T, Dm), device), _dev(q, q.shape, device)
    for ws in range(2):
        w = B[(np.arange(Sq * heads) * (1 + 6 * ws) + 11 * ws) % B.size].reshape(1, 1, Sq, heads)
        tw = _dev(w, w.shape, device)
        for fid, fp32 in (("HiFi4", False), ("HiFi2", False), ("LoFi", False), ("HiFi4", True)):
            tag = f"indexer_h{heads}_w{ws}_{fid}_{'d32' if fp32 else 'd16'}"
            _guard(tag, lambda: stage(tag, out_bits(ttnn.experimental.indexer_score_dsa(
                tq, tk, tw, chunk_start_idx=T, program_config=ttnn.IndexerScoreProgramConfig(q_chunk_size=32, k_chunk_size=64, head_group_size=0),
                compute_kernel_config=_ckc(device, fid, fp32))), extra=f"({time.time() - t0:.1f} s)"))


@pytest.mark.parametrize("src", ["finite", "all"])
def test_softmax(device, src):
    """scale_mask_softmax, x (10, 1, 64, 113) (4 tiles, 15 padded columns: MASK_PADDED_DATA, the dest-reuse add), rows of
    patterns; mask (10, 1, 32, 113) zeros with -inf, or the finite special and normal values."""
    t0 = time.time()
    Wd, Hh, Bt = 113, 64, 10
    pats = FINITE if src == "finite" else PATS
    x = np.resize(pats, Bt * Hh * Wd).reshape(Bt, 1, Hh, Wd)
    B = b16_set()
    bfin = B[cls_bits(B) <= 2]
    for mk in ("inf", "vals"):
        if mk == "inf":
            m = np.where(np.arange(Bt * 32 * Wd).reshape(Bt, 1, 32, Wd) % 5 == 0, 0xFF80, 0).astype(np.uint16)
        else:
            m = bfin[np.arange(Bt * 32 * Wd) % bfin.size].reshape(Bt, 1, 32, Wd)
        tx, tm = _dev(x, x.shape, device), _dev(m, m.shape, device)
        for fid, fp32 in (("HiFi4", True), ("HiFi4", False), ("LoFi", False)):
            for ns in (True, False):
                tag = f"softmax_{src}_{mk}_{fid}_{'d32' if fp32 else 'd16'}_{'ns' if ns else 'plain'}"
                _guard(tag, lambda: stage(tag, out_bits(ttnn.scale_mask_softmax(
                    tx, 0.75, tm, compute_kernel_config=_ckc(device, fid, fp32), numeric_stable=ns)), extra=f"({time.time() - t0:.1f} s)"))
