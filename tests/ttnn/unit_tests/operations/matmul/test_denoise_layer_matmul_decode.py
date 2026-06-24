# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Full pi0.5 denoise decoder LAYER: WITH vs WITHOUT matmul_decode.

This exercises a complete Gemma-300M action-expert layer (not just the MLP), so the
effect of matmul_decode is measured the way it actually lands in production -- diluted
by the attention that dominates the layer.

Layer (adaRMS / adaLN-Zero modulated decoder block, width=1024, mlp_dim=4096,
num_heads=8, num_kv_heads=1 (GQA), head_dim=256, M=32 suffix tokens attending to a
1024-token VLM prefix KV):

    h   = rms_norm(x) * (1 + scale_a) + shift_a
    qkv = h @ Wqkv ; split into q[8], k[1], v[1] heads ; RoPE(q), RoPE(k)
    k,v = concat(prefix_kv, suffix_kv)            # 1024 + 32 = 1056 keys
    a   = nlp_concat_heads( SDPA(q, k, v, mask) ) @ Wo
    x   = x + gate_a * a
    h2  = rms_norm(x) * (1 + scale_m) + shift_m
    m   = down( gelu(gate(h2)) * up(h2) )          # GeGLU MLP
    x   = x + gate_m * m

matmul_decode is applied to the **MLP** projections (gate/up/down) -- the projections
where resident-L1 weights are the candidate; the attention qkv/o projections stay
ttnn.linear in BOTH modes, so the only difference is the MLP path. Three modes, each
asserted against the same torch reference (PCC >= 0.99):

  * ``linear``        -- WITHOUT matmul_decode (production default).
  * ``decode``        -- MLP via matmul_decode (partial-WS + reshard-before-down).
  * ``decode_fused``  -- MLP via matmul_decode with ``fused_gelu=True``.

Each path's warm layer latency is printed so with/without is directly comparable.

Run:
    pytest tests/ttnn/unit_tests/operations/matmul/test_denoise_layer_matmul_decode.py -x -s
"""

import time

import pytest
import torch
import torch.nn.functional as F

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc

W = 1024
MLP_DIM = 4096
NH, NKV, HD = 8, 1, 256
QKV_OUT = NH * HD + 2 * NKV * HD  # 2560
M = 32  # suffix tokens (1 tile)
AH = 10  # real action tokens (rest of M is padding)
PREFIX = 1024  # VLM prefix KV length
KV = PREFIX + M  # 1056
EPS = 1e-6
SCALE = HD**-0.5
K_BLOCKS, N_BLOCKS, RESHARD_CORES = 2, 32, 2
PCC = 0.99
SEED = 0

_HIFI2 = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=False, fp32_dest_acc_en=False, packer_l1_acc=False
)
_LOFI = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.LoFi, math_approx_mode=False, fp32_dest_acc_en=False, packer_l1_acc=False
)
_SDPA_CK = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=False
)
_L1, _DRAM = ttnn.L1_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG


# --------------------------------------------------------------------------- weights / inputs
def _make_layer(seed=SEED):
    torch.manual_seed(seed)
    g = lambda *s: torch.randn(*s, dtype=torch.float32)
    return {
        "ln_attn": g(W) * 0.1 + 1.0,
        "ln_mlp": g(W) * 0.1 + 1.0,
        "wqkv": g(QKV_OUT, W) * 0.02,  # (out, in)
        "wo": g(W, NH * HD) * 0.02,
        "gate": g(MLP_DIM, W) * 0.02,
        "up": g(MLP_DIM, W) * 0.02,
        "down": g(W, MLP_DIM) * 0.02,
        # adaLN-Zero modulation (precomputed from the timestep cond once per inference)
        "scale_a": g(1, W) * 0.1,
        "shift_a": g(1, W) * 0.1,
        "gate_a": g(1, W) * 0.1,
        "scale_m": g(1, W) * 0.1,
        "shift_m": g(1, W) * 0.1,
        "gate_m": g(1, W) * 0.1,
    }


def _inputs(seed=SEED + 1):
    torch.manual_seed(seed)
    x = torch.randn(M, W)
    cos = torch.randn(1, 1, M, HD)  # arbitrary RoPE tables (ttnn applies x*cos+rotate_half(x)*sin)
    sin = torch.randn(1, 1, M, HD)
    prefix_k = torch.randn(1, NKV, PREFIX, HD) * 0.1
    prefix_v = torch.randn(1, NKV, PREFIX, HD) * 0.1
    mask = torch.zeros(1, 1, M, KV)
    mask[:, :, :, PREFIX + AH :] = -1e9  # mask the padded suffix key positions
    return x, cos, sin, prefix_k, prefix_v, mask


# --------------------------------------------------------------------------- torch reference
def _rmsnorm(x, w):
    return x / torch.sqrt(x.pow(2).mean(-1, keepdim=True) + EPS) * w


def _rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _rope(x, cos, sin):
    return x * cos + _rotate_half(x) * sin


def _reference(wts, x, cos, sin, pk, pv, mask):
    c, s = cos[0, 0], sin[0, 0]  # [M, HD]
    h = _rmsnorm(x, wts["ln_attn"]) * (1 + wts["scale_a"]) + wts["shift_a"]
    qkv = h @ wts["wqkv"].t()  # [M, 2560]
    q = qkv[:, : NH * HD].view(M, NH, HD).permute(1, 0, 2)  # [NH, M, HD]
    k = qkv[:, NH * HD : (NH + NKV) * HD].view(M, NKV, HD).permute(1, 0, 2)
    v = qkv[:, (NH + NKV) * HD :].view(M, NKV, HD).permute(1, 0, 2)
    q = _rope(q, c, s)
    k = _rope(k, c, s)
    k = torch.cat([pk[0], k], dim=1)  # [NKV, KV, HD]
    v = torch.cat([pv[0], v], dim=1)
    k = k.repeat_interleave(NH // NKV, dim=0)  # GQA expand -> [NH, KV, HD]
    v = v.repeat_interleave(NH // NKV, dim=0)
    scores = (q @ k.transpose(-1, -2)) * SCALE + mask[0]  # [NH, M, KV]
    a = torch.softmax(scores, dim=-1) @ v  # [NH, M, HD]
    a = a.permute(1, 0, 2).reshape(M, NH * HD)  # [M, 2048]
    x = x + wts["gate_a"] * (a @ wts["wo"].t())
    h2 = _rmsnorm(x, wts["ln_mlp"]) * (1 + wts["scale_m"]) + wts["shift_m"]
    m = (F.gelu(h2 @ wts["gate"].t()) * (h2 @ wts["up"].t())) @ wts["down"].t()
    return x + wts["gate_m"] * m


# --------------------------------------------------------------------------- ttnn helpers
def _crs(device, n):
    return ttnn.num_cores_to_corerangeset(n, device.compute_with_storage_grid_size(), True)


def _tt(device, t, dtype=ttnn.bfloat16, mem=_L1):
    return ttnn.from_torch(t, layout=ttnn.TILE_LAYOUT, device=device, dtype=dtype, memory_config=mem)


def _pws_B(device, w_kn, n_blocks):
    k, n = w_kn.shape
    kc, nc = k // K_BLOCKS, n // n_blocks
    br = w_kn.reshape(K_BLOCKS, kc, n).permute(1, 0, 2).reshape(kc, n * K_BLOCKS)
    mc = ttnn.create_sharded_memory_config(
        (kc, nc),
        core_grid=_crs(device, K_BLOCKS * n_blocks),
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    return ttnn.from_torch(br, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mc, dtype=ttnn.bfloat8_b)


# --------------------------------------------------------------------------- ttnn layer
def _build_layer(device, wts, ins, mode):
    """Upload weights once; return a run() closure that does the full layer forward.
    mode in {linear, decode, decode_fused} controls only the MLP path."""
    x_t, cos_t, sin_t, pk_t, pv_t, mask_t = ins
    decode = mode != "linear"
    fused = mode == "decode_fused"

    x0 = _tt(device, x_t.view(1, 1, M, W))
    ln_a = _tt(device, wts["ln_attn"])
    ln_m = _tt(device, wts["ln_mlp"])
    sca, sha, gta = (_tt(device, wts[k].view(1, 1, 1, W)) for k in ("scale_a", "shift_a", "gate_a"))
    scm, shm, gtm = (_tt(device, wts[k].view(1, 1, 1, W)) for k in ("scale_m", "shift_m", "gate_m"))
    wqkv = _tt(device, wts["wqkv"].t().contiguous(), dtype=ttnn.bfloat8_b)
    wo = _tt(device, wts["wo"].t().contiguous(), dtype=ttnn.bfloat8_b)
    cos = _tt(device, cos_t)
    sin = _tt(device, sin_t)
    pk = _tt(device, pk_t)
    pv = _tt(device, pv_t)  # bf16 to match the RoPE'd (bf16) suffix KV for concat
    mask = _tt(device, mask_t, mem=_DRAM)
    if decode:
        gate_b = _pws_B(device, wts["gate"].t().contiguous(), N_BLOCKS)
        up_b = _pws_B(device, wts["up"].t().contiguous(), N_BLOCKS)
        down_b = _pws_B(device, wts["down"].t().contiguous(), N_BLOCKS)
        a_mc = ttnn.create_sharded_memory_config(
            (M, W // RESHARD_CORES),
            core_grid=_crs(device, RESHARD_CORES),
            strategy=ttnn.ShardStrategy.WIDTH,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        mid_mc = ttnn.create_sharded_memory_config(
            (M, MLP_DIM // RESHARD_CORES),
            core_grid=_crs(device, RESHARD_CORES),
            strategy=ttnn.ShardStrategy.WIDTH,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
    else:
        gw = _tt(device, wts["gate"].t().contiguous(), dtype=ttnn.bfloat8_b)
        uw = _tt(device, wts["up"].t().contiguous(), dtype=ttnn.bfloat8_b)
        dw = _tt(device, wts["down"].t().contiguous(), dtype=ttnn.bfloat8_b)

    def _adarms(x, lnw, scale, shift):
        n = ttnn.rms_norm(x, weight=lnw, epsilon=EPS, memory_config=_L1)
        sc1 = ttnn.add(scale, 1.0)
        n = ttnn.multiply(n, sc1, memory_config=_L1)
        out = ttnn.add(n, shift, memory_config=_L1)
        ttnn.deallocate(n)
        ttnn.deallocate(sc1)
        return out

    def _mlp(h):
        if decode:
            hw = ttnn.to_memory_config(h, a_mc)
            gate = ttnn.matmul_decode(
                hw, gate_b, partial_width_sharded=True, compute_kernel_config=_HIFI2, fused_gelu=fused
            )
            up = ttnn.matmul_decode(hw, up_b, partial_width_sharded=True, compute_kernel_config=_HIFI2)
            if not fused:
                gate = ttnn.gelu(gate, fast_and_approximate_mode=False)
            hid = ttnn.multiply(gate, up, memory_config=gate.memory_config())
            hid2 = ttnn.to_memory_config(hid, mid_mc)
            ows = ttnn.matmul_decode(hid2, down_b, partial_width_sharded=True, compute_kernel_config=_LOFI)
            out = ttnn.to_memory_config(ows, _L1)
            for t in (hw, gate, up, hid, hid2, ows):
                ttnn.deallocate(t)
            return out
        gate = ttnn.linear(h, gw, memory_config=_L1, compute_kernel_config=_HIFI2)
        up = ttnn.linear(h, uw, memory_config=_L1, compute_kernel_config=_HIFI2)
        gate = ttnn.gelu(gate, fast_and_approximate_mode=False)
        hid = ttnn.multiply(gate, up, memory_config=_L1)
        out = ttnn.linear(hid, dw, memory_config=_L1, compute_kernel_config=_LOFI)
        for t in (gate, up, hid):
            ttnn.deallocate(t)
        return out

    def run():
        h = _adarms(x0, ln_a, sca, sha)
        qkv = ttnn.linear(h, wqkv, memory_config=_L1, compute_kernel_config=_HIFI2)
        ttnn.deallocate(h)
        q, k, v = ttnn.experimental.nlp_create_qkv_heads(
            qkv, num_heads=NH, num_kv_heads=NKV, transpose_k_heads=False, memory_config=_L1
        )
        ttnn.deallocate(qkv)
        q = ttnn.experimental.rotary_embedding(q, cos, sin, memory_config=_L1)
        k = ttnn.experimental.rotary_embedding(k, cos, sin, memory_config=_L1)
        kk = ttnn.concat([pk, k], dim=2, memory_config=_L1)
        vv = ttnn.concat([pv, v], dim=2, memory_config=_L1)
        ttnn.deallocate(k)
        ttnn.deallocate(v)
        a = ttnn.transformer.scaled_dot_product_attention(
            q, kk, vv, attn_mask=mask, is_causal=False, scale=SCALE, compute_kernel_config=_SDPA_CK, memory_config=_L1
        )
        ttnn.deallocate(q)
        ttnn.deallocate(kk)
        ttnn.deallocate(vv)
        a = ttnn.experimental.nlp_concat_heads(a, memory_config=_L1)
        o = ttnn.linear(a, wo, memory_config=_L1, compute_kernel_config=_HIFI2)
        ttnn.deallocate(a)
        og = ttnn.multiply(o, gta, memory_config=_L1)
        x1 = ttnn.add(x0, og, memory_config=_L1)
        ttnn.deallocate(o)
        ttnn.deallocate(og)
        h2 = _adarms(x1, ln_m, scm, shm)
        m = _mlp(h2)
        ttnn.deallocate(h2)
        mg = ttnn.multiply(m, gtm, memory_config=_L1)
        out = ttnn.add(x1, mg, memory_config=_L1)
        for t in (x1, m, mg):
            ttnn.deallocate(t)
        return out

    return run


def _latency_ms(device, run, reps=20):
    for _ in range(5):
        ttnn.deallocate(run())
        ttnn.synchronize_device(device)
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        o = run()
        ttnn.synchronize_device(device)
        ts.append((time.perf_counter() - t0) * 1e3)
        ttnn.deallocate(o)
    ts.sort()
    return ts[len(ts) // 2]


# --------------------------------------------------------------------------- test
@pytest.mark.parametrize("mode", ["linear", "decode", "decode_fused"])
def test_denoise_layer_with_without_matmul_decode(device, mode):
    """Full denoise layer, with vs without matmul_decode (MLP); each must match torch (PCC >= 0.99)."""
    wts = _make_layer()
    ins = _inputs()
    ref = _reference(wts, *ins)  # [M, W]

    run = _build_layer(device, wts, ins, mode)
    out = ttnn.to_torch(run()).float().reshape(M, W)
    lat = _latency_ms(device, run)
    print(f"\n[denoise LAYER {mode:13s}] latency median = {lat:.3f} ms  (PCC asserted >= {PCC})")
    assert_with_pcc(ref, out, PCC)
