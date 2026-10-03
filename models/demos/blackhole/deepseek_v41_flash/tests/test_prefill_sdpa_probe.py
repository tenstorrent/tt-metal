# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Probe: ttnn prefill SDPA at the DeepSeek-V4.1-Flash attention shapes (per device: U users, 8 q heads, ONE kv head
(K == V), head_dim 512, per-head sink, window 128), on the 4x8 mesh with replicated data (same L1 conditions as the model).

Variants: (win) is_causal + sliding_window_size=128 + sink; (mask) additive mask over [kv | latents] + sink.
Env DSV41_PROBE_CHUNKS="q,k;q,k" overrides the chunk sizes tried."""

import os

import pytest
import torch

import ttnn
from models.demos.blackhole.deepseek_v41_flash.reference.ref_layer import pcc

H, D, WINDOW = 8, 512, 128
SCALE = D**-0.5


def ref_attn(q, k, sink, mask):
    """q [U,H,S,D], k [U,1,Sk,D] (K==V), sink [H], additive mask [S,Sk] -> [U,H,S,D] (fp32)."""
    s = torch.einsum("uhsd,uktd->uhst", q.float(), k.float()) * SCALE + mask
    m = s.amax(-1, keepdim=True).clamp(min=-1e30)
    p = torch.exp(s - m)
    den = p.sum(-1, keepdim=True) + torch.exp(sink.float().view(1, H, 1, 1) - m)
    return torch.einsum("uhst,uktd->uhsd", p, k.float()) / den


def band_mask(S, Sk_extra=0, ratio=0):
    i = torch.arange(S).view(-1, 1)
    j = torch.arange(S).view(1, -1)
    m = torch.where((j <= i) & (j > i - WINDOW), 0.0, float("-inf"))
    if ratio:
        Sc = S // ratio
        jc = torch.arange(Sk_extra).view(1, -1)
        mc = torch.where(jc < (i + 1) // ratio, 0.0, float("-inf"))
        m = torch.cat([m, mc], dim=1)
    return m


@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [pytest.param({"l1_small_size": 16384, "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING}, id="ring")],
    indirect=True,
)
@pytest.mark.parametrize("S", [128, 256])
@pytest.mark.parametrize("variant", ["win", "mask_r2"])
def test_prefill_sdpa_probe(mesh_device, S, variant):
    md = mesh_device
    U = 4
    torch.manual_seed(0)
    q = torch.randn(U, H, S, D) * 0.5
    kv = torch.randn(U, 1, S, D) * 0.5
    sink = torch.randn(H) * 0.5
    up = lambda t, dt=ttnn.bfloat16: ttnn.from_torch(
        t.contiguous(),
        device=md,
        dtype=dt,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(md),
    )
    tq = up(q)
    sinks = up((sink / SCALE).reshape(1, H, 1, 1))
    ckc = ttnn.init_device_compute_kernel_config(
        md.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )
    chunks = [
        tuple(int(v) for v in c.split(","))
        for c in os.environ.get("DSV41_PROBE_CHUNKS", "32,32;64,64;128,128;128,256;256,256").split(";")
    ]
    if variant == "win":
        k_dev, k_ref = kv, kv
        mask_ref = band_mask(S)
        tmask, kw = None, dict(is_causal=True, sliding_window_size=WINDOW)
    else:
        Sc = S // 2
        lat = torch.randn(U, 1, Sc, D) * 0.5
        k_ref = torch.cat([kv, lat], dim=2)
        k_dev = k_ref
        mask_ref = band_mask(S, Sc, 2)
        tmask = up(mask_ref.clamp(min=-1e9).reshape(1, 1, S, -1))
        kw = dict(is_causal=False, attn_mask=tmask)
    tk = up(k_dev)
    ref = ref_attn(q, k_ref, sink, mask_ref.clamp(min=-1e9))
    for qc, kc in chunks:
        if S % qc or tk.shape[2] % kc:
            continue
        try:
            out = ttnn.transformer.scaled_dot_product_attention(
                tq,
                tk,
                tk,
                scale=SCALE,
                attention_sink=sinks,
                compute_kernel_config=ckc,
                program_config=ttnn.SDPAProgramConfig(
                    compute_with_storage_grid_size=md.compute_with_storage_grid_size(),
                    q_chunk_size=qc,
                    k_chunk_size=kc,
                    exp_approx_mode=False,
                ),
                **kw,
            )
            got = ttnn.to_torch(ttnn.get_device_tensors(out)[0]).float()
            print(f"PROBE {variant} S={S} q{qc} k{kc}: PCC {pcc(got, ref):.5f}", flush=True)
            ttnn.deallocate(out)
        except Exception as e:  # L1 overflow etc.
            print(f"PROBE {variant} S={S} q{qc} k{kc}: FAILED {str(e).splitlines()[0][:160]}", flush=True)
