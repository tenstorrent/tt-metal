# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Standalone repro for the Quasar PREFILL SDPA op, seen in llama32_1b prefill.

The llama32_1b prefill attention ends in a causal SDPA:

    ttnn.experimental.quasar.transformer.scaled_dot_product_attention(
        q_heads, k_heads, v_heads, is_causal=True, scale=..., program_config=SDPAProgramConfig(...))

The e2e run reaches this op (after QKV matmul -> create_qkv_heads -> RoPE -> KV fill) and FATALs at
program creation because the model's SDPAProgramConfig pins compute_with_storage_grid_size=(8,8) = 64
cores, but craq-sim exposes only 8x4 = 32 cores (sdpa_program_factory.cpp:383). This file exercises JUST
that op with a SMALL grid (1 and 2 cores) to see whether prefill SDPA runs at all on the Quasar sim once
the grid fits -- separate from the grid-size FATAL. If it runs on 1/2 cores, grid-coercion is a viable
on-device path; if it stalls/faults like decode SDPA did, prefill SDPA must also be host-sided.

Shapes (llama-3.2-1B prefill, one user's prompt):
    q      : [1, n_q_heads=32, seq, head_dim=64]  bf16 TILE, DRAM interleaved
    k / v  : [1, n_kv_heads=8, seq, head_dim=64]  bf16 TILE, DRAM interleaved
    is_causal=True (Sq == Sk), GQA group = n_q_heads // n_kv_heads = 4
    program_config: SDPAProgramConfig(compute_with_storage_grid_size=(gx,gy), q_chunk_size=64, k_chunk_size=64)

Prefill SDPA validate requires: TILE, bf16, NON-sharded (DRAM/L1 interleaved) inputs; no padding on the
batch / num_heads / head_dim dims (sdpa_device_operation.cpp). Inputs are built via a bf16 row-major upload
+ quasar.tilize (NOT from_torch(TILE), which hangs on the Quasar sim). bf16 throughout.

seq=128 with q_chunk=64 -> 2 Q chunks, so the 2-core grid actually distributes work (1 chunk/core), which
is what we want to probe vs the 1-core (serial) case.

Run (Quasar sim, with watcher):
    MESH_DEVICE=<qsr> TT_METAL_SIMULATOR=~/sim/libttsim.so TT_METAL_WATCHER=1 \
        pytest tests/ttnn/unit_tests/operations/test_quasar_sdpa_prefill.py
"""

import pytest
import torch
from loguru import logger

import ttnn

# llama-3.2-1B prefill attention dims
N_Q_HEADS = 32
N_KV_HEADS = 8
HEAD_DIM = 64
SEQ = 128  # tile-aligned prompt length; 2 Q chunks at q_chunk=64
Q_CHUNK = 64
K_CHUNK = 64
SCALE = HEAD_DIM**-0.5


def _tile_bf16_dram(t_bf16, mesh_device):
    """bf16 TILE, DRAM-interleaved without from_torch(TILE) (hangs on the Quasar sim): upload row-major,
    then tilize via the Gen2-native quasar op where available."""
    rm = ttnn.from_torch(
        t_bf16,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.replicate_tensor_to_mesh_mapper(mesh_device),
    )
    try:
        return ttnn.experimental.quasar.tilize(rm, memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=ttnn.bfloat16)
    except (AttributeError, RuntimeError) as e:
        logger.info(f"[sdpa-prefill-repro] quasar.tilize unavailable ({e}); using mainline ttnn.tilize")
        return ttnn.tilize(rm, memory_config=ttnn.DRAM_MEMORY_CONFIG)


def _compute_cfg():
    # fp32_dest_acc_en=False on Quasar (bf16->Tf32 unpack gap); HiFi2 matches the model.
    return ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )


def _ref_causal_gqa(q, k, v, scale):
    """Torch reference: causal GQA self-attention. q [1,nqh,S,hd], k/v [1,nkv,S,hd] -> out [1,nqh,S,hd]."""
    q = q.float()
    k = k.float()
    v = v.float()
    nqh, S = q.shape[1], q.shape[2]
    nkv = k.shape[1]
    group = max(nqh // nkv, 1)
    mask = torch.triu(torch.full((S, S), float("-inf")), diagonal=1)  # causal: row i attends to <= i
    out = torch.zeros_like(q)
    for h in range(nqh):
        kv = h // group
        scores = (q[0, h] @ k[0, kv].transpose(-1, -2)) * scale + mask  # [S, S]
        w = torch.softmax(scores, dim=-1)
        out[0, h] = w @ v[0, kv]
    return out


def _pcc(a, b):
    a = a.flatten().float()
    b = b.flatten().float()
    if torch.allclose(a, b):
        return 1.0
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def _run_prefill(mesh_device, gx, gy):
    """Causal prefill SDPA on a gx*gy grid. Probes whether prefill SDPA runs on the Quasar sim once the
    grid fits (the model pins 8x8=64 cores, which the 8x4 sim rejects at sdpa_program_factory.cpp:383)."""
    torch.manual_seed(0)
    q = torch.randn(1, N_Q_HEADS, SEQ, HEAD_DIM, dtype=torch.bfloat16)
    k = torch.randn(1, N_KV_HEADS, SEQ, HEAD_DIM, dtype=torch.bfloat16)
    v = torch.randn(1, N_KV_HEADS, SEQ, HEAD_DIM, dtype=torch.bfloat16)

    qt = _tile_bf16_dram(q, mesh_device)
    kt = _tile_bf16_dram(k, mesh_device)
    vt = _tile_bf16_dram(v, mesh_device)

    prog_cfg = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(gx, gy),
        exp_approx_mode=False,
        q_chunk_size=Q_CHUNK,
        k_chunk_size=K_CHUNK,
    )

    logger.info(
        f"[sdpa-prefill-repro] causal SDPA q[1,{N_Q_HEADS},{SEQ},{HEAD_DIM}] k/v[1,{N_KV_HEADS},{SEQ},{HEAD_DIM}] "
        f"grid {gx}x{gy} (num_cores={gx * gy}) q_chunk={Q_CHUNK} k_chunk={K_CHUNK}"
    )
    out = ttnn.experimental.quasar.transformer.scaled_dot_product_attention(
        qt,
        kt,
        vt,
        is_causal=True,
        scale=SCALE,
        program_config=prog_cfg,
        compute_kernel_config=_compute_cfg(),
    )
    ttnn.synchronize_device(mesh_device)
    o = ttnn.to_torch(out)  # [1, n_q_heads, seq, head_dim]
    assert torch.isfinite(o).all(), "prefill SDPA produced non-finite output"

    ref = _ref_causal_gqa(q, k, v, SCALE)
    pcc = _pcc(o, ref)
    logger.info(f"[sdpa-prefill-repro] out shape {tuple(o.shape)} PCC={pcc:.5f}")
    assert pcc > 0.99, f"prefill SDPA PCC too low: {pcc}"


@pytest.mark.parametrize("grid_xy", [(1, 1), (2, 1)], ids=["1core", "2core"])
def test_prefill_sdpa_grids(mesh_device, grid_xy):
    """Prefill causal SDPA on 1 and 2 compute cores. The model pins an 8x8 grid the 8x4 sim rejects; these
    small grids fit, isolating whether prefill SDPA RUNS on Quasar (vs the decode SDPA sim bugs) once the
    grid is legal. 1core = serial (both Q chunks on one core); 2core = one Q chunk per core."""
    gx, gy = grid_xy
    dev = mesh_device.compute_with_storage_grid_size()
    if dev.x < gx or dev.y < gy:
        pytest.skip(f"grid {gx}x{gy} needs a device >= that; device is {dev.x}x{dev.y}")
    _run_prefill(mesh_device, gx, gy)
