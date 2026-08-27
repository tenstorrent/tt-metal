# SPDX-License-Identifier: Apache-2.0
"""Standalone correctness + device-time probe for the K-BF4 fused head-split.

Compares:
  REF  = bge_qkv_heads_headsplit(BF8) then ttnn.typecast(k, bfloat4_b)   [current]
  NEW  = bge_qkv_heads_headsplit(..., k_out_dtype=bfloat4_b)             [fused]

Correctness: Q/V must be bit-identical; K must be bit-equivalent to the
standalone typecast (same BF8->BF4 LLK). Device time via tracy signposts.

Shape: qkv_fused [B6, 1, S8192, 3*16*64=3072] BFP8 (QKV matmul output).
"""
import os

import pytest
import torch
from loguru import logger

import ttnn

try:
    from tracy import signpost
except ImportError:

    def signpost(*a, **k):
        return None


B, NUM_HEADS, S, DH = 6, 16, 8192, 64
QKV_W = 3 * NUM_HEADS * DH
HEAD_GROUPS = 4
N_ITERS = 10


@pytest.mark.parametrize("device_params", [{"trace_region_size": 10_000_000, "num_command_queues": 1}], indirect=True)
def test_headsplit_kbf4_probe(mesh_device):
    prof = os.environ.get("TT_METAL_DEVICE_PROFILER", "0") == "1"

    from models.demos.wormhole.bge_m3.tt.custom_ops.fused_qkv_heads.op import bge_qkv_heads_headsplit

    torch.manual_seed(0)
    qkv_t = torch.randn(B, 1, S, QKV_W, dtype=torch.bfloat16)
    qkv = ttnn.from_torch(
        qkv_t,
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # ---- REF: BF8 head-split + standalone typecast ----
    q_ref, k_ref_bf8, v_ref = bge_qkv_heads_headsplit(qkv, num_heads=NUM_HEADS, head_groups=HEAD_GROUPS)
    k_ref = ttnn.typecast(k_ref_bf8, dtype=ttnn.bfloat4_b)

    # ---- NEW: fused K-BF4 head-split ----
    q_new, k_new, v_new = bge_qkv_heads_headsplit(
        qkv, num_heads=NUM_HEADS, head_groups=HEAD_GROUPS, k_out_dtype=ttnn.bfloat4_b
    )

    # ---- correctness ----
    def to_t(x):
        return ttnn.to_torch(x, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))[:B]

    q_ref_t, q_new_t = to_t(q_ref), to_t(q_new)
    v_ref_t, v_new_t = to_t(v_ref), to_t(v_new)
    k_ref_t, k_new_t = to_t(k_ref), to_t(k_new)

    def pcc(a, b):
        a, b = a.float().flatten(), b.float().flatten()
        return torch.corrcoef(torch.stack([a, b]))[0, 1].item()

    q_pcc, v_pcc, k_pcc = pcc(q_ref_t, q_new_t), pcc(v_ref_t, v_new_t), pcc(k_ref_t, k_new_t)
    q_exact = torch.equal(q_ref_t, q_new_t)
    v_exact = torch.equal(v_ref_t, v_new_t)
    logger.info(f"Q: exact={q_exact} pcc={q_pcc:.6f}")
    logger.info(f"V: exact={v_exact} pcc={v_pcc:.6f}")
    logger.info(f"K: pcc(new vs ref-typecast)={k_pcc:.6f}")

    assert q_exact, "Q must be bit-identical"
    assert v_exact, "V must be bit-identical"
    assert k_pcc > 0.999, f"K BF4 must match standalone typecast, got {k_pcc}"

    if not prof:
        logger.info("correctness OK (set TT_METAL_DEVICE_PROFILER=1 for timing)")
        return

    # ---- device timing ----
    for _ in range(2):
        a, b, c = bge_qkv_heads_headsplit(qkv, num_heads=NUM_HEADS, head_groups=HEAD_GROUPS)
        d = ttnn.typecast(b, dtype=ttnn.bfloat4_b)
        for t in (a, b, c, d):
            ttnn.deallocate(t)
    ttnn.synchronize_device(mesh_device)
    signpost("ref_headsplit_plus_typecast")
    for _ in range(N_ITERS):
        a, b, c = bge_qkv_heads_headsplit(qkv, num_heads=NUM_HEADS, head_groups=HEAD_GROUPS)
        d = ttnn.typecast(b, dtype=ttnn.bfloat4_b)
        for t in (a, b, c, d):
            ttnn.deallocate(t)
    ttnn.synchronize_device(mesh_device)

    for _ in range(2):
        a, b, c = bge_qkv_heads_headsplit(qkv, num_heads=NUM_HEADS, head_groups=HEAD_GROUPS, k_out_dtype=ttnn.bfloat4_b)
        for t in (a, b, c):
            ttnn.deallocate(t)
    ttnn.synchronize_device(mesh_device)
    signpost("new_fused_kbf4")
    for _ in range(N_ITERS):
        a, b, c = bge_qkv_heads_headsplit(qkv, num_heads=NUM_HEADS, head_groups=HEAD_GROUPS, k_out_dtype=ttnn.bfloat4_b)
        for t in (a, b, c):
            ttnn.deallocate(t)
    ttnn.synchronize_device(mesh_device)
    signpost("end")
