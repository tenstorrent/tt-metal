# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""QWEN36_VERIFY_HEADS_DENSE (TPAttention._verify_qkv_dense) against the legacy lean-verify op sequence, on device.

From the same q|k|v block (the fused QKV matmul's contiguous [1,1,R,(NH+2NKV)*HD] part), the legacy sequence (decode
head split, 3 sharded_to_interleaved, rms_norm + (1+w) multiply for q and for k, apply_partial_rope_decode for q and for
k) and the dense one must produce q [1,R,NH,HD], k and v [1,R,NKV,HD] that are equal byte for byte, at the verify row
counts of the full trace (32 rows: 8 users x 4 or 4 users x 8 candidates), of the 4-, 2- and 1-user buckets (16, 8, 4
rows) and of K=11 (24, 12 rows), with distinct long-context positions per row; likewise the head concat after the SDPA.
Each sequence is then captured in a trace and replayed to compare their device time (ops alone, no model around them).

Run: MESH_DEVICE=P150x8 pytest models/demos/blackhole/qwen36/tests/test_verify_heads_dense.py -v -s
"""

import time
import types

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.blackhole.qwen36.tests.test_factory import parametrize_mesh_tp, replicate_to_device
from models.demos.blackhole.qwen36.tests.test_rope_tp import _rope_params
from models.demos.blackhole.qwen36.tt import tp_common as tpc
from models.demos.blackhole.qwen36.tt.attention.rope_tp import apply_partial_rope_decode, rot_mats_decode
from models.demos.blackhole.qwen36.tt.attention.tp import TPAttention

REPLAYS = 200


def _legacy(qkv3, tw, cos_tt, sin_tt, NH, NKV, B, rd):
    """The legacy lean-verify sequence, op for op (TPAttention._forward_verify_lean, QWEN36_VERIFY_HEADS_DENSE off)."""
    _L1 = ttnn.L1_MEMORY_CONFIG
    q, k, v = ttnn.experimental.nlp_create_qkv_heads_decode(
        qkv3, num_heads=NH, num_kv_heads=NKV, memory_config=ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG
    )
    q = ttnn.sharded_to_interleaved(q, _L1)
    k = ttnn.sharded_to_interleaved(k, _L1)
    v = ttnn.sharded_to_interleaved(v, _L1)
    q = ttnn.multiply(ttnn.rms_norm(q, epsilon=1e-6, memory_config=_L1), tw["q_norm"], memory_config=_L1)
    k = ttnn.multiply(ttnn.rms_norm(k, epsilon=1e-6, memory_config=_L1), tw["k_norm"], memory_config=_L1)
    q = apply_partial_rope_decode(q, cos_tt, sin_tt, NH, B, rd)
    k = apply_partial_rope_decode(k, cos_tt, sin_tt, NKV, B, rd)
    return q, k, v


def _host(t, mesh_device):
    return ttnn.to_torch(t, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))


def _trace_us(mesh_device, fn):
    """Device time per replay of fn's ops: compile eagerly, capture, replay REPLAYS times."""
    for o in fn():
        ttnn.deallocate(o)
    ttnn.synchronize_device(mesh_device)
    tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    outs = fn()
    ttnn.end_trace_capture(mesh_device, tid, cq_id=0)
    ttnn.synchronize_device(mesh_device)
    t0 = time.perf_counter()
    for _ in range(REPLAYS):
        ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh_device)
    us = 1e6 * (time.perf_counter() - t0) / REPLAYS
    ttnn.release_trace(mesh_device, tid)
    for o in outs:
        ttnn.deallocate(o)
    return us


@torch.no_grad()
@parametrize_mesh_tp(trace_region_size=64 * 1024 * 1024)
@pytest.mark.parametrize("rows", [32, 24, 16, 12, 8, 4], ids=["r32", "r24", "r16", "r12", "r8", "r4"])
def test_verify_heads_dense_qkv_bit_identical(mesh_device, rows, monkeypatch, reset_seeds, ensure_gc):
    monkeypatch.setenv("QWEN36_VERIFY_HEADS_DENSE", "1")
    args, HD, rd, theta, NH, NKV = _rope_params(mesh_device, B=8)
    width = (NH + 2 * NKV) * HD
    qn = torch.randn(HD) * 0.1
    kn = torch.randn(HD) * 0.1
    tw = {
        "q_norm": tpc.replicate(qn.to(torch.float32) + 1.0, mesh_device, None),
        "k_norm": tpc.replicate(kn.to(torch.float32) + 1.0, mesh_device, None),
    }
    q1 = (qn.to(torch.float32) + 1.0).to(torch.bfloat16).reshape(1, 1, 1, -1)
    k1 = (kn.to(torch.float32) + 1.0).to(torch.bfloat16).reshape(1, 1, 1, -1)
    tw["qk_norm_dense"] = tpc.replicate(torch.cat([q1] * NH + [k1] * NKV, dim=1), mesh_device, None)
    stub = types.SimpleNamespace(NH=NH, NKV=NKV, HD=HD, rope_dim=rd, tw=tw)
    assert TPAttention._heads_dense_ok(stub, rows)

    positions = torch.arange(rows, dtype=torch.int32) * 37 + 32761
    cos_tt, sin_tt = rot_mats_decode(mesh_device, rd, 40960, theta, positions)
    cos_tt = ttnn.reshape(cos_tt, (1, 1, rows, rd))  # the layout the verify stages ([1,1,R,rd])
    sin_tt = ttnn.reshape(sin_tt, (1, 1, rows, rd))
    x = torch.randn(1, 1, rows, width, dtype=torch.bfloat16) * 4
    x_dev = replicate_to_device(mesh_device, x)

    def l1_copy():
        return ttnn.to_memory_config(x_dev, ttnn.L1_MEMORY_CONFIG)

    old = _legacy(l1_copy(), tw, cos_tt, sin_tt, NH, NKV, rows, rd)
    new = TPAttention._verify_qkv_dense(stub, l1_copy(), cos_tt, sin_tt, rows)
    for name, a, b in zip(("q", "k", "v"), old, new):
        ha, hb = _host(a, mesh_device), _host(b, mesh_device)
        assert tuple(ha.shape) == tuple(hb.shape), f"{name}: shape {tuple(ha.shape)} vs {tuple(hb.shape)}"
        same = torch.equal(ha.view(torch.int16), hb.view(torch.int16))
        logger.info(f"rows={rows} {name} {tuple(ha.shape)} bit-identical={same}")
        assert same, f"rows={rows}: {name} differs (max abs {(ha.float() - hb.float()).abs().max()})"
    for t in (*old, *new):
        ttnn.deallocate(t)

    us_old = _trace_us(mesh_device, lambda: _legacy(l1_copy(), tw, cos_tt, sin_tt, NH, NKV, rows, rd))
    us_new = _trace_us(mesh_device, lambda: TPAttention._verify_qkv_dense(stub, l1_copy(), cos_tt, sin_tt, rows))
    logger.info(
        f"[heads_dense] q|k|v rows={rows}: legacy {us_old:.1f} us/replay, dense {us_new:.1f} us/replay "
        f"(x16 layers: {16 * (us_old - us_new) / 1e3:.3f} ms saved per verify; ops alone, one L1 copy in each)"
    )


@torch.no_grad()
@parametrize_mesh_tp(trace_region_size=64 * 1024 * 1024)
@pytest.mark.parametrize("rows", [32, 24, 16, 12, 8, 4], ids=["r32", "r24", "r16", "r12", "r8", "r4"])
def test_verify_concat_heads_dense_bit_identical(mesh_device, rows, monkeypatch, reset_seeds, ensure_gc):
    """The head concat after the spec SDPA: _concat_heads_dense vs _concat_heads_decode on the SDPA output layout
    ([1,R,NH,HD] over a [1,R,32,HD] tile layout, L1)."""
    monkeypatch.setenv("QWEN36_VERIFY_HEADS_DENSE", "1")
    args, HD, rd, theta, NH, NKV = _rope_params(mesh_device, B=8)
    stub = types.SimpleNamespace(NH=NH, NKV=NKV, HD=HD, rope_dim=rd, tw={}, mesh=mesh_device)
    x = torch.randn(1, rows, NH, HD, dtype=torch.bfloat16) * 2
    x_dev = replicate_to_device(mesh_device, x)

    def l1_copy():
        return ttnn.to_memory_config(x_dev, ttnn.L1_MEMORY_CONFIG)

    old = TPAttention._concat_heads_decode(stub, l1_copy(), rows)
    new = TPAttention._concat_heads_dense(stub, l1_copy(), rows)
    ha, hb = _host(old, mesh_device), _host(new, mesh_device)
    assert tuple(ha.shape) == tuple(hb.shape), f"shape {tuple(ha.shape)} vs {tuple(hb.shape)}"
    same = torch.equal(ha.view(torch.int16), hb.view(torch.int16))
    logger.info(f"rows={rows} concat heads {tuple(ha.shape)} bit-identical={same}")
    assert same, f"rows={rows}: concat heads differ"
    ttnn.deallocate(old)
    ttnn.deallocate(new)
    us_old = _trace_us(mesh_device, lambda: (TPAttention._concat_heads_decode(stub, l1_copy(), rows),))
    us_new = _trace_us(mesh_device, lambda: (TPAttention._concat_heads_dense(stub, l1_copy(), rows),))
    logger.info(
        f"[heads_dense] concat rows={rows}: legacy {us_old:.1f} us/replay, dense {us_new:.1f} us/replay "
        f"(x16 layers: {16 * (us_old - us_new) / 1e3:.3f} ms per verify)"
    )
