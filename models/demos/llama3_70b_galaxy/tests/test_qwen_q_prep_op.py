# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""qwen_q_prep (custom_ops/qwen_q_prep) against the chain it replaces, TtLlamaAttention._qwen_local_q_prep.

The reference runs the production helper itself (_qwen_local_q_setup + _qwen_local_q_prep from
llama_attention.py, bound to a stub that carries only the attributes they read) with a real
models.common.rmsnorm.RMSNorm built the way llama_attention.py builds q_norm. Both paths update their own
copy of the same bf8 Q buffer in place; every device's whole buffer must match exactly (owned slices
rewritten identically, the other slices untouched by both). Prints the wall time of each path.

Run on the Wormhole Galaxy (8x4 mesh):
    pytest models/demos/llama3_70b_galaxy/tests/test_qwen_q_prep_op.py -s
"""

import time

import pytest
import torch
from loguru import logger

import ttnn
from models.common.rmsnorm import RMSNorm
from models.demos.llama3_70b_galaxy.tt.custom_ops.qwen_q_prep import QPrepPlan, column_slices, pair_cores, qwen_q_prep
from models.demos.llama3_70b_galaxy.tt.llama_attention import TtLlamaAttention
from models.demos.llama3_70b_galaxy.tt.llama_common import get_rot_transformation_mat

N_LOCAL_HEADS = 8
HEAD_DIM = 128
ROT_LEN = 40960  # rot mats are built for max_seq_len; shorter prompts read their first S rows
TIMED_ITERS = 5


class _RefAttention:
    """The attributes TtLlamaAttention._qwen_local_q_setup / _qwen_local_q_prep read, nothing else."""

    _qwen_local_q_setup = TtLlamaAttention._qwen_local_q_setup
    _qwen_local_q_prep = TtLlamaAttention._qwen_local_q_prep

    def __init__(self, mesh_device, gamma_torch, trans_mat):
        self.mesh_device = mesh_device
        self.tt_ccl = type("TtCclStub", (), {})()
        self.n_local_heads = N_LOCAL_HEADS
        self.head_dim = HEAD_DIM
        self.qk_norm = True
        # Same constructor arguments as llama_attention.py (q_norm), with Qwen3's rms_norm_eps.
        self.q_norm = RMSNorm(
            device=mesh_device,
            dim=HEAD_DIM,
            state_dict={"layers.0.attention.q_norm.weight": gamma_torch},
            state_dict_prefix=None,
            weight_dtype=ttnn.bfloat16,
            weight_key="layers.0.attention.q_norm",
            eps=1e-06,
        )
        self.transformation_mats = {"prefill": trans_mat}


def _replicated(mesh_device, t, dtype, layout=ttnn.TILE_LAYOUT):
    return ttnn.from_torch(
        t,
        dtype=dtype,
        layout=layout,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )


def _time(fn, iters):
    mesh_device = fn()  # warm-up / compile
    ttnn.synchronize_device(mesh_device)
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    ttnn.synchronize_device(mesh_device)
    return (time.perf_counter() - start) / iters * 1e3


@pytest.mark.parametrize("seq_len", [16384, 40960])
def test_qwen_q_prep_plan_covers_owned_rows(seq_len):
    """Host only: the per-pair row ranges cover each owned (head, tile row) exactly once."""
    cols = 4
    pairs = pair_cores(7, 10)
    plan = QPrepPlan(n_heads=N_LOCAL_HEADS, seq_len=seq_len, cols=cols, num_pairs=len(pairs))
    for col in range(cols):
        s0, s1 = column_slices(col, cols)
        seen = []
        for p in range(len(pairs)):
            start, count = plan.pair_rows(p)
            for row in range(start, start + count):
                h, j, tl = plan.row_coords(row)
                seen.append((h, (s0 if j == 0 else s1) * plan.sl_t + tl))
        expect = {(h, s * plan.sl_t + tl) for h in range(N_LOCAL_HEADS) for s in (s0, s1) for tl in range(plan.sl_t)}
        assert len(seen) == len(expect) and set(seen) == expect


@torch.no_grad()
@pytest.mark.parametrize("mesh_device", [(8, 4)], indirect=True)
@pytest.mark.parametrize("device_params", [{}], indirect=True)
@pytest.mark.parametrize("seq_len", [16384, 40960])
def test_qwen_q_prep_bit_exact(mesh_device, seq_len):
    torch.manual_seed(0)
    rows, cols = list(mesh_device.shape)
    assert (rows, cols) == (8, 4)

    q_torch = torch.randn(1, N_LOCAL_HEADS, seq_len, HEAD_DIM) * 4.0
    cos_torch = torch.rand(1, 1, ROT_LEN, HEAD_DIM) * 2 - 1
    sin_torch = torch.rand(1, 1, ROT_LEN, HEAD_DIM) * 2 - 1
    gamma_torch = torch.randn(HEAD_DIM).to(torch.bfloat16)

    cos = _replicated(mesh_device, cos_torch, ttnn.bfloat16)
    sin = _replicated(mesh_device, sin_torch, ttnn.bfloat16)
    trans_mat = _replicated(mesh_device, get_rot_transformation_mat(HEAD_DIM), ttnn.bfloat16)
    ref = _RefAttention(mesh_device, gamma_torch, trans_mat)
    q_ref = _replicated(mesh_device, q_torch, ttnn.bfloat8_b)
    q_op = _replicated(mesh_device, q_torch, ttnn.bfloat8_b)
    q_in = ttnn.to_torch(ttnn.get_device_tensors(q_ref)[0])  # bf8-quantised input, for the untouched slices

    ref._qwen_local_q_prep(q_ref, [cos, sin], seq_len)
    out = qwen_q_prep(q_op, cos, sin, trans_mat, ref.q_norm.weight, eps=ref.q_norm.eps)
    assert out.buffer_address() == q_op.buffer_address()
    ttnn.synchronize_device(mesh_device)

    sl = seq_len // (2 * cols)
    mismatches = []
    for i, (t_ref, t_op) in enumerate(zip(ttnn.get_device_tensors(q_ref), ttnn.get_device_tensors(q_op))):
        col = i % cols  # get_device_tensors is row-major over the mesh
        a = ttnn.to_torch(t_ref)
        b = ttnn.to_torch(t_op)
        owned = column_slices(col, cols)
        for s in range(2 * cols):
            rs = slice(s * sl, (s + 1) * sl)
            if s in owned:
                if not torch.equal(a[:, :, rs], b[:, :, rs]):
                    diff = (a[:, :, rs].float() - b[:, :, rs].float()).abs()
                    mismatches.append(
                        f"device {i} col {col} slice {s}: {int((diff > 0).sum())} values differ, max {diff.max():.3e}"
                    )
                assert not torch.equal(b[:, :, rs], q_in[:, :, rs]), f"device {i} slice {s} was not rewritten"
            else:
                assert torch.equal(b[:, :, rs], q_in[:, :, rs]), f"device {i} slice {s} (not owned) was modified"
                assert torch.equal(a[:, :, rs], b[:, :, rs])
    assert not mismatches, "\n".join(mismatches[:16])

    # Timing (both paths keep rewriting their own buffer in place; values stay bounded by the norm).
    def run_ref():
        ref._qwen_local_q_prep(q_ref, [cos, sin], seq_len)
        return mesh_device

    def run_op():
        qwen_q_prep(q_op, cos, sin, trans_mat, ref.q_norm.weight, eps=ref.q_norm.eps)
        return mesh_device

    t_ref = _time(run_ref, TIMED_ITERS)
    t_op = _time(run_op, TIMED_ITERS)
    logger.info(f"S={seq_len}: chain {t_ref:.3f} ms, qwen_q_prep {t_op:.3f} ms (wall, mean of {TIMED_ITERS})")

    for t in (q_ref, q_op, cos, sin, trans_mat):
        ttnn.deallocate(t)
