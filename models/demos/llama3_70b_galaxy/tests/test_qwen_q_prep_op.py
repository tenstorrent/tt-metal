# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""qwen_q_prep (custom_ops/qwen_q_prep) against the stock-op chain it replaces,
TtLlamaAttention._qwen_local_q_prep_stock.

The reference runs the production stock helper (_qwen_local_q_setup + _qwen_local_q_prep_stock from
llama_attention.py, bound to a stub that carries only the attributes they read) with a real
models.common.rmsnorm.RMSNorm built the way llama_attention.py builds q_norm. (_qwen_local_q_prep itself routes
qk_norm=True to the op, so it cannot be the reference.) Both paths update their own copy of the same bf8 Q buffer
in place; every device's whole buffer must match exactly (owned slices rewritten identically, the other slices
untouched by both). A second pair of fresh Q tensors (new buffer addresses) runs through the op's program-cache
hit path and must match too. Prints the wall time of each path.

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
    _qwen_local_q_prep_stock = TtLlamaAttention._qwen_local_q_prep_stock

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


@pytest.mark.host_only
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
    sl = seq_len // (2 * cols)

    def check_pair(q_torch_in, tag):
        q_ref = _replicated(mesh_device, q_torch_in, ttnn.bfloat8_b)
        q_op = _replicated(mesh_device, q_torch_in, ttnn.bfloat8_b)
        q_in = ttnn.to_torch(ttnn.get_device_tensors(q_ref)[0])  # bf8-quantised input, for the untouched slices
        ref._qwen_local_q_prep_stock(q_ref, [cos, sin], seq_len)
        out = qwen_q_prep(q_op, cos, sin, trans_mat, ref.q_norm.weight, eps=ref.q_norm.eps)
        assert out.buffer_address() == q_op.buffer_address()
        ttnn.synchronize_device(mesh_device)
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
                            f"{tag} device {i} col {col} slice {s}: {int((diff > 0).sum())} values differ, "
                            f"max {diff.max():.3e}"
                        )
                    assert not torch.equal(b[:, :, rs], q_in[:, :, rs]), f"{tag} device {i} slice {s} not rewritten"
                else:
                    assert torch.equal(b[:, :, rs], q_in[:, :, rs]), f"{tag} device {i} slice {s} (not owned) modified"
                    assert torch.equal(a[:, :, rs], b[:, :, rs])
        assert not mismatches, "\n".join(mismatches[:16])
        logger.info(f"S={seq_len} {tag}: op == stock chain on all {len(ttnn.get_device_tensors(q_op))} devices")
        return q_ref, q_op

    q_ref, q_op = check_pair(q_torch, "first call")
    # Fresh tensors at new buffer addresses: a new descriptor-cache key, and generic_op's program-cache hit path
    # (same program, new runtime-arg addresses).
    q_ref_keep, q_op_keep = q_ref, q_op
    q_ref, q_op = check_pair(torch.randn(1, N_LOCAL_HEADS, seq_len, HEAD_DIM) * 4.0, "fresh buffers")
    assert q_op.buffer_address() != q_op_keep.buffer_address()
    ttnn.deallocate(q_ref_keep)
    ttnn.deallocate(q_op_keep)

    # Timing (both paths keep rewriting their own buffer in place; values stay bounded by the norm).
    def run_ref():
        ref._qwen_local_q_prep_stock(q_ref, [cos, sin], seq_len)
        return mesh_device

    def run_op():
        qwen_q_prep(q_op, cos, sin, trans_mat, ref.q_norm.weight, eps=ref.q_norm.eps)
        return mesh_device

    t_ref = _time(run_ref, TIMED_ITERS)
    t_op = _time(run_op, TIMED_ITERS)
    logger.info(f"S={seq_len}: chain {t_ref:.3f} ms, qwen_q_prep {t_op:.3f} ms (wall, mean of {TIMED_ITERS})")

    for t in (q_ref, q_op, cos, sin, trans_mat):
        ttnn.deallocate(t)


@torch.no_grad()
@pytest.mark.parametrize("mesh_device", [(8, 4)], indirect=True)
# The model's dispatch layout (text_qwen_demo.py): dispatch on a column -> 7x10 worker grid, which the ring SDPA
# program config uses.
@pytest.mark.parametrize(
    "device_params", [{"dispatch_core_axis": ttnn.DispatchCoreAxis.COL, "worker_l1_size": 1345000}], indirect=True
)
@pytest.mark.parametrize("seq_len", [40960])
def test_qwen_ring_sdpa_ignores_unowned_q(mesh_device, seq_len):
    """qwen_q_prep only rewrites each column's two owned Q slices (column_slices); the other six keep stale
    pre-prep values. Prove ring SDPA never reads them: NaN-fill every non-owned slice on each column and check
    that the ring SDPA output is bit-identical (and NaN-free) to a run whose non-owned slices hold finite data.
    Uses the model's production long-prefill ring SDPA program config and compute kernel config."""
    from models.demos.llama3_70b_galaxy.tt.model_config import LlamaOptimizations
    from models.demos.llama3_70b_galaxy.tt.qwen_model_config import TtQwenModelArgs

    torch.manual_seed(1)
    rows, cols = list(mesh_device.shape)
    assert (rows, cols) == (8, 4)
    margs = TtQwenModelArgs(
        mesh_device, max_batch_size=32, max_seq_len=seq_len, optimizations=LlamaOptimizations.performance
    )
    progcfg = margs.get_model_config()["SDPA_RING_PROGCFG"](seq_len, chunk_start_idx=0)
    ckc = margs.compute_kernel_config_ring_sdpa_long
    assert ckc is not None
    sl = seq_len // (2 * cols)

    q_clean = torch.randn(1, N_LOCAL_HEADS, seq_len, HEAD_DIM)
    q_nan = q_clean.repeat(cols, 1, 1, 1)  # one copy per mesh column
    for col in range(cols):
        owned = column_slices(col, cols)
        for s in range(2 * cols):
            if s not in owned:
                q_nan[col, :, s * sl : (s + 1) * sl] = float("nan")
    col_mapper = ttnn.ShardTensor2dMesh(mesh_device, dims=(None, 0), mesh_shape=[rows, cols])

    def q_dev(t, mapper):
        return ttnn.from_torch(
            t,
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=mapper,
        )

    q_a = q_dev(q_clean, ttnn.ReplicateTensorToMesh(mesh_device))
    q_b = q_dev(q_nan, col_mapper)
    k = _replicated(mesh_device, torch.randn(1, 1, seq_len, HEAD_DIM), ttnn.bfloat8_b)
    v = _replicated(mesh_device, torch.randn(1, 1, seq_len, HEAD_DIM), ttnn.bfloat8_b)

    def sdpa(q):
        return ttnn.transformer.ring_distributed_scaled_dot_product_attention(
            q,
            k,
            v,
            ring_size=cols,
            scale=HEAD_DIM**-0.5,
            compute_kernel_config=ckc,
            program_config=progcfg,
            page_table=None,
            chunk_start_idx=None,
        )

    # The device copy really carries non-finite values in every non-owned slice (bf8 keeps NaN/Inf), finite elsewhere.
    for i, t in enumerate(ttnn.get_device_tensors(q_b)):
        qb = ttnn.to_torch(t)
        owned = column_slices(i % cols, cols)
        for s in range(2 * cols):
            finite = torch.isfinite(qb[:, :, s * sl : (s + 1) * sl])
            assert bool(finite.all()) if s in owned else not bool(finite.any()), f"device {i} slice {s}"

    out_a = sdpa(q_a)
    out_b = sdpa(q_b)
    ttnn.synchronize_device(mesh_device)
    bad = []
    for i, (ta, tb) in enumerate(zip(ttnn.get_device_tensors(out_a), ttnn.get_device_tensors(out_b))):
        a = ttnn.to_torch(ta)
        b = ttnn.to_torch(tb)
        if torch.isnan(b).any() or not torch.equal(a, b):
            bad.append(f"device {i} col {i % cols}: nan={bool(torch.isnan(b).any())} equal={torch.equal(a, b)}")
    assert not bad, "\n".join(bad[:16])
    logger.info(f"S={seq_len}: ring SDPA output {tuple(out_a.shape)} identical with NaN-filled non-owned Q slices")
    for t in (q_a, q_b, k, v, out_a, out_b):
        ttnn.deallocate(t)
