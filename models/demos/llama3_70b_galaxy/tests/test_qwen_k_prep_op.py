# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""qwen_q_prep(all_rows=True) on the K path against the stock chain it replaces,
TtLlamaAttention._k_prep_stock (typecast bf8 -> bf16, k_norm, rotary_embedding_llama, typecast bf16 -> bf8), bound
to a stub that carries only the attributes it reads. Also checks that the V typecast bf8 -> bf8 is a no-op (output
equals input on every device).

Every device's whole K buffer must match exactly, for a first call and for fresh tensors at new buffer addresses
(program-cache hit path). Prints the wall time of each path.

Run on the Wormhole Galaxy (8x4 mesh):
    pytest models/demos/llama3_70b_galaxy/tests/test_qwen_k_prep_op.py -s
"""

import time

import pytest
import torch
from loguru import logger

import ttnn
from models.common.rmsnorm import RMSNorm
from models.demos.llama3_70b_galaxy.tt.custom_ops.qwen_q_prep import qwen_q_prep
from models.demos.llama3_70b_galaxy.tt.llama_attention import TtLlamaAttention
from models.demos.llama3_70b_galaxy.tt.llama_common import get_rot_transformation_mat

HEAD_DIM = 128
ROT_LEN = 40960
TIMED_ITERS = 5


def _replicated(mesh_device, t, dtype, layout=ttnn.TILE_LAYOUT):
    return ttnn.from_torch(
        t,
        dtype=dtype,
        layout=layout,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )


class _RefAttention:
    """The attributes TtLlamaAttention._k_prep_stock reads, nothing else."""

    _k_prep_stock = TtLlamaAttention._k_prep_stock

    def __init__(self, k_norm, trans_mat):
        self.qk_norm = True
        self.k_norm = k_norm
        self.transformation_mats = {"prefill": trans_mat}


@torch.no_grad()
@pytest.mark.parametrize("mesh_device", [(8, 4)], indirect=True)
@pytest.mark.parametrize(
    "device_params", [{"dispatch_core_axis": ttnn.DispatchCoreAxis.COL, "worker_l1_size": 1345000}], indirect=True
)  # production dispatch / L1
@pytest.mark.parametrize("seq_len", [16384, 40960])
def test_qwen_k_prep_bit_exact(mesh_device, seq_len):
    torch.manual_seed(0)
    k_torch = torch.randn(1, 1, seq_len, HEAD_DIM) * 4.0
    v_torch = torch.randn(1, 1, seq_len, HEAD_DIM) * 4.0
    cos_torch = torch.rand(1, 1, ROT_LEN, HEAD_DIM) * 2 - 1
    sin_torch = torch.rand(1, 1, ROT_LEN, HEAD_DIM) * 2 - 1
    gamma_torch = torch.randn(HEAD_DIM).to(torch.bfloat16)

    cos_full = _replicated(mesh_device, cos_torch, ttnn.bfloat16)
    sin_full = _replicated(mesh_device, sin_torch, ttnn.bfloat16)
    # The model's rot mats for a 40960 prefill are [1, 1, S, 128]; the stock RoPE needs exactly S rows.
    cos = ttnn.slice(cos_full, [0, 0, 0, 0], [1, 1, seq_len, HEAD_DIM])
    sin = ttnn.slice(sin_full, [0, 0, 0, 0], [1, 1, seq_len, HEAD_DIM])
    trans_mat = _replicated(mesh_device, get_rot_transformation_mat(HEAD_DIM), ttnn.bfloat16)
    k_norm = RMSNorm(
        device=mesh_device,
        dim=HEAD_DIM,
        state_dict={"layers.0.attention.k_norm.weight": gamma_torch},
        state_dict_prefix=None,
        weight_dtype=ttnn.bfloat16,
        weight_key="layers.0.attention.k_norm",
        eps=1e-06,
    )

    ref = _RefAttention(k_norm, trans_mat)

    def check(k_in_torch, tag):
        k_ref_in = _replicated(mesh_device, k_in_torch, ttnn.bfloat8_b)
        k_op = _replicated(mesh_device, k_in_torch, ttnn.bfloat8_b)
        k_ref = ref._k_prep_stock(k_ref_in, [cos, sin])  # deallocates k_ref_in
        out = qwen_q_prep(k_op, cos_full, sin_full, trans_mat, k_norm.weight, eps=k_norm.eps, all_rows=True)
        assert out.buffer_address() == k_op.buffer_address()
        ttnn.synchronize_device(mesh_device)
        mismatches = []
        for i, (a, b) in enumerate(zip(ttnn.get_device_tensors(k_ref), ttnn.get_device_tensors(k_op))):
            a, b = ttnn.to_torch(a), ttnn.to_torch(b)
            if not torch.equal(a, b):
                diff = (a.float() - b.float()).abs()
                mismatches.append(f"{tag} K device {i}: {int((diff > 0).sum())} values differ, max {diff.max():.3e}")
        assert not mismatches, "\n".join(mismatches[:16])
        logger.info(f"S={seq_len} {tag}: qwen_q_prep(all_rows) == _k_prep_stock on all devices")
        ttnn.deallocate(k_ref)
        return k_op

    k_first = check(k_torch, "first call")
    k_op = check(torch.randn(1, 1, seq_len, HEAD_DIM) * 4.0, "fresh buffers")
    assert k_op.buffer_address() != k_first.buffer_address()
    ttnn.deallocate(k_first)

    v = _replicated(mesh_device, v_torch, ttnn.bfloat8_b)
    v_cast = ttnn.typecast(v, dtype=ttnn.bfloat8_b)
    for i, (vi, vo) in enumerate(zip(ttnn.get_device_tensors(v), ttnn.get_device_tensors(v_cast))):
        assert torch.equal(ttnn.to_torch(vi), ttnn.to_torch(vo)), f"V device {i}: typecast bf8 -> bf8 changed values"
    logger.info(f"S={seq_len}: V typecast bf8 -> bf8 == identity on all devices")

    k_ref_src = _replicated(mesh_device, k_torch, ttnn.bfloat8_b)

    def run_ref():
        k_in = ttnn.clone(k_ref_src)
        ttnn.deallocate(ref._k_prep_stock(k_in, [cos, sin]))
        return mesh_device

    def run_op():
        qwen_q_prep(k_op, cos_full, sin_full, trans_mat, k_norm.weight, eps=k_norm.eps, all_rows=True)
        return mesh_device

    def time_it(fn):
        fn()
        ttnn.synchronize_device(mesh_device)
        start = time.perf_counter()
        for _ in range(TIMED_ITERS):
            fn()
        ttnn.synchronize_device(mesh_device)
        return (time.perf_counter() - start) / TIMED_ITERS * 1e3

    t_ref = time_it(run_ref)
    t_op = time_it(run_op)
    logger.info(
        f"S={seq_len}: K chain (incl. a clone) {t_ref:.3f} ms, qwen_q_prep(all_rows) {t_op:.3f} ms (wall, mean of {TIMED_ITERS})"
    )
