# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""TtLlamaAttention._qwen_qkv_prep_direct (Q, K, V straight from the fused QKV buffer) against the stock path it
replaces: nlp_create_qkv_heads, then _qwen_local_q_prep_stock (Q, this column's two ring slices), _k_prep_stock
(K, all positions) and the V typecast bf8 -> bf8. All methods are the production ones, bound to a stub that carries
only the attributes they read.

Every device must match exactly: Q on the owned slices (the direct path leaves the other slices unwritten; ring SDPA
does not read them), K and V on all positions. A second call on fresh tensors (new buffer addresses) runs the
generic_op program-cache hit path. Prints the wall time of each path.

Run on the Wormhole Galaxy (8x4 mesh):
    pytest models/demos/llama3_70b_galaxy/tests/test_qwen_qkv_direct_op.py -s
"""

import time

import pytest
import torch
from loguru import logger

import ttnn
from models.common.rmsnorm import RMSNorm
from models.demos.llama3_70b_galaxy.tt.custom_ops.qwen_q_prep import column_slices
from models.demos.llama3_70b_galaxy.tt.llama_attention import TtLlamaAttention
from models.demos.llama3_70b_galaxy.tt.llama_common import get_rot_transformation_mat

N_LOCAL_HEADS = 8
HEAD_DIM = 128
TIMED_ITERS = 5


def _norm(mesh_device, name, gamma_torch):
    return RMSNorm(
        device=mesh_device,
        dim=HEAD_DIM,
        state_dict={f"layers.0.attention.{name}.weight": gamma_torch},
        state_dict_prefix=None,
        weight_dtype=ttnn.bfloat16,
        weight_key=f"layers.0.attention.{name}",
        eps=1e-06,
    )


class _Attention:
    """The attributes the methods below read, nothing else."""

    _qwen_local_q_setup = TtLlamaAttention._qwen_local_q_setup
    _qwen_local_q_prep_stock = TtLlamaAttention._qwen_local_q_prep_stock
    _k_prep_stock = TtLlamaAttention._k_prep_stock
    _qwen_qkv_prep_direct = TtLlamaAttention._qwen_qkv_prep_direct

    def __init__(self, mesh_device, q_gamma, k_gamma, trans_mat):
        self.mesh_device = mesh_device
        self.tt_ccl = type("TtCclStub", (), {})()
        self.n_local_heads = N_LOCAL_HEADS
        self.n_local_kv_heads = 1
        self.head_dim = HEAD_DIM
        self.qk_norm = True
        self.q_norm = _norm(mesh_device, "q_norm", q_gamma)
        self.k_norm = _norm(mesh_device, "k_norm", k_gamma)
        self.transformation_mats = {"prefill": trans_mat}

    def stock(self, xqkv, rot_mats, seq_len):
        q, k, v = ttnn.experimental.nlp_create_qkv_heads(
            xqkv,
            num_heads=self.n_local_heads,
            num_kv_heads=self.n_local_kv_heads,
            transpose_k_heads=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        q = self._qwen_local_q_prep_stock(q, rot_mats, seq_len)
        k = self._k_prep_stock(k, rot_mats)
        v8 = ttnn.typecast(v, dtype=ttnn.bfloat8_b)
        ttnn.deallocate(v)
        return q, k, v8


def _replicated(mesh_device, t, dtype):
    return ttnn.from_torch(
        t,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )


@torch.no_grad()
@pytest.mark.parametrize("mesh_device", [(8, 4)], indirect=True)
@pytest.mark.parametrize(
    "device_params", [{"dispatch_core_axis": ttnn.DispatchCoreAxis.COL, "worker_l1_size": 1345000}], indirect=True
)  # production dispatch / L1
@pytest.mark.parametrize("seq_len", [16384, 40960])
def test_qwen_qkv_direct_bit_exact(mesh_device, seq_len):
    torch.manual_seed(0)
    rows, cols = list(mesh_device.shape)
    assert (rows, cols) == (8, 4)
    width = (N_LOCAL_HEADS + 2) * HEAD_DIM
    cos = _replicated(mesh_device, torch.rand(1, 1, seq_len, HEAD_DIM) * 2 - 1, ttnn.bfloat16)
    sin = _replicated(mesh_device, torch.rand(1, 1, seq_len, HEAD_DIM) * 2 - 1, ttnn.bfloat16)
    trans_mat = _replicated(mesh_device, get_rot_transformation_mat(HEAD_DIM), ttnn.bfloat16)
    attn = _Attention(
        mesh_device,
        torch.randn(HEAD_DIM).to(torch.bfloat16),
        torch.randn(HEAD_DIM).to(torch.bfloat16),
        trans_mat,
    )
    rot = [cos, sin]
    sl = seq_len // (2 * cols)

    def check(tag):
        xqkv = _replicated(mesh_device, torch.randn(1, 1, seq_len, width) * 4.0, ttnn.bfloat8_b)
        q_ref, k_ref, v_ref = attn.stock(xqkv, rot, seq_len)
        q_new, k_new, v_new = attn._qwen_qkv_prep_direct(xqkv, rot, seq_len)
        ttnn.synchronize_device(mesh_device)
        bad = []
        devs = zip(*(ttnn.get_device_tensors(t) for t in (q_ref, q_new, k_ref, k_new, v_ref, v_new)))
        for i, (qr, qn, kr, kn, vr, vn) in enumerate(devs):
            qr, qn = ttnn.to_torch(qr), ttnn.to_torch(qn)
            for s in column_slices(i % cols, cols):
                rs = slice(s * sl, (s + 1) * sl)
                if not torch.equal(qr[:, :, rs], qn[:, :, rs]):
                    bad.append(f"{tag} device {i} Q slice {s} differs")
            if not torch.equal(ttnn.to_torch(kr), ttnn.to_torch(kn)):
                bad.append(f"{tag} device {i} K differs")
            if not torch.equal(ttnn.to_torch(vr), ttnn.to_torch(vn)):
                bad.append(f"{tag} device {i} V differs")
        assert not bad, "\n".join(bad[:16])
        logger.info(f"S={seq_len} {tag}: direct Q (owned slices), K, V == stock on all 32 devices")
        return xqkv, [q_ref, k_ref, v_ref, q_new, k_new, v_new]

    xqkv, keep = check("first call")
    xqkv2, keep2 = check("fresh buffers")
    assert keep2[3].buffer_address() != keep[3].buffer_address()
    for t in keep + keep2 + [xqkv2]:
        ttnn.deallocate(t)

    def time_it(fn):
        for t in fn():
            ttnn.deallocate(t)
        ttnn.synchronize_device(mesh_device)
        start = time.perf_counter()
        for _ in range(TIMED_ITERS):
            for t in fn():
                ttnn.deallocate(t)
        ttnn.synchronize_device(mesh_device)
        return (time.perf_counter() - start) / TIMED_ITERS * 1e3

    t_ref = time_it(lambda: attn.stock(xqkv, rot, seq_len))
    t_new = time_it(lambda: attn._qwen_qkv_prep_direct(xqkv, rot, seq_len))
    logger.info(f"S={seq_len}: stock {t_ref:.3f} ms, direct {t_new:.3f} ms (wall, mean of {TIMED_ITERS})")
