# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Unit checks for the Qwen context-parallel long prefill (tt/qwen_cp_prefill.py, design B) on the WH Galaxy 8x4.

Layout B: mesh column c owns ring slices c and 7 - c (S/4 tokens, local order [slice c ; slice 7 - c]); mesh row r
owns q heads 8r..8r+7 and kv head r.

test_cp_q_prep_local: qwen_q_prep(src_local=True) reading a column-local fused QKV buffer [1, 1, S/4, 1280]
    against _qwen_qkv_prep_direct on the natural buffer: Q on the owned slices, K (all_rows over the local rows with
    the local rot mats) and V (column slice) against the matching rows of today's K / V. torch.equal on all 32
    devices.

Run on the Wormhole Galaxy (8x4 mesh):
    pytest models/demos/llama3_70b_galaxy/tests/test_qwen_cp_prefill.py -s
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.common.rmsnorm import RMSNorm
from models.demos.llama3_70b_galaxy.tt.custom_ops.qwen_q_prep import column_slices, qwen_q_prep
from models.demos.llama3_70b_galaxy.tt.llama_attention import TtLlamaAttention
from models.demos.llama3_70b_galaxy.tt.llama_common import get_rot_transformation_mat

N_LOCAL_HEADS = 8
HEAD_DIM = 128


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
    """The attributes the production methods below read, nothing else."""

    _qwen_local_q_setup = TtLlamaAttention._qwen_local_q_setup
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


def _replicated(mesh_device, t, dtype):
    return ttnn.from_torch(
        t,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )


def _local_rows(t, c, cols, dim=2):
    """Column c's local order [slice c ; slice 2 * cols - 1 - c] of a natural-order tensor along dim."""
    sl = t.shape[dim] // (2 * cols)
    s0, s1 = column_slices(c, cols)
    return torch.cat([t.narrow(dim, s0 * sl, sl), t.narrow(dim, s1 * sl, sl)], dim=dim)


@torch.no_grad()
@pytest.mark.parametrize("mesh_device", [(8, 4)], indirect=True)
@pytest.mark.parametrize("device_params", [{}], indirect=True)
@pytest.mark.parametrize("seq_len", [16384, 40960])
def test_cp_q_prep_local(mesh_device, seq_len):
    torch.manual_seed(0)
    rows, cols = list(mesh_device.shape)
    assert (rows, cols) == (8, 4)
    H, D = N_LOCAL_HEADS, HEAD_DIM
    width = (H + 2) * D
    cos = _replicated(mesh_device, torch.rand(1, 1, seq_len, D) * 2 - 1, ttnn.bfloat16)
    sin = _replicated(mesh_device, torch.rand(1, 1, seq_len, D) * 2 - 1, ttnn.bfloat16)
    trans_mat = _replicated(mesh_device, get_rot_transformation_mat(D), ttnn.bfloat16)
    attn = _Attention(mesh_device, torch.randn(D).to(torch.bfloat16), torch.randn(D).to(torch.bfloat16), trans_mat)
    rot = [cos, sin]
    sl = seq_len // (2 * cols)

    # One natural fused QKV, quantized once on the host so the local copy holds exactly the same bf8 values.
    xh = ttnn.to_torch(
        ttnn.from_torch(torch.randn(1, 1, seq_len, width) * 4.0, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT)
    ).to(torch.bfloat16)
    xqkv = _replicated(mesh_device, xh, ttnn.bfloat8_b)
    xqkv_local = ttnn.from_torch(
        torch.cat([_local_rows(xh, c, cols) for c in range(cols)], dim=0),
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(None, 0), mesh_shape=(rows, cols)),
    )
    q_ref, k_ref, v_ref = attn._qwen_qkv_prep_direct(xqkv, rot, seq_len)
    rot_local = attn._qwen_local_q_setup(rot, seq_len)[3]

    def run_local():
        q_new = ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, H, seq_len, D]), ttnn.bfloat8_b, ttnn.TILE_LAYOUT, mesh_device, ttnn.DRAM_MEMORY_CONFIG
        )
        k_loc = ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, 1, seq_len // cols, D]),
            ttnn.bfloat8_b,
            ttnn.TILE_LAYOUT,
            mesh_device,
            ttnn.DRAM_MEMORY_CONFIG,
        )
        qwen_q_prep(
            q_new,
            cos,
            sin,
            trans_mat,
            attn.q_norm.weight,
            eps=attn.q_norm.eps,
            src=xqkv_local,
            src_col=0,
            src_local=True,
        )
        qwen_q_prep(
            k_loc,
            rot_local[0],
            rot_local[1],
            trans_mat,
            attn.k_norm.weight,
            eps=attn.k_norm.eps,
            all_rows=True,
            src=xqkv_local,
            src_col=H * D,
        )
        v_loc = ttnn.slice(
            xqkv_local,
            [0, 0, 0, (H + 1) * D],
            [1, 1, seq_len // cols, (H + 2) * D],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        return q_new, k_loc, v_loc

    for tag in ("first call", "fresh buffers"):
        q_new, k_loc, v_loc = run_local()
        ttnn.synchronize_device(mesh_device)
        bad = []
        devs = zip(*(ttnn.get_device_tensors(t) for t in (q_ref, q_new, k_ref, k_loc, v_ref, v_loc)))
        for i, (qr, qn, kr, kn, vr, vn) in enumerate(devs):
            c = i % cols
            qr, qn = ttnn.to_torch(qr), ttnn.to_torch(qn)
            for s in column_slices(c, cols):
                rs = slice(s * sl, (s + 1) * sl)
                if not torch.equal(qr[:, :, rs], qn[:, :, rs]):
                    bad.append(f"{tag} device {i} Q slice {s} differs")
            if not torch.equal(_local_rows(ttnn.to_torch(kr), c, cols), ttnn.to_torch(kn)):
                bad.append(f"{tag} device {i} local K differs from today's K rows")
            if not torch.equal(_local_rows(ttnn.to_torch(vr), c, cols), ttnn.to_torch(vn)):
                bad.append(f"{tag} device {i} local V differs from today's V rows")
        assert not bad, "\n".join(bad[:16])
        logger.info(f"S={seq_len} {tag}: local-src Q (owned slices), local K, local V == today's on all 32 devices")
        for t in (q_new, k_loc, v_loc):
            ttnn.deallocate(t)
