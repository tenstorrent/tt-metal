# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Full-width QK-norm building block (OLMo-2/3).

The norm is an RMSNorm over the WHOLE q / k projection (all heads), applied on the fused-QKV activation
before the head split. Under tensor parallelism the q columns are sharded across devices, so the
statistics must be reduced across the mesh (rms_norm_pre/post_all_gather). This test checks that
``models.common.rmsnorm.RMSNorm`` configured the way ``Attention`` configures it (sharded gamma,
distributed statistics on >1 device) matches a torch RMSNorm over the full width, on synthetic data —
no checkpoint needed. Real-weight coverage of the whole attention block is test_attention[_prefill].py.
"""
import os

import pytest
import torch

import ttnn
from models.common.rmsnorm import RMSNorm
from models.common.utility_functions import comp_pcc
from models.tt_transformers.tt.ccl import TT_CCL
from models.tt_transformers.tt.common import Mode


def torch_rms_norm(x, w, eps):
    xf = x.float()
    return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps) * w.float()).to(x.dtype)


@torch.no_grad()
@pytest.mark.parametrize(
    "mesh_device",
    [
        {"N150": (1, 1), "N300": (1, 2), "T3K": (1, 8), "P150": (1, 1), "P300": (1, 2), "P150x4": (1, 4)}.get(
            os.environ.get("MESH_DEVICE"), len(ttnn.get_device_ids())
        )
    ],
    indirect=True,
)
@pytest.mark.parametrize("rows", (32, 4096), ids=("decode-rows", "prefill-rows"))
@pytest.mark.parametrize("mode", [Mode.DECODE, Mode.PREFILL])
@pytest.mark.parametrize("device_params", [{"fabric_config": True}], indirect=True)
def test_qk_norm_full_width(mesh_device, rows, mode, reset_seeds, ensure_gc):
    n_heads, n_kv_heads, head_dim, eps = 40, 8, 128, 1e-6
    q_w, kv_w = n_heads * head_dim, n_kv_heads * head_dim
    num_devices = mesh_device.get_num_devices()
    assert q_w % (num_devices * 32) == 0 and kv_w % (num_devices * 32) == 0

    state_dict = {
        "layers.0.attention.q_norm.weight": torch.randn(q_w) * 0.5 + 1.0,
        "layers.0.attention.k_norm.weight": torch.randn(kv_w) * 0.5 + 1.0,
    }
    tt_ccl = TT_CCL(mesh_device)
    kwargs = dict(
        device=mesh_device,
        eps=eps,
        state_dict=state_dict,
        state_dict_prefix=None,
        weight_cache_path=None,
        weight_dtype=ttnn.bfloat16,
        is_distributed=(lambda m: True) if num_devices > 1 else None,
        tt_ccl=tt_ccl,
        ccl_topology=ttnn.Topology.Linear,
    )
    q_norm = RMSNorm(dim=q_w, weight_key="layers.0.attention.q_norm", **kwargs)
    k_norm = RMSNorm(dim=kv_w, weight_key="layers.0.attention.k_norm", **kwargs)

    # fused activation [1, 1, rows, q | k | v]; per-device shard = that device's head blocks (dim 3), as in Attention
    x = torch.randn(1, 1, rows, q_w + 2 * kv_w) * 3.0
    q_ref = torch_rms_norm(x[..., :q_w], state_dict["layers.0.attention.q_norm.weight"], eps)
    k_ref = torch_rms_norm(x[..., q_w : q_w + kv_w], state_dict["layers.0.attention.k_norm.weight"], eps)
    # shard the q block and the k block separately so each device holds a contiguous head chunk of both
    q_tt = ttnn.from_torch(
        x[..., :q_w],
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=3),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    k_tt = ttnn.from_torch(
        x[..., q_w : q_w + kv_w],
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=3),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    q_out = ttnn.to_torch(q_norm(q_tt, mode), mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=3))
    k_out = ttnn.to_torch(k_norm(k_tt, mode), mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=3))

    for name, out, ref in (("q", q_out, q_ref), ("k", k_out, k_ref)):
        passing, pcc = comp_pcc(ref, out.to(ref.dtype), 0.999)
        # a per-head norm would give a very different scale per head: check the global RMS is right too
        rms_ratio = (out.float().pow(2).mean() / ref.float().pow(2).mean()).sqrt().item()
        print(f"{name}: pcc={pcc} rms_ratio={rms_ratio:.4f} devices={num_devices} rows={rows} mode={mode}")
        assert passing, f"{name} PCC {pcc}"
        assert abs(rms_ratio - 1.0) < 0.02, f"{name} global RMS off by {rms_ratio}"
