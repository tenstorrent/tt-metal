# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""dit_fused_distributed_rmsnorm with cluster_axis=None: per-head RMSNorm + RoPE computed locally on every device."""

import pytest
import torch

import ttnn

from ...utils.mochi import get_rot_transformation_mat, stack_cos_sin

SINGLE_DEVICE = [pytest.param((1, 1), {"l1_small_size": 65536}, id="single_device")]
MESH = [
    pytest.param(
        (4, 8),
        {
            "fabric_config": ttnn.FabricConfig.FABRIC_1D,
            "require_exact_physical_num_devices": True,
            "l1_small_size": 65536,
        },
        id="mesh4x8",
    )
]
SEQ, HEAD_DIM, EPS = 1824, 64, 1e-6


def _pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.corrcoef(torch.stack([a.flatten().float(), b.flatten().float()]))[0, 1])


def _reference(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, trans: torch.Tensor) -> torch.Tensor:
    x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + EPS)
    rotated = (x.reshape(*x.shape[:-1], HEAD_DIM // 32, 32) @ trans).reshape(x.shape)
    return x * cos + rotated * sin


@pytest.mark.parametrize(
    ("mesh_device", "device_params"), SINGLE_DEVICE + MESH, indirect=["mesh_device", "device_params"]
)
@pytest.mark.parametrize("heads", [32, 64], ids=["heads32", "heads64_two_tiles_folded"])
def test_local_rmsnorm_rope(mesh_device, heads):
    torch.manual_seed(0)
    x = torch.randn(1, heads, SEQ, HEAD_DIM).to(torch.bfloat16).float()
    angles = torch.rand(1, 1, SEQ, HEAD_DIM // 2) * 2 * torch.pi
    cos, sin = stack_cos_sin(angles.cos(), angles.sin())
    trans = get_rot_transformation_mat()
    expected = _reference(x, cos, sin, trans[0, 0])

    def to_device(t, dtype):
        mapper = ttnn.ReplicateTensorToMesh(mesh_device)
        return ttnn.from_torch(t, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=mesh_device, mesh_mapper=mapper)

    x_tt = to_device(x, ttnn.bfloat16)
    cos_tt, sin_tt = to_device(cos, ttnn.float32), to_device(sin, ttnn.float32)
    trans_tt = to_device(trans, ttnn.bfloat16)
    rope = dict(transformation_mat=trans_tt, rope_cos=cos_tt, rope_sin=sin_tt)
    compute_config = ttnn.init_device_compute_kernel_config(
        mesh_device.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True
    )

    assert ttnn.experimental.dit_fused_distributed_rmsnorm_create_stats_buffer(x_tt, None, mesh_device, **rope) is None

    def run():
        out = ttnn.experimental.dit_fused_distributed_rmsnorm(
            x_tt,
            None,
            mesh_device,
            [],
            epsilon=EPS,
            num_heads_per_device=1,
            per_head_norm=False,
            compute_kernel_config=compute_config,
            **rope,
        )
        return ttnn.to_torch(out, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0)).float()

    first, second = run(), run()
    assert torch.equal(first, second), "local RMSNorm + RoPE is not deterministic across launches"
    assert first.shape == (mesh_device.get_num_devices(), heads, SEQ, HEAD_DIM)
    for device_index, out in enumerate(first):
        pcc = _pcc(out, expected[0])
        assert pcc > 0.999, f"device {device_index}: pcc={pcc}"
