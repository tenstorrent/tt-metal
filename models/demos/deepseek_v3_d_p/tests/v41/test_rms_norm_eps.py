# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""DeepSeek-V4.1 RMSNorm epsilon contract.

V4.1 sets ``rms_norm_eps=1e-20`` for every RMSNorm (attn/ffn/q/kv/compressor/indexer).
The reference (HF ``inference/model.py`` ``RMSNorm``) computes ``x * rsqrt(mean(x^2) + eps)`` in
fp32. These tests check that the device norms used by deepseek_v3_d_p honor that epsilon:
output matches the fp32 reference, stays finite (including all-zero rows) and repeats
bit-identically. The tiny-magnitude input has mean(x^2) ~ 1e-12, where the V4 default eps
(1e-6) shrinks the output ~1000x. PCC is scale-invariant, so the output magnitude is checked
separately, and the local test runs eps=1e-6 as a control to prove that check is sensitive.
"""

import pytest
import torch

import ttnn
from models.demos.deepseek_v3_d_p.tests.fabric_profiles import fabric2d_device_params
from models.demos.deepseek_v3_d_p.tt.tt_ccl import per_axis_topology
from models.demos.deepseek_v3_d_p.tt.tt_distributed_rms_norm import TtDistributedRmsNorm
from tests.ttnn.utils_for_testing import comp_pcc

V41_RMS_NORM_EPS = 1e-20
PCC = 0.999
NORM_RATIO_TOL = 0.02
SEQ = 256

MESH_2X4 = pytest.param(
    (2, 4),
    fabric2d_device_params(),
    marks=pytest.mark.requires_mesh_topology(mesh_shape=(2, 4), topology="mesh-2x4"),
    id="fabric2d-mesh-2x4",
)


def _reference_rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    x = x.float()
    return weight.float() * (x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps))


def _input(kind: str, dim: int) -> torch.Tensor:
    torch.manual_seed(1234)
    x = torch.randn(1, 1, SEQ, dim)
    if kind == "tiny":
        x = x * 1e-6
    elif kind == "zero_rows":
        x[..., ::7, :] = 0.0
    return x.to(torch.bfloat16)


def _norm_ratio(reference: torch.Tensor, actual: torch.Tensor) -> float:
    return (actual.float().norm() / reference.norm()).item()


def _check(reference: torch.Tensor, first: torch.Tensor, second: torch.Tensor, kind: str) -> None:
    assert torch.isfinite(first).all(), "device RMSNorm produced non-finite values"
    assert torch.equal(first, second), "device RMSNorm is not bit-identical across repeated runs"
    if kind == "zero_rows":
        assert torch.all(first[..., ::7, :] == 0), "all-zero rows must normalize to zero"
    passed, pcc = comp_pcc(reference, first.float(), PCC)
    assert passed, f"PCC {pcc} < {PCC}"
    ratio = _norm_ratio(reference, first)
    assert abs(ratio - 1) <= NORM_RATIO_TOL, f"output magnitude ratio {ratio} vs reference"


@pytest.mark.parametrize("kind", ["randn", "tiny", "zero_rows"])
@pytest.mark.parametrize("dim", [128, 512, 1280], ids=["index_k_norm", "kv_norm", "q_norm"])
@pytest.mark.parametrize("mesh_device, device_params", [MESH_2X4], indirect=True)
def test_local_rms_norm_eps(mesh_device, device_params, dim, kind):
    x = _input(kind, dim)
    weight = torch.empty(dim).uniform_(0.5, 1.5)
    reference = _reference_rms_norm(x, weight, V41_RMS_NORM_EPS)

    replicate = ttnn.ReplicateTensorToMesh(mesh_device)
    tt_x = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device, mesh_mapper=replicate)
    tt_w = ttnn.from_torch(
        weight.reshape(1, 1, dim // 32, 32).to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        mesh_mapper=replicate,
    )

    def run(eps: float = V41_RMS_NORM_EPS) -> torch.Tensor:
        out = ttnn.rms_norm(tt_x, weight=tt_w, epsilon=eps)
        return ttnn.to_torch(ttnn.get_device_tensors(out)[0])

    _check(reference, run(), run(), kind)
    if kind == "tiny":
        control = _norm_ratio(reference, run(eps=1e-6))
        assert abs(control - 1) > NORM_RATIO_TOL, f"eps=1e-6 control ratio {control}: tiny case is insensitive"


@pytest.mark.parametrize("kind", ["randn", "tiny", "zero_rows"])
@pytest.mark.parametrize("mesh_device, device_params", [MESH_2X4], indirect=True)
def test_distributed_rms_norm_eps(mesh_device, device_params, kind):
    emb_dim = 5120
    x = _input(kind, emb_dim)
    weight = torch.empty(emb_dim).uniform_(0.5, 1.5)
    reference = _reference_rms_norm(x, weight, V41_RMS_NORM_EPS)

    mesh_shape = tuple(mesh_device.shape)
    norm = TtDistributedRmsNorm(
        mesh_device=mesh_device,
        emb_dim=emb_dim,
        epsilon=V41_RMS_NORM_EPS,
        torch_weight=weight,
        cluster_axis=1,
        num_links=1,
        topology=per_axis_topology(device_params["fabric_config"])[1],
    )
    tt_x = ttnn.from_torch(
        x,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=mesh_shape, dims=(None, 3)),
    )

    def run() -> torch.Tensor:
        out = norm(tt_x)
        full = ttnn.to_torch(
            out, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, mesh_shape=mesh_shape, dims=(0, 3))
        )
        return full[:1]

    _check(reference, run(), run(), kind)
