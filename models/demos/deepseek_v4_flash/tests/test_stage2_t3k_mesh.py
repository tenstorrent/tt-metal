# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
import torch

from models.demos.deepseek_v4_flash.mesh_config import ModeConfig, mesh_for_shape, mesh_for_ttnn_mesh

ttnn = pytest.importorskip("ttnn")

pytestmark = pytest.mark.t3k_compat


def _skip_unless_t3k() -> None:
    try:
        cluster_type = ttnn.cluster.get_cluster_type()
        num_devices = ttnn.get_num_devices()
    except Exception as exc:
        pytest.skip(f"Unable to query TT cluster for T3K mesh test: {exc}")

    if cluster_type != ttnn.cluster.ClusterType.T3K or num_devices != 8:
        pytest.skip(f"Requires T3K with 8 devices, found cluster_type={cluster_type}, num_devices={num_devices}")


@pytest.fixture
def t3k_mesh(request):
    _skip_unless_t3k()
    mesh_shape = request.param
    mesh_device = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(*mesh_shape))
    try:
        yield mesh_device
    finally:
        for submesh in mesh_device.get_submeshes():
            ttnn.close_mesh_device(submesh)
        ttnn.close_mesh_device(mesh_device)


@pytest.mark.parametrize("t3k_mesh", [pytest.param((1, 8), id="1x8"), pytest.param((2, 4), id="2x4")], indirect=True)
def test_deepseek_v4_flash_t3k_mesh_semantics_and_roundtrip(t3k_mesh):
    mesh_shape = tuple(t3k_mesh.shape)
    mesh_config = mesh_for_ttnn_mesh(t3k_mesh)

    assert t3k_mesh.get_num_devices() == 8
    assert mesh_config.to_manifest_dict() == mesh_for_shape(mesh_shape).to_manifest_dict()
    if mesh_shape == (1, 8):
        assert mesh_config.decode == ModeConfig(tp=8, ep=1, sp=1)
        assert mesh_config.prefill == ModeConfig(tp=8, ep=1, sp=1)
    elif mesh_shape == (2, 4):
        assert mesh_config.decode == ModeConfig(tp=4, ep=2, sp=1)
        assert mesh_config.prefill == ModeConfig(tp=4, ep=1, sp=2)
    else:
        raise AssertionError(f"Unexpected DeepSeek T3K mesh shape: {mesh_shape}")

    rows, cols = mesh_shape
    torch_input = torch.arange(rows * 1 * 32 * cols * 32, dtype=torch.float32).reshape(rows, 1, 32, cols * 32)
    torch_input = torch_input.to(torch.bfloat16)

    tt_input = ttnn.from_torch(
        torch_input,
        device=t3k_mesh,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ShardTensor2dMesh(t3k_mesh, dims=(0, 3), mesh_shape=mesh_shape),
    )

    assert len(ttnn.get_device_tensors(tt_input)) == 8
    assert tuple(tt_input.shape) == (1, 1, 32, 32)

    torch_output = ttnn.to_torch(
        ttnn.from_device(tt_input),
        mesh_composer=ttnn.ConcatMesh2dToTensor(t3k_mesh, mesh_shape=mesh_shape, dims=(0, 3)),
    )
    torch.testing.assert_close(torch_output, torch_input)
