# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest
import torch

from models.demos.deepseek_v4_flash.converter import convert_hf_checkpoint
from models.demos.deepseek_v4_flash.cpu_reference import combine_routed_experts
from models.demos.deepseek_v4_flash.expert_abi import load_packed_expert_weight
from models.demos.deepseek_v4_flash.expert_plan import plan_batch1_decode_expert_placements
from models.demos.deepseek_v4_flash.manifest import load_tt_manifest
from models.demos.deepseek_v4_flash.synthetic import generate_tiny_hf_checkpoint

ttnn = pytest.importorskip("ttnn")

from models.demos.deepseek_v4_flash.ttnn_routed_expert import TtRoutedExpertMLP

pytestmark = pytest.mark.t3k_compat


def _skip_unless_t3k() -> None:
    try:
        cluster_type = ttnn.cluster.get_cluster_type()
        num_devices = ttnn.get_num_devices()
    except Exception as exc:
        pytest.skip(f"Unable to query TT cluster for T3K expert-plan test: {exc}")

    if cluster_type != ttnn.cluster.ClusterType.T3K or num_devices != 8:
        pytest.skip(f"Requires T3K with 8 devices, found cluster_type={cluster_type}, num_devices={num_devices}")


@pytest.fixture(scope="module")
def tiny_tt_preprocessed_checkpoint(tmp_path_factory: pytest.TempPathFactory) -> Path:
    artifact_root = os.environ.get("DSV4_FLASH_ARTIFACT_DIR")
    if artifact_root:
        base = Path(artifact_root) / "pytest" / f"deepseek_v4_flash_t3k_expert_plan_{os.getpid()}"
        if base.exists():
            shutil.rmtree(base)
        base.mkdir(parents=True)
    else:
        base = tmp_path_factory.mktemp("deepseek_v4_flash_t3k_expert_plan")

    source = generate_tiny_hf_checkpoint(base / "hf_source", num_hidden_layers=1)
    return convert_hf_checkpoint(source, base / "tt_preprocessed")


@pytest.fixture
def t3k_mesh():
    _skip_unless_t3k()
    mesh_device = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(2, 4))
    try:
        yield mesh_device
    finally:
        for submesh in mesh_device.get_submeshes():
            ttnn.close_mesh_device(submesh)
        ttnn.close_mesh_device(mesh_device)


def test_t3k_planned_primary_replica_runs_tiny_routed_expert(
    tiny_tt_preprocessed_checkpoint: Path,
    t3k_mesh,
) -> None:
    plan = plan_batch1_decode_expert_placements((2, 4), (0, 2), replicas_per_expert=1)
    expert_id = 2
    primary_replica = plan.placements[1].primary_replica

    assert primary_replica.expert_id == expert_id
    assert primary_replica.mesh_coord == (1, 0)
    assert plan.devices_for_expert(expert_id) == ((1, 0),)

    submesh = t3k_mesh.create_submesh(
        ttnn.MeshShape(1, 1),
        offset=ttnn.MeshCoordinate(*primary_replica.mesh_coord),
    )

    manifest = load_tt_manifest(tiny_tt_preprocessed_checkpoint)
    weights = {
        projection: load_packed_expert_weight(
            tiny_tt_preprocessed_checkpoint, layer=0, expert=expert_id, projection=projection
        ).dequantize(dtype=torch.bfloat16)
        for projection in ("w1", "w2", "w3")
    }

    hidden_size = weights["w1"].shape[-1]
    intermediate_size = weights["w1"].shape[0]
    torch_input = torch.linspace(-0.2, 0.2, steps=32 * hidden_size, dtype=torch.float32).reshape(1, 1, 32, hidden_size)
    torch_input = torch_input.to(torch.bfloat16)
    route_weights = torch.linspace(0.25, 1.0, steps=32, dtype=torch.float32).reshape(1, 32, 1)
    route_indices = torch.full((1, 32, 1), expert_id, dtype=torch.int64)
    expected = combine_routed_experts(
        torch_input.reshape(1, 32, hidden_size),
        route_weights,
        route_indices,
        {expert_id: (weights["w1"], weights["w2"], weights["w3"])},
        swiglu_limit=float(manifest["config"]["swiglu_limit"]),
    ).reshape_as(torch_input)

    tt_input = ttnn.from_torch(torch_input, device=submesh, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    tt_route_weights = ttnn.from_torch(
        route_weights.reshape(1, 1, 32, 1).expand(1, 1, 32, intermediate_size).to(torch.bfloat16),
        device=submesh,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
    )
    module = TtRoutedExpertMLP.from_preprocessed(
        tiny_tt_preprocessed_checkpoint,
        device=submesh,
        expert=expert_id,
    )
    torch_output = ttnn.to_torch(module(tt_input, route_weight=tt_route_weights))

    torch.testing.assert_close(torch_output.float(), expected.float(), rtol=5e-2, atol=5e-2)
