# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Reject unsupported meshes without opening devices or loading weights."""

from unittest.mock import Mock

import pytest

import ttnn
from models.demos.llama31_8b_qb2.tt import model
from models.demos.llama31_8b_qb2.tt.decoder import LlamaDecoder, validate_qb2_mesh


@pytest.mark.parametrize("entrypoint", ["model", "decoder"])
@pytest.mark.parametrize("unsupported", ["architecture", "product", "shape", "device_count"])
def test_rejects_unsupported_mesh_before_loading_weights(monkeypatch, expect_error, entrypoint, unsupported):
    mesh = Mock(shape=(2, 2) if unsupported == "shape" else (1, 4))
    mesh.arch.return_value = (
        ttnn.device.Arch.WORMHOLE_B0 if unsupported == "architecture" else ttnn.device.Arch.BLACKHOLE
    )
    mesh.get_num_devices.return_value = 3 if unsupported == "device_count" else 4
    monkeypatch.setattr(
        ttnn.cluster,
        "get_cluster_type",
        lambda: ttnn.cluster.ClusterType.P150_X4 if unsupported == "product" else ttnn.cluster.ClusterType.P300_X2,
    )
    policy_loader = Mock()
    checkpoint_loader = Mock()
    monkeypatch.setattr(model, "load_precision_config", policy_loader)
    monkeypatch.setattr(model, "checkpoint_path", checkpoint_loader)
    with expect_error(ValueError, "Requires a Blackhole P300_X2 QB2"):
        if entrypoint == "model":
            model.LlamaModel(mesh)
        else:
            # None inputs ensure rejection precedes config/weight processing.
            LlamaDecoder.from_state_dict(
                None, hf_config=None, layer_idx=0, mesh_device=mesh, precision_policy=None, ccl=None
            )
    policy_loader.assert_not_called()
    checkpoint_loader.assert_not_called()


def test_accepts_qualified_qb2_mesh(monkeypatch):
    mesh = Mock(shape=(1, 4))
    mesh.arch.return_value = ttnn.device.Arch.BLACKHOLE
    mesh.get_num_devices.return_value = 4
    monkeypatch.setattr(ttnn.cluster, "get_cluster_type", lambda: ttnn.cluster.ClusterType.P300_X2)
    validate_qb2_mesh(mesh)
