# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Reject unsupported meshes before loading weights or constructing the decoder."""

import re
from unittest.mock import Mock

import pytest

import ttnn
from models.demos.qwen38_27b_qb2.tt import model
from models.demos.qwen38_27b_qb2.tt.decoder_tp import Qwen38TPDecoder, resolve_mesh_tp

# A qualified platform is the whole tuple, so each rejection case perturbs exactly one element of
# an otherwise-valid mesh. The Wormhole rows mirror the Blackhole ones so neither platform's gate
# can regress unnoticed.
_QB2 = (ttnn.Arch.BLACKHOLE, ttnn.cluster.ClusterType.P300_X2, 4, (1, 4))
_T3K = (ttnn.Arch.WORMHOLE_B0, ttnn.cluster.ClusterType.T3K, 8, (1, 8))

_REJECTED = {
    "qb2_architecture": (ttnn.Arch.WORMHOLE_B0, ttnn.cluster.ClusterType.P300_X2, 4, (1, 4)),
    "qb2_product": (ttnn.Arch.BLACKHOLE, ttnn.cluster.ClusterType.P150_X4, 4, (1, 4)),
    "qb2_shape": (ttnn.Arch.BLACKHOLE, ttnn.cluster.ClusterType.P300_X2, 4, (2, 2)),
    "qb2_device_count": (ttnn.Arch.BLACKHOLE, ttnn.cluster.ClusterType.P300_X2, 3, (1, 4)),
    "t3k_architecture": (ttnn.Arch.BLACKHOLE, ttnn.cluster.ClusterType.T3K, 8, (1, 8)),
    "t3k_product": (ttnn.Arch.WORMHOLE_B0, ttnn.cluster.ClusterType.N300, 8, (1, 8)),
    "t3k_shape": (ttnn.Arch.WORMHOLE_B0, ttnn.cluster.ClusterType.T3K, 8, (2, 4)),
    "t3k_device_count": (ttnn.Arch.WORMHOLE_B0, ttnn.cluster.ClusterType.T3K, 4, (1, 8)),
}


def _mock_mesh(monkeypatch, arch, cluster_type, num_devices, shape):
    mesh = Mock(shape=shape)
    mesh.arch.return_value = arch
    mesh.get_num_devices.return_value = num_devices
    monkeypatch.setattr(ttnn.cluster, "get_cluster_type", lambda: cluster_type)
    return mesh


@pytest.mark.parametrize("entrypoint", ["model", "decoder"])
@pytest.mark.parametrize("unsupported", sorted(_REJECTED))
def test_rejects_unsupported_mesh_before_loading_weights(monkeypatch, expect_error, entrypoint, unsupported):
    mesh = _mock_mesh(monkeypatch, *_REJECTED[unsupported])
    precision_loader = Mock()
    checkpoint_loader = Mock()
    monkeypatch.setattr(model, "load_precision", precision_loader)
    monkeypatch.setattr(model, "checkpoint_path", checkpoint_loader)
    # expect_error matches its message as a regex -- the mesh shapes in the message are
    # parenthesised, so escape them rather than letting them read as capture groups.
    expected = re.escape("requires a Blackhole P300_X2 QB2 in a (1, 4) mesh or a Wormhole T3K in a (1, 8) mesh")
    with expect_error(ValueError, expected):
        if entrypoint == "model":
            model.Qwen38Model(mesh)
        else:
            Qwen38TPDecoder.from_state_dict(None, hf_config=None, layer_idx=0, mesh_device=mesh)
    precision_loader.assert_not_called()
    checkpoint_loader.assert_not_called()


@pytest.mark.parametrize("platform, expected_tp", [(_QB2, 4), (_T3K, 8)], ids=["qb2_tp4", "t3k_tp8"])
def test_accepts_qualified_mesh_and_reports_tp(monkeypatch, platform, expected_tp):
    assert resolve_mesh_tp(_mock_mesh(monkeypatch, *platform)) == expected_tp
