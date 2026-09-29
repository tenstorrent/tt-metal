# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Reject unsupported meshes before loading weights or constructing the decoder."""

import re
from unittest.mock import Mock

import pytest

import ttnn
from models.demos.qwen38_27b_t3k.tt import model
from models.demos.qwen38_27b_t3k.tt.decoder_tp import (
    _TP_POLICY,
    Qwen38TPDecoder,
    kv_head_owners,
    native_mesh_shape,
    resolve_mesh_tp,
)

# A qualified platform is the whole tuple, so each rejection case perturbs exactly one element of
# an otherwise-valid mesh. The Wormhole rows mirror the Blackhole ones so neither platform's gate
# can regress unnoticed.
_QB2 = (ttnn.Arch.BLACKHOLE, ttnn.cluster.ClusterType.P300_X2, 4, (1, 4))
_T3K = (ttnn.Arch.WORMHOLE_B0, ttnn.cluster.ClusterType.T3K, 8, (1, 8))

_REJECTED = {
    # A qualified Blackhole QB2 mesh belongs to the sibling implementation, not this one.
    "blackhole_qb2": _QB2,
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
    expected = re.escape("requires a Wormhole T3K in a (1, 8) mesh")
    with expect_error(ValueError, expected):
        if entrypoint == "model":
            model.Qwen38Model(mesh)
        else:
            Qwen38TPDecoder.from_state_dict(None, hf_config=None, layer_idx=0, mesh_device=mesh)
    precision_loader.assert_not_called()
    checkpoint_loader.assert_not_called()


@pytest.mark.parametrize("platform, expected_tp", [(_T3K, 8)], ids=["t3k_tp8"])
def test_accepts_qualified_mesh_and_reports_tp(monkeypatch, platform, expected_tp):
    assert resolve_mesh_tp(_mock_mesh(monkeypatch, *platform)) == expected_tp


# Qwen3.8-27B: 24 Q heads, 4 KV heads, GQA group 6.
_NUM_Q, _NUM_KV = 24, 4


def test_kv_heads_shard_evenly_when_devices_do_not_exceed_them():
    assert kv_head_owners(_NUM_KV, 4) is None


@pytest.mark.parametrize("tp", [4, 8], ids=["tp4", "tp8"])
def test_every_device_owns_the_kv_head_its_q_heads_need(tp):
    owners = kv_head_owners(_NUM_KV, tp) or list(range(tp))
    assert len(owners) == tp
    gqa_group = _NUM_Q // _NUM_KV
    per_device_q = _NUM_Q // tp
    for device, owned in enumerate(owners):
        needed = {q // gqa_group for q in range(device * per_device_q, (device + 1) * per_device_q)}
        # One KV head per device, and a device's Q heads never span two of them.
        assert needed == {owned}


def test_kv_head_owners_rejects_uneven_sharing(expect_error):
    with expect_error(ValueError, "cannot share"):
        kv_head_owners(3, 8)


def test_t3k_policy_fits_a_narrower_grid_and_one_link():
    # _width_memory builds a 10-wide rectangle for core counts divisible by ten, which cannot
    # exist on an 8x8 worker grid, and the second ethernet link per pair is the dispatch datapath.
    overlay = _TP_POLICY[8]
    assert overlay["rectangular_working"] is False
    assert overlay["num_links"] == 1


def test_the_measured_tp4_defaults_are_not_overlaid():
    # The overlay carries T3K deltas only; the TP4 measurement stays in measured_policy.
    assert 4 not in _TP_POLICY


@pytest.mark.parametrize("platform", [_T3K], ids=["t3k"])
def test_entry_points_can_size_the_mesh_before_opening_it(monkeypatch, platform):
    arch, cluster, _, shape = platform
    monkeypatch.setattr(ttnn, "get_arch_name", lambda: arch.name.lower())
    monkeypatch.setattr(ttnn.cluster, "get_cluster_type", lambda: cluster)
    assert native_mesh_shape() == shape


def test_unqualified_cluster_has_no_native_mesh_shape(monkeypatch, expect_error):
    monkeypatch.setattr(ttnn, "get_arch_name", lambda: "wormhole_b0")
    monkeypatch.setattr(ttnn.cluster, "get_cluster_type", lambda: ttnn.cluster.ClusterType.N300)
    with expect_error(ValueError, re.escape("requires a Wormhole T3K")):
        native_mesh_shape()
