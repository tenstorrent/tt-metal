# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import ttnn

from ....pipelines.wan import pipeline_wan, policy


@pytest.fixture
def arch(monkeypatch):
    def set_arch(blackhole: bool) -> None:
        monkeypatch.setattr(policy, "_is_blackhole", lambda: blackhole)
        monkeypatch.setattr(ttnn.device, "is_blackhole", lambda: blackhole)
        monkeypatch.setattr(pipeline_wan, "ensure_cached", lambda name: name)

    return set_arch


@pytest.mark.parametrize(
    "blackhole, presets",
    [(True, pipeline_wan._PRESETS_BH), (False, pipeline_wan._PRESETS_WH)],
    ids=["blackhole", "wormhole"],
)
def test_config_default_takes_the_served_resolutions(arch, blackhole, presets):
    arch(blackhole)
    for mesh_shape in presets:
        config = pipeline_wan.WanPipelineConfig.default(mesh_shape=mesh_shape)
        assert config.resolutions == policy.served_resolutions(mesh_shape), mesh_shape


@pytest.mark.parametrize(
    "blackhole, mesh_shape, expected",
    [
        (True, (2, 2), policy.WAN_RESOLUTIONS_480P),
        (True, (4, 8), policy.WAN_RESOLUTIONS_720P + policy.WAN_RESOLUTIONS_480P),
        (True, (4, 32), policy.WAN_RESOLUTIONS_720P + policy.WAN_RESOLUTIONS_480P),
        (True, (1, 4), (policy.WAN_DEFAULT_RESOLUTION,)),
        (False, (4, 8), (policy.WAN_DEFAULT_RESOLUTION,)),
    ],
)
def test_served_resolutions(arch, blackhole, mesh_shape, expected):
    arch(blackhole)
    assert policy.served_resolutions(mesh_shape) == expected


def test_validate_request_accepts_a_served_size(arch):
    arch(True)
    policy.validate_request(mesh_shape=(4, 8), width=1280, height=720)
    policy.validate_request(mesh_shape=(2, 2), width=480, height=832)


@pytest.mark.parametrize("blackhole, mesh_shape", [(True, (2, 2)), (False, (4, 8))])
def test_validate_request_rejects_an_unserved_size(arch, blackhole, mesh_shape, expect_error):
    arch(blackhole)
    with expect_error(ValueError, "unsupported size 1280x720; this deployment serves"):
        policy.validate_request(mesh_shape=mesh_shape, width=1280, height=720)
