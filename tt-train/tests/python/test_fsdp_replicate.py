# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for FSDP replicate patterns and their config; no device needed."""

from __future__ import annotations

import re

import pytest

from ttml.common.config import DeviceConfig
from ttml.fsdp import _matching_replicate_patterns


@pytest.mark.parametrize(
    "name, patterns, expected",
    [
        # A pattern can match anywhere in the name.
        ("self_attn.q_norm.weight", ["q_norm.weight"], ["q_norm.weight"]),
        ("self_attn.q_norm.weight", [r"(q|k)_norm\.weight$"], [r"(q|k)_norm\.weight$"]),
        ("self_attn.q_norm.weight", [r"^self_attn\."], [r"^self_attn\."]),
        # Anchors apply to the whole name.
        ("self_attn.q_norm.weight", [r"^q_norm"], []),
        ("mlp.up_proj.weight", ["q_norm.weight", "k_norm.weight"], []),
        # All matching patterns are returned.
        ("self_attn.k_norm.weight", ["k_norm", "norm"], ["k_norm", "norm"]),
    ],
)
def test_matching_replicate_patterns(name, patterns, expected):
    assert _matching_replicate_patterns(name, [re.compile(p) for p in patterns]) == expected


def test_device_config_fsdp_replicate_params_defaults_to_empty():
    assert DeviceConfig({"device_config": {"enable_fsdp": True}}).fsdp_replicate_params == []


def test_device_config_fsdp_replicate_params_parsed():
    cfg = DeviceConfig({"device_config": {"fsdp_replicate_params": ["q_norm.weight", "k_norm.weight"]}})
    assert cfg.fsdp_replicate_params == ["q_norm.weight", "k_norm.weight"]


@pytest.mark.parametrize("bad", ["q_norm.weight", ["q_norm.weight", 3]])
def test_device_config_fsdp_replicate_params_rejects_non_list_of_str(bad, expect_error):
    with expect_error(ValueError, "fsdp_replicate_params must be a list of regex patterns"):
        DeviceConfig({"device_config": {"fsdp_replicate_params": bad}})
