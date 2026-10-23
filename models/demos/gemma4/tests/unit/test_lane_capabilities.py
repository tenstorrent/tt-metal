# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The lane-serving declarations the vllm-tt-plugin reads from ``model_capabilities``
(``folds_dp_into_lanes``, ``lane_prefill_group``) follow the Gemma 4 env rails. Pure Python."""

import pytest

# generator_vllm imports vLLM at module scope; the tt-metal unit-test runners have no vLLM.
pytest.importorskip("vllm")
from models.demos.gemma4.tt import generator_vllm as gv  # noqa: E402

CLASSES = [
    gv.Gemma4ForCausalLM,
    gv.Gemma4VisionForConditionalGeneration,
    gv.Gemma4MTPForCausalLM,
    gv.Gemma4DFlashForCausalLM,
]


def _declared(cls, key):
    value = cls.model_capabilities[key]
    return value() if callable(value) else value


@pytest.mark.parametrize("cls", CLASSES, ids=lambda c: c.__name__)
def test_lane_declarations_follow_the_env_rails(cls, monkeypatch):
    monkeypatch.delenv("GEMMA4_GALAXY_LANES", raising=False)
    monkeypatch.delenv("GEMMA4_LANE_PREFILL_ROUTER", raising=False)
    assert _declared(cls, "folds_dp_into_lanes") is False
    assert _declared(cls, "lane_prefill_group") is False
    assert not cls._lanes_fold() and not cls._lane_prefill_group()

    monkeypatch.setenv("GEMMA4_LANE_PREFILL_ROUTER", "1")
    assert _declared(cls, "lane_prefill_group") is False  # needs the fold

    monkeypatch.setenv("GEMMA4_GALAXY_LANES", "1")
    assert _declared(cls, "folds_dp_into_lanes") is True
    assert _declared(cls, "lane_prefill_group") is True
    assert cls._lanes_fold() and cls._lane_prefill_group()


def test_lane_declarations_are_resolved_at_config_time():
    # Callables, not import-time values: the plugin evaluates them in
    # check_and_update_config, once the catalog's env is in place.
    caps = gv.Gemma4ForCausalLM.model_capabilities
    assert callable(caps["folds_dp_into_lanes"]) and callable(caps["lane_prefill_group"])
