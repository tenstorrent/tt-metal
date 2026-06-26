# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch

from models.experimental.diffusion_gemma.tt.self_conditioning import (
    build_self_conditioning,
    validate_self_conditioning_state,
)


def _state(hidden_size=8, intermediate_size=6):
    return {
        "pre_norm.weight": torch.ones(hidden_size),
        "gate_proj.weight": torch.ones(intermediate_size, hidden_size),
        "up_proj.weight": torch.ones(intermediate_size, hidden_size),
        "down_proj.weight": torch.ones(hidden_size, intermediate_size),
    }


def test_validate_self_conditioning_state_accepts_expected_shapes():
    validate_self_conditioning_state(_state(), hidden_size=8, intermediate_size=6)


def test_validate_self_conditioning_state_rejects_missing_weight():
    state = _state()
    del state["up_proj.weight"]

    with pytest.raises(ValueError, match="missing self-conditioning weights"):
        validate_self_conditioning_state(state, hidden_size=8, intermediate_size=6)


def test_validate_self_conditioning_state_rejects_wrong_shape():
    state = _state()
    state["down_proj.weight"] = torch.ones(6, 8)

    with pytest.raises(ValueError, match="down_proj.weight has shape"):
        validate_self_conditioning_state(state, hidden_size=8, intermediate_size=6)


def test_build_self_conditioning_uses_config_and_forwards_constructor_args():
    calls = {}

    class _FakeSelfConditioning:
        def __init__(self, device, state_dict, **kwargs):
            calls["ctor"] = (device, state_dict, kwargs)

    config = SimpleNamespace(hidden_size=8, intermediate_size=6, rms_norm_eps=1e-5)

    out = build_self_conditioning(
        "device",
        _state(),
        config=config,
        dtype="dtype",
        module_cls=_FakeSelfConditioning,
    )

    assert isinstance(out, _FakeSelfConditioning)
    assert calls["ctor"][0] == "device"
    expected_state = _state()
    assert calls["ctor"][1].keys() == expected_state.keys()
    for key, expected in expected_state.items():
        assert torch.equal(calls["ctor"][1][key], expected)
    assert calls["ctor"][2] == {
        "hidden_size": 8,
        "intermediate_size": 6,
        "eps": 1e-5,
        "dtype": "dtype",
    }


def test_build_self_conditioning_requires_dimensions_without_config():
    with pytest.raises(ValueError, match="hidden_size and intermediate_size"):
        build_self_conditioning("device", _state(), module_cls=object)
