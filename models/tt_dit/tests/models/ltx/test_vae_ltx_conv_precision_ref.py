# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU check of the LTX_VAE_CONV_FIDELITY knob: unset changes nothing, bad values fail loudly.
Run: python -m pytest --noconftest <this file>
"""

import pytest

import ttnn
from models.tt_dit.models.vae import vae_ltx


@pytest.fixture
def env(monkeypatch):
    monkeypatch.delenv("LTX_VAE_CONV_FIDELITY", raising=False)
    return monkeypatch


def test_unset_is_default(env):
    assert vae_ltx._decoder_conv_fidelity_from_env() is None


@pytest.mark.parametrize("name", ["LoFi", "HiFi2", "HiFi3", "HiFi4"])
def test_parse(env, name):
    env.setenv("LTX_VAE_CONV_FIDELITY", name)
    assert vae_ltx._decoder_conv_fidelity_from_env() == getattr(ttnn.MathFidelity, name)


@pytest.mark.parametrize("value", ["lofi", "HiFi1"])
def test_bad_value_raises(env, value):
    env.setenv("LTX_VAE_CONV_FIDELITY", value)
    with pytest.raises(ValueError):  # allow-pytest.raises: runs with --noconftest, no expect_error fixture
        vae_ltx._decoder_conv_fidelity_from_env()
