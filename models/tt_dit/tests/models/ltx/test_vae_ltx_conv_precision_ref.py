# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU check of the LTX_VAE_CONV_FIDELITY / LTX_VAE_CONV_WEIGHT_DTYPE knobs.

Unset knobs change nothing, bad values fail loudly, and a bf8 weight changes the conv3d weight-cache key
(so a cached bf16 tensor is never loaded into a bf8 parameter) while default keys stay byte-identical.
Run: python -m pytest --noconftest <this file>
"""

import pytest

import ttnn
from models.tt_dit.models.vae import vae_ltx
from models.tt_dit.utils.conv3d import conv3d_blocking_hash


class _Conv:
    def __init__(self, c_in_block, weight_dtype=ttnn.bfloat16):
        self.conv_config = type("Cfg", (), {"C_in_block": c_in_block})()
        self.dtype = ttnn.bfloat16
        self.weight = type("W", (), {"dtype": weight_dtype})()

    def named_children(self):
        return iter(())


class _Tree:
    def __init__(self, *children):
        self._children = children

    def named_children(self):
        return ((str(i), c) for i, c in enumerate(self._children))


@pytest.fixture
def env(monkeypatch):
    monkeypatch.delenv("LTX_VAE_CONV_FIDELITY", raising=False)
    monkeypatch.delenv("LTX_VAE_CONV_WEIGHT_DTYPE", raising=False)
    return monkeypatch


def test_unset_is_default(env):
    assert vae_ltx._decoder_conv_precision_from_env() == (None, None)


@pytest.mark.parametrize(
    "fidelity,dtype,want",
    [
        ("LoFi", "", (None, ttnn.MathFidelity.LoFi)),
        ("HiFi2", "bf16", (ttnn.bfloat16, ttnn.MathFidelity.HiFi2)),
        ("", "bf8", (ttnn.bfloat8_b, None)),
        ("LoFi", "bf8", (ttnn.bfloat8_b, ttnn.MathFidelity.LoFi)),
    ],
)
def test_parse(env, fidelity, dtype, want):
    env.setenv("LTX_VAE_CONV_FIDELITY", fidelity)
    env.setenv("LTX_VAE_CONV_WEIGHT_DTYPE", dtype)
    assert vae_ltx._decoder_conv_precision_from_env() == want


@pytest.mark.parametrize("var,value", [("LTX_VAE_CONV_FIDELITY", "lofi"), ("LTX_VAE_CONV_WEIGHT_DTYPE", "bfp8")])
def test_bad_value_raises(env, var, value):
    env.setenv(var, value)
    with pytest.raises(ValueError):  # allow-pytest.raises: runs with --noconftest, no expect_error fixture
        vae_ltx._decoder_conv_precision_from_env()


def test_blocking_hash_keys_on_weight_dtype():
    default = conv3d_blocking_hash(_Tree(_Conv(64), _Conv(128)))
    # Same key as before the dtype was part of it: sha256 of "64_128".
    assert default == "cin" + __import__("hashlib").sha256(b"64_128").hexdigest()[:8]
    bf8 = conv3d_blocking_hash(_Tree(_Conv(64), _Conv(128, ttnn.bfloat8_b)))
    assert bf8 != default
