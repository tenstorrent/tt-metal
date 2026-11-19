# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""CPU tests for the default-off LTX_FUSE_NORM_ADALN opt-in."""

import pytest
import torch

import models.tt_dit.models.transformers.ltx.transformer_ltx as transformer_ltx

D = 64


@pytest.mark.parametrize(
    "env,expected",
    [(None, False), ("0", False), ("", False), ("1", True), ("true", True), ("True", True)],
)
def test_fuse_norm_adaln_flag(monkeypatch, env, expected):
    if env is None:
        monkeypatch.delenv("LTX_FUSE_NORM_ADALN", raising=False)
    else:
        monkeypatch.setenv("LTX_FUSE_NORM_ADALN", env)
    assert transformer_ltx._fuse_norm_adaln_enabled() is expected


class _FakeNorm:
    def __init__(self):
        self.calls = []

    def __call__(self, x, **kwargs):
        self.calls.append(kwargs)
        return ("normed", kwargs)


@pytest.mark.parametrize(
    "fuse,x_shape,mod_shape,fused",
    [
        (False, (1, 1, 32, D), (1, 1, 1, D), False),  # default: unfused norm then addcmul
        (True, (1, 1, 32, D), (1, 1, 1, D), True),  # scalar B=1 modulation: fused
        (True, (1, 1, 32, D), (1, 1, 32, D), False),  # per-token modulation stays unfused
        (True, (2, 1, 32, D), (1, 1, 1, D), False),  # batched input stays unfused
    ],
)
def test_norm_adaln_route(monkeypatch, fuse, x_shape, mod_shape, fused):
    addcmul_calls = []
    monkeypatch.setattr(transformer_ltx.ttnn, "addcmul", lambda *args: addcmul_calls.append(args) or "addcmul")
    norm = _FakeNorm()
    x, shift, scale_p1 = torch.zeros(x_shape), torch.zeros(mod_shape), torch.ones(mod_shape)

    out = transformer_ltx._norm_adaln(norm, x, shift, scale_p1, fuse=fuse)

    if fused:
        assert norm.calls == [{"dynamic_weight": scale_p1, "dynamic_bias": shift}]
        assert addcmul_calls == []
        assert out == ("normed", norm.calls[0])
    else:
        assert norm.calls == [{}]
        assert addcmul_calls == [(shift, ("normed", {}), scale_p1)]
        assert out == "addcmul"
