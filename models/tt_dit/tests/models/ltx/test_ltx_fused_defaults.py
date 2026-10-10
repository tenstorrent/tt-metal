# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""CPU check that the fused gate and fused norm+AdaLN routes are on unless turned off."""

import pytest

from models.tt_dit.models.transformers.ltx import attention_ltx, transformer_ltx


@pytest.mark.parametrize(
    "var, enabled",
    [
        ("LTX_FUSE_GATE_ON_DEVICE", attention_ltx._fuse_gate_on_device_enabled),
        ("LTX_FUSE_NORM_ADALN", transformer_ltx._fuse_norm_adaln_enabled),
    ],
)
def test_fused_route_default_on(monkeypatch, var, enabled):
    monkeypatch.delenv(var, raising=False)
    assert enabled()
    for off in ("0", "false", "False"):
        monkeypatch.setenv(var, off)
        assert not enabled()
    monkeypatch.setenv(var, "1")
    assert enabled()
