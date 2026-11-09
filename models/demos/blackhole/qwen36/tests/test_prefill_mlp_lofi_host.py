# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""QWEN36_PREFILL_MLP_LOFI host checks (no device): LoFi only for prefill calls, only when every dense-MLP weight is
bfloat4_b, and the twin keeps the packer_l1_acc of the config today's code picks."""
from types import SimpleNamespace

import pytest

import ttnn
from models.demos.blackhole.qwen36.tt import mlp as mlp_mod
from models.demos.blackhole.qwen36.tt import tp_common as tpc

BF4, BF8 = ttnn.bfloat4_b, ttnn.bfloat8_b


def test_flag_off_by_default(monkeypatch):
    monkeypatch.delenv(tpc.PREFILL_MLP_LOFI_ENV, raising=False)
    assert tpc.prefill_mlp_lofi([BF4, BF4]) is False


def test_on_all_bf4(monkeypatch):
    monkeypatch.setenv(tpc.PREFILL_MLP_LOFI_ENV, "1")
    assert tpc.prefill_mlp_lofi([BF4, BF4, BF4]) is True


@pytest.mark.parametrize("dts", [[BF4, BF8], [BF8], []])
def test_on_refuses_non_bf4(monkeypatch, dts):
    monkeypatch.setenv(tpc.PREFILL_MLP_LOFI_ENV, "1")
    assert tpc.prefill_mlp_lofi(dts) is False


def test_bad_value(monkeypatch):
    monkeypatch.setenv(tpc.PREFILL_MLP_LOFI_ENV, "yes")
    with pytest.raises(ValueError):
        tpc.prefill_mlp_lofi([BF4])


def _x(rows):
    return SimpleNamespace(shape=(1, 1, rows, 640))


def test_prefill_ckc_swaps_only_prefill_rows(monkeypatch):
    monkeypatch.setattr(mlp_mod.ttnn, "TILE_SIZE", 32, raising=False)
    base, dec, lofi_a, lofi_b = object(), object(), object(), object()
    s = SimpleNamespace(_prefill_lofi={id(base): lofi_a, id(dec): lofi_b})
    f = mlp_mod.Qwen36MLP._prefill_ckc
    assert f(s, base, _x(8192)) is lofi_a
    assert f(s, dec, _x(128)) is lofi_b  # today's T-based pick (decode cfg) keeps its packer_l1_acc twin
    assert f(s, base, _x(32)) is base  # decode rows untouched
    assert f(s, object(), _x(8192)) is not lofi_a  # unknown config passes through


def test_prefill_ckc_off_is_identity(monkeypatch):
    monkeypatch.setattr(mlp_mod.ttnn, "TILE_SIZE", 32, raising=False)
    s = SimpleNamespace(_prefill_lofi=None)
    c = object()
    assert mlp_mod.Qwen36MLP._prefill_ckc(s, c, _x(8192)) is c
