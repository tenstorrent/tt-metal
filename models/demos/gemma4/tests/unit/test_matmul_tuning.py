# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest

from models.demos.gemma4.tt import matmul_tuning


def _patch_config_types(monkeypatch):
    monkeypatch.setattr(matmul_tuning.ttnn, "CoreCoord", lambda x, y: SimpleNamespace(x=x, y=y))
    monkeypatch.setattr(
        matmul_tuning.ttnn,
        "MatmulMultiCoreReuseMultiCast1DProgramConfig",
        lambda **kwargs: SimpleNamespace(**kwargs),
    )


@pytest.mark.parametrize("logical_m", [1, 16, 32])
def test_decode_config_accepts_one_logical_tile_row(monkeypatch, logical_m):
    _patch_config_types(monkeypatch)
    config = matmul_tuning.derive_decode_1d_config(logical_m, 256, 1024)
    assert config is not None
    assert config.per_core_M == 1
    assert 1 <= config.in0_block_w <= 8
    assert (256 // matmul_tuning.ttnn.TILE_SIZE) % config.in0_block_w == 0


@pytest.mark.parametrize("shape", [(33, 256, 1024), (1, 255, 1024), (1, 256, 1000), (1, 32, 32)])
def test_decode_config_rejects_incompatible_shapes(monkeypatch, shape):
    _patch_config_types(monkeypatch)
    assert matmul_tuning.derive_decode_1d_config(*shape) is None


@pytest.mark.parametrize(
    "value,draft,target",
    [
        (None, False, True),
        ("", False, True),
        ("0", False, False),
        ("off", False, False),
        ("draft", True, False),
        ("target", False, True),
        ("draft,target", True, True),
        ("all", True, True),
    ],
)
def test_tuner_scope_selection(monkeypatch, value, draft, target):
    if value is None:
        monkeypatch.delenv("GEMMA4_TUNE_MATMULS", raising=False)
    else:
        monkeypatch.setenv("GEMMA4_TUNE_MATMULS", value)
    assert matmul_tuning.DecodeMatmulTuner.from_env(scope="draft").enabled is draft
    assert matmul_tuning.DecodeMatmulTuner.from_env(scope="target").enabled is target


class _FakeMesh:
    def __init__(self, num_devices):
        self._num_devices = num_devices

    def get_num_devices(self):
        return self._num_devices

    def compute_with_storage_grid_size(self):
        return SimpleNamespace(x=8, y=8)


@pytest.mark.parametrize(
    "value,num_devices,target",
    [
        (None, 1, True),
        (None, 2, False),
        ("target", 2, True),
        ("0", 1, False),
    ],
)
def test_target_default_is_single_device_only(monkeypatch, value, num_devices, target):
    if value is None:
        monkeypatch.delenv("GEMMA4_TUNE_MATMULS", raising=False)
    else:
        monkeypatch.setenv("GEMMA4_TUNE_MATMULS", value)
    mesh = _FakeMesh(num_devices)
    assert matmul_tuning.DecodeMatmulTuner.from_env(mesh, scope="target").enabled is target


def test_tuner_caches_by_shape(monkeypatch):
    calls = []
    marker = object()

    def derive(*args):
        calls.append(args)
        return marker

    monkeypatch.setattr(matmul_tuning, "derive_decode_1d_config", derive)
    tuner = matmul_tuning.DecodeMatmulTuner(enabled=True)
    x = SimpleNamespace(shape=(1, 1, 1, 256))
    weight = SimpleNamespace(shape=(1, 1, 256, 1024))
    assert tuner.config_for(x, weight) is marker
    assert tuner.config_for(x, weight) is marker
    assert len(calls) == 1


def test_explicit_program_config_takes_precedence(monkeypatch):
    captured = {}

    def linear(x, weight, **kwargs):
        captured.update(kwargs)
        return "output"

    monkeypatch.setattr(matmul_tuning.ttnn, "linear", linear)
    tuner = matmul_tuning.DecodeMatmulTuner(enabled=True)
    tuner.config_for = lambda *args: pytest.fail("explicit config must bypass tuner derivation")
    explicit = object()
    assert tuner.linear(object(), object(), program_config=explicit) == "output"
    assert captured["program_config"] is explicit


def _patch_compute_config(monkeypatch):
    monkeypatch.setattr(
        matmul_tuning.ttnn,
        "init_device_compute_kernel_config",
        lambda arch, **kwargs: SimpleNamespace(arch=arch, **kwargs),
    )


def _tensor(dtype, shape=(1, 1, 1, 256)):
    return SimpleNamespace(dtype=dtype, shape=shape, device=lambda: SimpleNamespace(arch=lambda: "arch"))


def test_tuned_call_keeps_automatic_compute_config(monkeypatch):
    _patch_compute_config(monkeypatch)
    captured = {}

    def linear(x, weight, **kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(matmul_tuning.ttnn, "linear", linear)
    tuner = matmul_tuning.DecodeMatmulTuner(enabled=True)
    tuner.config_for = lambda *args: "tuned"
    tuner.linear(_tensor(matmul_tuning.ttnn.bfloat16), _tensor(matmul_tuning.ttnn.bfloat8_b))
    config = captured["compute_kernel_config"]
    assert captured["program_config"] == "tuned"
    assert config.math_fidelity == matmul_tuning.ttnn.MathFidelity.HiFi2
    assert config.packer_l1_acc and not config.fp32_dest_acc_en and not config.math_approx_mode


def test_caller_compute_config_and_untuned_calls_are_unchanged(monkeypatch):
    _patch_compute_config(monkeypatch)
    captured = []
    monkeypatch.setattr(matmul_tuning.ttnn, "linear", lambda x, weight, **kwargs: captured.append(kwargs))
    tuner = matmul_tuning.DecodeMatmulTuner(enabled=True)
    x, weight = _tensor(matmul_tuning.ttnn.bfloat16), _tensor(matmul_tuning.ttnn.bfloat16)
    explicit = object()
    tuner.config_for = lambda *args: "tuned"
    tuner.linear(x, weight, compute_kernel_config=explicit)
    tuner.config_for = lambda *args: None
    tuner.linear(x, weight)
    assert captured[0]["compute_kernel_config"] is explicit
    assert "compute_kernel_config" not in captured[1]


@pytest.mark.parametrize(
    "x_dtype,weight_dtype",
    [("bfloat8_b", "bfloat8_b"), ("bfloat4_b", "bfloat8_b"), ("float32", "float32")],
)
def test_compute_config_defers_where_ttnn_default_matches(monkeypatch, x_dtype, weight_dtype):
    _patch_compute_config(monkeypatch)
    tuner = matmul_tuning.DecodeMatmulTuner(enabled=True)
    ttnn = matmul_tuning.ttnn
    assert tuner.compute_config_for(_tensor(getattr(ttnn, x_dtype)), _tensor(getattr(ttnn, weight_dtype))) is None
