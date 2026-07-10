# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest

from models.experimental.diffusion_gemma.tt import prefill_moe as PM


def test_tuned_prefill_moe_defaults_on_and_can_be_disabled(monkeypatch):
    monkeypatch.delenv(PM.FLAG, raising=False)
    assert PM.tuned_prefill_moe_enabled()
    monkeypatch.setenv(PM.FLAG, "0")
    assert not PM.tuned_prefill_moe_enabled()


def _model(*, hidden_size=2816, intermediate_size=192, grid=(13, 10)):
    experts = SimpleNamespace(
        config=SimpleNamespace(hidden_size=hidden_size),
        weights=SimpleNamespace(intermediate_size_per_device=intermediate_size),
    )
    mesh = SimpleNamespace(
        compute_with_storage_grid_size=lambda: SimpleNamespace(x=grid[0], y=grid[1]),
    )
    return SimpleNamespace(
        mesh_device=mesh,
        layers=[SimpleNamespace(moe=SimpleNamespace(experts=experts))],
    )


@pytest.fixture
def fake_ttnn(monkeypatch):
    fake = SimpleNamespace(
        TILE_SIZE=32,
        CoreCoord=lambda x, y: (x, y),
        MatmulMultiCoreReuseMultiCast1DProgramConfig=lambda **kwargs: kwargs,
    )
    monkeypatch.setattr(PM, "ttnn", fake)
    return fake


def test_tuned_prefill_moe_uses_measured_qb2_geometry(monkeypatch, fake_ttnn):
    monkeypatch.setenv(PM.FLAG, "1")

    def original(m, n, in0_block_w=1):
        return ("original", m, n, in0_block_w)

    monkeypatch.setattr(PM.gemma4_prefill, "_build_sparse_matmul_config", original)

    with PM.use_tuned_prefill_moe(_model()):
        builder = PM.gemma4_prefill._build_sparse_matmul_config
        gate = builder(32, 192)
        down = builder(32, 2816)
        fallback = builder(64, 192, 7)

    assert gate["compute_with_storage_grid_size"] == (6, 1)
    assert gate["in0_block_w"] == 44
    assert gate["per_core_N"] == 1
    assert down["compute_with_storage_grid_size"] == (11, 4)
    assert down["in0_block_w"] == 3
    assert down["per_core_N"] == 2
    assert fallback == ("original", 64, 192, 7)
    assert PM.gemma4_prefill._build_sparse_matmul_config is original


@pytest.mark.parametrize(
    "model",
    [
        _model(hidden_size=2048),
        _model(intermediate_size=256),
        _model(grid=(8, 8)),
        SimpleNamespace(layers=[]),
    ],
)
def test_tuned_prefill_moe_leaves_unsupported_models_unchanged(monkeypatch, model):
    monkeypatch.setenv(PM.FLAG, "1")

    def original(*args):
        return args

    monkeypatch.setattr(PM.gemma4_prefill, "_build_sparse_matmul_config", original)
    with PM.use_tuned_prefill_moe(model):
        assert PM.gemma4_prefill._build_sparse_matmul_config is original


def test_tuned_prefill_moe_restores_builder_after_error(monkeypatch, fake_ttnn, expect_error):
    monkeypatch.setenv(PM.FLAG, "1")

    def original(*args):
        return args

    monkeypatch.setattr(PM.gemma4_prefill, "_build_sparse_matmul_config", original)
    with expect_error(RuntimeError, match="stop"):
        with PM.use_tuned_prefill_moe(_model()):
            raise RuntimeError("stop")
    assert PM.gemma4_prefill._build_sparse_matmul_config is original
