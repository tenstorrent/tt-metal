# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
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


@pytest.fixture
def contextual_builder(monkeypatch, fake_ttnn):
    def original(m, n, in0_block_w=1):
        return ("original", m, n, in0_block_w)

    monkeypatch.setattr(PM, "_original_builder", original)
    monkeypatch.setattr(PM.gemma4_prefill, "_build_sparse_matmul_config", PM._contextual_config_builder)
    return PM._contextual_config_builder


def test_tuned_prefill_moe_uses_measured_qb2_geometry(monkeypatch, contextual_builder):
    monkeypatch.setenv(PM.FLAG, "1")

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
    assert PM.gemma4_prefill._build_sparse_matmul_config is contextual_builder
    assert contextual_builder(32, 192) == ("original", 32, 192, 1)


@pytest.mark.parametrize(
    "model",
    [
        _model(hidden_size=2048),
        _model(intermediate_size=256),
        _model(grid=(8, 8)),
        SimpleNamespace(layers=[]),
    ],
)
def test_tuned_prefill_moe_leaves_unsupported_models_unchanged(monkeypatch, contextual_builder, model):
    monkeypatch.setenv(PM.FLAG, "1")

    with PM.use_tuned_prefill_moe(model):
        assert contextual_builder(32, 192) == ("original", 32, 192, 1)


def test_tuned_prefill_moe_resets_context_after_error(monkeypatch, contextual_builder, expect_error):
    monkeypatch.setenv(PM.FLAG, "1")

    with expect_error(RuntimeError, match="stop"):
        with PM.use_tuned_prefill_moe(_model()):
            raise RuntimeError("stop")
    assert contextual_builder(32, 192) == ("original", 32, 192, 1)


def test_tuned_prefill_moe_does_not_leak_across_threads(monkeypatch, contextual_builder):
    monkeypatch.setenv(PM.FLAG, "1")
    entered = Barrier(2)
    completed = Barrier(2)

    def tuned_call():
        with PM.use_tuned_prefill_moe(_model()):
            entered.wait()
            result = contextual_builder(32, 192)
            completed.wait()
            return result

    def stock_call():
        entered.wait()
        result = contextual_builder(32, 192)
        completed.wait()
        return result

    with ThreadPoolExecutor(max_workers=2) as executor:
        tuned = executor.submit(tuned_call)
        stock = executor.submit(stock_call)

    assert tuned.result()["compute_with_storage_grid_size"] == (6, 1)
    assert stock.result() == ("original", 32, 192, 1)
