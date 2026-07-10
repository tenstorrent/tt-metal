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


def _model(
    *,
    hidden_size=2816,
    intermediate_size=192,
    moe_intermediate_size=704,
    num_experts=128,
    top_k=8,
    dtype="bf16",
    arch="blackhole",
    mesh_shape=(1, 4),
    num_devices=4,
    grid=(13, 10),
    tp=4,
    ep=1,
    sp=1,
    tp_axis=1,
    mismatched_second_layer=False,
):
    def make_experts(layer_hidden_size):
        weight = SimpleNamespace(get_dtype=lambda: dtype)
        return SimpleNamespace(
            config=SimpleNamespace(
                hidden_size=layer_hidden_size,
                moe_intermediate_size=moe_intermediate_size,
                num_experts=num_experts,
                top_k=top_k,
            ),
            weights=SimpleNamespace(
                intermediate_size_per_device=intermediate_size,
                gate_proj=weight,
                up_proj=weight,
                down_proj=weight,
            ),
        )

    mesh = SimpleNamespace(
        arch=lambda: arch,
        shape=mesh_shape,
        get_num_devices=lambda: num_devices,
        compute_with_storage_grid_size=lambda: SimpleNamespace(x=grid[0], y=grid[1]),
    )
    mesh_config = SimpleNamespace(
        mesh_shape=mesh_shape,
        tp_axis=tp_axis,
        prefill=SimpleNamespace(tp=tp, ep=ep, sp=sp),
    )
    layer_hidden_sizes = [hidden_size, 2048 if mismatched_second_layer else hidden_size]
    return SimpleNamespace(
        mesh_device=mesh,
        mesh_config=mesh_config,
        layers=[
            SimpleNamespace(moe=SimpleNamespace(experts=make_experts(layer_hidden_size)))
            for layer_hidden_size in layer_hidden_sizes
        ],
    )


@pytest.fixture
def fake_ttnn(monkeypatch):
    fake = SimpleNamespace(
        TILE_SIZE=32,
        bfloat16="bf16",
        bfloat8_b="bfp8",
        device=SimpleNamespace(Arch=SimpleNamespace(BLACKHOLE="blackhole")),
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
        _model(moe_intermediate_size=768),
        _model(num_experts=64),
        _model(top_k=4),
        _model(dtype="bfp8"),
        _model(arch="wormhole"),
        _model(mesh_shape=(2, 2)),
        _model(num_devices=8),
        _model(tp=2),
        _model(ep=2),
        _model(sp=2),
        _model(tp_axis=0),
        _model(grid=(8, 8)),
        _model(mismatched_second_layer=True),
        SimpleNamespace(layers=[]),
    ],
)
def test_tuned_prefill_moe_leaves_unsupported_models_unchanged(monkeypatch, contextual_builder, model):
    monkeypatch.setenv(PM.FLAG, "1")

    with PM.use_tuned_prefill_moe(model):
        assert contextual_builder(32, 192) == ("original", 32, 192, 1)


def test_tuned_prefill_moe_requires_measured_chunk_size(monkeypatch, contextual_builder):
    monkeypatch.setenv(PM.FLAG, "1")
    monkeypatch.setattr(PM.gemma4_prefill, "PREFILL_CHUNK_SIZE", 64)

    with PM.use_tuned_prefill_moe(_model()):
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
