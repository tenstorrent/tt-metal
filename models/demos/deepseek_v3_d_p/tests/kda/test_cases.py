# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU-only contracts of the KDA case matrix and its cache keys (no device)."""

from __future__ import annotations

import hashlib
import itertools
from dataclasses import replace
from pathlib import Path

import pytest
import torch

import ttnn
from models.demos.deepseek_v3_d_p.reference.kda import KDAReferenceState, kda_forward_reference
from models.demos.deepseek_v3_d_p.reference.kda.config import KDAConfig
from models.demos.deepseek_v3_d_p.tests.kda import cases as kda_cases
from models.demos.deepseek_v3_d_p.tests.kda.cases import (
    KDA_CASES,
    KDAPreparedCacheMiss,
    KDATestCase,
    KDAWeightSource,
    build_kda_case,
    kda_weight_cache_dir,
    loudbox_kda_case,
    registered_kda_case,
)
from models.demos.deepseek_v3_d_p.tests.kda.decay_extremes import (
    DECAY_EXTREME_CASES,
    case_config,
    case_input_norm_weight,
    case_weights,
    check_reachable_input,
    crafted_hidden,
)
from models.demos.deepseek_v3_d_p.tests.kda.reference_cache import (
    cpu_reference_cache_path,
    cpu_references,
    prepare_cpu_references,
)
from models.demos.deepseek_v3_d_p.tests.kda.text_input import text_input_cache_path
from models.demos.deepseek_v3_d_p.utils.oracle_cache import SHARED_ORACLE_CACHE_ENV

_TOY_CONFIG = KDAConfig(hidden_size=64, num_heads=2, head_k_dim=32, head_v_dim=32, conv_kernel_size=4, norm_eps=1e-5)


@pytest.fixture
def oracle_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """An empty shared CPU oracle cache; the per-checkout model cache is a separate empty directory."""
    monkeypatch.setattr(ttnn.CONFIG, "model_cache_path", tmp_path / "model_cache")
    monkeypatch.setenv(SHARED_ORACLE_CACHE_ENV, str(tmp_path / "oracles"))
    return tmp_path / "oracles"


def _toy_source(**overrides) -> KDAWeightSource:
    return replace(KDAWeightSource(model="kimi_k3", layer_idx=1, kind="synthetic", config=_TOY_CONFIG), **overrides)


def _toy_hidden(tokens: int, seed: int = 3) -> torch.Tensor:
    return torch.randn(1, tokens, _TOY_CONFIG.hidden_size, generator=torch.Generator().manual_seed(seed)).bfloat16()


def _toy_state(scale: float) -> KDAReferenceState:
    history = _TOY_CONFIG.conv_kernel_size - 1
    return KDAReferenceState(
        recurrent=torch.full((1, 2, 32, 32), scale),
        q_convolution=torch.full((1, history, 64), scale),
        k_convolution=torch.full((1, history, 64), scale),
        v_convolution=torch.full((1, history, 64), scale),
    )


def test_registered_cases_are_unique_and_found_by_their_fields(expect_error) -> None:
    for name, spec in KDA_CASES.items():
        assert spec.name == name
        assert (
            registered_kda_case(
                spec.weights,
                spec.mesh_shape,
                spec.tensor_parallel_axis,
                spec.chunk_tokens,
                chunk_valid_tokens=spec.chunk_valid_tokens,
                head_slice=spec.head_slice,
                inputs=spec.inputs,
                model=spec.model,
            )
            is spec
        )
    with expect_error(KeyError, "not registered"):
        registered_kda_case("synthetic", (2, 4), 1, 2560)


def test_loudbox_matrix_is_registered() -> None:
    expected = {
        "LB-A": ((2, 4), False, {"single": (1280,), "chained3": (1280,) * 3, "ragged": (1280, 992)}),
        "LB-B": ((8, 1), True, {"single": (5120,), "chained3": (5120,) * 3, "ragged": (5120, 3872)}),
    }
    heads = {"kimi_k3": 96, "glm_5_3_flash": 64}
    for model, (weights, inputs) in itertools.product(
        heads, (("synthetic", "randn"), ("real", "randn"), ("real", "text"))
    ):
        for layout, (mesh_shape, quarter, schedules) in expected.items():
            for schedule, valid_tokens in schedules.items():
                spec = loudbox_kda_case(weights, layout, schedule, inputs, model)
                head_slice = (0, heads[model] // 4) if quarter else None
                assert (spec.model, spec.mesh_shape, spec.head_slice, spec.chunk_valid_tokens) == (
                    model,
                    mesh_shape,
                    head_slice,
                    valid_tokens,
                )
                assert spec.weight_source().config.num_heads == heads[model] // (4 if quarter else 1)


def test_case_spec_rejects_unpreparable_schedules(expect_error) -> None:
    spec = KDA_CASES["kimi_k3-synthetic-mesh2x4-tpaxis1-T1280"]
    with expect_error(ValueError, "only the last"):
        replace(spec, chunk_valid_tokens=(992, 1280))
    with expect_error(ValueError, "multiple of SP"):
        replace(spec, chunk_tokens=1296, chunk_valid_tokens=(1296,))
    with expect_error(ValueError, "head_slice"):
        replace(spec, head_slice=(90, 100)).weight_source()


def test_preparation_and_device_builders_produce_identical_keys() -> None:
    """Every registered random-input case built twice from scratch yields the same inputs and cache keys."""
    for spec in (spec for spec in KDA_CASES.values() if spec.inputs == "randn"):
        first, second = build_kda_case(spec), build_kda_case(spec)
        assert torch.equal(first.hidden, second.hidden), spec.name
        assert first.weights == second.weights, spec.name
        assert kda_weight_cache_dir(first.weights, spec.mesh_shape, spec.tensor_parallel_axis) == kda_weight_cache_dir(
            second.weights, spec.mesh_shape, spec.tensor_parallel_axis
        )
        assert cpu_reference_cache_path(first.weights, first.chunk_valid_hidden(0), None) == cpu_reference_cache_path(
            second.weights, second.chunk_valid_hidden(0), None
        ), spec.name


def test_weight_cache_dirs_separate_sources_and_placements() -> None:
    dirs = {}
    for spec in KDA_CASES.values():
        weights = spec.weight_source()
        key = (weights.identity, weights.config, spec.mesh_shape, spec.tensor_parallel_axis)
        dirs.setdefault(kda_weight_cache_dir(weights, spec.mesh_shape, spec.tensor_parallel_axis), set()).add(key)
    # The tensorbin stems add layer and config digest; within one directory only the config may differ.
    for path, keys in dirs.items():
        assert len({key[0] for key in keys}) == 1 and len({key[2:] for key in keys}) == 1, path
    synthetic_lb_b = KDA_CASES["kimi_k3-synthetic-mesh8x1-tpaxis1-T5120-heads0-24"].weight_source()
    assert synthetic_lb_b.config.num_heads == 24
    assert synthetic_lb_b.identity != KDA_CASES["kimi_k3-synthetic-mesh2x4-tpaxis1-T1280"].weight_source().identity


def test_reference_keys_distinguish_every_identity_field() -> None:
    hidden = _toy_hidden(64)
    baseline = (_toy_source(), hidden, None)
    variants = {
        "baseline": baseline,
        "model": (_toy_source(model="other_model"), hidden, None),
        "layer": (_toy_source(layer_idx=2), hidden, None),
        "head-slice": (_toy_source(head_slice=(0, 2)), hidden, None),
        "real-weights": (_toy_source(kind="real"), hidden, None),
        "config": (_toy_source(config=replace(_TOY_CONFIG, gate_lower_bound=-5.0)), hidden, None),
        "input-values": (_toy_source(), _toy_hidden(64, seed=4), None),
        "input-length": (_toy_source(), hidden[:, :32], None),
        "zero-initial-state": (_toy_source(), hidden, _toy_state(0.0)),
        "initial-state": (_toy_source(), hidden, _toy_state(0.5)),
    }
    keys = {name: cpu_reference_cache_path(*arguments) for name, arguments in variants.items()}
    assert len(set(keys.values())) == len(keys), keys
    assert cpu_reference_cache_path(_toy_source(), hidden.clone(), None) == keys["baseline"]


def _toy_chained_case(monkeypatch: pytest.MonkeyPatch) -> tuple[KDATestCase, dict[str, torch.Tensor]]:
    spec = KDA_CASES["kimi_k3-synthetic-mesh2x4-tpaxis1-T1280-chunks2-last992"]
    state_dict = kda_cases.random_weights(_TOY_CONFIG)
    monkeypatch.setattr(KDAWeightSource, "load_state_dict", lambda self: state_dict)
    return KDATestCase(spec=spec, weights=_toy_source(), hidden=_toy_hidden(2 * spec.chunk_tokens)), state_dict


def test_load_only_reference_miss_fails_fast_with_preparation_command(
    oracle_cache: Path, monkeypatch: pytest.MonkeyPatch, expect_error
) -> None:
    monkeypatch.delenv(kda_cases.CACHE_MISS_ENV, raising=False)
    case, _ = _toy_chained_case(monkeypatch)
    monkeypatch.setattr(
        KDAWeightSource, "load_state_dict", lambda self: pytest.fail("load-only must not build weights")
    )
    with expect_error(KDAPreparedCacheMiss, f"--case {case.spec.name}") as error:
        cpu_references(case)
    assert "CPU reference (chunk 0)" in str(error.value) and str(oracle_cache) in str(error.value)
    assert not list(oracle_cache.rglob("*.pt"))


def test_chained_references_carry_state_and_reuse_cache(oracle_cache: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(kda_cases.CACHE_MISS_ENV, raising=False)
    case, state_dict = _toy_chained_case(monkeypatch)

    cold = prepare_cpu_references(case)
    warm = cpu_references(case)  # load-only

    assert [reference.cache_hit for reference in cold] == [False, False]
    assert [reference.cache_hit for reference in warm] == [True, True]
    assert len(list(oracle_cache.rglob("*.pt"))) == 2
    for cold_reference, warm_reference in zip(cold, warm, strict=True):
        assert torch.equal(cold_reference.output, warm_reference.output)
        assert torch.equal(cold_reference.state.recurrent, warm_reference.state.recurrent)
    # Chained chunks equal the single-shot reference over the valid tokens, so chunk 1 used chunk 0's state.
    valid = torch.cat([case.chunk_valid_hidden(0), case.chunk_valid_hidden(1)], dim=1)
    single_output, single_state = kda_forward_reference(valid, state_dict, _TOY_CONFIG)
    chained_output = torch.cat([reference.output for reference in cold], dim=1)
    torch.testing.assert_close(chained_output, single_output, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(cold[-1].state.recurrent, single_state.recurrent, rtol=1e-4, atol=1e-4)


def test_load_only_text_input_miss_fails_fast_and_cached_input_is_used(
    oracle_cache: Path, monkeypatch: pytest.MonkeyPatch, expect_error
) -> None:
    monkeypatch.delenv(kda_cases.CACHE_MISS_ENV, raising=False)
    spec = loudbox_kda_case("real", "LB-A", "single", "text")
    monkeypatch.setattr(kda_cases, "build_text_input", lambda *args: pytest.fail("load-only must not build inputs"))
    with expect_error(KDAPreparedCacheMiss, f"--case {spec.name}") as error:
        build_kda_case(spec)
    assert "text input" in str(error.value) and str(oracle_cache) in str(error.value)

    # A prepared input (written as the preparation step writes it) is what the case uses.
    hidden = torch.randn(1, spec.chunk_tokens, 7168).bfloat16()
    path = text_input_cache_path(spec.model, spec.weight_source().layer_idx, spec.chunk_tokens)
    path.parent.mkdir(parents=True)
    digest = hashlib.sha256(memoryview(hidden.view(torch.uint8).numpy())).hexdigest()
    torch.save({"hidden": hidden, "sha256": digest}, path)
    assert torch.equal(build_kda_case(spec).hidden, hidden)
    assert text_input_cache_path(spec.model, 2, spec.chunk_tokens) != path
    assert text_input_cache_path(spec.model, 1, 2 * spec.chunk_tokens) != path


def test_decay_extreme_inputs_are_input_norm_outputs(expect_error) -> None:
    """A crafted layer input is w * u with per-token RMS(u) = 1 (an input_layernorm output); the builder's check
    rejects any token with RMS(x / w) > 1."""
    weight = torch.tensor([0.5, 0.25])
    check_reachable_input(torch.tensor([[[0.5, -0.25], [0.25, 0.125]]]), weight)  # RMS(u) = 1 and 0.5
    with expect_error(ValueError, "token 1 has RMS"):
        check_reachable_input(torch.tensor([[[0.5, -0.25], [0.5, 0.5]]]), weight)  # u = [1, 2]: RMS 1.58

    case = DECAY_EXTREME_CASES["synthetic-weak-h0-1-T1280x8"]  # nonzero beta target, so a nonzero center
    config = case_config(case)
    weight = case_input_norm_weight(case, config)
    hidden = crafted_hidden(case, case_weights(case), config, weight)
    token_rms = (hidden.double()[0] / weight).square().mean(dim=-1).sqrt()
    assert torch.allclose(token_rms, torch.ones_like(token_rms), atol=2.0**-8)
