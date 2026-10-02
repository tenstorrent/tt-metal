# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Native-context ceiling guard: the serving templates waive vLLM's own
max_model_len check (VLLM_ALLOW_LONG_MAX_MODEL_LEN), so gemma4 must refuse a
max_seq_len past max_position_embeddings itself -- at boot, not at runtime."""
from types import SimpleNamespace

import pytest

# generator_vllm imports vllm at module level; the unit tier has no vllm.
pytest.importorskip("vllm")

from models.demos.gemma4.tests.unit.dflash_contract_harness import make_expect_error  # noqa: E402
from models.demos.gemma4.tt.generator_vllm import _assert_within_native_context  # noqa: E402

expect_error = pytest.fixture(lambda: make_expect_error())

NATIVE = 262_144


def _args(native=NATIVE):
    return SimpleNamespace(_hf_text_config=SimpleNamespace(max_position_embeddings=native))


def test_at_native_passes(monkeypatch):
    monkeypatch.delenv("GEMMA4_ALLOW_BEYOND_NATIVE", raising=False)
    _assert_within_native_context(NATIVE, _args())


def test_below_native_passes(monkeypatch):
    monkeypatch.delenv("GEMMA4_ALLOW_BEYOND_NATIVE", raising=False)
    _assert_within_native_context(131_072, _args())


def test_one_past_native_raises(monkeypatch, expect_error):
    monkeypatch.delenv("GEMMA4_ALLOW_BEYOND_NATIVE", raising=False)
    with expect_error(ValueError, "native max_position_embeddings"):
        _assert_within_native_context(NATIVE + 1, _args())


def test_the_scenario_from_the_report_raises(monkeypatch, expect_error):
    # An operator writes max_context: 300000; the pool convention follows it.
    monkeypatch.delenv("GEMMA4_ALLOW_BEYOND_NATIVE", raising=False)
    with expect_error(ValueError, "VLLM_ALLOW_LONG_MAX_MODEL_LEN"):
        _assert_within_native_context(300_000, _args())


def test_explicit_env_waiver_passes(monkeypatch):
    monkeypatch.setenv("GEMMA4_ALLOW_BEYOND_NATIVE", "1")
    _assert_within_native_context(300_000, _args())


def test_unknown_native_stays_permissive(monkeypatch):
    # A model_args without the attribute (older adaptors) must not brick boots.
    monkeypatch.delenv("GEMMA4_ALLOW_BEYOND_NATIVE", raising=False)
    _assert_within_native_context(300_000, SimpleNamespace())
