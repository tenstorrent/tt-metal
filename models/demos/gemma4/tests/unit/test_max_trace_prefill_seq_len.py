# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Coverage for ``resolve_gemma4_max_trace_prefill_seq_len``.

The max traced-prefill length is a per-(device, model) memory limit derived in
model code (WH-T3K 31B cannot hold the full prefill-trace bucket sweep — see
tt-metal #49929). The helper is vllm-free and takes plain strings, so these run
in the Gemma-4 models-unit-tests pipeline with no device.
"""

import pytest

from models.demos.gemma4.tt.generator_trace import (
    GEMMA4_MAX_TRACE_PREFILL_SEQ_LEN,
    resolve_gemma4_max_trace_prefill_seq_len,
)


@pytest.fixture(autouse=True)
def _clear_override(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("GEMMA4_MAX_TRACE_PREFILL_SEQ_LEN", raising=False)


def test_wh_t3k_31b_is_capped():
    """WH-T3K 31B caps the traced prefill length so larger prefills run non-traced."""
    # base_model_name may arrive as a local snapshot dir; the "gemma-4-31B" marker
    # is present either way.
    assert (
        resolve_gemma4_max_trace_prefill_seq_len(
            device_name="T3K",
            base_model_name="/mnt/MLPerf/hub/models--google--gemma-4-31B-it/snapshots/abc",
        )
        == 128
    )


def test_qb2_31b_keeps_full_sweep():
    """BH-QB2 (P150x4) 31B has DRAM headroom — keeps the default (full sweep)."""
    assert (
        resolve_gemma4_max_trace_prefill_seq_len(device_name="P150x4", base_model_name="google/gemma-4-31B-it")
        == GEMMA4_MAX_TRACE_PREFILL_SEQ_LEN
    )


def test_t3k_non_31b_keeps_default():
    """The cap is 31B-specific; a smaller model on T3K keeps the default."""
    assert (
        resolve_gemma4_max_trace_prefill_seq_len(device_name="T3K", base_model_name="google/gemma-4-12B-it")
        == GEMMA4_MAX_TRACE_PREFILL_SEQ_LEN
    )


def test_env_override_wins(monkeypatch: pytest.MonkeyPatch):
    """GEMMA4_MAX_TRACE_PREFILL_SEQ_LEN overrides the derived value (per-deployment tuning)."""
    monkeypatch.setenv("GEMMA4_MAX_TRACE_PREFILL_SEQ_LEN", "512")
    assert resolve_gemma4_max_trace_prefill_seq_len(device_name="T3K", base_model_name="google/gemma-4-31B-it") == 512
