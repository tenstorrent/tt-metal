# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for adaptive speculative draft length (K) selection.

These are host-only (no device) — they validate the workload classifier and
env-driven band selection used by the FastAPI server / demo.
"""

from __future__ import annotations

import pytest

from models.demos.gemma4.tt.adaptive_draft_len import (
    AdaptiveDraftLenConfig,
    classify_workload,
    select_adaptive_draft_len,
)


@pytest.mark.parametrize(
    "prompt,expected",
    [
        ("Write a Python function called merge_sort", "high_accept"),
        ("Repeat the following sentence 10 times, numbering each line", "high_accept"),
        ("tell me a story in 500 words", "low_accept"),
        ("Write me a creative short story about a fox", "low_accept"),
        ("Summarize the following passage in three bullets", "medium_accept"),
        ("What is the capital of France?", "high_accept"),
    ],
)
def test_classify_workload_prompt_cues(prompt, expected):
    assert classify_workload(prompt) == expected


def test_classify_short_open_ended_as_low():
    assert classify_workload("Hello", prompt_tokens=20, max_new_tokens=512, short_prompt_tokens=128) == "low_accept"


def test_classify_long_prompt_as_high():
    assert classify_workload("continue", prompt_tokens=4096, long_prompt_tokens=2048) == "high_accept"


def test_select_bands_from_config():
    cfg = AdaptiveDraftLenConfig(enabled=True, k_high=16, k_mid=8, k_low=6)
    k, w = select_adaptive_draft_len("tell me a story", config=cfg)
    assert (k, w) == (6, "low_accept")
    k, w = select_adaptive_draft_len("Summarize this paragraph", config=cfg)
    assert (k, w) == (8, "medium_accept")
    k, w = select_adaptive_draft_len("Write a Python merge_sort", config=cfg)
    assert (k, w) == (16, "high_accept")


def test_select_override_skips_adaptive():
    cfg = AdaptiveDraftLenConfig(enabled=True, k_high=16, k_mid=8, k_low=6)
    k, w = select_adaptive_draft_len("tell me a story", config=cfg, override=12)
    assert (k, w) == (12, None)


def test_select_disabled_returns_high_band():
    cfg = AdaptiveDraftLenConfig(enabled=False, k_high=16, k_mid=8, k_low=6)
    k, w = select_adaptive_draft_len("tell me a story", config=cfg)
    assert (k, w) == (16, None)


def test_from_env_orders_bands(monkeypatch):
    monkeypatch.setenv("GEMMA4_SPEC_DRAFT_LEN", "16")
    monkeypatch.setenv("GEMMA4_SPEC_DRAFT_LEN_MID", "32")  # invalid > high → clamped
    monkeypatch.setenv("GEMMA4_SPEC_DRAFT_LEN_LOW", "8")
    monkeypatch.setenv("GEMMA4_SPEC_ADAPTIVE_K", "1")
    cfg = AdaptiveDraftLenConfig.from_env()
    assert cfg.enabled is True
    assert cfg.k_high == 16
    assert cfg.k_mid == 16  # clamped to high
    assert cfg.k_low == 8


def test_from_env_can_disable(monkeypatch):
    monkeypatch.setenv("GEMMA4_SPEC_ADAPTIVE_K", "0")
    cfg = AdaptiveDraftLenConfig.from_env(default_draft_len=16)
    assert cfg.enabled is False
    assert cfg.k_high == 16


def test_set_draft_len_releases_fused_trace(expect_error):
    """SpeculativeDecoder.set_draft_len must release a live fused trace on change."""
    from models.demos.gemma4.tt.spec_decode import SpeculativeDecoder

    # Minimal stub: bypass __init__ device requirements.
    spec = object.__new__(SpeculativeDecoder)
    spec.draft_len = 16
    released = {"called": False}

    def _release():
        released["called"] = True
        spec._fused_trace = None

    spec.release_fused_trace = _release  # type: ignore[method-assign]
    spec._verify_traces = {"old": object()}
    spec._draft_trace = object()
    spec._fused_trace = {"id": 1}

    assert spec.set_draft_len(16) is False
    assert released["called"] is False
    assert spec.draft_len == 16

    assert spec.set_draft_len(6) is True
    assert released["called"] is True
    assert spec.draft_len == 6
    assert spec._verify_traces == {}
    assert spec._draft_trace is None

    with expect_error(ValueError, "draft_len must be >= 1"):
        spec.set_draft_len(0)
