# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host-only tests for the pure speculative draft-length policies.

Pins the measured boundaries in ``auto_draft_len`` (prompt >= 1024 -> K=5)
and ``auto_draft_len_batched`` (K = min(context_K, 32 // B - 1); 0 means
speculation OFF) so a silent regression in either table cannot ship — both
are plain arithmetic that 24 device tests exercise only indirectly.
"""

import pytest

from models.demos.gemma4.tt.spec_decode import auto_draft_len, auto_draft_len_batched


@pytest.fixture(autouse=True)
def _no_draft_len_env(monkeypatch):
    monkeypatch.delenv("GEMMA4_SPEC_DRAFT_LEN", raising=False)


@pytest.mark.parametrize(
    ("prompt_len", "expected"),
    [
        (None, 3),  # unknown prompt -> short default
        (128, 3),
        (1023, 3),  # boundary: strictly below the knee
        (1024, 5),  # boundary: at the knee
        (131072, 5),
    ],
)
def test_auto_draft_len_prompt_boundary(prompt_len, expected):
    assert auto_draft_len(prompt_len) == expected


def test_auto_draft_len_env_override_wins(monkeypatch):
    monkeypatch.setenv("GEMMA4_SPEC_DRAFT_LEN", "7")
    assert auto_draft_len(128) == 7
    assert auto_draft_len(131072) == 7


def test_auto_draft_len_env_auto_defers_to_policy(monkeypatch):
    monkeypatch.setenv("GEMMA4_SPEC_DRAFT_LEN", "auto")
    assert auto_draft_len(128) == 3
    assert auto_draft_len(4096) == 5


def test_auto_draft_len_custom_default():
    assert auto_draft_len(None, default=2) == 2
    assert auto_draft_len(64, default=2) == 2


@pytest.mark.parametrize(
    ("batch", "prompt_len", "expected"),
    [
        (None, 4096, 5),  # no batch -> plain policy
        (1, 4096, 5),  # B=1 -> plain policy
        (8, 128, 3),  # short prompt K=3 within the 32-row cap (8*(3+1)=32)
        (8, 4096, 3),  # context K=5 CLIPPED to the row cap 32//8-1=3
        (16, 4096, 1),  # row cap 32//16-1=1
        (32, 4096, 0),  # past the compute knee: speculation OFF
        (33, 128, 0),  # row cap goes negative -> OFF, never negative K
    ],
)
def test_auto_draft_len_batched_row_cap(batch, prompt_len, expected):
    assert auto_draft_len_batched(prompt_len, batch) == expected


def test_auto_draft_len_batched_env_override_wins(monkeypatch):
    monkeypatch.setenv("GEMMA4_SPEC_DRAFT_LEN", "2")
    assert auto_draft_len_batched(4096, 32) == 2  # explicit K bypasses the cap


def test_auto_draft_len_batched_env_auto_defers_to_policy(monkeypatch):
    monkeypatch.setenv("GEMMA4_SPEC_DRAFT_LEN", "auto")
    assert auto_draft_len_batched(4096, 8) == 3
    assert auto_draft_len_batched(4096, 32) == 0
