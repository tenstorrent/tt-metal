# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The bounded-ring guard refuses a verify the ring cannot hold (tt-metal#57701).

A packed verify writes candidates at p+1..p+K before attention runs. On a ring
of exactly ``sliding_window`` those slots still hold live window positions, so
every sliding layer attends a shortened window and the committed tokens are
wrong from the first one past ``ring - verify_width``. This used to warn and
continue, which made the failure silent.
"""

import pytest

# generator_vllm imports vllm at module scope (through tt_transformers), so
# COLLECTING this file fails on a runner without vLLM -- which is the tt-metal
# unit job. Skip before the import, since the failure is at import.
pytest.importorskip("vllm")

from models.demos.gemma4.tt.attention import SPEC_RING_HEADROOM_ENV
from models.demos.gemma4.tt.generator_vllm import _ALLOW_UNSAFE_SPEC_RING_ENV, _reserve_spec_ring_headroom

WINDOW = 1024
SHIPPED_VERIFY_WIDTH = 6  # GEMMA4_DFLASH_VERIFY=5 -> P_v = K + 1


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv(SPEC_RING_HEADROOM_ENV, raising=False)
    monkeypatch.delenv(_ALLOW_UNSAFE_SPEC_RING_ENV, raising=False)


def test_exact_window_ring_refuses_a_packed_verify(expect_error):
    """The shipped 31B dFlash configuration: ring 1024, verify width 6."""
    with expect_error(RuntimeError, "exact-window ring"):
        _reserve_spec_ring_headroom(WINDOW, SHIPPED_VERIFY_WIDTH, "Gemma4DFlash")


def test_the_refusal_names_the_remedy_and_the_issue():
    try:
        _reserve_spec_ring_headroom(WINDOW, SHIPPED_VERIFY_WIDTH, "Gemma4DFlash")
    except RuntimeError as exc:
        message = str(exc)
    else:
        raise AssertionError("an exact-window ring must not be accepted")
    assert SPEC_RING_HEADROOM_ENV in message, "the remedy must be named"
    assert "GEMMA4_MAX_TOKENS_ALL_USERS" in message, "doubling the ring has to be paid for"
    assert "57701" in message, "the issue must be findable from the failure"
    assert _ALLOW_UNSAFE_SPEC_RING_ENV in message, "the opt-out must be discoverable"


def test_sufficient_headroom_is_accepted(monkeypatch):
    """16 blocks doubles a 1024 window to a 2048 ring, which clears any width."""
    monkeypatch.setenv(SPEC_RING_HEADROOM_ENV, "16")
    _reserve_spec_ring_headroom(WINDOW, SHIPPED_VERIFY_WIDTH, "Gemma4DFlash")


def test_width_one_needs_no_headroom():
    """Width 1 writes only the anchor, so it evicts nothing in the live window."""
    _reserve_spec_ring_headroom(WINDOW, 1, "Gemma4DFlash")


def test_full_attention_models_are_unaffected():
    _reserve_spec_ring_headroom(None, SHIPPED_VERIFY_WIDTH, "Gemma4DFlash")


def test_the_opt_out_downgrades_to_a_warning(monkeypatch):
    """Short-context deployments can accept the corruption knowingly."""
    monkeypatch.setenv(_ALLOW_UNSAFE_SPEC_RING_ENV, "1")
    _reserve_spec_ring_headroom(WINDOW, SHIPPED_VERIFY_WIDTH, "Gemma4DFlash")


def test_the_check_is_on_the_ring_not_on_the_env_var(monkeypatch, expect_error):
    """Headroom set too small is still unsafe.

    The previous version returned as soon as the env var was present, so an
    under-sized headroom passed. This asserts the guard measures the ring.
    A 1024 window with 32 blocks of headroom is a 3072 ring, which is not a
    power of two -- so the ring builder rejects it rather than the guard
    silently accepting a value that only LOOKS configured.
    """
    monkeypatch.setenv(SPEC_RING_HEADROOM_ENV, "32")
    with expect_error(ValueError, "power of two"):
        _reserve_spec_ring_headroom(WINDOW, SHIPPED_VERIFY_WIDTH, "Gemma4DFlash")
