# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The bounded ring is sized for the verify, and refused when it cannot hold it.

A packed verify writes the anchor and K drafts before attention runs. On a ring
of exactly ``sliding_window`` the draft slots still hold live window positions,
so every sliding layer attends a shortened window and the committed tokens are
wrong from the first one past ``ring - K``. This used to warn and continue,
which made the failure silent (tt-metal#57701).

The model now sizes the ring itself, so these tests cover the sizing, and the
guard that backstops an operator value too small to be safe.
"""

import os
from types import SimpleNamespace

import pytest

# generator_vllm imports vllm at module scope (through tt_transformers), so
# COLLECTING this file fails on a runner without vLLM -- which is the tt-metal
# unit job. Skip before the import, since the failure is at import.
pytest.importorskip("vllm")

from models.demos.gemma4.tt.attention import SPEC_RING_HEADROOM_ENV, bounded_ring_modulo
from models.demos.gemma4.tt.generator_vllm import (
    _auto_size_spec_ring,
    _reserve_spec_ring_headroom,
    _spec_ring_blocks_for,
)

WINDOW = 1024
SHIPPED_WIDTH = 6  # GEMMA4_DFLASH_VERIFY=5 -> P_v = K + 1


@pytest.fixture(autouse=True)
def _clean_env():
    """Restore the headroom env around every test.

    ``_auto_size_spec_ring`` writes ``os.environ`` directly, and monkeypatch
    only restores what monkeypatch itself changed -- ``delenv`` on an unset var
    records nothing. Without this, a sizing test leaks the headroom into every
    later test in the process and silently changes ``bounded_ring_modulo`` for
    the device suites.
    """
    saved = os.environ.get(SPEC_RING_HEADROOM_ENV)
    os.environ.pop(SPEC_RING_HEADROOM_ENV, None)
    yield
    if saved is None:
        os.environ.pop(SPEC_RING_HEADROOM_ENV, None)
    else:
        os.environ[SPEC_RING_HEADROOM_ENV] = saved


def _cls(width, *, declines=False):
    return SimpleNamespace(_SPEC_N=width, __name__="FakeSpecClass", _SPEC_DECLINES_AT_RING_WRAP=declines)


def _cfg(window=WINDOW):
    return SimpleNamespace(text_config=SimpleNamespace(sliding_window=window))


# --------------------------------------------------------------------------
# Sizing: the model does what the operator used to have to do by hand.
# --------------------------------------------------------------------------


def test_a_speculating_class_gets_a_ring_that_holds_its_drafts():
    """The shipped 31B dFlash / 12B MTP case: 1024 window, K=5 -> ring 2048."""
    _auto_size_spec_ring(_cls(SHIPPED_WIDTH), _cfg(), bounded_sliding=True)
    assert bounded_ring_modulo(WINDOW) == 2048
    _reserve_spec_ring_headroom(WINDOW, SHIPPED_WIDTH, "Gemma4DFlash")  # must not raise


def test_the_ring_is_sized_from_drafts_not_from_the_anchor():
    """K drafts need K slots, not K + 1.

    The anchor write at p lands in slot p % ring, which held p - ring -- already
    outside the window -- so only the drafts evict live history. This is the same
    test the contract rail applies per request (``_dflash_ring_advice``).

    The shipped 1024 window cannot show the difference (1024 + 5 and 1024 + 6
    both round to 2048), so this uses a window where counting the anchor crosses
    a power of two: 3776 + 320 IS 4096, while 3776 + 321 forces 8192. Counting
    one row too many would double the bounded pool for every sliding layer.
    """
    assert _spec_ring_blocks_for(3776, 320) == (4096 - 3776) // 64
    assert _spec_ring_blocks_for(3776, 321) == (8192 - 3776) // 64
    # The shipped case, for the record: 1024 window, K=5 -> 2048 ring -> 16 blocks.
    assert _spec_ring_blocks_for(WINDOW, 5) == 16


def test_a_baseline_class_is_left_alone():
    """No _SPEC_N: nothing speculates, so the exact-window ring is correct."""
    _auto_size_spec_ring(SimpleNamespace(__name__="Gemma4ForCausalLM"), _cfg(), bounded_sliding=True)
    assert bounded_ring_modulo(WINDOW) == WINDOW


def test_unbounded_serving_is_left_alone():
    """Without bounded sliding there is no ring to size."""
    _auto_size_spec_ring(_cls(SHIPPED_WIDTH), _cfg(), bounded_sliding=False)
    assert SPEC_RING_HEADROOM_ENV not in os.environ


def test_full_attention_models_are_left_alone():
    _auto_size_spec_ring(_cls(SHIPPED_WIDTH), _cfg(window=None), bounded_sliding=True)
    assert SPEC_RING_HEADROOM_ENV not in os.environ


def test_an_operator_value_wins(monkeypatch):
    """A pinned value is not overwritten -- it is checked instead."""
    monkeypatch.setenv(SPEC_RING_HEADROOM_ENV, "16")
    _auto_size_spec_ring(_cls(SHIPPED_WIDTH), _cfg(), bounded_sliding=True)
    assert os.environ[SPEC_RING_HEADROOM_ENV] == "16"


# --------------------------------------------------------------------------
# Guard: the backstop for an operator value that is too small.
# --------------------------------------------------------------------------


def test_an_exact_window_ring_is_refused(expect_error):
    with expect_error(RuntimeError, "cannot hold a verify"):
        _reserve_spec_ring_headroom(WINDOW, SHIPPED_WIDTH, "Gemma4DFlash")


def test_the_refusal_names_the_remedy_and_the_issue():
    try:
        _reserve_spec_ring_headroom(WINDOW, SHIPPED_WIDTH, "Gemma4DFlash")
    except RuntimeError as exc:
        message = str(exc)
    else:
        raise AssertionError("an exact-window ring must not be accepted")
    assert SPEC_RING_HEADROOM_ENV in message, "the remedy must be named"
    assert "57701" in message, "the issue must be findable from the failure"
    assert "16" in message, "the working value must be given, not just described"


def test_a_class_that_declines_at_the_wrap_is_not_refused():
    """The contract rail stops proposing before the wrap and carries on as plain
    decode, so an exact ring costs it speed, not correctness."""
    _reserve_spec_ring_headroom(WINDOW, SHIPPED_WIDTH, "Gemma4DFlashContract", declines_at_wrap=True)


def test_width_one_needs_no_headroom():
    """Width 1 writes only the anchor, so it evicts nothing in the live window."""
    _reserve_spec_ring_headroom(WINDOW, 1, "Gemma4DFlash")


def test_an_invalid_ring_is_reported_whatever_the_width(monkeypatch, expect_error):
    """The ring is validated before the width check.

    Width 1 needs no headroom, but a ring that is not a power of two is wrong for
    any width. Validating after the width check would have made this guard's
    error path depend on verify_width, leaving a bad env value to surface later
    from the model or trace path instead of at config time.
    """
    monkeypatch.setenv(SPEC_RING_HEADROOM_ENV, "32")  # 1024 + 32*64 = 3072
    with expect_error(ValueError, "power of two"):
        _reserve_spec_ring_headroom(WINDOW, 1, "Gemma4DFlash")


def test_full_attention_is_unaffected_by_the_guard():
    _reserve_spec_ring_headroom(None, SHIPPED_WIDTH, "Gemma4DFlash")


def test_the_check_is_on_the_ring_not_on_the_env_var(monkeypatch, expect_error):
    """Headroom set too small is still unsafe.

    The first version of this guard returned as soon as the env var was present,
    so an under-sized headroom passed. A 1024 window with 32 blocks is a 3072
    ring, not a power of two, so the ring builder rejects it rather than the
    guard silently accepting a value that only LOOKS configured.
    """
    monkeypatch.setenv(SPEC_RING_HEADROOM_ENV, "32")
    with expect_error(ValueError, "power of two"):
        _reserve_spec_ring_headroom(WINDOW, SHIPPED_WIDTH, "Gemma4DFlash")


def test_every_real_speculating_class_gets_sized():
    """The three shipped speculating classes, by their real _SPEC_N.

    The fakes above pin the arithmetic; this pins the wiring. A new serving
    class that forgets ``_SPEC_N``, or one whose K changes, silently falls back
    to the exact-window ring -- which is the #57701 defect, and is invisible
    until someone reads generated text past the ring.
    """
    from models.demos.gemma4.tt.generator_vllm import (
        Gemma4DFlashContractForCausalLM,
        Gemma4DFlashForCausalLM,
        Gemma4MTPForCausalLM,
    )

    for cls in (Gemma4DFlashForCausalLM, Gemma4DFlashContractForCausalLM, Gemma4MTPForCausalLM):
        os.environ.pop(SPEC_RING_HEADROOM_ENV, None)
        assert int(cls._SPEC_N) > 1, f"{cls.__name__} must declare a verify width"
        _auto_size_spec_ring(cls, _cfg(), bounded_sliding=True)
        ring = bounded_ring_modulo(WINDOW)
        assert ring - WINDOW >= cls._SPEC_N - 1, f"{cls.__name__}: ring {ring} cannot hold its drafts"
        # and the guard agrees, including for the rail that declines at the wrap
        _reserve_spec_ring_headroom(WINDOW, cls._SPEC_N, cls.__name__, declines_at_wrap=cls._SPEC_DECLINES_AT_RING_WRAP)
