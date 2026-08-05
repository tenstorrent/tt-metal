# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import pytest
import re
from contextlib import contextmanager


@contextmanager
def _expect_error(expected_exception, match=None):
    try:
        yield
    except expected_exception as exc:
        if match is not None and re.search(match, str(exc)) is None:
            raise AssertionError(f"Exception message did not match {match!r}: {exc}") from exc
    else:
        raise AssertionError(f"Expected {expected_exception} to be raised")


@pytest.fixture
def expect_error():
    return _expect_error


@pytest.fixture(autouse=True)
def _reset_default_reveal_pmax():
    """Clear the process-wide derived reveal span between tests.

    ``DiffusionGemmaForCausalLM.__init__`` registers the span it resolved into a module
    global so the controller (built deep inside the denoise-block entry point) can read it
    without re-requiring ``DG_DENOISE_REVEAL_PMAX``. That registration outlives the wrapper,
    so without this reset a test that builds a wrapper leaves a span behind that makes a
    later test's "unset must raise" assertion silently pass instead — an order-dependent
    false green.
    """
    yield
    try:
        from models.experimental.diffusion_gemma.tt import traced_denoise
    except Exception:  # ttnn unavailable in a host-only environment; nothing to reset
        return
    traced_denoise.set_default_reveal_pmax(None)
