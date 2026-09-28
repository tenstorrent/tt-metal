# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Test helper: build a tt_dit module with its legacy SDPA configuration, for parity checks.

On Blackhole every tt_dit SDPA call runs a named recipe (``sdpa_precision=None`` selects the module's
``sdpa_precision_default``). The legacy configuration (explicit chunks, ``compute_kernel_config``) is
what a module builds off Blackhole; ``sdpa_variant(LEGACY)`` builds exactly that on any device by
making ``sdpa_recipe.resolve_precision`` return ``None`` while the module is constructed. It is a test
hook only: models have no legacy flag.
"""

from __future__ import annotations

import contextlib
from unittest import mock

from ...utils import sdpa_recipe

LEGACY = "legacy"  # smoke-test precision marker: the module's legacy (non-Blackhole) SDPA setup


@contextlib.contextmanager
def sdpa_variant(precision):
    """Yield the ``sdpa_precision`` constructor argument for a smoke variant.

    ``LEGACY`` yields ``None`` with recipe resolution disabled (legacy SDPA); anything else is passed
    through (``None`` = the module's default recipe).
    """
    if precision != LEGACY:
        yield precision
        return
    with mock.patch.object(sdpa_recipe, "resolve_precision", lambda *args, **kwargs: None):
        yield None
