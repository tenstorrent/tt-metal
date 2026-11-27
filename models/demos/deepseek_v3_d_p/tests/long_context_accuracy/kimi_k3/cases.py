# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Kimi-K3 long-context accuracy cases (``block.LongContextCase``), run by ``test_long_context_accuracy.py`` and checked
against their goldens by ``test_cases_match_goldens.py``.

The KDA layer-0 case (golden ``k3-l0-kda-fp64-1m``, reader ``golden.KimiK3KdaGolden``, device Block
``KimiK3KdaBlock`` in ``blocks.py``) is added by m06n.11.3.
"""

from __future__ import annotations

from models.demos.common.prefill.long_context_accuracy.block import LongContextCase

CASES: tuple[LongContextCase, ...] = ()
