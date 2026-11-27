# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Kimi-K3.1 long-context accuracy cases (``block.LongContextCase``) against the FP64 chain golden ``k31-l0-3-fp64-1m``
(reader ``golden.KimiK31ChainGolden``), run by ``test_long_context_accuracy.py`` and checked against the golden by
``test_cases_match_goldens.py``: the MoE layer 1 and 2 cases (m06n.11.4) and the DSA layer 3 case (m06n.11.5).

Each block's cases, including their imports, stay inside their own section below.
"""

from __future__ import annotations

from models.demos.common.prefill.long_context_accuracy.block import LongContextCase

# ==== MoE (m06n.11.4) ===============================================================================================
MOE_CASES: tuple[LongContextCase, ...] = ()
# ==== end of MoE ====================================================================================================

# ==== DSA (m06n.11.5) ===============================================================================================
DSA_CASES: tuple[LongContextCase, ...] = ()
# ==== end of DSA ====================================================================================================

CASES = MOE_CASES + DSA_CASES
