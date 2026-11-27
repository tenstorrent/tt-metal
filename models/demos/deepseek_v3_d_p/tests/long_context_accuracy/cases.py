# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Every long-context accuracy case of the models in this folder: one model folder's ``cases.CASES`` each."""

from __future__ import annotations

from models.demos.common.prefill.long_context_accuracy.block import LongContextCase
from models.demos.deepseek_v3_d_p.tests.long_context_accuracy.kimi_k3 import cases as kimi_k3
from models.demos.deepseek_v3_d_p.tests.long_context_accuracy.kimi_k3_1 import cases as kimi_k3_1

ALL_CASES: tuple[LongContextCase, ...] = kimi_k3.CASES + kimi_k3_1.CASES
