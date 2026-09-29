# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Deprecated import path: the §4 prototype oracle is replaced by ``reference/deepseek_v41/oracle.py``
(``real_spec((2, 3, 20, 21, 24), 2048, candidate_topk_blocks=96)`` is the prototype schedule) and the
device-weight helper moved to ``tests/v41/reference_weights.py``. Remove once no test imports this module."""

from models.demos.deepseek_v3_d_p.tests.v41.reference_weights import device_weights  # noqa: F401
