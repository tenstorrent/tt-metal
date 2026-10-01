# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Quasar fork of tests/ttnn/unit_tests/operations/sdpa/test_windowed_sdpa.py.

Runs the same windowed (block-diagonal) SDPA cases against
ttnn.experimental.quasar.transformer.scaled_dot_product_attention by swapping the op entry point
before the reused tests call it.
"""

import ttnn
from ttnn.experimental.quasar.transformer import scaled_dot_product_attention

ttnn.transformer.scaled_dot_product_attention = scaled_dot_product_attention

from tests.ttnn.unit_tests.operations.sdpa.test_windowed_sdpa import *  # noqa: E402,F401,F403
