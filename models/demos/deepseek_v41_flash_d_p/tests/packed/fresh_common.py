# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""PB.MB: the fresh walls' shared parts live in ONE module, tests/fresh_common.py (PB.9F's packed walls and MB.1's
single-request walls through adapter.prefill_chunk); this name is that same module object (not a copy), so the packed
walls' imports keep working."""

import sys

from models.demos.deepseek_v41_flash_d_p.tests import fresh_common as _fresh_common

sys.modules[__name__] = _fresh_common
