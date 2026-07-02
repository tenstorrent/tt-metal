# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Ensure the repo root is on ``sys.path`` so the absolute
``models.demos.deepseek_v3_d_p.reference.cpu_deepseek_v32.*`` imports resolve under any pytest
invocation (plain ``pytest`` does not add the CWD, unlike ``python -m pytest``).
"""

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
