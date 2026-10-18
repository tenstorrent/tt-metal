# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import sys
from pathlib import Path

EXAMPLES_ROOT = Path(__file__).resolve().parents[1]

# On sys.path so `grpo_remote_rollout`, the package this directory defines, is importable.
sys.path.insert(0, str(EXAMPLES_ROOT))
