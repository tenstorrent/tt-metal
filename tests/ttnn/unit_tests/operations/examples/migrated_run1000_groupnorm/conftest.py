# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Make the vendored golden harness importable without external PYTHONPATH."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
