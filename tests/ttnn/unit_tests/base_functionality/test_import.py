# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import subprocess
import sys


def test_import_without_torch():
    # A fresh interpreter must import TTNN even when optional PyTorch is unavailable.
    result = subprocess.run(
        [sys.executable, "-c", "import sys; sys.modules['torch'] = None; import ttnn"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
