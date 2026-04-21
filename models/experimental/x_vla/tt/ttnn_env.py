# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Route ttnn imports to the xvla tt-metal tree.

This host has ttnn editable-installed from a sibling tt-metal
(`/home/ttuser/experiments/pi0_5/tt-metal`). Its runtime .so expects kernel
sources at paths that only exist in the xvla tree (different checkout).
Prepending the xvla tt-metal directories to sys.path and setting
TT_METAL_HOME makes the xvla ttnn package and kernels win.

Must be imported BEFORE `import ttnn` anywhere else in the process.
"""

from __future__ import annotations

import os
import sys

TT_METAL_HOME = "/home/ttuser/experiments/xvla/tt-metal"


def install() -> None:
    os.environ.setdefault("TT_METAL_HOME", TT_METAL_HOME)
    for p in (
        TT_METAL_HOME,
        f"{TT_METAL_HOME}/ttnn",
        f"{TT_METAL_HOME}/tools",
    ):
        if p not in sys.path:
            sys.path.insert(0, p)
