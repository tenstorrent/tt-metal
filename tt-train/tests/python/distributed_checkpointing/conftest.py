# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Makes this directory's helper (``dcp_oracle``) and the parent directory's (``tp_block``) importable whether pytest
collects this directory alone or the whole ``tt-train/tests/python`` tree."""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
for _path in (_HERE, os.path.dirname(_HERE)):
    if _path not in sys.path:
        sys.path.insert(0, _path)
