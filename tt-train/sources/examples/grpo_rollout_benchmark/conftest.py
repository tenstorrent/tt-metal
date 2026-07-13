# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import os
import sys

# Make ``benchmark_common`` and ``utils.*`` importable when tests/scripts run.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
