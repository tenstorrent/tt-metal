# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
# Round 3 eltwise binary CI tests open a device: only under the card lock (hwlock), in a CI container or on the mock cluster.
import os
import sys

if not (os.environ.get("HWLOCK_HELD") or os.environ.get("GITHUB_ACTIONS") or os.environ.get("TT_METAL_MOCK_CLUSTER_DESC_PATH")):
    sys.exit("not under hwlock")
