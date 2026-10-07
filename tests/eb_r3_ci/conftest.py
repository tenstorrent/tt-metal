# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Card rule 05: tests here open a device only under hwlock (HWLOCK_HELD), in the CI container (GITHUB_ACTIONS) or on a
mock cluster (TT_METAL_MOCK_CLUSTER_DESC_PATH)."""
import os
import sys

if not (os.environ.get("HWLOCK_HELD") or os.environ.get("GITHUB_ACTIONS") or os.environ.get("TT_METAL_MOCK_CLUSTER_DESC_PATH")):
    sys.exit("not under hwlock")
