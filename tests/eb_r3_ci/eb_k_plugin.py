# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Round 3 eltwise binary CI: a -k expression from EB_K_EXPR (the tracy wrapper splits command-line arguments at spaces)."""
import os


def pytest_configure(config):
    k = os.environ.get("EB_K_EXPR")
    if k:
        config.option.keyword = k
