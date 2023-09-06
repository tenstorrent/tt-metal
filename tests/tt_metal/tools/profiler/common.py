"""
SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

SPDX-License-Identifier: Apache-2.0
"""

import os


def get_repo_path():
    try:
        REPO_PATH = os.environ["TT_METAL_HOME"]
    except KeyError:
        assert False, "TT_METAL_HOME has to be setup. Please refer to getting started docs"

    return REPO_PATH
