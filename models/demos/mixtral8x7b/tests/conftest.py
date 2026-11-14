# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import gc

import pytest


@pytest.fixture(autouse=True)
def ensure_gc():
    gc.collect()
