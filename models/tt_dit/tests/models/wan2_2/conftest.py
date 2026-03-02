# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest


@pytest.fixture
def M_block_size():
    return 8


@pytest.fixture
def K_block_size():
    return 7  # Testing K=4096 with K_block=7


@pytest.fixture
def N_block_size():
    return 8


@pytest.fixture
def subblock_h():
    return 2


@pytest.fixture
def subblock_w():
    return 2
