# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest

import ttnn
from models.tt_dit.models.transformers.flux3_klein import transformer_flux3_klein  # noqa: F401  (registers the entries)
from models.tt_dit.utils.matmul import get_matmul_config


def blocking(m, n):
    c = get_matmul_config(m, 3072, n, ttnn.CoreCoord(12, 9), use_heuristic=True)
    return c.M_block_size, c.K_block_size, c.N_block_size


@pytest.mark.parametrize(
    ("m", "n", "expected"),
    [
        (5120, 2304, (11, 8, 8)),  # 10 references, early stage (226340: 14x8x8 clashed)
        (3200, 4608, (4, 8, 16)),  # 2048 + 2 references (226340: 9x8x16 past L1)
        (9344, 4608, (4, 8, 16)),  # 4096 + 2 references (226340: 16x8x16)
        (9344, 2304, (11, 8, 8)),
    ],
)
def test_overflowing_shapes_are_cut(m, n, expected):
    assert blocking(m, n) == expected


@pytest.mark.parametrize(
    ("m", "n", "expected"),
    [(2048, 4608, (6, 8, 16)), (2816, 2304, (8, 8, 8)), (1664, 4608, (5, 8, 16)), (5120, 4608, (6, 8, 16))],
)
def test_shapes_that_run_keep_the_heuristic(m, n, expected):  # 226340 ran these
    assert blocking(m, n) == expected


def test_tabled_shapes_keep_their_entry():
    assert blocking(8192, 2304) == (11, 8, 8)
    assert blocking(4224, 4608) == (4, 8, 16)


@pytest.mark.parametrize("n", [2304, 4608, 9216])
def test_every_m_fits(n):
    for m in range(128, 32768 + 1, 128):
        mb, kb, nb = blocking(m, n)
        assert mb * kb + kb * nb + mb * nb <= 272, (m, n, mb, kb, nb)
