# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest

import ttnn

from ...utils import matmul

_UNTABLED = (32 * 1001, 32 * 7, 32 * 5)


@pytest.fixture
def untabled_shape():
    yield _UNTABLED
    matmul.grid_88_configs.pop(_UNTABLED, None)
    matmul._clear_config_caches()


def _blocks(config):
    return config.M_block_size, config.K_block_size, config.N_block_size, config.subblock_h, config.subblock_w


def test_matmul_config_is_resolved_once_and_copied(untabled_shape) -> None:
    grid = ttnn.CoreCoord(8, 8)
    first = matmul.get_matmul_config(*untabled_shape, grid, default_block_size=[4, 4, 4])
    expected = _blocks(first)
    first.K_block_size = 1
    hits = matmul._resolve_matmul_config.cache_info().hits

    second = matmul.get_matmul_config(*untabled_shape, grid, default_block_size=(4, 4, 4))

    assert matmul._resolve_matmul_config.cache_info().hits == hits + 1
    assert _blocks(second) == expected


def test_agmm_config_is_copied() -> None:
    kwargs = dict(full_grid=ttnn.CoreCoord(12, 10), cluster_size=4, num_links=2, core_grid=ttnn.CoreCoord(12, 9))
    _, first, workers = matmul.get_agmm_config(4096, 3072, 9216, **kwargs)
    expected = _blocks(first)
    first.M_block_size = 99

    grid, second, second_workers = matmul.get_agmm_config(4096, 3072, 9216, **kwargs)

    assert _blocks(second) == expected
    assert (grid.x, grid.y, second_workers) == (12, 9, workers)


def test_register_matmul_configs_clears_the_cache(untabled_shape) -> None:
    grid = ttnn.CoreCoord(8, 8)
    matmul.get_matmul_config(*untabled_shape, grid)

    matmul.register_matmul_configs({"8x8": {untabled_shape: (1, 7, 5, (1, 1))}})

    assert _blocks(matmul.get_matmul_config(*untabled_shape, grid)) == (1, 7, 5, 1, 1)
