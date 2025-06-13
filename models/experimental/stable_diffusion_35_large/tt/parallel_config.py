# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from typing import NamedTuple


class ParallelConfig(NamedTuple):
    mesh_shape: tuple[int, int]
    factor: int
    mesh_axis: int

    # TODO: factor as property


class DiTParallelConfig(NamedTuple):
    cfg_parallel: ParallelConfig
    tensor_parallel: ParallelConfig
    # sequence_parallel: ParallelConfig
    # ring_parallel: ParallelConfig
    # ulysses_parallel: ParallelConfig


def create_dit_parallel_config(
    mesh_shape: tuple[int, int],
    cfg_parallel: ParallelConfig,
    tensor_parallel: ParallelConfig,
    # sequence_parallel: ParallelConfig,
    # ring_parallel: ParallelConfig,
    # ulysses_parallel: ParallelConfig
) -> DiTParallelConfig:
    # validate cfg config
    assert cfg_parallel.factor in [1, 2]
    assert cfg_parallel.mesh_axis in [0, 1]
    assert cfg_parallel.mesh_shape[cfg_parallel.mesh_axis] == mesh_shape[cfg_parallel.mesh_axis] // cfg_parallel.factor
    assert cfg_parallel.mesh_shape[1 - cfg_parallel.mesh_axis] == mesh_shape[1 - cfg_parallel.mesh_axis]

    # validate tensor config
    assert tensor_parallel.mesh_axis in [0, 1]
    assert (
        tensor_parallel.mesh_shape[tensor_parallel.mesh_axis]
        == cfg_parallel.mesh_shape[tensor_parallel.mesh_axis] // tensor_parallel.factor
    )
    assert (
        tensor_parallel.mesh_shape[1 - tensor_parallel.mesh_axis]
        == cfg_parallel.mesh_shape[1 - tensor_parallel.mesh_axis]
    )

    # TODO: Be very careful with validation here.

    return DiTParallelConfig(
        cfg_parallel=cfg_parallel,
        tensor_parallel=tensor_parallel,
        # sequence_parallel=sequence_parallel,
        # ring_parallel=ring_parallel,
        # ulysses_parallel=ulysses_parallel
    )
