# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn

OPS = ("sdpa_decode_bh",)


def _config_ok(program_config):
    if program_config is None:
        return False
    grid = program_config.compute_with_storage_grid_size
    return (
        (grid.x, grid.y) == (8, 8)
        and program_config.sub_core_grids is None
        and program_config.q_chunk_size == 0
        and program_config.k_chunk_size == 0
        and not program_config.exp_approx_mode
    )


def _q_rows_ok(q, rows):
    if not q.is_sharded():
        return False
    mem = q.memory_config()
    spec = mem.shard_spec
    if (
        mem.memory_layout != ttnn.TensorMemoryLayout.HEIGHT_SHARDED
        or mem.buffer_type != ttnn.BufferType.L1
        or spec is None
    ):
        return False
    if tuple(spec.shape) != (32, 128) or spec.orientation != ttnn.ShardOrientation.ROW_MAJOR:
        return False
    box = spec.grid.bounding_box()
    return spec.grid.num_cores() == rows and (box.start.x, box.start.y, box.end.x, box.end.y) == (0, 0, rows - 1, 0)


def covers(q, keys, program_config, sliding_window):
    return (
        sliding_window is None
        and _config_ok(program_config)
        and q.dtype == ttnn.bfloat16
        and tuple(q.padded_shape) == (1, 1, 32, 128)
        and _q_rows_ok(q, 1)
        and keys.dtype == ttnn.bfloat8_b
        and tuple(keys.padded_shape)[1:] == (8, 32, 128)
    )


def covers_rows(q, keys, program_config, sliding_window):
    shape = tuple(q.padded_shape)
    return (
        sliding_window is None
        and _config_ok(program_config)
        and q.dtype == ttnn.bfloat16
        and shape[0] == 1
        and 1 < shape[1] <= 8
        and shape[2:] == (32, 128)
        and _q_rows_ok(q, shape[1])
        and keys.dtype == ttnn.bfloat8_b
        and tuple(keys.padded_shape)[1:] == (8, 32, 128)
    )


def sdpa_decode_rows(q, keys, values, page_table, cur_pos, program_config, scale, compute_kernel_config):
    return ttnn.experimental.llama31_8b_v3.sdpa_decode_bh(
        q,
        keys,
        values,
        page_table,
        cur_pos,
        program_config,
        scale=scale,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        compute_kernel_config=compute_kernel_config,
        multi_pos=True,
    )


def sdpa_decode(q, keys, values, page_table, cur_pos, program_config, scale, compute_kernel_config):
    return ttnn.experimental.llama31_8b_v3.sdpa_decode_bh(
        q,
        keys,
        values,
        page_table,
        cur_pos,
        program_config,
        scale=scale,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        compute_kernel_config=compute_kernel_config,
    )
