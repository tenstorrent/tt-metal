# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0


import ttnn


def mesh_scatter(
    tensor: ttnn.Tensor,
    mesh_shape: tuple | ttnn.MeshShape,
    scatter_idx: tuple[int | None, int | None],
    semaphores: tuple[object | None, object | None],
) -> None:
    from_row_scatter_idx, from_col_scatter_idx = scatter_idx
    num_mesh_rows, num_mesh_cols = mesh_shape
    per_col_semaphore, per_row_semaphore = semaphores

    assert (
        from_row_scatter_idx is None or 0 <= from_row_scatter_idx < num_mesh_rows
    ), f"Scatter row index must be within [{0}, {num_mesh_rows})"
    assert (
        from_col_scatter_idx is None or 0 <= from_col_scatter_idx < num_mesh_cols
    ), f"Scatter column index must be within [{0}, {num_mesh_cols})"
    assert (
        from_row_scatter_idx is None or per_col_semaphore is not None
    ), "Semaphore for row scatter must be provided if row scatter index is specified"
    assert (
        from_col_scatter_idx is None or per_row_semaphore is not None
    ), "Semaphore for column scatter must be provided if column scatter index is specified"

    old_memory_config = tensor.memory_config()
    old_tensor = tensor
    if not old_memory_config.interleaved:  # TODO: Remove this once point2point supports sharded tensors
        tensor = ttnn.to_memory_config(
            tensor,
            ttnn.L1_MEMORY_CONFIG if old_memory_config.buffer_type == ttnn.BufferType.L1 else ttnn.DRAM_MEMORY_CONFIG,
        )
    else:
        old_memory_config = None
        old_tensor = None

    # Scatter row if needed
    for to_row_scatter_idx in range(0 if from_row_scatter_idx is None else num_mesh_rows):
        if to_row_scatter_idx == from_row_scatter_idx:
            continue

        for column_scatter_idx in range(num_mesh_cols):
            ttnn.point_to_point(
                tensor,
                ttnn.MeshCoordinate(to_row_scatter_idx, column_scatter_idx),
                ttnn.MeshCoordinate(from_row_scatter_idx, column_scatter_idx),
                ttnn.Topology.Linear,
                per_col_semaphore,
                optional_output_tensor=tensor,
            )

    # Scatter column if needed
    for to_col_scatter_idx in range(0 if from_col_scatter_idx is None else num_mesh_cols):
        if to_col_scatter_idx == from_col_scatter_idx:
            continue

        for row_scatter_idx in range(num_mesh_rows):
            ttnn.point_to_point(
                tensor,
                ttnn.MeshCoordinate(to_col_scatter_idx, row_scatter_idx),
                ttnn.MeshCoordinate(from_col_scatter_idx, row_scatter_idx),
                ttnn.Topology.Linear,
                per_row_semaphore,
                optional_output_tensor=tensor,
            )

    if old_memory_config is not None:  # TODO: Remove this once point2point supports sharded tensors
        reconfig_tensor = ttnn.to_memory_config(tensor, old_memory_config)
        ttnn.deallocate(tensor)

        ttnn.copy(reconfig_tensor, old_tensor)
        ttnn.deallocate(reconfig_tensor)
