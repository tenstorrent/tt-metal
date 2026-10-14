# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Tensor creation helpers that preserve model mesh placement."""

import torch

import ttnn


def allocate_replicated_zeros(
    shape,
    *,
    device,
    dtype,
    layout=ttnn.TILE_LAYOUT,
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
):
    """Create zeros with the same topology as ``ReplicateTensorToMesh``.

    ``shape`` is the shape on each device. TTNN owns the fill implementation
    and its size-based dispatch policy. No model tensor cache file is needed.
    A ``None`` dtype retains the inference from the old ``torch.zeros`` input.
    """
    if dtype is None:
        dtype = {
            torch.float32: ttnn.float32,
            torch.bfloat16: ttnn.bfloat16,
            torch.float16: ttnn.bfloat16,
        }[torch.get_default_dtype()]
    tensor = ttnn.zeros(shape, device=device, dtype=dtype, layout=layout, memory_config=memory_config)
    # Device allocation uses the physical mesh dimensions by default. The
    # legacy replication mapper uses one distribution dimension, with physical
    # device coordinates in row-major order. Preserve that metadata for users
    # that combine this cache with tensors created through the mapper.
    tensor.update_tensor_topology(
        ttnn.TensorTopology(
            ttnn.MeshShape([device.get_num_devices()]),
            [ttnn.PlacementReplicate()],
            tensor.tensor_topology().mesh_coords(),
        )
    )
    return tensor
