# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Common utilities for Panoptic DeepLab.

This module provides functions used across the Panoptic DeepLab model.
"""

import torch
import ttnn
import os

PDL_L1_SMALL_SIZE = 37 * 1024  # Minimum L1 small size for Panoptic DeepLab


def get_panoptic_deeplab_weights_path(model_location_generator=None, test_file_path=None):
    """
    Get the complete path to the Panoptic DeepLab model weights file.

    Args:
        model_location_generator: Optional model location generator function for CI environments
        test_file_path: Optional path to the test file (used for local path determination)

    Returns:
        str: Complete path to the model weights file
    """
    # Determine weights path based on environment
    if model_location_generator is None or "TT_GH_CI_INFRA" not in os.environ:
        # Use local path
        if test_file_path is not None:
            current_dir = os.path.dirname(os.path.abspath(test_file_path))
            # Navigate up from tests/pcc/ to models/experimental/panoptic_deeplab/
            while not current_dir.endswith("panoptic_deeplab"):
                parent_dir = os.path.dirname(current_dir)
                if parent_dir == current_dir:  # Reached root directory
                    break
                current_dir = parent_dir
        else:
            # Fallback to current working directory approach
            current_dir = os.path.dirname(os.path.abspath(__file__))
            # Navigate up from tt/ to models/experimental/panoptic_deeplab/
            current_dir = os.path.dirname(current_dir)
        complete_weights_path = os.path.join(current_dir, "weights", "model_final_bd324a.pkl")
    else:
        # Check if weights already exist in CI v2 cache first
        cached_weights_path = (
            "/tmp/ttnn_model_cache/model_weights/vision-models/panoptic_deeplab/model_final_bd324a.pkl"
        )
        if os.path.exists(cached_weights_path):
            complete_weights_path = cached_weights_path
        else:
            # Use CI v2 model location generator to download
            complete_weights_path = (
                model_location_generator("vision-models/panoptic_deeplab", model_subdir="", download_if_ci_v2=True)
                / "model_final_bd324a.pkl"
            )
    return complete_weights_path


def get_panoptic_deeplab_config():
    """
    Get the standard configuration for Panoptic DeepLab model.

    Returns:
        dict: Dictionary containing model configuration parameters
    """
    return {
        "batch_size": 1,
        "num_classes": 19,
        "project_channels": [32, 64],
        "decoder_channels": [256, 256, 256],
        "sem_seg_head_channels": 256,
        "ins_embed_head_channels": 32,
        "common_stride": 4,
        "train_size": (512, 1024),
    }


def from_torch_fast(
    t: torch.Tensor,
    *,
    device: ttnn.Device | ttnn.MeshDevice | None = None,
    layout: ttnn.Layout | None = None,
    dtype: ttnn.DataType | None = None,
    memory_config: ttnn.MemoryConfig | None = None,
    to_host: bool = False,
    mesh_mapper: ttnn.TensorToMesh | None = None,
    # The argument shard_dim is a bit problematic. If set, it creates a mesh mapper with the given
    # device. But for a host tensor, the device is None, so a mesh mapper can not be created.
    shard_dim: int | None = None,
) -> ttnn.Tensor:
    assert shard_dim is None or device is not None, "shard_dim requires device"

    if isinstance(device, ttnn.MeshDevice):
        if shard_dim is not None:
            mesh_mapper = ttnn.ShardTensorToMesh(device, dim=shard_dim)
        if mesh_mapper is None:
            mesh_mapper = ttnn.ReplicateTensorToMesh(device)
    elif isinstance(device, ttnn.Device):
        mesh_mapper = None

    float32_in = t.dtype == torch.float32
    float32_out = dtype == ttnn.float32 or (dtype is None and float32_in)

    # ttnn.to_layout does not support changing the datatype or memory_config if the layout already matches. ttnn.clone
    # does not support changing the datatype if the input is not tiled. An option could be to tilize the input before
    # changing the datatype and then untilize again, but it was not tested if this would be faster than converting the
    # datatype on the host. Also ttnn.to_dtype does not support device tensors. Additionally, `ttnn.to_layout` is lossy
    # for float32.
    if device is None or layout is None or layout == ttnn.ROW_MAJOR_LAYOUT or (float32_in and float32_out):
        return ttnn.from_torch(
            t,
            device=None if to_host else device,
            layout=layout,
            dtype=dtype,
            memory_config=memory_config,
            mesh_mapper=mesh_mapper,
        )

    tensor = ttnn.from_torch(t, device=device, mesh_mapper=mesh_mapper)

    if tensor.shape[-2] == 32 and t.shape[-2] == 1:
        # Work around the fact that the shape is erroneously set to the padded shape under certain conditions.
        assert isinstance(device, ttnn.MeshDevice)
        assert dtype in (ttnn.bfloat4_b, ttnn.bfloat8_b)
        tensor = tensor.reshape(ttnn.Shape(t.shape))

    tensor = ttnn.to_layout(tensor, layout, dtype=dtype, memory_config=memory_config)

    if to_host:
        tensor = tensor.cpu()

    return tensor
