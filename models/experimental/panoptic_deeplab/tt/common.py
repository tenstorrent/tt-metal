# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Common utilities for Panoptic DeepLab.

This module provides functions used across the Panoptic DeepLab model.
"""

from loguru import logger
import torch
import ttnn
import os

PDL_L1_SMALL_SIZE = 4 * 1024  # Minimum L1 small size for Panoptic DeepLab


def load_images(image_dir: str):
    """Load all images from a directory."""
    if not check_if_folder_contains_image(image_dir):
        logger.warning(f"No images found in directory: {image_dir}")
        return []

    supported_formats = (".png", ".jpg", ".jpeg", ".bmp")
    image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.lower().endswith(supported_formats)]
    return image_paths


def check_if_folder_contains_image(folder_path):
    """
    Check if the given folder contains any image files.

    Args:
        folder_path (str): Path to the folder to check.

    Returns:
        bool: True if the folder contains image files, False otherwise.
    """
    if not os.path.isdir(folder_path):
        return False

    image_extensions = (".png", ".jpg", ".jpeg")
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(image_extensions):
            return True
    return False


def get_panoptic_deeplab_resource_path(resource_type, model_location_generator=None, test_file_path=None):
    """
    Get the path to Panoptic DeepLab resources (weights or images).

    Args:
        resource_type: Either "weights" or "images"
        model_location_generator: Optional model location generator for CI
        test_file_path: Optional path to test file for local path determination

    Returns:
        str: Path to the requested resource
    """
    # All differences parametrized in one config
    config = {
        "weights": {
            "local_subdir": "weights",
            "filename": "model_final_bd324a.pkl",
            "ci_cache_path": "/tmp/ttnn_model_cache/model_weights/vision-models/panoptic_deeplab/model_final_bd324a.pkl",
            "ci_download_path": "vision-models/panoptic_deeplab",
            "ci_model_subdir": "",  # Use default
            "validate_func": None,  # No validation for weights
            "error_message": None,
        },
        "images": {
            "local_subdir": "images",
            "filename": None,  # Return directory, not specific file
            "ci_cache_path": "/tmp/ttnn_model_cache/model_weights/vision-models/panoptic_deeplab/images",
            "ci_download_path": "vision-models/panoptic_deeplab/images",
            "ci_model_subdir": "",  # Use default
            "validate_func": check_if_folder_contains_image,
            "error_message": "No images found in directory",
        },
    }[resource_type]

    # Get base directory (same logic for both)
    base_dir = _get_base_directory(test_file_path)

    # Environment-based path resolution
    if model_location_generator is None or "TT_GH_CI_INFRA" not in os.environ:
        # Local environment
        resource_path = os.path.join(base_dir, config["local_subdir"])
        if config["filename"]:
            resource_path = os.path.join(resource_path, config["filename"])

        # Validate local path immediately for images
        if config["validate_func"] and not config["validate_func"](resource_path):
            raise RuntimeError(f"{config['error_message']}: {resource_path}")
    else:
        # CI environment - check cache first
        cache_exists = os.path.exists(config["ci_cache_path"])
        cache_valid = config["validate_func"] is None or config["validate_func"](config["ci_cache_path"])
        if cache_exists and cache_valid:
            resource_path = config["ci_cache_path"]
            # Validate CI cached path immediately for images
        else:
            # Download using model location generator
            resource_path = model_location_generator(
                config["ci_download_path"], model_subdir=config["ci_model_subdir"], download_if_ci_v2=True
            )

            # Add filename if specified
            if config["filename"]:
                resource_path = os.path.join(resource_path, config["filename"])

            # Validate CI downloaded path immediately for images
            if config["validate_func"] and not config["validate_func"](resource_path):
                raise RuntimeError(f"{config['error_message']}: {resource_path}")

    return resource_path


def _get_base_directory(test_file_path):
    """Helper to get the panoptic_deeplab base directory."""
    if test_file_path is not None:
        current_dir = os.path.dirname(os.path.abspath(test_file_path))
        while not current_dir.endswith("panoptic_deeplab"):
            parent_dir = os.path.dirname(current_dir)
            if parent_dir == current_dir:
                break
            current_dir = parent_dir
    else:
        current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return current_dir


# Simple wrapper functions
def get_panoptic_deeplab_weights_path(model_location_generator=None, test_file_path=None):
    return get_panoptic_deeplab_resource_path("weights", model_location_generator, test_file_path)


def get_panoptic_deeplab_images_path(model_location_generator=None, test_file_path=None):
    return get_panoptic_deeplab_resource_path("images", model_location_generator, test_file_path)


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


def preprocess_nchw_input_tensor(device, torch_input):
    """
    Pad (if needed) and reshape to [1,1,N*H*W,C] for downstream convolution
    """

    assert len(torch_input.shape) == 4, f"Expected input tensor to be rank 4 (was {len(torch_input.shape)})"

    C = torch_input.shape[1]
    HW = torch_input.shape[2] * torch_input.shape[3]

    SHARD_WIDTH = 8

    # pad to min shard width
    torch_input = torch.nn.functional.pad(torch_input, (0, 0, 0, 0, 0, SHARD_WIDTH - C), mode="constant", value=0)
    # convert to channel last
    torch_input = torch_input.permute(0, 2, 3, 1)

    core_range_set = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(device.core_grid.x - 1, device.core_grid.y - 1))}
    )
    num_cores = device.core_grid.x * device.core_grid.y
    shard_height = (1 * HW + num_cores - 1) // num_cores  # = 26215

    sharded_memory_config = ttnn.create_sharded_memory_config_(
        shape=(shard_height, SHARD_WIDTH),
        core_grid=core_range_set,
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    return ttnn.from_torch(
        torch_input,
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=sharded_memory_config,
    )


def reshape_flattened_conv_output(tensor: ttnn.Tensor, batch_size: int = 1, layer_name: str = None) -> ttnn.Tensor:
    """
    Reshape a flattened convolution output from [1, 1, NHW, C] to [N, H, W, C].

    This function handles the case where convolutions output flattened format [1, 1, NHW, C]
    where NHW = N*H*W and W = 2*H. It calculates H and W from NHW and reshapes accordingly.

    Args:
        tensor: Input tensor that may be in flattened format [1, 1, NHW, C]
        batch_size: Batch size N (default: 1)
        layer_name: Optional layer name for debug logging

    Returns:
        Reshaped tensor [N, H, W, C] if reshape is valid, otherwise original tensor
    """
    if tensor.shape[1] == 1 and tensor.shape[2] > 1:
        # Flattened format: [1, 1, NHW, C] -> [N, H, W, C]
        NHW = tensor.shape[2]
        C = tensor.shape[3]
        # W = 2*H, so NHW = N*H*W = 1*H*2*H = 2*H^2, therefore H = sqrt(NHW/2)
        H_out = int((NHW // 2) ** 0.5)
        W_out = 2 * H_out
        if H_out * W_out == NHW:
            reshaped = ttnn.reshape(tensor, (batch_size, H_out, W_out, C))
            if layer_name:
                logger.debug(
                    f"{layer_name} reshaped output from [1, 1, {NHW}, {C}] to [{batch_size}, {H_out}, {W_out}, {C}]"
                )
            return reshaped
    return tensor
