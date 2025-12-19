# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Common utilities for Panoptic DeepLab.

This module provides functions used across the Panoptic DeepLab model,
including model creation, pipeline setup, input/output processing, and validation.
"""

from typing import Callable, Tuple, List, Optional, Union
from loguru import logger
import torch
import ttnn
import os
import numpy as np

from models.experimental.panoptic_deeplab.reference.pytorch_model import (
    PANOPTIC_DEEPLAB,
    DEEPLAB_V3_PLUS,
    PytorchPanopticDeepLab,
)
from models.experimental.panoptic_deeplab.tt.model_preprocessing import (
    create_panoptic_deeplab_parameters,
    fuse_conv_bn_parameters,
)
from models.experimental.panoptic_deeplab.tt.model_configs import ModelOptimisations

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


def convert_ttnn_output_to_numpy(
    ttnn_output: ttnn.Tensor,
    output_channels: Optional[int] = None,
) -> np.ndarray:
    """
    Convert TTNN output tensor to numpy format for visualization.

    Converts from NHWC device tensor to HWC numpy array, handling channel slicing if needed.

    Args:
        ttnn_output: TTNN output tensor (can be device or host tensor)
        output_channels: Optional number of channels to slice (removes padding)

    Returns:
        Numpy array in HWC format [H, W, C]
    """
    # Convert to torch
    ttnn_output_torch = ttnn.to_torch(ttnn_output)

    # Slice channels if needed (remove padding)
    if output_channels is not None:
        ttnn_output_torch = ttnn_output_torch[:, :output_channels, :, :]

    # Convert to numpy in HWC format for visualization
    # Shape: [1, C, H, W] -> [H, W, C]
    numpy_output = ttnn_output_torch.float().squeeze(0).permute(1, 2, 0).numpy()

    return numpy_output


def convert_pytorch_outputs_to_numpy(
    pytorch_semantic_logits: torch.Tensor,
    pytorch_center_logits: Optional[torch.Tensor] = None,
    pytorch_offset_logits: Optional[torch.Tensor] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Convert PyTorch model outputs to numpy format for visualization.

    Args:
        pytorch_semantic_logits: Semantic segmentation logits [1, C, H, W]
        pytorch_center_logits: Optional center heatmap logits [1, C, H, W]
        pytorch_offset_logits: Optional offset logits [1, C, H, W]

    Returns:
        Tuple of (semantic_np, center_np, offset_np) in HWC format
    """
    semantic_np = pytorch_semantic_logits.float().squeeze(0).permute(1, 2, 0).numpy()
    center_np = (
        pytorch_center_logits.float().squeeze(0).permute(1, 2, 0).numpy() if pytorch_center_logits is not None else None
    )
    offset_np = (
        pytorch_offset_logits.float().squeeze(0).permute(1, 2, 0).numpy() if pytorch_offset_logits is not None else None
    )
    return semantic_np, center_np, offset_np


# ============================================================================
# Model Creation Functions
# ============================================================================


def create_pytorch_model(
    weights_path: str,
    model_category: str,
    target_size: Tuple[int, int],
    num_classes: int,
    common_stride: int,
    project_channels: List[int],
    decoder_channels: List[int],
    sem_seg_head_channels: int,
    ins_embed_head_channels: int,
) -> PytorchPanopticDeepLab:
    """
    Create and load a PyTorch Panoptic DeepLab model.

    Args:
        weights_path: Path to model weights (.pkl file)
        model_category: Model category (PANOPTIC_DEEPLAB or DEEPLAB_V3_PLUS)
        target_size: Input size as (height, width)
        num_classes: Number of classes
        common_stride: Common stride
        project_channels: Project channels
        decoder_channels: Decoder channels
        sem_seg_head_channels: Semantic segmentation head channels
        ins_embed_head_channels: Instance embedding head channels

    Returns:
        Loaded PyTorch model
    """
    logger.info("Loading PyTorch model...")
    pytorch_model = PytorchPanopticDeepLab(
        num_classes=num_classes,
        common_stride=common_stride,
        project_channels=project_channels,
        decoder_channels=decoder_channels,
        sem_seg_head_channels=sem_seg_head_channels,
        ins_embed_head_channels=ins_embed_head_channels,
        train_size=target_size,
        weights_path=weights_path,
        model_category=model_category,
    )
    pytorch_model = pytorch_model.to(dtype=torch.bfloat16)
    pytorch_model.eval()
    return pytorch_model


def create_ttnn_model(
    device: ttnn.Device,
    pytorch_model: PytorchPanopticDeepLab,
    target_size: Tuple[int, int],
    batch_size: int,
    model_category: str,
    num_classes: int,
    common_stride: int,
    project_channels: List[int],
    decoder_channels: List[int],
    sem_seg_head_channels: int,
    ins_embed_head_channels: int,
    model_configs: Optional[ModelOptimisations] = None,
):
    """
    Create a TTNN Panoptic DeepLab model from PyTorch model.

    Args:
        device: TTNN device
        pytorch_model: PyTorch model to convert
        target_size: Input size as (height, width)
        batch_size: Batch size
        model_category: Model category (PANOPTIC_DEEPLAB or DEEPLAB_V3_PLUS)
        num_classes: Number of classes
        common_stride: Common stride
        project_channels: Project channels
        decoder_channels: Decoder channels
        sem_seg_head_channels: Semantic segmentation head channels
        ins_embed_head_channels: Instance embedding head channels
        model_configs: Optional model optimization configs

    Returns:
        TTNN model
    """
    # Import here to avoid circular import
    from models.experimental.panoptic_deeplab.tt.tt_model import TtPanopticDeepLab

    # Create TTNN parameters
    logger.info("Creating TTNN parameters...")
    ttnn_parameters = create_panoptic_deeplab_parameters(
        pytorch_model,
        device,
        input_height=int(target_size[0]),
        input_width=int(target_size[1]),
        batch_size=batch_size,
    )

    # Apply Conv+BatchNorm fusion
    logger.info("Applying Conv+BatchNorm fusion to parameters...")
    fused_parameters = fuse_conv_bn_parameters(ttnn_parameters, eps=1e-5)

    # Create model configurations if not provided
    if model_configs is None:
        model_configs = ModelOptimisations(
            conv_act_dtype=ttnn.bfloat8_b,
            conv_w_dtype=ttnn.bfloat8_b,
        )
        model_configs.setup_resnet_backbone()
        model_configs.setup_aspp()
        model_configs.setup_decoder()
        model_configs.setup_heads()

    # Create TTNN model
    ttnn_model = TtPanopticDeepLab(
        device=device,
        parameters=fused_parameters,
        num_classes=num_classes,
        common_stride=common_stride,
        project_channels=project_channels,
        decoder_channels=decoder_channels,
        sem_seg_head_channels=sem_seg_head_channels,
        ins_embed_head_channels=ins_embed_head_channels,
        train_size=target_size,
        model_configs=model_configs,
        model_category=model_category,
    )

    return ttnn_model


# ============================================================================
# Pipeline/Executor Functions
# ============================================================================


def create_model_wrapper(ttnn_model) -> Callable:
    """
    Create a model wrapper function for use with pipeline/executor.

    The wrapper takes an L1 input tensor and calls the model's forward method.
    The executor expects the model to accept L1 tensors and return device tensors.

    Args:
        ttnn_model: The TtPanopticDeepLab model instance

    Returns:
        A callable function that takes an L1 input tensor and returns model outputs
    """

    def model_forward(l1_input_tensor: ttnn.Tensor):
        """
        Forward pass wrapper for pipeline executor.

        Args:
            l1_input_tensor: Input tensor in L1 memory (expected by executor)

        Returns:
            Tuple of (semantic_logits, center_heatmap, offset_map)
            For DEEPLAB_V3_PLUS, returns only semantic_logits (not a tuple with None)
        """
        assert l1_input_tensor.storage_type() == ttnn.StorageType.DEVICE, "Model expects input tensor to be on device"
        assert (
            l1_input_tensor.memory_config().buffer_type == ttnn.BufferType.L1
        ), "Model expects input tensor to be in L1"

        # Call model forward
        semantic_logits, center_heatmap, offset_map, _ = ttnn_model.forward(l1_input_tensor, return_features=False)

        # For DEEPLAB_V3_PLUS, center_heatmap and offset_map are None
        # Return only semantic_logits to avoid None handling issues in executor
        if ttnn_model.model_category == DEEPLAB_V3_PLUS:
            return semantic_logits
        else:
            # Return as tuple for PANOPTIC_DEEPLAB
            return (semantic_logits, center_heatmap, offset_map)

    return model_forward


# ============================================================================
# Input Processing Functions
# ============================================================================


def create_host_input_tensors_from_torch(
    device: ttnn.Device,
    torch_inputs: List[torch.Tensor],
) -> Tuple[List[ttnn.Tensor], Optional[ttnn.MemoryConfig], Optional[ttnn.MemoryConfig]]:
    """
    Create host input tensors from PyTorch tensors.

    Preprocesses input tensors directly in Python:
    - Pads channels to SHARD_WIDTH (8)
    - Converts to channel last (NHWC)
    - Creates host tensors directly without device preprocessing

    Args:
        device: TTNN device
        torch_inputs: List of PyTorch tensors in NCHW format [N, C, H, W]

    Returns:
        Tuple of (list of host input tensors, dram_memory_config, l1_memory_config)
        - dram_memory_config: None (not used)
        - l1_memory_config: Sharded L1 with full grid
    """
    host_inputs = []
    dram_memory_config = None
    l1_memory_config = None

    SHARD_WIDTH = 8

    for i, torch_input in enumerate(torch_inputs):
        assert len(torch_input.shape) == 4, f"Expected input tensor to be rank 4 (was {len(torch_input.shape)})"

        C = torch_input.shape[1]
        H = torch_input.shape[2]
        W = torch_input.shape[3]
        HW = H * W

        # Pad channels to SHARD_WIDTH (8) if needed
        # Padding format: (pad_left, pad_right, pad_top, pad_bottom, pad_front, pad_back)
        # For channel dimension (dim=1), we pad at the end: pad_front=0, pad_back=SHARD_WIDTH-C
        torch_input = torch.nn.functional.pad(torch_input, (0, 0, 0, 0, 0, SHARD_WIDTH - C), mode="constant", value=0)

        # Convert to channel last (NHWC): [1, H, W, SHARD_WIDTH]
        torch_input = torch_input.permute(0, 2, 3, 1)

        # Create memory configs on first iteration
        if i == 0:
            # CustomTracedModelExecutor doesn't use DRAM memory config - it transfers directly to L1
            dram_memory_config = None

            # Create L1 sharded memory config (height sharding across full grid)
            core_range_set = ttnn.CoreRangeSet(
                {
                    ttnn.CoreRange(
                        ttnn.CoreCoord(0, 0),
                        ttnn.CoreCoord(device.core_grid.x - 1, device.core_grid.y - 1),
                    )
                }
            )
            num_cores = device.core_grid.x * device.core_grid.y
            shard_height = (1 * HW + num_cores - 1) // num_cores

            sharded_memory_config = ttnn.create_sharded_memory_config_(
                shape=(shard_height, SHARD_WIDTH),
                core_grid=core_range_set,
                strategy=ttnn.ShardStrategy.HEIGHT,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
            )

            # Use the L1 sharding config for the pipeline
            l1_memory_config = ttnn.MemoryConfig(
                sharded_memory_config.memory_layout,
                ttnn.BufferType.L1,
                sharded_memory_config.shard_spec,
            )

        # Convert to TTNN host tensor (not on device)
        # Use ROW_MAJOR_LAYOUT and bfloat16 dtype to match the original preprocessing
        host_input = ttnn.from_torch(
            torch_input,
            device=None,  # Host tensor, not on device
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        host_inputs.append(host_input)

    return host_inputs, dram_memory_config, l1_memory_config


def create_host_input_tensors_random(
    device: ttnn.Device,
    batch_size: int,
    input_height: int,
    input_width: int,
    num_inputs: int,
) -> Tuple[List[ttnn.Tensor], Optional[ttnn.MemoryConfig], Optional[ttnn.MemoryConfig]]:
    """
    Create random host input tensors for Panoptic DeepLab (for testing).

    Args:
        device: TTNN device
        batch_size: Batch size (should be 1 for Panoptic DeepLab)
        input_height: Input image height
        input_width: Input image width
        num_inputs: Number of input tensors to create

    Returns:
        Tuple of (list of host input tensors, dram_memory_config, l1_memory_config)
    """
    torch_inputs = []
    for _ in range(num_inputs):
        # Create random input in NCHW format
        torch_input = torch.randn(batch_size, 3, input_height, input_width, dtype=torch.bfloat16)
        torch_inputs.append(torch_input)

    return create_host_input_tensors_from_torch(device, torch_inputs)


def convert_host_input_to_torch_for_reference(host_input: ttnn.Tensor) -> torch.Tensor:
    """
    Convert host input tensor back to PyTorch format for reference model.

    Args:
        host_input: Host input tensor in NHWC format [1, H, W, C] where C=8

    Returns:
        PyTorch tensor in NCHW format [1, 3, H, W] (padding removed)
    """
    # Convert host input back to torch for reference
    torch_input = ttnn.to_torch(host_input)
    # Input is in NHWC format [1, H, W, C] where C=8 (3 original + 5 padding)
    # We need to remove padding and convert to NCHW for PyTorch model
    assert torch_input.shape[0] == 1, f"Expected batch size 1, got {torch_input.shape[0]}"
    # Remove padding: take only first 3 channels
    torch_input = torch_input[:, :, :, :3]  # [1, H, W, 3]
    # Convert NHWC -> NCHW
    torch_input = torch_input.permute(0, 3, 1, 2)  # [1, 3, H, W]
    return torch_input


# ============================================================================
# Reference Output Generation
# ============================================================================


def generate_reference_outputs(
    pytorch_model: PytorchPanopticDeepLab,
    host_inputs: List[ttnn.Tensor],
) -> List[Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]]:
    """
    Generate reference outputs from PyTorch model for given host inputs.

    Args:
        pytorch_model: PyTorch model
        host_inputs: List of host input tensors

    Returns:
        List of tuples (semantic_logits, center_heatmap, offset_map)
        For DEEPLAB_V3_PLUS, center_heatmap and offset_map are None
    """
    logger.info("Generating reference outputs from PyTorch model...")
    reference_outputs = []
    for host_input in host_inputs:
        torch_input = convert_host_input_to_torch_for_reference(host_input)
        with torch.no_grad():
            pytorch_semantic, pytorch_center, pytorch_offset, _ = pytorch_model.forward(torch_input)
        reference_outputs.append((pytorch_semantic, pytorch_center, pytorch_offset))
    return reference_outputs


# ============================================================================
# Output Processing Functions
# ============================================================================


def process_ttnn_outputs_for_visualization(
    ttnn_semantic_logits: ttnn.Tensor,
    ttnn_model,
    ttnn_center_logits: Optional[ttnn.Tensor] = None,
    ttnn_offset_logits: Optional[ttnn.Tensor] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Process TTNN model outputs for visualization.

    Converts TTNN outputs to numpy format, handling channel slicing and format conversion.

    Args:
        ttnn_semantic_logits: Semantic segmentation logits
        ttnn_model: TTNN model instance (for getting output channel counts)
        ttnn_center_logits: Optional center heatmap logits (for PANOPTIC_DEEPLAB)
        ttnn_offset_logits: Optional offset map logits (for PANOPTIC_DEEPLAB)

    Returns:
        Tuple of (semantic_np, center_np, offset_np) in HWC format
        For DEEPLAB_V3_PLUS, center_np and offset_np are None
    """
    # Process semantic output
    semantic_original_channels = ttnn_model.semantic_head.get_output_channels_for_slicing()
    semantic_np = convert_ttnn_output_to_numpy(ttnn_semantic_logits, semantic_original_channels)

    # Process instance outputs (only for PANOPTIC_DEEPLAB)
    center_np = None
    offset_np = None
    if (
        ttnn_model.model_category == PANOPTIC_DEEPLAB
        and ttnn_center_logits is not None
        and ttnn_offset_logits is not None
    ):
        center_original_channels = ttnn_model.instance_head.get_center_output_channels_for_slicing()
        offset_original_channels = ttnn_model.instance_head.get_offset_output_channels_for_slicing()

        center_np = convert_ttnn_output_to_numpy(ttnn_center_logits, center_original_channels)
        offset_np = convert_ttnn_output_to_numpy(ttnn_offset_logits, offset_original_channels)

    return semantic_np, center_np, offset_np


def load_and_preprocess_image_for_visualization(
    image_path: str,
    target_size: Tuple[int, int],
) -> np.ndarray:
    """
    Load and preprocess an image for visualization.

    Args:
        image_path: Path to input image
        target_size: Target size as (height, width)

    Returns:
        Preprocessed image as numpy array in RGB format [H, W, 3]
    """
    import cv2

    original_image = cv2.imread(image_path)
    if original_image is None:
        raise ValueError(f"Could not load image from {image_path}")

    # Convert BGR to RGB
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

    # Resize to target size
    original_image = cv2.resize(original_image, (int(target_size[1]), int(target_size[0])))

    return original_image


def extract_outputs_from_pipeline_result(
    ttnn_output: Union[ttnn.Tensor, Tuple, List],
    model_category: str,
    output_index: int = 0,
) -> Tuple[ttnn.Tensor, Optional[ttnn.Tensor], Optional[ttnn.Tensor]]:
    """
    Extract semantic, center, and offset outputs from pipeline result.

    Handles different output formats based on model category:
    - DEEPLAB_V3_PLUS: Single tensor (semantic only)
    - PANOPTIC_DEEPLAB: Tuple/list of 3 tensors (semantic, center, offset)

    Args:
        ttnn_output: Output from pipeline (can be tensor, tuple, or list)
        model_category: Model category (PANOPTIC_DEEPLAB or DEEPLAB_V3_PLUS)
        output_index: Index for error messages (for validation)

    Returns:
        Tuple of (semantic_logits, center_logits, offset_logits)
        For DEEPLAB_V3_PLUS, center_logits and offset_logits are None
    """
    if model_category == DEEPLAB_V3_PLUS:
        ttnn_semantic = ttnn_output
        assert (
            ttnn_semantic.storage_type() == ttnn.StorageType.HOST
        ), f"Semantic output {output_index} should be on host"
        return ttnn_semantic, None, None
    else:
        # PANOPTIC_DEEPLAB
        assert isinstance(
            ttnn_output, (tuple, list)
        ), f"Output {output_index} should be a tuple or list for PANOPTIC_DEEPLAB, got {type(ttnn_output)}"
        assert len(ttnn_output) == 3, f"Output {output_index} should have 3 elements, got {len(ttnn_output)}"
        ttnn_semantic, ttnn_center, ttnn_offset = ttnn_output

        assert (
            ttnn_semantic.storage_type() == ttnn.StorageType.HOST
        ), f"Semantic output {output_index} should be on host"
        assert ttnn_center.storage_type() == ttnn.StorageType.HOST, f"Center output {output_index} should be on host"
        assert ttnn_offset.storage_type() == ttnn.StorageType.HOST, f"Offset output {output_index} should be on host"

        return ttnn_semantic, ttnn_center, ttnn_offset


def create_visualization_from_outputs(
    semantic_np: np.ndarray,
    original_image: np.ndarray,
    model_category: str,
    center_np: Optional[np.ndarray] = None,
    offset_np: Optional[np.ndarray] = None,
    center_threshold: float = 0.05,
) -> Tuple[np.ndarray, dict]:
    """
    Create visualization from processed outputs.

    Args:
        semantic_np: Semantic segmentation predictions in HWC format
        original_image: Original input image in RGB format
        model_category: Model category (PANOPTIC_DEEPLAB or DEEPLAB_V3_PLUS)
        center_np: Optional center heatmap predictions (for PANOPTIC_DEEPLAB)
        offset_np: Optional offset predictions (for PANOPTIC_DEEPLAB)
        center_threshold: Center threshold for panoptic segmentation

    Returns:
        Tuple of (visualization_image, panoptic_info)
    """
    from models.experimental.panoptic_deeplab.demo.demo_utils import (
        create_panoptic_visualization,
        create_deeplab_v3plus_visualization,
    )

    if model_category == PANOPTIC_DEEPLAB:
        panoptic_vis, panoptic_info = create_panoptic_visualization(
            semantic_np,
            center_np,
            offset_np,
            original_image,
            center_threshold=center_threshold,
            score_threshold=center_threshold,
            stuff_area=1,
            top_k=1000,
            nms_kernel=11,
        )
    else:
        panoptic_vis, panoptic_info = create_deeplab_v3plus_visualization(
            semantic_np,
            original_image=original_image,
        )

    return panoptic_vis, panoptic_info


# ============================================================================
# Validation Functions
# ============================================================================


def validate_outputs_with_pcc(
    pytorch_semantic: torch.Tensor,
    ttnn_semantic: ttnn.Tensor,
    ttnn_model,
    model_category: str,
    pytorch_center: Optional[torch.Tensor] = None,
    ttnn_center: Optional[ttnn.Tensor] = None,
    pytorch_offset: Optional[torch.Tensor] = None,
    ttnn_offset: Optional[ttnn.Tensor] = None,
    layer_name_prefix: str = "",
    semantic_exp_pcc: float = 0.951,
    semantic_exp_abs_err: float = 1.952,
    semantic_exp_rel_err: float = 0.180,
    center_exp_pcc: float = 0.949,
    center_exp_abs_err: float = 0.008,
    center_exp_rel_err: float = 22.349,
    offset_exp_pcc: float = 0.966,
    offset_exp_abs_err: float = 12.160,
    offset_exp_rel_err: float = 2.405,
) -> List[bool]:
    """
    Validate TTNN outputs against PyTorch reference using PCC and relative error checks.

    Args:
        pytorch_semantic: PyTorch semantic logits
        ttnn_semantic: TTNN semantic logits
        ttnn_model: TTNN model instance
        model_category: Model category (PANOPTIC_DEEPLAB or DEEPLAB_V3_PLUS)
        pytorch_center: Optional PyTorch center logits
        ttnn_center: Optional TTNN center logits
        pytorch_offset: Optional PyTorch offset logits
        ttnn_offset: Optional TTNN offset logits
        layer_name_prefix: Prefix for layer names in validation messages
        semantic_exp_pcc: Expected PCC for semantic output
        semantic_exp_abs_err: Expected absolute error for semantic output
        semantic_exp_rel_err: Expected relative error for semantic output
        center_exp_pcc: Expected PCC for center output
        center_exp_abs_err: Expected absolute error for center output
        center_exp_rel_err: Expected relative error for center output
        offset_exp_pcc: Expected PCC for offset output
        offset_exp_abs_err: Expected absolute error for offset output
        offset_exp_rel_err: Expected relative error for offset output

    Returns:
        List of boolean results for each validation check
    """
    from models.experimental.panoptic_deeplab.tests.pcc.common import check_ttnn_output

    all_passed = []

    # Check semantic output
    passed = check_ttnn_output(
        layer_name=f"{layer_name_prefix}semantic",
        pytorch_output=pytorch_semantic,
        ttnn_output=ttnn_semantic,
        to_channel_first=False,
        output_channels=ttnn_model.semantic_head.get_output_channels_for_slicing(),
        exp_pcc=semantic_exp_pcc,
        exp_abs_err=semantic_exp_abs_err,
        exp_rel_err=semantic_exp_rel_err,
    )
    all_passed.append(passed)

    # Check instance outputs (only for PANOPTIC_DEEPLAB)
    if model_category == PANOPTIC_DEEPLAB:
        if pytorch_center is not None and ttnn_center is not None:
            passed_center = check_ttnn_output(
                layer_name=f"{layer_name_prefix}center",
                pytorch_output=pytorch_center,
                ttnn_output=ttnn_center,
                to_channel_first=False,
                output_channels=ttnn_model.instance_head.get_center_output_channels_for_slicing(),
                exp_pcc=center_exp_pcc,
                exp_abs_err=center_exp_abs_err,
                exp_rel_err=center_exp_rel_err,
            )
            all_passed.append(passed_center)

        if pytorch_offset is not None and ttnn_offset is not None:
            passed_offset = check_ttnn_output(
                layer_name=f"{layer_name_prefix}offset",
                pytorch_output=pytorch_offset,
                ttnn_output=ttnn_offset,
                to_channel_first=False,
                output_channels=ttnn_model.instance_head.get_offset_output_channels_for_slicing(),
                exp_pcc=offset_exp_pcc,
                exp_abs_err=offset_exp_abs_err,
                exp_rel_err=offset_exp_rel_err,
            )
            all_passed.append(passed_offset)

    return all_passed
