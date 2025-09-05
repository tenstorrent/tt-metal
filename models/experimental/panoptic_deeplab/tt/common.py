# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Common utilities for ResNet weight management in Panoptic DeepLab.

This module provides functions to create state dictionaries for ResNet models:
- Random weight initialization functions (create_random_*_state_dict)
- Real weight loading from R-52.pkl file (create_real_resnet_state_dict)

The real weights are loaded from the R-52.pkl file which contains trained ResNet-52
weights that can be used instead of random initialization for better performance.
"""

import torch
import torch.nn.init
import pickle
import os
import ttnn


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


def create_random_stem_state_dict():
    """Create random state dict for DeepLabStem."""
    state_dict = {}

    # Conv1: 3 -> 64 channels, 3x3 kernel, stride=2
    conv1_weight = torch.empty(64, 3, 3, 3, dtype=torch.bfloat16)
    torch.nn.init.kaiming_normal_(conv1_weight, mode="fan_out", nonlinearity="relu")
    state_dict["conv1.weight"] = conv1_weight
    state_dict["conv1.bias"] = torch.zeros(64, dtype=torch.bfloat16)
    state_dict["conv1.norm.weight"] = torch.randn(64, dtype=torch.bfloat16)
    state_dict["conv1.norm.bias"] = torch.randn(64, dtype=torch.bfloat16)
    state_dict["conv1.norm.running_mean"] = torch.randn(64, dtype=torch.bfloat16)
    state_dict["conv1.norm.running_var"] = torch.abs(torch.randn(64, dtype=torch.bfloat16)) + 1e-5

    # Conv2: 64 -> 64 channels, 3x3 kernel, stride=1
    conv2_weight = torch.empty(64, 64, 3, 3, dtype=torch.bfloat16)
    torch.nn.init.kaiming_normal_(conv2_weight, mode="fan_out", nonlinearity="relu")
    state_dict["conv2.weight"] = conv2_weight
    state_dict["conv2.bias"] = torch.zeros(64, dtype=torch.bfloat16)
    state_dict["conv2.norm.weight"] = torch.randn(64, dtype=torch.bfloat16)
    state_dict["conv2.norm.bias"] = torch.randn(64, dtype=torch.bfloat16)
    state_dict["conv2.norm.running_mean"] = torch.randn(64, dtype=torch.bfloat16)
    state_dict["conv2.norm.running_var"] = torch.abs(torch.randn(64, dtype=torch.bfloat16)) + 1e-5

    # Conv3: 64 -> 128 channels, 3x3 kernel, stride=1
    conv3_weight = torch.empty(128, 64, 3, 3, dtype=torch.bfloat16)
    torch.nn.init.kaiming_normal_(conv3_weight, mode="fan_out", nonlinearity="relu")
    state_dict["conv3.weight"] = conv3_weight
    state_dict["conv3.bias"] = torch.zeros(128, dtype=torch.bfloat16)
    state_dict["conv3.norm.weight"] = torch.randn(128, dtype=torch.bfloat16)
    state_dict["conv3.norm.bias"] = torch.randn(128, dtype=torch.bfloat16)
    state_dict["conv3.norm.running_mean"] = torch.randn(128, dtype=torch.bfloat16)
    state_dict["conv3.norm.running_var"] = torch.abs(torch.randn(128, dtype=torch.bfloat16)) + 1e-5

    return state_dict


def create_random_bottleneck_state_dict(in_channels, bottleneck_channels, out_channels, has_shortcut=False):
    """Create random state dict for BottleneckBlock."""
    state_dict = {}

    # Conv1 weights (1x1 conv)
    conv1_weight = torch.empty(bottleneck_channels, in_channels, 1, 1, dtype=torch.bfloat16)
    torch.nn.init.kaiming_normal_(conv1_weight, mode="fan_out", nonlinearity="relu")
    state_dict["conv1.weight"] = conv1_weight
    state_dict["conv1.bias"] = torch.zeros(bottleneck_channels, dtype=torch.bfloat16)
    state_dict["conv1.norm.weight"] = torch.randn(bottleneck_channels, dtype=torch.bfloat16)
    state_dict["conv1.norm.bias"] = torch.randn(bottleneck_channels, dtype=torch.bfloat16)
    state_dict["conv1.norm.running_mean"] = torch.randn(bottleneck_channels, dtype=torch.bfloat16)
    state_dict["conv1.norm.running_var"] = torch.abs(torch.randn(bottleneck_channels, dtype=torch.bfloat16)) + 1e-5

    # Conv2 weights (3x3 conv)
    conv2_weight = torch.empty(bottleneck_channels, bottleneck_channels, 3, 3, dtype=torch.bfloat16)
    torch.nn.init.kaiming_normal_(conv2_weight, mode="fan_out", nonlinearity="relu")
    state_dict["conv2.weight"] = conv2_weight
    state_dict["conv2.bias"] = torch.zeros(bottleneck_channels, dtype=torch.bfloat16)
    state_dict["conv2.norm.weight"] = torch.randn(bottleneck_channels, dtype=torch.bfloat16)
    state_dict["conv2.norm.bias"] = torch.randn(bottleneck_channels, dtype=torch.bfloat16)
    state_dict["conv2.norm.running_mean"] = torch.randn(bottleneck_channels, dtype=torch.bfloat16)
    state_dict["conv2.norm.running_var"] = torch.abs(torch.randn(bottleneck_channels, dtype=torch.bfloat16)) + 1e-5

    # Conv3 weights (1x1 conv)
    conv3_weight = torch.empty(out_channels, bottleneck_channels, 1, 1, dtype=torch.bfloat16)
    torch.nn.init.kaiming_normal_(conv3_weight, mode="fan_out", nonlinearity="relu")
    state_dict["conv3.weight"] = conv3_weight
    state_dict["conv3.bias"] = torch.zeros(out_channels, dtype=torch.bfloat16)
    state_dict["conv3.norm.weight"] = torch.randn(out_channels, dtype=torch.bfloat16)
    state_dict["conv3.norm.bias"] = torch.randn(out_channels, dtype=torch.bfloat16)
    state_dict["conv3.norm.running_mean"] = torch.randn(out_channels, dtype=torch.bfloat16)
    state_dict["conv3.norm.running_var"] = torch.abs(torch.randn(out_channels, dtype=torch.bfloat16)) + 1e-5

    # Shortcut weights (if needed)
    if has_shortcut:
        shortcut_weight = torch.empty(out_channels, in_channels, 1, 1, dtype=torch.bfloat16)
        torch.nn.init.kaiming_normal_(shortcut_weight, mode="fan_out", nonlinearity="relu")
        state_dict["shortcut.weight"] = shortcut_weight
        state_dict["shortcut.bias"] = torch.zeros(out_channels, dtype=torch.bfloat16)
        state_dict["shortcut.norm.weight"] = torch.randn(out_channels, dtype=torch.bfloat16)
        state_dict["shortcut.norm.bias"] = torch.randn(out_channels, dtype=torch.bfloat16)
        state_dict["shortcut.norm.running_mean"] = torch.randn(out_channels, dtype=torch.bfloat16)
        state_dict["shortcut.norm.running_var"] = torch.abs(torch.randn(out_channels, dtype=torch.bfloat16)) + 1e-5

    return state_dict


def create_full_resnet_state_dict():
    """Create complete random state dict for ResNet with all layers."""
    state_dict = {}

    # Stem layers
    stem_state_dict = create_random_stem_state_dict()
    for key, value in stem_state_dict.items():
        state_dict[f"stem.{key}"] = value

    # res2 (3 blocks): 128->256 channels
    for i, (in_channels, has_shortcut) in enumerate([(128, True), (256, False), (256, False)]):
        bottleneck_state_dict = create_random_bottleneck_state_dict(in_channels, 64, 256, has_shortcut)
        for key, value in bottleneck_state_dict.items():
            state_dict[f"res2.{i}.{key}"] = value

    # res3 (4 blocks): 256->512 channels
    for i, (in_channels, has_shortcut) in enumerate([(256, True), (512, False), (512, False), (512, False)]):
        bottleneck_state_dict = create_random_bottleneck_state_dict(in_channels, 128, 512, has_shortcut)
        for key, value in bottleneck_state_dict.items():
            state_dict[f"res3.{i}.{key}"] = value

    # res4 (6 blocks): 512->1024 channels
    for i, (in_channels, has_shortcut) in enumerate(
        [(512, True), (1024, False), (1024, False), (1024, False), (1024, False), (1024, False)]
    ):
        bottleneck_state_dict = create_random_bottleneck_state_dict(in_channels, 256, 1024, has_shortcut)
        for key, value in bottleneck_state_dict.items():
            state_dict[f"res4.{i}.{key}"] = value

    # res5 (3 blocks): 1024->2048 channels
    for i, (in_channels, has_shortcut) in enumerate([(1024, True), (2048, False), (2048, False)]):
        bottleneck_state_dict = create_random_bottleneck_state_dict(in_channels, 512, 2048, has_shortcut)
        for key, value in bottleneck_state_dict.items():
            state_dict[f"res5.{i}.{key}"] = value

    return state_dict


def load_resnet_weights_from_pickle(pickle_path: str = None) -> dict:
    """
    Load real ResNet weights from R-52.pkl file.

    Args:
        pickle_path: Path to the pickle file. If None, uses default path relative to this file.

    Returns:
        Dictionary containing the ResNet state dict with real weights.
    """
    if pickle_path is None:
        # Default path relative to this file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        pickle_path = os.path.join(current_dir, "..", "weights", "R-52.pkl")

    if not os.path.exists(pickle_path):
        raise FileNotFoundError(f"Pickle file not found at: {pickle_path}")

    print(f"Loading ResNet weights from: {pickle_path}")

    with open(pickle_path, "rb") as f:
        data = pickle.load(f)

    if "model" not in data:
        raise ValueError("Pickle file does not contain 'model' key")

    model_state = data["model"]
    print(f"Loaded {len(model_state)} weight tensors")

    # Convert to bfloat16 to match the random weight functions
    state_dict = {}
    for key, value in model_state.items():
        if isinstance(value, torch.Tensor):
            # Convert to bfloat16 and ensure it's on CPU
            state_dict[key] = value.to(dtype=torch.bfloat16, device="cpu")
        elif hasattr(value, "shape") and hasattr(value, "dtype"):
            # Handle numpy arrays by converting to torch tensor first
            import numpy as np

            if isinstance(value, np.ndarray):
                tensor = torch.from_numpy(value.copy())
                state_dict[key] = tensor.to(dtype=torch.bfloat16, device="cpu")
            else:
                state_dict[key] = value
        else:
            # Keep non-tensor values as-is (like num_batches_tracked)
            state_dict[key] = value

    print("Converted all tensors to bfloat16")
    return state_dict


def create_real_resnet_state_dict(pickle_path: str = None) -> dict:
    """
    Create ResNet state dict using real weights from R-52.pkl file.
    This function provides the same interface as create_full_resnet_state_dict()
    but uses real weights instead of random ones.

    Args:
        pickle_path: Path to the pickle file. If None, uses default path.

    Returns:
        Dictionary containing the ResNet state dict with real weights.
    """
    state_dict = load_resnet_weights_from_pickle(pickle_path)

    # Add missing bias terms that are expected by the model but not present in real weights
    # ResNet typically doesn't use bias in conv layers when using batch normalization,
    # but the TtResNet implementation expects them to be zero

    # Get all weight keys to determine what bias terms we need
    weight_keys = [k for k in state_dict.keys() if k.endswith(".weight") and "norm" not in k]

    for weight_key in weight_keys:
        bias_key = weight_key.replace(".weight", ".bias")
        if bias_key not in state_dict:
            # Create zero bias with the appropriate size (number of output channels)
            weight_tensor = state_dict[weight_key]
            if len(weight_tensor.shape) == 4:  # Conv2d weight: [out_channels, in_channels, H, W]
                out_channels = weight_tensor.shape[0]
                state_dict[bias_key] = torch.zeros(out_channels, dtype=torch.bfloat16)

    # Remove any keys that shouldn't be there
    keys_to_remove = [
        k
        for k in state_dict.keys()
        if k.startswith("stem.fc.") or k.endswith(".num_batches_tracked")  # FC layer not used in our model
    ]  # Not needed by TtResNet
    for key in keys_to_remove:
        del state_dict[key]

    print(f"Added missing bias terms and cleaned up state dict. Final keys: {len(state_dict)}")
    return state_dict
