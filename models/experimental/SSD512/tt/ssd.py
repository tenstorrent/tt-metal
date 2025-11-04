# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0


"""
Full SSD Network - TTNN Implementation

This module integrates VGG backbone, extras backbone, L2Norm, and multibox heads
to create the complete SSD network implementation using TTNN operations.
"""

import ttnn
import torch
from models.experimental.SSD512.tt.layers.vgg_backbone import (
    apply_vgg_backbone,
    create_vgg_layers_with_weights,
    vgg_backbone,
)
from models.experimental.SSD512.tt.layers.extras_backbone import (
    apply_extras_backbone,
    create_extras_layers_with_weights,
    extras_backbone,
    extras,
)
from models.experimental.SSD512.tt.layers.multibox_heads import (
    apply_multibox_heads,
    build_multibox_heads,
    create_multibox_layers_with_weights,
)
from models.experimental.SSD512.reference.ssd import base
from models.experimental.SSD512.tt.layers.l2norm import l2norm


def apply_layers_partial(
    input_tensor, layers_with_weights, start_idx, end_idx, device=None, dtype=ttnn.bfloat16, memory_config=None
):
    """
    Apply a subset of layers from a layer list.

    Args:
        input_tensor: Input tensor
        layers_with_weights: Full list of layers
        start_idx: Start index (inclusive)
        end_idx: End index (exclusive)
        device: TTNN device object
        dtype: TTNN data type
        memory_config: TTNN memory config

    Returns:
        Output tensor after applying layers[start_idx:end_idx]
    """
    subset_layers = layers_with_weights[start_idx:end_idx]

    # Use the appropriate apply function based on layer type
    # Check first layer to determine which apply function to use
    if len(subset_layers) > 0:
        # Check if these are VGG layers (have pool layers) or extras layers (only conv+relu)
        has_pool = any(layer.get("type") == "pool" for layer in subset_layers)

        if has_pool:
            # VGG layers
            return apply_vgg_backbone(
                input_tensor, subset_layers, device=device, dtype=dtype, memory_config=memory_config
            )
        else:
            # Extras layers
            return apply_extras_backbone(
                input_tensor, subset_layers, device=device, dtype=dtype, memory_config=memory_config
            )

    return input_tensor


def ssd_forward_ttnn(
    input_tensor,
    vgg_layers_with_weights,
    extras_layers_with_weights,
    loc_layers_with_weights,
    conf_layers_with_weights,
    device=None,
    dtype=ttnn.bfloat16,
    memory_config=None,
    phase="train",
):
    """
    Full SSD forward pass using TTNN operations.

    This function implements the complete SSD forward pass matching torch_reference_ssd.py:
    1. Apply VGG backbone up to conv4_3 (layer 23)
    2. Apply L2Norm to conv4_3 output
    3. Continue VGG to fc7 (layer 23 onwards)
    4. Apply extras layers (extract sources at k % 2 == 1)
    5. Apply multibox heads to source layers
    6. Flatten and concatenate predictions

    Args:
        input_tensor: Input tensor (torch.Tensor in NCHW format)
        vgg_layers_with_weights: List of VGG backbone layers with weights
        extras_layers_with_weights: List of extras layers with weights
        loc_layers_with_weights: List of location head layers with weights
        conf_layers_with_weights: List of confidence head layers with weights
        device: TTNN device object
        dtype: TTNN data type (default: bfloat16)
        memory_config: TTNN memory config
        phase: "train" or "test" (for future use)

    Returns:
        Tuple of (loc_preds, conf_preds) as torch.Tensor:
            - loc_preds: Flattened location predictions [batch, num_priors*4]
            - conf_preds: Flattened confidence predictions [batch, num_priors*num_classes]
    """
    if memory_config is None:
        memory_config = ttnn.DRAM_MEMORY_CONFIG

    sources = []

    # Split VGG at layer 23 (conv4_3)
    # Layer 23 is the relu after conv4_3
    vgg_split_idx = 23

    # Part 1: Apply VGG up to and including layer 23 (conv4_3 + relu)
    x = apply_layers_partial(
        input_tensor,
        vgg_layers_with_weights,
        0,
        vgg_split_idx,
        device=device,
        dtype=dtype,
        memory_config=memory_config,
    )

    # Apply L2Norm to conv4_3 output
    # Convert to torch NCHW format for L2Norm
    conv4_3_torch = ttnn.to_torch(x)
    # conv4_3_torch = conv4_3_torch.permute(0, 3, 1, 2)  # NHWC -> NCHW

    # Apply L2Norm (returns TTNN tensor in NCHW format)
    # l2norm function expects torch tensor in NCHW and returns TTNN tensor in NCHW
    conv4_3_norm_ttnn = l2norm(conv4_3_torch, num_channels=512, scale=20.0, device=device)

    # Convert to NHWC format for consistency with other tensors
    # The l2norm returns TTNN tensor in NCHW format, convert to torch then NHWC
    conv4_3_norm_torch = ttnn.to_torch(conv4_3_norm_ttnn)
    conv4_3_norm_torch = conv4_3_norm_torch.permute(0, 2, 3, 1)  # NCHW -> NHWC
    conv4_3_norm_ttnn = ttnn.from_torch(
        conv4_3_norm_torch,
        device=device,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        memory_config=memory_config,
    )

    sources.append(conv4_3_norm_ttnn)

    # Part 2: Continue VGG from layer 23 onwards (to fc7)
    x = apply_layers_partial(
        x,
        vgg_layers_with_weights,
        vgg_split_idx,
        len(vgg_layers_with_weights),
        device=device,
        dtype=dtype,
        memory_config=memory_config,
    )

    # Add fc7 to sources
    sources.append(x)

    # Apply extras layers
    # In PyTorch: for k, v in enumerate(self.extras): x = F.relu(v(x), inplace=True); if k % 2 == 1: sources.append(x)
    # So k is the index of the conv layer in extras list (which contains only conv layers)
    # We extract sources after applying every 2nd conv layer (k=1, 3, 5, ...)
    # In our extras_layers_with_weights, we have conv+relu pairs
    # So we need to count conv layers and extract after every 2nd conv's relu

    extras_k = 0  # Count of conv layers (0-indexed)

    i = 0
    while i < len(extras_layers_with_weights):
        layer = extras_layers_with_weights[i]

        if layer["type"] == "conv":
            # Collect conv and its following relu
            temp_layers = [layer]
            if i + 1 < len(extras_layers_with_weights) and extras_layers_with_weights[i + 1]["type"] == "relu":
                temp_layers.append(extras_layers_with_weights[i + 1])

            # Apply conv+relu pair
            x = apply_extras_backbone(
                x,
                temp_layers,
                device=device,
                dtype=dtype,
                memory_config=memory_config,
            )

            extras_k += 1

            # Extract source after relu if k % 2 == 1 (after 2nd, 4th, 6th conv layer)
            # This means after extras_k = 1, 3, 5, ... (0-indexed: 2nd, 4th, 6th)
            if extras_k % 2 == 1:
                sources.append(x)

            # Skip the relu layer since we already processed it
            if i + 1 < len(extras_layers_with_weights) and extras_layers_with_weights[i + 1]["type"] == "relu":
                i += 2  # Skip both conv and relu
            else:
                i += 1  # Only conv, no relu (shouldn't happen)
        else:
            # Shouldn't have standalone relu (should be paired with conv)
            i += 1

    # Apply multibox heads to source layers
    tt_loc_preds, tt_conf_preds = apply_multibox_heads(
        sources,
        loc_layers_with_weights,
        conf_layers_with_weights,
        device=device,
        dtype=dtype,
        memory_config=memory_config,
    )

    # Flatten and concatenate predictions
    # Convert TTNN tensors to torch for concatenation
    loc_torch_list = []
    conf_torch_list = []

    for loc_pred in tt_loc_preds:
        loc_torch = ttnn.to_torch(loc_pred)
        loc_torch = loc_torch.permute(0, 3, 1, 2)  # NHWC -> NCHW
        loc_torch = loc_torch.contiguous()
        loc_torch = loc_torch.view(loc_torch.size(0), -1)  # Flatten: [batch, H*W*C]
        loc_torch_list.append(loc_torch)

    for conf_pred in tt_conf_preds:
        conf_torch = ttnn.to_torch(conf_pred)
        conf_torch = conf_torch.permute(0, 3, 1, 2)  # NHWC -> NCHW
        conf_torch = conf_torch.contiguous()
        conf_torch = conf_torch.view(conf_torch.size(0), -1)  # Flatten: [batch, H*W*C]
        conf_torch_list.append(conf_torch)

    # Concatenate all predictions
    loc_concat = torch.cat(loc_torch_list, dim=1)  # [batch, total_priors*4]
    conf_concat = torch.cat(conf_torch_list, dim=1)  # [batch, total_priors*num_classes]

    return loc_concat, conf_concat


def build_ssd_network_ttnn(
    phase="train",
    size=300,
    num_classes=21,
    device=None,
    dtype=ttnn.bfloat16,
):
    """
    Build complete SSD network using TTNN operations.

    This function creates all components of the SSD network:
    - VGG backbone
    - Extras backbone
    - Multibox heads

    Args:
        phase: "train" or "test"
        size: Input size (300 or 512)
        num_classes: Number of classes
        device: TTNN device object
        dtype: TTNN data type (default: bfloat16)

    Returns:
        Dictionary containing all network components with weights:
        {
            'vgg_layers': vgg_layers_with_weights,
            'extras_layers': extras_layers_with_weights,
            'loc_layers': loc_layers_with_weights,
            'conf_layers': conf_layers_with_weights,
            'phase': phase,
            'size': size,
            'num_classes': num_classes,
        }
    """
    if size not in [300, 512]:
        raise ValueError(f"Size must be 300 or 512, got {size}")

    # Build VGG backbone
    vgg_cfg = base[str(size)]
    vgg_layers_config = vgg_backbone(vgg_cfg, input_channels=3, batch_norm=False, device=device)
    vgg_layers_with_weights = create_vgg_layers_with_weights(vgg_layers_config, device=device, dtype=dtype)

    # Build extras backbone
    extras_cfg = extras[str(size)]
    extras_layers_config = extras_backbone(extras_cfg, input_channels=1024, batch_norm=False, device=device)
    extras_layers_with_weights = create_extras_layers_with_weights(extras_layers_config, device=device, dtype=dtype)

    # Build multibox heads
    loc_layers_config, conf_layers_config = build_multibox_heads(
        size=size,
        num_classes=num_classes,
        device=device,
    )
    loc_layers_with_weights, conf_layers_with_weights = create_multibox_layers_with_weights(
        loc_layers_config,
        conf_layers_config,
        device=device,
        dtype=dtype,
    )

    return {
        "vgg_layers": vgg_layers_with_weights,
        "extras_layers": extras_layers_with_weights,
        "loc_layers": loc_layers_with_weights,
        "conf_layers": conf_layers_with_weights,
        "phase": phase,
        "size": size,
        "num_classes": num_classes,
    }
