# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import ttnn


class TtConv2D:
    def __init__(
        self,
        conv,
        conv_pth,
        device=None,
        activation=None,
        activation_dtype=ttnn.bfloat16,
        weights_dtype=ttnn.bfloat8_b,
        shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        is_blk=False,
        dealloc_act=False,
        act_block_h=None,
    ):
        if is_blk:
            shard_layout = ttnn.TensorMemoryLayout.BLOCK_SHARDED
        self.conv = conv
        self.device = device
        self.in_channels = conv.in_channels
        self.out_channels = conv.out_channels
        self.kernel_size = conv.kernel_size
        self.padding = conv.padding
        self.stride = conv.stride
        self.groups = conv.groups
        self.activation_dtype = activation_dtype
        self.compute_config = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.LoFi,
            fp32_dest_acc_en=False,
            packer_l1_acc=False,
            math_approx_mode=True,
        )
        self.conv_config = ttnn.Conv2dConfig(
            weights_dtype=weights_dtype,
            shard_layout=shard_layout,
            deallocate_activation=dealloc_act,
            enable_act_double_buffer=False,
            reshard_if_not_optimal=True,
            activation=activation,
        )
        if act_block_h is not None:
            self.conv_config.act_block_h_override = act_block_h
        if conv_pth.bias is not None:
            self.bias = conv_pth.bias
        else:
            self.bias = None

        self.weight = conv_pth.weight

    def __call__(self, x):
        input_height = self.conv.input_height
        input_width = self.conv.input_width
        batch_size = self.conv.batch_size
        [x, [output_height, output_width], [self.weight, self.bias]] = ttnn.conv2d(
            input_tensor=x,
            weight_tensor=self.weight,
            bias_tensor=self.bias,
            device=self.device,
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            input_height=input_height,
            input_width=input_width,
            batch_size=batch_size,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            conv_config=self.conv_config,
            slice_config=ttnn.Conv2dL1FullSliceConfig,
            groups=self.groups,
            compute_config=self.compute_config,
            return_output_dim=True,
            return_weights_and_bias=True,
            dtype=self.activation_dtype,
        )
        return x, output_height, output_width


def _ensure_uint32(tensor):
    """
    Ensure tensor is UINT32 dtype for use as gather indices
    ttnn.gather accepts only UINT16 or UINT32 for indices
    """
    if tensor.dtype == ttnn.uint32:
        return tensor

    # Save original layout to restore after typecast
    original_layout = tensor.layout
    needs_layout_restore = original_layout != ttnn.TILE_LAYOUT

    if needs_layout_restore:
        tensor = ttnn.to_layout(tensor, ttnn.TILE_LAYOUT)

    tensor = ttnn.typecast(tensor, dtype=ttnn.uint32)

    if needs_layout_restore:
        tensor = ttnn.to_layout(tensor, original_layout)

    return tensor


def advanced_indexing(tensor, indices):
    """
    Implements tensor[range(N), indices] for 2D/3D tensors using ttnn operations.

    2D case [N, C] with indices [N]:
        Returns [N] where result[i] = tensor[i, indices[i]]

    3D case [N, C, D] with indices [N]:
        Returns [N, D] where result[i, :] = tensor[i, indices[i], :]

    Returns:
        [N] or [N, D] tensor

    """
    original_layout = tensor.layout
    N = tensor.shape[0]
    is_3d = len(tensor.shape) == 3

    # Work in ROW_MAJOR to avoid TILE padding issues during reshape
    # (TILE layout pads dimensions to multiples of 32, breaking reshape math)
    tensor = ttnn.to_layout(tensor, ttnn.ROW_MAJOR_LAYOUT)
    indices = ttnn.to_layout(indices, ttnn.ROW_MAJOR_LAYOUT)

    if is_3d:
        # 3D: [N, C, D] -> [N, D]
        # Strategy: Flatten to [N*C, D], compute flat indices, then gather
        C, D = tensor.shape[1], tensor.shape[2]
        tensor_flat = ttnn.reshape(tensor, (N * C, D))

        # Compute flat indices: i*C + indices[i] for each i
        batch_offset = ttnn.mul(ttnn.arange(0, N, 1, device=tensor.device()), float(C))
        batch_offset = _ensure_uint32(batch_offset)  # mul changes dtype to BFLOAT16
        flat_indices = ttnn.add(batch_offset, indices)  # [N]

        # Expand to [N, D] to gather all D elements for each of N rows
        flat_indices = ttnn.reshape(flat_indices, (N, 1))
        flat_indices = ttnn.repeat(flat_indices, (1, D))

        # ttnn.gather requires TILE layout for both tensors
        tensor_flat = ttnn.to_layout(tensor_flat, ttnn.TILE_LAYOUT)
        flat_indices = ttnn.to_layout(flat_indices, ttnn.TILE_LAYOUT)
        result = ttnn.gather(tensor_flat, dim=0, index=flat_indices)

    else:
        # 2D: [N, C] -> [N]
        # Strategy: Gather along dim 1 using indices
        indices_expanded = ttnn.reshape(indices, (N, 1))

        # ttnn.gather requires TILE layout for both tensors
        tensor = ttnn.to_layout(tensor, ttnn.TILE_LAYOUT)
        indices_expanded = ttnn.to_layout(indices_expanded, ttnn.TILE_LAYOUT)
        result = ttnn.gather(tensor, dim=1, index=indices_expanded)
        result = ttnn.reshape(result, (N,))

    # Restore original layout
    if original_layout == ttnn.ROW_MAJOR_LAYOUT:
        result = ttnn.to_layout(result, ttnn.ROW_MAJOR_LAYOUT)

    return result


def boolean_indexing(tensor, mask):
    """
    Implements tensor[mask] using ttnn operations (gather with nonzero indices).

    Args:
        tensor: [N], [N, C], or [B, N, C] input tensor
        mask: [N] boolean mask

    Returns:
        [M], [M, C], or [B, M, C] tensor where M = number of True values in mask

    """
    original_layout = tensor.layout
    original_shape = tensor.shape
    N = mask.shape[0]

    # Handle 3D input by flattening batch dimension (B=1 case)
    is_3d = len(original_shape) == 3
    if is_3d:
        B, N_tensor, C = original_shape
        if B == 1:
            tensor = ttnn.reshape(tensor, (N_tensor, C))
        else:
            raise NotImplementedError("boolean_indexing only supports B=1 for 3D tensors")

    # Get nonzero indices from mask
    # ttnn.nonzero requires 4D input and returns [M, 4] coordinates
    mask = ttnn.to_layout(mask, ttnn.ROW_MAJOR_LAYOUT)
    mask_4d = ttnn.reshape(mask, (1, 1, 1, N))
    indices_result = ttnn.nonzero(mask_4d)

    # Unwrap from list if necessary
    if isinstance(indices_result, list) and len(indices_result) > 0:
        indices_result = indices_result[0]

    # Extract indices: result is [M, 4], we need column 3 (the N dimension)
    result_shape = indices_result.shape
    if len(result_shape) == 4:
        # Result is [1, 1, 1, M*4], reshape to [M, 4]
        M = result_shape[3] // 4
        indices_2d = ttnn.reshape(indices_result, (M, 4))
    else:
        # Already [M, 4]
        M = result_shape[0]
        indices_2d = indices_result

    # Slice column 3 (last dimension index): [M, 4] -> [M, 1] -> [M]
    indices = ttnn.slice(indices_2d, (0, 3), (M, 4))
    indices = ttnn.reshape(indices, (M,))
    indices = _ensure_uint32(indices)

    # Perform gather (requires TILE layout for both tensors)
    tensor = ttnn.to_layout(tensor, ttnn.TILE_LAYOUT)

    if len(tensor.shape) == 1:
        # 1D: [N] -> [M]
        indices = ttnn.to_layout(indices, ttnn.TILE_LAYOUT)
        result = ttnn.gather(tensor, dim=0, index=indices)
    else:
        # 2D: [N, C] -> [M, C]
        # Repeat indices for all C columns
        C = tensor.shape[-1]
        indices_expanded = ttnn.reshape(indices, (M, 1))
        indices_expanded = ttnn.repeat(indices_expanded, (1, C))
        indices_expanded = _ensure_uint32(indices_expanded)  # repeat may change dtype
        indices_expanded = ttnn.to_layout(indices_expanded, ttnn.TILE_LAYOUT)
        result = ttnn.gather(tensor, dim=0, index=indices_expanded)

    # Reshape back to 3D if input was 3D
    if is_3d and original_shape[0] == 1:
        # Convert to ROW_MAJOR to check actual shape
        result = ttnn.to_layout(result, ttnn.ROW_MAJOR_LAYOUT)
        result_shape = result.shape

        if len(result_shape) == 4:
            # Some operations may return 4D [1, 1, M, C], flatten to [1, M, C]
            result = ttnn.reshape(result, (1, result_shape[2], result_shape[3]))
        elif len(result_shape) == 2:
            # Result is [M, C], add batch dimension to get [1, M, C]
            result = ttnn.reshape(result, (1, result_shape[0], result_shape[1]))

        # Convert back to original layout
        if original_layout == ttnn.TILE_LAYOUT:
            result = ttnn.to_layout(result, ttnn.TILE_LAYOUT)
    else:
        # Restore original layout for non-3D
        if original_layout != ttnn.TILE_LAYOUT:
            result = ttnn.to_layout(result, original_layout)

    return result
