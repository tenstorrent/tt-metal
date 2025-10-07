# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch


class Conv:
    def __init__(
        self,
        conv_params,
        parameters,
        *,
        act_block_h=None,
        reshard=False,
        deallocate=True,
        height_sharding=True,
        width_sharding=None,
        activation="",
        dtype=ttnn.bfloat16,
        groups=1,
        dilation=1,
        use_shallow_conv_variant=False,
    ) -> None:
        self.weights = parameters["weight"]
        self.bias = parameters["bias"]

        self.kernel_size = (self.weights.shape[2], self.weights.shape[3])
        self.conv_params = conv_params
        self.out_channels = self.weights.shape[0]
        self.act_block_h = act_block_h
        self.reshard = reshard
        self.deallocate = deallocate
        self.activation = activation
        self.dtype = dtype
        if width_sharding == True:
            self.shard_layout = ttnn.TensorMemoryLayout.WIDTH_SHARDED
        else:
            self.shard_layout = (
                ttnn.TensorMemoryLayout.HEIGHT_SHARDED if height_sharding else ttnn.TensorMemoryLayout.BLOCK_SHARDED
            )
        self.groups = groups
        self.dilation = dilation
        self.use_shallow_conv_variant = use_shallow_conv_variant

    def __call__(self, device, input_tensor):
        batch_size = input_tensor.shape[0]
        input_height = input_tensor.shape[1]
        input_width = input_tensor.shape[2]
        input_channels = input_tensor.shape[3]

        # Check for small spatial dimensions that might cause issues
        is_small_spatial = input_height < 32 or input_width < 32

        input_tensor = ttnn.to_memory_config(input_tensor, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        # if input_tensor.shape[3] == 1024 or input_tensor.shape[3] == 384:
        #     input_tensor = ttnn.to_memory_config(input_tensor, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        # else:
        #     input_tensor = ttnn.from_device(input_tensor)

        # check input is in interleaved format
        if hasattr(input_tensor, "memory_config") and input_tensor.memory_config().is_sharded():
            # input_tensor = ttnn.sharded_to_interleaved(input_tensor, ttnn.DRAM_MEMORY_CONFIG)
            input_tensor = ttnn.sharded_to_interleaved(input_tensor, ttnn.L1_MEMORY_CONFIG)

        # check weights are also properly formatted
        if hasattr(self.weights, "memory_config") and self.weights.memory_config().is_sharded():
            self.weights = ttnn.sharded_to_interleaved(self.weights, ttnn.DRAM_MEMORY_CONFIG)

        compute_config = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
            math_approx_mode=False,
        )

        output_tensor = None
        _out_height = None
        _out_width = None

        try:
            if self.kernel_size[0] == 3 and self.kernel_size[1] == 3:
                # For 3x3 convolutions, use the minimal configuration that works
                temp_input = ttnn.from_device(input_tensor) if hasattr(input_tensor, "device") else input_tensor
                temp_input = ttnn.to_device(temp_input, device, memory_config=ttnn.DRAM_MEMORY_CONFIG)

                # Minimal config that works reliably
                conv_config = ttnn.Conv2dConfig(
                    weights_dtype=ttnn.bfloat16,
                    output_layout=ttnn.TILE_LAYOUT,
                    deallocate_activation=True,
                    reallocate_halo_output=True,
                    enable_act_double_buffer=True,
                    activation=None,  # Handle activation separately if needed
                )

                [output_tensor, [_out_height, _out_width]] = ttnn.conv2d(
                    input_tensor=temp_input,
                    weight_tensor=self.weights,
                    bias_tensor=self.bias,
                    device=device,
                    in_channels=input_channels,
                    out_channels=self.out_channels,
                    batch_size=batch_size,
                    input_height=input_height,
                    input_width=input_width,
                    kernel_size=self.kernel_size,
                    stride=(self.conv_params[0], self.conv_params[1]),
                    padding=(self.conv_params[2], self.conv_params[3]),
                    dilation=(self.dilation, self.dilation),
                    groups=self.groups,
                    conv_config=conv_config,
                    compute_config=compute_config,
                    return_output_dim=True,
                    return_weights_and_bias=False,
                )

                # Apply activation after convolution if needed
                if self.activation == "relu":
                    output_tensor = ttnn.relu(output_tensor)
                elif self.activation == "gelu":
                    output_tensor = ttnn.gelu(output_tensor)
                # Add other activations as needed

            else:
                # For 1x1 convolutions, try DRAM approach first for small tensors
                if is_small_spatial:
                    temp_input = ttnn.from_device(input_tensor) if hasattr(input_tensor, "device") else input_tensor
                    temp_input = ttnn.to_device(temp_input, device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                    conv_input = temp_input
                else:
                    conv_input = input_tensor

                [output_tensor, [_out_height, _out_width]] = ttnn.conv2d(
                    input_tensor=conv_input,
                    weight_tensor=self.weights,
                    bias_tensor=self.bias,
                    device=device,
                    in_channels=input_channels,
                    out_channels=self.out_channels,
                    batch_size=batch_size,
                    input_height=input_height,
                    input_width=input_width,
                    kernel_size=self.kernel_size,
                    stride=(self.conv_params[0], self.conv_params[1]),
                    padding=(self.conv_params[2], self.conv_params[3]),
                    dilation=(self.dilation, self.dilation),
                    groups=self.groups,
                    compute_config=compute_config,
                    return_output_dim=True,
                    return_weights_and_bias=False,
                )

                # Apply activation if needed
                if self.activation == "relu":
                    output_tensor = ttnn.relu(output_tensor)
                elif self.activation == "gelu":
                    output_tensor = ttnn.gelu(output_tensor)

        except Exception as e:
            # If the optimized approach fails, try a more conservative fallback
            try:
                print(f"Conv2d failed ({e}), trying fallback...")

                # Force everything to DRAM and use minimal config
                temp_input = ttnn.from_device(input_tensor) if hasattr(input_tensor, "device") else input_tensor
                temp_input = ttnn.to_device(temp_input, device, memory_config=ttnn.DRAM_MEMORY_CONFIG)

                temp_weights = ttnn.from_device(self.weights) if hasattr(self.weights, "device") else self.weights
                temp_weights = ttnn.to_device(temp_weights, device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                if not hasattr(self.weights, "device") or self.weights.device != device:
                    self.weights = ttnn.to_device(self.weights, device)
                    self.bias = ttnn.to_device(self.bias, device)
                [output_tensor, [_out_height, _out_width]] = ttnn.conv2d(
                    input_tensor=temp_input,
                    weight_tensor=temp_weights,
                    bias_tensor=self.bias,
                    device=device,
                    in_channels=input_channels,
                    out_channels=self.out_channels,
                    batch_size=batch_size,
                    input_height=input_height,
                    input_width=input_width,
                    kernel_size=self.kernel_size,
                    stride=(self.conv_params[0], self.conv_params[1]),
                    padding=(self.conv_params[2], self.conv_params[3]),
                    dilation=(self.dilation, self.dilation),
                    groups=self.groups,
                    compute_config=compute_config,
                    return_output_dim=True,
                    return_weights_and_bias=False,
                )

                # Apply activation if needed
                if self.activation == "relu":
                    output_tensor = ttnn.relu(output_tensor)
                elif self.activation == "gelu":
                    output_tensor = ttnn.gelu(output_tensor)

            except Exception as fallback_e:
                raise RuntimeError(
                    f"Conv2d failed. Primary: {e}, Fallback: {fallback_e}. "
                    f"Kernel size: {self.kernel_size}, Input shape: {[batch_size, input_height, input_width, input_channels]}, "
                    f"Conv params: {self.conv_params}, Small spatial: {is_small_spatial}"
                )

        # Post-processing
        if hasattr(output_tensor, "memory_config") and output_tensor.memory_config().is_sharded():
            output_tensor = ttnn.sharded_to_interleaved(output_tensor, ttnn.DRAM_MEMORY_CONFIG)

        output_tensor = ttnn.to_layout(output_tensor, layout=ttnn.ROW_MAJOR_LAYOUT)
        output_tensor = ttnn.reshape(output_tensor, (batch_size, _out_height, _out_width, output_tensor.shape[3]))
        output_tensor = ttnn.to_layout(output_tensor, layout=ttnn.TILE_LAYOUT)
        del _out_height, _out_width

        return output_tensor


class Conv_with_split:
    def __init__(
        self,
        conv_params,
        parameters,
        *,
        act_block_h=None,
        reshard=False,
        deallocate=False,
        height_sharding=True,
        width_sharding=None,
        activation="",
        dtype=ttnn.bfloat16,
        groups=1,
        dilation=1,
        use_shallow_conv_variant=False,
        split_factor=2,
    ) -> None:
        self.weights = parameters["weight"]
        self.bias = parameters["bias"]
        input_channels = self.weights.shape[1]
        self.output_channels = self.weights.shape[0]
        assert input_channels % split_factor == 0
        self.split_input_channels = input_channels // split_factor
        if width_sharding == True:
            self.shard_layout = ttnn.TensorMemoryLayout.WIDTH_SHARDED
        else:
            self.shard_layout = (
                ttnn.TensorMemoryLayout.HEIGHT_SHARDED if height_sharding else ttnn.TensorMemoryLayout.BLOCK_SHARDED
            )
        self.dtype = dtype
        self.split_factor = split_factor
        self.act_block_h = act_block_h
        self.conv_params = conv_params
        self.activation = activation
        self.kernel_size = (self.weights.shape[2], self.weights.shape[3])

    def __call__(self, device, input_tensor):
        batch, height, width, channel = input_tensor.shape
        original_input = input_tensor
        input_tensor = ttnn.to_torch(input_tensor)
        self.weights = ttnn.to_torch(self.weights)
        split_input_tensors = torch.split(input_tensor, self.split_input_channels, 3)
        split_weight_tensors = torch.split(self.weights, self.split_input_channels, 1)

        weights_dtype = ttnn.float32

        compute_config = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
            math_approx_mode=False,
        )

        for i in range(self.split_factor):
            tt_weight_tensor = ttnn.from_torch(
                split_weight_tensors[i], weights_dtype if weights_dtype != ttnn.bfloat8_b else ttnn.float32
            )
            if i == 0:
                tt_bias_tensor = self.bias

            tt_input_tensor = ttnn.from_torch(split_input_tensors[i], ttnn.bfloat16)

            [tt_output_tensor_on_device, [out_height, out_width]] = ttnn.conv2d(
                input_tensor=tt_input_tensor,
                weight_tensor=tt_weight_tensor,
                bias_tensor=tt_bias_tensor,
                device=device,
                in_channels=self.split_input_channels,
                out_channels=self.output_channels,
                batch_size=batch,
                input_height=height,
                input_width=width,
                kernel_size=self.kernel_size,
                stride=(self.conv_params[0], self.conv_params[1]),
                padding=(self.conv_params[2], self.conv_params[3]),
                dilation=(1, 1),
                groups=1,
                compute_config=compute_config,
                return_output_dim=True,
                return_weights_and_bias=False,
            )
            tt_conv_output_tensor = ttnn.from_device(tt_output_tensor_on_device)
            torch_conv_output_tensor = ttnn.to_torch(tt_conv_output_tensor)

            if i == 0:
                torch_output_tensor = torch_conv_output_tensor
            else:
                torch_output_tensor = torch.add(torch_output_tensor, torch_conv_output_tensor)

        if len(torch_output_tensor.shape) == 2:
            # Reshape to 4D if it got flattened
            torch_output_tensor = torch_output_tensor.reshape(batch, out_height, out_width, self.output_channels)
        elif torch_output_tensor.shape[1] == 1 and torch_output_tensor.shape[2] != out_width:
            # Shape is corrupted [batch, 1, h*w, channels]
            torch_output_tensor = torch_output_tensor.reshape(batch, out_height, out_width, self.output_channels)

        output_tensor = ttnn.from_torch(torch_output_tensor, dtype=ttnn.bfloat16, device=device)
        # output_tensor = ttnn.to_layout(output_tensor, layout=ttnn.ROW_MAJOR_LAYOUT)
        if output_tensor.get_layout() != ttnn.ROW_MAJOR_LAYOUT:
            output_tensor = ttnn.to_layout(output_tensor, layout=ttnn.ROW_MAJOR_LAYOUT)

        # output_tensor = ttnn.reshape(output_tensor, (batch, out_height, out_width, output_tensor.shape[3]))
        # output_tensor = ttnn.to_layout(output_tensor, layout=ttnn.TILE_LAYOUT)
        if output_tensor.shape[1] != out_height or output_tensor.shape[2] != out_width:
            output_tensor = ttnn.reshape(output_tensor, (batch, out_height, out_width, self.output_channels))

        expected_shape = (batch, out_height, out_width, self.output_channels)
        if output_tensor.shape != expected_shape:
            output_tensor = ttnn.reshape(output_tensor, expected_shape)

        output_tensor = ttnn.to_layout(output_tensor, layout=ttnn.TILE_LAYOUT)

        # Apply activation if specified
        if self.activation == "relu":
            output_tensor = ttnn.relu(output_tensor)

        del out_height, out_width
        return output_tensor
