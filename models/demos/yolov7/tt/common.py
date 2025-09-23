# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.

# SPDX-License-Identifier: Apache-2.0

import ttnn
from tests.ttnn.ttnn_utility_fuction import get_shard_grid_from_num_cores


class TtYOLOv7Conv2D:
    def __init__(
        self,
        input_params,
        conv_params,
        parameters,
        *,
        act_block_h=None,
        deallocate=False,
        deallocate_activation=False,
        height_sharding=True,
        activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.SILU),
        groups=1,
        dtype=ttnn.bfloat8_b,
        num_cores_nhw=None,
        is_reshape=False,
        enable_act_double_buffer=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
        math_approx_mode=False,
        input_channels_alignment=32,
        use_1d_systolic_array=True,
        weights_dtype=ttnn.bfloat8_b,
    ) -> None:
        self.weights = parameters["weight"]
        self.bias = parameters["bias"]
        self.dtype = dtype
        self.deallocate_activation = deallocate_activation
        self.fp32_dest_acc_en = fp32_dest_acc_en
        self.packer_l1_acc = packer_l1_acc
        self.math_approx_mode = math_approx_mode
        self.input_channels_alignment = input_channels_alignment
        self.use_1d_systolic_array = use_1d_systolic_array
        self.input_params = input_params
        self.kernel_size = (self.weights.shape[2], self.weights.shape[3])
        self.conv_params = conv_params
        self.out_channels = self.weights.shape[0]
        self.act_block_h = act_block_h
        self.groups = groups
        self.deallocate = deallocate
        self.activation = activation
        self.shard_layout = (
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED if height_sharding else ttnn.TensorMemoryLayout.BLOCK_SHARDED
        )
        self.num_cores_nhw = num_cores_nhw
        self.is_reshape = is_reshape
        self.enable_act_double_buffer = enable_act_double_buffer
        self.weights_dtype = weights_dtype

    def __call__(self, device, input_tensor):
        conv_config = ttnn.Conv2dConfig(
            weights_dtype=self.weights_dtype,
            activation=self.activation,
            shard_layout=self.shard_layout,
            reshard_if_not_optimal=True if self.use_1d_systolic_array else False,
            enable_act_double_buffer=True,
            enable_weights_double_buffer=True,
            deallocate_activation=self.deallocate_activation,
        )
        compute_config = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.LoFi,
            fp32_dest_acc_en=self.fp32_dest_acc_en,
            packer_l1_acc=self.packer_l1_acc,
            math_approx_mode=self.math_approx_mode,
        )
        if self.num_cores_nhw is not None:
            shard_grid = get_shard_grid_from_num_cores(self.num_cores_nhw, device)
            conv_config.core_grid = shard_grid
            conv_config.override_sharding_config = True

        if self.act_block_h is not None:
            conv_config.act_block_h_override = self.act_block_h

        [output_tensor, [_out_height, _out_width], [self.weights, self.bias]] = ttnn.conv2d(
            input_tensor=input_tensor,
            weight_tensor=self.weights,
            bias_tensor=self.bias,
            in_channels=self.input_params[3],
            out_channels=self.out_channels,
            device=device,
            kernel_size=(self.conv_params[0], self.conv_params[1]),
            stride=(self.conv_params[2], self.conv_params[3]),
            padding=(self.conv_params[4], self.conv_params[5]),
            dilation=(self.conv_params[6], self.conv_params[7]),
            batch_size=self.input_params[0],
            input_height=self.input_params[1],
            input_width=self.input_params[2],
            conv_config=conv_config,
            compute_config=compute_config,
            groups=self.groups,
            return_weights_and_bias=True,
            return_output_dim=True,
            dtype=self.dtype,
        )
        if self.is_reshape:
            output_tensor = ttnn.sharded_to_interleaved(output_tensor, ttnn.L1_MEMORY_CONFIG)
            output_tensor = ttnn.to_layout(output_tensor, ttnn.ROW_MAJOR_LAYOUT)
            output_tensor = ttnn.reshape(
                output_tensor, (input_tensor.shape[0], _out_height, _out_width, output_tensor.shape[-1])
            )
            output_tensor = ttnn.permute(output_tensor, (0, 3, 1, 2))
        return output_tensor


class TtYOLOv7Matmul:
    def __init__(
        self,
        input_params,
        parameters,
        activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.SILU),
        input_dtype=ttnn.bfloat8_b,
        weight_dtype=ttnn.bfloat8_b,
        bias_dtype=ttnn.bfloat8_b,
        output_dtype=ttnn.bfloat8_b,
        pad_input=0,
        # Memory configuration: "dram", "height_sharded", "block_sharded"
        memory_config_type="height_sharded",
        # Matmul type: "1d" (default) or "2d"
        matmul_type="1d",
        # Matmul configuration parameters
        compute_grid_size=(8, 8),
        per_core_M=4,
        per_core_N=16,
        in0_block_w=4,
        out_subblock_h=1,
        out_subblock_w=8,
        out_block_h=4,
        out_block_w=16,
        # Shard configuration
        shard_grid_cores=((0, 0, 7, 7)),  # ((start_x, start_y, end_x, end_y), ...)
        input_shard_shape=(128, 512),
        output_shard_shape=(128, 512),
        # Compute configuration
        math_fidelity=ttnn.MathFidelity.LoFi,
        math_approx_mode=True,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
        fuse_batch=True,
        mcast_in0=False,
        transpose_mcast=False,
        tile_size=32,
    ):
        self.weights = parameters["weight"]
        self.bias = parameters["bias"]
        self.input_params = input_params
        self.out_features = self.weights.shape[0]
        self.activation = activation
        self.pad_input = pad_input

        # Data types
        self.input_dtype = input_dtype
        self.weight_dtype = weight_dtype
        self.bias_dtype = bias_dtype
        self.output_dtype = output_dtype

        # Memory and matmul configuration
        self.memory_config_type = memory_config_type
        self.matmul_type = matmul_type

        # Matmul configuration
        self.compute_grid_size = compute_grid_size
        self.per_core_M = per_core_M
        self.per_core_N = per_core_N
        self.in0_block_w = in0_block_w
        self.out_subblock_h = out_subblock_h
        self.out_subblock_w = out_subblock_w
        self.out_block_h = out_block_h
        self.out_block_w = out_block_w

        # Shard configuration
        self.shard_grid_cores = shard_grid_cores
        self.input_shard_shape = input_shard_shape
        self.output_shard_shape = output_shard_shape

        # Compute configuration
        self.math_fidelity = math_fidelity
        self.math_approx_mode = math_approx_mode
        self.fp32_dest_acc_en = fp32_dest_acc_en
        self.packer_l1_acc = packer_l1_acc
        self.fuse_batch = fuse_batch
        self.mcast_in0 = mcast_in0
        self.transpose_mcast = transpose_mcast
        self.tile_size = tile_size

        # Pre-process weights and bias once during initialization
        self._weights_processed = False
        self._bias_processed = False

    def _get_activation_function(self):
        """Get the activation function based on activation string."""
        activation_map = {
            "silu": ttnn.UnaryWithParam(ttnn.UnaryOpType.SILU),
            "relu": ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            "gelu": ttnn.UnaryWithParam(ttnn.UnaryOpType.GELU),
        }
        return activation_map.get(self.activation.lower())

    def _create_shard_grid(self):
        """Create shard grid from core ranges."""
        core_ranges = []

        if isinstance(self.shard_grid_cores, tuple) and len(self.shard_grid_cores) == 4:
            self.shard_grid_cores = [self.shard_grid_cores]

        for core_range in self.shard_grid_cores:
            start_x, start_y, end_x, end_y = core_range
            core_ranges.append(ttnn.CoreRange(ttnn.CoreCoord(start_x, start_y), ttnn.CoreCoord(end_x, end_y)))
        return ttnn.CoreRangeSet(set(core_ranges))

    def _prepare_weights_and_bias(self, device):
        """Prepare weights and bias tensors for computation."""
        if not self._weights_processed:
            # Convert weights to tiled layout
            self.weights = ttnn.convert_conv_weight_tensor_to_tiled_layout(
                self.weights,
                self.weights.shape[1] // self.tile_size,
                self.weights.shape[0] // self.tile_size,
                output_dtype=ttnn.bfloat16,
            )
            self.weights = ttnn.to_dtype(self.weights, self.weight_dtype)
            self.weights = ttnn.to_device(self.weights, device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            self._weights_processed = True

        if not self._bias_processed:
            # Convert bias to tiled layout
            self.bias = ttnn.to_layout(self.bias, ttnn.TILE_LAYOUT)
            self.bias = ttnn.to_dtype(self.bias, self.bias_dtype)
            self.bias = ttnn.to_device(self.bias, device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            self._bias_processed = True

    def _create_compute_config(self, device):
        """Create compute kernel configuration."""
        return ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=self.math_fidelity,
            math_approx_mode=self.math_approx_mode,
            fp32_dest_acc_en=self.fp32_dest_acc_en,
            packer_l1_acc=self.packer_l1_acc,
        )

    def _create_matmul_config(self):
        """Create matmul program configuration based on matmul type."""
        fused_activation = self.activation

        if self.matmul_type == "1d":
            return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                compute_with_storage_grid_size=self.compute_grid_size,
                in0_block_w=self.in0_block_w,
                out_subblock_h=self.out_subblock_h,
                out_subblock_w=self.out_subblock_w,
                out_block_h=self.out_block_h,
                out_block_w=self.out_block_w,
                per_core_M=self.per_core_M,
                per_core_N=self.per_core_N,
                fuse_batch=self.fuse_batch,
                fused_activation=fused_activation,
                mcast_in0=self.mcast_in0,
                num_global_cb_receivers=0,
            )
        elif self.matmul_type == "2d":
            return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                compute_with_storage_grid_size=self.compute_grid_size,
                in0_block_w=self.in0_block_w,
                out_subblock_h=self.out_subblock_h,
                out_subblock_w=self.out_subblock_w,
                out_block_h=self.out_block_h,
                out_block_w=self.out_block_w,
                per_core_M=self.per_core_M,
                per_core_N=self.per_core_N,
                fuse_batch=self.fuse_batch,
                fused_activation=fused_activation,
                transpose_mcast=self.transpose_mcast,
            )
        else:
            raise ValueError(f"Unsupported matmul type: {self.matmul_type}")

    def _calculate_shard_shapes(self, input_tensor):
        """Calculate shard shapes based on memory config type and input tensor."""
        batch_size, _, nhw, input_channels = input_tensor.shape
        output_channels = self.out_features

        if self.memory_config_type == "height_sharded":
            # Height sharded: shard along height dimension
            num_cores = self._create_shard_grid().num_cores()
            shard_height = (nhw + num_cores * 32 - 1) // (num_cores * 32) * 32
            input_shard_shape = [shard_height, input_channels]
            output_shard_shape = [shard_height, output_channels]

        elif self.memory_config_type == "block_sharded":
            # Block sharded: shard along both height and width dimensions
            grid_h, grid_w = self.compute_grid_size
            shard_height = (nhw + grid_h * 32 - 1) // (grid_h * 32) * 32
            shard_width_in = (input_channels + grid_w * 32 - 1) // (grid_w * 32) * 32
            shard_width_out = (output_channels + grid_w * 32 - 1) // (grid_w * 32) * 32
            input_shard_shape = [shard_height, shard_width_in]
            output_shard_shape = [shard_height, shard_width_out]

        else:  # DRAM - no sharding
            input_shard_shape = None
            output_shard_shape = None

        return input_shard_shape, output_shard_shape

    def _create_memory_config(self, shard_shape=None):
        """Create memory configuration based on memory config type."""
        if self.memory_config_type == "dram":
            return ttnn.DRAM_MEMORY_CONFIG
        elif self.memory_config_type == "height_sharded":
            if shard_shape is None:
                raise ValueError("Shard shape required for height sharded memory config")
            shard_grid = self._create_shard_grid()
            shard_spec = ttnn.ShardSpec(shard_grid, shard_shape, ttnn.ShardOrientation.ROW_MAJOR)
            return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, shard_spec)
        elif self.memory_config_type == "block_sharded":
            if shard_shape is None:
                raise ValueError("Shard shape required for block sharded memory config")
            shard_grid = self._create_shard_grid()
            shard_spec = ttnn.ShardSpec(shard_grid, shard_shape, ttnn.ShardOrientation.ROW_MAJOR)
            return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.BLOCK_SHARDED, ttnn.BufferType.L1, shard_spec)
        else:
            raise ValueError(f"Unsupported memory config type: {self.memory_config_type}")

    def _calculate_per_core_values(self, input_tensor):
        """Calculate per_core_M and per_core_N values based on tensor dimensions and grid size."""
        batch_size, _, nhw, input_channels = input_tensor.shape
        output_channels = self.out_features
        grid_h, grid_w = self.compute_grid_size

        if self.memory_config_type == "height_sharded":
            # For 1D height sharded, only M dimension is distributed
            num_cores = self._create_shard_grid().num_cores()
            per_core_M = (nhw + 32 * num_cores - 1) // (32 * num_cores)
            per_core_N = output_channels // 32
        else:  # For DRAM and block sharded
            # Both M and N dimensions can be distributed
            per_core_M = (nhw + 32 * grid_h - 1) // (32 * grid_h)
            per_core_N = (output_channels + 32 * grid_w - 1) // (32 * grid_w)

        return per_core_M, per_core_N

    def _pad_input_tensor(self, input_tensor, device):
        """Pad input tensor if required."""

        if self.pad_input > 0:
            input_tensor = ttnn.to_layout(input_tensor, ttnn.ROW_MAJOR_LAYOUT)
            pad_width = ((0, 0), (0, 0), (0, self.pad_input), (0, 0))
            input_tensor = ttnn.pad(input_tensor, pad_width, value=0.0)
            input_tensor = ttnn.to_layout(input_tensor, ttnn.TILE_LAYOUT)
        return input_tensor

    def __call__(self, device, input_tensor):
        # Prepare weights and bias (only done once)
        self._prepare_weights_and_bias(device)

        # Calculate dynamic shard shapes and per-core values based on input tensor
        input_shard_shape, output_shard_shape = self._calculate_shard_shapes(input_tensor)
        per_core_M, per_core_N = self._calculate_per_core_values(input_tensor)

        # Update per-core values for this specific call
        self.per_core_M = per_core_M
        self.per_core_N = per_core_N

        # Create configurations
        compute_config = self._create_compute_config(device)
        matmul_config = self._create_matmul_config()

        # Create memory configurations based on type
        if self.memory_config_type == "dram":
            input_memory_config = ttnn.DRAM_MEMORY_CONFIG
            output_memory_config = ttnn.DRAM_MEMORY_CONFIG
        else:
            input_memory_config = self._create_memory_config(input_shard_shape)
            output_memory_config = self._create_memory_config(output_shard_shape)

        # Apply memory configuration to input tensor if not DRAM
        if self.memory_config_type != "dram":
            input_tensor = ttnn.to_memory_config(input_tensor, input_memory_config)

        if input_tensor.layout != ttnn.TILE_LAYOUT:
            # Convert to interleaved, tilize, then reshard to block
            input_tensor = ttnn.sharded_to_interleaved(input_tensor, ttnn.DRAM_MEMORY_CONFIG)
            input_tensor = ttnn.to_layout(input_tensor, ttnn.TILE_LAYOUT)
            input_tensor = ttnn.interleaved_to_sharded(input_tensor, input_memory_config)

        # Pad input tensor if needed
        input_tensor = self._pad_input_tensor(input_tensor, device)

        # Perform linear operation
        linear_args = {
            "input_tensor_a": input_tensor,
            "input_tensor_b": self.weights,
            "bias": self.bias,
            "program_config": matmul_config,
            "dtype": self.output_dtype,
            "compute_kernel_config": compute_config,
        }

        # Add memory_config only if not DRAM (for DRAM, let ttnn decide)
        if self.memory_config_type != "dram":
            linear_args["memory_config"] = output_memory_config

        output_tensor = ttnn.linear(**linear_args)

        output_tensor = output_tensor[:, :, : -self.pad_input, :] if self.pad_input > 0 else output_tensor

        return output_tensor


def get_mesh_mappers(device):
    if device.get_num_devices() > 1:
        inputs_mesh_mapper = ttnn.ShardTensorToMesh(device, dim=0)
        weights_mesh_mapper = ttnn.ReplicateTensorToMesh(device)
        output_mesh_composer = ttnn.ConcatMeshToTensor(device, dim=0)
    else:
        inputs_mesh_mapper = None
        weights_mesh_mapper = None
        output_mesh_composer = None
    return inputs_mesh_mapper, weights_mesh_mapper, output_mesh_composer


def sharded_concat(
    input_tensors,
    num_cores=56,
    dim=3,
    shard_strategy=ttnn.ShardStrategy.HEIGHT,
    shard_orientation=ttnn.ShardOrientation.ROW_MAJOR,
):
    max_cores = 0
    max_id = 0
    for i, tensor in enumerate(input_tensors):
        cores = tensor.memory_config().shard_spec.grid.num_cores()
        if cores > max_cores:
            max_cores = cores
            max_id = i
    id = max_id

    shard_grid = input_tensors[id].memory_config().shard_spec.grid
    num_cores = shard_grid.num_cores()

    in_shard_width = input_tensors[id].shape[-1]
    shard_height = (input_tensors[id].shape[2] + num_cores - 1) // num_cores
    shard_height = ((shard_height + 31) // 32) * 32

    input_sharded_memory_config = ttnn.create_sharded_memory_config_(
        (shard_height, in_shard_width),
        core_grid=shard_grid,
        strategy=shard_strategy,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    out_shard_width = 0
    for i in range(len(input_tensors)):
        out_shard_width += input_tensors[i].shape[-1]

        if input_tensors[i].shape[-1] != in_shard_width:
            temp_input_shared_memory_config = ttnn.create_sharded_memory_config_(
                (shard_height, input_tensors[i].shape[-1]),
                core_grid=shard_grid,
                strategy=shard_strategy,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
            )
            input_tensors[i] = ttnn.to_memory_config(input_tensors[i], temp_input_shared_memory_config)
        else:
            # If the input tensor already has the correct memory config, we can skip this step
            input_tensors[i] = ttnn.to_memory_config(input_tensors[i], input_sharded_memory_config)

    output_sharded_memory_config = ttnn.create_sharded_memory_config_(
        (shard_height, out_shard_width),
        core_grid=shard_grid,
        strategy=shard_strategy,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    output = ttnn.concat(input_tensors, dim, memory_config=output_sharded_memory_config)
    return output
