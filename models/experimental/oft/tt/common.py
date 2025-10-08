import ttnn
import torch
import math
from loguru import logger


def _nearest_32_per_core(x, core):
    return math.ceil(x / core / 32) * 32 * core


def _nearest_32(x):
    return math.ceil(x / 32) * 32


def calculate_matmul_shard_dims(per_core_M, per_core_N, in_ch, out_ch, core_grid, sharding_strategy):
    if sharding_strategy == "height":
        shard_height = per_core_M * ttnn.TILE_SIZE
        in_shard_width = (in_ch + ttnn.TILE_SIZE - 1) // ttnn.TILE_SIZE * ttnn.TILE_SIZE
        out_shard_width = (out_ch + ttnn.TILE_SIZE - 1) // ttnn.TILE_SIZE * ttnn.TILE_SIZE
    else:
        shard_height = per_core_M * ttnn.TILE_SIZE
        in_shard_width = (in_ch // core_grid.x + ttnn.TILE_SIZE - 1) // ttnn.TILE_SIZE * ttnn.TILE_SIZE
        out_shard_width = per_core_N * ttnn.TILE_SIZE
    return shard_height, in_shard_width, out_shard_width


def infer_per_core_dims(n, h, w, in_ch, out_ch, core_grid, sharding_strategy):
    """Infer per-core M and N dimensions based on sharding strategy.
    For height sharded, go for the largest possible core number. starting from core_grid.y*core_grid.x and then search for divisors of n*h*w/32. If not found, search the closest value to the whole divisor

    """
    nhw = n * h * w

    nhw_padded = _nearest_32(nhw)
    in_ch_padded = _nearest_32(in_ch)
    out_ch_padded = _nearest_32(out_ch)

    if sharding_strategy == ttnn.TensorMemoryLayout.HEIGHT_SHARDED:
        # For HEIGHT_SHARDED, determine optimal number of cores
        total_cores = core_grid.y * core_grid.x
        cores_to_use = total_cores
        found_divisor = False

        # Try all values from total_cores down to total_cores/2
        for cores_to_try in range(total_cores, 15 - 1, -1):
            if (nhw_padded // 32) % cores_to_try == 0:
                cores_to_use = cores_to_try
                found_divisor = True
                break

        if found_divisor:
            print(f"Using {cores_to_use} cores for HEIGHT_SHARDED")
            per_core_M = (nhw_padded // 32) // cores_to_use
        else:
            # Default to using max cores if no exact divisor found
            print(f"No exact divisor found, using {total_cores} cores for HEIGHT_SHARDED {nhw_padded}")
            per_core_M = math.ceil((nhw_padded // 32) / total_cores)

        per_core_N = math.ceil(out_ch_padded / ttnn.TILE_SIZE)

    elif sharding_strategy == ttnn.TensorMemoryLayout.BLOCK_SHARDED:
        per_core_M = math.ceil(nhw_padded / core_grid.x / ttnn.TILE_SIZE)
        per_core_N = math.ceil(out_ch_padded / core_grid.y / ttnn.TILE_SIZE)

    return per_core_M, per_core_N


def infer_out_subblock(per_core_M, per_core_N, dtype=None):
    """
    Infer optimal output subblock dimensions for given per-core dimensions.

    Args:
        per_core_M (int): Height dimension per core
        per_core_N (int): Width dimension per core
        dtype (ttnn.DataType, optional): Data type that affects constraints.
            Defaults to None (which uses BFloat16 constraint).

    Returns:
        tuple: (best_h, best_w) optimal subblock dimensions

    Constraints:
        - out_subblock_h * out_subblock_w <= max_product (8 for BFloat16, 4 for Float32)
        - out_subblock_h must divide per_core_M evenly
        - out_subblock_w must divide per_core_N evenly
        - out_subblock_w must equal per_core_N OR out_subblock_h must equal 1
    """

    # Determine max product based on data type
    max_product = 4 if dtype == ttnn.float32 else 8

    # Strategy 1: Set out_subblock_w = per_core_N
    max_h = max_product // per_core_N if per_core_N > 0 else 0
    if max_h > 0:
        # Find largest divisor of per_core_M that is <= max_h
        for h in range(min(max_h, per_core_M), 0, -1):
            if per_core_M % h == 0:
                return h, per_core_N

    # Strategy 2: Set out_subblock_h = 1
    max_w = min(max_product, per_core_N)
    for w in range(max_w, 0, -1):
        if per_core_N % w == 0:
            return 1, w

    # Fallback (shouldn't normally reach here)
    return 1, 1


class Conv:
    def __init__(
        self,
        parameters,
        conv_pt,
        *,
        stride=1,
        padding=1,
        act_block_h=32,
        reshard=False,
        deallocate=False,
        height_sharding=None,
        activation=None,
        width_sharding=False,
        block_sharding=False,
        dtype=ttnn.bfloat8_b,
        weights_dtype=ttnn.bfloat8_b,
        output_layout=ttnn.TILE_LAYOUT,
        is_sliced=False,
    ) -> None:
        self.weights = parameters.weight
        # logger.debug(f"Conv weights: {self.weights.shape}")
        # logger.debug(f"Conv parameters: {self.weights}")
        self.conv_pt = conv_pt
        # logger.debug(f"Conv: {self.conv_pt}")

        # Automatically detect if bias is present in parameters
        try:
            self.has_bias = hasattr(parameters, "bias") and parameters.bias is not None
        except (KeyError, AttributeError):
            self.has_bias = False

        if self.has_bias:
            logger.debug(f"Conv: bias found in parameters")

        # handle comparison mode that requires bias
        if ttnn.CONFIG.enable_comparison_mode:
            if self.has_bias:
                try:
                    self.bias = parameters.bias
                except (KeyError, AttributeError):
                    self.bias = None
            else:
                # Create bias tensor with proper shape for TTNN conv2d
                bias_tensor = torch.zeros(conv_pt.out_channels)
                self.bias = bias_tensor.view(1, 1, 1, -1)
                # Convert bias to ttnn tensor
                self.bias = ttnn.from_torch(self.bias, dtype=ttnn.DataType.BFLOAT16, layout=ttnn.ROW_MAJOR_LAYOUT)
            # In comparison mode, we always have bias (either real or zero)
            self.has_bias = True
        else:
            if self.has_bias:
                try:
                    self.bias = parameters.bias
                except (KeyError, AttributeError):
                    self.bias = None

        self.kernel_size = (self.weights.shape[2], self.weights.shape[3])
        self.stride = conv_pt.stride  # stride
        self.padding = conv_pt.padding  # padding
        self.out_channels = conv_pt.out_channels
        # logger.debug(f"Conv out channels: {self.out_channels}")
        self.act_block_h = act_block_h
        self.reshard = reshard
        self.dtype = dtype
        self.weights_dtype = weights_dtype
        self.output_layout = output_layout

        if width_sharding:
            self.shard_layout = ttnn.TensorMemoryLayout.WIDTH_SHARDED
        elif height_sharding:
            self.shard_layout = ttnn.TensorMemoryLayout.HEIGHT_SHARDED
        elif block_sharding:
            self.shard_layout = ttnn.TensorMemoryLayout.BLOCK_SHARDED
        else:
            self.shard_layout = None

        self.deallocate = deallocate
        self.activation = activation
        self.is_sliced = is_sliced
        self.slice_config = None
        if is_sliced:
            self.slice_config = ttnn.Conv2dSliceConfig(
                slice_type=ttnn.Conv2dSliceHeight,
                num_slices=2,
            )

    def __str__(self) -> str:
        return f"Conv: {self.weights.shape} {self.bias.shape if self.has_bias else ''} {self.kernel_size}"

    def __call__(self, device, input_tensor):
        conv_config = ttnn.Conv2dConfig(
            weights_dtype=self.weights_dtype,
            shard_layout=self.shard_layout,
            deallocate_activation=self.deallocate,
            activation=self.activation,
            # reshard_if_not_optimal=True,
            output_layout=self.output_layout,
        )
        compute_config = ttnn.init_device_compute_kernel_config(
            device.arch(),
            # TODO(mbezulj): explore fidelity/fp32 settings. affects frontend, latents, scores.
            math_fidelity=ttnn.MathFidelity.HiFi4,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )
        if self.act_block_h is not None:
            conv_config.act_block_h_override = self.act_block_h

        [output_tensor, [out_h, out_w], [self.weights, self.bias]] = ttnn.conv2d(
            input_tensor=input_tensor,
            weight_tensor=self.weights,
            bias_tensor=self.bias if self.has_bias else None,
            in_channels=self.conv_pt.in_channels,
            out_channels=self.conv_pt.out_channels,
            device=device,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            batch_size=self.conv_pt.batch_size,
            input_height=self.conv_pt.input_height,
            input_width=self.conv_pt.input_width,
            conv_config=conv_config,
            compute_config=compute_config,
            return_output_dim=True,
            return_weights_and_bias=True,
            slice_config=self.slice_config,
        )
        # logger.debug(f"Output tensor shape: {output_tensor.shape}, out_h: {out_h}, out_w: {out_w}")
        # logger.debug(
        #     f"Output tensor dtype: {output_tensor.dtype}, layout: {output_tensor.layout}, memory config: {output_tensor.memory_config}"
        # )
        return output_tensor, out_h, out_w


class GroupNorm:
    def __init__(self, parameters, num_groups, channels, eps=1e-5, dtype=ttnn.bfloat16, is_sliced=False):
        self.weight = parameters.weight
        self.bias = parameters.bias
        self.num_groups = num_groups
        self.channels = channels
        self.eps = eps
        self.dtype = dtype
        self.is_sliced = is_sliced
        self.num_splited_groups = num_groups
        self.num_splited_channels = channels

    def __call__(self, device, input_tensor, H, W, shard="HS", num_splits=1):
        compute_grid = device.compute_with_storage_grid_size()
        grid_size = ttnn.CoreGrid(y=compute_grid.y, x=compute_grid.x)
        grid_y = grid_size.y
        grid_x = grid_size.x
        logger.debug(f"{grid_x=}, {grid_y=}, {shard=}, {num_splits=} {self.is_sliced=}")
        # spliting tensor into multiple splits for very large tensors

        if shard == "HS":
            grid_x *= grid_y
            grid_y = 1

        # Generate input mask
        input_mask_tensor = ttnn.create_group_norm_input_mask(self.channels, self.num_groups, grid_y)
        input_mask_tensor = ttnn.from_torch(
            input_mask_tensor,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        # Generate gamma/beta tensors
        gamma = ttnn.create_group_norm_weight_bias_rm(self.weight, self.channels, grid_y)
        beta = ttnn.create_group_norm_weight_bias_rm(self.bias, self.channels, grid_y)

        gamma_t = ttnn.from_torch(
            gamma,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        beta_t = ttnn.from_torch(
            beta,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        # Generate shard config
        grid_coord = ttnn.CoreCoord(grid_size.x - 1, grid_size.y - 1)
        shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), grid_coord)})
        if shard == "HS":
            # logger.debug(f"Shard height: {H}, width: {W}, grid_size: {grid_size}")
            shard_shape = (H * W) // grid_size.x // grid_size.y, self.channels
            # logger.debug(f"Shard shape: {shard_shape}")
            shard_spec = ttnn.ShardSpec(shard_grid, shard_shape, ttnn.ShardOrientation.ROW_MAJOR)
            sharded_mem_config = ttnn.MemoryConfig(
                ttnn.types.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.types.BufferType.L1, shard_spec
            )
        elif shard == "BS":
            shard_shape = (H * W) // grid_size.x, self.channels // grid_size.y
            shard_spec = ttnn.ShardSpec(shard_grid, shard_shape, ttnn.ShardOrientation.COL_MAJOR)
            sharded_mem_config = ttnn.MemoryConfig(
                ttnn.types.TensorMemoryLayout.BLOCK_SHARDED, ttnn.types.BufferType.L1, shard_spec
            )
        # logger.debug(
        #     f"input tensor shape: {input_tensor.shape}, layout: {input_tensor.layout} memory config: {input_tensor.memory_config}"
        # )
        input_tensor = ttnn.to_memory_config(input_tensor, memory_config=sharded_mem_config)
        tt_output_tensor = ttnn.group_norm(
            input_tensor,
            num_groups=self.num_groups,
            input_mask=input_mask_tensor,
            weight=gamma_t,
            bias=beta_t,
            memory_config=sharded_mem_config,
            core_grid=grid_size,
            epsilon=1e-5,
            # inplace=False,
        )
        return tt_output_tensor


class GroupNormDRAM:
    def __init__(self, parameters, num_groups, channels, eps=1e-5, dtype=ttnn.bfloat16, is_sliced=False):
        self.weight = parameters.weight
        self.bias = parameters.bias
        self.num_groups = num_groups
        self.channels = channels
        self.eps = eps
        self.dtype = dtype
        self.is_sliced = is_sliced
        self.num_splited_groups = num_groups
        self.num_splited_channels = channels

    def __call__(self, device, input_tensor, H, W, shard="HS", num_splits=1):
        compute_grid = device.compute_with_storage_grid_size()
        grid_x, grid_y = compute_grid.x, compute_grid.y
        logger.debug(f"DRAM {grid_x=}, {grid_y=}, {shard=}, {num_splits=} {self.is_sliced=}")
        if num_splits > 4:
            grid_y = 2

        grid_x = 4
        grid_size = ttnn.CoreGrid(y=grid_y, x=grid_x)

        # torch input tensor
        unpadded_shape = input_tensor.shape
        out_shape = [
            unpadded_shape[0],
            unpadded_shape[1],
            _nearest_32_per_core(unpadded_shape[2], grid_x),
            _nearest_32_per_core(unpadded_shape[3], grid_y),
        ]
        logger.debug(f"unpadded_shape: {unpadded_shape} out_shape: {out_shape}")
        input_tensor_tilized = ttnn.tilize_with_val_padding(
            input_tensor, output_tensor_shape=out_shape, pad_value=0, use_multicore=True
        )
        logger.debug(
            f"input_tensor_tilized shape: {input_tensor_tilized.shape} padded shape: {input_tensor_tilized.padded_shape}"
        )
        [gamma_t, beta_t], input_mask_tensor = ttnn.dram_group_norm_params_from_torch(
            [self.weight, self.bias], self.channels, self.num_groups, device, core_grid=grid_size, return_mask=True
        )

        # groupnorm
        logger.debug(f"DRAM {grid_size=}")
        output_tensor = ttnn.group_norm(
            input_tensor_tilized,
            num_groups=self.num_groups,
            input_mask=input_mask_tensor,
            weight=gamma_t,
            bias=beta_t,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            output_layout=ttnn.TILE_LAYOUT,
            core_grid=grid_size,
            inplace=False,
            num_out_blocks=num_splits,
            epsilon=1e-5,
        )

        # ttnn.synchronize_device(device)
        return output_tensor


class GroupNorm_fallback:
    def __init__(self, parameters, num_groups, channels, eps, dtype, is_sliced=False):
        import torch.nn as nn

        self.gn = nn.GroupNorm(num_groups, channels, eps=eps)
        self.gn.weight = nn.Parameter(parameters.weight.clone())
        self.gn.bias = nn.Parameter(parameters.bias.clone())

    def __call__(self, device, input_tensor, H, W, shard="HS", num_splits=1):
        torch_input_nchw = ttnn.to_torch(input_tensor, dtype=torch.float32).permute(0, 3, 1, 2)
        torch_output_nhwc = self.gn(torch_input_nchw).permute(0, 2, 3, 1)
        tt_output = ttnn.from_torch(torch_output_nhwc, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        return tt_output


class Conv_fallback:
    def __init__(
        self,
        parameters,
        conv_pt,
        *,
        stride=1,
        padding=1,
        has_bias=False,
        act_block_h=32,
        dtype=ttnn.bfloat16,
        activation=None,
        output_layout=ttnn.TILE_LAYOUT,
        **kwargs,
    ):
        import torch.nn as nn

        self.out_channels = conv_pt.out_channels
        self.kernel_size = (parameters.weight.shape[2], parameters.weight.shape[3])
        self.stride = conv_pt.stride
        self.padding = conv_pt.padding
        self.activation = activation
        self.output_layout = output_layout
        self.dtype = dtype
        self.parameters = parameters
        self.conv_pt = conv_pt

        # Create PyTorch Conv2d
        self.conv = nn.Conv2d(
            in_channels=conv_pt.in_channels,
            out_channels=conv_pt.out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            bias=has_bias,
        )

        # Set weights
        self.conv.weight = nn.Parameter(ttnn.to_torch(parameters.weight, dtype=torch.float32))
        if has_bias:
            self.conv.bias = nn.Parameter(ttnn.to_torch(parameters.bias, dtype=torch.float32))

        # Handle activation
        if self.activation == "relu":
            self.act_fn = nn.ReLU()
        else:
            self.act_fn = nn.Identity()

    def __call__(self, device, input_tensor):
        # Convert NHWC to NCHW for PyTorch
        torch_input_nchw = (
            ttnn.to_torch(input_tensor, dtype=torch.float32)
            .permute(0, 3, 1, 2)
            .reshape(
                self.conv_pt.batch_size, self.conv_pt.in_channels, self.conv_pt.input_height, self.conv_pt.input_width
            )
        )

        # Apply convolution and activation
        torch_output = self.conv(torch_input_nchw)
        torch_output = self.act_fn(torch_output)

        # Convert back from NCHW to NHWC for TTNN
        torch_output_nhwc = torch_output.permute(0, 2, 3, 1)  # .reshape(1, 1, -1, self.out_channels)

        # Calculate output dimensions
        out_h = torch_output.shape[2]
        out_w = torch_output.shape[3]

        # Convert to TTNN tensor
        tt_output = ttnn.from_torch(torch_output_nhwc, device=device, dtype=self.dtype, layout=self.output_layout)

        return tt_output, out_h, out_w


class Linear:
    def __init__(self, linear_weight, linear_bias, linear_pt, dtype=ttnn.bfloat16) -> None:
        self.linear_weight = linear_weight
        self.linear_bias = linear_bias
        self.output_dtype = dtype

        self.nhw = linear_pt["nhw"]
        self.height_sharding = linear_pt["height_sharding"]
        self.nhw = linear_pt["nhw"]
        self.in_ch = linear_pt["in_channels"]
        self.out_ch = linear_pt["out_channels"]

        # Set sharding layout based on configuration
        if self.height_sharding:
            self.shard_layout = ttnn.TensorMemoryLayout.HEIGHT_SHARDED
        else:
            self.shard_layout = ttnn.TensorMemoryLayout.BLOCK_SHARDED

    def get_program_config(self, device):
        compute_grid = device.compute_with_storage_grid_size()
        self.core_grid = ttnn.CoreGrid(y=compute_grid.y, x=compute_grid.x)
        total_cores = self.core_grid.x * self.core_grid.y

        if self.shard_layout == ttnn.TensorMemoryLayout.HEIGHT_SHARDED:
            self.per_core_M = math.ceil((self.nhw // ttnn.TILE_SIZE) / total_cores)
            self.per_core_N = math.ceil(self.out_ch // ttnn.TILE_SIZE)

            self.in0_block_w = 8
            self.out_subblock = infer_out_subblock(self.per_core_M, self.per_core_N)
            self.out_block = (self.per_core_M, self.per_core_N)

            self.shard_height = self.per_core_M * ttnn.TILE_SIZE
            self.in_shard_width = (self.in_ch + ttnn.TILE_SIZE - 1) // ttnn.TILE_SIZE * ttnn.TILE_SIZE
            self.out_shard_width = self.per_core_N * ttnn.TILE_SIZE

            return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                compute_with_storage_grid_size=(self.core_grid.x, self.core_grid.y),
                in0_block_w=self.in0_block_w,
                out_subblock_h=self.out_subblock[0],
                out_subblock_w=self.out_subblock[1],
                per_core_M=self.per_core_M,
                per_core_N=self.per_core_N,
                out_block_h=self.out_block[0],
                out_block_w=self.out_block[1],
                fuse_batch=True,
                fused_activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
                mcast_in0=False,
            )
        elif self.shard_layout == ttnn.TensorMemoryLayout.BLOCK_SHARDED:
            self.per_core_M = math.ceil((self.nhw // ttnn.TILE_SIZE) / self.core_grid.y)
            self.per_core_N = math.ceil((self.out_ch // ttnn.TILE_SIZE) / self.core_grid.x)

            self.in0_block_w = 4

            self.out_subblock = infer_out_subblock(self.per_core_M, self.per_core_N)
            self.out_block = (self.per_core_M, self.per_core_N)

            self.shard_height = self.per_core_M * ttnn.TILE_SIZE
            self.in_shard_width = (
                (self.in_ch // self.core_grid.x + ttnn.TILE_SIZE - 1) // ttnn.TILE_SIZE * ttnn.TILE_SIZE
            )
            self.out_shard_width = self.per_core_N * ttnn.TILE_SIZE

            print(
                f"Linear BLOCK_SHARDED: {self.per_core_M=}, {self.per_core_N=}, {self.out_subblock=}, {self.shard_height=}, {self.in_shard_width=}, {self.out_shard_width=}"
            )

            return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                compute_with_storage_grid_size=(self.core_grid.x, self.core_grid.y),
                in0_block_w=self.in0_block_w,
                out_subblock_h=self.out_subblock[0],
                out_subblock_w=self.out_subblock[1],
                per_core_M=self.per_core_M,
                per_core_N=self.per_core_N,
                out_block_h=self.out_block[0],
                out_block_w=self.out_block[1],
                fused_activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
                transpose_mcast=False,
            )
        else:
            return None

    def __call__(self, input_tensor, device):
        self.device = device

        program_config = self.get_program_config(device)

        if self.shard_layout is ttnn.TensorMemoryLayout.BLOCK_SHARDED:
            memory_config = ttnn.create_sharded_memory_config(
                (self.shard_height, self.in_shard_width),
                self.core_grid,
                ttnn.ShardStrategy.HEIGHT
                if self.shard_layout == ttnn.TensorMemoryLayout.HEIGHT_SHARDED
                else ttnn.ShardStrategy.BLOCK,
                ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
            )
            input_tensor = ttnn.reshard(input_tensor, output_memory_config=memory_config)
            print(f"Resharded input tensor to {memory_config}")

        memory_config = ttnn.create_sharded_memory_config(
            (self.shard_height, self.out_shard_width),
            self.core_grid,
            ttnn.ShardStrategy.HEIGHT
            if self.shard_layout == ttnn.TensorMemoryLayout.HEIGHT_SHARDED
            else ttnn.ShardStrategy.BLOCK,
            ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )

        compute_config = ttnn.init_device_compute_kernel_config(
            self.device.arch(),
            math_fidelity=ttnn.MathFidelity.LoFi,
            math_approx_mode=True,
            fp32_dest_acc_en=False,
            packer_l1_acc=False,
            dst_full_sync_en=False,
        )

        output_tensor = ttnn.linear(
            input_tensor,
            self.linear_weight,
            bias=self.linear_bias,
            program_config=program_config,
            memory_config=memory_config,
            dtype=self.output_dtype,
            compute_kernel_config=compute_config,
        )

        return output_tensor
