import ttnn
import torch
import torch.nn.functional as F
from loguru import logger

try:
    from tracy import signpost

    use_signpost = True
except ModuleNotFoundError:
    use_signpost = False

from models.experimental.oft.reference.oft import EPSILON
from models.experimental.oft.reference.utils import perspective


def calculate_per_core_dims(n, h, w, in_ch, out_ch, core_grid, sharding_strategy):
    nhw = n * h * w
    if sharding_strategy == "height":
        per_core_M = (nhw // (core_grid.y * core_grid.x) + ttnn.TILE_SIZE - 1) // ttnn.TILE_SIZE
        per_core_N = (out_ch + ttnn.TILE_SIZE - 1) // ttnn.TILE_SIZE
    else:
        per_core_M = (nhw // core_grid.y + ttnn.TILE_SIZE - 1) // ttnn.TILE_SIZE
        per_core_N = (out_ch // core_grid.x + ttnn.TILE_SIZE - 1) // ttnn.TILE_SIZE
    return per_core_M, per_core_N


def calculate_shard_dims(per_core_M, per_core_N, in_ch, out_ch, core_grid, sharding_strategy):
    if sharding_strategy == "height":
        shard_height = per_core_M * ttnn.TILE_SIZE
        in_shard_width = (in_ch + ttnn.TILE_SIZE - 1) // ttnn.TILE_SIZE * ttnn.TILE_SIZE
        out_shard_width = (out_ch + ttnn.TILE_SIZE - 1) // ttnn.TILE_SIZE * ttnn.TILE_SIZE
    else:
        shard_height = per_core_M * ttnn.TILE_SIZE
        in_shard_width = (in_ch // core_grid.x + ttnn.TILE_SIZE - 1) // ttnn.TILE_SIZE * ttnn.TILE_SIZE
        out_shard_width = per_core_N * ttnn.TILE_SIZE
    return shard_height, in_shard_width, out_shard_width


def get_matmul_config(core_grid, in0_block_w, out_subblock, per_core_M, per_core_N, out_block, sharding_strategy):
    if sharding_strategy == "height":
        return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=(core_grid.x, core_grid.y),
            in0_block_w=in0_block_w,
            out_subblock_h=out_subblock[0],
            out_subblock_w=out_subblock[1],
            per_core_M=per_core_M,
            per_core_N=per_core_N,
            out_block_h=out_block[0],
            out_block_w=out_block[1],
            fuse_batch=True,
            fused_activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            mcast_in0=False,
        )
    else:
        return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(core_grid.x, core_grid.y),
            in0_block_w=in0_block_w,
            out_subblock_h=out_subblock[0],
            out_subblock_w=out_subblock[1],
            per_core_M=per_core_M,
            per_core_N=per_core_N,
            out_block_h=out_block[0],
            out_block_w=out_block[1],
            fused_activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            transpose_mcast=False,
        )


def calculate_initialization_parameters(
    device, channels, cell_size, grid_height, feature_shape_hw, calib, grid, scale, use_precomputed_grid
):
    y_corners = torch.arange(0, grid_height, cell_size) - grid_height / 2.0
    y_corners = F.pad(y_corners.view(-1, 1, 1, 1), [1, 1])
    # Expand the grid in the y dimension
    corners = grid.unsqueeze(1) + y_corners.view(-1, 1, 1, 3)

    # Project grid corners to image plane and normalize to [-1, 1]
    img_corners = perspective(calib.view(-1, 1, 1, 1, 3, 4), corners, dtype=torch.float32)
    feature_height, feature_width = feature_shape_hw
    # Normalize to [-1, 1]
    img_size = corners.new([feature_width, feature_height]) / scale
    norm_corners = (2 * img_corners / img_size - 1).clamp(-1, 1)
    # Get top-left and bottom-right coordinates of voxel bounding boxes
    bbox_corners = torch.cat(
        [
            torch.min(norm_corners[:, :-1, :-1, :-1], norm_corners[:, :-1, 1:, :-1]),
            torch.max(norm_corners[:, 1:, 1:, 1:], norm_corners[:, 1:, :-1, 1:]),
        ],
        dim=-1,
    )
    batch, height, depth, width, _ = bbox_corners.size()
    bbox_corners = bbox_corners.flatten(2, 3).permute(0, 2, 1, 3)
    area = (
        (bbox_corners[..., 2:] - bbox_corners[..., :2]).prod(dim=-1) * feature_height * feature_width * 0.25 + EPSILON
    ).unsqueeze(1)
    visible = area > EPSILON

    area = 1 / area
    area_nhwc = (
        area.permute(0, 2, 3, 1)
        .repeat(1, 1, 1, channels)
        .reshape(1, depth * width, 1, channels * height)
        .permute(0, 2, 1, 3)
    )  # Convert to N1HW*C format
    visible_nhwc = (
        visible.permute(0, 2, 3, 1)
        .repeat(1, 1, 1, channels)
        .reshape(1, depth * width, 1, channels * height)
        .permute(0, 2, 1, 3)
    )  # Convert to N1HW*C format
    top_left_bc = bbox_corners[..., [0, 1]]
    top_left_bc = torch.nn.functional.pad(top_left_bc, ((0, 0, 0, 0, 0, 63, 0, 0)), value=0.0)
    btm_right_bc = bbox_corners[..., [2, 3]]
    btm_right_bc = torch.nn.functional.pad(btm_right_bc, ((0, 0, 0, 0, 0, 63, 0, 0)), value=0.0)
    top_right_bc = bbox_corners[..., [2, 1]]
    top_right_bc = torch.nn.functional.pad(top_right_bc, ((0, 0, 0, 0, 0, 63, 0, 0)), value=0.0)
    btm_left_bc = bbox_corners[..., [0, 3]]
    btm_left_bc = torch.nn.functional.pad(btm_left_bc, ((0, 0, 0, 0, 0, 63, 0, 0)), value=0.0)

    batch_size, grid_h, grid_w, _ = top_left_bc.shape
    input_shape_nhwc = [batch_size, feature_height, feature_width, channels]

    # reshape
    if use_precomputed_grid:
        prepare_grid_lambda = lambda torch_grid_bf16, input_shape_nhwc: ttnn.to_device(
            ttnn.prepare_grid_sample_grid(
                ttnn.from_torch(torch_grid_bf16, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.float32),
                input_shape_nhwc,
                padding_mode="zeros",
                output_dtype=ttnn.bfloat16,
            ),
            device,
        )

        top_left_bc_tt = prepare_grid_lambda(top_left_bc, input_shape_nhwc)
        btm_right_bc_tt = prepare_grid_lambda(btm_right_bc, input_shape_nhwc)
        top_right_bc_tt = prepare_grid_lambda(top_right_bc, input_shape_nhwc)
        btm_left_bc_tt = prepare_grid_lambda(btm_left_bc, input_shape_nhwc)

    else:
        top_left_bc_tt = ttnn.from_torch(top_left_bc, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
        btm_right_bc_tt = ttnn.from_torch(
            btm_right_bc, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
        )
        top_right_bc_tt = ttnn.from_torch(
            top_right_bc, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
        )
        btm_left_bc_tt = ttnn.from_torch(btm_left_bc, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

    top_left_bc_tt = ttnn.reshape(top_left_bc_tt, [batch_size, grid_h, 1, grid_w * top_left_bc_tt.shape[-1]])
    btm_right_bc_tt = ttnn.reshape(btm_right_bc_tt, [batch_size, grid_h, 1, grid_w * btm_right_bc_tt.shape[-1]])
    top_right_bc_tt = ttnn.reshape(top_right_bc_tt, [batch_size, grid_h, 1, grid_w * top_right_bc_tt.shape[-1]])
    btm_left_bc_tt = ttnn.reshape(btm_left_bc_tt, [batch_size, grid_h, 1, grid_w * btm_left_bc_tt.shape[-1]])

    visible_tt = ttnn.from_torch(visible_nhwc, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    area = torch.nn.functional.pad(area_nhwc * visible_nhwc, ((0, 0, 0, 63, 0, 0, 0, 0)), value=0.0)
    area_tt = ttnn.from_torch(area, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    return (
        [top_left_bc_tt, btm_right_bc_tt, top_right_bc_tt, btm_left_bc_tt],
        visible_tt,
        area_tt,
        [batch, height, depth, width],
    )
    # return area_tt


class OFT:
    """
    Orthographic Feature Transform (OFT) class with refactored configuration management.

    This class separates configuration parameters into:
    1. Sharding configurations for slicing operations (setup in _setup_sharding_configs)
    2. Linear layer configurations (setup in _setup_linear_configs)
    3. Dynamic configurations calculated per forward pass (_calculate_dynamic_configs)

    All linear layer parameters (matmul config, output memory config, compute config)
    are now calculated outside the forward function for better performance and clarity.
    """

    def __init__(
        self,
        device,
        parameters,
        channels,
        cell_size,
        grid_height,
        features_shape_hw,
        calib,
        grid,
        scale,
        use_precomputed_grid,
    ):
        # params for conv3d
        self.linear_weight = parameters.conv3d.weight
        self.linear_bias = parameters.conv3d.bias
        self.scale = scale
        self.use_precomputed_grid = use_precomputed_grid
        self.features_shape_hw = features_shape_hw
        self.device = device

        self.num_slices = 11

        self.bbox_corners, self.visible, self.area, self.shape = calculate_initialization_parameters(
            device, channels, cell_size, grid_height, features_shape_hw, calib, grid, self.scale, use_precomputed_grid
        )

        in_channels = self.linear_weight.shape[0]
        self.linear_weight = ttnn.reshape(self.linear_weight, [in_channels // self.shape[1], self.shape[1], channels])
        self.linear_weight = ttnn.permute(self.linear_weight, (1, 0, 2))
        self.linear_weight = ttnn.reshape(self.linear_weight, [in_channels, channels])

        # integral_image_quantization_strategy
        # None - no quantization
        # "to_uint32" - quantize to uint32 before integral image, dequantize after
        # "to_float32" - quantize to float32 before integral image, dequantize after
        self.integral_image_quantization_strategy = "to_uint32"
        logger.info(f"Integral image quantization strategy: {self.integral_image_quantization_strategy}")
        if self.integral_image_quantization_strategy == "to_uint32":
            self.prescaler = ttnn.from_torch(torch.tensor(1024 * 1024), device=device, dtype=ttnn.bfloat16)
            self.postscaler = ttnn.from_torch(torch.tensor(1 / 1024 / 1024), device=device, dtype=ttnn.bfloat16)

        # Initialize sharding and linear layer configurations
        self._setup_sharding_configs()
        self._setup_linear_configs()

    def _setup_sharding_configs(self):
        """Setup sharding configurations for slicing operations"""
        self.core_grid = ttnn.CoreGrid(y=4, x=4)
        self.sharding_strategy = "block"

        # Sharding parameters for slicing
        self.slice_memory_layout = (
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED
            if self.sharding_strategy == "height"
            else ttnn.TensorMemoryLayout.BLOCK_SHARDED
        )
        self.shard_orientation = ttnn.ShardOrientation.ROW_MAJOR

    def _setup_linear_configs(self):
        """Setup configurations for linear layer operations"""
        # Linear layer parameters
        self.in0_block_w = 2
        self.out_subblock = (1, 2)

        # Data types
        self.input_dtype = ttnn.bfloat16
        self.output_dtype = ttnn.bfloat16
        self.weight_dtype = ttnn.bfloat16

        # Compute configuration
        self.compute_config = ttnn.init_device_compute_kernel_config(
            self.device.arch(),
            math_fidelity=ttnn.MathFidelity.LoFi,
            math_approx_mode=True,
            fp32_dest_acc_en=False,
            packer_l1_acc=False,
            dst_full_sync_en=False,
        )

    def create_sharded_tensor(
        self,
        input_tensor,
        target_shard_height=None,
        target_shard_width=None,
        sharding_strategy=None,
        memory_layout=None,
        use_all_cores=True,
    ):
        """
        Helper function to create a sharded tensor using available cores.

        Args:
            input_tensor: Input tensor to be sharded
            target_shard_height: Target height for each shard (optional, calculated if None)
            target_shard_width: Target width for each shard (optional, calculated if None)
            sharding_strategy: "height", "block", or None (uses self.sharding_strategy)
            memory_layout: Memory layout or None (uses self.slice_memory_layout)
            use_all_cores: Whether to use all available cores (default: True)

        Returns:
            Sharded tensor using all available cores

        Example usage:
            # Basic usage with automatic shard size calculation
            sharded_tensor = self.create_sharded_tensor(my_tensor)

            # Use specific shard dimensions
            sharded_tensor = self.create_sharded_tensor(
                my_tensor,
                target_shard_height=1024,
                target_shard_width=512
            )

            # Use height sharding strategy with all cores
            sharded_tensor = self.create_sharded_tensor(
                my_tensor,
                sharding_strategy="height",
                use_all_cores=True
            )
        """
        # Use class defaults if not specified
        strategy = sharding_strategy if sharding_strategy is not None else self.sharding_strategy
        mem_layout = memory_layout if memory_layout is not None else self.slice_memory_layout

        # Get tensor dimensions
        tensor_shape = input_tensor.shape
        n, h, w, channels = tensor_shape

        # Calculate core grid - use all available cores if requested
        if use_all_cores:
            grid_size = self.device.compute_with_storage_grid_size()
            core_grid = ttnn.CoreGrid(y=grid_size.y, x=grid_size.x)
        else:
            core_grid = self.core_grid

        # Calculate shard dimensions
        if target_shard_height is None or target_shard_width is None:
            if strategy == "height":
                # Height sharding: distribute height across cores
                total_elements = n * h * w
                per_core_elements = (total_elements + (core_grid.y * core_grid.x) - 1) // (core_grid.y * core_grid.x)
                shard_height = ((per_core_elements + ttnn.TILE_SIZE - 1) // ttnn.TILE_SIZE) * ttnn.TILE_SIZE
                shard_width = (channels + ttnn.TILE_SIZE - 1) // ttnn.TILE_SIZE * ttnn.TILE_SIZE
            else:  # block sharding
                # Block sharding: distribute both height and width
                per_core_h = (h + core_grid.y - 1) // core_grid.y
                per_core_w = (w + core_grid.x - 1) // core_grid.x
                shard_height = ((per_core_h + ttnn.TILE_SIZE - 1) // ttnn.TILE_SIZE) * ttnn.TILE_SIZE
                shard_width = ((per_core_w * channels + ttnn.TILE_SIZE - 1) // ttnn.TILE_SIZE) * ttnn.TILE_SIZE
        else:
            shard_height = target_shard_height
            shard_width = target_shard_width

        # Create shard strategy
        shard_strategy = ttnn.ShardStrategy.HEIGHT if strategy == "height" else ttnn.ShardStrategy.BLOCK

        # Create sharded memory configuration
        sharded_mem_config = ttnn.create_sharded_memory_config(
            (shard_height, shard_width),
            core_grid,
            shard_strategy,
            self.shard_orientation,
            use_height_and_width_as_shard_shape=True,
        )

        # Convert to sharded tensor
        if input_tensor.is_sharded():
            # If already sharded, convert to interleaved first
            interleaved_tensor = ttnn.to_memory_config(input_tensor, ttnn.DRAM_MEMORY_CONFIG)
            sharded_tensor = ttnn.to_memory_config(interleaved_tensor, sharded_mem_config)
        else:
            # Convert directly to sharded
            sharded_tensor = ttnn.to_memory_config(input_tensor, sharded_mem_config)

        logger.debug(
            f"Created sharded tensor with shape {sharded_tensor.shape}, "
            f"shard_height: {shard_height}, shard_width: {shard_width}, "
            f"core_grid: {core_grid}, strategy: {strategy}"
        )

        return sharded_tensor

    def create_sharded_slice(
        self,
        input_tensor,
        shard_height,
        shard_width,
        num_slices,
        slice_index,
        core_grid=None,
        sharding_strategy=None,
        memory_layout=None,
    ):
        """
        Helper function to create a sharded slice from an input tensor (similar to interleaved_to_sharded_partial).

        Args:
            input_tensor: Input tensor to be sliced and sharded
            shard_height: Height of each shard
            shard_width: Width of each shard
            num_slices: Total number of slices
            slice_index: Index of the current slice
            core_grid: Core grid to use (optional, uses self.core_grid if None)
            sharding_strategy: Sharding strategy (optional, uses self.sharding_strategy if None)
            memory_layout: Memory layout (optional, uses self.slice_memory_layout if None)

        Returns:
            Sharded slice tensor

        Example usage:
            # Process tensor in slices (e.g., for memory management)
            for i in range(num_slices):
                slice_tensor = self.create_sharded_slice(
                    large_tensor,
                    shard_height=1024,
                    shard_width=512,
                    num_slices=11,
                    slice_index=i
                )
                # Process slice_tensor...
                ttnn.deallocate(slice_tensor)
        """
        # Use class defaults if not specified
        grid = core_grid if core_grid is not None else self.core_grid
        strategy = sharding_strategy if sharding_strategy is not None else self.sharding_strategy
        mem_layout = memory_layout if memory_layout is not None else self.slice_memory_layout

        # Create sharded slice using ttnn function
        sharded_slice = ttnn.interleaved_to_sharded_partial(
            input_tensor,
            (grid.x, grid.y),
            [shard_height, shard_width],
            num_slices,
            slice_index,
            mem_layout,
            self.shard_orientation,
        )

        logger.debug(
            f"Created sharded slice {slice_index}/{num_slices} with "
            f"shard dimensions [{shard_height}, {shard_width}], "
            f"core_grid: {grid}, strategy: {strategy}"
        )

        return sharded_slice

    def _calculate_dynamic_configs(self, n, h, w, in_ch, out_ch):
        """Calculate dynamic configurations based on tensor dimensions"""
        w_sliced = w // self.num_slices

        # Calculate per-core dimensions
        per_core_M, per_core_N = calculate_per_core_dims(
            n, h, w_sliced, in_ch, out_ch, self.core_grid, self.sharding_strategy
        )

        # Calculate shard dimensions
        shard_height, in_shard_width, out_shard_width = calculate_shard_dims(
            per_core_M, per_core_N, in_ch, out_ch, self.core_grid, self.sharding_strategy
        )

        out_block = (per_core_M, per_core_N)

        # Matmul configuration
        matmul_config = get_matmul_config(
            self.core_grid,
            self.in0_block_w,
            self.out_subblock,
            per_core_M,
            per_core_N,
            out_block,
            self.sharding_strategy,
        )

        # Shard strategy
        shard_strategy = ttnn.ShardStrategy.HEIGHT if self.sharding_strategy == "height" else ttnn.ShardStrategy.BLOCK

        # Output memory configuration
        output_mem_config = ttnn.create_sharded_memory_config(
            (shard_height, out_shard_width),
            self.core_grid,
            shard_strategy,
            ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )

        return {
            "per_core_M": per_core_M,
            "per_core_N": per_core_N,
            "shard_height": shard_height,
            "in_shard_width": in_shard_width,
            "out_shard_width": out_shard_width,
            "matmul_config": matmul_config,
            "output_mem_config": output_mem_config,
        }

    def forward(self, device, features, calib, grid):
        if use_signpost:
            signpost(header="OFT block started")

        features = ttnn.reshape(features, [1, self.features_shape_hw[0], self.features_shape_hw[1], -1])
        if features.get_layout() == ttnn.ROW_MAJOR_LAYOUT:
            features = ttnn.to_layout(features, ttnn.TILE_LAYOUT)

        if self.integral_image_quantization_strategy == None:
            integral_image = ttnn_integral_image_channel_last(features)
        elif self.integral_image_quantization_strategy == "to_uint32":
            features = ttnn.mul(features, self.prescaler, dtype=ttnn.bfloat16)
            features = ttnn.typecast(features, ttnn.uint32)
            integral_image = ttnn_integral_image_channel_last(features)
            integral_image = ttnn.typecast(integral_image, ttnn.bfloat16)
            integral_image = ttnn.mul(integral_image, self.postscaler, dtype=ttnn.bfloat16)
        elif self.integral_image_quantization_strategy == "to_float32":
            features = ttnn.typecast(features, ttnn.float32)
            integral_image = ttnn_integral_image_channel_last(features)
            integral_image = ttnn.typecast(integral_image, ttnn.bfloat16)

        if integral_image.get_layout() == ttnn.TILE_LAYOUT:
            integral_image = ttnn.to_layout(integral_image, ttnn.ROW_MAJOR_LAYOUT)

        integral_image = ttnn.to_memory_config(integral_image, ttnn.L1_MEMORY_CONFIG)

        logger.debug(f"Integral image shape: {integral_image.shape}")
        logger.debug(f"Bounding box corners shape: {self.bbox_corners[0].shape}")

        top_left = ttnn.grid_sample(
            integral_image,
            self.bbox_corners[0],
            use_precomputed_grid=self.use_precomputed_grid,
            batch_output_channels=True,
        )

        top_left = ttnn.reshape(top_left, [top_left.shape[0], top_left.shape[2], top_left.shape[1], top_left.shape[3]])
        n, h, w, in_ch = top_left.shape
        # top_left = ttnn.tilize_with_val_padding(top_left, [n, h, w + 63, in_ch], 0.0)
        top_left = ttnn.to_layout(top_left, ttnn.TILE_LAYOUT)

        btm_right = ttnn.grid_sample(
            integral_image,
            self.bbox_corners[1],
            use_precomputed_grid=self.use_precomputed_grid,
            batch_output_channels=True,
        )
        btm_right = ttnn.reshape(
            btm_right, [btm_right.shape[0], btm_right.shape[2], btm_right.shape[1], btm_right.shape[3]]
        )
        # btm_right = ttnn.tilize_with_val_padding(btm_right, [n, h, w + 63, in_ch], 0.0)
        btm_right = ttnn.to_layout(btm_right, ttnn.TILE_LAYOUT)

        top_right = ttnn.grid_sample(
            integral_image,
            self.bbox_corners[2],
            use_precomputed_grid=self.use_precomputed_grid,
            batch_output_channels=True,
        )
        top_right = ttnn.reshape(
            top_right, [top_right.shape[0], top_right.shape[2], top_right.shape[1], top_right.shape[3]]
        )
        # top_right = ttnn.tilize_with_val_padding(top_right, [n, h, w + 63, in_ch], 0.0)
        top_right = ttnn.to_layout(top_right, ttnn.TILE_LAYOUT)

        btm_left = ttnn.grid_sample(
            integral_image,
            self.bbox_corners[3],
            use_precomputed_grid=self.use_precomputed_grid,
            batch_output_channels=True,
        )
        btm_left = ttnn.reshape(btm_left, [btm_left.shape[0], btm_left.shape[2], btm_left.shape[1], btm_left.shape[3]])
        # btm_left = ttnn.tilize_with_val_padding(btm_left, [n, h, w + 63, in_ch], 0.0)
        btm_left = ttnn.to_layout(btm_left, ttnn.TILE_LAYOUT)

        integral_image = ttnn.to_memory_config(integral_image, ttnn.DRAM_MEMORY_CONFIG)

        # Get output channels from linear weight
        out_ch = self.linear_weight.shape[1]

        # Calculate dynamic configurations based on tensor dimensions
        configs = self._calculate_dynamic_configs(n, h, w, in_ch, out_ch)

        logger.debug(f"OFT Block per_core_M: {configs['per_core_M']}, per_core_N: {configs['per_core_N']}")
        logger.debug(
            f"OFT Block shard_height: {configs['shard_height']}, in_shard_width: {configs['in_shard_width']}, out_shard_width: {configs['out_shard_width']}"
        )

        out_initial = torch.randn([n, h, w, out_ch], dtype=torch.float32)
        ortho_feats = ttnn.from_torch(
            out_initial, self.input_dtype, device=device, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )

        for i in range(self.num_slices):
            # Using the new helper function for sharded slicing
            vox_feats_slice = self.create_sharded_slice(
                top_left, configs["shard_height"], configs["in_shard_width"], self.num_slices, i
            )

            top_right_slice = self.create_sharded_slice(
                top_right, configs["shard_height"], configs["in_shard_width"], self.num_slices, i
            )

            ttnn.sub_(vox_feats_slice, top_right_slice)

            ttnn.deallocate(top_right_slice)

            bottom_right_slice = self.create_sharded_slice(
                btm_right, configs["shard_height"], configs["in_shard_width"], self.num_slices, i
            )

            ttnn.add_(vox_feats_slice, bottom_right_slice)
            ttnn.deallocate(bottom_right_slice)

            bottom_left_slice = self.create_sharded_slice(
                btm_left, configs["shard_height"], configs["in_shard_width"], self.num_slices, i
            )

            ttnn.sub_(vox_feats_slice, bottom_left_slice)
            ttnn.deallocate(bottom_left_slice)

            area_slice = self.create_sharded_slice(
                self.area, configs["shard_height"], configs["in_shard_width"], self.num_slices, i
            )

            ttnn.mul_(vox_feats_slice, area_slice)
            ttnn.deallocate(area_slice)

            vox_feats_slice = ttnn.linear(
                vox_feats_slice,
                self.linear_weight,
                bias=self.linear_bias,
                program_config=configs["matmul_config"],
                memory_config=configs["output_mem_config"],
                dtype=self.output_dtype,
                compute_kernel_config=self.compute_config,
            )

            ttnn.sharded_to_interleaved_partial(
                vox_feats_slice, ortho_feats, self.num_slices, i, memory_config=ttnn.DRAM_MEMORY_CONFIG
            )

        ortho_feats = ortho_feats[:, :, : w - 63, :]

        if use_signpost:
            signpost(header="OFT block ended")
        # return ortho_feats
        return (
            ortho_feats,
            integral_image,
            self.bbox_corners[0],
            self.bbox_corners[1],
            self.bbox_corners[2],
            self.bbox_corners[3],
        )


def ttnn_integral_image_channel_last(features_nhwc):
    assert len(features_nhwc.shape) == 4, "Input tensor must be 4D"
    assert features_nhwc.shape[0] == 1, "Batch size must be 1"
    tmp = ttnn.cumsum(features_nhwc, dim=1, dtype=features_nhwc.dtype)
    # ttnn.deallocate(features_nhwc) remove if needed, for now it work without move
    # tmp = ttnn.move(tmp)
    return ttnn.cumsum(tmp, dim=2, dtype=features_nhwc.dtype)
