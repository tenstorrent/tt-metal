"""TTNN implementation of ResnetBlock for VocosBackbone prior_net and post_net.

Architecture: GroupNorm(32) -> swish -> Conv1d(k=3) -> GroupNorm(32) -> swish -> Conv1d(k=3) + residual.

All ops on device:
- ttnn.group_norm with height-sharded L1 input (no host roundtrip)
- ttnn.silu for swish
- ttnn.conv1d for convolutions
"""

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.inworld_tts.tt.model_config import RESNET_NUM_GROUPS, VOCOS_DIM


def _prepare_group_norm_params(weight_1d, bias_1d, num_channels, num_groups, device):
    """Prepare GroupNorm weight, bias, and input_mask for ttnn.group_norm.

    Uses height-sharded mode (num_cores_across_channel=1) following the
    official ttnn.group_norm docs. Uses block-sharded mode (8 x-cores) for
    better parallelism — 1.7x faster than height-sharded (2 cores) for
    channels=1024, groups=32.

    Returns:
        (weight_tt, bias_tt, mask_tt) on device
    """
    # Block-sharded: num_cores_across_channel = num_x_cores
    # For channels=1024, groups=32: 8 x-cores gives 128 channels per core
    num_x_cores = 8
    gamma = ttnn.create_group_norm_weight_bias_rm(
        input_tensor=weight_1d, num_channels=num_channels, num_cores_x=num_x_cores
    )
    beta = ttnn.create_group_norm_weight_bias_rm(
        input_tensor=bias_1d, num_channels=num_channels, num_cores_x=num_x_cores
    )
    mask = ttnn.create_group_norm_input_mask(
        num_channel=num_channels,
        num_groups=num_groups,
        num_cores_across_channel=num_x_cores,
        data_type=ttnn.bfloat8_b,
    )

    weight_tt = ttnn.from_torch(
        gamma,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    bias_tt = ttnn.from_torch(
        beta,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    mask_tt = ttnn.to_device(mask, device, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    return weight_tt, bias_tt, mask_tt


class TtResnetBlock(LightweightModule):
    """ResnetBlock: norm1->swish->conv1->norm2->swish->conv2 + residual.

    All ops on device:
    - GroupNorm via ttnn.group_norm with block-sharded L1 input (8x2 grid)
    - SiLU via ttnn.silu
    - Conv1d via ttnn.conv1d
    """

    def __init__(
        self,
        device,
        state_dict,
        block_prefix,
        channels=VOCOS_DIM,
        num_groups=RESNET_NUM_GROUPS,
        dtype=ttnn.bfloat16,
    ):
        super().__init__()
        self.device = device
        self.channels = channels
        self.num_groups = num_groups

        # GroupNorm weights prepared for ttnn.group_norm (height-sharded, on device)
        norm1_w = state_dict[block_prefix + "norm1.weight"].to(torch.bfloat16).to(torch.float32)
        norm1_b = state_dict[block_prefix + "norm1.bias"].to(torch.bfloat16).to(torch.float32)
        norm2_w = state_dict[block_prefix + "norm2.weight"].to(torch.bfloat16).to(torch.float32)
        norm2_b = state_dict[block_prefix + "norm2.bias"].to(torch.bfloat16).to(torch.float32)

        self.norm1_weight, self.norm1_bias, self.norm1_mask = _prepare_group_norm_params(
            norm1_w, norm1_b, channels, num_groups, device
        )
        self.norm2_weight, self.norm2_bias, self.norm2_mask = _prepare_group_norm_params(
            norm2_w, norm2_b, channels, num_groups, device
        )

        # Cache sharded config per T (lazily populated in forward)
        self._gn_config_cache = {}

        # Conv1d weights as host tensors for ttnn.conv1d
        conv1_w = state_dict[block_prefix + "conv1.weight"].to(torch.bfloat16).to(torch.float32)
        conv1_b = state_dict[block_prefix + "conv1.bias"].to(torch.bfloat16).to(torch.float32)
        conv2_w = state_dict[block_prefix + "conv2.weight"].to(torch.bfloat16).to(torch.float32)
        conv2_b = state_dict[block_prefix + "conv2.bias"].to(torch.bfloat16).to(torch.float32)

        self.conv1_weight = ttnn.from_torch(conv1_w, dtype=ttnn.float32)
        self.conv1_bias = ttnn.from_torch(conv1_b.reshape(1, 1, 1, channels), dtype=ttnn.float32)
        self.conv2_weight = ttnn.from_torch(conv2_w, dtype=ttnn.float32)
        self.conv2_bias = ttnn.from_torch(conv2_b.reshape(1, 1, 1, channels), dtype=ttnn.float32)

        # Conv config
        self.conv_config = ttnn.Conv1dConfig(
            weights_dtype=ttnn.float32,
            deallocate_activation=True,
            shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            act_block_h_override=32,
        )
        self.compute_config = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            fp32_dest_acc_en=True,
        )

        # Cache device conv weights
        self._conv1_dw = None
        self._conv1_db = None
        self._conv2_dw = None
        self._conv2_db = None

    def _get_gn_config(self, T):
        """Get (or compute and cache) the block-sharded memory config + grid for GroupNorm.

        The config depends on the spatial size T, so we cache per T value.
        Uses block sharding (is_height_sharded=False) for better core utilization.
        """
        if T not in self._gn_config_cache:
            sharded_mem_config, grid_size = ttnn.determine_expected_group_norm_sharded_config_and_grid_size(
                device=self.device,
                num_channels=self.channels,
                num_groups=self.num_groups,
                input_nhw=1 * T * 1,  # N=1, H=T, W=1
                is_height_sharded=False,
                is_row_major=True,
            )
            self._gn_config_cache[T] = (sharded_mem_config, grid_size)
        return self._gn_config_cache[T]

    def _group_norm_device(self, x_tt, weight, bias, mask, T):
        """GroupNorm on device via ttnn.group_norm with block-sharded L1 input.

        Args:
            x_tt: [1, 1, T, C] ttnn tensor (any layout/memory)
            weight, bias, mask: prepared GroupNorm params on device
            T: sequence length
        Returns:
            [1, 1, T, C] ttnn tensor in TILE_LAYOUT, DRAM
        """
        sharded_mem_config, grid_size = self._get_gn_config(T)

        # Move input to block-sharded L1 in ROW_MAJOR
        x_rm = ttnn.to_layout(x_tt, ttnn.ROW_MAJOR_LAYOUT)
        x_sharded = ttnn.to_memory_config(x_rm, sharded_mem_config)

        out = ttnn.group_norm(
            x_sharded,
            num_groups=self.num_groups,
            input_mask=mask,
            weight=weight,
            bias=bias,
            memory_config=sharded_mem_config,
            core_grid=grid_size,
            compute_kernel_config=self.compute_config,
        )

        # Move back to interleaved L1 TILE_LAYOUT for downstream ops
        out = ttnn.to_memory_config(out, ttnn.L1_MEMORY_CONFIG)
        out = ttnn.to_layout(out, ttnn.TILE_LAYOUT)
        return out

    def _run_conv1d(self, x_nhwc, weight, bias, dw_attr, db_attr, T):
        """Run Conv1d(k=3, padding=1) on [1, 1, T, C] NHWC input.

        Args:
            x_nhwc: [1, 1, T, C] ttnn tensor (channels-last, NHWC format)
        Returns:
            [1, 1, T, C] ttnn tensor in TILE_LAYOUT
        """
        w = getattr(self, dw_attr) or weight
        b = getattr(self, db_attr) or bias

        result = ttnn.conv1d(
            input_tensor=x_nhwc,
            weight_tensor=w,
            in_channels=self.channels,
            out_channels=self.channels,
            device=self.device,
            bias_tensor=b,
            kernel_size=3,
            stride=1,
            padding=1,
            batch_size=1,
            input_length=T,
            dtype=ttnn.bfloat16,
            conv_config=self.conv_config,
            compute_config=self.compute_config,  # fp32_dest_acc_en=True
            groups=1,
            return_output_dim=True,
            return_weights_and_bias=True,
        )
        output_tensor, out_length, [wd, bd] = result
        setattr(self, dw_attr, wd)
        setattr(self, db_attr, bd)

        # Output is [1, 1, out_length, C] sharded -> interleaved L1
        output_tensor = ttnn.sharded_to_interleaved(output_tensor, ttnn.L1_MEMORY_CONFIG)
        return output_tensor  # [1, 1, T, C]

    def forward(self, x):
        """Forward pass. All ops on device: GroupNorm, SiLU, Conv1d.

        Args:
            x: [1, 1, T, C] ttnn tensor in TILE_LAYOUT
        Returns:
            [1, 1, T, C] ttnn tensor in TILE_LAYOUT
        """
        residual = x
        T = x.shape[2]

        # GroupNorm 1 (on device, height-sharded L1)
        h = self._group_norm_device(x, self.norm1_weight, self.norm1_bias, self.norm1_mask, T)

        # SiLU on device
        h = ttnn.silu(h)

        # Conv1d 1: input is [1, 1, T, C] TILE -> need ROW_MAJOR for conv1d
        h = ttnn.to_layout(h, ttnn.ROW_MAJOR_LAYOUT)
        h = self._run_conv1d(h, self.conv1_weight, self.conv1_bias, "_conv1_dw", "_conv1_db", T)
        h = ttnn.to_layout(h, ttnn.TILE_LAYOUT)

        # GroupNorm 2 (on device, height-sharded L1)
        h = self._group_norm_device(h, self.norm2_weight, self.norm2_bias, self.norm2_mask, T)

        # SiLU on device
        h = ttnn.silu(h)

        # Conv1d 2
        h = ttnn.to_layout(h, ttnn.ROW_MAJOR_LAYOUT)
        h = self._run_conv1d(h, self.conv2_weight, self.conv2_bias, "_conv2_dw", "_conv2_db", T)
        h = ttnn.to_layout(h, ttnn.TILE_LAYOUT)

        # Residual on device
        return ttnn.add(residual, h)
