"""TTNN implementation of ResnetBlock for VocosBackbone prior_net and post_net.

Architecture: GroupNorm(32) -> swish -> Conv1d(k=3) -> GroupNorm(32) -> swish -> Conv1d(k=3) + residual.

All ops on device:
- GroupNorm via host roundtrip (single bf16 conversion per block, not per op)
- ttnn.silu for swish
- ttnn.conv1d for convolutions
"""

import torch
import torch.nn.functional as F

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.inworld_tts.tt.model_config import RESNET_NUM_GROUPS, VOCOS_DIM


class TtResnetBlock(LightweightModule):
    """ResnetBlock: norm1->swish->conv1->norm2->swish->conv2 + residual.

    Conv1d runs natively on device via ttnn.conv1d.
    SiLU runs natively on device via ttnn.silu.
    GroupNorm uses a single host roundtrip per norm (unavoidable without complex
    sharding setup) but no extra bf16 quantization since data stays in bf16 throughout.
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

        # GroupNorm weights (CPU, for host-side norm)
        self.norm1_weight = state_dict[block_prefix + "norm1.weight"].to(torch.bfloat16).to(torch.float32)
        self.norm1_bias = state_dict[block_prefix + "norm1.bias"].to(torch.bfloat16).to(torch.float32)
        self.norm2_weight = state_dict[block_prefix + "norm2.weight"].to(torch.bfloat16).to(torch.float32)
        self.norm2_bias = state_dict[block_prefix + "norm2.bias"].to(torch.bfloat16).to(torch.float32)

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

    def _group_norm_host(self, x_tt, weight, bias):
        """GroupNorm via host roundtrip. Stays in bf16 to avoid requantization loss.

        Args:
            x_tt: [1, 1, T, C] ttnn bfloat16 tensor
        Returns:
            [1, 1, T, C] ttnn bfloat16 tensor
        """
        x_torch = ttnn.to_torch(x_tt)  # bf16 from device
        # Reshape for F.group_norm: needs [N, C, *] format
        h = x_torch.squeeze(0).permute(0, 2, 1)  # [1, C, T] bf16
        # Run GroupNorm in bf16 to avoid requantization (float32 GroupNorm + bf16 cast adds error)
        h = F.group_norm(h, self.num_groups, weight.to(h.dtype), bias.to(h.dtype), eps=1e-6)
        # Back to [1, 1, T, C]
        h = h.permute(0, 2, 1).unsqueeze(0)
        return ttnn.from_torch(h, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device)

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

        # Output is [1, 1, out_length, C] sharded -> interleaved TILE
        output_tensor = ttnn.sharded_to_interleaved(output_tensor, ttnn.DRAM_MEMORY_CONFIG)
        return output_tensor  # [1, 1, T, C]

    def forward(self, x):
        """Forward pass. Conv1d and SiLU on device, GroupNorm via host roundtrip.

        Args:
            x: [1, 1, T, C] ttnn tensor in TILE_LAYOUT
        Returns:
            [1, 1, T, C] ttnn tensor in TILE_LAYOUT
        """
        residual = x
        T = x.shape[2]

        # GroupNorm 1 (host roundtrip -- no extra bf16 loss since data is already bf16)
        h = self._group_norm_host(x, self.norm1_weight, self.norm1_bias)

        # SiLU on device
        h = ttnn.silu(h)

        # Conv1d 1: input is [1, 1, T, C] TILE -> need ROW_MAJOR for conv1d
        h = ttnn.to_layout(h, ttnn.ROW_MAJOR_LAYOUT)
        h = self._run_conv1d(h, self.conv1_weight, self.conv1_bias, "_conv1_dw", "_conv1_db", T)
        h = ttnn.to_layout(h, ttnn.TILE_LAYOUT)

        # GroupNorm 2
        h = self._group_norm_host(h, self.norm2_weight, self.norm2_bias)

        # SiLU on device
        h = ttnn.silu(h)

        # Conv1d 2
        h = ttnn.to_layout(h, ttnn.ROW_MAJOR_LAYOUT)
        h = self._run_conv1d(h, self.conv2_weight, self.conv2_bias, "_conv2_dw", "_conv2_db", T)
        h = ttnn.to_layout(h, ttnn.TILE_LAYOUT)

        # Residual on device
        return ttnn.add(residual, h)
