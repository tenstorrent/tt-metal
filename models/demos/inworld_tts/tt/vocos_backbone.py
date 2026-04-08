"""TTNN implementation of VocosBackbone.

Pipeline: embed Conv1d -> 2x ResnetBlock (prior_net) -> 12 TransformerBlocks ->
          2x ResnetBlock (post_net) -> final LayerNorm.

ALL ops run on device (TTNN). No PyTorch fallback except for the initial
input conversion and the embed Conv1d (which uses ttnn.conv1d).
Data stays in channels-last [1, 1, T, C] TILE_LAYOUT throughout the backbone.
"""

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.inworld_tts.tt.model_config import (
    VOCOS_DEPTH,
    VOCOS_DIM,
    VOCOS_HEADS,
    VOCOS_POS_EMB_DIM,
    get_compute_kernel_config_hifi4,
)
from models.demos.inworld_tts.tt.resnet_block import TtResnetBlock
from models.demos.inworld_tts.tt.transformer_block import TtTransformerBlock


class TtVocosBackbone(LightweightModule):
    """VocosBackbone: Conv embed -> ResnetBlocks -> TransformerBlocks -> ResnetBlocks -> LayerNorm.

    All ops on device. Data format: [1, 1, T, C] TILE_LAYOUT (channels-last).
    """

    def __init__(
        self,
        device,
        state_dict,
        dim=VOCOS_DIM,
        depth=VOCOS_DEPTH,
        n_heads=VOCOS_HEADS,
        pos_emb_dim=VOCOS_POS_EMB_DIM,
        dtype=ttnn.bfloat16,
        state_dict_prefix="",
    ):
        super().__init__()
        self.device = device
        self.dim = dim
        self.depth = depth

        prefix = state_dict_prefix

        # Embed Conv1d(1024, 1024, k=7, padding=3) weights
        embed_w = state_dict[prefix + "embed.weight"].to(torch.bfloat16).to(torch.float32)
        embed_b = state_dict[prefix + "embed.bias"].to(torch.bfloat16).to(torch.float32)
        self.embed_weight = ttnn.from_torch(embed_w, dtype=ttnn.float32)
        self.embed_bias = ttnn.from_torch(embed_b.reshape(1, 1, 1, dim), dtype=ttnn.float32)
        self._embed_device_weight = None
        self._embed_device_bias = None

        self.embed_conv_config = ttnn.Conv1dConfig(
            weights_dtype=ttnn.float32,
            deallocate_activation=True,
            shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            act_block_h_override=32,
        )
        self.embed_compute_config = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            fp32_dest_acc_en=True,
        )

        # Prior net: 2x ResnetBlock
        self.prior_net = []
        for i in range(2):
            block = TtResnetBlock(
                device=device,
                state_dict=state_dict,
                block_prefix=f"{prefix}prior_net.{i}.",
                channels=dim,
                dtype=dtype,
            )
            self.prior_net.append(block)

        # Transformer blocks
        self.transformers = []
        for i in range(depth):
            block = TtTransformerBlock(
                device=device,
                state_dict=state_dict,
                layer_num=i,
                dim=dim,
                n_heads=n_heads,
                pos_emb_dim=pos_emb_dim,
                dtype=dtype,
                state_dict_prefix=prefix,
            )
            self.transformers.append(block)

        # Post net: 2x ResnetBlock
        self.post_net = []
        for i in range(2):
            block = TtResnetBlock(
                device=device,
                state_dict=state_dict,
                block_prefix=f"{prefix}post_net.{i}.",
                channels=dim,
                dtype=dtype,
            )
            self.post_net.append(block)

        # Final LayerNorm weights -- [1, 1, dim//32, 32] in ROW_MAJOR
        self.final_ln_weight = ttnn.from_torch(
            state_dict[prefix + "final_layer_norm.weight"].reshape(1, 1, dim // 32, 32),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self.final_ln_bias = ttnn.from_torch(
            state_dict[prefix + "final_layer_norm.bias"].reshape(1, 1, dim // 32, 32),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        self.compute_kernel_config = get_compute_kernel_config_hifi4()

    def _run_embed_conv(self, x_nhwc, T):
        """Embed Conv1d(1024, 1024, k=7, padding=3).

        Args:
            x_nhwc: [1, 1, T, C] ttnn tensor in ROW_MAJOR (NHWC format)
        Returns:
            [1, 1, T, C] ttnn tensor
        """
        w = self._embed_device_weight if self._embed_device_weight is not None else self.embed_weight
        b = self._embed_device_bias if self._embed_device_bias is not None else self.embed_bias

        result = ttnn.conv1d(
            input_tensor=x_nhwc,
            weight_tensor=w,
            in_channels=self.dim,
            out_channels=self.dim,
            device=self.device,
            bias_tensor=b,
            kernel_size=7,
            stride=1,
            padding=3,
            batch_size=1,
            input_length=T,
            dtype=ttnn.bfloat16,
            conv_config=self.embed_conv_config,
            compute_config=self.embed_compute_config,
            groups=1,
            return_output_dim=True,
            return_weights_and_bias=True,
        )

        output_tensor, out_length, [weights_device, bias_device] = result
        self._embed_device_weight = weights_device
        self._embed_device_bias = bias_device

        # Output is [1, 1, out_length, C] sharded -> interleaved L1
        output_tensor = ttnn.sharded_to_interleaved(output_tensor, ttnn.L1_MEMORY_CONFIG)
        return output_tensor  # [1, 1, T, C]

    def forward(self, x):
        """Forward pass.

        Args:
            x: [B, T, C] torch tensor or [1, 1, T, C] ttnn tensor
        Returns:
            [1, 1, T, C] ttnn tensor in TILE_LAYOUT
        """
        # Convert input to device if needed
        if isinstance(x, torch.Tensor):
            if x.dim() == 3:
                x = x.unsqueeze(0)  # [1, 1, T, C]
            x = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device)

        T = x.shape[2]
        C = x.shape[3]

        # Embed Conv1d: [1, 1, T, C] TILE -> ROW_MAJOR for conv -> back to TILE
        h = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)  # [1, 1, T, C] NHWC
        h = self._run_embed_conv(h, T)  # [1, 1, T, C]
        h = ttnn.to_layout(h, ttnn.TILE_LAYOUT)

        # Prior net: 2x ResnetBlock (all on device)
        for block in self.prior_net:
            h = block(h)

        # 12 Transformer blocks (float32 activations for precision)
        for block in self.transformers:
            h = block(h)

        # Activations are already bf16 (no float32 typecast in transformer blocks)

        # Post net: 2x ResnetBlock (all on device)
        for block in self.post_net:
            h = block(h)

        # Final LayerNorm -> L1
        output = ttnn.layer_norm(
            h,
            weight=self.final_ln_weight,
            bias=self.final_ln_bias,
            epsilon=1e-6,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_config,
        )

        return output
