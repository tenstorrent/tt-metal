# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
TTNN Swin-L backbone — standalone, reusable.
Multi-scale output: returns feature maps for selected stages after per-stage norms.

Architecture (Swin-L defaults):
  PatchEmbed(4x4 conv, stride 4) -> 4 stages:
    Stage 0: 2 blocks, dim=192,  heads=6,  window=12 -> C2
    Stage 1: 2 blocks, dim=384,  heads=12, window=12 -> C3
    Stage 2: 18 blocks, dim=768, heads=24, window=12 -> C4
    Stage 3: 2 blocks, dim=1536, heads=48, window=12 -> C5
  Downsample (PatchMerge) between stages 0->1, 1->2, 2->3.
  Per-stage output norms applied before returning feature maps.

Use `out_indices` to control which stages produce outputs:
  DINO-5scale: out_indices=(0, 1, 2, 3) -> 4 feature maps
  ATSS-DyHead: out_indices=(1, 2, 3)    -> 3 feature maps

Input: [B, 3, H, W] in NCHW (converted to NHWC internally).
Output: list of len(out_indices) tensors in NCHW.
"""

import ttnn
from models.experimental.swin_l.tt.tt_swin_block import TtSwinBlock
from models.experimental.swin_l.tt.tt_swin_patch_merge import TtSwinPatchMerge
from tests.ttnn.ttnn_utility_fuction import get_shard_grid_from_num_cores


class TtSwinLBackbone:
    """TTNN Swin-L backbone producing multi-scale feature maps for selected stages."""

    def __init__(
        self,
        device,
        parameters,
        embed_dim=192,
        depths=(2, 2, 18, 2),
        num_heads=(6, 12, 24, 48),
        window_size=12,
        mlp_ratio=4.0,
        attn_masks=None,
        out_indices=(0, 1, 2, 3),
    ):
        self.device = device
        self.parameters = parameters
        self.embed_dim = embed_dim
        self.depths = depths
        self.num_heads = num_heads
        self.window_size = [window_size, window_size] if isinstance(window_size, int) else list(window_size)
        self.mlp_ratio = mlp_ratio
        self.out_indices = out_indices

        # Patch embedding conv (4x4 stride 4)
        self.patch_embed_weight = parameters["patch_embed"]["projection"]["weight"]
        self.patch_embed_bias = parameters["patch_embed"]["projection"]["bias"]
        self.patch_embed_weight = ttnn.from_device(self.patch_embed_weight)
        self.patch_embed_bias = ttnn.from_device(self.patch_embed_bias)

        # Build stages: each stage = list of TtSwinBlock
        self.stages = []
        for s in range(4):
            dim = embed_dim * (2**s)
            heads = num_heads[s]
            blocks = []
            for b in range(depths[s]):
                shift = [0, 0] if b % 2 == 0 else [self.window_size[0] // 2, self.window_size[1] // 2]
                mask = attn_masks[s] if attn_masks is not None else None
                blocks.append(
                    TtSwinBlock(
                        device,
                        parameters["stages"][s]["blocks"][b],
                        dim=dim,
                        num_heads=heads,
                        window_size=self.window_size,
                        shift_size=shift,
                        mlp_ratio=mlp_ratio,
                        attn_mask=mask,
                    )
                )
            self.stages.append(blocks)

        # Downsamples between stages 0->1, 1->2, 2->3
        self.downsamples = []
        for s in range(3):
            dim = embed_dim * (2**s)
            self.downsamples.append(TtSwinPatchMerge(device, parameters["stages"][s]["downsample"], dim=dim))

    def __call__(self, input_tensor):
        """
        input_tensor: [B, 3, H, W] NCHW on device.
        Returns: list of len(out_indices) NCHW feature maps.
        """
        N, C, H, W = input_tensor.shape
        patch_size = 4

        # Pad input to multiple of patch_size (mmdet PatchEmbed uses padding='corner')
        pad_h = (patch_size - H % patch_size) % patch_size
        pad_w = (patch_size - W % patch_size) % patch_size
        min_channels = 16
        pad_c = max(0, min_channels - C)
        if pad_h > 0 or pad_w > 0 or pad_c > 0:
            nchw = ttnn.pad(
                input_tensor,
                ((0, 0), (0, pad_c), (0, pad_h), (0, pad_w)),
                value=0.0,
            )
        else:
            nchw = input_tensor
        nhwc = ttnn.permute(nchw, (0, 2, 3, 1))
        ttnn.deallocate(nchw)
        nhwc = ttnn.reallocate(nhwc)

        # Patch embedding: 4x4 conv stride 4
        conv_config = ttnn.Conv2dConfig(
            weights_dtype=ttnn.bfloat16,
            shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            deallocate_activation=True,
            enable_act_double_buffer=True,
            enable_weights_double_buffer=True,
            enable_kernel_stride_folding=False,
        )
        shard_grid = get_shard_grid_from_num_cores(64, self.device)
        conv_config.core_grid = shard_grid
        conv_config.override_sharding_config = True

        compute_config = ttnn.init_device_compute_kernel_config(
            self.device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi2,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
            math_approx_mode=False,
        )

        [output, [out_h, out_w], [self.patch_embed_weight, self.patch_embed_bias]] = ttnn.conv2d(
            input_tensor=nhwc,
            weight_tensor=self.patch_embed_weight,
            bias_tensor=self.patch_embed_bias,
            in_channels=nhwc.shape[3],
            out_channels=self.embed_dim,
            device=self.device,
            kernel_size=(4, 4),
            stride=(4, 4),
            padding=(0, 0),
            batch_size=N,
            input_height=nhwc.shape[1],
            input_width=nhwc.shape[2],
            conv_config=conv_config,
            compute_config=compute_config,
            return_output_dim=True,
            return_weights_and_bias=True,
            dtype=ttnn.bfloat16,
        )
        output = ttnn.sharded_to_interleaved(output, ttnn.DRAM_MEMORY_CONFIG)
        output = ttnn.reshape(output, (N, out_h, out_w, self.embed_dim))

        # Post-embed layer norm
        output = ttnn.to_layout(output, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        output = ttnn.layer_norm(
            output,
            weight=self.parameters["patch_embed"]["norm"]["weight"],
            bias=self.parameters["patch_embed"]["norm"]["bias"],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # Run 4 stages, collecting features only for stages in out_indices
        features = []
        for s in range(4):
            # Run all blocks in this stage
            for block in self.stages[s]:
                output = block(output)

            # Only collect output for stages in out_indices
            if s in self.out_indices:
                normed = ttnn.layer_norm(
                    output,
                    weight=self.parameters[f"norm{s}"]["weight"],
                    bias=self.parameters[f"norm{s}"]["bias"],
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
                # NHWC -> NCHW
                feat_nchw = ttnn.permute(normed, (0, 3, 1, 2), memory_config=ttnn.DRAM_MEMORY_CONFIG)
                features.append(feat_nchw)
                ttnn.deallocate(normed)

            # Downsample (except after last stage)
            if s < 3:
                output = self.downsamples[s](output)

        return features
