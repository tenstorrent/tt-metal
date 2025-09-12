# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.experimental.swin_s.tt.common import StridedConv

try:
    from tracy import signpost

    use_signpost = True
except ModuleNotFoundError:
    use_signpost = False

program_configs = {
    "linear_config_1": ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=(8, 8),
        in0_block_w=1,
        out_subblock_h=1,
        out_subblock_w=3,
        per_core_M=2,
        per_core_N=6,
        fuse_batch=True,
        fused_activation=None,
        mcast_in0=False,
    ),
    "linear_config_2": ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=(8, 8),
        in0_block_w=1,
        out_subblock_h=1,
        out_subblock_w=3,
        per_core_M=4,
        per_core_N=12,
        transpose_mcast=False,
        fused_activation=None,
    ),
}


class TtPatchMerging:
    def __init__(self, device, parameters, dim):
        self.dim = dim
        self.device = device
        self.parameters = parameters

        out_channels = [96, 192, 384]
        positions = {
            "tl": (0, 0),
            "tr": (0, 1),
            "bl": (1, 0),
            "br": (1, 1),
        }
        for out_channel in out_channels:
            for name, (r, c) in positions.items():
                key = f"conv_{out_channel}_weights_{name}"
                if out_channel == 384:
                    conv = StridedConv(
                        [2, 2, 0, 0],
                        parameters=parameters[f"conv_{out_channel}_weights_{name}"],
                        groups=out_channel,
                        shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
                    )
                else:
                    conv = StridedConv(
                        [2, 2, 0, 0],
                        parameters=parameters[f"conv_{out_channel}_weights_{name}"],
                        reshard=True,
                        groups=out_channel,
                    )
                setattr(self, f"conv_{out_channel}_{name}", conv)

    def __call__(self, input_tensor):
        if use_signpost:
            signpost(header="patchmerging")
        _, H, W, _ = input_tensor.shape
        input_tensor = ttnn.pad(input_tensor, input_tensor.shape, [0, 0, 0, 0], 0)

        channel = input_tensor.shape[-1]
        if channel == 96:
            input_tensor_0 = self.conv_96_tl(self.device, input_tensor)
            input_tensor_1 = self.conv_96_bl(self.device, input_tensor)
            input_tensor_2 = self.conv_96_tr(self.device, input_tensor)
            input_tensor_3 = self.conv_96_br(self.device, input_tensor)
        elif channel == 192:
            input_tensor_0 = self.conv_192_tl(self.device, input_tensor)
            input_tensor_1 = self.conv_192_bl(self.device, input_tensor)
            input_tensor_2 = self.conv_192_tr(self.device, input_tensor)
            input_tensor_3 = self.conv_192_br(self.device, input_tensor)
        elif channel == 384:
            input_tensor_0 = self.conv_384_tl(self.device, input_tensor)
            input_tensor_1 = self.conv_384_bl(self.device, input_tensor)
            input_tensor_2 = self.conv_384_tr(self.device, input_tensor)
            input_tensor_3 = self.conv_384_br(self.device, input_tensor)

        input_tensor_0 = ttnn.sharded_to_interleaved(input_tensor_0, ttnn.L1_MEMORY_CONFIG)
        input_tensor_1 = ttnn.sharded_to_interleaved(input_tensor_1, ttnn.L1_MEMORY_CONFIG)
        input_tensor_2 = ttnn.sharded_to_interleaved(input_tensor_2, ttnn.L1_MEMORY_CONFIG)
        input_tensor_3 = ttnn.sharded_to_interleaved(input_tensor_3, ttnn.L1_MEMORY_CONFIG)

        output_tensor = ttnn.concat(
            [input_tensor_0, input_tensor_1, input_tensor_2, input_tensor_3], -1, memory_config=ttnn.L1_MEMORY_CONFIG
        )
        output_tensor = ttnn.to_layout(output_tensor, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG)
        output_tensor = ttnn.layer_norm(
            output_tensor,
            weight=self.parameters.norm["weight"],
            bias=self.parameters.norm["bias"],
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        ttnn.deallocate(input_tensor_0)
        ttnn.deallocate(input_tensor_1)
        ttnn.deallocate(input_tensor_2)
        ttnn.deallocate(input_tensor_3)
        if output_tensor.shape[-1] == 384:
            output_tensor = ttnn.to_memory_config(
                output_tensor,
                memory_config=ttnn.create_sharded_memory_config(
                    output_tensor.shape,
                    core_grid=ttnn.CoreGrid(y=8, x=8),
                    strategy=ttnn.ShardStrategy.HEIGHT,
                    orientation=ttnn.ShardOrientation.ROW_MAJOR,
                ),
                dtype=ttnn.bfloat16,
            )

            output_tensor = ttnn.linear(
                output_tensor,
                self.parameters.reduction["weight"],
                memory_config=ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG,
                dtype=ttnn.bfloat16,
                compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                    math_fidelity=ttnn.MathFidelity.LoFi,
                ),
                program_config=program_configs["linear_config_1"],
            )
        elif output_tensor.shape[-1] == 768:
            output_tensor = ttnn.to_memory_config(
                output_tensor,
                memory_config=ttnn.create_sharded_memory_config(
                    output_tensor.shape,
                    core_grid=ttnn.CoreGrid(y=8, x=8),
                    strategy=ttnn.ShardStrategy.BLOCK,
                    orientation=ttnn.ShardOrientation.ROW_MAJOR,
                ),
                dtype=ttnn.bfloat16,
            )
            output_tensor = ttnn.linear(
                output_tensor,
                self.parameters.reduction["weight"],
                memory_config=ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG,
                dtype=ttnn.bfloat16,
                compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                    math_fidelity=ttnn.MathFidelity.LoFi,
                ),
                program_config=program_configs["linear_config_2"],
            )
        else:
            output_tensor = ttnn.linear(
                output_tensor,
                self.parameters.reduction["weight"],
                dtype=ttnn.bfloat16,
                compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                    math_fidelity=ttnn.MathFidelity.LoFi,
                ),
                core_grid=ttnn.CoreGrid(y=8, x=8),
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )

        output_tensor = ttnn.to_memory_config(output_tensor, ttnn.L1_MEMORY_CONFIG, dtype=ttnn.bfloat16)
        return output_tensor
