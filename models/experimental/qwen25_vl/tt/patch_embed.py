"""
This is the patch embedding implementation for Qwen-VL-7B.

The existing TtLlamaConv2dPatch from tt_transformers uses Conv2d, but Qwen needs Conv3d instead.
Since the stride size is the same as the kernel size for this operation, we can use a matrix
multiplication (matmul) instead of a convolution. This is necessary because
`ttnn.experimental.conv3d` currently only supports Conv3d with stride (1, 1, 1).
"""

import ttnn


class TTQwen2_5_VisionPatchEmbed:
    def __init__(
        self,
        device,
        patch_size,
        temporal_patch_size,
        in_channels,
        embed_dim,
        state_dict,
        weight_key,
        layer_num=None,
        state_dict_prefix="",
        weight_cache_path=None,
        weight_memory_config=ttnn.DRAM_MEMORY_CONFIG,
        weight_dtype=ttnn.bfloat16,
        mode="decode",
    ):
        super().__init__()
        self.mode = mode
        self.device = device
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.weight_memory_config = weight_memory_config
        self.weight_dtype = weight_dtype

        weight_name_1 = f"{state_dict_prefix}{weight_key}proj.weight"
        torch_weight = state_dict[weight_name_1]

        weight_matrix = torch_weight.view(self.embed_dim, -1)
        self.weight = ttnn.from_torch(
            weight_matrix.T,
            device=self.device,
            dtype=self.weight_dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=self.weight_memory_config,
        )
        self.compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
            dst_full_sync_en=False,
        )

    def __call__(self, x: ttnn.Tensor) -> ttnn.Tensor:
        x_flattened = ttnn.reshape(x, (x.shape[2], -1))
        output = ttnn.matmul(x_flattened, self.weight, compute_kernel_config=self.compute_kernel_config)

        return output
