"""
This is the patch merger implementation used in the Qwen-VL-7B model.

There's no existing implementation for this in tt_transformers,
so it was written specifically based on Qwen-VL's architecture.
"""

import ttnn
from models.experimental.qwen25_vl.tt.rmsnorm import RMSNorm
from models.tt_transformers.tt.model_config import ModelArgs


class TTQwen2_5_VLPatchMerger:
    def __init__(
        self,
        device,
        dim,
        state_dict,
        weight_key,
        layer_num=None,
        state_dict_prefix="",
        weight_cache_path=None,
        weight_memory_config=ttnn.DRAM_MEMORY_CONFIG,
        weight_dtype=ttnn.bfloat16,
        is_distributed=None,
        eps: float = 1e-06,
        dims=3584,
        context_dim=1280,
        spatial_merge_size=2,
        mode="decode",
    ):
        super().__init__()
        self.eps = eps
        self.mode = mode

        tt_model_args = ModelArgs(
            device,
            max_batch_size=1,
            max_seq_len=128,
        )

        weight_name_1 = f"{state_dict_prefix}{weight_key}ln_q.weight"
        weight_name_2 = f"{state_dict_prefix}{weight_key}feed_forward.0.weight"
        weight_name_3 = f"{state_dict_prefix}{weight_key}feed_forward.2.weight"

        bias_name_2 = f"{state_dict_prefix}{weight_key}feed_forward.0.bias"
        bias_name_3 = f"{state_dict_prefix}{weight_key}feed_forward.2.bias"

        self.weight_1 = ttnn.as_tensor(
            state_dict[weight_name_1],
            device=device,
            dtype=weight_dtype,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=weight_memory_config,
        )

        self.weight_2 = ttnn.as_tensor(
            state_dict[weight_name_2],
            device=device,
            dtype=weight_dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=weight_memory_config,
        )

        self.weight_3 = ttnn.as_tensor(
            state_dict[weight_name_3],
            device=device,
            dtype=weight_dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=weight_memory_config,
        )

        self.hidden_size = context_dim * (spatial_merge_size**2)
        self.ln_q = RMSNorm(
            device=device,
            dim=1280,
            state_dict=state_dict,
            state_dict_prefix="",
            weight_key="visual.merger.ln_q",
            weight_dtype=ttnn.bfloat16,
            is_distributed=False,
            sharded_program_config=tt_model_args.get_model_config()["SHARDED_NORM_ATTN_PRGM_CFG"],
            sharded_output_config=False,
        )

        self.weight_3 = ttnn.transpose(self.weight_3, 0, 1)

        self.weight_2 = ttnn.transpose(self.weight_2, 0, 1)

        self.compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
            dst_full_sync_en=False,
        )

    def __call__(self, x):
        x = self.ln_q(x, mode=self.mode)

        x = ttnn.reshape(x, (-1, self.hidden_size))

        x = ttnn.linear(
            x,
            self.weight_2,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_config,
        )
        x = ttnn.gelu(x)

        x = ttnn.linear(
            x,
            self.weight_3,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_config,
        )

        return x
