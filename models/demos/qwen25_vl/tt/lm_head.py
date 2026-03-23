# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import math

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.tt_transformers.tt.ccl import tt_all_reduce


class QwenLMHead(LightweightModule):
    """LM head for Qwen2.5-VL on TG (8,4) mesh.

    Uses ShardTensor2dMesh to shard the weight with K (dim) across columns
    and vocab across rows, matching the 2D hidden-state layout on TG.
    Follows the Llama Galaxy pattern: DRAM interleaved memory for prefill,
    and a simple ttnn.linear + tt_all_reduce forward.
    """

    def __init__(
        self,
        args,
        mesh_device,
        tt_ccl,
        dtype,
        state_dict,
        state_dict_prefix,
        weight_cache_path,
    ):
        super().__init__()
        self.args = args
        self.mesh_device = mesh_device
        self.tt_ccl = tt_ccl
        self.dtype = dtype
        self.vocab_size = args.vocab_size
        self.padded_vocab_size = args.padded_vocab_size if args.padded_vocab_size else args.vocab_size

        tile_size = 32
        self.padded_vocab_size = math.ceil(self.padded_vocab_size / tile_size) * tile_size

        torch_output_weights = state_dict[f"{state_dict_prefix}output.weight"].permute(1, 0)

        if self.vocab_size < self.padded_vocab_size:
            padding_size = self.padded_vocab_size - self.vocab_size
            torch_output_weights = torch.cat(
                [
                    torch_output_weights,
                    torch.zeros(torch_output_weights.shape[0], padding_size, dtype=torch_output_weights.dtype),
                ],
                dim=-1,
            )

        padded_lm_head = torch_output_weights.unsqueeze(0).unsqueeze(0)

        cache_file_name = (
            None if args.dummy_weights else weight_cache_path / f"output_lm_head_tg_2d_{self.padded_vocab_size}"
        )

        self.output_weight = ttnn.as_tensor(
            padded_lm_head,
            device=mesh_device,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(3, 2), mesh_shape=args.cluster_shape),
            layout=ttnn.TILE_LAYOUT,
            dtype=dtype,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cache_file_name=cache_file_name,
        )

        self.compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

    def forward(self, x: ttnn.Tensor):
        output = ttnn.linear(
            x,
            self.output_weight,
            compute_kernel_config=self.compute_kernel_config,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=ttnn.bfloat16,
        )

        ttnn.deallocate(x)

        output = tt_all_reduce(
            output,
            self.mesh_device,
            self.tt_ccl,
            cluster_axis=1,
            dim=3,
            memory_config=output.memory_config(),
            dtype=ttnn.bfloat16,
            sharded=False,
            use_composite=True,
        )

        return output
