"""
This is the ImageTransformer block for Gemma-3-4b-it.
gemma attention/MLP components and additional multi-device communication
"""

# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.demos.gemma3.tt.gemma_image_attention import TtGemmaImageAttention
from models.demos.gemma3.tt.gemma_image_mlp import TtGemmaImageFeedForward
from models.tt_transformers.tt.multimodal.llama_image_block import TtLlamaImageTransformerBlock


def create_gemma_block_forward():
    """
    Creates gemma-specific forward function that adds multi-device communication.
    """

    def gemma_forward_wrapper(llama_block):
        def gemma_forward(x_11SH, mask=None):
            seq_len = x_11SH.shape[-2]
            # Should this be % 128 following gemma_image_transformer.py?
            assert seq_len % 32 == 0 and seq_len > 0, "Seqlen must be divisible by 32"

            attn_out = llama_block.attn(llama_block.ln_1(x_11SH), mask=mask)
            if llama_block.gated:
                attn_out = ttnn.mul(attn_out, ttnn.tanh(llama_block.gate_attn))

            # Gemma-specific: Additional multi-device communication
            if llama_block.num_devices > 1:
                attn_out = ttnn.experimental.all_gather_async(
                    attn_out,
                    persistent_output_buffer=None,
                    dim=3,
                    multi_device_global_semaphore=llama_block.tt_ccl.get_and_cycle_ag_semaphore_handles(),
                    num_links=1,
                    topology=ttnn.Topology.Linear,
                    barrier_semaphore=llama_block.tt_ccl.get_and_cycle_barrier_semaphore_handle(),
                    chunks_per_sync=10,
                    num_workers_per_link=2,
                    num_buffers_per_channel=2,
                )
            res = ttnn.add(x_11SH, attn_out)

            mlp_out = llama_block.mlp(llama_block.ln_2(res))
            if llama_block.gated:
                mlp_out = ttnn.mul(mlp_out, ttnn.tanh(llama_block.gate_ffn))
            out = ttnn.add(res, mlp_out)

            ttnn.deallocate(mlp_out)
            ttnn.deallocate(attn_out)
            ttnn.deallocate(res)
            return out

        return gemma_forward

    return gemma_forward_wrapper


class TtGemmaImageTransformerBlock:
    def __init__(
        self,
        mesh_device,
        state_dict,
        tt_ccl,
        state_dict_prefix,
        weight_cache_path,
        dtype,
        configuration,
        gated=False,
    ):
        self.llama_block = TtLlamaImageTransformerBlock(
            mesh_device=mesh_device,
            tt_ccl=tt_ccl,
            state_dict=state_dict,
            state_dict_prefix=state_dict_prefix,
            weight_cache_path=weight_cache_path,
            dtype=dtype,
            configuration=configuration,
            gated=gated,
        )

        self.llama_block.attn = TtGemmaImageAttention(
            mesh_device=mesh_device,
            tt_ccl=tt_ccl,
            state_dict=state_dict,
            state_dict_prefix=f"{state_dict_prefix}attn.",
            weight_cache_path=weight_cache_path,
            dtype=dtype,
            configuration=configuration,
        )

        self.llama_block.mlp = TtGemmaImageFeedForward(
            mesh_device=mesh_device,
            tt_ccl=tt_ccl,
            args=configuration,
            state_dict=state_dict,
            state_dict_prefix=f"{state_dict_prefix}mlp.",
            weight_cache_path=weight_cache_path,
            dtype=dtype,
        )

        # Replace forward method with gemma-specific version
        forward_wrapper = create_gemma_block_forward()
        self.llama_block.forward = forward_wrapper(self.llama_block)

    def forward(self, x_11SH, mask=None):
        return self.llama_block.forward(x_11SH, mask)

    def __call__(self, x_11SH, mask=None):
        return self.forward(x_11SH, mask)
