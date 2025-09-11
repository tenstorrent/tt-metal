"""
This is the ImageTransformer block for Gemma-3-4b-it.
We have reused the TtLlamaImageTransformerBlock with incorporating the
TtGemmaImageAttention and TtGemmaImageFeedForward
"""

# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import tracy

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.gemma3.tt.gemma_image_attention import TtGemmaImageAttention

# from models.tt_transformers.tt.multimodal.llama_image_attention import TtLlamaImageAttention
from models.demos.gemma3.tt.gemma_image_mlp import TtGemmaImageFeedForward
from models.tt_transformers.tt.multimodal.llama_layernorm import TtLayerNorm


class TtGemmaImageTransformerBlock(LightweightModule):
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
        super().__init__()

        self.state_dict = state_dict
        self.mesh_device = mesh_device
        self.tt_ccl = tt_ccl
        self.num_devices = configuration.num_devices
        self.hidden_size = configuration.vision_dim
        self.gated = gated

        self.ln_1 = TtLayerNorm(
            device=mesh_device,
            dim=configuration.vision_dim,
            state_dict=state_dict,
            state_dict_prefix=f"{state_dict_prefix}ln_1.",
            weight_cache_path=weight_cache_path,
            weight_dtype=dtype,
            eps=configuration.norm_eps,
        )

        self.attn = TtGemmaImageAttention(
            mesh_device=mesh_device,
            tt_ccl=tt_ccl,
            state_dict=state_dict,
            state_dict_prefix=f"{state_dict_prefix}attn.",
            weight_cache_path=weight_cache_path,
            dtype=dtype,
            configuration=configuration,
        )

        self.ln_2 = TtLayerNorm(
            device=mesh_device,
            dim=configuration.vision_dim,
            state_dict=state_dict,
            state_dict_prefix=f"{state_dict_prefix}ln_2.",
            weight_cache_path=weight_cache_path,
            weight_dtype=dtype,
            eps=configuration.norm_eps,
        )

        self.mlp = TtGemmaImageFeedForward(
            mesh_device=mesh_device,
            tt_ccl=tt_ccl,
            args=configuration,
            state_dict=state_dict,
            state_dict_prefix=f"{state_dict_prefix}mlp.",
            weight_cache_path=weight_cache_path,
            dtype=dtype,
        )

        if gated:
            # Gate tensors must be expanded to hidden dim or we get a PCC error
            self.gate_attn = ttnn.as_tensor(
                state_dict[f"{state_dict_prefix}gate_attn"].unsqueeze(0).expand(1, self.hidden_size),
                dtype=ttnn.bfloat16,
                device=self.mesh_device,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            self.gate_ffn = ttnn.as_tensor(
                state_dict[f"{state_dict_prefix}gate_ffn"].unsqueeze(0).expand(1, self.hidden_size),
                dtype=ttnn.bfloat16,
                device=self.mesh_device,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

    def forward(self, x_11SH, mask=None):
        seq_len = x_11SH.shape[-2]
        assert seq_len % 32 == 0 and seq_len > 0, "Seqlen must be divisible by 32"

        print("Calling gemma attention, shape: ", x_11SH.shape)
        print("Calling gemma attention, tensor: ", x_11SH)
        # Replicated tensor
        attn_out = self.attn(self.ln_1(x_11SH), mask=mask)
        if self.gated:
            attn_out = ttnn.mul(attn_out, ttnn.tanh(self.gate_attn))

        if self.num_devices > 1:
            print("Calling AG")
            print("Attn out shape pre AG before MLP: ", attn_out.shape)
            tracy.signpost("AG async in between begin")
            # attn_out = ttnn.to_layout(attn_out, ttnn.ROW_MAJOR_LAYOUT)
            # attn_out = ttnn.reshape(attn_out, [1, 1, 1, -1])
            # print("Post new reshape: ", attn_out.shape)
            # print("Post new reshape tensor: ", attn_out.padded_shape)
            attn_out = ttnn.experimental.all_gather_async(
                attn_out,
                persistent_output_buffer=None,
                dim=1,
                multi_device_global_semaphore=self.tt_ccl.get_and_cycle_ag_semaphore_handles(),
                num_links=1,
                topology=ttnn.Topology.Ring,
                # topology=ttnn.Topology.Linear,
                barrier_semaphore=self.tt_ccl.get_and_cycle_barrier_semaphore_handle(),
                chunks_per_sync=10,
                num_workers_per_link=2,
                num_buffers_per_channel=2,
            )
            tracy.signpost("AG async in between end")
            print("AG done pre MLP")
            print("Attn out shape post AG before MLP: ", attn_out.shape)
            print("Attn out tensor post AG before MLP: ", attn_out)
            # attn_out = ttnn.to_layout(attn_out, ttnn.ROW_MAJOR_LAYOUT)
            # attn_out =  ttnn.experimental.nlp_concat_heads(attn_out)

            attn_out = ttnn.untilize(attn_out)
            attn_out = ttnn.transpose(attn_out, 1, 2)
            attn_out = ttnn.reshape(attn_out, [1, 1, seq_len, -1])
            attn_out = ttnn.tilize(attn_out)
            # attn_out = ttnn.to_layout(attn_out, ttnn.TILE_LAYOUT)
            print("Attn out shape post reshape: ", attn_out.shape)
            print("Attn out tensor post reshape padded shape: ", attn_out.padded_shape)
        res = ttnn.add(x_11SH, attn_out)

        print("Calling gemma MLP, shape: ", res.shape)
        mlp_out = self.mlp(self.ln_2(res))
        if self.gated:
            mlp_out = ttnn.mul(mlp_out, ttnn.tanh(self.gate_ffn))
        out = ttnn.add(res, mlp_out)

        ttnn.deallocate(mlp_out)
        ttnn.deallocate(attn_out)
        ttnn.deallocate(res)
        return out
