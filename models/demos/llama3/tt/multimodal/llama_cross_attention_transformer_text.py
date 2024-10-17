# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import os
import math
import ttnn
import torch
import torch.nn as nn
from models.demos.llama3.tt.llama_decoder import TtTransformerBlock
from models.demos.llama3.tt.multimodal.llama_cross_block import TtLlamaCrossAttentionTransformerBlock
from models.demos.llama3.tt.llama_model import LMHead
from models.common.rmsnorm import RMSNorm
import ttnn
from typing import Optional
from models.common.lightweightmodule import LightweightModule
from models.demos.llama3.tt.llama_embedding import TtLlamaEmbedding


def _get_full_row_masked_out_mask(
    attn_bias,
    negative_inf_value,
):
    """
    attn_bias should be a 4D tensor of shape [B, H, S1, S2]
    where B is the batch size, H is the number of heads,
    and S1/S2 are the sequence lengths. This returns
    a 4D tensor of shape [B, H, S1, 1] which stores boolean
    values which are 0 if the a full row in the last dimension
    contains negative infinity values, otherwise it's 1.
    """
    return (attn_bias != negative_inf_value).any(dim=-1).type_as(attn_bias)[..., None]


class TtLlamaCrossAttentionTransformerText(LightweightModule):
    def __init__(
        self,
        mesh_device,
        state_dict,
        state_dict_prefix,
        weight_cache_path,
        dtype,
        configuration,
    ):
        super().__init__()
        self.vocab_size = configuration.vocab_size
        assert self.vocab_size > 0
        self.n_layers = configuration.n_layers
        self.mesh_device = mesh_device
        self.dtype = dtype
        self.model_config = configuration.get_model_config()
        self.grid_size = configuration.max_grid_size
        state_dict_prefix = configuration.get_state_dict_prefix("", None)
        self.configuration = configuration
        self.model_config = configuration.get_model_config()
        self.state_dict = state_dict

        # self.tok_embeddings = TtLlamaEmbedding(
        #     mesh_device=mesh_device,
        #     args=configuration,
        #     weight_cache_path=configuration.weight_cache_path(dtype),
        #     state_dict=state_dict,
        #     dtype=ttnn.bfloat16,  # Row major layout requires bfloat16
        # )
        # NOTE: Running all embeddings in torch for now since learnable embeddings use complex indexing ops which must be in torch
        self.tok_embeddings = torch.nn.Embedding(configuration.vocab_size, configuration.dim)
        tok_embedding_prefix = f"{state_dict_prefix}tok_embeddings."
        self.tok_embeddings.load_state_dict(
            {k[len(tok_embedding_prefix) :]: v for k, v in state_dict.items() if k.startswith(tok_embedding_prefix)}
        )

        self.norm = RMSNorm(
            device=mesh_device,
            dim=configuration.dim,
            state_dict=state_dict,
            state_dict_prefix=configuration.get_state_dict_prefix("", None),
            weight_cache_path=weight_cache_path,
            weight_dtype=dtype,
            weight_key="norm",
        )

        # # self.output layer weight
        # TODO: Generalize LMHead, maybe use llama_model's single-tile-sequence LMHead
        # self.output = LMHead(
        #     configuration,
        #     mesh_device,
        #     ttnn.bfloat8_b,
        #     state_dict,
        #     state_dict_prefix,
        #     weight_cache_path,
        # )

        # torch_weight = lambda name, suffix: torch.transpose(
        #     self.state_dict[f"{state_dict_prefix}{name}.{suffix}"], -2, -1
        # )

        lm_head_torch = self.state_dict[f"{state_dict_prefix}output.weight"].transpose(-1, -2)
        num_splits = 4  # arbitrary, reasonable number
        lm_head_torch = torch.chunk(lm_head_torch, num_splits, dim=-1)

        cache_name = lambda name, suffix, split: weight_cache_path / (state_dict_prefix + f"{name}{suffix}{split}")
        as_interleaved_tensor = lambda name, suffix, split, type, dim: ttnn.as_tensor(
            lm_head_torch[split],
            dtype=type,
            device=self.mesh_device,
            mesh_mapper=ttnn.ShardTensorToMesh(self.mesh_device, dim=dim),
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cache_file_name=cache_name(name, suffix, split),
        )

        # Sharded weights
        self.outputs = [
            as_interleaved_tensor("output", "weight", idx, ttnn.bfloat8_b, dim=-1) for idx in range(len(lm_head_torch))
        ]
        # self.output = as_interleaved_tensor("output", "weight", ttnn.bfloat8_b, dim=-1)

        self.n_llama_layers = configuration.n_layers
        self.model_dim = configuration.dim

        self.fusion_schedule = self._init_fusion_schedule(configuration.vision_num_cross_attention_layers)

        self.learnable_embedding = torch.nn.Embedding(
            8,
            configuration.dim,
        )
        learn_embedding_prefix = f"{state_dict_prefix}learnable_embedding."
        self.learnable_embedding.load_state_dict(
            {k[len(learn_embedding_prefix) :]: v for k, v in state_dict.items() if k.startswith(learn_embedding_prefix)}
        )
        self.num_frozen_embeddings = self.tok_embeddings.num_embeddings
        self._thresh = self.num_frozen_embeddings - 1

        # transformer blocks
        self.layers = []
        self.cross_attention_layers = []
        for i in range(configuration.n_layers):
            layer_id = i
            block = TtTransformerBlock(configuration, mesh_device, dtype, state_dict, layer_id, weight_cache_path)
            self.layers.append(block)
            if layer_id in self.fusion_schedule:
                xa_layer_id = self.fusion_schedule.index(layer_id)
                block = TtLlamaCrossAttentionTransformerBlock(
                    mesh_device,
                    state_dict,
                    f"{state_dict_prefix}cross_attention_layers.{xa_layer_id}.",
                    weight_cache_path,
                    dtype,
                    configuration,
                )
                self.cross_attention_layers.append(block)

        # add xattn and dummy layers to avoid conditionals in forward()
        self.text_and_xattn_layers = []

        for idx, layer in enumerate(self.layers):
            if idx in self.fusion_schedule:
                xattn_layer_idx = self.fusion_schedule.index(idx)
                xattn_layer = self.cross_attention_layers[xattn_layer_idx]
            else:
                xattn_layer_idx = 0
                xattn_layer = lambda x, *args, **kwargs: x

            self.text_and_xattn_layers.append(
                (
                    layer,
                    xattn_layer,
                    xattn_layer_idx,
                )
            )

        self.max_seq_len = configuration.max_seq_len

    def _init_fusion_schedule(
        self,
        num_layers: int,
    ):
        llama_layers = list(range(self.n_llama_layers))

        # uniformly spread the layers
        k = math.ceil(len(llama_layers) / num_layers)
        return llama_layers[::-1][::k][:num_layers][::-1]

    def get_partially_trainable_embedding(self, x):
        xz = torch.zeros_like(x, device=x.device)
        oz = torch.ones_like(x, device=x.device)
        x_orig = torch.minimum(x, torch.tensor(self._thresh, device=x.device))
        x_new = torch.maximum(x, torch.tensor(self._thresh + 1, device=x.device)) - self.num_frozen_embeddings

        mask_orig = torch.where(x >= self.num_frozen_embeddings, xz, oz).unsqueeze(-1)
        mask_new = torch.where(x < self.num_frozen_embeddings, xz, oz).unsqueeze(-1)

        x_orig = self.tok_embeddings(x_orig)
        x_new = self.learnable_embedding(x_new).type_as(x_orig)
        return x_orig * mask_orig.type_as(x_orig) + x_new * mask_new.type_as(x_new)

    def _get_xattn_mask(
        self,
        num_tokens,
        text_device,
        text_dtype,
        vision_tokens,
        cross_attention_masks,
    ):
        assert vision_tokens is not None, "Vision tokens must be provided"
        vision_seqlen = vision_tokens.shape[3]
        assert (
            vision_tokens.shape[1] == cross_attention_masks.shape[2]
        ), f"Mismatch in number of images given and number of masks given {vision_tokens.shape} {cross_attention_masks.shape}"
        assert (
            vision_tokens.shape[2] == cross_attention_masks.shape[3]
        ), f"Vision tokens shape {vision_tokens.shape} mismatch with xattn shape {cross_attention_masks.shape}"
        assert (
            num_tokens == cross_attention_masks.shape[1]
        ), f"Mismatch in text sequence length and cross attention mask sequence length {num_tokens} {cross_attention_masks.shape}"
        _, _, _, num_image_tokens, image_token_dim = tuple(vision_tokens.shape)
        bsz, ntext, nimg, nchunks = cross_attention_masks.shape
        cross_attention_masks = (
            cross_attention_masks.repeat_interleave(vision_seqlen, dim=3).view(bsz, ntext, -1).unsqueeze(1)
        )
        full_text_row_masked_out_mask = _get_full_row_masked_out_mask(
            cross_attention_masks,
            torch.finfo(cross_attention_masks.dtype).min,
        )
        cross_attention_masks *= full_text_row_masked_out_mask

        return (
            cross_attention_masks.to(device=text_device, dtype=text_dtype),
            full_text_row_masked_out_mask,
        )

    def setup_cache(self, max_batch_size, dtype):
        self.cache_is_setup = True

    def forward(
        self,
        h: ttnn.Tensor,
        xattn_mask,
        full_text_row_masked_out_mask_1NSH: ttnn.Tensor,
        full_text_row_masked_out_mask_11SD: ttnn.Tensor,
        xattn_caches,
        current_pos,
        rot_mat=None,
        transformation_mats=None,
        user_id=0,
        mode="decode",
        page_table=None,
        # get_last_token=-1,
        text_only_inference=False,
    ):
        for idx, (
            layer,
            xattn_layer,
            xattn_layer_idx,
        ) in enumerate(self.text_and_xattn_layers):
            if not text_only_inference:
                h = xattn_layer(
                    h,
                    xattn_mask=xattn_mask,
                    xattn_cache=xattn_caches[xattn_layer_idx],
                    full_text_row_masked_out_mask_1NSH=full_text_row_masked_out_mask_1NSH,
                    full_text_row_masked_out_mask_11SD=full_text_row_masked_out_mask_11SD,
                    mode=mode,
                )
            h = layer(
                h,
                current_pos,
                rot_mat=rot_mat,
                transformation_mats=transformation_mats,
                user_id=user_id,
                mode=mode,
            )

        h = self.norm(h)

        seq_len = h.shape[2]
        MAX_MM_SEQ_LEN = 1024
        if seq_len >= MAX_MM_SEQ_LEN:  # Too big to compute. Set different program configs based on seqlen
            # Reshape input to to fit on device and parallelize computation
            h = ttnn.reshape(h, [1, seq_len // MAX_MM_SEQ_LEN, MAX_MM_SEQ_LEN, -1])
        pc = self.model_config["CROSS_TRANSFORMER_TEXT_OUTPUT_PROGCFG"](seq_len, MAX_MM_SEQ_LEN)

        outputs = []
        for out_weight in self.outputs:
            output = ttnn.linear(
                h,
                out_weight,
                compute_kernel_config=self.model_config["SDPA_DECODE_COMPUTE_PROGCFG"],
                core_grid=None,
                dtype=ttnn.bfloat16,
                program_config=pc,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

            if self.configuration.num_devices > 1:
                output = ttnn.all_gather(output, dim=3, num_links=1, topology=ttnn.Topology.Linear)
            outputs.append(output)

        output = ttnn.concat(outputs, dim=-1)

        return output
