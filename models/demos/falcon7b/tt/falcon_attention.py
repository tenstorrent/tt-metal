# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import math
from torch import nn
from typing import Optional, Tuple

import tt_lib

from models.utility_functions import (
    torch2tt_tensor,
    tt2torch_tensor,
    pad_by_zero,
    nearest_32,
    is_wormhole_b0,
)

from models.demos.falcon7b.tt.model_utils import get_weights_cached


class TtFalconRotaryEmbedding(torch.nn.Module):
    """
    See FalconRotaryEmbedding from hf_modeling_falcon.py
    """

    def __init__(
        self,
        tt_devices,
        dim,
        base_url,
        layer_num,
        max_position_embeddings=2048,
        base=10000,
        model_config=None,
        tt_cache_path=None,
    ):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))

        self.max_seq_len_cached = max_position_embeddings
        self.model_config = model_config
        t = torch.arange(
            self.max_seq_len_cached,
            device=inv_freq.device,
            dtype=inv_freq.dtype,
        )
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)

        layer_name = f"{base_url}.{layer_num}.rotary_embedding"
        cos_str = f"{layer_name}.cos_cached"
        sin_str = f"{layer_name}.sin_cached"

        overwrite_cos, overwrite_sin = False, False

        for _ in range(2):
            self.tt_cos_cached = get_weights_cached(
                tt_devices,
                model_config,
                tt_cache_path,
                cos_str,
                weight_config_str="COS_CACHED_WEIGHTS",
                weights_to_cache=emb.cos()[None, None, :, :],
                overwrite=overwrite_cos,
            )
            overwrite_cos = (
                tt2torch_tensor(self.tt_cos_cached[0]).shape[-2] != self.max_seq_len_cached
            )  # Verify cached tensor has same max seq len
            if not overwrite_cos:
                break

        for _ in range(2):
            self.tt_sin_cached = get_weights_cached(
                tt_devices,
                model_config,
                tt_cache_path,
                sin_str,
                weight_config_str="SIN_CACHED_WEIGHTS",
                weights_to_cache=emb.sin()[None, None, :, :],
                overwrite=overwrite_sin,
            )
            overwrite_sin = (
                tt2torch_tensor(self.tt_sin_cached[0]).shape[-2] != self.max_seq_len_cached
            )  # Verify cached tensor has same max seq len
            if not overwrite_sin:
                break

    def forward(self, layer: tt_lib.tensor.Tensor, token_idx: Optional[int] = None) -> tt_lib.tensor.Tensor:
        seq_len = layer[0].get_legacy_shape()[2]
        assert seq_len <= self.max_seq_len_cached, "seq_len exceeds max_seq_len_cached in RotaryEmbedding!"

        output = []
        for i in range(len(layer)):
            output.append(
                tt_lib.tensor.rotary_embedding(
                    layer[i],
                    self.tt_cos_cached[i],
                    self.tt_sin_cached[i],
                    token_idx,
                    output_mem_config=self.model_config["ROTARY_EMBEDDING_OUTPUT_MEMCFG"],
                )
            )
        return output


class TtFalconAttention(nn.Module):
    """Mulit-Query Attention: https://arxiv.org/pdf/1911.02150.pdf"""

    def __init__(
        self,
        devices,
        state_dict,
        base_url,
        layer_num,
        hidden_size: int,
        num_heads: int,
        max_position_embeddings: int = 2048,
        model_config=None,
        tt_cache_path=None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.max_position_embeddings = max_position_embeddings
        self.devices = devices
        self.num_devices = len(devices)
        self.state_dict = state_dict
        self.model_config = model_config
        self.padded_local_heads = nearest_32(num_heads)

        if (self.head_dim * num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {num_heads})."
            )

        layer_name = f"{base_url}.{layer_num}.self_attention"
        query_key_value_str = f"{layer_name}.query_key_value.weight"
        selfout_str = f"{layer_name}.dense.weight"

        self.query_key_value_weights = get_weights_cached(
            devices,
            model_config,
            tt_cache_path,
            query_key_value_str,
            weight_config_str="FUSED_QKV_MM_WEIGHTS",
            weights_to_cache=(torch.transpose(state_dict[query_key_value_str], -2, -1) if state_dict else None),
        )
        self.dense_weights = get_weights_cached(
            devices,
            model_config,
            tt_cache_path,
            selfout_str,
            weight_config_str="SELFOUT_MM_WEIGHTS",
            weights_to_cache=(torch.transpose(state_dict[selfout_str], -2, -1) if state_dict else None),
        )

        self.rotary_embedding = TtFalconRotaryEmbedding(
            self.devices,
            self.head_dim,
            base_url,
            layer_num,
            max_position_embeddings=self.max_position_embeddings,
            model_config=model_config,
            tt_cache_path=tt_cache_path,
        )

        self.scale = 1 / math.sqrt(self.head_dim)
        self.scalar = []
        for device in self.devices:
            self.scalar.append(pad_by_zero(torch.Tensor([self.scale]), device)[0])

    def forward(
        self,
        hidden_states: tt_lib.tensor.Tensor,
        alibi: torch.Tensor,
        attention_mask: torch.Tensor,
        llm_mode: str,
        user_id: int = 0,
        layer_past: Optional[Tuple[tt_lib.tensor.Tensor]] = None,
        layer_past_len: int = 0,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[tt_lib.tensor.Tensor, Optional[Tuple[tt_lib.tensor.Tensor]]]:
        """
        Prefill input shape: [batch, 1, seq_len, hidden_size]
        Decode input shape: [seq_len, 1, batch, hidden_size]
        """

        assert not output_attentions

        padded_layer_past_len = nearest_32(layer_past_len + 1)

        if llm_mode == "prefill":
            batch = hidden_states[0].get_legacy_shape()[0]
            q_len = hidden_states[0].get_legacy_shape()[2]
            assert layer_past is not None
        elif llm_mode == "decode":
            batch = hidden_states[0].get_legacy_shape()[2]
            q_len = hidden_states[0].get_legacy_shape()[0]
            # We always store max_position_embeddings for kv_cache,
            # so we need separate variable to store the actual len of the kv_cache
            assert layer_past is not None
            assert layer_past_len > 0 and layer_past_len <= self.max_position_embeddings
        else:
            raise NotImplementedError(f"Llm mode {llm_mode} is not supported! Must be one of prefill or decode.")

        #################
        ### FUSED QKV ###
        #################
        fused_query_key_value = []
        for i in range(self.num_devices):
            fused_query_key_value.append(
                tt_lib.tensor.falcon_fused_qkv_matmul(
                    hidden_states[i],
                    self.query_key_value_weights[i],
                    output_mem_config=self.model_config["FUSED_QKV_MM_OUTPUT_MEMCFG"],
                    output_dtype=self.model_config["FUSED_QKV_MM_OUTPUT_DTYPE"],
                )
            )

        ###########
        ### TMs ###
        ###########
        query_layer, key_layer, value_layer = [], [], []
        for i in range(self.num_devices):
            query_layer_i, key_layer_i, value_layer_i = tt_lib.tensor.nlp_create_qkv_heads_falcon7b(
                fused_query_key_value[i],
                output_mem_config=self.model_config["CREATE_QKV_HEADS_OUTPUT_MEMCFG"],
            )
            fused_query_key_value[i].deallocate()
            query_layer.append(query_layer_i)
            key_layer.append(key_layer_i)
            value_layer.append(value_layer_i)

        #########################
        ### ROTARY EMBEDDINGS ###
        #########################
        if llm_mode == "prefill":
            query_layer = self.rotary_embedding(query_layer)
            key_layer = self.rotary_embedding(key_layer)
        elif llm_mode == "decode":
            query_layer = self.rotary_embedding(query_layer, layer_past_len)
            key_layer = self.rotary_embedding(key_layer, layer_past_len)

        ######################
        ### K CACHE UPDATE ###
        ######################
        if llm_mode == "prefill":
            for i in range(self.num_devices):
                tt_lib.tensor.fill_cache(layer_past[i][0], key_layer[i], user_id)

        elif llm_mode == "decode":
            for i in range(self.num_devices):
                # Update kv_cache in place
                tt_lib.tensor.update_cache(layer_past[i][0], key_layer[i], layer_past_len)
            for i in range(self.num_devices):
                # key and value layers will have kv_seq_len padded to nearest 32
                key_layer[i] = tt_lib.tensor.unpad(
                    layer_past[i][0],
                    [0, 0, 0, 0],
                    [batch - 1, 0, nearest_32(layer_past_len + 1) - 1, self.head_dim - 1],
                    output_mem_config=self.model_config["K_CACHE_SLICE_OUTPUT_MEMCFG"],
                )

            if self.model_config["l1_sharded"]:
                for i in range(self.num_devices):
                    key_layer[i] = tt_lib.tensor.interleaved_to_sharded(
                        key_layer[i],
                        sharded_mem_config=self.model_config["ATTN_BATCH_SHARDED_MEMCFG"](
                            padded_layer_past_len, self.head_dim
                        ),
                    )

                # Pad and transpose Q for batched matmul
                for i in range(self.num_devices):
                    query_layer[i] = tt_lib.tensor.pad(
                        query_layer[i], [1, self.padded_local_heads, batch, self.head_dim], [0, 0, 0, 0], 0.0
                    )

                for i in range(self.num_devices):
                    query_layer[i] = tt_lib.tensor.transpose(
                        query_layer[i],
                        -2,
                        -3,
                    )

                for i in range(self.num_devices):
                    query_layer[i] = tt_lib.tensor.reshape(
                        query_layer[i],
                        batch,
                        1,
                        self.padded_local_heads,
                        self.head_dim,  # Batch must be in dim 0 to match K cache
                    )

                for i in range(self.num_devices):
                    query_layer[i] = tt_lib.tensor.interleaved_to_sharded(
                        query_layer[i],
                        sharded_mem_config=self.model_config["ATTN_BATCH_SHARDED_MEMCFG"](
                            self.padded_local_heads, self.head_dim
                        ),
                    )

        ######################
        ### PRE-SOFTMAX MM ###
        ######################
        key_layer_transposed = []
        for i in range(self.num_devices):
            key_layer_transposed.append(
                tt_lib.tensor.transpose(
                    key_layer[i],
                    -2,
                    -1,
                    output_mem_config=(
                        self.model_config["K_TRANSPOSED_OUTPUT_MEMCFG"]
                        if llm_mode == "prefill" or self.model_config["l1_sharded"] == False
                        else tt_lib.tensor.MemoryConfig(
                            tt_lib.tensor.TensorMemoryLayout.HEIGHT_SHARDED, tt_lib.tensor.BufferType.L1
                        )
                    ),
                ),
            )
            key_layer[i].deallocate()

        attn_weights = []
        if llm_mode == "prefill":
            for i in range(self.num_devices):
                attn_weights.append(
                    tt_lib.tensor.matmul(
                        query_layer[i],
                        key_layer_transposed[i],
                        output_mem_config=self.model_config["PRE_SOFTMAX_MM_OUTPUT_MEMCFG"],
                    )
                )
                query_layer[i].deallocate()
                key_layer_transposed[i].deallocate()
        elif self.model_config["l1_sharded"]:
            for i, device in enumerate(self.devices):
                attn_weights.append(
                    tt_lib.operations.primary.matmul(
                        query_layer[i],
                        key_layer_transposed[i],
                        program_config=self.model_config["ATTN_BATCHED_MM_PROGCFG"](
                            self.head_dim // 32, self.padded_local_heads // 32, padded_layer_past_len // 32
                        ),
                        output_mem_config=self.model_config["ATTN_BATCH_SHARDED_MEMCFG"](
                            self.padded_local_heads, padded_layer_past_len
                        ),
                        output_dtype=self.model_config["PRE_SOFTMAX_MM_OUTPUT_DTYPE"],  # Must be BFLOAT16
                        compute_kernel_config=self.model_config["COMPUTE_KERNEL_CONFIG"],
                    )
                )
                query_layer[i].deallocate()
                key_layer_transposed[i].deallocate()
        elif is_wormhole_b0():
            for i, device in enumerate(self.devices):
                attn_weights.append(
                    tt_lib.operations.primary.transformers.attn_matmul(
                        query_layer[i],
                        key_layer_transposed[i],
                        compute_with_storage_grid_size=device.compute_with_storage_grid_size(),
                        output_mem_config=self.model_config["PRE_SOFTMAX_MM_OUTPUT_MEMCFG"],
                        output_dtype=self.model_config["PRE_SOFTMAX_MM_OUTPUT_DTYPE"],  # Must be BFLOAT16
                    )
                )
                query_layer[i].deallocate()
                key_layer_transposed[i].deallocate()
        else:
            for i, device in enumerate(self.devices):
                attn_weights.append(
                    tt_lib.operations.primary.transformers.group_attn_matmul(
                        query_layer[i],
                        key_layer_transposed[i],
                        compute_with_storage_grid_size=device.compute_with_storage_grid_size(),
                        output_mem_config=self.model_config["PRE_SOFTMAX_MM_OUTPUT_MEMCFG"],
                        output_dtype=self.model_config["PRE_SOFTMAX_MM_OUTPUT_DTYPE"],  # Must be BFLOAT16
                    )
                )
                query_layer[i].deallocate()
                key_layer_transposed[i].deallocate()

        if llm_mode == "prefill" or self.model_config["l1_sharded"] == False:
            for i in range(self.num_devices):
                attn_weights[i] = tt_lib.tensor.bcast(
                    attn_weights[i],
                    self.scalar[i],
                    tt_lib.tensor.BcastOpMath.MUL,
                    tt_lib.tensor.BcastOpDim.HW,
                    output_mem_config=self.model_config["PRE_SOFTMAX_SCALE_OUTPUT_MEMCFG"],
                )

            if attention_mask is not None:
                for i in range(self.num_devices):
                    attn_weights[i] = tt_lib.tensor.add(
                        attn_weights[i],
                        attention_mask[i],
                        output_mem_config=self.model_config["PRE_SOFTMAX_MASK_OUTPUT_MEMCFG"],
                    )
            ###############
            ### SOFTMAX ###
            ###############
            for i in range(self.num_devices):
                attn_weights[i] = tt_lib.operations.primary.transformers.scale_mask_softmax_in_place(attn_weights[i])
        else:
            ###############
            ### SOFTMAX ###
            ###############
            for i in range(self.num_devices):
                attn_weights[i] = tt_lib.operations.primary.transformers.scale_mask_softmax_in_place(
                    attn_weights[i],
                    scale=self.scale,
                    mask=attention_mask[i],
                    program_config=tt_lib.operations.primary.transformers.SoftmaxShardedMultiCoreProgramConfig(
                        compute_with_storage_grid_size=(8, 4),
                        subblock_w=1,
                        block_h=self.padded_local_heads // 32,
                        block_w=padded_layer_past_len // 32,
                    ),
                    is_causal_mask=True,
                )

        ######################
        ### V CACHE UPDATE ###
        ######################
        if llm_mode == "prefill":
            for i in range(self.num_devices):
                tt_lib.tensor.fill_cache(layer_past[i][1], value_layer[i], user_id)

        elif llm_mode == "decode":
            for i in range(self.num_devices):
                # Update kv_cache in place
                tt_lib.tensor.update_cache(layer_past[i][1], value_layer[i], layer_past_len)
            for i in range(self.num_devices):
                value_layer[i] = tt_lib.tensor.unpad(
                    layer_past[i][1],
                    [0, 0, 0, 0],
                    [batch - 1, 0, nearest_32(layer_past_len + 1) - 1, self.head_dim - 1],
                    output_mem_config=self.model_config["V_CACHE_SLICE_OUTPUT_MEMCFG"],
                )
            if self.model_config["l1_sharded"]:
                for i in range(self.num_devices):
                    value_layer[i] = tt_lib.tensor.interleaved_to_sharded(
                        value_layer[i],
                        sharded_mem_config=self.model_config["ATTN_BATCH_SHARDED_MEMCFG"](
                            padded_layer_past_len, self.head_dim
                        ),
                    )

        layer_present = layer_past if use_cache else None

        ########################
        ### POST-SOFTMAX MM ###
        ########################
        attn_output = []
        if llm_mode == "prefill":
            for i in range(self.num_devices):
                attn_output.append(
                    tt_lib.tensor.matmul(
                        attn_weights[i],
                        value_layer[i],
                        output_mem_config=self.model_config["POST_SOFTMAX_MM_OUTPUT_MEMCFG"],
                    )
                )
                attn_weights[i].deallocate()
                value_layer[i].deallocate()

        elif self.model_config["l1_sharded"]:
            for i in range(self.num_devices):
                attn_output.append(
                    tt_lib.operations.primary.matmul(
                        attn_weights[i],
                        value_layer[i],
                        program_config=self.model_config["ATTN_BATCHED_MM_PROGCFG"](
                            padded_layer_past_len // 32,
                            self.padded_local_heads // 32,
                            self.head_dim // 32,
                        ),
                        output_mem_config=self.model_config["ATTN_BATCH_SHARDED_MEMCFG"](
                            self.padded_local_heads,
                            self.head_dim,
                        ),
                        output_dtype=self.model_config["POST_SOFTMAX_MM_OUTPUT_DTYPE"],
                        compute_kernel_config=self.model_config["COMPUTE_KERNEL_CONFIG"],
                    )
                )
                attn_weights[i].deallocate()
                value_layer[i].deallocate()

            for i in range(self.num_devices):
                attn_output[i] = tt_lib.tensor.sharded_to_interleaved(
                    attn_output[i], output_mem_config=self.model_config["POST_SOFTMAX_MM_OUTPUT_MEMCFG"]
                )

            # Get batch in dim 1
            for i in range(self.num_devices):
                attn_output[i] = tt_lib.tensor.reshape(attn_output[i], 1, batch, self.padded_local_heads, self.head_dim)

            # Get batch in dim 2
            for i in range(self.num_devices):
                attn_output[i] = tt_lib.tensor.transpose(
                    attn_output[i],
                    -2,
                    -3,
                )

            # UNPAD
            attn_output_shape = attn_output[0].get_legacy_shape()
            for i in range(self.num_devices):
                attn_output[i] = tt_lib.tensor.unpad(
                    attn_output[i],
                    [0, 0, 0, 0],
                    [
                        attn_output_shape[0] - 1,
                        self.num_heads - 1,
                        attn_output_shape[2] - 1,
                        attn_output_shape[3] - 1,
                    ],
                    output_mem_config=self.model_config["POST_SOFTMAX_MM_OUTPUT_MEMCFG"],
                )
        else:
            for i in range(self.num_devices):
                # TODO: switch to group_attn_matmul once multiple q heads is supported (issue #5318)
                if is_wormhole_b0():
                    attn_output.append(
                        tt_lib.operations.primary.transformers.attn_matmul(
                            attn_weights[i],
                            value_layer[i],
                            compute_with_storage_grid_size=device.compute_with_storage_grid_size(),
                            output_mem_config=self.model_config["POST_SOFTMAX_MM_OUTPUT_MEMCFG"],
                            output_dtype=self.model_config["POST_SOFTMAX_MM_OUTPUT_DTYPE"],  # Must be BFLOAT16
                        )
                    )
                else:
                    attn_output.append(
                        tt_lib.operations.primary.transformers.group_attn_matmul(
                            attn_weights[i],
                            value_layer[i],
                            compute_with_storage_grid_size=device.compute_with_storage_grid_size(),
                            output_mem_config=self.model_config["POST_SOFTMAX_MM_OUTPUT_MEMCFG"],
                            output_dtype=self.model_config["POST_SOFTMAX_MM_OUTPUT_DTYPE"],  # Must be BFLOAT16
                        )
                    )
                attn_weights[i].deallocate()
                value_layer[i].deallocate()

        #########################
        ### ATTENTION SELFOUT ###
        #########################
        for i in range(self.num_devices):
            attn_output[i] = tt_lib.tensor.nlp_concat_heads(
                attn_output[i],
                output_mem_config=self.model_config["CONCAT_HEADS_OUTPUT_MEMCFG"],
            )

        for i in range(self.num_devices):
            attn_output[i] = tt_lib.tensor.falcon_selfout_matmul(
                attn_output[i],
                self.dense_weights[i],
                output_mem_config=self.model_config["SELFOUT_MM_OUTPUT_MEMCFG"],
                output_dtype=self.model_config["SELFOUT_MM_OUTPUT_DTYPE"],
            )

        return attn_output, layer_present
