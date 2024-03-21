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

        self.scalar = [pad_by_zero(torch.Tensor([1 / math.sqrt(self.head_dim)]), device)[0] for device in devices]

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
                    output_mem_config=self.model_config["K_TRANSPOSED_OUTPUT_MEMCFG"],
                )
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

        elif llm_mode == "decode":
            for i, device in enumerate(self.devices):
                # TODO: switch to group_attn_matmul once multiple q heads is supported (issue #5318)
                if is_wormhole_b0():
                    attn_weights.append(
                        tt_lib.operations.primary.transformers.attn_matmul(
                            query_layer[i],
                            key_layer_transposed[i],
                            compute_with_storage_grid_size=device.compute_with_storage_grid_size(),
                            output_mem_config=self.model_config["PRE_SOFTMAX_MM_OUTPUT_MEMCFG"],
                            output_dtype=self.model_config["PRE_SOFTMAX_MM_OUTPUT_DTYPE"],  # Must be BFLOAT16
                        )
                    )
                else:
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
        # TODO: Replace with scaled_softmax_attention_mask from BERT
        for i in range(self.num_devices):
            attn_weights[i] = tt_lib.operations.primary.softmax_in_place(
                attn_weights[i],
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

        elif llm_mode == "decode":
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
