# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import math
from torch import nn
from typing import List, Optional, Tuple

from models.demos.falcon7b_common.tt.model_utils import get_falcon_default_core_grid
import ttnn

from models.utility_functions import (
    tt2torch_tensor,
    pad_by_zero,
    nearest_32,
    is_wormhole_b0,
)

from models.demos.falcon7b_common.tt.model_utils import get_weights_cached
from models.utility_functions import torch_tensors_to_tt_tensors


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

    def forward(
        self, layer: ttnn.experimental.tensor.Tensor, token_idx: Optional[int] = None
    ) -> ttnn.experimental.tensor.Tensor:
        seq_len = layer[0].get_legacy_shape()[2]
        assert seq_len <= self.max_seq_len_cached, "seq_len exceeds max_seq_len_cached in RotaryEmbedding!"

        output = []
        for i in range(len(layer)):
            output.append(
                ttnn.experimental.tensor.rotary_embedding(
                    layer[i],
                    self.tt_cos_cached[i],
                    self.tt_sin_cached[i],
                    token_idx,
                    output_mem_config=self.model_config["ROTARY_EMBEDDING_OUTPUT_MEMCFG"],
                )
            )
        return output


class TtFalconAttentionPrefill(nn.Module):
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
        weights_dict=None,
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
            weights_dict=weights_dict,
        )
        self.dense_weights = get_weights_cached(
            devices,
            model_config,
            tt_cache_path,
            selfout_str,
            weight_config_str="SELFOUT_MM_WEIGHTS",
            weights_to_cache=(torch.transpose(state_dict[selfout_str], -2, -1) if state_dict else None),
            weights_dict=weights_dict,
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

        scale = 1 / math.sqrt(self.head_dim)
        self.scalar = []
        for device in self.devices:
            self.scalar.append(pad_by_zero(torch.Tensor([scale]), device)[0])

        # optimized version can utilize single float value for softmax
        if self.model_config["PREFILL_OPTIMIZED_MODE"]:
            self.scalar_for_optimized_prefill = 1 / math.sqrt(self.head_dim)

        # generate output buffer on device
        if self.model_config["PREFILL_OPTIMIZED_MODE"] and "ATTN_OUTPUT_TENSORS" not in self.model_config:
            self.model_config["ATTN_OUTPUT_TENSORS"] = {}
            # create output tensors
            for seq_len in [128, 1024, 2048]:
                tensor = torch.zeros((1, self.num_heads, seq_len, self.head_dim)).bfloat16().float()

                tt_tensors = torch_tensors_to_tt_tensors(
                    [tensor.detach().clone() for _ in range(self.num_devices)],
                    ttnn.experimental.tensor.Layout.TILE,
                    ttnn.experimental.tensor.DataType.BFLOAT16,
                    self.model_config["ATTN_OPTIMIZED_MEMCFG"],
                    self.devices,
                )
                self.model_config["ATTN_OUTPUT_TENSORS"][seq_len] = tt_tensors

    def forward(
        self,
        hidden_states: ttnn.experimental.tensor.Tensor,
        alibi: torch.Tensor,
        attention_mask: torch.Tensor,
        llm_mode: str,
        user_id: int = 0,
        layer_past: Optional[Tuple[ttnn.experimental.tensor.Tensor]] = None,
        layer_past_len: int = 0,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[ttnn.experimental.tensor.Tensor, Optional[Tuple[ttnn.experimental.tensor.Tensor]]]:
        """
        Prefill input shape: [1, 1, seq_len, hidden_size]
        """
        assert not output_attentions

        seq_len = hidden_states[0].get_legacy_shape()[2]

        if self.model_config["PREFILL_OPTIMIZED_MODE"] and seq_len in [128, 1024, 2048]:
            attn_output, layer_present = self._optimized_forward(
                hidden_states,
                attention_mask,
                user_id,
                layer_past,
                use_cache,
            )
            return attn_output, layer_present

        #################
        ### FUSED QKV ###
        #################
        fused_query_key_value = []
        for i in range(self.num_devices):
            fused_query_key_value.append(
                ttnn.matmul(
                    hidden_states[i],
                    self.query_key_value_weights[i],
                    memory_config=self.model_config["FUSED_QKV_MM_OUTPUT_MEMCFG"],
                    dtype=self.model_config["FUSED_QKV_MM_OUTPUT_DTYPE"],
                    core_grid=get_falcon_default_core_grid(hidden_states[i].device()),
                )
            )

        ###########
        ### TMs ###
        ###########
        query_layer, key_layer, value_layer = [], [], []
        for i in range(self.num_devices):
            query_layer_i, key_layer_i, value_layer_i = ttnn.experimental.nlp_create_qkv_heads_falcon7b(
                fused_query_key_value[i],
                memory_config=self.model_config["CREATE_QKV_HEADS_OUTPUT_MEMCFG"],
            )
            fused_query_key_value[i].deallocate()
            query_layer.append(query_layer_i)
            key_layer.append(key_layer_i)
            value_layer.append(value_layer_i)

        #########################
        ### ROTARY EMBEDDINGS ###
        #########################
        query_layer = self.rotary_embedding(query_layer)
        key_layer = self.rotary_embedding(key_layer)

        ######################
        ### K CACHE UPDATE ###
        ######################
        for i in range(self.num_devices):
            ttnn.experimental.tensor.fill_cache(layer_past[i][0], key_layer[i], user_id)

        ######################
        ### PRE-SOFTMAX MM ###
        ######################
        key_layer_transposed = []
        for i in range(self.num_devices):
            key_layer_transposed.append(
                ttnn.transpose(
                    key_layer[i],
                    -2,
                    -1,
                    memory_config=(
                        self.model_config["K_TRANSPOSED_OUTPUT_MEMCFG"]
                        if llm_mode == "prefill" or self.model_config["l1_sharded"] == False
                        else ttnn.experimental.tensor.MemoryConfig(
                            ttnn.experimental.tensor.TensorMemoryLayout.HEIGHT_SHARDED,
                            ttnn.experimental.tensor.BufferType.L1,
                        )
                    ),
                ),
            )
            key_layer[i].deallocate()

        attn_weights = []
        for i in range(self.num_devices):
            attn_weights.append(
                ttnn.matmul(
                    query_layer[i],
                    key_layer_transposed[i],
                    memory_config=self.model_config["PRE_SOFTMAX_MM_OUTPUT_MEMCFG"],
                )
            )
            query_layer[i].deallocate()
            key_layer_transposed[i].deallocate()

        for i in range(self.num_devices):
            attn_weights[i] = ttnn.experimental.tensor.bcast(
                attn_weights[i],
                self.scalar[i],
                ttnn.experimental.tensor.BcastOpMath.MUL,
                ttnn.experimental.tensor.BcastOpDim.HW,
                output_mem_config=self.model_config["PRE_SOFTMAX_SCALE_OUTPUT_MEMCFG"],
            )

        if attention_mask is not None:
            for i in range(self.num_devices):
                attn_weights[i] = ttnn.add(
                    attn_weights[i],
                    attention_mask[i],
                    memory_config=self.model_config["PRE_SOFTMAX_MASK_OUTPUT_MEMCFG"],
                )
        ###############
        ### SOFTMAX ###
        ###############
        for i in range(self.num_devices):
            attn_weights[i] = ttnn.scale_mask_softmax_in_place(attn_weights[i])

        ######################
        ### V CACHE UPDATE ###
        ######################
        for i in range(self.num_devices):
            ttnn.experimental.tensor.fill_cache(layer_past[i][1], value_layer[i], user_id)

        layer_present = layer_past if use_cache else None

        ########################
        ### POST-SOFTMAX MM ###
        ########################
        attn_output = []
        for i in range(self.num_devices):
            attn_output.append(
                ttnn.matmul(
                    attn_weights[i],
                    value_layer[i],
                    memory_config=self.model_config["POST_SOFTMAX_MM_OUTPUT_MEMCFG"],
                )
            )
            attn_weights[i].deallocate()
            value_layer[i].deallocate()

        #########################
        ### ATTENTION SELFOUT ###
        #########################
        for i in range(self.num_devices):
            attn_output[i] = ttnn.experimental.nlp_concat_heads(
                attn_output[i],
                memory_config=self.model_config["CONCAT_HEADS_OUTPUT_MEMCFG"],
            )

        for i in range(self.num_devices):
            attn_output[i] = ttnn.matmul(
                attn_output[i],
                self.dense_weights[i],
                memory_config=self.model_config["SELFOUT_MM_OUTPUT_MEMCFG"],
                dtype=self.model_config["SELFOUT_MM_OUTPUT_DTYPE"],
                core_grid=get_falcon_default_core_grid(attn_output[i].device()),
            )

        return attn_output, layer_present

    def _optimized_forward(
        self,
        hidden_states: ttnn.experimental.tensor.Tensor,
        attention_mask: torch.Tensor,
        user_id: int = 0,
        layer_past: Optional[Tuple[ttnn.experimental.tensor.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[ttnn.experimental.tensor.Tensor, Optional[Tuple[ttnn.experimental.tensor.Tensor]]]:
        seq_len = hidden_states[0].get_legacy_shape()[2]

        #################
        ### FUSED QKV ###
        #################
        if seq_len == 2048:
            fused_query_key_value = [
                ttnn.matmul(
                    hidden_states[device_id],
                    self.query_key_value_weights[device_id],
                    program_config=self.model_config["FUSED_QKV_MM_OPTIMIZED_PROGCFG"],
                    memory_config=self.model_config["FUSED_QKV_MM_OPTIMIZED_MEMCFG"],
                    dtype=ttnn.experimental.tensor.DataType.BFLOAT16,
                    compute_kernel_config=self.model_config["FUSED_QKV_MM_OPTIMIZED_KERNEL_CONFIG"],
                )
                for device_id in range(self.num_devices)
            ]
        else:
            fused_query_key_value = [
                ttnn.matmul(
                    hidden_states[device_id],
                    self.query_key_value_weights[device_id],
                    memory_config=self.model_config["FUSED_QKV_MM_OUTPUT_MEMCFG"],
                    dtype=self.model_config["FUSED_QKV_MM_OUTPUT_DTYPE"],
                    compute_kernel_config=self.model_config["FUSED_QKV_MM_OPTIMIZED_KERNEL_CONFIG"],
                    core_grid=ttnn.CoreGrid(y=7, x=8),
                )
                for device_id in range(self.num_devices)
            ]

        ###########
        ### TMs ###
        ###########
        query_layer, key_layer, value_layer = [], [], []
        for i in range(self.num_devices):
            query_layer_i, key_layer_i, value_layer_i = ttnn.experimental.nlp_create_qkv_heads_falcon7b(
                fused_query_key_value[i],
                memory_config=self.model_config["CREATE_QKV_HEADS_OUTPUT_MEMCFG"],
            )
            fused_query_key_value[i].deallocate()
            query_layer.append(query_layer_i)
            key_layer.append(key_layer_i)
            value_layer.append(value_layer_i)

        #########################
        ### ROTARY EMBEDDINGS ###
        #########################
        query_layer = self.rotary_embedding(query_layer)
        key_layer = self.rotary_embedding(key_layer)

        ######################
        ### K CACHE UPDATE ###
        ######################
        for i in range(self.num_devices):
            ttnn.experimental.tensor.fill_cache(layer_past[i][0], key_layer[i], user_id)

        ######################
        ### PRE-SOFTMAX MM ###
        ######################
        key_layer_transposed = []
        for i in range(self.num_devices):
            key_layer_transposed.append(
                ttnn.transpose(
                    key_layer[i],
                    -2,
                    -1,
                    memory_config=self.model_config["K_TRANSPOSED_OUTPUT_MEMCFG"],
                )
            )
            key_layer[i].deallocate()

        grid_size = self.model_config["ATTN_OPTIMIZED_GRID_SIZE"]
        allowed_num_cores = self.model_config["ATTN_OPTIMIZED_ALLOWED_NUM_CORES"]
        num_slices = {128: 1, 1024: 4, 2048: 16}[seq_len]

        tiles_per_shard = math.ceil((((self.num_heads * seq_len) / allowed_num_cores) / num_slices) / 32)
        mm_activations_height_shard_spec = [tiles_per_shard * 32, 2 * 32]
        mm_output_height_shard_spec = [tiles_per_shard * 32, seq_len]

        # Define output buffer on devices
        attention_outputs_concatenated = self.model_config["ATTN_OUTPUT_TENSORS"][seq_len]

        # Slice inputs and operate on each slice separately
        for i in range(num_slices):
            slices = [
                ttnn.experimental.tensor.interleaved_to_sharded_partial(
                    query_layer[device_id],
                    grid_size,
                    mm_activations_height_shard_spec,
                    num_slices,  # num_slices
                    i,  # slice_index
                    ttnn.experimental.tensor.TensorMemoryLayout.HEIGHT_SHARDED,
                    ttnn.experimental.tensor.ShardOrientation.ROW_MAJOR,
                )
                for device_id in range(self.num_devices)
            ]

            subblock_h = 1
            subblock_w = 1
            if seq_len == 2048:
                subblock_w = 8  # best option

            qkt_prg_cfg = self.model_config["QKT_OPTIMIZED_PROGCFG"](tiles_per_shard, seq_len, subblock_h, subblock_w)

            ### QKT MATMUL ###
            mm_slices = [
                ttnn.matmul(
                    slices[device_id],
                    key_layer_transposed[device_id],
                    program_config=qkt_prg_cfg,
                    memory_config=self.model_config["QKTV_MM_OPTIMIZED_MEMCFG"],
                    dtype=ttnn.experimental.tensor.DataType.BFLOAT16,
                    compute_kernel_config=self.model_config["QKTV_AND_SOFTMAX_OPTIMIZED_KERNEL_CONFIG"],
                )
                for device_id in range(self.num_devices)
            ]

            softmax_program_config = self.model_config["SOFTMAX_OPTIMIZED_PROGCFG"](
                grid_size, subblock_w, mm_output_height_shard_spec[0] // 32, mm_output_height_shard_spec[1] // 32
            )
            ### SOFTMAX ###
            mm_slices = [
                ttnn.scale_causal_mask_hw_dims_softmax_in_place(
                    mm_slices[device_id],
                    self.scalar_for_optimized_prefill,
                    attention_mask[device_id][i],
                    program_config=softmax_program_config,
                    compute_kernel_config=self.model_config["QKTV_AND_SOFTMAX_OPTIMIZED_KERNEL_CONFIG"],
                )
                for device_id in range(self.num_devices)
            ]

            ### QKTV MATMUL ###
            qktv_prg_cfg = self.model_config["QKTV_MM_OPTIMIZED_PROGCFG"](tiles_per_shard, seq_len, subblock_h)
            attn_out_slices = [
                ttnn.matmul(
                    mm_slices[device_id],
                    value_layer[device_id],
                    program_config=qktv_prg_cfg,
                    memory_config=self.model_config["QKTV_MM_OPTIMIZED_MEMCFG"],
                    dtype=ttnn.experimental.tensor.DataType.BFLOAT16,
                    compute_kernel_config=self.model_config["QKTV_AND_SOFTMAX_OPTIMIZED_KERNEL_CONFIG"],
                )
                for device_id in range(self.num_devices)
            ]

            for device_id in range(self.num_devices):
                ttnn.experimental.tensor.sharded_to_interleaved_partial(
                    attn_out_slices[device_id],
                    attention_outputs_concatenated[device_id],
                    num_slices,
                    i,
                    self.model_config["ATTN_OPTIMIZED_MEMCFG"],
                )
            for device_id in range(self.num_devices):
                attn_out_slices[device_id].deallocate(True)
                mm_slices[device_id].deallocate(True)
                slices[device_id].deallocate(True)

        # V cache update
        for device_id in range(self.num_devices):
            ttnn.experimental.tensor.fill_cache(layer_past[device_id][1], value_layer[device_id], user_id)

        layer_present = layer_past if use_cache else None

        attn_outputs = [
            ttnn.experimental.nlp_concat_heads(
                attention_outputs_concatenated[device_id],
                memory_config=self.model_config["CONCAT_HEADS_OUTPUT_MEMCFG"],
            )
            for device_id in range(self.num_devices)
        ]
        attn_outputs = [
            ttnn.matmul(
                attn_outputs[device_id],
                self.dense_weights[device_id],
                memory_config=self.model_config["SELFOUT_MM_OUTPUT_MEMCFG"],
                dtype=self.model_config["SELFOUT_MM_OUTPUT_DTYPE"],
                compute_kernel_config=self.model_config["SELFOUT_MM_OPTIMIZED_KERNEL_CONFIG"],
                core_grid=ttnn.CoreGrid(y=7, x=8),
            )
            for device_id in range(self.num_devices)
        ]

        return attn_outputs, layer_present


class TtFalconAttentionDecode(nn.Module):
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
        weights_dict=None,
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
            weights_dict=weights_dict,
        )
        self.dense_weights = get_weights_cached(
            devices,
            model_config,
            tt_cache_path,
            selfout_str,
            weight_config_str="SELFOUT_MM_WEIGHTS",
            weights_to_cache=(torch.transpose(state_dict[selfout_str], -2, -1) if state_dict else None),
            weights_dict=weights_dict,
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
        hidden_states: ttnn.experimental.tensor.Tensor,
        alibi: torch.Tensor,
        attention_mask: torch.Tensor,
        llm_mode: str,
        user_id: int = 0,
        layer_past: Optional[Tuple[ttnn.experimental.tensor.Tensor]] = None,
        layer_past_len: int = 0,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[ttnn.experimental.tensor.Tensor, Optional[Tuple[ttnn.experimental.tensor.Tensor]]]:
        """
        Prefill input shape: [batch, 1, seq_len, hidden_size]
        Decode input shape: [seq_len, 1, batch, hidden_size]
        """

        assert not output_attentions

        padded_layer_past_len = nearest_32(layer_past_len + 1)

        batch = hidden_states[0].get_legacy_shape()[2]
        q_len = hidden_states[0].get_legacy_shape()[0]
        # We always store max_position_embeddings for kv_cache,
        # so we need separate variable to store the actual len of the kv_cache
        assert layer_past is not None
        assert layer_past_len <= self.max_position_embeddings

        #################
        ### FUSED QKV ###
        #################
        fused_query_key_value = []
        for i in range(self.num_devices):
            fused_query_key_value.append(
                ttnn.matmul(
                    hidden_states[i],
                    self.query_key_value_weights[i],
                    memory_config=self.model_config["FUSED_QKV_MM_OUTPUT_MEMCFG"],
                    dtype=self.model_config["FUSED_QKV_MM_OUTPUT_DTYPE"],
                    core_grid=get_falcon_default_core_grid(hidden_states[i].device()),
                )
            )

        ###########
        ### TMs ###
        ###########
        query_layer, key_layer, value_layer = [], [], []
        for i in range(self.num_devices):
            query_layer_i, key_layer_i, value_layer_i = ttnn.experimental.nlp_create_qkv_heads_falcon7b(
                fused_query_key_value[i],
                memory_config=self.model_config["CREATE_QKV_HEADS_OUTPUT_MEMCFG"],
            )
            fused_query_key_value[i].deallocate(True)
            query_layer.append(query_layer_i)
            key_layer.append(key_layer_i)
            value_layer.append(value_layer_i)

        #########################
        ### ROTARY EMBEDDINGS ###
        #########################
        query_layer = self.rotary_embedding(query_layer, layer_past_len)
        key_layer = self.rotary_embedding(key_layer, layer_past_len)

        ######################
        ### K CACHE UPDATE ###
        ######################
        for i in range(self.num_devices):
            # Update kv_cache in place
            ttnn.experimental.tensor.update_cache(layer_past[i][0], key_layer[i], layer_past_len)
            key_layer[i].deallocate(True)
        for i in range(self.num_devices):
            # key and value layers will have kv_seq_len padded to nearest 32
            key_layer[i] = ttnn.slice(
                layer_past[i][0],
                [0, 0, 0, 0],
                [batch - 1, 0, nearest_32(layer_past_len + 1) - 1, self.head_dim - 1],
                memory_config=self.model_config["K_CACHE_SLICE_OUTPUT_MEMCFG"],
            )

        if self.model_config["l1_sharded"]:
            for i in range(self.num_devices):
                key_layer[i] = ttnn.experimental.tensor.interleaved_to_sharded(
                    key_layer[i],
                    sharded_mem_config=self.model_config["ATTN_BATCH_SHARDED_MEMCFG"](
                        padded_layer_past_len, self.head_dim
                    ),
                )

            # Pad and transpose Q for batched matmul
            for i in range(self.num_devices):
                query_layer[i] = ttnn.pad(
                    query_layer[i],
                    [1, self.padded_local_heads, batch, self.head_dim],
                    [0, 0, 0, 0],
                    0.0,
                )

            for i in range(self.num_devices):
                query_layer[i] = ttnn.transpose(
                    query_layer[i],
                    -2,
                    -3,
                )

            for i in range(self.num_devices):
                query_layer[i] = ttnn.experimental.tensor.reshape(
                    query_layer[i],
                    batch,
                    1,
                    self.padded_local_heads,
                    self.head_dim,  # Batch must be in dim 0 to match K cache
                )

            for i in range(self.num_devices):
                query_layer[i] = ttnn.experimental.tensor.interleaved_to_sharded(
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
                ttnn.transpose(
                    key_layer[i],
                    -2,
                    -1,
                    memory_config=(
                        self.model_config["K_TRANSPOSED_OUTPUT_MEMCFG"]
                        if self.model_config["l1_sharded"] == False
                        else ttnn.experimental.tensor.MemoryConfig(
                            ttnn.experimental.tensor.TensorMemoryLayout.HEIGHT_SHARDED,
                            ttnn.experimental.tensor.BufferType.L1,
                        )
                    ),
                ),
            )
            key_layer[i].deallocate()

        attn_weights = []
        if self.model_config["l1_sharded"]:
            for i, device in enumerate(self.devices):
                attn_weights.append(
                    ttnn.matmul(
                        query_layer[i],  # [batch, 1, padded_local_heads, head_dim]
                        key_layer_transposed[i],  # [batch, 1, head_dim, padded_layer_past_len]
                        program_config=self.model_config["ATTN_BATCHED_MM_PROGCFG"](
                            self.head_dim // 32, self.padded_local_heads // 32, padded_layer_past_len // 32
                        ),
                        memory_config=self.model_config["ATTN_BATCH_SHARDED_MEMCFG"](
                            self.padded_local_heads, padded_layer_past_len
                        ),
                        dtype=self.model_config["PRE_SOFTMAX_MM_OUTPUT_DTYPE"],  # Must be BFLOAT16
                        compute_kernel_config=self.model_config["PRE_SOFTMAX_MM_COMPUTE_KERNEL_CONFIG"],
                    )
                )
                query_layer[i].deallocate()
                key_layer_transposed[i].deallocate(True)
        elif is_wormhole_b0():
            for i, device in enumerate(self.devices):
                attn_weights.append(
                    ttnn.experimental.operations.primary.transformers.attn_matmul(
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
                    ttnn.experimental.operations.primary.transformers.group_attn_matmul(
                        query_layer[i],
                        key_layer_transposed[i],
                        compute_with_storage_grid_size=device.compute_with_storage_grid_size(),
                        output_mem_config=self.model_config["PRE_SOFTMAX_MM_OUTPUT_MEMCFG"],
                        output_dtype=self.model_config["PRE_SOFTMAX_MM_OUTPUT_DTYPE"],  # Must be BFLOAT16
                    )
                )
                query_layer[i].deallocate()
                key_layer_transposed[i].deallocate(True)

        if self.model_config["l1_sharded"] == False:
            for i in range(self.num_devices):
                attn_weights[i] = ttnn.experimental.tensor.bcast(
                    attn_weights[i],
                    self.scalar[i],
                    ttnn.experimental.tensor.BcastOpMath.MUL,
                    ttnn.experimental.tensor.BcastOpDim.HW,
                    output_mem_config=self.model_config["PRE_SOFTMAX_SCALE_OUTPUT_MEMCFG"],
                )

            if attention_mask is not None:
                for i in range(self.num_devices):
                    attn_weights[i] = ttnn.add(
                        attn_weights[i],
                        attention_mask[i],
                        memory_config=self.model_config["PRE_SOFTMAX_MASK_OUTPUT_MEMCFG"],
                    )
            ###############
            ### SOFTMAX ###
            ###############
            for i in range(self.num_devices):
                attn_weights[i] = ttnn.scale_mask_softmax_in_place(attn_weights[i])
        else:
            ###############
            ### SOFTMAX ###
            ###############
            for i in range(self.num_devices):
                attn_weights[i] = ttnn.scale_mask_softmax_in_place(
                    attn_weights[i],
                    scale=self.scale,
                    mask=attention_mask[i],
                    program_config=ttnn.SoftmaxShardedMultiCoreProgramConfig(
                        compute_with_storage_grid_size=(8, 4),
                        subblock_w=1,
                        block_h=self.padded_local_heads // 32,
                        block_w=padded_layer_past_len // 32,
                    ),
                    is_causal_mask=False,  # causal_mask=False will broadcast attention mask across all heads
                )

        ######################
        ### V CACHE UPDATE ###
        ######################
        for i in range(self.num_devices):
            # Update kv_cache in place
            ttnn.experimental.tensor.update_cache(layer_past[i][1], value_layer[i], layer_past_len)
        for i in range(self.num_devices):
            value_layer[i] = ttnn.slice(
                layer_past[i][1],
                [0, 0, 0, 0],
                [batch - 1, 0, nearest_32(layer_past_len + 1) - 1, self.head_dim - 1],
                memory_config=self.model_config["V_CACHE_SLICE_OUTPUT_MEMCFG"],
            )
        if self.model_config["l1_sharded"]:
            for i in range(self.num_devices):
                value_layer[i] = ttnn.experimental.tensor.interleaved_to_sharded(
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
        if self.model_config["l1_sharded"]:
            for i in range(self.num_devices):
                attn_output.append(
                    ttnn.matmul(
                        attn_weights[i],  # [batch, 1, padded_local_heads, padded_layer_past_len]
                        value_layer[i],  # [batch, 1, padded_layer_past_len, head_dim]
                        program_config=self.model_config["ATTN_BATCHED_MM_PROGCFG"](
                            padded_layer_past_len // 32,
                            self.padded_local_heads // 32,
                            self.head_dim // 32,
                        ),
                        memory_config=self.model_config["ATTN_BATCH_SHARDED_MEMCFG"](
                            self.padded_local_heads,
                            self.head_dim,
                        ),
                        dtype=self.model_config["POST_SOFTMAX_MM_OUTPUT_DTYPE"],
                        compute_kernel_config=self.model_config["POST_SOFTMAX_MM_COMPUTE_KERNEL_CONFIG"],
                    )
                )
                attn_weights[i].deallocate(True)
                value_layer[i].deallocate()

            for i in range(self.num_devices):
                attn_output[i] = ttnn.experimental.tensor.sharded_to_interleaved(
                    attn_output[i], output_mem_config=self.model_config["POST_SOFTMAX_MM_OUTPUT_MEMCFG"]
                )

            # Get batch in dim 1
            for i in range(self.num_devices):
                attn_output[i] = ttnn.experimental.tensor.reshape(
                    attn_output[i], 1, batch, self.padded_local_heads, self.head_dim
                )

            # Get batch in dim 2
            for i in range(self.num_devices):
                attn_output[i] = ttnn.transpose(
                    attn_output[i],
                    -2,
                    -3,
                )

            # UNPAD
            attn_output_shape = attn_output[0].get_legacy_shape()
            for i in range(self.num_devices):
                attn_output[i] = ttnn.slice(
                    attn_output[i],
                    [0, 0, 0, 0],
                    [
                        attn_output_shape[0] - 1,
                        self.num_heads - 1,
                        attn_output_shape[2] - 1,
                        attn_output_shape[3] - 1,
                    ],
                    memory_config=self.model_config["POST_SOFTMAX_MM_OUTPUT_MEMCFG"],
                )
        else:
            for i in range(self.num_devices):
                # TODO: switch to group_attn_matmul once multiple q heads is supported (issue #5318)
                if is_wormhole_b0():
                    attn_output.append(
                        ttnn.experimental.operations.primary.transformers.attn_matmul(
                            attn_weights[i],
                            value_layer[i],
                            compute_with_storage_grid_size=device.compute_with_storage_grid_size(),
                            output_mem_config=self.model_config["POST_SOFTMAX_MM_OUTPUT_MEMCFG"],
                            output_dtype=self.model_config["POST_SOFTMAX_MM_OUTPUT_DTYPE"],  # Must be BFLOAT16
                        )
                    )
                else:
                    attn_output.append(
                        ttnn.experimental.operations.primary.transformers.group_attn_matmul(
                            attn_weights[i],
                            value_layer[i],
                            compute_with_storage_grid_size=device.compute_with_storage_grid_size(),
                            output_mem_config=self.model_config["POST_SOFTMAX_MM_OUTPUT_MEMCFG"],
                            output_dtype=self.model_config["POST_SOFTMAX_MM_OUTPUT_DTYPE"],  # Must be BFLOAT16
                        )
                    )
                attn_weights[i].deallocate(True)
                value_layer[i].deallocate()

        #########################
        ### ATTENTION SELFOUT ###
        #########################
        for i in range(self.num_devices):
            attn_output[i] = ttnn.experimental.nlp_concat_heads(
                attn_output[i],
                memory_config=self.model_config["CONCAT_HEADS_OUTPUT_MEMCFG"],
            )

        for i in range(self.num_devices):
            attn_output[i] = ttnn.matmul(
                attn_output[i],
                self.dense_weights[i],
                memory_config=self.model_config["SELFOUT_MM_OUTPUT_MEMCFG"],
                dtype=self.model_config["SELFOUT_MM_OUTPUT_DTYPE"],
                core_grid=get_falcon_default_core_grid(attn_output[i].device()),
            )

        return attn_output, layer_present
