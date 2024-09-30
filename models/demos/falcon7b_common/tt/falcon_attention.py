# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import math
from torch import nn
from typing import Optional, Tuple

from models.demos.falcon7b_common.tt.model_utils import get_falcon_default_core_grid
import ttnn
from ttnn import ReplicateTensorToMesh

from models.utility_functions import (
    nearest_32,
    is_wormhole_b0,
)

from models.demos.falcon7b_common.tt.model_utils import get_weights_cached
from models.demos.falcon7b_common.tests.test_utils import tt_from_torch


class TtFalconRotaryEmbedding(torch.nn.Module):
    def __init__(
        self,
        mesh_device,
        dim,
        base_url,
        layer_num,
        max_position_embeddings=2048,
        base=10000,
        model_config=None,
        tt_cache_path=None,
        weights_dict=None,
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

        layer_name = f"{base_url}.{layer_num}.rotary_embedding_maxseq{max_position_embeddings}"
        cos_str = f"{layer_name}.cos_cached"
        sin_str = f"{layer_name}.sin_cached"

        self.tt_cos_cached = get_weights_cached(
            mesh_device,
            model_config,
            tt_cache_path,
            cos_str,
            weight_config_str="COS_CACHED_WEIGHTS",
            weights_to_cache=emb.cos()[None, None, :, :],
            weights_dict=weights_dict,
        )
        self.tt_sin_cached = get_weights_cached(
            mesh_device,
            model_config,
            tt_cache_path,
            sin_str,
            weight_config_str="SIN_CACHED_WEIGHTS",
            weights_to_cache=emb.sin()[None, None, :, :],
            weights_dict=weights_dict,
        )

    def forward(self, layer: ttnn.Tensor, token_idx: Optional[int] = None) -> ttnn.Tensor:
        seq_len = layer.shape.with_tile_padding()[2]
        assert seq_len <= self.max_seq_len_cached, "seq_len exceeds max_seq_len_cached in RotaryEmbedding!"

        output = ttnn.experimental.rotary_embedding(
            layer,
            self.tt_cos_cached,
            self.tt_sin_cached,
            token_idx,
            memory_config=self.model_config["ROTARY_EMBEDDING_OUTPUT_MEMCFG"],
        )
        return output


class TtFalconAttentionPrefill(nn.Module):
    """Mulit-Query Attention: https://arxiv.org/pdf/1911.02150.pdf"""

    def __init__(
        self,
        mesh_device,
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
        self.mesh_device = mesh_device
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
            mesh_device,
            model_config,
            tt_cache_path,
            query_key_value_str,
            weight_config_str="FUSED_QKV_MM_WEIGHTS",
            weights_to_cache=(torch.transpose(state_dict[query_key_value_str], -2, -1) if state_dict else None),
            weights_dict=weights_dict,
        )
        self.dense_weights = get_weights_cached(
            mesh_device,
            model_config,
            tt_cache_path,
            selfout_str,
            weight_config_str="SELFOUT_MM_WEIGHTS",
            weights_to_cache=(torch.transpose(state_dict[selfout_str], -2, -1) if state_dict else None),
            weights_dict=weights_dict,
        )

        self.rotary_embedding = TtFalconRotaryEmbedding(
            mesh_device,
            self.head_dim,
            base_url,
            layer_num,
            max_position_embeddings=self.max_position_embeddings,
            model_config=model_config,
            tt_cache_path=tt_cache_path,
            weights_dict=weights_dict,
        )

        self.scalar = tt_from_torch(
            torch.tensor(1 / math.sqrt(self.head_dim)).view(1, 1),
            dtype=model_config["DEFAULT_DTYPE"],
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ReplicateTensorToMesh(mesh_device),
        )

        # optimized version can utilize single float value for softmax
        if self.model_config["PREFILL_OPTIMIZED_MODE"]:
            self.scalar_for_optimized_prefill = 1 / math.sqrt(self.head_dim)

        # generate output buffer on device
        if self.model_config["PREFILL_OPTIMIZED_MODE"] and "ATTN_OUTPUT_TENSORS" not in self.model_config:
            self.model_config["ATTN_OUTPUT_TENSORS"] = {}
            # create output tensors
            for seq_len in [128, 1024, 2048]:
                tensor = torch.zeros((1, self.num_heads, seq_len, self.head_dim)).bfloat16().float()

                tt_tensors = tt_from_torch(
                    tensor.detach(),
                    ttnn.bfloat16,
                    device=self.mesh_device,
                    layout=ttnn.TILE_LAYOUT,
                    memory_config=self.model_config["ATTN_OPTIMIZED_MEMCFG"],
                    mesh_mapper=ReplicateTensorToMesh(self.mesh_device),
                )
                self.model_config["ATTN_OUTPUT_TENSORS"][seq_len] = tt_tensors

    def forward(
        self,
        hidden_states: ttnn.Tensor,
        alibi: torch.Tensor,
        attention_mask: torch.Tensor,
        llm_mode: str,
        user_id: int = 0,
        layer_past: Optional[Tuple[ttnn.Tensor]] = None,
        layer_past_len: int = 0,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[ttnn.Tensor, Optional[Tuple[ttnn.Tensor]]]:
        """
        Prefill input shape: [1, 1, seq_len, hidden_size]
        """
        assert not output_attentions

        seq_len = hidden_states.shape.with_tile_padding()[2]

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
        fused_query_key_value = ttnn.matmul(
            hidden_states,
            self.query_key_value_weights,
            memory_config=self.model_config["FUSED_QKV_MM_OUTPUT_MEMCFG"],
            dtype=self.model_config["FUSED_QKV_MM_OUTPUT_DTYPE"],
            core_grid=get_falcon_default_core_grid(hidden_states.device()),
        )

        ###########
        ### TMs ###
        ###########
        query_layer, key_layer, value_layer = ttnn.experimental.nlp_create_qkv_heads_falcon7b(
            fused_query_key_value,
            memory_config=self.model_config["CREATE_QKV_HEADS_OUTPUT_MEMCFG"],
        )
        fused_query_key_value.deallocate()

        #########################
        ### ROTARY EMBEDDINGS ###
        #########################
        query_layer = self.rotary_embedding(query_layer)
        key_layer = self.rotary_embedding(key_layer)

        ######################
        ### K CACHE UPDATE ###
        ######################
        ttnn.fill_cache(layer_past[0], key_layer, user_id)

        ######################
        ### PRE-SOFTMAX MM ###
        ######################
        key_layer_transposed = ttnn.transpose(
            key_layer,
            -2,
            -1,
            memory_config=(
                self.model_config["K_TRANSPOSED_OUTPUT_MEMCFG"]
                if llm_mode == "prefill" or self.model_config["l1_sharded"] == False
                else ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                    ttnn.BufferType.L1,
                )
            ),
        )
        key_layer.deallocate()

        attn_weights = ttnn.matmul(
            query_layer,
            key_layer_transposed,
            memory_config=self.model_config["PRE_SOFTMAX_MM_OUTPUT_MEMCFG"],
        )
        query_layer.deallocate()
        key_layer_transposed.deallocate()

        attn_weights = ttnn.mul(
            attn_weights, self.scalar, memory_config=self.model_config["PRE_SOFTMAX_SCALE_OUTPUT_MEMCFG"]
        )

        if attention_mask is not None:
            attn_weights = ttnn.add(
                attn_weights,
                attention_mask,
                memory_config=self.model_config["PRE_SOFTMAX_MASK_OUTPUT_MEMCFG"],
            )

        ###############
        ### SOFTMAX ###
        ###############
        attn_weights = ttnn.scale_mask_softmax_in_place(attn_weights)

        ######################
        ### V CACHE UPDATE ###
        ######################
        ttnn.fill_cache(layer_past[1], value_layer, user_id)

        layer_present = layer_past if use_cache else None

        ########################
        ### POST-SOFTMAX MM ###
        ########################
        attn_output = ttnn.matmul(
            attn_weights,
            value_layer,
            memory_config=self.model_config["POST_SOFTMAX_MM_OUTPUT_MEMCFG"],
        )
        attn_weights.deallocate()
        value_layer.deallocate()

        #########################
        ### ATTENTION SELFOUT ###
        #########################
        attn_output = ttnn.experimental.nlp_concat_heads(
            attn_output,
            memory_config=self.model_config["CONCAT_HEADS_OUTPUT_MEMCFG"],
        )

        attn_output = ttnn.matmul(
            attn_output,
            self.dense_weights,
            memory_config=self.model_config["SELFOUT_MM_OUTPUT_MEMCFG"],
            dtype=self.model_config["SELFOUT_MM_OUTPUT_DTYPE"],
            core_grid=get_falcon_default_core_grid(attn_output.device()),
        )

        return attn_output, layer_present

    def _optimized_forward(
        self,
        hidden_states: ttnn.Tensor,
        attention_mask: torch.Tensor,
        user_id: int = 0,
        layer_past: Optional[Tuple[ttnn.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[ttnn.Tensor, Optional[Tuple[ttnn.Tensor]]]:
        seq_len = hidden_states.shape.with_tile_padding()[2]

        #################
        ### FUSED QKV ###
        #################
        if seq_len == 2048:
            fused_query_key_value = ttnn.matmul(
                hidden_states,
                self.query_key_value_weights,
                program_config=self.model_config["FUSED_QKV_MM_OPTIMIZED_PROGCFG"],
                memory_config=self.model_config["FUSED_QKV_MM_OPTIMIZED_MEMCFG"],
                dtype=ttnn.bfloat16,
                compute_kernel_config=self.model_config["FUSED_QKV_MM_OPTIMIZED_KERNEL_CONFIG"],
            )
        else:
            fused_query_key_value = ttnn.matmul(
                hidden_states,
                self.query_key_value_weights,
                memory_config=self.model_config["FUSED_QKV_MM_OUTPUT_MEMCFG"],
                dtype=self.model_config["FUSED_QKV_MM_OUTPUT_DTYPE"],
                compute_kernel_config=self.model_config["FUSED_QKV_MM_OPTIMIZED_KERNEL_CONFIG"],
                core_grid=ttnn.CoreGrid(y=7, x=8),
            )

        ###########
        ### TMs ###
        ###########
        query_layer, key_layer, value_layer = ttnn.experimental.nlp_create_qkv_heads_falcon7b(
            fused_query_key_value,
            memory_config=self.model_config["CREATE_QKV_HEADS_OUTPUT_MEMCFG"],
        )
        fused_query_key_value.deallocate()

        #########################
        ### ROTARY EMBEDDINGS ###
        #########################
        query_layer = self.rotary_embedding(query_layer)
        key_layer = self.rotary_embedding(key_layer)

        ######################
        ### K CACHE UPDATE ###
        ######################
        ttnn.fill_cache(layer_past[0], key_layer, user_id)

        ######################
        ### PRE-SOFTMAX MM ###
        ######################

        key_layer_transposed = ttnn.transpose(
            key_layer,
            -2,
            -1,
            memory_config=self.model_config["K_TRANSPOSED_OUTPUT_MEMCFG"],
        )
        key_layer.deallocate()

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
            slices = ttnn.interleaved_to_sharded_partial(
                query_layer,
                grid_size,
                mm_activations_height_shard_spec,
                num_slices,  # num_slices
                i,  # slice_index
                ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                ttnn.ShardOrientation.ROW_MAJOR,
            )

            subblock_h = 1
            subblock_w = 1
            if seq_len == 2048:
                subblock_w = 8  # best option

            qkt_prg_cfg = self.model_config["QKT_OPTIMIZED_PROGCFG"](tiles_per_shard, seq_len, subblock_h, subblock_w)

            ### QKT MATMUL ###
            mm_slices = ttnn.matmul(
                slices,
                key_layer_transposed,
                program_config=qkt_prg_cfg,
                memory_config=self.model_config["QKTV_MM_OPTIMIZED_MEMCFG"],
                dtype=ttnn.bfloat16,
                compute_kernel_config=self.model_config["QKTV_AND_SOFTMAX_OPTIMIZED_KERNEL_CONFIG"],
            )

            softmax_program_config = self.model_config["SOFTMAX_OPTIMIZED_PROGCFG"](
                grid_size, subblock_w, mm_output_height_shard_spec[0] // 32, mm_output_height_shard_spec[1] // 32
            )
            ### SOFTMAX ###
            mm_slices = ttnn.scale_causal_mask_hw_dims_softmax_in_place(
                mm_slices,
                self.scalar_for_optimized_prefill,
                attention_mask[i],
                program_config=softmax_program_config,
                compute_kernel_config=self.model_config["QKTV_AND_SOFTMAX_OPTIMIZED_KERNEL_CONFIG"],
            )

            ### QKTV MATMUL ###
            qktv_prg_cfg = self.model_config["QKTV_MM_OPTIMIZED_PROGCFG"](tiles_per_shard, seq_len, subblock_h)
            attn_out_slices = ttnn.matmul(
                mm_slices,
                value_layer,
                program_config=qktv_prg_cfg,
                memory_config=self.model_config["QKTV_MM_OPTIMIZED_MEMCFG"],
                dtype=ttnn.bfloat16,
                compute_kernel_config=self.model_config["QKTV_AND_SOFTMAX_OPTIMIZED_KERNEL_CONFIG"],
            )

            ttnn.sharded_to_interleaved_partial(
                attn_out_slices,
                attention_outputs_concatenated,
                num_slices,
                i,
                memory_config=self.model_config["ATTN_OPTIMIZED_MEMCFG"],
            )
            attn_out_slices.deallocate(True)
            mm_slices.deallocate(True)
            slices.deallocate(True)

        # V cache update
        ttnn.fill_cache(layer_past[1], value_layer, user_id)

        layer_present = layer_past if use_cache else None

        attn_outputs = ttnn.experimental.nlp_concat_heads(
            attention_outputs_concatenated,
            memory_config=self.model_config["CONCAT_HEADS_OUTPUT_MEMCFG"],
        )
        attn_outputs = ttnn.matmul(
            attn_outputs,
            self.dense_weights,
            memory_config=self.model_config["SELFOUT_MM_OUTPUT_MEMCFG"],
            dtype=self.model_config["SELFOUT_MM_OUTPUT_DTYPE"],
            compute_kernel_config=self.model_config["SELFOUT_MM_OPTIMIZED_KERNEL_CONFIG"],
            core_grid=ttnn.CoreGrid(y=7, x=8),
        )

        return attn_outputs, layer_present


class TtFalconAttentionDecode(nn.Module):
    """Mulit-Query Attention: https://arxiv.org/pdf/1911.02150.pdf"""

    def __init__(
        self,
        mesh_device,
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
        self.mesh_device = mesh_device
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
            mesh_device,
            model_config,
            tt_cache_path,
            query_key_value_str,
            weight_config_str="FUSED_QKV_MM_WEIGHTS",
            weights_to_cache=(torch.transpose(state_dict[query_key_value_str], -2, -1) if state_dict else None),
            weights_dict=weights_dict,
        )
        self.dense_weights = get_weights_cached(
            mesh_device,
            model_config,
            tt_cache_path,
            selfout_str,
            weight_config_str="SELFOUT_MM_WEIGHTS",
            weights_to_cache=(torch.transpose(state_dict[selfout_str], -2, -1) if state_dict else None),
            weights_dict=weights_dict,
        )

        self.rotary_embedding = TtFalconRotaryEmbedding(
            mesh_device,
            self.head_dim,
            base_url,
            layer_num,
            max_position_embeddings=self.max_position_embeddings,
            model_config=model_config,
            tt_cache_path=tt_cache_path,
            weights_dict=weights_dict,
        )

        self.scalar = 1 / math.sqrt(self.head_dim)
        if self.model_config["l1_sharded"] == False:
            self.tt_scalar = tt_from_torch(
                torch.tensor(self.scalar).view(1, 1),
                dtype=model_config["DEFAULT_DTYPE"],
                device=mesh_device,
                layout=ttnn.TILE_LAYOUT,
                mesh_mapper=ReplicateTensorToMesh(mesh_device),
            )

    def forward(
        self,
        hidden_states: ttnn.Tensor,
        alibi: torch.Tensor,
        attention_mask: torch.Tensor,
        llm_mode: str,
        user_id: int = 0,
        layer_past: Optional[Tuple[ttnn.Tensor]] = None,
        layer_past_len: int = 0,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[ttnn.Tensor, Optional[Tuple[ttnn.Tensor]]]:
        """
        Prefill input shape: [batch, 1, seq_len, hidden_size]
        Decode input shape: [seq_len, 1, batch, hidden_size]
        """

        assert not output_attentions

        padded_layer_past_len = nearest_32(layer_past_len + 1)

        batch = hidden_states.shape.with_tile_padding()[2]
        q_len = hidden_states.shape.with_tile_padding()[0]
        # We always store max_position_embeddings for kv_cache,
        # so we need separate variable to store the actual len of the kv_cache
        assert layer_past is not None
        assert layer_past_len <= self.max_position_embeddings

        #################
        ### FUSED QKV ###
        #################
        fused_query_key_value = ttnn.matmul(
            hidden_states,
            self.query_key_value_weights,
            memory_config=self.model_config["FUSED_QKV_MM_OUTPUT_MEMCFG"],
            dtype=self.model_config["FUSED_QKV_MM_OUTPUT_DTYPE"],
            core_grid=get_falcon_default_core_grid(hidden_states.device()),
        )

        ###########
        ### TMs ###
        ###########
        query_layer, key_layer, value_layer = ttnn.experimental.nlp_create_qkv_heads_falcon7b(
            fused_query_key_value,
            memory_config=self.model_config["CREATE_QKV_HEADS_OUTPUT_MEMCFG"],
        )
        fused_query_key_value.deallocate(True)

        #########################
        ### ROTARY EMBEDDINGS ###
        #########################
        query_layer = self.rotary_embedding(query_layer, layer_past_len)
        key_layer = self.rotary_embedding(key_layer, layer_past_len)

        ######################
        ### K CACHE UPDATE ###
        ######################

        # Update kv_cache in place
        ttnn.update_cache(layer_past[0], key_layer, layer_past_len)
        key_layer.deallocate(True)

        # key and value layers will have kv_seq_len padded to nearest 32
        key_layer = ttnn.slice(
            layer_past[0],
            starts=(0, 0, 0, 0),
            ends=(batch, 1, nearest_32(layer_past_len + 1), self.head_dim),
            steps=(1, 1, 1, 1),
            memory_config=self.model_config["K_CACHE_SLICE_OUTPUT_MEMCFG"],
        )

        if self.model_config["l1_sharded"]:
            key_layer = ttnn.interleaved_to_sharded(
                key_layer,
                self.model_config["ATTN_BATCH_SHARDED_MEMCFG"](padded_layer_past_len, self.head_dim),
            )

            # Pad and transpose Q for batched matmul
            query_layer = ttnn.pad(
                query_layer,
                [1, self.padded_local_heads, batch, self.head_dim],
                [0, 0, 0, 0],
                0.0,
            )
            query_layer = ttnn.transpose(
                query_layer,
                -2,
                -3,
            )
            query_layer = ttnn.reshape_on_device(
                query_layer,
                batch,
                1,
                self.padded_local_heads,
                self.head_dim,  # Batch must be in dim 0 to match K cache
            )
            query_layer = ttnn.interleaved_to_sharded(
                query_layer,
                self.model_config["ATTN_BATCH_SHARDED_MEMCFG"](self.padded_local_heads, self.head_dim),
            )

        ######################
        ### PRE-SOFTMAX MM ###
        ######################

        key_layer_transposed = ttnn.transpose(
            key_layer,
            -2,
            -1,
            memory_config=(
                self.model_config["K_TRANSPOSED_OUTPUT_MEMCFG"]
                if self.model_config["l1_sharded"] == False
                else ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                    ttnn.BufferType.L1,
                )
            ),
        )
        key_layer.deallocate()

        if self.model_config["l1_sharded"]:
            attn_weights = ttnn.matmul(
                query_layer,  # [batch, 1, padded_local_heads, head_dim]
                key_layer_transposed,  # [batch, 1, head_dim, padded_layer_past_len]
                program_config=self.model_config["ATTN_BATCHED_MM_PROGCFG"](
                    self.head_dim // 32, self.padded_local_heads // 32, padded_layer_past_len // 32
                ),
                memory_config=self.model_config["ATTN_BATCH_SHARDED_MEMCFG"](
                    self.padded_local_heads, padded_layer_past_len
                ),
                dtype=self.model_config["PRE_SOFTMAX_MM_OUTPUT_DTYPE"],  # Must be BFLOAT16
                compute_kernel_config=self.model_config["PRE_SOFTMAX_MM_COMPUTE_KERNEL_CONFIG"],
            )
            query_layer.deallocate()
            key_layer_transposed.deallocate(True)
        elif is_wormhole_b0():
            attn_weights = ttnn.experimental.attn_matmul(
                query_layer,
                key_layer_transposed,
                compute_with_storage_grid_size=self.mesh_device.compute_with_storage_grid_size(),
                memory_config=self.model_config["PRE_SOFTMAX_MM_OUTPUT_MEMCFG"],
                dtype=self.model_config["PRE_SOFTMAX_MM_OUTPUT_DTYPE"],  # Must be BFLOAT16
            )
            query_layer.deallocate()
            key_layer_transposed.deallocate()
        else:
            attn_weights = ttnn.experimental.group_attn_matmul(
                query_layer,
                key_layer_transposed,
                compute_with_storage_grid_size=self.mesh_device.compute_with_storage_grid_size(),
                memory_config=self.model_config["PRE_SOFTMAX_MM_OUTPUT_MEMCFG"],
                dtype=self.model_config["PRE_SOFTMAX_MM_OUTPUT_DTYPE"],  # Must be BFLOAT16
            )
            query_layer.deallocate()
            key_layer_transposed.deallocate(True)

        if self.model_config["l1_sharded"] == False:
            attn_weights = ttnn.mul(
                attn_weights, self.tt_scalar, memory_config=self.model_config["PRE_SOFTMAX_SCALE_OUTPUT_MEMCFG"]
            )
            if attention_mask is not None:
                attn_weights = ttnn.add(
                    attn_weights,
                    attention_mask,
                    memory_config=self.model_config["PRE_SOFTMAX_MASK_OUTPUT_MEMCFG"],
                )
            ###############
            ### SOFTMAX ###
            ###############
            attn_weights = ttnn.scale_mask_softmax_in_place(attn_weights)
        else:
            ###############
            ### SOFTMAX ###
            ###############
            attn_weights = ttnn.scale_mask_softmax_in_place(
                attn_weights,
                scale=self.scalar,
                mask=attention_mask,
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

        # Update kv_cache in place
        ttnn.update_cache(layer_past[1], value_layer, layer_past_len)

        value_layer = ttnn.slice(
            layer_past[1],
            starts=(0, 0, 0, 0),
            ends=(batch, 1, nearest_32(layer_past_len + 1), self.head_dim),
            steps=(1, 1, 1, 1),
            memory_config=self.model_config["V_CACHE_SLICE_OUTPUT_MEMCFG"],
        )
        if self.model_config["l1_sharded"]:
            value_layer = ttnn.interleaved_to_sharded(
                value_layer,
                self.model_config["ATTN_BATCH_SHARDED_MEMCFG"](padded_layer_past_len, self.head_dim),
            )

        layer_present = layer_past if use_cache else None

        ########################
        ### POST-SOFTMAX MM ###
        ########################
        if self.model_config["l1_sharded"]:
            attn_output = ttnn.matmul(
                attn_weights,  # [batch, 1, padded_local_heads, padded_layer_past_len]
                value_layer,  # [batch, 1, padded_layer_past_len, head_dim]
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
            attn_weights.deallocate(True)
            value_layer.deallocate()

            attn_output = ttnn.sharded_to_interleaved(
                attn_output, memory_config=self.model_config["POST_SOFTMAX_MM_OUTPUT_MEMCFG"]
            )

            # Get batch in dim 1
            attn_output = ttnn.reshape_on_device(attn_output, 1, batch, self.padded_local_heads, self.head_dim)

            # Get batch in dim 2
            attn_output = ttnn.transpose(
                attn_output,
                -2,
                -3,
            )

            # UNPAD
            attn_output_shape = attn_output.shape.with_tile_padding()
            attn_output = ttnn.slice(
                attn_output,
                starts=(0, 0, 0, 0),
                ends=(
                    attn_output_shape[0],
                    self.num_heads,
                    attn_output_shape[2],
                    attn_output_shape[3],
                ),
                steps=(1, 1, 1, 1),
                memory_config=self.model_config["POST_SOFTMAX_MM_OUTPUT_MEMCFG"],
            )
        else:
            # TODO: switch to group_attn_matmul once multiple q heads is supported (issue #5318)
            if is_wormhole_b0():
                attn_output = ttnn.experimental.attn_matmul(
                    attn_weights,
                    value_layer,
                    compute_with_storage_grid_size=self.mesh_device.compute_with_storage_grid_size(),
                    memory_config=self.model_config["POST_SOFTMAX_MM_OUTPUT_MEMCFG"],
                    dtype=self.model_config["POST_SOFTMAX_MM_OUTPUT_DTYPE"],  # Must be BFLOAT16
                )
            else:
                attn_output = ttnn.experimental.group_attn_matmul(
                    attn_weights,
                    value_layer,
                    compute_with_storage_grid_size=self.mesh_device.compute_with_storage_grid_size(),
                    memory_config=self.model_config["POST_SOFTMAX_MM_OUTPUT_MEMCFG"],
                    dtype=self.model_config["POST_SOFTMAX_MM_OUTPUT_DTYPE"],  # Must be BFLOAT16
                )
            attn_weights.deallocate(True)
            value_layer.deallocate()

        #########################
        ### ATTENTION SELFOUT ###
        #########################
        attn_output = ttnn.experimental.nlp_concat_heads(
            attn_output,
            memory_config=self.model_config["CONCAT_HEADS_OUTPUT_MEMCFG"],
        )

        attn_output = ttnn.matmul(
            attn_output,
            self.dense_weights,
            memory_config=self.model_config["SELFOUT_MM_OUTPUT_MEMCFG"],
            dtype=self.model_config["SELFOUT_MM_OUTPUT_DTYPE"],
            core_grid=get_falcon_default_core_grid(attn_output.device()),
        )

        return attn_output, layer_present
