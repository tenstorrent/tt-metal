import torch
import math
from torch import nn
from typing import Optional, Tuple

import tt_lib

from models.helper_funcs import Linear as TTLinear
from models.utility_functions import (
    torch2tt_tensor,
    tt2torch_tensor,
    pad_by_zero,
    nearest_32,
)


class TtFalconRotaryEmbedding(torch.nn.Module):
    """
    See FalconRotaryEmbedding from hf_modeling_falcon.py
    """

    def __init__(
        self,
        tt_device,
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

        # Build here to make `torch.jit.trace` work.
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
        if tt_cache_path is not None:
            # if 0:
            self.tt_cos_cached = tt_lib.tensor.load_tensor(
                str(
                    tt_cache_path
                    / f"{layer_name}.cos_cached_{self.model_config['COS_CACHED_WEIGHTS_DTYPE'].name}.bin"
                )
            ).to(tt_device, self.model_config["COS_CACHED_WEIGHTS_MEMCFG"])
            self.tt_sin_cached = tt_lib.tensor.load_tensor(
                str(
                    tt_cache_path
                    / f"{layer_name}.sin_cached_{self.model_config['SIN_CACHED_WEIGHTS_DTYPE'].name}.bin"
                )
            ).to(tt_device, self.model_config["SIN_CACHED_WEIGHTS_MEMCFG"])
        else:
            self.tt_cos_cached = torch2tt_tensor(
                emb.cos()[None, None, :, :],
                tt_device,
                tt_memory_config=self.model_config["COS_CACHED_WEIGHTS_MEMCFG"],
                tt_dtype=self.model_config["COS_CACHED_WEIGHTS_DTYPE"],
            )
            self.tt_sin_cached = torch2tt_tensor(
                emb.sin()[None, None, :, :],
                tt_device,
                tt_memory_config=self.model_config["SIN_CACHED_WEIGHTS_MEMCFG"],
                tt_dtype=self.model_config["SIN_CACHED_WEIGHTS_DTYPE"],
            )

    def forward(self, layer: tt_lib.tensor.Tensor, token_idx: Optional[int] = None) -> tt_lib.tensor.Tensor:
        # x: [bs, num_attention_heads, seq_len, head_size]
        # seq_len > self.max_seq_len_cached block is unlikely to be run after we build sin/cos in `__init__`. Keep the logic here just in case.
        seq_len = layer.shape()[2]
        assert (
            seq_len <= self.max_seq_len_cached
        ), "seq_len exceeds max_seq_len_cached in RotaryEmbedding!"

        return tt_lib.tensor.rotary_embedding(
            layer,
            self.tt_cos_cached,
            self.tt_sin_cached,
            token_idx,
            output_mem_config=self.model_config["ROTARY_EMBEDDING_OUTPUT_MEMCFG"],
            # output_dtype=self.model_config["ROTARY_EMBEDDING_OUTPUT_DTYPE"], # Not currently supported
        )


class TtFalconAttention(nn.Module):
    """Mulit-Query Attention: https://arxiv.org/pdf/1911.02150.pdf"""

    def __init__(
        self,
        device,
        state_dict,
        base_url,
        layer_num,
        hidden_size: int,
        num_heads: int,
        max_position_embeddings: int = 2048,
        llm_mode: str = "prefill",
        model_config=None,
        tt_cache_path=None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.max_position_embeddings = max_position_embeddings
        self.llm_mode = llm_mode
        self.device = device
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
        if tt_cache_path is not None:
            # if 0:
            self.query_key_value_weights = tt_lib.tensor.load_tensor(
                str(
                    tt_cache_path
                    / f"{query_key_value_str}_{self.model_config['FUSED_QKV_MM_WEIGHTS_DTYPE'].name}.bin"
                )
            ).to(device, self.model_config["FUSED_QKV_MM_WEIGHTS_MEMCFG"])
            self.dense_weights = tt_lib.tensor.load_tensor(
                str(
                    tt_cache_path
                    / f"{selfout_str}_{self.model_config['SELFOUT_MM_WEIGHTS_DTYPE'].name}.bin"
                )
            ).to(device, self.model_config["SELFOUT_MM_WEIGHTS_MEMCFG"])
        else:
            # TODO: Take in model_config instead of hardcoding dtypes/mem_configs
            # self.query_key_value_weights = torch2tt_tensor(torch.rand(4544, 4672), self.device)
            self.query_key_value_weights = torch2tt_tensor(
                torch.transpose(
                    self.state_dict[query_key_value_str],
                    -2,
                    -1,
                ),
                self.device,
                tt_memory_config=self.model_config["FUSED_QKV_MM_WEIGHTS_MEMCFG"],
                tt_dtype=self.model_config["FUSED_QKV_MM_WEIGHTS_DTYPE"],
            )

            # self.dense_weights = torch2tt_tensor(torch.rand(4544, 4544), self.device)
            self.dense_weights = torch2tt_tensor(
                torch.transpose(
                    self.state_dict[selfout_str],
                    -2,
                    -1,
                ),
                self.device,
                tt_memory_config=self.model_config["SELFOUT_MM_WEIGHTS_MEMCFG"],
                tt_dtype=self.model_config["SELFOUT_MM_WEIGHTS_DTYPE"],
            )

        self.rotary_embedding = TtFalconRotaryEmbedding(
            self.device,
            self.head_dim,
            base_url,
            layer_num,
            max_position_embeddings=self.max_position_embeddings,
            model_config=model_config,
            tt_cache_path=tt_cache_path,
        )

        self.scalar = pad_by_zero(
            torch.Tensor([1 / math.sqrt(self.head_dim)]), self.device
        )[0]

    def forward(
        self,
        hidden_states: tt_lib.tensor.Tensor,
        alibi: torch.Tensor,
        attention_mask: torch.Tensor,
        layer_past: Optional[Tuple[tt_lib.tensor.Tensor]] = None,
        layer_past_len: int = 0,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[tt_lib.tensor.Tensor, Optional[Tuple[tt_lib.tensor.Tensor]]]:
        """
        Prefill input shape: [batch, 1, seq_len, hidden_size]
        Decode input shape: [seq_len, 1, batch, hidden_size]
        """

        assert (
            not output_attentions
        )  # hf_reference Falcon Attention doesn't support this

        if self.llm_mode == "prefill":
            batch = hidden_states.shape()[0]
            q_len = hidden_states.shape()[2]
            assert layer_past is None
        elif self.llm_mode == "decode":
            batch = hidden_states.shape()[2]
            q_len = hidden_states.shape()[0]
            # We always store max_position_embeddings for kv_cache,
            # so we need separate variable to store the actual len of the kv_cache
            # TODO: Can layer_past_len be zero??
            assert layer_past is not None
            assert layer_past_len > 0 and layer_past_len <= self.max_position_embeddings
        else:
            raise NotImplementedError(
                f"Llm mode {llm_mode} is not supported! Must be one of prefill or decode."
            )

        #################
        ### FUSED QKV ###
        #################
        fused_query_key_value = tt_lib.tensor.falcon_fused_qkv_matmul(
            hidden_states,
            self.query_key_value_weights,
            output_mem_config=self.model_config["FUSED_QKV_MM_OUTPUT_MEMCFG"],
            output_dtype=self.model_config["FUSED_QKV_MM_OUTPUT_DTYPE"],
        )  # b, 1, seq_len, 73 * head_dim

        ###########
        ### TMs ###
        ###########
        query_layer, key_layer, value_layer = tt_lib.tensor.nlp_create_qkv_heads(
            fused_query_key_value,
            output_mem_config=self.model_config["CREATE_QKV_HEADS_OUTPUT_MEMCFG"],
            # output_dtype=self.model_config["CREATE_QKV_HEADS_OUTPUT_DTYPE"], # Not currently supported
        )
        fused_query_key_value.deallocate()

        #########################
        ### ROTARY EMBEDDINGS ###
        #########################
        if self.llm_mode == "prefill":
            query_layer = self.rotary_embedding(query_layer)
            key_layer = self.rotary_embedding(key_layer)
        elif self.llm_mode == "decode":
            query_layer = self.rotary_embedding(query_layer, layer_past_len + 1)
            key_layer = self.rotary_embedding(key_layer, layer_past_len + 1)

        ######################
        ### K CACHE UPDATE ###
        ######################
        if self.llm_mode == "prefill":
            # TODO: Fill kv_cache
            pass

        elif self.llm_mode == "decode":
            # Update kv_cache in place
            tt_lib.tensor.update_cache(layer_past[0], key_layer, layer_past_len)
            # key and value layers will have kv_seq_len padded to nearest 32
            key_layer = tt_lib.tensor.unpad(
                layer_past[0],
                [0, 0, 0, 0],
                [batch - 1, 0, nearest_32(layer_past_len + 1) - 1, self.head_dim - 1],
                output_mem_config=self.model_config["K_CACHE_SLICE_OUTPUT_MEMCFG"],
            )

        kv_seq_len = key_layer.shape()[-2]
        layer_present = layer_past if use_cache else None

        ######################
        ### PRE-SOFTMAX MM ###
        ######################
        # TT implementation for:
        # attn_weights = torch.matmul(query_layer, key_layer.transpose(2, 3)) / math.sqrt(self.head_dim)
        key_layer_transposed = tt_lib.tensor.transpose(
            key_layer,
            output_mem_config=self.model_config["K_TRANSPOSED_OUTPUT_MEMCFG"],
            # output_dtype=self.model_config["K_TRANSPOSED_OUTPUT_DTYPE"], # Not currently supported
        )

        if self.llm_mode == "prefill":
            attn_weights = tt_lib.tensor.matmul(
                query_layer,
                key_layer_transposed,
                output_mem_config=self.model_config["PRE_SOFTMAX_MM_OUTPUT_MEMCFG"],
                # output_dtype=self.model_config["PRE_SOFTMAX_MM_OUTPUT_DTYPE"], # Not currently supported
            )
        elif self.llm_mode == "decode":
            attn_weights = tt_lib.operations.primary.transformers.attn_matmul(
                query_layer,
                key_layer_transposed,
                compute_with_storage_grid_size=tt_lib.tensor.CoreCoord(12, 9),
                output_mem_config=self.model_config["PRE_SOFTMAX_MM_OUTPUT_MEMCFG"],
                output_dtype=self.model_config[
                    "PRE_SOFTMAX_MM_OUTPUT_DTYPE"
                ],  # Must be BFLOAT16
            )
        query_layer.deallocate()
        key_layer_transposed.deallocate()

        attn_weights = tt_lib.tensor.bcast(
            attn_weights,
            self.scalar,
            tt_lib.tensor.BcastOpMath.MUL,
            tt_lib.tensor.BcastOpDim.HW,
            output_mem_config=self.model_config["PRE_SOFTMAX_SCALE_OUTPUT_MEMCFG"],
            # output_dtype=self.model_config["PRE_SOFTMAX_SCALE_OUTPUT_DTYPE"], # Not currently supported
        )  # b, self.num_heads, q_len, kv_seq_len

        ###############
        ### SOFTMAX ###
        ###############
        # TODO: Replace with scaled_softmax_attention_mask from BERT

        # TODO: C can be 1 if we have bcast add along C; otherwise; we need to repeat along C
        attn_weights = tt_lib.tensor.add(
            attn_weights,
            attention_mask,
            output_mem_config=self.model_config["PRE_SOFTMAX_MASK_OUTPUT_MEMCFG"],
            # output_dtype=self.model_config["PRE_SOFTMAX_MASK_OUTPUT_DTYPE"], # Not currently supported
        )

        # TT implementation for:
        # PyTorch: upcast attention to fp32
        # attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_layer.dtype)
        attn_weights = tt_lib.operations.primary.softmax_in_place(
            attn_weights,
            # output_mem_config=self.model_config["SOFTMAX_OUTPUT_MEMCFG"], # Not needed since in place
            # output_dtype=self.model_config["SOFTMAX_OUTPUT_DTYPE"],
        )

        ######################
        ### V CACHE UPDATE ###
        ######################
        if self.llm_mode == "prefill":
            # TODO: Fill kv_cache
            pass

        elif self.llm_mode == "decode":
            # Update kv_cache in place
            tt_lib.tensor.update_cache(layer_past[1], value_layer, layer_past_len)
            # key and value layers will have kv_seq_len padded to nearest 32
            value_layer = tt_lib.tensor.unpad(
                layer_past[1],
                [0, 0, 0, 0],
                [batch - 1, 0, nearest_32(layer_past_len + 1) - 1, self.head_dim - 1],
                output_mem_config=self.model_config["V_CACHE_SLICE_OUTPUT_MEMCFG"],
            )

        ########################
        ### POST-SOFTMAX MM ###
        ########################
        if self.llm_mode == "prefill":
            attn_output = tt_lib.tensor.matmul(
                attn_weights,
                value_layer,
                output_mem_config=self.model_config["POST_SOFTMAX_MM_OUTPUT_MEMCFG"],
                # output_dtype=self.model_config["POST_SOFTMAX_MM_OUTPUT_DTYPE"], # Not currently supported
            )
        elif self.llm_mode == "decode":
            attn_output = tt_lib.operations.primary.transformers.attn_matmul(
                attn_weights,
                value_layer,
                compute_with_storage_grid_size=tt_lib.tensor.CoreCoord(12, 9),
                output_mem_config=self.model_config["POST_SOFTMAX_MM_OUTPUT_MEMCFG"],
                output_dtype=self.model_config[
                    "POST_SOFTMAX_MM_OUTPUT_DTYPE"
                ],  # Must be BFLOAT16
            )
        attn_weights.deallocate()

        #########################
        ### ATTENTION SELFOUT ###
        #########################
        attn_output = tt_lib.tensor.nlp_concat_heads(
            attn_output,
            output_mem_config=self.model_config["CONCAT_HEADS_OUTPUT_MEMCFG"],
            # output_dtype=self.model_config["CONCAT_HEADS_OUTPUT_DTYPE"], # Not currently supported
        )
        attn_output = tt_lib.tensor.falcon_selfout_matmul(
            attn_output,
            self.dense_weights,
            output_mem_config=self.model_config["SELFOUT_MM_OUTPUT_MEMCFG"],
            output_dtype=self.model_config["SELFOUT_MM_OUTPUT_DTYPE"],
        )

        return attn_output, layer_present
