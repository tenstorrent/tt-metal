import torch
import math
from torch import nn
from typing import Optional, Tuple

import tt_lib

from tt_models.helper_funcs import Linear as TTLinear
from tt_models.utility_functions import torch2tt_tensor, tt2torch_tensor, pad_by_zero


"""
Taken from LlamaRotaryEmbedding
"""


class FalconRotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float().to(device) / dim))
        self.register_buffer("inv_freq", inv_freq)

        # Build here to make `torch.jit.trace` work.
        self.max_seq_len_cached = max_position_embeddings
        t = torch.arange(
            self.max_seq_len_cached,
            device=self.inv_freq.device,
            dtype=self.inv_freq.dtype,
        )
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer(
            "cos_cached", emb.cos()[None, None, :, :], persistent=False
        )
        self.register_buffer(
            "sin_cached", emb.sin()[None, None, :, :], persistent=False
        )

    def forward(self, device, dtype, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        # This `if` block is unlikely to be run after we build sin/cos in `__init__`. Keep the logic here just in case.
        if seq_len > self.max_seq_len_cached:
            self.max_seq_len_cached = seq_len
            t = torch.arange(
                self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype
            )
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            # Different from paper, but it uses a different permutation in order to obtain the same calculation
            emb = torch.cat((freqs, freqs), dim=-1).to(device)
            self.register_buffer(
                "cos_cached", emb.cos()[None, None, :, :], persistent=False
            )
            self.register_buffer(
                "sin_cached", emb.sin()[None, None, :, :], persistent=False
            )
        return (
            self.cos_cached[:, :, :seq_len, ...].to(dtype=dtype),
            self.sin_cached[:, :, :seq_len, ...].to(dtype=dtype),
        )


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    # Falcon doesn't seem to use position_ids
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class TtFalconRotaryEmbedding(torch.nn.Module):
    """
    rotary_emb = FalconRotaryEmbedding(
        self.head_dim, max_position_embeddings=self.max_position_embeddings
    )
    cos, sin = rotary_emb(
        query_layer.device,
        query_layer.dtype,
        seq_len=q_len,
    )
    query_layer, key_layer = apply_rotary_pos_emb(query_layer, key_layer, cos, sin)
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

    def forward(self, layer: tt_lib.tensor.Tensor) -> tt_lib.tensor.Tensor:
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
        model_config=None,
        tt_cache_path=None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.max_position_embeddings = max_position_embeddings
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
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[tt_lib.tensor.Tensor, Optional[Tuple[torch.Tensor]]]:
        """Input shape: [batch, 1, seq_len, hidden_size]"""

        assert (
            not output_attentions
        )  # hf_reference Falcon Attention doesn't support this

        bsz = hidden_states.shape()[0]
        q_len = hidden_states.shape()[2]

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
        query_layer = self.rotary_embedding(query_layer)
        key_layer = self.rotary_embedding(key_layer)

        ################
        ### KV CACHE ###
        ################
        if past_key_value is not None:
            # TODO: This needs to be on device
            # reuse k, v, self_attention
            key_layer = tt_lib.fallback_ops.concat(
                [past_key_value[0], key_layer], dim=2
            )
            value_layer = tt_lib.fallback_ops.concat(
                [past_key_value[1], value_layer], dim=2
            )

        kv_seq_len = key_layer.shape()[-2]

        past_key_value = (key_layer, value_layer) if use_cache else None

        ########################
        ### PRE-SOFTMAX MM ###
        ########################
        # TT implementation for:
        # attn_weights = torch.matmul(query_layer, key_layer.transpose(2, 3)) / math.sqrt(self.head_dim)
        key_layer_transposed = tt_lib.tensor.transpose(
            key_layer,
            output_mem_config=self.model_config["K_TRANSPOSED_OUTPUT_MEMCFG"],
            # output_dtype=self.model_config["K_TRANSPOSED_OUTPUT_DTYPE"], # Not currently supported
        )

        attn_weights = tt_lib.tensor.matmul(
            query_layer,
            key_layer_transposed,
            output_mem_config=self.model_config["PRE_SOFTMAX_MM_OUTPUT_MEMCFG"],
            # output_dtype=self.model_config["PRE_SOFTMAX_MM_OUTPUT_DTYPE"], # Not currently supported
        )
        query_layer.deallocate()

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

        # Falcon Attention generates attention mask if attention_mask is None
        if attention_mask is None:
            attention_mask = (
                torch.ones(bsz, self.num_heads, q_len, kv_seq_len) * -100000
            ).triu(diagonal=1)
            attention_mask = torch2tt_tensor(attention_mask, self.device)

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

        ########################
        ### POST-SOFTMAX MM ###
        ########################
        attn_output = tt_lib.tensor.matmul(
            attn_weights,
            value_layer,
            output_mem_config=self.model_config["POST_SOFTMAX_MM_OUTPUT_MEMCFG"],
            # output_dtype=self.model_config["POST_SOFTMAX_MM_OUTPUT_DTYPE"], # Not currently supported
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

        return attn_output, past_key_value
