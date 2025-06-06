# Copyright 2023 DeepSeek-AI and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" PyTorch DeepSeek Attention modules."""

import warnings
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from transformers.cache_utils import Cache
from transformers.utils import is_flash_attn_2_available, is_flash_attn_greater_or_equal_2_10, logging

import ttnn

from .rmsnorm import DeepseekV3RMSNorm
from .rope import (
    DeepseekV3DynamicNTKScalingRotaryEmbedding,
    DeepseekV3LinearScalingRotaryEmbedding,
    DeepseekV3RotaryEmbedding,
    DeepseekV3YarnRotaryEmbedding,
    apply_rotary_pos_emb,
    yarn_get_mscale,
)

if is_flash_attn_2_available():
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # noqa


logger = logging.get_logger(__name__)


def _get_unpad_data(attention_mask):
    """Extract unpadded data indices for Flash Attention variable-length processing.

    This function is used by Flash Attention to efficiently handle sequences with padding
    by extracting only the non-padded tokens and their cumulative sequence lengths.

    Args:
        attention_mask: [batch_size, seq_len] - 1 for valid tokens, 0 for padding

    Returns:
        indices: [total_tokens] - flat indices of non-padded tokens
        cu_seqlens: [batch_size + 1] - cumulative sequence lengths (0, seq1_len, seq1_len+seq2_len, ...)
        max_seqlen_in_batch: int - maximum sequence length in the batch

    Example:
        attention_mask = [[1, 1, 1, 0],    # seq_len=3
                         [1, 1, 0, 0]]     # seq_len=2

        Returns:
        - indices = [0, 1, 2, 4, 5]  # positions of non-padded tokens in flattened tensor
        - cu_seqlens = [0, 3, 5]     # cumulative lengths: 0, 3, 3+2=5
        - max_seqlen_in_batch = 3
    """
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.torch.int32), (1, 0))
    return (
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )


# Copied from transformers.models.llama.modeling_llama.repeat_kv
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)

    Used for converting from Multi-Query Attention (MQA) or Grouped-Query Attention (GQA)
    to full Multi-Head Attention by repeating key/value heads.

    Args:
        hidden_states: [batch, num_kv_heads, seq_len, head_dim]
        n_rep: number of times to repeat each key/value head

    Returns:
        [batch, num_kv_heads * n_rep, seq_len, head_dim]

    Example:
        Input: [1, 8, 100, 64] with n_rep=16
        Output: [1, 128, 100, 64] (each of 8 heads repeated 16 times)

    Note: In DeepSeekV3, this would be used if num_key_value_heads != num_attention_heads,
    but the config shows both are 128, so n_rep=1 (no actual repetition needed).
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


# Copied from transformers.models.llama.modeling_llama.LlamaAttention with Llama->DeepseekV3
class DeepseekV3Attention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper

    This implementation uses LoRA compression for query projections and MQA (Multi-Query Attention)
    style compression for key-value projections to reduce memory and computation.

    Pseudo-code:
    ```
    def forward(hidden_states, attention_mask, position_ids, past_key_value, use_cache):
        # Input: hidden_states [batch, seq_len, hidden_size=7168]
        # Output: attn_output [batch, seq_len, hidden_size=7168]

        # === Query Projection with LoRA ===
        # Step 1: Down-project to low rank
        q_compressed = q_a_proj(hidden_states)  # [batch, seq, q_lora_rank=1536]
        q_compressed = q_a_layernorm(q_compressed)  # [batch, seq, 1536]

        # Step 2: Up-project to full query size
        q = q_b_proj(q_compressed)  # [batch, seq, num_heads*q_head_dim=128*192=24576]
        q = q.reshape(batch, seq, num_heads=128, q_head_dim=192)
        q = q.transpose(1, 2)  # [batch, 128, seq, 192]

        # Step 3: Split into non-rotary and rotary parts
        q_nope = q[..., :qk_nope_head_dim]  # [batch, 128, seq, 128]
        q_pe = q[..., qk_nope_head_dim:]    # [batch, 128, seq, 64]

        # === Key-Value Projection with Compression ===
        # Step 1: Project to compressed KV + rope key
        kv_compressed = kv_a_proj_with_mqa(hidden_states)  # [batch, seq, kv_lora_rank+qk_rope_head_dim=512+64=576]

        # Step 2: Split compressed KV and rope key
        compressed_kv = kv_compressed[..., :kv_lora_rank]     # [batch, seq, 512]
        k_pe = kv_compressed[..., kv_lora_rank:]              # [batch, seq, 64]
        k_pe = k_pe.unsqueeze(2).transpose(1, 2)              # [batch, 1, seq, 64] (MQA)

        # Step 3: Up-project compressed KV
        compressed_kv = kv_a_layernorm(compressed_kv)  # [batch, seq, 512]
        kv = kv_b_proj(compressed_kv)  # [batch, seq, num_heads*(qk_nope_head_dim+v_head_dim)=128*(128+128)=32768]
        kv = kv.reshape(batch, seq, num_heads=128, 256).transpose(1, 2)  # [batch, 128, seq, 256]

        # Step 4: Split into key (non-rotary) and value
        k_nope = kv[..., :qk_nope_head_dim]           # [batch, 128, seq, 128]
        value_states = kv[..., qk_nope_head_dim:]     # [batch, 128, seq, 128]

        # === RoPE Application ===
        cos, sin = rotary_emb(value_states, seq_len=kv_seq_len)  # [seq_len, 64]
        q_pe, k_pe = apply_rotary_pos_emb(q_pe, k_pe, cos, sin, position_ids)

        # === Concatenate Non-Rotary and Rotary Parts ===
        query_states = concat([q_nope, q_pe], dim=-1)  # [batch, 128, seq, 192]
        key_states = concat([k_nope, k_pe.expand(-1, num_heads, -1, -1)], dim=-1)  # [batch, 128, seq, 192]

        # === KV Cache Update ===
        if past_key_value is not None:
            key_states, value_states = past_key_value.update(key_states, value_states, layer_idx)
            kv_seq_len = key_states.shape[-2]

        # === Standard Attention ===
        attn_weights = query_states @ key_states.transpose(-1, -2)  # [batch, 128, seq, kv_seq_len]
        attn_weights = attn_weights * softmax_scale  # q_head_dim^(-0.5) * mscale
        attn_weights = attn_weights + attention_mask
        attn_weights = softmax(attn_weights, dim=-1)

        attn_output = attn_weights @ value_states  # [batch, 128, seq, v_head_dim=128]
        attn_output = attn_output.transpose(1, 2)  # [batch, seq, 128, 128]
        attn_output = attn_output.reshape(batch, seq, num_heads*v_head_dim=16384)

        # === Output Projection ===
        attn_output = o_proj(attn_output)  # [batch, seq, hidden_size=7168]

        return attn_output, attn_weights, past_key_value
    ```

    Shape Examples:

    **Prefill (seq_len=100):**
    - Input: [1, 100, 7168]
    - Q LoRA: [1, 100, 7168] -> [1, 100, 1536] -> [1, 100, 24576] -> [1, 128, 100, 192]
    - KV Compression: [1, 100, 7168] -> [1, 100, 576] -> [1, 128, 100, 256] + [1, 1, 100, 64]
    - Attention: [1, 128, 100, 192] @ [1, 128, 100, 192]^T -> [1, 128, 100, 100]
    - Output: [1, 100, 7168]

    **Decode (seq_len=1, with cache):**
    - Input: [1, 1, 7168]
    - New Q: [1, 128, 1, 192]
    - New K,V: [1, 128, 1, 192], [1, 128, 1, 128]
    - Cached K,V: [1, 128, prev_len, 192], [1, 128, prev_len, 128]
    - Combined K,V: [1, 128, prev_len+1, 192], [1, 128, prev_len+1, 128]
    - Attention: [1, 128, 1, 192] @ [1, 128, prev_len+1, 192]^T -> [1, 128, 1, prev_len+1]
    - Output: [1, 1, 7168]

    Key Features:
    - **LoRA Compression**: Query projection uses rank-1536 bottleneck (7168->1536->24576)
    - **KV Compression**: Key-value projection uses rank-512 bottleneck (7168->576->32768+64)
    - **MQA for RoPE**: Key RoPE part is single-headed [1, 1, seq, 64], expanded to all heads
    - **Separate RoPE/NoRoPE**: Splits Q,K into rotary (64d) and non-rotary (128d) parts
    - **YARN Scaling**: Uses mscale factor in softmax_scale for length extrapolation

    Memory Savings:
    - Standard attention: 3 * (7168 * 128 * 192) = ~525M parameters
    - This implementation: (7168*1536 + 1536*24576) + (7168*576 + 512*32768) ≈ 67M parameters
    - ~8x reduction in attention projection parameters
    """

    def __init__(self, config, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing `layer_idx` is not recommended and will "
                "to errors during the forward call, if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads

        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.q_lora_rank = config.q_lora_rank
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.kv_lora_rank = config.kv_lora_rank
        self.v_head_dim = config.v_head_dim
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.q_head_dim = config.qk_nope_head_dim + config.qk_rope_head_dim

        self.is_causal = True

        if self.q_lora_rank is None:
            self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.q_head_dim, bias=False)
        else:
            self.q_a_proj = nn.Linear(self.hidden_size, config.q_lora_rank, bias=config.attention_bias)
            self.q_a_layernorm = DeepseekV3RMSNorm(config.q_lora_rank)
            self.q_b_proj = nn.Linear(config.q_lora_rank, self.num_heads * self.q_head_dim, bias=False)

        self.kv_a_proj_with_mqa = nn.Linear(
            self.hidden_size,
            config.kv_lora_rank + config.qk_rope_head_dim,
            bias=config.attention_bias,
        )
        self.kv_a_layernorm = DeepseekV3RMSNorm(config.kv_lora_rank)
        self.kv_b_proj = nn.Linear(
            config.kv_lora_rank,
            self.num_heads * (self.q_head_dim - self.qk_rope_head_dim + self.v_head_dim),
            bias=False,
        )

        self.o_proj = nn.Linear(
            self.num_heads * self.v_head_dim,
            self.hidden_size,
            bias=config.attention_bias,
        )
        self._init_rope()

        self.softmax_scale = self.q_head_dim ** (-0.5)
        if self.config.rope_scaling is not None:
            mscale_all_dim = self.config.rope_scaling.get("mscale_all_dim", 0)
            scaling_factor = self.config.rope_scaling["factor"]
            if mscale_all_dim:
                mscale = yarn_get_mscale(scaling_factor, mscale_all_dim)
                self.softmax_scale = self.softmax_scale * mscale * mscale

    def _init_rope(self):
        if self.config.rope_scaling is None:
            self.rotary_emb = DeepseekV3RotaryEmbedding(
                self.qk_rope_head_dim,
                max_position_embeddings=self.max_position_embeddings,
                base=self.rope_theta,
            )
        else:
            scaling_type = self.config.rope_scaling["type"]
            scaling_factor = self.config.rope_scaling["factor"]
            if scaling_type == "linear":
                self.rotary_emb = DeepseekV3LinearScalingRotaryEmbedding(
                    self.qk_rope_head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            elif scaling_type == "dynamic":
                self.rotary_emb = DeepseekV3DynamicNTKScalingRotaryEmbedding(
                    self.qk_rope_head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            elif scaling_type == "yarn":
                kwargs = {
                    key: self.config.rope_scaling[key]
                    for key in [
                        "original_max_position_embeddings",
                        "beta_fast",
                        "beta_slow",
                        "mscale",
                        "mscale_all_dim",
                    ]
                    if key in self.config.rope_scaling
                }
                self.rotary_emb = DeepseekV3YarnRotaryEmbedding(
                    self.qk_rope_head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                    **kwargs,
                )
            else:
                raise ValueError(f"Unknown RoPE scaling type {scaling_type}")

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.v_head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )
        bsz, q_len, _ = hidden_states.size()

        if self.q_lora_rank is None:
            q = self.q_proj(hidden_states)
        else:
            q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
        q = q.view(bsz, q_len, self.num_heads, self.q_head_dim).transpose(1, 2)
        q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
        compressed_kv, k_pe = torch.split(compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        k_pe = k_pe.view(bsz, q_len, 1, self.qk_rope_head_dim).transpose(1, 2)
        kv = (
            self.kv_b_proj(self.kv_a_layernorm(compressed_kv))
            .view(bsz, q_len, self.num_heads, self.qk_nope_head_dim + self.v_head_dim)
            .transpose(1, 2)
        )

        k_nope, value_states = torch.split(kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
        kv_seq_len = value_states.shape[-2]
        if past_key_value is not None:
            if self.layer_idx is None:
                raise ValueError(
                    f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                    "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                    "with a layer index."
                )
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)

        q_pe, k_pe = apply_rotary_pos_emb(q_pe, k_pe, cos, sin, position_ids)

        query_states = k_pe.new_empty(bsz, self.num_heads, q_len, self.q_head_dim)
        query_states[:, :, :, : self.qk_nope_head_dim] = q_nope
        query_states[:, :, :, self.qk_nope_head_dim :] = q_pe

        key_states = k_pe.new_empty(bsz, self.num_heads, q_len, self.q_head_dim)
        key_states[:, :, :, : self.qk_nope_head_dim] = k_nope
        key_states[:, :, :, self.qk_nope_head_dim :] = k_pe
        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.softmax_scale

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )
        assert attention_mask is not None
        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.v_head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.v_head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.reshape(bsz, q_len, self.num_heads * self.v_head_dim)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


# Copied from transformers.models.llama.modeling_llama.LlamaFlashAttention2 with Llama->DeepseekV3
class DeepseekV3FlashAttention2(DeepseekV3Attention):
    """
    DeepseekV3 flash attention module. This module inherits from `DeepseekV3Attention` as the weights of the module stays
    untouched. The only required change would be on the forward pass where it needs to correctly call the public API of
    flash attention and deal with padding tokens in case the input contains any of them.

    Flash Attention optimizes attention computation using:
    - Tiled computation to reduce memory usage from O(N²) to O(N)
    - Fused kernels for better GPU utilization
    - Support for variable-length sequences without padding

    Pseudo-code:
    ```
    def forward(hidden_states, attention_mask, position_ids, past_key_value, use_cache):
        # Input: hidden_states [batch, seq_len, hidden_size=7168]
        # Output: attn_output [batch, seq_len, hidden_size=7168]

        # === Same Q, K, V projections as base DeepseekV3Attention ===
        # (See DeepseekV3Attention docstring for details)
        query_states = process_queries(hidden_states)  # [batch, num_heads, seq, q_head_dim]
        key_states = process_keys(hidden_states)       # [batch, num_heads, seq, q_head_dim]
        value_states = process_values(hidden_states)   # [batch, num_heads, seq, v_head_dim]

        # === Handle dimension mismatch for Flash Attention ===
        if q_head_dim != v_head_dim:
            # Pad value_states to match query dimensions
            value_states = F.pad(value_states, [0, q_head_dim - v_head_dim])  # [batch, heads, seq, q_head_dim]

        # === KV Cache Update ===
        if past_key_value is not None:
            key_states, value_states = past_key_value.update(key_states, value_states, layer_idx)

        # === Transpose for Flash Attention format ===
        # Flash Attention expects [batch, seq_len, num_heads, head_dim]
        query_states = query_states.transpose(1, 2)  # [batch, seq, num_heads, q_head_dim]
        key_states = key_states.transpose(1, 2)      # [batch, seq, num_heads, q_head_dim]
        value_states = value_states.transpose(1, 2)  # [batch, seq, num_heads, q_head_dim]

        # === Flash Attention Computation ===
        if attention_mask is not None:
            # Variable length sequences - unpad, compute, repad
            attn_output = flash_attn_varlen_func(
                query_states, key_states, value_states,
                cu_seqlens_q, cu_seqlens_k,
                max_seqlen_q, max_seqlen_k,
                dropout_p=dropout_rate,
                softmax_scale=softmax_scale,
                causal=True
            )
        else:
            # Fixed length sequences
            attn_output = flash_attn_func(
                query_states, key_states, value_states,
                dropout=dropout_rate,
                softmax_scale=softmax_scale,
                causal=True
            )

        # === Restore original dimensions ===
        if q_head_dim != v_head_dim:
            attn_output = attn_output[..., :v_head_dim]  # [batch, seq, heads, v_head_dim]

        # === Output Processing ===
        attn_output = attn_output.reshape(batch, seq, num_heads * v_head_dim)
        attn_output = o_proj(attn_output)  # [batch, seq, hidden_size]

        return attn_output, None, past_key_value  # No attention weights returned
    ```

    Shape Examples (same as base class):

    **Prefill (seq_len=100):**
    - Input: [1, 100, 7168]
    - Q,K,V: [1, 100, 128, 192], [1, 100, 128, 192], [1, 100, 128, 128]
    - Flash Attention: Computed in tiles, memory O(100) instead of O(10000)
    - Output: [1, 100, 7168]

    **Decode (seq_len=1, with cache):**
    - Input: [1, 1, 7168]
    - Q,K,V: [1, 1, 128, 192], [1, prev_len+1, 128, 192], [1, prev_len+1, 128, 128]
    - Flash Attention: Optimized for decode pattern
    - Output: [1, 1, 7168]

    Key Differences from Standard Attention:
    - **Memory Efficient**: O(N) memory instead of O(N²) for attention matrix
    - **Fused Kernels**: Combines softmax, dropout, and matrix multiply
    - **Variable Length**: Handles padded sequences efficiently with cu_seqlens
    - **No Attention Weights**: Cannot return attention weights due to tiled computation
    - **Layout Requirements**: Needs [batch, seq, heads, dim] format

    Performance Benefits:
    - ~2-4x faster than standard attention for long sequences
    - ~3-8x less memory usage
    - Better GPU utilization through kernel fusion
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # TODO: Should be removed once Flash Attention for RoCm is bumped to 2.1.
        # flash_attn<2.1 generates top-left aligned causal mask, while what is needed here is bottom-right alignement, that was made default for flash_attn>=2.1. This attribute is used to handle this difference. Reference: https://github.com/Dao-AILab/flash-attention/releases/tag/v2.1.0.
        # Beware that with flash_attn<2.1, using q_seqlen != k_seqlen (except for the case q_seqlen == 1) produces a wrong mask (top-left).
        self._flash_attn_uses_top_left_mask = not is_flash_attn_greater_or_equal_2_10()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        # DeepseekV3FlashAttention2 attention does not support output_attentions
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )

            # overwrite attention_mask with padding_mask
            attention_mask = kwargs.pop("padding_mask")

        output_attentions = False

        bsz, q_len, _ = hidden_states.size()

        if self.q_lora_rank is None:
            q = self.q_proj(hidden_states)
        else:
            q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
        q = q.view(bsz, q_len, self.num_heads, self.q_head_dim).transpose(1, 2)
        q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        # Flash attention requires the input to have the shape
        # batch_size x seq_length x head_dim x hidden_dim
        # therefore we just need to keep the original shape
        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
        compressed_kv, k_pe = torch.split(compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        k_pe = k_pe.view(bsz, q_len, 1, self.qk_rope_head_dim).transpose(1, 2)
        kv = (
            self.kv_b_proj(self.kv_a_layernorm(compressed_kv))
            .view(bsz, q_len, self.num_heads, self.qk_nope_head_dim + self.v_head_dim)
            .transpose(1, 2)
        )

        k_nope, value_states = torch.split(kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
        kv_seq_len = value_states.shape[-2]

        kv_seq_len = value_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)

        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        q_pe, k_pe = apply_rotary_pos_emb(q_pe, k_pe, cos, sin, position_ids)

        query_states = k_pe.new_empty(bsz, self.num_heads, q_len, self.q_head_dim)
        query_states[:, :, :, : self.qk_nope_head_dim] = q_nope
        query_states[:, :, :, self.qk_nope_head_dim :] = q_pe

        key_states = k_pe.new_empty(bsz, self.num_heads, q_len, self.q_head_dim)
        key_states[:, :, :, : self.qk_nope_head_dim] = k_nope
        key_states[:, :, :, self.qk_nope_head_dim :] = k_pe

        if self.q_head_dim != self.v_head_dim:
            value_states = F.pad(value_states, [0, self.q_head_dim - self.v_head_dim])

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # TODO: These transpose are quite inefficient but Flash Attention requires the layout [batch_size, sequence_length, num_heads, head_dim]. We would need to refactor the KV cache
        # to be able to avoid many of these transpose/reshape/view.
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        dropout_rate = self.attention_dropout if self.training else 0.0

        # In PEFT, usually we cast the layer norms in float32 for training stability reasons
        # therefore the input hidden states gets silently casted in float32. Hence, we need
        # cast them back in the correct dtype just to be sure everything works as expected.
        # This might slowdown training & inference so it is recommended to not cast the LayerNorms
        # in fp32. (DeepseekV3RMSNorm handles it correctly)

        input_dtype = query_states.dtype
        if input_dtype == torch.float32:
            # Handle the case where the model is quantized
            if hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            elif torch.is_autocast_enabled():
                target_dtype = torch.get_autocast_gpu_dtype()
            else:
                target_dtype = self.q_proj.weight.dtype if self.q_lora_rank is None else self.q_a_proj.weight.dtype

            logger.warning_once(
                f"The input hidden states seems to be silently casted in float32, this might be related to"
                f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
                f" {target_dtype}."
            )

            query_states = query_states.to(target_dtype)
            key_states = key_states.to(target_dtype)
            value_states = value_states.to(target_dtype)

        attn_output = self._flash_attention_forward(
            query_states,
            key_states,
            value_states,
            attention_mask,
            q_len,
            dropout=dropout_rate,
            softmax_scale=self.softmax_scale,
        )
        if self.q_head_dim != self.v_head_dim:
            attn_output = attn_output[:, :, :, : self.v_head_dim]

        attn_output = attn_output.reshape(bsz, q_len, self.num_heads * self.v_head_dim).contiguous()
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

    def _flash_attention_forward(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        query_length,
        dropout=0.0,
        softmax_scale=None,
    ):
        """
        Calls the forward method of Flash Attention - if the input hidden states contain at least one padding token
        first unpad the input, then computes the attention scores and pad the final attention scores.

        Args:
            query_states (`torch.Tensor`):
                Input query states to be passed to Flash Attention API
            key_states (`torch.Tensor`):
                Input key states to be passed to Flash Attention API
            value_states (`torch.Tensor`):
                Input value states to be passed to Flash Attention API
            attention_mask (`torch.Tensor`):
                The padding mask - corresponds to a tensor of size `(batch_size, seq_len)` where 0 stands for the
                position of padding tokens and 1 for the position of non-padding tokens.
            dropout (`int`, *optional*):
                Attention dropout
            softmax_scale (`float`, *optional*):
                The scaling of QK^T before applying softmax. Default to 1 / sqrt(head_dim)
        """
        if not self._flash_attn_uses_top_left_mask:
            causal = self.is_causal
        else:
            # TODO: Remove the `query_length != 1` check once Flash Attention for RoCm is bumped to 2.1. For details, please see the comment in DeepseekV3FlashAttention2 __init__.
            causal = self.is_causal and query_length != 1

        # Contains at least one padding token in the sequence
        if attention_mask is not None:
            batch_size = query_states.shape[0]
            (
                query_states,
                key_states,
                value_states,
                indices_q,
                cu_seq_lens,
                max_seq_lens,
            ) = self._upad_input(query_states, key_states, value_states, attention_mask, query_length)

            cu_seqlens_q, cu_seqlens_k = cu_seq_lens
            max_seqlen_in_batch_q, max_seqlen_in_batch_k = max_seq_lens

            attn_output_unpad = flash_attn_varlen_func(
                query_states,
                key_states,
                value_states,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_in_batch_q,
                max_seqlen_k=max_seqlen_in_batch_k,
                dropout_p=dropout,
                softmax_scale=softmax_scale,
                causal=causal,
            )

            attn_output = pad_input(attn_output_unpad, indices_q, batch_size, query_length)
        else:
            attn_output = flash_attn_func(
                query_states,
                key_states,
                value_states,
                dropout,
                softmax_scale=softmax_scale,
                causal=causal,
            )

        return attn_output

    def _upad_input(self, query_layer, key_layer, value_layer, attention_mask, query_length):
        indices_k, cu_seqlens_k, max_seqlen_in_batch_k = _get_unpad_data(attention_mask)
        batch_size, kv_seq_len, num_key_value_heads, head_dim = key_layer.shape

        key_layer = index_first_axis(
            key_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim),
            indices_k,
        )
        value_layer = index_first_axis(
            value_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim),
            indices_k,
        )
        if query_length == kv_seq_len:
            query_layer = index_first_axis(
                query_layer.reshape(batch_size * kv_seq_len, self.num_heads, head_dim),
                indices_k,
            )
            cu_seqlens_q = cu_seqlens_k
            max_seqlen_in_batch_q = max_seqlen_in_batch_k
            indices_q = indices_k
        elif query_length == 1:
            max_seqlen_in_batch_q = 1
            cu_seqlens_q = torch.arange(
                batch_size + 1, dtype=torch.int32, device=query_layer.device
            )  # There is a memcpy here, that is very bad.
            indices_q = cu_seqlens_q[:-1]
            query_layer = query_layer.squeeze(1)
        else:
            # The -q_len: slice assumes left padding.
            attention_mask = attention_mask[:, -query_length:]
            query_layer, indices_q, cu_seqlens_q, max_seqlen_in_batch_q = unpad_input(query_layer, attention_mask)

        return (
            query_layer,
            key_layer,
            value_layer,
            indices_q,
            (cu_seqlens_q, cu_seqlens_k),
            (max_seqlen_in_batch_q, max_seqlen_in_batch_k),
        )


ATTENTION_CLASSES = {
    "eager": DeepseekV3Attention,
    "flash_attention_2": DeepseekV3FlashAttention2,
}


# Proposed flash_mla API for TTNN
def ttnn_flash_mla_prefill(q, k, v, is_causal=True, scale=None, program_config=None, compute_kernel_config=None):
    """
    MISSING OP - Proposed API for MLA (Multi-head Latent Attention) in prefill mode

    Optimized attention for DeepSeek's compressed KV with different Q/K/V dimensions.
    Handles q_head_dim=192, k_head_dim=192, v_head_dim=128 efficiently.

    Args:
        q: Query tensor [1, num_heads=128, seq_len, q_head_dim=192]
        k: Key tensor [1, num_heads=128, seq_len, k_head_dim=192]
        v: Value tensor [1, num_heads=128, seq_len, v_head_dim=128]
        is_causal: Whether to apply causal mask
        scale: Softmax scale factor (default: 1/sqrt(q_head_dim))

    Returns:
        output: [1, num_heads=128, seq_len, v_head_dim=128]

    This operation would:
    1. Handle different Q/K dimensions vs V dimension efficiently
    2. Apply causal masking if needed
    3. Use tiled computation for memory efficiency
    4. Support variable sequence lengths
    """


def ttnn_flash_mla_decode(
    q, k_cache, v_cache, cur_pos, page_table=None, scale=None, program_config=None, compute_kernel_config=None
):
    """
    MISSING OP - Proposed API for MLA decode with paged KV cache

    Optimized decode attention for compressed KV cache with MQA-style rope keys.

    Args:
        q: Query tensor [1, batch_size, num_heads=128, q_head_dim=192]
        k_cache: Paged key cache [num_blocks, block_size, num_heads=128, k_head_dim=192]
        v_cache: Paged value cache [num_blocks, block_size, num_heads=128, v_head_dim=128]
        cur_pos: Current position tensor [batch_size]
        page_table: Page table for KV cache [batch_size, max_blocks]
        scale: Softmax scale factor

    Returns:
        output: [1, batch_size, num_heads=128, v_head_dim=128]

    Handles:
    1. Different head dimensions for K (192) vs V (128)
    2. Paged KV cache access
    3. Batched decode with different positions per sequence
    """


class DeepseekV3AttentionTTNN(nn.Module):
    """TTNN implementation of Multi-headed attention with LoRA compression

    This implementation uses LoRA compression for query projections and MQA (Multi-Query Attention)
    style compression for key-value projections to reduce memory and computation.

    Pseudo-code:
    ```
    def forward(hidden_states, position_ids, kv_cache=None, is_decode=False):
        # Input: hidden_states
        #   Prefill: [1, 1, SEQ_LEN, 7168]
        #   Decode: [1, 1, BATCH_SIZE, 7168]

        # === Query Projection with LoRA ===
        # Step 1: Down-project to low rank
        q_compressed = ttnn.linear(hidden_states, q_a_weight)  # [1, 1, seq/batch, 1536]
        q_compressed = ttnn.rms_norm(q_compressed, weight=q_a_layernorm_weight)

        # Step 2: Up-project to full query size
        q = ttnn.linear(q_compressed, q_b_weight)  # [1, 1, seq/batch, 24576]

        # === Key-Value Projection with Compression ===
        # Step 1: Project to compressed KV + rope key
        kv_compressed = ttnn.linear(hidden_states, kv_a_weight)  # [1, 1, seq/batch, 576]

        # Step 2: Split compressed KV and rope key
        compressed_kv = kv_compressed[..., :512]  # [1, 1, seq/batch, 512]
        k_pe = kv_compressed[..., 512:]          # [1, 1, seq/batch, 64]

        # Step 3: Up-project compressed KV
        compressed_kv = ttnn.rms_norm(compressed_kv, weight=kv_a_layernorm_weight)
        kv = ttnn.linear(compressed_kv, kv_b_weight)  # [1, 1, seq/batch, 32768]

        # === Reshape and split heads ===
        if is_decode:
            # Decode: use nlp_create_qkv_heads_decode for height sharding
            q, k_nope, v = ttnn.experimental.nlp_create_qkv_heads_decode(
                ttnn.concat([q, kv], dim=-1),
                num_heads=128, num_kv_heads=128
            )
            # q: [1, batch, 128, 192], k_nope: [1, batch, 128, 128], v: [1, batch, 128, 128]

            # Reshape k_pe for MQA broadcast
            k_pe = ttnn.reshape(k_pe, [1, batch, 1, 64])
            k_pe = ttnn.broadcast(k_pe, [1, batch, 128, 64])
        else:
            # Prefill: standard head creation
            q = ttnn.reshape(q, [1, 1, seq, 128, 192])
            q = ttnn.transpose(q, 2, 3)  # [1, 128, seq, 192]

            kv = ttnn.reshape(kv, [1, 1, seq, 128, 256])
            kv = ttnn.transpose(kv, 2, 3)  # [1, 128, seq, 256]
            k_nope = kv[..., :128]
            v = kv[..., 128:]

            k_pe = ttnn.reshape(k_pe, [1, 1, seq, 64])
            k_pe = ttnn.broadcast(k_pe, [1, 128, seq, 64])

        # === Split Q into nope and pe parts ===
        q_nope = q[..., :128]
        q_pe = q[..., 128:]

        # === Apply RoPE ===
        cos, sin = rotary_emb(position_ids, seq_len, is_decode)
        q_pe, k_pe = apply_rotary_pos_emb_ttnn(q_pe, k_pe, cos, sin, is_decode)

        # === Concatenate rotary and non-rotary parts ===
        query_states = ttnn.concat([q_nope, q_pe], dim=-1)
        key_states = ttnn.concat([k_nope, k_pe], dim=-1)
        value_states = v

        # === KV Cache Update ===
        if kv_cache is not None:
            if is_decode:
                # Update cache at current positions
                ttnn.experimental.paged_update_cache(
                    kv_cache.k_cache, key_states,
                    update_idxs=position_ids, page_table=kv_cache.page_table
                )
                ttnn.experimental.paged_update_cache(
                    kv_cache.v_cache, value_states,
                    update_idxs=position_ids, page_table=kv_cache.page_table
                )
            else:
                # Fill cache for prefill
                ttnn.experimental.paged_fill_cache(
                    kv_cache.k_cache, key_states,
                    page_table=kv_cache.page_table, batch_idx=0
                )
                ttnn.experimental.paged_fill_cache(
                    kv_cache.v_cache, value_states,
                    page_table=kv_cache.page_table, batch_idx=0
                )

        # === Attention Computation ===
        if is_decode:
            # Use flash MLA for decode with KV cache
            attn_output = ttnn_flash_mla_decode(
                query_states, kv_cache.k_cache, kv_cache.v_cache,
                cur_pos=position_ids, page_table=kv_cache.page_table,
                scale=softmax_scale
            )
            # Concat heads
            attn_output = ttnn.experimental.nlp_concat_heads_decode(attn_output, num_heads=128)
        else:
            # Prefill
            attn_output = ttnn_flash_mla_prefill(
                query_states, key_states, value_states,
                is_causal=True, scale=softmax_scale
            )
            # Concat heads
            attn_output = ttnn.experimental.nlp_concat_heads(attn_output)
            attn_output = ttnn.reshape(attn_output, [1, 1, seq, 128 * 128])

        # === Output Projection ===
        attn_output = ttnn.linear(attn_output, o_weight)  # [1, 1, seq/batch, 7168]

        return attn_output
    ```

    Shape Examples:

    **Prefill (SEQ_LEN=100):**
    - Input: [1, 1, 100, 7168]
    - Q LoRA: [1, 1, 100, 7168] -> [1, 1, 100, 1536] -> [1, 1, 100, 24576] -> [1, 128, 100, 192]
    - KV Compression: [1, 1, 100, 7168] -> [1, 1, 100, 576] -> [1, 128, 100, 256] + [1, 1, 100, 64]
    - Attention: [1, 128, 100, 192] @ [1, 128, 100, 192]^T -> [1, 128, 100, 100]
    - Output: [1, 1, 100, 7168]

    **Decode (BATCH_SIZE=32):**
    - Input: [1, 1, 32, 7168]
    - New Q: [1, 32, 128, 192]
    - New K,V: [1, 32, 128, 192], [1, 32, 128, 128]
    - Attention with cache: [1, 32, 128, 192] @ cached_kv
    - Output: [1, 1, 32, 7168]
    """

    def __init__(self, config, layer_idx=None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.hidden_size = config.hidden_size  # 7168
        self.num_heads = config.num_attention_heads  # 128
        self.q_lora_rank = config.q_lora_rank  # 1536
        self.qk_rope_head_dim = config.qk_rope_head_dim  # 64
        self.kv_lora_rank = config.kv_lora_rank  # 512
        self.v_head_dim = config.v_head_dim  # 128
        self.qk_nope_head_dim = config.qk_nope_head_dim  # 128
        self.q_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim  # 192

        # Weights will be loaded as ttnn tensors
        self.q_a_weight = None
        self.q_a_layernorm_weight = None
        self.q_b_weight = None
        self.kv_a_weight = None
        self.kv_a_layernorm_weight = None
        self.kv_b_weight = None
        self.o_weight = None

        # Rotary embeddings
        self.rotary_emb = None  # Will be DeepseekV3YarnRotaryEmbeddingTTNN

        # Softmax scale with YARN mscale
        self.softmax_scale = self.q_head_dim ** (-0.5)
        if config.rope_scaling is not None:
            mscale_all_dim = config.rope_scaling.get("mscale_all_dim", 0)
            scaling_factor = config.rope_scaling["factor"]
            if mscale_all_dim:
                from .rope import yarn_get_mscale

                mscale = yarn_get_mscale(scaling_factor, mscale_all_dim)
                self.softmax_scale = self.softmax_scale * mscale * mscale

    def forward(
        self,
        hidden_states,
        position_ids=None,
        kv_cache=None,
        is_decode_mode=False,
        memory_config=None,
        compute_kernel_config=None,
    ):
        """
        Args:
            hidden_states: TTNN tensor
                Prefill: [1, 1, SEQ_LEN, 7168]
                Decode: [1, 1, BATCH_SIZE, 7168]
            position_ids: TTNN tensor with positions
                Prefill: [1, 1, SEQ_LEN] or None
                Decode: [1, 1, BATCH_SIZE, 1]
            kv_cache: KV cache object with paged cache tensors
            is_decode_mode: Whether in decode mode
            memory_config: Memory configuration
            compute_kernel_config: Compute configuration

        Returns:
            TTNN tensor of same shape as input
        """
        if is_decode_mode:
            batch_size = hidden_states.shape[2]
            seq_len = 1
        else:
            batch_size = 1
            seq_len = hidden_states.shape[2]

        # Query projection with LoRA
        q_compressed = ttnn.linear(
            hidden_states,
            self.q_a_weight,
            bias=None,
            memory_config=memory_config,
            compute_kernel_config=compute_kernel_config,
        )
        q_compressed = ttnn.rms_norm(
            q_compressed, epsilon=1e-6, weight=self.q_a_layernorm_weight, memory_config=memory_config
        )
        q = ttnn.linear(
            q_compressed,
            self.q_b_weight,
            bias=None,
            memory_config=memory_config,
            compute_kernel_config=compute_kernel_config,
        )

        # KV projection with compression
        kv_compressed = ttnn.linear(
            hidden_states,
            self.kv_a_weight,
            bias=None,
            memory_config=memory_config,
            compute_kernel_config=compute_kernel_config,
        )

        # Split compressed KV and rope key
        compressed_kv = kv_compressed[..., : self.kv_lora_rank]
        k_pe = kv_compressed[..., self.kv_lora_rank :]

        # Normalize and up-project KV
        compressed_kv = ttnn.rms_norm(
            compressed_kv, epsilon=1e-6, weight=self.kv_a_layernorm_weight, memory_config=memory_config
        )
        kv = ttnn.linear(
            compressed_kv,
            self.kv_b_weight,
            bias=None,
            memory_config=memory_config,
            compute_kernel_config=compute_kernel_config,
        )

        # Create heads
        if is_decode_mode:
            # Fuse Q and KV for decode head creation
            qkv_fused = ttnn.concat([q, kv], dim=-1)
            q_heads, k_nope, v = ttnn.experimental.nlp_create_qkv_heads_decode(
                qkv_fused, num_heads=self.num_heads, num_kv_heads=self.num_heads, memory_config=memory_config
            )
            # Shapes: q [1, batch, 128, 192], k_nope [1, batch, 128, 128], v [1, batch, 128, 128]

            # Handle k_pe for MQA broadcast
            k_pe = ttnn.reshape(k_pe, [1, batch_size, 1, self.qk_rope_head_dim])
            k_pe = ttnn.transpose(k_pe, 1, 2)  # [1, 1, batch, 64]
        else:
            # Prefill mode
            q_heads = ttnn.reshape(q, [1, 1, seq_len, self.num_heads, self.q_head_dim])
            q_heads = ttnn.transpose(q_heads, 2, 3)  # [1, 128, seq, 192]

            kv_heads = ttnn.reshape(kv, [1, 1, seq_len, self.num_heads, self.qk_nope_head_dim + self.v_head_dim])
            kv_heads = ttnn.transpose(kv_heads, 2, 3)  # [1, 128, seq, 256]

            k_nope = kv_heads[..., : self.qk_nope_head_dim]
            v = kv_heads[..., self.qk_nope_head_dim :]

            k_pe = ttnn.reshape(k_pe, [1, 1, seq_len, self.qk_rope_head_dim])
            k_pe = ttnn.unsqueeze(k_pe, 1)  # [1, 1, seq, 64]
            k_pe = ttnn.transpose(k_pe, 2, 3)  # [1, 1, seq, 64]

        # Split Q into nope and pe parts
        q_nope = q_heads[..., : self.qk_nope_head_dim]
        q_pe = q_heads[..., self.qk_nope_head_dim :]

        # Apply RoPE
        if is_decode_mode:
            cos, sin = self.rotary_emb(position_ids=position_ids, memory_config=memory_config)
        else:
            kv_seq_len = seq_len
            if kv_cache is not None:
                kv_seq_len = kv_cache.get_seq_length(self.layer_idx) + seq_len
            cos, sin = self.rotary_emb(seq_len=kv_seq_len, memory_config=memory_config)

        q_pe, k_pe = apply_rotary_pos_emb_ttnn(q_pe, k_pe, cos, sin, is_decode_mode)

        # Expand k_pe to all heads after RoPE
        if not is_decode_mode:
            k_pe = ttnn.broadcast(k_pe, [1, self.num_heads, seq_len, self.qk_rope_head_dim])
        else:
            k_pe = ttnn.broadcast(k_pe, [1, batch_size, self.num_heads, self.qk_rope_head_dim])

        # Concatenate rotary and non-rotary parts
        query_states = ttnn.concat([q_nope, q_pe], dim=-1)
        key_states = ttnn.concat([k_nope, k_pe], dim=-1)
        value_states = v

        # KV cache operations
        if kv_cache is not None:
            if is_decode_mode:
                # Update cache
                ttnn.experimental.paged_update_cache(
                    kv_cache.k_cache, key_states, update_idxs=position_ids, page_table=kv_cache.page_table
                )
                ttnn.experimental.paged_update_cache(
                    kv_cache.v_cache, value_states, update_idxs=position_ids, page_table=kv_cache.page_table
                )
            else:
                # Fill cache
                ttnn.experimental.paged_fill_cache(
                    kv_cache.k_cache, key_states, page_table=kv_cache.page_table, batch_idx=0
                )
                ttnn.experimental.paged_fill_cache(
                    kv_cache.v_cache, value_states, page_table=kv_cache.page_table, batch_idx=0
                )

        # Attention computation
        if is_decode_mode:
            # Decode with cache
            attn_output = ttnn_flash_mla_decode(
                query_states,
                kv_cache.k_cache,
                kv_cache.v_cache,
                cur_pos=position_ids,
                page_table=kv_cache.page_table,
                scale=self.softmax_scale,
                program_config=compute_kernel_config,
                compute_kernel_config=compute_kernel_config,
            )
            # Concat heads
            attn_output = ttnn.experimental.nlp_concat_heads_decode(attn_output, num_heads=self.num_heads)
        else:
            # Prefill
            attn_output = ttnn_flash_mla_prefill(
                query_states,
                key_states,
                value_states,
                is_causal=True,
                scale=self.softmax_scale,
                program_config=compute_kernel_config,
                compute_kernel_config=compute_kernel_config,
            )
            # Concat heads
            attn_output = ttnn.experimental.nlp_concat_heads(attn_output)
            attn_output = ttnn.reshape(attn_output, [1, 1, seq_len, self.num_heads * self.v_head_dim])

        # Output projection
        attn_output = ttnn.linear(
            attn_output,
            self.o_weight,
            bias=None,
            memory_config=memory_config,
            compute_kernel_config=compute_kernel_config,
        )

        return attn_output


# KV Cache helper class for TTNN
class TTNNPagedKVCache:
    """Helper class to manage paged KV cache for TTNN"""

    def __init__(
        self, batch_size, max_seq_len, num_layers, num_heads, head_dim_k, head_dim_v, page_size=128, device=None
    ):
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim_k = head_dim_k
        self.head_dim_v = head_dim_v
        self.page_size = page_size

        # Allocate cache tensors
        num_blocks = (max_seq_len + page_size - 1) // page_size * batch_size

        # K cache: [num_blocks, page_size, num_heads, head_dim_k]
        self.k_cache = None  # Would be allocated on device

        # V cache: [num_blocks, page_size, num_heads, head_dim_v]
        self.v_cache = None  # Would be allocated on device

        # Page table: [batch_size, max_blocks_per_seq]
        self.page_table = None  # Would be allocated on device

        # Track sequence lengths
        self.seq_lengths = [0] * batch_size

    def get_seq_length(self, layer_idx, batch_idx=0):
        return self.seq_lengths[batch_idx]

    def update_seq_length(self, layer_idx, batch_idx, new_len):
        self.seq_lengths[batch_idx] = new_len
