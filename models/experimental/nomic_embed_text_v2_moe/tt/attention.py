# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Bidirectional MHA, the TTNN form of reference.NomicBertAttention.

    x (B, 1, S, H)
      Wqkv                    -> (B, 1, S, 3H)   three-major: [q | k | v]
      nlp_create_qkv_heads    -> three (B, A, S, D)
      rotary_embedding_hf     -> q and k rotated, v untouched
      SDPA, is_causal=False   -> (B, A, S, D)
      nlp_concat_heads        -> (B, 1, S, H)
      out_proj                -> (B, 1, S, H)

The rotary tables and the additive mask are built once per forward pass by the caller, not here:
both depend only on S, and building them per block would repeat the same host work 12 times.
"""

from __future__ import annotations

from typing import Optional

import ttnn

from models.common.lightweightmodule import LightweightModule
from models.experimental.nomic_embed_text_v2_moe.tt.common import to_device, transpose_linear_weight


class TtNomicBertAttention(LightweightModule):
    """Fused three-major QKV projection, full-head rotary, then bidirectional SDPA.

    Two defaults differ from torch and both fail silently:

      - ttnn's SDPA defaults is_causal to True where torch defaults it to False. Left alone it
        applies a decoder mask to an encoder and still returns finite output, PCC 0.44.
      - the mask must materialise the query axis as (B, 1, S, S). Torch broadcasts (B, 1, 1, S)
        over queries; ttnn rejects that shape outright, which at least is loud.

    No explicit scale is passed: SDPA's own default is already 1/sqrt(head_dim), which is what
    the reference relies on too.
    """

    def __init__(self, device, config, tt_config, state_dict, state_dict_prefix):
        super().__init__()
        self.tt_config = tt_config
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim

        def weight(name):
            return to_device(
                transpose_linear_weight(state_dict[f"{state_dict_prefix}{name}.weight"]),
                device,
                dtype=tt_config.weight_dtype,
            )

        def bias(name):
            return to_device(state_dict[f"{state_dict_prefix}{name}.bias"], device, dtype=tt_config.weight_dtype)

        self.qkv_weight, self.qkv_bias = weight("Wqkv"), bias("Wqkv")
        self.out_weight, self.out_bias = weight("out_proj"), bias("out_proj")

    def _rotate(self, x: ttnn.Tensor, cos: ttnn.Tensor, sin: ttnn.Tensor) -> ttnn.Tensor:
        """Apply rotary position embedding to one of q or k.

        rotary_embedding_hf's prefill mode wants a leading batch of 1, so the batch is folded
        into the head axis; cos/sin are (1, 1, S, D) and broadcast over it, applying the same
        table to every row.

        Args:
            x: (B, A, S, D) queries or keys.
            cos: (1, 1, S, D) cosine table from tt.common.rotary_tables.
            sin: (1, 1, S, D) sine table.

        Returns:
            ttnn.Tensor: (B, A, S, D), rotated.
        """
        batch, heads, seqlen, head_dim = x.shape
        folded = ttnn.reshape(x, (1, batch * heads, seqlen, head_dim))
        rotated = ttnn.experimental.rotary_embedding_hf(folded, cos, sin, is_decode_mode=False)
        return ttnn.reshape(rotated, (batch, heads, seqlen, head_dim))

    def forward(
        self,
        x: ttnn.Tensor,
        rot_mats: tuple[ttnn.Tensor, ttnn.Tensor],
        attn_mask: Optional[ttnn.Tensor] = None,
    ) -> ttnn.Tensor:
        """Project, rotate, attend and project back.

        Args:
            x: (B, 1, S, H) block input.
            rot_mats: (cos, sin), each (1, 1, S, D), from tt.common.rotary_tables.
            attn_mask: (B, 1, S, S) additive mask from tt.common.additive_attention_mask, or
                None for no masking.

        Returns:
            ttnn.Tensor: (B, 1, S, H).
        """
        qkv = ttnn.linear(
            x,
            self.qkv_weight,
            bias=self.qkv_bias,
            compute_kernel_config=self.tt_config.compute_kernel_config,
        )

        # Three-major, heads contiguous inside each of q, k and v. transpose_k_heads stays False
        # because SDPA wants K as (B, A, S, D), not pre-transposed.
        query, key, value = ttnn.experimental.nlp_create_qkv_heads(
            qkv, num_heads=self.num_heads, num_kv_heads=self.num_heads, transpose_k_heads=False
        )
        ttnn.deallocate(qkv)

        cos, sin = rot_mats
        rotated_query, rotated_key = self._rotate(query, cos, sin), self._rotate(key, cos, sin)
        # Freed here rather than inside _rotate: the reshape there aliases this buffer, so the
        # pre-rotation tensor is the one thing that owns it.
        ttnn.deallocate(query)
        ttnn.deallocate(key)

        context = ttnn.transformer.scaled_dot_product_attention(
            rotated_query,
            rotated_key,
            value,
            attn_mask=attn_mask,
            is_causal=False,
            compute_kernel_config=self.tt_config.compute_kernel_config,
        )
        for tensor in (rotated_query, rotated_key, value):
            ttnn.deallocate(tensor)

        concatenated = ttnn.experimental.nlp_concat_heads(context)
        ttnn.deallocate(context)

        out = ttnn.linear(
            concatenated,
            self.out_weight,
            bias=self.out_bias,
            compute_kernel_config=self.tt_config.compute_kernel_config,
        )
        ttnn.deallocate(concatenated)
        return out
