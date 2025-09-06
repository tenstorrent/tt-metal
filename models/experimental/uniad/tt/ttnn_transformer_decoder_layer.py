# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from torch import Tensor
from typing import Union, Callable, Optional
import torch.nn.functional as F

import ttnn


from models.experimental.uniad.tt.ttnn_multi_head_attention import TtMultiheadAttention


class TtTransformerDecoderLayer:
    __constants__ = ["norm_first"]

    def __init__(
        self,
        parameters,
        device,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
        layer_norm_eps: float = 1e-5,
        batch_first: bool = False,
        norm_first: bool = False,
        bias: bool = True,
        dtype=None,
    ) -> None:
        self.parameters = parameters
        self.device = device
        factory_kwargs = {"device": device, "dtype": dtype}
        self.self_attn = TtMultiheadAttention(
            device, parameters.self_attn, embed_dim=d_model, num_heads=nhead, batch_first=batch_first
        )
        self.multihead_attn = TtMultiheadAttention(
            device, parameters.multihead_attn, embed_dim=d_model, num_heads=nhead, batch_first=batch_first
        )

        # Implementation of Feedforward model
        self.linear1 = ttnn.linear
        self.linear2 = ttnn.linear

        self.norm_first = norm_first
        self.norm1 = ttnn.layer_norm
        self.norm2 = ttnn.layer_norm
        self.norm3 = ttnn.layer_norm

        self.activation = ttnn.relu

    def __call__(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        tgt_is_causal: bool = False,
        memory_is_causal: bool = False,
    ) -> Tensor:
        x = tgt

        if self.norm_first:
            x = x + self.self_attn(
                self.norm1(x), self.norm1(x), self.norm1(x), attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask
            )
            x = x + self.multihead_attn(
                self.norm2(x), memory, memory, attn_mask=memory_mask, key_padding_mask=memory_key_padding_mask
            )
            x = x + self._ff_block(self.norm3(x))
        else:
            x = self.norm1(
                (x + self.self_attn(x, x, x)[0]),
                weight=self.parameters.norm1.weight,
                bias=self.parameters.norm1.bias,
            )

            x = self.norm2(
                (x + self.multihead_attn(x, memory, memory)[0]),
                weight=self.parameters.norm2.weight,
                bias=self.parameters.norm2.bias,
            )

            x = self.norm3(x + self._ff_block(x), weight=self.parameters.norm3.weight, bias=self.parameters.norm3.bias)

        return x

    # self-attention block
    def _sa_block(
        self,
        x: Tensor,
        attn_mask: Optional[Tensor],
        key_padding_mask: Optional[Tensor],
        is_causal: bool = False,
    ) -> Tensor:
        x = self.self_attn(
            x,
            x,
            x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            is_causal=is_causal,
            need_weights=False,
        )[0]
        return x

    # multihead attention block
    def _mha_block(
        self,
        x: Tensor,
        mem: Tensor,
        attn_mask: Optional[Tensor],
        key_padding_mask: Optional[Tensor],
        is_causal: bool = False,
    ) -> Tensor:
        x = self.multihead_attn(
            x,
            mem,
            mem,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            is_causal=is_causal,
            need_weights=False,
        )[0]
        return x

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        y = self.linear1(x, self.parameters.linear1.weight, bias=self.parameters.linear1.bias, dtype=ttnn.bfloat16)
        x = self.linear2(
            self.activation(y), self.parameters.linear2.weight, bias=self.parameters.linear2.bias, dtype=ttnn.bfloat16
        )
        return x
