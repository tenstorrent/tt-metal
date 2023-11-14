# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import torch
import torch.nn as nn
import tt_lib
from typing import Optional
from models.experimental.mistral.tt.mistral_attention import TtAttention
from models.experimental.mistral.tt.mistral_feed_forward import TtFeedForward
from models.experimental.mistral.tt.mistral_rms_norm import TtRMSNorm
from models.experimental.mistral.tt.mistral_configuration import TtModelArgs


class TtTransformerBlock(nn.Module):
    def __init__(
        self,
        args: TtModelArgs,
        state_dict=None,
        device=None,
        base_address=None,
    ):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.device = device
        self.attention = TtAttention(args, f"{base_address}attention.", device, state_dict)
        self.feed_forward = TtFeedForward(args, f"{base_address}feed_forward.", device, state_dict)
        self.attention_norm = TtRMSNorm(
            args.dim,
            base_address=f"{base_address}attention_norm.",
            state_dict=state_dict,
            device=device,
            eps=args.norm_eps,
        )
        self.ffn_norm = TtRMSNorm(
            args.dim, base_address=f"{base_address}ffn_norm.", state_dict=state_dict, device=device, eps=args.norm_eps
        )
        self.args = args

    def forward(
        self,
        x: tt_lib.tensor.Tensor,
        freqs_cis: torch.Tensor,
        positions: tt_lib.tensor.Tensor,
        mask: Optional[torch.Tensor],
    ) -> tt_lib.tensor.Tensor:
        r = self.attention.forward(self.attention_norm(x), freqs_cis, positions, mask)
        h = tt_lib.tensor.add(x, r)
        x.deallocate()
        r = self.feed_forward.forward(self.ffn_norm(h))
        out = tt_lib.tensor.add(h, r)
        h.deallocate()
        return out
