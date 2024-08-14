# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import torch
import torch.nn as nn

import ttnn
from typing import Optional
from models.experimental.mistral.tt.mistral_attention import TtAttention
from models.experimental.mistral.tt.mistral_feed_forward import TtFeedForward
from models.experimental.mistral.tt.mistral_rms_norm import TtRMSNorm
from models.experimental.mistral.tt.mistral_configuration import TtModelArgs


class TtTransformerBlock(nn.Module):
    def __init__(
        self,
        args: TtModelArgs,
        device=None,
        base_address=None,
        tt_cache_path=None,
        output_mem_config=None,
    ):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.device = device
        self.output_mem_config = output_mem_config
        self.attention = TtAttention(
            args,
            f"{base_address}attention.",
            device,
            tt_cache_path=tt_cache_path,
            output_mem_config=self.output_mem_config,
        )
        self.feed_forward = TtFeedForward(
            args,
            f"{base_address}feed_forward.",
            device,
            tt_cache_path=tt_cache_path,
            output_mem_config=self.output_mem_config,
        )
        self.attention_norm = TtRMSNorm(
            args.dim,
            base_address=f"{base_address}attention_norm.",
            eps=args.norm_eps,
            tt_cache_path=tt_cache_path,
            device=self.device,
            output_mem_config=self.output_mem_config,
        )
        self.ffn_norm = TtRMSNorm(
            args.dim,
            base_address=f"{base_address}ffn_norm.",
            eps=args.norm_eps,
            tt_cache_path=tt_cache_path,
            device=self.device,
            output_mem_config=self.output_mem_config,
        )
        self.args = args

    def forward(
        self,
        x: ttnn.Tensor,
        bcast_freq_xq: ttnn.complex_tensor,
        bcast_freq_xk: ttnn.complex_tensor,
        positions: ttnn.Tensor,
        mask: Optional[torch.Tensor],
        seqlen: int,
    ) -> ttnn.Tensor:
        r = self.attention.forward(self.attention_norm(x), bcast_freq_xq, bcast_freq_xk, positions, mask, seqlen)
        h = ttnn.add(x, r)
        x.deallocate()
        r = self.feed_forward.forward(self.ffn_norm(h))
        out = ttnn.add(h, r)
        h.deallocate()
        return out
