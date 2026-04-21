# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""TT-NN implementation of the xvla DomainAwareLinear layer.

DomainAwareLinear holds per-domain weights in an `nn.Embedding` (so weight
lookup is just a row gather). At inference the domain_id is fixed for the
whole batch (and in our benchmark always a single scalar), so we can
pre-select the domain's weight/bias, upload to device once, and reuse.
"""

from __future__ import annotations

import torch
from torch import nn


class TTNNDomainLinear(nn.Module):
    """Drop-in replacement for xvla `DomainAwareLinear` that runs one
    per-domain linear on the Blackhole device.

    The original signature is `(x: [B,I] or [B,T,I], domain_id: [B]) -> [...]`.
    We restrict to batch=1 (the benchmark setting) so we can pick exactly
    one domain's weights. The upload happens once per distinct domain_id.
    """

    def __init__(self, torch_module: nn.Module, device) -> None:
        super().__init__()
        import ttnn

        self._ttnn = ttnn
        self.device = device
        self.input_size = int(torch_module.input_size)
        self.output_size = int(torch_module.output_size)

        # Keep raw tables on CPU; materialize per-domain ttnn tensors on demand.
        # torch_module.fc.weight shape: [num_domains, output*input]
        # torch_module.bias.weight shape: [num_domains, output]
        self._fc_weight_cpu = torch_module.fc.weight.detach().to(torch.bfloat16).contiguous()
        self._bias_cpu = torch_module.bias.weight.detach().to(torch.bfloat16).contiguous()
        self._cache: dict[int, tuple] = {}

    def _get_domain_weights(self, domain_id: int):
        if domain_id in self._cache:
            return self._cache[domain_id]
        ttnn = self._ttnn
        w = self._fc_weight_cpu[domain_id].view(self.input_size, self.output_size).contiguous()
        b = self._bias_cpu[domain_id].contiguous()
        w_tt = ttnn.from_torch(w, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device)
        b_tt = ttnn.from_torch(
            b.unsqueeze(0), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device
        )
        self._cache[domain_id] = (w_tt, b_tt)
        return w_tt, b_tt

    def forward(self, x: torch.Tensor, domain_id: torch.LongTensor) -> torch.Tensor:
        ttnn = self._ttnn
        assert domain_id.shape[0] == 1, "TTNNDomainLinear only supports batch=1"
        d = int(domain_id[0].item())
        w_tt, b_tt = self._get_domain_weights(d)

        squeeze_seq = x.dim() == 2
        if squeeze_seq:
            x = x.unsqueeze(1)

        x_bf16 = x.to(torch.bfloat16).contiguous()
        x_tt = ttnn.from_torch(x_bf16, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device)
        out_tt = ttnn.linear(x_tt, w_tt, bias=b_tt, dtype=ttnn.bfloat16)
        out = ttnn.to_torch(out_tt).to(x.dtype)
        ttnn.deallocate(x_tt)
        ttnn.deallocate(out_tt)

        if squeeze_seq:
            out = out.squeeze(1)
        return out
