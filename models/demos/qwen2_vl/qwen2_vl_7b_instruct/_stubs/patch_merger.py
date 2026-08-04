# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Native TTNN port of `patch_merger` of Qwen/Qwen2-VL-7B-Instruct.

Reference: `visual.merger`, a `PatchMerger`:

    x = self.mlp(self.ln_q(x).view(-1, self.hidden_size))
    # ln_q: LayerNorm(context_dim)
    # mlp: Linear(hidden_size, hidden_size) -> GELU -> Linear(hidden_size, dim)
    # hidden_size == context_dim * spatial_merge_size**2
"""

from __future__ import annotations

import ttnn

_DRAM = ttnn.DRAM_MEMORY_CONFIG


def build(device, torch_module):
    ln_q = torch_module.ln_q
    fc1, gelu, fc2 = torch_module.mlp[0], torch_module.mlp[1], torch_module.mlp[2]
    hidden_size = int(torch_module.hidden_size)

    # float32: the merger produces the final image_embeds injected into the LM;
    # keep it in float32 to match the float32 vision blocks feeding it.
    ln_q_weight = ttnn.from_torch(ln_q.weight.detach(), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    ln_q_bias = ttnn.from_torch(ln_q.bias.detach(), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    ln_q_eps = float(ln_q.eps)

    fc1_weight = ttnn.from_torch(
        fc1.weight.detach().T.contiguous(), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device
    )
    fc1_bias = ttnn.from_torch(
        fc1.bias.detach().reshape(1, -1), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device
    )
    fc2_weight = ttnn.from_torch(
        fc2.weight.detach().T.contiguous(), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device
    )
    fc2_bias = ttnn.from_torch(
        fc2.bias.detach().reshape(1, -1), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device
    )

    def forward(x, *args, **kwargs):
        if x.get_dtype() != ttnn.float32:
            x = ttnn.typecast(x, ttnn.float32)
        x = ttnn.layer_norm(x, epsilon=ln_q_eps, weight=ln_q_weight, bias=ln_q_bias)
        n = int(x.shape[0]) * int(x.shape[-1]) // hidden_size
        x = ttnn.reshape(x, (n, hidden_size))
        x = ttnn.matmul(x, fc1_weight, memory_config=_DRAM)
        x = ttnn.add(x, fc1_bias, memory_config=_DRAM)
        x = ttnn.gelu(x)
        x = ttnn.matmul(x, fc2_weight, memory_config=_DRAM)
        x = ttnn.add(x, fc2_bias, memory_config=_DRAM)
        return x

    return forward


def patch_merger(*args, **kwargs):
    raise RuntimeError(
        "patch_merger requires build(device, torch_module) to bind trained weights; the bare callable has no parameters."
    )
