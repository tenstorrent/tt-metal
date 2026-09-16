# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Native TTNN port of `patch_embed` of Qwen/Qwen2-VL-7B-Instruct.

Reference: `visual.patch_embed`, a `PatchEmbed` module that projects flattened
image patches with a `Conv3d(in_channels, embed_dim, kernel_size=(T, P, P),
stride=(T, P, P), bias=False)`:

    hidden_states = hidden_states.view(-1, in_channels, T, P, P)
    hidden_states = proj(hidden_states).view(-1, embed_dim)

Because `stride == kernel_size` and the reshaped input exactly fills one
kernel window, the conv3d degenerates to a single dot product per output
channel per patch -- i.e. a plain matmul of the flattened patch vector
`(N, in_channels*T*P*P)` against the flattened conv weight
`(in_channels*T*P*P, embed_dim)`. No sliding window, no bias (bias=False).
"""

from __future__ import annotations

import ttnn

_DRAM = ttnn.DRAM_MEMORY_CONFIG


def build(device, torch_module):
    conv = torch_module.proj  # nn.Conv3d(in_channels, embed_dim, kernel_size=(T, P, P), stride=(T, P, P), bias=False)
    embed_dim = int(conv.out_channels)
    weight = conv.weight.detach().reshape(embed_dim, -1).T.contiguous()  # (flat, embed_dim)
    # float32: this projection feeds 32 stacked vision blocks (also float32); a
    # bf16 patch_embed at the front compounds enough rounding error to drop the
    # merged image_embeds PCC below the e2e bar.
    weight_ttnn = ttnn.from_torch(weight, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)

    bias_ttnn = None
    if conv.bias is not None:
        bias = conv.bias.detach().reshape(1, embed_dim)
        bias_ttnn = ttnn.from_torch(bias, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)

    def forward(hidden_states, *args, **kwargs):
        # hidden_states: (N, in_channels*T*P*P) flattened patches.
        if hidden_states.get_dtype() != ttnn.float32:
            hidden_states = ttnn.typecast(hidden_states, ttnn.float32)
        out = ttnn.matmul(hidden_states, weight_ttnn, memory_config=_DRAM)
        if bias_ttnn is not None:
            out = ttnn.add(out, bias_ttnn, memory_config=_DRAM)
        return out

    return forward


def patch_embed(*args, **kwargs):
    raise RuntimeError(
        "patch_embed requires build(device, torch_module) to bind trained weights; the bare callable has no parameters."
    )
