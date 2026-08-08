# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Native TTNN port of `albert_layer_group` (hexgrad/Kokoro-82M
`bert.encoder.albert_layer_groups.0`, a HF `AlbertLayerGroup`).

Reference torch forward simply chains its inner `albert_layers`:

    for albert_layer in self.albert_layers:
        hidden_states = albert_layer(hidden_states, attention_mask, **kwargs)
    return hidden_states

Each inner layer is a full `AlbertLayer`, so this port reuses the graduated
native `albert_layer` port for every sub-layer and threads the hidden state
through them. Everything runs natively in ttnn (float32).
"""

from __future__ import annotations

import ttnn
from models.demos.kokoro_82m._stubs.albert_layer import build as _build_albert_layer

_DRAM = ttnn.DRAM_MEMORY_CONFIG


HF_MODEL_ID = "hexgrad/Kokoro-82M"


def build(device, torch_module):
    """Build a native forward for each inner AlbertLayer and chain them."""
    m = torch_module
    layer_fwds = [_build_albert_layer(device, layer) for layer in m.albert_layers]

    def forward(hidden_states, attention_mask=None, *args, **kwargs):
        h = hidden_states
        if not isinstance(h, ttnn.Tensor):
            h = ttnn.from_torch(
                h.contiguous().float(),
                dtype=ttnn.float32,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=_DRAM,
            )
        for fwd in layer_fwds:
            h = fwd(h, attention_mask)
        return h

    return forward


def albert_layer_group(*args, **kwargs):
    raise RuntimeError(
        "albert_layer_group requires build(device, torch_module) to bind trained "
        "weights; the bare callable has no parameters."
    )
