# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Native TTNN port of `albert_transformer` (hexgrad/Kokoro-82M
`bert.encoder`, a HF `AlbertTransformer`).

Reference torch forward (ALBERT weight-sharing: one layer group reused for every
hidden layer):

    hidden_states = embedding_hidden_mapping_in(hidden_states)   # embed_size -> hidden
    for i in range(num_hidden_layers):
        group_idx = int(i / (num_hidden_layers / num_hidden_groups))
        hidden_states = albert_layer_groups[group_idx](hidden_states, attention_mask)
    return BaseModelOutput(last_hidden_state=hidden_states)

Native ttnn: a single `ttnn.linear` up-projection followed by `num_hidden_layers`
applications of the reused layer group (via the graduated native
`albert_layer_group` port). Everything runs natively in float32.
"""

from __future__ import annotations

import ttnn
from models.demos.kokoro_82m._stubs.albert_layer_group import build as _build_layer_group

_DRAM = ttnn.DRAM_MEMORY_CONFIG


HF_MODEL_ID = "hexgrad/Kokoro-82M"


def build(device, torch_module):
    """Bind the up-projection + shared layer groups; return a native forward."""
    m = torch_module
    cfg = m.config
    num_hidden_layers = int(cfg.num_hidden_layers)
    num_hidden_groups = int(cfg.num_hidden_groups)

    compute_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )

    map_w = ttnn.from_torch(
        m.embedding_hidden_mapping_in.weight.detach().t().contiguous().float(),
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=_DRAM,
    )
    map_b = ttnn.from_torch(
        m.embedding_hidden_mapping_in.bias.detach().reshape(1, -1).contiguous().float(),
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=_DRAM,
    )

    group_fwds = [_build_layer_group(device, g) for g in m.albert_layer_groups]

    def forward(hidden_states, attention_mask=None, *args, **kwargs):
        h = hidden_states
        if not isinstance(h, ttnn.Tensor):
            h = ttnn.from_torch(
                h.contiguous().float(), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device, memory_config=_DRAM
            )
        if h.get_dtype() != ttnn.float32:
            h = ttnn.typecast(h, ttnn.float32)

        h = ttnn.linear(h, map_w, bias=map_b, compute_kernel_config=compute_config, memory_config=_DRAM)

        for i in range(num_hidden_layers):
            group_idx = int(i / (num_hidden_layers / num_hidden_groups))
            h = group_fwds[group_idx](h, attention_mask)
        return h

    return forward


def albert_transformer(*args, **kwargs):
    raise RuntimeError(
        "albert_transformer requires build(device, torch_module) to bind trained "
        "weights; the bare callable has no parameters."
    )
