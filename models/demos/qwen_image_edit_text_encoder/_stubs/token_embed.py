# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Native TTNN port of the Qwen2.5-VL token embedding (`model.language_model.embed_tokens`,
nn.Embedding(152064, 3584)). A lookup, not a matmul; on a mesh the table is split by hidden dim over the TP axis.

Input ids arrive as a uint32 row-major device tensor [B, S]; output is [B, S, 3584] bf16 tile.
"""

from __future__ import annotations

import ttnn


class TtTokenEmbed:
    """On a mesh the table is sharded over the TP (column) axis by hidden dim -- 3584 / 4 columns per
    chip -- and replicated over rows; the lookup runs per chip and an all_gather on the column axis
    rebuilds the full row (a lookup is exact, so this changes no value)."""

    def __init__(self, device, torch_module):
        self.device = device
        from models.demos.qwen_image_edit_text_encoder._stubs.attention import mesh_shape, shard_mapper

        _, self.tp = mesh_shape(device) if isinstance(device, ttnn.MeshDevice) else (1, 1)
        kw = {}
        if isinstance(device, ttnn.MeshDevice):
            kw["mesh_mapper"] = shard_mapper(device, -1) if self.tp > 1 else ttnn.ReplicateTensorToMesh(device)
        self.weight = ttnn.from_torch(
            torch_module.weight.detach().bfloat16(),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            **kw,
        )

    def __call__(self, input_ids, **kwargs):
        if input_ids.dtype != ttnn.uint32:
            input_ids = ttnn.typecast(input_ids, ttnn.uint32)
        if input_ids.layout != ttnn.ROW_MAJOR_LAYOUT:
            input_ids = ttnn.to_layout(input_ids, ttnn.ROW_MAJOR_LAYOUT)
        out = ttnn.embedding(input_ids, self.weight, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
        if self.tp > 1:
            out = ttnn.all_gather(
                out, dim=len(out.shape) - 1, cluster_axis=1, num_links=1, topology=ttnn.Topology.Linear
            )
        return out


def build(device, torch_module=None):
    return TtTokenEmbed(device, torch_module)


def token_embed(device, torch_module=None):
    return TtTokenEmbed(device, torch_module)
