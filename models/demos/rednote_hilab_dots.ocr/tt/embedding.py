# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""TTNN text token embedding for dots.ocr.

dots.ocr text embedding (Qwen2-style ``model.embed_tokens``): a plain
``F.embedding`` lookup on a [vocab=151936, hidden=1536] table, untied from the
lm_head. Maps onto the single ``ttnn.embedding`` op (KB entry ttnn_embedding:
indices uint32 ROW_MAJOR, weight table kept as-is in DRAM ROW_MAJOR,
TILE_LAYOUT output for direct fusion into downstream ops).

Parallelism plan (ARCHITECTURE.md): placement=shard (per-token cadence). The
reference_impl models/tt_transformers/tt/embedding.py shards the embedding
TABLE across the mesh on the hidden dim (``ShardTensor2dMesh dims=(None, 3)``
on the [1, 1, vocab, hidden] weight) — each device looks up the full vocab and
produces its hidden-dim slice, then a CCL combine (``ttnn.all_gather`` on the
hidden dim, Topology.Linear) reconstructs the replicated [.., hidden]
activation. We follow that exact table-shard + CCL pattern (the plan's
"shard ... with all-reduce, tt_transformers pattern"); sharding the hidden dim
rather than the vocab dim is what the named reference pattern does — a
vocab-dim shard would need per-shard index offsetting/masking that
ttnn.embedding (and the reference) does not implement. On a single device the
mesh_mapper and the gather degenerate gracefully.
"""

import ttnn
from models.common.lightweightmodule import LightweightModule


class TtEmbedding(LightweightModule):
    """dots.ocr text token embedding: ttnn.embedding on a hidden-dim-sharded table.

    Args:
        mesh_device: ttnn mesh device handle (table sharded on hidden dim).
        state_dict: {"weight": [vocab, hidden]} torch tensor (HF key
            model.embed_tokens.weight, untied from lm_head).
        dtype: on-device table dtype.
    """

    def __init__(self, mesh_device, state_dict, dtype=ttnn.bfloat16):
        super().__init__()
        self.mesh_device = mesh_device
        self.num_devices = mesh_device.get_num_devices()

        weight = state_dict["weight"]  # [vocab, hidden]
        # tt_transformers Embedding keeps the table [1, 1, vocab, hidden]
        # ROW_MAJOR in DRAM, sharded on the hidden dim across the mesh.
        self.weight = ttnn.from_torch(
            weight.unsqueeze(0).unsqueeze(0),
            dtype=dtype,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=-1),
        )

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """x: [1, 1, 1, batch*seq] uint32 ROW_MAJOR token ids, replicated.

        Returns: [1, 1, batch*seq, hidden] TILE_LAYOUT, replicated across the
        mesh (per-device hidden slices recombined via all_gather).
        """
        e = ttnn.embedding(x, self.weight, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        e = ttnn.unsqueeze_to_4D(e)  # embedding emits rank 3; all_gather + callers expect [B, 1, S, H]
        if self.num_devices > 1:
            e = ttnn.all_gather(e, dim=3, topology=ttnn.Topology.Linear)
        return e
