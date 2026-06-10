# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""TTNN text token embedding for dots.ocr.

dots.ocr text embedding (Qwen2-style ``model.embed_tokens``): a plain
``F.embedding`` lookup on a [vocab=151936, hidden=1536] table, untied from the
lm_head. Maps onto the single ``ttnn.embedding`` op (KB entry ttnn_embedding:
indices uint32 ROW_MAJOR, weight table kept as-is in DRAM ROW_MAJOR,
TILE_LAYOUT output for direct fusion into downstream ops).

Parallelism plan (ARCHITECTURE.md) said placement=shard (per-token cadence),
following reference_impl models/tt_transformers/tt/embedding.py: table sharded
on the hidden dim, per-device slices recombined by ``ttnn.all_gather``. The
optimization tick (occupancy redo, tracy under --traced at the production
shapes) measured the recombining all_gather at 80% of the decode-step block
kernel time (15.5 us/chip at ids [1,1,1,32], link-latency-bound: num_links
1->2 moved it only -6%) and 96% at the 2.8k prefill shape (331 us/chip at
1 link, 167 us at the 2-link QB HW ceiling). Replicating the table instead
(the tp-guidance "embedding (small models)" row; 467 MB bf16 per chip of the
34 GB DRAM) deletes the CCL outright: decode 18.4 -> 9.2 us/chip, prefill
180.9 -> 45.2 us/chip vs the best sharded variant. placement="replicate" is
therefore the production default; the original shard + all_gather(num_links)
path is kept behind placement="shard" for DRAM-tight configurations. Lookup
occupancy is the embedding kernel's row-tile axis: 88/110 cores at prefill
(one core per 32-row tile), 1 core at the single-row-tile decode shape — the
kernel ceiling for that shape. On a single device the mesh_mapper degenerates
gracefully either way.
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
        num_links: ethernet links for the recombining all_gather (sharded
            placement only). QB hops expose 2 links (TT_FATAL at 4,
            vision_attention tick evidence); 2 measured ~2x faster than the
            1-link default at the prefill shape ([1,1,2816,1536]).
        placement: "replicate" (default) keeps the full table on every chip
            so the lookup needs NO CCL — the all_gather was 80% of the
            decode-step block kernel time (15.5 us/chip) and 96% at prefill
            (331 us/chip), while the replicated table costs 467 MB of the
            34 GB chip DRAM. "shard" is the original hidden-dim-sharded
            table + all_gather recombine (tt_transformers pattern) — keep it
            if DRAM budget ever gets tight.
    """

    def __init__(self, mesh_device, state_dict, dtype=ttnn.bfloat16, num_links=2, placement="replicate"):
        super().__init__()
        assert placement in ("replicate", "shard"), placement
        self.mesh_device = mesh_device
        self.num_devices = mesh_device.get_num_devices()
        self.num_links = num_links
        self.placement = placement

        weight = state_dict["weight"]  # [vocab, hidden]
        # Table kept [1, 1, vocab, hidden] ROW_MAJOR in DRAM. replicate: full
        # table per chip, CCL-free lookup. shard: tt_transformers pattern —
        # hidden-dim shards recombined by all_gather in forward.
        mapper = (
            ttnn.ReplicateTensorToMesh(mesh_device)
            if placement == "replicate"
            else ttnn.ShardTensorToMesh(mesh_device, dim=-1)
        )
        self.weight = ttnn.from_torch(
            weight.unsqueeze(0).unsqueeze(0),
            dtype=dtype,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=mapper,
        )

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """x: [1, 1, 1, batch*seq] uint32 ROW_MAJOR token ids, replicated.

        Returns: [1, 1, batch*seq, hidden] TILE_LAYOUT, replicated across the
        mesh (replicated-table lookup, or hidden-dim slices recombined via
        all_gather under placement="shard").
        """
        e = ttnn.embedding(x, self.weight, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        e = ttnn.unsqueeze_to_4D(e)  # embedding emits rank 3; callers expect [B, 1, S, H]
        if self.num_devices > 1 and self.placement == "shard":
            e = ttnn.all_gather(e, dim=3, num_links=self.num_links, topology=ttnn.Topology.Linear)
        return e
