# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The Llama-3.1-8B prefill caches declared as a ``CacheLayoutModel`` (the cache tensor layout spec).

Each K/V head is a ``CacheConfig``: the cache tensor's ``KvLayoutSpec`` (its logical shape, dtype and DRAM
ND-shard spec, and how its sequence is retained), the mesh distribution its chunks really have, the writer's
``GenerationPolicy``, and the mesh, layer range and base address. ``ttnn.experimental.disaggregation.to_chunk_map``
derives the KV chunk address table from the model.

The allocated tensors carry a replicated topology (which head or sequence block a chip holds is decided at
write time), so the distribution is declared here instead:

    configs k_h0..k_h7   k head h:   the sequence sharded over the SP rows of TP column h
    configs v_h0..v_h7   v head h:   likewise

The caches are ``[num_users * num_layers, 1, max_seq_len / sp, head_dim]`` per chip, user-major, so
``to_chunk_map`` folds the layers into the batch axis. The writer's layout is the Natural scheme with the
Identity bank order: 32-token blocks round-robin over the banks, and SP row r owning ``[n*chunk + r*chunk/sp,
+chunk/sp)`` of each prefill chunk n (chunk-modulo ownership with ``device_chunk_size = chunk/sp``).
"""

import ttnn
from models.demos.common.prefill.runners.cache_layout import row_major_mesh_id
from models.demos.llama_3p1_8b_d_p.reference.llama_3p1_8b_config import Llama31_8BConfig as Model
from models.demos.llama_3p1_8b_d_p.tt.kv_cache import _cache_memory_config
from models.demos.llama_3p1_8b_d_p.tt.prefill_geometry import PREFILL_LAYOUT as prefill_layout

disaggregation = ttnn.experimental.disaggregation


def _spec(tensor, memory_config, *, num_users, num_layers, max_seq_len, arch):
    # The global sequence: the topology shards axis 2 over the SP rows.
    shape = ttnn.Shape([num_users * num_layers, 1, max_seq_len, Model.HEAD_DIM])
    tensor_spec = ttnn.TensorSpec(shape, tensor.dtype, ttnn.TILE_LAYOUT, memory_config)
    return disaggregation.KvLayoutSpec(tensor_spec, disaggregation.TemporalDense(), arch)


def _head_topology(sp, head):
    """K/V head ``head``: the sequence sharded over the SP rows of TP column ``head``."""
    coords = [ttnn.MeshCoordinate(row, head) for row in range(sp)]
    return ttnn.TensorTopology(ttnn.MeshShape([sp]), [ttnn.PlacementShard(2)], coords)


def llama_cache_layout_model(*, mesh_device, kv_cache, chunk_size):
    """The ``CacheLayoutModel`` of the Llama prefill K/V caches on ``mesh_device`` (one full-model rank).

    Raises CacheLayoutNotDerivable for a mesh whose chips are not numbered row-major."""
    sp, tp = prefill_layout.sp, prefill_layout.tp
    fnids = [[mesh_device.get_fabric_node_id(ttnn.MeshCoordinate(r, c)) for c in range(tp)] for r in range(sp)]
    mesh_id = row_major_mesh_id(fnids, sp, tp, "the Llama prefill mesh")
    policy = disaggregation.GenerationPolicy(
        bank_scheme=disaggregation.BankScheme.Natural,
        bank_order=disaggregation.BankOrder.Identity,
        # Natural/Identity pages the sequence at k_chunk_size tokens: one cache page.
        k_chunk_size=prefill_layout.cache_page_size,
        device_chunk_size=chunk_size // sp,
    )
    memory_config = _cache_memory_config(mesh_device)
    arch = mesh_device.arch()
    caches = []
    for kv, tensor in (("k", kv_cache.k), ("v", kv_cache.v)):
        spec = _spec(
            tensor,
            memory_config,
            num_users=kv_cache.num_users,
            num_layers=kv_cache.num_layers,
            max_seq_len=kv_cache.max_seq_len,
            arch=arch,
        )
        for head in range(Model.NUM_KEY_VALUE_HEADS):
            caches.append(
                disaggregation.CacheConfig(
                    spec,
                    _head_topology(sp, head),
                    policy,
                    int(tensor.buffer_address()),
                    name=f"{kv}_h{head}",
                    first_layer=0,
                    num_layers=kv_cache.num_layers,
                    mesh_id=mesh_id,
                )
            )
    geometry = disaggregation.MapGeometry(
        kv_cache.num_layers,
        kv_cache.num_users,
        kv_cache.max_seq_len,
        position_step=prefill_layout.cache_page_size,
    )
    return disaggregation.CacheLayoutModel(caches, mesh_id, ttnn.MeshShape(sp, tp), geometry)
