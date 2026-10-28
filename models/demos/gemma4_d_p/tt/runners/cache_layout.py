# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The Gemma 4 ring caches declared as a ``CacheLayoutModel`` (the cache tensor layout spec).

Every layer owns its own tensors, so every (config, layer) is a ``CacheConfig`` holding that one layer at
that tensor's base address:

    00..03_global_hH       global packed K|V head H: the sequence sharded over the CP rows of TP column H
    04..19_sliding_k_hH    sliding K head H of 16: the sequence sharded over the CP rows, the 16 heads over
    20..35_sliding_v_hH    the TP columns (4 per chip); each config selects its one head

The allocated tensors carry a Replicate topology (which head or sequence block a chip holds is decided at
write time), so the distribution is declared here instead. The writer's layout is the Natural scheme with
the Identity bank order: 32-token blocks round-robin over the banks in the order (slot, local head, block),
CP row r owning ``[n*chunk + r*chunk/cp, +chunk/cp)`` of each prefill chunk n (``device_chunk_size =
chunk/cp``). Sliding layers keep the full sequence (ring attention, not a ring buffer), so they are Dense.
"""

import ttnn
from models.demos.common.prefill.runners.cache_layout import row_major_mesh_id
from models.demos.gemma4_d_p.tt.attention.global_kv_cache import GLOBAL_PACKED_DIM, SLIDING_HEAD_DIM
from models.demos.gemma4_d_p.tt.attention.ring_prefill import TILE_HEIGHT, migration_ring_memory_config

disaggregation = ttnn.experimental.disaggregation

NUM_GLOBAL_HEADS = 4
NUM_SLIDING_HEADS = 16


def _coordinate(mesh_config, cp_row, tp_column):
    coordinates = [0, 0]
    coordinates[mesh_config.cp_axis] = cp_row
    coordinates[mesh_config.tp_axis] = tp_column
    return ttnn.MeshCoordinate(*coordinates)


def _spec(mesh_device, heads, row_dim, *, num_users, max_seq_len, arch):
    shape = ttnn.Shape([num_users, heads, max_seq_len, row_dim])
    memory_config = migration_ring_memory_config(mesh_device, row_dim)
    tensor_spec = ttnn.TensorSpec(shape, ttnn.bfloat8_b, ttnn.TILE_LAYOUT, memory_config)
    return disaggregation.KvLayoutSpec(tensor_spec, disaggregation.TemporalDense(), arch)


def _global_topology(mesh_config, head):
    """Global head ``head``: the sequence sharded over the CP rows of TP column ``head``."""
    cp = mesh_config.cp_degree
    coords = [_coordinate(mesh_config, row, head) for row in range(cp)]
    return ttnn.TensorTopology(ttnn.MeshShape([cp]), [ttnn.PlacementShard(2)], coords)


def _sliding_topology(mesh_device, mesh_config):
    """The sequence sharded over the CP axis and the heads over the TP axis."""
    placements = [None, None]
    placements[mesh_config.cp_axis] = ttnn.PlacementShard(2)
    placements[mesh_config.tp_axis] = ttnn.PlacementShard(1)
    shape = ttnn.MeshShape(*tuple(mesh_device.shape))
    coords = [ttnn.MeshCoordinate([c[i] for i in range(c.dims())]) for c in ttnn.MeshCoordinateRange(shape)]
    return ttnn.TensorTopology(shape, placements, coords)


def gemma4_cache_layout_model(*, mesh_device, mesh_config, kv_caches, chunk_size, config_names):
    """The ``CacheLayoutModel`` of ``kv_caches`` (``Gemma4KvCaches``) on ``mesh_device``, its configs named
    ``config_names`` (4 global, 16 sliding K, 16 sliding V)."""
    rows, cols = tuple(mesh_device.shape)
    fnids = [[mesh_device.get_fabric_node_id(ttnn.MeshCoordinate(r, c)) for c in range(cols)] for r in range(rows)]
    mesh_id = row_major_mesh_id(fnids, rows, cols, "the mesh")
    arch = mesh_device.arch()
    policy = disaggregation.GenerationPolicy(
        bank_scheme=disaggregation.BankScheme.Natural,
        bank_order=disaggregation.BankOrder.Identity,
        # Natural/Identity pages the sequence at k_chunk_size tokens: one DRAM block.
        k_chunk_size=TILE_HEIGHT,
        device_chunk_size=chunk_size // mesh_config.cp_degree,
    )
    spec_args = dict(num_users=kv_caches.num_users, max_seq_len=kv_caches.max_seq_len, arch=arch)
    global_spec = _spec(mesh_device, 1, GLOBAL_PACKED_DIM, **spec_args)
    sliding_spec = _spec(mesh_device, NUM_SLIDING_HEADS, SLIDING_HEAD_DIM, **spec_args)
    sliding_topology = _sliding_topology(mesh_device, mesh_config)

    caches = []

    def add(name, spec, topology, tensor, layer, head=None):
        caches.append(
            disaggregation.CacheConfig(
                spec,
                topology,
                policy,
                int(tensor.buffer_address()),
                name=name,
                first_layer=layer,
                num_layers=1,
                mesh_id=mesh_id,
                head=head,
            )
        )

    global_names = config_names[:NUM_GLOBAL_HEADS]
    k_names = config_names[NUM_GLOBAL_HEADS : NUM_GLOBAL_HEADS + NUM_SLIDING_HEADS]
    v_names = config_names[NUM_GLOBAL_HEADS + NUM_SLIDING_HEADS :]
    for layer in kv_caches.global_layers:
        for head, name in enumerate(global_names):
            add(name, global_spec, _global_topology(mesh_config, head), kv_caches[layer].kv, layer)
    for layer in kv_caches.sliding_layers:
        cache = kv_caches[layer]
        for head in range(NUM_SLIDING_HEADS):
            add(k_names[head], sliding_spec, sliding_topology, cache.k, layer, head)
            add(v_names[head], sliding_spec, sliding_topology, cache.v, layer, head)
    geometry = disaggregation.MapGeometry(
        len(kv_caches), kv_caches.num_users, kv_caches.max_seq_len, position_step=TILE_HEIGHT
    )
    return disaggregation.CacheLayoutModel(caches, mesh_id, ttnn.MeshShape(rows, cols), geometry)
