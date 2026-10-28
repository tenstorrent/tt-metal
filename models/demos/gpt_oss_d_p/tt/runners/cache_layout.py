# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""The GPT-OSS prefill caches declared as a ``CacheLayoutModel`` (the cache tensor layout spec).

Each head of each cache is a ``CacheConfig``: the tensor's ``KvLayoutSpec`` (its logical shape, dtype and
DRAM ND-shard spec, and how its sequence is retained), the mesh distribution its chunks really have, the
writer's ``GenerationPolicy``, and the mesh, layer range and base address.
``ttnn.experimental.disaggregation.to_chunk_map`` derives the KV chunk address table from the model.

The allocated tensors carry a 1-D Replicate topology (which head or sequence block a chip holds is decided
at write time), so the distribution is declared here instead:

    configs 00..N-1   k head h:   the sequence sharded over the SP rows of TP column h
    configs N..2N-1   v head h:   likewise

The caches are ``[num_users * num_layers, 1, max_seq_len, head_dim]``, user-major, so ``to_chunk_map``
folds the layers into the batch axis. Every layer, sliding or full, has a full-length slot and publishes
every position. The writer's layout is the Natural scheme with the Identity bank order: 32-token blocks
round-robin over the banks, and SP row r owning ``[n*chunk + r*chunk/sp, +chunk/sp)`` of each prefill chunk
n (chunk-modulo ownership with ``device_chunk_size = chunk/sp``).

The bounded sliding cache (``GPT_OSS_BOUNDED_SLIDING_KV=1``) has no table and is not declared.
"""

import ttnn
from models.demos.common.prefill.runners.cache_layout import CacheLayoutNotDerivable, row_major_mesh_id
from models.demos.common.prefill.runners.migration import get_num_dram_banks
from models.demos.gpt_oss_d_p.tt.attention.kv_cache import NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK, cache_memory_config

disaggregation = ttnn.experimental.disaggregation


def _spec(dtype, *, num_users, num_layers, seq_len, head_dim, num_dram_banks, arch):
    shape = ttnn.Shape([num_users * num_layers, 1, seq_len, head_dim])
    tensor_spec = ttnn.TensorSpec(shape, dtype, ttnn.TILE_LAYOUT, cache_memory_config(head_dim, num_dram_banks))
    return disaggregation.KvLayoutSpec(tensor_spec, disaggregation.TemporalDense(), arch)


def _head_topology(sp, head):
    """K/V head ``head``: the sequence sharded over the SP rows of TP column ``head``."""
    coords = [ttnn.MeshCoordinate(row, head) for row in range(sp)]
    return ttnn.TensorTopology(ttnn.MeshShape([sp]), [ttnn.PlacementShard(2)], coords)


def gpt_oss_cache_layout_model(
    *,
    mesh_device,
    kv_cache,
    sp_axis,
    sp,
    cols,
    num_users,
    num_layers,
    seq_len,
    chunk_size,
    num_kv_heads,
    head_dim,
):
    """The ``CacheLayoutModel`` of the GPT-OSS prefill caches on ``mesh_device``: one config per (cache, head),
    k heads then v heads, named by zero-padded position, each over this rank's ``num_layers`` layers.

    Raises CacheLayoutNotDerivable for a bounded sliding cache, a sequence sharded over the mesh columns, or a
    mesh whose chips are not numbered row-major."""
    from models.demos.gpt_oss_d_p.tt.runners.kv_chunk_table import _stable_config_name

    if getattr(kv_cache, "bounded_sliding", False):
        raise CacheLayoutNotDerivable("a bounded sliding cache keeps its sliding layers in a circular cache")
    if sp_axis != 0:
        raise CacheLayoutNotDerivable(f"the sequence is sharded over mesh axis {sp_axis}, not the rows")
    assert num_kv_heads == cols, f"num_kv_heads ({num_kv_heads}) must equal the TP column count ({cols})"
    fnids = [[mesh_device.get_fabric_node_id(ttnn.MeshCoordinate(r, c)) for c in range(cols)] for r in range(sp)]
    mesh_id = row_major_mesh_id(fnids, sp, cols, "the prefill mesh")
    num_dram_banks = get_num_dram_banks(mesh_device)
    arch = mesh_device.arch()
    policy = disaggregation.GenerationPolicy(
        bank_scheme=disaggregation.BankScheme.Natural,
        bank_order=disaggregation.BankOrder.Identity,
        # Natural/Identity pages the sequence at k_chunk_size tokens: one DRAM block.
        k_chunk_size=NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK,
        device_chunk_size=chunk_size // sp,
    )
    num_configs = 2 * num_kv_heads
    caches = []
    for cache_idx, tensor in enumerate((kv_cache.k, kv_cache.v)):
        spec = _spec(
            tensor.dtype,
            num_users=num_users,
            num_layers=num_layers,
            seq_len=seq_len,
            head_dim=head_dim,
            num_dram_banks=num_dram_banks,
            arch=arch,
        )
        for head in range(num_kv_heads):
            caches.append(
                disaggregation.CacheConfig(
                    spec,
                    _head_topology(sp, head),
                    policy,
                    int(tensor.buffer_address()),
                    name=_stable_config_name(cache_idx * num_kv_heads + head, num_configs),
                    layers=[],
                    first_layer=0,
                    num_layers=num_layers,
                    mesh_id=mesh_id,
                )
            )
    geometry = disaggregation.MapGeometry(
        num_layers, num_users, seq_len, position_step=NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK
    )
    return disaggregation.CacheLayoutModel(caches, mesh_id, ttnn.MeshShape(sp, cols), geometry)
