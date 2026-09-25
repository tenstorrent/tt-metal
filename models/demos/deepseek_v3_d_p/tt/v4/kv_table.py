# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The V4-Flash KvChunkAddressTable: one config per contract group, LINEAR addressing, SP-replicated caches.

Why not the MLA builder: ``populate_kv_chunk_address_table_kimi`` replays a block-cyclic TOKEN layout (a 5120-token
period split over the SP rows into 32-row bank shards). V4's compressed axes are ENTRY axes (one per 4 or 128 tokens),
so a chunk holds 40 HCA entries -- not 32 per SP shard -- and the V4 caches are replicated over SP anyway
(``tt/v4/kv_cache.py``). Here each group tensor ``[users * layers_in_group, 1, rows, width]`` is ND-sharded in
32-row shards round-robin over the DRAM banks (``init_kvpe_cache``), which makes the address of unified row ``r`` of
(slot, layer_in_group) a pure function of its flat 32-row chunk index::

    chunk  = ((slot * L + layer) * rows + r) // 32
    bank   = chunk % num_banks
    offset = base + (chunk // num_banks) * chunk_size_bytes

The table's "layer" for a config is the KIND-RANK (0..L-1 over the whole model, so PP stages merge), its
"position" is the unified row (multiples of 32), and every chip of a stage holds a replica, so ONE device group
per stage covers all its chips. ``walk_linear`` is the pure address walk (device-free, tested); the ``populate_*``
and ``build_*`` functions wrap it in the ttnn table API.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

from models.demos.deepseek_v3_d_p.tt.v4.kv_contract import CHUNK_N_TOKENS, CONTRACT, KvGroupSpec


@dataclass(frozen=True)
class ChunkAddr:
    layer: int  # kind-rank
    position: int  # unified row of the chunk's first row
    slot: int
    bank: int
    offset: int  # absolute DRAM byte offset within the bank

    @property
    def noc_addr(self) -> int:
        return (self.bank << 32) | self.offset


def walk_linear(
    *,
    num_slots: int,
    num_layers: int,
    rows: int,
    num_banks: int,
    base_addr: int,
    chunk_size_bytes: int,
    first_layer: int = 0,
    extent: int | None = None,
) -> Iterator[ChunkAddr]:
    """Every (slot, layer, 32-row chunk) of one group tensor, in storage order. ``rows`` is the tensor's ALLOCATED
    row count (a multiple of 32; it fixes the addresses); only chunks whose position is below ``extent`` (default
    ``rows``) are emitted -- the writers' headroom rows are never migrated. ``first_layer`` offsets the table's
    layer index (a PP stage's first kind-rank); the tensor's own batch index is ``slot * num_layers + local_layer``."""
    if rows % CHUNK_N_TOKENS:
        raise ValueError(f"rows {rows} must be a multiple of {CHUNK_N_TOKENS}")
    extent = rows if extent is None else int(extent)
    if extent % CHUNK_N_TOKENS or extent > rows:
        raise ValueError(f"extent {extent} must be a multiple of {CHUNK_N_TOKENS} and <= rows {rows}")
    chunks_per_layer = rows // CHUNK_N_TOKENS
    flat = 0
    for slot in range(int(num_slots)):
        for local_layer in range(int(num_layers)):
            for c in range(chunks_per_layer):
                bank = flat % num_banks
                offset = int(base_addr) + (flat // num_banks) * int(chunk_size_bytes)
                position = c * CHUNK_N_TOKENS
                if position < extent:
                    yield ChunkAddr(
                        layer=int(first_layer) + local_layer, position=position, slot=slot, bank=bank, offset=offset
                    )
                flat += 1


def kind_rank_range(all_layers_of_kind: list[int], my_layers_of_kind: tuple) -> tuple[int, int]:
    """(first kind-rank, count) of this rank's slice of one kind: PP ranks own contiguous global layer ranges, so
    their layers of a kind are contiguous in kind-rank too."""
    if not my_layers_of_kind:
        return 0, 0
    first = all_layers_of_kind.index(my_layers_of_kind[0])
    assert list(all_layers_of_kind[first : first + len(my_layers_of_kind)]) == list(my_layers_of_kind)
    return first, len(my_layers_of_kind)


def group_table_config(spec: KvGroupSpec, *, max_seq_len: int, num_layers_total: int, num_slots: int):
    """The ``KvChunkAddressTableConfig`` for one contract group (imported lazily: needs ttnn)."""
    import ttnn

    cfg = ttnn.experimental.disaggregation.KvChunkAddressTableConfig()
    cfg.num_layers = int(num_layers_total)
    cfg.max_sequence_length = int(spec.extent(max_seq_len))
    cfg.num_slots = int(num_slots)
    cfg.chunk_n_tokens = CHUNK_N_TOKENS
    cfg.chunk_size_bytes = int(spec.chunk_size_bytes)
    return cfg


def populate_group(
    table, config_id: int, *, spec: KvGroupSpec, rows: int, num_slots: int, stages: list, extent: int | None = None
) -> None:
    """Fill config ``config_id`` from per-stage descriptors ``{first_layer, count, base_addr, num_banks, host_tag,
    fnids}`` (``allgather_kv_stage_layout`` output, with the layer fields in KIND-RANK units). One device group
    per stage = every chip of the stage (the rows are replicated)."""
    import ttnn

    for st in stages:
        if int(st["count"]) == 0:
            continue
        fnids = [fid for row in st["fnids"] for fid in row]
        group_idx = table.add_device_group(fnids)
        host_name = f"host-{st['host_tag']:08x}"
        for fid in fnids:
            table.set_fabric_node_host(fid, host_name=host_name)
        for a in walk_linear(
            num_slots=num_slots,
            num_layers=int(st["count"]),
            rows=rows,
            num_banks=int(st["num_banks"]),
            base_addr=int(st["base_addr"]),
            chunk_size_bytes=spec.chunk_size_bytes,
            first_layer=int(st["first_layer"]),
            extent=extent,
        ):
            loc = ttnn.experimental.disaggregation.KvCacheLocation()
            loc.noc_addr = a.noc_addr
            loc.size_bytes = spec.chunk_size_bytes
            loc.device_group_index = group_idx
            table.set(a.layer, a.position, a.slot, loc, config_id)


def _stage_layout(mesh_device, base: int, mesh_shape, first: int, count: int) -> list:
    """The per-stage descriptors: the runner's cross-rank all-gather when a distributed context with more than one
    rank is up, else this process's own single stage (the mock-migration / in-process Gate 1 path)."""
    import ttnn
    from models.demos.common.prefill.runners.migration import allgather_kv_stage_layout, get_num_dram_banks

    try:
        size = int(ttnn.distributed_context_get_size())
    except Exception:  # no distributed context in this process
        size = 1
    if size > 1:
        return allgather_kv_stage_layout(mesh_device, base, mesh_shape, first, count)
    rows, cols = int(mesh_shape[0]), int(mesh_shape[1])
    fnids = [[mesh_device.get_fabric_node_id(ttnn.MeshCoordinate(r, c)) for c in range(cols)] for r in range(rows)]
    return [
        {
            "rank": 0,
            "first_layer": int(first),
            "count": int(count),
            "base_addr": int(base),
            "num_banks": int(get_num_dram_banks(mesh_device)),
            "host_tag": 0,
            "fnids": fnids,
        }
    ]


def build_v4_kv_chunk_table(
    *, mesh_device, caches, hf_config, num_slots: int, path: str, include_pending: bool = True
) -> str:
    """COLLECTIVE over the runner's ranks: build the merged multi-config table for the contract groups this
    model has and serialize it to ``path``. Every rank calls this (the per-config stage all-gather is
    symmetric); ``caches`` is this rank's ``V4FlashKvCaches``. Config order = contract order, pending last."""
    import ttnn
    from models.demos.common.prefill.runners.migration import serialize_prebuilt_kv_chunk_table
    from models.demos.deepseek_v3_d_p.tt.v4.layer_kinds import layers_of_kind

    geom = caches.geometry
    specs = [g for g in CONTRACT if include_pending or not g.pending]
    all_of_kind = {g.name: layers_of_kind(hf_config, g.kind) for g in specs}
    configs, plan = [], []
    for g in specs:
        n_total = len(all_of_kind[g.name])
        if n_total == 0:
            continue
        configs.append(
            group_table_config(g, max_seq_len=geom.max_seq_len, num_layers_total=n_total, num_slots=num_slots)
        )
        plan.append(g)
    table = ttnn.experimental.disaggregation.KvChunkAddressTable(configs)
    mesh_shape = list(mesh_device.shape)
    for config_id, g in enumerate(plan):
        tensor = caches.group_tensors().get(g.name)
        first, count = kind_rank_range(all_of_kind[g.name], geom.layers(g.name))
        base = int(tensor.buffer_address()) if tensor is not None else 0
        stages = _stage_layout(mesh_device, base, mesh_shape, first, count)
        populate_group(
            table,
            config_id,
            spec=g,
            rows=geom.rows(g.name),
            num_slots=num_slots,
            stages=stages,
            extent=geom.extent(g.name),
        )
    return serialize_prebuilt_kv_chunk_table(table=table, path=path)
