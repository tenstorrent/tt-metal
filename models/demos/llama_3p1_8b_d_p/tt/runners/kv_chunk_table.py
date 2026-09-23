# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Build the native/legacy shared KvChunkAddressTable for Llama prefill."""

import socket

from .kv_layout import PrefillKVLayout


def build_address_table(*, layout, base_addresses, fabric_nodes, host_name, api):
    """Populate the shared table API from explicit cache geometry and chip ownership."""
    required = {(row, col) for row in range(layout.sp) for col in range(layout.tp)}
    if set(fabric_nodes) != required or len(base_addresses) != 2:
        raise ValueError(f"table requires every SP{layout.sp}/TP{layout.tp} fabric node and separate K/V bases")
    configs = {}
    for name in layout.config_names:
        cfg = api.KvChunkAddressTableConfig()
        cfg.num_layers = layout.num_layers
        cfg.max_sequence_length = layout.max_seq_len
        cfg.num_slots = layout.num_slots
        cfg.chunk_n_tokens = 32
        cfg.chunk_size_bytes = layout.chunk_size_bytes
        configs[name] = cfg
    table = api.KvChunkAddressTable(configs)
    if tuple(table.config_name(i) for i in range(table.num_configs())) != layout.config_names:
        raise RuntimeError("shared table changed the K/V config order")
    groups = {}
    for coord, node in fabric_nodes.items():
        groups[coord] = table.add_device_group([node])
        table.set_fabric_node_host(node, host_name=host_name)
    for config in range(len(layout.config_names)):
        for slot in range(layout.num_slots):
            for layer in range(layout.num_layers):
                for position in range(0, layout.max_seq_len, 32):
                    coord, bank, offset = layout.locate(config, layer, position, slot, base_addresses[config // 8])
                    location = api.KvCacheLocation()
                    location.noc_addr = (bank << 32) | offset
                    location.size_bytes = layout.chunk_size_bytes
                    location.device_group_index = groups[coord]
                    table.set(layer, position, slot, location, config)
    return table


def build_kv_chunk_address_table(*, mesh_device, kv_cache, chunk_size):
    import ttnn
    from models.demos.common.prefill.runners.migration import get_num_dram_banks
    from models.demos.llama_3p1_8b_d_p.tt.prefill_geometry import PREFILL_LAYOUT as prefill_layout

    if tuple(mesh_device.shape) != prefill_layout.mesh_shape or kv_cache.sp != prefill_layout.sp:
        raise ValueError(f"Llama prefill table requires SP{prefill_layout.sp}/TP{prefill_layout.tp}")
    layout = PrefillKVLayout(
        max_seq_len=kv_cache.max_seq_len,
        num_slots=kv_cache.num_users,
        num_layers=kv_cache.num_layers,
        num_banks=get_num_dram_banks(mesh_device),
        chunk_size=chunk_size,
    )
    shape = (layout.num_slots * layout.num_layers, 1, layout.max_seq_len // layout.sp, layout.head_dim)
    for tensor in (kv_cache.k, kv_cache.v):
        spec = tensor.memory_config().nd_shard_spec
        if tuple(tensor.shape) != shape or tensor.dtype != ttnn.bfloat8_b or tensor.layout != ttnn.TILE_LAYOUT:
            raise ValueError("migration requires metadata-matching BF8_B TILE K/V caches")
        if tensor.device() != mesh_device or tensor.memory_config().buffer_type != ttnn.BufferType.DRAM:
            raise ValueError("migration cache must reside in DRAM on the supplied mesh")
        if (
            spec is None
            or tuple(spec.shard_shape) != (1, 1, 32, 128)
            or spec.orientation != ttnn.ShardOrientation.ROW_MAJOR
            or spec.shard_distribution_strategy != ttnn.ShardDistributionStrategy.ROUND_ROBIN_1D
        ):
            raise ValueError("migration requires 32-token round-robin NdShard cache pages")
        expected_grid = ttnn.CoreRangeSet(
            [ttnn.CoreRange(ttnn.CoreCoord(bank, 0), ttnn.CoreCoord(bank, 0)) for bank in range(layout.num_banks)]
        )
        if spec.grid != expected_grid:
            raise ValueError("NdShard DRAM bank grid differs from table bank order")
    nodes = {
        (row, col): mesh_device.get_fabric_node_id(ttnn.MeshCoordinate(row, col))
        for row in range(prefill_layout.sp)
        for col in range(prefill_layout.tp)
    }
    return build_address_table(
        layout=layout,
        base_addresses=tuple(int(t.buffer_address()) for t in (kv_cache.k, kv_cache.v)),
        fabric_nodes=nodes,
        host_name=socket.gethostname(),
        api=ttnn.experimental.disaggregation,
    )


def build_and_serialize_kv_chunk_table(*, mesh_device, kv_cache, chunk_size, path):
    from models.demos.common.prefill.runners.migration import serialize_prebuilt_kv_chunk_table

    table = build_kv_chunk_address_table(mesh_device=mesh_device, kv_cache=kv_cache, chunk_size=chunk_size)
    return serialize_prebuilt_kv_chunk_table(table=table, path=path)
