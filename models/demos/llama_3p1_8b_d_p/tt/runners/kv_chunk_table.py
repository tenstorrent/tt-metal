# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Build the native/legacy shared KvChunkAddressTable for Llama prefill.

The caches' declared layout (``cache_layout.py``, a ``CacheLayoutModel`` over the same caches) derives the
same table with ``to_chunk_map``. Until that replaces the walk (``LLAMA_KV_TABLE_FROM_LAYOUT_MODEL=1``), the
walked table is published and the derived one is checked against it; the model is written next to the table
(``<path>.cache_layout.pb``), for a consumer that derives the table itself, only when the two agree.
"""

import socket

from loguru import logger

from models.demos.common.prefill.runners.cache_layout import publish_kv_chunk_table

from .cache_layout import llama_cache_layout_model
from .kv_layout import PrefillKVLayout

# Set to 1 to publish the table derived from the caches' CacheLayoutModel without walking the classic one.
LAYOUT_MODEL_ENV = "LLAMA_KV_TABLE_FROM_LAYOUT_MODEL"


def _validated_layout(*, mesh_device, kv_cache, chunk_size):
    """The caches' ``PrefillKVLayout``; raises ValueError for caches the table cannot describe."""
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
    return layout


def build_kv_chunk_address_table(*, mesh_device, kv_cache, chunk_size):
    import ttnn
    from models.demos.llama_3p1_8b_d_p.tt.prefill_geometry import PREFILL_LAYOUT as prefill_layout

    layout = _validated_layout(mesh_device=mesh_device, kv_cache=kv_cache, chunk_size=chunk_size)
    nodes = {
        (row, col): mesh_device.get_fabric_node_id(ttnn.MeshCoordinate(row, col))
        for row in range(prefill_layout.sp)
        for col in range(prefill_layout.tp)
    }
    api = ttnn.experimental.disaggregation
    base_addresses = tuple(int(t.buffer_address()) for t in (kv_cache.k, kv_cache.v))
    host_name = socket.gethostname()
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
    for coord, node in nodes.items():
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


def build_and_serialize_kv_chunk_table(*, mesh_device, kv_cache, chunk_size, path):
    import ttnn
    from models.demos.llama_3p1_8b_d_p.tt.prefill_geometry import PREFILL_LAYOUT as prefill_layout

    def walk():
        return build_kv_chunk_address_table(mesh_device=mesh_device, kv_cache=kv_cache, chunk_size=chunk_size)

    def layout_model():
        return llama_cache_layout_model(mesh_device=mesh_device, kv_cache=kv_cache, chunk_size=chunk_size)

    # The caches are validated even when the derived table is published without walking.
    layout = _validated_layout(mesh_device=mesh_device, kv_cache=kv_cache, chunk_size=chunk_size)
    host_name = socket.gethostname()
    hosts = [
        (mesh_device.get_fabric_node_id(ttnn.MeshCoordinate(row, col)), host_name)
        for row in range(prefill_layout.sp)
        for col in range(prefill_layout.tp)
    ]
    publish_kv_chunk_table(
        path=path,
        walk=walk,
        layout_model=layout_model,
        hosts=hosts,
        enable_env=LAYOUT_MODEL_ENV,
        model_name="Llama-3.1-8B",
    )
    num_configs = len(layout.config_names)
    entries = num_configs * layout.num_slots * layout.num_layers * (layout.max_seq_len // 32)
    logger.info(f"[migration] KV chunk address table serialized to {path} (configs={num_configs}, entries={entries})")
    return path
