# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""KV chunk address table builder for Llama 3.1 8B GQA prefill.

Handles n_kv_local > 1 (2 KV heads per TP column), unlike the GPT-OSS builder which
assumes n_kv_local == 1 (num_kv_heads == cols).

Per-chip cache shape: ``[num_users * num_layers, n_kv_local, seq_local, head_dim]``
with DRAM NdShard ROUND_ROBIN_1D over 32-token banks.

The DRAM bank walk is linearized: for each batch index (slot * num_layers + layer),
local head 0's sequence banks come first, then local head 1's. The shard index for
a given (batch, local_head, seq_bank) is::

    shard_idx = batch * (n_kv_local * seq_banks_per_chip)
              + local_head * seq_banks_per_chip
              + seq_bank

Config layout (id order is the src<->dst migration contract)::

    config 0..N-1   -> k head 0..N-1   (device group: head h's TP column = h // n_kv_local)
    config N..2N-1  -> v head 0..N-1   (device group: head h's TP column)

``N == num_kv_heads`` (8 for Llama 3.1 8B). Each config maps one global KV head to
the TP column that holds it (column = h // n_kv_local) and the local head index
within that column (h % n_kv_local).
"""

from __future__ import annotations

import socket

from loguru import logger

import ttnn
from models.demos.common.prefill.runners.migration import get_num_dram_banks
from models.demos.llama_3_1_8b.tt.attention.kv_cache import NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK

_TILE_BYTES = {ttnn.bfloat8_b: 1088, ttnn.bfloat16: 2048}


def _chunk_size_bytes(dtype, head_dim: int) -> int:
    assert head_dim % 32 == 0
    return (head_dim // 32) * _TILE_BYTES[dtype]


def _make_config(*, num_layers, max_seq_len, num_users, chunk_size_bytes):
    cfg = ttnn.experimental.disaggregation.KvChunkAddressTableConfig()
    cfg.num_layers = num_layers
    cfg.max_sequence_length = max_seq_len
    cfg.num_slots = num_users
    cfg.chunk_n_tokens = NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK
    cfg.chunk_size_bytes = chunk_size_bytes
    return cfg


def _stable_config_name(config_id: int, num_configs: int) -> str:
    width = max(2, len(str(max(num_configs - 1, 0))))
    return f"{config_id:0{width}d}"


def build_kv_chunk_address_table(
    *,
    mesh_device,
    kv_cache,
    seq_len,
    num_layers,
    mesh_shape,
    sp_axis,
    num_users,
    chunk_size,
    num_kv_heads,
    head_dim,
):
    """Build the Llama multi-config block-cyclic KV chunk address table.

    Handles n_kv_local > 1 by computing shard indices directly from (batch, local_head, seq_bank)
    rather than using a running bank walk that assumes one head per column.
    """
    tp_axis = 1 - sp_axis
    sp = mesh_shape[sp_axis]
    cols = mesh_shape[tp_axis]
    num_dram_banks = get_num_dram_banks(mesh_device)

    assert num_kv_heads % cols == 0, f"num_kv_heads ({num_kv_heads}) must be divisible by TP columns ({cols})"
    n_kv_local = num_kv_heads // cols
    assert seq_len % chunk_size == 0
    tokens_per_chunk_local = chunk_size // sp
    assert tokens_per_chunk_local % NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK == 0
    seq_local = seq_len // sp
    seq_banks_per_chip = seq_local // NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK
    banks_per_chunk_local = tokens_per_chunk_local // NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK
    num_chunks_per_seq_len = seq_len // chunk_size

    for name, t in (("k", kv_cache.k), ("v", kv_cache.v)):
        assert (
            t.shape[0] == num_users * num_layers
        ), f"{name} cache batch dim {t.shape[0]} != num_users({num_users}) * num_layers({num_layers})"

    # Config layout: k_h0..k_h{N-1}, v_h0..v_h{N-1}.
    # Each spec: (label, tensor, target_col, target_local_head, dtype)
    specs = []
    for h in range(num_kv_heads):
        specs.append((f"k_h{h}", kv_cache.k, h // n_kv_local, h % n_kv_local, kv_cache.k.dtype))
    for h in range(num_kv_heads):
        specs.append((f"v_h{h}", kv_cache.v, h // n_kv_local, h % n_kv_local, kv_cache.v.dtype))

    num_configs = len(specs)
    configs_by_name = {
        _stable_config_name(i, num_configs): _make_config(
            num_layers=num_layers,
            max_seq_len=seq_len,
            num_users=num_users,
            chunk_size_bytes=_chunk_size_bytes(dtype, head_dim),
        )
        for i, (_, _, _, _, dtype) in enumerate(specs)
    }
    table = ttnn.experimental.disaggregation.KvChunkAddressTable(configs_by_name)
    assert table.num_configs() == num_configs
    for i in range(num_configs):
        assert table.config_name(i) == _stable_config_name(i, num_configs)

    host_name = socket.gethostname()
    hosts_set = set()

    for config_id, (label, tensor, target_col, target_lh, dtype) in enumerate(specs):
        base_addr = tensor.buffer_address()
        chunk_bytes = _chunk_size_bytes(dtype, head_dim)

        for global_row in range(sp):
            fabric_node_ids = [mesh_device.get_fabric_node_id(ttnn.MeshCoordinate(global_row, target_col))]
            group_idx = table.add_device_group(fabric_node_ids)
            for fid in fabric_node_ids:
                key = (int(fid.mesh_id), int(fid.chip_id))
                if key not in hosts_set:
                    table.set_fabric_node_host(fid, host_name=host_name)
                    hosts_set.add(key)

            for slot in range(num_users):
                for layer in range(num_layers):
                    batch = slot * num_layers + layer

                    for seq_chunk in range(num_chunks_per_seq_len):
                        chunk_token_start = seq_chunk * chunk_size + global_row * tokens_per_chunk_local

                        for bank_in_chunk in range(banks_per_chunk_local):
                            position = chunk_token_start + bank_in_chunk * NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK

                            # Direct shard index: [batch, n_kv_local, seq_banks_per_chip] linearized
                            seq_bank = seq_chunk * banks_per_chunk_local + bank_in_chunk
                            shard_idx = (
                                batch * (n_kv_local * seq_banks_per_chip) + target_lh * seq_banks_per_chip + seq_bank
                            )

                            bank_id = shard_idx % num_dram_banks
                            bank_offset = (shard_idx // num_dram_banks) * chunk_bytes

                            location = ttnn.experimental.disaggregation.KvCacheLocation()
                            location.noc_addr = (bank_id << 32) | (base_addr + bank_offset)
                            location.size_bytes = chunk_bytes
                            location.device_group_index = group_idx
                            table.set(layer, position, slot, location, config_id)

    logger.info(
        f"[llama-kv-table] multi-config table built "
        f"(configs={num_configs} [{', '.join(s[0] for s in specs)}], entries={table.total_entries()}, "
        f"banks={num_dram_banks}, n_kv_local={n_kv_local})"
    )
    return table


def build_and_serialize_kv_chunk_table(
    *,
    mesh_device,
    kv_cache,
    seq_len,
    num_layers,
    mesh_shape,
    sp_axis,
    num_users,
    chunk_size,
    num_kv_heads,
    head_dim,
    path,
) -> str:
    table = build_kv_chunk_address_table(
        mesh_device=mesh_device,
        kv_cache=kv_cache,
        seq_len=seq_len,
        num_layers=num_layers,
        mesh_shape=mesh_shape,
        sp_axis=sp_axis,
        num_users=num_users,
        chunk_size=chunk_size,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
    )
    ttnn.experimental.disaggregation.export_to_protobuf_file(table, path)
    logger.info(
        f"[migration] Llama KV chunk address table serialized to {path} "
        f"(configs={table.num_configs()}, entries={table.total_entries()})"
    )
    return path
