# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""KV chunk address table builder for Mistral-Medium-3.5 dense-GQA prefill.

Adapted from ``gpt_oss_d_p/tt/runners/kv_chunk_table.py`` — the ``kv_cache`` cluster's
``address_table`` role, and the reason the cluster must come from ONE donor package: these addresses
have to describe the cache that ``tt/attention/kv_cache.allocate_kv_cache`` actually built.

The layout it walks (all of it fixed by that allocator):

  * per-chip shape ``[num_users * num_layers, 1, seq_local, head_dim]``, user-major
  * DRAM NdShard ROUND_ROBIN_1D over 32-token banks
    (``NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK``, imported from the allocator rather than re-declared —
    two copies of that constant is how a table silently stops matching its cache)
  * TP-head-sharded K/V: column ``c`` holds head ``c``, not a replica
  * SP block-cyclic sequence over the ``sp`` rows

Config layout (the id order IS the src<->dst migration contract)::

    config 0..N-1   -> k head 0..N-1   (single-member device group: that head's TP column)
    config N..2N-1  -> v head 0..N-1   (single-member device group)

``N == num_kv_heads == TP columns == 8`` here (one KV head per column).

Protobuf note, carried over verbatim because it is a real trap: ``import_from_protobuf`` rebuilds
configs through a ``std::map``, i.e. in LEXICOGRAPHIC name order. The list constructor auto-names
configs ``"0".."N-1"``, which reorders for ``N > 10`` (``"10"`` sorts before ``"2"``) and breaks
producer lookups by integer ``config_id``. With 8 KV heads this table has 16 configs, so it is
squarely in the affected range — hence the zero-padded names below.
"""

from __future__ import annotations

import socket

from loguru import logger

import ttnn
from models.demos.common.prefill.runners.migration import get_num_dram_banks
from models.demos.mistral_3_5_d_p.tt.attention.kv_cache import NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK

# bf8_b / bf16 TILE byte sizes (32x32 tile). bf8_b = 1024 mantissa + 64 exponent; bf16 = 2048.
_TILE_BYTES = {ttnn.bfloat8_b: 1088, ttnn.bfloat16: 2048}


def _chunk_size_bytes(dtype, head_dim: int) -> int:
    """Bytes for one ``[1, 1, 32, head_dim]`` chunk in the cache dtype / TILE layout."""
    assert head_dim % 32 == 0, f"head_dim ({head_dim}) must be a multiple of the 32-wide tile"
    try:
        tile_bytes = _TILE_BYTES[dtype]
    except KeyError as e:
        raise AssertionError(f"unsupported KV cache dtype {dtype}; expected bfloat8_b or bfloat16") from e
    return (head_dim // 32) * tile_bytes


def _make_config(*, num_layers, max_seq_len, num_users, chunk_size_bytes):
    cfg = ttnn.experimental.disaggregation.KvChunkAddressTableConfig()
    cfg.num_layers = num_layers
    cfg.max_sequence_length = max_seq_len
    cfg.num_slots = num_users
    cfg.chunk_n_tokens = NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK
    cfg.chunk_size_bytes = chunk_size_bytes
    return cfg


def _stable_config_name(config_id: int, num_configs: int) -> str:
    """Zero-padded decimal name, so ``std::map`` lexicographic order matches numeric config_id."""
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
    """Build the multi-config block-cyclic KV chunk address table (does not serialize).

    ``kv_cache`` is a ``MistralKVCache`` (``.k`` / ``.v``). ``seq_len`` is the ALLOCATED capacity and
    ``chunk_size`` the block-cyclic period (tokens per ``prefill_chunk``).
    """
    tp_axis = 1 - sp_axis
    sp = mesh_shape[sp_axis]
    cols = mesh_shape[tp_axis]
    num_dram_banks = get_num_dram_banks(mesh_device)

    assert seq_len % chunk_size == 0, f"seq_len {seq_len} must be a multiple of chunk_size {chunk_size}"
    tokens_per_chunk_local = chunk_size // sp
    assert tokens_per_chunk_local % NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK == 0, (
        f"chunk_size {chunk_size} / sp {sp} = {tokens_per_chunk_local}, "
        f"not a multiple of {NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK}"
    )
    # Head h lives on TP column h (the write path shards GQA heads over the columns), which needs a
    # 1:1 mapping. At the spec's TP=8 with 8 KV heads that holds exactly.
    assert num_kv_heads == cols, (
        f"this table maps head h -> TP column h, so num_key_value_heads ({num_kv_heads}) must equal "
        f"the TP column count ({cols})"
    )
    for name, tensor in (("k", kv_cache.k), ("v", kv_cache.v)):
        assert (
            tensor.shape[0] == num_users * num_layers
        ), f"{name} cache batch dim {tensor.shape[0]} != num_users({num_users}) * num_layers({num_layers})"

    num_chunks_per_seq_len = seq_len // chunk_size

    # Config layout: k_h0..k_hN-1, then v_h0..v_hN-1.
    specs = [(f"k_h{h}", kv_cache.k, [h], kv_cache.k.dtype) for h in range(num_kv_heads)]
    specs += [(f"v_h{h}", kv_cache.v, [h], kv_cache.v.dtype) for h in range(num_kv_heads)]

    # Dict ctor + zero-padded names: protobuf import uses std::map, and unpadded "0".."15" would
    # reorder config_ids (see the module docstring).
    num_configs = len(specs)
    configs_by_name = {
        _stable_config_name(i, num_configs): _make_config(
            num_layers=num_layers,
            max_seq_len=seq_len,
            num_users=num_users,
            chunk_size_bytes=_chunk_size_bytes(dtype, head_dim),
        )
        for i, (_, _, _, dtype) in enumerate(specs)
    }
    table = ttnn.experimental.disaggregation.KvChunkAddressTable(configs_by_name)
    assert table.num_configs() == num_configs
    for i in range(num_configs):
        assert table.config_name(i) == _stable_config_name(i, num_configs), (
            f"config_id {i} name {table.config_name(i)!r} != {_stable_config_name(i, num_configs)!r} "
            "(protobuf-safe naming broken)"
        )

    host_name = socket.gethostname()
    hosts_seen = set()

    for config_id, (_label, tensor, group_cols, dtype) in enumerate(specs):
        base_addr = tensor.buffer_address()
        chunk_bytes = _chunk_size_bytes(dtype, head_dim)
        for global_row in range(sp):
            fabric_node_ids = [mesh_device.get_fabric_node_id(ttnn.MeshCoordinate(global_row, c)) for c in group_cols]
            group_idx = table.add_device_group(fabric_node_ids)
            for fid in fabric_node_ids:
                key = (int(fid.mesh_id), int(fid.chip_id))
                if key not in hosts_seen:
                    table.set_fabric_node_host(fid, host_name=host_name)
                    hosts_seen.add(key)

            # Replay the ND-shard ROUND_ROBIN_1D walk: 32-token blocks round-robin across the DRAM
            # banks (per chip, per tensor), advancing the per-bank offset after each full sweep.
            # Addresses are identical on every column of a row (identical allocation); only the
            # device group's column differs per head.
            curr_bank_id = 0
            curr_bank_offset = 0
            for slot in range(num_users):
                for layer in range(num_layers):
                    for seq_chunk in range(num_chunks_per_seq_len):
                        chunk_token_start = seq_chunk * chunk_size + global_row * tokens_per_chunk_local
                        chunk_token_end = chunk_token_start + tokens_per_chunk_local
                        for position in range(chunk_token_start, chunk_token_end, NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK):
                            location = ttnn.experimental.disaggregation.KvCacheLocation()
                            location.noc_addr = (curr_bank_id << 32) | (base_addr + curr_bank_offset)
                            location.size_bytes = chunk_bytes
                            location.device_group_index = group_idx
                            table.set(layer, position, slot, location, config_id)

                            curr_bank_id = (curr_bank_id + 1) % num_dram_banks
                            if curr_bank_id == 0:
                                curr_bank_offset += chunk_bytes

    logger.info(
        f"[mistral-kv-table] multi-config table built "
        f"(configs={len(specs)} [{', '.join(s[0] for s in specs)}], entries={table.total_entries()}, "
        f"banks={num_dram_banks}, chunk_bytes={configs_by_name[_stable_config_name(0, num_configs)].chunk_size_bytes})"
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
    """Build the multi-config table and serialize it to ``path`` for SET_TABLE. Returns ``path``."""
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
        f"[migration] Mistral KV chunk address table serialized to {path} "
        f"(configs={table.num_configs()}, entries={table.total_entries()})"
    )
    return path
