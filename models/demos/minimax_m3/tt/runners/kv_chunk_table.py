# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""KV chunk address table builder for the MiniMax-M3 GQA prefill cache.

Describes the on-device KV layout as a ``KvChunkAddressTable`` so the migration worker can copy the right
chunks. ``TtPrefillRuntime.build_kv_chunk_table`` calls this; the runner publishes the serialized table.

The worker treats a chunk's device group as REPLICAS (reads one member, writes the same bytes to all
destinations). M3's ``k`` / ``v`` are TP-head-sharded — column ``c`` holds a different head, not a replica —
so each (tensor, head) needs its OWN config with a single-member device group; ``index_k`` is TP-replicated
(like DeepSeek's kvpe) and uses a full-row replica group. Hence a multi-config table:

    config 0..N-1   -> k head 0..N-1   (single-member group: the head's column, per SP row)
    config N..2N-1  -> v head 0..N-1   (single-member group)
    config 2N       -> index_k         (replica group: all TP columns of the SP row)

The per-chip DRAM addressing (32-token blocks round-robin across the DRAM banks, block-cyclic positions,
user-major ``slot*num_layers+layer`` fold) matches DeepSeek's ``create_kv_chunk_address_table_kimi``, just
repeated per config with each tensor's own ``buffer_address()`` / ``chunk_size_bytes`` and column set.
"""

import socket

from loguru import logger

import ttnn
from models.demos.minimax_m3.tt.attention.kv_cache import BH_NUM_DRAM_BANKS, NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK

# bf8_b / bf16 TILE byte sizes (32x32 tile). bf8_b = 1024 mantissa + 64 exponent bytes; bf16 = 2048.
_TILE_BYTES = {ttnn.bfloat8_b: 1088, ttnn.bfloat16: 2048}


def _chunk_size_bytes(dtype, head_dim: int) -> int:
    """Bytes for one ``[1, 1, 32, head_dim]`` chunk in the cache's dtype/TILE layout — ``head_dim/32``
    tiles wide, one 32-token tile tall. Matches the migration read size and the producer's bfp8 decode."""
    assert head_dim % 32 == 0, f"head_dim ({head_dim}) must be a multiple of the 32-wide tile"
    try:
        tile_bytes = _TILE_BYTES[dtype]
    except KeyError:
        raise AssertionError(f"unsupported KV cache dtype {dtype}; expected bfloat8_b or bfloat16")
    return (head_dim // 32) * tile_bytes


def _make_config(*, num_layers, max_seq_len, num_users, chunk_size_bytes):
    cfg = ttnn.experimental.disaggregation.KvChunkAddressTableConfig()
    cfg.num_layers = num_layers
    cfg.max_sequence_length = max_seq_len
    cfg.num_slots = num_users
    cfg.chunk_n_tokens = NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK
    cfg.chunk_size_bytes = chunk_size_bytes
    return cfg


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
    """Build the M3 multi-config block-cyclic KV chunk address table and serialize it to ``path`` for the
    inference server's SET_TABLE. Returns the path on success.

    ``chunk_size`` is the block-cyclic period (the runtime's per-``prefill_chunk`` token count), which the
    KV writer / the indexed rope use as the period — NOT a hardcoded constant (unlike the kimi builder).
    ``kv_cache`` is the ``MiniMaxKVCache`` (``.k`` / ``.v`` / ``.index_k`` device tensors). Single-rank only.
    """
    tp_axis = 1 - sp_axis
    sp = mesh_shape[sp_axis]
    cols = mesh_shape[tp_axis]

    assert seq_len % chunk_size == 0, f"seq_len {seq_len} must be a multiple of chunk_size {chunk_size}"
    tokens_per_chunk_local = chunk_size // sp
    assert tokens_per_chunk_local % NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK == 0, (
        f"chunk_size {chunk_size} / sp {sp} = {tokens_per_chunk_local}, "
        f"not a multiple of {NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK}"
    )
    # Head h lives on TP column h (the write path shards the GQA heads over the columns; gather_layer
    # reads head c from column c). That 1:1 mapping requires num_kv_heads == number of TP columns.
    assert num_kv_heads == cols, (
        f"M3 KV chunk table maps head h -> TP column h, so num_key_value_heads ({num_kv_heads}) must equal "
        f"the TP column count ({cols}). A different head:column ratio needs a generalized column map."
    )
    for name, t in (("k", kv_cache.k), ("v", kv_cache.v), ("index_k", kv_cache.index_k)):
        assert (
            t.shape[0] == num_users * num_layers
        ), f"{name} cache batch dim {t.shape[0]} != num_users({num_users}) * num_layers({num_layers})"

    num_chunks_per_seq_len = seq_len // chunk_size

    # Config layout (id order is the src<->dst migration contract): k_h0..k_hN-1, v_h0..v_hN-1, index_k.
    # Each entry: (label, device tensor, TP columns forming its device group, dtype).
    specs = []
    for h in range(num_kv_heads):
        specs.append((f"k_h{h}", kv_cache.k, [h], kv_cache.k.dtype))
    for h in range(num_kv_heads):
        specs.append((f"v_h{h}", kv_cache.v, [h], kv_cache.v.dtype))
    specs.append(("index_k", kv_cache.index_k, list(range(cols)), kv_cache.index_k.dtype))

    configs = [
        _make_config(
            num_layers=num_layers,
            max_seq_len=seq_len,
            num_users=num_users,
            chunk_size_bytes=_chunk_size_bytes(dtype, head_dim),
        )
        for (_, _, _, dtype) in specs
    ]
    table = ttnn.experimental.disaggregation.KvChunkAddressTable(configs)

    host_name = socket.gethostname()
    hosts_set = set()

    for config_id, (label, tensor, group_cols, dtype) in enumerate(specs):
        base_addr = tensor.buffer_address()
        chunk_bytes = _chunk_size_bytes(dtype, head_dim)
        for global_row in range(sp):
            fabric_node_ids = [mesh_device.get_fabric_node_id(ttnn.MeshCoordinate(global_row, c)) for c in group_cols]
            group_idx = table.add_device_group(fabric_node_ids)
            for fid in fabric_node_ids:
                key = (int(fid.mesh_id), int(fid.chip_id))
                if key not in hosts_set:
                    table.set_fabric_node_host(fid, host_name=host_name)
                    hosts_set.add(key)

            # Replay the ND-shard ROUND_ROBIN_1D walk: 32-token blocks round-robin across the DRAM banks
            # (per chip / per tensor), advancing the per-bank offset after each full bank sweep. Same
            # arithmetic as DeepSeek's kimi builder — the addresses are identical on every column of a row
            # (the tensor is allocated identically everywhere); only the device group's column differs.
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

                            curr_bank_id = (curr_bank_id + 1) % BH_NUM_DRAM_BANKS
                            if curr_bank_id == 0:
                                curr_bank_offset += chunk_bytes

    ttnn.experimental.disaggregation.export_to_protobuf_file(table, path)
    logger.info(
        f"[migration] M3 KV chunk address table serialized to {path} "
        f"(configs={len(specs)} [{', '.join(s[0] for s in specs)}], entries={table.total_entries()})"
    )
    return path
