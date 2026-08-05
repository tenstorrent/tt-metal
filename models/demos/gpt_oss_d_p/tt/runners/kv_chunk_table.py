# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""KV chunk address table builder for the GPT-OSS GQA prefill cache (``GptOssKVCache``).

Mirrors ``minimax_m3/tt/runners/kv_chunk_table.py`` — the on-device DRAM layout is identical (M3's
``k`` / ``v`` and our ``GptOssKVCache.k`` / ``.v`` share the same user-major ``slot*num_layers+layer``
fold, 32-token round-robin bank walk, and block-cyclic SP write). The only omissions vs M3 are:

  * NO ``index_k`` — GPT-OSS is dense GQA with no sparse lightning indexer, so the config list is
    just ``num_kv_heads`` K configs + ``num_kv_heads`` V configs (no third replicated tensor).
  * Different table builder file (this one) so future GPT-OSS-specific tweaks stay local.

The chunk-address-table utility in ``utils/kv_cache_table.py`` — cherry-picked from
``lgalasTT/gpt-oss-d-p-chunk-address-table`` — is written against a per-layer cache-tensor list
(``kv_caches[layer] == [k, v]``) allocated by ``init_kv_cache`` in
``tt/attention/kv_cache_prefill_only.py``. Our runtime's ``GptOssKVCache`` stores ONE ``k`` and ONE
``v`` tensor covering all layers via a user-major flattened leading batch dim
(``[num_users * num_layers, 1, seq_local, head_dim]``). That's the same layout M3 uses, so we
follow M3's builder verbatim (with index_k dropped) rather than the per-layer utility.

The per-chip DRAM addressing (32-token blocks round-robin across the DRAM banks, block-cyclic
positions, user-major ``slot*num_layers+layer`` fold) matches DeepSeek/M3 exactly.
"""

import socket

from loguru import logger

import ttnn
from models.demos.deepseek_v3_d_p.utils.kv_cache_utils import BH_NUM_DRAM_BANKS, NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK

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
    """Build the GPT-OSS multi-config block-cyclic KV chunk address table and serialize it to ``path``
    for the migration worker's SET_TABLE. Returns the path on success. Single-rank only.

    ``chunk_size`` is the block-cyclic period (the runtime's per-``prefill_chunk`` token count) — same
    period the KV writer (``update_padded_kv_cache``) and the indexed rope use.
    ``kv_cache`` is the ``GptOssKVCache`` (``.k`` / ``.v`` device tensors, both packed
    ``[num_users*num_layers, 1, seq_local, head_dim]``).
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
    # Head h lives on TP column h — the writer shards the GQA heads over the columns; gather_layer
    # reads head c from column c. That 1:1 mapping requires num_kv_heads == number of TP columns.
    assert num_kv_heads == cols, (
        f"GPT-OSS KV chunk table maps head h -> TP column h, so num_key_value_heads ({num_kv_heads}) "
        f"must equal the TP column count ({cols}). A different head:column ratio needs a generalized "
        f"column map."
    )
    for name, t in (("k", kv_cache.k), ("v", kv_cache.v)):
        assert (
            t.shape[0] == num_users * num_layers
        ), f"{name} cache batch dim {t.shape[0]} != num_users({num_users}) * num_layers({num_layers})"

    num_chunks_per_seq_len = seq_len // chunk_size

    # Config layout (id order is the src<->dst migration contract): k_h0..k_hN-1, v_h0..v_hN-1.
    # Each entry: (label, device tensor, TP columns forming its device group, dtype).
    specs = []
    for h in range(num_kv_heads):
        specs.append((f"k_h{h}", kv_cache.k, [h], kv_cache.k.dtype))
    for h in range(num_kv_heads):
        specs.append((f"v_h{h}", kv_cache.v, [h], kv_cache.v.dtype))

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

            # Replay the ND-shard ROUND_ROBIN_1D walk: 32-token blocks round-robin across DRAM banks
            # (per chip / per tensor), advancing per-bank offset after each full bank sweep. Same
            # arithmetic as DeepSeek's kimi builder and M3's builder — the addresses are identical on
            # every column of a row (the tensor is allocated identically everywhere); only the device
            # group's column differs.
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
        f"[migration] GPT-OSS KV chunk address table serialized to {path} "
        f"(configs={len(specs)} [{', '.join(s[0] for s in specs)}], entries={table.total_entries()})"
    )
    return path
