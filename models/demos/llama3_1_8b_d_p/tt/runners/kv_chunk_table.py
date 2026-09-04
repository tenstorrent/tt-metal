# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""KV chunk address table for Llama 3.1 8B GQA prefill.

Turns ``(slot, layer, token range)`` into a DRAM address for every KV head, describing exactly the
caches ``tt/attention/kv_cache.allocate_kv_cache`` allocated:

  * per-chip shape ``[num_users * num_layers, num_kv_heads_local, seq_local, head_dim]`` (user-major)
  * DRAM NdShard ``ROUND_ROBIN_1D`` over 32-token banks, shard ``[1, 1, 32, head_dim]``
  * KV heads sharded on the TP columns, sequence SP block-cyclic on the rows

Config layout (id order is the src<->dst migration contract)::

    config 0 .. H-1     -> k head 0 .. H-1
    config H .. 2H-1    -> v head 0 .. H-1

with ``H == num_kv_heads == 8``.

**Where this departs from ``gpt_oss_d_p``'s table, and why it had to.** The donor asserts
``num_kv_heads == TP columns`` — one KV head per column — so a head maps to a column and the cache's
head dim is always 1. Llama has 8 KV heads on 4 columns, so head ``h`` lives on column
``h // n_kv_local`` at cache head index ``h % n_kv_local``, and the DRAM walk has to step over the
head dim as well as the batch and sequence dims.

That also forces the walk to be written differently. The donor advances a running
``(bank_id, bank_offset)`` counter through nested loops, which is only correct because its loop
nesting happens to match the ND-shard's flattening order exactly. With a real head dim that
coincidence stops holding, so this file computes the flat shard index in closed form::

    shard_index = ((slot * num_layers + layer) * n_kv_local + h_local) * seq_blocks + seq_block
    bank_id     = shard_index % num_dram_banks
    bank_offset = (shard_index // num_dram_banks) * chunk_size_bytes

which reduces to the donor's counter when ``n_kv_local == 1``. A mistake here produces
valid-looking addresses pointing at the wrong tokens, so ``tests/test_kv_cache_table.py`` checks it
bit-exactly against bytes read back from device DRAM.

Protobuf note: ``import_from_protobuf`` rebuilds configs through a ``std::map`` (lexicographic name
order), so unpadded names ``"0".."15"`` would put ``"10"`` before ``"2"`` and silently renumber the
config ids. Names are zero-padded so map order == config_id order across export/import.
"""

from __future__ import annotations

import socket

from loguru import logger

import ttnn
from models.demos.common.prefill.runners.migration import KvCacheStage, get_num_dram_banks
from models.demos.llama3_1_8b_d_p.tt.attention.kv_cache import NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK

# TILE (32x32) byte sizes. bf8_b = 1024 mantissa + 64 exponent bytes; bf16 = 2048.
_TILE_BYTES = {ttnn.bfloat8_b: 1088, ttnn.bfloat16: 2048}


def chunk_size_bytes(dtype, head_dim: int) -> int:
    """Bytes for one ``[1, 1, 32, head_dim]`` cache chunk in TILE layout."""
    assert head_dim % 32 == 0, f"head_dim ({head_dim}) must be a multiple of the 32-wide tile"
    try:
        tile_bytes = _TILE_BYTES[dtype]
    except KeyError as e:
        raise AssertionError(f"unsupported KV cache dtype {dtype}; expected bfloat8_b or bfloat16") from e
    return (head_dim // 32) * tile_bytes


def _make_config(*, num_layers, max_seq_len, num_users, chunk_bytes):
    cfg = ttnn.experimental.disaggregation.KvChunkAddressTableConfig()
    cfg.num_layers = num_layers
    cfg.max_sequence_length = max_seq_len
    cfg.num_slots = num_users
    cfg.chunk_n_tokens = NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK
    cfg.chunk_size_bytes = chunk_bytes
    return cfg


def stable_config_name(config_id: int, num_configs: int) -> str:
    """Zero-padded decimal name so std::map lexicographic order matches numeric config_id."""
    width = max(2, len(str(max(num_configs - 1, 0))))
    return f"{config_id:0{width}d}"


def config_specs(num_kv_heads: int) -> list[tuple[str, str, int]]:
    """``(label, cache_name, global_head)`` per config, in config_id order: all K heads, then all V.

    ``"k"`` sorts before ``"v"``, matching the spec's
    ``interfaces.decode.config_names_and_order == ["k", "v"]``: config ids ``[0, H)`` are K, ``[H, 2H)``
    are V.
    """
    return [(f"k_h{h}", "k", h) for h in range(num_kv_heads)] + [(f"v_h{h}", "v", h) for h in range(num_kv_heads)]


def dram_location(
    *,
    base_addr: int,
    slot: int,
    layer: int,
    h_local: int,
    seq_block: int,
    num_layers: int,
    n_kv_local: int,
    seq_blocks: int,
    num_dram_banks: int,
    chunk_bytes: int,
) -> tuple[int, int]:
    """Replay the ND-shard ROUND_ROBIN_1D placement: -> ``(bank_id, byte_offset_within_bank)``.

    The shard is ``[1, 1, 32, head_dim]``, so shards enumerate in row-major order over
    ``(batch, head, seq_block)`` with ``batch = slot * num_layers + layer``. Blocks round-robin
    across banks, and the per-bank offset advances once per full sweep.
    """
    batch = slot * num_layers + layer
    shard_index = (batch * n_kv_local + h_local) * seq_blocks + seq_block
    bank_id = shard_index % num_dram_banks
    bank_offset = (shard_index // num_dram_banks) * chunk_bytes
    return bank_id, bank_offset


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

    ``chunk_size`` is the block-cyclic period — the tokens per ``prefill_chunk`` call. It sets where
    a global token position lands in a chip's local cache rows, so it must be the SAME value the
    runtime prefills with, or every address is off by a block.
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
    assert num_kv_heads % cols == 0, (
        f"num_kv_heads ({num_kv_heads}) must be divisible by the TP column count ({cols}); "
        f"KV-head replication is not implemented"
    )
    n_kv_local = num_kv_heads // cols
    for name, t in (("k", kv_cache.k), ("v", kv_cache.v)):
        assert (
            t.shape[0] == num_users * num_layers
        ), f"{name} cache batch dim {t.shape[0]} != num_users({num_users}) * num_layers({num_layers})"
        assert t.shape[1] == n_kv_local, f"{name} cache head dim {t.shape[1]} != num_kv_heads/cols ({n_kv_local})"

    seq_local = seq_len // sp
    seq_blocks = seq_local // NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK
    num_chunks_per_seq = seq_len // chunk_size
    specs = config_specs(num_kv_heads)
    num_configs = len(specs)

    tensors = {"k": kv_cache.k, "v": kv_cache.v}
    configs_by_name = {
        stable_config_name(i, num_configs): _make_config(
            num_layers=num_layers,
            max_seq_len=seq_len,
            num_users=num_users,
            chunk_bytes=chunk_size_bytes(tensors[cache_name].dtype, head_dim),
        )
        for i, (_, cache_name, _) in enumerate(specs)
    }
    table = ttnn.experimental.disaggregation.KvChunkAddressTable(configs_by_name)
    assert table.num_configs() == num_configs
    for i in range(num_configs):
        assert table.config_name(i) == stable_config_name(i, num_configs), (
            f"config_id {i} name {table.config_name(i)!r} != {stable_config_name(i, num_configs)!r} "
            "(protobuf-safe naming broken)"
        )

    host_name = socket.gethostname()
    hosts_set = set()

    for config_id, (label, cache_name, head) in enumerate(specs):
        tensor = tensors[cache_name]
        base_addr = tensor.buffer_address()
        chunk_bytes = chunk_size_bytes(tensor.dtype, head_dim)
        col = head // n_kv_local  # which TP column carries this head
        h_local = head % n_kv_local  # its index within that chip's head dim
        for global_row in range(sp):
            fabric_node_ids = [mesh_device.get_fabric_node_id(ttnn.MeshCoordinate(global_row, col))]
            group_idx = table.add_device_group(fabric_node_ids)
            for fid in fabric_node_ids:
                key = (int(fid.mesh_id), int(fid.chip_id))
                if key not in hosts_set:
                    table.set_fabric_node_host(fid, host_name=host_name)
                    hosts_set.add(key)

            for slot in range(num_users):
                for layer in range(num_layers):
                    for seq_chunk in range(num_chunks_per_seq):
                        # Global positions this row carries in this chunk, and the local cache rows
                        # they occupy (the block-cyclic map: local row = chunk*C + j).
                        chunk_token_start = seq_chunk * chunk_size + global_row * tokens_per_chunk_local
                        for j in range(0, tokens_per_chunk_local, NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK):
                            position = chunk_token_start + j
                            local_row = seq_chunk * tokens_per_chunk_local + j
                            bank_id, bank_offset = dram_location(
                                base_addr=base_addr,
                                slot=slot,
                                layer=layer,
                                h_local=h_local,
                                seq_block=local_row // NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK,
                                num_layers=num_layers,
                                n_kv_local=n_kv_local,
                                seq_blocks=seq_blocks,
                                num_dram_banks=num_dram_banks,
                                chunk_bytes=chunk_bytes,
                            )
                            location = ttnn.experimental.disaggregation.KvCacheLocation()
                            location.noc_addr = (bank_id << 32) | (base_addr + bank_offset)
                            location.size_bytes = chunk_bytes
                            location.device_group_index = group_idx
                            table.set(layer, position, slot, location, config_id)

    logger.info(
        f"[llama3_1_8b-kv-table] built (configs={num_configs} [{specs[0][0]}..{specs[-1][0]}], "
        f"entries={table.total_entries()}, banks={num_dram_banks}, heads/chip={n_kv_local})"
    )
    return table


def build_and_serialize_kv_chunk_table(*, path, **kwargs) -> str:
    """Build the table and serialize it to ``path`` for SET_TABLE. Returns ``path``."""
    table = build_kv_chunk_address_table(**kwargs)
    ttnn.experimental.disaggregation.export_to_protobuf_file(table, path)
    logger.info(
        f"[migration] Llama 3.1 8B KV chunk address table serialized to {path} "
        f"(configs={table.num_configs()}, entries={table.total_entries()})"
    )
    return path


def build_migration_stages(kv_cache, *, mesh_device, first_layer_idx, num_layers):
    """One ``KvCacheStage`` per migratable cache tensor, in config order (K then V).

    The engine all-gathers a layout per stage across pipeline ranks and merges them into one table.
    Llama 3.1 8B is single-stage, so this is a two-element list describing this rank's whole cache.
    """
    return [
        KvCacheStage(base_addr=int(kv_cache.k.buffer_address()), first_layer=first_layer_idx, count=num_layers),
        KvCacheStage(base_addr=int(kv_cache.v.buffer_address()), first_layer=first_layer_idx, count=num_layers),
    ]


def serialize_table(runtime, kv_cache, path: str) -> str:
    """Thin forwarder used by ``TtPrefillRuntime.build_kv_chunk_table``."""
    cfg = runtime.config
    return build_and_serialize_kv_chunk_table(
        mesh_device=runtime.mesh_device,
        kv_cache=kv_cache,
        seq_len=cfg.max_seq_len,
        num_layers=cfg.num_layers,
        mesh_shape=cfg.mesh_shape,
        sp_axis=cfg.sp_axis,
        num_users=cfg.num_users,
        chunk_size=cfg.default_chunk_size,
        num_kv_heads=runtime.hf_config.num_key_value_heads,
        head_dim=runtime.hf_config.head_dim,
        path=path,
    )
