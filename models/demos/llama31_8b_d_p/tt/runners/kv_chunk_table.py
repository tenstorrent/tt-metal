# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""KV chunk address table for Llama-3.1-8B prefill (`llama31_8b_d_p`) — P10.4, closes ``R-030``.

The address table is the migration contract: it maps ``(layer, token position, slot, config)`` to the
DRAM ``(bank, offset)`` holding those 32 tokens of one KV head, so a reader with no device — the
migration worker, or the producer's ``read_dram_umd`` PCC check — can find the bytes prefill wrote.
``TtPrefillRuntime.build_kv_chunk_table`` calls :func:`build_and_serialize_kv_chunk_table`; the engine
publishes whatever comes back and never inspects it. **A structurally valid but wrong table is the
worst failure mode in this file** — it migrates the wrong DRAM ranges and surfaces as a corrupted
decode long after prefill, with nothing pointing here — which is why ``build_kv_chunk_table`` raised
rather than returning a placeholder for three phases (``R-030``).

What it describes — the layout ``tt/attention/kv_cache.allocate_kv_cache`` builds, gated by ``G-KV``
and ``G-KV-TP8``:

  * per-chip shape ``[num_users * num_layers, 1, seq_local, head_dim]``, user-major
    (``slot = user_id * num_layers + layer_idx``);
  * DRAM ``NdShard`` ``ROUND_ROBIN_1D``, shard row ``[1, 1, 32, head_dim]`` over
    ``mesh_device.dram_grid_size().x`` banks (measured **8** on this Blackhole galaxy);
  * K/V heads TP-sharded: column ``c`` holds head ``c``, one head per chip, *not* a replica
    (asserted bit-exactly by ``G-KV-TP8``);
  * the sequence SP-sharded **block-cyclic** over the ``sp`` rows, period ``chunk_size``.

Config layout (config-id order **is** the src<->dst migration contract, and is what the producer's
packed-GQA reader indexes by, ``prefill_producer.py:566`` / ``:573``)::

    config 0..N-1    -> k head 0..N-1   (single-member device group: that head's TP column)
    config N..2N-1   -> v head 0..N-1   (single-member device group)

``N == num_key_value_heads == the TP column count == 8``.

**Protobuf naming.** ``import_from_protobuf_file`` rebuilds configs through a ``std::map``, i.e. in
lexicographic *name* order. Auto-naming would give ``"0".."15"``, where ``"10"`` sorts before ``"2"``
and every ``config_id`` above 9 comes back pointing at a different head — a reader that looks up by
integer id then silently PCCs head 10's bytes against head 2's golden. Zero-padded names
(``"00".."15"``) make map order equal numeric order. Copied deliberately from
``models/demos/gpt_oss_d_p/tt/runners/kv_chunk_table.py:20-23``, which is where this trap is
documented; asserted below rather than trusted.

**Relationship to the gpt-oss builder.** This is the same geometry (``DEC-105``): P2 committed to
keeping gpt-oss's ``NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK = 32`` and ``[1, 1, 32, head_dim]`` shard row
precisely so the producer's existing packed-K/V reader could be reused instead of a fourth one being
written. The differences are ``head_dim`` 64 -> 128 (Appendix F.6) and that this module drives the
config population through the engine's shared ``serialize_kv_chunk_table`` helper rather than
hand-rolling it, as ``ADDING_A_PREFILL_MODEL.md`` §2 asks.
"""

from __future__ import annotations

import socket

from loguru import logger

import ttnn
from models.demos.common.prefill.runners.migration import get_num_dram_banks, serialize_kv_chunk_table
from models.demos.llama31_8b_d_p.tt.attention.kv_cache import NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK

# Bytes per 32x32 TILE. bf8_b = 1024 mantissa bytes + 64 exponent bytes; bf16 = 2048.
# The producer decodes with exactly these sizes (`_decode_bfp8_chunk` / `_decode_bf16_chunk`), so a
# wrong value here is not a slow path, it is a misaligned read.
_TILE_BYTES = {ttnn.bfloat8_b: 1088, ttnn.bfloat16: 2048}


def chunk_size_bytes(dtype, head_dim: int) -> int:
    """Bytes of one ``[1, 1, 32, head_dim]`` DRAM-bank chunk in the cache's dtype + TILE layout."""
    assert head_dim % ttnn.TILE_SIZE == 0, f"head_dim ({head_dim}) must be a multiple of the 32-wide tile"
    try:
        tile_bytes = _TILE_BYTES[dtype]
    except KeyError as exc:
        raise AssertionError(
            f"unsupported KV cache dtype {dtype}; this table can size bfloat8_b and bfloat16 chunks "
            f"only. Add it to _TILE_BYTES rather than letting the address walk stride by a guess."
        ) from exc
    return (head_dim // ttnn.TILE_SIZE) * tile_bytes


def stable_config_name(config_id: int, num_configs: int) -> str:
    """Zero-padded decimal name, so the protobuf ``std::map`` order equals numeric ``config_id``."""
    width = max(2, len(str(max(num_configs - 1, 0))))
    return f"{config_id:0{width}d}"


def _clone_config(template, name_count: int) -> list:
    """``name_count`` fresh ``KvChunkAddressTableConfig`` objects, field-copied from ``template``.

    ``serialize_kv_chunk_table`` populates exactly one config (num_layers / max_sequence_length /
    num_slots / chunk_n_tokens / chunk_size_bytes) and hands it to the builder. Llama's table needs
    one config per KV head, and every one of those five fields is identical across heads — K and V
    share a dtype, a head_dim and a slot geometry — so each head's config is a field-for-field copy
    of the helper's. Copying the five fields (rather than reusing one object 2N times, or
    re-populating them here) keeps the engine helper as the single place they are set while still
    giving the table 2N distinct objects, which is what the gpt-oss builder's dict constructor also
    passes.
    """
    out = []
    for _ in range(name_count):
        cfg = ttnn.experimental.disaggregation.KvChunkAddressTableConfig()
        cfg.num_layers = template.num_layers
        cfg.max_sequence_length = template.max_sequence_length
        cfg.num_slots = template.num_slots
        cfg.chunk_n_tokens = template.chunk_n_tokens
        cfg.chunk_size_bytes = template.chunk_size_bytes
        out.append(cfg)
    return out


def build_kv_chunk_address_table(
    *,
    config,
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
    """Build the multi-config block-cyclic KV chunk address table. Does not serialize, issues no comms.

    ``config`` is the single populated config ``serialize_kv_chunk_table`` built from the geometry;
    it is replicated per head by :func:`_clone_config`. ``kv_cache`` is a ``LlamaKVCache``
    (``.k`` / ``.v``). ``chunk_size`` is the block-cyclic period — tokens per ``prefill_chunk`` call —
    and it must be the size the cache was actually written with, because the period is what decides
    which chip holds a given global position.
    """
    tp_axis = 1 - sp_axis
    sp = mesh_shape[sp_axis]
    cols = mesh_shape[tp_axis]
    num_dram_banks = get_num_dram_banks(mesh_device)

    assert seq_len % chunk_size == 0, (
        f"seq_len {seq_len} must be a multiple of chunk_size {chunk_size}: the block-cyclic period "
        f"has to tile the cache or the last partial period's rows do not line up"
    )
    tokens_per_chunk_local = chunk_size // sp
    assert tokens_per_chunk_local % NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK == 0, (
        f"chunk_size {chunk_size} / sp {sp} = {tokens_per_chunk_local} tokens per chip per chunk, "
        f"not a multiple of the {NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK}-token DRAM bank block"
    )
    # Head h lives on TP column h — one head per chip, decided by the write path's mesh mapper
    # (G-KV-TP8 asserts it bit-exactly). Anything else makes every device group below point at the
    # wrong chip.
    assert num_kv_heads == cols, (
        f"this table maps KV head h -> TP column h, so num_key_value_heads ({num_kv_heads}) must "
        f"equal the TP column count ({cols}) of mesh_shape={tuple(mesh_shape)}. R-027."
    )
    for label, tensor in (("k", kv_cache.k), ("v", kv_cache.v)):
        assert tensor.shape[0] == num_users * num_layers, (
            f"{label} cache batch dim {tensor.shape[0]} != num_users({num_users}) * "
            f"num_layers({num_layers}); the user-major slot arithmetic below would address the "
            f"wrong rows"
        )
        assert tensor.shape[-1] == head_dim, f"{label} cache head_dim {tensor.shape[-1]} != {head_dim}"

    num_chunks_per_seq_len = seq_len // chunk_size

    # Config order IS the migration contract: k_h0..k_hN-1 then v_h0..v_hN-1.
    specs = [(f"k_h{h}", kv_cache.k, h) for h in range(num_kv_heads)]
    specs += [(f"v_h{h}", kv_cache.v, h) for h in range(num_kv_heads)]
    num_configs = len(specs)

    configs = _clone_config(config, num_configs)
    configs_by_name = {stable_config_name(i, num_configs): cfg for i, cfg in enumerate(configs)}
    table = ttnn.experimental.disaggregation.KvChunkAddressTable(configs_by_name)
    assert table.num_configs() == num_configs, f"table has {table.num_configs()} configs, expected {num_configs}"
    for i in range(num_configs):
        expected = stable_config_name(i, num_configs)
        assert table.config_name(i) == expected, (
            f"config_id {i} came back named {table.config_name(i)!r}, expected {expected!r}: the "
            f"protobuf-safe zero-padded naming is broken and config ids will not survive export"
        )

    host_name = socket.gethostname()
    hosts_set = set()
    # All 2N configs share one chunk_size_bytes (they are field-copies of the single config the
    # engine helper populated), which is only sound while K and V have the same dtype. They always
    # do -- `allocate_kv_cache` builds both from one `cache_dtype` -- but a divergence would size
    # half the table's chunks wrong, and a wrongly-sized chunk is a misaligned read, not an error.
    assert kv_cache.k.dtype == kv_cache.v.dtype, (
        f"K cache dtype {kv_cache.k.dtype} != V cache dtype {kv_cache.v.dtype}; this table gives "
        f"every config one chunk_size_bytes, so split dtypes need per-config sizing first"
    )
    expected_bytes = chunk_size_bytes(kv_cache.k.dtype, head_dim)
    assert config.chunk_size_bytes == expected_bytes, (
        f"the caller sized a chunk at {config.chunk_size_bytes} B but this cache's "
        f"{kv_cache.k.dtype} / head_dim {head_dim} needs {expected_bytes} B"
    )

    for config_id, (_label, tensor, column) in enumerate(specs):
        base_addr = tensor.buffer_address()
        for global_row in range(sp):
            # Single-member device group: head `column` lives on exactly this chip.
            fabric_node_ids = [mesh_device.get_fabric_node_id(ttnn.MeshCoordinate(global_row, column))]
            group_idx = table.add_device_group(fabric_node_ids)
            for fid in fabric_node_ids:
                key = (int(fid.mesh_id), int(fid.chip_id))
                if key not in hosts_set:
                    table.set_fabric_node_host(fid, host_name=host_name)
                    hosts_set.add(key)

            # Replay the ND-shard ROUND_ROBIN_1D allocation: consecutive 32-token blocks of this
            # chip's buffer go round-robin across the DRAM banks, and the per-bank offset advances
            # after each full sweep. The iteration order (slot, layer, seq_chunk, offset) is exactly
            # the buffer's own row order, and the position it maps to is the inverse of
            # update_padded_kv_cache's block-cyclic write -- local row lr on chip c holds global
            # position (lr // chunk_local) * chunk_size + c * chunk_local + (lr % chunk_local),
            # which is `blockcyclic_positions` (proved an exact inverse host-side by G-KV).
            # Addresses are identical on every column of a row (identical allocations); only the
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
                            location.size_bytes = expected_bytes
                            location.device_group_index = group_idx
                            table.set(layer, position, slot, location, config_id)

                            curr_bank_id = (curr_bank_id + 1) % num_dram_banks
                            if curr_bank_id == 0:
                                curr_bank_offset += expected_bytes

    logger.info(
        f"[llama31_8b_d_p-kv-table] built: configs={num_configs} "
        f"(k_h0..k_h{num_kv_heads - 1}, v_h0..v_h{num_kv_heads - 1}), entries={table.total_entries()}, "
        f"banks={num_dram_banks}, chunk_bytes={expected_bytes}, sp={sp} cols={cols} "
        f"chunk_size={chunk_size} seq_len={seq_len} layers={num_layers} users={num_users}"
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
    """Build the table and serialize it to ``path`` for SET_TABLE. Returns ``path``.

    The config population and the protobuf write are both the engine's
    ``serialize_kv_chunk_table`` (``common/prefill/runners/migration.py:220``); this module supplies
    only the table builder and the chunk geometry, which is what ``ADDING_A_PREFILL_MODEL.md`` §2
    asks a model to own.
    """

    def _builder(*, config, chunk_size_bytes, num_users):  # noqa: ARG001 - engine-fixed signature
        return build_kv_chunk_address_table(
            config=config,
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

    return serialize_kv_chunk_table(
        table_builder=_builder,
        num_layers=num_layers,
        max_seq_len=seq_len,
        num_users=num_users,
        chunk_n_tokens=NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK,
        chunk_size_bytes=chunk_size_bytes(kv_cache.k.dtype, head_dim),
        path=path,
    )
