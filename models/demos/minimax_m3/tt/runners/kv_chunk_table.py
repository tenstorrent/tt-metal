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

Configs are named by zero-padded position ("00".."08"), the names blaze's decode table uses: the KV manager
pairs the two tables by config name and id, so both must match. The worker copies raw bytes and the decode peer
stores index_k in bf8, so migration requires a bf8 index_k (``M3_INDEX_CACHE_BF16`` unset).

With index_k TP-deduped (``kv_cache.index_k_tp_axis``, ``M3_INDEX_K_TP_SHARD=1``) config 2N stays the one
index_k config, but each chip holds a DISTINCT stripe of its SP row's slab, so its entries use single-chip
groups: chip (row, col) owns tokens ``[n*chunk + row*chunk_local + col*chunk_local/tp, +chunk_local/tp)``, and
walks the DRAM banks over its own rows only (the same layout as DeepSeek's ``tp_axis`` table). The config
list and order are unchanged, so the destination side keeps its own layout (a replicated destination still
receives every stripe on all its columns).

The per-chip DRAM addressing (32-token blocks round-robin across the DRAM banks, block-cyclic positions,
user-major ``slot*num_layers+layer`` fold) matches DeepSeek's ``create_kv_chunk_address_table_block_cyclic``, just
repeated per config with each tensor's own ``buffer_address()`` / ``chunk_size_bytes`` and column set.

Every config is STRIDED_ROWS: one ``set_strided_row`` per (slot, layer), each residue of the row's period an
affine address on one device group, so a 1M-token table never materializes its per-chunk grid.
"""

import math

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


def config_name(config_id: int, num_configs: int) -> str:
    """Zero-padded positional config name, as blaze's ``kv_chunk_migration_helpers`` names its decode configs
    (at least two digits). Sorted names keep ids in position order."""
    width = max(2, len(str(num_configs - 1)))
    return f"{config_id:0{width}d}"


def _page_noc_addr(page: int, *, base_addr: int, num_banks: int, chunk_bytes: int) -> int:
    """NoC address of a chip's ``page``-th 32-token block in its ROUND_ROBIN_1D bank walk."""
    return ((page % num_banks) << 32) | (base_addr + (page // num_banks) * chunk_bytes)


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
    stage_layouts=None,
    index_k_layers=None,
) -> str:
    """Build the M3 multi-config block-cyclic KV chunk address table and serialize it to ``path`` for the
    inference server's SET_TABLE. Returns the path on success.

    ``chunk_size`` is the block-cyclic period (the runtime's per-``prefill_chunk`` token count), which the
    KV writer / the indexed rope use as the period — NOT a hardcoded constant (unlike the kimi builder).
    ``kv_cache`` is the ``MiniMaxKVCache`` (``.k`` / ``.v`` / ``.index_k`` device tensors); ``num_layers``
    is THIS rank's layer count (== the global total single-rank).

    ``stage_layouts`` (one gathered layout per ``kv_migration_stages`` entry — k, v, index_k) selects the
    pipeline-parallel merge: one table spanning every stage's layers, each stage's chunks placed on ITS
    mesh at ITS cache's base address, written at the GLOBAL layer index. With stage_layouts None the
    table covers this rank's mesh alone via local ``buffer_address()`` / fabric-node lookups — the
    single-rank path.

    ``index_k_layers`` (global layer ids that carry an index_k, i.e. the MSA layers) limits the index_k
    config's rows to those layers. The cache allocates index_k for every layer but dense layers never write
    it, and the decode peer has no index_k on dense layers; a row published on one side only makes the KV
    Manager reject the whole migration ("covered by only one side"). None publishes every layer.
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
    # Only THIS rank's cache is inspectable; in a pipeline every stage allocates the same way (same
    # model/env), so its dtypes and the slot fold generalize to the gathered stages.
    # The migration worker copies raw chunk bytes and blaze's decode index_k is bfp8 (DsaIndexerDirectKCacheUpdate
    # only supports bfp8 tiles): a bf16 source would overrun its rows.
    if kv_cache.index_k.dtype != ttnn.bfloat8_b:
        raise ValueError(
            f"KV migration needs a bfloat8_b index_k (the decode peer's dtype), got {kv_cache.index_k.dtype}; "
            "unset M3_INDEX_CACHE_BF16 for migrating runs"
        )
    index_k = kv_cache.index_k
    for name, t in (("k", kv_cache.k), ("v", kv_cache.v), ("index_k", index_k)):
        assert (
            t.shape[0] == num_users * num_layers
        ), f"{name} cache batch dim {t.shape[0]} != num_users({num_users}) * num_layers({num_layers})"

    num_chunks_per_seq_len = seq_len // chunk_size

    # Config layout (id order is the src<->dst migration contract): k_h0..k_hN-1, v_h0..v_hN-1, index_k.
    # Each entry: (label, cache index into stage_layouts, device groups per SP row as lists of TP columns,
    # dtype). A TP-replicated tensor has one group (its replicas); a TP-deduped index_k has one single-chip
    # group per column, each owning its 1/tp stripe of the row's slab.
    index_k_tp_sharded = getattr(kv_cache, "index_k_tp_axis", None) is not None
    specs = []
    for h in range(num_kv_heads):
        specs.append((f"k_h{h}", 0, [[h]], kv_cache.k.dtype))
    for h in range(num_kv_heads):
        specs.append((f"v_h{h}", 1, [[h]], kv_cache.v.dtype))
    index_k_groups = [[c] for c in range(cols)] if index_k_tp_sharded else [list(range(cols))]
    specs.append(("index_k", 2, index_k_groups, index_k.dtype))
    if index_k_tp_sharded:
        assert tokens_per_chunk_local % (cols * NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK) == 0, (
            f"TP-deduped index_k needs chunk_local {tokens_per_chunk_local} split into {cols} stripes of whole "
            f"{NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK}-token chunks"
        )

    if stage_layouts is None:
        from models.demos.common.prefill.runners.migration import _host_tag_int

        # Single-rank: synthesize one one-stage layout per cache from the local mesh so both paths
        # share the build loop below. base_addr identical on every chip of the mesh (each cache is
        # allocated mesh-wide).
        local_fnids = [
            [mesh_device.get_fabric_node_id(ttnn.MeshCoordinate(r, c)) for c in range(cols)] for r in range(sp)
        ]
        stage_layouts = [
            [
                {
                    "first_layer": 0,
                    "count": num_layers,
                    "base_addr": int(t.buffer_address()),
                    "num_banks": BH_NUM_DRAM_BANKS,
                    "host_tag": _host_tag_int(),
                    "fnids": local_fnids,
                }
            ]
            for t in (kv_cache.k, kv_cache.v, index_k)
        ]

    from models.demos.common.prefill.runners.migration import validate_stage_layout_contiguous

    assert len(stage_layouts) == 3, f"expected one gathered layout per cache (k, v, index_k), got {len(stage_layouts)}"
    total_layers = validate_stage_layout_contiguous(stage_layouts[0])
    for cache_name, layout in zip(("v", "index_k"), stage_layouts[1:]):
        # M3's three caches share one layer-index space (index_k allocates a slot for every layer, zeros
        # on dense ones), so every cache must gather the identical per-rank ranges.
        if (
            len(layout) != len(stage_layouts[0])
            or validate_stage_layout_contiguous(layout) != total_layers
            or any(
                (a["first_layer"], a["count"]) != (b["first_layer"], b["count"])
                for a, b in zip(stage_layouts[0], layout)
            )
        ):
            raise RuntimeError(
                f"{cache_name} layout's layer ranges differ from k's; M3's caches share one layer-index "
                f"space, so kv_migration_stages must report the same range for all three."
            )

    disagg = ttnn.experimental.disaggregation
    names = [config_name(i, len(specs)) for i in range(len(specs))]
    table = disagg.KvChunkAddressTable(
        {
            name: _make_config(
                num_layers=total_layers,
                max_seq_len=seq_len,
                num_users=num_users,
                chunk_size_bytes=_chunk_size_bytes(dtype, head_dim),
            )
            for name, (_, _, _, dtype) in zip(names, specs)
        },
        {name: disagg.ChunkCompression.STRIDED_ROWS for name in names},
    )

    page_tokens = NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK
    slab_pages = chunk_size // page_tokens
    row_pages = tokens_per_chunk_local // page_tokens
    hosts_set = set()

    for config_id, (label, tensor_idx, row_groups, dtype) in enumerate(specs):
        chunk_bytes = _chunk_size_bytes(dtype, head_dim)
        # Pages one group (a whole SP row, or one TP stripe of it) owns per slab.
        group_pages = row_pages // len(row_groups)
        filter_layers = label == "index_k" and index_k_layers is not None
        for stage in stage_layouts[tensor_idx]:
            base_addr = stage["base_addr"]
            num_banks = stage["num_banks"]
            # Both gathered and local layouts use the same crc32 identity as the KV manager.
            host_name = stage.get("host_name") or f"host-{stage['host_tag']:08x}"
            groups = {}
            for global_row in range(sp):
                for stripe, group_cols in enumerate(row_groups):
                    fabric_node_ids = [stage["fnids"][global_row][c] for c in group_cols]
                    groups[global_row, stripe] = table.add_device_group(fabric_node_ids)
                    for fid in fabric_node_ids:
                        key = (int(fid.mesh_id), int(fid.chip_id))
                        if key not in hosts_set:
                            table.set_fabric_node_host(fid, host_name=host_name)
                            hosts_set.add(key)

            # A group's chip places its 32-token pages round-robin over its banks in slot -> local layer ->
            # slab -> page order. The row period spans enough slabs for every residue to advance by whole
            # bank sweeps, so each residue's address is affine in the chunk index.
            period_slabs = min(num_banks // math.gcd(group_pages, num_banks), num_chunks_per_seq_len)
            step = slab_pages * period_slabs
            stride = period_slabs * group_pages // num_banks * chunk_bytes
            residue_groups, residue_pages = [], []
            for chunk in range(step):
                slab, in_slab = divmod(chunk, slab_pages)
                row, in_row = divmod(in_slab, row_pages)
                stripe, page = divmod(in_row, group_pages)
                residue_groups.append(groups[row, stripe])
                residue_pages.append(slab * group_pages + page)

            for slot in range(num_users):
                for local_layer in range(stage["count"]):
                    global_layer = stage["first_layer"] + local_layer
                    if filter_layers and global_layer not in index_k_layers:
                        continue
                    walk_start = (slot * stage["count"] + local_layer) * num_chunks_per_seq_len * group_pages
                    table.set_strided_row(
                        global_layer,
                        slot,
                        chunk_bytes,
                        [
                            _page_noc_addr(
                                walk_start + p, base_addr=base_addr, num_banks=num_banks, chunk_bytes=chunk_bytes
                            )
                            for p in residue_pages
                        ],
                        [stride] * step,
                        residue_groups,
                        config_id,
                    )

    disagg.export_to_protobuf_file(table, path)
    logger.info(
        f"[migration] M3 KV chunk address table serialized to {path} "
        f"(configs={len(specs)} [{', '.join(s[0] for s in specs)}], stages={len(stage_layouts[0])}, "
        f"layers={total_layers}, entries={table.total_entries()}, index_k="
        f"{'TP-deduped (single-chip groups)' if index_k_tp_sharded else f'TP-replicated ({cols}-chip groups)'})"
    )
    return path
