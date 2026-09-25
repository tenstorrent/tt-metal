# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""KV chunk address table for Gemma-4 (two KV geometries), consumed by the tt-d-gen KV Manager.

Caches (``tt/attention/kv_cache.py``), one per layer type, per chip
``[num_users * n_type_layers, n_kv_local, seq_local, head_dim]``, bf8, DRAM ND-shard ``[1,1,32,D]``
ROUND_ROBIN_1D over the DRAM banks; sequence block-cyclic over SP rows; KV heads over TP cols
(sliding: 8 heads split over TP; full: 2 heads, split if TP <= 2 else the GQA head replicated per col).

Configs (id order is the prefill<->decode contract; names zero-padded so protobuf's std::map order
== config id)::

    sliding K h0..h7, sliding V h0..h7, full K h0..h1, full V h0..h1     (20 configs)

Every config spans ``num_layers = 30`` GLOBAL layer ids but is populated only on the layers of its
type (tt-d-gen skips a config for layers it does not cover). ``chunk_size_bytes`` differs per
geometry (sliding D=256: 8 tiles = 8704 B; full D=512: 17408 B). The device group of a (head, SP row)
is every TP column holding that head (first = designated reader).

Shard address (closed form of ROUND_ROBIN_1D): shard index
``i = ((slot * L_type + type_layer) * n_kv_local + h_local) * blocks_local + blk`` lives in bank
``i % nbanks`` at offset ``base + (i // nbanks) * shard_bytes``, where ``blk`` walks the chip's
block-cyclic rows: global position ``seq_chunk * C + row * C_local + 32 * j`` -> ``blk = seq_chunk * C_local/32 + j``.
"""

from __future__ import annotations

import socket

from loguru import logger

import ttnn
from models.demos.common.prefill.runners.migration import get_num_dram_banks
from models.demos.gemma4_26b_d_p.reference.config import FULL, SLIDING
from models.demos.gemma4_26b_d_p.tt.attention.attention import kv_heads_for_col
from models.demos.gemma4_26b_d_p.tt.attention.kv_cache import NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK

_TILE_BYTES = {ttnn.bfloat8_b: 1088, ttnn.bfloat16: 2048}
BLK = NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK


def shard_bytes(dtype, head_dim: int) -> int:
    return (head_dim // 32) * _TILE_BYTES[dtype]


def config_specs(cfg, tp: int):
    """[(name, layer_type, 'k'|'v', global_kv_head)] in config-id order."""
    specs = []
    for t in (SLIDING, FULL):
        li = cfg.layer_types.index(t)
        n_kv = cfg.layer_kv_heads(li)
        for kv in ("k", "v"):
            for g in range(n_kv):
                specs.append((f"{t[:4]}_{kv}_h{g}", t, kv, g))
    return specs


def _name(i, n):
    return f"{i:0{max(2, len(str(n - 1)))}d}"


def build_kv_chunk_address_table(*, mesh_device, cfg, caches: dict, cache_layer: dict, seq_len: int, chunk_size: int, num_users: int):
    """``caches``: {SLIDING|FULL: Gemma4KVCache}; ``cache_layer``: global layer id -> index within its type's cache."""
    sp, tp = tuple(mesh_device.shape)
    C_local = chunk_size // sp
    assert seq_len % chunk_size == 0 and C_local % BLK == 0
    nbanks = get_num_dram_banks(mesh_device)
    blocks_local = (seq_len // sp) // BLK
    n_layers = cfg.num_hidden_layers
    specs = config_specs(cfg, tp)
    n_cfg = len(specs)

    table_cfgs = {}
    for i, (_, t, _, _) in enumerate(specs):
        c = ttnn.experimental.disaggregation.KvChunkAddressTableConfig()
        c.num_layers = n_layers
        c.max_sequence_length = seq_len
        c.num_slots = num_users
        c.chunk_n_tokens = BLK
        c.chunk_size_bytes = shard_bytes(caches[t].k.dtype, caches[t].head_dim)
        table_cfgs[_name(i, n_cfg)] = c
    table = ttnn.experimental.disaggregation.KvChunkAddressTable(table_cfgs)
    for i in range(n_cfg):
        assert table.config_name(i) == _name(i, n_cfg)

    host = socket.gethostname()
    hosts = set()
    layers_of = {t: [l for l in range(n_layers) if cfg.layer_types[l] == t and l in cache_layer] for t in (SLIDING, FULL)}

    for cid, (label, t, kv, g) in enumerate(specs):
        cache = caches[t]
        tensor = cache.k if kv == "k" else cache.v
        base = tensor.buffer_address()
        sb = shard_bytes(tensor.dtype, cache.head_dim)
        L_type = cache.num_layers
        n_kv = cfg.layer_kv_heads(cfg.layer_types.index(t))
        cols = [c for c in range(tp) if g in kv_heads_for_col(c, tp, cfg.num_attention_heads, n_kv)]
        h_local = kv_heads_for_col(cols[0], tp, cfg.num_attention_heads, n_kv).index(g)
        for row in range(sp):
            fids = [mesh_device.get_fabric_node_id(ttnn.MeshCoordinate(row, c)) for c in cols]
            gidx = table.add_device_group(fids)
            for f in fids:
                key = (int(f.mesh_id), int(f.chip_id))
                if key not in hosts:
                    table.set_fabric_node_host(f, host_name=host)
                    hosts.add(key)
            for slot in range(num_users):
                for gl in layers_of[t]:
                    tl = cache_layer[gl]
                    first = ((slot * L_type + tl) * cache.n_kv_local + h_local) * blocks_local
                    for seq_chunk in range(seq_len // chunk_size):
                        for j in range(C_local // BLK):
                            blk = seq_chunk * (C_local // BLK) + j
                            idx = first + blk
                            loc = ttnn.experimental.disaggregation.KvCacheLocation()
                            loc.noc_addr = ((idx % nbanks) << 32) | (base + (idx // nbanks) * sb)
                            loc.size_bytes = sb
                            loc.device_group_index = gidx
                            pos = seq_chunk * chunk_size + row * C_local + j * BLK
                            table.set(gl, pos, slot, loc, cid)
    logger.info(f"[gemma4-kv-table] {n_cfg} configs [{', '.join(s[0] for s in specs)}], entries={table.total_entries()}, banks={nbanks}")
    return table


def build_and_serialize_kv_chunk_table(*, path: str, **kw) -> str:
    table = build_kv_chunk_address_table(**kw)
    ttnn.experimental.disaggregation.export_to_protobuf_file(table, path)
    logger.info(f"[migration] Gemma-4 KV chunk table -> {path} (configs={table.num_configs()}, entries={table.total_entries()})")
    return path
