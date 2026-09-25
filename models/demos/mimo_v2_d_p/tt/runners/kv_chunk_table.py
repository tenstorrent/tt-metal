# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""KV chunk address table for MiMo-V2 (two KV geometries), consumed by the tt-d-gen KV Manager.

Caches (``tt/attention/kv_cache.py``), one per attention type, per chip
``[num_users * n_type_layers, n_kv_local, seq_local, D]``, bf8, DRAM ND-shard ``[1,1,32,D]`` ROUND_ROBIN_1D
over the DRAM banks; sequence block-cyclic over SP rows; KV heads over TP cols (GA 4 heads, SWA 8; split
if TP <= n_kv else the GQA head replicated per col).

Configs (id order is the prefill<->decode contract; zero-padded names so protobuf's map order == id)::

    GA K h0..h3, GA V h0..h3, SWA K h0..h7, SWA V h0..h7        (24 configs)

Every config spans all 48 GLOBAL layer ids but is populated only on this rank's layers of its type.
``chunk_size_bytes`` per geometry: K D=192 (6 bf8 tiles, 6528 B); V D=128 (4 tiles, 4352 B) on both types.
K is stored rope-rotated in the Meta interleaved order of the first 64 dims (``rope.rope_perm``).

Shard address (closed form of ROUND_ROBIN_1D): shard ``i = ((slot * L_type + type_layer) * n_kv_local + h_local)
* blocks_local + blk`` lives in bank ``i % nbanks`` at ``base + (i // nbanks) * shard_bytes``, where ``blk``
walks the chip's block-cyclic rows: global position ``seq_chunk * C + row * C_local + 32 * j`` ->
``blk = seq_chunk * C_local/32 + j``.
"""

from __future__ import annotations

import socket

from loguru import logger

import ttnn
from models.demos.common.prefill.runners.migration import get_num_dram_banks
from models.demos.mimo_v2_d_p.reference.config import GA, SWA
from models.demos.mimo_v2_d_p.tt.attention.attention import kv_heads_for_col
from models.demos.mimo_v2_d_p.tt.attention.kv_cache import NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK

_TILE_BYTES = {ttnn.bfloat8_b: 1088, ttnn.bfloat16: 2048}
BLK = NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK


def shard_bytes(dtype, head_dim: int) -> int:
    return (head_dim // 32) * _TILE_BYTES[dtype]


def config_specs(cfg):
    """[(name, attn_type, 'k'|'v', global_kv_head)] in config-id order."""
    return [(f"{t[:4]}_{kv}_h{g}", t, kv, g) for t in (GA, SWA) for kv in ("k", "v") for g in range(cfg.attn_spec(t).n_kv)]


def _name(i, n):
    return f"{i:0{max(2, len(str(n - 1)))}d}"


def build_kv_chunk_address_table(*, mesh_device, cfg, caches: dict, cache_layer: dict, seq_len: int, chunk_size: int, num_users: int):
    sp, tp = tuple(mesh_device.shape)
    C_local = chunk_size // sp
    assert seq_len % chunk_size == 0 and C_local % BLK == 0
    nbanks = get_num_dram_banks(mesh_device)
    blocks_local = (seq_len // sp) // BLK
    n_layers = cfg.num_hidden_layers
    specs = [s for s in config_specs(cfg) if s[1] in caches]
    n_cfg = len(specs)

    table_cfgs = {}
    for i, (_, t, kv, _) in enumerate(specs):
        cache = caches[t]
        c = ttnn.experimental.disaggregation.KvChunkAddressTableConfig()
        c.num_layers = n_layers
        c.max_sequence_length = seq_len
        c.num_slots = num_users
        c.chunk_n_tokens = BLK
        c.chunk_size_bytes = shard_bytes(cache.k.dtype, cache.k_dim if kv == "k" else cache.v_dim)
        table_cfgs[_name(i, n_cfg)] = c
    table = ttnn.experimental.disaggregation.KvChunkAddressTable(table_cfgs)

    host = socket.gethostname()
    hosts = set()
    layers_of = {t: [l for l in range(n_layers) if cfg.layer_type(l) == t and l in cache_layer] for t in (GA, SWA)}

    for cid, (_, t, kv, g) in enumerate(specs):
        cache = caches[t]
        tensor = cache.k if kv == "k" else cache.v
        base = tensor.buffer_address()
        sb = shard_bytes(tensor.dtype, cache.k_dim if kv == "k" else cache.v_dim)
        spec = cfg.attn_spec(t)
        cols = [c for c in range(tp) if g in kv_heads_for_col(c, tp, spec.n_q, spec.n_kv)]
        h_local = kv_heads_for_col(cols[0], tp, spec.n_q, spec.n_kv).index(g)
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
                    first = ((slot * cache.num_layers + cache_layer[gl]) * cache.n_kv_local + h_local) * blocks_local
                    for seq_chunk in range(seq_len // chunk_size):
                        for j in range(C_local // BLK):
                            idx = first + seq_chunk * (C_local // BLK) + j
                            loc = ttnn.experimental.disaggregation.KvCacheLocation()
                            loc.noc_addr = ((idx % nbanks) << 32) | (base + (idx // nbanks) * sb)
                            loc.size_bytes = sb
                            loc.device_group_index = gidx
                            table.set(gl, seq_chunk * chunk_size + row * C_local + j * BLK, slot, loc, cid)
    logger.info(f"[mimo-kv-table] {n_cfg} configs, entries={table.total_entries()}, banks={nbanks}")
    return table


def build_and_serialize_kv_chunk_table(*, path: str, **kw) -> str:
    table = build_kv_chunk_address_table(**kw)
    ttnn.experimental.disaggregation.export_to_protobuf_file(table, path)
    logger.info(f"[migration] MiMo-V2 KV chunk table -> {path} (configs={table.num_configs()}, entries={table.total_entries()})")
    return path
