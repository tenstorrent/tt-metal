# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""The KV address table of the served cache (serving_contract.md, "Adapter and table") and its read-back.

One table entry is one 32-token record of one (slot, layer, config) in ONE DRAM bank of one chip. The cache
(tt/kv_cache.py TtKVCache) is ND-sharded [1, 1, 32, head_dim] ROUND_ROBIN_1D per chip, so shard j of a chip's
[batch, 1, max_seq / sp, head_dim] tensor sits in bank j % banks at buffer_address + (j // banks) * record bytes, with
j = batch * (max_seq / sp / 32) + local_row / 32. Global position p is on SP row (p / (chunk / sp)) % sp at local row
(p // chunk) * (chunk / sp) + p % (chunk / sp) (the engine's block-cyclic spread); KV head h is on TP column h.

Configs k_h0..k_h{n-1}, v_h0..v_h{n-1}: config c < n is K head c, else V head c - n. Table layers are served layers
(table layer i = served layer first + i); only attention layers have records, the Gated DeltaNet layers stay unset
(size 0): their fixed-size state is not migrated (serving_answers gdn_state_migration false).
"""

from __future__ import annotations

import socket
import zlib
from pathlib import Path

RECORD = 32  # tokens per record (chunk_n_tokens)
BFP8_TILE_BYTES = 1088


def config_names(n_kv: int) -> list[str]:
    return [f"{kind}_h{h}" for kind in ("k", "v") for h in range(n_kv)]


def record_bytes(head_dim: int) -> int:
    """One bfloat8_b TILE record of 32 tokens: head_dim / 32 tiles of 1088 B (8704 B at head_dim 256)."""
    return head_dim // 32 * BFP8_TILE_BYTES


def worker_host_name() -> str:
    """The host key the migration worker expects (as gemma4_d_p's kv_chunk_table.py)."""
    return f"host-{zlib.crc32(socket.gethostname().encode()) & 0x7FFFFFFF:08x}"


def record_location(cache, attn_layer: int, slot: int, pos: int, num_banks: int) -> tuple[int, int, int]:
    """(SP row, DRAM bank, offset from the buffer address) of the record at global position pos (a multiple of 32)."""
    local = cache.chunk // cache.sp
    row = (pos // local) % cache.sp
    lr = (pos // cache.chunk) * local + pos % local
    shard = cache.batch(slot, attn_layer) * (cache.max_seq // cache.sp // RECORD) + lr // RECORD
    return row, shard % num_banks, shard // num_banks * record_bytes(cache.head_dim)


def entry_noc_addr(cache, attn_of_layer: dict, layer: int, pos: int, slot: int, config: int, num_banks: int) -> int:
    """noc_addr ((bank << 32) | local address) of table entry (layer, pos, slot, config); 0 where the table leaves the
    entry unset (a table layer with no KV records)."""
    if layer not in attn_of_layer:
        return 0
    _, bank, off = record_location(cache, attn_of_layer[layer], slot, pos, num_banks)
    tensor = cache.k if config < cache.n_kv else cache.v
    return bank << 32 | int(tensor.buffer_address()) + off


def build_kv_chunk_table(mesh, cache, attn_of_layer: dict, num_layers: int):
    """The table over every (slot, attention layer, 32-position, config) record. attn_of_layer: {table layer: the
    layer's index in the cache's batch dim}; num_layers: table layers (the served layers)."""
    import ttnn

    D = ttnn.experimental.disaggregation
    sp, tp = mesh.shape[0], mesh.shape[1]
    assert cache.n_kv == tp, f"one KV head per TP column: {cache.n_kv} heads, {tp} columns"
    size = record_bytes(cache.head_dim)

    def config():
        c = D.KvChunkAddressTableConfig()
        c.num_layers = num_layers
        c.max_sequence_length = cache.max_seq
        c.num_slots = cache.num_slots
        c.chunk_n_tokens = RECORD
        c.chunk_size_bytes = size
        return c

    names = config_names(cache.n_kv)
    table = D.KvChunkAddressTable({n: config() for n in names})
    got = [table.config_name(i) for i in range(table.num_configs())]
    if got != names:
        raise RuntimeError(f"table config order {got}, want {names}")
    host = worker_host_name()
    groups = {}
    for r in range(sp):
        for h in range(tp):
            fnid = mesh.get_fabric_node_id(ttnn.MeshCoordinate(r, h))
            groups[(r, h)] = table.add_device_group([fnid])
            table.set_fabric_node_host(fnid, host_name=host)
    num_banks = mesh.dram_grid_size().x
    loc = D.KvCacheLocation()  # set() copies it
    loc.size_bytes = size
    for c in range(len(names)):
        h = c % cache.n_kv
        base = int((cache.k if c < cache.n_kv else cache.v).buffer_address())
        for layer, a in attn_of_layer.items():
            for slot in range(cache.num_slots):
                for pos in range(0, cache.max_seq, RECORD):
                    row, bank, off = record_location(cache, a, slot, pos, num_banks)
                    loc.device_group_index = groups[(row, h)]
                    loc.noc_addr = bank << 32 | base + off
                    table.set(layer, pos, slot, loc, c)
    return table


# ---------------------------------------------------------------- read-back (the producer's KV check, device-less)
def _pcc(a, b) -> float:
    import torch

    a, b = a.flatten().double(), b.flatten().double()
    if torch.equal(a, b):
        return 1.0
    return float(torch.corrcoef(torch.stack([a, b]))[0, 1])


def golden_kv(trace_dir, layer: int, length: int) -> dict:
    """{"k", "v": [n_kv, length, head_dim]} of a layer from a golden: the framework's (kv_cache/layer_<i>.safetensors,
    key_cache_layer_<i>) or the GPU trace's (cache/layer_<ii>.safetensors, past_key_values.layers.<i>.keys)."""
    from safetensors import safe_open

    d = Path(trace_dir)
    p = d / "kv_cache" / f"layer_{layer}.safetensors"
    if p.exists():
        keys = {"k": f"key_cache_layer_{layer}", "v": f"value_cache_layer_{layer}"}
    else:
        p = d / "cache" / f"layer_{layer:02d}.safetensors"
        keys = {"k": f"past_key_values.layers.{layer}.keys", "v": f"past_key_values.layers.{layer}.values"}
    with safe_open(str(p), framework="pt") as f:
        out = {}
        for kind, name in keys.items():
            t = f.get_tensor(name).float()
            out[kind] = t.reshape(-1, *t.shape[-2:])[:, :length]
    return out


def read_slot_kv(table, device_map: dict, layer: int, slot: int, config: int, length: int, head_dim: int):
    """[length, head_dim] of one (layer, slot, config) through the table, over UMD (as the KV Manager reads it)."""
    import torch

    import ttnn
    from models.demos.common.prefill.runners import prefill_producer as producer

    rows = []
    for pos in range(0, length, RECORD):
        loc = table.lookup(layer, pos, slot, config)
        uid = producer._resolve_unique_id(table.get_device_group(loc.device_group_index).fabric_node_ids, device_map)
        raw = bytes(ttnn.experimental.disaggregation.read_dram_umd(uid, loc.noc_addr, loc.size_bytes))
        if len(raw) != record_bytes(head_dim):
            raise ValueError(f"record (layer {layer}, pos {pos}, slot {slot}, config {config}) is {len(raw)} bytes")
        rows.append(producer._decode_bfp8_chunk(raw, head_dim))
    return torch.cat(rows, dim=0)[:length]


def read_slot_kv_and_check_pcc(table, device_map: dict, slot_id: int, real_len: int, trace_dir) -> dict:
    """{"k", "v": min PCC} of slot_id's KV over [0, real_len) on every table layer with records, vs the golden."""
    import torch
    from loguru import logger

    n_kv = table.num_configs() // 2
    head_dim = table.config(0).chunk_size_bytes // BFP8_TILE_BYTES * 32
    mins, checked = {"k": 1.0, "v": 1.0}, []
    for layer in range(table.config(0).num_layers):
        if not table.lookup(layer, 0, slot_id, 0).size_bytes:
            continue  # a Gated DeltaNet layer: no records
        gold = golden_kv(trace_dir, layer, real_len)
        for kind, base in (("k", 0), ("v", n_kv)):
            dev = torch.stack(
                [read_slot_kv(table, device_map, layer, slot_id, base + h, real_len, head_dim) for h in range(n_kv)]
            )
            p = _pcc(dev, gold[kind])
            mins[kind] = min(mins[kind], p)
            logger.info(f"[qwen38_27b] slot {slot_id} layer {layer} {kind.upper()} [0, {real_len}): PCC {p:.6f}")
        checked.append(layer)
    if not checked:
        raise RuntimeError(f"slot {slot_id}: no table layer with KV records (nothing verified)")
    return mins
