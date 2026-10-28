# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""The KV address table of MiMo's served cache (serving_contract.md "Adapter and table") and the producer's KV
read-back through it.

One config per (state, global KV head): ``key_h0..key_h7`` (192 wide, 6 bfp8 tiles = 6528 B per 32-token record) and
``value_h0..value_h7`` (128 wide, 4352 B). Heads 0-3 exist on every layer, heads 4-7 only on the sliding layers; the
other layers' entries of those configs stay unset (size 0). The protobuf orders configs by name, so config ids are
key_h0-7 = 0-7, value_h0-7 = 8-15, and the migration base address (entry 0) is layer 0's key head 0.

Record address (the ROUND_ROBIN_1D walk of a [slots, local_heads, local_seq, width] tensor with [1, 1, 32, width]
shards): shard = (slot * local_heads + local_head) * local_seq / 32 + local_row / 32, at bank ``shard % banks``,
``base + shard // banks * record_bytes``. Global position g sits on SP row ``(g // W) % sp`` at local row
``(g // chunk) * W + g % W`` (W = chunk / sp); global head h on TP column ``h // local_heads``.
"""

from __future__ import annotations

import socket
import zlib

import torch
from loguru import logger

import ttnn

RECORD = 32  # tokens per record (tt-d-gen tools/launch_harness/tables.py: chunk_n_tokens)
BFP8_TILE_BYTES = 1088
STATES = ("key", "value")


def worker_host_name(hostname: str | None = None) -> str:
    """Stable host key the migration worker expects (as gemma4_d_p's kv_chunk_table.py)."""
    hostname = socket.gethostname() if hostname is None else hostname
    return f"host-{zlib.crc32(hostname.encode()) & 0x7FFFFFFF:08x}"


def config_names(cfg: dict, layers) -> list[str]:
    from .kv_caches import kv_heads

    n = max(kv_heads(cfg, i) for i in layers)
    return [f"{state}_h{h}" for state in STATES for h in range(n)]


def record_bytes(width: int) -> int:
    return width // ttnn.TILE_SIZE * BFP8_TILE_BYTES


def _streams(kv):
    """(config name, layer, tensor, head, local heads, width) of every (config, layer) the table covers."""
    from .kv_caches import kv_heads, widths

    out = []
    for layer in kv.layer_ids:
        lkv = kv[layer]
        w = widths(kv.cfg, layer)
        for state in STATES:
            t = lkv.tensors()[state]
            if t.dtype != ttnn.bfloat8_b:
                raise ValueError(f"served cache must be bfloat8_b, got {t.dtype}")
            for h in range(kv_heads(kv.cfg, layer)):
                out.append((f"{state}_h{h}", layer, t, h, lkv.local_heads, w[state]))
    return out


def build_kv_chunk_address_table(*, mesh_device, kv, first_layer_idx: int = 0):
    """The table over every served layer (table layer = global layer - first_layer_idx)."""
    D = ttnn.experimental.disaggregation
    sp, tp, chunk, max_seq = kv.sp, kv.tp, kv.chunk_size, kv.max_seq_len
    if tuple(mesh_device.shape) != (sp, tp):
        raise ValueError(f"cache built for mesh {(sp, tp)}, got {tuple(mesh_device.shape)}")
    w_local = chunk // sp
    local_seq = max_seq // sp
    names = config_names(kv.cfg, kv.layer_ids)
    streams = _streams(kv)
    sizes = {}
    for name, _, _, _, _, width in streams:
        sizes.setdefault(name, record_bytes(width))
    configs = {}
    for name in names:
        c = D.KvChunkAddressTableConfig()
        c.num_layers = len(kv)
        c.max_sequence_length = max_seq
        c.num_slots = kv.num_users
        c.chunk_n_tokens = RECORD
        c.chunk_size_bytes = sizes[name]
        configs[name] = c
    table = D.KvChunkAddressTable(configs)
    order = [table.config_name(i) for i in range(table.num_configs())]
    cid = {name: i for i, name in enumerate(order)}
    if order[0] != "key_h0":
        raise RuntimeError(f"protobuf config order {order}: the migration base must be key_h0 (config 0)")

    banks = mesh_device.dram_grid_size().x
    host = worker_host_name()
    groups, hosted = {}, set()

    def group(r, c):
        if (r, c) not in groups:
            fnid = mesh_device.get_fabric_node_id(ttnn.MeshCoordinate(r, c))
            groups[(r, c)] = table.add_device_group([fnid])
            key = (int(fnid.mesh_id), int(fnid.chip_id))
            if key not in hosted:
                table.set_fabric_node_host(fnid, host_name=host)
                hosted.add(key)
        return groups[(r, c)]

    blocks_local = local_seq // RECORD
    table_set = table.set
    for name, layer, t, h, lh, width in streams:
        base = int(t.buffer_address())
        size = record_bytes(width)
        loc = D.KvCacheLocation()
        loc.size_bytes = size
        col, local_head = h // lh, h % lh
        for slot in range(kv.num_users):
            first_shard = (slot * lh + local_head) * blocks_local
            for g in range(0, max_seq, RECORD):
                r = (g // w_local) % sp
                shard = first_shard + ((g // chunk) * w_local + g % w_local) // RECORD
                loc.device_group_index = group(r, col)
                loc.noc_addr = (shard % banks) << 32 | base + shard // banks * size
                table_set(layer - first_layer_idx, g, slot, loc, cid[name])
    return table


def build_and_serialize_kv_chunk_table(*, path: str, mesh_device, kv, first_layer_idx: int = 0) -> str:
    from models.demos.common.prefill.runners.migration import serialize_prebuilt_kv_chunk_table

    table = build_kv_chunk_address_table(mesh_device=mesh_device, kv=kv, first_layer_idx=first_layer_idx)
    return serialize_prebuilt_kv_chunk_table(table=table, path=path)


# ---------------------------------------------------------------- the producer's read-back (prefill_producer.py)
def read_config_rows(table, device_map: dict, layer: int, slot: int, config_id: int, length: int) -> torch.Tensor:
    """[length, width] float32 of one (layer, config) through the table, as the KV Manager reads it."""
    from models.demos.common.prefill.runners.prefill_producer import _decode_bfp8_chunk, _resolve_unique_id

    size = int(table.config(config_id).chunk_size_bytes)
    width = size // BFP8_TILE_BYTES * ttnn.TILE_SIZE
    rows = []
    for pos in range(0, length, RECORD):
        loc = table.lookup(layer, pos, slot, config_id)
        if int(loc.size_bytes) != size:
            raise ValueError(
                f"layer {layer} config {config_id} position {pos}: record of {loc.size_bytes} B, want {size}"
            )
        uid = _resolve_unique_id(table.get_device_group(loc.device_group_index).fabric_node_ids, device_map)
        raw = ttnn.experimental.disaggregation.read_dram_umd(uid, loc.noc_addr, loc.size_bytes)
        rows.append(_decode_bfp8_chunk(bytes(raw), width))
    return torch.cat(rows, dim=0)[:length]


def read_slot_kv_and_check_pcc(table, device_map: dict, slot_id: int, real_len: int, trace_dir, num_layers=None):
    """{"key": min PCC, "value": min PCC} of the slot's rows [0, real_len) of every layer and head the table covers,
    against the golden's kv_cache/layer_<i>.safetensors (key_cache_layer_<i> [KV, seq, 192] after RoPE,
    value_cache_layer_<i> [KV, seq, 128] after the value scale). Table layer i is model layer i (one rank)."""
    from pathlib import Path

    from safetensors import safe_open

    from tests.ttnn.utils_for_testing import comp_pcc

    names = {table.config_name(c): c for c in range(table.num_configs())}
    n_layers = int(table.config(0).num_layers) if num_layers is None else int(num_layers)
    mins = {state: 1.0 for state in STATES}
    checked = 0
    for layer in range(n_layers):
        path = Path(trace_dir) / "kv_cache" / f"layer_{layer}.safetensors"
        with safe_open(str(path), framework="pt") as f:
            gold = {state: f.get_tensor(f"{state}_cache_layer_{layer}").float() for state in STATES}
        length = min(real_len, gold["key"].shape[-2])
        for state in STATES:
            for h in range(gold[state].shape[0]):
                c = names.get(f"{state}_h{h}")
                if c is None or int(table.lookup(layer, 0, slot_id, c).size_bytes) == 0:
                    raise ValueError(f"layer {layer} {state} head {h}: no table record")
                dev = read_config_rows(table, device_map, layer, slot_id, c, length)
                pcc = float(comp_pcc(gold[state][h, :length], dev, 0.0)[1])
                mins[state] = min(mins[state], pcc)
                checked += 1
        logger.info(f"  layer {layer}: key {mins['key']:.5f} value {mins['value']:.5f} (running min)")
    if not checked:
        raise RuntimeError(f"slot {slot_id}: no KV record checked")
    logger.info(f"[producer] slot {slot_id} MiMo KV PCC over [0, {real_len}) of {n_layers} layers -> {mins}")
    return mins
