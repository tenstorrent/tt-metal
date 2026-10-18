# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""The KV chunk address table of the served cache (tt.attention.TtKVCache) and its read-back over UMD.

Every layer's K and V are [slots, Hkv/TP, max_seq, D] bfp8 per chip, ND-sharded [1, 1, 32, D] ROUND_ROBIN_1D over the
DRAM banks: shard n = ((slot * Hkv/TP) + h) * (max_seq / 32) + pos / 32 lives in bank n % banks at
base + (n // banks) * shard_bytes. Mesh column c holds its own KV heads; the 8 mesh rows hold the same bytes (one
device group of 8 replicas per column).

Sixteen configs, all 256 wide (8704 bytes = 8 bfp8 tiles per record), in the order column c x {K, V} x sub 0/1:
  - sliding layer (2 KV heads x 256 per chip): sub = the chip's local head, so config (c, kind, sub) is KV head 2c+sub;
  - global layer (1 KV head x 512 per chip, head c // 2 on columns 2g, 2g+1): sub 0 is columns [0, 256) of the
    32 x 512 record (its first 8 tiles), sub 1 columns [256, 512) (the next 8), at +8704.
"""

from __future__ import annotations

import socket
import zlib

import numpy as np
import torch

import ttnn

TILE = 32
BFP8_TILE_BYTES = 1088
WIDTH = 256  # every config's record width
RECORD_BYTES = WIDTH // TILE * BFP8_TILE_BYTES  # 8704
KINDS = ("key", "value")


def config_id(column: int, kind: int, sub: int) -> int:
    return column * 4 + kind * 2 + sub


def config_names(tp: int) -> tuple:
    return tuple(
        f"{config_id(c, k, s):02d}_c{c}_{'kv'[k]}{s}" for c in range(tp) for k in range(len(KINDS)) for s in range(2)
    )


def worker_host_name(hostname: str | None = None) -> str:
    """Host key the migration worker expects (as gemma4_d_p's table)."""
    hostname = socket.gethostname() if hostname is None else hostname
    return f"host-{zlib.crc32(hostname.encode()) & 0x7FFFFFFF:08x}"


def _record(cache, slot: int, sub: int, pos: int, banks: int, base: int) -> tuple[int, int]:
    """(noc_addr, size) of config sub `sub` of the 32-token record at `pos` of `slot` in one cache tensor."""
    blocks = cache.max_seq // TILE
    shard_bytes = cache.d // TILE * BFP8_TILE_BYTES
    head, offset = (sub, 0) if cache.nkv == 2 else (0, sub * RECORD_BYTES)
    assert cache.nkv in (1, 2) and cache.d * cache.nkv == 2 * WIDTH, (cache.nkv, cache.d)
    shard = (slot * cache.nkv + head) * blocks + pos // TILE
    return (shard % banks) << 32 | base + (shard // banks) * shard_bytes + offset, RECORD_BYTES


def build_table(mesh, caches: dict, num_slots: int, max_seq: int):
    """The table over `caches` ({layer: TtKVCache}, layers 0..n-1)."""
    D = ttnn.experimental.disaggregation
    rows, tp = (int(x) for x in mesh.shape)
    names = config_names(tp)
    configs = {}
    for name in names:
        cfg = D.KvChunkAddressTableConfig()
        cfg.num_layers = len(caches)
        cfg.max_sequence_length = max_seq
        cfg.num_slots = num_slots
        cfg.chunk_n_tokens = TILE
        cfg.chunk_size_bytes = RECORD_BYTES
        configs[name] = cfg
    table = D.KvChunkAddressTable(configs)
    got = tuple(table.config_name(i) for i in range(table.num_configs()))
    if got != names:
        raise RuntimeError(f"table config order {got} != {names}")
    host = worker_host_name()
    groups = []
    for c in range(tp):
        fnids = [mesh.get_fabric_node_id(ttnn.MeshCoordinate(r, c)) for r in range(rows)]
        for f in fnids:
            table.set_fabric_node_host(f, host_name=host)
        groups.append(table.add_device_group(fnids))
    banks = mesh.dram_grid_size().x
    loc = D.KvCacheLocation()
    for layer in sorted(caches):
        cache = caches[layer]
        assert cache.max_seq == max_seq and cache.slots == num_slots, (cache.max_seq, cache.slots)
        for k, t in enumerate((cache.k, cache.v)):
            base = int(t.buffer_address())
            for c in range(tp):
                loc.device_group_index = groups[c]
                for sub in range(2):
                    cid = config_id(c, k, sub)
                    for slot in range(num_slots):
                        for pos in range(0, max_seq, TILE):
                            loc.noc_addr, loc.size_bytes = _record(cache, slot, sub, pos, banks, base)
                            table.set(layer, pos, slot, loc, cid)
    return table


def export_table(mesh, caches: dict, num_slots: int, max_seq: int, path: str) -> str:
    ttnn.experimental.disaggregation.export_to_protobuf_file(build_table(mesh, caches, num_slots, max_seq), path)
    return path


def decode_bfp8(raw: bytes, width: int) -> torch.Tensor:
    """One TILE-layout bfp8 record of 32 rows x width -> fp32 [32, width] (shared exponent per 16-value face row)."""
    n = width // TILE
    u8 = np.frombuffer(raw, dtype=np.uint8).reshape(n, BFP8_TILE_BYTES)
    exp = u8[:, :64].astype(np.int32).reshape(n, 4, 16)
    man = u8[:, 64:].reshape(n, 4, 16, 16)
    mag = (man & 0x7F).astype(np.float32) * np.exp2((exp - 133).astype(np.float32))[..., None]
    val = np.where((man >> 7) > 0, -mag, mag)
    tiles = val.reshape(n, 2, 2, 16, 16).transpose(0, 1, 3, 2, 4).reshape(n, TILE, TILE)
    return torch.from_numpy(np.ascontiguousarray(tiles.transpose(1, 0, 2).reshape(TILE, n * TILE)))


def _read(table, dmap: dict, layer: int, slot: int, cid: int, length: int) -> torch.Tensor:
    """Config `cid` of `layer` / `slot` over rows [0, length) -> [length, 256] (first chip of the group in dmap)."""
    D = ttnn.experimental.disaggregation
    rows = []
    uid = None
    for pos in range(0, length, TILE):
        loc = table.lookup(layer, pos, slot, cid)
        if uid is None:
            nodes = table.get_device_group(loc.device_group_index).fabric_node_ids
            keys = [(int(n.mesh_id), int(n.chip_id)) for n in nodes]
            uid = next((dmap[k] for k in keys if k in dmap), None)
            if uid is None:
                raise KeyError(f"no chip of device group {keys} in the device map")
        rows.append(decode_bfp8(bytes(D.read_dram_umd(uid, loc.noc_addr, loc.size_bytes)), WIDTH))
    return torch.cat(rows, dim=0)[:length]


def read_slot_kv(table, dmap: dict, slot: int, length: int, sliding: dict) -> dict:
    """{layer: {"key", "value"}} position-major [length, Hkv * D] (the reference state layout) through the table.

    sliding: {layer: True for a sliding layer (8 KV heads x 256), False for a global one (2 KV heads x 512)}."""
    tp = table.num_configs() // 4
    out = {}
    for layer, is_sliding in sliding.items():
        out[layer] = {}
        for k, name in enumerate(KINDS):
            if is_sliding:  # head 2c + sub = config (c, k, sub)
                heads = [_read(table, dmap, layer, slot, config_id(c, k, s), length) for c in range(tp) for s in (0, 1)]
            else:  # global head g from column 2g: sub 0 | sub 1
                heads = [
                    torch.cat([_read(table, dmap, layer, slot, config_id(2 * g, k, s), length) for s in (0, 1)], -1)
                    for g in range(tp // 2)
                ]
            out[layer][name] = torch.cat(heads, dim=-1)
    return out


def pcc(a, b) -> float:
    """Pearson correlation in float64."""
    a = a.double().flatten()
    b = b.double().flatten()
    a = a - a.mean()
    b = b - b.mean()
    den = a.norm() * b.norm()
    return float((a @ b) / den) if den > 0 else float(bool((a == b).all()))


def kv_pcc_mins(got: dict, golden_of, length: int) -> dict:
    """{"k": min PCC over layers, "v": ...} of read-back `got` vs golden_of(layer) -> {"key", "value"} [>= length, :]."""
    mins = {"k": 1.0, "v": 1.0}
    for layer, kv in got.items():
        want = golden_of(layer)
        for short, name in (("k", "key"), ("v", "value")):
            w = want[name].reshape(-1, want[name].shape[-1])[:length].float()
            mins[short] = min(mins[short], pcc(kv[name][:length], w))
    return mins
