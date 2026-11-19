# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""The served caches, their address table and the producer's read-back of them (serving contract, "KV cache" and
"Adapter and table").

Caches (tt/model.py alloc_caches, DeepSeek's init_kvpe_cache): per chip [slots * regions, 1, max_seq / SP, width],
bfloat8_b TILE, ND-sharded [1, 1, 32, width] ROUND_ROBIN_1D over the chip's B DRAM banks, replicated over TP. The
region of (slot, layer) is batch slot * regions + region, as the device model indexes them (tt/model.py State):

- config 0 (``"00"``, kind ``kvpe``), width 576 = [kv_latent 512 | k_rope 64]: one region per layer that owns its KV
  (NoPE layers keep the pe columns zero); shared-KV layers read their source layer's region and have no entry.
- config 1 (``"01"``, kind ``index``), width 128, the DSA indexer key: one region per full-indexer layer; shared-index
  layers have no entry. With INDEXER_HADAMARD (the P/D deployment) the keys are rotated by H128 / sqrt(128), as the
  Reload decode caches them.

Shard j of a cache (flat over [batch, rows / 32]) sits in bank j % B at buffer_address + (j // B) * record_bytes: one
32-token record is one shard, contiguous in one bank. Position g of a slot lives on SP row (g // W) % SP at local row
(g // chunk) * W + g % W (W = chunk / SP: the engine's ring_sdpa_reshuffle, block-cyclic with period chunk). A record
never crosses a chip (W % 32 == 0); its device group is the TP chips of that SP row, which hold identical replicas.

Table layer k is model layer first + k (the served layers are one contiguous run). Config names "00" / "01" keep the
KVPE cache at config id 0 and the index cache at 1 (the table numbers configs in name order; tt-metal's
deepseek_v3_d_p kv_chunk_table uses the same names).
"""

from __future__ import annotations

import zlib
from dataclasses import dataclass
from pathlib import Path

import torch

import ttnn

RECORD_TOKENS = 32
CONFIGS = ("00", "01")  # table config names, in id order
KINDS = {"00": "kvpe", "01": "index"}
_TILE_BYTES = {"BFLOAT8_B": 1088, "BFLOAT16": 2048}


@dataclass(frozen=True)
class Geometry:
    """What the caches and the table hold: layers (model layer of table layer k), kv_owners / index_layers (the
    layers with a region in config 0 / 1, in region order, as tt/model.py TtModel kv_region / index_region), and the
    serving shape."""

    layers: tuple
    kv_owners: tuple
    index_layers: tuple
    sp: int
    chunk: int
    max_seq: int
    slots: int
    kvpe_width: int
    index_width: int

    @classmethod
    def of(cls, cfg, layers, *, sp: int, chunk: int, max_seq: int, slots: int) -> "Geometry":
        """cfg: reference/model.py Config."""
        layers = tuple(layers)
        assert layers == tuple(range(layers[0], layers[0] + len(layers))), f"served layers must be contiguous: {layers}"
        owners = tuple(i for i in layers if cfg.owns_kv(i))
        for i in layers:
            assert cfg.owns_kv(i) or cfg.kv_source(i) in owners, f"layer {i} reads layer {cfg.kv_source(i)}'s KV"
        assert chunk % (sp * RECORD_TOKENS) == 0 and max_seq % chunk == 0, (sp, chunk, max_seq)
        return cls(
            layers,
            owners,
            tuple(i for i in layers if cfg.is_full(i)),
            sp,
            chunk,
            max_seq,
            slots,
            cfg.kv_lora + cfg.qk_rope,
            cfg.index_dim,
        )

    def width(self, name: str) -> int:
        return self.kvpe_width if KINDS[name] == "kvpe" else self.index_width

    def owners(self, name: str) -> tuple:
        return self.kv_owners if KINDS[name] == "kvpe" else self.index_layers

    def regions(self, name: str) -> int:
        return len(self.owners(name))

    def region(self, name: str, layer: int):
        """The region of a model layer in a config's cache; None: the layer has no entry in that config."""
        own = self.owners(name)
        return own.index(layer) if layer in own else None

    def rows_local(self) -> int:
        return self.max_seq // self.sp

    def place(self, pos: int) -> tuple[int, int]:
        """(SP row, local row) of position pos."""
        w = self.chunk // self.sp
        return (pos // w) % self.sp, (pos // self.chunk) * w + pos % w


def record_bytes(cache) -> int:
    """Bytes of one 32-token record (one [32, width] shard) of a TILE cache."""
    return int(cache.shape[-1]) // ttnn.TILE_SIZE * _TILE_BYTES[str(cache.dtype).split(".")[-1]]


def record_addr(geo: Geometry, name: str, k: int, pos: int, slot: int, base: int, banks: int, size: int):
    """(SP row, noc_addr) of the record (table layer k, position pos, slot) in config ``name``; None if unset."""
    region = geo.region(name, geo.layers[k])
    if region is None:
        return None
    row, local = geo.place(pos)
    shard = (slot * geo.regions(name) + region) * (geo.rows_local() // RECORD_TOKENS) + local // RECORD_TOKENS
    return row, (shard % banks) << 32 | (base + (shard // banks) * size)


def host_name() -> str:
    """The worker host key the migration code uses (migration._host_tag_int)."""
    import socket

    return f"host-{zlib.crc32(socket.gethostname().encode()) & 0x7FFFFFFF:08x}"


def build_table(geo: Geometry, caches: dict, *, bases: dict, banks: int, row_fnids: list, host: str):
    """The KvChunkAddressTable of the caches ({name: tensor}). bases: {name: buffer address}; row_fnids: per SP row
    the fabric node ids of its TP chips (one device group each)."""
    D = ttnn.experimental.disaggregation
    configs = {}
    for name in CONFIGS:
        c = D.KvChunkAddressTableConfig()
        c.num_layers = len(geo.layers)
        c.max_sequence_length = geo.max_seq
        c.num_slots = geo.slots
        c.chunk_n_tokens = RECORD_TOKENS
        c.chunk_size_bytes = record_bytes(caches[name])
        configs[name] = c
    table = D.KvChunkAddressTable(configs)
    got = tuple(table.config_name(i) for i in range(table.num_configs()))
    assert got == CONFIGS, f"table config order {got}, expected {CONFIGS}"
    groups = []
    for fnids in row_fnids:
        groups.append(table.add_device_group(list(fnids)))
        for f in fnids:
            table.set_fabric_node_host(f, host_name=host)
    for name in CONFIGS:
        cid, size = table.config_id_of(name), record_bytes(caches[name])
        loc = D.KvCacheLocation()
        loc.size_bytes = size
        for k in range(len(geo.layers)):
            if geo.region(name, geo.layers[k]) is None:
                continue  # the layer owns no such cache: its entries stay unset
            for slot in range(geo.slots):
                for pos in range(0, geo.max_seq, RECORD_TOKENS):
                    row, addr = record_addr(geo, name, k, pos, slot, int(bases[name]), banks, size)
                    loc.device_group_index = groups[row]
                    loc.noc_addr = addr
                    table.set(k, pos, slot, loc, cid)
    return table


def base_address(geo: Geometry, caches: dict, banks: int) -> int:
    """The migration base: the local address of table entry (layer 0, position 0, slot 0, config 0)."""
    name = CONFIGS[0]
    at = record_addr(geo, name, 0, 0, 0, int(caches[name].buffer_address()), banks, record_bytes(caches[name]))
    return 0 if at is None else at[1] & 0xFFFFFFFF


# ---------------------------------------------------------------- the producer's read-back
def config_rows(st: dict, n: int, kvpe_width: int) -> dict:
    """{config: [n, width]} of a layer's golden state (reference state_tensors: kv_latent, k_rope, index_key; a cache
    the layer does not own is [n, 0])."""
    out = {}
    lat, kr = st["kv_latent"][:n].float(), st["k_rope"][:n].float()
    if lat.shape[-1]:
        out["00"] = torch.nn.functional.pad(torch.cat([lat, kr], -1), (0, kvpe_width - lat.shape[1] - kr.shape[1]))
    ik = st["index_key"][:n].float()
    if ik.shape[-1]:
        from models.demos.deepseek_v3_d_p.tt.mistral_large_4.indexer import hadamard_matrix
        from models.demos.deepseek_v3_d_p.tt.mistral_large_4.settings import settings

        # INDEXER_HADAMARD: the cache holds the keys rotated (tt/indexer.py TtIndexer._rotate)
        out["01"] = ik @ hadamard_matrix(ik.shape[-1]) if settings.get("INDEXER_HADAMARD") else ik
    return out


def read_slot_kv_and_check_pcc(table, device_map: dict, slot_id: int, real_len: int, trace_dir, layers, kvpe_width):
    """The slot's records of positions [0, real_len) through the table, each from the first chip of its device group
    in device_map ({(mesh_id, chip_id): unique id}), decoded and compared with the golden state in trace_dir
    (kv_cache/layer_<i>.safetensors: <name>_cache_layer_<i> for kv_latent, k_rope, index_key) -> {kind: min PCC over
    the layers}. layers: model layer of table layer k."""
    from loguru import logger
    from safetensors.torch import load_file

    from models.demos.common.prefill.runners.prefill_producer import _decode_bfp8_chunk
    from tests.ttnn.utils_for_testing import comp_pcc

    D = ttnn.experimental.disaggregation
    cid = {table.config_name(c): c for c in range(table.num_configs())}
    end = -(-real_len // RECORD_TOKENS) * RECORD_TOKENS
    mins = {}
    for k, layer in enumerate(layers):
        d = load_file(str(Path(trace_dir) / "kv_cache" / f"layer_{layer}.safetensors"))
        st = {
            n: d.get(f"{n}_cache_layer_{layer}", torch.zeros(real_len, 0)) for n in ("kv_latent", "k_rope", "index_key")
        }
        for name, gold in config_rows(st, real_len, kvpe_width).items():
            recs = []
            for pos in range(0, end, RECORD_TOKENS):
                loc = table.lookup(k, pos, slot_id, cid[name])
                fnids = table.get_device_group(loc.device_group_index).fabric_node_ids
                uid = next(
                    device_map[key]
                    for key in ((int(f.mesh_id), int(f.chip_id)) for f in fnids)
                    if key in device_map
                )
                recs.append(
                    _decode_bfp8_chunk(bytes(D.read_dram_umd(uid, loc.noc_addr, loc.size_bytes)), gold.shape[-1])
                )
            dev = torch.cat(recs)[:real_len]
            pcc = float(comp_pcc(gold, dev.float(), 0.0)[1])
            kind = KINDS[name]
            mins[kind] = min(mins.get(kind, 1.0), pcc)
            logger.info(f"[read-back] slot {slot_id} table layer {k} (model layer {layer}) {kind}: pcc {pcc:.6f}")
    logger.info(f"[read-back] slot {slot_id} KV over [0, {real_len}): min pcc {mins}")
    return mins
