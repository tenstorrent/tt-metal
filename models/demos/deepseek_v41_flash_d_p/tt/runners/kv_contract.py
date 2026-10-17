# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Prefill-server KV contract of DeepSeek-V4.1-Flash on an SP x TP mesh (bringup/serving_contract.md "KV cache",
"Adapter and table"; SP over mesh axis 0, TP over axis 1, both read from mesh.shape).

The engine-owned caches ARE the model's served state: every slot's ``DS41DeviceState`` is bound to its (slot, layer)
slabs of these tensors (tt/served.CacheSlab), and the window / compressor modules write them in place. One tensor per
table config, in table order (the names sort in it: ``KvChunkAddressTable``'s name -> config constructor numbers
configs in sorted name order, DS41-6):

    0_window_kv       every served layer               32 rows / record  x 512   TP-dedup
    1_compress_kv_r2  served KV sources of ratio 2     16                x 512   TP-dedup
    2_index_k_r2      the same layers                  16                x 128   TP-replicated
    3_compress_kv_r1  served KV sources of ratio 1     32                x 512   TP-dedup
    4_index_k_r1      the same layers                  32                x 128   TP-replicated

Each tensor: [slots x Lc, 1, local rows, width] bf16 ROW_MAJOR, DRAM ND-sharded [1, 1, 32 / r, width] ROUND_ROBIN_1D
over every DRAM bank (tt/served.alloc_served); batch row b = slot x Lc + the layer's row among the config's Lc
layers. Placement (the window path, tt/attention.py; the plan): token g on mesh row (g / W) % SP (W = chunk / SP),
TP-dedup: column (g mod W) / stripe (stripe = W / TP), local token row (g / chunk) stripe + g mod stripe;
TP-replicated: every column, local token row (g / chunk) W + g mod W. A compressed row j sits at the local row of
its first token / r. So the record of tokens [p, p + 32) is shard b x (local token rows / 32) + lr / 32 of its
chip, in bank (shard mod banks) at base + (shard div banks) x record bytes.

Address table: one config per tensor (chunk_n_tokens 32, chunk_size_bytes = one record, num_slots = slots,
max_sequence_length = max_seq, num_layers = the served (engine) layers L); a config sets entries only for the
layers it covers (config 0: all). Table layer = engine index k (= first_layer_idx + k; a non-contiguous bring-up
subset is served as engine layers 0..L-1, PREFILL_LAYER_IDS naming the model layers, F100). Device groups: a TP-dedup
record's chip alone; a TP-replicated record's mesh row (every chip holding the same bytes).
"""

from __future__ import annotations

import socket

import torch

import ttnn

from ..attention import WindowGeometry
from ..served import CacheSlab, alloc_served

RECORD = 32  # tokens per record (tools/launch_harness/tables.py: chunk_n_tokens)
CONFIG_ORDER = ("0_window_kv", "1_compress_kv_r2", "2_index_k_r2", "3_compress_kv_r1", "4_index_k_r1")


def served_configs(cfg, layer_ids: list[int]) -> list[dict]:
    """The table configs of these served model layers (engine index k = position in ``layer_ids``), table order."""
    out = [
        dict(name=CONFIG_ORDER[0], kind="window_kv", ratio=1, width=cfg.head_dim, dedup=True, layers=list(layer_ids))
    ]
    for r, (cn, ik) in ((2, CONFIG_ORDER[1:3]), (1, CONFIG_ORDER[3:5])):
        src = [i for i in layer_ids if cfg.is_kv_source(i) and cfg.ratio(i) == r]
        if src:
            out.append(dict(name=cn, kind="compress_kv", ratio=r, width=cfg.head_dim, dedup=True, layers=src))
            out.append(dict(name=ik, kind="index_k", ratio=r, width=cfg.index_dim, dedup=False, layers=src))
    for c in out:
        c["rows"] = RECORD // c["ratio"]  # rows per record
        c["record_bytes"] = c["rows"] * c["width"] * 2
        c["engine"] = [layer_ids.index(i) for i in c["layers"]]
    return out


class DS41ContractKV:
    """The engine-owned served caches + their address table (see the module docstring)."""

    def __init__(self, mesh, cfg, layer_ids: list[int], max_seq: int, chunk: int, num_users: int):
        from models.demos.common.prefill.runners.migration import get_num_dram_banks

        self.mesh, self.cfg, self.layer_ids = mesh, cfg, list(layer_ids)
        self.max_seq, self.chunk, self.num_users = int(max_seq), int(chunk), int(num_users)
        self.geo = g = WindowGeometry(mesh, chunk)
        assert self.max_seq % self.chunk == 0, (max_seq, chunk)
        self.num_banks = get_num_dram_banks(mesh)
        self.configs = served_configs(cfg, self.layer_ids)
        self.by_config = {}
        for c in self.configs:
            stripe = g.stripe if (c["dedup"] and g.tp > 1) else g.w
            c["stripe"] = stripe
            c["tok_rows"] = (self.max_seq // self.chunk) * stripe  # local token rows per chip
            assert c["tok_rows"] % RECORD == 0 and stripe % RECORD == 0, (c["name"], stripe)
            self.by_config[c["name"]] = alloc_served(
                mesh, c["width"], c["tok_rows"] // c["ratio"], c["rows"], self.num_users * len(c["layers"]), c["dedup"]
            )

    def slabs(self, slot: int) -> dict:
        """{model layer: {config name: CacheSlab}} of one slot (what DS41DeviceState binds)."""
        assert 0 <= slot < self.num_users, (slot, self.num_users)
        out = {}
        for c in self.configs:
            t, n = self.by_config[c["name"]], len(c["layers"])
            for row, layer in enumerate(c["layers"]):
                out.setdefault(layer, {})[c["name"]] = CacheSlab(t, slot, row, n)
        return out

    def base_address(self) -> int:
        return int(self.by_config[CONFIG_ORDER[0]].buffer_address())

    def stage_bases(self) -> list[int]:
        return [int(self.by_config[c["name"]].buffer_address()) for c in self.configs]

    # ------------------------------------------------------------------ placement
    def locate(self, c: dict, pos: int) -> tuple[int, int, int]:
        """Token position (record start) -> (mesh row, mesh column or -1 for every column, local token row)."""
        g = self.geo
        r = (pos // g.w) % g.sp
        if c["dedup"] and g.tp > 1:
            return r, (pos % g.w) // g.stripe, (pos // g.chunk) * g.stripe + pos % g.stripe
        return r, -1, (pos // g.chunk) * g.w + pos % g.w

    def address_table(self, seq_len: int, first_layer_idx: int = 0, stage_layouts=None):
        D = ttnn.experimental.disaggregation
        assert seq_len <= self.max_seq and seq_len % RECORD == 0, (seq_len, self.max_seq)
        L = len(self.layer_ids)
        g = self.geo
        host = socket.gethostname()
        fnids = [[self.mesh.get_fabric_node_id(ttnn.MeshCoordinate(r, c)) for c in range(g.tp)] for r in range(g.sp)]
        bases, banks = self.stage_bases(), [self.num_banks] * len(self.configs)
        if stage_layouts is not None:
            assert len(stage_layouts) == len(self.configs), (len(stage_layouts), len(self.configs))
            for cid, layout in enumerate(stage_layouts):
                assert len(layout) == 1, "single-rank table: one stage per config"
                st = layout[0]
                assert (st["first_layer"], st["count"]) == (first_layer_idx, L), (st["first_layer"], st["count"], L)
                assert int(st["base_addr"]) == bases[cid], (cid, st["base_addr"], bases[cid])
                banks[cid] = int(st["num_banks"])
            st = stage_layouts[0][0]
            host, fnids = f"host-{st['host_tag']:08x}", st["fnids"]
        table = D.KvChunkAddressTable(
            {c["name"]: self._config(L, seq_len, self.num_users, c["record_bytes"]) for c in self.configs}
        )
        for r in range(g.sp):
            for c in range(g.tp):
                table.set_fabric_node_host(fnids[r][c], host_name=host)
        chip_group = [[table.add_device_group([fnids[r][c]]) for c in range(g.tp)] for r in range(g.sp)]
        row_group = [table.add_device_group(list(fnids[r])) if g.tp > 1 else chip_group[r][0] for r in range(g.sp)]
        for cid, c in enumerate(self.configs):
            assert table.config_name(cid) == c["name"], (cid, table.config_name(cid), c["name"])
            base, nbytes, n = bases[cid], c["record_bytes"], len(c["layers"])
            per_batch = c["tok_rows"] // RECORD  # records per (slot, layer) slab on a chip
            for slot in range(self.num_users):
                for row, k in enumerate(c["engine"]):
                    b = slot * n + row
                    for pos in range(0, seq_len, RECORD):
                        mr, mc, lr = self.locate(c, pos)
                        j = b * per_batch + lr // RECORD  # ROUND_ROBIN_1D: shard j in bank j % B at j // B
                        loc = D.KvCacheLocation()
                        loc.noc_addr = ((j % banks[cid]) << 32) | (base + (j // banks[cid]) * nbytes)
                        loc.size_bytes = nbytes
                        loc.device_group_index = row_group[mr] if mc < 0 else chip_group[mr][mc]
                        table.set(first_layer_idx + k, pos, slot, loc, cid)
        return table

    @staticmethod
    def _config(num_layers: int, seq_len: int, slots: int, nbytes: int):
        cfg = ttnn.experimental.disaggregation.KvChunkAddressTableConfig()
        cfg.num_layers = num_layers
        cfg.max_sequence_length = seq_len
        cfg.num_slots = slots
        cfg.chunk_n_tokens = RECORD
        cfg.chunk_size_bytes = nbytes
        return cfg


# ---------------------------------------------------------------- read-back (host, through the table over UMD)
def _pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.double().flatten(), b.double().flatten()
    a, b = a - a.mean(), b - b.mean()
    den = a.norm() * b.norm()
    return float((a @ b) / den) if den > 0 else float(torch.equal(a, b))


def read_rows(table, device_map: dict, config_id: int, layer: int, slot: int, length: int, width: int) -> torch.Tensor:
    """Natural-order rows of the records covering token positions [0, ceil32(length)) of one table layer (bf16
    ROW_MAJOR records; a record of 32 / r rows), as the KV Manager reads them."""
    from models.demos.common.prefill.runners import prefill_producer as P

    D = ttnn.experimental.disaggregation
    out = []
    for pos in range(0, -(-length // RECORD) * RECORD, RECORD):
        loc = table.lookup(layer, pos, slot, config_id)
        assert loc.size_bytes, f"config {config_id}: no record for layer {layer} position {pos} slot {slot}"
        uid = P._resolve_unique_id(table.get_device_group(loc.device_group_index).fabric_node_ids, device_map)
        raw = bytes(D.read_dram_umd(uid, loc.noc_addr, loc.size_bytes))
        out.append(torch.frombuffer(bytearray(raw), dtype=torch.bfloat16).reshape(-1, width))
    return torch.cat(out).float() if out else torch.zeros(0, width)


def golden_rows(golden, c: dict, layer: int, length: int) -> torch.Tensor:
    """The golden rows of config ``c`` for the complete groups of tokens [0, length): window KV per position (the
    chunks' "kv" boundary), compressed KV / index keys from the layer's final state."""
    if c["kind"] == "window_kv":
        kv = torch.cat([golden.layer(ch, layer)["kv"] for ch in range(golden.n_chunks)])
        return kv[:length].float()
    return golden.state(layer)[c["kind"]][: length // c["ratio"]].float()


def read_slot_kv_and_check_pcc(table, device_map, slot_id: int, real_len: int, trace_dir, cfg, layer_ids) -> dict:
    """Min PCC per cache kind over [0, real_len) of every served layer vs the bring-up golden at ``trace_dir``
    (table layer k = layer_ids[k]): {window_kv, compress_kv, index_k}."""
    from loguru import logger

    from models.demos.common.bringup.reference.golden import Golden

    golden = Golden(trace_dir)
    names = [table.config_name(i) for i in range(table.num_configs())]
    mins = {}
    for c in served_configs(cfg, list(layer_ids)):
        cid = names.index(c["name"])
        for k, layer in zip(c["engine"], c["layers"]):
            want = golden_rows(golden, c, layer, real_len)
            got = read_rows(table, device_map, cid, k, slot_id, real_len, c["width"])[: want.shape[0]]
            assert got.shape == want.shape, (c["name"], layer, got.shape, want.shape)
            p = _pcc(got, want)
            mins[c["kind"]] = min(mins.get(c["kind"], 1.0), p)
            logger.info(f"  {c['name']} layer {layer} (table layer {k}): PCC {p:.6f}")
    logger.info(f"[ds41] slot {slot_id} KV PCC over [0, {real_len}) of layers {list(layer_ids)} -> {mins}")
    return mins
