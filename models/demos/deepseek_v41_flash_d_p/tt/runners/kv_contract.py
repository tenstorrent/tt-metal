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

INDEX_K_SERVED=fp8_row192 (P.39, the default; owner 2026-11-06): the index configs hold Blaze decode's 192-byte rows
(uint8 [.., rows, 192]: e4m3 of key / scale | fp32 scale | zeros; records 16 x 192 / 32 x 192 B) and also cover the
re-index layers that read the source's keys (4_index_k_r1: 20, 24, 28, 32, 36; every one the same rows, written
with the source's: tt/compressor.reindex_layers). INDEX_K_SERVED=bf16: bf16 rows of 128, KV sources only.

Each tensor: [slots x Lc, 1, local rows, width] bf16 ROW_MAJOR, DRAM ND-sharded [1, 1, 32 / r, width] ROUND_ROBIN_1D
over every DRAM bank (tt/served.alloc_served); batch row b = slot x Lc + the layer's row among the config's Lc
layers. Placement (the window path, tt/attention.py; the plan): token g on mesh row (g / W) % SP (W = chunk / SP),
TP-dedup: column (g mod W) / stripe (stripe = W / TP), local token row (g / chunk) stripe + g mod stripe;
TP-replicated: every column, local token row (g / chunk) W + g mod W. A compressed row j sits at the local row of
its first token / r. So the record of tokens [p, p + 32) is shard b x (local token rows / 32) + lr / 32 of its
chip, in bank (shard mod banks) at base + (shard div banks) x record bytes.

WINDOW_KV_KEEP=snapshot128 (P.42, the default; owner 2026-11-07, decode's review): 0_window_kv is a SNAPSHOT config
(KvChunkTensorType.Snapshot; records at positions 0, 32, 64, 96 only, so tt-d-gen's KV Manager reads it as a
128-position snapshot: kv_chunk_table.cpp conformSnapshot -> SnapshotOfExtent(128); its declared
max_sequence_length is 128, decode's table (P.43; tt-metal's table refuses a lookup at or past it,
kv_chunk_address_table.cpp:138, so every reader stops at 128: testing/contract.py reads a Snapshot config at the
request's last chunk only, its 4 records): per (slot, layer) 128 rows, position p at row p % 128 for
p < cap = (prompt_len - 1) // kv_block_size x kv_block_size = actual_end // kv_block_size x kv_block_size under PD
(P.49, DS41-219; decode receives [0, cap) and replays [cap, prompt_len)),
4 records of 32 rows (32 KiB each, one bank), the same bytes on every chip (device group: the whole mesh); tensor
[slots x L, 1, 128, 512], replicated. REINDEX_SERVE=alias (P.42, the default): 4_index_k_r1's entries of the re-index
layers are their key source's records (the same noc_addr, size and device group at the same (slot, position)); the
tensor holds the KV sources' slabs only (``alloc``).

Address table: one config per tensor (chunk_n_tokens 32, chunk_size_bytes = one record, num_slots = slots,
max_sequence_length = max_seq (the snapshot: 128), num_layers = the served (engine) layers L); a config sets entries only for the
layers it covers (config 0: all). Table layer = engine index k (= first_layer_idx + k; a non-contiguous bring-up
subset is served as engine layers 0..L-1, PREFILL_LAYER_IDS naming the model layers, F100). Device groups: a TP-dedup
record's chip alone; a TP-replicated record's mesh row (every chip holding the same bytes).
"""

from __future__ import annotations

import socket

import torch

import ttnn

from ..attention import SNAP, WindowGeometry, snapshot_cap, window_keep
from ..compressor import FP8_ROW_BYTES, fp8_rows_decode, index_k_fp8, reindex_alias, reindex_layers
from ..served import CacheSlab, alloc_served

RECORD = 32  # tokens per record (tools/launch_harness/tables.py: chunk_n_tokens)
CONFIG_ORDER = ("0_window_kv", "1_compress_kv_r2", "2_index_k_r2", "3_compress_kv_r1", "4_index_k_r1")


def served_configs(cfg, layer_ids: list[int], keep: str | None = None) -> list[dict]:
    """The table configs of these served model layers (engine index k = position in ``layer_ids``), table order.
    ``keep`` (WINDOW_KV_KEEP; None: the setting): full = 0_window_kv of every position; snapshot128 (P.42) = the
    128-row snapshot (``snapshot``: tensor_type Snapshot, ``seq`` 128); ring | none (P.32) = no 0_window_kv (the slot's
    window rows are its state's ring). Each config's ``alloc``: the layers its tensor holds slabs for (REINDEX_SERVE=
    alias: the KV sources only; ``of`` maps an alias layer to its source)."""
    keep = keep or window_keep()
    out = []
    if keep == "full":
        out.append(
            dict(
                name=CONFIG_ORDER[0], kind="window_kv", ratio=1, width=cfg.head_dim, dedup=True, layers=list(layer_ids)
            )
        )
    elif keep == "snapshot128":
        out.append(
            dict(
                name=CONFIG_ORDER[0],
                kind="window_kv",
                ratio=1,
                width=cfg.head_dim,
                dedup=False,
                layers=list(layer_ids),
                snapshot=True,
                seq=SNAP,
            )
        )
    fp8 = index_k_fp8()
    for r, (cn, ik) in ((2, CONFIG_ORDER[1:3]), (1, CONFIG_ORDER[3:5])):
        src = [i for i in layer_ids if cfg.is_kv_source(i) and cfg.ratio(i) == r]
        if src:
            out.append(dict(name=cn, kind="compress_kv", ratio=r, width=cfg.head_dim, dedup=True, layers=src))
            idx = dict(name=ik, kind="index_k", ratio=r, width=cfg.index_dim, dedup=False, layers=src)
            if fp8:  # P.39: Blaze's rows, also under the re-index layers reading these keys (source of each: "of")
                mirrors = {j: i for i in src for j in reindex_layers(cfg, layer_ids, i)}
                idx.update(fp8=True, layers=sorted(src + list(mirrors)), of=mirrors)
                if mirrors and reindex_alias():  # P.42: their entries are the source's records; slabs for src only
                    idx.update(alias=True, alloc=list(src))
            out.append(idx)
    for c in out:
        c.setdefault("alloc", list(c["layers"]))
        c.setdefault("seq", None)  # (None: max_seq, a KV config)
        c["rows"] = RECORD // c["ratio"]  # rows per record
        c["row_bytes"] = FP8_ROW_BYTES if c.get("fp8") else c["width"] * 2
        c["alloc_width"] = FP8_ROW_BYTES if c.get("fp8") else c["width"]  # elements a row of the tensor
        c["dtype"] = ttnn.uint8 if c.get("fp8") else ttnn.bfloat16
        c["record_bytes"] = c["rows"] * c["row_bytes"]
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
        from ..attention import window_keep

        self.window_keep = window_keep()  # (P.32: ring | none hold no 0_window_kv here; P.42: snapshot128 its snapshot)
        self.configs = served_configs(cfg, self.layer_ids, self.window_keep)
        self.by_config = {}
        for c in self.configs:
            stripe = g.stripe if (c["dedup"] and g.tp > 1) else g.w
            c["stripe"] = stripe
            # local token rows per chip (P.42 snapshot: its 128 rows, every chip)
            c["tok_rows"] = SNAP if c.get("snapshot") else (self.max_seq // self.chunk) * stripe
            assert c["tok_rows"] % RECORD == 0 and stripe % RECORD == 0, (c["name"], stripe)
            self.by_config[c["name"]] = alloc_served(
                mesh,
                c["alloc_width"],
                c["tok_rows"] // c["ratio"],
                c["rows"],
                self.num_users * len(c["alloc"]),
                c["dedup"],
                dtype=c["dtype"],
            )

    def slabs(self, slot: int) -> dict:
        """{model layer: {config name: CacheSlab}} of one slot (what DS41DeviceState binds)."""
        assert 0 <= slot < self.num_users, (slot, self.num_users)
        out = {}
        for c in self.configs:
            t, n = self.by_config[c["name"]], len(c["alloc"])
            for row, layer in enumerate(c["alloc"]):
                out.setdefault(layer, {})[c["name"]] = CacheSlab(t, slot, row, n)
        return out

    def base_address(self) -> int:
        return int(self.by_config[self.configs[0]["name"]].buffer_address())

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
            {
                c["name"]: self._config(L, c["seq"] or seq_len, self.num_users, c["record_bytes"], c.get("snapshot"))
                for c in self.configs
            }
        )
        for r in range(g.sp):
            for c in range(g.tp):
                table.set_fabric_node_host(fnids[r][c], host_name=host)
        chip_group = [[table.add_device_group([fnids[r][c]]) for c in range(g.tp)] for r in range(g.sp)]
        row_group = [table.add_device_group(list(fnids[r])) if g.tp > 1 else chip_group[r][0] for r in range(g.sp)]
        mesh_group = None  # (P.42 snapshot: every chip holds the same bytes)
        for cid, c in enumerate(self.configs):
            assert table.config_name(cid) == c["name"], (cid, table.config_name(cid), c["name"])
            base, nbytes, n = bases[cid], c["record_bytes"], len(c["alloc"])
            per_batch = c["tok_rows"] // RECORD  # records per (slot, layer) slab on a chip
            if c.get("snapshot") and mesh_group is None:
                mesh_group = table.add_device_group([f for row in fnids for f in row])
            for slot in range(self.num_users):
                for layer, k in zip(c["layers"], c["engine"]):
                    # P.42 REINDEX_SERVE=alias: a re-index layer's entries are its key source's records
                    b = slot * n + c["alloc"].index(c.get("of", {}).get(layer, layer) if c.get("alias") else layer)
                    for pos in range(0, c["seq"] or seq_len, RECORD):
                        if c.get("snapshot"):  # P.42: record pos / 32 of the slab's 128 rows, on every chip
                            j, dg = b * per_batch + pos // RECORD, mesh_group
                        else:
                            mr, mc, lr = self.locate(c, pos)
                            j = b * per_batch + lr // RECORD  # ROUND_ROBIN_1D: shard j in bank j % B at j // B
                            dg = row_group[mr] if mc < 0 else chip_group[mr][mc]
                        loc = D.KvCacheLocation()
                        loc.noc_addr = ((j % banks[cid]) << 32) | (base + (j // banks[cid]) * nbytes)
                        loc.size_bytes = nbytes
                        loc.device_group_index = dg
                        table.set(first_layer_idx + k, pos, slot, loc, cid)
        return table

    @staticmethod
    def _config(num_layers: int, seq_len: int, slots: int, nbytes: int, snapshot: bool = False):
        D = ttnn.experimental.disaggregation
        cfg = D.KvChunkAddressTableConfig()
        cfg.num_layers = num_layers
        cfg.max_sequence_length = seq_len
        cfg.num_slots = slots
        cfg.chunk_n_tokens = RECORD
        cfg.chunk_size_bytes = nbytes
        if snapshot:  # P.42: fixed-size state (tt-d-gen KvTensorType::SNAPSHOT: moved whole, positions 0..96)
            cfg.tensor_type = D.KvChunkTensorType.Snapshot
        return cfg


# ---------------------------------------------------------------- read-back (host, through the table over UMD)
SNAPSHOT_ENDS = {}  # P.42: {slot: the end of the last chunk the in-process runtime ran} (the snapshot's cap source)


def _pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.double().flatten(), b.double().flatten()
    a, b = a - a.mean(), b - b.mean()
    den = a.norm() * b.norm()
    return float((a @ b) / den) if den > 0 else float(torch.equal(a, b))


def read_rows(
    table, device_map: dict, config_id: int, layer: int, slot: int, length: int, width: int, fp8: bool = False
) -> torch.Tensor:
    """Natural-order rows of the records covering token positions [0, ceil32(length)) of one table layer (bf16
    ROW_MAJOR records; a record of 32 / r rows), as the KV Manager reads them. ``fp8`` (an index config under
    INDEX_K_SERVED=fp8_row192): 192-byte rows, decoded to the values they hold (tt/compressor.fp8_rows_decode)."""
    from models.demos.common.prefill.runners import prefill_producer as P

    D = ttnn.experimental.disaggregation
    out = []
    for pos in range(0, -(-length // RECORD) * RECORD, RECORD):
        loc = table.lookup(layer, pos, slot, config_id)
        assert loc.size_bytes, f"config {config_id}: no record for layer {layer} position {pos} slot {slot}"
        uid = P._resolve_unique_id(table.get_device_group(loc.device_group_index).fabric_node_ids, device_map)
        raw = bytes(D.read_dram_umd(uid, loc.noc_addr, loc.size_bytes))
        if fp8:
            out.append(fp8_rows_decode(torch.frombuffer(bytearray(raw), dtype=torch.uint8).reshape(-1, FP8_ROW_BYTES)))
        else:
            out.append(torch.frombuffer(bytearray(raw), dtype=torch.bfloat16).reshape(-1, width))
    return torch.cat(out).float() if out else torch.zeros(0, width)


def golden_window(golden, layer: int, lo: int, hi: int) -> tuple[torch.Tensor, torch.Tensor]:
    """(positions, rows) of layer's golden window KV in [lo, hi) the golden holds: every position of a layer with
    per-chunk dumps, else the positions its ring snapshots hold (kv_cache/layer_i[_at_p], position q at ring row
    q % 128; a position two snapshots hold: the later one; F264 handoff snapshots included). Test-side only."""
    if all(golden.has_layer(ch, layer) for ch in range(golden.n_chunks)):
        kv = torch.cat([golden.layer(ch, layer)["kv"] for ch in range(golden.n_chunks)])
        pos = torch.arange(max(0, lo), min(hi, kv.shape[0]))
        return pos, kv[pos].float()
    from models.demos.common.bringup.reference.handoff import index as handoff_index

    rows = {}
    # (FIX.26, F264: the golden's serving handoff snapshots too, e.g. at decode's cap: the rows below the cap that
    # neither the request-end nor the final ring holds)
    handoffs = set((handoff_index(golden.dir) or {}).get("points", []))
    for at in sorted(set(golden.meta.get("state_snapshots", [])) | {golden.seq} | handoffs, reverse=True):
        try:
            ring = golden.state(layer, at=None if at == golden.seq else at)["window_kv"]
        except FileNotFoundError:
            continue
        ring = ring.reshape(-1, ring.shape[-1])
        for q in range(max(lo, at - ring.shape[0], 0), min(hi, at)):
            rows.setdefault(q, ring[q % ring.shape[0]])
    pos = sorted(rows)
    return torch.tensor(pos, dtype=torch.int64), (torch.stack([rows[q] for q in pos]).float() if pos else None)


def snapshot_failures(snap: torch.Tensor, golden, layer: int, cap: int) -> tuple[torch.Tensor, torch.Tensor, list]:
    """P.42: a served snapshot [128, 512] vs the golden window KV of positions [cap - 128, cap) at row p % 128: (the
    device rows, the golden rows of the positions the golden holds, failures: rows of no position < cap not zero).
    Test-side only."""
    lo = max(0, cap - SNAP)
    pos, want = golden_window(golden, layer, lo, cap)
    fails = []
    held = {q % SNAP for q in range(lo, cap)}
    empty = [r for r in range(SNAP) if r not in held and bool(snap[r].abs().sum() > 0)]
    if empty:
        fails.append(f"snapshot layer {layer} cap {cap}: rows {empty[:4]}.. of no position < cap are not zero")
    got = snap[pos % SNAP].float() if len(pos) else torch.zeros(0, snap.shape[-1])
    return got, (want if want is not None else torch.zeros(0, snap.shape[-1])), fails


def golden_rows(golden, c: dict, layer: int, length: int) -> tuple[torch.Tensor, int, str]:
    """(golden rows, first row they stand for, comparison) of config ``c`` for the complete groups of tokens
    [0, length). Window KV: a layer with per-chunk dumps (the golden's step_dump_layers) gives every position (the
    chunks' "kv" boundary, comparison "full"); any other layer gives the last min(window, length) positions from its
    ring state as of ``length`` (kv_cache/layer_i[_at_length]: position p at ring row p % window, comparison
    "last-window"; the golden keeps no earlier window KV of that layer). Compressed KV / index keys: the layer's final
    state, every complete group (comparison "full"). Test-side only: never called from the model path."""
    if c["kind"] == "window_kv":
        if all(golden.has_layer(ch, layer) for ch in range(golden.n_chunks)):
            kv = torch.cat([golden.layer(ch, layer)["kv"] for ch in range(golden.n_chunks)])
            return kv[:length].float(), 0, "full"
        ring = golden.state(layer, at=length)["window_kv"]  # snapshot at ``length`` (FileNotFoundError if none)
        ring = ring.reshape(-1, ring.shape[-1])
        n = min(ring.shape[0], length)
        pos = torch.arange(length - n, length)
        return ring[pos % ring.shape[0]].float(), length - n, "last-window"
    src = c.get("of", {}).get(layer, layer)  # (P.39: a re-index layer serves its KV source's index keys)
    return golden.state(src)[c["kind"]][: length // c["ratio"]].float(), 0, "full"


def read_slot_kv_and_check_pcc(table, device_map, slot_id: int, real_len: int, trace_dir, cfg, layer_ids) -> dict:
    """Min PCC per cache kind over [0, real_len) of every served layer vs the bring-up golden at ``trace_dir``
    (table layer k = layer_ids[k]): {window_kv, compress_kv, index_k}. Every served layer is compared; the window KV of
    a layer without per-chunk golden dumps over its last window only (golden_rows), logged per layer."""
    from loguru import logger

    from models.demos.common.bringup.reference.golden import Golden

    golden = Golden(trace_dir)
    names = [table.config_name(i) for i in range(table.num_configs())]
    mins, modes = {}, {}
    # P.42 snapshot: the cap of the request's real end (the runtime's last chunk end in this process; else the
    # caller's 32-aligned length, whose cap end // 64 x 64 is the same, P.49)
    cap = snapshot_cap(SNAPSHOT_ENDS.get(int(slot_id), real_len))
    for c in served_configs(cfg, list(layer_ids)):
        cid = names.index(c["name"])
        for k, layer in zip(c["engine"], c["layers"]):
            if c.get("snapshot"):
                snap = read_rows(table, device_map, cid, k, slot_id, SNAP, c["width"])
                got, want, bad = snapshot_failures(snap, golden, layer, cap)
                for b in bad:  # (rows of no position < cap not zero: the read-back fails the window kind)
                    logger.error(f"  {c['name']}: {b}")
                    mins[c["kind"]] = 0.0
                if want.shape[0]:
                    p = _pcc(got, want)
                    mins[c["kind"]] = min(mins.get(c["kind"], 1.0), p)
                    modes.setdefault(f"{c['name']}:snapshot", []).append(layer)
                    logger.info(f"  {c['name']} layer {layer}: snapshot at cap {cap}, {want.shape[0]} rows PCC {p:.6f}")
                continue
            want, lo, mode = golden_rows(golden, c, layer, real_len)
            got = read_rows(table, device_map, cid, k, slot_id, real_len, c["width"], c.get("fp8", False))
            got = got[lo : lo + want.shape[0]]
            assert got.shape == want.shape, (c["name"], layer, got.shape, want.shape)
            p = _pcc(got, want)
            mins[c["kind"]] = min(mins.get(c["kind"], 1.0), p)
            modes.setdefault(f"{c['name']}:{mode}", []).append(layer)
            logger.info(
                f"  {c['name']} layer {layer} (table layer {k}): {mode} rows [{lo}, {lo + want.shape[0]}) PCC {p:.6f}"
            )
    logger.info(f"[ds41] slot {slot_id} KV comparison per config and kind: {modes}")
    logger.info(f"[ds41] slot {slot_id} KV PCC over [0, {real_len}) of layers {list(layer_ids)} -> {mins}")
    return mins
