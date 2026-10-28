# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""The served KV caches of DeepSeek-V4.1-Flash (serving_contract.md "KV cache", "Adapter and table") and the pieces
the contract tests share: the table configs and their record geometry, the golden rows of each config, the table
rules (tt-d-gen's KV Manager and launch harness), the table read-back a KV Manager does, and the adapter build the
prefill runner does.

Served configs, in this order (the names sort in it: KvChunkAddressTable's name -> config constructor assigns ids in
sorted key order), each present when at least one served layer has it (record = 32 token positions,
tables.py:37; bf16 ROW_MAJOR; one record = one ND shard [1, 1, rows, width] in one DRAM bank):

    name            layers                                  rows / record   width   golden (R.GOLDEN_RUNG)
    0_window_kv     every layer                             32              512     chunk_XX/layer_YY "kv" (dumped
                                                                                    layers), else the ring snapshots
    1_compress_kv_r2  KV sources, compress ratio 2        16              512     kv_cache "compress_kv"
    2_index_k_r2    the same layers                         16              128     kv_cache "index_k"
    3_compress_kv_r1  KV sources, compress ratio 1        32              512     kv_cache "compress_kv"
    4_index_k_r1    the same layers                         32              128     kv_cache "index_k"

A compressed row j stands for tokens [j * r, j * r + r): the record of token positions [p, p + 32) holds rows
[p / r, (p + 32) / r). RoPE dims are the last 64 of a 512 vector and of a 128 index key (inference/model.py
apply_rotary_emb(x[..., -rd:])): the per-channel PCC splits there (kv_dump_compare.tensor_pcc, nope / pe).
"""

from __future__ import annotations

import inspect
import json
import os
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

from models.demos.deepseek_v41_flash_d_p.tests.bringup.contract import server_rules as R

ROPE = 64
CONFIG_ORDER = ("0_window_kv", "1_compress_kv_r2", "2_index_k_r2", "3_compress_kv_r1", "4_index_k_r1")
PREFILL_CHUNK_KWARGS = ("slot_id", "actual_start", "actual_end", "request_id", "d2h_service", "metadata_msg")


class NotBuilt(Exception):
    pass


# ---------------------------------------------------------------- model facts
@lru_cache(1)
def model_args() -> dict:
    """The checkpoint's inference/config.json (compress ratios, KV sources, dims)."""
    from models.demos.common.bringup.reference.golden import hf_path

    return json.loads((Path(hf_path(R._spec())) / "inference" / "config.json").read_text())


def ratio(layer: int) -> int:
    return int(model_args()["compress_ratios"][layer])


def is_kv_source(layer: int) -> bool:
    return layer in model_args()["kv_source_layers"]


def served_prefix() -> list[int]:
    """Layers the runner tests serve: the spec's layers from 0 up to the first gap (the whole model in a full run;
    0-3 in this partial run, whose later layers need golden inputs the runner cannot give). Runner layer k = model
    layer k."""
    sel = R._spec().layers()
    n = 0
    while n < len(sel) and sel[n] == n:
        n += 1
    if n == 0:
        raise ValueError(f"the spec's layers {sel} do not start at layer 0: nothing the runner can serve")
    return list(range(n))


def served_configs(layers: list[int]) -> list[dict]:
    """The table configs for these served layers (runner layer index = model layer), in CONFIG_ORDER."""
    a = model_args()
    head, idx = int(a["head_dim"]), int(a["index_head_dim"])
    out = [{"name": CONFIG_ORDER[0], "kind": "window", "ratio": 1, "rows": 32, "width": head, "layers": list(layers)}]
    for r, (cn, ik) in ((2, CONFIG_ORDER[1:3]), (1, CONFIG_ORDER[3:5])):
        src = [i for i in layers if is_kv_source(i) and ratio(i) == r]
        if src:
            out.append({"name": cn, "kind": "compress", "ratio": r, "rows": 32 // r, "width": head, "layers": src})
            out.append({"name": ik, "kind": "index", "ratio": r, "rows": 32 // r, "width": idx, "layers": src})
    for c in out:
        c["split"] = c["width"] - ROPE
        c["record_bytes"] = c["rows"] * c["width"] * 2
    return out


def layer_configs(layer: int) -> list[dict]:
    """The configs a layer owns (window_kv always; compress / index on a KV source)."""
    return [c for c in served_configs([layer]) if layer in c["layers"]]


def rows_of(cfg: dict, lo_tok: int, hi_tok: int) -> tuple[int, int]:
    """Cache rows of the complete groups in token range [lo_tok, hi_tok)."""
    r = cfg["ratio"]
    return -(-lo_tok // r), hi_tok // r


# ---------------------------------------------------------------- golden
class GoldenKV:
    """The R.GOLDEN_RUNG golden (the spec's shortest full-dump rung: s4096 / chunk 2048 on the seed, s10240 / 5120 on
    galaxy_8x4). The reference is chunk-invariant (DS41 reference docstring, "Determinism"), so a row at position p is
    the golden of any chunk plan, and position p's window KV depends on tokens [0, p] only.

    What the golden holds per layer (a full run keeps per-chunk dumps only for the manifest's step_dump_layers; F164
    dropped the other layers' "in", equal to the previous layer's "out"):
      dumped layer (every chunk dumped)  window KV of every position ("kv"), block input ("in")         -> "full"
      any other layer                    window KV only in its ring snapshots (kv_cache/layer_i[_at_p],
                                         position q at ring row q % 128): the last 128 positions before
                                         each snapshot (metadata state_snapshots) and before seq          -> "last-window"
                                         block input = the previous layer's "out" when that one is dumped
    Compressed KV / index keys: the layer's final state, every complete group ("full"). Every comparison goes through
    ``select`` / ``golden_failures``, which counts what was compared per (config, layer, mode): ``coverage_failures``
    fails a served (config, layer) never compared, so no layer is skipped.

    ``window="last"`` (owner decision B, 2026-10-27, OV-16): a dumped layer's window KV is compared on the positions the
    reference drift measures, whose limit R.state_threshold applies (drift state_window_kv_L*: the ring = the last
    `window` positions): the last min(window, hi - lo) positions of the compared range [lo, hi), mode "last-window".
    Accuracy over every position stays covered by the ladder's per-layer err. ``window="all"``: every position."""

    def __init__(self, window: str = "all"):
        assert window in ("all", "last"), window
        self.window = window
        from models.demos.common.bringup.reference.golden import Golden

        s = R._spec()
        rung = s.rung(R.GOLDEN_RUNG)
        assert (rung["seq"], rung["chunk"]) == (R.MAX_SEQ, R.CHUNK), (rung, R.MAX_SEQ, R.CHUNK)
        self.g = Golden.for_rung(s, R.GOLDEN_RUNG)
        assert sorted(self.g.dumped_chunks) == list(range(self.g.n_chunks)), "the contract needs every chunk dumped"
        self._cat, self._ring, self._dumped = {}, {}, {}
        self.coverage = {}  # (config name, layer) -> {mode: rows compared}

    def tokens(self) -> torch.Tensor:
        return self.g.tokens()

    def dumped(self, layer: int) -> bool:
        """Every chunk of this layer dumped (per-position window KV and block input in the golden)."""
        if layer not in self._dumped:
            self._dumped[layer] = all(self.g.has_layer(c, layer) for c in range(self.g.n_chunks))
        return self._dumped[layer]

    def _boundary(self, layer: int, name: str) -> torch.Tensor:
        key = (layer, name)
        if key not in self._cat:
            self._cat[key] = torch.cat([self.g.layer(c, layer)[name] for c in range(self.g.n_chunks)])
        return self._cat[key]

    def has_block_input(self, layer: int) -> bool:
        return self.dumped(layer) or (layer > 0 and self.dumped(layer - 1))

    def block_input(self, layer: int) -> torch.Tensor:
        """[MAX_SEQ, 4 * 5120 + 4] the layer's block input (streams + pending pre), every position: its own "in", or
        the previous layer's "out" (F164: equal; the golden drops the "in" of a layer without dumps)."""
        if self.dumped(layer):
            return self._boundary(layer, "in")
        if layer > 0 and self.dumped(layer - 1):
            return self._boundary(layer - 1, "out")
        raise KeyError(
            f"golden {self.g.dir}: no block input of layer {layer} (neither it nor layer {layer - 1} dumped)"
        )

    def _window_ring(self, layer: int) -> tuple[torch.Tensor, torch.Tensor]:
        """(positions, rows) of a layer without dumps: every position some ring snapshot of the golden holds. A
        position two snapshots hold takes the final state's row: the snapshot after a shorter prompt (at 10208 on
        galaxy_8x4) is a separate reference run, equal to fp32 noise only (max |diff| 7.7e-4 of 2.7, layer 39)."""
        if layer not in self._ring:
            rows = {}
            for at in sorted(set(self.g.meta.get("state_snapshots", [])) | {self.g.seq}, reverse=True):
                ring = self.g.state(layer, at=None if at == self.g.seq else at)["window_kv"]
                ring = ring.reshape(-1, ring.shape[-1])
                n = ring.shape[0]
                for q in range(max(0, at - n), at):
                    rows.setdefault(q, ring[q % n])
            pos = sorted(rows)
            self._ring[layer] = (
                torch.tensor(pos, dtype=torch.int64),
                torch.stack([rows[q] for q in pos]) if pos else torch.zeros(0, 512),
            )
        return self._ring[layer]

    def ring_size(self) -> int:
        """The reference's sliding window (rows of its window_kv ring state)."""
        if not hasattr(self, "_ring_size"):
            ring = self.g.state(0)["window_kv"]
            self._ring_size = int(ring.reshape(-1, ring.shape[-1]).shape[0])
        return self._ring_size

    def select(self, cfg: dict, layer: int, lo_tok: int, hi_tok: int) -> tuple[np.ndarray, np.ndarray, str]:
        """(local row indices into the rows of the complete groups of [lo_tok, hi_tok), golden rows there float32,
        mode "full" | "last-window")."""
        a, b = rows_of(cfg, lo_tok, hi_tok)
        if cfg["kind"] != "window" or (self.dumped(layer) and self.window == "all"):
            return np.arange(b - a), self.rows(cfg, layer, lo_tok, hi_tok), "full"
        if self.dumped(layer):  # window == "last": the ring the drift measures, the last positions of the range
            first = max(lo_tok, hi_tok - self.ring_size())
            return np.arange(first - lo_tok, hi_tok - lo_tok), self.rows(cfg, layer, first, hi_tok), "last-window"
        pos, ring = self._window_ring(layer)
        keep = (pos >= lo_tok) & (pos < hi_tok)
        return (pos[keep] - lo_tok).numpy(), ring[keep].float().numpy(), "last-window"

    def rows(self, cfg: dict, layer: int, lo_tok: int, hi_tok: int) -> np.ndarray:
        """Golden rows of config `cfg` for the complete groups in token range [lo_tok, hi_tok), float32 (every row:
        the window KV of a dumped layer, or the compressed caches)."""
        if cfg["kind"] == "window":
            t = self._boundary(layer, "kv")[lo_tok:hi_tok]
        else:
            name = "compress_kv" if cfg["kind"] == "compress" else "index_k"
            a, b = rows_of(cfg, lo_tok, hi_tok)
            t = self.g.state(layer)[name][a:b]
        return t.float().numpy()

    def note(self, cfg: dict, layer: int, mode: str, n: int) -> None:
        c = self.coverage.setdefault((cfg["name"], layer), {})
        c[mode] = c.get(mode, 0) + int(n)

    def coverage_failures(self, cfgs: list[dict], what: str) -> list[str]:
        """A served (config, layer) whose rows were never compared with the golden in this test."""
        return [
            f"{what}: {c['name']} layer {layer} never compared with the golden"
            for c in cfgs
            for layer in c["layers"]
            if not sum(self.coverage.get((c["name"], layer), {}).values())
        ]

    def coverage_summary(self) -> str:
        """Per config and mode: the layers and the rows compared (full: every row; last-window: the positions the
        golden's ring snapshots hold)."""
        by = {}
        for (name, layer), modes in sorted(self.coverage.items()):
            for mode, n in modes.items():
                by.setdefault(f"{name}:{mode}", []).append(f"{layer}({n})")
        return "; ".join(f"{k} layers(rows) {' '.join(v)}" for k, v in by.items())


# ---------------------------------------------------------------- records
def decode(raw: bytes, cfg: dict) -> np.ndarray:
    """One record -> [rows, width] float32 (bf16 ROW_MAJOR; for 32-row records this is kv_dump_compare.decode_chunk
    with dtype bf16, storage row-major)."""
    if len(raw) != cfg["record_bytes"]:
        raise ValueError(
            f"{cfg['name']}: record of {len(raw)} B, expected {cfg['record_bytes']} B "
            f"({cfg['rows']} rows x {cfg['width']} bf16, row-major)"
        )
    return (np.frombuffer(raw, dtype="<u2").astype(np.uint32) << 16).view(np.float32).reshape(cfg["rows"], cfg["width"])


def assemble(records: dict, cfg: dict, layer: int, lo_tok: int, hi_tok: int) -> np.ndarray:
    """Rows of the records covering token range [align32(lo), ceil32(hi)) of one layer, in position order.
    records: {(layer, pos): bytes}."""
    out = []
    for pos in range(R.align_down(lo_tok, 32), R.ceil_to(hi_tok, 32), 32):
        if (layer, pos) not in records:
            raise KeyError(f"{cfg['name']}: no record for layer {layer} position {pos}")
        out.append(decode(records[(layer, pos)], cfg))
    return np.concatenate(out) if out else np.zeros((0, cfg["width"]), np.float32)


def valid_and_pad(records: dict, cfg: dict, layer: int, lo_tok: int, hi_tok: int) -> tuple[np.ndarray, np.ndarray]:
    """(rows of the complete groups in [lo_tok, hi_tok), the pad rows after them up to the end of the last record)."""
    rows = assemble(records, cfg, layer, lo_tok, hi_tok)
    base = R.align_down(lo_tok, 32) // cfg["ratio"]
    a, b = rows_of(cfg, lo_tok, hi_tok)
    return rows[a - base : b - base], rows[b - base :]


def pcc_failures(kvc, cfg: dict, got: np.ndarray, want: np.ndarray, layer: int, thr, what: str) -> list[str]:
    """kv_dump_compare.tensor_pcc (nope / pe channels scored separately) -> failure strings. ``thr``: a number (the
    server's own rule) or R.state_threshold()'s per (config, layer) limit."""
    thr = thr(cfg, layer) if callable(thr) else thr
    if got.shape != want.shape:
        return [f"{what}: {cfg['name']} layer {layer}: shape {got.shape} vs golden {want.shape}"]
    if not got.size:
        return []
    out = []
    for f in kvc.tensor_pcc(got.astype(np.float32), want.astype(np.float32), layer, thr, cfg["split"]):
        if not f["passed"]:
            out.append(f"{what}: {cfg['name']} layer {layer} {f['channel']} pcc {f['pcc']} < {thr}")
    return out


def golden_failures(kvc, gk: GoldenKV, cfg: dict, got: np.ndarray, layer: int, lo: int, hi: int, thr, what: str):
    """``got`` (the rows of the complete groups of [lo, hi)) vs the golden rows ``gk.select`` has there: every row of a
    dumped layer or a compressed cache; for another layer's window KV the positions in its ring snapshots. Counted in
    ``gk.coverage``."""
    idx, want, mode = gk.select(cfg, layer, lo, hi)
    if mode == "full":
        gk.note(cfg, layer, mode, want.shape[0])
        return pcc_failures(kvc, cfg, got, want, layer, thr, what)
    a, b = rows_of(cfg, lo, hi)
    if got.shape[0] != b - a:
        return [f"{what}: {cfg['name']} layer {layer}: {got.shape[0]} rows, expected {b - a}"]
    gk.note(cfg, layer, mode, len(idx))
    return pcc_failures(kvc, cfg, got[idx], want, layer, thr, f"{what} ({mode} {len(idx)} rows)")


# ---------------------------------------------------------------- table (CPU, on the exported .pb)
def parse_table(path: str):
    from launch_harness.kv_chunk_address_table_pb2 import KvChunkAddressTable

    R.harness()
    t = KvChunkAddressTable()
    t.ParseFromString(Path(path).read_bytes())
    return t


def table_entries(t, config_idx: int) -> dict:
    """{(slot, layer, position): (noc_addr, size_bytes, device_group_index)} of one config, as the KV Manager loads it
    (tt-metal import_from_protobuf_file, proto_kv_chunk_table.cpp:103): UNROLLED entries (a later entry for a cell
    replaces an earlier one, tables.py:40-44), or STRIDED_ROWS runs (chunk c of a row at base_noc_addr + addr_stride *
    (c / chunk_step) of the run with start_chunk = c % chunk_step). Raises ValueError where the import would."""
    cfg = t.configs[config_idx]
    if cfg.compression == 0:
        return {
            (e.slot, e.layer, e.position): (e.noc_addr, e.size_bytes, e.device_group_index)
            for e in t.entries
            if e.config_idx == config_idx
        }
    if cfg.compression != 1:
        raise ValueError(f"unknown chunk compression {cfg.compression}")
    npc = -(-cfg.max_sequence_length // cfg.chunk_n_tokens)
    rows = {}
    for r in t.runs:
        if r.config_idx != config_idx:
            continue
        where = f"run (slot {r.slot}, layer {r.layer}, residue {r.start_chunk})"
        if not 1 <= r.chunk_step <= npc or r.count == 0 or r.start_chunk >= r.chunk_step:
            raise ValueError(f"{where}: malformed (chunk_step {r.chunk_step}, count {r.count})")
        if r.count != -(-(npc - r.start_chunk) // r.chunk_step):
            raise ValueError(f"{where}: count {r.count} does not tile the {npc}-chunk row")
        row = rows.setdefault((r.slot, r.layer), {})
        if r.start_chunk in row:
            raise ValueError(f"{where}: duplicate residue")
        row[r.start_chunk] = r
    out = {}
    for (slot, layer), row in rows.items():
        step = next(iter(row.values())).chunk_step
        if set(row) != set(range(step)):
            raise ValueError(f"row (slot {slot}, layer {layer}): runs do not cover residues 0..{step - 1}")
        for c in range(npc):
            r = row[c % step]
            addr = (r.base_noc_addr + r.addr_stride * (c // step)) & 0xFFFFFFFFFFFFFFFF
            out[(slot, layer, c * cfg.chunk_n_tokens)] = (addr, r.size_bytes, r.device_group_index)
    return out


def table_rules(table_path: str, layers: list[int], num_users: int, bank_limits: dict | None = None) -> list[str]:
    """The table rules, on the exported protobuf. From tt-d-gen:
      tables.read_table accepts it (chunk_n_tokens 32, compression 0/1, every chip has an owner host: tables.py:27-59);
      config 0 covers exactly the served layers (tables.py:67-74), every config >= max_slots slots (75-77);
      config names unique, one num_slots for all configs (kv_chunk_table.cpp:41-72), one host per (config, layer)
      (100-112); the configs are the served ones in CONFIG_ORDER (the KVM pairs source and destination configs by name,
      id and chunk_n_tokens: kv_chunk_table_loader.cpp:26-51), each covering exactly its layers (a layer covered by one
      side only fails the migration: migration_strategy_builder.cpp:107-121);
      every (slot, layer, position < max_seq) record present, size = chunk_size_bytes = one record of the config
      (migration_strategy_builder.cpp:143-166; size 0 = no entry: proto_kv_chunk_table.cpp:79-82);
      the address: bank = noc_addr >> 32, offset = noc_addr & 0xFFFFFFFF (kv_manager/device_io/src/noc_addr.cpp:12-18),
      offset % 64 == 0 (noc_address_resolver.cpp:18-22), [offset, offset + size) inside the bank's allocation
      (bank_limits {config name: (lo, hi)} when known, else below 2**32), and no two records overlapping on one chip and
      bank (one record = one address + size in ONE bank, so a record must not share bytes with another).
    Returns failures."""
    tables, _, _ = R.harness()
    fails = []
    try:
        hc = tables.read_table(table_path)
    except Exception as e:
        return [f"tables.read_table rejects the table: {e}"]
    want = served_configs(layers)
    t = parse_table(table_path)
    names = [c.name for c in t.configs]
    if names != [c["name"] for c in want]:
        fails.append(f"configs {names}, expected {[c['name'] for c in want]} (serving_contract.md 'KV cache')")
    if hc[0]["layers"] != layers:
        fails.append(f"config 0 covers layers {hc[0]['layers']}, expected exactly {layers}")
    slots = {c.num_slots for c in t.configs}
    if len(slots) != 1:
        fails.append(f"num_slots differs across configs: {sorted(slots)}")
    hosts = {h.host_name for h in t.fabric_node_hosts}
    if len(hosts) != 1:
        fails.append(f"the table names {len(hosts)} hosts {sorted(hosts)}: every (config, layer) on one host")
    chips = {}  # (mesh, chip, bank) -> [(lo, hi, key)]
    groups = {i: [(n.mesh_id, n.chip_id) for n in g.fabric_node_ids] for i, g in enumerate(t.device_groups)}
    for idx, (cfg, w) in enumerate(zip(t.configs, want)):
        name = cfg.name
        if cfg.chunk_n_tokens != R.RECORD_TOKENS:
            fails.append(f"{name}: chunk_n_tokens {cfg.chunk_n_tokens} != {R.RECORD_TOKENS}")
        if cfg.num_slots < num_users:
            fails.append(f"{name}: {cfg.num_slots} slots < max_slots {num_users}")
        if cfg.max_sequence_length < R.MAX_SEQ:
            fails.append(f"{name}: max_sequence_length {cfg.max_sequence_length} < max_seq {R.MAX_SEQ}")
        if name != w["name"]:
            continue
        if cfg.chunk_size_bytes != w["record_bytes"]:
            fails.append(
                f"{name}: chunk_size_bytes {cfg.chunk_size_bytes} != {w['record_bytes']} "
                f"({w['rows']} rows x {w['width']} bf16 per 32 tokens)"
            )
        if hc[idx]["layers"] != w["layers"]:
            fails.append(f"{name}: covers layers {hc[idx]['layers']}, expected exactly {w['layers']}")
        try:
            entries = table_entries(t, idx)
        except ValueError as e:
            fails.append(f"{name}: {e}")
            continue
        miss = [
            (s, l, p)
            for s in range(num_users)
            for l in w["layers"]
            for p in range(0, R.MAX_SEQ, R.RECORD_TOKENS)
            if (s, l, p) not in entries or entries[(s, l, p)][1] == 0
        ]
        if miss:
            fails.append(f"{name}: {len(miss)} (slot, layer, position) records missing, e.g. {miss[:3]}")
        wrong = [k for k, e in entries.items() if e[1] and e[1] != cfg.chunk_size_bytes]
        if wrong:
            fails.append(f"{name}: {len(wrong)} records not chunk_size_bytes {cfg.chunk_size_bytes}, e.g. {wrong[:3]}")
        lim = (bank_limits or {}).get(name)
        bad_align, bad_bank = [], []
        for k, (addr, size, dg) in entries.items():
            if not size:
                continue
            bank, off = addr >> 32, addr & 0xFFFFFFFF
            if off % R.DRAM_ALIGN:
                bad_align.append(k)
            hi = lim[1] if lim else 1 << 32
            lo = lim[0] if lim else 0
            if off < lo or off + size > hi:
                bad_bank.append((k, hex(off), size))
            for chip in groups.get(dg, []):
                chips.setdefault((*chip, bank), []).append((off, off + size, (name, *k)))
        if bad_align:
            fails.append(
                f"{name}: {len(bad_align)} record offsets not {R.DRAM_ALIGN}-byte aligned, e.g. {bad_align[:3]}"
            )
        if bad_bank:
            fails.append(f"{name}: {len(bad_bank)} records outside the bank's allocation {lim}, e.g. {bad_bank[:3]}")
    for (m, c, bank), iv in chips.items():
        iv.sort()
        for (a0, a1, ka), (b0, b1, kb) in zip(iv, iv[1:]):
            if b0 < a1:
                fails.append(
                    f"chip ({m}, {c}) bank {bank}: records {ka} and {kb} overlap ([{a0:#x}, {a1:#x}) / {b0:#x})"
                )
                break
    return fails


# ---------------------------------------------------------------- table read-back (on the device host)
class TableReader:
    """KV records through the exported table, as the KV Manager reads them (out of band, over UMD)."""

    def __init__(self, table_path: str, dmap: dict):
        import ttnn

        self.D = ttnn.experimental.disaggregation
        self.table = self.D.import_from_protobuf_file(table_path)
        self.dmap = dmap
        self.cid = {self.table.config_name(i): i for i in range(self.table.num_configs())}
        self.replica_mismatch = []

    def record(self, name: str, slot: int, layer: int, pos: int) -> bytes:
        """One 32-token record; every chip of its device group must hold the same bytes (the KVM reads any one)."""
        loc = self.table.lookup(layer, pos, slot, self.cid[name])
        if not loc.size_bytes:
            raise KeyError(f"{name}: no table entry for slot {slot} layer {layer} position {pos}")
        raws = []
        for n in self.table.get_device_group(loc.device_group_index).fabric_node_ids:
            key = (int(n.mesh_id), int(n.chip_id))
            if key in self.dmap:
                raws.append(bytes(self.D.read_dram_umd(self.dmap[key], loc.noc_addr, loc.size_bytes)))
        if not raws:
            raise KeyError(f"no chip of device group {loc.device_group_index} in the device map")
        if any(r != raws[0] for r in raws[1:]):
            self.replica_mismatch.append((name, slot, layer, pos))
        return raws[0]

    def records(self, name: str, slot: int, layers, lo: int, hi: int) -> dict:
        return {
            (l, p): self.record(name, slot, l, p)
            for l in layers
            for p in range(R.align_down(lo, 32), R.ceil_to(hi, 32), 32)
        }

    def dump(self, out_dir: Path, slot: int, layers: list[int], lo: int, hi: int) -> None:
        """kv_dram_poke --dump files <out>/<config>/s<slot>_l<layer>_p<pos>.bin for the records covering [lo, hi) of
        every config the layers own (one directory per config, as kv_dump_compare expects)."""
        for cfg in served_configs(layers):
            d = Path(out_dir) / cfg["name"]
            d.mkdir(parents=True, exist_ok=True)
            for (l, p), raw in self.records(cfg["name"], slot, cfg["layers"], lo, hi).items():
                (d / R.record_name(slot, l, p)).write_bytes(raw)


def load_dump(dir_: Path, cfg: dict, slot: int) -> dict:
    """{(layer, pos): bytes} of one config's dump (kv_dump_compare.load_dump with this config's record size)."""
    _, kvc, _ = R.harness()
    return kvc.load_dump(str(Path(dir_) / cfg["name"]), slot=slot, chunk_bytes=cfg["record_bytes"])


def bytes_equal(a: dict, b: dict, cfg: dict, layers, lo_tok: int, hi_tok: int) -> list[str]:
    """Byte compare of the rows of complete groups in [lo_tok, hi_tok) (kv_dump_compare bytecmp, per config rows)."""
    out = []
    for l in layers:
        ra, _ = valid_and_pad(a, cfg, l, lo_tok, hi_tok)
        rb, _ = valid_and_pad(b, cfg, l, lo_tok, hi_tok)
        if ra.tobytes() != rb.tobytes():
            bad = np.nonzero((ra.view(np.uint32) != rb.view(np.uint32)).any(axis=1))[0]
            out.append(
                f"{cfg['name']} layer {l}: {bad.size} rows of tokens [{lo_tok}, {hi_tok}) differ, first row {bad[:3]}"
            )
    return out


# ---------------------------------------------------------------- the adapter, as prefill_runner.main uses it
def adapter_name() -> str:
    return R._spec().get("contract.adapter") or R._spec().model


def check_runtime(runtime) -> list[str]:
    """What prefill_runner.py calls on the runtime (437-447 prefill_chunk, 689 compile, 762-765 / 847
    set_d2h_ack_service, 859 set_layer_completion_sink, 936 kv_migration_stages, 955-990 build_kv_chunk_table), as
    missing items."""
    miss = []
    for name in (
        "compile",
        "prefill_chunk",
        "set_layer_completion_sink",
        "set_d2h_ack_service",
        "build_kv_chunk_table",
        "kv_migration_stages",
    ):
        if not callable(getattr(runtime, name, None)):
            miss.append(f"runtime.{name}()")
    if not hasattr(runtime, "mesh_device") or not hasattr(runtime, "config"):
        miss.append("runtime.mesh_device / runtime.config (is_first_rank, is_last_rank, use_trace, first_layer_idx)")
    if callable(getattr(runtime, "prefill_chunk", None)):
        p = inspect.signature(runtime.prefill_chunk).parameters
        if not any(v.kind == v.VAR_KEYWORD for v in p.values()):
            lost = [k for k in PREFILL_CHUNK_KWARGS if k not in p]
            if lost:
                miss.append(f"prefill_chunk(input, kv, *, {', '.join(PREFILL_CHUNK_KWARGS)}) lacks {lost}")
    if callable(getattr(runtime, "build_kv_chunk_table", None)):
        p = inspect.signature(runtime.build_kv_chunk_table).parameters
        if not any(v.kind == v.VAR_KEYWORD for v in p.values()):
            lost = [k for k in ("first_layer_idx", "num_my_layers", "stage_layouts") if k not in p]
            if lost:
                miss.append(
                    f"build_kv_chunk_table(kv, path, *, first_layer_idx, num_my_layers, stage_layouts) lacks {lost}"
                )
    return miss


def check_kv(kv, layers: list[int]) -> list[str]:
    """The KvCaches handle names its served tensors: kv.by_config {config name: ttnn.Tensor}."""
    by = getattr(kv, "by_config", None)
    if not isinstance(by, dict):
        return ["KvCaches.by_config {config name: device tensor} (serving_contract.md 'KV cache')"]
    want = [c["name"] for c in served_configs(layers)]
    if list(by) != want:
        return [f"KvCaches.by_config has {list(by)}, expected {want}"]
    return []


def engine_env(num_layers: int, num_users: int) -> dict:
    s = R._spec()
    env = {
        "PREFILL_MODEL": adapter_name(),
        "PREFILL_SP": str(R.SP),
        "PREFILL_TP": str(R.TP),
        "PREFILL_CHUNK_SIZE": str(R.CHUNK),
        "PREFILL_MAX_SEQ_LEN": str(R.MAX_SEQ),
        "PREFILL_NUM_LAYERS": str(num_layers),
        "PREFILL_NUM_USERS": str(num_users),
        "PREFILL_FABRIC_MODE": fabric_mode(s),
    }
    os.environ.update(env)
    return env


def fabric_mode(s) -> str:
    """The runner's PREFILL_FABRIC_MODE for the spec's fabric (runner_utils.open_mesh_device defaults to 2d_torus_xy,
    which this box does not have; the owner's rule: always a 2D fabric)."""
    fc = str((s.get("box.device_params") or {}).get("fabric_config", "FABRIC_2D"))
    return {"FABRIC_2D": "2d", "FABRIC_2D_TORUS_X": "2d_torus_x", "FABRIC_2D_TORUS_Y": "2d_torus_y"}.get(
        fc, "2d_torus_xy" if fc == "FABRIC_2D_TORUS_XY" else "2d"
    )


def build(mesh_device, layers: list[int], num_users: int):
    """adapter -> runtime -> engine-owned KV cache -> compile, as prefill_runner.main does (lines 626-689)."""
    engine_env(len(layers), num_users)
    from models.demos.common.prefill.adapter import PrefillRunParams, get_adapter

    try:
        adapter = get_adapter(adapter_name())
    except Exception as e:
        raise NotBuilt(f"adapter {adapter_name()!r} not registered / importable (adapter.py ADAPTER_PATHS): {e}")
    hf = adapter.load_hf_config()
    hf.max_seq_len = R.MAX_SEQ
    params = PrefillRunParams(
        mesh_shape=(R.SP, R.TP),
        num_layers=len(layers),
        first_layer_idx=0,
        is_first_rank=True,
        is_last_rank=True,
        max_seq_len=R.MAX_SEQ,
        chunk_size=R.CHUNK,
        num_users=num_users,
        capacity_factor=8,
        num_links=2,
        gate_mode_name=adapter.default_gate_mode,
        kv_only_last_layer=True,
        weight_cache_path=adapter.weight_cache_path((R.SP, R.TP)),
    )
    runtime = adapter.build_runtime(mesh_device=mesh_device, hf_config=hf, params=params)
    miss = check_runtime(runtime)
    if miss:
        raise NotBuilt("; ".join(miss))
    kv = adapter.allocate_kv_cache(mesh_device=mesh_device, hf_config=hf, params=params)
    miss = check_kv(kv, layers)
    if miss:
        raise NotBuilt("; ".join(miss))
    runtime.compile(kv)
    return adapter, runtime, kv


def export_table(runtime, kv, mesh_device, path: str, num_layers: int) -> str:
    """The runner's migration path (PREFILL_ENABLE_MIGRATION=1, mock; prefill_runner.py 936-960): stages -> gathered
    layouts -> table."""
    from models.demos.common.prefill.runners.migration import allgather_kv_stage_layouts

    stages = runtime.kv_migration_stages(kv, 0, num_layers)
    layouts = allgather_kv_stage_layouts(mesh_device, stages, (R.SP, R.TP))
    out = runtime.build_kv_chunk_table(kv, path, first_layer_idx=0, num_my_layers=num_layers, stage_layouts=layouts)
    return out or path


def device_map(mesh_device, path: str) -> dict:
    from models.demos.common.prefill.runners.migration import serialize_device_map

    serialize_device_map(mesh_device, path)
    return read_device_map(path)


def read_device_map(path: str) -> dict:
    text = Path(path).read_text()
    try:
        return {tuple(int(x) for x in k.split(":")): int(v) for k, v in json.loads(text).items()}
    except json.JSONDecodeError:  # "mesh chip unique_id" lines
        out = {}
        for line in text.splitlines():
            p = line.replace(":", " ").split()
            if len(p) >= 3:
                out[(int(p[0]), int(p[1]))] = int(p[2])
        return out


def payload_tensor(mesh_device, payload: np.ndarray):
    """The H2D global tensor [sp, 1, chunk / sp] uint32 ROW_MAJOR DRAM, sharded over mesh axis 0, replicated over
    axis 1 (prefill_runner.py H2D_MAPPER_CONFIG)."""
    import ttnn

    return ttnn.from_torch(
        torch.from_numpy(payload.astype(np.int64)).to(torch.uint32),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.create_mesh_mapper(
            mesh_device, ttnn.MeshMapperConfig(placements=[ttnn.PlacementShard(0), ttnn.PlacementReplicate()])
        ),
    )


# ---------------------------------------------------------------- the cache tensors' memory config (layout test)
def layout_failures(tensor, cfg: dict, num_banks: int) -> tuple[list[str], tuple[int, int] | None]:
    """The record rule (serving_contract.md "KV cache"): bf16 ROW_MAJOR, DRAM ND-sharded with shard [1, 1, rows, width],
    ROUND_ROBIN_1D over every DRAM bank, never DRAM interleaved. Returns (failures, (lo, hi) of each bank's
    allocation: base address .. base + ceil(shards / banks) * shard bytes)."""
    import ttnn

    name = cfg["name"]
    fails = []
    mc = tensor.memory_config()
    if mc.buffer_type != ttnn.BufferType.DRAM:
        fails.append(f"{name}: buffer type {mc.buffer_type}, not DRAM")
    if tensor.dtype != ttnn.bfloat16:
        fails.append(f"{name}: dtype {tensor.dtype}, the served dtype is bfloat16")
    if tensor.layout != ttnn.ROW_MAJOR_LAYOUT:
        fails.append(f"{name}: layout {tensor.layout}, records are bf16 ROW_MAJOR rows")
    nd = mc.nd_shard_spec
    if nd is None:
        fails.append(f"{name}: not ND-sharded ({mc.memory_layout}): an interleaved cache spreads a record over banks")
        return fails, None
    shape = list(nd.shard_shape)
    if shape[-2:] != [cfg["rows"], cfg["width"]] or any(int(d) != 1 for d in shape[:-2]):
        fails.append(f"{name}: shard shape {shape}, expected [1, 1, {cfg['rows']}, {cfg['width']}]")
    if nd.shard_distribution_strategy != ttnn.ShardDistributionStrategy.ROUND_ROBIN_1D:
        fails.append(f"{name}: shard distribution {nd.shard_distribution_strategy}, expected ROUND_ROBIN_1D")
    cores = nd.num_cores() if callable(nd.num_cores) else nd.num_cores
    if int(cores) != num_banks:
        fails.append(f"{name}: sharded over {cores} banks, expected all {num_banks} DRAM banks")
    page = int(tensor.buffer_aligned_page_size())
    pages = int(tensor.buffer_num_pages())
    shards = -(-pages // cfg["rows"])
    base = int(tensor.buffer_address())
    return fails, (base, base + -(-shards // num_banks) * cfg["rows"] * page)
