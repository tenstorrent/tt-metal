# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Shared pieces of the adapter-level contract tests: the prefill runner's calls into the adapter / runtime, the
table read-back a KV Manager does (table lookup -> read_dram_umd), and the launch harness's table rules.

Everything a runtime must offer is checked up front, so a missing piece fails as "not built" instead of a TypeError
deep in the runner (models/demos/common/prefill/runners/prefill_runner.py is the caller these mirror)."""

from __future__ import annotations

import inspect
import json
import os
from pathlib import Path

import numpy as np

from models.demos.xing40_a4b_d_p.tests.bringup.contract import server_rules as R

MODEL = "xing40_a4b_d_p"

PREFILL_CHUNK_KWARGS = ("slot_id", "actual_start", "actual_end", "request_id", "d2h_service", "metadata_msg")


class NotBuilt(Exception):
    pass


def check_runtime(runtime) -> list[str]:
    """What prefill_runner.py calls on the runtime (lines 318-327, 534, 693, 767-769, 796-830), as missing items."""
    miss = []
    for name in ("compile", "prefill_chunk", "set_layer_completion_sink", "build_kv_chunk_table"):
        if not callable(getattr(runtime, name, None)):
            miss.append(f"runtime.{name}()")
    if not (hasattr(runtime, "kv_migration_stages") or hasattr(runtime, "kv_migration_base_address")):
        miss.append("runtime.kv_migration_base_address(kv) or runtime.kv_migration_stages(kv, first, n)")
    if not hasattr(runtime, "mesh_device") or not hasattr(runtime, "config"):
        miss.append("runtime.mesh_device / runtime.config (is_last_rank, use_trace, first_layer_idx, num_layers)")
    if callable(getattr(runtime, "prefill_chunk", None)):
        p = inspect.signature(runtime.prefill_chunk).parameters
        lost = [k for k in PREFILL_CHUNK_KWARGS if k not in p]
        if lost:
            miss.append(f"prefill_chunk(input, kv, {', '.join(f'{k}=' for k in PREFILL_CHUNK_KWARGS)}) lacks {lost}")
    if callable(getattr(runtime, "build_kv_chunk_table", None)):
        p = inspect.signature(runtime.build_kv_chunk_table).parameters
        need = ["first_layer_idx", "num_my_layers"]
        lost = [k for k in need if k not in p] + (
            [] if ("stage_layout" in p or "stage_layouts" in p) else ["stage_layout"]
        )
        if lost and not any(v.kind == v.VAR_KEYWORD for v in p.values()):
            miss.append(
                "build_kv_chunk_table(kv, path, *, first_layer_idx, num_my_layers, stage_layout) lacks "
                f"{lost} (the runner's migration path calls it with those, prefill_runner.py:796-830)"
            )
    return miss


# The runner opens the fabric with max_packet_payload_size_bytes = model_config.FABRIC_PAYLOAD_SIZE
# (runner_utils.py:41-53). Below the fabric's own default (4352 B, what every bring-up test ran with) a CCL whose page is
# an fp32 tile (4096 B, e.g. the mHC [S/4, 32] fp32 all_reduce) gets 0 pages per packet: SIGFPE building reduce_scatter.
# Blackhole's ceiling is 15232 B (tt_metal/fabric/erisc_datamover_builder.hpp:460-483).
MIN_FABRIC_PAYLOAD, MAX_FABRIC_PAYLOAD = 4352, 15232


def check_model_config(cfg) -> list[str]:
    n = int(getattr(cfg, "FABRIC_PAYLOAD_SIZE", 0))
    if not MIN_FABRIC_PAYLOAD <= n <= MAX_FABRIC_PAYLOAD:
        return [
            f"model_config.FABRIC_PAYLOAD_SIZE = {n}: the runner's fabric packet payload must be in "
            f"[{MIN_FABRIC_PAYLOAD}, {MAX_FABRIC_PAYLOAD}] B (an fp32 tile is 4096 B; below it reduce_scatter divides by 0)"
        ]
    return []


def run_params(num_layers: int, num_users: int):
    from models.demos.common.prefill.adapter import PrefillRunParams

    return PrefillRunParams(
        mesh_shape=(R.SP, R.TP),
        num_layers=num_layers,
        first_layer_idx=0,
        is_first_rank=True,
        is_last_rank=True,
        max_seq_len=R.MAX_SEQ,
        chunk_size=R.CHUNK,
        num_users=num_users,
        capacity_factor=8,
        num_links=2,
        gate_mode_name="DEVICE_FP32",
        kv_only_last_layer=True,
        weight_cache_path=None,
    )


def build(mesh_device, num_layers: int, num_users: int):
    """adapter -> runtime -> engine-owned KV cache -> compile, as prefill_runner.main does (lines 497-534)."""
    from models.demos.common.prefill.adapter import get_adapter

    try:
        adapter = get_adapter(MODEL)
    except Exception as e:
        raise NotBuilt(f"adapter {MODEL} not registered / importable: {e}")
    hf = adapter.load_hf_config()
    hf.max_seq_len = R.MAX_SEQ
    params = run_params(num_layers, num_users)
    runtime = adapter.build_runtime(mesh_device=mesh_device, hf_config=hf, params=params)
    miss = check_runtime(runtime)
    if miss:
        raise NotBuilt("; ".join(miss))
    kv = adapter.allocate_kv_cache(mesh_device=mesh_device, hf_config=hf, params=params)
    runtime.compile(kv)
    return adapter, runtime, kv


def export_table(runtime, kv, mesh_device, path: str, num_layers: int) -> str:
    """The runner's migration path (PREFILL_ENABLE_MIGRATION=1, mock): stages -> allgather layouts -> table."""
    from models.demos.common.prefill.runners.migration import KvCacheStage, allgather_kv_stage_layouts

    if hasattr(runtime, "kv_migration_stages"):
        stages = runtime.kv_migration_stages(kv, 0, num_layers)
        kw = {"stage_layouts": allgather_kv_stage_layouts(mesh_device, stages, (R.SP, R.TP))}
    else:
        stages = [KvCacheStage(runtime.kv_migration_base_address(kv), 0, num_layers)]
        kw = {"stage_layout": allgather_kv_stage_layouts(mesh_device, stages, (R.SP, R.TP))[0]}
    out = runtime.build_kv_chunk_table(kv, path, first_layer_idx=0, num_my_layers=num_layers, **kw)
    return out or path


def device_map(mesh_device, path: str) -> dict:
    from models.demos.common.prefill.runners.migration import serialize_device_map

    serialize_device_map(mesh_device, path)
    return {tuple(int(x) for x in k.split(":")): int(v) for k, v in json.loads(Path(path).read_text()).items()}


def payload_tensor(mesh_device, payload: np.ndarray):
    """The H2D global tensor [sp, 1, chunk/sp] uint32 ROW_MAJOR DRAM, sharded over axis 0, replicated over axis 1
    (prefill_runner.py:60 H2D_MAPPER_CONFIG, runner_utils.make_global_spec)."""
    import torch

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


class TableReader:
    """KV records through the exported table, as the KV Manager reads them (out of band, over UMD)."""

    def __init__(self, table_path: str, dmap: dict, config_id: int = 0):
        import ttnn

        self.D = ttnn.experimental.disaggregation
        self.table = self.D.import_from_protobuf_file(table_path)
        self.dmap, self.cid = dmap, config_id
        self.replica_mismatch = []

    def record(self, slot: int, layer: int, pos: int) -> bytes:
        """One 32-token record; every chip of its device group must hold the same bytes (the KVM may read any)."""
        loc = self.table.lookup(layer, pos, slot, self.cid)
        nodes = list(self.table.get_device_group(loc.device_group_index).fabric_node_ids)
        raws = []
        for n in nodes:
            key = (int(n.mesh_id), int(n.chip_id))
            if key in self.dmap:
                raws.append(bytes(self.D.read_dram_umd(self.dmap[key], loc.noc_addr, loc.size_bytes)))
        if not raws:
            raise KeyError(f"no chip of device group {loc.device_group_index} in the device map")
        if any(r != raws[0] for r in raws[1:]):
            self.replica_mismatch.append((slot, layer, pos))
        return raws[0]

    def dump(self, out_dir: Path, slot: int, layers, lo: int, hi: int) -> None:
        """kv_dram_poke --dump files s<slot>_l<layer>_p<pos>.bin for the records covering [lo, hi)."""
        out_dir.mkdir(parents=True, exist_ok=True)
        for layer in layers:
            for pos in range(R.align_down(lo, 32), R.ceil_to(hi, 32), 32):
                (out_dir / R.record_name(slot, layer, pos)).write_bytes(self.record(slot, layer, pos))


# ---------------------------------------------------------------- table rules (CPU, on the exported .pb)
def table_entries(t, config_idx: int = 0) -> dict:
    """{(slot, layer, position): (noc_addr, size_bytes, device_group_index)} for one config of a parsed
    KvChunkAddressTable, as the KV Manager sees it. The KVM loads the table with tt-metal's import_from_protobuf_file
    (kv_manager/src/control_plane/maps/proto_kv_chunk_table.cpp:103); for a STRIDED_ROWS config that import
    (tt_metal/impl/internal/disaggregation/kv_chunk_address_table_protobuf.cpp:498-560) takes the runs only, rejects a
    malformed row, and resolves chunk c of a row to the run of residue r = c % chunk_step at
    base_noc_addr + addr_stride * (c / chunk_step) (kv_chunk_address_table.cpp StridedRowMap::lookup). Raises
    ValueError with the import's reason when the runs would not load."""
    cfg = t.configs[config_idx]
    if cfg.compression == 0:  # UNROLLED: explicit entries, a later one replaces an earlier one
        return {
            (e.slot, e.layer, e.position): (e.noc_addr, e.size_bytes, e.device_group_index)
            for e in t.entries
            if e.config_idx == config_idx
        }
    if cfg.compression != 1:
        raise ValueError(f"unknown chunk compression {cfg.compression} (the import fails closed)")
    npc = -(-cfg.max_sequence_length // cfg.chunk_n_tokens)
    rows = {}
    for r in t.runs:
        if r.config_idx != config_idx:
            continue
        where = f"run (slot {r.slot}, layer {r.layer}, residue {r.start_chunk})"
        if not 1 <= r.chunk_step <= npc:
            raise ValueError(f"{where}: chunk_step {r.chunk_step} out of [1, {npc}]")
        if r.count == 0:
            raise ValueError(f"{where}: count 0")
        if r.layer >= cfg.num_layers or r.slot >= cfg.num_slots:
            raise ValueError(f"{where}: layer / slot out of range")
        if r.start_chunk >= r.chunk_step:
            raise ValueError(f"{where}: start_chunk >= chunk_step {r.chunk_step}")
        if r.count != -(-(npc - r.start_chunk) // r.chunk_step):
            raise ValueError(f"{where}: count {r.count} does not tile the {npc}-chunk row")
        row = rows.setdefault((r.slot, r.layer), {})
        if row and any(o.chunk_step != r.chunk_step or o.size_bytes != r.size_bytes for o in row.values()):
            raise ValueError(f"{where}: chunk_step / size_bytes differ from the row's other runs")
        if r.start_chunk in row:
            raise ValueError(f"{where}: duplicate residue")
        row[r.start_chunk] = r
    if not rows:
        raise ValueError("tagged STRIDED_ROWS but has no runs")
    out = {}
    for (slot, layer), row in rows.items():
        step = next(iter(row.values())).chunk_step
        if set(row) != set(range(step)):
            raise ValueError(f"row (slot {slot}, layer {layer}): runs do not cover residues 0..{step - 1}")
        for c in range(npc):
            r = row[c % step]
            addr = (r.base_noc_addr + r.addr_stride * (c // step)) & 0xFFFFFFFFFFFFFFFF  # uint64 wrap, as in C++
            out[(slot, layer, c * cfg.chunk_n_tokens)] = (addr, r.size_bytes, r.device_group_index)
    return out


def bank_overlaps(entries: dict) -> list[str]:
    """No two records in one DRAM bank of one device group overlap. noc_addr = (bank << 32) | local address
    (tt_metal/impl/internal/disaggregation/noc_addr.hpp addr_channel / addr_local); a record must also stay in its
    bank (local + size <= 2**32)."""
    fails, banks = [], {}
    for k, (addr, size, dg) in entries.items():
        if not size:
            continue
        lo = addr & 0xFFFFFFFF
        if lo + size > 1 << 32:
            fails.append(f"record {k} [{lo:#x}, +{size}) runs past the end of bank {addr >> 32}")
        banks.setdefault((dg, addr >> 32), []).append((lo, lo + size, k))
    for (dg, bank), iv in banks.items():
        iv.sort()
        for (a0, a1, ka), (b0, b1, kb) in zip(iv, iv[1:]):
            if b0 < a1:
                fails.append(
                    f"records {ka} [{a0:#x}, {a1:#x}) and {kb} [{b0:#x}, {b1:#x}) overlap in device group {dg} "
                    f"bank {bank}"
                )
                break
    return fails


def table_rules(table_path: str, num_layers: int, num_users: int) -> tuple[list[str], dict]:
    """The launch harness's table rules (tables.read_table / paired_tables / layout) and the KV Manager's migration
    plan rules (migration_strategy_builder.cpp:103-170) on the exported table. Returns (failures, geometry)."""
    tables, _ = R.harness()
    fails = []
    try:
        configs = tables.read_table(table_path)
    except Exception as e:
        return [f"tables.read_table rejects the table: {e}"], {}
    geom = {}
    try:
        geom = tables.layout(MODEL, 0, configs[0]["chunk_bytes"])
    except Exception as e:
        fails.append(f"tables.layout: {e} (a 576-wide MLA cache is bfp8 TILE 19584 B or bf16 row-major 36864 B)")
    c0 = configs[0]
    if c0["layers"] != list(range(num_layers)):
        fails.append(f"config 0 covers layers {c0['layers'][:3]}..{c0['layers'][-3:]}, not exactly 0..{num_layers - 1}")
    if c0["slots"] < num_users:
        fails.append(f"config 0 has {c0['slots']} slots < max_slots {num_users}")
    if c0["capacity"] < R.MAX_SEQ:
        fails.append(f"config 0 max_sequence_length {c0['capacity']} < max_seq {R.MAX_SEQ}")

    from launch_harness.kv_chunk_address_table_pb2 import KvChunkAddressTable

    t = KvChunkAddressTable()
    t.ParseFromString(Path(table_path).read_bytes())
    try:
        entries = table_entries(t, 0)
    except ValueError as e:
        return fails + [f"config 0: {e}"], geom
    cbytes = t.configs[0].chunk_size_bytes
    want = [(s, l, p) for s in range(num_users) for l in range(num_layers) for p in range(0, R.MAX_SEQ, 32)]
    missing = [k for k in want if k not in entries]
    if missing:
        fails.append(f"{len(missing)} (slot, layer, pos) records missing, e.g. {missing[:3]}")
    wrong = [k for k, e in entries.items() if e[1] != cbytes]
    if wrong:
        fails.append(f"{len(wrong)} records with size != chunk_size_bytes {cbytes}, e.g. {wrong[:3]}")
    seen = {}
    for k, (addr, _, dg) in entries.items():
        a = (dg, addr)
        if a in seen:
            fails.append(f"records {seen[a]} and {k} alias one DRAM address (slots / layers must not share a region)")
            break
        seen[a] = k
    fails += bank_overlaps(entries)
    return fails, geom


def decode(kvc, raw: bytes, geom: dict) -> np.ndarray:
    return kvc.decode_chunk(raw, dtype=geom["dtype"], width=geom["width"], storage=geom["storage"])


def harness_geometry_note(geom: dict) -> str:
    return (
        f"the launch harness decodes this cache as {geom.get('dtype')} {geom.get('storage')} (tables.py LAYOUTS keys "
        "the geometry on the record size: 36864 B = bf16 ROW-MAJOR, 19584 B = bfp8 TILE); a bf16 TILE cache has the "
        "36864 B size of a row-major one and reads back scrambled"
    )


def env_flag(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default) == "1"
