# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Serving contract: drive the model only through the prefill engine's adapter/runtime API, the way
models/demos/common/prefill/runners/prefill_runner.py does, and check what the Blaze prefill CI checks.

Checks (each failure is listed in ``contract_failures``; metric ``contract_checks_failed``):
  engine input      every chunk's input is the H2D shape: uint32 ROW_MAJOR [sp, 1, chunk/sp], tail padded with
                    ``contract.pad_token`` (default 0xFFFFFFFF, what the Blaze engine sends) past actual_end; the last
                    chunk ends ``tests.contract_tail_pad`` tokens early (default 32)
  acks              one ack per (layer, chunk), in layer order within each chunk, through the engine's sink
                    (``sink.layers_completed(layer_start, layer_end, slot_id, pos_start, pos_end)``: GLOBAL layers,
                    the chunk's slot and [actual_start, actual_end)); with an adapter that has ``kv_slot_layer_ids``
                    (hybrids: only the layers that own a KV slab ack), one per slab layer
  ack timing        at each ack, the table's blocks for that layer and chunk already hold their final bytes
                    (re-read after a full sync must be byte-identical); ``acks_early`` counts violations. A Snapshot
                    config (tensor_type Snapshot: fixed-size state rewritten by design on every chunk) is checked the
                    way tt-d-gen ships it (prefill_reader.cpp: non-KV tensors only with the request's last chunk): at
                    the last chunk's acks only, every record of its declared positions (AckReads)
  table             build_kv_chunk_table round-trips through protobuf; the migration base address is entry 0
  producer          the producer's own KV read-back PCC vs the golden (``pcc_producer_kv_*``), plus the model's
                    independent read-back (``pcc_kv_independent_min``) when the hooks provide one, and their gap
  fixed state       tensors in spec state.fixed (recurrent state, conv tail) are not in the KV table: the hooks'
                    ``contract_state_pcc(spec, runtime, kv, slot, length, golden) -> {name: pcc}`` compares them with
                    the golden snapshot at ``length`` (``pcc_contract_state_*``); a spec with fixed state and no such
                    hook fails the contract
  handoff (F264)    a model whose serving path stops before the engine's actual_end (hooks.contract_end, e.g. decode's
                    cap) is judged there: the producer / independent read-back over [0, it rounded down to 32), the
                    fixed state at it, limits from the official drift at it (reference/handoff.py,
                    ``handoff-drift-<rung>.json``); ``contract_state_at`` records the position

Engine env (PREFILL_MODEL, PREFILL_SP/TP, PREFILL_CHUNK_SIZE, PREFILL_MAX_SEQ_LEN, PREFILL_NUM_LAYERS,
PREFILL_NUM_USERS) must be set before the adapter is imported: call ``engine_env(spec)`` at module import.
"""

from __future__ import annotations

import os
import tempfile

import torch

from models.demos.common.bringup.core import metrics
from models.demos.common.bringup.reference.generate_golden import run_starts
from models.demos.common.bringup.reference.golden import Golden
from models.demos.common.bringup.testing import e2e_limits as E

PAD = 0xFFFFFFFF
BLOCK = 32


def contract_rung(s) -> str:
    return s.get("contract.rung") or s.data["ladder"][0]["name"]


LAYER_IDS_ENV = "PREFILL_LAYER_IDS"


def served_layers(s) -> tuple[int, int]:
    """(first layer, layer count) the engine serves. A contiguous subset is served as itself (F41). A non-contiguous one
    (a partial run's representative layers) is served as ``count`` engine layers 0..count-1, the model layer of each
    listed in PREFILL_LAYER_IDS (F100): acks, the KV table and the producer's read-back use the engine index; the
    runtime maps it with ``layer_ids_from_env`` and exposes the list as ``layer_ids``. A run of layers after a gap takes
    its input from ``gap_feed`` (F102), never from the previous layer's output."""
    sel = s.layers()
    if sel == list(range(sel[0], sel[0] + len(sel))):
        return sel[0], len(sel)
    return 0, len(sel)


def layer_ids_from_env(first_layer_idx: int, num_layers: int) -> list[int]:
    """The model layers an adapter serves: PREFILL_LAYER_IDS when the engine env sets it (F100), else the range."""
    ids = os.environ.get(LAYER_IDS_ENV)
    if ids:
        out = [int(x) for x in ids.split(",")]
        assert len(out) == num_layers, f"{LAYER_IDS_ENV}={ids} lists {len(out)} layers, the engine serves {num_layers}"
        return out
    return list(range(first_layer_idx, first_layer_idx + num_layers))


class GapFeed:
    """F102: the input of a served layer run that follows a gap in a non-contiguous subset (a partial bring-up's
    representative layers): the golden block input of the run's first layer, as the ladder feeds it (the gaps never
    run on the device). Framework-owned, so no model reads a golden: an adapter asks ``gap_feed(layer_ids, max_seq,
    chunk)`` once and, at a layer ``i in feed.starts`` other than 0 (a run's first layer), feeds ``feed.rows(layer, start)``
    [chunk, width] fp32 (zero rows past the golden) to the layer instead of the previous layer's output."""

    def __init__(self, s, layer_ids: list[int], max_seq: int, chunk: int):
        self.starts = set(run_starts(layer_ids))  # the first layer of each contiguous run
        self.chunk, self.g, self._rows = chunk, None, {}
        for rung in s.data.get("ladder", []):
            try:
                g = Golden.for_rung(s, rung["name"])
            except FileNotFoundError:
                continue
            if g.seq >= max_seq and sorted(g.dumped_chunks) == list(range(g.n_chunks)):
                self.g = g
                break
        assert self.g is not None, f"a gapped layer subset needs a golden rung with every chunk dumped up to {max_seq}"

    def rows(self, layer: int, start: int) -> torch.Tensor:
        if layer not in self._rows:
            self._rows[layer] = torch.cat([self.g.layer(c, layer)["in"] for c in range(self.g.n_chunks)]).float()
        x = self._rows[layer][start : start + self.chunk]
        return torch.cat([x, x.new_zeros(self.chunk - x.shape[0], x.shape[1])]) if x.shape[0] < self.chunk else x


def gap_feed(layer_ids: list[int], max_seq: int, chunk: int) -> GapFeed | None:
    """F102: the bring-up's GapFeed when the engine serves a gapped subset under the bring-up contract
    (PREFILL_LAYER_IDS set, a run starting at a layer other than 0); None otherwise (a served model)."""
    if not os.environ.get(LAYER_IDS_ENV) or run_starts(layer_ids) == [0]:
        return None
    from models.demos.common.bringup.reference.golden import load_spec

    return GapFeed(load_spec(), layer_ids, max_seq, chunk)


def engine_env(s) -> dict:
    rung = s.rung(contract_rung(s))
    env = {
        "PREFILL_MODEL": s.get("contract.adapter", s.model),
        "PREFILL_SP": str(s.mesh[0]),
        "PREFILL_TP": str(s.mesh[1]),
        "PREFILL_CHUNK_SIZE": str(rung["chunk"]),
        "PREFILL_MAX_SEQ_LEN": str(rung["seq"]),
        "PREFILL_NUM_LAYERS": str(served_layers(s)[1]),
        LAYER_IDS_ENV: ",".join(str(i) for i in s.layers()),
        "PREFILL_NUM_USERS": str(s.get("contract.num_users")),
    }
    os.environ.update(env)
    return env


def engine_input(mesh, tokens: list[int], chunk: int, sp: int, pad: int = PAD):
    """The chunk input exactly as the engine's H2D service delivers it."""
    import ttnn

    padded = list(tokens) + [pad] * (chunk - len(tokens))
    t = torch.tensor(padded, dtype=torch.int64).reshape(sp, 1, chunk // sp).to(torch.uint32)  # as prefill_producer
    return ttnn.from_torch(
        t,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.create_mesh_mapper(
            mesh, ttnn.MeshMapperConfig(placements=[ttnn.PlacementShard(0), ttnn.PlacementReplicate()])
        ),
    )


def device_map(mesh) -> dict:
    import ttnn

    out = {}
    rows, cols = mesh.shape
    for r in range(rows):
        for c in range(cols):
            fid = mesh.get_fabric_node_id(ttnn.MeshCoordinate(r, c))
            out[(int(fid.mesh_id), int(fid.chip_id))] = int(
                ttnn.cluster.get_chip_unique_id_from_fabric_node_id(mesh, int(fid.mesh_id), int(fid.chip_id))
            )
    return out


# F131: the producer's read-back keys -> golden state names; an adapter's kv_state_names() or the hooks' (spec) override
KV_STATE_NAMES = {"k": "key", "v": "value", "kvpe": "kv_latent"}


def kv_state_names(s, adapter) -> dict:
    names = dict(KV_STATE_NAMES)
    names.update(getattr(adapter, "kv_state_names", dict)())
    hooks = s.hooks()
    if hasattr(hooks, "kv_state_names"):
        names.update(hooks.kv_state_names(s))
    return names


def producer_limits(s, adapter, mins: dict, drift: dict, failed: list[str]) -> None:
    """F128 / F131: each producer key's ``pcc_producer_kv_<key>_limit`` from the drift of the golden state it reads
    back. A key that is a state name maps to itself; one that maps to no state with a drift is a failure, never a
    skipped check."""
    names = kv_state_names(s, adapter)
    states = {m.group(2) for m in map(E.STATE_KEY.match, drift) if m and m.group(1) == "contract_state"}
    for k, v in mins.items():
        name = k if k in states else names.get(k)
        if name not in states:
            failed.append(
                f"producer KV key {k!r} maps to no golden state with a drift ({sorted(states)}): give the adapter "
                "kv_state_names() -> {producer key: state name} (testing/contract.py, F131)"
            )
            continue
        lim = E.state_limit(drift, name, "contract_state")
        metrics.record(f"pcc_producer_kv_{k}_limit", lim)
        if v < lim:
            failed.append(f"producer read-back PCC {k} ({name}) {v:.5f} < limit {lim:.5f}")


def is_snapshot_config(cfg) -> bool:
    """A table config of tensor_type Snapshot (KvChunkTensorType.Snapshot; a table without the field: False)."""
    t = getattr(cfg, "tensor_type", None)
    return t is not None and str(getattr(t, "name", t)).split(".")[-1] == "Snapshot"


class AckReads:
    """The ack-timing check (``acks_early``). ``on_ack(row, chunk, start, end)`` reads the blocks an ack claims ready;
    ``early()`` re-reads them all (after the request and a full sync) and counts those whose bytes changed.
    History (KV) configs: the chunk's first and last 32-token block, at every chunk's ack (unchanged rule). Snapshot
    configs (``is_snapshot_config``): rewritten by design on every chunk and shipped by tt-d-gen with the request's
    last chunk only, so read at the last chunk's acks only (``last_chunk``), every record of the config's declared
    positions [0, max_sequence_length) that has a table entry; never at a position past it (tt-metal's table refuses
    that lookup). ``table``: read_device_chunk(layer=, position=, slot=, config_id=), config(cid), lookup(l, p, s, c),
    num_configs()."""

    def __init__(self, table, slot: int, last_chunk: int, block: int = BLOCK):
        self.table, self.slot, self.last_chunk, self.block = table, int(slot), int(last_chunk), int(block)
        self.n_cfg = table.num_configs()
        self.snap = {}  # snapshot config id -> its record positions
        for cid in range(self.n_cfg):
            cfg = table.config(cid)
            if is_snapshot_config(cfg):
                self.snap[cid] = list(range(0, int(cfg.max_sequence_length), int(cfg.chunk_n_tokens)))
        self.reads = {}

    def _read(self, row: int, pos: int, cid: int) -> bytes:
        return bytes(self.table.read_device_chunk(layer=row, position=pos, slot=self.slot, config_id=cid))

    def on_ack(self, row: int, chunk: int, start: int, end: int) -> None:
        for cid in range(self.n_cfg):
            if cid in self.snap:
                if chunk != self.last_chunk:
                    continue
                for pos in self.snap[cid]:
                    if self.table.lookup(row, pos, self.slot, cid).size_bytes:
                        self.reads[(row, chunk, pos, cid)] = self._read(row, pos, cid)
                continue
            for pos in {start, ((end - 1) // self.block) * self.block}:
                self.reads[(row, chunk, pos, cid)] = self._read(row, pos, cid)

    def clear(self) -> None:
        self.reads.clear()

    def early(self) -> int:
        return sum(self._read(r, p, cid) != data for (r, _chunk, p, cid), data in self.reads.items())


def run_contract_test(s, mesh) -> list[str]:
    import ttnn
    from models.demos.common.prefill.adapter import PrefillRunParams, get_adapter
    from models.demos.common.prefill.runners import prefill_producer as producer
    from models.demos.common.prefill.runners.layer_completion_ring_writer import LayerCompletionSink

    D = ttnn.experimental.disaggregation
    rung = s.rung(contract_rung(s))
    g = Golden.for_rung(s, rung["name"])
    seq, chunk = rung["seq"], rung["chunk"]
    n_chunks, sp = seq // chunk, s.mesh[0]
    slot = int(s.get("contract.slot"))
    tail = int(s.get("tests.contract_tail_pad"))
    actual_len = seq - tail
    first, L = served_layers(s)
    failed = []

    adapter = get_adapter(os.environ["PREFILL_MODEL"])
    hf = adapter.load_hf_config()
    params = PrefillRunParams(
        mesh_shape=tuple(s.mesh),
        num_layers=L,
        first_layer_idx=first,
        is_first_rank=True,
        is_last_rank=True,
        max_seq_len=seq,
        chunk_size=chunk,
        num_users=int(os.environ["PREFILL_NUM_USERS"]),
        capacity_factor=1,
        num_links=int(s.get("contract.num_links")),
        gate_mode_name=adapter.default_gate_mode,
        kv_only_last_layer=False,
        weight_cache_path=adapter.weight_cache_path(tuple(s.mesh)),
    )
    kv = adapter.allocate_kv_cache(mesh_device=mesh, hf_config=hf, params=params)
    rt = adapter.build_runtime(mesh_device=mesh, hf_config=hf, params=params)
    sel = s.layers()
    if first == 0 and sel != list(range(len(sel))):  # F100: a non-contiguous subset served by engine index
        served = getattr(rt, "layer_ids", None)
        if served is None or list(served) != sel:
            failed.append(
                f"non-contiguous layer subset {sel}: the runtime must serve the model layers {LAYER_IDS_ENV} lists and "
                f"expose them as rt.layer_ids (got {served}; models/demos/common/bringup/testing/contract.py:"
                "layer_ids_from_env)"
            )

    with tempfile.TemporaryDirectory() as td:
        table_path = rt.build_kv_chunk_table(kv, os.path.join(td, "kv_table.pb"))
        table = D.import_from_protobuf_file(table_path)
        n_cfg = table.num_configs()
        t2 = D.import_from_protobuf_file(table_path)
        if t2.num_configs() != n_cfg or t2.total_entries() != table.total_entries():
            failed.append("table protobuf round-trip changed the table")

    ack_reads, acks, cur = AckReads(table, slot, n_chunks - 1), [], {"chunk": None, "start": 0, "end": 0}
    slab_layers = getattr(adapter, "kv_slot_layer_ids", lambda n: None)(L)
    # the table's layer of an acked global layer: its KV slot on a hybrid, else the layer itself
    table_row = {g: k for k, g in enumerate(slab_layers)} if slab_layers is not None else None

    class Sink(LayerCompletionSink):
        def layers_completed(self, layer_start, layer_end, slot_id, pos_start, pos_end):
            for layer in range(layer_start, layer_end):
                acks.append((layer, cur["chunk"], (slot_id, pos_start, pos_end)))
                row = layer if table_row is None else table_row.get(layer)
                if row is None:
                    continue  # not a slab layer: the ack list check reports it
                # Read the blocks this ack claims are ready: first and last 32-token block of the chunk, every history
                # config; a Snapshot config's records at the request's last chunk only (AckReads)
                ack_reads.on_ack(row, cur["chunk"], cur["start"], cur["end"])

    rt.set_layer_completion_sink(Sink())
    rt.compile(kv)
    acks.clear()
    ack_reads.clear()

    tokens = g.tokens().tolist()
    for c in range(n_chunks):
        a, b = c * chunk, min((c + 1) * chunk, actual_len)
        cur.update(chunk=c, start=a, end=b)
        inp = engine_input(mesh, tokens[a:b], chunk, sp, int(s.get("contract.pad_token")))
        rt.prefill_chunk(inp, kv, slot_id=slot, actual_start=a, actual_end=b, request_id=c)
    ttnn.synchronize_device(mesh)

    acked = list(range(first, first + L)) if slab_layers is None else list(slab_layers)
    want = [(layer, c, (slot, c * chunk, min((c + 1) * chunk, actual_len))) for c in range(n_chunks) for layer in acked]
    if acks != want:
        failed.append(f"acks {acks[:6]}... != one per layer in order per chunk ({len(acks)} vs {len(want)})")
    early = ack_reads.early()
    metrics.record("acks_early", early)
    metrics.record("ack_blocks_checked", len(ack_reads.reads))
    if early:
        failed.append(f"{early} of {len(ack_reads.reads)} blocks changed after their layer was acknowledged")
    if rt.kv_migration_base_address(kv) != table.lookup(0, 0, 0, 0).noc_addr & 0xFFFFFFFF:
        failed.append("migration base address != table entry (layer 0, pos 0, slot 0, config 0)")

    # F264: the state and the served caches are judged where the model's serving path hands off (hooks.contract_end:
    # e.g. decode's cap when prefill stops there), never at a position it does not compute; their limits come from
    # the official drift at that point (reference/handoff.py), the golden's snapshot there from the same pipeline
    from models.demos.common.bringup.reference import handoff as HO

    at = HO.contract_end(s, actual_len)
    metrics.record("contract_state_at", at)
    if not 0 < at <= actual_len:
        failed.append(f"hooks.contract_end({actual_len}) = {at}: not in (0, {actual_len}]")
        at = actual_len
    print(f"contract: state and served caches judged at {at} (request end {actual_len})")
    dmap = device_map(mesh)
    mins = producer._read_slot_kv_and_check_pcc(table, dmap, slot, at - at % BLOCK, str(g.dir))
    for k, v in mins.items():
        metrics.record(f"pcc_producer_kv_{k}", v)
    hooks = s.hooks()
    if hasattr(hooks, "contract_independent_pcc"):
        # Guards against a stubbed comp_pcc inside the producer (run_safe_pytest's precompile pass stubs it).
        ind = hooks.contract_independent_pcc(s, table, dmap, slot, at - at % BLOCK, g)
        metrics.record("pcc_kv_independent_min", min(ind.values()))
        gap = max(abs(mins[k] - ind[k]) for k in mins if k in ind)
        metrics.record("producer_vs_independent_pcc_gap", gap)
        if gap > 1e-4:
            failed.append(f"producer PCC {mins} disagrees with the independent read-back {ind}")
    # F128 / F134: calib x the reference's drift as the contract reads it back (F264: at the handoff point)
    drift = E.load(g.dir, rung["name"]) if at == actual_len else HO.load_drift(g.dir, rung["name"], at)
    producer_limits(s, adapter, mins, drift, failed)
    if s.state_fixed:
        if not hasattr(hooks, "contract_state_pcc"):
            failed.append(f"fixed-size state {s.state_fixed} unchecked: hooks.contract_state_pcc is missing")
        else:
            fixed = hooks.contract_state_pcc(s, rt, kv, slot, at, g)
            if not fixed:
                failed.append("fixed-size state read-back: hooks.contract_state_pcc returned nothing")
            for k, v in fixed.items():
                metrics.record(f"pcc_contract_state_{k}", v)
                lim = E.state_limit(drift, k, "contract_state")  # F128: the hook's keys are golden state names
                metrics.record(f"pcc_contract_state_{k}_limit", lim)
                if v < lim:
                    failed.append(f"fixed-size state read-back PCC {k} {v:.5f} < limit {lim:.5f}")
    metrics.record("contract_checks_failed", len(failed))
    for f in failed:
        print(f"FAIL contract: {f}")
    return failed


def gqa_independent_pcc(s, table, dmap, slot, length, g, n_kv: int, head_dim: int) -> dict:
    """Independent read-back for a GQA cache whose configs are K heads 0..n_kv-1 then V heads (the gpt_oss_d_p
    and ernie45_d_p layout). A model with that layout can return this from its contract_independent_pcc hook."""
    from models.demos.common.prefill.runners import prefill_producer as producer

    out = {"k": 1.0, "v": 1.0}
    for layer in s.layers():
        st = g.state(layer)
        for kind, base, name in (("k", 0, "key"), ("v", n_kv, "value")):
            dev = torch.stack(
                [
                    producer._read_kv_slice(
                        table, dmap, base + h, layer, slot, length, head_dim, producer._decode_bfp8_chunk
                    )
                    for h in range(n_kv)
                ]
            )
            gold = st[name].reshape(n_kv, -1, head_dim)[:, :length]
            out[kind] = min(out[kind], metrics.pcc(dev.float(), gold.float()))
    return out
