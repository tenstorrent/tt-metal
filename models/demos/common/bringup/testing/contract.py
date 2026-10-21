# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Serving contract: drive the model only through the prefill engine's adapter/runtime API, the way
models/demos/common/prefill/runners/prefill_runner.py does, and check what the Blaze prefill CI checks.

Checks (each failure is listed in ``contract_failures``; metric ``contract_checks_failed``):
  engine input      every chunk's input is the H2D shape: uint32 ROW_MAJOR [sp, 1, chunk/sp], tail padded with
                    ``contract.pad_token`` (default 0xFFFFFFFF, what the Blaze engine sends) past actual_end; the last
                    chunk ends ``tests.contract_tail_pad`` tokens early (default 32)
  acks              one ack per (layer, chunk), in layer order within each chunk; with an adapter that has
                    ``kv_slot_layer_ids`` (hybrids: only the layers that own a KV slab ack), one per slab layer
  ack timing        at each ack, the table's blocks for that layer and chunk already hold their final bytes
                    (re-read after a full sync must be byte-identical); ``acks_early`` counts violations
  table             build_kv_chunk_table round-trips through protobuf; the migration base address is entry 0
  producer          the producer's own KV read-back PCC vs the golden (``pcc_producer_kv_*``), plus the model's
                    independent read-back (``pcc_kv_independent_min``) when the hooks provide one, and their gap
  fixed state       tensors in spec state.fixed (recurrent state, conv tail) are not in the KV table: the hooks'
                    ``contract_state_pcc(spec, runtime, kv, slot, length, golden) -> {name: pcc}`` compares them with
                    the golden snapshot at ``length`` (``pcc_contract_state_*``); a spec with fixed state and no such
                    hook fails the contract

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
                ttnn.cluster.get_chip_unique_id_from_fabric_node_id(int(fid.mesh_id), int(fid.chip_id))
            )
    return out


def run_contract_test(s, mesh) -> list[str]:
    import ttnn
    from models.demos.common.prefill.adapter import PrefillRunParams, get_adapter
    from models.demos.common.prefill.runners import prefill_producer as producer

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

    early_reads, acks, cur = {}, [], {"chunk": None, "start": 0, "end": 0}

    def sink(layer, rid):
        acks.append((layer, rid))
        # Read the blocks this ack claims are ready: first and last 32-token block of the chunk, every config.
        for pos in {cur["start"], ((cur["end"] - 1) // BLOCK) * BLOCK}:
            for cid in range(n_cfg):
                early_reads[(layer, rid, pos, cid)] = bytes(
                    table.read_device_chunk(layer=layer, position=pos, slot=slot, config_id=cid)
                )

    rt.set_layer_completion_sink(sink)
    rt.compile(kv)
    acks.clear()
    early_reads.clear()

    tokens = g.tokens().tolist()
    for c in range(n_chunks):
        a, b = c * chunk, min((c + 1) * chunk, actual_len)
        cur.update(chunk=c, start=a, end=b)
        inp = engine_input(mesh, tokens[a:b], chunk, sp, int(s.get("contract.pad_token")))
        rt.prefill_chunk(inp, kv, slot_id=slot, actual_start=a, actual_end=b, request_id=c)
    ttnn.synchronize_device(mesh)

    slab_layers = getattr(adapter, "kv_slot_layer_ids", lambda n: None)(L)
    n_acks = L if slab_layers is None else len(slab_layers)
    want = [(layer, c) for c in range(n_chunks) for layer in range(n_acks)]
    if acks != want:
        failed.append(f"acks {acks[:6]}... != one per layer in order per chunk ({len(acks)} vs {len(want)})")
    early = sum(
        bytes(table.read_device_chunk(layer=l_, position=p, slot=slot, config_id=cid)) != data
        for (l_, _rid, p, cid), data in early_reads.items()
    )
    metrics.record("acks_early", early)
    metrics.record("ack_blocks_checked", len(early_reads))
    if early:
        failed.append(f"{early} of {len(early_reads)} blocks changed after their layer was acknowledged")
    if rt.kv_migration_base_address(kv) != table.lookup(0, 0, 0, 0).noc_addr & 0xFFFFFFFF:
        failed.append("migration base address != table entry (layer 0, pos 0, slot 0, config 0)")

    dmap = device_map(mesh)
    mins = producer._read_slot_kv_and_check_pcc(table, dmap, slot, actual_len - actual_len % BLOCK, str(g.dir))
    for k, v in mins.items():
        metrics.record(f"pcc_producer_kv_{k}", v)
    hooks = s.hooks()
    if hasattr(hooks, "contract_independent_pcc"):
        # Guards against a stubbed comp_pcc inside the producer (run_safe_pytest's precompile pass stubs it).
        ind = hooks.contract_independent_pcc(s, table, dmap, slot, actual_len - actual_len % BLOCK, g)
        metrics.record("pcc_kv_independent_min", min(ind.values()))
        gap = max(abs(mins[k] - ind[k]) for k in mins if k in ind)
        metrics.record("producer_vs_independent_pcc_gap", gap)
        if gap > 1e-4:
            failed.append(f"producer PCC {mins} disagrees with the independent read-back {ind}")
    drift = E.load(g.dir)  # F128: each state's limit = calib x the reference's own drift on its worst layer
    for k, v in mins.items():
        lim = E.state_limit(drift, k)
        metrics.record(f"pcc_producer_kv_{k}_limit", lim)
        if v < lim:
            failed.append(f"producer read-back PCC {k} {v:.5f} < limit {lim:.5f}")
    if s.state_fixed:
        if not hasattr(hooks, "contract_state_pcc"):
            failed.append(f"fixed-size state {s.state_fixed} unchecked: hooks.contract_state_pcc is missing")
        else:
            fixed = hooks.contract_state_pcc(s, rt, kv, slot, actual_len, g)
            if not fixed:
                failed.append("fixed-size state read-back: hooks.contract_state_pcc returned nothing")
            for k, v in fixed.items():
                metrics.record(f"pcc_contract_state_{k}", v)
                lim = E.state_limit(drift, k)  # F128: keyed by state name, like the producer's
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
