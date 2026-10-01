# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Serving contract, adapter test: the engine input, the per-layer acks (host sink) and the KV table, in process
(serving_contract.md, "Input", "KV cache", "Acks", "Adapter and table").

The adapter builds a 2-layer runtime (PrefillRunParams.num_layers = 2) with 2 slots, exports its table the way the
runner's migration path does (stages -> allgather_kv_stage_layouts -> build_kv_chunk_table(..., first_layer_idx,
num_my_layers, stage_layout)), and gets the server's chunks as the H2D stream delivers them (PAD_ID tail,
ring_sdpa_reshuffle by actual_start, [4, 1, 1280] uint32 sharded over axis 0). Plan, interleaved round-robin
(server_rules.interleave): slot 0 turns 3000 then 9000 (follow-up from the 2944 resident prefix), slot 1 turn 6000.

Inside every ack (the host layer-completion sink, the shipped single-rank transport) the test reads that layer's
records for the chunk through the table over UMD, as the KV Manager does, and checks them. Pass:
  - acks: one per layer per chunk, in layer order, global layer ids, the request_id passed through
  - at each ack: [actual_start, actual_end) of that layer vs the golden (kv_dump_compare per-channel PCC >= 0.97),
    the pad rows of the last record zero, every chip of the record's device group holding the same bytes
  - at the end: each slot's [0, end) vs the golden; slot 0's reused prefix [0, 2944) byte-identical to what turn 1's
    acks shipped, its last block [2944, 3000) PCC >= 0.99 against it (the harness's source / destination rule)
  - the table: tables.read_table and layout accept it, every (slot, layer, pos < max_seq) record present, one size,
    no two records on one address; decoded with the harness's geometry it is the KV the model wrote
"""

import numpy as np
import pytest

from models.demos.common.bringup.testing.harness import device_timeout, mesh_parametrize, spec
from models.demos.xing40_a4b_d_p.tests.bringup.contract import engine as E
from models.demos.xing40_a4b_d_p.tests.bringup.contract import server_rules as R

S = spec()
pytestmark = device_timeout(S)

N_LAYERS, N_SLOTS = 2, 2
TURNS = {0: [3000, 9000], 1: [6000]}


@mesh_parametrize
def test_adapter_acks(mesh_device, tmp_path):
    if mesh_device is None:
        pytest.skip("device test")
    R.self_check()
    tables, kvc = R.harness()
    g = R.golden()
    tokens = g.tokens().numpy()
    thr = R.state_threshold()
    try:
        _, runtime, kv = E.build(mesh_device, N_LAYERS, N_SLOTS)
    except E.NotBuilt as e:
        pytest.fail(f"not built: {e}", pytrace=False)
    table_path = E.export_table(runtime, kv, mesh_device, str(tmp_path / "table.pb"), N_LAYERS)
    fails, geom = E.table_rules(table_path, N_LAYERS, N_SLOTS)
    assert not fails, "table rules:\n" + "\n".join(fails)
    reader = E.TableReader(table_path, E.device_map(mesh_device, str(tmp_path / "device_map.json")))
    note = E.harness_geometry_note(geom)

    acks, cur = [], {}
    snap = tmp_path / "snap_slot0_turn0"

    def sink(layer_idx, request_id):
        acks.append((int(layer_idx), int(request_id)))
        s, lo, hi = cur["slot"], cur["start"], cur["end"]
        rows = []
        for pos in range(R.align_down(lo, 32), R.ceil_to(hi, 32), 32):
            raw = reader.record(s, layer_idx, pos)
            rows.append(E.decode(kvc, raw, geom))
            if cur["snap"]:
                snap.mkdir(exist_ok=True)
                (snap / R.record_name(s, layer_idx, pos)).write_bytes(raw)
        got = np.concatenate(rows)[lo % 32 :][: hi - lo]
        want = R.golden_kv(g, layer_idx)[lo:hi].numpy()
        f = R.kv_pcc_failures(kvc, got, want, layer_idx, thr, f"ack slot {s} [{lo}, {hi})")
        cur["fails"] += [f"{x}; {note}" for x in f]
        pad = np.concatenate(rows)[lo % 32 + hi - lo :]
        if pad.size and not np.all(pad == 0):
            cur["fails"].append(f"ack slot {s} layer {layer_idx}: pad rows [{hi}, {R.ceil_to(hi, 32)}) not zero")

    runtime.set_layer_completion_sink(sink)
    pushes = R.interleave(TURNS)
    ends = {}
    for k, p in enumerate(pushes):
        cur.update(slot=p.slot, start=p.start, end=p.end, snap=(p.slot == 0 and p.turn == 0), fails=[])
        inp = E.payload_tensor(mesh_device, R.server_payload(tokens[p.start : p.start + R.CHUNK], p.start, p.end))
        n0 = len(acks)
        try:
            runtime.prefill_chunk(
                inp, kv, slot_id=p.slot, actual_start=p.start, actual_end=p.end, request_id=k, d2h_service=None
            )
        except Exception as err:
            pytest.fail(f"not built: chunk {k} slot {p.slot} [{p.start}, {p.end}) rejected: {err}", pytrace=False)
        got = acks[n0:]
        if got != [(i, k) for i in range(N_LAYERS)]:
            fails.append(f"chunk {k}: acks {got}, want one per layer in order {[(i, k) for i in range(N_LAYERS)]}")
        fails += [f"chunk {k}: {x}" for x in cur["fails"]]
        ends[p.slot] = p.end

    final = tmp_path / "final"
    for s, e in ends.items():
        reader.dump(final, s, range(N_LAYERS), 0, e)
    for s, e in ends.items():
        dump = kvc.load_dump(str(final), slot=s, chunk_bytes=len(reader.record(s, 0, 0)))
        for layer in range(N_LAYERS):
            got = kvc.reassemble(dump, layer, 0, e, **{k: geom[k] for k in ("dtype", "width", "storage")})
            fails += R.kv_pcc_failures(kvc, got, R.golden_kv(g, layer)[:e].numpy(), layer, thr, f"final slot {s}")
    a = kvc.load_dump(str(snap), slot=0, chunk_bytes=len(reader.record(0, 0, 0)))
    b = kvc.load_dump(str(final), slot=0, chunk_bytes=len(reader.record(0, 0, 0)))
    gk = {k: geom[k] for k in ("dtype", "width", "storage")}
    prefix = R.follow_up_resident(TURNS[0][0], TURNS[0][1])
    res = kvc.compare(a, b, method="bytecmp", layers=list(range(N_LAYERS)), start=0, end=prefix, **gk)
    if res["status"] != "passed":
        fails.append(f"slot 0 reused prefix [0, {prefix}) changed after turn 1 was acked: {res['findings'][:3]}")
    res = kvc.compare(
        a,
        b,
        method="pcc",
        layers=list(range(N_LAYERS)),
        start=prefix,
        end=TURNS[0][0],
        threshold=R.LAST_BLOCK_PCC,
        **gk,
    )
    if res["status"] != "passed":
        fails.append(
            f"slot 0 last block [{prefix}, {TURNS[0][0]}) PCC vs turn 1 < {R.LAST_BLOCK_PCC}: {res['findings']}"
        )
    if reader.replica_mismatch:
        fails.append(
            f"device-group replicas differ at {len(reader.replica_mismatch)} records: {reader.replica_mismatch[:3]}"
        )
    assert not fails, "\n".join(fails)
