# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Serving contract, adapter test: the engine input, the per-layer acks (host sink) and the KV read through the
table, in process (serving_contract.md "Input", "Acks", "Adapter and table").

The adapter builds the served layers (caches.served_prefix) with 2 slots at chunk 2048 / max_seq 4096, exports its
table the way the runner's migration path does, and gets the server's chunks as the H2D stream delivers them
(server_rules.server_payload: PAD_ID tail, ring_sdpa_reshuffle by actual_start, [sp, 1, chunk / sp] uint32 sharded
over mesh axis 0). Plan: server_rules.interleave(ACK_TURNS): slot 0 1000, then 3000 over its 960 resident prefix;
slot 1 cold 3000 over two chunks: (0, 0, 1000), (1, 0, 2048), (0, 960, 3000), (1, 2048, 3000).

Inside every ack (the host layer-completion sink, prefill_runner.build_layer_completion_sink's callback) the test reads
that layer's records of the chunk through the table over UMD, as the KV Manager does. Pass:
  - acks: one per served layer per chunk, in layer order, request_id passed through (prefill_reader.cpp:121-160:
    the k-th ack of a chunk migrates layer k)
  - at each ack: the chunk's rows of every cache the layer owns vs the golden (server per-channel PCC >= 0.97), the pad
    rows of the last record zero; the same records re-read after the chunk returns and the mesh syncs hold the same
    bytes (an ack before the KV is in DRAM ships stale KV: K-71)
  - at the end: each slot's [0, end) vs the golden; slot 0's reused prefix [0, 960) byte-identical to what turn 1's
    acks shipped, its last block [960, 1000) PCC >= 0.99 against it (the harness's prefix / last-block rule,
    validation.py:55-61); every chip of a record's device group holds the same bytes
"""

import pytest

from models.demos.common.bringup.testing.harness import device_timeout, mesh_parametrize, spec
from models.demos.deepseek_v41_flash_d_p.tests.bringup.contract import caches as C
from models.demos.deepseek_v41_flash_d_p.tests.bringup.contract import server_rules as R

S = spec()
pytestmark = device_timeout(S)
N_SLOTS = 2


@mesh_parametrize
def test_adapter_acks(mesh_device, tmp_path):
    if mesh_device is None:
        pytest.skip("device test")
    import ttnn

    R.self_check()
    _, kvc, _ = R.harness()
    gk = C.GoldenKV()
    tokens = gk.tokens().numpy()
    thr = R.state_threshold()
    layers = C.served_prefix()
    cfgs = C.served_configs(layers)
    try:
        _, runtime, kv = C.build(mesh_device, layers, N_SLOTS)
    except C.NotBuilt as e:
        pytest.fail(f"not built: {e}", pytrace=False)
    path = C.export_table(runtime, kv, mesh_device, str(tmp_path / "table.pb"), len(layers))
    tfails = C.table_rules(path, layers, N_SLOTS)
    assert not tfails, "table rules:\n" + "\n".join(tfails)
    reader = C.TableReader(path, C.device_map(mesh_device, str(tmp_path / "device_map.json")))

    acks, cur, fails = [], {}, []
    shipped = {}  # (slot, turn) -> {config: {(layer, pos): bytes}} what the acks of that turn shipped

    def sink(layer_idx, request_id):
        acks.append((int(layer_idx), int(request_id)))
        s, lo, hi = cur["slot"], cur["start"], cur["end"]
        for cfg in cfgs:
            if layer_idx not in cfg["layers"]:
                continue
            recs = reader.records(cfg["name"], s, [layer_idx], lo, hi)
            cur["reads"][(cfg["name"], layer_idx)] = recs
            shipped.setdefault((s, cur["turn"]), {}).setdefault(cfg["name"], {}).update(recs)
            got, pad = C.valid_and_pad(recs, cfg, layer_idx, lo, hi)
            want = gk.rows(cfg, layer_idx, lo, hi)
            cur["fails"] += C.pcc_failures(kvc, cfg, got, want, layer_idx, thr, f"ack slot {s} [{lo}, {hi})")
            if pad.size and not (pad == 0).all():
                cur["fails"].append(f"ack slot {s} {cfg['name']} layer {layer_idx}: pad rows after {hi} not zero")

    runtime.set_layer_completion_sink(sink)
    ends = {}
    for k, p in enumerate(R.interleave(R.ACK_TURNS)):
        cur.update(slot=p.slot, start=p.start, end=p.end, turn=p.turn, fails=[], reads={})
        inp = C.payload_tensor(mesh_device, R.server_payload(tokens[p.start : p.start + R.CHUNK], p.start, p.end))
        n0 = len(acks)
        try:
            runtime.prefill_chunk(
                inp,
                kv,
                slot_id=p.slot,
                actual_start=p.start,
                actual_end=p.end,
                request_id=k,
                d2h_service=None,
                metadata_msg=None,
            )
        except Exception as err:
            pytest.fail(f"not built: chunk {k} slot {p.slot} [{p.start}, {p.end}) rejected: {err}", pytrace=False)
        ttnn.synchronize_device(mesh_device)
        want = [(i, k) for i in range(len(layers))]
        if acks[n0:] != want:
            fails.append(f"chunk {k} [{p.start}, {p.end}): acks {acks[n0:]}, want one per layer in order {want}")
        fails += [f"chunk {k}: {x}" for x in cur["fails"]]
        early = [
            (name, layer)
            for (name, layer), recs in cur["reads"].items()
            if reader.records(name, p.slot, [layer], p.start, p.end) != recs
        ]
        if early:
            fails.append(
                f"chunk {k}: records changed after their layer's ack (acked before the KV was in DRAM): {early}"
            )
        ends[p.slot] = p.end
        print(f"[adapter acks] chunk {k} slot {p.slot} [{p.start}, {p.end}): {len(acks) - n0} acks")

    for s, e in ends.items():
        for cfg in cfgs:
            recs = reader.records(cfg["name"], s, cfg["layers"], 0, e)
            for layer in cfg["layers"]:
                got, _ = C.valid_and_pad(recs, cfg, layer, 0, e)
                fails += C.pcc_failures(
                    kvc, cfg, got, gk.rows(cfg, layer, 0, e), layer, thr, f"final slot {s} [0, {e})"
                )
    t1, t2 = R.ACK_TURNS[0]
    prefix = R.follow_up_resident(t1, t2)
    for cfg in cfgs:
        final = reader.records(cfg["name"], 0, cfg["layers"], 0, t1)
        first = shipped[(0, 0)][cfg["name"]]
        fails += [f"slot 0 reused prefix: {x}" for x in C.bytes_equal(first, final, cfg, cfg["layers"], 0, prefix)]
        for layer in cfg["layers"]:
            a, _ = C.valid_and_pad(first, cfg, layer, prefix, t1)
            b, _ = C.valid_and_pad(final, cfg, layer, prefix, t1)
            fails += C.pcc_failures(kvc, cfg, b, a, layer, R.LAST_BLOCK_PCC, f"slot 0 last block [{prefix}, {t1})")
    if reader.replica_mismatch:
        fails.append(
            f"device-group replicas differ at {len(reader.replica_mismatch)} records: {reader.replica_mismatch[:3]}"
        )
    print(f"[adapter acks] layers {layers}, configs {[c['name'] for c in cfgs]}: {len(fails)} failures")
    assert not fails, "\n".join(fails[:40])
