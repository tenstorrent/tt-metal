# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Serving contract, adapter test: the engine input, the per-layer acks (host sink) and the KV read through the
table, in process (serving_contract.md "Input", "Acks", "Adapter and table").

The adapter builds the served layers (caches.served_prefix) with 2 slots at R.CHUNK / R.MAX_SEQ (the spec's golden
rung: 2048 / 4096 on the seed, 5120 / 10240 on galaxy_8x4), exports its table the way the runner's migration path
does, and gets the server's chunks as the H2D stream delivers them (server_rules.server_payload: PAD_ID tail,
ring_sdpa_reshuffle by actual_start, [sp, 1, chunk / sp] uint32 sharded over mesh axis 0). Plan:
server_rules.interleave(ACK_TURNS) (H = CHUNK / 2): slot 0 H - 24, then CHUNK + 952 over its H - 64 resident prefix;
slot 1 cold CHUNK + 952 over two chunks. Seed: (0, 0, 1000), (1, 0, 2048), (0, 960, 3000), (1, 2048, 3000).

Inside every ack (the host layer-completion sink: sink.layers_completed(layer_start, layer_end, slot_id, pos_start,
pos_end), what prefill_runner's LayerCompletionRingWriter receives) the test reads that layer's records of the chunk
through the table over UMD, as the KV Manager does. Pass:
  - acks: one per served layer per chunk, in layer order, naming the chunk (slot_id, actual_start, actual_end) the
    engine passed (prefill_reader.cpp:121-160: the k-th ack of a chunk migrates layer k)
  - at each ack: the chunk's rows of every cache the layer owns vs the golden (caches.golden_failures: every row of the
    compressed caches; a dumped layer's window KV over every position of the range, per channel against calib x the
    official reference's drift on the same rows (caches.GoldenKV(window="drift"), FIX.19, owner 2026-11-01; it
    replaces decision B's last 128 rows vs the ring's drift; P.26: every window-KV check also records its relative L2
    error, and the official reference's on the same rows where it has them, as metrics err_acks_wkv_* /
    err_acks_wkv_*_official, never gated); another layer's window KV at the positions its ring
    snapshots hold; every served layer compared at least once, the per-layer mode printed) (server per-channel PCC >= 0.93), the pad
    rows of the last record zero; the same records re-read after the chunk returns and the mesh syncs hold the same
    bytes (an ack before the KV is in DRAM ships stale KV: K-71). INDEX_K_SERVED=fp8_row192 (P.39): the index configs'
    192-byte rows decoded (caches.decode) and compared with the golden at the same limit; their bytes' format
    (caches.fp8_row_failures: zero padding, a positive scale, the row's largest |e4m3| 416 or 448, all-zero pad rows);
    a re-index layer's rows (24 / 28 / 32 / 36) byte-identical to its key source's (layer 20)
  - at the end: each slot's [0, end) vs the golden; slot 0's reused prefix [0, 960) byte-identical to what turn 1's
    acks shipped, its last block [960, 1000) PCC >= 0.99 against it (the harness's prefix / last-block rule,
    validation.py:55-61); every chip of a record's device group holds the same bytes
  - WINDOW_KV_KEEP=snapshot128 (P.42, owner-authorized contract change 2026-11-07): 0_window_kv is the 128-row
    snapshot. At every ack and at the end its 4 records hold the golden window KV of positions [cap - 128, cap) at row
    p % 128, cap = end // 64 x 64 of the chunk's end (P.49) (caches.snapshot_failures: the golden rows in position
    order through golden_failures, the same limits as before; rows of no position < cap zero). Ends not multiples of
    64: 2536, 6072 and the chunk ends; then one more request shorter than 128 (slot 1, [0, 100): cap 64, rows 64..127
    zero), its acks and snapshot checked the same way. The snapshot is rewritten every chunk by design, so the reused-
    prefix byte check and the last-block check apply to the per-position caches only. REINDEX_SERVE=alias (P.42): a
    re-index layer's records are read through its own table entries, which alias layer 20's records (byte-equal by
    construction; table_rules checks the entries)
  - PREFILL_END=cap (P.49, owner-authorized contract change 2026-11-12; DS41-219): the runtime computes and writes a
    chunk only below decode's cap of its actual_end (caches.served_end: actual_end // 64 x 64; the engine's
    actual_end is prompt_len - 1 under PD). Every per-position check above then covers [lo, cap) and [0, cap); rows
    in [cap, actual_end) are decode's replay and are not compared; the snapshot's cap is the same in both modes.
    The slot-0 last-block check covers [prefix, cap(t1)), empty in this plan (prefix 2496 = cap(2536)): printed, not
    run. PREFILL_END=end keeps every range as before.
"""

import pytest

from models.demos.common.bringup.testing.harness import device_timeout, mesh_parametrize, spec
from models.demos.common.prefill.runners.layer_completion_ring_writer import LayerCompletionSink
from models.demos.deepseek_v41_flash_d_p.tests.bringup.contract import caches as C
from models.demos.deepseek_v41_flash_d_p.tests.bringup.contract import server_rules as R

S = spec()
BATCH_SHARED = True  # FIX.29: takes the model of a bring-up batch process (core/batch.shareable)
pytestmark = device_timeout(S)
N_SLOTS = 2


@mesh_parametrize
def test_adapter_acks(mesh_device, tmp_path):
    if mesh_device is None:
        pytest.skip("device test")
    import ttnn

    R.self_check()
    _, kvc, _ = R.harness()
    gk = gk_full = C.GoldenKV(window="drift")  # FIX.19 (owner 2026-11-01): every position vs the official drift
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

    def on_layer(layer_idx, chunk):
        acks.append((int(layer_idx), chunk))
        s, lo, hi = cur["slot"], cur["start"], cur["end"]
        gk = cur.get("gk") or gk_full  # (FIX.28: the decode requests' golden also holds the handoff snapshots)
        reader.where = f"ack layer {layer_idx} slot {s} [{lo}, {hi})"
        for cfg in cfgs:
            if layer_idx not in cfg["layers"]:
                continue
            recs = reader.records(cfg["name"], s, [layer_idx], lo, hi)
            cur["reads"][(cfg["name"], layer_idx)] = recs
            if cfg.get("snapshot") and layer_idx in cur.get("unsnap", ()):
                continue  # (FIX.28: DECODER_SKIP=decode never writes the decoder's snapshot; decode does not read it)
            if cfg.get("snapshot"):  # P.42: the 128-row snapshot at the chunk end's cap
                cur["fails"] += C.snapshot_failures(
                    kvc, gk, cfg, recs, layer_idx, C.snapshot_cap(hi), thr, f"ack slot {s} [{lo}, {hi})"
                )
                continue
            shipped.setdefault((s, cur["turn"]), {}).setdefault(cfg["name"], {}).update(recs)
            hs = C.served_end(hi)  # (P.49: PREFILL_END=cap computes and writes [lo, cap(hi)) only)
            if hs <= lo:
                continue
            got, pad = C.valid_and_pad(recs, cfg, layer_idx, lo, hs)
            cur["fails"] += C.golden_failures(kvc, gk, cfg, got, layer_idx, lo, hs, thr, f"ack slot {s} [{lo}, {hs})")
            if pad.size and not (pad == 0).all():
                cur["fails"].append(f"ack slot {s} {cfg['name']} layer {layer_idx}: pad rows after {hs} not zero")
            if cfg.get("fp8"):  # P.39: Blaze's 192-byte rows; a re-index layer's are its key source's, byte for byte
                rb, pb = C.valid_and_pad(recs, cfg, layer_idx, lo, hs, raw_rows=True)
                cur["fails"] += C.fp8_row_failures(rb, pb, cfg, layer_idx, f"ack slot {s} [{lo}, {hs})")
                src = cfg["of"].get(layer_idx)
                if src is not None:
                    mine = {(src, p): b for (_, p), b in recs.items()}
                    theirs = reader.records(cfg["name"], s, [src], lo, hi)
                    cur["fails"] += [
                        f"ack slot {s} [{lo}, {hs}): re-index layer {layer_idx} vs its source: {x}"
                        for x in C.bytes_equal(mine, theirs, cfg, [src], lo, hs)
                    ]

    class Sink(LayerCompletionSink):
        def layers_completed(self, layer_start, layer_end, slot_id, pos_start, pos_end):
            for layer_idx in range(layer_start, layer_end):
                on_layer(layer_idx, (int(slot_id), int(pos_start), int(pos_end)))

    runtime.set_layer_completion_sink(Sink())
    ends = {}

    def run_chunk(k, p, **header):
        inp = C.payload_tensor(mesh_device, R.server_payload(tokens[p.start : p.start + R.CHUNK], p.start, p.end))
        n0 = len(acks)
        out = []
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
                **header,  # (FIX.28: DECODER_SKIP=decode's global_write_start, last=False)
            )
        except Exception as err:
            pytest.fail(f"not built: chunk {k} slot {p.slot} [{p.start}, {p.end}) rejected: {err}", pytrace=False)
        ttnn.synchronize_device(mesh_device)
        want = [(i, (p.slot, p.start, p.end)) for i in range(len(layers))]
        if acks[n0:] != want:
            out.append(f"chunk {k} [{p.start}, {p.end}): acks {acks[n0:]}, want one per layer in order {want}")
        out += [f"chunk {k}: {x}" for x in cur["fails"]]
        early = [
            (name, layer)
            for (name, layer), recs in cur["reads"].items()
            if reader.records(name, p.slot, [layer], p.start, p.end) != recs
        ]
        if early:
            out.append(f"chunk {k}: records changed after their layer's ack (acked before the KV was in DRAM): {early}")
        ends[p.slot] = p.end
        print(f"[adapter acks] chunk {k} slot {p.slot} [{p.start}, {p.end}): {len(acks) - n0} acks")
        return out

    plan = R.interleave(R.ACK_TURNS)
    for k, p in enumerate(plan):
        cur.update(slot=p.slot, start=p.start, end=p.end, turn=p.turn, fails=[], reads={})
        fails += run_chunk(k, p)

    for s, e in ends.items():
        for cfg in cfgs:
            recs = reader.records(cfg["name"], s, cfg["layers"], 0, e)
            if cfg.get("snapshot"):  # P.42: the request's final snapshot at its cap
                for layer in cfg["layers"]:
                    mine = {key: b for key, b in recs.items() if key[0] == layer}
                    fails += C.snapshot_failures(
                        kvc, gk, cfg, mine, layer, C.snapshot_cap(e), thr, f"final slot {s} [0, {e})"
                    )
                continue
            es = C.served_end(e)  # (P.49: decode's cap under PREFILL_END=cap)
            for layer in cfg["layers"]:
                if es <= 0:
                    continue
                got, _ = C.valid_and_pad(recs, cfg, layer, 0, es)
                fails += C.golden_failures(kvc, gk, cfg, got, layer, 0, es, thr, f"final slot {s} [0, {es})")
                if cfg.get("fp8"):  # P.39: the served bytes' format over the whole request
                    rb, pb = C.valid_and_pad(recs, cfg, layer, 0, es, raw_rows=True)
                    fails += C.fp8_row_failures(rb, pb, cfg, layer, f"final slot {s} [0, {es})")
    t1, t2 = R.ACK_TURNS[0]
    prefix = R.follow_up_resident(t1, t2)
    # P.49: under PREFILL_END=cap turn 1 computes [0, cap(t1)); its last block [prefix, cap(t1)) is then empty when
    # prefix == cap(t1) (this plan: 2496 = cap(2536)): nothing turn 1 shipped there to compare (reported, never hidden)
    t1_end = C.served_end(t1)
    if t1_end <= prefix:
        print(
            f"[adapter acks] last-block check [{prefix}, {t1}) not run: turn 1 computed [0, {t1_end}) (PREFILL_END=cap)"
        )
    for cfg in cfgs:
        if cfg.get("snapshot"):  # (P.42: rewritten every chunk by design; checked at every ack and at the end above)
            continue
        final = reader.records(cfg["name"], 0, cfg["layers"], 0, t1)
        first = shipped[(0, 0)][cfg["name"]]
        fails += [f"slot 0 reused prefix: {x}" for x in C.bytes_equal(first, final, cfg, cfg["layers"], 0, prefix)]
        for layer in [] if t1_end <= prefix else cfg["layers"]:
            a, _ = C.valid_and_pad(first, cfg, layer, prefix, t1_end)
            b, _ = C.valid_and_pad(final, cfg, layer, prefix, t1_end)
            what = f"slot 0 last block [{prefix}, {t1_end})"
            fails += C.pcc_failures(kvc, cfg, b, a, layer, R.LAST_BLOCK_PCC, what)
            gk.note_err("lastblock", cfg, layer, b, a, what)  # P.26, never gated
    if any(c.get("snapshot") for c in cfgs):  # P.42: a request shorter than 128 (cap 64) on slot 1, cold
        short = R.Push(slot=1, chunk_idx=0, start=0, end=100, is_last=True, turn=len(R.ACK_TURNS[1]))
        cur.update(slot=1, start=0, end=100, turn=short.turn, fails=[], reads={})
        fails += run_chunk(len(plan), short)
    fails += decoder_skip_requests(mesh_device, runtime, reader, cfgs, layers, cur, run_chunk, kvc, thr)
    if reader.replica_mismatch:
        print(
            f"[adapter acks] replica mismatches (config, slot, layer, pos, chips, group, where): {reader.replica_detail}"
        )
        fails.append(
            f"device-group replicas differ at {len(reader.replica_mismatch)} records: {reader.replica_mismatch[:3]}"
        )
    fails += gk.coverage_failures(cfgs, "golden")
    print(f"[adapter acks] golden comparison per config and mode: {gk.coverage_summary()}")
    print(f"[adapter acks] window KV of the dumped layers, worst range per channel: {gk.drift_summary()}")
    # P.26 (owner 2026-11-01): the relative L2 error next to every window-KV PCC check, recorded, never gated
    print(
        f"[adapter acks] window KV relative error, worst per check kind / layer / channel: {gk.record_errors('acks')}"
    )
    print(f"[adapter acks] layers {layers}, configs {[c['name'] for c in cfgs]}: {len(fails)} failures")
    assert not fails, "\n".join(fails[:40])


def decoder_skip_requests(mesh_device, runtime, reader, cfgs, layers, cur, run_chunk, kvc, thr) -> list[str]:
    """FIX.28 (OWNER 2026-11-13): DECODER_SKIP=decode (P.51, Kartik's cut) as tt-d-gen sends it, on the model this test
    built (loaded with the shipped DECODER_SKIP, off; ``use_decoder_skip("decode")``, back to the loaded value after).
    A prompt of N = MAX_SEQ - 32 golden tokens (the contract's request) stops at P = (N - 128) // 64 x 64
    (tt/attention.decoder_skip_stop); every chunk [k chunk, min((k + 1) chunk, end)) carries the header's
    global_write_start = actual_start and last=False. Two requests, cold:
      slot 0  end = P            (what tt-d-gen sends)
      slot 1  end = P + 40       (an actual_end not a multiple of 64: serving_contract.md P.49's rule, which P.51 keeps:
                                  the runtime stops at decode's cap = end // 64 x 64 = P)
    Checked: every ack as in the default plan (one per layer in order naming the engine's chunk, each layer's history
    rows of the chunk below its served end vs the golden at the ack, re-read after the chunk: same bytes) with layers
    0-19's snapshot at the chunk end's cap and layers 20-39's snapshot not read (decode never writes it and does not
    read it); the served end = decode's cap of the request's end = P; at the end the history caches over [0, P) and
    layers 0-19's snapshot at P vs the golden (layers without dumps: the golden's handoff snapshot at P, reference/
    handoff.py; caches.GoldenKV(handoffs=True)), every (config, layer) decode is served compared at least once; and no
    history record of [P, N) written (the same bytes before and after the request). Metrics decode_acks_failures,
    decode_served_end, decode_unwritten_records_changed (asserted 0 through the test's failures)."""
    from models.demos.common.bringup.core import metrics
    from models.demos.deepseek_v41_flash_d_p.tt.attention import decoder_skip_stop

    model = runtime.model
    gkd = C.GoldenKV(window="drift", handoffs=True)
    n = R.MAX_SEQ - 32
    p_stop = decoder_skip_stop(n)
    hist = [c for c in cfgs if not c.get("snapshot")]
    out, changed, checked = [], 0, 0
    model.use_decoder_skip("decode")
    try:
        enc = [i for i in layers if i not in model.decoder]
        dec = [i for i in layers if i in model.decoder]
        served = [dict(c, layers=enc) if c.get("snapshot") else c for c in cfgs]
        for slot, end in ((0, p_stop), (1, p_stop + 40)):
            cap = C.served_end(end)
            what = f"decode slot {slot} [0, {end})"
            if cap != C.snapshot_cap(end) or cap != p_stop:
                out.append(f"{what}: served end {cap}, decode's cap {C.snapshot_cap(end)}, P {p_stop}")
            lo_w = -(-cap // 32) * 32
            before = {c["name"]: reader.records(c["name"], slot, c["layers"], lo_w, n) for c in hist}
            bounds = [(a, min(a + R.CHUNK, end)) for a in range(0, end, R.CHUNK)]
            for k, (a, b) in enumerate(bounds):
                push = R.Push(slot=slot, chunk_idx=k, start=a, end=b, is_last=k == len(bounds) - 1, turn=100 + slot)
                cur.update(slot=slot, start=a, end=b, turn=push.turn, fails=[], reads={}, gk=gkd, unsnap=set(dec))
                out += [f"{what}: {x}" for x in run_chunk(1000 + k, push, global_write_start=a, last=False)]
            cur.update(gk=None, unsnap=set())
            for c in hist:
                after = reader.records(c["name"], slot, c["layers"], lo_w, n)
                diff = sorted(key for key in before[c["name"]] if after.get(key) != before[c["name"]][key])
                changed += len(diff)
                checked += len(before[c["name"]])
                if diff:
                    out.append(f"{what}: {c['name']} records of [{lo_w}, {n}) written past the stop: {diff[:6]}")
            for c in served:
                recs = reader.records(c["name"], slot, c["layers"], 0, cap)
                for layer in c["layers"]:
                    if c.get("snapshot"):
                        mine = {key: b for key, b in recs.items() if key[0] == layer}
                        out += C.snapshot_failures(kvc, gkd, c, mine, layer, cap, thr, f"{what} final")
                        continue
                    got, _ = C.valid_and_pad(recs, c, layer, 0, cap)
                    out += C.golden_failures(kvc, gkd, c, got, layer, 0, cap, thr, f"{what} final [0, {cap})")
                    if c.get("fp8"):
                        rb, pb = C.valid_and_pad(recs, c, layer, 0, cap, raw_rows=True)
                        out += C.fp8_row_failures(rb, pb, c, layer, f"{what} final [0, {cap})")
            print(f"[adapter acks] DECODER_SKIP=decode slot {slot} [0, {end}): served end {cap} (P {p_stop})")
    finally:
        model.use_decoder_skip(model.loaded_skip)
    out += gkd.coverage_failures(served, "decode golden")
    print(f"[adapter acks] DECODER_SKIP=decode golden comparison per config and mode: {gkd.coverage_summary()}")
    print(f"[adapter acks] DECODER_SKIP=decode window KV, worst range per channel: {gkd.drift_summary()}")
    metrics.record("decode_acks_failures", len(out))
    metrics.record("decode_served_end", p_stop)
    metrics.record("decode_unwritten_records_changed", changed)
    metrics.record("decode_unwritten_records_checked", checked)
    print(f"[adapter acks] DECODER_SKIP=decode: {checked} records past P checked unwritten, {len(out)} failures")
    return out
