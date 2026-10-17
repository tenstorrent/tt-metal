# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""serving_contract.md, "Acks": one ack per layer per chunk, each after that layer's KV is in DRAM, on the hybrid cache.

tt-d-gen's reader retires a chunk after layers_per_chunk acks (prefill_reader.cpp:157) and migrates what an ack
covers. The framework's adapter test (testing/contract.py) checks acks on config 0 only at the chunk's first and last
record; Gemma's table has 36 configs and a global layer writes configs 0-3 only, a sliding layer 4-35 only. Here, on
the server's interleave of two slots (a cold 8017-token prompt on slot 0, a follow-up 2944 -> 9000 on slot 1, so the
chunk ends mid-record and starts off a chunk multiple), at every ack the sink reads, for its layer, the first and the
last record of the chunk of every config of the layer's own type. Pass: acks are (layer 0..59, request) in order for
every chunk; no acknowledged record changes afterwards; the last record's pad rows are zero at the ack.
"""

import pytest
from bringup_fw.testing.harness import device_timeout, mesh_parametrize, spec

S = spec()


@device_timeout(S)
@mesh_parametrize
def test_ack_timing(mesh_device):
    if mesh_device is None:
        pytest.skip("CPU proof mode: no mesh")
    import tempfile

    from bringup_fw.core import metrics
    from bringup_fw.testing import contract
    from bringup_fw.testing import server_rules as R

    import ttnn
    from models.demos.gemma4_31b_glx.tests.bringup.contract import serving_helpers as H

    chunk, max_seq, sp = H.geometry(S)
    g = H.target_golden(S)
    tokens = g.tokens().tolist()
    _, n_layers = contract.served_layers(S)
    adapter, kv, rt = H.adapter_runtime(S, mesh_device)
    with tempfile.TemporaryDirectory() as td:
        table = H.export_table(rt, kv, td)
    pushes = [(0, lo, hi) for lo, hi in R.chunk_plan(8017, 0, chunk, max_seq)]
    pushes += [(1, lo, hi) for lo, hi in R.chunk_plan(9000, 2944, chunk, max_seq)]
    pushes = [pushes[0], pushes[2], pushes[1], pushes[3]]  # the server's round robin of two queued slots
    acks, early, cur = [], {}, {}

    def read(layer, pos, slot, cid):
        return bytes(table.read_device_chunk(layer=layer, position=pos, slot=slot, config_id=cid))

    def sink(layer, rid):
        acks.append((layer, rid))
        if rid not in cur:
            return
        slot, lo, hi = cur[rid]
        for pos in {R.align_down(lo, 32), R.align_down(hi - 1, 32)}:
            for cid in H.layer_configs(layer):
                early[(layer, rid, slot, pos, cid)] = read(layer, pos, slot, cid)

    rt.set_layer_completion_sink(sink)
    rt.compile(kv)
    acks.clear()
    # slot 1's resident prefix [0, 2944), as an earlier turn left it
    pre = [(1, lo, hi) for lo, hi in R.chunk_plan(2944, 0, chunk, max_seq)]
    for rid, (slot, lo, hi) in enumerate(pre + pushes):
        if rid >= len(pre):
            cur[rid] = (slot, lo, hi)
        inp = H.server_input(mesh_device, tokens[lo:hi], lo, hi, chunk, sp)
        rt.prefill_chunk(inp, kv, slot_id=slot, actual_start=lo, actual_end=hi, request_id=rid)
    ttnn.synchronize_device(mesh_device)

    fails = []
    want = [(layer, rid) for rid in range(len(pre) + len(pushes)) for layer in range(n_layers)]
    if acks != want:
        fails.append(f"acks {acks[:4]}... != one per layer in order per chunk ({len(acks)} vs {len(want)})")
    changed = [k for k, data in early.items() if read(k[0], k[3], k[2], k[4]) != data]
    metrics.record("ack_blocks_checked", len(early))
    metrics.record("acks_early", len(changed))
    if not early:
        fails.append("no record read at an ack (the sink was never called for the checked chunks)")
    if changed:
        fails.append(
            f"{len(changed)} of {len(early)} records changed after their layer's ack, e.g. "
            f"(layer, request, slot, pos, config) {changed[:3]}"
        )
    _, kvc = H.dgen_checks()
    dtype = H.DUMP_DTYPE[H.kv_dtype_name(S)]
    for (layer, rid, slot, pos, cid), data in early.items():
        hi = cur[rid][2]
        if pos <= hi - 1 < pos + 32 and hi % 32:
            rows = kvc.decode_chunk(data, dtype=dtype, width=H.config_width(cid), storage="tile")
            if (rows[hi - pos :] != 0).any():
                fails.append(
                    f"layer {layer} {H.CONFIG_NAMES[cid]} slot {slot}: pad rows [{hi}, {pos + 32}) not zero "
                    "at the ack"
                )
                break
    metrics.record("ack_checks_failed", len(fails))
    for f in fails:
        print(f"FAIL ack_timing: {f}")
    assert not fails, f"{len(fails)} ack checks failed: {fails[:4]}"
