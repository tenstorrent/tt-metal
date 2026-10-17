# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""serving_contract.md, "Input": the chunk the H2D stream delivers, for any start the server sends.

tt-d-gen fills one chunk of `chunk` uint32 tokens, PAD_ID (0xFFFFFFFF) past actual_end (prefill_writer.cpp:96), and
reorders it across the SP chips by actual_start (prefill_pipeline.cpp:149-150, ring_sdpa_reshuffle.hpp): chip c gets
the 640 tokens whose absolute positions are its block-cyclic rows. The runner hands that [sp, 1, chunk / sp] tensor,
sharded on the SP axis, to runtime.prefill_chunk(input, kv, slot_id=, actual_start=, actual_end=, request_id=).

The adapter's runtime prefills the same prompt twice into two slots: slot 0 in natural chunks from 0 (start 0,
5120), slot 1 with the server's follow-up plan over a golden-identical prefix, i.e. the starts 2944 (chip 4 holds
the first token) and 8064, ending mid-record at 9000. Pass: slot 1's KV through the table equals slot 0's for rows
[2944, 9000) (PCC >= max(serving.min_kv_pcc, thresholds.state) against each other and the golden), rows [9000, 9024)
zero. A model that reads the payload in natural order, or treats PAD_ID as a token id, fails at the starts that are
not chunk multiples.
"""

import pytest
from bringup_fw.testing.harness import device_timeout, mesh_parametrize, spec

S = spec()
PROMPT, RESIDENT = 9000, 2944


@device_timeout(S)
@mesh_parametrize
def test_input_payload(mesh_device):
    if mesh_device is None:
        pytest.skip("CPU proof mode: no mesh")
    import tempfile

    from bringup_fw.core import metrics
    from bringup_fw.testing import chunk_starts as CS
    from bringup_fw.testing import contract, kv_table
    from bringup_fw.testing import server_rules as R

    import ttnn
    from models.demos.gemma4_31b_glx.tests.bringup.contract import serving_helpers as H

    chunk, max_seq, sp = H.geometry(S)
    g = H.target_golden(S)
    tokens = g.tokens()
    adapter, kv, rt = H.adapter_runtime(S, mesh_device)
    rt.set_layer_completion_sink(lambda layer, rid: None)
    rt.compile(kv)
    cold = R.chunk_plan(PROMPT, 0, chunk, max_seq)
    warm = R.chunk_plan(PROMPT, RESIDENT, chunk, max_seq)
    assert warm[0][0] == RESIDENT and RESIDENT % chunk, warm
    rid = 0
    for slot, plan in ((0, cold), (1, [(0, RESIDENT)]), (1, warm)):
        for lo, hi in plan:
            inp = H.server_input(mesh_device, tokens[lo:hi].tolist(), lo, hi, chunk, sp)
            rt.prefill_chunk(inp, kv, slot_id=slot, actual_start=lo, actual_end=hi, request_id=rid)
            rid += 1
    ttnn.synchronize_device(mesh_device)
    with tempfile.TemporaryDirectory() as td:
        table = H.export_table(rt, kv, td)
    reader = kv_table.TableReader(table, contract.device_map(mesh_device))
    _, kvc = H.dgen_checks()
    thr, dtype = CS.kv_threshold(S), H.DUMP_DTYPE[H.kv_dtype_name(S)]
    layers = H.sample_layers(contract.served_layers(S)[1])
    fails = []
    for slot in (0, 1):
        f, scores = H.read_back_failures(reader, kvc, g.state, slot, PROMPT, layers, thr, dtype, "input")
        fails += f
        for k, v in scores.items():
            metrics.record(f"input_pcc_s{slot}_{k.replace(':', '_c')}", v)
    lo, hi = R.align_down(RESIDENT, 32), R.ceil_to(PROMPT, 32)
    for layer in layers:
        for c in H.SAMPLE_CONFIGS["global" if H.is_global(layer) else "sliding"]:
            w = H.config_width(c)
            a = {(layer, p): reader.record(0, layer, p, c) for p in range(lo, hi, 32)}
            b = {(layer, p): reader.record(1, layer, p, c) for p in range(lo, hi, 32)}
            res = kvc.compare(
                a,
                b,
                method="pcc",
                layers=[layer],
                start=RESIDENT,
                end=PROMPT,
                threshold=thr,
                dtype=dtype,
                width=w,
                split=0,
                storage="tile",
            )
            if res["status"] != "passed":
                fails.append(
                    f"input: layer {layer} {H.CONFIG_NAMES[c]}: the follow-up plan's KV != the cold plan's "
                    f"over [{RESIDENT}, {PROMPT}): {res['findings'][:2]}"
                )
    metrics.record("input_checks_failed", len(fails))
    for f in fails:
        print(f"FAIL {f}")
    assert not fails, f"{len(fails)} checks failed: {fails[:4]}"
