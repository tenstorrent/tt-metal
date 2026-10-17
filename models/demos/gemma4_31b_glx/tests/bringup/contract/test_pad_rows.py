# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""serving_contract.md, "Attention and cache writes": a follow-up chunk over a global layer, ending mid-record.

The framework's chunk-start tests (L.starts) cover layers 0-1, both sliding. This one adds Gemma's global layer
(layer 5: the packed 640-wide K/V cache, partial RoPE, its own head count) on the server's chunk for a follow-up turn:
resident 2944 (64-aligned, not chunk-aligned: chip 4 of 8 holds the first token, tt-d-gen ring_sdpa_reshuffle.hpp),
prompt 8017 (ends 17 rows into a 32-token record), so one chunk [2944, 8017) (prefill_writer.cpp chunk plan).
Layers 0-5 start from the golden KV for [0, 2944). Pass, every layer:
  - rows below 2944 unchanged; rows [2944, 8017) vs the golden KV, PCC >= max(serving.min_kv_pcc, thresholds.state);
  - rows [8017, 8032) exactly zero (the last record ships whole; tt-d-gen backend_runtime.cpp kv_block_size % 32);
  - the same chunk sent again rewrites every row with the same bytes (a recompute the server may send).
"""

import pytest
from bringup_fw.testing.harness import device_timeout, mesh_parametrize, spec

S = spec()
LAYERS = list(range(6))
RESIDENT, PROMPT = 2944, 8017


@device_timeout(S)
@mesh_parametrize
def test_pad_rows(mesh_device):
    if mesh_device is None:
        pytest.skip("CPU proof mode: no mesh")
    import torch
    from bringup_fw.core import metrics
    from bringup_fw.testing import chunk_starts as CS
    from bringup_fw.testing import server_rules as R

    from models.demos.gemma4_31b_glx.tests.bringup.contract import serving_helpers as H

    chunk, max_seq, _ = H.geometry(S)
    plan = R.chunk_plan(PROMPT, RESIDENT, chunk, max_seq)
    assert plan == [(RESIDENT, PROMPT)], plan  # the server's plan, not the model's
    g = H.target_golden(S)
    try:
        mod = CS._Device(S, mesh_device, LAYERS, max_seq, chunk)
    except NotImplementedError as e:
        H.not_built(f"hooks.device_model: {e}")
    except CS.NotBuilt as e:
        H.not_built(str(e))
    tokens = g.tokens()
    held = CS.load_prefix(S, mod, g, LAYERS, RESIDENT)
    try:
        mod.run(tokens, plan)
    except CS.NotBuilt as e:
        H.not_built(str(e))
    fails = CS.check_kv(S, mod, g, LAYERS, held, RESIDENT, PROMPT, "pad_rows")
    pad_end = R.ceil_to(PROMPT, R.RECORD_TOKENS)
    first = {i: {k: CS._rows(v, pad_end).clone() for k, v in mod.read(i, pad_end).items()} for i in LAYERS}
    mod.run(tokens, plan)
    same = True
    for i in LAYERS:
        for k, v in mod.read(i, pad_end).items():
            if not torch.equal(CS._rows(v, pad_end), first[i][k]):
                same = False
                fails.append(f"pad_rows: layer {i} {k}: the same chunk sent twice wrote different bytes")
    metrics.record("pad_rows_rewrite_identical", int(same))
    mod.free()
    CS.report("pad_rows", fails)
    assert not fails, f"{len(fails)} checks failed: {fails[:4]}"
