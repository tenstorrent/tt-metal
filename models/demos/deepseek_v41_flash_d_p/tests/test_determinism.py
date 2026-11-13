# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Determinism (FIX.23, owner 2026-11-12): identical inputs give identical outputs, bit for bit.

FIX.25: the check itself is the framework's (testing/model_determinism.py: novel and code prompts interleaved, cold
first sequence after the load, warm ones after the other prompt, every chunk's final hidden and every layer's served
state bit for bit), which every model-level gate now runs (core/gate.py with_determinism). This file keeps the gates
that name it (P.49, FIX.25, FIX.26) and P.50's A/B metric named by the COMBINE_SPLIT side.

DS41-233 / DS41-234 / DS41-235: the MoE combine sums the routed terms in packet-arrival order; with sums above 32 the
single snapped fp32 sum depended on that order (fixed by COMBINE_SPLIT=on).

FIX.28 (OWNER 2026-11-13): the same check also runs DECODER_SKIP=decode (P.51, Kartik's cut), the path the serving
runtime takes under it, on the same loaded model (``use_decoder_skip("decode")``; back to the loaded value, off, after:
the full-path check above is unchanged and runs first). Per prompt (novel, code; cold, then warm after the other) one
sequence the way tt-d-gen drives a slot (``decode_sequence``): a prompt of N = DET_CHUNKS x chunk tokens stops at
P = (N - 128) // 64 x 64 (tt/attention.decoder_skip_stop, exclusive); request A [0, Xp) (Xp = 1.5 chunks, the
frontier its caches then hold), then request B on that prefix from Xp - 128 with global_write_start Xp (its first
chunk replays [Xp - 128, Xp) with no history-cache write) up to P; every chunk last=False, global_write_start =
actual_start elsewhere, PAD_ID past its end, the slot's id history noted as the adapter does. Compared bit for bit:
after every chunk, layers 0-19's served window snapshots (every chip); after the request, every history cache decode
reads over [0, P) (compressed KV and index keys of every KV source, layer 20's included) and layers 0-19's window
rings. Metric determinism_max_abs_diff_decoder_skip (== 0 gates; the other determinism_*_decoder_skip metrics as the
framework's).

    PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all --no-precompile \\
        models/demos/deepseek_v41_flash_d_p/tests/test_determinism.py
"""

from __future__ import annotations

from models.demos.common.bringup.core import metrics
from models.demos.common.bringup.testing import batch
from models.demos.common.bringup.testing.harness import mesh_parametrize, spec
from models.demos.common.bringup.testing.model_determinism import check
from models.demos.deepseek_v41_flash_d_p.tt.settings import settings

S = spec()
BATCH_SHARED = True  # FIX.29: takes the model of a bring-up batch process (core/batch.shareable)


PAD_ID = 0xFFFFFFFF


def decode_plan(n_tokens: int, chunk: int) -> list[tuple[int, int, int]]:
    """(actual_start, actual_end, global_write_start) of every chunk of DECODER_SKIP=decode's sequence (module
    docstring): request A [0, Xp), then request B from Xp - 128 (the replay) with global_write_start Xp, up to P."""
    from models.demos.deepseek_v41_flash_d_p.tt.attention import SNAP, decoder_skip_stop

    p = decoder_skip_stop(n_tokens)
    xp = (chunk + chunk // 2) // 64 * 64
    assert 0 < xp - SNAP and xp < p, (xp, p)
    plan = [(a, min(a + chunk, xp), a) for a in range(0, xp, chunk)]
    b0 = xp - SNAP
    plan += [(a, min(a + chunk, p), max(a, xp)) for a in range(b0, p, chunk)]
    return plan


def decode_sequence(model, layers, tokens, chunk: int, n: int, final_norm: bool):
    """DECODER_SKIP=decode's sequence on a fresh slot state (testing/model_determinism.check's ``run``): (layers
    0-19's served snapshots after each chunk, [(name, tensor)] of what decode reads after the request)."""
    import torch

    from models.demos.common.bringup.testing.model_determinism import read_states
    from models.demos.common.bringup.testing.positions import _free

    assert model.skip_mode == "decode", model.skip_mode
    plan = decode_plan(n * chunk, chunk)
    p = plan[-1][1]
    enc = [i for i in layers if i not in model.decoder]
    state = model.new_state(-(-max(a + chunk for a, _, _ in plan) // chunk) * chunk)
    outs = []
    try:
        for a, b, g in plan:
            ids = torch.full((chunk,), PAD_ID, dtype=torch.int64)
            ids[: b - a] = tokens[a:b].long()
            state.note_tokens(ids, a, b)  # (as the adapter: the slot's id history and every engram tail at a)
            model.begin_chunk(ids, a, last=False, write_start=g)
            assert model._skip, "DECODER_SKIP=decode: the chunk must skip the decoder (layers 21-39)"
            h = model.embed(ids, a)
            for i in layers:
                h2 = model.layer(i, h, a, state, end=b)
                model.free(h)
                h = h2
            model.free(h)
            model.sync()
            outs.append(torch.stack([torch.stack(state.window[i].snapshot_host()) for i in enc]))
        model.sync()
        st = [x for x in read_states(state, enc, p)]
        for i in model.kv_only:  # (layer 20: its compressed KV and index keys; its window never runs under decode)
            st += [(f"L{i:02d}.{k}", t) for k, t in state.served(i, p).items() if k != "0_window_kv"]
        # (what is compared holds data: every chunk's snapshots and every served history cache non-zero)
        empty = [k for k, o in enumerate(outs) if not bool(o.float().abs().sum() > 0)]
        empty += [name for name, t in st if "0_window_kv" not in name and not bool(t.float().abs().sum() > 0)]
        assert not empty, f"DECODER_SKIP=decode: all-zero served tensors {empty[:6]}"
    finally:
        _free(state)
    return outs, st


@mesh_parametrize(timeout_s=2 * 3600)
def test_determinism(mesh_device):
    layers = S.layers()
    model = batch.device_model(S, mesh_device, layers, lm_head=False)
    err, worst = check(S, model, layers, label=f"COMBINE_SPLIT={settings.get('COMBINE_SPLIT')}")
    # (P.50's A/B, owner 2026-11-12: the same metric named by the side, COMBINE_SPLIT off / on)
    metrics.record(f"determinism_max_abs_diff_{settings.get('COMBINE_SPLIT')}", worst)
    # FIX.28: DECODER_SKIP=decode, the same model (module docstring)
    model.use_decoder_skip("decode")
    try:
        err_d, worst_d = check(
            S, model, layers, label="DECODER_SKIP=decode", run=decode_sequence, suffix="_decoder_skip"
        )
    finally:
        model.use_decoder_skip(model.loaded_skip)
    print(f"DETERMINISM: full path max |diff| {worst}, DECODER_SKIP=decode max |diff| {worst_d}")
    assert err is None, err
    assert err_d is None, f"DECODER_SKIP=decode: {err_d}"
