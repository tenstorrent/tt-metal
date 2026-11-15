# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""P.70 (DS41-280): MOE_REPLICAS=prev under DECODER_SKIP=decode - the plans' period from the MoE layers a chunk calls.

Under decode a not-last chunk calls the MoE of layers 0-19 only (layer 20 its KV head, 21-39 never): 20 calls a chunk,
so a layer's next call comes 20 ordinals after its plan. Before P.70 every plan assumed 40 (every MoE layer built) and
the guard never let one be used (P.69: ready 0 of 20). This test runs real back-to-back chunks of the ladder's prompt
(novel), last=False, on the model run as decode (use_decoder_skip, FIX.28), with the shipped settings plus
MOE_REPLICAS=prev MOE_FILL=piggyback ENGRAM_TABLE=resident MOE_COUNTS=on (test_moe_prev_replicas' load: the counts
are the host mirror's input), sides switched in the loaded model as test_moe_prev_replicas does:

    off      prev switched off (the empty map, no planner): two passes, the reference (layer 19's output per chunk)
             and a repeat that must equal it bit for bit
    prev     one warm pass, then two passes: each chunk's layer 19 output must equal off's bit for bit; the second pass
             replays every plan on the host (expert_prev.plan_host at the chunk's period) and compares the global
             state, every layer's state and slot map, and the pending job pages

Metrics: prev_decode_ready_<c> (MoE calls whose plan was used, chunk c), prev_decode_calls_<c>, prev_decode_period
(the period the blocks took), prev_decode_maxdiff (prev vs off), prev_decode_repeat_maxdiff (prev vs prev),
prev_decode_off_repeat_maxdiff, prev_decode_mirror_mismatch, prev_decode_mirror_checks. Asserts: every diff 0, no
mirror mismatch, and chunks 1.. use plans (ready > 0).

    PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all --no-precompile \\
        models/demos/deepseek_v41_flash_d_p/tests/test_moe_prev_decode.py
"""

from __future__ import annotations

import time

from loguru import logger

import ttnn
from models.demos.common.bringup.core import metrics
from models.demos.common.bringup.testing import batch
from models.demos.common.bringup.testing.harness import mesh_parametrize, spec
from models.demos.common.bringup.testing.positions import _free

S = spec()
BATCH_SHARED = True
CHUNKS = 6
SEQ = CHUNKS * 5120


def _pass(model, mesh, layers, tokens, side, enc_last, ref=None, keep=None, mirror=None) -> list:
    """Chunks 0..CHUNKS-1 of one fresh request (last=False: the decoder skipped) -> per chunk {diff, prev stats}."""
    from models.demos.deepseek_v41_flash_d_p.tt import moe_block_main_adapter as A

    chunk = int(S.get("target.chunk"))
    state = model.new_state(SEQ)
    rows = []
    try:
        for c in range(CHUNKS):
            st = c * chunk
            t = tokens[st : st + chunk]
            ps0 = dict(A.prev_stats(mesh))
            model.begin_chunk(t, st, last=False)
            h = model.embed(t)
            out = None
            for i in layers:
                h2 = model.layer(i, h, st, state)
                model.free(h)
                h = h2
                if i == enc_last:
                    out = ttnn.clone(h.streams)
            model.free(h)
            ttnn.synchronize_device(mesh)
            row = {"chunk": c}
            if ref is not None:
                d = ttnn.max(ttnn.abs(ttnn.sub(out, ref[c])))
                row["diff"] = max(float(ttnn.to_torch(x).float().abs().max()) for x in ttnn.get_device_tensors(d))
                ttnn.deallocate(d)
            if keep is not None:
                keep[c] = out
            else:
                ttnn.deallocate(out)
            rec = A.flush_counts(mesh, st, chunk)
            ps1 = A.prev_stats(mesh)
            row["prev"] = {k: ps1.get(k, 0) - ps0.get(k, 0) for k in ps1}
            if mirror is not None and rec is not None:
                mirror.chunk(rec, row["prev"])
            logger.info(f"P.70 decode {side} chunk {c}: diff {row.get('diff')}, prev {row['prev']}")
            rows.append(row)
        if mirror is not None:
            mirror.pages()
    finally:
        _free(state)
    return rows


@mesh_parametrize(timeout_s=2 * 3600)
def test_moe_prev_decode(mesh_device, monkeypatch):
    from models.demos.deepseek_v41_flash_d_p.tests import test_moe_prev_replicas as T
    from models.demos.deepseek_v41_flash_d_p.tests.test_moe_counts import _ids
    from models.demos.deepseek_v41_flash_d_p.tt.settings import settings

    monkeypatch.setenv("DS41_MOE_REPLICAS", "prev")
    monkeypatch.setenv("DS41_MOE_FILL", "piggyback")
    monkeypatch.setenv("DS41_ENGRAM_TABLE", "resident")
    monkeypatch.setenv("DS41_MOE_COUNTS", "on")
    monkeypatch.delenv("DS41_MOE_COUNTS_DIR", raising=False)
    settings.thaw()
    layers = list(S.layers())
    model = batch.device_model(S, mesh_device, layers, lm_head=False)
    loaded = model.skip_mode
    model.use_decoder_skip("decode")
    t0 = time.time()
    import torch

    tokens = torch.tensor(_ids("novel"))
    enc = [i for i in layers if i not in model._decoder_set]
    enc_last = enc[-1]
    ref: dict = {}
    fails = []
    try:
        T._side(mesh_device, "off")
        _pass(model, mesh_device, layers, tokens, "off", enc_last)  # (warm)
        _pass(model, mesh_device, layers, tokens, "off", enc_last, keep=ref)
        off_rep = _pass(model, mesh_device, layers, tokens, "off", enc_last, ref=ref)
        T._side(mesh_device, "prev")
        _pass(model, mesh_device, layers, tokens, "prev", enc_last)  # (warm: the planner, the exact combine)
        p1 = _pass(model, mesh_device, layers, tokens, "prev", enc_last, ref=ref)
        mirror = T._Mirror(mesh_device, enc)
        p2 = _pass(model, mesh_device, layers, tokens, "prev", enc_last, ref=ref, mirror=mirror)
        periods = sorted({pr.period for pr in T._prevs(mesh_device)})
        nmain = len([L for L in enc if L in T._main()])
    finally:
        T._side(mesh_device, "prev")
        for v in ref.values():
            ttnn.deallocate(v)
        model.use_decoder_skip(loaded)

    mx = lambda rows: max(r["diff"] for r in rows)  # noqa: E731
    metrics.record("prev_decode_off_repeat_maxdiff", mx(off_rep))
    metrics.record("prev_decode_maxdiff", mx(p1))
    metrics.record("prev_decode_repeat_maxdiff", mx(p2))
    metrics.record("prev_decode_period", periods[0] if len(periods) == 1 else str(periods))
    metrics.record("prev_decode_mirror_mismatch", len(mirror.mismatch))
    metrics.record("prev_decode_mirror_checks", mirror.checks)
    for r in p2:
        metrics.record(f"prev_decode_ready_{r['chunk']}", r["prev"].get("ready", 0))
        metrics.record(f"prev_decode_calls_{r['chunk']}", r["prev"].get("calls", 0))
    print(
        "P.70 decode: period "
        + str(periods)
        + "; ready / calls a chunk "
        + ", ".join(f"c{r['chunk']} {r['prev'].get('ready', 0)}/{r['prev'].get('calls', 0)}" for r in p2)
        + f"; maxdiff prev-off {mx(p1)} / {mx(p2)}, off repeat {mx(off_rep)}; mirror {len(mirror.mismatch)} mismatches"
        f" of {mirror.checks} checks; {time.time() - t0:.0f} s"
    )
    if mx(off_rep) != 0.0:
        fails.append(f"two off passes differ ({mx(off_rep)})")
    if mx(p1) != 0.0 or mx(p2) != 0.0:
        fails.append(f"prev differs from off ({mx(p1)}, {mx(p2)})")
    if mirror.mismatch:
        fails.append(f"{len(mirror.mismatch)} mirror mismatches, first {mirror.mismatch[:3]}")
    if mirror.checks == 0:
        fails.append("the mirror checked nothing")
    if periods != [nmain]:
        fails.append(f"periods {periods}, want [{nmain}] (the MoE layers a decode chunk calls)")
    if any(r["prev"].get("ready", 0) == 0 for r in p2[1:]):
        fails.append("a chunk after the first used no plan")
    assert not fails, fails
