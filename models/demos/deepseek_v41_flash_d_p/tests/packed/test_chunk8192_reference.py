# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""PB.2 (owner 2026-11-19): ONE request in 8192-row chunks on today's path (PACKED=off), against the s56320 golden with
the EXISTING limits. Written in PB.1 (bringup/PACKED_DESIGN.md "8192 on today's tree"), not run there (no device).

The goldens are 5120-chunked (s56320_c5120: chunk_00..chunk_10 outputs, growing states final, the fixed-size states
- window ring, engram tail - snapshotted at 51200 only). The reference's output at a position does not depend on the
chunking (every GEMM in fixed row tiles, kernels.py "GEMM tiling"; exact causal attention over the slot's state; the
compressor's groups are 2-aligned and no 32-aligned start leaves one pending), so a golden row is the target of the same
position however the device chunks the prompt. Two cases (parametrized), each with its limit from an existing drift
file (testing/e2e_limits: calib x the official reference's drift on that rung, F128 / F134), no new limit:

  from0   every 8192 chunk from position 0: [0, 8192), ..., [40960, 49152) full, then [49152, 57344) with actual_end
          56320 (7168 real rows, 1024 pad: the padded tail at 8192). Rows [51200, 56320) of the last chunk (chunk rows
          [2048, 7168)) vs golden chunk_10, per layer (err_layer_L<i>) and final (err_final_hidden, top5_overlap at
          the ladder's sampled rows); every state vs the golden (check_states). Limits: drift-s56320.json (the rung run
          from 0 on the device, as here). Rows [49152, 51200) vs golden chunk_09 rows [3072, 5120): rel recorded as a
          diagnostic (no drift entry covers them).
  last    the golden state at 51200 (the only fixed-size snapshot), then ONE 8192 chunk [51200, 59392) with actual_end
          56320 (5120 real rows, 3072 pad) vs golden chunk_10 (all 5120 rows). Limits: drift-last.json (rung last).
max_seq: 57344 (from0) / 59392 (last) > target.seq 56320, raised in memory before the load (DS41-279), within deploy.seq.
Every layer runs on every chunk (begin_chunk without ``last``: the decoder too, whose windows the last chunk reads).

Metrics (suffix _c8192_<case>): err_layer_L<i>, err_final_hidden, top5_overlap (gated, each with <metric>_limit);
pcc_final_hidden, top1_match, err_chunk09_rows (from0, diagnostic), chunk_seconds_c<k>, model_load_s.

    BRINGUP_CHUNK=8192 python -m models.demos.common.bringup.resident submit --spec $BRINGUP_SPEC --cmd \\
        "BRINGUP_CHUNK=8192 scripts/run_safe_pytest.sh --run-all --no-precompile \\
         models/demos/deepseek_v41_flash_d_p/tests/packed/test_chunk8192_reference.py"
(a model load of its own: the chunk is a load-time geometry; 40 layers, ~4-5 min on the 8x4.)
"""

from __future__ import annotations

import time

import pytest
import torch

from models.demos.common.bringup.core import metrics
from models.demos.common.bringup.reference.golden import Golden
from models.demos.common.bringup.testing import e2e_limits as E
from models.demos.common.bringup.testing.harness import device_timeout, mesh_parametrize, spec
from models.demos.common.bringup.testing.ladder import check_states, rel, sampled_rows

S = spec()
pytestmark = device_timeout(S)
BATCH_SHARED = True  # (shares a held model of the same chunk; BRINGUP_CHUNK=8192 is the load's geometry)
CHUNK = 8192
GOLDEN_RUNG = "s56320"  # its golden: s56320_c5120 (outs every chunk, states final + the 51200 snapshot)
PAD_ID = 0xFFFFFFFF  # the engine's pad past actual_end (K-72)
CASES = {"from0": "s56320", "last": "last"}  # case -> the drift rung whose limits apply


def _raise_max_seq(monkeypatch, need: int) -> None:
    """DS41-279: the model's position tables reach target.seq; an 8192 chunk past 56320 needs more (capped at
    deploy.seq)."""
    deploy = int((S.data.get("deploy") or {}).get("seq") or need)
    assert need <= deploy, (need, deploy)
    if int(S.data["target"]["seq"]) < need:
        monkeypatch.setitem(S.data["target"], "seq", need)


def _chunk_ids(tokens: torch.Tensor, s0: int, end: int) -> torch.Tensor:
    ids = torch.full((CHUNK,), PAD_ID, dtype=torch.long)
    ids[: end - s0] = tokens[s0:end].long()
    return ids


@pytest.mark.parametrize("case", list(CASES))
@mesh_parametrize
def test_chunk8192_reference(mesh_device, case, monkeypatch):
    if mesh_device is None:
        pytest.skip("device test")
    assert int(S.data["target"]["chunk"]) == CHUNK, "run with BRINGUP_CHUNK=8192 (the load's chunk)"
    g = Golden.for_rung(S, GOLDEN_RUNG)
    layers = [i for i in S.layers() if i in g.layers]
    assert layers == list(range(S.num_layers)), f"needs every layer (final norm, top-5): {layers}"
    gchunk, seq = g.chunk, g.seq  # 5120, 56320
    last_c = seq // gchunk - 1  # golden chunk_10 = [51200, 56320)
    g0 = last_c * gchunk
    sfx = f"_c8192_{case}"
    if case == "from0":
        starts = list(range(0, seq, CHUNK))  # 0, 8192, ..., 49152
    else:
        starts = [g0]
    max_seq = starts[-1] + CHUNK
    _raise_max_seq(monkeypatch, max_seq)
    model = S.hooks().device_model(mesh_device, S, layers, lm_head=True)
    metrics.record("model_load_s" + sfx, round(getattr(model, "load_seconds", 0.0), 1))
    assert getattr(model, "chunk", CHUNK) == CHUNK, f"the held model's chunk is {model.chunk}, not {CHUNK}"
    tokens = g.tokens()
    state = model.new_state(max_seq)
    if case == "last":
        for i in layers:
            state.load_prefix(i, g.state(i, at=g0), g0)
        state.load_tokens(tokens[:g0])
    if hasattr(model, "steady_state"):
        model.steady_state(None)

    trail, hidden = {}, None
    for k, s0 in enumerate(starts):
        end = min(s0 + CHUNK, seq)
        ids = _chunk_ids(tokens, s0, end)
        t0 = time.time()
        state.note_tokens(ids, s0, end)  # (the engram tails of the 3 ids before s0)
        model.begin_chunk(ids, s0)  # (no ``last``: every layer, the decoder's windows included)
        h = model.embed(ids)
        final = s0 == starts[-1]
        for i in layers:
            h2 = model.layer(i, h, s0, state, end)
            model.free(h)
            h = h2
            if final:
                trail[i] = model.to_host(h).float()
        if final:
            hidden = model.final_norm(h)
        model.free(h)
        model.sync()
        metrics.record(f"chunk_seconds_c{k:02d}" + sfx, round(time.time() - t0, 3))
        print(f"chunk {k} [{s0}, {s0 + CHUNK}) actual_end {end}: {time.time() - t0:.2f} s")

    s_last = starts[-1]
    a, b = g0 - s_last, seq - s_last  # chunk rows of golden chunk_10's positions
    drift = E.load(g.dir, CASES[case])
    failed = []
    for i in layers:
        want = g.layer(last_c, i)["out"].float()
        e = rel(trail[i][a:b], want)
        metrics.record(f"err_layer_L{i:02d}" + sfx, e)
        metrics.record(f"err_layer_L{i:02d}" + sfx + "_limit", E.err_limit(drift, f"layer_L{i:02d}"))
        E.check(drift, f"layer_L{i:02d}", f"err_layer_L{i:02d}" + sfx, e, failed)
    gm = g.model(last_c)
    final = model.to_host(hidden).float()
    e = rel(final[a:b], gm["final_norm"])
    metrics.record("pcc_final_hidden" + sfx, metrics.pcc(final[a:b], gm["final_norm"].float()))
    metrics.record("err_final_hidden" + sfx, e)
    metrics.record("err_final_hidden" + sfx + "_limit", E.err_limit(drift, "final_hidden"))
    E.check(drift, "final_hidden", "err_final_hidden" + sfx, e, failed)
    rows = sampled_rows(gchunk)
    lg = model.logits(hidden, rows + a).float()
    tt5 = lg.topk(5, dim=-1).indices
    gtop = gm["top32_ids"][rows].long()
    top1 = (tt5[:, 0] == gtop[:, 0]).float().mean().item()
    top5 = (tt5 == gtop[:, :1]).any(-1).float().mean().item()
    metrics.record("top1_match" + sfx, top1)
    metrics.record("top5_overlap" + sfx, top5)
    metrics.record("top5_overlap" + sfx + "_limit", E.overlap_limit(drift, "top5"))
    E.check(drift, "top5", "top5_overlap" + sfx, top5, failed)
    if case == "from0" and a > 0:  # rows [s_last, g0) = golden chunk_09's rows [s_last - (g0 - gchunk), gchunk)
        prev = g.model(last_c - 1)["final_norm"].float()
        lo = s_last - (g0 - gchunk)
        metrics.record("err_chunk09_rows" + sfx, rel(final[:a], prev[lo:]))
        check_states(state, g, layers, seq, drift, failed, S.get("tests.waivers"))
    model.free(hidden)
    print(f"8192 {case}: final hidden rel {e:.5f} top5 {top5:.4f} top1 {top1:.4f}")
    state.free()
    assert not failed, "\n".join(failed)
