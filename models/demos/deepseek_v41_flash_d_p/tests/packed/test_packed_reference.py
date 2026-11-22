# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""PB.3 (owner 2026-11-19): every segment of a packed chunk against the CPU reference's golden, within the EXISTING
limits (bringup/PACKED_DESIGN.md s9 item 3). PACKED=on (the gate sets DS41_PACKED=on; the test sets it for its call).

The goldens hold one fixed-state snapshot (s56320_c5120: the state at 51200), so every segment continues that point:
four distinct slots, each loaded with the golden state at 51200 (as test_chunk8192_reference's ``last`` case), lane k
carrying golden positions [51200, 51200 + n_k), n = 2048, 2048, 1600, 777 (full and padded lanes; PREFILL_END=cap: the
last lane computes [51200, 51968) of 777). Segment k vs golden chunk_10 rows [0, run_k): every layer's output
(err_seg<k>_layer_L<i>), the final norm (err_seg<k>_final_hidden) and top-5 at the ladder's sampled rows below run_k
(top5_overlap_seg<k>), each with ``_limit`` from drift-last.json (rung ``last``: the limits the ladder gates; no new
limit). Lanes 0 and 1 hold the same request at the same positions in different slots and lanes: their outputs must be
bit-identical (packed_ref_lane_spread_max_abs_diff, recorded; the oracle gates lane equivalence).

    BRINGUP_CHUNK=8192 DS41_PACKED=on scripts/run_safe_pytest.sh --run-all --no-precompile \\
        models/demos/deepseek_v41_flash_d_p/tests/packed/test_packed_reference.py
(the gate's accuracy load: 40 layers, one load with the oracle and the determinism check, OV-61.)
"""

from __future__ import annotations

import importlib
import time

import pytest
import torch

from models.demos.common.bringup.core import metrics
from models.demos.common.bringup.reference.golden import Golden
from models.demos.common.bringup.testing import e2e_limits as E
from models.demos.common.bringup.testing.harness import device_timeout, mesh_parametrize, spec
from models.demos.common.bringup.testing.ladder import rel, sampled_rows
from models.demos.deepseek_v41_flash_d_p.tests.packed import packed_common as _pc

_pc = importlib.reload(_pc)  # (the resident runner keeps helper modules of an earlier request: take this file's)
CHUNK = _pc.CHUNK
boundary_rows = _pc.boundary_rows
lane_rows = _pc.lane_rows
packed = _pc.packed
from models.demos.deepseek_v41_flash_d_p.tt.packed import ChunkDescriptor, Segment

S = spec()
pytestmark = device_timeout(S)
BATCH_SHARED = True
GOLDEN_RUNG = "s56320"
LENGTHS = (2048, 2048, 1600, 777)  # lane k's new tokens
SLOTS = (0, 1, 2, 3)


def _raise_max_seq(monkeypatch, need: int) -> None:
    """As test_chunk8192_reference (DS41-279 / OV-61: the same max_seq in every packed accuracy test): target.seq is
    raised only when the 8192 tables (runners/adapter.table_seq) do not reach ``need``."""
    from models.demos.deepseek_v41_flash_d_p.tt.runners.adapter import table_seq

    if table_seq(int(S.data["target"]["seq"]), CHUNK) < need:
        monkeypatch.setitem(S.data["target"], "seq", need)


@mesh_parametrize
def test_packed_reference(mesh_device, monkeypatch):
    if mesh_device is None:
        pytest.skip("device test")
    from models.demos.deepseek_v41_flash_d_p.tt.attention import prefill_end
    from models.demos.deepseek_v41_flash_d_p.tt.packed_device import run_chunk

    assert int(S.data["target"]["chunk"]) == CHUNK, "run with BRINGUP_CHUNK=8192 (the load's chunk)"
    g = Golden.for_rung(S, GOLDEN_RUNG)
    layers = [i for i in S.layers() if i in g.layers]
    assert layers == list(range(S.num_layers)), f"needs every layer (final norm, top-5): {layers}"
    gchunk, seq = g.chunk, g.seq
    last_c = seq // gchunk - 1
    g0 = last_c * gchunk  # 51200
    max_seq = g0 + CHUNK
    _raise_max_seq(monkeypatch, max_seq)
    model = S.hooks().device_model(mesh_device, S, layers, lm_head=False)
    assert model.chunk == CHUNK, model.chunk
    if model.skip_mode != "off":
        model.use_decoder_skip("off")  # (every layer; FIX.28)
    tokens = g.tokens()
    states = []
    for k in range(len(LENGTHS)):
        st = model.new_state(max_seq)
        for i in layers:
            st.load_prefix(i, g.state(i, at=g0), g0)
        st.load_tokens(tokens[:g0])
        states.append(st)
    segs = [Segment(SLOTS[k], g0, g0 + n) for k, n in enumerate(LENGTHS)]
    desc = ChunkDescriptor.packed(CHUNK, segs)
    ids = desc.pack_ids([tokens[g0 : g0 + n] for n in LENGTHS])
    runs = [max(0, prefill_end(s.end) - s.start) for s in segs]

    trail = {k: {} for k in range(len(segs))}
    spread = [0.0]
    hidden = {}

    def on_layer(i, h):  # (errors per layer as the layers run: no trail of every layer kept on the host)
        rows = {}
        want = g.layer(last_c, i)["out"].float()
        for k, n in enumerate(runs):
            rows[k] = boundary_rows(model, h, desc, k, n)
            trail[k][i] = rel(rows[k], want[:n])
        d = (rows[0] - rows[1]).abs().max().item()
        spread[0] = max(spread[0], d)
        if i == layers[-1]:
            fin = model.final_norm(h)
            for k, n in enumerate(runs):
                hidden[k] = lane_rows(model, fin, desc, k, n)
            model.free(fin)

    t0 = time.time()
    with packed("on"):
        run_chunk(model, desc, ids, states, layers, on_layer=on_layer)
    model.sync()
    print(f"packed chunk [{g0}, {g0 + CHUNK}) lanes {LENGTHS}: {time.time() - t0:.1f} s")

    drift = E.load(g.dir, "last")
    gm = g.model(last_c)
    head_w = getattr(model, "head_w", None)
    if head_w is None:
        head_w = model.ck.weight("head.weight", torch.float32)
    failed = []
    srows = sampled_rows(gchunk)
    for k, n in enumerate(runs):
        sfx = f"seg{k}"
        for i in layers:
            e = trail[k][i]
            name = f"err_{sfx}_layer_L{i:02d}"
            metrics.record(name, e)
            metrics.record(name + "_limit", E.err_limit(drift, f"layer_L{i:02d}"))
            E.check(drift, f"layer_L{i:02d}", name, e, failed)
        fin = hidden[k]
        e = rel(fin, gm["final_norm"][:n])
        metrics.record(f"err_{sfx}_final_hidden", e)
        metrics.record(f"err_{sfx}_final_hidden_limit", E.err_limit(drift, "final_hidden"))
        E.check(drift, "final_hidden", f"err_{sfx}_final_hidden", e, failed)
        metrics.record(f"pcc_{sfx}_final_hidden", metrics.pcc(fin, gm["final_norm"][:n].float()))
        rows = srows[srows < n]
        lg = torch.nn.functional.linear(fin[rows], head_w).float()
        tt5 = lg.topk(5, dim=-1).indices
        gtop = gm["top32_ids"][rows].long()
        top5 = (tt5 == gtop[:, :1]).any(-1).float().mean().item()
        top1 = (tt5[:, 0] == gtop[:, 0]).float().mean().item()
        metrics.record(f"top5_overlap_{sfx}", top5)
        metrics.record(f"top5_overlap_{sfx}_limit", E.overlap_limit(drift, "top5"))
        metrics.record(f"top1_match_{sfx}", top1)
        E.check(drift, "top5", f"top5_overlap_{sfx}", top5, failed)
        print(f"seg{k} ({n} rows): final rel {e:.5f} top5 {top5:.4f} top1 {top1:.4f} ({len(rows)} sampled rows)")
    metrics.record("packed_ref_lane_spread_max_abs_diff", spread[0])
    print(f"lane 0 vs lane 1 (same request, other slot and lane): max abs diff {spread[0]}")
    del head_w
    for st in states:
        st.free()
    assert not failed, "\n".join(failed)
