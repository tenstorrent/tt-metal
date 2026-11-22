# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""PB.7 (owner 2026-11-20): every segment of a GLUED chunk against the CPU reference's golden, within the EXISTING
limits (as test_packed_reference.py; PACKED=on, PACKED_GLUED=on).

The goldens hold one fixed-state snapshot (s56320_c5120: the state at 51200), so every segment continues that point:
distinct slots, each loaded with the golden state at 51200, carrying golden positions [51200, 51200 + n_k). Chunk 1 is
glued: n = 1600, 2048, 777, 1300, 2000 back to back (model rows [51200, cap) each: 1600, 2048, 768, 1280, 1984 at model
rows 0, 1600, 3648, 4416, 5696: segments start and end inside chips and span chips), then segment 5 (n = 1400) does
not fit: its first part [51200, 51648) ends chunk 1 (split at the last kv block that fits) and the rest
[51648, 52600) starts chunk 2, glued with segment 6 (n = 1600, another slot). Segment k vs golden chunk_10 rows
[0, run_k) (the split segment: its two parts' rows together): every layer's output (err_gseg<k>_layer_L<i>), the final
norm (err_gseg<k>_final_hidden) and top-5 at the ladder's sampled rows below run_k (top5_overlap_gseg<k>), each with
``_limit`` from drift-last.json (rung ``last``: the limits the ladder gates; no new limit).

    BRINGUP_CHUNK=8192 DS41_PACKED=on DS41_PACKED_GLUED=on scripts/run_safe_pytest.sh --run-all --no-precompile \\
        models/demos/deepseek_v41_flash_d_p/tests/packed/test_glued_reference.py
(the gate's accuracy load: 40 layers, one load with the oracles, OV-61.)
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
from models.demos.common.bringup.testing.ladder import sampled_rows
from models.demos.deepseek_v41_flash_d_p.tests.packed import packed_common as _pc

_pc = importlib.reload(_pc)  # (the resident runner keeps helper modules of an earlier request: take this file's)
CHUNK = _pc.CHUNK
packed, glued = _pc.packed, _pc.glued
from models.demos.deepseek_v41_flash_d_p.tt.packed import GluedChunk, Segment, split

S = spec()
pytestmark = device_timeout(S)
BATCH_SHARED = True
GOLDEN_RUNG = "s56320"
CHUNK1 = (1600, 2048, 777, 1300, 2000)  # segments 0-4: new tokens, glued in chunk 1
SPLIT_N = 1400  # segment 5: split over chunk 1 / chunk 2
CHUNK2 = (1600,)  # segment 6: glued after segment 5's rest in chunk 2


def _raise_max_seq(monkeypatch, need: int) -> None:
    """As test_packed_reference (DS41-279 / OV-61: the same max_seq in every packed accuracy test)."""
    from models.demos.deepseek_v41_flash_d_p.tt.runners.adapter import table_seq

    if table_seq(int(S.data["target"]["seq"]), CHUNK) < need:
        monkeypatch.setitem(S.data["target"], "seq", need)


class _Err:
    """rel L2 over a segment's rows gathered from one or two chunks (sums of squares, float64)."""

    def __init__(self):
        self.d2, self.w2 = {}, {}

    def add(self, key, got: torch.Tensor, want: torch.Tensor) -> None:
        w = want.double()
        self.d2[key] = self.d2.get(key, 0.0) + float(((got.double().reshape(w.shape) - w) ** 2).sum())
        self.w2[key] = self.w2.get(key, 0.0) + float((w**2).sum())

    def rel(self, key) -> float:
        return (self.d2[key] ** 0.5) / max(self.w2[key] ** 0.5, 1e-300)


@mesh_parametrize
def test_glued_reference(mesh_device, monkeypatch):
    if mesh_device is None:
        pytest.skip("device test")
    from models.demos.deepseek_v41_flash_d_p.tt.packed_device import run_chunk

    assert int(S.data["target"]["chunk"]) == CHUNK, "run with BRINGUP_CHUNK=8192 (the load's chunk)"
    g = Golden.for_rung(S, GOLDEN_RUNG)
    layers = [i for i in S.layers() if i in g.layers]
    assert layers == list(range(S.num_layers)), f"needs every layer (final norm, top-5): {layers}"
    gchunk, seq = g.chunk, g.seq
    last_c = seq // gchunk - 1
    g0 = last_c * gchunk  # 51200
    _raise_max_seq(monkeypatch, g0 + CHUNK)
    model = S.hooks().device_model(mesh_device, S, layers, lm_head=False)
    assert model.chunk == CHUNK, model.chunk
    if model.skip_mode != "off":
        model.use_decoder_skip("off")  # (every layer; FIX.28)
    tokens = g.tokens()
    lens = list(CHUNK1) + [SPLIT_N] + list(CHUNK2)
    states = []
    for k in range(len(lens)):
        st = model.new_state(g0 + CHUNK)
        for i in layers:
            st.load_prefix(i, g.state(i, at=g0), g0)
        st.load_tokens(tokens[:g0])
        states.append(st)
    segs = [Segment(k, g0, g0 + n) for k, n in enumerate(lens)]
    x1, x2 = split(segs[5], CHUNK - sum(CHUNK1))
    assert x1 is not None and x2.start == x1.end, (x1, x2)
    chunks = [
        ([0, 1, 2, 3, 4, 5], [segs[0], segs[1], segs[2], segs[3], segs[4], x1]),
        ([5, 6], [x2, segs[6]]),
    ]
    err = _Err()
    fin_rows = {}
    for ci, (ks, sg) in enumerate(chunks):
        desc = GluedChunk(CHUNK, tuple(sg))
        ids = desc.pack_ids([tokens[s.start : s.end] for s in sg])
        print(
            f"chunk {ci + 1}: segments {ks} positions {[(s.start, s.end) for s in sg]}, model rows "
            f"{[(desc.lane_off(j), desc.rows(j)) for j in range(desc.lanes)]}, lane chunk {desc.as_lanes() is not None}"
        )

        def on_layer(i, h, desc=desc, ks=ks, sg=sg):
            want = g.layer(last_c, i)["out"].float()
            for j, k in enumerate(ks):
                n = desc.rows(j)
                if n <= 0:
                    continue
                r0 = sg[j].start - g0  # (golden chunk_10 row of the part's first row)
                got = _pc.glued_boundary_rows(model, h, desc.lane_off(j), n)
                err.add((k, i), got, want[r0 : r0 + n])
            if i == layers[-1]:
                fin = model.final_norm(h)
                for j, k in enumerate(ks):
                    n = desc.rows(j)
                    if n > 0:
                        fin_rows.setdefault(k, []).append(
                            (sg[j].start - g0, _pc.model_rows(model, fin, desc.lane_off(j), n))
                        )
                model.free(fin)

        t0 = time.time()
        with packed("on"), glued("on"):
            run_chunk(model, desc, ids, [states[k] for k in ks], layers, on_layer=on_layer)
        model.sync()
        print(f"  {time.time() - t0:.1f} s")

    drift = E.load(g.dir, "last")
    gm = g.model(last_c)
    head_w = getattr(model, "head_w", None)
    if head_w is None:
        head_w = model.ck.weight("head.weight", torch.float32)
    failed = []
    srows = sampled_rows(gchunk)
    for k in range(len(lens)):
        sfx = f"gseg{k}"
        for i in layers:
            e = err.rel((k, i))
            name = f"err_{sfx}_layer_L{i:02d}"
            metrics.record(name, e)
            metrics.record(name + "_limit", E.err_limit(drift, f"layer_L{i:02d}"))
            E.check(drift, f"layer_L{i:02d}", name, e, failed)
        parts = sorted(fin_rows[k], key=lambda x: x[0])
        assert parts[0][0] == 0, parts[0][0]
        fin = torch.cat([p for _, p in parts])
        n = fin.shape[0]
        assert all(
            parts[j][0] == sum(p.shape[0] for _, p in parts[:j]) for j in range(len(parts))
        ), "parts not contiguous"
        e = float((fin.double() - gm["final_norm"][:n].double()).norm() / gm["final_norm"][:n].double().norm())
        metrics.record(f"err_{sfx}_final_hidden", e)
        metrics.record(f"err_{sfx}_final_hidden_limit", E.err_limit(drift, "final_hidden"))
        E.check(drift, "final_hidden", f"err_{sfx}_final_hidden", e, failed)
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
        what = " (split over 2 chunks)" if len(parts) > 1 else ""
        print(f"gseg{k} ({n} rows{what}): final rel {e:.5f} top5 {top5:.4f} top1 {top1:.4f} ({len(rows)} sampled rows)")
    del head_w
    for st in states:
        st.free()
    assert not failed, "\n".join(failed)
