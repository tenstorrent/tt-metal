# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""P.33 (owner 2026-11-05, Kartik / Nikhil's updated proposal): prefill never runs the decoder; it stops at an even
cut 128 tokens short of the prompt's end, and decode runs the last 128 tokens through all 40 layers. A test only (no
model change): what prefill hands decode at the cut is checked against the golden.

Rung s10240's tokens (end = 10240), every layer loaded, DECODER_SKIP=exact. The prompt is prefilled up to
cut = (end - 128) rounded down to even (10112), in the model's chunks from 0, every chunk begin_chunk(.., last=False):
[0, 5120) and the partial chunk [5120, 10112) (its end passed to every layer, the ids past it the engine's pad).
Layers 0-19 run in full, layer 20 only its KV projection (TtDS41Block.kv_only), layers 21-39 are skipped.

Checked at the cut against the golden, each with its limit from the rung's drift (testing/e2e_limits.py: calib x the
official reference's drift, F128):
    cut_err_window_kv_L<i>     i in 0-19: the last 128 window rows before the cut, [cut - 128, cut), as the
                               reference's ring at ``cut`` (ring_host(cut)); limit state_window_kv_L<i> (the drift of
                               the same layer's 128-row ring at the rung's end)
    cut_err_compress_kv_L20 / cut_err_index_k_L20   layer 20's compressed KV / index keys over [0, cut) (ratio 1);
                               limit state_compress_kv_L20 / state_index_k_L20
The golden's window rows before the cut: the golden dumps the kv step of chunk 1 only for the step-dump layers
(0, 1, 2, 3, 14, 20, 21, 24); for the other encoder layers the rows are the reference's own attn_collapse
-> attn_norm -> kv (numerics native, per row: no attention across positions) on the golden's input of that layer (the
previous layer's dumped output), the window weights alone. Cross-check on the dumped non-engram layers (2, 3): the
recomputed rows against the dump, recorded as cut_diag_golden_recompute_L<i>; above one bf16 rounding (the drift's
rel_res) the derivation is wrong and the test fails.
Structural: the decoder layers (20-39) wrote no window row of [0, cut) (skipped / kv_only); layer 20's compressed KV
of the last chunk's rows before the cut is not zero (its KV projection ran up to the cut). Diagnostic: the number of
layer 20 compressed-KV rows past the cut that are not zero (cut_diag_ckv_rows_past_cut: the pad rows of the partial
chunk; decode writes those positions).

    DS41_ENGRAM_TABLE=resident DS41_DECODER_SKIP=exact scripts/run_safe_pytest.sh --run-all --no-precompile \\
        models/demos/deepseek_v41_flash_d_p/tests/bringup/test_decoder_cut.py
"""

from __future__ import annotations

import pytest
import torch

from models.demos.common.bringup.core import metrics
from models.demos.common.bringup.reference.golden import Golden, hf_path
from models.demos.common.bringup.reference.interface import Ctx
from models.demos.common.bringup.testing import e2e_limits as E
from models.demos.common.bringup.testing.harness import device_timeout, mesh_parametrize, spec
from models.demos.common.bringup.testing.ladder import rel

S = spec()
pytestmark = device_timeout(S)
RUNG = "s10240"
DECODE_ROWS = 128  # the tokens decode runs through all 40 layers (the owner's proposal)
PAD_ID = 0xFFFFFFFF  # the engine's pad (K-72): past the chunk's end
WINDOW_WEIGHTS = ("attn_norm.weight", "attn.wkv.weight", "attn.kv_norm.weight")  # (attn_collapse reads the input's pre)


def cut_of(end: int) -> int:
    return (end - DECODE_ROWS) // 2 * 2


def check(drift: dict, key: str, metric: str, value: float, failed: list[str]) -> None:
    """Records <metric> and <metric>_limit (calib x the drift's rel, one rounding at least: e2e_limits.err_limit)."""
    lim = E.err_limit(drift, key)
    metrics.record(metric, value)
    metrics.record(f"{metric}_limit", lim)
    if not value <= lim:
        failed.append(f"FAIL {metric}: {value:.5g} > limit {lim:.5g} (calib {E.CALIB} x drift {key})")


def golden_windows(g: Golden, layers: list[int], cut: int, drift: dict, failed: list[str]) -> dict:
    """{layer: the golden's window rows [cut - 128, cut) as the reference's ring at ``cut`` [128, 512]} (module doc)."""
    from models.demos.deepseek_v41_flash_d_p.reference.ds41_ref import DS41Reference, LayerWeights

    w = DECODE_ROWS
    c = (cut - w) // g.chunk
    assert c == (cut - 1) // g.chunk and c in g.dumped_chunks, f"rows [{cut - w}, {cut}) not in one dumped chunk"
    a, b = cut - w - c * g.chunk, cut - c * g.chunk
    ref = DS41Reference(hf_path(S), layers=[], numerics="native", lm_head=False)
    cfg = ref.cfg

    def recompute(i: int) -> torch.Tensor:
        assert i > 0 and not cfg.is_engram(i), f"layer {i}: no kv dump and not recomputable (engram / first layer)"
        ref.w[i] = LayerWeights({n: ref.ck.weight(f"layers.{i}.{n}", torch.float32) for n in WINDOW_WEIGHTS})
        x = g.layer(c, i - 1)["out"][a:b].float()
        ctx = Ctx(i, cut - w, w, None)
        y = ref.component(i, "attn_norm")(ctx, ref.component(i, "attn_collapse")(ctx, x))
        kv = ref.component(i, "kv")(ctx, y)
        del ref.w[i]
        return kv.float()

    pos = torch.arange(cut - w, cut)
    out = {}
    for i in layers:
        dump = g.layer(c, i)
        rows = dump["kv"][a:b].float().clone() if "kv" in dump else recompute(i)
        if "kv" in dump and i > 0 and not cfg.is_engram(i):
            d = rel(recompute(i), rows)
            res = drift[f"state_window_kv_L{i:02d}"]["rel_res"]
            metrics.record(f"cut_diag_golden_recompute_L{i:02d}", d)
            print(f"golden window rows L{i:02d}: recomputed vs dumped rel {d:.3g} (one rounding {res:.3g})")
            if not d <= res:
                failed.append(
                    f"golden derivation L{i:02d}: recomputed window rows rel {d:.3g} > one rounding {res:.3g}"
                )
        ring = torch.zeros(w, rows.shape[-1])
        ring[pos % w] = rows
        out[i] = ring
    g._layer.cache_clear()
    return out


@mesh_parametrize
def test_decoder_cut(mesh_device, monkeypatch):
    if mesh_device is None:
        pytest.skip("device test")
    monkeypatch.setenv("DS41_DECODER_SKIP", "exact")  # (the gate sets it too; the model reads it at load)
    g = Golden.for_rung(S, RUNG)
    layers = [i for i in S.layers() if i in g.layers]
    assert layers == list(range(S.num_layers)), f"the cut needs every layer: {layers}"
    end, chunk = g.seq, g.chunk
    cut = cut_of(end)
    assert cut % chunk, f"cut {cut} on a chunk boundary: the test is the partial non-last chunk"
    drift = E.load(g.dir, RUNG)
    failed: list[str] = []

    model = S.hooks().device_model(mesh_device, S, layers, lm_head=False)
    assert model.decoder and model.skip_mode == "exact", "DECODER_SKIP=exact did not take"
    src = model.cfg.cand_source
    assert model.decoder[0] == src and model.kv_only == {src}, (model.decoder, model.kv_only)
    encoder = [i for i in layers if i < src]
    metrics.record("model_load_s", round(getattr(model, "load_seconds", 0.0), 1))
    metrics.record("cut_position", cut)

    want_win = golden_windows(g, encoder, cut, drift, failed)
    gs = g.state(src)
    assert model.cfg.ratio(src) == 1, model.cfg.ratio(src)

    # prefill [0, cut): every chunk not the request's last
    tokens = g.tokens()
    state = model.new_state(-(-cut // chunk) * chunk)
    model.steady_state(None)
    for s in range(0, cut, chunk):
        e = min(s + chunk, cut)
        ids = torch.full((chunk,), PAD_ID, dtype=torch.long)
        ids[: e - s] = tokens[s:e].long()
        state.note_tokens(ids, s, e)
        model.begin_chunk(ids, s, last=False)
        h = model.embed(ids)
        for i in layers:
            h2 = model.layer(i, h, s, state, e)
            model.free(h)
            h = h2
        model.free(h)
        model.sync()
        print(f"chunk [{s}, {e}) last=False")

    # the encoder windows at the cut
    worst = (None, 0.0)
    for i in encoder:
        key = f"state_window_kv_L{i:02d}"
        v = rel(state.window[i].ring_host(cut, model.cfg.window).float(), want_win[i])
        check(drift, key, f"cut_err_window_kv_L{i:02d}", v, failed)
        print(f"L{i:02d} window KV [{cut - DECODE_ROWS}, {cut}) rel {v:.5f} limit {E.err_limit(drift, key):.5f}")
        if v / E.err_limit(drift, key) > worst[1]:
            worst = (i, v / E.err_limit(drift, key))
    # layer 20's compressed KV and index keys over [0, cut)
    cc = state.compress[src]  # (as DS41DeviceState.to_torch: its top-k / candidates never ran on a skipped chunk)
    for name, t in (("compress_kv", cc.ckv), ("index_k", cc.index_k)):
        want = gs[name].float()[:cut]
        got = t.to_host(cut)[: cut // cc.ratio].float()
        m = min(got.shape[0], want.shape[0])
        assert m == cut, (name, got.shape, want.shape, cut)
        v = rel(got[:m].reshape(m, -1), want[:m].reshape(m, -1))
        key = f"state_{name}_L{src:02d}"
        check(drift, key, f"cut_err_{name}_L{src:02d}", v, failed)
        print(f"L{src} {name} [0, {cut}) rel {v:.5f} limit {E.err_limit(drift, key):.5f}")
    print(f"worst encoder window KV: L{worst[0]} at {worst[1]:.3f} of its limit")

    # structural: the decoder never wrote a window row; layer 20's KV projection ran up to the cut
    last0 = (cut - 1) // chunk * chunk
    written = [i for i in model.decoder if not bool((state.window[i].to_host(cut).float() == 0).all())]
    if written:
        failed.append(f"decoder layers {written} wrote window rows of [0, {cut}) (the decoder must not run)")
    ckv = cc.ckv.to_host(-(-cut // chunk) * chunk).float()
    if bool((ckv[last0:cut] == 0).all()):
        failed.append(f"layer {src}'s compressed KV of [{last0}, {cut}) is all zero (its KV projection did not run)")
    past = int((ckv[cut:].abs().sum(-1) > 0).sum())
    metrics.record("cut_diag_ckv_rows_past_cut", past)
    print(f"layer {src} compressed KV: {past} non-zero rows past the cut (of {ckv.shape[0] - cut})")
    state.free()
    assert not failed, "\n".join(failed)
