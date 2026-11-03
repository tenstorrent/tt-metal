# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""P.30 (owner 2026-11-03): DECODER_SKIP=exact keeps what the request's last chunk must produce exact.

Rung s10240 (two 5120-token chunks from position 0, the golden's tokens), every layer, twice in one process:

    skip   chunk 0 with last=False: layers 0-19 in full, layer 20 its KV projection only (TtDS41Block.kv_only),
           layers 21-39 skipped (their window rows of [0, 5120) never written); chunk 1 with last=True: every layer,
           each decoder layer's window carry masked (its rows before 5120 were skipped)
    off    both chunks with last=True (every layer, today's path; the switch's ``off`` behaviour)

Checked against the golden, each with its limit from the rung's drift (testing/e2e_limits.py, F128), on the rows the
skip keeps exact (a decoder layer 20 + k is exact from chunk start + (k + 1) x 127 on: rows >= 5120 + 20 x 127):
    skip_err_final_hidden       the final norm's rel L2 over chunk 1's rows [20 x 127, 5120) (drift final_hidden)
    skip_top5_overlap           the device's top-5 holds the golden's top-1, over the ladder's sampled rows there
    skip_err_window_kv_L<i>     every layer's window KV ring at 10240 (the last 128 rows; drift state_window_kv_L<i>)
    skip_err_compress_kv_L20 / skip_err_index_k_L20   layer 20's compressed KV / index keys over [0, 10240): what
                                its KV projection must keep (chunk 0) and the full layer writes (chunk 1)
The off path's same values are recorded as off_skip_<metric> (diagnostic, not gated). Structural: on the skip path the
decoder layers' served window rows of chunk 0 stay zero (they never ran there) and layer 20's served compressed KV of
chunk 0 does not (its KV projection ran).

    DS41_ENGRAM_TABLE=resident DS41_DECODER_SKIP=exact scripts/run_safe_pytest.sh --run-all --no-precompile \\
        models/demos/deepseek_v41_flash_d_p/tests/bringup/test_decoder_skip.py
"""

from __future__ import annotations

import pytest

from models.demos.common.bringup.core import metrics
from models.demos.common.bringup.reference.golden import Golden
from models.demos.common.bringup.testing import e2e_limits as E
from models.demos.common.bringup.testing.harness import device_timeout, mesh_parametrize, spec
from models.demos.common.bringup.testing.ladder import rel, sampled_rows

S = spec()
pytestmark = device_timeout(S)
RUNG = "s10240"


def check(drift: dict, key: str, metric: str, value: float, failed: list[str]) -> None:
    """e2e_limits.check for a ``skip_*`` metric (its limit kind is read from an ``err_`` prefix): an error against
    calib x the drift's rel (err_limit), an overlap against overlap_limit; records <metric> and <metric>_limit."""
    err = "_err_" in metric
    lim = E.err_limit(drift, key) if err else E.overlap_limit(drift, key)
    metrics.record(metric, value)
    metrics.record(f"{metric}_limit", lim)
    if value > lim if err else value < lim:
        failed.append(
            f"FAIL {metric}: {value:.5g} {'>' if err else '<'} limit {lim:.5g} (calib {E.CALIB} x drift {key})"
        )


def run(model, g: Golden, layers: list[int], skip: bool) -> dict:
    """Both chunks of the rung; chunk c < last is marked not last when ``skip``. Returns the host values checked."""
    chunk, n = g.chunk, g.seq // g.chunk
    tokens = g.tokens()
    state = model.new_state(g.seq)
    model.steady_state(None)
    hidden = None
    for c in range(n):
        s0 = c * chunk
        last = c == n - 1
        model.begin_chunk(tokens[s0 : s0 + chunk], s0, last=last or not skip)
        h = model.embed(tokens[s0 : s0 + chunk])
        for i in layers:
            h2 = model.layer(i, h, s0, state)
            model.free(h)
            h = h2
        hd = model.final_norm(h)
        model.free(h)
        if last:
            hidden = hd
        else:
            model.free(hd)
        model.sync()
        print(f"{'skip' if skip else 'off '} chunk {c} [{s0}, {s0 + chunk}) last={last or not skip}")
    out = {"final": model.to_host(hidden).float(), "window": {}, "zero_chunk0": {}}
    w = model.cfg.window
    rows = sampled_rows(chunk)
    out["rows"] = rows
    out["top5"] = model.logits(hidden, rows).float().topk(5, dim=-1).indices
    model.free(hidden)
    for i in layers:
        out["window"][i] = state.window[i].ring_host(g.seq, w).float()
    for i in model.decoder or [i for i in layers if i >= model.cfg.cand_source]:
        out["zero_chunk0"][i] = bool((state.window[i].to_host(chunk).float() == 0).all())
    src = model.cfg.cand_source
    st = state.to_torch(src, g.seq)
    out["compress_kv"], out["index_k"] = st["compress_kv"].float(), st["index_k"].float()
    out["ckv_chunk0_zero"] = bool((state.compress[src].ckv.to_host(chunk).float() == 0).all())
    state.free()
    return out


@mesh_parametrize
def test_decoder_skip(mesh_device, monkeypatch):
    if mesh_device is None:
        pytest.skip("device test")
    monkeypatch.setenv("DS41_DECODER_SKIP", "exact")  # (the gate sets it too; the model reads it at load)
    g = Golden.for_rung(S, RUNG)
    layers = [i for i in S.layers() if i in g.layers]
    assert layers == list(range(S.num_layers)), f"the skip needs every layer: {layers}"
    model = S.hooks().device_model(mesh_device, S, layers, lm_head=True)
    assert model.decoder, "DECODER_SKIP=exact did not take (no decoder layers)"
    src = model.cfg.cand_source
    assert model.decoder[0] == src and model.kv_only == {src}, (model.decoder, model.kv_only)
    metrics.record("model_load_s", round(getattr(model, "load_seconds", 0.0), 1))
    chunk, n = g.chunk, g.seq // g.chunk
    assert n == 2, (g.seq, g.chunk)
    t0 = len(model.decoder) * (model.cfg.window - 1)  # the first chunk row every decoder layer computes exactly
    assert t0 + model.cfg.window <= chunk, (t0, chunk)
    drift = E.load(g.dir, RUNG)
    gm = g.model(n - 1)
    want_final = gm["final_norm"].float()
    gtop = gm["top32_ids"].long()

    res = {"skip": run(model, g, layers, skip=True), "off": run(model, g, layers, skip=False)}
    failed, table = [], {}
    for path, r in res.items():
        pre = "" if path == "skip" else "off_"
        e = rel(r["final"][t0:], want_final[t0:])
        rows = r["rows"]
        sel = rows >= t0
        top5 = (r["top5"][sel] == gtop[rows[sel]][:, :1]).any(-1).float().mean().item()
        top1 = (r["top5"][sel][:, 0] == gtop[rows[sel]][:, 0]).float().mean().item()
        last_hit = int(r["top5"][-1, 0]) == int(gtop[rows[-1], 0])
        vals = {"skip_err_final_hidden": (e, "final_hidden"), "skip_top5_overlap": (top5, "top5")}
        for i in layers:
            want = g.state(i)["window_kv"].float()
            vals[f"skip_err_window_kv_L{i:02d}"] = (rel(r["window"][i], want), f"state_window_kv_L{i:02d}")
        gs = g.state(src)
        for name in ("compress_kv", "index_k"):
            want = gs[name].float()
            got = r[name].reshape(want.shape) if r[name].numel() == want.numel() else r[name]
            m = min(got.shape[0], want.shape[0])  # (the rows of [0, seq): ladder.check_states compares the same)
            got, want = got[:m].reshape(m, -1), want[:m].reshape(m, -1)
            vals[f"skip_err_{name}_L{src:02d}"] = (rel(got, want), f"state_{name}_L{src:02d}")
        table[path] = {k: v for k, (v, _) in vals.items()}
        for metric, (v, key) in vals.items():
            if path == "skip":
                check(drift, key, metric, v, failed)
            else:  # (diagnostic: the off path beside it)
                metrics.record(pre + metric, v)
        metrics.record(f"{pre}skip_top1_match", top1)
        metrics.record(f"{pre}skip_last_token_top1", int(last_hit))
        worst = max((k for k in vals if "window_kv" in k), key=lambda k: vals[k][0] / E.err_limit(drift, vals[k][1]))
        print(
            f"{path}: final hidden (rows [{t0}, {chunk})) rel {e:.5f} limit {E.err_limit(drift, 'final_hidden'):.5f}; "
            f"top5 {top5:.4f} (limit {E.overlap_limit(drift, 'top5'):.4f}) top1 {top1:.4f} over {int(sel.sum())} rows, "
            f"last token top1 {'match' if last_hit else 'MISS'}; worst window KV {worst} {vals[worst][0]:.5f} "
            f"(limit {E.err_limit(drift, vals[worst][1]):.5f}); L{src} compress_kv "
            f"{vals[f'skip_err_compress_kv_L{src:02d}'][0]:.5f} index_k {vals[f'skip_err_index_k_L{src:02d}'][0]:.5f}"
        )
    for i in layers:
        print(
            f"L{i:02d} window KV rel skip {table['skip'][f'skip_err_window_kv_L{i:02d}']:.5f} off "
            f"{table['off'][f'skip_err_window_kv_L{i:02d}']:.5f} limit "
            f"{E.err_limit(drift, f'state_window_kv_L{i:02d}'):.5f}"
        )
    # structural: the skip ran (decoder windows of chunk 0 unwritten; layer 20's compressed KV written), off did not
    unwritten = [i for i, z in res["skip"]["zero_chunk0"].items() if not z]
    if unwritten:
        failed.append(f"skip: decoder layers {unwritten} wrote window rows of chunk 0 (not skipped)")
    if res["skip"]["ckv_chunk0_zero"]:
        failed.append(f"skip: layer {src}'s compressed KV of chunk 0 is all zero (its KV projection did not run)")
    if all(res["off"]["zero_chunk0"].values()):
        failed.append("off: the decoder layers' window rows of chunk 0 are zero (the off path skipped them)")
    assert not failed, "\n".join(failed)
