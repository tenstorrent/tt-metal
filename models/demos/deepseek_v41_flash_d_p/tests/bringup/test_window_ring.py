# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""P.32 (owner 2026-11-04 / 11-05): WINDOW_KV_KEEP=ring | none against full, on a prompt and its follow-up turn.

Rung s10240's tokens (the golden with every layer's window ring at 10240 and full dumps of both chunks), every layer,
one model (loaded with DECODER_SKIP=replay128, its decoder tail built; ``use_decoder_skip`` runs it as exact too), a
fresh state per run:

    prompt     [0, 5120) not the request's last (last=False: the decoder skipped), [5120, 8021) the last (2901 rows
               >= exact's 2688: the decoder exact from 5120 + 20 x 127 on, so its rows before the follow-up are too)
    follow-up  a second request on the slot, the prompt's continuation up to 10240, starting at align_down(8021, 64) =
               8000 as tt-d-gen starts it (21 rows computed again), minus the pull-back the mode needs where the carry
               is not kept:
                   ring        8000: every layer's carry [7872, 8000) read from the ring (it holds [7840, 8021): the
                               last block's pad rows past 8021 zero); no replay
                   none        exact 8000 - 2688 = 5312, replay128 8000 - 128 = 7872, begin_chunk(.., replay=True):
                               every window truncated there (nothing kept between requests)
                   full        (beside them) exact 8000 (P.30), replay128 7872 with replay=True (P.31)
               the request's last chunk (last=True): exact every layer; replay128 layers 0-19, layer 20's KV
               projection, the decoder tail on the last 128 rows

Checked against the golden with the end-to-end limits (testing/e2e_limits.py: calib x the rung's drift), the full
path's values recorded beside as full_<metric>_<mode> (diagnostic):
    ring_err_final_hidden_<keep>_<mode>    the final norm's rel L2: exact over the turn's rows [8000, 10240);
                                           replay128 at the last position (the first token; P.31's gated metric)
    ring_top5_overlap_<keep>_<mode>        the device's top-5 holds the golden's top-1: exact over the sampled rows
                                           of [8000, 10240), replay128 at the last position
    ring_err_window_kv_L<i>_<keep>_<mode>  every layer's window KV at 10240 (its last 128 rows, ring_host) vs the
                                           golden's ring; under replay128 the decoder layers' (20-39) rows are the
                                           tail's, truncated to it by design (P.31: not gated, owner): recorded as
                                           ring_diag_window_kv_L<i>_<keep>_replay128
Memory: ring_window_bytes_<keep> = the window KV bytes a chip the slot keeps, the larger of the two modes (replay128:
its decoder tail's caches included; under ring / none those are the tail model's working rings, never carried); none:
0, the model's working rings counted as ring_window_working_bytes_none; ring_window_bytes_full_1m = full's at the
deployment context (1M, 40 layers).
Refusal (DECODER_SKIP=off): after a 5120-token prompt, a chunk whose carry is not kept raises in begin_chunk (ring:
start 2048, older than the ring; none: the follow-up at 4992), and one whose carry the ring holds (5056) does not.

    DS41_ENGRAM_TABLE=resident scripts/run_safe_pytest.sh --run-all --no-precompile \\
        models/demos/deepseek_v41_flash_d_p/tests/bringup/test_window_ring.py
"""

from __future__ import annotations

import pytest
import torch

from models.demos.common.bringup.core import metrics
from models.demos.common.bringup.reference.golden import Golden
from models.demos.common.bringup.testing import e2e_limits as E
from models.demos.common.bringup.testing.harness import device_timeout, mesh_parametrize, spec
from models.demos.common.bringup.testing.ladder import rel, sampled_rows

S = spec()
pytestmark = device_timeout(S)
RUNG = "s10240"
PROMPT_END = 8021  # the prompt's end: its last chunk [5120, 8021) holds >= 2688 rows; not 64-aligned (a follow-up
# starts before it)
PAD_ID = 0xFFFFFFFF  # the engine's pad (K-72): past actual_end
DEPLOY_CONTEXT = 1 << 20


def chunk(model, state, tokens, s: int, e: int, last: bool, replay: bool = False):
    """One chunk [s, e) of the slot (every layer): its final norm (Skipped / TailHidden / the chunk's rows)."""
    ids = torch.full((model.chunk,), PAD_ID, dtype=torch.long)
    ids[: e - s] = tokens[s:e].long()
    state.note_tokens(ids, s, e)
    model.begin_chunk(ids, s, last=last, replay=replay)
    h = model.embed(ids)
    for i in model.layers:
        h2 = model.layer(i, h, s, state, e)
        model.free(h)
        h = h2
    hidden = model.final_norm(h)
    model.free(h)
    model.sync()
    return hidden


def capacity(model, g: Golden) -> int:
    """The state's max_seq: a chunk at a start inside the rung writes a whole chunk's table rows (K-69: any multiple of
    the chunk), so the follow-up at 8000 needs 8000 + 5120 rows."""
    return -(-(g.seq + model.chunk) // model.chunk) * model.chunk


def follow_start(model, keep: str, mode: str, f: int) -> tuple[int, bool]:
    """(start, replay) of the follow-up turn at f = align_down(prompt end, 64)."""
    if keep == "ring":
        return f, False
    if keep == "none":
        return f - (model.last_rows if mode == "exact" else model.replay_rows), True
    return (f, False) if mode == "exact" else (f - model.replay_rows, True)


def run(model, g: Golden, keep: str, mode: str) -> dict:
    tokens, seq = g.tokens(), g.seq
    state = model.new_state(capacity(model, g), keep=keep)
    model.steady_state(None)
    for s, e, last in ((0, g.chunk, False), (g.chunk, PROMPT_END, True)):
        model.free(chunk(model, state, tokens, s, e, last))
    f = PROMPT_END // 64 * 64
    start, replay = follow_start(model, keep, mode, f)
    assert seq - start <= model.chunk, (start, seq)
    hidden = chunk(model, state, tokens, start, seq, True, replay)
    first = getattr(hidden, "start", start)  # (replay128: the tail chunk's first position)
    final = model.to_host(hidden).float()
    if mode == "exact":
        pos = torch.arange(f, seq)
        top_pos = f + sampled_rows(seq - f)
    else:
        pos = torch.arange(seq - 1, seq)
        top_pos = pos
    out = {"start": start, "final": final[pos - first], "pos": pos, "top_pos": top_pos}
    out["top5"] = model.logits(hidden, top_pos - first).float().topk(5, dim=-1).indices
    out["final128"] = final[torch.arange(seq - 128, seq) - first]
    model.free(hidden)
    tail = state.tail_state
    out["window"] = {}
    for i in model.layers:
        src = tail if (mode == "replay128" and i in model.decoder and tail is not None) else state
        out["window"][i] = src.window[i].ring_host(seq, model.cfg.window).float()
    out["bytes"] = state.window_bytes()
    state.free()
    print(f"{keep}/{mode}: follow-up [{start}, {seq}) replay={replay}, window bytes a chip {out['bytes']}")
    return out


def check(drift: dict, key: str, metric: str, value: float, failed: list[str] | None) -> None:
    """Records <metric> and <metric>_limit (calib x the drift's rel, or the overlap limit); ``failed`` None: diagnostic."""
    err = "_err_" in metric or "_diag_" in metric
    lim = E.err_limit(drift, key) if err else E.overlap_limit(drift, key)
    metrics.record(metric, value)
    metrics.record(f"{metric}_limit", lim)
    bad = not (value <= lim) if err else not (value >= lim)
    if bad and failed is not None:
        failed.append(
            f"FAIL {metric}: {value:.5g} {'>' if err else '<'} limit {lim:.5g} (calib {E.CALIB} x drift {key})"
        )


def refusal(model, g: Golden, expect_error) -> None:
    """DECODER_SKIP=off: a start whose carry is not kept is refused before any device work; one the ring holds is not."""
    model.use_decoder_skip("off")
    tokens = g.tokens()
    for keep, bad in (("ring", 2048), ("none", 4992)):
        state = model.new_state(capacity(model, g), keep=keep)
        model.free(chunk(model, state, tokens, 0, g.chunk, None))
        ids = torch.full((model.chunk,), PAD_ID, dtype=torch.long)
        with expect_error(ValueError, "which are not kept"):
            model.begin_chunk(ids, bad)
        if keep == "ring":
            model.begin_chunk(ids, g.chunk - 64)  # (the ring holds [4928, 5120): accepted)
        print(f"refusal {keep}: start {bad} refused" + (f", {g.chunk - 64} accepted" if keep == "ring" else ""))
        state.free()


@mesh_parametrize
def test_window_ring(mesh_device, monkeypatch, expect_error):
    if mesh_device is None:
        pytest.skip("device test")
    monkeypatch.setenv("DS41_DECODER_SKIP", "replay128")  # (the model reads it at load; exact via use_decoder_skip)
    g = Golden.for_rung(S, RUNG)
    layers = [i for i in S.layers() if i in g.layers]
    assert layers == list(range(S.num_layers)), f"the follow-up needs every layer: {layers}"
    assert g.seq // g.chunk == 2 and PROMPT_END - g.chunk >= 2688, (g.seq, g.chunk)
    model = S.hooks().device_model(mesh_device, S, layers, lm_head=True)
    model.build_tail()
    assert model.decoder and model.tail is not None, "DECODER_SKIP=replay128 did not take"
    metrics.record("model_load_s", round(getattr(model, "load_seconds", 0.0), 1))
    drift = E.load(g.dir, RUNG)
    gm = g.model(g.seq // g.chunk - 1)
    c0 = g.seq - g.chunk  # the golden chunk's first position
    want_final, gtop = gm["final_norm"].float(), gm["top32_ids"].long()
    want_win = {i: g.state(i)["window_kv"].float() for i in layers}
    failed, nbytes = [], {}
    for mode in ("exact", "replay128"):
        model.use_decoder_skip(mode)
        res = {keep: run(model, g, keep, mode) for keep in ("full", "ring", "none")}
        for keep, r in res.items():
            pre, suf = ("full_", f"_{mode}") if keep == "full" else ("ring_", f"_{keep}_{mode}")
            gate = None if keep == "full" else failed
            e = rel(r["final"], want_final[r["pos"] - c0])
            hit = (r["top5"] == gtop[r["top_pos"] - c0][:, :1]).any(-1).float().mean().item()
            check(drift, "final_hidden", f"{pre}err_final_hidden{suf}", e, gate)
            check(drift, "top5", f"{pre}top5_overlap{suf}", hit, gate)
            metrics.record(f"{pre}diag_final_hidden_last128{suf}", rel(r["final128"], want_final[-128:]))
            worst = (None, 0.0)
            for i in layers:
                key = f"state_window_kv_L{i:02d}"
                v = rel(r["window"][i], want_win[i])
                diag = mode == "replay128" and i in model.decoder
                name = f"{pre}{'diag' if diag else 'err'}_window_kv_L{i:02d}{suf}"
                check(drift, key, name, v, None if diag else gate)
                ratio = v / E.err_limit(drift, key)
                if not diag and ratio > worst[1]:
                    worst = (i, ratio)
            nbytes[keep] = max(nbytes.get(keep, 0), r["bytes"])
            print(
                f"{keep}/{mode}: final hidden rel {e:.5f} (limit {E.err_limit(drift, 'final_hidden'):.5f}) over "
                f"{len(r['pos'])} rows, top5 {hit:.4f} (limit {E.overlap_limit(drift, 'top5'):.4f}); worst gated "
                f"window KV L{worst[0]} at {worst[1]:.3f} of its limit"
            )
        for i in layers:
            print(
                f"{mode} L{i:02d} window KV rel full {rel(res['full']['window'][i], want_win[i]):.5f} ring "
                f"{rel(res['ring']['window'][i], want_win[i]):.5f} none {rel(res['none']['window'][i], want_win[i]):.5f}"
                f" limit {E.err_limit(drift, f'state_window_kv_L{i:02d}'):.5f}"
            )
    for keep, b in nbytes.items():
        metrics.record(f"ring_window_bytes_{keep}", b)
    working = sum(r.nbytes for m in (model, model.tail) for r in m.tables._rings.values())
    metrics.record("ring_window_working_bytes_none", working)
    full_1m = (-(-DEPLOY_CONTEXT // model.chunk)) * model.tables.geo.stripe * 512 * 2 * len(layers)
    metrics.record("ring_window_bytes_full_1m", full_1m)
    print(
        f"window KV a chip a slot: full {nbytes['full']} B at {g.seq} ({full_1m / 2**30:.3f} GiB at 1M), ring "
        f"{nbytes['ring']} B ({nbytes['ring'] / 2**20:.2f} MiB), none {nbytes['none']} B (+ the model's working rings "
        f"{working} B)"
    )
    refusal(model, g, expect_error)
    assert not failed, "\n".join(failed)
