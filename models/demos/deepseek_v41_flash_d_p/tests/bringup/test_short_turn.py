# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""P.31 (owner 2026-11-03): a short turn's time to first token, and its accuracy, under the switch DECODER_SKIP.

A 512-token suffix appended to a 51200-token cached prefix (rung ``last``: the golden state of [0, 51200) loaded for
every layer, the golden's own tokens [51200, 51712) as the suffix), one chunk, the request's last:

    exact      replay_rows 0: the chunk starts at the prefix's end (its caches are complete: the prefix's last chunk
               ran every layer), every layer over the chunk (begin_chunk(.., last=True))
    replay128  replay_rows 128: the chunk starts 128 tokens before the prefix's end, every layer's window truncated
               there (begin_chunk(.., replay=True): DeepSeek's SWA Bounded Replay), layers 0-19 over the chunk, layer
               20's KV projection over the chunk, then the decoder (layers 20-39) on the last 128 rows only, as the
               decoder tail model's own chunk (tt/model.py TtDS41Model.build_tail; windows truncated to those rows)

Recorded (the A/B runs it with the switch on and off):
    short_turn_ms              wall of the chunk, warm (begin_chunk .. final norm, synced; median of TIMED_RUNS, the
                               first run discarded); the host LM head is not in it
    short_turn_device_ms       one run under the real-time profiler: every program, the slowest chip a call
    short_turn_err_final_hidden   the final norm's rel L2 vs the golden at the last position (the first token's),
                               limit calib x the rung's drift (final_hidden): gated here
    short_turn_top5_overlap    the device's top-5 holds the golden's top-1 at the last position (1 / 0; limit: the
                               rung's top5): gated here
    short_turn_err_tail_hidden, short_turn_top5_tail   the same over the last 128 positions (the rows both values
                               compute; under replay128 the first of them see a truncated window: diagnostic)
    short_turn_top1_last       the golden's top-1 is the device's top-1 at the last position (diagnostic)
    short_turn_start, short_turn_rows, model_load_s, tail_load_s

    DS41_ENGRAM_TABLE=resident DS41_DECODER_SKIP=replay128 scripts/run_safe_pytest.sh --run-all --no-precompile \\
        models/demos/deepseek_v41_flash_d_p/tests/bringup/test_short_turn.py
"""

from __future__ import annotations

import time

import pytest
import torch

from models.demos.common.bringup.core import metrics
from models.demos.common.bringup.reference.golden import Golden
from models.demos.common.bringup.testing import e2e_limits as E
from models.demos.common.bringup.testing import profiler
from models.demos.common.bringup.testing.harness import device_timeout, mesh_parametrize, spec
from models.demos.common.bringup.testing.ladder import rel

S = spec()
pytestmark = device_timeout(S)
RUNG = "last"
SUFFIX = 512  # the new tokens
ROWS = 128  # the rows checked: the last 128 positions (the decoder tail of replay128)
TIMED_RUNS = 5
PAD_ID = 0xFFFFFFFF  # the engine's pad (K-72): past actual_end


def _median(v):
    v = sorted(v)
    return v[len(v) // 2]


@mesh_parametrize
def test_short_turn(mesh_device):
    if mesh_device is None:
        pytest.skip("device test")
    g = Golden.for_rung(S, RUNG)
    layers = [i for i in S.layers() if i in g.layers]
    assert layers == list(range(S.num_layers)), f"a short turn needs every layer: {layers}"
    chunk, n = g.chunk, g.seq // g.chunk
    prefix = (n - 1) * chunk  # the cached prefix: the golden state at the rung's last chunk start
    end = prefix + SUFFIX
    model = S.hooks().device_model(mesh_device, S, layers, lm_head=True)
    metrics.record("model_load_s", round(getattr(model, "load_seconds", 0.0), 1))
    mode = "replay128" if getattr(model, "replay", False) else ("exact" if getattr(model, "decoder", None) else "off")
    if mode == "replay128":
        model.build_tail()  # (load time: the decoder tail model, before any timed run)
    metrics.record("tail_load_s", round(getattr(model, "tail_load_seconds", 0.0), 1))
    start = max(0, (prefix - getattr(model, "replay_rows", 0)) // 32 * 32)
    assert end - start <= chunk, (start, end, chunk)
    metrics.record("short_turn_start", start)
    metrics.record("short_turn_rows", end - start)
    print(
        f"DECODER_SKIP={mode}: chunk [{start}, {end}) on a cached prefix of {prefix} (model load {model.load_seconds:.0f} s)"
    )

    tokens = g.tokens()
    ids = torch.full((chunk,), PAD_ID, dtype=torch.long)
    ids[: end - start] = tokens[start:end].long()
    state = model.new_state(g.seq)
    for i in layers:
        state.load_prefix(i, g.state(i, at=prefix), prefix)
    state.load_tokens(tokens[:prefix])
    model.steady_state(None)

    def run():
        state.note_tokens(ids, start, end)  # (the engram tails of the 3 ids before start)
        model.begin_chunk(ids, start, last=True, replay=True)
        h = model.embed(ids)
        for i in layers:
            h2 = model.layer(i, h, start, state, end)
            model.free(h)
            h = h2
        hidden = model.final_norm(h)
        model.free(h)
        return hidden

    # accuracy: the first run
    hidden = run()
    model.sync()
    first = getattr(hidden, "start", start)  # (replay128: the tail chunk's first position)
    final = model.to_host(hidden).float()
    sel = torch.arange(end - ROWS - first, end - first)
    lg = model.logits(hidden, sel).float()
    model.free(hidden)
    gm = g.model(n - 1)
    want = gm["final_norm"].float()[end - ROWS - prefix : end - prefix]
    gtop = gm["top32_ids"].long()[end - ROWS - prefix : end - prefix]
    got = final[sel]
    top5 = lg.topk(5, dim=-1).indices
    hit = (top5 == gtop[:, :1]).any(-1).float()
    drift = E.load(g.dir, RUNG)
    lim_e, lim_o = E.err_limit(drift, "final_hidden"), E.overlap_limit(drift, "top5")
    vals = {
        "short_turn_err_final_hidden": (rel(got[-1:], want[-1:]), lim_e, "err"),
        "short_turn_top5_overlap": (hit[-1].item(), lim_o, "overlap"),
    }
    failed = []
    for m, (v, lim, kind) in vals.items():
        metrics.record(m, v)
        metrics.record(f"{m}_limit", lim)
        if not (v <= lim if kind == "err" else v >= lim):
            failed.append(
                f"FAIL {m}: {v:.5g} {'>' if kind == 'err' else '<'} limit {lim:.5g} (calib {E.CALIB} x drift)"
            )
    tail_err, tail_top5 = rel(got, want), hit.mean().item()
    metrics.record("short_turn_err_tail_hidden", tail_err)
    metrics.record("short_turn_top5_tail", tail_top5)
    metrics.record("short_turn_top1_last", int(top5[-1, 0] == gtop[-1, 0]))
    print(
        f"accuracy at the last position: final hidden rel {vals['short_turn_err_final_hidden'][0]:.5f} (limit "
        f"{lim_e:.5f}), top5 {int(vals['short_turn_top5_overlap'][0])} (limit {lim_o:.4f}), top1 "
        f"{int(top5[-1, 0] == gtop[-1, 0])}; over the last {ROWS} rows (diagnostic): rel {tail_err:.5f}, top5 "
        f"{tail_top5:.4f}"
    )

    # time to first token (warm): wall, then device
    walls = []
    for _ in range(TIMED_RUNS + 1):
        model.sync()
        t0 = time.time()
        model.free(run())
        model.sync()
        walls.append(round((time.time() - t0) * 1e3, 1))
    walls = walls[1:]
    model.sync()
    profiler.enable(mesh_device, rt=True)
    try:
        profiler.set_layer(None)
        profiler.signpost("run")
        model.free(run())
        profiler.signpost("end")
        profiler.finish_rt()
        dev = sum(profiler.result()["kernel_ns"].values()) / 1e6
    finally:
        profiler.disable()
    metrics.record("short_turn_ms", _median(walls))
    metrics.record("short_turn_device_ms", round(dev, 2))
    print(f"short turn ({mode}): wall {_median(walls)} ms (runs {walls}), device {dev:.2f} ms")
    state.free()
    assert not failed, "\n".join(failed)
