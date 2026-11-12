# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Determinism (FIX.23, owner 2026-11-12): identical inputs give identical outputs, bit for bit.

One loaded model (every switch at its shipped default; P.50's gate sets DS41_COMBINE_SPLIT=on), both prompts (novel: the spec's prompt file; code: tt-metal
C++ sources, testing/b2b.py), DET_CHUNKS chunks each (default 4) from position 0, DET_RUNS sequences of each prompt
(default 3), the prompts interleaved (novel, code, novel, code, ...) and every sequence on a fresh state, as a server
runs requests. The first sequence after load is the cold one (DS41-233). Every chunk's final hidden (after the final
norm, or the last layer's output) of every later sequence must equal the prompt's first sequence's bit for bit:
cold vs warm and warm vs warm, after the other prompt ran in between. Metric determinism_max_abs_diff (== 0 gates;
also _novel / _code and per-chunk lists).

DS41-233 / DS41-234 / DS41-235: the MoE combine sums the routed terms in packet-arrival order; with sums above 32 the
single snapped fp32 sum depended on that order (fixed by COMBINE_SPLIT=on).
Ported from FIX.23 (exp/fix23_snapshot 4e90f611e86) by P.50; the A/B: DS41_COMBINE_SPLIT=off vs on.

    PYTHONPATH=$PWD DS41_COMBINE_SPLIT=on scripts/run_safe_pytest.sh --run-all --no-precompile \\
        models/demos/deepseek_v41_flash_d_p/tests/test_determinism.py
"""

from __future__ import annotations

import os

import torch
from loguru import logger

from models.demos.common.bringup.core import metrics
from models.demos.common.bringup.testing import b2b, batch
from models.demos.common.bringup.testing.harness import begin_chunk, mesh_parametrize, spec
from models.demos.common.bringup.testing.positions import _free
from models.demos.deepseek_v41_flash_d_p.tt.settings import settings

S = spec()
PROMPTS = ("novel", "code")


def _sequence(model, layers, tokens: torch.Tensor, chunk: int, n: int, final_norm: bool) -> list[torch.Tensor]:
    """One prefill of n chunks from position 0 on a fresh state; every chunk's final hidden read back after the
    chunk (no sync inside a chunk)."""
    state = model.new_state(n * chunk)
    outs = []
    try:
        for c in range(n):
            s0 = c * chunk
            t = tokens[s0 : s0 + chunk]
            begin_chunk(model, t, s0)
            h = model.embed(t)
            for i in layers:
                h2 = model.layer(i, h, s0, state)
                model.free(h)
                h = h2
            if final_norm:
                h2 = model.final_norm(h)
                model.free(h)
                h = h2
            outs.append(model.to_host(h))
            model.free(h)
        model.sync()
    finally:
        _free(state)
    return outs


@mesh_parametrize(timeout_s=2 * 3600)
def test_determinism(mesh_device):
    chunk = int(S.get("target.chunk"))
    n = int(os.environ.get("DET_CHUNKS") or 4)
    runs = int(os.environ.get("DET_RUNS") or 3)
    assert runs >= 2, "DET_RUNS: at least two sequences a prompt"
    layers = S.layers()
    final_norm = layers[-1] == S.num_layers - 1
    tokens = {wl: torch.tensor(b2b.ids(S, wl, n * chunk)) for wl in PROMPTS}
    model = batch.device_model(S, mesh_device, layers, lm_head=False)
    logger.info(
        f"DETERMINISM: {len(layers)} layers, prompts {PROMPTS}, {n} chunks of {chunk}, {runs} runs each, "
        f"COMBINE_SPLIT={settings.get('COMBINE_SPLIT')}"
    )

    first: dict = {}
    diffs = {wl: [[0.0] * n for _ in range(runs)] for wl in PROMPTS}
    rows = {wl: [[0] * n for _ in range(runs)] for wl in PROMPTS}
    for r in range(runs):
        for wl in PROMPTS:
            outs = _sequence(model, layers, tokens[wl], chunk, n, final_norm)
            if r == 0:
                first[wl] = outs
                continue
            for c in range(n):
                d = (outs[c].float() - first[wl][c].float()).abs()
                diffs[wl][r][c] = float(d.max())
                rows[wl][r][c] = int((d.reshape(d.shape[0], -1).amax(1) > 0).sum())
            logger.info(
                f"DETERMINISM {wl} run {r} vs run 0{' (cold)' if wl == PROMPTS[0] else ''}: max |diff| per chunk "
                f"{diffs[wl][r]}, rows differing {rows[wl][r]} of {chunk}"
            )
    worst = 0.0
    for wl in PROMPTS:
        m = max(max(x) for x in diffs[wl])
        worst = max(worst, m)
        metrics.record(f"determinism_max_abs_diff_{wl}", m)
        metrics.record(
            f"determinism_chunk_max_abs_diff_{wl}", [max(diffs[wl][r][c] for r in range(runs)) for c in range(n)]
        )
        metrics.record(f"determinism_rows_{wl}", [max(rows[wl][r][c] for r in range(runs)) for c in range(n)])
    metrics.record("determinism_max_abs_diff", worst)
    # (P.50's A/B, owner 2026-11-12: the same metric named by the side, COMBINE_SPLIT off / on)
    metrics.record(f"determinism_max_abs_diff_{settings.get('COMBINE_SPLIT')}", worst)
    metrics.record("determinism_chunks", n)
    metrics.record("determinism_runs", runs)
    assert worst == 0.0, (
        f"identical inputs gave different outputs: max |diff| {worst} "
        f"(novel per run / chunk {diffs['novel']}, code {diffs['code']})"
    )
