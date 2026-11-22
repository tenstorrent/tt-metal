# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Model-level determinism (FIX.25, owner 2026-11-12: "determinism is the highest of all priorities"): the loaded
device model, the same prompts run again in one process, every chunk's final hidden and every layer's served state
bit for bit.

Why: an accuracy gate passes a run-to-run difference smaller than its limits. DeepSeek-V4.1's MoE combine summed in
packet-arrival order (DS41-233 / 234 / 235: up to 0.26 between two identical prefills) for weeks under passing gates.

What runs (``check``): two prompts, ``novel`` (the spec's prompt file, the ladder's ids) and ``code`` (C++ source as
a model turn, testing/b2b.py; DS41-234's prompt), interleaved novel, code, novel, code, ... ``runs`` times each, every
sequence ``chunks`` chunks from position 0 on a fresh state, as a server runs requests. The first sequence after the
model load is the cold one (DS41-233: chunk 0 differed cold vs warm); every later sequence of a prompt runs warm,
right after the other prompt (determinism.py's A, B, A). Each later sequence must equal its prompt's first one bit
for bit: every chunk's final hidden (after the final norm when the stack ends at the model's last layer) and, after
its last chunk, every layer's served state (``state.served(layer, length)``, the caches the serving engine hands to
decode; a model without it: ``state.to_torch``). The comparison is testing/determinism.py's (snapshot / diff: bits,
so -0 vs +0 and NaN payloads count).

Coverage and cost (``coverage``; defaults.yaml tests.model_determinism, DET_CHUNKS / DET_RUNS override for a probe):
4 chunks x 3 runs x 2 prompts. 4 chunks reach the chunks DS41-233 / 234 moved (0-3), 3 runs give cold vs warm and
warm vs warm after the other prompt. In a gate it runs first in the process of the gate's first shared-model unit
(core/gate.py ``with_determinism``, core/batch.py): no model load of its own, its cold sequence pays the program
compiles the unit's next run would have paid.

Metrics: determinism_max_abs_diff (== 0 gates, every gate: core/gate.py), _novel / _code, determinism_state_max_abs_
diff, determinism_elements_differ (bits), determinism_chunk_max_abs_diff_<prompt>, determinism_rows_<prompt> (per
chunk), determinism_chunks / _runs, determinism_s (the check's seconds, its cost in the gate).

No exemption for the model (owner): tests.nondeterministic exempts component steps only (testing/component.py).
"""

from __future__ import annotations

import os
import time

import torch

from models.demos.common.bringup.testing import determinism as D

PROMPTS = ("novel", "code")
TEST = "models/demos/common/bringup/tests/test_determinism.py"
CMD = f"scripts/run_safe_pytest.sh --run-all --no-precompile {TEST}"


def coverage(s) -> tuple[int, int]:
    """(chunks, runs) per prompt: DET_CHUNKS / DET_RUNS, else tests.model_determinism; chunks fit target.seq."""
    chunk = int(s.get("target.chunk"))
    n = int(os.environ.get("DET_CHUNKS") or s.get("tests.model_determinism.chunks") or 4)
    runs = int(os.environ.get("DET_RUNS") or s.get("tests.model_determinism.runs") or 3)
    n = max(1, min(n, int(s.get("target.seq")) // chunk))
    assert runs >= 2, "model determinism: at least two sequences a prompt"
    return n, runs


def read_states(state, layers, length: int) -> list[tuple[str, torch.Tensor]]:
    """Host copies of every layer's served state (layer order), as determinism.snapshot."""
    get = state.served if hasattr(state, "served") else state.to_torch
    return [(p[len("out.") :], t) for p, t in D.snapshot({f"L{i:02d}": get(i, length) for i in layers})]


def sequence(model, layers, tokens: torch.Tensor, chunk: int, n: int, final_norm: bool, states: bool = True):
    """One prefill of n chunks from position 0 on a fresh state: (every chunk's final hidden on the host, the states
    after the last chunk). No sync inside a chunk: the hidden is read back after it, as the harness does."""
    from models.demos.common.bringup.testing.harness import begin_chunk
    from models.demos.common.bringup.testing.positions import _free

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
            outs.append(model.to_host(h).detach().cpu().clone())
            model.free(h)
        model.sync()
        st = read_states(state, layers, n * chunk) if states else []
    finally:
        _free(state)
    return outs, st


def _max_abs(x: torch.Tensor, y: torch.Tensor) -> float:
    if not x.is_floating_point():
        return float((x.long() - y.long()).abs().max()) if x.numel() else 0.0
    d = (x.float() - y.float()).abs().nan_to_num(float("inf"))
    return float(d.max()) if d.numel() else 0.0


def compare(first: tuple, got: tuple) -> dict:
    """One sequence against its prompt's first: per chunk max |diff| and differing rows, the first differing chunk
    and state (determinism.diff's text), the worst state difference and the count of elements whose bits differ."""
    outs0, st0 = first
    outs, st = got
    res = {"chunk_diff": [], "chunk_rows": [], "first_chunk": None, "first_state": None, "state_diff": 0.0}
    bits = 0
    for c, (a, b) in enumerate(zip(outs0, outs)):
        ne = D._bits(a) != D._bits(b)
        bits += int(ne.sum())
        res["chunk_diff"].append(_max_abs(a, b))
        res["chunk_rows"].append(int(ne.reshape(a.shape[0], -1).any(1).sum()) if a.dim() else int(ne.any()))
        if res["first_chunk"] is None and bool(ne.any()):
            res["first_chunk"] = (c, D.diff([("final_hidden", a)], [("final_hidden", b)]))
    if [p for p, _ in st0] != [p for p, _ in st]:
        res["first_state"] = ("structure", D.diff(st0, st))
        res["state_diff"] = float("inf")
        bits += 1
    else:
        for (p, a), (_, b) in zip(st0, st):
            ne = D._bits(a) != D._bits(b) if a.shape == b.shape and a.dtype == b.dtype else torch.ones(1, dtype=bool)
            if bool(ne.any()):
                bits += int(ne.sum())
                res["state_diff"] = max(res["state_diff"], _max_abs(a, b) if a.shape == b.shape else float("inf"))
                if res["first_state"] is None:
                    res["first_state"] = (p, D.diff([(p, a)], [(p, b)]))
    res["bits"] = bits
    return res


def check(
    s,
    model,
    layers,
    *,
    label: str = "model",
    record: bool = True,
    tokens: dict | None = None,
    run=None,
    suffix: str = "",
) -> tuple[str | None, float]:
    """Run the prompts (module docstring) on ``model``; record the metrics. (None when every later sequence equals its
    prompt's first bit for bit, else what differs first (prompt, run, chunk, layer state); the worst |diff|).
    ``tokens``: {prompt: ids} instead of the spec's prompts (the selftest's fake model).
    ``run`` (FIX.28, F266): another serving mode's sequence, ``run(model, layers, tokens, chunk, n, final_norm) ->
    (per-chunk host tensors, [(name, host tensor)] of the state that mode serves)`` instead of ``sequence`` (e.g.
    DeepSeek-V4.1's DECODER_SKIP=decode: chunks marked not last, a replay below global_write_start); ``suffix`` is
    appended to every metric name (``determinism_max_abs_diff<suffix>``, ...), so the default mode's stay as they are.
    The same prompts, order, cold / warm runs and bit comparison."""
    seq_fn = sequence if run is None else run
    from models.demos.common.bringup.core import metrics
    from models.demos.common.bringup.testing import b2b

    chunk = int(s.get("target.chunk"))
    n, runs = coverage(s)
    final_norm = layers[-1] == s.num_layers - 1
    if tokens is None:
        tokens = {wl: torch.tensor(b2b.ids(s, wl, n * chunk)) for wl in PROMPTS}
    assert tuple(tokens) == PROMPTS, f"model determinism: prompts {tuple(tokens)}, want {PROMPTS}"
    print(f"DETERMINISM ({label}): {len(layers)} layers, prompts {PROMPTS}, {n} chunks of {chunk}, {runs} runs each")
    t0 = time.time()
    first: dict = {}
    diffs = {wl: [[0.0] * n for _ in range(runs)] for wl in PROMPTS}
    rows = {wl: [[0] * n for _ in range(runs)] for wl in PROMPTS}
    worst = {wl: 0.0 for wl in PROMPTS}
    state_worst, bits, errors = 0.0, 0, []
    for r in range(runs):
        for wl in PROMPTS:
            got = seq_fn(model, layers, tokens[wl], chunk, n, final_norm)
            if r == 0:
                first[wl] = got
                k = len(got[0])  # (a ``run`` may make another count of per-chunk tensors than n)
                diffs[wl] = [[0.0] * k for _ in range(runs)]
                rows[wl] = [[0] * k for _ in range(runs)]
                continue
            res = compare(first[wl], got)
            diffs[wl][r], rows[wl][r] = res["chunk_diff"], res["chunk_rows"]
            worst[wl] = max(worst[wl], max(res["chunk_diff"]), res["state_diff"])
            state_worst = max(state_worst, res["state_diff"])
            bits += res["bits"]
            cold = " (run 0: the cold sequence, the first after the load)" if wl == PROMPTS[0] else ""
            what = f"prompt {wl} run {r} vs run 0{cold}, after prompt {PROMPTS[1 - PROMPTS.index(wl)]}"
            print(
                f"DETERMINISM {what}: max |diff| per chunk {res['chunk_diff']}, rows differing {res['chunk_rows']} "
                f"of {chunk}, states max |diff| {res['state_diff']}"
            )
            if res["bits"]:
                where = []
                if res["first_chunk"]:
                    where.append(f"first differing chunk {res['first_chunk'][0]}: {res['first_chunk'][1]}")
                if res["first_state"]:
                    where.append(f"first differing layer state {res['first_state'][1]}")
                errors.append(f"{what}: " + "; ".join(where))
    secs = time.time() - t0
    total = max(worst.values())
    if record:
        for wl in PROMPTS:
            metrics.record(f"determinism_max_abs_diff_{wl}{suffix}", worst[wl])
            k = len(diffs[wl][0])
            metrics.record(
                f"determinism_chunk_max_abs_diff_{wl}{suffix}",
                [max(diffs[wl][r][c] for r in range(runs)) for c in range(k)],
            )
            metrics.record(
                f"determinism_rows_{wl}{suffix}", [max(rows[wl][r][c] for r in range(runs)) for c in range(k)]
            )
        # bits differing with equal values (-0 / +0, NaN payloads) still fail: the metric is then at least 2^-149
        metrics.record(f"determinism_max_abs_diff{suffix}", total if total or not bits else 2.0**-149)
        metrics.record(f"determinism_state_max_abs_diff{suffix}", state_worst)
        metrics.record(f"determinism_elements_differ{suffix}", bits)
        metrics.record(f"determinism_chunks{suffix}", n)
        metrics.record(f"determinism_runs{suffix}", runs)
        metrics.record(f"determinism_s{suffix}", round(secs, 1))
    print(f"DETERMINISM ({label}): {'FAIL' if errors else 'bit-identical'} in {secs:.1f} s")
    if not errors:
        return None, total
    more = f" (and {len(errors) - 1} more of {2 * (runs - 1)} sequences differ)" if len(errors) > 1 else ""
    return f"identical inputs gave different outputs (max |diff| {total}, {bits} elements): {errors[0]}{more}", total
