# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""tt-d-gen restarts a slot below the GDN state's position (serving_contract.md, "Attention and cache writes"; gates
assemble).

The framework's chunk-start tests (L.starts) skip this model: its first layers are Gated DeltaNet, whose state cannot
be loaded at an arbitrary resident position. This test runs the server's own plan on the device model from position 0,
on the s10240 rung's geometry (chunk 5120, max_seq 10240), every brought-up layer:

  turn 1   a cold 5101-token prompt: one chunk [0, 5101), pad rows [5101, 5120) of the attention KV exactly zero
  turn 2   the follow-up over all 10240 tokens: tt-d-gen restarts at align_down(5101, 64) = 5056 (prefix_indexer.hpp:75,
           prefill_reader.cpp:192) and pulls the last chunk back to max_seq - chunk (prefill_writer.cpp:63-67):
           [5056, 10176) then [5120, 10240). The GDN layers must restore their state at 5056 and at 5120 (snapshots),
           the attention layer must keep rows [0, 5056) unchanged.

Pass: attention K/V rows [0, 5101) after turn 1 and [0, 10240) after turn 2 vs the golden, GDN conv/recurrent after
turn 2 vs the golden final state (PCC >= max(serving.min_kv_pcc, thresholds.state)); attention rows [0, 5056)
byte-identical across turn 2; pad rows zero; a restart at a position with no snapshot (1024) is refused.
Metrics: hybrid_starts_failures, hybrid_starts_pcc_<name>_L<i>.
"""

from __future__ import annotations

import pytest
import torch
from bringup_fw.testing.harness import device_timeout, mesh_parametrize, seq_major, spec

S = spec()
RUNG = "s10240"
KV_BLOCK = 64
FIRST_TURN = 5101
NO_SNAPSHOT_START = 1024


def plans(chunk: int, max_seq: int):
    from bringup_fw.testing import server_rules as R

    first = R.chunk_plan(FIRST_TURN, 0, chunk, max_seq)
    resident = R.follow_up_resident(first[-1][1], max_seq, KV_BLOCK)
    second = R.chunk_plan(max_seq, resident, chunk, max_seq)
    assert resident == 5056 and R.pulled_back(second), (resident, second)  # the server's plan, not the model's
    return first, resident, second


@device_timeout(S)
@mesh_parametrize
def test_hybrid_chunk_starts(mesh_device):
    if mesh_device is None:
        pytest.skip("no mesh (BRINGUP_IMPL=reference): the restart rule is the device model's")
    from bringup_fw.core import metrics
    from bringup_fw.reference.golden import Golden
    from bringup_fw.testing import server_rules as R
    from bringup_fw.testing.chunk_starts import kv_threshold

    g = Golden.for_rung(S, RUNG)
    chunk, max_seq = g.chunk, g.seq
    first, resident, second = plans(chunk, max_seq)
    tokens = g.tokens()
    layers = S.layers()
    fixed = set(S.state_fixed)
    thr = kv_threshold(S)
    try:
        model = S.hooks().device_model(mesh_device, S, layers, lm_head=False)
    except NotImplementedError as e:
        pytest.fail(f"not built: {e}", pytrace=False)
    import inspect

    if (
        "start" not in inspect.signature(model.embed).parameters
        or "end" not in inspect.signature(model.layer).parameters
    ):
        pytest.fail(
            "not built: the device model has no serving entry points embed(tokens, start=) / "
            "layer(i, h, start, state, end=) (serving_contract.md, 'Attention and cache writes')",
            pytrace=False,
        )
    state = model.new_state(max_seq)

    def run(lo: int, hi: int) -> None:
        t = torch.full((chunk,), R.PAD_ID, dtype=torch.int64)
        t[: hi - lo] = tokens[lo:hi]
        h = model.embed(t, start=lo)
        for i in layers:
            h2 = model.layer(i, h, lo, state, end=hi)
            model.free(h)
            h = h2
        model.free(h)
        model.sync()

    fails = []

    def check(n: int, case: str, names_of) -> dict:
        got_all = {}
        for i in layers:
            got = state.to_torch(i, n)
            want = g.state(i)
            for name in names_of(i):
                if name in fixed:
                    a, b = got[name].float().flatten(), want[name].float().flatten()
                else:
                    a, b = seq_major(got[name])[:n], seq_major(want[name])[:n]
                p = metrics.pcc(a, b)
                metrics.record(f"hybrid_starts_pcc_{case}_{name}_L{i:02d}", p)
                ok = p >= thr
                print(f"{'ok  ' if ok else 'FAIL'} {case} layer {i} {name} [0, {n}): pcc {p:.6f} (>= {thr})")
                if not ok:
                    fails.append(f"{case}: layer {i} {name} pcc {p:.5f} < {thr}")
            got_all[i] = got
        return got_all

    def growing(i):
        return [n for n in S.state_names(i) if n not in fixed]

    for lo, hi in first:
        print(f"turn 1 chunk [{lo}, {hi})", flush=True)
        run(lo, hi)
    check(FIRST_TURN, "turn1", growing)
    pad_end = R.ceil_to(FIRST_TURN, R.RECORD_TOKENS)
    held = {}
    for i in layers:
        if not growing(i):
            continue
        got = state.to_torch(i, pad_end)
        held[i] = {}
        for name in growing(i):
            rows = seq_major(got[name])[:pad_end]
            if not torch.all(rows[FIRST_TURN:] == 0):
                fails.append(f"turn1: layer {i} {name} pad rows [{FIRST_TURN}, {pad_end}) not zero before the ack")
            held[i][name] = rows[:resident].clone()

    for lo, hi in second:
        print(f"turn 2 chunk [{lo}, {hi}) (resident {resident})", flush=True)
        try:
            run(lo, hi)
        except Exception as e:  # a model without GDN snapshots may refuse the restart: that is "not built" here
            fails.append(f"turn2: chunk [{lo}, {hi}) rejected: {type(e).__name__}: {e}")
            break
    else:
        got = check(max_seq, "turn2", lambda i: S.state_names(i))
        for i, by_name in held.items():
            for name, rows in by_name.items():
                if not torch.equal(seq_major(got[i][name])[:resident], rows):
                    fails.append(
                        f"turn2: layer {i} {name} rows [0, {resident}) changed; a chunk may only write "
                        "[actual_start, actual_end)"
                    )
        try:
            run(NO_SNAPSHOT_START, NO_SNAPSHOT_START + chunk)
            fails.append(
                f"a restart at {NO_SNAPSHOT_START} (no GDN snapshot there) was accepted: the GDN state would "
                "fold the prefix twice; it must be refused"
            )
        except Exception as e:
            print(f"ok   restart at {NO_SNAPSHOT_START} refused: {type(e).__name__}: {str(e)[:200]}")
    if hasattr(state, "free"):
        state.free()
    metrics.record("hybrid_starts_failures", len(fails))
    for f in fails:
        print(f"FAIL {f}")
    assert not fails, f"{len(fails)} problems, first: {fails[0]}"
