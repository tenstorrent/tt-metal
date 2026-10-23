# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Serving contract, restarts of the fixed state (serving_contract.md "Attention and cache writes"): one slot, several
requests, every chunk plan from tt-d-gen's rules (testing/server_rules.py), on the s10240 golden (max_seq 10240).

The framework's chunk-start tests (L.starts) skip layers with fixed-size state; this test covers them.
  slot_reuse   a request, then a new request on the same slot from 0: its first chunk resets conv / recurrent.
               Pass: state at 5120 and at 10240 vs the golden, KV vs the golden.
  follow_up    (prefix_reuse true only) turn 1 of 3001 tokens, turn 2 of 10240 tokens on the same slot: the engine
               starts turn 2 at 2944 and pulls its last chunk back to 5120 (prefill_writer.cpp:63). Pass: the final
               conv / recurrent vs the golden (the state is restored to each chunk's start, never advanced twice), the
               rewrite of [5120, 8064) bit-identical, KV vs the golden.
"""

from __future__ import annotations

import torch
from bringup_fw.reference.golden import Golden
from bringup_fw.testing import server_rules as R
from bringup_fw.testing.harness import device_timeout, mesh_parametrize, seq_major, spec

from models.demos.qwen38_27b.tests.bringup.contract import _common as C

S = spec()
RUNG = "s10240"


def _check(s, state, g, n: int, case: str, fails: list) -> None:
    thr = C.kv_threshold(s)
    for i in C.linear_layers(s):
        got, want = state.to_torch(i, n), g.state(i, at=n)
        for name in ("conv", "recurrent"):
            C.pcc_check(f"turns_{case}_{name}_L{i:02d}_at{n}", got[name], want[name], thr, fails)
    for i in C.attn_layers(s):
        got, want = state.to_torch(i, n), g.state(i)
        for name in ("key", "value"):
            C.pcc_check(
                f"turns_{case}_{name}_L{i:02d}_at{n}", seq_major(got[name])[:n], seq_major(want[name])[:n], thr, fails
            )


def _run(model, layers, state, tokens, plan, chunk):
    try:
        for lo, hi in plan:
            C.run_chunk(model, layers, state, tokens, lo, hi, chunk)
    except TypeError as e:
        C.not_built(f"serving entry points embed(tokens, start=) / layer(i, h, start, state, end=): {e}")


@device_timeout(S)
@mesh_parametrize
def test_turns(mesh_device):
    if mesh_device is None:
        return  # CPU proof mode
    s = S
    g = Golden.for_rung(s, RUNG)
    tokens, chunk, max_seq = g.tokens(), g.chunk, g.seq
    layers = s.layers()
    model = C.device_model(s, mesh_device, layers)
    C.require(model, "embed", "layer", "new_state", "free", "sync")
    fails = []

    # slot_reuse: a first request of 5120 tokens, then a new 10240-token request on the same state from 0
    state = model.new_state(max_seq)
    C.require(state, "to_torch", what="device model state")
    _run(model, layers, state, tokens, R.chunk_plan(chunk, 0, chunk, max_seq), chunk)
    plan = R.chunk_plan(max_seq, 0, chunk, max_seq)
    _run(model, layers, state, tokens, plan[:1], chunk)
    _check(s, state, g, chunk, "slot_reuse", fails)
    _run(model, layers, state, tokens, plan[1:], chunk)
    _check(s, state, g, max_seq, "slot_reuse", fails)
    if hasattr(state, "free"):
        state.free()

    if C.answer(s, "prefix_reuse", False) in (True, "true", "True", 1):
        first = 3001
        state = model.new_state(max_seq)
        _run(model, layers, state, tokens, R.chunk_plan(first, 0, chunk, max_seq), chunk)
        res = R.follow_up_resident(first, max_seq)
        plan = R.chunk_plan(max_seq, res, chunk, max_seq)
        assert R.pulled_back(plan), plan  # the server's plan: (2944, 8064) then (5120, 10240)
        print(f"follow_up: resident {res}, chunks {plan}", flush=True)
        _run(model, layers, state, tokens, plan[:1], chunk)
        lo, hi = plan[1][0], plan[0][1]
        a = C.attn_layers(s)[0]
        before = {k: seq_major(v)[lo:hi].clone() for k, v in state.to_torch(a, hi).items()}
        _run(model, layers, state, tokens, plan[1:], chunk)
        after = {k: seq_major(v)[lo:hi] for k, v in state.to_torch(a, hi).items()}
        if not all(torch.equal(before[k], after[k]) for k in before):
            fails.append(f"follow_up: layer {a} rewrote [{lo}, {hi}) with different bytes")
        _check(s, state, g, max_seq, "follow_up", fails)
        if hasattr(state, "free"):
            state.free()
    else:
        print("follow_up: skipped (prefix_reuse false: every turn starts at 0, kv_block_size 0)", flush=True)
    C.report("turns", fails)
