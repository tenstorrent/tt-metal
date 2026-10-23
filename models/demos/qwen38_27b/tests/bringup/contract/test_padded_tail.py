# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Serving contract, pad rows (serving_contract.md "Input", "Attention and cache writes"): a chunk that ends early.

One chunk of the contract rung (5120) carries 5120 - tests.contract_tail_pad real tokens, then PAD_ID (0xFFFFFFFF) up
to the chunk, as the engine sends it (prefill_writer.cpp:93). Pass:
  - linear layers: conv and recurrent equal the golden snapshot at actual_end (pad rows never enter the fold);
  - full-attention layers: KV rows [0, actual_end) vs the golden with PCC >= max(serving.min_kv_pcc, thresholds.state),
    rows [actual_end, ceil32(actual_end)) exactly zero (the record ships whole).
"""

from __future__ import annotations

from bringup_fw.reference.golden import Golden
from bringup_fw.testing.contract import contract_rung
from bringup_fw.testing.harness import device_timeout, mesh_parametrize, seq_major, spec

from models.demos.qwen38_27b.tests.bringup.contract import _common as C

S = spec()


@device_timeout(S)
@mesh_parametrize
def test_padded_tail(mesh_device):
    if mesh_device is None:
        return  # CPU proof mode
    import torch

    s = S
    rung = s.rung(contract_rung(s))
    g = Golden.for_rung(s, rung["name"])
    chunk = rung["chunk"]
    end = chunk - int(s.get("tests.contract_tail_pad"))
    pad_end = -(-end // 32) * 32
    layers = s.layers()
    model = C.device_model(s, mesh_device, layers)
    C.require(model, "embed", "layer", "new_state", "free", "sync")
    state = model.new_state(rung["seq"])
    C.require(state, "to_torch", what="device model state")
    try:
        C.run_chunk(model, layers, state, g.tokens(), 0, end, chunk)
    except TypeError as e:  # no start= / end= keywords yet
        C.not_built(f"serving entry points embed(tokens, start=) / layer(i, h, start, state, end=): {e}")
    thr, fails = C.kv_threshold(s), []
    for i in C.linear_layers(s):
        got = state.to_torch(i, end)
        want = g.state(i, at=end)
        for name in ("conv", "recurrent"):
            C.pcc_check(f"tail_{name}_L{i:02d}", got[name], want[name], thr, fails)
    for i in C.attn_layers(s):
        got = state.to_torch(i, pad_end)
        want = g.state(i)
        for name in ("key", "value"):
            rows = seq_major(got[name])[:pad_end]
            C.pcc_check(f"tail_{name}_L{i:02d}", rows[:end], seq_major(want[name])[:end], thr, fails)
            pad = rows[end:pad_end]
            if pad.numel() and not torch.all(pad == 0):
                fails.append(
                    f"layer {i} {name}: pad rows [{end}, {pad_end}) not zero (max |x| " f"{pad.abs().max().item():.4g})"
                )
    if hasattr(state, "free"):
        state.free()
    C.report("padded_tail", fails)
