# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Serving contract, part test: the KV cache written at every start tt-d-gen sends (serving_contract.md, "Attention
and cache writes").

Layers 0 and 1 of the device model (hooks.device_model) run the server's chunk plan (server_rules.chunk_plan) for:

  cold_mid_end     a fresh 8017-token prompt: (0, 5120), (5120, 8017); the second chunk ends mid-record
  follow_up_block  a follow-up turn over a 2944-token resident prefix (46 blocks of 64, the shipped kv_block_size):
                   (2944, 8064), (8064, 9000): every chunk starts off the SP period and crosses a 1280-row slab
  follow_up_tile   a 32-aligned start (kv_block_size 32 is legal, backend_runtime.cpp:66-69): (1312, 6432), (6432, 7001)

Pass: rows below actual_start bit-identical to what the state held, rows [start, end) vs the golden kv_latent with the
server's per-channel PCC (kv_dump_compare.tensor_pcc, nope / pe) >= max(server 0.93, spec state 0.97), the pad rows of
the last 32-token record exactly zero.
"""

import pytest

from models.demos.common.bringup.testing.harness import device_timeout, mesh_parametrize, spec
from models.demos.xing40_a4b_d_p.tests.bringup.contract import parts
from models.demos.xing40_a4b_d_p.tests.bringup.contract import server_rules as R

S = spec()
pytestmark = device_timeout(S)

CASES = {
    "cold_mid_end": (0, 8017),
    "follow_up_block": (2944, 9000),
    "follow_up_tile": (1312, 7001),
}


@mesh_parametrize
def test_cache_starts(mesh_device):
    if mesh_device is None:
        pytest.skip("device test")
    g = R.golden()
    tokens = g.tokens()
    thr = R.state_threshold()
    model = parts.build_model(mesh_device, S)
    fails = []
    for name, (prefix, prompt_len) in CASES.items():
        plan = R.chunk_plan(prompt_len, prefix)
        print(f"case {name}: resident {prefix}, chunks {plan}")
        state = model.new_state(R.MAX_SEQ)
        held = parts.load_prefix(state, g, prefix)
        parts.run_chunks(model, state, tokens, plan)
        fails += [f"{name}: {f}" for f in parts.check_kv(state, g, held, prefix, prompt_len, thr)]
        if hasattr(state, "free"):
            state.free()
    assert not fails, "\n".join(fails)
