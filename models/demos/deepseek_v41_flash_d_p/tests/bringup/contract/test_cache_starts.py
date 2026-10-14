# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Serving contract, part test: the input layout and every served cache written at the starts tt-d-gen sends
(serving_contract.md "Input", "KV cache", "Attention and cache writes").

The device model (hooks.device_model) runs the server's chunk plans (server_rules.chunk_plan / follow_up_resident) at
chunk 2048 / max_seq 4096, one state per scenario (one slot), for the layers of the gate's block type
(parts.run_layers: at C.<block type>.attention its representative and that layer's state sources; by hand every
spec layer). Layer 0 starts from model.embed(tokens, start) (the server's ids, PAD_ID past actual_end), every other
layer from its golden block input; the golden is the s4096 rung (every position dumped).

  cold_mid_end   cold 3017: (0, 2048), (2048, 3017): a chunk ending mid-record, inside a ratio-2 group
  follow_block   1000, then 3000 over the 960 resident prefix: (960, 3000), a start off the SP period
  follow_tile    kv_block_size 32: 1330, then 3001 over 1312: (1312, 3001), a start that is not a multiple of 64
  pulled_back    1000, then 4000 over 960: (960, 3008), then (2048, 4000) pulled back over rows already written

Pass, for every served cache of every layer run (caches.layer_configs: window_kv; compress_kv / index_k on a KV
source), after every turn: rows of [0, actual_start) bit-identical to what the state held before the turn; rows of
[actual_start, actual_end) vs the golden with the server's per-channel PCC (kv_dump_compare.tensor_pcc) >= max(0.93
server, spec state 0.97); the pad rows of the last 32-token record exactly zero. Pulled back: rows of [960, 2048)
unchanged by the pulled-back chunk and layer 0's window_kv rows of [2048, 3008) rewritten with the same bytes.
"""

import pytest

from models.demos.common.bringup.testing.harness import device_timeout, mesh_parametrize, spec
from models.demos.deepseek_v41_flash_d_p.tests.bringup.contract import caches as C
from models.demos.deepseek_v41_flash_d_p.tests.bringup.contract import parts
from models.demos.deepseek_v41_flash_d_p.tests.bringup.contract import server_rules as R

S = spec()
pytestmark = device_timeout(S)


@pytest.mark.parametrize("scenario", list(R.SCENARIOS))
@mesh_parametrize
def test_cache_starts(mesh_device, scenario):
    if mesh_device is None:
        pytest.skip("device test")
    _, kvc, _ = R.harness()
    gk = C.GoldenKV()
    thr = R.state_threshold()
    bt, layers = parts.run_layers(S)
    turns = R.scenario_plan(scenario)
    print(f"[cache starts] {scenario}: block type {bt}, layers {layers}, SP {R.SP} (W {R.W}), turns {turns}")
    model = parts.build_model(mesh_device, S, layers)
    state = parts.new_state(model)
    fails, held, prev_end = [], None, 0
    for t, (resident, plan) in enumerate(turns):
        start, end = plan[0][0], plan[-1][1]
        assert start == R.align_down(resident, R.TILE)
        before_last = None
        for k, (lo, hi) in enumerate(plan):
            if k == len(plan) - 1 and k > 0:
                before_last = {i: parts.served(state, i, plan[k - 1][1]) for i in layers}
            parts.run_chunk(model, state, gk, layers, lo, hi)
        for i in layers:
            got = parts.served(state, i, end)
            fails += [
                f"turn {t} {plan}: {f}" for f in parts.check_turn(kvc, gk, i, got, held and held[i], start, end, thr)
            ]
            if before_last is not None:  # the pulled-back last chunk (prefill_writer.cpp:63-67)
                lo, prev_hi = plan[-1][0], plan[-2][1]
                for cfg in C.layer_configs(i):
                    a = C.rows_of(cfg, start, lo)
                    old, new = before_last[i][cfg["name"]], got[cfg["name"]]
                    if not parts.bits(new[a[0] : a[1]]).equal(parts.bits(old[a[0] : a[1]])):
                        fails.append(
                            f"{cfg['name']} layer {i}: rows of [{start}, {lo}) changed by the pulled-back chunk "
                            f"({lo}, {end}), which starts after them"
                        )
                if i == 0:
                    old, new = before_last[0][C.CONFIG_ORDER[0]][lo:prev_hi], got[C.CONFIG_ORDER[0]][lo:prev_hi]
                    if not parts.bits(new).equal(parts.bits(old)):
                        n = (parts.bits(new) != parts.bits(old)).any(dim=1).sum().item()
                        fails.append(
                            f"window_kv layer 0: the pulled-back chunk rewrote {n} rows of [{lo}, {prev_hi}) with other "
                            "bytes (layer 0's KV depends on the token alone: a rewrite must give the same bytes)"
                        )
        held = {i: parts.served(state, i, end) for i in layers}
        prev_end = end
    if hasattr(state, "free"):
        state.free()
    print(f"[cache starts] {scenario}: {len(fails)} failures (end {prev_end})")
    assert not fails, "\n".join(fails[:40])
