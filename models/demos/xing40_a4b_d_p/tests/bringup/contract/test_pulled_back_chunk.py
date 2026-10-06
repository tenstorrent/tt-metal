# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Serving contract, part test: the pulled-back last chunk (serving_contract.md, "Attention and cache writes").

A follow-up turn (server_rules.SCENARIOS "pulled_back") whose last chunk's padded end would pass max_seq, so the server
moves it back to max_seq - chunk (prefill_writer.cpp:63-67, tt-d-gen #430). At chunk 2048 / max_seq 4096: 4000 tokens
over a 960-token resident prefix -> (960, 3008), then (2048, 4000) [5120 / 56320: 56000 over 2944 -> ...,
(49024, 54144), (51200, 56000)]. The last chunk recomputes [2048, 3008), which the chunk before it wrote and the server
has already shipped once (prefill_reader.cpp:94 queues [actual_start, actual_end) per ack, so the range ships twice).

Layers 0 and 1 of the device model start from the golden prefix up to the next-to-last chunk and run the last two
chunks. Pass: rows below that chunk's start bit-identical; layer-0 rows of the overlap bit-identical before and after
the rewrite (layer-0 KV is a function of the token and its position only, so the recompute is deterministic); the rows
from that start to the prompt end vs the golden kv_latent with the per-channel PCC >= max(server 0.93, spec state
0.97) at both layers.
"""

import pytest
import torch

from models.demos.common.bringup.testing.harness import device_timeout, mesh_parametrize, spec
from models.demos.xing40_a4b_d_p.tests.bringup.contract import parts
from models.demos.xing40_a4b_d_p.tests.bringup.contract import server_rules as R

S = spec()
pytestmark = device_timeout(S)

RESIDENT, PROMPT = R.scenario()["pulled_back"]


@mesh_parametrize
def test_pulled_back_chunk(mesh_device):
    if mesh_device is None:
        pytest.skip("device test")
    plan = R.chunk_plan(PROMPT, RESIDENT)
    before, last = plan[-2], plan[-1]
    # the server's plan, not the model's: the last chunk moved back over rows the one before it wrote
    assert last == (R.MAX_SEQ - R.CHUNK, PROMPT) and last[0] < before[1], plan
    g = R.golden()
    tokens = g.tokens()
    thr = R.state_threshold()
    model = parts.build_model(mesh_device, S)
    state = model.new_state(R.MAX_SEQ)
    held = parts.load_prefix(state, g, before[0])
    parts.run_chunks(model, state, tokens, [before])
    lo, hi = last[0], before[1]
    first = parts.read_kv(state, 0, hi)[lo:hi].clone()
    parts.run_chunks(model, state, tokens, [last])
    again = parts.read_kv(state, 0, hi)[lo:hi]
    fails = []
    if not torch.equal(first, again):
        rows = (first != again).any(dim=1).nonzero().flatten()
        fails.append(
            f"layer 0: the pulled-back chunk rewrote [{lo}, {hi}) with different bytes ({rows.numel()} rows differ, "
            f"first at {lo + int(rows[0])}); the recompute of a shipped range must be deterministic"
        )
    fails += parts.check_kv(state, g, held, before[0], PROMPT, thr)
    assert not fails, "\n".join(fails)
