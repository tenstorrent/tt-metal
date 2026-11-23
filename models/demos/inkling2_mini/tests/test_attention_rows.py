# SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
# SPDX-License-Identifier: Apache-2.0
"""Multi-token verify for local attention layers (Inkling2AttentionFast.decode_rows + commit_rows): R tokens in one
pass against R plain decode steps on a copy of the state. The o-projection input rows (row t = token t) and, after
commit_rows, the k / v rings and the conv history are bit-equal. Seeded prefix (bfp8 rings, max_ctx 128K), positions
around a ring wrap and far from it.

    pytest -q models/demos/inkling2_mini/tests/test_attention_rows.py
"""

import pytest
import torch

import models.demos.inkling2_mini.tests.test_attention as ta
import models.demos.inkling2_mini.tests.test_attention_fast as T
import ttnn
from models.demos.inkling2_mini.tests.test_attention_fast import DEVICE_PARAMS
from models.demos.inkling2_mini.tt.attention import ROWS
from models.demos.inkling2_mini.tt.attention_fast import DecodePositionsFast, Inkling2AttentionFast


def _one(attn, x, state, pos):
    """plain decode of one token up to the o-projection input (decode's local path minus w_o)"""
    if attn.window is None:
        proj, mode = attn._window(state, x, x, x)
    else:
        mode = "single"
        proj = attn.w_all(attn.window(state.x_hist, [x]), ttnn.bfloat16, attn.mm_cfg)
    r = ROWS[mode]
    q, mask, sink = attn.prep(proj, pos, {"q": 0, "k": r["k"], "v": r["v"]}, state)
    a = ttnn.transformer.scaled_dot_product_attention_decode(
        q,
        state.k_ring,
        state.v_ring,
        is_causal=False,
        attn_mask=mask,
        attention_sink=sink,
        scale=1.0,
        program_config=attn.local_cfg,
        compute_kernel_config=attn.hifi4,
    )
    return ttnn.to_torch(attn.out(a, proj)).float()[0, 0, 0]


@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
@pytest.mark.parametrize("p", [9000, 1023 + 9 * 1024])
@pytest.mark.parametrize("R", [2, 3])
def test_decode_rows(device, p, R):
    sd = ta.random_state_dict(False, seed=7)
    attn = Inkling2AttentionFast(device, T.host_weights_from_hf(sd, False), False)
    pk, pv = T.seeded_prefix(p, seed=p + 1)
    states = []
    for _ in range(2):
        st = attn.new_state(T.MAX_CTX, ttnn.bfloat8_b)
        st.fill(pk, pv)
        states.append(st)
    ref_st, st = states
    k0 = ttnn.to_torch(st.k_ring).float()
    g = torch.Generator().manual_seed(p + R)
    xs_host = [torch.randn(T.HIDDEN, generator=g) for _ in range(R)]
    xs = [ttnn.to_device(ta._x_host(x), device) for x in xs_host]

    pos = DecodePositionsFast(device, T.MAX_CTX)
    ref = []
    for t in range(R):
        pos.update(p + t)
        pos.prepare()
        ref.append(_one(attn, xs[t], ref_st, pos))

    positions = []
    for t in range(R):
        pt = DecodePositionsFast(device, T.MAX_CTX)
        pt.update(p + t)
        pt.prepare()
        positions.append(pt)
    rows = attn.rows_out(R, st)
    act = ttnn.to_torch(attn.decode_rows(xs, st, positions, rows)).float()[0, 0]
    for t in range(R):
        assert ref[t].abs().max() > 0, t  # a real output row
        assert torch.equal(act[t], ref[t]), t
    assert not act[R:].any(), "rows past R stay zero"
    assert torch.equal(ttnn.to_torch(st.k_ring).float(), k0), "the pass must not write the ring"
    attn.commit_rows(st, rows, positions[0], R)
    assert not torch.equal(ttnn.to_torch(st.k_ring).float(), k0), "commit writes the ring"
    for a, b in ((st.k_ring, ref_st.k_ring), (st.v_ring, ref_st.v_ring), (st.x_hist, ref_st.x_hist)):
        assert torch.equal(ttnn.to_torch(a), ttnn.to_torch(b))
