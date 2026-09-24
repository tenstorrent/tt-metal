# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Gated DeltaNet token mixer vs the torch reference at real dims, SP=8 x TP=4, random weights.

Covers: projections -> SP gather -> causal conv (with carry) -> gates -> fused delta rule -> gated
norm -> out_proj -> TP all-reduce, plus both carried states (recurrent, conv) in the golden trace's
layout. The chunked case pushes 2 chunks through the same module with the state carried, which is
the GDN half of chunked prefill. (No minimax pattern exists for GDN; structure follows
test_attention_vs_ref.py / test_attention_chunked_vs_ref.py.)
"""

import os

import pytest
import torch

import ttnn
from models.demos.qwen_3_8_27b.config import QWEN38, PrefillSpec, ttnn_dtype
from models.demos.qwen_3_8_27b.reference import qwen3_8_ref as ref
from models.demos.qwen_3_8_27b.tests.common import assert_pcc, from_sp, to_sp
from models.demos.qwen_3_8_27b.tt.gdn import TtGatedDeltaNet


@pytest.fixture(scope="module", params=["composed", "fused"])
def gdn_pair(mesh_config, request):
    """core: the composed fp32 delta rule (default, tt/gdn_core.py) or the fused ttnn op."""
    torch.manual_seed(0)
    m = ref.init_random_(ref.GatedDeltaNet(QWEN38), seed=11).to(torch.bfloat16).eval()
    spec = PrefillSpec.load()
    old = os.environ.get("QWEN38_GDN_CORE")
    os.environ["QWEN38_GDN_CORE"] = request.param
    try:
        tt = TtGatedDeltaNet(
            mesh_config, QWEN38, m.state_dict(), weight_dtype=ttnn_dtype(spec.weight_dtype_attention), cache=None
        )
    finally:
        if old is None:
            os.environ.pop("QWEN38_GDN_CORE")
        else:
            os.environ["QWEN38_GDN_CORE"] = old
    tt.tag = request.param
    return m, tt


def _x(T, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(1, T, QWEN38.hidden_size, generator=g).to(torch.bfloat16)


@pytest.mark.parametrize("seq", [2048, 10240], ids=["2k", "10k"])
def test_gdn_vs_ref(mesh_config, gdn_pair, seq):
    m, tt = gdn_pair
    x = _x(seq, 1)
    with torch.no_grad():
        want, want_conv, want_rec = m(x)
    state = {}
    out = tt(to_sp(x[None], mesh_config), state)
    got = from_sp(out, mesh_config)[0]
    assert_pcc(f"gdn_{tt.tag}_out_{seq}", got, want.float())
    rec, conv = tt.read_state(state)
    assert_pcc(f"gdn_{tt.tag}_recurrent_state_{seq}", rec, want_rec)
    assert_pcc(f"gdn_{tt.tag}_conv_state_{seq}", conv, want_conv.float())
    # every SP row ran the same recurrence: the last row's copy must equal row 0's
    rec7, _ = tt.read_state(state, sp_row=mesh_config.sp - 1)
    assert torch.equal(rec, rec7)
    for t in state.values():
        ttnn.deallocate(t)


def test_gdn_chunked_vs_ref(mesh_config, gdn_pair):
    """2 chunks through the same module, state carried == the reference one-shot over both."""
    m, tt = gdn_pair
    chunk = 2048
    x = _x(2 * chunk, 2)
    with torch.no_grad():
        want, want_conv, want_rec = m(x)
    state = {}
    outs = [
        from_sp(tt(to_sp(x[None, :, c * chunk : (c + 1) * chunk], mesh_config), state), mesh_config)[0]
        for c in range(2)
    ]
    assert_pcc(f"gdn_{tt.tag}_chunked_out_chunk1", outs[1], want[:, chunk:].float())
    assert_pcc(f"gdn_{tt.tag}_chunked_out_all", torch.cat(outs, 1), want.float())
    rec, conv = tt.read_state(state)
    assert_pcc(f"gdn_{tt.tag}_chunked_recurrent_state", rec, want_rec)
    assert_pcc(f"gdn_{tt.tag}_chunked_conv_state", conv, want_conv.float())
    for t in state.values():
        ttnn.deallocate(t)
