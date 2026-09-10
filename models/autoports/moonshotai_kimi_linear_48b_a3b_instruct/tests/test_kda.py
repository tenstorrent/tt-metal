# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""KDA layer: decode step, prefill (padded / chunked), prefill-then-decode continuity — vs the fp32 torch oracle."""
from __future__ import annotations

import os

import pytest
import torch

import ttnn
from models.autoports.moonshotai_kimi_linear_48b_a3b_instruct.reference.kda_ref import kda_layer_reference
from models.autoports.moonshotai_kimi_linear_48b_a3b_instruct.tests.utils import (
    assert_pcc,
    first_shard,
    gather_conv_carry,
    gather_dim,
    replicated,
)
from models.autoports.moonshotai_kimi_linear_48b_a3b_instruct.tt.kda.decode_step import recurrent_kda_decode_ttnn
from models.autoports.moonshotai_kimi_linear_48b_a3b_instruct.tt.kda.layer import KimiKDA, ceil32
from models.demos.deepseek_v3_d_p.reference.kda.config import KDAConfig
from models.demos.deepseek_v3_d_p.reference.kda.ops import kda_recurrent_reference
from models.demos.deepseek_v3_d_p.tests.kda.utils import random_weights

FULL = KDAConfig(hidden_size=2304, num_heads=32, head_k_dim=128, head_v_dim=128, conv_kernel_size=4, norm_eps=1e-5)


def _weights(request, checkpoint_or_none, layer_idx=0):
    """Real layer weights when a snapshot is available, else deterministic random weights of the full shape."""
    if checkpoint_or_none is not None:
        return checkpoint_or_none.attention_state_dict(layer_idx), f"layer{layer_idx}"
    torch.manual_seed(0)
    return random_weights(FULL), "random"


@pytest.fixture(scope="module")
def maybe_checkpoint():
    s = os.environ.get("KIMI_SNAPSHOT")
    if s and os.path.isfile(os.path.join(s, "model.safetensors.index.json")):
        from models.autoports.moonshotai_kimi_linear_48b_a3b_instruct.reference.weights import KimiCheckpoint

        return KimiCheckpoint(s)
    return None


@pytest.mark.parametrize("B", [1, 8])
def test_decode_step_matches_reference(mesh_device, B):
    torch.manual_seed(1)
    H, K, V = 8, 128, 128
    q, k, v = (torch.randn(B, 1, H, K), torch.randn(B, 1, H, K), torch.randn(B, 1, H, V))
    g = -torch.rand(B, 1, H, K) * 2.0  # log decay <= 0, per channel
    beta = torch.rand(B, 1, H)
    state = torch.randn(B, H, K, V) * 0.1
    ref_o, ref_s = kda_recurrent_reference(q, k, v, g, beta, state)  # [B,1,H,V], [B,H,K,V]
    dev = lambda t, dt=ttnn.bfloat16: ttnn.from_torch(t, dtype=dt, layout=ttnn.TILE_LAYOUT, device=mesh_device)
    o, s = recurrent_kda_decode_ttnn(
        dev(q), dev(k), dev(v), dev(beta, ttnn.float32), dev(g), dev(state, ttnn.float32), device=mesh_device
    )
    assert_pcc(ref_o, ttnn.to_torch(o), 0.999, f"decode step output B={B}")
    assert_pcc(ref_s, ttnn.to_torch(s), 0.999, f"decode step state B={B}")


def _real_hidden(goldens, n):
    """Real post-norm activations entering the layer-0 KDA (stage-02 goldens), tiled to n tokens."""
    h = goldens["runs"]["prefill128"]["hooks"]["layer0.attn"]["in"]
    h = (h if torch.is_tensor(h) else h[0]).float()[0]  # [128, H]
    reps = -(-n // h.shape[0])
    return h.repeat(reps, 1)[:n].unsqueeze(0).bfloat16()


@pytest.mark.parametrize("T,valid", [(32, 32), (64, 61), (128, 128), (2048, 2000)])
def test_prefill_matches_reference(mesh_device, ccl, maybe_checkpoint, goldens, request, T, valid):
    weights, tag = _weights(request, maybe_checkpoint)
    hidden = _real_hidden(goldens, valid)
    ref_out, ref_state = kda_layer_reference(hidden, weights, FULL)
    layer = KimiKDA(mesh_device, FULL, weights, layer_idx=0, ccl=ccl)
    st = layer.allocate_prefill_state()
    hidden_pad = torch.zeros(1, T, FULL.hidden_size, dtype=torch.bfloat16)
    hidden_pad[:, :valid] = hidden
    out, new_state = layer.forward_prefill(replicated(mesh_device, hidden_pad), st, valid_len=valid)
    out_t = first_shard(out).float()[0, 0, :valid]
    assert_pcc(ref_out[0], out_t, 0.995, f"prefill out T={T} valid={valid} ({tag})")
    rec = gather_dim(new_state.recurrent, mesh_device, dim=1)
    assert_pcc(ref_state.recurrent, rec, 0.999, "prefill recurrent state")
    conv = gather_conv_carry(new_state.convolution, layer.tp, FULL.q_dim, FULL.k_dim, FULL.v_dim)
    ref_conv = torch.cat([ref_state.q_convolution, ref_state.k_convolution, ref_state.v_convolution], -1)
    assert_pcc(ref_conv, conv.float(), 0.999, "prefill conv carry")


def test_prefill_then_decode(mesh_device, ccl, maybe_checkpoint, goldens, request):
    weights, tag = _weights(request, maybe_checkpoint)
    P, D = 61, 6
    hidden = _real_hidden(goldens, P + D)
    ref_out, _ = kda_layer_reference(hidden, weights, FULL)
    layer = KimiKDA(mesh_device, FULL, weights, layer_idx=0, ccl=ccl)
    st = layer.allocate_prefill_state()
    T = ceil32(P)
    pad = torch.zeros(1, T, FULL.hidden_size, dtype=torch.bfloat16)
    pad[:, :P] = hidden[:, :P]
    _, st = layer.forward_prefill(replicated(mesh_device, pad), st, valid_len=P)
    ds = layer.allocate_decode_state(batch=1)
    layer.prefill_state_to_decode(st, ds)
    for i in range(D):
        x = hidden[:, P + i : P + i + 1].reshape(1, 1, 1, -1)
        out = layer.forward_decode(replicated(mesh_device, x), ds)
        assert_pcc(ref_out[0, P + i], first_shard(out).float().reshape(-1), 0.995, f"decode token {i} ({tag})")
    ref_state = kda_layer_reference(hidden, weights, FULL)[1]
    rec = gather_dim(ds.recurrent, mesh_device, dim=1)
    assert_pcc(ref_state.recurrent, rec, 0.999, "state after prefill+decode")
