# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""CPU check that LTX_AUDIO_REPLICATE rebuilds the unsharded QKV weight from the TP shards.

The replicated islands all-gather the loaded per-device QKV shards and regroup them with
qkv_regroup_columns. Gathering column shards restores the TP-prepared weight as is, so regrouping it
must give exactly what _prepare_torch_state builds for TP=1.
"""

from types import SimpleNamespace

import pytest
import torch

from models.tt_dit.models.transformers.ltx.attention_ltx import LTXAttention
from models.tt_dit.models.transformers.ltx.audio_replicate_ltx import audio_replicate_enabled, qkv_regroup_columns


def _prepared_qkv(state, num_heads, head_dim, tp):
    attn = SimpleNamespace(
        num_heads=num_heads,
        head_dim=head_dim,
        n_local_heads=num_heads // tp,
        parallel_config=SimpleNamespace(tensor_parallel=SimpleNamespace(factor=tp)),
        fuse_gate=False,
        is_self=True,
    )
    state = dict(state)
    LTXAttention._prepare_torch_state(attn, state)
    # ColParallelLinear stores the weight [in, out]; the bias as a row.
    return state["to_qkv.weight"].T, state["to_qkv.bias"].reshape(1, -1)


@pytest.mark.parametrize("tp", [2, 4])
def test_regrouped_gather_matches_tp1(tp):
    num_heads, head_dim, dim_in = 32, 64, 96
    dim = num_heads * head_dim
    gen = torch.Generator().manual_seed(0)
    state = {}
    for name in ("to_q", "to_k", "to_v"):
        state[f"{name}.weight"] = torch.randn(dim, dim_in, generator=gen)
        state[f"{name}.bias"] = torch.randn(dim, generator=gen)

    gathered_w, gathered_b = _prepared_qkv(state, num_heads, head_dim, tp)
    want_w, want_b = _prepared_qkv(state, num_heads, head_dim, 1)

    regroup = qkv_regroup_columns(tp, dim // tp)
    got_w = torch.cat([gathered_w[:, a:b] for ranges in regroup for a, b in ranges], dim=-1)
    got_b = torch.cat([gathered_b[:, a:b] for ranges in regroup for a, b in ranges], dim=-1)
    assert torch.equal(got_w, want_w)
    assert torch.equal(got_b, want_b)


def test_knob_defaults_off(monkeypatch):
    monkeypatch.delenv("LTX_AUDIO_REPLICATE", raising=False)
    assert not audio_replicate_enabled()
    monkeypatch.setenv("LTX_AUDIO_REPLICATE", "1")
    assert audio_replicate_enabled()
