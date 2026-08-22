# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Hardware-independent contracts for Laguna prefill MoE dispatch.

The opt-in Stage-1 candidate is tile-sparse: it creates one local-expert mask per
32 token rows for gate/up while the established down path remains unchanged.
``_build_expert_row_dispatch`` pins the stronger token-sparse packing contract so
a later device kernel can replace that stage without changing router weighting or
combine semantics.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

import ttnn
from models.autoports.poolside_laguna_xs_2_1.tt.multichip_decoder import (
    MOE_PREFILL_TILE_SPARSE_ENV,
    TOKEN_DISPATCH_BUCKETS,
    TOKEN_DISPATCH_ENV,
    TOKEN_DISPATCH_MOE_LAYERS,
    MultichipDecoder,
    _build_expert_row_dispatch,
    _parse_binary_env,
    _token_dispatch_eligibility,
)


def _qualified_dispatch_kwargs(**overrides):
    values = {
        "enabled": True,
        "layer_idx": 1,
        "is_moe": True,
        "mesh_devices": 2,
        "seq_len": 8192,
        "sharded": False,
        "pack_gate_up": True,
        "global_experts": 256,
        "local_experts": 128,
        "hidden": 2048,
        "intermediate": 512,
        "top_k": 8,
        "activation_dtype": ttnn.bfloat16,
        "moe_ff13_dtype": ttnn.bfloat4_b,
        "moe_ff2_dtype": ttnn.bfloat4_b,
        "ccl_dtype": ttnn.bfloat16,
        "moe_fidelity": "LoFi",
    }
    values.update(overrides)
    return values


def _expert(x, gate, up, down):
    return (torch.nn.functional.silu(x @ gate) * (x @ up)) @ down


def _dense_routed_reference(x, topk_indices, topk_weights, gate, up, down):
    """Union-sparsity numerical reference: every expert sees every token."""
    tokens, experts = x.shape[0], gate.shape[0]
    dense = torch.zeros(tokens, experts, dtype=x.dtype)
    dense.scatter_(1, topk_indices, topk_weights)
    out = torch.zeros_like(x)
    for expert in range(experts):
        out += _expert(x, gate[expert], up[expert], down[expert]) * dense[:, expert : expert + 1]
    return out


def _packed_routed_reference(x, topk_indices, topk_weights, gate, up, down, devices):
    """Execute only rows named by the stable dispatch plan, then EP-sum."""
    experts = gate.shape[0]
    assert experts % devices == 0
    local_experts = experts // devices
    out = torch.zeros_like(x)
    processed = []
    for device in range(devices):
        expert_start = device * local_experts
        plan = _build_expert_row_dispatch(
            topk_indices,
            topk_weights,
            expert_start=expert_start,
            expert_count=local_experts,
        )
        for local_expert in range(local_experts):
            begin = int(plan["expert_offsets"][local_expert])
            end = int(plan["expert_offsets"][local_expert + 1])
            rows = plan["token_indices"][begin:end]
            weights = plan["weights"][begin:end]
            if not rows.numel():
                continue
            global_expert = expert_start + local_expert
            selected = _expert(x[rows], gate[global_expert], up[global_expert], down[global_expert])
            out.index_add_(0, rows, selected * weights.unsqueeze(1))
            processed.extend((int(row), global_expert) for row in rows)
    return out, processed


def test_row_dispatch_has_stable_selected_row_semantics():
    indices = torch.tensor([[3, 0, 5], [4, 1, 3], [2, 5, 0], [3, 4, 1]])
    weights = torch.tensor([[0.3, 0.2, 0.5], [0.4, 0.1, 0.5], [0.0, 0.7, 0.3], [0.6, 0.4, 0.0]])

    plan = _build_expert_row_dispatch(indices, weights, expert_start=3, expert_count=2)

    # Local expert 0 (global 3) receives tokens 0,1,3; local expert 1
    # (global 4) receives tokens 1,3.  Global expert 2's zero-weight route is
    # deliberately absent as are all non-local experts.
    assert plan["token_indices"].tolist() == [0, 1, 3, 1, 3]
    assert plan["slot_indices"].tolist() == [0, 2, 0, 0, 1]
    assert plan["local_expert_indices"].tolist() == [0, 0, 0, 1, 1]
    torch.testing.assert_close(plan["weights"], torch.tensor([0.3, 0.5, 0.6, 0.4, 0.4]))
    assert plan["expert_counts"].tolist() == [3, 2]
    assert plan["expert_offsets"].tolist() == [0, 3, 5]


@pytest.mark.parametrize("env_name", (TOKEN_DISPATCH_ENV, MOE_PREFILL_TILE_SPARSE_ENV))
def test_moe_experimental_flags_are_strict_and_default_off(monkeypatch, env_name, expect_error):
    monkeypatch.delenv(env_name, raising=False)
    assert not _parse_binary_env(env_name)
    for value, expected in (("0", False), ("1", True)):
        monkeypatch.setenv(env_name, value)
        assert _parse_binary_env(env_name) is expected
    for invalid in ("true", "yes", "2", "", " 1 "):
        monkeypatch.setenv(env_name, invalid)
        with expect_error(ValueError, "exactly '0' or '1'"):
            _parse_binary_env(env_name)


def test_token_dispatch_supported_bucket_and_layer_matrix():
    assert TOKEN_DISPATCH_BUCKETS == {1024, 2048, 4096, 8192}
    assert TOKEN_DISPATCH_MOE_LAYERS == set(range(1, 40))
    for layer_idx in range(40):
        for seq_len in (32, 64, 128, 256, 512, 1024, 2048, 4096, 8192):
            eligible, _ = _token_dispatch_eligibility(
                **_qualified_dispatch_kwargs(
                    layer_idx=layer_idx,
                    is_moe=layer_idx != 0,
                    seq_len=seq_len,
                )
            )
            assert eligible is (layer_idx in TOKEN_DISPATCH_MOE_LAYERS and seq_len in TOKEN_DISPATCH_BUCKETS)


@pytest.mark.parametrize(
    "override,reason",
    [
        ({"enabled": False}, "feature flag"),
        ({"mesh_devices": 1, "local_experts": 256}, "p150x2"),
        ({"mesh_devices": 4, "local_experts": 64}, "p150x2"),
        ({"seq_len": 512}, "bucket"),
        ({"seq_len": 16384}, "bucket"),
        ({"sharded": True}, "decode/sharded"),
        ({"pack_gate_up": False}, "packed gate/up"),
        ({"top_k": 4}, "dimensions"),
        ({"activation_dtype": ttnn.bfloat8_b}, "BF16"),
        ({"moe_ff13_dtype": ttnn.bfloat8_b}, "BFP4/BFP4"),
        ({"moe_ff2_dtype": ttnn.bfloat8_b}, "BFP4/BFP4"),
        ({"ccl_dtype": ttnn.bfloat8_b}, "BF16 CCL"),
        ({"moe_fidelity": "HiFi2"}, "LoFi"),
    ],
)
def test_token_dispatch_guards_fall_back_closed(override, reason):
    eligible, detail = _token_dispatch_eligibility(**_qualified_dispatch_kwargs(**override))
    assert not eligible
    assert reason in detail


def test_mlp_uses_established_moe_for_an_ineligible_bucket(monkeypatch):
    """The runtime guard does not partially enter dispatch before fallback."""
    decoder = object.__new__(MultichipDecoder)
    decoder.cfg = SimpleNamespace(is_moe=True, hidden=2048)
    decoder.MOE_PREFILL_CHUNK = 256
    decoder._token_dispatch_requested = True
    decoder._token_dispatch_fallback_reason = ""
    decoder._token_dispatch_guard = lambda seq_len, sharded: (False, "prefill bucket is not supported")
    decoder._moe_token_dispatch = lambda *_args: pytest.fail("dispatch must not run")
    baseline = object()
    decoder._moe = lambda *_args: baseline
    monkeypatch.setattr(ttnn, "reshape", lambda tensor, _shape: tensor)

    assert decoder._mlp(object(), 256, False) is baseline
    assert decoder._token_dispatch_fallback_reason == "prefill bucket is not supported"


def test_stacked_gate_up_expert_selection_matches_separate_weight_views():
    """Pin the reader's [expert,K,2N] gate/up and [expert,N,K] down contract."""
    torch.manual_seed(2208)
    experts, hidden, intermediate, tokens = 5, 16, 12, 7
    x = torch.randn(tokens, hidden)
    gate = torch.randn(experts, hidden, intermediate)
    up = torch.randn(experts, hidden, intermediate)
    down = torch.randn(experts, intermediate, hidden)
    gate_up = torch.cat((gate, up), dim=-1).unsqueeze(0)
    down_stacked = down.unsqueeze(0)

    for local_expert in range(experts):
        packed = gate_up[0, local_expert]
        got = (torch.nn.functional.silu(x @ packed[:, :intermediate]) * (x @ packed[:, intermediate:])) @ down_stacked[
            0, local_expert
        ]
        expected = _expert(x, gate[local_expert], up[local_expert], down[local_expert])
        torch.testing.assert_close(got, expected, atol=0, rtol=0)


@pytest.mark.parametrize("devices", [1, 2, 4])
def test_token_packed_dispatch_matches_union_reference(devices):
    torch.manual_seed(2026 + devices)
    tokens, hidden, intermediate, experts, top_k = 37, 16, 12, 32, 4
    x = torch.randn(tokens, hidden)
    gate = torch.randn(experts, hidden, intermediate) / hidden**0.5
    up = torch.randn(experts, hidden, intermediate) / hidden**0.5
    down = torch.randn(experts, intermediate, hidden) / intermediate**0.5
    router = torch.randn(tokens, experts)
    _, indices = torch.topk(router, k=top_k, dim=-1, sorted=True)
    weights = torch.rand(tokens, top_k)
    weights /= weights.sum(dim=-1, keepdim=True)

    union = _dense_routed_reference(x, indices, weights, gate, up, down)
    packed, processed = _packed_routed_reference(x, indices, weights, gate, up, down, devices)

    torch.testing.assert_close(packed, union, rtol=2e-5, atol=2e-5)
    expected_pairs = sorted((token, int(expert)) for token, row in enumerate(indices) for expert in row)
    assert sorted(processed) == expected_pairs
    assert len(processed) == tokens * top_k


@pytest.mark.parametrize("devices", [1, 2, 4])
def test_tile_sparse_gate_up_stage_reduces_union_work_without_dropping_routes(devices):
    torch.manual_seed(11 + devices)
    tokens, experts, top_k, group = 256, 256, 8, 32
    local_experts = experts // devices
    indices = torch.topk(torch.randn(tokens, experts), k=top_k, dim=-1).indices

    old_row_expert_work = 0
    tile_row_expert_work = 0
    selected_pairs = 0
    for device in range(devices):
        start = device * local_experts
        local = indices - start
        selected = (local >= 0) & (local < local_experts)
        selected_pairs += int(selected.sum())

        dense = torch.zeros(tokens, local_experts, dtype=torch.bool)
        rows = torch.arange(tokens).unsqueeze(1).expand_as(local)[selected]
        dense[rows, local[selected]] = True
        old_row_expert_work += tokens * int(dense.any(dim=0).sum())
        tile_row_expert_work += group * int(dense.reshape(tokens // group, group, local_experts).any(dim=1).sum())

    # Stage 1 still performs extra gate/up rows within an active tile, but every
    # real token/expert route is covered and substantially less gate/up work
    # survives than the full-chunk expert union. The production down projection
    # deliberately remains on the original full-T union until TTNN can preserve
    # group×expert batches through SwiGLU.
    assert tile_row_expert_work >= selected_pairs
    assert tile_row_expert_work < old_row_expert_work


@pytest.mark.parametrize(
    "indices,weights,error",
    [
        (torch.zeros(2, 2, dtype=torch.int64), torch.zeros(2, 3), "shapes must match"),
        (torch.zeros(2, dtype=torch.int64), torch.zeros(2), "rank >= 2"),
    ],
)
def test_row_dispatch_rejects_invalid_router_tensors(indices, weights, error, expect_error):
    with expect_error(ValueError, error):
        _build_expert_row_dispatch(indices, weights, expert_start=0, expert_count=1)
