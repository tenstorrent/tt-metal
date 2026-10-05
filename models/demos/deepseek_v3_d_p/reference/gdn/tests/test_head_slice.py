# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU tests: a whole-K-group GDN head slice is one TP rank's shard and its reference is that rank's oracle."""

from dataclasses import replace

import pytest
import torch

from models.demos.deepseek_v3_d_p.reference.gdn.config import GDNConfig
from models.demos.deepseek_v3_d_p.reference.gdn.head_slice import (
    gdn_head_slice_channels,
    gdn_head_slice_config,
    slice_gdn_heads,
)
from models.demos.deepseek_v3_d_p.reference.gdn.layer import GDNReferenceState, gdn_forward_reference
from models.demos.deepseek_v3_d_p.reference.gdn.tests.helpers import TINY, random_weights

# Four K heads over TP2 puts two K heads (six V heads) on a rank, so the slice must keep the within-rank K-to-V
# grouping (V head j -> K head j // 3), not only the rank boundary.
_CONFIG = replace(TINY, num_key_heads=4, num_value_heads=12)
_TP = 2


def _nonzero_state(config: GDNConfig, seed: int) -> GDNReferenceState:
    generator = torch.Generator().manual_seed(seed)
    return GDNReferenceState(
        conv=torch.randn(config.conv_kernel_size - 1, config.conv_dim, generator=generator),
        recurrent=0.1 * torch.randn(config.num_value_heads, config.head_k_dim, config.head_v_dim, generator=generator),
    )


def _restrict_state(state: GDNReferenceState, config: GDNConfig, key_head_start: int, num_key_heads: int):
    channels = gdn_head_slice_channels(config, key_head_start=key_head_start, num_key_heads=num_key_heads)
    v_heads = slice(key_head_start * config.group, (key_head_start + num_key_heads) * config.group)
    return GDNReferenceState(conv=state.conv[:, channels], recurrent=state.recurrent[v_heads])


def _rank_partial_reference(hidden, weights, config, state, key_head_start: int, num_key_heads: int) -> torch.Tensor:
    """TP rank partial defined on the full layer: out_proj restricted to the rank's V-head input columns."""
    width = config.group * config.head_v_dim
    columns = slice(key_head_start * width, (key_head_start + num_key_heads) * width)
    masked = torch.zeros_like(weights["out_proj.weight"])
    masked[:, columns] = weights["out_proj.weight"][:, columns]
    output, _ = gdn_forward_reference(hidden, weights | {"out_proj.weight": masked}, config, state)
    return output


@pytest.mark.parametrize("activation", ["silu", "sigmoid"])
def test_slice_reference_is_tp_rank_partial(activation: str) -> None:
    """Each rank's sliced reference (nonzero carried state) equals the full layer's rank partial and the full state
    restricted to the rank; the partials sum to the full output."""
    config = replace(_CONFIG, output_gate_activation=activation)
    weights = {name: tensor.float() for name, tensor in random_weights(config).items()}
    hidden = torch.randn(48, config.hidden_size, generator=torch.Generator().manual_seed(5))
    state = _nonzero_state(config, seed=6)
    full_output, full_state = gdn_forward_reference(hidden, weights, config, state)
    heads = config.num_key_heads // _TP
    partials = []
    for rank in range(_TP):
        start = rank * heads
        sliced_output, sliced_state = gdn_forward_reference(
            hidden,
            slice_gdn_heads(weights, config, key_head_start=start, num_key_heads=heads),
            gdn_head_slice_config(config, heads),
            _restrict_state(state, config, start, heads),
        )
        partials.append(_rank_partial_reference(hidden, weights, config, state, start, heads))
        expected_state = _restrict_state(full_state, config, start, heads)
        # FP32 summation-order noise only: outputs reach |10|, observed differences 2e-6.
        torch.testing.assert_close(sliced_output, partials[-1], atol=1e-5, rtol=1e-5, msg=f"rank {rank} output")
        torch.testing.assert_close(sliced_state.recurrent, expected_state.recurrent, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(sliced_state.conv, expected_state.conv, atol=0, rtol=0)
    torch.testing.assert_close(sum(partials), full_output, atol=1e-5, rtol=1e-5)
    # Negative control: the partials are distinct, so matching the wrong rank cannot pass.
    assert not torch.allclose(partials[0], partials[1], atol=1e-3)


def test_slice_selects_whole_key_groups() -> None:
    """Hand-checked channel map for K heads [1, 2) of 4, G = 3, K = V = 16: q 16..31, k 64+16..64+31, v heads 3..5
    -> 128 + 48..128 + 95."""
    channels = gdn_head_slice_channels(_CONFIG, key_head_start=1, num_key_heads=1)
    expected = torch.cat([torch.arange(16, 32), torch.arange(80, 96), torch.arange(176, 224)])
    assert torch.equal(channels, expected)
    weights = random_weights(_CONFIG)
    sliced = slice_gdn_heads(weights, _CONFIG, key_head_start=1, num_key_heads=1)
    assert torch.equal(sliced["A_log"], weights["A_log"][3:6])
    assert torch.equal(sliced["out_proj.weight"], weights["out_proj.weight"][:, 48:96])


def test_slice_does_not_mutate_or_alias_input() -> None:
    weights = random_weights(_CONFIG)
    original = {name: tensor.clone() for name, tensor in weights.items()}
    sliced = slice_gdn_heads(weights, _CONFIG, key_head_start=2, num_key_heads=2)
    for tensor in sliced.values():
        tensor.add_(1)
    for name, tensor in weights.items():
        assert torch.equal(tensor, original[name]), name


def test_slice_config_changes_only_heads() -> None:
    assert gdn_head_slice_config(_CONFIG, 1) == replace(_CONFIG, num_key_heads=1, num_value_heads=3)


def test_slice_rejects_invalid_ranges(expect_error) -> None:
    weights = random_weights(_CONFIG)
    with expect_error(ValueError, "num_key_heads"):
        slice_gdn_heads(weights, _CONFIG, key_head_start=0, num_key_heads=0)
    with expect_error(ValueError, "num_key_heads"):
        slice_gdn_heads(weights, _CONFIG, key_head_start=0, num_key_heads=5)
    with expect_error(ValueError, "exceeds"):
        slice_gdn_heads(weights, _CONFIG, key_head_start=3, num_key_heads=2)
    with expect_error(ValueError, "missing GDN weights"):
        slice_gdn_heads(
            {k: v for k, v in weights.items() if k != "dt_bias"}, _CONFIG, key_head_start=0, num_key_heads=2
        )
