# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU tests: a contiguous KDA head slice is one TP rank's shard and its reference is that rank's oracle."""

from dataclasses import replace
from pathlib import Path

import pytest
import torch

from models.demos.deepseek_v3_d_p.reference.glm_5_3_flash_config import glm_5_3_flash_kda_config
from models.demos.deepseek_v3_d_p.reference.kda import KDAReferenceState, kda_forward_reference
from models.demos.deepseek_v3_d_p.reference.kda.config import KDAConfig
from models.demos.deepseek_v3_d_p.reference.kimi_k3_config import kimi_k3_kda_config
from models.demos.deepseek_v3_d_p.tests.kda.head_slice import (
    galaxy_chip_head_slice_config,
    kda_head_slice_config,
    slice_kda_heads,
)
from models.demos.deepseek_v3_d_p.tests.kda.utils import make_kimi_k3_test_case, random_weights
from models.demos.deepseek_v3_d_p.tt.kda.weights import _prepare_kda_host_weights

_TP = 4

_GATE_CASES = {
    # Low-rank output gate (GLM-5.3-Flash) with the bounded decay used by both production models.
    "low_rank_gate-bounded_decay": {"use_full_rank_gate": False, "gate_lower_bound": -5.0},
    # Full-rank output gate (Kimi K3) with the softplus decay path.
    "full_rank_gate-softplus_decay": {"use_full_rank_gate": True, "gate_lower_bound": None},
}


def _config(**gate: object) -> KDAConfig:
    return KDAConfig(
        hidden_size=64, num_heads=8, head_k_dim=32, head_v_dim=32, conv_kernel_size=4, norm_eps=1e-5, **gate
    )


def _nonzero_state(config: KDAConfig, seed: int) -> KDAReferenceState:
    generator = torch.Generator().manual_seed(seed)
    history = config.conv_kernel_size - 1
    return KDAReferenceState(
        recurrent=0.1 * torch.randn(1, config.num_heads, config.head_k_dim, config.head_v_dim, generator=generator),
        q_convolution=torch.randn(1, history, config.q_dim, generator=generator),
        k_convolution=torch.randn(1, history, config.k_dim, generator=generator),
        v_convolution=torch.randn(1, history, config.v_dim, generator=generator),
    )


def _restrict_state(state: KDAReferenceState, config: KDAConfig, head_start: int, heads: int) -> KDAReferenceState:
    stop = head_start + heads
    return KDAReferenceState(
        recurrent=state.recurrent[:, head_start:stop],
        q_convolution=state.q_convolution[..., head_start * config.head_k_dim : stop * config.head_k_dim],
        k_convolution=state.k_convolution[..., head_start * config.head_k_dim : stop * config.head_k_dim],
        v_convolution=state.v_convolution[..., head_start * config.head_v_dim : stop * config.head_v_dim],
    )


def _rank_partial_reference(
    hidden: torch.Tensor,
    weights: dict[str, torch.Tensor],
    config: KDAConfig,
    state: KDAReferenceState,
    head_start: int,
    heads: int,
) -> torch.Tensor:
    """TP rank partial defined on the full layer: o_proj restricted to the rank's input columns."""
    columns = slice(head_start * config.head_v_dim, (head_start + heads) * config.head_v_dim)
    masked_output_projection = torch.zeros_like(weights["o_proj.weight"])
    masked_output_projection[:, columns] = weights["o_proj.weight"][:, columns]
    output, _ = kda_forward_reference(hidden, weights | {"o_proj.weight": masked_output_projection}, config, state)
    return output


def _assert_slice_is_rank_oracle(
    hidden: torch.Tensor,
    weights: dict[str, torch.Tensor],
    config: KDAConfig,
    state: KDAReferenceState,
    tp: int,
    ranks: tuple[int, ...],
    *,
    atol: float,
    rtol: float,
) -> None:
    heads = config.num_heads // tp
    _, full_state = kda_forward_reference(hidden, weights, config, state)
    for rank in ranks:
        head_start = rank * heads
        sliced_config = kda_head_slice_config(config, heads)
        sliced_output, sliced_state = kda_forward_reference(
            hidden,
            slice_kda_heads(weights, config, num_heads=heads, head_start=head_start),
            sliced_config,
            _restrict_state(state, config, head_start, heads),
        )
        expected_output = _rank_partial_reference(hidden, weights, config, state, head_start, heads)
        expected_state = _restrict_state(full_state, config, head_start, heads)
        torch.testing.assert_close(sliced_output, expected_output, atol=atol, rtol=rtol, msg=f"rank {rank} output")
        for field in ("recurrent", "q_convolution", "k_convolution", "v_convolution"):
            torch.testing.assert_close(
                getattr(sliced_state, field),
                getattr(expected_state, field),
                atol=atol,
                rtol=rtol,
                msg=f"rank {rank} {field}",
            )


@pytest.mark.parametrize("gate", _GATE_CASES.values(), ids=_GATE_CASES.keys())
def test_slice_reference_is_tp_rank_partial(gate: dict[str, object]) -> None:
    """Each rank's sliced reference equals the full layer's rank partial; the partials sum to the full output."""
    config = _config(**gate)
    weights = random_weights(config)
    hidden = torch.randn(1, 48, config.hidden_size, generator=torch.Generator().manual_seed(5))
    state = _nonzero_state(config, seed=6)
    heads = config.num_heads // _TP

    _assert_slice_is_rank_oracle(hidden, weights, config, state, _TP, tuple(range(_TP)), atol=1e-6, rtol=1e-5)

    full_output, _ = kda_forward_reference(hidden, weights, config, state)
    partials = [_rank_partial_reference(hidden, weights, config, state, rank * heads, heads) for rank in range(_TP)]
    torch.testing.assert_close(sum(partials), full_output, atol=1e-6, rtol=1e-5)
    # Negative control: the partials are distinct, so matching the wrong rank cannot pass.
    assert not torch.allclose(partials[0], partials[1], atol=1e-3)


# Shard dimension of every prepared tensor, as mapped onto the TP axis by tt/kda/weights.py; None = replicated.
_PREPARED_SHARD_DIMS = {
    "input_projection": -1,
    "decay_output_projection": -1,
    "output_projection": -2,
    "decay_scale_flat": -1,
    "decay_bias_flat": -1,
    "norm": None,
}


@pytest.mark.parametrize("gate", _GATE_CASES.values(), ids=_GATE_CASES.keys())
def test_slice_equals_tp_rank_device_shard(gate: dict[str, object]) -> None:
    """TP rank r's shard of the packed TP4 weights equals the TP1 packing of head slice r."""
    config = _config(**gate)
    weights = random_weights(config)
    heads = config.num_heads // _TP
    packed = _prepare_kda_host_weights(weights, config, _TP)
    for rank in range(_TP):
        sliced_config = kda_head_slice_config(config, heads)
        sliced = _prepare_kda_host_weights(
            slice_kda_heads(weights, config, num_heads=heads, head_start=rank * heads), sliced_config, 1
        )
        for name, shard_dim in _PREPARED_SHARD_DIMS.items():
            tensor = getattr(packed, name)
            expected = tensor if shard_dim is None else tensor.chunk(_TP, dim=shard_dim)[rank]
            torch.testing.assert_close(getattr(sliced, name), expected, atol=0, rtol=0, msg=f"rank {rank} {name}")
        for tap, (tensor, sliced_tap) in enumerate(zip(packed.convolution_taps, sliced.convolution_taps, strict=True)):
            torch.testing.assert_close(
                sliced_tap, tensor.chunk(_TP, dim=-1)[rank], atol=0, rtol=0, msg=f"rank {rank} conv tap {tap}"
            )


def test_slice_rejects_invalid_head_ranges(expect_error) -> None:
    config = _config(use_full_rank_gate=False)
    weights = random_weights(config)
    with expect_error(ValueError, "num_heads"):
        slice_kda_heads(weights, config, num_heads=0)
    with expect_error(ValueError, "num_heads"):
        slice_kda_heads(weights, config, num_heads=config.num_heads + 1)
    with expect_error(ValueError, "exceeds"):
        slice_kda_heads(weights, config, num_heads=2, head_start=7)
    with expect_error(ValueError, "missing KDA weight"):
        slice_kda_heads({k: v for k, v in weights.items() if k != "g_b_proj.weight"}, config, num_heads=2)


def test_slice_config_changes_only_heads() -> None:
    config = _config(use_full_rank_gate=True, gate_lower_bound=-5.0)
    assert kda_head_slice_config(config, 2) == replace(config, num_heads=2)


@pytest.mark.parametrize(
    "build, chip_heads",
    [
        pytest.param(kimi_k3_kda_config, 24, id="kimi_k3"),
        pytest.param(glm_5_3_flash_kda_config, 16, id="glm_5_3_flash"),
    ],
)
def test_lb_b_chip_config_is_galaxy_tp4_share(build, chip_heads: int) -> None:
    """LB-B per-chip configs built from the real config.json: K3 96 -> 24 heads, GLM 64 -> 16 heads."""
    config = build()
    assert galaxy_chip_head_slice_config(config) == replace(config, num_heads=chip_heads)


def test_lb_b_chip_config_rejects_indivisible_heads(expect_error) -> None:
    with expect_error(ValueError, "not divisible"):
        galaxy_chip_head_slice_config(replace(_config(), num_heads=6))


def test_kimi_k3_layer_slice_is_tp_rank_partial(kimi_k3_checkpoint_dir: Path) -> None:
    """Real K3 layer 1 (local only): the LB-B quarter slice (24 heads) is the TP4 rank oracle for ranks 0 and 3."""
    case = make_kimi_k3_test_case(kimi_k3_checkpoint_dir, sequence=32)
    weights = {name: tensor.float() for name, tensor in case.state_dict.items()}
    state = _nonzero_state(case.config, seed=7)
    _assert_slice_is_rank_oracle(
        case.hidden.float(), weights, case.config, state, _TP, (0, _TP - 1), atol=1e-4, rtol=1e-4
    )
