# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Critical on-device correctness tests for the concat-experts denoise MoE."""

import os
from types import SimpleNamespace

import pytest
import torch

import ttnn
from models.common.utility_functions import comp_pcc
from models.experimental.diffusion_gemma.tt import concat_moe


_requires_device = pytest.mark.skipif(
    os.environ.get("DG_RUN_DEVICE") != "1",
    reason="set DG_RUN_DEVICE=1 to run on a Tenstorrent device",
)
_module_device = pytest.mark.use_module_device

# Structurally faithful but small: expert count is a multiple of 32's half-tile,
# sequence length is the production canvas length, and all matmul dimensions are tiled.
_E, _H, _I, _S, _TOPK = 16, 256, 64, 256, 4


def _rand(*shape, seed):
    return torch.randn(*shape, generator=torch.Generator().manual_seed(seed))


def _make_weights(seed=3):
    return SimpleNamespace(
        gate_proj=_rand(1, _E, _H, _I, seed=seed),
        up_proj=_rand(1, _E, _H, _I, seed=seed + 1),
        down_proj=_rand(1, _E, _I, _H, seed=seed + 2),
        intermediate_size_per_device=_I,
    )


def _make_routing(seed=9, zero_unselected=True):
    generator = torch.Generator().manual_seed(seed)
    logits = torch.randn(1, 1, _S, _E, generator=generator)
    probs = torch.softmax(logits, dim=-1)
    if not zero_unselected:
        return probs
    topk = torch.topk(probs, _TOPK, dim=-1).indices
    mask = torch.zeros_like(probs)
    mask.scatter_(-1, topk, 1.0)
    return probs * mask


def _torch_oracle(x, weights, routing):
    """Per-expert host oracle for the folded concat implementation."""
    num_experts, hidden = weights.gate_proj.shape[1], weights.gate_proj.shape[2]
    sequence = x.shape[-2]
    out = torch.zeros(1, 1, sequence, hidden, dtype=torch.float32)
    x_float = x.float().reshape(sequence, hidden)
    for expert in range(num_experts):
        gate = x_float @ weights.gate_proj[0, expert].float()
        up = x_float @ weights.up_proj[0, expert].float()
        activation = torch.nn.functional.gelu(gate, approximate="tanh") * up
        out[0, 0] += (activation @ weights.down_proj[0, expert].float()) * routing[0, 0, :, expert : expert + 1].float()
    return out


def _to_device(value, device):
    return ttnn.from_torch(value, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)


def _fake_experts(weights, device):
    return SimpleNamespace(
        weights=SimpleNamespace(
            gate_proj=_to_device(weights.gate_proj, device),
            up_proj=_to_device(weights.up_proj, device),
            down_proj=_to_device(weights.down_proj, device),
            intermediate_size_per_device=int(weights.gate_proj.shape[3]),
        ),
        mesh_config=None,
        ccl_manager=None,
    )


def _run_concat(device, weights, routing, x):
    experts = _fake_experts(weights, device)
    tt_x = _to_device(x, device)
    tt_routing = _to_device(routing, device)
    try:
        out = concat_moe.concat_experts_forward(experts, tt_x, tt_routing)
        hidden, sequence = weights.gate_proj.shape[2], x.shape[-2]
        return ttnn.to_torch(out).float().reshape(1, 1, sequence, hidden)
    finally:
        cached = getattr(experts, "_dg_concat_weights", None)
        if cached is not None:
            cached.deallocate()
        for tensor in (
            tt_x,
            tt_routing,
            experts.weights.gate_proj,
            experts.weights.up_proj,
            experts.weights.down_proj,
        ):
            tensor.deallocate(True)


@_requires_device
@_module_device
def test_concat_matches_per_expert_oracle(device):
    weights = _make_weights()
    routing = _make_routing()
    x = _rand(1, 1, _S, _H, seed=21) * 0.1

    got = _run_concat(device, weights, routing, x)
    expected = _torch_oracle(x, weights, routing)

    passing, pcc = comp_pcc(expected, got, 0.99)
    assert passing, f"concat MoE disagrees with the per-expert oracle: {pcc}"


@_requires_device
@_module_device
def test_fold_requires_zero_for_unselected_experts(device):
    weights = _make_weights()
    x = _rand(1, 1, _S, _H, seed=22) * 0.1
    masked = _make_routing(zero_unselected=True)
    unmasked = _make_routing(zero_unselected=False)

    got_masked = _run_concat(device, weights, masked, x)
    got_unmasked = _run_concat(device, weights, unmasked, x)
    expected = _torch_oracle(x, weights, masked)

    passing, pcc = comp_pcc(expected, got_masked, 0.99)
    assert passing, f"masked fold disagrees with the oracle: {pcc}"
    assert (got_unmasked - got_masked).abs().max().item() > 1e-3


@_requires_device
@_module_device
def test_padded_intermediate_columns_contribute_zero(device):
    weights = _make_weights()
    routing = _make_routing()
    x = _rand(1, 1, _S, _H, seed=23) * 0.1
    baseline = _run_concat(device, weights, routing, x)

    pad = 32
    padded = SimpleNamespace(
        gate_proj=torch.nn.functional.pad(weights.gate_proj, (0, pad)),
        up_proj=torch.nn.functional.pad(weights.up_proj, (0, pad)),
        down_proj=torch.nn.functional.pad(weights.down_proj, (0, 0, 0, pad)),
        intermediate_size_per_device=_I + pad,
    )
    got = _run_concat(device, padded, routing, x)

    passing, pcc = comp_pcc(baseline, got, 0.999)
    assert passing, f"zero-padded intermediate changed the result: {pcc}"


@_requires_device
@_module_device
def test_down_concat_is_a_pure_reshape(device):
    weights = _make_weights()
    source = _to_device(weights.down_proj, device)
    try:
        info = concat_moe.verify_down_concat_is_free(SimpleNamespace(down_proj=source))
        assert info["values_match"], f"down concat is not byte-order preserving: {info}"
    finally:
        source.deallocate(True)
