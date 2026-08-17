# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Critical host-oracle and device-parity tests for self-conditioning."""

import os

import pytest
import torch

import ttnn
from models.experimental.diffusion_gemma.reference.self_conditioning import SelfConditioning
from models.experimental.diffusion_gemma.tt.self_conditioning import TtSelfConditioning
from tests.ttnn.utils_for_testing import assert_with_pcc


def _generator(seed=0):
    return torch.Generator().manual_seed(seed)


def test_zero_signal_is_post_norm_of_embeds_not_identity():
    batch, length, vocab, hidden = 2, 8, 32, 16
    module = SelfConditioning(hidden, intermediate_size=24)
    embeddings = torch.randn(batch, length, hidden, generator=_generator(3))
    embedding_weight = torch.randn(vocab, hidden, generator=_generator(4))

    out = module.condition(embeddings, None, embedding_weight, enabled=False)

    assert torch.allclose(out, module.post_norm(embeddings), atol=1e-6)
    assert torch.allclose(out, module(embeddings, torch.zeros_like(embeddings)), atol=1e-6)
    assert not torch.allclose(out, embeddings, atol=1e-3)


def test_soft_embedding_onehot_recovers_scaled_token_row():
    vocab, hidden = 20, 12
    embedding_weight = torch.randn(vocab, hidden, generator=_generator(5))
    logits = torch.full((1, 1, vocab), -1e4)
    logits[..., 7] = 1e4

    soft = SelfConditioning.soft_embedding(logits, embedding_weight)

    assert torch.allclose(soft[0, 0], embedding_weight[7] * (hidden**0.5), atol=1e-3)


# 26B-A4B dimensions: self-conditioning uses the dense intermediate size, not MoE.
HIDDEN, INTERMEDIATE, EPSILON = 2816, 2112, 1e-6
SEQUENCE_LENGTH = 256
VOCAB_SIZE = 256

_requires_device = pytest.mark.skipif(
    os.environ.get("DG_RUN_DEVICE") != "1",
    reason="set DG_RUN_DEVICE=1 to run on a Tenstorrent device (needs sfpi >= 7.60.0)",
)
_module_device = pytest.mark.use_module_device


def _build(seed):
    torch.manual_seed(seed)
    reference = SelfConditioning(
        HIDDEN,
        INTERMEDIATE,
        eps=EPSILON,
        activation="gelu_pytorch_tanh",
    ).eval()
    state = {
        "pre_norm.weight": reference.pre_norm.weight.data.clone(),
        "gate_proj.weight": reference.gate_proj.weight.data.clone(),
        "up_proj.weight": reference.up_proj.weight.data.clone(),
        "down_proj.weight": reference.down_proj.weight.data.clone(),
    }
    device_module_kwargs = {
        "hidden_size": HIDDEN,
        "intermediate_size": INTERMEDIATE,
        "eps": EPSILON,
    }
    return reference, state, device_module_kwargs


def _to_device(value, device):
    return ttnn.from_torch(
        value.unsqueeze(0),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )


def _embedding_to_device(embedding_weight, device):
    return ttnn.from_torch(
        embedding_weight.unsqueeze(0).unsqueeze(0),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )


@_requires_device
@_module_device
def test_self_conditioning_pcc(device):
    reference, state, kwargs = _build(0)
    device_module = TtSelfConditioning(device, state, **kwargs)
    embeddings = torch.randn(1, SEQUENCE_LENGTH, HIDDEN)
    signal = torch.randn(1, SEQUENCE_LENGTH, HIDDEN)

    with torch.no_grad():
        golden = reference(embeddings, signal)
    out = ttnn.to_torch(device_module.forward(_to_device(embeddings, device), _to_device(signal, device)))[0]

    assert_with_pcc(golden, out, 0.99)


@_requires_device
@_module_device
def test_zero_signal_is_post_norm_of_embeds(device):
    reference, state, kwargs = _build(1)
    device_module = TtSelfConditioning(device, state, **kwargs)
    embeddings = torch.randn(1, SEQUENCE_LENGTH, HIDDEN)
    signal = torch.zeros(1, SEQUENCE_LENGTH, HIDDEN)

    with torch.no_grad():
        golden = reference(embeddings, signal)
    out = ttnn.to_torch(device_module.forward(_to_device(embeddings, device), _to_device(signal, device)))[0]

    assert_with_pcc(golden, out, 0.99)
    assert not torch.allclose(out.float(), embeddings, atol=1e-3)


@_requires_device
@_module_device
def test_condition_full_path_pcc(device):
    reference, state, kwargs = _build(2)
    device_module = TtSelfConditioning(device, state, **kwargs)
    embeddings = torch.randn(1, SEQUENCE_LENGTH, HIDDEN)
    previous_logits = torch.randn(1, SEQUENCE_LENGTH, VOCAB_SIZE)
    embedding_weight = torch.randn(VOCAB_SIZE, HIDDEN)

    with torch.no_grad():
        golden = reference.condition(embeddings, previous_logits, embedding_weight, enabled=True)
    out = ttnn.to_torch(
        device_module.condition(
            _to_device(embeddings, device),
            _to_device(previous_logits, device),
            _embedding_to_device(embedding_weight, device),
        )
    )[0]

    assert_with_pcc(golden, out, 0.99)


@_requires_device
@_module_device
def test_condition_none_logits_is_post_norm(device):
    reference, state, kwargs = _build(3)
    device_module = TtSelfConditioning(device, state, **kwargs)
    embeddings = torch.randn(1, SEQUENCE_LENGTH, HIDDEN)
    embedding_weight = torch.randn(VOCAB_SIZE, HIDDEN)

    with torch.no_grad():
        golden = reference.condition(embeddings, None, embedding_weight, enabled=False)
    out = ttnn.to_torch(
        device_module.condition(
            _to_device(embeddings, device),
            None,
            _embedding_to_device(embedding_weight, device),
        )
    )[0]

    assert_with_pcc(golden, out, 0.99)
