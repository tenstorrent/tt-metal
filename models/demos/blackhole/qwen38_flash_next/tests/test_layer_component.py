# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import os
from pathlib import Path

import torch

from models.demos.blackhole.qwen38_flash_next.checkpoint import Qwen38Checkpoint
from models.demos.blackhole.qwen38_flash_next.config import Qwen38Placement
from models.demos.blackhole.qwen38_flash_next.tt.layer import Qwen38DecoderLayer

CHECKPOINT = Path(os.environ.get("QWEN38_CHECKPOINT", "/nonexistent/Qwen3.8-Flash-Next"))


def _position_embeddings(sequence):
    positions = torch.arange(sequence, dtype=torch.float32).view(1, sequence, 1)
    frequencies = torch.linspace(0.001, 0.037, 32).view(1, 1, 32)
    angles = positions * frequencies
    return torch.cat((angles.cos(), angles.cos()), dim=-1).to(torch.bfloat16), torch.cat(
        (angles.sin(), angles.sin()), dim=-1
    ).to(torch.bfloat16)


def _causal_mask(sequence):
    visible = torch.ones(sequence, sequence, dtype=torch.bool).tril()
    return torch.where(visible, torch.tensor(0.0), torch.tensor(torch.finfo(torch.float32).min)).view(
        1, 1, sequence, sequence
    )


def _checkpoint_and_placement():
    checkpoint = Qwen38Checkpoint(CHECKPOINT)
    placement = Qwen38Placement(checkpoint.config, mesh_shape=(1, 4), physical_ids=(0, 1, 2, 3))
    return checkpoint, placement


def test_exact_gdn_decoder_layer_prefill_equals_tokenwise_state_transition():
    checkpoint, placement = _checkpoint_and_placement()
    layer = Qwen38DecoderLayer.from_checkpoint(checkpoint, placement, layer_index=0)
    generator = torch.Generator().manual_seed(43)
    hidden = (torch.randn((1, 2, 10240), generator=generator) * 0.02).to(torch.bfloat16)

    full, full_state, full_aux = layer.forward(hidden)
    state = None
    pieces = []
    for position in range(hidden.shape[1]):
        output, state, aux = layer.forward(hidden[:, position : position + 1], state=state)
        pieces.append(output)

    torch.testing.assert_close(torch.cat(pieces, dim=1), full, rtol=0.03, atol=0.03)
    torch.testing.assert_close(state.attention.recurrent, full_state.attention.recurrent, rtol=0.03, atol=0.002)
    assert full_aux.routing.indices.shape == (2, 10)
    assert aux.routing.scores.shape == (1, 10)


def test_exact_ple_layer_and_qsa_layer_form_a_short_alternating_stack():
    checkpoint, placement = _checkpoint_and_placement()
    ple_layer = Qwen38DecoderLayer.from_checkpoint(checkpoint, placement, layer_index=1)
    qsa_layer = Qwen38DecoderLayer.from_checkpoint(checkpoint, placement, layer_index=3)
    generator = torch.Generator().manual_seed(47)
    hidden = (torch.randn((1, 2, 10240), generator=generator) * 0.02).to(torch.bfloat16)
    input_ids = torch.tensor([[17, 248044]], dtype=torch.long)

    hidden, ple_state, ple_aux = ple_layer.forward(hidden, input_ids=input_ids)
    hidden, qsa_state, qsa_aux = qsa_layer.forward(
        hidden,
        position_embeddings=_position_embeddings(2),
        attention_mask=_causal_mask(2),
    )

    assert hidden.shape == (1, 2, 10240)
    assert ple_state.ple.token_context.shape == (1, 2)
    assert ple_state.ple.conv.shape == (1, 10240, 9)
    assert qsa_state.attention.length == 2
    assert qsa_aux.selected_tokens.shape == (1, 1, 2, 2)
    assert ple_aux.routing.indices.shape == qsa_aux.routing.indices.shape == (2, 10)
