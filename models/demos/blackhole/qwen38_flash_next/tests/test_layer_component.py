# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0


import torch

from models.demos.blackhole.qwen38_flash_next.tests.helpers.model import qsa_causal_mask, qsa_position_embeddings
from models.demos.blackhole.qwen38_flash_next.tt.layer import Qwen38DecoderLayer


def test_exact_gdn_decoder_layer_prefill_equals_tokenwise_state_transition(release_checkpoint, release_placement):
    layer = Qwen38DecoderLayer.from_checkpoint(release_checkpoint, release_placement, layer_index=0)
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


def test_exact_ple_layer_and_qsa_layer_form_a_short_alternating_stack(release_checkpoint, release_placement):
    ple_layer = Qwen38DecoderLayer.from_checkpoint(release_checkpoint, release_placement, layer_index=1)
    qsa_layer = Qwen38DecoderLayer.from_checkpoint(release_checkpoint, release_placement, layer_index=3)
    generator = torch.Generator().manual_seed(47)
    hidden = (torch.randn((1, 2, 10240), generator=generator) * 0.02).to(torch.bfloat16)
    input_ids = torch.tensor([[17, 248044]], dtype=torch.long)

    hidden, ple_state, ple_aux = ple_layer.forward(hidden, input_ids=input_ids)
    hidden, qsa_state, qsa_aux = qsa_layer.forward(
        hidden,
        position_embeddings=qsa_position_embeddings(2),
        attention_mask=qsa_causal_mask(2),
    )

    assert hidden.shape == (1, 2, 10240)
    assert ple_state.ple.token_context.shape == (1, 2)
    assert ple_state.ple.conv.shape == (1, 10240, 9)
    assert qsa_state.attention.length == 2
    assert qsa_aux.selected_tokens.shape == (1, 1, 2, 2)
    assert ple_aux.routing.indices.shape == qsa_aux.routing.indices.shape == (2, 10)
