# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn
from models.demos.wormhole.bge_m3.tests.test_utils import (
    SEQUENCE_LENGTHS,
    assert_pcc,
    make_lazy_weight,
    require_single_device,
    to_torch,
    to_ttnn_ids,
)
from models.demos.wormhole.bge_m3.tt.embeddings import BgeM3Embedding

VOCAB_SIZE = 4096
HIDDEN_SIZE = 1024
MAX_POSITION_EMBEDDINGS = 8192
PAD_TOKEN_ID = 1
BATCH_SIZE = 1


@pytest.mark.parametrize("seq_len", SEQUENCE_LENGTHS, ids=[f"S{seq_len}" for seq_len in SEQUENCE_LENGTHS])
def test_embeddings_vs_pytorch(device, seq_len):
    require_single_device(device)
    torch.manual_seed(42)

    word_embeddings = torch.nn.Embedding(VOCAB_SIZE, HIDDEN_SIZE, padding_idx=PAD_TOKEN_ID)
    position_embeddings = torch.nn.Embedding(MAX_POSITION_EMBEDDINGS, HIDDEN_SIZE)
    token_type_embeddings = torch.nn.Embedding(1, HIDDEN_SIZE)

    with torch.no_grad():
        word_embeddings.weight.copy_(torch.randn_like(word_embeddings.weight))
        word_embeddings.weight[PAD_TOKEN_ID].zero_()
        position_embeddings.weight.copy_(torch.randn_like(position_embeddings.weight))
        token_type_embeddings.weight.copy_(torch.randn_like(token_type_embeddings.weight))

    tt_model = BgeM3Embedding(
        word_embeddings_weight=make_lazy_weight(
            word_embeddings.weight.detach().clone(),
            device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        ),
        position_embeddings_weight=make_lazy_weight(
            position_embeddings.weight.detach().clone(),
            device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        ),
        token_type_embeddings_weight=make_lazy_weight(
            token_type_embeddings.weight.detach().clone(),
            device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        ),
        vocab_size=VOCAB_SIZE,
        max_position_embeddings=MAX_POSITION_EMBEDDINGS,
        hidden_size=HIDDEN_SIZE,
        pad_token_id=PAD_TOKEN_ID,
    )

    input_ids = torch.randint(PAD_TOKEN_ID + 1, VOCAB_SIZE, (BATCH_SIZE, seq_len), dtype=torch.long)
    input_ids[:, -min(8, seq_len) :] = PAD_TOKEN_ID
    token_type_ids = torch.zeros_like(input_ids)
    position_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0).expand(BATCH_SIZE, -1)

    tt_output = tt_model.forward(
        to_ttnn_ids(input_ids, device),
        token_type_ids=to_ttnn_ids(token_type_ids, device),
        position_ids=to_ttnn_ids(position_ids, device),
    )
    tt_output_torch = to_torch(tt_output, expected_shape=(BATCH_SIZE, 1, seq_len, HIDDEN_SIZE))

    reference_output = (
        word_embeddings(input_ids) + position_embeddings(position_ids) + token_type_embeddings(token_type_ids)
    ).unsqueeze(1)
    assert_pcc(reference_output.to(torch.float32), tt_output_torch, 0.999)
