# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import torch.nn.functional as F

import ttnn
from models.common.auto_compose import to_torch_auto_compose
from models.common.modules.lazy_weight import LazyWeight
from models.common.utility_functions import comp_allclose, comp_pcc
from models.demos.wormhole.bge_m3.tt.embeddings import BgeM3Embedding


def _lazy_weight_from_2d(weight_2d: torch.Tensor, device) -> LazyWeight:
    return LazyWeight(
        source=weight_2d.unsqueeze(0).unsqueeze(0),
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def _to_ttnn_ids(input_ids: torch.Tensor, rank: int, device) -> ttnn.Tensor:
    if rank == 4:
        input_ids = input_ids.reshape(input_ids.shape[0], 1, 1, input_ids.shape[1])

    return ttnn.from_torch(
        input_ids.to(torch.int32),
        device=device,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )


def _to_torch_output(tt_output: ttnn.Tensor, batch_size: int, seq_len: int, hidden_size: int) -> torch.Tensor:
    out = to_torch_auto_compose(tt_output).to(torch.float32)

    expected_shape = (batch_size, 1, seq_len, hidden_size)
    assert tuple(out.shape) == expected_shape, f"Expected output shape {expected_shape}, got {tuple(out.shape)}"
    return out


def _reference_sum_embeddings(
    input_ids: torch.Tensor,
    word_weight: torch.Tensor,
    position_weight: torch.Tensor,
    token_type_weight: torch.Tensor,
    pad_token_id: int,
) -> torch.Tensor:
    batch_size, seq_len = input_ids.shape

    word_weight_ref = word_weight.clone()
    word_weight_ref[pad_token_id] = 0
    word = F.embedding(input_ids, word_weight_ref)

    position_ids = torch.arange(0, seq_len, dtype=torch.long).unsqueeze(0).repeat(batch_size, 1)
    pos = F.embedding(position_ids, position_weight)

    token_type_ids = torch.zeros_like(input_ids)
    typ = F.embedding(token_type_ids, token_type_weight)

    return word + pos + typ


@pytest.mark.parametrize("input_rank", [2, 4], ids=["rank2-input", "rank4-input"])
def test_ttnn_bge_m3_embeddings_vs_reference(device, input_rank):
    """
    Validation target:
      - Numerical parity (sum of token + position + token_type embeddings).
      - Batch behavior for B=2.
      - Input adapter behavior for both [B,S] and [B,1,1,S].
    """
    torch.manual_seed(42)

    vocab_size = 256
    hidden_size = 64
    max_position_embeddings = 64
    pad_token_id = 0
    batch_size = 2
    seq_len = 32

    word_weight = torch.randn(vocab_size, hidden_size, dtype=torch.bfloat16)
    word_weight[pad_token_id] = 0
    position_weight = torch.randn(max_position_embeddings, hidden_size, dtype=torch.bfloat16)
    token_type_weight = torch.randn(1, hidden_size, dtype=torch.bfloat16)

    model = BgeM3Embedding(
        word_embeddings_weight=_lazy_weight_from_2d(word_weight, device),
        position_embeddings_weight=_lazy_weight_from_2d(position_weight, device),
        token_type_embeddings_weight=_lazy_weight_from_2d(token_type_weight, device),
        vocab_size=vocab_size,
        max_position_embeddings=max_position_embeddings,
        hidden_size=hidden_size,
        pad_token_id=pad_token_id,
    )

    input_ids = torch.randint(1, vocab_size, (batch_size, seq_len), dtype=torch.long)
    tt_input_ids = _to_ttnn_ids(input_ids, rank=input_rank, device=device)
    tt_output = model.forward(tt_input_ids)
    tt_output_torch = _to_torch_output(tt_output, batch_size=batch_size, seq_len=seq_len, hidden_size=hidden_size)

    ref_output = _reference_sum_embeddings(
        input_ids=input_ids,
        word_weight=word_weight.to(torch.float32),
        position_weight=position_weight.to(torch.float32),
        token_type_weight=token_type_weight.to(torch.float32),
        pad_token_id=pad_token_id,
    ).unsqueeze(1)

    passing, pcc_message = comp_pcc(ref_output, tt_output_torch, 0.999)
    allclose, allclose_message = comp_allclose(ref_output, tt_output_torch)
    assert passing, f"PCC check failed: {pcc_message}; {allclose_message}; allclose={allclose}"
