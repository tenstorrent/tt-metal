# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch

import ttnn
from models.tt_transformers.tt.model import Transformer


def _torch_reference_apply_grammar_bitmask(logits: torch.Tensor, grammar_bitmask: torch.Tensor) -> torch.Tensor:
    # Equivalent to vllm/v1/worker/tt_model_runner.py::apply_grammar_bitmask
    structured_output_arange = torch.arange(32, dtype=torch.int32, device=grammar_bitmask.device)
    unpacked_bitmask = (
        torch.bitwise_right_shift(
            grammar_bitmask[:, :, None],
            structured_output_arange[None, None, :],
        )
        & 1
    ) == 0
    unpacked_bitmask = unpacked_bitmask.reshape(grammar_bitmask.shape[0], -1)[:, : logits.shape[-1]]
    out = logits.clone()
    out.masked_fill_(unpacked_bitmask, -float("inf"))
    return out


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_unpack_bitmask_matches_torch_reference(mesh_device):
    batch_size = 2
    vocab_size = 50
    padded_vocab_dim = 64
    packed_vocab_dim = (vocab_size + 31) // 32

    grammar_bitmask = torch.randint(0, 2**31 - 1, (batch_size, packed_vocab_dim), dtype=torch.int32)
    grammar_bitmask_tt = ttnn.from_torch(
        grammar_bitmask, device=mesh_device, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT
    )
    bitmask_arange_tt = ttnn.from_torch(
        torch.arange(32, dtype=torch.int32).view(1, 1, 32),
        device=mesh_device,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )

    transformer = SimpleNamespace(
        bitmask_arange=bitmask_arange_tt,
        vocab_size=vocab_size,
    )

    unpacked_tt = Transformer.unpack_bitmask(transformer, grammar_bitmask_tt, padded_vocab_dim=padded_vocab_dim)
    unpacked_tt_torch = ttnn.to_torch(unpacked_tt)

    # Build expected mask from the known-good vLLM logic (then pad to match TT shape).
    zeros_logits = torch.zeros((batch_size, vocab_size), dtype=torch.float32)
    expected_unpadded = _torch_reference_apply_grammar_bitmask(zeros_logits, grammar_bitmask)
    expected = torch.nn.functional.pad(expected_unpadded, (0, padded_vocab_dim - vocab_size), value=0.0)

    assert torch.equal(unpacked_tt_torch, expected)


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_apply_bitmask_to_logits_matches_torch_reference(mesh_device):
    batch_size = 3
    vocab_size = 96
    padded_vocab_dim = 128
    packed_vocab_dim = (vocab_size + 31) // 32
    grammar_bitmask = torch.randint(0, 2**31 - 1, (batch_size, packed_vocab_dim), dtype=torch.int32)
    grammar_bitmask_tt = ttnn.from_torch(
        grammar_bitmask, device=mesh_device, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT
    )
    bitmask_arange_tt = ttnn.from_torch(
        torch.arange(32, dtype=torch.int32).view(1, 1, 32),
        device=mesh_device,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )

    # TT logits are shaped [1, 1, batch, padded_vocab].
    tt_logits_torch = torch.randn((1, 1, batch_size, padded_vocab_dim), dtype=torch.float32)
    tt_logits = ttnn.from_torch(
        tt_logits_torch.clone(), device=mesh_device, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT
    )
    expected_unpadded = _torch_reference_apply_grammar_bitmask(tt_logits_torch[0, 0, :, :vocab_size], grammar_bitmask)

    transformer = SimpleNamespace(
        bitmask=grammar_bitmask_tt,
        bitmask_arange=bitmask_arange_tt,
        vocab_size=vocab_size,
        args=SimpleNamespace(padded_vocab_size=padded_vocab_dim),
    )
    transformer.unpack_bitmask = lambda bitmask, padded_vocab_dim: Transformer.unpack_bitmask(
        transformer, bitmask, padded_vocab_dim
    )

    out = Transformer.apply_bitmask_to_logits(transformer, tt_logits)
    out_torch = ttnn.to_torch(out)

    # Method is documented as in-place.
    assert out is tt_logits
    # Non-padded vocab logits must match the known-good torch application exactly.
    assert torch.equal(out_torch[0, 0, :, :vocab_size], expected_unpadded)
    # Padded logits are unchanged by mask application.
    assert torch.equal(out_torch[0, 0, :, vocab_size:], tt_logits_torch[0, 0, :, vocab_size:])
