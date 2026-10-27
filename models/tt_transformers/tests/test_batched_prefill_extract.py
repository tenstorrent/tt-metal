# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch

import ttnn
from models.tt_transformers.tt.model import Transformer


@pytest.mark.parametrize("mesh_device", [len(ttnn.get_device_ids())], indirect=True)
@pytest.mark.parametrize(
    "padded_batch,prefill_seq_len,target_batch,last_token_idx",
    [
        (4, 128, 32, [99, 59, 119, 79]),
        (4, 128, 4, [63, 63, 63, 63]),
        (4, 128, 32, [63, 0, 63, 63]),
        (8, 1024, 32, [599, 899, 299, 999, 0, 1023, 32, 31]),
        (32, 128, 32, [(7 * i + 5) % 128 for i in range(32)]),
    ],
    ids=["mixed_lengths", "same_length", "one_token_prompt", "long_prompts", "batch_32"],
)
def test_extract_last_tokens_batched_prefill_returns_source_rows(
    mesh_device, padded_batch, prefill_seq_len, target_batch, last_token_idx
):
    torch.manual_seed(0)
    dim = 256 * mesh_device.get_num_devices()
    hidden = torch.randn(padded_batch, 1, prefill_seq_len, dim, dtype=torch.bfloat16)
    hidden_tt = ttnn.from_torch(
        hidden,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=-1),
    )

    model = SimpleNamespace(mesh_device=mesh_device)
    out = Transformer.extract_last_tokens_batched_prefill(
        model, hidden_tt, last_token_idx, padded_batch, prefill_seq_len, target_batch=target_batch
    )
    got = ttnn.to_torch(out, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=-1))

    expected = torch.zeros(1, 1, target_batch, dim, dtype=torch.bfloat16)
    for slot in range(padded_batch):
        expected[0, 0, slot] = hidden[slot, 0, last_token_idx[slot]]
    assert list(got.shape) == [1, 1, target_batch, dim]
    assert torch.equal(got.to(torch.bfloat16), expected)
