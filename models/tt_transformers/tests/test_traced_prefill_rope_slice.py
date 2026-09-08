# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import MagicMock, call

import pytest
import torch

import ttnn
from models.tt_transformers.tt.common import Mode
from models.tt_transformers.tt.model import Transformer, _get_trace_rope_table_len, _pad_prefill_rope_tables


def test_resumed_prefill_rope_slice_matches_trace_length_near_context_limit(device):
    max_seq_len = 32768
    prefill_seq_len = 8192
    chunk_start_idx = 28672
    head_dim = 32
    trace_prefill_seq_lens = [128, 1024, 2048, 4096, prefill_seq_len]
    trace_rope_table_len = _get_trace_rope_table_len(max_seq_len, trace_prefill_seq_lens)

    valid_positions = (
        torch.arange(max_seq_len, dtype=torch.float32)
        .to(torch.bfloat16)
        .reshape(1, 1, max_seq_len, 1)
        .expand(1, 1, max_seq_len, head_dim)
        .contiguous()
    )
    rope_setup = SimpleNamespace()
    rope_setup.cos_matrix_prefill = ttnn.from_torch(
        valid_positions,
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
    )
    rope_setup.sin_matrix_prefill = ttnn.from_torch(
        -valid_positions,
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
    )
    _pad_prefill_rope_tables([rope_setup], max_seq_len, trace_prefill_seq_lens)
    full_rot_cos = rope_setup.cos_matrix_prefill
    full_rot_sin = rope_setup.sin_matrix_prefill
    positions = torch.nn.functional.pad(valid_positions, (0, 0, 0, trace_rope_table_len - max_seq_len))

    model = object.__new__(Transformer)
    # Simulate the eight-device mesh width that the old code passed as the partition count.
    model.args = SimpleNamespace(max_seq_len=max_seq_len, num_devices=8)
    model._tt_seq_len_buffer = ttnn.from_torch(
        torch.tensor([1, 1, max_seq_len, head_dim], dtype=torch.int32),
        device=device,
    )
    model._tt_slice_start_zeros_4 = ttnn.from_torch(
        torch.zeros(4, dtype=torch.int32),
        device=device,
    )
    tt_chunk_start_idx = ttnn.from_torch(
        torch.tensor([chunk_start_idx], dtype=torch.int32),
        device=device,
    )

    rot_cos, rot_sin = model._slice_prefill_rot_mats(
        (full_rot_cos, full_rot_sin),
        tt_chunk_start_idx,
        prefill_seq_len,
    )

    expected_positions = positions[:, :, chunk_start_idx : chunk_start_idx + prefill_seq_len, :]
    assert rot_cos.shape[2] == prefill_seq_len
    assert rot_sin.shape[2] == prefill_seq_len
    torch.testing.assert_close(ttnn.to_torch(rot_cos), expected_positions)
    torch.testing.assert_close(ttnn.to_torch(rot_sin), -expected_positions)


@pytest.mark.parametrize(
    "batch_size, expected_prefill_seq_len",
    [(1, 8192), (8, 1024)],
    ids=["single_user", "batched"],
)
def test_prefill_forward_passes_per_user_length_to_global_and_local_rope_slices(batch_size, expected_prefill_seq_len):
    model = object.__new__(Transformer)
    model.prefetcher = None
    model.layers = []
    model._slice_prefill_rot_mats = MagicMock(side_effect=lambda rot_mats, *_: rot_mats)

    # Batched prefill hands forward the flattened [1, 1, batch_size * S_per_user, dim] activation.
    x = SimpleNamespace(shape=(1, 1, 8192, 32))
    rot_mats_global = object()
    rot_mats_local = object()
    chunk_start_idx = object()

    result = model.forward(
        x,
        current_pos=None,
        rot_mats_global=rot_mats_global,
        rot_mats_local=rot_mats_local,
        mode=Mode.PREFILL,
        chunk_start_idx=chunk_start_idx,
        batch_size=batch_size,
    )

    assert result is x
    assert model._slice_prefill_rot_mats.call_args_list == [
        call(rot_mats_global, chunk_start_idx, expected_prefill_seq_len),
        call(rot_mats_local, chunk_start_idx, expected_prefill_seq_len),
    ]


def test_trace_rope_table_supports_every_bucket_and_dynamic_start():
    max_seq_len = 30000
    trace_prefill_seq_lens = [128, 1024, 2048, 4096, 8192]

    table_len = _get_trace_rope_table_len(max_seq_len, trace_prefill_seq_lens)

    assert table_len >= max_seq_len + max(trace_prefill_seq_lens)
    assert all(table_len % seq_len == 0 for seq_len in trace_prefill_seq_lens)
