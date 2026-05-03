# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Focused tests for Gemma4 vLLM paged-attention metadata."""

import pytest
import torch

from models.demos.gemma4.tt.generator_vllm import layer_kv_cache_shape, validate_page_table
from models.demos.gemma4.tt.model_config import Gemma4ModelArgs


def test_vllm_layer_kv_cache_shape_tracks_mixed_attention_geometry():
    args = Gemma4ModelArgs()

    sliding_shape = layer_kv_cache_shape(args, 0, (1, 8), max_num_blocks=16, block_size=64)
    global_shape = layer_kv_cache_shape(args, 5, (1, 8), max_num_blocks=16, block_size=64)

    assert sliding_shape == (16, 1, 64, 256)
    assert global_shape == (16, 1, 64, 512)
    assert sliding_shape[-1] != global_shape[-1]


def test_vllm_page_table_validation_accepts_noncontiguous_physical_blocks():
    page_table = torch.tensor([[7, 3, 2, 5]], dtype=torch.int32)

    validate_page_table(page_table, block_size=64, max_seq_len=128, batch_size=1)


@pytest.mark.parametrize(
    "page_table, error",
    [
        (torch.arange(4, dtype=torch.int64).reshape(1, 4), TypeError),
        (torch.arange(4, dtype=torch.int32), ValueError),
        (torch.tensor([[-1, 0, 1, 2]], dtype=torch.int32), ValueError),
        (torch.arange(2, dtype=torch.int32).reshape(1, 2), ValueError),
    ],
)
def test_vllm_page_table_validation_rejects_bad_metadata(page_table, error):
    with pytest.raises(error):
        validate_page_table(page_table, block_size=64, max_seq_len=256, batch_size=1)
