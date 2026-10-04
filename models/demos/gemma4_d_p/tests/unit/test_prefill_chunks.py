# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host checks for fixed-shape prefill traces with partial final chunks."""

from types import SimpleNamespace

import pytest
import torch

from models.demos.gemma4_d_p.demo import text_demo_prefill as demo
from models.demos.gemma4_d_p.tt.attention.ring_prefill import ring_sdpa_chunk_sizes
from models.demos.gemma4_d_p.tt.model import _cp_chunk_major_row_order, prefill_chunk_geometry_error


@pytest.mark.parametrize("chunk_size,n_chunks", [(3328, 79), (6656, 40), (8192, 32), (9984, 27)])
@pytest.mark.parametrize("cp", [4, 8])
def test_partial_final_chunk_preserves_context_and_rope_positions(chunk_size, n_chunks, cp):
    context_len = 262144
    tokens = torch.arange(context_len, dtype=torch.int32).unsqueeze(0)
    padded = demo._pad_prefill_tokens(tokens, chunk_size)
    assert padded.shape == (1, n_chunks * chunk_size)
    assert padded.dtype == tokens.dtype
    torch.testing.assert_close(padded[:, :context_len], tokens)
    assert torch.count_nonzero(padded[:, context_len:]) == 0
    assert prefill_chunk_geometry_error(chunk_size, cp, padded.shape[-1]) is None

    # Reassemble all CP-local RoPE shards in replay order, including the final trace.
    order = _cp_chunk_major_row_order(padded.shape[-1], cp, chunk_size)
    positions = order.reshape(cp, n_chunks, chunk_size // cp).transpose(0, 1).reshape(-1)
    torch.testing.assert_close(positions[:context_len], torch.arange(context_len))
    assert positions[-1] == padded.shape[-1] - 1
    assert 0 < context_len - (n_chunks - 1) * chunk_size <= chunk_size


@pytest.mark.parametrize(
    "cp,chunk_size,expected",
    [
        (4, 3328, (32, 416)),
        (8, 3328, (32, 416)),
        (4, 6656, (64, 320)),
        (8, 6656, (64, 320)),
        (4, 8192, (96, 256)),
        (8, 8192, (96, 256)),
        (4, 9984, (96, 256)),
        (8, 9984, (96, 256)),
    ],
)
def test_global_sdpa_uses_requested_chunks(chunk_size, expected, cp):
    assert ring_sdpa_chunk_sizes(chunk_size // cp, sliding=False, cp_degree=cp) == expected


@pytest.mark.parametrize("chunk_size", demo.PREFILL_CHUNK_SIZES)
@pytest.mark.parametrize("cp", [4, 8])
def test_sliding_sdpa_blocks_divide_each_rank_slab(chunk_size, cp):
    slab = chunk_size // cp
    q_chunk, k_chunk = ring_sdpa_chunk_sizes(slab, sliding=True, cp_degree=cp)
    assert (q_chunk, k_chunk) in ((128, 128), (64, 64), (32, 32))
    assert slab % q_chunk == slab % k_chunk == 0


@pytest.mark.parametrize("chunk_size,capacity", [(3328, 262912), (6656, 266240), (8192, 262144), (9984, 269568)])
@pytest.mark.parametrize("configured_length", [None, "262144"])
def test_demo_allocates_whole_chunk_model_capacity(monkeypatch, chunk_size, capacity, configured_length):
    if configured_length is None:
        monkeypatch.delenv("GEMMA4_MAX_SEQ_LEN", raising=False)
    else:
        monkeypatch.setenv("GEMMA4_MAX_SEQ_LEN", configured_length)
    config = SimpleNamespace(num_hidden_layers=60)
    monkeypatch.setattr(demo.Gemma4ModelArgs, "load_hf_config", lambda _: config)
    monkeypatch.setattr(demo.Gemma4ModelArgs, "from_hf_config", lambda _: config)
    calls = []

    def create_model(**kwargs):
        calls.append(kwargs)
        return config, "model", "cache", None

    monkeypatch.setattr(demo, "create_tt_model", create_model)
    demo._build_prefill_model(SimpleNamespace(cp_degree=8, tp_degree=4), "unused", chunk_size, 262144)
    assert calls[0]["max_seq_len"] == capacity
    assert calls[0]["prefill_chunk_size"] == chunk_size
