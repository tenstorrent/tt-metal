# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The Q chunk of the chunked full-attention prefill SDPA depends on the number of query rows, and never changes a result."""

from types import SimpleNamespace

import pytest
import torch

import ttnn
from models.demos.gemma4_31b_qb2.tt.decoder import Decoder
from models.demos.gemma4_31b_qb2.tt.generator import build_generator


def full_layer():
    layer = object.__new__(Decoder)
    layer.sliding = False
    layer.pref_compute = SimpleNamespace(fp32_dest_acc_en=True)
    layer.pref_sdpa_by_q = {q: SimpleNamespace(q_chunk_size=q) for q in (32, 64, 128, 256)}
    layer.pref_sdpa_selections = {}
    return layer


def chosen(layer, rows, start=8192):
    q = SimpleNamespace(shape=(1, 8, rows, 512), dtype=ttnn.bfloat8_b)
    cache = SimpleNamespace(dtype=ttnn.bfloat8_b)
    page_table = SimpleNamespace(buffer_aligned_page_size=lambda: 160)
    return layer._prefill_sdpa_config(q, cache, page_table, start=start).q_chunk_size


@pytest.mark.parametrize("rows", [256, 512, 768, 1024])
def test_short_suffixes_with_an_even_number_of_128_row_chunks_use_q_chunk_64(rows):
    assert Decoder._short_suffix_q_chunk(rows) == 64
    assert chosen(full_layer(), rows) == 64


@pytest.mark.parametrize("rows", [128, 384, 640, 896, 1280, 1536, 2048, 4096, 6656, 8192])
def test_other_sizes_keep_the_default_choice(rows, monkeypatch):
    assert Decoder._short_suffix_q_chunk(rows) is None
    layer = full_layer()
    assert chosen(layer, rows) == chosen(layer, rows)  # deterministic
    assert chosen(layer, rows) in (128, 256)
    assert chosen(layer, rows) != 64


def test_the_choice_never_depends_on_the_start_and_sliding_layers_are_unchanged():
    layer = full_layer()
    assert {chosen(layer, 512, start=s) for s in (128, 8192, 49152)} == {64}
    sliding = full_layer()
    sliding.sliding = True
    sliding.pref_sdpa_by_q = {
        q: SimpleNamespace(q_chunk_size=q) for q in (32, 64, 128)
    }  # sliding layers offer up to 128
    assert chosen(sliding, 512) == 128


@pytest.fixture
def generator(qb2_mesh):
    gen = build_generator(qb2_mesh, layer_indices=[0, 5], max_seq_len=16384)
    yield gen
    gen.close()


@pytest.mark.parametrize("rows", [256, 384, 512, 768, 1024, 2048])
def test_every_q_chunk_choice_gives_bit_identical_outputs(generator, monkeypatch, rows):
    """The rule against the default choice (largest chunk that fits L1), same inputs, continuation after 8K cached tokens."""
    gen = generator
    state = gen.allocate_cache(batch=1, capacity=16384)
    tokens = torch.randint(1000, 50000, (1, 16384), generator=torch.Generator().manual_seed(3))
    key = iter(range(1, 100))

    def run():
        k = next(key)
        gen.prefill_forward(tokens, page_table=state.page_tables, kv_cache=state, prompt_lens=[8192], history_keys=[k])
        return gen.prefill_forward(
            tokens,
            page_table=state.page_tables,
            kv_cache=state,
            prompt_lens=[8192 + rows],
            start_positions=[8192],
            history_keys=[k],
        )

    with_rule = run()
    monkeypatch.setattr(Decoder, "_short_suffix_q_chunk", staticmethod(lambda rows: None))
    default = run()
    assert torch.equal(with_rule, default)
