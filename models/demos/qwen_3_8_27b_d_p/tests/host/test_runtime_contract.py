# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""P2: the runtime's chunk-range contract. Host only — every assertion fires before the model is
touched, so a stub model is enough and no device is needed.

The interesting one is the last test. For a pure attention model a padded final chunk is inert:
causality means no real token attends the pad. Three quarters of this model is a *recurrent*
mixer, where the pad tokens are folded into the carried state and every later chunk inherits
them. So a short chunk is accepted only as the final one, and the runtime refuses the next call
rather than quietly producing a wrong state.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from models.demos.qwen_3_8_27b_d_p.config import MeshConfig
from models.demos.qwen_3_8_27b_d_p.reference.config import Qwen35TextConfig
from models.demos.qwen_3_8_27b_d_p.tt.tt_prefill_runtime import PrefillRuntimeConfig, TtPrefillRuntime

CHUNK = 512
MAX_SEQ = 2048


class _StubModel:
    """Records the chunks it is asked for; the runtime's assertions run before any of this."""

    def __init__(self) -> None:
        self.calls: list[tuple[int, int]] = []

    def prefill_chunk(self, ids, *, start_pos, caches, user_id, skip_lm_head, on_layer_complete):
        self.calls.append((start_pos, ids.numel()))
        return None


@pytest.fixture
def runtime():
    cfg = Qwen35TextConfig.from_json()
    mesh_config = MeshConfig((8, 4), tp=4)
    model = _StubModel()
    rt = TtPrefillRuntime(
        mesh_device=None,
        model=model,
        cfg=cfg,
        mesh_config=mesh_config,
        config=PrefillRuntimeConfig(chunk_size=CHUNK, max_seq_len=MAX_SEQ, num_users=2),
    )
    return rt, model


@pytest.fixture
def caches():
    return SimpleNamespace(kv=None, gdn={})


def test_chunk_size_must_be_block_cyclic_aligned(expect_error):
    """``chunk_size % (32 * sp)`` is the KV table's addressing period. A misaligned value corrupts
    addresses silently, so the runtime refuses to be constructed with one."""
    mesh_config = MeshConfig((8, 4), tp=4)
    PrefillRuntimeConfig(chunk_size=512, max_seq_len=2048).validate(mesh_config)
    with expect_error(AssertionError, "block-cyclic"):
        PrefillRuntimeConfig(chunk_size=500, max_seq_len=2048).validate(mesh_config)
    with expect_error(AssertionError, "whole number"):
        PrefillRuntimeConfig(chunk_size=512, max_seq_len=2304).validate(mesh_config)


def test_make_chunk_input_pads_to_chunk_size(runtime, expect_error):
    rt, _model = runtime
    assert rt.make_chunk_input(list(range(CHUNK))).numel() == CHUNK
    padded = rt.make_chunk_input(list(range(10)))
    assert padded.numel() == CHUNK and int(padded[0, -1]) == 0
    with expect_error(AssertionError, "longer than chunk_size"):
        rt.make_chunk_input(list(range(CHUNK + 1)))


@pytest.mark.parametrize(
    "start, end, match",
    [
        (1, CHUNK, "not a multiple of chunk_size"),
        (MAX_SEQ, MAX_SEQ + CHUNK, "runs past the per-user cache"),
        (0, 0, "not within one"),
        (0, CHUNK + 1, "not within one"),
        (CHUNK, CHUNK - 1, "not within one"),
    ],
)
def test_out_of_contract_chunk_ranges_fail_loudly(runtime, caches, start, end, match, expect_error):
    rt, _model = runtime
    with expect_error(AssertionError, match):
        rt.prefill_chunk(torch.zeros(CHUNK, dtype=torch.int64), caches, actual_start=start, actual_end=end)


def test_user_id_is_range_checked(runtime, caches, expect_error):
    rt, _model = runtime
    with expect_error(AssertionError, "user_id"):
        rt.prefill_chunk(torch.zeros(CHUNK, dtype=torch.int64), caches, user_id=2, actual_start=0, actual_end=CHUNK)


def test_in_contract_chunks_are_accepted_in_order(runtime, caches):
    rt, model = runtime
    for start in range(0, MAX_SEQ, CHUNK):
        rt.prefill_chunk(torch.zeros(CHUNK, dtype=torch.int64), caches, actual_start=start, actual_end=start + CHUNK)
    assert model.calls == [(s, CHUNK) for s in range(0, MAX_SEQ, CHUNK)]


def test_a_padded_chunk_must_be_the_last_one(runtime, caches, expect_error):
    """A short final chunk is legal; a chunk *after* it is not.

    Without this the Gated DeltaNet layers would continue from a recurrent state that absorbed the
    pad tokens — a wrong answer with no shape error and no obviously bad intermediate.
    """
    rt, _model = runtime
    rt.prefill_chunk(torch.zeros(CHUNK, dtype=torch.int64), caches, actual_start=0, actual_end=CHUNK)
    rt.prefill_chunk(torch.zeros(300, dtype=torch.int64), caches, actual_start=CHUNK, actual_end=CHUNK + 300)
    with expect_error(AssertionError, "was short"):
        rt.prefill_chunk(torch.zeros(CHUNK, dtype=torch.int64), caches, actual_start=2 * CHUNK, actual_end=3 * CHUNK)


def test_prefill_sequence_splits_a_prompt_into_chunks(runtime, caches):
    rt, model = runtime
    rt.prefill_sequence(torch.zeros(CHUNK * 3, dtype=torch.int64), caches)
    assert [c[0] for c in model.calls] == [0, CHUNK, 2 * CHUNK]
    assert all(c[1] == CHUNK for c in model.calls)


def test_prefill_sequence_rejects_a_prompt_longer_than_the_cache(runtime, caches, expect_error):
    rt, _model = runtime
    with expect_error(AssertionError, "exceeds max_seq_len"):
        rt.prefill_sequence(torch.zeros(MAX_SEQ + CHUNK, dtype=torch.int64), caches)
