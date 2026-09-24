# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host-only: the runtime refuses out-of-contract chunk ranges loudly (no device needed)."""

from types import SimpleNamespace

import pytest

from models.demos.qwen_3_8_27b.tt.runtime import TtPrefillRuntime

CHUNK, MAX, CAP = 5120, 262144, 266240


class _StubModel:
    def __init__(self):
        self.mc = SimpleNamespace(sp=8)
        self.embedding = SimpleNamespace(make_tokens=lambda t: t)
        self.calls = []

    def forward(self, x, ctx):
        self.calls.append((ctx.start, ctx.valid_end))
        return None


class _StubCaches:
    def reset_gdn(self, user_id=None):
        pass


@pytest.fixture
def rt():
    return TtPrefillRuntime(_StubModel(), chunk_size=CHUNK, max_seq_len=MAX, capacity=CAP, num_users=2)


def test_in_contract_chunks_run(rt):
    c = _StubCaches()
    rt.prefill_chunk(None, c, 0, 0, CHUNK)
    rt.prefill_chunk(None, c, 0, CHUNK, 2 * CHUNK)
    rt.prefill_chunk(None, c, 1, 0, 1024)  # padded final chunk for another user
    assert rt.model.calls == [(0, CHUNK), (CHUNK, 2 * CHUNK), (0, 1024)]


@pytest.mark.parametrize(
    "slot,start,end,message",
    [
        (2, 0, CHUNK, "out of range"),  # slot out of range
        (-1, 0, CHUNK, "out of range"),
        (0, 32, CHUNK, "not a multiple of chunk_size"),  # start not chunk aligned
        (0, 0, CHUNK + 32, "chunk range"),  # end past start + chunk
        (0, 0, 0, "chunk range"),  # empty
        (0, 0, 100, "tile aligned"),  # not tile aligned
        (0, CAP, CAP + CHUNK, "past the KV cache capacity"),  # past capacity
    ],
)
def test_out_of_contract_chunks_assert(rt, expect_error, slot, start, end, message):
    with expect_error(AssertionError, message):
        rt.prefill_chunk(None, _StubCaches(), slot, start, end)


def test_gdn_chunks_must_be_sequential(rt, expect_error):
    c = _StubCaches()
    with expect_error(AssertionError, "GDN state is sequential"):
        rt.prefill_chunk(None, c, 0, CHUNK, 2 * CHUNK)  # skipped chunk 0
    rt.prefill_chunk(None, c, 0, 0, 1024)  # padded => sequence ended
    with expect_error(AssertionError, "GDN state is sequential"):
        rt.prefill_chunk(None, c, 0, CHUNK, 2 * CHUNK)


def test_make_chunk_input_requires_exact_chunk(rt, expect_error):
    with expect_error(AssertionError, "chunk input must be exactly chunk_size"):
        rt.make_chunk_input(list(range(CHUNK - 1)))
