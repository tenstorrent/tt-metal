# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""P2: the chunk-schedule contract of :class:`~...tt.runtime.PrefillRuntime`. Host-only.

The recipe's P2 step is "assertions on ``actual_start`` / ``actual_end`` so an out-of-contract chunk
fails loudly". This file is the proof that they do, and it runs without hardware because a schedule
violation is rejected *before* any device work — which is the whole point of putting the check on
the runtime rather than discovering it as a bad PCC forty minutes later.

The model is a stand-in that records the ``(user_id, cached_len)`` it was called with. That is
exactly the seam worth testing: :class:`~...tt.model.MistralModel` takes ``cached_len`` on faith
and addresses the block-cyclic cache from it, so what matters is that a wrong offset never reaches
it and that a right one arrives unchanged. The numerics of the chunks that *do* get through are
``tests/galaxy_prefill_kv_pcc.py``'s job, on real weights against the golden trace.

``make_chunk_input`` is not exercised here: it pushes a tensor to a mesh, so it has no host-only
behavior to test. It is covered on device by the P2 run.
"""

import pytest

from models.demos.mistral_medium_3_5_128b.tt.runtime import PrefillRuntime, RuntimeConfig

CHUNK = 512
DEPTH = 2


class FakeModel:
    """The attributes the runtime reads, and a ``__call__`` that records the offsets it was given."""

    def __init__(self, chunk_size=CHUNK, num_layers=DEPTH):
        self.chunk_size = chunk_size
        self.num_layers = num_layers
        self.mesh_device = "mesh"
        self.mesh_config = "mesh_config"
        self.calls = []
        self.returned = []

    def __call__(self, tokens, *, kv_cache, user_id, cached_len, want_logits):
        assert want_logits is False, "the prefill runtime has no use for logits"
        self.calls.append((user_id, cached_len))
        self.returned.append(FakeActivation())
        return self.returned[-1]


class FakeActivation:
    """A device tensor stand-in that knows whether it was freed."""

    def __init__(self):
        self.freed = False

    def deallocate(self, force=False):
        self.freed = True


@pytest.fixture
def runtime():
    return PrefillRuntime(FakeModel(), RuntimeConfig(chunk_size=CHUNK, max_seq_len=4 * CHUNK, num_layers=DEPTH))


def test_a_full_sequence_of_chunks_passes_its_offsets_through(runtime):
    """The happy path: four chunks in order, each handed the model the offset it should write at."""
    for i in range(4):
        out = runtime.prefill_chunk(
            object(), None, slot_id=0, actual_start=i * CHUNK, actual_end=(i + 1) * CHUNK, request_id=i
        )
        assert out is None, "the last rank returns None — the populated cache is the output"
    assert runtime.model.calls == [(0, 0), (0, CHUNK), (0, 2 * CHUNK), (0, 3 * CHUNK)]
    assert runtime.filled == {0: 4 * CHUNK}


def test_the_hidden_state_is_freed_on_the_last_rank(runtime):
    """Nobody reads a prefill chunk's output activation, so the runtime must not leak it.

    At 10240 tokens x 12288 hidden that is ~250 MB per chunk on a mesh with ~1.4 GiB spare after
    the weights — a leak here is an out-of-memory failure two chunks later, attributed to whatever
    allocated next.
    """
    assert runtime.prefill_chunk(object(), None, slot_id=0, actual_start=0, actual_end=CHUNK) is None
    assert runtime.model.returned[-1].freed, "the last rank's chunk output was returned to nobody and not freed"

    # A non-last rank hands the activation on over the D2D socket, so it must NOT be freed here.
    onward = PrefillRuntime(
        FakeModel(), RuntimeConfig(chunk_size=CHUNK, max_seq_len=4 * CHUNK, num_layers=DEPTH, is_last_rank=False)
    )
    out = onward.prefill_chunk(object(), None, slot_id=0, actual_start=0, actual_end=CHUNK)
    assert out is onward.model.returned[-1] and not out.freed


def test_an_unaligned_start_is_rejected(runtime):
    """A start that is not a whole number of chunks in makes the per-chip cache write non-contiguous.

    The cache would still accept it and the read-back permutation would still invert *something* —
    this is the failure that looks like a numerics bug.
    """
    with pytest.raises(AssertionError, match="multiple of chunk_size"):  # allow-pytest.raises: host-side
        runtime.prefill_chunk(object(), None, slot_id=0, actual_start=CHUNK // 2, actual_end=CHUNK)
    assert runtime.model.calls == [], "the model must not have been called"


def test_a_gap_and_a_repeat_are_both_rejected(runtime):
    """Chunk N attends the prefix chunks 0..N-1 in the cache, so the order is not advisory."""
    runtime.prefill_chunk(object(), None, slot_id=0, actual_start=0, actual_end=CHUNK)
    with pytest.raises(AssertionError, match="in order and without gaps"):  # allow-pytest.raises: host-side
        runtime.prefill_chunk(object(), None, slot_id=0, actual_start=2 * CHUNK, actual_end=3 * CHUNK)
    with pytest.raises(AssertionError, match="in order and without gaps"):  # allow-pytest.raises: host-side
        runtime.prefill_chunk(object(), None, slot_id=0, actual_start=0, actual_end=CHUNK)
    assert runtime.model.calls == [(0, 0)]


def test_slots_are_tracked_independently(runtime):
    """Two users interleaved: each slot's schedule is its own, and neither sees the other's offset."""
    rt = PrefillRuntime(
        FakeModel(), RuntimeConfig(chunk_size=CHUNK, max_seq_len=4 * CHUNK, num_layers=DEPTH, num_users=2)
    )
    rt.prefill_chunk(object(), None, slot_id=0, actual_start=0, actual_end=CHUNK)
    rt.prefill_chunk(object(), None, slot_id=1, actual_start=0, actual_end=CHUNK)
    rt.prefill_chunk(object(), None, slot_id=0, actual_start=CHUNK, actual_end=2 * CHUNK)
    assert rt.model.calls == [(0, 0), (1, 0), (0, CHUNK)]
    assert rt.filled == {0: 2 * CHUNK, 1: CHUNK}
    with pytest.raises(AssertionError, match="outside the 2 allocated user slots"):  # allow-pytest.raises: host-side
        rt.prefill_chunk(object(), None, slot_id=2, actual_start=0, actual_end=CHUNK)


def test_a_padded_tail_chunk_ends_the_sequence(runtime):
    """``actual_end < actual_start + chunk_size`` is legal once, as the last chunk.

    Anything after it would have to start mid-chunk. Rejecting it with *that* reason rather than
    letting the alignment assertion fire is the difference between a caller fixing its padding and a
    caller wondering why an aligned-looking offset was refused.
    """
    runtime.prefill_chunk(object(), None, slot_id=0, actual_start=0, actual_end=CHUNK)
    runtime.prefill_chunk(object(), None, slot_id=0, actual_start=CHUNK, actual_end=CHUNK + 100)
    assert runtime.filled == {0: CHUNK + 100}, "filled reports real tokens, not the padded chunk"
    with pytest.raises(AssertionError, match="ended the sequence"):  # allow-pytest.raises: host-side
        runtime.prefill_chunk(object(), None, slot_id=0, actual_start=CHUNK + 100, actual_end=CHUNK + 200)
    runtime.reset_slot(0)
    runtime.prefill_chunk(object(), None, slot_id=0, actual_start=0, actual_end=CHUNK)


def test_an_over_long_or_empty_chunk_is_rejected(runtime):
    with pytest.raises(AssertionError, match="longer than"):  # allow-pytest.raises: host-side
        runtime.prefill_chunk(object(), None, slot_id=0, actual_start=0, actual_end=CHUNK + 1)
    with pytest.raises(AssertionError, match="empty or negative"):  # allow-pytest.raises: host-side
        runtime.prefill_chunk(object(), None, slot_id=0, actual_start=0, actual_end=0)
    with pytest.raises(AssertionError, match="empty or negative"):  # allow-pytest.raises: host-side
        runtime.prefill_chunk(object(), None, slot_id=0, actual_start=CHUNK, actual_end=0)


def test_running_past_the_cache_capacity_is_rejected(runtime):
    """The cache holds ``max_seq_len``; the block-cyclic write would wrap onto chunk 0's rows."""
    for i in range(4):
        runtime.prefill_chunk(object(), None, slot_id=0, actual_start=i * CHUNK, actual_end=(i + 1) * CHUNK)
    with pytest.raises(AssertionError, match="past the cache's"):  # allow-pytest.raises: host-side
        runtime.prefill_chunk(object(), None, slot_id=0, actual_start=4 * CHUNK, actual_end=5 * CHUNK)


def test_the_model_and_the_runtime_must_agree_on_the_period(runtime):
    """``MistralModel.chunk_size`` *is* the cache's addressing period; two values is two layouts."""
    with pytest.raises(AssertionError, match="must equal the runtime's"):  # allow-pytest.raises: host-side
        PrefillRuntime(FakeModel(chunk_size=256), RuntimeConfig(chunk_size=CHUNK, max_seq_len=CHUNK, num_layers=DEPTH))
    with pytest.raises(AssertionError, match="layers, runtime config says"):  # allow-pytest.raises: host-side
        PrefillRuntime(FakeModel(num_layers=1), RuntimeConfig(chunk_size=CHUNK, max_seq_len=CHUNK, num_layers=DEPTH))


def test_the_config_validates_itself():
    with pytest.raises(AssertionError, match="multiple of the tile height"):  # allow-pytest.raises: host-side
        RuntimeConfig(chunk_size=100, max_seq_len=1000, num_layers=1)
    with pytest.raises(AssertionError, match="below one chunk"):  # allow-pytest.raises: host-side
        RuntimeConfig(chunk_size=CHUNK, max_seq_len=CHUNK // 2, num_layers=1)


def test_num_chunks_counts_a_padded_tail(runtime):
    assert runtime.num_chunks(4 * CHUNK) == 4
    assert runtime.num_chunks(4 * CHUNK - 1) == 4
    assert runtime.num_chunks(1) == 1
