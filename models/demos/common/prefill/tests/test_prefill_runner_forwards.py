# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host-only: the runner's request loop over packed forwards, with the socket and runtime faked."""

from types import SimpleNamespace

import pytest

from models.demos.common.prefill.prefill_chunk import (
    SHUTDOWN_WORD,
    PrefillChunk,
    forward_metadata_words,
    pad_chunks,
    sentinel_words,
)
from models.demos.common.prefill.runners import prefill_runner as runner

C, SP, F = 1024, 8, 4


class _Tensor:
    def is_allocated(self):
        return True

    def device(self):
        return None


class _PackedRuntime:
    def __init__(self):
        self.config = SimpleNamespace(is_first_rank=True, is_last_rank=True, use_trace=False)
        self.mesh_device = None
        self.calls = []

    def prefill_forward(self, inp, kv, chunks, request_id=0, d2h_service=None, *, metadata_msg=None):
        self.calls.append((request_id, list(chunks)))


def _configure(monkeypatch, *, supports_packed_forward):
    monkeypatch.setattr(runner, "FORWARD_CHUNKS", F)
    monkeypatch.setattr(runner, "CHUNK_SIZE", C)
    monkeypatch.setattr(runner, "NUM_USERS", 8)
    monkeypatch.setattr(runner, "MAX_SEQ_LEN", 16 * C)
    monkeypatch.setattr(runner, "MTP_LEVELS", 0)
    monkeypatch.setattr(runner, "ADAPTER", SimpleNamespace(supports_packed_forward=supports_packed_forward))
    monkeypatch.setattr(runner, "PER_CHUNK_FORWARDS", F > 1 and not supports_packed_forward)
    monkeypatch.setattr(runner.ttnn, "deallocate", lambda t: None)
    monkeypatch.setattr(runner.ttnn, "synchronize_device", lambda d: None)


@pytest.fixture
def loop(monkeypatch):
    def run(runtime, forwards, *, supports_packed_forward, num_ranks=1, d2d_out=None):
        _configure(monkeypatch, supports_packed_forward=supports_packed_forward)
        words = [forward_metadata_words(pad_chunks(f, F), F) for f in forwards]
        words.append(sentinel_words(SHUTDOWN_WORD, F))
        it = iter(words)

        def socket_next(service, sockets, n_mtp=0):
            return _Tensor(), None, runner._meta_from_words(next(it))

        monkeypatch.setattr(runner, "_socket_next", socket_next)
        runner.run_request_loop(
            runtime,
            None,
            0,
            num_ranks,
            d2d_rows=0,
            d2d_width=0,
            h2d_service=object(),
            d2d_out=d2d_out,
            sockets=runner.SocketBuffers(metadata=None),
        )

    return run


FORWARDS = [
    [PrefillChunk(0, 0, C), PrefillChunk(0, C, 2 * C), PrefillChunk(1, 0, 500)],
    [PrefillChunk(2, 0, C)],
    [PrefillChunk(3, 0, C), PrefillChunk(4, 0, C), PrefillChunk(5, 0, C), PrefillChunk(6, 0, 40)],
]


def test_request_id_counts_forwards(loop):
    runtime = _PackedRuntime()
    loop(runtime, FORWARDS, supports_packed_forward=True)
    assert [(rid, [c for c in chunks if not c.is_empty]) for rid, chunks in runtime.calls] == [
        (0, FORWARDS[0]),
        (1, FORWARDS[1]),
        (2, FORWARDS[2]),
    ]


def test_first_rank_refuses_a_bad_forward(loop, expect_error):
    split = [PrefillChunk(0, 0, C), PrefillChunk(1, 0, C), PrefillChunk(0, C, 2 * C)]
    with expect_error(ValueError, "two separate runs"):
        loop(_PackedRuntime(), [split], supports_packed_forward=True)


@pytest.mark.parametrize(
    "is_last_rank, returns, message",
    [(False, None, "returned no activation"), (True, _Tensor(), "the last rank's prefill_chunk returned")],
)
def test_a_wrong_chunk_output_shuts_down_the_next_rank(loop, monkeypatch, expect_error, is_last_rank, returns, message):
    shutdowns = []
    monkeypatch.setattr(runner, "_lease_reclaim", lambda *a: None)
    monkeypatch.setattr(runner, "_forward_shutdown", lambda d2d_out, *a: shutdowns.append(d2d_out))
    monkeypatch.setattr(runner, "_chunk_rows", lambda inp, k, rows: _Tensor())
    monkeypatch.setattr(runner, "_device_record", lambda words, dev: _Tensor())

    class _Wrong(_PerChunkRuntime):
        def prefill_chunk(self, inp, kv, **kwargs):
            super().prefill_chunk(inp, kv, **kwargs)
            return returns

    runtime = _Wrong()
    runtime.config.is_last_rank = is_last_rank
    d2d_out = object()
    with expect_error(AssertionError, message):
        loop(runtime, FORWARDS[:1], supports_packed_forward=False, num_ranks=2, d2d_out=d2d_out)
    assert shutdowns == [d2d_out]
    assert len(runtime.calls) == 1


def test_a_refused_forward_shuts_down_the_next_rank(loop, monkeypatch, expect_error):
    shutdowns = []
    monkeypatch.setattr(runner, "_lease_reclaim", lambda *a: None)
    monkeypatch.setattr(runner, "_forward_shutdown", lambda d2d_out, *a: shutdowns.append(d2d_out))
    runtime = _PackedRuntime()
    runtime.config.is_last_rank = False
    d2d_out = object()
    split = [PrefillChunk(0, 0, C), PrefillChunk(1, 0, C), PrefillChunk(0, C, 2 * C)]
    with expect_error(ValueError, "two separate runs"):
        loop(runtime, [split], supports_packed_forward=True, num_ranks=2, d2d_out=d2d_out)
    assert shutdowns == [d2d_out]
    assert runtime.calls == []


def test_index_endpoints_use_the_chunk_metadata(monkeypatch):
    made = []
    services = SimpleNamespace(
        create_receiver=lambda **kw: made.append(kw["metadata_size_bytes"]),
        create_sender=lambda **kw: made.append(kw["metadata_size_bytes"]),
    )
    monkeypatch.setattr(runner.ttnn, "D2DStreamService", services)
    monkeypatch.setattr(runner.ttnn, "create_mesh_mapper", lambda *a: None)
    monkeypatch.setattr(runner, "indexer_indices_global_spec", lambda *a: None)
    monkeypatch.setattr(runner, "D2D_METADATA_SIZE_BYTES", 16)
    runner.build_d2d_index_endpoints(None, 1, (1, 1, 32, 8), inbound=True, outbound=True)
    assert made == [12, 12]


def _request_ids(forwards):
    return [c * F + k for c, f in enumerate(forwards) for k in range(len(f))]


class _PerChunkRuntime(_PackedRuntime):
    def prefill_chunk(
        self, inp, kv, *, slot_id, actual_start, actual_end, request_id=0, d2h_service=None, metadata_msg=None
    ):
        assert not self.calls or request_id > self.calls[-1][0], "request_id must increase"
        self.calls.append((request_id, PrefillChunk(slot_id, actual_start, actual_end)))


@pytest.mark.parametrize("per_chunk, rank, size", [(True, 0, 12), (True, 1, 12), (False, 0, 56), (False, 1, 60)])
def test_layer_ack_record_size(monkeypatch, per_chunk, rank, size):
    monkeypatch.setattr(runner, "PER_CHUNK_FORWARDS", per_chunk)
    monkeypatch.setattr(runner, "CHUNK_METADATA_SIZE_BYTES", 56)
    monkeypatch.setattr(runner, "D2D_METADATA_SIZE_BYTES", 60)
    assert runner._layer_ack_record_bytes(rank) == size


def test_per_chunk_forwards_run_each_chunk_once(loop, monkeypatch):
    monkeypatch.setattr(runner, "_chunk_rows", lambda inp, k, rows: _Tensor())
    monkeypatch.setattr(runner, "_device_record", lambda words, dev: _Tensor())
    monkeypatch.setattr(runner, "_fill_record", lambda record, words: None)
    runtime = _PerChunkRuntime()
    loop(runtime, FORWARDS, supports_packed_forward=False)
    assert runtime.calls == list(zip(_request_ids(FORWARDS), (c for f in FORWARDS for c in f)))


class _Act:
    dtype = layout = None

    def __init__(self, shape, origin=None):
        self.shape = list(shape)
        self.origin = origin

    def device(self):
        return None

    def is_allocated(self):
        return True


class _D2D:
    def get_backing_tensor(self):
        return _Act([])

    def wait_for_fabric_links(self):
        pass

    def release_fabric_links(self):
        pass


class _EchoRuntime(_PackedRuntime):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(is_first_rank=False, is_last_rank=False, use_trace=False)
        self.records = []

    def prefill_chunk(
        self, inp, kv, *, slot_id, actual_start, actual_end, request_id=0, d2h_service=None, metadata_msg=None
    ):
        self.calls.append((request_id, PrefillChunk(slot_id, actual_start, actual_end)))
        self.records.append(metadata_msg.origin.flatten().tolist())
        return inp


def test_downstream_rank_writes_chunks_into_one_buffer_and_forwards_the_metadata(monkeypatch):
    _configure(monkeypatch, supports_packed_forward=False)
    monkeypatch.setattr(runner, "_sp", SP)
    rows, hidden = C // SP, 64
    zeros, writes, fills, sent = [], [], [], []

    def make_zeros(shape, **kw):
        zeros.append(list(shape))
        return _Act(shape)

    def fill_cache(cache, act, batch_idx, update_idx):
        assert batch_idx == 0
        writes.append((cache, "zero" if act.origin is None else act.origin, update_idx))

    def copy_host_to_device_tensor(src, dst):
        fills.append(dst)
        dst.origin = src.origin

    def outbound(service, activation, metadata):
        sent.append((activation, metadata.origin.flatten().tolist()))

    monkeypatch.setattr(
        runner.ttnn, "slice", lambda t, s, e, memory_config=None: _Act([1, 1, e[2] - s[2], hidden], s[2])
    )
    monkeypatch.setattr(runner.ttnn, "zeros", make_zeros)
    monkeypatch.setattr(runner.ttnn, "fill_cache", fill_cache)
    monkeypatch.setattr(runner.ttnn, "from_torch", lambda t, **kw: _Act(t.shape, t))
    monkeypatch.setattr(runner.ttnn, "copy_host_to_device_tensor", copy_host_to_device_tensor)
    monkeypatch.setattr(runner.ttnn, "create_mesh_mapper", lambda *a: None)
    monkeypatch.setattr(runner.ttnn, "ReplicateTensorToMesh", lambda dev: None)
    monkeypatch.setattr(runner.ttnn.experimental.deepseek_prefill, "outbound_socket_service_sync", outbound)
    words = [forward_metadata_words(pad_chunks(f, F), F) for f in FORWARDS] + [sentinel_words(SHUTDOWN_WORD, F)]
    it = iter(words)
    monkeypatch.setattr(
        runner,
        "_d2d_recv",
        lambda d2d_in, sockets: (_Act([1, 1, F * rows, hidden]), runner._meta_from_words(next(it))),
    )

    runtime = _EchoRuntime()
    sockets = runner.SocketBuffers(metadata=None)
    runner.run_request_loop(
        runtime,
        None,
        1,
        3,
        d2d_rows=0,
        d2d_width=0,
        d2d_in=_D2D(),
        d2d_out=_D2D(),
        sockets=sockets,
    )

    chunks = [c for f in FORWARDS for c in f]
    assert runtime.calls == list(zip(_request_ids(FORWARDS), chunks))
    assert runtime.records == [[c.slot_id, c.actual_start, c.actual_end] for c in chunks]
    assert len(sockets.per_chunk.records) == F and len(fills) == 4
    out = sockets.per_chunk.out
    assert zeros == [[1, 1, rows, hidden], [1, 1, F * rows, hidden]]
    # Chunk k lands at rows k * rows; the 1-chunk forward re-zeroes the two rows forward 0 left behind.
    own = [(out, k * rows, k * rows) for k in range(F)]
    assert writes == own[:3] + own[:1] + [(out, "zero", rows), (out, "zero", 2 * rows)] + own
    assert all(a is out for a, _ in sent[:-1])
    assert [w for _, w in sent] == words
