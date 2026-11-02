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

C, F = 1024, 4


class _Tensor:
    def is_allocated(self):
        return True


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
