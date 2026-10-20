# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host-only: the runner's request loop over packed forwards, with the socket and runtime faked."""

from types import SimpleNamespace

import pytest

from models.demos.common.prefill.forward_entries import (
    SHUTDOWN_WORD,
    PrefillEntry,
    forward_metadata_words,
    pad_entries,
    sentinel_words,
)
from models.demos.common.prefill.runners import prefill_runner as runner

C, SP, F = 1024, 8, 4


class _Tensor:
    def is_allocated(self):
        return True


class _PackedRuntime:
    def __init__(self):
        self.config = SimpleNamespace(is_first_rank=True, is_last_rank=True, use_trace=False)
        self.mesh_device = None
        self.calls = []

    def prefill_forward(self, inp, kv, entries, request_id=0, d2h_service=None, *, metadata_msg=None):
        self.calls.append((request_id, list(entries)))


@pytest.fixture
def loop(monkeypatch):
    def run(runtime, forwards, *, supports_packed_forward):
        monkeypatch.setattr(runner, "FORWARD_CHUNKS", F)
        monkeypatch.setattr(runner, "CHUNK_SIZE", C)
        monkeypatch.setattr(runner, "CHUNK_METADATA_SIZE_BYTES", 200)
        monkeypatch.setattr(runner, "_sp", SP)
        monkeypatch.setattr(runner, "NUM_USERS", 8)
        monkeypatch.setattr(runner, "MAX_SEQ_LEN", 16 * C)
        monkeypatch.setattr(runner, "MTP_LEVELS", 0)
        monkeypatch.setattr(runner, "ADAPTER", SimpleNamespace(supports_packed_forward=supports_packed_forward))
        monkeypatch.setattr(runner.ttnn, "deallocate", lambda t: None)
        monkeypatch.setattr(runner.ttnn, "synchronize_device", lambda d: None)
        words = [forward_metadata_words(pad_entries(f, F)) for f in forwards]
        words.append(sentinel_words(SHUTDOWN_WORD, F))
        it = iter(words)

        def socket_next(service, n_mtp=0):
            return _Tensor(), None, runner._meta_from_words(next(it)), _Tensor()

        monkeypatch.setattr(runner, "_socket_next", socket_next)
        runner.run_request_loop(runtime, None, 0, 1, d2d_rows=0, d2d_width=0, h2d_service=object())

    return run


FORWARDS = [
    [PrefillEntry(0, 0, C), PrefillEntry(0, C, 2 * C), PrefillEntry(1, 0, 500)],
    [PrefillEntry(2, 0, C)],
    [PrefillEntry(3, 0, C), PrefillEntry(4, 0, C), PrefillEntry(5, 0, C), PrefillEntry(6, 0, 40)],
]


def test_request_id_counts_chunks(loop):
    runtime = _PackedRuntime()
    loop(runtime, FORWARDS, supports_packed_forward=True)
    assert [(rid, [e for e in entries if not e.is_empty]) for rid, entries in runtime.calls] == [
        (0, FORWARDS[0]),
        (3, FORWARDS[1]),
        (4, FORWARDS[2]),
    ]
