# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import ttnn

from ...parallel.manager import CCLManager


def _manager() -> CCLManager:
    manager = CCLManager.__new__(CCLManager)  # no device: the ping-pong bookkeeping only
    manager._ping_pong_buffer_cache = {"rs": [["i0", None], ["i1", "o1"]], "ag": ["p", "q"]}
    manager._ping_pong_buffer_indices = {"rs": 1, "ag": 0}
    manager._used_ping_pong_keys = set()
    return manager


def test_free_ping_pong_buffers_frees_only_those_keys(monkeypatch) -> None:
    freed = []
    monkeypatch.setattr(ttnn, "deallocate", freed.append)
    manager = _manager()
    manager.free_ping_pong_buffers(["rs", "rs"])  # the second is already freed: skipped
    assert freed == ["i0", "i1", "o1"]
    assert manager.ping_pong_keys() == {"ag"}
    assert manager._ping_pong_buffer_indices == {"ag": 0}


def test_take_used_ping_pong_keys_returns_each_use_once() -> None:
    manager = _manager()
    assert manager._ping_pong_entry("ag") == ["p", "q"]
    assert manager.take_used_ping_pong_keys() == {"ag"}
    assert manager.take_used_ping_pong_keys() == set()
