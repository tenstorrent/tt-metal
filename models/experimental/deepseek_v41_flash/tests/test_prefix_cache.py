# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""The decode prefix cache, host only: a token prefix loads on any session, and a replay stops to record one."""

from collections import deque
from types import SimpleNamespace

from models.experimental.deepseek_v4_flash.demo.server import _PrefixCache, _Scheduler, _Turn


class _Snap:
    def __init__(self, pos, nbytes=8):
        self.pos = pos
        self.nbytes = nbytes


def test_lookup_takes_the_longest_prefix_and_leaves_the_last_token():
    cache = _PrefixCache(1 << 20)
    short, long, whole = _Snap(2), _Snap(4), _Snap(5)
    assert cache.store([1, 2], short)
    assert cache.store([1, 2, 3, 4], long)
    assert cache.store([1, 2, 3, 4, 5], whole)
    # An identical prompt still replays its last token, so the stored length is not a hit.
    assert cache.lookup([1, 2, 3, 4, 5], vocab=100) == (4, long)
    # A longer prompt reuses the whole stored prompt.
    assert cache.lookup([1, 2, 3, 4, 5, 9], vocab=100) == (5, whole)
    assert cache.lookup([1, 2, 3, 4, 5], vocab=0) is None
    assert cache.lookup([9, 9, 9], vocab=100) is None


def test_lookup_stops_before_an_image_token():
    cache = _PrefixCache(1 << 20)
    text, image = _Snap(3), _Snap(4)
    cache.store([1, 2, 3], text)
    cache.store([1, 2, 3, 100], image)
    assert cache.lookup([1, 2, 3, 100, 4], vocab=100) == (3, text)


def test_store_drops_the_least_recently_used_past_the_budget():
    cache = _PrefixCache(10)
    assert cache.store([1], _Snap(1, nbytes=6))
    assert cache.store([1, 2], _Snap(2, nbytes=6))
    assert len(cache) == 1
    assert cache.lookup([1, 2, 3], vocab=100)[0] == 2
    assert not cache.store([7], _Snap(1, nbytes=11))
    assert len(cache) == 1


def test_decoded_prefix_hit_imports_and_replays_the_tail():
    cache = _PrefixCache(1 << 20)
    snap = _Snap(4)
    cache.store([10, 11, 12, 13], snap)
    loaded = []

    class Model:
        class config:
            vocab_size = 1000

        def import_session(self, sid, snapshot):
            loaded.append((sid, snapshot))

        def reset_session(self, sid):
            loaded.append(("reset", sid))

    user = SimpleNamespace(pos=0, sid=3, tokens=[], index=0)
    turn = _Turn("alice", 0, {"messages": [{"role": "user", "content": "hi"}]}, None, 8)
    turn.ids = [10, 11, 12, 13, 14, 15]
    scheduler = _Scheduler.__new__(_Scheduler)
    scheduler.server = SimpleNamespace(prefix_cache=cache, prefiller=None)
    scheduler.engine = SimpleNamespace(model=Model(), users=[user])

    assert scheduler._share_prefix(user, turn)
    assert loaded == [(3, snap)]
    assert user.pos == 4 and user.tokens == [10, 11, 12, 13]
    assert turn.ids == [14, 15]
    assert turn.from_prefix_cache and turn.resumed_from == 4


def test_failed_import_replays_the_prompt_from_zero():
    cache = _PrefixCache(1 << 20)
    cache.store([1, 2, 3], _Snap(3))

    class Model:
        class config:
            vocab_size = 100

        def import_session(self, sid, snapshot):
            raise RuntimeError("pools full")

        def reset_session(self, sid):
            self.reset = sid

    model = Model()
    user = SimpleNamespace(pos=0, sid=1, tokens=[], index=0)
    turn = _Turn("bob", 0, {"messages": [{"role": "user", "content": "hi"}]}, None, 4)
    turn.ids = [1, 2, 3, 4]
    scheduler = _Scheduler.__new__(_Scheduler)
    scheduler.server = SimpleNamespace(prefix_cache=cache, prefiller=None)
    scheduler.engine = SimpleNamespace(model=model, users=[user])

    assert not scheduler._share_prefix(user, turn)
    assert model.reset == 1
    assert user.pos == 0 and turn.ids == [1, 2, 3, 4]


def test_replay_stops_on_a_prefix_and_records_it_once_idle():
    cache = _PrefixCache(1 << 20)
    exported = []

    class Model:
        class config:
            vocab_size = 100

        def export_session(self, sid):
            exported.append(sid)
            return _Snap(user.pos)

    user = SimpleNamespace(pos=0, sid=7, tokens=[], index=0, messages=[], _next_render=0, pending_id=None)
    turn = _Turn(
        "carol",
        0,
        {"messages": [{"role": "system", "content": "be brief"}, {"role": "user", "content": "hi"}]},
        None,
        4,
    )
    turn.ids = [1, 2, 3, 4, 5]
    scheduler = _Scheduler.__new__(_Scheduler)
    scheduler._posted = deque()
    scheduler._inflight = deque()
    scheduler.server = SimpleNamespace(
        prefix_cache=cache,
        prefiller=None,
        _render_ids=lambda user, include_assistant, closing: [1, 2, 3],
    )
    scheduler.engine = SimpleNamespace(model=Model(), users=[user])

    scheduler._remember_prefix(user, turn)
    assert turn.prefix_cuts == {3, 4}  # history ends at 3; the last prompt token stays for decode
    assert scheduler._prefix_batch(turn, user, 16) == 3

    user.pos = 3
    user.tokens = [1, 2, 3]
    scheduler._inflight.append(turn)
    assert scheduler._capture_due(turn, user)
    assert exported == []

    scheduler._inflight.clear()
    assert not scheduler._capture_due(turn, user)
    assert exported == [7]
    assert cache.lookup([1, 2, 3, 4, 5], vocab=100)[0] == 3
    assert 3 not in turn.prefix_cuts
