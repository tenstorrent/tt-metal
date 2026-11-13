# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for device-RAM page-table parking, cache hit rate, and TTFT (device hit vs host hit)."""

from __future__ import annotations

import threading
import time
from types import SimpleNamespace

from models.experimental.deepseek_v4_flash.tt.decode.paged_cache import (
    ZERO_BLOCK,
    PagedGroup,
    PagedKVManager,
)
from models.experimental.deepseek_v41_flash.demo.server import (
    _DeviceParkedConversation,
    _SlotReuse,
    _V41SlotPool,
)
from models.experimental.deepseek_v4_flash.demo.server import _HostConversation


def test_paged_kv_detach_attach_and_free():
    groups = {
        "g0": PagedGroup(layer_type="g0", block_size=32, sliding_window=32, max_seq=256, compress_rate=2),
    }
    mgr = PagedKVManager(groups, pool_blocks={"g0": 10})

    sid0 = mgr.open_session()
    mgr.ensure_capacity(sid0, pos=128)  # allocates compressed blocks
    free_after_alloc = mgr.pools["g0"].free_blocks
    orig_row = list(mgr._sessions[sid0].rows["g0"])

    # Detach page-table rows: blocks stay allocated in pool, session row tail is cleared.
    detached = mgr.detach_session_rows(sid0)
    assert mgr.pools["g0"].free_blocks == free_after_alloc
    ring_blocks = groups["g0"].ring_blocks
    comp_blocks = groups["g0"].compressed_blocks
    assert mgr._sessions[sid0].rows["g0"][ring_blocks:] == [ZERO_BLOCK] * comp_blocks

    # Re-attach to another session and verify exact restoration.
    sid1 = mgr.open_session()
    free_after_open1 = mgr.pools["g0"].free_blocks
    mgr.attach_session_rows(sid1, detached)
    assert mgr._sessions[sid1].rows["g0"][ring_blocks:] == orig_row[ring_blocks:]

    # Freeing detached rows returns blocks to the pool.
    mgr.detach_session_rows(sid1)
    mgr.free_detached_rows(detached)
    assert mgr.pools["g0"].free_blocks == free_after_open1 + sum(1 for b in orig_row[ring_blocks:] if b != ZERO_BLOCK)


class _FakeModel:
    def __init__(self, device_reload_s: float = 0.002, host_restore_s: float = 0.020):
        self.device_reload_s = device_reload_s
        self.host_restore_s = host_restore_s
        self.config = SimpleNamespace(vocab_size=1000)
        self._session_pos = {0: 64, 1: 64}
        self.freed = []

    def evict_session_to_device_park(self, sid: int):
        self._session_pos[sid] = 0
        return SimpleNamespace(pos=64, rows={"g0": [1, 2]})

    def reload_session_from_device_park(self, sid: int, snapshot):
        self._session_pos[sid] = snapshot.pos
        time.sleep(self.device_reload_s)

    def export_device_park(self, snapshot):
        self.freed.append(snapshot)
        return SimpleNamespace(pos=snapshot.pos, nbytes=1024)

    def free_device_park(self, snapshot):
        self.freed.append(snapshot)

    def reset_session(self, sid: int):
        self._session_pos[sid] = 0

    def import_session(self, sid: int, snapshot):
        self._session_pos[sid] = snapshot.pos
        time.sleep(self.host_restore_s)


def _make_user(sid: int, messages: list[dict], pos: int = 64):
    return SimpleNamespace(
        sid=sid,
        messages=list(messages),
        tokens=list(range(pos)),
        marks=[],
        pending_id=None,
        _next_render=len(messages),
        thinking_mode=None,
        reasoning_effort=None,
        system_prompt=None,
        pos=pos,
    )


class _FakeBaseScheduler:
    def __init__(self):
        pass

    def _offload(self, slot: int, owner: str | None) -> None:
        pool = self.server.pool
        user = self.engine.users[slot]
        pool.store(
            _HostConversation(
                owner=owner,
                messages=list(user.messages),
                tokens=list(user.tokens),
                marks=list(user.marks),
                pending_id=user.pending_id,
                next_render=user._next_render,
                thinking_mode=user.thinking_mode,
                reasoning_effort=user.reasoning_effort,
                system_prompt=user.system_prompt,
                snapshot=SimpleNamespace(pos=user.pos, nbytes=1024),
                stored_at=pool._used_at[slot],
            )
        )

    def _pool_summary(self) -> str:
        return f"tokens_left={self.engine.tokens_left()}"


class _FakeScheduler(_SlotReuse, _FakeBaseScheduler):
    def __init__(self, server, engine):
        self.server = server
        self.engine = engine
        self._cut_short = set()
        super().__init__()

    def _restore(self, user, turn) -> bool:
        if isinstance(turn.restore, _DeviceParkedConversation):
            return super()._restore(user, turn)
        if isinstance(turn.restore, _HostConversation):
            self.engine.model.import_session(user.sid, turn.restore.snapshot)
            user.pos = turn.restore.snapshot.pos
            user.messages = list(turn.restore.messages)
            return True
        return False


def test_device_park_hit_rate_and_ttft():
    model = _FakeModel(device_reload_s=0.002, host_restore_s=0.020)
    conv_a = [{"role": "user", "content": "hello A"}]
    conv_b = [{"role": "user", "content": "hello B"}]
    users = [_make_user(0, conv_a, pos=64), _make_user(1, conv_b, pos=64)]
    for u in users:
        u.reset = lambda u=u: setattr(u, "pos", 0)
    engine = SimpleNamespace(model=model, users=users, tokens_left=lambda: 1000)

    pool = _V41SlotPool(engine, host_users=4, host_bytes=1 << 30, max_running=2, idle_timeout=300.0)
    server = SimpleNamespace(pool=pool, prefiller=None, thinking_mode=None, reasoning_effort=None)
    sched = _FakeScheduler(server, engine)

    # All slots full, idle_timeout not reached, but page table has space: acquire takes oldest slot.
    pool._used_at[0] = time.time() - 5.0
    pool._used_at[1] = time.time() - 1.0
    lease = pool.acquire("user_c", [{"role": "user", "content": "hello C"}])
    assert lease.slot == 0
    sched._offload(lease.slot, lease.previous)
    users[0].messages = [{"role": "user", "content": "hello C"}]
    users[0].pos = 16
    pool.release(0)
    assert len(pool.device_parked) == 1
    assert len(pool.host) == 0

    # Measure device-park cache hit rate and time to first output token (TTFT).
    device_hits = 0
    host_hits = 0
    total_queries = 4
    device_ttft = []
    host_ttft = []

    for idx in range(2):
        lease = pool.acquire("user_a", conv_a + [{"role": "user", "content": f"follow-up {idx}"}])
        assert isinstance(lease.restore, _DeviceParkedConversation)
        device_hits += 1
        t0 = time.perf_counter()
        turn = SimpleNamespace(slot=lease.slot, user_key="user_a", restore=lease.restore, cancelled=threading.Event())
        assert sched._restore(users[lease.slot], turn)
        device_ttft.append(time.perf_counter() - t0)
        pool.release(lease.slot)
        # Re-park for next iteration.
        sched._offload(lease.slot, "user_a")
        users[lease.slot].messages = [{"role": "user", "content": "other user"}]
        users[lease.slot].pos = 16

    # Now simulate host-RAM cache hits for comparison.
    for idx in range(2):
        host_conv = _HostConversation(
            owner="user_h",
            messages=conv_b,
            tokens=list(range(64)),
            marks=[],
            pending_id=None,
            next_render=1,
            thinking_mode=None,
            reasoning_effort=None,
            system_prompt=None,
            snapshot=SimpleNamespace(pos=64, nbytes=1024),
            stored_at=time.time(),
        )
        host_hits += 1
        t0 = time.perf_counter()
        turn = SimpleNamespace(slot=1, user_key="user_h", restore=host_conv, cancelled=threading.Event())
        assert sched._restore(users[1], turn)
        host_ttft.append(time.perf_counter() - t0)

    hit_rate = (device_hits + host_hits) / total_queries
    avg_device_ttft_ms = (sum(device_ttft) / len(device_ttft)) * 1000
    avg_host_ttft_ms = (sum(host_ttft) / len(host_ttft)) * 1000

    assert hit_rate == 1.0
    assert avg_device_ttft_ms < avg_host_ttft_ms
    print(
        f"cache_hit_rate={hit_rate:.2%}, "
        f"device_hit_ttft={avg_device_ttft_ms:.2f}ms, "
        f"host_hit_ttft={avg_host_ttft_ms:.2f}ms"
    )


def test_evict_oldest_to_host_only_when_pool_full():
    model = _FakeModel()
    users = [
        _make_user(0, [{"role": "user", "content": "u0"}], pos=64),
        _make_user(1, [{"role": "user", "content": "u1"}], pos=64),
    ]
    for u in users:
        u.reset = lambda u=u: setattr(u, "pos", 0)
    free_tokens = [0]
    engine = SimpleNamespace(model=model, users=users, tokens_left=lambda: free_tokens[0])

    pool = _V41SlotPool(engine, host_users=4, host_bytes=1 << 30, max_running=2, idle_timeout=0.0)
    pool.owner = ["u0", "u1"]
    pool._used_at = [50.0, 100.0]
    pool.device_parked.append(
        _DeviceParkedConversation(
            owner="u_oldest",
            messages=[{"role": "user", "content": "oldest"}],
            tokens=list(range(64)),
            marks=[],
            pending_id=None,
            next_render=1,
            thinking_mode=None,
            reasoning_effort=None,
            system_prompt=None,
            snapshot=SimpleNamespace(pos=64, rows={"g0": [3]}),
            used_at=10.0,
        )
    )
    server = SimpleNamespace(pool=pool, prefiller=None, thinking_mode=None, reasoning_effort=None)
    sched = _FakeScheduler(server, engine)

    def tokens_left_with_eviction():
        return free_tokens[0]

    # When pool is full (tokens_left == 0), _make_room evicts oldest user (u_oldest) to host RAM first.
    orig_export = model.export_device_park

    def export_and_free(snap):
        free_tokens[0] += 64
        return orig_export(snap)

    model.export_device_park = export_and_free
    sched._make_room(64, keep=1)
    assert len(pool.device_parked) == 0
    assert [c.owner for c in pool.host] == ["u_oldest"]

    # If still short, next oldest parked slot (slot 0 at used_at=50.0, skipping keep=1) goes to host RAM.
    free_tokens[0] = 0
    orig_reset = users[0].reset

    def reset_and_free():
        free_tokens[0] += 64
        orig_reset()

    users[0].reset = reset_and_free
    sched._make_room(64, keep=1)
    assert [c.owner for c in pool.host] == ["u_oldest", "u0"]


def test_parse_args_256_slots_keeps_kv_pool_size():
    from models.experimental.deepseek_v41_flash.demo.server import parse_args

    args = parse_args([])
    assert args.num_users == 256
    assert args.total_context == 40 * 204800
