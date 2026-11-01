# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host-only: the reference producer's forward schedule and H2D payloads, at one chunk per forward (unchanged
pushes and bytes) and packed (every forward meets the order contract)."""

import math
import struct

import numpy as np
import pytest
import torch

from models.demos.common.prefill.prefill_chunk import (
    EMPTY_CHUNK,
    PrefillChunk,
    decode_forward_metadata,
    pad_chunks,
    validate_chunks,
)
from models.demos.common.prefill.runners import prefill_producer as producer

C = 1024
MAX_SEQ_LEN = 8 * C


@pytest.fixture
def env(monkeypatch):
    def set_env(forward_chunks=1, sp=8):
        monkeypatch.setattr(producer, "CHUNK_SIZE", C)
        monkeypatch.setattr(producer, "MAX_SEQ_LEN", MAX_SEQ_LEN)
        monkeypatch.setattr(producer, "GLOBAL_MESH_SHAPE", (sp, 4))
        monkeypatch.setattr(producer, "MTP_LEVELS", 0)
        monkeypatch.setattr(producer, "FORWARD_CHUNKS", forward_chunks)

    return set_env


def make_cfg(**kw):
    base = dict(
        num_users=3,
        chunks_min=1,
        chunks_max=4,
        max_requests=6,
        duration_s=float("inf"),
        p_gap=0.0,
        p_burst=0.0,
        gap_ms=(200.0, 2000.0),
        mid_chunk_end_prob=0.0,
        seed=0,
        verify=False,
        pcc_threshold=0.9,
    )
    base.update(kw)
    return producer.ProducerConfig(**base)


def run(cfg):
    forwards = []

    def push_fn(pushes):
        forwards.append(list(pushes))
        return 0.0

    stats = producer.run_schedule(cfg, push_fn=push_fn, now_fn=lambda: 0.0, sleep_fn=lambda s: None)
    return forwards, stats


# run_schedule pushes (slot, chunk_idx, start, end, actual_isl) recorded before packed forwards existed.
RECORDED = {
    "random_gap_burst_mid_multiturn": (
        dict(
            num_users=3,
            max_requests=6,
            p_gap=0.2,
            p_burst=0.3,
            mid_chunk_end_prob=0.5,
            multi_turn_prob=0.5,
            seed=7,
            interleave="random",
        ),
        [
            (0, 0, 0, 1024, 3072),
            (2, 0, 0, 1024, 1024),
            (2, 0, 0, 1024, 1024),
            (2, 0, 1024, 2048, 2192),
            (2, 1, 2048, 2192, 2192),
            (2, 0, 2176, 3200, 3663),
            (2, 1, 3200, 3663, 3663),
            (1, 0, 0, 1024, 4096),
            (1, 1, 1024, 2048, 4096),
            (1, 2, 2048, 3072, 4096),
            (0, 1, 1024, 2048, 3072),
            (0, 2, 2048, 3072, 3072),
            (1, 3, 3072, 4096, 4096),
        ],
    ),
    "round_robin_burst_mid": (
        dict(
            num_users=2,
            chunks_min=2,
            chunks_max=3,
            max_requests=4,
            p_burst=0.5,
            mid_chunk_end_prob=0.3,
            seed=1,
            interleave="round_robin",
        ),
        [
            (0, 0, 0, 1024, 2048),
            (1, 0, 0, 1024, 1540),
            (1, 1, 1024, 1540, 1540),
            (1, 0, 0, 1024, 2018),
            (0, 1, 1024, 2048, 2048),
            (1, 1, 1024, 2018, 2018),
            (0, 0, 0, 1024, 3072),
            (0, 1, 1024, 2048, 3072),
            (0, 2, 2048, 3072, 3072),
        ],
    ),
    "random_multiturn": (
        dict(
            num_users=4,
            chunks_max=3,
            max_requests=7,
            mid_chunk_end_prob=0.6,
            multi_turn_prob=0.7,
            seed=1234,
            interleave="random",
        ),
        [
            (2, 0, 0, 660, 660),
            (1, 0, 0, 1024, 2362),
            (0, 0, 0, 1024, 1955),
            (1, 1, 1024, 2048, 2362),
            (3, 0, 0, 13, 13),
            (0, 1, 1024, 1955, 1955),
            (1, 2, 2048, 2362, 2362),
            (3, 0, 0, 1024, 2479),
            (3, 1, 1024, 2048, 2479),
            (2, 0, 640, 1664, 2688),
            (2, 1, 1664, 2688, 2688),
            (0, 0, 1952, 2976, 4000),
            (3, 2, 2048, 2479, 2479),
            (0, 1, 2976, 4000, 4000),
        ],
    ),
}


@pytest.mark.parametrize("name", RECORDED)
def test_single_chunk_schedule_unchanged(env, name):
    env(forward_chunks=1)
    kw, expected = RECORDED[name]
    forwards, stats = run(make_cfg(**kw))
    assert all(len(f) == 1 for f in forwards)
    assert [tuple(f[0]) for f in forwards] == expected
    assert stats.total_pushes == stats.total_chunks == len(expected)


@pytest.mark.parametrize("start", [0, 2 * C, 2 * C + 96, 3 * C + 512])
def test_single_chunk_payload_unchanged(env, start):
    env(forward_chunks=1)
    pool = list(range(10, 10 + 5 * C))
    isl = start + C - 100
    rows, mtp, metadata = producer._forward_payload([producer.ChunkPush(2, 0, start, isl, isl)], lambda s: pool)
    assert np.array_equal(rows, producer._h2d_rows(producer._chunk_slice(pool, start, isl), start))
    assert mtp is None
    assert metadata == struct.pack("<3I", 2, start, isl)


def test_single_chunk_shutdown_unchanged(env):
    env(forward_chunks=1)
    rows, mtp, metadata = producer._shutdown_payload()
    assert np.array_equal(rows, producer._h2d_rows(producer._chunk_slice([], 0)))
    assert mtp is None
    assert metadata == struct.pack("<iii", -1, -1, -1)


def test_packed_payload(env):
    forward_chunks, sp = 4, 8
    env(forward_chunks=forward_chunks, sp=sp)
    pools = {0: list(range(100_000, 100_000 + 4 * C)), 1: list(range(200_000, 200_000 + 4 * C))}
    pushes = [
        producer.ChunkPush(1, 2, 2 * C + 96, 3 * C + 96, 3 * C + 500),
        producer.ChunkPush(1, 3, 3 * C + 96, 3 * C + 500, 3 * C + 500),
        producer.ChunkPush(0, 0, 0, C, 2 * C),
    ]
    rows, mtp, metadata = producer._forward_payload(pushes, pools.__getitem__)
    assert mtp is None
    assert rows.shape == (sp, 1, forward_chunks * C // sp)
    chunks = decode_forward_metadata(list(struct.unpack(f"<{len(metadata) // 4}i", metadata)), forward_chunks)
    assert chunks == [PrefillChunk(p.slot_id, p.actual_start, p.actual_end) for p in pushes] + [EMPTY_CHUNK]
    s = C // sp
    for k, p in enumerate(pushes):
        chunk = producer._h2d_rows(
            producer._chunk_slice(pools[p.slot_id], p.actual_start, p.actual_isl), p.actual_start
        )
        assert np.array_equal(rows[:, :, k * s : (k + 1) * s], chunk)
    assert np.all(rows[:, :, 3 * s :] == 1)

    rows, _, metadata = producer._shutdown_payload()
    assert rows.shape == (sp, 1, forward_chunks * C // sp)
    assert metadata == struct.pack(f"<{2 + 3 * forward_chunks}i", *[-1] * (2 + 3 * forward_chunks))


def check_packed_run(forwards, stats, cfg):
    f = cfg.forward_chunks
    last = {}  # slot -> its last ChunkPush
    for i, pushes in enumerate(forwards):
        assert 1 <= len(pushes) <= f
        chunks = pad_chunks([PrefillChunk(p.slot_id, p.actual_start, p.actual_end) for p in pushes], f)
        validate_chunks(chunks, chunk_size=C, forward_chunks=f, num_users=cfg.num_users, max_seq_len=MAX_SEQ_LEN)
        for p in pushes:
            prev = last.get(p.slot_id)
            if p.chunk_idx == 0:
                assert (
                    prev is None or prev.actual_end == prev.actual_isl
                ), f"forward {i}: {p} starts before {prev} ended"
                assert (
                    p.actual_start == 0 or p.actual_start == prev.actual_isl // 32 * 32
                ), f"forward {i}: {p} resumes off-tile"
            else:
                assert (p.chunk_idx, p.actual_start) == (
                    prev.chunk_idx + 1,
                    prev.actual_end,
                ), f"forward {i}: {p} after {prev}"
                assert p.actual_isl == prev.actual_isl
            assert p.actual_end == min(p.actual_start + C, p.actual_isl)
            last[p.slot_id] = p
        if len(pushes) < f:
            # Short only when no slot had more: each run ends its request and absent slots are idle for good.
            assert all(p.actual_end == p.actual_isl for p in pushes if p is last[p.slot_id])
            present = {p.slot_id for p in pushes}
            assert all(p.slot_id in present for later in forwards[i + 1 :] for p in later)
    assert {s: fill.real_len for s, fill in stats.resident.items()} == {s: p.actual_end for s, p in last.items()}
    assert stats.total_pushes == len(forwards)
    assert stats.total_chunks == sum(len(p) for p in forwards)


@pytest.mark.parametrize("interleave", ["round_robin", "random"])
@pytest.mark.parametrize("forward_chunks", [2, 4])
def test_packed_schedule(env, forward_chunks, interleave):
    env(forward_chunks=forward_chunks)
    for seed in range(50):
        gen = torch.Generator().manual_seed(seed)
        num_users = int(torch.randint(3, 6, (1,), generator=gen))
        cfg = make_cfg(
            num_users=num_users,
            chunks_max=6,
            max_requests=num_users + int(torch.randint(0, 7, (1,), generator=gen)),
            p_gap=0.1,
            mid_chunk_end_prob=0.5,
            multi_turn_prob=0.5,
            seed=seed,
            interleave=interleave,
            forward_chunks=forward_chunks,
        )
        forwards, stats = run(cfg)
        check_packed_run(forwards, stats, cfg)
        assert stats.completed == cfg.max_requests


@pytest.mark.parametrize("interleave", ["round_robin", "random"])
@pytest.mark.parametrize("forward_chunks", [2, 3, 4, 8])
def test_packed_schedule_fills_forwards(env, forward_chunks, interleave):
    env(forward_chunks=forward_chunks)
    for seed in range(20):
        cfg = make_cfg(
            num_users=4,
            chunks_max=7,
            max_requests=4,
            mid_chunk_end_prob=0.5,
            seed=seed,
            interleave=interleave,
            forward_chunks=forward_chunks,
        )
        forwards, stats = run(cfg)
        check_packed_run(forwards, stats, cfg)
        assert len(forwards) == math.ceil(stats.total_chunks / forward_chunks)
        for pushes in forwards:
            ends = [p for i, p in enumerate(pushes) if i + 1 == len(pushes) or pushes[i + 1].slot_id != p.slot_id]
            partial = [p.actual_end < p.actual_start + C for p in ends]
            assert partial == sorted(partial), f"partial-ending run before a full-ending one: {pushes}"


def test_round_robin_rotates_first_slot(env):
    env(forward_chunks=2)
    cfg = make_cfg(num_users=3, chunks_min=6, chunks_max=6, max_requests=3, interleave="round_robin", forward_chunks=2)
    forwards, _ = run(cfg)
    assert [[p.slot_id for p in f] for f in forwards[:3]] == [[0, 0], [1, 1], [2, 2]]


@pytest.mark.parametrize("interleave", ["round_robin", "random"])
def test_packed_respects_max_requests(env, interleave):
    env(forward_chunks=4)
    cfg = make_cfg(num_users=3, chunks_max=1, max_requests=1, interleave=interleave, forward_chunks=4)
    forwards, stats = run(cfg)
    assert stats.completed == 1
    assert [[p.slot_id for p in f] for f in forwards] == [[0]]


def test_import_config_defers_budget_check(monkeypatch, expect_error):
    for name in ("GLOBAL_MESH_SHAPE", "CHUNK_SIZE", "MAX_SEQ_LEN", "FORWARD_CHUNKS", "CHUNK_METADATA_SIZE_BYTES"):
        monkeypatch.setattr(producer, name, getattr(producer, name))  # restored after the reloads below
    monkeypatch.delenv("PREFILL_CHUNK_SIZE", raising=False)
    monkeypatch.setenv("PREFILL_FORWARD_TOKENS_BUDGET", str(8 * C))
    producer._load_env_config(check_budget=False)  # the import-time call, before a manifest sets the chunk size
    with expect_error(AssertionError, "PREFILL_FORWARD_TOKENS_BUDGET"):
        producer._load_env_config()
    monkeypatch.setenv("PREFILL_CHUNK_SIZE", str(2 * C))
    producer._load_env_config()
    assert producer.FORWARD_CHUNKS == 4


def test_packed_rejects_burst(env, expect_error):
    env(forward_chunks=2)
    with expect_error(AssertionError, "p_burst"):
        run(make_cfg(p_burst=0.5, forward_chunks=2))
