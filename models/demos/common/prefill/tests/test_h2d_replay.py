# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import pytest

from models.demos.common.prefill.runners.h2d_replay import InjectRecord, load_inject_log, parse_inject_log

CHUNK = 5120


def _line(ts: str, slot: int, start: int, end: int) -> str:
    return (
        f"{ts} INFO     Pipeline  tt-d-gen/4242 prefill_pipeline.cpp:155  "
        f"prefill inject slot_id={slot} start={start} end={end}\n"
    )


def _parse(lines, num_slots=4, max_seq_len=CHUNK * 11):
    return parse_inject_log(lines, chunk_size=CHUNK, max_seq_len=max_seq_len, num_slots=num_slots)


def test_runs_share_isl_and_count_chunks_per_slot():
    lines = [
        _line("2026-09-29T10:00:00.000", 0, 0, CHUNK),
        _line("2026-09-29T10:00:00.250", 1, 0, CHUNK),
        "2026-09-29T10:00:00.260 INFO     Pipeline  tt-d-gen/4242 other.cpp:1  unrelated line\n",
        _line("2026-09-29T10:00:00.500", 0, CHUNK, CHUNK + 100),
        _line("2026-09-29T10:00:01.000", 1, CHUNK, 2 * CHUNK),
        _line("2026-09-29T10:00:01.500", 0, 96, 96 + CHUNK),
    ]

    records = _parse(lines)

    assert [(r.slot_id, r.actual_start, r.actual_end, r.actual_isl, r.chunk_idx) for r in records] == [
        (0, 0, CHUNK, CHUNK + 100, 0),
        (1, 0, CHUNK, 2 * CHUNK, 0),
        (0, CHUNK, CHUNK + 100, CHUNK + 100, 1),
        (1, CHUNK, 2 * CHUNK, 2 * CHUNK, 1),
        (0, 96, 96 + CHUNK, 96 + CHUNK, 0),
    ]
    assert records[1].t_s - records[0].t_s == pytest.approx(0.25)
    assert records[4].t_s - records[0].t_s == pytest.approx(1.5)


def test_untimestamped_lines_and_shutdown_sentinel():
    records = _parse(
        [
            "[worker] prefill inject slot_id=2 start=0 end=10\n",
            f"prefill inject slot_id={0xFFFFFFFF} start={0xFFFFFFFF} end={0xFFFFFFFF}\n",
        ]
    )

    assert records == [InjectRecord(2, 0, 10, 10, 0, None)]


@pytest.mark.parametrize(
    "slot, start, end, match",
    [
        (0, 0, CHUNK + 1, "not a chunk"),
        (0, 10, 10, "not a chunk"),
        (0, CHUNK * 10, CHUNK * 11 + 1, "not a chunk"),
        (4, 0, CHUNK, "num_users"),
    ],
)
def test_rejects_pushes_the_runner_cannot_take(slot, start, end, match):
    with pytest.raises(ValueError, match=match):
        _parse([_line("2026-09-29T10:00:00.000", slot, start, end)])


def test_load_rejects_log_without_injects(tmp_path):
    path = tmp_path / "engine.log"
    path.write_text("nothing to see\n")

    with pytest.raises(ValueError, match="no 'prefill inject' lines"):
        load_inject_log(str(path), chunk_size=CHUNK, max_seq_len=CHUNK, num_slots=1)


class _FakeClock:
    def __init__(self):
        self.t = 100.0
        self.sleeps = []

    def now(self):
        return self.t

    def sleep(self, s):
        self.sleeps.append(s)
        self.t += s


def _replay(records, speed, push_cost_s=0.0):
    from models.demos.common.prefill.runners.prefill_producer import replay_schedule

    clock = _FakeClock()
    pushes = []

    def push_fn(*args):
        pushes.append((clock.t, args))
        clock.t += push_cost_s
        return push_cost_s * 1000.0

    stats = replay_schedule(records, push_fn=push_fn, speed=speed, now_fn=clock.now, sleep_fn=clock.sleep)
    return stats, pushes


def _timed_records():
    return _parse(
        [
            _line("2026-09-29T10:00:00.000", 0, 0, CHUNK),
            _line("2026-09-29T10:00:01.000", 1, 0, 300),
            _line("2026-09-29T10:00:03.000", 0, CHUNK, 2 * CHUNK),
        ]
    )


@pytest.mark.parametrize("speed, expected_offsets", [(1.0, [0.0, 1.0, 3.0]), (2.0, [0.0, 0.5, 1.5]), (0, [0, 0, 0])])
def test_replay_honours_recorded_gaps(speed, expected_offsets):
    stats, pushes = _replay(_timed_records(), speed)

    assert [t - 100.0 for t, _ in pushes] == pytest.approx(expected_offsets)
    assert [args for _, args in pushes] == [
        (0, 0, 0, CHUNK, 2 * CHUNK),
        (1, 0, 0, 300, 300),
        (0, 1, CHUNK, 2 * CHUNK, 2 * CHUNK),
    ]
    assert stats.total_pushes == 3
    assert stats.completed == 2
    assert {s: f.real_len for s, f in stats.resident.items()} == {0: 2 * CHUNK, 1: 300}


def test_replay_catches_up_after_a_slow_push():
    _, pushes = _replay(_timed_records(), 1.0, push_cost_s=2.0)

    assert [t - 100.0 for t, _ in pushes] == pytest.approx([0.0, 2.0, 4.0])


def test_replay_rejects_pcc_check(tmp_path):
    from models.demos.common.prefill.runners import prefill_producer as producer

    path = tmp_path / "engine.log"
    path.write_text(_line("2026-09-29T10:00:00.000", 0, 0, 10))
    cfg = producer._config_from_env()
    cfg.replay_log = str(path)
    cfg.verify = True

    with pytest.raises(ValueError, match="CHECK_PCC"):
        producer._load_replay_records(cfg)

    cfg.verify = False
    assert len(producer._load_replay_records(cfg)) == 1
