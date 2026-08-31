# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

from tests.ttnn.profiling import realtime_profiler_utils


def test_profile_realtime_program_merged_preserves_chip_completeness(monkeypatch) -> None:
    records = [
        {"runtime_id": 7, "chip_id": 2, "duration_ns": 5.0, "kernel_sources": ("reader.cpp",)},
        {"runtime_id": 7, "chip_id": 1, "duration_ns": 8.0, "kernel_sources": ("reader.cpp",)},
        {"runtime_id": 8, "chip_id": 1, "duration_ns": 2.0, "kernel_sources": ("writer.cpp",)},
    ]

    def profile(device, run_fn, *, collect_all, record_timeout_seconds):
        del device, run_fn, record_timeout_seconds
        assert collect_all
        return "result", records

    monkeypatch.setattr(realtime_profiler_utils, "profile_realtime_program", profile)
    result, programs = realtime_profiler_utils.profile_realtime_program_merged(None, lambda: None)

    assert result == "result"
    assert programs[7] == {
        "duration_ns": 8.0,
        "kernel_sources": ("reader.cpp",),
        "chip_ids": (1, 2),
        "record_count": 2,
    }
    assert programs[8]["chip_ids"] == (1,)
    assert programs[8]["record_count"] == 1
