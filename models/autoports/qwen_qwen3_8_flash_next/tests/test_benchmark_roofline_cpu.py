# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only regressions for completion attribution and missing-work rejection."""

import importlib.util
import sys
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))
spec = importlib.util.spec_from_file_location("qwen_roofline_test", TOOLS / "benchmark_roofline.py")
roofline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(roofline)


def test_delayed_earlier_host_completion_is_inside_full_timeline():
    events = [
        dict(
            ticket=0,
            phase="decode",
            begin_ns=10,
            completion_lower_ns=90,
            completion_observed_ns=100,
            host_complete_ns=400,
        ),
        dict(
            ticket=1,
            phase="prefill",
            begin_ns=50,
            completion_lower_ns=190,
            completion_observed_ns=200,
            host_complete_ns=205,
        ),
    ]
    seconds, intervals = roofline.partition(events)
    assert intervals[-1]["end_ns"] == 400
    assert sum(seconds.values()) == pytest.approx(390 / 1e9)
    assert intervals[0]["end_ns"] == intervals[1]["start_ns"]


def test_missing_decode_host_completion_rejected():
    request = dict(request=1, first_submission_ns=1_010_000_000, prompt_sha256="different", first_page=1)
    events = [
        dict(
            ticket=0,
            phase="prefill",
            end_order=0,
            completion_observed_ns=1_020_000_000,
            metadata={"requests": [dict(request, start=0, end=4096)]},
        )
    ]
    for i in range(127):
        events.append(
            dict(
                ticket=i + 1,
                phase="decode",
                end_order=i + 1,
                completion_observed_ns=1_030_000_000 + i,
                metadata={"requests": [dict(request, position=4096 + i)]},
            )
        )
    perf = dict(start_times=[1.0], ttfts=[0.1], itls=[[0.01] * 127], completed=1)
    with pytest.raises(ValueError, match="host readback completion"):  # allow-pytest.raises: pure CPU error contract
        roofline.select_events({"events": events}, perf)
    for e in events[1:]:
        e["host_complete_ns"] = e["completion_observed_ns"]
    selected, requests, _ = roofline.select_events({"events": events}, perf)
    assert len(selected) == 128
    assert requests[0]["queued_extra_decode_steps"] == 0
    events[65]["metadata"]["requests"][0]["position"] += 1
    with pytest.raises(ValueError, match="consecutive"):  # allow-pytest.raises: pure CPU error contract
        roofline.select_events({"events": events}, perf)


def native_event(event_id, order, *, device=0, queue=0, range_end=3):
    return dict(
        ticket=order,
        end_order=order,
        event_repr=(
            f"MeshEvent(id={event_id}, device_id={device}, mesh_cq_id={queue}, "
            f"device_range=MeshCoordinateRange(start=MeshCoordinate([0, 0]), end=MeshCoordinate([0, {range_end}])))"
        ),
    )


def test_native_fifo_accepts_gaps_and_retains_identity_and_order():
    result = roofline.validate_native_fifo([native_event(1050, 3), native_event(1067, 4)])
    assert result["cq_id"] == 0
    assert result["device_id"] == 0
    assert result["checked_count"] == 2
    assert [row["native_event_id"] for row in result["events"]] == [1050, 1067]


@pytest.mark.parametrize(
    "second",
    [
        native_event(1049, 4),
        native_event(1050, 4),
        native_event(1067, 2),
        native_event(1067, 4, device=1),
        native_event(1067, 4, queue=1),
        native_event(1067, 4, range_end=2),
        dict(ticket=4, end_order=4, event_repr="MeshEvent(unrecognized)"),
    ],
)
def test_native_fifo_rejects_order_or_mesh_mismatch(second):
    with pytest.raises(ValueError):  # allow-pytest.raises: pure CPU error contract
        roofline.validate_native_fifo([native_event(1050, 3), second])
