# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Two tracy switches were wired to "did the drain plugin load", which answers neither question.

`profiler_drain_plugin()` returns a module name whenever the tool's own tree is importable, so it
is true on every box the tool runs on. Hanging other decisions off it makes them unconditional
while reading like a condition. Both switches below now default off and are asked for explicitly.

THE MID-RUN DUMP (--dump-device-data-mid-run) is a HOST MEMORY remedy.

tracy's --dump-device-data-mid-run writes every profiler read out instead of holding it in host
memory until exit. It was introduced for a 32-chip WH Galaxy whose profiled worker reached 320 GB
resident, and it was wired to `mid_run_dump=bool(_drain)` -- true on every box that loads the drain
plugin, which is every box.

It is not free. Measured on a T3K 2x4 (2026-10-27), the same capture with the drain loaded and only
this flag changed: vision_encode 1m43s without it, 24+ min with it. The whole-pipeline capture went
from 27 minutes to over ten hours and was killed at the discovery ceiling with no baseline -- on a
503 GB box whose capture peaked well inside its memory.

So the default is off, and a host that genuinely cannot hold the markers says so by killing the
attempt; the caller then retries with the dump on. No threshold is picked and no box is named.
"""

from __future__ import annotations

from agent.probes import (
    build_tracy_command,
    mid_run_dump_default,
    push_device_to_tracy_default,
)

_MID_RUN_FLAG = "--dump-device-data-mid-run"


def test_loading_the_drain_no_longer_turns_the_dump_on():
    """The regression in one line: a plugin being present is not evidence about memory."""
    assert mid_run_dump_default(True) is False
    assert mid_run_dump_default(False) is False


def test_an_operator_can_force_it_either_way():
    assert mid_run_dump_default(True, {"PERF_MCP_PROFILER_MID_RUN_DUMP": "1"}) is True
    assert mid_run_dump_default(True, {"PERF_MCP_PROFILER_MID_RUN_DUMP": "0"}) is False


def test_forcing_it_on_still_requires_the_drain():
    """The dump releases what the drain reads; without a drain there are no reads to release."""
    assert mid_run_dump_default(False, {"PERF_MCP_PROFILER_MID_RUN_DUMP": "1"}) is False


def test_the_flag_reaches_the_command_only_when_asked_for():
    """The decision is worth nothing if the command does not carry it."""
    off = build_tracy_command("t.py", None, "/tmp/out", plugins=("p",), mid_run_dump=False)
    on = build_tracy_command("t.py", None, "/tmp/out", plugins=("p",), mid_run_dump=True)
    assert _MID_RUN_FLAG not in off
    assert _MID_RUN_FLAG in on
    # ...and the plugin is loaded either way: the drain is not what this flag controls.
    assert off.count("-p") == on.count("-p")


# --- the second switch: device markers in the tracy stream ------------------------------------
#
# THE COST, measured on this box 2026-10-27. A capture run with the drain plugin absent pushed
# device markers into the tracy stream, and tracy-capture died of SIGSEGV after 31 minutes of
# completed device work -- "killed by SIGSEGV (signal 11) before writing a trace", no .tracy at
# all. The same pipeline with the push disabled wrote a 41 MB cpp_device_perf_report.csv holding
# 167,888 device rows. The device data the ops report is built from comes from the device log
# either way; the tracy stream only needs the host side. So pushing buys nothing and can lose the
# whole capture -- and whether the drain loaded says nothing about either.

_PUSH_ENV = "PERF_MCP_PROFILER_PUSH_DEVICE_TO_TRACY"
_NO_PUSH_FLAG = "--disable-device-data-push-to-tracy"


def test_device_markers_stay_out_of_the_tracy_stream_by_default():
    assert push_device_to_tracy_default() is False
    assert push_device_to_tracy_default({}) is False


def test_an_operator_who_wants_device_zones_in_the_gui_can_have_them():
    """Not forbidden -- just not the default, because the default is what runs unattended."""
    assert push_device_to_tracy_default({_PUSH_ENV: "1"}) is True
    assert push_device_to_tracy_default({_PUSH_ENV: "0"}) is False


def test_both_switches_read_an_override_the_same_way():
    """One parser, so "off" cannot mean one thing to one switch and another to the other."""
    for raw, wanted in (("1", True), ("yes", True), ("0", False), ("false", False), ("OFF", False)):
        assert mid_run_dump_default(True, {"PERF_MCP_PROFILER_MID_RUN_DUMP": raw}) is wanted, raw
        assert push_device_to_tracy_default({_PUSH_ENV: raw}) is wanted, raw


def test_the_no_push_flag_reaches_the_command():
    pushing = build_tracy_command("t.py", None, "/tmp/out", push_device_to_tracy=True)
    not_pushing = build_tracy_command("t.py", None, "/tmp/out", push_device_to_tracy=False)
    assert _NO_PUSH_FLAG not in pushing
    assert _NO_PUSH_FLAG in not_pushing
