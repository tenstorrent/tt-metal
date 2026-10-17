#!/usr/bin/env python3

# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import ast
import inspect

import pytest

from tracy import process_device_log

# The names of the zones and riscs a dispatch core reports, read from the module under test so
# this file never has to spell out a name the profiler could rename.
_DISPATCH_ZONES = (process_device_log.DISPATCH_ZONE_NAME, process_device_log.PREFETCH_ZONE_NAME)
_MASTER = process_device_log.DISPATCH_MASTER_RISC
_SUBORDINATE = process_device_log.DISPATCH_SUBORDINATE_RISC

_HEADER = (
    "ARCH: blackhole, CHIP_FREQ[MHz]: 1350, Max Compute Cores: 120\n"
    "PCIe slot, core_x, core_y, RISC processor type, timer_id, time[cycles since reset], data, "
    "run host ID, trace id, trace id counter, zone name, type, source line, source file, meta data\n"
)


@pytest.fixture(autouse=True)
def _forget_dispatch_cores():
    """The set of dispatch cores is module state, and a stale entry would change what a test sees."""
    process_device_log.dispatchCores.clear()
    yield
    process_device_log.dispatchCores.clear()


def _write_log(path, rows):
    path.write_text(_HEADER + "".join(row + "\n" for row in rows))
    return str(path)


def _row(core, risc, marker, time, zone, kind, meta=""):
    return "0,%d,%d,%s,%d,%d,0,1024,,,%s,%s,433,/x/src.cc,%s" % (core[0], core[1], risc, marker, time, zone, kind, meta)


def _dispatch_zone(core, risc, time, command):
    """One dispatch zone. Its first and last markers are identical between zones, by design."""
    return [
        _row(core, risc, 900, time, _DISPATCH_ZONES[0], "ZONE_START"),
        _row(core, risc, 910, time + 1, _DISPATCH_ZONES[0], "ZONE_START", "{'workers_runtime_id': %d}" % (time,)),
        _row(core, risc, 911, time + 2, _DISPATCH_ZONES[0], "ZONE_START", "{'dispatch_command_type': '%s'}" % command),
        _row(core, risc, 901, time + 3, _DISPATCH_ZONES[0], "ZONE_END"),
    ]


def test_renaming_one_dispatch_zone_does_not_rename_the_others(tmp_path):
    """A zone name rewritten on one zone must not reach the identical marker in another.

    get_dispatch_core_ops rewrites the zone name of the markers bounding each dispatch zone in
    place. import_device_profile_log hands the same marker to every row reporting it, so without
    a private copy for these cores one rewrite would rename every zone that shares the marker --
    silently, and every op measured off those names would be attributed to the wrong command.
    """
    core = (1, 1)
    rows = []
    for risc in (_MASTER, _SUBORDINATE):
        rows += _dispatch_zone(core, risc, 1000, "CQ_DISPATCH_CMD_WRITE_LINEAR")
        rows += _dispatch_zone(core, risc, 2000, "CQ_DISPATCH_CMD_WRITE_PAGED")

    devicesData = process_device_log.import_device_profile_log(_write_log(tmp_path / "log.csv", rows))
    assert (0, core) in process_device_log.dispatchCores

    process_device_log.risc_to_core_timeseries(devicesData, True)
    bounding = [
        entry[0]["zone_name"]
        for entry in devicesData["devices"][0]["cores"][core]["riscs"][_MASTER]["timeseries"]
        if entry[0]["id"] in (900, 901)
    ]
    assert bounding == [
        "CQ_DISPATCH_CMD_WRITE_LINEAR",
        "CQ_DISPATCH_CMD_WRITE_LINEAR",
        "CQ_DISPATCH_CMD_WRITE_PAGED",
        "CQ_DISPATCH_CMD_WRITE_PAGED",
    ]


def test_rows_reporting_the_same_marker_are_given_the_same_one(tmp_path):
    """Every distinct marker is built once. A dict per row is what makes a large log unreadable."""
    core = (2, 2)
    rows = [_row(core, _MASTER, 500, 1000 + n, "TEST-ZONE", "ZONE_START") for n in range(8)]

    devicesData = process_device_log.import_device_profile_log(_write_log(tmp_path / "log.csv", rows))

    timeseries = devicesData["devices"][0]["cores"][core]["riscs"][_MASTER]["timeseries"]
    assert len(timeseries) == len(rows)
    assert len({id(entry[0]) for entry in timeseries}) == 1


def test_a_marker_is_not_shared_across_differing_rows(tmp_path):
    """Sharing keys on every field a marker carries, so rows that differ must not collide."""
    core = (3, 3)
    rows = [
        _row(core, _MASTER, 500, 1000, "TEST-ZONE", "ZONE_START"),
        _row(core, _MASTER, 500, 1001, "TEST-ZONE", "ZONE_END"),
        _row(core, _MASTER, 501, 1002, "OTHER-ZONE", "ZONE_START"),
    ]

    devicesData = process_device_log.import_device_profile_log(_write_log(tmp_path / "log.csv", rows))

    timeseries = devicesData["devices"][0]["cores"][core]["riscs"][_MASTER]["timeseries"]
    assert [entry[0]["zone_name"] for entry in timeseries] == ["TEST-ZONE", "TEST-ZONE", "OTHER-ZONE"]
    assert [entry[0]["type"] for entry in timeseries] == ["ZONE_START", "ZONE_END", "ZONE_START"]
    assert len({id(entry[0]) for entry in timeseries}) == 3


def test_a_log_longer_than_one_batch_reads_the_same_as_a_short_one(tmp_path):
    """Rows are read in batches; a marker seen in an earlier batch is still reused in a later one."""
    core = (4, 4)
    rows = [_row(core, _MASTER, 500, 1000 + n, "TEST-ZONE", "ZONE_START") for n in range(10)]
    log = _write_log(tmp_path / "log.csv", rows)

    whole = process_device_log.import_device_profile_log(log)
    original_batch = process_device_log.CSV_CHUNK_ROWS
    try:
        process_device_log.CSV_CHUNK_ROWS = 3
        batched = process_device_log.import_device_profile_log(log)
    finally:
        process_device_log.CSV_CHUNK_ROWS = original_batch

    def timeseries(devicesData):
        return devicesData["devices"][0]["cores"][core]["riscs"][_MASTER]["timeseries"]

    assert timeseries(batched) == timeseries(whole)
    assert len({id(entry[0]) for entry in timeseries(batched)}) == 1


def _marker_field_writes(tree):
    """Every place the source writes one of a marker's own fields in place, by function.

    The fields are read from the dict import_device_profile_log stores in its marker cache, so
    this follows the marker if it ever grows or loses one.
    """
    cache_name = "markers"
    fields = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Dict):
            continue
        for target in node.targets:
            if (
                isinstance(target, ast.Subscript)
                and isinstance(target.value, ast.Name)
                and target.value.id == cache_name
            ):
                fields = {key.value for key in node.value.keys if isinstance(key, ast.Constant)}
    assert fields, "could not find where a marker is built; this test needs updating"

    writes = {}
    for parent in ast.walk(tree):
        if not isinstance(parent, ast.FunctionDef):
            continue
        for node in ast.walk(parent):
            targets = node.targets if isinstance(node, ast.Assign) else [getattr(node, "target", None)]
            for target in targets:
                if (
                    isinstance(target, ast.Subscript)
                    and isinstance(target.slice, ast.Constant)
                    and target.slice.value in fields
                ):
                    writes.setdefault(parent.name, set()).add(target.slice.value)
    return writes


def test_nothing_writes_to_a_marker_that_is_still_shared():
    """A marker is shared between every row reporting it, so writing to one writes to all of them.

    The one pass that does rewrite a marker is get_dispatch_core_ops, and the cores it rewrites
    are unshared first. Any other in-place write would silently change rows it was never meant
    to touch, so this fails the moment a new one appears rather than at the wrong answer.
    """
    tree = ast.parse(inspect.getsource(process_device_log))

    writes = _marker_field_writes(tree)

    assert set(writes) == {process_device_log.get_dispatch_core_ops.__name__}, (
        "these write a marker field in place: %s. A marker is shared, so unshare the core first "
        "(see unshare_markers) or build a new marker instead." % sorted(writes)
    )
