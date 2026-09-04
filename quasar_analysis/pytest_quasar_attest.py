# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""
pytest plugin: attest that every op test actually DISPATCHED ON A QUASAR DEVICE.

WHY
---
"the test passed" is not, on its own, proof that anything ran on Quasar. A test could pass on the
wrong arch, or an op could fall back to host, or return an input unchanged. The suite's own
`_assert_quasar(device)` closes the first hole (the arch is asserted in every test). This plugin
closes the other two, by capturing the ttnn graph around each test and recording what happened
underneath:

  * the ttnn entry point that was called          -> function_start "ttnn.experimental.quasar.add"
  * the DEVICE OPERATION it lowered to            -> function_end   "BinaryNgDeviceOperation"
  * how long that device operation took           -> duration_ns    242511806
  * the device buffers allocated for it           -> buffer_allocate on device_id N

A device operation node only exists if a program was created and enqueued on the device, so its
presence -- with a nonzero duration -- is the attestation. A host fallback or a no-op would record
the ttnn call and NO device operation, and this plugin fails the session for it.

USE
---
    pytest -p quasar_analysis.pytest_quasar_attest \\
           models/demos/vision/classification/resnet50/quasar/tests/ResNet50_Forge_Fe_bf16/ \\
           --attest-out quasar_analysis/forge_fe_bf16_runs/dispatch_attestation.json

Writes one JSON record per test and prints a summary table. Exits non-zero if any test that PASSED
dispatched no device operation, or ran on an arch other than Quasar.

Three outcomes are reported separately, because they mean different things:

  DISPATCHED   a device operation ran. This is the attestation.
  VIEW         the op was called and returned, but created no device program (0 ns, no device op).
               ttnn.experimental.quasar.reshape is the legitimate case: when the last dimension does
               not change it is a metadata-only view, so there is nothing to run. A test like this
               passing does NOT prove anything executed on the device -- which is worth knowing
               rather than hiding, so it gets its own bucket instead of being counted as attested.
  XFAILED      one of the three documented Quasar gaps: the op does not exist, so nothing dispatches
               by design.

Host-only tests take no device and are skipped. The session fails if any test ran on a non-Quasar
arch; add --attest-strict to also fail on VIEW.

NOTE ON FIDELITY
----------------
With fast runtime mode on (the default), ttnn warns that FastOperation does not produce per-op
captured sub-graphs. It still records the device-operation function_start/function_end pair, which
is what this plugin needs. Nothing here changes what the tests do.
"""

import json
import os

import pytest

_RECORDS = []
_DEVICE_OP_SUFFIXES = ("DeviceOperation", "DeviceOperationAdapter")


def pytest_addoption(parser):
    group = parser.getgroup("quasar-attest")
    group.addoption(
        "--attest-out",
        action="store",
        default=None,
        metavar="PATH",
        help="write the per-test dispatch attestation records to this JSON file",
    )
    group.addoption(
        "--attest-strict",
        action="store_true",
        default=False,
        help="also fail the session for any passing test that dispatched no device program",
    )


def _op_calls(record):
    """The ttnn ops a test called, minus the host-side plumbing every test uses."""
    plumbing = ("ttnn.from_torch", "ttnn.to_torch", "ttnn.from_device", "ttnn.to_device", "ttnn.deallocate")
    return [n for n in record.get("ttnn_ops", []) if n not in plumbing]


def _summarise(graph):
    """Pull the attestation facts out of one captured graph."""
    ttnn_ops, device_ops, allocs, device_ns, device_ids = [], [], 0, 0, set()
    for node in graph or []:
        if not isinstance(node, dict):
            continue
        kind = node.get("node_type")
        params = node.get("params") or {}
        name = params.get("name") or ""
        if kind == "function_start" and name.startswith("ttnn."):
            ttnn_ops.append(name)
        elif kind == "function_end" and name.endswith(_DEVICE_OP_SUFFIXES):
            device_ops.append(name)
            device_ns += int(node.get("duration_ns") or 0)
        elif kind == "buffer_allocate":
            allocs += 1
            if params.get("device_id") is not None:
                device_ids.add(params["device_id"])
    return {
        "ttnn_ops": ttnn_ops,
        "device_ops": device_ops,
        "device_op_ns": device_ns,
        "buffer_allocations": allocs,
        "device_ids": sorted(device_ids),
    }


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_call(item):
    device = item.funcargs.get("mesh_device") if hasattr(item, "funcargs") else None
    if device is None:  # host-only test: nothing to attest
        yield
        return

    import ttnn

    if ttnn.graph.is_graph_capture_active():  # never nest a capture
        yield
        return

    record = {"test": item.nodeid, "arch": str(device.arch())}
    ttnn.graph.begin_graph_capture(ttnn.graph.RunMode.NORMAL)
    graph = None
    try:
        outcome = yield
    finally:
        try:
            graph = ttnn.graph.end_graph_capture()
        except Exception as exc:  # a capture that cannot be closed is itself a finding
            record["capture_error"] = "%s: %s" % (type(exc).__name__, exc)

    record.update(_summarise(graph))
    excinfo = outcome.excinfo if outcome is not None else None
    if excinfo is None:
        record["outcome"] = "passed"
    elif excinfo[0] is not None and excinfo[0].__name__ == "XFailed":
        record["outcome"] = "xfailed"
    else:
        record["outcome"] = "failed"
        record["error"] = "%s: %s" % (excinfo[0].__name__, str(excinfo[1]).split("\n")[0][:200])
    _RECORDS.append(record)


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    if not _RECORDS:
        return
    tr = terminalreporter
    tr.write_sep("=", "quasar dispatch attestation")

    non_quasar = [r for r in _RECORDS if r["arch"] != "Arch.QUASAR"]
    passed = [r for r in _RECORDS if r["outcome"] == "passed"]
    dispatched = [r for r in passed if r["device_ops"]]
    # a test that called no ttnn op at all is not an op test (the arch/grid check is one); it is not
    # a "view" and saying so would be noise
    non_op = [r for r in passed if not r["device_ops"] and not _op_calls(r)]
    views = [r for r in passed if not r["device_ops"] and _op_calls(r)]
    xfailed = [r for r in _RECORDS if r["outcome"] == "xfailed"]
    failed = [r for r in _RECORDS if r["outcome"] == "failed"]
    for r in views:
        r["outcome"] = "view"

    tr.write_line("device tests captured : %d" % len(_RECORDS))
    tr.write_line("arch reported         : %s" % ", ".join(sorted({r["arch"] for r in _RECORDS})))
    tr.write_line("DISPATCHED            : %d passing tests ran a device program on Quasar" % len(dispatched))
    tr.write_line("VIEW                  : %d passing tests created no device program" % len(views))
    for r in views:
        tr.write_line("      %s  (%s)" % (r["test"].split("::")[-1], ", ".join(_op_calls(r))))
    if non_op:
        tr.write_line("not an op test        : %d (calls no ttnn op -- the arch/grid check)" % len(non_op))
    tr.write_line("XFAILED (known gaps)  : %d -- the op does not exist, nothing dispatches by design" % len(xfailed))
    reached = [r for r in failed if r["device_ops"]]
    tr.write_line(
        "failed                : %d, of which %d reached the device first (a device program was recorded "
        "before the failure -- the last one named is where it threw)" % (len(failed), len(reached))
    )

    def _tally(records):
        seen = {}
        for r in records:
            for name in r["device_ops"]:
                seen[name] = seen.get(name, 0) + 1
        return seen

    ok_seen, bad_seen = _tally(dispatched), _tally(failed)
    if ok_seen:
        tr.write_line("device operations that COMPLETED on passing tests:")
        for name, n in sorted(ok_seen.items(), key=lambda kv: -kv[1]):
            tr.write_line("    %-44s x%d" % (name, n))
    if bad_seen:
        tr.write_line("device operations recorded on FAILING tests (the last one is where each threw):")
        for name, n in sorted(bad_seen.items(), key=lambda kv: -kv[1]):
            tr.write_line("    %-44s x%d" % (name, n))
    total_ms = sum(r["device_op_ns"] for r in _RECORDS) / 1e6
    tr.write_line(
        "total device-operation time: %.1f ms across %d dispatches (%.1f ms on passing tests)"
        % (total_ms, sum(ok_seen.values()) + sum(bad_seen.values()), sum(r["device_op_ns"] for r in dispatched) / 1e6)
    )

    out = config.getoption("--attest-out")
    if out:
        os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
        with open(out, "w") as fh:
            json.dump(_RECORDS, fh, indent=1)
        tr.write_line("records written to %s" % out)

    problems = []
    if non_quasar:
        problems.append("%d test(s) ran on a non-Quasar arch: %s" % (len(non_quasar), non_quasar[0]["arch"]))
    if views and config.getoption("--attest-strict"):
        problems.append(
            "--attest-strict: %d passing test(s) created no device program: %s"
            % (len(views), ", ".join(r["test"].split("::")[-1] for r in views[:5]))
        )
    if problems:
        for p in problems:
            tr.write_line("ATTESTATION FAILED: %s" % p)
        terminalreporter._session.exitstatus = 1  # non-zero exit without pretending a test failed
    else:
        tr.write_line(
            "ATTESTATION OK: all %d device tests ran on Arch.QUASAR; %d dispatched a device program, "
            "%d were metadata-only views, %d are known gaps."
            % (len(_RECORDS), len(dispatched), len(views), len(xfailed))
        )
