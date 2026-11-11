# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Render the sweep as junit XML, including the cases that hung.

Written here rather than by pytest's --junit-xml for two reasons, both of which
only bite on the runs that matter:

  - pytest writes its XML once, at session end. When a perturbation wedges a core
    the supervisor kills that session, so the file is never written and every
    result in that attempt is lost. Building from the append-only result log
    workers keep as they go survives the kill, and spans the several attempts a
    resumed sweep is split across.

  - a wedged case is never reported by anyone. The worker that would have
    reported it is blocked inside a device read and will not return; xdist
    notices the missing worker but names it a crashed node, not a failing test.
    The supervisor knows which case and which variant did it, so those go in
    here as ordinary failures — which is the whole point of the exercise.
"""

import re
from pathlib import Path
from xml.etree import ElementTree

# XML 1.0 forbids most control characters outright, and a failure message can
# carry them straight out of a device dump. ElementTree will not escape these
# for us, so a single stray byte would otherwise produce a file no CI can parse.
_ILLEGAL = re.compile(r"[^\x09\x0a\x0d\x20-\ud7ff\ue000-\ufffd]")

# junit consumers put this in the failure list; the body carries the detail.
_SUMMARY_LIMIT = 200


def _clean(text) -> str:
    return _ILLEGAL.sub("", str(text or ""))


def _summary(text: str) -> str:
    head = _clean(text).strip().splitlines()
    return (head[0] if head else "")[:_SUMMARY_LIMIT]


def _testcase(nodeid: str, duration: float) -> ElementTree.Element:
    # Split a pytest node id into the class/name pair junit expects, so results
    # group by file in a viewer rather than arriving as one flat list.
    # No fallbacks: nodeids always carry "::" and durations are rounded floats.
    path, _, name = _clean(nodeid).partition("::")
    return ElementTree.Element(
        "testcase",
        classname=path.replace("/", ".").removesuffix(".py"),
        name=name,
        time=f"{duration:.3f}",
    )


def _wedge_case(wedge: dict) -> ElementTree.Element:
    """One hung case, as the failure pytest was never in a position to report."""
    variant = wedge["variant"]
    # Baseline and reproducibility hangs happen before a variant is published.
    label = _clean(variant.get("label") or "baseline/reproducibility run")
    element = _testcase(wedge["case"], 0.0)
    failure = ElementTree.SubElement(
        element,
        "failure",
        type="Wedge",
        message=f"hang: {label} wedged the core"[:_SUMMARY_LIMIT],
    )
    failure.text = _clean(
        f"The Tensix core stopped answering for {wedge['age']:.0f}s while this case "
        f"was running, and the worker had to be killed. The perturbation in flight "
        f"at the time was:\n\n    {label}\n\n"
        f"Full variant: {variant}\n"
        f"See the ttnop report artifact for the resolved source location."
    )
    return element


def _result_case(nodeid: str, record: dict) -> tuple:
    element = _testcase(nodeid, record["duration"])
    outcome = record["outcome"]
    message = record["message"]
    if outcome == "failed":
        failure = ElementTree.SubElement(
            element, "failure", type="AssertionError", message=_summary(message)
        )
        failure.text = _clean(message)
    elif outcome in ("skipped", "xfailed"):
        attributes = {"message": _summary(message)}
        if outcome == "xfailed":
            attributes["type"] = "pytest.xfail"
        ElementTree.SubElement(element, "skipped", **attributes)
    return element, outcome


def render(results: dict, wedges: list, path: Path) -> Path:
    """Write one junit file covering every case the sweep reached.

    A case that wedged is reported only as a wedge: it may also carry a result
    line from an earlier attempt, but the hang is the more useful story and
    duplicate node ids confuse junit viewers.
    """
    cases = []
    failed = skipped = 0
    total_time = 0.0

    wedged_ids = set()
    for wedge in wedges:
        cases.append(_wedge_case(wedge))
        wedged_ids.add(wedge["case"])
        failed += 1

    for nodeid, record in sorted(results.items()):
        if nodeid in wedged_ids:
            continue
        element, outcome = _result_case(nodeid, record)
        cases.append(element)
        total_time += record["duration"]
        if outcome == "failed":
            failed += 1
        elif outcome in ("skipped", "xfailed"):
            skipped += 1

    suite = ElementTree.Element(
        "testsuite",
        name="ttnop",
        tests=str(len(cases)),
        failures=str(failed),
        errors="0",
        skipped=str(skipped),
        time=f"{total_time:.3f}",
    )
    suite.extend(cases)
    root = ElementTree.Element("testsuites")
    root.append(suite)

    path.parent.mkdir(parents=True, exist_ok=True)
    ElementTree.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)
    return path
