#!/usr/bin/env python3

# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Turn a `tt-bh-glx-cluster-debug` dump into health-check findings.

`tt-bh-glx-cluster-debug collect` (the syseng cluster-debug .deb) writes a JSONL
snapshot of one Galaxy: flat records for the chassis, its 4 UBBs, 32 ASICs, 448
ETH ports, 56 QSFP cages and their modules. That file is an inventory, not a
verdict — by design it states what the topology *expects* of every link and what
the firmware actually *found*, side by side, and leaves the comparison to the
reader. Nothing in it says PASS or FAIL.

This is that reader. Each fault class below is one of the access patterns the
dump was shaped to answer (cluster_debug_spec.md §10.3, declared as SQL in the
tool's own ``visualizer/queries.py``); they are re-derived here in Python over a
single dump so the health check needs neither the merge step nor the SQLite
database. Where the two could drift — the all-zero ``remote_info`` guard, which
records read as "unread", how a both-ends-blind pair is collapsed — this file
mirrors the query and says so at the point it matters.

Output is the shape `tools/scaleout/kmd_triage/triage_json.sh` emits and
`diag_runner.py` already ingests:

    {"tool": ..., "version": N,
     "checks": [{"name", "status", "details", "ip", "data"}]}

One shape for every externally-produced finding, so the rollup to a single
verdict stays computed in one place (diag_runner's Phase) rather than here.
Statuses are advisory: the ingesting phase holds FAIL at WARN unless it was
asked to let this tool gate the run.

Runnable on its own against a stored dump, with the same
``--json <path> -o <path>`` interface the triage scripts take, which is what
makes a dump collected on a sick machine reviewable on a desk:

    python3 qsfp_ingest.py qsfp_dump_host.jsonl --json out.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

TOOL_NAME = "tt-bh-glx-cluster-debug"

# Bumped when the derivation below changes what it calls a fault, so a stored
# report says which rules produced it. Not the dump's SCHEMA_VER, which the
# collector owns and `qsfp_inventory` records separately.
INGEST_VERSION = 1

PASS, WARN, FAIL, SKIP = "PASS", "WARN", "FAIL", "SKIP"

# A BH Galaxy 6U, per cluster_debug_spec.md §4. eth_port is 14 per ASIC
# including the harvested and non-link tiles, so it scales with the ASIC count
# rather than being a constant of its own.
EXPECTED_UBBS = 4
EXPECTED_ASICS = 32
ETH_PORTS_PER_ASIC = 14

# COLLECTION.STATUS, from the collector's schema/base.py CollectionStatus.
#
# The split is the one the tool's own queries draw: a record that was not read
# says nothing about the part, and counting it makes an unread port a down one.
# HARVESTED is not a failure — the tile is disabled in ENABLED_ETH and PORT_TYPE
# says whether that is a real harvest or a PCIe tile.
READ_OK = ("OK", "HARVESTED")
# The part did not answer, or answered and the read failed. A hardware fault.
HARD_FAILURES = ("READ_FAILED", "UNREACHABLE", "ABSENT")
# PARTIAL: the record exists but some DATA blobs are missing. SKIPPED: an opt-in
# stage did not run (the cage sweep, typically, when ipmitool is unusable).
# Neither is a statement about the hardware — both are lost coverage.
SOFT_FAILURES = ("PARTIAL", "SKIPPED")

# PORT_TYPEs that never carry a link, so a port holding one is not "down".
NO_LINK_PORT_TYPES = ("PCIE", "UNCONNECTED", "INVALID_LOCATION")

# PORT_TYPEs that leave the chassis through a QSFP cage, and so cannot train
# until somebody plugs a cable into that cage. A partly populated Galaxy is a
# supported configuration, which makes "this port did not train" meaningless
# for one of these until the cage is known to hold a module.
#
# The EXAMAX types are deliberately absent: they are the inter-UBB links inside
# the chassis, soldered to the backplane, and they train with nothing plugged
# in. So are CHIP_TO_CHIP's, which is why only these two are listed.
CAGED_PORT_TYPES = ("CHIP_TO_QSFPDD", "CHIP_TO_WARP400")

# COLLECTION.STATUS values that make a cage's PRESENT meaningless: the cage was
# never read, so it is not empty, it is unknown.
CAGE_UNREAD = ("SKIPPED", "READ_FAILED", "UNREACHABLE")

TRAIN_PASS = "LINK_TRAIN_PASS"

# Offending parts named in a check's one-line `details`. The full list always
# goes to `data`, which is what the JSON report and a ticket carry.
MAX_LISTED = 4


# ─────────────────────────────────────────────────────────────────────────────
# Reading the dump
# ─────────────────────────────────────────────────────────────────────────────


def load_dump(path: str | Path) -> tuple[list[dict], int]:
    """Records from a JSONL dump, plus a count of lines that were not records.

    Returns ``(records, malformed)``. The collector writes JSONL precisely so a
    file truncated by a host that fell over mid-run stays valid up to its last
    newline, so a short final line is dropped rather than taken as a broken
    file — but it is counted, because a dump quietly missing records would
    otherwise read as a galaxy quietly missing parts.
    """
    records: list[dict] = []
    malformed = 0
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except ValueError:
                malformed += 1
                continue
            if isinstance(record, dict):
                records.append(record)
            else:
                malformed += 1
    return records, malformed


def group_by_schema(records: Iterable[dict]) -> dict[str, list[dict]]:
    """Records bucketed by their SCHEMA (entity type)."""
    groups: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        groups[str(record.get("SCHEMA") or "")].append(record)
    return dict(groups)


def _blob(record: dict, name: str) -> dict:
    """A DATA blob's VALUE, or {} when the blob is absent or carries an ERROR.

    A blob that failed to read carries ``{SOURCE, ERROR}`` in place of VALUE, so
    reaching straight for VALUE would raise on exactly the records a failing
    machine produces most of.
    """
    blob = (record.get("DATA") or {}).get(name)
    if not isinstance(blob, dict):
        return {}
    value = blob.get("VALUE")
    return value if isinstance(value, dict) else {}


def collection_status(record: dict) -> str:
    return str((record.get("COLLECTION") or {}).get("STATUS") or "")


def port_read(port: dict) -> bool:
    """Whether this port's record is worth drawing a conclusion from.

    A PARTIAL port has no STATUS block at all — the collector drops it when any
    blob fails — so unlike a cage, a port counts only when its collection was
    clean. Mirrors PORT_READ in the tool's queries.py.
    """
    return collection_status(port) in READ_OK


def in_service(port: dict) -> bool:
    """A port that is meant to carry a link: not harvested, not a PCIe tile."""
    return not port.get("HARVESTED") and str(port.get("PORT_TYPE") or "") not in NO_LINK_PORT_TYPES


def behind_a_cage(port: dict) -> bool:
    """Whether this port only carries a link once a cable is plugged in.

    Keyed on PORT_TYPE, the static topology, rather than on the port's QSFP_UID:
    the cage pointer arrives with the opt-in cage stage and is null on a dump
    collected with --skip-qsfp, where the port is still just as cage-attached.
    """
    return str(port.get("PORT_TYPE") or "") in CAGED_PORT_TYPES


def cage_occupancy(cages: list[dict]) -> dict[str, bool]:
    """ENTRY_ID → whether a module answered in that cage.

    Only cages whose own read succeeded are in the map. PRESENT off a cage that
    never answered says nothing, so a port behind one is left as unjudgeable
    rather than quietly counted as uncabled — the difference between "there is
    no cable here" and "nobody looked".
    """
    return {
        str(cage["ENTRY_ID"]): bool(cage.get("PRESENT"))
        for cage in cages
        if cage.get("ENTRY_ID") and collection_status(cage) not in CAGE_UNREAD
    }


def port_status(port: dict, field: str) -> Any:
    """One field of a port's promoted STATUS block, or None if it has none."""
    return (port.get("STATUS") or {}).get(field)


def observed_partner(port: dict) -> str | None:
    """The partner the firmware actually found, from the raw struct dump.

    An all-zero ``asic_id`` is no answer, not an answer: an untrained port still
    carries a zero-filled ``remote_info``, and reading it as a partner makes
    every untrained link look trained to the same imaginary far end. Measured on
    bh-glx6u-37, 320 of 384 ports were in that state, and before the collector's
    own query grew this guard it called all 208 internal links miscabled.

    The result is the same UID form as ETH_PARTNER_UID, so expectation and
    observation compare directly with no lookup.
    """
    info = _blob(port, "eth_status").get("remote_info")
    if not isinstance(info, dict):
        return None
    asic_id = str(info.get("asic_id") or "")
    # Test the digits, not the field's presence: lower, drop the 0x, strip
    # leading zeros, and see whether anything is left.
    if not asic_id.lower().replace("0x", "").lstrip("0"):
        return None
    eth_id = info.get("eth_id")
    if not isinstance(eth_id, int):
        return None
    return f"asic:{asic_id}/eth{eth_id:02d}"


def short_path(record: dict) -> str:
    """A record's PATH without the leading ``glx=<host>/`` segment.

    Every record in one dump shares that segment, so repeating it in a details
    line spends the console's width saying the same hostname four times.
    """
    path = str(record.get("PATH") or record.get("UID") or "?")
    head, sep, tail = path.partition("/")
    return tail if sep and head.startswith("glx=") else path


# ─────────────────────────────────────────────────────────────────────────────
# Emitting checks
# ─────────────────────────────────────────────────────────────────────────────


def check(
    name: str,
    status: str,
    details: str,
    ip: str = "other",
    data: dict | None = None,
    console_visible: bool = True,
) -> dict:
    """One check in the shape diag_runner ingests.

    ``console_visible`` is honoured by the ingesting phase and defaults to true,
    so the triage scripts — which never set it — are unaffected. It is what
    keeps store-only forensics out of the console summary while leaving them in
    the JSON, the same treatment the snapshot phase gives its ``gddr_info_*``
    counters.
    """
    payload = {"name": name, "status": status, "details": details, "ip": ip, "data": data or {}}
    if not console_visible:
        payload["console_visible"] = False
    return payload


def _listing(items: list[str]) -> str:
    """The first few offenders, with a count of the rest."""
    if not items:
        return ""
    shown = ", ".join(items[:MAX_LISTED])
    remaining = len(items) - MAX_LISTED
    return f"{shown}, +{remaining} more" if remaining > 0 else shown


# ─────────────────────────────────────────────────────────────────────────────
# The checks
# ─────────────────────────────────────────────────────────────────────────────


def _inventory(groups: dict[str, list[dict]], envelope: dict, malformed: int) -> dict:
    """What the dump found, against what a 6U Galaxy has.

    Deliberately overlapping the snapshot phase's ``pcie_enum_count``: this
    reaches the chips over a different path (the collector's own PCI
    enumeration and BMC reads, not tt-smi), so the two agreeing is worth more
    than either alone, and the absent-slot reasons the collector records are
    finer-grained than a count.
    """
    counts = {entity: len(groups.get(entity, [])) for entity in ("ubb", "asic", "eth_port", "qsfp_port", "module")}
    expected_ports = ETH_PORTS_PER_ASIC * counts["asic"]

    short: list[str] = []
    if counts["ubb"] != EXPECTED_UBBS:
        short.append(f"ubb {counts['ubb']}/{EXPECTED_UBBS}")
    if counts["asic"] != EXPECTED_ASICS:
        short.append(f"asic {counts['asic']}/{EXPECTED_ASICS}")
    if counts["eth_port"] != expected_ports:
        short.append(f"eth_port {counts['eth_port']}/{expected_ports}")

    # Numbered children are keyed by position precisely so absence has somewhere
    # to live, which on a broken galaxy is the most informative field in the
    # record — it carries the reason the slot came up empty.
    absent: list[str] = []
    for galaxy in groups.get("galaxy", []):
        for num, ref in sorted((galaxy.get("UBBS") or {}).items()):
            if isinstance(ref, dict) and not ref.get("PRESENT"):
                absent.append(f"UBB{num}: {ref.get('ABSENT_REASON') or 'absent'}")
    for ubb in groups.get("ubb", []):
        for loc, ref in sorted((ubb.get("ASICS") or {}).items()):
            if isinstance(ref, dict) and not ref.get("PRESENT"):
                absent.append(f"UBB{ubb.get('UBB_NUM', '?')}/U{loc}: {ref.get('ABSENT_REASON') or 'absent'}")

    galaxies = groups.get("galaxy", [])
    identity = {
        "galaxy_uid": (galaxies[0].get("UID") if galaxies else None),
        "chassis_serial": (galaxies[0].get("CHASSIS_SERIAL") if galaxies else None),
        # The join key for cross-snapshot board history: a UBB's UID outlives
        # the slot and the chassis it currently sits in.
        "ubb_uids": {str(u.get("UBB_NUM")): u.get("UID") for u in groups.get("ubb", [])},
    }

    details = f"{counts['asic']} ASIC(s) on {counts['ubb']} UBB(s), {counts['eth_port']} ETH port(s)"
    if counts["qsfp_port"]:
        details += f", {counts['qsfp_port']} cage(s)"
    if short:
        details += "; short: " + ", ".join(short)
    if absent:
        details += "; absent: " + _listing(absent)
    if malformed:
        details += f"; {malformed} unparseable line(s) in the dump"

    if short or absent:
        status = FAIL
    elif malformed:
        status = WARN
    else:
        status = PASS

    return check(
        "qsfp_inventory",
        status,
        details,
        ip="asic",
        data={
            "counts": counts,
            "expected": {"ubb": EXPECTED_UBBS, "asic": EXPECTED_ASICS, "eth_port": expected_ports},
            "absent": absent,
            "malformed_lines": malformed,
            "identity": identity,
            "snapshot_id": envelope.get("SNAPSHOT_ID"),
            "schema_ver": envelope.get("SCHEMA_VER"),
            "duration_s": envelope.get("DURATION_S"),
            "tool": envelope.get("TOOL"),
        },
    )


def _findings(envelope: dict) -> dict:
    """The collector's own account of what it could not do.

    A finding is how `collect` records a descriptor it could not open, a BMC
    that would not answer, or a cage sweep that fell over — it exits non-zero
    only when nothing at all could be collected, because a recovery script
    calling it on a sick cluster needs whatever was readable. So a clean exit
    code is not a clean run, and this is the field that says so. WARN, not FAIL:
    it is lost coverage, not a statement about the hardware.
    """
    if not envelope:
        return check("qsfp_findings", WARN, "snapshot envelope missing; collector findings unavailable", ip="other")
    findings = [str(f) for f in (envelope.get("FINDINGS") or [])]
    if not findings:
        return check("qsfp_findings", PASS, "collector reported no findings", ip="other")
    return check(
        "qsfp_findings",
        WARN,
        f"{len(findings)} finding(s): " + _listing(findings),
        ip="other",
        data={"findings": findings},
    )


def _collection_failures(records: list[dict]) -> list[dict]:
    """Parts that did not collect, split by whether that is a hardware fault.

    A failure is a state and not an omission here — the record is still emitted,
    with its parents, its PATH and its static topology — so this is the one
    place that answers "which chips stopped answering during the run".
    """
    hard: list[dict] = []
    soft: list[dict] = []
    for record in records:
        entity = str(record.get("SCHEMA") or "")
        # The envelopes describe a run rather than a part and carry no COLLECTION.
        if entity in ("", "snapshot", "cluster_snapshot"):
            continue
        status = collection_status(record)
        entry = {
            "entity": entity,
            "status": status,
            "path": short_path(record),
            "reason": (record.get("COLLECTION") or {}).get("REASON"),
            "error": (record.get("COLLECTION") or {}).get("ERROR"),
        }
        if status in HARD_FAILURES:
            hard.append(entry)
        elif status in SOFT_FAILURES:
            soft.append(entry)

    def _render(entries: list[dict]) -> str:
        mix = Counter(f"{e['entity']} {e['status']}" for e in entries)
        summary = ", ".join(f"{kind} x{n}" for kind, n in sorted(mix.items()))
        first = _listing([f"{e['path']}: {e['reason'] or e['status']}" for e in entries])
        return f"{summary}; {first}" if first else summary

    checks = [
        check(
            "qsfp_collection_failures",
            FAIL if hard else PASS,
            _render(hard) if hard else "every part collected or was harvested",
            ip="other",
            data={"failures": hard},
        ),
        check(
            "qsfp_collection_partial",
            WARN if soft else PASS,
            (
                # Distinct from the check above on purpose: this is coverage the
                # run did not get, and a run that reports it as a fault teaches
                # its readers to ignore it.
                f"{len(soft)} part(s) not fully read (lost coverage, not a fault): {_render(soft)}"
                if soft
                else "no partial or skipped records"
            ),
            ip="other",
            data={"partial": soft},
            console_visible=bool(soft),
        ),
    ]
    return checks


def _board_rev(ubbs: list[dict]) -> dict:
    """Board revision, read independently of tt-smi.

    The revision selects the internal topology table, and the wrong table
    resolves 24 confident wrong inter-UBB partners per galaxy — so a mixed or
    disagreeing read invalidates the partner-based checks below, not just this
    one. ``BOARD_ID_ASIC_AGREEMENT`` is the collector's record of whether all
    eight ASICs on a UBB reported the same board_id, which is the failure mode
    worth catching: board_id can be reset to a uniform default, taking the
    revision with it, and nothing else looks wrong when it happens.
    """
    if not ubbs:
        return check("qsfp_board_rev", SKIP, "no UBB records in the dump", ip="board")

    revs = Counter(str(u.get("BOARD_REV") or "unknown") for u in ubbs)
    disagreeing = [short_path(u) for u in ubbs if u.get("BOARD_ID_ASIC_AGREEMENT") is False]
    single = next(iter(revs)) if len(revs) == 1 else None

    if single is not None and single != "unknown" and not disagreeing:
        return check(
            "qsfp_board_rev",
            PASS,
            f"{single} on all {len(ubbs)} UBB(s)",
            ip="board",
            data={"rev": single, "distribution": dict(revs)},
        )

    reasons = []
    if single is None:
        reasons.append(f"mixed revisions across UBBs: {dict(revs)}")
    elif single == "unknown":
        reasons.append("no UBB reported a board revision")
    if disagreeing:
        reasons.append(f"ASICs disagree on board_id within: {', '.join(disagreeing)}")
    return check(
        "qsfp_board_rev",
        FAIL,
        "; ".join(reasons),
        ip="board",
        data={"rev": single, "distribution": dict(revs), "asic_disagreement": disagreeing},
    )


def _link_training(ports: list[dict], occupancy: dict[str, bool]) -> dict:
    """Every link that should have trained and did not.

    The broadest of the link checks and the one that needs no expectation at
    all: it asks what is down, not what was lost, which is the question to ask
    when the descriptor is itself in doubt.
    """
    judged: list[dict] = []
    empty_cage: list[dict] = []
    unread_cage: list[dict] = []
    for port in ports:
        if not (in_service(port) and port_read(port)):
            continue
        if not behind_a_cage(port):
            judged.append(port)
            continue
        occupied = occupancy.get(str(port.get("QSFP_ENTRY") or ""))
        if occupied is True:
            judged.append(port)
        elif occupied is False:
            empty_cage.append(port)
        else:
            unread_cage.append(port)

    # What was set aside, said out loud: a check that quietly stopped looking at
    # a third of the ports would read as coverage it no longer has.
    aside = []
    if empty_cage:
        aside.append(f"{len(empty_cage)} behind an empty cage")
    if unread_cage:
        aside.append(f"{len(unread_cage)} behind a cage that was not read")
    tail = f" ({', '.join(aside)} not counted)" if aside else ""

    data = {
        "judged": len(judged),
        "skipped_empty_cage": [short_path(p) for p in empty_cage],
        "skipped_unread_cage": [short_path(p) for p in unread_cage],
    }

    if not judged:
        return check(
            "qsfp_link_training",
            SKIP,
            f"no port could be judged{tail or ': none were read'}",
            ip="eth",
            data=data,
        )

    failed = [p for p in judged if port_status(p, "TRAIN_STATUS") != TRAIN_PASS]
    if not failed:
        return check(
            "qsfp_link_training",
            PASS,
            f"{len(judged)}/{len(judged)} cabled and internal ports trained{tail}",
            ip="eth",
            data=data,
        )

    entries = [
        {
            "path": short_path(p),
            "port_type": p.get("PORT_TYPE"),
            "qsfp": p.get("QSFP_NAME"),
            "train_status": port_status(p, "TRAIN_STATUS"),
            "port_status": port_status(p, "PORT_STATUS"),
            "retrain_count": port_status(p, "RETRAIN_COUNT"),
        }
        for p in failed
    ]
    return check(
        "qsfp_link_training",
        FAIL,
        f"{len(judged) - len(failed)}/{len(judged)} cabled and internal ports trained{tail}; down: "
        + _listing([f"{e['path']} ({e['train_status'] or 'no train status'})" for e in entries]),
        ip="eth",
        data={**data, "failures": entries},
    )


def _link_asymmetry(ports: list[dict], by_entry: dict[str, dict]) -> dict:
    """Links that are up at one end and down at the other.

    A link is one thing and its two ends are two records, so the ends can
    disagree — and that asymmetry names the end at fault, which neither end's
    own status does. Only resolvable within this dump: a partner in another
    galaxy has no record here to disagree with.
    """
    pairs: dict[tuple, tuple[dict, dict]] = {}
    for port in ports:
        far = by_entry.get(port.get("ETH_PARTNER_ENTRY") or "")
        if far is None or not port_read(port) or not port_read(far):
            continue
        # `is not` in the tool's SQL is null-safe inequality; in Python plain
        # `!=` already is, and two unread ends never reach here.
        if port_status(port, "LINK_UP") != port_status(far, "LINK_UP"):
            key = tuple(sorted((str(port.get("ENTRY_ID")), str(far.get("ENTRY_ID")))))
            pairs.setdefault(key, (port, far))

    if not pairs:
        return check(
            "qsfp_link_asymmetry",
            PASS,
            "no link disagrees with itself end to end",
            ip="eth",
            console_visible=False,
        )

    entries = [
        {
            "path": short_path(near),
            "link_up": port_status(near, "LINK_UP"),
            "far_path": short_path(far),
            "far_link_up": port_status(far, "LINK_UP"),
        }
        for near, far in pairs.values()
    ]
    return check(
        "qsfp_link_asymmetry",
        FAIL,
        f"{len(entries)} link(s) up at one end and down at the other: "
        + _listing([f"{e['path']}={e['link_up']} vs {e['far_path']}={e['far_link_up']}" for e in entries]),
        ip="eth",
        data={"asymmetric": entries},
    )


def _missing_channel(ports: list[dict], by_entry: dict[str, dict]) -> dict:
    """Channels the topology expected that the firmware never saw.

    A cable missing is a fact about both of its ends, and two rows saying the
    same thing is half the answer wasted — so when both ends are blind the pair
    collapses onto the end whose path sorts first, and each row says which case
    it is. Mirrors the collapse in the tool's `missing_channel` query.
    """
    expecting = [p for p in ports if p.get("ETH_PARTNER_UID") and port_read(p)]
    if not expecting:
        # No descriptor matched this host *and* no internal table applied. Saying
        # PASS here would report coverage this run did not have.
        return check(
            "qsfp_missing_channel",
            SKIP,
            "no port in this dump carries an expected partner",
            ip="eth",
        )

    entries = []
    for port in expecting:
        if observed_partner(port) is not None:
            continue
        far = by_entry.get(port.get("ETH_PARTNER_ENTRY") or "")
        if far is None:
            ends = "this end only; the far end is not in this dump"
        elif not port_read(far):
            ends = "this end only; the far end was not read"
        elif observed_partner(far) is None and far.get("ETH_PARTNER_UID"):
            # Both ends blind: one row for the pair, on the lower path.
            if str(port.get("PATH") or "") > str(far.get("PATH") or ""):
                continue
            ends = "both ends"
        else:
            ends = "this end only"
        entries.append(
            {
                "path": short_path(port),
                "ends": ends,
                "qsfp": port.get("QSFP_NAME"),
                "expected": short_path(far) if far is not None else port.get("ETH_PARTNER_UID"),
                "train_status": port_status(port, "TRAIN_STATUS"),
            }
        )

    if not entries:
        return check(
            "qsfp_missing_channel",
            PASS,
            f"all {len(expecting)} expected channel(s) were seen by firmware",
            ip="eth",
        )
    return check(
        "qsfp_missing_channel",
        FAIL,
        f"{len(entries)} of {len(expecting)} expected channel(s) never came up: "
        + _listing([f"{e['path']} -> {e['expected']} ({e['ends']})" for e in entries]),
        ip="eth",
        data={"missing": entries, "expected_total": len(expecting)},
    )


def _miscabled(ports: list[dict], by_uid: dict[str, dict], expectations_complete: bool) -> dict:
    """Channels joining two ends of this dump that nothing asked for.

    Two cases, and they are not equally strong. *Wrong end* — an expectation
    exists and the firmware found a different one — is miscabling, full stop.
    *Undescribed* — the firmware found an end nobody expected — is only a
    finding when an expectation was available to be missing: without a matched
    factory descriptor the internal topology table covers the soldered links and
    nothing describes the cabled ones, so every cabled link inside the chassis
    would otherwise be reported as a surprise. Hence the split on
    ``expectations_complete``.
    """
    wrong_end = []
    undescribed = []
    for port in ports:
        observed = observed_partner(port)
        if observed is None or observed not in by_uid:
            continue
        expected = port.get("ETH_PARTNER_UID")
        if expected is None:
            undescribed.append({"path": short_path(port), "observed": short_path(by_uid[observed])})
        elif observed != expected:
            wrong_end.append(
                {
                    "path": short_path(port),
                    "qsfp": port.get("QSFP_NAME"),
                    "expected": expected,
                    "observed": short_path(by_uid[observed]),
                }
            )

    data = {"wrong_end": wrong_end, "undescribed": undescribed, "expectations_complete": expectations_complete}
    if wrong_end:
        return check(
            "qsfp_miscabled",
            FAIL,
            f"{len(wrong_end)} channel(s) trained to the wrong end: "
            + _listing([f"{e['path']} -> {e['observed']} (expected {e['expected']})" for e in wrong_end]),
            ip="eth",
            data=data,
        )
    if undescribed and expectations_complete:
        return check(
            "qsfp_miscabled",
            WARN,
            f"{len(undescribed)} channel(s) trained where nothing was expected: "
            + _listing([f"{e['path']} -> {e['observed']}" for e in undescribed]),
            ip="eth",
            data=data,
        )
    details = "every channel trained to the end that was expected"
    if undescribed:
        details = (
            f"no channel trained to the wrong end; {len(undescribed)} trained where this dump had no "
            f"expectation (no factory descriptor matched this host, so the cabled links are undescribed)"
        )
    return check("qsfp_miscabled", PASS, details, ip="eth", data=data)


def _partner_disagreement(ports: list[dict], by_uid: dict[str, dict]) -> dict:
    """Ends that contradict each other about who they are talking to.

    Stronger than miscabling and needs no expectation at all: A says it is
    talking to B while B says it is talking to someone else. Both ends were read
    independently, so this is the hardware contradicting itself.
    """
    entries = []
    for port in ports:
        observed = observed_partner(port)
        far = by_uid.get(observed or "")
        if far is None or not port_read(far):
            continue
        far_observed = observed_partner(far)
        if far_observed != port.get("UID"):
            entries.append(
                {
                    "path": short_path(port),
                    "far_path": short_path(far),
                    "far_observed": far_observed,
                }
            )

    if not entries:
        return check(
            "qsfp_partner_disagreement",
            PASS,
            "both ends of every observed link name each other",
            ip="eth",
            console_visible=False,
        )
    return check(
        "qsfp_partner_disagreement",
        FAIL,
        f"{len(entries)} end(s) disagree about their partner: "
        + _listing([f"{e['path']} says {e['far_path']}, which says {e['far_observed'] or 'nothing'}" for e in entries]),
        ip="eth",
        data={"disagreements": entries},
    )


def _outside_channel(ports: list[dict], by_uid: dict[str, dict]) -> dict:
    """Trained channels leading to hardware this dump did not read.

    Store-only, and that is the point: the health check collects one galaxy, so
    every inter-galaxy cable in a working rack lands here. It is a count of how
    far the chassis can see, not a fault — the same rows only become a finding
    once several galaxies have been merged, which this phase deliberately does
    not do.
    """
    outside = [p for p in ports if (o := observed_partner(p)) is not None and o not in by_uid]
    return check(
        "qsfp_outside_channel",
        PASS,
        f"{len(outside)} trained channel(s) lead outside this dump "
        f"(expected: one galaxy was collected, so every inter-galaxy cable does)",
        ip="eth",
        data={
            "count": len(outside),
            "ports": [{"path": short_path(p), "observed": observed_partner(p)} for p in outside],
        },
        console_visible=False,
    )


def _cage_gaps(cages: list[dict]) -> dict:
    """Cages that are empty, or hold a cable nobody described.

    The one question whose answer is a cage rather than a channel: a cage with
    nothing in it has no channel to report. WARN rather than FAIL — a partly
    populated system is normal and which cages should be filled is a deployment
    fact, not a board one.
    """
    if not cages:
        return check("qsfp_cage_gaps", SKIP, "no cage records (the QSFP sweep did not run)", ip="eth")
    # A cage that was not read is not empty; the collection checks have it.
    readable = [c for c in cages if collection_status(c) not in CAGE_UNREAD]
    if not readable:
        return check("qsfp_cage_gaps", SKIP, f"none of the {len(cages)} cage(s) were read", ip="eth")
    if not any(c.get("QSFP_PARTNER_UID") for c in readable):
        return check(
            "qsfp_cage_gaps",
            SKIP,
            f"no cage carries an expected partner, so an empty one cannot be told from an "
            f"unused one ({len(readable)} cage(s) read)",
            ip="eth",
        )

    entries = []
    for cage in readable:
        present, expected = bool(cage.get("PRESENT")), cage.get("QSFP_PARTNER_UID")
        if present and expected:
            continue
        if not present and expected:
            gap = "empty, a cable was expected"
        elif not present:
            gap = "empty, and nothing expected"
        else:
            gap = "cabled, and nothing expected"
        entries.append({"path": short_path(cage), "qsfp": cage.get("QSFP_NAME"), "gap": gap})

    if not entries:
        return check("qsfp_cage_gaps", PASS, f"all {len(readable)} cage(s) match the expected cabling", ip="eth")
    return check(
        "qsfp_cage_gaps",
        WARN,
        f"{len(entries)} of {len(readable)} cage(s) do not match the expected cabling: "
        + _listing([f"{e['qsfp'] or e['path']} ({e['gap']})" for e in entries]),
        ip="eth",
        data={"gaps": entries},
    )


def _eth_counters(ports: list[dict]) -> dict:
    """Per-port error and retrain counters, recorded and never alerted on.

    Forensics and regression baselines, in the same spirit as the snapshot
    phase's ``gddr_info_*`` checks: a threshold here would be a policy decision
    nobody has made, and a counter that WARNs on its first non-zero reading
    teaches its readers to ignore it. The top offenders are named so the numbers
    are usable from the report without reopening the dump.
    """
    read = [p for p in ports if port_read(p) and p.get("STATUS")]
    if not read:
        return check("qsfp_eth_counters", SKIP, "no port carries a STATUS block", ip="eth", console_visible=False)

    def _total(field: str) -> int:
        return sum(v for p in read if isinstance(v := port_status(p, field), int))

    totals = {
        "retrain_count": _total("RETRAIN_COUNT"),
        "corr_cw": _total("CORR_CW"),
        "uncorr_cw": _total("UNCORR_CW"),
    }
    erroring = [
        {"path": short_path(p), "uncorr_cw": v}
        for p in read
        if isinstance(v := port_status(p, "UNCORR_CW"), int) and v > 0
    ]
    worst = sorted(erroring, key=lambda e: e["uncorr_cw"], reverse=True)[:MAX_LISTED]
    details = (
        f"{len(read)} port(s): retrain={totals['retrain_count']} "
        f"corr_cw={totals['corr_cw']} uncorr_cw={totals['uncorr_cw']}"
    )
    if worst:
        details += "; worst uncorr: " + ", ".join(f"{e['path']}={e['uncorr_cw']}" for e in worst)
    return check(
        "qsfp_eth_counters",
        PASS,
        details,
        ip="eth",
        data={"totals": totals, "ports_counted": len(read), "worst_uncorr_cw": worst},
        console_visible=False,
    )


def _modules(cages: list[dict], modules: list[dict]) -> dict:
    """The transceiver inventory: what is plugged in, and where.

    Store-only, like the snapshot phase's FRU capture. A module's identity is
    the part and its location is a field, which is what lets "this module has
    thrown errors in three different cages" be answered later from stored runs.
    """
    if not cages and not modules:
        return check("qsfp_modules", SKIP, "the QSFP sweep did not run", ip="eth", console_visible=False)
    populated = sum(1 for c in cages if c.get("PRESENT"))
    vendors = Counter(str(m.get("VENDOR_PN") or "unknown") for m in modules)
    inventory = [
        {
            "path": short_path(m),
            "qsfp": m.get("QSFP_NAME"),
            "vendor_pn": m.get("VENDOR_PN"),
            "vendor_sn": m.get("VENDOR_SN"),
            "length_m": m.get("LENGTH_M"),
        }
        for m in modules
    ]
    return check(
        "qsfp_modules",
        PASS,
        f"{len(modules)} module(s) in {populated}/{len(cages)} populated cage(s); "
        f"types: {', '.join(f'{pn} x{n}' for pn, n in sorted(vendors.items())) or 'none'}",
        ip="eth",
        data={"modules": inventory, "populated_cages": populated, "cages": len(cages), "vendors": dict(vendors)},
        console_visible=False,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Assembly
# ─────────────────────────────────────────────────────────────────────────────


def descriptor_matched(envelope: dict) -> bool:
    """Whether a factory descriptor covering this host was actually read.

    Both halves matter and the collector records them separately: a descriptor
    that was supplied but does not name this host leaves the cage-attached links
    with no expected partner, exactly as if none had been supplied at all. The
    checks that compare against an expectation consult this rather than assuming
    coverage they may not have.
    """
    source = (envelope.get("SOURCES") or {}).get("FACTORY_SYSTEM_DESCRIPTOR")
    if not isinstance(source, dict):
        return False
    return bool(source.get("PRESENT")) and bool(source.get("MATCHED_HOSTNAME"))


def summarize(records: list[dict], malformed: int = 0) -> list[dict]:
    """Every check derivable from one galaxy's dump, in console order."""
    groups = group_by_schema(records)
    envelopes = groups.get("snapshot", [])
    envelope = envelopes[0] if envelopes else {}
    ports = groups.get("eth_port", [])
    cages = groups.get("qsfp_port", [])

    if not envelope and not ports:
        return [
            check(
                "qsfp_inventory",
                FAIL,
                f"the dump holds no snapshot envelope and no ETH ports ({len(records)} record(s), "
                f"{malformed} unparseable line(s))",
                ip="asic",
                data={"records": len(records), "malformed_lines": malformed},
            )
        ]

    # ENTRY_ID joins within a snapshot, UID joins across them; the partner
    # pointers use the first and the firmware's own answer resolves through the
    # second, so both indexes are needed.
    occupancy = cage_occupancy(cages)
    by_entry = {str(p["ENTRY_ID"]): p for p in ports if p.get("ENTRY_ID")}
    by_uid = {str(p["UID"]): p for p in ports if p.get("UID")}
    complete = descriptor_matched(envelope)

    checks = [
        _inventory(groups, envelope, malformed),
        _board_rev(groups.get("ubb", [])),
        _findings(envelope),
        *_collection_failures(records),
        _link_training(ports, occupancy),
        _link_asymmetry(ports, by_entry),
        _missing_channel(ports, by_entry),
        _miscabled(ports, by_uid, complete),
        _partner_disagreement(ports, by_uid),
        _outside_channel(ports, by_uid),
        _cage_gaps(cages),
        _eth_counters(ports),
        _modules(cages, groups.get("module", [])),
    ]
    return checks


def build_report(records: list[dict], malformed: int = 0) -> dict:
    """The full payload, in the shape the health check ingests."""
    return {"tool": TOOL_NAME, "version": INGEST_VERSION, "checks": summarize(records, malformed)}


def rollup(checks: list[dict]) -> str:
    """FAIL > WARN > PASS > SKIP over a set of checks.

    For the standalone text report only. The health check computes the phase's
    verdict itself, from the same precedence, so that a run's status is decided
    in one place whatever produced the findings.
    """
    statuses = {entry.get("status") for entry in checks}
    for status in (FAIL, WARN, PASS):
        if status in statuses:
            return status
    return SKIP


def render(report: dict, source: str = "") -> str:
    """The same findings as readable text, for the run's log directory."""
    lines = [f"{report['tool']} ingest v{report['version']}"]
    if source:
        lines.append(f"dump: {source}")
    lines.append("")
    for entry in report["checks"]:
        lines.append(f"  {entry['name']:34} {entry['status']:5}  {entry['details']}")
    lines += ["", f"  {'rollup':34} {rollup(report['checks'])}"]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Derive health-check findings from a tt-bh-glx-cluster-debug JSONL dump."
    )
    parser.add_argument("dump", help="The .jsonl file written by `tt-bh-glx-cluster-debug collect`")
    parser.add_argument("--json", dest="json_out", help="Write the checks JSON here (default: stdout)")
    parser.add_argument("-o", dest="text_out", help="Write the human-readable report here")
    args = parser.parse_args()

    try:
        records, malformed = load_dump(args.dump)
    except OSError as err:
        print(f"could not read {args.dump}: {err}", file=sys.stderr)
        return 1

    report = build_report(records, malformed)
    text = render(report, source=args.dump)
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(report, indent=2, default=str))
    else:
        print(json.dumps(report, indent=2, default=str))
    if args.text_out:
        Path(args.text_out).write_text(text)
    else:
        print(text, file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
