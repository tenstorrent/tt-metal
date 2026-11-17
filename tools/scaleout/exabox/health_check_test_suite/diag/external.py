# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Ingestion of findings written by tools outside this suite.

The triage scripts and the QSFP ingest both emit
``{"checks": [{"name", "status", "details", "ip", "data"}]}``; this turns each
entry into a Check.
"""
from __future__ import annotations

from .model import FAIL, IP_ORDER, PASS, SKIP, WARN, Check

EXTERNAL_STATUSES = (PASS, WARN, FAIL, SKIP)


def normalize_external_check(
    payload: dict,
    *,
    prefix: str,
    hold_fail_at: str | None = None,
    hold_warn_at: str | None = None,
    gating_flag: str = "",
) -> Check:
    """Turn one entry of an external tool's JSON into a Check, defensively.

    Shared by every phase that ingests findings produced outside this suite — the
    triage scripts and the QSFP tests' dump — because they write one shape
    (see ``kmd_triage/triage_json.sh``) and the defences below are properties of
    ingesting *any* of it, not of either producer.

    Two normalizations are unconditional, both because the report is consumed by
    code that doesn't know these phases exist:

    * Names get a per-phase prefix. CHECK_CATEGORY, EXCLUDED_CHECKS,
      ACKNOWLEDGED_CHECKS and _find_check() in the analyzer are all keyed on the
      bare check name with no phase qualifier, so an unprefixed ``pcie_gen``
      from an external tool would silently inherit that check's routing.
    * An unrecognised status becomes WARN. Anything outside PASS/WARN/FAIL/SKIP
      falls through the analyzer's SEVERITY map to UNKNOWN and won't validate
      against the CheckStatus enum, so a typo would land as a junk CSV row.

    ``hold_fail_at`` and ``hold_warn_at`` are how a phase says how much authority
    its tool has yet earned: each names the status to record that finding as, or
    is None to leave it alone. They are applied to the tool's own verdict and
    never to each other's output, so holding a FAIL at PASS is one step and not
    a FAIL quietly sliding down through WARN. Both annotate the details with
    what was done, because a status that reads lower than the finding beneath it
    sounds is otherwise indistinguishable from a bug.

    An `ip` outside IP_ORDER is folded to "other" for the same reason:
    print_phase_summary() iterates IP_ORDER, so an unrecognised group would
    reach the JSON and the CSV but never appear in the console summary.

    ``console_visible`` is optional and defaults to true, so a producer that
    never sets it (the triage scripts) is unaffected; a producer that does can
    keep store-only forensics in the JSON without crowding the console, which is
    how the snapshot phase treats its own ``gddr_info_*`` counters.
    """
    name = str(payload.get("name") or "unnamed")
    if not name.startswith(prefix):
        name = f"{prefix}{name}"
    details = str(payload.get("details") or "")
    status = str(payload.get("status") or "").upper()
    if status not in EXTERNAL_STATUSES:
        details = f"[unrecognised status {payload.get('status')!r}] {details}".rstrip()
        status = WARN
    # On the tool's own verdict, so the two are alternatives rather than a chain.
    if status == FAIL and hold_fail_at:
        flag = f", {gating_flag} off" if gating_flag else ""
        details = f"[advisory: FAIL recorded as {hold_fail_at}{flag}] {details}".rstrip()
        status = hold_fail_at
    elif status == WARN and hold_warn_at:
        details = f"[advisory: WARN recorded as {hold_warn_at}] {details}".rstrip()
        status = hold_warn_at
    data = payload.get("data")
    ip = str(payload.get("ip") or "other")
    return Check(
        name=name,
        status=status,
        details=details,
        data=data if isinstance(data, dict) else {},
        ip=ip if ip in IP_ORDER else "other",
        console_visible=payload.get("console_visible", True) is not False,
    )
