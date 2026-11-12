# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Phase 1 checks: everything judged from a single tt-smi snapshot.

Also run after every pre-test reset, against a fresh snapshot (see resets.py).
"""
from __future__ import annotations

from collections import Counter, defaultdict

from .galaxy import (
    BOARD_ID_PREFIX_REVAB,
    BOARD_ID_PREFIX_REVC,
    BOARD_REV_EXPECTATIONS,
    ETH_EXAMAX_BY_ASIC,
    ETH_INTERNAL_BY_ASIC,
    EXPECTED_CHIP_COUNT,
    TRAY_TO_UBB_NIBBLE,
    UBB_NIBBLE_TO_TRAY,
    expected_bdf,
    format_bdf,
    parse_bdf_physical,
)
from .model import FAIL, PASS, SKIP, WARN, Check, Phase

# GDDR train+BIST bitmask layout — tt-system-firmware telemetry.h:103-118.
# Exposed by tt-smi -s as smbus_telem.DDR_STATUS (legacy WH name; on BH this is
# really TAG_GDDR_STATUS, tag 22). All 8 channels packed into one u32.
#   bit [2*ch    ] = train_done   (ch=0..7)
#   bit [2*ch + 1] = error
#   bit [16 + 2*ch    ] = bist_done
#   bit [16 + 2*ch + 1] = bist_failed
# Fully-trained + BIST-passed = 0x55555555.


def detect_board_rev(devs: list[dict]) -> tuple[str | None, dict[str, list[str]]]:
    """Classify chips by board ID prefix → revision.

    Returns (rev_label, per_rev_bdfs).
    rev_label is 'RevA/B', 'RevC', or None when chips are mixed or any chip
    has an unrecognised board_id (callers should treat None as indeterminate
    and surface a FAIL on the board_rev check).
    per_rev_bdfs maps each observed bucket label (including 'unknown:<prefix>')
    to its list of BDFs, for diagnostic output.
    """
    per_rev: dict[str, list[str]] = defaultdict(list)
    for d in devs:
        bid = (d.get("board_info") or {}).get("board_id") or ""
        bdf = (d.get("board_info") or {}).get("bus_id", "?")
        if bid.startswith(BOARD_ID_PREFIX_REVAB):
            per_rev["RevA/B"].append(bdf)
        elif bid.startswith(BOARD_ID_PREFIX_REVC):
            per_rev["RevC"].append(bdf)
        else:
            per_rev[f"unknown:{bid[:8] or '<empty>'}"].append(bdf)
    buckets = list(per_rev.keys())
    if len(buckets) == 1 and buckets[0] in BOARD_REV_EXPECTATIONS:
        return buckets[0], dict(per_rev)
    return None, dict(per_rev)


def validate_snapshot(snap: dict, phase: Phase) -> str | None:
    """Run all snapshot checks. Returns the detected board revision
    ('RevA/B', 'RevC') or None if indeterminate, so the caller can record it
    on the report.
    """
    devs = snap.get("device_info") or []

    # 1. Enumeration count — on short count, identify which physical slots didn't
    # show up (set diff against the full 32-BDF Galaxy 6U topology). Helps ops
    # localise the missing chip(s) without manual cross-referencing.
    n = len(devs)
    actual_bdfs = sorted((d.get("board_info") or {}).get("bus_id", "") for d in devs)
    all_expected_bdfs = sorted(expected_bdf(t, a) for t in range(1, 5) for a in range(1, 9))
    missing_bdfs_enum = sorted(set(all_expected_bdfs) - set(actual_bdfs))
    enum_details = f"{n}/{EXPECTED_CHIP_COUNT} chips enumerated"
    if missing_bdfs_enum:
        enum_details += "; missing: " + ", ".join(format_bdf(b) for b in missing_bdfs_enum)
    phase.add(
        Check(
            name="pcie_enum_count",
            status=PASS if n == EXPECTED_CHIP_COUNT else FAIL,
            details=enum_details,
            data={
                "actual": n,
                "expected": EXPECTED_CHIP_COUNT,
                "missing_bdfs": missing_bdfs_enum,
                "actual_bdfs": actual_bdfs,
            },
            ip="pcie",
        )
    )
    if n == 0:
        return None  # nothing else we can check

    # 1b. Board revision — gates GDDR speed + PCIe gen expectations below.
    # RevA/B (board_id 00000471xxxxxxxx): 14G GDDR, Gen4 PCIe.
    # RevC   (board_id 00000473xxxxxxxx): 16G GDDR, Gen5 PCIe.
    # Mixed-rev or unknown-prefix board_ids => FAIL here, downstream
    # rev-dependent checks => SKIP. Spec: Confluence SYS-4055.
    rev, per_rev_bdfs = detect_board_rev(devs)
    rev_dist = {k: len(v) for k, v in per_rev_bdfs.items()}
    if rev is not None:
        prefix = BOARD_REV_EXPECTATIONS[rev]["board_id_prefix"]
        phase.add(
            Check(
                name="board_rev",
                status=PASS,
                details=f"Rev: {rev} ({rev_dist[rev]}/{n} chips, board_id prefix {prefix})",
                data={
                    "rev": rev,
                    "distribution": rev_dist,
                    "per_rev_bdfs": per_rev_bdfs,
                    "expectations": BOARD_REV_EXPECTATIONS[rev],
                },
                ip="board",
            )
        )
    else:
        phase.add(
            Check(
                name="board_rev",
                status=FAIL,
                details=f"indeterminate board revision: {rev_dist}",
                data={"rev": None, "distribution": rev_dist, "per_rev_bdfs": per_rev_bdfs},
                ip="board",
            )
        )

    # 1c. BDF -> physical location mapping. Visible topology readout for the
    # enumerated chips, so any BDF appearing in subsequent failure details can
    # be correlated to its UBB/U position without referring to docs.
    by_ubb_bdfs: dict[int, list[tuple[str, int]]] = defaultdict(list)
    for d in devs:
        bdf = (d.get("board_info") or {}).get("bus_id") or ""
        pos = parse_bdf_physical(bdf)
        if pos is None:
            continue
        tray, asic = pos
        by_ubb_bdfs[tray].append((bdf, asic))
    topo_lines: list[str] = []
    for tray in sorted(by_ubb_bdfs):
        nibble = TRAY_TO_UBB_NIBBLE.get(tray, 0)
        chips = sorted(by_ubb_bdfs[tray], key=lambda x: x[1])
        rendered = "  ".join(f"{bdf}=U{a}" for bdf, a in chips)
        topo_lines.append(f"UBB{tray} (bus {nibble:X}X): {rendered}")
    phase.add(
        Check(
            name="bdf_topology",
            status=PASS,
            details=f"{n} chips mapped\n" + "\n".join(topo_lines),
            data={"per_ubb": {str(t): [bdf for bdf, _ in v] for t, v in by_ubb_bdfs.items()}},
            ip="board",
        )
    )

    # 2. dram_status
    dram_ok = [bool(d.get("board_info", {}).get("dram_status")) for d in devs]
    bad = [d["board_info"]["bus_id"] for d, ok in zip(devs, dram_ok) if not ok]
    phase.add(
        Check(
            name="dram_status",
            status=PASS if all(dram_ok) else FAIL,
            details=f"{sum(dram_ok)}/{n} dram_status=True"
            + (f"; failed: [{', '.join(format_bdf(b) for b in bad)}]" if bad else ""),
            data={"failed_chips": bad},
            ip="gddr",
        )
    )

    # 3. ENABLED_GDDR uniform == 0xff
    gddr_masks = [d.get("smbus_telem", {}).get("ENABLED_GDDR") for d in devs]
    gddr_bad = [(d["board_info"]["bus_id"], m) for d, m in zip(devs, gddr_masks) if m != "0xff"]
    phase.add(
        Check(
            name="enabled_gddr_full",
            status=PASS if not gddr_bad else FAIL,
            details=f"{n - len(gddr_bad)}/{n} chips have ENABLED_GDDR=0xff"
            + (f"; partial: [{', '.join(f'{format_bdf(b)}={m}' for b, m in gddr_bad[:5])}]" if gddr_bad else ""),
            data={"partial_chips": gddr_bad},
            ip="gddr",
        )
    )

    # 4. HARVESTING_STATE: 0x0 or 0x1 valid; other values are a failure.
    # Hidden from console (kept in JSON for forensics).
    HARV_OK = ("0x0", "0", "0x1", "1")
    harv = [d.get("smbus_telem", {}).get("HARVESTING_STATE") for d in devs]
    harv_bad = [(d["board_info"]["bus_id"], h) for d, h in zip(devs, harv) if h not in (None, *HARV_OK)]
    phase.add(
        Check(
            name="harvesting_state",
            status=PASS if not harv_bad else FAIL,
            details=f"{n - len(harv_bad)}/{n} chips have HARVESTING_STATE in {{0,1}}",
            data={"harvested_chips": harv_bad, "distribution": dict(Counter(harv))},
            ip="asic",
            console_visible=False,
        )
    )

    # 5. ASIC_LOCATION per UBB completeness (1..8 each)
    by_ubb = defaultdict(set)
    for d in devs:
        bus = d["board_info"]["bus_id"].split(":")[1]
        tray = UBB_NIBBLE_TO_TRAY.get(int(bus[0], 16))
        if tray is None:
            continue
        try:
            loc = int(d["smbus_telem"]["ASIC_LOCATION"], 16)
        except (KeyError, TypeError, ValueError):
            continue
        by_ubb[tray].add(loc)
    expected = set(range(1, 9))
    missing = {tray: sorted(expected - locs) for tray, locs in by_ubb.items() if expected - locs}
    missing_trays = sorted(set(range(1, 5)) - set(by_ubb.keys()))
    asic_loc_ok = not missing and not missing_trays
    # Build expected-BDF list for every missing chip — entire-tray-missing means all 8 asics.
    missing_bdfs: list[str] = []
    for tray in missing_trays:
        missing_bdfs.extend(expected_bdf(tray, a) for a in range(1, 9))
    for tray, asics in missing.items():
        missing_bdfs.extend(expected_bdf(tray, a) for a in asics)
    details = []
    if missing_trays:
        details.append(f"missing UBBs: {missing_trays}")
    if missing:
        details.append(f"missing ASICs: {missing}")
    if missing_bdfs:
        details.append(f"expected BDFs: {missing_bdfs}")
    phase.add(
        Check(
            name="asic_location_per_ubb",
            status=PASS if asic_loc_ok else FAIL,
            details="all UBBs complete" if asic_loc_ok else "; ".join(details),
            data={
                "per_ubb": {t: sorted(s) for t, s in by_ubb.items()},
                "missing": missing,
                "missing_trays": missing_trays,
                "missing_bdfs": missing_bdfs,
            },
            ip="asic",
        )
    )

    # 6. Physical BDF ↔ firmware ASIC_LOCATION agreement
    mismatches = []
    for d in devs:
        bus = d["board_info"]["bus_id"].split(":")[1]
        phys_asic = int(bus[1], 16)
        try:
            fw_asic = int(d["smbus_telem"]["ASIC_LOCATION"], 16)
        except (KeyError, TypeError, ValueError):
            continue
        if phys_asic != fw_asic:
            mismatches.append(
                {
                    "bdf": d["board_info"]["bus_id"],
                    "physical_asic": phys_asic,
                    "fw_asic_location": fw_asic,
                }
            )

    def _fmt_mismatch(m: dict) -> str:
        return f"{format_bdf(m['bdf'])} phys=U{m['physical_asic']} " f"fw=U{m['fw_asic_location']}"

    phase.add(
        Check(
            name="physical_vs_fw_location",
            status=PASS if not mismatches else FAIL,
            details=f"{n - len(mismatches)}/{n} chips match"
            + (f"; mismatches: [{'; '.join(_fmt_mismatch(m) for m in mismatches[:3])}]" if mismatches else ""),
            data={"mismatches": mismatches},
            ip="asic",
        )
    )

    # 7. Firmware versions consistent across chips (warn-only per ticket).
    # gddr_fw is the M-RISC (GDDR memory controller) firmware — 5.1+ only.
    # dm_app_fw / dm_bl_fw don't apply on Galaxy; always report SKIP (kept in JSON for completeness).
    FW_FIELDS_CHECKED = ("fw_bundle_version", "cm_fw", "eth_fw", "gddr_fw")
    FW_FIELDS_NA_ON_GALAXY = ("dm_app_fw", "dm_bl_fw")
    for fw_field in FW_FIELDS_CHECKED:
        vals = [d.get("firmwares", {}).get(fw_field) for d in devs]
        # If the field is absent on every chip, gate as SKIP (older tt-smi).
        if all(v is None for v in vals):
            phase.add(
                Check(
                    name=f"{fw_field}_consistent",
                    status=SKIP,
                    details=f"{fw_field} not present in snapshot (requires tt-smi 5.1+)",
                    data={},
                    ip="fw",
                )
            )
            continue
        unique = Counter(vals)
        if len(unique) == 1:
            ((only_val, _),) = unique.most_common(1)
            phase.add(
                Check(
                    name=f"{fw_field}_consistent",
                    status=PASS,
                    details=f"all chips: {only_val}",
                    data={"value": only_val},
                    ip="fw",
                )
            )
        else:
            phase.add(
                Check(
                    name=f"{fw_field}_consistent",
                    status=WARN,
                    details=f"mismatch across chips: {dict(unique)}",
                    data={"distribution": dict(unique)},
                    ip="fw",
                )
            )
    for fw_field in FW_FIELDS_NA_ON_GALAXY:
        vals = [d.get("firmwares", {}).get(fw_field) for d in devs]
        unique = Counter(vals)
        phase.add(
            Check(
                name=f"{fw_field}_consistent",
                status=SKIP,
                details="not applicable to Galaxy",
                data={"distribution": dict(unique)},
                ip="fw",
                console_visible=False,
            )
        )

    # 8. GDDR per-channel training + BIST decoded from smbus_telem.DDR_STATUS.
    # Layout in header comment at top of file.
    train_failures: list[dict] = []
    bist_failures: list[dict] = []
    for d in devs:
        bdf = d["board_info"]["bus_id"]
        status_str = d.get("smbus_telem", {}).get("DDR_STATUS")
        if status_str is None:
            train_failures.append({"bdf": bdf, "reason": "DDR_STATUS missing"})
            continue
        try:
            status = int(status_str, 16)
        except (ValueError, TypeError):
            train_failures.append({"bdf": bdf, "reason": f"DDR_STATUS unparseable: {status_str!r}"})
            continue
        for ch in range(8):
            train_done = (status >> (ch * 2)) & 1
            err = (status >> (ch * 2 + 1)) & 1
            bist_done = (status >> (16 + ch * 2)) & 1
            bist_fail = (status >> (17 + ch * 2)) & 1
            if not train_done or err:
                train_failures.append({"bdf": bdf, "channel": ch, "train_done": train_done, "error": err})
            if not bist_done or bist_fail:
                bist_failures.append({"bdf": bdf, "channel": ch, "bist_done": bist_done, "bist_failed": bist_fail})

    def _fmt_chan_fail(f: dict) -> str:
        bdf = format_bdf(f.get("bdf", "?"))
        if "reason" in f:
            return f"{bdf}: {f['reason']}"
        return (
            f"{bdf} ch={f.get('channel','?')} " f"train_done={f.get('train_done','?')} err={f.get('error','?')}"
            if "train_done" in f
            else f"{bdf} ch={f.get('channel','?')} "
            f"bist_done={f.get('bist_done','?')} bist_failed={f.get('bist_failed','?')}"
        )

    total_channels = n * 8
    phase.add(
        Check(
            name="gddr_training_per_channel",
            status=PASS if not train_failures else FAIL,
            details=f"{total_channels - len(train_failures)}/{total_channels} channels trained"
            + (
                f"; first failures: [{'; '.join(_fmt_chan_fail(f) for f in train_failures[:3])}]"
                if train_failures
                else ""
            ),
            data={"failures": train_failures},
            ip="gddr",
        )
    )
    phase.add(
        Check(
            name="gddr_bist_per_channel",
            status=PASS if not bist_failures else FAIL,
            details=f"{total_channels - len(bist_failures)}/{total_channels} channels BIST passed"
            + (
                f"; first failures: [{'; '.join(_fmt_chan_fail(f) for f in bist_failures[:3])}]"
                if bist_failures
                else ""
            ),
            data={"failures": bist_failures},
            ip="gddr",
        )
    )

    # 9. PCIe lane width — U6 chips (the x8 host-PCIe-attached ASIC on each UBB)
    # should be at x8; all other ASICs should be x1. Lane mismatch is a hard fail.
    #
    # Uses BDF-derived physical position rather than FW ASIC_LOCATION so this
    # check is robust to FW telemetry bugs (e.g., multiple chips wrongly
    # reporting the same ASIC_LOCATION). Any such telemetry issue is surfaced
    # by physical_vs_fw_location; this check evaluates against physical reality.
    U6_ASIC = 0x6

    def _phys_asic_loc(d: dict) -> int | None:
        pos = parse_bdf_physical((d.get("board_info") or {}).get("bus_id", ""))
        return pos[1] if pos is not None else None

    def _is_u6(d: dict) -> bool:
        return _phys_asic_loc(d) == U6_ASIC

    lane_mismatches = []
    for d in devs:
        loc = _phys_asic_loc(d)
        width = str(d["board_info"].get("pcie_width"))
        expected_width = "8" if loc == U6_ASIC else "1"
        if width != expected_width:
            lane_mismatches.append(
                {
                    "bdf": d["board_info"]["bus_id"],
                    "physical_asic": loc,
                    "actual_width": width,
                    "expected_width": expected_width,
                }
            )

    def _fmt_lane_mm(m: dict) -> str:
        return f"{format_bdf(m['bdf'])} " f"x{m['actual_width']} (expected x{m['expected_width']})"

    phase.add(
        Check(
            name="pcie_lane_width",
            status=PASS if not lane_mismatches else FAIL,
            details=(
                f"{n - len(lane_mismatches)}/{n} chips at expected lane width "
                f"(x8 on physical U6, x1 elsewhere)"
                + (
                    f"; mismatches: [{'; '.join(_fmt_lane_mm(m) for m in lane_mismatches[:3])}]"
                    if lane_mismatches
                    else ""
                )
            ),
            data={"mismatches": lane_mismatches},
            ip="pcie",
        )
    )

    # 10. PCIe gen — U6 chips should all be at the rev's expected gen
    # (Gen4 on RevA/B, Gen5 on RevC). Known bug: chips occasionally train down
    # to Gen1; flag as WARN so it surfaces without failing the run. If the
    # board rev is indeterminate we SKIP — no safe baseline to check against.
    # Selects U6 chips by BDF-derived physical position (not FW ASIC_LOCATION).
    u6_chips = [d for d in devs if _is_u6(d)]
    per_chip_gen = [(d["board_info"]["bus_id"], d["board_info"].get("pcie_speed")) for d in u6_chips]
    if rev is None:
        phase.add(
            Check(
                name="pcie_gen",
                status=SKIP,
                details="board_rev indeterminate; cannot pick expected gen (see board_rev)",
                data={"per_chip": dict(per_chip_gen), "expected": None, "rev": None},
                ip="pcie",
            )
        )
    else:
        expected_gen = BOARD_REV_EXPECTATIONS[rev]["pcie_gen"]

        # WARN only on chips trained *below* the rev's expected gen (the known
        # down-train bug). Above-expected is silicon-out-of-spec but not a
        # regression — note it in details, leave status unaffected.
        def _gen_marker(g) -> str:
            if not isinstance(g, int):
                return "(?)"
            if g < expected_gen:
                return "(!)"
            if g > expected_gen:
                return "(^)"
            return ""

        under = [(bdf, g) for bdf, g in per_chip_gen if isinstance(g, int) and g < expected_gen]
        over = [(bdf, g) for bdf, g in per_chip_gen if isinstance(g, int) and g > expected_gen]
        listing = ", ".join(f"{format_bdf(bdf)}=Gen{g}{_gen_marker(g)}" for bdf, g in per_chip_gen)
        bits = [f"U6 chips (expected Gen{expected_gen} on {rev}): {listing}"]
        if under:
            bits.append(f"below: {', '.join(format_bdf(bdf) for bdf, _ in under)}")
        if over:
            bits.append(f"above-spec: {', '.join(format_bdf(bdf) for bdf, _ in over)}")
        phase.add(
            Check(
                name="pcie_gen",
                status=WARN if under else PASS,
                details="; ".join(bits),
                data={
                    "per_chip": dict(per_chip_gen),
                    "under_expected": under,
                    "above_expected": over,
                    "expected": expected_gen,
                    "rev": rev,
                },
                ip="pcie",
            )
        )

    # 11. GDDR speed — RevA/B chips expected at 14G; RevC chips expected at 16G.
    # Rev-indeterminate => SKIP (don't FAIL with a wrong baseline).
    speeds = Counter(d["board_info"].get("dram_speed") for d in devs)
    if rev is None:
        phase.add(
            Check(
                name="gddr_speed",
                status=SKIP,
                details="board_rev indeterminate; cannot pick expected speed (see board_rev)",
                data={"distribution": dict(speeds), "expected": None, "rev": None},
                ip="gddr",
            )
        )
    else:
        expected_speed = BOARD_REV_EXPECTATIONS[rev]["gddr_speed"]
        gddr_speed_bad = [
            (d["board_info"]["bus_id"], d["board_info"].get("dram_speed"))
            for d in devs
            if d["board_info"].get("dram_speed") != expected_speed
        ]
        phase.add(
            Check(
                name="gddr_speed",
                status=PASS if not gddr_speed_bad else FAIL,
                details=(
                    f"{n - len(gddr_speed_bad)}/{n} chips at {expected_speed} ({rev})"
                    + (
                        f"; off-speed: [{', '.join(f'{format_bdf(b)}={s}' for b, s in gddr_speed_bad[:3])}]"
                        if gddr_speed_bad
                        else ""
                    )
                ),
                data={
                    "distribution": dict(speeds),
                    "off_speed": gddr_speed_bad,
                    "expected": expected_speed,
                    "rev": rev,
                },
                ip="gddr",
            )
        )

    # 12. ETH speed — no eth speed field in tt-smi 4.1.2 or 5.1.1 schema.
    phase.add(
        Check(
            name="eth_speed",
            status=SKIP,
            details="no eth speed field in tt-smi snapshot schema",
            data={},
            ip="eth",
        )
    )

    # 12b. ETH links up — verify internal (non-QSFP) ports report as live.
    # Expected per chip = (Internal | ExaMAX) & ENABLED_ETH. Compare to
    # ETH_LIVE_STATUS masked to the same non-QSFP set.
    #
    # The port-topology tables (ETH_INTERNAL_BY_ASIC, ETH_EXAMAX_BY_ASIC) are
    # indexed by *physical* ASIC_LOCATION (the chip's slot on the UBB, fixed
    # by wiring). We derive that from the BDF rather than FW telemetry so the
    # check stays correct when FW ASIC_LOCATION is misreported — that's
    # separately surfaced by physical_vs_fw_location.
    #
    # ETH_LIVE_STATUS is only populated by FW bundle >= 19.9. On older FW
    # it stays 0x0 regardless of actual link state, so we SKIP rather than
    # WARN to avoid misattributing a firmware capability gap to a real link
    # issue. On unknown/newer FW we keep the original WARN-on-all-zero path.
    ETH_LIVE_MIN_FW = (19, 9)
    eth_live_supported = True
    detected_fw = None
    for d in devs:
        ver = d.get("firmwares", {}).get("fw_bundle_version")
        if ver:
            detected_fw = ver
            try:
                major, minor = (int(p) for p in ver.split(".")[:2])
                eth_live_supported = (major, minor) >= ETH_LIVE_MIN_FW
            except (ValueError, IndexError):
                pass
            break

    live_status_vals: list[int | None] = []
    eth_link_failures: list[dict] = []
    for d in devs:
        loc = _phys_asic_loc(d)
        live_str = d.get("smbus_telem", {}).get("ETH_LIVE_STATUS")
        try:
            live = int(live_str, 16) if live_str is not None else None
        except (ValueError, TypeError):
            live = None
        live_status_vals.append(live)
        if loc is None or live is None or loc not in ETH_INTERNAL_BY_ASIC:
            continue
        try:
            enabled = int(d["smbus_telem"]["ENABLED_ETH"], 16)
        except (KeyError, ValueError, TypeError):
            continue
        non_qsfp = (ETH_INTERNAL_BY_ASIC[loc] | ETH_EXAMAX_BY_ASIC[loc]) & enabled
        actual = live & non_qsfp
        missing = non_qsfp & ~actual
        if missing:
            down_ports = [b for b in range(12) if missing & (1 << b)]
            eth_link_failures.append(
                {
                    "bdf": d["board_info"]["bus_id"],
                    "physical_asic": loc,
                    "down_ports": down_ports,
                    "expected_mask": f"0x{non_qsfp:03x}",
                    "actual_mask": f"0x{actual:03x}",
                }
            )
    all_zero = all(v == 0 for v in live_status_vals if v is not None)
    any_present = any(v is not None for v in live_status_vals)
    if not any_present:
        phase.add(
            Check(
                name="eth_links_up", status=SKIP, details="ETH_LIVE_STATUS not present in snapshot", data={}, ip="eth"
            )
        )
    elif all_zero and not eth_live_supported:
        phase.add(
            Check(
                name="eth_links_up",
                status=SKIP,
                details=f"ETH_LIVE_STATUS requires FW bundle >= 19.9 "
                f"(detected: {detected_fw}) — firmware does not "
                f"populate this field; cannot validate links",
                data={"detected_fw": detected_fw},
                ip="eth",
            )
        )
    elif all_zero:
        phase.add(
            Check(
                name="eth_links_up",
                status=WARN,
                details=f"ETH_LIVE_STATUS=0x0 on all {len(devs)} chips — "
                f"FW {detected_fw} should populate this field; "
                f"check firmware/tt-smi",
                data={"detected_fw": detected_fw},
                ip="eth",
            )
        )
    elif eth_link_failures:
        first = eth_link_failures[0]
        phase.add(
            Check(
                name="eth_links_up",
                status=FAIL,
                details=(
                    f"{len(devs) - len(eth_link_failures)}/{len(devs)} chips OK; "
                    f"first failure: {format_bdf(first['bdf'])} "
                    f"down_ports={first['down_ports']}"
                ),
                data={"failures": eth_link_failures},
                ip="eth",
            )
        )
    else:
        phase.add(
            Check(
                name="eth_links_up",
                status=PASS,
                details=f"{len(devs)}/{len(devs)} chips: all non-QSFP links up",
                data={},
                ip="eth",
            )
        )

    # 13. GDDR thermal + error counters (5.1+ only). Store-only — surfaces values
    # but never alerts; useful for forensics / regression baselines.
    GDDR_INFO_FIELDS = (
        "MAX_GDDR_TEMP",
        "GDDR_0_1_TEMP",
        "GDDR_2_3_TEMP",
        "GDDR_4_5_TEMP",
        "GDDR_6_7_TEMP",
        "GDDR_0_1_CORR_ERRS",
        "GDDR_2_3_CORR_ERRS",
        "GDDR_4_5_CORR_ERRS",
        "GDDR_6_7_CORR_ERRS",
        "GDDR_UNCORR_ERRS",
    )
    for tag in GDDR_INFO_FIELDS:
        vals = {d["board_info"]["bus_id"]: d.get("smbus_telem", {}).get(tag) for d in devs}
        present = [v for v in vals.values() if v is not None]
        if not present:
            phase.add(
                Check(
                    name=f"gddr_info_{tag.lower()}",
                    status=SKIP,
                    details=f"{tag} not present (requires tt-smi 5.1+)",
                    data={},
                    ip="gddr",
                    console_visible=False,
                )
            )
            continue
        # Store every value verbatim; no thresholds. Hidden from console — too noisy.
        phase.add(
            Check(
                name=f"gddr_info_{tag.lower()}",
                status=PASS,
                details=f"{tag}: {len(present)} chips reporting",
                data={"per_chip": vals},
                ip="gddr",
                console_visible=False,
            )
        )

    # 14. ASIC thermal pre-check — kept in JSON only.
    temps = []
    for d in devs:
        t = d.get("telemetry", {}).get("asic_temperature")
        if t is not None:
            try:
                temps.append((d["board_info"]["bus_id"], float(t)))
            except (ValueError, TypeError):
                pass
    if temps:
        hottest_bdf, hottest_t = max(temps, key=lambda x: x[1])
        thm_limit = float(devs[0].get("limits", {}).get("thm_limit", 110))
        threshold = thm_limit - 20  # idle should be well below throttle
        status = PASS if hottest_t < threshold else WARN
        phase.add(
            Check(
                name="asic_thermal_precheck",
                status=status,
                details=f"hottest chip {hottest_bdf} @ {hottest_t:.1f}°C (limit {thm_limit}°C)",
                data={
                    "hottest_chip": hottest_bdf,
                    "hottest_temp_c": hottest_t,
                    "thm_limit_c": thm_limit,
                    "all_temps_c": dict(temps),
                },
                ip="thermal",
                console_visible=False,
            )
        )

    return rev
