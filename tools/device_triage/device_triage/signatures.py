# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Failure taxonomy and the signatures that map evidence onto it.

Two kinds of evidence feed the same categories:
- machine HC checks (structured): `classify_check()` turns a diag_report check into findings;
- log lines (unstructured): `match_line()` against `LINE_SIGNATURES`, first match wins,
  so specific signatures are listed before generic ones.
"""

import json
import re
from dataclasses import dataclass, field

from . import hosts

CATEGORIES = {
    "gddr": "GDDR / DRAM",
    "pcie": "PCIe / enumeration",
    "tensix": "Tensix / compute / NOC",
    "bmc": "BMC / IPMI",
    "reset": "Resets",
    "eth": "Ethernet links / cabling",
    "arc_fw": "ARC / firmware",
    "host": "Host OS / network",
    "power_thermal": "Power / thermal",
    "infra": "Tooling / config",
    "unknown": "Unclassified",
}
# The categories the DIIM ask is about; the rest are tracked so evidence isn't misfiled.
PRIMARY_CATEGORIES = ("gddr", "pcie", "tensix", "bmc", "reset")
SEVERITY_WEIGHT = {"fail": 3.0, "warn": 1.0, "info": 0.0}


@dataclass(frozen=True)
class Signature:
    name: str
    category: str
    severity: str
    pattern: re.Pattern
    help: str = ""


def _sig(name: str, category: str, severity: str, regex: str, help: str = "", flags: int = 0) -> Signature:
    return Signature(name, category, severity, re.compile(regex, flags), help)


I = re.IGNORECASE

LINE_SIGNATURES: tuple[Signature, ...] = (
    # Resets (listed first: reset errors often mention PCIe or devices too)
    _sig("post_reset_failed", "reset", "fail", r"POST_RESET failed for device (?P<dev>\d+)"),
    _sig("reset_ioctl_failed", "reset", "fail", r"Sending reset ioctl to device (?P<dev>\d+) failed"),
    _sig("glx_reset_failed", "reset", "fail", r"Error in resetting galaxy 6u trays"),
    _sig("reinit_failed", "reset", "fail", r"Error when re-initializing chips"),
    _sig(
        "reset_failed",
        "reset",
        "fail",
        r"Reset failed \| Exit code|Cleanup reset failed|failed to reset|reset (?:did not|didn't) (?:complete|succeed)",
        flags=I,
    ),
    _sig(
        "chip_not_back_after_reset",
        "reset",
        "fail",
        r"(?:did not|didn't|failed to) come back after (?:a )?reset|not (?:enumerated|present) after reset",
        flags=I,
    ),
    _sig("tray_i2c_error", "reset", "fail", r"\bI2C\b.{0,60}(?:fail|error|timeout|NACK)", flags=I),
    # PCIe
    _sig("pcie_read_ffffffff", "pcie", "fail", r"Read 0xffffffff over PCIe(?: ID (?P<dev>\d+))?"),
    # All-ones version: the firmware registers read back as 0xff, i.e. the chip isn't answering over PCIe.
    _sig(
        "fw_bundle_unreadable",
        "pcie",
        "warn",
        r"(?:fw_bundle_version=|firmware bundle version:?\s*)255\.255\.255\.255",
        flags=I,
    ),
    # UMD host->device barrier: `Memory barrier timed out ... on device 30 core ...: expected 0xaa, read 0xffffffff`.
    _sig(
        "membar_read_ffffffff",
        "pcie",
        "fail",
        r"barrier timed out.{0,80}?\bon device (?P<dev>\d+)\b.{0,120}read 0xffffffff",
        flags=I,
    ),
    _sig("pcie_hang", "pcie", "fail", r"PcieHangError|PCIe (?:link )?hang", flags=I),
    _sig("query_mappings_failed", "pcie", "fail", r"Query mappings failed on device (?P<dev>\d+)"),
    _sig("device_open_failed", "pcie", "fail", r"Failed to open device /dev/tenstorrent/(?P<dev>\d+)"),
    _sig("kmd_enodev", "pcie", "fail", r"KMD returned: -19\b"),
    _sig(
        "no_devices_detected",
        "pcie",
        "fail",
        r"Error in detecting devices|No Tenstorrent devices (?:found|detected)",
        flags=I,
    ),
    _sig(
        "devices_short",
        "pcie",
        "fail",
        r"(?<![\d/])(?:[0-9]|[12][0-9]|3[01])\s*/\s*32\s+(?:devices|chips)\b(?! OK)",
        flags=I,
    ),
    # UMD cluster descriptor: `Board 0x47330000000 has 24 chips, but expected 32 chips for board type ubb_blackhole`.
    _sig("board_chips_short", "pcie", "fail", r"Board 0x[0-9a-f]+ has \d+ chips, but expected \d+ chips"),
    _sig("set_power_state_enodev", "pcie", "fail", r"SET_POWER_STATE failed on device (?P<dev>\d+): No such device"),
    _sig(
        "pcie_aer_uncorrectable",
        "pcie",
        "fail",
        r"AER: (?:Uncorrect(?:ed|able)|Fatal)|PCIe Bus Error: severity=(?:Uncorrect|Fatal)",
        flags=I,
    ),
    _sig("pcie_aer_corrected", "pcie", "warn", r"AER: Correct(?:ed|able)|PCIe Bus Error: severity=Correct", flags=I),
    _sig(
        "pcie_link_down",
        "pcie",
        "fail",
        r"Surprise Down|DPC: containment event|pciehp.{0,40}(?:Link Down|Card not present)",
        flags=I,
    ),
    _sig("iommu_fault", "pcie", "warn", r"IO_PAGE_FAULT|AMD-Vi.{0,40}(?:fault|Event logged)|DMAR:.{0,40}fault"),
    _sig("sigbus", "pcie", "warn", r"Signal: Bus error|SIGBUS"),
    # GDDR. ARC reports why it failed to boot: `smc_init_status=0x10 (GDDR training failure)`.
    _sig(
        "fw_startup_gddr_training",
        "gddr",
        "fail",
        r"startup error(?: on device (?P<dev>\d+))? at core.*\(GDDR training failure\)",
        flags=I,
    ),
    _sig("dram_training_failed", "gddr", "fail", r"DRAM training failed|gddr issue", flags=I),
    _sig("gddr_training_failed", "gddr", "fail", r"GDDR\s*\d*\s+(?:training|train)\w*\s+(?:fail|error)", flags=I),
    _sig("dram_bist_failed", "gddr", "fail", r"\bBIST\b.{0,40}\bfail", flags=I),
    _sig(
        "dram_test_failed",
        "gddr",
        "fail",
        r"\[\s+FAILED\s+\]\s+\S*Dram\S*|(?:DRAM|GDDR) tests?\s+loop \d+ failed",
        flags=I,
    ),
    # Tensix / compute
    _sig("aiclk_settle_timeout", "tensix", "fail", r"Waiting for AICLK value to settle failed on timeout"),
    _sig("physical_cores_timeout", "tensix", "fail", r"Timeout.*waiting for physical cores"),
    _sig("no_asic_id", "tensix", "warn", r"No ASIC ID found at hostname"),
    _sig(
        "noc_sanity_failed",
        "tensix",
        "fail",
        r"noc_sanity.{0,40}(?:FAIL|did not answer)|NOC (?:sanity|read) (?:failed|timed out)",
        flags=I,
    ),
    _sig(
        "nondeterminism",
        "tensix",
        "fail",
        r"non-?determinis|PCC (?:mismatch|check failed)|determinism (?:re-?)?check failed",
        flags=I,
    ),
    _sig("core_hang", "tensix", "fail", r"(?:Tensix|worker) cores? .{0,40}(?:hang|hung|stuck|not responding)", flags=I),
    _sig("watcher_error", "tensix", "warn", r"Watcher detected|watcher.{0,40}(?:assert|NOC error|sanitiz)", flags=I),
    # Core firmware not in the state the host expects: a failed reset or a dead chip.
    _sig("run_mailbox_unexpected", "tensix", "warn", r"unexpected run_mailbox value"),
    # BMC
    _sig(
        "bmc_unreachable",
        "bmc",
        "fail",
        r"BMC.{0,60}(?:unreachable|not reachable|unavailable|not responding|is down|timed out|connection refused)",
        flags=I,
    ),
    _sig(
        "ipmi_failed",
        "bmc",
        "fail",
        r"ipmitool.{0,80}(?:error|failed|unable)|Unable to establish (?:IPMI|LAN)|IPMI.{0,30}(?:error|failed|timed out|timeout)|Get Device ID command failed",
        flags=I,
    ),
    _sig("redfish_failed", "bmc", "warn", r"redfish.{0,40}(?:error|fail|timeout)", flags=I),
    # Ethernet / cabling
    _sig("unhealthy_links", "eth", "fail", r"Found Unhealthy Links|FAULTY LINKS REPORT"),
    _sig("missing_connections", "eth", "fail", r"missing (?:port/cable|channel) connections"),
    # Deployment test: `missing links: chip[0 (0000:01:00.0), ubb: 1, chip: 1]: expected 10 links, got 8`.
    _sig("chip_links_missing", "eth", "fail", r"missing links: chip\[\d+ \("),
    # Older fabric health check: `Chip 11 Tray: 2 N4 has 1 connections to Chip 3 Tray: 1 N4, expected 2`.
    _sig(
        "chip_connections_short",
        "eth",
        "fail",
        r"Chip \d+ Tray: \d+ \w+ has \d+ connections to Chip \d+ Tray: \d+ \w+, expected \d+",
    ),
    _sig("eth_test_loop_failed", "eth", "fail", r"Ethernet tests?\s+loop \d+ failed", flags=I),
    _sig("extra_connections", "eth", "warn", r"extra (?:port/cable|channel) connections"),
    _sig("missing_global_connection", "eth", "fail", r"Global Connection.*is not found in the host connectivity graph"),
    _sig("link_train_failed", "eth", "fail", r"LINK_TRAIN_(?:TIMEOUT|FAIL)\w*"),
    _sig("mgd_topology_mismatch", "eth", "fail", r"Graph specified in MGD could not fit"),
    _sig("eth_data_mismatch", "eth", "fail", r"Data Mismatch"),
    _sig("eth_crc_error", "eth", "warn", r"CRC Error"),
    _sig("eth_uncorrected_cw", "eth", "warn", r"Uncorrected CW"),
    _sig(
        "eth_core_timeout",
        "eth",
        "fail",
        r"Timed out (?:while )?waiting for (?:active )?eth(?:ernet)? core|ETH core.{0,30}timeout",
        flags=I,
    ),
    _sig("port_down_failed", "eth", "warn", r"cross-host port down FAILED"),
    _sig("links_retrained", "eth", "info", r"Rediscovering ethernet links after successful link retraining"),
    # ARC / firmware
    _sig(
        "arc_startup_error",
        "arc_fw",
        "fail",
        r"ArcStartupError|(?:Firmware|ARC) startup error(?: on device (?P<dev>\d+))?|"
        r"ARC (?:core )?(?:\([^)]*\) )?(?:failed to start|not ready|did not start|startup timed out)",
    ),
    _sig("arc_timeout", "arc_fw", "fail", r"ARC.*[Tt]imeout|ARC message timed out"),
    _sig("heartbeat_stalled", "arc_fw", "fail", r"heartbeat.{0,40}(?:stall|stuck|not advanc|frozen)", flags=I),
    _sig("telemetry_bad", "arc_fw", "warn", r"telemetry.{0,30}(?:unpublished|malformed|not published)", flags=I),
    # UMD throws on this, so every workload on the host fails until the odd tray is reflashed.
    _sig("fw_mismatch", "arc_fw", "fail", r"(?:firmware|FW) (?:bundle )?(?:version )?mismatch", flags=I),
    # Power / thermal
    _sig(
        "power_fault",
        "power_thermal",
        "warn",
        r"\bPSU\b.{0,40}(?:fail|fault|lost|absent)|power supply.{0,30}(?:fail|fault|lost)|power (?:loss|fault|failure)|brownout|\bVRM\b.{0,30}fault",
        flags=I,
    ),
    _sig(
        "thermal_event",
        "power_thermal",
        "warn",
        r"thermal (?:trip|shutdown|throttl)|over-?temp|temperature (?:too high|exceeded|critical)",
        flags=I,
    ),
    # Host
    _sig(
        "machine_check", "host", "warn", r"mce: \[Hardware Error\]|Machine check events logged|Machine Check Exception"
    ),
    _sig("disk_full", "host", "fail", r"No space left on device"),
    _sig("oom", "host", "warn", r"Out of memory: Kill|oom-kill:|oom_reaper"),
    _sig(
        "ssh_unreachable",
        "host",
        "warn",
        r"ssh: connect to host .{0,80}(?:refused|timed out|No route to host)",
        flags=I,
    ),
    _sig(
        "network_unreachable",
        "host",
        "warn",
        r"Unable to connect to the peer.*Network is unreachable|btl_tcp_endpoint.*Unable to connect",
    ),
    # Host memory mapped for the chips at the wrong NOC address, or left mapped: IOMMU/driver state, all chips at once.
    _sig("sysmem_noc_mismatch", "host", "warn", r"Expected NOC address: 0x[0-9a-f]+, but got 0x[0-9a-f]+"),
    _sig("sysmem_unmap_failed", "host", "warn", r"Failed to unmap sysmem buffer"),
    _sig("set_power_state_einval", "host", "warn", r"SET_POWER_STATE failed on device \d+: Invalid argument"),
    # Tooling / config
    _sig("fsd_error", "infra", "fail", r"Hostname not found in FSD"),
    _sig("ssh_auth_failed", "infra", "info", r"Permission denied \(publickey\)"),
    _sig("workload_timeout", "infra", "warn", r"Workload execution timed out|MPIEXEC_TIMEOUT|DUE TO TIME LIMIT"),
    _sig(
        "docker_failed",
        "infra",
        "warn",
        r"docker: Error response from daemon|Cannot connect to the Docker daemon|Docker startup error",
    ),
    _sig("segfault", "infra", "warn", r"Segmentation fault"),
    _sig("mpi_abort", "infra", "info", r"MPI_ABORT was invoked|exited on signal \d+"),
    _sig("tt_fatal", "unknown", "info", r"TT_FATAL|TT_THROW"),
)

# Cheap case-insensitive keyword gate: a line that has none of these can't match.
PREFILTER = re.compile(
    r"fail|error|timeout|timed out|unhealthy|faulty|missing|extra |0xffffffff|hang|hardware error|aer:|bmc|ipmi|"
    r"reset|dram|gddr|aiclk|arc|crc|uncorrect|mismatch|fatal|throw|no space|out of memory|oom|unreachable|denied|"
    r"link_train|retrain|bus error|sigbus|determinis|pcc |i2c|psu|power|thermal|temp|segmentation|query mappings|"
    r"/ ?32 |global connection|physical cores|asic id|watcher|redfish|surprise down|dpc:|io_page_fault|dmar|"
    r"come back|not present|detecting devices|mpi|time limit|docker|machine check|heartbeat|telemetry|firmware|fw |"
    r"255\.255\.255\.255|chips, but expected|connections to chip",
    re.IGNORECASE,
)


def match_line(line: str) -> tuple[Signature, re.Match] | None:
    if not PREFILTER.search(line):
        return None
    for sig in LINE_SIGNATURES:
        m = sig.pattern.search(line)
        if m:
            return sig, m
    return None


SIGNATURES_BY_NAME = {s.name: s for s in LINE_SIGNATURES}
# Lines about a remote target ("ssh: connect to host X", "BMC of X unreachable"):
# blame the host the line names, not the host that logged it.
TARGETED_SIGNALS = {"ssh_unreachable", "bmc_unreachable", "ipmi_failed", "network_unreachable", "redfish_failed"}


@dataclass
class Finding:
    """One piece of evidence before it is stored: what, how bad, and on which chips."""

    category: str
    signal: str
    severity: str
    detail: str
    chips: set[str] = field(default_factory=set)
    origin: str = ""


# analyze_validation_results.py exit codes, as recorded in cluster HC `analyzer_code`.
ANALYZER_CODES: dict[int, tuple[str, str, str]] = {
    1: ("eth", "unhealthy_links_repeated", "fail"),
    2: ("eth", "unhealthy_links_scattered", "fail"),
    3: ("gddr", "dram_training_failed", "fail"),
    4: ("eth", "missing_connections", "fail"),
    5: ("eth", "extra_connections", "warn"),
    6: ("eth", "missing_global_connection", "fail"),
    7: ("infra", "fsd_error", "fail"),
    8: ("infra", "mgd_error", "fail"),
    9: ("infra", "workload_timeout", "warn"),
    10: ("arc_fw", "arc_timeout", "fail"),
    11: ("tensix", "aiclk_timeout", "fail"),
    12: ("host", "network_error", "warn"),
    13: ("pcie", "missing_devices", "fail"),
    50: ("unknown", "inconclusive", "warn"),
    66: ("infra", "analyzer_input_error", "info"),
    124: ("infra", "wrapper_timeout", "warn"),
    126: ("infra", "wrapper_exec_error", "info"),
}

# Machine HC check -> (category, signal). Statuses: FAIL -> fail, WARN -> warn.
CHECK_RULES: dict[str, tuple[str, str]] = {
    "dram_status": ("gddr", "dram_status_false"),
    "enabled_gddr_full": ("gddr", "gddr_disabled"),
    "gddr_training_per_channel": ("gddr", "gddr_channel_untrained"),
    "gddr_bist_per_channel": ("gddr", "gddr_bist_failed"),
    "gddr_speed": ("gddr", "gddr_off_speed"),
    "gddr_fw_consistent": ("arc_fw", "gddr_fw_inconsistent"),
    "pcie_enum_count": ("pcie", "chips_missing_from_bus"),
    "pcie_lane_width": ("pcie", "pcie_lane_width_low"),
    "pcie_gen": ("pcie", "pcie_gen_low"),
    "asic_location_per_ubb": ("pcie", "asic_location_missing"),
    "qsfp_inventory": ("pcie", "qsfp_inventory_short"),
    "triage_hostside_chips_present": ("pcie", "chips_missing_from_bus"),
    "triage_hostside_pcie_link": ("pcie", "pcie_link_degraded"),
    "triage_hostside_pcie_bar": ("pcie", "pcie_bar_unassigned"),
    "triage_hostside_pcie_aer": ("pcie", "pcie_aer_endpoint"),
    "triage_hostside_bridge_aer": ("pcie", "pcie_aer_bridge"),
    "triage_hostside_driver_bound": ("pcie", "driver_not_bound"),
    "triage_hostside_chip_reachable": ("pcie", "chip_unreachable"),
    "triage_deviceside_devices_present": ("pcie", "devices_missing"),
    "triage_hostside_driver_loaded": ("host", "driver_not_loaded"),
    "triage_hostside_kernel_log": ("host", "kernel_log_faults"),
    "triage_hostside_arc_heartbeat": ("arc_fw", "arc_heartbeat_stalled"),
    "triage_deviceside_liveness": ("arc_fw", "chip_not_answering"),
    "triage_deviceside_info": ("arc_fw", "device_info_degraded"),
    "triage_deviceside_arc_scratch": ("arc_fw", "arc_scratch_bad"),
    "triage_deviceside_telemetry": ("arc_fw", "telemetry_bad"),
    "triage_deviceside_noc_sanity": ("tensix", "noc_sanity_failed"),
    "harvesting_state": ("tensix", "harvesting_state_bad"),
    "physical_vs_fw_location": ("arc_fw", "location_mismatch"),
    "fw_bundle_version_consistent": ("arc_fw", "fw_bundle_inconsistent"),
    "cm_fw_consistent": ("arc_fw", "cm_fw_inconsistent"),
    "eth_fw_consistent": ("arc_fw", "eth_fw_inconsistent"),
    "board_rev": ("arc_fw", "board_rev_mixed"),
    "qsfp_board_rev": ("arc_fw", "board_rev_inconsistent"),
    "qsfp_board_rev_agrees": ("arc_fw", "board_rev_disagrees"),
    "eth_links_up": ("eth", "eth_ports_down"),
    "qsfp_findings": ("eth", "qsfp_collector_findings"),
    "qsfp_collection_failures": ("eth", "qsfp_read_failed"),
    "qsfp_collection_partial": ("eth", "qsfp_partial"),
    "qsfp_link_training": ("eth", "link_training_failed"),
    "qsfp_link_asymmetry": ("eth", "link_asymmetry"),
    "qsfp_missing_channel": ("eth", "missing_channel"),
    "qsfp_miscabled": ("eth", "miscabled"),
    "qsfp_partner_disagreement": ("eth", "partner_disagreement"),
    "qsfp_cage_gaps": ("eth", "cage_gaps"),
    "asic_thermal_precheck": ("power_thermal", "thermal_precheck"),
    "post_test_reset_ok": ("reset", "post_test_reset_degraded"),
    "cpld_auto_recover": ("reset", "cpld_auto_recover_failed"),
    "post_reset_state_stable": ("reset", "post_reset_state_unstable"),
    "cpld_fw_old": ("arc_fw", "cpld_fw_old"),
    "host_fru_info": ("bmc", "bmc_fru_unavailable"),
    "snapshot_capture": ("unknown", "snapshot_failed"),
}
# Fleet-wide WARNs that say nothing about one unit: kept, never scored.
NOISE_CHECKS = {"cpld_fw_old", "post_reset_state_stable"}
# Host state accumulated over a boot: WARN on units everything else calls healthy.
ADVISORY_CHECKS = {"triage_hostside_pcie_aer", "triage_hostside_kernel_log", "pcie_gen"}
TEST_DEFAULTS = (
    ("gddr", "gddr", "gddr_test_failed"),
    ("dram", "gddr", "gddr_test_failed"),
    ("eth", "eth", "eth_test_failed"),
    ("didt", "tensix", "didt_test_failed"),
)
RESET_CHECK_RE = re.compile(r"^reset_\d+_(r|glx_reset)$")
CHIP_DATA_KEYS = (
    "missing_bdfs",
    "failed_chips",
    "failures",
    "off_speed",
    "findings",
    "failed",
    "warned",
    "missing",
    "under_spec",
    "degraded",
    "errors",
    "absent",
    "down",
)


def _chips_from_check(details: str, data: dict, bdf_map: dict[str, str]) -> set[str]:
    # pcie_gen lists every U6 chip with its speed, then the off-speed ones after "below:".
    if "below:" in details:
        details = details.split("below:", 1)[1]
    chips = {chip for _, chip in hosts.bdf_chip_pairs(details)}
    chips |= {hosts.chip_label(m.group(1), m.group(2)) for m in hosts.UC_RE.finditer(details)}
    for key in CHIP_DATA_KEYS:
        if key in data and data[key]:
            blob = json.dumps(data[key])
            chips |= {bdf_map[b.lower()] for b in hosts.BDF_RE.findall(blob) if b.lower() in bdf_map}
            chips |= {hosts.chip_label(m.group(1), m.group(2)) for m in hosts.UC_RE.finditer(blob)}
    for tray in data.get("missing_trays") or []:
        chips.add(f"u{int(tray)}")
    return chips


def classify_check(phase: str, check: dict, bdf_map: dict[str, str], report_checks: dict[str, str]) -> list[Finding]:
    """Findings for one diag_report check. `report_checks` maps check name -> worst status in the report."""
    name = check.get("name", "")
    status = check.get("status", "")
    details = str(check.get("details") or "")
    data = check.get("data") if isinstance(check.get("data"), dict) else {}
    origin = f"check:{phase}/{name}"
    chips = _chips_from_check(details, data, bdf_map)
    findings: list[Finding] = []

    if RESET_CHECK_RE.match(name):
        if status not in ("FAIL", "WARN"):
            return []
        flag = RESET_CHECK_RE.match(name).group(1)
        severity = "fail" if status == "FAIL" else "warn"
        # tt-smi -r needs CPLD >= 1.16; with the old CPLD the suite falls back to
        # -glx_reset, so a -r failure only matters if that fallback failed too.
        if flag == "r" and report_checks.get("cpld_auto_recover") != "FAIL":
            severity = "warn"
        findings.append(Finding("reset", f"reset_{flag}_failed", severity, details, chips, origin))
        post_pcie = data.get("post_pcie_count")
        if isinstance(post_pcie, int) and post_pcie < 32:
            findings.append(
                Finding(
                    "pcie",
                    "chip_dropped_after_reset",
                    "fail",
                    f"{post_pcie}/32 chips on the bus after {name}",
                    chips,
                    origin,
                )
            )
        return findings

    if phase == "tests" or name.startswith(("gddr_fast", "eth_link_up", "eth_bandwidth", "didt")):
        if status not in ("FAIL", "WARN"):
            return []
        category, signal = "unknown", "test_failed"
        for needle, cat, sig in TEST_DEFAULTS:
            if needle in name:
                category, signal = cat, sig
                break
        # Category is provisional: the indexer reclassifies from the test log.
        return [Finding(category, signal, "fail" if status == "FAIL" else "warn", details, chips, origin)]

    rule = CHECK_RULES.get(name)
    if rule is None:
        if status == "FAIL":
            return [Finding("unknown", f"check_{name}", "fail", details, chips, origin)]
        return []
    category, signal = rule

    if name == "host_fru_info":
        if status == "SKIP" and re.search(r"bmc|ipmi|timed out|unable|refused|unreachable", details, re.I):
            return [Finding("bmc", signal, "warn", details, chips, origin)]
        return []
    if status not in ("FAIL", "WARN"):
        return []
    severity = "fail" if status == "FAIL" else "warn"
    if name in NOISE_CHECKS:
        severity = "info"
    elif name in ADVISORY_CHECKS and severity == "warn":
        severity = "info" if name == "pcie_gen" and "(!)" not in details else "warn"
    if name == "snapshot_capture":
        hit = match_line(details)
        if hit:
            category, signal = hit[0].category, hit[0].name
    return [Finding(category, signal, severity, details, chips, origin)]
