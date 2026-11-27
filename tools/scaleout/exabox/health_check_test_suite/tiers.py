# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""What each health-check tier runs. The one file to edit to add a tier, or to
add, move or drop a step in one.

The phases always run in the same order, fixed in diag_runner.run_diag():

    snapshot -> resets -> tests -> post_test_resets -> triage -> eth_status -> qsfp_tests -> sys_triage

The order is load-bearing (the triage probes, the ETH status read, the QSFP
collector and sys-triage all open every chip, so they must not overlap the tests,
and must come after a discrete reset; see the comments on each field below), so a tier does not
reorder phases. It only picks what each one does. A phase a tier leaves empty is
still in the report, as a SKIP that names the tier — except post_test_resets,
whose phase is left out of the report when there is nothing to run.

Where the steps a tier can name are defined:

    resets, post_test_resets  tt-smi flags: "-r" or "-glx_reset"
    tests                     TESTS (gtest) and PYTESTS (pytest) in diag/device_tests.py
    triage                    TRIAGE_TOOLS in diag/triage.py

A name that isn't defined there fails at import, so a typo stops
`diag_runner.py --help` (and the image's build-time smoke test) rather than a run.

Adding or renaming a tier also means updating, by hand, the tier lists in
run_diag.sh (its case statement and --help), run_diag_docker.sh (both case
statements, the sys-triage tier check and --help) and run_health_check.py's
--tier choices, plus the tier table in HEALTH_CHECK.md.
tools/scaleout/exabox/tests/test_diag_tiers.py fails until they agree with
TIERS.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

from diag.device_tests import PYTESTS, TESTS
from diag.resets import RESET_FLAGS
from diag.triage import TRIAGE_TOOLS


@dataclass(frozen=True)
class Tier:
    # Pre-test reset loop, in order. Per SYS-4365 the first reset is -r, then
    # stick to -glx_reset. Each is followed by a fresh snapshot, revalidated.
    resets: tuple[str, ...] = ()

    # Device tests, in order: names from TESTS / PYTESTS.
    tests: tuple[str, ...] = ()

    # Reset run between the tests and the triage phase. Two reasons it sits
    # there: the triage probes open every chip read-write, so they must not
    # overlap the tests, and the tools have no SIGBUS handler — tt-kmd zaps
    # every mapping on reset, so a reset concurrent with a probe kills it
    # outright rather than being reported. A discrete reset that completes
    # first rules both out. Bare: no revalidation snapshot (see
    # diag.resets.run_post_test_reset). Skipped by --skip-reset and --skip-triage.
    post_test_resets: tuple[str, ...] = ()

    # First-step triage tools: names from TRIAGE_TOOLS.
    triage: tuple[str, ...] = ()

    # Read every on-host ETH link twice with tt-triage's check_eth_status and
    # judge the two reads (diag/eth_status.py). It opens every chip, so it sits
    # after the post-test reset like the triage tools; it never resets, and
    # stops before opening a chip if one has dropped off PCIe.
    eth_status: bool = False

    # Print the ETH link summary at the end of the run (diag/eth_summary.py), for
    # the person at the console rather than the dashboard: a verdict, what is
    # broken and where, and the cabling found. Kept as logs/eth_status_summary.txt
    # too. Needs eth_status.
    eth_summary: bool = False

    # Collect a tt-bh-glx-cluster-debug ETH dump and check it. A minutes-long
    # ETH sweep; it sits in the same slot as triage, after the post-test reset.
    qsfp_tests: bool = False

    # Collect the syseng sys-triage bundle. It is the record of a unit's state
    # taken before the BMC power cycles it, and its vuart and scratch reads are
    # worth most on a unit that has not been reset since it misbehaved.
    sys_triage: bool = False


_WEEKLY = Tier(
    resets=("-r", "-glx_reset", "-glx_reset"),
    tests=("gddr_full", "eth_link_up", "eth_bandwidth", "didt_matmul_galaxy"),
    post_test_resets=("-glx_reset",),
    triage=("host_side", "device_side"),
    qsfp_tests=True,
)

TIERS: dict[str, Tier] = {
    # Quick ETH link health, based on Veljko Cankovic's script: the unit read as
    # found, the snapshot and two ETH status reads 30 s apart. No resets and no
    # tests, so nothing changes the state being read, and a fault is still
    # there for whoever picks it up. The sys-triage bundle comes last, with the
    # unit still as found. The one tier anyone may run by hand, so it ends with
    # the ETH link summary.
    "flash": Tier(
        eth_status=True,
        eth_summary=True,
        sys_triage=True,
    ),
    # The scheduled run.
    "daily": Tier(
        resets=("-r", "-glx_reset"),
        tests=("gddr_fast", "eth_link_up", "eth_bandwidth"),
        post_test_resets=("-glx_reset",),
        triage=("host_side", "device_side"),
        qsfp_tests=True,
    ),
    # daily, with every test and the full GDDR pattern set.
    "weekly": _WEEKLY,
    # weekly plus the sys-triage bundle: everything, for a unit with an issue,
    # so the run's logs can go straight on its DIIM ticket.
    "full": replace(_WEEKLY, sys_triage=True),
    # Data collection before a BMC reboot. No resets and no tests: a reset
    # would wipe the state the triage, QSFP and sys-triage phases are there to
    # record, and a stress test on a unit already headed for a power cycle only
    # disturbs it. With no tests there is nothing for triage to overlap, so no
    # post-test reset either.
    "pre_bmc_reboot": Tier(
        triage=("host_side", "device_side"),
        qsfp_tests=True,
        sys_triage=True,
    ),
    # Data collection after a BMC reboot. The same steps as pre_bmc_reboot for
    # now, as its own entry so the two can change independently.
    "post_bmc_reboot": Tier(
        triage=("host_side", "device_side"),
        qsfp_tests=True,
        sys_triage=True,
    ),
}

# Old names, accepted until the exabox-infra and tt-orchestration callers move
# over. The report records the new name.
DEPRECATED_TIER_ALIASES: dict[str, str] = {
    "light": "daily",
    "medium": "daily",
    "deploy": "weekly",
    "pre_reboot": "pre_bmc_reboot",
}


def _validate() -> None:
    problems = []
    for name, tier in TIERS.items():
        problems += [f"{name}: unknown reset flag {f!r}" for f in tier.resets if f not in RESET_FLAGS]
        problems += [
            f"{name}: unknown post-test reset flag {f!r}" for f in tier.post_test_resets if f not in RESET_FLAGS
        ]
        problems += [f"{name}: unknown test {t!r}" for t in tier.tests if t not in TESTS and t not in PYTESTS]
        problems += [f"{name}: unknown triage tool {t!r}" for t in tier.triage if t not in TRIAGE_TOOLS]
        if tier.eth_summary and not tier.eth_status:
            problems.append(f"{name}: eth_summary needs eth_status")
    problems += [
        f"{old}: alias for unknown tier {new!r}" for old, new in DEPRECATED_TIER_ALIASES.items() if new not in TIERS
    ]
    if problems:
        raise ValueError("tiers.py: " + "; ".join(problems))


_validate()
