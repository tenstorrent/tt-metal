# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""What each health-check tier runs. The one file to edit to add a tier, or to
add, move or drop a step in one.

The phases always run in the same order, fixed in diag_runner.run_diag():

    snapshot -> resets -> tests -> post_test_resets -> triage -> qsfp_tests -> sys_triage

The order is load-bearing (the triage probes, the QSFP collector and sys-triage
all open every chip, so they must not overlap the tests, and must come after a
discrete reset; see the comments on each field below), so a tier does not
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
run_diag.sh (its case statement and --help) and run_diag_docker.sh (both case
statements and --help), plus the tier table in HEALTH_CHECK.md.
tools/scaleout/exabox/tests/test_diag_tiers.py fails until the two scripts agree
with TIERS.
"""
from __future__ import annotations

from dataclasses import dataclass

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

    # Collect a tt-bh-glx-cluster-debug ETH dump and check it. A minutes-long
    # ETH sweep; it sits in the same slot as triage, after the post-test reset.
    qsfp_tests: bool = False

    # Collect the syseng sys-triage bundle. It is the record of a unit's state
    # taken before the BMC power cycles it, and its vuart and scratch reads are
    # worth most on a unit that has not been reset since it misbehaved.
    sys_triage: bool = False


TIERS: dict[str, Tier] = {
    # Smoke check. No triage, QSFP or sys-triage: each costs minutes.
    "light": Tier(
        resets=("-r",),
        tests=("eth_link_up",),
    ),
    "medium": Tier(
        resets=("-r", "-glx_reset"),
        tests=("gddr_fast", "eth_link_up", "eth_bandwidth"),
        post_test_resets=("-glx_reset",),
        triage=("host_side", "device_side"),
        qsfp_tests=True,
    ),
    "deploy": Tier(
        resets=("-r", "-glx_reset", "-glx_reset"),
        tests=("gddr_full", "eth_link_up", "eth_bandwidth", "didt_matmul_galaxy"),
        post_test_resets=("-glx_reset",),
        triage=("host_side", "device_side"),
        qsfp_tests=True,
    ),
    # Data collection before a BMC reboot. No resets and no tests: a reset
    # would wipe the state the triage, QSFP and sys-triage phases are there to
    # record, and a stress test on a unit already headed for a power cycle only
    # disturbs it. With no tests there is nothing for triage to overlap, so no
    # post-test reset either. The only tier that collects sys-triage.
    "pre_reboot": Tier(
        triage=("host_side", "device_side"),
        qsfp_tests=True,
        sys_triage=True,
    ),
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
    if problems:
        raise ValueError("tiers.py: " + "; ".join(problems))


_validate()
