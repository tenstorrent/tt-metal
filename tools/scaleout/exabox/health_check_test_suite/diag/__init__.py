# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""The Blackhole Galaxy health check's phases and the helpers they share.

diag_runner.py runs the phases in order and assembles the report; tiers.py
decides what each tier asks of each phase. The phase modules:

    snapshot.py, snapshot_checks.py   1  tt-smi snapshot + FRU inventory, and the checks over it
    resets.py                         2  pre-test reset loop; 4  post-test reset
    device_tests.py                   3  deployment gtests and pytest stress tests
    triage.py                         5  kmd_triage first-step tools
    qsfp.py, qsfp_ingest.py           6  QSFP tests (ETH dump collection and its conversion to checks)
    sys_triage.py                     7  syseng sys-triage bundle

and what they share: model.py (Check, Phase), common.py (subprocess, console and
path helpers), galaxy.py (BH Galaxy topology), external.py (ingesting checks
written by external tools).
"""
