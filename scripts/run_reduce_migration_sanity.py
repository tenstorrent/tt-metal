#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Run one selected numerical case per covered reduce-migration kernel using run_safe_pytest.

Use --list to see exact cases, kernel coverage and hardware requirements.
Use --lane to choose hardware lanes, or --kernel to run a kernel's primary case.
No arguments select the complete suite, including cases requiring other hardware.
"""

import sys

from run_reduce_migration_tests import ROOT, main

MANIFEST = ROOT / ("ttnn/cpp/ttnn/kernel_lib/reduce_migration_inventory_2026-09-08_f808380a87b/sanity_test_suite.json")


if __name__ == "__main__":
    sys.exit(main(default_manifest=MANIFEST))
