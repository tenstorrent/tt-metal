# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import sys

# The pytest plugin lives in this package, so --codegen-host-only must reach it
# without importing the device harness (ttexalens) first.
if "--codegen-host-only" not in sys.argv:
    from ttexalens import Verbosity

    Verbosity.set(Verbosity.ERROR)
