# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Source A's REFERENCE chains -- the other side of every PCC comparison.

Deliberately NOT under `tt/`. `tt/` is the device port: everything in it is either the forward
path or the build that feeds it, and tooling reads the whole package as such. The golden is torch,
runs on the host, and is the thing the port is measured AGAINST -- so it lives beside `tt/`, not
inside it. `tests/e2e/test_gates.py` asserts nothing in `tt/` imports this at module scope.
"""
