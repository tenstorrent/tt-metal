<!--
SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

SPDX-License-Identifier: Apache-2.0
-->

# hc: the Galaxy health-check core

This folder holds the part of tray-test that other repos can carry unchanged: the stage engine behind
`run.sh` and the programs and helpers it runs. `CORE` lists those files. They move here in steps; until a
file has moved, `CORE` names it at its current path.

Rules for files in this folder:

- Every source file starts with the SPDX header above (`#` comments in shell and python).
- No site names, host names, internal ticket ids, chat links, personal names or private IP addresses.
  Comments explain the behaviour in plain words instead of pointing at a tracker.
- Local features (hold files, serving gate, pinned wheels, agent loops) are hooks that are off by
  default; tray-test turns them on from its wrapper. `../HC_STAYS` lists the files that stay local.

`tests/test_hc_clean.sh` enforces the header and pattern rules here and checks that `CORE` plus
`HC_STAYS` cover every file `run.sh` uses.
