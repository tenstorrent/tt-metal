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

## Hooks

`hc/lib/hc.sh` loads the hooks. It sources `lib/hc_hooks.sh` once if that file exists; tray-test's copy sets
every hook below, and a copy of the core without it runs with every hook off. A hook set in the
environment wins; set to an empty value, it stays off.

| Hook | Off (unset or empty) | On (tray-test's value) |
| --- | --- | --- |
| `HC_SETUP` | setup only checks `python3 -c 'import ttnn'` and uses the caller's `python3`; a failure still stops the run with exit 6 | a shell command that builds the env (`bash ./setup.sh`, the `./venv` python) |
| `HC_HOLDFILE` | no hold files are read; no host is held | a script that defines `hf_check` (the user's holds and allowlist; a held host runs nothing, exit 10) |
| `HC_HOLD` | `--hold` is refused with one line and the run goes on without it | a script that defines `hold_init`, `hold_point`, `hold_done` and `hold_broker` (`--hold`) |
| `HC_BROKER_YIELD` | the device is never given up to a queued broker job | a script that defines `by_on`, `by_start`, `by_check` and `by_stop` |
| `HC_SITE_FILE` | no site data; no site file is read, `~/.config/tray-test/site` included | a `:`-separated list of `key=value` files; `hc_site <key>` prints the first match (tray-test: `$TRAY_SITE_FILE`, default `~/.config/tray-test/site`, then `site.txt`) |

Site data keys read through `hc_site`:

- `glx_hosts`: an extended regex of short hostnames where the tray reset (`tt-smi -glx_reset`) may run.
  Unset, no host qualifies.
- `maxutil_skip`: the reason the max-util step is skipped on this box (printed as a WARNING).
- `maxutil_skip_tag`, `maxutil_skip_pv`: the text the skip WARNINGs add on this site. Unset, the core prints
  them in plain words.
- `temp_trip_src`: where the default thermal trip comes from, printed with each THERMAL line.
- `warn_<host>`: a WARNING the load test prints on that host before its load.
- `h3_tt_metal_build`: the folder holding the H3 tt-metal builds, one per pin.

ETH maps pick a host's platform from `H <platform> <regex>` lines in `fleet/eth_map_<platform>.txt`;
with no such map, every host reads `hc/fleet/eth_map.txt`.

The serving gate stays in `TRAY_REPRO_RESET_GATE`.

Other settings for running the core on its own (all unset by default, which keeps today's behaviour):

- `TRAY_MAXUTIL_BIN`: a built `unit_tests_didt_max_util`, used in place of the pinned max-util archive
  (`hc/fleet/maxutil.pin`). Its tt-metal tree is `TT_METAL_HOME`, else the first folder above the binary that
  has a `tt_metal/` folder.
- `H3_PYTHON_ENV`: a Python env folder (with `bin/python`). Set, `traytest/hc/h3/h3.sh` uses the tt-metal tree at
  `TT_METAL_HOME` with that env, and skips the pinned wheel, the site build and the archive.
- `TRAY_RUN_PARENT`: the folder `run.sh` makes its run folder in. Unset, the run folder goes beside `run.sh`.

`tests/test_hc_core_hooks_off.sh` copies the `CORE` files alone and checks that `run.sh --stub
--mode=smoke` passes with every hook off and runs the same stages as tray-test's own stub run.
`tests/test_hc_core_needs.sh` checks the settings above, the site-file rule and the plain WARNINGs without
a device.
