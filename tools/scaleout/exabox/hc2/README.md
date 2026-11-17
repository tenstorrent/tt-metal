<!--
SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

SPDX-License-Identifier: Apache-2.0
-->

# hc2: single-galaxy hardware health check

hc2 checks one Blackhole galaxy (32 chips in 4 trays) and names any failing part by PCIe address (BDF) and tray.
It is tray-test's regular `run.sh`. It runs as one host-local test of about 30 minutes, needs no other host, and ends with:

- a verdict per tray,
- the chips and trays lost,
- a `Repair recommendation:` line.

## How it differs from `health_check_test_suite/` and `run_diag.sh`

| | `health_check_test_suite/run_diag.sh` | hc2 |
|---|---|---|
| Scope | cluster: MPI across hosts, fabric and dispatch tests | one galaxy host, nothing off the box |
| Load | functional tests | power and bus-bar load stages (max-util, power virus), numeric checks, link tests, synthetic H3 |
| Result | test pass/fail per suite | verdict per tray, fault named by BDF and tray, cause and repair line |

`run_diag.sh` and the rest of `health_check_test_suite/` are not changed by hc2. The two are run side by side.

## Run

Build tt-metal and its python_env first, either with `./build_metal.sh` and `./create_venv.sh` or with
`tools/scaleout/exabox/build_metal_exabox.sh`. Then run:

```
tools/scaleout/exabox/hc2/glue/run_hc2.sh               # plain run, about 30 min
tools/scaleout/exabox/hc2/glue/run_hc2.sh --mode=full   # same stages, longer load
tools/scaleout/exabox/hc2/glue/run_hc2.sh --stub        # fake galaxy, no device: proves the code paths only
```

Run folders go to `$TT_METAL_HOME/generated/hc2/`, or to `HC2_OUT_DIR` if it is set.

## Layout

- `core/`: the test. A byte-identical copy of tray-test's `traytest/hc`, written by tray-test's
  `traytest/tools/hc2_sync.sh export`. Do not edit it here. Change it in tray-test and export again.
- `SYNC`: the manifest of that copy: source repo, path, commit and one sha256 per file.
- `glue/`: the tt-metal side, owned here.
  - `run_hc2.sh` is the entry point.
  - `check_env.sh` is the env check it hands to the core.
- `tests/`: deviceless pytest checks.

## What the glue sets

- `TT_METAL_HOME` is this checkout, whatever was set before.
- The Python is tt-metal's own `python_env` (`PYTHON_ENV_DIR` moves it). hc2 runs no pip install and uses no pinned wheel.
- `HC_SETUP` is `glue/check_env.sh`. It passes only when `import ttnn` resolves to a file inside `TT_METAL_HOME`.
- `HC_HOLDFILE`, `HC_HOLD` and `HC_SITE_FILE` stay unset, so hold scripts and site data are off. A site may set them.
- The glue refuses a core without the `HC_SETUP` hook (outside `--stub`), because such a core installs ttnn itself.
- Synthetic H3 runs on this checkout, through a view at `<out dir>/.h3-build`, not on a pinned tt-metal sha.
- max-util runs the core's pinned prebuilt archive (`core/fleet/maxutil.pin`), the same build and load tray-test runs.
  The core finds the archive in `~/tray-test-dist`, `/data/$USER/tray-test-dist` or `TRAY_MAXUTIL_DIST`; without it
  the step ends INCOMPLETE with a setup error, never a silent skip. This checkout's own
  `build/test/didt/unit_tests_didt_max_util` is not the default: tt-metal's fast dispatch now refuses kernels on active
  Ethernet cores, and the workload streams DRAM from them. A caller may still set `TRAY_MAXUTIL_BIN`.
  There is no shared download location for the archive yet, and the pinned tree is not published. The archive was
  built from a tt-metal tree that predates the fast-dispatch refusal above, with a per-core CRC-32 log added to the
  test and the SFPI version bumped to 7.85.0. To pack your own, build `unit_tests_didt_max_util` in a Release build of
  such a tree (one from before that refusal, not this checkout), then run
  `core/fleet/maxutil_pack.sh pack <tt-metal dir> <out dir> [<mpi prefix>]`. It writes `maxutil-<sha12>.tar.zst`
  into `<out dir>`, which must be one of the dirs above (`TRAY_MAXUTIL_DIST` takes a colon-separated list). A build of
  this checkout packs and passes the pin check, but cannot run the load. The step accepts an archive only when its
  sha256 is the one in `maxutil.pin`, so a build of your own also needs `maxutil.pin` replaced with the lines
  `maxutil_pack.sh pin <archive>` prints. Such a build is not the pinned one: without the per-core CRC-32 log the
  output-CRC vote has nothing to compare on it. `maxutil_pack.sh verify <archive>` and `check <archive> <dir>` test an
  archive without opening a device.

## Resets

hc2 never runs `tt-smi -r`. Its resets are `tt-smi -glx_reset` followed by a rescan of the root ports and a recount
of the chips by BDF and tray.

## Tests

```
pytest tools/scaleout/exabox/hc2/tests
```

The tests need no device. They check that:

- `SYNC` matches `core/`;
- a `--stub --mode=smoke` run through `glue/run_hc2.sh` finishes in under 60 s;
- the glue sets only `HC_SETUP` and the env check works;
- every hc2 file outside `core/` carries SPDX.

`HC2_CORE` points the stub run at another core, either a `core/` copy or a tray-test checkout.
