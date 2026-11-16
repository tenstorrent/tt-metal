<!--
SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

SPDX-License-Identifier: Apache-2.0
-->

# Keeping the hc2 copy in sync

This folder is the core of the hc2 health check. It is developed in the tray-test repository
(`traytest/hc/`) and copied byte-identically into tt-metal at `<hc2>/core/`, where `<hc2>` is the
hc2 folder in tt-metal (`HC2_DEST`, default set in `traytest/tools/hc2_sync.sh`).
The copy is a plain folder, not a submodule, so a normal clone of tt-metal has everything it needs.

`<hc2>/SYNC` records where the copy came from: the source repository, the source
path, the source commit, and the sha256 of every file. Do not edit it by hand.

Binaries the glue supplies, and files of tray-test's own routes, are never copied: `hc2_sync.sh` leaves out every path in its `EXCLUDE` rule and writes
one `exclude:` line per path to SYNC. Today that is `fleet/pv/run-ttx`, the power virus binary. The core finds it through
`HC_RUN_TTX` (else `fleet/pv/run-ttx` beside itself); when it is absent, run.sh prints a WARNING as its first line, at
each power virus part and in the summary, and runs the bus-bar chunk load in its place.

The second path, `h3/requirements-wheel.txt`, is left out too: it holds the pip pins of tray-test's own wheel route (`h3.sh wheel_setup`),
which hc2 never runs (the glue sets `H3_TT_METAL_BUILD` and uses the checkout's `python_env`). Keeping it out of metal
keeps a pin that cannot be installed there (the ttnn wheel is CPython 3.10) out of metal's dependency scan.

Only `core/` and `SYNC` are managed. Everything else under `<hc2>/` (for example `glue/`) belongs to
tt-metal and is never touched.

## Commands

Run these from a tray-test checkout. `<metal>` is the root of a tt-metal checkout.

| Command | What it does | Exit codes |
|---|---|---|
| `traytest/tools/hc2_sync.sh check <metal>` | Lists files changed in tray-test since the SYNC commit, files changed in tt-metal against the SYNC hashes, and files changed on both sides. | 0 in sync, 1 drift, 2 error |
| `traytest/tools/hc2_sync.sh import <metal>` | For each file changed in tt-metal, does a 3-way merge (`git merge-file`) into `traytest/hc/`. The base is the core at the SYNC commit. New and deleted files are carried over. Conflicts are reported and both sides are kept. | 0 merged, 1 conflicts left, 2 error |
| `traytest/tools/hc2_sync.sh export <metal>` | Copies the core at tray-test's HEAD to `<metal>/<hc2>/core/` and rewrites SYNC. It refuses when the core has uncommitted changes, or when tt-metal has changes that are not in tray-test's HEAD yet. `--force` drops those tt-metal changes. It also refuses, with one line, when HEAD is on no `origin/*` branch (SYNC must name a commit readers can fetch); `--allow-unpublished` exports it anyway with a WARNING. | 0 done, 2 refused or error |

Environment overrides: `HC2_CORE` (core path in tray-test, default `traytest/hc`), `HC2_DEST`
(the `<hc2>` path in tt-metal), `HC2_SOURCE` (repository name written
to SYNC).

## Workflow

- **Update tt-metal from tray-test:** commit in tray-test, run `check`, then `export`, then commit the
  result in tt-metal as one change.
- **Bring a fix made in tt-metal back:** run `import`, resolve any conflicts (look for `<<<<<<<`
  markers), commit in tray-test, then `export` so SYNC points at the new commit.
- **Review drift:** `check`, or simply `diff -r traytest/hc <metal>/<hc2>/core`.

Python caches (`__pycache__`, `*.pyc`) in the tt-metal copy are ignored. Only file contents are
hashed, so a permission-only change in tt-metal is not reported.
