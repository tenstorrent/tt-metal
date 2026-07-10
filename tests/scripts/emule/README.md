<!-- SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# tt-emule smoke test

[tt-emule](https://github.com/tenstorrent/tt-emule) is a C++ software emulator of
the Tenstorrent device APIs. It lets tt-metal kernel and host code run on a plain
x86-64 Linux host with no hardware, integrating through
`-DTT_METAL_USE_EMULE=ON`. The host-side integration lives in this repo
(`tt_metal/impl/emulation/`, plus the `TargetDevice::Emule` guards in
`tt_metal/distributed/`).

## What this catches

A tt-metal change can break the emule integration while every silicon and mock
test stays green — the emule path simply isn't exercised by tt-metal CI. (For
example, a guard change routed `TargetDevice::Emule` through the RISC-V kernel
compile path, which emule cannot satisfy; it was only caught downstream in the
tt-emule regression, not on the tt-metal PR.) `run_emule_smoke.sh` runs a small
emule build-and-run so that class of break fails on the tt-metal side.

## Usage

```bash
# Build tt-metal with the emule backend and run wormhole + blackhole smoke sets.
tests/scripts/emule/run_emule_smoke.sh

# One arch only:
tests/scripts/emule/run_emule_smoke.sh wormhole

# Reuse an already-built tree and/or a local tt-emule checkout:
EMULE_SKIP_BUILD=1 TT_EMULE_PATH=/path/to/tt-emule \
  tests/scripts/emule/run_emule_smoke.sh
```

If `TT_EMULE_PATH` is not set, the script clones `tenstorrent/tt-emule@main`
into `$HOME/.cache/tt-metal-emule-smoke/tt-emule` (refreshed on re-run). See the
script header for all environment variables.

**Prerequisites:** clang-20 + libstdc++ (gcc-13+), CMake ≥ 3.24, Ninja, ccache.
The `ghcr.io/tenstorrent/tt-mlir/tt-mlir-ci-ubuntu-24-04` image has them all.

## What it runs

1. Resolves a tt-emule checkout (supplies `include/jit_hw` + `include/tt_emule`).
2. Initializes the `umd` and `tracy` submodules.
3. Builds tt-metal with `-DTT_METAL_USE_EMULE=ON` via tt-emule's maintained
   `ci-build.sh` (clang-20 + libstdc++), falling back to an inline `cmake`
   recipe if that script is absent.
4. Runs the per-arch smoke sets **sequentially** (they share a JIT cache):
   - **wormhole** — the curated PR tier (`CI_TIER=pr`): tilize/format, L1 + DRAM
     buffers, a Tensix JIT kernel, DRAM channels, a DM loopback, a ttnn reduction.
   - **blackhole** — the full (small) suite: host-only checks + INT32 ttnn ops.

The build recipe and test lists are owned by tt-emule and reused as-is, so the
smoke set tracks upstream automatically and this stays a thin driver.

## Notes

- Not wired into a GitHub Actions workflow yet — this is a runnable entry point
  for local use and as the basis for a future CI job.
- It builds tt-emule `main` against the current tt-metal checkout, so unrelated
  drift between the two `main` branches can occasionally surface here. Pin
  `TT_EMULE_REF` to a known-good ref if you need a stable baseline.
