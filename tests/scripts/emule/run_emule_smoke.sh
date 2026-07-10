#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
#
# tt-emule smoke test — build tt-metal (the current checkout) with the tt-emule
# software backend and run the curated wormhole + blackhole smoke sets.
#
# Why this exists
# ---------------
# The emule host-integration lives in this repo (tt_metal/impl/emulation/, the
# TargetDevice::Emule guards in tt_metal/distributed/). A tt-metal change can
# break it while every silicon/mock test stays green, because tt-metal CI runs
# no emule tests — the dependency is one-way (the tt-emule repo pins and tests a
# tt-metal SHA, not the reverse). This script is the minimal signal in the other
# direction: run it against a tt-metal change to catch an emule-integration
# break here, instead of downstream in the tt-emule regression.
#
# The actual build recipe and test lists are owned by tt-emule and reused as-is
# (its .github/scripts/ci-build.sh and scripts/run_regression_<arch>.sh), so the
# smoke set tracks upstream automatically and this script stays a thin driver.
#
# Usage
# -----
#   tests/scripts/emule/run_emule_smoke.sh [arch ...]
#
#   arch            wormhole | blackhole (default: both, run sequentially —
#                   they share a JIT cache and cannot run concurrently)
#
# Environment
# -----------
#   TT_METAL_HOME     tt-metal source root         (default: git root of this script)
#   TT_EMULE_PATH     existing tt-emule checkout    (default: clone TT_EMULE_REPO@TT_EMULE_REF)
#   TT_EMULE_REPO     repo to clone if no checkout  (default: https://github.com/tenstorrent/tt-emule)
#   TT_EMULE_REF      ref to clone / fetch          (default: main)
#   BUILD_DIR         build tree                    (default: $TT_METAL_HOME/build_emule)
#   EMULE_SKIP_BUILD  =1 to reuse an existing build (skip configure + compile)
#
# Prerequisites: clang-20 + libstdc++ (gcc-13+), CMake >= 3.24, Ninja, ccache,
# and network access to clone tt-emule when TT_EMULE_PATH is not supplied. The
# tt-mlir CI docker image (ghcr.io/tenstorrent/tt-mlir/tt-mlir-ci-ubuntu-24-04)
# ships all of these.

set -euo pipefail

log()  { printf '\n\033[1;34m== %s ==\033[0m\n' "$*"; }
warn() { printf '\033[1;33mWARN: %s\033[0m\n' "$*" >&2; }
die()  { printf '\033[1;31mERROR: %s\033[0m\n' "$*" >&2; exit 1; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------
TT_METAL_HOME="${TT_METAL_HOME:-$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null)}"
[ -n "${TT_METAL_HOME:-}" ] && [ -d "$TT_METAL_HOME" ] || die "cannot locate tt-metal root; set TT_METAL_HOME"
TT_METAL_HOME="$(cd "$TT_METAL_HOME" && pwd)"

BUILD_DIR="${BUILD_DIR:-$TT_METAL_HOME/build_emule}"
TT_EMULE_REPO="${TT_EMULE_REPO:-https://github.com/tenstorrent/tt-emule}"
TT_EMULE_REF="${TT_EMULE_REF:-main}"
EMULE_SKIP_BUILD="${EMULE_SKIP_BUILD:-0}"

# Arches: positional args override the default of both, in the required order.
if [ "$#" -gt 0 ]; then
    ARCHES=("$@")
else
    ARCHES=(wormhole blackhole)
fi
for a in "${ARCHES[@]}"; do
    case "$a" in
        wormhole|blackhole) ;;
        *) die "unknown arch '$a' (expected: wormhole | blackhole)" ;;
    esac
done

log "tt-emule smoke test"
echo "  TT_METAL_HOME: $TT_METAL_HOME"
echo "  BUILD_DIR:     $BUILD_DIR"
echo "  arches:        ${ARCHES[*]}"

# ---------------------------------------------------------------------------
# Resolve a tt-emule checkout (supplies include/jit_hw + include/tt_emule and
# the build/test scripts). Use TT_EMULE_PATH if given; otherwise clone/refresh
# a cached checkout under $HOME/.cache.
# ---------------------------------------------------------------------------
if [ -n "${TT_EMULE_PATH:-}" ]; then
    [ -d "$TT_EMULE_PATH" ] || die "TT_EMULE_PATH does not exist: $TT_EMULE_PATH"
    TT_EMULE_PATH="$(cd "$TT_EMULE_PATH" && pwd)"
    log "Using tt-emule at $TT_EMULE_PATH"
else
    TT_EMULE_PATH="${TT_EMULE_CACHE:-$HOME/.cache/tt-metal-emule-smoke/tt-emule}"
    log "Fetching tt-emule ($TT_EMULE_REPO @ $TT_EMULE_REF) into $TT_EMULE_PATH"
    if [ -d "$TT_EMULE_PATH/.git" ]; then
        git -C "$TT_EMULE_PATH" fetch --depth=1 origin "$TT_EMULE_REF"
        git -C "$TT_EMULE_PATH" checkout -q FETCH_HEAD
    else
        mkdir -p "$(dirname "$TT_EMULE_PATH")"
        git clone --depth=1 --branch "$TT_EMULE_REF" "$TT_EMULE_REPO" "$TT_EMULE_PATH"
    fi
fi

CI_BUILD="$TT_EMULE_PATH/.github/scripts/ci-build.sh"
WH_SCRIPT="$TT_EMULE_PATH/scripts/run_regression_wormhole.sh"
BH_SCRIPT="$TT_EMULE_PATH/scripts/run_regression_blackhole.sh"
[ -f "$WH_SCRIPT" ] || die "missing $WH_SCRIPT — is $TT_EMULE_PATH a tt-emule checkout?"
[ -f "$BH_SCRIPT" ] || die "missing $BH_SCRIPT — is $TT_EMULE_PATH a tt-emule checkout?"

# ---------------------------------------------------------------------------
# Submodules: emule needs UMD (SWEmuleChip + the cluster_descriptor_examples the
# regression scripts read) and tracy. Init shallow; no-op if already present.
# ---------------------------------------------------------------------------
log "Initializing tt-metal submodules (umd, tracy)"
git -C "$TT_METAL_HOME" submodule update --init --depth=1 tt_metal/third_party/umd tt_metal/third_party/tracy

CLUSTER_EXAMPLES="$TT_METAL_HOME/tt_metal/third_party/umd/tests/cluster_descriptor_examples"
[ -f "$CLUSTER_EXAMPLES/wormhole_N150.yaml" ] || warn "wormhole_N150.yaml not found under $CLUSTER_EXAMPLES"
[ -f "$CLUSTER_EXAMPLES/blackhole_P100.yaml" ] || warn "blackhole_P100.yaml not found under $CLUSTER_EXAMPLES"

# ---------------------------------------------------------------------------
# Build tt-metal with the emule backend. Delegate to tt-emule's maintained
# recipe when present (single source of truth); otherwise inline an equivalent.
# ---------------------------------------------------------------------------
if [ "$EMULE_SKIP_BUILD" = "1" ]; then
    log "EMULE_SKIP_BUILD=1 — reusing existing build at $BUILD_DIR"
elif [ -f "$CI_BUILD" ]; then
    log "Building via tt-emule ci-build.sh"
    TT_EMULE_DIR="$TT_EMULE_PATH" TT_METAL_DIR="$TT_METAL_HOME" BUILD_DIR="$BUILD_DIR" \
        bash "$CI_BUILD"
else
    warn "ci-build.sh not found in tt-emule checkout — using inline build recipe"
    log "Configuring tt-metal (emule ON, clang-20 + libstdc++)"
    cmake -B "$BUILD_DIR" -S "$TT_METAL_HOME" -G Ninja \
        -DCMAKE_TOOLCHAIN_FILE="$TT_METAL_HOME/cmake/x86_64-linux-clang-20-libstdcpp-toolchain.cmake" \
        -DCMAKE_BUILD_TYPE=Release \
        -DTT_METAL_USE_EMULE=ON \
        -DTT_EMULE_PATH="$TT_EMULE_PATH" \
        -DWITH_PYTHON_BINDINGS=ON \
        -DENABLE_TRACY=OFF \
        -DENABLE_DISTRIBUTED=OFF \
        -DTT_METAL_BUILD_TESTS=ON \
        -DTTNN_BUILD_TESTS=ON \
        -DTT_INSTALL=OFF \
        -DTT_USE_SYSTEM_SFPI=OFF
    log "Building smoke-test binaries"
    cmake --build "$BUILD_DIR" -j"$(nproc)" --target \
        unit_tests_api unit_tests_data_movement unit_tests_ttnn ttnn
fi

# ---------------------------------------------------------------------------
# Run the per-arch smoke sets sequentially (shared JIT cache). Wormhole runs its
# curated PR tier (CI_TIER=pr); blackhole runs its full suite (small — Tier 1 +
# INT32 Tier 4). A failure in either arch fails the whole run.
# ---------------------------------------------------------------------------
declare -A RESULT
overall=0

run_arch() {
    local arch="$1" script="$2"; shift 2
    log "Regression: $arch"
    if TT_METAL_DIR="$TT_METAL_HOME" BUILD_DIR="$BUILD_DIR" "$@" bash "$script"; then
        RESULT[$arch]=PASS
    else
        RESULT[$arch]=FAIL
        overall=1
    fi
}

for arch in "${ARCHES[@]}"; do
    case "$arch" in
        wormhole)  run_arch wormhole  "$WH_SCRIPT" env CI_TIER=pr ;;
        blackhole) run_arch blackhole "$BH_SCRIPT" env ;;
    esac
done

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
log "tt-emule smoke summary"
for arch in "${ARCHES[@]}"; do
    printf '  %-10s %s\n' "$arch" "${RESULT[$arch]:-SKIP}"
done

if [ "$overall" -ne 0 ]; then
    die "tt-emule smoke test FAILED"
fi
echo "tt-emule smoke test PASSED"
