#!/bin/bash
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Sweep a ttnn op test instead of an LLK kernel test.
#
#   ./metal.sh path/to/test_op.py [pytest args...]
#   ./metal.sh 'path/to/test_op.py::test_case[params]'
#
# Env (see README): TTNOP_DELAYS TTNOP_THREADS TTNOP_SITE_MODE TTNOP_FILLER
#                   TTNOP_SITES TTNOP_REPEATS TTNOP_METAL_KERNEL
#
# The cave has to exist in the kernel image before any of this works:
#   make metal_cave     (once, and again after rebuilding the hw_toolchain target)

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export TT_METAL_HOME="${TT_METAL_HOME:-$(cd "$HERE/../../../../.." && pwd)}"

if [[ $# -eq 0 ]]; then
    echo "usage: $0 <test file or nodeid> [pytest args...]" >&2
    exit 4
fi

# ttnn first (so `import ttnn` hits ttnn/ttnn, not the outer namespace), then
# tools/ (tracy), then ttnop itself so the plugin resolves by bare name, then the
# repo root for models/ and tests/.
export PYTHONPATH="$TT_METAL_HOME/ttnn:$TT_METAL_HOME/tools:$HERE:$TT_METAL_HOME${PYTHONPATH:+:$PYTHONPATH}"
# Prefer build_Release/{tt_metal,ttnn} over the possibly-stale build_Release/lib copies.
export LD_LIBRARY_PATH="$TT_METAL_HOME/build_Release/tt_metal:$TT_METAL_HOME/build_Release/ttnn:$TT_METAL_HOME/build_Release/tt_stl:$TT_METAL_HOME/build_Release/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

export TTNOP_METAL=1
# ttnop reads the arch as CHIP_ARCH; tt-metal calls it ARCH_NAME. Reports only.
export CHIP_ARCH="${CHIP_ARCH:-${ARCH_NAME:-wormhole}}"
# Required. Under fast dispatch the image is staged into a DRAM buffer on the first
# enqueue, so every later poke into the host image is invisible.
export TT_METAL_SLOW_DISPATCH_MODE="${TT_METAL_SLOW_DISPATCH_MODE:-1}"
# The scan reads the post-XIP dump metal writes beside each kernel ELF.
unset TT_METAL_DISABLE_XIP_DUMP

make --silent -C "$HERE" scan
make --silent -C "$HERE" metal_shim

echo ">> delays=${TTNOP_DELAYS:-1-100} threads=${TTNOP_THREADS:-unpack,math}" \
     "sites=${TTNOP_SITE_MODE:-sync} filler=${TTNOP_FILLER:-auto}"
echo ">> target=$1 kernel=${TTNOP_METAL_KERNEL:-<most recently loaded>}"

# A found race turns the case red, so a non-zero exit is a result, not a crash.
exec python3 -m pytest -p ttnop_plugin -p no:randomly -q "$@"
