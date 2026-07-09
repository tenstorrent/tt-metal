#!/usr/bin/env bash
#
# Prime the Blackhole X280 (L2CPU) L3 LIM ECC so the SPSC kernel-profiler drainer firmware can run.
#
# Why this is needed: on a fresh board (or after any cold power cycle) the SRAM backing the X280's
# LIM has uninitialized ECC, so the drainer FW's loads/stores fault silently and it never starts
# (the real-time profiler logs "X280 drainer FW did not start ... heartbeat=0x0"). This is a
# device-only step that CANNOT be folded into metal init: priming requires routing L3 through the
# cache controller to write valid data+ECC into the SRAM (test_x280_profcons --primeecc), and then
# a FULL ASIC reset to activate it (WayEnable=0xF is irreversible until that reset — an L2CPU reset
# does NOT clear it). The primed ECC persists across the reset and across subsequent `tt-smi -r`,
# but NOT across a cold power cycle — so run this once after each cold power cycle.
#
# Usage:  ./prime_x280_ecc.sh [device] [l2cpu]     (defaults: device 0, l2cpu 0)
#
# See also: tools/x280_bm/PRIMER.md, and the --primeecc branch of test_x280_profcons.cpp.

set -euo pipefail

device="${1:-0}"
l2cpu="${2:-0}"

# Resolve the tt-metal repo root from this script's location:
# tt_metal/programming_examples/profiler/test_x280_profcons/ -> repo root is 4 dirs up.
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root="$(cd "$script_dir/../../../.." && pwd)"
cd "$root"
export TT_METAL_HOME="$PWD"
export ARCH_NAME=blackhole

bin="build_Release/programming_examples/profiler/test_x280_profcons"
if [ ! -x "$bin" ]; then
    echo "[prime-x280] $bin not built — building ..."
    ninja -C build_Release test_x280_profcons
fi

# Metal shared libs aren't on the default loader path for a bare invocation; add them.
export LD_LIBRARY_PATH="$(find build_Release -name '*.so*' -type f -exec dirname {} \; 2>/dev/null \
    | sort -u | tr '\n' ':')${LD_LIBRARY_PATH:-}"

echo "[prime-x280] priming L3 LIM ECC (device $device, l2cpu $l2cpu) ..."
"./$bin" --primeecc --l2cpu "$l2cpu" --device "$device"

echo "[prime-x280] activating via ASIC reset: tt-smi -r $device ..."
tt-smi -r "$device"

echo "[prime-x280] done — X280 LIM ECC primed (valid until the next cold power cycle)."
