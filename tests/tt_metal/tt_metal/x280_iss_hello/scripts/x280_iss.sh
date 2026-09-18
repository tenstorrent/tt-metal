#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
#
# Launch the X280 ISS.
#
# Blackhole L2CPU X280 cluster (ISA-level):
#   4 identical harts, rv64gcv, VLEN=512, ELEN=64
#   LIM SRAM at 0x08000000 (1.875 MiB), matching tt-llm-engine x280.h
#
# If X280_ISS is set to a vendor SiFive ISS binary, that is used instead.
# Otherwise Spike is invoked with the X280 ISA/memory configuration.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
WORK="${WORK:-$ROOT/build}"
PREFIX="$WORK/iss-prefix"

# SiFive X280 on Blackhole: rv64gcv, VLEN=512 (coreip_21G3 / llm-engine x280).
# zvl512b sets VLEN>=512 on modern Spike (no --varch flag).
X280_ISA="${X280_ISA:-rv64gcv_zicsr_zifencei_zvl512b}"
X280_HARTS="${X280_HARTS:-4}"
# LIM_BASE=0x08000000, LIM_SIZE=0x1E0000 (1.875 MiB)
X280_MEM="${X280_MEM:-0x8000000:0x1E0000}"

if [[ -n "${X280_ISS:-}" && -x "${X280_ISS}" ]]; then
    exec "$X280_ISS" "$@"
fi

SPIKE=""
if [[ -x "$PREFIX/bin/spike" ]]; then
    SPIKE="$PREFIX/bin/spike"
elif command -v spike >/dev/null; then
    SPIKE="$(command -v spike)"
else
    printf 'ERROR: no ISS. Run scripts/fetch_iss.sh or set X280_ISS.\n' >&2
    exit 1
fi

# Local dtc from fetch_iss.sh, if present.
if [[ -x "$WORK/dtc/usr/bin/dtc" ]]; then
    export PATH="$WORK/dtc/usr/bin:$PATH"
    export LD_LIBRARY_PATH="$WORK/dtc/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
fi

exec "$SPIKE" \
    --isa="$X280_ISA" \
    --disable-dtb \
    -p"$X280_HARTS" \
    -m"$X280_MEM" \
    --pc=0x8001000 \
    "$@"
