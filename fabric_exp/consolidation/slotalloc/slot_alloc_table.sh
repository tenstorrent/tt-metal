#!/bin/bash
# Build and run the offline slot allocation table (SLOT_ALLOCATOR.md). CPU only, no device, no tree build: compiles
# slot_alloc_table.cpp with the host's allocator source into ~/.tt-metal-slotalloc/ and runs it with the given args
# (--sizes, --policy, --reasons, --only; see slot_alloc_table.cpp).
set -e
TREE=/home/aagarwal/tt-metal
OUT=${SLOTALLOC_BIN:-/home/aagarwal/.tt-metal-slotalloc}
mkdir -p "$OUT"
SRC=(fabric_exp/consolidation/slotalloc/slot_alloc_table.cpp tt_metal/fabric/builder/fabric_slot_allocator.cpp
     tt_metal/fabric/builder/fabric_trid_allocator.cpp)
BIN=$OUT/slot_alloc_table
newer=0
for f in "${SRC[@]}" tt_metal/fabric/builder/fabric_slot_allocator.hpp tt_metal/fabric/builder/fabric_trid_allocator.hpp; do [ "$TREE/$f" -nt "$BIN" ] && newer=1; done
if [ ! -x "$BIN" ] || [ $newer = 1 ]; then
    g++ -std=c++17 -O1 -Wall -Wextra -Werror -I"$TREE" "${SRC[@]/#/$TREE/}" -o "$BIN"
fi
exec "$BIN" "$@"
