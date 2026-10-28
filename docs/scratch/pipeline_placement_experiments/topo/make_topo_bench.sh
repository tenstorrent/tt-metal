#!/usr/bin/env bash

# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# Build topo_solver_bench without libtt_metal: compile the topology solver's three source files
# straight into the benchmark. solve_topology_mapping is header-only (topology_solver.tpp); the
# non-template helpers live in topology_solver.cpp / topology_solver_sat.cpp /
# topology_solver_sat_solver.cpp. The parts of topology_solver.cpp that touch FabricNodeId,
# MeshGraph and the PSD are never called by the benchmark, so they are dropped by --gc-sections
# instead of being linked against the library.
#
# Usage: make_topo_bench.sh <build_dir> <output_binary> [source.cpp]
#   <build_dir> supplies third_party/libcadical.a (RelWithDebInfo/Release: optimized).
#   [source.cpp] defaults to topo_solver_bench.cpp; carve_footprint_bench.cpp builds the same way.
# Header-only dependencies: taken from the CPM cache when present, else from $TOPO_DEPS, a
# directory holding checkouts of fmt (11.1.4), spdlog (v1.15.2), tt-logger (v1.1.8), enchantum
# (8ca5b0e) and cadical (rel-2.2.1; header only). TT_METAL_HOME must point at the checkout.
set -euo pipefail
BUILD_DIR="${1:?build dir}"
OUT_BIN="${2:?output binary}"
SRC="${3:-topo_solver_bench.cpp}"
: "${TT_METAL_HOME:?set TT_METAL_HOME}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEPS="${TOPO_DEPS:-$HERE/../.out/deps}"
C="$TT_METAL_HOME/.cpmcache"
U="$TT_METAL_HOME/tt_metal/third_party/umd"

pick() {  # pick <name> <relative header> <cache glob> <deps dir> -> include dir
    local hdr="$2"
    for d in $3 "$4"; do
        [[ -f "$d/$hdr" ]] && { echo "$d"; return; }
    done
    echo "make_topo_bench: cannot find $hdr for $1 (checked CPM cache and $4)" >&2
    exit 1
}
FMT=$(pick fmt include/fmt/format.h "$C/fmt/*" "$DEPS/fmt")
SPDLOG=$(pick spdlog include/spdlog/spdlog.h "$C/spdlog/*" "$DEPS/spdlog")
TTLOG=$(pick tt-logger include/tt-logger/tt-logger.hpp "$C/tt-logger/*" "$DEPS/tt-logger")
ENCH=$(pick enchantum enchantum/include/enchantum/enchantum.hpp "$C/enchantum/*" "$DEPS/enchantum")
JSON=$(pick json include/nlohmann/json.hpp "$C/json/*" "$DEPS/json")
REFLECT=$(pick reflect reflect "$C/reflect/*" "$DEPS/reflect" 2>/dev/null || echo "$DEPS/reflect")  # header <reflect>; optional
CADICAL=$(pick cadical src/cadical.hpp "$C/cadical/*" "$DEPS/cadical")
LIB="${CADICAL_LIB:-$BUILD_DIR/third_party/libcadical.a}"
[[ -f "$LIB" ]] || { echo "make_topo_bench: no $LIB" >&2; exit 1; }

INC=(-I "$TT_METAL_HOME/tt_metal/api" -I "$TT_METAL_HOME/tt_stl" -I "$TT_METAL_HOME/tt_metal" -I "$TT_METAL_HOME/tt_metal/fabric"
    -I "$TT_METAL_HOME/tt_metal/hostdevcommon/api" -I "$TT_METAL_HOME/tt_metal/include"
    -I "$FMT/include" -I "$SPDLOG/include" -I "$TTLOG/include" -I "$ENCH/enchantum/include" -I "$JSON/include" -I "$REFLECT"
    -I "$U/device/api" -I "$U/tt-umd-utils/api" -I "$U/tt-umd/api" -I "$CADICAL/src")
DEFS=(-DFMT_HEADER_ONLY -DSPDLOG_FMT_EXTERNAL)
CXX="${CXX:-clang++-20}"
OBJ="$(dirname "$OUT_BIN")/topo_obj"
BENCH_OBJ="$OBJ/$(basename "$SRC" .cpp).o"
mkdir -p "$OBJ"
pids=()
for f in topology_solver topology_solver_sat topology_solver_sat_solver; do
    "$CXX" -std=c++20 -O2 -ffunction-sections -fdata-sections -c "${DEFS[@]}" "${INC[@]}" -o "$OBJ/$f.o" "$TT_METAL_HOME/tt_metal/fabric/$f.cpp" &
    pids+=($!)
done
"$CXX" -std=c++20 -O2 -ffunction-sections -fdata-sections -c "${DEFS[@]}" "${INC[@]}" -o "$BENCH_OBJ" "$HERE/$SRC" &
pids+=($!)
for p in "${pids[@]}"; do wait "$p"; done
"$CXX" -fuse-ld=lld -Wl,--gc-sections -o "$OUT_BIN" "$BENCH_OBJ" "$OBJ/topology_solver.o" "$OBJ/topology_solver_sat.o" "$OBJ/topology_solver_sat_solver.o" "$LIB" -lpthread
echo "built $OUT_BIN"
