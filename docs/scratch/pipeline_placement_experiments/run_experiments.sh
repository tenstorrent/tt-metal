#!/usr/bin/env bash

# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# Handoff runner for the pipeline-placement experiments. See README.md for what each one checks.
#
#   ./run_experiments.sh setup       # venv + generated protobuf module + MGD list (once)
#   ./run_experiments.sh quick       # every experiment at its small size (a few minutes)
#   ./run_experiments.sh <name>      # one experiment: tier_b | tier_b_designed | tier_b_timing |
#                                    #   carve | carve_repo | planner | resolver | resolver_sat |
#                                    #   topo [all|multi|multi2|slots] | carve_footprint [quick|multi|repo|all]
#   ./run_experiments.sh <name> big  # carve, carve_repo, resolver, resolver_sat: the slow cases.
#                                    #   Minutes to hours; run on a spare machine, not a shared box.
#
# Everything runs on CPU and touches no hardware. Outputs go to $OUT (default ./.out, gitignored).
# Sizes are kept small by default; raise them with the environment knobs listed in README.md.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -z "${TT_METAL_HOME:-}" && -d "$HERE/../../../tt_metal/fabric" ]]; then
    TT_METAL_HOME="$(cd "$HERE/../../.." && pwd)"
fi
: "${TT_METAL_HOME:?outside a tt-metal checkout: set TT_METAL_HOME (used for the MGD files and the .proto)}"
export TT_METAL_HOME
OUT="${OUT:-$HERE/.out}"
VENV="$OUT/venv"
PY="$VENV/bin/python"
BUILD_DIR="${BUILD_DIR:-$TT_METAL_HOME/build_Debug}"

export PER_MGD="${PER_MGD:-3}"
export MAX_MESHES="${MAX_MESHES:-8}"
export CASE_TIMEOUT="${CASE_TIMEOUT:-1}"
export SEED="${SEED:-0}"
export MGD_LIST="$OUT/mgd_files.txt"
export PYTHONPATH="$OUT/py:$HERE/tier_b${PYTHONPATH:+:$PYTHONPATH}"

log() { printf '\n== %s\n' "$*"; }

setup() {
    mkdir -p "$OUT/py"
    if [[ ! -x "$PY" ]]; then
        python3 -m venv "$VENV"
        # 3.20.x reads the module protoc generates here; newer protobuf rejects it.
        "$VENV/bin/pip" install -q 'protobuf==3.20.3' python-sat
    fi
    protoc --experimental_allow_proto3_optional \
        -I "$TT_METAL_HOME/tt_metal/fabric/protobuf" --python_out="$OUT/py" \
        "$TT_METAL_HOME/tt_metal/fabric/protobuf/mesh_graph_descriptor.proto"
    (cd "$TT_METAL_HOME" && git ls-files '*.textproto' | grep -Ei 'mesh_graph|mgd') > "$MGD_LIST"
    echo "setup done: $(wc -l < "$MGD_LIST") MGD files, venv $VENV"
}

need_setup() { [[ -x "$PY" && -s "$MGD_LIST" ]] || setup; }

tier_b() {
    log "Tier B: simple checks vs full search on random pipelines (PER_MGD=$PER_MGD MAX_MESHES=$MAX_MESHES CASE_TIMEOUT=${CASE_TIMEOUT}s)"
    (cd "$OUT" && "$PY" "$HERE/tier_b/run.py" 2>/dev/null | tee "$OUT/tier_b.txt")
}

tier_b_designed() {
    log "Tier B: the ring each MGD is built for, plus the degree check (MAX_MESHES=$MAX_MESHES)"
    (cd "$OUT" && "$PY" "$HERE/tier_b/run2.py" 2>/dev/null | tee "$OUT/tier_b_designed.txt")
}

tier_b_timing() {
    log "Tier B: time split, simple checks vs full search on passing vs rejected cases"
    (cd "$OUT" && "$PY" "$HERE/tier_b/run3.py" 2>/dev/null | tee "$OUT/tier_b_timing.txt")
}

carve() {
    log "Carve inside one mesh: DFS vs CaDiCaL (${1:-quick}; each solve killed after ${CARVE_TIMEOUT:-20} s)"
    "$PY" "$HERE/carve/carve_bench.py" "${1:-quick}" | tee "$OUT/carve.txt"
}

carve_repo() {
    log "Carve on repo-shaped slice grids (4x8, 8x4, 2x16, 8x8); each solve killed after ${CARVE_TIMEOUT:-30} s"
    "$PY" "$HERE/carve/carve_bench_repo.py" "${1:-quick}" | tee "$OUT/carve_repo.txt"
}

# Four SAT encodings of the carve (positions, positions + symmetry breaking, slice-level grouping,
# grouping + symmetry) against the DFS, on larger and tighter grids. 'open' reruns the cases every
# solver capped; CARVE_ENCODINGS=dfs,pos limits the columns.
carve_encodings() {
    log "Carve SAT encodings vs DFS (${1:-quick}; each solve killed after ${CARVE_TIMEOUT:-20} s)"
    "$PY" "$HERE/carve/carve_bench_encodings.py" "${1:-quick}" | tee "$OUT/carve_encodings.txt"
}

# The same single-slice ring instances through the two C++ engines: the resolver search copied out
# of pipeline_builder.cpp, and solve_topology_mapping (needs libtt_metal from a build newer than
# the solver sources; set BUILD_DIR, default build_RelWithDebInfo for this group).
topo() {
    local bdir="${TOPO_BUILD_DIR:-$TT_METAL_HOME/build_RelWithDebInfo}"
    local group="${1:-all}"  # all | multi | multi2 | slots | quick
    log "resolver vs solve_topology_mapping, group $group (${TOPO_TIMEOUT:-60} s cap per solve)"
    extract_resolver
    g++ -O2 -std=c++20 -I "$OUT/resolver" -I "$HERE/topo" -o "$OUT/resolver_same_cases" "$HERE/topo/resolver_same_cases.cpp"
    "$OUT/resolver_same_cases" "$group" | tee "$OUT/topo_resolver_$group.txt"
    if "$HERE/topo/make_topo_bench.sh" "$bdir" "$OUT/topo_solver_bench"; then
        "$OUT/topo_solver_bench" "$group" | grep -v "| info" | tee "$OUT/topo_solver_$group.txt"
    else
        echo "topo: could not build topo_solver_bench (needs libcadical.a from a tt-metal build and the header-only deps); skipping"
    fi
}

# The carve as a footprint mapping through solve_topology_mapping (Auto, SAT, DFS): one global node
# per stage-shape position, the pipeline as the target, a resource constraint for disjointness.
# Built like topo_solver_bench; leaves the topo group untouched so the two can be compared.
carve_footprint() {
    local bdir="${TOPO_BUILD_DIR:-$TT_METAL_HOME/build_RelWithDebInfo}"
    local group="${1:-quick}"  # quick | multi | repo | all
    log "carve as footprint mapping, group $group (${TOPO_TIMEOUT:-20} s cap per solve)"
    "$HERE/topo/make_topo_bench.sh" "$bdir" "$OUT/carve_footprint_bench" carve_footprint_bench.cpp
    "$OUT/carve_footprint_bench" "$group" | grep -v "| info" | tee "$OUT/carve_footprint_$group.txt"
}

planner() {
    : "${TT_BLAZE:?set TT_BLAZE to a tt-blaze checkout}"
    log "Blaze carve planner vs a brute-force ring of rectangles (TT_BLAZE=$TT_BLAZE)"
    TT_BLAZE="$TT_BLAZE" "$PY" "$HERE/carve/carve_probe.py" | tee "$OUT/planner.txt"
}

# bench.cpp includes the resolver's search verbatim: everything in the anonymous namespace of
# tt::tt_fabric::detail in pipeline_builder.cpp, through resolve_pipeline_placement. It is copied out
# of the checkout at build time, so the benchmark always measures the current code.
extract_resolver() {
    mkdir -p "$OUT/resolver"
    awk '
        /^namespace tt::tt_fabric::detail \{$/ { in_detail = 1; next }
        in_detail && /^namespace \{$/ { copying = 1 }
        copying && /^\}  \/\/ namespace tt::tt_fabric::detail$/ { exit }
        copying { print }
    ' "$TT_METAL_HOME/tt_metal/fabric/pipeline_builder.cpp" >"$OUT/resolver/search_body.inc"
    grep -q resolve_pipeline_placement "$OUT/resolver/search_body.inc" ||
        { echo "could not extract the resolver from pipeline_builder.cpp" >&2; exit 1; }
}

resolver() {
    local groups=(feas cables)
    [[ "${1:-}" == big ]] && groups=(odd big)
    log "pipeline_builder.cpp resolver search on synthetic grids, groups ${groups[*]}; 60 s cap per group"
    extract_resolver
    g++ -O2 -std=c++20 -I "$OUT/resolver" -o "$OUT/resolver_bench" "$HERE/resolver/bench.cpp"
    : > "$OUT/resolver.txt"
    for group in "${groups[@]}"; do
        timeout 60 "$OUT/resolver_bench" "$group" | tee -a "$OUT/resolver.txt" || echo "  ($group: stopped at 60 s)"
    done
    [[ "${1:-}" == big ]] || echo "  skipped: 'odd' (5x7 takes > 120 s) and 'big' (> 120 s); run: $0 resolver big"
}

# CaDiCaL comes from the tt-metal build: the header from the CPM source checkout and the archive from
# $BUILD_DIR/third_party. CADICAL_SRC overrides the source dir (the CPM cache can be gone while the
# archive is still there), CADICAL_LIB the archive. A Debug build's archive is unoptimized and makes
# every SAT time here about 10x slower; use a RelWithDebInfo/Release build, or build CaDiCaL with -O3
# yourself and point CADICAL_LIB at it.
resolver_sat() {
    local src="${CADICAL_SRC:-}"
    [[ -n "$src" ]] || src="$(sed -n 's/^CPM_PACKAGE_cadical_SOURCE_DIR:INTERNAL=//p' "$BUILD_DIR/CMakeCache.txt" 2>/dev/null || true)"
    local lib="${CADICAL_LIB:-$BUILD_DIR/third_party/libcadical.a}"
    if [[ ! -f "$src/src/cadical.hpp" || ! -f "$lib" ]]; then
        echo "resolver_sat: needs cadical.hpp (CADICAL_SRC=$src) and libcadical.a (CADICAL_LIB=$lib); skipping"
        return 0
    fi
    if [[ -z "${CADICAL_LIB:-}" ]] && grep -q '^CMAKE_BUILD_TYPE:STRING=Debug$' "$BUILD_DIR/CMakeCache.txt" 2>/dev/null; then
        echo "resolver_sat: $BUILD_DIR is a Debug build; its CaDiCaL is unoptimized (times about 10x slower)"
    fi
    log "CaDiCaL encoding of the same ring placement ${1:-quick}; 60 s cap (the big case takes > 230 s)"
    g++ -O2 -std=c++20 -I "$src/src" -o "$OUT/resolver_sat" "$HERE/resolver/sat.cpp" "$lib"
    timeout 60 "$OUT/resolver_sat" "${1:-quick}" | tee "$OUT/resolver_sat.txt" || echo "  (stopped at 60 s)"
}

quick() {
    tier_b
    tier_b_designed
    carve
    resolver
    resolver_sat
    if [[ -n "${TT_BLAZE:-}" ]]; then planner; else echo "planner: set TT_BLAZE to include it"; fi
}

cmd="${1:-quick}"
case "$cmd" in
    setup) setup ;;
    quick | tier_b | tier_b_designed | tier_b_timing | carve | carve_repo | carve_encodings | planner | resolver | resolver_sat | topo)
        need_setup
        "$cmd" "${@:2}"
        ;;
    carve_footprint) "$cmd" "${@:2}" ;;  # C++ only, no venv needed
    *)
        sed -n '2,13p' "$0"
        exit 1
        ;;
esac
