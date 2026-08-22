#!/usr/bin/env bash
# Dump the inter-mesh base-embedding CNF for one 2x4 pipeline stage count onto SC36, via the 36-rank producer.
# Usage: dump_cnf.sh <stages> <out.cnf> <paths_file>   (paths_file = 36 lines of per-rank mock cluster-desc paths)
set -eu
STAGES=$1; CNF=$2; PATHS_FILE=$3; ROOT=$(git rev-parse --show-toplevel); cd "$ROOT"
GRB=$(readlink -f build/tools/scaleout/generate_rank_bindings)
LIB=$(readlink -f build/tt_metal); UMD=$(readlink -f build/tt_metal/third_party/umd/lib)
mapfile -t PATHS < "$PATHS_FILE"
MGD=$(readlink -f "generated/mgd/pipeline_sweep/sweep_2x4_pipeline_${STAGES}stage_mesh_graph_descriptor.textproto")
OUT=$(mktemp -d); rm -f "$CNF"
CMD=(mpirun --allow-run-as-root --oversubscribe --tag-output
   -x TT_METAL_HOME="$ROOT" -x "LD_LIBRARY_PATH=$LIB:$UMD"
   -x TT_TOPO_SAT_NO_MINHOST=1 -x "TT_TOPO_SAT_DUMP_DIMACS=$CNF" -x OMPI_MCA_mpi_yield_when_idle=1)
for i in "${!PATHS[@]}"; do [ "$i" -gt 0 ] && CMD+=(":"); CMD+=(-np 1 -x "TT_METAL_MOCK_CLUSTER_DESC_PATH=${PATHS[$i]}" "$GRB" -m "$MGD" -o "$OUT"); done
timeout 120 "${CMD[@]}" >/dev/null 2>&1 || true
head -1 "$CNF"
