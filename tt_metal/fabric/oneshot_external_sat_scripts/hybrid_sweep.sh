#!/usr/bin/env bash
# HYBRID integrated-speed test: our CaDiCaL drives the full solve (host cap, occupancy, descent, decode) but each
# heavy SAT solve is delegated to gimsatul (TT_TOPO_SAT_GIMSATUL=1). Measures integrated inter-mesh solve time with
# ALL infrastructure intact, vs pure-CaDiCaL cold cap. One solve each, host cap ON throughout.
set -u
cd /data/rsong/tt-metal2
SC=/tmp/claude-4010/-data-rsong-tt-metal2/00fa4a3b-aa18-4f70-b82a-ae18fdf91071/scratchpad
RES="$SC/hybrid_results.txt"; : > "$RES"
GRB=$(readlink -f build/tools/scaleout/generate_rank_bindings)
LIB=$(readlink -f build/tt_metal); UMD=$(readlink -f build/tt_metal/third_party/umd/lib)
GIMBIN="$SC/gimsatul/gimsatul"
mapfile -t PATHS < "$SC/paths_SC36.txt"
CAP=1200; TH=32

declare -A CFG=(
  [cad_mode3_coldcap]="-x TT_TOPO_SAT_MIN_MODE=3"
  [gim_mode3_coldcap]="-x TT_TOPO_SAT_MIN_MODE=3 -x TT_TOPO_SAT_GIMSATUL=1 -x TT_TOPO_SAT_GIMSATUL_BIN=$GIMBIN -x TT_TOPO_SAT_GIMSATUL_THREADS=$TH"
  [gim_mode0_warmdescent]="-x TT_TOPO_SAT_MIN_MODE=0 -x TT_TOPO_SAT_GIMSATUL=1 -x TT_TOPO_SAT_GIMSATUL_BIN=$GIMBIN -x TT_TOPO_SAT_GIMSATUL_THREADS=$TH"
)
ORDER=(cad_mode3_coldcap gim_mode3_coldcap gim_mode0_warmdescent)

printf "%-6s %-7s %-24s %-12s %-6s %s\n" shape stages config intermesh_s rc occupied | tee -a "$RES"
run() { local shape=$1 stages=$2 cfg=$3
  local MGD=$(readlink -f "generated/mgd/pipeline_sweep/sweep_${shape}_pipeline_${stages}stage_mesh_graph_descriptor.textproto")
  [ -f "$MGD" ] || { printf "%-6s %-7s %-24s %s\n" "$shape" "$stages" "$cfg" NO_MGD | tee -a "$RES"; return; }
  local OUT="$SC/HYB_${shape}_${stages}_${cfg}"; rm -rf "$OUT"; mkdir -p "$OUT"
  local CMD=(/usr/bin/mpirun --allow-run-as-root --oversubscribe --tag-output
     -x TT_LOGGER_LEVEL=info -x TT_METAL_HOME="$PWD" -x "LD_LIBRARY_PATH=$LIB:$UMD" -x OMPI_MCA_mpi_yield_when_idle=1 ${CFG[$cfg]})
  local i; for i in "${!PATHS[@]}"; do [ "$i" -gt 0 ] && CMD+=(":"); CMD+=(-np 1 -x "TT_METAL_MOCK_CLUSTER_DESC_PATH=${PATHS[$i]}" "$GRB" -m "$MGD" -o "$OUT"); done
  timeout "$CAP" "${CMD[@]}" > "$OUT/log" 2>&1; local rc=$?
  local im=$(grep -aoE "intermesh-solve\] attempt 1 : [0-9.]+ ms" "$OUT/log" | head -1 | grep -oE "[0-9.]+ ms" | grep -oE "[0-9.]+")
  local occ=$(grep -aoE "hard-capped host-group usage at [0-9]+" "$OUT/log" | grep -oE "[0-9]+" | tail -1)
  local secs="TIMEOUT/FAIL"; [ -n "$im" ] && secs=$(awk "BEGIN{printf \"%.1f\", $im/1000}")
  printf "%-6s %-7s %-24s %-12s %-6s %s\n" "$shape" "$stages" "$cfg" "$secs" "$rc" "${occ:-?}" | tee -a "$RES"
}
for stages in 64 96 112 128; do for cfg in "${ORDER[@]}"; do run 2x4 "$stages" "$cfg"; done; done
echo "[$(date -u +%T)] HYBRID SWEEP DONE" | tee -a "$RES"
