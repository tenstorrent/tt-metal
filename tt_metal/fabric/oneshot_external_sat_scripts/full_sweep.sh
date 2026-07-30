#!/usr/bin/env bash
# COMPREHENSIVE sweep: full CaDiCaL ablation + gimsatul-hybrid variants (a couple thread counts), host cap ON,
# ONE solve each. Writes results incrementally so partial data is always available. Ordered fast->slow per stage.
set -u
cd /data/rsong/tt-metal2
SC=/tmp/claude-4010/-data-rsong-tt-metal2/00fa4a3b-aa18-4f70-b82a-ae18fdf91071/scratchpad
RES="$SC/full_results.txt"; : > "$RES"
GRB=$(readlink -f build/tools/scaleout/generate_rank_bindings)
LIB=$(readlink -f build/tt_metal); UMD=$(readlink -f build/tt_metal/third_party/umd/lib)
GIMBIN="$SC/gimsatul/gimsatul"
mapfile -t PATHS < "$SC/paths_SC36.txt"
CAP=1500

declare -A CFG=(
  [cad_mode3]="-x TT_TOPO_SAT_MIN_MODE=3"
  [cad_mode3_seed7]="-x TT_TOPO_SAT_MIN_MODE=3 -x TT_TOPO_SAT_SEED=7"
  [cad_mode3_fastsat]="-x TT_TOPO_SAT_MIN_MODE=3 -x TT_TOPO_SAT_FASTSAT=1"
  [cad_mode1]="-x TT_TOPO_SAT_MIN_MODE=1"
  [cad_mode0]="-x TT_TOPO_SAT_MIN_MODE=0"
  [gim_mode3_t8]="-x TT_TOPO_SAT_MIN_MODE=3 -x TT_TOPO_SAT_GIMSATUL=1 -x TT_TOPO_SAT_GIMSATUL_BIN=$GIMBIN -x TT_TOPO_SAT_GIMSATUL_THREADS=8"
  [gim_mode3_t16]="-x TT_TOPO_SAT_MIN_MODE=3 -x TT_TOPO_SAT_GIMSATUL=1 -x TT_TOPO_SAT_GIMSATUL_BIN=$GIMBIN -x TT_TOPO_SAT_GIMSATUL_THREADS=16"
  [gim_mode3_t32]="-x TT_TOPO_SAT_MIN_MODE=3 -x TT_TOPO_SAT_GIMSATUL=1 -x TT_TOPO_SAT_GIMSATUL_BIN=$GIMBIN -x TT_TOPO_SAT_GIMSATUL_THREADS=32"
  [gim_first_t32]="-x TT_TOPO_SAT_MIN_MODE=0 -x TT_TOPO_SAT_GIMSATUL=1 -x TT_TOPO_SAT_GIM_FIRST=1 -x TT_TOPO_SAT_GIMSATUL_BIN=$GIMBIN -x TT_TOPO_SAT_GIMSATUL_THREADS=32"
)
# fast/informative first so partial data is useful early
ORDER=(cad_mode3 gim_mode3_t32 gim_mode3_t16 gim_mode3_t8 cad_mode3_seed7 cad_mode3_fastsat gim_first_t32 cad_mode1 cad_mode0)

printf "%-6s %-7s %-20s %-12s %-6s %s\n" shape stages config intermesh_s rc occupied | tee -a "$RES"
run() { local shape=$1 stages=$2 cfg=$3
  local MGD=$(readlink -f "generated/mgd/pipeline_sweep/sweep_${shape}_pipeline_${stages}stage_mesh_graph_descriptor.textproto")
  [ -f "$MGD" ] || return
  local OUT="$SC/FS_${shape}_${stages}_${cfg}"; rm -rf "$OUT"; mkdir -p "$OUT"
  local CMD=(/usr/bin/mpirun --allow-run-as-root --oversubscribe --tag-output
     -x TT_LOGGER_LEVEL=info -x TT_METAL_HOME="$PWD" -x "LD_LIBRARY_PATH=$LIB:$UMD" -x OMPI_MCA_mpi_yield_when_idle=1 ${CFG[$cfg]})
  local i; for i in "${!PATHS[@]}"; do [ "$i" -gt 0 ] && CMD+=(":"); CMD+=(-np 1 -x "TT_METAL_MOCK_CLUSTER_DESC_PATH=${PATHS[$i]}" "$GRB" -m "$MGD" -o "$OUT"); done
  timeout "$CAP" "${CMD[@]}" > "$OUT/log" 2>&1; local rc=$?
  local im=$(grep -aoE "intermesh-solve\] attempt 1 : [0-9.]+ ms" "$OUT/log" | head -1 | grep -oE "[0-9.]+ ms" | grep -oE "[0-9.]+")
  local occ=$(grep -aoE "hard-capped host-group usage at [0-9]+" "$OUT/log" | grep -oE "[0-9]+" | tail -1)
  local secs="TIMEOUT/FAIL"; [ -n "$im" ] && secs=$(awk "BEGIN{printf \"%.1f\", $im/1000}")
  printf "%-6s %-7s %-20s %-12s %-6s %s\n" "$shape" "$stages" "$cfg" "$secs" "$rc" "${occ:-?}" | tee -a "$RES"
  # clean the big per-cell output dir to save disk (keep only the log)
  find "$OUT" -type f ! -name log -delete 2>/dev/null
}
for stages in 64 96 112 128; do for cfg in "${ORDER[@]}"; do run 2x4 "$stages" "$cfg"; done; done
for stages in 32 48 64;      do for cfg in "${ORDER[@]}"; do run 4x4 "$stages" "$cfg"; done; done
echo "[$(date -u +%Y-%m-%dT%H:%M:%S)] FULL SWEEP DONE" | tee -a "$RES"
