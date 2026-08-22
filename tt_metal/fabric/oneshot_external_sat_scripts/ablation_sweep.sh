#!/usr/bin/env bash
# In-solver optimization ablation, ONE solve each, HOST CAP applied throughout (via the minimize/hardcap path).
# Each config toggles one optimization on the real solve path (preferred N/A here: pipeline MGDs have no pinnings;
# preferred is measured separately later). Records the [intermesh-solve] time per (shape, stages, config).
set -u
cd /data/rsong/tt-metal2
SC=/tmp/claude-4010/-data-rsong-tt-metal2/00fa4a3b-aa18-4f70-b82a-ae18fdf91071/scratchpad
RES="$SC/ablation_results.txt"; : > "$RES"
GRB=$(readlink -f build/tools/scaleout/generate_rank_bindings)
LIB=$(readlink -f build/tt_metal); UMD=$(readlink -f build/tt_metal/third_party/umd/lib)
mapfile -t PATHS < "$SC/paths_SC36.txt"
CAP=1500  # per-solve timeout (s)

# config name -> env flags (all keep the host cap on; they differ only in HOW the capped solve is driven)
declare -A CFG=(
  [mode3_coldcap]="-x TT_TOPO_SAT_MIN_MODE=3"
  [mode3_fastsat]="-x TT_TOPO_SAT_MIN_MODE=3 -x TT_TOPO_SAT_FASTSAT=1"
  [mode3_seed7]="-x TT_TOPO_SAT_MIN_MODE=3 -x TT_TOPO_SAT_SEED=7"
  [mode1_warmlock]="-x TT_TOPO_SAT_MIN_MODE=1"
  [mode0_warmdescent]="-x TT_TOPO_SAT_MIN_MODE=0"
)
ORDER=(mode3_coldcap mode3_fastsat mode3_seed7 mode1_warmlock mode0_warmdescent)

printf "%-6s %-7s %-20s %-12s %-6s %s\n" shape stages config intermesh_s rc occupied | tee -a "$RES"
run() { # shape stages config
  local shape=$1 stages=$2 cfg=$3
  local MGD=$(readlink -f "generated/mgd/pipeline_sweep/sweep_${shape}_pipeline_${stages}stage_mesh_graph_descriptor.textproto")
  [ -f "$MGD" ] || { printf "%-6s %-7s %-20s %-12s %-6s %s\n" "$shape" "$stages" "$cfg" "NO_MGD" "-" "-" | tee -a "$RES"; return; }
  local OUT="$SC/AB_${shape}_${stages}_${cfg}"; rm -rf "$OUT"; mkdir -p "$OUT"
  local CMD=(/usr/bin/mpirun --allow-run-as-root --oversubscribe --tag-output
     -x TT_LOGGER_LEVEL=info -x TT_METAL_HOME="$PWD" -x "LD_LIBRARY_PATH=$LIB:$UMD" -x OMPI_MCA_mpi_yield_when_idle=1 ${CFG[$cfg]})
  local i; for i in "${!PATHS[@]}"; do [ "$i" -gt 0 ] && CMD+=(":"); CMD+=(-np 1 -x "TT_METAL_MOCK_CLUSTER_DESC_PATH=${PATHS[$i]}" "$GRB" -m "$MGD" -o "$OUT"); done
  timeout "$CAP" "${CMD[@]}" > "$OUT/log" 2>&1; local rc=$?
  local im=$(grep -aoE "intermesh-solve\] attempt 1 : [0-9.]+ ms" "$OUT/log" | head -1 | grep -oE "[0-9.]+" | head -1)
  local occ=$(grep -aoE "hard-capped host-group usage at [0-9]+|occupied=[0-9]+" "$OUT/log" | grep -oE "[0-9]+" | tail -1)
  local secs="TIMEOUT"; [ -n "$im" ] && secs=$(awk "BEGIN{printf \"%.1f\", $im/1000}")
  printf "%-6s %-7s %-20s %-12s %-6s %s\n" "$shape" "$stages" "$cfg" "$secs" "$rc" "${occ:-?}" | tee -a "$RES"
}

for stages in 64 96 112 128; do for cfg in "${ORDER[@]}"; do run 2x4 "$stages" "$cfg"; done; done
for stages in 32 48 64;      do for cfg in "${ORDER[@]}"; do run 4x4 "$stages" "$cfg"; done; done
echo "[$(date -u +%T)] ABLATION SWEEP DONE" | tee -a "$RES"
