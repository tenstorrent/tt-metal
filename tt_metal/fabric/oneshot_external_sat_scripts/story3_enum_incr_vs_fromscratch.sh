#!/usr/bin/env bash
# Story 3 (SMALL): incremental vs from-scratch enumeration, CaDiCaL-only hardcap, -n 5, ONE seed (7).
# Stages 64/80/96, one seed each first, incr then fromscratch. Confirms path ran + wall time + #sols.
set -u; cd /data/rsong/tt-metal2
SC=/tmp/claude-4010/-data-rsong-tt-metal2/00fa4a3b-aa18-4f70-b82a-ae18fdf91071/scratchpad
RES="$SC/story3_small_results.txt"; : > "$RES"
GRB=$(readlink -f build/tools/scaleout/generate_rank_bindings)
LIB=$(readlink -f build/tt_metal); UMD=$(readlink -f build/tt_metal/third_party/umd/lib)
mapfile -t PATHS < "$SC/paths_SC36.txt"; CAP=900; NSOL=5; SEED=7
printf "%-7s %-12s %-8s %-6s %s\n" stages mode wall_s sols pathOK | tee -a "$RES"
run() { local stages=$1 mode=$2
  local MGD=$(readlink -f "generated/mgd/pipeline_sweep/sweep_2x4_pipeline_${stages}stage_mesh_graph_descriptor.textproto")
  local OUT="$SC/S3S_${stages}_${mode}"; rm -rf "$OUT"; mkdir -p "$OUT"
  local EXTRA=(-x TT_TOPO_SAT_MIN_MODE=3 -x "TT_TOPO_SAT_SEED=$SEED")
  [ "$mode" = "fromscratch" ] && EXTRA+=(-x TT_TOPO_SAT_ENUM_FROMSCRATCH=1)
  local CMD=(/usr/bin/mpirun --allow-run-as-root --oversubscribe --tag-output -x TT_LOGGER_LEVEL=info -x TT_METAL_HOME="$PWD" -x "LD_LIBRARY_PATH=$LIB:$UMD" -x OMPI_MCA_mpi_yield_when_idle=1 "${EXTRA[@]}")
  local i; for i in "${!PATHS[@]}"; do [ "$i" -gt 0 ] && CMD+=(":"); CMD+=(-np 1 -x "TT_METAL_MOCK_CLUSTER_DESC_PATH=${PATHS[$i]}" "$GRB" -m "$MGD" -o "$OUT" -n "$NSOL"); done
  local st=$(date +%s); timeout $CAP "${CMD[@]}" > "$OUT/log" 2>&1; local rc=$?; local et=$(date +%s)
  local sols=$(grep -aoE "Enumerated [0-9]+ solution" "$OUT/log" | grep -oE "[0-9]+" | tail -1); [ -z "$sols" ] && sols=$(grep -acE "Wrote solution" "$OUT/log")
  local pathOK="?"
  if [ "$mode" = "fromscratch" ]; then grep -qa "ENUM path = FROM-SCRATCH" "$OUT/log" && pathOK="FS-OK" || pathOK="FS-MISSING"; else grep -qa "ENUM path = INCREMENTAL" "$OUT/log" && pathOK="INCR-OK" || pathOK="INCR-MISSING"; fi
  printf "%-7s %-12s %-8s %-6s %s\n" "$stages" "$mode" "$((et-st))" "${sols:-0}" "$pathOK" | tee -a "$RES"
  find "$OUT" -type f ! -name log -delete 2>/dev/null; find "$OUT" -mindepth 1 -type d -empty -delete 2>/dev/null
}
for stages in 64 80 96; do
  run "$stages" incr
  run "$stages" fromscratch
done
echo "[$(date -u +%T)] STORY3 SMALL DONE" | tee -a "$RES"
