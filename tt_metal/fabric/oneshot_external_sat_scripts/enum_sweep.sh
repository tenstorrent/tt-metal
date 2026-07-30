#!/usr/bin/env bash
# ENUMERATION (search_n) experiments. Enumerate N=5 solutions; metric = WALL time to finish (the [intermesh] timer
# only covers the first solve). Key comparison: does fastsat speed the incremental .next steps? Also hardcap vs warm
# prime, and gimsatul-prime. Host cap on throughout. Records solutions found (in case of timeout).
set -u; cd /data/rsong/tt-metal2
SC=/tmp/claude-4010/-data-rsong-tt-metal2/00fa4a3b-aa18-4f70-b82a-ae18fdf91071/scratchpad
RES="$SC/enum_results.txt"; : > "$RES"
GRB=$(readlink -f build/tools/scaleout/generate_rank_bindings)
LIB=$(readlink -f build/tt_metal); UMD=$(readlink -f build/tt_metal/third_party/umd/lib)
GIMBIN="$SC/gimsatul/gimsatul"; mapfile -t PATHS < "$SC/paths_SC36.txt"; CAP=800; NSOL=5
declare -A CFG=(
  [enum_hardcap]="-x TT_TOPO_SAT_MIN_MODE=3"
  [enum_hardcap_fastsat]="-x TT_TOPO_SAT_MIN_MODE=3 -x TT_TOPO_SAT_FASTSAT=1"
  [enum_gim_prime_t32]="-x TT_TOPO_SAT_MIN_MODE=3 -x TT_TOPO_SAT_GIMSATUL=1 -x TT_TOPO_SAT_GIMSATUL_BIN=$GIMBIN -x TT_TOPO_SAT_GIMSATUL_THREADS=32"
  [enum_warmdescent]="-x TT_TOPO_SAT_MIN_MODE=0"
)
ORDER=(enum_hardcap enum_hardcap_fastsat enum_gim_prime_t32 enum_warmdescent)
printf "%-7s %-24s %-8s %-6s %-6s %s\n" stages config wall_s rc sols_found note | tee -a "$RES"
for stages in 64 96; do
 MGD=$(readlink -f "generated/mgd/pipeline_sweep/sweep_2x4_pipeline_${stages}stage_mesh_graph_descriptor.textproto")
 for cfg in "${ORDER[@]}"; do
  OUT="$SC/EN_${stages}_${cfg}"; rm -rf "$OUT"; mkdir -p "$OUT"
  CMD=(/usr/bin/mpirun --allow-run-as-root --oversubscribe --tag-output -x TT_LOGGER_LEVEL=info -x TT_METAL_HOME="$PWD" -x "LD_LIBRARY_PATH=$LIB:$UMD" -x OMPI_MCA_mpi_yield_when_idle=1 ${CFG[$cfg]})
  for i in "${!PATHS[@]}"; do [ "$i" -gt 0 ] && CMD+=(":"); CMD+=(-np 1 -x "TT_METAL_MOCK_CLUSTER_DESC_PATH=${PATHS[$i]}" "$GRB" -m "$MGD" -o "$OUT" -n "$NSOL"); done
  st=$(date +%s); timeout $CAP "${CMD[@]}" > "$OUT/log" 2>&1; rc=$?; et=$(date +%s)
  sols=$(ls -d "$OUT"/*/ 2>/dev/null | grep -c solution 2>/dev/null); [ "${sols:-0}" = "0" ] && sols=$(grep -aoE "found [0-9]+ solution" "$OUT/log" | grep -oE "[0-9]+" | sort -rn | head -1)
  note=$([ "$rc" = "124" ] && echo "TIMEOUT@${CAP}s" || echo "done")
  printf "%-7s %-24s %-8s %-6s %-6s %s\n" "$stages" "$cfg" "$((et-st))" "$rc" "${sols:-0}" "$note" | tee -a "$RES"
  find "$OUT" -type f ! -name log -delete 2>/dev/null; find "$OUT" -mindepth 1 -type d -empty -delete 2>/dev/null
 done
done
echo "[$(date -u +%T)] ENUM SWEEP DONE" | tee -a "$RES"
