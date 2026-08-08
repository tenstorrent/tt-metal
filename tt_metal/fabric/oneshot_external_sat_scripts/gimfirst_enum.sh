#!/usr/bin/env bash
# gim-first-solve + incremental-CaDiCaL enumeration on the HARD 2x4 cases where the production baseline gets
# 0 solutions (96/112/128/144 all fail to find even 1 in the 15-min budget). MIN_MODE=3 + GIMSATUL=1 => gimsatul
# primes the hardcap first solve, then CaDiCaL enumerates .next. Target -n 20. Logs time-to-first + #solutions.
set -u; cd /data/rsong/tt-metal2
SC=/tmp/claude-4010/-data-rsong-tt-metal2/00fa4a3b-aa18-4f70-b82a-ae18fdf91071/scratchpad
RES="$SC/gimfirst_enum_results.txt"; : > "$RES"
GRB=$(readlink -f build/tools/scaleout/generate_rank_bindings)
LIB=$(readlink -f build/tt_metal); UMD=$(readlink -f build/tt_metal/third_party/umd/lib)
GIMBIN="$SC/gimsatul/gimsatul"; mapfile -t PATHS < "$SC/paths_SC36.txt"
CAP=2400; NSOL=20  # 40 min/cell, aim for 20 solutions
printf "%-7s %-8s %-10s %-8s %-6s %s\n" stages 1st_sol total_s sols rc note | tee -a "$RES"
for stages in 128 96 112 144; do
 MGD=$(readlink -f "generated/mgd/pipeline_sweep/sweep_2x4_pipeline_${stages}stage_mesh_graph_descriptor.textproto")
 OUT="$SC/GFE_${stages}"; rm -rf "$OUT"; mkdir -p "$OUT"
 CMD=(/usr/bin/mpirun --allow-run-as-root --oversubscribe --tag-output -x TT_LOGGER_LEVEL=info -x TT_METAL_HOME="$PWD" -x "LD_LIBRARY_PATH=$LIB:$UMD" -x OMPI_MCA_mpi_yield_when_idle=1 \
    -x TT_TOPO_SAT_MIN_MODE=3 -x TT_TOPO_SAT_GIMSATUL=1 -x "TT_TOPO_SAT_GIMSATUL_BIN=$GIMBIN" -x TT_TOPO_SAT_GIMSATUL_THREADS=32)
 for i in "${!PATHS[@]}"; do [ "$i" -gt 0 ] && CMD+=(":"); CMD+=(-np 1 -x "TT_METAL_MOCK_CLUSTER_DESC_PATH=${PATHS[$i]}" "$GRB" -m "$MGD" -o "$OUT" -n "$NSOL"); done
 st=$(date +%s); timeout $CAP "${CMD[@]}" > "$OUT/log" 2>&1; rc=$?; et=$(date +%s)
 # time to first solution (epoch of "1 written" - enum start), #solutions written, total wall
 t_enum_start=$(grep -aE "Enumerating topology mapping solutions \(max" "$OUT/log" | head -1 | grep -oE "[0-9]{2}:[0-9]{2}:[0-9]{2}\.[0-9]+" | head -1)
 t_first=$(grep -aE "1 written|\[1 written so far\]|Wrote solution.*\[1 " "$OUT/log" | head -1 | grep -oE "[0-9]{2}:[0-9]{2}:[0-9]{2}\.[0-9]+" | head -1)
 first="?"; [ -n "$t_enum_start" ] && [ -n "$t_first" ] && first=$(awk "BEGIN{print int($(date -d "$t_first" +%s.%3N 2>/dev/null)-$(date -d "$t_enum_start" +%s.%3N 2>/dev/null))\"s\"}")
 sols=$(grep -aoE "Enumerated [0-9]+ solution" "$OUT/log" | grep -oE "[0-9]+" | tail -1)
 [ -z "$sols" ] && sols=$(ls -d "$OUT"/*/ 2>/dev/null | grep -c . 2>/dev/null)
 note=$([ "$rc" = "124" ] && echo "TIMEOUT@${CAP}s" || echo "done")
 printf "%-7s %-8s %-10s %-8s %-6s %s\n" "$stages" "${first:-?}" "$((et-st))" "${sols:-0}" "$rc" "$note" | tee -a "$RES"
 find "$OUT" -type f ! -name log -delete 2>/dev/null; find "$OUT" -mindepth 1 -type d -empty -delete 2>/dev/null
done
echo "[$(date -u +%T)] GIMFIRST ENUM DONE" | tee -a "$RES"
