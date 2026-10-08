#!/usr/bin/env bash
# Round 3 eltwise binary (#58818 review): run the exhaustive bit dumps of a spec file. Each line:
# "<pairs>|||<pytest args>[|||<extra env K=V ...>]", pairs "A|B;A|B" with each side "K=V,K=V" of EB_R3_* toggles (side A the
# reference). usage: dump_run.sh <spec file>
[[ -n "${HWLOCK_HELD:-}" || -n "${GITHUB_ACTIONS:-}" ]] || { echo "not under hwlock" >&2; exit 2; }
cd /work
SPEC=$1
export PYTHONPATH=/work:/work/tests/eb_r3_ci:${PYTHONPATH:-}
export EB_R3_LOG_RULE=1 TT_METAL_CACHE=${TT_METAL_CACHE:-/tmp/ebdump_cache}
mkdir -p "$TT_METAL_CACHE"
F='(/DUMP|EB_R3_RULE|passed|failed|skipped|rror|Traceback|^E |Timeout|timed out/) && !seen[$0]++ {print substr($0, 1, 1500); fflush()}'
echo "##### spec $SPEC $(date -u +%T)"; grep -v '^#' "$SPEC"
BASE=$TT_METAL_CACHE; n=0
while IFS= read -r line; do
  [[ -z "$line" || "$line" == \#* ]] && continue
  n=$((n + 1)); export TT_METAL_CACHE=$BASE/line_$n; mkdir -p $TT_METAL_CACHE
  pairs=${line%%|||*}; rest=${line#*|||}; args=${rest%%|||*}; extra=""; [[ "$rest" == *"|||"* ]] && extra=${rest#*|||}
  echo "##### $(date -u +%T) pairs=[$pairs] args=[$args] env=[$extra]"
  export EB_DUMP_PAIRS="$pairs"
  eval "$extra timeout -s INT -k 60 ${EB_RUN_LIMIT:-6600} python3 -u -m pytest -p no:cacheprovider --timeout=0 -q -s -rfE $args" < /dev/null 2>&1 | sed -u -E 's/^tests\/eb_r3_ci\/[^ ]* //' | awk "$F"
  # the two sides of each comparison ran different code: variants grouped by defines without the toggle lines
  python3 /work/tests/eb_r3_ci/elf_ab.py "$TT_METAL_CACHE" ${EB_ELFAB:-"eltwise_binary_no_bcast=EB_R3_|BINARY_NG_BLOCK" "eltwise_binary_col_bcast=EB_R3_|BCAST_OTHER_CHUNK" "eltwise_binary_scalar_bcast=EB_R3_|BCAST_OTHER_CHUNK" "eltwise_binary_row_bcast=EB_R3_" "eltwise_binary_row_col_bcast=EB_R3_" "eltwise_binary_scalar=EB_R3_" "eltwise_binary=EB_R3_" "eb_dump_reuse=EB_DUMP_PER_TILE"} < /dev/null 2>&1 | grep -v " variants 0,"
done < "$SPEC"
echo "##### end $(date -u +%T)"
