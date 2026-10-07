#!/usr/bin/env bash
# Round 3 eltwise binary (#58818 review): run the exhaustive bit dumps of a spec file. Each line:
# "<pairs>|||<pytest args>[|||<extra env K=V ...>]", pairs "A|B;A|B" with each side "K=V,K=V" of EB_R3_* toggles (side A the
# reference). usage: dump_run.sh <spec file>
cd /work
SPEC=$1
export PYTHONPATH=/work:/work/tests/eb_r3_ci:${PYTHONPATH:-}
export EB_R3_LOG_RULE=1 TT_METAL_CACHE=${TT_METAL_CACHE:-/tmp/ebdump_cache}
mkdir -p "$TT_METAL_CACHE"
F='(/DUMP|EB_R3_RULE|passed|failed|skipped|rror|Traceback|^E |Timeout|timed out/) && !seen[$0]++ {print substr($0, 1, 1500); fflush()}'
echo "##### spec $SPEC $(date -u +%T)"; grep -v '^#' "$SPEC"
while IFS= read -r line; do
  [[ -z "$line" || "$line" == \#* ]] && continue
  pairs=${line%%|||*}; rest=${line#*|||}; args=${rest%%|||*}; extra=""; [[ "$rest" == *"|||"* ]] && extra=${rest#*|||}
  echo "##### $(date -u +%T) pairs=[$pairs] args=[$args] env=[$extra]"
  export EB_DUMP_PAIRS="$pairs"
  eval "$extra timeout -s INT -k 60 ${EB_RUN_LIMIT:-6600} python3 -u -m pytest -p no:cacheprovider -q -s -rfE $args" < /dev/null 2>&1 | sed -u -E 's/^tests\/eb_r3_ci\/[^ ]* //' | awk "$F"
done < "$SPEC"
echo "##### end $(date -u +%T)"
