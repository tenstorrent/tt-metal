#!/usr/bin/env bash
[[ -n ${HWLOCK_HELD:-} || -n ${GITHUB_ACTIONS:-} ]] || { echo "not under hwlock" >&2; exit 2; }
# Round 3 eltwise binary (#58723 third review): the exhaustive dumps of tests/eb_r3_ci/test_eb_dump_wt.py whose ids match the
# pytest -k expression $1 (per-face against whole-tile program, every bf16 pair), then the kernel variants the cache holds.
cd /work
export TT_METAL_RUNTIME_ROOT=/work PYTHONPATH=/work:/work/tests/eb_r3_ci:${PYTHONPATH:-} TT_METAL_CACHE=/tmp/ebdump8_cache
mkdir -p "$TT_METAL_CACHE"
F='(/DUMP|passed|failed|skipped|rror|Traceback|^E |Timeout|timed out/) && !seen[$0]++ {print substr($0, 1, 1500); fflush()}'
echo "##### dump8 [$1] $(date -u +%T)"
timeout -s INT -k 60 ${EB_RUN_LIMIT:-6300} python3 -u -m pytest -p no:cacheprovider --timeout=0 -q -s -rfE tests/eb_r3_ci/test_eb_dump_wt.py -k "$1" < /dev/null 2>&1 | awk "$F"
echo "##### end $(date -u +%T)"
