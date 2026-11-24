#!/usr/bin/env bash
# One merge-gate compile session, timed, from a clean build tree.
# Usage: session.sh <label> <cache dir|""> <mode>   (env: ARCH GROUP NGROUPS OUT LLK_PERF_COMPILE_WORKERS)
set -euo pipefail
LABEL=$1 CACHE=$2 MODE=$3
TRIAL=$(cd "$(dirname "$0")" && pwd)
BUILD="$RUNNER_TEMP/tt-llk-build"
DIR="$OUT/$LABEL"
mkdir -p "$DIR/stats"
rm -rf "$BUILD"
cd "$GITHUB_WORKSPACE/tt_metal/tt-llk"
start=$(date +%s.%N)
set +e
env LLK_ELF_CACHE="$CACHE" LLK_ELF_CACHE_MODE="$MODE" LLK_ELF_CACHE_STATS="$DIR/stats" LLK_PERF_PHASE=compile \
  timeout -k 30 3000 ./tests/run_llk_perf_"$ARCH".sh "$GROUP" "$NGROUPS" > "$DIR/pytest.log" 2>&1
rc=$?
set -e
end=$(date +%s.%N)
tail -3 "$DIR/pytest.log"
( cd "$BUILD" && find . -name '*.elf' -print0 | sort -z | xargs -0 -r -P8 -n 500 sha256sum ) | sort -k2 > "$DIR/elfs.sha256"
python3 "$TRIAL/summarize.py" "$DIR/stats" > "$DIR/stats.json"
python3 - "$DIR" "$LABEL" "$start" "$end" "$rc" "$(git rev-parse HEAD)" "$MODE" "${CACHE:+on}" <<'EOF'
import json, sys
d, label, start, end, rc, sha, mode, on = sys.argv[1:]
stats = json.load(open(f"{d}/stats.json"))
elfs = sum(1 for _ in open(f"{d}/elfs.sha256"))
out = {"label": label, "commit": sha, "cache": bool(on), "mode": mode if on else None,
       "wall_s": round(float(end) - float(start), 1), "rc": int(rc), "elfs": elfs, **stats}
json.dump(out, open(f"{d}/session.json", "w"))
print(json.dumps(out))
EOF
if [[ "${KEEP_TREE:-0}" == 1 ]]; then rm -rf "$RUNNER_TEMP/tree_$LABEL"; mv "$BUILD" "$RUNNER_TEMP/tree_$LABEL"; fi
# Keep the per-ELF stats small for the artifact.
zstd -q --rm -T0 "$DIR"/stats/tu.*.jsonl 2>/dev/null || true
