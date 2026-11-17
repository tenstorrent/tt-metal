#!/bin/bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
#
# The multimodal suite against one endpoint, every output kept: per bench a smoke (--limit SMOKE) then the full set,
# each in its own directory with run_eval.sh's journal, raw request / response log, lmms-eval samples and summaries.
#
#   run_suite.sh <dir> <base_url> <model> [--target vllm|openrouter] [--provider slug] [--concurrency C]
#                [--smoke N] [--benches "b1 b2 ..."] [--phase smoke|full|both] [--max-retries N] [--conformance]
#
#   <dir>/<bench>/smoke<N>/   <dir>/<bench>/full/   <dir>/conformance/ (with --conformance: test_api_conformance.py,
#   every exchange in exchanges.jsonl)   <dir>/meta.json (code SHAs, harness versions, env)   <dir>/suite.log
#   <dir>/suite.jsonl (one record per finished step: rc, wall, score, requests / failures, cost)
#
# Resume: rerun the same command; finished requests come from each step's journal. A full set with failed requests is
# rerun up to RESUME_PASSES (2) more times, which resends only the failed ones. A step whose smoke has failed
# requests stops the suite (rc 3) on target vllm (a broken server should not burn the full sets); on openrouter the
# failures are recorded and the suite goes on (rerunning retries them). A full step whose run_eval.sh still exits
# non-zero on its last pass has no summary.json (an earlier pass's is kept as summary.json.prev) and the suite ends
# with rc 4 after the other benches. Default benches, in order:
#   ocrbench mmau_mini librispeech_clean videomme_short mmmu_pro_aa
# Env: as run_eval.sh; SUITE_PYTHON (python with httpx + pytest for --conformance, default the eval venv's).

set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
[ $# -ge 3 ] || { sed -n '4,20p' "$0" | sed 's/^# \{0,1\}//'; exit 2; }
DIR=$(realpath -m "$1") BASE_URL=${2%/} MODEL=$3
shift 3
TARGET="" PROVIDER="" CONC=32 SMOKE=5 PHASE=both RETRIES="" CONFORMANCE=0
BENCHES="ocrbench mmau_mini librispeech_clean videomme_short mmmu_pro_aa"
while [ $# -gt 0 ]; do
    case $1 in
        --target) TARGET=$2; shift 2 ;;
        --provider) PROVIDER=$2; shift 2 ;;
        --concurrency) CONC=$2; shift 2 ;;
        --smoke) SMOKE=$2; shift 2 ;;
        --benches) BENCHES=$2; shift 2 ;;
        --phase) PHASE=$2; shift 2 ;;
        --max-retries) RETRIES=$2; shift 2 ;;
        --conformance) CONFORMANCE=1; shift ;;
        *) echo "unknown option $1" >&2; exit 2 ;;
    esac
done
[ -n "$TARGET" ] || case $BASE_URL in *openrouter.ai*) TARGET=openrouter ;; *) TARGET=vllm ;; esac
EVALENV=${MM_EVALENV:-/data/ttuser-agentx/mm/evalenv}
PY=${SUITE_PYTHON:-$EVALENV/bin/python}
mkdir -p "$DIR"
LOG=$DIR/suite.log
log() { echo "$(date -u +%FT%TZ) $*" | tee -a "$LOG"; }

# ------------------------------------------------------------------------------------------------ meta
{
    echo "{"
    echo " \"started\": \"$(date -u +%FT%TZ)\", \"host\": \"$(hostname)\", \"dir\": \"$DIR\","
    echo " \"base_url\": \"$BASE_URL\", \"model\": \"$MODEL\", \"target\": \"$TARGET\", \"provider\": \"$PROVIDER\","
    echo " \"concurrency\": $CONC, \"smoke\": $SMOKE, \"phase\": \"$PHASE\", \"benches\": \"$BENCHES\","
    echo " \"eval_code\": {\"path\": \"$HERE\", \"sha\": \"$(git -C "$HERE" rev-parse HEAD 2>/dev/null)\","
    echo "   \"dirty_files\": $(git -C "$HERE" status --porcelain -- . 2>/dev/null | wc -l)},"
    echo " \"harness\": \"$("$EVALENV/bin/python" -m pip list 2>/dev/null | grep -iE '^(lmms[-_]eval|openai|httpx|datasets) ' | tr -s ' ' | tr '\n' ';')\","
    echo " \"argv\": \"$0 $*\""
    echo "}"
} > "$DIR/meta.$(date -u +%Y%m%dT%H%M%SZ).json"

record() {  # $1 bench, $2 step, $3 out dir, $4 rc, $5 wall s
    "$EVALENV/bin/python" - "$@" >> "$DIR/suite.jsonl" <<'EOF'
import json, os, sys
bench, step, out, rc, wall = sys.argv[1:6]
rec = {"bench": bench, "step": step, "out": out, "rc": int(rc), "wall_s": int(wall)}
try:
    s = json.load(open(os.path.join(out, "summary.json")))
    rec["metrics"] = s.get("metrics")
    rec["n_samples"] = s.get("n_samples")
    req = s.get("requests") or {}
    for k in ("requests", "from_journal", "ok", "failed", "status", "cost_usd", "effective_concurrency",
              "max_in_flight", "attempt_status", "empty_content", "finish_length", "latency_p50_s", "latency_p95_s"):
        rec[k] = req.get(k)
except (OSError, ValueError):
    rec["summary"] = None
print(json.dumps(rec))
EOF
}

run_step() {  # $1 bench, $2 step name, $3 limit or "", returns run_eval.sh's rc
    local bench=$1 step=$2 limit=$3 out=$DIR/$1/$2 t0 rc args
    mkdir -p "$out"
    args=("$bench" "$BASE_URL" "$MODEL" --target "$TARGET" --concurrency "$CONC" --out "$out")
    [ -n "$PROVIDER" ] && args+=(--provider "$PROVIDER")
    [ -n "$RETRIES" ] && args+=(--max-retries "$RETRIES")
    [ -n "$limit" ] && args+=(--limit "$limit")
    # What an earlier pass or run left: a pass that dies before its results step must not be read through them (a
    # stale journal.stats.json with failed=0 ended the resume loop, a stale summary.json was reported as the score).
    for f in summary.json journal.stats.json; do
        [ -f "$out/$f" ] && mv -f "$out/$f" "$out/$f.prev"
    done
    log "start $bench $step -> $out"
    t0=$(date +%s)
    bash "$HERE/run_eval.sh" "${args[@]}" >> "$out/stdout.log" 2>&1
    rc=$?
    record "$bench" "$step" "$out" "$rc" "$(( $(date +%s) - t0 ))"
    log "end $bench $step rc=$rc after $(( $(date +%s) - t0 )) s: $(tail -n 1 "$DIR/suite.jsonl" | cut -c1-400)"
    return $rc
}

failed_requests() {  # $1 out dir -> failed request count of the last run (from the journal stats)
    "$EVALENV/bin/python" -c '
import json, sys
try:
    print(json.load(open(sys.argv[1] + "/journal.stats.json")).get("failed", 0))
except (OSError, ValueError):
    print(-1)' "$1"
}

if [ "$CONFORMANCE" = 1 ]; then
    out=$DIR/conformance
    mkdir -p "$out"
    log "start conformance -> $out"
    t0=$(date +%s)
    REPO=$(cd "$HERE/../../../../.." && pwd)
    (cd "$REPO" && BASE_URL=$BASE_URL MODEL=$MODEL MM_TARGET=$TARGET MM_CONFORMANCE_LOG=$out/exchanges.jsonl \
        MM_ASSETS_DIR=${MM_ASSETS_DIR:-/data/ttuser-agentx/mm/evals/assets} \
        "$PY" -m pytest -q -p no:cacheprovider --confcutdir="$HERE" "$HERE/test_api_conformance.py" \
        --junitxml="$out/junit.xml" -rA > "$out/pytest.log" 2>&1)
    rc=$?
    echo "{\"bench\": \"conformance\", \"step\": \"pytest\", \"out\": \"$out\", \"rc\": $rc, \"wall_s\": $(( $(date +%s) - t0 )), \"result\": \"$(tail -n 1 "$out/pytest.log" | tr -d '"')\"}" >> "$DIR/suite.jsonl"
    log "end conformance rc=$rc: $(tail -n 1 "$out/pytest.log")"
fi

FAILED_STEPS=""
for bench in $BENCHES; do
    if [ "$PHASE" != full ]; then
        run_step "$bench" "smoke$SMOKE" "$SMOKE"
        rc=$?
        nfail=$(failed_requests "$DIR/$bench/smoke$SMOKE")
        if [ "$rc" != 0 ] || [ "$nfail" != 0 ]; then
            log "smoke of $bench: rc=$rc, failed requests=$nfail"
            if [ "$TARGET" = vllm ]; then log "stopping the suite (smoke failed on our server)"; exit 3; fi
        fi
    fi
    [ "$PHASE" = smoke ] && continue
    # Failed requests are not journaled: up to RESUME_PASSES reruns resend only those (the rest come from the journal).
    # A pass that exits non-zero (lmms-eval died, e.g. on a missing video) has no results of its own: it is run again
    # (it resumes from the journal), and if the last pass still fails the step is reported failed, not done.
    for pass in $(seq 0 "${RESUME_PASSES:-2}"); do
        run_step "$bench" full ""
        rc=$?
        if [ "$rc" != 0 ]; then
            log "$bench full: run_eval.sh exited $rc on pass $pass; no summary of this pass"
            continue
        fi
        nfail=$(failed_requests "$DIR/$bench/full")
        [ "$nfail" = 0 ] && break
        log "$bench full: $nfail failed requests after pass $pass"
    done
    if [ "$rc" != 0 ]; then
        log "$bench full: FAILED (run_eval.sh exited $rc on its last pass); it has no summary.json"
        FAILED_STEPS="$FAILED_STEPS $bench"
    fi
done
if [ -n "$FAILED_STEPS" ]; then
    log "suite done, with failed steps:$FAILED_STEPS"
    exit 4
fi
log "suite done"
