#!/usr/bin/env bash
# Copy the small TTI release artifacts back into the autoport's evidence tree.
#
#   copy_back.sh [<release-run-stamp>]
#
# <release-run-stamp> selects one release run by the timestamp TTI puts in its
# file names, e.g. 2026-08-16_05-11-42. Without it the newest release report is
# used and its stamp is derived from the file name, so a stale earlier run
# cannot leak in through a "newest match" glob on a different directory.
#
# Copied: the final release markdown and its report-data JSON, the run spec TTI
# wrote (which is what proves which implementation was evaluated), the spec this
# stage handed to run.py, the per-task eval results JSON, every per-sweep-point
# benchmark JSON, and the run log.
#
# NOT copied: .env, the Hugging Face cache, the persistent volume, weights,
# tensor dumps, profiler CSVs, and the per-sample eval dumps (tens of MB of
# model text). The raw server.log is not copied either; logs/server_excerpt.log
# carries the configuration and the TT-side lines instead.
set -uo pipefail

REPO=/home/ttuser/dev/muse-glimmer/tt-metal
MODEL_DIR=models/autoports/meta_models_muse_glimmer_30b
DOC=$REPO/$MODEL_DIR/doc/tti_release
WORK_ROOT=/home/ttuser/dev/muse-glimmer/tti-release/muse-glimmer-30b
W=$WORK_ROOT/cache_root/workflow_logs
REL=$W/reports_output/release

mkdir -p "$DOC"/{report,run_spec,evals,benchmarks,logs}

# --- pick the release run -----------------------------------------------------
if [ "$#" -ge 1 ]; then
  STAMP=$1
  REPORT_MD=$(ls -t "$REL"/report_*"$STAMP"*.md 2>/dev/null | head -1)
else
  REPORT_MD=$(ls -t "$REL"/report_*.md 2>/dev/null | head -1)
  STAMP=$(basename "${REPORT_MD:-}" .md | sed 's/.*_\([0-9-]\{10\}_[0-9-]\{8\}\)$/\1/')
fi
if [ -z "${REPORT_MD:-}" ] || [ ! -f "$REPORT_MD" ]; then
  echo "no release report found under $REL" >&2
  exit 1
fi
echo "release run stamp: $STAMP"
echo "release report   : $(basename "$REPORT_MD")"

copy() { cp "$1" "$2/" && echo "  $(basename "$1") -> ${2##*/}"; }

echo "== final release report =="
copy "$REPORT_MD" "$DOC/report"
for f in "$REL"/data/report_data_*"$STAMP"*.json; do
  [ -f "$f" ] && copy "$f" "$DOC/report"
done

echo "== run spec (proves the evaluated implementation path) =="
# TTI names it runtime_model_spec_<launch-stamp>_<model_id>_<rand>.json; the
# launch stamp is in the report's metadata, so read it from there rather than
# guessing.
SPEC_FROM_REPORT=$(python3 - "$REPORT_MD" <<'PY'
import json, re, sys
txt = open(sys.argv[1]).read()
m = re.search(r'"runtime_model_spec_json":\s*"([^"]+)"', txt)
print(m.group(1) if m else "")
PY
)
if [ -n "$SPEC_FROM_REPORT" ] && [ -f "$SPEC_FROM_REPORT" ]; then
  copy "$SPEC_FROM_REPORT" "$DOC/run_spec"
else
  echo "  !! the report's runtime_model_spec_json path did not resolve: $SPEC_FROM_REPORT" >&2
fi
[ -f "$WORK_ROOT/specs/muse_glimmer_30b_autoport_release.json" ] &&
  copy "$WORK_ROOT/specs/muse_glimmer_30b_autoport_release.json" "$DOC/run_spec"

echo "== eval results (aggregate JSON only; per-sample dumps are left behind) =="
# One results file per task, and this stage runs two. Earlier release runs leave
# their own files in the same directory, so take the newest two, not all of them.
EVAL_DIR=$REL/Muse-Glimmer-30B_p300x2_release/eval_id_muse-glimmer-30b-autoport_Muse-Glimmer-30B_p300x2/meta-models__Muse-Glimmer-30B
for f in $(ls -t "$EVAL_DIR"/results_*.json 2>/dev/null | head -2); do copy "$f" "$DOC/evals"; done

echo "== per-document eval health (a few KB; no model text) =="
# The samples_*.jsonl are tens of MB of model output and stay in the TTI cache.
# This is the part a reviewer needs: response length, score, and whether each
# turn reached the model's visible channel, which is what makes an
# under-budgeted max_gen_toks visible.
for task in ifeval aime25; do
  python3 "$DOC/bench/eval_sample_health.py" --samples-dir "$EVAL_DIR" --task "$task" \
    --out "$DOC/evals/${task}_sample_health.json" || true
done

echo "== per-sweep-point benchmark JSON for this run =="
BENCH_DIR=$REL/Muse-Glimmer-30B_p300x2_release/llm
DAY=${STAMP%%_*}
# The sweep writes one file per point, each stamped with its own start time.
# Take only files newer than the release run's own launch stamp, so a later
# partial sweep cannot be mixed in with the graded one, and say how many.
LAUNCH=$(basename "$SPEC_FROM_REPORT" | sed 's/runtime_model_spec_\([0-9-]*_[0-9-]*\)_.*/\1/')
n=0
for f in "$BENCH_DIR"/benchmark_*.json; do
  [ -f "$f" ] || continue
  [ "$f" -nt "$SPEC_FROM_REPORT" ] || continue
  [ "$f" -ot "$REPORT_MD" ] || continue
  cp "$f" "$DOC/benchmarks/" && n=$((n+1))
done
echo "  $n benchmark JSON file(s) -> benchmarks/ (sweep points between the run's launch stamp $LAUNCH and its report)"
[ "$n" = 18 ] || echo "  !! expected 18 sweep points, got $n -- check for a partial sweep" >&2

# Each benchmark JSON carries `itls` (one inter-token latency per generated
# token) and `generated_texts` (every request's full output) alongside the
# summary statistics computed from them -- 6.8 MB of raw payload across the
# sweep, for 0.1 MB of metrics. Drop the two arrays, record that we did, keep
# every metric.
python3 - "$DOC/benchmarks" <<'TRIM'
import glob, json, os, sys
for f in sorted(glob.glob(f"{sys.argv[1]}/benchmark_*.json")):
    d = json.load(open(f)); dropped = {}
    for k in ("itls", "generated_texts"):
        if k in d:
            v = d.pop(k)
            dropped[k] = {"n": len(v), "note": "dropped by copy_back.sh: raw per-token/per-request payload; this file's summary statistics are computed from it"}
    if dropped:
        d["_trimmed_for_evidence"] = dropped
        json.dump(d, open(f, "w"), indent=1)
TRIM
echo "  trimmed raw per-token/per-request payloads out of the benchmark JSON"

echo "== run log =="
RUN_LOG=$(ls -t "$W"/run_logs/run_*release*.log 2>/dev/null | head -1)
[ -n "$RUN_LOG" ] && copy "$RUN_LOG" "$DOC/logs"

echo "== server excerpt (the raw server.log is gitignored and stays out of the repo) =="
SERVER_LOG=$DOC/server/server.log
if [ -f "$SERVER_LOG" ]; then
  { head -400 "$SERVER_LOG"
    echo "... [truncated] ..."
    grep -nE "reasoning|autoports|generator_vllm|KV cache|max_model_len|max_num_seqs|sample_on_device|DEGRADED|prefill trace|Maximum concurrency" \
      "$SERVER_LOG" | head -400
    echo "... [tail] ..."
    tail -200 "$SERVER_LOG"; } > "$DOC/logs/server_excerpt.log" 2>/dev/null
  du -h "$SERVER_LOG" > "$DOC/server/server_log_size.txt"
  echo "  logs/server_excerpt.log written ($(du -h "$SERVER_LOG" | cut -f1) raw)"
fi

echo "== sizes =="
du -sh "$DOC"
echo "== files over 2 MB (raw server logs are expected here and are gitignored) =="
find "$DOC" -size +2M -printf "  %s %p\n" 2>/dev/null || true
