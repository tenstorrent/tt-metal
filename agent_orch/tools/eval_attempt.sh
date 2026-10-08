#!/usr/bin/env bash
# Evaluate one attempt (or the campaign root) on the device.
#
#   eval_attempt.sh --campaign C --node N            # worker: run from inside your worktree
#   eval_attempt.sh --campaign C --baseline [--runs 3]
#   eval_attempt.sh --campaign C --measure <ref> --label L   # e.g. re-measure the root at round start
#
# Worker mode snapshots the worktree (committed + uncommitted changes) into a
# temporary commit, then under the machine-wide device lock: checks it out in
# the campaign's eval checkout, rebuilds only if non-kernel files changed,
# runs the campaign test under the profiler with a timeout, and writes
# <node dir>/eval/{score.json,summary.md,ops.csv,error.txt}. Full profiler
# output goes to $DREAM_HOME/<c>/reports/<label>/.
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/dream_env.sh"

CAMPAIGN="" NODE="" MODE="node" RUNS="" REF="" LABEL=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --campaign) CAMPAIGN="$2"; shift 2 ;;
    --node) NODE="$2"; shift 2 ;;
    --baseline) MODE="baseline"; shift ;;
    --runs) RUNS="$2"; shift 2 ;;
    --measure) MODE="measure"; REF="$2"; shift 2 ;;
    --label) LABEL="$2"; shift 2 ;;
    *) dream_die "unknown argument $1" ;;
  esac
done
[[ -n "$CAMPAIGN" ]] || dream_die "--campaign is required"

cfg() { dream_py -c "from dream.campaign import load_campaign as L; import json,sys; v=L('$CAMPAIGN').cfg
for k in sys.argv[1].split('.'): v=v[k]
print(v if not isinstance(v,(dict,list)) else json.dumps(v))" "$1"; }

CHOME="$DREAM_HOME/$CAMPAIGN"
EV="$CHOME/eval"
TEST="$(cfg test)"
TIMEOUT="$(cfg eval.timeout_s)"
[[ -d "$EV" ]] || dream_die "no eval checkout at $EV; run setup_campaign.sh first"

# run_eval <commit> <label> -> sets EVAL_STATUS (ok|build_error|hang), EVAL_LOG, EVAL_RC, BUILD_S, RUN_S
run_eval() {
  local snap="$1" label="$2"
  local rep="$CHOME/reports/$label"
  exec 9>"$DREAM_LOCK"
  if ! flock -n 9; then
    echo "[eval] waiting for the device lock ($DREAM_LOCK)..."
    flock 9
  fi
  echo "[eval] $label: checking out ${snap:0:12} in $EV"
  git -C "$EV" checkout --detach -f -q "$snap" || { EVAL_STATUS=infra; EVAL_LOG=""; flock -u 9; return; }
  git -C "$EV" submodule update --init --recursive -q

  local last need=1 t0 t1
  last="$(cat "$EV/.dream_last_build" 2>/dev/null || true)"
  if [[ -n "$last" && -f "$EV/ttnn/ttnn/_ttnn.so" ]]; then
    if ! git -C "$EV" diff --name-only "$last" "$snap" | grep -vE '/kernels/|^agent_orch/' | grep -q .; then
      need=0
    fi
  fi
  mkdir -p "$CHOME/logs"
  BUILD_S=0
  if [[ $need == 1 ]]; then
    echo "[eval] building (log: $CHOME/logs/build_$label.log)"
    t0=$(date +%s)
    (cd "$EV" && ./build_metal.sh --release --enable-ccache --cpm-source-cache "$DREAM_CPM_CACHE") \
      >"$CHOME/logs/build_$label.log" 2>&1
    local brc=$?
    t1=$(date +%s); BUILD_S=$((t1 - t0))
    if [[ $brc != 0 ]]; then
      EVAL_STATUS=build_error; EVAL_LOG="$CHOME/logs/build_$label.log"; EVAL_RC=$brc
      git -C "$EV" checkout --detach -f -q "$last" 2>/dev/null  # keep the tree matching the last good build
      flock -u 9; return
    fi
    echo "$snap" >"$EV/.dream_last_build"
  else
    echo "[eval] kernel-only change since last build: skipping host build"
  fi

  rm -rf "$rep" "$CHOME/jitcache"; mkdir -p "$rep" "$CHOME/jitcache"
  echo "[eval] running $TEST (timeout ${TIMEOUT}s, log: $rep/run.log)"
  t0=$(date +%s)
  (
    cd "$EV" && source "$DREAM_PYENV/bin/activate" &&
      TT_METAL_HOME="$EV" PYTHONPATH="$EV:$EV/ttnn:$EV/tools" TT_METAL_CACHE="$CHOME/jitcache" \
        TRACY_NO_WEB_SERVER=1 timeout --kill-after=30 "$TIMEOUT" \
        python -m tracy -t "$DREAM_TRACY_PORT" -o "$rep" -r -p -v -m pytest "$EV/$TEST"
  ) >"$rep/run.log" 2>&1
  EVAL_RC=$?
  t1=$(date +%s); RUN_S=$((t1 - t0))
  EVAL_LOG="$rep/run.log"
  if [[ $EVAL_RC == 124 || $EVAL_RC == 137 ]]; then
    EVAL_STATUS=hang
    echo "[eval] timed out; resetting the device"
    tt-smi -r >>"$rep/run.log" 2>&1 || echo "[eval] WARNING: tt-smi -r failed" | tee -a "$rep/run.log"
  else
    EVAL_STATUS=ok
  fi
  flock -u 9
}

case "$MODE" in
node)
  [[ -n "$NODE" ]] || dream_die "--node is required"
  WT="$(git rev-parse --show-toplevel)" || dream_die "run from inside your worktree"
  NODE_DIR="$WT/agent_orch/campaigns/$CAMPAIGN/attempts/$NODE"
  OUT="$NODE_DIR/eval"
  mkdir -p "$NODE_DIR"
  score() { dream_py "$DREAM_TOOLS/score.py" --campaign "$CAMPAIGN" --node "$NODE" --out-dir "$OUT" "$@"; }

  # 1. only allowed paths + own node dir may change
  bad="$( { git -C "$WT" diff --name-only HEAD; git -C "$WT" ls-files --others --exclude-standard; } | sort -u |
    dream_py -c "
import sys
from dream.campaign import load_campaign
c = load_campaign('$CAMPAIGN')
print('\n'.join(p for p in sys.stdin.read().split() if not c.allowed(p, '$NODE')))")"
  if [[ -n "$bad" ]]; then
    echo "[eval] forbidden edits:"; echo "$bad"
    score --status forbidden_edit --error "changed files outside allowed_paths: $(echo $bad)"
    exit 0
  fi

  # 2. snapshot the worktree without touching its index
  tmpidx="$(mktemp)"; trap 'rm -f "$tmpidx"' EXIT
  GIT_INDEX_FILE="$tmpidx" git -C "$WT" read-tree HEAD
  GIT_INDEX_FILE="$tmpidx" git -C "$WT" add -A -- . ":!agent_orch/campaigns/$CAMPAIGN/attempts/$NODE/eval"
  tree="$(GIT_INDEX_FILE="$tmpidx" git -C "$WT" write-tree)"
  snap="$(git -C "$WT" commit-tree "$tree" -p HEAD -m "dream eval snapshot $NODE")"

  run_eval "$snap" "$NODE"
  meta="{\"commit_under_test\": \"$snap\", \"build_seconds\": $BUILD_S, \"eval_seconds\": ${RUN_S:-0}, \"report_dir\": \"$CHOME/reports/$NODE\"}"
  if [[ $EVAL_STATUS == ok ]]; then
    score --status ok --log "$EVAL_LOG" --report-dir "$CHOME/reports/$NODE" --rc "$EVAL_RC" --meta "$meta"
  else
    score --status "$EVAL_STATUS" ${EVAL_LOG:+--error-file "$EVAL_LOG"} --meta "$meta"
  fi
  echo "[eval] wrote $OUT"
  ;;
baseline | measure)
  if [[ $MODE == baseline ]]; then
    snap="$(git -C "$EV" rev-parse "dream/$CAMPAIGN/root")"; RUNS="${RUNS:-$(cfg eval.baseline_runs)}"; LABEL="baseline"
  else
    snap="$(git -C "$EV" rev-parse "$REF")"; RUNS="${RUNS:-1}"; [[ -n "$LABEL" ]] || dream_die "--label is required"
  fi
  ms=()
  for i in $(seq 1 "$RUNS"); do
    run_eval "$snap" "${LABEL}_$i"
    [[ $EVAL_STATUS == ok ]] || dream_die "$LABEL run $i: $EVAL_STATUS (see $EVAL_LOG)"
    m="$CHOME/reports/${LABEL}_$i/measure.json"
    dream_py "$DREAM_TOOLS/score.py" --campaign "$CAMPAIGN" --measure-only --log "$EVAL_LOG" \
      --report-dir "$CHOME/reports/${LABEL}_$i" --rc "$EVAL_RC" --out "$m"
    ms+=("$m")
  done
  if [[ $MODE == baseline ]]; then
    dream_py "$DREAM_TOOLS/score.py" --campaign "$CAMPAIGN" --make-baseline "${ms[@]}" --out "$CHOME/ledger/baseline.json"
    echo "[eval] wrote $CHOME/ledger/baseline.json (commit it on the ledger)"
  else
    dream_py - "$CHOME/ledger/baseline.json" "${ms[@]}" <<'PY'
import json, sys
base = json.load(open(sys.argv[1]))
for m in sys.argv[2:]:
    r = json.load(open(m))
    for sid, v in r["shapes"].items():
        b = base["shapes"][sid]["us_chip_mean"]
        d = (v["us_chip_mean"] - b) / b * 100
        flag = "  DRIFT" if abs(d) > base["noise_pct"] else ""
        print(f"{sid:28s} {v['us_chip_mean']:8.3f} us  vs baseline {b:8.3f}  ({d:+.2f}%){flag}")
PY
  fi
  ;;
esac
