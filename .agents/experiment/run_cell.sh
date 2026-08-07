#!/usr/bin/env bash
# skillexp — ONE entry point for one cell. PIPELINE-HARDENING.md §3.
#
#   bash run_cell.sh <arm> <model_dir> [--dry-run] [--base-sha SHA] [--no-publish]
#
# Replaces the ~8 hand-typed git commands per cell in RUN-PLAN §4 (~130 per machine). Everything
# between "start a cell" and "the cell is tagged" happens here, so consecutive runs differ only in
# their arguments and there is no state carried between them -- the cell root is destroyed at the end.
#
# --dry-run exercises EVERY step except the model call itself (multigoal --dry-run), so the whole
# harness -- isolation, locks, gates, diff audit, publish preconditions -- can be validated in
# seconds instead of waiting 3h+ for a stage to finish and then discovering the harness is broken.
# That is not a lesser test: B13, B23 and B30 were all harness bugs, not model bugs, and every one
# of them would have been caught by a dry run.
set -uo pipefail
LOGROOT=${SKILLEXP_LOGROOT:-$HOME/skillexp-logs}
ROOT=${SKILLEXP_ROOT:-$HOME/skillexp}
CANON=${SKILLEXP_CANON:-$HOME/tt-metal}

ARM=${1:?arm}; MD=${2:?model_dir}; shift 2
DRY=0; BASE_SHA=""; PUBLISH=1
while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run)    DRY=1; shift ;;
    --no-publish) PUBLISH=0; shift ;;
    --base-sha)   BASE_SHA=${2:?}; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done
CELL=$ROOT/cells/${ARM}__${MD}
LOCK=${SKILLEXP_ARMS:-$(dirname "$0")/arms}/$ARM.lock
STEP=0
die()  { echo; echo "RUN_CELL FAILED at step $STEP: $*" >&2; exit 1; }
step() { STEP=$((STEP+1)); echo; echo "── step $STEP: $*"; }

[ -f "$LOCK" ] || die "no lockfile $LOCK"
# shellcheck disable=SC1090
lockval() { grep -E "^$1=" "$LOCK" | head -1 | cut -d= -f2-; }

echo "cell: arm=$ARM model=$MD dry_run=$DRY"

# ---------------------------------------------------------------- assertions shared by pre and post
assert_lock() {  # assert_lock <when> <root>
  local when=$1 r=$2 bad=0 pat
  pat=$(lockval must_be_absent)
  if [ -n "$pat" ]; then
    # repo-wide, every model -- NOT just the model under test. base ships a worked
    # openai_gpt_oss_20b advisor report, which a noadvise arm must not carry.
    local hits
    hits=$(cd "$r" && compgen -G "$pat" 2>/dev/null | head -5)
    if [ -n "$hits" ]; then
      echo "    must_be_absent VIOLATED ($when):"; echo "$hits" | sed 's/^/      /'; bad=1
    else echo "    must_be_absent ok ($when): no $pat"; fi
  fi
  if [ "$when" = post ]; then
    local e; e=$(lockval must_exist_after);  e=${e//MODEL_DIR/$MD}
    [ -n "$e" ] && { [ -e "$r/$e" ] && echo "    must_exist ok: $e" || { echo "    MISSING: $e"; bad=1; }; }
    e=$(lockval must_exist_after2); e=${e//MODEL_DIR/$MD}
    [ -n "$e" ] && { [ -e "$r/$e" ] && echo "    must_exist ok: $e" || { echo "    MISSING: $e"; bad=1; }; }
    e=$(lockval must_not_exist_after); e=${e//MODEL_DIR/$MD}
    [ -n "$e" ] && { [ -e "$r/$e" ] && { echo "    MUST NOT EXIST: $e"; bad=1; } || echo "    must_not_exist ok: $e"; }
  fi
  return $bad
}

# ---------------------------------------------------------------- 1. fresh isolated root
step "create fresh isolated cell root"
args=("$ARM" "$MD"); [ -n "$BASE_SHA" ] && args+=(--base-sha "$BASE_SHA")
bash "$(dirname "$0")/newcell.sh" "${args[@]}" 2>&1 | sed 's/^/    /' || die "newcell refused"

# ---------------------------------------------------------------- 2. preflight lock
step "preflight lock assertions"
assert_lock pre "$CELL" || die "preflight lock violated -- refusing to start the stage"

# ---------------------------------------------------------------- 3. record the start SHA
step "pin start state"
START=$(git -C "$CELL" rev-parse HEAD)
echo "    start_sha=$START"

# ---------------------------------------------------------------- 4. activate + run
step "activate canonical path and run the stage"
if [ -d "$CANON" ] && [ ! -L "$CANON" ]; then
  echo "    NOTE: $CANON is still a real directory (legacy layout)."
  echo "          Not swapping it. Running the stage IN PLACE would defeat isolation, so in a real"
  echo "          run this is a hard stop until it is moved to $ROOT/admin/tt-metal."
  [ "$DRY" -eq 1 ] || die "cannot activate: $CANON is a real dir, move it to $ROOT/admin/tt-metal first"
else
  ln -sfn "$CELL" "$CANON"; echo "    activated $CANON -> $CELL"
fi

PROMPTS=(.agents/prompts/model_bringup_multigoal/02-optimized-decoder.txt)
[ "$(lockval graph_fusing)" = on ] && PROMPTS=(.agents/prompts/model_bringup_multigoal/01b-fused-decoder.txt "${PROMPTS[@]}")
echo "    prompts: ${PROMPTS[*]}"
if [ "$DRY" -eq 1 ]; then
  echo "    DRY RUN: invoking multigoal --dry-run (no model call, no device)"
  # `python` is not on PATH outside the venv -- the first dry run got rc=127 here and the script
  # still printed "run_cell OK" at the end. A swallowed non-zero rc is exactly the B13/B23 failure
  # (capturing the wrong command's status), so resolve the interpreter and then ACT on the code.
  PY=""
  for c in "$CELL/python_env/bin/python" /usr/bin/python3 "$(command -v python3 2>/dev/null)"; do
    [ -n "$c" ] && [ -x "$c" ] && { PY=$c; break; }
  done
  [ -n "$PY" ] || die "no python interpreter found (tried the cell venv, /usr/bin/python3, PATH)"
  echo "    interpreter: $PY"
  ( cd "$CELL" && timeout 120 "$PY" .agents/scripts/multigoal --repo "$CELL" --dry-run \
      --replace MODEL_DIR="models/autoports/$MD" --replace DECODE_BATCH=32 \
      --start-index 2 --log-dir "$LOGROOT/dryrun-$ARM-$MD" "${PROMPTS[@]}" 2>&1 ) \
    | tail -8 | sed 's/^/    /'
  rc=${PIPESTATUS[0]}
  echo "    multigoal --dry-run rc=$rc"
  [ "$rc" -eq 0 ] || die "multigoal --dry-run exited $rc -- the harness is broken, fix before a real run"
else
  MOE=0; case "$MD" in *north_mini*|*gemma_4_26b*) MOE=1 ;; esac
  DECODE_BATCH=32; [ "$MOE" -eq 1 ] && DECODE_BATCH=1
  MOE=$MOE DECODE_BATCH=$DECODE_BATCH bash "$LOGROOT/run_stage.sh" "$ARM" "$MD" || true
fi

# ---------------------------------------------------------------- 5. postflight lock + diff audit
step "postflight lock assertions"
if [ "$DRY" -eq 1 ]; then
  echo "    (dry run: no stage output, so must_exist_after cannot hold -- checking must_be_absent only)"
  assert_lock pre "$CELL" || die "postflight: forbidden path appeared"
else
  assert_lock post "$CELL" || die "postflight lock violated -- NOT publishing"
fi

step "writable-paths diff audit"
W=$(lockval writable); W=${W//MODEL_DIR/$MD}
CHANGED=$(git -C "$CELL" diff --name-only "$START" HEAD 2>/dev/null)
OUT=$(echo "$CHANGED" | grep -v "^$W" | grep -v '^$' | head -10)
if [ -n "$OUT" ]; then
  echo "    CHANGES OUTSIDE $W:"; echo "$OUT" | sed 's/^/      /'
  die "stage modified paths outside its writable set (this is how B16 restored both factors)"
fi
echo "    ok: $(echo "$CHANGED" | grep -c . ) changed path(s), all under $W"

# ---------------------------------------------------------------- 6. publish
step "publish"
if [ "$DRY" -eq 1 ] || [ "$PUBLISH" -eq 0 ]; then
  echo "    skipped (dry run / --no-publish). Real run would call publish.sh, which re-asserts the"
  echo "    lock in a throwaway worktree before creating any ref."
else
  bash "$LOGROOT/publish.sh" "$ARM" "$MD" "$(git -C "$CELL" rev-parse --abbrev-ref HEAD)" || die "publish refused"
fi

# ---------------------------------------------------------------- 7. teardown
step "teardown"
if [ "$DRY" -eq 1 ]; then
  echo "    dry run: leaving $CELL in place for inspection"
else
  [ -L "$CANON" ] && rm -f "$CANON"
  rm -rf "$CELL"; echo "    destroyed $CELL -- nothing carries to the next cell"
fi
echo; echo "run_cell OK (arm=$ARM model=$MD dry_run=$DRY)"
