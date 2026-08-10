#!/usr/bin/env bash
# skillexp stage launcher, machine A. Runs ONE stage for ONE model in a detached tmux
# session inside the mvasiljevic-ttxla container. One hardware-facing run at a time --
# never launch two of these concurrently, they share the devices and the latency numbers
# are the whole point.
#
#   ./run_stage.sh <branch> <hf_id> <model_dir> <logtag> <start-index> <prompt-relpath>...
#
# Example (phase 1, phi):
#   ./run_stage.sh mvasiljevic/qb2/skillexp/base microsoft/Phi-3.5-mini-instruct \
#     microsoft_phi_3_5_mini_instruct p1-fd-phi 1 \
#     .agents/prompts/model_bringup_multigoal/01-functional-decoder.txt
set -euo pipefail

BR=$1; HF=$2; MD=$3; TAG=$4; IDX=$5; shift 5
PROMPTS="$*"
SESSION="skillexp-$TAG"
LOGDIR="/home/mvasiljevic/skillexp-logs/$TAG"
mkdir -p "$LOGDIR"
LOGDIR_PRE="$LOGDIR"
LOCK=/home/mvasiljevic/tt-metal/.skillexp-STAGE-RUNNING
# P27: a live stage and manual git bookkeeping shared this working tree, and a `git checkout base`
# three minutes into a run restored every factor file under it -- the stage then ran 1h17m with both
# factors present and committed onto the wrong branch. The stage cannot be moved (it needs
# build_Release + python_env at this exact path), so the bookkeeping moved instead:
# use the /home/mvasiljevic/skillexp-book worktree. This lock makes the hazard visible.
if [ -e "$LOCK" ]; then
  echo "PIN GUARD: a stage lock already exists:" >&2; cat "$LOCK" >&2
  echo "  If that stage is dead, remove the lock. Refusing to launch over it." >&2
  exit 1
fi
CONTAINER=${CONTAINER:-mvasiljevic-ttxla}

if docker exec -u mvasiljevic "$CONTAINER" tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "session $SESSION already exists -- refusing to double-launch" >&2
  exit 1
fi
# Check INSIDE the container, where the real multigoal python process lives. A host-side `pgrep -f`
# matches any host process whose command line merely CONTAINS the pattern -- including an operator's own
# diagnostic (`docker exec ... pgrep -c -f 'scripts/multigoal'`). That happened: a monitoring command left
# running alongside the driver made this guard report a live run and refused all 5 challenger cells in
# 12 seconds, with zero multigoal processes actually running. The container-side check cannot be poisoned
# from the host.
if docker exec -u mvasiljevic "${CONTAINER:-mvasiljevic-ttxla}" pgrep -f "scripts/multigoal" >/dev/null 2>&1; then
  echo "a multigoal is already running on this host -- refusing (one hardware run at a time)" >&2
  exit 1
fi

# SKIP-PREP: the caller has an already-running session for this cell and is only attaching. Touching
# the branch here would move the tree under a live stage -- the P27 failure, from the driver.
if [ "$BR" = "SKIP-PREP" ]; then
  echo "attach-only: leaving the checkout untouched"
  exit 0
fi

# The stage runs on the skill branch; do this on the host where git is configured.
git -C /home/mvasiljevic/tt-metal checkout -q "$BR"
HEAD_SHA=$(git -C /home/mvasiljevic/tt-metal rev-parse --short=11 HEAD)
echo "checked out $BR -> $HEAD_SHA"

# PIN GUARD. A stage's own commits land on whatever branch is checked out, so a skill branch can
# silently drift off its pinned SHA and the next stage then builds on the previous model's work.
# That happened once: phi's FD commits landed on skillexp/base and qwen nearly started on top of
# them. Refuse to launch when a skill branch does not match the record.
declare -A PINNED=(
  [mvasiljevic/qb2/skillexp/base]=6f0bf9ad6e1
  [mvasiljevic/qb2/skillexp/fuse-advise]=23b31c3cc8d
  [mvasiljevic/qb2/skillexp/fuse-noadvise]=03a8221501d
  [mvasiljevic/qb2/skillexp/nofuse-advise]=612219d0acd
  [mvasiljevic/qb2/skillexp/nofuse-noadvise]=51b17c3da34
)
want=${PINNED[$BR]:-}
if [ -n "$want" ] && [ "$HEAD_SHA" != "$want" ]; then
  echo "PIN GUARD: $BR is at $HEAD_SHA but the record pins $want." >&2
  echo "  Reset it (git reset --hard $want) or update the record. Refusing to launch." >&2
  exit 1
fi
[ -n "$want" ] && echo "pin guard: $BR == $want OK"

# --- FACTOR GUARD (machine B's B16) -----------------------------------------------------------
# A stage-02 agent on machine B ran `git rebase --onto <fd-tip> <arm+fd-merge>` mid-stage. Because
# fd/ descends from base, that restored every factor file the arm deletes: the run burned 8h28m
# and 5.8M tokens, then blocked on the shard-advise hard gate it had itself resurrected -- on the
# arm that is supposed to have no advisor at all. It was not a nofuse-noadvise measurement.
#
# The factor manipulation lives in tracked .agents/, so any ordinary history rewrite erases it and
# nothing notices. So: assert the state before, and re-assert after (see the tmux block below).
#
# grep -c exits 1 when the count is 0 -- never chain `|| echo` here, it appends a second token and
# corrupts the state string (cost machine B a false FATAL).
factor_state() {
  ( cd /home/mvasiljevic/tt-metal || exit
    printf '%s %s %s %s %s %s' \
      "$(grep -icE 'shard-advise|OPT-015|ttnn-advise' .agents/skills/optimize/SKILL.md 2>/dev/null)" \
      "$(grep -icE 'shard-advise|HARD GATE' .agents/prompts/model_bringup_multigoal/02-optimized-decoder.txt 2>/dev/null)" \
      "$(ls .agents/skills/shard-advise 2>/dev/null | wc -l)" \
      "$(ls .agents/skills/graph-fusing 2>/dev/null | wc -l)" \
      "$([ -e .agents/prompts/model_bringup_multigoal/01b-fused-decoder.txt ] && echo 1 || echo 0)" \
      "$([ -e .agents/prompts/model_bringup_multigoal/02-optimized-decoder.check.sh ] && echo 1 || echo 0)" )
}
# measured per branch: base [7 1 3 2 1 1] · nofuse-noadvise [0 0 0 0 0 0]
#                      nofuse-advise [7 1 3 0 0 1] · fuse-noadvise [0 0 0 2 1 0]
#                      fuse-advise [7 1 3 2 1 1]
BEFORE=$(factor_state)
echo "factor_state(before) = [$BEFORE]"
echo "$BEFORE" > "$LOGDIR_PRE/factor_before" 2>/dev/null || true
if [ -n "${EXPECT_FACTORS:-}" ] && [ "$BEFORE" != "$EXPECT_FACTORS" ]; then
  echo "FATAL: factor state [$BEFORE] != expected [$EXPECT_FACTORS] BEFORE the run. Refusing." >&2
  exit 70
fi


docker exec -u mvasiljevic -w /home/mvasiljevic/tt-metal "$CONTAINER" \
  tmux new-session -d -s "$SESSION" bash -lc "
    set -o pipefail
    source /home/mvasiljevic/tt-metal/python_env/bin/activate
    export TT_METAL_HOME=/home/mvasiljevic/tt-metal
    # The advise arms need the advisor reachable; the no-advise arms must not see it even by
    # accident. ADVISOR=1 is set per stage by the driver, never globally.
    if [ \"${ADVISOR:-0}\" = 1 ]; then
      export TTMLIR_ADVISOR_HOME=/home/mvasiljevic/tt-mlir
      echo \"advisor: TTMLIR_ADVISOR_HOME=\$TTMLIR_ADVISOR_HOME\"
    else
      unset TTMLIR_ADVISOR_HOME
    fi
    echo \"=== stage $TAG started \$(date -Iseconds) ===\"
    # PIN THE AGENT (advchal-v3 / A3). --model and --effort appeared nowhere in v2, so every cell
    # inherited the Codex account default. The advchal-v2 corpus ran gpt-5.6-sol with effort UNSET
    # (recovered from the cells' JSONLs and the driver log). An account-default change would silently
    # invalidate the v2-vs-v3 comparison, so the model is now explicit and effort is deliberately left
    # unset -- pinning it to a value v2 did not use would introduce a new variable rather than fix one.
    python .agents/scripts/multigoal \
      --repo /home/mvasiljevic/tt-metal \
      --codex-bin /home/mvasiljevic/.local/bin/codex \
      --codex-home /home/mvasiljevic/.codex \
      --model ${AGENT_MODEL:-gpt-5.6-sol} \
      --sandbox danger-full-access --approval-policy never \
      --replace HF_MODEL=$HF \
      --replace MODEL_DIR=models/autoports/$MD \
      --replace DECODE_BATCH=${DECODE_BATCH_OVERRIDE:-32} \
      --start-index $IDX \
      --log-dir $LOGDIR \
      $PROMPTS 2>&1 | tee -a $LOGDIR/console.log
    rc=\$?
    # re-assert the factor state: a mid-stage history rewrite silently restores the factor files
    AFTER=\$(cd /home/mvasiljevic/tt-metal && printf '%s %s %s %s %s %s' \
      \"\$(grep -icE 'shard-advise|OPT-015|ttnn-advise' .agents/skills/optimize/SKILL.md 2>/dev/null)\" \
      \"\$(grep -icE 'shard-advise|HARD GATE' .agents/prompts/model_bringup_multigoal/02-optimized-decoder.txt 2>/dev/null)\" \
      \"\$(ls .agents/skills/shard-advise 2>/dev/null | wc -l)\" \
      \"\$(ls .agents/skills/graph-fusing 2>/dev/null | wc -l)\" \
      \"\$([ -e .agents/prompts/model_bringup_multigoal/01b-fused-decoder.txt ] && echo 1 || echo 0)\" \
      \"\$([ -e .agents/prompts/model_bringup_multigoal/02-optimized-decoder.check.sh ] && echo 1 || echo 0)\")
    echo \"\$AFTER\" > $LOGDIR/factor_after
    BEF=\$(cat $LOGDIR/factor_before 2>/dev/null)
    echo \"factor_state(after) = [\$AFTER]  HEAD=\$(cd /home/mvasiljevic/tt-metal && git rev-parse --short=11 HEAD)\" | tee -a $LOGDIR/console.log
    if [ -n \"\$BEF\" ] && [ \"\$AFTER\" != \"\$BEF\" ]; then
      echo \"*** FACTOR DRIFT [\$BEF] -> [\$AFTER]: run CONTAMINATED, do not tag\" | tee -a $LOGDIR/console.log
      rc=71
    fi
    # B17 (machine B): the ancestry test is wrong by construction -- a work branch built from the
    # fd/ tip legitimately never has the arm as an ancestor. The invariant that matters is that the
    # .agents TREE is byte-identical to the arm's: indifferent to history shape, and it catches B16
    # just as well. Both machines independently measured 33ada6f46a4... for nofuse-noadvise, so this
    # doubles as proof the two machines run the same skills.
    if [ -n \"${ARM_REF:-}\" ]; then
      W=\$(cd /home/mvasiljevic/tt-metal && git rev-parse \"${ARM_REF}:.agents\" 2>/dev/null)
      G=\$(cd /home/mvasiljevic/tt-metal && git rev-parse HEAD:.agents 2>/dev/null)
      echo \"agents-tree: arm=\$W head=\$G\" | tee -a $LOGDIR/console.log
      if [ -n \"\$W\" ] && [ \"\$W\" != \"\$G\" ]; then
        echo \"*** .agents TREE DIVERGED from ${ARM_REF}: run CONTAMINATED, do not tag\" | tee -a $LOGDIR/console.log
        rc=71
      fi
    fi
    echo \"=== stage $TAG exited rc=\$rc \$(date -Iseconds) ===\" | tee -a $LOGDIR/console.log
    rm -f /home/mvasiljevic/tt-metal/.skillexp-STAGE-RUNNING
    echo \$rc > $LOGDIR/exit_code
  "

sleep 5
# ORDERING BUG, fixed: the lock used to be written unconditionally after this sleep, but the tmux stage
# removes the lock and writes exit_code when it finishes -- so a stage that died inside the first 5s (a
# rejected prompt, an over-limit objective) was already gone by the time the lock appeared. The result was
# a lock with no owner, which every later cell then waited on forever: one bad launch stalled the whole
# queue. Only claim the lock if the session is actually alive, and report the death as a launch failure.
if ! docker exec -u mvasiljevic "$CONTAINER" tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "stage $TAG died within 5s of launch -- no stage lock written." >&2
  echo "  cause is usually in $LOGDIR/console.log (prompt rejected, objective over the 4000-char limit)." >&2
  tail -3 "$LOGDIR/console.log" 2>/dev/null | sed 's/^/    /' >&2
  rm -f "$LOCK"
  exit 1
fi
printf 'stage=%s branch=%s expect_factors=[%s] launched=%s\nDO NOT run git checkout/branch/reset in /home/mvasiljevic/tt-metal while this exists.\nUse the /home/mvasiljevic/skillexp-book worktree for bookkeeping instead (P27).\n' \
  "$TAG" "$BR" "${EXPECT_FACTORS:-unset}" "$(date -Iseconds)" > "$LOCK"
echo "launched tmux session: $SESSION"
echo "stage lock written: $LOCK"
docker exec -u mvasiljevic "$CONTAINER" tmux ls 2>&1 | sed 's/^/  /'
echo "logs: $LOGDIR"
