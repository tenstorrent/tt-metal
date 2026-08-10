#!/usr/bin/env bash
# Stage 02b (advisor-challenger) across the finished no-advise optimized decoders.
#
# Operator instruction 2026-07-30: run all 5, in value order. Output is a FASTER optimized_decoder.py
# (or an explicit no-change result), not just an audit.
#
# TREE CONSTRUCTION -- read this before changing it. Each cell's tree is:
#     .agents/       from the challenger skill branch (off base: has shard-advise AND advisor-challenger)
#   + models/autoports/<md>/   from the INCUMBENT CELL TAG (the finished no-advise decoder)
# built by `git checkout <tag> -- models/autoports/<md>` rather than by merging the arm branch. Merging
# a *-noadvise arm would drag in its factor deletions (it removes skills/shard-advise/ entirely), and the
# challenger needs the advisor present. This is deliberately a MIXED tree and is therefore NOT a 2x2 arm
# measurement -- which is why it publishes to its own namespace and the status page says so in a box.
#
# NAMESPACE: writes only  skillexp/done/challenger/<arm>/<md>  (tag)
#                         run/challenger  ·  cell/challenger/<arm>/<md>  ·  claim/challenger/<arm>/<md>
# Machine B's refs and all five skillexp arm branches are READ-ONLY here. Pushes are NON-FORCE; a
# rejection stops the cell and is reported.
set -uo pipefail

METAL=/home/mvasiljevic/tt-metal
LOGROOT=/home/mvasiljevic/skillexp-logs
CONTAINER=mvasiljevic-ttxla
LOG="$LOGROOT/challenger-driver.log"
SKILL_BR=mvasiljevic/qb2/skillexp/challenger-skill-v3
PUB=advchal-v3
HOST=$(hostname)
PROMPT=.agents/prompts/model_bringup_multigoal/02b-advisor-challenger.txt
GATE=.agents/prompts/model_bringup_multigoal/02b-advisor-challenger.check.sh
ADVISOR_HOME=/home/mvasiljevic/tt-mlir
TIMING="$LOGROOT/challenger-timing.tsv"

log() { echo "$(date -Iseconds) $*" | tee -a "$LOG"; }

push_safe() {
  local refspec=$1 out
  out=$(docker exec -u mvasiljevic -w "$METAL" "$CONTAINER" bash -c "timeout 600 git push origin $refspec" 2>&1)
  echo "$out" | grep -E '\->|rejected|error|up to date' | sed 's/^/    /'
  if echo "$out" | grep -qE 'rejected|error:|! \['; then
    # Say WHICH kind of rejection. A generic "rejected" sent me hunting ref contention for a cell that
    # had actually tripped the 100MB file limit, and hunting a lost race for one that had hit a transient
    # network failure. The three need completely different responses.
    if echo "$out" | grep -qiE 'exceeds .* file size limit|GH001|Large files detected'; then
      log "  PUSH REJECTED (FILE TOO LARGE) for $refspec -- a blob exceeds the remote's 100MB limit;"
      log "    it must be removed from history, not deleted in a later commit"
    elif echo "$out" | grep -qiE 'non-fast-forward|fetch first|behind its remote'; then
      log "  PUSH REJECTED (NON-FAST-FORWARD) for $refspec -- the ref moved; park the old value then re-point"
    elif echo "$out" | grep -qiE 'pre-receive hook declined'; then
      log "  PUSH REJECTED (HOOK DECLINED) for $refspec -- see the remote's message above"
    else
      log "  PUSH REJECTED (OTHER) for $refspec -- not forcing; full output in $LOG"
      echo "$out" | tail -6 >> "$LOG"
    fi
    return 1
  fi
  return 0
}

require_no_stage() {
  while [ -e "$METAL/.skillexp-STAGE-RUNNING" ]; do
    log "  a stage is live ($(head -1 "$METAL/.skillexp-STAGE-RUNNING" 2>/dev/null)) -- waiting 60s"
    sleep 60
  done
}

# The advisor toolchain, verified 2026-07-30. Two traps cost real time finding:
#   1. env/activate derives every path from $(pwd), so it MUST be sourced from the tt-mlir checkout.
#      Sourcing it from tt-metal produced a PYTHONPATH of "//third_party/..." and ttnn failed to import.
#   2. Never pipe the bootstrap (`| tail`) -- a pipeline is a subshell, so its PATH/PYTHONPATH exports
#      are discarded and ttnn-advise "disappears" while the bootstrap prints "ready".
# Also note: activate OVERWRITES TT_METAL_HOME with tt-mlir's VENDORED tt-metal, so the capture traces a
# different ttnn build than the incumbent was measured with. The capture target must reach the model
# source through TT_METAL_ROOT instead. That provenance gap applies to all four historical captures too.
advisor_env_probe() {
  docker exec -u mvasiljevic "$CONTAINER" bash -lc "
    export TTMLIR_ADVISOR_HOME=$ADVISOR_HOME
    cd $ADVISOR_HOME || exit 1
    source \$TTMLIR_ADVISOR_HOME/tools/ttnn-jit/integrations/agentic-research/shard-advise/scripts/bootstrap.sh >/dev/null 2>&1
    command -v ttnn-advise >/dev/null 2>&1 || { echo 'ttnn-advise unavailable'; exit 1; }
    ttnn-advise --help >/dev/null 2>&1 || { echo 'ttnn-advise cannot run'; exit 1; }
    echo ok" 2>&1 | tail -1
}

log "=== challenger driver start (pid $$) ==="
probe=$(advisor_env_probe)
if [ "$probe" != ok ]; then
  log "ADVISOR TOOLCHAIN NOT USABLE ($probe) -- refusing to start. Nothing can be captured."
  exit 1
fi
log "advisor toolchain OK (ttnn-advise runs; tt-mlir pinned $(git -C $ADVISOR_HOME rev-parse --short=11 HEAD))"

[ -f "$TIMING" ] || printf 'cell\tarm\tmodel_dir\tincumbent_tag\tstart\tend\tseconds\tgate\ttagged\n' > "$TIMING"

# value order, per the findings doc + operator confirmation
#   arm|model_dir|short|incumbent_tag_sha|decode_batch|hf_model
# phiFN is FIRST here only because its stage is finishing now and it must be published before any other
# cell runs: phiB and phiA share its model_dir, so their build purges doc/advisor_challenger and would
# destroy any of phiFN's output that the stage had not yet committed.
CELLS='
fuse-noadvise|microsoft_phi_3_5_mini_instruct|phiFN|6e04e475cf41|32|microsoft/Phi-3.5-mini-instruct
nofuse-noadvise|google_gemma_4_26b_a4b_it|g26B|3a006fa031dc|1|google/gemma-4-26B-A4B-it
nofuse-noadvise-onA|google_gemma_4_26b_a4b_it|g26onA|e578352fc071|1|google/gemma-4-26B-A4B-it
fuse-noadvise|coherelabs_north_mini_code_1_0|nmFN|55b77536191d|1|CohereLabs/North-Mini-Code-1.0
nofuse-noadvise|coherelabs_north_mini_code_1_0|nmB|1604664b424a|1|CohereLabs/North-Mini-Code-1.0
nofuse-noadvise-onA|coherelabs_north_mini_code_1_0|nmOnA|ac0f349992f0|1|CohereLabs/North-Mini-Code-1.0
nofuse-noadvise|qwen_qwen3_6_27b|qwenB|ce1b1b13f752|32|Qwen/Qwen3.6-27B
fuse-noadvise|qwen_qwen3_6_27b|qwen|c5c4223d83cb|32|Qwen/Qwen3.6-27B
fuse-noadvise|google_gemma_4_26b_a4b_it|g26FN|851add5b57fa|1|google/gemma-4-26B-A4B-it
nofuse-noadvise|microsoft_phi_3_5_mini_instruct|phiB|c2331f8bccfb|32|microsoft/Phi-3.5-mini-instruct
nofuse-noadvise-onA|microsoft_phi_3_5_mini_instruct|phiA|7b050e2281f7|32|microsoft/Phi-3.5-mini-instruct
'

echo "$CELLS" | while IFS='|' read -r arm md short inc db hf; do
  [ -n "${arm:-}" ] || continue

  if [ -e "$LOGROOT/.challenger-STOP" ] \
     && ! docker exec -u mvasiljevic "$CONTAINER" tmux has-session -t "skillexp-p-$PUB-$short" 2>/dev/null; then
    log "=== STOP SENTINEL: holding before $PUB/$arm/$md. Remove $LOGROOT/.challenger-STOP to continue."
    break
  fi
  if git -C "$METAL" rev-parse -q --verify "refs/tags/skillexp/done/$PUB/$arm/$md" >/dev/null 2>&1; then
    log "CELL $PUB/$arm/$md already tagged -- skipping"; continue
  fi

  log "=== CELL $PUB/$arm/$md (incumbent $inc, DECODE_BATCH=$db) ==="
  git -C "$METAL" fetch origin --quiet 2>/dev/null || true
  git -C "$METAL" fetch origin --tags --quiet 2>/dev/null || true

  # the incumbent must exist and must genuinely be a no-advise decoder
  if ! git -C "$METAL" rev-parse --verify -q "$inc^{commit}" >/dev/null 2>&1; then
    log "  incumbent commit $inc not found -- cell not attempted"
    echo "$PUB/$arm/$md (no incumbent)" >> "$LOGROOT/challenger-NOT-ATTEMPTED"; continue
  fi
  if [ "$(git -C "$METAL" ls-tree -d --name-only "$inc" -- "models/autoports/$md/doc/optimized_decoder/shard_advise" | wc -l)" != 0 ]; then
    log "  REFUSING: incumbent $inc already carries a shard_advise/ tree, so it is not a no-advise"
    log "  decoder and the whole premise of this stage does not hold for it."
    echo "$PUB/$arm/$md (incumbent not advisor-free)" >> "$LOGROOT/challenger-NOT-ATTEMPTED"; continue
  fi

  if ! CLAIM_BASE="$SKILL_BR" MACHINE=a bash "$LOGROOT/skillexp_claim.sh" claim "$PUB/$arm" "$md" >>"$LOG" 2>&1; then
    log "  CLAIM REFUSED for $PUB/$arm/$md -- skipping"; continue
  fi
  log "  claimed $PUB/$arm/$md"

  tag="p-$PUB-$short"
  work="skillexp-cell/$PUB/$short"

  # Decide whether a live stage for this cell can be ADOPTED. Only if it is running the current skill
  # commit: a stage launched under a superseded method must not be adopted, because gating a stale run with
  # the new gate would publish it as though it had been measured the new way. It is never killed -- we wait
  # for it and then rebuild the cell from scratch.
  # PUBLISH-ONLY: a stage that already FINISHED under the current method but was never published must be
  # published, not re-run. Without this the driver only ever adopts a LIVE session, so a completed run whose
  # driver was stopped gets silently rebuilt and its device hours thrown away -- which is what happened to
  # an earlier re-run in this project whose publish failed. Guarded hard: clean exit, complete status,
  # current skill tree, and real output on the work branch. The freshness check still runs afterwards.
  adopt=0
  publish_only=0
  if ! docker exec -u mvasiljevic "$CONTAINER" tmux has-session -t "skillexp-$tag" 2>/dev/null \
     && [ -f "$LOGROOT/$tag/exit_code" ] \
     && [ "$(cat "$LOGROOT/$tag/exit_code" 2>/dev/null)" = 0 ] \
     && git -C "$METAL" rev-parse --verify -q "$work" >/dev/null 2>&1; then
    st=$(grep -hoE 'stage_[0-9]+_terminal_status=[a-zA-Z]+' "$LOGROOT/$tag/manifest.txt" 2>/dev/null | tail -1 | cut -d= -f2)
    ag=$(git -C "$METAL" rev-parse "$work:.agents" 2>/dev/null)
    cur=$(git -C "$METAL" rev-parse "$SKILL_BR:.agents" 2>/dev/null)
    # Count artifacts in the WORKING TREE as well as on the branch. The stage does NOT commit its own
    # output -- the driver commits it at publish time -- so a finished run's artifacts sit uncommitted in
    # the working tree. Checking only `git ls-tree "$work"` reported 0 for a cleanly completed 46-minute
    # run, publish-only declined, the rebuild's purge deleted the files, and the work was gone. Untracked
    # files removed by `git clean` are not in the object store, so that loss was unrecoverable.
    # The working-tree count is only meaningful if $work is the branch actually checked out.
    out=$(git -C "$METAL" ls-tree -r --name-only "$work" -- "models/autoports/$md/doc/advisor_challenger" 2>/dev/null | wc -l)
    cur_br=$(git -C "$METAL" symbolic-ref --short HEAD 2>/dev/null)
    if [ "$cur_br" = "$work" ]; then
      wt=$(find "$METAL/models/autoports/$md/doc/advisor_challenger" -type f 2>/dev/null | wc -l)
      out=$((out + wt))
      [ "${wt:-0}" -gt 0 ] && log "  ($wt uncommitted artifact(s) in the working tree on $work)"
    fi
    if [ "$st" = complete ] && [ -n "$cur" ] && [ "$ag" = "$cur" ] && [ "${out:-0}" -gt 0 ]; then
      log "  a COMPLETED unpublished run exists for $short ($out artifacts, current method) -- publishing"
      log "  it instead of re-running; its device hours are not discarded."
      publish_only=1
      t0=$( [ -f "$LOGROOT/$tag/start_epoch" ] && cat "$LOGROOT/$tag/start_epoch" || stat -c %Y "$LOGROOT/$tag" )
      rc=0
    fi
  fi

  if [ "$publish_only" = 0 ] && docker exec -u mvasiljevic "$CONTAINER" tmux has-session -t "skillexp-$tag" 2>/dev/null; then
    want_agents=$(git -C "$METAL" rev-parse "$SKILL_BR:.agents" 2>/dev/null)
    have_agents=$(git -C "$METAL" rev-parse "$work:.agents" 2>/dev/null)
    if [ -n "$want_agents" ] && [ "$have_agents" = "$want_agents" ]; then
      adopt=1
    else
      log "  a stage for $short is live but its .agents ($have_agents) is not the current skill tree"
      log "  ($want_agents): superseded method. NOT adopting, NOT killing -- waiting for it to end."
      while docker exec -u mvasiljevic "$CONTAINER" tmux has-session -t "skillexp-$tag" 2>/dev/null; do
        sleep 60
      done
      log "  superseded stage for $short ended; rebuilding this cell from scratch"
      rm -f "$LOGROOT/$tag/exit_code"
    fi
  fi

  if [ "$publish_only" = 1 ]; then
    :   # nothing to launch or wait for; fall straight through to the gate + publish
  elif [ "$adopt" = 1 ]; then
    log "  already running the current method -- attaching, leaving its branch and tree alone"
    t0=$( [ -f "$LOGROOT/$tag/start_epoch" ] && cat "$LOGROOT/$tag/start_epoch" || stat -c %Y "$LOGROOT/$tag" )
  else
    require_no_stage
    if ! ( cd "$METAL" || exit 1
           git merge --abort 2>/dev/null; git reset -q --hard HEAD 2>/dev/null
           git checkout -q -B "$work" "$SKILL_BR" || exit 1
           # the incumbent's finished decoder, source only -- see TREE CONSTRUCTION above
           git checkout "$inc" -- "models/autoports/$md" || exit 1
           # PURGE THE STAGE'S OWN OUTPUT DIR *BEFORE* STAGING. A superseded stage keeps writing files
           # after its last commit, so those files are UNTRACKED -- and a branch reset does not remove
           # untracked files. Deleting after `git add -A` (as this did) staged the leftovers into the
           # rebuild commit, and the isolation check then refused every cell for carrying output it had
           # just committed itself. Purge from index, worktree and ignored files, then stage.
           git rm -r -q --cached --ignore-unmatch "models/autoports/$md/doc/advisor_challenger" 2>/dev/null || true
           rm -rf "models/autoports/$md/doc/advisor_challenger"
           git clean -fdxq "models/autoports/$md/doc/advisor_challenger" 2>/dev/null || true
           git add -A "models/autoports/$md"
           git diff --cached --quiet || git commit -q --no-verify \
             -m "challenger $arm/$md: incumbent decoder from $inc onto the challenger skill tree" ) >>"$LOG" 2>&1; then
      log "  CANNOT BUILD WORK BRANCH -- cell not attempted"
      echo "$PUB/$arm/$md (work branch)" >> "$LOGROOT/challenger-NOT-ATTEMPTED"; continue
    fi
    # isolation: the stage's own output must not pre-exist, or it measures already-challenged -> more
    if [ "$(git -C "$METAL" ls-tree -d --name-only HEAD -- "models/autoports/$md/doc/advisor_challenger" | wc -l)" != 0 ]; then
      log "  REFUSING: advisor_challenger/ already present on the work branch"; continue
    fi
    log "  work branch $work: skills=$(git -C "$METAL" rev-parse --short=11 "$SKILL_BR") incumbent=$inc"
    mkdir -p "$LOGROOT/$tag"
    t0=$(date +%s); echo "$t0" > "$LOGROOT/$tag/start_epoch"
    rm -f "$LOGROOT/$tag/exit_code"
    # ADVISOR=1: this stage REQUIRES the advisor, unlike the arm it took the incumbent from
    # ARM_REF drives run_stage.sh's .agents tree-divergence check (B17), and run_stage.sh dereferences
    # it WITHOUT a default, so under `set -u` an unset ARM_REF kills the launch outright -- it took all
    # five cells down in 7 seconds on the first attempt. For this stage the reference tree is the
    # challenger SKILL branch: the work branch takes .agents from it verbatim, so any mid-run change to
    # .agents is a contamination signal exactly as it is on an arm.
    if ! ADVISOR=1 TTMLIR_ADVISOR_HOME="$ADVISOR_HOME" DECODE_BATCH_OVERRIDE="$db" \
         ARM_REF="$SKILL_BR" \
         "$LOGROOT/run_stage.sh" "$work" "$hf" "$md" "$tag" 2 "$PROMPT" >>"$LOG" 2>&1; then
      log "  LAUNCH FAILED"; continue
    fi
    log "  launched (DECODE_BATCH=$db), waiting"
  fi

  if [ "$publish_only" = 0 ]; then
    while [ ! -f "$LOGROOT/$tag/exit_code" ]; do sleep 60; done
    rc=$(cat "$LOGROOT/$tag/exit_code")
  fi
  t1=$(date +%s); secs=$((t1 - t0))
  ts=$(grep -hoE 'stage_[0-9]+_terminal_status=[a-zA-Z]+' "$LOGROOT/$tag/manifest.txt" 2>/dev/null | tail -1 | cut -d= -f2)
  log "  stage rc=$rc terminal_status=${ts:-unknown} wall=${secs}s"


# --- FRESHNESS: the re-run must build its own artifacts, not recover the old attempt's ------------
# The clean-tree preflight is not sufficient on its own. A previous re-run elsewhere in this project
# satisfied exactly that check and then cherry-picked its predecessor's commits out of the object store
# two minutes later. Three independent tests, because they fail differently:
#   1. author date -- cherry-pick and rebase PRESERVE it, so a commit touching this stage's output that
#      predates the stage start came from somewhere else;
#   2. reflog -- a cherry-pick or foreign merge onto the work branch is recorded verbatim;
#   3. blob identity vs every parked OLD-METHOD copy of the same cell -- catches hand-copied files,
#      which have fresh commits but stale content.
challenger_is_fresh() {   # challenger_is_fresh <md> <work> <t0epoch> -> 0 ok, 1 reject
  local md=$1 work=$2 t0=$3
  # NOT on the same `local` line: bash expands every argument to `local` BEFORE performing any
  # assignment, so P would interpolate the OLD $md. It only worked so far because the driver loop
  # happens to set a global `md` to the same value -- a masked landmine, not correctness.
  local P="models/autoports/$md/doc/advisor_challenger"
  local bad=0 n=0 c ad floor=$((t0 - 120))
  while read -r c ad; do
    [ -n "$c" ] || continue
    n=$((n + 1))
    if [ "$ad" -lt "$floor" ]; then
      log "  FRESHNESS REJECT: commit ${c:0:11} touching $P authored $(date -Iseconds -d "@$ad"), before"
      log "    this stage started ($(date -Iseconds -d "@$t0")) -- inherited history, not built here"
      bad=1
    fi
  done < <(git -C "$METAL" log --format='%H %at' "$SKILL_BR..$work" -- "$P" 2>/dev/null)
  # Zero artifact-touching commits used to be treated as fraud here. It is the NORMAL case when the
  # driver has just committed the output itself, and reading it as fraud rejected five clean cells. What
  # matters is whether artifacts EXIST and whether any commit that touches them predates this stage.
  local have
  have=$(git -C "$METAL" ls-tree -r --name-only "$work" -- "$P" 2>/dev/null | wc -l)
  if [ "${have:-0}" = 0 ]; then
    log "  FRESHNESS REJECT: no artifacts under $P at all -- the stage produced nothing"; return 1
  fi
  [ "$n" = 0 ] && log "  FRESHNESS: no pre-existing commits touch $P (expected: the driver commits the output)"
  local cp
  cp=$(git -C "$METAL" reflog show "$work" 2>/dev/null | grep -icE 'cherry-pick|rebase' || true)
  if [ "${cp:-0}" -gt 0 ]; then
    log "  FRESHNESS REJECT: work branch reflog records $cp cherry-pick/rebase event(s)"
    bad=1
  fi
  local other shared ident pth mine theirs
  for other in $(git -C "$METAL" for-each-ref --format='%(refname)' \
                   'refs/heads/mvasiljevic/qb2/skillexp/parked/*challenger*' 2>/dev/null); do
    # EVERY parked challenger copy, not just OLDMETHOD-*. Three method revisions produced three families of
    # parked artifacts (OLDMETHOD-*, RUN2-*, RUN3PARTIAL-*), and a glob that only covered the first would
    # have let a re-run inherit the other two byte-for-byte and still be called fresh.
    case "$other" in *"$md"*|*"$short"*|*run-challenger*) ;; *) continue ;; esac
    shared=0; ident=0
    while IFS= read -r pth; do
      [ -n "$pth" ] || continue
      mine=$(git -C "$METAL" rev-parse "$work:$pth" 2>/dev/null)
      theirs=$(git -C "$METAL" rev-parse "$other:$pth" 2>/dev/null)
      [ -n "$mine" ] && [ -n "$theirs" ] || continue
      shared=$((shared + 1))
      [ "$mine" = "$theirs" ] && { ident=$((ident + 1)); log "  FRESHNESS: identical to $other: $pth"; }
    done < <(git -C "$METAL" ls-tree -r --name-only "$work" -- "$P" 2>/dev/null)
    [ "$shared" = 0 ] && continue
    log "  FRESHNESS: vs $other -- shared=$shared identical=$ident"
    if [ "$ident" -ge 2 ]; then
      log "  FRESHNESS REJECT: $ident byte-identical artifacts shared with the OLD-METHOD copy $other"
      bad=1
    fi
  done
  [ "$bad" = 0 ] && log "  FRESHNESS OK: $n commit(s) authored in-stage, no cherry-pick/rebase, no old-method reuse"
  return "$bad"
}

  # ---- COMMIT THE STAGE'S OUTPUT FIRST, before any check ----------------------------------------
  # The stage writes doc/advisor_challenger/ into the WORKING TREE and does not commit it; the driver
  # commits at publish time. Every check that reasoned over commits therefore saw nothing:
  #   * the freshness check rejected all five cells with "no commit touches <path> at all" -- the normal
  #     case, read as fraud -- then parked a work branch that contained no artifacts, and the next cell's
  #     purge deleted the files. ~103 minutes of device time lost across five cells.
  #   * the publish-only guard had the same blind spot (fixed earlier, separately).
  # Committing here makes the artifacts durable before anything can reject or purge them, and gives every
  # later check a real commit to reason about.
  if [ "$rc" = 0 ]; then
    ( cd "$METAL" || exit 1
      git add -f -A "models/autoports/$md" 2>/dev/null || true
      git ls-files "models/autoports/$md" | grep -E '__pycache__|\.pyc$' | xargs -r git rm -q --cached 2>/dev/null || true
      git diff --cached --quiet || git commit -q --no-verify -m "challenger $arm/$md: stage output (committed by the driver on completion)"
    ) >>"$LOG" 2>&1
    nart=$(git -C "$METAL" ls-tree -r --name-only "$work" -- "models/autoports/$md/doc/advisor_challenger" 2>/dev/null | wc -l)
    log "  committed the stage output: $nart artifact(s) now durable on $work"
    if [ "${nart:-0}" = 0 ]; then
      log "  STAGE PRODUCED NOTHING under doc/advisor_challenger -- not gating, not tagging"
      rc=98
    fi
  fi

    # ---- BLOB SIZE GUARD ------------------------------------------------------------------------
    # GitHub hard-refuses any file over 100 MB, and a push validates EVERY REACHABLE blob -- so deleting
    # an oversized file in a later commit does not help; it has to leave the history. One cell's advisor
    # decision trace was 112.56 MB and its publish was rejected with a generic "remote rejected" that
    # looked like ref contention. Compress in place and rewrite THIS commit, before anything is pushed.
    # gzip is not a loss: the trace went 112.6 MB -> 1.1 MB and gunzip restores it byte for byte.
    big=$(git -C "$METAL" ls-tree -r -l "$work" -- "models/autoports/$md" | awk '$4>95000000{print $5}')
    if [ -n "$big" ]; then
      log "  BLOB GUARD: file(s) over 95MB would be refused by the remote; compressing before publish:"
      ( cd "$METAL" || exit 1
        for f in $big; do
          case "$f" in
            *.json|*.log|*.txt|*.csv|*.mlir)
              sz=$(stat -c %s "$f" 2>/dev/null)
              gzip -9 "$f" 2>/dev/null && {
                nsz=$(stat -c %s "$f.gz" 2>/dev/null)
                git rm -q --cached "$f" 2>/dev/null || true
                git add -f "$f.gz"
                echo "    gzipped $f  $((sz/1048576))MB -> $((nsz/1048576))MB"
                printf '%s\n' "\`$(basename "$f")\` is stored gzipped: the raw file was $((sz/1048576))MB," \
                  "over GitHub's 100MB per-file limit, and a push validates every reachable blob so a later" \
                  "deletion would not have helped. Content is unchanged; gunzip restores it byte for byte." \
                  > "$(dirname "$f")/README-compressed.md"
                git add -f "$(dirname "$f")/README-compressed.md"
              } ;;
            *) echo "    CANNOT compress $f (not a text artifact) -- publish will fail; investigate" ;;
          esac
        done
        git diff --cached --quiet || git commit -q --no-verify --amend --no-edit ) >>"$LOG" 2>&1
      still=$(git -C "$METAL" ls-tree -r -l "$work" -- "models/autoports/$md" | awk '$4>95000000{print $5}')
      [ -n "$still" ] && log "  BLOB GUARD: STILL oversized after compression: $still"
      log "  BLOB GUARD: largest artifact now $(git -C "$METAL" ls-tree -r -l "$work" -- "models/autoports/$md" | awk '{if($4>m)m=$4}END{printf "%.1fMB", m/1048576}')"
    fi

  # ---- the gate decides, not the stage's self-report -------------------------------------------
  gate=fail
  if [ "$rc" = 0 ] && ! challenger_is_fresh "$md" "$work" "$t0"; then
    log "  NOT FRESH -- parking, not gating and not tagging"
    ( cd "$METAL" && git branch -f "mvasiljevic/qb2/skillexp/parked/NOTFRESH-challenger-$arm-$md" "$work" ) 2>/dev/null || true
    push_safe "refs/heads/mvasiljevic/qb2/skillexp/parked/NOTFRESH-challenger-$arm-$md" || true
    rc=99
  fi
  if [ "$rc" = 0 ]; then
    # CHALLENGER_DECODE_BATCH lets the gate assert capture_batch == incumbent decode_batch == requested
    if docker exec -u mvasiljevic -w "$METAL" "$CONTAINER" \
         bash -lc "TT_METAL_HOME=$METAL CHALLENGER_DECODE_BATCH=$db bash $GATE $md" >>"$LOG" 2>&1; then
      gate=pass; log "  GATE PASSED"
    else
      log "  GATE FAILED -- see $LOG. Not tagging."
    fi
  fi

  tagged=no
  if [ "$gate" = pass ]; then
    ( cd "$METAL"
      # output was already committed on completion; just pick up anything written since
      git add -f -A "models/autoports/$md" 2>/dev/null || true
      git ls-files "models/autoports/$md" | grep -E '__pycache__|\.pyc$' | xargs -r git rm -q --cached 2>/dev/null || true
      git diff --cached --quiet || git commit -q --no-verify -m "challenger $arm/$md: late stage output"
      git fetch origin --quiet 2>/dev/null || true
      # ONE RUN BRANCH PER CELL, not per experiment. phiB, phiFN and phiA are all
      # microsoft_phi_3_5_mini_instruct and write the same doc/advisor_challenger path, so a shared run
      # branch made each phi overwrite the previous one's evidence on merge. Keyed by arm AND model so no
      # two cells can ever land on the same ref.
      base=$(git rev-parse --verify --quiet "origin/mvasiljevic/qb2/skillexp/run/$PUB/$arm/$md" || echo "$work")
      git checkout -q -B "mvasiljevic/qb2/skillexp/run/$PUB/$arm/$md" "$base"
      git merge --no-edit -q "$work" ) >>"$LOG" 2>&1 \
      && push_safe "HEAD:refs/heads/mvasiljevic/qb2/skillexp/run/$PUB/$arm/$md" \
      && ( cd "$METAL" && git tag -a "skillexp/done/$PUB/$arm/$md" -m \
             "machine=a host=$HOST cell=$PUB/$arm/$md incumbent=$inc DECODE_BATCH=$db
skills=$SKILL_BR@$(git rev-parse --short=11 $SKILL_BR) advisor=tt-mlir@$(git -C $ADVISOR_HOME rev-parse --short=11 HEAD)
stage=02b-advisor-challenger gate=passed invariant=final_ms<=incumbent_ms
NOT a 2x2 arm result -- mixed tree: base .agents + incumbent model source
tagged=$(date -Iseconds)" ) \
      && push_safe "refs/tags/skillexp/done/$PUB/$arm/$md" \
      && { tagged=yes; log "  PUBLISHED + TAGGED skillexp/done/$PUB/$arm/$md"; }
    # PARK-THEN-FORCE the cell pointer. A plain push is rejected non-fast-forward whenever this ref
    # already exists from an earlier round, which left the tag pointing at new work while cell/ still
    # asserted RETRACTED work was the cell's history. Forcing blind would have destroyed 22 unpreserved
    # artifacts on one such ref, so: if the existing remote value is not already reachable from some
    # parked branch, park it first, and only then move the pointer.
    cellref="refs/heads/mvasiljevic/qb2/skillexp/cell/$PUB/$arm/$md"
    curval=$(docker exec -u mvasiljevic -w "$METAL" "$CONTAINER" \
               bash -c "timeout 120 git ls-remote origin $cellref 2>/dev/null" | cut -f1)
    if [ -n "$curval" ]; then
      git -C "$METAL" fetch origin "$cellref" --quiet 2>/dev/null || true
      preserved=""
      for pk in $(git -C "$METAL" for-each-ref --format='%(refname:short)' \
                    'refs/heads/mvasiljevic/qb2/skillexp/parked/*' 2>/dev/null); do
        [ "$(git -C "$METAL" rev-parse "$pk" 2>/dev/null)" = "$curval" ] && preserved="$pk"
      done
      if [ -z "$preserved" ]; then
        pref="mvasiljevic/qb2/skillexp/parked/PREV-cell-$PUB-$arm-$md"
        git -C "$METAL" branch -f "$pref" "$curval" 2>/dev/null || true
        push_safe "refs/heads/$pref" \
          && log "  parked the previous cell pointer ${curval:0:11} at $pref before moving it" \
          || { log "  could NOT park the previous cell pointer ${curval:0:11} -- leaving cell/ untouched"; curval=SKIP; }
      else
        log "  previous cell pointer ${curval:0:11} already preserved at $preserved"
      fi
    fi
    if [ "$curval" != SKIP ]; then
      ( cd "$METAL" && git branch -f "mvasiljevic/qb2/skillexp/cell/$PUB/$arm/$md" "$work" ) 2>/dev/null || true
      docker exec -u mvasiljevic -w "$METAL" "$CONTAINER" \
        bash -c "timeout 300 git push -f origin refs/heads/mvasiljevic/qb2/skillexp/cell/$PUB/$arm/$md" \
        >/dev/null 2>&1 && log "  cell pointer -> $(git -C "$METAL" rev-parse --short=11 "$work")" \
        || log "  cell pointer push failed (tag and run branch are already published)"
    fi
  else
    ( cd "$METAL"
      git checkout -q -B "mvasiljevic/qb2/skillexp/wip/$PUB/$arm/$md" 2>/dev/null
      git add -A "models/autoports/$md" 2>/dev/null
      git diff --cached --quiet || git commit -q --no-verify -m "WIP challenger $arm/$md: gate not passed" ) >>"$LOG" 2>&1
    push_safe "HEAD:refs/heads/mvasiljevic/qb2/skillexp/wip/$PUB/$arm/$md" || true
  fi

  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$PUB/$arm/$md" "$arm" "$md" "$inc" \
    "$(date -Iseconds -d "@$t0")" "$(date -Iseconds -d "@$t1")" "$secs" "$gate" "$tagged" >> "$TIMING"

  if [ "$tagged" = yes ]; then
    CLAIM_BASE="$SKILL_BR" MACHINE=a bash "$LOGROOT/skillexp_claim.sh" release "$PUB/$arm" "$md" >>"$LOG" 2>&1 \
      && log "  claim released"
  else
    log "  keeping the claim: $PUB/$arm/$md not tagged"
  fi
  PATH=/home/mvasiljevic/.skillexp-bin:$PATH MACHINE=a METAL="$METAL" LOGROOT="$LOGROOT" \
    bash "$LOGROOT/skillexp_status.sh" >>"$LOGROOT/status-render.log" 2>&1 || true
done

log "=== challenger driver done ==="
