# SDPA autofix config — sourced by fixer.sh.
#
# The fixer reads the SDPA watcher's state (~/.sdpa-watch/state.json), triages
# every new ❌ run into per-test regression signatures, and for eligible ones
# asks a headless Claude to write a fix in an isolated worktree — no device
# runs, no builds. It then opens a DRAFT PR marked for human review and
# dispatches the failing pipeline legs on the PR branch.
#
# The watcher's config is sourced first: REPO, BRANCH, TT_METAL_DIR, MODEL,
# PIPELINES, Slack channel/token, OAuth settings and the PATH self-heal all
# come from there, so the two can never disagree on what is watched.
source "$HOME/.sdpa-watch/config.sh"

# ---- Mode ------------------------------------------------------------------
#   dryrun — everything except push / PR / dispatch: writes the patch, PR body
#            and dispatch commands under proposals/, posts a "would open PR"
#            note to Slack. Ledger records the signature as handled.
#   live   — pushes the branch, opens the draft PR, dispatches pipelines and
#            follows them up on later ticks. Signatures that were only
#            proposed in dryrun become eligible once more.
FIX_MODE="${FIX_MODE:-dryrun}"

# ---- Models ----------------------------------------------------------------
TRIAGE_MODEL="$MODEL"            # structured classification of failing logs
FIX_MODEL="claude-opus-5-5"      # writes the fix; the harder task

# ---- Eligibility & caps ----------------------------------------------------
# A signature is attempted only when it failed in >= MIN_STREAK consecutive
# runs of its workflow, OR triage pinned a culprit commit with evidence.
MIN_STREAK="${MIN_STREAK:-1}"
MAX_NEW_PER_DAY="${MAX_NEW_PER_DAY:-3}"               # PRs (live) or proposals (dryrun) per UTC day
MAX_FIX_PER_TICK="${MAX_FIX_PER_TICK:-1}"              # fix attempts per tick (each takes minutes)
FIX_TIMEOUT_SEC=2400             # hard wall-clock cap on one fix agent run
TRIAGE_TIMEOUT_SEC=900
SCAN_MAX_JUDGE_PER_TICK="${SCAN_MAX_JUDGE_PER_TICK:-8}"  # human-fix candidates judged per tick

# ---- Guards on the agent's diff -------------------------------------------
MAX_DIFF_LINES=300

# ---- Git / GitHub ----------------------------------------------------------
FIX_WORKTREE="/localdev/skrstic/sdpa-fix-wt"   # isolated; never your checkout
BRANCH_PREFIX="skrstic/autofix"
PR_LABELS="made-by-ai,automated"
PR_TITLE_PREFIX="[autofix · needs human review]"
STATUS_CONTEXT="autofix/human-review"
CI_STATUS_CONTEXT="autofix/targeted-ci"

# ---- Slack -----------------------------------------------------------------
# Reuses the watcher's bot token + channel. FIX_SLACK=0 silences the fixer.
FIX_SLACK="${FIX_SLACK:-1}"

# fixlib.py checks commit ancestry in the main checkout.
export TT_METAL_DIR
