---
description: |
  `/test` — an on-demand CI pipeline selector for tt-metal pull requests.

  tt-metal has ~34 optional pipelines that are NOT part of pr-gate. Today a developer
  who wants one has to know it exists, find it in the Actions tab, click "Run workflow",
  and remember to point it at their own branch instead of the default `main`. Most
  people either skip that entirely (and find out at merge-gate) or run far more than
  the change warrants.

  This workflow closes that gap: comment `/test` on a PR and an agent reads the diff,
  reasons about which subsystems and which *hardware* the change can actually affect,
  and dispatches only the matching pipelines — narrowed to the relevant platforms —
  against the PR's own branch. Selecting **zero** pipelines is a valid, expected
  outcome for a docs-only or comment-only change.

  It never merges, never pushes, and never modifies the PR. Its only effects are
  dispatching allowlisted workflows and posting one summary comment.

on:
  slash_command:
    name: test
    # PR comments only. `/test` is meaningless on a plain issue — there is no diff to
    # reason about and no head branch to dispatch against — and allowing `issues` would
    # spend a model turn just to reply "not a PR". Restricting the trigger is cheaper
    # and clearer than handling that case in the prompt.
    events: [pull_request_comment]
  # Acknowledge the command immediately. These pipelines take minutes to even start
  # queueing, so without a reaction the developer has no signal that `/test` was seen.
  reaction: "eyes"
  # DEFAULT-DENY on who can burn hardware. This mirrors gh-aw's default
  # ([admin, maintainer, write]) but is written out because it is the primary
  # authorization control for this workflow, not an incidental one: a Galaxy or T3000
  # run occupies scarce physical silicon that the whole org shares. Anyone who can
  # invoke `/test` could already dispatch these same workflows by hand from the Actions
  # tab, so this grants no new capability — it just refuses to extend that capability
  # to drive-by commenters on a public repo.
  roles: [admin, maintainer, write]

# The agent only reads a diff and reasons; it never builds. The long pole is model
# latency, not compute. Dispatched pipelines run in their own workflow runs and are
# NOT bounded by this timeout.
timeout-minutes: 20

permissions:
  contents: read
  pull-requests: read
  actions: read
  copilot-requests: write

# One in-flight `/test` per pull request. gh-aw disables `cancel-in-progress` for
# command triggers, so without a PR-scoped group a second `/test` on PR #2 would queue
# behind an unrelated one on PR #1. Keyed on the PR number rather than the default
# workflow+ref because `issue_comment` events all carry the same `github.ref` (`main`),
# which would collapse every PR into a single shared slot.
concurrency:
  group: "gh-aw-${{ github.workflow }}-${{ github.event.issue.number }}"

engine: copilot
model: claude-sonnet-5

# Cost backstop, matching the skills-reviewer workflows. `/test` is invoked by hand, so
# spend scales with how often developers reach for it rather than with repo activity —
# this caps a bad day (or a loop of retried invocations) without throttling normal use.
max-daily-ai-credits: 10000

network: defaults

tools:
  github:
    # Read-only inspection of workflows and runs (`actions`), the PR and its files
    # (`pull_requests`, `repos`), and event context. No `issues` toolset: this workflow
    # never files or edits issues.
    toolsets: [actions, repos, pull_requests, search, context]
  # Needed to read the pre-fetched PR data from disk and to inspect the
  # `workflow_dispatch` input schema of each candidate pipeline before dispatching it.
  # Unrestricted, as in `silencer.md`, because it does not widen what this workflow can
  # actually do: the agent job is read-only and network-firewalled, and every effect it
  # can have is bounded by the `safe-outputs` allowlists below (34 named workflows,
  # branch refs only, at most 8 dispatches and one comment). A prompt-injected agent
  # gains nothing from a shell that it could not already reach through those.
  bash: true

# Deterministic pre-fetch. Everything here is computed by the runner, NOT by the model:
# the PR head branch in particular must be a fact, not an inference, because it is the
# ref every dispatch is aimed at.
pre-agent-steps:
  - name: Pre-fetch PR metadata and diff
    env:
      GH_TOKEN: ${{ github.token }}
      # `github.event.issue.number` is the only branch that can ever be taken: this
      # workflow compiles to a single `issue_comment` trigger, and the generated guard
      # additionally requires `github.event.issue.pull_request != null`. The
      # `aw_context` fallback chain that the skills-reviewer workflows carry here is
      # deliberately omitted — those need it because they also run under
      # `workflow_dispatch`/`repository_dispatch`, and copying it into a
      # comment-only workflow would just imply a manual entry point that does not exist.
      PR_NUMBER: ${{ github.event.issue.number }}
      EXPR_GITHUB_REPOSITORY: ${{ github.repository }}
      PR_DIFF_MAX_LINES: "3000"
    run: |
      set -euo pipefail
      mkdir -p /tmp/gh-aw/agent

      gh pr view "$PR_NUMBER" --repo "$EXPR_GITHUB_REPOSITORY" \
        --json number,title,body,headRefName,headRefOid,baseRefName,isCrossRepository,headRepositoryOwner,additions,deletions,changedFiles \
        > /tmp/gh-aw/agent/pr-meta.json

      # Full changed-file list. Pipeline selection is driven far more by WHICH paths
      # changed than by the contents of the hunks, so this list must not be truncated
      # even when the diff below is.
      #
      # Deliberately `gh api --paginate` and NOT `gh pr view --json files`: the latter
      # issues a GraphQL query with `files(first: 100)` and silently returns only the
      # first 100 entries with no error and no indication it truncated. tt-metal
      # routinely has PRs well past that (a 174-file PR returns exactly 100), and the
      # missing paths are invisible to the selector — which is precisely how a change
      # that needed Galaxy coverage would get none. The REST endpoint paginates
      # properly. It caps at 3000 files, which no realistic PR reaches; the guard below
      # catches it if one ever does.
      gh api --paginate "repos/$EXPR_GITHUB_REPOSITORY/pulls/$PR_NUMBER/files" \
        --jq '.[].filename' > /tmp/gh-aw/agent/pr-files.txt

      # Cross-check the fetched count against what GitHub reports for the PR, and tell
      # the agent when they disagree so it can widen its selection rather than reason
      # confidently from a partial list.
      FILE_COUNT="$(wc -l < /tmp/gh-aw/agent/pr-files.txt)"
      REPORTED_FILES="$(jq -r '.changedFiles' /tmp/gh-aw/agent/pr-meta.json)"
      if [ "$FILE_COUNT" -lt "$REPORTED_FILES" ]; then
        echo "::warning::Changed-file list is incomplete (${FILE_COUNT} of ${REPORTED_FILES})."
        printf 'INCOMPLETE: fetched %s of %s changed files\n' "$FILE_COUNT" "$REPORTED_FILES" \
          > /tmp/gh-aw/agent/pr-files-truncated.txt
      fi

      # Diff body is best-effort context and IS truncated. Generated lock files are
      # excluded: they are megabytes of compiler output that would crowd out real
      # signal, and `.gitattributes` already marks them linguist-generated.
      gh pr diff "$PR_NUMBER" --repo "$EXPR_GITHUB_REPOSITORY" \
        --exclude '**/*.lock.yml' \
        > /tmp/gh-aw/agent/pr-diff.full
      head -n "${PR_DIFF_MAX_LINES}" /tmp/gh-aw/agent/pr-diff.full > /tmp/gh-aw/agent/pr-diff.patch

      HEAD_REF="$(jq -r '.headRefName' /tmp/gh-aw/agent/pr-meta.json)"
      IS_FORK="$(jq -r '.isCrossRepository' /tmp/gh-aw/agent/pr-meta.json)"

      # Belt-and-braces guard on the ONE invariant that matters (see *The ref rule*).
      # `gh pr view` should never report a head branch of `main` for a real PR, but if
      # anything upstream ever regressed into returning the base branch, this fails the
      # run loudly instead of letting the agent dispatch a fleet of hardware pipelines
      # at `main`. Cheap, and it turns a silent wrong-target into a visible error.
      case "$HEAD_REF" in
        main|master|refs/heads/main|refs/heads/master)
          echo "::error::Refusing to continue: PR head branch resolved to '$HEAD_REF'." >&2
          exit 1
          ;;
      esac
      if [ -z "$HEAD_REF" ] || [ "$HEAD_REF" = "null" ]; then
        echo "::error::Could not resolve PR head branch for #${PR_NUMBER}." >&2
        exit 1
      fi

      printf '%s\n' "$HEAD_REF" > /tmp/gh-aw/agent/pr-head-ref.txt
      printf '%s\n' "$IS_FORK"  > /tmp/gh-aw/agent/pr-is-fork.txt

      # Authoritative copies for the post-agent enforcement step, kept OUTSIDE the
      # agent-writable mount. The agent sandbox mounts /tmp (and /tmp/gh-aw) rw with
      # unrestricted bash, so the two files above are model context, not facts — a
      # prompt-injected agent could rewrite them before the post-step reads them.
      # ${RUNNER_TEMP}/gh-aw is mounted read-only into the sandbox and other
      # ${RUNNER_TEMP} paths are not mounted at all, so a sibling directory there is
      # host-owned for the whole job: written here (before the agent starts), read
      # only by the enforcement post-step (after it exits).
      FACTS_DIR="${RUNNER_TEMP:?}/gh-aw-facts"
      mkdir -p "$FACTS_DIR"
      printf '%s\n' "$HEAD_REF" > "$FACTS_DIR/pr-head-ref.txt"
      printf '%s\n' "$IS_FORK"  > "$FACTS_DIR/pr-is-fork.txt"

      echo "PR #${PR_NUMBER}: head=${HEAD_REF} fork=${IS_FORK} files=$(wc -l < /tmp/gh-aw/agent/pr-files.txt) diff_lines=$(wc -l < /tmp/gh-aw/agent/pr-diff.patch)"

# Deterministic enforcement of *The ref rule* (see the prompt below). The rule is
# executed by a model, and a model can skip it: in run 32947659949 the agent omitted
# `ref` on both of its dispatch calls (while correctly naming the PR branch in its
# summary comment), and gh-aw's fallback chain silently dispatched them against
# `main`. When a dispatch_workflow item carries no `ref`, gh-aw resolves one as
# target-ref > GITHUB_HEAD_REF > GITHUB_REF; an `issue_comment` event sets neither of
# the first two, so the fallback is always `refs/heads/main` — and `allowed-refs` is
# only checked against *explicit* refs, so no safe-outputs configuration can make
# that fallback fail (github/gh-aw dispatch_workflow.cjs). Until gh-aw fails closed
# or resolves the PR head itself, close the gap on our side: after the agent runs,
# rewrite the collected safe-output items so every dispatch_workflow item's `ref` is
# the runner-resolved PR head branch — missing, wrong, or right, it becomes the fact
# computed in the pre-agent step above.
#
# Placement is load-bearing: gh-aw emits post-steps after its "Ingest agent output"
# step (which materializes /tmp/gh-aw/agent_output.json from the safe-outputs JSONL)
# and before the artifact upload that the safe_outputs job downloads and dispatches
# from — so the file rewritten here is exactly the one the dispatcher reads.
#
# The head-ref and fork facts are read from ${RUNNER_TEMP}/gh-aw-facts, which the
# agent sandbox cannot write (see the pre-agent step): the /tmp/gh-aw copies exist
# only as model context and are treated as untrusted here.
post-steps:
  - name: Enforce PR head ref on dispatch_workflow items
    if: always()
    run: |
      set -euo pipefail
      OUT=/tmp/gh-aw/agent_output.json
      FACTS_DIR="${RUNNER_TEMP:?}/gh-aw-facts"
      REF_FILE="$FACTS_DIR/pr-head-ref.txt"
      FORK_FILE="$FACTS_DIR/pr-is-fork.txt"

      # FAIL CLOSED. The agent artifact upload after this step runs
      # `if: always()`, and the safe_outputs job runs whenever the agent job
      # was not skipped — a failed agent job still gets its collected items
      # dispatched. A plain `exit 1` here would therefore ship the
      # un-rewritten items downstream and reopen the exact hole this step
      # exists to close. Instead, any exit that is not an explicit success —
      # including unexpected command failures under `set -euo pipefail` —
      # first empties the item list: no dispatches, no comment, and a red
      # step pointing at what broke.
      neutralize() {
        echo '{"items":[]}' > "$OUT" || true
      }
      finish_ok=0
      trap '[ "$finish_ok" = 1 ] || { echo "::error::Ref enforcement did not complete; discarding all safe-output items." >&2; neutralize; }' EXIT

      # gh-aw's placeholder step (which writes '{"items":[]}' when the agent
      # produced nothing) runs before post-steps, so this file normally exists
      # by now even on a no-dispatch run. Guard anyway: if it is absent there is
      # nothing to enforce and nothing the safe_outputs job could dispatch.
      if [ ! -s "$OUT" ]; then
        echo "No agent output collected; nothing to enforce."
        finish_ok=1
        exit 0
      fi

      # Deterministic fork stop. `workflow_dispatch` only accepts refs that
      # exist in this repository, and a fork's head branch name can *also*
      # exist here by coincidence — forcing it would then green-light a run
      # of the wrong code. The prompt already tells the agent to dispatch
      # nothing for forks, but that rule is executed by a model; enforce it
      # here by stripping every dispatch item while keeping the agent's
      # explanatory comment. Anything other than a literal "false"
      # (including a missing file) is treated as a fork.
      IS_FORK="$(cat "$FORK_FILE" 2>/dev/null || echo unknown)"
      if [ "$IS_FORK" != "false" ]; then
        echo "PR is from a fork (pr-is-fork.txt: '$IS_FORK'); stripping all dispatch_workflow items."
        jq '.items = [ .items[]? | select(.type != "dispatch_workflow") ]' "$OUT" > "$OUT.tmp"
        mv "$OUT.tmp" "$OUT"
        finish_ok=1
        exit 0
      fi

      # The pre-agent step hard-fails the run before the agent ever starts if the
      # head ref cannot be resolved, so an empty file here means something upstream
      # changed shape — discard the items (via the EXIT trap) and fail loudly
      # rather than let a dispatch fall back to main.
      if [ ! -s "$REF_FILE" ]; then
        echo "::error::pr-head-ref.txt is missing or empty; cannot enforce dispatch refs." >&2
        exit 1
      fi
      HEAD_REF="$(cat "$REF_FILE")"
      case "$HEAD_REF" in
        ""|null|main|master|refs/heads/main|refs/heads/master)
          echo "::error::Refusing to enforce dispatch ref '$HEAD_REF'." >&2
          exit 1
          ;;
      esac

      BEFORE="$(jq -c '[.items[]? | select(.type == "dispatch_workflow") | {workflow_name, ref: (.ref // "MISSING")}]' "$OUT")"
      jq --arg ref "$HEAD_REF" \
        '.items = [ .items[]? | if .type == "dispatch_workflow" then .ref = $ref else . end ]' \
        "$OUT" > "$OUT.tmp"
      mv "$OUT.tmp" "$OUT"
      echo "dispatch_workflow refs as emitted by the agent: $BEFORE"
      echo "All dispatch_workflow items now target: $HEAD_REF"
      finish_ok=1

safe-outputs:
  mentions: false
  add-comment:
    # Exactly one report per invocation: what was selected, why, and links to the runs.
    max: 1
    hide-older-comments: true
  dispatch-workflow:
    # COMPILE-TIME ALLOWLIST of every pipeline `/test` may launch. The compiler verifies
    # each of these exists and declares `workflow_dispatch`; a typo or a renamed pipeline
    # is a build error rather than a runtime surprise. Entries are bare filename stems.
    #
    # This is deliberately the set of *optional* pipelines only. `pr-gate` and
    # `merge-gate` are absent by design: they run on their own and re-running them from
    # here would duplicate work the PR already does.
    workflows:
      - sanity-tests
      - blackhole-e2e-tests

      - galaxy-profiler-tests
      - galaxy-multi-user-isolation-tests
      - galaxy-unit-tests
      - galaxy-integration-tests
      - galaxy-stress-tests
      - galaxy-e2e-tests
      - galaxy-sanity
      - galaxy-health

      - t3000-e2e-tests
      - t3000-integration-tests
      - t3000-profiler-tests
      - t3000-unit-tests

      - single-card-profiler-tests
      - pipeline-select-profiler

      - models-t1-e2e-tests
      - models-t1-unit-tests
      - models-t2-e2e-tests
      - models-t2-unit-tests
      - models-t3-e2e-tests
      - models-t3-unit-tests

      - perf-device-models
      - tt-metal-l2-nightly
      - ttnn-run-sweeps
      - vllm-model-tests
      - metal-run-microbenchmarks

      - runtime-sanity-tests
      - runtime-unit-tests
      - runtime-integration-tests
      - runtime-perf-tests
    # REQUIRED for the PR-branch targeting this whole workflow exists to provide, and
    # the reason it cannot be narrower than "any branch":
    #
    # `/test` arrives as an `issue_comment` event. That event sets no `GITHUB_HEAD_REF`,
    # so gh-aw's ref-resolution chain (message.ref > target-ref > GITHUB_HEAD_REF >
    # GITHUB_REF) falls all the way through to `GITHUB_REF` — which on an issue_comment
    # is `refs/heads/main`. Every dispatch would silently test `main` instead of the PR.
    # The agent must therefore pass `ref` explicitly per call, and per-call refs are
    # refused outright unless `allowed-refs` is set. `target-ref` cannot substitute: it
    # is a single static string, and the correct branch differs on every invocation.
    #
    # The pattern is `**` (which normalizes to `refs/heads/**`) rather than `*`, because
    # in path-mode globbing `*` does not cross `/` — `refs/heads/*` would match `main`
    # and `some-branch` but NOT `user/my-feature`, which is the shape of most tt-metal
    # PR branches. `**` also excludes `refs/tags/*` for free, so a tag can never be
    # dispatched.
    #
    # Residual risk, assessed and accepted: this glob permits any branch, `main`
    # included, so it is a namespace restriction rather than an enforcement of "PR
    # branch only". The layers that actually constrain it are: `roles` above (only
    # write-access users can invoke at all); the pre-fetch step, which resolves the
    # branch deterministically and hard-fails if it ever comes back as `main`, so the
    # agent copies a supplied value instead of inventing one; and the summary comment,
    # which prints the ref for every dispatch so a wrong target is visible and
    # cancellable within seconds. The worst outcome is optional CI running on
    # already-trusted code — every branch in this repo was pushed by someone with write
    # access — which is wasted machine time, not a correctness or security event.
    allowed-refs: ["**"]
    # Upper bound on hardware committed by a single `/test`. Chosen to fit the widest
    # legitimate fan-out — a `tt_metal/api` change plausibly wants all four `runtime-*`
    # pipelines plus `sanity-tests`; a broad `models/` change wants the six
    # `models-t{1,2,3}-{e2e,unit}` pipelines — while still capping a
    # misreasoned "run everything" at 8 rather than all 34.
    max: 8
---

# `/test` — CI pipeline selector for tt-metal

You are the `/test` agent for `${{ github.repository }}`. A developer with write access
commented `/test` on a pull request. Your job is to decide **which optional CI pipelines
this change actually needs, on which hardware**, and launch exactly those against the
PR's own branch.

You select and launch existing pipelines; you do not write tests. You do not modify the
PR, push commits, or comment on anything other than the PR that invoked you.

## What the developer asked for

The full text of the triggering comment is:

```
${{ steps.sanitized.outputs.text }}
```

If it is bare `/test`, choose entirely on your own judgement. If it carries a hint —
`/test blackhole`, `/test just galaxy demos`, `/test t3000 + profiler` — **treat that
hint as authoritative** and narrow to it. The developer knows something about their
change that the diff may not show. Only override an explicit request if it is impossible
(e.g. they named a pipeline that is not in your allowlist), and say so in your comment.

## Inputs already on disk

These were fetched deterministically before you started. **Read them; do not re-derive
them.**

| Path | Contents |
|---|---|
| `/tmp/gh-aw/agent/pr-meta.json` | PR number, title, body, head/base branch, fork flag, line counts |
| `/tmp/gh-aw/agent/pr-files.txt` | Every changed file path, one per line |
| `/tmp/gh-aw/agent/pr-files-truncated.txt` | **Only exists if the file list is incomplete.** If present, read it |
| `/tmp/gh-aw/agent/pr-diff.patch` | The diff, truncated to 3000 lines |
| `/tmp/gh-aw/agent/pr-head-ref.txt` | **The branch every dispatch must target** |
| `/tmp/gh-aw/agent/pr-is-fork.txt` | `true` if the PR comes from a fork |

## Stop condition: fork pull requests

If `pr-is-fork.txt` is `true`, **dispatch nothing**.

GitHub's `workflow_dispatch` API only accepts a ref that exists in this repository, and a
fork's head branch does not. There is no workaround: `refs/pull/<N>/head` exists here but
is not a branch, and dispatch rejects it. Dispatching anyway would either error or — worse
— fall back to testing `main`, which tells the developer nothing about their change.

Post your comment explaining this, and point them at the Actions tab to run a pipeline by
hand against a local copy of the branch if they need one. Then stop.

A deterministic post-step strips any dispatch you emit for a fork PR (your comment still
posts), so a mistake here cannot reach the dispatcher — but the comment you write must
match that reality: never describe a pipeline as dispatched on a fork PR.

## Selection procedure

1. **Read the changed-file list first.** Paths determine which subsystems and which
   silicon are reachable; the diff body only refines *how much*. A change confined to
   `docs/`, `tech_reports/`, `*.md`, or comments needs **nothing** — say so and dispatch
   zero pipelines. That is a correct and common outcome, not a failure.

   If `pr-files-truncated.txt` exists, you are reasoning from a partial list. Lean
   **wider** than the visible paths justify, and say so in your comment so the developer
   knows to check whether anything was missed.

2. **Map paths to affected hardware and subsystems** using the table below.

3. **Read the diff** for the files that matter, to judge blast radius. A one-line
   guard-clause fix in a Blackhole-only code path does not justify the Galaxy fleet. A
   change to a shared dispatch primitive does.

4. **Shortlist pipelines**, then cut. Prefer the narrowest pipeline that would actually
   catch a regression in what changed. Ask of each candidate: *if this change is broken,
   would this pipeline fail?* If you cannot answer yes, drop it.

5. **Narrow each survivor to the relevant platforms _and suites_** via its inputs (next
   section). Running `runtime-unit-tests` across every SKU when only Blackhole code
   changed wastes hours of scarce silicon — and so does running the fabric and T3000
   suites inside `sanity-tests` for a single-device op change.

   Narrowing is not only about architecture. Several pipelines bundle independent test
   suites behind their own toggles, and those default to *on*. Reach step 6 with an
   explicit answer for each survivor: which suites can this change actually break?

6. **Respect the cap of 8.** If more than 8 look justified, you are almost certainly
   being too broad — re-cut to the highest-signal ones and note in your comment what you
   left out and why, so the developer can dispatch the rest by hand.

## Pipeline catalogue

| Pipeline | Hardware | Reach for it when |
|---|---|---|
| `sanity-tests` | WH + BH + simulator | First-line signal on core `tt_metal/` or `ttnn/` changes. Bundles eight independent suites — select them, do not take the default of all eight |
| `blackhole-e2e-tests` | Blackhole (P150/P300/BH QuietBox) | Anything under a `blackhole/` path or BH-specific HAL/SoC descriptor |
| `galaxy-sanity`, `galaxy-health` | Galaxy (WH/BH) | Quick Galaxy-reachability check before committing to the heavier Galaxy suites |
| `galaxy-unit-tests`, `galaxy-integration-tests`, `galaxy-e2e-tests` | Galaxy | Fabric, CCL, multi-device, or large-mesh code paths |
| `galaxy-profiler-tests` | Galaxy | Galaxy profiler instrumentation changes |
| `galaxy-stress-tests`, `galaxy-multi-user-isolation-tests` | Galaxy | Stability, long-run, or multi-tenant isolation behaviour |
| `t3000-unit-tests`, `t3000-integration-tests`, `t3000-e2e-tests` | T3000 (8×WH) | Multi-chip work that does not need a full Galaxy |
| `t3000-profiler-tests`, `single-card-profiler-tests`, `pipeline-select-profiler` | T3K / single card / selectable | `tt_metal/tools/profiler/**`, tracy, or profiling instrumentation |
| `models-t1-*` | Selectable SKU | Tier-1 (highest-priority) model changes under `models/` |
| `models-t2-*`, `models-t3-*` | Selectable SKU | Tier-2/3 model changes |
| `perf-device-models` | Single card | Device-perf regressions from op or kernel changes |
| `tt-metal-l2-nightly` | WH + BH | Broad L2 coverage for wide-reaching `tt_metal/` changes |
| `ttnn-run-sweeps` | Selectable | `ttnn/` op changes where sweep coverage is the real signal |
| `vllm-model-tests` | Selectable SKU | vLLM serving integration |
| `metal-run-microbenchmarks` | Single card | Low-level metal performance primitives |
| `runtime-sanity-tests`, `runtime-unit-tests`, `runtime-integration-tests`, `runtime-perf-tests` | WH / BH / multichip | `tt_metal/impl/**`, `llrt/**`, `api/**`, `jit_build/**`, dispatch and runtime layers |

Path orientation: `tt_metal/hw/**` and `tt_metal/tt-llk/**` are kernel/LLK; `tt_metal/fabric/**`
and `tt_metal/distributed/**` are multi-device; `tt_metal/impl/**`, `tt_metal/llrt/**`,
`tt_metal/api/**`, `tt_metal/jit_build/**` are runtime/dispatch; `ttnn/**` is the op library;
`models/**` is model code; `tt-train/**` is training; `tests/**` is test-only (map to whichever
pipeline owns the tests being touched); `docs/**`, `tech_reports/**`, `.md` are documentation.

## Narrowing inputs

**Each dispatch tool's own input schema is authoritative.** It is generated at compile
time from that pipeline's `workflow_dispatch` block and already tells you which fields are
required, which are `enum`-constrained, and what each one defaults to. It also sets
`additionalProperties: false`, so an input name that is not in the schema is rejected
before the dispatch is attempted. Work from the schema in front of you; do not infer input
names from the pipeline's YAML on disk, because the two can disagree (see the maintenance
note below) and the schema is what validation enforces.

**Eleven pipelines have required inputs — a no-input dispatch of these will fail
validation.** You must supply at least:

| Pipeline | Must supply |
|---|---|
| `galaxy-sanity` | `arch` |
| `models-t1-e2e-tests`, `models-t1-unit-tests` | `model` |
| `models-t2-e2e-tests`, `models-t2-unit-tests` | `model` |
| `models-t3-e2e-tests`, `models-t3-unit-tests` | `model` |
| `t3000-integration-tests`, `t3000-unit-tests` | `model` |
| `vllm-model-tests` | `model` |
| `ttnn-run-sweeps` | `arch`, `log-level`, `runner-label`, `sweep_name` |

Each of these still has a sensible default in the schema — passing the default explicitly
is fine when you have no reason to narrow further. The other 23 pipelines take no required
inputs and can be dispatched bare.

The defaults are usually *maximal*, and that is where the waste is. Recurring shapes:

- **`all` defaults to `true` on the `runtime-*` pipelines.** Setting `blackhole: true`
  alone does **not** narrow anything — `all` is still true and everything runs. You must
  set `all: false` *and* the specific platform. This is the single easiest way to
  accidentally run the full matrix.
- `wormhole` / `blackhole` / `multichip` booleans select architecture on the `runtime-*`,
  `galaxy-e2e-tests`, and `galaxy-health` pipelines.
- `model` and `sku` are `choice` inputs on the `models-t*` and `vllm-model-tests`
  pipelines, both defaulting to `all`. If the change touches one model, name it. SKU
  values carry a human-readable suffix — use the option string exactly as written
  (e.g. `wh_n150 (N150)`, `bh_p150 (P150)`).
- **Suite and board toggles: `run-<something>` booleans that default to `true`.** Three
  pipelines bundle independent suites this way, and taking the defaults runs all of them:

  | Pipeline | Toggles (all default `true`) |
  |---|---|
  | `sanity-tests` | `run-ttnn-sanity-tests`, `run-ops-sanity-tests`, `run-fabric-sanity-tests`, `run-t3000-sanity-tests`, `run-umd-sanity-tests`, `run-ttsim-sanity-tests`, `run-blackhole-multi-card-sanity-tests`, `run-models-sanity-tests` |
  | `single-card-profiler-tests` | `run-n150-profiler`, `run-n300-profiler`, `run-blackhole-profiler` |
  | `pipeline-select-profiler` | `run-n150-profiler`, `run-n300-profiler`, `run-blackhole-profiler`, `run-t3k-profiler` |

  The names say what each covers, so map them the same way you mapped paths to pipelines:
  a single-device `ttnn` op change reaches `run-ttnn-sanity-tests` and `run-ops-sanity-tests`
  and does **not** reach fabric, T3000, UMD, or multi-card. Set the ones it cannot reach to
  `false`. Leaving all seven on is the same mistake as dispatching seven pipelines when one
  would do — it is just hidden inside a single dispatch.

- **Do not touch inputs that change behaviour rather than scope.** `mlperf-read-only`,
  `mlperf-write-access`, `upload_results`, `skip_on_timeout`, `build-inplace-wheel`,
  `enable-watcher`, `enable-llk-asserts`, and `run_triage_tests` are not narrowing knobs;
  flipping them changes what the run *does* or where it writes, not how much of it runs.
  Leave them alone.
- Leave `platform`, `build-type`, and `enable-lto` at their defaults unless the change is
  specifically about a build configuration.

### Consistency check before you dispatch

Read back the reason you are about to write for each pipeline. **If your reason says a
subsystem is not reachable by this change, no input you are passing may still enable it.**
Saying "nothing multi-chip or fabric is reachable here" and then dispatching with
`run-fabric-sanity-tests` and `run-t3000-sanity-tests` left at their defaults contradicts
your own analysis and spends shared silicon on it. Either narrow the inputs to match the
reason, or widen the reason to admit why you kept the suite on.

> **Maintenance note (for humans, not the agent).** Because those schemas are baked into
> `test-command.lock.yml` at compile time, they are a *snapshot*. If an allowlisted
> pipeline later adds an input, renames one, or adds a value to a `choice` list, `/test`
> cannot use the new shape until this workflow is recompiled — `additionalProperties:
> false` and the compiled `enum` lists will reject it. The failure mode is narrow and
> visible (that one dispatch is refused with a validation error, nothing else breaks), but
> it does mean **changing a dispatch input on any of the 34 pipelines requires re-running
> `gh aw compile test-command` and committing the lock file.**

## The ref rule

**Every dispatch must set `ref` to the exact contents of `/tmp/gh-aw/agent/pr-head-ref.txt`.**

This is the single most important rule in this workflow. If you omit `ref`, the dispatch
does **not** fail — it silently runs against `main`, because the `issue_comment` event
that triggered you carries no PR branch. The developer would get a green pipeline that
tested none of their code, which is worse than no result at all.

Copy the branch name from that file verbatim. Do not reconstruct it from the PR title, the
comment, or your memory of the diff. Never dispatch `main`, `master`, or a release branch.

A deterministic post-step also rewrites the `ref` of every dispatch you emit to the
contents of that file before anything is dispatched, so an omitted or mistyped `ref`
cannot actually reach `main` — but that backstop is not a reason to skip the rule. Your
summary comment quotes the ref, and it must match what actually runs.

## Reporting

Post exactly one comment. Open it with exactly this heading, verbatim:

```
### `/test` — dispatched pipelines
```

Do not reword it per run — a stable heading is what makes these comments scannable when
several land on the same PR. In particular do not describe this as "triage": that word is
heavily overloaded in this repo (CI triage, issue triage, `run_triage_tests`) and it is not
what this comment is. It is a record of what was launched.

Then keep the body short enough to read at a glance:

- **What you dispatched** — a table with exactly these three columns, one row per
  dispatched pipeline:

  | Pipeline | Inputs | Reason |
  |---|---|---|
  | the **badge**, built from the template below — *never* the bare pipeline name | `key: value, …`, or `defaults` | one line |

  **Column 1 is a badge image.** The badge already renders the pipeline's name, so a plain
  name in that column is strictly worse than a badge: it throws away the live status and
  the link, which are the entire reason the column exists. If you are about to write
  `` `sanity-tests` `` there, stop — you have dropped the badge; build it from the template
  under *Status badges* instead.

  **Column 2 is the exact inputs you supplied**, verbatim as `key: value`, comma-separated,
  in a code span — not a prose summary like "Blackhole only". A reader has no other way to
  find out what a dispatched run was scoped to: `workflow_dispatch` inputs are not shown on
  the run page, so if this cell does not say it, the information is gone. Where you
  deliberately left a suite off, that `false` is the most useful thing in the row — it is
  the record of a decision, and the reviewer's chance to catch you having narrowed too far.

  If you passed nothing, write `defaults` — and be aware that for the pipelines with suite
  toggles above, `defaults` means *everything*, which is rarely what you intended.
- **The ref** every dispatch targeted, stated explicitly so it is auditable.
- **What you deliberately skipped** and why, when a reader might expect it — especially
  anything you dropped to stay under the cap of 8.
- If you dispatched **nothing**, say so plainly and give the reason (docs-only change,
  fork PR, nothing reachable by the optional pipelines). Do not pad it. No badges in that
  case — a badge for a pipeline you did not launch is worse than no badge.

### Status badges

The badge **goes in column 1 of the table above** — it is not a separate section, and not a
caption. Build one per dispatched pipeline, linked to that pipeline's runs filtered to this
branch, substituting the workflow's **filename** and the branch from `pr-head-ref.txt`:

```
[![](${{ github.server_url }}/${{ github.repository }}/actions/workflows/<file>.yaml/badge.svg?branch=<branch>)](${{ github.server_url }}/${{ github.repository }}/actions/workflows/<file>.yaml?query=branch:<branch>)
```

This is the same badge-plus-filtered-link convention `pr-description-inject-branch-name.yaml`
already uses for the always-on pipelines, so the two read as one system. The badge image is
re-rendered by GitHub on every page load, so it reports the *live* status of that pipeline
on this branch — queued, passing, or failing — and keeps doing so long after this comment
is written. Clicking it lands on that pipeline's runs for this branch.

Both values are known to you right now: the filename is the allowlist entry plus `.yaml`,
and the branch is the file you already read. Nothing here needs a run ID.

**Do not state a run ID, a run URL, or a start time.** You are writing this comment
*before* any dispatch has happened — your comment and your dispatches are both outputs of
this turn and are only carried out afterwards, so no run exists yet and no ID has been
assigned. Anything in that shape would be invented. The badge is what makes that
unnecessary: it resolves the run on the reader's behalf, later, from facts you do have.

For the same reason a badge may read **"no status"** for the first minute or so, before the
run is created — and if that pipeline ran on this branch previously, it will briefly show
that older result instead. Add one line under the table saying badges go live shortly after
posting, so nobody reads a cold badge as a failure to launch.

That caveat line is a caption for the badges. **Only include it if the table actually
contains badges** — printing "badges go live shortly" above a table of plain pipeline names
tells the reader to wait for something that is never going to appear, and is how a dropped
badge column disguises itself as a slow one.

Close by noting that these are optional pipelines: they do not gate the PR, and a failure
here means the change needs another look, not that the PR is blocked from merging.
