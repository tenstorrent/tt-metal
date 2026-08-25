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
  group: "gh-aw-${{ github.workflow }}-${{ github.event.issue.number || github.event.pull_request.number }}"

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
      PR_NUMBER: ${{ github.event.issue.number || github.event.pull_request.number || fromJSON(github.event.inputs.aw_context || github.event.client_payload.aw_context || '{}').item_number }}
      EXPR_GITHUB_REPOSITORY: ${{ github.repository }}
      PR_DIFF_MAX_LINES: "3000"
    run: |
      set -euo pipefail
      mkdir -p /tmp/gh-aw/agent

      gh pr view "$PR_NUMBER" --repo "$EXPR_GITHUB_REPOSITORY" \
        --json number,title,body,headRefName,headRefOid,baseRefName,isCrossRepository,headRepositoryOwner,additions,deletions,changedFiles \
        > /tmp/gh-aw/agent/pr-meta.json

      # Full changed-file list, unabridged. Pipeline selection is driven far more by
      # WHICH paths changed than by the contents of the hunks, so this list must never
      # be truncated even when the diff below is.
      gh pr view "$PR_NUMBER" --repo "$EXPR_GITHUB_REPOSITORY" \
        --json files --jq '.files[].path' > /tmp/gh-aw/agent/pr-files.txt

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

      echo "PR #${PR_NUMBER}: head=${HEAD_REF} fork=${IS_FORK} files=$(wc -l < /tmp/gh-aw/agent/pr-files.txt) diff_lines=$(wc -l < /tmp/gh-aw/agent/pr-diff.patch)"

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
      - galaxy-deepseek-tests
      - galaxy-perf-tests
      - galaxy-demo-tests
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

You are a triage agent, not a test author. You do not modify the PR, push commits, or
comment on anything other than the PR that invoked you.

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
| `/tmp/gh-aw/agent/pr-files.txt` | Every changed file path, one per line (complete, never truncated) |
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

## Selection procedure

1. **Read the changed-file list first.** Paths determine which subsystems and which
   silicon are reachable; the diff body only refines *how much*. A change confined to
   `docs/`, `tech_reports/`, `*.md`, or comments needs **nothing** — say so and dispatch
   zero pipelines. That is a correct and common outcome, not a failure.

2. **Map paths to affected hardware and subsystems** using the table below.

3. **Read the diff** for the files that matter, to judge blast radius. A one-line
   guard-clause fix in a Blackhole-only code path does not justify the Galaxy fleet. A
   change to a shared dispatch primitive does.

4. **Shortlist pipelines**, then cut. Prefer the narrowest pipeline that would actually
   catch a regression in what changed. Ask of each candidate: *if this change is broken,
   would this pipeline fail?* If you cannot answer yes, drop it.

5. **Narrow each survivor to the relevant platforms** via its inputs (next section).
   Running `runtime-unit-tests` across every SKU when only Blackhole code changed wastes
   hours of scarce silicon.

6. **Respect the cap of 8.** If more than 8 look justified, you are almost certainly
   being too broad — re-cut to the highest-signal ones and note in your comment what you
   left out and why, so the developer can dispatch the rest by hand.

## Pipeline catalogue

| Pipeline | Hardware | Reach for it when |
|---|---|---|
| `sanity-tests` | WH + BH + simulator | Broad, cheap first signal on core `tt_metal/` or `ttnn/` changes |
| `blackhole-e2e-tests` | Blackhole (P150/P300/BH QuietBox) | Anything under a `blackhole/` path or BH-specific HAL/SoC descriptor |
| `galaxy-sanity`, `galaxy-health` | Galaxy (WH/BH) | Quick Galaxy-reachability check before committing to the heavier Galaxy suites |
| `galaxy-unit-tests`, `galaxy-integration-tests`, `galaxy-e2e-tests` | Galaxy | Fabric, CCL, multi-device, or large-mesh code paths |
| `galaxy-demo-tests`, `galaxy-deepseek-tests` | Galaxy | Model-level Galaxy demos; DeepSeek-specific model code |
| `galaxy-perf-tests`, `galaxy-profiler-tests` | Galaxy | Galaxy performance or profiler instrumentation changes |
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

Every pipeline in the allowlist has **only optional** `workflow_dispatch` inputs, so
dispatching with no inputs is always safe and runs that pipeline's defaults. But the
defaults are usually *maximal*, and that is where the waste is.

**Before dispatching a pipeline, read its `on.workflow_dispatch.inputs` block** from
`.github/workflows/<name>.yaml` and pick the narrowing inputs that fit the change. Read
the file rather than guessing — these schemas change, and an invalid `choice` value makes
the dispatch fail outright.

Recurring shapes:

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
- Leave `platform`, `build-type`, and `enable-lto` at their defaults unless the change is
  specifically about a build configuration.

## The ref rule

**Every dispatch must set `ref` to the exact contents of `/tmp/gh-aw/agent/pr-head-ref.txt`.**

This is the single most important rule in this workflow. If you omit `ref`, the dispatch
does **not** fail — it silently runs against `main`, because the `issue_comment` event
that triggered you carries no PR branch. The developer would get a green pipeline that
tested none of their code, which is worse than no result at all.

Copy the branch name from that file verbatim. Do not reconstruct it from the PR title, the
comment, or your memory of the diff. Never dispatch `main`, `master`, or a release branch.

## Reporting

Post exactly one comment. Keep it short enough to read at a glance:

- **What you dispatched** — a table of pipeline, the platform narrowing you applied, and a
  one-line reason. Link each dispatched run.
- **The ref** every dispatch targeted, stated explicitly so it is auditable.
- **What you deliberately skipped** and why, when a reader might expect it — especially
  anything you dropped to stay under the cap of 8.
- If you dispatched **nothing**, say so plainly and give the reason (docs-only change,
  fork PR, nothing reachable by the optional pipelines). Do not pad it.

Close by noting that these are optional pipelines: they do not gate the PR, and a failure
here means the change needs another look, not that the PR is blocked from merging.
