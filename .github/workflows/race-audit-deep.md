---
description: |
  On-demand / deeper LLK race-hazard sweep, triggered by commenting `/race-audit`
  on a pull request (or via workflow_dispatch).

  Same skill as the automatic `race-audit.md` workflow
  (tt_metal/tt-llk/.claude/skills/race-audit-all), but this is the ONLY path
  permitted to use the skill's exhaustive Workflow tier — the multi-phase
  enumerate -> deep audit -> adversarial verify -> reconcile -> closer-loop ->
  thin-grounding -> synthesize pipeline with its file-manifest + coverage ledger.
  That tier is too slow and too expensive to run automatically on every push, but
  it is the only mode that can actually satisfy the skill's "no cap without a
  closer" rule, so it stays available behind an explicit human request.

  ADVISORY ONLY, exactly like the automatic workflow: COMMENT reviews only, never
  REQUEST_CHANGES, never a required status check.

on:
  # NOTE: gh-aw v0.84.0 rejects `slash_command` in the same workflow as
  # `pull_request`, which is why this is a separate file from `race-audit.md`
  # rather than a second trigger on it. `slash_command` + `workflow_dispatch` is a
  # permitted combination (the same pairing `repo-assist.md` uses).
  slash_command:
    name: race-audit
    events: [pull_request_comment, pull_request_review_comment]
  workflow_dispatch:
    inputs:
      exhaustive:
        description: "Run the skill's exhaustive Workflow tier (slower, far more expensive) instead of the default parallel tier"
        required: false
        type: boolean
        default: false
  reaction: "eyes"

concurrency:
  # Not `cancel-in-progress` (unlike the automatic workflow): a human explicitly
  # asked for this run, so a subsequent push should not silently kill it.
  group: race-audit-deep-${{ github.event.issue.number }}
  cancel-in-progress: false

# The exhaustive tier loops gap-closers until the coverage ledger is clear, so it
# needs materially more headroom than the automatic sweep's 90 minutes. This is a
# ceiling, not a target; the run must still state its bound if it hits the budget.
timeout-minutes: 240

permissions:
  contents: read
  pull-requests: read

# See race-audit.md for the rationale: `engine: claude` because `race-audit-all` is
# a Claude Code skill whose default execution tier is the engine's native
# concurrent `Agent` fan-out. Reuses the existing `LLK_PR_REVIEW_API_KEY` secret.
engine:
  id: claude
  env:
    ANTHROPIC_API_KEY: ${{ secrets.LLK_PR_REVIEW_API_KEY }}

# `awmgmcpg` is gh-aw's own internal MCP Gateway sidecar hostname — benign, not a
# real missing dependency, and not silenceable via `network.allowed` at this
# compiler version. Handled at the instruction level; see the guidelines in the
# body and the fuller NOTE in `race-audit.md`.
#
# Declared `mcp-servers` hosts are auto-added to the compiled firewall allowlist.
network: defaults

tools:
  github:
    toolsets: [pull_requests, repos, context]
    lockdown: false
    min-integrity: none
  bash: true
  cache-memory: true

mcp-servers:
  # Mirrors `tt_metal/tt-llk/.mcp.json`, same set as `race-audit.md`. The same
  # caveat applies: `atlassian` and `glean_default` carry no credentials here, so
  # authenticated Confluence/Glean access is unverified and the skill will mark the
  # affected coverage bounded rather than guess.
  atlassian:
    url: "https://mcp.atlassian.com/v1/mcp"
  deepwiki:
    url: "https://mcp.deepwiki.com/mcp"
    allowed:
      - read_wiki_structure
      - read_wiki_contents
      - ask_question
  glean_default:
    url: "https://tenstorrent-be.glean.com/mcp/default"

safe-outputs:
  mentions: false
  create-pull-request-review-comment:
    # Higher than the automatic sweep's 20: an exhaustive run legitimately produces
    # more, and this path was explicitly requested by a human.
    max: 30
    side: "RIGHT"
  submit-pull-request-review:
    max: 1
    # ADVISORY ONLY. Compiler-enforced, not just prompt-enforced.
    allowed-events: [COMMENT]
  messages:
    footer: "> 🔎 *Deep race audit by [{workflow_name}]({run_url}) — advisory only, never blocking.*"
    run-started: "🔎 [{workflow_name}]({run_url}) is running an on-demand LLK race-hazard sweep. The exhaustive tier can take a long time — I'll report back here."
    run-success: "🔎 [{workflow_name}]({run_url}) finished the sweep — see the inline comments and summary review for findings."
    run-failure: "🔎 [{workflow_name}]({run_url}) {status}. Will need another look."
---

# LLK Race Audit — on-demand / deep (tt-metal)

You are running the repository's own **`race-audit-all`** LLK hazard sweep against a
pull request, because a human explicitly asked for it.

## Current Context

- **Repository**: `${{ github.repository }}`
- **Pull Request**: #${{ github.event.issue.number }}
- **Request text**: "${{ steps.sanitized.outputs.text }}"

## Step 1: Load the skill

Read, in full, before anything else:

```
tt_metal/tt-llk/.claude/skills/race-audit-all/SKILL.md
```

It is the authority for this run. Then read the SKILL.md of each per-class
sub-audit you need, from the sibling directories under
`tt_metal/tt-llk/.claude/skills/`: `mmio-race-audit`, `reconfig-stall-audit`,
`cfg-word-overlap-audit`, `semaphore-handshake-audit`, `mailbox-sync-audit`,
`dataflow-cb-sync-audit`, `srcreg-bank-sync-audit`, `noc-sync-audit`,
`instruction-latency-audit` (plus `arch-lookup`).

Execute the skills faithfully — do not reimplement them from memory or from this
file's summary.

## Step 2: Read the request and choose the tier

The request text above is the human's instruction. Interpret it:

- **Exhaustive requested** — the text asks for an "exhaustive", "no-skip", "full",
  "deep", or "multi-agent" run (or `workflow_dispatch` was invoked with
  `exhaustive: true`): run the skill's **exhaustive Workflow tier** — phases 0-6,
  with the mandatory **file manifest + coverage ledger**, the adversarial verify
  phase, and the **closer loop until the ledger is clear**. Every in-scope file must
  end in exactly one of `audited` / `abstained` / `out-of-scope`; the run may not
  report "done" while any in-scope file is `not-opened`. If the 240-minute budget is
  hit first, stop and state the explicit residual count — that makes it a *bounded*
  run, not an exhaustive one, and it must be labelled as such.
- **Scope narrowed** — the text names specific classes, files, or an arch (e.g.
  "just the noc and dataflow classes", "Quasar only"): honour that scope, and say in
  the summary that the scope was user-narrowed.
- **Otherwise** — run the same **default parallel tier** as the automatic workflow:
  concurrent `Agent` fan-out per `(class, arch, file-group)`, JOIN inline.

State plainly at the top of the summary which tier you ran and why.

Unless the request says otherwise, scope to the PR's changed files plus whatever the
method requires you to open to judge them. If an exhaustive run is requested, the
declared scope boundary must be made explicit and the ledger must prove nothing
inside it was skipped.

## Step 3: Source preflight — run it, log it, DO NOT PAUSE (override)

Identical override to the automatic workflow, and for the same reason: a
`/race-audit` comment is a single-shot request, not an interactive session. Nobody
is waiting to answer a follow-up question, so a pause just burns the timeout.

1. Run the skill's preflight and build the source reachability table
   (tt-isa-docs MCP, DeepWiki MCP, Atlassian/Confluence MCP, Glean, pinned
   `sfpi-gcc`, dataflow API source, in-repo `tt_llk_*` code).
2. **Auto-choose "proceed with the reachable set".** Never block.
3. Reproduce the table and the per-source consequences ("Confluence unreachable ->
   Quasar HW = `[code-only]`", "sfpi-gcc unfetchable -> latency abstains") in the
   summary review body.

All other grounding rules remain binding: **ground-or-abstain**, **never infer a
negative from a missing doc** (use `UNCERTAIN — needs HW/owner confirmation`), and
**confirm reachability before flagging a value-gated race** (program factory and
device-op validation invariants).

Note that if the human's request was specifically to reach a source that the
automatic run could not (for example "re-run with Confluence"), and it is still
unreachable, say so explicitly and prominently — that is the answer to their
question.

## Step 4: Map findings to review comments

Preserve the skill's **monotonic contract**: the JOIN may only add or escalate,
never silently drop or downgrade a per-audit verdict.

**Inline review comments** (`create-pull-request-review-comment`, max 30) — one per
finding at its `file:line`, `RIGHT` side:

- Lead with the verdict tag: `EMERGENT-RACE`, `RACE`, `LATENT`, `HARDENING-GAP`,
  `INIT-BUG`, `UNCERTAIN`, `ANNOTATED-SAFE`.
- Name the class(es) and arch.
- For `EMERGENT-RACE`, give the full cross-class chain: resource, composed
  guarantees, and the gap.
- State the grounding source and revision, or mark it `[code-only]`.
- Give a concrete fix or the specific confirmation needed.

A comment must land on a line present in the PR diff; if the true site is outside
the diff, put it in the summary with its `file:line` spelled out rather than forcing
it onto an unrelated line. If you exceed 30, prioritise `EMERGENT-RACE` > `RACE` >
`INIT-BUG` > `LATENT` > `HARDENING-GAP` > `UNCERTAIN` > `ANNOTATED-SAFE`, report the
remainder in the summary, and state how many were not inlined.

**Summary review** (`submit-pull-request-review`, exactly one, event `COMMENT`):

- Which tier ran, and the declared scope boundary.
- Per-verdict, per-class totals plus the emergent count.
- The cross-reference worklist: every "safe because <other class>" clause and
  whether it was discharged.
- The source reachability table and every coverage bound.
- **For an exhaustive run**: the coverage ledger tallies
  (`audited` / `abstained` / `out-of-scope`) and **0** `not-opened` — or, if the
  budget was hit, the explicit residual count, marking the run *bounded*.
- Any findings that could not be inlined.
- The plain statement that no per-class finding was dropped or downgraded.
- A one-line note that this is advisory and does not block the merge.

## Step 5: Dedup against the automatic run

Use cache memory at `/tmp/gh-aw/cache-memory/`. The automatic `race-audit` workflow
writes `pr-<number>.json`; read it so you do not simply repeat what it already
posted. Write your own state to `pr-<number>-deep.json`.

Focus your inline comments on findings that are **new relative to the automatic
sweep**, and in the summary say how many previously-reported findings this deeper
run **confirmed**, **refuted**, or **escalated**. Refuting or escalating an earlier
finding is one of the most valuable things this deeper tier produces — report it
explicitly. Reconcile monotonically, per the skill's rules: every prior finding maps
to confirmed / refuted / false-positive-correct / not-found.

## Guidelines

- **Advisory, never blocking.** `COMMENT` reviews only. Never `REQUEST_CHANGES`,
  never `APPROVE`.
- **Never forward firewall boilerplate into what you post.** Do not copy any
  `⚠️ Firewall blocked …` block — in particular the benign `awmgmcpg` MCP-gateway
  notice, which is gh-aw's own internal sidecar hostname and not a real missing
  dependency — into a review comment or review body. Strip it from anything public.
- **Ground or abstain**, and record the source and revision behind every verdict.
- **A missing doc page is "undocumented", never "absent" or "unordered".** Use
  `UNCERTAIN — needs HW/owner confirmation` rather than manufacturing a confirmed
  race; a refuted confirmed-flag trains reviewers to dismiss the auditor.
- **An exhaustive run that skipped files is a bounded run.** Say so. Do not let
  "sampled N of M" read as complete.
- **Comment on the code, not the author.**
- **Identify yourself as automation** in the summary review.
- **Do not modify the repository.** Read-only job; all writes go through the declared
  safe outputs.
