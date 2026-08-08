---
description: |
  Automated LLK race/hazard sweep on pull requests. Runs the repo-local
  `race-audit-all` Claude Code skill (tt_metal/tt-llk/.claude/skills/race-audit-all)
  over the PR diff and posts its findings as inline review comments plus a single
  summary review.

  The skill orchestrates nine per-class LLK hazard audits — mmio-race,
  reconfig-stall, cfg-word-overlap, semaphore-handshake, mailbox-sync,
  dataflow-cb-sync, srcreg-bank-sync, noc-sync, instruction-latency — across four
  synchronization surfaces (cross-thread, RISC<->Tensix, cross-core/NoC,
  intra-thread micro-architectural), then adds a cross-class JOIN pass that
  discharges "safe because <invariant owned by another audit>" cross-references and
  surfaces EMERGENT-RACE findings no single audit can see.

  ADVISORY ONLY. This workflow never blocks a merge: it posts COMMENT reviews and
  never REQUEST_CHANGES, and it must not be configured as a required status check.
  The skill's stated posture is to over-report every suspicion and never suppress
  for uncertainty, so false positives are expected by design.

on:
  # Auto-run on PRs that touch LLK code or the dataflow API the skill's
  # dataflow-cb/noc classes ground against.
  #
  # NOTE: gh-aw v0.84.0 rejects `slash_command` in the same workflow as
  # `pull_request` ("cannot use 'slash_command' with 'pull_request' in the same
  # workflow"). The on-demand / deeper-pass path therefore lives in the sibling
  # workflow `race-audit-deep.md` (`/race-audit` + workflow_dispatch), which is the
  # only place the skill's exhaustive Workflow tier may be used.
  pull_request:
    types: [opened, synchronize, reopened]
    paths:
      - "tt_metal/tt-llk/**"
      - "tt_metal/hw/inc/api/dataflow/**"

concurrency:
  # A new push supersedes an in-flight audit of a now-stale diff. Findings are
  # keyed to file:line, so finishing an audit of an outdated diff would place
  # comments on lines that no longer exist.
  group: race-audit-${{ github.event.pull_request.number }}
  cancel-in-progress: true

# Matches the real-world budget of the existing manual `llk-pr-review.yaml`
# (timeout-minutes: 80). The default parallel tier fans out concurrent Agent calls
# per (class, arch, file-group) and saturates a ~10-16 concurrency cap, so a sweep
# is a substantial run, not a lightweight lint.
timeout-minutes: 90

permissions:
  contents: read
  pull-requests: read

# `engine: claude` is required here (the repo's other gh-aw workflows use
# `copilot`): `race-audit-all` is a Claude Code skill, and the sweep depends on the
# engine's native concurrent `Agent` fan-out, which is the skill's default
# execution tier.
#
# Reuses the existing `LLK_PR_REVIEW_API_KEY` secret already provisioned for
# `llk-pr-review.yaml` rather than adding a second Anthropic key to the repo. The
# claude engine reads `ANTHROPIC_API_KEY`, so it is mapped here; the compiled
# manifest records only `LLK_PR_REVIEW_API_KEY`.
engine:
  id: claude
  env:
    ANTHROPIC_API_KEY: ${{ secrets.LLK_PR_REVIEW_API_KEY }}

# NOTE on the `awmgmcpg` firewall warning: `awmgmcpg` is gh-aw's own internal MCP
# Gateway sidecar hostname (image github/gh-aw-mcpg, container awmg-mcpg), flagged
# by gh-aw's own firewall. It is benign and not a real missing external dependency.
# It cannot be silenced via `network.allowed` — the compiler (v0.84.0) rejects a
# bare `awmgmcpg` token (not a valid ecosystem id and no dot), and the gateway's
# real transport `host.docker.internal` is already in the `defaults` allowlist, so
# allowlisting changes nothing. It is handled at the instruction level instead —
# see "Never forward firewall boilerplate into what you post" in the body.
#
# The three `mcp-servers` domains below do NOT need to be listed here: gh-aw
# automatically adds declared MCP server hosts to the compiled firewall allowlist.
network: defaults

tools:
  github:
    # Only what a PR review needs: read the PR, its files and diff, and run context.
    toolsets: [pull_requests, repos, context]
    # tt-metal is public; `lockdown: false` lets the agent read PR content from
    # third-party contributors, which is exactly the code it must audit.
    lockdown: false
    min-integrity: none
  # Required: the skill's method is deterministic enumeration (glob the in-scope
  # trees, grep for primitives, resolve macros/wrappers, follow the call graph) and
  # it resolves the pinned sfpi-gcc commit via `gh`.
  bash: true
  # Dedup across re-runs. `synchronize` re-fires on every push, and without memory
  # the agent would re-post the same inline findings on each one.
  cache-memory: true

mcp-servers:
  # The same three servers the existing `llk-pr-review.yaml` already relies on for
  # LLK grounding, mirrored from `tt_metal/tt-llk/.mcp.json` (gh-aw declares MCP
  # servers in frontmatter rather than via an `--mcp-config` file path). Together
  # they cover the skill's ground-truth ladder: DeepWiki and tt-isa-docs serve the
  # same `tenstorrent/tt-isa-documentation` corpus for WH/BH ISA semantics, and
  # Atlassian reaches the Quasar Confluence HW pages.
  #
  # CAVEAT: neither `atlassian` nor `glean_default` is given credentials here, and
  # gh-aw's compiled job has no interactive OAuth step, so authenticated
  # Confluence/Glean access is UNVERIFIED in this automated context. This is not a
  # silent failure: the skill's ground-or-abstain contract means it will label the
  # affected coverage bounded (e.g. `[Quasar: code-only]`) rather than fabricate a
  # verdict. See the PR description for the Tailscale/egress discussion.
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
    # Nine sub-audits plus a JOIN pass routinely surface more than
    # grumpy-reviewer's 5. Still capped so a pathological run cannot bury the PR.
    max: 20
    side: "RIGHT"
  submit-pull-request-review:
    max: 1
    # ADVISORY ONLY — the hard guarantee that this workflow can never gate a merge.
    # `COMMENT` is the only permitted event; APPROVE and REQUEST_CHANGES are
    # rejected by the compiler-generated output handler, not merely discouraged in
    # the prompt. Do not add REQUEST_CHANGES here, and do not make this a required
    # status check.
    allowed-events: [COMMENT]
  messages:
    footer: "> 🔎 *Race audit by [{workflow_name}]({run_url}) — advisory only, never blocking.*"
    run-started: "🔎 [{workflow_name}]({run_url}) is running the LLK race-hazard sweep on this PR..."
    run-success: "🔎 [{workflow_name}]({run_url}) finished the sweep — see the inline comments and summary review for findings."
    run-failure: "🔎 [{workflow_name}]({run_url}) {status}. Will need another look."
---

# LLK Race Audit (tt-metal)

You are running the repository's own **`race-audit-all`** LLK hazard sweep against a
pull request, and reporting what it finds as PR review comments.

## Current Context

- **Repository**: `${{ github.repository }}`
- **Pull Request**: #${{ github.event.pull_request.number }}
- **Head SHA**: `${{ github.event.pull_request.head.sha }}`
- **Base SHA**: `${{ github.event.pull_request.base.sha }}`

## Step 1: Load the skill

The skill is checked out with the repository. Read it in full before doing anything
else:

```
tt_metal/tt-llk/.claude/skills/race-audit-all/SKILL.md
```

It is the authority for this run — its method, its monotonic contract, its
ground-truth source ladder, and its output format all govern what you do. Follow it
as written, except where this workflow explicitly overrides it below.

Then read the SKILL.md of each per-class sub-audit you actually need. They live as
sibling directories under `tt_metal/tt-llk/.claude/skills/`:

`mmio-race-audit`, `reconfig-stall-audit`, `cfg-word-overlap-audit`,
`semaphore-handshake-audit`, `mailbox-sync-audit`, `dataflow-cb-sync-audit`,
`srcreg-bank-sync-audit`, `noc-sync-audit`, `instruction-latency-audit`
(plus `arch-lookup` for arch resolution).

Do **not** reimplement the audits from memory or from this file's summary — read the
skills and execute them faithfully. Do not paste whole SKILL.md files into your
findings.

## Step 2: Scope to the PR diff

Get the PR's changed files and diff for `${{ github.repository }}` PR
#${{ github.event.pull_request.number }}. Scope the sweep to **the code this PR
touches**, plus whatever the skill's method requires you to open in order to judge
those changes (callers, the other thread's writer of a shared CONFIG word, the
matching semaphore post, the program factory that supplies a value-gated bound, and
so on). Following a hazard out of the diff is correct and expected; auditing
unrelated trees is not.

Declare the scope boundary explicitly in your summary so a reader can see where the
sweep stopped.

## Step 3: Tier — default parallel only (MANDATORY)

Run the skill's **default (parallel, non-exhaustive) tier**: concurrent `Agent`
fan-out per `(class, arch, file-group)`, then the JOIN inline.

**Do NOT escalate to the exhaustive Workflow tier in this workflow.** The multi-phase
exhaustive pipeline (enumerate -> deep audit -> adversarial verify -> reconcile ->
closer-loop -> thin-grounding pass -> synthesize) with its loop-until-dry closer
phase cannot fit the 90-minute budget on an automatic per-PR trigger, and its cost
is not appropriate as a routine CI gate. The exhaustive tier is reserved for the
sibling `race-audit-deep` workflow, where a human has explicitly asked for it.

Because this is a bounded run, apply the skill's own rule honestly: a bounded sweep
must never read as exhaustive. State the bound.

## Step 4: Source preflight — run it, log it, DO NOT PAUSE (override)

The skill instructs you to emit a source-manifest preflight and then **pause and ask
the user** to choose between proceeding, reaching a missing source, or adding a
source. **That interactive pause is overridden here.** There is no user attached to
a CI run; waiting would hang the job until the timeout and post nothing.

Instead:

1. Run the preflight exactly as specified — probe each source in the ladder
   (tt-isa-docs MCP, DeepWiki MCP, Atlassian/Confluence MCP, Glean, the pinned
   `sfpi-gcc` source, the dataflow API source, in-repo `tt_llk_*` code) and build
   the reachability table.
2. **Auto-choose option (a): proceed with the reachable set.** Never block.
3. Reproduce the reachability table and the per-source consequence list ("Confluence
   unreachable -> Quasar HW = `[code-only]`", "sfpi-gcc unfetchable -> latency
   abstains", ...) in the summary review body, so the bound is visible to reviewers
   rather than buried in the run log.

Everything else about grounding is unchanged and still binding: **ground-or-abstain**
(if no applicable authority is reachable, emit no verdict and label the coverage
hole — never substitute a weaker basis), **never infer a negative from a missing
doc** (emit `UNCERTAIN — needs HW/owner confirmation` instead of a confirmed race),
and **confirm reachability before flagging a value-gated race** (check the program
factory and device-op validation invariants).

Expect Atlassian/Confluence and Glean to be unauthenticated in this environment. If
they are unreachable, that is a coverage bound to report, not a reason to guess and
not a reason to fail the run.

## Step 5: Map findings to review comments

Preserve the skill's **monotonic contract**: the JOIN may only add findings or
escalate severity; it must never silently drop or downgrade a per-audit verdict. The
mapping below is a reporting format, not a licence to summarize findings away.

**Inline review comments** (`create-pull-request-review-comment`, max 20) — one per
finding, anchored at its `file:line` on the `RIGHT` side of the diff:

- Lead with the verdict tag: `EMERGENT-RACE`, `RACE`, `LATENT`, `HARDENING-GAP`,
  `INIT-BUG`, `UNCERTAIN`, or `ANNOTATED-SAFE`.
- Name the class(es) that produced it and the arch it applies to.
- For an `EMERGENT-RACE`, give the full cross-class chain: the physical resource, the
  composed guarantees, and exactly where the gap is.
- State the grounding: which source and revision the verdict rests on (ISA-doc page,
  DeepWiki, Confluence page id + date, `sfpi-gcc` commit) — or say plainly that it is
  `[code-only]`.
- Give a concrete fix or the specific confirmation needed.
- Keep each comment tight. Detail belongs in the summary, not in twenty long threads.

A comment must land on a line that exists in this PR's diff. If a finding's true site
is outside the diff, do **not** force it onto an unrelated line — put it in the
summary review body with its `file:line` written out.

**Prioritise if you exceed 20.** Order: `EMERGENT-RACE` > `RACE` > `INIT-BUG` >
`LATENT` > `HARDENING-GAP` > `UNCERTAIN` > `ANNOTATED-SAFE`. Then report the
remainder in the summary and say how many were not inlined — do not silently drop
them. Dropping a finding for space is a reporting bound and must be stated.

**Summary review** (`submit-pull-request-review`, exactly one, event `COMMENT`):

- Per-verdict, per-class totals plus the emergent count.
- The cross-reference worklist: every "safe because <other class>" clause and whether
  it was discharged.
- The source reachability table and every coverage bound, including the explicit
  statement that this was the **default parallel tier, not an exhaustive run**.
- Any findings that could not be inlined.
- The plain statement the skill requires: that no per-class finding was dropped or
  downgraded.
- A one-line note that this is advisory and does not block the merge, and that
  `/race-audit` can be used for a deeper on-demand pass.

If the sweep finds nothing, still submit the summary review saying so, with the
coverage bounds — a clean result is only meaningful alongside its scope.

## Step 6: Dedup

Use cache memory at `/tmp/gh-aw/cache-memory/` keyed on this PR
(`pr-${{ github.event.pull_request.number }}.json`). Record the head SHA and a
stable fingerprint of each finding you post (class + file + line + verdict).

On a re-run for a new push, only post findings that are **new or changed** relative
to what you already posted, and note in the summary how many previously-reported
findings still stand. Do not re-post an identical inline comment on every push.

## Guidelines

- **Advisory, never blocking.** Only ever submit a `COMMENT` review. Never
  `REQUEST_CHANGES`, never `APPROVE`. The skill deliberately over-reports and never
  suppresses for uncertainty, so false positives are expected — gating merges on that
  would be wrong. Say so in the summary.
- **Never forward firewall boilerplate into what you post.** Do not copy any
  `⚠️ Firewall blocked …` block — in particular the benign `awmgmcpg` MCP-gateway
  notice — into a review comment or review body. `awmgmcpg` is gh-aw's own internal
  MCP Gateway sidecar hostname, not a real missing dependency, and it cannot be
  silenced via `network.allowed` at this compiler version (see the NOTE in the
  frontmatter). Treat any such block as internal-only noise and strip it from
  anything posted publicly.
- **Ground or abstain.** A verdict with no reachable authority is a labelled coverage
  hole, not a guess. Record the source and revision behind every verdict.
- **A missing doc page is "undocumented", never "absent" or "unordered".** Do not turn
  doc-silence into a confirmed `RACE` — and equally not into `SAFE`. Use `UNCERTAIN —
  needs HW/owner confirmation`. A fabricated mechanism filed as confirmed is
  anti-conservative: each refuted flag trains reviewers to dismiss the auditor.
- **Comment on the code, not the author.** Be specific, technical, and neutral.
- **Identify yourself as automation** in the summary review, and note that findings
  need maintainer confirmation.
- **Do not modify the repository.** This job is read-only; every write goes through
  the declared safe outputs.
