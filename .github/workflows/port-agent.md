---
description: |
  Drives the tt-dm-codegen porting pipeline for a single op from one place: sync a shipped port to a
  newer generator revision, verify it on real hardware, and report the results onto its PR.

  The agent reasons, routes, and writes; it never touches a card. All measurement happens in
  `port-device.yaml`, which holds the hardware and runs the pipeline. This workflow decides what to
  run, reads back what happened, and turns it into PR comments and charts.

  Staged by default: every run prints what it would post and writes nothing until `staged: false`.

on:
  workflow_dispatch:
    inputs:
      # --- identity -------------------------------------------------------------------------
      op:
        description: "op name, e.g. pad"
        required: true
        type: string
      mode:
        description: "what to do"
        required: true
        type: choice
        default: sync
        options: [sync, verify, port, improve, upstream, collect]
      pr:
        description: "tt-metal PR number this port lives on (its body holds the results block)"
        required: false
        type: string
        default: ""
      run_id:
        description: "pipeline run id — reuse one to continue a port, blank to start a new one"
        required: false
        type: string
        default: ""
      # --- sync -----------------------------------------------------------------------------
      codegen_ref:
        description: "tt-dm-codegen ref to sync from"
        required: false
        type: string
        default: "main"
      sync_apply:
        description: "report only, apply onto a branch, or apply and open a PR"
        required: false
        type: choice
        default: report
        options: [report, apply, pr]
      in_place:
        description: "commit onto the PR's own branch instead of stacking a sync branch"
        required: false
        type: boolean
        default: false
      no_translate:
        description: "copy kernels only; do not regenerate the host C++"
        required: false
        type: boolean
        default: false
      rescope:
        description: "re-run intake scoping for this op"
        required: false
        type: boolean
        default: false
      avoid_syncing:
        description: "declared divergence, '<paths>: <why>' per line — reported, never overwritten"
        required: false
        type: string
        default: ""
      # --- verify ---------------------------------------------------------------------------
      verify:
        description: "where to verify"
        required: false
        type: choice
        default: none
        options: [none, local, cross-arch, both]
      archs:
        description: "comma-separated archs, e.g. wormhole_b0,blackhole"
        required: false
        type: string
        default: "wormhole_b0"
      band:
        description: "which gate to run"
        required: false
        type: choice
        default: both
        options: [both, correctness, performance]
      post_charts:
        description: "post perf charts to the PR"
        required: false
        type: boolean
        default: true
      chart_format:
        description: "markdown bars, uploaded PNGs, or both"
        required: false
        type: choice
        default: both
        options: [mermaid, png, both]
      # --- upstream (accepted, not yet wired) -----------------------------------------------
      improve_codegen:
        description: "NOT YET WIRED — propose fixes back to tt-dm-codegen"
        required: false
        type: boolean
        default: false
      upstream_mode:
        description: "NOT YET WIRED — how to raise findings upstream"
        required: false
        type: choice
        default: none
        options: [none, issue, pr]
      # --- safety and budget ----------------------------------------------------------------
      staged:
        description: "print what would be posted and write nothing (default: true)"
        required: false
        type: boolean
        default: true
      auto_ship:
        description: "let the pipeline push and open/update the PR itself"
        required: false
        type: boolean
        default: false
      max_tokens:
        description: "pipeline token ceiling"
        required: false
        type: string
        default: ""
      max_wall_clock_s:
        description: "pipeline wall-clock ceiling in seconds"
        required: false
        type: string
        default: ""
      model:
        description: "model for the pipeline's own LLM phases"
        required: false
        type: string
        default: ""
      extra_flags:
        description: "extra run.py flags (allowlisted; anything else fails the run)"
        required: false
        type: string
        default: ""

timeout-minutes: 45

# Read-only, and gh-aw enforces it: the agent job may not hold a write scope. `actions: read` is
# what lets it poll device runs and read their logs; dispatching one is a *write*, so it goes
# through the `dispatch-workflow` safe output below rather than a token this job holds.
permissions:
  contents: read
  issues: read
  pull-requests: read
  actions: read
  # Not a repo write: it bills inference to the Actions token, as repo-assist does, rather than
  # requiring a personal COPILOT_GITHUB_TOKEN.
  copilot-requests: write

network: defaults

engine: copilot

tools:
  github:
    toolsets: [actions, repos, issues, pull_requests, search, context]
    lockdown: false
  # A port is a chain of runs across days. Memory is what makes the second run a continuation:
  # which run_id belongs to which PR, which device runs are still in flight, what was already said.
  repo-memory: true
  bash: true

safe-outputs:
  mentions: false
  add-comment:
    max: 5
    target: "*"
  # The only way this agent starts a device run. It cannot hold `actions: write` itself, so the
  # dispatch is brokered here and the allowlist is what stops it from triggering anything else in
  # a repo whose workflows include full release pipelines.
  dispatch-workflow:
    workflows: [port-device]
    max: 2
  # Charts must be hosted to be visible: a PR comment pointing at a runner-local path renders as a
  # broken image for every human who reads it. `assets/` prefix is required by gh-aw for new branches.
  upload-asset:
    branch: "assets/codegen-port-charts"
    max: 10
    allowed-exts: [.png]
  create-issue:
    title-prefix: "[codegen-port] "
    labels: [automation]
    max: 2

# Deterministic pre-work. Routing is computed here, not inferred by the model: which mode maps to
# which device command is a fixed table, and a model that occasionally picks `apply` when the
# operator said `report` would write to a branch nobody asked it to touch.
#
# The decision reaches the agent as a file, not as `${{ steps.route.outputs.* }}` in the prompt:
# these steps run in the agent job, after the activation job has already rendered the prompt, so
# any step output interpolated above would render empty. gh-aw rejects that at compile time.
#
# A non-zero exit here fails the job before the agent starts, which is what makes the validation
# below a gate rather than a suggestion.
steps:
  - name: Route and validate
    id: route
    env:
      MODE: ${{ inputs.mode }}
      VERIFY: ${{ inputs.verify }}
      SYNC_APPLY: ${{ inputs.sync_apply }}
      STAGED: ${{ inputs.staged }}
      EXTRA_FLAGS: ${{ inputs.extra_flags }}
      ARCHS: ${{ inputs.archs }}
      OP: ${{ inputs.op }}
      RUN_ID: ${{ inputs.run_id }}
      IMPROVE: ${{ inputs.improve_codegen }}
      UPSTREAM_MODE: ${{ inputs.upstream_mode }}
      AUTO_SHIP: ${{ inputs.auto_ship }}
    run: |
      mkdir -p /tmp/gh-aw/agent
      python3 - <<'PY'
      import json, os, re, shlex, sys, time

      def fail(msg):
          sys.exit(f"::error::{msg}")

      mode = os.environ["MODE"]
      op = os.environ["OP"].strip()
      if not re.fullmatch(r"[a-z][a-z0-9_]*", op):
          fail(f"op {op!r} must be a bare lowercase identifier — it becomes a path and a C++ filename")

      # v1 scope. These fail loudly rather than no-oping: an operator who asked for an upstream PR
      # and got a silent success would believe one exists.
      unwired = []
      if mode in ("improve", "upstream"):
          unwired.append(f"mode={mode}")
      if os.environ.get("IMPROVE") == "true":
          unwired.append("improve_codegen=true")
      if os.environ.get("UPSTREAM_MODE", "none") != "none":
          unwired.append(f"upstream_mode={os.environ['UPSTREAM_MODE']}")
      if unwired:
          fail(f"not wired yet: {', '.join(unwired)}. The inputs exist so the surface is stable, "
               "but nothing implements them. Use mode=sync/verify/port/collect.")

      staged = os.environ["STAGED"] == "true"
      sync_apply = os.environ["SYNC_APPLY"]
      if staged and sync_apply != "report":
          fail(f"sync_apply={sync_apply} writes, but staged=true. Set staged=false to allow writes.")

      for arch in [a.strip() for a in os.environ["ARCHS"].split(",") if a.strip()]:
          if arch not in ("wormhole_b0", "blackhole"):
              fail(f"unknown arch {arch!r}")

      BOOL = {"--allow-narrow", "--skip-intake", "--refresh-manifest", "--update", "--rescope",
              "--no-cross-arch-sweep", "--collect-cross-arch", "--render-png-charts"}
      VALUED = {"--stack-on", "--from-main", "--manifest", "--baseline-ref", "--max-tokens",
                "--max-wall-clock-s", "--phase-timeout-s", "--max-phases", "--build-timeout-s",
                "--review-verdict", "--ttmetal-base-ref", "--claude-model"}
      raw = os.environ.get("EXTRA_FLAGS", "").strip()
      if raw:
          try:
              tokens = shlex.split(raw)
          except ValueError as e:
              fail(f"extra_flags is not parseable as shell words: {e}")
          expect_value = False
          for token in tokens:
              if expect_value:
                  expect_value = False
              elif token in BOOL:
                  pass
              elif token in VALUED:
                  expect_value = True
              elif token.split("=", 1)[0] in VALUED:
                  pass
              else:
                  fail(f"refusing extra_flags token {token!r}: not in the allowlist "
                       f"({', '.join(sorted(BOOL | VALUED))})")
          if expect_value:
              fail(f"trailing flag {tokens[-1]!r} expects a value")

      # The fixed mode -> device-command table.
      device_mode = {"sync": "sync", "verify": "verify", "port": "port", "collect": ""}[mode]
      needs_device = bool(device_mode) and not (mode == "verify" and os.environ["VERIFY"] == "none")

      # `port-device` has one `staged` switch but it gates two different writes: `--apply` in sync
      # mode and `--auto-ship` in verify/port mode. Un-staging to get one would silently enable the
      # other, so the second opt-in is folded in here and the device job is handed a single
      # already-decided boolean. Writing is then the conjunction of two explicit consents.
      auto_ship = os.environ.get("AUTO_SHIP") == "true"
      if mode == "sync":
          device_staged = staged or sync_apply == "report"
      else:
          device_staged = staged or not auto_ship

      routing = {
          "op": op,
          "mode": mode,
          "device_mode": device_mode,
          "needs_device": needs_device,
          # Pass THIS to port-device's `staged` input, not `staged` below.
          "device_staged": device_staged,
          # A generated id must be stable for the whole port, so it is echoed back in the report
          # for the operator to pass to the next run. Losing it orphans the state artifact.
          "run_id": os.environ["RUN_ID"].strip() or f"ci-{op}-{int(time.time())}",
          "run_id_was_generated": not os.environ["RUN_ID"].strip(),
          "staged": staged,
          "sync_apply": sync_apply,
          "archs": [a.strip() for a in os.environ["ARCHS"].split(",") if a.strip()],
      }
      with open("/tmp/gh-aw/agent/routing.json", "w") as fh:
          json.dump(routing, fh, indent=2)
      print(json.dumps(routing, indent=2))
      PY
---

# Codegen port agent

You drive the tt-dm-codegen porting pipeline for `${{ inputs.op }}` in `${{ github.repository }}`.

**Routing is already decided.** Before you do anything else, read `/tmp/gh-aw/agent/routing.json`. It holds:

| field | meaning |
| --- | --- |
| `op` | the op being ported |
| `device_mode` | which command the device workflow should run |
| `needs_device` | whether to dispatch a device run at all |
| `run_id` | the pipeline run id keying this port's state artifact |
| `run_id_was_generated` | true if it was minted this run — if so, surface it so the operator can reuse it |
| `staged` | whether **you** may write (post comments, upload assets, open issues) |
| `device_staged` | what to pass as `port-device`'s `staged` input — **not** `staged` |
| `sync_apply`, `archs` | validated pass-throughs |

These come from a fixed table and already-validated inputs. **Use them as given.** Do not re-derive routing from the prose below, and do not substitute your own judgement about what the operator "really meant" — that is exactly how a `report` becomes an `apply` against a branch nobody asked you to touch.

If the file is missing, stop and say so rather than guessing; its absence means the validation gate did not run.

## What this pipeline is

`tt-dm-codegen` is a private generator that emits data-movement kernels. A *port* takes one op's generated implementation into tt-metal as a real `ProgramFactory`, behind a routing predicate that falls back to the native path. Once ported, the two repos drift: the generator keeps improving, and the ported copy does not follow on its own.

That is what you are here for. A **sync** brings a shipped port up to a newer generator revision. A **verify** re-measures it on real hardware. Neither is safe to assume: a sync that builds is not a sync that computes the right answer, and only a device run settles it.

## The staged contract

`staged` is the difference between a dry run and a real one, and it is **true unless the operator set it false**.

When staged, you may read anything and dispatch device runs — those measure and never write to the PR. You may **not** post comments, upload assets, open issues, or let the pipeline push. Instead, print to the run log exactly what you *would* have posted, in full, so the operator can review the real thing rather than a description of it.

The device workflow enforces its own half of this (it withholds `--auto-ship` and `--apply`), so a staged run cannot write even if you ask it to. Do not treat that as licence to ask.

## How you reach things

Your `bash` sandbox has **no GitHub credentials**, and `tt-dm-codegen` is private — you cannot clone it. So:

- **GitHub reads** (PR bodies, comments, workflow runs, logs, artifacts) → the `github` MCP tool.
- **GitHub writes** → safe-outputs only.
- **tt-dm-codegen** → you never read it directly. The device workflow checks it out with a scoped token; anything you need to know about the generator side comes back in that run's log and artifacts.
- **bash** → local filesystem only: reading the tt-metal checkout, parsing JSON you have downloaded.

## Step 1 — read the port's current state

Read memory first (see **Memory**), then reconcile it against reality; memory can be stale.

If `pr` is set, fetch the PR body with the `github` tool and find the `codegen-port-results` block — an HTML comment carrying a JSON payload. It is the port's durable record: the op, its arch, its pins, its per-arch performance, and its `avoid_syncing` divergence registry. Everything you report should be consistent with it.

**Accept `v1` or `v2`.** Ports shipped before the registry existed carry `v1`, which has no `avoid_syncing` key; the pipeline reads both and rewrites the block as `v2` the next time it ships. A `v1` block is a normal older port, not a problem to report — treat a missing `avoid_syncing` as an empty registry and carry on.

If the PR body has no results block at all, say so and stop. Without it there is no port to sync — you would be guessing at what was ever measured.

## Step 2 — dispatch the device run

Only when `needs_device` is `true`. Use the **`dispatch_workflow` safe output** targeting `port-device` — you do not have permission to dispatch it any other way, and the `github` tool will refuse. Pass:

- `op`, `run_id` (from routing above), `codegen_ref`, `avoid_syncing`, `extra_flags` — straight through
- `mode` — the computed device command
- `arch` — the first entry in `archs`
- `commit` — the PR's head SHA when `pr` is set, so the run measures the tree under review rather than `main`
- `staged` — pass **`device_staged`** here, not `staged`. They differ on purpose: the device job's one switch gates both `--apply` and `--auto-ship`, and routing has already folded the operator's separate consents into the right answer.
- `state_from_run_id` — the device run id recorded in memory for this `run_id`, if any, so the run continues rather than restarts

Record the dispatched run id in memory **immediately**, before doing anything else. A build plus a device sweep takes hours — far longer than your 45-minute budget — so you will not see it finish. If you lose the id, the next run cannot find the work and will start it again from scratch.

Then **stop waiting**. Report that the run is queued, with its link, and end. Checking on it is the next run's job.

## Step 3 — collect a finished run

On a later run (or `mode: collect`), for each device run id in memory:

1. Get the run's status with the `github` tool. Still going? Leave the memory entry and move on.
2. Finished? Download its artifacts: `codegen-port-state-<op>-<run_id>` (the pipeline state JSON), `codegen-port-charts-...` (PNGs), and the logs.
3. Parse the state file. What matters: `measurements.perf` per arch, `measurements.correctness`, `flags.verdict`, `spend`, and `phase_status` for where it stopped.

If the run **failed**, read the failing job's log and say why in plain language — which phase, and what it hit. A failure with a real explanation is worth more than a success you cannot account for. Do not guess a cause you have not read.

## Step 4 — report

Compose a comment for the PR (post it only when not staged; otherwise print it).

Lead with the outcome: did the sync change anything, did it still pass, did performance move. Then the detail:

- **Sync** — what was copied, what was re-translated, and what was left alone because `avoid_syncing` declared it. Call out the declared divergences explicitly: an exemption nobody can see is an exemption nobody will ever retire.
- **Correctness** — pass/fail counts per arch. Never describe a sync as verified without one.
- **Performance** — device time versus native, per arch. Every ratio must come from the state file. Do not restate numbers already in the PR body as if this run measured them.
- **Charts** — when `post_charts` is true. `chart_format` chooses: `mermaid` (markdown, always renders), `png` (upload the artifact PNGs via the `upload-assets` safe output and embed the returned URLs), or `both`. PNGs must be uploaded to be visible — a local path in a comment is unreadable to everyone but the runner that wrote it.

Say what was **not** established as plainly as what was. A green build proves compilation, not numerics. A single-arch run says nothing about the other arch.

## Memory

Keep, per op:

- `run_id`, the PR it belongs to, and the codegen ref it was last synced to
- device run ids dispatched, with status and what they were for
- what you have already posted, so a re-run does not repeat itself
- the `avoid_syncing` registry last seen in the PR body

Read at the start, write at the end, and re-verify against live state before acting on any of it.

## Guidelines

- **Measure, do not assert.** Every number comes from a state file this pipeline produced. You have no card and cannot reproduce anything yourself.
- **Staged means silent.** Print, do not post, unless `staged` is false.
- **Never bypass routing.** The pre-step's decisions are final.
- **Declared divergence is intentional.** Never propose "fixing" a path listed in `avoid_syncing`; it differs on purpose. Report it, and say why it is listed.
- **One comment per run.** Fold everything into a single message rather than a series.
- **Identify yourself** as the codegen port agent, with 🤖, on anything posted.
- **Do not open PRs against tt-metal.** Only the pipeline pushes port branches, and only when explicitly told to.
