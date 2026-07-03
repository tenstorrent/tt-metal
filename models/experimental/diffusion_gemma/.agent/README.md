# DiffusionGemma Bring-up Agents (Claude Code)

A **DiffusionGemma-specific** set of Claude Code skills, slash commands, and gate scripts. It
points Claude Code at the in-repo DiffusionGemma module (`models/experimental/diffusion_gemma/`)
and drives the remaining bring-up work — correctness, long-context fit, precision, perf, and
block-granular vLLM serving on QB2.

Originally forked from a generic *autoregressive* LLM bring-up pipeline, then adapted twice: for
the **text-diffusion** path, and (this directory) for **Claude Code** conventions instead of Codex.
The keystone is the `diffusion-gemma` skill (`skills/diffusion-gemma/SKILL.md`) — load it first in
every stage; it is the authority that maps each generic autoregressive assumption onto the diffusion
path.

```text
models/experimental/diffusion_gemma/.agent/
  skills/                     # Claude Code skills (SKILL.md each) — the engineering knowledge
    diffusion-gemma/          #   keystone: shared model context + hard rules (load first)
  commands/                   # 10 stage slash-commands (/dg-01-… … /dg-10-…)
  scripts/                    # deterministic gate scripts + the context-contract checker
```

There are no Codex artifacts here: no `agents/openai.yaml`, no `multigoal` runner, no
`$skill`/`/goal` syntax, no 4000-char objective cap. Skills are referenced by name (Claude Code
auto-invokes them from their `description`, or you type `/name`); stages are ordinary slash commands.

## Wiring it into Claude Code

These files live with the module, so they travel with it — but Claude Code only auto-discovers
skills/commands under `.claude/`. Expose them once (project-level) with symlinks from the repo root:

```bash
cd <repo-root>
mkdir -p .claude/skills .claude/commands
for d in models/experimental/diffusion_gemma/.agent/skills/*/; do
  ln -sfn "../../$d" ".claude/skills/$(basename "$d")"
done
for f in models/experimental/diffusion_gemma/.agent/commands/*.md; do
  ln -sfn "../../$f" ".claude/commands/$(basename "$f")"
done
```

(Copy instead of symlink if you prefer, or point `~/.claude/skills` at them for personal use.)
After that, `/help` lists the `dg-*` commands and the skills auto-trigger on their descriptions.

## This is not a greenfield bring-up

The DiffusionGemma module is **already substantially built and has RUN on QB2**. Foundation, the
KV-phase machine, bidirectional masked SDPA (incl. the >32768 long-prompt path), on-device canvas
sampling, and the decode-loop control flow are built and validated; the first real-26B multi-block
prompt→text run passed on QB2. The stages are therefore **validate / harden / optimize / integrate
the existing code**, ordered by the issue-map roadmap — not "author a decoder from scratch". Read
`models/experimental/diffusion_gemma/plan.md` Part 0 for live status.

## The two hard rules (read before anything)

1. **Never edit the shared Gemma-4 backbone.** The backbone is `models/demos/gemma4/`, reused
   as-is. Every DiffusionGemma fix stays inside `models/experimental/diffusion_gemma/`. The gate
   `scripts/check_no_shared_gemma4_edits.sh` enforces this (set `DG_BASE_REF` to the ref the work
   branched from — `main` on a main-based dev branch).
2. **Correctness is judged on diffusion decisions, not teacher-forcing top-k.** No AIME24 top-1/5.
   Validate entropy values, Gumbel-max argmax agreement, and accept/renoise agreement vs the torch
   reference, with the torch run's Gumbel noise injected for token-exact comparison. See the
   `diffusion-gemma` skill.

## The stages

Each stage is a slash command; run them in order (or jump to the one you need).

| Command | Delivers | Issues |
|---|---|---|
| `/dg-01-backbone-parity` | Reused gemma4 backbone validated against the DiffusionGemma checkpoint (weight mapping + causal PCC on QB2) | #47461 #47468 |
| `/dg-02-diffusion-delta` | Net-new device pieces validated: bidirectional attention, three-phase KV, self-conditioning, canvas sampling | #47462 #47474 #47472 |
| `/dg-03-denoise-loop` | Discrete-diffusion decode loop proven correct and trace-safe (fixed step budget, on-device cutoff mask) | #47463 |
| `/dg-04-block-diffusion-run` | End-to-end prompt→text multi-block RUN hardened on QB2 (RUN-first; degenerate output OK) | #47464 |
| `/dg-05-decision-fidelity` | The #48291 decision-fidelity bar measured and decided (engineering vs product-accept) | #48291 |
| `/dg-06-qb2-longcontext` | Full 256K memory budget + long-context fit on the (1,4) mesh, incl. canvas scratch + non-causal mask | #47487 |
| `/dg-07-datatype-sweep` | Fastest precision config that preserves the diffusion decisions | #47475 #47465 |
| `/dg-08-optimize-perf` | Denoise-step / block path optimized and traced | #47465 |
| `/dg-09-vllm-integration` | Block-granular serving through the tenstorrent/vllm TT plugin | #47466 #47488 |
| `/dg-10-release-ci` | Release handoff + tiered models-CI wiring | tti-release #47489 |

Every stage command tells Claude to use the `diffusion-gemma` skill first; hardware-facing stages
also use the `tt-device-usage` skill. Each command names the generic skill whose knowledge that
stage reuses, and that skill's "DiffusionGemma adaptation" section states the overrides.

## How the generic skills map

- **Reused as-is** (model-agnostic TTNN/infra): `autodebug`, `autotriage`, `models-ci`, `tt-lang`,
  `tt-device-usage`, and the bulk of `optimize` / `tt-enable-tracing` TTNN knowledge.
- **Lightly noted** (reused process, with a short `## DiffusionGemma note` re-pointing scope/examples
  to the diffusion path): `autofix`, `code_quality_review`, `beautify`, `qualitative-check`.
- **Adapted** (a `## DiffusionGemma adaptation` section overrides the autoregressive assumptions):
  `functional-decoder`, `full-model`, `datatype-sweep`, `multichip`, `optimize`,
  `tt-enable-tracing`, `vllm-integration`, `tti-release`, `stage-review`.
- **Not applicable**: `forge-functional-decoder` — the backbone is the hand-written gemma4 code,
  not a tt-forge emit; this pipeline does not use it.

## Context-length contract

The target is the HF-advertised 262144 (256K) context for `prompt + generated`; that is a capacity
limit, not a requirement that prompts be 256K tokens. Do not quietly reduce it. A reduction is
acceptable only for a hard physical device limit (QB2 DRAM), with a byte calculation or a failed
capacity probe as evidence — and the budget must include the per-step canvas K/V scratch and the
non-causal long-context mask buffers, not just weights + KV. The artifact is
`models/experimental/diffusion_gemma/doc/context_contract.json` (validate it with
`scripts/check_context_contract.py`); stages create/verify it. Stage evidence goes under
`models/experimental/diffusion_gemma/doc/<stage>/` (README.md + work_log.md + artifacts).

## Verification gates

There is no runner auto-chaining stages in Claude Code — run the gate scripts yourself after a
stage, or wire them as Claude Code hooks (a `Stop` / `PostToolUse` hook in `.claude/settings.json`).
They follow a simple exit-code convention: `0` pass, `1` advisory, `2` critical, `3` checker error.

- `scripts/check_no_shared_gemma4_edits.sh` — the reusable backbone-isolation gate (hard rule #1).
  Run it after any stage: `DG_BASE_REF=main bash .../scripts/check_no_shared_gemma4_edits.sh`.
- `scripts/04-block-diffusion-run.check.sh` — the RUN-stage gate; calls the shared-edits gate plus
  RUN-artifact checks.

Prefer a gate over prose — enforcement generalizes. Add diffusion-appropriate gates (e.g. a
decision-agreement floor) rather than an autoregressive degeneracy check.

## Extending it

- **New knowledge** goes in a skill (`skills/<name>/SKILL.md` with `name` + `description`
  frontmatter). Keep model facts and hard rules in the `diffusion-gemma` skill so every stage
  inherits them.
- **New stage** = a `commands/dg-<n>-<name>.md` slash command with a `description` frontmatter line.
- **New gate** = a `scripts/*.sh` with the exit-code convention above, scoped to
  `models/experimental/diffusion_gemma`. Calibrate any threshold against known-good and known-bad
  artifacts before trusting it.
