# PR #46283 evaluation — full record

Everything from driving tenstorrent/tt-metal PR #46283 (`tt_hw_planner` + `perf_automation`)
end to end against Voxtral-TTS (4B, three stacks) on a single p150b, and comparing the result
against our hand-written TTNN port.

Start here if picking this up cold.

## Read in this order

| file | what it is |
|---|---|
| `../PRSuggestions.md` | **the deliverable.** 8 points, 12 optimizations. This is what goes to the PR author. |
| `../PRSuggestions.txt` | same words, plain text. **Source of truth** — the `.md` is generated from it, so edit the `.txt` and regenerate. |
| `../TOOL_FINDINGS.md` | ~5600 lines, 48 findings with evidence, plus a Corrections section listing claims we retracted. The receipts for anything in the deliverable. |
| `logs/RUN_PLAN.md` | stage-by-stage state of the run, the commands used, and the resolver test |
| `../PR46283_SUGGESTIONS.md` | an earlier, longer PR-facing draft. Superseded by `PRSuggestions`. |

## What is here

    logs/                 run logs for bring-up, emit-e2e, optimize; PCC verification;
                          ms/frame measurements before and after; the optimizer session ledger
    logs/prior_run_state/ state captured from the first (aborted) run, including the overlay
                          store — which is itself the evidence for the overlay-store finding.
                          The unrelated minimax_h3 patches in there are what the store held.
    tracy/                the two large profiler logs, gzipped
    resolver_test/        the loader-resolver experiment (see below)

Logs are named `.log.txt` rather than `.log` because the repo's `.gitignore` drops `*.log`.
Two large logs are gzipped. Nothing here is silently excluded — verified with `git check-ignore`.

## The resolver test, in short

The tool can write its own PyTorch reference for a model `transformers` cannot load. We tested it
twice:

- `loader_run1_adapter/_reference_loader.py` — 729 lines, written with our reference reachable.
  It adapted ours, and wrote itself a self-check that passed (`selfcheck.log.txt`).
- `loader_run2_from_scratch/_reference_loader.py` — 1064 lines, written with **every** PyTorch
  implementation on the machine hidden (`isolated_run.sh` does the hiding and restores via an EXIT
  trap). It implemented all three stacks itself, importing only torch/json/math/os/safetensors.

`test_iso_loader.py` is the correctness test we wrote for run 2, since it emitted no self-check of
its own. Result, against our restored reference on the same weights: **7/7**, embedding exact,
backbone PCC 0.99999988, codec PCC 1.0, and the flow stage's 37 audio codes bit-identical.

Not included: `native_iso/`, the 7.5 GB Mistral-native checkpoint. Reproduce by pointing
`isolated_run.sh` at a local copy.

## Resuming

Everything needed to pick this up is committed. The chat transcripts are deliberately **not** in the
repo — they live only at `/localdev/lserbedzija/pr46283_evidence/session_transcripts/` on this
machine, and nothing here depends on them.

To resume a session in this project:

    cd /localdev/lserbedzija && claude --resume

Run it from `/localdev/lserbedzija` or the picker will not list the sessions. If that fails, read
the table above in order; the deliverable and the findings doc together carry every conclusion.

## Repo layout note

This checkout is a **git worktree** of `/localdev/lserbedzija/repos/tt-metal`, so it shares one
object store and history with the hand-port. Branch here: `lserbedzija/pr46283-findings`.
The hand-port's cross-comparison work is on `lserbedzija/voxtral-pr46283-xfer`.

## Deliberate omissions in the deliverable — do not "restore" these as oversights

- No listening-pass recommendation. An automated pipeline cannot perform one.
- The measurement traps (predictor scores being uncalibrated so only paired deltas mean anything;
  short prompts being seed noise; the recogniser's 30 s window silently truncating; MCD failing its
  own self-test) are held back for a follow-up. They are in `TOOL_FINDINGS.md` and in the hand-port's
  `STATUS.md` §6.59.
- No closing or contact line, by choice.
