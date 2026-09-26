---
name: bringup
description: Interview the user about a model to bring up (prefill on Tenstorrent), write and get approval for its spec, launch the gated bring-up framework in models/demos/common/bringup, and supervise the run. Use when someone wants to bring up a new model, asks how a bring-up is going, or wants to pause, resume, rerun or fork one.
---

# /bringup: intake and supervision for the gated bring-up framework

You interview the person, write the model spec, get it approved, launch the orchestrator, and supervise the run. You
do not implement the model yourself; agents do, one gated step at a time. Framework: `models/demos/common/bringup/`
(README.md there). A worked run: `models/demos/gemma4_a4b_d_p/bringup/` (spec, ledger, BREADCRUMBS.md, supervision.md).

Always: `export PYTHONPATH=$PWD` (the shell's default points at another checkout); `B="python -m models.demos.common.bringup"`.

## 1. Interview

Ask in one message for what you cannot find out yourself, with defaults proposed:
- **Model**: HF id. You will pin its revision (commit sha).
- **Target**: context length and chunk size (default 56320 tokens in 5120-token chunks).
- **Ladder**: default four rungs: 2 chunks of 2048 with full dumps, 2 chunks of 8192, the last chunk after a golden
  prefix, then the full target.
- **Layer subset**, only if the model does not fit the box.
- **Owner rules** for every agent, e.g. "always use 2D fabric". They go in `agents.rules` and appear in every brief.
- **Dashboard style**: `standard` (the ERNIE look), `teletext` (a 90s teletext service on a CRT), or `both`. Goes in
  `dashboard.styles`.
- **Retry policy**, only if they want to change it: per role, attempts (default 3) and whether it escalates to
  `ttnn-expert-debugger` (only implement and device fixes do; that agent is for TTNN ops, never CPU code).

## 2. Check on the spot and report

- Checkpoint: reachable, size, gated or not; it downloads to `/localdev/$USER/bringup/<model>/hf/`
  (`snapshot_download(<id>, local_dir=...)`). The download metadata records the revision; put it in `hf.revision`.
- Box: `ls /dev/tenstorrent | wc -l` chips; `tt-smi -ls` for the architecture. Never `tt-smi -r`.
- Config (the text config for multimodal checkpoints): layers, hidden, heads, KV heads, head_dim, experts, attention
  type per layer, sliding window. These give `block_types` (one entry per distinct block, every layer exactly once, a
  representative layer each) and `checkpoint.expect` / `checkpoint.config`.
- Base or instruction-tuned: a chat template plus `-it` / `-Instruct` in the name means instruction-tuned. That decides
  the input wrap: `raw` for base checkpoints, `model_turn` for instruction-tuned ones (the book becomes the model's
  reply to a short request; an -it model fed raw text is not predicting anything it was trained on).
- The model card's usage example: becomes `intake.smoke: {prompt, expect}` (e.g. "capital of France" -> "Paris").
- What the repo already has for this family (grep `models/demos`, the repo map); list it for the planner.

## 3. Write the spec, get it approved, launch

1. `$B new --model <slug> --hf-id <id>`, then fill `models/demos/<slug>/bringup/spec.yaml` (the template comments say
   what each field is): the interview answers, `hf.revision`, `hf.parity_seq` longer than any sliding window,
   `text.wrap`, `intake.smoke`, `agents.rules`, `agents.read` (HF modeling file for the reference role, similar repo
   models for plan and implement), `dashboard.styles`.
2. Validate: `python -c "from models.demos.common.bringup.core.spec import Spec; print(Spec.load('<spec>').validate())"`.
3. Show the spec. Wait for an explicit yes. Then `$B approve intake --spec <spec>`.
4. `python -m models.demos.common.bringup.plan.ledger_gen --spec <spec> --early --write` and `$B init-run run1 --spec <spec>`.
5. Launch: `python -u -m models.demos.common.bringup.orchestrator run --spec <spec> >> $ART/runs/run1/orchestrator.log 2>&1`
   as a background process, and watch the log (step 4). The first task, R.1, builds the canonical prompt and runs the
   HF sanity gate (revision, usage-example smoke, next-token accuracy floor).
6. Publish the dashboard(s): `python -m models.demos.common.bringup.dashboard.export --spec <spec>` writes
   `<bringup_dir>/dashboard/index.html` and/or `teletext.html`; publish them as artifacts and give the links.

## 4. Supervise the run

Watch the orchestrator log for gate results, attempts, stops and problems. After every gate, re-export and republish
the dashboards. Classify every failure before acting, and log every intervention in `<bringup_dir>/supervision.md`
(time, task, trigger, classification, action, resulting commit):

| Failure | Action |
|---|---|
| the model's code (a gate misses a threshold) | none: the retry loop and the brief handle it |
| the check before an agent starts (no implementation yet) | none: expected |
| the framework (a false violation, a file not committed, a wrong rule) | pause, fix with a selftest, gate the fix in `dev/`, resume |
| the box (device open fails, ethernet or fabric timeouts) | the orchestrator stops by itself; ask the person to reset the board, then a quick box check, then resume |
| the input or data (implausible accuracy, degenerate generations) | stop, show the evidence, ask the person |
| an approval point (plan, performance picks) | bring the person the plan or list; record their decision with `$B approve` |

Never:
- edit anything in the tree while an agent step runs. `python -m models.demos.common.bringup.orchestrator pause --spec <spec>`
  stops it before its next task; the path check would charge your edit to the running agent;
- approve for the person, loosen a threshold, or edit a frozen test;
- change shared code or the spec without asking (a spec edit voids the intake approval; re-approve on their word);
- run `tt-smi -r`, or use long timeouts for a device check (the box test takes seconds).

Report briefly on each gate the person would care about; say plainly when you stopped something and why.

## 5. Resume, rerun, fork

- `python -m models.demos.common.bringup.orchestrator resume --spec <spec>` (after a stop, a pause, or a fix).
- `$B approve plan|perf --spec <spec>`; `$B status --spec <spec>`.
- `$B rerun --from <id> --spec <spec>`; `$B fork --from <id> --name <run> --spec <spec>`;
  `$B compare --spec <specA> --other <specB>`.
