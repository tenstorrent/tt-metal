---
name: bringup
description: Interview the user about a model to bring up (prefill on Tenstorrent), write and get approval for its spec, launch the gated bring-up framework in models/demos/common/bringup, and supervise the run. Use when someone wants to bring up a new model, asks how a bring-up is going, or wants to pause, resume, rerun or fork one.
---

# /bringup: intake and supervision for the gated bring-up framework

You interview the person, write the model spec, get it approved, launch the orchestrator, and supervise the run. You
do not implement the model yourself; agents do, one gated step at a time. Framework: `models/demos/common/bringup/`
(README.md there). A worked run: `models/demos/gemma4_a4b_d_p/bringup/` (spec, ledger, BREADCRUMBS.md, supervision.md).

Always: `export PYTHONPATH=$PWD` (the shell's default points at another checkout); `B="python -m models.demos.common.bringup"`.

## The rule for a box smaller than the deployment box (read this first)

The deployment box exists for ONE thing: the scale factors. On a box with fewer chips than the deployment box, a
partial bring-up runs the SCALED model, and only the scaled model. There is no real-size track on this box.

- The factors come from the plan: each role's mesh share on this box over the deployment box's (`plan.yaml` roles,
  `plan/boxes.py`). Heads and experts are cut so that each chip of this box holds what a deployment chip holds; the
  chunk is the deployment chunk times the sequence factor, so each mesh row holds the deployment's tokens per row.
- Once the plan is approved, SG.1 cuts the checkpoint (`plan/scale.py`, `scaling.yaml`). From then on EVERYTHING on
  this box uses it: the goldens (made from the cut checkpoint, after SG.1), every component and swap test, the ladder,
  the contract tests, the profile and the perf picks. Fewer experts, fewer heads, the scaled chunk.
- Only these read the real checkpoint (their gates carry `BRINGUP_SCALED=0`): the reference checks against the
  model's own code (R.*) and the plan's memory check (PL.*), which is about the deployment box.
- The spec's `target` and `ladder` are written in this box's units: the scaled chunk, a context this box runs. The spec
  check refuses a target chunk that is not the scaled chunk.
- Never propose, plan or run a real-size version of the model on a smaller box. "Real model on this box" is only the
  CPU reference. A full-model run at real size happens on a box where it fits (mode `full`), not as a second track.

## 1. Interview

Ask in one message for what you cannot find out yourself, with defaults proposed:
- **Model**: HF id. You will pin its revision (commit sha).
- **Deployment box and this box**: the box the model is for (normally a Galaxy 8x4) and its context (e.g. 1M
  tokens), and the box this bring-up runs on. When they differ, the spec gets `deploy` / `boxes` / `run` (F64,
  `core/spec.py`): the plan is made for the deployment box and checked on every listed box. List every box the model
  may run on (with its mesh, DRAM per chip and any overrides, e.g. 4-bit experts on a smaller box).
- **Mode**: `partial` (the first layers that cover every layer type bring up the layout; on a box smaller than the
  deployment box the whole run is the scaled model, see the rule above) or `full` (every layer at real size, every
  end-to-end check; only on a box where the full model fits, which the plan gate checks).
- **Serving chunk**: the chunk size the server sends on the deployment box (`deploy.chunk`). Always ask; propose 4096.
  It never grows on a bigger box (more rows, fewer tokens per row) and never exceeds `serving.max_chunk` (5120, the
  owner's limit). A smaller box (the proxy) shrinks it by its sequence factor so each mesh row holds the deployment's
  tokens per row (4096 on a 4x4 -> 1024 per row -> 2048 on a 2x2) (F67). Show the per-row numbers when you ask.
- **Target**: this run's context in this box's units, in whole chunks of this box's (scaled) chunk (e.g. 55296 = 27 x
  2048 on a 2x2 standing in for a 4x4 with 4096-token chunks).
- **Ladder**: every rung at this box's chunk (no golden for a chunk this box never runs): default three rungs: 2 chunks
  with full dumps (they feed the component and swap tests), the last chunk after a golden prefix, then the full target.
  At most the canonical prompt's length. On a smaller box the goldens are the scaled model's (after SG.1). Goldens are
  made only for this box's rungs; another box's goldens come from that box's own run.
- **Layer subset**: in partial mode, the first layers that cover every layer type; in full mode, every layer.
- **Owner rules** for every agent of this model (beyond the standing ones in the agent definition, such as always
  2D fabric). They go in `agents.rules` and appear in every brief.
- **Dashboard style**: `standard` (the ERNIE look), `teletext` (a 90s teletext service on a CRT), or `both`. Goes in
  `dashboard.styles`.
- **Prior bring-up**: has this checkpoint been brought up before in this repo (another mesh or configuration)? If so,
  it becomes `prior`: the new bring-up shares its checkpoint download, goldens and CPU reference, and every step starts
  from the prior's matching files. Name the new one with its variant, e.g. `<prior>_2x2`; the prior keeps its name.
- **Framework switches**: every one, with its default, is in `models/demos/common/bringup/defaults.yaml` (component and
  swap review, retry policy, thresholds, timeouts, contract and profile options). A model overrides one in its
  spec under the same key; change the default itself only in that file.
- **Model switches**: every choice of the model (fidelities, chunk sizes, KV dtype, implementation choices, fork
  options) is a `Setting` in `models/demos/<model>/tt/settings.py` (core/model_settings.py), with its default and the
  owner decision behind it. Agents add switches only there (the orchestrator lints every agent's changes); the last
  task, Z.1, runs the `settings-audit` agent (a smaller model) to move any that ended up elsewhere. Perf picks name
  their A/B switch from that file. Component and swap tests always run at the model's max precision
  (`max_precision` in that file); a precision drop from a perf pick is accepted on the end-to-end numbers (ladder,
  contract tests), never judged by component limits.
- **tt-d-gen**: should the runner tests drive the model with tt-d-gen's real engine? Propose yes when the box is one
  tt-d-gen serves on (LoudBox, Galaxy), no otherwise (e.g. a QuietBox, where tt-d-gen does not run; the runner tests
  then use tt-metal's `prefill_producer` as a stand-in, and the dashboard says so). The answer goes in
  `serving.require_engine` (true | false). If yes: where the checkout is or should go (default
  `/localdev/$USER/tt-d-gen`), in `serving.server_repo` (leave the template's default when they take it); as soon as
  they answer, during the spec review, start the build (section 3). If no: no build. SC.1 still reads the tt-d-gen
  source (a clone is enough, no build).
- **Retry policy**, only if they want to change it: per role, attempts (default 3) and whether it escalates to
  `ttnn-expert-debugger` (only implement and device fixes do; that agent is for TTNN ops, never CPU code).

## 2. Check on the spot and report

- Checkpoint: reachable, size, gated or not; it downloads to `/localdev/$USER/bringup/<model>/hf/`
  (`snapshot_download(<id>, local_dir=...)`). The download metadata records the revision; put it in `hf.revision`.
  A layer subset still needs the whole checkpoint once, for the HF sanity (smoke, accuracy floor); task R.4 then trims
  it to layers 0..last subset layer (F47; `checkpoint.trim_drop` globs drop MTP or vision towers too, `checkpoint.trim:
  false` keeps everything). Check the free disk for the whole download; the trimmed size is what stays.
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
   With a prior bring-up: `$B new --prior <prior slug> --mesh R,C` instead. It copies the prior's spec with `prior` set
   and the new mesh, and writes hooks that reuse the prior's CPU side; the R.* tasks then pass on their first check
   and the G.* tasks reuse the prior's goldens. Review every copied field (the box name, rules, reads).
   **tt-d-gen build**, only when `serving.require_engine` is true, as soon as the path is known and in parallel with everything else: start the `dgen-build` agent
   in the background (Agent tool, `subagent_type: dgen-build`, prompt: `REPO=<path>`). It clones or pulls tt-d-gen and
   builds its engine bindings next to it (`<path>-build/`); it never touches the device or the model repo, so it runs
   alongside the spec review and the run. Skip it when `python -m models.demos.common.bringup.testing.dgen_engine
   find` already prints a build at that path's current commit. The runner tests (K.1, K.4) fail without a build
   (`serving.require_engine`, default true); check the agent's reply before K.1 and tell the owner in one line if it
   failed.
2. Validate: `python -c "from models.demos.common.bringup.core.spec import Spec; print(Spec.load('<spec>').validate())"`.
3. Show the spec. Wait for an explicit yes. Then `$B approve intake --spec <spec>`.
4. `python -m models.demos.common.bringup.plan.ledger_gen --spec <spec> --early --write` and `$B init-run run1 --spec <spec>`.
5. Launch: `python -u -m models.demos.common.bringup.orchestrator run --spec <spec> >> $ART/runs/run1/orchestrator.log 2>&1`
   from the checkout being brought up (it pins TT_METAL_HOME / PYTHONPATH / TT_METAL_CACHE / PYTHON_ENV_DIR to that
   checkout itself, F68), in a tmux session (`tmux new-session -d -s <name> "<command>"`; it survives the session; to restart, kill the
   session and start a new one: keys sent to a finished session's pane go nowhere), then arm the watch (section 4,
   "Always be watching"). The first task, R.1, builds the canonical prompt and runs the HF sanity gate (revision,
   usage-example smoke, next-token accuracy floor).
6. The serving contract comes first. Task SC.1 runs the `serving-contract` agent (`agents/serving-contract.md`): it
   reads how tt-d-gen drives a prefill model at its latest commit and writes `serving_contract.md` (a how-to for
   each part of the model) and the frozen contract tests (`contract_tests.yaml` says which step's gate runs each).
   The plan must follow it, each step's brief carries its section, and each step's gate runs its tests; the runner
   test is part of K.1. When SC.1 passes, bring the owner its `## Questions for the owner` (KV dtype for the decode
   side, slot count, prefix reuse); their answers go in the spec's `serving:` section (re-approve the intake on
   their word). Needs a tt-d-gen clone at `/localdev/$USER/tt-d-gen` (ask the owner to clone it if missing).
7. Publish the dashboard(s): `python -m models.demos.common.bringup.dashboard.export --spec <spec>` writes
   `<bringup_dir>/dashboard/index.html` and/or `teletext.html`; publish them as artifacts and give the links.

## 4. Supervise the run

Watch the orchestrator log for gate results, attempts, stops and problems. After every gate, re-export and republish
the dashboards.

### Always be watching

While a run is live, a watch is always armed: the run must never sit at a stop (an approval, a STOPPED task, a pause,
a crash) with nobody told. The one wait is

```
python -m models.demos.common.bringup.watch --spec <spec>      # Bash, run_in_background, timeout 2400000
```

It returns on the next event (gate result, a second attempt, STOPPED, WAITING FOR A PERSON, paused, DEFERRED,
KNOWLEDGE CONTRADICTED, a traceback, the end), with exit 2 after 30 minutes with nothing to act on, and with exit 3
at once when the orchestrator is not running (it reads the pid file `run` / `resume` write), printing the log's last
event. Rules:
- Re-arm it in the same turn you handle its result, every time: after an event (a DASHBOARD STALE / RED one too),
  after exit 2, after your own resume or relaunch. Never end a turn with a live run and no watch armed; never write
  your own grep loop.
- Three layers, set up by every overseer session (at launch, at resume, and first thing when a new session takes
  over a run), so no run waits on the overseer's memory (F93, owner):
  1. the watch (below);
  2. a 30-minute timer: `CronCreate` (cron `7,37 * * * *`, recurring) with the prompt `bringup heartbeat: check
     <spec>: orchestrator alive, last log line recent, the running step progressing; act on a stop; re-arm the watch;
     republish a stale dashboard`. Check `CronList` first (one per session). It fires even when no watch is armed;
     it expires after 7 days, so a longer run re-creates it;
  3. the stop guard: `~/.claude/settings.json` has the Stop hook `python3 ~/.claude/skills/bringup/heartbeat_stop.py`
     (add it if missing, `{"hooks": {"Stop": [{"hooks": [{"type": "command", "command": "...", "timeout": 30}]}]}}`).
     It refuses to let the session go idle while a run it named has no watch, or stopped without an answer.
- Heartbeat (owner, 2026-10-16): the watch is the heartbeat; never wait longer than 30 minutes (`--timeout 1800`, its
  default). On every wake, an event or not, check the run yourself before re-arming: the orchestrator is alive, the
  log's last line is recent, the running step is making progress (not the same failure again). Never sit waiting for
  a signal: a run that stopped unseen for hours is the overseer's failure (DeepSeek-V4.1: a stop at 20:33 went unseen
  because a watch that woke on DASHBOARD STALE was not re-armed).
- Exit 3 means act now: WAITING FOR A PERSON is an approval (yours if the owner delegated it, else tell the owner at
  once and say what they must decide); STOPPED and a crash are classified below. A run you paused yourself is the one
  exit 3 you expect: do the work, then resume and re-arm.
- When the person asks "status?", run the watch once (it returns at once if the run stopped) before answering. Classify every failure before acting, and log every intervention in `<bringup_dir>/supervision.md`
(time, task, trigger, classification, action, resulting commit):

| Failure | Action |
|---|---|
| the model's code (a gate misses a threshold) | none: the retry loop and the brief handle it |
| the check before an agent starts (no implementation yet) | none: expected |
| the framework (a false violation, a file not committed, a wrong rule) | pause, fix with a selftest, gate the fix in `dev/`, resume |
| the box (device open fails, ethernet or fabric timeouts) | the orchestrator stops by itself; ask the person to reset the board, then a quick box check, then resume |
| the input or data (implausible accuracy, degenerate generations) | stop, show the evidence, ask the person |
| an approval point (plan, performance picks) | bring the person the plan or list; record their decision with `$B approve`. With a deployment box, lead with plan.md's `## Boxes`: whether the full model fits the deployment box at its context, every per-chip path that differs on a smaller box, and the deployment-only checks |
| an agent needs something its step does not allow (a shared file, a wider path, a framework change) | decide yourself (below) |
| someone edits the tree while an agent step runs | the step's tree diff fails it; finish or revert the edit, then rerun the task |
| a task became DEFERRED (the agent requested an op from op-gen) | review the evidence like a gate commit; accept it and tell the owner in one line, or reject it as cheating (below); section 5 |

### Steer the agents; do not do their work

You oversee; the agents do the work. Every correction of the model's code, its knowledge files or the way a step is
done reaches the agents as a note, so the fix and its lesson land where the next step and the next model find them.

- **Feedback: a note.** `python -m models.demos.common.bringup.overseer note --spec <spec> --tasks '<glob>' [--roles
  <role>] "<what is wrong; what to do or check>"` (repeat `--tasks` / `--roles` for more than one). Every open note heads the brief of every matching step; the agent
  acts on it and answers it in its knowledge-use report (done / n/a / disagree with evidence). Your watch wakes on
  `OVERSEER NOTE <id> DISAGREED` (read the evidence: accept it, or reword the note) and on `unanswered`. Close a note
  once it is settled (`overseer close OV-n --why "..."`); `overseer list` shows them all. Notes are for things the
  agents should know or do, e.g. "the spec's scaled ladder is wrong; do not work around it", "shorten DS41-12 to 600
  characters", "check the other block sizes too", "this fork option already exists".
- **Point agents at the bring-up forks first.** Before a note or a fix job names an op, check
  `ttnn/ttnn/bringup/INDEX.md`: an existing fork (often faster and more precise) is used, extended or optimized
  before any library op is forked or composed (owner; DeepSeek-V4.1 FIX.4 was first sent to fork
  `ttnn.experimental.deepseek_prefill.mhc_split_sinkhorn` while `ttnn.bringup.mhc_pre_xing` already had an fp32
  SFPU Sinkhorn).
- **Change code that passes today: a fix job, at once.** When something must change and no failing test will start
  an agent for it (a per-chunk weight rebuild, a precision raise that a wrong limit forced, a slow path the owner
  rejects), never wait for a later step or the end of the run: `python -m models.demos.common.bringup.overseer fix
  --spec <spec> --gate <passed task> [--gate ...] "<the change, the evidence, what to measure>"`. At its next task
  boundary the orchestrator places task FIX.<n> before everything else and starts a bring-up engineer on it (no
  precheck); its gate reruns the frozen tests of every --gate task (name each passed step whose code the change
  touches, plus the swap tests through whole blocks), and a fix that stops stops the run. It keeps the plan approval.
  A note is for an agent that will run anyway; a fix job is for a change that must happen now. Queue every decision
  the moment you make it, never only in your plan or the log: one that must wait (for a framework fix you are about
  to install) is queued with `--hold "<what it waits for>"` and released (`overseer release FIX.<n>`) when that lands.
  A decision that lives only in your head is lost with the session.
- **Stop a step that goes the wrong way: abort.** `overseer abort --spec <spec> --reason "..."` ends the agent at
  work, records the attempt as aborted (no gate, not counted against the task's attempts), and turns the reason into
  a note for that task's next attempt. A running agent cannot take messages; abort plus a note is how one reaches it.
  Use it for an agent working around a spec or framework error, one in a dead end, or cheating in progress; after
  three aborts of one task it stops for a person.
- **What you change yourself.** The owner-delegated decisions (spec, plan, approvals: edit, re-approve, regenerate
  the ledger with ledger_gen); framework fixes (models/demos/common/bringup, at a pause, with a selftest, F-entry in
  dev/BREADCRUMBS.md); pause, resume, and reject (revert + note). Never the model's code, its knowledge files,
  `state.json` or `results/`: write a note and let the agent do it.
- **Your files.** `overseer.yaml`, `supervision.md` and the dashboards are yours; the agent path check ignores them, so
  write them at any time. Anything else in the tree only at a pause.

### Accuracy limits come from the noise floor, never a fixed bar

Owner rule (DeepSeek-V4.1 run1). Every accuracy limit of the component and swap tests is calib (2.0; 3.5 for the low
precision kinds) x the step's noise floor on the same comparison: the larger of the official code's own arithmetic
on the same inputs (task R.5 taps it per step; the reference role writes the hook and `official_taps.yaml`) and the
input nudge (the fp32 step on its inputs moved inside their official dtype's rounding cell; testing/noise_floor.py),
and never below one rounding of the output to its official dtype. A selection flip must be explained by the nudge. The
gated metrics are errors (rel L2 for float outputs, the missed fraction for integer ones, `err_*` `<= limit`); PCC is
recorded as a diagnostic only, being blind to a scale or an offset. After a change of limits, rerun every passed step's
gates once without touching the run (`python -m models.demos.common.bringup.core.verify --spec <spec> --tasks 'C.*'
...`) and queue a fix job for each failure.

**Numerics that cannot be fixed: leave them, with a note (owner's general guidance).** When a step misses its noise
floor and no bug or lower-precision choice explains it at the device's normal settings, do not chase it with
emulation (multi-pass or split matmuls, exact kernels, CPU-side arithmetic): no production run would ever allow
them, and a model brought up that way is useless. Leave the step as it is, with a note (a known_issues entry with
the evidence; the owner's waiver in `tests.waivers`, reported on the dashboard and in the end-of-run report), keep
the run going, and debug it once the full bring-up produces output tokens, where its real effect shows. The gate metrics read `>= limit` (the `<metric>_limit` the test
records). A fixed bar either lets real errors through or, when the reference's own dtypes cannot meet it, makes the
agents raise the device precision above the reference to clear it, which the owner rejects: match the reference's
precision, never exceed it to pass a test. When a gate fails on a fixed number, the number is the bug: fix the
framework (a pause, a selftest), then queue a fix job that takes back any precision raise it forced.

### Decide; do not ask

You are the overseer. Use your judgement and keep the run moving; the person does not want to be asked for routine
permissions. Ask the person only for: the intake spec, the plan, the performance picks, a board reset, a push or PR,
an op request's approval and the op-gen launch (section 5), and evidence that the model or its data is wrong (implausible accuracy, degenerate output). Everything else (an agent
needing a shared file, a wider allowed path, a framework fix, a rerun, a retry after a stop) you decide, do, log in
supervision.md, and report in one line.

Verify, do not ask. When you doubt an agent's change (a fork edit, an unusual precision or config choice, a claim in
a CHANGELOG or a commit message), check it yourself: pause at the next task boundary, run the fork's tests, a sweep
over the shapes the model uses (every ladder rung, other block sizes), bit-identical reruns, a comparison with the
source op; then resume. Never ask the person whether to run a check. A change that passes its gates and your checks
stands, also when it departs from the plan or from the literal wording of an owner instruction while meeting its goal
(record why); a change that breaks an explicit owner rule without the tests forcing it is rejected (below). The
person sees every such change in the end-of-run report (section 4, "At the end").

Judge every agent's work before you let it stand, most of all when a gate passes after a struggle. In the plan and in
every gate commit, check the ops against `ttnn/ttnn/bringup/INDEX.md` first: a step built on a library op or a composed
path where a bring-up fork already does the job (or could, extended behind an option) goes back as a note or a fix
job (agent rule 16: forks first, then TTNN, composing last). Read the diff of
the gate commit (`git show --stat`, then the parts that matter), not only the verdict. Reject it as cheating if it:
- loosens a threshold, edits a frozen test, a golden, `tasks.yaml`, `state.json` or `results/`, or skips a check;
- special-cases the test: recognizes the golden input or layer, hard-codes outputs, or reads the golden inside the model;
- hides CPU work in a device module (torch math in a forward, a host round-trip per chunk, per-call constant rebuilds);
  the one exception is a host input the plan places `host` (agent rule 5), and only its host part (e.g. a table lookup
  in `begin_chunk`), never work a device op could do;
- silences the problem instead of fixing it (a try/except that swallows it, a fallback path only the test takes);
- fakes an interface (attributes set only to get past a check, e.g. an MLA field on a non-MLA model);
- defers a step to op-gen (task DEFERRED, a commit "(deferred to op-gen: <op>)") on thin evidence, or when a TTNN op,
  a composition or a fork clearly does the job. Review every deferral like a gate commit: read request.yaml's
  evidence and check it yourself (the repo map, a grep for the op).
Accept it if the change is the honest fix, even when it touches shared code: a generic hook in the engine, a new
branch for a new layout, a fix in a shared op with its own test. To reject: pause, revert the gate commit
(`git revert`), write a note for the task saying what was wrong and what the honest fix is, then
`$B rerun --from <task>`. Log the decision either way.

Never:
- edit anything in the tree while an agent step runs, except your own files (above). `python -m
  models.demos.common.bringup.orchestrator pause --spec <spec>` stops it before its next task; the path check would
  charge your edit to the running agent;
- approve the intake, plan or performance picks for the person, loosen a threshold, or edit a frozen test. When the
  person delegates the picks, add each as a perf task (step `perf`, role `perf`, `brief.details` with the exact
  change, deps on X.2 or the previous pick, gate = an accuracy rung plus the profile with a time threshold below the
  X.1 baseline); perf tasks do not void the plan approval;
- brief a perf pick to measure anything beyond the e2e chunk time and the changed section's time, before and after,
  from the plain profile (minutes). Per-op mode (BRINGUP_PROFILE_OPS) syncs after every ttnn call, about a minute per
  layer: it runs on the representative layers only, never the whole model (Xing P.3: a per-op before/after over 40
  layers ran 40 minutes for a change to the experts, and was then killed by the agent's task time limit);
- run device measurements for a perf pick by hand. Every perf task gets an `ab` switch in tasks.yaml (`ab: {env:
  {<VAR>: <old value>}}`, the old path; optional `ab_rungs`). After its agent (which stops at the first frozen-test
  failure), the orchestrator itself produces the on/off table (frozen tests, ladder at `last` and the full-target
  rung, one plain profile each) at `runs/<run>/ab/<task>/table.md` and waits for the owner; bring the owner the table
  and record their answer with `orchestrator decide --task <id> --accept|--reject`, then resume;
- change the spec without asking (a spec edit voids the intake approval; re-approve on their word);
- run `tt-smi -r`, or use long timeouts for a device check (the box test takes seconds).

Report briefly on each gate the person would care about; say plainly what you decided, stopped or rejected, and why.
The orchestrator re-exports the dashboards after every gate. Publishing is yours: only an interactive session can
publish to claude.ai (a `claude -p` run has no Artifact tool). Publish each style once (Artifact), record its link
with `python -m models.demos.common.bringup.dashboard.links --spec <spec> set --style <style> --url <url>`; from then
on the watch wakes you with `DASHBOARD STALE` when the local files are 15 minutes ahead of the last publish: publish
every link (same file paths, or `url` from a new session), then `dashboard.links --spec <spec> published`. When gates pass every few minutes, wait with the watch's `--quiet-passes` (problems only; its timeout prints the
progress) and report progress at each timeout instead of every gate.

### At the end

Curate the model's knowledge before reporting. Go through `<bringup_dir>/known_issues.md` and `repo_map.md`: promote
every lesson that would change what an agent does on a different model (TTNN / fork behaviour, op limits, precision
mechanisms, testing method, framework / infra / hardware, serving rules) into the shared
`models/demos/common/bringup/knowledge/` files, merged with the entry it overlaps, at most three lines, with
`Applies to:` and `Seen at:` (`python -m models.demos.common.bringup.knowledge.check --model-dir <model_dir>` checks
both). Delete model entries the model's code has since fixed. Promoted or moved entries keep their id.
Then tally what the agents used: `python -m models.demos.common.bringup.knowledge.usage --out runs/<run>/knowledge`
(it reads every step's knowledge-use report, all runs and models) and act on `usage.md`:
- contradicted: re-check each like a `changed` entry (`skill/rebase-bringup/staleness.md` step 2), fix or delete;
- unused over 5 relevant runs: cut it, or move a shared entry that only one model needs to that model's file;
- a model entry used in 2+ runs: a promotion candidate (above);
- a missing topic that recurs: write the entry, if the run has the evidence;
- uses never traced to the entry's code: say so in the report (the agent may cite entries it did not apply).
Show the owner the shared-file diff and the usage summary for approval.

Report where the run's time went: `python -m models.demos.common.bringup.timing --spec <spec>` (the model, the
agents' device runs and other tools, gates on device and on CPU, idle time by reason: waiting for a person,
stopped, paused, not running; also on the dashboard). Use it during the run too, when a step seems slow.

Besides the metrics the person asked for, report every change the agents made beyond the model's own code, one line
each with its commit and your verdict: fork changes (ttnn/ttnn/bringup: what, why, which tests), framework, test or
knowledge edits, departures from the plan or from owner rules and the reason the gates forced them, the checks you ran
and their results, and anything you rejected or reverted. The person approves these after the fact, not during the run.

### Keep the session's context small

A run takes hours and dozens of gates; the supervising session must not fill its context with logs.
- Pull numbers, not files: extract the metrics you need from `state.json` / `results/<task>.json` with a short
  `python -c` or `grep`, and `tail -n` the orchestrator log. Never `cat` a gate log, an agent transcript
  (`runs/<run>/agents/*.jsonl`), a golden or a dashboard page.
- Wait with the watch (section 4, "Always be watching"), which already filters to the events you act on; do not
  stream step starts, freezes or agent transcripts.
- One line per routine gate ("C.x passed, 29/56, commit abc123"); detail only for failures, stops and decisions.
- A `KNOWLEDGE CONTRADICTED <id>: <evidence>` line: tell the owner at once (the entry, what the agent says is wrong,
  the evidence); the entry is re-checked at the run end (contradicted, below), or sooner if the owner asks.
- Keep checks short: the box test takes seconds; never give a device check a long timeout.
- Hand-off: everything that matters is on disk (tasks.yaml, state.json, results/, BREADCRUMBS.md, supervision.md,
  git). When the context passes about 75%, pause the run, append a "hand-off" entry to supervision.md (where the run
  is, what is pending, open decisions), and tell the person to continue in a fresh session with `/bringup` and "report
  on <model>"; a new session picks up from the files, not from memory.

## 5. Deferred steps and op-gen

An implement agent may defer a component step TTNN has no proper op for: it writes `<bringup_dir>/op_requests/<op>/`
(request.yaml with the evidence, op_prompt.txt, feature_spec.py, reference.py, bind.py), the task becomes DEFERRED, the
step stays on the CPU bridge and the run goes on ("complete with N deferred" at the end). Launching op-gen is always the
owner's call; you never launch it and never push without asking. The owner will not type command names: map what they
say to the commands and drive the flow yourself.

| The owner says | You do |
|---|---|
| "approve the pooling request", "the X spec looks fine", "go ahead with that op" | `$B approve op-request <op>`, then `$B op-export <op>` right away; tell them the next steps it printed (submodule commit, gitlink bump, push, the `run_eval.py` command). Do the commits if they want; ask before any push; the launch is theirs |
| "what's pending", "what did the agents ask for" | `$B op-requests`, then one short paragraph per request: the op, which model step, why TTNN lacks it, the shapes, where the files are |
| "op-gen finished X (and Y) at <path or run>", "the codegen came up with these ops, retry" | find the generated `ttnn/ttnn/operations/<op>/` folders (from the eval run's logs or clone; ask for the path only if you cannot find it), `$B op-ready <ops> --from <folder(s)>`, then `orchestrator resume`; report which tasks rerun |
| "use shape X", "the math is wrong: ..." | edit the request files yourself between agent steps (pause the orchestrator if one runs), `python -m models.demos.common.bringup.plan.op_request refresh <dir>` after a shape or placement change, rerun `... check <dir>`, and ask for approval again: an edit voids it |

Unprompted: when a task becomes DEFERRED, review it (above), then tell the owner in one line with the request folder.
In every status report and at the end of the run, list the pending requests with their status (draft, approved,
exported, delivered), so the owner never has to remember them. Log each approval, export and delivery in supervision.md.

## 6. Resume, rerun, fork

- `python -m models.demos.common.bringup.orchestrator resume --spec <spec>` (after a stop, a pause, or a fix).
- `$B approve plan|perf --spec <spec>`; `$B status --spec <spec>` (DEFERRED tasks are listed with their op requests).
- `$B rerun --from <id> --spec <spec>` (a deferred task downstream stays DEFERRED; name it to redo it);
  `$B fork --from <id> --name <run> --spec <spec>`;
  `$B compare --spec <specA> --other <specB>`.
