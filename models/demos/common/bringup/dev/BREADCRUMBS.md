# Bring-up framework build: breadcrumbs

Append-only log, one section per framework task. The design is
`models/demos/ernie45_d_p/bringup/docs/pipeline_design.html` (draft 4); the plan is section 7 of
`models/demos/ernie45_d_p/bringup/HANDOFF.md`. The framework is validated with synthetic ledgers and fixtures
and by reading ERNIE's existing records; nothing of ERNIE is regenerated. The first real use is the next model.

Drive this ledger with
`python -m models.demos.common.bringup gate <id> --spec models/demos/common/bringup/dev/spec.yaml --commit`.

## F1 (2026-09-25): core
- `core/spec.py`: model spec (YAML). Only `model` is needed to run gates; `validate()` checks the full model schema
  (layers, block types covering every layer once, ladder rungs, state tensors, mesh). Paths: `repo`, `bringup_dir`
  (ledger in git), `art` (`/localdev/$USER/bringup/<model>`, outside git) with `hf/ golden/ tt_cache/<mesh>/<version>/
  profiles/ runs/<run>/`.
- `core/ledger.py`: tasks.yaml + state.json + flock. Topological order, downstream closure (for rerun), runnable set.
- `core/gate.py`: the verdict order is deps -> frozen hashes -> device policy -> rc -> metrics -> artifacts.
  HANG is its own verdict (safe runner exit 2) and the triage report is copied next to the log. Logs go to
  `$ART/runs/<run>/logs/`, not git. The commit stages only the ledger files and the task's `paths`, and nothing
  is written to state after the commit, so the tree stays clean.
- Device policy parses the command quote-aware (shlex): a device gate must use `scripts/run_safe_pytest.sh` or
  `scripts/tt-probe.sh`; any direct `pytest` / `python -m pytest` fails; `VAR=x scripts/run_safe_pytest.sh` is allowed.
- `core/metrics.py`: `record(name, value)`; the task and the results dir come from `BRINGUP_TASK` / `BRINGUP_RESULTS_DIR`,
  which the gate sets. Own `pcc()` (the precompile pass stubs `comp_pcc`).
- Gotcha: the ledger `.lock` showed up as untracked after a commit; the ledger writes a `.gitignore` for it.
- Selftest gate: `selftest/test_core.py` (31 tests, temp git repos, no device). The selftest conftest records
  `selftest_passed` / `selftest_failed`, so an empty collection cannot pass.

## F2 (2026-09-25): freeze, runs
- `freeze <id>`: formats the task's `tests` with the repo's pre-commit hooks first, then runs the gate twice without
  touching state: `BRINGUP_IMPL=reference` must PASS and `BRINGUP_IMPL=stub` (zeros) must FAIL. Then it hashes the
  files into the task's `frozen` block and commits them as `[<tag>][<id>][freeze]`. `stub_check: false` skips the two
  runs for tests that do not wrap a module (goldens, plan).
- Why format first: F1's gate commit showed black reformatting five files inside `git commit`. A hash taken before
  that would never match again. For the same reason `gate --commit` formats the task's `paths` before running, so the
  tested bytes are the committed bytes.
- `task_commit` greps `[<tag>][<id>] ` with a trailing space, so it finds the gate commit and skips freeze commits.
- Runs: `init-run <name>` stores `_run` (name, branch) in state.json. The resume point is the first unpassed task
  in dependency order (FAIL, HANG and STOPPED resume at themselves). `rerun --from X` resets X and its downstream closure.
  `fork --from X --name B` adds a git worktree on branch `bringup/<model>/B` at X's gate commit under
  `$ART/<model>/runs/B/worktree`, writes a spec copy that points at it with the same `$ART` (goldens and weight caches
  shared by path), and resets the downstream verdicts in the fork. The main checkout never switches branch.
- `compare --other <spec>`: per task, status, attempts, debugger attempts, wall time, changed agent definitions
  (blob hashes from `agent_hashes`), metric deltas.

## F3 (2026-09-25): reference side
- `reference/interface.py`: the reference contract (`forward_chunk`, `new_state`, `state_tensors`, `load_state`,
  `block_graph`, `component`, `chunk_context`) and `run_block`. A block is a list of `Step(name, inputs, output, kind,
  stateful)`; the recorder names are `L{i}.in`, `L{i}.<step output>` (last = `out`), `embed`, `final_norm`, `logits`.
  The graph is what the swap tests execute, so it must be the code: `check_reference` replays each block type's
  representative layer from the recorded input and a state holding the recorded prefix, and requires an exact match
  of every boundary (`graph_replay_maxabs == 0`).
- `check_hf.py`: per-layer and logits parity against HF (forward hooks on `model.model.layers`, or the model's
  `hf_layers` hook), with `--num-layers` truncation for checkpoints too large for fp32 on the host.
- `generate_golden.py`: same on-disk format as the ERNIE goldens (prefill-server golden trace), plus `content_hash`
  in the manifest, `state_tensors` in the metadata, and an atomic `.partial` -> final rename. It refuses to overwrite.
  With a layer subset it stores only the selected layers, plus every chunk's block input at each contiguous run start.
- `golden.py` `Golden` reader. Checked read-only against ERNIE's existing `s4096_c2048` (no content hash there;
  `pinned_hash()` falls back to the manifest file's sha256).
- Tests run the generic scripts on `selftest/fixture_model.py`: a synthetic 3-layer, 1-head fixture with an
  independent one-shot "HF" twin and a byte-level tokenizer. It is a unit-test fixture, not a model bring-up.

## F4 (2026-09-25): device-side test helpers
- `testing/harness.py`: `BRINGUP_IMPL` = device | reference | stub picks the module under test; `mesh_parametrize`
  uses the spec's mesh and `box.device_params` (or the `device_params` hook), and opens no device under reference or
  stub, so freezing a test does not hold the box. Comparisons: `pcc`, `match` (integer outputs), `topk_overlap`.
- Device hooks a model provides: `device_component(mesh, spec, layer, step)` -> fn(ctx, *host inputs), with the golden
  state prefix in `ctx.extra`; `device_model(mesh, spec, layers)` with embed / from_host / layer / final_norm / to_host
  / logits / free / sync and a state with load_prefix / to_torch.
- `testing/component.py`: the component test (one step, golden in, golden out) and the swap test (the block of the
  representative layer with the listed steps on device, the rest on the CPU reference; per-boundary trail recorded).
  Both read the component rung's last dumped chunk, so stateful steps always see a non-empty prefix.
- `testing/ladder.py`: stacks device layers over a rung. Non-contiguous subsets restart each contiguous run from the
  golden block input. Final hidden and top-k only when the stack ends at the model's last layer.
- `testing/contract.py`: engine-path contract test. Adds the two checks the ERNIE run missed: the engine's uint32
  [sp, 1, chunk/sp] input with a padded tail (`contract.pad_token`, default 0xFFFFFFFF), and ack timing, checked
  model-agnostically: the blocks a layer ack covers are read at ack time and must be byte-identical after a full sync.
  Note: `prefill_producer` itself pads with real pool tokens; 0xFFFFFFFF follows the tt-d-gen audit of the Blaze engine.
- `testing/profiler.py` + `profile.py`: the Tracy-free section profiler from ERNIE, now generic (`signpost("L{i}.phase.op")`),
  and a warm profile of the last chunk of the profile rung, written to `results/<task>_profile.json`.
- `testing/templates.py`: component and swap test templates rendered into `<model_dir>/tests/bringup/`; never overwrites.
- Generic pytest entry points in `tests/` (ladder, contract, profile) take the spec from `BRINGUP_SPEC`.
- Selftests use fake device hooks in `selftest/fixture_model.py` (reference + deterministic noise). Real device runs
  start with the first model.

## F5 (2026-09-25): plan, approvals, ledger generator, knowledge
- Owner's requirement: the planning step must check its plan against the memory available on the device. `plan/memory.py`
  does it from the checkpoint, not from the planner's numbers: the plan (`plan.yaml`) gives a placement and dtype per
  tensor pattern (replicate / shard / expert / shard_rows / shard_cols / skip); the check reads every tensor's shape
  from the safetensors headers (no data loaded), fails on any unplaced tensor or unused pattern, computes per-layer
  state from heads per chip x head_dim x dtype x min(target seq, window) x users (and checks the heads cover the
  config's KV heads), adds the planner's activation estimate (must be > 0) and extras, and compares with
  `box.chip_dram_gb` minus headroom (default 15%). Result in `results/plan_memory.json` for the dashboard.
- `plan/components.py`: every step of every block type and embed / final_norm / lm_head mapped once; COMPOSED or CPU
  needs `searched` (what was looked for). Findings live in `findings.yaml`, so appending one keeps the plan approved.
- `plan/approvals.py` + `approve <point>`: approvals store the hashes of the approved files (formatted first) and, for
  the plan, a signature of the ledger's structure (ids, deps, gate commands, thresholds, tests). Freezing and picked perf
  items keep the approval; any other change voids it, and the plan gate (`plan_approved == 1`) fails.
- `plan/ledger_gen.py`: the standard ledger R.1-3, G.<rung>, B.1, PL.1, C.<bt>.<step> (parallel), S.<bt>.<nn>
  (sequential swap order), L.<rung>, K.1, X.1-2. C and S tasks carry `tests`, the component golden's manifest as
  `freeze_extra`, and a `brief`. The gate now fails a task that declares tests but is not frozen.
- `intake/check_checkpoint.py`: expected tensor globs with shapes, counts, config fields vs the spec.
- `knowledge/repo_map.md` (every path checked to exist), `knowledge/known_issues.md` seeded with 20 entries from the
  ERNIE run and this build, `knowledge/check.py` format check. `plan/opportunities.py` ranks profiled sections by
  device ms and attaches matching repo-map rows and performance known issues; it never overwrites a list with picks.
- `tests/test_box.py`: mesh opens, all_gather exact and all_reduce error per axis, DRAM per chip.

## F6 (2026-09-25): orchestrator, agent definition, briefs, intake
- `orchestrator.py run|resume`: one process runs the ledger in dependency order. Scripted steps (goldens, box, integrate,
  perf) run their gate and hand a failure to a `fix` agent. Agent steps write a brief (`briefs/brief.md` + role text from
  `briefs/roles.yaml`) and start a fresh `claude -p --output-format stream-json --agents <run>/agents.json --agent
  bringup-engineer`. Tests are rendered, reviewed by the test role and frozen before the implement role starts. Three
  failed attempts -> WIP commit `[<tag>][<id>][wip]` -> `ttnn-expert-debugger` (the repo's own agent) with the WIP sha,
  logs and triage, three attempts -> STOPPED (exit 1). The plan waits for `approve plan` and X.2 waits for picks (exit 3).
- The agent definition lives in `agents/bringup-engineer.md` and is passed with `--agents`, so nothing under `.claude/`
  changes. Each agent run is recorded under the task: role, attempt, session id, model, problems, and the blob hashes
  of the definition and brief templates (for `compare` after a fork).
- After every agent run: the tree diff must stay inside the brief's allowed paths (snapshot of `git status` before and
  after, content-hashed; generated/, probes and caches ignored), the stream-json Bash log must not reach the device except
  through the safe runners (direct pytest, or python on a file / heredoc / -c importing ttnn), and known_issues.md must keep
  its format. Any problem fails the attempt even if the gate passes.
- `BRINGUP_AGENT_CMD` swaps the CLI; `selftest/mock_agent.py` is a scripted stand-in. 11 orchestrator tests cover pass,
  retry with the previous failure in the brief, debugger hand-over and rescue, STOPPED and resume, plan approval, picks,
  path and device-command violations.
- `new --model --hf-id` scaffolds `models/demos/<model>/bringup/` (spec template, hooks skeleton, breadcrumbs).
  `ledger_gen --early --write` writes R, G, B and PL.0; PL.0 runs `ledger_gen --extend` after R.3, when the block
  graphs exist, and appends the C, S, L, K, X tasks.
- `skill/bringup/SKILL.md`: the conversational intake. It is not installed under `.claude/skills/` (shared config); it
  needs the owner's go-ahead to copy or link it there.

## F7 (2026-09-25): dashboard
- Owner's requirement: every new model gets a dashboard with the ERNIE dashboard's look and sections.
  `dashboard/template.html` keeps the ERNIE CSS, layout and interactions (progress meter, clickable gate ladder, model
  graph + layer strip, op coverage, findings, chunk timing, per-chip sharding cards with memory bars, the Kimi-style
  "where the time goes" timeline with the per-chip grid, the PCC trail on a log scale). Everything ERNIE-specific
  became data: the model graph is drawn from the reference's block graphs (`results/block_graphs.json`, written by
  `check_reference`), one column per block type with residual skips; the layer strip and trail x-axis size to the
  model; sharding reads `results/plan_memory.json` (the plan gate's computation) plus optional `chips` and
  `ccl_per_layer` in plan.yaml; profile phases come from the section names, with the bound (compute / memory /
  comm / other) guessed from the name unless plan.yaml `profile_sections` describes them; the chip grid follows the mesh.
- New in the ladder: STOPPED and HANG states, "waiting for a person", frozen-test and agent-run summaries per task.
- `dashboard/export.py --spec S` writes `<bringup_dir>/dashboard/index.html`.
- Test: the page script runs under node with a small DOM stand-in (`selftest/dom_shim.js`) on synthetic records and on
  an empty bring-up, so a runtime error in any section fails the gate.

## F8 (2026-09-25): README
- `README.md`: start, steps and task ids, layout, the rules the runner enforces. Known issue added: templated YAML
  fails check-yaml at commit (the F6 commit did not land the first time).
- Next: the first real use, Gemma-4 26B-A4B on the 1x4 Blackhole box (the owner's choice), ledger in
  `models/demos/gemma4_a4b_d_p/bringup/`.

## F9 (2026-09-25): fixes from the Gemma-4 intake
- A rung with `golden: X` no longer has to come after X in the ladder (ERNIE's order runs the cheap "last chunk after a
  golden prefix" rung before the full rung); it must name a rung that owns its golden with the same seq/chunk.
- R.1 is a scripted intake step (it was mapped to the reference role and would have started an agent) and also checks
  that the intake approval matches the current spec (`intake_approved == 1`).
- Agent steps try their gate once before starting an agent when their paths already exist (R.3 after R.2, swap tests
  whose components already pass). The failed pre-check is handed to the agent as the previous attempt.
- `agents.read.<role>` in the spec lists files every brief of that role names (for Gemma: the HF modeling code for the
  reference role, `models/demos/gemma4/tt` for plan and implement).

## F10 (2026-09-25): brief and commit details
- The reference-role brief now says what R.3 and the goldens step will hold the reference to (exact graph replay,
  chunked == one-shot, a CPU run over the longest rung) and where to split the block into steps.
- Gate commits also stage the model's `approvals.yaml` and `findings.yaml` when present.
- Gemma-4 26B-A4B: R.1 passed (63d8dc763b8); the orchestrator runs from R.2 as run `run1`.

## F11 (2026-09-25): first false violation in a real run
- Gemma R.2 attempt 1 was flagged for "changed files outside the allowed paths": my own F10 commit and the dashboard
  re-export loop ran during the agent step. The orchestrator was stopped before attempt 2 (whose brief would have told
  an agent to revert framework files). Now a file that is clean after the step is not attributed to the agent (agents
  never commit), `*/dashboard/index.html` is ignored, and `resume` also resets tasks an interrupted run left RUNNING.

## F12 (2026-09-25): per-role retry policy
- Owner's correction: ttnn-expert-debugger debugs TTNN ops (hangs, CB sync, kernel numerics) and cannot debug a CPU
  reference. The design's "3 attempts, then WIP commit and the debugger" applies to the implement step; F6 had applied
  it to every agent step. Now `DEFAULT_POLICY` in orchestrator.py: implement and fix-after-a-device-gate escalate to the
  debugger; reference, plan, contract, test and fix-after-a-CPU-gate stop for a person after their attempts. The spec
  overrides per role (`agents.policy.<role>: {attempts, escalate, debugger_attempts}`); `--attempts` overrides all.
- Gemma R.3 (for the record): chunked != one-shot at PCC 0.99983 was MKL matmul results depending on the row count;
  per-expert token groups differ between chunked and one-shot, and routing flips amplified it. The reference agent
  pads each expert group to a multiple of 32 rows; chunked now matches one-shot bit for bit.

## F13 (2026-09-25): chat-wrapped golden text
- Gemma-4 26B-A4B-it predicted the raw book at 16-19% top-1 (ERNIE: 69-87%); HF itself degenerates on raw text
  ("the age of times, it was the age of times") but answers and quotes Dickens correctly inside its chat template.
  The owner chose chat-wrapped input: `text.chat_template: true` puts the book in one user turn (the template
  supplies BOS) and truncates inside the turn. Worth a check at intake for any "-it" checkpoint: the goldens record
  `text_top1_acc`, and a value far below ERNIE's is the signal.

## F14 (2026-09-25): canonical prefill prompt
- Owner's requirement: the framework needs one deterministic input that prefill is tested on. Before this, every
  script tokenized the text from the spec itself, so a spec edit silently changed every step's input.
- `reference/prompt.py`: R.1 builds `$ART/<model>/input/prompt.json` once (token ids, wrap mode, source sha256, the
  chat prefix, sha256 of the ids) and refuses to overwrite it with a different prompt. `text_tokens` now reads it,
  so HF parity, the chunked check, every golden, the ladder and the contract prefill prefixes of the same tokens.
  Goldens refuse to run without it and record `prompt_sha256`; frozen tests pin the golden manifest, hence the prompt.
- Wrap modes: `raw` (BOS + text, base models; ERNIE's input), `user_turn`, `model_turn` (chat template with a short
  request, then the text as the model's reply). Measured on Gemma-4 26B-A4B-it, top-1 next-token over 2048 tokens:
  raw 18%, user_turn 13%, model_turn 67%. Instruction-tuned models are trained on model turns only, so
  `model_turn` is the choice for `-it` checkpoints; F13's user-turn wrapping was the wrong one.

## F15 (2026-09-25): parity length vs sliding window
- Owner asked whether Gemma's 43% next-token accuracy at 8k-16k was a red flag. It was not (HF agrees with the golden
  100% past the window; the model's own accuracy falls with position), but it showed R.2's 512-token parity never
  reached the 1024-token sliding window, and R.3 (reference vs itself) cannot catch a window bug. `validate()` now
  rejects `hf.parity_seq` <= `checkpoint.config.sliding_window`.

## F16 (2026-09-25): first device run of a generic test
- Gemma B.1 failed on the framework's own `tests/test_box.py`: `MeshDevice.get_devices()` does not exist (the selftests
  never open a device). It uses `get_num_devices()` now and records the mesh shape; the DRAM size comes from the spec.
  The orchestrator had handed the failure to a fix agent that is not allowed to edit framework code; stopped it and
  fixed it here. Lesson: a generic device test needs one real device run before a model depends on it.
- Also committed: the R.3 agent's known-issues entry on M-dependent CPU sgemm (padding expert groups to 32 rows).
- The F16 gate itself was refused by the device policy: a device gate could not run any `python`, not even the CPU-only
  knowledge check. Now a python command in a device gate is refused only if its script or `-m` module imports ttnn
  (or cannot be resolved).

## F17 (2026-09-25): approval scope
- Gemma PL.0 (ledger extension) had step "plan", so after its gate passed the orchestrator stopped for a plan
  approval before the plan agent had written anything. Now PL.0 is a scripted `ledger` step, PL.1 declares
  `approval: plan`, and only tasks with an `approval` field wait for a person.

## F18 (2026-09-25): device-command check and package files
- Gemma C.sliding.attn_norm attempt 1 passed its gate but was failed as "python on device code": the agent edited
  hooks.py with a python heredoc whose text contained "import ttnn". The check now parses the code (heredoc, -c,
  or script) with `ast` and flags only real imports of ttnn, directly or through a repo module that imports it.
  The orchestrator was restarted before its needless second attempt did anything.
- Gate commits now also stage the ledger's `__init__.py` and `results/block_graphs.json`; freeze commits stage the
  `__init__.py` files of the frozen tests' packages (they were left untracked).

## F19 (2026-09-25): what counts as device access
- Second false positive in a real run: a test-role agent ran a CPU-only golden analysis heredoc that imported the test
  harness, and harness.py imports ttnn lazily inside a function; F18's import tracing flagged it and burned one of
  the test role's attempts. Importing ttnn does not take the box; opening a device does. The check now flags python
  whose own code opens a device (open_mesh_device, open_device, CreateDevice, ttnn.MeshDevice, synchronize_device, ...);
  direct pytest stays flagged because that is how device tests run.
- Operating rule (learned twice): stop the orchestrator before editing framework code. The path check diffs the tree,
  so an edit made while an agent runs is charged to that agent, and the next brief tells it to revert the edit. Both
  times the run was stopped before an agent acted on it.

## F20 (2026-09-26): owner rules in every brief
- Gemma C.sliding.experts: two gate runs failed at device open (active ethernet core 31-25 timeout on all 4 chips; the
  safe runner's reset did not clear it) right after the first runs of the new GeluTanh expert pipeline. The owner reset
  the board and set the rule "always use 2D fabric". Spec `agents.rules` is now rendered at the top of every brief;
  Gemma's spec opens the mesh with FABRIC_2D (box check passes on it).

## F21 (2026-09-26): formal intake, supervision, dashboard styles
- `intake/check_hf_sanity.py` (added to R.1 by the ledger generator): records the checkpoint revision from the HF
  download metadata and checks it against `hf.revision`; runs the model card's usage example (`intake.smoke`) through
  the chat template and requires the expected answer; requires HF next-token accuracy on the canonical prompt above
  `text.min_top1` (default 0.4). HF has no standard golden input per model; the usage example is the closest thing,
  and the accuracy floor would have caught Gemma's raw-text input at intake.
- Orchestrator: `pause` (stops before the next task; edit the framework only while paused), and an infrastructure
  stop: a gate log with a device-open signature (active ethernet core timeout, "Try resetting the board") stops the run
  with "reset the board" instead of burning attempts.
- Dashboard styles: `dashboard/teletext_template.html` (the teletext service built by a subagent, now data-driven from
  the same `build()` as the standard page); `export --style standard|teletext|both`, default from `dashboard.styles`.
- `/bringup` skill rewritten: interview (target, ladder, owner rules, dashboard style, retry policy), on-the-spot checks
  (revision, base vs instruction-tuned -> input wrap, usage example -> smoke), spec and approval, launch, and a
  supervision section (failure classes and actions, a `supervision.md` log per model, the never-list). Installed as
  `~/.claude/skills/bringup` -> this directory: the repo's `.claude` is a symlink into the tt_ops_code_gen submodule,
  so a project-level install would modify another repository.
- Gate commits also stage the model's `spec.yaml` and `supervision.md`.

## F22 (2026-09-26): context discipline in the skill
- The supervising session reached 78% context after ~40 gates, mostly from printed logs. The skill now says: pull
  numbers not files, filter the monitor, one line per routine gate, short device checks, and at ~75% pause, write a
  hand-off entry in supervision.md and continue in a fresh session from the files.
- The `pause` command worked on its first real use (Gemma paused cleanly before C.global.attention).

## F23 (2026-09-26): 2D fabric in the agent definition
- Owner: "always use fabric_2d" belongs in the agent definition, not in supervisor instructions or one model's spec.
  It is now rule 4 of `agents/bringup-engineer.md` (every agent, every model), and the spec template's
  `box.device_params` defaults to FABRIC_2D. `agents.rules` in a spec stays for model-specific rules.

## F24 (2026-09-26): teletext carousel
- Owner: the teletext screen should switch between Model graph and Index every minute (then 20 s). While page 100 or 102 is on
  screen and hold is off, the screen rolls to the other one after 20 s (`CAROUSEL_S`). A key press restarts the count, hold stops it,
  and other pages stay put. With prefers-reduced-motion, hold starts on, so it does not flip.
- `selftest/teletext_shim.js` runs the screen script under node with a fake DOM and clock;
  `test_teletext_carousel` checks the timeline (the flip, hold, a key press, another page).

## F25 (2026-09-26): device-test timeout
- Gemma L.s56320 passed every check (worst layer 0.9937, state 0.9895, top-5 1.0), then pytest-timeout killed it at
  300 s, the repo `pytest.ini` default. The orchestrator took that for a failed check and started a fix agent (stopped
  by the supervisor before it changed anything).
- The ladder, contract and profile tests now carry `pytestmark = device_timeout(S)`, from spec `box.test_timeout_s`
  (default 3600 s; in the spec template). run_safe_pytest still catches hangs.
- A `Timeout (>Ns) from pytest-timeout` in a gate log now stops the task for a person, like a failed device open;
  no attempt is used and no agent starts.

## F26 (2026-09-26): shared code needs the owner's approval
- K.1's allowed paths include `models/demos/common/prefill/adapter.py`, shared by every prefill model. Nothing made the
  run wait for the owner; the contract agent started and was stopped by the supervisor (no edits made).
- `plan/approvals.shared_paths`: a task's paths outside the model dir (plus spec `paths.own`). The orchestrator waits
  for a person before any gate or agent of such a task until `approve shared:<task id>` is recorded; the approval is
  bound to that exact list of paths. The skill's supervision table says so.

## F27 (2026-09-26): no host work in the forward path
- Owner: no ugly shortcuts; nothing told agents to avoid host round-trips per chunk. Gemma's attention rebuilt RoPE
  tables on the host and uploaded them in every layer, every chunk; the global layers re-uploaded a page table.
- Agent definition rule 5 (every agent, every model): no from_torch / to_torch / synchronize_device, host compute or
  per-call constant rebuilds in a forward; shape/position constants are built once at load for max_seq, kept on the
  device and sliced there per chunk. Later rules renumbered 6-10.
- `testing/host_transfers.HostTransfers` wraps those ttnn entry points. The ladder counts inside `model.layer` on warm
  chunks (after the first) and records `host_transfers_per_layer`; ledger_gen gates it `== 0` on multi-chunk rungs
  (not on a golden-prefix rung, whose one chunk is cold). The profile records it from an extra warm run.
- Gemma's ledger is already approved and its rungs passed, so for Gemma it is reported by X.1 only; the fixes go on
  the X.2 list. Owner approved K.1 touching adapter.py (`approve shared:K.1`).

## F28 (2026-09-26): clean full-prefill measurement
- X.1 (`testing/profile.full_prefill`) now also runs the whole target prefill warm: embed -> all layers -> final norm
  per chunk, nothing read back, one sync at the end (a compile pass first). Records prefill_ms_full, prefill_tok_s
  and, from a pass that syncs after each chunk, prefill_chunk_ms_c<nn>. Excludes the LM head (Gemma's is on the host).

## F29 (2026-09-26): no approval formalities for agents; stale waiting note
- Owner: no approvals for agents to do their step's work. F26's wait is removed; the contract step's agent may change
  `models/demos/common/prefill` (orchestrator.CONTRACT_SHARED, allowed and staged for every contract task). ledger_gen's
  K.1 paths drop adapter.py (covered by the above).
- Gemma K.1: the engine's producer picks a KV reader by adapter name and sent Gemma to the MLA reader (KV_LORA_RANK).
  The agent's finding proposes an adapter hook; the next K.1 attempt can now make it.
- A task's "waiting for a person" note is cleared when the task runs again (the teletext kept flashing it on K.1).
