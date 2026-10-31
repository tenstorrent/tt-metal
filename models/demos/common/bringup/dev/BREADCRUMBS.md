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

## F30 (2026-09-26): the overseer decides
- Owner: use judgement, do not ask for trivial permissions; the overseer judges whether an agent is cheating and
  approves or rejects on its own. The skill's new "Decide; do not ask" section lists what still goes to the person
  (intake spec, plan, perf picks, board reset, push/PR, evidence the model or data is wrong), what counts as cheating,
  and how to reject (pause, revert the gate commit, add a finding, rerun).

## F31 (2026-09-26): perf role
- Picked performance items (the owner delegated the picks for Gemma) run as role `perf`: the brief carries
  `brief.details` (the exact change), the policy escalates to the debugger like implement, and the agent keeps the old
  behaviour selectable for comparison. The skill says how to add a pick; the contract role text no longer says
  "add only the registry line" (F29).

## F32 (2026-09-26): a fast perf loop
- Owner: 15-minute iterations are unacceptable. X.1 took that long because (1) run_safe_pytest's precompile pass runs
  the whole test once with stubbed checks before the real pass, and (2) F28's full-target prefill ran three times.
- The full prefill is now opt-in (spec `perf.full_prefill` or BRINGUP_FULL_PREFILL=1), for a final number only. The
  profile's untimed counting run also reads back the chunk's last-layer output: `pcc_chunk_out` vs the golden, so a
  perf gate is one ~5k chunk on the golden 51k prefix (accuracy + host transfers + time) in one test. ledger_gen runs
  the profile with `--no-precompile` (kernels compile on first use into the same disk cache).

## F33 (2026-09-26): profile sections; a standing assemble step
- Gemma X.1 failed "device profiler returned no durations": nothing marked sections (only "end"). `run_block` now
  signposts every step, so any model whose blocks go through the block graph is profiled per step.
- The ladder, profile and contract were measuring the hybrid harness (CPU reference, host in / host out per device
  step): about 20 s per 5k chunk of host glue. New standing step M.1 (role `assemble`, after the last swap test,
  before the first rung): one all-device model from the validated modules, every block through `run_block`, gated on
  host_transfers_per_layer == 0 and accuracy on a multi-chunk rung. The hybrid stays selectable for debugging.
- Gemma: P.1 (step perf, role assemble) wraps tt/model.py's TtGemma4Model as the device_model hook; X.1 now
  profiles it (real baseline); P.2 (SDPA config A) follows with its threshold from X.1.

## F34 (2026-09-26): false device-command alarm; no retry after a pass; profile timings on the dashboard
- P.1's agent rewrote hooks.py through a python heredoc whose string carried `ttnn.synchronize_device`; the regex
  called it device access, and the problem made the orchestrator start attempt 2 of a task whose gate had passed
  (stopped). `code_opens_device` now parses the code and counts only real calls (strings do not count). A passing
  gate is never redone for a command problem: the problems go to state `review` for the overseer. Out-of-path
  changes still retry.
- The timing section showed only ladder chunks (with readbacks). It now also shows the profile's warm last chunk
  (chunk_wall_ms) and the warm full prefill (prefill_chunk_ms_c<nn>), labelled.

## F35 (2026-09-26): readable timing, honest status, lean full-model gates
- Timing is one headline row per measurement: token range in k = 1024 (0->55k, 50k->55k), total and per-chunk time,
  tokens/s, how it was measured (warm, no readback / ladder with readbacks), and the model (all-device / hybrid
  harness, from `device_model_hybrid`). The ladder and profile record rung_seq/chunk/start, chunk_start/len,
  prefill_seq/chunk for this. Hybrid rows are greyed (standard) or folded into one line (teletext).
- While an agent works a task shows RUNNING (attempt n), not the FAIL of the check before it started.
- Full-model gates (integrate, assemble, contract, perf) run without run_safe_pytest's precompile pass
  (`gate.gate_command`; tasks.yaml is unchanged, so approvals hold).
- X.2 does not stop for picks when perf tasks after it are already in the ledger.
- The profile records sub-sections a module marks itself (`attention.sdpa` -> device_ms_attention_sdpa).

## F36 (2026-09-26): timing is performance only
- Owner: ladder rows (accuracy runs that read every layer back) never belong in the timing section. The exporter
  drops them for every model; only warm, no-readback measurements appear (profile chunk, full prefill). The selftest
  pins it. Token counts below 1k print as plain numbers.

## F37 (2026-09-26): chunk time by position
- Owner: the chunk-time-by-position table (one warm chunk at 0, 50k, 100k, 150k, 200k) belongs on the timing page,
  measured at the end. `testing/positions.py` + `tests/test_positions.py`: positions from spec `perf.positions`, default
  0 and 1..4 x the start of the target's last chunk; the device model is built with target.seq raised in memory so its
  position tables cover the longest start; zeros prefix and random ids (timing only); compile run, then a timed run.
  Records pos_chunk, pos_ms_<start>.
- ledger_gen adds the final X.3 (after X.2; the overseer points its deps at the last pick): full-target ladder, warm
  full prefill, position sweep. A pick is now a perf task with a role, so X.2 still stops when only X.3 follows it.
- Dashboards: a line chart plus table under Chunk timing (standard), a bar block on the teletext Timing page.
- Gemma X.4 runs the sweep (its X.3 predates F37). Profiler-off probe: 443 / 615 / 781 / 946 / 1112 ms.

## F38 (2026-09-26): chat-template arguments for thinking models
- MiMo-V2.6's template opens a think block after the generation prompt unless `enable_thinking=false`. Spec
  `text.template_kwargs` (e.g. `{enable_thinking: false}`) now reaches every `apply_chat_template` call: the canonical
  prompt (user_turn, model_turn; recorded in prompt.json) and the intake smoke. With it the book is the model's
  non-thinking reply (`<think></think>` then the text). Selftest pins both paths.

## F39 (2026-09-26): HF sanity after the model's own loader
- MiMo-V2.6's checkpoint (fp8 block-scaled dense, per-expert mxfp4, ~618 GB in bf16 vs 503 GB host RAM) cannot be run
  by the stock `from_pretrained` at R.1, and R.1 has no agent to write `hooks.hf_model`. Spec `hf.custom_loader: true`
  moves the HF sanity (revision, smoke, accuracy floor) from R.1's gate into R.2's, before `check_hf`; R.1 keeps the
  checkpoint check and the canonical prompt. Thresholds are unchanged. The reference brief says when and how to write
  `hf_model` (full model with `num_layers=None`, packed where it does not fit). Selftest pins the move.

## F40 (2026-09-26): a layer subset never runs layers after its last one
- MiMo layers 0-5 of 48: G.s56320 ran all 48 layers on CPU (10-13 min per 5k chunk, ~2 h, 212 GB RSS) because
  generate_golden always built the full reference; R.2's check_hf also compared all 48. Owner: fix the framework.
- generate_golden builds the reference for layers 0..max(selected) when the subset stops before the last layer and skips
  the model-level outputs (final norm, top-32, logits tail), which the ladder only compares when the stack ends at the
  last layer. check_hf defaults `--num-layers` to max(selected)+1 for such subsets. A subset that ends at the last layer
  (e.g. "0-1,29") still runs everything. The full-model HF sanity (R.1/R.2 with custom_loader) is unchanged.
- Selftests: the reference is built with [0, 1] for a [0, 1] subset of 3 and no model outputs are stored; [0, 2] still
  builds the full stack; check_hf's default prefix.
- Also: CPU gates set torch threads to physical cores (`core.metrics.cpu_threads()`), not `os.cpu_count()` (SMT
  siblings made MoE GEMMs ~6x slower, found by the MiMo R.2 agent).

## F41 (2026-09-27): the serving contract serves the layer subset
- MiMo K.1 (layers 0-5 of 48) failed three attempts on "acks ... (12 vs 96)": `testing/contract.py` expected acks and
  set `PREFILL_NUM_LAYERS` for all 48 layers while the device model and golden cover 6. The contract agent found it and
  proposed the fix (outside its paths). `served_layers(s)` returns (first, count) of a contiguous subset (a
  non-contiguous one is refused); `run_contract_test`, `engine_env` and `gqa_independent_pcc` use it. Selftest.
- Same class as F40: every framework path that counts layers must use `spec.layers()`, not `num_layers`. Remaining
  `num_layers` uses were checked: ladder, profile and positions compare against it only to detect a full stack.

## F42 (2026-09-27): the full prefill times a prefix subset
- MiMo X.3 (layers 0-5 of 48): every accuracy and position metric passed but `prefill_ms_full` was MISSING, because
  `testing/profile.py:full_prefill` returned None unless the stack was all 48 layers. The fix agent diagnosed it
  (framework, outside its paths); the overseer stopped it. Now any stack starting at layer 0 without gaps is timed
  (final norm only when it ends at the last layer), and `prefill_layers` records the layer count. Selftest.
- Third subset bug in a row (F40 goldens/parity, F41 contract): a sweep of `num_layers` uses in `testing/` found no
  other; ladder, profile and positions use it only to detect a full stack.

## F43 (2026-09-27): the TTNN operations behind every profile section, per layer
- Owner: clicking a section (e.g. attention.qkv) should show the ttnn ops it was made of; different layer types get
  their own tabs under "Where the time goes".
- `testing/profiler.py` op mode (`enable(mesh, ops=True)`): wraps `ttnn.decorators.FastOperation.__call__`; every
  outermost ttnn call syncs, drains the device profiler and books its programs to (layer, section, op, per-chip
  shapes of the first two tensor args), merging only back-to-back repeats, so rows are in execution order. Programs
  launched outside a wrapped op go to "(other)". `profile.py` sets the layer per `model.layer` call and, with
  `BRINGUP_PROFILE_OPS=1` / spec `perf.op_profile`, adds one extra warm op-mode run and writes `ops` into the profile
  JSON ("L3.attention.qkv" -> rows). Kernel durations are device-side, so the per-op syncs do not change them: the
  MiMo op-mode total (225.6 ms) equals the section profile (225.6 ms).
- The final X.3 task now sets BRINGUP_PROFILE_OPS=1 (ledger_gen). Exporter: `profile.views` = all layers (summed)
  plus one tab per block type (per layer, mean over its profiled layers); each step carries its `ops`. Standard page:
  tabs above the bar, and an execution-order op list (op, per-chip shapes, calls, ms, share) in the detail panel.
- MiMo re-measured ad hoc (tuned model, 51200->56320 chunk): results/X.3_ops_profile.json.

## F44 (2026-09-27): operation time, not only kernel time: the pipelined device timeline per op
- Owner: per op, show dispatch and host cost on top of device kernel time. DeepWiki's options: Tracy
  (`ops_perf_results` op-to-op / dispatch columns; needs a `-p` build and its host capture crashed on this box),
  `ttnn.graph` capture (host wall per call, no device side), perf_counter + sync per op (serializes the pipeline and
  inflates exactly the dispatch time being measured). Chosen: rebuild the device timeline of an un-synced run.
- Device perf records carry start/end timestamps (device clock; cycles per ns taken from end-start vs duration).
  `profiler.enable(mesh, timeline=True)`: no syncs; each outermost ttnn call's host dispatch time is timed; one read
  at the end (`collect_timeline`). `align_timeline` splits each chip's programs (launch order) per call with the
  op-mode run's per-chip program counts (the model is deterministic; sequences and totals are checked, a mismatch
  is an error, not a guess). Per call on the critical chip (longest timeline): kernel, idle gap before it, slot
  (end - previous end); kernels + gaps = the device timeline. The critical chip's own kernel is used because in a
  pipelined run an early chip's collective waits inside its kernel (MiMo all_reduce: 10.6 ms synced, 25.0 ms as max
  over chips pipelined).
- `op_profile` runs op mode, then the timeline run; op rows get gap_ms, slot_ms, host_ms; the profile gets
  `timeline` (device_timeline_ms, kernel_ms, gap_ms, host_dispatch_ms, host_wall_ms). Metrics op_rows,
  timeline_ok, timeline_*; the final X.3 gates op_rows >= 1 and timeline_ok == 1, so every bring-up ends with them.
- Dashboard: a pipeline line under "Where the time goes" and, per op, "device idle before it · host dispatch".
- MiMo 50k->55k: device timeline 225.6 ms = kernels 225.4 + gaps 0.2 ms; host dispatch 27.8 ms for 611 calls, hidden
  behind device work (host wall 226.6 ms): device-bound.
- Layout fix: the profile section is a grid; its children kept min-width auto, so the phase-label row under the bar
  (nine nowrap labels) widened the section to 1427 px inside a 1140 px frame and pushed the bar, notes and op bars past
  it. `#prof-sec>*{min-width:0}`; labels truncate with a tooltip; op names break after "." and "_"; op rows pin their
  cells on narrow screens. Checked with headless Chrome at 1400 / 1000 / 390 px (no horizontal overflow).

## F45 (2026-09-27): where each op's tensors live
- Owner: per op, where its inputs and outputs are (DRAM interleaved, L1 sharded). Op mode and the timeline run record
  `mem` per call: the first two tensor inputs and the outputs as "DRAM interleaved", "L1 height-sharded 64 cores",
  "+ RM" for row-major (else tiled). It is part of the row key (a placement change is a new row). Dashboard: in / out
  chips per op row (DRAM green, L1 blue) with a legend.
- `timeline_ok` now also requires every op row to have its timeline columns: a black reformat had silently dropped a
  field and left 0 of 247 rows with timeline data while the alignment itself passed.
- MiMo finding: every tensor of every op is DRAM interleaved (nothing in L1), a perf lead for the matmuls, norms and
  residual adds.

## F46 (2026-09-28): defer a step to op-gen, keep it on the CPU, continue the bring-up
- Owner: an implement agent that finds no proper TTNN op for a step (GLM-5.3's indexer key pooling) stopped the whole
  run. Now it may defer the step: the step stays on the CPU reference through a bridge, the agent leaves a request for
  the op code generator (op-gen, tt_metal/third_party/tt_ops_code_gen), and the rest of the bring-up continues.
  Launching op-gen is always the owner's call; the framework never launches it or pushes.
- Ledger: status DEFERRED counts like PASS for dependents (runnable, gate deps, resume point). A run whose only
  unfinished tasks are DEFERRED ends DONE, "complete with N deferred". rerun keeps a downstream deferral DEFERRED (an
  upstream change does not deliver the op; name it with --from to redo it); fork may start at a deferred task.
- `plan/op_request.py`: `<bringup_dir>/op_requests/<op>/` = request.yaml (op, model, tasks, component, layers,
  status draft|approved|exported|delivered, evidence searched/tried/why_not_fork, interface with per-device shapes
  from each tensor's mesh placement and the chunk lengths, tolerance from the component gate, acceptance = the frozen
  test and its golden), op_prompt.txt (op-gen's format, `# golden: <op>` first, `Import path:` last, `## Rules`),
  feature_spec.py (single-value TARGET axes as strings, one INPUTS case per chunk length, INVALID = []),
  reference.py (standalone `pytorch_<op>`, pure torch), bind.py (model side, not exported: the layer's weights and
  scalars). `new` fills the mechanical parts from the spec, ledger, reference and component golden, and leaves
  `<<AGENT` markers for the evidence, prose and the reference body. `check` rejects: missing files or fields, a
  marker left, thin evidence (empty lists, short outcomes, a one-word why_not_fork), a bad op name or one TTNN has, a
  task that is not a component task, a malformed prompt or feature spec, a reference that imports anything but
  torch/math/numpy, and a reference that does not reproduce the model's CPU step on a random 32-row input (bind.py
  supplies the weights; its shapes and params must match the interface).
- Deviation from the spec as written: the generator cannot turn a reference step (usually a closure over the layer's
  weights) into a standalone function mechanically, so it scaffolds reference.py with the step's source as a comment
  and the checker proves the agent's function equivalent numerically. bind.py was added for the weights.
- Orchestrator: after an implement attempt on a C task whose gate fails, a request naming the task that passes the
  check makes it DEFERRED (reason = the evidence summary), and the request plus the task paths are committed as
  `[tag][C.x] <title> (deferred to op-gen: <op>)`. A rejected request is a failed attempt; its errors go into the next
  brief. After the debugger's attempts, one last implement attempt (brief .201) may defer instead of STOPPED
  (policy `defer_after_debugger`, default on). Only the implement role of a component task may write op_requests/
  (path check); the debugger's brief has no defer section. A step tagged OPGEN in components.yaml (needs `searched`,
  block steps only) makes the implement brief say "defer it from the start". Every brief lists the deferred steps.
- `testing/cpu_bridge.py`: `CpuBridge(mesh, spec, step, fn, inputs=[placements], output=placement)` gathers the device
  inputs (replicate / shard:<dim> / shard2d:<r>,<c>) with get_device_tensors + to_torch, runs the reference step in
  fp32, and places the output with from_torch and the matching mapper. Its transfers run inside
  `host_transfers.bridged()` and count in `bridge_calls`, never in `total`, so `host_transfers_per_layer` stays 0.
  ladder, profile and positions record `deferred_cpu_steps` and `deferred_cpu_ms`; the profile JSON lists
  `bridged_steps`. `device_model_hybrid` is unchanged (the bridge is not the hybrid harness).
- Harness: a DEFERRED step runs on the reference in the swap tests; its own component test fails in device mode
  (while deferred, and whenever device_component returns a bridge), so a bridged step can never PASS its gate.
- Approvals: `approve op-request <op>` hashes the request files with request.yaml minus status/exported/delivered.
- `plan/op_export.py`: `op-export` (needs the approval) writes eval/prompts/<op>.txt and eval/golden_tests/<op>/
  (feature_spec with ttnn enums, helpers with the request's reference verbatim plus INPUT_SPECS / PARAMS /
  TOLERANCES and run_<op>, axes, test_golden, conftest, test_regression) and prints the owner's steps (submodule
  commit, push, gitlink bump, push, run_eval.py). `op-ready <op>... --from` copies op-gen's
  ttnn/ttnn/operations/<op> to ttnn/ttnn/bringup/<op> (imports and kernel paths moved), registers PYTHON_OPS,
  writes CHANGELOG.md and an INDEX.md row, marks the request delivered, sets the task's brief details to "use
  ttnn.bringup.<op> ...", and resets it and everything downstream with rerun_from. Configurable roots
  (--codegen-root, --bringup-ops); selftests run on tmp copies only.
- Dashboard (both styles): "Deferred to op-gen" section / teletext page 108, a banner that the results include N CPU
  steps, OPGEN and DEFERRED tags, timing rows and profile sections marked where the bridge ran.
- Overseer skill: a "Deferred steps and op-gen" section (plain-language phrases mapped to the commands, proactive
  reports of pending requests, editing a request on the owner's word voids the approval), deferral review in the
  cheating list. README steps and commands; docs/pipeline_design.html section "Steps TTNN cannot do yet".
- Also: `stage_paths` skips a task path that is an empty folder (git add refused it; hit by a deferral with no code
  yet), and knowledge/repo_map.md lost a brace path the format check rejected (the one selftest failing before F46).
- Selftests: 155 passed + 1 failing before, 192 passed after (test_opgen.py adds 36).

## F47 (2026-09-28): trim a layer subset's checkpoint after the HF sanity
Owner (Hy4 intake): the whole checkpoint of a layer-subset bring-up is needed only for the one-time HF sanity (smoke
"Paris", accuracy floor); afterwards the unneeded weights should go. Hy4 Preview is 1.56 TB for 6 of 78 layers.
- `intake/trim_checkpoint.py`, ledger task R.4 (step intake, scripted, deps R.2: the sanity has passed, at R.1 or at
  R.2 with custom_loader, and the reference's parity too). Only when the spec owns its checkpoint (no `prior`), the
  kept prefix stops before the last layer or `checkpoint.trim_drop` names tensors, and `checkpoint.trim` is not false.
  Keeps layers 0..K-1, K = max(last subset layer + 1, hf.parity_layers) (the reference runs the layers before a
  subset layer), and every non-layer tensor except trim_drop globs. Layer match `(^|.)layers.<n>.`, so
  `mtp_layers.0` is a non-layer tensor unless dropped by glob.
- Shards: only-kept stays, only-dropped is deleted (and its download metadata), mixed is rewritten as
  `subset-<shard>` and verified byte for byte (uint8 views, NaN-safe) before the original goes. The index lists only
  kept tensors (metadata kept, e.g. MiMo's tp_size); the original is `model.safetensors.index.full.json`.
- `bringup_trim.json` in the checkpoint dir, written before anything is removed: the full tensor map, the sanity
  metrics, keep_layers. `plan.memory.checkpoint_tensors` returns that map (checkpoint gate and plan memory still see
  the whole model); `check_hf_sanity` replays the recorded metrics (revision still read live, records `replayed`), so
  R.1 and R.2 still rerun. trim_checkpoint.main re-checks the sanity metrics itself (ledger_gen.sanity_metrics, now
  shared) and refuses to trim if they do not pass. Resumable: a truncated subset file is rewritten.
- Reference brief: read every tensor through the index, never by shard file name.
- The MiMo checkpoint was trimmed by hand the same way before this existed (160 -> 20 GB, commit 023405b50b0).
- Selftests: 192 -> 198 (test_trim.py adds 6; test_hf_sanity_records_revision_and_accuracy needed the marker lookup
  guarded like the revision lookup).

## F48 (2026-09-29): gate commits stage changed forks and knowledge entries
Hy4 run1: the C.dense_full.indexer agent made the `indexer_score` fork (ttnn/ttnn/bringup, INDEX.md, the nanobind
registration) and agents from R.2 on added 19 known issues and repo-map entries. Agents may write both (allowed paths:
BRINGUP_OPS and common_paths), but `stage_paths` staged only the task's own paths, so none of it was committed; the
overseer committed them by hand (5da28095718, and the knowledge commit after it). `stage_paths` now adds every changed
file under ttnn/ttnn/bringup and the two shared knowledge files (`_dirty_under`, git status, no __pycache__, no deletes):
one agent runs at a time, so those changes are the task's. Only changed files, so the gate's formatting pass does not
reformat other forks. Selftest test_fork_cases.py::test_gate_commit_stages_changed_forks_and_knowledge_only; 199 pass.

## F49 (2026-09-29): swap tests gate every swapped step; no test-role review for swap tasks
Owner (Hy4 run1): the test-role review of the swap tests cost the most agent time and found no bug. From the Hy4
state.json in this tree: 42 swap-test reviews (S.*.test.1), 8.1 h, 50 % of all agent time (16.1 h); the owner's
count was 41 reviews, 7.8 h, 51 %, 4 changed tests, 0 bugs. Note: in this tree all 42 frozen Hy4 swap tests carry
hand-added checks (165-1658 lines, the template is 25), not 4; the reviews added the same things each time: the
swapped step vs golden and vs the CPU step on the same (device) inputs with rel L2 / worst row / row norm ratio
limits, finiteness, "not a CPU bridge", and downstream / block-out rel limits.
- `testing/component.py`: `run_swap_test(..., checks=None)` is unchanged (block-out PCC gate + ungated trail).
  `checks="steps"`: each swapped step's override first runs the CPU step on the same inputs (a stateful step on a
  context whose state is a deep copy, before the swapped step, so the block's own state evolves exactly as before;
  side effects on the reference object itself, e.g. Hy4's per-chunk top-k record, are not isolated), then gates
  `swap_<step>_vs_cpu` / `_vs_golden` (COMPARE / THRESHOLD of the step's frozen component test, read with ast),
  and for float outputs `_finite`, `_rel` (<= swap_step_rel 0.02), `_row` (<= swap_step_row 0.05), `_ratio_min/max`
  (1 +- swap_step_ratio 0.02), and `_cpu_bridge` in device mode. An optional hook `swap_context(spec, ref, layer,
  golden, chunk, rctx, dctx)` fills what one block cannot compute (Hy4 moe_shared: the source layer's top-k).
- Added after the proof showed gaps (all generic, spec-overridable): `_bias` (median row norm ratio within 1 +-
  swap_step_bias 0.01: scale1.02 / 0.98 sat exactly on the 0.02 rel and ratio limits); the rel limit tightened to
  swap_step_calib 2 x the step's own component-gate error (sqrt(2 (1 - pcc)) from results/C.<bt>.<step>.json),
  never below swap_step_floor 0.005 (noise1e-2 on an fp32-exact step: rel ~0.01; the Hy4 device vs-CPU errors are at
  most 1.07 x the component errors); `swap_<last>_out_rel` (the block out with vs without the last swapped step's
  error: the tail re-run on the CPU from the CPU step's output, context snapshot after the step; <= swap_out_rel
  0.01; noise on the attn_hc gates moves out by 0.028 while the gates themselves move 0.006); selection outputs
  (>= 75 % exact zeros, e.g. dense router weights): rows whose top-k support moved are counted (`_reselected` <=
  swap_step_reselect 0.02) and left out of the row numbers (the Hy4 device router flips 0.07 % of its choices vs
  the CPU on the same input, which would fail a worst-row limit).
- `testing/mutate.py`, `BRINGUP_IMPL=mutate:<kind>` (+ `BRINGUP_MUTATE_STEP`): the CPU step with its output altered:
  scale1.02, scale0.98, halfswap, rowshift, quarterzero, noise1e-2, sign; idxshift for integer outputs; control
  `bf16` (bf16-rounded inputs and output) must pass. mesh_parametrize opens no device in these modes.
- `testing/templates.py`: the swap template has `CHECKS = "steps"`; the component template is unchanged; rendered
  files are untouched (all existing frozen swap tests keep their behaviour).
- `orchestrator.py`: `freeze_tests` freezes a swap task at once (reference PASS, stub FAIL still required), logging
  `[S.x] swap test frozen without review (F49)`, unless `agents.swap_review: all | [block types]` names it; a failed
  unreviewed freeze starts the test role with the failure. Component tasks keep their test agent.
- `dev/f49_mutation_proof.py` / `.md`: the proof (below). Selftests: test_swap_checks.py.

Merged from dnijemcevic/glm53_prefill on 2026-09-30; renumbered F48-F52 -> F50-F54; their commit messages keep the
old numbers. (The selftest counts in F50-F54 are the glm53 branch's own, 197 -> 218, before the merge.)

## F50 (2026-09-28): mixed per-layer state (GLM-5.3 intake)
GLM-5.3-Flash has sparse-MLA layers (latent cache + pooled indexer keys, growing along the sequence) and KDA layers
(recurrent state [heads, 128, 128] + conv tail, fixed size). The framework assumed one list of state names for every
layer and stored only the final state, which every "prefix at chunk start" consumer sliced to `start`: exact for a
KV cache, wrong for a recurrence (the final state already includes the chunk under test).
- Spec: `state.by_block_type: {bt: [names]}` (names per layer type; default `state.tensors` for all) and
  `state.fixed: [names]`; `Spec.state_names(layer)`, `Spec.state_fixed`; validation (known block types, all covered,
  names in `state.tensors`).
- Golden: per-layer names; fixed tensors also stored before every dumped chunk as `kv_cache/layer_{i}_at_{p}`, and on
  the serving-contract rung after `seq - contract_tail_pad` tokens (the last chunk rerun, cut short, from a copy of
  the state before it). `metadata.json` adds `state_by_layer`, `state_fixed`, `state_snapshots`.
  `Golden.state(i, at=p)`: growing tensors are the final ones (consumer slices), fixed ones the snapshot at p (error
  if there is none).
- Consumers pass `at=start`: component and swap contexts (harness), the ladder's golden prefix, the profile. The
  profile reloads the prefix before every run when the spec has fixed state (a recurrence advances on each rerun).
  check_reference takes the replay prefix from the state before the last chunk, not from the final state.
- Contract: acks expected per slab layer when the adapter has `kv_slot_layer_ids` (hybrids; the engine already acks
  that way); fixed state needs `hooks.contract_state_pcc(...)` against the tail snapshot, else the contract fails.
  Engine-side migration of fixed state (a table kind without a position axis) is not done: K.1 of such a model.
- Memory: state entries take `per_token: false` (fixed size, no seq factor), `seq_stride` (pooled rows),
  `seq_divisor` (sequence split over chips); fixed entries are checked only against their own `kv_heads`.
- Dashboard: any `pcc_state_<name>_L<i>` is per-layer.
- Device-model contract (harness docstring): `layer(i, h, 0, state)` starts a new sequence, so a fixed state resets.
- Selftests: 197 -> 212 (test_mixed_state.py adds 15; the fixture gains `fixture.recurrent_layers`, a linear
  recurrence with an HF twin). 14 of the 15 fail on the pre-F50 code.

## F51 (2026-09-29): gate commits carry the shared paths agents may change (GLM-5.3 run)
superseded by F48 (this branch); code not taken, see merge commit

- Symptom: GLM-5.3 C.dsa_moe.attention extended the ttnn.bringup sdpa fork (`high_precision` for sparse_sdpa: C++,
  CHANGELOG, INDEX, source.yaml/baseline, a new unit test) and its gate passed, but the gate commit 3ef3d142602 held
  only the model files; the fork edits stayed uncommitted in the working tree. Earlier gates had left agents'
  known_issues.md / repo_map.md entries uncommitted the same way.
- Cause: `orchestrator.allowed_paths` lets every step write `ttnn/ttnn/bringup` (BRINGUP_OPS) and the knowledge files
  (common_paths), but `core/gate.py:stage_paths` never listed them, so `git_commit` (explicit paths) skipped them.
- Fix: stage_paths appends `ttnn/ttnn/bringup` and the two knowledge files when they exist. `git add -A -- <dir>` also
  picks up new files under the fork (the new unit test).
- Selftests 212 -> 214 (test_fork_cases: the shared paths are staged and committed, including an untracked test
  file; missing shared paths are skipped). The first fails on the pre-F51 gate.py.
- Recovered by hand for GLM: the sdpa fork change and the knowledge entries committed after C.dsa_moe.attention
  (supervision.md).

## F52 (2026-09-29): state.json keeps only the gated metrics of a test that records many (GLM-5.3 run)

- Symptom: the S.kda_moe.10 gate passed but its commit failed: the repo's pre-commit `check-large-files` refused
  `models/demos/glm53_flash_d_p/bringup/state.json (506 KB) exceeds the 500 KB limit`; the orchestrator exited 1.
- Cause: run_gate copied every recorded metric into state.json. GLM's swap tests record 300-400 informational
  metrics each (per-step coefficients vs CPU and golden), so after ~60 of 96 tasks state.json passed 500 KB.
- Fix: `gate.state_metrics`: up to STATE_METRICS_MAX (64) metrics are all kept; above that, only the ones the task's
  thresholds match. results/<task>.json still holds every metric (the dashboard trails, prior view and profile read
  those); state.json metrics feed only the ladder row detail and runs.compare deltas, which use gated metrics.
- Selftests 214 -> 216 (test_core: many metrics -> gated only, results keep all; few metrics -> all kept). The first
  fails on the pre-F52 gate.py. The GLM state.json was migrated in place with the same rule (518 KB -> see commit).

## F53 (2026-09-29): the contract step may change the model's hooks.py (GLM-5.3 run)

- Symptom: GLM K.1 failed twice on "fixed-size state ['kda_recurrent', 'kda_conv'] unchecked: hooks.contract_state_pcc is
  missing". The agent had written the read-back in tt/runners/adapter.py, but K.1's allowed paths (task paths
  [tt] + prefill engine + forks + knowledge) left bringup/hooks.py read-only, so it could not register the hook; it
  rightly refused to monkeypatch the hooks module at run time.
- Cause: F50 made the contract gate require a `contract_state_pcc` hook for spec state.fixed, but the contract role's
  allowed paths and the gate's stage_paths never included the model's hooks.py.
- Fix: allowed_paths adds `<bringup>/hooks.py` for the contract step; stage_paths commits it for that step.
- Selftests 216 -> 217 (test_orchestrator: the contract step may change and commits hooks.py). Fails on the pre-F53 code.

## F54 (2026-09-29): profile gates size the device profiler for large models (GLM-5.3 run)

- Symptom: GLM X.3 failed only `timeline_ok = 0`: "device programs per chip 1281 != op-mode counts 1319 (profiler
  buffer too small?)"; the log said "Profiler DRAM buffers were full, markers were dropped".
- Cause: the pipelined timeline reads the device profiler once per chunk; TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT
  defaults to 1000 programs per core, and a 5-layer GLM chunk launches 1319 per chip (MiMo: 611).
- Fix: ledger_gen.PROFILE_ENV adds TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT=4000 (the X.3 fix agent confirmed 3000
  gives timeline_ok 1: 423.7 ms device timeline, 0.7 ms gaps). GLM's tasks.yaml profile commands (X.1, P.1, X.3)
  patched the same way.
- Selftests 217 -> 218 (test_plan: PROFILE_ENV sizes the profiler; fails on the pre-F54 ledger_gen).

## F56 (2026-09-30): component tests check themselves by output kind; freeze sweep; optional review skip
Branch `dnijemcevic/f56-component-checks` (from `dnijemcevic/ernie45_prefill` at 9996fb84734). Owner goal: drop most
of the component-test reviews (Hy4 run1: 42 reviews, ~4 h of agent time) without letting through any mistake a
reviewed test catches. The skip is built but OFF by default (`agents.component_review: all`) until the owner says OK.
Everything below is done and proven; a cherry-pick needs no rerun of the proofs (`dev/f56_mutation_proof.md`, device
numbers below). To reproduce: `python models/demos/common/bringup/dev/f56_mutation_proof.py` (CPU, ~35 min with two
processes) and `dev/f56_device_check.py` (device, ~10 min).

**What changed**
- `testing/component_checks.py` (new) + `testing/component.py`: `run_component_test(..., checks=None | "auto")`.
  `None` is exactly the old behaviour (every existing frozen test unchanged). `"auto"`: the golden gate as before
  (index outputs compared order-free, `topk_overlap`), plus checks vs the CPU step on the same inputs and vs the
  golden, chosen by output kind:
  - float: finite, rel L2, worst row rel L2 (0.03), every row's norm ratio (1 +- 0.015), median row norm ratio
    (1 +- 0.004: a systematic scale). The rel limit follows the step's precision: the CPU step run with every
    input-derived fp32 intermediate rounded to bf16 (constants such as weights and RoPE tables stay exact) has rel
    error e; limit = 2 e (3.5 e for Step.kind in `tests.low_precision_kinds`, default [moe]: bfp8 experts), within
    [0.003, 0.015], never below 1.2 e; vs the golden + the fp32 step's own error vs the bf16 golden. The fixed limits
    are also never below 1.2 x what the precision model reaches. Narrow outputs (<= 64 columns, iHC gates): also
    the worst column's rel L2 (0.015).
  - selection (>= 75 % zeros, dense router weights): same nonzero count per row, no negative weight, mean selection
    overlap >= 0.995, rel L2 on rows with the same selection <= 0.005, their row sums within 0.4 %.
  - index (integer rows that are sets, pads -1 / 0xFFFFFFFF): order-free overlap (mean >= thr, worst row >= 0.95),
    same valid count per row, no repeats, causal and own-position when the reference is.
  - int (anything else): match as before, and the sweep sends it to a review.
  Second inputs (vs the CPU step on the same inputs; rel >= 0.03, fixed limits x 1.5, no bias limit: they are for
  structural bugs, errors >= 0.1): `chunk0` (stateful steps), `layer<N>` (the farthest other golden layer), `mixed`
  (rows and columns of every float input permuted, seeded: removes layer 0's equal iHC streams and near-1 pre gates,
  under which a stream-order bug moves nothing on any real input), `small` (float inputs scaled by a power of 2 to
  RMS 1e-3: a norm epsilon), `big` (x 2: a clamp). A model hook `swap_context` (F49) fills both contexts if present.
  All limits are spec-overridable: `thresholds.component_*` (COMPONENT_DEFAULTS).
- Freeze sweep `BRINGUP_IMPL=mutations` (CPU, one process): the reference passes; every standard mistake of
  testing/mutate.py that applies to the output, applied on the golden and on every second input, fails; the bf16
  controls pass. It also lists, per float input, whether a 1.02 x scale of it is seen (informational: a norm is
  scale-free). `core/runs.py: freeze_task(..., mutations=True)` requires it (the test's rc; the sweep records no pcc).
- `testing/templates.py`: component template `CHECKS = "auto"`. `orchestrator.py`: a component task freezes without
  the test role when `agents.component_review` (all | none | [block types]; default `COMPONENT_REVIEW_DEFAULT =
  "all"`) leaves its block type out AND the sweep passes; a failed sweep starts the test role with the sweep's log.
- `core/metrics.py`: the temp file is per process (two proof processes on the adhoc task collided on it).
- Selftests: `selftest/test_component_checks.py` (48 + 1 skip); all selftests 298 passed, 1 skipped.

**Design decisions and dead ends (do not redo)**
- A fixed rel limit cannot work: noise1e-2 (rel 0.010) sits between the Hy4 device norms (0.002-0.003) and the
  bfp8 experts (0.008). The bf16 precision model was checked against all 39 recorded Hy4 device component results:
  device / model error 0.29-1.74 (bf16 steps), 2.09 (bfp8 experts); every device result is inside its golden limit
  with >= 2 x headroom (tightest: experts 0.0080 vs 0.0158, dense ffn_residual 0.0044 vs 0.0094).
- The first precision model rounded every fp32 op output; it rounded RoPE positions too (bf16 positions above 256
  are coarse), so chunk 0 attention showed e = 0.44. It now rounds only values derived from the step's inputs
  (taint by tensor identity). It also restores the reference object's attributes after its run: Hy4 caches its RoPE
  table on the reference, and a table cached in bf16 mode made the plain reference miss its own output by 7.8 %.
- An "input contribution" absolute limit (error vs min_i ||cpu - cpu(input i = 0)||) was dropped: with limits tied
  to the precision model it was never tighter, and a 2 % error in an addend that is 10-20 % of a residual stream is
  at the output's bf16 rounding for any test (the reviewed tests have the same bound).
- The generic "scale each input and weight by 1.02, flag blind parts" check of the plan is reporting only, for
  inputs: weights are not reachable generically (`ref.w[layer]` is Hy4's own), and for float outputs "slipped" and
  "below the limit" are the same thing. The second inputs do the work instead (they close every known blind spot).
- Second inputs with calibrated (tight) limits false-alarmed on the device: Hy4 attention L1 with every float input
  x 1e-3 has rel 0.009 and a median norm ratio of -0.75 % (the reviewed test scaled only attn_norm); permuted
  inputs rel 0.0055. Hence the loose second-input limits.
- `layer<N>`: the nearest layer (1) could not show a layer-0 stream swap (rel 0.004); the farthest (5) neither
  (1e-5: all pre gates ~1). `mixed` shows it (rel 0.67).
- A step whose bf16 precision is too coarse to see 1 % noise fails its sweep and goes to review (the fixture's tiny
  attention). That is intended.

**Proofs (final code, 2026-09-30; supersede every earlier number in this section)**
Per model: `dev/f56_proof/<model>.md` (every row; `dev/f56_mutation_proof.py --spec <spec> --all`). Device:
`dev/f56_device_check.py` runs checks="auto" on every component test of the spec's model, at the bring-up settings
(GEMMA4_SDPA_CFG=base, MIMO_SDPA_CFG=base: the settings the component tests gated; later perf picks lowered them).

| model | tests | mistakes | reviewed catches | new catches | missed by new, caught by reviewed | controls pass | freeze without review | device check |
|---|---|---|---|---|---|---|---|---|
| hy4_preview_d_p | 42 | 309 | 307 | 308 | 0 | 84/84 | 41/42 | 42/42 |
| glm53_flash_d_p | 38 | 260 | 257 | 258 | 0 | 76/76 | 36/38 | 38/38 |
| gemma4_a4b_d_p | 28 | 196 | 118 | 195 | 0 | 56/56 | 27/28 | 26/28 |
| mimo_v2_6_d_p | 20 | 140 | 115 | 140 | 0 | 40/40 | 19/20 | 18/20 |
| mimo_v2_6_d_p_2x2 | 20 | 140 | 115 | 140 | 0 | 40/40 | 19/20 | 18/20 |
| all | 148 | 1045 | 912 | 1041 | 0 | 296/296 | 142/148 | 142/148 |

- Mistakes: the 7 float kinds or idxshift of testing/mutate.py on every test, plus the Hy4 hard cases (eps, stream
  swap, addend x 1.02, no clamp, no router bias, post gates halved; Hy4 only: they patch its reference).
- The 6 tests that go to a review (sweep fails): attention steps whose bf16 precision limit is above 1 % (Hy4 moe_full,
  Gemma sliding, GLM kda_dense / kda_moe) and MiMo full_moe experts: 1 % noise is within their limit, so neither the
  new checks nor the reviewed tests catch it. Intended.
- The 6 device failures are real findings, not false alarms (each passes its own reviewed test, whose row limits are
  +-3-5 %):
  1. Gemma experts L0 / L5: a +1.0-1.5 % systematic scale (every row too large on the layer29 / mixed inputs), there
     since bring-up (recorded gate row ratio 1.022). Not investigated further.
  2. MiMo and MiMo 2x2 full attention L0 / L5: +0.3-1.0 % on every row (row norm ratio [1.0029, 1.0103]; at its gate
     [0.9938, 1.0073]); it came with a change to MiMo's attention after its gate. Not the sdpa fork: MIMO_V_PAD=1 (stock
     SDPA, V padded) gives bit-identical output. Candidates: the split qkv / kv weights and q_scale folding of commit
     5cb48fa24c9 and its neighbours (not bisected).
- Also found: Gemma's and MiMo's perf picks P.2 (SDPA config A: HiFi2, fp32 dest acc off, approx exp) cost Gemma
  attention 2.5-3 x its bring-up precision (global 0.0080 -> 0.0205, sliding 0.0051 -> 0.0164); their gates only
  reran the layer PCC rung, and no component test reruns after a perf pick.
- Device-driven fixes, all generic (commits on this branch): per-column limits on the golden only and never below
  4 x the whole-output limit; columns against their own norm; systematic scale up to half the step's error budget;
  second-input floors (worst row 0.15, row norm 0.1); router row sums vs golden allow the golden's own rounding; the
  precision model keeps aliases (views / in-place) and rounds only input-derived values.
- Not proven: weight-only bugs no output shows; recipes whose device precision the bf16 model misses by far (bfp8
  activations: those steps get loose limits or go to review via the sweep).

## F57 (2026-09-30): perf picks: accuracy first, the orchestrator's A/B report, the owner decides (Xing run1 P.3)
Status: DONE (2f64c73fb7d, 508858edd42, 2765a6a6c33). Incident: P.3 (routed experts at HiFi2) spent ~1 h: the agent
saw the frozen experts test fail and still ran a per-op profile over all 40 layers (killed by its task limit); the
first A/B then measured HiFi4 twice (the agent had put the default back on the old path, "on" = the tree's default).
- Per-op profiling (`BRINGUP_PROFILE_OPS`) runs on one representative layer per block type (`profile.op_layers`,
  `BRINGUP_PROFILE_LAYERS`); the whole model is refused.
- Perf role (`briefs/roles.yaml`): frozen tests first, stop at the first failure, never profile a failing change,
  keep the switch the task names (`$ab_switch`), make the change the default, never revert it on a failure.
- Orchestrator `perf_step`: agent -> A/B report -> WAITING FOR A PERSON -> `orchestrator decide --task <id>
  --accept|--reject|--rerun` -> `resume`. The report (runs/<run>/ab/<task>/table.md) runs per side, off
  (`tasks.yaml ab.env`) and on (`ab.change`), both set explicitly: every frozen test of the gate, the ladder on
  `ab_rungs` (default the gate's rung + the spec's last rung), one plain profile. Invalid (no runs / flagged) when the
  switch appears in none of the task's code or both sides give bit-identical metrics. A side measured for the same
  code and switch (side.json) is reused. Accept runs the gate and commits; reject reverts the task's files, status
  REJECTED (counts as done). Profile guard: `testing/accuracy_guard.py` (ACCURACY_FAIL marker; BRINGUP_AB=1 exempt).
- Open: the table lists the F56 mistake sweep's expected control lines (`pcc=0.000000`, `rel=1 row=1`) as failures,
  and its header says "on = the tree as the agent left it" (should name ab.change).

## F59 (2026-10-13): two-tier knowledge: a short anchored shared file, each model's own lessons in its bringup dir

- Symptom: every agent step read `knowledge/known_issues.md` (209 KB, ~50k tokens) and `repo_map.md` (22 KB) in full;
  205 of 245 known issues and 42 of 64 map rows sat under "Proposed" because nothing ever merged them, and many were
  about one model's layers, tests or code. Entries had no tie to the code they describe, so a rebase could leave them
  stale unnoticed.
- Fix: a reviewed proposal (bucketing, merges, rewrites, staleness against 9b393736f86) applied in one pass. Shared
  `known_issues.md` keeps 80 durable lessons in six sections (new: Testing; no Proposed), each with `Applies to:` and
  `Seen at:` anchors, at most SHARED_MAX_CHARS (800); shared `repo_map.md` keeps general rows. Model-specific entries
  moved to `models/demos/<model>/bringup/known_issues.md` / `repo_map.md` (6 models); 18 entries deleted (duplicates;
  framework bugs since fixed: F40-F42, 06c216f0224, 3c9fdf4ee27, 8a8d3452ee2, the contract hooks path).
- Process: agents write only their model's files (engineer rule 11, brief "When you finish"; `common_paths` and the
  gate's staging changed accordingly, shared files are no longer swept into task commits); briefs and PRIOR_READS list
  the model files; `knowledge/check.py` checks the shared file (sections, fields, anchors, length) and, with
  `--model-dir`, the model's files (the orchestrator checks both after every step). The overseer skill curates at run
  end: promote durable lessons into the shared files, delete fixed ones, owner approves the diff.
- References: claude.ai artifact links are unreadable to agents and other people; `knowledge/references/` holds the
  Kimi K2.7 4x4 layer plan (page + text version), which the repo map's sharding row now points at.
- Still open: the staleness re-check after a rebase / fork sync (the anchors make it mechanical) belongs in a
  `/rebase-bringup` skill.

## F60 (2026-10-13): /rebase-bringup skill; knowledge staleness lister and procedure

- The rebase of 2026-10-12 (llk_helper_library 73027b6e6ff -> bdc576fa652) done by hand, written down as
  `skill/rebase-bringup/SKILL.md`: baseline, rebase (blob prefetch in a partial clone, original-op conflict policy,
  tree verification), fork syncs by 3-way merge of fork_op.py copies, API ports of source-less forks, fork test and
  signature re-keys, fork_source re-records, per-model verification, publish with a backup.
- Staleness, isolated as `skill/rebase-bringup/staleness.md` (self-contained, runnable by an agent on its own) on top
  of `knowledge/stale.py`: per entry, `Seen at:` (else the blame commit; entries moved into a model's file keep the
  shared file's age), `Applies to:` resolved to repo paths (paths, partial paths by suffix, op names through
  `bind_function<"name">` and the fork folders), the commits since on them; status changed / anchor-missing /
  unanchored / unchanged; repo-map rows naming missing paths. At a6b2f86ab9a: 211 entries, 42 changed, 18 unanchored.

## F61 (2026-10-13): staleness re-check: device checks in the procedure, the first run's gaps closed

- First agent run of `staleness.md` (alone, at 864ec8b0891): 60 candidates, 37 still true, 9 rewritten, 5 left for a
  device, 13 procedure gaps. The procedure forbade device runs, so those 5 stayed open; step 4 "Device checks" now runs
  them (probe from `Check:`, safe runners, one at a time, bounded; confirmed / refuted / inconclusive; probes kept in
  `knowledge/probes/`; an entry whose mesh this box cannot open names the box and stays `Unverified since:`).
- Fields fixed: one `Check:`; `Unverified since: <sha>.` before `Found:`; `Applies to: method (...)` for testing /
  process knowledge; model entries get anchors when verified; rules for measured numbers, new options, repo-map rows,
  splitting a large pass.
- `knowledge/stale.py`: statuses needs-probe and method; partial paths resolve in the entry's own model first; arch
  copies (quasar) are not anchors; commits older than `since` (reordered by a rebase) are dropped.
- Device checks run at 864ec8b0891: FPU fp32 (confirmed: matmul rel 1.2e-3 vs fp64, all_reduce ~2 TF32 ulps, SFPU adds
  exact); MiMo streaming SDPA (re-timed with V 128: q256/k256 56.5 ms, q512/k128 44.9 ms, q512/k256 still over L1;
  k128 accuracy cost confirmed, 0.99842 vs 0.99962); as_tensor cache spin (not reproduced on 1x4 / 2x2, left open for
  the 4x2 LoudBox with the two Xing ring_mla entries).

## F62 (2026-10-13): short per-model known issues, full text in known_issues_detail.md

- Per-model files were what an agent still read in bulk after F59: GLM 70 KB, Hy4 55 KB (median ~1000 chars per
  entry). Every bullet over 600 chars is now short (claim, the number that is the lesson, fix, pointer; GLM 65 ->
  30 KB, Hy4 52 -> 25 KB, MiMo 14 -> 8 KB, Xing 5 -> 2.4 KB), its original text kept byte for byte under
  `## <title>` in the model's `known_issues_detail.md`, the bullet ending its Fix with `Detail: known_issues_detail.md.`
  An agent now reads ~13-21k tokens of knowledge per step (was ~58k).
- check.py: model entries <= MODEL_MAX_CHARS (600); a Detail pointer needs its section. Engineer rule 11 and
  staleness.md say so; orchestrator / gate let agents write and commit the detail file.
- stale.py: knowledge files are never anchors (the Detail pointer); a shortened entry keeps its age through its
  detail section's original title. Ages and statuses identical before and after the trim (211 entries).

## F63 (2026-10-13): knowledge-use tracking: entry ids, per-step reports, usage tally

- Owner asked how we know the knowledge is used. Every entry now starts with an id (`- **[K-12] Title.**`; models:
  GEM / GLM / HY4 / MIMO / M22 / XING), unique over all files, never reused; a moved or promoted entry keeps it.
  check.py checks presence and uniqueness and prints each file's NEXT ID; the length limits leave the id out.
- Every agent step writes `bringup/results/knowledge_use/<task>.<role>.<attempt>.json` (the brief names it, engineer
  rule 11): used (id, how it changed the step), contradicted (id, evidence), missing (topics). The orchestrator
  validates it, adds meta (model, run, task, HEAD, the files the step read / searched / edited / changed, from the
  transcript) and marks each use traced (the step touched the entry's Applies-to code), untraced, or unchecked
  (no anchors). A missing or malformed report is recorded in the agent run and echoed, not a failed attempt. The gate
  commits the reports with the step.
- knowledge/usage.py tallies all committed reports: uses, last use, contradictions, relevant runs (a run of the
  entry's model, or any model for a shared entry, that touched its code; every such run for an entry without anchors)
  and the unused streak; flagged at 5 (owner), model entries used in 2+ runs are promotion candidates, missing topics
  grouped. stale.py gets a `contradicted` status (a report's contradiction after the entry's Seen at), staleness.md
  verifies it first. The run-end curation (skill/bringup) acts on usage.md; both dashboards get a knowledge panel.
- Cite at use (owner): agents name an entry's id where it decides something (their reasoning, messages, BREADCRUMBS)
  and write a contradiction into the report as soon as they find it. Thinking is not in the transcript, but naming the
  id there makes the end-of-step report's recall better; ids in the visible text (messages, text written to files)
  mark the reported use `cited`, and cited ids missing from the report are kept as `cited_only` (counted as uses,
  `unreported`). The orchestrator prints `KNOWLEDGE CONTRADICTED <id>: <evidence>` when the step ends and records
  the ids on the agent run; the supervisor relays it to the owner at once.

## F64 (2026-10-14): plan for the deployment box, fit to every box; scaled runs on smaller boxes

- Owner: the target of every model is a Galaxy 8x4; QuietBox / LoudBox stand in for it during the bring-up. Modes:
  partial (the first layers bring up the layout; on a smaller box scaled so each chip sees the Galaxy's per-chip
  shapes, plus a real-size run for accuracy) and full (every layer, real dimensions, every end-to-end check, on any box
  where it fits). Marko's GLM changes (split caches, head-split weights) were plan decisions made for 56k on 4 chips;
  a 1M target on 32 chips forces them in the memory check.
- Spec (core/spec.py): optional `deploy: {box, seq, chunk}`, `boxes: {name: {mesh, chip_dram_gb, ..., overrides}}`,
  `run: {box, mode}`. The run's box becomes `box` and its overrides (spec keys) are merged, so every `box.*` reader is
  unchanged; `raw` is the file (run forks copy it). Without these keys nothing changes (`raw is data`).
- Plan (plan/boxes.py, memory.py): `roles` (role -> mesh axes, `sequence` required), placement `split` over a role,
  state entries with total `heads` and `split_heads` / `split_seq`, per-box `dtypes` / activations under `boxes`, and
  `conditions` (per-chip counts that pick a path: max / min / multiple_of, `else` = the other path). Per box:
  scale per role, the scaled context (exact or not), full model weights and longest context, the run's layers scaled
  (per chip as on the deployment box) and at real size, and each condition's path vs the deployment box's. Errors:
  the deployment box does not fit at its context, a condition without `else` fails on the deployment or run box,
  a `skip` drops decoder layers (the check applies the subset). check_plan writes results/plan_boxes.json, prints the
  table, records deploy_fits / box_path_differences, and requires a `## Boxes` section in plan.md. The run's own
  memory check keeps the run's layers.
- Briefs: a "Boxes" section (run box and mode, the deployment box, every box); the plan role also gets how to plan for
  the deployment box. Engineer rule 15: code runs on every listed box (no chip-count constants; sizes for every box;
  per-chip path choices as the plan's conditions say).
- Steps 4-6 (same day): the scaled run. plan/scale.py writes the reduced checkpoint (`<art>/hf_scaled/<run box>/`)
  from `scaling.yaml` next to the plan (config counts -> roles; per tensor: cut a dimension to the kept heads, or keep
  per-expert tensors by index; config.json counts scaled), so the CPU reference and the device model see a smaller
  model with no model-code change, provided they read their dimensions from the checkpoint's config. scaling.yaml is
  not an approved plan file: the scale step (role `scale`, task SG.1) may fix it. BRINGUP_SCALED=1 gives the spec's
  scaled view: the reduced checkpoint, `golden_scaled/<box>`, a `-scaled` weight cache, the deployment's context
  scaled by the sequence factor, and a two-rung ladder (`scaled` all chunks, `scaled_last` last chunk after a golden
  prefix; `scaled.ladder` overrides). Tasks, when a partial run's box has fewer chips than the deployment box: SG.1
  (reduced checkpoint + golden, after PL.1), SL.<rung> (after the real ladder), SX.1 (scaled profile; X.2 ranks
  opportunities from it), and O.1 captures the fork calls of both runs (fork_cases takes several capture files).
  The plan gate checks scaling.yaml against the checkpoint and prints the kept counts. Full mode needs every layer, so
  the end-to-end gates (final hidden, top-k, the device smoke) come from the existing rules. Ladder and profile
  record `scaled_run`. Dashboard: deployment and mode in the header, a Boxes section (per-box table, per-chip paths,
  differing ones marked), teletext page 111. The intake asks for the deployment box, the boxes and the mode.
- Not done here: results per box. Done in F183 (a seeded run's own ledger).
  Perf picks (P tasks) still A/B on the real-size profile; only the opportunity list uses the scaled one.

## F65 (2026-10-14): layers that read other layers' state

- DeepSeek-V4.1-Flash's attention shares state across layers: KV sources (2, 8, 14, 20) own the compressed KV and
  indexer keys, reuse layers attend over a source's cache with its top-k indices, re-index layers (24, ...) use the
  candidate pool of layer 20. The framework assumed every layer owns its state.
- hooks `state_sources(spec, layer) -> [earlier layers]` (spec.state_sources). A reader keeps no state (empty
  `state.by_block_type` entry); per-chunk results it reuses are fixed-size state tensors of its source. Tests load the
  sources' state as of the END of the chunk (they have run it): harness reference_ctx / device_ctx
  (`ctx.extra["source_state"]`, `source_len`), component / swap / mutation tests build the reference with the sources;
  check_reference's replay loads them after the last chunk; generate_golden refuses a subset without a reader's
  sources and snapshots fixed state at the end of every dumped chunk too; the memory check needs state entries only for
  layers that have state.
- Selftests: test_shared_state.py on the fixture (`fixture.shared`: a KV source and a recurrent source).
- Token ids for steps keyed on them (DeepSeek-V4.1's engram: an n-gram hash of the token ids selects table rows):
  `ctx.extra["tokens"]` in every test context (harness reference_ctx / device_ctx, the reference check's replay;
  forward_chunk sets it itself), and an optional device-model hook `begin_chunk(tokens, start)` that the ladder,
  profile, positions, smoke and op-report loops call before a chunk's first layer (a stack that starts from a golden
  block input never calls `embed`). Fixture `token_layers`; 2 more selftests.

## F66 (2026-10-15): the supervisor always watches; host placement; runner smoke only for whole models

- `watch.py`: `python -m models.demos.common.bringup.watch --spec S` is the supervisor's one wait. It returns on the
  next event of `runs/<run>/orchestrator.log` (gate result, attempt >= 2, STOPPED, WAITING FOR A PERSON, paused,
  DEFERRED, KNOWLEDGE CONTRADICTED, traceback, the end; exit 0), after its timeout (exit 2, re-arm), or at once when
  the orchestrator is not running (exit 3, the log's last event). `run` / `resume` write `runs/<run>/ORCHESTRATOR.pid`
  and remove it on exit. Why: DeepSeek-V4.1 run1 sat at the plan approval (WAITING FOR A PERSON, orchestrator exited
  3) for 7 hours because the supervisor's hand-rolled log grep was not re-armed. The /bringup skill (section 4,
  "Always be watching") makes the watch mandatory: re-armed in the turn that handles its result, never a turn ended
  with a live run unwatched, run once before answering "status?". Selftests: test_watch.py.
- Plan placement `host` (plan/memory.py): a tensor used from host memory (DeepSeek-V4.1's engram tables, memory-mapped
  per chunk) is placed but not counted per chip; needs a `why`; reported as `host_gb`. Before, the only off-device
  placement was `skip`, which the deployment check rejects on decoder-layer tensors.
- K.1 requires `smoke_runner_ok` only when the run covers the whole model (as L.smoke): a layer subset cannot answer
  the runner smoke, which skips itself there.
- Agent rule 5 (bringup-engineer.md) reworded to "no unnecessary host work": a host input (data the model keeps on the host, keyed only on token ids, e.g. the engram tables) is allowed in begin_chunk, placed `host` in the plan, tagged CPU, host part only; the skill's cheating list names that exception.

## F67 (2026-10-15): the serving chunk never grows; a proxy box shrinks it; goldens only for this box

- The chunk is the serving chunk (`deploy.chunk`). plan/boxes.py `context`: chunk = deploy.chunk x min(1, sequence
  factor), context = deploy.seq x factor. A bigger box keeps the chunk (its rows get fewer tokens); a smaller proxy box
  shrinks it so each mesh row holds the deployment's tokens per row (owner: 4096 on a 4x4 = 1024 per row -> 2048 on a
  2x2). Before, the chunk grew with the box (8192 on a 4x4 became 16384 on an 8x4). Owner: "16k is insanely large;
  never above 4-5K".
- `serving.max_chunk` (defaults.yaml, 5120): a spec with a deployment box fails validation when deploy.chunk,
  target.chunk or any ladder / scaled.ladder rung is above it. Specs without `deploy` are unchanged.
- The /bringup intake asks for the serving chunk and shows the per-row numbers; every ladder rung runs at this box's
  chunk (default three rungs); goldens only for this box's rungs (another box's come from its own run).
- Selftests: the proxy's context keeps the tokens per row, a bigger box keeps the chunk; the max_chunk check.

## F68 (2026-10-15): the framework pins the environment to its own checkout

- core/checkout_env.py `pin()`: TT_METAL_HOME, PYTHONPATH, TT_METAL_CACHE (<root>/built) and PYTHON_ENV_DIR to the
  checkout the framework runs from; called by `orchestrator run|resume` and the CLI's gate / rerun / sweep / compare,
  so every gate, agent and test inherits them (eval/run_eval.py does the same per clone). Why: ~/.bashrc exported the
  primary tree's values to every shell, so DeepSeek-V4.1's run in tt-metal2 shared the primary tree's JIT kernel cache
  (entries recompile on a tree mismatch, but two trees write the same entries) and profiler paths.
- Selftests: test_checkout_env.py.

## F69 (2026-10-15): the overseer steers the agents with notes and aborts

- `overseer.py`: notes in `<bringup_dir>/overseer.yaml` (`note --tasks GLOB --roles ROLE "text"`, `close`, `list`).
  Every open note matching a step heads its brief ("Overseer notes"); the agent answers each in its knowledge-use
  report (`overseer`: done / n/a / disagree + evidence); the orchestrator echoes `OVERSEER NOTE <id> DISAGREED` and
  `unanswered`, which wake the watch. `abort --reason`: ends the agent at work (its own process group; AGENT.json in
  the run dir), records the attempt ABORTED (no gate, not against the budget, at most 3 per task), and adds the reason
  as a note for the next attempt. The orchestrator's SIGTERM now unwinds and takes its agent with it.
- The path check ignores `overseer.yaml` and `supervision.md` (the overseer's files). The orchestrator re-exports the
  dashboards after every gate (background; BRINGUP_DASHBOARD=0 off). watch.py `--quiet-passes`.
- Why (DeepSeek-V4.1 run1): the overseer's interventions went through the live tree (a known_issues edit failed an
  agent's attempt; a supervision.md line would have been charged to one), kills by hand (lost attempts, an orphan
  agent) and hand-edited run state; the owner: "agents should see your feedback; the overseer should not intervene
  directly". Agent rule 1 and the skill ("Steer the agents; do not do their work") say how.
- Selftests: test_overseer.py.

## F70 (2026-10-15): where a run's time went

- `timing.py` (`python -m models.demos.common.bringup.timing --spec S [--run R]`): from the agents' transcripts (the
  result's wall and API time; each tool call's time, a run_safe_pytest / tt-probe call counted as a device run),
  every gate run's seconds (state history entries carry `s` now; older runs: the last duration only; device / CPU by
  the task's `device` flag), the run's timeline (runs/<run>/timeline.jsonl: the orchestrator's start and exit with
  its code; idle time between them named waiting for a person / stopped / paused / not running) and the device log
  (runs/<run>/device_timing.jsonl: the safe runner's lock wait and run time; the orchestrator sets
  TT_DEVICE_TIMING_LOG). Writes runs/<run>/timing.json. Dashboard: "Where the time went" (teletext page 112).
- Owner asked for a breakdown of device time vs LLM API time vs other (DeepSeek-V4.1 run1). Selftests:
  test_timing.py.

## F71 (2026-10-15): the published dashboard stays current

- Only an interactive Claude session can publish to claude.ai: a `claude -p` run (what the orchestrator starts) has no
  Artifact tool (tested). So the orchestrator refreshes the local files (F69) and the overseer publishes.
  `dashboard/links.py`: `<bringup_dir>/dashboard/links.yaml` keeps the links (every session publishes to the same
  pages) and the last publish time (`set`, `published`, `show`); the watch wakes the overseer with DASHBOARD STALE when
  the local files are more than 15 minutes newer than the last publish, also under --quiet-passes. The path check
  ignores dashboard yaml too. Owner: "I want this built into the framework", not a per-session timer.
- overseer.py CLI: `--tasks` / `--roles` repeat (action=append) instead of a greedy list that swallowed the note text.
- Selftests: test_links.py.

## F72 (2026-10-15): the swap tail limit is the noise floor, not 1 %

- testing/component.py `_check_tail`: the block-out check after the last swapped step was a fixed swap_out_rel (1 %).
  It now runs the same CPU tail from the last step's bf16 precision model (component_checks.precision_model, on the
  step's inputs) and limits the block out at component_calib x max(that error, the bf16 resolution of the output),
  with no fixed floor or cap; swap_out_rel only when the model cannot run. Metrics `<tag>_model_rel`,
  `<tag>_rel_limit`.
- Why (DeepSeek-V4.1 S.swa.06, owner): the fixed 1 % failed a correct device attention (1.6 %; the model's own fp8
  roundings amplify bf16 differences) and pushed the agents into "exact" kernels (25x matmuls, 64 ms per layer).
  Owner: "thresholds should be derived from the noise floor of the exact dtype with the CPU op, not fixed."
- Selftest: test_the_tail_limit_follows_the_precision_model.

## F73 (2026-10-15): the swap step limit is the step's noise floor, no fixed cap or floor

- testing/component.py `_check_step`: a swapped step vs the CPU step on the same inputs was limited at
  min(swap_step_rel 2 %, max(swap_step_floor 0.5 %, 2 x its component error)). Now 2 x max(its component error, the
  bf16 resolution of its output); swap_step_rel only when the component error is unknown. The worst-row, ratio and
  bias limits are still fixed (owner's thresholds review pending).
- Why (DeepSeek-V4.1 S.swa.06, OV-3 answered "disagree" with evidence by C.swa.ffn_collapse's agent, DS41-20): the
  fast window attention's precision model is 2.2 %, the step measured 2.1 %, the 2 % cap failed it.

## F74 (2026-10-15): the overseer's git use can no longer crash or fool a run

- core/gate.py `_git_retry`: the framework's git add / commit wait out another process's index.lock (up to a minute)
  instead of raising. Why: the overseer committed links.yaml while the orchestrator committed a gate; the
  orchestrator's `git add` hit the lock and the run crashed (DeepSeek-V4.1 run1, C.swa_engram.q_a).
- dashboard/links.py: the publish time lives in the run dir (`dashboard_published.json`, outside git); links.yaml keeps
  only the links. Why: every gate commit's pre-commit hooks stash unstaged changes, so an uncommitted publish time
  read as the old committed one for seconds (false DASHBOARD STALE twice). The overseer never commits during a run.
- Selftests: test_publish_time_lives_in_the_run_dir, test_git_commit_waits_out_an_index_lock.

## F75 (2026-10-15): on a smaller box the run IS the scaled model

- Owner: "the deployment box is ONLY for computing the scale factor; once computed EVERYTHING goes scaled: goldens,
  model (fewer experts, fewer heads), smaller chunk." F64 had built two tracks on a smaller box (real-size tests plus
  a separate scaled run, SG.1 golden / SL.* / SX.1 / a second fork capture); the overseer proposed it at intake without
  calling it out. Removed.
- core/spec.py: `scaling` (partial, fewer chips than the deployment box); once plan.yaml exists the spec IS the
  scaled view (scaled checkpoint, golden_scaled/<box>, -scaled cache) with no env; BRINGUP_SCALED=0 forces the real
  checkpoint. The target / ladder are written in this box's units; validate() reports a target chunk that is not the
  scaled chunk. An old `scaled.ladder` is still read.
- plan/ledger_gen.py: on a scaled run SG.1 (cut the checkpoint, after PL.1) comes before the goldens, which use it;
  no SL.* / SX.1 / scaled capture; R.* and PL.1 gates carry BRINGUP_SCALED=0. reference/generate_golden.py: an
  existing golden that matches the rung is reused (any run), a mismatch stops.
- Skill: "The rule for a box smaller than the deployment box (read this first)"; roles / agent definition / briefs:
  every agent on a scaled run is told the run is the scaled model and to read dimensions from its config.
- Selftests: test_scaled_spec_view, test_ledger_of_a_scaled_run, test_golden_full_layers (reuse).
- F75 (2): plan/scale.py copies the checkpoint's subfolders without weights (a model's own code and configs, e.g.
  DeepSeek's inference/) and rewrites the extra configs scaling.yaml `config_files` names with the kept counts (every
  run; a config-only change does not re-cut the weights). It never rewrites a file in place: a hard link to the real
  checkpoint (an agent had linked DeepSeek's inference/ files) is broken first. Why: the frozen serving-contract tests
  read inference/config.json directly and failed on the scaled checkpoint (C.swa.attention). Selftests: 2 more.

## F76 (2026-10-16): every component and swap limit from the noise floor

- Owner: "we should really not have fixed thresholds like 1% … they should be derived from the noise floor coming from
  using the exact dtype with the particular cpu torch op". F72/F73 moved two limits (the swap tail, the swap step's
  rel) onto the noise floor; the rest stayed fixed, and S.swa.11 (DeepSeek-V4.1 run1) failed only a fixed PCC 0.99 of
  the router vs the golden (0.9887 after 10 device steps; its noise-floor checks passed with 10x margin). The agents
  then raised the attention precision above the reference (fp32 scores, TF32-rounded q) to clear it.
- testing/noise_floor.py: value / gate_limit / bf16_res. A limit = calib x the precision model's miss on the same
  comparison (calib squared for 1 - pcc), never below one bf16 rounding of the output or one element of an integer
  output.
- Component tests (component_checks.Expect): the precision model runs on every input for every output kind. The golden
  gate's limit (THRESHOLD None) is the noise floor vs the golden, recorded as `<metric>_limit`; float rel / row /
  ratio / bias / per-column limits are calib x the model's statistic (vs the golden: the model's own vs the golden)
  with no cap and no floor; selection (overlap, matched-row rel, row sums) and index (overlap, worst row) limits
  likewise. The second-input probe floors stay: they only loosen (structural probes on synthetic scales).
- Swap tests: every swapped step's precision model on its own inputs gives its vs-CPU PCC limit and its rel / row /
  ratio / bias / reselect limits (rel never below its component-gate error); the block run again with its swapped
  steps as their precision models from the golden input gives each step's vs-golden limit and pcc_swap_out's
  (`pcc_swap_out_limit`).
- core/gate.check_metrics: a condition `>= limit` compares the metric with the `<metric>_limit` the test recorded.
  ledger_gen writes it for C.* and S.*. The spec's fixed thresholds remain only as the fallback when the precision
  model cannot run (printed). The end-to-end bars (pcc_layer, pcc_state_min, top5) are still fixed: next.
- Selftests: test_swap_checks.py (test_every_swap_limit_comes_from_the_noise_floor), test_component_checks.py.

## F77 (2026-10-16): fix jobs; the implement brief shows a task's details; an abort is not charged as state.json

- Owner: a fix must not wait for a failing test or the run's end ("a fixing step can't be injected on the next pause,
  what the fuck"). DeepSeek-V4.1 run1: the overseer's note to store the experts as bfp8 at load never reached an agent,
  because every later experts step passed without one.
- `overseer fix --gate TASK ... "text"` queues FIX.<n> in overseer.yaml; at its next task boundary the orchestrator
  places it before the first unpassed task (step overseer_fix, role overseer: a bring-up engineer, no precheck), its
  gate = the --gate tasks' gates chained, paths = theirs; a stopped fix stops the run; fix tasks keep the plan approval
  (approvals.ledger_signature skips them). Role text `overseer` in briefs/roles.yaml.
- The implement and fix roles now show the task's `brief.details` (FIX.1, injected by hand as an implement task,
  started without its instructions).
- run_agent took the tree diff after recording an abort, so the orchestrator's own state.json write was charged to
  the aborted agent ("changed files outside the allowed paths (revert them): state.json" in the next brief).
- Selftests: test_overseer.py.

## F78 (2026-10-16): the noise floor at the reference's own dtypes, per step

- Owner: "match the reference precision, that's the general guidance". F76's precision model rounded every intermediate
  to bf16; the official code computes some steps in fp32 (DeepSeek-V4.1's router: linear(x.float(), w.float()),
  softplus, top-k, weights all fp32), so their limits were far too loose (router vs CPU PCC limit 0.27, reselect 1.67).
- `<bringup_dir>/precision.yaml` (component_checks.step_precision): per step (or `<block type>.<step>`) `compute: fp32 |
  bf16`, `out: fp32 | bf16`; precision_model runs fp32 steps without intermediate rounding and keeps an fp32 output.
  The reference role writes it from the official code; only the reference and overseer roles may change it (it sets
  how strict every limit is); gates commit it. A step mixing both (fp32 scores, bf16 probabilities) stays bf16.
- Selftests: test_component_checks.py::test_the_precision_model_follows_the_references_dtypes.

## F79 (2026-10-16): a fix job's gate runs each step's tests as that step; F76 fixes from the first device gate

- FIX.2 (DeepSeek-V4.1 run1) reran C.swa.attention's gate under BRINGUP_TASK=FIX.2: its contract test picks its layers
  by `C.<block type>.attention` in BRINGUP_TASK, fell back to every layer and hit the half-built layer 2. Now each
  segment of a fix job's gate runs with `BRINGUP_TASK=<its task>`, and the gate runner's BRINGUP_RECORD_TASK (=
  FIX.<n>) decides the results file (core/metrics.record_task; profile.py writes under it too).
- Frozen swap tests (swa_engram 14-18) patch component._check_step with an eight-argument wrapper: F76's two new
  arguments now travel in component._STEP_CTX; the call keeps the old signature.
- The bf16-resolution floor came from the golden (already bf16: 0) and set a vs-golden limit of PCC 1.0 for an exact
  step (engram_kv); it now comes from the fp32 CPU output (the step on the same inputs; the block's last step).
- Selftests: test_overseer.py (record task, the fix gate's prefixes), test_swap_checks.py (eight-argument wrapper).

## F80 (2026-10-16): the noise floor = max(the official arithmetic, the input nudge); explained selection flips

- The precision brainstorm (DeepSeek-V4.1 run1, owner-approved; /localdev/dnijemcevic/bringup/
  handoff_precision_noise_floor_2026-10-16.md sections 6-9) measured F76's bf16-everywhere model as wrong both ways:
  2x too loose on norms, ~10^4x on the hc mixes (the official code is fp32 there), and near-empty for the router
  (PCC limit 0.27). The target stays the fp32 CPU step; its floor, per statistic, is the larger of
  - the reference arithmetic: the official code at its shipped dtypes on the same inputs vs the fp32 step, tapped per
    step by the model hook official_taps and cached next to the component goldens (testing/official_floor.py,
    official_floor.json; task R.5, which also checks the tap map: the fp32 step on the tapped inputs must give the
    official output within calib x its input-nudge floor), and
  - the input nudge: the fp32 step on its inputs moved uniformly inside their rounding cell, in the dtype the official
    code holds each input (official_taps.yaml `inputs`: bf16 / fp32 / fixed for a quantizer's output), 3 seeds, the
    worst (defaults.yaml noise_floor.seeds).
  The vs-CPU resolution floor is one rounding to the official output dtype (`out`: bf16 / fp32; none for fp32).
- Selections: the CPU step's own top-k is tapped (noise_floor.TopkTap); a device flip must be explained (both swapped
  entries within noise_floor.flip_spread = 2x the nudge's score spread of the cutoff) and the flipped rows are at
  most calib x the nudge's. A selection step without torch.topk keeps the count limit only.
- Swap tests: every swapped step's floor is its nudge on the same inputs (+ the cached official term); the vs-golden
  limits and pcc_swap_out come from the block run with every swapped step nudged, once per seed. The freeze sweep's
  controls are now a fresh-seed nudge and (bf16 outputs only) a bf16 output; the bf16-everywhere control is gone.
- Removed: F76's _Bf16Everywhere precision model and F78's precision.yaml / step_precision.
- Not yet (handoff section 7): end-to-end limits from the official run's drift (pcc_layer, pcc_state_min, top5, the
  contract's pcc_producer_kv_*); judging a step at the next stored tensor; the fp8 flip rule at attention.
- Selftests: test_noise_floor.py (nudge, dtypes, flips, no-topk selections, the official floor and its check);
  test_component_checks.py, test_swap_checks.py updated.
- R.5's first run (DeepSeek-V4.1): 129 of 136 steps within the check, 7 out: every stateful one past the first layers
  (compressors 1.04-1.23x, attention 1.04-2.3x at layers 3-24): the fp32 step read the golden's prefix caches while
  the official output came from the official run's own. The hook now also returns each layer's official state
  (`_state`: start / end, the reference's format); official_floor uses it (and the sources' end state) when given.
- With the official state, attention passed (L3-L24 0.032-0.036 vs 0.055-0.075); the 3 compressors stayed out (rel
  0.037 vs 2 x the nudge 0.015): the official compressor rounds to bf16 inside, and the fp32 step with those roundings
  emulated reproduces it bit for bit (the agent's evidence). The check now passes a step over calib x its nudge only
  with an `explained` entry in official_taps.yaml (the official arithmetic is then its floor; the overseer reviews).

## F81 (2026-10-16): gate the rel error, not PCC

- Owner (DeepSeek-V4.1 run1): switch the gated accuracy metrics from PCC to rel L2. PCC is blind to a scale or an
  offset (a 2 % scale keeps PCC at 1; the hc mixes' -4e-4 bias), and 1 - pcc ~ rel^2 / 2 squashes the scale.
- Component tests gate `err_<step>_L<layer>` (rel L2 vs the golden for float outputs, the missed fraction 1 - overlap
  or match for integer ones) `<= limit` (calib x the floor's rel vs the golden: the input nudge, the official
  arithmetic, one bf16 rounding). Swap tests gate `err_swap_out` (the block out's rel vs the golden, calib x the
  nudged block's) and every swapped float step's `_vs_golden_err`; vs the CPU step the per-statistic checks (rel, row,
  ratio, bias, flips) already gate. PCC is still recorded (pcc_<step>_L.., pcc_swap_out, the trail) as a diagnostic.
  A frozen test's literal THRESHOLD still gates its PCC too. component_error reads err_ (the rel itself) first.
- Swap tests' CPU time: the floor reruns every swapped step; noise_floor.swap_seeds (1) for the earlier steps and the
  nudged block, the full seeds for the newly swapped step (DeepSeek-V4.1 S.swa.15 on CPU: 35 s before F80, > 300 s
  with 3 seeds everywhere, 148 s now).
- Not yet: the end-to-end gates (pcc_layer, pcc_state_min, top5, the contract's pcc_producer_kv_*), with their floor.
- Observation for the precision work: downstream of a router, the nudged block's vs-golden rel is dominated by routing
  flips (S.swa.15: router 0.185, experts 0.154, block out 0.023), so those vs-golden limits are loose; the same-input
  checks (explained flips, matched-row rel) are the tight gate there.
- Selftests: test_component_checks.py (a 2 % scale: PCC 1, err over its limit), test_swap_checks.py.

## F82 (2026-10-16): verify: re-gate passed tasks without touching the run

- `python -m models.demos.common.bringup.core.verify --spec S --tasks GLOB ...`: runs the gate commands of the matching
  PASS tasks as their gates would, metrics and logs into runs/<run>/verify/<stamp>/ (summary.json); state.json,
  results/ and the commits are untouched. After a change of limits (F76-F81) the overseer reruns every passed step's
  frozen tests once and turns each failure into a fix job (overseer fix), instead of a gate run that would flip the
  task to FAIL mid-run or overwrite its committed results. Selftest: test_verify.py.

## F83 (2026-10-16): every row and column against its own floor; the signed additive bias

- Owner: rel is unreliable at small magnitudes (worst row / per column behind a fixed 1e-6 floor, near-zero rows
  "equal"), and the worst row's limit came from the floor's worst row (one tiny noisy row set it for all).
  noise_floor.own_floor: per row (and per column), ||dev_r - ref_r|| / max(the floor's ||nudged_r - ref_r||, one
  rounding of ref_r), times 1 / (1 + sqrt(2 ln 2N / m)) for the worst of N rows of length m: no division by a small
  norm, no fixed floor, an exact row counts 0. Additive bias per column: |mean_r(dev - ref)| over the larger of the
  floor's |mean|, and sqrt(2 ln 2C) x rms / sqrt(rows) of the floor's and of one rounding's error (the largest random
  mean of C columns). The official arithmetic's excess over the nudge (whole-output) widens every row's floor.
- Gated <= calib (x the probe factor on second inputs; columns and bias on the golden input only) in component and
  swap tests; they replace the worst-row, row-norm-ratio and per-column limits. Kept: whole-output rel, the median
  row-norm ratio (a scale bias), selections' explained flips.
- Synthetic check (2048 x 256, 5 seeds): a correct output scores <= 1.3; a 1 % row, an error on a 1e-5 row, a 3e-4
  column offset each fail at their own place. Selftest: test_noise_floor.py.

## F84 (2026-10-16): the overseer's "stopped because" on the dashboard

- `overseer status --spec S "why"` writes runs/<run>/OVERSEER_STATUS.json; the dashboards show it (standard: the
  snapshot line; teletext: HOLD); the orchestrator clears it when it starts. The owner saw a dashboard "still on R.5"
  for an hour while the overseer re-verified passed steps with the run stopped.

## F85 (2026-10-16): owner waivers of a step's noise-floor checks

- Owner (DeepSeek-V4.1 run1): no multi-pass / split / exact matmuls to match an fp32 reference; "if it doesn't work,
  let it be, continue and we'll fix it later". Spec `tests.waivers: {<step> | <block type>.<step>: reason}`: the step's
  noise-floor checks still run and print; a failure is reported `WAIVED` with the reason and recorded
  (`<metric>_waived`), the gate passes it (core/gate.check_metrics) as long as the output is whole (finite, the right
  size). Component tests, every swapped step's checks, and the tail check of a waived last step. Only the owner adds
  a waiver (a spec edit, re-approved); the dashboard and the end-of-run report list every waived step.
- Selftest: test_noise_floor.py::test_an_owner_waiver_passes_a_steps_noise_floor_checks_and_records_it.
- First device check (DeepSeek-V4.1, 6 passed steps): attention passed on its real inputs (row 1.76, column 0.85,
  bias 1.22 x its floor) but its `small` probe (inputs at RMS 1e-3) had one row 36x: at synthetic scales the device's
  absolute limits (fp8 scale floors, epsilons) dominate and the relative nudge does not model them. The own-floor
  checks now run on inputs with real value scales only (golden, chunk0, another layer, mixed); `small` / `big` keep
  their structural-probe limits.

## F86 (2026-10-16): the bring-up forks first

- Owner: agents must look into the bring-up ops (ttnn/ttnn/bringup/INDEX.md) and fork / extend / optimize them before
  looking in TTNN. DeepSeek-V4.1 FIX.4 was sent (by the overseer) to fork ttnn.experimental.deepseek_prefill.
  mhc_split_sinkhorn while ttnn.bringup.mhc_pre_xing (mhc_pre_ttnn, AI-generated) already had an fp32 SFPU Sinkhorn.
- Agent rule 16 (forks first; then TTNN, forked when it needs changing; composing last, with the reason); the plan
  role maps components to the forks first; the skill: check INDEX.md before a note or fix job names an op, and in
  every plan / gate commit review.

## F87-F89 (2026-10-16): resumed fix jobs gate first; the dashboard never paints a passed step red; own-floor warns
- F87: a task resumed after a stop (a fix job too, precheck False) runs its gate before an agent: FIX.4's restart
  started an agent on work already done (about 30 min). orchestrator.py, selftest test_a_resumed_fix_gates_before_an_agent.
- F88: F81 renamed every gate metric (pcc_* -> err_*), so the 85 steps passed before it had no metric of the new name and
  the page painted them red (the word 'limit' reached the page). resolve_thresholds never passes 'limit'; a passed step
  gated under an earlier rule gets a note. export.red_passed (the page's own check) writes dashboard/CHECK.txt; the
  watch reports DASHBOARD RED; STALE_AFTER 15 -> 2 min.
- F89 (owner): the own-floor row / column / additive-bias checks (F83) print WARN and record <tag>_floor_warn; only the
  overall error (and the other existing checks) gate. S.swa.15 failed on kv's column bias 2.27 x (limit 2) and the
  router's own floor (1e3 x, likely a flaw for selection outputs: review later).
- F90 (owner): noise1e-2 dropped from the mistake sweep (within the noise limit by design).
- F91 (owner): the sweep marks an uncaught mistake "within limit", not SLIPPED, when its own size (rel vs the CPU
  output) is within the step's rel limit on every input: a correct step may be that far off. DeepSeek-V4.1
  compressor: scale1.02 / 0.98 vs rel limits 0.031-0.042 started a test-role agent for nothing.
- F92: the block-out check after the last swapped step (_check_tail) takes the noise floor for int outputs too (a
  top-k): the nudged fp32 step's own selection flips. Was the fixed 0.01, which the CPU's own nudged indexer misses
  (0.026); DeepSeek-V4.1 S.enc_source.07 failed 6 attempts at 0.023 (DS41-50, the debugger's probe_006).
- F93 (owner): heartbeat in three layers, set up by every overseer session: the watch (default --timeout 1800, 30 min), a 30-min CronCreate timer, and the Stop hook skill/bringup/heartbeat_stop.py (no idle session on an unwatched or silently stopped run). The skill says to check the run on every
  wake and re-arm after every event. A stop at 20:33 went unseen: a DASHBOARD STALE wake was not re-armed.
- F91 fix: the sweep's "within limit" compares with the limit check() applies (second inputs: at least
  component_second_rel), not the bare rel_lim (DeepSeek-V4.1 attention: `small` rel_lim 0.0057, check limit 0.03).
- F94: after the overseer aborts a test agent, the orchestrator re-renders the template and retries the freeze
  without review before any further test agent (the overseer may have fixed the check that sent it there).
- F95 (owner): resumable after any kill. Start clears a stale AGENT.json / ABORT.json; `resume` records which role and
  attempt a crash cut off; attempt numbers continue (attempts_so_far: no brief or transcript overwritten) and the
  next brief says the attempt was cut off; a test frozen without review is always rendered afresh (an aborted test
  agent's half-written coef check survived a restart and sent a second agent to fix it: DeepSeek-V4.1 attention).
  Skill: stop doomed agents at once, the run is resumable.
- F96: the scaled probes (small RMS 1e-3, big x 2) gate rel at component_scaled_rel 0.2 (structural mistakes: 0.5-2):
  at tiny scales a correct device meets absolute floors (fp8 scale, eps) the nudge model does not cover. DeepSeek-V4.1
  attention L2: small 0.035 > 0.03 with every other input well inside; golden 0.035 vs 0.16. Every standard mistake
  is still caught (sweep selftests).
- F97: an owner waiver passes a step only when its output is whole AND within component_scaled_rel (0.2) of the
  reference: it covers a precision gap, never a wrong module. F85 passed any finite output, so DeepSeek-V4.1 ffn_hc
  (waived) passed with a zero stub and its freeze sent a test agent to a test it could not fix.
- F98: the after-step (tail) floor of a float step also runs the tail from the CPU step output rounded to bf16 (a
  correct device's output is at least that): a top-k downstream flips near-ties on it. DeepSeek-V4.1 S.enc_source_
  engram.02: engram_kv 8e-5 vs the CPU, block-out tail 0.041 vs a 0.0033 limit (the nudge moved nothing: floor 1e-6).
- F99: ladder states: integer states (top-k index lists) are checked by the mean per-row selection overlap with the
  golden (thresholds.state_overlap 0.9; a wrong selector gives ~k/n), not by the PCC of the ids; pcc_state_min covers
  float states only. DeepSeek-V4.1 M.1 precheck: topk_idx PCC 0.52-0.78 while every float state was >= 0.9959.
- F99 (adjusted): state_overlap 0.75, not 0.9: the check is structural (a wrong selector gives ~k/n <= 0.5); near-tie
  flips grow with the pool (DeepSeek-V4.1 L.s55296: 0.897 while every layer stayed at PCC 0.9966). The fix agent
  that 0.9 started was stopped.
- F100: serving contract with a non-contiguous layer subset (a partial run's representative layers): served as
  count engine layers 0..count-1, the model layer of each in PREFILL_LAYER_IDS (engine_env); adapters map with
  contract.layer_ids_from_env and expose rt.layer_ids, which the contract checks. Was a ValueError (DeepSeek-V4.1 K.1,
  layers 0-3,14,20,21,24).
- F101: tie-aware top-k recall (owner, after Momcilo's MoE gate rule), recorded by default for every top-k output
  (router weights and index lists) in component and swap tests: the reference's scores are tapped from every top-k
  the CPU step takes (row blocks stacked, matched to the output by holding all its ids), and recall_tie counts a swap
  whose both scores lie within noise_floor.tie_rtol (0.002) of the cutoff as a hit. Recorded, not gated (F76: no fixed
  bars); index lists now also get the F80 "every flip explained" gate. The nudge band is not a tie rule: with an fp4
  query it spans ~650 of the 512-of-3072 row (median +-94 % of the cutoff). Ladder index states stay plain overlap
  (no scores in the golden). DeepSeek-V4.1 device, vs the CPU step: indexers recall 0.9887-0.9996, recall_tie
  0.9911-0.9998, 0 unexplained; routers recall_tie 1.0 everywhere.
- F102: the input of a served layer run after a gap (a partial run's non-contiguous subset under the contract) comes
  from the framework, testing/contract.gap_feed (the golden block input of the run's first layer, as the ladder
  feeds it); None without PREFILL_LAYER_IDS or for a contiguous range. DeepSeek-V4.1 K.1 had put the golden reader in
  the model's adapter (GoldenGapFeed): a model never reads a golden, even behind a switch.
- F103: a perf pick's ab switch may name a model setting (tt/settings.py): the A/B report sets its variable (prefix +
  name) and refuses a value outside its choices; other names stay plain variables the code must read. DeepSeek-V4.1
  P.1 named EXPERTS_IMPL, the code reads DS41_EXPERTS_IMPL: both sides ran bfp8 and the table was identical.
- F104: accepting a perf pick whose ab.change is a model setting ships it: its default in tt/settings.py becomes the
  accepted value (rewritten in place via the AST) before the accept gate, so the gate, the commit and every later task
  run the change. DeepSeek-V4.1 P.2's opt-in EXPERTS_KERNEL stayed `loop`: the accept gate measured the old path.
- F105: the accept gate reuses the A/B report's 'on' side (same commands, same code, every command rc 0, the gate's
  rung among the A/B rungs) instead of running again; the code key now hashes the committed files minus the run's own
  records (state, results, breadcrumbs, supervision, tasks.yaml), so a ledger commit in between is no code change.
  DeepSeek-V4.1 P.2: the accept re-ran ~50 min of tests the A/B had just measured.
- F106: the settings lint on a fork flags only the environment reads a change adds (vs HEAD): an imported fork's own
  knobs are not the editing agent's. DeepSeek-V4.1 P.3: 173 flags on Marko's MIMO_FL_* reads after an edit to
  flat_routed_expert; attempt 2 began converting them (aborted, OV-15). Model code stays strict.
- F107: a perf pick's agent attempts continue the numbering across a resume or a new brief (F95's rule, which the
  perf loop missed): DeepSeek-V4.1 P.3 restarted at attempt 1 after an overseer stop, reusing attempt 1's brief name.
- F108: a perf pick's A/B report measures every uncommitted change under the task's paths (any attempt's), not only
  the last attempt's edits: DeepSeek-V4.1 P.4 attempt 2 (the code already in the tree from attempt 1) was reported as
  "the agent changed nothing" and nothing was measured.
- F109: a task waiting for a person no longer stops the run when other runnable tasks exist and the waiting task has
  nothing uncommitted in the tree (another task's commit would take it along); its dependents stay blocked. Owner:
  run P.5 regardless of the P.4 decision.

- F110: a gate whose commit git refuses (a pre-commit hook) is FAIL with the hook's output as its reason, not PASS
  with the change uncommitted. DeepSeek-V4.1 P.5: prefer-expect-error refused the accept gate's commit, the
  orchestrator crashed with P.5 marked PASS.

- F112 (owner rule, 2026-10-19): no parity-only rounding. Agents' rule 17 (bringup-engineer.md; rule 14 and the
  reference / plan roles point at it): a model never rounds to a format the device then neither computes nor stores
  in. `hooks.reference` may take `numerics` ("official" for the goldens, "native" with those roundings off);
  per-step checks (component, swap) use `reference.interface.step_reference`, which asks for `tests.step_numerics`
  (default native) when the hook takes it; older hooks keep their only arithmetic. DeepSeek-V4.1: ~12 fp8 / fp4
  quant-dequant passes (tt/quant.py, ~25 eltwise ops each) were added only to pass per-step checks against the
  quantizing reference.

- F113 (owner, 2026-10-19): bring-up ops before stock TTNN. ttnn/ttnn/bringup/INDEX.md is one entry per op folder
  with "Consider instead of: <stock ops>", each op folder a README.md (what it does / consider it instead of /
  options / measured / limits / how to call / used by). testing/ops_lint.py: a step that adds a call to a stock op an
  INDEX entry lists as a standalone item needs `stock <op> instead of <ttnn.bringup.op>: <reason>` in the model's
  BREADCRUMBS.md (soft: justified, never banned); a changed op folder keeps its README sections and INDEX entry;
  `python -m models.demos.common.bringup.testing.ops_lint` checks every folder. fork_op.py writes the README skeleton
  and the new INDEX entry; rule 6 / 16 and the fork-op skill say so. DeepSeek-V4.1: combine_rf (the weighted fp32
  combine) had no INDEX row, and our combine (unweighted) was followed by ~16 ops of sums.

- F114 (owner, 2026-10-19): clean wall time. The profile gate ran its wall timing in the profiled process
  (TT_METAL_DEVICE_PROFILER=1, mid-run dumps): instrumented kernels and buffer dumps; DeepSeek-V4.1's same code
  measured 787 / 925 ms wall. gate_command (core/gate.py with_clean_wall) now precedes every profiled test_profile.py
  segment by the same command without the profiler variables and BRINGUP_PROFILE_WALL=1 (compile + 3 timed runs,
  chunk_wall_ms = median, chunk_wall_clean = 1); the profiled run (BRINGUP_WALL_CLEAN=1) keeps that and records its
  own as chunk_wall_ms_profiled. The A/B report runs both and shows chunk_wall_ms and profiled_programs.

- F115: `pytest --collect-only` / `--co` is not a direct pytest (it lists tests, runs no fixture, opens no device).
  DeepSeek-V4.1 P.8 attempt 1 was failed after 2h20m of passing work for one `pytest --collect-only -q`.

- F116 (owner, 2026-10-19): a perf agent's own checks are the A/B report's change-on side. ab_measure keeps every
  run (frozen test command, ladder rung, profile) in side.json with its own stamp (code key, switch env, command) and
  reuses a matching one; `orchestrator measure --task <id> [--part <test stem | rung | profile | test | ladder>]` runs
  them as the report's 'on' side (its switch, environment and results dirs). The perf brief tells the agent to run its
  frozen tests that way after its last edit. F105's accept reuse reads the per-run stamps. DeepSeek-V4.1 P.8: the
  agent's ~1 h of frozen tests and the report's on side were the same runs, twice.

- F117 (owner rule, 2026-10-19): weights at the checkpoint's own width (agents' rule 7): fp4 -> bfp4, fp8 -> bfp8,
  16-bit -> bf16 in the shipped defaults; the exact wider format (bfp8 / bf16) in max_precision where the narrow one
  is lossy. DeepSeek-V4.1 had widened every fp8 weight to bf16 at load (2x memory and DRAM traffic).

- F114b / F119 (2026-10-19): the clean wall run times WALL_RUNS = 12 chunks and takes the median after WALL_WARM = 4
  (chunk_wall_ms_spread records the rest's range); the harness's golden-input uploads at layer gaps (a subset's
  non-contiguous layers) are timed apart (chunk_wall_gap_ms) and subtracted (an approximation when device-bound: the
  device may run queued work during an upload). DeepSeek-V4.1 P.8: 3-run medians gave 861 (new, still warming: the
  experts + combine program keys its cache on buffer addresses) vs 785 (old); 25 runs: ~546 vs ~845; the gap uploads
  were ~280 ms of every run.

- F120 (owner, 2026-10-19): the accept reuse takes each gate run from the side that measured its configuration: the
  frozen tests run at max precision, so per switch the max-precision value (else the on value); when that is every
  switch's off value, the tests come from the off side (gate_tests_side), the ladder rung and profile from the on side.
  The A/B code key ignores files that cannot change a measurement (NO_MEASURE: the orchestration, agents, briefs,
  skills, selftests, dev notes, dashboards, knowledge, any markdown). DeepSeek-V4.1 P.8: the on side's swap test failed
  by design (rounding mismatch) and the overseer's framework commits had changed the key: the accept would have rerun
  ~1 h of tests the A/B had already run; done by hand once. WALL_RUNS 20 / WALL_WARM 10 (P.8: ~10 warm-up chunks).

- F121 (owner, 2026-10-20): when max precision pins every switch of a pick (switches_pinned) to one side's values
  (gate_tests_side), the A/B runs the frozen tests once, on that side, and shows the result for both: they cannot see
  the change. DeepSeek-V4.1 P.9 (COMBINE_ACC, pinned fp32, with EXPERTS_COMBINE pinned separate): ~40 min of identical
  tests per side.

- F122 (owner, 2026-10-20): quick A/B by default (defaults.yaml perf.ab_mode: quick; a task's ab_mode overrides).
  One process a side: BRINGUP_QUICK=1 BRINGUP_PROFILE_WALL=1 test_profile.py (the last chunk after the golden prefix,
  the clean wall of F114b, then every layer vs the golden: pcc_layer_L*, pcc_chunk_out, host transfers). The accept
  checks the thresholds the quick run measured (measured_only) and marks the pick full_pending; before the final
  scripted perf task (X.3) the orchestrator runs every pending pick's full gate once, picks with the same gate command
  sharing one run (pending_full_gates); a failure stops the run. The perf brief makes `orchestrator measure` the
  agent's final check. DeepSeek-V4.1: a full A/B was ~2 h 20 min a pick (16 frozen tests, two rungs, profile, both
  sides); the owner: "it takes a day only to try a few simple things".

- F122b: the dashboard narrows a quick-accepted pick's (full_pending) thresholds to the metrics its quick A/B measured
  (the rest wait for the full gate before the final task); without it the watch stopped on "DASHBOARD RED: P.9:
  pcc_state_min missing; device_ms_experts missing".

- F123 (owner, 2026-10-20): realistic precision from the start. Agents' rule 7: the shipped defaults are realistic
  from the first step (checkpoint-width weights, HiFi2 / LoFi, fp32 accumulation only where measured to matter, stock
  kernels unless a measured end-to-end gap, bf16 cross-chip sums; reference GLM-5.3 production); max_precision is the
  per-step tests' bug-check setting only (rule 14). The bringup skill's intake asks for the reference point.
  DeepSeek-V4.1: built at max precision, then P.4 / P.6 / P.7 / P.10 / P.13 to get down to realistic; the
  sparse_sdpa high-precision options existed only to beat the parity fp8 quantizers (removed in P.8).
- F124 (owner, 2026-10-21): an owner waiver of the ladder's index-state overlap. `tests.waivers` key `ladder.state_overlap`
  (reason) lowers the limit from `state_overlap` to the structural floor `state_overlap_waived` (0.5), reported WAIVED with
  `overlap_state_min_waived`; a wrong selector still fails. Origin: DeepSeek-V4.1 X.3, native numerics (F112) at 55k gave
  L24 overlap 0.695 vs goldens whose selection used the reference's fp4 index q-dq (NUMERICS=official: 0.875, PASS); every
  layer PCC >= 0.9944. Proper fix: F118 (goldens follow NUMERICS). Selftest: test_ladder_states.py.
- F125 (overseer, 2026-10-21): X.3 asks for `prefill_ms_full` only when the run's layers are a stack from layer 0 with no
  gaps (the condition `profile.full_prefill` times it under); a gapped subset (DeepSeek-V4.1's 0-3,14,20,21,24) could never
  record it, so X.3 failed MISSING after every test passed. Selftest: test_plan.py.
- F126 (overseer, 2026-10-21): a fix job gated on a task that has not passed yet takes that task's deps and runs before
  it; the task gains the fix as a dep. Before, the fix depended on its gate task, so the unpassed task re-ran its known-
  failing gate first (DeepSeek-V4.1 X.3: 45 min) and could exhaust its attempts before the fix started.
  Selftest: test_overseer.py::test_a_fix_gated_on_an_unpassed_task_runs_before_it.
- F127 (overseer, 2026-10-21): a fix job carries its own write paths (`overseer fix --paths`, default the model's tt/ dir)
  on top of its gate tasks' paths. DeepSeek-V4.1 FIX.5 was gated on X.3, a scripted task with no paths, so its agent could
  not edit tt/ and stopped without a change. Selftest: test_overseer.py.
- F128 (owner, 2026-10-21; the end-to-end half of F80, deferred on 10-16 and lost): the device is judged against the
  full-precision reference; the reference at its own shipped precision only sets the margin, per step AND end to end.
  - Goldens: `golden.numerics: native` (fp32 compute, no quantize-dequantize, stored fp32); `generate_golden --set`
    (the G tasks) also makes `<rung>-official` (numerics official at `reference.shipped_dtype`, default bf16) and
    `drift.json` (reference/drift.py: per layer, state, top-k overlap, final hidden, top-5; plus a bf16 resolution
    floor). content_hash leaves out the derived drift.json / official_floor.json.
  - Limits (testing/e2e_limits.py): 1 - e2e.calib (2.0) x max(1 - drift, 1 - res); ladder (pcc_layer_L*,
    pcc_state_*_L*, overlap_state_*_L*, pcc_final_hidden, top5_overlap) and contract (pcc_producer_kv_*,
    pcc_contract_state_*) record `<metric>_limit`; ledger_gen writes `>= limit` for M.1, L.*, K.1, X.3, Z.1. The fixed
    thresholds layer / state / state_overlap / final_hidden / top5 are gone (defaults.yaml); F124's waiver is gone.
  - No silent fallback: reference.golden.reference_at stops when hooks.reference has no `numerics` switch and the run
    asks for native (unless `reference.no_rounding: true`); per-step references (interface.step_reference) go through
    it too. F112 had fallen back to the rounded arithmetic, so DeepSeek-V4.1's per-step checks were rounded as well.
  - Docs: the skill's accuracy section, agents' rules 14 / 17 and the reference role, the hooks template.
  - Guards (selftest/test_e2e_limits.py): the limit math, no fallback, drift on synthetic goldens, no generated device
    gate with a fixed accuracy number, no threshold() call in ladder / contract, defaults at native.
- F129 (overseer, 2026-10-21): state.json keeps only each task's gated metrics and their limits (core/gate.state_metrics;
  results/<task>.json keeps every metric, and the dashboard reads both). Ledger._save_state trims older entries on the
  next save. DeepSeek-V4.1 FIX.6's gate passed every check and then failed its commit: state.json 502 KB > the repo's
  500 KB large-file hook (128 KB of it were ungated metrics). Selftests: test_core.py, test_dashboard_limits.py.
- F130 (overseer, 2026-10-21): the dashboard notes, instead of painting red, a passed task whose gate gained a fixed
  check after it passed (F128 added drift_metrics to the G tasks); the check runs at the task's rerun. Before, the
  export's consistency check flagged it as an export bug and the watch woke at once on every arm.
  Selftest: test_dashboard_limits.py::test_a_fixed_check_added_after_a_pass_is_a_note_not_red.
- F131 (overseer, 2026-10-21; audit V2-V4): F128's limits at run time.
  - Contract (testing/contract.producer_limits): a producer read-back key maps to the golden state it reads (itself when
    it is a state name, else KV_STATE_NAMES k -> key, v -> value, kvpe -> kv_latent, overridden by the adapter's
    `kv_state_names()` or hooks.kv_state_names(spec)); a key with no state in the drift fails the contract with the
    fix named. Was E.state_limit(drift, "k"): KeyError for Gemma-4 / MiMo (k / v vs key / value). e2e_limits.state_limit
    matches the state name exactly (a prefix match took index_key's layers for "index").
  - harness.threshold(s, "state" | "layer" | ...) raises a KeyError naming e2e_limits (was a bare KeyError 'state').
    DeepSeek's and Xing's runner contract tests (server_rules.state_threshold) take per (state, layer) the stricter of
    the server's prefill-golden PCC (the server's own rule) and calib x the drift on their golden rung.
  - The quick A/B profile records each layer's and each state's limit (E.check, ladder.check_states on the last chunk):
    a `>= limit` gate could never pass a quick pick ("no pcc_layer_L00_limit recorded"). The A/B report's quick and
    ladder results carry {metric: [value, limit]} and the table lists value / limit / share of the margin used (worst
    first).
  - Selftests: test_e2e_runtime.py (key map, unmapped stop, removed thresholds, no threshold() in any model's contract
    tests, quick profile limits), test_perf_ab.py::test_a_quick_ab_judges_limit_gates_on_the_limits_it_recorded.
- F132 (overseer, 2026-10-21; audit V6, V11, V12): one entry point into the CPU reference.
  - reference.golden.reference_at(spec, numerics, *, layers, dtype) is the only caller of hooks.reference; the hook
    must name a `numerics` parameter (a `**kw`-only hook stops: it would take the switch and may ignore it). A hook
    without one stops for either numerics unless `reference.no_rounding: true` (F136 checks such a model has a margin).
  - The target is a constant (golden.TARGET_NUMERICS = native): `golden.numerics` and `tests.step_numerics` are gone
    from defaults.yaml and Spec.validate refuses them (core/spec.REFUSED). `official` is reached only from
    generate_golden --numerics official (the G task's --set) and its drift.
  - official_floor (R.5) compares the official output with the native step (was the hook's default arithmetic, not
    the step the tests use); op_request's golden function is native; ledger_gen / check_plan read the native graph.
  - R.2 (check_hf) gates the official numerics against HF (the HF code is the official arithmetic) and records the
    native target beside it (native_pcc_hidden_L*, ...); R.3 (check_reference) checks both (plain names native,
    official_ prefix; ledger_gen gates both).
  - Selftests: test_step_numerics.py (rewritten: native always, **kw stops, spec keys refused, AST: no `.reference(`
    call anywhere in the framework), test_e2e_limits.py.
- F133 (overseer, 2026-10-21; audit V7, V8): every resolution floor is one rounding to the tensor's real dtype.
  - noise_floor.round_to(t, "fp32") moves t inside its fp32 rounding cell (float64, seed 0; was the identity, a 0 floor);
    rel / float_errors / col_errors / step_errors compute in float64 so the move survives.
  - Per step, vs the full-precision golden (F128 stores fp32): `gres`, `gold_res`, the selection floor, the gate's
    resolution and the swap tests' vs-golden limits round the CPU step's output to its official `out` dtype
    (official_taps.yaml), not bf16. An `out: fp32` step (router, hc mixes, compress_partial) was 10^3-10^4 too loose
    vs the golden (the floor F78 / F80 removed vs the CPU). The "golden is already bf16" comments are gone.
  - End to end (reference/drift.py): a float metric's floor is one rounding to the dtype the official golden stores
    that tensor in (bf16, fp32 for golden.keep_fp32); an overlap's is one miss among the selected entries (rows for
    top-5). `e2e.overlap_res` (a fixed 1 %: every overlap limit <= 0.98) is gone.
  - Selftests: test_noise_floor.py::test_an_fp32_steps_golden_limit_is_one_fp32_rounding_not_bf16,
    test_e2e_limits.py::test_the_drift_floor_is_one_rounding_in_the_stored_dtype_and_one_miss_per_selection.
  - Existing drift.json files keep their old floors until regenerated (`reference.drift --rung R`).
- F134 (overseer, 2026-10-21; audit V5, V26): the drift is measured on the comparison it gates.
  - reference/drift.py replays the official reference (numerics official at reference.shipped_dtype) the way the
    ladder feeds the device on each rung: a prefix_from_golden rung loads the golden's state prefix and runs the last
    chunk; a gapped subset starts each run from the golden block input of its first layer (written into the block
    input forward_chunk records, which run_block hands to the steps; a forward that copies it stops, checked on the
    first step). Was the official golden's own full propagation: prefix and gap drift the device never has (too loose
    for X.3's last-chunk rungs, Z.1, DeepSeek's 0-3,14,20,21,24).
  - One file per rung, `drift-<rung>.json` in the golden it reads (E.load(dir, rung)); the G task writes every reader
    rung's (`generate_golden --set`); the -official golden is no longer made (`--numerics official` still exists).
  - A prefix rung's states are compared over the positions the device computed: the entry's `region` (the axis
    state_tensors slices), applied by the ladder and the quick profile too.
  - The contract rung adds `contract_state_<name>_L<i>`: growing states over [0, seq - tail_pad rounded down to 32)
    (what the producer reads back), fixed ones after the last chunk ending there; the contract's limits use them.
  - Pinned: a G task's pass writes every drift file's sha into the frozen block of each device task gated on an
    end-to-end `limit` (drift.pin, core/gate); a drift changed outside its G task fails those gates. (The report
    proposed freeze_extra; the e2e tasks have no frozen tests, the frozen block is what the gate checks.)
  - Not done: the state PCC over [0, seq) of a full rung keeps the prefix computed by the device itself (nothing to
    exclude); the DeepSeek / Xing runner contract tests keep the rung's pcc_state limits (their chunk plans are their
    own, not the contract rung's tail).
  - Selftests: test_drift_matched.py (prefix seeding and region, gap feed, a forward that ignores the feed stops,
    contract entries, per-reader files and the pin), test_e2e_limits.py (entries()).
- F135 (overseer, 2026-10-21; audit V9): end to end, the per-step convention. The ladder and the quick profile gate
  every float tensor by its rel L2 error vs the full-precision golden: `err_layer_L<i>`, `err_state_<name>_L<i>`,
  `err_final_hidden` `<= limit`, limit = e2e.calib x max(the drift's rel, one rounding's rel); index states and top-5
  keep their overlap, 1 - calib x max(miss, one miss). PCC (pcc_layer_*, pcc_state_*, pcc_final_hidden, pcc_state_min)
  is recorded as a diagnostic, without a limit. The contract keeps PCC (the producer's read-back and the
  contract_state_pcc hook report nothing else) at 1 - calib^2 x its miss (1 - pcc ~ rel^2 / 2; F128 used calib, ~1.41x
  the per-step margin). Drift entries are per tensor (`layer_L<i>`, `state_<name>_L<i>`, `final_hidden`, `top5`,
  `contract_state_*`) with rel / rel_res / pcc / pcc_res (or overlap / overlap_res). ledger_gen gates the err_*.
  - Ledgers written before F135 gate `pcc_layer_L* >= limit` etc.: those limits are no longer recorded (MISSING);
    rename to err_layer_L* / err_state_*_L* / err_final_hidden `<= limit` (overlap_state_*, top5_overlap and
    pcc_producer_kv_* stay).
  - Selftests: test_e2e_limits.py (the convention; a 2 % scale fails with PCC 1), test_device_side.py,
    test_e2e_runtime.py, test_drift_matched.py.
- F136 (overseer, 2026-10-21; audit V12, V13): the margin's source must exist and be the shipped code.
  - The G task stops with "no margin source" when the reference at numerics official, reference.shipped_dtype moves
    no layer of the rung beyond one rounding (official identical to native, a hook that ignores numerics or dtype,
    reference.no_rounding): every limit would be one rounding of the target, the device fails by design and agents
    raise precision. Records official_identical, drift_layers, drift_layers_above_res (gated >= 1 in G tasks).
  - `reference.margin_source: hf_official` (owner-signed, defaults.yaml `reference`): the official modelling code
    (hooks.hf_model) at shipped_dtype in one pass, runs after a gap fed from the golden; layer outputs and top-5 only.
    The report proposed an `-official` golden from it; the smaller version: the drift entries directly, marked
    `source: hf_official` (`matched: false` on a prefix rung). A state or the final hidden has no drift then, and its
    gate stops with the source named (e2e_limits._entry), never a fallback.
  - R.5 also runs the reference's own official numerics on every tapped step's official inputs and requires the
    tapped official output within the tap check's limit (`official_ref_vs_taps_errors == 0`): the -official
    arithmetic shown to be the shipped code's, per step (V13: two definitions of the "real reference" are reconciled).
  - Intake asks for reference.shipped_dtype (skill; spec template). The fixture's official numerics at a 16-bit
    dtype round every step's output to it (a margin exists in every golden selftest).
  - Selftests: test_drift_matched.py (an ignoring hook stops G; hf_official), test_noise_floor.py
    (test_r5_checks_the_references_official_numerics_against_the_taps).
- F137 (owner, 2026-10-21; audit V1, V23): the device always runs the shipped realistic settings, in every gated test,
  per step included; no device run at max precision, ever, not even as a diagnostic.
  - Gone: tests.component_max_precision, model_precision.apply / overrides (component and swap tests set no
    overrides), Settings(max_precision=...) and max_precision_env (a TypeError now), F120 / F121's split (the A/B runs
    every frozen test on both sides with each side's switch; the accept reuses only the 'on' side's runs).
    DeepSeek's and Xing's tt/settings.py in this tree lose MAX_PRECISION.
  - The "logic bug or precision?" diagnosis is CPU-only (testing/devfmt.py): a failing component test runs the step's
    CPU reference at the device's real formats on the same inputs (hooks.devfmt_component(spec, layer, step): weights
    through their stored block formats, bf16 activations, the fidelity as far as the CPU models it; without the hook,
    bf16 activations only, said so) and records `<err metric>_devfmt_cpu`, `_devfmt_dev` and `_devfmt_precision`,
    never gated (the device within half its distance to full precision of the device-format step: precision).
  - Setting(emulation=...) declares the choices that only emulate a reference format (rule 17); settings_lint refuses
    one as a default and an undeclared choice named like one (qdq, tf32, emul, parity); the orchestrator runs it when
    an agent changes tt/settings.py, and in Z.1. This tree's DeepSeek snapshot flags COMPRESSOR_X round_tf32 and
    ENGRAM_KV_QDQ qdq (defaults).
  - Text: agents' rules 7 / 10 / 14, the settings audit, README, the skill (owner rules 1 and 2 in the accuracy
    section).
  - Not done: the diagnosis for a failing swap-test step (component tests only).
  - Selftests: test_model_settings.py (no max-precision setting or machinery, the emulation lint, the CPU diagnosis),
    test_perf_ab.py::test_every_frozen_test_runs_on_both_sides_at_the_shipped_settings.
- F138 (owner, 2026-10-21; audit V10, V14, V16, V20-V22): no fixed accuracy number anywhere.
  - Gone: defaults.yaml `thresholds` (component, block, swap_step_*, swap_out_rel), harness DEFAULT_THRESHOLDS and
    Spec.threshold (harness.threshold raises with the reason), tests.low_precision_kinds / component_calib_low, and
    the fixed component limits (component_rel / floor / margin / row / ratio / bias / select / select_rel / rowsum /
    index_row). Spec.validate refuses `thresholds` and `tests.low_precision_kinds`.
  - No floor means FAIL ("cannot judge: no noise floor"), never a fixed fallback: float_limits needs the floor, the
    swap step / tail / block-out checks fail without one; selection / index checks without a floor give statistics
    only.
  - The second-input probe factors are structural, renamed struct_* (struct_probe, struct_second_rel / row / ratio,
    struct_scaled_rel, struct_col_factor; the waiver's structural bound), kept apart from the accuracy limits.
  - run_component_test: checks="auto" only (the golden-only path is gone); a frozen test's literal THRESHOLD is
    recorded (`<step>_threshold_literal`, `pcc_swap_out_threshold_literal`), never gated; the freeze refuses a test
    with a numeric THRESHOLD; the templates write `THRESHOLD = None`.
  - Ledger.validate and every gate run (core/gate pre-checks, so an overseer-added task is caught too) refuse an
    accuracy metric (pcc_ / err_ / overlap_ / top1 / top5) gated on a number in a device task or a perf pick; the
    reference's own CPU checks (intake, reference, goldens) keep theirs.
  - Every ladder state gates as err_state_<name>_L<i> (index states: the missed fraction 1 - overlap, as per step);
    ledger_gen's `overlap_state_*_L* >= limit` (F128) went MISSING on every model without an index state.
  - op_request's tolerance is the step's recorded err_*_limit only (no PCC 0.99 / rel 0.02 fallback; refused
    without one).
  - Dashboard (V16): the trail is each layer's rel error (sqrt(2 (1 - PCC)) where only PCC was recorded) with the
    per-layer limit curve its run recorded; a layer is red when over its limit; the final row shows the worst layer
    with its limit. The removed D.thresholds (fixed lines at undefined values) are gone from both templates.
  - The device-format weight term (V20) is F137's CPU diagnosis (testing/devfmt.py), not a separate number.
  - Selftests: test_no_fixed_numbers.py (ledger, spec, freeze, AST, every `limit` condition recorded by its gate's
    test on the fixture for C / S / L / M.1; the committed-ledger lint is xfail strict until F140); updated
    component / swap / opgen / dashboard tests.
- F139 (owner, 2026-10-21; audit V17, V18): fork tests on a noise floor. testing/fork_cases.noise_limit(fn_ref, inputs,
  dtypes, out_dtype): calib (2.0) x the rel L2 (and calib^2 on the PCC miss) of the torch reference at the op's real
  formats (inputs rounded to bf16 / bfp8 / bfp4 on the CPU, a model of the 16-element shared-exponent blocks; fp32
  sums; the output rounded) vs fp64. A math case declares `noise` (its formats), never a pcc / rtol / atol / rel of its
  own; data movement stays `exact`. `fork_cases --lint` checks the model's own cases (O.1: fork_cases_lint == 0 in
  newly generated ledgers; other models' legacy cases are F140's). The fork-tests skill says so; the rebase skill
  judges a rebase by every metric within its limit and its move as a share of the margin, never identical bits.
  - Not done: the 122 existing math cases (their tests read case["pcc"] / ["rtol"]) are F140's migration.
  - Selftests: test_fork_cases.py (the floor on a matmul: correct passes, 1 % scale fails, bfp8 wider; the lint; the
    legacy cases as xfail strict until F140).
- F141 (owner, 2026-10-21; audit section 3, V15, V16, V19, V25, V27-V29): the text says what the framework does.
  - Skill accuracy section: the owner's six precision rules (no max-precision device path; shipped settings vs the
    full-precision reference with the real reference's precision as margin; bit-identity never a criterion; bring-up
    ops before stock TTNN; packer L1 accumulation is not fp32 accumulation unless the packer writes fp32; a deferred
    part of an owner rule queued at once); the end-to-end margin as F134 / F135 / F136 build it; no fixed number
    anywhere (F138); waivers only for a device-format limit on a correct target, owner-signed; perf picks gate
    accuracy on `limit` only. Agents: rule 7 (packer L1), rule 16 (bring-up ops, the rms_norm example), the test role
    (no numeric THRESHOLD). Briefs: test, reference (numerics, R.5) and fix roles. README, serving-contract, the
    settings audit, the prior's test text, the accept message, the spec template (no `thresholds:`, which F138
    refuses), golden.py's docstring.
  - Knowledge: K-22, K-25, K-27, K-30, K-31, K-39, K-47 and repo_map's routed experts no longer prescribe raises above
    the reference; knowledge/check.py refuses a shared entry whose Fix raises precision (HiFi4, fp32 accumulation /
    dest / intermediates, high_precision, exact kernels) without "only where the reference ..." or the owner's decision.
  - Dashboard: CPU-bridged runs are marked provisional (trail label, final row); per-layer limits came with F138.
  - Selftests: test_text_guard.py (no stale phrase in agents / skills / briefs / README / templates / knowledge /
    orchestrator text; the six owner rules present in the skill), test_dashboard_limits.py (the trail's limits, no
    fixed line).
- F142 (overseer, 2026-10-21): batched re-check. `core.verify --batch` runs the selected C.* / S.* tasks whose gate is
  one test command over their frozen files grouped by block type (and runner words) in ONE pytest process
  (`--no-precompile` for run_safe_pytest), the mesh opened and the layer loaded once per group instead of per task
  (DeepSeek-V4.1: 266 per-step gates, ~8.75 h of device, mostly setup). testing/batch_tasks.py (a pytest plugin) sets
  BRINGUP_TASK / BRINGUP_RECORD_TASK per test from its file's task, so the metrics land in each task's results file,
  and records each task's outcome; each task is judged exactly as unbatched (outcome + gate metrics). A task whose
  test never reported (the batch died) fails with the batch's exit code (rerun it unbatched). Compound gates, the
  ladder and the contract run alone. `--tasks` order is kept (M.1 / L.* / K.1 first), dependency order within a
  pattern. Only the verify dir is written (state.json, results/ untouched, as before).
  - Selftest: test_verify.py::test_a_batched_verify_judges_every_task_as_its_own_gate (fixture model: the same
    verdicts and metrics batched and unbatched, one process for the three test files, a failing task attributed).
- F143 (overseer, 2026-10-21): the run loop places queued overseer fixes before anything else, so `resume --only
  FIX.<n>` can name a fix still in the queue (DeepSeek-V4.1 FIX.8 printed "nothing left to run": the --only filter ran
  before placement). Note: `--until X` stops only X's downstream tasks; independent tasks still run (use --only).
  Selftest: test_overseer.py::test_only_can_name_a_queued_fix.
- F144 (overseer, 2026-10-21): `bringup freeze goldens` re-hashes only the golden files in every task's frozen block,
  and only when the golden verifies against its own content_hash; test files keep their hashes (an edited test still
  fails). DeepSeek-V4.1: F128's full-precision goldens replaced the frozen ones, so every C.* / S.* gate failed "frozen
  file changed: .../manifest.json"; a full re-freeze would have run every test twice more on the device.
  Selftest: test_reference.py::test_rehash_goldens_updates_only_verified_golden_files.
- F145 (overseer, 2026-10-21): R.5's old tap check (official output vs the fp32 step within calib x the input nudge)
  compared against the full-precision step since F132, so every step whose official code rounds to fp8 / fp4 inside
  failed (DeepSeek-V4.1: 40 of 129 steps, q_a / kv / attention / experts / shared expert at 1-5 %). That distance is
  the step's margin: recorded (`in_step_rounding`), never an error. The tap map is checked by F136's official_vs_taps
  (the reference's official numerics on the tapped inputs reproduce each tapped output; a mis-mapped tap or a stale
  state fails there, and an `explained` entry cannot hide it). Selftests: test_noise_floor.py.
- F146 (overseer, 2026-10-22): component_checks.limits keeps the structural column factor under its pre-F138 name
  (`component_col_factor` = `struct_col_factor`): DeepSeek-V4.1's frozen experts tests (swa, swa_engram) read it and
  failed FIX.9's gate with KeyError. Only that structural factor is aliased; the removed fixed accuracy numbers stay gone.
  Selftest: test_component_checks.py::test_frozen_tests_keep_the_structural_column_factor_under_its_old_name.
- F147 (overseer, 2026-10-22): one orchestrator per run (the pid file names a live other one: refuse, exit STOPPED),
  and an overseer abort with the pause file set stops the attempt loop (exit PAUSED) instead of starting the next
  attempt. DeepSeek-V4.1 FIX.9: abort + pause still started attempt 3 in the old orchestrator while a resumed second
  one began the gate. Selftest: test_overseer.py::test_a_second_orchestrator_on_a_live_run_is_refused.
- F148 (overseer, 2026-10-22): a profile's per-op rows go gzipped beside it (`<task>_ops.json.gz`, `ops_file` in the
  profile; gate_outputs commits it; the dashboard reads it). DeepSeek-V4.1 FIX.5's gate passed every metric, then git
  refused results/FIX.5_profile.json (613 KB, 498 KB of op rows) under the 500 KB hook. Selftest: test_dashboard.py.
- F149 (owner, 2026-10-22): a refused commit never means a re-run. A gate whose checks pass but whose commit git refuses
  (a pre-commit hook) is COMMIT_PENDING (core/ledger.py; was F110's FAIL): state.json keeps the gate's metrics, its
  lines and `commit_pending` {hook output, too_large files, the task files' key}; results/ and the other gate outputs
  stay on disk. COMMIT_PENDING is not a done status (dependents wait for the commit; a downstream gate never builds on
  an uncommitted pass), and `ledger.reset` (rerun) drops it. `gate <id> --commit-only` (core/gate.commit_only) re-stages
  and commits, running nothing; a gate run on a pending task does the same instead of running. Refused again, it stays
  pending and names the files too large (the overseer fixes the framework, then commit-only). It refuses when the
  task's own files (paths, frozen block) changed since the gate ran. The orchestrator stops the run on a pending commit
  (no further agent attempt; the watch wakes on "COMMIT PENDING") and, on the next run, commits it before anything
  else. Dashboards show COMMIT_PENDING. DeepSeek-V4.1 FIX.6 (35 min, state.json 502 KB) and FIX.5 (70 min, a 613 KB
  profile) were recorded FAIL and re-run in full only to commit. Selftests: test_core.py
  (test_a_refused_commit_is_pending_and_commits_without_a_rerun, replacing F110's; too large; changed files),
  test_orchestrator.py::test_a_refused_commit_stops_the_run_and_resume_commits_without_a_rerun.
- F150 (owner, 2026-10-22): no re-run of unchanged work. core/stamps.py stamps every gate run: per unit (a fix job's
  segments of one source task each, F79's BRINGUP_TASK prefix now set as the unit's env; else the whole command) the
  hash of its command as run, its source task's frozen block, the gate env's BRINGUP_* / TT_* variables (not where it
  files results), the code key and the goldens key. Code key: F105 / F116's file set (the tree minus the run's records
  and NO_MEASURE, now in core/stamps.py; plus overseer.yaml, the model's dashboard, verify / stamps / CLI / timing),
  but by content (ls-tree blobs, changed and untracked files hashed), so the gate's own commit keeps it; the whole
  tree, never a task's paths alone (a C.* test reads the reference, a ladder all of tt/). Goldens key: every golden
  dir's own records (manifest with content_hash, drift / floor files). Every run unit is kept in
  runs/<run>/stamps/<unit>.json (verdict, rc, metrics, side outputs copied aside; 8 stamps a unit). A unit whose stamp
  has a PASS (rc 0, its metrics within today's limits: all of them for a one-unit gate) is restored instead of run
  ("reused: same code" in the log, res.reused, state stamp.reused, the orchestrator's echo); a FAIL or HANG is never
  reused; a changed stamp, a missing declared artifact or `gate --fresh` runs it. This covers F87's resumed task, a fix
  job's units of passed tasks on unchanged code (and the units a stopped fix job already passed), `rerun` / `sweep`,
  and core.verify (a task whose every unit is kept PASS is reused; `--force` runs it; unbatched runs are kept). Not
  done: the A/B's ab_code_key keeps its own key (commit-dependent) so the live run's in-flight A/B sides stay valid;
  untracked files are in the gate key only. DeepSeek-V4.1: resumed fix jobs re-ran 45-70 min gates on code unchanged
  since their last run. Selftests: test_stamps.py (reuse on resume, a one-byte change runs, FAIL never reused, the key
  follows content, fix-job units, verify --force, the orchestrator's resumed task without an agent), test_verify.py.
- F151 (owner, 2026-10-22): a fix job runs its gate tasks' commands exactly. Cause: overseer._as_task copies each gate
  task's tasks.yaml command, but a full-model task's --no-precompile is not in that text for L.* and X.3 (ledger_gen
  writes it literally only for M.1, K.1, the profile and smoke): gate_command adds it at run time by the task's step
  (integrate / assemble / contract / perf), and a fix job's step is overseer_fix. So DeepSeek-V4.1 FIX.8's L.last,
  FIX.9's L.last and FIX.5's X.3 segments ran run_safe_pytest's precompile pass (the whole test once more) inside 45-70
  min gates, while M.1 / K.1 segments did not. Fix: core/gate.gate_units builds each unit's command with its source
  task's step (step_command), so a unit's command is exactly gate_command(source task); fixes already placed in a live
  tasks.yaml are covered at run time (no rewrite needed). Component tasks keep their precompile pass, as their own gates
  do. Selftest: test_overseer.py::test_a_fix_job_runs_its_gate_tasks_commands_exactly.
- F152 (owner, 2026-10-22): `orchestrator run|resume --until X` runs X and what X needs (Ledger.upstream: X and its
  transitive deps, recomputed each step since a placed fix may become a dep, F126), then stops ("reached X"); an
  unknown X stops at once. Was F143's note: it excluded only X's downstream tasks, so every independent task still ran
  (DeepSeek-V4.1: FIX.5's 70 min gate ran after `resume --until FIX.7`). STOPPED tasks and stopped fix jobs outside
  that set no longer hold the run. Selftest: test_overseer.py::test_until_runs_the_task_and_what_it_needs_then_stops.
- F153 (owner, 2026-10-22): two owner rules in the bringup skill, section 4. At its top: the overseer watches the
  orchestrator itself for waste and wrong behaviour first (stop it, do the cheap correct thing by hand, fix the
  orchestrator with a selftest; "the orchestrator does X" is never a reason given to the owner for waiting). After
  "Stop doomed work at once": time is the owner's most important constraint (reuse before any re-run over ~10 min:
  commit-only, the stamped gate result, the A/B sides, the kept profiles; never re-run for an administrative reason;
  say what re-runs, why and how long before starting it). Agents' rule 8: run the full gate once, after the last edit.
  Owner: "THE SYSTEM SHOULD NOT ACT AS TIME IS NOT IMPORTANT".

## F154: commit-only takes the fix the refusing hook asked for (2026-10-22)
- DeepSeek-V4.1 P.18: the gate passed, git's prefer-expect-error hook refused the commit (a fork unit test used
  pytest.raises). Fixing the named test file changed the task's files, so F149's commit-only refused it and asked for
  a full re-run: the hook's own fix was treated as a code change.
- Now the pending commit keeps a per-file hash map; commit-only commits without a re-run when every file changed
  since the gate is one the hook's output names, and lists them in the commit. Any other changed file still needs
  the gate. A pending commit without the map (older) records it at its next refusal.
- core/gate.py (task_files_map, _hook_fixed, _refused, commit_only); selftest
  test_commit_only_takes_a_fix_to_the_files_the_hook_named.

## F155: device time from the program real-time profiler, everywhere (owner 2026-10-22)
- Owner: "best if the framework could use the realtime profiler everywhere". The profile (X.1, the perf gates, the A/B
  profile, X.3) used the device profiler: an instrumented build, a sync and a profiler read at every section
  signpost, and for the per-op breakdown a sync after every ttnn call (about a minute per layer, so only the
  representative layers) plus a separate timeline run. The GLM-5.3 op report already had a real-time mode (rt, 10-09:
  per-call device-operation id ranges joined with the real-time records).
- Now testing/profiler.py ``backend()`` is "rt" by default (defaults.yaml perf.profiler; BRINGUP_PROFILER for one
  run). rt mode records the device-operation id at every signpost and around every outermost ttnn call; finish_rt ->
  charge_rt charges each record (per chip: end - start at dispatch, summed per runtime id) to its section (the same
  kernel_ns / kernel_ns_dev / programs schema: per range the slowest chip), its call, the op rows (op_ns, "(other)" for
  programs no wrapped call launched) and the timeline on the critical chip (gap / slot / host per call; unwrapped
  programs move the clock). testing/profile.py makes one rt run: sections, and with op mode the op rows of every
  layer (BRINGUP_PROFILE_LAYERS narrows) and the timeline, written in the old schema (profile.json "profiler": "rt",
  metric profiler_rt, chunk_wall_ms_rt = the rt run's wall without op mode). Dropped records and an inactive real-time
  profiler fail (no fallback). The device-profiler path stays behind perf.profiler: device for cross-checks
  (require_device_profiler checks its environment).
- Gates: ledger_gen puts no profiler environment on profile gates (DEVICE_PROFILE_ENV with BRINGUP_PROFILER=device
  only on perf.profiler: device); core/gate.with_clean_wall now adds the clean wall run to any test_profile.py
  segment and drops the device profiler's variables from the profiled run unless it carries BRINGUP_PROFILER=device,
  so ledgers written before F155 measure with rt without being regenerated (their gate commands change, so their
  F150 stamps change once). tests/test_op_report.py defaults to rt (BRINGUP_OPREP_RT kept as an override).
- Semantics: rt's time is dispatch start to end, including waits inside a program (a CCL waiting on a neighbour); the
  device profiler's is kernel time. Expect rt >= device per section, closest on compute-bound sections.
- Selftests: test_realtime_profiler.py (sections as id ranges without syncs, op rows / "(other)" / timeline from one
  run, inactive profiler fails, dropped records fail, backend switch, ledger env and clean-wall rewrite per switch);
  test_clean_wall.py and test_plan.py updated.
- Not changed: tests/test_all_gather_bench.py (a CCL bench that wants kernel time; it sets the device profiler
  itself).

## F157: agents run the repo's commit hooks on their changes before they finish (2026-10-22)
- DeepSeek-V4.1 P.18 and P.19: both gates passed and both commits were refused by prefer-expect-error (new fork unit
  tests with pytest.raises); F154 committed them without a re-run, but the overseer had to fix 11 lines by hand.
- agents/bringup-engineer.md rule 12: run `pre-commit run --files <changed and new files>` and fix every failure before
  finishing; use the expect_error fixture with the real message text.
- F155 device validation (2026-10-22, DS41 2x2, P.19 code, last chunk): real-time vs device profiler per section within
  0.2 % in total (91.94 vs 91.74 ms), program counts identical (3776), every section within ±2 % except moe_add +9 %
  (its reduce-scatter waits inside the program); per-op profile of all 8 layers in 6 min (device-profiler op mode ~13).

## F156: the stop guard accepts an OVERSEER HOLD line (2026-10-22)
- skill/bringup/heartbeat_stop.py: a stopped run whose last log line says OVERSEER HOLD (a stop the overseer chose and
  told the owner about) no longer blocks every turn.

## F158: a waived step covers the block out of its swap test (2026-10-22, owner)
- DeepSeek-V4.1 test_swap_swa_12_experts: the owner waived the bfp4 experts step, but the swap test's block out
  (err_swap_out) carries the same gap and read no waiver, so no spec entry could express "the waiver extends to the
  blocks containing the experts" (owner 2026-10-22).
- testing/component.py run_swap_test: when err_swap_out misses its noise floor and a swapped step has an owner waiver,
  the block out goes through NF.waive under that step (printed, err_swap_out_waived recorded; the structural bound
  still fails a wrong module). Selftest test_a_waived_step_covers_the_block_out_of_its_swap_test.

## F167 (committed as F159, 0035ffecd5c): a rung's step_dump_layers (per-step boundaries only for named layers) (2026-10-22)
- DeepSeek-V4.1 full model on galaxy_8x4 (40 layers, chunk 5120): storing every step boundary of every layer at the
  dumped chunks would take ~330 GB; the step tests use only the representative layers, while the ladder and the
  contract read every layer's in / out and state.
- Rung key `step_dump_layers: [..]`: at the dumped chunks, those layers store every boundary, the other run layers only
  `in` and `out` (state and KV caches unchanged, every layer). Recorded in the manifest; the spec check refuses a layer
  the run does not have. reference/generate_golden.py, core/spec.py; selftest
  test_golden_step_dumps_only_for_the_named_layers.

## F163 (committed as F160, a7d06380170): a rung's outs_every_chunk (2026-10-22)
- DeepSeek-V4.1 full model (owner): every layer's output on every chunk of the long rung, without per-step dumps on
  those chunks (the per-layer errors of every chunk). Rung key `outs_every_chunk: true`; manifest records it.
  reference/generate_golden.py; selftest test_golden_outs_every_chunk_without_dumps.

## F159: an all-zero target gets a zero limit, not an absurd one (2026-10-22)
- DeepSeek-V4.1 err_state_compress_partial: the golden state was exactly zero; drift.py's one-rounding rel divided by
  a 1e-300 clamp (9e255) and the end-to-end limit became 1.8e256, so any device value passed.
- reference/drift.py marks an all-zero target (zero_target, rel_res 0); testing/e2e_limits.py treats rel_res > 1 (impossible
  for a rounding of a nonzero value) as such a target in drift files written before, for the rel and the PCC limits.
  Selftest test_an_all_zero_state_gets_a_zero_limit_not_an_absurd_one.

## F164 (committed as F161, 603afee8bac): a layer without step dumps stores only its out (2026-10-22)
- DS41 full model short rung: a non-step layer's file held in and out, fp32 [5120, 20484] each (839 MB); its in is the
  previous layer's out bit for bit (checked at L5, L15, L30) and no check reads it (the ladder, drift and profile read
  in only at a run start, the step tests only on step layers). Now 420 MB; a run start keeps its in.

## F160: a top-k step's ids are compared as sets in the swap checks (2026-10-22)
- DeepSeek-V4.1 swap enc_source_14 / enc_source_engram_17: the indexer's per-step check compared ids by position
  (default_mode: "match" for an integer output); the reference returns ascending ids, the device's top-k returns
  them by score, so the same sets scored 0.0015 (limit 0.92) while recall was 0.994 (DS41-102).
- testing/component.py step_mode: an integer output that is a top-k (component_checks.output_kind == "index") is
  compared by per-row overlap (topk_overlap), as the component checks already did (K-42). Selftest
  test_a_topk_step_is_compared_as_a_set_not_by_position.

## F161: an owner waiver reaches a step's integer (top-k) output (2026-10-22)
- DeepSeek-V4.1 swap enc_source_14: the owner-delegated indexer waiver never applied: the per-step check's integer
  path returned before NF.waive. Now a waived top-k passes when its per-row overlap with the CPU step is structural
  (>= 1 - struct_scaled_rel, F97); a wrong id set still fails. Selftest test_a_waiver_reaches_an_index_step_but_not_a_wrong_one.

## F162: a gate unit re-runs only when what it reads changed (2026-10-22, owner: time first)
- DeepSeek-V4.1 FIX.10 re-ran 25-45 min of passed component tests three times in one afternoon: two spec waiver
  edits (for other steps), an intake re-approval (approvals.yaml) and framework fixes; F150's code key was the whole
  tree minus records and orchestration.
- core/stamps.py: code_key(repo, spec) also leaves out approvals.yaml and another model's bring-up records and frozen
  tests (its bringup/ non-Python files, its tests/bringup/), and counts the spec without tests.waivers (spec_text).
  waiver_key(spec, cmd) stamps per unit the waivers its tests read (testing/component.py is the only reader): a
  component test its step's, a swap test or a whole tests/bringup run all, ladder / contract / profile none.
  Stamps gain a "waivers" field: stamps from before F162 match nothing once (one re-run), never a false reuse.
  gate.unit_stamps and core/verify use it. Selftests test_a_unit_is_rerun_only_when_what_it_reads_changed,
  test_a_waiver_reruns_only_the_units_that_read_it. Still conservative: an edit to testing/component.py re-runs every
  component and swap unit (file-level, not function-level).

## F165: BRINGUP_CPU_THREADS caps a CPU job's torch threads (2026-10-22)
- DS41 full-model goldens shared the box with the live 2x2 run and starved its swap tests' CPU references (300 s
  timeouts). core/metrics.py cpu_threads() reads BRINGUP_CPU_THREADS first; run such jobs with `taskset -c <cores>
  nice -n 19` and the override. Selftest test_cpu_threads_env_override.

## F166: a looped gate is one stamped unit per test (2026-10-22, owner: time first)
- DeepSeek-V4.1 FIX.10: `for t in swa ...; do run_safe_pytest ... test_c_${t}_experts.py || exit 1; done` was one gate
  unit, so a stop or failure at the 7th test re-ran all 8 (~25 min) on resume.
- core/gate.py split_segment: a for loop over a literal list of one command template, and a run_safe_pytest call
  over several test files (flags without values only), become one unit per test (<task>#<label>, unit_base strips
  the label for BRINGUP_TASK and the frozen block); anything else stays whole, and a task without such segments has
  the same single unit as before. gate_command unrolls the loop (same commands, && stops at the first failure).
  A resumed gate reuses the tests that passed (F150 stamps) and starts at the first that did not; with F162 a waiver
  re-runs only its step's looped test. Selftests test_a_looped_gate_reuses_the_tests_that_passed,
  test_a_waiver_reruns_only_the_looped_test_of_its_step, test_only_a_literal_loop_or_a_plain_file_list_is_split.


## F169: a gate never restarts; it runs the delta (2026-10-22, owner)
- DeepSeek-V4.1 FIX.10 ran ~3 h for ~2 h of work: its ~40-min gate restarted from the first test four times, after a
  spec waiver edit, two framework fixes (testing/component.py) and a pytest timeout; F150's stamps keyed on the tree
  (F162 narrowed it, still counting the framework and waivers) and the gate retried as a whole. Owner: "a gate never
  restarts; if something fails, fix the framework, then only run the delta".
- core/stamps.py: under_test_key (the model's code and tests, the task's and a fix job's gate tasks' paths, the spec's
  paths.own, ttnn/ and tt_metal/ incl. the forks, the goldens) and Delta (runs/<run>/delta/<task>.json: each unit's
  verdict PASS / FAIL / STOPPED on which code under test). core/gate.py run_units: a unit that passed in this task's gate
  on the same code under test, command and environment is kept ("reused: same code under test"); a unit whose metrics
  miss stops the gate (as a non-zero exit did); a timeout or box error is STOPPED, retried alone; on_unit progress
  callback; state `units` and "units passed / units remaining" lines at a stop. run_units_only (gate --unit), rerun
  (gate --rerun), delta_status (gate --status); `--continue` is the default gate run, with a commit.
- overseer classify --task <id> --as framework|waiver|infra|model; orchestrator: a classified failure resumes into the
  gate's delta and starts no agent (stop_for_overseer when it still fails); the classification clears on a pass. The
  orchestrator logs "gate unit k/n <unit> PASS|FAIL|STOPPED (s)"; watch.PATTERN wakes on a FAIL / STOPPED unit.
- Skill section 4 opens with the overseer's duty (monitor and improve the framework, never its mouthpiece) and the five
  gate rules. Selftests: selftest/test_delta_gate.py (delta after a waiver + framework edit, model-code change re-runs
  all, infra stop retried alone, --unit / --rerun, orchestrator resume without an agent, the watch event); two F162
  selftests in test_stamps.py now state the within-task F169 behaviour.


## F170: the framework-waste audit, group A (2026-10-22, owner: time first)
- Source: reports/framework_waste_audit.md (22 items, audit of d24349ce0ba). Group A, the most time:
- Item 2: `gate <task> --check` (run_gate(check=True)): the agent's run of its gate, kept per unit in the stamp store and
  the task's delta, no state, no commit. The brief shows that command and the gate's units (briefs/brief.md, rule 8),
  so the orchestrator's gate after an attempt runs only what the agent did not pass on the same code.
- Item 5: core/gate.py gate_units: every test-running `&&` segment (TEST_RUN) is its own unit <task>#<label>
  (segment_label: the test file's stem); a non-test segment travels with the test after it (a trailing one with the
  last); a one-segment gate keeps its one unit named <task>. X.3 stopping in test_positions no longer re-runs the
  ladder and both profile runs.
- Item 6: Orchestrator.pause_point at the top of every agent loop (implement / fix, test, debugger, last-chance deferral,
  perf); the debugger and perf loops treat an overseer's abort as attempt_loop does (no gate, not counted).
- Item 7 + 17: harness.mesh_parametrize puts the spec's box.test_timeout_s on every device test (frozen component and
  swap tests had pytest.ini's 300 s); Orchestrator.gate retries a timeout or a box error once (a box error after a
  quick box check, tests/test_box.py through the safe runner) before InfraStop.
- Item 4: an attempt that changed nothing a gate reads (stamps.code_key) is not measured again (gate_unless_unchanged);
  an agent that crashed (rc != 0 / is_error) without changes is retried (CRASH_RETRIES, CRASH_WAIT_S), not charged; a
  programming error whose innermost frame is the framework's code (framework_error) on a task that passed before raises
  FrameworkStop: the run stops for the overseer, no agent (a first check before an implementation still starts one).
- Item 9: the debugger keeps a passing gate with problems (filed for review), as attempt_loop does.
- Item 16: stamps._git raises GitError on a failed git call (an empty file map gave every tree one key). The selftest
  fixture repo (fx) is now a git checkout.
- Item 21: run_agent(report=(role, attempt)): the debugger's knowledge-use report is read where its brief named it.
- Selftests: selftest/test_waste_fixes.py (one per item); test_stamps / test_overseer updated to the per-segment units.


## F171: the framework-waste audit, group B (2026-10-22)
- Item 15: run_units re-stamps a unit (stamp id, F169 under-test key) when it starts, not once at the gate's start.
- Item 22: a committing gate runs pre-commit only on the task's changed files (_dirty_under); none changed: no run.
- Item 12: stamps.env_key ignores BRINGUP_CPU_THREADS and TT_METAL_JIT_SERVER_* (where kernels compile, not what runs).
- Item 11: core.verify pins the checkout env (checkout_env.pin, as gate / rerun do) and run_batch keeps each batched
  task's outcome in the stamp store under its gate unit's stamp (a gate after a --batch re-check reuses it).
- Item 14: Orchestrator.ab_code_key = stamps.code_key (untracked files counted, commits of the same content not; was
  git ls-tree + git diff, blind to an agent's new untracked module). In-flight A/B sides re-measure once.
- Item 10: run_agent charges only the knowledge-file errors the attempt introduced (knowledge_errors before / after).
  Not done: overseer edits during an attempt are still charged as "outside the allowed paths" (rule: edit at pauses).
- Item 19: metrics.cpu_threads: outside a gate (no BRINGUP_TASK) while an orchestrator or run_safe_pytest runs on the
  box, a quarter of the physical cores and nice 19 (device_work_running, psutil).
- Item 20: dashboard/links.py STALE_AFTER 30 min (was 2: every gate woke the overseer); watch.first_report returns a
  DASHBOARD STALE (per publish) or RED (per content) line once (runs/<run>/watch_reported.json).
- Item 8 not done: the checks read the nudged outputs in several places (not only statistics) and caching them is
  ~GBs per model; swap tests' floors depend on device outputs; measure the CPU share first.
- Items 13 (stamps per model, not per run) and 18 (resume an agent's session) not in this round.
- Selftests: test_waste_fixes.py group B; test_verify's batched test checks the batch's runs are kept.

## F172: freeze's validation runs write to a scratch dir, not the task's results (2026-10-22)
- DeepSeek-V4.1 C.swa.shared_expert / C.swa_engram.shared_expert: the re-freeze's reference and zero-stub runs
  (record=False) wrote their metrics into results/<task>.json, so the task's record showed rel 1.0 on every check
  (read as "the shared expert returns zeros") while the committed device result was rel 0.002.
- core/gate.py run_gate: a validation run (record=False, not --check) sets BRINGUP_RESULTS to
  runs/<run>/validation/<task>, resets and reads metrics there, and deletes no gate outputs. Selftest
  test_freeze_validation_runs_leave_the_task_results_alone (fails without the fix).

## F173: the code under test is hashed by behaviour for the model's Python (2026-10-22)
- DeepSeek-V4.1 P.22 attempt 2 re-ran 18 accuracy units (~40 min) because it edited a Setting's explanation text in
  tt/settings.py: F169's under-test key hashed the model's files byte for byte.
- core/stamps.py under_test_key: the model's own .py files count by semantic_py_hash (the AST without comments,
  formatting, docstrings and a Setting's `why` text); unparsable files hash as text. Code, defaults and choices still
  change the key. Selftest test_a_docstring_or_setting_text_edit_keeps_the_code_under_test.

## F174: the bring-up-ops lint reads real calls, not names in strings (2026-10-22)
- DeepSeek-V4.1 P.22: attempt 1 was failed because a Setting's help text in tt/settings.py named the stock
  `ttnn.experimental.deepseek_prefill.moe_grouped_topk(...)`; the code already called the fork. Attempt 2 (~1 h) only
  reworded the text.
- testing/ops_lint.py stock_problems: calls come from the parsed AST (dotted name of each Call); strings, docstrings
  and comments never count; an unparsable file falls back to the line pattern. Selftest
  test_a_stock_op_named_in_a_string_is_not_a_call.

## F175: the final task runs the newest pick's full gate first; byte-key records stay valid (2026-10-23)
- DeepSeek-V4.1 X.3: the pending quick-accepted picks' full gates ran oldest first (P.17), so the shared gate re-ran 18
  units that P.22's agent had just passed on the same code (gate --check). Now the newest pick runs first (its delta),
  the older picks with the same command take its result.
- F173 changed the under-test key's formula, which made every earlier record stale on an identical tree. The key now
  comes as (semantic, bytes); a record matching either is kept; new records carry the semantic one. Selftests
  test_a_record_made_under_the_byte_key_stays_valid_on_an_identical_tree, the F122 test (newest first).

## F177: the stamp store finds the same test filed by another task (2026-10-23)
- DeepSeek-V4.1 X.3: four quick-accepted picks share one 18-unit frozen test set; units are `<task>#<test>` and the
  store looked a unit up only under its own name, so the identical command on the identical code (same stamp id) ran
  once per pick (45 min each).
- core/stamps.py Store.get falls back to `*#<test>` records with the same stamp id (record marked _from); restore
  copies the outputs from that unit's dir. Selftest test_the_store_finds_the_same_test_filed_by_another_task.

## F178: the warm full prefill times a gapped subset; never a silent skip (2026-10-23)
- DeepSeek-V4.1 X.3 set BRINGUP_FULL_PREFILL=1 but its layers (0-3, 14, 20, 21, 24) are not a prefix, so
  testing/profile.py full_prefill returned None and the gate passed without the number (no silent fallback, F138).
- full_prefill now runs a gapped subset back to back (prefill_gapped = 1) when every layer's state sources are in it,
  and raises otherwise. Selftest in test_device_side.py (full prefill).

## F179: a fork call's signature leaves out volatile arguments (2026-10-23)
- DeepSeek-V4.1 O.1: one flat_routed_expert call passes extra_key (a program-cache key built from buffer addresses);
  its value changed between the agent's capture and the gate's, so the signature changed and no case could match it;
  two agent attempts were spent on it.
- testing/fork_capture.py: VOLATILE_KWARGS (extra_key) are recorded but left out of `sig`; the pre-F179 signature is kept
  as `sig_full`, and testing/fork_cases.py counts a call covered by either, so existing cases still match. Selftest
  test_a_volatile_cache_key_does_not_change_a_calls_signature.

## F180: the attempt and A/B histories live beside state.json, uncommitted (2026-10-23)
- DeepSeek-V4.1 Z.1: the gate passed, the commit was refused: state.json 504 KB > the repo's 500 KB hook (agent_runs
  137 KB, ab 85 KB grow with every attempt and pick).
- core/ledger.py: SIDE_KEYS (agent_runs, ab) are saved to bringup/history_state.json (kept out of git through the
  clone's info/exclude; ignored by the agent path check); state() merges them back, so every reader is unchanged.
  Selftest test_attempt_and_ab_histories_live_beside_state_json.

## F183: a run's ledger apart from the model's shared files (2026-10-23)
- Growing DeepSeek-V4.1 from its 2x2 partial run to the whole model on galaxy_8x4 (reports/full_from_seed_design.md,
  F-S1): one ledger per model meant the second run would overwrite the first's tasks, results and approvals.
- core/spec.py: `ledger_dir` (tasks.yaml, state.json, results/, approvals.yaml, overseer.yaml, supervision.md,
  dashboard/, op_requests/) is `bringup_dir` except for a seeded run: `bringup_dir/ledgers/<run box>-<mode>/`, where
  its spec.yaml lives too; `run_dir` is `runs/<box>-<mode>/<run>`. The model's shared files (hooks.py, plan,
  components, serving contract, official taps, known issues, repo map, BREADCRUMBS, dashboard.yaml) stay in
  `bringup_dir`. Every ledger reader moved to `ledger_dir`; approvals: intake = the run's spec, perf = its
  opportunities.md, plan = the model's plan; gate.stage_paths stages each from its own dir. A spec without
  `run.seed` resolves exactly as before. Selftest test_a_seeded_run_has_its_own_ledger_and_run_dirs,
  test_approvals_are_the_runs_except_the_plan.

## F184: run.seed: a finished run of the same model this run grows from (2026-10-23)
- F-S2. `run: {box, mode, seed: <the seed's spec path>}`: the same model dir, tt/, hooks, CPU reference, plan, forks
  (unlike `prior`, which copies the model to another dir). The spec check refuses a seed of another model, `prior`
  with a seed, a seeded spec outside its ledger, and a seeded spec without ladder rung `seed` over the seed's layers.
- Briefs get a `## Seed` section (extend the seed's code in place, both boxes keep working); role `port`
  (briefs/roles.yaml, defaults agents.policy.port) for W.1 and M.1. The dashboard reads plan_boxes / plan_memory /
  block_graphs from the seed's results (its PL and R tasks are carried). Selftests
  test_a_seed_is_the_same_model_and_its_layers_are_the_port_rung, test_the_port_brief_names_the_seed.

## F185: the seeded task graph (2026-10-23)
- F-S3. plan/ledger_gen.py generate_seeded: CK.1 (the checkpoint on this host, intake approval, prompt; no HF sanity,
  the seed passed it), G.<rung> imported (F186), B.1, W.1 (the seed's layers at this run's size, each contiguous run
  from its golden input: every block type in its new mesh at once), M.1 (every layer, E2E_LAST), then the standard
  ladder, smoke, K.1, X.*, O.1, Z.1 (`integrate`, shared with the standard graph). No R/SC/PL/C/S/SG tasks: tasks.yaml
  `seed:` records the seed's spec, the commit of its last gate and the carried tasks (R.1-3, R.5, SC.1, PL.0-1) it
  passed. The seed's component and swap tests stay the diagnosis of a failing layer (they read the spec and its
  goldens), not gates. Selftest test_the_seeded_graph_carries_the_seed_and_ports_its_layers.
- Not done: F-S4 (the orchestrator runs a failing layer's block chain by itself), F-S6 (golden budget at intake),
  F-S8 (a tt/ commit of the seeded run flags the seed's L tasks for re-verification); the port agent is told to keep
  the seed box working, nothing checks it yet.

## F186: goldens made on another host, imported (2026-10-23)
- F-S7. Real-size goldens take CPU hours and the whole checkpoint in RAM (DS41 8x4: s10240 ~50 min, s56320 13.5 h), so
  they are made once on a big host and copied, often in part (only what the run's tests read).
- reference/golden.py: files.json (every file's sha256; derived, not in content_hash) and verify_copy (the index gives
  the manifest's content_hash, every copied file its line, the missing files listed). reference/golden_import.py: the
  seeded G task's gate (manifest is this rung's, verify_copy, the drift file of every rung reading the golden present,
  records golden_hash_ok / golden_files_missing / drift_metrics / the margin); `--index` writes files.json where the
  golden is complete. Spec `paths.*` relative paths resolve under the artifact root (one spec on several hosts).
  Selftests test_an_imported_golden_verifies_as_a_partial_copy, test_relative_artifact_paths_are_under_the_artifact_root.

## F187: every real-time profile records its per-op rows (2026-10-23, owner)
- DeepSeek-V4.1 P.23: the dashboard shows the newest profile; a pick's gate profile had no op rows, so the
  per-section op list (with each op's one-line note, F182) disappeared after the pick. Owner: never show an older
  profile's ops instead; record them every time.
- testing/profile.py want_ops(s, rt): on the real-time profiler the rows come from the same unsynced run (only host
  bookkeeping is added; the clean wall is measured in its own run), so defaults.yaml perf.op_profile is now true. The
  device profiler's op mode (a sync per call, about a minute per layer) still runs only when a spec sets
  perf.op_profile itself or BRINGUP_PROFILE_OPS=1. Selftest test_every_real_time_profile_records_op_rows.

## F188: a pick accepted after the final task passed still gets its full gate (2026-10-23)
- DeepSeek-V4.1 P.23 was added after X.3 had passed and accepted on its quick A/B; its full gate stayed pending
  (pending_full_gates runs only when the final task steps) and nothing ran it; the overseer ran it by hand.
- orchestrator.py: when nothing is left to run and the final task (step perf, no role) has passed, the run's end
  runs the pending full gates (the same code as before the final task: newest pick first, shared commands share a
  run). Selftest test_a_pick_accepted_after_the_final_task_runs_its_full_gate_at_the_end.

## F189: the overseer files of a seeded run ledger are the overseer (2026-10-26)
- DeepSeek-V4.1 galaxy_8x4-full W.1: the gate passed, but the overseer supervision.md in bringup/ledgers/<run>/ counted as
  the agent change outside its allowed paths, so a needless attempt 2 started (and an abort wrote overseer.yaml, flagged
  the same way). F183 moved the overseer files into the ledger; orchestrator.IGNORED and core/stamps.SKIP still matched
  only bringup/<file>.
- Both now also match bringup/ledgers/<run>/ (overseer.yaml, supervision.md, history_state.json; stamps: overseer.yaml,
  approvals.yaml, dashboard/). Selftest test_seeded_ledger_overseer_files.py.
- Not done: a gate whose units all passed, with only an out-of-scope-path problem, still starts a new attempt; it could
  commit and stop for the overseer instead.

## F190: a pass with out-of-scope changes stops for the overseer, not a new attempt (2026-10-26, owner)
- DeepSeek-V4.1 galaxy_8x4-full W.1 (F189's not-done): every unit passed and the gate committed, the only problem a file
  outside the paths; attempt 2 started with nothing to do and deleted results/W.1.json, edited state.json.
- attempt_loop and debug: a PASS whose problems include "changed files outside" now commits (as before), files the
  problems under `review` and stops the task (reason "gate passed and committed, ..."). The overseer reverts or keeps
  those files and resumes: the units are reused, no agent starts. A failing gate with such files is unchanged (the
  next attempt reverts them). Selftest test_orchestrator.py::test_a_pass_with_changes_outside_the_paths_stops_...

## F191: launch preflight: a gate's commit can go through (2026-10-26, owner)
- DeepSeek-V4.1 galaxy_8x4-full CK.1: a passed gate's commit was refused (no git identity on the node), found only after
  the gate. `orchestrator run|resume` now runs orchestrator.commit_preflight first: author and committer identity
  (`git var`), a merge / rebase in progress, a stale index.lock. Any problem prints `PREFLIGHT: ...` and the run does not
  start (exit STOPPED; the watch reports it at once). Selftest test_launch_preflight.py.

## F192: a run mesh smaller than the box sees only its cards; no subset is reported at launch (2026-10-26, owner)
- DeepSeek-V4.1: a 2x2 run on a Galaxy opened all 32 chips (gates never applied testing/cards.py, which fork tests use),
  and nothing said that no 2x2 subset of a Galaxy exists.
- At `run|resume` (no TT_VISIBLE_DEVICES from the caller): run_mesh_cards(spec.mesh) finds the subset (probed once,
  cached per box) and sets TT_VISIBLE_DEVICES for every gate and agent of the process; with no subset the launch stops
  with a PREFLIGHT line (a 2x2 there needs a mesh-graph descriptor or a submesh of the whole box). A miss is cached for
  this path (a resume would otherwise probe up to 40 subsets again); deleting "2x2:none" from the cache clears
  it. `gate` (__main__) applies a cached subset to a gate run by hand (no probe). A mesh of every card is unchanged.
- Placement: commit_preflight, run_mesh_cards, cached_cards live in orchestrator.py, the hand-run gate hook in
  __main__.py: both stamp-skipped (core/stamps.py), so landing F191 / F192 re-runs no passed unit; core/gate.py and
  testing/cards.py (measured) are unchanged.
- Not done: running a smaller mesh on a Galaxy at all (mesh-graph descriptor / submesh of the system mesh).
- Selftest isolation: the first suite run probed the Galaxy's real cards (test_resume_after_a_fix calls main resume)
  next to a live run's device work. selftest/conftest.py now hides /dev/tenstorrent and the card cache from every
  selftest (autouse); test_cards.py::test_no_selftest_sees_the_real_cards.

## F193: a fix job cut off mid-agent restarts its agent; only a STOP resumes into a gate first (2026-10-26)
- DeepSeek-V4.1 galaxy_8x4-full FIX.3 (root-cause the RT profiler failure): the overseer killed the run as its agent
  started (to land F190-F192, kill + resume per F95); on resume F87's "a resumed task gates first" ran the job's gate
  (B.1 + W.1, both passed before the job was queued), it passed on reused units and committed FIX.3 with no agent.
- A task with `precheck: False` (fix jobs) now gates first on resume only when it resumed from STOPPED (its agents
  failed; the work may be done meanwhile, F87); from RUNNING (cut off) its agent starts again. Tasks with a precheck
  are unchanged. Selftest test_orchestrator.py::test_a_fix_job_cut_off_mid_agent_restarts_its_agent_not_a_passing_gate
  (fails without the fix); test_overseer.py's F87 test now writes the resume entry's `from` as `resume` does.

## F194: a box with mixed IOMMU domains is reported at launch; a profiling task does not start on it (2026-10-26, owner)
- DeepSeek-V4.1 galaxy_8x4-full: every 8x4 mesh open logged "Real-time profiler disabled on device 7 / 10" and nothing
  stopped X.1 from profiling without them. FIX.3 found the cause: two cards re-enumerated after a board reset came back
  in an `identity` IOMMU domain among `DMA-FQ` ones; Metal keeps one IOMMU flag for the cluster, so their pinned D2H
  buffer fails (EINVAL). The fix is a root command (on that node `sudo systemctl restart tt-iommu-domains`).
- A host fact, so a framework check, not model code (the overseer first routed it as a model fix job; its agent wired a
  check into the model's hooks, which voided W.1's stamp: reverted). orchestrator.iommu_mismatch / iommu_problem read
  sysfs (/sys/class/tenstorrent/*/device/iommu_group/type): the cards translated differently from the majority, with
  the fix. `run|resume` prints `PREFLIGHT (warning): ...`; a task with step perf or a test_profile.py gate stops
  ("not started: ...") while the box is mixed. Stamp-skipped file: no passed unit re-runs. Selftest test_host_iommu.py.

## F194: a box with mixed IOMMU domains is reported at launch; a profiling task does not start on it (2026-10-26, owner)
- DeepSeek-V4.1 galaxy_8x4-full FIX.3: two Galaxy cards re-enumerated after a board reset came back in an `identity`
  IOMMU domain among `DMA-FQ` ones; Metal keeps one IOMMU flag for the cluster, so their pinned D2H buffer fails and the
  real-time profiler loses chips 7 / 10. Fix is root-only (here `sudo systemctl restart tt-iommu-domains`).
- A host fact, so a framework check (the agent's model-side check was reverted: it voided W.1's stamp).
  orchestrator.iommu_mismatch / iommu_problem read sysfs; `run|resume` prints `PREFLIGHT (warning): ...`; a task with
  step perf or a test_profile.py gate stops ("not started: ...") while the box is mixed. Stamp-skipped file.
  Selftest test_host_iommu.py.

## F195: the device smoke decodes the CPU smoke's token budget (DeepSeek-V4.1 galaxy_8x4 L.smoke, 2026-10-27)
- Symptom: L.smoke FAIL, device greedy 'We need answer one word. Capital of' (8 tokens, coherent), no 'Paris'; a
  fix agent started for a model that was right. Cause: testing/smoke.py run_smoke decoded a fixed max_new=8, the CPU
  smoke (intake/check_hf_sanity.py, R.1) 24; a thinking model reasons first (CPU: '...Capital of France =
  Paris...</think>Paris'). Fix: one budget, check_hf_sanity.SMOKE_TOKENS (24) or spec `intake.smoke.max_new`, via
  smoke_tokens(spec) in both; the device loop still stops as soon as the answer appears.
- Selftest: selftest/test_smoke_budget.py (fails without the fix).
- Not done: R.1 does not record the CPU's greedy token ids, so the device smoke cannot report its matching prefix
  with the CPU (would tell a budget miss from a divergence); needs an R.1 CPU re-run (~13 min decode), next model.

## F196: the clean wall is timed only when proven warm (owner, DeepSeek-V4.1 galaxy_8x4, 2026-10-27)
- Symptom: one 40-layer 5120 chunk: first timed run after compile 26.3 s, the fourth (profiled) 15.8 s; the owner:
  "verify the cache is hit for everything, then call that a warm run". F114b's fixed WALL_WARM (10) was a guess.
- Fix (testing/profile.py, BRINGUP_PROFILE_WALL=1): warm_up() runs the chunk until a run adds no program-cache entry
  (mesh.num_program_cache_entries(); at most WALL_MAX_WARM = 100, else an assertion naming the growth), then
  WALL_RUNS = 10 timed runs with mesh.set_program_cache_misses_allowed(False): a completed timed run hit the cache for
  every program. Records chunk_wall_warm_runs, chunk_wall_warm_ms, program_cache_entries(_runs),
  chunk_wall_cache_hits_only. A mesh without the API (selftests) keeps the fixed WALL_WARM.
- Selftest: selftest/test_warm_wall.py. Also: host numbers of the real-time-profiler run with op mode on are not the
  model's (per-call bookkeeping); only the clean wall is a host number.

## F197: no run is called warm unless it built no program (owner, DeepSeek-V4.1 galaxy_8x4, 2026-10-27)
- Symptom: F196's first use found no warm run at all: ttnn.bringup.flat_routed_expert_with_program keyed its program
  on xg's DRAM address, a new one every call (56 addresses in 5 runs; census plugin): every "warm" host number of the
  run (15-43 s a 40-layer chunk) was program rebuilds; with the address pinned the verified warm wall is 860 ms.
- Fix: testing/profile.py record_warm(mesh, before, what): the plain timed run, the real-time-profiled run and every
  positions.py timed run record <what>_new_programs (program-cache entries it added; -1 when the mesh cannot tell) and
  print "NOT WARM ..." when > 0. Selftest test_warm_wall.py::test_a_timed_run_that_built_programs_is_reported_not_warm.
- Also seen: with programs built mid-run, the real-time profiler's per-chip program spans include waiting for peers
  that started late in collectives (500k chunk: experts + moe_add 182.2 -> 166.9 ms device once warm). Device times
  from a run that built programs are inflated too, not only host times.
- Not done: the selftest suite takes a UMD chip lock (CHIP_IN_USE_0) somewhere: a box test waited on it (2026-10-27).

## F198: every fork test is counted; a failing test is never accepted because it failed before (owner, 2026-10-27)
- Symptom (pre-rebase baseline, 2x2, 2026-10-27; reports/fork_test_gaps_2026-10-27.md): 12 flat_routed_expert subgrid
  tests crashed for 16 days, re-recorded six times as "pre-existing"; O.1 (`fork_cases --run-tests`) returned before
  running anything when one fork had no case file (update_padded_kv_cache), so 8 forks' cases never ran, and it
  reported 1 failure; a test file that failed to import was dropped from the collection; skips counted as passes;
  `fork_source` failed only on regressions, so tests failing on the original and the fork (dispatch 2, the
  update_padded_kv_cache selection's collection error, indexer_score's fp32-refusal test) passed for good; F137 removed
  Settings(max_precision) and missed xing40_a4b_d_p, whose import error broke every test in the DeepSeek-V3 test folder
  (the source tests of combine, dispatch, offset_cumsum, update_padded_kv_cache) from 10-21; one dead Ethernet core
  left a device open and 175 all-gather tests errored at setup, reported as results.
- Owner: a failing fork test is never accepted (no "same as baseline"); transparency on what works and what does not.
- Fix: testing/fork_tests.py counts every test per fork from a junit report per group (passed, failed, errors,
  skipped with pytest's reason, not run = collected but never reported), collection errors (an import error, nothing
  collected) are failures, a group in which every test errored or none reported (a box error) runs once more after
  run_safe_pytest's reset and says so. testing/fork_cases.py: a fork without a case file is a failure of its own and
  the others still run; records fork_tests_failed (failed + errors + not run + collection errors + forks without
  cases), fork_tests_passed, fork_tests_skipped, fork_tests_not_run; O.1 shows passed and skipped in its gate output.
  testing/fork_source.py: every test that does not pass on the fork fails the check, labelled regression / new /
  failing at baseline too; skips listed. indexer_score's source selection filters out the one test the fork can
  never pass (its change), with the reason. xing40_a4b_d_p/tt/settings.py: max_precision removed.
- Selftests: test_fork_cases.py (counting, a fork without cases, import errors, junit names, the box-error retry, the
  strict source report), test_model_imports.py (every model's settings and serving adapter import; fails without
  the xing fix).
- Not done here (owner's list, 2026-10-27): cases for update_padded_kv_cache (57 calls) and the unpadded shared-expert
  call; the flat_routed_expert perf band (passes when faster, numbers per box); the strict all-fork run at the end of
  a completed run and in the rebase skill; skips that are not "this box is too small" (parametrize combinations, dead
  rms_norm perf tests, a known rms_norm bug, the indexer source entry run on 1 card, missing moe_ag perf numbers);
  test time (a mesh opened per test, weights built per test).

## F199: every fork's every test, strictly, once a run is complete (task F.1) and in the rebase skill (owner, 2026-10-27)
- Owner: transparency on what works; the strict all-fork run runs after a completed bring-up run and in the rebase
  skill, not on every fork change (it takes about an hour on a 2x2).
- testing/fork_suite.py: per fork, each tests/unit file (perf files included), the model cases (fork_tests.py) and the
  original op's tests (fork_source.py), every test counted (F198's tallies), every skip and failure listed. A failure
  passes only through an owner waiver, spec `tests.fork_waivers: {<test id>: reason}` (defaults.yaml), reported
  WAIVED. Records fork_suite_failed (gate == 0), _passed, _skipped, _waived and <task>_fork_suite.json.
- plan/ledger_gen.py: task F.1 after Z.1 (role optests: an agent fixes a failure, three attempts, then stops for the
  owner); briefs/roles.yaml tells it never to skip, filter, delete or loosen a test, and that only the owner waives.
  Existing ledgers get it with `ledger_gen --extend`.
- Skills: rebase-bringup step 1 (baseline: fork_suite, the full list to the owner before rebasing) and step 5 (done
  when fork_suite_failed == 0; a failure is never accepted because step 1 had it); bringup-fork-op ("must match the
  baseline" is gone: every test passes, a by-design divergence is filtered with its reason, skips are reported);
  bringup (fork-test transparency, F.1).
- Timing tests (flat_routed_expert indexed perf, moe_ag): fail when slower than the margin, log a recalibration note
  when faster (owner); the shared helper tests/ttnn/profiling/realtime_profiler_utils.py (upstream) is used to measure
  only.
- Selftests: test_fork_cases.py (the suite's waivers and totals; the skills name the rule).

## F200: the clean wall is timed once the walls settle, not only once the program cache is warm (owner, 2026-10-27)
- Symptom: DeepSeek-V4.1 2x2 MoE-block A/B: F196 called every run warm after 1 warm-up run (no new program), but the
  host walls kept falling 10-25% over the next 5-6 runs (flat 142 -> 118 ms, the median 138.0 came mostly from the
  falling part), and one configuration measured 102.2 / 120.0 / 125.2 ms in three processes.
- testing/profile.py warm_up: after the cache is warm, it keeps running until the last WALL_SETTLE_WINDOW (5) walls
  are within WALL_SETTLE_PCT (5%) of their median, at most WALL_MAX_SETTLE (30) runs; not settled is flagged
  (chunk_wall_settled 0, "F200 NOT SETTLED"), not refused. Also records chunk_wall_cache_warm_runs and
  chunk_wall_loadavg (other host load skews a wall).
- Left: the spread between processes (the same setup 102 vs 120 ms) is not a warm-up effect; compare walls only from
  back-to-back alternating runs (A, B, A, B) until that is understood.
- Selftests: test_warm_wall.py (the flat path's real walls settle at run 12; never-settling walls are flagged; the
  window rule).

## F201: the generic program runners' calls are recorded and tested, by program, never by address (owner, 2026-10-27)
- Symptom: DeepSeek-V4.1 O.1 under MOE_IMPL=block passed with 0 uncovered calls while Marko's fused MoE head and the
  exchange's sparse dispatch ran through ttnn.bringup.prepare_program / run_prepared: plain nanobind functions, not
  ttnn operations, so fork_capture never saw them (and their program argument printed as an object address).
- testing/fork_capture.py: patch() also wraps ttnn.bringup.prepare_program / run_prepared (PROGRAM_RUNNERS). The
  program is recorded as program_summary(desc): per (mesh range, program) its kernels (a file's path, or the inline
  source's hash with integer literals >= 2^16 masked as addresses), defines and compile-time args (masked the same way),
  core ranges, config, how many cores take runtime args (never their values), CBs (size, cores, formats, buffer-backed)
  and semaphores; `digest` names it. run_prepared carries the summary of the descriptor it was prepared from. Shapes
  and dtypes of the io tensors as for any call.
- testing/fork_cases.py: op_to_fork also maps a fork's plain `mod.def` functions; a case may cover several captured
  calls (`sigs`, e.g. one program's prepare_program and run_prepared).
- core/gate.py outputs_in: every `fork_calls_<name>.json` an optests gate names is a gate output (deleted before the
  gate, committed by it), not only fork_calls_scaled.json (DS41's O.1 now also records MOE_IMPL=block).
- prepared_program/tests: one case per distinct program; the program built as the model builds it, run by the plain
  generic_op and prepared with the call site's moved tensors at other addresses (bit-identical), and the determinism check.
- Selftests: test_fork_capture_programs.py (the same program at two addresses -> one summary and one signature; another
  define -> another program; masking keeps shapes).

## F202: a mesh a test opens itself carries every device param (DeepSeek-V4.1 galaxy_8x4 K.1 decode hang, 2026-10-27)
Problem: the K.1 runner smoke's stand-in decode opens a fresh mesh by hand (test_runner_smoke.open_mesh: set_fabric_config
+ open_mesh_device) and dropped the spec's fabric_router_config (hooks.device_params: 10880 B payload for the weighted
combine's 10272 B packets). Every combine packet exceeded the router default 4352 B: the first MoE call hung (2/2 on
b06u08, every run here), while the runner (sets the payload itself) and the fixture-based tests passed. Hours went into
a "flat_routed_expert decode hang" that was a test's mesh setup. xing40's copy of the test had the same latent bug.
Fix: testing/harness.open_mesh(spec) / close_mesh: every device_params entry applied as the repo fixture applies it
(conftest set_fabric: STRICT_INIT, tensix extension off, router config passed; the rest to open_mesh_device). Both
smoke tests use it. Selftest test_open_mesh.py: the helper's calls, and a lint that refuses ttnn.set_fabric_config /
ttnn.open_mesh_device in any models/demos/*/tests/bringup file. The fork side (combine_rf) now refuses a payload too
small for its packets at build time, so this can never be a silent hang again.

## F203: a runner test waits for its engine build instead of failing into an agent (DeepSeek-V4.1 galaxy_8x4, 2026-10-27; F200 on the galaxy branch before the merge)
- Seen: the owner turned on serving.require_engine for the 8x4 full run while tt-d-gen was still being built (dgen-build
  in parallel); K.1's gate would have failed on the missing build and started an agent attempt that cannot succeed.
- orchestrator.engine_missing: a task whose gate names a test file that tt-d-gen's engine drives (it names dgen_engine
  or use_engine) with require_engine and no build (dgen_engine.find_build) does not gate; it waits ("tt-d-gen build
  missing: <why>", WAITING FOR A PERSON) and F109 runs the independent tasks meanwhile. parked_waiting releases it as
  soon as the build is found; when only it is left, the run stops waiting and `resume` re-checks.
- Selftest: test_engine_wait.py (waits, no gate, no agent, independent work passes, runs once a build appears; no wait
  without the requirement or without a runner test).
- Also: testing/dgen_engine.server_repo expands $USER / ~ like the spec's own paths (core/spec._expand); the 8x4 spec's
  `/scratch/$USER/tt-d-gen` was taken literally and the built engine never found (selftest
  test_the_specs_server_repo_expands_like_its_other_paths).
- Also: testing/dgen_prefill_driver passes device_prefill_pipeline its arguments by name, layers_per_chunk only to a build
  whose signature declares it (tt-d-gen 1cf343887 dropped it; by position the timeout became sp_factor and the engine
  died, ENGINE_FATAL ... sp_factor=1800000). Selftest test_the_engine_driver_names_its_pipeline_arguments.
- Seeded runs (F185 follow-up, not done): the seed's contract tests are carried unchecked; box literals in them
  (server_rules CHUNK 2048 / MAX_SEQ 4096 / GOLDEN_RUNG s4096) broke K.1 on 8x4. The rule for tt/ (sizes from the spec
  and mesh, never one box's literals) applies to the contract tests too.
- Also: tt-d-gen attaches to the runner's layer-completion ring /tt_prefill_layer_completions_<id>
  (dgen_engine.ack_shm_name = layer_completion_ring_reader.scheduler_shm_name), not the pre-ring counter channel
  /tt_prefill_layer_acks_<id>: with the old name the engine waited out its 30 min connect timeout while the runner
  waited for its first chunk. Selftest test_the_engine_attaches_to_the_runners_completion_ring. Other models' plans
  (xing40 serve/server.py, contract test) still name the old channel: fix at their next run.

## F204: a hang's triage and Inspector dump survive the next run and every framework kill (DeepSeek-V4.1 galaxy_8x4, 2026-10-27; F201 on the galaxy branch before the merge)
Problem: the K.1 runner-smoke decode hang lost its evidence twice in a row: run 2's generated/tt-triage was wiped by
run 3, and run 3's was cut off by a kill while tt-triage was still writing. Both dirs (generated/tt-triage, rewritten by
the next hang; generated/inspector, rewritten by every mesh open) live at fixed repo paths; the gate kept only
triage.txt, and only at the gate's end, after later units had reopened the mesh.
Fix: core/hangdump.py. keep() copies both dirs whole into <logs>/<log stem>.hang-<unit>-<time>/ the moment a unit
returns the safe runner's HANG (it returns only after triage finished), before the next unit runs; a single-command
gate keeps it at its end; the gate's result names the dir. wait_triage() blocks (up to 300 s) while a tt-triage.py
of this user runs, before every framework kill: overseer abort, an agent's timeout, a stopped orchestrator (_killpg).
Selftest test_hangdump.py (a three-unit gate whose middle unit hangs and whose next unit rewrites inspector; abort
waits before it kills). Overseer side (skill): never kill a device run by hand while tt-triage runs; diagnostic runs
outside a gate go through a serialized wrapper that copies the same dirs before the next run (diag/run.sh on the node).

## F205: a wall is settled only without a downward drift (DeepSeek-V4.1 galaxy_8x4 5-position profile, 2026-10-28)
Problem: F200's settle test (the last 5 walls within 5 % of their median) passes a slow drift: walls falling ~1 % a run
stay within a 5 % band of 5 runs. The 8x4 flat series was called settled at 1042 ms (200k, 11 runs) and 842 ms (100k,
8 runs) while 0k fell to 640 ms over 28 runs and 500k to 721 over 31: the host walls were not comparable.
Fix: testing/profile.settled also requires the last window's median at most WALL_TREND_PCT (2 %) below the window
before it (two windows of 5 needed); WALL_MAX_SETTLE 30 -> 60. Selftests test_warm_wall.py (the drift case; the F200
2x2 sequence now settles on the 119 ms plateau, not on the 123.6 .. 123.5 window still 16 % below its predecessor).

## F206: a fix job runs before the task it is placed before, also under --until (DeepSeek-V4.1 FIX.6, 2026-10-28)
Problem: _place_fixes inserted the fix before the first unpassed task in tasks.yaml, but with every gate task passed
nothing depended on it; under `resume --until X.2` the scope is X.2's upstream, so FIX.6 (the MoE default change for
the new torus spec) was out of scope and X.1 profiled the old default on the torus, where it hangs (stopped by hand).
Fix: the task a fix is placed before gets the fix as a dep (topo order and --until scope both include it); the plan
approval's ledger signature ignores deps on overseer fixes (the edge is the fix's, not the plan's). Selftest
test_overseer.py::test_a_queued_fix_runs_before_the_next_task_under_until; full selftests 639 pass (1 failure from
before: xing40 server_rules.py:393 fixed threshold).

## F207: an A/B side whose process died of a signal runs once more (DeepSeek-V4.1 P.2, 2026-10-28)
Problem: P.2's off side (the old path) died with "Fatal Python error: Bus error" inside load_tensor_flatbuffer while
loading cached expert weights, at model build, before the changed code ran; the files re-read cleanly afterwards
(transient). ab_measure has no crash pattern (INFRA_FAILURES covers board errors only), so the side was recorded as
failed (rc 1, no metrics) and the owner got a table with no comparison. Gate units are retried once on box errors
(F170); A/B sides were not.
Fix: PROCESS_CRASH (Fatal Python error: Bus error / Segmentation fault / Aborted, "(core dumped)"); a side item that
exits non-zero with it runs once more (the crashed log kept as <name>.crashed.log, the results reset). A second crash
is recorded as the side's result. Selftest test_perf_ab.py::test_a_side_whose_process_crashed_runs_once_more.

## F208: realistic precision covers every tensor the reference rounds; a spec rule that contradicts it is refused (owner, 2026-10-28)
Problem: DeepSeek-V4.1's KV caches (window fp8, compressed fp4, index keys fp4 in the reference) and its GEMM activation
inputs (fp8 in the reference) run in bf16 on the device. Cause: the 2x2 intake spec (2026-10-14) carried "Precision
during bring-up: HiFi4, fp32 accumulation, bf16 KV cache" in agents.rules; SC.1 wrote the serving contract to it ("the
owner's rule is a bf16 KV cache"); F123 (2026-10-20) changed agents' rule 7 but neither the spec line nor the contract;
the seeded Galaxy spec copied the line. Rule 7 spelled out weights only, rule 17 only permits low-bit caches; nothing
checked. Owner: "realistic precision = whatever the reference uses".
Fix: agents' rule 7 names caches and GEMM inputs at the reference's width (device 8-/4-bit formats), a narrower
reference makes bf16 wrong even where a contract or spec says otherwise; core/spec.py stale_precision_rules refuses
agents.rules entries with HiFi4 / "fp32 accumulation" / "bf16 KV cache" / full or maximum precision / "precision during
bring-up" unless marked 'owner exception' (selftest test_core.py::test_a_spec_rule_that_contradicts_realistic_precision_is_refused).
Not done: an automated audit of the device's stored / GEMM-input formats against the reference's rounding list (needs a
model hook; see the DS41 fix plan). Flags today on other models: glm53_flash_d_p(_lb) ("HiFi4 during functional
bring-up"), xing40_a4b_d_p(_8x4) (SDPA "still HiFi4": an owner decision?) - owner to decide.

## F209 (pending): a rejected perf pick reverts its fork changes too, and keeps its knowledge notes (DeepSeek-V4.1 P.3)
Problem: `decide --reject` reverted P.3's model paths but left ~230 lines in ttnn/ttnn/bringup/prepared_program (C++,
bindings, test, docs) and INDEX.md - forks every agent may write (F48) are outside the pick's paths; and a full revert
would also drop the pick's finding (known_issues DS41-131, BREADCRUMBS). Overseer reverted the fork code by hand and
kept the notes. To do: reject reverts every file the pick changed except the knowledge files.

## F210: a gate commit leaves the generated dashboard pages out (DeepSeek-V4.1 FIX.8, 2026-10-29)
Problem: git_commit staged everything under the task's paths, and the ledger dir holds dashboard/index.html and
teletext.html, re-exported after every gate (published as artifacts). At ~0.9 MB each they hit the repo's 500 KB hook,
and FIX.8's passed gate (4-bit KV, 11 units) went COMMIT_PENDING; the same pages had been swept into a commit by hand
before (the overseer's `git add -u`). Fix: core/gate.py GENERATED = `:(exclude,glob)**/dashboard/*.html` on the add,
diff and commit pathspecs (and unstaged), so no gate commit takes them. Selftest
test_a_gate_commit_leaves_the_generated_dashboard_pages_out.
Second try (P.6, same night): an exclude pathspec makes `git add -A` skip an untracked file named outright (P.6's
results/P.6.json: the commit failed with "pathspec did not match"; the sandbox test used a directory and passed).
git_commit now lists the changed files under the paths (ls-files -m -o -d --exclude-standard + staged), drops
GENERATED matches, and adds / commits exactly that list.

## F211: commit-only ignores the records the framework writes under a task's paths (DeepSeek-V4.1 FIX.8, 2026-10-29)
Problem: after F210's fix, `gate FIX.8 --commit-only` still refused: "the task's files changed since its gate passed:
state.json (not named by the hook)". FIX.8's paths are the whole model dir, which holds its ledger; the refusal itself
wrote state.json (and the dashboards, results) after hashing the task's files, so commit-only could never commit it
(only `rerun --from`, ~1.5 device-hours). Fix: core/gate.py task_files_map and _hook_fixed drop files stamps.SKIP
matches (state.json, results/, dashboards, overseer.yaml, approvals.yaml, records) - the same files that never void
a stamp. Selftest test_commit_only_ignores_the_records_the_framework_writes_under_the_tasks_paths.

## F212: a gate needing device_ms_* with only wall-only profile segments is refused up front (DeepSeek-V4.1 P.6, 2026-10-29)
Problem: the overseer's perf gates (P.1-P.7) ran test_profile.py with BRINGUP_PROFILE_WALL=1 and gated device_ms_total.
with_clean_wall (F114) adds the clean wall run and the profiled run only to a segment without that flag, so no run
recorded device_ms_total: P.6's passing change (all accuracy in limits, wall 424.8 < 434) showed the gate FAIL on a
MISSING metric, which would have started a second agent attempt for nothing. Fix: ledger.wall_only_device_metrics,
in Ledger.validate (unfinished tasks) and in the gate's pre-check; P.4-P.7's commands lost the flag (the gate adds the
wall run itself). Selftest test_a_gate_that_needs_device_time_but_only_profiles_the_wall_is_refused_up_front.

## F213: a released fix is not placed before a perf pick whose A/B is measured (DeepSeek-V4.1 FIX.9 / P.6, 2026-10-29)
Problem: P.6 was accepted (A/B measured, its change in the tree, not yet gated / committed) and FIX.9 released; at the
next boundary _place_fixes put FIX.9 before the first unpassed task, P.6, made P.6 depend on it (F206) and started its
agent on a tree holding P.6's uncommitted change, whose gate commit would have swept P.6 in. Fix: _first_open skips a
perf pick whose state has `waiting` or `ab` (the decision pending or made); the fix goes before the next open task.
Selftest test_a_fix_is_not_placed_before_a_perf_pick_whose_ab_is_measured. (By hand: FIX.9 aborted, P.6's dep on it
dropped, P.6 gated and committed directly.)

## F214 (pending): an accepted A/B decision was lost from state.json (DeepSeek-V4.1 P.6, 2026-10-29)
`decide --task P.6 --accept` printed "accept recorded" at ~02:23:20; at 02:26 no task in state.json had an `ab` record
(P.6 back to its WAITING entry). Writers in between: the orchestrator resume (02:23:38: placed FIX.9, started and lost
its agent to an abort), `overseer abort`, a BLOCKED `gate P.6`. ledger.update merges under the lock, so a stale
whole-state write is suspected (a process saving a state it read before the decide). To do: find the writer; a
selftest that a decide survives a resume. Workaround used: the A/B table on disk + `gate P.6` (commits on PASS).

## F220: a model switch set in the environment is part of a unit's stamp (DeepSeek-V4.1 2x2 engram bfp8 A/B, 2026-10-28)

`DS41_ENGRAM_ROWS=bfp8 verify --tasks L.s4096 L.last L.s55296` printed "reused: same code, kept PASS" for all three
rungs: the stamp's environment key kept only `BRINGUP_` / `TT_` variables, so the bf16 run's PASS stood in for the
bfp8 run and the A/B compared a run with itself. `stamps.env_key` now also keeps every prefix a model declares in its
`tt/settings.py` (`Settings("DS41_", ...)`, found by `stamps.settings_prefixes()`). With no such variable set the key
is unchanged, so no kept result is invalidated. Selftest: `test_a_model_switch_in_the_environment_changes_the_stamp`.

## F221: a waived step covers the golden check of the later steps it feeds (DeepSeek-V4.1 2x2 engram bfp8 default, 2026-10-28)

The owner waived the engram_rows step when its rows became bfp8 (rel 3.5e-3 vs the exact rows, limit 0). Swap tests
S.swa_engram.02 / .03 / .18 then failed on one check only: swap_engram_kv_vs_golden_err 0.00350 > 0.00335, the rows'
gap carried into the next step, while engram_kv matched the CPU on the same inputs (0.0029 <= 0.0034). F158 passed a
waiver on to the block out, not to a later step's vs-golden check. `testing/component.run_swap_test` now records, per
swapped step, the first waived swapped step before it in the block (`_UPSTREAM_WAIVED`); `_check_step` waives that
step's vs-golden failure under it (`NF.waive`: structural bound kept, `<tag>_vs_golden_err_waived` recorded). Its
vs-CPU checks still gate; a later step's waiver covers nothing before it. Selftest
`test_swap_checks.py::test_a_waived_step_covers_the_golden_check_of_the_steps_it_feeds` (fixture: per-step noise,
`fixture_model.NOISE["steps"]`); full selftests 650 pass (1 failure from before: xing40 fixed threshold). Device:
S.swa_engram.18 and S.enc_source_engram.20 pass with ENGRAM_ROWS=bfp8.

## F222: a setting may name a test value when it is bit-identical to the default (DeepSeek-V4.1 ENGRAM_TABLE, 2026-10-28)

The owner made ENGRAM_TABLE=resident the served default: each engram table is built in RAM as bfp8 at process start
(~45-84 s a layer). Every device test touching layers 1 or 14 is a new process and paid it (the engram_rows step test
37 s -> 87 s). `core/model_settings.Setting` gains `test`: the value a pytest process gets instead of the default,
only for a choice proven to give the default's numbers bit for bit (here `random`, read from disk with read-ahead
off; tests/test_engram_table_modes.py). The environment variable still overrides both; a process without pytest (the
server) gets the default. Selftest `test_model_settings.py::test_a_test_value_applies_only_under_pytest_and_the_
environment_still_wins`; full selftests 651 pass (1 failure from before: xing40 fixed threshold).

## F223: an fp4 quantizer's output is nudged inside its fp4 cell, so a 4-bit cache is judged against the reference's own 4-bit rounding (DeepSeek-V4.1 2x2 CKV_FORMAT=bfp4_rows, owner 2026-10-28)

With the compressed KV stored as bfloat4_b rows, the attention step failed one check: chunk0 rel 0.0321 > 0.03. The
golden check passed (0.029 <= 0.148), because its floor includes the official arithmetic, whose fp4 cache moves the
output by 0.025-0.047. The synthetic inputs (chunk0, layer14, mixed) have no official arithmetic. Their floor was the
input nudge alone, and the compressed input was `fixed` (a quantizer output, not nudged). So the floor was 0.0033 and
the structural minimum of 0.03 applied: a bar the reference's own 4-bit cache would not clear. Owner: count the
reference's fp4 in the limit. `noise_floor.nudge` takes the input dtype `fp4/<block>`. The value moves uniformly
inside its e2m1 cell: the grid step at |x| / s, with s = the block's amax / 6 along the last dim. This is the size of
the official quantizer's rounding, which the selftest checks against an fp4 quant-dequant (0.5-2x). `input_dtypes`
accepts it and refuses `fp4/<non-number>`. DeepSeek-V4.1's official_taps.yaml marks the attention step's compressed
input `fp4/16` (e2m1, e4m3 scale per 16). The indexer's compressed input stays `fixed`, since the device does not
re-round it. Result: chunk0 limit 0.0493 (was 0.03), pass; layer14 / mixed stay at 0.03 (their floor is lower).
Selftest `test_noise_floor.py::test_an_fp4_input_is_nudged_by_the_size_of_its_quantizers_rounding`;
noise_floor + swap_checks selftests 56 pass.

## F224: the fp4 cell with a power-of-two scale, and a step's own quantizers in the nudge (DeepSeek-V4.1 2x2 lowbit, owner 2026-10-29)

Lowbit phase 2 puts the index keys and the index query at the reference's width (bfloat4_b on device; the reference:
e2m1 with an e8m0 power-of-two scale per 32, ds41_ref.py `_q4(..., 32)`). F223's `fp4/<block>` cell assumes a free
scale (amax / 6, the compressed KV's e4m3 one). A power-of-two scale is 2^ceil(log2(amax / 6)), up to 2x coarser, so
that cell understates the reference's rounding. And the index query is rounded inside the indexer step, not at an
input: the CPU target (native, fp32) skips it, the golden check counts it through the official arithmetic, but the
second inputs (chunk0 / layer / mixed) only see the input nudge, the F223 failure again once the device rounds the
query. Fix: `noise_floor.nudge` takes `fp4e8/<block>` (the e2m1 grid step on the power-of-two scale), accepted by
`input_dtypes`; and an opt-in `nudge_inner: true` per official_taps.yaml step makes its nudged runs (every caller:
`nudged`, the component and swap selection paths, the swap chain) run under `noise_floor.inner`, where a reference's
internal quantizer calling `noise_floor.nudge_inner(x, dtype)` moves inside its cell (one seed a call); the exact run,
unmarked steps and calls outside a nudged run get x unchanged. DeepSeek-V4.1's `_q4` in native mode calls it (fp4/16
for the e4m3-scale compressed KV, fp4e8/32 for the index keys / query). No taps entry is marked yet, so no limit
changes today. Phase 2 (with INDEX_K_FORMAT / INDEX_Q_FORMAT): `indexer: {inputs: [bf16, bf16, fp4e8/32],
nudge_inner: true}`. Not covered: an internal fp8 (e4m3, `_q8`) cell. Selftests `test_noise_floor.py::
test_a_pow2_scale_fp4_input_is_nudged_by_the_size_of_its_quantizers_rounding` (fails with the power-of-two scale off)
and `::test_nudge_inner_moves_a_steps_own_quantizer_only_in_its_marked_nudged_runs`; noise_floor + swap_checks +
component_checks 109 pass; the whole selftest suite 660 pass, 1 failing before and after (test_e2e_runtime
test_no_runner_contract_test_takes_a_fixed_threshold: xing40_a4b_d_p/tests/bringup/contract/server_rules.py:393).

## F225: an owner waiver for named failures of a frozen test that has no limit to waive (DeepSeek-V4.1 FIX.12, owner 2026-10-29)
Trigger: FIX.12's gate (P.10's units) failed test_adapter_acks and test_runner_contract on one contract check (layer 20
window KV, last 128 rows of a 2536-token prompt, PCC 0.98855 < 0.98957, from P.9; DS41-141). The owner waived it for
now, but only component / swap tests read `tests.waivers`; a contract test's check had no waiver at all. Fix:
`tests.unit_waivers: {<test file stem>: {match: [regex, ...], why}}` (core/stamps.unit_waivers). After a unit fails
(not a hang, not a box stop), core/gate.waived_failures passes it only when every failure in its pytest output is an
AssertionError whose every message line matches a pattern; a second failure, an ERROR, a crash with no failure lines
is never waived. The log gets "# <unit>: WAIVED (owner, tests.unit_waivers): ..." and the stamp record `waived`; the
waiver is part of the unit's stamp (waiver_key), so a changed waiver re-runs only its own unit. Selftest
test_core.py::test_an_owner_unit_waiver_passes_only_the_named_contract_failures; checked against the real FIX.12 logs
of both tests. Selftest suite: 651 pass, 2 failing before and after (test_e2e_runtime
test_no_runner_contract_test_takes_a_fixed_threshold: xing40 server_rules.py:393; test_fork_cases
test_a_volatile_cache_key_does_not_change_a_calls_signature: timeout > 300 s) - to look at.

## F226: owner waivers for end-to-end states (DeepSeek-V4.1 2x2 lowbit, owner 2026-10-29)

A step's owner waiver (F85, spec tests.waivers) covers its component and swap checks, but the ladder's and the
profile's state checks (err_state_<name>_L<i> vs calib x the reference's drift) had no waiver path: an accepted format
change that moves a state past its limit (the 4-bit index keys / query: the top-k state of L14 0.185 vs 0.165) could
only fail the gate or loosen the limit. Fix: `ladder.check_states(..., waivers)` (callers pass spec tests.waivers):
a state over its limit with a waiver `state.<name>` is printed WAIVED with the reason and recorded
`<metric>_waived = 1`, which core/gate.py already reports as WAIVED; every other state, and the same state without a
waiver, still fails. Limits unchanged. Selftest: test_e2e_limits.py::
test_a_state_over_its_limit_passes_only_with_its_owner_waiver_and_is_reported.

## F228: an A/B side whose run did not finish is never reused (DeepSeek-V4.1 P.16, 2026-10-30)

The P.16 'on' side was SIGKILLed at model load (host RAM, rc 1, no metric); `decide --rerun` then reused it as "same
code and switch" (the stamp matched) and the report showed nothing for the change. Fix: `ab_measure` reuses a kept run
only when `ab_finished(result)`: it passed, or it recorded a measurement or a failed check; a run killed before any
test ran is measured again, the finished side stays reused. (The 09:55 "F229" was not a bug: the second resume
correctly waited, the rerun decision having been consumed by the first.) Selftest: test_perf_ab.py::
test_a_side_killed_before_any_test_is_measured_again_on_rerun.

## F230: a perf pick without a switch (a branch merge) measures 'before' first; profiles at positions in the ledger (DeepSeek-V4.1, owner 2026-10-30)

The overseer took a MoE block drop (another branch) by merging it itself and profiled 50k / 500k with scripts outside
the tree (posprof plugins); none of it reached the ledger. Owner: the overseer queues, agents change code, and a branch
merge runs "before" before merging and "after" after. Fix:
- `ab: {before: true[, env: {...}]}` (orchestrator.ab_before): before the agent starts, perf_step measures the 'off'
  side on the tree as it is (ab_measure_before; `ab.env` is what both sides run with); then the agent merges, then
  'on'. The 'before' runs are kept whatever code the tree holds later (stamp without the code key), never re-measured on
  the changed tree; missing 'before' runs on a tree already changed under the task's paths wait for a person.
- `ab_positions: [<start>, ...]`: per position one item (test_profile.py at BRINGUP_POS with BRINGUP_PROFILE_HOST=1,
  plus the clean-wall split): device time per section, warm wall, host cProfile, on and off, in rec.positions and the
  table ("One chunk at position ...", with on - off).
- testing/profile.py: BRINGUP_POS = one chunk at that start, every layer, zero prefix and random tokens (timing only,
  nothing compared; the plugin's logic, now in the tree); BRINGUP_PROFILE_HOST=1 adds HOST_RUNS warm runs under cProfile
  after the timed wall runs (host_cprofile_ms, host_launch_ms / _calls, <task>_host<sfx>.json top functions, .pstats).
  core/metrics: BRINGUP_METRIC_SUFFIX (set to _p<pos> by a position run) names every metric, so one gate holds several.
- skill: "The overseer queues; it never does the work".
Selftests: test_perf_ab.py::test_a_merge_pick_measures_before_then_lets_the_agent_change_the_tree,
test_a_merge_picks_before_side_is_never_measured_on_the_changed_tree,
test_a_merge_pick_without_its_before_side_on_a_changed_tree_waits; test_e2e_runtime.py::
test_a_profile_at_a_position_records_timing_with_its_suffix_and_the_host_profile.

## F231 / F232 / F233: an imported branch's knobs, the settings check over the whole change, a switch-flip pick (DeepSeek-V4.1 P.17 / P.20, 2026-10-30)

- F231: P.17 took a MoE block drop as is; its new environment knobs in the combine_rf fork were charged to the agent
  (F106 knew only lines already at HEAD), whose only ways out (convert or mark them diagnostic) altered the drop.
  settings_lint.violations(imports=...) / tasks.yaml `imports: [<ref>]`: a fork line verbatim in an imported ref's
  version of the file is the import's own. Selftest test_model_settings.py::
  test_lint_counts_an_imported_branchs_knobs_as_the_forks_own.
- F232: the check ran on the last attempt's diff only, so P.17's attempt 3 passed because attempt 1 had made the
  edits. It now checks everything uncommitted under the task's paths (last_changed | changed_under).
- F233: a perf pick whose agent changes nothing measured nothing; `ab: {flip: true}` measures both sides anyway (a pick
  that only flips switches the code already has). Selftests test_perf_ab.py::
  test_a_flip_pick_measures_existing_switches_with_no_code_change, test_without_flip_an_unchanged_tree_is_not_measured.
- Open: a pause does not stop a task's A/B measurement (P.17's ran after PAUSE was set); only agent attempts and task
  boundaries honour it.

## F235: a decided perf pick is applied before a queued fix is placed (DeepSeek-V4.1 FIX.15 after P.20, 2026-10-31)

The overseer rejected P.20 and released FIX.15 in one pause; on resume the fix was placed first and its agent started
on P.20's uncommitted (rejected) changes, which its commit would have swept in. run() now applies decided picks (an
accept / reject recorded, not yet committed or reverted: decided_picks) before place_fixes, and runs them first.
Selftest test_perf_ab.py::test_a_decided_pick_is_applied_before_a_queued_fix_is_placed (fails without the fix).
Also (no code): agents' rule 19 and the skill: no single-card device job on a Galaxy whose mesh a run depends on (a hung
one gets a single-card reset that breaks the mesh), and no reset by any back door (dirty files, provoked runs).

## F236: timed profile runs take the consecutive-chunk path (model hook steady_state; DeepSeek-V4.1, owner 2026-10-31)

Every warm-wall, cProfile and profiled run repeats the same chunk, so a model whose state has a fast path for a chunk
that follows a whole previous chunk paid the slow path in every measurement (DS41: the window carry rebuilt, 120 ops,
1.2 ms device / 16 ms host a chunk, that consecutive serving chunks skip). testing/profile.py: timed_reload() calls
model.steady_state(start) (when the device model has it) before every timed run; accurate_reload() calls
steady_state(None) before the quick per-layer accuracy run and the counting run, so their values still come from the
golden prefix on the normal path. A model without the hook is unchanged. Selftest test_e2e_runtime.py::
test_timed_runs_take_the_steady_state_path_and_accuracy_runs_the_normal_one. DS41's hook: FIX.16.
