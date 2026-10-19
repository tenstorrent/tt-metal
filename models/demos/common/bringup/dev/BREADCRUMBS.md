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
- Not done: results per box. One ledger per model, so running the same model on a second box re-runs its tasks in
  the same results folder; separating them means a ledger per box, worth doing when a model first moves boxes.
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
