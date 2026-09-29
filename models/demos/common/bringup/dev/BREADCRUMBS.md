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
## F48 (2026-09-28): mixed per-layer state (GLM-5.3 intake)
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
  recurrence with an HF twin). 14 of the 15 fail on the pre-F48 code.

## F49 (2026-09-29): gate commits carry the shared paths agents may change (GLM-5.3 run)

- Symptom: GLM-5.3 C.dsa_moe.attention extended the ttnn.bringup sdpa fork (`high_precision` for sparse_sdpa: C++,
  CHANGELOG, INDEX, source.yaml/baseline, a new unit test) and its gate passed, but the gate commit 3ef3d142602 held
  only the model files; the fork edits stayed uncommitted in the working tree. Earlier gates had left agents'
  known_issues.md / repo_map.md entries uncommitted the same way.
- Cause: `orchestrator.allowed_paths` lets every step write `ttnn/ttnn/bringup` (BRINGUP_OPS) and the knowledge files
  (common_paths), but `core/gate.py:stage_paths` never listed them, so `git_commit` (explicit paths) skipped them.
- Fix: stage_paths appends `ttnn/ttnn/bringup` and the two knowledge files when they exist. `git add -A -- <dir>` also
  picks up new files under the fork (the new unit test).
- Selftests 212 -> 214 (test_fork_cases: the shared paths are staged and committed, including an untracked test
  file; missing shared paths are skipped). The first fails on the pre-F49 gate.py.
- Recovered by hand for GLM: the sdpa fork change and the knowledge entries committed after C.dsa_moe.attention
  (supervision.md).
