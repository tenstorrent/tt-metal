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
