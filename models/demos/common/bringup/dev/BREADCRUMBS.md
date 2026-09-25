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
