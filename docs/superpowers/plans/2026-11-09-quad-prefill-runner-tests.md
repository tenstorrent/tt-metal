# Quad prefill-runner hardware screen Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `gwdt --quad` runs three weight-free prefill-runner screens on 1 or 4 galaxies and names the bad galaxy or link.

**Architecture:** Two fake adapters (`quad_passthrough`, `quad_block_random`) plug into the real `prefill_runner.py`. Each rank's runtime reads back its input/output per device, writes `rank{r}.json` + `rank{r}.hashes.jsonl`. A new `gwdt_quad.py` launches runner + producer, and turns the rank files into per-stage verdicts.

**Tech Stack:** Python 3.10+, ttnn, torch, tt-run/mpirun, pytest.

**Spec:** `docs/superpowers/specs/2026-11-09-quad-prefill-runner-tests-design.md`

## Global Constraints

- Shapes as `block-random`: chunk 5120, 2 chunks per request, K2.7 L0 dense + L1 MoE, weight seed 42, input seed 1234.
- `PREFILL_NUM_USERS=16` (same prompt rotates over 16 KV slots so no in-flight slot is re-sent), `PREFILL_MAX_SEQ_LEN=10240`, `PREFILL_USE_TRACE=0`, `PREFILL_LAYER_ACK_D2H=0`, `PREFILL_TP_SHARD_KV=1`.
- `PREFILL_NUM_LAYERS = 2 * ranks`.
- Budgets: passthrough 3 requests; block-random smoke 300 s; soak 900 s default, `--soak-minutes`.
- Fabric `2d_torus_xy` first, fallback `2d`.
- Watchdog: 300 s no progress → HANG; setup > 2700 s → HANG.
- No weka, HF download, or TTNN cache on any path.
- Code comments: why only, present tense (`tt-buddy` `code-comments.md`).
- `PREFILL_*` env set inside the `bash -lc` string under tt-run.

## Review Focus

- Rank file half-written when read → reader must tolerate and retry (Task 1 test `test_load_rank_tolerates_partial_json`).
- Producer stops mid-request at budget → last partial iteration must not count as mismatch (Task 2 test `test_partial_last_iteration_ok`).
- Iteration 0 all zeros (vacuous) → FAIL, not PASS (Task 2 test `test_vacuous_baseline_fails`).
- Rank never wrote a file (died at bring-up) → FAIL "rank k never started" (Task 1 test `test_missing_rank_fails`).
- Wrong `--hosts` order with descriptors present → exit 2, no FAIL row (Task 5 test `test_ring_order_mismatch_exits`).

---

### Task 1: Rank-file verdict logic

**Files:**
- Create: `models/demos/deepseek_v3_d_p/scripts/gwdt_quad_verdict.py`
- Test: `models/demos/deepseek_v3_d_p/scripts/test_gwdt_quad_verdict.py`

**Interfaces:**
- Produces:
  - Rank file schema (`rank{r}.json`): `{"rank": int, "host": str, "mode": "passthrough"|"block_random", "fabric": str, "chunks_done": int, "iterations": int, "last_in": [iter, chunk] | null, "last_out": [iter, chunk] | null, "last_progress": float, "first_bad": {...} | null, "mismatches": [{"iter", "chunk", "what": "input"|"output", "devices": [[r, c], ...], "n_diff": int}], "error": str | null, "done": bool}`.
  - Hash file (`rank{r}.hashes.jsonl`): one line per chunk: `{"iter", "chunk", "in": [hex per device], "out": [hex per device]}`.
  - `load_ranks(results_dir: Path, n_ranks: int) -> dict[int, dict | None]`
  - `stage_verdicts(ranks: dict[int, dict | None], hashes: dict[int, list[dict]], hosts: list[str], hung: bool) -> list[tuple[str, Result]]` — one `("stage k host", Result)` per stage, pipeline order.
  - `overall(rows) -> Result` — earliest non-PASS row wins; reason names galaxy/link.

- [ ] **Step 1: Write failing tests** — `test_all_pass`, `test_output_mismatch_is_galaxy_compute` (stage 2 output differs → `FAIL galaxy 2 (<host>) compute, chips (1,3)`), `test_input_mismatch_is_link` (stage 2 input differs, stage 1 output OK → `FAIL link 1→2`), `test_stage0_input_is_h2d`, `test_cross_stage_hash_mismatch_is_link`, `test_hang_link` (k-1 `last_out` ahead of k `last_in`), `test_hang_compute` (k `last_in` ahead of `last_out`), `test_missing_rank_fails`, `test_load_rank_tolerates_partial_json`, `test_rank_error_fails`.
- [ ] **Step 2: Run** `pytest -q --noconftest test_gwdt_quad_verdict.py` → FAIL (module missing).
- [ ] **Step 3: Implement** `gwdt_quad_verdict.py`; import `Result, Status` from `gwdt_checks`.
- [ ] **Step 4: Run** → PASS.
- [ ] **Step 5: Commit** `deepseek_v3_d_p/scripts: add quad screen per-stage verdicts`.

### Task 2: StageMonitor (host-side check + rank files)

**Files:**
- Create: `models/demos/deepseek_v3_d_p/tt/runners/quad_stage_monitor.py`
- Test: `models/demos/deepseek_v3_d_p/tests/test_quad_stage_monitor.py` (torch only, no device)

**Interfaces:**
- Consumes: rank-file schema from Task 1.
- Produces: `class StageMonitor(rank: int, mode: str, results_dir: Path, mesh_cols: int, fabric: str)` with
  - `check(what: str, chunk: int, shards: list[torch.Tensor], expected: list[torch.Tensor] | None = None) -> None` — `what` in `{"input","output"}`; `expected=None` means vs iteration-0 baseline; `chunk == 0 and what == "input"` advances the iteration.
  - `flush() -> None` (atomic write of `rank{r}.json`, append hash line once both in/out for a chunk are seen).
  - `fail(error: str) -> None`, `finish() -> None` (`done=True`).
- Readback helper (same file): `device_shards(t: ttnn.Tensor) -> list[torch.Tensor]` via `ttnn.get_device_tensors` + `ttnn.to_torch`.
- Device i → `(i // mesh_cols, i % mesh_cols)`.

- [ ] **Step 1: Failing tests** — `test_baseline_then_equal_passes`, `test_mismatch_records_device_coords`, `test_expected_pattern_mode`, `test_vacuous_baseline_fails`, `test_partial_last_iteration_ok`, `test_hash_line_written_per_chunk`, `test_mismatch_list_capped_at_20`.
- [ ] **Step 2: Run** (laptop: `uv run --with torch --with loguru pytest -q --noconftest`) → FAIL.
- [ ] **Step 3: Implement.** Hash = `hashlib.sha1(t.contiguous().view(torch.uint8).numpy()).hexdigest()[:16]` on raw bytes. Import `ttnn` lazily inside `device_shards` so tests run without ttnn.
- [ ] **Step 4: Run** → PASS.
- [ ] **Step 5: Commit** `deepseek_v3_d_p: add quad screen stage monitor`.

### Task 3: Adapters + runtimes + manifests

**Files:**
- Create: `models/demos/deepseek_v3_d_p/tt/runners/adapters/quad_screen.py`
- Create: `models/demos/deepseek_v3_d_p/tt/runners/quad_screen_runtime.py`
- Create: `models/demos/deepseek_v3_d_p/tt/runners/manifests/quad_passthrough.json`, `quad_block_random.json`
- Modify: `models/demos/common/prefill/adapter.py` (`ADAPTER_PATHS`: two lines)

**Interfaces:**
- Consumes: `StageMonitor`, `device_shards` (Task 2); `TtPrefillRuntime`, `TtPrefillRuntimeConfig`; `create_hf_model`, `extract_layer_state_dict` (`utils/transformer_helpers.py`); `prepare_prefill_input_tensor`.
- Produces:
  - `QuadPassthroughAdapter(KimiK27Adapter)`, `name="quad_passthrough"`; `QuadBlockRandomAdapter(KimiK27Adapter)`, `name="quad_block_random"`.
  - `embedding_like_hidden(total_len: int, hidden: int, seed: int = 1234) -> torch.Tensor` (moved copy of the test helper; test keeps its own).
  - Env read by runtimes: `PREFILL_QUAD_RESULTS_DIR` (required), `PREFILL_FABRIC_MODE` (recorded only).
- Runtime rules (spec "Runtimes"):
  - Inner `TtPrefillRuntime` built with `is_first_rank=False, is_last_rank=False, kv_only_last_layer=False, weight_cache_path=None`, `model_cfg` = subclass of `KimiK27Config` with `NUM_DENSE_LAYERS = first_layer_idx + 1`.
  - State dict: `torch.manual_seed(42)`; `hf = create_hf_model(variant, cfg, 2)`; layers `[extract_layer_state_dict(..., 0, ...), extract_layer_state_dict(..., 1, ...)]`.
  - Wrapper exposes `config` with real roles; `__getattr__` delegates the rest to inner.
  - `prefill_chunk`: readback input → `check("input")`; free input; copy pre-uploaded hidden slice for `actual_start // chunk_size`; inner forward; readback output → `check("output")`; last rank frees and returns None.
  - Passthrough: subclass `TtPrefillRuntime`, `_build_model` sets `self.model=None; self.model_built=True`; layers = `ttnn.clone`, fire completion per global layer; check vs host pattern shards.
  - `compile()` must not record checks (monitor enabled after compile).

- [ ] **Step 1: Write** `quad_screen.py`, `quad_screen_runtime.py`, manifests, registry lines.
- [ ] **Step 2: Import check (cluster):** `python -c "from models.demos.common.prefill.adapter import get_adapter; get_adapter('quad_block_random'); get_adapter('quad_passthrough')"` → no error.
- [ ] **Step 3: Commit** `deepseek_v3_d_p: add quad screen fake adapters for the prefill runner`.

### Task 4: Producer drains acks without PCC

**Files:**
- Modify: `models/demos/common/prefill/runners/prefill_producer.py` (main: completion channel + final check)

**Interfaces:**
- Produces: env `PREFILL_PRODUCER_DRAIN_ACKS=1` → connect the completion channel when `verify` is off; exit 1 if received ≠ expected completions.

- [ ] **Step 1: Implement** (≤10 lines; default off, no behaviour change otherwise).
- [ ] **Step 2: Run** existing `models/demos/common/prefill/tests/test_prefill_producer_forwards.py` on cluster → PASS.
- [ ] **Step 3: Commit** `prefill/producer: drain layer acks without PCC when asked`.

### Task 5: `gwdt --quad` launcher and steps

**Files:**
- Create: `models/demos/deepseek_v3_d_p/scripts/gwdt_quad.py`
- Modify: `models/demos/deepseek_v3_d_p/scripts/galaxy_workload_deployment_test.py` (`--quad`, `--hosts`, `--soak-minutes`; QUAD section in `Screen`)
- Test: `models/demos/deepseek_v3_d_p/scripts/test_gwdt_quad.py`

**Interfaces:**
- Consumes: Task 1 verdicts; `Step, State, Ctx, sh, with_fallbacks`-style fallback from `gwdt_steps`.
- Produces:
  - `build_quad_steps(hosts: list[str], soak_s: int) -> list[Step]`: keys `quad:ring-order`, `quad:reset`, `quad:passthrough-smoke`, `quad:block-random-smoke`, `quad:block-random-soak`; test steps group `"quad"`.
  - `ring_order(repo: Path, hosts: list[str]) -> list[str] | None` (None when descriptors missing).
  - `runner_cmd(repo, hosts, model, fabric, results_dir, n_layers) -> str`, `producer_cmd(repo, rank0_host, model, results_dir, tokens_dir, duration_s, max_requests) -> str`.
  - `write_tokens(dir: Path, n: int, vocab: int, seed: int = 1234) -> Path` (metadata.json `token_ids`).
  - MGD: 4 hosts torus `ci/sc4_mgd.textproto`, mesh `pipeline_prefill_4galaxy_connected_fabric2d_mesh_graph_descriptor.textproto`; 1 host `ci/sc1_mgd.textproto`.
  - Stage sub-rows go in `state.rows` with labels in `Result.details` (`"stage k host: reason"`).

- [ ] **Step 1: Failing tests** — `test_quad_keys_in_order`, `test_runner_cmd_exports_prefill_env_inside_bash` (asserts `PREFILL_MANIFEST`, `PREFILL_FABRIC_MODE`, `PREFILL_QUAD_RESULTS_DIR`, `PREFILL_NUM_LAYERS=8` all inside the `bash -lc` string), `test_mgd_by_host_count_and_fabric`, `test_producer_cmd_budget_and_drain`, `test_write_tokens_deterministic`, `test_ring_order_mismatch_exits`, `test_passthrough_blocks_only_when_both_modes_fail`, `test_cli_quad_requires_hosts`.
- [ ] **Step 2: Run** `pytest -q --noconftest test_gwdt_quad.py` → FAIL.
- [ ] **Step 3: Implement.** Flow per test step: per-host `tt-smi -glx_reset_auto` + shm purge over ssh (parallel) → Popen runner → poll ranklogs for `setup complete, entering request loop` × n ranks (or runner exit / 2700 s) → producer via `mpirun -np 1 --host <rank0>` (rank 0 host from `generated/ttrun/*/rankfile`, else `hosts[0]`) → poll rank files (300 s no-progress watchdog) → wait runner ≤120 s → verdict. Fallback: rerun with `fabric="2d"` as a `State` fallback sub-row.
- [ ] **Step 4: Run** → PASS; also run `test_gwdt_checks.py` → still 80 pass.
- [ ] **Step 5: Commit** `deepseek_v3_d_p/scripts: add gwdt --quad prefill-runner screen`.

### Task 6: Device bring-up on one galaxy

**Files:** fixes only, as found.

- [ ] **Step 1:** Sync branch to `/data/mbezulj/tt-metal-3-quad`; run Task 2–5 unit tests in cluster venv → PASS.
- [ ] **Step 2:** `gwdt --quad --hosts <1 host> --only quad:passthrough-smoke` via sbatch → PASS, 1 stage row.
- [ ] **Step 3:** same for `quad:block-random-smoke` → PASS.
- [ ] **Step 4:** Manual 4-rank passthrough on one galaxy with `pipeline_prefill_request_intragalaxy_4rank.yaml` (`PREFILL_SP=2`) → 4 stage files, all PASS.
- [ ] **Step 5:** Fault check: env `PREFILL_QUAD_INJECT=<rank>:<what>` flips one bf16 bit in that rank's readback copy → verdict names that stage. (Test-only hook in `StageMonitor`; reads env once.)
- [ ] **Step 6:** Commit fixes; sync to the cluster worktree; note results via `tt-buddy:note`.
