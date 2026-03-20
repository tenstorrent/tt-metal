# Build / benchmark protocol memory (tt-metal)

## Matmul N150 — Milestone 1 (local protocol scaffolding)

### Flow (confirmed in code)

1. **`sweeps_parameter_generator.py`** imports `sweeps.model_traced.matmul_model_traced` -> `MasterConfigLoader().get_suite_parameters("matmul")` fills `parameters["model_traced"]` from `model_tracer/traced_operations` master JSON.
2. **Permutations** expand the suite; vectors get **`input_hash`** (SHA224 of serialized vector, or **`config_hash`** from DB when present) and are written under **`tests/sweep_framework/vectors_export/`** (base name + optional **`__mesh_*`** suffix by mesh shape).
3. **`sweeps_runner.py`** loads vectors (`vectors_export` or **`--vector-source file`**), runs `run()` in the sweep module, exports **`tests/sweep_framework/results_export/<short>_<digest>_<ts>.json`** as a **JSON list** of OpTest-shaped records (`status`, `input_hash`, `metrics` with `e2e_perf_ms` when `--perf`).

### What we added

- **`tests/sweep_framework/benchmark_protocol/`**: deterministic **smoke / train / holdout** split (`matmul_n150_partition.py`), **`matmul_n150_protocol.py`** CLI (`partition` | `write-json` | `report`), **`matmul_n150_report.py`** aggregates pass/timeout/p50/p95/memory vs manifest.
- **`run_matmul_n150_protocol.sh`**: `generate` -> `partition` -> `write-json` -> `run` (from `tests/sweep_framework/` so `framework.*` imports resolve) -> `report`.
- **Docs**: `docs/benchmark_protocol/matmul_n150_milestone1.md`, `docs/benchmark_protocol/matmul_n150_acceptance_policy.md`.

### Master trace resolution (gotcha)

`MasterConfigLoader` used to only pick `ttnn_operations_master_v2_reconstructed.json` or `ttnn_operations_master_UF_EV_B9_GWH01_deepseek.json`. Many trees only have **`model_tracer/traced_operations/ttnn_operations_master.json`** (includes `ttnn::matmul` configs). Fallback for that file is now in `master_config_loader_v2.py`.

Bundled trace can put `device_series` / `card_count` inside single-element lists; `sweeps_parameter_generator.get_mesh_shape_from_vector` used them as a dict key and crashed (`unhashable type: 'list'`). `_scalar_from_trace_field` now unwraps them.

Bundled `ttnn_operations_master.json` matmul args use `{ "Tensor": { ... } }` under `arg0` / `arg1`. `_get_generic_parameters` was calling `_extract_tensor_config` (which only accepts `type == "ttnn.Tensor"`), so tensors were never promoted to `input_a_*` / `input_b_*`. Fix: use `extract_tensor_config` in that path.

`matmul_n150_protocol.py report`: with `--results-glob` only, argparse leaves `--results` as `None`; use `(args.results or [])` before iterating.

Bundled trace `shard_spec.grid` is not always `{start,end}`: some entries are a single core or a corner pair list. `parse_memory_config` now normalizes those forms.

When report input includes multiple `results_export/model_traced_*.json` files, duplicate `input_hash` rows can inflate `results_matched` above `vectors_expected`. Reporter now dedupes by `input_hash` (latest timestamp wins) and emits:
- `total_result_rows_raw`
- `total_result_rows_deduped`
- `duplicate_result_rows_dropped`

Some matmul vectors only had `arg0`/`arg1` in JSON because `input_a_*` were `__ABSENT__` and dropped on export, so `run()` saw missing positional args. `matmul_model_traced.run` now defaults tensor args to `None` and promotes `arg0`/`arg1` via `MasterConfigLoader.extract_tensor_config` + `parse_memory_config`.

### Environment notes

- `sweeps_runner` requires `ARCH_NAME` in `{wormhole_b0, blackhole}`.
- `build_metal.sh` not run: vector generation and hardware sweeps need working tt-metal Python + device; partition/report can still be tested with fake JSON.
- Use `python3` here (no `python` shim).
- Always activate venv: `source /home/ubuntu/tt-metal/python_env/bin/activate`.
- In this environment, `tt-smi` is reliably found only from `python_env/bin`; unactivated runs can fail with `SWEEPS: Unable to locate tt-smi executable`.

### DeepSeek-focused perf loop (small set)

- Strict `--source-include 'deepseek'` partition yields 3 vectors in this repo's `model_traced.matmul_model_traced` export.
- Kept improvement: in `matmul_model_traced.run`, use direct `ttnn.to_torch(output_tensor)` for non-mesh path (better p50/p95 on 3-vector set).
- Reverted non-improvement: `ttnn.to_torch(..., cq_id=0)` did not help.

### Larger DeepSeek-oriented test pool (still N150-safe)

- To reduce tiny-sample noise while staying near decoder-style traces, use `--source-include 'simple_text_demo.py'`.
- Working split override: `--smoke-max 4 --train-fraction 0.5` -> 4 smoke / 4 train / 4 holdout.

### Deeper matmul direction

- DeepSeek-specific `_get_prefill_pc` right-sizing was positive on typical latency for short-M shapes.
- Next broader experiments target:
  - shared TTNN C++ auto matmul config path,
  - shared `models/common/modules/mlp/mlp_1d.py` prefill config path.

### Latest A/B validation snapshot (N150, stash-based before/after)

- Shared C++ auto matmul experiment (`matmul_program_config.cpp`):
  - Used `matmul_n150_deepseek_oriented_manifest.json` (12 vectors), running clean baseline via temporary stash and then after-restore run.
  - All runs passed correctness (12/12).
  - First baseline run had one-time compile overhead and was discarded for fairness.
  - Fair comparison (second baseline vs after) showed no clear win; p95 values were worse after on this sample.
- Shared `MLP1D` experiment (`mlp_1d.py`):
  - Added `tests/sweep_framework/benchmark_protocol/mlp1d_prefill_n150_benchmark.py` for targeted device timing.
  - Expected grid behavior observed: `[8, 8] -> [8, 1]` on short-`seq_len` cases.
  - Measured latency was mixed/slightly worse overall on this N150 sample.

### Additional findings (N150, after initial snapshot)

- Follow-up attempt (C++ auto path, Y-only cap) regressed on the 12-vector DeepSeek-oriented protocol and was reverted.
- Follow-up attempt (C++ transpose-mcast divisibility fix in auto path) did not improve targeted non-square-grid benchmark and was reverted.
- Follow-up attempt (`MLP1D` `prefill_w2_prg_config`, Y-only cap for w2) was mixed:
  - Single A/B snapshot: `overall_p50` slightly worse, `overall_p95` improved.
  - Repeat check (3x vs 3x baseline): still mixed, with mean `overall_p50` worse and mean `overall_p95` better.
  - Conclusion: no stable net win yet under a strict no-regression criterion.
- Boundary-memory A/B (DRAM vs L1 intermediate tensors for two chained matmuls) on 3 repeated runs each:
  - Setup: `tests/sweep_framework/benchmark_protocol/matmul_boundary_memory_n150_benchmark.py`.
  - Mean `overall_p50`: improved with L1 (~-4.2% vs DRAM).
  - Mean `overall_p95`: regressed with L1 (~+9.4% vs DRAM), including one notable tail spike.
  - Conclusion: memory-bound behavior is visible, but naive L1 forcing is not universally safe under strict p50+p95 no-regression requirements.
- Post-rebuild baseline-first adaptive boundary-memory check (3x baseline first, then 3x adaptive):
  - Baseline: DRAM intermediate tensors.
  - Adaptive rule: L1 only for short/moderate shapes (`seq_len <= 64`, `dim <= 1536`, `hidden_dim <= 3072`), DRAM otherwise.
  - Mean `overall_p50`: slightly worse (~+0.4% vs baseline).
  - Mean `overall_p95`: better (~-4.6% vs baseline).
  - Conclusion: adaptive policy improves tail but still fails strict objective criterion because `overall_p50` does not improve.
- Post-rebuild baseline-first subblock-width check in shared `MLP1D` helper (`_get_out_subblock_w`, 4 -> 8) with 3x baseline and 3x after:
  - Setup: `tests/sweep_framework/benchmark_protocol/mlp1d_prefill_n150_benchmark.py`.
  - Mean `overall_p50`: regressed (~+6.4% vs baseline).
  - Mean `overall_p95`: regressed (~+3.1% vs baseline).
  - Conclusion: this change clearly regressed and was reverted.
- Post-rebuild baseline-first boundary-benchmark harness adjustment (force final output to DRAM, keep adaptive policy only on intermediate) with 3x baseline and 3x after:
  - Setup: `tests/sweep_framework/benchmark_protocol/matmul_boundary_memory_n150_benchmark.py`.
  - Mean `overall_p50`: regressed (~+1.5% vs baseline).
  - Mean `overall_p95`: regressed (~+5.9% vs baseline).
  - Conclusion: no improvement under strict criterion; reverted.
- Post-rebuild baseline-first adaptive gate tightening (L1 only at `dim == 1536`, not for smaller `dim`) with 3x baseline and 3x after:
  - Setup: `tests/sweep_framework/benchmark_protocol/matmul_boundary_memory_n150_benchmark.py`.
  - Mean `overall_p50`: improved (~-5.8% vs baseline).
  - Mean `overall_p95`: slightly regressed (~+0.7% vs baseline).
  - Conclusion: mixed result; reverted under strict both-metrics-must-improve policy.

### Exploration infra additions (N150)

- Added regime manifest for low-noise screening:
  - `tests/sweep_framework/benchmark_protocol/matmul_n150_regimes.json`.
  - 8 representative cases spanning decode small-M DRAM-bound, prefill/FF larger-M, and tiny/irregular edge shapes.
- Added kernel-focused microbench harness:
  - `tests/sweep_framework/benchmark_protocol/matmul_n150_kernelbench.py`.
  - Resolves real traced vectors by `input_hash` from `matmul_n150_protocol_all.json`.
  - Reports per-case e2e (`ms`) and optional device-kernel duration (`ns`) in one JSON output.
- Added exploration scoreboard (separate from strict merge gate):
  - `matmul_exploration_scoreboard.md`.
  - Keeps mixed but informative candidates instead of discarding them.
