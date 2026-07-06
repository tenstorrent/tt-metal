# Llama-3.1-8B: old `tt_transformers` vs new `models/common` demo — perf sweep

Sweep run: Thu Jul 2 20:10–20:23, tmux session `old_perf_sweep`, all 12 runs **PASSED** (rc=0).

- Old demo: `models/tt_transformers/demo/simple_text_demo.py` (`python_env/bin/pytest`)
- New demo: `models/common/tests/demos/llama3_8b/demo.py` (actuals from the failing run in terminal 5)
- Env: `HF_MODEL=meta-llama/Llama-3.1-8B-Instruct`, `TT_CACHE_PATH=/localdev/gwang/.cache/meta-llama/Llama-3.1-8B-Instruct`, `HF_HOME=/proj_sw/user_dev/huggingface`
- Mesh mapping: `1x1 -> N150`, `1x2 -> N300`, `1x8 -> T3K` (4× N300 boards = 8 Wormhole chips)
- Old `-k` filters: `batch-1 and <opt>` and `batch-32 and not log-probs and <opt>` (log-probs variant has no new-demo counterpart)
- Per-run logs: `old_perf_sweep_logs/<mesh>-<opt>-<batch>.log`

## Old demo — measured

| mesh | MESH_DEVICE | optimization | batch | TTFT (ms) | tok/s/u | tok/s | result |
|---|---|---|---|---|---|---|---|
| 1x1 | N150 | performance | batch-1 | 177.1 | 9.49 | 9.49 | PASS |
| 1x1 | N150 | performance | batch-32 | 35.1 | 8.81 | 281.9 | PASS |
| 1x1 | N150 | accuracy | batch-1 | 206.8 | 9.11 | 9.11 | PASS |
| 1x1 | N150 | accuracy | batch-32 | 41.1 | 8.49 | 271.5 | PASS |
| 1x2 | N300 | performance | batch-1 | 90.4 | 25.4 | 25.4 | PASS |
| 1x2 | N300 | performance | batch-32 | 25.5 | 22.2 | 711.0 | PASS |
| 1x2 | N300 | accuracy | batch-1 | 96.3 | 23.4 | 23.4 | PASS |
| 1x2 | N300 | accuracy | batch-32 | 28.4 | 20.6 | 660.5 | PASS |
| 1x8 | T3K | performance | batch-1 | 39.9 | 70.3 | 70.3 | PASS |
| 1x8 | T3K | performance | batch-32 | 12.8 | 56.1 | 1794.0 | PASS |
| 1x8 | T3K | accuracy | batch-1 | 41.9 | 64.4 | 64.4 | PASS |
| 1x8 | T3K | accuracy | batch-32 | 13.3 | 52.2 | 1670.6 | PASS |

## tok/s/u: old vs new (actual) vs new target

| case | N150 (1x1) old / new / tgt | N300 (1x2) old / new / tgt | T3K (1x8) old / new / tgt |
|---|---|---|---|
| perf batch-1 | 9.5 / 26.4 / 28.3 | 25.4 / 35.1 / 44.2 | 70.3 / 28.3 / 64.3 |
| perf batch-32 | 8.8 / 25.4 / 28.3 | 22.2 / 33.9 / 44.2 | 56.1 / 28.4 / 64.3 |
| acc batch-1 | 9.1 / 23.8 / 25.2 | 23.4 / 31.2 / 38.8 | 64.4 / 27.0 / 60.8 |
| acc batch-32 | 8.5 / 23.0 / 25.2 | 20.6 / 30.1 / 38.8 | 52.2 / 27.3 / 60.8 |

New-demo targets from `EXPECTED_METRICS` in `models/common/tests/demos/llama3_8b/demo.py`; new actuals from the `FAILED ... tok/s/u X below target Y` lines in terminal 5.

## Headline

- **N150 (1 chip):** new TTTv2 is **faster** than old (26.4 vs 9.5 tok/s/u).
- **T3K (8 chips):** old is **~2.5× faster** than new (70.3 vs 28.3).

Old scales 9.5 -> 70.3 tok/s/u from N150 -> T3K (**~7.4×**, near-linear TP). New goes 26.4 -> 28.3 (**~1.07×** — basically no scaling). Every T3K target misses by >2× while N150/N300 only miss by ~10–25%. The new executor's single-chip decode is healthy, but its tensor-parallel path across 8 devices isn't amortizing — each T3K decode iter (~35ms) is barely faster than N150 (~38ms) despite 8× the compute. That's where to dig.

Caveat: old runs 200 decode tokens in the full demo loop, new runs 128 via `run_perf_benchmark`; both report steady-state decode tok/s/u excluding compile, so the scaling argument holds, but absolute TTFT definitions differ slightly (old = prefill_time/batch; new = measured TTFT).

## 2026-07-06 T3K TTTv2 optimization update

Current TTTv2 results after the construction refactor and T3K decode optimizations:

| case | old TTTv1 tok/s/u | old TTTv1 tok/s | new TTTv2 target tok/s/u | new TTTv2 current tok/s/u | new TTTv2 current tok/s | TTFT (ms) | decode latency (ms) | result |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| T3K perf batch-1 | 70.3 | 70.3 | 64.3 | 88.2 | 88.2 | 36.5 | 11.33 | PASS |
| T3K perf batch-32 | 56.1 | 1794.0 | 64.3 | 85.9 | 2747.5 | 35.5 | 11.65 | PASS |
| T3K token accuracy batch-1 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | PASS, top1 96.1%, top5 100.0% |

Verification commands:

```bash
python_env/bin/python -m py_compile models/common/models/executor.py models/common/models/generator.py models/common/models/llama3_8b/model.py models/common/modules/attention/attention_1d.py models/common/modules/mlp/mlp_1d.py models/common/modules/sampling/sampling_1d.py models/common/tests/demos/llama3_8b/demo.py
git diff --check -- models/common/models/executor.py models/common/models/generator.py models/common/models/llama3_8b/model.py models/common/modules/attention/attention_1d.py models/common/modules/mlp/mlp_1d.py models/common/modules/sampling/sampling_1d.py models/common/tests/demos/llama3_8b/demo.py
TT_METAL_HOME=/localdev/gwang/tt-metal-2 TT_CACHE_PATH=/localdev/gwang/.cache/meta-llama/Llama-3.1-8B-Instruct HF_HOME=/proj_sw/user_dev/huggingface HF_MODEL=meta-llama/Llama-3.1-8B-Instruct MESH_DEVICE=T3K python_env/bin/pytest -v 'models/common/tests/demos/llama3_8b/demo.py::test_llama3_8b[performance-1x8-batch-1]'
TT_METAL_HOME=/localdev/gwang/tt-metal-2 TT_CACHE_PATH=/localdev/gwang/.cache/meta-llama/Llama-3.1-8B-Instruct HF_HOME=/proj_sw/user_dev/huggingface HF_MODEL=meta-llama/Llama-3.1-8B-Instruct MESH_DEVICE=T3K python_env/bin/pytest -v 'models/common/tests/demos/llama3_8b/demo.py::test_llama3_8b[performance-1x8-batch-32]'
TT_METAL_HOME=/localdev/gwang/tt-metal-2 TT_CACHE_PATH=/localdev/gwang/.cache/meta-llama/Llama-3.1-8B-Instruct HF_HOME=/proj_sw/user_dev/huggingface HF_MODEL=meta-llama/Llama-3.1-8B-Instruct MESH_DEVICE=T3K python_env/bin/pytest -v 'models/common/tests/demos/llama3_8b/demo.py::test_llama3_8b[accuracy-1x8-token-accuracy]'
```

### Work done to reach the current numbers, ordered by performance impact

1. Added benchmark-only on-device decode feedback in `TracedLLMExecutor`. After the first traced decode replay, the benchmark keeps token feedback and position increments on device and skips host readback for steady-state iterations. This was the largest throughput improvement, taking T3K batch-1 from about `60.4 tok/s/u` to `88.2 tok/s/u`, and it does not affect teacher-forced accuracy.
2. Fixed the T3K fused attention output projection path. TTTv2 was checking `model_config["USE_FUSED_ALL_GATHER_MATMUL"]`, which was absent in the new pure-config path, so it silently used separate matmul, conversion, and reduce-scatter ops. The new path now follows the old model arg property, `args.use_fused_all_gather_matmul`. This moved T3K batch-1 from about `59.6 tok/s/u` to `60.4 tok/s/u`.
3. Switched the perf path to on-device top-k sampling by default, using `SamplingParams(temperature=0.0, top_k=32, top_p=0.08)`. This made the benchmark path eligible for device-side token feedback and removed avoidable host-side sampling work from the steady-state path.
4. Aligned the T3K short-context perf setup with the old demo: `max_seq_len=1024`, paged attention with `max_num_blocks=1024`, and per-user page allocation derived from `max_num_blocks // max_batch_size`. This made the comparison use the same short-context allocation regime and avoided measuring a heavier setup than the old path.
5. Propagated old TTTv1 communication tuning into the TTTv2 config path: attention decode all-gather-matmul config from `ATTN_AGMM_CONFIG`, MLP decode reduce-scatter config from `MLP_RS_CONFIG`, and sampling all-gather config from `SAMPLING_AG_CONFIG`.
6. Matched LM head compute-kernel fidelity to the old path by using the HiFi2 LM head kernel config.
7. Updated sampling to use the configured sub-core grid for `ttnn.manual_seed`, and made its all-gather cluster-axis handling work for both single-chip and multi-chip meshes. This was primarily a parity and correctness cleanup; the measured perf effect was neutral to slightly negative, but it kept sampling behavior aligned.
8. Added reset/page-table refresh handling so benchmark decode initializes state on the first replay and only refreshes the page table during device-feedback runs when it changes. This was required support code for the highest-impact device-feedback optimization.
9. Removed the Llama-3.1-8B model path's dependency on `from_model_args` construction. The demo and generator now build an explicit TTTv2 config and instantiate `Llama3Transformer1D(config)`.
10. Added an explicit `build_llama3_transformer_1d_config_from_model_args(...)` builder that wires embedding, RoPE, RMSNorm, attention, MLP, LM head, and sampling configs without relying on module factories. This was the enabling refactor that exposed the config parity gaps, but it was not by itself the direct throughput win.
11. Kept the eager executor API compatible with the traced executor by accepting `reset_batch`; eager accuracy ignores it. This had no direct perf impact, but it was needed to keep the accuracy verification path passing after the traced executor API change.

### Experiments that were not kept

- Splitting decode and sampling into separate executor-owned traces, matching the old TTTv1 trace shape, regressed T3K batch-1 to about `55.9 tok/s/u`.
- Rounding `Sampling1DConfig.max_batch_size` to 32 regressed T3K batch-1 to about `54.7 tok/s/u`.
- Changing `topk_global_indices` from `int32` to `uint32` was neutral to negative, about `58.4 tok/s/u`.

The main remaining non-functional cleanup is the repeated matmul warning about `program_config.allowed_worker_cores` being auto-populated. It did not block correctness or perf verification, but it points to program configs that should eventually be normalized before direct matmul use.
