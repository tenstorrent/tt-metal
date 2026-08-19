# Optimized Decoder Work Log

Model: `Qwen/Qwen3.6-35B-A3B`

Base commit: `4a94c15a3830a08bfb2ff517fd8d83e95fa8cff6`

Branch: `vkovacevic/agentic-research/qb2-qwen36-35b-a3b`

Stage scope: `models/autoports/qwen_qwen3_6_35b_a3b/tt/optimized_decoder.py`, `tests/test_optimized_decoder.py`, and `doc/optimized_decoder/*` only. No multichip, full-model, or vLLM work was started.

## Device Safety

```bash
tt-smi -ls --local
```

Result: four local Blackhole p300c devices visible and resettable. Artifacts:

- `logs/tt_smi_initial.log`
- `logs/tt_smi_before_candidates.log`
- `logs/tt_smi_final_sparsew4_exactnnz.log`

Watcher runs used `TT_METAL_WATCHER_DISABLE_ETH=1` because this stage exercises a single-device decoder-layer path and no Ethernet/fabric data path. The watcher still attached to all four local devices and checked worker progress.

## Static And Collection Checks

```bash
./python_env/bin/python -m py_compile \
  models/autoports/qwen_qwen3_6_35b_a3b/tt/optimized_decoder.py \
  models/autoports/qwen_qwen3_6_35b_a3b/tests/test_optimized_decoder.py \
  2>&1 | tee models/autoports/qwen_qwen3_6_35b_a3b/doc/optimized_decoder/logs/py_compile_final_exactnnz_sparsew4.log
```

Result: passed.

```bash
./python_env/bin/python -m pytest \
  --confcutdir=models/autoports/qwen_qwen3_6_35b_a3b/tests \
  --collect-only -q \
  models/autoports/qwen_qwen3_6_35b_a3b/tests/test_optimized_decoder.py \
  2>&1 | tee models/autoports/qwen_qwen3_6_35b_a3b/doc/optimized_decoder/logs/pytest_collect_final_exactnnz_sparsew4.log
```

Result: 46 optimized tests collected.

## Final Correctness And Watcher

```bash
timeout 1200 env TT_METAL_WATCHER=10 TT_METAL_WATCHER_APPEND=1 \
  TT_METAL_WATCHER_NOINLINE=1 TT_METAL_WATCHER_DISABLE_ETH=1 \
  RUN_QWEN36_REAL_WEIGHTS=1 \
  ./python_env/bin/python -m pytest \
  --confcutdir=models/autoports/qwen_qwen3_6_35b_a3b/tests \
  --timeout=900 \
  models/autoports/qwen_qwen3_6_35b_a3b/tests/test_optimized_decoder.py \
  -k 'not test_perf_qwen36_optimized and not candidate' -s \
  2>&1 | tee models/autoports/qwen_qwen3_6_35b_a3b/doc/optimized_decoder/logs/watcher_correctness_final_sparsew4_exactnnz.log
```

Result: `14 passed, 32 deselected, 2 warnings in 158.77s`.

| Case | Prefill PCC | Traced decode PCC |
| --- | ---: | ---: |
| synthetic linear layer 0, seq 5 | 0.9995203064456892 | 0.9994662304973918 |
| synthetic full layer 3, seq 33 | 0.9995777703980826 | 0.9993898329949268 |
| synthetic batch-2 linear, seq 5 | 0.9994317553119714 | 0.9992362429708346 |
| synthetic batch-2 full, seq 33 | 0.9995916971087269 | 0.9992974376522705 |
| synthetic linear non-aligned seq 65 | 0.9975411512233607 | 0.9994416658202645 |
| synthetic full non-aligned seq 33 | 0.9995777703980826 | 0.9993898329949268 |
| real linear layer 0, seq 1 | 0.9991292848497009 | 0.9987610738665994 |
| real full layer 3, seq 1 | 0.9996961087208918 | 0.9995498281934312 |
| real linear layer 0, seq 5 | 0.9996973622269505 | 0.9986944187478957 |
| real full layer 3, seq 5 | 0.9996233758352069 | 0.9994969304756183 |

Repeated decode determinism: linear PCC `1.0`, full PCC `1.0`.

The run also passed `test_optimized_runtime_fallback_audit_source`, which rejects `torch`, `from_torch`, `to_torch`, fallback-call use, and functional decoder delegation in measured optimized runtime functions.

## Dynamic Fallback Audit

```bash
timeout 900 env TTNN_CONFIG_OVERRIDES='{"throw_exception_on_fallback": true}' \
  RUN_QWEN36_REAL_WEIGHTS=1 \
  ./python_env/bin/python -m pytest \
  --confcutdir=models/autoports/qwen_qwen3_6_35b_a3b/tests \
  --timeout=600 \
  models/autoports/qwen_qwen3_6_35b_a3b/tests/test_optimized_decoder.py \
  -k 'test_real_weight_optimized_decoder_prefill_decode_against_hf or test_optimized_runtime_fallback_audit_source' -s \
  2>&1 | tee models/autoports/qwen_qwen3_6_35b_a3b/doc/optimized_decoder/logs/runtime_fallback_audit_final_sparsew4_exactnnz.log
```

Result: `5 passed, 41 deselected, 2 warnings in 59.40s`. This proves the final real-weight optimized path runs without TTNN dynamic fallback. Hugging Face reference-side Torch warnings in the log are unrelated to the TTNN optimized path.

## Final Performance

```bash
timeout 900 env RUN_QWEN36_OPTIMIZED_PERF=1 RUN_QWEN36_PERF_REAL_WEIGHTS=1 \
  ./python_env/bin/python -m tracy -r -p -v \
  --no-runtime-analysis --op-support-count=5000 --check-exit-code \
  --output-folder models/autoports/qwen_qwen3_6_35b_a3b/doc/optimized_decoder/tracy/final_sparsew4_exactnnz_raw \
  ./python_env/bin/python -m pytest \
  --confcutdir=models/autoports/qwen_qwen3_6_35b_a3b/tests \
  --timeout=600 \
  models/autoports/qwen_qwen3_6_35b_a3b/tests/test_optimized_decoder.py \
  -k test_perf_qwen36_optimized -s \
  2>&1 | tee models/autoports/qwen_qwen3_6_35b_a3b/doc/optimized_decoder/logs/tracy_perf_final_sparsew4_exactnnz_summary.log
```

Result: `4 passed, 42 deselected`.

Warmed walls:

| Window | Wall time |
| --- | ---: |
| `OPT_LINEAR_PREFILL` | 20.414 ms |
| `OPT_FULL_PREFILL` | 8.973 ms |
| `OPT_LINEAR_DECODE` traced | 1.537 ms |
| `OPT_FULL_DECODE` traced | 1.213 ms |

The raw profiler CSV was normalized before report generation because legacy rows had blank architecture metadata:

```python
import csv
from pathlib import Path

raw = Path("models/autoports/qwen_qwen3_6_35b_a3b/doc/optimized_decoder/tracy/final_sparsew4_exactnnz_raw/reports/2026_08_19_02_29_34/ops_perf_results_2026_08_19_02_29_34.csv")
normalized = raw.with_name(raw.stem + "_blackhole.csv")
with raw.open(newline="") as inf, normalized.open("w", newline="") as outf:
    reader = csv.DictReader(inf)
    writer = csv.DictWriter(outf, fieldnames=reader.fieldnames)
    writer.writeheader()
    for row in reader:
        row["DEVICE ARCH"] = row.get("DEVICE ARCH") or "blackhole"
        row["AVAILABLE WORKER CORE COUNT"] = row.get("AVAILABLE WORKER CORE COUNT") or "110"
        writer.writerow(row)
```

The normalized CSV is committed as `tracy/final_sparsew4_exactnnz_raw/reports/2026_08_19_02_29_34/ops_perf_results_2026_08_19_02_29_34_blackhole.csv.parts/` with `SHA256SUMS` to satisfy the 500 KB repository hook.

Report command shape:

```bash
python_env/bin/tt-perf-report \
  --start-signpost OPT_LINEAR_PREFILL \
  --end-signpost OPT_LINEAR_PREFILL_END \
  --arch blackhole --active-experts 8 --no-color --raw-op-codes \
  --csv models/autoports/qwen_qwen3_6_35b_a3b/doc/optimized_decoder/tracy/final/linear_attention/prefill_perf_report.csv \
  --summary-file models/autoports/qwen_qwen3_6_35b_a3b/doc/optimized_decoder/tracy/final/linear_attention/prefill_summary.csv \
  models/autoports/qwen_qwen3_6_35b_a3b/doc/optimized_decoder/tracy/final_sparsew4_exactnnz_raw/reports/2026_08_19_02_29_34/ops_perf_results_2026_08_19_02_29_34_blackhole.csv \
  > models/autoports/qwen_qwen3_6_35b_a3b/doc/optimized_decoder/tracy/final/linear_attention/prefill_perf_report.txt
```

The same command shape was run for `OPT_FULL_PREFILL`, `OPT_LINEAR_DECODE`, and `OPT_FULL_DECODE`, and for the current review candidate matrix.

| Window | Rows | Device time | Movement rows | Report |
| --- | ---: | ---: | ---: | --- |
| linear prefill | 492 | 12.123002 ms | 0.171698 ms | `tracy/final/linear_attention/prefill_perf_report.csv` |
| full prefill | 100 | 7.669389 ms | 0.074263 ms | `tracy/final/full_attention/prefill_perf_report.csv` |
| linear traced decode | 93 | 1.445070 ms | 0.038009 ms | `tracy/final/linear_attention/decode_perf_report.csv` |
| full traced decode | 73 | 1.125983 ms | 0.022648 ms | `tracy/final/full_attention/decode_perf_report.csv` |

Final report fallback scan:

| Window | Fallback-like rows | Notes |
| --- | ---: | --- |
| linear prefill | 0 | Small `Tilize`/`Untilize` device layout bridges only |
| full prefill | 0 | Small `Tilize`/`Untilize` device layout bridges only |
| linear decode | 0 | Small `Tilize`/`Untilize` device layout bridges only |
| full decode | 0 | Two `InterleavedToShardedDeviceOperation` rows total `0.001447 ms`, plus small tilize/untilize rows |

The final source audit and reports show no unnecessary host fallback, Torch conversion, or reshard rows in measured prefill/decode.

## Before/After Versus Fused

Fused baseline values come from the completed fused-decoder stage artifacts.

| Case | Fused wall ms | Final wall ms | Wall delta | Fused device ms | Final device ms | Device delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| linear prefill, seq 5 | 33.628 | 20.414 | -39.3% | 25.289 | 12.123 | -52.1% |
| full prefill, seq 33 | 23.038 | 8.973 | -61.1% | 21.684 | 7.669 | -64.6% |
| linear traced decode after seq 5 | 2.463 | 1.537 | -37.6% | 2.368 | 1.445 | -39.0% |
| full traced decode after seq 33 | 2.121 | 1.213 | -42.8% | 2.036 | 1.126 | -44.7% |

The final optimized runtime beats the best correct fused traced-decode baseline and the earlier optimized sparsew2 path.

## Candidate Evidence

### Exact `nnz` And Sparse Geometry

Correctness:

```bash
timeout 900 env TT_METAL_WATCHER=10 TT_METAL_WATCHER_APPEND=1 \
  TT_METAL_WATCHER_NOINLINE=1 TT_METAL_WATCHER_DISABLE_ETH=1 \
  RUN_QWEN36_OPTIMIZED_CANDIDATES=1 RUN_QWEN36_REAL_WEIGHTS=1 \
  ./python_env/bin/python -m pytest \
  --confcutdir=models/autoports/qwen_qwen3_6_35b_a3b/tests \
  --timeout=600 \
  models/autoports/qwen_qwen3_6_35b_a3b/tests/test_optimized_decoder.py \
  -k 'exact_nnz' -s \
  2>&1 | tee models/autoports/qwen_qwen3_6_35b_a3b/doc/optimized_decoder/logs/candidate_exact_nnz_combo_correctness.log
```

Result: `12 passed, 34 deselected`.

```bash
timeout 900 env TT_METAL_WATCHER=10 TT_METAL_WATCHER_APPEND=1 \
  TT_METAL_WATCHER_NOINLINE=1 TT_METAL_WATCHER_DISABLE_ETH=1 \
  RUN_QWEN36_OPTIMIZED_CANDIDATES=1 RUN_QWEN36_REAL_WEIGHTS=1 \
  ./python_env/bin/python -m pytest \
  --confcutdir=models/autoports/qwen_qwen3_6_35b_a3b/tests \
  --timeout=600 \
  models/autoports/qwen_qwen3_6_35b_a3b/tests/test_optimized_decoder.py \
  -k 'routed_bfp4 or decode_l1_sparse or prefill_l1_sparse or sparse_in0_block_w4 or sparse_cores16_out2' -s \
  2>&1 | tee models/autoports/qwen_qwen3_6_35b_a3b/doc/optimized_decoder/logs/candidate_sparse_geometry_correctness.log
```

Result: `10 passed, 36 deselected`.

Perf commands used this shape with `QWEN36_OPTIMIZED_POLICY=<policy>` and output folders under `tracy/candidate_<policy>_raw`:

```bash
timeout 900 env RUN_QWEN36_OPTIMIZED_PERF=1 RUN_QWEN36_PERF_REAL_WEIGHTS=1 \
  QWEN36_OPTIMIZED_POLICY=sparse_in0_block_w4_exact_nnz \
  ./python_env/bin/python -m tracy -r -p -v \
  --no-runtime-analysis --op-support-count=5000 --check-exit-code \
  --output-folder models/autoports/qwen_qwen3_6_35b_a3b/doc/optimized_decoder/tracy/candidate_sparse_in0_block_w4_exact_nnz_raw \
  ./python_env/bin/python -m pytest \
  --confcutdir=models/autoports/qwen_qwen3_6_35b_a3b/tests \
  --timeout=600 \
  models/autoports/qwen_qwen3_6_35b_a3b/tests/test_optimized_decoder.py \
  -k test_perf_qwen36_optimized -s \
  2>&1 | tee models/autoports/qwen_qwen3_6_35b_a3b/doc/optimized_decoder/logs/tracy_perf_candidate_sparse_in0_block_w4_exact_nnz_summary.log
```

| Policy | Linear prefill wall | Full prefill wall | Linear decode wall | Full decode wall | Decision |
| --- | ---: | ---: | ---: | ---: | --- |
| old sparsew2 default | 24.846 | 13.907 | 2.136 | 1.813 | Superseded |
| `decode_exact_nnz` | 24.445 | 13.457 | 1.688 | 1.350 | Kept as part of final |
| `routed_bfp4_exact_nnz` | 24.943 | 13.416 | 1.683 | 1.350 | Rejected, no win |
| `decode_l1_sparse_inputs_exact_nnz` | 24.808 | 13.454 | 1.693 | 1.373 | Rejected, slower |
| `prefill_l1_sparse_inputs_exact_nnz` | 24.781 | 13.370 | 1.675 | 1.349 | Rejected, no prefill win |
| `sparse_in0_block_w4_exact_nnz` | 20.976 | 9.060 | 1.537 | 1.220 | Kept |
| `sparse_cores16_out2_exact_nnz` | 23.942 | 12.933 | 1.648 | 1.332 | Rejected, slower |
| final no-env default | 20.414 | 8.973 | 1.537 | 1.213 | Accepted |

### Precision And Fidelity

Candidate correctness command:

```bash
timeout 1200 env RUN_QWEN36_OPTIMIZED_CANDIDATES=1 RUN_QWEN36_REAL_WEIGHTS=1 \
  ./python_env/bin/python -m pytest \
  --confcutdir=models/autoports/qwen_qwen3_6_35b_a3b/tests \
  --timeout=900 \
  models/autoports/qwen_qwen3_6_35b_a3b/tests/test_optimized_decoder.py \
  -k test_candidate_qwen36_optimized_policy_real_weight_pcc -s \
  2>&1 | tee models/autoports/qwen_qwen3_6_35b_a3b/doc/optimized_decoder/logs/candidate_policy_correctness.log
```

| Candidate | Correctness | Perf evidence | Decision |
| --- | --- | --- | --- |
| Routed MoE BFP4 | Passed linear/full correctness; exact-`nnz` rerun also passed. | `routed_bfp4_exact_nnz` did not beat final sparsew4/exact-`nnz`. | Keep layer-kind auto policy: linear BFP8, full BFP4. |
| All MoE BFP4 | Failed real linear prefill PCC `0.9890764118945616`. | Not retained. | Rejected for real-weight PCC; synthetic/random PCC was not used to veto or accept wins. |
| Default dense/shared BFP8 and BF16 sparse outputs | Final real PCC all above bar. | Final report tables. | Kept. |

### Dense DRAM-Sharded Projection

First attempt hit a TTNN API signature issue, so the candidate was adapted and rerun rather than rejected on first error.

```bash
timeout 900 env RUN_QWEN36_OPTIMIZED_CANDIDATES=1 RUN_QWEN36_REAL_WEIGHTS=1 \
  ./python_env/bin/python -m pytest \
  --confcutdir=models/autoports/qwen_qwen3_6_35b_a3b/tests \
  --timeout=600 \
  models/autoports/qwen_qwen3_6_35b_a3b/tests/test_optimized_decoder.py \
  -k test_candidate_decode_dense_projection_dram_sharded -s \
  2>&1 | tee models/autoports/qwen_qwen3_6_35b_a3b/doc/optimized_decoder/logs/candidate_dense_dram_sharded_adapted.log
```

| Projection | Baseline | Candidate | PCC | Decision |
| --- | ---: | ---: | ---: | --- |
| linear packed `qkv/z/b/a`, padded width 12544 | 0.233 ms | 0.274 ms | 0.9998905590570133 | Rejected, slower after legal sharding/reshard boundary cost. |
| full packed `q/k/v`, padded width 9216 | 0.139 ms | 0.263 ms | 0.9998712665130002 | Rejected, slower after legal sharding/reshard boundary cost. |

The candidate used `MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig` and LoFi compute kernel config, covering the material DRAM-sharded dense decode matmul advice.

### SDPA K64

Correctness artifact: `logs/candidate_sdpa_k64_correctness.log`, full prefill PCC `0.999624418357692`, traced decode PCC `0.9994981617821127`.

Perf artifact: `tracy/candidate_sdpa_k64/full_attention/decode_perf_report.csv`, full decode `1.999055 ms` before sparsew2. K64 was slightly faster than the previous full-decode SDPA config and was kept. Final K64 plus sparsew4/exact-`nnz` full decode is `1.125983 ms` device time.

## Final Optimized Configuration

```python
OptimizedDecoderPolicy(
    attention_weight_dtype=ttnn.bfloat8_b,
    linear_attention_weight_dtype=ttnn.bfloat8_b,
    shared_moe_weight_dtype=ttnn.bfloat8_b,
    routed_moe_weight_dtype=AUTO_ROUTED_MOE_WEIGHT_DTYPE,
    sparse_decode_output_dtype=ttnn.bfloat16,
    sparse_prefill_output_dtype=ttnn.bfloat16,
    sparse_decode_memory_config=ttnn.DRAM_MEMORY_CONFIG,
    sparse_prefill_memory_config=ttnn.DRAM_MEMORY_CONFIG,
    sparse_in0_block_w=4,
    sparse_core_count_cap=None,
    sparse_out_subblock_h=1,
    sparse_out_subblock_w=1,
    use_decode_exact_nnz=True,
    use_decode_l1_sparse_inputs=False,
    use_prefill_l1_sparse_inputs=False,
    use_decode_sdpa_program_config=True,
    decode_sdpa_q_chunk_size=32,
    decode_sdpa_k_chunk_size=64,
    decode_sdpa_max_cores_per_head_batch=16,
)
```

`AUTO_ROUTED_MOE_WEIGHT_DTYPE` resolves to `ttnn.bfloat8_b` for linear-attention layers and `ttnn.bfloat4_b` for full-attention layers.

## Context Contract

No `doc/context_contract.json` change was made. No capacity-affecting memory layout, cache dtype, cache block size, or public shape contract changed. Non-aligned logical sequence lengths remain valid; optimized MoE prefill pads internally and slices back to the requested logical sequence.

## Stage Review

Initial stage-review subagent `01a017b4-c3e0-7573-bd95-f7a0c87219e3` returned `more-work-needed`:

- rerun sparse geometry evidence under the final geometry;
- prove/measure exact decode `nnz`;
- add dynamic fallback evidence;
- clarify watcher ETH-disabled rationale and remaining movement rows.

Fixes in this pass:

- added exact decode `nnz`;
- added `in0_block_w=4` and smaller-core/output-subblock candidates;
- reran real-weight correctness and perf for exact-`nnz` combinations;
- selected final sparsew4/exact-`nnz` policy;
- ran final watcher and dynamic fallback audits;
- regenerated final and candidate `tt-perf-report` tables.

Final stage-review subagent `01a017e0-74af-7dd2-965e-c77fb5f06cef` returned `clean-pass`. Residual risks were limited to read-only review scope, small classified device-side layout bridges, and non-blocking environment log noise.

## Local Commits

- `tt-metal`, branch `vkovacevic/agentic-research/qb2-qwen36-35b-a3b`: `0b076f14a0be2ac5fb0cd24f98766acfbf3eb17f` (`Optimize qwen3.6 decoder`). Local checkpoint only; not pushed.

## Artifacts

- Final code: `models/autoports/qwen_qwen3_6_35b_a3b/tt/optimized_decoder.py`
- Final tests: `models/autoports/qwen_qwen3_6_35b_a3b/tests/test_optimized_decoder.py`
- README: `models/autoports/qwen_qwen3_6_35b_a3b/doc/optimized_decoder/README.md`
- Final correctness/watcher: `logs/watcher_correctness_final_sparsew4_exactnnz.log`
- Dynamic fallback audit: `logs/runtime_fallback_audit_final_sparsew4_exactnnz.log`
- Final perf log: `logs/tracy_perf_final_sparsew4_exactnnz_summary.log`
- Final Blackhole report input: `tracy/final_sparsew4_exactnnz_raw/reports/2026_08_19_02_29_34/ops_perf_results_2026_08_19_02_29_34_blackhole.csv.parts/`
- Final report tables: `tracy/final/linear_attention/*_perf_report.{csv,txt}`, `tracy/final/full_attention/*_perf_report.{csv,txt}`
- Candidate report tables: `tracy/candidate_decode_exact_nnz/`, `tracy/candidate_routed_bfp4_exact_nnz/`, `tracy/candidate_decode_l1_sparse_inputs_exact_nnz/`, `tracy/candidate_prefill_l1_sparse_inputs_exact_nnz/`, `tracy/candidate_sparse_in0_block_w4_exact_nnz/`, `tracy/candidate_sparse_cores16_out2_exact_nnz/`
- Candidate correctness logs: `logs/candidate_policy_correctness.log`, `logs/candidate_exact_nnz_combo_correctness.log`, `logs/candidate_sparse_geometry_correctness.log`, `logs/candidate_dense_dram_sharded_adapted.log`, `logs/candidate_sdpa_k64_correctness.log`

Oversized Tracy internals are report-generation byproducts. The durable artifacts for review are the tee logs, Blackhole-normalized final report input CSV parts, and generated `tt-perf-report` CSV/TXT tables.
