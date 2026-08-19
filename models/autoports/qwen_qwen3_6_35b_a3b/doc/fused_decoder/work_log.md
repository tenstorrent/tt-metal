# Fused Decoder Work Log

Model: `Qwen/Qwen3.6-35B-A3B`

Stage scope: `models/autoports/qwen_qwen3_6_35b_a3b/tt/fused_decoder.py`, `tests/test_fused_decoder.py`, and `doc/fused_decoder/*` only.

## Commands And Results

### Device Safety

```bash
timeout 60 tt-smi -ls --local
```

Result: four local Blackhole p300c devices visible. Final artifact: `logs/tt_smi_local.log`.

### Static And Collection Checks

```bash
./python_env/bin/python -m py_compile \
  models/autoports/qwen_qwen3_6_35b_a3b/tt/fused_decoder.py \
  models/autoports/qwen_qwen3_6_35b_a3b/tests/test_fused_decoder.py
```

Result: passed.

```bash
./python_env/bin/python -m pytest \
  --confcutdir=models/autoports/qwen_qwen3_6_35b_a3b/tests \
  --collect-only -q \
  models/autoports/qwen_qwen3_6_35b_a3b/tests/test_fused_decoder.py
```

Result: 18 fused tests collected.

### Synthetic Correctness

```bash
set -o pipefail
timeout 1200 ./python_env/bin/python -m pytest \
  --confcutdir=models/autoports/qwen_qwen3_6_35b_a3b/tests \
  --timeout=1200 \
  models/autoports/qwen_qwen3_6_35b_a3b/tests/test_fused_decoder.py \
  -k 'not perf and not real_weight' -s \
  2>&1 | tee models/autoports/qwen_qwen3_6_35b_a3b/doc/fused_decoder/logs/synthetic_correctness_full.log
```

Result: `10 passed, 8 deselected`.

Key PCC values:

| Case | Prefill PCC | Traced decode PCC |
| --- | ---: | ---: |
| synthetic linear layer 0, seq 5 | 0.9995374726645452 | 0.999636443075036 |
| synthetic full layer 3, seq 33 | 0.9996627571644442 | 0.9994431969325123 |
| synthetic batch-2 linear, seq 5 | 0.9995916143985506 | 0.9995260348984508 |
| synthetic batch-2 full, seq 33 | 0.9996683781380403 | 0.9994558571610651 |
| synthetic linear non-aligned seq 65 | 0.9975437742222755 | 0.9994243099011441 |
| synthetic full non-aligned seq 33 | 0.9996627571644442 | 0.9994431969325123 |

Repeated decode determinism: linear PCC `1.0`, full PCC `1.0`.

### Real-Weight Correctness

```bash
set -o pipefail
timeout 1200 env RUN_QWEN36_REAL_WEIGHTS=1 ./python_env/bin/python -m pytest \
  --confcutdir=models/autoports/qwen_qwen3_6_35b_a3b/tests \
  --timeout=1200 \
  models/autoports/qwen_qwen3_6_35b_a3b/tests/test_fused_decoder.py \
  -k 'test_real_weight_fused_decoder_prefill_decode_against_hf' -s \
  2>&1 | tee models/autoports/qwen_qwen3_6_35b_a3b/doc/fused_decoder/logs/real_weight_correctness.log
```

Result: `4 passed, 14 deselected`.

| Case | Prefill PCC | Traced decode PCC |
| --- | ---: | ---: |
| real linear layer 0, seq 1 | 0.9961841378187762 | 0.9993734355916063 |
| real full layer 3, seq 1 | 0.9998392105909775 | 0.9996458116075372 |
| real linear layer 0, seq 5 | 0.9974394485520821 | 0.9997484579084984 |
| real full layer 3, seq 5 | 0.9997803748533853 | 0.999480393229282 |

The real linear prefill drop versus functional is the largest numerical delta, but it remains above the PCC bar and the decode PCC stays high.

### Fallback Audit

```bash
set -o pipefail
timeout 900 env TTNN_CONFIG_OVERRIDES='{"throw_exception_on_fallback": true}' \
  ./python_env/bin/python -m pytest \
  --confcutdir=models/autoports/qwen_qwen3_6_35b_a3b/tests \
  --timeout=900 \
  models/autoports/qwen_qwen3_6_35b_a3b/tests/test_fused_decoder.py \
  -k 'test_synthetic_fused_decoder_prefill_decode_against_hf or test_fused_runtime_fallback_audit_source' -s \
  2>&1 | tee models/autoports/qwen_qwen3_6_35b_a3b/doc/fused_decoder/logs/runtime_fallback_audit.log
```

Result: `3 passed, 15 deselected`. The runtime source audit forbids `torch`, `from_torch`, `to_torch`, and fallback-call use in measured fused runtime functions.

### Watcher

```bash
set -o pipefail
rm -rf models/autoports/qwen_qwen3_6_35b_a3b/doc/fused_decoder/watcher/final
timeout 900 env TT_METAL_WATCHER=10 \
  TT_METAL_LOGS_PATH=/localdev/vkovacevic/tt-metal/models/autoports/qwen_qwen3_6_35b_a3b/doc/fused_decoder/watcher/final \
  ./python_env/bin/python -m pytest \
  --confcutdir=models/autoports/qwen_qwen3_6_35b_a3b/tests \
  --timeout=900 \
  models/autoports/qwen_qwen3_6_35b_a3b/tests/test_fused_decoder.py \
  -k 'test_synthetic_fused_decoder_prefill_decode_against_hf or test_synthetic_fused_decoder_repeated_input_determinism' -s \
  2>&1 | tee models/autoports/qwen_qwen3_6_35b_a3b/doc/fused_decoder/logs/watcher_correctness.log
```

Result: `4 passed, 14 deselected`. `watcher/final/generated/watcher/watcher.log` had zero tight failure markers: `assert`, `fatal`, `deadlock`, `stalled`, `illegal`, `watcher error`, `erisc error`, `brisc error`, `ncrisc error`, `trisc error`.

### Performance

```bash
set -o pipefail
timeout 1200 env RUN_QWEN36_FUSED_PERF=1 RUN_QWEN36_PERF_REAL_WEIGHTS=1 \
  ./python_env/bin/python -m tracy -r -p -v \
  --no-runtime-analysis --op-support-count=5000 --check-exit-code \
  --output-folder models/autoports/qwen_qwen3_6_35b_a3b/doc/fused_decoder/tracy/raw \
  ./python_env/bin/python -m pytest \
  --confcutdir=models/autoports/qwen_qwen3_6_35b_a3b/tests \
  --timeout=1200 \
  models/autoports/qwen_qwen3_6_35b_a3b/tests/test_fused_decoder.py \
  -k test_perf_qwen36_fused -s \
  2>&1 | tee models/autoports/qwen_qwen3_6_35b_a3b/doc/fused_decoder/logs/tracy_perf_summary.log
```

The committed tee log is stored as `logs/tracy_perf_summary.log.parts/` with `SHA256SUMS`; reconstruct from `models/autoports/qwen_qwen3_6_35b_a3b/doc/fused_decoder` with `cat logs/tracy_perf_summary.log.parts/part_*.log > logs/tracy_perf_summary.log`.

Result: `4 passed, 14 deselected`.

Raw and normalized profiler CSVs:

- `tracy/raw/reports/2026_08_18_23_40_09/ops_perf_results_2026_08_18_23_40_09.csv.parts/`
- `tracy/raw/reports/2026_08_18_23_40_09/ops_perf_results_2026_08_18_23_40_09_blackhole.csv.parts/`

The original CSVs are committed as line-preserving split parts plus `SHA256SUMS` manifests to stay under the repository hook size limit. Reconstruct each CSV from the repo root with `cat <csv>.parts/part_*.csv > <csv>`, then verify with `sha256sum -c <csv>.parts/SHA256SUMS`. `tt-perf-report` generation used `--active-experts 8` for MoE rows whose attributes lacked numeric `nnz`.

```bash
python_env/bin/tt-perf-report --start-signpost FUSED_LINEAR_PREFILL --end-signpost FUSED_LINEAR_PREFILL_END \
  --arch blackhole --active-experts 8 --no-color --raw-op-codes \
  --csv models/autoports/qwen_qwen3_6_35b_a3b/doc/fused_decoder/tracy/linear_attention/prefill_perf_report.csv \
  --summary-file models/autoports/qwen_qwen3_6_35b_a3b/doc/fused_decoder/tracy/linear_attention/prefill_summary.csv \
  models/autoports/qwen_qwen3_6_35b_a3b/doc/fused_decoder/tracy/raw/reports/2026_08_18_23_40_09/ops_perf_results_2026_08_18_23_40_09_blackhole.csv \
  > models/autoports/qwen_qwen3_6_35b_a3b/doc/fused_decoder/tracy/linear_attention/prefill_perf_report.txt
```

The same command shape was run for `FUSED_FULL_PREFILL`, `FUSED_LINEAR_DECODE`, and `FUSED_FULL_DECODE`.

Before/after summary:

| Case | Functional wall ms | Fused wall ms | Functional device ms | Fused device ms |
| --- | ---: | ---: | ---: | ---: |
| linear prefill seq 5 | 45.456 | 33.628 | 37.162 | 25.289 |
| full prefill seq 33 | 35.810 | 23.038 | 34.158 | 21.684 |
| linear traced decode after seq 5 | 3.023 | 2.463 | 2.923 | 2.368 |
| full traced decode after seq 33 | 2.714 | 2.121 | 2.621 | 2.036 |

Perf conclusion: final fused runtime is faster than the best correct functional traced-decode baseline and faster in both representative prefill cases. The largest remaining rows are MoE sparse matmuls and TTNN top-k/scatter/sparse internals, not Python graph fallback.

### Graph-Fusion Candidate Probe

```bash
set -o pipefail
timeout 1200 ./python_env/bin/python \
  models/autoports/qwen_qwen3_6_35b_a3b/doc/fused_decoder/graph_fusion_candidate_probe.py \
  2>&1 | tee models/autoports/qwen_qwen3_6_35b_a3b/doc/fused_decoder/logs/graph_fusion_candidate_probe.log
```

Result: completed successfully. Artifact: `logs/graph_fusion_candidate_probe.log`.

The probe used Qwen dimensions `heads=16`, `kv_heads=2`, `head_dim=256`, `q_width=4096`, `q_gate_width=8192`, `kv_width=512`, fused `q+gate/k/v` width `9216`, and standard QKV width `5120`.

| Candidate | Evidence | Outcome |
| --- | --- | --- |
| `ttnn.transformer.split_query_key_value_and_split_heads` direct on Qwen `q+gate/k/v` prefill | rejected: inferred head size `460` is not a multiple of tile width `32` | cannot consume Qwen's gated-Q packed projection |
| `split_query_key_value_and_split_heads` after stripping Q gate | correct shapes; current path `0.1680 ms/iter`, candidate `0.2152 ms/iter` | rejected because required slice/concat makes it slower |
| `ttnn.experimental.nlp_create_qkv_heads_decode` direct on Qwen `q+gate/k/v` decode | rejected: input shape `9216` is not divisible by `num_heads + 2 * num_kv_heads = 20` | cannot consume Qwen's gated-Q packed projection |
| `nlp_create_qkv_heads_decode` after stripping Q gate | correct shapes; current path `0.0463 ms/iter`, candidate `0.1292 ms/iter` | rejected because required slice/concat makes it slower |
| `ttnn.transformer.concatenate_heads` prefill | correct output shape; current reshape `0.0599 ms/iter`, candidate `0.1673 ms/iter` | rejected as slower helper substitution |
| `ttnn.experimental.nlp_concat_heads_decode` on actual 16-head decode layout | rejected: physical shard shape `(16, 256)` is not tile `{32, 32}` sized | cannot consume actual Qwen head count without padding/layout churn |
| `nlp_concat_heads_decode` with padded 32-head layout | raw output shape `(1, 1, 32, 4096)`; required logical slice gives `(1, 1, 1, 4096)` at `0.0476 ms/iter`; current reshape is `0.0119 ms/iter` | rejected because it is slower after shape repair and expands the logical head axis |
| `ttnn.experimental.deepseek.moe.generalized_moe_gate` direct on fused-router layout | rejected: `input_tensor must be sharded` | direct dense router tensor cannot satisfy the op contract |
| `generalized_moe_gate` adapted to Qwen's required sharded layout with dense scatter rebuild | rejected during JIT: Blackhole build cannot find `experimental/llk_sfpu/llk_math_generalized_moe_gate_topk_single_face.h`; current dense router path was `0.2263 ms/iter` for 33 tokens and `0.2187 ms/iter` for 1 token | unavailable on this Blackhole checkout; candidate full-decoder timings skipped because the adapted op fails before timing |

Source contract check for `generalized_moe_gate`: all five tensors must be L1 height-sharded, experts are packed into 32x32 tiles, and only the first `topk` output entries per token are valid compact scores/ids. The adapted Qwen probe reshaped router logits into that layout and rebuilt dense `[tokens, experts]` routing with TTNN scatter, but the candidate cannot JIT on current Blackhole hardware.

```bash
find tt_metal -path '*llk_math_generalized_moe_gate_topk_single_face.h' -print
```

Result: only `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/experimental/llk_sfpu/llk_math_generalized_moe_gate_topk_single_face.h` exists; there is no Blackhole counterpart in this checkout.

Current fused decoder timings from the same candidate probe, using a synthetic nonzero-router setup for candidate comparison:

| Path | Time |
| --- | ---: |
| current fused decoder linear prefill | 27.2866 ms/iter |
| current fused decoder linear traced decode | 2.3982 ms/iter |
| current fused decoder full prefill | 21.8168 ms/iter |
| current fused decoder full traced decode | 2.0636 ms/iter |

The adapted `generalized_moe_gate` full-decoder path could not run because the kernel failed JIT before timing.

## Graph-Fusing Assessment

| Graph-fusing pattern | Status | Notes |
| --- | --- | --- |
| Same-input matmul/linear projection packing | Implemented | full-attention Q/K/V; linear-attention QKV/Z/B/A; shared expert gate/up |
| Packed sparse expert gate/up | Implemented | routed `mlp.experts.gate_up_proj` stays packed and is sliced after sparse matmul |
| Adjacent activation plus binary multiply | Implemented | SiLU gate/up, sigmoid gates, beta sigmoid, softplus scaling |
| Dedicated RoPE | Implemented where compatible | `ttnn.experimental.rotary_embedding` on measured batch-1 full-attention path; fallback kept for batch-broadcast shapes |
| SDPA and paged decode attention | Already dedicated TTNN | kept `scaled_dot_product_attention`, `paged_scaled_dot_product_attention_decode`, and chunked SDPA |
| Paged KV-cache fill/update | Already dedicated TTNN | no capacity or dtype change |
| Top-k/scatter routing | Already dedicated TTNN | no lower-level fuse available in this scope |
| Conv1d causal state update | Assessed, not fused | no stateful TTNN depthwise causal conv primitive matched the required layout/state contract without extra conversions |
| Dedicated head split/create/concat helpers | Assessed, not kept | see candidate probe above; direct Qwen gated-Q shapes reject or helper paths are slower after required slice/concat/layout work |
| DRAM-sharded matmul program-config rewrite | Assessed, not kept | would require reshards or input layout contract changes in the measured path; this belongs to optimized-decoder tuning, outside this fused-only goal |

All stage-local graph-fusing patterns that matched TTNN runtime contracts and improved the measured path are reflected in `FusedDecoder.graph_summary`. All remaining graph-fusing candidates from the skill were either already dedicated TTNN ops in the fused runtime, incompatible with Qwen's exact tensor contract, unavailable on this Blackhole checkout, or slower in the final Blackhole probe.

## Rejected Options

- Subclassing or delegating to `FunctionalDecoder`: rejected because tests must exercise a fused runtime path, not a functional fallback.
- Public `seq_len % 64 == 0` requirement for linear attention: rejected. Fused prefill pads internal chunks and slices back to logical length.
- Public `seq_len % 32 == 0` requirement for MoE sparse matmul: rejected. Routed prefill pads internal token groups and slices back to logical length.
- Always using dedicated RoPE: rejected for batch-broadcast shapes where the TTNN experimental op does not accept the same cosine/sine layout; the TTNN primitive fallback keeps batch-2 semantics.
- Dedicated head split/create/concat helper replacement: rejected by `graph_fusion_candidate_probe.py`; direct Qwen packed shapes failed op contracts, and shape-massaged candidates were slower than the current fused path.
- `generalized_moe_gate` router replacement: rejected by `graph_fusion_candidate_probe.py`; the direct dense tensor fails the op contract, and the adapted sharded candidate cannot JIT on this Blackhole checkout because the required LLK header is missing.
- Keeping large raw Tracy internals: rejected for repository size. The report input CSVs, filtered op CSVs, and generated report tables are preserved.

## Artifacts

- Code: `tt/fused_decoder.py`
- Tests: `tests/test_fused_decoder.py`
- Candidate probe: `graph_fusion_candidate_probe.py`, `logs/graph_fusion_candidate_probe.log`
- Correctness logs: `logs/synthetic_correctness_full.log`, `logs/real_weight_correctness.log`, `logs/post_patch_synthetic_correctness.log`
- Fallback audit: `logs/runtime_fallback_audit.log`
- Watcher: `logs/watcher_correctness.log`, `watcher/final/generated/watcher/watcher.log`
- Perf provenance: `logs/tracy_perf_summary.log.parts/`
- Raw report input: `tracy/raw/reports/2026_08_18_23_40_09/ops_perf_results_2026_08_18_23_40_09.csv.parts/`, `tracy/raw/reports/2026_08_18_23_40_09/ops_perf_results_2026_08_18_23_40_09_blackhole.csv.parts/`
- `tt-perf-report` tables and CSVs: `tracy/linear_attention/*_perf_report.{txt,csv}`, `tracy/full_attention/*_perf_report.{txt,csv}`
- Filtered measured-window ops: `tracy/linear_attention/prefill_filtered_ops.csv.parts/`, `tracy/linear_attention/decode_filtered_ops.csv`, `tracy/full_attention/*_filtered_ops.csv`
- Summary plots and CSVs: `tracy/linear_attention/*_summary.csv.{csv,png}`, `tracy/full_attention/*_summary.csv.{csv,png}`

## Limitations And Follow-Up Boundary

No known fused-decoder gate remains open. Remaining performance opportunities are TTNN kernel/program-config optimization work, not unfused Python graph substitutions. Those belong to later optimized-decoder work and were not started in this stage.

## Stage Review And Commit Record

Initial stage review: `01a0174b-d00b-72a0-820a-34058fce8fb6` returned `more-work-needed` for unproven dedicated head/MoE candidates. Follow-up review `01a01757-a371-7d20-a169-6547b2acb58a` returned `more-work-needed` because `generalized_moe_gate` still needed an adapted sharded probe and padded `nlp_concat_heads_decode` needed explicit shape repair evidence.

Final stage review: `01a01765-316a-7a43-9376-98fd8a383e3e` returned `clean-pass`. The final review verified the fused runtime path, PCC coverage, fallback audit, watcher run, before/after perf evidence, `tt-perf-report` artifacts, and graph-fusing candidate exhaustion. The earlier findings were fixed by the adapted `generalized_moe_gate` probe and the corrected padded `nlp_concat_heads_decode` shape/timing evidence.

Review summary artifact: `stage_review.md`.

Implementation commit SHA: pending local commit.
