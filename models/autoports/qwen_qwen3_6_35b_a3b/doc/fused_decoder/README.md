# Fused Decoder

This directory records the fused-decoder state for `Qwen/Qwen3.6-35B-A3B` under `models/autoports/qwen_qwen3_6_35b_a3b`. The scope is single-device TTNN decoder-layer fusion only. No optimized-decoder, multichip-decoder, full-model, or vLLM work is included.

## Implementation

`tt/fused_decoder.py` implements `FusedDecoder(LightweightModule)` with the same public contract as `FunctionalDecoder`:

| Method | Input shape | Required state | Return |
| --- | --- | --- | --- |
| `prefill_forward` | `[1, batch, seq, 2048]` | full attention: `position_embeddings`, `page_table`, optional `kv_cache`; linear attention: `linear_state` | `FunctionalDecoderResult` |
| `decode_forward` | `[1, 1, batch, 2048]` | `current_pos`; full attention also needs `position_embeddings`, `page_table`, `kv_cache`; linear attention needs `linear_state` | `FunctionalDecoderResult` |

The fused runtime path is not a wrapper around the functional decoder. It reuses low-level shared helpers and result/cache dataclasses, but measured prefill/decode enter `FusedDecoder` modules directly.

Implemented graph fusions:

| Pattern | Fused action |
| --- | --- |
| Full-attention same-input projections | packed `q_proj`, `k_proj`, `v_proj` into one `qkgv_proj`, then sliced on device |
| Full-attention gate activation | folded output gate sigmoid into `ttnn.mul(..., input_tensor_b_activations=[SIGMOID])` |
| Full-attention RoPE | uses `ttnn.experimental.rotary_embedding` for batch-1 measured path; keeps a TTNN primitive fallback for batch-broadcast shapes |
| Linear-attention same-input projections | packed `in_proj_qkv`, `in_proj_z`, `in_proj_b`, `in_proj_a` into one projection |
| Linear-attention activation chains | folded `softplus(alpha + dt_bias) * neg_exp_a_log`, beta sigmoid multiplies, and `silu(z)` output gate |
| Shared expert MLP | packed shared `gate_proj` and `up_proj`, fused `silu(gate) * up`, folded shared-expert gate sigmoid |
| Routed expert MLP | uses packed `mlp.experts.gate_up_proj` sparse matmul output, sliced on device, fused `silu(gate) * up` |

The fused path preserves batch-1 and batch-2 semantics, full-attention paged KV-cache behavior, linear-attention recurrent state behavior, traced decode, non-aligned logical lengths, and deterministic repeated decode.

## Context Contract

No update was made to `doc/context_contract.json`: the fused decoder does not reduce the advertised capacity, public shapes, cache block size, cache dtype, KV-cache layout contract, or linear-attention logical-length support. Internal packing and padding are local to fused runtime kernels and preserve valid non-aligned logical sequence lengths.

## Correctness

Acceptance bar: PCC >= 0.995. Main artifact: `logs/synthetic_correctness_full.log` (`10 passed, 8 deselected`). Real-weight artifact: `logs/real_weight_correctness.log` (`4 passed, 14 deselected`).

Command:

```bash
set -o pipefail
timeout 1200 ./python_env/bin/python -m pytest \
  --confcutdir=models/autoports/qwen_qwen3_6_35b_a3b/tests \
  --timeout=1200 \
  models/autoports/qwen_qwen3_6_35b_a3b/tests/test_fused_decoder.py \
  -k 'not perf and not real_weight' -s \
  2>&1 | tee models/autoports/qwen_qwen3_6_35b_a3b/doc/fused_decoder/logs/synthetic_correctness_full.log
```

Real-weight command:

```bash
set -o pipefail
timeout 1200 env RUN_QWEN36_REAL_WEIGHTS=1 ./python_env/bin/python -m pytest \
  --confcutdir=models/autoports/qwen_qwen3_6_35b_a3b/tests \
  --timeout=1200 \
  models/autoports/qwen_qwen3_6_35b_a3b/tests/test_fused_decoder.py \
  -k 'test_real_weight_fused_decoder_prefill_decode_against_hf' -s \
  2>&1 | tee models/autoports/qwen_qwen3_6_35b_a3b/doc/fused_decoder/logs/real_weight_correctness.log
```

| Case | Functional prefill PCC | Fused prefill PCC | Functional traced decode PCC | Fused traced decode PCC |
| --- | ---: | ---: | ---: | ---: |
| synthetic linear layer 0, seq 5 | 0.9994461003286241 | 0.9995374726645452 | 0.9994787951042461 | 0.999636443075036 |
| synthetic full layer 3, seq 33 | 0.9996404835641908 | 0.9996627571644442 | 0.9994230880969464 | 0.9994431969325123 |
| synthetic batch-2 linear, seq 5 | 0.9995132077531952 | 0.9995916143985506 | 0.9995844476837497 | 0.9995260348984508 |
| synthetic batch-2 full, seq 33 | 0.9996329431454074 | 0.9996683781380403 | 0.9994690230596567 | 0.9994558571610651 |
| real linear layer 0, seq 1 | 0.9996229995741831 | 0.9961841378187762 | 0.9988370795673545 | 0.9993734355916063 |
| real full layer 3, seq 1 | 0.9998681212753325 | 0.9998392105909775 | 0.9995886582000745 | 0.9996458116075372 |
| real linear layer 0, seq 5 | 0.9993761564364843 | 0.9974394485520821 | 0.9997758895133866 | 0.9997484579084984 |
| real full layer 3, seq 5 | 0.9997494253841673 | 0.9997803748533853 | 0.9996446766105499 | 0.999480393229282 |

Additional fused coverage:

| Case | Result |
| --- | ---: |
| synthetic linear non-aligned seq 65 prefill PCC | 0.9975437742222755 |
| synthetic linear non-aligned seq 65 traced decode PCC | 0.9994243099011441 |
| synthetic full non-aligned seq 33 prefill PCC | 0.9996627571644442 |
| synthetic full non-aligned seq 33 traced decode PCC | 0.9994431969325123 |
| synthetic linear repeated decode determinism PCC | 1.0 |
| synthetic full repeated decode determinism PCC | 1.0 |

The real linear prefill PCC delta is the largest material numerical change. It remains above the acceptance bar and is attributed to changed TTNN accumulation/rounding order from packed projection and folded activation paths; decode PCC improves for real seq 1 and remains effectively unchanged for real seq 5.

## Performance

Performance was captured with warmed prefill and warmed traced decode. The fused results beat the functional traced baseline in all measured cases. `tt-perf-report` CSV `Device Time` is in milliseconds after summing the report CSV microsecond rows.

Command:

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

Raw ops CSVs were captured as `tracy/raw/reports/2026_08_18_23_40_09/ops_perf_results_2026_08_18_23_40_09.csv` and the `tt-perf-report` input `tracy/raw/reports/2026_08_18_23_40_09/ops_perf_results_2026_08_18_23_40_09_blackhole.csv`, a metadata-normalized copy with Blackhole architecture and 110 worker cores. To keep every committed artifact below the repo hook size limit, those two CSVs and the largest filtered window CSV are committed as `.parts/` directories with `SHA256SUMS`. Reconstruct from the repo root with `cat <csv>.parts/part_*.csv > <csv>`, then run `sha256sum -c <csv>.parts/SHA256SUMS`. Oversized profiler internals were removed; the committed perf artifacts are the split report input CSVs, filtered measured-window CSVs, and `tt-perf-report` tables/summaries.

| Case | Functional wall ms | Fused wall ms | Wall delta | Functional device ms | Fused device ms | Device delta | Table | Filtered ops CSV |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| linear prefill, seq 5 | 45.456 | 33.628 | -26.0% | 37.162 | 25.289 | -32.0% | `tracy/linear_attention/prefill_perf_report.txt` | `tracy/linear_attention/prefill_filtered_ops.csv.parts/` |
| full prefill, seq 33 | 35.810 | 23.038 | -35.7% | 34.158 | 21.684 | -36.5% | `tracy/full_attention/prefill_perf_report.txt` | `tracy/full_attention/prefill_filtered_ops.csv` |
| linear traced decode after seq 5 | 3.023 | 2.463 | -18.5% | 2.923 | 2.368 | -19.0% | `tracy/linear_attention/decode_perf_report.txt` | `tracy/linear_attention/decode_filtered_ops.csv` |
| full traced decode after seq 33 | 2.714 | 2.121 | -21.8% | 2.621 | 2.036 | -22.3% | `tracy/full_attention/decode_perf_report.txt` | `tracy/full_attention/decode_filtered_ops.csv` |

`tt-perf-report` conclusions:

| Window | Rows | Device ms | Overall DRAM roofline | Dominant rows |
| --- | ---: | ---: | --- | --- |
| linear prefill | 492 | 25.289 | 14.2%, 73 GB/s | all-expert `SparseMatmulDeviceOperation` gate/up and down projections |
| full prefill | 100 | 21.684 | 16.5%, 84 GB/s | all-expert sparse MoE rows, then active-expert sparse MoE rows |
| linear decode | 93 | 2.368 | 11.1%, 57 GB/s | active-expert sparse MoE rows, then packed input projection |
| full decode | 73 | 2.036 | 11.2%, 57 GB/s | active-expert sparse MoE rows, then packed attention projection |

The report still shows small `Tilize*`/`Untilize*` rows from TTNN internal layout conversions around sparse/top-k/scatter/rotary-compatible kernels. There are no `from_torch`, `to_torch`, `Reshard`, or runtime host fallback rows in the measured reports, and the fused runtime source audit forbids Torch conversion calls in the measured path.

## Fallback And Watcher

Fallback audit command:

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

Result: `3 passed, 15 deselected`. The source audit also passed in `logs/synthetic_correctness_full.log`.

Watcher command:

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

Result: `4 passed, 14 deselected`. Tight watcher failure-marker scan on `watcher/final/generated/watcher/watcher.log` found zero `assert`, `fatal`, `deadlock`, `stalled`, `illegal`, or processor error markers.

Hardware provenance: `logs/tt_smi_local.log` shows four local Blackhole p300c devices. The repeated warning about unknown motherboard `B850M-C` is platform metadata only and did not block device open, tests, Tracy profiling, or watcher.

## Candidate Exhaustion

The first stage review requested explicit proof for the remaining TTNN dedicated head and MoE-gate candidates. The final probe is reproducible with:

```bash
set -o pipefail
timeout 1200 ./python_env/bin/python \
  models/autoports/qwen_qwen3_6_35b_a3b/doc/fused_decoder/graph_fusion_candidate_probe.py \
  2>&1 | tee models/autoports/qwen_qwen3_6_35b_a3b/doc/fused_decoder/logs/graph_fusion_candidate_probe.log
```

Result: completed successfully. The probe used Qwen attention dimensions `heads=16`, `kv_heads=2`, `head_dim=256`, `q_width=4096`, `q_gate_width=8192`, `kv_width=512`, fused Qwen `q+gate/k/v` width `9216`, and standard QKV width `5120`.

| Candidate | Probe result | Decision |
| --- | --- | --- |
| `ttnn.transformer.split_query_key_value_and_split_heads` direct on Qwen `q+gate/k/v` prefill | rejected: inferred head size `460` is not tile-width aligned | not applicable to Qwen's gated Q projection |
| `split_query_key_value_and_split_heads` after stripping Q gate | correct shapes, but `0.2152 ms/iter` versus current `0.1680 ms/iter` | rejected; requires extra slice/concat and is slower |
| `ttnn.experimental.nlp_create_qkv_heads_decode` direct on Qwen `q+gate/k/v` decode | rejected: input shape `9216` is not divisible by `num_heads + 2 * num_kv_heads = 20` | not applicable to Qwen's gated Q projection |
| `nlp_create_qkv_heads_decode` after stripping Q gate | correct shapes, but `0.1292 ms/iter` versus current `0.0463 ms/iter` | rejected; requires extra slice/concat and is slower |
| `ttnn.transformer.concatenate_heads` prefill | correct output shape, but `0.1673 ms/iter` versus current reshape `0.0599 ms/iter` | rejected as a slower helper substitution |
| `ttnn.experimental.nlp_concat_heads_decode` on actual 16-head layout | rejected: physical shard shape `(16, 256)` is not tile sized | not applicable without padding/layout churn |
| `nlp_concat_heads_decode` with padded 32-head layout | raw padded output shape `(1, 1, 32, 4096)`; adding the required logical slice gives `(1, 1, 1, 4096)` at `0.0476 ms/iter` versus current reshape `0.0119 ms/iter` | rejected; slower after shape repair and expands the logical head axis |
| `ttnn.experimental.deepseek.moe.generalized_moe_gate` direct on fused-router layout | rejected: `input_tensor must be sharded` | direct dense router tensor is incompatible |
| `generalized_moe_gate` adapted to Qwen's required sharded layout with dense scatter rebuild | rejected during JIT: Blackhole build cannot find `experimental/llk_sfpu/llk_math_generalized_moe_gate_topk_single_face.h`; current dense router path is `0.2263 ms/iter` for 33 tokens and `0.2187 ms/iter` for 1 token | not available on this Blackhole checkout; candidate full-decoder timings were skipped because the adapted op fails before timing |

The `generalized_moe_gate` source contract was also checked: all five tensors must be L1 height-sharded, the input packs experts into 32x32 tiles, and only the first `topk` output entries per token are valid compact scores/ids. The adapted Qwen probe reshapes router logits into that layout and scatters compact top-k scores/ids back to dense `[tokens, experts]` routing, but the kernel cannot JIT on this Blackhole checkout. The required LLK header exists only under the Wormhole path in this tree, so the candidate is unavailable for the current hardware.

For reference, the probe also timed the current fused decoder path with the same synthetic nonzero-router setup used for the adapted candidate: linear prefill `27.2866 ms/iter`, linear traced decode `2.3982 ms/iter`, full prefill `21.8168 ms/iter`, and full traced decode `2.0636 ms/iter`. The adapted `generalized_moe_gate` full-decoder path could not run because its kernel failed JIT first.

All graph-fusing patterns from the skill were therefore either implemented in `FusedDecoder.graph_summary`, already covered by dedicated TTNN runtime ops, or rejected with op-contract, Blackhole availability, and timing evidence. No remaining stage-local graph substitution improved the current hardware path.

## Limitations

- This is a fused decoder layer, not an optimized decoder. Program-config tuning, sharded matmul, multichip distribution, full-model assembly, and serving are deliberately left for later stages.
- `ttnn.experimental.rotary_embedding` is used for the batch-1 measured full-attention path. Batch-broadcast RoPE shapes keep the TTNN primitive fallback to preserve batch-2 semantics without host fallback.
- Top-k, scatter, sparse matmul, paged cache fill/update, and SDPA are already dedicated TTNN operations. Remaining bottlenecks are TTNN op internals rather than stage-local Python graph patterns.
- Long advertised-context capability is inherited from the functional decoder contract. This fused stage did not rerun the 262143/262144-token context probes because no capacity-affecting layout, sharding, or cache dtype change was introduced.
