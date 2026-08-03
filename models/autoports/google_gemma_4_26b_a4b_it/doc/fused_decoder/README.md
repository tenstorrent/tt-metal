# Gemma-4 26B A4B fused decoder

Stage 02 delivers a single-device `FusedDecoder` for
`google/gemma-4-26B-A4B-it`. It owns the measured prefill and decode bodies;
neither body calls a functional-forward fallback. The functional decoder is
retained only as the setup/semantic base and as the unfused A/B control.

## Result

The retained graph has two measured changes:

1. The 128 expert gate and up projections are packed once in DRAM and executed
   by one `sparse_matmul` with a 1,408-wide output. Two views feed a multiply
   whose input activation is GELU. The down projection remains one
   `sparse_matmul`. Source gate/up buffers are released after packing, so this
   does not consume capacity that belongs to KV cache.
2. Natural-layout decode caches use `paged_fused_update_cache` for concurrent
   K/V writes. V is placed on a disjoint height-sharded core grid, as required
   by the fused op. Shared physical cache views and modulo-addressed sliding
   caches retain two `paged_update_cache` calls because the fused op has no
   `num_kv_heads`, `block_size`, or `cache_position_modulo` view arguments.
Sequence length 1,024 warmed host medians (five samples) are the latency
authority. Prefill is single-user by contract; decode is reported at batch 1
and serving batch 32.

| layer kind | batch | functional prefill ms | fused prefill ms | prefill speedup | functional traced decode ms | fused traced decode ms | decode speedup |
|---|---:|---:|---:|---:|---:|---:|---:|
| sliding | 1 | 680.937 | 401.403 | 1.696x | 3.013 | 2.458 | 1.226x |
| sliding | 32 | — | — | — | 68.858 | 51.308 | 1.342x |
| full | 1 | 681.934 | 402.500 | 1.694x | 3.198 | 2.648 | 1.208x |
| full | 32 | — | — | — | 68.690 | 51.141 | 1.343x |

The four `candidate_ab_*.json` files contain every sample and bind the result
to exact fused and functional implementation/test hashes. The final graph also
beat the strongest correct rejected candidate, adapted dense gate/up
`minimal_matmul_split`, in 101-repeat alternating same-process trace replay for
every layer/batch case. Batch-1 prefill was also measured with 21 alternating
repetitions:

| layer kind | batch | final decode ms | dense decode ms | final prefill ms | dense prefill ms | candidate PCC |
|---|---:|---:|---:|---:|---:|---:|
| sliding | 1 | 2.4613 | 2.4707 | 401.488 | 401.519 | 0.999961 |
| sliding | 32 | 51.2994 | 51.3373 | — | — | 0.999927 |
| full | 1 | 2.6622 | 2.6677 | 402.645 | 402.654 | 0.999970 |
| full | 32 | 51.1315 | 51.2067 | — | — | 0.999962 |

The `final_vs_dense_split_layer*.json` artifacts contain all samples,
ordering, PCC, and source provenance. Thus the delivered graph beats both the
functional traced baseline and the best numerically correct candidate.

The composite GeGLU lowering was also rerun for 101 traced replays across the
complete layer/batch matrix. Explicit lowering won three of four raw medians
and the aggregate (107.550 versus 107.670 ms); the 7.2-us serving-case
inversion has a paired 95% interval crossing zero, and no interval
significantly favors the candidate. Current-source prefill comparisons have
PCC 1.0 and favor explicit lowering by 0.949 ms sliding and 0.980 ms full.
Selection is based on measured latency with a predeclared significance rule,
not op count. `final_vs_composite_geglu_matrix.json` and its four input
artifacts retain every sample and exact source provenance.

## Correctness and contracts

The unchanged Stage 01 HF acceptance oracle is run with `FusedDecoder`
substituted at construction time. This prevents a functional fallback while
keeping the reference, masks, caches, shapes, and thresholds identical.

| layer/cache case | prefill PCC | decode PCC | threshold |
|---|---:|---:|---:|
| sliding/shared | 0.998617 | 0.999678 | 0.995 |
| full/natural | 0.997773 | 0.999868 | 0.995 |
| full/shared view | 0.997773 | 0.999868 | 0.995 |

Fused-versus-functional PCC is 1.0 for prefill and 0.999948 sliding /
0.999968 full for decode. Traced batch 1/32 outputs meet the HF bar, eager to
trace PCC is 1.0, and two replay outputs have PCC 1.0 for both layer kinds.

Non-aligned logical lengths 31, 33, and 1,025 pass for both kinds. The lowest
PCC is 0.996493 at sliding length 1,025. The bounded 1,025-token modulo test
also proves that wrapping slot zero preserves slots 1..1,023.

The advertised 262,144-token context remains unchanged. Batch-1 traced decode
at position 262,143 passes for sliding and full caches, preserves cache
sentinels, produces finite output, and repeats with PCC 1.0. The packing step
releases its source buffers, while cache layout and dtype are unchanged.
Real-weight fused prefill also passes with finite last-token readback at both
262,143 and 262,144 for sliding and full attention. The corresponding artifact
times are 103.3 s sliding and 190.7 s full, versus functional evidence of about
175 s and 262 s. Therefore `doc/context_contract.json` was deliberately not
modified.

## Operation-topology audit

The measured graph is entirely device-resident after test input setup:

| segment | input / movement | operation sequence | output |
|---|---|---|---|
| attention input | DRAM hidden | RMSNorm → QKV linear → dedicated head split | Q/K/V heads |
| RoPE/cache | L1 sharded heads | dedicated RoPE → fused K/V paged update for natural caches | resident KV cache |
| attention | sharded Q + paged cache | paged SDPA → dedicated head concat → output linear | DRAM attention branch |
| dense branch | normalized hidden | gate linear + up linear → GELU → multiply → down linear | dense branch |
| router | residual | RMSNorm → scaled linear → softmax → top-k | sparse routing weights |
| experts | normalized hidden + routing | packed sparse up/gate → slices → activation-folded multiply → sparse down → score multiply → reduce | MoE branch |
| output | two branches + residual | add → RMSNorm → residual add → layer scalar | decoder output |

The retained `tt-perf-report` tables show 57 device ops for a 32-token prefill
tile, 71 sliding decode ops, and 73 full decode ops. The packed expert up/gate
matmul is the dominant operation (about 66% of prefill device time and 45–48%
of batch-1 decode device time); the sparse down matmul is next. The reports
also prove the fused cache operation is present. There are no host operations
or reshard operations inside the signposted ranges.

The visible tilize/untilize and slice operations surround the canonical router
top-k/sparsity representation and `sparse_matmul` contracts. Removing or
merging them was not expressible by the available sparse/gate ops without
changing routing or layout. Attention sharding conversions likewise satisfy
the dedicated RoPE, cache-update, SDPA, and head-concat contracts. There is no
`torch`, `from_torch`, `to_torch`, or host fallback in the owned hot methods.
The remaining decode RMSNorm layout conversions were tested directly: passing
the height-sharded input through removed conversions fails in the device
layernorm implementation with `Height sharded inputs are not supported`.

`tt-perf-report` uses a clean 32-token topology capture because a combined
1,024-token device-op capture filled the fixed profiler marker buffer. The
repository's `--force-legacy-device-logs` recovery produced complete device-op
correlation; the incomplete raw capture was discarded. Warmed sequence-1,024
host A/B remains the performance decision source. Device trace replay is
captured separately, as required by the device-usage workflow.

## Fusion pattern disposition

All patterns in the graph-fusing skill were assessed against this decoder:

| class / pattern | disposition |
|---|---|
| activation, softmax, RMSNorm, SDPA | already dedicated in Stage 01 |
| QKV/head split, prefill/decode head concat, RoPE, top-k | already dedicated in Stage 01 |
| residual add + RMSNorm | correct but slower; rejected |
| MoE expert dedicated path | packed sparse up/gate retained; a target-shaped `moe_compute` run with real layer-0 weights failed its BF4 numerical gate (expert 127 PCC 0.977038), and `compute_only` exposes final double buffers rather than token-ordered score-reduced output; the combining path requires a collective/fabric contract |
| MoE gate/router dedicated path | available gate ops do not match Gemma's scale/top-k contract; two scale merges failed PCC |
| paged KV update | `paged_fused_update_cache` retained for its supported natural-cache contract |
| shared-LHS matmul | expert up/gate retained; dense gate/up default and adapted `minimal_matmul_split` lost all four alternating same-run traced comparisons at candidate PCC 0.999927–0.999970 |
| permute/reshape simplification | expert reshape/transpose chain is ordering-bearing; no identity composition remains |
| decode RoPE/layout folding | adapted full-attention decode-mode form failed PCC (0.946838) |
| input activation into binary | expert GELU folded into multiply; current-source dense always-fold candidate regressed decode and had no material prefill improvement (<0.1%), so explicit dense GELU remains |
| composite `geglu` | 3D form was adapted to 4D; explicit lowering won 3/4 raw decode medians, the aggregate with no significant candidate win, and both prefill comparisons |
| bias/activation/transpose into matmul | no bias; dense activation fold was rejected by the complementary current-source control; weight orientation is already native |
| slice after matmul | expert slices select both required halves, so narrowing the operand would recreate two matmuls |
| reduction/reshape and scaled-sum/mean | routed weighted sum is not an unweighted mean; existing fast reduction is canonical |
| layout conversion / reshard elimination | no measured reshard remains; removing decode RMSNorm interleaved conversions hits the device op's explicit height-sharded-input rejection |
| conv, pooling, batchnorm, spatial mean, RepVGG | not present |
| distributed RMSNorm and fused collectives | out of scope for this single-device stage |

Rejected-candidate JSON files preserve the numerical or timing evidence. No
further applicable single-device decoder fusion beat the retained graph.

## Evidence and commands

Primary artifacts are the PCC, trace, boundary, context, A/B, and rejected
candidate JSON files in this directory; `tracy/` contains filtered
`tt-perf-report` CSV tables plus device-trace replay CSV/text data.
`watcher_clean.log` is the separate
watcher run (7 passed, zero watcher faults). `tracy/provenance.json` and
`artifacts/watcher_provenance.json` bind those fresh captures to all four fused
and functional implementation/test files. `source_binding.json` binds the
default final suite to the same bytes, and `final_suite.log` retains its
17-passed/21-opt-in-skipped terminal result.

```bash
GEMMA4_RANGE_DOWNLOAD=1 TTNN_CONFIG_OVERRIDES='{"throw_exception_on_fallback": true}' \
  pytest -q models/autoports/google_gemma_4_26b_a4b_it/tests/test_fused_decoder.py::test_fused_hf_acceptance

GEMMA4_RANGE_DOWNLOAD=1 GEMMA4_FUSED_DECODER_PERF=1 \
  GEMMA4_FUSED_DECODER_SEQ_LEN=1024 \
  TTNN_CONFIG_OVERRIDES='{"throw_exception_on_fallback": true}' \
  pytest -q models/autoports/google_gemma_4_26b_a4b_it/tests/test_fused_decoder.py::test_fused_decoder_functional_ab_latency

GEMMA4_RANGE_DOWNLOAD=1 GEMMA4_FUSED_DECODER_CANDIDATE_AB=1 \
  TTNN_CONFIG_OVERRIDES='{"throw_exception_on_fallback": true}' \
  pytest -q models/autoports/google_gemma_4_26b_a4b_it/tests/test_fused_decoder.py::test_final_vs_best_candidates \
  models/autoports/google_gemma_4_26b_a4b_it/tests/test_fused_decoder.py::test_candidate_matrix_selection

GEMMA4_RANGE_DOWNLOAD=1 GEMMA4_MOE_COMPUTE_CANDIDATE=1 \
  pytest -q models/autoports/google_gemma_4_26b_a4b_it/tests/test_fused_decoder.py::test_moe_compute_gemma4_target_shape_candidate

GEMMA4_PREFILL_CAPACITY_LENGTH=262144 GEMMA4_RANGE_DOWNLOAD=1 \
  TTNN_CONFIG_OVERRIDES='{"throw_exception_on_fallback": true}' \
  pytest -q models/autoports/google_gemma_4_26b_a4b_it/tests/test_fused_decoder.py::test_fused_prefill_capacity_probe

GEMMA4_PREFILL_CAPACITY_LENGTH=262143 GEMMA4_RANGE_DOWNLOAD=1 \
  TTNN_CONFIG_OVERRIDES='{"throw_exception_on_fallback": true}' \
  pytest -q models/autoports/google_gemma_4_26b_a4b_it/tests/test_fused_decoder.py::test_fused_prefill_capacity_probe

GEMMA4_RANGE_DOWNLOAD=1 GEMMA4_FUSED_DECODER_PROFILE=1 \
  GEMMA4_FUNCTIONAL_DECODER_SEQ_LEN=32 \
  TTNN_CONFIG_OVERRIDES='{"throw_exception_on_fallback": true}' \
  python -m tracy -r -p --check-exit-code \
  -o models/autoports/google_gemma_4_26b_a4b_it/doc/fused_decoder/tracy/current_raw \
  -m pytest models/autoports/google_gemma_4_26b_a4b_it/tests/test_fused_decoder.py::test_fused_decoder_perf_profile -q

python -m tracy.process_ops_logs \
  -o models/autoports/google_gemma_4_26b_a4b_it/doc/fused_decoder/tracy/current_raw \
  --force-legacy-device-logs

GEMMA4_RANGE_DOWNLOAD=1 TT_METAL_WATCHER=10 \
  TTNN_CONFIG_OVERRIDES='{"throw_exception_on_fallback": true}' \
  pytest -q models/autoports/google_gemma_4_26b_a4b_it/tests/test_fused_decoder.py::test_fused_hf_acceptance \
  models/autoports/google_gemma_4_26b_a4b_it/tests/test_fused_decoder.py::test_fused_traced_decode_batch_contract
```
