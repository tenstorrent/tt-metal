# Corrected matmul transpose fusion: candidate assessment

All three transpose sites remain eligible for whole-decoder testing. They are **not selected** by this source-only build. The mandatory remaining checks are whole-decoder correctness/semantics and full prefill/decode latency against the best correct candidate.

## Matched component result

`v6_corrected_matmul_transpose_matched_timing_device.json` passed and closed cleanly. Every one of36 paired traced differences was negative, across both capture orders and all three sites. Each adapter/order has120 replays with exact outputs checked after every batch.

| Site / capture order | Baseline traced ms | Candidate traced ms | Gain | Eager baseline→candidate ms |
|---|---:|---:|---:|---|
| actual_chunk0_output_attention / candidate→baseline | 6.496596 | 6.483852 | 0.196% | 6.779650→6.873974 |
| actual_chunk0_output_attention / baseline→candidate | 6.496589 | 6.483996 | 0.194% | 6.719094→6.689877 |
| actual_chunk0_state_update / candidate→baseline | 12.965291 | 12.933574 | 0.245% | 13.248228→13.188094 |
| actual_chunk0_state_update / baseline→candidate | 12.965786 | 12.934255 | 0.243% | 13.250494→13.192272 |
| actual_chunk0_gram / candidate→baseline | 6.498091 | 6.484575 | 0.208% | 6.838712→6.799998 |
| actual_chunk0_gram / baseline→candidate | 6.496153 | 6.484090 | 0.186% | 6.883442→6.811411 |

Both input buffer addresses were5848704 and5979776 across all three fixtures. Candidate Gram/attention output remained6176384 in either capture order; corresponding native output moved6372992→6307456. State candidate output moved6110848→6766208; native moved6766208→6241920. Gains persisted across these allocation/capture differences. The test ran output attention first, reversing original site order, and alternated balanced paired measurements while both traces remained live. No source arithmetic, geometry, fidelity, or ordering changed.

The earlier attention candidate trace7.247840ms regression did not recur; its cause remains unproven. Original five slow samples and initial AICLK1343-versus-requested1350 warning remain preserved in `v6_corrected_matmul_transpose_device.json/.log`. No per-sample clock telemetry established clock causality. Matched eager attention remains mixed: first capture order regressed about0.094ms; second improved about0.029ms. Thus this assessment does not infer warmed whole-prefill benefit from traced component data alone.

## Complete chunk result

Both `v6_chunk64_transpose_output_zero_device.json` and `v6_chunk64_transpose_output_nonzero_device.json` report `complete_adapter_exact` and clean closure. All three sites enabled, logical lengths127,65,1; nonzero state uses the canonical prefix64 state. Complete BF16 output, FP32 output, final FP32 state, and all observed intermediates match the unchanged803 graph exactly. Candidate HF output/state gates also remain exact. The wholechunk wrapper preserves every unmodified exp/cumsum/triangular recurrence and padding boundary.

The zero-state length127 complete adapter timing (candidate measured first) improved warmed eager168.689→164.375ms (~2.56%) and traced160.427→159.540ms (~0.55%). These are chunk fixture timings, not whole-decoder timings. Nonzero run is correctness-only. Exact detailed samples and artifact source hashes remain in those JSON files.

## Frozen standalone integration

Candidate: `v6_transport_conv_norm_decay_transpose_integration.py.snapshot`, SHA256 `85adacf5df348f815e684a6ad1b1a36ecf12e862b72ef156681bfabb86dccd92`.
Base44f87: `44f87c04349836a4ad8b09c7a6c573fe3afe1e558af285e0482cce2c0fe5b2b3`.
Builder: `tests/v6_build_matmul_transpose_snapshot.py`, SHA256 `fc4a38c99377a79ba4066b3675cc490e2a1fcc68049b4fb5535a3295fc8a6f04`.
Build proof: `v6_transport_conv_norm_decay_transpose_integration_build.json`; adjacent static reader/compute/writer and executed builder snapshots preserve exact source bytes.

The builder appends `TransposeCorrectedMatmul` and `TRANSPOSE_CORRECTED_MM_READER`, adds one nonallocating constructor to `Chunk64StateGraph`, and replaces exactly three call expressions (Gram key transpose, state-update weighted-key transpose, output-attention key transpose). Other matmul calls still use the original corrected helper. Precise inverse substitutions, setup removal and helper removal restore the entire44f87 AST exactly. All addressed source methods/compute/writer/planner definitions were proved identical between the measured803 component base and44f87. The complete `FusedDecoder` AST remains unchanged. Static runtime helper contains no diagnostic test imports or AST/source generation.

Pinned autoflake2.3.1/isort5.13.2/Black23.10.1 ran before snapshot freeze; formatted AST is exact and all hook checks pass. No production source or frozen helper was edited. CPU source preparation did not open a device.

Audit extension references: helper AST `ad92b3e06964b793f0daa48c6259c0f465315b75c2d5cc88cf63a417ef923b01`, constructor AST `b7a0368ac991fbcbbe48b9083a3275247e6f7f824abd90b0be82377c3bcc3412`. Constructor only assigns references/source strings and queries grid metadata; no setup exemption is warranted. Pin the new reader constant and audit the complete helper runtime. The compute and writer strings remain the original corrected matmul constants.

No persistent device allocation is added; the MM CB footprint remains40960bytes/core on at most64cores. Operand transpose temporaries are removed, with no change to output/state shape, context contract, or padding capability. Source analysis suggests no capacity reduction; final stage capacity qualification remains parent-owned.

Rebuild on fresh paths:

```bash
PYTHONPATH=. python_env/bin/python models/autoports/qwen_qwen3_6_35b_a3b/tests/v6_build_matmul_transpose_snapshot.py \
  --output models/autoports/qwen_qwen3_6_35b_a3b/doc/fused_decoder/endurance/v6_transport_conv_norm_decay_transpose_integration_fresh.py.snapshot \
  --report models/autoports/qwen_qwen3_6_35b_a3b/doc/fused_decoder/endurance/v6_transport_conv_norm_decay_transpose_integration_fresh_build.json
```

Next: parent serial whole-layer34 cached correctness and S256 warmed prefill/traced decode. Retain only if full acceptance remains unchanged and final measured performance beats the best correct candidates. Current component and chunk evidence authorizes testing, not promotion.
