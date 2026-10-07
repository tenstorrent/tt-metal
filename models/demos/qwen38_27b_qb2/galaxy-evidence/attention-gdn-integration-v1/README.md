# Attention precision and GDN model boundary, 2026-10-07

All hardware measurements use one TP4 submesh of the allocated Galaxy `.98`,
the pinned `a08819ddbe23077f8037d3802303939064868ff6` runtime, and five warm
samples of 100 trace replays. Times include dispatch. No full-model speedup or
GPQA gain is established by these experiments.

## Attention

The original HiFi4/FP32/accurate-exp test used six BF16 query heads. The pinned
`sdpa_decode_program_factory.cpp` selects half tiles for this geometry.
`compute_common.hpp::sub_exp_block_bcast_cols_inplace` forces approximate exp
when `vector_mode != VectorMode::RC`, even when `EXP_APPROX_MODE` is false.
FP32 destination accumulation also does not make its BF16 intermediate and
statistics circular buffers FP32.

The full-tile control keeps the same six live queries, random seed, quantized
KV and shuffled page table, adding 26 zero query rows after generating inputs.
Full-tile approximate exp has the same error as half-tile approximate exp.
Full-tile accurate exp passes at 8K and 128K. At near 256K, increasing the
chunk to 256 also passes the unchanged per-user PCC >= 0.999 and relative-RMS
<= 0.02 gates. Approximate chunk/core tuning alone passed none of 48 variants.

| ISL / batch | Full-tile accurate chunk 256, max 16 cores: us | Worst-user relative RMS |
|---|---:|---:|
| 8192 / 1 | 74.99 | 0.00791 |
| 8192 / 16 | 236.53 | 0.00834 |
| 131072 / 8 | 1567.29 | 0.01289 |
| 262016 / 4 | 1515.85 | 0.01631 |

The opt-in model helper includes device Q padding and output slicing inside
the trace. Its `model-attention-v2/attention.json` passes eight cases: caches
of 32, 128 and 288 tokens, 1024/B16, 8K/B1/B16, 128K/B8 and near-256K/B4.
At 128K/B8 it measures 1569.57 us; near-256K/B4, 1518.12 us. Prefill math,
weights and the default `config/precision.json` remain unchanged.

Select the candidate with `QWEN_PRECISION_CONFIG` pointing to
`config/precision_accurate_decode.json`. New G0 qualification is required.
The G0 hashes now bind environment-selected precision artifacts, and the
serving supervisor explicitly forwards that qualified override to workers.
Previously its environment sanitization would have discarded the override.

## GDN

The experimental adapter accepts the model's raw convolution outputs and
performs device Q/K normalization, head expansion, gate/layout preparation,
four-way partitioned recurrence with two buffered items, and conversion to
the existing head-major output layout. It updates one FP32 state allocation
in place. Convolution, output gated RMSNorm and projection are excluded.
It remains outside the model and is not the final fused P1/P2 implementation.

All B1/8/16/32/64 candidate cases pass eager and 64-step trace checks against
FP32 reference on every rank, including unchanged inputs and a state view
without copying. The separate native control uses the current model's chunked
scan, batch splitting, concatenation and state copy with identical inputs.

| Batch | Candidate boundary us | Native boundary us | Native / candidate |
|---|---:|---:|---:|
| 1 | 148.19 | 85.99 | 0.58x |
| 8 | 211.71 | 234.73 | 1.11x |
| 16 | 283.60 | 643.64 | 2.27x |
| 32 | 400.91 | 1290.30 | 3.22x |
| 64 | 656.02 | 2488.18 | 3.79x |

The candidate regresses B1. The native control passes the first two eager
steps, then fails the strict per-head 0.5% relative-RMS gate after 64 traced
updates at every batch (worst-head output error 7.20–8.43%). The candidate's
worst-head output error remains below 0.10% after the same updates. This is
a timing comparison with the
existing implementation, not a claim that both paths passed the same accuracy
qualification. All native metrics, including failures, are retained. The
candidate gate was not relaxed. Native-control accuracy does not prove the
cause of the model's GPQA gap.

## Validation and retained failures

The integration CPU suite passes 169 tests and 40 subtests; one tokenizer test
is skipped without the checkpoint environment. Initial model-wrapper execution
failed because native `Shape` does not accept slicing; converting to a tuple
fixed it. Its devices closed cleanly. The first native-control run stopped at
the accuracy assertion; the diagnostic now records every control failure while
continuing timing, with candidate checks still required to pass.

Raw initial failure logs and launch receipts remain at
`/home/ttuser/qwen38-artifacts-20261007/{model-attention-v1,gdn-native-control-v1}`
and matching `.log` paths. Earlier numerical failures are published here.
`manifest.json` records SHA256 of the original uncompressed bytes. The two
`.snapshot` files preserve the test source used before integration edits.

Run the bounded tests through `demo/run_model_decode_attention.sh` and
`demo/run_gdn_model_adapter.sh`, giving a task root and a new output directory.
Set `QWEN_GDN_NATIVE_CONTROL=1` for the latter to collect the native comparison.
The wrappers use the shared device lock and the existing safe pytest runner.

## Full-model evaluation queued

The exact persistent launcher, systemd command receipt and frozen source
manifest are included alongside this README. The source manifest pins Metal
commit `e663e4d6d18edd09af8f48b828e9ca4e3ebbac72`. The service
`qwen38-accurate-decode-qualification-v1-20261007.service` starts new eight-replica
G0, then API checks and full GPQA at the unchanged 32K output budget and 89.2%
gate, conditional on G0 passing. It has an eight-hour outer deadline. Source
verification passed and model loading started; no new G0 or GPQA result is
claimed. The GDN candidate is excluded from this attention-only comparison.
