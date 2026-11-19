# Fused corrected matmul: propagated graph and timing probes

The parent measured the frozen helper `tests/v6_fused_corrected_matmul_probe.py` (SHA256 `845c959110a713f808d1475eb22ded5ffa4e6ddd8209f02170fb90ed02b353dc`) on the actual prefix matrices. `v6_fused_corrected_matmul_gram_device.json` passed one-head Gram FP32 comparison against both the frozen adapter and CPU. `v6_fused_corrected_matmul_all_device.json` passed all10 actual matrix cases at32 heads and the additional leading-axis broadcast case bit-for-bit against the frozen adapter. Its previously known tiny CPU differences are unchanged. Both reports record clean device close. These component results do not establish propagated output/state parity or latency.

Two new diagnostic wrappers preserve that helper and all existing sources:

- `tests/v6_chunk64_fused_corrected_output_probe.py`, SHA256 `ae0e11d3c683be1cede6ee18bea429c1b2cb6d06a7da6ac1917fd24c3d75713d`. Its successful source evidence is `v6_chunk64_fused_corrected_output_source_v2.json`; dependency and literal kernel snapshots accompany it. The first source-audit attempt's partial artifacts remain preserved. Its failure was an assertion comparing a pre-attachment state-function identity: the unchanged existing helper regenerates that function during metadata setup. The corrected assertion compares the attached state call to the current frozen helper's installed function, matching the existing wrapper's proof.
- `tests/v6_fused_corrected_matmul_timing_probe.py`, SHA256 `43b8d7bff62b7bb53bda4526988d1019add3782b7f0b8a8eb6dd88ce93589ebe`. Source-only evidence is `v6_fused_corrected_matmul_timing_source.json`, with exact dependency snapshots. Its default execution does not import Torch/TTNN; hardware requires `--run-device`.

The output wrapper changes only construction of the matrix adapter: the unchanged corrected-FMA output wrapper installs its existing `mm` method, which calls `self.corrected_mm`, and that object is now the fused helper. Output equations, state equations, canonical triangular sums, double-single cumulative sums, scaled MKL exponentials, initial state, padding and trimming are unchanged. The earlier corrected wrapper still constructs its now-unused `CorrectedFma` setup object, but no per-K adapter is called by the graph. This temporary setup redundancy is confined to the diagnostic. The constructor substitution is restored on exit, and hashes of the frozen helper/wrapper are checked.

Each output-wrapper run pins the11-case matrix report and records every wrapper/helper source plus all three literal fused-matmul kernel strings. Run the full output/state comparator with fresh paths:

```bash
python_env/bin/python -m models.autoports.qwen_qwen3_6_35b_a3b.tests.v6_chunk64_fused_corrected_output_probe --lengths 128 --initial-state zero --variant sfpu_fma --sum-order canonical --output models/autoports/qwen_qwen3_6_35b_a3b/doc/fused_decoder/endurance/v6_chunk64_fused_corrected_output_device.json
python_env/bin/python -m models.autoports.qwen_qwen3_6_35b_a3b.tests.v6_chunk64_fused_corrected_output_probe --lengths 127 65 1 --initial-state prefix64 --variant sfpu_fma --sum-order canonical --output models/autoports/qwen_qwen3_6_35b_a3b/doc/fused_decoder/endurance/v6_chunk64_fused_corrected_output_nonzero_device.json
```

The timing probe compares complete warmed adapters from identical persistent DRAM FP32 inputs to DRAM FP32 outputs. Baseline timing includes its slicing, WHERE expansion, resharding, per-K corrected generic calls, and final placement. Fused timing includes its single generic operation and descriptor preparation. Setup/uploads/readback are outside timing. Defaults are32 heads, Gram/transformed-V/state-update matrix kinds,3 warmups,7 eager samples, and5 groups of100 traced replays per variant. It records eager and traced medians separately, exact first/last replay equality and determinism, actual input fixture hashes and variant order. Watcher/device profiler/trace-allocation tracking must be off. These are matrix microbenchmarks, not clean decoder latency or full-model throughput claims.

```bash
python_env/bin/python -m models.autoports.qwen_qwen3_6_35b_a3b.tests.v6_fused_corrected_matmul_timing_probe --run-device --output models/autoports/qwen_qwen3_6_35b_a3b/doc/fused_decoder/endurance/v6_fused_corrected_matmul_timing_device.json
```

The static decoder integration described below embeds literal kernels after the propagated component gates passed. It retains the exact increasing-K correction body and local FP32 feedback pack/unpack boundary. Whole-decoder correctness, long-context coverage and before/after warmed performance remain required after integration.

## Measured complete-adapter timing

The parent completed the warmed microbenchmark in `v6_fused_corrected_matmul_timing_device.json`, SHA256 `1ed303e43dbbec0a3e8dc4ed9b1802c3bbb62c7d4173ca7550130d24f4b310d4`. All eager and first/last traced outputs match the frozen adapter bit-for-bit;500 timed trace replays per variant preserve exact output determinism. The report records clean device close. These are complete matrix adapter measurements on32 heads; they are not integrated decoder latency.

| Actual-prefix matrix | Eager baseline ms | Eager fused ms | Traced baseline ms | Traced fused ms |
|---|---:|---:|---:|---:|
| chunk0_gram | 62.51995685 | 6.70577725 | 13.45461332 | 6.48915449 |
| chunk0_transformed_v | 30.71019193 | 6.71368185 | 10.27650309 | 6.49258298 |
| chunk0_state_update | 30.97665077 | 13.38012237 | 17.04831992 | 12.95708924 |

The parent also completed the propagated output/state gates: zero initial state at128 and prefix64 initial state at127,65,1 all have zero FP32 output, BF16 output and final FP32 state code differences against HF. The exact reports are `v6_chunk64_fused_corrected_output_zero_device.json` and `v6_chunk64_fused_corrected_output_nonzero_device.json`.

## Static integration from700c

The new builder `tests/v6_build_fused_corrected_matmul_snapshot.py` emitted `v6_corrected_fma_full_v5_streaming_scan_norm_tail_fused_matmul.py.snapshot`, SHA256 `8cda999c18f9245c872e40d7f67e027838b56396056ede43e162b7a8386f3faa`. Its `.provenance.json` and appended builder/reader/compute/writer snapshots record every source/evidence hash.

The base is the frozen streaming-scan/norm-tail integration SHA256 `700c10db90b23fac1d40427c9bbcd82a6924c7704699d54768a0160c2b641e05`. The builder validates the11 matrix cases and all four propagated output/state cases before writing. It substitutes only `Chunk64StateGraph.__init__` adapter construction and embeds the measured helper with literal kernel sources. The old `CorrectedFma`, `CorrectedMatmul` and their old compute constant are removed only after verifying no references remain. Their retained input tuple and cached WHERE predicates are no longer allocated. The new helper's `last_plan` is metadata only.

Reverting that one setup substitution, removing the new embedded helpers/literals and reinserting the three old top-level nodes restores the complete baseline AST. `FusedDecoder`, `Chunk64OutputGraph` and `Chunk64StateGraph.mm` are individually unchanged. Thus decode, original V5 full-attention branches, public singleton prefill, input projection policy, norm-tail fusion, cache/padding contracts and output/state equations are preserved at source level. The measured helper is alpha-equivalent after collision-avoiding global renames; only its setup source-generator call becomes a literal source constant. No test imports, runtime AST execution or source-generation calls enter the snapshot.

This snapshot has not been integrated-device tested by the builder agent. Component passes and matrix speedups alone do not establish whole-decoder correctness, warmed latency, long-context feasibility or stage acceptance. The independent local-LUT repair is intentionally absent; that experiment can be combined after its own proof.
