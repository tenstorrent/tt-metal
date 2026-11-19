# Diagnostic: two-token prefill projection with sequential K1024 blocks

Immutable baseline [v5_prefill_pair_blocks_baseline.py.snapshot](v5_prefill_pair_blocks_baseline.py.snapshot) is selected v5 SHA256 `8ac733159befba919abf705603f3aada9d105195c7b495d8cd25aa65e531769f`. Isolated candidate [v5_prefill_pair_blocks.py.snapshot](v5_prefill_pair_blocks.py.snapshot) is SHA256 `980501c216e00693d920149bf95f14e78ea9e29d92745eccd65e6b0b4e1ca62a`; [preparation provenance](v5_prefill_pair_blocks.py.provenance.json) binds both. Production, prior controls and core harnesses are untouched. No hardware was run by the preparer.

**Measured decision: reject.** The complete projection adapter and matched whole-decoder prefill are slower; see the source-bound results below. The original hypothesis was a source-legal adapter saving. It replaces two one-token/two-block scans with two two-token/one-block scans. Each kernel still scans K1024 sequentially, but its independent outer dimension represents tokens rather than K blocks. The two block results are added once in FP32 in the original order, and only then cast to the requested BF16 or FP32 output. Public singleton K512 and explicit internal-singleton K1024 paths are AST-identical to baseline. Odd tails call that unchanged singleton path with K1024; no public cache boundary or recurrent arithmetic changes.

## Shape and arithmetic contract

For each pair of logical input rows:

1. Slice the existing RM input and reshape `[1,2,2048] -> [2,64,32]`. Gather the same even/odd pair ordering with setup-prepared RM UINT16 indices `[2,64,32]`, then RM-reshape to `[2,2048]`.
2. For block b=0 then b=1, RM-slice columns `[1024*b:1024*(b+1)]` and reshape/tilize coefficients to `[2,1,1024,1]`. The setup BF16 DRAM half-weight `[1,1,1024,6208]` broadcasts over the token axis. Accurate mixed multiply produces FP32 L1 `[2,1,1024,6208]`.
3. Run the unchanged forward `cumsum(dim=-2, dtype=FP32, disable_compensated_sum=True, memory_config=L1)`. Crop aligned rows `[992:1024]`, release the full prefix, and extract final row31. Keep the first block's FP32 partial while computing the second.
4. Compute `partial0 + partial1` in FP32, cast once, untilize to compact DRAM RM, reshape `[2,1,1,6208] -> [1,2,6208]` and retain the output. The final concat/tilize includes all retained pairs and an optional singleton tail.

Exact native contracts, relative to repository root:

- `data_movement/gather/device/gather_device_operation.cpp:48–96,167` requires matching ranks and derives output shape from **indices**, not input. Reusing `[1,64,32]` indices would produce only one token; the new `[2,64,32]` setup allocation is required. Both input and indices remain RM (`:98–108`).
- `eltwise/binary_ng/device/binary_ng_device_operation.cpp:440–466` supports ordinary outer-axis singleton broadcast for these rank-4 inputs. Both tile dimensions remain32×32, satisfying `:480–501`. There is no repeated weight-row expansion into L1.
- `reduction/accumulation/device/accumulation_program_factory.cpp:442–554` handles forward height scans independently by outer plane/column. H1024 means32 tiles/run; token batching does not alter that run. FP32 input uses direct-destination unpack and32-bit destination. `device/kernels/compute/accumulation_tile_axis_compute.cpp:25–87` starts each independent run with `first=true`, then carries the FP32 running total across its32 tiles. Compensation remains disabled.
- `experimental/reduction/fast_reduce_nc/device/kernels/reduce_nc.cpp:42–59` copies partial0 then adds partial1 for the baseline's two-block reduction. BinaryNG FP32+FP32 ADD selects SFPU (`binary_ng_device_operation.cpp:49–58`) with FP32 direct-destination unpack. The candidate leaves the fast flag unset; explicitly false is not accepted for FP32 ADD. This is the same two-partial adaptation that passed the older sequential-block probe, not FMA or reassociated multi-block reduction.

## Setup residency, live L1 bound and prior controls

The candidate adds two compact DRAM BF16 half-weight uploads totaling25,427,968 bytes (24.25MiB), plus8,192 bytes of RM UINT16 indices. The original compact weight stays resident for unchanged singleton/decode behavior. No supported zero-copy tensor offset view was identified for the two half ranges; the additional residency is explicit, not mislabeled as a view. This is a bounded tradeoff to avoid a weight slice/copy on every runtime pair, much smaller than the historical776MiB filled-row prefill representation. A retained candidate would still require whole-model capacity gates; source arithmetic cannot prove that capacity.

No weights are staged into L1 during runtime. The two-token one-block product and prefix each contain50,855,936 bytes; their simultaneous pair is101,711,872 bytes (97MiB), equal to the current one-token/two-block product-plus-prefix payload. The retained first partial adds1,589,248 bytes (1.515625MiB) of padded FP32 L1 allocation while block1 runs. Runtime input/output slices, gathers, coefficient tilizes, terminal extraction, addition, output conversions and concat are all included in the adapter.

This differs from both exhausted controls:

- [Transient L1 weights](AUTOFIX_v4_prefill_weight_l1.md) staged24.25MiB into L1 and doubled scans for each individual token. It was exact but S128 trace30.579791 ->33.885947ms, rejected. Here the independent two-token dimension preserves the baseline scan payload/parallelism while DRAM weights remain resident.
- [Earlier DRAM microbatches](AUTOFIX_v3_prefill_serial_l1.md) put large product/prefix tensors in DRAM; chunks4/16/32 were slower. This control keeps the full scans in L1 and batches only two tokens while serializing the K blocks.

The possible gain comes from fewer row gathers/output retention conversions/concats and replacing two FastReduce calls per pair with one FP32 addition. Additional coefficient slicing, outer weight broadcasting and retained-partial pressure may offset it. No speedup is inferred from op count or traffic alone.

## Why the three-token extension is not prepared under the same policy

Blackhole has **1536KiB physical L1 per worker**, not a hard-coded110MiB total (`soc_descriptors/blackhole_140_arch.yaml:151–152`, `hw/inc/internal/tt-1xx/blackhole/dev_mem_map.h:33`). A CPU preprocessor evaluation of the current memory map gives `MEM_MAP_END=41888`. `llrt/hal/tt-1xx/blackhole/bh_hal_tensix.cpp:44,70–71` reserves69KiB kernel configuration and aligns to64 bytes, yielding default unreserved start112576. With the probe's `l1_small_size=32768`, `impl/allocator/l1_banking_allocator.cpp:119–122,249–254` leaves1,427,520 bytes per interleaved bank. These are source-derived bounds, not live allocator measurements.

Using the established110-bank device geometry and4096-byte FP32 tile pages, per-allocation bank rounding is decisive (`impl/allocator/bank_manager.cpp:459–482`):

| Tokens per block | Tiles per product/prefix | Bytes per bank per product/prefix | Retained partial per bank | Pair + partial + cumsum buffers per bank | Remaining versus1,427,520 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 12416 | 462848 | 16384 | 974848 | +452672 |
| 3 | 18624 | 696320 | 24576 | 1449984 | **−22464** |

Cumsum allocates four input and four output FP32 tiles,32KiB per participating core (`accumulation_program_factory.cpp:474–475,551–553`). Three-token block1 therefore already exceeds the default per-bank budget by22,464 bytes before other model tensors, binary buffers, alignment or fragmentation. Only10,304 bytes remain after its product/prefix/partial, less than the required32KiB cumsum buffers. The two-token nominal margin is not an allocation guarantee, but three-token same-policy execution is not a justified probe here. Moving the partial to DRAM or changing allocator reservations would be a separate adapter/control with different costs; neither is silently applied. The root should test the prepared two-token control first.

## Complete projection probe

[tests/v5_prefill_pair_blocks_probe.py](../../../tests/v5_prefill_pair_blocks_probe.py) calls both actual snapshot `linear` methods on real checkpoint K/V/A/B weights and explicitly labeled synthetic normalized BF16 inputs. It executes the candidate's **actual AST setup assignments** to prepare DRAM halves and indices, checks the uploaded halves by readback, and records setup scope/time separately from runtime latency. No host work, weight staging or omitted adapter is inserted inside the measured method.

Default device cases are singleton K512 and K1024, then S2/3/32/33/128/129 with internal K1024. All6208 FP32 and BF16 output codes are compared, with separate K/V and A/B summaries. Odd3/33/129 cases exercise the unchanged singleton tail. Mismatched cases skip timing; diagnostic flags and model gates are not weakened. Complete synchronized eager and warmed traced BF16 adapter timings include all runtime work, with post-eager and first/last replay output checks. Source, setup AST, checkpoint/input and helper hashes are recorded. Physical chip0, trace32MiB, L1-small32768 and clean timing environment are explicit; trace release/device close are protected by `finally`.

[CPU setup report](v5_prefill_pair_blocks_cpu.json) passes source-scope/singleton-AST checks and selected-column FP32 partial/final mapping on33 rows. This does not prove device allocation or all-input exactness. Pre-commit passes all applicable hooks for the new probe. The baseline and candidate have not been run on hardware by the preparer.

Root-only commands after current validation:

```bash
python_env/bin/python -m models.autoports.qwen_qwen3_6_35b_a3b.tests.v5_prefill_pair_blocks_probe \
  --output models/autoports/qwen_qwen3_6_35b_a3b/doc/fused_decoder/endurance/v5_prefill_pair_blocks_probe.json

for variant in baseline candidate; do
  if [ "$variant" = baseline ]; then suffix=_baseline; else suffix=''; fi
  python_env/bin/python -m models.autoports.qwen_qwen3_6_35b_a3b.tests.benchmark_fused_snapshot \
    --snapshot "models/autoports/qwen_qwen3_6_35b_a3b/doc/fused_decoder/endurance/v5_prefill_pair_blocks${suffix}.py.snapshot" \
    --layer 0 --length 256 --context 512 --prefill-repeats 10 --decode-repeats 60 \
    --output "models/autoports/qwen_qwen3_6_35b_a3b/doc/fused_decoder/endurance/v5_prefill_pair_blocks_${variant}_s256.json"
done
```

If the adapter passes and wins, S4097 whole-prefill timing, unchanged decode behavior, added-residency capacity and the full unchanged acceptance queue remain separate requirements. That conditional validation plan is superseded by the measured rejection below; no additional long test is needed for this slower control.

## Measured rejection: root-run device results

The root completed the [projection probe](v5_prefill_pair_blocks_probe.json) with all eight cases exact in both requested output dtypes and `device_closed=true`. These are finite-fixture device results, not a proof of all-input equality. [Compact decision and SHA256 evidence bindings](v5_prefill_pair_blocks.decision.json) verify the immutable baseline, candidate and probe hashes against the measured artifacts. Original reports and provenance remain unchanged.

| Complete traced projection | Baseline ms | Candidate ms | Change |
| --- | ---: | ---: | ---: |
| S1, K512 | 0.278988 | 0.279176 | +0.07% |
| S1, K1024 | 0.250972 | 0.250890 | -0.03% |
| S2, K1024 | 0.501190 | 0.630133 | +25.73% |
| S3, K1024 | 0.740209 | 0.873234 | +17.97% |
| S32, K1024 | 7.660235 | 9.861208 | +28.73% |
| S33, K1024 | 7.903164 | 10.108249 | +27.90% |
| S128, K1024 | 30.584268 | 39.408567 | +28.85% |
| S129, K1024 | 30.839057 | 39.666552 | +28.62% |

Every modified multirow case regresses. The S128 adapter grows from **30.584268 to 39.408567 ms** (+28.85%). Singleton K512/K1024 are unchanged code and their small timing differences do not establish a gain.

The matched whole-decoder [baseline](v5_prefill_pair_blocks_baseline_s256.json) and [candidate](v5_prefill_pair_blocks_candidate_s256.json) use layer0, S256/context512, ten prefill samples and sixty decode samples:

| Whole decoder | Baseline ms | Candidate ms | Interpretation |
| --- | ---: | ---: | --- |
| Prefill | 107.813201 | 121.481796 | +12.68%; reject |
| Traced decode | 1.285217237 | 1.285358099 | Unchanged path; no measured gain |

Both report identical prefill PCC0.999962395060449, prefill contribution PCC0.9999594863488576, HF prefill-state PCC0.9999985691639764 and eager/traced decode PCC0.9999819053980984; both are deterministic and pass the runtime fallback audit. These snapshot benchmarks are explicitly selection diagnostics, not acceptance artifacts. **Reject the paired-block candidate for measured performance.** Its extra24.25MiB weight residency is consequently not retained. Further long-stream testing cannot repair this clear latency regression and is unnecessary for this rejection.

## Source follow-up: no supported small correction identified

For two tokens, both graphs execute two product/cumsum pairs of shape `[2,1,1024,6208]`; the rewrite did not reduce the dominant scan count or scan payload. It changes the outer-axis meaning, introduces two RM coefficient-width slices, changes weight addressing to an outer broadcast, and retains the first FP32 partial during the second scan. The expected saving from output/gather adapters demonstrably loses to the complete adapter's other costs. Without an operation profile, the observed regression cannot be assigned to one of those costs.

A suspected hidden weight expansion is **not supported by the selected native path**: `ttnn/cpp/ttnn/operations/eltwise/binary/binary.cpp:654–860` invokes `prim::binary_ng` without calling the legacy `preprocess_inputs` repeat helper at584–607. Native outer singleton broadcast is supported at `binary_ng/device/binary_ng_device_operation.cpp:440–466`. Removing a hypothetical materialized repeat would therefore not be a concrete repair. The probe also verifies both gathered token rows and all6208 output columns. No redundant mistaken conversion or slice was identified that credibly explains and removes the +28.85% adapter regression. No revised candidate or hardware run is proposed on this evidence.

## Separate lightweight layer-scale follow-up

The [checkpoint norm-vector survey](AUTOFIX_v5_representative_layers.md), with [compact numeric evidence](v5_layer_norm_scale_survey.json), proposes at most two additional linear-attention layers for native decode norm generalization: **34 first,14 second**. Layer34 has the largest linear gated-norm gamma (1.2890625, versus layer0 maximum0.95703125) and substantially larger postnorm effective-gamma RMS (1.543634 versus0.908804). Layer14 supplies a middle-depth attenuation extreme: the smallest gated-norm gamma (0.29296875) and wider scale range. These recommendations use only337.5KiB of norm vectors across the checkpoint; no projection matrices or activation distributions were read. They are supplemental diagnostic choices, do not replace existing gates, and make no full-model correctness claim. This note is confined to the owned work memo; root work_log is untouched.
