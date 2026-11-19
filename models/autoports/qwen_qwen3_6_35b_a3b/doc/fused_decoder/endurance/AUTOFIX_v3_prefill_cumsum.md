# Diagnostic: compact BF16 prefill projection with token-batched cumsum

Prepared against immutable `selected_fused_decoder_v3.py.snapshot`, SHA256 `73ba9aabb10a2c34f3b10b5ff1fc93418470c0a624dd863e87a8c8c2af2901ba`. This is an unselected source/CPU diagnostic; the preparer has not run hardware. Three independent snapshots vary only projection microbatch size: `v3_prefill_cumsum_chunk{4,16,32}.py.snapshot`. Each has a patch and provenance sidecar. Production, old candidates, and core harnesses are unchanged.

The singleton v3 FastReduce projection is unchanged, including its K512/K1024 policy. Multirow projection always keeps the existing K1024 order, even for a final one-token internal microbatch. Projection subdivision changes neither public prefill chunks nor recurrent cache rounding boundaries. AST comparison confirms that only setup (`from_state_dict`) and `linear` differ.

## Arithmetic and native contract

Setup uploads ordered BF16 weights `[1,2,1024,6208]`. Runtime gathers the same odd/even K order and reshapes each token microbatch into BF16 coefficients `[rows,2,1024,1]`. Accurate BinaryNG multiplication produces FP32 `[rows,2,1024,6208]` in interleaved DRAM. Native uncompensated FP32 cumsum scans height1024 in order, and the last row supplies two partials per token. FastReduce over dimension1 folds those two partials in the original order. Cast, row repack, and concat return `[1,S,6208]`.

Relevant source contracts:

- `operations/reduction/accumulation/device/accumulation_device_operation.cpp:16–111` accepts interleaved TILE forward FP32 height scans on Blackhole, with unchanged input/output dtype and32x32 tiles. This rank4 shape follows the direct height path.
- `accumulation_program_factory.cpp:442–545` uses FP32 destination/unpack and the tile-axis SFPU implementation. `blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_cumsum.h:69–99` performs sequential SFPADD with a carried FP32 accumulator. Compensation is explicitly disabled.
- Native cumsum writes every prefix and allocates a separate output; `accumulation_common.cpp` invokes the primitive with no preallocated output, then rebinds any optional public output. There is no claimed in-place or last-only optimization.
- FastReduce's FP32 branch copies its first operand then adds sequentially (`fast_reduce_nc/device/kernels/reduce_nc.cpp:42–59`). Cumsum instead starts from positive zero. This preserves ordinary finite arithmetic but is not a universal signed-zero bit-equivalence proof. The diagnostic requires exact final FP32 and BF16 bits, without a signed-zero waiver.
- Before selecting the final row, the adapter slices the aligned terminal32-row band. `data_movement/slice/slice.cpp` can untilize a complete input for a nonaligned slice; this two-stage selection confines that work to the band.
- The final FastReduce receives an explicit FP32 TILE DRAM output `[rows,1,1,6208]`, allocated inside the measured method. `fast_reduce_nc_device_operation.cpp:52–66,117–146` accepts this same-rank output and preserves its TensorSpec. Physical geometry remains `[rows,1,32,6208]`, but logical height stays1. This incorporates the source-backed correction validated separately by the parent's singleton cumsum probe; it avoids the default padded-logical-height reshape failure.

## Capacity and comparison with selected prefill

The selected v3 filled-row PF weight occupies813,694,976 bytes (776MiB). This candidate replaces that PF-only allocation with25,427,968 bytes (24.25MiB), while retaining original singleton weights. Coefficients need `rows*131072` physical bytes. Each FP32 product or full-prefix allocation needs `rows*50,855,936` bytes:

| Projection rows | Product bytes | Prefix bytes | Simultaneous product + prefix |
| ---: | ---: | ---: | ---: |
| 4 | 203,423,744 | 203,423,744 | 406,847,488 |
| 16 | 813,694,976 | 813,694,976 | 1,627,389,952 |
| 32 | 1,627,389,952 | 1,627,389,952 | 3,254,779,904 |

These are DRAM allocations, not an L1 capacity claim. The adapter releases product and prefix references as soon as possible, but actual allocator/trace reuse remains a device check. The terminal band adds1,589,248 bytes per row before its row extraction. Small chunks trade lower peak allocation for more gathers, conversions, kernel dispatches, row repacks, and concat inputs. The full-prefix write remains real overhead; the singleton result alone does not establish PF speed.

This deliberately revisits an old scan algorithm under a different complete adapter. Historical `cumsum_reductions.py.snapshot` already used PF microbatches4 and K-height scans, with full12352-column FP32 operands; `cumsum_reductions_0.json` measured332.131ms PF/5.773ms decode, and `last_only_reduction_0.json` measured332.241/5.157ms. Those results remain valid and preserved. The present discriminator uses the selected6208-column split, compact BF16 operands, unchanged v3 singleton, bounded terminal extraction, explicit logical output, and independent4/16/32 chunk choices. It is not a claim that token-batched cumsum itself is new.

## Probe and CPU evidence

`tests/v3_prefill_cumsum_probe.py` invokes the actual baseline and candidate `linear` methods using layer0 checkpoint K/V/A/B weights. Inputs are explicitly labelled synthetic normalized BF16 rows (seed154), rather than recorded activations. CPU-only artifacts `v3_prefill_cumsum_chunk{4,16,32}_cpu.json` pass exact FP32 partial and final checks on33 rows and12 columns spanning semantic/tile boundaries. The CPU checks execute each proposed microbatch subdivision and compare copy-first reduction against the positive-zero-start scan. They establish fixture mapping/arithmetic only, not whole-model acceptance.

The device discriminator compares all6208 columns at FP32 and BF16 boundaries for S2/32/33/128/129, plus unchanged singleton K512/K1024. Timing is skipped on any mismatch. Complete warmed traced timing includes gather, coefficient creation, dtype/layout changes, DRAM product and all-prefix scan, both terminal slices, explicit output allocation, final block reduction, cast, row repack, and concat. First/last replay outputs are checked. Physicalchip0, trace32MiB, L1-small32768, synchronized trace release, and `finally` device close are explicit. Source/weight/input/candidate hashes are recorded.

Run each candidate serially with a distinct output (replace4 by16 or32 for the other two):

```sh
python_env/bin/python -m models.autoports.qwen_qwen3_6_35b_a3b.tests.v3_prefill_cumsum_probe --candidate-snapshot models/autoports/qwen_qwen3_6_35b_a3b/doc/fused_decoder/endurance/v3_prefill_cumsum_chunk4.py.snapshot --output models/autoports/qwen_qwen3_6_35b_a3b/doc/fused_decoder/endurance/v3_prefill_cumsum_chunk4_probe.json
```

All three source snapshots and the diagnostic parse successfully; source/probe/CPU proof hashes are in the sidecars. Hardware exactness, allocation behavior, and complete-adapter latency remain pending. A passing diagnostic still requires the parent's full-path benchmark and fresh-HF acceptance before selection.

## Parent device results: DRAM variants rejected for speed

The parent subsequently ran all three frozen variants. Each `v3_prefill_cumsum_chunk{4,16,32}_probe.json` reports exact outputs on all seven cases. S128 complete traced adapter medians were:

| Microbatch | Selected-v3 baseline ms | DRAM cumsum ms |
| ---: | ---: | ---: |
| 4 | 48.520157 | 75.148055 |
| 16 | 48.389867 | 68.125666 |
| 32 | 48.518207 | 67.021932 |

Thus none earns selection. The preparation evidence above remains historical; these are parent-owned hardware results, not execution by this investigator. Old snapshots, probes and reports remain immutable.
