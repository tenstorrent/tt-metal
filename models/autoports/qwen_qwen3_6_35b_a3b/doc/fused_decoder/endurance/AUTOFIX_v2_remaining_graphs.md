# Independent graph candidates after v2

Base is immutable `../selected_fused_decoder_v2.py.snapshot`, SHA256 `0b96b6204c5c1b276b14c5f29d8dd4789569608e092cec90b7a3898ee1cfa26c`. Three independent snapshots below change only their stated expressions. Adjacent patches/provenance bind every snapshot to the base. Production, shared native source, existing harnesses and pattern assessment were not edited. No device execution was performed by this investigation.

## 1. Mixed inputnorm operands

`v2_norm_mixed_operands.py.snapshot` removes the initial full-input BF16-to-FP32 typecast from the exact linear inputnorm. Both `multiply(x,x)` and `multiply(x,inverse)` explicitly request FP32 output and `fast_and_approximate_mode=False`; all reductions, root/inverse refinements, gamma multiplication and output boundaries remain as in v2. The original DRAM placement adapter remains.

BF16 is exactly representable in FP32. Widening each BF16 input at the accurate SFPU multiply instead of materializing the widened tensor preserves the FP32 product boundary. This is the same operand-storage route already validated for ordered projection and the first convolution tap, applied to different consumers. It does **not** square in BF16 or narrow the FP32 inverse. Native `binary_ng_device_operation.cpp:55–67` selects SFPU for accurate MUL; `binary_ng_program_factory.cpp:1370–1386` requests direct FP32 destination unpack for SFPU inputs, and its output-format-dependent destination configuration preserves FP32 products. Unlike ADD/SUB, MUL allows the explicit accurate flag with FP32 output.

The saved allocation is the standalone FP32 input buffer, physically 262,144 bytes for a single 2048-channel tiled row or 1,048,576 bytes for 128 rows, plus its dispatch/read-write traffic. This is a modest decode candidate; no latency improvement is assumed. It is independent of the other agent's prefill norm reduction rewrite.

`tests/v2_norm_mixed_probe.py` is ready. It compares all 65,280 finite BF16 values through the old/new square; then captures the actual old/new FP32 square and normalized product before gamma, and compares complete BF16 norm outputs on five real canonical singleton rows and 32-/128-row prefixes. Every FP32 code, including signed zero, must match. `v2_norm_mixed_cpu.json` validates fixtures only; native exactness remains unverified.

## 2. Rank-five prefill state correction

`v2_prefill_rank5_state.py.snapshot` applies the already-discriminated decode head-group representation to the separately inlined prefill final-state scan. Prepared K becomes `[S,16,1,128,1]`; each state is viewed `[1,16,2,128,128]`, and delta `[1,16,2,1,128]`. Products return to `[1,32,128,128]` before the existing ordered contraction/add. This removes prefill K duplication; every value-head pair reads its original key head by outer singleton broadcast. The FP32 sequential state arithmetic, native chunk output, public BF16 cache boundary and prefill chunk plan remain unchanged. No query scale or output arithmetic is involved in this state-only loop.

Every added view preserves H/W and tile padding (`reshape_view/reshape.cpp:757–770`); BinaryNG outer broadcasting supports rank five (`binary_ng_device_operation.cpp:450–470`, `binary_ng_program_factory.cpp:45–53`). State/output products have the same payload and placement as before. The prepared K storage halves, but the 1024/128-step scan arithmetic is not reduced, so benefit needs measurement.

`tests/v2_prefill_rank5_probe.py` extracts the actual inlined scan statements from each immutable snapshot's AST rather than maintaining a second device implementation. It runs deterministic length-1 and length-32 scans constructed by repeating saved real operands at positions128/130/7703, explicitly not a canonical prefix accuracy test. It compares decayed state, memory product, update product, memory contraction and updated state at first/second/last steps, plus the complete final state. Intermediate references are bounded to those steps; host conversions occur after the full scan. CPU old/new final states are bit-exact (`v2_prefill_rank5_cpu.json`). Device probe remains required, followed by full decoder qualification if faster.

## 3. Token-height packed ordered prefill projection

`v2_prefill_token_height.py.snapshot` is the larger opportunity. Current prefill products place one token per outer axis and 1024 output coefficients in each tile. The candidate instead places 32 token rows in tile height and 32 output coefficients in tile width:

- Setup BF16 weights: pair-permuted checkpoint state weight `[2,1024,1,6208]` (ordinary tiled, H padded32).
- Runtime BF16 coefficients: original pair gather, untilize, transpose `[1,rows,2048]` to `[1,2048,rows]`, reshape `[2,1024,rows,1]`, tilize.
- Accurate BF16-by-BF16-to-FP32 `multiply` broadcasts one weight row and one coefficient column into `[2,1024,rows,6208]`.
- `fast_reduce_nc(dims=[0,1])` folds K1024 first, then two partials, retaining FP32 until the original final BF16 cast. Return `[1,rows,6208]` directly.

No K split, partial reassociation, native matmul, reduced precision, or reference change is introduced. `fast_reduce_nc.cpp` sorts descending dimensions; `device/kernels/reduce_nc.cpp:42–59` copies the first tile then adds subsequent tiles sequentially. The coefficient/weight lane remapping changes which independent dot products share a tile, not the arithmetic within any dot.

The path applies only when the existing projection microbatch has a row count divisible by32. Existing singleton paths, public projection K policy, and nonaligned microbatches remain unchanged. Thus S129 uses the new128-row microbatch plus the original one-row remainder; S33 retains its old path. The old decode/PF weights remain resident for these paths. Public state/chunk boundaries do not change.

Native row/column broadcasting is explicitly supported as `ROW_A_COL_B` (`binary_ng_device_operation.cpp:217–246`). On Blackhole FP32 destination, the factory deliberately disables the MOVB2D LLK path for this broadcast and selects software broadcast (`binary_ng_program_factory.cpp:1450–1468`), avoiding the source-documented conversion issue. The alternative's software-broadcast cost is a performance uncertainty to measure, not a reason to assume a win.

Physical allocation comparison for one128-row microbatch (ordinary32x32 tiles):

| Tensor | v2 | Candidate |
| --- | ---: | ---: |
| FP32 product | 7,516,192,768 B | 6,509,559,808 B |
| BF16 coefficient tiles | 536,870,912 B | 16,777,216 B |
| Extra resident BF16 weight view | 0 B | 813,694,976 B |

The product shrinks13.39% by removing960 output-padding columns. Coefficients shrink32-fold because each tile now contains32 meaningful token coefficients. Weight tiles can serve32 token rows rather than one; this may reduce repeated weight reads, but no exact DRAM-traffic claim is made without profiling.

Capacity check: extra weight is776MiB. At the existing maximum128-row microbatch, new product plus extra weight is184MiB smaller than the original product alone; coefficient shrink saves another496MiB. Thus the dominant product allocation does not present an obvious new nominal capacity blocker. The extra weight remains resident during other decoder stages, so this arithmetic is **not** an allocator or full-context proof. Both the isolated probe and a winning complete decoder must still demonstrate allocation feasibility, including later-stage peaks and the unchanged largest supported context. A blanket L1 product placement would not fit either geometry; products remain DRAM.

`tests/v2_prefill_token_height_probe.py` loads only real K/V/A/B checkpoint weights and calls the actual snapshot `linear` methods. It compares all6208 outputs at both FP32 and BF16 boundaries for S1/K512, S1/K1024, S2/S32/S33/S128/S129, including unchanged fallbacks. Complete traced timing includes pair gather, coefficient conversions/transpose/tilization, multiply, both reductions, cast and final concat. CPU mapping proof checks real weights on representative columns spanning every projection and tile boundary; both products and ordered reductions are exact (`v2_prefill_token_height_cpu.json`). This is a mapping proof, not an all-output device proof.

## Commands for the serialized hardware owner

```sh
python_env/bin/python -m models.autoports.qwen_qwen3_6_35b_a3b.tests.v2_norm_mixed_probe
python_env/bin/python -m models.autoports.qwen_qwen3_6_35b_a3b.tests.v2_prefill_rank5_probe
python_env/bin/python -m models.autoports.qwen_qwen3_6_35b_a3b.tests.v2_prefill_token_height_probe
```

Each script reserves physical chip0 and closes in `finally`. Norm/scan are bitwise discriminators only; the projection probe additionally performs warmed traces. Complete decoder benchmarking can use the existing `benchmark_fused_snapshot` entry with each snapshot independently. Keep current queue/device controls; no device invocation was executed here.

## Exhausted or rejected alternatives

The native GDN `output_final_state=False` flag does not eliminate final-state work: `chunk_gdn_device_operation.cpp:265–289` unconditionally creates output and state tensors; the wrapper's flag controls its optional returned state reshape. The final-state writer path remains active, as independently audited by the norm agent. There is no supported model-only output-only flag that removes this work.

Previously measured controls are not reproposed: packed A/B output-padding removal (exact but slower at every tested complete adapter), all-norm L1, transient weight L1 staging, full-row/flat/transposed LUT layouts, native depthwise convolution, native recurrent-FDR, and L2-boundary folding. Dividing the K1024 fold or changing the 16-then8 recurrence association breaks the ordered arithmetic contract; BF16 product storage and FMA residual/update replacements likewise change required boundaries. Smaller tile geometries remain blocked by BinaryNG/FastReduce contracts. The new token-height path uses standard tiles and changes independent-dot packing, so it is distinct from these rejected controls.

These are useful remaining experiments, not an optimization completion claim. Conservative device-session accounting remains below eight hours and measured useful-byte roofline below70%; elapsed wall time is not substituted for either gate.

## Subsequent parent-owned device results

The original source/CPU preparation above is retained; its pending-device language describes that preparation state. The parent subsequently produced `v2_norm_mixed_device.json` (**22/22** strict checks), `v2_prefill_rank5_device.json` (**24/24** strict checks), and `v2_prefill_token_height_probe.json` (all6208 FP32/BF16 outputs exact in all seven shape/K cases). Adjacent candidate provenance now binds these artifacts and their full-decoder benchmark hashes.

| Full decoder, S256/context512 | Prefill ms | Decode ms |
| --- | ---: | ---: |
| v2_adapter_baseline | 154.133333 | 1.406199 |
| v2_norm_mixed_operands | 154.644486 | 1.401315 |
| v2_prefill_rank5_state | 154.475637 | 1.405493 |
| v2_prefill_token_height | 256.019585 | 1.405748 |

All recorded prefill/contribution/eager/traced PCCs equal the baseline. Mixed norm is a small observed decode improvement; prefill rank5 has no short-prefill speedup and remains unselected pending the parent's longer-context comparison. The first token-height geometry is a clear performance regression: isolated complete-adapter S32 14.012849→25.536081 ms, S128 55.463183→106.274018 ms, S129 56.245227→106.910357 ms; its full decoder likewise regresses to256.019585 ms prefill. Exactness did not make this geometry fast. The adaptation below tests a distinct removable source cost before closing the layout family. No production selection is made here.

## Adaptation: fill the physical weight rows during setup

`v2_prefill_token_height_filled_rows.py.snapshot` retains the v2 base, untouched fallback paths, and the original aligned-token geometry's arithmetic. Its setup uploads pair-permuted BF16 weight values as:

```python
state_weight.reshape(2,1024,1,1,6208).expand(2,1024,1,32,6208).contiguous()
```

The weight's logical shape is `[2,1024,1,32,6208]`. Its physical bytes remain813,694,976: the old H1 operand already padded H to32, but stored zeros in31 rows. The new setup fills those rows with the repeated coefficients. Existing tiled runtime coefficients receive a metadata-only view `[2,1024,rows//32,32,1]`. Weight outer-axis broadcasting supplies each token block while native subtile broadcast becomes **COL_B only**, replacing ROW_A_COL_B. The product is `[2,1024,rows//32,32,6208]`; the same K1024-then2 reduction yields `[1,1,rows//32,32,6208]`, which views directly to `[1,rows,6208]`. All views preserve tiled H/W or merge tile-aligned H; no token permutation or extra runtime replication is required. Memory/product byte counts are unchanged from the first token-height candidate.

This is a real source-supported change: `binary_ng/device/kernels_ng/dataflow/reader_interleaved_row_col_mixed_bcast.cpp:113–143` reads and then software-expands the weight's first row **for every output width tile** when BCAST_LLK is disabled. At S128 there are1,589,248 output tiles, corresponding to about3.25GB of BF16 scalar L1 stores for these weight expansions, apart from the operand read. `reader_interleaved_col_bcast.cpp` reads already-filled weight tiles without that expansion, while retaining the coefficient-column fill once per tile row. Blackhole still uses the software COL_B route for BF16→FP32 (`binary_ng_program_factory.cpp:329–354`); this candidate does not rely on the source-documented problematic LLK conversion. It removes the repeated weight-row expansion, not all software broadcasting.

A second possible cost remains unchanged: `reader_reduce_nc.cpp` advances each K input tile by `inner_tile_size`. Original per-token geometry has stride7 tiles; token-height S32 has194 and S128 has776. This changes DRAM bank/page locality. For a hypothetical eight-bank simple interleave, stride776 stays on one bank per dot versus stride7 walking all banks. Actual tensor-accessor mapping and concurrent owners matter, so this is a source-supported locality concern, **not proof of the observed slowdown**. The filled-row adaptation isolates the explicit software expansion cost before changing reducer geometry or association.

The original probe is preserved byte-for-byte as `v2_prefill_token_height_probe_initial.py.snapshot`. The live diagnostic now accepts `--candidate-snapshot`, defaulting to the original token-height candidate. Setup recognizes the filled-row view and uploads the corresponding real weight geometry. The added CPU proof checks exact rank5 products and sequential reductions for all32 token lanes on real selected columns under both K512/K1024; `v2_prefill_token_height_filled_rows_cpu.json` passes. The device comparison still gates **all** output columns and both FP32/BF16 boundaries, and its complete trace contains every adapter operation. Original probe results and all original snapshots remain unchanged.

```sh
python_env/bin/python -m models.autoports.qwen_qwen3_6_35b_a3b.tests.v2_prefill_token_height_probe --candidate-snapshot models/autoports/qwen_qwen3_6_35b_a3b/doc/fused_decoder/endurance/v2_prefill_token_height_filled_rows.py.snapshot --output models/autoports/qwen_qwen3_6_35b_a3b/doc/fused_decoder/endurance/v2_prefill_token_height_filled_rows_probe.json
```

CPU proof was completed before notifying the parent to start its serial long benchmark; no TTNN/device commands were executed by this investigation.

## Filled-row adaptation: measured exactness and improvement

The parent-owned `v2_prefill_token_height_filled_rows_probe.json` reports **all6208 output codes exact at FP32 and BF16** across all seven shape/K cases, including unchanged fallback paths. First and last timed trace replays are also exact. The complete adapter includes every coefficient conversion, layout change, multiply, reduction and output operation.

| Rows | v2 adapter ms | Filled-row adapter ms | Latency reduction |
| --- | ---: | ---: | ---: |
| 32 | 14.006013 | 11.066432 | 20.99% |
| 128 | 55.475110 | 48.441761 | 12.68% |
| 129 | 56.233308 | 49.144565 | 12.61% |

The complete decoder benchmark `v2_prefill_token_height_filled_rows_s256.json` is now available: **139.468036 ms prefill / 1.406044 ms decode**, versus the matched v2 baseline **154.133333 / 1.406199 ms**. Prefill improves by **14.665297 ms (9.51%)**; the decode path is unchanged and its median difference is negligible. All reported prefill/contribution/eager/traced PCCs match the baseline exactly. The source-bound hashes and measurements are recorded in the adaptation's provenance.

This earns advancement of the filled-row adaptation to the parent's combined candidate and subsequent acceptance gates. It does not rehabilitate the original unfilled ROW_A_COL_B geometry, which remains a measured regression. The observed improvement is consistent with eliminating the explicit per-output-tile weight-row fill; no uncollected operator profile is claimed to establish the full causal attribution. Added resident weight remains776MiB, so largest-context and combined-path allocation qualification still apply. No production selection or stop-rule completion follows from these short diagnostic benchmarks. No additional experiments were prepared for this update, and no hardware was executed by this investigation.
