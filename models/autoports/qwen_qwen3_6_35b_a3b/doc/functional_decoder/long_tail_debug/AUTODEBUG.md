# AutoDebug: long-tail decode repeatedly stores its recurrence in BF16

**Leading diagnosis:** the model's grid-1/K32 decode configuration runs 8,192 serial key-chunk reductions per tail query through BF16 score, denominator, and output-accumulator buffers. `fp32_dest_acc_en=True` does not change those buffer formats. The passing chunked-prefill path uses FP32 score and denominator buffers. This is a concrete precision discrepancy and the strongest source-supported explanation for the long-tail error. It is **not a hardware-confirmed attribution of the measured PCC**.

I found no ordinary position/page-index overflow above 65,536, incorrect final-page mask, or unmatched circular-buffer count in the selected branch. The exact 65,536-tile K/V totals are handled by intentional modulo counters. No source changes, device operations, hardware reproductions, resets, or launcher invocations were performed. This report is the sole file written by this investigation.

## Evidence and its limits

The recovered log, `models/autoports/qwen_qwen3_6_35b_a3b/doc/functional_decoder/logs/full_long_recovered.log`, directly records:

| Measurement | PCC |
| --- | ---: |
| Prefill, logical length 262143, whole output | 0.9992695572285034 |
| Row 262110 | 0.9995316096968718 |
| Row 262142 | 0.9459396225707403 |
| Other sampled rows 0, 31, 255, 256, 512 | 0.999895 or higher |

The run fails the existing aggregate-and-selected-row requirement of PCC >= 0.995 at `tests/long_context.py:120-134`. It then exits: this log supplies neither a 262144-prefill result nor a final-position traced-decode result. It measures **one tail row**, not all 31. It does not establish the first bad position, a discontinuity at 65,536, or that every other row passes. Aggregate PCC alone cannot locate or characterize the error.

At initial inspection, the source SHA256 values for `tt/functional_decoder.py`, `tests/long_context.py`, and `tests/reference.py` exactly matched the recovered-run sidecar `doc/functional_decoder/full_long.json.provenance.json`. Committed HEAD is `a407e7d3ec6a00693e1be369d8a395f63c9c2c62`, also matching that sidecar; its decoder SHA256 is `b4fadd1c273c777764d7f533ba8838672bcc3a65dd0fb42880e27c95904fb362`. During report review, concurrent external edits changed the worktree decoder's grid to (8,8), retained K32, and added four comment lines. This investigation did not make those edits. **Decoder line references and the original failure analysis below describe the committed/recovered grid-(1,1) snapshot**, not that later worktree configuration.

The sidecar records Blackhole P300c chip 0, Torch 2.11.0+cpu, Transformers 5.12.1, and TTNN `0.1.dev1+ga464944`. Python-source correspondence for the original failure is verified; the version string alone does not prove which native binary/kernel build executed. Native findings below describe the prepared source tree.

The log says the HF oracle cache identity was verified. The runner checks exact config, source/version/weight identity and expected shape/dtype (`tests/long_context.py:32-53`). HF uses explicit absolute-position causal masks (`tests/reference.py:53-61`), so reusing the 262144-row oracle for its 262143-row prefix is logically valid.

All model-relative paths below are under `models/autoports/qwen_qwen3_6_35b_a3b`. For native references:

- `D` = `ttnn/cpp/ttnn/operations/transformer/sdpa_decode/device`
- `P` = `ttnn/cpp/ttnn/operations/transformer/sdpa/device/sdpa_program_factory.cpp`
- `C` = `ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/compute/compute_common.hpp`
- `F` = `ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/dataflow/dataflow_common.hpp`

## Exact execution path

`prepare_prefill` distinguishes logical and physical lengths (`tt/functional_decoder.py:326-344`). For length 262143, start 0, and default chunk size 256, its actual plan is:

1. 1023 chunks of 256, covering positions 0 through 261887.
2. One chunk of 224, covering 261888 through 262111.
3. 31 individual tokens at 262112 through 262142.

`prefill_forward` selects decode exactly when `n == 1` (`:358-365`). Row 262110 is inside the last aligned chunk; its own index need not be divisible by 32. Row 262142 uses decode. Each tail call begins a new attention reduction over the full accumulated history; the statistics do **not** recur across the 31 calls.

The local chain is input RMSNorm -> BF16 Q/gate/K/V projections -> per-head Q/K RMSNorm -> 64-dimensional partial RoPE -> BF16 paged K/V write -> attention -> sigmoid gate -> output projection -> residual -> post-attention RMSNorm -> MoE -> residual (`tt/functional_decoder.py:98-110,255-324`). Q has 16 heads; K/V have two; head dimension is 256; attention scale is exactly 1/16. Cache shape is `[8192, 2, 32, 256]`, with an unsharded INT32 table containing 8192 distinct permuted physical pages (`:204-230`; `tests/long_context.py:101-106`). Absolute positions are uploaded as INT32 (`:232-253`).

For a tail token, the model converts Q/K/V to a height shard of `[32,256]` on core (0,0), updates the cache with `paged_update_cache`, and calls `paged_scaled_dot_product_attention_decode` with grid (1,1), Q32/K32, HiFi4, FP32 destination accumulation and DRAM output (`tt/functional_decoder.py:53-59,269-295`). There is no operand swap or precision promotion hidden in this wrapper: the supplied K32 and compute configuration are forwarded by `sdpa_decode/sdpa_decode.cpp:133-172`.

The selected native kernels are `D/kernels/dataflow/reader_decode_all.cpp`, `D/kernels/dataflow/writer_decode_all.cpp`, and `D/kernels/compute/sdpa_flash_decode.cpp` (`D/sdpa_decode_program_factory.cpp:918-950`). No MLA, speculative multi-position mode, sliding window, explicit mask tensor, or dynamic chunking is selected.

| Lowered value | Passing short decode at position 512 | Last long tail at position 262142 |
| --- | ---: | ---: |
| Live tokens | 513 | 262143 |
| Scanned tokens, rounded to K32 | 544 | 262144 |
| PNHt / Sq_chunk_t | 1 / 1 | 1 / 1 |
| DHt / vDHt / Sk_chunk_t | 8 / 8 / 1 | 8 / 8 / 1 |
| Cores per KV head | 1 | 1 |
| KV heads processed serially on the sole core | 2 | 2 |
| Key chunks per head | 17 | 8192 |
| Local softmax corrections per head | 16 | 8191 |
| Cross-core tree rounds | 0 | 0 |
| Last virtual page / valid columns | 16 / 1 | 8191 / 31 |

Factory arithmetic is at `D/sdpa_decode_program_factory.cpp:122-171,201-216`; all kernels use `D/kernels/rt_args_common.hpp:35-107` to derive the same workload. Every long tail position 262112..262142 scans 8192 chunks; only the final mask's valid columns change.

The Python calls enqueue work. The relevant completion boundary is the output readback before PCC: `tests/long_context.py:119-121` -> `ttnn/ttnn/operations/core.py:464-467` -> `from_device`, whose default is blocking (`ttnn/cpp/ttnn/operations/core/core.hpp:34-35`). Thus this is a completed readback mismatch, not an inference from reaching the end of an enqueue-only function. Trace capture/replay occurs after the failing assertion and was not reached in this recovered run.

## Why decode precision is the leading diagnosis

### The running state is BF16 despite FP32 destination accumulation

`D/sdpa_decode_program_factory.cpp:507-508` unconditionally sets both `im_df` and `stats_df` to `Float16_b`. CB allocations at `:694-710` therefore store:

| CB | Value | Storage |
| --- | --- | --- |
| c24 | QK scores, then exponential probabilities | BF16 |
| c25 | Current probability-times-V output | BF16 |
| c26 | Running output numerator | BF16 |
| c27/c28 | Current/previous maximum | BF16 |
| c29/c30 | Current/previous denominator | BF16 |
| c31 | Maximum correction factor | BF16 |

FP32 destination accumulation is separately passed to the compute descriptor at `:946-950`. It improves calculations in destination registers; packing into a BF16 CB still rounds away information before the next chunk reloads it. This is not a claim that the API promises FP32 storage whenever that flag is enabled.

Ignoring notation for tile layouts, let `R` mean a BF16 store. The decode loop implements the algebraically valid online-softmax recurrence, but with approximately these storage boundaries:

```text
S_j = R(Q @ K_j^T)
m_j = R(max(m_previous, max(S_j)))
P_j = R(exp_approx((S_j - m_j) / 16))
alpha_j = R(exp((m_previous - m_j) / 16))
l_j = R(R(sum(P_j)) + R(alpha_j * l_previous))
U_j = R(R(P_j @ V_j) + R(alpha_j * U_previous))
output = R(U_last * R(1 / l_last))
```

The stages are explicit in `D/kernels/compute/sdpa_flash_decode.cpp:415-430,487-574,687-714`. Shared helpers perform the intermediate packs (`C:317-339,412-425,487-502,636-662,697-718,739-765`). The actual moves at the end of every chunk leave the latest max/sum in the **previous** buffers. The contrary comment at decode compute `:656` is stale; the actual final use of `cb_prev_sum` is correct.

At 8192 chunks, small positive denominator increments can disappear against an already large BF16 sum. The numerator also incurs repeated rounding of signed, data-dependent values; its error need not cancel the denominator error. Final gating, projection, residual addition and MoE can transform scale/direction error into a decoder-row PCC failure. A scalar rescaling of an isolated vector alone would not lower its PCC; the diagnosis does not rely on that incorrect argument.

A cheap host-only arithmetic illustration, **not a model reproduction**, used BF16 round-to-nearest-even after every addition with an unchanged maximum and exact unit weights:

| Updates `l <- BF16(l + 32)` | Emulated sum | Exact sum |
| ---: | ---: | ---: |
| 17 | 544 | 544 |
| 256 | 8192 | 8192 |
| 512 | 8192 | 16384 |
| 8192 | 8192 | 262144 |

At 8192 the BF16 spacing is 64; adding 32 reaches a tie. This establishes the representational failure mode, not the model's actual denominator, exact hardware rounding behavior, or measured error magnitude. It predicts sensitivity to serial reduction depth and attention distribution, rather than an intrinsic integer boundary at 65536.

### The passing full-chunk path does not have the same recurrence precision

For this model's FP32 destination setting, `P:177-178,1750-1754,1858-1865` allocate **FP32 QK/probability and denominator buffers**. Maxima and output numerator remain BF16. Full prefill accumulates partial denominator vectors and performs their final row reduction outside the key loop (`C:1947-1958,1992-2021`); its numerator correction uses L1 accumulation (`C:2002-2007`). Decode instead explicitly row-reduces each BF16 probability chunk and stores the scalar sum in BF16.

Consequently the passing aligned row is consistent with a decode-specific precision problem even though both paths use K32 and `self.compute`. It does not prove full prefill has no precision limitations, nor does it quantify numerator versus denominator contributions. FP32 storage also does not mean lossless FP32 recurrence arithmetic: full prefill's ordinary FPU multiply/add helpers (`C:1997,2000,643-653,676-688`) reload through source registers, whose FP32 conversion uses TF32 on Blackhole (`tt_metal/tt-llk/tt_llk_blackhole/common/inc/cunpack_common.h:280-287,315-317`). The grid difference by itself does not mean full prefill splits each query across 64 workers; the confirmed contrast here is its buffer formats and arithmetic schedule.

### Selected Blackhole math paths were checked

The decode factory defaults `exp_approx_mode` to true independently of `math_approx_mode=False` (`D/sdpa_decode_program_factory.cpp:219-223`). With 16 BF16 Q heads, the factory selects 16x32 half tiles (`:513-551`), and compute selects `VectorMode::R` (`D/kernels/compute/sdpa_flash_decode.cpp:301-304`). `C:369-407` forces approximate score exponentiation for this vector mode even if `exp_approx_mode=False` is supplied. Changing that option alone therefore cannot promise accurate score exponentials.

The maximum-correction exponential follows a different helper: on Blackhole, `calculate_exponential_first_column<true,...>` calls the scalar accurate exponential (`tt_metal/hw/ckernels/blackhole/metal/llk_api/experimental/llk_sfpu/ckernel_sfpu_sdpa.h:193-203`). The FP32 scalar implementation uses literal constants (`.../ckernel_sfpu_exp.h:512-514,668-703`). Final reciprocal explicitly initializes its constants (`C:324`) and takes the FP32 two-iteration branch (`.../ckernel_sfpu_sdpa.h:28-38`). No concrete mode-lifetime or programmable-constant violation was found on these selected paths. The scale 1/16 is exactly BF16-representable, so truncating the correction helper's scale to BF16 is harmless here.

## Indexing, masking and buffer audit

### No position or physical tile-ID truncation found

The current position travels through a `uint32_t` reader and separate one-entry mailboxes for writer c8 and compute c15 (`D/kernels/dataflow/reader_decode_all.cpp:124-170`; writer `:122-147`; compute `:172-184`). The shared helper's signed `int` parameter also safely holds 262142. Fixed `Sk_chunk_t=1` bypasses the dynamic helper's `uint8_t` power-of-two calculation.

The unsharded table selects a **uint32_t** lookup (`D/kernels/dataflow/reader_decode_all.cpp:270-294`; `D/kernels/dataflow/dataflow_common.hpp:755-763,837-845`). For this cache, `F:53-68` reduces to:

```text
physical_tile = page_table[virtual_page] * 16 + kv_head * 8 + hidden_tile
```

Virtual and physical pages are 0..8191; the maximum physical tile ID is 131071. This exceeds 16 bits but all the mapping operands/result and the NoC page ID remain uint32_t (`tt_metal/hw/inc/api/tensor/noc_traits.h:137-148`; `api/tensor/tensor_accessor.h:649-652`). The final page update writes offsets 0..30 and remains inside the allocated 32-row page.

### Final-page mask matches the concrete boundary

The writer creates one retained causal mask (`D/kernels/dataflow/writer_decode_all.cpp:254-256`). For position 262142, remainder 30 means only column 31 is masked; for position 512, remainder 0 means columns 1..31 are masked (`D/kernels/dataflow/dataflow_common.hpp:47-100,251-304`). The single reducer applies it on each serial head's final chunk without consuming it (`D/kernels/compute/sdpa_flash_decode.cpp:129-130,449-453`). Fixed chunking disables the dynamic mask-fusion branch. No off-by-one discrepancy was found.

### The 65536-tile wrap has a matching consumer

K/V CBs each hold 16 tiles and transfer eight per chunk (`D/sdpa_decode_program_factory.cpp:480-482,639-640`). With 8192 chunks, each transfers exactly 65536 tiles per head. Inspection and an independent host-only event/count model give these paired total pushes/pops per head, including normalization and writer drain:

| CB | Formula for N chunks | N=17 | N=8192 |
| --- | ---: | ---: | ---: |
| c1, c2 | 8N | 136 | 65536 |
| c24 | 2N+1 | 35 | 16385 |
| c25 | 8(N-1) | 128 | 65528 |
| c26 | 16N | 272 | 131072 |
| c27, c28 | N | 17 | 8192 |
| c29 | 2N-1 | 33 | 16383 |
| c30 | 2N | 34 | 16384 |
| c31 | N-1 | 16 | 8191 |
| c20 | 8 | 8 | 8 |

The recurrence/finalization references are the compute loop cited above and writer `:514-516`; matmul consumption is at `C:1106,1155`. Working CBs drain after each head. Q remains available across both heads and is popped once at compute `:733-734`; constants/mask/page-table buffers intentionally remain resident. Reader prefetch can queue the next head's K/V without changing their stream order.

Local received/acknowledged counters are uint16_t (`tt_metal/hw/inc/internal/circular_buffer_interface.h:94-100`). Both reserve and consume explicitly perform modulo occupancy arithmetic (`tt_metal/hw/inc/api/dataflow/dataflow_api.h:406-425`; `tt_metal/hw/ckernels/blackhole/metal/llk_io/llk_io_unpack.h:18-33`; `llk_io_pack.h:33-44`). For logical received=65536 and acknowledged=65528, the consumer sees `(0 - 65528) mod 65536 = 8` live tiles; reserve sees eight free slots. No wait requests 65536 live tiles. A host-only check of every eight-aligned received count through 262160, with legal K/V occupancies 0/8/16, confirmed these formulas. This is a count/capacity audit, not execution of the asynchronous kernels or proof against an unobserved device/compiler fault.

## Candidate ranking after adversarial review

| Candidate | Source evidence and match to observations | Missing evidence / smallest falsifier | Rank |
| --- | --- | --- | --- |
| Excessively deep BF16 decode recurrence | Confirmed BF16 state, 8192 serial chunks versus <=17 in short cases; passing full path has FP32 scores/sums | Compare the same attention inputs before MoE across shorter/deeper reductions; attention could already pass | Leading; runtime attribution remains unverified |
| Position/page overflow above 65536 | No selected-path narrowing; maximum address fits uint32_t | Physical-page-coded V and paired page permutations at boundary positions | Not supported by audited source |
| 65536 cumulative CB tiles | Real boundary, but paired modulo arithmetic and balanced counts | Device/compiler behavior beyond the source contract remains untested | Not supported as a source bug |
| Bad causal tail/paged cache update | Correct absolute INT32 index, page/row bounds and final mask | Poison future entries; compare unpaged/readback KV against reference | No source contradiction found |
| Half-tile approximate score exp | Confirmed selected behavior, independent of compute math_approx_mode | Hold recurrence length fixed and compare an accurate full-tile implementation | Possible contributor; not isolated |
| MoE route disagreement | Different finite-precision logits can change the eighth/ninth expert boundary | Inspect pre-MoE output and actual top8 IDs/weights for row262142 | Secondary explanation/amplifier |

The evidence does not justify a claim that 0.94594 is inevitable from BF16, that the final tail token is uniquely wrong, or that no model-local downstream contribution exists.

## Separately observed concurrent probe artifacts

These artifacts appeared in the shared worktree during report review. They were **not created or executed by this investigator**, are separate from the recovered log, and were inspected as provisional reported results together with their scripts:

| Artifact / configuration | Reported PCC | Other reported metric |
| --- | ---: | --- |
| `sdpa_precision_probe.json`, grid1/K32 | 0.7700665 | Relative RMS error 48.0465 |
| Same probe, grid8/K32 | 0.9978367 | Relative RMS error 0.18078; norm ratio 1.16626 |
| Same probe, grid1/K256 | 0.9876653 | Relative RMS error 1.6520 |
| Same probe, grid8/K256 | 0.9997216 | Relative RMS error 0.02363 |
| `real_tail_probe.json`, grid1/K32 | 0.8429102 | Context262144, position262143 |
| Same real-weight probe, grid8/K32 | 0.9998276 | Same probe inputs |
| Same real-weight probe, grid8/K256 | 0.9999453 | Same probe inputs |

`tests/sdpa_precision_probe.py:17-25,34-74` prepares identical BF16 random operands, an identity page table and position262142, compares to FP32 host attention, and varies the grid/K chunk. `tests/real_tail_probe.py:23-38,51-66` prepares HF K/V, uses identity page order, and overrides the SDPA configuration for each final-token decoder call. That script tests **position262143**, not the original failing262142, and bypasses the oracle-identity check when loading expected output. Its metrics are recorded without acceptance assertions. Neither artifact establishes that the original TT-filled, shuffled-cache prefill or its traced continuation now passes.

The synthetic grid1/K256 versus grid1/K32 contrast holds cores/head and serial-head ownership fixed while shortening the recurrence, supporting a numerical contribution. The grid-only contrast also changes two serial KV heads/core into one head/core and adds tree reduction, so it is not an isolated storage-precision experiment. The grid8/K32 norm error illustrates why a high PCC alone should not be described as accurate magnitudes. These results strengthen the reduction-depth hypothesis and support trying the smaller grid-only intervention first; they do not isolate a single arithmetic stage or close the original acceptance gate.

## Suggested model-local intervention and decisive tests

**First try the smallest configuration change at `tt/functional_decoder.py:290-292`:** retain the current one-core Q shard, BF16 cache, Q32/K32 and existing compute configuration; widen decode to grid (8,8), with `max_cores_per_head_batch=16` documented or explicit. This reduces 8192 local chunks to 512 plus four tree rounds, preserves arbitrary existing cache capacities, and is supported by the separately observed probe contrast above. It has already appeared as a concurrent external worktree edit; this report does not implement it or certify its full-context validation.

If its magnitude/row checks require further improvement, a bounded-reduction experiment can additionally use K512 for this 262144-token allocated capacity. This preserves every context token and changes reduction depth/order, rather than the mathematical attention domain or acceptance threshold.

With that explicit configuration, factory arithmetic gives 32 active cores, 16 per KV head, 512 total key chunks, 32 local chunks/core, and four tree rounds. Grid8/K32/max16 alone gives 512 local chunks; grid8/K128/max16 gives 128. The default max-cores value is **16**, not the grid's core count (`ttnn/cpp/ttnn/operations/transformer/sdpa_config.hpp:15-21`). Report or set it explicitly when interpreting experiments.

The existing Q shard is supported: other workers remotely read Q from output core (0,0), using the sharded-Q branch (`D/sdpa_decode_program_factory.cpp:342-355`; `D/kernels/dataflow/dataflow_common.hpp:575-617`). These are unicast reads. K512 can span 16 shuffled 32-token pages because each K/V tile row resolves its own page (`:755-763,837-845`). Moving Q to DRAM is unnecessary for this candidate.

**Preserve arbitrary cache capacities.** Workload rounds the live length up to K size and the reader fetches every rounded row. Unconditional K512 can over-read a small or non-512-aligned allocation accepted by `new_kv_state`. A safe simple model-local policy is K512 only when the logical page-table length is divisible by 16, otherwise K32. A more general policy chooses the largest power of two <=512 that divides `page_table.shape[-1] * 32`. This uses setup metadata and requires no device readback. Do not silently extend attention bounds, write padded live state, or reduce the advertised context.

This configuration is **a candidate repair to validate**, not a demonstrated pass. Use these controls before declaring it fixed:

1. **Locate the first divergence.** For the identical final query/state, compare Q after norm/RoPE, logical cached K/V, raw attention output, gated/projected attention, post-attention residual, router IDs/weights and final decoder output. Compare attention against a host FP32/FP64 reference built from the *same BF16 operands* to isolate kernel error, then against HF to include projection error. Record PCC together with absolute/relative error and norms; a scale error can be invisible to attention-only PCC.
2. **Hold operands, position and page order fixed; vary reduction depth.** Compare grid1/K32, grid1/K128, grid8/K32/max16, grid8/K128/max16 and grid8/K512/max16 on the long allocation. Also test grid(2,1)/K32/max1: it gives one core per KV head with one head/core, retaining 8192 local chunks, and separates the two-head lifecycle from reduction depth. Keep exponent settings fixed initially. Then separately examine `exp_approx_mode`; remember the half-tile score path remains approximate when it is false. A large improvement before gating/MoE supports recurrence precision without conflating cache or router changes. No improvement weakens the leading diagnosis.
3. **Use a synthetic test that cannot hide denominator loss.** Set Q=K=0, V nonzero only in the first 32 valid positions, and all remaining V zero. Correct output is the prefix contribution divided by `cur_pos+1` (for V=1 in that prefix, 32/(cur_pos+1)). Compare norms/values, not just PCC. Constant V over the whole cache is a poor sole control because numerator and denominator errors can cancel. Keep a large allocation and test positions 512, 8191, 8192, 65534, 65535, 65536, 131071, 262111, 262112, 262142 and 262143.
4. **Separate addressing/masking from arithmetic.** Repeat with identity versus correctly paired shuffled page layouts; use distinguishable head/page/hidden-coordinate V patterns. Poison future final-page entries, especially position262143 while decoding262142, and verify no contribution. This probes the proposed index boundary using known correspondence rather than inferred output samples.
5. **Rerun the original real-weight contract unchanged.** Require PCC >=0.995 for 262143 and 262144 prefill and traced decode at262143, with the original full context. Inspect all 31 tail rows and neighboring rows, not only the final sample; record attention diagnostics before MoE. Also rerun short/continuation/page-permutation cases and capacity33/257/513 regressions for the chunk-size policy. Complete trace replay/readback and determinism checks; enqueue completion alone is insufficient.

If a bounded stock reduction still fails, use a **model-local FP32 recurrence implementation** rather than treating HiFi4 or an FP32 Q cast as sufficient. Decode validation currently rejects FP32 inputs (`D/sdpa_decode_device_operation.cpp:39-46`), and its buffer formats are hard-coded regardless.

There is an inspected in-tree precedent at `models/demos/k2_horizon_7b_qb2/tt/accurate_attention`: `binding.cpp:346-423` promotes half tiles, changes accumulator/exchange CB formats and sizes, and uses `UnpackToDestFp32`; `build.py:89-166` replaces per-chunk/tree recurrence, row sum, moves and output scaling with the matching helpers in `accurate_decode.hpp`. It is a design reference, not evidence of a Qwen pass. Adapt the relevant code locally and validate this model's shapes/architecture; direct import carries a hard-coded (11,10) grid (`binding.cpp:300-305`). Merely changing CB formats is incomplete: ordinary FPU source reloads can truncate FP32, partial-tile support and tree transfer sizes also need to match, and helpers must agree on formats.

## Other Potential Issues / test gaps

- **Router precision differs from HF.** TT requests FP32 router logits before top8 (`tt/functional_decoder.py:178-180`). Installed HF computes checkpoint-typed `F.linear`, then FP32 softmax/top8/renormalization (`python_env/lib/python3.12/site-packages/transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py:784-790`); the reference loads weights unchanged (`tests/reference.py:39-49`). Near ties can select different experts or amplify an attention difference. Top8 on logits versus softmax followed by renormalization is mathematically equivalent in exact arithmetic, so that ordering alone is not an algorithmic omission. There is no measured expert correspondence for the failing row; keep this secondary until the stagewise comparison.
- **Existing tests do not establish this numerical corner.** `tests/ttnn/unit_tests/operations/sdpa/test_sdpa_decode.py:35-73` checks FP32 destination/half-sync merging only at positions31/32. Its ordinary matrix uses different grids/shapes/cache dtypes (`:77-100`). The shared test helper selects K512 for long contexts (`tests/ttnn/unit_tests/operations/sdpa/sdpa_test_utils.py:48-60`), unlike the model's fixed K32. These are useful sibling-contract clues, not proof that grid1/K32 is unsupported or that wider-grid/K512 passes this model.
- **No independent model-logic error was found** in Q/gate splitting, head grouping, learned RMSNorm +1, partial RoPE64/theta1e7, text-only equal mRoPE axes, scale1/16 or causal history. This audit does not replace comparison of actual intermediate tensors.

## Inspection checks and report review

Three isolated source-only investigations separately audited model/reference logic, numerical helpers, and indexing/CB contracts. Their claims were revisited against the selected callers, factories, kernels and Blackhole helpers. Independently executed cheap host-only checks verified Python SHA256 correspondence, the exact prefill plan, workload/page/tile counts, a two-head CB event/count ledger for N=17 and N=8192, modulo occupancy arithmetic through262160 transferred tiles, and the illustrative BF16 sum. None imported TTNN or accessed a device.

Review demoted attractive but unproven explanations: 16-bit position overflow, a missing last-column mask, cumulative-CB overflow, an exp/reciprocal init leak, a single-option accurate-exp fix, and router differences as an established root cause. The remaining headline is the confirmed recurrence-precision discrepancy, with its model-level causal attribution explicitly awaiting the controlled attention-only and unchanged full-context checks above.
