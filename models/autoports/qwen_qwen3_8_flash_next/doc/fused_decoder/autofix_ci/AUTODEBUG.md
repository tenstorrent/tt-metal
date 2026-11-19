# AutoDebug: fused GDN synthetic decode regression

## Finding

**The completed owner controls isolate the regression to the native GDN convolution path. That replacement changes the model's numerical contract in three coupled ways: accumulator precision, the convolution-output BF16 rounding boundary, and SiLU precision.** Grouped accurate projections and the direct recurrence reshapes can remain enabled: their combination passes the exact failed case when native convolution is disabled.

The strongest repair direction is an opt-in native convolution mode that accumulates in FP32, explicitly rounds the convolution result to BF16 before activation, evaluates SiLU accurately in FP32, and rounds the final output to BF16. Preserve the fused implementation and the unchanged 0.995 model threshold. Merely switching `fp32_dest_acc_en=True` does not restore the missing boundary and is not an adequate repair recommendation.

Confidence is **high for localization to the native-convolution branch and for the source-level numerical discrepancies**, and **moderate for this proposed repair resolving the final model PCC failure**. No saved intermediate tensors identify which discrepancy first changes the decisive output. A downstream expert-selection change is plausible but unproven.

This investigation performed source/evidence inspection and one scalar, CPU-only arithmetic check. It did not run TTNN, hardware tests, a model reproduction, device resets, or the AutoDebug launcher. No runtime source was edited.

## Evidence and revision ledger

Repository: `/home/ttuser/qwen38-flash/tt-metal`; HEAD `70dc6f5dd24298741e2f108a281160677e986155`.

Abbreviations used for source references:

- `M/`: `models/autoports/qwen_qwen3_8_flash_next/`.
- `C/`: `ttnn/cpp/ttnn/operations/experimental/kda/qkv_causal_conv1d_silu/`.
- `BH/`: `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/`.
- `HF`: `python_env/lib/python3.12/site-packages/transformers/models/qwen4_exp/modeling_qwen4_exp.py`.

The current `M/tt/fused_decoder.py` SHA256 is `3088b9b9fc1b32cd28b88656e566dda3a3dbfe23c581689c72f5a5a6342df23e`. It matches the original synthetic provenance and every control below. The shared synthetic tests, fixture, functional decoder, decoder components, and recurrence source also match the original failure provenance. HEAD alone would be insufficient because the fused source is untracked; these file hashes establish correspondence.

All artifacts below are under `M/doc/fused_decoder/`. The owner produced the controls during this investigation; I read their completed XMLs and provenance files. Each provenance is `status=finished`, with exit code matching the result.

| Artifact | Native conv | Direct recurrence reshapes | Decode step 1 PCC | Result |
|---|---:|---:|---:|---|
| `ci_control_functional_v1.xml` | No | No | 0.9999936819076538 | Pass |
| `ci_control_best_group64_v1.xml` | No | No | 0.9999937415122986 | Pass |
| `ci_control_best_group64_layout_v1.xml` | No | Yes | 0.9999937415122986 | Pass |
| `ci_control_best_group64_conv_v1.xml` | Yes | No | **0.994683563709259** | Fail |
| `final_v1_synthetic.xml`, `gdn-1` | Yes | Yes | **0.994683563709259** | Fail |

The two fused conv-off controls match every reported PCC. The conv-only control reproduces the final default's prefill PCC, decode-0 PCC, and failure PCC exactly. Equal PCCs do not establish bitwise tensor equality. Effective policies come from `M/tests/run_fused_probe.py:20-97`; `M/tests/probe_fused_ci_policy.py:22-32` injects them into the unchanged fused fixture. The three fused controls retain grouped accurate projections, packed GDN/shared projections, fused expert/shared activations, and fused expert reduction. Additional QSA defaults in the final constructor do not execute for layer 0.

The original command ran all eight cases: GDN and indexed QSA, each with sequence lengths 1/31/33/65. Seven passed. For `gdn-1`, prefill PCC was `0.9999908208847046`, decode 0 was `0.9999849200248718`, and decode 1/repeat 0 failed. Later decode/reset checks were not reached. GDN 31/33/65 decode PCCs are approximately 0.99995. This is a sharp drop in the measured whole-decoder output, not observed gradual deterioration. Different lengths use different random inputs, so the matrix does **not** prove a length-specific kernel bug.

The failing fixture uses layer 0, seed 1901, BF16 hidden input `[1,4,10240]`, normal standard deviation 0.02, and identical synthesized weights for HF and TTNN (`M/tests/test_synthetic_decoder.py:92-118`). `gdn-1` means a one-token prompt, not batch size one as a distinguishing parameter: all these cases use one request. Failure is the third token overall, the second decoded token. The unchanged comparison flattens both outputs after FP32 conversion and requires Pearson correlation >=0.995 (`:36-40`).

Warmup explicitly synchronizes; replay uses `blocking=True` and synchronous `to_torch`; both convolution and recurrent buffers are restored before capture/repeated runs (`:125-154`). Thus this assertion observes a completed decode result. The failed case itself does not establish repeat determinism because it aborts during the first repetition.

Same-source `final_v1_layer{0,1,3}_hf` short results pass. Existing live continuation0/1 results use older fused SHA `a423836fd21a27912b0ebb986cf445a0d3008c1658cfdf1ef518daba383cc0ae` and `best_group64_conv`, with direct recurrence layout disabled. They support that earlier convolution variant, not exact-source live validation of the final default. Layer 1's live report retains its documented fresh-PLE backend qualification (`literal_hf_strict_passed=false`); resumed/decode checks remain strict. None of these distinctions justify discarding the successful real-weight path.

## The numerical contract that changed

### Functional/HF boundary

For channel c and output token r, the intended four-tap expression is

`z[r,c] = sum(j=0..3, x[r-3+j,c] * w[j,c])`.

The functional implementation explicitly computes FP32 products, pairwise FP32 sums, casts z to BF16, promotes that rounded z to FP32 for SiLU, then casts the activated result to BF16 (`M/tt/fused_decoder.py:653-666`; the inherited functional GDN has the same sequence).

HF's selected CPU fallback similarly obtains a BF16 `F.conv1d` output before applying SiLU, both for cached update and prefill (`HF:213-219,233-242`). This establishes the intermediate dtype boundary, not HF's exact internal accumulation order. The original synthetic log records fallback use at `M/doc/fused_decoder/logs/final_v1_synthetic.log:31-35`. This matters independently of whether a fused mathematical expression is more accurate relative to infinite precision: the BF16 intermediate is part of the reference execution being preserved.

### Native boundary

`M/tt/fused_decoder.py:607-608,630-647,783-789` supplies BF16 inputs/taps, channel chunk 128, HiFi4, `math_approx_mode=False`, `fp32_dest_acc_en=False`, `packer_l1_acc=False`. The public wrapper preserves those supplied settings and input ordering; it only tilizes input if necessary (`C/qkv_causal_conv1d_silu.cpp:15-51,73-95`). Outputs default to interleaved DRAM. The factory forwards the compute configuration at `C/device/qkv_causal_conv1d_silu_program_factory.cpp:467-474`.

The compute kernel builds diagonal BF16 matrices from the taps, performs four matmuls into the same destination tile, applies pack-thread SiLU directly to that destination, then packs the output (`C/device/kernels/compute/qkv_causal_conv1d_silu.cpp:44-77`). The false mode exposes a low-precision destination result to SiLU, with rounding during accumulation; the true mode accumulates in FP32 but has **no explicit BF16 rounding of that completed FP32 sum before SiLU**. Neither setting implements the functional sequence. In particular, the failing false path should not be described simply as retaining excess preactivation precision: its early rounding is itself a discrepancy.

There is a second, independently visible difference: `silu_tile_pack` passes `DST_ACCUM_MODE` to `calculate_silu` (`tt_metal/hw/inc/api/compute/compute_kernel_api.h:738-749`). On Blackhole, the false arm chooses `exp_21f` and one reciprocal refinement, whereas the true arm chooses the accurate FP32 exponential and two refinements (`BH/ckernel_sfpu_sigmoid.h:35-49`). SiLU explicitly rounds its result for false mode (`BH/ckernel_sfpu_silu.h:23-33`). Consequently `math_approx_mode=False` does **not** make these two paths arithmetically identical.

| Path | Convolution accumulation | One BF16 boundary after FP32 sum | SiLU variant |
|---|---|---|---|
| Functional model | FP32 products/pairwise sum | Yes | FP32 |
| Native, current false mode | Non-FP32 destination; earlier rounding | No | BF16-oriented exp/reciprocal arm |
| Native, true mode control | FP32 destination | No | Accurate FP32 arm |
| Proposed model-compatible native mode | FP32 destination | Yes | Accurate FP32 arm |

A scalar check demonstrates why the missing boundary is substantive. With BF16-representable inputs `[1,1,0,0]` and taps `[1,0.00390625,0,0]`, the exact sum is `1.00390625`, whose round-to-nearest-even BF16 value is 1.0. `BF16(SiLU(BF16(z)))` is `0.73046875`; `BF16(SiLU(z))` is `0.734375`. This was checked with Python `struct`/`math`, not a device/model reproduction. It proves non-equivalence of the expressions, not that this particular value occurs in the failed run or that the false-mode hardware accumulator equals z.

The earlier real-weight contribution PCCs, 0.9945418239 with true mode and 0.9990801811 with false mode, are compatible with coupled changes in accumulation, missing intermediate rounding, and activation precision (`gdn_conv_shared_activation_v2.json`, `gdn_layout_conv_bf16_v1.json`). They do not prove that lower precision implements the reference contract, and cannot select one of the three differences as the sole cause.

## Lowered geometry, history, buffers, and completion

Every GDN decode in this matrix, including passing lengths, reaches the native op with the same physical geometry. Logical token count 1 is deliberately exposed as `[1,32,10240]` using its existing tile padding. History is `[1,3,10240]` BF16 row-major, taps are BF16 tiled row vectors, output widths are Q=2048/K=2048/V=6144. `actual_start=0`, predecessor equals the supplied history tensor, mesh is 1x1, and there is no fabric exchange (`M/tt/fused_decoder.py:604-647`). Validation accepts this layout/dtype/alignment combination (`C/device/qkv_causal_conv1d_silu_device_operation.cpp:30-90,115` onward).

Factory calculations (`C/device/qkv_causal_conv1d_silu_program_factory.cpp:295-308`) give `Mt=1`, `Qt=64`, `Kt=64`, `Vt=192`, `Ct=320`, `block_ct=4`, `num_blocks=80`, and 80 work items. False mode gives destination limit 8 and group 4; true mode gives destination limit 4 and the same group 4. Diagonal construction groups are respectively 8 and 4 for 16 weight tiles. These values divide exactly; no partial-group case occurs. The same work range is supplied to reader, compute, and writer (`:480-489`).

For each work item/channel block, the concrete buffer ledger is:

| Buffer | Capacity (tiles) | Producer | Consumer / advancement |
|---|---:|---|---|
| Identity | 1 | Reader reserves/pushes 1 tile once | Compute waits once; reuses the tile |
| Taps | 16 | Reader reserves/pushes 16 tiles after read barrier | Compute waits 16, creates diagonals, pops 16 |
| Diagonal weights | 16 | Compute reserves/pushes 16 tiles | Waits 16, indexes them across four taps/channel; pops 16 on block change or completion |
| Shifted activations | 32 | Reader reserves/pushes 16 tiles, ordered channel then tap | Compute waits 16, consumes four taps for each of four output tiles, pops 16 |
| Output | 8 | Compute reserves/pushes 4 tiles after activation/pack | Writer waits 4, writes all four with correct Q/K/V-relative indices, waits for NoC writes, pops 4 |

References: `C/device/kernels/dataflow/reader_qkv_causal_conv1d_silu.cpp:104-145,223-240` (reader); `C/device/kernels/compute/qkv_causal_conv1d_silu.cpp:33-86` (compute); `C/device/kernels/dataflow/writer_qkv_causal_conv1d_silu.cpp:117-142` (writer). Capacities come from `C/device/qkv_causal_conv1d_silu_program_factory.hpp:66-74`. The last channel block starts at tile 316 and writes V tiles 188..191, within its 192-tile width. A full final group exists. No producer/consumer count contradiction was found.

The reader maps output row r/tap j to history/current row `r-3+j`; for native output row 0, only the three carried rows and current row 0 participate (`reader:149-168,180-240`). BF16 diagonal matmul has no cross-token reduction, so later padded rows do not enter earlier logical output rows for finite inputs. Only logical rows are retained, and caller state is updated from `history = concat(conv_state, projected)`, not from the native physical padding (`M/tt/fused_decoder.py:631,648-651`). At prompt/decode0/decode1 the row-0 tap windows are respectively `[0,0,0,p0]`, `[0,0,p0,p1]`, `[0,p0,p1,p2]`. This supplies a data-dependent numerical explanation opportunity without asserting an off-by-one error.

Pack-thread activation waits on the math/pack semaphore, selects the packer destination half, then stalls pack until SFPU completion (`ttnn/cpp/ttnn/operations/matmul/device/kernels/compute/bmm_fused_activation.hpp:110-123`). This is an explicit handoff, not an unguarded read of queued matmul output. Startup, broadcast diagonal construction, and matmul all use BF16 full-tile buffers, so no asymmetric format change was found. SiLU initializes Prgm0=2 and Prgm1=1/ln2; the intervening FPU operations do not overwrite those SFPU constants (`BH/ckernel_sfpu_sigmoid.h:94-109`). No concrete synchronization or buffer-lifetime violation was established in this path. A native-output provenance/consumer problem remains logically possible until exact-input comparisons, but has less direct evidence than the arithmetic contract changes.

## Minimal repair and decisive follow-up

Keep group64 accurate projections, fused expert reduction, and direct recurrence reshapes. Localize the repair to the convolution call/kernel contract:

1. Add an explicit opt-in model-compatible convolution rounding mode, or a model-local specialization. In that mode use FP32 destination accumulation, round the completed z to BF16 with nearest-even semantics, evaluate SiLU from that rounded value using the accurate FP32 math path, and pack BF16. This can remain fused; the boundary can be a register conversion rather than a DRAM intermediate. Do not globally change the unrelated default SiLU primitive.
2. Preserve the existing native op's ordinary semantics for other callers. Its cancellation test explicitly targets `SiLU(sum products)` without this intermediate rounding, so changing the generic behavior silently could invalidate an intentional contract.
3. Keep the current false and true modes as diagnostic contrasts. Changing only `fp32_dest_acc_en` conflates accumulator and activation precision and leaves the reference boundary absent. Disabling convolution is a proven control, not evidence that the faster native approach should be abandoned.

Implementation detail that matters: `BH/ckernel_sfpu_typecast.h:272-287` demonstrates BF16 nearest-even rounding followed by masking and FP32 storage. Its initialization at `:608-613` overwrites Prgm0/1/2, so reusing it requires restoring SiLU initialization before activation. Public `typecast_tile` wrappers execute on MATH (`tt_metal/hw/inc/api/compute/eltwise_unary/typecast.h`); placing one inside a PACK block will not perform the intended conversion. Prefer a local fused PACK SFPU body that rounds x while preserving the SiLU constants, or explicitly supported PACK wrappers with the correct reinitialization and destination addressing. Do not change the destination-format template to false merely to introduce rounding: that also selects the lower-accuracy activation path.

The repair is a source-supported proposal, not a verified patch. If sequential FP32 matmul accumulation still differs at BF16 ties from the functional pairwise sum, reproduce the four-product/pairwise order inside the specialization; measure this before adding complexity.

For the owner's next hardware experiment, capture the **same** projected BF16 QKV, taps, and three-row history immediately before convolution for prompt/decode0/decode1 of seed1901. Compare the current false mode, true mode, functional four-product reference, and proposed boundary-preserving mode on those identical operands. Record pre-SiLU values if available, post-SiLU Q/K/V, normalized Q/K, recurrence output/state, GDN output projection, MLP router logits and top10 IDs. Use the identical incoming recurrence state when comparing a single step; separately test each variant's full evolving state. This distinguishes local producer differences from accumulated state and downstream amplification.

Then rerun the unchanged eight-case synthetic matrix and real-HF layers0/1/3 short gates, followed by the existing live continuation checks and warmed latency measurements for the repaired source. A passing synthetic-only repair is insufficient if it regresses the already-passing real-weight path. No test relaxation, synthetic-specific branch, or data-dependent fallback is indicated.

## Other Potential Issues / hypotheses adjudicated

| Candidate | Evidence and preconditions | Assessment / smallest discriminating intervention |
|---|---|---|
| Native convolution numerical contract | Direct source discrepancy; exact conv-only failure and conv-off pass | Leading diagnosis. Test explicit pre-SiLU BF16 boundary with FP32 accumulation/activation on identical inputs. Which subdifference is decisive remains unresolved. |
| Direct recurrence reshapes corrupt head/scalar layout | Same recurrence compute/reader/writer; layout-only control passes with identical reported PCCs | Not needed for this failure. `M/tt/fused_decoder.py:464-597` removes conversion roundtrips; decode q/k/v `[48,1,128]` remains correctly tiled. Scalar reshapes still dispatch through TTNN rather than assuming an invalid metadata view. |
| Grouped projection mixes output rows or silently uses approximate reduction | Setup concatenates independent rows; products reduce only K; FP32 accurate SFPU reduction selected; conv-off group64 passes | No demonstrated structural defect. `M/tt/fused_decoder.py:31-68,815-829`; `ttnn/cpp/ttnn/operations/reduction/generic/device/reduce_op.cpp:202-217`. Different summation order exists but does not independently reproduce this failure. |
| Expert routing amplifies a small GDN difference | GDN feeds residual write, MLP mixing, BF16-rounded logits, then discrete top10 (`M/tt/functional_decoder.py:134-142`; `M/tt/fused_decoder.py:155-170`) | Plausible explanation for the sharp whole-decoder PCC drop, not an observed route flip. Failure artifacts contain no paired router logits/IDs. Compare rank10/rank11 and selected IDs at the failing token. |
| History shift / tail padding / CB underflow | Lowered reader, writer, caller history copy and exact 4-tile groups are coherent | No concrete violation found. Passing/failing native decode geometry is identical. The different test seeds prevent a pure length-based deduction. |
| Trace corruption or incomplete asynchronous execution | Failing output follows blocking replay/readback; conv-off controls use the same trace harness | Unsupported as headline. A producer replacement changes allocations too, so controls alone do not prove arithmetic; exact-input eager/native-output comparisons can close this gap if necessary. |

The primitive test coverage is a material gap. `tests/ttnn/unit_tests/operations/experimental/kda/kda_test_utils.py:370-372,415-423` generates BF16 operands and forms its general golden with BF16 products and sequential BF16 additions, a third rounding contract. The general native test accepts PCC 0.999 and NRMSE 0.008 (`tests/ttnn/nightly/unit_tests/operations/experimental/kda/test_qkv_causal_conv1d_silu.py:239-241`). Its cancellation test explicitly requires the false-mode arm to breach the FP32 bound (`:790-825`). Passing that suite does not establish the model's FP32-products/single-BF16-boundary equivalence. Add a targeted golden for the proposed mode, including BF16 tie/boundary and cancellation inputs, without weakening either existing contract.

## Report review

After the first draft, headline claims were checked again against the completed factorial controls, actual selected source/configuration, reader/compute/writer contracts, installed HF fallback, and primitive tests, including an independent read-only review. All five evidence logs matched their provenance SHA256 values; the frozen fused-source hash remained unchanged. The investigation does not promote a padding bug, recurrence-layout bug, incorrect grouped reduction, observed router flip, or a specific accumulator-bit-width explanation: their required evidence is absent or contradicted by the controls. The confirmed result is convolution-branch localization plus a concrete model numerical-contract mismatch. The exact arithmetic contributor and the proposed repair's model accuracy/performance remain hardware-verification work for the owner.
