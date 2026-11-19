# AutoDebug: gated-residual RMSNorm fusion accuracy

Investigation date: 2026-11-13. Inspection only; no device execution, resets, source edits, or changes to other stages. This report is the only investigator-created file. The AutoDebug launcher was not invoked. Three independent investigators reviewed native normalization, baseline arithmetic, and downstream amplification.

## Findings

**The native RMSNorm replacement does not preserve the original FP32 arithmetic, despite FP32 tensors and FP32 destination accumulation.** On the inspected Blackhole path it uses FPU operations with TF32 source intake for squared-value reduction, mean-plus-epsilon, and normalized-value multiplication. The original expression selects full-FP32 SFPU operations and a different reduction order. Native epsilon is also stored as truncated BF16. These are concrete arithmetic discrepancies at the replacement boundary, not evidence of a wrong normalization axis or missing weight.

**The leading explanation for the isolated decoder failure is amplification across BF16 rounding and MoE expert selection.** Both residual normalization sites change; their BF16 results feed precise GR projections and eventually a router that itself rounds logits to BF16 before top-10. Precise projections preserve computations on their actual inputs; they cannot undo changed inputs. A changed expert at the selection boundary can produce a large output change. The saved artifacts contain no logits or expert IDs, so expert-selection changes at the failing row remain unproven.

**The component PCC gate cannot establish decoder safety.** PCC is insensitive to positive row-wise gain. A normalization-scale error can therefore have nearly perfect component PCC, change BF16 values, and disturb the relative contribution of four streams. The component probe also exercises only attention-norm weights on separate random inputs, not the MLP norm on actual post-attention states.

The smallest reliable policy is to retain the original residual norm: `fuse_residual_norm=False` is already the factory default; the later-added `merge_norm` flag must also remain false for that control. A smaller fusion that preserves the original square/mean and merges only epsilon addition with rsqrt is a reasonable in-scope candidate, subject to the same decoder gates. No change to the router, tolerances, or low-level layernorm implementation is justified within this stage.

## Evidence and revision boundary

Repository: `/home/ttuser/qwen38-flash/tt-metal`; inspected HEAD `70dc6f5dd24298741e2f108a281160677e986155`.

In source references below, **A** means `models/autoports/qwen_qwen3_8_flash_next`, **LN** means `ttnn/cpp/ttnn/operations/normalization/layernorm`, and **NG** means `ttnn/cpp/ttnn/operations/eltwise/binary_ng`. Line numbers are inspection anchors; function names identify the code if the live workspace moves.

Direct observations from [norm_v1.json](norm_v1.json) and [candidates_layer0_v1.json](candidates_layer0_v1.json):

| Case | Minimum whole-output PCC | Minimum contribution PCC | Other observations |
|---|---:|---:|---|
| Component norm, 1 / 31 / 32 tokens | 0.999998569 / 0.999998391 / 0.999998689 | Not measured | All three outputs differ bitwise from baseline |
| Layer 0 baseline, prefill33 | 1.0 | 1.0 | Baseline compared with itself |
| Layer 0 activation-only | 1.0 | 1.0 | Reported control passes |
| Layer 0 norm-only | 0.994016469 | 0.991623998 | Fails both 0.995 minimum-row gates |
| Layer 0 combined | 0.994016469 | 0.991623998 | Entire reported prefill PCC structure equals norm-only |

- The sole contribution row below 0.995 is zero-based **row 8**, the ninth token. Its PCC is 0.9916239977. Row 32 is 0.9999936819; several other rows have smaller excursions, including row 29 at 0.9985944033. The artifact gives no per-row whole-output PCC, so it does not establish which row owns that separate minimum.
- Norm-only final prefix-state PCC is 0.9999976754 for convolution state and 0.9999907017 for recurrent state. These are flattened state correlations, not per-token attention-output comparisons.
- Its two standalone decode inputs pass, with contribution PCC approximately 0.99998945 and 0.99998581. All eight trace replays equal that variant's eager output/state. They are not bitwise equal to the baseline merely because `exact_output` / `exact_state` say true.
- Both saved logs explicitly identify `blackhole` topology. Their SHA256 values match their provenance files. No hardware identity was queried during this investigation.

Revision qualification matters. Both provenance files name the same HEAD, but include hashes of uncommitted files. `A/tt/decoder_components.py`, `functional_decoder.py`, `probe_fused_norm.py`, and the candidate run's precise projection/router sources match the inspected files. `fused_decoder.py` differs from both historical hashes, and `probe_fused_candidates.py` differs from its recorded hash. The historical fused hashes are `7909d669a10a34b3f1eccfc09e32d3cdfecae457727ac8ad11ceceafdafd722c` (component) and `6a6a4d62e1deab090238b5a571c87718abb985f0af981f5c19190bd404d9d393` (decoder).

The first inspected full-file hashes were `c49ab7d029ef8af0be6baa44f4a0195ad4cd1b4d7295083baf22b675f023c536` for `fused_decoder.py` and `ff8599f762995e972b1197a6619af7d94339893952ebddcfed06d8ca7400e843` for `probe_fused_candidates.py`. The live files also changed during inspection, without investigator edits. In particular, a `merged_rms` helper and `merged_norm` test option appeared, followed by a `merge_norm` factory flag. They are not measured by the two named v1 artifacts. An AST comparison before the alternate norm branch appeared established that `FusedGatedResidual.read` differed from `GatedResidual.read` only in the RMS call and its compute argument. The later native branch still calls the same `fused_rms` helper; the new merged branch is separate. Historical measurements and current-source findings are therefore kept distinct; no archived historical source body or compiled-kernel hash was available to prove every historical lowering detail.

## The exact model contract

The reference implementation at `/home/ttuser/qwen38-flash/reference/modeling_qwen4_exp.py:141–161` groups width 10240 into four groups of 2560, computes RMS in FP32, multiplies by `1 + weight.float()`, and only then casts to the input dtype. The configured epsilon is `1e-6`.

`A/tt/decoder_components.py:27–30,57–61,77–97` implements:

```text
v = FP32(x)
m = mean(FP32_multiply(v, v), last dimension)
s = rsqrt(FP32_add(m, epsilon))
normalized = BF16(FP32_multiply(FP32_multiply(v, s), weight))
```

The weight loader already applies the `+1` offset and stores `[1,4,1,2560]` FP32 weights. The native candidate at `A/tt/fused_decoder.py:18–22` keeps the same external learned-weight multiplication and final cast, replacing only the unweighted normalization with `ttnn.rms_norm`.

For a chunk of T tokens, the grouped tensor is logical `[1,4,T,2560]`, tiled, interleaved DRAM. T is at most 32. The original scale is `[1,4,T,1]`; normalization is a column broadcast for T>1 and scalar broadcast for T=1. The learned weight is a row broadcast for T>1 and has matching trailing dimensions at T=1. Both operands are FP32 for the final weight multiplication. The native output is also FP32, tiled, with output memory defaulting to input memory; this is an arithmetic-path change, not a deliberate dtype/layout change at the consumer boundary.

The native call receives `HiFi4`, `math_approx_mode=False`, `fp32_dest_acc_en=True`, and `packer_l1_acc=False` from `WeightLoader.compute` (`decoder_components.py:37–43`). The standalone rms_norm API has different defaults, but the candidate supplies a config, so those defaults do not explain this failure.

## Lowered arithmetic comparison

| Phase | Original expression | Native unweighted RMSNorm |
|---|---|---|
| Square | FP32 binary SFPU multiply, direct unpack to DEST | FPU square, FP32 storage/destination but TF32 source intake |
| Width mean | Accurate FP32 SFPU reduction; sum then FP32 multiply by `1/2560` | FPU tile reduction of squared values; multiply accumulated sum by FP32 `1/2560` |
| Add epsilon | FP32 scalar SFPU add | Reload stored mean through TF32; FPU add with BF16 epsilon tile |
| Rsqrt | Precise Blackhole SFPU algorithm on original mean-plus-epsilon | Same precise algorithm family, but on already changed mean-plus-epsilon |
| Multiply input by scale | FP32 binary SFPU broadcast multiply | Reload stored inverse RMS through TF32, then FPU broadcast multiply |
| Learned weight and BF16 cast | FP32 per-stream multiply, then cast | Same external operations |

Source chain establishing the comparison:

1. **Original binary operations:** `ttnn/cpp/ttnn/operations/eltwise/binary/binary.cpp:1390–1414` and `NG/device/binary_ng_device_operation.cpp:51–65` select the FP32 SFPU path. `NG/device/binary_ng_program_factory.cpp:1134–1146,1374–1381` enables FP32 destination and direct FP32 unpack to DEST. The Blackhole FP32 broadcasts use software broadcast rather than the Wormhole-only LLK FP32 broadcast optimization (`:347–378`); they remain SFPU arithmetic. Classification is in `binary_ng_device_operation.cpp:217–244`.
2. **Original reduction:** `ttnn/cpp/ttnn/operations/reduction/generic/device/reduce_op.cpp:148–153,198–217,239–269` selects accurate FP32 SFPU reduction on Blackhole, lowers AVG to SUM, and applies reciprocal width afterward. `ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.inl:486–565` accumulates corresponding lanes across width tiles 0 through 79, then reduces the tile columns and applies the scalar. It is a different reduction tree from native tile reduction. The public mean default has fast/approximate mode disabled (`generic_reductions_nanobind.hpp:126–141`).
3. **Native dispatch:** `ttnn/cpp/ttnn/operations/normalization/rmsnorm/rmsnorm.cpp:67–83` forwards the explicit config, input memory, default program config, and RMSNORM type into the shared layernorm device operation. `LN/device/layernorm_types.hpp:19–21` defaults to `legacy_reduction=False`, `use_welford=False`. The tensors are interleaved, so the interleaved factory is relevant.
4. **Native FPU arithmetic:** `LN/device/kernels/compute/layernorm.cpp:271–303,314–334` explicitly uses FPU square, `row_wise_mean`, `BinaryFpu::Add`, and `BinaryFpu::Mul`. The square helper is documented as FPU in `ttnn/cpp/ttnn/kernel_lib/eltwise/api/convenience.hpp:73–80`. `normalization/kernel_util/compute/numeric.h:77–110,228–239` implements the FPU reduction and post-scaling; its `FLOAT32_REDUCTION` template argument does not turn that inner loop into the generic SFPU reducer.
5. **Why FP32 destination is insufficient:** `LN/device/layernorm_op_multi_core.cpp:1179–1187,1233–1241` pins ordinary FP32 buffers to `UnpackToSrc`. `tt_metal/jit_build/genfiles.cpp:1069–1075` chooses TF32 source format for this FP32-destination/BF16-scaler configuration; `tt_metal/jit_build/data_format.cpp:147–154` applies it to Float32 input unpacking. The precision loss is before arithmetic consumes the operands. The layernorm selector itself documents ordinary tile reducer/finalizer TF32 intake (`LN/device/layernorm_stats_selector.hpp:43–45`).
6. **Epsilon:** `LN/device/layernorm_op_multi_core.cpp:694–695` allocates its tile as `Float16_b`; the reader generates it at `reader_unary_interleaved_ln.cpp:135–137`. `ttnn/cpp/ttnn/kernel/dataflow/generate_bcast_scalar_metal2.hpp:14–24` takes the high 16 bits of the FP32 scalar. Static bit calculation gives `9.98377799987793e-7`, versus FP32 `9.999999974752427e-7` for the original epsilon.

Here, full FP32 describes operand and destination precision, not a guarantee of correctly rounded IEEE arithmetic or bitwise agreement with PyTorch. BF16 input values are exactly representable in TF32, so the native square's *input* conversion alone is not the demonstrated first error. The squared results generally require more mantissa bits; their reduction intake, the mean reload, and inverse-RMS reload are real precision-loss opportunities. The latter two remain discrepancies even if a particular square or reduction happened to match. Neither FP32 tensor dtype nor HiFi4 removes them.

Epsilon truncation is real but ranks below TF32 intake as a likely dominant source. For an illustrative mean square of `0.02² = 0.0004`, its isolated relative effect on inverse RMS is about two parts per million. That is an estimate for the probe's input scale, not a measurement of actual norm rows, and it does not establish that epsilon can never trigger a sensitive route.

The rsqrt implementation itself is not shown to select a low-precision fast approximation. `unary/device/unary_program_factory.cpp:532,761–764` uses precise mode for standalone unary; native uses the supplied `math_approx_mode=False`. `tt_metal/hw/inc/api/compute/eltwise_unary/rsqrt.h:29–32` distinguishes Fast/Default domain handling from kernel approximation precision. Blackhole `ckernel_sfpu_sqrt.h:62–90` implements the precise 23-bit refinement branch. Changing the rsqrt enum alone does not restore the operands lost in the native FPU path.

## Dimensions, buffers, and completion boundaries

`A/tt/functional_decoder.py:157–174` splits prefill33 into chunks of 32 and 1. Each chunk calls both residual reads. There is no native normalization call with 33 logical token rows.

| Quantity | T=1 component / tail / decode | T=31 component | T=32 component / first prefill chunk |
|---|---:|---:|---:|
| Padded grouped shape | `[1,4,32,2560]` | Same | Same |
| NC / Ht / Wt | 4 / 1 / 80 | Same | Same |
| FP32 width block size | 4 tiles | 4 | 4 |
| Width blocks | 20 full blocks | 20 | 20 |
| Tile rows assigned | 4 | 4 | 4 |
| Partial width / scaler tiles needed | No / 1 | No / 1 | No / 1 |

The default grid normally assigns one tile row to each of four cores. For these parameters, the compact factory's modeled CB footprint is 737,280 bytes per core. Its actual selection depends on live usable L1 (`LN/device/layernorm_op_multi_core.cpp:265–345`), which the saved artifacts do not record. If it does not fit, the streaming variant is selected. That variant retains the relevant BF16 epsilon and FPU mean-add/normalization arithmetic (`layernorm_large_tensor.cpp:398–409,435 onward`); therefore the arithmetic finding does not depend on assuming the compact kernel ran historically.

For the compact path the per-core queue ledger is coherent: reader produces 80 input tiles plus one scaler and one epsilon tile; square retains input and produces 80 squared tiles; mean consumes all 80 and produces one mean tile; add/rsqrt consumes the mean and produces one scale tile; normalization produces 20 blocks of four output tiles; writer barriers precede matching output pops. Input 80 and scale one then drain; scaler and epsilon each drain once at exit (`layernorm.cpp:416–441`; `writer_unary_interleaved_start_id_blocked.cpp:29–39`). No width remainder exists. Height padding is in independent rows and is not included in a width reduction. No buffer-count contradiction was found for this concrete case.

The saved comparisons are not merely host enqueue success. Component outputs are read with `to_torch`; traced timing waits with `synchronize_device`. The decoder probe reads prefill output and states back; replay calls synchronize before comparisons. Thus the reported PCC failures reflect completed outputs. However, trace-vs-eager equality only establishes repeatability within a variant. Each decode restores that variant's own prefix state, then tests one of two independent inputs; it does not establish identical baseline state or a long chained continuation. See `A/tests/probe_fused_candidates.py`, functions `snapshot`, `restore`, and the eager/replay loops; initially lines 173–232, subsequently approximately 186–245.

## Plausible amplification chain and its limits

| Candidate | Source evidence and fit to observations | Unverified condition / smallest intervention |
|---|---|---|
| Native arithmetic mismatch | Proven current-source delta at exactly the norm replacement; consistent with nonexact components and norm-only failure | Historical compiled path not archived; retain original norm or test partial SFPU fusion |
| Expert-membership amplification | Actual BF16/top-10 discontinuity; fits a few unusually affected tokens | Row-8 IDs/margins absent; fixed-route control must demonstrate recovery |
| Continuous GR/attention/expert amplification | All are downstream consumers with nonlinearities and rounding; final state PCC does not exclude them | No intermediate localization; split sites and capture first large divergence |
| Epsilon-only explanation | Proven smaller numeric delta, same replacement boundary | No evidence it dominates; change epsilon alone in baseline |
| Tail/queue/trace defect | No source contradiction found for the instantiated compact counts; row 8 is in a full chunk | Unrecorded runtime internals remain possible; matched-input boundary controls before any protocol rewrite |

```text
native arithmetic changes per-stream inverse RMS
  → different BF16 normalized values / relative stream gains
  → changed GR down/up gates, mixed input and injection
  → attention + residual write + second GR read
  → changed precise router inputs
  → BF16-rounded router logits and potentially changed top-10 membership
  → different weighted expert contribution
  → residual output fails at a sensitive token
```

The arrows before changed expert membership are supported as possible data propagation by `A/tt/decoder_components.py:77–109` and `functional_decoder.py:134–142`. They are not captured intermediate measurements from row 8.

The discontinuity is concrete: `A/tt/precise_projection.py:60–75` forms products and balanced sums in FP32, then casts logits to BF16. `SparseMoe.router_logits` casts those rounded logits back to FP32 (`decoder_components.py:135–145`). `router_topk.py` invokes the ATen-compatible selector; `tt/kernels/router_topk_algorithm.hpp:15` onward compares values with the intended tie behavior. The model uses 512 experts and top-10. Small logit changes can cross the rank-10/rank-11 boundary or alter a boundary tie. FP32 re-expansion does not restore pre-rounding logits.

This predicts data-dependent isolated token failures while many rows and two decode inputs pass. The norm-only/combined equality and activation-only control support the norm replacement as the intervention boundary. They do not prove which of the two norms first produces the damaging perturbation, or that routing rather than attention/nonlinear arithmetic first produces the large error.

PCC adds an important blind spot: for positive `c`, `PCC(y, c*y) = 1`. The component probe flattens every stream/token into its own width-2560 row before correlation (`probe_linear_contract.py:46–50`). Different scalar gains across the four streams are therefore hidden by separate row correlations but can change the flattened GR projections and mixed-stream direction. BF16 rounding then makes even a scalar gain change nonuniform. Component PCC above 0.999998 is not a bound on absolute error, gain error, or route stability.

High final state PCC is insufficient to rule out attention: GDN convolution state retains the final three projected rows (`decoder_components.py:242`), which do not directly describe row 8; recurrent state is compared only as one flattened tensor. Attention output normalization/gating also lies outside those state metrics. Layer 0 uses GDN, so QSA index selection is not involved in this case.

## Minimal interventions within the allowed scope

1. **Retain the original norm policy.** Keep `fuse_residual_norm=False, merge_norm=False` in the current factory; if a norm candidate must remain selectable, explicitly classify native RMS as rejected for the demonstrated layer-0 prefill33 contract. The original `GatedResidual.read` remains available without modifying `decoder_components.py`. Both defaults are already false, so this is not a claim that a new default fix is needed.
2. **Prefer the smallest partial fusion experiment.** Retain the exact original FP32 square, generic mean, two multiply order, external weight, and BF16 output boundary. Merge scalar epsilon addition and its immediately following rsqrt:

   ```python
   variance = ttnn.mean(ttnn.multiply(value, value), dim=-1, keepdim=True)
   scale = ttnn.add(
       variance,
       eps,
       activations=[ttnn.UnaryWithParam(ttnn.UnaryOpType.RSQRT, 0.0)],
   )
   result = ttnn.typecast(
       ttnn.multiply(ttnn.multiply(value, scale), weight), x.dtype
   )
   ```

   The live workspace's newly added `merged_rms` has this structure. Scalar add remains the FP32 SFPU path and the post-activation runs before its final pack (`NG/device/kernels/compute/eltwise_binary_sfpu_scalar.cpp:60–81`). The binary descriptor defaults to `math_approx_mode=False` (`NG/device/binary_ng_program_factory.cpp:1511–1514`; `tt_metal/api/tt-metalium/program_descriptors.hpp:103–111`). This removes one separate rsqrt dispatch. It is a source-supported candidate, not a validated fix or speed claim; neither named v1 artifact tests it. Compare the scale itself, final BF16 values, and full decoder results before accepting it.
3. **If useful afterward, test a larger partial fusion.** Preserve square/mean, but pass the mean as the RHS of `multiply(value, mean_sq)` with ordered RHS activations `ADD_UNARY_SFPU(eps)` then `RSQRT(0.0)`. The public activation constructors/bindings exist; FP32 RHS preprocessing and final SFPU multiplication are supported (`binary_nanobind.cpp:1700–1713`, `NG/device/binary_ng_program_factory.cpp:1228–1239`, `NG/device/kernels/compute/eltwise_utils_sfpu.hpp:19–48`). This can remove two standalone dispatches, but introduces more preprocessing/broadcast work and needs separate accuracy/timing evidence. Do not substitute a different mean or reorder learned-weight multiplication.

Do not attempt to repair this by loosening the 0.995 threshold, removing the router's BF16 boundary, changing tie-breaking, dropping precise projections, or casting the norm result to BF16 before multiplying the weight. Those interventions change the established computation rather than restore it.

## Discriminating experiments, in priority order

These are proposed device follow-ups for the owning stage, not experiments run by AutoDebug. Instrumentation belongs in tests/docs and any implementation candidate stays in `fused_decoder.py`.

| Experiment | Control and measurements | What it decides |
|---|---|---|
| 1. Re-establish revision-consistent contrast | Archive exact source hashes, architecture/config, input seed 2071, layer 0, prefill33; compare baseline, native norm, and partial-fusion candidate with identical initial state. Record effective native kernel/factory if available. | Confirms that the current branch reproduces the historical boundary; prevents attributing a newer helper to v1 results. |
| 2. Split the two residual sites | Baseline, attention-only native norm, MLP-only native norm, both. Also feed exactly the same saved baseline grouped operands into each norm, especially actual post-attention inputs and both sets of weights. | Separates direct MLP-norm error from upstream attention/state propagation. |
| 3. Measure norm error beyond PCC | Save original square/mean/scale, original/native unweighted FP32 outputs, weighted FP32 outputs and final BF16 outputs. Report per-token/per-stream gain or norm ratio, max/RMS error, BF16 equality fraction and ULP changes. Preserve coordinates. | Tests whether the perturbation is predominantly scale drift and whether the component random probe understates the actual MLP error. |
| 4. Locate row 8's first large divergence | Capture mixed inputs and injections at both sites, attention output, post-attention residual, router input, BF16-rounded logits, top-10 IDs/scores and weights, routed/shared expert outputs, final contribution. Read host summaries outside runtime audit and outside timing. | Determines whether the jump precedes routing; checks the complete causal chain rather than inferring it from final state PCC. |
| 5. Fix routes diagnostically | Compare baseline/candidate selected sets and paired expert weights. On candidate expert inputs, substitute baseline membership with recomputed candidate scores, then baseline membership plus baseline weights. Rebuild both `dense` and its `active` union before sparse execution so every substituted expert is computed. Record rank-10/rank-11 margins and boundary ties. | Recovery localizes membership versus weight amplification. Identical sets refute a membership-flip story; continued failure with fixed routes requires attention/expert/input investigation. Different order alone is insufficient. |
| 6. Isolate arithmetic discrepancies | Original expression with only epsilon changed to the native BF16-truncated value; native-vs-original unweighted FP32 results; original-vs-partial-fusion scale/output. Optional CPU TF32 emulation is explanatory only. | Tests epsilon alone and validates partial fusion without pretending CPU emulation reproduces device reduction/packing exactly. |
| 7. Check nearby supported shapes and continuation | Same underlying prompt prefix at lengths 1,31,32,33,65; restore common baseline prefix for localization, then retain candidate-own-prefix replay checks. Include both norm weight sets. | Distinguishes data sensitivity from tile boundaries; avoids changing random inputs while treating length as the sole variable. |

Acceptance remains finite outputs plus **every logical row** meeting both whole-output and residual-subtracted contribution PCC ≥0.995. Exact component equality is a useful diagnostic, not a replacement acceptance bar. Preserve trace-vs-eager output/state checks and the existing precise projections/router. Benchmark only candidates that pass, with fresh same-harness measurements; the v1 decoder timing was allocation-tracked and explicitly not clean timing. Component speedups alone do not establish decoder benefit.

## Other Potential Issues and rejected shortcuts

- **Directly passing the per-stream weight to native RMS is unsafe.** The standard reader loads gamma once at `block.start()` with no stream offset (`LN/device/kernels/dataflow/reader_unary_interleaved_ln.cpp:176–187`), then reuses it across tile rows. The current external `[1,4,1,2560]` multiplication avoids this hazard. A direct gamma fusion would require a different supported mapping, not merely supplying `weight=`. This is a proposed-fix hazard, not the current bug.
- **`use_welford=True` is not an available RMS accuracy switch.** The device validator explicitly rejects RMSNorm with it (`LN/device/layernorm_device_operation.cpp:320–333`). The FP32 SFPU finalizer eligibility also excludes RMSNorm (`layernorm_op_multi_core.cpp:240–244`). Existing high-fidelity flags are already enabled.
- **Epsilon truncation alone is not established as the cause.** It is a smaller independent discrepancy; use the controlled epsilon-only experiment before assigning it the full decoder failure.
- **No evidence supports an incorrect sparse expert union.** The union schedules experts across the token tile, but dense routing weights remain per token and are applied before reduction (`A/tt/decoder_components.py:146–195`). A changed union can be a consequence of changed routes. The candidate preserves the relevant routing computation, and activation-only passes the reported control.
- **No demonstrated axis, group, offset, or BF16 cast-placement error.** Group width, four streams, `1+weight`, external weight dtype and final cast placement agree with the reference. Native arithmetic precision differs below that correct high-level graph.
- **No demonstrated tail, queue, trace-lifetime, or low-level state-mode defect.** Concrete width counts are the same in passing component cases and the failing first chunk, and the worst contribution occurs within that first full chunk. This lowers the rank of a pure tail hypothesis without claiming all state-dependent execution defects are impossible. No target-specific rsqrt state violation was found.
- **Native op tests do not establish this model contract.** `tests/ttnn/unit_tests/operations/fused/test_rms_norm.py:117` explicitly notes TF32 unpack. Its FP32 test allows PCC 0.999, relative tolerance 0.052 and absolute tolerance 0.064 (`:130–145`). Those tests neither require original-expression equality nor exercise downstream BF16 expert decisions.

## Claim review

The first draft was revisited against the code and independently challenged by the downstream and baseline-arithmetic investigators. Review clarified the new factory flag, mutable-source hashes, precise-mode ownership, and the need to rebuild sparse scheduling metadata in fixed-route controls. TF32/FPU arithmetic versus original SFPU arithmetic is established in the inspected source and fits the intervention boundary. Missing historical source bodies and native kernel-selection records limit exact historical attribution. Expert-selection amplification explains the sparse token pattern but remains a hypothesis until logits/IDs and fixed-route controls exist. Wrong axes/weights, fast-rsqrt precision, a 33-row kernel boundary, and sparse-union bugs were not promoted to causes.

Static checks performed: JSON inspection; provenance/log SHA256 comparison; relevant source hash comparison; initial GR-method AST comparison; verification that norm and combined have identical reported prefill metrics and only contribution row 8 misses 0.995; epsilon bit conversion; source tracing through Python, wrapper, program factory, dataflow, compute, and Blackhole helpers. No reproduced numerical result or candidate performance is claimed.
