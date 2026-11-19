# AutoDebug: Qwen3.6 fused decoder, actual layer34 / seed154

## Headline findings

**The severe layer34 failure is already present in V4. This is a shared decoder correctness gap with an additional V5 numerical change, not an established V5-only gated-norm regression.** The same-oracle V4 control completed during this investigation: 282 failing decode positions, minimum contribution PCC 0.9126810471. V5 has 296 unique failing positions and minimum 0.9125414665. Both minima occur at absolute position7944. There are 196 shared failures; 86 positions fail only V4 and 100 only V5.

**V5's native decode gated norm does change HF's rounding contract.** It removes the BF16 boundaries after unweighted normalization and gamma multiplication, changes multiplication association, and introduces a BF16 intermediate before the final Z multiply. The source and an existing isolated probe establish a numerical difference. They do not establish that this change causes the shared severe failure. Gamma's maximum crossing1 is not a demonstrated root cause.

**The leading symptom-level hypothesis is a small shared attention/postnorm discrepancy amplified by MoE selection or combination, with V5 perturbing which tokens cross a routing boundary.** Bidirectional isolated pass/fail flips, early failures and very similar worst outputs support this hypothesis. No layer34 intermediate capture establishes actual expert membership, logit margins, or the earliest divergent operation. Large weight-specific attention/core error remains a competing explanation and must be tested on equal terms.

**The existing routing diagnostic has two capture defects relevant to follow-up:** its saved incoming state aliases the state V5 overwrites, and its decay observation can be overwritten by the later Q-product. These affect diagnostic interpretation, not the stress test that detected this failure.

Inspection only: no TTNN import/execution, hardware access, model reproduction, implementation edits, native edits, or full-model/multichip changes. CPU work consisted of reading preserved evidence, recomputing metrics from saved tensors, hashing files, and inspecting small checkpoint parameter vectors.

## Scope, identities and evidence

Paths below are repository-relative unless abbreviated:

- M = models/autoports/qwen_qwen3_6_35b_a3b
- S = M/tt/fused_decoder.py
- V4 = M/doc/fused_decoder/selected_fused_decoder_v4.py.snapshot
- E = M/doc/fused_decoder/endurance
- HF = python_env/lib/python3.12/site-packages/transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py
- N = ttnn/cpp/ttnn/operations

Prepared native Git HEAD is 4c643884bf51a39fc1166e0b86b86bd373bedfee. The existing worktree contains model/tests/docs changes; this investigation leaves them intact.

| Object | SHA256 / identity |
| --- | --- |
| Selected V5 S, independently hashed | 8ac733159befba919abf705603f3aada9d105195c7b495d8cd25aa65e531769f |
| Executed V4 snapshot, independently hashed | b1633d51621b4c45d9514a66fad21faa57ad562185502342c3d87bd5ebc2701c |
| Stress harness | a3ac90f98a6b3ccb9b05e9e463aa4014760a276fdc8b3e726c6df53b5ca93f2e |
| HF checkpoint revision | 345d467b117d1fd5eda9b879a5988de9ae5f47a7 |
| Reported target architecture | Blackhole, single-device mesh |

Primary evidence is E/v5_representative_layer34_stress_seed154.json and its log, provenance and tensor sidecars; E/v4_layer34_stress_seed154.json and corresponding sidecars; E/v5_representative_layer34_s257.json; E/v5_layer_norm_scale_survey.json; and the shared oracle M/doc/fused_decoder/oracle_cache/v5_layer34_seed154_8192.pt.

V4's provenance also lists the live V5 source because the generic recorder hashes that file. The actual executed class comes from the snapshot, correctly recorded as measured_decoder_source_sha256 by M/tests/stress_fused_snapshot.py:29–43,51–55. Do not attribute the V4 result to the live file.

The V5 log records fresh HF reference computation; the V4 log explicitly reuses that generated oracle. The four source fingerprints in the oracle match current disk. The source-level native lowering below describes the prepared tree; artifact provenance does not independently fingerprint every native binary or compiled kernel.

### Direct observations

All stress cases below use prompt128 and continuation33. Listed failure counts use the existing threshold, min(raw PCC, contribution PCC) <0.995.

| Case | Decode × repeats | Prefill PCC | Minimum raw decode PCC | Minimum contribution PCC | Unique failing decode positions |
| --- | ---: | ---: | ---: | ---: | ---: |
| Selected V5 layer0, seed913 | 8192 ×3 | .9999701164 | .9966482650 | .9962079142 | 0 |
| Selected V5 layer0, seed42 | 8192 ×3 | .9999717837 | .9969917499 | .9958185879 | 0 |
| Selected V5 layer0, seed154 | 8192 ×3 | .9999681973 | .9974715758 | .9970268219 | 0 |
| V5 actual layer34, seed154 | 8192 ×3 | .9997163801 | .9278896333 | .9125414665 | 296 |
| V4 actual layer34, seed154 | 8192 ×1 | .9997163801 | .9279779596 | .9126810471 | 282 |

The selected-source layer0/3 shorter stress gates and layer3 seed913 8192 gate also pass. Copies under qualified_v5 are archives of these results, not additional experiments. The separate reported context262144 gate is not an8192-decode trajectory on layer34.

| Absolute position | V4 raw / contribution PCC | V5 raw / contribution PCC | Meaning |
| ---: | --- | --- | --- |
| 137 | .9999741407 / .9999668904 | .9941642208 / .9928203651 | First V5 failure; decode index9, tenth decode token |
| 165 | .9915619755 / .9897280534 | .9999683885 / .9999604595 | First V4 failure, V5 passes |
| 181 | contribution .9928846984 | contribution .9928810990 | Early shared failure |
| 7944 | .9279779596 / .9126810471 | .9278896333 / .9125414665 | Shared worst position |

V5's888 recorded failures are296 identical failing positions across three bit-equal repeats. V4 has one repeat; its unconditional repeated_bit_equal=true field is not evidence of repeat determinism. There are236 V5 raw-PCC failures and296 contribution-PCC failures; V4 has227 and282, respectively.

Failures per successive1024 decode tokens:

| Version | 0–1023 | 1024–2047 | 2048–3071 | 3072–4095 | 4096–5119 | 5120–6143 | 6144–7167 | 7168–8191 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| V5 | 37 | 34 | 35 | 40 | 41 | 37 | 38 | 34 |
| V4 | 28 | 35 | 34 | 39 | 34 | 40 | 35 | 37 |

These counts show no monotonic increase in failure density. They cannot establish that recurrent state is correct: state error can be selectively amplified by normalization and routing.

Independent CPU checks of saved tensors established:

- V4/V5 prefixes, all8192 decode inputs and all8192 HF outputs are bit-equal to the shared oracle. Those input tensors and oracle continuation also match the layer0 seed154 oracle's input corpus.
- All inspected tensors and metrics are finite. Every recorded decode PCC/contribution PCC recomputes within7.6e−15.
- No V4/V5 full output token is bit-equal. Their pairwise output PCC has median .9999809073 and minimum .9288438467: small widespread perturbations coexist with sparse large differences. At the shared worst position7944, their output-to-output PCC is .9999736250.
- Continuation33 passes in both JSONs; minimum raw/contribution PCC is .9990089525/.9989415306 in both. Metrics match exactly for32/33 positions; final8352 differs by about3e−6 contribution PCC. Failure tensor sidecars do not retain device continuation outputs, so independent saved-tensor verification covers decode.

### Interpretations that are not yet observations

Sparse bidirectional flips suggest a discontinuous downstream operation such as top-k expert membership. They do not prove a route swap at137,165,181 or7944. Likewise, the worst PCC being nearly equal is not proof that the two implementations chose identical experts or had identical recurrent state there.

The short layer34 s257 benchmark uses default seed42,257-token prefill and one decode input repeatedly evaluated from restored state. Its context8192 is capacity, not8192 decode steps. The eager/traced PCC .9999671066 therefore does not contradict the seed154 stress failure at137. See M/tests/run_decoder.py:27–38,136–180,214.

## Execution and numerical contract

### Harness and synchronization ledger

M/tests/stress_fused_decoder.py:30–34 generates fixed external prefix/tokens/continuation. HF consumes prefix,8192 one-token calls, then continuation (:69–76). The TT output is never fed back as the next input. Only convolution and recurrent state persist between decode calls.

The TT sequence is:

1. Prefill and blocking host output read; save both state tensors (:107–112).
2. Restore, warm decode, restore, synchronize, capture one decode (:114–127).
3. At every repeat, restore the saved prefix state (:138–139).
4. Refresh the input tensor, execute the same trace with blocking=True, copy output to CPU before replay (:140–155).
5. Assert every later repeat's output equals its first-repeat counterpart (:158–161).
6. Release trace, run continuation on final decode state, read outputs (:168–184).
7. Write JSON/tensors, assert failure, close mesh in finally (:185–219).

Thus the JSON failure follows completed replay/readback; this is not a host assertion racing ahead of unfinished device output. No concrete input-refresh, missing-restore, trace-output-alias or queue-count defect was found. Bit equality across repeats does not rule out a deterministic kernel error.

“Contribution” is full decoder output minus original input, including attention and MoE, not a direct attention/core/gated-norm metric. The subtraction occurs in BF16 before M/tests/reference.py:64–69 converts to double for mean-centered Pearson correlation. Recomputing subtraction in FP32 leaves the severe failure: minimum .9125618917 V5/.9126772179 V4. It changes the V5 threshold count296→297, not the diagnosis.

### Actual decode geometry and ownership

| Boundary | Effective shape, dtype/layout and ownership |
| --- | --- |
| External hidden state | BF16 TILE [1,1,2048], physically H32, fixed per token |
| Input norm | Custom FP32 arithmetic, effective gamma1+weight; BF16 result |
| Precise state projection | K/V/A/B packed6208 columns:2048+4096+32+32; BF16 operands, FP32 product/scan/reduction, BF16 result |
| Native output branch | Q/Z packed6144 columns:2048+4096; BF16 output from sharded native linear |
| QKV convolution |8192 channels, history3 plus current row, four taps; BF16 post-convolution/SiLU boundary |
| Q/K |16 heads×128, BF16-equivalent rounded L2, widened for recurrence; each shared by2 adjacent value heads |
| V / beta / decay |32 value heads×128; BF16 V/beta semantics, FP32 decay lookup |
| State | [1,32,128,128] FP32 storage; consumed BF16-rounded once per public decode |
| Core | [1,1,32,128] FP32 arithmetic, explicitly BF16 before output norm |
| V5 norm input | Head-first physical [32,32,128]; native processes all32 rows, only output row0 is consumed downstream |
| V5 gate tensor | Packed Q/Z physical [1,32,6144], Z offset2048 |
| Attention output / postnorm | BF162048 residual; postnorm width-sharded32 cores, each physical32×64, FP32 effective gamma |
| Router / experts |256 BF16 logits,8 selected experts with paired scores; sparse matmuls and expert reduction |

For a passing early decode and failing137/7944, geometry/work counts are unchanged. No token-position branch changes the decode factory or buffer capacity. Layer0 and34 select the same architecture/shape branches; the parameters and resulting data differ.

## Path-by-path adjudication

### 1. Input norm and projections: earliest candidates, no proven layer34 defect

S:80–82 correctly uploads effective ordinary RMSNorm gamma1+weight, while gated norm uses direct weight. S:355–447 computes input normalization and gamma multiplication in FP32 before the BF16 output, matching HF:823–831's semantic boundaries. Its reduction, square-root correction and reciprocal correction use a different arithmetic schedule from Torch. Rare BF16 midpoint differences are possible, but there is no saved layer34 normalized input proving one here. Existing layer0 observations must not be assigned to layer34.

S:96–108,278–323's precise K/V/A/B branch applies the same adjacent-pair permutation to inputs and weight rows. Their dot product is invariant under that paired permutation. Decode uses four K512 blocks; prefill uses two K1024 blocks, so FP32 addition order differs from each other and from CPU F.linear. That is a numerical contrast, not an identified packing/index bug.

The lowered precise multiply explicitly disables the approximate path. N/eltwise/binary_ng/device/binary_ng_device_operation.cpp:51–65 selects SFPU; binary_ng_program_factory.cpp:1134–1146,1374–1381 uses FP32 DEST and direct-to-DEST operands. The Float32 cumsum uses direct-to-DEST SFPU scan in N/reduction/accumulation/device/accumulation_program_factory.cpp:479–501 and kernels/compute/accumulation_tile_axis_compute.cpp:56–67. No unintended BF16 product spill or TF32 state-projection input contraction was identified.

The Q/Z branch at S:106,334–346 is a different, native matmul path. It does not promise the precise branch's CPU-matching arithmetic. S:527–532 assembles the correct Q/K/V/A/B/Z column ranges. Q affects current output and Q convolution history; Z affects the output gate. Neither Q nor Z enters the recurrent state update. Their errors can alter current attention and routing, but cannot cause accumulating recurrent-state error in this external-input harness.

Smallest discriminator: compare both projection branches against canonical projection on the same captured normalized input, including BF16 code differences per Q/K/V/A/B/Z channel. Do not attribute a projection mismatch to matmul until controlling the norm input.

### 2. Convolution, L2, gate lookup and recurrence: contracts largely agree

S:533–545 forms history=[three old projected QKV rows,current input] and uses four taps. V5 retains history[:,s:s+3] directly in persistent BF16 RM state; V4 converts the same carry back to TILE. At decode s=1 the carry is old rows1,2 plus current row. At a32-token interior it is rows32,33,34. New state does not alias the concatenated history. No off-by-one carry or padded-row persistence was found.

S:549–558 intentionally reproduces the BF16 boundaries for square, sum, epsilon, rsqrt and product in HF l2norm; exact equality of all primitive/reduction results is not proved for layer34. It uses sum+1e−6, not an incorrect mean+1e−6. The gate table is built per layer from all65536 BF16 projection-A codes (S:109–123). Compact compression has an exhaustive CPU bit-equality assertion for the remapping (S:124–157). It falls back to the full table if the compact window is too large. Neither assuming a layer0 table nor clipping layer34's gamma is supported by source.

For head h=2g+m, state coordinate[k,v], the recurrence at S:483–523 is:

    D_h = B(S_h) * decay_h             (public decode only)
    memory_h[v] = sum_k D_h[k,v] * K_g[k]
    delta_h[v] = (V_h[v] - memory_h[v]) * beta_h
    Snext_h[k,v] = D_h[k,v] + K_g[k] * delta_h[v]
    core_h[v] = sum_k Snext_h[k,v] * (Q_g[k] * 128**-0.5)

Here B denotes BF16 rounding followed by widening for arithmetic. The orientation, shared-head grouping and scale match HF's recurrent fallback. The new state remains FP32 until next public call; rounding on consumption is equivalent to HF storing its BF16 cache, while preserving the current output's unrounded update. Internal prefill chunks/tails deliberately do not add public cache boundaries (S:1016–1030). Removing that boundary would be a regression.

_recurrent_sum at S:475–481 groups8×16 coordinates and reduces the128-key axis. N/experimental/reduction/fast_reduce_nc/device/fast_reduce_nc_program_factory.cpp:289–302 and kernels/reduce_nc.cpp:43–53 preserve FP32 through direct-to-DEST SFPU addition. This is not the old fused-delta-rule TF32 state-decay defect.

V5 targets its final add directly into state["recurrent"] (S:511,595–604), and omits the later decode copy (S:636–637). The state is read to a distinct decayed tensor before the write; the final core consumes new. Same-CQ program ordering gives no obvious read-after-overwrite contradiction. Existing recurrent-direct-state and persistent-RM probes report exact results, but cover bounded layer0/saved fixtures, not this layer34 trajectory.

Remaining uncertainty: equal final-output prefill PCC is not proof of equal prefill state. The stress harness saves no HF/V4/V5 prefix state for direct comparison. A layer-specific initial state error, projection mismatch, or local FP32 reduction difference can still cross later BF16 cache boundaries.

### 3. V5 gated norm: a verified HF rounding discrepancy

HF:191–200 and V4:643–675 have these intended dtype boundaries, with c and gamma BF16 and z the BF16 projected gate:

    n  = B(c * rsqrt(mean(FP32(c)^2) + epsilon))
    w  = B(FP32(n) * FP32(gamma))
    m4 = B(FP32(w) * SiLU(FP32(z)))

V5 S:652–672 invokes the KDA sigmoid-gated norm and then multiplies Z externally:

    n  = FP32_buffer(FPU(c * inverse_rms))
    w  = FP32_buffer(FPU(n * gamma))
    sig = FP32_buffer(sigmoid(z))
    g  = B(FPU(w * sig))
    m5 = B(FP32(g) * FP32(z))

This removes the first two BF16 boundaries, changes association, and inserts rounding before the external Z product. It is not numerically equivalent to HF despite the real-arithmetic identity SiLU(z)=z*sigmoid(z). Choosing native output_dtype=float32 would remove only the last intermediate BF16 boundary; it would not restore HF ordering.

Lowering:

- N/experimental/kda/sigmoid_gated_rms_norm/device/sigmoid_gated_rms_norm_program_factory.cpp:78–88 creates Float32 tmp/stats/inv/norm buffers; :136–145 selects UnpackToSrc for their consumers.
- Its kernels/compute/sigmoid_gated_rms_norm.cpp:51,69,85,104,137 uses FPU normalization/gamma/sigmoid-product stages. Ordinary FP32 FPU operands reload through TF32 source formats; see tt_metal/jit_build/genfiles.cpp:1069–1086.
- V4's external precise multiplies select BinaryNG SFPU/direct-to-DEST, preserving the explicit round_widen_chain at S:61–63.
- Both ordinary RMSNorm and fused norm receive self.compute: HiFi4, approximation false, FP32 DEST true, packer L1 accumulation false (S:53–59). RMSNorm's approximate defaults are overridden at N/normalization/rmsnorm/rmsnorm.cpp:69–83.
- Both use FPU reduction paths, the same BF16-truncating epsilon generator, and default rsqrt under approximation=false. V4 does SUM then scalar1/N; fused does AVG with exact scaler1/128. Do not describe V4 as full-FP32 SFPU reduction or blame a different approximate-rsqrt default.

The native layout is consistent. For head h, Mt=1 and Vt=4: reader_sigmoid_gated_rms_norm.cpp:45–62 computes x tile base4h and gate base64+4h. This selects Z columns2048+128h through2048+128h+127. writer_sigmoid_gated_rms_norm.cpp:17–29 writes four tiles per head in increasing order. It reduces channels per row, not rows across tokens.

E/v4_decode_native_norm_device.json already shows this is a small numerical rewrite: three seeds give head-BF16 versus V4 PCC .9999936784/.9999927301/.9999930533, relative L2 about .00393–.00412, and1906–1925 differing BF16 codes among4096. Head/row geometries have identical recorded comparison metrics; all18 consumed-output padding-sentinel checks pass, including poisoned unused rows. M/tests/v4_decode_native_norm_probe.py:165–181 constructs the comparisons and :224–268 constructs the sentinel checks. Its core/Z are synthetic and gamma is layer0's, so it cannot quantify actual layer34 sensitivity.

This is a concrete discrepancy worth isolating, but V4's shared severe failures falsify it as the sole root cause. It executes after recurrent-state update and cannot feed state drift in this harness.

### 4. Shared postnorm, router and experts: leading amplification hypothesis

S:976–981 adds attention residual, normalizes with post_attention_layernorm, then adds MoE. Decode postnorm input is BF16 width-sharded across32 cores of32×64. Its gamma is FP321+weight. N/normalization/rmsnorm/rmsnorm.cpp:70–83 lowers to RMSNORM; layernorm_device_operation.cpp:20–25 selects the sharded factory.

The sharded kernel packs/reloads the normalized intermediate and applies gamma via FPU row broadcast (N/normalization/layernorm/device/kernels/compute/layernorm_sharded.cpp:627–674). sharded_layernorm_factory_helpers.cpp:117–122,699–700 preserves actual gamma dtype; :1232–1256 gives ordinary FP32 consumers UnpackToSrc. Thus FP32 storage/accumulation does not establish HF's uninterrupted FP32 normalization/affine arithmetic. The32 width shards cover all2048 channels; there is no demonstrated wrong normalization axis.

HF:784–791 computes FP32 softmax over BF16 logits, top8, renormalizes selected scores and casts BF16. S:678–697 selects top8 BF16 logits then softmaxes selected scores. For the same expert set the full-softmax denominator cancels in real arithmetic; this reordering alone is not an algebraic bug.

Canonical native parameters are grouped=false (nanobind default), enable_sigmoid=false, topk=8, output_softmax=true, scale1, bias0. S:252–269,680–694 supplies one core with face0 containing256 logits, transposed global IDs, and distinct score/ID outputs. The descriptor builder passes those flags directly and computes num_blocks=(32/32)*(32/32)=1; N/experimental/deepseek/moe/generalized_moe_gate/device/generalized_moe_gate_program_descriptor_builder.cpp:63–64,128–154. The selected single-block kernel waits one input/bias/ID tile and produces one score and one ID tile (: corresponding unified_kernels/generalized_moe_gate.hpp:216–253). The same counts apply to passing128 and failing137/7944. No multiblock merge path or topk6 mask is involved.

Native score arithmetic differs from HF FP32 softmax: descriptor compute leaves fp32_dest_acc_en at its false default (tt_metal/api/tt-metalium/program_descriptors.hpp:103–111), and Blackhole's finalize helper stores shifted logits, exponentials and normalization intermediates through16-bit DEST. See tt_metal/tt-llk/tt_llk_blackhole/common/inc/sfpu/experimental/ckernel_sfpu_generalized_moe_gate_topk_single_face.h:695–713,767–797. Max subtraction is present, so raw-logit overflow from missing stabilization is not the identified issue.

The native gate explicitly permits tied rank8 membership to differ from Torch; models/common/tests/moe/test_generalized_moe_gate.py:194–207 tests ranking/score consistency instead of identical IDs. Equal selected logits do not make different experts interchangeable. Small upstream changes can also exchange near-tied8th/9th experts.

S:730–739 keeps IDs and scores paired and sums expert contributions. Merely changing their order is not evidence of dropped or misweighted experts. Expert arithmetic also differs from HF's per-expert BF16 index_add sequence (HF:755–771), and can matter even with identical expert membership. No actual layer34 route/score/expert tensor is saved, so do not declare all296 failures to be routing swaps.

The smallest intervention implied by this hypothesis is a diagnostic common-input/common-route replay, not a broad expert rewrite or relaxed PCC threshold.

## What the parameter survey can and cannot show

The gamma survey selects layer34 because its direct gated gamma max1.2890625 exceeds layer0's .95703125. All direct gated gamma values are BF16, positive and finite; both V4's FP32 upload and V5's BF16 upload represent them exactly (S:80–82,275). No missing +1, wrong gamma dtype or gamma>1 branch exists in this path.

A CPU-only read of small norm/A_log/dt_bias checkpoint vectors additionally found:

| Static descriptor | Layer0 | Layer14 | Layer34 |
| --- | ---: | ---: | ---: |
| Gated gamma RMS | .885739 | .898365 | .944870 |
| Effective input gamma entries not TF32-exact |1518/2048 |642/2048 |530/2048 |
| Effective postnorm gamma entries not TF32-exact |388/2048 |139/2048 |14/2048 |
| A_log min/max | −3.984375/4.28125 | −2.109375/3.078125 | −1.40625/1.46875 |
| dt_bias min/max | −5.28125/15.5625 | −7.21875/2.234375 | −5.59375/2.90625 |

Representability was checked by low13 mantissa bits of FP32 effective gamma. These counts are not measured device errors; in particular, the custom input norm uses precision-preserving SFPU arithmetic. Layer34's stronger postnorm scale and different learned dynamics/projections can affect routing margins and head sensitivity. None of these static descriptors establishes its runtime activation distribution or an accumulating-state cause. Gamma's maximum alone is insufficient to explain PCC, and V4 already uses the same actual layer34 gamma.

## Competing hypotheses and falsifying probes

| Candidate | Direct evidence / contradiction | Observations explained | Missing precondition and smallest discriminator |
| --- | --- | --- | --- |
| Small shared attention/postnorm error amplified by top-k membership or MoE arithmetic — leading hypothesis | Shared native Q/Z/postnorm/score arithmetic differs from HF; native tied membership need not match Torch | Sparse early failures, V4/V5 flips both ways, shared severe floor, layer-specific sensitivity | No layer34 logits/IDs/expert outputs. Capture137,165,181 plus neighbors; replay exact logits and then exact IDs/weights |
| Shared native Q/Z or output projection error, possibly amplified by gated norm | Q/Z uses native linear while K/V/A/B uses precise scan | Layer-dependent parameters, output-only changes, early onset without requiring state drift | Actual projections/core errors unknown. Compare on identical normalized inputs; compare pre/post gated norm per head |
| Layer-specific initial/recurrent state error from input norm, K/V/A/B projection or recurrence | Stateful path and different FP32 summation schedules exist; no shape/layout contradiction found | Shared V4/V5 failures and parameter-specific response; could be selectively amplified | No mapped prefix/incoming/next state. Compare rounded incoming state and next BF16 cache image, then one-step common-input replay |
| V5 gated-norm arithmetic mismatch — verified secondary discrepancy | Explicit BF16 boundaries/association differ from HF/V4; isolated probe differs | Extra V5 perturbations and changed failing positions | Cannot explain V4 failures alone. Extract tails on identical actual core/Z/gamma, then replay same downstream input |
| V5 direct state writer or RM carry bug — lower priority | Storage/output placement changed, bounded exact probes pass; no local alias/count contradiction found | Could add a V5-only state difference if actual stream violates a hidden contract | V4's severe failure persists without these changes. Compare V4/V5 state/carry bitwise after prefix and each early decode |
| Gamma maximum1.289 as sole cause — unsupported | Same gamma in failing V4; no source branch at1; parameter survey lacks activations | At most a possible sensitivity modifier | Hold core/Z fixed and compare gammas/rounding boundaries; scale swap alone is not proof of production causality |
| Missing sync, stale trace input, gradual endurance-only corruption — unsupported as stated | Blocking replay/readback, restores and three exact repeats; failure at tenth token | Does not predict actual early/sparse sequence without extra assumptions | Same-prefix eager versus traced single-step on independently restored state |

A shared low-precision discrepancy is not automatically acceptable. The unchanged0.995 gates fail materially; no threshold relaxation is proposed.

### Smallest useful experiment sequence for the parent

1. **Capture the early discriminators from the exact saved stream.** Preserve prefix128, seed154 and original saved inputs; run through positions136–138,164–166 and180–182. Position181 is a shared failure,137 a V5-only failure and165 a V4-only failure. Reusing the actual oracle tensors avoids accidental RNG/length changes. Use7944 as the second-stage severe-case check if early captures do not explain the failure family.
2. **Record a complete mapped boundary chain.** For HF/V4/V5, save normalized input; Q/Z and K/V/A/B projections; convolution carry and post-conv values; Q/K, beta, decay; consumed rounded incoming state; memory, delta, raw next state, next BF16 cache image and BF16 core; gated output, attention projection, residual, postnorm, logits, paired expert IDs/scores and final MoE/output. Include dtype/layout/padded shape, absolute position and head/column mapping.
3. **Separate local error from state history.** On identical captured operands replay one recurrence step; compare raw FP32 output and rounded cache codes. Independently compare native Q/Z against canonical projection and each packed state-projection channel on the same normalized input. This can falsify the favored downstream story if the core is already badly wrong.
4. **Isolate the V5 tail.** Apply extracted V4/V5 tails and canonical HF arithmetic to identical actual BF16 core/Z and layer34 gamma. Compare per-head RMS/relative error and merged BF16 codes before output projection. Hold output projection and MoE input fixed when attributing the change.
5. **Test routing and combination separately.** On identical BF16 logits compare selected expert sets, paired weights and rank8/9 margin. If sets differ, replay with one common expert set and scores. If the error persists, compare expert matmuls and combination on identical MoE input/IDs/weights. A recovery after forcing routes identifies a local contribution; it does not prove the earlier producer was wrong.
6. **Only pursue storage/trace changes if state diverges between V4/V5.** Compare cloned carry and rounded cache codes at prefix and early positions, then one eager decode versus a traced decode from separate identical restored states. Change one storage/norm feature at a time; the full V4/V5 comparison changes three features.

All are proposed follow-ups, not hardware experiments performed by this investigator. Keep experiments in fused-decoder/tests/docs and preserve original acceptance gates.

## Other potential issues and diagnostic hazards

1. **Incoming state capture aliases V5's overwritten output.** M/tests/routing_probe.py:79 stores delta_state_input=state without cloning; S:511,604 then writes into that persistent buffer. Readback after the call sees next state, not incoming state. Capture a clone before invoking recurrence and also record the BF16-rounded state actually consumed. This defect is not present in the stress metric path.
2. **Decay capture can be relabeled Q-products.** routing_probe.py:108–109 accepts any multiply with input_tensor_b_activations. S:514–518's final Q multiply has the scale activation and overwrites decayed_state. Restrict the capture to the actual state-decay producer. memory_products and reduced_output_products captured by _recurrent_sum's call ordinal are unambiguous (:88–96).
3. **Existing routing probe is layer0-specific.** It hardcodes weights(0), layer(...,0), decoder layer_idx=0 and cache layer0 (routing_probe.py:23–24,66,202). routing_probe.py:155–157 now captures separate K/V/A/B6208 and Q/Z6144 producers. A layer34 discriminator must update layer selection and prefix/state mapping, and reconstruct QKV/A/B/Z from those separate captures rather than assume projection_packed contains the former12352-column tensor; blindly running it does not inspect this failure.
4. **V4 determinism field is overbroad.** stress_fused_decoder.py:199 writes repeated_bit_equal=true even for one repeat. The final checker separately requires at least3 repeats; this V4 run is a diagnostic control, not a fresh qualification.
5. **Stress prefill gate is aggregate-only.** :109–110 checks flattened raw PCC without per-token/contribution or state checks. Passing prefill PCC cannot exonerate prefix state.
6. **Continuation device tensors are omitted from failure dumps.** :214–216 retains prefix/tokens/references/first-repeat decode outputs only.
7. **NaN failure predicate is incomplete.** :185 does not explicitly reject non-finite metrics; check_fused_evidence.py:73–75 does. All inspected tensors/metrics here are finite, so this does not explain these failures.
8. **Acceptance eligibility differs from diagnostic value.** check_fused_evidence.py:41–46 excludes a populated oracle-cache path; V4 also declares acceptance_eligible=false. These results remain valid same-oracle diagnostic contrasts, not evidence that either version passed final acceptance.

## Report review

After the first draft, claims were rechecked against the exact V4/V5 source diff, the completed V4 control, saved-tensor metric correspondence, HF dtype boundaries and native factory/kernel paths. Headline claims are limited to shared observed failure, the concrete V5 rounding discrepancy and diagnostic capture defects. The following attractive claims were explicitly rejected or demoted:

- Gamma>1, a missing +1, or gamma upload precision as a demonstrated root cause.
- “s257 passed, so the first137 stress failure cannot be numerical.”
- Full-FP32 V4 reduction versus TF32-only V5 reduction, or an approximate-rsqrt default accidentally selected.
- Unsorted top-k order by itself causing missing/misweighted experts.
- Native FP32 state automatically being consumed through TF32 in the current SFPU recurrence.
- An accumulating gated-norm feedback error in a harness with independent external inputs.
- Inferring identical layer34 state/route membership from matching final PCC or from older layer0 captures.

Independent xhigh post-draft reviews checked evidence/harness, upstream recurrence/projections, and native norm/router claims. Review corrections preserved Q scaling before contraction, distinguished native processing of padded rows from downstream row0 use, corrected the reduction source path, and limited geometry comparisons to the metrics actually recorded.

The exact earliest divergent calculation and the causal share of norm, projection, recurrent state and MoE remain unresolved without mapped layer34 intermediates. The report supplies targeted falsifiers instead of assigning that uncertainty to gamma.

