# AutoDebug: the 8192-token failure also occurs in the old control; shared precision boundaries remain unresolved

Inspection-only investigation, 2026-11-12. No target execution, target imports, device operations, resets, or source edits. Only this report was written. Checks used source reads, diffs, hashes, JSON summaries, and other cheap static inspection. The parent owns all hardware. Three isolated xhigh source audits covered frontend numerics, recurrence, and normalization/routing.

**Result:** no single root cause is proved by the available 8192-token artifacts. The old `validated_chunk4096.py.snapshot` control completed during this investigation and **also fails**, including six of the new candidate's seven bad positions. Therefore the new matmul state read, combined Q/K L2 call, carry API, native prefill gated norm, and sharded norms are not necessary causes of the common failure. They can still change individual token outcomes.

The strongest source-supported finding is a collection of **shared, earlier numerical discrepancies from the actual HF fallback**, beginning at input RMSNorm and continuing through gate projections, convolution, Q/K normalization, and output gated normalization. Native FP32 storage/accumulation does not ensure FP32 operand arithmetic in RMSNorm. A BF16 router cutoff tie can then change the selected expert set, even with identical logits. These are concrete discrepancies and plausible contributors, not a demonstrated causal explanation of positions 3979/7703. A matched intermediate capture is still needed to rank their runtime impact. Preserve the established BF16 recurrent-cache boundary exactly once per public call.

## Evidence and revision boundary

Repository root: `/home/ttuser/qwen36-35b-a3b/tt-metal`, HEAD `4c643884bf51a39fc1166e0b86b86bd373bedfee`. Below, `M` means `models/autoports/qwen_qwen3_6_35b_a3b/`, and `D` means `M/doc/fused_decoder/`; `S` means `D/mlp_norm_shard_dram.py.snapshot`; `V` means `D/validated_chunk4096.py.snapshot`. Model `tests/...` references are relative to M. Native operation paths abbreviate `ttnn/cpp/ttnn/operations/`; repeated factory/kernel basenames use their preceding operation directory. `tt_metal/...` and `models/common/...` paths are repository-relative. `HF` means the installed `python_env/lib/python3.12/site-packages/transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py`; `Cache` means the adjacent `transformers/cache_utils.py`.

| Artifact | Recorded source / result |
|---|---|
| [Failing snapshot](../mlp_norm_shard_dram.py.snapshot) | SHA256 `6c552444ffb96d002fdbdb8be37d07e55c380fba350ca9be33d9ee622c9d50fb`, exactly matching the failing run's sidecar |
| [Old control](../validated_chunk4096.py.snapshot) | SHA256 `42ac1e4b7a3a6556771b7fa97bdbc0439eb1545852aa1c8715e1a62f88681eec`, exactly matching its 8192 run's sidecar |
| [Candidate result](../endurance/mlp_norm_shard_8192_0_results.json) | Layer 0, seed 913, prompt 128, decode 8192, repeats 3, one capture; 21 failed rows = seven positions repeated three times |
| [Old-control result](../endurance/validated_chunk4096_8192_0_results.json) | Same layer, seed, prompt and decode lengths; **one** repeat, ten failed positions |
| Sidecars | Blackhole P300c chip 0, MeshShape(1,1), serial; Torch 2.11.0+cpu, Transformers 5.12.1; trace allocation tracking and program cache included |

The current `tests/reference.py` and `tests/stress_fused_decoder.py` hashes match the failing sidecar. The live model can be switched by the parent; all model claims here use the frozen snapshots. Native source references describe the inspected checkout; the sidecars hash model/test Python files, not the complete native build. The inspected operations/hardware source paths had no reported Git modifications.

The log confirms the Torch HF fallback, not the optional FLA/causal-conv fast path. The reference loads pinned checkpoint revision `345d467b117d1fd5eda9b879a5988de9ae5f47a7`, assigns checkpoint parameter dtypes, and feeds BF16 inputs (`tests/reference.py:14-20,46-61`; stress harness:31-37).

### Direct observations

| Candidate failing position | Output PCC | Contribution PCC | Old control contribution PCC |
|---|---:|---:|---:|
| 3979 | 0.995280057 | 0.993250277 | 0.988747965 |
| 5011 | 0.997229888 | 0.994590837 | 0.999780281 (passes) |
| 6595 | 0.995097537 | 0.994531127 | 0.993845370 |
| 7551 | 0.987617426 | 0.984317884 | 0.983082642 |
| 7668 | 0.991116415 | 0.989211603 | 0.989954716 |
| 7703 | 0.984020062 | 0.983477155 | 0.984173411 |
| 8157 | 0.995429067 | 0.994781857 | 0.994340474 |

- Candidate prefill PCC is `0.9999459169034807`. Its first 1024 decode outputs pass both thresholds: minima `0.9978808797444716` output / `0.9977203236604616` contribution. Its first 128 also pass. This directly establishes a short passing prefix in the **exact** failing run without relying on another snapshot's short gate.
- The candidate's three repeats are bit-identical, verified by `torch.equal`, not only equal PCC (`tests/stress_fused_decoder.py:110-113`). Every failing position repeats with identical metrics. The old 8192 control has only one repeat; no three-repeat endurance claim is made for it.
- Both continuations pass all 33 tokens, positions 8320-8352. Candidate minima are `0.9990500042739142` / `0.9989119695337052`; old minima are `0.9991700289822928` / `0.9991165092989898`.
- Old-only failures: 1849, 6474, 7082, 8287. New-only failure: 5011. Six common positions: 3979, 6595, 7551, 7668, 7703, 8157. Old worst contribution is `0.9830826419366696` at 7551; candidate worst is at 7703.
- Adjacent candidate positions 3978 and 3980 have contribution PCC `0.9997706220` and `0.9999295824`. These are isolated threshold failures, not evidence that the state suddenly becomes permanently corrupted at 3979.

The unchanged gate is `min(output_pcc, contribution_pcc) >= 0.995` for every token. Contribution is computed by subtracting the external BF16 input from both final outputs in the harness (`106-137`). It is an observed residual-contribution metric; the BF16 subtractions are not an exact decomposition of internal attention/MoE tensors.

### Interpretation limits

3979 is the first **threshold failure** in the candidate, not the first numerical divergence. Every decode call has the same shapes and program inputs apart from tensor values; 8192 is the number of calls, not a decoder buffer dimension. There is no layer-0 branch switching at token 4096 or 8192. Earlier shorter runs do not exercise the later input/state combinations.

The shared positions support looking first at common numerical paths. They do not prove a routing change, monotonic state drift, a particular head, or one dominant primitive. No position-matched 8192 intermediate capture is present in the named results. Historical position-237/876 probes and [hypothesis memos](../AUTOFIX_conv_hypothesis.md) predate the cache correction and cannot supply those missing measurements.

## Exact execution and state boundaries

The harness generates a fixed external prefix and fixed external decode tensors. It does **not** feed the previous decoder output back as the next input. Attention's recurrent/conv state persists; post-attention norm, router, and MoE only affect the current returned output. They can amplify a discrepancy locally, but cannot drive next-step recurrent-state drift in this test.

The harness warms decode, restores state, synchronizes before capture, records one decode, and restores state before each repeat (`stress_fused_decoder.py:64-91`). Each replay uses `blocking=True`, followed by device-to-host output conversion (`106-107`). Continuation output is also read before the final gate. Thus the reported numerical comparisons occur after completion, not merely after enqueue. Determinism does not exclude a deterministic native bug, but no concrete buffer/lifetime mismatch was found that explains the common positions.

For ordinary layer-0 decode, the effective chain is:

| Stage | Logical tensor / selected behavior |
|---|---|
| Input/post-attention norm | BF16 `[1,1,2048]`; current input is L1 width-sharded across 32 cores, each physical shard `[32,64]`; FP32 gamma; old uses interleaved norm |
| Packed projection | BF16 activation/weight, HiFi4, FP32 destination and FP32 `[1,1,12352]` output; explicit DRAM-sharded decode projection |
| Convolution | BF16 packed input padded to `[1,32,12352]`; Q/K/V widths 2048/2048/4096; BF16 RM history `[1,3,8192]`; logical end 1 |
| Q/K L2 | BF16 conv values widened to FP32; new batches Q/K as `[2,1,16,128]`, normalizes only last 128, slices halves |
| Recurrence | Q/K repeated to 32 value heads; state `[1,32,128,128]`; FP32 computation from BF16-representable consumed public state |
| Output norm/gate | FP32 core `[1,32,128]` into weighted RMSNorm, BF16 norm output and Z; fused SiLU/multiply; BF16 projection/residual |
| Router/MoE | BF16 256 logits; generalized gate selects eight paired IDs/scores; BF16 sparse expert outputs and weighted combine |

### Preserve the already-proven cache fix

`Cache:925-937` chooses dtype from convolution state and allocates recurrent cache in that dtype; `986-989` copies FP32 recurrence results into that cache. HF updates it once after recurrence (`HF:546-548`). With the actual BF16 reference, next-call consumed state is BF16.

Both snapshots now match that boundary: current decode `S:331-335`, old `V:310-314`; public prefill `S:713-716` / `V:658-661` rounds once, then internal segments pass `cache_boundary=False` (`S:726`, `V:671`). The current token's output uses unrounded new state. The persistent TT allocation remains FP32, so comparing its raw stored tensor directly to HF's BF16 cache overstates a semantic difference: compare **BF16-rounded consumed TT state** to HF cache.

Do not restore native FDR or add rounding between internal prefill chunks. The prior [AUTOFIX evidence](../AUTOFIX.md) established separate TF32 native-FDR and public-cache problems; neither is reopened by this report.

## Shared discrepancies, ordered from the earliest calculation

### 1. Input RMSNorm already has narrower arithmetic than its Python formula suggests

The formula is correct: HF normalizes in FP32, multiplies by `1 + weight.float()`, and casts to input dtype (`HF:823-831`). TT uploads that same FP32 gamma (`S:76-78`) and calls weighted RMSNorm (`S:241-255`). There is no missing `+1`.

However, the generic kernel normalizes with FPU operations, packs its normalized intermediate, and applies gamma with `BinaryFpu::Mul` (`normalization/layernorm/device/kernels/compute/layernorm.cpp:271-303,314-378`, under `ttnn/cpp/ttnn/operations/`). The planner's possible large-tensor variant has the same FPU normalization/affine boundary (`layernorm_large_tensor.cpp:459-479`); no claim depends on assuming a width-only large-tensor selection. The factory preserves gamma's FP32 dtype (`layernorm_op_multi_core.cpp:491-496,742`) and pins ordinary FP32 consumers to `UnpackToSrc`; the precision-preserving Welford aliases exclude RMSNorm (`617-646,1179-1241`). Metal2 maps `UnpackToSrc` to default unpack (`tt_metal/impl/metal2_host_api/program_spec/construction/kernel_lowering.cpp:140-143`), and the JIT's FP32-destination path uses TF32 sources (`tt_metal/jit_build/genfiles.cpp:1069-1086`, `data_format.cpp:147-155,211-219`). Thus FP32 intermediates/gamma can lose mantissa bits before multiplication. HiFi4/FP32 accumulation does not recover them.

The sharded RMSNorm also applies gamma with FPU row broadcast (`normalization/layernorm/device/kernels/compute/layernorm_sharded.cpp:632-666`). Its gamma format comes from the actual weight dtype and its ordinary FP32 consumers likewise use `UnpackToSrc` (`device/sharded_layernorm_factory_helpers.cpp:117-122,699-700,1232-1256`). The 32 width shards each cover 64 columns; the local/global mean factors multiply to `1/64 * 1/32 = 1/2048`, so this is not an incorrect per-shard normalization axis. Sharding can change reduction order, but does not establish exact HF FP32 arithmetic.

Static checkpoint-byte inspection confirms that gamma narrowing is non-vacuous: **1518/2048 input-norm gamma values and 388/2048 post-attention gamma values are not exactly TF32-representable** after forming `1 + BF16_weight.float()`. All 128 linear-attention norm weights are BF16 and therefore TF32-exact. I independently checked these counts by reading the safetensors index/header and only those BF16 arrays under `/home/ttuser/qwen36-35b-a3b/hf/hub/models--Qwen--Qwen3.6-35B-A3B/snapshots/345d467b117d1fd5eda9b879a5988de9ae5f47a7`, decoding each BF16 word, forming FP32 gamma, and testing its low 13 mantissa bits. No Torch/model imports were used. These are representability counts, not measured device errors. BF16 input values themselves are TF32-exact; the extra narrowing arises when FP32 computed intermediates/statistics or gamma are reloaded. The final BF16 pack can mask some differences.

The earliest plausible divergence is therefore already input norm, before projection or recurrence. Runtime impact at the bad tokens is unmeasured; this is a precision-contract difference, not proof of the root cause.

**Discriminator:** normalize the identical external BF16 token with exact HF FP32 arithmetic and each TT norm path; compare BF16 output codes. Project the identical normalized tensor through each backend separately. This separates input-norm rounding from projection arithmetic instead of attributing both to a low final attention PCC.

### 2. A/B projection and beta boundaries differ from HF

HF's A/B projection outputs are BF16; it computes BF16 sigmoid beta, but widens A before adding dt_bias and applying softplus (`HF:464-465,514-516`). TT requests FP32 packed projection and consumes FP32 A/B directly (`S:291,321-329`; same effective semantics in V). It omits HF's BF16 projection-result and beta-result rounding.

The gate equation/sign and beta placement match. New decode adds dt_bias only to the A slice, while old adds the mostly-zero packed bias before slicing; prefill still receives the packed bias through `linear` (`S:218-239,321-327`). There is no supported missing/double bias here.

Lowering checks exclude an assumed accidental fast gate: FP32/FP32 ADD/MUL select binary-ng SFPU (`eltwise/binary_ng/device/binary_ng_device_operation.cpp:51-65`); input and activation scratch use full FP32 destination unpack (`binary_ng_program_factory.cpp:1371-1400`). Softplus uses beta 1, threshold 20 and receives `INP_FLOAT32` (`unary/common/unary_op_utils.cpp:547-557`; `binary_ng_utils.cpp:639-643`). Default sigmoid is Accurate (`unary/unary_nanobind.cpp:1100`); Blackhole's FP32 implementations are in `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_softplus.h:212-255` and `ckernel_sfpu_sigmoid.h:35-49`. Polynomial approximation remains, without a demonstrated intrinsic defect.

**Discriminator:** from identical projected values, round raw A/B to BF16 before A widening/dt_bias and BF16-round sigmoid's output. Compare g/beta, then next state/core from identical consumed state. Do not merely cast the already bias-added A result: HF rounds before adding dt_bias.

### 3. Native convolution omits HF's BF16 pre-SiLU result boundary

The native compute accumulates all four tap matmuls in destination, applies SiLU in place, then packs BF16 (`experimental/kda/qkv_causal_conv1d_silu/device/kernels/compute/qkv_causal_conv1d_silu.cpp:64-81`; output dtype in `device_operation.cpp:190-200`). Its effective boundary is approximately `BF16(SiLU(sum_fp32(x*w)))`. HF fallback obtains BF16 `conv1d` output first, then SiLU (`HF:230-234,495`): `BF16(SiLU(BF16(sum(x*w))))`.

Both old and new APIs use this compute kernel. The validator rejects approximate math, and Blackhole SiLU uses its accurate sigmoid route (`device_operation.cpp:185-187`; Blackhole `ckernel_sfpu_silu.h:13-43`). This supports a rounding mismatch rather than an incorrect convolution formula.

**Discriminator:** use identical BF16 projected QKV/history/taps and compare preactivation, BF16 preactivation, and post-SiLU values. Introduce only the missing BF16 boundary in a parent-owned diagnostic control, retain the corrected cache policy, and compare Q/K/V plus per-head recurrence output. A final Q/K/V cast after SiLU does not test this boundary.

### 4. Q/K L2 has different rounding semantics; concatenation does not mix heads

HF performs BF16 square, returned BF16 sum, epsilon addition, rsqrt and final product before widening for recurrence (`HF:238-241,328-334`). TT widens first and uses RMSNorm with epsilon `1e-6/128`, gamma `128**-0.5` (`S:312-320`; `V:295-301`). The mean-to-sum transformation is algebraically correct, but the operation boundaries differ. Its FP32 RMSNorm also has the FPU/TF32 intermediate boundaries described above.

New concatenation is along batch, producing `[2,1,16,128]`; normalization width is the final dimension, 128 (`layernorm_op_multi_core.cpp:446-452`), with independent rows (`compute/layernorm.cpp:196`). I found no normalization across Q/K or across 2048 channels. Separate versus combined calls can still change scheduling/reduction rounding.

**Discriminator:** start from identical BF16 post-conv Q/K and compare HF's complete BF16-boundary L2, current RMSNorm, and a full FP32 SFPU sum-based L2. Then feed each result into a shared-input recurrence step. Casting only final FP32-normalized Q/K to BF16 does not reproduce BF16 squaring and reduction boundaries.

### 5. Output gated norm both changes boundaries and can amplify small-head error

HF returns BF16 core from recurrence (`HF:365`), widens it for unweighted RMS normalization, casts normalized values back to BF16 **before** multiplying BF16 norm weight, computes `SiLU(z.float())`, and finally casts BF16 (`HF:191-200`). Both TT snapshots pass FP32 core into weighted RMSNorm and cast only after gamma (`S:365-368`, `V:330-334`).

There is another lowered boundary: fused multiply's BF16 SiLU operand is activated and packed to a BF16 intermediate before multiplication (`eltwise/binary_ng/device/binary_ng_program_factory.cpp:1228-1241`). Canonical multiply defaults to `fast_and_approximate_mode=false` (`eltwise/binary/binary.cpp:1390-1416`), so even BF16/BF16 MUL selects SFPU in this checkout (`binary_ng_device_operation.cpp:63-64`). Tensor/tensor canonicalization preserves Z as physical RHS and its activation list (`binary_ng_device_operation.cpp:753-767`). The selected equal-shape kernel preprocesses RHS before loading it for multiplication (`compute/eltwise_binary_sfpu_no_bcast.cpp:49-74`); its helper copies, activates, then packs to the operand-format intermediate (`compute/eltwise_utils_sfpu.hpp:19-48`). This is a BF16 storage boundary, not an allegation of a FPU multiply on this path. HF keeps SiLU's FP32 result until the final product. The generic TT norm/gate is therefore not an exact HF substitute even when core values match.

Headwise normalization can give a small absolute core error in a low-RMS head substantial output weight. Prior position-876 substitution evidence demonstrates this mechanism for the **older cache-bug run**, not for the corrected 8192 run. Its use here is a reason to inspect per-head RMS/relative error, not a claim that the same heads or cause recur.

**Discriminator:** apply exact HF gated norm to saved TT core/Z, then project with identical weights; separately use HF core with TT gated norm. This distinguishes upstream core error amplified by normalization from norm/gate arithmetic itself.

### 6. A tied router cutoff can change experts; unsorted pairing is not a bug

HF computes FP32 softmax over BF16 logits, selects eight, renormalizes selected scores, then casts BF16 (`HF:784-791`). TT selects top eight BF16 logits then computes their softmax (`S:385-406`). BF16 decode-logit dtype itself matches HF. In real arithmetic these normalized scores agree for the same selected set; a common softmax denominator cancels. Native score normalization still has narrower storage: its descriptor leaves FP32 destination disabled by default (`experimental/deepseek/moe/generalized_moe_gate/device/generalized_moe_gate_program_descriptor_builder.cpp:149-154`; `tt_metal/api/tt-metalium/program_descriptors.hpp:103-111`), and Blackhole's single-face helper stores shifted logits, exponential sums and scores through 16-bit destination (`tt_metal/tt-llk/tt_llk_blackhole/common/inc/sfpu/experimental/ckernel_sfpu_generalized_moe_gate_topk_single_face.h:695-713,767-797`). This differs from HF FP32 softmax, without implying every SFPU temporary register is 16-bit. The shared MoE code keeps IDs and weights paired and sums expert contributions (`S:407-444`). Merely changing top-k order does not drop or misweight an expert.

Tie **membership** is different: native tests explicitly document BF16 cutoff ties selecting different experts from Torch and validate key multisets/score consistency instead of identical IDs (`models/common/tests/moe/test_generalized_moe_gate.py:194-207`). Different tied experts have different outputs despite equal gate values. Upstream perturbations can also swap nearly tied ranks eight/nine. This is a concrete mechanism for a sparse deterministic output discrepancy, but the 8192 result JSON does not record logits, rank gaps, or expert IDs. It cannot establish a tie or routing change at any bad position.

**Discriminator:** capture identical-position BF16 logits, the top 16 IDs/values, rank-8/9 gap, actual native IDs, and paired scores. Compare expert **sets**, preserving weights, not only ordering. Run the native gate and Torch selection on the exact same logits. Then replay expert computation with the same IDs/weights in both implementations. This separates upstream logit error, tie policy, score rounding, and expert arithmetic.

## Adjudicated alternatives and other potential issues

**New matmul read: real precision delta, insufficient common-cause explanation.** `S:281-283` replaces `V:264-265`'s FP32 SFPU `sum(decayed*k)` with `exp(g)*(k^T*state)`. Shapes are K `[1,32,128,1]`, K-transpose operand `[1,32,1,128]`, state `[1,32,128,128]`, and memory result `[1,32,1,128]`; orientation and scale match HF. Ordinary FPU matmul can narrow FP32 K to TF32 (`matmul/device/factory/matmul_multicore_reuse_optimized_program_factory.cpp:595-612`, JIT references above). **Consumed public BF16 state is already exactly TF32-representable**, so this does not repeat the old FDR's persistent-state narrowing at ordinary decode entry. Internal single-token prefill tails may consume unrounded FP32 state and are a separate exposure. The exact auto-selected matmul program/grid was not captured; the batched-reuse selection is source-derived, not runtime-verified. Old full-SFPU control failure demotes this from the common root cause.

**Shared addcmul rounding:** actual Blackhole `ckernel_sfpu_addcmul.h:37-59` uses MUL/MAD chains with FP32 output; HF materializes multiply then add (`HF:360`). FP32 binary operations and default FP32 reductions use precision-preserving SFPU paths (`binary_ng_program_factory.cpp:1371-1400`; `reduction/generic/device/reduce_op.cpp:143-153,198-217`; `reduce_op_multi_core_h_program_factory.cpp:604-611`). No original FDR-style TF32 state-decay/copy defect was found in this decomposition. Tiny arithmetic differences could cross later BF16 cache midpoints, but no saved code-level trajectory proves this. Compare rounded cache codes in shared-input replay; PCC alone misses scalar shrink or quantized-state differences.

**Convolution carry/stride adjudication:** both APIs lower to the same device op (`qkv_causal_conv1d_silu.cpp:55-164`). Input validation allows packed width >=8192. Factory derives `input_wt` from actual padded width (`device/qkv_causal_conv1d_silu_program_factory.cpp:295-308`): new 386 tiles, old 256; both convolve only `Ct=64+64+128=256` tiles. Reader addresses use `mt*input_wt+offset+ct` (`reader:180-189`), and carry staging uses the same input stride (`writer:74-81`). There is no supported Z/A/B bleed into QKV.

For every single-token decode, physical rows=32, `Mt=1`, block width=256 channels=8 tiles, 32 channel blocks/work items. `actual_end=1` makes valid rows=1. Carry assembly takes `[history[1],history[2],input[0]]` (`experimental/kda/device/kernels/chronology.hpp:58-63,112-118`; `history_exchange_qkv_causal_conv1d_silu.cpp:97-118`). Padding is not persisted. In compute, each group reserves/pushes `group` output tiles and waits/pops `4*group` activation tiles; each block builds and later pops 32 diagonal-weight tiles (`compute:21,37-87`). No token-count-dependent FIFO imbalance was identified: a passing decode and position 7703 have the same counts. HF's cache4 plus new token, followed by keeping the last conv output, consumes the newest three old tokens plus the current token; TT history3 is equivalent. Continuation33 passing covers one 32-token segment and a one-token tail, not all possible carry intervals or numerical states.

**Native prefill gated norm/new sharding:** they can alter initial state/output numerics and individual token results, but old control fails without them. Prefill output gated norm is after recurrent-state creation, so it does not itself corrupt the state carried into decode. No source contradiction was proved that warrants headlining these changes.

**Earlier failed boundary controls are confounded.** `gates_exact.py.snapshot`, `l2_exact.py.snapshot`, `conv_round.py.snapshot`, and `combined_round.py.snapshot` lack the corrected public BF16 consumed-state boundary. Their failed 1024 JSONs do not refute frontend-boundary changes on the corrected-cache 8192 baseline. Conversely, the earlier cache-fixed 1024 pass does not establish 8192 correctness. Source hashes and public-call policies must remain attached to each result.

## Candidate ledger

| Candidate | Direct source evidence / symptom coverage | Unverified condition and smallest discriminating intervention |
|---|---|---|
| Input norm and Q/K L2 precision | Shared FP32-to-TF32 reloads and HF rounding differences; can affect recurrent inputs in both failing controls | Actual bad-token/head error is missing; isolate exact shared-input normalization before changing recurrence |
| A/B/beta and convolution boundaries | Shared missing BF16 boundaries before persistent recurrence; compatible with both controls failing | No corrected-cache 8192 boundary A/B exists; change one boundary while preserving public cache semantics |
| Output gated norm | Shared differing dtype sequence; can amplify small-head error without feeding it back | Relative contribution is unknown; substitute exact HF norm with saved core/Z |
| Router ties/score precision | Shared documented cutoff tie policy and narrower score storage; can create isolated output spikes | Bad-token tie/route membership is unobserved; compare identical BF16 logits and freeze paired routes |
| EXP/addcmul/reduction rounding | Common recurrence arithmetic differs in rounding order, without a proved equation or state-storage error | No identical-input failure isolated; compare step intermediates and rounded next-cache codes |
| New matmul/carry/sharding/prefill gate | Present only in new candidate; six failures survive without them | Can explain only differential outcomes unless an independent shared cause exists; isolate only after shared-path localization |

No candidate is promoted merely because its maximum error could be quantization-sized. The first four rows have direct precision/tie discrepancies but lack failing-position intermediate evidence; the recurrence-rounding row lacks a demonstrated harmful numerical effect. None uniquely predicts the seven exact positions from source alone.

## Smallest useful next investigation, owned by the parent

1. **Capture one common failure and adjacent passes**, preferably 3978/3979/3980 and 7702/7703/7704, using the exact source hash/seed/token indices. Record actual consumed BF16-rounded recurrent state, projected QKV/A/B/Z, history, post-conv Q/K/V, normalized Q/K, g/beta, next state, pre-gated-norm core, attention output, post norm, logits and paired routes. Existing saved tensor files contain external inputs/final outputs, not these intermediate correspondences. Preserve exact dtype/layout metadata and per-head RMS/relative errors.
2. **Run a shared-input recurrence replay**: identical consumed state and Q/K/V/g/beta through HF equations, old SFPU read, and new matmul read. Compare decay, memory, delta, new state, core, and BF16-rounded next-state codes. Feed already-normalized Q/K to the bare HF equations or disable HF's internal L2; apply head repetition/query scaling once and do not normalize TT inputs twice. Then cross-substitute HF inputs versus HF pre-state. This distinguishes inherited prefill/history error, current frontend differences, and recurrence arithmetic. Keep the once-per-public-call cache rule in every variant.
3. **Test the earliest divergent boundary**, using the isolated controls above. If only full-model runs are available, independently test A/B/beta rounding, convolution pre-SiLU BF16, and full BF16 L2 on the cache-correct baseline before combining changes. For a norm discrepancy, compare exact shared-input FP32 normalization/gamma with current native paths. A final-output improvement alone does not prove which producer changed.
4. **Separate amplification** with exact HF gated norm on saved core/Z and identical-logit/identical-route MoE replay. Do not assume every bad output is a top-k swap; route membership, route scores and core/head errors must be measured independently.
5. **Retain the full original gate** on all 8192 decode tokens plus continuation33 and deterministic repeats after any repair. The old-control result removes a misleading passing contrast; it does not justify weakening 0.995 or advancing this unresolved layer to another pipeline stage.

## Draft review

The first draft was rechecked against the frozen models, the completed old-control JSON/sidecar, HF source, native factory/kernel contracts, and independent source audits. Two independent xhigh follow-up reviews verified the numerical table/cache/recurrence claims and the norm/router lowering. The review corrected a FPU-versus-SFPU multiply citation, covered both interleaved RMSNorm kernel variants, and made replay normalization/scaling boundaries explicit. Formula/axis/carry suspicions were demoted when invariants held. Headline claims are limited to reproduced overlap and concrete shared precision discrepancies. No hardware causal confirmation is claimed; no diagnosis relies on a historical probe being mapped to an unobserved 8192 intermediate.
