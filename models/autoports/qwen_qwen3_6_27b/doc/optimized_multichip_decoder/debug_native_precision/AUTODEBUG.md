# AutoDebug: native recurrent precision

## Finding and recommendation

**The leading explanation for the B1 zero-state, step-0 failure is TF32-class operand narrowing when the FPU readout re-reads the live FP32 rank-one state, amplified by cancellation.** FP32 buffers, FP32 destination accumulation, HiFi4, and `math_approx_mode=False` do not make this contraction a full-FP32 multiplication. The custom implementation explicitly avoids that narrowing with direct-to-Dest loads and SFPU arithmetic. Compact preparation retains the native contraction, so compact preparation alone does not remove this risk.

**A full-FP32 SFPU readout is the smallest justified first intervention. The available evidence does not establish that prediction or update must change to repair the earliest failure, or that readout alone will pass the complete recurrent trajectory.** At zero state, decay and prediction are identically zero; the native outer product can retain rank-one structure despite quantizing its two factors. Requantizing that matrix entry by entry for readout is a stronger explanation for a loss of correlation across value channels than a scalar Q/K normalization error.

There is also an **independent source-contract conflict in the prepared BF16 configuration**: CB8 is configured for full-FP32 direct-to-Dest unpack for persistence copying, yet is consumed by the FPU readout matmul. That configuration should be resolved in any production adaptation. It is not the cause of the observed public-FP32 failure, and its actual hardware effect is unverified here.

Keep BF16 persistent state, FP32 live readout before persistence rounding, and the existing **0.995 per-user and per-head PCC gates**. Do not replace them with aggregate PCC, lower the threshold, or promote FP32 persistence as a substitute for a numerical fix.

## Scope and evidence integrity

This was an inspection-only investigation in the already-isolated AutoDebug session. No AutoDebug launcher, TTNN import, device operation, build, test, or CPU numerical reproduction was run. Only source/artifact reads, JSON summaries, hashes, and this report were produced. Another investigator owns persistence reproduction and probe instrumentation; no source was edited here. Two xhigh inspection subagents independently audited native lowering and the custom/preparation contrast. No applicable ancestor `AGENTS.md` was found.

Repository root: `/home/ttuser/qwen36-27b/tt-metal`. In the following, `D` is `models/autoports/qwen_qwen3_6_27b/doc/optimized_multichip_decoder`. Source links point into this checkout; line numbers describe the inspection snapshot.

Direct observations and interpretations are kept separate:

| Direct artifact observation | What it establishes / does not establish |
|---|---|
| [prepared_native_wide22.log](/home/ttuser/qwen36-27b/tt-metal/models/autoports/qwen_qwen3_6_27b/doc/optimized_multichip_decoder/prepared_native_wide22.log:26) passes B1 normal and zero-QKV, then fails B1 zero-state step 0 at rank 1/user 0/head 6, PCC `0.9892162833519499`, threshold `.995`. | A real per-head failure in the public native FP32 control. It does not identify which intermediate first diverges. |
| [prepared_native_wide22.json](/home/ttuser/qwen36-27b/tt-metal/models/autoports/qwen_qwen3_6_27b/doc/optimized_multichip_decoder/prepared_native_wide22.json) saves both custom controls for that step: output minimum `0.9999999995199629`; state minima BF16 `0.9999999841999524`, FP32 `0.9999999999999972`. | Both custom arithmetic paths pass against their own dtype-matched FP64 trajectories. BF16 persistence is not intrinsically incompatible with this case. |
| The failing step has no saved public-native state metric or prepared-path metrics. | Public output validation stopped the original metric loop. **The prepared BF16/FP32 results for this failing case are unknown**, not failures or passes. All five paths had been invoked before the explicit synchronization and readbacks; the missing coverage is validation, not proof that their kernels never executed. |
| [native_recurrent_wide18.json](/home/ttuser/qwen36-27b/tt-metal/models/autoports/qwen_qwen3_6_27b/doc/optimized_multichip_decoder/native_recurrent_wide18.json) passes 18 cases, but contains per-rank/user metrics only. B1 zero-state step-0 native output minimum is `0.9999990713048387`; rank 1/user 0 has that same value. | Whole-user PCC can hide a small, weak head among 12 heads. This does not contradict the later per-head failure. No per-head number can be reconstructed from the saved aggregates. |
| [native_recurrent_perf19_v2.json](/home/ttuser/qwen36-27b/tt-metal/models/autoports/qwen_qwen3_6_27b/doc/optimized_multichip_decoder/native_recurrent_perf19_v2.json) fails B10 after 702 same-input updates on native-FP32 versus custom-BF16 output: minimum `0.9897577435687838`, first rank/user entry. | This is a separate trajectory/persistence-policy comparison. It cannot explain a zero-state first-step failure. |

The wide22 requested matrix was batches `[1,3,9,10,17,31,32]`, three modes, three eager steps, changed-input trace checks and 50 queued steps. Execution stopped in the third B1 case. Only B1 normal and zero-QKV completed; later batches are untested in this artifact. Wide18 covers `[1,3,9,10,17,32]` in all three modes under its older metric. Perf19_v2 reached B1/B3/B9 successfully and B10 incompletely; B17/B32 were not reached.

Revision check: wide22's prepared candidate, native candidate, custom source, recurrent variants, native compute kernel and factory hashes match the inspected files. Its probe hash was `84103adc1e894c46018c8ca603553c84ccc7076c3706a842526e5a5f99d4e19b`; the current probe has concurrent failure-collection changes. Its current line numbers and collector behavior are not retroactive runtime evidence. Perf19_v2's native probe still matches its recorded hash. Wide18's probe and the older runs' `multichip_decoder.py` hashes differ from today's versions; use their saved metrics as historical evidence, without claiming identical complete control flow across revisions. Native candidate/kernel/factory hashes remain consistent across these artifacts. The recorded hashes do not pin lower-level JIT/LLK headers or prove the exact executed binary; the lowering analysis below describes the inspected source.

## Lowered execution and precision

The mathematical target is, for each head,

```text
d = exp(g)
Sd = S * d
p[j] = sum_i k[i] * Sd[i,j]
u[j] = (v[j] - p[j]) * beta
L[i,j] = Sd[i,j] + k[i] * u[j]
y[j] = sum_i q[i] * L[i,j]
S_next = round_to_persistence_dtype(L)
```

Q and K are normalized with epsilon `1e-6`; Q is separately scaled by `128**-0.5`. Output uses the live update `L`, not the rounded persistent state. The FP64 oracle explicitly implements this ordering in [probe_prepared_native_recurrent.py](/home/ttuser/qwen36-27b/tt-metal/models/autoports/qwen_qwen3_6_27b/tests/probe_prepared_native_recurrent.py:135).

The public control starts with BF16 row-major packed `[B,1,2560]`, containing four Q heads, four K heads, and twelve V heads, each width 128. It normalizes widened Q/K, passes `scale=128**-0.5`, `use_qk_l2norm=False`, and uses persistent FP32 tile/DRAM wave-local rings with identity block indices. It explicitly requests HiFi4, exact-mode math and FP32 accumulation: [native_recurrent_candidate.py](/home/ttuser/qwen36-27b/tt-metal/models/autoports/qwen_qwen3_6_27b/tt/native_recurrent_candidate.py:60), [step](/home/ttuser/qwen36-27b/tt-metal/models/autoports/qwen_qwen3_6_27b/tt/native_recurrent_candidate.py:108).

The C++ [wrapper](/home/ttuser/qwen36-27b/tt-metal/ttnn/cpp/ttnn/operations/transformer/fused_recurrent_gated_delta_rule/fused_recurrent_gated_delta_rule.cpp:87) separately scales Q, splits heads, repeats each Q/K head three times, widens V/beta, computes decay, and creates one padded row per head. For T=1, recurrent inputs are FP32 tile/DRAM Q/K/V `[BH,1,128]`, decay/beta `[BH,1,1]`, state `[BH,128,128]`. The ring is passed through without a typecast or copy, and its output aliases the ring. The [factory](/home/ttuser/qwen36-27b/tt-metal/ttnn/cpp/ttnn/operations/transformer/fused_recurrent_gated_delta_rule/device/fused_recurrent_gated_delta_rule_program_factory.cpp:78) allocates FP32 arithmetic CBs, but its compute configuration has **no** direct-FP32 unpack modes.

This selects reduced-precision FPU operands: [genfiles.cpp](/home/ttuser/qwen36-27b/tt-metal/tt_metal/jit_build/genfiles.cpp:1069) chooses `Tf32` for the conditional unpack destination under FP32 accumulation with these formats. [data_format.cpp](/home/ttuser/qwen36-27b/tt-metal/tt_metal/jit_build/data_format.cpp:211) preserves `Float32` only for explicitly direct-to-Dest buffers. [base_types.hpp](/home/ttuser/qwen36-27b/tt-metal/tt_metal/api/tt-metalium/base_types.hpp:33) distinguishes FP32 Dest preservation from reduced SrcA/B precision. HiFi4 does not recover bits already discarded on that input path.

| Phase | Native kernel | Custom SFPU control |
|---|---|---|
| Decay | FPU scalar broadcast multiply; FP32 packed result | Separately rounded FP32 SFPU `S*d` |
| Prediction | Four K-tile matmuls into FP32 accumulator; state is narrowed on FPU input | FP32 products, ordered fold of four tile segments, then SFPU reduction of 32 rows |
| Residual | FPU `sub_tiles(v,p)`, beta copied from its CB, then SFPU multiply | Separate SFPU subtraction and beta multiply |
| Update | Copy decayed state into Dest, then accumulate FPU outer product; copy can also narrow FP32 input | Separate SFPU `k*u`, then add decayed state; no replacement with FMA |
| Readout | Re-read packed live FP32 matrix through FPU matmul input, narrowing its entries | Multiply live FP32 update by Q, fold four segments, then SFPU reduction |
| Persistence | Public ring is FP32; prepared BF16 branch casts only after live readout | BF16 RNE rounds a copy; readout retains unrounded live registers |

Native evidence: [mm](/home/ttuser/qwen36-27b/tt-metal/ttnn/cpp/ttnn/operations/transformer/fused_recurrent_gated_delta_rule/device/kernels/compute/fused_recurrent_gated_delta_rule.cpp:39), [update](/home/ttuser/qwen36-27b/tt-metal/ttnn/cpp/ttnn/operations/transformer/fused_recurrent_gated_delta_rule/device/kernels/compute/fused_recurrent_gated_delta_rule.cpp:63), [residual/decay](/home/ttuser/qwen36-27b/tt-metal/ttnn/cpp/ttnn/operations/transformer/fused_recurrent_gated_delta_rule/device/kernels/compute/fused_recurrent_gated_delta_rule.cpp:104), [recurrence/readout](/home/ttuser/qwen36-27b/tt-metal/ttnn/cpp/ttnn/operations/transformer/fused_recurrent_gated_delta_rule/device/kernels/compute/fused_recurrent_gated_delta_rule.cpp:186).

Custom evidence: [direct-FP32 unpack configuration](/home/ttuser/qwen36-27b/tt-metal/models/autoports/qwen_qwen3_6_27b/tt/optimized_decoder.py:4814), [decay/prediction](/home/ttuser/qwen36-27b/tt-metal/models/autoports/qwen_qwen3_6_27b/tt/optimized_decoder.py:5142), [update/readout and persistence-copy RNE](/home/ttuser/qwen36-27b/tt-metal/models/autoports/qwen_qwen3_6_27b/tt/optimized_decoder.py:5176), [ordered contraction helper](/home/ttuser/qwen36-27b/tt-metal/models/autoports/qwen_qwen3_6_27b/tt/optimized_decoder.py:5359), [current fused execution](/home/ttuser/qwen36-27b/tt-metal/models/autoports/qwen_qwen3_6_27b/tt/optimized_decoder.py:5483). The `cumulative_exp_scalar` variant changes preparation/scheduling and the exp execution extent, while retaining this recurrence arithmetic: [recurrent_variants.py](/home/ttuser/qwen36-27b/tt-metal/models/autoports/qwen_qwen3_6_27b/tt/recurrent_variants.py:33).

Public normalization is **not automatically a TF32 path** because it calls public operations. In this checkout FP32 multiply/add select SFPU ([binary_ng_device_operation.cpp](/home/ttuser/qwen36-27b/tt-metal/ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/binary_ng_device_operation.cpp:55)); the factory configures direct-FP32 unpack ([binary_ng_program_factory.cpp](/home/ttuser/qwen36-27b/tt-metal/ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/binary_ng_program_factory.cpp:1374)). Sum defaults to accurate mode and selects SFPU for eligible Blackhole FP32 inputs ([generic_reductions.hpp](/home/ttuser/qwen36-27b/tt-metal/ttnn/cpp/ttnn/operations/reduction/generic/generic_reductions.hpp:32), [reduce_op.cpp](/home/ttuser/qwen36-27b/tt-metal/ttnn/cpp/ttnn/operations/reduction/generic/device/reduce_op.cpp:198)). Unary FP32 inputs request precision preservation ([unary.cpp](/home/ttuser/qwen36-27b/tt-metal/ttnn/cpp/ttnn/operations/eltwise/unary/unary.cpp:56)). These general-operation sources are not hashed by the old reports, so this is a current-source dispatch finding, not a claim about every historical build.

## Why readout is the leading first-step explanation

With finite inputs and `S=0`, `Sd=0` and `p=0`. Thus `u=beta*v` and the exact output is `y = (q dot k)*u`. BF16 and FP32 persistent initial states coincide; neither a long recurrence nor a previous BF16 store is involved. Increasing decay/prediction accuracy cannot restore information lost in this step because those terms are zero.

Let `Qa`, `Qb` denote the reduced operand formats of physical FPU SrcA and SrcB. An idealized zero-state native update is

```text
Lhat[i,j] = Qb(kcol[i]) * Qa(u[j])
```

Here `kcol` includes the native transpose's source-format rounding. Quantizing the two vectors independently still gives an outer product. The checked-in [Src-format/fidelity model](/home/ttuser/qwen36-27b/tt-metal/tt_metal/tt-llk/tests/python_tests/helpers/golden_generators.py:921) truncates FP32 to TF32 and applies four phase masks ([masks](/home/ttuser/qwen36-27b/tt-metal/tt_metal/tt-llk/tests/python_tests/helpers/golden_generators.py:1066)); the effective factors have at most eleven and ten significant bits. Their scalar products fit FP32 for ordinary finite exponent ranges; the inner dimension is one live value, with zero padding. This is a model-based representability argument, not a measured reconstruction of this head's hardware products. Therefore **the existence of quantization in the update alone is not proof that update must be replaced to fix PCC**. In exact arithmetic, contraction of this matrix remains a scalar times the quantized value vector; SFPU contraction introduces its own smaller FP32 rounding. A positive scalar amplitude error alone leaves PCC unchanged, though a sign flip or an almost-zero scalar remains a concern.

The existing readout instead computes approximately

```text
yhat[j] = FPU_sum_i Qb(q[i]) * Qa(Lhat[i,j])
```

`Qa(Lhat[i,j])` generally cannot be factored into a row-only and a column-only factor. It introduces entry-dependent rounding. If `q dot k` is small from cancellation, the desired signal is small while the rounding contributions can vary by channel and need not cancel. This predicts an isolated head with poor PCC even when its enclosing user's PCC is excellent. The matrix is the logical second matmul operand and maps to hardware SrcA; the API explicitly swaps logical in0/in1 onto physical SrcB/SrcA: [llk_unpack_AB_matmul_api.h](/home/ttuser/qwen36-27b/tt-metal/tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_unpack_AB_matmul_api.h:86).

This mechanism is source-supported, **not numerically confirmed for rank 1/head 6**: the report does not save that head's Q/K dot product, live state, or individual output vector. No missing tensor correspondence has been invented. FPU accumulation order, residual/value-factor rounding, and update imperfections remain possible contributors. A readout-only controlled replacement is the falsifier: it must improve the failing head while leaving the native update/state unchanged. If it does not pass, inspect the existing live state before changing unrelated stages.

For nonzero state and later steps, the rank-one simplification no longer holds. Native decay narrowing, prediction contraction, residual subtraction, the decayed-state seed copy, and the outer update can all perturb the persisted matrix and feed error into later predictions. An SFPU readout cannot reconstruct those lost state bits. This establishes a remaining risk, not proof that every phase needs replacement to satisfy the gates.

## Compact preparation, ownership and completion audit

The prepared candidate copies the native helper definitions verbatim and replaces only the main scheduling/persistence loop: [recurrence_source](/home/ttuser/qwen36-27b/tt-metal/models/autoports/qwen_qwen3_6_27b/tt/prepared_native_recurrent_candidate.py:301). Its first launch widens BF16 QKV by shifting the stored bits, normalizes four Q plus four K vectors with the same SFPU square/four-tile fold/reduce/epsilon/rsqrt/normalization/separate-Q-scale sequence as the custom path, and stores a compact FP32 row record:

```text
floats 0..511       normalized/scaled Q
floats 512..1023    normalized K
floats 1024..2559   widened V
floats 2560..2571   exp(g)
floats 2572..2583   beta
floats 2584..2591   zero padding
```

See [preparation](/home/ttuser/qwen36-27b/tt-metal/models/autoports/qwen_qwen3_6_27b/tt/prepared_native_recurrent_candidate.py:93), [record writer](/home/ttuser/qwen36-27b/tt-metal/models/autoports/qwen_qwen3_6_27b/tt/prepared_native_recurrent_candidate.py:165). The payload is `2592*4=10368` bytes/user. The second launch obtains `user=h//12`, `hv=h%12`, `hk=hv//3`, restores row tiles, and executes native helpers. BF16 state is expanded locally in L1, the live FP32 update is read out, then persistence is rounded with the FP32-to-BF16 typecast. There is no required full-state DRAM widening/narrowing pass.

Concrete failing geometry: B1 has 12 heads, T=1, `Kt=Vt=4`, 16 state tiles/head. Head 6 selects grouped Q/K head 2. Its prepared Q/K/V byte offsets are respectively `1024`, `3072`, `7168`; its scalars are decay index 6 and beta index 18 in the 32-float tail. Public native maps head 6 to core `(0,6)` on the 11x10 grid; prepared maps it to `(6,0)`. These are different valid one-owner mappings; no collective routes recurrence data between ranks.

At B10 the public owner uses waves of 9 and 1 users, hence 108 and 12 native cores, each processing one head. Prepared assigns 120 heads to 110 cores: ten cores get two heads and 100 get one. B17 assigns two heads to 94 cores and one to 16; B31 assigns four to 42 and three to 68; B32 assigns four to 54 and three to 56. Those later prepared geometries were not validated by wide22.

Per prepared head, the source count ledger is balanced: reader produces state 16, Q/K/V 4 each, and decay/beta 1 each; compute consumes initial state/decay, produces and consumes decayed state 16, prediction 4, residual 4 and transposed K 4; update produces live state 16 and persistence 16; readout produces output 4. Writer consumes output 4 and persistence 16 after write barriers. Compute explicitly pops the live state 16 before the next independent head ([REC_MAIN](/home/ttuser/qwen36-27b/tt-metal/models/autoports/qwen_qwen3_6_27b/tt/prepared_native_recurrent_candidate.py:260)). Thus a two-head B10 core does not inherit the native multi-token loop's retained state. No count/ownership contradiction explaining the first public-native numerical failure was found.

The probe invokes all paths and explicitly synchronizes the mesh before reading outputs/states and checking metrics. In perf19_v2, queued replays also have a final synchronization/readback; timing batches synchronize before/after replay loops. `702 = 2 + 7*100` updates follows an explicit reset and a fixed input bank entry: [timing loop](/home/ttuser/qwen36-27b/tt-metal/models/autoports/qwen_qwen3_6_27b/tests/probe_native_recurrent.py:350). These are completed numerical comparisons, not assertions made merely after enqueuing work.

## Minimal production adaptation and decisive checks

1. **Implement the first discriminator only at T=1 readout.** Replace the final `mm(q,live,...)` in the prepared recurrence with FP32 direct-to-Dest copies, SFPU products, an ordered fold of the four K tiles, and SFPU column reduction. The custom `contract()` helper supplies a concrete numerical topology. Q must be represented as column tiles for this helper: the compact reader can materialize Q in that form, or a separate full-FP32 transpose stage can create it. Do not assume the current row-tile Q fits the helper's broadcast contract.
2. **Configure every SFPU-loaded FP32 readout operand for direct unpack, with explicit phase transitions.** In the T1 prepared candidate, CB0 (Q) and CB8 (live state) can use direct-to-Dest once their FPU readout consumer is removed. An SFPU multiply after an ordinary narrowing copy is insufficient. Keep remaining FPU input CBs SrcA/B compatible. The original startup uses CB0 as its format source, and the BF16-expansion branch restores SrcA from CB0. Both must be reconsidered when CB0 becomes direct-only: initialize from the first FPU phase's compatible buffers, explicitly establish readout's direct-copy formats, and restore compatible FPU formats before the next head's decay/prediction/update. The copied native assumption that no per-op format reconfiguration is needed no longer holds. [Startup's contract](/home/ttuser/qwen36-27b/tt-metal/tt_metal/hw/inc/api/compute/compute_kernel_hw_startup.h:31) requires matching the first operation or explicitly reconfiguring; `state_configure` cannot be relied on when the [compute sentinel is disabled](/home/ttuser/qwen36-27b/tt-metal/tt_metal/hw/inc/api/compute/sentinel/compute_kernel_sentinel.h:272). Do not mark every CB direct-to-Dest. Do not reuse a direct-only Q-column buffer as the existing FPU K-column input without separate format ownership. In the public T>1 kernel, CB8/CB9 also feed the next token's FPU decay; this T1-only CB-mode change is not directly transferable there.
3. **Preserve state semantics.** Keep BF16 persistent storage and exact local expansion, perform readout from the live FP32 matrix, and round only the persistence copy with RNE. Keep compact FP32 output and existing ownership/drain/barrier behavior. Do not read from a newly BF16-rounded matrix to simplify readout. Production code should use maintained helpers or explicit kernel variants rather than rely on runtime source-text splitting.
4. **Use complete existing gates to decide whether more precision is necessary.** Record all five paths at the original zero-state failure and verify unchanged state for a readout-only comparison. Then require per-head and per-user `.995` for output and state against dtype-matched FP64 oracles, cross-controls, normal/zero cases, changed inputs, later steps, multihead-core boundaries, and trace/queued execution. Retain existing absolute-error checks wherever the surrounding suite applies them. No gate relaxation is justified by an aggregate pass.
5. **If a later state or output gate fails, isolate the remaining recurrence phases.** First distinguish inaccurate readout of the same live state from an already-inaccurate state. A prediction replacement needs precise operands and SFPU contraction; retaining FPU residual subtraction can still narrow that new prediction. Likewise, a precise update needs a precise decayed-state seed and separate SFPU multiply/add. For a conservative complete adaptation, reuse the custom FP32 SFPU decay/prediction/residual/update/readout order inside the compact prepared ownership path, with BF16 persistence. The existing custom controls provide the best numerical reference; the newly combined two-launch implementation would still require validation and performance measurement. Do not claim a readout-only or full-SFPU compact variant has already passed.

The 702-update observation deserves a separate check. At B10/rank0/user0 the saved dtype-matched oracle PCCs are:

| Path | Output PCC | State PCC |
|---|---:|---:|
| Custom BF16 | `0.9999999138872465` | `0.9999999927905696` |
| Custom FP32 | `0.999999999998453` | `0.9999999999998987` |
| Native FP32 | `0.9985420973641315` | `0.9999159293167666` |

The native-versus-BF16-custom output PCC is `0.9897577435687838`. Its two paths follow different persistence trajectories, so excellent agreement with different oracles does not guarantee excellent mutual agreement. PCC gates are not transitive. Native also has measurable arithmetic drift from its own FP32 oracle. The artifact alone does not apportion the cross-control gap between that drift and BF16-versus-FP32 persistence. Readout-only changes leave the native persistent trajectory unchanged, so they cannot be assumed to repair this comparison. The other CPU investigator's persistence work is the appropriate discriminator; it was not duplicated here.

## Other potential issues and test gaps

**Prepared BF16 CB8 mixes incompatible consumption contracts.** [step configuration](/home/ttuser/qwen36-27b/tt-metal/models/autoports/qwen_qwen3_6_27b/tt/prepared_native_recurrent_candidate.py:399) marks CB8 `UnpackToDestFp32`, and [REC_MAIN](/home/ttuser/qwen36-27b/tt-metal/models/autoports/qwen_qwen3_6_27b/tt/prepared_native_recurrent_candidate.py:280) passes it to FPU `mm` before persistence copying it. JIT infers `Float32 -> Float32` for that CB. The low-level format contract allows this conversion only when unpacking to Dest ([cunpack_common.h](/home/ttuser/qwen36-27b/tt-metal/tt_metal/tt-llk/tt_llk_blackhole/common/inc/cunpack_common.h:558)), while matmul consumes SrcA/B. Actual format reconfiguration and hardware consequences were not measured. Earlier prepared normal/zero-QKV passes are evidence against claiming an unconditional catastrophic failure; the conflict still needs removal or explicit compatible staging. The public FP32 path does not set this mode and therefore cannot fail for this reason. SFPU readout removes the conflicting consumer naturally.

**Tests can miss cancellation-sensitive heads.** Native operation utilities explicitly document TF32-class input truncation and apply whole-tensor PCC plus a global maximum-relative-error guard ([gdn_decode_test_utils.py](/home/ttuser/qwen36-27b/tt-metal/tests/ttnn/unit_tests/operations/transformers/gdn_decode_test_utils.py:47), [assert_accuracy](/home/ttuser/qwen36-27b/tt-metal/tests/ttnn/unit_tests/operations/transformers/gdn_decode_test_utils.py:125)). Their random-input matrix includes zero state, but it does not establish the prepared probe's per-head contract on this seed or 702 repeated updates. A numerically stricter aggregate threshold still does not imply each head passes. Compatibility digests also intentionally freeze current arithmetic; a deliberate precision variant should have distinct coverage rather than silently reinterpret old bitwise expectations.

**No proven compact-preparation corruption was found.** The current prepared reader/writer offsets, grouped-head selection, payload sizes, normalization sequence and output segments agree at the inspected boundary. Public/prepared normalization can still differ in reduction topology or exp/rsqrt approximation details; these are weaker unresolved alternatives, not established root causes. No low-level timing/scoreboarding violation was established by this inspection. The leading diagnosis is intended source-format narrowing interacting with a stronger numerical contract, not proof of a faulty FPU instruction implementation.

## Review of headline claims

The draft was checked again against the source: zero-state excludes recurrence prediction/decay contribution; no BF16 persistent round precedes the first readout; native FPU input narrowing is selected by JIT configuration; custom loads explicitly preserve FP32; a quantized outer product need not itself destroy rank-one PCC; the readout does requantize matrix entries; the original assertion hides prepared metrics; and the BF16 CB8 conflict is confined to that prepared configuration. Claims that prediction/update are already proven necessary, that normalization is automatically TF32, that the prepared failing case is known, or that the later cross-dtype failure has the same demonstrated cause were rejected. A third independent xhigh reviewer checked the draft against the source/artifacts; its startup/format-transition correction was incorporated, and its final review found no remaining material corrections. Runtime confirmation of the cancellation hypothesis and the smallest gate-passing adaptation remains with the parent/device owner.
