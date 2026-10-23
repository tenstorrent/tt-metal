# AutoDebug: long-context linear-attention numerical failure

## Verdict

**The leading device-side finding is repeated TF32 truncation inside a recurrence advertised as carrying FP32 state. FP32 tensor storage and `fp32_dest_acc_en=True` do not preserve FP32 state through this kernel's unpack paths.** The kernel decays the old state through SrcA/SrcB, spills it, then copies the decayed state back through the default TF32 unpack path before adding the next update. This happens every token, including inside a 128-token call. It is a concrete precision defect, with a credible accumulation mechanism and the correct short-versus-long exposure pattern. Its contribution to the exact reported PCCs is **not hardware-verified**.

**The HF comparison also has a separate, confirmed state-precision mismatch:** installed `DynamicCache` stores recurrent state in **BF16**, inherited from the convolution cache. The long-reference helper introduces sixteen intermediate cache writes before evaluating the final 128 tokens; the short-reference helper introduces none before its prefill outputs. TT keeps its persistent recurrent tensor in FP32. Thus the long test changes both context length and the reference's rounding schedule. It cannot isolate device accumulation from reference-cache rounding.

No missing prefix, convolution-history offset, incorrect GQA grouping, final-token padding update, or reader/writer count mismatch was established. The report does **not** conclude that TF32 is the sole cause, that HF is necessarily the numerically better oracle, or that a passing second decode proves recovery of the state. Preserve the .995 acceptance threshold while resolving these concrete precision differences.

## Scope, source identity, and direct observations

Inspection date: 2026-10-23. Repository HEAD: `0611d6e393882ce28e811b023b5f7d23bba1aced`. The autoport is untracked at this HEAD, so HEAD alone does not identify its source. No target, model, hardware test, reset, or implementation edit was run. Read-only `docker exec` read the installed HF source. Static source inspection and small Python standard-library metadata/dimension/bit checks were used. Installed package metadata confirms Transformers 5.12.1 and no installed `flash-linear-attention`, `fla`, or `causal-conv1d` distribution, consistent with the Torch reference path. No other Qwen TT implementation was inspected.

Source abbreviations used below:

- **FD**: [tt/functional_decoder.py](../../../tt/functional_decoder.py).
- **TEST**: [tests/run_decoder.py](../../../tests/run_decoder.py).
- **WRAP**, **FACTORY**, **VALIDATE**, **COMPUTE**, **READER**, **WRITER**: the public `.cpp`, device program factory, device operation, and three kernels under `ttnn/cpp/ttnn/operations/transformer/fused_recurrent_gated_delta_rule/`.
- **HF**: installed `/opt/venv/lib/python3.10/site-packages/transformers/models/qwen3_5/modeling_qwen3_5.py` in `qwen38-functional-build`.
- **CACHE**: installed `/opt/venv/lib/python3.10/site-packages/transformers/cache_utils.py` in that container. HF/CACHE line references refer to these installed files, **not** the saved `hf_modeling_reference.py`.

| Inspected artifact | SHA-256 |
| --- | --- |
| FD | `b7884e9d9f4019c8a4973bce5fa13344b5fec814ddaceb8d774da2a18e3950e3` |
| TEST | `369eec392644641b6be0e98b45dfa74652e8ff92d46ec30b76424c27f21c04b6` |
| Installed HF | `395439341ea5ba4dd14c103e830383caa75686cfaf6b693dd7efb58622224da6` |
| Installed CACHE | `2783b2702910dbb6e70aa05a093f71e1c51c110d7fa73c904fc38d12e5882545` |
| Earlier gate-contract AUTODEBUG.md, preserved | `24fe15914a211447fba38d9922ddb3328cea6c3c887f25b06b354780ac80d748` |

The failing log is timestamped approximately 13:02 UTC; at initial inspection FD's modification time was 13:00:49 and TEST's was 13:03:54. The traceback's assertion is at line 170, whereas the initially inspected TEST assertion is at 224. **Even the initially inspected TEST is not an exact archived reproduction revision.** Its final metrics and execution structure agree with the log, but unlogged options, especially `--debug-states` and profiling, are not evidence that those paths ran.

**Concurrent-source warning:** external work changed both Python files during this investigation. A later read at FD mtime13:13:30 showed `_delta` calling a newly imported `.fp32_recurrence.fp32_recurrence` instead of the fused op; its hash was `de8e6b1aba798932723251372c2ef0baed0e321acac2af46114048b70b76be20`. TEST at mtime13:12:00 had hash `4e2dee710f5c1ff32c9a9072faef6ebb6687c01ae06a19c1d270db57d6083cc5`. These edits were **not made by this investigation**. All FD/TEST line numbers and fused-call conclusions below refer to the initial hashes in the table. The replacement recurrence is not reviewed or certified here; live links can now show shifted lines or changed calls. The failing artifact remains the subject, and no new result is inferred from the replacement's existence.

Recorded results, not reruns:

| Artifact | B | S | Prefill PCC | First decode PCC | Changed-input second decode | Resume |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `linear_smoke.json` | 1 | 32 | .9999571443 | .9999789596 | not recorded | not recorded |
| `linear_129.json` | 1 | 129 | .9996660948 | .9998660684 | not recorded | not recorded |
| `linear_b32_129_resume_fixed.json` | 32 | 129 | .9997943640 | .9996516705 | .9993154407 | 1.0 |
| `linear_16385.json` | 1 | 16385 | **.9929262996** | **.9906601310** | .9993519783 | not run |

All record deterministic replay. The long log identifies Blackhole, reports the Torch fallback warning, and has identical eager/trace first-decode PCC. Its prefill comparison covers only the last 128 outputs. These are whole-decoder outputs after projection, residual, normalization, and MLP, not isolated recurrent-state measurements. There are no per-head gates, intermediate errors, or state PCCs in the supplied long result.

## 1. Leading finding: state precision is lost on every fused step

### The actual lowered path

FD:51–54 passes HiFi4, `math_approx_mode=False`, `fp32_dest_acc_en=True`, `packer_l1_acc=True`. The Wormhole-named configuration is also valid on Blackhole: `operations/core/compute_kernel/compute_kernel_config.hpp:36–39` aliases the configurations. FD:145–159 supplies FP32 Q/K/V/log-decay/beta and FP32 initial state; memory is interleaved DRAM.

WRAP:85–114 scales Q by `1/sqrt(128)`, makes inputs head-major, repeats each of sixteen Q/K heads three times to match 48 value heads, exponentiates **log** decay once, and relayouts to per-token tiles. It does not interchange g and beta. VALIDATE:88–99 requires the resulting shapes and FP32 dtype. The single selected program factory uses one core per batch/value-head pair.

FACTORY:78–101 allocates every arithmetic CB as Float32. However, its `ComputeConfigDescriptor` at :153–157 omits `unpack_to_dest_mode`; the descriptor's vector is empty by default (`tt_metal/api/tt-metalium/program_descriptors.hpp:103–111`). `tt_metal/impl/program/program.cpp:559–567` copies it unchanged to ComputeConfig, and `tt_metal/impl/kernels/kernel.cpp:1070–1075` copies it to the JIT options. The actual JIT conversion is:

1. `tt_metal/jit_build/genfiles.cpp:999–1005`: FP32 accumulation plus all-FP32 CBs selects conditional unpack destination **Tf32**.
2. `tt_metal/jit_build/data_format.cpp:215–221`: only an explicitly nondefault unpack-to-destination mode overrides that with Float32. This factory supplies none.
3. Blackhole `tt_metal/tt-llk/tt_llk_blackhole/common/inc/cunpack_common.h:311–329,339–340` distinguishes FP32 in L1, TF32 in SrcA/SrcB, and full FP32 direct-to-DST. TF32 has ten fraction bits. The LLK source-format golden explicitly truncates thirteen FP32 mantissa bits at `tt_metal/tt-llk/tests/python_tests/helpers/golden_generators.py:883–917`.

The issue is therefore not inferred from a tensor's dtype label or from a generic belief about matrix hardware. It follows from the selected descriptor and generated formats.

### Where truncation enters the recurrence

The intended equations agree with HF: `D = exp(g) * S`, `u = beta * (v - kᵀD)`, `S_next = D + k uᵀ`, `o = qᵀS_next`.

| COMPUTE phase | Actual precision exposure |
| --- | --- |
| :186–192, helper :132–144 | `mul_tiles_bcast_scalar` reads both old S and decay through TF32 SrcA/SrcB, then packs FP32 D. The old state and near-unity decay have already lost bits. |
| :194–197, helper :39–56 | `k @ D` reads TF32 operands and accumulates into FP32 DST. |
| :199–206, helper :104–128 | FPU subtraction consumes reduced-precision v/read operands; beta's `copy_tile` also uses the default conversion before the SFPU multiplication. |
| :213–218, helper :71–88 | `copy_init(D); copy_tile(D,...)` seeds DST from a **TF32-truncated** D; the rank-one matmul then accumulates into that seed. |
| :220–225 | Q reads the updated state through the same matrix-input precision. The state survives the token in a Float32 CB, not permanently in DST. |

The validation comment/assertion at VALIDATE:47–53 says state is carried across tokens in FP32 DST. COMPUTE:180–225 instead carries it in ping/pong CBs and reloads it every token. HiFi4 and FP32 accumulation improve arithmetic **after** these input conversions; neither recovers discarded mantissa bits. `math_approx_mode=False` does not select full-precision unpacking.

This is exactly the accumulator-reload hazard documented in `tech_reports/op_kernel_dev/accuracy_tips/accuracy_tips.md:58–77`. A simple symbolic contract check isolates it: take g=0, beta=0, finite zero Q/K/V, and a state element `1 + 2^-12 = 1.000244140625`. The recurrence should preserve that element. The configured FP32-to-TF32 decay operand path truncates it to 1.0 even though input, output, and DST are labeled FP32. This is a source-derived counterexample, not a measured device run.

For an illustrative slow-decay scalar `g=-0.0001`, `exp(g)≈.999900005`, but TF32 truncation gives `.99951171875`. Ignoring all updates and all other rounding, retention after 16385 steps is approximately .19427 versus .00033464. **This is not a model simulation or evidence that the checkpoint produces that g.** It demonstrates why quantizing decay near one can change memory lifetime, rather than merely cause a one-time small output perturbation. The supplied weight means/standard deviations do not establish any particular per-head effective gate.

### Causal confidence and smallest intervention

The same loss occurs in every case. S129 has only 129 sequential updates; S16385 has 16385. Batch32 adds independent states rather than lengthening any state trajectory. That is consistent with short cases passing and the long case degrading. TF32 state and decay errors are direction/head dependent, so they can survive output normalization. However, the report cannot quantitatively derive .992926 or .990660, prove which head dominates, or prove that this contribution exceeds HF cache rounding.

The intervention belongs in the fused recurrence's **state/decay arithmetic and accumulator reload**, not the Python tensor dtype. Preserve full-precision state through decay and the additive state seed; a full-FP32 SFPU path with appropriate direct-to-DST staging is a targeted candidate. Keep matrix operands and direct-to-DST-only buffers/configuration distinct.

**Do not apply a blanket `UnpackToDestFp32` flag to the current CBs.** The accuracy guide at :77 states that such CBs cannot then be used by SrcA/SrcB operations. Here `cb_sd` is both a matrix operand and a copied accumulator seed; the state buffers also feed FPU decay and matrix reads. A correct fix must respect those consumers, with separate staging or valid phase-specific configuration. Changing only `copy_tile` of D would leave the earlier lossy state-times-decay step intact. This investigation does not edit or claim validation of that redesign.

## 2. Confirmed HF cache mismatch and long-reference boundary effect

TEST:78 constructs the HF decoder in BF16. Its QKV projection and convolution cache are BF16. HF:485–487 updates convolution cache before HF:548–550 updates recurrent cache. `DynamicCache(config=cfg)` maps `linear_attention` to `LinearAttentionLayer` (CACHE:1035,1472–1474).

CACHE:925–937 obtains `self.dtype` from convolution state and allocates recurrent storage with that same dtype. CACHE:989 uses `copy_`, so the rule's FP32 result is converted to **BF16**. Both HF rule implementations explicitly convert that cached state back to FP32 for arithmetic (HF:298–302,346–350). The cast back cannot restore discarded bits. A configuration label suggesting FP32 SSM state does not override this implementation: this cache constructor does not read such a policy.

FD:101–102 allocates an FP32 recurrent cache, and FD:164 copies the fused FP32 result into it. Accordingly, the persistent-cache contracts differ even after addressing the device's internal truncation.

Initially inspected TEST:33–53 derives the long reference as follows:

| S | HF calls before/including measured prefill | TT fused token counts per request |
| --- | --- | --- |
| 32 | one 32-token call; internal HF chunk width 64 | one T=32 |
| 129 | one 129-token call; three internal 64-token chunks including padding | T=128, then T=1 |
| 16385 | prefix 16257 = fifteen 1024-token calls + one 897-token call; then measured tail128 | 128 calls with T=128, then T=1 |

At S16385, there are **sixteen BF16 cache boundaries before the measured tail**; each prefix call's internal 64-token chunks otherwise keep FP32 state. The tail begins at zero-based position 16257 and ends at 16384. Its final cache write is the seventeenth prefill cache write and affects subsequent decode, not the already computed tail outputs. For S32/S129, the first prefill cache write happens after the measured prefill outputs; it can affect the first decode but cannot explain their prefill discrepancy.

Thus the helper's “all valid prefix state” is semantically correct, but does not make the streamed BF16-cache reference numerically identical to one monolithic HF prefill. This candidate specifically tracks the reference-path switch at `length > 4096`; pure device accumulation need not have a discontinuity there. Neither boundary is isolated by the supplied matrix.

Do not blindly make TT state BF16 every 128 tokens to imitate HF: that would introduce a **different** rounding schedule from the current HF helper. First establish the intended oracle/cache contract, and compare identical hidden inputs with matched boundaries and with explicitly FP32 HF recurrent storage. Those are discriminating follow-ups, not results of this inspection.

## 3. Other potential issues: precision differences upstream and downstream

- **Q/K normalization:** HF:258–264 and :330–335 apply `l2norm` while Q/K remain BF16, then promote to FP32. The function at :240–243 computes multiply, reduction, epsilon addition, reciprocal square root, and final multiply in that incoming tensor dtype at its materialized boundaries. FD:141–148 promotes first and computes normalization in FP32. Matching only the final output dtype is not the same computation.
- **Beta:** HF:516 evaluates sigmoid on BF16 b; FD:147 promotes b before sigmoid. Both implement the same gate equation, with different rounding.
- **Convolution:** FD:132–136 materializes BF16 products and partial sums across four taps. HF:497 uses a grouped convolution followed by SiLU, with different accumulation/materialization boundaries. No tap-order error was found.
- **Layer norms/gating:** FD precombines `1 + norm_weight` into BF16 at setup (:69–70); HF's zero-centered RMS norm uses its own FP32 arithmetic before casting. HF:187–202 evaluates the post-normalization SiLU gate in FP32; FD:165–167 materializes BF16 normalization/SiLU/product. These feed the final projection and MLP and can amplify/mask recurrent discrepancies. They are shared by passing cases and are not established long-only roots.
- **Chunk algebra versus recurrence:** HF uses a triangular chunk solution and cumulative log-decay inside 64-token chunks (:287–317), whereas TT uses a sequential recurrence. They are algebraically equivalent but have different finite-precision operation order. Chunk size alone is not a missing-state bug.
- **Softplus implementation:** Blackhole's FP32 `ckernel_sfpu_softplus.h:34–42,107` uses a polynomial for the central range. Its constant at input0 is approximately .69310557842 versus ln(2)≈.69314718056. This is a source-coefficient illustration, not a device measurement. It can perturb decay but has not been isolated as a root cause.

These are actual calculation differences, but none has been isolated as responsible for the failing threshold. They belong below the concrete recurrent-state precision findings rather than being called generic “acceptable BF16 drift.”

The external FP32 chain does **not** share the fused kernel's default-unpack problem. Actual binary dispatch preserves operand order: a+dt uses row-broadcast B, neg_A*softplus uses row-broadcast A, and q/k*inverse_norm uses column-broadcast B for multi-token inputs (`binary.cpp:796–876`). Matching FP32 operands select SFPU (`binary_ng_device_operation.cpp:55–58`) and direct-to-DST FP32 modes (`binary_ng_program_factory.cpp:1261–1268`); Blackhole excludes the FP32 LLK broadcast fast path at :363–371 and uses byte/word-preserving software broadcast. FP32 width-sum selects accurate reduction and direct-to-DST input (`reduction/generic/device/reduce_op.cpp:147–152,197–216`; `reduce_op_multi_core_w_program_factory.cpp:477–482`). Unary FP32 preservation is selected in `unary.cpp:56` and `unary_program_factory.cpp:648–669`, including the accurate exp path. Thus the confirmed repeated TF32 issue is localized to the fused recurrence, not inferred for every FP32 TTNN operation.

## 4. Shape, state, and queue audit: candidates not supported by source

### Layout and GQA

For each B=1,T=128 call, WRAP makes Q/K/V `[48,128,128]`, then `[6144,1,128]`; decay/beta become `[6144,1,1]`; state is `[48,128,128]`. For the final T=1 call they are `[48,1,128]` and `[48,1,1]`. Physical tile height remains 32 for the per-token rows; it is **not** an extra 31 recurrent updates. The kernel loop bound is logical T.

GQA repeats head `h` as value heads `3h,3h+1,3h+2`, matching HF's `repeat_interleave(3,dim=2)` at :519–521. Flattening B and H before repetition preserves request boundaries. Reader/writer indexing uses `(h*T+t)` for token tiles and `h*16` for state tiles. The output is restored through row-major reshape and permutation to `[B,T,48,128]` (WRAP:160–164).

FD's reshapes explicitly go through row-major (:30–32), so the 16-head and 48-head tile padding is not intentionally folded into logical features. WRAP's T>1 reshape also goes through row-major (:47–51,106–114); state reshapes preserve the tiled K,V dimensions. `to_layout_op.cpp:169–182` unpads to logical extents before row-major reshape. No evidence supports an accidental BF16 dtype conversion merely from requesting row-major layout. S129 already exercises both T=128 and T=1 paths, substantially weakening a final-token-only layout explanation.

### Convolution and HF padding

HF retains four projected tokens, TT three. This is sufficient and equivalent for a width-four causal convolution: a new output uses exactly three predecessors plus the new token. HF prepends cached context then keeps the last `seq_len` outputs (HF:469–499); TT forms `[old_last3,new_qkv]`, applies offsets 0..3, and retains `history[:,length:length+3]` (FD:131–135). No prefix is dropped at a 128-token boundary.

HF pads the 897-token prefix call to 960 inside the chunk rule; Q/K/V/beta/g padding is zero (HF:268–273). A padded timestep has exp(0)=1 and zero key/update, so it leaves state unchanged in exact arithmetic. The same reasoning covers S32 and S129 padding. Numerical chunk algebra can still differ; “63 spurious decay updates” is not a supported diagnosis.

### Concrete queue ledger

All these cases have Kt=Vt=4, sixteen state tiles per head, `output_per_token_state=false`, and no ring/index state. FACTORY:53–55,114 sets compute compile-time arguments `{4,4,0}`. Runtime T is 32,128,or1, **never 16385**. A B1 launch maps 48 heads to 48 cores using `{h/grid_y,h%grid_y}` (FACTORY:65–74). FD:151–155 splits B32 into batches no larger than `floor(grid.x*grid.y/48)`; no batch state is shared. The logs do not record the exact usable compute grid, so an exact B32 sublaunch count is not asserted.

| Per-head buffer | Production/consumption in each token |
| --- | --- |
| Q,K,V | Reader supplies 4 tiles each; Q consumed after output, K after read/transpose, V after residual. |
| decay,beta | Reader supplies 1 each; each consumed once by its matching phase. Capacity 2 for T>1,1 for T=1. |
| current S | Initial reader supplies 16; each token waits16, decays, and pops16. |
| D,read,u,k-column | Compute produces and consumes respectively16,4,4,4; none persists into the next iteration. |
| next S | Rank-one update produces16. Output matmul reads it without popping; next iteration consumes it as current S. Ping/pong alternates correctly. |
| output | Compute pushes4/token; writer drains4/token. |
| state output | Compute emits16 only on the final token; writer drains16 after all token outputs. |

Sources: COMPUTE:180–225 and the corresponding READER/WRITER loops. For T128, token0 consumes initial S and produces ping; token1 consumes ping and produces pong; token127 consumes ping and produces pong plus final state. T32 has the same final parity. T1 consumes initial S and emits ping plus final state. The last next-state CB remains live only until kernel completion, intentionally; no next token waits on it. CB allocation is 132 FP32 tiles/core for T>1 and130 for T1, independent of total model context. No boundary count, stale-format, or capacity discrepancy was found for these instantiated values.

### Completion and replay

TEST:116 reads prefill to host; :184 reads eager decode; :186 synchronizes before capture; :193 executes replays with `blocking=True` and reads outputs; :199 does the same for the changed input. These metrics therefore represent completed device work, not merely enqueued work. Restoring the prefill snapshot before each first-step replay and then executing the changed-input step **without restore** matches the sequential HF expected_decode/expected_decode2 calls (:95–96,191–200).

The second decode changes both timestep and hidden input. A different q/z/residual/MLP can be much less sensitive to the same state error. Also, x is sampled before xd (:82–84), so short and long cases do not hold decode input fixed despite sharing the seed. The higher second PCC cannot establish state healing, trace-only corruption, or a single bad token. Determinism and matching eager/trace results make a replay-only explanation less likely, without certifying every asynchronous subsystem.

## 5. Candidate adjudication and follow-up controls

| Candidate | Direct evidence | Fit to contrast | Remaining uncertainty / falsifier | Smallest intervention boundary |
| --- | --- | --- | --- | --- |
| Per-token TF32 state/decay truncation | Selected JIT format and exact consuming primitives | 16385 versus 32/129 updates; B32 states remain short | Per-head contribution unmeasured; isolate fused recurrence with identical prepared inputs and full-precision state path | Fused state arithmetic/reload, preserving consumer-format contracts |
| HF BF16 cache plus helper boundaries | Cache allocation/copy and 16 long-prefix writes | Reference control flow changes only for long tests | Same-input monolithic/streamed/FP32-cache HF comparison not run | Reference cache precision and call schedule |
| BF16 normalization, beta, convolution, output gating differences | Different explicit dtype/materialization order | Shared by short/long; can alter a recurrent trajectory | No intermediate attribution; align one boundary at a time | Model preprocessing/postprocessing |
| Wrong shape/GQA/history/padded recurrence | No demonstrated contradiction; invariants traced | S129 already covers128+1; B32 resume1.0 | Device layout bit checks would further test implementation, not an established fix | No structural rewrite justified |
| Trace restore or asynchronous output artifact | Current restore/step/synchronization order is coherent | Same eager/trace failure, deterministic replays | Archived TEST revision not identical | No trace change justified by these metrics |

Prioritized follow-ups for a subsequent execution-authorized repair session:

1. **Localize before judging final PCC.** Save prepared Q/K/V/g/beta and recurrent states at selected prefix lengths, with head/token coordinates and actual dtypes. Compare state relative error, per-head PCC, and core output before norm/projection. Current `--debug-states` helps but the supplied run has no such measurements.
2. **Test the precise recurrence contract.** Use identical prepared inputs for the device and FP32/FP64 reference, including a zero-update identity state, near-unity gates, and long chains split into the same 128-token calls. Check state after 32, 129, and 16385 updates. A full-FP32 state-decay/add path is the relevant controlled contrast; simply changing Python state dtype or the final cast is not.
3. **Isolate the HF oracle difference.** With identical fixed hidden inputs, compare the existing 16-prefix-boundary reference to matched 128-token boundaries and to a reference preserving FP32 recurrent cache. Separate prefill-output changes from the last cache cast's effect on decode. Do not silently replace the acceptance reference and call that a device fix.
4. **Then align other numeric boundaries independently.** BF16 Q/K normalization, beta, convolution accumulation, and gate materialization should each have a measured contribution. Use unchanged inputs for both decode steps and compare the full prefix/state history, not just an aggregate PCC.

The registered-golden tests in `tests/ttnn/unit_tests/operations/transformers/test_fused_recurrent_gdn_golden.py` explicitly run on the host (:5), cover small T values (:32,99–114), and check FP32 versus FP64 at T8 (:155–171). They do not validate this device's long-state precision or the installed HF BF16 cache. Passing those tests would not falsify either leading finding.

## Report review

The initial causal story was challenged against the complete observed matrix and lowered code. Structural padding/history/GQA suspicions were demoted after tracing the invariants. The TF32 finding was traced beyond the FP32 dtype and compute-config declarations into JIT-selected formats and Blackhole consumers. The HF rounding finding was traced through the actual cache class rather than inferred from the rule's FP32 temporary state. The claimed second-decode recovery was limited to its observed output metric because input and timestep both changed.

An independent xhigh subagent reviewed the first draft against the descriptor/JIT/Blackhole consumers, HF cache, and reader/compute/writer code. It confirmed the source-level findings, scalar counterexample, boundary counts, CB ledger, and qualified causal ranking; it found no blocking correction. Its wording correction to distinguish initially inspected TEST from concurrently modified source is included.

The diagnosis remains source-proven precision discrepancies with **unverified attribution of the exact .995 failure**. No target execution or implementation change is included in this report.
