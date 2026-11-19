# AutoDebug: layer 34, seed 154, continuation row 276

## Findings

**The exact short fused control also fails at row 276. This is an inherited continuation-path correctness failure; the selected BFP8/LoFi optimization is not necessary to produce it.** While this inspection was running, additional caller-generated controls appeared in this directory. `diagnostic34_default.json` reproduces the original result exactly. `diagnostic34_fused.json`, using the same 128-token prefix, 128 decode steps and continuation33, fails at PCC **0.9905532110**, contribution PCC **0.9882746633**. The original optimized result is **0.9907021120 / 0.9884516143**. Both are below the existing 0.995 gate. Their similar metrics do not establish identical intermediate errors or expert selections.

**A concrete inherited discrepancy worth testing first is the multi-token router's precision contract.** Both decoders select experts from FP32 projection outputs for continuation33, whereas HF projects to BF16 logits before FP32 softmax/top-k, and rounds the normalized routing weights back to BF16. The exact singleton router does not run for this continuation. This can change cutoff membership and amplify a small numerical difference into a single-token output spike. Actual row-276 logits and expert IDs have not been captured, so this is a leading mechanism to discriminate, not a proven root cause.

**No new recurrent-state dtype or cache-boundary bug was established.** Storage is FP32, with deliberate BF16 rounding at public-call boundaries, in both implementations. The supplied isolated-layer inputs prevent expert/shared output errors from accumulating into future recurrent state. Do not remove the boundary rounding or lower the gate based on this investigation.

This session used source/evidence inspection and cheap host-only checks. It did not import or execute TTNN, open devices, run reproductions, change implementation/tests, or launch AutoDebug again. The only written file is this report. Existing fused-stage reports are preserved. Runtime results below were read from artifacts produced outside this investigation.

## Evidence and revision consistency

Paths in source citations are relative to the model directory (`M`), except `N = ttnn/cpp/ttnn/operations` and `HF = python_env/lib/python3.12/site-packages/transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py`, relative to the repository root. Additional native aliases are `G = N/experimental/nomic_embed_text_v2_moe/grouped_matmul`, `R = N/experimental/nomic_embed_text_v2_moe/routing_metadata`, and `T = N/reduction/topk`. Paths beginning `tt_metal/` are repository-relative. Line numbers refer to inspected source; test harnesses changed during the investigation.

| Object | SHA256 |
| --- | --- |
| `tt/fused_decoder.py`, original failure and new controls | `d8106611f65edccdf069baab44acc9c0088baa709090b017a6a6df75f0a7bb87` |
| `tt/optimized_decoder.py`, original failure and new controls | `99e2a2fa1d4cb78f4dd0eb95079cc8288e8f7e0ebd92e6ffbc44ffb3be05cb00` |
| `tests/reference.py` | `8c60281ea7849c7463f16f721ff8d553ee3014c7324db49b38fb8fa45f1b2dc3` |
| Original failure's `tests/stress_fused_decoder.py` | `8d228111908823faaa40ace58f7d527fd488c21e1cef560fe782d4e0cf0d8e37` |
| New controls' `tests/stress_fused_decoder.py` | `aec934fe1cb4752184ba24c3cbf3d1684f5f79152a0ec120c5cf3e110a51633d` |
| Original failure's optimized stress wrapper | `1e6cd388c59bfcaf56be0dfa9001f697d0cd1cf2ef96c824b1c2af6421240786` |
| New controls' optimized stress wrapper | `f5943b4d0d80b6cab90b6636fc3b545b5d549436d788cb2470041f72b6c8378e` |
| Local HF implementation | `f738a9eb46d1351a751f36a1cae5f7435d589e8c3218e7315182419c4c0230f1` |

The checkpoint revision is `345d467b117d1fd5eda9b879a5988de9ae5f47a7`. The original failure's provenance records Blackhole P300c chip0, a 1x1 mesh, allocation tracking enabled, program-cache allocations included, and no Watcher/profiler. Model source hashes were independently checked against disk. Native source lowering describes this prepared tree; provenance does not independently fingerprint every compiled native kernel.

The new harness adds policy/activation inputs and aggregate metrics. All new diagnostic controls have the same CPU-oracle key, including sequence parameters, Torch version, reference/HF/cache sources and harness hash. The default control exactly reproduces the original prefill and row-276 scores. This bridges the changed harness; the diagnosis does not splice an old failure onto a different model implementation.

### Direct observations

Unless stated otherwise, these are layer34, seed154, prefix128 + decode128 + continuation33, three repeats, one decode capture. Values are the failing continuation row276, relative row20.

| Artifact | Effective change | Raw PCC | Contribution PCC | Gate |
| --- | --- | ---: | ---: | --- |
| `stress_34.json` | Original selected optimized defaults | .9907021120 | .9884516143 | fail |
| `diagnostic34_default.json` | New harness, same defaults | .9907021120 | .9884516143 | fail |
| `diagnostic34_expert_hifi2.json` | Experts BFP8/HiFi2 | .9905490880 | .9882709435 | fail |
| `diagnostic34_expert_bf16.json` | Experts BF16/HiFi4 | .9905532290 | .9882724975 | fail |
| `diagnostic34_shared_bf16.json` | Shared BF16/HiFi4 | .9906875244 | .9884481907 | fail |
| `diagnostic34_prefill_auto.json` | `prefill_block_w=null` | .9907021120 | .9884516143 | fail |
| `diagnostic34_fused.json` | Actual `FusedDecoder`, inherited precisions and geometry | .9905532110 | .9882746633 | fail |

The individual JSONs, provenance and `logs/diagnostic34_*.log` support these results; `stress34_controls.json` is a convenience index, not the sole evidence. The fused log records the assertion and clean device closure. The higher expert precision improves aggregate decode correlation but does not repair the offending row. The automatic-prefill control has identical reported row metrics; no claim of bit-identical continuation tensors is made.

Additional observations:

- Original prefill PCC is .9997121180. All 128 decode steps pass: minimum raw/contribution PCC .9996675880/.9996295043. Decode outputs are bit-equal across three replays.
- Continuation row257 is a smaller shared anomaly: optimized .9976867687/.9973492558 and fused .9976890190/.9973467008. It still passes. Neighbors275 and277 pass, so a broad failure beginning at276 is not observed.
- Optimized layer0 and full-attention layer3 pass their same-shape seed913 short stress cases (`stress_0.json`, `stress_3.json`). This excludes neither layer-specific numeric sensitivity nor data-dependent routing errors.
- `natural_initial_34.json` passes on actual HF-propagated natural257 activations, but uses older optimized source `f3138675…`, prefix256 and one held-out decode token. Its policy lacks the later projection/residual additions. It is evidence for the initial expert policy on that corpus, not coverage of this continuation transition.
- The newer `natural_stress_34.json` uses current model hashes and actual HF-propagated `natural290.pt` inputs, follows the same 128/128/33 modes, and passes: minimum continuation raw/contribution PCC .9999151085/.9995842567. This is a useful data-dependent passing contrast, not a waiver of the random-input failure.
- Aggregate continuation PCC is .9997940348 for the default control and .9998550659 for fused, despite both failing row276. Aggregate PCC cannot replace the per-row gate.

### What the older fused pass actually covered

`../fused_decoder/stress_representative_34_seed154.json` (archived gzip parts plus provenance) uses prefix128 + **decode8192** + continuation33 and the same fused source. Absolute row276 was a **single-token decode** there; the continuation begins at8320. Its row276 PCC is .9999954792.

A cheap host-only BF16 RNG shape-partition check, starting from seed154, produced equal short-continuation33 and long-decode128:161 tensors, with equal prefix/first128 decode inputs. This supports input equivalence given the same RNG state at input generation; it is not a byte comparison against archived passing-run tensors. Thus the older pass is not dismissed as necessarily different data at276. It is nevertheless a different execution/reference mode: repeated public recurrent calls versus one chunked continuation, with different internal cache-rounding and projection/router paths. Its distant continuation8320 is different data/state. The newly completed exact fused128 control resolves the formerly missing comparison without depending on the older corpus mapping.

## State, execution and ownership ledger

`FusedDecoder.new_recurrent_state` (`tt/fused_decoder.py:1116-1128`) allocates:

| Object | Logical shape | Storage |
| --- | --- | --- |
| Convolution history | `[1,3,8192]` | BF16, row-major, DRAM |
| Recurrent state | `[1,32,128,128]` | FP32, tiled, DRAM |

History positions below identify pre-convolution QKV rows, not external hidden-state rows.

The policy string `"kv":"bf16"` is not a claim that linear recurrent storage is BF16. The base `self.compute` remains HiFi4, approximation disabled, FP32 DEST enabled and packer L1 accumulation disabled (`fused_decoder.py:327-333`). Optimized expert/shared configs are separate objects (`optimized_decoder.py:44-57`).

| Phase | Effective behavior |
| --- | --- |
| Prefix128 | Plans `(0,64,0,None)` and `(64,64,64,None)`. Public entry rounds recurrent BF16 then widens FP32 once; the two internal chunks retain FP32 carry. |
| Save/warm/capture | Prefix state cloned. Restore before warmup, restore and synchronize before capture, restore before every measured repeat. |
| Decode128, positions128-255 | Every call consumes BF16-rounded incoming recurrent state, computes/stores FP32 updated state and computes current output from the unrounded update. |
| End of final repeat | Conv contains positions253-255; recurrent contains FP32 state after255. The last replay blocks and its output is read back before trace release. |
| Continuation33, start256 | One plan `(0,33,256,None)`, `decode=False`, `cache_boundary=False`. Public entry rounds state after255 once. Chunk graph pads33 to64 internally with31 neutral rows; outputs crop to33. |
| Exit288 | Conv contains positions286-288. Mathematically, FP32 recurrent carry corresponds to the logical33 updates: zero K/V/beta and zero g padding adds no logical update. This is not proof of bit equality to an unpadded arithmetic schedule. |

Source: `fused_decoder.py:1138-1177,1179-1204,1255-1277,1828-1833,1883-1899,2053-2067,2325-2372`; `tests/stress_fused_decoder.py:122-145,157-195`.

HF initializes its cache dtype from the BF16 convolution state, allocates recurrent storage using that dtype, and copies updated states into it (`cache_utils.py:921-940,976-989`). Its chunk rule widens inputs/initial state to FP32, retains FP32 arithmetic within chunks, and converts core outputs back to BF16 (`HF:256-321`). Its recurrent versus chunk choice is controlled by cached state and `seq_len==1` (`HF:521-544`). TT's round-on-consumption policy implements this public BF16 cache boundary while retaining unrounded FP32 state for the current output. No contradictory state narrowing unique to optimized was found.

Critically, `_forward` updates attention state **before** post-attention norm and `_moe`; `_moe` has no state argument (`fused_decoder.py:1795-1805`). Stress refreshes pre-generated external inputs, never the preceding decoder output (`tests/stress_fused_decoder.py:38-50,160`). Ordinary expert/shared quantization error therefore cannot feed back into future recurrent state or upstream router logits in this harness. Different states under an exact expert-only A/B would require an additional alias/lifetime/dispatch corruption; it would not be ordinary model feedback.

The failure follows completed work: replay uses `blocking=True`, followed by host output copies; continuation also performs host readback before scoring. No concrete unfinished-work, restore, extra-capture-update or trace-release lifetime defect was found. Three repeats establish decode determinism only: continuation executes once after the final repeat, outside the trace. The separate default rerun reproduces its failure.

## Projection and expert changes that actually execute

Layer34 is linear attention. `_ProjectionDecoder` installs/replaces attention Q/K/V/gate/output candidates only for `full_attention` (`optimized_decoder.py:444-459,481-490,607-620`). `OptimizedDecoder`'s residual override is also full-attention-only (`:696-707`). Linear-attention Q/K/V/A/B/Z and output projection, input/post-attention norms, state graph and singleton ordered router delegate to fused code. Changing `attention_dtype` or `residual_mode` is not a meaningful layer34 discriminator.

### Shared projections

The changed shared branch packs gate512 + up512 + scalar padded32, yielding weights `[2048,1056]`; down weights are `[512,2048]`. Default packed-all quantizes the scalar alongside gate/up to **BFP8/HiFi2**. The separately exposed scalar BF16/HiFi4 defaults apply only to split topologies (`optimized_decoder.py:461-505`). There is no missing scalar column or gate/up reorder in the inspected packing.

At continuation33 these are BF16 tiled activations, BFP8 weights, BF16 outputs, DRAM interleaved tensors, explicit 2D multicast matmuls (`optimized_decoder.py:565-605`):

| Role | Logical M,K,N | Tiles M,K,N | Configured grid | K block | Per-core/output M,N | Subblock | K blocks |
| --- | --- | --- | --- | ---: | --- | --- | ---: |
| Shared packed | 33,2048,1056 | 2,64,33 | 8x8 | 8 | 1,5 | 1,1 | 8 |
| Shared down | 33,512,2048 | 2,16,64 | 8x8 | 8 | 1,8 | 1,1 | 2 |

The partial last N block of packed projection is a normal padded output block. `MatmulMultiCoreReuseMultiCastProgramConfig` selects the 2D factory (`N/matmul/device/matmul_device_operation.cpp:2748`; `N/matmul/device/factory/matmul_multicore_reuse_mcast_2d_program_factory.cpp:361,451-454,1838-1842`). The factory configures FP32 intermediate reload with UnpackToDest (`:1467-1502`); blanket TF32-spill accusations are not supported.

The three-reader DRAM-sharded programs apply only to logical singleton inputs. Their changed memory layout/reader counts execute during decode, but **row276 uses the interleaved 2D prefill path**, not that decode factory. Fused uses BF16/HiFi4 shared projections with automatic matmul configuration. Those differences are real; however, exact fused and `prefill_block_w=null` controls still fail, so neither new geometry nor new shared precision is necessary for this failure.

### Routed experts

The optimized `_moe` AST preserves fused routing, activation and combine algebra. Differences are sparse geometry/output placement and expert compute configuration, plus constructor weight conversions. At continuation33, the **grouped** branch executes (`optimized_decoder.py:299-350`), not singleton sparse matmul. Sparse blocks16/16,32/64 cores and `expert_memory` do not configure the offending grouped matmuls.

Each of four64-expert bank passes consumes IDs/weights `[1,1,33,8]`. Metadata capacity is2304 gathered rows: `(ceil(33*8/32)+64-1)*32`, not an observed active-row count (`N/experimental/nomic_embed_text_v2_moe/routing_metadata/device/routing_metadata_device_operation.cpp:105-110`). Dispatched activations are `[1,1,2304,2048]`; grouped gate/up produces `[1,1,2304,1024]`; activated width512 feeds grouped down output width2048. Actual expert counts/order depend on routing and were not saved.

Both grouped outputs explicitly use BF16; dispatched activations and grouped outputs select L1 for `s<=256`, in both decoders, while routed-combine output is DRAM. The default difference is BFP8 expert weights with LoFi versus BF16/HiFi4. Mixed BF16/BFP8 inputs are accepted independently by grouped validation (`G/device/grouped_matmul_device_operation.cpp:30-35,68-69,111-114`). The supplied compute config is retained and forwarded (`G/grouped_matmul.cpp:22-28`; `G/device/grouped_matmul_program_factory.cpp:83-84,257-261`). LoFi remains a lower product fidelity despite FP32 accumulation; FP32 DEST does not restore omitted fidelity phases (`tt_metal/tt-llk/tt_llk_blackhole/llk_lib/llk_math_matmul.h:433-434`). This is a supported approximation, not an identified dtype-contract violation. BF16/HiFi4 expert and actual fused controls retain the row276 failure.

## Leading unresolved source discrepancy: continuation router

For `s=33`, both `_moe` implementations request FP32 logits, then run `ttnn.topk(logits,k=8,sorted=True)` and softmax the selected values using the inherited HiFi4 compute config (`fused_decoder.py:1502-1519`; `optimized_decoder.py:238-254`). `ordered_router` and terminal-scan paths require singleton input (`fused_decoder.py:688-750`); they are not selected by this continuation.

HF instead computes:

```text
BF16 hidden and BF16 gate weights -> F.linear -> BF16 logits
FP32 softmax over 256 -> top-k8 -> divide by selected sum -> BF16 scores
```

See `HF:784-792`. TT's corresponding multi-token path has no BF16 logit boundary before selection, and retains FP32 scores rather than HF's BF16 routing-weight boundary.

The prepared native path preserves this discrepancy rather than silently rounding it away. Top-k preprocessing normalizes rank without changing value dtype (`T/topk.cpp:596-607`). For logical33, padded64 rows, width256 and requested k8 (internally32), the large-width routes are ineligible; `TopKSingleCoreProgramFactory` splits the two tile rows over two cores. It keeps Float32 value buffers, explicitly unpacks input/intermediate values to DEST, and enables32-bit DEST (`T/device/topk_single_core_program_factory.cpp:40-42,64-89,300-311,403-410`). Blackhole's SFPU sort loads/comparisons preserve FP32 values; there is no implied BF16/TF32 conversion. Top-k values retain input dtype (`T/device/topk_device_operation.cpp:436-438`), softmax output retains its input dtype (`N/normalization/softmax/device/softmax_device_operation.cpp:379-384`), and routing metadata requires FP32 scores and emits FP32 combine weights (`R/device/routing_metadata_device_operation.cpp:58-59,135-138`). Its weight kernel copies32-bit words (`R/device/kernels/dataflow/routing_metadata_weights.cpp:74-91`). Thus FP32 score precision also survives into the grouped combine setup.

Selecting logits first and applying softmax only to the selected eight is mathematically valid when membership is the same. It is not, by itself, the discrepancy. Likewise, permutations of correctly paired IDs and scores preserve the routed sum in exact arithmetic; reordered finite-precision reductions can still change low bits. The stronger harmful possibility is **different membership** after BF16 rounding or differing finite-precision cutoff/tie handling, or different rounded weights. Expert identities are not interchangeable at a tie.

This mechanism fits a common fused/optimized, data-dependent continuation spike, survives every completed policy rollback, and does not require corrupted persistent state. It remains unproved at276 because final-output PCC does not reveal logits, route sets, rank8/9 gaps, shared/routed split, or the earliest upstream divergence. A small inherited attention/core/postnorm difference can also change routing before this explicit dtype discrepancy is reached. Those candidates need the same failing-row evidence; neither is accepted solely because it sounds numerically plausible.

| Candidate | Source evidence and observed coverage | Missing precondition / smallest discriminator | Rank |
| --- | --- | --- | --- |
| Inherited continuation router precision/selection | Explicit BF16-boundary discrepancy; shared by both short failures; singleton exact selector bypassed | Actual row276 membership/weight discrepancy; identical-input reference router replay | First discriminating control, causal confidence unproved |
| Inherited attention, chunk output or postnorm numerics | Chunk path differs from passing public singletons; state arithmetic uses different schedules from HF | Earliest bad state/core/output boundary; same-input chunk and state cross-substitution | Competing common cause, including possible upstream route trigger |
| New expert/shared precision or projection geometry | Real precision/tiling differences, but high-precision, auto-prefill and actual fused controls fail | Could affect differential metrics, not needed for failure; separate routed/shared outputs only after common localization | Demoted as sole cause |
| Trace-release alias/lifetime corruption | No source violation found; last decode blocks and state has persistent ownership | Copied-state continuation must change behavior across release boundary | Secondary until a concrete state mismatch exists |

## Minimal discriminating follow-up

The broad expert/shared/geometry rollbacks and exact fused control are already completed; do not repeat them as if they were missing. The smallest useful next work is one preserved failing continuation, not another large precision sweep.

1. **Retain the actual failure and freeze its incoming state.** Save continuation inputs, HF/device outputs, and conv/recurrent tensors after255, before the public continuation cast. Clone tensors before any in-place update. Re-run the same33-token call from copied state, preserving the one-public-call boundary. Compare Fused and Optimized attention output, residual, postnorm and router inputs at relative rows19/20/21 (positions275/276/277); include row1/position257 as a second anomaly. Model outputs do not mathematically feed state, so these upstream boundaries should agree between implementations absent additional corruption.
2. **Test routing on identical inputs before altering state math.** On the captured BF16 TT postnorm tensor, compute the reference BF16 linear/FP32 softmax/top-k/BF16 weights. Separately apply reference routing to BF16-rounded TT logits. Capture production logits, top16 values/IDs, rank8/9 gap, paired normalized scores and actual routed IDs. This separates upstream error, GEMM accumulation, missing BF16 boundary and tie-selection policy. Compare sets and per-ID weights, not only order. Replay MoE with reference paired routes on that same input; widen the BF16-rounded weights back to FP32 before `routing_metadata`, as its API requires. Merely casting logits then calling TT top-k is not a decisive negative control if its tie policy still differs.
3. **If routes agree, localize the common numeric error.** Capture pre-gated-norm chunk output, gated output, attention projection, routed expert sum and shared output. Replay common chunk64 arithmetic with identical Q/K/V/g/beta and incoming state, then cross-substitute HF versus TT incoming state. Compare BF16 cache codes as well as per-head error. This distinguishes inherited frontend/state error from the chunk-output/projection path and from expert/combine error. Preserve cache rounding once per public call; normalizing Q/K or applying query scale twice invalidates the control.
4. **Use a mode contrast only with a mode-matched oracle.** From copied state after255, compare one33-token call against33 public singleton calls, each against its own HF execution of that same partition. The two modes legitimately have different cache-rounding boundaries. A singleton-only pass would locate the exposed mode but would not prove which chunk primitive is faulty. If state copied before versus after trace release changes the result, then investigate lifetime/aliasing with that concrete contrast.

Each intervention should retain exact row metrics and source fingerprints. A fix must restore the original per-row0.995 gate for this random corpus and retain passing natural/stress cases. Neither aggregate PCC nor the natural-text pass establishes correctness for row276.

## Other potential issues and evidence gaps

- **Failure artifact omits the failure tensor.** `tests/stress_fused_decoder.py:242-245` saves prefix, decode tokens, decode references and decode outputs, but no continuation inputs/reference/output. Here the only failing row is in continuation. Existing sidecars cannot independently recompute its PCC or identify its offending expert/head. Add those tensors to future diagnostic evidence; this omission does not cause the production error.
- **Contribution PCC is not an attention-state metric.** It subtracts the original input from the complete decoder output, including attention and MoE. Subtraction occurs in BF16 before `tests/reference.py:64-69` widens for correlation. Both raw and contribution PCC fail, so the finding cannot be dismissed as contribution-subtraction rounding alone.
- **Padding is not a demonstrated root cause.** Row276 is relative row20, not logical tail32 or tile boundary31/32. Both HF and TT neutral-pad the chunk to64. Causal computation and neutral updates make that structure coherent. A harmful lowered implementation defect remains possible, but no concrete padding/count contradiction was found.
- **Approximation is not an acceptance verdict.** Expert LoFi, BFP8 packing including shared scalar, score precision and changed shared tiling can perturb outputs. Their existence does not explain the shared fused failure by itself. No architecture-specific scoreboard/mode-state violation was established, and no assertion that .99 is harmless numeric drift is justified.

## Draft review

The draft was checked against the executed model hashes, individual control JSONs/provenance/logs, local HF/cache source, the model call graph and native operation contracts. Three independent xhigh subagents reviewed state/baseline, expert/router lowering, and shared projections; their review corrected native filenames, dispatch citations, RNG-evidence scope, and exact-arithmetic qualifications. The newly completed matched fused failure changed the headline from an optimization suspicion to an inherited continuation failure. State narrowing, new full-attention/residual paths, sparse decode geometry and the three-reader shared decode factory were rejected as standalone explanations for row276. The router dtype discrepancy is kept separate from an unobserved actual route swap. No hardware root-cause confirmation is claimed.
