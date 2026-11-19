# AutoDebug: BFP8 resumed-prefill PCC failure

Inspection date: 2026-11-16. Checkout HEAD: `a8fdbbc11b9bf58e597e5d4003dec559e3a50a99`.
Paths abbreviated `M/` below mean `models/autoports/qwen_qwen3_6_35b_a3b/`.
`R/` means `/home/ttuser/qwen36-35b-a3b/codex_home/plugins/cache/tenstorrent-skills/tt-model-bringup/0.1.24/runtime/readiness_check/`.
`HF/` means `python_env/lib/python3.12/site-packages/transformers/`.

## Findings

**No source-level token/page offset, lost-history, or residual-padding defect was established. There are multiple real changes in numerical execution across the split. The new matched HF evidence reproduces the start-68 and start-2528 failures, but does not explain away TT's start-2559 failure: HF passes that exact singleton case.** The current TT gate remains **FAIL**. There is no verified minimal production fix from this inspection.

A concrete TT-specific numerical candidate is **grouped-MoE arithmetic depending on the other rows in its launch**, through counts-dependent stochastic rounding and internal BFP8 activation boundaries. This can perturb prefix activations before a resumed attention state is constructed. It is a demonstrated source mechanism, not a demonstrated cause of the observed PCC minimum or a measured ranking above projection/collective differences. The public singleton additionally changes projection reduction grouping, recurrent kernel, gated activation, collective, and expert implementation; start2559 is now the cleanest case for localizing an additional TT contribution.

Do **not** remove the recurrent BF16 conversion on the theory that FP32 storage makes it accidental: the original HF BF16 cache also rounds the nonzero recurrent state at this boundary. Do **not** call the entire failure unavoidable HF variation: the matching failing positions differ, and the singleton control passes on HF. The distinction matters for selecting the next experiment.

This session and its three inspection subagents read source and artifacts, extracted the planner with Python AST, and analyzed saved logits on CPU. We launched no model, accessed no device, reset/terminated no process, and edited no source. Only this report was written. The TT run and separately prepared CPU HF controls completed while inspection was in progress; their completed artifacts are included below.

## Direct observations and their limits

The installed runner defaults are length2560, block32, tile32, window256. Its actual starts are32,68,2528,2559 (`R/run_prefill_resume_check.py:80–116,435–439`). It compares one fresh full call with prefix followed by suffix, scoring absolute `[start,min(start+256,2560))` plus2559. It is a comparison of each implementation with itself, not TT against HF. The diagnostic wrapper saves the first compared fresh/resumed pair without modifying scoring (`M/tests/datatype_resume_check.py:22–46`).

The completed TT artifact is `M/doc/datatype_sweep/baseline_resume_check.json`. Its recorded command in `M/doc/datatype_sweep/work_log.md` specifies all40 layers, context4096, and explicit `precision_config="baseline"`; the failed-case runtime precision summaries confirm baseline and mesh1x4:

| Start | Scored rows | Mean PCC | Minimum PCC | Top1 agreement | Result |
|---:|---:|---:|---:|---:|---|
| 32 | 257 | .995132673 | .980385446 | .992217898 | PASS |
| 68 | 257 | .994176119 | .968224176 | .988326848 | FAIL minimum |
| 2528 | 32 | .989604529 | .973850864 | 1 | FAIL mean/minimum |
| 2559 | 1 | .986428019 | .986428019 | 1 | FAIL mean |

All last-position top1 checks pass. Both negative controls fail correctly for starts32/68. Controls are not run for the shorter windows at2528/2559. Negative-control failure establishes sensitivity to absent context; it does not prove the retained state is numerically correct.

The saved tensors are finite CPU float32, with BF16-representable values: start68 `[1,257,248320]`, start2528 `[1,32,248320]`, start2559 `[1,1,248320]`. The fresh last-position full-vocabulary vector is bit-identical across these three saved runs. Their resumed last-position vectors differ. This supplies an actual correspondence check against an uncontrolled fresh-reference explanation, without proving intermediate device lifetime correctness.

For start68, minimum failures occur at absolute positions105,118,257,292,300,304; the worst is304. Top1 differs at132,172,304. These are spread through the scored window, including positions after full attention returns to its aligned path at256. This is not an observed single corrupt row at the page handoff. Values at304 are fresh token257=25 versus token13=23.875, resumed token13=24.5 versus token257=24.125. A narrow top1 margin can explain an argmax flip; it does not exempt the vocabulary-wide PCC failure. The start68 final2559 PCC is .989069352.

The separately completed controls are `M/doc/datatype_sweep/hf_resume_control/L2560_s{68,2528,2559}/result.json` and `summary.json`:

| Start | HF mean/min PCC | HF top1 | HF result | Interpretation |
|---:|---|---:|---|---|
| 68 | .988563302 / .497114083 | .988326848 | FAIL mean/min | Exact-case HF BF16 split sensitivity exists |
| 2528 | .991039354 / .977858609 | 1 | FAIL minimum | Exact-case HF sensitivity also exists at an aligned split |
| 2559 | .992442349 / .992442349 | 1 | PASS | A blanket HF explanation does not cover TT's singleton failure |

These are completed external measurements, not runs performed by AutoDebug. I checked exact token and position equality against the TT diagnostic JSONs, the three forward cache lengths, recorded unchanged Torch RNG, and the hashes of the runner, HF harness, config, HF source, and diagnostic JSONs. The source is `Qwen3_5MoeForCausalLM`, revision `345d467b117d1fd5eda9b879a5988de9ae5f47a7`, CPU, Transformers5.12.1, Torch2.11.0+cpu, SDPA. The new harness explicitly sets `dtype=torch.bfloat16` (`M/tests/datatype_hf_resume_control.py:92`); this matches the checkpoint's text-config dtype and the historical default-load control's recorded BF16 dtype. Installed loading also resolves omitted dtype through `None→auto→config.dtype` (`HF/modeling_utils.py:836–839,4120–4121`), using the text config for this model. It is accurate to call this a matched BF16 control, not a constructor with no explicit dtype argument.

Position correspondence now exists and contradicts an identical-error claim:

| Position in s68 window | TT fresh/resumed PCC | HF fresh/resumed PCC |
|---:|---:|---:|
| 105 | .974138 | .994668 |
| 118 | .979579 | .581901 |
| 135 | .993936 | .497114 |
| 257 | .976881 | .966376 |
| 292 | .978708 | .981754 |
| 300 | .974548 | .981725 |
| 304 | .968224 | .974685 |

Only118,257,304 overlap between the six TT and sixteen HF rows below.98. HF's top1 changes at113,130,304; TT changes at132,172,304. At304 both switch257→13; HF resumed has an exact top-logit tie. At132 HF fresh also has a tie, but both HF runs select16. At172 both HF runs select11, agreeing with TT resumed rather than TT fresh. At2528 the TT worst position is2528; HF's is2531. These controls demonstrate sensitivity, not common intermediate causes or a waiver. Prior stage6 cases at lengths257/1025 are historical context only.

## Actual continuation and padding geometry

`M/tt/generator.py:688–700` resets once before each fresh/resumed experiment, calls prefix with absolute end `start`, then suffix with the same token tensor/cache and nonzero start. Only `start==0` zeroes conv/recurrent state (`:403–406`). Validation uses absolute ends, and the executor receives exactly `tokens[row,start:end]` with `start=start` (`:421–428`). Residual row0 intentionally means the first suffix token; it is not absolute token0.

The following is derived by evaluating the actual `attention_chunks` function without importing TTNN (`M/tt/prefill.py:26–39`). MoE physical rows follow `:213–219`:

| Call interval | Linear attention | Full attention | MoE valid → physical rows |
|---|---|---|---|
| fresh `[0,2560)` | 40×64 | 10×256 | 2048 +512 |
| s32 prefix `[0,32)` | 32 | 32 | 32→32 |
| s32 suffix `[32,2560)` | 39×64 +32 | 224 singleton updates32–255, then9×256 | 2048 +480→480 |
| s68 prefix `[0,68)` | 64 +4 | 64 then singletons64–67 | 68→96 |
| s68 suffix `[68,2560)` | 38×64 +60 | 188 singletons68–255, then9×256 | 2048 +444→448 |
| s2528 prefix | 39×64 +32 | 9×256 +224 | 2048 +480 |
| s2528 suffix | 32 | 32 singletons2528–2559 | 32 |
| s2559 prefix | 39×64 +63 | 9×256 +224 then31 singletons2528–2558 | 2048 +511→512 |
| s2559 suffix | public singleton | singleton2559 | singleton sparse MoE |

Linear/MoE grouping restarts relative to each public call. Full-attention alignment is absolute. **Passing start32 already uses more singleton full-attention updates than failing start68.** A singleton-full-attention bug cannot be inferred merely from their presence. The page-aligned2528 failure also rules out a diagnosis requiring only `start%32 != 0`.

For s68, full-attention prefix fills pages0–1 and updates page2 rows0–3. Suffix starts page2 row4, updates through page7, then at absolute256 fills pages8–15. Subsequent fills cover16–23 through72–79. Every multirow suffix fill starts at an absolute multiple256; none fills a partial first page. `_positions` selects `absolute//32` for its eight-page window, while singleton cache updates receive the absolute token position. SDPA uses the full table and absolute chunk start (`M/tt/prefill.py:111–125`; `M/tt/multichip_decoder.py:1948–1994`). `page_window` copies `table[first+i]` with small-read address alignment; its clamp is irrelevant to pages0–79 in the128-entry allocation (`M/tt/prefill_transport.py:63–75`). No page-offset contradiction was found.

Residual copy moves4096 bytes per BF16 row, with distinct source/destination, a read barrier before reusing scratch, and a write barrier before the next iteration. Its metadata is `(source_row,destination_row,count,extent)`. Every invalid row below `extent` is explicitly zeroed (`M/tt/prefill_transport.py:18–46,114–160`). The executor writes every valid row after attention and MoE before the next layer reads it; source/destination alternate and the attended workspace is separate (`M/tt/prefill.py:197–220`). Logical linear tails4/60 are internally zero-padded to64 for DeltaNet and cropped; their zeros do not advance the public token interval.

The final s68 logit block starts relative2464/absolute2532 and contains28 valid rows; token2559 occupies physical row27. Fresh uses absolute2528 and row31. Selected-index conversion and host cropping are consistent (`M/tt/prefill.py:221–233`; `M/tt/generator.py:408–435`). Row-lane numerical behavior is testable, but there is no demonstrated off-by-four output slice.

## Earliest plausible divergence, module by module

### Linear attention: BF16 state boundary matches HF

At every public prefill call, `BoundedPrefill._begin_layer` performs FP32→BF16→FP32 on recurrent storage (`M/tt/prefill.py:127–131`), matching the accepted decoder (`M/tt/multichip_decoder.py:1763–1764`). Fresh converts zero state; resumed additionally rounds a nonzero state. FP32 backing therefore does not mean FP32 persistence across public calls.

The original HF recurrence computes and carries internal chunks in FP32 (`HF/models/qwen3_5_moe/modeling_qwen3_5_moe.py:260–315`). Its cache initializes dtype from BF16 convolution state, creates recurrent storage with that same dtype, then copies the FP32 final state into it (`HF/cache_utils.py:921–941,974–991`). The next call widens that BF16 state to FP32. TT delays this rounding until consumption; numerically that is the same public-call boundary for this path. Removing it would alter reference semantics, not fix an established porting error.

The first layer's s68 prefix begins with the same `[0:64)` attention chunk as fresh. Input projections for multirow linear attention loop over individual rows with K1024 (`M/tt/multichip_decoder.py:678–688`). Convolution uses the last three raw QKV rows plus the current row and four taps; three cached rows suffice for kernel width4 despite HF storing four. The slice updating history is `[s:s+3)` of `old3+new_s` (`:462–471`). No off-by-one was found.

Prefix rows64–67 run a four-valid-row chunk padded to64; fresh uses `[64:128)`. The causal triangular/output dependency graph does not consume future rows for those first four outputs. Padding k/v/g/beta with zero makes padded state updates mathematically neutral (`M/tt/fused_decoder.py:2053–2105,2325–2372`). The **first explicit split-dependent state/chunk boundary is token68**: resumed starts a new chunk from rounded S68, while fresh computes row4 of the chunk beginning64 with internal FP32 carry. This is a genuine finite-precision difference, not an indexing defect. It also occurs under HF's chunk64/BF16-cache implementation. The **earliest plausible layer0 attention numerical difference is already token64**, because the four-row and64-row norm/epilogue/output-projection geometries differ; native output `ttnn.linear` bypasses the rowwise input scan (`M/tt/multichip_decoder.py:578,671–677`). Source causality does not prove those outputs bit-identical. Only rows0–63 share identical layer0 attention call geometry. Neither the first divergent tensor nor attribution of final PCC .968224 is measured.

These comparisons concern the layer0 attention chain, **not necessarily the earliest divergence in the entire stack**: layer0's differently grouped MoE can already change prefix rows before layer1 constructs its state, including rows0–63 whose attention call geometry is identical.

### Grouped MoE: a TT-only source of launch dependence

`BoundedPrefill._moe` leaves `batched=False`, so multirow `_moe` calls `_grouped_moe(...,stochastic_rounding=True)` (`M/tt/prefill.py:147–151`; `M/tt/multichip_decoder.py:1612,1615–1675`). The native argument survives the public wrapper, device attributes, cache key, and factory define. The compute kernel enables FPU stochastic rounding and seeds `0x5EED1234` once per invocation, then loops over experts and counts-dependent blocks (`ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/moe_fused_swiglu/device/kernels/moe_fused_swiglu_compute.cpp:446–459,519–564`; factory `:591–611`). A fixed seed reproduces a fixed schedule; it does not give a row the same random rounding sequence after the schedule changes.

Fresh first MoE launch has2048 rows; s68 prefix has96 physical rows, including28 zero rows. Router top8 is still computed for zero rows; scatter/count/alignment and dispatch have no valid-row mask (`M/tt/multichip_decoder.py:1553–1566,1615–1657`). Zero rows have zero mathematical expert contribution in this bias-free model, but they change expert counts, tiling, and potentially the stochastic stream consumed by valid rows. Removing suffix rows or restarting at token68 also changes row placement and counts even without padding. Start32 passes despite those schedule changes, so the mechanism is not a sufficient prediction of gate failure by itself.

Precision is lower inside this path than `activation_dtype="bf16"` alone suggests: BF16 row-major routed activations are tilized into **BFP8** `CB_X_TILES`; gate/up accumulators, gathered partials, and hidden buffers also use BFP8. Slice/intermediate/output buffers include BF16. These formats are explicit in native `device/moe_fused_swiglu_geometry.cpp:319–348`; conversion and activation packing occur in compute `:606–624,762–777`. The public BF16 boundary is real, but it is not an all-BF16 activation pipeline. A BF16-weight-only sweep does not remove these internal BFP8 boundaries.

Thus layer0 post-MoE prefix outputs are an early useful observation point. No actual layerwise tensor capture currently proves this path owns the failing rows. Disabling stochastic rounding would be a diagnostic/policy change, not a validated policy-preserving production fix. It cannot resolve the shared HF mechanism by definition.

### Public singleton: additional TT sensitivity remains unresolved

At start2559 the suffix has one token. `prefill.py:133–143` selects `decode=True`, `cache_boundary=False`, `projection_k=512`. An uninterrupted64-row linear chunk uses K1024 and the chunk64 output graph. The singleton instead uses compensated singleton norm, K512 scan grouping, `_recurrent_step`, and the singleton gated epilogue (`M/tt/multichip_decoder.py:387–440,442–635,659–711,1473–1481`). Its initial state is rounded once by `_begin_layer`; `_recurrent_step(round_state=False)` avoids another round.

The optimized recurrent path must be traced through `mesh_state_slab_recurrence.py`, not assumed to be native FDR. `full_model_optimized=True` is passed at `M/tt/model.py:98`; the decoder installs its state-slab helper (`multichip_decoder.py:1239–1252`). Its shape/dtype guards select separate FP32 operations with HiFi4, FP32 destination, and explicit FP32 unpack (`M/tt/mesh_state_slab_recurrence.py:263–293,335–341`). No accidental BFP8/LoFi/TF32 recurrence downgrade was found. FP32 storage alone does not prove equality with the chunk graph's corrected matmuls/reduction order. The singleton's `cache_boundary=False` gated epilogue uses SFPU SILU, whereas the multirow prefill epilogue uses the BF16-code lookup table (`:549–582,599–633`). Both intend the same function; a wrong result is not established merely from this implementation difference. Historical `M/doc/fused_decoder/endurance/bf16_unary_precision.json` finds ordinary BF16-rounded SiLU outputs matching, apart from tiny/subnormal differences; any new test must compare the gated product before its final rounding, not assume a large standalone SiLU error.

Every layer's MoE also switches from grouped EP (LoFi, BF16 destination, internal BFP8, stochastic rounding) to indexed sparse matmul (LoFi, FP32 destination) (`:1567–1613`). Full attention switches projection/normalization/SDPA paths, and singleton reduction can use direct all-reduce instead of multirow reduce-scatter/gather. Prefix2559 already substitutes one zero pad row for the real final token inside its last512-row MoE group, so its retained prefix state can differ before the singleton runs. A singleton-only replacement cannot exonerate that prefix state.

HF also selects a cached recurrent rather than chunked DeltaNet path for a one-token suffix (`HF/models/qwen3_5_moe/modeling_qwen3_5_moe.py:467–475,521–548`), but its matched case passes. That makes TT's additional singleton/prefix precision differences a priority, not proof that any one kernel is defective.

## Effective dtype/fidelity ledger

Runtime precision in all failed diagnostic JSONs agrees with `M/tt/precision.py:20–57`:

| Component | Selected execution |
|---|---|
| Embedding, residual, public activations/logits, conv state | BF16 |
| Recurrent storage/internal chunk carry | FP32; BF16 consumption at each public call |
| Linear-attention weights/projections | BF16; ordered coefficient scans/FP32 arithmetic and documented BF16 boundaries |
| Full-attention/shared/head weights | BFP8; HiFi2, FP32 destination, approximate math off |
| Indexed expert GU/down | BFP8 weights; LoFi, FP32 destination |
| Grouped EP | BFP8 weights/internal buffers; LoFi, BF16 destination, stochastic FPU rounding |
| Router | FP32 weights/logits using native BF16/ordered scan machinery; top8 scores BF16 |
| KV | BF16 TILE, page32; no BFP8 KV compression in this baseline |
| CCL | BF16 reduction partials; FP32 input partials preserved until intended BF16 residual output |
| Norm | FP32/native BF16 helpers, layer HiFi4; terminal FP32 RMSNorm HiFi2 then BF16 |

Terminal uses the same32-physical-row DRAM head variant for selected prefill blocks (`M/tt/model.py:237–258`). Dense projection and collective factory selection still depend on row shape and memory layout. No blanket assertion that FP32 destination makes all these branches exact is justified.

## Discriminating minimal follow-ups and conditional fixes

These are proposed follow-ups for the hardware-owning caller, not experiments performed here. They should preserve raw failed evidence and use the unchanged gate for final validation.

1. **Localize once with actual full-depth data before changing policy.** For s68 and especially s2559, retain the requested row immediately before/after attention and after MoE for each layer, plus the exact pre-update conv/recurrent state. Include layer0 prefix rows0–63 and post-MoE output in the s68 comparison: their attention chunk is identical while the MoE launch differs. Record logical/physical rows, branch flags, and state before/after the public BF16 cast. Compare prefix state separately from suffix execution. This finds the first additional TT deviation; aggregate logits cannot identify it.

2. **Same-operand singleton control at the first divergent linear layer.** Feed the *same captured* normalized token, conv history, and post-BF16 recurrent state to the existing K512 and K1024 scan variants; compare projections. Separately feed identical q/k/v/g/beta/state into `_recurrent_step` and the existing chunk64 graph with one valid row/zero padding, recording FP32 state and core output before gating. Finally compare SFPU SILU against the existing lookup on identical gate inputs. Change one component per contrast. This separates a producer error from downstream consequences of a changed intermediate and avoids attributing prefix differences to the singleton. A captured-operand HF FP32 scalar/chunk reference can adjudicate unexpected differences without rerunning35B weights. If a helper violates its arithmetic contract, fix that helper or the singleton prefill call site; do not infer that merely regrouping all operations is a bug fix.

3. **Same-input grouped-MoE schedule control.** Capture real layer0 pre-MoE x/IDs/scores for fresh first2048 and prefix68→96, verify common valid inputs first, then compare grouped routed outputs before shared expert/reduction. Repeat only with `stochastic_rounding=False`, preserving weights, dtype, fidelity, dispatch, and operand values. If common inputs differ before the grouped operation, first use one captured input in both padded/grouped contexts. Record counts and row mapping; include32→32 as the passing contrast and68→96 as the failing contrast. A collapse of common-row differences only when SR is disabled identifies stochastic scheduling, not a faulty transport or proof that all full-model error is explained. A production remedy would require an explicitly evaluated rounding policy or row-independent rounding design, including padding exclusion; globally changing accepted prefill rounding without that evidence is not justified.

4. **Only if the mismatch first appears at transport/terminal:** compare transported rows with the direct decoder input/output for s68's4/60 attention tails and68→96/444→448 MoE tails. For terminal, place the same captured hidden row at physical row0 and row27/31 and compare logits. This tests lane/packing behavior independently of upstream state. Fix count/extent/indexing only if the bytes disagree. Static offsets currently agree.

Keeping FP32 state across calls, forcing every prefill through singleton decode, recomputing prefixes, or relaxing the .98/.99 gates can reduce a symptom while changing the contract. None is selected as a repair. The narrow inherited HF sensitivity is real for s68/2528 under the measured implementation; it is neither a proof of mathematical impossibility nor evidence that TT must fail the same gate. After any justified change, rerun all four default starts, applicable controls, and the selected normal precision construction.

## Other potential issues and claims rejected on review

| Candidate | Source/contrast assessment | Disposition |
|---|---|---|
| Suffix written at residual row0 loses absolute position | Relative residual offsets and independent absolute cache/RoPE positions are consistent | No demonstrated bug |
| Partial page68 fill overwrites prefix | It uses singleton page updates until256, not a page fill at68 | Refuted by selected branch |
| Zero padding reaches conv/recurrent cache as extra tokens | DeltaNet uses explicit neutral padding; transport commits valid rows only | No logical state-advance defect found |
| Padded rows are entirely irrelevant | They are irrelevant to ideal bias-free output, but participate in router counts/SR scheduling | Concrete numerical test gap |
| FP32 recurrence is accidentally narrowed | Narrowing matches HF BF16 cache semantics | Expected boundary, not isolated bug |
| Full-attention singleton implementation alone explains s68 | Passing s32 uses224 such updates versus188 | Insufficient diagnosis |
| All failures reproduced by HF, therefore waive | s2559 HF passes; s68/2528 worst positions differ | False; gate remains FAIL |
| Historical reduced bounded-versus-legacy parity proves all40 split invariance | It compared different implementations of continuation on two layers/older sources, not uninterrupted versus split all40 | Limited historical evidence only |
| LoFi/BFP8 alone proves a bug | Both paths intentionally use these policies, and start32 passes | Amplifier/candidate, not proof |
| Snapshot/callback proves queued work is complete | Source queue ordering plus host-logit readback matters; enqueuing alone does not prove completion | Keep explicit completion boundaries |

The transport kernels place barriers before scratch reuse and writes complete before their kernel ends; state updates, metadata copies, attention, and cache consumers use the ordered command queue. Host logits are materialized in `receive_logits`, then compared on CPU (`M/tt/generator.py:410–418`; `M/tt/model.py:374`); these are stronger completion observations than a host assertion after enqueue. No concrete outstanding queue/count/lifetime contradiction was found. This is not hardware proof against all races.

## Provenance and report review

The report was revisited against planner output, selected Python dispatch, HF cache dtype ownership, native MoE argument lowering and formats, saved position mappings, the full four-case matrix, and the new HF singleton pass. Claims of a state-rounding bug, page68 fill error, common HF/TT worst row, or inevitable gate failure were rejected. The earliest divergence remains a **source-supported numerical mechanism**, not an observed all40 intermediate root cause. The next experiment should supply that missing intermediate evidence rather than another broad precision sweep.

Current HF model source SHA256 is `f738a9eb46d1351a751f36a1cae5f7435d589e8c3218e7315182419c4c0230f1`; installed runner SHA256 is `c1d66dc5e416f67f0cb209353fbadcaf6be8cada21727102d1879d46d8dbf2f1`. The matched controls record artifact/source hashes and exact tokens/positions. Their existing `.pt` hashes were not recomputed in this root pass; subagents read the raw saved tensors for finite/value/mapping checks. The TT run's provenance records commit and device visibility, not a complete source/binary hash set; the hashes below pin this inspection's current source, not all historical execution dependencies. Existing workspace edits and externally completed artifacts were left intact.

| File under M/ | SHA256 at inspection |
|---|---|
| tt/prefill.py | eda33d5d9023ca28d785a871ae58032685aff37558fc4f119cf439e42f28356c |
| tt/prefill_transport.py | bf436f0a8b82daa47bf0ffd6bc06f09b6ad876aaba7abbd555f432e606240061 |
| tt/multichip_decoder.py | 5b5696d6d2028f3ac5ccc5f0349f35916915d43130010ef9f835f7921da873fe |
| tt/fused_decoder.py | d8106611f65edccdf069baab44acc9c0088baa709090b017a6a6df75f0a7bb87 |
| tt/model.py | b314e1972dc8c3ed8131dc6a46e45ed453aa4d83712ca19ca4f2325e87e9a93c |
| tt/precision.py | d892fbd51e696e605101d5a19d48a53f197d16a3dcb4b7963a0d2f8b9d23ac18 |
| tt/mesh_state_slab_recurrence.py | 372daefaafc9d935d9d9bc298716858d8c973bf2f2ae46fe82db92c7b83ed7a5 |
| tt/generator.py | 5f9d84f0d9b9878260159600458b54512c3f06d77d67bebe6c4a986fc4043dbd |
| doc/datatype_sweep/baseline_resume_check.json | 89343adaad123f0e6a7ce6d6c06da019d44d2be7a10b40873ed70c06cc227d53 |
| doc/datatype_sweep/hf_resume_control/summary.json | ccf194a07bb328e9ad65d704fc1809d7b95137ff2addb05f5658de2fcd6763e4 |
