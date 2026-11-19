# AutoDebug: Qwen resumed prefill at L257 / start68

**The failure follows a change in chunk geometry; no incorrect cache offset, padded-token state advance, or causal-key omission was established. The precise numerical producer is unresolved.** At start68, the generator splits the reference's `[64,96)` block into a four-token prefix tail and a continuation block beginning at68, and keeps every later block shifted by four tokens. Both passing starts preserve the reference's overall chunk schedule. This makes chunk-dependent execution the leading investigation boundary, but does not make an arbitrary start invalid or prove that a particular kernel is wrong.

**Do not waive the readiness failure or change accepted precision.** The smallest proposed intervention is to process a 28-token continuation head `[68,96)`, then resume absolute 32-token boundaries. This is a discriminating experiment and candidate fix, not a verified repair: it cannot undo an already-different prefix state at68. The reduced `0,1,3` observer already present in the tree can identify where to investigate next without altering the installed readiness scorer.

## Scope and provenance

Inspection only: no device opens, model execution, reproductions, resets, source edits, or AutoDebug launcher invocation. Only this root report was written by the primary investigator. Three xhigh investigators independently checked the wrapper/scorer, GDN/PLE/hash state, and QSA/lowered cache paths. Cheap local checks parsed source and executed only the runner's two pure schedule/window helpers extracted with AST; no model or TTNN imports were used for those checks.

Repository HEAD is `2589dcde879a81213c03e5f5eaef5c85f68dc6eb`, matching the readiness record. Important full-model files are untracked, so that commit alone does **not** identify the runtime used by the recorded run. `tt/generator.py` changed externally several times during inspection, including its preparation/decode code and line numbering. The prefill semantics discussed below were consistent across the reads; current source cannot prove historical identity. The reduced probe also changed externally during final review: its initial fixed capacity384 was replaced with a capacity option defaulting to262144. Recommendations below reflect that latest probe. Line references to the generator refer to the snapshot below and include method names to survive line shifts. Paths abbreviated `tt/`, `tests/`, and `doc/` are under `models/autoports/qwen_qwen3_8_flash_next/`.

Inspected SHA256 values:

| File | SHA256 |
|---|---|
| `tt/generator.py` | `5129665ca4ec064fd2069eb0c3f3f256f5b3df7e5a5c03a14054b520acf19623` |
| `tt/model.py` | `96dce35d98ee4dbe0ec4f0ee674bfa0e51c8fef2cc38b03386bcb7d152abb875` |
| `tt/multichip_decoder.py` | `c2426640ad16075b14474faffb6edecb0488ecd1864cddf39ac807433df9876c` |
| `tt/fused_decoder.py` | `d638f10da2929e6898f63bac36151590c64eb5feb1ad184c3942053f59ae1b5b` |
| `tests/probe_prefill_resume.py` (final review) | `f39028425320b4830934aceba7e2859c844a823964aaa0084a7e2140a8d82add` |
| `readiness/prefill_resume_check.json` | `b52e8568501ff7a0226f3bd892e1c44070b04b8d5d4a26d31575404c89238534` |
| `doc/full_model/full_accuracy_v1.log` | `74a9e8811ee9ae2d817af60d3cffe0a8221fb170f5a6c2515b2c4264a079dbb2` |

The installed scorer inspected is `/home/ttuser/qwen38-flash/codex_home/plugins/cache/tenstorrent-skills/tt-model-bringup/0.1.24/runtime/readiness_check/run_prefill_resume_check.py`, SHA256 `c1d66dc5e416f67f0cb209353fbadcaf6be8cada21727102d1879d46d8dbf2f1`. References to “runner” below mean that file.

Final hash review found another external generator change to `b0218038a644a26a97bddca07715022e4addd73e23c01a5253d16e8707dc93d1`. Its `_prefill_chunk`, continuation helper, and relative32-token loop were re-read and retain the behavior described here. Other six model/decoder/probe/evidence hashes in the table still matched. The table retains the generator snapshot used for line references; this report does not certify later unrelated runtime edits.

## Direct observations

The JSON and `full_accuracy_v1.log:7522–7529` agree:

| Start / kind | Scored absolute positions | Mean row PCC | Minimum row PCC | Top1 agreement | Result |
|---|---|---:|---:|---:|---|
| 32 / block_aligned | 32–159 and256, 129 rows | 1.000003686 | 0.999964887 | 1.0 | Pass |
| 68 / tile_unaligned | 68–195 and256, 129 rows | 0.995711197 | 0.951207715 | 1.0 | Fail |
| 256 / prefix_cache_hit | 256, one row | 0.999984587 | 0.999984587 | 1.0 | Pass |

The floor is mean row PCC >=0.99, minimum row PCC >=0.98, top1 agreement >=0.98, and equal last-position top1 (`runner:65–69,150–165`). Only the minimum-row floor fails. Small PCC values above1 are possible because the runner's floating-point formula is not clamped (`runner:119–127`); printed “1.0000” is not proof of exact tensor equality.

Both negative controls fail as expected for starts32 and68. At68, no-history minimum/mean PCC is0.19967/0.80590; fresh-suffix is0.12931/0.80382. These controls show the retained history has substantial effect. They do not prove each retained-state element is correct. No controls are required at256 because the compared window has only one row.

The failing artifacts retain aggregate statistics only: no worst position, row PCC vector, intermediate tensor, routing IDs, or state comparison. It is impossible to determine from them whether divergence begins at68, grows gradually, spikes at a later boundary, or originates in a specific layer. Do not assign the minimum to position256 without a mapping. A final-row-only explanation is insufficient:128 rows at PCC1 plus one at0.951207715 would give mean0.999621765, appreciably above the observed0.995711197.

The “prefix_cache_hit” name comes from deduplicated runner geometry (`runner:94–110`); this case actually recomputes the prefix using the same generator. It does not demonstrate KV-only prefix reuse. The generator advertises `supports_prefix_caching=False` because recurrent state is also required (`tt/generator.py:33–42`).

The full accuracy JSON reports48 layers, serial execution, cache capacity262144, and top1/top5/top100 of0.95/1.0/1.0 for the separate prefill and teacher-forcing checks. Those are different comparisons from TT full-prefill versus TT resumed-prefill; they do not close this failed gate. Full-model status remains in progress.

## Exact boundary and ownership trace

All intervals here are half-open; tokens are zero-indexed.

| Path | Prefix calls | Continuation calls |
|---|---|---|
| Full reference | — | `[0,32),[32,64),[64,96),[96,128),[128,160),[160,192),[192,224),[224,256),[256,257)` |
| Start32 | `[0,32)` | Same remaining blocks as reference |
| Start68 | `[0,32),[32,64),[64,68)` | `[68,100),[100,132),[132,164),[164,196),[196,228),[228,257)` |
| Start256 | First eight32-token blocks | `[256,257)` |

The first changed **invocation** begins at64, before the resume itself. Start68 also changes the reference's final one-token invocation to a29-token invocation containing position256. Thus it changes prefix-tail count, row placement, subsequent group composition, and final-tail implementation together. The evidence does not isolate “68 crosses a cache page” from those other differences. Starts32 and256 reproduce the original overall schedule; their success is consistent with geometry sensitivity, without proving why it occurs.

`QwenGenerator.prefill_logits_continuation` (`tt/generator.py:585–600`) retains the full token tensor, prepares through exclusive prefix end `start`, then calls `prefill_forward` with the same cache, full prompt end257, and absolute start68. `prefill_forward` resets model state only for start0 (`:419–425`), binds the absolute attention position, and slices exact logical token ranges using `range(start,end,32)` (`:435–450`). It does not pass only the suffix while retaining an absolute offset. Returned rows are selected by `position-start`, matching the runner's convention (`runner:174–185`). No wrapper off-by-one or reference/candidate row mismatch was found.

`QwenModel.forward_chunk` (`tt/model.py:151–156`) runs these <=32-token chunks through every selected decoder before producing logits. It passes complete rank-local BF16 hyperconnection streams directly between layers. `MultichipDecoder.prefill_forward` (`tt/multichip_decoder.py:951–978`) takes its direct `_chunk` branch for every generator chunk; the decoder's larger-input output writer and `chunk_input.py` are not the extraction path for this reproduction. Post-forward logit padding to32 is for the persistent output buffer, after decoder state updates (`QwenGenerator._prefill_body`, snapshot `:304–309`).

For the same absolute position68, the reference is local row4 of `[64,96)`, while the continuation is local row0 of `[68,100)`. Both have logical count32 and physical one-tile height, but token placement and neighboring rows differ. The prefix fragment count4 uses a physical32-row tensor while histories have different physical heights, as detailed below. These are concrete lowered differences; whether one triggers a defect or changes ordinary rounding requires measurements.

## Stateful paths: suspected alignment bugs adjudicated

### GDN, PLE, and token history

The model uses `MeshGDN`/`MeshPLE`, not just the analogous functional decoder. On identical inputs and initial state, source expresses the same per-token causal operation for count4 and the first four rows of count32:

* GDN exact QKV reads token rows independently and preserves the same fixed product tree (`tt/fused_decoder.py:685–829`). Z/A/B uses the same prepared DRAM rows policy for Q4 and Q32, both with one physical M tile (`tt/multichip_decoder.py:38–49,224–249`; `tt/dram_projection.py:193–220,250–257`). No activation precision switch is selected at Q4.
* GDN concatenates three previous projected rows with exactly `tokens` current rows; applies four convolution taps; copies the last three logical rows; and normalizes over128 features (`tt/multichip_decoder.py:255–340`). The alternate experimental `conv_taps` branch is not installed by `MeshGDN.__init__`.
* Mesh recurrence passes logical `count=T`, with FP32 state `[1,12,128,128]`; the reader indexes `h*T+t`, and compute loops exactlyT times (`tt/multichip_decoder.py:353–488`; `tt/kernels/gdn_compute_sfpu.cpp:196–239`). Saving/reloading that FP32 state introduces no additional BF16 state boundary. This is source evidence, not a silicon-equality claim.
* PLE convolution has dilation3, four taps, and nine history rows (`tt/multichip_decoder.py:1490–1509`; `tt/ple.py:60–66`). N-gram buckets use logicalS=n, share the persistent history tensor, and advance only thoseS positions (`tt/ngram_hash.py:125–156`). The inspected reference `/home/ttuser/qwen38-flash/reference/modeling_qwen4_exp.py:1195–1226` agrees on PLE context/dilation, and `:1109–1153` agrees on two-token n-gram history/EOS semantics. The target config supplies ngram3/kernel4 (`doc/functional_decoder/target_text_config.json:91,102`). This is source correspondence, not measured numerical equivalence.

Expected state after consuming tokens0..67 is GDN conv projections65..67, PLE conv inputs59..67, n-gram raw IDs66..67, and68 recurrent updates. The shape ledger is:

| Logical count | GDN history logical/padded height | PLE history logical/padded height |
|---:|---:|---:|
| 32 | 35 /64 | 41 /64 |
| 4 | 7 /32 | 13 /32 |
| 29 | 32 /32 | 38 /64 |
| 1 | 4 /32 | 10 /32 |

Native concat handles padding along the concatenated axis by untilizing through logical endpoints, concatenating row-major values, then retilizing (`ttnn/cpp/ttnn/operations/data_movement/concat/concat.cpp:97–139`). Unaligned tiled slice starts use the row-major path (`.../slice/slice.cpp:225–239`). Therefore appending28 phantom padding tokens to these histories is **not** what this source does. No concrete advance-by32 or state-tail indexing violation was found. Different primitive/program variants remain candidates only if the direct split experiment finds a difference.

### QSA positions, compressed keys, and paged KV

Start68 is divisible by4. It is unaligned to the32-token tile/page, but aligned to QSA compression groups; blaming a partial four-token group at this split is incorrect.

Production `MeshQSA.forward_chunk` computes absolute `positions=state.position+row`, `group=floor(position/4)`, and a compressed-key write only when `position%4==3` (`tt/multichip_decoder.py:1057–1114`). Three raw-history rows bridge chunks. The caller advances by the logical count (`:941–942`; `tt/qsa_chunked.py:75–80`). Concrete checks:

* At68, full local row4 and resumed local row0 both have group17, no compressed-key write,17 complete groups, and causal tail token68.
* At71, full row7 and resumed row3 both pool raw68..71, write compressed group17, and apply RoPE at68. Their different history-array indices refer to the same four logical inputs.
* KV updates receive the absolute position tensor (`:1104–1113`; `tt/qsa.py:173–176`). With page32 and identity page table, token68 is page2, row4 in either path. The lowered update reader computes page-table entry `update_idx/block_size`; the writer uses `(update_idx%32)*Wbytes`. At local D256 BF16, that is page2's first tile plus row byte offset `4*512=2048` in both paths (`ttnn/cpp/ttnn/operations/experimental/paged_cache/device/kernels/dataflow/reader_update_cache_interleaved_start_id.cpp:123–128`; sibling `writer_update_cache_interleaved_start_id.cpp:98–115`). Both reader and writer recognize the -1 incomplete-group sentinel and skip it (`:86–89` and `:76–78`, respectively). The local selected-KV reader uses the selected token's absolute page and row, not the query's chunk-local row (`tt/multichip_decoder.py:1193–1256`). Query-row coordinates only select that query's ID list and output batch.

The fused selector actually selected by production is `FusedChunkedQSA._selected_ids` (`tt/fused_decoder.py:1388–1391`), not the unfused `ttnn.topk` path. It admits all complete groups until its512-group heap fills (`:1258–1308`). Here `floor((p+1)/4)<=64` for everyp<=256. Consequently every causal key is retained: complete groups plus the uncompressed causal tail. A score perturbation can change the order of selected keys, but cannot remove a causal group at this length. Identical relevant scores and positions imply identical ordered IDs: the heap produces descending scores with ascending group indices at ties (`:1225–1227,1297–1302`); moving a query row alone does not change that order. This rules against a changed sparse top-k membership as the direct explanation. It does **not** rule out order-sensitive finite-precision attention, or bad upstream Q/K/V values.

Production SDPA remains HiFi2 with FP32 destination accumulation, exact exponential mode, Q chunk32, K chunk128; QKV/WO retain their separately selected HiFi4 DRAM policies (`tt/multichip_decoder.py:1035–1054`). The number of queries becomes the SDPA batch axis, so full versus partial chunks can also alter work placement. No contradiction with the lowered cache/selector coordinate contracts was found.

## Completion, trace, and numeric-path limits

`_prefill_chunk` lazily captures a trace per `(count,slot)` and replays it nonblocking onCQ0. The host continuation requests all logits and reads each chunk back using `ttnn.to_torch`; preparation explicitly synchronizes. The failure is a completed host-logit comparison, not evidence of a later teardown hang. Prefix work uses the same CQ, preserving enqueue order before continuation and readback.

Do not infer that first-use trace capture executes a chunk twice. FD trace begin enables bypass (`tt_metal/distributed/fd_mesh_command_queue.cpp:1626–1643`); enqueue records trace nodes and returns without normal dispatch (`:443–456`). The explicit replay is the execution. Capture/cache/lifetime problems are not categorically excluded, but no sequence-matched violation was identified. Q4 andQ29 first-use traces versus already-usedQ32/Q1 are a useful separate control if eager execution differs from replay.

There is a definite count1/count>1 MoE reduction change: `MeshMoe._reduce_single` uses an ownership-masked feature-parallel matmul; multirow `_reduce` pads only the reduction to32 and calls native `post_combine_reduce` (`tt/multichip_decoder.py:748–801`). The native kernel streams each token's10 expert slots, skips nonlocal experts, overwrites the first local contribution, then accumulates later ones through L1 packing (`ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/post_combine_reduce/device/kernels/deepseek_moe_post_combine_reduce_compute.cpp:80–158`). Thus position256 takes genuinely different arithmetic at start68. That difference alone cannot explain the saved mean if all other rows matched; it belongs in the numeric-path investigation, not as a complete root cause.

Router scores, BF16 rounding, expert assignments, attention key ordering, and subsequent state can amplify an earlier small difference through48 layers. None of the existing artifacts records that causal chain. In particular, all top1 agreement is compatible with substantial changes elsewhere in a vocabulary vector and is not evidence that0.9512 is harmless quantization noise.

## Ranked diagnosis and falsifiers

| Candidate | Direct evidence and fit | Missing precondition / decisive check | Status |
|---|---|---|---|
| Chunk-dependent numeric/primitive execution, potentially amplified through the stack | Only failing case changes prefix-tail shape and all subsequent chunk placement; code has count/row-dependent execution, including final MoE reduction | Locate first differing output/state and compare eager versus trace on identical inputs | Leading **boundary**, producer unresolved |
| Prefix-tail state error at68 | Count4 has different history physical extent and programs than32 | Source logical counts/tails are coherent; compare full32 against4+28 from common state64 | Unproven; test first |
| QSA key permutation / query-batch execution | All causal keys retained, but their order and SDPA batch geometry can differ | Show selected-ID order or Q/K/V divergence at the same absolute query and compare attention output | Secondary numeric hypothesis |
| Wrong cache page, reset, compression phase, suffix offset, or padded state advance | Would explain a generic resume failure | Concrete wrapper/state/cache ledgers above contradict those proposed mechanisms | Not supported as headline bugs |
| First-use trace/cache state | Counts4/29 introduce trace keys absent from the canonical schedule | Paired warmed-eager and traced split replay; first-use versus repeated execution | Test gap, no observed contract violation |

This ranking does not claim normal low-precision drift has been established. A primitive correctness defect remains possible. Missing indexed runtime evidence prevents ranking a particular low-level instruction or architecture-specific mode as the root cause.

## Discriminating reduced test and minimal proposed fix

No test below was run in this inspection. Use the hardware owner's normal serialized lane environment and restore the accepted mesh descriptor after sourcing it. Keep the accepted1x4 mesh/ring setup and all dtype, fidelity, weight, KV, recurrence, and routing policies.

1. **Run the already-added transparent observer with layers `0,1,3` at full cache capacity.** It invokes the installed readiness runner with natural-book L257, starts32/68/256, window128, both negative controls, and unchanged thresholds (`tests/probe_prefill_resume.py:144–189`). It records prompt IDs and per-position PCC, maximum/RMS logit difference, top1 margin, and both chunk starts from the existing returned host logits (`:42–107`), adding zero model forwards. Keep the current failing acceptance artifact intact.

   ```bash
   timeout 1800 python -m models.autoports.qwen_qwen3_8_flash_next.tests.probe_prefill_resume \
     --layer-indices 0,1,3 --capacity 262144 \
     --output models/autoports/qwen_qwen3_8_flash_next/doc/full_model/reduced_resume_0_1_3.json
   ```

   If needed, run the same command serially with `--layer-indices 0`, `1`, and `3`, each with its own output. Layer0 isolates GDN withoutPLE; layer1 includesPLE/hash/GDN; layer3 includesQSA. These still share MoE, hyperconnections, final mixer/head, and generator, so failure in one layer kind does not uniquely implicate its attention. Omitting layer2 and using reduced residual distributions also limits correspondence to the48-layer run.

2. **Control the intervention boundary before changing precision.** Pair the existing schedule with a continuation whose first chunk ends at the next absolute32 boundary:

   ```python
   offset = start
   while offset < end:
       n = min(end - offset, 32 - (offset % 32))
       # Existing exact-token upload, model trace, state updates and logits read.
       offset += n
   ```

   At68 this yields `[68,96),[96,128),[128,160),[160,192),[192,224),[224,256),[256,257)`. Prefix processing remains32,32,4; starts32 and256, and start0, retain their current schedules. AllQ1..32 input buckets already exist and are warmed. This is the smallest proposed wrapper change: normalize the chunk plan to absolute positions while retaining arbitrary public start positions. It does not pad or replay already-consumed tokens, reset recurrent state, or require aligned caller input.

   The current observer's `continuation_chunk_start` formula assumes the old relative schedule; update that diagnostic formula or record the actual chunk plan in a future experiment. Otherwise an improved run would carry incorrect boundary metadata. Pair both schedules using identical prompt IDs and equivalent fresh state, retain all129 row statistics, and compare state at96. Passing this control would support the wrapper boundary as a useful intervention; it still requires full-stack revalidation.

3. **If the control fails, locate the first producer from a common input/state.** Save or reconstruct the same module input `[64,96)` and state64. Compare full32 with split4+28. Compare the first4 outputs, all32 outputs, and state96 separately. For GDN inspect projectedQKV/ZAB, conv, normalizedQ/K, gates, FP32 recurrent state/output; forPLE inspect exact hash IDs/history, embedding, gated input, conv and state; forQSA inspect projectedQ/K/V, absolute positions, raw history, compressed keys, KV rows, selected-ID ordered arrays and sets, then SDPA. Include MoE router logits/IDs and contribution, residual writes, final mixer and head until the first actual divergence is found. Use identical module inputs to distinguish local implementation error from inherited upstream drift. If the traced path alone fails, repeat the same prepared buffers/state eagerly and audit capture/cache state at that boundary.

4. **Protect the scope of reduced results.** The initially inspected probe used fixed capacity384, while the failure used262144. An external edit during final review added `--capacity`, now default262144 (`tests/probe_prefill_resume.py:114,128,156`); use that full-capacity setting for the primary diagnostic. At384, compressed-key width is96 versus65536 (`tt/multichip_decoder.py:872–886`); Q>1 merged-head score work is one block versus32 blocks of2048 (`tt/fused_decoder.py:1183–1205`). A deliberately smaller-capacity pass remains diagnostic, not validation of the original cache/program path. If the reduced stack at full capacity does not reproduce, collect the same row observer on the original full stack. Save source hashes, effective configuration, and concrete chunk intervals with every run. The external probe edit is preparation for a future test, not new runtime evidence.

If the aligned-head experiment fixes the reduced and full48-layer cases without damaging the controls, the minimalfix is that generator chunk-plan change, plus correct diagnostic metadata. Otherwise do not ship it as the diagnosis; repair the first proven producer/consumer contract instead. Re-run the unchanged original L257 readiness gate at full capacity and retain the original failed artifact alongside the result. Required floors remain0.99 mean row PCC,0.98 minimum row PCC,0.98 top1, and matching last top1.

## Accepted policy and other potential issues

The accepted optimized-multichip decoder explicitly permits nonaligned public lengths and owns internal padding/chunking (`doc/optimized_multichip_decoder/README.md:46–48,67–71`). Its whole-output PCC0.995 gate and diagnostic per-row PCC (`:78–83`) are different from the full-model readiness minimum-row gate. The saved **mean of row correlations** is not whole-output correlation;0.995711 cannot be used to claim that either original contract has been satisfied by the other statistic.

Preserve BF16 activations/interlayer residuals and page32 KV, FP32 recurrent state, selected DRAM policies, QSA remaining-attention HiFi2, and routed-expert BFP4/LoFi (`tt/multichip_decoder.py:38–49,629–635,815,1039–1047`; policy README). Raising fidelity/dtype, changing top-k/tie behavior, disabling a layer type, changing context support, or lowering the PCC floor would confound this diagnosis and alter accepted policy.

Repeated matmul memory-config warnings in the historical log are not an independent diagnosis. The configured DRAM projection path explicitly restores interleaved BF16 output after native matmul (`tt/dram_projection.py:210–228`), and the warnings also occur outside the failing case. No position-correlated corruption was demonstrated.

The current minimal artifacts omit exact runtime-source provenance and indexed logits/state evidence. Those are evidence gaps, not additional model bugs. After drafting this report, the headline candidates were checked again against wrapper ownership, logical/padded history lowering, production fused selector dispatch, concrete68/71 coordinates, and trace-recording behavior. Suspected padding advance, compression-phase failure, causal-key loss, duplicate trace execution, and a final-token-only explanation were rejected or narrowed accordingly. The unresolved producer is stated explicitly rather than promoted from a plausible numerical story.
