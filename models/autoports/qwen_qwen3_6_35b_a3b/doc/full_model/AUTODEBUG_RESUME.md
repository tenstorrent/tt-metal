# AutoDebug: Stage6 full-model retained-state resumed prefill

## Finding

**No source-supported wrapper, cache, position, or continuation-contract defect was isolated that explains the three reported PCC failures while preserving the accepted optimized decoder policy.** The packaged all40 check remains **FAIL**. The pinned HF BF16 control independently fails numerical gates in the same three cases. This supports the reported known limitation of numerical sensitivity to continuation; it does not support diagnosing TT-specific qualitative degradation, declaring the TT gate passed, or changing its thresholds or prompt.

The inspected continuation builds a prefix once and consumes its retained KV, convolution and recurrent state for the new suffix. No hidden prefix recomputation, intervening reset, double subtraction of the start position, missing logical token, or cache-page mismatch was found. The extra recurrent-state BF16 boundary and different full/split operation groupings also exist in the accepted decoder. Removing them would change the required policy, not repair a demonstrated wrapper inconsistency.

**No defect-backed reduced runtime experiment or implementation change is selected.** There is no isolated source defect whose causal intervention can currently be specified. This is an inspection result with a remaining numerical gate failure, not proof that no lower-level defect exists or that passing the gate under the stated constraints is impossible.

## Scope and revision consistency

Inspection only: source reading, JSON/hash verification, and standard-library arithmetic to reconstruct the case matrix. No target/model/device/HF execution, target imports, device reset, implementation edits, or AutoDebug relaunch occurred. Three independent xhigh investigators audited the wrapper/checker, decoder state/positions, and HF control. Their findings were checked against source and artifacts.

Git HEAD is `121ab5df0f494787236fff2b371e422e6edb599d`; full-model and low-level work is uncommitted. The prior mixed-serving issue remains separate in `doc/full_model/AUTODEBUG_MIXED.md`.

Unless an absolute path or repository-root prefix is given, paths below are relative to `models/autoports/qwen_qwen3_6_35b_a3b/`.

- `CHECKER`: `/home/ttuser/qwen36-35b-a3b/codex_home/plugins/cache/tenstorrent-skills/tt-model-bringup/0.1.24/runtime/readiness_check/run_prefill_resume_check.py`.
- `HF_IMPL`: repository-root `python_env/lib/python3.12/site-packages/transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py`.
- `HF_CACHE`: repository-root `python_env/lib/python3.12/site-packages/transformers/cache_utils.py`.

Every source hash recorded by `readiness/prefill_resume_check.failed.json` matches the inspected bytes: wrapper `c2265164…`, checker `c1d66dc5…`, generator `bcf6f84d…`, model `a0e5eb31…`, prefill `1979a8ad…`, and book `25d33cc3…`. The failed artifact SHA-256 is `cce222c5dd38c329839310732c038a41d835e39238e5210539509f0cf102ed62`, matching the HF control's input pin. The HF script, checker, book and `HF_IMPL` hashes also match their recorded pins. The paired verdict correctly pins both its result and the TT failure artifact.

The failed wrapper artifact does **not** individually pin the decoder, transport, every helper/kernel, or loaded native binary. Current decoder SHA-256 is `a57112adc86cb65b42f57974c22dc561cf792a384501b866eaafb5c21269923f`, also recorded by the B3 teacher artifact. This strengthens correspondence for that file but does not retroactively pin the failing run's whole stack. `HF_CACHE` is inspectable now but likewise lacks a recorded runtime hash. Older reduced-equivalence/mixed reports have different pins and are not substituted for this run.

## Direct observations

The TT artifact covers seven all40 cases on the1x4 mesh, context2048, raw natural-book tokens, starts derived from block sizes32/128, tile32, window64 and both packaged controls. Gates remain mean PCC >= .99, minimum PCC >= .98, top1 agreement >= .98, and equal final-position top1. PCC is computed separately over each selected vocabulary row, then averaged/minimized; this is not one flattened whole-window correlation (`CHECKER:119–165`).

| Length / start | Selected rows | TT mean PCC | TT min PCC | TT top1 agreement | Result |
|---|---:|---:|---:|---:|---|
| 257 / 32 | 65 | .996340726 | .990075736 | 65/65 | PASS |
| 257 / 128 | 65 | .993512290 | .858570058 | 64/65 | FAIL: minimum PCC |
| 257 / 256 | 1 | .993464653 | .993464653 | 1/1 | PASS |
| 1025 / 32 | 65 | .995993575 | .987099819 | 65/65 | PASS |
| 1025 / 128 | 65 | .995988802 | .983661673 | 65/65 | PASS |
| 1025 / 260 | 65 | .993122153 | .968665329 | 65/65 | FAIL: minimum PCC |
| 1025 / 1024 | 1 | .987020509 | .987020509 | 1/1 | FAIL: mean PCC |

Every final-position top1 matches. “All top1 gates pass” does not mean exact equality everywhere:257/128 has one disagreement. Both negative controls are rejected in each of the five65-row cases, giving ten expected rejections. Controls are intentionally skipped for the two singleton windows because `control_min_window=64` (`CHECKER:213–230`). They demonstrate sensitivity to absent or shifted history; they do not prove every retained state value is correct.

Program-cache entries remain9602 after preparation/checks. Model and sampling replay counters are0. The failed comparisons use eager prefill, not model/sampler trace replay. Stable counts establish program reuse, not numerical equivalence or correct runtime data.

The passing full40 prefill and B1/B3 teacher top5/top100 results, and B32 capacity/state check, remain separate evidence. They neither negate these PCC failures nor localize them. B32 is not a B32 HF-accuracy or resumed-prefill check (`doc/full_model/README.md:62–74,114–119`).

## HF control: supported correspondence and limits

Read together: `doc/full_model/qualitative/hf_resume_control/{result.json,verdict.json,README.md,run_control.py,blind_records.json,blind_scores.json,blind_labels.json,rubric.json,prompt_tokens.json}`.

The control uses original pinned revision `345d467b117d1fd5eda9b879a5988de9ae5f47a7`, BF16, CPU eight threads, Transformers5.12.1 and SDPA, with no model-policy overrides. Tokenizer/book and the installed natural-token, window and numerical evaluator helpers correspond to the TT path (`run_control.py:42–81,114–142`). Full references are independent of each prefix/suffix request; the latter retain their actual HF cache, with cache lengths checked after prefix and continuation.

| Length / start | HF mean / min PCC | HF top1 | TT mean / min PCC | HF failed gate(s) |
|---|---|---:|---|---|
| 257 / 128 | .987027380 / .496342693 | 64/65 | .993512290 / .858570058 | Mean and minimum |
| 1025 / 260 | .991617427 / .952856380 | 64/65 | .993122153 / .968665329 | Minimum |
| 1025 / 1024 | .988465489 / .988465489 | 1/1 | .987020509 / .987020509 | Mean |

HF is worse by both aggregate PCC metrics in the first two cases; TT is slightly worse in the singleton. Stored HF per-row PCC values reproduce the saved means/minima. All six blind metric records match the corresponding labeled source metrics and scores. The paired verdict preserves `tt_gate_waived=false`.

Important limits:

- This compares **each implementation's full versus split execution**, not TT logits directly against HF logits. HF uses its default contiguous cache and Torch recurrence; it does not emulate TT paged kernels or all selected arithmetic. The checker argument `[32,128]` chooses starts, not two physical cache implementations.
- HF stores per-row PCC/top1 and identifies worst positions135 and300 in the first two cases. TT stores aggregates only; its worst positions/mismatch IDs cannot be recovered. Case-level correspondence does not establish the same failing TT/HF rows or a shared causal mechanism. For the singleton the position is unambiguously1024.
- TT did not archive its token IDs. Source/tokenizer revision/book/helper correspondence supports intended token matching; independent equality of two archived token arrays cannot be checked because the TT array is absent.
- There is one deterministic HF measurement per case, explicitly reused under ten seed labels. Eight forwards report unchanged Torch RNG. These are not ten independent trials, repeated-run stability, a statistical failure-rate waiver, or generated-text quality evidence. HF does not repeat TT's negative controls.

The qualitative-check skill was applied only to distinguish numerical evidence from generated-content claims. Existing paired chat smokes (`qualitative/tt_qualitative_verdict.json`, `qualitative/tt_greedy/paired_review.md`) do not explain or waive this raw-book numerical gate.

## Wrapper and state contract audit

`tests/run_full_model_resume_check.py:80–104` constructs the required model/generator, prepares it and adapts only the checker's constructor. Packaged defaults and controls are unchanged; the checker owns evaluation/teardown. The wrapper asserts PASS and saves a failed artifact on exception (`:105–147`), so this is not a mislabeled success.

`CHECKER:203–209` resets before independent full/candidate calls. `tt/generator.py:683–694` resets once at helper entry, fills `[0,start)` when requested, then continues on the same `self.cache` with end=`len(tokens)` and `start_pos=start`. No reset intervenes. `_prefill_forward` clears convolution/recurrent history only when `start==0` (`:398–401`). The already-prepared executor returns from preparation without warmup/reset (`tt/prefill.py:263–267`).

The public contract carries a full token row but consumes `tokens[row,start:end]` (`tt/prefill_validation.py:65–70`; `tt/generator.py:416–422`). Absolute requested positions become `p-start` once (`:693`); the callback extracts and restores those requested relative rows (`:405–430`). The checker accepts the requested-row count directly, without indexing it again (`CHECKER:174–185`). Selected rows are the first64 continuation positions plus the final position, deduplicated (`CHECKER:113–116`). No off-by-one or double-offset discrepancy was found.

## Position, page and padding contract audit

Physical KV page size32 is fixed in `tt/multichip_decoder.py:1651–1687`. At context2048 each full-attention layer has64 physical pages of `[1,32,256]` per chip and an identity table; batch1 has no cross-request sharing. The checker `[32,128]` argument selects starts through an LCM (`CHECKER:90–110`), not a128-token physical KV format.

`tt/prefill.py:110–123` refreshes absolute position/RoPE indices per full-attention chunk. Multirow chunks supply a table window beginning at `absolute//32`. `tt/multichip_decoder.py:1746–1765` uses it for fill and the same absolute tensor for chunked SDPA. Singletons use the full table plus absolute position for paged update and decode SDPA (`:1725–1743`). Multirow chunks begin only at multiples of256, satisfying page and SDPA alignment.

The lowered path agrees: the legacy scalar-start path validates alignment to both query and key chunk sizes (repository-root `ttnn/cpp/ttnn/operations/transformer/sdpa/device/sdpa_device_operation.cpp:388–403`). Production bounded prefill instead supplies a tensor start. `.../sdpa/device/kernels/dataflow/reader_interleaved.cpp:350–371` reads it and divides by query chunk size32; this caller supplies starts aligned to256. The reader uses that offset for absolute cache bounds and query coordinates (`:409–416,575`). `ttnn/cpp/ttnn/operations/experimental/paged_cache/device/kernels/dataflow/writer_fill_cache_interleaved.cpp:22–38` maps local sequence tiles through the shifted table to physical fill pages. No relative/absolute-coordinate contradiction was found.

Transport copies exactly valid BF16 rows and zeroes only temporary padding (`tt/prefill_transport.py:34–43,113–129`); public logits exclude padding (`tt/prefill.py:212–230`). For a61-token recurrent tail, K/V/g/beta are zero-padded to64 (`tt/fused_decoder.py:2058–2064`), the final state uses cumulative row63 (`:2092–2094`), and output crops to valid length (`:2366–2371`). Those padded rows are algebraically neutral: zero g adds no decay and zero K/V/beta adds no update. This is not a bitwise-equivalence proof for every kernel. Convolution commits only real history using `history[:,s:s+3]` (`tt/multichip_decoder.py:435–445`). No padded logical-token commit or prefix overwrite was isolated.

## Complete case geometry and numerical policy

The bounded planner (`tt/prefill.py:25–38`) matches the accepted planner (`tt/fused_decoder.py:1807–1833`). Standard-library arithmetic reconstructed these suffix counts without executing target code:

| Length / start | Suffix tokens | Linear chunks relative to this call | Full-attention suffix work |
|---|---:|---|---|
| 257 / 32 | 225 | 3×64 +33 | 225 singletons |
| 257 / 128 | 129 | 2×64 +1 | 129 singletons |
| 257 / 256 | 1 | 1 | 1 singleton |
| 1025 / 32 | 993 | 15×64 +33 | 224 singletons, 3×256, final singleton |
| 1025 / 128 | 897 | 14×64 +1 | 128 singletons, 3×256, final singleton |
| 1025 / 260 | 765 | 11×64 +61 | 252 singletons, 2×256, final singleton |
| 1025 / 1024 | 1 | 1 | 1 singleton |

Full references use linear4×64+1/full256+1 at257 and linear16×64+1/full4×256+1 at1025. In the260 suffix, aligned full chunks start at512/768, fill-page offsets16/24, followed by page32/offset0 at1024. The128 suffix at257 visits positions128..256 exactly once. No hole, overlap or prefix-relative KV write was found.

Each public linear prefill rounds recurrent state BF16 then widens to its FP32 carrier (`tt/prefill.py:126–130`), exactly matching `tt/multichip_decoder.py:1544–1545`. Internal chunks pass `cache_boundary=False`; projection width is1024 for a multi-token public call,512 for a public singleton. A final one-row chunk within a longer linear call deliberately stays on the chunk path (`tt/prefill.py:132–142`; `tt/multichip_decoder.py:1547–1557`). These are accepted public-call rules, not accidentally derived from padded output width.

The inspected HF source also computes chunk recurrence internally in FP32 (`HF_IMPL:260–322`), initializes its persistent recurrent buffer with the BF16 convolution-cache dtype and copies returned state into it (`HF_CACHE:925–937,976–990`). Splitting introduces a stored-state rounding boundary. A cached singleton additionally switches to HF's recurrent single-token path (`HF_IMPL:521–544`). These are source-backed sensitivity mechanisms, not isolated causes of particular TT outliers.

Passing257/256 and failing1025/1024 share the public singleton branch and extra state boundary; both uninterrupted references end on a one-row chunk tail. Passing1025/128 and failing257/128 share prefix token/length geometry and the first two linear chunk geometries, but differ in extent, attention grouping and MoE grouping. This does not establish identical intermediate activations or states. Neither “singleton path,” “start128,” nor “an extra state round” alone completes the causal chain.

Batched decode explicitly selects deterministic grouped-MoE rounding; selected prefill retains its existing stochastic policy (`tt/multichip_decoder.py:1393,1452–1456`). Preserve that repair, recurrent consumption semantics, BF16 residual/KV, FP32 recurrent carriers, accepted weight dtypes/fidelity/layout and TP4/EP4 ownership (`doc/full_model/preserved_policy.md:3–19`). No blanket precision increase, less optimized path, host fallback, single-chip fallback or prefix recomputation is justified.

## Completion boundaries

Each compatibility callback calls `model.host_logits` before its temporary logits are dropped (`tt/prefill.py:226–231`; `tt/model.py:308–311`). `ttnn.to_torch` calls `from_device` (repository-root `ttnn/ttnn/operations/core.py:464–467`), whose binding defaults to `blocking=True` (`ttnn/cpp/ttnn-nanobind/operations/core.cpp:164–176`). Transport is serial on one command queue and barriers protect scratch reuse (`tt/prefill_transport.py:7–9,32–43`). No missing completion boundary was identified in this eager comparison path. This does not prove every native synchronization implementation correct.

## Candidate adjudication and other potential issues

| Candidate | Adjudication | Smallest justified intervention |
|---|---|---|
| Prefix reset or dropped history | Same cache spans prefix/suffix; reset precedes the pair; ten corrupted-history controls are rejected | None; no wrapper defect found |
| Wrong suffix/output index/page offset | Explicit absolute slice, one relative-index conversion, matching page/position inputs | None; no source contradiction |
| Recurrent BF16 boundary | Identical accepted decoder behavior, analogous HF storage, present in passing cases too | Preserve it; removing it changes policy |
| Singleton/chunk/projection switch | Matches inherited rules; singleton257/256 passes | Numerical mechanism hypothesis only |
| Padding/transport/cache-hit corruption | Valid-row bounds and serial metadata consumption are coherent; no divergent lowered input identified | Unproven; stable counts alone neither prove nor disprove it |
| Prior stochastic batched-MoE issue | Capacity1 and no decode replay here; accepted fixed-batch repair is separate | Keep repair; no global SR change |
| TT-specific qualitative degradation | Raw-book numerical check; HF also fails; paired chat review selects no quality failure | Unsupported classification |
| Native defect not exposed by inspection | Entire native stack unpinned; TT saved intermediates/per-row data absent | Residual uncertainty, not a ranked root cause |

## Smallest causal checks and disposition

The smallest supported checks were static and are complete: trace one prefix/suffix pair across reset/slicing; reconstruct all seven planner/page geometries; compare state boundaries with the accepted decoder and HF source; verify control pins, metrics and failure correspondence. None exposed a caller/callee contradiction.

**No defect-backed reduced device experiment is selected.** Running layers0/3 with altered precision, boundary, chunking or rounding would test a different path without establishing a port defect. A reduced pass cannot establish full40 passing behavior. The runner's `--reduced` option is explicitly diagnostic (`tests/run_full_model_resume_check.py:8–12,33,57`).

The missing evidence is identifiable: TT per-position PCC/top1 IDs and, if necessary, the earliest differing layer/state under original all40 execution. Saved aggregates cannot supply that correspondence. This report neither invents it nor requires a speculative hardware sweep. Any later observation must retain original tokens, positions, gates, optimized arithmetic and genuine continuation, and be treated as new evidence before selecting a causal experiment.

The user's acceptance of HF-equivalent sensitivity permits recognizing the known limitation and continuing other authorized work. It does not satisfy the separate explicit requirement that resumed prefill pass. Both HF result and paired verdict leave that TT requirement unwaived. Preserve the raw failure and **do not mark this gate or Stage6 passed** on this investigation's authority. Under the present constraints and available evidence, no supported minimal source fix can be isolated.

## Report review

After the first draft, an independent xhigh review and a source/evidence recheck verified the seven-case matrix, HF arrays/blind scores, provenance, cache boundaries and completion path. The review corrected a scalar-only SDPA validation citation and clarified that shared chunk geometry does not prove identical intermediate values. Arithmetic differences remain mechanisms/hypotheses, not bugs; historical mixed-serving and terminal-singleton issues remain separate. The headline is limited to no isolated source-supported defect and the continued failed gate.
