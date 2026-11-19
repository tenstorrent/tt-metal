# Batch geometry requirement reconciliation

Independent CPU-only analysis, 2026-11-14. This is not `$stage-review`, an acceptance verdict, a threshold change, a limitation-ledger disposition, or a promotion. The frozen `shared_pool_generator_v5` cross-geometry diagnostic remains **FAILED**. No production code or existing threshold was changed for this analysis. Existing artifacts were read and a separately labeled offline TT-to-HF top-ID report was generated.

The central distinction is between **reproducibility at the same execution history**, **agreement between different finite-precision execution histories**, and **accuracy against HF**. They test different properties. Neither replacing one with another nor treating TT B1 as a perfect HF oracle is justified.

## Requirements and their source

The original user requires preserving the optimized multichip decoder's dtype/fidelity/cache/activation/CCL policy and residual layout, with no fallback to single-chip, replicated, host-side or less optimized behavior. It also requires larger-batch serving state, explicit positions/page tables, inactive rows, traced device sampling/feedback, HF-referenced prefill/decode top5 >=98% and top100 =100%, qualitative HF controls, and work on unmet requirements or failed gates. The prompt does not state a universal PCC/top1 equality threshold between TT B1 and TT B2.

The installed `$full-model` skill adds explicit requirements for:

- deterministic greedy behavior on repeated identical inputs;
- logits for a given prompt reproducible across repeated runs and batch positions;
- batch >1 coverage for prefill, decode, cache/page-table indexing, feedback and output formatting, up to B32 when feasible;
- mixed-length batched prefill whose per-row outputs equal single-row outputs;
- resumed prefill and preservation of the decoder's selected policy/layout.

These requirements cannot be discarded merely because some other numerical comparison is overstrong. In particular, exact mixed-prefill and same-history permutation/lifecycle evidence remain required and are distinct from the historical later-token cross-width failure.

The exact local frozen decode threshold originates in `tests/probe_full_model_batch.py` (`result["thresholds"]` and the assertions immediately after `BATCH_DECODE_LOGITS`): finite outputs, whole PCC >=0.995, minimum row PCC >=0.99, top1 agreement =1, top5 >=0.98 and top100 =1 against TT solo. Its first decode consumes one forced token per prompt type; `probe_shared_pool_generator.py::decode_geometry_comparison` applies those numbers to later 42/43/44 histories as well. The local source explicitly says that numerical differences do not excuse changed greedy tokens or failed agreement. That is real recorded test intent, not something to delete retrospectively. It is not, however, a demonstrated HF guarantee or an explicit universal promise in the original user prompt.

The original batch comparator still contains the `argmax` versus `topk(...)[0]` tie-order pattern in its source. The parent separately corrected only the shared-pool comparator and archived its original; the V3 HF rescore proves only top1 and its derived verdict changed. No PCC, top5/top100 or numeric threshold changed. This analysis neither edits the original comparator nor treats a tie fix as a solution to the actual forced 43/survivor discrepancies.

## Four different invariants

| Property | What stays the same | Required interpretation and evidence |
| --- | --- | --- |
| Repeat/reset reproducibility | Logical input history, seed, precision path, active-width schedule and bindings | Repeated logits/tokens and request state must be reproducible. Existing same-geometry controls are exact; further current full48 integration evidence still matters. |
| Batch-position/permutation reproducibility | Each request's history and precision/width schedule; only its row/physical ownership changes | A reorder must preserve its cache, positions, RNG, tokens and outputs. Pooled/private and migrated/uninterrupted B2 exactness specifically test this. |
| Mixed-prefill and prefix continuation | Same logical prefix, masking and position semantics | The skill explicitly requires batched-equals-solo and resumed-prefill evidence. It is not waived by this decode geometry discussion. |
| B1 versus B2, or B2-history survivor versus uninterrupted B1 | Logical tokens/positions match, but the arithmetic branch and/or earlier cache-rounding history can change | Exact logits and invariant chosen top1 IDs are not automatic finite-precision properties. The frozen numerical gate tests a stronger bounded-agreement condition; its observed failure must remain visible and be reconciled explicitly. |

“Batch positions” naturally covers row relocation at unchanged batch geometry. A reasonable counterargument is that users also expect a prompt's answer to be insensitive to scheduler concurrency, so the phrase can be read more broadly. The analysis does not resolve that ambiguity by silently narrowing a declared public capability. It identifies which interpretation has mathematical and empirical support: the selected implementation and original HF do not provide a universal different-width exactness/greedy-ID guarantee. A serving implementation still has to meet its accuracy and state-isolation contract at every supported width it claims to have validated.

## Arithmetic policy and what AutoFix established

`MeshMoe._reduce` selects the accepted `_reduce_single` matmul for n=1 and native `post_combine_reduce` for n>1. The former uses the loader's HiFi4 FP32 DEST policy and BF16 output. The native factory explicitly uses HiFi4 with `fp32_dest_acc_en=false`, with the serial BF16 packing/accumulation observed in the ISA diagnosis. These branches can round a weighted expert sum differently even for identical expert values and routing weights. Retaining that split cannot simultaneously imply identical floating-point results for all inputs.

`AUTOFIX.md`, `batch_reduction_AUTODEBUG.md` and `moe_decode_capture_v1.json` establish more than a speculation: migration versus uninterrupted B2 and pooled versus private B2 are exact, while the first same-input TT difference is layer0 routed reduction. Exact ISA emulation explains the saved native values. A pure indexing/cache corruption explanation was not found in those controls.

The counterfactuals distinguish cause from an authorized repair:

- The BF16 batched-matmul schedule preserves BF16 DEST and improves forced 43 whole PCC to about 0.99012, but still fails the frozen gate. It is faster for B2 and slower for B32, and was not selected.
- The FP32 batched counterfactual makes the reduced comparisons exact, but changes the accepted batched precision. Its artifact deliberately remains acceptance-ineligible. It does not authorize production widening.
- A per-row B1 fallback would likewise evade the selected batched execution rather than satisfy the user's policy/performance preservation instruction.

The user has not authorized solving this by changing the dtype/fidelity policy or lowering the gate. Conversely, identifying intentional arithmetic does not prove every future same-policy optimization is exhausted or certify a failure as harmless. No claim of an exhaustive AutoFix failure follows from these two counterfactuals alone.

The prior optimized-multichip report validated selected layer outputs against its accepted reference using whole-output PCC and scoped B32/stack cases. It did not establish universal full-model B1/B2 vocabulary-logit or greedy-token equality over arbitrary decode histories. The stronger later local gate must be understood in that provenance.

## What the matched HF controls establish

The pinned original-HF reduced 0/1/3 CPU control runs the exact 3/33-token prompts and forced 42/43/44 histories. V1 shows deterministic B1/B2 differences; V2 adds the exact B2→B1 survivor history. Repeats, initial row reversal and native cache permutation are exact at the saved boundaries. Padding-only controls pass the numerical gate, and matching physical padding does not remove the batch-geometry failure. V3 separately corrects the known tie-ID scorer defect from stored tensors; forced 43 and every survivor comparison remain unchanged.

For forced 43, HF natural-B1 versus B2 has whole/minimum PCC 0.992546833/0.987764712 and top1/5/100=1/1/1. TT has 0.979649656/0.970201804 and 0.5/0.5/1. HF survivor versus natural B1 has PCC 0.995195317 but a real unique argmax change 83908→193953. TT survivor has PCC 0.994277325 and retains its own solo top1. Thus original HF itself rejects a universal version of the local different-width invariant on this backend. It does not reproduce identical failure magnitude or identical failed metrics.

The explicit user instruction says to run the matched HF control before chasing a qualitative/eval failure and to record a known limitation if HF does the same. This evidence supports a **narrow documented numerical limitation**: original-HF CPU outputs may deterministically vary with active-width history. It does not establish that TT's larger forced 43 self-distance has the same cause or severity, or that its top5 loss relative to TT B1 is harmless.

It also is not a substitute for the formal `$qualitative-check` exception procedure. That skill calls for matched prompt/sampling settings, at least ten seeds, blind scoring and a fixed comparative failure margin for qualitative/shared-suite failures. These deterministic tensor controls use two repeated seed-zero executions and no blind text rubric. That is appropriate numerical diagnostic evidence, but it must not be described as completion of the ten-seed blind qualitative protocol. The independently completed qualitative suite is separate evidence. Neither the broad “HF also varies” observation nor the narrow successful text suite automatically disposes of this failed numerical gate.

## Direct TT-to-HF top-ID evidence, kept separate

The local self-comparison asks whether a **TT B1** top token is inside **TT B2** top-k. The readiness accuracy direction is different: whether the **TT prediction** is inside **HF reference** top-k. A larger TT self-distance can arise because B1 is less faithful to HF than B2. Without this direction check, describing TT43 as a proven HF-accuracy failure overstates the evidence.

The new `saved_geometry_tt_to_hf_v1.json` computes the latter direction for every available per-row saved ranked ID with a matched HF history, plus exact argmax from the separate saved full-TT-logit capture:

- 12 rows have full TT logits at prefill/42/43, from the historical eager real-layer capture. Exact argmax and complete tie IDs are computed; each has a unique TT maximum.
- 54 rows are explicitly labeled saved `double().flatten().topk(5)[0]` IDs from the tracked shared-pool JSON. These include duplicate expected/actual and lifecycle controls, continued-B2 token44 and the token44 survivor. Full TT tie sets and true argmax are unavailable for these rows; they are ranked-ID evidence, not sampled-token proof.
- All 66 recorded IDs belong to their matching-history HF top5 and top100. This is not a 66-example independent accuracy score: it contains duplicates, two experiments and different evidence strengths. The report deliberately makes no aggregate acceptance claim.
- Three flattened multi-row baseline rebuild summaries are excluded from per-row extraction; their individual requests are covered by explicit row comparisons. The later unchanged-table resident decode has no matching saved HF continuation and is explicitly excluded. No unavailable token44 full TT tensors or solo-request1 token44 logits are invented.

The critical identities are:

| Phase/request/history | TT ID and evidence | Matching HF observation |
| --- | --- | --- |
| Forced43, short request, B1 | 78917; exact full-capture argmax and pool ranked ID | HF argmax 71111; TT ID remains in HF top5 |
| Forced43, short request, B2 | 71111; exact full-capture argmax and pool ranked ID | HF argmax 71111, top1 match |
| Forced43, long request, B1/B2 | 77693; exact full-capture argmax and pool ranked IDs | HF argmax 77693 in both histories |
| Forced44, short uninterrupted B1 | 81760; ranked-ID evidence only | HF top5 includes 81760 |
| Forced44, short survivor | 81760; ranked-ID evidence only | HF top5 includes 81760; HF unique argmax 193953 |

Therefore the TT43 top5=0.5 **self-comparison** does not mean the TT B2 prediction missed HF top5. In the saved rows, it did not. The larger PCC drift remains an observed TT numerical discrepancy requiring explicit disposition; this positive direct top-ID evidence neither deletes it nor supplies a universal bound on distribution/logprob error.

Reproducible offline command (no model or device execution):

```bash
python_env/bin/python -m models.autoports.qwen_qwen3_8_flash_next.tests.compare_saved_geometry_to_hf --output models/autoports/qwen_qwen3_8_flash_next/doc/full_model/saved_geometry_tt_to_hf_v1.json
```

The output refuses overwrite and records input hashes, source hash, exact HF backend, prompt/history mapping, all top100 IDs, strict token rank, deterministic argmax and tie-set distinctions. Its SHA256 is `52448e86c5c9200424b80d25d41dcf95133517c5cc392d7d6b6e017221191082`; helper SHA256 is `10dc1d8016a894dfe80e4d12551f252dd245ea86cd59d97a051d6907849b0ef2`. Repository-pinned cached formatting/import checks passed. Raw HF/TT artifacts and the failed gate are untouched.

## Evidence still required and counterarguments

1. **Complete current full48 HF accuracy.** Finish the current traced B32 run and inspect aggregate, per-phase and per-request top1/5/100, first misses, active widths, row permutations, metadata stability and state isolation. At assignment the parent reported no top5/top100 misses through 832 records; that is incomplete progress, not final evidence. Its 32 correlated prefixes from one AIME sequence are not 32 independent prompts, and the exact denominator must remain visible. Final B1 readiness on the final source is separate. The user top5/top100 bars remain unchanged.
2. **Close the precise reduced-history evidence boundary.** The offline artifact is useful historical evidence, but ranked IDs at44 cannot prove the actual common sampler's greedy choice or current production feedback. If needed to dispose of this specific diagnostic, a minimal current reduced run should save true device predictions/full logits for matched B1/B2/survivor histories and compare to the pinned HF tensors. Do not spend another all48 run merely to recover a few missing ranked-ID/tie facts.
3. **Retain same-history correctness requirements.** Exact repeated/migrated/permuted outputs, dormant request state, unchanged/changed page tables, public mixed-prefill/resume semantics and resident `reload_inputs=False` sampled feedback must pass at the current integration. Teacher forcing cannot prove feedback because it overrides inputs. Successful HF top-k accuracy cannot excuse a lifecycle or trace bug.
4. **Do not infer free-running quality from teacher forcing.** The completed matched/blind qualitative suite and AIME completions have their own revision/scope; current B1 final-source compatibility must be recorded. A single correlated B32 teacher-forcing sequence cannot guarantee quality on every prompt or bound all logprob differences. That limitation is an evidence scope, not grounds to invent an infinite test obligation.
5. **Address the contradictory gate intent transparently.** One defensible counterargument is that a locally frozen gate remains an acceptance obligation once adopted, especially under the user's “failed gates: work” instruction. Another is that enforcing a universal unsupported cross-width invariant can create pressure for a forbidden precision change even while direct HF accuracy is adequate. Neither is resolved by silently weakening thresholds, relabeling the old failure as passed, or declaring that every numerical difference is a known limitation. The stage owner/reviewer must explicitly reconcile the tested claim with the original requirements and evidence; this preparatory analysis does not perform that disposition.

If the cross-width gate is retained as an acceptance requirement, its real failures remain work despite current HF evidence. If a later review determines that its universal interpretation was overbroad, that determination must retain the original failed artifact and rationale, preserve required same-history and HF-accuracy gates, and state the supported numerical capability honestly. No such determination or capability change is made here.

## Sources reviewed

- Original user goal and its explicit dtype/fidelity preservation, HF accuracy, known-limitation and failed-gate instructions.
- Installed `full-model/SKILL.md`, SHA256 `5ff4cc73e9b1005e66ef86c2db473b9ae546f1753d296fd111d83f10a79ef14b`; `qualitative-check/SKILL.md`, SHA256 `5ca94560d885504e60cb94aaafa38433582335d41e956eb2c0ff4d736e81e8ed`, under the pinned 0.1.24 bringup plugin.
- Original threshold source `tests/probe_full_model_batch.py`, SHA256 `646a75a01d28a50b9333e678e6a8ace94248678f9b34c1feb3452614b05c1d94`; shared-pool original/corrected scorer and preserved v5 result.
- `AUTOFIX.md`, SHA256 `0b6a9f0492277b54a20631745ed391039c8017123ad6cf2540102fbb917b46cf`; arithmetic AutoDebug, real decode capture, BF16/FP32 counterfactuals and component timing.
- `shared_pool_hf_control_v1.json`, V2 JSON/PT, V3 separate rescore, `shared_pool_hf_control_interpretation.md`, current batch HF harness, full-model README and selected optimized-multichip README.

No stage-review verdict or acceptance promotion is provided.
