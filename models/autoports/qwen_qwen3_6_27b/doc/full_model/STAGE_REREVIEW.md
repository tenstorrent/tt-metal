# Stage Review

Verdict: more-work-needed

## Required Work

- P1: Inactive fixed slots are not inactive for full-attention KV state, and there is no safe per-slot reset/reuse contract.
  Evidence: `tt/generator.py` passes `active_mask` into every decoder layer and advances positions conditionally, but `tt/multichip_decoder.py::_full_attention_decode` unconditionally calls `paged_update_cache` for both K and V using every row's `current_positions`. The mask is applied only to linear-attention convolution/recurrent state. `tests/full_model_mixed_slots.py` asserts only that the inactive row's position remains 63; it never snapshots or compares that row's paged K/V. The README narrows inactive rows to “empty/reset slots” and says paused occupied slots are unsupported, while the only public reset clears every layer and every slot.
  Why this matters: the requested serving-ready low-level API explicitly includes fixed slots and inactive rows. An inactive request can currently overwrite its paged cache at the unchanged position on every replay. A completed slot also cannot be cleared and reused without destroying active requests, so the advertised scheduler-facing state contract is incomplete.
  Required next step: make full-attention paged updates preserve inactive rows (or route inactive rows to proven disposable blocks), implement an exact per-slot reset/reuse operation for paged K/V plus linear recurrent/conv state, and add a B2 public-wrapper test that snapshots inactive state across several replays and proves one slot can be reset/reused without changing the other.

- P1: The advertised 192,511-token public prompt maximum is not proven by a full-model/public-generator run and the terminal path makes the inherited decoder capacity result insufficient.
  Evidence: `doc/context_contract.json` and `Qwen36Generator.MAX_PREFILL_TOKENS` agree on 192,511, but the cited 192,511 pass / 194,559 failure comes from the earlier decoder prefill path. The full-model capacity artifacts allocate resident weights, state, KV, and a workspace reserve; they do not execute `Qwen36Generator.prefill_forward` near that length. The public wrapper always invokes `model.terminal_forward` for the entire physical sequence and concatenates vocabulary logits for all rows before host slicing, even when `return_all_logits=False`. At S192,511 this adds an enormous full-sequence TP4 vocabulary result not present in the inherited decoder capacity proof. Final public-wrapper evidence covers S261/B1 and S65/B2, not a non-aligned length near the advertised limit.
  Why this matters: the goal requires the model/generator to accept every valid non-aligned prompt through its supported context and permits a reduction only with evidence for the largest feasible full-stack value. Static agreement between a constant and JSON does not prove this materially larger wrapper allocation can run.
  Required next step: avoid materializing full-sequence vocabulary logits when only final logical prompt rows are requested, then run the public full wrapper at a non-aligned length near 192,511. If the terminal/full-wrapper allocation imposes a smaller hard limit, bracket the adjacent pass/failure and update the public contract to the largest full-stack value rather than inheriting the decoder-only result.

- P1: Final correctness and qualitative evidence is stale relative to the sampler-boundary optimization, while the only post-change all-layer token-out artifact collapses to EOS for every replay.
  Evidence: `models/common/sampling/tt_sampling.py` and `tt/generator.py` were changed at 19:49 to slice force-argmax logits to active rows and pad/copy only sampled tokens. The AIME autoregressive artifact predates this change (18:44) and the six-case shared suite predates it (19:25). The post-change all-64-layer artifact `artifacts/full_model_perf_active_row_b1.json` emits token 248046 (`<|im_end|>`) for the initial prediction and all 128 replays. Its harness truncates a rendered 170-token chat request at token 128, in the middle of user content and before the chat close/generation prompt, so this is not a valid chat control and cannot establish that the new sampling contract is correct. The post-change reduced profile repeats token 220 and proves timing/trace operation, not target-model free-running quality.
  Why this matters: the sampler/LM-head boundary is stage-critical, and the optimization was made after all coherent generation evidence. The visible repeated EOS anomaly must be controlled on the actual final code before the profiler win can close the stage. Older quality results cannot prove a later token-selection implementation.
  Required next step: rerun the fresh-reference autoregressive comparison and shared qualitative suite on the current active-row sampler implementation, read the outputs, and rerun the degeneracy check. Use a complete valid chat prompt for final token-out performance (choose a prompt whose rendered length is at most the requested benchmark length, or truncate user content before rendering), and preserve repeated-run deterministic greedy evidence on the final implementation.

## Other Concerns

- The public `setup_token_out_decode` API accepts a caller-owned cache and page table and the non-greedy B2 reduced wrapper passes, which closes the prior private-only sampling finding. However, changed page tables are accepted only as host `torch.Tensor` values after setup; device-owned scheduler updates are not supported. This is acceptable for the stated changed/unchanged refresh contract, but should remain explicit for later serving integration.
- `tests/full_model_perf.py` drives private trace fields directly rather than the public `token_out_decode_step`. The operations are equivalent today, but a benchmark through the public method would prevent performance evidence from silently bypassing future public-boundary work.
- The worktree contains unrelated untracked decoder-profiler, Falcon, and third-party paths. The eventual stage checkpoint must isolate stage-owned files only.

## Hard-Check Gaps

- Watcher cannot initialize the required TP4 fabric in this checkout: `logs/full_model_watcher_reduced_final.log` records ACTIVE_ETH firmware size 27,920 bytes exceeding the 25,600-byte kernel-config buffer before model construction. This is an exact instrumentation/toolchain blocker rather than evidence of a model failure. Named device profiling and ordinary traced runs cover the new terminal/sampler path, but no full-wrapper Watcher run exists.
- The B32 72,192/72,256 capacity bracket is a physical allocation probe, not execution at that context. The JSON/README now label it as physical residency, so it does not by itself create another required-work finding.
- The parser regression test covers an orphan END with no BEGIN. It does not test the stricter branch where a BEGIN exists and a mismatched END must still fail, though the implementation retains that assertion. This is a testing gap rather than a contradiction in the generated named reports.
- No local stage checkpoint exists yet. That is correctly deferred until an eventual clean-pass rereview.

## Anomaly Ledger

- Observed anomaly: the final all-layer performance run produces `<|im_end|>` for the first token and every one of 128 traced replays.
  Evidence: `artifacts/full_model_perf_active_row_b1.json`, token id 248046 repeated 129 times.
  Affected path: final active-row force-argmax LM-head/sampling boundary and device token feedback used for the reported 17.5168 t/s/u.
  Control or comparison: coherent AIME and six-prompt outputs exist, but both were generated before the active-row sampling change. The performance prompt is invalid because the harness slices a fully rendered 170-token chat conversation to 128 tokens mid-user-message.
  Likely subsystem: invalid benchmark prompt is a plausible cause of the first EOS; stale post-change correctness leaves the sampler-boundary change as an unexcluded cause of continued collapse.
  Investigation performed: decoded token 248046, reconstructed the tokenizer-rendered prompt, verified its length and token-128 truncation point, compared artifact/file timestamps, and inspected the active-row sampling code and older output artifacts.
  Resolution: more-work-needed.

- Observed anomaly: an inactive B2 fixed slot keeps its position unchanged but full-attention cache updates remain unconditional.
  Evidence: `logs/full_model_mixed_slots_lm_head_fix_v4.log` reports positions `[66,63]`; source calls both paged K/V updates without consuming the active mask; the test has no cache comparison.
  Affected path: public mixed-slot split-trace decode.
  Control or comparison: linear recurrent and convolution state have explicit active-mask preservation logic and prior PCC evidence; full attention does not.
  Likely subsystem: full-attention cache-update masking/page-table ownership.
  Investigation performed: traced the active mask from generator through model and both decoder layer kinds and inspected test assertions.
  Resolution: more-work-needed.

- Observed anomaly: full-wrapper Watcher setup terminates with a firmware-size fatal and then segfaults during teardown.
  Evidence: `logs/full_model_watcher_reduced_final.log` records the ACTIVE_ETH 27,920/25,600 fatal and teardown stack.
  Affected path: diagnostic instrumentation startup, before model construction.
  Control or comparison: normal TP4 fabric opens and all non-Watcher correctness/profile runs complete; inherited decoder Watcher evidence covers the unchanged decoder kernels.
  Likely subsystem: Watcher-instrumented fabric firmware and fatal-path teardown.
  Investigation performed: inspected the complete failure log and confirmed failure precedes model construction.
  Resolution: controlled as a tooling limitation; not treated as a model correctness failure.

## Scope Inspected

- Goal/skill paths: original Qwen3.6-27B full-model goal; `.agents/skills/stage-review/SKILL.md`; `.agents/skills/full-model/SKILL.md`; `.agents/skills/tt-device-usage/SKILL.md`; `.agents/skills/tt-enable-tracing/SKILL.md`; `.agents/skills/qualitative-check/SKILL.md`.
- Artifact paths: `doc/full_model/{README.md,work_log.md,STAGE_REVIEW.md,AUTOFIX.md}`, accuracy/autoregressive/trace/mixed-slot/Watcher logs, fresh-reference metadata, capacity JSON, qualitative shared suite and verdict, fallback audit, full performance JSON, and all three final profiler-report directories.
- Code paths: `tt/{model.py,generator.py,functional_decoder.py,optimized_decoder.py,multichip_decoder.py}`, full-model tests, shared readiness reference fix, common sampling change, Tracy parser fix, and its regression test.
- Commands run: read-only `git status/diff`, `find`, `rg`, `sed`, JSON/CSV inspection scripts, tokenizer prompt reconstruction, and `pytest -q tests/ttnn/tracy/test_process_ops_logs.py models/autoports/qwen_qwen3_6_27b/tests/test_full_model_public_contract.py` (7 passed, 1 unrelated pre-existing skip). No TT device was opened and no server or hardware experiment was launched by this review.

## Residual Risk

- The repaired profiler pipeline is credible: its regression passes, v4 reports contain named four-device operations, the common candidate-gather greedy path is materially slower, and the selected active-row path removes the 1.319 ms argmax while leaving sampler all-gather below the real LM-head cost. Accuracy gates, page-table refresh counters, and the public non-greedy B2 path are also strong for their tested pre-change/current shapes. The remaining risks are not evidence-format preferences: they are an unconditional inactive-slot state mutation, an unverified advertised maximum public prompt, and missing post-sampler-change target-model quality evidence in the presence of a visible EOS-collapse artifact.
