# AutoDebug: Stage9 native reasoning-budget control

Inspection date: 2026-11-16. Repository HEAD: `c285ee189e21b9810ef97d62c83d5ab3f0d5834b` (existing dirty worktree). Plugin HEAD: `629f07cb385494b14d0ee3d138de039d71d09546` (clean). This investigation used source reads, existing logs, JSON/hash comparisons, and checkpoint metadata reads only. No target modules were imported, no target or device workload was executed, and the already-running native process was not modified. Only this report was written.

## Findings

**No concrete reference-runner, loader, mask, cache, RNG, or protocol bug was found that explains bad_words_3. The evidence identifies a sampled native EOS before the reasoning-end marker, not exhausted reasoning allowance.** The simplest remaining explanation is an ordinary early-EOS sample under the native model's constrained distribution. That interpretation is provisional: terminal logits and RNG state were not recorded, so inspection cannot certify that the underlying model distribution was numerically correct.

The failed sample contains 16 generated tokens, ends on 248046 (`<|im_end|>`), and has no 248069 (`</think>`), tool-call marker, or final answer. It stopped in 13.443 seconds with a 4096-token/7200-second discovery allowance. The runner correctly records established=false and error “No reasoning-end token.” It does not silently count this as a usable completion. Raising the cap cannot repair this already-recorded trajectory.

**Keep two separate issues visible:** successful native choice/regex/plain and bad_words_4 trajectories exceed the tests' original total caps, while native bad_words_3 fails the final-answer contract despite ample room. Neither fact proves why initial served bad-word response 0 had content=None. Native bad-word seeds 0/1/2 fit cap 100, seed 4 needs 770 tokens, and matching HF/vLLM seeds do not promise matching sampled trajectories.

**The last native case completed during report review.** Final evidence has eight assertion-valid samples and one early-EOS failure, complete=true, budget_established=false, and no suggested_extra_allowance. Applying the sizing formula only to successful samples yields a conditional diagnostic extra allowance of **1,528**, superseding the earlier seven-success calculation of 180. **This run did not establish a usable allowance for all nine cases.**

## Evidence scope and direct observations

Path abbreviations:

- M = models/autoports/qwen_qwen3_8_flash_next in this checkout.
- E = M/readiness_vllm/hf_reasoning_budget_v1.
- H = python_env/lib/python3.12/site-packages/transformers in this checkout.
- P = /home/ttuser/qwen38-flash/vllm-tt-plugin.
- V = /home/ttuser/qwen38-flash/vllm-env/lib/python3.12/site-packages/vllm.

At the final inspection snapshot, **2026-11-16 16:13:38 UTC**, E/manifest.json had complete=true, budget_established=false, nine rows, and no suggested_extra_allowance. bad_words_4 had completed with EOS after 770 tokens in 506.722 seconds. An earlier snapshot at 16:05:51 had only 572 logged tokens for that case; the completed artifacts supersede that partial observation. “Established” below means the runner's answer assertions passed, not a broader semantic-quality judgment.

| Case | Base cap | Reasoning | Boundary | Answer including EOS | Total generated | Result |
|---|---:|---:|---:|---:|---:|---|
| choice_0 | 8 | 69 | 1 | 2 | 72 | Established; blue |
| regex_1 | 16 | 67 | 1 | 5 | 73 | Established; LANE-1 |
| json_2 | 64 | 41 | 1 | 13 | 55 | Established; status ok, lane 2 |
| plain_3 | 16 | 43 | 1 | 7 | 51 | Established; Request 3 acknowledged. |
| bad_words_0 | 100 | 89 | 1 | 5 | 95 | Established; Greet! |
| bad_words_1 | 100 | 16 | 1 | 4 | 21 | Established; Something. |
| bad_words_2 | 100 | 29 | 1 | 4 | 34 | Established; Hallo! |
| bad_words_3 | 100 | **No boundary found** | 0 | 0 | 16 | Unestablished; early EOS |
| bad_words_4 | 100 | 763 | 1 | 6 | 770 | Established by assertions; Pineapple! |

Counts come from the individual E/<case>.json records and matching manifest rows. All nine completed JSON records exactly match the corresponding JSONL token sequences with indices 0..N-1. All nine end in EOS 248046. None contains any of the 12 banned token IDs. Successful plain/bad-word answers have leading newlines retained in the artifacts. The null reasoning count for seed 3 means no completed reasoning span was established; it does **not** mean zero generated reasoning.

E/bad_words_3.json:4-30 and JSONL line 16 give the exact anomaly:

```text
generated IDs:
[1596, 1144, 310, 5707, 310, 1156, 25, 328,
 44240, 264, 4145, 40719, 310, 728, 30, 248046]

raw text:
We need to respond to user: "Say a simple greeting to me?<|im_end|>

finish_reason: stop
reasoning_tokens: null
answer: ""
established: false
error: No reasoning-end token
```

All five bad-word cases have the **same 56 prompt token IDs** and exact user message “Say hello to me.” Rendered prompts include the checkpoint's default xhigh system instruction and end with the assistant role and opening think marker. References to “greet,” “something,” or German inside generated text are generated paraphrases; they are not evidence that the request prompt changed. E/requests.json exactly matches the preparation validation's preflight_requests.

Provenance checks:

- All **12** files listed in E/manifest.json.source_sha256 match their launch hashes, including both focus scripts, plugin test files, checkpoint template/generation config, and the recorded vLLM/xgrammar source files.
- Manifest versions/paths identify original torch 2.11.0+cpu, original transformers 5.22.2, and serving-installation xgrammar 0.2.7. The runner imports torch/transformers before appending the grammar site (M/tests/run_hf_reasoning_budget.py:176-182). It does not import vLLM to execute this control.
- E/hf_reference_loader/loader.json records Qwen4ExpForCausalLM, **48 layers**, original per-tensor dtypes, eager attention, and **251,487,307,800 mmap parameter bytes**, at checkpoint revision 812d96e10a0dd82b69d749e68693698cf6c36daa.
- Native Transformers model/cache/generation sources are **not** among those 12 launch hashes. Conclusions about those installed files assume they have remained unchanged since launch. This is a provenance limitation, not evidence of a change.

## Executable path producing early EOS

### Inputs and masks

M/tests/run_hf_reasoning_budget.py:48-85,203-210 fixes the nine requests and renders the checkpoint template without overrides. Bad-word cases use temperature 1, explicit top-p 1, checkpoint top-k 20, repetition penalty 1, and seeds 0..4 (265-286). P/tests/tt/utils.py:38-40,103-104,118-126 supplies top-p 1 and omits unspecified top-k. V/entrypoints/openai/chat_completion/protocol.py:660-674 resolves request values before server defaults; V/config/model.py:1819-1846 includes checkpoint top-k. No top-k=0 versus top-k=20 discrepancy was found.

The runner tokenizes the accepted unprefixed/space-prefixed forms (88-100), matching V/sampling_params.py:740-773. Recorded banned IDs:

```text
14556, 23066, 9419, 21251, 5834, 15131,
12675, 20375, 34385, 33076, 18103, 26659
```

All are **single-token** bans. None is EOS 248046/248044 or end-think 248069. The bad-word operation therefore sets those 12 scores to negative infinity on every generated step. No multi-token suffix or empty-prefix ambiguity explains a seed-specific failure. The code matches V/v1/sample/ops/bad_words.py:9-27; V/v1/sample/sampler.py:381-404 supplies output_token_ids.

“Output-only” excludes prompt history; it does **not** mean final-answer-only. Bans apply during generated reasoning in both paths. This can affect reasoning that tries to repeat “hello” and alter subsequent text/probabilities. It is an intended constraint, not a discrepancy. Because these bans are singletons, changing the history itself would not change this particular mask.

Plugin ownership was also traced: P/src/vllm_tt_plugin/model_runner.py:4104-4116 routes bad-word requests to host sampling; output history is materialized at 2718-2719, banned-word row indices are remapped at 2770-2776, and metadata reaches host sampling through 4439-4523. This supports the source-level semantic comparison, not a claim about unrecorded live logits.

The presence penalty applies only to choice_0, once per unique generated token (runner:125-127), matching V/model_executor/layers/utils.py:64-79. It is zero for bad_words_3. A grammar matcher exists only for the three grammar cases (runner:106-110,268-270), so **xgrammar cannot directly mask the anomalous bad-word case**. For grammar cases, masks begin immediately after generated end-think (133-139), with both checkpoint EOS IDs passed as matcher stop IDs. The structured-output manager defers grammar until reasoning ends (V/v1/structured_output/__init__.py:235-290); backend_xgrammar.py:84-100,125-130 matches compiler settings and stop-token override.

The preparation's CPU semantic validation checks penalties/bad-word histories, grammar gating, canonical JSON whitespace, both EOS IDs, and equivalent decoded vocabulary across tokenizer installations. These are useful semantics checks. They did not run the full model or validate seed 3's actual terminal logits.

### Selection precedes stopping; no EOS insertion

The runner explicitly passes eos_token_id=[248046,248044], forced_eos_token_id=None, pad_token_id=248044 (282-284). The saved generation config has no stop strings, forced BOS/EOS, or positive minimum output length. H/generation/utils.py:2064-2104 resolves per-call kwargs last, after checkpoint/global defaults.

H/generation/utils.py:1586-1614 merges the custom processor **before** sampling top-k/top-p filtering. The bad-word path is:

```text
model last-position logits -> FP32 copy -> mask 12 banned IDs
-> top-k 20 -> softmax -> torch.multinomial -> append selected ID
-> EOS/length/time stopping criteria
```

Selection/append/stop occurs at H/generation/utils.py:3240-3281. H/generation/stopping_criteria.py:590-592 checks the latest generated token against EOS IDs; EOS already in the prompt does not stop generation. This is one CPU sequence, so completed-other-row padding cannot manufacture recorded 248046; padding would be 248044 anyway. Deferred/asynchronous stopping is MPS-only (generation/utils.py:442-450); CPU StopCheck.finish() does not undo a step (385-400).

The token logger sees generated tokens on the next processor call, but a post-generate loop writes any remaining terminal token (runner:119-124,288-292). Logging EOS there is expected, not evidence of a lost end-think. The final sequence and JSONL agree exactly.

Given this path and the artifacts, **248046 was the selected EOS**, not a rendering artifact, cap-generated EOS, or padding. Whether the logits making it selectable were correct remains a separate question that token IDs cannot settle.

### Loader and request state

M/tests/full_model_hf_reference.py:59-104 pins the snapshot, loads original tensors with assign=True, checks that only explicitly replaced PLE embedding keys are missing, replaces those embeddings with checkpoint-row lookup, rematerializes nonpersistent RoPE buffers, rejects remaining meta tensors, and sets evaluation mode.

Read-only checkpoint header inspection verified **128** PLE shards, numbered 0..127, each BF16 [2500012,160]. The three small I64 PLE buffers are present; cumulative head offsets and total vocabulary 320001446 agree, with padded size 320001536 = 128 * 2500012. Thus the loader's divmod(row,2500012) and width 160 (44-52) agree with this pin. HF's canonical conversion concatenates those row shards (H/conversion_mapping.py:2131-2140). The checkpoint already contains fused expert gate_up_proj/down_proj tensors; converters for separately stored experts do not imply a missing conversion here (1200-1213).

The persistent ExactCheckpointRows.rows dictionary caches cloned **weight rows by global row ID**, not request history (loader:43-52). Its dummy weight is not used by replacement forward. Timing hooks (126-137) record time/count only.

Each generate call gets new inputs and a new processor, with no past_key_values supplied (runner:262-287). Default cache_implementation=None creates a **new DynamicCache** (H/generation/utils.py:2206-2226,2245-2276,2316-2318); it does not reuse the static-cache path. Qwen separates GatedDeltaNet convolution state 0, PLE convolution state 1, and n-gram state 2 (H/models/qwen4_exp/configuration_qwen4_exp.py:169-171; modeling_qwen4_exp.py:514-517,1127-1150,1214-1217). Cache state initialization is independent (H/cache_utils.py:1042-1053), and n-gram forward clones previous context before updating its cache. No collision or update-before-read corruption was found. CPU decode does not switch expert implementations (H/generation/utils.py:2493-2515).

These checks reject specific loader/cross-case-cache hypotheses. They are not a new full-model numerical parity result. Cached recurrence versus full prefill at the exact failing prefix remains a useful runtime discriminator.

### RNG comparison limits

The runner resets the seed per request immediately before inputs/processor/generation (265), after model construction. Earlier samples do not simply consume the later case's random stream. No model-forward randomness explaining order-dependent contamination was found on the inspected evaluation path.

HF uses the global CPU generator through torch.multinomial (H/generation/utils.py:3265-3269). Plugin host sampling uses per-request generators and exponential noise; prefetch uses cloned states and verifies live state before reuse (P/src/vllm_tt_plugin/host_sampler.py:39-48,84-119,149-156). This is not a token-identity contract across HF/vLLM versions, implementations, logits, or batching. It also does **not** prove that these particular implementations must disagree under identical distributions and compatible states. No shared-logit comparison was performed.

Native seed 3 and served response/seed 0 are different observations, not a paired failing request. A native seed-0 pass does not guarantee a served seed-0 pass. Changing seeds would hide this distinction rather than diagnose it.

### Parser classification

Installed qwen3 uses the engine adapter, not the older BasicThinkingReasoningParser: V/reasoning/__init__.py:143-145 -> qwen3_engine_reasoning_parser.py:4 -> parser/engine/registered_adapters.py:66-69 -> parser/engine/adapters.py:77-84 -> parser/qwen3.py.

The actual grammar starts in REASONING with default thinking enabled (parser/qwen3.py:102,230-242). End-think switches to CONTENT; tool-call start can implicitly end reasoning (130-143). Neither appears in seed 3. parser/engine/parser_engine.py:509-534 collects reasoning/content events separately and returns None when no content events exist. This stream would legitimately have content=None. A parser cannot recover an answer that was never generated.

The runner rejects duplicate end markers and tool-call output as unestablished parser equivalence (300-305). Neither explains this failure. No evidence supports promoting reasoning to final content or synthesizing a boundary.

## Initial served failure: known and unknown

M/readiness_vllm/sampling_tests.full_initial_failure.log:159,309,368 records **max_batch_size=2**, **reasoning_token_budget=0**:

- Bad-word test: five requests, seeds 0..4, cap 100. First reported failure is response 0 content=None (165-186); this does not establish the other four outcomes.
- Full-capacity mixed test: IDs 0 and 1 (choice/regex). It exposes choice 0 content=None at cap 8 (341-363).
- Slot-reuse mixed test: IDs 0..3 (choice/regex/JSON/plain). It also exposes choice 0 content=None (401-423), not separate JSON/plain/regex failures.

This matrix follows P/tests/tt/test_structured_output_dp1.py:128-163. The four native structured/plain requests cover every distinct prompt in this initial two-slot matrix. Serial native execution does not reproduce concurrent scheduling or slot reuse. Future larger-batch matrices add request IDs and would require additional coverage.

Native choice/regex/plain totals 72/73/51 exceed caps 8/16/16; even their reasoning spans exceed those caps. This supports insufficient allowance as a plausible explanation for structured/plain empty content, especially choice 0. It does not prove the original TT stream's finish reason. Native JSON takes 55, below cap 64.

Native bad-word seeds 0/1/2 total **95, 21, 34**, all within cap 100. Completed seed 4 totals **770**, demonstrating that another native sampled trajectory exceeds cap 100. Seed 3 stops invalidly at 16 despite cap 4096. Thus “served bad-word None is only reasoning truncation” remains **unestablished**. The failing served request is known: **response 0 corresponds to seed 0**, because configs use seed=i and asyncio.gather preserves input order (P/tests/tt/test_host_only_params.py:39-55; tests/tt/utils.py:151-168). Its associated raw stream, boundary, finish reason, and completion usage were not retained in the pytest failure log. Do not infer those missing fields from the assertion or substitute native seed 4's trajectory for served seed 0.

The initial suite has **7 failed, 60 passed, 75 skipped** (530-537). Four other failures concern unseeded variety, top-k 19 variety, repetition sensitivity, and frequency sensitivity. The reasoning-budget option does not alter those workloads (P/README.md:897-903). Chat allowance alone cannot close all Stage9 failures.

## What completed counts justify

P/README.md:897-912 documents **--tt-reasoning-token-budget=N**, an optional additive cap with default 0, for bad-word and mixed structured/plain tests, retaining all assertions. P/tests/tt/conftest.py:43-50,109-114 defines and validates the nonnegative option. It calls for native completions with **normal EOS stopping**, separate reasoning/answer counts, and framing/answer room. Discovery ceiling/runtime hits do not establish a usable limit. Normal EOS alone is not a valid final answer.

The prepared control explicitly also rejects missing reasoning-end or failed answer validation (M/doc/vllm_integration/hf_reasoning_budget_preparation.md:54-63). Implementation matches:

- Per-case validation requires EOS, exactly one reasoning-end, no tool marker, and the answer contract (runner:293-308).
- Loop completion sets complete=true, then budget_established=all(row.established) (344-345). Execution completion is not success.
- Only all-case success emits suggested_extra_allowance and exits 0 (346-352). The final manifest actually records budget_established=false and no suggestion; the code therefore takes exit 1. This conclusion does not rely on a separately captured process exit status.

For **eight successful samples only**, including completed seed 4:

```text
N = max over successful rows of
    max(2 * (reasoning_tokens + framing_tokens), total_tokens - base_cap)

per-row values = 140, 136, 84, 88, 180, 34, 60, 1528
conditional maximum = 1528
```

This would give choice/regex/JSON/plain/bad-word total caps **1536/1544/1592/1544/1628**, fitting the eight assertion-valid completions with the empirical margin. Structured/plain-only arithmetic yields 140. Neither is an upper bound for other trajectories. The earlier seven-success maximum was 180; seed 4's completion shows why that partial-run calculation could not be treated as the final estimate. The value 1528 is our conditional arithmetic, **not an allowance emitted by the failed control**.

The measurements may support a **separate, clearly labeled cap diagnostic** with the existing option, all seeds and assertions intact. This can test the independently demonstrated short structured/plain allowance. It must not be presented as the prepared all-nine control passing, an established global budget, or a repair for early EOS. The preparation's “after a fully established reference” acceptance condition (100-105) is unmet. A later served pass would be evidence for that served run, not a retroactive native pass. No served default, assertion, seed, EOS policy, or thinking policy change is justified here.

## Competing explanations

| Candidate | Direct support or contradiction | Unverified precondition | Adjudication and smallest check/intervention |
|---|---|---|---|
| Ordinary sampled early EOS | Actual EOS at index 15; eligible EOS; sampling on; no grammar; ample cap | Correct terminal logits give EOS selectable mass | Leading interpretation. Capture exact-prefix logits/probability; no behavior change implied. |
| Mask bug | 12 matching singleton bans; no grammar for this case | Live mask differs from inspected code or corrupts other scores | Compare exact masked-index set and raw/processed logits. Fix only a demonstrated discrepancy. |
| Cross-case state/RNG contamination | Fresh request cache and per-case seed reset; PLE memo stores weights | Hidden persistent mutation changes the request | Same seed-3 request alone versus ordered replay, unchanged controls. |
| Loader/PLE mapping bug | Shapes, shards, offsets, keys and canonical mapping agree | Loaded rows/logits differ despite invariants | Check accessed rows or prefix logits if needed; correct hardcoded constants are not a defect. |
| Parser lost an answer | No boundary or final answer in raw IDs; native path never invokes vLLM | Actual served stream differs and contains valid final text | Obtain served raw response; parse saved stream later. No parser fix indicated. |
| Original cap too small | Native choice/regex/plain exceed caps | Actual served failure hits cap before final content | Obtain finish/usage/raw stream; optional cap diagnostic. Cannot explain EOS at 16/4096. |
| Wrong model/cache logits | No concrete source contradiction; terminal logits absent | Numerical defect changes EOS support/probability | Same-prefix cached versus fresh full-prefix logits, then localize consequential divergence. |

## Minimal discriminating follow-ups — not performed

1. **Inspect the exact failing distribution.** Saved prompt length 56 plus first **15** generated IDs gives a **71-token context**, predicting generated index 15 (zero-based). Keep weights, dtype, prompt, masks, top-k 20, top-p 1, temperature 1, and seed 3. Capture raw logits **before in-place processing**, the 12 masked indices, post-mask/top-k scores, finite/NaN status, both EOS ranks/probabilities, end-think rank/probability, and sampling RNG state. Zero postprocessing EOS probability contradicts sampling EOS from that recorded distribution; an unexpected mask establishes a separate processing discrepancy. Positive finite EOS probability supports sampling eligibility but does not alone prove model correctness.
2. **Separate logit checking from draw replay.** Teacher forcing checks the distribution. To replay the recorded draw, reproduce the preceding 15 sampling draws or restore the corresponding captured RNG state. Seeding 3 then teacher-forcing 15 tokens does not consume those draws. A fresh otherwise identical seed-3 run is the smallest order-control experiment; compare full prefixes and terminal pre-draw RNG state, not unpaired “last tokens.”
3. **If needed, compare cache paths on the same history.** Compare fresh cached prefill/decode replay against fresh full-prefix forward, with correct PLE/recurrent initialization. Capture full-vector differences and EOS/top-k membership. BF16 recurrent versus prefill arithmetic need not be bit-identical; a nonzero difference alone is not a bug. A consequential eligibility change calls for layer/state localization. Independently sampled continuations confound cache behavior with changed history.
4. **Classify the actual served None response.** Recover or later record the full response for known index/seed 0, raw IDs/text, reasoning/content, finish/stop reason, and completion usage. Length at cap before the boundary supports cap exhaustion; stop on EOS before the boundary supports incomplete reasoning; raw valid final text lost in parsing supports protocol loss. The current content-only failure log cannot distinguish these. Preserve request correspondence and original assertions.

Instrumentation caveat for check 1: H/generation/utils.py:3240-3250 passes next_token_logits to processors that can mutate it in place **before** storing output_logits. Enabling output_logits=True alone does not necessarily preserve pristine pre-mask logits on this custom path. Capture or clone before mutation. This affects future diagnostic quality, not the current output.

## Other Potential Issues and evidence gaps

- **Native validation is intentionally stronger.** The control requires EOS and nonempty answer; plugin bad-word validation checks content is not None and absence of whole banned words (P/tests/tt/test_host_only_params.py:54-65). Structured tests can accept valid final content at a cap. The native control is conservative budget discovery, not a serving-test substitute. This does not cause the missing boundary.
- **Assertion validity does not establish greeting quality.** Seed 4's “Pineapple!” meets the implemented nonempty/no-banned-word checks, but the checks do not evaluate whether it fulfills “Say hello to me.” Its 763-token reasoning and invented context are model output, not a changed saved prompt. This quality limitation is separate from seed 3's missing reasoning boundary and must not be called a semantic greeting pass.
- **Token mask versus word assertion.** The test deliberately excludes “>” from punctuation stripping for certain BPE-merged forms. No banned ID or prohibited final word appears in the successful artifacts. This general distinction is not evidence for early EOS.
- **Preparation prose is historical.** Its opening “no HF model execution” describes preparation time. The live manifest/logs supersede it.
- **Artifact publication is not atomic.** write() uses Path.write_text, rewriting the manifest per case (runner:40-41,323-326). A concurrent reader can transiently see partial JSON or a manifest lagging a case file. All nine final checked records agree; this did not manufacture seed 3's EOS.
- **Terminal logits/RNG and full served failure responses are absent.** This limits a definitive numerical root cause and served classification. Token IDs do not establish EOS probability, and earlier successes do not prove all model/cache calculations correct.

## Claim review

The draft was checked against executable source paths and both passing and failing observations. Independent source/evidence audits covered loader/cache state, masks/protocol, and budget/matrix coverage. Review rejected: “all controls pass,” “seed 3 exhausted its allowance,” “bad-word masking begins only after reasoning,” “xgrammar caused this EOS,” “EOS was padding,” “earlier cases reused the same DynamicCache,” “same seed implies matching HF/vLLM output,” and “the initial matrix needs IDs beyond 0..3.” The older basic reasoning parser was also rejected as evidence for the active Qwen3 path. Review corrected the known response-0/seed-0 mapping and updated all sizing conclusions after seed 4 completed.

The headline is limited to claims surviving those checks: **an actual native sampled EOS ended an invalid reasoning-only response; no concrete source defect has been established; the all-nine budget gate remains unestablished; successful lengths inform only a qualified cap diagnostic.** Terminal-prefix numerical correctness and the initial served response's termination require the proposed runtime evidence. No hardware result is claimed.

Selected artifact hashes at the stated snapshot:

```text
run_hf_reasoning_budget.py  08950e483195bc2e1612e3da6dff5032c859b84e18759d83d2f7bd123499fabb
full_model_hf_reference.py  fc2fc3fd0435793da1efaba234495c1e5b98e353be7a37cad5f18e6992dc8bfb
requests.json              0a41c226c5a1b947bf56ea49bd182ccc2e2205ca6c28d96e90b17d6dc489b32f
manifest.json              7f15b369bd81852f63a2034cdc0d070f8777b60e93c0e8cebcabc84b8ea377f8
bad_words_3.json           f3e49ae9db954a9db92f3685b96f5aee221c02f8a3613137e0cee6fa197e0b2d
bad_words_3.tokens.jsonl    8e48c98db67ddb50e23468a25475e55e947fc737ff0dc61506664e36478958ab
bad_words_4.json           a473a80f5db09ed226d8de0fba700740165f2efa3d48b9805b3d6d16d630c1f3
bad_words_4.tokens.jsonl    1f61c6dca97faec434217b1d3340cb4fd6c69720a4b22de5c8f53376437b1a69
```
