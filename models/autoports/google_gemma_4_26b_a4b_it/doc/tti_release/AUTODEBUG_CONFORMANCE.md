# AutoDebug: Gemma 4 vLLM parameter conformance failures

## Verdict

The release failure combines two distinct problems:

1. `test_non_uniform_seeding` is, on the available evidence, a **transport/test-timeout failure**, not evidence that seeding is ignored. One of 32 concurrent requests (seed 6) exceeded the fixture's hard 30-second read timeout before the test reached either seed assertion.
2. All four `test_penalties` failures are best classified as **generic test-heuristic flaws**. Each failed baseline/penalized pair produced materially different text, five neighboring penalty cases passed, and the inspected request-builder path forwards all three penalty values into vLLM `SamplingParams`. The failures are caused by proxy metrics that are neither necessary nor consistently directional consequences of a working penalty.

There is a separate, real seed-conformance concern in the checked-out Forge adapter: `tt-media-server/utils/sampling_params_builder.py` explicitly replaces every request seed with `None`. That is not the cause proved by this run—the failing request timed out—but it means a timeout-fixed rerun against this exact implementation is expected not to establish true per-request seed conformance. Confirm the deployed source revision before assigning that defect to the release binary.

## Direct observations

- Release summary: `test_non_uniform_seeding` passed 0/1; `test_penalties` passed 5/9; the other listed parameter groups passed.
- The seed failure is `Request failed for seed 6 ... Read timed out. (read timeout=30)`. No response content was returned and neither the seed-0 determinism assertion nor the nonzero-seed uniqueness assertion ran.
- `test_non_uniform_seeding` launches 32 requests concurrently, each asking for up to 50 tokens. It calls `api_client(payload)` and therefore inherits the fixture default `timeout=30`.
- The penalty test deliberately calls `api_client(..., timeout=None)` for both baseline and penalized requests, so it does not share the seed test's 30-second ceiling.
- Every penalty comparison uses the same prompt, `temperature=0.1`, `max_tokens=1024`, and `seed=1234`; only the tested penalty field differs.
- `tt-media-server/utils/sampling_params_builder.py` reads `presence_penalty`, `frequency_penalty`, and `repetition_penalty` from the request and passes each to vLLM `SamplingParams`.
- The same builder sets `seed = None` unconditionally, with a comment saying the Forge device sampler ignores per-request seeds. Thus request-seed propagation is absent in the inspected checkout even though the API accepts the field.
- The conformance wrapper only launches pytest and reshapes its report. It does not alter request parameters or timeouts.

## Ranked diagnosis by failure

### 1. `test_non_uniform_seeding`: transport timeout (high confidence)

The immediate and sufficient cause is the fixture timeout. A 32-way burst reached a request whose response took longer than 30 seconds. The test failed inside the request helper, before it had a complete result set and before it evaluated seed behavior.

Ranked hypotheses:

1. **Hard client read timeout under queued concurrent service — high confidence.** The exception and fixture source directly establish this. With service serialization or constrained batching, a later request can exceed 30 seconds even though the server remains healthy.
2. **Checked-out Forge backend ignores seeds — high code confidence, but not demonstrated by this failure.** The builder forces `seed=None`. This would cause a semantic conformance failure only after all requests complete. It cannot explain a client read timeout by itself.
3. **Server stall specific to seed 6 — low confidence.** Nothing in the evidence isolates seed 6 as causal; it is simply the request whose client observed the timeout.

Focused isolated experiments:

- Repeat only this test with `api_client(payload, timeout=None)` or a release-level timeout comfortably above worst-case queueing. Preserve 32-way concurrency. This distinguishes transport budget from semantics.
- Record per-request enqueue, first-token, and completion latency plus seed. If all complete, then evaluate the existing seed assertions.
- Before interpreting the semantic result, log the effective `SamplingParams.seed` at the vLLM boundary and record the deployed git revision. On the inspected Forge builder it will be `None` for every request.
- If semantic coverage is desired independently of saturation, first run a small sequential matrix (`seed=0` twice, two distinct nonzero seeds), then separately test concurrent batching. This avoids conflating seed propagation with overload behavior.

Minimal fix boundary:

- Test/transport fix: change only `test_non_uniform_seeding` to use an explicit no-timeout or realistic timeout, as the penalty test already does. Do not globally disable the fixture timeout.
- Backend fix, if the deployed Forge path matches this checkout and seed support is required: stop discarding `request.seed` and make the selected sampling implementation honor per-request seeds. This is separate from the timeout fix and requires runtime validation.

### 2. Frequency penalty, natural-repetition prompt: invalid equal-length heuristic (high confidence)

Failure: baseline and penalized outputs were both 72 whitespace tokens. The texts nevertheless changed materially; for example, `have good vitamins` became `contain important vitamins`. Equal total length does not imply the penalty was ignored.

Ranked hypotheses:

1. **Test heuristic flaw — high confidence.** The assertion requires `test_stats["len"] != base_stats["len"]`, but output length is not a required effect of frequency penalty.
2. **Penalty ignored — low confidence from this case.** Changed deterministic output is evidence that the request affected generation, although it is not by itself mathematical proof that the exact penalty was applied.

Focused experiment: compare token IDs/logprobs at the first divergence and verify that logits for previously occurring tokens receive the expected count-dependent adjustment. A black-box fallback is to sweep `frequency_penalty` over `0, 0.5, 1.2, 2.0` with fixed seed and score repeated-token counts, without requiring lengths to differ.

Minimal fix boundary: remove the unequal-length assertion. Test frequency penalty using tokenized repetition counts or first-divergence logprob deltas across a parameter sweep.

### 3. Presence penalty, natural-repetition prompt: invalid equal-length heuristic (high confidence)

Failure: both outputs were 72 whitespace tokens, but their wording differed. Presence penalty changes logits for tokens that have appeared; it does not promise a different completion length.

Ranked hypotheses:

1. **Test heuristic flaw — high confidence.** Exact length inequality is unrelated to the parameter contract.
2. **Penalty ignored — low confidence from this case.** The changed text and successful presence-penalty semantic-repetition case argue against a blanket omission.

Focused experiment: with fixed prompt/seed, compare the probability or rank of already-seen tokens immediately before the first output divergence for `presence_penalty=0` and `1.2`. Sweep values and assess token-set reuse, not total length.

Minimal fix boundary: remove unequal-length as a mandatory condition and replace it with a metric tied to whether a token has appeared at least once.

### 4. Presence penalty, repeat-trap prompt: invalid diversity-direction heuristic (high confidence)

Failure: whitespace `unique_ratio` fell from `0.243333...` to `0.146718...`, below the required 90% floor. The penalized output was materially different. Presence penalty is a per-token binary logit adjustment; it does not guarantee that whole-completion whitespace unique ratio increases, especially when length, punctuation, and a deliberately repetitive writing style vary.

Ranked hypotheses:

1. **Test heuristic flaw — high confidence.** Lowercased whitespace splitting keeps punctuation attached and conflates length with diversity. More importantly, global unique ratio is not a contract of presence penalty.
2. **Prompt/test instability — medium confidence.** `Write a very repetitive story` explicitly asks the model to oppose the penalty, making coarse completion-level direction checks brittle.
3. **Penalty ignored — low confidence.** One failing proxy does not establish omission, while another presence-penalty case passed and the builder forwards the field.

Focused experiment: use a controlled forced-prefix prompt and inspect next-token logprobs for tokens that are present versus absent in generated history. Alternatively, compare set reuse over equal-length prefixes across a penalty sweep with a tokenizer-aware metric.

Minimal fix boundary: replace the 90% whitespace unique-ratio floor; do not change the adapter based on this result alone.

### 5. Repetition penalty, repeat-trap prompt: invalid raw top-token-count heuristic (high confidence)

Failure: the most common whitespace token appeared 57 times in the penalized output versus 42 in baseline, violating `57 <= 42`. The penalized story is much longer and substantially different, so raw counts are not normalized for exposure. A longer output can contain more instances while having a lower rate. Punctuation-sensitive whitespace tokens also make the selected “top token” unstable between outputs.

Ranked hypotheses:

1. **Test heuristic flaw — high confidence.** Comparing unnormalized maximum counts across unequal-length outputs cannot establish increased repetition.
2. **Prompt/test instability — medium confidence.** The prompt requests repetition, and the model can satisfy it through different phrases or punctuation that defeat whitespace statistics.
3. **Repetition penalty ignored — low confidence.** The repetition-penalty natural and semantic cases both passed, the output changed strongly, and source forwards the value.

Focused experiment: compare repetition rate over equal token counts, preferably using model token IDs; separately measure repeated n-grams and normalize each count by completion length. The decisive white-box check is the sign-dependent vLLM repetition-penalty transform on logits for previously generated token IDs.

Minimal fix boundary: normalize metrics over a fixed-length prefix or assert the expected logit transformation. Do not require an absolute most-common-token count to fall across outputs of different lengths.

## Source/version boundary

The checked-out generic Forge runner builds vLLM `SamplingParams` with all penalty fields, so there is no source-only basis for claiming that the autoport adapter drops penalties. Conversely, the same builder intentionally drops seeds. The release traceback references environments under `/home/mvasiljevic/tt-inference-server`, while the inspected branch is `/home/mvasiljevic/tti-gemma4-stage11`; therefore exact deployed-revision identity must be established before turning the seed code discrepancy into a release-binary claim.

The observed changed penalty outputs are consistent with the penalties being applied, but they do not alone prove it: if seeds are dropped, ordinary stochastic variation can also change outputs. That is why the recommended decisive checks inspect effective `SamplingParams` and first-divergence logits rather than relying on prose-level output differences.

## Recommended disposition

- Classify the recorded non-uniform-seeding result as **inconclusive due to transport timeout**. Fix its request timeout, then rerun semantic checks.
- Track the inspected Forge builder's forced `seed=None` as a **separate likely backend conformance defect**, gated on deployed-version confirmation.
- Classify all four recorded penalty failures as **generic heuristic false failures**. Repair the assertions and retain adapter/backend changes only if an isolated effective-parameter or logit-level experiment proves omission.
- No hardware or server was run, and no production source was edited during this investigation.

## AutoDebug provenance

The mandated fresh-context AutoDebug runner was invoked first. Its independent Codex session and explorer were both blocked before shell startup because `bubblewrap` was unavailable, and even its attempt to write `AUTODEBUG.md` failed for the same sandbox-launch reason. The findings above were therefore completed as the skill's local fallback and checked directly against the supplied release log, release report, conformance test, fixture, wrapper, runner, and sampling-parameter builder.
