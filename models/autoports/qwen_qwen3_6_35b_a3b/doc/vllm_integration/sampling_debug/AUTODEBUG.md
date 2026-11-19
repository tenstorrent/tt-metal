# AutoDebug: Stage 9 chat sampling failures

## Findings and disposition

**The three recorded failures are final-chat-content assertions under an open-thinking template and a total generation cap. They do not demonstrate broken bad-word masking or broken structured-output masking.** Native HF and TT produce identical short greedy reasoning prefixes for the captured plain16 and choice8 prompts; neither reaches the reasoning terminator. The upstream structured-output contract deliberately permits reasoning before applying the final-answer grammar.

**The extended bad-word quality concern remains open.** The fixed blind review records HF 4/10 versus TT 7/10 secondary failures, exceeding its predeclared 20-percentage-point margin. Short-budget HF reproduction does not waive this result. There is also a material evidence qualification: the blind review scores reconstructed token text, and one alleged final-channel control-token leak is absent from the actual API `message.content`. Preserve the original review and verdict; do not silently turn this qualification into a passing score.

**A supported serving-policy candidate uses the existing upstream flag** `--default-chat-template-kwargs '{"enable_thinking": false}'`, if this deployment is intended to give direct answers to requests that omit a reasoning preference. The inspected vLLM merges request choices over that default and propagates the effective choice to rendering, response parsing, and the engine reasoning gate. This is a legitimate supported policy, not a proven repair of TT reasoning quality. It cannot establish readiness without matched quality controls and the original concurrent tests under the declared policy, plus retained explicit-thinking coverage.

No adapter, sampler, model-math, or kernel source change is justified by the inspected failure evidence. No source was changed, no hardware/model/HTTP workload was run, and the live server was not disturbed. This report does not mark Stage 9 complete.

## Scope, source identity, and evidence boundaries

Abbreviations below:

- `R = /home/ttuser/qwen36-35b-a3b`
- `M = R/tt-metal/models/autoports/qwen_qwen3_6_35b_a3b`
- `P = R/vllm-tt-plugin`
- `V = R/tt-metal/python_env/lib/python3.12/site-packages/vllm`
- `H = R/hf/hub/models--Qwen--Qwen3.6-35B-A3B/snapshots/345d467b117d1fd5eda9b879a5988de9ae5f47a7`
- `C = M/readiness_vllm/sampling_controls`

Inspected tt-metal HEAD is `47c0bbd606a691e6111b46c5ee1eb9e01fd423d8`; plugin HEAD is `5d1785b76d17a8989c9495f6939abea6cef62f09`, with a clean plugin worktree at inspection. The model adapter and local launch/control scripts are untracked model work, so these findings describe the worktree, not just committed HEAD. Installed upstream vLLM is **0.31.0** (`V/_version.py:23`), matching the live server's recorded engine version (`M/readiness_vllm/final_server/server.log:37`). Do not substitute the older plugin revision or earlier sentinel exception described in `M/doc/vllm_integration/AUTODEBUG_sampling.md` for this investigation.

The original six paired control artifacts were complete before this inspection. A parallel evidence audit froze those at 2026-11-16 17:32:07 UTC; their mtimes were no later than 17:21:36. The focused 1024-allowance retry was still incomplete when read: its log showed bad-words failed and the first structured test starting, without final traceback or structured verdict (`M/readiness_vllm/sampling_targeted_1024.log:12-13`). Later-running files are not inferred to have passed. The subsequently completed native thinking-off control is discussed separately below.

## Direct observations

1. The completed canonical run reports **65 passed, 3 failed, 74 skipped** (`M/readiness_vllm/sampling_initial.log:306-310`; `sampling_initial_execution.json`). Bad-words fails at `text is not None`, before its forbidden-word check (`sampling_initial.log:155-188`). Both mixed DP1 tests fail on choice request 0's `content in CHOICES`, with `content=None` (`:223-246`, `:282-305`). These are successful response objects with missing final content, not recorded model exceptions.
2. The execution artifact records device and host decode sampling steps and no runtime-error or out-of-vocabulary lines in the captured interval. That corroborates running sampling paths; it is not a universal device-health or correctness proof.
3. The saved short plain and choice HTTP controls finish with `finish_reason="length"`, all generated tokens in reasoning, and `message.content=null`. The corresponding native HF token prefixes match exactly. Ten greedy seeds repeat one deterministic prompt outcome per case; they are not ten independent quality samples.
4. In extended bad-word controls, HF seed 9 has no reasoning end within 1024 generated tokens; TT seeds 1, 5, and 6 have none and finish for length. All twelve recorded banned single-token IDs occur zero times anywhere in either complete generated-token corpus, including reasoning.
5. The 60 blind items, key entries, and review scores join exactly by ID. All blind full texts match pinned-tokenizer decoding of the recorded generated token IDs after removal of the end-of-turn marker and edge whitespace. Primary and secondary counts recompute to the saved summary.

| Captured case | Native HF empty final | TT HTTP empty final | Native HF secondary quality failures | TT secondary quality failures |
|---|---:|---:|---:|---:|
| Plain, 16 total tokens | 10/10 | 10/10 | 0/10 | 0/10 |
| Choice, 8 total tokens | 10/10 | 10/10 | 0/10 | 0/10 |
| Bad words, 1024 total tokens | 1/10 | 3/10 | 4/10 | 7/10 |

The last two columns are the **fixed reconstructed-text blind review**, not a fresh assessment of literal served content. See `C/blind_review.json`, `blind_key.json`, and `unblinded_summary.json`. The rubric treats coherent short reasoning truncation as a primary availability failure without assigning it a secondary quality failure.

## Exact test contracts and effective budgets

| Test or sender | Requests and relevant parameters | Required final answer |
|---|---|---|
| `test_bad_words` | Five concurrent chats, seeds 0-4, temperature 1, top-p 1; ban hello/Hello/hi/Hi/hey/Hey; total cap `100 + allowance` | Content must be non-`None`; then six exact word spellings must be absent |
| Mixed choice | Request IDs divisible by 4; temperature 0, presence penalty 0.5; total cap `8 + allowance` | Exactly red, green, blue, or yellow |
| Mixed regex | IDs congruent to 1; temperature 0; total cap `16 + allowance` | Full match of `LANE-[0-9]` |
| Mixed JSON | IDs congruent to 2; temperature 0; total cap `64 + allowance` | JSON object with status ok and integer lane 0-31 |
| Mixed plain | IDs congruent to 3; temperature 0; total cap `16 + allowance` | Truthy final content |

Source: `P/tests/tt/test_host_only_params.py:34-65`; `test_structured_output_dp1.py:22-110`. For observed capacity 32, the first mixed test submits 32 requests and the slot-reuse test submits 64 (`:113-163`). All four sender types are part of the test contract; a failure propagated from one choice sender does not establish whether the regex, JSON, plain, or remaining choice senders succeeded. `asyncio.gather` propagates the sender assertion before the aggregate result check.

None of these failing requests supplies `chat_template_kwargs` or a reasoning preference. `P/tests/tt/utils.py:118-133` returns just `message.content` for bad-words, dropping reasoning, finish reason, and token usage from the assertion context. Most passing seed/logprob/penalty/host-parameter cases use the legacy completions endpoint, which returns generated text directly (`utils.py:55-93,163`). Thus 65 passes can coexist with these chat-final-content failures.

`--tt-reasoning-token-budget` defaults to zero and is explicitly an **extra total generation allowance** (`P/tests/tt/conftest.py:42-51,108-114`). It does not force reasoning to end or reserve answer tokens. With value 1024 the actual caps are **1124 bad-words, 1032 choice, 1040 regex/plain, and 1088 JSON**. The paired extended bad-words artifact instead uses **1024 total**, not 1124. Its failure seeds must not be presented as the retry's exact responses.

The canonical clients inherit OpenAI defaults of 600-second timeout and two retries (`P/tests/tt/conftest.py:144-154`; installed `openai/_constants.py:7-8`). Retries do not apply to an HTTP success whose content later fails a pytest assertion. They can affect traffic in long concurrent runs, unlike the sequential collector's 1800-second requests, so a future concurrency comparison should record or disable retries deliberately.

## Causal chain: template to missing final content

### 1. The pinned template starts an unfinished thinking span

`H/chat_template.jinja:151-158` renders:

```text
omitted or enable_thinking=true: <|im_start|>assistant\n<think>\n
enable_thinking=false:          <|im_start|>assistant\n<think>\n\n</think>\n\n
```

This is documented model behavior, including a supported nonthinking API example (`H/README.md:663-671,787-834`). The launch script selects `--reasoning-parser qwen3` and supplies no default-template kwargs (`M/tests/run_vllm_server.sh:21-26`). The recorded server pins model and tokenizer to the same HF revision and uses `enable_in_reasoning=False` (`M/readiness_vllm/final_server/server.log:14,37`). Checkpoint generation defaults include top-k 20 (`H/generation_config.json`); bad-words explicitly supplies top-p 1 through the test helper, so the controls do not accidentally use checkpoint top-p 0.95.

### 2. The parser correctly separates reasoning from final content

`V/parser/qwen3.py:100-104,224-242` defaults thinking to true and starts in `REASONING`. `</think>` transitions to `CONTENT` (`:130-132`). The parser collects reasoning events separately and produces `None` when no content events occurred (`V/parser/engine/parser_engine.py:509-534`; full response parsing `:746-771`). `V/entrypoints/openai/chat_completion/serving.py:995-1029` places those separate results into the response.

The generated-token budget includes reasoning; it is not a final-content budget. `serving.py:320-340` passes the requested output cap into sampling. Truncating a coherent reasoning prefix before `</think>` therefore produces a valid length-limited response with `reasoning` and no `content`. Copying reasoning into content or stripping the parser would change this API contract rather than repair it.

### 3. Structured grammar intentionally waits for reasoning end

`V/v1/structured_output/__init__.py:235-258` defers constraints while the reasoning gate is false; `:202-213` fills the mask with all-allowed bits in that state. Once the reasoning boundary is consumed, `:456-465` latches the ended state and advances the final-answer grammar. Thus `structured_outputs.choice` does not force the initial thinking tokens to be a color.

The captured native choice control is deliberately valid only before that transition: `M/tests/run_vllm_reasoning_budget.py:101-104` aborts if HF emits `</think>`, because it does not implement the native choice grammar. It proves short pre-grammar truncation, not long-budget grammar correctness. The shared tests also include regex/JSON and mixed state reuse, which these single-prompt controls do not cover.

### 4. Bad-word constraints can lengthen or distort reasoning on either backend

`V/sampling_params.py:353-357,739-773` specifies forbidden token sequences and constructs both unprefixed and space-prefixed encodings. This is a token-sequence contract, not every lexical spelling inside every larger BPE token. It applies to generated reasoning as well as final text. For example HTTP seed 3's `Say_hello` contains distinct token 93383 (`_hello`), not a recorded banned singleton; it does not show that masking was bypassed.

Native first `</think>` indices, zero-based, are `[40,261,186,164,687,118,675,398,477,None]`. Only seed 0 has final text in its first 100 generated tokens. Therefore the reported native **9/10 absence at 100** is a prefix measurement from the extended trajectories, not a separate completed 100-token experiment. For actual test seeds 0-4, four of five cannot finish reasoning within 100. Native seed 0 later degenerates despite its early boundary.

TT indices are `[349,None,308,319,350,None,None,133,268,167]`; seed 2 emits a second close token at index 330. None of these TT trajectories has final text at 100. A larger allowance addresses a finite cap, but neither a 1024 nor a 1124 cap guarantees termination or good content. The observed concurrent retry failure is compatible with this mechanism, although its exact pending traceback is unavailable.

## Adapter and host-sampling adjudication

The following source checks challenge alternative explanations instead of assuming the template is the only possible cause:

| Candidate | Inspected contract and disposition |
|---|---|
| Bad-words dropped by adapter/device sampler | `P/src/vllm_tt_plugin/model_runner.py:4046-4054` forces bad-words to host sampling. `:4413-4441` supplies the per-rank bans and output history to the host sampler. `V/v1/sample/sampler.py:395-397` applies the processor. Zero banned IDs in both captured corpora supports enforcement for these singleton bans. Multi-token suffix bans remain untested by this case. |
| Grammar switches to a faulty device path at reasoning end | Structured parameters are detected even before a restrictive bitmask exists (`P/src/vllm_tt_plugin/structured_output.py:17-53`). Qwen declares `supports_device_grammar=False` (`M/tt/serving_capabilities.py:37`); `model_runner.py:4057-4060` keeps the request on host. The end marker changes the mask, not the host/device route. |
| Adapter accidentally samples host-mode logits | `M/tt/generator_vllm.py` preserves `sampling_params=None`; `M/tt/serving.py:489-497,550-556` selects unsampled full logits in that mode. The plugin owns penalties, grammar, bans, sampling, and stopping. No contrary forwarding was found. |
| Stale resident feedback during host sampling | Host steps force input reload (`P/src/vllm_tt_plugin/async_decode.py:464-474`); structured/host steps are excluded from resident overlap (`:610-615,674-675`). This does not prove all mixed-request state correctness, but the proposed stale resident-token mechanism is not shown here. |
| Host inspects unfinished accelerator work or wrong logit width | Decode enqueues trace then snapshot copy on CQ0 (`M/tt/serving.py:379-390`). Output-pool `read/process` does blocking read or records and synchronizes readback events before conversion, then returns independent `[submitted_rows,1,real_vocab]` storage (`M/tt/serving_state.py:490-534`). The plugin validates width and consumes the last time row (`model_runner.py:4326-4332`). No concrete missing fence or geometry contradiction was found. |
| Equal integer seeds require identical HF and TT samples | False. The control uses HF `generate` with `torch.manual_seed`; installed Transformers samples via `torch.multinomial` (`transformers/generation/utils.py:2835-2841`). The plugin host path uses upstream exponential-noise selection (`P/src/vllm_tt_plugin/host_sampler.py:106-120`; `V/v1/sample/ops/topk_topp_sampler.py:535-566`). Free-running stochastic divergence does not localize a numerical or RNG defect. |

These checks do not support the specific proposed failure mechanisms; they do not exclude every possible adapter/model issue or runtime race. No lower-level kernel hypothesis has the required sequence-matched evidence to headline it. The host-logit route still executes the TT model; using a host sampler is not an HF model fallback.

## Blind-review discrepancy and unresolved quality

`C/blind_key.json:32-36` maps `item_60931890` to HTTP bad-words seed 2. `C/blind_review.json:196-215` marks it as a quality failure solely for literal `</think>` contamination in reconstructed final text; it explicitly says the two greeting copies are not a sustained loop. The actual `message.content` at `C/http_bad_words_1024.json:1633` has the two copies but **no literal close marker**. The generated token arrays really contain close markers at 308 and 330, and `C/blind_outputs.json:82` retains the latter after splitting at the first marker. Upstream explicitly absorbs duplicate close markers in content (`V/parser/qwen3.py:134-139`). This explains the discrepancy without inventing lost tokens or an API parser leak.

Consequences:

- Keep the fixed 7/10 versus 4/10 reconstructed-text scores and their failed 30-point comparison. The report's original assertion of seven **served-content** quality failures is too broad.
- Excluding that single unsupported served-token-leak claim would leave 6/10 versus 4/10, exactly 20 points. This is a sensitivity calculation, **not a replacement verdict** or permission to change the preset rule after observing results. Assess actual served responses separately with backend labels hidden and the rubric fixed beforehand.
- Other TT failures remain substantive: sustained reasoning loops (seeds 1 and 6), unrelated drafting (5), farewells instead of greeting (4 and 8), and prompt echo (7). Native HF also has loops/gibberish (0,9) and nonresponsive final answers (5,7). Neither backend is clean under this constraint.
- There is no identified cross-request content or language drift in the fixed review. Invented constraints or a mangled paraphrase of the prompt alone do not prove contamination from another request.

Pairs match recorded model/revision, messages, rendered prompt tokens, banned IDs, caps, temperature/top-p/top-k, and seeds. All 30 HTTP responses capture `prompt_token_ids`, and each exactly matches its collector-side `prompt_tokens` and reported prompt usage: 14 tokens for bad-words, 18 for choice, and 20 for plain (for example `C/http_choice_8.json:112-131`). The native model's resolved dtype/config were not fully serialized. The controls are sequential; canonical bad-words is five concurrent requests and mixed tests are 32/64. Treat the finite-sample quality difference as open evidence requiring follow-up, not as localization to quantization, MoE routing, recurrence, cache corruption, or sampling.

## Supported policy candidate and minimum repair boundary

The existing upstream CLI describes exactly this policy (`V/entrypoints/launchers/cli_args.py:101-106`):

```text
--default-chat-template-kwargs '{"enable_thinking": false}'
```

The minimum repair boundary supported by this investigation is to align the declared chat mode with the tests' final-answer budgets, then validate the remaining quality and concurrency behavior. Source evidence does not select a deployment-wide default change over per-request test configuration; that depends on the intended serving contract.

If a direct-answer default is the intended product contract, place that argument in the **model-owned launch configuration**, alongside the existing reasoning parser. This does not need a new plugin capability, environment variable, `--additional-config` key, model-name gate, tokenizer edit, or custom parser. The readiness wrapper parses its extra argument string with `shlex.split` (installed `tt-model-bringup/0.1.24/runtime/readiness_check/run_vllm_server.py:1167`), so retain the JSON as one quoted argument when incorporating it into `--additional-server-args`.

The exact override chain in inspected vLLM is:

1. `ChatCompletionRequest.build_chat_params` preserves explicit `chat_template_kwargs.enable_thinking`. If that key is absent, a supplied `reasoning_effort` produces `enable_thinking = effort != "none"` (`V/entrypoints/openai/chat_completion/protocol.py:568-594`).
2. `ChatParams.with_defaults` merges request kwargs **over** server defaults; `False` survives the merge (`V/renderers/params.py:36-48,104-123`).
3. The renderer receives these defaults (`V/renderers/online_renderer.py:250-258`). Chat serving builds its parser from the same effective kwargs (`V/entrypoints/openai/chat_completion/serving.py:191-214,281-285`), initializes the reasoning-ended state from the actual prompt, and forwards kwargs to the engine (`:366-394`).

| Request | Effective thinking with proposed default |
|---|---|
| Omitted preference or empty kwargs | false |
| `chat_template_kwargs={"enable_thinking": true}` | true |
| `chat_template_kwargs={"enable_thinking": false}` | false |
| `reasoning_effort="high"`, no explicit thinking key | true |
| `reasoning_effort="none"`, no explicit thinking key | false |
| Explicit thinking false plus effort high | false; explicit template choice wins |

Local cheap checks confirmed this table by extracting the installed pure merge/request methods with AST and rendering the pinned Jinja template. No vLLM engine or model was imported. The generation annotation used a no-op Jinja extension, and unused media arguments were constrained to `None`; this validates default resolution and prompt suffixes only, not inference or complete request processing.

This policy is legitimate **only when disclosed as a changed default** and validated as such. It leaves explicit reasoning available, preserves bad-word/grammar requirements, and uses the checkpoint's documented nonthinking mode. If default-thinking behavior is part of the required deployment contract, retain it and make the generic tests' own short-answer mode explicit instead; changing the serving default would not repair that contract.

It would mask a regression to declare the current quality gap solved because nonthinking requests now produce any non-`None` content, to remove explicit-thinking coverage, to skip the three tests, or to combine old thinking-on and new thinking-off results as one passing run. `test_bad_words` even permits `""` through its non-`None` check and compares six spellings despite a case-insensitive comment (`P/tests/tt/test_host_only_params.py:54-65`). It is a constraint check, not a quality gate.

The completed native `C/hf_bad_words_off_100.json` is a useful caution: the closed-thinking prefix does not prevent the model from generating a fresh `<think>` (seeds 1 and 9) or confused narration (seed 3). Its `status="PASS"` means the collector completed, not that quality passed (`M/tests/run_vllm_reasoning_budget.py:141-142`). The template is a generation cue, not a hard token ban. Nonthinking parsing starts in content; it does not guarantee that every produced token is a good direct answer. No matched TT thinking-off pass is asserted here.

## Required follow-up before a readiness claim

These are proposed validations for the caller, not workloads executed by this investigation:

1. Preserve original artifacts, blind scores, configuration, and failed gate. Keep actual API content distinct from decoded generated-token text, with finish reason, reasoning counts, and request identity in failure diagnostics.
2. Under the declared nonthinking-default candidate, run matched native HF and TT bad-word controls at the original cap and a predeclared extended cap. Blindly assess **actual served TT final content** against native output processed with the same pinned parser, and assess full generated-token output separately; retain the existing margin and quality categories. Do not equate collection status with quality.
3. Verify omitted preference, explicit true/false, and reasoning-effort overrides in both full and streaming chat responses. Explicit-thinking tests need measured budgets for their actual prompts and must check correct reasoning/final separation. Do not expect every arbitrarily small thinking cap to yield final content.
4. Exercise choice, regex, JSON, and plain singly and then through the original 32-request mixed and 64-request slot-reuse tests. Verify every response and both immediate grammar enforcement in nonthinking mode and post-reasoning grammar activation in thinking mode. The short native choice control supplies no evidence for either transition.
5. Re-run the complete canonical sampling profile under one recorded server policy, preserving the successful actual sampling/logprob/seed tests. Add targeted token-sequence checks for real multi-token bad-words if claiming that broader feature; these twelve singleton bans do not test suffix history across steps or slot reuse.
6. If quality remains worse, compare HF and TT logits on identical teacher-forced prefixes at the first divergence, with resolved dtype and generation configuration recorded. The paired bad-word first-divergence indices are `[26,69,45,16,16,45,40,57,0,26]`. Control randomness before blaming model arithmetic; escalate to module/kernel comparison only with a localized numerical discrepancy.

## Other Potential Issues and claim review

- **Long-budget concurrent performance/retry behavior:** the canonical client timeout can cause extra requests during slow 32/64-request runs. No transport-retry record was established for the initial content assertions, so this is not their cause. Preserve exact retry and request identity evidence in follow-up runs.
- **Broader model quality:** the extended failure distribution is real for the fixed review, but unlocalized. No source-level math defect or race was proved in this inspection. The earlier sentinel and continuation-slot findings belong to separate revisions/sequences; they are not promoted as causes of these successful-but-contentless responses.
- **Test diagnostics and coverage:** helpers discard the metadata needed to distinguish reasoning truncation from no generated output; `asyncio.gather` failures do not report the entire mixed matrix. These are test gaps, not proof that the other request types passed.

After drafting, headline claims were checked against the actual failing assertion sites, pinned template, upstream merge order, both parser and structured gate, host routing before/after reasoning end, output snapshot/readback completion boundary, and raw blind/API correspondence. The short-budget diagnosis predicts the observed HF/TT identical prefixes and length-limited null content without requiring a hidden sampler defect. It does **not** explain away the extended TT quality excess or establish mixed-request correctness. Those remain work; the proposed default-policy intervention is supported by source contracts but has not been runtime-validated by this investigator.
