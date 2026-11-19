# AutoDebug: exact-length random prompt transport

## Leading finding

**The vLLM 0.26 random text preparation path does not guarantee an exact 4096-token completion input. A model-owned, process-local sample adapter can preserve the upstream benchmark runner and request function while sending the original sampled integer IDs directly.** No shared vLLM source or serving/model code needs to change for that proposed boundary.

The source proves two missing guarantees: random text normalization can exhaust its retries with a nonzero length mismatch, and server alignment uses only the first prompt to decide whether to check the remaining prompts. Neither establishes a final exact-length invariant. Matching tokenizer revisions does not make arbitrary token sequences losslessly decode/re-encode.

**Causality limit:** the artifacts establish incorrect final benchmark input lengths, but do not contain the corresponding original IDs, local per-request lengths, wire prompts, or raw usage streams. They do not prove that retry exhaustion specifically produced the observed 4658. The precise division between local normalization mismatch and subsequent client/server text-tokenization differences remains unresolved. The proposed ID boundary bypasses both possibilities.

Inspection only, 2026-11-14. No benchmark, tokenizer, vLLM target, model, or hardware execution; no HTTP requests, signals, process inspection, or server changes. Only this report and the requested root `AUTODEBUG.md` were written. Local checks read source and recomputed supplied JSON counts with the Python standard library.

## Evidence and source identity

All paths below refer to inspected local files, not inferred behavior from a different release.

| Label | Path |
|---|---|
| `D` | `/home/ttuser/ornith-9b/benchmark-clients/vllm-v0.26.0/vllm/benchmarks/datasets/datasets.py` |
| `U` | Same client, `vllm/benchmarks/datasets/utils.py` |
| `E` | Same client, `vllm/benchmarks/lib/endpoint_request_func.py` |
| `B` | Same client, `vllm/benchmarks/serve.py` |
| `S` | `/home/ttuser/ornith-9b/tt-metal/python_env/lib/python3.12/site-packages/vllm/` |
| `A` | `/home/ttuser/ornith-9b/tt-metal/models/autoports/ornith_ai_ornith_1_5_9b/doc/benchmark/` |

Client HEAD is `568afb3a13806beb53bb2e6bd518269357b237c0`; the four client files above have no diff from HEAD. tt-metal HEAD is `391f81a143ca327bdb9a576d4bc090a843445683`. `A/identity.json:5-13,58` records model/tokenizer revision `489cb97981b8654bcfcf30ce1f94ed1b62e07b53` and server distribution `0.31.0+empty`. Server conclusions below use the installed `S` files, including any local changes, rather than assuming an untouched upstream tag.

Direct observations, separated from interpretation:

| Observation | Supported conclusion |
|---|---|
| `A/transport_failure.json` records 32 completed, zero failed, input lengths 4096..4658, and 128 output tokens for every request. | Requests succeeded but the recorded workload was not fixed at 4096 input tokens. |
| Recomputed input sum is 137589; ten requests equal 4096 and 22 exceed it. Exact 32-request total would be 131072. | The artifact contains 6517 extra input tokens, about 4.97% above the intended aggregate. |
| `A/pilot_performance/perf-b32-warmup.log:7` reports sampling `[4096,4096]`; its argument record at line 6 has prefix 0, range ratio 0, batch size 1, random dataset, and `/v1/completions`. | Initial length sampling is fixed, not a configured random-length range. With `U:70-73`, this also implies zero subtracted special tokens in that recorded run. |
| The warmup JSON input-length array and aggregate equal the focused artifact. The log reports the same total at line 21. | The focused artifact agrees with the saved benchmark result. |
| No aggregate mismatch warning appears between sampling and the benchmark in that warmup log. | Retry exhaustion cannot be asserted as the observed cause. Also, absence of that warning cannot prove every local mismatch was zero: `D:650,681-690` checks their sum. |
| `B:1204` saves `output.prompt_len`; `E:194,244-247` initializes that from local metadata, then replaces it if a usage-only SSE chunk supplies `prompt_tokens`. | Saved lengths are final request-function counts, normally server usage here, but the JSON alone does not establish whether each count came from usage or fallback metadata. |

The separate 96-request pilot has lengths 4096..4741 and 412665 input tokens. It corroborates the missing fixed-length contract; this investigation makes no per-index correspondence between the two runs.

## Trace the first loss of the exact-length contract

1. **Length and content sampling.** `U:70-100` subtracts `tokenizer.num_special_tokens_to_add()` from the input-length budget only, then draws input lengths, output lengths, and offsets. `D:620-632` obtains those arrays, then builds `allowed_tokens` from vocabulary IDs excluding `all_special_ids`. Do not replace that array with a sorted array if preserving this client's sequence matters.
2. **Original token sequence.** `D:753-756` constructs the body exactly as:

   ```python
   inner = allowed_tokens[
       (offset + index + np.arange(input_len)) % len(allowed_tokens)
   ].tolist()
   raw = prefix_token_ids + inner
   ```

   The actual modulus is the number of allowed tokens, not `vocab_size`. For the recorded zero-prefix, zero-special, zero-range case, this original body has exactly 4096 allowed IDs.
3. **Representation changes to text.** `D:759-767` calls `gen_prompt_decode_to_target_len(..., target_token_len=prefix_len+input_len, add_special_tokens=False)`. This can change token identities, even when the final count matches. Its returned adjusted IDs are no longer necessarily the original allowed sequence.
4. **The retry budget is not a bound.** `D:518-547` decodes then re-encodes before checking exhaustion. With default `max_retry=10`, it can perform eleven decode/encode passes. On the last pass, `remain_num_try <= 0` exits at lines 523-526 **before** the truncation branch at lines 544-545. If that last encoding has 4658 tokens for target 4096, it returns the text, all 4658 adjusted IDs, and mismatch 562. This is a conditional code trace, not a reproduction or proof that this particular request took that branch.
5. **Metadata accepts the mismatch.** `D:769-770,657-664` sets `SampleRequest.prompt_len` to the actual adjusted length, not a hard requested length, and still appends the request. A warning does not reject it.
6. **Server alignment has no all-request invariant.** `B:2040-2045` invokes `_align_prompts_to_server_tokenizer`. At `B:117-124`, if the first server-tokenized prompt equals its recorded local `prompt_len`, the whole list returns unchanged. That equality can hold even if the first local length itself is not the configured target. If alignment does run, it targets each request's potentially already-inflated `prompt_len`, leaves short prompts alone, truncates oversized IDs, detokenizes back to text, and performs no final token-count validation (`B:132-146`). A first saved length of 4096 does not prove the alignment fast path ran; its source condition compares an earlier `/tokenize` result with local metadata.
7. **Completion payload contains text and no requested input limit.** `B:1004-1026` forwards the prompt and length separately. `E:175-188` inserts the prompt directly into JSON. `prompt_len` is not a server truncation field. `max_tokens` controls output length. No input truncation is added by this request function.

Even if the exact same tokenizer implementation/settings were used at both ends, step 4 permits local overlength requests. Conversely, if every returned local prompt really encodes to 4096 under exactly the server's tokenizer semantics, the server should count 4096: plain "decode/re-encode instability" alone would not explain an additional divergence. Identical revision is weaker than identical implementation/settings. The current artifacts do not establish that stronger condition or identify which requests diverged at which stage.

## Integer-ID transport is supported by both inspected sides

**Client:** `E:179,201` serializes `request_func_input.prompt` directly with `session.post(..., json=payload)`. It neither requires a string at runtime nor re-encodes a list. `SampleRequest` (`D:76-82`) and `RequestFuncInput` (`E:65-69`) are ordinary dataclasses; their annotations omit `list[int]`, but there is no dataclass runtime validation. A model-local typing adapter/cast can contain this static type mismatch.

**Server:**

- `S/entrypoints/openai/completion/protocol.py:50-56` explicitly accepts a flat nonnegative integer list as `prompt`.
- `S/renderers/online_renderer.py:657-708` sends it through `prompt_to_seq`, model prompt parsing, and `render_cmpl_async`. `S/renderers/inputs/preprocess.py:46-54` treats a nonempty integer list as **one prompt**, not a batch of singleton prompts; lines 129-138 convert it to `TokensPrompt(prompt_token_ids=...)`.
- `S/renderers/base.py:650-674` skips text tokenization when `prompt_token_ids` already exists. With normal `echo=False`, `build_tok_params` requests no detokenization (`completion/protocol.py:265-275`). Optional detokenization would add text without re-encoding these IDs.
- Post-tokenization validation applies truncation, padding, and length checking (`S/renderers/params.py:512-541`). With the inspected completion defaults, no prompt truncation or padding is requested. `add_special_tokens=True` is a text-encoding option and does not prepend tokens in this ID path. Explicit `add_special_tokens=False` is still useful to document the wrapper's intended semantics.
- For text-only inputs, `S/renderers/base.py:940-959` builds `tokens_input(prompt_token_ids)` directly. The subsequent expansion-truncation helper returns immediately for non-multimodal inputs (lines 988-992).
- `S/entrypoints/openai/completion/serving.py:308-322,441-488` counts actual prompt IDs and emits final usage with `choices=[]` when `include_usage=True`. That is exactly the usage-only form the older client parses.

Therefore a validated native-Python `list[int]` of length 4096 remains a 4096-ID engine input through the inspected server preparation path under these settings. This is a source-level conclusion, not live-server or TT hardware validation.

## Smallest model-owned intervention

Use a model-owned CLI wrapper in the **existing 0.26 performance-client interpreter**, installing a temporary in-process sample-generation override and then calling the original vLLM CLI. Keep the original `vllm`/`openai` backend key, `/v1/completions`, request function, benchmark loop, and metrics. This is a proposed change only; none was implemented here.

The narrowest reusable placement is a wrapper of the **`serve.get_samples` binding** (`B:44,2040`). In that wrapper, validate the fixed random-text configuration, temporarily replace `RandomDataset.generate_token_sequence`, invoke the saved upstream `get_samples`, restore the method in `finally`, and validate every returned sample. Restore the outer `serve.get_samples` binding in `finally` when the CLI finishes or raises as well. The replacement method should use the original formula above, concatenate the original prefix, slice the IDs to the requested length, and return `(ids, len(ids), 0)` without any decode/encode helper. For this specific configuration, the raw sequence already has length 4096, so the slice leaves its contents unchanged.

Required behavior for this intentionally narrow wrapper:

- Accept only plain random completion requests with prefix 0, range ratio 0, batch size 1, no LoRA/chat/multimodal preparation, and the expected tokenizer pin. Assert `num_special_tokens_to_add()==0` and sampled `input_len==requested_length`. A slice cannot repair a short raw sequence. Supporting other special-token budgets or prefixes would need a separately specified policy; the current upstream prefix helper itself normalizes through text (`D:694-725`).
- Retain upstream seed/RNG initialization, allowed-token array, offsets, request IDs, output lengths, and sampling order. `U:98-100` draws all request lengths and offsets before token normalization. Removing helper retries therefore does not change those arrays in this single-sample-call, zero-prefix path. Do not claim arbitrary later dataset calls preserve RNG state after removing retry draws.
- Use the actual `allowed_tokens` argument and native integers (`.tolist()` for this integer NumPy array), then require `len(ids)==requested_length`, `type(id) is int`, allowed-set membership, and no prohibited IDs for every sample. Do not recover IDs by encoding the old text or use the helper's adjusted token list; those choices lose the requested original allowed sequence.
- Set `prompt_len=len(ids)` and preserve other `SampleRequest` fields. Assert every prompt is an integer list, rather than relying on the first sample. Because the first is non-string, existing alignment returns immediately at `B:81-82`; no alignment patch is necessary.
- Validate effective body overrides. `E:128-135,188` merges `extra_body` after constructing the payload. CLI/per-request body overrides can replace `prompt`, change truncation, streaming, model, or output/EOS settings (`B:62-66,831,1011,2087-2088`). Reject conflicting workload overrides instead of allowing them to invalidate the fixed-input claim.
- Install hooks in the same process that runs the CLI; a subprocess loses them. `vllm/entrypoints/cli/main.py:57-70` preserves the normal benchmark CPU-platform setup. Replacing sample preparation avoids the direct-import alias trap in `entrypoints/cli/benchmark/serve.py:5,21-22` that would affect a late replacement of `serve.main`.

There is **no need to patch the request registry** when the sampler supplies ID lists. If a separate request adapter is chosen instead, patch the selected entry in `serve.ASYNC_REQUEST_FUNCS` before `benchmark()` selects it at `B:800`; rebinding the endpoint module's function name alone leaves the dictionary's stored callable unchanged. Such a late adapter also misses the earlier alignment stage and is less minimal here.

## What preserving upstream metrics means

Sample preparation finishes before measured benchmark timing begins (`B:974`). Unchanged upstream code handles the configured readiness check, warmups, concurrency semaphore, request scheduling, shared HTTP session, and result collection (`B:800-894,968-1043`). The original completions function keeps streaming and `include_usage=True` (`E:183-186`). Conversion is not added inside the timed request function.

The timing definitions remain upstream definitions: first nonempty `choices` chunk sets TTFT, subsequent choice chunks produce ITL, and latency ends at the last choice-chunk timestamp (`E:227-257`). Final usage-only traffic is excluded from that latency. Prompt usage replaces local `prompt_len` at `E:244-247`; aggregate input totals use those returned counts (`B:606`), and TPOT uses `(latency-TTFT)/(output_len-1)` (`B:608-610`). Do not rewrite returned lengths to force a passing report.

Input-ID transport preserves these algorithms, not the old numeric benchmark results. It intentionally changes the workload to the requested exact input length and removes server-side input tokenization work. Old variable-input text measurements are not valid fixed-4096 baselines.

## Other potential issues and limits

- **Retry exhaustion is a proven missing guarantee, not proven attribution of the recorded outliers.** The helper's returned prompt and adjusted IDs come from the same final decode/encode pair (`D:521-522,549`); there is no evidence of stale post-truncation length metadata. It is incorrect to assume the helper always returns exactly 4096 IDs and only its returned text is wrong.
- **Tokenizer pin evidence is incomplete in the old pilot.** The requester stipulates a matching pin. The inspected CLI loader passes tokenizer ID/mode/trust settings without a revision (`B:1971-1977`), and the old pilot uses the bare model ID. This does not prove a different revision was used. The proposed wrapper should record an immutable local tokenizer snapshot or explicitly pinned loader and resolved tokenization dependencies; version mismatch is not the headline diagnosis.
- **Special-token settings differ across the text boundary.** The random helper and `/tokenize` alignment explicitly use `add_special_tokens=False`, while the server completion schema defaults to true (`D:765`, `B:99`, server protocol lines 115-121). The recorded client's zero special-token budget does not demonstrate a server-side insertion, and a fixed BOS explanation alone does not account for input-dependent excess of hundreds of tokens. The integer path avoids this uncertainty.
- **The original allowed-token policy can already be lost in text normalization.** On a short encoding, helper extension samples the full vocabulary rather than `allowed_tokens` (`D:530-543`); re-encoding can also change IDs. This is an additional reason to intercept the original formula, not evidence that a prohibited token caused the measured length excess.
- **Missing usage can hide an accounting problem.** The old client reads usage only when `choices` is empty/absent (`E:244`). Without usable prompt usage, its supplied `prompt_len` survives; without completion usage it falls back to tokenizing output text (`B:589-604`). Exact input lists do not by themselves prove raw server usage was observed. The inspected 0.31 server's trailing usage chunk is compatible, but no raw stream was examined here.
- **Existing tests do not establish this transport contract.** `tests/benchmarks/test_random_dataset.py:17-20,63-136` tests sampling reproducibility with GPT-2, not Ornith exact-length ID transport to a separate server. No tests were executed in this investigation.

## Claim review and follow-up boundary

The first-draft findings were rechecked against helper exit ordering, sample length assignment, first-prompt alignment, JSON serialization, server list parsing, tokenizer bypass, post-tokenization validation, and final usage construction. The leading conclusion survives: changing only model-owned sample preparation can enforce exact IDs while retaining the upstream runner/metrics. The narrower claim that the observed 4658 must be helper exhaustion does **not** survive the evidence check and is explicitly not asserted.

Suggested implementation validation, not performed here: statically/mocked check the original formula, allowed-ID membership, every sample's exact length, and payload type; then have the caller perform a bounded integer-prompt smoke against its owned server and archive actual `usage.prompt_tokens==4096` plus 128 completion tokens. For a later measured 32-request run, require all input lengths 4096 and total input 131072, together with raw-usage evidence so local fallback cannot make the check pass. Repeat the intended supported request-count/seed configurations without changing the upstream metric definitions. No server restart or model change is implied by this transport proposal.

## Source fingerprints

SHA-256 at inspection:

| File | SHA-256 |
|---|---|
| `D` | `f4fed1d7135231d0941bea47a79b6c9901a8d21c3d21ad668a6b662b1c9c3f42` |
| `U` | `14efee4c78ea60b38e1be458c192ff1972c3295dbebe91bacff414ebcf89c3b6` |
| `E` | `e9ab4df19c5d89e1b73f116b4e21f2cc6c7da077266367c4b1b2bd7e015fda59` |
| `B` | `8938b64b273470a3b3fb3b5d8bbd22e8ab4df7640295f55435ac4384283ec589` |
| `A/transport_failure.json` | `7d5fa620fcbabb2b81b30b5f25f3b30df8b10ae7c8c3603a1d3d465730b7d779` |
| `S/entrypoints/openai/completion/protocol.py` | `4616cd6887ef0a23fd8a77468e82f2668e3d0b53075112f2d97f3de960c22c81` |
| `S/entrypoints/openai/completion/serving.py` | `aac889978efb034b95ab81ec59695d83ba7c879ab48cd5be2448c980f109a3bd` |
| `S/renderers/online_renderer.py` | `1ada26191425fd9d56c3525949f999826dad9f2c82df94ad586a1cf6f67b089e` |
| `S/renderers/inputs/preprocess.py` | `9f288d4203874d6421703e66b1c78e9be1a547201ec9b58bb22cf94c49b2cc57` |
| `S/renderers/base.py` | `0f440d8fcfe416c8a4665924d0602fd290152a1f2246cf0ad5d6e953b49ff86d` |
| `S/renderers/params.py` | `ad38852674ebc5bbe69aa9df91602847d518382cc2ceb0a25aac1d07be66b504` |
