# Mixed sampling routing audit after resume

Source-only independent audit, 2026-11-16. No hardware, server requests, implementation edits, or test changes were made. Scope: the one failure in the final full sampling run; this is not a stage-review verdict.

**Finding: admission-dependent host/device routing is a concrete source-supported explanation, but the saved failing run lacks the request-level evidence needed to establish it. Keep the failure unresolved.** Do not label it harmless or claim that all other requests in this test reproduced.

## Observed failure and evidence limit

`readiness_vllm/sampling_final_summary.json`, `sampling_execution.json`, and `sampling_tests.log` record 67 passed, 74 canonical skips, and one failed test in 2299.30 seconds. `TestBatchIsolation::test_mixed_params_batch` fails on the first checked shuffled request, `TopP: `, temperature 1, top-p 0.5, seed 7, eight output tokens. The two strings are `0.7\nTemperature: 0` and `0.7\nTopK: `. The assertion stops at shuffled index zero; the log does not establish agreement for the other nine requests.

I inspected the exact server byte interval recorded by the execution artifact: `[607400, 839963)`, 232563 bytes / 4259 lines. It has 699 device and 1615 host decode markers, but **zero `QWEN_VLLM_EVENT` entries** and no prompt, request ID, seed-7, or admission records. HTTP access lines identify connections and responses, not model request-to-slot ownership. Thus the interval does not establish the target's first-token route, mid-request route, shuffle order, companion admission/removal, or first divergent token's sampler. Aggregate mode counts cannot establish causality.

## Source chain

- `../vllm-tt-plugin/tests/tt/test_request_isolation.py:13` runs ten requests concurrently, shuffles using an unrecorded Python random permutation, and runs them again. Its comment correctly identifies one sampler per forward and different host/device streams, but constrains only top-k. Four companions have nonneutral repetition, presence, or frequency penalties. All explicit top-k values fit 32; absent top-k inherits the recorded server default 20.
- `tt/serving_capabilities.py` declares the real boundary `supports_device_penalties=False`. `tt/serving.py` rejects nonneutral device controls, and its reset path explicitly preserves neutral penalty buffers. `tt/generator.py::_sampling_forward` calls the canonical sampler with `penalties_on=False`. Changing this declaration alone would be false capability advertising.
- `../vllm-tt-plugin/src/vllm_tt_plugin/model_runner.py:3983` chooses the sampler per forward. At lines 4042–4045, any active penalty request plus that capability forces the whole submitted forward to host. `input_batch.py:384` defines `no_penalties` over active request sets. Unlike the top-k branch, this penalty predicate does not narrow itself to `sampling_rows`. Therefore even a resident penalty companion not selected for a particular prefill can affect routing.
- The host CPU generator is request-owned (`input_batch.py:152`, `host_sampler.py:67`); device RNG is generator-owned (`tt/generator.py:241`, canonical seed update path). No shared RNG stream synchronizes draws across the two samplers. The adapter selects device tokens exactly when sampling parameters are present and otherwise delegates logits through the explicit host compatibility path. It does not choose a persistent per-request sampler.

Consequently either a different first-token route or a route transition at companion admission/removal can change a seeded request's output without changed logits. These are two distinct hypotheses; the old log proves neither occurred for this target. A penalty-only resident batch may also remain host throughout, in which case another cause must be investigated.

## Smallest useful follow-up

Do **one bounded targeted reproduction** of this exact pytest node on the final server configuration, retaining both ten-request batches, their explicit permutation, full responses, and timestamps. Do not repeat the full suite merely to gather this information. Before running it, ensure diagnostic events can map each request to its slot and every actual draw, including the prefill draw: route, request/slot identity, output position, seed, active companions and their penalty eligibility, and sampling-state reset/reload commands. Existing audit-off compact markers are insufficient; even audit-on slot events need an unambiguous prompt/request-to-slot mapping. Prefer model-owned diagnostic logging over a shared plugin modification; any plugin instrumentation still follows plugin-change rules. Do not request logprobs as a diagnostic shortcut: that changes this model's route to host and invalidates the experiment.

The decisive comparison is the target's route/position timeline between the two runs. If output diverges after different routes or an admission-linked transition, record a routing defect and its exact admission pattern. If both timelines are identical, refute this routing explanation for that reproduction and check logits/RNG ownership at the first divergent step. Existing `logit_determinism/{http,standalone}.json` PASS evidence is useful prior control, but it is not a same-prompt, same-prefix capture of this failure. A passing uninstrumented retry would not retrospectively explain the recorded failure.

Do not force every seeded request to host, remove penalty companions, widen the declared top-k bound, or alter the canonical assertion to obtain a pass. Canonical device penalty support would require a real counted warmup, trace-safe history/count handling, and evidence; it is an implementation project, not a source-only conclusion of this audit.

## Classification under the selected skills

`vllm-integration/SKILL.md:257–261` allows reproducibility-only failures to be separated only when correctness, logprobs, crash-free serving, and qualitative output pass; it explicitly says a seeded request must not change sampler mid-request and admission evidence should identify routing defects. `stage-review/SKILL.md:243–248` calls the passing/failing admission comparison decisive. Those conditions are not supplied by the compact log. The independently recorded constrained thinking-on qualitative failure is also not cleared by diagnosing this sampling mismatch. Keep raw full-suite status **FAIL** and report the suspected routing issue separately until this focused evidence exists.

## Prepared bounded probe, 2026-11-17

Added only `tests/run_vllm_mixed_route_probe.py`; no adapter/plugin implementation was edited. It sends exactly two concurrent ten-request batches using the canonical configurations, first in canonical order and then in an explicitly saved shuffled order. The default client shuffle seed 7 gives `[8,3,1,4,7,0,9,6,2,5]`; this is a reproducible new reproduction order, **not** the unknown historical failure order. `--second-order` accepts an exact alternative permutation if another run's recorded order becomes available. There is no retry loop. All request seeds, token caps, penalties, omitted top-k defaults, and text-equality assertions remain canonical. Comparisons report every mismatch rather than stopping collection at the first assertion failure.

The only extra HTTP body fields are a diagnostic `request_id` and `return_token_ids=True`. The pinned completion protocol documents the latter as returning IDs alongside generated text; it does not request logprobs or switch sampler eligibility. Each response's actual ID, complete JSON, output/prompt token IDs, exact payload, client submit/completion UTC and monotonic timestamps, and prompt-token digest are saved atomically as requests finish. Requested IDs and actual response IDs are separate fields. Submission order is explicitly not claimed to be engine admission order. The collector fails if output IDs, prompt IDs, or actual response IDs are missing.

Parent-only execution, after outstanding hardware work finishes and an audited server is idle:

```bash
source python_env/bin/activate
source ../lane-env.sh
timeout 600 python -m models.autoports.qwen_qwen3_6_35b_a3b.tests.run_vllm_mixed_route_probe \
  --output models/autoports/qwen_qwen3_6_35b_a3b/readiness_vllm/mixed_route_probe.json \
  --request-timeout 180 --shuffle-seed 7
```

The collector requires a fresh output filename to preserve previous evidence. It does not query hardware, start or stop a server, reset devices, change sampler policy, or alter serving settings. No live request was sent during implementation. CPU verification: canonical configuration list compared exactly with the AST-extracted list from the pinned plugin test; offline `httpx.MockTransport` covered an equal pair and a deliberate case-9 mismatch, including response IDs/token IDs and all 20 records; plan-only and compile checks passed; file pre-commit hooks passed.

### Minimal adapter-only diagnostic addition proposed to parent

Existing audit events already expose prefill `starts`, `ends`, `slots`, route and active decode slots; decode exposes route, `slot_remap`, reload flags, and post-submission active slots. They do not expose which logical request owns each slot. Under `QWEN_VLLM_AUDIT=1`, add:

- A monotonically increasing adapter submission/event ID linking each decode event to its post-submission event. Existing `monotonic` time remains useful; no device read is needed.
- At prefill, each CPU row's physical slot and exact logical token IDs `tokens[row, start:end]` (these probe prompts are short), or SHA256 of compact JSON `json.dumps(ids, separators=(",", ":"))`. The collector's `prompt_token_digest` uses that encoding. Include original logical interval. This maps response request IDs to admitted slots even though the adapter API does not receive request IDs.
- At decode, submitted `slots` (including the explicit default row order), host `start_pos` marked **advisory**, `reload_sampling_params`, `reset_sampling_state`, and sampling parameter seeds when present. Keep the existing full slot permutation and post-call active slots. Derive the logical request's output-step index offline from its terminal prefill and subsequent active decode submissions, applying the permutation; do not read device positions or tokens to make this log.

Those fields permit an offline table `response ID -> prompt digest -> slot ownership -> submission/output index -> route` for both orders. Companion penalty eligibility comes from the saved exact request payloads and their mapped active ownership; the adapter need not reconstruct or override plugin routing. Host-only forwards have no sampling parameter payload, so seed-only logging would be insufficient. For this short-prompt workload every prefill is terminal; do not generalize prefill host-logit events into sampled draws for intermediate chunks in other workloads.

Remaining work: parent adds those optional CPU-only diagnostic fields, captures one pair on the same server, and judges the observed admission/route timelines. No causality or passing hardware result is claimed by the prepared script. If the pair passes, preserve that result without discarding the original failure or claiming that it reproduced an old admission pattern.

## CPU route-attribution analyzer prepared

`tests/analyze_vllm_mixed_routes.py` consumes the probe JSON and the parent's new optional adapter audit events. It uses a full prompt-token digest plus the response's overlapping client monotonic interval to identify each terminal prefill. The command requires explicit confirmation that client and server clocks share the same host boot. Repeated prompts across the two batches are distinguished by these intervals, not by arrival/submission order. Overlapping identical prompts remain ambiguous.

The analyzer pairs `decode` with `decode_submitted` by submission ID, requires adjacent relevant events, applies `new_slot[i] = old_slot[slot_remap[i]]`, then uses the submitted event's actual active slots. New prefill ownership replaces the prior slot owner. It records real adapter admission order, per-request host/device route sequences and transitions, paired submission line numbers, active companion response IDs, and differences between the two cases' timelines. It does not use advisory positions to update ownership or infer a host sampler from output strings.

Missing identity, unsupported chunked-prefill attribution, ambiguous prompts, unknown active slots, invalid permutations, missing pairs, or events outside a request interval produce `INCOMPLETE` (CLI exit 2). `COMPLETE_ATTRIBUTION` means only that the optional adapter events can be attributed confidently. It is not a sampling-test pass or causal proof. Events identify the route of a model forward; host sampling occurs later in the plugin, and async submitted forwards can outnumber emitted response tokens. Full sequences are preserved without silently trimming speculative/trailing submissions. Original token/text assertions and logit evidence remain separate.

After the parent runs the pair, invoke without hardware access:

```bash
python -m models.autoports.qwen_qwen3_6_35b_a3b.tests.analyze_vllm_mixed_routes \
  --probe models/autoports/qwen_qwen3_6_35b_a3b/readiness_vllm/mixed_route_probe.json \
  --server-log models/autoports/qwen_qwen3_6_35b_a3b/readiness_vllm/mixed_route_server/server.log \
  --output models/autoports/qwen_qwen3_6_35b_a3b/readiness_vllm/mixed_route_analysis.json \
  --same-host-clock
```

Seven CPU fixtures in `tests/test_vllm_mixed_route_analysis.py` pass: non-self-inverse three-slot permutation, later slot reuse, overlapping duplicate prompt ambiguity, mismatched submission pairing, explicit clock-domain requirement, missing prompt attribution, missing actual response identity, and invalid-remap rejection (some covered together). Exact final fixture command: `python -m pytest --noconftest -o addopts='' -q models/autoports/qwen_qwen3_6_35b_a3b/tests/test_vllm_mixed_route_analysis.py` -> 7 passed in 0.02 seconds. No server requests, device accesses, or production edits were performed for the analyzer.

## Actual diagnostic attribution, 2026-11-17 02:49 UTC

**The new pair establishes admission-dependent routing and a seeded mid-request sampler change for case 1 (`Random: `).** This is now observed behavior, not just the earlier source hypothesis. The original full-suite `TopP: ` failure remains historically unattributed; that case passes identically on host throughout this new pair.

Corrected analysis is preserved separately as `readiness_vllm/mixed_route_analysis_legacy_parsed.json`: `COMPLETE_ATTRIBUTION`, 20 confidently mapped requests, 18 paired decode submissions, zero attribution issues. Every mapped request has exactly as many terminal-prefill-plus-decode forwards as returned tokens in this pair. The original `mixed_route_analysis.json` remains untouched. The new analysis preserves each raw tensor repr as `slot_remap_raw` alongside its parsed integer permutation.

The old adapter serialized CPU remaps as complete multiline `tensor([...], dtype=torch.int32)` strings. The analyzer now accepts only a bounded full-match grammar for this exact CPU int32 representation, followed by `ast.literal_eval` and the existing full-permutation validation. It rejects ellipses, alternate dtypes/devices, expressions, booleans, floats, negative values, trailing code and oversized values. Nine CPU fixtures pass, including raw preservation and negative grammar cases. This repairs evidence parsing only; no production source, server setting or request was changed, and no HTTP/device retry was run.

The request parameters are identical: `Random: `, temperature 2, top-k 32, top-p 1, seed 42, max_tokens 5, neutral penalties, no logprobs. Response IDs are `cmpl-mixed-route-4422fb852cd9-b0-c1` and `cmpl-mixed-route-4422fb852cd9-b1-c1`. Both returned prompt tokens `[13597,25,220]`, digest `ff013bcc07f2f3ad29efd696b4620d0ffe0b42b379fdd79d973bf9328e4d44d9`.

| Evidence | Batch 0 | Batch 1 |
| --- | --- | --- |
| Actual prefill admission | Nine-row group after standalone `Count: ` | Standalone first prefill, despite client submit index 2 |
| Prefill slot / server log line | Slot 1 / 2757 | Slot 0 / 2796 |
| Prefill route | Host | Device |
| Subsequent decode routes | Host, host, host, host | Host, host, host, host |
| Decode submission IDs | 1–4 | 10–13 |
| Decode slots | 0,0,0,0 | 9,9,0,0 |
| Returned token IDs | 95772,14605,95789,220,110458 | 102038,96931,198,16,220 |
| First divergent output index | 0 | 0 |

Actual prefill records in `readiness_vllm/mixed_route_server/server.log` show:

- Line 2755: batch-0 case 0 alone on device. Line 2757: cases `[1,2,3,5,4,6,7,8,9]` together on host. This group includes the four penalty-bearing cases 2, 3, 7 and 8.
- Line 2796: batch-1 case 1 alone on device. Line 2798: cases `[4,7,8,0,3,9,2,5,6]` together on host. The same four penalty-bearing requests are present before the next decode.
- Lines 2801–2802: decode submission 10 includes all ten batch-1 requests on host, with the complete permutation moving case 1 from slot 0 to slot 9. This is the first host decode after that request's device prefill. Its later move to slot 0 is captured at submission 12.

The first unequal token was generated by different prefill samplers; it occurs **before** the observed device-to-host transition. Therefore the first mismatch must not be described as first appearing when the host stream advances on decode. The later transition is an independently demonstrated violation of stable seeded sampler ownership. The pinned routing source and real penalty boundary explain why companion membership selects host for a whole forward; the two RNG streams are different by construction. This supports classification as an admission-dependent seeded routing defect, not evidence by itself of bad logits or cross-request token contamination.

Classification boundaries: the probe remains canonical **FAIL**, with one unequal case and nine exact token/text matches; diagnosing it does not convert that failure into a passing test. No same-prefix logit/noise control was collected for this exact `Random: ` pair, so exclusive numerical causality is not claimed. Five-token high-temperature legacy completions are not a qualitative acceptance suite. The independent constrained thinking-on quality failure and other required gates remain unaffected. The original uninstrumented `TopP: ` failure cannot be retroactively assigned this pair's actual admission pattern.
