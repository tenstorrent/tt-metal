# Applied overhead validation and matched HTTP control

Source-only follow-up, 2026-11-17. No device jobs, HTTP requests, or production edits were performed by this audit. Parent applied the proposal after the baseline ended. The previously generated patch must not be regenerated against applied sources.

## CPU validation

`test_overhead_patch.py --applied` reads actual production sources directly; it does not import or execute the patch builder. Six tests PASS. Coverage: immutable remap validation/capacity rejection, one conversion across repeated normalization, direct-generator default event waits, actual adapter process hook forwarding its new keyword through the actual generator method, two simultaneously pending independently consumed leases, synchronous published-output blocking read, stale token/position ignorance and page/remap command suite, and the unchanged per-step sampling ledger AST hash. The existing seven `test_vllm_adapter_host.py` tests PASS without mock changes: none has a keyword-intolerant process-hook mock. New HTTP parser tests PASS3; they reject token/prompt/usage mismatch and do not invent per-token ITL for multi-token chunks.

```bash
python_env/bin/python models/autoports/qwen_qwen3_6_35b_a3b/doc/vllm_integration/sampling_debug/test_overhead_patch.py --applied
python_env/bin/python models/autoports/qwen_qwen3_6_35b_a3b/doc/vllm_integration/sampling_debug/test_matched_b2_http.py
python_env/bin/python -m pytest -q models/autoports/qwen_qwen3_6_35b_a3b/tests/test_vllm_adapter_host.py
```

## Synchronous ordering is safe, but not always event-based

Pinned plugin source root: `/home/ttuser/qwen36-35b-a3b/vllm-tt-plugin`.

- `src/vllm_tt_plugin/async_decode.py:1653–1677`: `finalize_decode` synchronizes every returned read event before invoking `process_decode_output_host`, independent of scheduling mode.
- `src/vllm_tt_plugin/model_runner.py:4192–4198`: ordinary synchronous decode submits `read_from_device=False, async_read=False` then finalizes. It has no read event. The adapter receives a published handle; `DecodeOutputPool.process` still calls its blocking read before host conversion. The new flag only skips the subsequent redundant event loop.
- `src/vllm_tt_plugin/model_runner.py:3017–3029`: synchronous lane submits `read_from_device=True` in the ordinary supported path. Low-level conversion blocks before returning an already-host output; finalization skips the adapter process hook.
- `src/vllm_tt_plugin/async_decode.py:1311–1319`: selective host readback may instead force deferred async reads even for a synchronous caller. The same finalizer waits on those events.
- `tests/run_serving_commands_probe.py:230–234` likewise waits for returned events before the adapter process hook.

Therefore the patch does not assert that every synchronous path has an event. It retains both valid forms: externally completed read events or a blocking read performed by the output pool. Calling the adapter process hook early on a reading handle would violate the documented plugin contract; direct generator callers retain waiting by default.

## Matched probe validation and timing limits

`run_matched_b2_overhead.py` resolves `ROOT` to the model autoport and `REPO` to tt-metal correctly. Its source-derived server geometry requires exactly one initialization and cache allocation, all40 layers, capacity32, context262144, ten BF16 attention caches, and actual physical shape `[8224,1,32,256]` in the baseline log. It validates selected precision and actual allocated cache identity/shapes. It intentionally does not substitute standalone8192 pages. `ROOT`/`REPO` resolution, baseline fixture length, and parser schema were checked on CPU.

The workload is exactly128 input/128 output tokens at one user. The first output comes from prefill; decode timing covers127 subsequent steps, including the first authoritative input reload and final drain. Serial native elapsed/127 is serial token-out wall time. Adapter async2 elapsed/127 is queued throughput; its recorded per-submission-to-consumption latencies are not HTTP ITL or serial TPOT. Native and adapter use the same process/cache/traces; they exclude HTTP scheduler and network costs. The identity physical-page mapping does not reproduce arbitrary server allocator history. Modes execute in groups, so temporal ordering can matter.

The earlier primary random prompt is a different token fixture. Use the command below for exact-input comparison. The new HTTP helper requires baseline PASS, sends the saved raw128 IDs, checks the returned prompt IDs, verifies all128 output IDs equal the native baseline, requests no logprobs, and preserves canonical greedy sampling. Every request uses a distinct supported `cache_salt`, so repeated prompts cannot gain a prefix-cache TTFT advantage over the reset/native prefill. It saves timestamped raw SSE JSONL and records first-to-last-token TPOT/127, TTFT, throughput, actual per-token ITL when available, and P50/P99. Two measured samples give descriptive percentiles, not a stable tail estimate. A warmup request is excluded.

## Exact final HTTP comparison command

Run against the owner's otherwise-idle final server with selected precision, capacity32, context262144, retained B2/B32 and the intended final sampling implementation. No server is launched by this script.

```bash
python_env/bin/python models/autoports/qwen_qwen3_6_35b_a3b/doc/vllm_integration/sampling_debug/run_matched_b2_http.py \
  --base-url http://127.0.0.1:8000 \
  --model Qwen/Qwen3.6-35B-A3B \
  --baseline-json models/autoports/qwen_qwen3_6_35b_a3b/readiness_vllm/serving_overhead/baseline.json \
  --prompt-json models/autoports/qwen_qwen3_6_35b_a3b/readiness_vllm/serving_overhead/baseline.prompt.json \
  --repetitions 2 \
  --output models/autoports/qwen_qwen3_6_35b_a3b/readiness_vllm/serving_overhead/http_final.json
```

The helper is CPU-tested but has not been run against a server. HTTP TPOT includes delivery scheduling and uses the first received token as its boundary; native timing begins immediately before the first decode submission. HTTP TTFT includes transport, scheduling and prefill, whereas native `prefill_seconds` excludes transport and scheduler. Report these distinctions rather than attributing their difference entirely to device performance.
