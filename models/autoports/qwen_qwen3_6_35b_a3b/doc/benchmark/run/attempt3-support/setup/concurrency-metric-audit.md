# Upstream concurrency metric audit

Source and saved-JSON review, 2026-11-18 UTC. No network, model imports, device calls, hooks, commits, or live-artifact changes.

**Finding:** `perf-b16.json`'s `max_concurrent_requests=18` is a one-second bucket occupancy statistic, not evidence that 18 requests ran simultaneously. The installed upstream client enforces the configured request bound of 16 with an asyncio semaphore. The same behavior explains b32's derived value of 34 with a configured bound of 32.

Installed sources (absolute paths):

- `/home/ttuser/qwen36-35b-a3b/benchmark-vllm-env/lib/python3.12/site-packages/vllm/benchmarks/serve.py`: the caller passes `args.max_concurrency` into `benchmark()` at line 2114 and saves the same CLI setting at line 2156. `benchmark()` constructs `asyncio.Semaphore(max_concurrency)` at lines 962–966; its nested `limited_request_func()` holds the semaphore across the complete awaited request function at lines 968–972. Every measured task is created through that wrapper, and `asyncio.gather(*tasks)` awaits them all at line 1043. The shared `aiohttp.TCPConnector` additionally sets both connection limits to the same configured bound at lines 807–809.
- The same file's `calculate_metrics()` (line 556) computes the derived metric at lines 669–706. It truncates each successful request's start and end offsets to integer seconds, then increments **every inclusive bucket** from start through end. `max_concurrent_requests` is the largest bucket count. Sequential requests touching different parts of one second both contribute, so this statistic can exceed the semaphore limit.
- `/home/ttuser/qwen36-35b-a3b/benchmark-vllm-env/lib/python3.12/site-packages/vllm/benchmarks/lib/endpoint_request_func.py`: `ASYNC_REQUEST_FUNCS` maps `vllm` to `async_request_openai_completions` at line 874. That function records `start_time` after entering the wrapper, consumes the response stream, and returns after the response context exits (lines 158–268). Its `latency` ends at the last choices-bearing stream chunk, not semaphore release or server slot release (line 257).

Saved evidence and lightweight reproduction using only JSON and Python standard-library arithmetic:

| Artifact | CLI/semaphore bound | Completed / failed | Upstream inclusive-second peak | Reconstructed stream-interval peak |
| --- | ---: | ---: | ---: | ---: |
| `attempt3/perf-b16.json` | 16 | 48 / 0 | 18 | 16 |
| `attempt3/perf-b32.json` | 32 | 96 / 0 | 34 | 32 |

Reconstruction used `start_times`, `ttfts`, and `itls`, with an end estimate of `start + ttft + sum(itls)`. Sorting starts/ends without second rounding gives the rightmost column; applying the upstream inclusive integer-second binning exactly reproduces both saved peaks. This is a stream timing reconstruction, not an independently captured HTTP completion or server occupancy trace. TTFT's timestamp is taken just after the first chunk timestamp, so reconstructed ends have a tiny positive offset relative to the client's unsaved per-request latency; this does not affect the reproduced counts here.

Concrete b16 example, offsets relative to its earliest start: request 0 finishes its measured stream at approximately 1459.093801 seconds and request 16 starts at 1459.094142; request 1 finishes at 1459.352407 and request 17 starts at 1459.352705. Bucket 1459 counts all four plus the other 14 original requests, yielding 18 even though each replacement starts after its predecessor's stream ends. B32 likewise counts two replacements in bucket 3041, yielding 34. B16 maximum buckets are 1459 and 2926; b32 maximum buckets are 3041 and 6085.

Reporting implication: report **server slots** from validated `max-num-seqs` separately from **configured concurrent requests** (16 or 32). Retain upstream raw `max_concurrent_requests` unchanged; if displayed, label it “peak requests touching one-second bucket” and explain the rounding. Do not label 18 or 34 as actual simultaneous requests or use these derived counts to claim a concurrency-contract failure. The source-enforced measured-client bound and saved successful request counts support the requested workload settings; this audit makes no separate claim about scheduler occupancy.
