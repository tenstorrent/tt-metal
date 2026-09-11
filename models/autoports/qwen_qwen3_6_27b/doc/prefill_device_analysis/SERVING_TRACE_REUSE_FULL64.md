# Full-model guarded decode-trace reuse

**The 64-layer closed-envelope experiment passed.** Repeated request setup
retained one model/sampler capture and refreshed resident inputs instead of
warming and recapturing. Median setup work fell from **532.666 ms to 0.171 ms**.
The prototype remains experimental: this is a tracked adapter measurement,
not production trace-reuse integration or an HTTP TTFT result.

## Scope and provenance

- Qwen3.8-27B revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`, all 64 layers,
  TP4, batch allocation 32, C1 in slot 0, logical/physical S128, maximum context
  256. Native prefill, device-fill logits scatter and the selected precision
  are the current production implementations. Precision and source hashes are
  recorded in the artifact.
- Repository HEAD `4ea57c41431a1e80318c4d32d249d5552c4f165f`. The experimental
  harness extension is an uncommitted source change recorded by its SHA256;
  the instance-only runtime policy and production defaults were not changed.
- Unseeded greedy sampling, top-k 1, top-p 0, temperature 1, no penalties or
  logprobs. Two different fixed-shape prompts alternate A/B/A/B. Their truncated
  chat formatting is a lifecycle test, not qualitative-generation evidence.
- Original and reuse arms each perform four correctness requests, one discarded
  timing warmup and three measured requests. Each request performs prefill and
  three decode steps. Full cache readback/hashing, about 4.4 seconds per
  correctness request, is outside the timed requests.
- The original four-layer artifacts remain unchanged. `--num-layers` defaults
  to 4 and now explicitly accepts 64; the build uses and verifies that count.

## First attempt: corrected harness gate

The first full-model attempt stopped on candidate request 0 **before any trace
reuse**. That request followed the ordinary release/capture path, with
`setup_reuses=0`. Active token history and active tokens on all four ranks were
exact, all **512 cache tensor/rank digests** were exact, and every slot's
position was exact. Only sampled tokens for inactive slots differed: 21, 21
and 20 inactive entries per rank at the three decode steps.

The original four-layer harness compared all 32 sampled token slots as a hard
gate. That is broader than the serving contract. `generator_vllm.py` supplies
`active_mask=(start_pos >= 0)` at setup; inactive positions are -1/UINT_MAX.
The plugin's `model_runner.py:2148–2156` selects the active request range
`[start, start + sz)` from returned tokens, which is only slot 0 for this C1
case. Inactive sampled values are not delivered or treated as live requests.

The correction explicitly gates **active slot 0 on every rank** and retains
complete raw token vectors plus inactive mismatch counts as diagnostics. It
does not relax comparison of any cache slot, any position, cache ownership,
trace ownership, program growth or unsafe allocations. The corrected run still
records inactive-token differences (17–22 entries per rank/step). The first
artifact and complete log are preserved as a harness-gate correction, not a
trace-reuse failure or an erased experiment.

## Correctness and allocation result

All four corrected candidate requests pass active-token history, active tokens
on every rank, every cache digest and all positions against the corresponding
original request. The 512 cache tensor/rank digests cover all fixed slots of
48 linear layers' conv/recurrent state and 16 full-attention layers' K/V state.
All 512 digests change between prompt A and B, while matching the corresponding
original request exactly. Digests cover exact host bytes of decoded logical
values, not unused physical padding.

Active position is 131 after three resident decode steps. Host token/position
inputs intentionally remain stale after the first authoritative reload, so
these checks exercise device token feedback and position advance. Timed token
histories also match their corresponding correctness prompt histories.

Across eight requests per arm, the original captures **8 times**; the candidate
captures **once and reuses 7 times**. The three reuses inside the correctness
phase are therefore real preserved request boundaries. There are no guard
fallbacks. All seven preserved-prefill checks report:

- Surviving ordinary allocations: **0**.
- Surviving program-owned allocations: **0**.
- Program-cache entries: **283 → 283**.
- Stable cache and persistent trace owners; exact active-slot/mask envelope.

The runtime also checks ownership and unsafe allocations before setup reuse,
and TTNN independently verifies allocations before every replay. Tracking
includes program-owned buffers; the harness rejects an environment that skips
them. No global hooks, allocator acknowledgments or check bypasses were added.

## Timing and its limits

Each value below is the median of three warmed samples. The control arm runs
first. Both arms retain mandatory trace allocation tracking and instrumented
host request boundaries including token readback.

| Span | Original | Reuse |
| --- | ---: | ---: |
| Prefill work inside adapter | 166.260 ms | 159.320 ms |
| Prototype prefill guard | 1.103 ms | 3.794 ms |
| Setup work, excluding prototype guard | 532.666 ms | 0.171 ms |
| Prototype setup guard | 0.017 ms | 0.800 ms |
| Whole first decode | 1171.084 ms | 635.311 ms |
| Prefill + first decode | 1337.089 ms | 798.832 ms |

The measured request-boundary reduction is **538.257 ms**. This verifies that
the complete model's approximately 533 ms warm/capture setup can be avoided
inside the tested envelope while preserving the checked state and outputs.

The table profiles the prototype guards separately. TTNN's intrinsic
`UnsafeAllocationTracker.verify_before_replay` still calls `gc.collect()` for
each model and sampler replay. That intrinsic time was **not independently
instrumented**; the remaining approximately 635 ms first-decode total includes
it and actual model/sampler/readback work. Neither subtracting an estimated GC
cost nor transferring the 538 ms saving to the HTTP benchmark would be a
measured production result.

Remaining work before promotion includes a production ownership policy without
the debug replay cost, actual serving KV/page-table geometry and content
changes, other slots, prompt lengths, remaps, C>1, and separately validated
sampling modes. No new HTTP performance or broader transition claim follows
from this closed-envelope result.

## Commands and evidence

The corrected run used:

```bash
env TT_METAL_HOME=/home/mvasiljevic/tt-metal \
  PYTHONPATH=/tmp/qwen-ci-vllm-plugin/src:/home/mvasiljevic/tt-metal \
  HF_HUB_OFFLINE=1 \
  QWEN_AUTOPORT_MODEL_ID=Qwen/Qwen3.8-27B \
  QWEN_AUTOPORT_MODEL_REVISION=1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0 \
  TT_METAL_DEVICE_PROFILER=0 TT_METAL_TRACE_ALLOC_TRACKING=1 \
  TT_METAL_TRACE_ALLOC_TRACEBACKS=0 TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0 \
  timeout 600 python_env/bin/python \
  models/autoports/qwen_qwen3_6_27b/doc/prefill_device_analysis/validate_serving_trace_reuse.py \
  --num-layers 64 --timed-requests 3 \
  --output models/autoports/qwen_qwen3_6_27b/doc/prefill_device_analysis/artifacts/serving_trace_reuse_full64_active_slots.json
```

- [Compact summary and source provenance](artifacts/serving_trace_reuse_full64_summary.json)
- [Complete measured JSONs and both logs](artifacts/serving_trace_reuse_full64_evidence.tar.gz),
  compressed without changing their bytes; the summary records entry hashes.
  Passing data is `serving_trace_reuse_full64_active_slots.json`; the first
  attempt is `serving_trace_reuse_full64.json`. Uncompressed copies remain in
  the workspace, but exceed the repository's 500 KB added-file limit.
- [Earlier four-layer results](SERVING_TRACE_REUSE_RESULTS.md)

The first process exited 1 with clean device closure at 15:52:52.927 UTC. The
corrected run exited **0**, closing every device at **15:59:12.672 UTC** on
2026-09-11. Hardware was handed back immediately; no further device run or
production trace-reuse edit was performed by this agent.
