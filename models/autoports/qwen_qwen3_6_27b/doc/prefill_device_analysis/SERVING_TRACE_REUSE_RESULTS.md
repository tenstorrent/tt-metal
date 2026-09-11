# Reduced serving-adapter trace reuse experiment

**Status: experimental, unselected.** The instance-only prototype preserves a
warmed decode trace across an in-place slot reset and eager prefill. Four-layer
correctness passed on four devices, including changed prompt contents. The
initial guarded implementation regressed request time; removing redundant
explicit garbage collection recovered a modest measured gain. No production
defaults changed, and neither full-model nor HTTP improvement is established.

The source hypothesis and ownership ledger remain in
[SERVING_TTFT_FOLLOWUP.md](SERVING_TTFT_FOLLOWUP.md). This experiment does not
resolve the approximately 521 ms output-limit HTTP delta or the remaining
prefill floor. It measures a bounded mechanism that could address part of that
cost. Separate full-model adapter measurements reported 529–535 ms setup; the
32.9 ms setup measured here is for four layers and cannot establish that the
same reuse policy succeeds on 64 layers.

## Implementation and closed envelope

[serving_trace_reuse_runtime.py](serving_trace_reuse_runtime.py) installs methods
only on the supplied adapter and generator instances. It uses the reviewed
[host policy](serving_trace_reuse_plan.py); it restores original methods on exit.
There are no global hooks, deallocation overrides, corruptible-allocation
acknowledgments or RNG changes.

- Real Qwen3.8-27B checkpoint, revision
  `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`, TP4, layers 0–3: three linear
  attention layers and one full attention layer. Native prefill and selected
  precision are recorded in each JSON artifact.
- Batch allocation 32, one active request in slot 0, exact logical and physical
  S128, maximum context 256, identical cache owners and page-table shape.
- Unseeded greedy sampling: top-k 1, top-p 0, temperature 1, no penalties or
  logprobs. Two different prompts alternate A/B/A/B. They are truncated to the
  exact shape for this test and are not qualitative or full-model accuracy
  evidence. The four-layer model returned token 220 for every active step.
- A successful first capture warms the exact reset/prefill/sampler envelope.
  Compatible later requests drain work, reset linear caches in place, run the
  existing eager prefill and request-state handling, and leave decode inputs
  requiring their existing explicit reload. Setup then uses the existing
  page-table refresh and token/position seed copies without warmup or capture.
- Cache, trace input/output and page-table tensor owners, addresses and specs
  must match. Sampler ownership is read from its actual `_trace_states` dict.
  The active slot and both masks stay fixed. Contract-format exceptions reject
  reuse. New programs, changed owners, incomplete prefill or any tracked unsafe
  allocation cause release before replay and ordinary recapture.

Allocation tracking is required at process startup, including program-owned
buffers. The prototype checks both model and sampler traces after prefill and
again before setup reuse. The runtime also checks before each replay. The
refinement removes only an explicit `gc.collect()` before the prototype's
direct allocation query: uncollected cycles conservatively appear unsafe and
force recapture. The zero-unsafe condition is unchanged.

## Correctness and allocation evidence

The first correctness-only run and both timing runs each separately validated
four original requests followed by four reuse requests. Every request performs
prefill, three decode steps and the next slot reset. Comparison reads are
outside the later timing samples.

- All four reuse requests match their corresponding original request's first
  token, all-rank decode token vectors, and all-rank final positions. The active
  position is 131 after the three decode steps. Host decode inputs deliberately
  remain stale after the first authoritative reload, exercising resident
  feedback.
- All **32 cache tensor/rank digests per request** match exactly: every fixed
  slot of all conv, recurrent, key and value tensors. Digests cover the logical
  values returned to the host, including exact host byte representation; they
  do not inspect unused physical padding. All 32 digests change between the two
  prompts, so the equality result includes genuinely different cache contents
  despite the identical sampled active tokens.
- The refined run preserves six request boundaries, with **zero surviving
  ordinary allocations and zero surviving program-owned allocations** observed
  at each boundary. Program-cache entries remain **282 → 282**. Original cache
  and persistent trace owners remain stable. There are no guard recaptures or
  tracker errors.
- In that run, four correctness requests plus one discarded warmup and two
  timed requests produce **7 captures in the original arm versus 1 capture and
  6 reuses** in the candidate. The correctness-only subset has 4 versus 1
  capture. This proves reuse occurred; it is not a silent fallback comparison.

Page-table contents are unchanged in these runs. Changed mappings, different
slots, lengths, masks, remaps, seeds and non-greedy modes remain unvalidated on
hardware. Host policy checks cover rejection decisions, not those device
transitions. Full 64-layer cache ownership and the server's larger KV geometry
remain unvalidated.

## Timing, including the failed attempt

These are host adapter boundaries including token readback, **not HTTP TTFT**.
Each arm has separate correctness requests, one timing warmup and then the
listed samples. Original runs first in each process. Small sample counts and
fixed arm order limit conclusions about differences of a few milliseconds.

| Attempt | Samples/arm | Original prefill + first decode | Reuse prefill + first decode | Result |
| --- | ---: | ---: | ---: | --- |
| Explicit GC in two candidate guards | 3 | 857.591 ms | 1455.190 ms | Regression of 597.599 ms; rejected |
| Direct allocation query, no explicit GC | 2 | 847.928 ms | 821.355 ms | Reduction of 26.573 ms (3.13%); experimental |

The initial candidate spent median 314.858 ms in its prefill guard and 313.456
ms in its setup guard. It reduced setup work from 33.021 to 0.418 ms while
making the total request substantially slower. That result is retained and is
not presented as a performance win.

| Refined run span, median | Original | Reuse |
| --- | ---: | ---: |
| Prefill work inside adapter | 183.139 ms | 188.277 ms |
| Prefill guard | 0.230 ms | 0.888 ms |
| Setup work, excluding prototype guard | 32.867 ms | 0.111 ms |
| Setup guard | 0.018 ms | 0.189 ms |
| Whole first decode, including readback | 664.522 ms | 632.154 ms |

TTNN's own `UnsafeAllocationTracker.verify_before_replay` still runs
`gc.collect()` for each trace replay (`ttnn/ttnn/unsafe_allocation_tracker.py:77`).
Both arms therefore retain substantial debug overhead, including the model
and sampler checks. The separate setup spans establish removal of the reduced
capture work; subtracting debug overhead from these totals would not produce a
measured production latency. A production ownership policy without debug
tracking, followed by full-model correctness and actual HTTP A/B, is required
before selection. No such policy or performance extrapolation is selected here.

## Reproduction and preserved evidence

Run only in the coordinating agent's exclusive hardware window. From the repo
root, the final bounded run was:

```bash
env TT_METAL_HOME=/home/mvasiljevic/tt-metal \
  PYTHONPATH=/tmp/qwen-ci-vllm-plugin/src:/home/mvasiljevic/tt-metal \
  HF_HUB_OFFLINE=1 \
  QWEN_AUTOPORT_MODEL_ID=Qwen/Qwen3.8-27B \
  QWEN_AUTOPORT_MODEL_REVISION=1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0 \
  TT_METAL_TRACE_ALLOC_TRACKING=1 \
  TT_METAL_TRACE_ALLOC_TRACEBACKS=0 \
  TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0 \
  timeout 180 python_env/bin/python \
  models/autoports/qwen_qwen3_6_27b/doc/prefill_device_analysis/validate_serving_trace_reuse.py \
  --timed-requests 2 \
  --output models/autoports/qwen_qwen3_6_27b/doc/prefill_device_analysis/artifacts/serving_trace_reuse_without_gc_timing.json
```

For the earlier correctness-only run, use `--correctness-only`. The initial
timing run used `--timed-requests 3` and the same helper with `import gc` plus
`gc.collect()` at the beginning of `_allocations`; that exact behavioral
difference is preserved here. This experiment did not modify production code.

- [Compact summary, commands and current helper hash](artifacts/serving_trace_reuse_summary.json)
- [Initial correctness JSON](artifacts/serving_trace_reuse_correctness.json),
  [log](artifacts/qwen_serving_trace_reuse_correctness.log)
- [Rejected timing JSON](artifacts/serving_trace_reuse_timing.json),
  [log](artifacts/qwen_serving_trace_reuse_timing.log)
- [Refined timing/correctness JSON](artifacts/serving_trace_reuse_without_gc_timing.json),
  [log](artifacts/qwen_serving_trace_reuse_without_gc_timing.log)
- [Host policy checks](artifacts/serving_trace_reuse_host_validation.json)

All runs exited successfully. The final run cleanly closed every device at
2026-09-11 **15:09:00.711 UTC**, after which hardware ownership returned to the
coordinator/seed validation agent. No full-model or server reuse run followed
under this agent's hardware window.
