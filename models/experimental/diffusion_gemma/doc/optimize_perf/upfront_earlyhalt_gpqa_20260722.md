# Up-front capture + traced early halt on real GPQA (2026-07-22)

## Result

Passed eight sequential `r1_gpqa_diamond` requests on full 30-layer
DiffusionGemma, 4× Blackhole p300c, with one startup capture and no recapture.
All eight requests released cleanly and halted in 10–43 denoise steps.

The original second-request hang was reproduced exactly:

- request 0 completed at K=13;
- request 1 reached `prefill_device_begin` but never `prefill_device_end`;
- live `tt-triage` found all four devices in the causal-prefill
  `AllBroadcast` writer, waiting on its semaphore.

## Root cause

The failure is not early-halt-specific:

1. Replaying all skipped early-halt traces before request teardown did not fix
   the next prefill.
2. Fixed-K up-front capture hung on the same second GPQA prefill.
3. Warming the 160-token prefill program before trace capture made both GPQA
   requests pass.

Up-front capture had only warmed the 32-token mock prefill. The first real
160-token prompt compiled a new prefill shape while denoise traces were already
active. That post-capture compilation/allocation corrupted trace/CCL state; the
next causal prefill stalled in `AllBroadcast`.

The fix warms every admitted aligned prefill length during vLLM's compile-only
warmup phase, before denoise trace capture. Runtime requests whose aligned
length was not warmed fail loudly instead of risking device corruption.

## Real GPQA validation

Configuration:

- `DG_UPFRONT_CAPTURE=1`
- `DG_DENOISE_REVEAL_MASK=1`
- `DG_DENOISE_EARLY_HALT=1`
- `DG_DENOISE_REVEAL_PMAX=1024`
- `DG_UPFRONT_PREFILL_WARMUP_LENS=160,192,256,384,480`
- argmax, K≤48, early-halt window 1
- 10 GiB trace region
- `num_concurrent=1`, one 256-token output block per question

| sample | prompt/cache | realized K | TTFT (s) |
|---:|---:|---:|---:|
| 0 | 154/160 | 13 | 6.62 |
| 1 | 159/160 | 16 | 7.76 |
| 2 | 184/192 | 12 | 6.08 |
| 3 | 165/192 | 43 | 18.63 |
| 4 | 380/384 | 10 | 5.38 |
| 5 | 231/256 | 13 | 6.49 |
| 6 | 191/192 | 19 | 8.85 |
| 7 | 478/480 | 17 | 8.18 |

`capture_events` remained 48 throughout; eight `request_release` events were
recorded. The lm-eval exact-match score was 0 because the one-block 256-token
cap truncates GPQA reasoning before its final answer. This experiment is the
multi-request trace-lifecycle gate, not an accuracy claim.

Machine-readable evidence:
`upfront_earlyhalt_gpqa_20260722.json`. Exact hang triage:
`triage/upfront_earlyhalt_gpqa_hang_{tt-triage,summary}.txt`.
