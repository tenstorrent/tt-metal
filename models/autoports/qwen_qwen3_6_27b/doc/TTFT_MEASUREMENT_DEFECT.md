# Every TTFT number on this port is a single cold observation

Operator analysis, 2026-08-17, written while the corrected measurement is
queued. Two findings, both derivable from committed artifacts and source on this
branch without running anything.

## Finding 1: the TTFT measurement includes one-time program build

`tests/full_model_perf.py` times the **first and only** `prefill_forward` in the
process:

```
generator.reset()
ttnn.synchronize_device(mesh)
started = time.perf_counter()
logits = generator.prefill_forward(...)
ttnn.synchronize_device(mesh)
ttft_seconds = time.perf_counter() - started
```

`generator.reset()` releases traces and resets the cache; it does **not**
populate tt-metal's program cache. So the measured interval contains kernel
compilation and program build for every distinct prefill op shape across all 64
layers. Trace capture is excluded — it is timed separately as
`trace_capture_seconds` — but prefill program build is not, because it happens
inside the timed call.

The token-out number in the same file is **not** affected: it replays a captured
trace 128 times, so it is a genuine steady-state measurement. Only TTFT is
contaminated.

## Finding 2: the vLLM TTFT figures are n=1

| stage | recorded | requests |
|---|---|---:|
| 09 vllm_integration | TTFT **P50/P99 4,139/4,139** ms | 1 |
| 10 optimized_vllm | TTFT **P50/P99 3,784/3,784** ms | 1 |

A P50 identical to a P99 to the millisecond is the signature of a single
observation, and both stages state the shape explicitly: "128 input / 128 output,
**1 request**, concurrency 1". The same rows' ITL P50/P99 **do** differ
(55.840/56.850 ms), which is what a real distribution looks like — ITL has 127
samples per request, TTFT has one.

## What this invalidates

1. **Stage 08's precision ranking.** `TTFT_SELECTION.md` on this branch already
   records that stage 08 ranked on traced teacher forcing — which its own README
   calls "the selection metric only", and which differs from token-out by 2.5× —
   and selected the row with the worst TTFT of the ~5.13 s cluster, +502 ms
   (+9.2 %). That note assumed the 502 ms was serving latency. It may not be: a
   precision change alters which matmul program configs get compiled (BFP4 vs
   BFP8 kernels, LoFi vs HiFi2 fidelity variants), so it moves **program-build
   cost directly**. A ~10 % spread between single cold observations is consistent
   with compile-cost differences.
2. **Stage 10's claimed TTFT gain.** "to the same integration baseline it
   improves TTFT by 8.6 %" is 4,139 → 3,784 ms, a difference between two n=1
   cold numbers. The TPOT (12.5 %) and ITL parts of that claim are well
   sampled and unaffected; the TTFT part is not supported by the measurement.

To be explicit about what is *not* claimed: I have not yet measured warm TTFT, so
I am not asserting how large the program-build share is. It is standard tt-metal
behaviour for a first-call op to compile, but the magnitude here is unquantified.
What is established is that **no measurement on this branch can distinguish
serving TTFT from cold-start TTFT**, so no TTFT claim on this branch is currently
supported — including the one in `TTFT_SELECTION.md`.

## The corrected measurement

`full_model_perf_warm.py` runs the identical prefill six times from one weight
load with `generator.reset()` between iterations, so the only thing persisting
across iterations is the program cache — exactly the variable under test. It
reports `ttft_ms_cold` (iteration 0, comparable to the recorded numbers),
`ttft_ms_warm_median`, `ttft_ms_warm_min`, and `cold_overhead_ms`, then runs the
canonical token-out replay with the same call sequence as the stage harness so
that number stays comparable.

It runs across the shipped config and the two candidates
`TTFT_SELECTION.md` names, so it answers both questions at once: how much of the
recorded TTFT is compile cost, and whether the candidates differ on **warm**
TTFT.

Reads: **warm ≈ cold** → prefill really is ~4 s and the ranking question stands
on its own; **warm ≪ cold** → the recorded TTFTs are program-build dominated, the
502 ms ranking spread and the 8.6 % gain both evaporate, and this model's real
serving TTFT is better than anything recorded on the branch.

## Finding 3: the long-prefill lever, separate from the above

`LINEAR_PREFILL_CHUNK_SIZE = 32` (`tt/functional_decoder.py:34`). Per chunk,
`generator.prefill_forward` builds and uploads **five** host tensors — one
sequence mask plus four conv-state lane selectors:

```
for start in range(0, physical_len, LINEAR_PREFILL_CHUNK_SIZE):
    ...
    sequence_mask_tt.append(self._upload(mask, dtype=ttnn.bfloat16))
    for lane in range(4):
        ...
        chunk_selectors.append(self._upload(selector, dtype=ttnn.bfloat16))
```

So host→device uploads before the model runs scale as **5·⌈S/32⌉**. At S=128
that is 20 uploads and cannot explain a 4 s TTFT. At S=32,768 it is **5,120
uploads**, and at the advertised 262,144 context it is **40,960**. The code
comment at `generator.py:211` acknowledges the term
("the O(ceil(S/LINEAR_PREFILL_CHUNK_SIZE)) uploaded metadata") without bounding
its cost.

This is a real long-context TTFT scaling term and it is untested: the longest
prefill measured for latency anywhere on this branch is S=161. Two cheap
mitigations, neither implemented: build the masks and selectors for all chunks as
**one** batched tensor and slice on device, or generate them on device from
`prompt_lens` instead of uploading per chunk. Worth measuring before it matters
to a long-context serving claim.

---

## Correction, same day: the vLLM TTFTs are warm, not compile-dominated

Added 2026-08-17 after reading the plugin. The inference above that the stage
09/10 vLLM TTFTs might be program-build dominated is **wrong**, and the reason is
worth recording because it changes what the number means.

`vllm_tt_plugin/worker.py:compile_or_warm_up_model` calls
`model_runner.warmup_model()`, which is a deliberate **two-phase** warmup
(`model_runner.py:3264`):

> "Phase 1 compiles all op variants (prefill + decode) into the program cache
> WITHOUT capturing any traces. Phase 2 then captures traces with every op
> already compiled, so no new kernel-cache allocations occur that could corrupt
> trace memory."

and it states as an explicit requirement:

> "2. Prefill warmup must cover all supported sequence lengths. If a new sequence
> length appears during inference, its first compilation will allocate new kernel
> cache entries (including reshape caches) that can corrupt active traces."

So prefill programs are compiled before the server accepts requests, by design.
The startup log confirms it ran: `TTModelRunner: trace_mode=all,
sample_on_device_mode=all, enable_model_warmup=True`.

### What survives, and what it now implies

Still correct, and unaffected:

- `tests/full_model_perf.py` TTFT **is** cold. That is a property of the script —
  it times the first `prefill_forward` with no preceding warmup — and it is where
  stage 08's ~5.13 s TTFT column and its 502 ms ranking spread come from. That
  spread remains contaminated by compile cost.
- The vLLM TTFTs are **n=1** (P50 identical to P99). That limitation stands
  regardless of warmth: there is no distribution behind 3,784 ms.

Now different, and more interesting: **~3,784 ms is a real warm prefill.** A
128-token prompt costs ~3.8 s while a decode step costs ~56 ms, so prefilling 128
tokens costs about as much as **68 decode steps** — roughly 30 ms of prefill per
prompt token, against 56 ms per generated token. Prefill should be far more
efficient per token than decode, because it processes the whole prompt in batched
passes rather than one token at a time. It is not.

That makes prefill a genuine optimisation target rather than a measurement
artifact, and it promotes the chunk-size term from a long-context footnote to the
first thing to test. See `PREFILL_CHUNK_LEVER.md`.

The queued cold-vs-warm sweep is still worth running: it quantifies the compile
share on the `full_model_perf` path, and its warm number is the cross-check
against the vLLM 3,784 ms. If warm TTFT there lands near 3.8 s, the two harnesses
agree and prefill is simply slow.
