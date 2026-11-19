# Completed b1 accounting audit

**Saved-artifact audit passed.** This covers the completed attempt-3 b1 profile,
not an overall stage verdict.

- Raw serving evidence: 8 completed, zero failed; every input length 4096 and
  output length 128; totals 32768 input and 1024 output tokens. Every streaming
  timing row has 127 inter-token intervals. Configured concurrent requests and
  recorded server capacity both equal 1. Raw duration: 693.218901627 s.
- Independent event reconstruction: 3087 ordered, contiguous events;
  1032 complete calls (16 prefill, 1016 decode);
  no duplicate or missing starts, accepts or completions. All calls lie within
  the measured client HTTP envelope.
- Replacement ordering: 14 prefill starts follow the first decode,
  each with zero open decode intervals. Zero cross-phase overlaps; 1000
  same-phase overlaps correctly unioned; zero unfinished intervals.
- All 8 request lifetimes reconcile to contiguous 4096-token prompts across
  16 chunks. Saved calls, chunks and lifetime records agree. Decode members
  match contiguous positions beginning at 4096; position-count distribution per
  request: `{127: 8}`. These executed decode positions are distinct from the
  exactly 128 emitted output tokens per request.
- Reconstructed interval unions and documented next-phase host-gap assignment
  exactly match phase seconds: prefill 671.012540608, decode
  22.195170609, total 693.207711217. The HTTP envelope is
  693.218495650 seconds. Separately recorded leading/trailing time is
  0.010377845 /
  0.000406588 seconds. No internal gap is lost
  or counted twice under the recorded convention.
- Matching work: 182452719976640 modeled prefill FLOPs, with term/fidelity sums agreeing;
  4756008036352.0000 modeled decode DRAM bytes, with one accounting
  row per decode call and matching trace key, positions and active-member count.
  All row term sums and bounds reconcile.
- SHA-256 checks passed for all 37 detail pieces, raw performance, server
  identity, referenced configuration, phase evidence, collector source and
  work-accounting source. Recorder instance matches the ready record; all saved
  detail row counts match.

## Request overlap

Sorting exact request start and reconstructed completion times
(`start + TTFT + sum(ITL)`, ends before starts at equal times) yields maximum
concurrent overlap **1**. Reproducing inclusive one-second request buckets yields
**2**, matching the unchanged upstream `max_concurrent_requests=2` field.
Replacement requests share a bucket with already-ended predecessors; the bucket
peak is not evidence of two simultaneously active requests.

The upstream algorithm was inspected in the preceding b8 audit
(`vllm/benchmarks/serve.py:673-706`). It uses `output.latency` for bucket ends;
this reconstruction uses available streamed-last-token end times and reproduces
the observed peak. It does not independently measure semaphore occupancy or
post-response teardown.

Bindings: performance SHA-256 `6dba58b3bdddd877fd3ff97ea49598aae53598feff52c65c6294f08bfe6c85af`; server identity digest
`3387530a0d43321df943396b34702c24b884b3230bc12c17039a5fe4d9ea0d7f`; evidence SHA-256
`4af3c9573a1b1592d1a38eb2c2f3d614443a4be261c301f576ae65eebd8a4da8`; accounting source SHA-256
`ab4fc3a30f277af42d916829d7c2e95506ec3a34b5199b03a390df85506c2de6`.

This bounded CPU audit read about 5.70 MB of saved detail pieces
with Python standard-library parsing, arithmetic and hashes. No model, HTTP,
device, live snapshot/flush, heavy ML import, source edit or raw mutation was
performed. DRAM bytes remain modeled logical streaming traffic, not hardware
counters; this verifies arithmetic and evidence attachment rather than actual
physical transactions or theoretical peak specifications anew.
