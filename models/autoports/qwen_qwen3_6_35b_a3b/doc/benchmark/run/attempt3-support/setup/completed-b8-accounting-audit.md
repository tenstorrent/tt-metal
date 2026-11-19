# Completed b8 accounting audit

**Saved-artifact audit passed.** This covers the completed attempt-3 b8 profile,
not other batch profiles or overall stage completion.

- Raw serving evidence: 24 completed, zero failed; every input length 4096 and
  output length 128; totals 98304 input and 3072 output tokens. Every streaming
  timing row has 127 inter-token intervals. Configured concurrent requests and
  recorded server capacity both equal 8.
- Independent event reconstruction: 1295 ordered, contiguous events;
  440 complete calls (48 prefill, 392 decode);
  no duplicate or missing starts, accepts or completions. All calls lie within
  the measured client HTTP envelope.
- Replacement ordering: 46 prefill starts follow the first decode,
  each with zero open decode intervals. Zero cross-phase overlaps; 354
  same-phase overlaps correctly unioned; zero unfinished intervals.
- All 24 request lifetimes reconcile to contiguous 4096-token prompts across
  48 chunks. Saved calls, chunks and lifetime records agree. Decode members
  match saved contiguous positions beginning at 4096; position-count distribution
  per request: `{127: 24}`. These executed decode positions are distinct from the
  exactly 128 emitted output tokens per request.
- Reconstructed interval unions and documented next-phase host-gap assignment
  exactly match phase seconds: prefill 2120.819512908, decode
  53.989997583, total 2174.809510491. The HTTP envelope is
  2174.825070190 seconds. Separately recorded leading/trailing time is
  0.013720643 /
  0.001839056 seconds. No internal gap is lost
  or counted twice under the recorded convention.
- Matching work: 547358159929920 modeled prefill FLOPs, with term/fidelity sums agreeing;
  10732133410658.7695 modeled decode DRAM bytes, with one accounting
  row per decode call and matching trace key, positions and active-member count.
  All row term sums and bounds reconcile.
- SHA-256 checks passed for all 21 detail pieces, raw performance, server
  identity, referenced configuration, phase evidence, collector source and
  work-accounting source. Recorder instance matches the ready record; all saved
  detail row counts match.

## Request overlap

Sorting the exact recorded request start and reconstructed completion times
(`start + TTFT + sum(ITL)`, no time buckets; ends precede starts at equal times)
yields maximum overlap **8**, versus upstream
`max_concurrent_requests=10`. The unrounded
request intervals therefore agree with the configured concurrency 8. The
upstream field is retained unchanged; its larger value is not evidence of more
than 8 simultaneously active requests in these recorded intervals. This
comparison uses streamed-last-token completion timestamps, not an independent
measurement of the semaphore or any post-response teardown time.

Bindings: performance SHA-256 `be1ab049f2f72ddc676148829ccba52c5819e43ac08a950244393f6c6d479822`; server identity digest
`9958e8964472ccefbe9d5c226b31d8d054c56d240651f6f40c193f8c46614e97`; evidence SHA-256
`1650739b23c87871d7f1d6bb9862ca52a5ca3a71bb277b47b537b7784ad16a8b`; accounting source SHA-256
`ab4fc3a30f277af42d916829d7c2e95506ec3a34b5199b03a390df85506c2de6`.

This bounded CPU audit read about 3.05 MB of saved detail pieces
with Python standard-library parsing, arithmetic and hashes. No model, HTTP,
device, live snapshot/flush, heavy ML import, source edit or experiment was
performed. DRAM bytes remain modeled logical streaming traffic, not hardware
counters; this verifies arithmetic and evidence attachment rather than actual
physical transactions or theoretical peak specifications anew.

The installed upstream benchmark source confirms inclusive one-second bucket
counting (`vllm/benchmarks/serve.py:673-706`): it floors each relative request
start/end and increments every bucket through the end bucket. Reproducing that
algorithm from saved start/TTFT/ITL values gives **10**, while exact interval
overlap gives **8**. Replacement requests can share a bucket with predecessors
that have already ended. The source uses `output.latency` for its end time;
this reconstruction uses the available streamed-last-token end time. Both yield
the observed peak here. Source SHA-256: `8938b64b273470a3b3fb3b5d8bbd22e8ab4df7640295f55435ac4384283ec589`.
