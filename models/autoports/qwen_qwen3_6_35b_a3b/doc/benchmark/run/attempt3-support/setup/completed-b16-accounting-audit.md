# Completed b16 accounting audit

**Saved-artifact audit passed.** This covers the completed attempt-3 b16 profile,
not other batch profiles or overall stage completion. Files reside under
`run/attempt3/`; no canonical-root b16 file was present at inspection.

- Raw serving evidence: 48 completed, zero failed; every input length 4096 and
  output length 128; totals 196608 input and 6144 output tokens. Every streaming
  timing row has 127 inter-token intervals. Concurrent requests and server
  capacity both equal 16.
- Independent event reconstruction: 1439 ordered, contiguous events;
  496 complete calls (96 prefill, 400 decode);
  no duplicate or missing starts, accepts or completions. All calls lie within
  the measured client HTTP envelope.
- Observed replacement ordering: 94 prefill starts follow the first
  decode, and each begins with zero open decode intervals. Zero cross-phase
  overlaps; 330 same-phase overlaps correctly unioned; zero unfinished
  intervals at the measured cohort's end.
- All 48 request lifetimes reconcile to contiguous 4096-token prompts across
  96 chunks. Saved calls, chunks and lifetime records agree. Decode members
  match saved contiguous position sequences beginning at 4096; position-count
  distribution per request: `{127: 48}`. These are executed decode positions,
  distinguished from the exactly 128 emitted output tokens per request.
- Reconstructed interval unions and documented next-phase host-gap assignment
  exactly match saved phase seconds: prefill 4283.864746724, decode
  106.186722298, total 4390.051469022. The HTTP envelope is
  4390.069686688 seconds; separately recorded leading/trailing time is
  0.016510244 /
  0.001707422 seconds. No internal gap is lost
  or counted twice under the recorded convention.
- Matching work: 1094716319859840 modeled prefill FLOPs; term and fidelity sums agree.
  Decode totals 20518017126111.2070 modeled DRAM bytes; every call has
  one accounting row with matching trace key, positions and active-member count.
  Row term sums and bounds reconcile.
- SHA-256 checks passed for all 24 detail pieces, raw performance, server
  identity, referenced configuration, phase evidence, collector source and
  work-accounting source. Recorder instance matches the ready record. Saved
  detail row counts match parsed records.

Bindings: performance SHA-256 `a635de429387fa6e6116a0901864525b5806ff0f57ddc3912d486d04a99f7dbd`; server identity digest
`02603e8050be8b9680901acb98982d77d9ed9891a5b4d9f071abf7a1c266e722`; evidence SHA-256
`eac9706c462a3a7eb46d0b38a1930ed4b80b28ebb027249414dc4776fa10e64b`; accounting source SHA-256
`ab4fc3a30f277af42d916829d7c2e95506ec3a34b5199b03a390df85506c2de6`.

This bounded CPU audit read about 4.04 MB of saved detail pieces
with Python standard-library parsing, arithmetic and hashes. No model, HTTP,
device, live snapshot/flush, hook, heavy ML import, source edit or commit was
performed. DRAM bytes remain modeled logical streaming traffic, not hardware
counters; this audit checks arithmetic and evidence attachment without measuring
actual physical transactions or independently revalidating theoretical peaks.
