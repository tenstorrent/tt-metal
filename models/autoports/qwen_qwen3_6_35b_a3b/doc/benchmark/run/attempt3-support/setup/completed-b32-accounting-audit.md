# Completed b32 accounting audit

**Saved-artifact audit passed.** This covers the completed attempt-3 b32 profile,
not other batch profiles or overall stage completion.

- Raw serving evidence: 96 completed, zero failed; every input length 4096 and
  output length 128; totals 393216 input and 12288 output tokens. Each of 96
  streaming timing rows has 127 inter-token intervals. Server concurrency is 32.
- Independent event reconstruction: 1727 ordered, contiguous events;
  608 complete calls (192 prefill, 416 decode);
  no duplicate or missing starts, accepts or completions. All calls lie within
  the measured client HTTP envelope.
- Replacement ordering is now observed: 190 prefill starts occur after
  the first decode; every one begins with zero open decode intervals. There are
  zero cross-phase overlaps, 282 same-phase overlaps correctly unioned,
  and zero unfinished intervals. This confirms the earlier source-only audit
  for this completed workload.
- All 96 request lifetimes reconcile to contiguous 4096-token prompts across
  192 chunks. Call chunks equal saved chunk/lifetime records. Decode members
  equal saved contiguous position sequences beginning at 4096. The distribution
  of observed decode positions per request is `{127: 96}`. These are executed
  decode steps, distinct from the exactly 128 emitted tokens per request;
  accepted pipeline steps are included in the work accounting.
- Recomputed phase union plus the documented next-phase host-gap assignment
  exactly matches saved intervals and phase seconds: prefill
  8918.901484694, decode 222.369775299; total
  9141.271259993 seconds. The HTTP envelope is 9141.293923221
  seconds; explicitly excluded leading/trailing time is
  0.020714801 /
  0.001948427 seconds. No internal gap is
  lost or counted twice under that convention.
- Work matches the same calls: 2189432639719680 modeled prefill FLOPs, with term and
  fidelity sums matching; 39562640716909.3984 modeled decode DRAM
  bytes, with one accounting row per decode call and matching positions,
  active-member counts and trace key. All row term sums and bounds reconcile.
- SHA-256 checks passed for all 31 detail pieces, raw performance,
  server identity, referenced server configuration, phase evidence, collector
  source and work-accounting source. Recorder instance matches the saved ready
  record. All saved detail row counts match.

Bindings: performance SHA-256 `5dd876868a58a0c82aceecdd5027fa4da79682a796caf8de8e0bd5d03ce7218d`; server identity digest
`06b1030d8546a69ce8900c5a1f24a8dfeebffb779df7df6b14c7445a384b7185`; evidence SHA-256
`eba5cd22b91a65d52b4cb480fd132ef60d8181f6112586b90598df0b17de0356`; accounting source SHA-256
`ab4fc3a30f277af42d916829d7c2e95506ec3a34b5199b03a390df85506c2de6`.

This is a bounded CPU audit of saved JSON pieces (about 6.06 MB
of detail files), using only Python standard-library parsing, arithmetic and
hashes. No large live snapshot, model/HTTP/device call, live flush, ML import or
source edit was used. Work remains a source-derived model: DRAM bytes are not
hardware counters, expert routing/reuse assumptions remain explicit, and this
audit verifies their attachment and arithmetic rather than measuring actual
physical transactions or validating theoretical peaks anew.
