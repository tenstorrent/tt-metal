# Prefill trace capacity diagnosis and candidate

Source analysis, 2026-11-14. Hardware failure is
[batch32_hf_v1.json](batch32_hf_v1.json); this candidate is not yet hardware accepted.

The first B32 prefill call failed in `_record_trace -> end_trace_capture`,
allocating a 143,278,080-byte trace command buffer per rank. Prepared free DRAM
was 4,169,844,224 bytes/rank; allocated growth before failure was
4,001,431,552 bytes/rank. Available contiguous bytes per bank were 17,688,000,
below the requested 17,915,904. This is command-buffer retention pressure, not
an evidence-backed context or KV dtype limit.

The original `(count, slot)` key retains one whole-stack trace for each logical
chunk length and slot. Prefix lengths203..234 require32 full-chunk keys and31
tail keys. Multiplying63 by the failed trace's size gives9,026,519,040 bytes;
actual tails vary, so this is an illustrative size estimate, not measured total.
Every key remained until close. Later decode geometry signatures are a separate
potential retention pressure; this change does not claim to fix those.

An LRU eviction workaround was rejected before implementation. Installed
`tt-enable-tracing/SKILL.md`, Trace Lifetime And Reuse, requires: “Supported
requests must not invalidate, retire, evict, or recapture traces.” Synchronizing
before release would prevent an in-flight use-after-free but would not satisfy
that persistent-lifetime contract.

The candidate reuses a canonical prefill cache binding: B1 uses its own cache;
B>1 uses the existing migration scratch. At each submitted row, device copies
stage its private recurrent/index/PLE state and page tables into scratch and
copy them back after all its chunks. Shared KV tensors stay in the shared pool;
page tables select that row's pages. Private-KV mode copies its KV state too.
No cache/token state travels through the host and no per-chunk state migration
is needed. Canonical output aliases an existing persistent tensor; each final
row output is copied once to its existing slot output. Full-logit mode reads
and slices every logical chunk before the next replay overwrites that output.

Public logical intervals are covered by buckets32/16/8/4/2/1 without crossing
the existing absolute32 grid. No padding advances state. Count alone selects a
trace, independent of slot or request. This bounds prefill to six captures over
the generator lifetime, preserving every captured trace until close. Preparation and persistent input allocation cover only these six physical
buckets. Standalone synthetic diagnostics must warm their own variants before
capture; an explicit-slot `_prefill_body` remains available for that purpose.

CPU validation:

```bash
python_env/bin/python -m unittest models.autoports.qwen_qwen3_8_flash_next.tests.test_prefill_validation_cpu -v
```

10 tests pass: malformed inputs precede mutation; mixed slots and sampled first
tokens preserve order; nonaligned continuation all-logit slices and padding are
correct;6,144 phase/length pairs cover precisely their logical interval within
six sizes and absolute32 boundaries; B1 performs no state copy; B>1 copies state
once each direction per row;384 calls across32 slots/two requests capture only
six signatures and reuse the same IDs. These are orchestration tests with CPU
stubs, not numerical TT or allocation-tracker evidence.

Costs requiring hardware adjudication: dyadic tails can issue up to five chunks
instead of one arbitrary-size tail, and B>1 copies private state at row
boundaries. Dtypes, fidelities, expert arithmetic, inter-layer residual layout,
KV ownership and CCL kernels are unchanged. Compare original viable B1/B2
prefill timings and full outputs, run reduced tracked mixed/resume checks, then
retry B32. Any latency regression remains optimization work; this record does
not waive the user's preserved-optimization requirement. Original generator
SHA256 `8da9d84e637f2e7db1bc210cc1f050a7afd6efdfd95095cbf94d46c1ee4a6233`
is archived under `tests/diagnostic_sources/prefill_trace_v1/generator.py` for
separate-process baseline measurements only.
