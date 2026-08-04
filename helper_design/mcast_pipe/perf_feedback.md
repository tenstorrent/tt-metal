# `mcast_pipe` performance feedback

This is the intake log for performance risks introduced or exposed by
`mcast_pipe` migrations. An entry records a required comparison; it is not
evidence of a regression until the relevant kernels are profiled against their
pre-migration baseline.

## Status values

- **Open** — the performance comparison has not been completed.
- **No regression** — the comparison found no material regression; preserve the
  measurements and configurations.
- **Regression** — a material regression was measured and needs follow-up.
- **Resolved** — a measured regression was addressed and the fix was verified.

## PERF-001 — Profile GroupNorm's fixed three-rectangle multicast path

- **Date:** 2026-08-04
- **Status:** Open
- **Migration commit:** `09afe010fb3` (`Apply mcast host helpers to sharded groupnorm v2`)
- **Host file:**
  `ttnn/cpp/ttnn/operations/normalization/groupnorm/device/groupnorm_sharded_program_factory.cpp`
- **Sender kernels:** `reader_mcast_sender_unary_sharded_gn_v2.cpp` and
  `welford_reader_mcast_sender_unary_sharded_gn_v2.cpp`

### Performance risk

Before the migration, the sender always constructed and sent the middle
rectangle, but constructed and sent the first and last edge rectangles only
when `has_mcast_first_group` or `has_mcast_last_group` was true.

The migrated factory emits one fixed wire containing three `Mcast2D` blocks for
every group: middle, first edge, and last edge. Missing edge rectangles are
represented as sender-only singleton rectangles. Consequently, both sender
kernels now construct three `SenderPipe`s and call all three `send()` methods on
every reduction iteration:

```cpp
mid_pipe.send(l1_read_addr_ex, l1_read_addr_ex, num_bytes_read);
first_pipe.send(l1_read_addr_ex, l1_read_addr_ex, num_bytes_read);
last_pipe.send(l1_read_addr_ex, l1_read_addr_ex, num_bytes_read);
```

For an absent edge, `SenderPipe::send()` recognizes the singleton as
degenerate, enters the local-copy path, observes `src_l1 == dst_l1`, and returns
without issuing NoC traffic. It is therefore behaviorally a no-op, but it still
adds pipe construction and a call/branch path. The common rectangular-group
case changed from one send call per reduction value to three, with two
degenerate calls. A group with one edge rectangle changed from two calls to
three. A group requiring all three rectangles does not add send calls, and may
benefit from pipe construction having moved out of the hot loop.

This needs a performance check; correctness coverage alone is not sufficient
to close the migration.

### Required comparison

Profile the migration against its parent using identical device, build, input,
and profiler settings. Record at least:

- legacy and Welford GroupNorm v2;
- a rectangular group, where both edge rectangles are absent;
- a wrapped group with one edge rectangle, if a supported configuration reaches
  it;
- a wrapped group with both edge rectangles, if a supported configuration
  reaches it;
- the sender data-movement kernel's `Kernel duration (ns)` and end-to-end
  operation duration.

Start with the exact legacy and Welford 8x4 nodes used to validate the migration,
then select configurations that deliberately cover each rectangle shape. Do not
average together shape classes, because the number of newly introduced
degenerate calls differs between them.

No acceptable regression threshold has been agreed yet. Preserve raw results
and variance so a threshold can be chosen from evidence.

### Follow-up if a regression is measured

Retain the helper-owned coordinate conversion and fixed, chainable argument
layout, but investigate a way for the sender to skip absent rectangle sends.
Possible formulations include per-core rectangle-presence metadata or a helper
operation that makes optional destinations explicit. Choose the formulation
only after measuring where the cost occurs; do not assume that host argument
size, pipe construction, and degenerate calls contribute equally.
