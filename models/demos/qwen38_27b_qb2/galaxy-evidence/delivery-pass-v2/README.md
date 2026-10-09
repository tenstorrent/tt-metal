# DRAM delivery diagnostic: correctness passes, redistribution costs remain

Oct 8 17:36 UTC: all six variants passed byte validation, alternate allocations,
delayed consumers and three trace replays on four physical ranks. Large timing
volumes check packet markers and aggregate order; they do not compare every
payload byte. Hardware test: 31.88 seconds; CPU: 355 tests plus 40 subtests.

| MiB/chip/call | Raw read GB/s | Best delivered GB/s |
|---:|---:|---:|
| 272 | 499.23 | 245.08 |
| 544 | 505.06 | 247.52 |
| 1088 | 507.90 | 248.81 |

The best delivery variant uses nearby consumers, 15-page packets and depth four.
The observed delivered rate is about half the read-only rate, with dispatch and
receiver work included. The complete attention path already achieves higher
useful-KV bytes/time; this diagnostic provides no model-speedup justification.
The next 54-variant sweep tests packet aggregation, buffering, placement and
backpressure before redesigning or integrating the mover. No attention math,
arbitrary production page table, or new KV layout is implemented here.

v1 stopped before device work because the transient unit PATH omitted
`/usr/local/bin/tt-smi`. v2 includes the existing executable directory and passed
after the authorized reset. No tool installation or firmware change was needed.
