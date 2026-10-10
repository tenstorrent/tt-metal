# Batched prefill attention prototype, October 10, 2026

`QWEN_PREFILL_BATCHED_ATTENTION=1` selects an experimental TP4 prefill path;
the default stays off. B16 cache-fill/attention calls decrease from 32+16 to
2+1, eliminating per-user slicing and final attention concatenation. BFP8 KV,
BF16 Q/K/V and existing native attention settings are preserved. Local page
table row indices are allocated at model setup and shared within one replica,
before any trace reserves scratch. Non-page-aligned continuations retain the
existing single-token decode route.

Engineering target: 1.25–1.5x at the cache-fill/attention boundary. If this
boundary occupied 25% of prefill, that would mean roughly 5–8% less TTFT.
The stage fraction and speedup are unmeasured; isolated decode TSU does not
improve from this change. No overall throughput gain is credited.

Host preflight passed **556 tests, 69 subtests, one skip**. Added tests cover
nonzero request slots, shuffled physical pages, carried prefixes, logical
tails, causal masking and unchanged inactive pages. The physical comparison
is collected but has not run. It requires before/batched/after timing, exact
full-cache equality against expected writes, exact all-rank output equality,
and first/middle/last query checks against a dense full-prefix reference for
every user on every rank. Cases include B16 chunks at 16K/32K and B32/32K.

The initial persistent unit `qwen38-prefill-attention-v1-20261010.service`
(invocation `9df6962d09c44beabf1037326bb33f2e`) was queued behind projection v4.
The compact predecessor subsequently failed with an L1 buffer collision in
the vocabulary-head matmul during prefill. All followers, including this one,
terminated before starting hardware, as required by the dependency checks.
This is not a failed batched-attention hardware result. Recovery must use a
new run directory and clean audited predecessor; the failed receipts remain.

Full-model prefill/decode comparisons, GPQA and serving promotion remain
required after a successful boundary test. Current qualified decode remains
16.55 TSU with GPQA 177/198. The compact candidate still has no full-model
timing result. Native installation and active benchmark sources were unchanged.
