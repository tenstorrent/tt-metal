# AutoFix record

The initial stage implementation was correct but its 62.94 TPOT-derived
tokens/s/user appeared far below the 110.38 tokens/s optimized full-model
number.  AutoDebug first localized steady serving time to the deferred device
completion boundary: 224 steady steps, pending depth two, 13.736 ms median
queue-to-wait time, 14.560 ms event wait, and sub-0.04 ms host extraction.

Experiments:

1. Making the sampled-token read synchronous reduced primary performance to
   57.5 tokens/s/user.  Rejected.
2. Reading one replicated sampled-token shard asynchronously was performance
   neutral, but it is the minimal correct plugin readback and was retained.
3. Reusing immutable scheduler payload objects and skipping steady page-table
   normalization was neutral.  The external-vLLM payload patch was removed;
   page-table reuse was retained because scheduler resets remain authoritative.
4. An exact non-serving harness revealed the comparison error: the 110.38
   number used physical batch 1, while vLLM executes physical batch 32.  With
   the serving shapes, canonical model plus sampling traces measure 14.732 ms
   caller-visible, or 67.88 tokens/s/user.  Final vLLM median ITL is 14.573 ms,
   or 68.62 tokens/s/user.  This closes the actual comparable-work gap.

Final exact-runner sampling, qualitative, non-aligned, primary, and CI gates
all passed.  The final primary result is 62.36 TPOT-derived tokens/s/user and
is reported as flat, not as a performance win.
