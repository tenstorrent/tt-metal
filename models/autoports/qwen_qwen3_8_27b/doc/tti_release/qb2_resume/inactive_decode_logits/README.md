# Define inactive decode logits before sampling

Sparse decode leaves skipped SDPA rows unwritten, but the generator sampled their logits. Serial and grouped B2 runs therefore produced different inactive token feedback despite identical active outputs. The five-line repair gives inactive logical rows finite zero logits before existing batch padding and native sampling. Active row order, dense decode, seeds, history and state updates remain unchanged.

The unchanged reduced FP8 layers0/3 retry passed all 768 comparisons exactly (PCC 1, maximum absolute difference 0), exact all 32 token/seed/history metadata, both slot orders and sampling modes, state/KV canaries, history and device-table rebind. Its 24 warmups establish unchanged active logits and finite-zero inactive sampler inputs; 32 replay snapshots and 48 guarded native captures pass. Actual child/wrapper/observer exits are 0, actual mapped native libraries are pinned, 704 passive ownership samples contain 0 errors, and all four devices close cleanly. The 100 ms observer does not exclude shorter events. Original ownership-interfered evidence remains rejected.

Independent source and runtime reviews approve only this reduced repair. Seven candidate host tests, five retry host tests and the post-promotion APC 18/fairness 27 host tests pass. Applied-source verification changes exactly one of 69 serving hashes and preserves benchmark policy. Python-only change; no C++ or CMake build is required.

`summary.json` binds the preserved external evidence, and `REVIEW.md` contains the independent acceptance review. The immutable retry used the candidate method in isolation before exact source promotion; old pre-promotion source pins must not be rewritten to the promoted hash.

This checkpoint establishes no full-model, serving, APC/fairness metric, throughput or Stage11 release pass. Recurring unknown external device openers remain a blocker for sustained measurements despite the host telemetry guard. Cold performance remains 12 failures/0 CSV passes, and the 262144+252 row remains unresolved.
