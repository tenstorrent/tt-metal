# Phase review, round 2 (optimization-loop, iterations 1-3), 2026-10-29

Verdict: clean-pass

All three round-1 required items resolved and re-derived from the raw artefacts: the TP=4 gradient check (identically seeded 1-layer model with the loader's rank-interleaved fused rows; PCC 0.99990-0.99992, relative error 1.25-1.42e-2, bias 1e-5 to 1e-7 of the mean magnitude on every adapter for both the `LoraApply` and the transposed + fused TP paths; loss identical); the exit restated at the clocks iteration 3 delivered (1661-1739 ms wall vs the target's 2006-2101 ms, ratio 0.83, 37.3-39.1 % MFU, all re-derived) with the falling-clock anomaly classified; the gap decomposition and ranked-table arithmetic (744 / 491 / 36 / 31 / 37 / 0 = 1339; tail writes 37.1; SDPA 7.6x / 9.6x; threshold 1 % with rung and reason per remaining item; commit SHAs). Other round-1 concerns verified: 16-layer point and 2/4/8/16 fit (byte-identical re-run, 1345.8 ms, residuals ≤ 0.4 %), wasted = 0 by attribution, hot-JSON note, op sequence, isolation script, force-add note.

Other concerns (text, applied after the review): the TP descent-check artefacts predate the interleave fix (relabelled, `.npz` removed); "771 vs 517" big-matmul roofline is 555; the ledger's "tracks the kernel efficiency, not the order of the runs" overclaims (the same code delivered ~1096 vs 1041 MHz an hour apart; confound admitted); nits (README harness count, work-log file names, iteration-1 MFU clock 1104, `--no-transpose` label, RoPE roofline 3.3, a comment on what the `param:` control proves).

Hard-check gaps: the `param:` control only proves the gather round-trip (commented); collective attribution under TP unchanged (not required by the single-chip contract).

Residual risk: the delivered clock, not kernel work, sets the wall-clock gap and the power-vs-thermal question is open; the TP=4 path is gradient-verified at 1 layer, loss-trajectory only at 40; ~105 ms of removable overhead remains listed with rung and reason.
