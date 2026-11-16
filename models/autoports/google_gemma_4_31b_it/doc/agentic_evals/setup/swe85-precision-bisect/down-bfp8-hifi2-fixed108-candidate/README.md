# Inert full108 down BFP8/HiFi2 candidate

This directory is a host-only candidate, not admitted and not run. It is a fixed108 full-mode diagnostic using the exact P3 manifest bytes (36 source cases × seeds 42/43/44, same fixture paths, per-row caps and one free trial). The candidate differs from P3 only in `down=bfp8` and `down=HiFi2`; attention, MLP, head, KV, activation/residual/CCL/logits/sampling policies and batch size remain as in P3. This is the same down projection policy tested by the P4 seven-case panel.

Forecast uses P3's measured 2140.193s panel (load 137.960s; 28953 decode tokens in 1017.393s, 28.458 tokens/s) and all 7 observed P4 rows (4461 decode tokens in 166.296s, weighted 26.826 tokens/s). The central same-output-count scenario is 2205.9s. A separate high sensitivity (2511.0s) applies the matched-seven output-count ratio 1.239 and slowest P4 row rate among runs with at least500 decode tokens (25.921 tokens/s). That ratio comes from only seven selected cases and is not an estimate for all108; the range is planning context, not a confidence interval. See `forecast.json` for the measured rows, arithmetic, and caveats.

The command shape reuses P3's `precision_repro_batch` invocation with one model load and a separate output path. It is inert and does not authorize execution.

P4 policy gate note: the existing seven-case panel has generated semantic anomalies according to root review. This full108 candidate is inert contingency planning only; it is not recommended for launch or an admission substitute. Root should decide whether to pursue this policy after completing the seven-case adjudication and considering the next precision arm.
