# Original fused-decoder goal

Use $graph-fusing and $tt-device-usage for the Qwen/Qwen3.6-35B-A3B functional decoder in the current tt-metal checkout.

As the user, I explicitly authorize you to use subagents when a selected skill requests or recommends them.

This goal creates the fused-decoder state for the repo-local TTNN autoport pipeline. Start from the completed functional decoder and work only on models/autoports/<model>/tt/fused_decoder.py plus its tests and docs; do not begin optimized-decoder, multichip-decoder, full-model, or vLLM work in this goal.

Fused decoder completion requirements:
- models/autoports/<model>/tt/fused_decoder.py exists and the delivered tests exercise the fused path, not a functional fallback.
- The fused decoder preserves the functional decoder's prefill/decode semantics, paged KV-cache behavior, determinism, and representative layer-kind coverage.
- The fused decoder preserves the context contract from functional decoder. If fusing optimization changes memory layout, sharding, or KV-cache dtype enough to change capacity, update doc/context_contract.json. Do not complete after reducing advertised capability unless a hard physical device limit, e.g. DRAM, requires it and evidence records the largest feasible value.
- The fused decoder preserves support for valid non-aligned logical sequence lengths; internal padding/chunking is fine, public `seq_len % chunk_size == 0` restrictions are not.
- Prefill and decode PCC remain at the functional acceptance bar for every meaningful layer kind, with any material delta explained.
- Warmed prefill and traced warmed decode latency are reported before and after fusing optimization.
- Final code must beat the best correct traced-decode baseline/candidates; fewer ops or cleaner topology is not enough.
- tt-perf-report tables/CSV exist for prefill and decode.
- The final fused runtime path has no unnecessary torch/from_torch/to_torch, tilize/untilize, reshard, or host fallback in measured prefill/decode.
- Stress or repeated-run coverage and a watcher-clean fused correctness run exist.
- models/autoports/<model>/doc/fused_decoder/README.md and work_log.md record commands, before/after PCC and perf, chosen configs, tt-perf-report conclusions, rejected options, limitations, and exact artifacts.
- Every $graph-fusing pattern is assessed for this model; applicable ones are tried and kept or rejected with before/after evidence, until the $optimize Stop Rule (≥70% roofline and last 2 kept <3%, or 8 h device time); untried items go in the work log as hand-off notes; quality gates are unchanged.
- $stage-review returns `clean-pass` (or `pass-with-carried` after 3 rounds); findings fixed/rereviewed. Local commit stage-owned touched-repo changes; never push; log SHAs.

Unmet requirements, review findings, failed gates: work. Stop only after `$autofix` fails.
