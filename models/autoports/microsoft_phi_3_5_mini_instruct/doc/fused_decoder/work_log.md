# Fused decoder work log

Date: 2026-07-29 UTC

- Started from functional-decoder stage commit `8adaf288329`.
- Confirmed four local Blackhole p300c boards were visible with
  `timeout 60 tt-smi -ls --local`; no reset or recovery was needed.
- Audited TTNN operations, tests, and model usages for every graph-fusing
  pattern.
- Captured current functional host baselines: prefill b1 1.797 ms, traced
  decode b1 1.050 ms, and traced decode b32 1.218 ms.
- Implemented the fused Binary-NG SwiGLU candidate. PCC was >= 0.999997, but
  five-sample host timing was noisy, so no conclusion was made from op count.
- Implemented and validated paired paged K/V update. At b1 it reduced traced
  decode to about 1.012 ms. At b32 it regressed to about 1.228 ms because the
  fused op requires disjoint input grids and an added V reshard; the b32
  candidate was rejected and the faster two-write path restored.
- Applied the dedicated prefill `experimental.nlp_concat_heads` kernel at
  batch 32. A six-run/100-sample alternating-order rerun exposed a batch-1
  regression, so batch 1 retains the faster generic kernel.
- Final alternating-order A/B (`ab_alternating_final_6x100.log`) produced mean
  functional -> fused latency: prefill b1 1.5900 -> 1.5798 ms, prefill b32
  37.6748 -> 37.3281 ms, traced decode b1 1.0503 -> 1.0126 ms, and traced
  decode b32 1.2161 -> 1.2134 ms. Paired wins were 5/6, 6/6, 6/6, and 5/6.
- Copied the functional suite into a fused-path integration suite by a
  mechanical class replacement, then retained an additional structural test
  that proves the three fused overrides and dedicated calls.
- Final integrated correctness: 26 passed in 69.68 s; exact log
  `fused_decoder_tests_final_v2.log`. PCC/context details are in `README.md`.
- Final watcher-separated run on the selected topology: three passed. Exact
  console: `watcher_final_console.log`; device log:
  `watcher_final/generated/watcher/watcher.log`. No watcher fault was reported.
- First Tracy attempt profiled 20 trace replays and overflowed profiler DRAM
  markers. It is preserved as rejected evidence in `tracy/profile_console.log`
  and `tracy/ops_rejected_overflow.csv`.
- After the first independent review, `tracy/profile_console_final_v2.log`
  overfilled device evidence because prefill was also repeated; the next
  attempt (`v3`) hit a transient profiler marker-pairing assertion during
  device teardown. `tt-smi -ls --local` remained healthy and an immediate
  clean retry succeeded.
- Final Tracy used one prefill measurement and five decode trace replays,
  passed eight A/B cases, and had zero profiler DRAM-buffer-loss warnings.
  Exact console: `tracy/profile_console_final_v4.log`; raw op CSV:
  `tracy/ops_final.csv`.
- Generated signpost-delimited `tt-perf-report` tables, CSVs, summaries, and
  plots for functional/fused prefill/decode at batch 1/32. Device totals and
  host timings are summarized in `README.md`.
- The functional context contract remains unchanged: BF16 cache, page size
  32, prefill/decode maximum 131072, no capability reduction.
- The first independent stage review returned `more-work-needed`: it found
  stale unsupported timing claims and missing adapted RoPE evidence. Timing
  was replaced by the recoverable alternating six-run artifact. The native
  HF-96, padded-HF-128, and llama transformation candidates are now executable
  in `tests/fused_decoder_rope_candidate.py`; `rope_candidate.log` records the
  native width fatal, changed padded midpoint, semantic rejection, and 2-pass
  result.
- Independent stage review remediation:
  - round 1: `more-work-needed` for stale timing evidence and unearned RoPE
    rejection;
  - round 2: `more-work-needed` for one stale 20-replay README sentence;
  - focused final rereview: `clean-pass`, with no required work or hard-check
    gaps.
- Local checkpoint commit SHA is recorded below after commit creation.
