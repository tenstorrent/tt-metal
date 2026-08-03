# Work log

Date: 2026-07-30 UTC

- Read `optimize`, `tt-device-usage`, `stage-review`, and `autofix`.
- Preserved unrelated dirty GPT-OSS and prompt artifacts.
- Confirmed four healthy Blackhole p300c devices; used one-device tests
  serially.
- Started from reviewed fused checkpoint `ab87d0fee6280402444593bb5d0f437d8bc19782`.
- Added an explicit tensor-group policy and screened BFP8 attention, BFP4
  gate/up, BFP8 down, BF16 cache.
- Real-weight PCC passed for non-aligned prefill 31/33/65 and decode batch
  1/32; repeated decode was deterministic.
- Traced batch-1 decode improved 1.046973 -> 0.791344 ms and batch 32 improved
  1.215818 -> 0.939042 ms.
- Rejected the first Tracy run because device marker buffers overflowed.
  Bounded rerun: `OPT_PROFILE_ITERATIONS=2 python -m tracy -r -p -v -m pytest
  -q -s models/autoports/microsoft_phi_3_5_mini_instruct/tests/optimized_decoder_perf.py`.
- Generated advice-backed B1/B32 reports. They prove the selected weight
  dtypes reached runtime and identify the down matmul and norm/residual stream
  as unfinished must-attack items.
- Separate watcher command:
  `TT_METAL_WATCHER=10 TT_METAL_LOGS_PATH=$PWD/models/autoports/microsoft_phi_3_5_mini_instruct/doc/optimized_decoder/watcher pytest -q -s models/autoports/microsoft_phi_3_5_mini_instruct/tests/test_optimized_decoder.py -k decode_real_reference_and_determinism`.
- The first review occurred before the optimize checklist was complete; its
  findings were remediated before final rereview.
- First independent review returned `more-work-needed` and triggered AutoFix.
- Initial AutoFix selected a 16-shard, `in0_block_w=8`, BFP8/LoFi
  DRAM-sharded down projection. That intermediate checkpoint measured
  0.667609 ms at B1 and 0.830255 ms at B32.
- Sharded norm/residual was refuted. QKV/output/gate-up DRAM-sharded
  candidates won isolated rows but not cumulative whole-layer timing and were
  reverted. Full matrices and PCC evidence are in `AUTOFIX.md`.
- Completion audit added direct paged prefill-to-decode, advertised-context,
  and five-replay trace tests. The shipped BFP4 suite passed all 10 tests;
  minimum PCC was 0.998697 at context 131072.
- Intermediate BFP8 profiler: `tracy/ops_final.csv`, `decode_b1_final.txt`,
  and `decode_b32_final.txt`. Its down row was 57-58 us, BFP8/LoFi, at
  85-87% modeled DRAM utilization.
- The BFP8 checkpoint result was 0.667609 ms B1 and 0.830255 ms B32. It was
  superseded by the later, fully validated BFP4 precision frontier below.
- Independent final stage review: `clean-pass`; see `stage_review.md`.
- Local stage checkpoint on branch `skillexp-cell/fuse-noadvise/phi`:
  `04d269d3601`. No push was performed.
- Completion audit found subclass policy overrides were silently discarded by
  the base constructor. AutoFix repaired propagation and tested real-weight
  BFP4 attention/down separately and cumulatively. Combined BFP4 became the
  default: traced decode is 0.646986 ms B1 and 0.810999 ms B32; warmed prefill
  is 1.395797 ms B1 and 30.266554 ms B32.
- Final watcher-clean ten-test evidence is `watcher_bfp4_final.txt`; final
  runtime rows are in `tracy/decode_bfp4_final.txt`. Prefill large-program
  evidence is in `tracy/prefill_program_config_final.txt`.
## 2026-07-30 AutoFix stage-review closure

- Repaired the precision frontier so all BFP8/BFP4 policies are explicit and
  preserved the successful raw 4-test runner output.
- Re-ran the shipped BFP4/LoFi decode geometry matrix at B1/B32 across 16/32
  cores and legal K-blocks. Promoted the proven down K-block 16 winner.
- Measured explicit prefill K-block 4/8 configurations for all material
  projections. Preserved exact B32 QKV/gate-up L1 blockers and promoted the
  legal explicit down configuration.
- Final correctness: 10 passed. Final performance: decode
  B1/B32 0.642772/0.807652 ms; warmed S128 prefill
  B1/B32 1.351156/24.148464 ms.
- Evidence: `bfp4_precision_frontier_runner.txt`,
  `bfp4_lofi_decode_geometry_runner.txt`,
  `prefill_explicit_config_runner.txt`,
  `correctness_prefill_down_explicit_runner.txt`, and
  `perf_bfp4_block16_prefill_down_runner.txt`.
- Recollected bounded Tracy on the final shipped state. Stage-owned
  `tracy/decode_b{1,32}_bfp4_final.csv` proves block16 decode down at
  47.879/47.992 us and zero host ops; `tracy/prefill_b{1,32}_bfp4_final.csv`
  proves explicit block8 down at 136.433 us (32 cores) / 768.119 us (64 cores)
  and zero host ops.
- Final clean-pass implementation/evidence checkpoint on branch
  `skillexp-cell/fuse-noadvise/phi`: `4f659ed8531`. No push was performed.
