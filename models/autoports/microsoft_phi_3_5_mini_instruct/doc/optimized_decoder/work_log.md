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
- AutoFix selected a 16-shard, `in0_block_w=8`, BFP8/LoFi DRAM-sharded down
  projection. Final warning-free traced decode is 0.667609 ms at B1 and
  0.830255 ms at B32.
- Sharded norm/residual was refuted. QKV/output/gate-up DRAM-sharded
  candidates won isolated rows but not cumulative whole-layer timing and were
  reverted. Full matrices and PCC evidence are in `AUTOFIX.md`.
- Final aggregate correctness: `correctness_final.log`, 6 passed.
- Final profiler: `tracy/ops_final.csv`, `decode_b1_final.txt`, and
  `decode_b32_final.txt`. The shipped-default down row is 57-58 us,
  BFP8/LoFi, at 85-87% modeled DRAM utilization.
- Final shipped-default result is 0.667609 ms B1 and 0.830255 ms B32. Faster
  numbers in projection screening logs are rejected experimental composites,
  not the shipped default.
- Independent final stage review: `clean-pass`; see `stage_review.md`.
- Local stage checkpoint on branch `skillexp-cell/fuse-noadvise/phi`:
  `04d269d3601`. No push was performed.
