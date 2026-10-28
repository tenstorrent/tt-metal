# E5c work log (optimized audio tower)

Host sjc-snva-tp100, one P150a, 2026-10-27 19:30-22:00 (PDT). Branch
`gtobarTT/embedding_gemma2_bringup`, start HEAD `182629fea4`. Env: `e1/env.sh` +
`HF_HUB_OFFLINE=1 HF_MODEL=<914f7f89 snapshot>` (`e5c/scripts/env.sh`). One device job at a time;
card-free check before each (`e5c/scripts/bench.sh`, `tracy_case.sh`, `gates.sh` exit when a
process holds `/dev/tenstorrent`).

Harness, frozen for every candidate:
- perf: `tests/perf/bench_audio.py --cases 1x256,1x768,4x256` (`e5c/scripts/bench.sh <label>`), burst
  5 windows x 20 replays after 3 s idle; sustained 5 x 40 after a 10 s soak (baseline and final
  only). Records in `e5c/bench.jsonl`.
- accuracy screen: `e5c/scripts/acc.py --save <tag> --cmp <prev>`: eager `AudioEncoder.encode` on
  s10, s30, b2_s10_s2, short; soft-token PCC / worst-token cosine / hidden PCC vs the HF fp32
  reference (cached in `e5c/ref/`), and max abs diff against the previous kept candidate.
- profile: `e5c/scripts/tracy_case.sh <label> <b1_s768|b4_s256>` (`TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT=4000`:
  one forward is ~1350 programs, more than the default 1000 between flushes; the first "before"
  attempt failed on that, `logs/tracy_before.log`).

Noise: window spread 0.01-0.05 ms at every shape (`bench.jsonl`); a change was kept when its
median beat the previous kept median by more than the window spread and accuracy was identical
(or, for c2 / c15b, not lower in aggregate).

## Timeline (burst medians ms: 1 x 256 / 1 x 768 / 4 x 256)

Times are approximate (log timestamps); the order is exact.

| time | step | result | decision |
|---|---|---|---|
| 19:34 | baseline, E3 code (`9195af74d6`), burst + sustained | 25.775 / 54.284 / 88.338 (sustained 25.771 / 54.285 / 88.328) | reference |
| 19:37 | `mm_micro.py`: batched QK^T / PV / selection matmuls default vs MatmulMultiCoreReuseProgramConfig | 447 -> 65, 232 -> 65, 275 -> 94 us, bit-identical | |
| 19:40 | c1 block-matmul configs | 23.180 / 45.597 / 76.655 | keep (`089d0f1fec`) |
| 19:45 | `bounds.py`: clip bounds of every layer | q/k/v input bounds equal in all 12 layers; no clamp is provably a no-op (silu range vs ffw2 input bounds, v range vs post input bounds) | |
| 19:47 | c2 2D-multicast ClippableLinear configs | 19.140 / 38.555 / 56.972; not bit-identical, soft PCC s10 0.99841 -> 0.99900, s30 0.99905 -> 0.99900, short 0.99853 -> 0.99858 | keep (`572a64a480`) |
| 19:50 | c3 output clamp fused as HARDTANH | 18.231 / 36.347 / 53.365, identical | keep |
| 19:53 | c4 FFN silu + clamp unary_chain | 17.799 / 35.436 / 52.320, identical | keep |
| 19:56 | c5 fused QKV + nlp_create_qkv_heads | 15.606 / 31.681 / 47.572, identical | keep |
| 19:58 | c6 nlp_concat_heads | 14.950 / 30.440 / 45.884, identical | keep |
| 20:00 | c7 softcap as add post-activations | 14.715 / 30.107 / 45.285, identical | keep |
| 20:02 | c8 softmax exp / recip activations | 14.589 / 29.933 / 44.988, identical | keep |
| 20:03 | `relshift_micro.py`: rel shift row-major trick 249 us, gather 126-137 us; 32-wide Q E^T matmul 16 us vs 28-88 us | | |
| 20:05 | c9 gather rel shift | 13.560 / 27.619 / 41.063, identical | keep |
| 20:07 | `dwconv_micro.py`: depthwise conv2d for the lconv | fp32 output: `found_valid_config` fatal; bf16 output 39 us but max abs err inf; L1 input: event sync during trace capture | reject |
| 20:08 | `taps_micro.py`: addcmul vs mul + add (fp32) | 35.9 vs 53.0 us, FMA not bit-identical, needs fp32 slices | not pursued |
| 20:10 | c10 lconv selection bf16 output, pm 1 | 13.284 / 27.073 / 39.974, identical | keep |
| 20:12 | c11 lconv GLU sigmoid pre-activation, silu + clamp chain | 13.162 / 26.767 / 39.369, identical | keep |
| 20:15 | c12 mel uploaded in tile layout | 12.908 / 25.989 / 38.371, identical | keep |
| 20:17 | c13 subsample relu into the mask multiply | 12.853 / 25.898 / 38.263, identical | keep |
| 20:19 | c14 FFN residual scale pre-activation | 12.730 / 25.663 / 37.790, identical | keep |
| 20:22 | c15 input_proj + output_proj + embed 2D configs | 12.650 / 25.510 / 37.449; short soft PCC 0.99858 -> 0.99839 | split |
| 20:24 | c15b output_proj + embed only | 12.663 / 25.540 / 37.531; max abs 1.6e-2, soft PCC equal to 6 digits | keep (input_proj part rejected) |
| 20:27 | `shift_micro.py`: lconv shift block path 145 us vs [4S, S] 2D matmul 82 us / 4 x [S, S] 84 us (768); 70 vs 16 / 35 us (256) | | |
| 20:32 | c16 lconv full shift at batch 1 | 11.981 / 24.351 / 37.534, identical | keep |
| 20:34 | `norm_micro.py`: interleaved RMSNorm 18-19 us at 256-1024 rows (S/32 cores); block-sharded round trip 11.4 (256), 15.8 (768), 19.1 (1024) us | | |
| 20:36 | `shape_micro.py`: eltwise in DRAM at 1024 x 1024 is 2x the 768 x 1024 time (DRAM-bound) | | |
| 20:40 | c17 L1 intermediates: FFN, lconv, residual | 10.512 / 20.223 / 29.157, identical | keep |
| 20:42 | precheck: `pytest tests/pcc/audio` on c18 code | 116 passed | |
| 20:44 | c18 L1 attention intermediates | 9.102 / 16.500 / 21.642, identical | keep |
| 20:47 | c19 L1 masked mel / conv0 input (first try: layer_norm rejects the height-sharded conv output; fixed with an explicit S2I) | 8.819 / 15.661 / 19.996, identical | keep |
| 20:49 | c20 L1 conv flatten + input_proj output | 8.815 / 15.713 / 20.006; not identical, s30 soft PCC 0.99900 -> 0.99833 | reject |
| 20:52 | c21 block-sharded RMSNorm for <= 24 tile rows | 8.511 / 15.358 / 19.991; soft PCC s30 0.99823, b2 0.99874, short 0.99844 (s10 0.99918) | reject (accuracy), E6 |
| 20:54 | c22 fused QKV fp32 output in L1 (96 KB cap) | 8.821 / 15.217 / 19.424, identical | keep |
| 20:55 | c23 FFN wide outputs with the 96 KB cap | 8.821 / 15.217 / 19.420 (no change) | reject |
| 20:57 | c24 conv0 output path in L1 (256 KB cap) | 8.741 / 15.015 / 19.156, identical | keep |
| 20:58 | c25 LoFi for the lconv shift matmuls | not exact: soft PCC s10 0.99834, short 0.99562 | reject (precision), E6 |
| 21:00 | E6 screen: HiFi2 for all audio projections | 8.644 / 13.578 / 17.061; soft PCC s10 0.99889, s30 0.99836, short 0.99556 | not kept, E6 |
| 21:02 | final perf on `fa62290e86` (docstrings only after c24) | 8.739 / 15.021 / 19.154; sustained 8.737 / 15.016 / 19.154 | headline |
| 21:04 | tracy after b1_s768 / b4_s256; gates | see README | |

## Notes

- Reverts of rejected candidates used `git checkout <file>` on uncommitted edits (c20, c21, c23,
  c25, E6 screen); the rejected sharded-norm module is kept at `e5c/scripts/rejected_norm.py`.
- The rel-shift gather index is a load-time uint16 constant; `ttnn.repeat` broadcasts it per
  forward (6 us on 2 cores, from DRAM). The [S, S] shift matrices are created for every token
  bucket at load time, so no allocation happens after trace capture.
- `nlp_create_qkv_heads` and `nlp_concat_heads` accept fp32 (validated in their device ops).
- The bench prints one "Allocating device buffers is potentially unsafe due to the existence of
  an active trace" warning after capture. Its source in the bench was not investigated **[G]**:
  probably the bench's own host round trip (`replay` + `read_rows`) outside the timed windows. The
  trace-allocation gate (tracker enabled, test-suite replay order) is the safety evidence.

## Gates

See README "Gates" (`e5c/logs/gates.log`, `e5c/logs/gate_*.log`).
