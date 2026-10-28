# E5c: optimized audio tower (stage report)

Scope: the EmbeddingGemma 2 audio path (`tt/audio/`: conv subsampling, 12 conformer layers,
`output_proj`, `embed_audio`) on one Blackhole P150a (13x10 grid), sjc-snva-tp100, 2026-10-27.
The functional baseline is the E3 audio code, unchanged through E4 / E5a / E5b (`tt/audio` at
`182629fea4`). Text and vision paths are unchanged: no shared file (`tt/linear.py`,
`tt/rms_norm.py`, `tt/config.py`, `tt/weights.py`) was edited. The audio tower has its own helpers
in `tt/audio/` (`linear.py`, `program_configs.py`, the L1 rule in `config.py`).

Labels: **[M]** measured (command and log named), **[I]** inferred from code or data, **[G]** guess.
Logs, sweeps, micro-benches and raw data: `tt-ops/runs/embeddinggemma-2/e5c/` (`bench.jsonl`,
`results_*.jsonl`, `logs/`, `scripts/`, `tracy/`).

## Result

Traced device time per forward (`tests/perf/bench_audio.py`: inputs resident, `execute_trace`
back to back, one sync per window). Burst = median of 5 windows of 20 replays, each after 3 s idle.
Sustained = median of 5 windows of 40 replays after a 10 s soak. Inputs: real in-tree speech
(`audio_common.load_case`: s10 = 10 s, s30 = 30 s, 4 x s10).

| batch x bucket | E3 code today burst / sustained ms | E3 published ms | E5c burst ms [window min, max] | E5c sustained ms | speedup vs today / vs published |
|---|---|---|---|---|---|
| 1 x 256 (10 s) | 25.775 / 25.771 | 23.67 | **8.739** [8.738, 8.740] | 8.737 | 2.95x / 2.71x |
| 1 x 768 (30 s) | 54.284 / 54.285 | 47.79 | **15.021** [15.012, 15.023] | 15.016 | 3.61x / 3.18x |
| 4 x 256 (4 x 10 s) | 88.338 / 88.328 | 80.48 | **19.154** [19.147, 19.159] | 19.154 | 4.61x / 4.20x |

**[M]** `e5c/bench.jsonl` labels `baseline_E3` (HEAD `9195af74d6`, the E3/E4 audio code,
`logs/bench_baseline.log`) and `final_E5c` (HEAD `fa62290e86`, the final code,
`logs/final_perf.log`). Window spread is under 0.1 % in every row, and sustained equals burst
within 0.01 ms: the card did not throttle. Per clip, 4 x 10 s is now 4.79 ms / clip against
8.74 ms for one 10 s clip.

The E3 code re-measures slower today than in the E3 report (+2.1 / +6.5 / +7.9 ms). Tracy of the
E3 code today at 1 x 768: 53,940 us device time over the same 1357 ops, against 47,482 us in the E3
report; matmuls are 25,702 us vs 19,257 us, the rest within a few percent **[M]**
`tracy/before/b1_s768` vs `doc/functional_audio/tracy/b1_s768`. It is the drift E5b saw on the E2
vision code (default-config HiFi4 bf16 matmuls slower than on the day of the functional stage). The
cause is unexplained. Every before / after number here is from the same card on the same day.

## What changed (cumulative, in order)

All rows are bit-identical to the previous kept row (soft tokens and tower output, max abs diff 0 on
s10, s30, b2_s10_s2 and short, `e5c/scripts/acc.py`) unless the "accuracy" column says otherwise.
Burst medians, ms. Full ledger with rejects: `work_log.md`.

| # | change | 1 x 256 | 1 x 768 | 4 x 256 | accuracy |
|---|---|---|---|---|---|
| — | E3 code (today) | 25.775 | 54.284 | 88.338 | |
| c1 | block-matmul program configs for QK^T, PV and the lconv selection matmul (`program_configs.py`): they ran on 2-4 cores | 23.180 | 45.597 | 76.655 | identical |
| c2 | 2D-multicast program configs for every ClippableLinear (`linear.py`; bf16 weights, HiFi4 + fp32 acc unchanged) | 19.140 | 38.555 | 56.972 | not identical (accumulation order): soft PCC s10 0.99841 -> 0.99900, s30 0.99905 -> 0.99900 |
| c3 | output clamp fused into the matmul (HARDTANH) | 18.231 | 36.347 | 53.365 | identical |
| c4 | FFN: silu + ffw2 input clamp as one `unary_chain` | 17.799 | 35.436 | 52.320 | identical |
| c5 | fused QKV (one input clamp: q/k/v input bounds are equal in all 12 layers) + `nlp_create_qkv_heads`; output clamps as pre-activations of the scale multiplies | 15.606 | 31.681 | 47.572 | identical |
| c6 | head merge with `nlp_concat_heads` | 14.950 | 30.440 | 45.884 | identical |
| c7 | logit softcap as post-activations of the logits add | 14.715 | 30.107 | 45.285 | identical |
| c8 | softmax: exp / reciprocal as activations of subtract / multiply (hand-written fp32 softmax kept) | 14.589 | 29.933 | 44.988 | identical |
| c9 | rel shift: 32-wide Q E^T (column 12 - d) + one `ttnn.gather` instead of the row-major reshape / slice trick | 13.560 | 27.619 | 41.063 | identical |
| c10 | lconv selection matmul output bf16 (exact), one block per core | 13.284 | 27.073 | 39.974 | identical |
| c11 | lconv GLU sigmoid as multiply pre-activation; silu + linear_end input clamp as one chain | 13.162 | 26.767 | 39.369 | identical |
| c12 | mel uploaded in tile layout [B,1,T,128] (drops a 0.75 ms row-major reshape at 1 x 768) | 12.908 | 25.989 | 38.371 | identical |
| c13 | subsample relu as pre-activation of the layer-1 mask multiply | 12.853 | 25.898 | 38.263 | identical |
| c14 | FFN 0.5 residual scale as pre-activation of the residual add | 12.730 | 25.663 | 37.790 | identical |
| c15b | 2D configs for output_proj and embed_audio | 12.663 | 25.540 | 37.531 | not identical: max abs 1.6e-2, soft PCC equal to 6 digits |
| c16 | lconv causal shift at batch 1: four [S, S] 0/1 shift matmuls per bucket (7.5 MB shared constants) on the 2D config; batch > 1 keeps the block path | 11.981 | 24.351 | 37.534 | identical |
| c17 | L1 intermediates (FFN, lconv, residual stream): `audio_act_memory_config`, <= 64 KB per bank | 10.512 | 20.223 | 29.157 | identical |
| c18 | L1 intermediates in the attention | 9.102 | 16.500 | 21.642 | identical |
| c19 | masked mel and conv0 input in L1 | 8.819 | 15.661 | 19.996 | identical |
| c22 | fused QKV fp32 output in L1 up to 96 KB per bank | 8.821 | 15.217 | 19.424 | identical |
| c24 | conv0 output path (S2I, LayerNorm, ReLU * mask) in L1 up to 256 KB per bank | 8.741 | 15.015 | 19.156 | identical |
| final | same code, final run (`fa62290e86`, docstrings only since c24) | 8.739 | 15.021 | 19.154 | |

Two kept rows change values: c2 and c15b. Both keep dtypes and math fidelity (bf16 weights, HiFi4,
fp32 accumulation); only the K blocking and core split of the matmuls change. They move the
soft-token PCC by noise-level amounts in both directions (Accuracy below).

## Rejected or measured for E6

| candidate | result | decision |
|---|---|---|
| c20: conv-stack flatten + input_proj output in L1 | 15.661 -> 15.713 ms at 1 x 768; not identical (the default `ttnn.linear` picks another config for an L1 input): s30 soft PCC 0.99900 -> 0.99833 | rejected |
| c15 (input_proj on the 2D config) | -0.03 ms; short-clip soft PCC 0.99858 -> 0.99839 | rejected (kept output_proj / embed only, c15b) |
| c21: block-sharded RMSNorm round trip for <= 24 tile rows (`norm_micro.py`: 11.4 vs 18.1 us at 256 rows) | -0.31 / -0.30 / 0 ms; soft PCC s30 0.99900 -> 0.99823, b2 clip 0 0.99900 -> 0.99874, short 0.99858 -> 0.99844, s10 0.99900 -> 0.99918 | rejected (lowers accuracy); for E6 |
| c23: FFN 4096-wide outputs with the 96 KB cap | no change (already in L1 at every benched shape) | rejected |
| c25: LoFi for the 0/1 lconv shift matmuls | not exact (h is the truncated operand): soft PCC s10 0.99834, short 0.99562 | rejected (precision); for E6 |
| E6 screen: HiFi2 for every audio projection (ClippableLinear + fused QKV) | 8.644 / 13.578 / 17.061 ms (-0.10 / -1.44 / -2.09 ms); soft PCC s10 0.99889, s30 0.99836, short 0.99556 | not kept (precision); for E6 |
| depthwise `ttnn.conv2d` (groups = 1024, k = (1, 5)) for the lconv | 39 us vs 145 us micro, but fp32 output fails (`op_slicing.cpp:266 found_valid_config`), bf16 output is a precision drop and returned garbage (max abs err inf) in the micro; with an L1 input the conv issues an event sync inside trace capture | rejected (`dwconv_micro.py`) |
| `ttnn.addcmul` for the lconv tap accumulation | 35.9 vs 53.0 us per tap in fp32 (micro), but needs fp32 slices (bf16 slices are cheaper) and is not bit-identical (fused multiply-add) | not tried in the model; net gain ~4 us per tap **[I]** |

The micro-benches are in `e5c/scripts/*_micro.py` with logs in `e5c/logs/*_micro*.log`.

## Op-topology audit (one conformer layer)

Ops per layer, before (E3) -> after, from the tt-perf-report rows **[M]**:

| block | E3 | E5c |
|---|---|---|
| FFN (x2) | norm, clamp, matmul, clamp, silu, clamp, matmul, clamp, norm, mul, add (11) | norm, clamp, matmul+clamp, chain(silu, clamp), matmul+clamp, norm, add(pre-act scale) (7) |
| attention projections | 3 x (clamp, matmul, clamp) + 2 scale muls + 3 x (reshape, transpose) (17) | clamp, fused QKV matmul, create_heads, mul(pre-act clamp) x 2, clamp (6) |
| relative position | matmul (24 cores) + untilize, reshape, slice, reshape, slice, tilize (7) | block matmul, repeat, gather (3) |
| QK^T / PV | 2 batched matmuls on 2 and 4 cores | same matmuls on 130 cores |
| softcap + softmax | add, mul, tanh, mul, add, max, sub, exp, sum, recip, mul (11) | add(+softcap), add(mask), max, sub(+exp), sum, mul(+recip) (6) |
| merge + post | transpose, reshape, clamp, matmul, clamp (5) | concat_heads, clamp, matmul+clamp (3) |
| lconv | norm, clamp, matmul, clamp, 2 slices, sigmoid, mul, block context (5), selection matmul (4 cores), mul, 4 x (slice, mul, add), norm, silu, clamp, matmul, clamp, add (33) | norm, clamp, matmul+clamp, 2 slices, mul(+sigmoid), mul, 4 x (shift matmul, mul, add), norm, chain, matmul+clamp, add (23) at batch 1 |

Per forward at 1 x 768: 1357 -> 874 device ops. The remaining reshape / slice / concat work is
mostly the two block contexts of the attention (about 28 us each per layer, in L1) and the GLU
slices.

## L1 placement

`tt/audio/config.py::audio_act_memory_config`: an intermediate goes to L1 interleaved when it
needs at most 64 KB per bank (the E5a text rule; 130 banks), else DRAM. At 1 x 768 that is every
[S, 1024] tensor (12 KB per bank), the [S, 4096] FFN intermediates (48 KB), q / k / v heads and
the block contexts (24-48 KB), the logits and softmax (12 KB). Two named exceptions get a wider
cap: the fp32 fused QKV output (72 KB per bank at 1 x 768, 96 KB cap) and the conv0 output path in
the subsampling (193 KB per bank at 1 x 768, 256 KB cap). Persistent inputs (mel, masks), the
trace outputs (hidden, soft tokens), weights and constants stay in DRAM. `EG2_AUDIO_L1_BYTES_PER_BANK=0`
puts every intermediate back in DRAM (A/B switch). The eltwise ops were DRAM-bound: a bf16
clamp on [1024, 1024] took 19.7 us in DRAM against 10.0 us for [768, 1024] (`shape_micro.py`),
and the whole change (c17 + c18 + c19 + c22 + c24) took 1 x 768 from 24.35 to 15.02 ms.

## Matmul configs

- Batched block matmuls (QK^T [B, 8*nb, 32, 128] x [B, 8*nb, 64, 128]^T, PV, the rel-pos Q E^T,
  the batch > 1 lconv selection): `MatmulMultiCoreReuseProgramConfig` with one 32-row block per
  work unit over the full grid (`program_configs.block_bmm_config`). Micro at 1 x 768 **[M]**
  `mm_micro_768.log`: QK^T 447 -> 65 us, PV 232 -> 65 us, selection 275 -> 94 us, all
  bit-identical; in0_block_w 4 beat 2 and 1 (65 vs 88 / 89 us).
- Projections: `MatmulMultiCoreReuseMultiCastProgramConfig` on the full grid, per_core_M / N from
  the shape, in0_block_w up to 8 under a 900 KB circular-buffer budget, fp32-destination subblocks
  of at most 4 tiles, the ClippableLinear output clamp as the fused activation. In the final
  profile the FFN matmuls run at 72-80 % of the HiFi4 FLOP roofline (768 x 1024 x 4096: 62 us;
  768 x 4096 x 1024: 66 us) **[M]** `tracy/after/b1_s768`.
- Precision policy unchanged from E3: bf16 weights, HiFi4, fp32 accumulation for every matmul;
  q / k / v / logits / softmax / depthwise taps in fp32.

## Accuracy

The gates pass on the final code (Gates). Soft tokens vs the HF fp32 reference, per clip, E3 code
(`e3/final_results.jsonl`) and E5c (`e5c/results_gates.jsonl`), with HF's own bf16 run for
context **[M]** (`e5c/scripts/audio_acc_cmp.py`):

| clip | soft-token PCC E3 / E5c / HF bf16 | worst-token cosine E3 / E5c |
|---|---|---|
| s10 (10 s; also b2_s10_s2 clip 0) | 0.99841 / 0.99900 / 0.99919 | 0.764 / 0.834 (token 126) |
| s30 (30 s, 750 tokens) | 0.99905 / 0.99900 / 0.99924 | 0.773 / 0.773 (token 267) |
| s2 (also b2_s10_s2 clip 1) | 0.99990 / 0.99990 / 0.99992 | 0.9994 / 0.9996 |
| short (0.64 s, 16 tokens) | 0.99853 / 0.99858 / 0.99847 | 0.982 / 0.982 |
| tone3 (synthetic) | 0.99806 / 0.99769 / 0.99844 | 0.975 / 0.967 |

Full-model embedding cosine vs HF fp32 for the audio items (`test_full_model.py`, E5b gates
`e5b/results_review_fix.jsonl` -> E5c gates): speech 2 s 0.99975 -> 0.99976, speech 10 s
0.99980 -> 0.99979, tone 3 s 0.99926 -> 0.99895, speech 2 s + text 0.99974 -> 0.99975, speech
10 s + text 0.99977 -> 0.99976, image + audio 0.99953 -> 0.99943 (bar 0.99) **[M]**.

Per-module PCC (teacher-forced FFN / attention / lconv / layer / subsample / heads) is unchanged
at the printed precision: per-layer minimum 0.999986 (short, layer 2), attention 0.999998, lconv
0.999996, conv stack 0.999700 (crop13), all against a bar of 0.995. The two value-changing keeps
(c2, c15b) move the per-clip soft-token PCC by -5e-5 to +5.9e-4 on speech and by -3.7e-4 on the
synthetic tone (tone3); the tone item's full-model cosine moves by -3.1e-4. The worst-token
sensitivity (s10 token 126, s30 token 267) is the known E6 item and is not addressed here; its
cosine moves with any change of accumulation order (s10: 0.764 -> 0.834).

## Gates (final code `fa62290e86`)

`e5c/scripts/gates.sh` on HEAD `fa62290e86` (clean tree for `tt/`, `tests/`, `demo/`; HF
`refs/main` untouched), log `e5c/logs/gates.log`, per-gate logs `e5c/logs/gate_*.log` **[M]**:

| gate | command | result |
|---|---|---|
| audio suite | `pytest tests/pcc/audio -q` | 116 passed (53 s): per-layer / per-module PCC, crops, 12-layer stack, soft tokens (bar 0.99), fp32-act variant, trace == eager (max abs 0.0), determinism, fallback audit, perf |
| full model | `pytest tests/pcc/test_full_model.py tests/pcc/test_full_model_trace.py -q` | 41 passed; audio items cosine 0.99895-0.99979 (bar 0.99) |
| demo | `pytest demo/demo.py -s` | 2 passed; `retrieval_argmax_agree` 1.0 |
| watcher | `TT_METAL_WATCHER=10 pytest tests/pcc/audio -k "not perf and not reference_matches"` | 107 passed, 9 deselected (397 s); `watcher.log` 653 lines, 0 lines matching `error|assert|sanitiz|overflow|hang|exception|fault` |
| trace allocation | `TT_METAL_TRACE_ALLOC_TRACKING=1 pytest tests/pcc/audio -q` | 116 passed, no "still alive" report |
| vision (comment-only edits in `tt/vision`, carry-over commit) | `pytest tests/pcc/vision -q` | 91 passed; all 67 vision PCC records identical to E5b HEAD (`results_review_fix.jsonl`) |
| all PCC | `pytest tests/pcc -q -k "not s32768"` | 354 passed, 1 deselected (438 s) |

Fallback audit: `test_audio_fallback_audit.py` passes (no torch call, no host-transfer entry
point in `AudioModel.forward`), and the after profile has 1765 `tt_dnn_device` rows and no host op
**[M]** `tracy/after/b1_s768/ops_perf_results.csv` (not committed).

## tt-perf-report (before / after)

`tracy/{before,after}/{b1_s768,b4_s256}/`: `forward_perf_report.txt` (table with advice),
`forward_perf_report.csv`, `forward_perf_report_stacked.csv`, `forward_perf_report.console.log`.
One warmed eager forward between the `PERF_AUDIO` signposts
(`test_audio_perf.py::test_tracy_audio_forward`, `TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT=4000`).
"before" is the E3 code (`tt/audio` checked out at `182629fea4`, `scripts/before_tracy.sh`). The
eager profile's op-to-op gaps are host dispatch; the traced replay does not pay them.

| case | ops | device us | matmul | eltwise binary | eltwise unary | norm | gather | data movement (reshape, transpose, slice, concat, (un)tilize, head split / merge, repeat) |
|---|---|---|---|---|---|---|---|---|
| before 1 x 768 | 1357 | 53,940 | 25,702 | 5,530 | 6,055 | 2,313 | — | 13,192 |
| after 1 x 768 | 874 | 14,665 | 6,049 | 2,490 | 1,073 | 2,042 | 1,399 | 1,297 |
| before 4 x 256 | 1369 | 88,054 | 45,183 | 10,162 | 9,760 | 2,616 | — | 18,490 |
| after 4 x 256 | 970 | 18,903 | 7,694 | 3,740 | 1,403 | 2,136 | 1,404 | 2,131 |

Advice in the after report and what was done with it:
- "HiFi2 is sufficient ... 2x the throughput": a precision change, measured above (E6 screen) and
  left for E6.
- "place input 0 in L1" on input_proj (tried, c20, rejected), embed_audio (its input is the fp32
  tower output, a trace output kept in DRAM) and the lconv shift matmuls (in0 is the constant shift
  matrix in DRAM; 7.5 MB of constants are not pinned in L1).
- "High op-to-op gap": eager dispatch, not present in the traced replay (bench matches the device
  time: 15.02 ms vs 14.67 ms profiled).

## Remaining (not done)

- RMSNorm is 14 % of 1 x 768 (108 calls, 17-19 us each on 24-32 cores). The block-sharded variant
  is faster at <= 768 rows but lowered accuracy (c21); a sharded residual stream end to end was not
  built.
- The gather of the rel shift is 9.5 % (117 us per layer). No exact cheaper formulation was found:
  the shift is a per-row skew; the row-major view trick costs 244 us.
- The 0/1 shift matmuls of the lconv do dense work on band matrices (4 x 17-20 us per layer at
  1 x 768); LoFi is not exact for them.
- Batch > 1 lconv still uses block context + selection matmul + slices.
- The subsampling conv0 input still pays a 232 us row-major reshape to [B, T, 128, 1] at 1 x 768
  (C = 1 conv input).
- Precision levers for E6: HiFi2 projections (-1.4 ms at 1 x 768), sharded RMSNorm, LoFi shift
  matmuls; the fp32-activation variant (`act_dtype=float32`) still works and passes its tests.

## Context contract

`doc/context_contract.json` -> `audio` -> `optimized_e5c`: no capability change (buckets 32-768,
750 tokens per clip, batch); new persistent constants (7.5 MB shift matrices shared by the layers,
the [1, 1, 32, 64] uint16 shift index), and the mel input contract is now [B, 1, 4S, 128] fp32
TILE.
