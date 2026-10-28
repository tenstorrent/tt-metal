# E5b: optimized vision tower (stage report)

Scope: the EmbeddingGemma 2 vision path (`tt/vision/`: patch embedder, 16 encoder layers, pooler,
`embed_vision`) on one Blackhole P150a (13x10 grid), sjc-snva-tp100, 2026-10-27. The functional
baseline is the E2 vision code (unchanged through E5a; `tt/vision` at `86c6b9f9f7`). Text and
audio code paths are unchanged: no shared file (`tt/mlp.py`, `tt/rms_norm.py`, `tt/linear.py`,
`tt/config.py`) was edited; the vision tower imports `tt/linear.py` helpers.

Labels: **[M]** measured (command and log named), **[I]** inferred from code or data, **[G]** guess.
Logs, sweeps and raw data: `tt-ops/runs/embeddinggemma-2/e5b/` (`bench.jsonl`, `results_*.jsonl`,
`logs/`, `scripts/`, `tracy/`).

## Result

Traced device time per forward (`tests/perf/bench_vision.py`: inputs resident, `execute_trace`
back to back, one sync per window). Burst = median of 5 windows of 10 replays, each after 3 s idle.
Sustained = median of 5 windows of 20 replays after a 10 s soak. Window spread is under 0.2 % in
every row; sustained equals burst within 0.1 ms, so the card did not throttle on these runs.
Inputs: real processor outputs (b1 landscape; b4 landscape, portrait, square, small; b8 the 8
video frames).

| batch x bucket | E2 code burst ms | E2 code sustained ms | E5b burst ms | E5b sustained ms | speedup | ms per image |
|---|---|---|---|---|---|---|
| 1 x 2560 (image) | 62.797 | 62.787 | **26.271** | 26.277 | 2.39x | 62.8 -> 26.3 |
| 4 x 2560 (images) | 323.257 | 323.296 | **116.020** | 116.153 | 2.79x | 80.8 -> 29.0 |
| 8 x 1280 (video frames) | 228.314 | 228.262 | **88.142** | 88.164 | 2.59x | 28.5 -> 11.0 |

**[M]** `e5b/bench.jsonl` labels `baseline_E2` (HEAD `86c6b9f9f7`, the E2 vision code) and
`r_c4_mlp_split` (the final code: the E5b review fix, per-image MLP for batches, on top of
`47824a6e41`). `logs/bench_baseline.log`, `logs/review_b4_code.log`. Before the review fix the E5b
code (`47824a6e41`, label `c3_wide_l1`, `logs/final3.log`) measured 26.263 / 121.619 / 94.643 ms.

Speedup against both baselines (E5b burst median):

| batch x bucket | vs E2 code re-measured today | vs E2 published (E2 report) |
|---|---|---|
| E5b as reviewed (`47824a6e41`) | | |
| 1 x 2560 | 62.797 / 26.263 = 2.39x | 55.60 / 26.263 = 2.12x |
| 4 x 2560 | 323.257 / 121.619 = 2.66x | 294.47 / 121.619 = 2.42x |
| 8 x 1280 | 228.314 / 94.643 = 2.41x | 199.61 / 94.643 = 2.11x |
| E5b with the review fix (per-image MLP) | | |
| 1 x 2560 | 62.797 / 26.271 = 2.39x | 55.60 / 26.271 = 2.12x |
| 4 x 2560 | 323.257 / 116.020 = 2.79x | 294.47 / 116.020 = 2.54x |
| 8 x 1280 | 228.314 / 88.142 = 2.59x | 199.61 / 88.142 = 2.26x |

E2 published 55.60 / 294.47 / 199.61 ms for the same code. Re-measured today it is 62.80 / 323.26 /
228.31 ms (+7.2 ms at 1 x 2560). Tracy of the E2 code today vs the E2 report: SDPA (2002 us per
call), RMSNorm and eltwise rows are identical; the extra time is in the default-config HiFi4 bf16
matmuls, 1.3-2.4x slower with the same program config (e.g. MLP down 2560x3072x768: 175 us in E2,
417 us today) **[M]** `tracy/before/b1_s2560` vs `doc/functional_vision/tracy/b1_s2560`. The
cause is **unexplained**. What it is not: the ttnn / tt_metal libraries predate E2 (built
11:59-12:01, E2 ran from 13:00; file times of `build/lib/libtt_metal.so`, `ttnn/ttnn/_ttnn.so`,
`e2/logs/`) **[M]**, and the text control reproduces: the E5a text forward at 1 x 512 measured
7.315 ms in E5a (`e5a/logs/final.log`) and 7.315 ms again in the E5b gates
(`results_gates_a410c73278.jsonl`, `full_latency_device_ms` b1_s512) **[M]**. The text path runs
tuned HiFi2 matmuls, not default-config HiFi4 bf16 ones, so the control does not cover the slowed
op type **[I]**. Every before / after number in this report is from today, same card, same harness.

## Op-topology audit (one encoder layer, 1 x 2560)

| step | E2 ops (device us, Tracy) | candidate | action | evidence |
|---|---|---|---|---|
| input norm | rms_norm (26) | L1 output | kept: 15 us | Tracy after |
| QKV | one fused linear, bf16 HiFi4 default config (166) | 2D multicast config, HiFi2 + fp32 acc; bfp8 weights; L1 output | kept 2D/HiFi2 with L1 output (43 us); bfp8 rejected (accuracy) | "Precision" |
| head split | nlp_create_qkv_heads (57) | L1 input and output | kept: 18 us | Tracy after |
| q/k/v norm | 3 x rms_norm (27/27/25) | fuse with RoPE / pre-split | no fused head-norm + RoPE op for prefill; L1 outputs (21/21/19) | Tracy after |
| RoPE | x @ R (44) + 2 mul + add (28 x 3), per tensor | `rotary_embedding_llama` with trans_mat = 32x32 block of R | kept: 29 us per tensor, same output | `logs/rope_b4_lost_sdpa.log`, `sdpa_b4_b8.log` (rope_check) |
| SDPA | dense bf16 [B,1,S,S] mask, 8x8 q256 k256, HiFi4 (2002) | chunk/grid sweep, HiFi2, bfp4 mask, no-mask bound, windowed mode | kept 13x10 q128 k512 HiFi2 + bfp4 mask (944); windowed at 8 x 1280 | "SDPA" |
| concat heads | nlp_concat_heads (20) | SDPA `output_concat_heads=True` | kept: op removed | |
| o_proj | linear default (118) | 2D config / minimal_matmul | kept 2D (20 us) | "Matmul search" |
| post-attn norm + residual add | rms_norm + add | rms_norm `residual_input_tensor` computes norm(x + r), not x + norm(a) | not applicable **[I]**; L1 placement kept (15 + 11 us) | |
| MLP gate/up | 2 linear (192/192) + gelu (90) + mul (105) | gelu in the matmul packer, gelu as multiply pre-activation, packed gate|up, L1 outputs | kept: 2D configs with L1 outputs (55/56) + gelu as multiply pre-activation (135) | "MLP" |
| MLP down | linear default (417) | minimal_matmul / 2D config | kept minimal_matmul (110); 2D 213 us with the DRAM input | Tracy L9 |
| per forward | mask: subtract, multiply, bf16 repeat (251), tilize (117) | bfp4 broadcast add | kept (45 us) | |

Ops per forward at 1 x 2560: 448 (E2) -> 320 (E5b) **[M]** Tracy. Decode, KV cache, paging,
sampling, LM head, MoE and CCL items of the optimize skill do not apply to this one-pass encoder
on one chip.

## SDPA

Sweep: `e5b/scripts/sdpa_sweep.py` (q chunk 64-256, k chunk 64-512, grids 8x8 / 10x10 / 12x10 /
13x10, HiFi4 / HiFi2 / LoFi, fp32 accumulation on/off, mask bf16 / bfp8 / bfp4, no mask, windowed)
and `sdpa_windowed.py` (windowed vs masked, per config). Real shapes: 12 heads x 64, valid counts
of the test images, PCC vs an fp32 torch golden over valid rows. Eager, back to back, one sync
**[M]** `logs/sdpa_b1.log`, `logs/sdpa_b4_b8.log`, `logs/sdpa_win.log`.

| shape | E2 config (us) | best kept (us) | config kept | PCC per op |
|---|---|---|---|---|
| 1 x 2560 | 2011 | 964-982 | 13x10 q128 k512, HiFi2 + fp32 acc, bfp4 mask | 0.999937 |
| 4 x 2560 | 12643 | 3659 | same | 0.999940 |
| 8 x 1280 | 6241 | 1993 | 12x10 q128 k256, windowed (no mask) | 0.999948 |
| 1 x 1280 | — | 267 | 12x10 q128 k256, bfp4 mask | 0.999939 |

At 1 x 2560 with the kept config:

| variant | us | PCC | decision |
|---|---|---|---|
| HiFi4, fp32 acc, bf16 mask | 1046 | 0.999982 | — |
| HiFi2, fp32 acc, bf16 / bfp8 / bfp4 mask | 974 / 964 / 964 | 0.999937 | kept HiFi2 + bfp4 |
| LoFi, fp32 acc, bfp4 mask | 997 | 0.997975 | rejected: slower and less accurate |
| HiFi2, fp32 acc off, bfp4 mask | 456 | 0.999358 | rejected: end-to-end soft-token PCC 0.990183, worst-token cosine 0.519 (row X_sdpa16) |
| no mask (lower bound, wrong output) | 910 | — | the mask costs about 5 % |
| windowed (`cu_window_seqlens`), no mask | 988 | 0.999937 | not kept at 2560 |

Windowed vs masked (best config of each): 8 x 1280 1993 vs 2635 us (padding is 16 % of each
frame, so the windows skip real work); 1 x 1280 403 vs 267; 4 x 2560 3770 vs 3655; 2 x 2560 1862
vs 1850. Kept: windowed only for batch >= 8 at bucket 1280 (`attention.WINDOWED_MIN_BATCH`).
The windowed path flattens the batch into one sequence [1, 12, B*S, 64]: `cu_window_seqlens` =
[0, n_0, S, S + n_1, ..., B*S] puts each frame's valid prefix in its own window, so no mask and no
cross-frame attention. It needs the valid patches to be a prefix (processor output always is;
`prepare_inputs` raises otherwise). Batches 2-7 at 1280 were not measured and use the mask **[G]**.

### Windowed SDPA unit test (E5b review fix)

`tests/pcc/vision/test_vision_windowed_attention.py` (10 tests) checks the windowed path directly.

- **Numerics, layers 0 and 15.** Input: 8 frames at bucket 1280 with different valid lengths (1080,
  1260, 972, 1278, 81, 1170, 630, 1152; tile aligned and not, 2 to 1199 padding rows). The test
  runs `VisionAttention` twice on the same input: once through `PaddingWindows`, once through the
  bfp4 key-padding mask, as `VisionModel.forward_hidden` builds each. It checks four things:
  - The valid rows match: PCC >= `PCC_LAYER` (0.995) per frame.
  - The valid rows do not change when every padding row of the input is overwritten with noise
    (x50), on either path.
  - The windowed padding rows are exactly 0 (zero input, own window), while the masked padding
    rows are not. So the padding rows differ by design.
  - The pooling matrix from `prepare_inputs` has zero weight on every padding row and weight 1 on
    every valid row. So the padding rows are never consumed.
- **Host.** `padding_windows` structure (2B+1 int32 entries, windows 2b / 2b+1 = valid prefix /
  padding of frame b, all non-empty), its `ValueError` for n in {0, S, S+1, -1}, the
  `use_windowed_attention` gate (8 / 7 frames, 1280 / 2560), and the non-prefix `ValueError` in
  `prepare_inputs`.

Result **[M]**: 10 passed. Windowed and masked outputs are bit-identical on every valid row (max abs
diff 0, PCC 1.0), and the noise changes no valid row (max abs diff 0).
`e5b/logs/review_windowed_pass1.log`. The test catches a wrong window: with frames 0 and 1 merged
into one window (planted in `padding_windows`, not committed), 3 tests fail
(`test_padding_windows_structure` and both layers of `test_windowed_matches_masked`: PCC 0.830 /
0.866 on frames 0 / 1 at layer 0, padding leak up to 14.5). Reverted, 10 pass again.
`e5b/logs/review_windowed_bite_{planted,reverted}.log`.

## Matmul search

`e5b/scripts/mm_sweep.py` (M = 2560 with L1 input, M = 10240 DRAM; every vision role; bf16 weights,
HiFi2 + fp32 acc; minimal_matmul vs 2D multicast on 13x10 / 12x10 / 13x8 / 12x8 / 8x10 / 10x10
with in0_block_w 2-8) **[M]** `logs/mm_sweep.log`. Kept per role (`tt/vision/linear.py`):

| role | M = 2560 (one image) | M = 10240 (batch 4 / 8 frames) |
|---|---|---|
| QKV 768 -> 2304 | 2D 13x10 bw8: 43 us with L1 output, 76 with DRAM output (E2 166) | minimal_matmul 433 us |
| o_proj 768 -> 768 | 2D 13x10 bw8: 20 us (minimal 48) | minimal_matmul 145 us (2D 222) |
| gate / up 768 -> 3072 | 2D 13x10 bw8: 55 us each with L1 output, 101 with DRAM output (E2 192) | minimal_matmul 516 us each |
| down 3072 -> 768 | minimal_matmul 109 us (2D 213 with its DRAM input) | minimal_matmul 329 us (2D 840) |

(device us from the after Tracy at 1 x 2560 and 4 x 2560; the DRAM-output figures are from the
mid-stage Tracy `e5b/tracy/mid_w16`, the 2D-down figure from `e5b/tracy/L9`.) The text rule in `tt/linear.py` sends
narrow outputs (N <= 768) at M >= 2048 to minimal_matmul; `vision_linear` uses the 2D config for
M <= 4096 when K <= 1024 (o_proj), and the text rule otherwise. The first version also moved down
to the 2D config: in the micro sweep (L1 input) it won 64 vs 108 us, but in the model its input is
in DRAM and it took 213 us; whole forward 30.02 vs 28.98 ms at 1 x 2560 (row L8) -> rejected for down.

## MLP

`VisionMLP` (`tt/vision/mlp.py`): up and gate projections, then `gelu_tanh(gate) * up` as one
multiply with gelu as its SFPU pre-activation, then down. Gelu in the gate matmul's packer (the
text choice above 1024 tokens) cost 52 us per gate matmul at M = 2560 (153 vs 101 us, Tracy mid);
the pre-activation form was faster in every shape: 28.98 -> 28.70 ms at 1 x 2560, 128.58 ->
121.67 at 4 x 2560, 112.19 -> 105.28 at 8 x 1280 (rows L7 -> L10). Packed gate|up was not built:
N = 6144 needs two output slices and loses the separate 2D configs **[I]**.

## Precision

Weights stay bf16 (the checkpoint dtype). Matmuls run HiFi2 with fp32 accumulation, SDPA HiFi2
with fp32 accumulation, norms HiFi4 (unchanged). Soft-token accuracy on the 7 vision test cases
(min over every image / frame, vs HF fp32; `test_vision_model.py::test_soft_tokens`) **[M]**
`results_<label>.jsonl`:

| row | change | soft-token PCC min | worst-token cosine | 1 x 2560 ms |
|---|---|---|---|---|
| L0 | E2 path | 0.997739 | 0.926977 | 62.47 |
| L3 | SDPA table + HiFi2, bfp4 mask, fused RoPE | 0.997316 | 0.929685 | 42.68 |
| L4 | + attention projections bfp8 | 0.995885 | 0.824403 | 39.95 |
| L5 | + MLP bfp8 (text `MLP(optimized=True)`) | 0.996286 | 0.623919 | 31.31 |
| L6 | L5 with bf16 weights | 0.997025 | 0.909138 | 31.51 |
| X | final with SDPA fp32 acc off | 0.990183 | 0.519016 | 20.16 |
| final | kept path (`47824a6e41`) | 0.997163 | 0.863832 | 26.26 |

bfp8 weights pass the 0.99 gate but cut the worst soft token from 0.909 to 0.624 cosine and are
only 0.6 % (1 x 2560) to 4.6 % (8 x 1280) faster than bf16 weights with the same configs, so bf16
was kept; the bfp8 trade is left to the E6 precision sweep. SDPA without fp32 accumulation is 30 %
faster end to end at 1 x 2560 but drops the soft-token PCC to 0.990, at the gate: rejected here,
for E6.

**Accuracy cost of the kept path.** The E5b path loses a small, systematic amount of soft-token
accuracy against E2; it is not at the level of HF's own bf16 run. Per image or frame, vs the HF fp32
reference (E2 = row L0, E5b = the final E5b code with the per-image MLP review fix, HF bf16 = HF
run in bf16 against the same reference) **[M]** `results_L0_none.jsonl`, `results_review_fix.jsonl`
(`vision_soft_token_*`, `hf_bf16_*`; `e5c/scripts/softtok_stats.py`):

| image | soft-token PCC E2 / E5b / HF bf16 | worst-token cosine E2 / E5b / HF bf16 |
|---|---|---|
| portrait | 0.99908 / 0.99804 / 0.99843 | 0.9429 / 0.8638 / 0.8662 |
| video frame 7 (largest loss) | 0.99904 / 0.99769 / 0.99927 | 0.9853 / 0.9505 / 0.9952 |
| video frame 3 (lowest PCC) | 0.99774 / 0.99653 / 0.99841 | 0.9523 / 0.8656 / 0.9654 |
| video frame 5 | 0.99895 / 0.99830 / 0.99876 | 0.9862 / 0.9648 / 0.9816 |
| mean of the 15 images / frames | 0.99896 / 0.99856 / 0.99895 | — |

- Mean soft-token PCC drops by 4.0e-4 against E2, and by up to 1.34e-3 on one image (video frame
  7). E5b is below HF bf16 on 12 of the 15 images.
- Worst-token cosine falls from 0.943 to 0.864 on the portrait (HF bf16 0.866) and from 0.952 to
  0.866 on video frame 3, where HF bf16 holds 0.965. (HF bf16's own worst token is 0.815, on
  video frame 4, where E5b scores 0.955.)
- The ablation rows (run on the reviewed code `47824a6e41`, before the per-image MLP fix) attribute
  the loss mostly to two changes. The gelu-as-multiply row (L8 -> L9) moves frame 5 from PCC
  0.99827 to 0.99733 (cosine 0.970 to 0.909) and the portrait cosine from 0.887 to 0.862. The
  bf16-weight projection / MLP config rows (L3 -> L6, HiFi2 configs) move the portrait cosine from
  0.962 to 0.909. The SDPA HiFi2 table (L0 -> L1) also moves frame 5 from PCC 0.99895 to 0.99791.
- Every gate still passes: the minimum soft-token PCC is 0.99653 (video frame 3) against a bar of
  0.99.
- The worst-token cosine is a noisy tail metric: it moves by +-0.03 between rows whose op outputs
  are identical to 1e-6 (L2 -> L3).
- To revisit in the E6 datatype sweep: HiFi4 for the projections and MLP, and the gelu placement.

The per-image MLP review fix changes only the batched cases (b2, video8), because their MLP matmuls
now use the one-image configs; single images are bit-identical. Against the reviewed code
(`47824a6e41`, `results_gates.jsonl`) the mean soft-token PCC moves from 0.99859 to 0.99856, video
frame 5 recovers (PCC 0.99733 -> 0.99830, cosine 0.909 -> 0.965) and video frame 3 drops (PCC
0.99716 -> 0.99653, cosine 0.923 -> 0.866). The portrait is unchanged.

## L1 placement

`vision_act_memory_config` (`tt/vision/config.py`, the E5a text rule): an activation goes to L1
interleaved when it needs <= 64 KB per bank and the forward has <= 4096 patches. At 1 x 2560 this
puts the residual stream, the norm outputs, the heads, RoPE and SDPA outputs and the
[*, 768]-wide projections in L1: 31.51 -> 28.98 ms at 1 x 2560; batch 4 / 8 frames (10240
patches) stay in DRAM, unchanged **[M]** rows L6 -> L7. Two named wide intermediates also go to
L1 up to 2560 patches (`VISION_L1_WIDE_MAX_TOKENS`): the MLP up / gate outputs (121 KB per bank
each) and the fused QKV output (91 KB per bank): 28.25 -> 26.26 ms (row 14). The gelu-multiply
output stays in DRAM (no gain in L1). Raising the cap for every tensor failed with a
circular-buffer / L1-buffer clash: at 128 KB per bank and 10240 patches (row L12) and at 128 KB
per bank for one image ("Statically allocated dataflow buffers in program 43 clash with L1
buffers ... L1 buffer allocated at 1142784 and static dataflow buffer region ends at 1160576")
**[M]** `logs/ablate6.log`, `logs/exp_l1_128b.log`. The bfp4 mask cannot move to L1: SDPA
asserts a DRAM mask (`sdpa_device_operation.cpp:165`) **[M]** `logs/exp_mask_l1b.log`.

## Batch-4 per-image cost

E2 code: 80.8 ms per image at batch 4 vs 62.8 ms at batch 1 (+18.0 ms). Tracy of the E2 code
**[M]** `tracy/before/{b1,b4}_s2560`: SDPA 202.05 ms at batch 4 (50.5 ms per image) vs 32.04 ms at
batch 1. The SDPA excess (18.5 ms per image) is the whole gap; the other op classes cost the same
per image (matmul 19.0 vs 19.1, eltwise 5.2 vs 5.4, norm 2.3 vs 3.0 ms). The E2 SDPA config (8x8
grid, q256 / k256) runs 12.64 ms per call at batch 4 vs 2.01 ms at batch 1 (`sdpa_b4_b8.log`).
Fixed by the SDPA table: 3.66 ms per call at batch 4 (0.91 ms per image, below batch 1's 0.96).

E5b: 30.4 ms per image at batch 4 vs 26.26 at batch 1 (+4.1 ms, 16 %). Tracy after (final code)
**[M]**: SDPA is now cheaper per image at batch 4 (14.6 vs 15.1 ms); the gap is the ops that use
L1 only at batch 1 and the M = 10240 projections: matmul 8.0 vs 5.1 ms per image, eltwise binary
3.2 vs 2.7, norm 2.3 vs 2.0, RoPE 1.3 vs 0.9, heads 0.9 vs 0.3. Batch 1 keeps its activations,
the QKV output and the MLP up / gate outputs in L1 (-4.5 ms together, rows L7 and 14); at batch 4
(10240 patches, 4x the bytes per bank) the same placement clashes with op circular buffers (row
L12), and the 2D matmul configs exceed L1 at M = 10240 (`mm_sweep.log`), so the projections run
minimal_matmul.

### E5b review fix: residual-only L1 and per-image MLP

Two levers, one change per measurement, screened by monkeypatch (`e5b/scripts/exp_b4.py`) with
`bench_vision.py` (burst + sustained, 5 windows each) against a same-session control of the
HEAD code (`none`) **[M]** `e5b/bench.jsonl` labels `r_b4_*`, `logs/review_b4_screens.log`,
`logs/review_b4_both*.log`:

| variant | 1 x 2560 | 4 x 2560 | 8 x 1280 | result |
|---|---|---|---|---|
| control (HEAD `4c14f6c978` = `47824a6e41` code) | 26.265 | 121.631 | 94.669 | — |
| (a) residual stream (h1, layer out) in L1 up to 10240 patches | 26.266 | 120.319 (-1.31) | 92.506 (-2.16) | faster alone; clashes when combined with (b) |
| (b) per-image MLP (slice -> one-image path, up / gate in L1 -> concat) | 26.269 | 115.692 (-5.94) | 87.813 (-6.86) | **kept** |
| (a) + (b) | — | TT_THROW | 85.700 (8 x 1280 alone) | rejected: CB clash at 4 x 2560 |
| (a) + (b), per-image MLP output in DRAM | TT_THROW at capture (all three shapes in one process; shape not isolated) | | | rejected: CB clash (L1 buffer at 958464) |

The window spread is at most 0.08 ms on every row, so both levers beat noise. Combined, they fail
at capture with "Statically allocated dataflow buffers in program 39 clash with L1 buffers on core
range [0-0 - 11-9]. L1 buffer allocated at 1019904 and static dataflow buffer region ends at
1029504". The combination runs at 8 x 1280 alone (85.70 ms). It is not kept because no single rule
covers both shapes. Lever (b) is the larger win and works at every shape, so the code keeps only
(b) (`VisionMLP._forward_per_image`, `tt/vision/mlp.py`). It applies to batches above 2560 patches
whose bucket is at most 2560: 2+ images or 3+ frames.

Committed code, `bench_vision.py`, label `r_c4_mlp_split` **[M]** `logs/review_b4_code.log`:

| batch x bucket | before (control) | after | change |
|---|---|---|---|
| 1 x 2560 | 26.265 | 26.271 | unchanged (path not taken) |
| 2 x 2560 | 65.057 (same-session control, `nosplit`) | 58.991 | -6.07 ms (-9.3 %) |
| 4 x 2560 | 121.631 | 116.020 | -5.61 ms (-4.6 %) |
| 8 x 1280 | 94.669 | 88.142 | -6.53 ms (-6.9 %) |

Batch 4 now costs 29.0 ms per image, against 26.27 at batch 1: the gap shrinks from +4.1 ms to
+2.7 ms. The remaining gap is attention-side: the QKV and o_proj projections at M = 10240, plus
norms, RoPE and heads in DRAM **[I]** (from the Tracy breakdown above; there is no Tracy capture of
the new code). The committed numbers are 0.3 ms above the screen at 4 x 2560 and 8 x 1280. The
committed run also captured the 2 x 2560 trace; the cause of the difference is not investigated
**[G]**. Gates on the new code: see `work_log.md` "E5b review fixes".

## Change ledger

Traced device ms, burst median (sustained median in brackets for the end rows); each row is
cumulative over the kept rows above it. `e5b/bench.jsonl` (label in the last column). Rows L0-L11
were A/B runs through a temporary switch (`EG2_VISION_LEVERS`, removed before commit `c504dd3268`).

| # | change | 1 x 2560 | 4 x 2560 | 8 x 1280 | decision | label |
|---|---|---|---|---|---|---|
| 0 | E2 code | 62.797 (62.787) | 323.257 (323.296) | 228.314 (228.262) | — | `baseline_E2` |
| 0' | E2 path through the switch (`output_concat_heads` on) | 62.473 | 322.021 | 226.889 | — | `L0_none` |
| 1 | SDPA chunk/grid table, HiFi2 + fp32 acc | 46.046 | 178.591 | 169.748 | kept | `L1_sdpa` |
| 2 | bfp4 key-padding mask (broadcast add) | 45.287 | 177.646 | 168.987 | kept | `L2_mask` |
| 3 | fused axial RoPE (`rotary_embedding_llama`) | 42.684 | 167.197 | 155.867 | kept | `L3_rope` |
| 4 | attention projections: `tt.linear`, bfp8 | 39.953 | 156.590 | 143.873 | bfp8 rejected (accuracy) | `L4_proj` |
| 5 | MLP: text `MLP(optimized=True)`, bfp8 | 31.313 | 123.392 | 107.003 | bfp8 rejected (accuracy) | `L5_mlp` |
| 6 | rows 4-5 with bf16 weights | 31.513 | 128.576 | 112.179 | kept | `L6_w16` |
| 7 | L1 activations and residual (64 KB/bank, <= 4096 patches) | 28.980 | 128.583 | 112.187 | kept | `L7_l1` |
| X | SDPA fp32 acc off | 20.157 (20.626) | 104.706 | 83.848 | rejected (soft-token PCC 0.990) | `X_sdpa16` |
| 8 | 2D config for o_proj and down at M <= 4096 | 30.019 | 128.572 | 112.196 | rejected (down 213 us) | `L8_vlin` |
| 9 | gelu as multiply pre-activation (VisionMLP) | 28.695 | 121.669 | 105.280 | kept | `L10_gelumul_novlin` |
| 10 | 2D config for o_proj only (K <= 1024) | 28.246 | 121.660 | 105.261 | kept | `L11_vlin_o` |
| 11 | L1 rule 128 KB/bank, 10240 patches | — | — | — | rejected (CB clash) | `L12_l1big` |
| 12 | clean code, commit `c504dd3268` | 28.247 (28.262) | 121.616 (121.676) | 105.243 (105.259) | — | `final_c1` |
| 13 | windowed SDPA for >= 8 frames at 1280, commit `a410c73278` | 28.250 (28.267) | 121.623 (121.704) | 94.650 (94.650) | kept | `c2_windowed` |
| 14a | screen: 128 KB/bank L1 cap for every tensor (<= 4096 patches) | clash | — | — | rejected (CB region end 1160576 B vs L1 buffer at 1142784 B) | `logs/exp_l1_128b.log` |
| 14b | screen: gelu-multiply output in L1 | 28.285 | — | — | rejected (no gain) | `exp_mlp_h_l1` |
| 14c | screen: MLP up/gate outputs in L1 | 26.923 | — | — | kept | `exp_mlp_upgate_l1` |
| 14d | screen: + QKV output in L1 | 26.268 | — | — | kept | `exp_wide_mlp_qkv` |
| 14e | screen: bfp4 mask in L1 | fails | — | — | blocked: SDPA requires a DRAM mask (`sdpa_device_operation.cpp:165`) | `logs/exp_mask_l1b.log` |
| 14 | 14c + 14d in code, commit `47824a6e41` | 26.263 (26.277) | 121.619 (121.673) | 94.643 (94.662) | kept | `c3_wide_l1` |
| 15a | review screen: residual stream in L1 up to 10240 patches | 26.266 | 120.319 | 92.506 | faster alone; clashes with 15b | `r_b4_resid_l1` |
| 15b | review screen: per-image MLP for batches > 2560 patches | 26.269 | 115.692 | 87.813 | kept | `r_b4_mlp_split` |
| 15c | review screen: 15a + 15b | — | clash | 85.700 | rejected (CB clash at 4 x 2560) | `r_b4_both_*` |
| 15 | 15b in code (E5b review fix) | 26.271 (26.277) | 116.020 (116.153) | 88.142 (88.164) | kept | `r_c4_mlp_split` |

## Gates

See "Gates" in `work_log.md` for the exact commands and logs (`e5b/logs/gate_*.log`).

## tt-perf-report (before / after)

One warmed eager forward between the `PERF_VISION` / `PERF_VISION_END` signposts
(`tests/pcc/vision/test_vision_perf.py::test_tracy_vision_forward[b1_s2560|b4_s2560]`, run by
`e5b/scripts/tracy_case.sh`; "before" = the E2 `tt/vision` files checked out over HEAD by
`e5b/scripts/before_tracy.sh`, restored afterwards):

```
python -m tracy -r -p -v -m pytest "models/demos/blackhole/embedding_gemma2/tests/pcc/vision/test_vision_perf.py::test_tracy_vision_forward[b1_s2560]" -q -s
tt-perf-report <ops_perf_results.csv> --start-signpost PERF_VISION --end-signpost PERF_VISION_END --csv forward_perf_report.csv
tt-perf-report <ops_perf_results.csv> --start-signpost PERF_VISION --end-signpost PERF_VISION_END --no-color > forward_perf_report.txt
```

Tables with advice, CSVs and stacked CSVs: `tracy/{before,after}/{b1_s2560,b4_s2560}/`. The
"after" captures are the final commit `47824a6e41`. Device time in us (sum of kernel time), op
count in brackets.

1 x 2560: 448 -> 320 ops, 62,628 -> 26,135 us

| op class | before | after |
|---|---|---|
| SDPA | 32035 (16) | 15105 (16) |
| matmul (before incl. the RoPE x@R matmuls) | 19110 (115) | 5068 (83) |
| eltwise binary | 5365 (151) | 2718 (57) |
| RMSNorm | 3001 (113) | 1952 (113) |
| gelu (unary) | 1443 (16) | 0 |
| create / concat heads | 1240 (32) | 298 (16) |
| RoPE (fused) | 0 | 934 (32) |
| mask repeat, tilize, embeddings, transpose | 434 (5) | 61 (3) |

4 x 2560: 448 -> 322 ops, 322,992 -> 121,503 us

| op class | before | after |
|---|---|---|
| SDPA | 202054 (16) | 58366 (16) |
| matmul | 76004 (115) | 32176 (83) |
| eltwise binary | 20719 (151) | 12614 (57) |
| RMSNorm | 9121 (113) | 9220 (113) |
| gelu (unary) | 9176 (16) | 0 |
| create / concat heads | 5081 (32) | 3758 (16) |
| RoPE (fused) | 0 | 5009 (32) |
| other | 837 (5) | 358 (5) |

Every op in both windows is a device op (no host op rows). Remaining advice in the "after"
tables: "High Op-to-Op Gap" (eager dispatch: 386 us of gaps vs 26.1 ms of kernels; the traced
forward is 26.26 ms) and "SLOW" on the matmuls (QKV at 63 % of FLOP peak, o_proj 44 %, down via
minimal_matmul at 31 % with "no program_config"; down's 2D config was measured slower, see
"Matmul search"). The patch-embedding matmul (fp32 input, HiFi4, default config, 216 us, 0.8 %)
was left as is. The largest remaining rows at 1 x 2560 per layer: SDPA 944 us, gelu multiply
135 us (DRAM in L1 out was no faster), down 109 us, gate / up 55 us each.

## Context contract

`doc/context_contract.json` -> `vision`: unchanged capability (buckets 1280 / 2560, 2520 patches
per image, 1260 per frame); adds the E5b policy (bf16 weights, bfp4 mask, windowed SDPA for
batch >= 8 at 1280) and the new `windows` device input. No reduction.

## Not done / remaining

- SDPA is 58 % of the 1 x 2560 forward (944 us per layer at HiFi2 + fp32 acc). The 2x lever is
  fp32 accumulation off, rejected here on accuracy (soft-token PCC 0.990): E6.
- bfp8 weights (0.6-4.6 % faster, worst-token cosine 0.62): E6.
- Batch-4 L1 placement. The per-image MLP is kept (review fix). A residual-only L1 policy is
  faster alone, but it clashes with the per-image MLP at 4 x 2560. It works with it at 8 x 1280
  alone (85.70 vs 88.14 ms), which would need a per-bucket rule. The batch-4 gap is now +2.7 ms per
  image, attention-side.
- Windowed SDPA for batches 2-7 at bucket 1280 (not measured; masked there).
- Per-layer tests (`test_vision_attention`, `test_vision_encoder_layer`) build the dense mask with
  `key_padding_mask`, so they run the masked SDPA path. The windowed path has its own unit test
  (E5b review fix, `tests/pcc/vision/test_vision_windowed_attention.py`, see "Windowed SDPA unit
  test") and is covered end to end (`test_soft_tokens[video8]`, trace replay == eager, full-model
  video tests).
