# Prefill towers (E10.1): vision and audio tower prefill at 12 x 10, batch 1

Vision tower (patch embedder, 16 encoder layers, pooler, `embed_vision`) and audio tower (subsampling,
12 conformer layers, `output_proj`, `embed_audio`) of `google/embeddinggemma-2` on one P150a (150 W
default TDP, firmware 19.8.0.0) opened on the 12 x 10 compute grid (120 cores, the Galaxy target;
`EMBEDDING_GEMMA2_GRID`, `conftest.open_eg2_device`). Contract: batch 1, vision 1 x 2560 (one image),
audio 1 x 256 (10 s) and 1 x 768 (30 s), real processor inputs (`landscape`, `s10`, `s30`). Metric:
traced device latency per forward (`execute_trace` replays, inputs resident), burst and sustained.
Precision unchanged: vision `v_exp_bfp8w4`, audio `a_h2b8_qkvbf16_shift2_sharded`. Stages applied:
baseline, topology (`prefill-topology`), matmul (`prefill-matmul`). Text code and shared helpers are
untouched. Labels: **[M]** measured (run named in `work_log.md`), **[I]** inferred. Run folder:
`tt-ops/runs/embeddinggemma-2/e10/`.

## Before / after (ms per forward) [M]

Before = same-session control on the final commit with every E10.1 switch off
(`EG2_VISION_MM_TUNED=0 EG2_AUDIO_MM_TUNED=0 EG2_VISION_PATCH_ONE_OP=0 EG2_VISION_PATCH_L1=0
EG2_AUDIO_SHIFT_IDX_ONCE=0`, which issues the `7febde8744` tower calls: `config_dump.py --tuned-off`
0 of 89 entries differ, and the topology switches restore the old code paths). After = code defaults.
Harness: `EMBEDDING_GEMMA2_BENCH_ALL=1 bench_vision.py / bench_audio.py` (burst, sustained with
telemetry, HF fp32 check of the traced replay output), after then control, back to back. Vision at
`a1501dbbb0` (`run/final2.jsonl`), audio at `36bed7c5b2` (`run/final.jsonl`; the audio files did not
change after it). Sustained headline; AICLK and board power are the medians of the last third.

| tower | grid | shape | burst before | burst after | change | sustained before (AICLK MHz, W) | sustained after (AICLK MHz, W) | change | soft-token PCC / embedding cos after (before) |
|---|---|---|---|---|---|---|---|---|---|
| vision | 12x10 | 1x2560 (landscape) | 16.279 | 15.133 | -7.04 % | 17.807 (1225, 127.5) | **16.167** (1256, 127.5) | **-9.21 %** | 0.99778 / 0.99975 (0.99730 / 0.99973) |
| audio | 12x10 | 1x256 (s10) | 7.584 | 7.457 | -1.68 % | 7.582 (1350, 99.0) | **7.457** (1350, 90.0) | **-1.65 %** | 0.99907 / 0.99991 (0.99849 / 0.99992) |
| audio | 12x10 | 1x768 (s30) | 12.192 | 12.095 | -0.79 % | 12.191 (1350, 111.0) | **12.097** (1350, 108.0) | **-0.77 %** | 0.99767 / 0.99992 (0.99767 / 0.99992) |
| vision | 13x10 | 1x2560 | 17.364 | 17.276 | -0.51 % | - | - | - | 0.99658 / 0.99974 (identical) |
| audio | 13x10 | 1x256 | 7.310 | 7.303 | -0.09 % | - | - | - | 0.99849 / 0.99992 (identical) |
| audio | 13x10 | 1x768 | 11.952 | 11.901 | -0.43 % | - | - | - | 0.99767 / 0.99992 (identical) |

- Gates: soft-token PCC >= 0.99 and embedding cosine (cosine of the mean soft token) >= 0.99 vs HF fp32
  on the traced replay pass for every row. 13 x 10 runs only the topology rewrites (vision: max abs 0 vs
  the switch off; audio: same PCC, cosine and minimum token cosine as the control, see "13 x 10 calls unchanged").
- Vision throttles under sustained load (AICLK 1225-1256 MHz at the 150 W limit); the after run holds a
  31 MHz higher clock at the same power (1256 vs 1225 MHz), so the sustained gain (-9.2 %) exceeds the
  burst gain (-7.0 %) **[M]** (telemetry). The -9.2 % includes that clock difference (about 2.5 % of
  clock **[I]**); the code effect is closer to the burst -7.0 % (review E10.1 M1). Audio stays at 1350 MHz; the after run draws 9 W (256) / 3 W (768) less.
- Window spread under 0.1 % (audio) and under 1.6 % (vision sustained) in every row.
- The stage-1 baseline (`run/baseline.jsonl`, harness commit `36736a6ca8`) matches the controls: vision
  16.277 burst, audio 7.585 / 12.192. Vision sustained differs between sessions (17.513 vs 17.807 ms,
  AICLK 1250 vs 1225 MHz: thermal history); the A/B above is same session.

## Baseline at 12 x 10 and the grid cost [M]

Stage-1 harness, commit `36736a6ca8`, `run/baseline.jsonl` (13 x 10 in the same session):

| shape | 12x10 burst / sustained (AICLK) | 13x10 burst / sustained (AICLK) | 12x10 vs 13x10 |
|---|---|---|---|
| vision 1x2560 | 16.277 / 17.513 (1250) | 17.364 / 18.713 (1243) | **-6.3 % / -6.4 %** (12 x 10 is faster) |
| audio 1x256 | 7.585 / 7.582 (1350) | 7.310 (closure control) | +3.8 % |
| audio 1x768 | 12.192 / 12.188 (1350) | 11.954 / 11.950 (1350) | +2.0 % |

Vision is faster on 120 cores: SDPA runs 240 (12 heads x 20 q-chunks of 128) work units, exactly two waves
on 120 cores, against 1.85 waves on 130 **[I]** (SDPA is 315 us per layer at 12 x 10, Tracy). Audio pays
for the smaller grid in the 2D matmuls (88-96 of 120 cores used, `roles_before.md`).

## Top costs per shape (before, 12 x 10 Tracy, one warmed eager forward) [M]

Profiles `profiles_before/<shape>/op_table.md` (`prefill_op_table` roofline model, 120 cores, 1350 MHz).

| shape | device kernel sum / ops | 1st | 2nd | 3rd |
|---|---|---|---|---|
| vision 1x2560 | 16.11 ms / 320 | SDPA 5.04 ms, 31 %, 5.2x model floor: **04 attention** | gelu(gate) * up multiply 2.35 ms, 15 %, NoC-bound on L1-interleaved inputs: **05 memory** (sharded MLP chain) | MLP matmuls: up / gate 1.74 ms (11 %, 1.5x floor), down 1.63 ms (10 %, 2.8x): **03 matmul** (down kept here) |
| audio 1x256 | 6.19 ms / 1090; wall 7.58 ms | FFN down 0.73 ms, 12 %, DRAM-bound 3.4x: **03 matmul** (kept), next DRAM-sharded weights (**05**) | rel-shift `ttnn.gather` 0.59 ms, 9.5 %, 49 us per layer: **06 custom kernel** | FFN up 0.57 ms, 9.2 %: **03** (no win found); sharded RMSNorm + reshards 0.86 ms (14 %): **05 / 07** |
| audio 1x768 | 11.30 ms / 1090; wall 12.19 ms | rel-shift gather 1.19 ms, 10.6 %, 99 us per layer: **06** | FFN down 0.95 ms, 8.4 %, compute 2.0x: **03** (kept) | FFN up 0.95 ms, 8.4 %: **03** (no win); norms + reshards 1.28 ms (11 %): **05 / 07**; conv0 input relayout 0.36 ms (3.2 %): **05 / 06** |

At 1 x 256 the wall time exceeds the kernel sum by 1.39 ms (1.3 us per op): op count matters there.

## Topology (stage 2)

One layer's op sequence, before and after, with the data-movement-only ops marked: `layer_ops.md`.
Vision: 19 ops per layer, 1 data-movement-only (`nlp_create_qkv_heads`, 22 us). Audio: 89 ops per
layer, 31 data-movement-only (110 us at 1 x 256, 203 us at 1 x 768, of which the gather is 49 / 99 us
and the 18 block-sharded RMSNorm reshards 37 / 55 us).

| # | tower | candidate (catalogue class) | action | accuracy | burst (same-session control) | kept | reason / blocker |
|---|---|---|---|---|---|---|---|
| t1 | audio | rel-shift index `ttnn.repeat` built in every layer (1: mask / constant built per layer) | build once per forward in `AudioModel.forward_hidden`, pass to each layer | bit-identical (max abs 0) | 1x256 -0.26 %, 1x768 -0.42 % (`search/audio_a0`) | yes, every grid | 12 ops per forward -> 1 |
| t2 | vision | `2 * px - 1` as multiply + subtract on fp32 [2560, 768] in DRAM, 55 us each (2: spelled-out math) | one `ttnn.unary_chain` (MUL_UNARY_SFPU 2, SUB_UNARY_SFPU 1) | bit-identical | -0.35 % (`vision_v3`), +0.34 % when turned off at the final code (`vision_v3off_g12`) | yes, every grid | |
| t3 | vision | patch-embedder position lookups and sums in DRAM (5: placement of short-lived tensors) | lookups, `table_x + table_y` and `h + pe` in L1 (vision rule) | bit-identical | +0.46 % when turned off at the final code (`v3off_g12`); 13 x 10 +0.23 % | yes, every grid | first version also put the projection output in L1: ttnn then picked another default config for it (30 cores, 97.7 vs 64.2 us, not bit-identical at 13 x 10); fixed in `a1501dbbb0` |
| - | vision | head split before SDPA (1) | none | | | no | q / k RMSNorm and RoPE run on the split heads before SDPA; no SDPA entry reads a fused QKV |
| - | vision | concat-heads after SDPA (1) | already `output_concat_heads=True` (E5b) | | | - | |
| - | vision | residual add + RMSNorm (2) | none | | | no | sandwich norm: the model adds `norm(a)` to `x`; `rms_norm(residual_input_tensor=)` computes `norm(x + a)` |
| - | vision | gate / up packing (3) | not built | | | no | the multiply needs two output slices (2 x ~25 us **[I]**) for one fewer 54 us matmul at its floor (1.5x); E5b same conclusion |
| - | vision | gelu into the gate matmul (4) | not re-measured | | | no | E5b A/B: the multiply pre-activation form wins at every shape |
| - | audio | GLU: linear_start then two slices (4: narrow the matmul) | not built | | | no | two N = 1024 matmuls cost 2 x 10.4 us at 256 vs 15.5 us + 2 x 2.5 us slices **[I]** from the profile: no gain |
| - | audio | interleaved -> block-sharded -> RMSNorm -> interleaved, 9 per layer (1) | not changed | | | no | the producers (2D matmul on 88-96 cores) and consumers do not use the norm's 8 x 8 shard grid; changing it is a placement change (stage 5) or the E6b precision choice (`norms.sharded`, stage 7) |
| - | audio | rel-shift gather (1) | not changed | | | no | 99 us per layer at 768; no stock op does the per-row skew; the E3 row-major trick was slower; a reader-side shift is a custom kernel (stage 6) |
| - | audio | conv0 input: TILE -> ROW_MAJOR -> reshape to width 1 -> interleaved-to-sharded (253 + 109 us at 768) (1) | not changed | | | no | `ttnn.conv2d` needs the C = 1 NHWC input (one 4-byte page per pixel); alternatives (host masking, conv0 as banded matmul, channel folding) change the input contract or numerics: stage 5 / 6 candidate |

Op counts per forward (Tracy): vision 320 -> 319, audio 1090 -> 1079 at both buckets.

## Matmul (stage 3)

Every matmul role per shape with M x K x N, calls, device time, share, cores, bound class and gap to the
model floor: `roles_before.md`, `roles_after.md`. In-model search tables (every candidate, including
failures with the exact error): `search_tables.md`. Method: `tt-ops e10/scripts/sweep.py`, one device
session per set, overrides per (role, shape), trace capture, 5 burst windows after 2 s idle, control first
and last (drift 0.000-0.007 ms in every set), replay output vs the control and vs HF fp32.

Material roles and outcome (12 x 10, batch 1):

| shape | role | M x K x N | before (us per call, cores) | searched | kept | after (us per call) |
|---|---|---|---|---|---|---|
| vision 1x2560 | pool (one-hot pooler) | 288 x 2560 x 768 | 279.3, 6 cores (ttnn's pick) | 2D 12x9 / 12x3, in0_block_w 8 / 16 | 2D on 12 x 9 | 21.8, 108 cores |
| vision 1x2560 | MLP down | 2560 x 3072 x 768 | 102.1, minimal_matmul, h in DRAM | minimal blocks (4,8,4) (4,8,2) (8,8,4) (8,4,8: TT_FATAL subblock > dest), h in L1, 2D in0_block_w 4 / 8 / 12 / 16 / 24 (24: CB clash) | 2D in0_block_w 12, h in L1 | 58.8 |
| vision 1x2560 | up / gate | 2560 x 768 x 3072 | 54.5, 1.5x floor | in0_block_w 4 / 12, subblock 2x2 | none (all slower) | 54.5 |
| vision 1x2560 | qkv | 2560 x 768 x 2304 | 43.0, 1.6x | subblock 2x2, in0_block_w 4 / 12 | none | 43.0 |
| vision 1x2560 | o | 2560 x 768 x 768 | 20.4, 2.2x | in0_block_w 12 / 24, minimal | none | 20.4 |
| audio 1x256 | FFN down | 256 x 4096 x 1024 | 30.4, 88 cores, DRAM 3.4x | 1D, 1D bw16, transposed 2D, in0_block_w 4 / 16 / 32 | 2D in0_block_w 32 | 26.7 (3.0x) |
| audio 1x256 | FFN up, qkv, lconv start, post / lconv end, shift | 256 x 1024 x {4096, 3072, 2048, 1024}, 256 x 256 x 1024 | 23.7 / 17.8 / 15.5 / 10.4 / 4.0 | 1D, 1D on 8x8, transposed 2D, in0_block_w 4 / 16 | none (all slower or tied) | |
| audio 1x256 | rel-pos Q E^T (block bmm) | 8 x 256 x 128 x 32 | 5.0, 16 cores | per_core_M 1 / 2 | per_core_M 2 | 3.6, 32 cores |
| audio 1x768 | FFN down | 768 x 4096 x 1024 | 39.5, 88 cores, compute 2.0x | transposed 2D, in0_block_w 4 / 16 / 32 | 2D in0_block_w 16 | 37.9 |
| audio 1x768 | FFN up, qkv, lconv start, post / lconv end, shift, rel-pos | | 39.5 / 28.3 / 23.1 / 13.8-18.0 / 13.4 / 5.6 | transposed 2D (per role and all: +17 %), per_core_N 13 / 14 / 16, in0_block_w 4 / 16, per_core_M 1 / 2 | none | |

Final config table (12 x 10 only; every other grid keeps the E5b / E5c rule):

| tower | key | config | file |
|---|---|---|---|
| vision | ("pool", 288) | `MMSpec("2d", grid=(12, 9))`: per_core (1, 2), in0_block_w 8 | `tt/vision/linear.py` `VISION_TUNED` |
| vision | ("down", 2560) | `MMSpec("2d", in0_block_w=12)`: per_core (8, 2), subblock (2, 2) | `VISION_TUNED` |
| vision | MLP h placement | L1 up to 2560 patches (`"h": 2560`) | `tt/vision/config.py` `VISION_L1_WIDE_TUNED` |
| audio | (256, 4096, 1024) | `AudioMMSpec("2d", in0_block_w=32)` | `tt/audio/linear.py` `AUDIO_TUNED` |
| audio | (768, 4096, 1024) | `AudioMMSpec("2d", in0_block_w=16)` | `AUDIO_TUNED` |
| audio | rel-pos per_core_M at S = 256 | 2 (rule: 4) | `tt/audio/attention.py` `REL_PM_TUNED` |

The tables apply at batch 1 only (the batched vision MLP slices pass `tuned=False`). Switches:
`EG2_VISION_MM_TUNED=0`, `EG2_AUDIO_MM_TUNED=0`; `VISION_MM_OVERRIDES`, `AUDIO_MM_OVERRIDES`,
`VISION_L1_WIDE_OVERRIDES`, `REL_PM_OVERRIDES` are the in-process search hooks (empty by default).

### 13 x 10 calls unchanged (proof) [M]

`tt-ops e10/scripts/config_dump.py` (host only: ttnn ops replaced by recorders, device stub reporting the
grid) records what `vision_linear` (5 roles), the pooler, the MLP h placement, `audio_linear` (6 roles) and
the rel-pos per_core_M issue for 5 vision and 8 audio shapes (batch 1 and batched) per grid, for the
current tree and for the same files at `7febde8744`:

    grid 12x10: 6 of 89 entries differ (vision down / pool / h_mem b1_s2560, audio down b1_s256 / b1_s768, rel_pm b1_s256)
    grid 13x10: 0 of 89 entries differ
    --tuned-off: 0 of 89 differ at both grids

The topology rewrites run on every grid and every shape (batched shapes included). Bit identity (max abs 0
against the switch off) was proven only at batch 1: vision 1 x 2560 on both grids (`vision_v3off_g12`,
`vision_v3off_g13`) and audio 1 x 256 / 1 x 768 at 12 x 10 (`audio_a0`). At 13 x 10 the audio bench check
gives the control's soft-token PCC, embedding cosine and minimum token cosine to every printed digit
(table above), which is consistent with bit identity but not a max-abs proof. The batched shapes were not
compared against the switch off; their existing PCC tests pass (review E10.1 M3).

## Accuracy and gates

Kept matmul changes at 12 x 10 change the accumulation order (K blocking) and are not bit-identical
(max abs vs control 0.65 vision, 2.0 / 0.78 audio, `search_tables.md`); the HF fp32 soft-token PCC goes
up (vision 0.99730 -> 0.99778, audio 256 0.99849 -> 0.99907, audio 768 unchanged 0.99767) and the
embedding cosine stays at 0.99975 / 0.99991 / 0.99992. Trace allocation tracking
(`TT_METAL_TRACE_ALLOC_TRACKING=1`, fails on unsafe allocations) passes on the replay of every contract
shape (`track2_v12`, `track_a12`).

Closure gates on `a1501dbbb0` (tree clean apart from these docs), `tt-ops e10/scripts/gates.sh`,
`logs/gates_summary.log` **[M]**:

| grid | tests/pcc/vision | tests/pcc/audio | test_full_model + test_full_model_trace | demo | text smoke test_text_model[b1_s512] |
|---|---|---|---|---|---|
| 12 x 10 | 91 passed | 117 passed | 41 passed (incl. test_retrieval[audio], the E6 text -> audio near tie) | 2 passed | 1 passed |
| 13 x 10 | 91 passed | 117 passed | 41 passed | 2 passed | 1 passed |

The batched tower shapes (4 x 2560, 8 x 1280, 4 x 256) were not tuned or timed; their existing PCC tests
are in the suites above and pass. No test, threshold, reference or golden was changed; the only test edit is
an added Tracy case (`test_tracy_audio_forward[b1_s256]`).

## Not done / next stages

- 04 attention: vision SDPA is 33.6 % of the forward after this unit (315 us per layer, 5.2x the model
  floor); its chunk / grid table still holds the 13 x 10 entry capped to 12 x 10.
- 05 memory: the vision gelu(gate) * up multiply (144 us per layer, NoC-bound on interleaved L1 inputs,
  31.7x its floor after h moved to L1) wants a block-sharded MLP chain; audio block-sharded RMSNorm reshards
  (18 ops per layer); DRAM-sharded FFN weights at 1 x 256 (weight-bound, 3.4x); the conv0 input relayout.
- 06 custom kernel: the audio rel-shift gather (49 / 99 us per layer).
- 07 precision is out of scope here (precision policy unchanged).
- The 13 x 10 matmul configs were not re-tuned (P150 path kept byte-identical by design).
