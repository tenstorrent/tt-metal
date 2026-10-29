# Prefill towers, stage 4 (E10.2): attention at 12 x 10, batch 1

Vision tower 1 x 2560 (one image, `landscape`, 2520 valid patches) and audio tower 1 x 256 (`s10`) / 1 x 768
(`s30`) of `google/embeddinggemma-2` on one P150a opened on the 12 x 10 grid (Galaxy target). Starting point:
E10.1 (`../README.md`). Precision unchanged (vision `v_exp_bfp8w4`, audio `a_h2b8_qkvbf16_shift2_sharded`).
Labels: **[M]** measured (run named in `work_log.md`), **[I]** inferred. Run folder:
`tt-ops/runs/embeddinggemma-2/e10/02-attention/`.

## Before / after (ms per forward) [M]

Before = same-session control on the final code with both stage switches off
(`EG2_VISION_SDPA_TUNED=0 EG2_AUDIO_ATTN_TUNED=0`, which issue the E10.1 attention calls). After = code
defaults. Harness `EMBEDDING_GEMMA2_BENCH_ALL=1 bench_vision.py / bench_audio.py` at 12 x 10 (burst, sustained
with telemetry, HF fp32 check of the traced replay output), after then control, back to back; 13 x 10:
`EMBEDDING_GEMMA2_BENCH_CHECK=1` (burst + check). Vision: commit `fb6cfaefd5`, `run/final2.jsonl`; audio:
commit `b7b727c1b7`, `run/final.jsonl` (the audio files did not change after it). AICLK / board power: medians
of the last third of the sustained run.

| tower | grid | shape | burst before | burst after | change | sustained before (AICLK MHz, W) | sustained after (AICLK MHz, W) | change | soft-token PCC / embedding cos after (before) |
|---|---|---|---|---|---|---|---|---|---|
| vision | 12x10 | 1x2560 (landscape) | 15.135 | 14.872 | -1.74 % | 16.100 (1262, 137.0) | **15.686** (1275, 135.5) | **-2.57 %** | 0.99787 / 0.99978 (0.99778 / 0.99975) |
| audio | 12x10 | 1x256 (s10) | 7.460 | 7.270 | -2.55 % | 7.456 (1350, 90.0) | **7.267** (1350, 92.0) | **-2.54 %** | 0.99907 / 0.99991 (0.99907 / 0.99991) |
| audio | 12x10 | 1x768 (s30) | 12.098 | 11.144 | -7.89 % | 12.097 (1350, 112.5) | **11.141** (1350, 112.0) | **-7.90 %** | 0.99767 / 0.99992 (0.99767 / 0.99992) |
| vision | 13x10 | 1x2560 | 17.279 | 17.280 | +0.01 % | - | - | - | 0.99658 / 0.99974 (identical) |
| audio | 13x10 | 1x256 | 7.303 | 7.302 | -0.01 % | - | - | - | 0.99849 / 0.99992 (identical) |
| audio | 13x10 | 1x768 | 11.900 | 11.900 | +0.00 % | - | - | - | 0.99767 / 0.99992 (identical) |

- Audio: bit-identical to the control (max abs 0, `search_tables.md`); PCC, cosine and minimum token cosine
  unchanged to every printed digit. Vision: k 256 changes the softmax accumulation order (max abs 0.667 vs the
  control in the search); soft-token PCC 0.99778 -> 0.99787, embedding cosine 0.99975 -> 0.99978.
- Vision sustained: the after run held a 13 MHz higher clock (1275 vs 1262 MHz, about 1 % **[I]**); the code
  effect is closer to the burst -1.7 %. Audio stays at 1350 MHz in both runs.
- 13 x 10 is unchanged by design: both tables are keyed by grid and have no 13 x 10 entry
  (`tests/perf/test_tuned_config_identity.py`, host-only, below).
- Window spread: under 0.1 % in every burst row and audio sustained; vision sustained 1.0 % (after) and 1.7 %
  (control) under thermal throttling at the 150 W limit.
- First closure (commit `b7b727c1b7`, vision q 256 / k 512, `run/final.jsonl`): vision -3.02 % burst / -3.65 %
  sustained, bit-identical, but the 12 x 10 full-model gate failed (see "Vision: q 256 / k 512 does not fit in
  the full model"); that entry was replaced by k 256 in `fb6cfaefd5`.

## What changed

| # | tower | change | where | 12 x 10 effect | kept |
|---|---|---|---|---|---|
| a1 | vision | SDPA q 128 / k 512 -> q 256 / k 256: 12 heads x 10 q chunks = 120 work units, one per core (was 240 = two waves) | `tt/vision/attention.py` `VISION_SDPA_TUNED[(12, 10)][2560]` | SDPA 315.1 -> 297.9 us per layer (Tracy `after2`); -1.7 % burst | yes, 12 x 10, batch 1 |
| a2 | audio | rel-shift: `ttnn.gather` replaced by transpose / batched one-hot matmul / transpose (stock ops) | `tt/audio/attention.py` `REL_SHIFT_MM_TUNED[(12, 10)] = {256, 768}`; `audio_model.py` skips the gather index | 49.3 -> 10.7 us per layer (256), 99.5 -> 19.5 us (768); -2.5 % / -7.9 % | yes, 12 x 10, batch 1, bf16 X |

a2 in detail. The relative-position logits are X = Q E^T on a [32 query, 32 distance] tile per block; the
Transformer-XL shift needs `bd[i, j] = X[i, idx[i, j]]` with a different shift per query row i. Grouping row i
of every block into batch i (`transpose(X, 1, 2)`: [1, M, 32, 32] -> [1, 32, M, 32]) makes the shift a fixed
per-batch selection: batch i times a one-hot P[i] [32, 64] (`P[i, c, j] = 1` iff `c = idx[i, j]`, a 4 K
element constant loaded once), then `transpose` back. Each output is one bf16 value times 1.0 plus zeros, so at
HiFi4 it is exact; P selects the gather's clamped off-band column too, so the result equals the gather bit for
bit everywhere (standalone `relshift_micro.py`: equal to `torch.gather` at both shapes; LoFi is not exact, max
abs 0.125; in-model max abs 0). It applies only to a bf16 X (the selected policy); an fp32 X (the E5c safe
baseline policy) keeps the gather. Switches: `EG2_VISION_SDPA_TUNED=0`, `EG2_AUDIO_ATTN_TUNED=0`;
search hooks `VISION_SDPA_OVERRIDES`, `WINDOWED_OVERRIDES`, `REL_SHIFT_MM_OVERRIDES` (empty by default).

## Per-shape attention table (after; 12 x 10) [M]

| shape | attention path | work units | cores | units / core | chunks | grid | dtypes | attention time per layer (Tracy) | burst / sustained ms | PCC / cos |
|---|---|---|---|---|---|---|---|---|---|---|
| vision 1x2560 | `ttnn.transformer.scaled_dot_product_attention`, dense key-padding mask (40 padding patches) | 1 x 12 heads x 10 q chunks = 120 (was 240) | 120 | 1 (was 2) | q 256, k 256 (10 k chunks; was q 128, k 512) | 12 x 10 | Q/K/V bf16, mask bfp4, HiFi2 + fp32 acc, exp approx per policy | SDPA 297.9 us (was 315.1); model floor 60.7 us (4.9x, was 5.2x) | 14.872 / 15.686 | 0.99787 / 0.99978 |
| audio 1x256 | band attention as block matmuls (8 heads x 8 blocks of 32 queries x 64-key context) | 64 blocks | rel 32, QK / PV 64 | 1 block (rel: 2) | fixed 32-query block, 64-key context | 12 x 10 | bf16 Q/K/V, fp32-acc HiFi4 | 108 us of attention ops (was 147), rel-shift 10.7 (was 49.3) | 7.270 / 7.267 | 0.99907 / 0.99991 |
| audio 1x768 | same | 192 blocks | rel 48, QK / PV 120 | 1.6 (2 waves; rel 4) | same | 12 x 10 | same | 187 us (was 267), rel-shift 19.5 (was 99.5) | 11.144 / 11.141 | 0.99767 / 0.99992 |

Audio attention ops per layer and their cost (Tracy, `attention_ops.md`): QKV clamp + matmul, `nlp_create_qkv_heads`,
q per-dim scale multiply, k scale, v clamp, rel-pos matmul `Q E^T`, rel-shift (now 3 ops), two block contexts
(slice / zero / slice / concat / concat each), QK^T block matmul, add + softcap, mask add, softmax (max,
sub+exp, sum, mul+recip), PV block matmul, `nlp_concat_heads`, post clamp + matmul. The largest after this unit
at 1 x 768: QKV matmul 28.4 us, the two block contexts 33 us (data movement), the q per-dim scale multiply
17.9 us (a [1, 1, 1, 128] broadcast operand; the scalar k multiply is 8.8 us), rel-shift 19.5 us, softmax 13.4 us.

## Requirements of the stage goal

| requirement | status |
|---|---|
| Dedicated SDPA op on every path, or the cost and blocker of the one that does not | Vision: SDPA. Audio: no SDPA; blocker below |
| Per-shape table (units, cores, units / core, chunks, grid, dtype, burst, sustained, PCC) | above |
| Chunk sizes, grid and core limits swept per shape in the full tower, failures with exact errors | vision: 36 candidate runs in 3 sets plus 4 full-model fit runs (`search_tables.md`, L1 section); audio: rel-shift path at both grids, rel-pos per_core_M in E10.1 |
| Mask preparation once per forward | vision: `key_padding_mask` once per forward (5 ops, 54.7 us, 0.4 %); audio: host-built [B, 1, S, 64] input, no device preparation; applied by one add per layer (2.7 / 6.9 us) |
| No separate head-split / concat op, or cost + blocker / stage-6 entry | vision split stays (below); audio split and concat stay (below) |
| Every contract shape passes its PCC gate; re-measured burst + sustained; no regression | table at top; gates below |

### Audio cannot use SDPA (exact reasons)

1. Logit softcap: the logits are `tanh((ac + bd) / 50) * 50` before the mask and softmax. The SDPA op
   (`ttnn/cpp/ttnn/operations/transformer/sdpa/`, `SDPAProgramConfig`, the Python binding) has no softcap or
   score-modifier argument; the only additive hook is `attn_mask`, added after QK^T with no nonlinearity between.
2. The relative-position term `bd` depends on the query values (Q E^T, shifted per row) and sits inside the
   tanh, so it cannot be passed as a precomputed additive mask; even without the softcap it would be a dense
   per-head [1, 8, S, S] input (18.9 MB fp32 per layer at 768) against the 64-wide band the block layout reads.
3. The band is 12 keys wide (causal-local, left context 12); SDPA's `sliding_window_size` could express the
   band, but not 1 or 2.

### Vision mask path

One image fills 2520 of the 2560 patches, so 40 keys must be masked: the dense bfp4 key-padding mask is needed.
The only stock path without a dense mask is the windowed SDPA (`cu_window_seqlens` [0, 2520, 2560]). It gives the
same output (max abs 0 vs the masked path, `vision_vs2` `win_q128_k512`) and is 23.7 % slower per forward
(18.725 vs 15.133 ms); with q 256 it is 71.5 % slower (q256 k256) or does not fit L1 (q256 k512). Not kept; the
masked path stays the one path at batch 1.

### Head layout

- Vision `nlp_create_qkv_heads` before SDPA: 22.0 us per layer, 352 us per forward (2.4 %) **[M]**. It stays:
  q / k RMSNorm (and the v norm) and the axial RoPE run on the split heads, and no stock SDPA entry reads a
  fused QKV. Stage-6 candidate: one kernel that writes the split, normed, rotated heads (or norm + RoPE fused
  into the head split). Output side: none (SDPA `output_concat_heads=True`).
- Audio `nlp_create_qkv_heads` (3.8 / 7.6 us per layer) and `nlp_concat_heads` (2.3 / 3.8 us) stay: the band
  attention runs on [B, H, S, 128] heads (block matmuls per head). The two block contexts (previous + own block of
  K and V: 16.7 / 33 us per layer, data movement only) are the larger layout cost. Stage-5 / 6 candidate: build
  the 64-key context in the block matmul's reader (read rows s - 32 .. s + 31 of K / V directly).

## Search summary (12 x 10, 1 x 2560, in-model burst, control first and last, drift <= 0.008 ms) [M]

| candidate | units / core | burst vs control | max abs | note |
|---|---|---|---|---|
| q256 k512 | 1 | -3.03 % / -2.98 % (two sessions) | 0 | best in the tower; L1 clash in the full model (below) |
| q256 k256 (kept) | 1 | -1.76 % | 0.667 | fits in the full model |
| q128 k256 | 2 | -0.90 % | 0.667 | |
| q128 k640 / q160 k512 / q320 k256 / q256 k320 | 2 / 2 / 1 / 1 | +1.07 / +3.55 / +4.51 / +1.89 % | | |
| q64 (k 256 / 512 / 1280) | 4 | +10 to +16 % | | |
| k 128 (q 128 / 256) | 2 / 1 | +7.5 / +5.2 % | | |
| sub-grids 12x8, 10x10, 12x5 | 2.4 - 4 | +16 to +25 % | 0 | fewer cores, more waves |
| max_cores_per_head_batch 4 / 8 / 32 | 2 or 1 | +-0.01 % | 0 | no effect: 10 cores per head at 12 heads on 120 cores |
| windowed (no dense mask) | 1 - 2 | +23.7 to +71.5 % | 0 | see Vision mask path |
| q128 k1280, q128 k2560, q256 k1280, win q256 k512 | | error | | CBs grow to 1750400 / 3245440 / 2696576 / 1680832 B, beyond the 1572864 B L1 |
| q256 k640, q512 k256, q320 k512 | | error | | static CBs clash with L1 buffers allocated at 1343488 (CB region ends at 1529216 / 1410432 / 1525120) |

Audio rel-shift (in-model, `audio_ar1`): 12 x 10 gather +2.65 % (256) / +8.55 % (768) against the matmul path;
13 x 10 (`audio_ar1_g13`, not adopted, 13 x 10 is kept unchanged) matmul path +1.63 % (256) / -8.32 % (768),
max abs 0 at both: a P150 candidate for 1 x 768 only.

### Vision: q 256 / k 512 does not fit in the full model [M]

The first closure (`b7b727c1b7`, q 256 / k 512) passed the tower suites but failed 3 of 41 full-model tests at
12 x 10 (`gates_12x10_full`, reproduced in `gates_12x10_full_rerun`): `test_full_model_item[mixed_image_audio]`,
`test_batched_mixed_modalities`, `test_no_host_fallback_in_full_forward[mixed_image_audio]`, all in the vision
SDPA: `Statically allocated circular buffers in program 67 clash with L1 buffers on core range [0-0 - 11-9].
L1 buffer allocated at 1286144 and static circular buffer region ends at 1295744`. Alone, the tower's lowest L1
buffer sits at 1343488; in the full model, after earlier items, other L1 buffers sit 57 KB per bank lower, and
the k 512 CBs overrun them by 9.6 KB. The same test file with the 2560 entry forced per candidate
(`scripts/test_full_sdpa_fit.py`, 12 x 10, `fit12_*`): q256 k512 6 failed / 24 passed; q256 k256, q128 k512,
q128 k256 30 passed each. Kept q 256 / k 256 (`fb6cfaefd5`), the best candidate that fits there.

L1 fit after the change: every contract shape ran its traced forward and every gate at 12 x 10 with the E10.1
matmul configs and placements unchanged; trace allocation tracking (`TT_METAL_TRACE_ALLOC_TRACKING=1`) passes on
the replay of the three contract shapes (`track2_v12` on `fb6cfaefd5`, `track_a12` on `b7b727c1b7`).

## Gates [M]

Final commit `fb6cfaefd5` (tree clean apart from these docs), `scripts/gates.sh`, `logs/gates_summary.log`
(rows `*_r2`); host-only identity test `tests/perf/test_tuned_config_identity.py`: 14 passed (vision SDPA and
audio rel-shift choices: 13 x 10 identical with the tables on and off, 12 x 10 differs only in the listed
entries, planted 13 x 10 entries detected); the two 13 x 10 checks fail with a planted 13 x 10 entry
(`scripts/planted_fail.py`: 2 failed, `logs/identity_planted.log`).

| grid | tests/pcc/vision | tests/pcc/audio | test_full_model + test_full_model_trace | demo | text smoke test_text_model[b1_s512] |
|---|---|---|---|---|---|
| 12 x 10 | 91 passed | 117 passed | 41 passed (incl. test_retrieval[audio], the E6 near tie) | 2 passed | 1 passed |
| 13 x 10 | 91 passed | 117 passed | 41 passed | 2 passed | 1 passed |

The batched tower shapes are not tuned (both tables apply at batch 1 only); their existing PCC tests are in the
suites above and pass. No test, threshold, reference or golden changed; the only test edit is the added identity
cases in `tests/perf/test_tuned_config_identity.py`.

## Stage-6 / later candidates left by this unit

- Vision head split + q / k / v norms + RoPE on the split heads (split 22.0 us, RoPE 60.8 us per layer; the head
  norms are part of the 124.7 us of RMSNorm per layer): one fused kernel (stage 6).
- Vision SDPA remains 4.9x its compute floor at q256 k256; q256 k512 (-1.3 % more) needs about 10 KB per core
  of L1 that the full model does not leave free (stage 5: placement of the full model's resident L1 buffers or
  of the vision attention activations); larger k chunks do not fit even in the tower (errors above).
- Audio block contexts (33 us per layer at 768) and the q per-dim broadcast multiply (17.9 us): stage 5.
- P150 (13 x 10): the matmul rel-shift wins at 1 x 768 (-8.3 %) but not at 1 x 256 (+1.6 %); not adopted here.
- Audio band attention as a single custom kernel (block context, rel-shift, softcap, mask, softmax, PV) only if
  stage 5 leaves it in the top costs (stage 6).
