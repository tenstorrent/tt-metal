# LTX-2.3 8+3: which ltx-rt/t48 gains over main are safe (v2, task #506, 2026-10-14)

Scope: LTX-2.3 distilled AV, BH Galaxy 4x8 Ring, 1088x1920, 145 frames, 8+3 steps.
LTX 2.5, H3 and 8+2 are out of scope. S2 3->2 steps (b21f12b93a2) is not a gain (#190: degrading).

Refs: main 8afb5d0661c (code reading), t48 1c5a2127400, PR 60209 head 2981ef5d2a9 (on main 14565689d1e).
v1 (#296) was code reading only. v2 adds the device results of #301, #305, #343, #428, #435, #444, #446,
#447 and settles items 8, 9, 11 and 12 by reading main against t48.

**All device numbers below were taken on blx01 under the 900 MHz AICLK clamp. Use them as ratios
only.** Run-to-run noise on the same main build (jobs 903 vs 930): total 8.12 vs 8.08 s, S1 2.94 vs
2.90 s, S2 and VAE equal. So a single-stage delta under ~0.05 s is noise.

Classes: **a** = bit-identical, **b** = quality-neutral with evidence, **c** = numerics change, no
evidence vs main yet (untested), **d** = degrading. **main** = already on main, not a gain.

## Final class table

| # | Gain | Commits | In PR 60209? | Class | Evidence | Measured (900 MHz, relative) |
|---|---|---|---|---|---|---|
| 1 | QK-RoPE fusion (bf16-preserving) + active-core RoPE | e2efcea16a1 (PR); t48 f8a8611de5a | yes | a | commit: latents and mp4 unchanged on 4x8 | part of PR S1/S2 gain below |
| 2 | Drop per-step replay syncs + latent stats | 4194cd98852 | no | a (logging only) | main has no per-step syncs in the denoise loop (only under LTX_TIME_STAGES); what is left is the default-on latent stats, ~17 ms per stage per the commit | ~0.03 s, unmeasured |
| 3 | Seeded noise once, on a host thread | 0a6f0ebccf3 | yes | a | same values with an explicit generator; #305: used, part of the S1/S2 gain | in PR gain |
| 4 | Reuse the S1 prompt in traced S2 | f26872c2664 | yes | a | same buffers, same data | in PR gain |
| 5 | Skip the V2A pad-mask multiply | 3eb94707517 | yes | a | md5-identical, blx03 job 161 | in PR gain |
| 6 | Async mp4 encode (audio track alongside) | 533c60ffa5d | yes | a | mp4 byte-identical | outside the test's timing table (#305) |
| 7 | x264 ultrafast crf 20 export | be3d6b2dc03 | yes | b | Y PSNR vs source 47.9 dB (main 45.6 dB); #301: main vs PR decoded frames PCC 0.9983-0.9988, PSNR 38.0-39.6 dB over 3 gens (lossy-vs-lossy, includes rows 1-5), stills match | outside the timing table (#305) |
| 8 | Audio VAE + BWE traces | 2ddbdbc5788, dropped again in 2981ef5d2a9 | no (dropped) | a | **settled, see below**: main already traces vocoder and BWE; only the mel-VAE trace is off on main, and it is worth ~2 ms (#305) | ~0 |
| 9 | Keep the latent upsampler resident | ab67c9e86b2 | no | a | **settled, see below**: main never evicts it on 4x8 | 0 on 4x8 |
| 10 | On-device YUV export default (LTX_YUV_EXPORT=1) + wide_rows fix for T>32 | main code; t447 84538e83ee6 + aa26f9e6e14 (not pushed) | no (on branch ttp/t447-...) | **b** (was c) | #446: 5 seeds x 145 frames, lossless yuv420p vs float path PSNR 53.4-55.0 dB, worst frame Y >= 52.5 dB, <= 0.14% of samples off by > 1 LSB, stills identical; #447: on vs t446-on bit-identical | VAE decode 0.94 -> 0.70-0.72 s (#446 jobs 950/951, #447 job 963) |
| 11 | Fold the gated residuals into to_out (LTX_FOLD_GATED_RESIDUAL) | ltx-rt 4e7d0cefcdb | n/a | **main** | **settled, see below**: main folds all six gated residuals unconditionally. Not a gain over main. (Not bit-identical vs an unfused addcmul, but main is already on the folded path.) | 0 |
| 12 | Fused RMSNorm+AdaLN (LTX_FUSE_NORM_ADALN) | aef81cd4a10 + f6b806516cc (v1 wrongly listed a613d669eef, which is the on-device gate fold) | no | c | **settled, see below**: not superseded by #58676; main's LTX block does not use it. Only evidence: t170, measured together with the gate fold (PCC 0.87 vs ref, VBench same). Not bit-identical: the normalized intermediate stays fp32 instead of rounding to bf16 before addcmul | commit claims ~0.1 s per 1080p gen; likely most of t48's S2 lead over the PR (gap table) |
| 13 | t48 conv VAE trims (exact shard, T-pad/W-mask fold, halo-only reader, blockings) + 11cdcb2caea | see v1 row 13 | no | a vs ltx-rt; c vs main | md5-identical vs ltx-rt only; never compared with main's halo/persist-pad VAE | ~0.23 s of VAE decode (gap table: t48 0.47 s vs PR+YUV ~0.70 s) |

Class counts: bit-identical 1, 2, 3, 4, 5, 6, 8, 9; neutral 7, 10; untested vs main 12, 13; already on main 11.

### Not gains / degrading

- **8-bit weights+activations (LTX_QUANT=all_bf8_lofi, PR 50361): d.** #435: S1 -10.1%, S2 -11.1%, total
  -7.2%, but it crashes on today's main unless DISABLE_FABRIC_AGMM=1. #444, 5 seeds: bf8 vs bf16 PSNR 15.8-25.7 dB
  against a bf16 noise floor of 22.0-28.9 dB; seed 4 swaps the whole background. Keep it out of the PR.
  Weight-only bf8: no speed gain (denoise 6.26 -> 6.24 s) and PSNR 19.4 dB on one seed. Drop it.
- Gate fold into QKV: main already folds the gate unconditionally on the fused-AGMM Ring route (attention_ltx.py
  `fuse_gate`). Fabric AGMM to_out: main already uses the strided AG-matmul to_out. Neither is a gain.
- v1's other "not gains" list is unchanged (host pinning, #57265 sweeps, device-resident hand-off, warmup-only
  items, rejected knobs).

## Items 8, 9, 11, 12 (code reading, main 8afb5d0661c vs t48 1c5a2127400)

**Item 8: are the audio VAE and BWE traced on main? Mostly yes.**
`models/tt_dit/models/audio_vae/audio_decoder_ltx.py:185-187` on main:
vocoder `use_trace = traced and LTX_VOC_TRACE (default True)`, BWE `use_trace_bwe = traced`, mel-VAE
`use_trace = traced and LTX_VAE_TRACE (default False)`. The standard test runs traced on 4x8
(`traced_default`). t48 (`:189-191`) differs only by defaulting LTX_VAE_TRACE (and a new LTX_BWE_TRACE gate)
to True. #305 measured the mel-VAE trace at ~2 ms (main's audio decode is 0.37 s at full clock), and
#428 shows audio decode 0.48 / 0.47 / 0.47 s for main / t48 / PR. Verdict: nothing left to port. The PR was
right to drop 2ddbdbc5788.

**Item 9: does main evict and reload the upsampler on every generation? No, not on 4x8.**
`pipeline_ltx.py` on main: the BH (4, 8) device config sets `dynamic_load=False` (only BH (2, 4) sets True).
`_register_coresident_exclusions` returns at once when `dynamic_load` is False, so nothing ever evicts the
upsampler, and `_prepare_upsampler` -> `LTXLatentUpsampler.reload_weights` is a no-op when `is_loaded()`
(`latent_upsampler_ltx.py:311-314`). #428: main's "Latent upsample" row is 0.14 s in both runs, with no 1.5 s
reload. Verdict: ab67c9e86b2 helps only the 2x4 dynamic_load path. Out of scope for this PR.

**Item 11: does LTX_FOLD_GATED_RESIDUAL change rounding? Yes, but main already has the fold.**
- Rounding: the t48 test (`test_transformer_ltx.py:104-110`) says the fold "re-associates the bf16
  accumulation, so the two paths agree to rounding, not to the bit". It bounds fused vs unfused at about
  PCC 99.9996 / RMSE 0.3% (a wrong gate gives 99.85 / 6.0%). So it is not bit-identical against a standalone
  `addcmul`, despite the "math-identical" comment at `transformer_ltx.py:36`.
- Main: `transformer_ltx.py` passes `addcmul_residual`/`addcmul_gate` into to_out for all six residuals,
  with no flag: video attn1 (:389), video attn2 (:410), audio attn1 (:442), audio attn2 (:462), A2V (:504) and
  V2A (:527). The fold has been there since #45797 and #50361. It goes through the same to_out epilogue
  primitive that t48 uses (`attention_ltx.py` `fused_ternary_input_a` / `addcmul_input_tensor1`). ltx-rt's
  4e7d0cefcdb squash had lost three of the folds, and t48 re-added them behind the flag.
- Verdict: **already on main, not a gain, nothing to port, no A/B needed.**

**Item 12: does main's tile-row AdaLN (#58676) supersede the fused RMSNorm+AdaLN? No.**
- #58676 (fbd0af7e10f) touches only MiniMax-H3 model files, `layers/normalization.py` and the
  `dit_fused_distributed_rmsnorm` op. It adds an `affine_tile_row_map` for per-token (tile-row) modulation.
  LTX's AdaLN is scalar per block (shift/scale shape (1,1,1,D)) and does not need that map.
- Main's LTX block still runs the norm and the modulation as two programs: `self.norm1(x)` and then
  `ttnn.addcmul(shift, normed, scale_p1)` (`transformer_ltx.py:381-382, 396, 434-435, 450`, plus
  `_modulated_ffn` :321).
- t48's `_norm_adaln` calls `norm(x, dynamic_weight=scale_p1, dynamic_bias=shift)`. Main's
  `DistributedRMSNorm.forward` already accepts exactly these (`normalization.py:201-291`, present since
  before #58676). With `norm_elementwise_affine=False`, `dynamic_weight` becomes the weight.
- Verdict: the port is Python-only (aef81cd4a10's `transformer_ltx.py` hunk plus the f6b806516cc default).
  No C++ is needed, since the op change it relies on is already on main. It changes numerics (fp32
  intermediate), so it stays class c until the A/B below.

## PR 0.973 vs t48 0.884: gap by stage (#428, blx01, 900 MHz, relative only)

The main column shows jobs 903 / 930. t48 is job 905, a gen #1 table from t48's own test with a different
prompt. The PR is job 933, gen #2.

| Stage | main | t48 | PR | t48 - PR | Where it comes from |
|---|---|---|---|---|---|
| Encoder | 0.24 / 0.24 | 0.24 | 0.26 | -0.02 | noise |
| Stage 1 (8 steps) | 2.94 / 2.90 | 2.78 | 2.85 | -0.07 | near noise (main varies 0.04 s run to run); row 12 is the only default-on DiT difference left between t48 and the PR (env scan of both trees) |
| Latent upsample | 0.14 / 0.14 | 0.18 | 0.14 | +0.04 | t48 is slower; unexplained, one sample |
| Stage 2 (3 steps) | 3.36 / 3.36 | 3.04 | 3.20 | -0.16 | above noise; row 12 is the likely main source. Rows 11 (already on main), the gate fold and fabric AGMM (on main) are not differences |
| VAE decode | 0.96 / 0.96 | 0.47 | 0.94 | -0.47 | -0.24 row 10 YUV export (#446: 0.94 -> 0.70 on the PR); about -0.23 row 13 conv VAE trims vs main's halo VAE |
| Audio decode | 0.48 / 0.48 | 0.47 | 0.47 | 0 | none: rows 8 and 9 are worth nothing here |
| Total | 8.12 / 8.08 | 7.18 | 7.86 | -0.68 | sum checks: -0.02 -0.07 +0.04 -0.16 -0.47 +0 = -0.68 |

So about 70% of the gap is the VAE (YUV export plus conv VAE trims) and about 30% is the denoise stages
(probably row 12). The export-side gains (rows 6 and 7) do not show in this table at all, because they run
after it is filled (#305).

## What the PR should add next (in order)

1. **Row 10, YUV export on by default, with the T>32 wide_rows fix**: t447 84538e83ee6 + aa26f9e6e14.
   It is quality-neutral (53-55 dB lossless, 5 seeds) and worth -0.22 to -0.24 s of VAE at 900 MHz. It is the
   largest proven-neutral gain not yet in the PR. It touches shared code (`yuv_d2h.py`), so its unit test
   (#447 job 960) goes with it. The branch is ready; it needs review and then a fast-forward onto the PR branch.
2. **Row 2 remainder**: latent stats opt-in (logging only, bit-identical, ~0.03 s). Cheap, but it is optional.
3. **Row 12**, only if A/B-12 below passes: Python-only, likely -0.1 to -0.2 s of denoise.
4. **Row 13**, only after its own A/B against main's halo VAE (v1 A/B-4). It brings shared-op C++
   (neighbor_pad_async, conv3d) that needs op tests. It is about -0.23 s but the most work.
5. Do not add rows 8, 9 or 11 (no gain on main 4x8), or 8-bit (d).

## Device A/B specs

Common to every A/B:
- Box: blx01 (it has the LTX-2.3 checkpoint, Gemma, the python env and the t446 harness), through its own
  tt-device-mcp broker, behind other tenants, after its health check passes. One project device job at a
  time on the box.
- Test: the standard test, unmodified:
  `models/tt_dit/tests/models/ltx/test_pipeline_ltx_distilled.py::test_pipeline_distilled -k 4x8sp1tp0nl2_ring_is_fsdp0`,
  8+3, 1088x1920, 145 frames, traced.
- Harness: copy `tt-project/t446/run446.sh` (fill/timed modes, the env unset list, stall watchdog, disk
  guard) and change only the arm switch. Timed mode uses `SEED=0 RUN_VBENCH=1 VBENCH_EXPORT_DIR`, which
  renders gen #2 plus warm seeds 1-4, and `T446_LOSSLESS=1` (hook/sitecustomize.py), which writes lossless
  mp4s, so the frames compare before x264.
- **One arm per broker job, run back to back on the same box, build and JIT cache.** A warm arm takes about
  320 s (#428 job 930: 319.6 s), so a pair (~640 s) does not fit the 600 s cap. #447 used the same split.
  Arm limits: fill 560 s (pytest timeout 540 s), timed 400 s (pytest 390 s).
- Pass bar (quality): off arm vs plain main must be md5-identical on the lossless clips. On arm vs off arm,
  per seed: PCC/PSNR from `cmp446.py` / `cmp301.py`. If PSNR is >= 45 dB on all 5 seeds, the item is neutral.
  Otherwise, compare against the bf16 noise floor from #444 (22.0-28.9 dB), then run VBench (5 dims, `vb444.py`)
  and look at frame-72 stills on all 5 seeds. Any visible content change makes it d.
- Speed: the gen #2 stage table, quoted verbatim, on vs off. Ratios only while blx01 is clamped at 900 MHz.

**A/B-11: none.** Main already folds all six gated residuals, so applying item 11 alone on top of main is
an empty diff. An "unfold" arm would only measure a regression of main. No job is proposed.

**A/B-12: fused RMSNorm+AdaLN alone on top of main.**
- Build: a branch from current main (8afb5d0661c or newer) plus **one commit**. Port `_norm_adaln` and the
  `LTX_FUSE_NORM_ADALN` gate from t48 `transformer_ltx.py` (aef81cd4a10 + f6b806516cc: `_fuse_norm_adaln_enabled`,
  `self._fuse_norm_adaln` fixed at construction, and the five call sites: video norm1/norm2, audio
  norm1/norm2, and `_modulated_ffn`). Nothing else from t48. No C++.
- Pre-check (CPU, no device): the off path must be textually main's (`ttnn.addcmul(shift, norm(x), scale_p1)`),
  so the off arm equals main.
- Jobs (blx01, in order, one at a time):
  1. fill: `LTX_FUSE_NORM_ADALN=0`, 560 s. JIT for the shared build. Then one more fill with `=1`, because
     the fused norm compiles a different kernel variant. Skip either fill if its arm's JIT is already warm.
  2. timed off: `LTX_FUSE_NORM_ADALN=0`, 400 s, 5 seeds, lossless.
  3. timed on: `LTX_FUSE_NORM_ADALN=1`, 400 s, 5 seeds, lossless, right after job 2.
  4. Reference (optional, only if no lossless main clips at the same seeds exist): plain main timed, 400 s, to
     show the off arm is md5-equal to main.
- Report: S1/S2 ratios on/off. Expect S2 about 0.95 and a total of about -0.1 to -0.2 s at the clamp, if row 12
  explains the gap. Per-seed PCC/PSNR, stills, VBench if under 45 dB, and a verdict a/b/d.
- If it passes as b: add it to PR 60209 as a default-on commit, after one standard e2e on the PR branch.

**A/B-13 (for completeness, v1 A/B-4):** main + t48's conv VAE commits (row 13 + 11cdcb2caea) vs main, the
same harness. Only the "VAE decode" row and lossless md5/PSNR matter. It needs the shared-op unit tests
first, so it is a code task before it is a device task.
