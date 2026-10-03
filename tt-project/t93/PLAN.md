# t93: next LTX conv VAE decode cuts (off-device plan)

Base: ttp/t48-ltx25-integrated @ b0b95926402. Shape: 1080p/145f on 4x8 = 544x960/145f on a
2x4 submesh (same per-chip work). No device jobs were run for this plan.

## Measurement caveat (fix first)

The 2x4 A/Bs behind #46, #65, #75 and #79, and the "1.99 s conv decode" baseline, ran
without `LTX_CONV3D_BLOCKING_MESH=4,8`. Without it, 544x960 on 2x4 misses the swept table and
conv3d runs on channel-fallback blockings (~1.5 s of the 2 s; t43). The production-equivalent
number is t61 / job 029 (override on, pre-folds): 828 ms device per chip, conv3d 308 ms (42
calls), NeighborPad 119, ReshapeView 156 (139 in the output path), BinaryNg 89, Permute 60,
LayerNorm 46, Concat 38.

Estimated t48 now: ~490 ms device per chip (conv3d ~308 + non-conv ~180, after #58 -170,
fold-time-pad -38, fold-w-mask -67, halo-only -62). Unmeasured. The fold savings measured
without the override still hold (they touch non-conv ops), but their % of the decode is
understated. #84 (LoFi/bf8 conv3d) must set the override or its conv3d numbers mean nothing.

Device-profiler op shares overstate non-conv ops. kevinmi's traced ablation (identity-patch
an op type, re-trace) found RMSNorm 8.7% of the traced decode (profiler said 21%), BinaryNg
4.2% (said 18.7%), NeighborPad ~0 (origin/kevinmi/neighbor-pad-conv3d-fused, 7504dc803bc).
Judge every cut by traced wall time, not op sums.

## Ranked tasks

| # | Task | Expected saving | Bit-exact? | 2x4 job? |
|---|------|-----------------|------------|----------|
| 1 | Rebaseline with the override, traced + profiled | 0 (fixes all numbers) | n/a | 1 short |
| 2 | Exact-shard rebalance after s0 upsample | -40 to -50 ms device | expected yes | 1 short |
| 3 | ~~Port fused neighbor_pad+conv3d onto the halo-only path~~ DROPPED (t98) | ~0 (halo cap ~12.5 ms) | n/a | done (job 455) |
| 4 | Fold the resnet residual add into the next norm (dual-output RMSNorm) | -10 to -20 ms | likely not; PCC 0.99999 | 1-2 short |
| 5 | Re-sweep conv3d blockings for the halo-only reader (top 4 shapes) | -15 to -30 ms | C_out/T/H/W yes; C_in block may round differently | several short |

### 1. Rebaseline (prerequisite)

One 2x4 job on t48 tip: `LTX_CONV3D_BLOCKING_MESH=4,8`, `LTX_VIDEO_VAE_TRACE=1` traced warm
replays (wall), then one profiled eager pass (op table). Gives the real per-chip budget and
shows whether the decode is still device-bound once conv3d uses tuned blockings (#87 found
trace = eager only on fallback blockings). Everything below is ranked on estimates until then.

### 2. Exact-shard rebalance after s0 upsample

At 1080p the latent is 34x60. On 4x8 that is 8.5x7.5 per chip, padded to 9x8, and the pad
doubles at each upsample: runtime per-chip tensors are 18x16, 36x32, 72x64 while the logical
data is 17x15, 34x30, 68x60. Every op from s1 on does 1 - (17*15)/(18*16) = 11.5% wasted work.
After s0's depth-to-space the logical dims divide evenly (68/4, 120/8), so one re-shard there
makes every later stage exact. The swept `_BLOCKINGS` keys are already the unpadded dims
(sweep runs at key + kernel - 1), so today's runtime shapes do not match what was tuned.

- Saving: ~34 ms conv3d (11.5% of ~293 ms s1+ conv) + ~15-20 ms non-conv, minus 3-5 ms for the
  re-shard. Re-shard is a rightward shift (chip i sends i rows/cols to its neighbor): cheapest
  as a one-direction neighbor send; simplest as all_gather per axis + slice on the s1 input
  (39x18x16x512 bf16, 11.5 MB per chip, the smallest post-s0 tensor).
- Also removes most of the logical-pad masking (pad_offset / logical_h/w masks become no-ops
  from s1 on) and simplifies the output crop.
- Code: `vae_ltx.py` (`_compute_ltx_decoder_dims`, the s0 up block, output crop), `LTX_VAE_EXACT_SHARD`
  opt-in. CPU test: torch reference of the re-shard + decoder parity on a small mesh mock.
- Bit-exact if the pad region was exactly zero-masked before (folds already guarantee that).

### 3. Fused neighbor_pad + conv3d: DROPPED (t98)

Result (t98, ttp/t98-t93-3-port-fused-neighbor-pad-conv3d-ont @ 4330b76f589; blx03 2x4,
544x960/145f, job 455): skipping the halo exchange entirely on all 31 routed convs saves at
most 12.5 ms of a ~530 ms decode. The fused op's 8 reserved cores (+4-6 ms) and its ~262 us
per-call overhead (~8 ms) cancel that out: net 0 to -7 ms, inside the 5-12 ms noise. Any
halo-exchange work is capped at ~12.5 ms. Original plan text below.


origin/kevinmi/neighbor-pad-conv3d-fused (last 2026-06-30) has a fused NeighborPad+Conv3d op
that overlaps the halo exchange with interior compute (halo_last two-pass). Measured there:
traced whole decode 1525 -> 1444 ms (-5.6%) at 1088x1920 on 2x4, per-op 1.05-1.21x on s3/s4,
PCC >= 99.999% vs standalone. t48 still carries its routing tables (`_HALO_LAST_KEYS`,
`_FORCE_SPATIAL_KEYS` in `utils/conv3d.py`) but nothing reads them: the fused op never merged.
Halo-only (#79) already removed the interior copy, so the remaining gain is the exchange
latency hidden behind compute; at our per-chip shapes expect roughly -20 to -60 ms. Highest
effort: 3-month-old C++ op to rebase onto halo-mode conv3d (#52514) and t48's conv3d changes,
plus a build. Do a CPU/build-only port first, then one A/B job per routed shape group.

### 4. Residual add folded into the next norm

Each LTXResnetBlock3D ends with `ttnn.add(residual, h)` (18 calls, ~21 ms in t61's profile),
and the next block starts with RMSNorm+SiLU on that sum. kevinmi's WIP f9cc61e24ad adds a
dual-output RMSNorm (normed + pre-add sum): +16% vs add+norm in TILE layout, +23-37%
row-major. It was blocked by row misalignment at non-tile-aligned W (15) with TILIZE_IN +
FUSE_PRE_ADD. Note task 2 makes W=15 everywhere, so fix the misalignment rather than rely on
padded dims. Alternative: residual add in the conv3d writer epilogue (one rounding instead of
two; not bit-exact). Saving -10 to -20 ms. Traced ablation puts all BinaryNg at 4.2% of decode,
so this caps out near there.

### 5. Blocking re-sweep for the halo-only reader

Production blockings (PR #53633 sweep) were tuned on the padded-input reader. Halo-only reads
the interior from the unpadded input plus a compact halo buffer: different NoC pattern, so the
optimum can move. Re-sweep s2_res (8 x 13.8 ms), s3_res, s4_res, s1_up with the
bruteforce harness (`models/tt_dit/tests/models/ltx/bruteforce_conv3d_sweep_ltx.py` on top of
`wan2_2/bruteforce_conv3d_sweep.py`; add halo mode). Wan's re-sweep after a reader change gave
1.16x (PR #45975). Run after #84 settles fidelity/dtype and after task 2 settles shapes, or the
sweep is wasted. Each sweep job must stay short (one shape per job).

Hint (t98, job 455): one grid column fewer slows the routed convs by only ~1.5-2%, so
blockings that use fewer columns cost little and can free cores for other work.

## Looked at and set aside

- Trace the conv decode: done in #87, no gain on fallback blockings; recheck in task 1 only.
- Winograd / FFT conv: bf16 accuracy risk, big kernel work, not worth it at ~40% conv share.
- Fusing the norm into the conv3d reader: the 27-tap gather would redo the norm 27x.
- Streamed T-chunk decode overlapped with export (Wan cached mode): a pipeline change that
  only pays if the VAE is on the e2e critical path after audio/export overlap; check after an
  e2e 4x8 run is allowed.
- #56922 (MiniMax-H3 decoders, not in t48): `rgb_to_yuv(wide_rows=True)` and deferred readback
  cut H3's D2H 6.2 -> 2.3 ms per wave. LTX's output path is already 38 ms after #58; worth a
  cherry-pick only if task 1 shows D2H on the critical path.
- The user's premise that the VAE is the main underperformer: on the estimates above the
  conv decode is ~0.5 s of a ~6 s e2e (denoise ~68%). Tasks 2-5 together are ~-0.1 to -0.15 s.
