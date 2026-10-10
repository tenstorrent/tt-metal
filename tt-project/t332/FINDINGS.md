# #332 LTX-2.5: Lightricks' recommended setup, and a conv VAE optimization plan (user #228)

Date: 2026-10-10. This is research only: no device jobs and no code changes.

Sources:
- the HF card https://huggingface.co/Lightricks/LTX-2.5 (the card text is public; the files and README raw are gated, and we have no token);
- LTX-2 GitHub `packages/ltx-pipelines` at commit 9ec55f9f22 (2026-10-02): `utils/constants.py`, `distilled.py`, `args.py`;
- our branch origin/ttp/t48-ltx25-integrated @ f6547442b30;
- earlier project tasks #12, #19, #20, #84, #96, #97, #98, #99, #208, #303.

## 1. What Lightricks ships and recommends

| Item | HF card / LTX-2 repo |
|---|---|
| DiT | `ltx-2.5-distilled-transformer-bf16` ("fixed 8-step schedule, CFG=1"); also the dev transformer, ComfyUI int8 variants, and nvfp4 (Blackwell only). Distilled bf16 is the fast-inference pick. |
| Text encoder | `gemma4-12b-with-proj-ltx-2.5-bf16` |
| Video VAEs | `vae/ltx-2.5-video-vae-bf16.safetensors` = DiffVAE, "higher quality, heavier" (1,472,223,346 B). `vae/ltx-2.5-video-vae-conv-bf16.safetensors` = conv VAE, "faster, lighter" (1,452,269,922 B). The card's sample command uses the DiffVAE. |
| Upsamplers | `ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0` ("required for multi-stage pipeline", 995,778,752 B; LTX-2.3's 1.1 is 995,743,560 B, so the two differ). Temporal x2 upscaler: the same size as 2.3's. |
| Extras | `duration-head` (picks the clip length when `--num-frames` is not given), `distilled-lora-450`, audio VAE, DFR pipeline, multishot, prompt enhancer, 4K via `--spatial-upscalings 2`. |
| Pipeline | `ltx_pipelines.distilled`: stage 1 at half resolution, then the x2 spatial upsampler, then stage 2 at full resolution. |
| Sigmas | S1 `[1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0]` (8 steps). S2 `[0.909375, 0.725, 0.421875, 0.0]` (3 steps). That is 8+3, the same as 2.3. There is no 2.5-specific params class: `LTX_2_4_PARAMS` inherits from 2.3. |
| Guidance | CFG 1, STG 0, modality scale 1 (distilled). |
| Sampler | Ancestral Euler (eta 1, s_noise 1) for checkpoints with model_version >= 2.5, in **both** stages. Noise seed offsets: +10000 for S1, +20000 for S2. |
| Resolution / fps | CLI default: stage 1 at 512x768, so the output is 1024x1536 at 24 fps. The Diffusers example: stage 1 at 544x960, so the output is 1088x1920 at 24 fps. Width and height must be divisible by 32; frames must satisfy % 8 == 1. Default 121 frames (5 s); 6 s = 145 frames, 10 s = 241 frames. |
| VAE decode | `AUTO_TILING` (tiled decode) and chunked denoise/decode. |
| Precision | bf16 (fp8-cast + CPU offload only for low VRAM). |

### Is the 2.5 conv VAE the same as the 2.3 conv VAE we swap in?
**Unknown. It cannot be checked without HF access.**
- The conv file is not on any box: the MLPerf snapshot 28dac7acdc has the DiffVAE but no conv VAE, no duration head and no LoRA.
- No HF token exists on g15blx02 or blx03.
- #12 found the 2.5 VAE encoder and latent stats byte-identical to 2.3 (86/86 tensors), and t48's loader calls the two "arch-identical".
- The 2.5 conv decoder weights may still be retrained.
- Today, `default_ltx25_video_vae(diffusion=False)` falls back to the 2.3 monolith `ltx-2.3-22b-distilled-1.1.safetensors`.
- Effect: none on speed (same architecture); unknown on quality.

## 2. Our t48 LTX-2.5 path compared with the recommendation

| # | Item | t48 today | Recommended | Speed effect | Quality effect |
|---|---|---|---|---|---|
| 1 | Conv VAE weights | 2.3 monolith conv VAE (fallback) | `ltx-2.5-video-vae-conv-bf16` | none (same arch) | unknown until the weights are diffed; #19 conv vs DiffVAE was 38.8 dB |
| 2 | S2 sampler | S1 ancestral, **S2 deterministic** (t48 comment: "3-step schedule too short to clear fresh noise") | ancestral in both stages (seed +20000 in S2) | ~0 (one randn + axpy per step) | a real divergence from the reference; needs a 5-seed A/B. t48 has no knob for this, so it is a code task |
| 3 | Resolution | 1088x1920 (S1 544x960) | 1088x1920 (Diffusers) or 1024x1536 (CLI default) | none | none; 1088x1920 matches the Diffusers recipe |
| 4 | Frames | 145 (6 s); 241 for 10 s via NUM_FRAMES | 121 by default, or the duration head picks | linear in frames | none |
| 5 | Steps / sigmas | 8+3, same values (f6547442b30) | 8+3 | match | match |
| 6 | fps / CFG / STG | 24 / 1 / 0 | 24 / 1 / 0 | match | match |
| 7 | DiT, TE, spatial upsampler | 2.5 distilled bf16, Gemma-4 12B, 2.5 x2 upsampler | same | match | match |
| 8 | VAE decode | untiled, whole mesh | auto-tiled, chunked | ours is faster | ours is seam-free; at most tiny differences at tile seams |
| 9 | Duration head, LoRA, temporal upscaler | not used | optional | none | none (with a fixed frame count) |
| 10 | Weights location | `default_ltx25_root()` falls back to the /mnt/MLPerf hub snapshot | – | – | device jobs must NOT read /mnt/MLPerf (charter); set `LTX25_ROOT` and `LTX25_VIDEO_VAE` to local copies |

The only mismatches that matter are #1 (needs HF access) and #2 (needs a code change plus a quality A/B). Neither changes speed.

## 3. Standard e2e command (the repo test, unmodified; env vars only)

Run from the t48-ltx25-integrated build on a 4x8 BH galaxy:

```
LTX_VERSION=2.5 LTX25_DIFFVAE=0 \
LTX25_ROOT=/var/tmp/fasth3/<local copy of the LTX-2.5 snapshot> \
LTX25_VIDEO_VAE=/var/tmp/fasth3/<local conv VAE: the 2.5 conv file if we get it, else ltx-2.3-22b-distilled-1.1.safetensors> \
NUM_FRAMES=145 HEIGHT=1088 WIDTH=1920 FPS=24 SEED=10 \
pytest -sv models/tt_dit/tests/models/ltx/test_pipeline_ltx_distilled.py::test_pipeline_distilled -k bh_4x8sp1tp0_ring
```

- `bh_4x8sp1tp0_ring` is the BH 4x8 ring-trace parametrization. It is the id the project's standard timing runs used (#293, #301, #305, #315).
- RUN_VBENCH and RUN_CLIP stay at their defaults (1).
- For 10 s, use `NUM_FRAMES=241` in its own job.
- Expected broker wall time:
  - about 380 s cold (#303, 2.3, 8+3), so submit with `-t 570`;
  - about 145-220 s with warm caches (#208).
  - A 241-frame job is unmeasured, so leave it at `-t 600`.
- The job script uses setsid plus a process-group kill trap.
- Local weights must be copied and checked readable before the broker job opens the device.
- Quote the test's timing table verbatim (#175).

## 4. Conv VAE decode: where the time goes

Measured on t48 (#96, job 435/436):
- Setup: 2x4 at 544x960 with 145 frames, which is the same per-chip shape as 4x8 at 1080p. Blocking override 4,8; HALO_ONLY, FOLD_TIME_PAD and EXACT_SHARD on (all three are now defaults on t48).
- Decode wall time: **445 ms**, eager or traced.
- The e2e "VAE" stage in #208 was 504 ms; the extra ~60 ms is upload, uint8/YUV readback and host work.

Per-chip device op sum (profiler, slowest chip): 486 ms over 161 ops.

| Op class | ms | % | ops |
|---|---|---|---|
| conv3d | 359.7 | 74 | 42 |
| layout: permute 28.8 + reshape 16.7 (depth-to-space/unpatchify) + slice 1.1 | 46.6 | 9.6 | 18 |
| norm (RMSNorm+SiLU, LayerNormDeviceOperation) | 45.7 | 9.4 | 37 |
| eltwise (BinaryNg residual adds 21.5 + 1 unary) | 22.6 | 4.6 | 21 |
| halo / neighbor_pad CCL (NpHalo) | 8.9 | 1.8 | 42 |
| rgb2yuv | 2.8 | 0.6 | 1 |

Not measured:
- the 4x8 profile itself: the CCL halo crosses more links on 4x8, and blocking 4,8 was tuned there;
- per-layer conv3d times (the CSV exists only as ops_perf.csv.gz on blx03 /var/tmp/fasth3/t96).

### Conv3d FLOPs and roofline (my estimate)

Decoder (reversed `decoder_blocks`), latent 19x34x60 at 1080p/145f. Each conv is k=3x3x3, 2*27*Cin*Cout FLOP per output voxel.

| Stage | Convs | Shape (T,H,W), ch | TFLOP |
|---|---|---|---|
| conv_in 128->1024 | 1 | 19x34x60 | 0.3 |
| res_x 2 @1024 | 4 | 19x34x60 | 8.8 |
| up compress_all /2 (1024->4096, d2s) | 1 | 19x34x60 | 8.8 |
| res_x 2 @512 | 4 | 37x68x120 | 17.1 |
| up compress_all x1 (512->4096, d2s) | 1 | 37x68x120 | 34.2 |
| **res_x 4 @512** | 8 | 73x136x240 | **269.8** |
| up compress_time (512->512, d2s) | 1 | 73x136x240 | 33.7 |
| **res_x 6 @256** | 12 | 145x136x240 | **201.0** |
| up compress_space (256->512, d2s) | 1 | 145x136x240 | 33.5 |
| **res_x 4 @128** | 8 | 145x272x480 | **134.0** |
| conv_out 128->48 | 1 | 145x272x480 | 6.3 |
| total | 42 | | **~747 TFLOP; 23.4 TFLOP per chip on 32 chips** |

- BH peak per chip (about 130 Tensix cores at 1.35 GHz, 4096 FLOP/cycle/core at LoFi): about 720 TFLOPS at LoFi, 360 at HiFi2 and 180 at HiFi4.
- The decoder convs run at HiFi4 with fp32 dest accumulation.
- Compute floor at HiFi4: 23.4 TFLOP / 180 TFLOPS ≈ **130 ms**; at HiFi2, about 65 ms.
- Measured conv3d time is 360 ms, which is about **36% of HiFi4 peak**.
- DRAM traffic is about 10-12 GB per chip at ~512 GB/s, which is about 25 ms. So decode is not DRAM-bound in theory.
- #84 shows the conv3d is limited by data movement (the reader's 27-tap gather, L1 traffic and packing), not by math: LoFi, a 4x cut in math passes, saved only 9%.
- Rough floor for the whole decode: about 130 ms of conv at HiFi4 plus about 40 ms of norm, eltwise and layout at bandwidth, which is **≈ 0.17-0.2 s**.
- Today it is 0.445 s, so about 0.25 s is recoverable in principle. Almost all of it is inside conv3d.

## 5. Ranked levers

The following are excluded:
- #98: fused neighbor_pad+conv3d. The halo ceiling is only 12.5 ms.
- #99: LTX_FUSE_NORM_ADD.
- the #170 knobs.
- tracing: no gain in #87/#96, and the DiffVAE result #241 agrees.
- EXACT_SHARD: already on by default on t48 (#97).

DiffVAE ideas that do not transfer:
- edge ordering and halo bricks: halo is already 9 ms;
- sync removal: eager equals traced, so there is no host gap left.

| Rank | Lever | Gain (est.) | Quality risk | Effort |
|---|---|---|---|---|
| 1 | **Conv3d kernel efficiency on the three big res_x stages** (603 of 747 TFLOP). First, a 4x8 per-layer profile to see which shapes run below about 30% of HiFi4. Then fix the reader: reuse the L1 input window across the kh/kw taps (a sliding window instead of a re-gather per tap); a larger Cout block for the 128- and 256-channel layers; a T-axis input reuse over the causal 3-frame window; re-tune the blocking per stage, not one global 4,8 (#17 found that one global H16xW2 re-sweep was slower, which says nothing about per-stage tuning). | 100-200 ms (goal: 50-60% of HiFi4) | none if the accumulation order stays the same; otherwise last-bit differences (PCC ≥ 0.9999 vs the unoptimized decode) | high (C++ conv3d reader/compute) |
| 2 | **HiFi2 for the up-block convs** (`LTX_VAE_CONV_FIDELITY=HiFi2`; the knob exists). Re-measure with the production blocking. #84's LoFi gain (-177 ms) was on the old fallback blocking, at a PSNR minimum of 45 dB. HiFi2 should keep PSNR above 50 dB. Also try `fp32_dest_acc_en=False` for the 128/256-channel stages (more dest tiles, so bigger output blocks). | 20-60 ms | low; numeric change; gate on PSNR ≥ 45 dB vs the HiFi4 decode on 5 seeds, then VBench | low (env A/B) |
| 3 | **Fold depth-to-space and unpatchify into the conv3d writer** (5 permutes and 5 reshapes = 45.5 ms). The conv after each upsample writes its output channels straight to their spatial positions, so no permute or reshape is needed. | 30-45 ms | bit-identical | medium |
| 4 | **Residual add in the conv3d epilogue** (20 BinaryNg = 21.5 ms). The second conv of each ResnetBlock adds the skip tensor in the packer instead of running a separate eltwise op. This differs from #99, which fused the norm with the add. | 15-20 ms | near bit-identical (one bf16 rounding less) | medium |
| 5 | **Hide the ~60 ms VAE stage tail and the audio decode**. Decode in temporal chunks so readback, YUV and export of chunk k overlap the device decode of chunk k+1. Separately, start the 0.39 s audio decode (#208) so it overlaps the video VAE instead of running after it, if a submesh or host split allows. | 40-60 ms (VAE tail); up to 0.39 s e2e (audio) | none (chunked decode with exact halos is bit-identical) | medium-high |

Realistic target: 0.445 s now, about 0.33 s after levers 2-4, and about 0.2-0.25 s with lever 1.
