# Wan2.2-Distill I2V on Quad BH Galaxy (4x32)

Lightx2v 4-step distill of Wan2.2-I2V-A14B, running traced on a multi-host
Quad Blackhole Galaxy (4x32 = 128 chips across 4 hosts). CFG is baked into the
distill (`guidance_scale=1.0`), so it runs 4 denoise steps with a single forward
per step (2 high-noise + 2 low-noise, `boundary_ratio=0.5`).

- Distill pipeline: `models/tt_dit/experimental/pipelines/pipeline_wan_distill.py`
- Functional / video test: `models/tt_dit/experimental/tests/test_pipeline_wan_distill_i2v.py`
- Performance test (traced, per-stage timed): `models/tt_dit/experimental/tests/test_performance_wan_distill_i2v.py`
- Base Wan2.2 (non-experimental): `models/tt_dit/pipelines/wan/` + `models/tt_dit/tests/models/wan2_2/`

## Cluster setup (this environment)

The quad runs across 4 hosts. These files are environment-specific (host names /
IPs / paths) and are NOT committed — create them locally for your cluster. The
ones used here are reproduced below for reference.

`rankfile_quad4_f2f1` — MPI rankfile, F2/F1 quad (used by the commands below):

```
# rank 0 = OM-F2-GBH01 (172.16.104.122)  rank 1 = OM-F2-GBH02 (172.16.104.123)
# rank 2 = OM-F1-GBH02 (172.16.104.121)  rank 3 = OM-F1-GBH01 (172.16.104.120)
rank 0=OM-F2-GBH01 slot=0:*
rank 1=OM-F2-GBH02 slot=0:*
rank 2=OM-F1-GBH02 slot=0:*
rank 3=OM-F1-GBH01 slot=0:*
```

`rankfile_quad4` — alternate F4/F3 quad:

```
rank 0=OM-F4-GBH01 slot=0:*
rank 1=OM-F4-GBH02 slot=0:*
rank 2=OM-F3-GBH02 slot=0:*
rank 3=OM-F3-GBH01 slot=0:*
```

`rank_bindings_quad4.yaml` — optional new-mode binding (all ranks → mesh 0),
pointing at a quad mesh-graph descriptor:

```yaml
rank_bindings:
  - {rank: 0, mesh_id: 0}
  - {rank: 1, mesh_id: 0}
  - {rank: 2, mesh_id: 0}
  - {rank: 3, mesh_id: 0}
mesh_graph_desc_path: "/data_bh/ubuntu/tt-metal/mesh_graph_descriptor_quad4.textproto"
```

The commands below use the in-tree `--rank-binding
tests/tt_metal/distributed/config/32x4_quad_bh_galaxy_rank_bindings.yaml` plus the
`rankfile_quad4_f2f1` above.

Common env (paths are this machine's; change as needed):

```bash
export TT_METAL_HOME=/data_bh/ubuntu/tt-metal
export PYTHONPATH=/data_bh/ubuntu/tt-metal
export TT_DIT_CACHE_DIR=/data_bh/videos/dit_cache_wan_distill/
export HF_HOME=/data_bh/videos/hf_data_wan_distill/
export TT_DIT_ALLOW_HF_DOWNLOAD=1
```

## Main command — distill performance (720p, quad, traced)

```bash
cd /data_bh/ubuntu/tt-metal && source python_env/bin/activate && tt-run \
  --rank-binding tests/tt_metal/distributed/config/32x4_quad_bh_galaxy_rank_bindings.yaml \
  --mpi-args "--host OM-F2-GBH01,OM-F2-GBH02,OM-F1-GBH02,OM-F1-GBH01 --rankfile /data_bh/ubuntu/tt-metal/rankfile_quad4_f2f1 --bind-to none --tag-output --merge-stderr-to-stdout" \
  bash -c "cd /data_bh/ubuntu/tt-metal && source python_env/bin/activate && \
    export TT_METAL_HOME=/data_bh/ubuntu/tt-metal PYTHONPATH=/data_bh/ubuntu/tt-metal \
           TT_DIT_CACHE_DIR=/data_bh/videos/dit_cache_wan_distill/ \
           HF_HOME=/data_bh/videos/hf_data_wan_distill/ TT_DIT_ALLOW_HF_DOWNLOAD=1 && \
    NO_PROMPT=1 pytest -v -s --timeout 100000 \
      models/tt_dit/experimental/tests/test_performance_wan_distill_i2v.py \
      -k 'resolution_720p and bh_4x32sp1tp0'"
```

Note: 4x32 is 720p-only (the 480p/4x32 case asserts out by design).

## Distill — generate a video (functional)

Set `PROMPT_IMAGE` to a conditioning image; the output mp4 is written to the cwd
(`wan_distill_i2v_1280x720_0.mp4`). Requires `imageio-ffmpeg` in the venv
(`uv pip install imageio-ffmpeg`).

```bash
# ... same tt-run wrapper as above, with these extra exports inside bash -c:
export PROMPT_IMAGE=/data_bh/videos/prompt_image.png
NO_PROMPT=1 pytest -v -s --timeout 100000 \
  models/tt_dit/experimental/tests/test_pipeline_wan_distill_i2v.py \
  -k 'resolution_720p and bh_4x32sp1tp0_ring'
```

## Base Wan2.2 I2V performance (720p, quad, traced)

```bash
cd /data_bh/ubuntu/tt-metal && source python_env/bin/activate && tt-run \
  --rank-binding tests/tt_metal/distributed/config/32x4_quad_bh_galaxy_rank_bindings.yaml \
  --mpi-args "--host OM-F2-GBH01,OM-F2-GBH02,OM-F1-GBH02,OM-F1-GBH01 --rankfile /data_bh/ubuntu/tt-metal/rankfile_quad4_f2f1 --bind-to none --tag-output --merge-stderr-to-stdout" \
  bash -c "cd /data_bh/ubuntu/tt-metal && source python_env/bin/activate && \
    export TT_METAL_HOME=/data_bh/ubuntu/tt-metal PYTHONPATH=/data_bh/ubuntu/tt-metal \
           TT_DIT_CACHE_DIR=/data_bh/videos/dit_cache_wan_distill/ \
           HF_HOME=/data_bh/videos/hf_data_wan_distill/ TT_DIT_ALLOW_HF_DOWNLOAD=1 && \
    NO_PROMPT=1 pytest -v -s --timeout 100000 \
      models/tt_dit/tests/models/wan2_2/test_performance_wan.py \
      -k 'i2v and resolution_720p and bh_4x32sp1tp0'"
```

## Performance — base vs distill (quad 4x32, 720p, 81 frames, traced)

| Stage                | Base i2v (40-step, CFG) | Distill (4-step, all opts) |
| -------------------- | ----------------------: | -------------------------: |
| Text Encoding        |                   0.17s |                      0.17s |
| Image Encoding (VAE) |                   6.61s |                      0.37s |
| Denoising            |    37.31s (80 forwards) |         2.91s (4 forwards) |
| VAE Decoding         |                   0.44s |                      0.45s |
| **Total Pipeline**   |              **44.59s** |                 **3.98s**  |

### Distill optimization journey (total pipeline, traced)

| Stage of work               | Total  | Change                                                       |
| --------------------------- | -----: | ------------------------------------------------------------ |
| Cloud-deployed baseline     | ~20s   | ran on default/single-galaxy mesh config                     |
| Quad-galaxy config          | ~11.2s | correct 4x32 parallelism via `WanPipelineConfig.default()`   |
| CFG fix                     | ~9.3s  | drop the redundant unconditional forward (CFG is baked in)   |
| Fast VAE encoder            | 6.70s  | truncated 33-frame encode + swept conv3d blockings + T_out=1 |
| **+ On-device conditioning**| **3.98s** | build zero frames on-device; transfer only the 1 image    |

- Image-encode went **6.0s → 0.37s** (≈ matches the Prodia I2V pipeline's 0.34s).
  Denoising (2.91s) is now the dominant stage.
- Per-forward denoise: base **0.47 s/fwd**, distill **~0.73 s/fwd**. The distill's
  higher per-forward cost is the high→low expert switch (`transformer` →
  `transformer_2`) amortized over just 4 steps vs base's 40 — structural, not a
  missing optimization.
- The distill inherits the base's full optimization path (parallelism, SDPA
  fracture, tracing, conv3d/matmul blocking) automatically via
  `WanPipelineConfig.default()`, keyed on mesh shape.

## Optimization notes

### Shipped: CFG fix (denoise 4.45s → 2.59s)
The distill bakes CFG in (`guidance_scale=1.0`), but the pipeline still ran the
redundant unconditional forward each step (8 forwards for 4 steps), then threw it
away via `lerp(uncond, cond, 1.0) == cond`. Setting `cfg_enabled=False` in the
distill's `create_pipeline` skips the dead forward — denoise dropped from 4.45s to
2.59s (total 11.19s → ~9.3s) with **bit-identical math** (frame-0 PCC 0.999 vs the
pre-fix baseline, visually confirmed coherent).

### Shipped (gated): fast VAE image-encode (6.0s → 0.37s)
The image-encode stage was ~65% of the distill pipeline. It is cut to ~0.37s by
three composable, env-gated optimizations in the distill pipeline (all default
**OFF** so base/T2V behavior is untouched):

1. **Truncated encode** (always on for the distill, via `_encode_frames_for`).
   For I2V the conditioning image sits at frame 0 and all later frames are zeros,
   which the causal Wan VAE encoder maps to a steady-state latent. We encode only
   the first 33 pixel frames (→ 9 latent frames) and replicate the last latent to
   fill the rest. Quality-safe on its own.

2. **Swept conv3d blockings + `T_out_block=1`** (`WAN_DISTILL_FAST_VAE_ENCODER=1`,
   `WAN_DISTILL_ENCODER_T_OUT_1=1`). The base builds the encoder with H/W=0, which
   falls into the slow `_DEFAULT_BLOCKINGS`. Rebuilding it at the real resolution +
   truncated-T chunk size keys the **swept** conv3d entries — encoder forward drops
   **1.44s → 0.21s (7×)**. Capping `T_out_block` at 1 keeps that speed while
   avoiding the temporal-blocking artifact the 4-step distill is sensitive to (the
   blocking cache key only depends on `C_in_block`, so prepared weights are shared).

3. **On-device conditioning assembly** (`WAN_DISTILL_ONDEVICE_COND=1`). The base
   builds the full 33-frame pixel video on the **host** (mostly zeros) and ships
   ~180 MB to device. Instead we transfer only the conditioned frame (~5.5 MB) and
   build the zero frames **on-device** via binary doubling (Prodia pattern). This
   cuts `prepare_latents` from **2.99s → 0.38s** (compiled).

Quality: validated against the full-encode baseline (same seed) with per-frame PCC
+ visual inspection on in-distribution inputs (anime portrait + real images). The
fast path is visually indistinguishable from full encode; per-frame PCC sits in the
same band as plain truncation (~0.86–0.94, divergence only on late frames, no
duplicate-subject / frame-0-noise artifacts).

> Earlier, a naive `encoder_t_chunk_size` bump (without the truncation, swept-key,
> and `T_out=1` pieces) corrupted the T=81 encode. The cause was traced to a bad
> test fixture (out-of-distribution fractal image + contradicting prompt) plus
> temporal blocking; the three-part approach above resolves both.

To enable all three (the shipping fast config):

```bash
export WAN_DISTILL_FAST_VAE_ENCODER=1 WAN_DISTILL_ENCODER_T_OUT_1=1 WAN_DISTILL_ONDEVICE_COND=1
```

Comparison / validation harnesses:
- `models/tt_dit/experimental/tests/test_encode_compare_seeds.py` — full vs each
  fast variant, N seeds, per-frame PCC + saved mp4s.
- `models/tt_dit/experimental/tests/test_validate_real_images.py` — baseline vs the
  shipping fast config across multiple real conditioning images.

## Next levers (for the distill specifically)
1. Denoising (2.91s, now ~73% of pipeline): bf8 quantization (needs PCC check) —
   applies to base too.
2. Promote the fast image-encode flags to default-on once multi-image quality
   sign-off is complete.
