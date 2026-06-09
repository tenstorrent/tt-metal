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

| Stage                | Base i2v (40-step, CFG) | Distill (4-step, CFG fix) |
| -------------------- | ----------------------: | ------------------------: |
| Text Encoding        |                   0.17s |                     0.17s |
| Image Encoding (VAE) |                   6.61s |                     6.00s |
| Denoising            |    37.31s (80 forwards) |        2.59s (4 forwards) |
| VAE Decoding         |                   0.44s |                     0.42s |
| **Total Pipeline**   |              **44.59s** |                **~9.3s**  |

- Per-forward denoise: base **0.47 s/fwd**, distill **0.65 s/fwd**. The distill's
  higher per-forward cost is the high→low expert switch (`transformer` →
  `transformer_2`) amortized over just 4 steps vs base's 40 — structural, not a
  missing optimization.
- Image-encode (~6s) is the **same shared, un-tuned VAE encoder path** for both.
  It's only ~15% of the base pipeline but ~65% of the distill, so it's the #1
  remaining lever for the distill.
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

### Tried but REVERTED: encoder chunking (~8s, failed quality)
We attempted to cut the ~6s image-encode by building the VAE encoder with real
height/width and a larger `encoder_t_chunk_size` (16 instead of the default 4), so
it would (a) hit the swept 4x32 conv3d blocking entries and (b) run ~6 passes
instead of ~20. This **did** drop the pipeline to ~8.0–8.4s.

**However, a visual/PCC check showed it corrupted the conditioning encode** on the
production T=81 / 4x32 path: frame 0 (the decoded conditioning) came out as noise
(PCC ~0.57 then garbage), and later frames were flat grey. The swept 4x32 encoder
blockings / chunked-cache path were validated for a T=33 sweep on 4x8, not the
T=81 path used here. **Because output quality broke, the encoder change was fully
reverted** — only the CFG fix remains. Properly capturing this win needs a conv3d
sweep at the real T=81 shapes (or a cheaper conditioning path), not just a chunk
bump.

## Next levers (for the distill specifically)
1. Image-encode (~6s, ~65% of pipeline): conv3d sweep at production T=81 shapes,
   or reduce the number of conditioning frames encoded.
2. bf8 quantization on denoise (needs PCC check) — applies to base too.
