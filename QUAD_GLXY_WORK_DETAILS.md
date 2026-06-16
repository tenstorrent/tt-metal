# Wan2.2 I2V on Quad BH Galaxy — Work Details

High-level gist of the quad-galaxy (4×32 Blackhole) image-to-video work: what was
done, the perf results, where the code lives, how to run it, and the
tt-inference-server changes that power the deployed endpoints.

Author: tvardhineni · Last updated: 2026-06-16

---

## 1. TL;DR

Three Wan2.2 I2V variants were brought up on single bh-galaxy and ported and optimized on **Quad BH Galaxy
(4×32 mesh, multi-host across 4 hosts)** at 720p / 81 frames:

| Model | Steps | Total (traced) | Endpoint | Notes |
|---|---|---|---|---|
| Wan2.2 base I2V | 40 | 44.6s | (reference only) | full CFG, baseline |
| **Wan2.2 Distill** (LightX2V) | 4 | **3.9s** | `ai-and-wan-3` | CFG baked in (guidance 1.0) |
| **AniSora V3.2** (anime) | 8 | **9.3s** | `ai-and-wan-2` | real CFG (guidance 3.5) |

The big wins: (a) actually running on the **quad config** (4×32, not 4×8),
(b) a **CFG fix** on the distill, and (c) a **fast VAE image-encode** path
(~6s → ~0.35s) shared by distill and AniSora. All validated for quality
(per-frame PCC + visual) — no quality regression.

---

## 2. Hardware / environment

- **Mesh:** 4×32 Blackhole Galaxy, multi-host over 4 hosts:
  - `OM-F2-GBH01` (172.16.104.122, rank 0) · `OM-F2-GBH02` (.123, rank 1)
  - `OM-F1-GBH02` (.121, rank 2) · `OM-F1-GBH01` (.120, rank 3)
- **Repo (shared across ranks):** `/data_bh/ubuntu/tt-metal` — *always work here*,
  not `~/tt-metal`, because the multi-host ranks read this path.
- **Python env:** `/data_bh/ubuntu/tt-metal/python_env` (`source python_env/bin/activate`)
- **Caches (shared, multi-model — names are legacy, not distill-only):**
  - `TT_DIT_CACHE_DIR=/data_bh/videos/dit_cache_wan_distill/` (prepared weights,
    namespaced per model: `Index-anisora-V3.2`, `Wan2.2-Distill-lightx2v-4step`, …)
  - `HF_HOME=/data_bh/videos/hf_data_wan_distill/` (HF downloads)
- **Branches:**
  - tt-metal: `tvardhineni/wan2_distill_quad` (based on Neel's `subtorus-wan` commits → main)
  - tt-inference-server: `wan2_distill_new`

---

## 3. What we did (chronological)

1. **Quad config enablement** — the original "20s on quad" was simply the
   pipeline defaulting to the **4×8** mesh. Adding the **(4,32)** parametrization
   (ring topology, `ring_params_8k`) + device **tracing** dropped distill 20s → 11s.
   This is the single biggest "fix" and was purely config, not model code.

2. **CFG fix (distill only)** — the distill bakes CFG in (`guidance_scale=1.0`),
   but `cfg_enabled` defaulted to `True`, so every step ran a wasted unconditional
   forward. Setting `cfg_enabled=False` removed 4 redundant forwards →
   11.2s → 9.3s, **bit-identical** output (lerp at weight 1.0 = cond).

3. **Fast VAE image-encode (~6s → ~0.35s)** — the dominant remaining cost. Three
   gated sub-optimizations (all quality-validated):
   - **Truncated encode** — I2V conditioning is image@frame0 + zeros; the causal
     VAE maps the zero tail to a steady-state latent, so we encode only the first
     ~33 pixel frames and replicate the tail latent (bit-equivalent to 81).
   - **Swept conv3d blockings** — rebuild the VAE encoder at the real resolution +
     truncated-T so it keys the tuned conv3d blocking tables instead of the slow
     `H/W=0` default. Paired with **`T_out_block=1`** to avoid a temporal-blocking
     "duplicate subject" artifact few-step models are sensitive to.
   - **On-device conditioning** — build the (mostly-zero) conditioning video on
     device via binary doubling instead of shipping it host→device.

4. **Ported to AniSora** — the three opts were factored into a reusable
   `FastImageEncodeMixin` so AniSora (and later LoRA) reuse the exact distill
   logic without touching the deployed distill code.

---

## 4. Performance (quad 4×32, 720p, 81 frames, traced)

| Stage | Base (40-step) | Distill (4-step) | AniSora (8-step) |
|---|---:|---:|---:|
| Text encoding | 0.17s | 0.18s | 0.17s |
| Image encoding | 6.61s | 0.36s | 0.34s |
| Denoising | 37.31s | 2.88s | 8.25s |
| VAE decoding | 0.44s | 0.50s | 0.50s |
| **Total** | **44.59s** | **3.93s** | **9.32s** |

Per-step denoise is ~same across models (~0.5s/forward, shared transformer); the
totals differ by step count + whether CFG is baked (distill, 1 fwd/step) or real
(AniSora/base, 2 fwd/step). Endpoint round-trips measured: distill ~3.8s, AniSora ~9.1s.

---

## 5. Key files (tt-metal, branch `tvardhineni/wan2_distill_quad`)

- `models/tt_dit/experimental/pipelines/fast_image_encode.py` — **shared mixin**
  (truncation + swept encoder + on-device cond), gated by per-model env flags.
- `models/tt_dit/experimental/pipelines/pipeline_wan_distill.py` — distill pipeline
  (CFG fix, inline fast-encode + its own flags).
- `models/tt_dit/experimental/pipelines/pipeline_anisora.py` — AniSora pipeline
  (inherits the mixin; flags `WAN_ANISORA_*`).
- `models/tt_dit/pipelines/wan/pipeline_wan_i2v.py` — base I2V; exposes the hooks
  (`_encode_frames_for`, `_encode_image_condition`, `_vae_encode_to_torch`).
- `models/tt_dit/utils/conv3d.py` — `set_force_t_out_block_1()` + swept blocking tables.
- Tests / harnesses:
  - `experimental/tests/test_performance_wan_distill_i2v.py` (traced perf)
  - `experimental/tests/test_performance_wan_anisora_i2v.py` (traced perf, `NUM_STEPS`)
  - `experimental/tests/test_validate_real_images.py` / `test_validate_anisora_images.py`
    (baseline-vs-fast PCC + mp4s)
  - `experimental/tests/test_pipeline_anisora.py`, `test_pipeline_lora.py` (added (4,32) row)
- `models/tt_dit/experimental/quad_glxy_wan2.2_distill.md` — distill doc (in-repo).

### Env flags (default OFF in code; ON by default in the server, see §7)
- Distill: `WAN_DISTILL_FAST_VAE_ENCODER`, `WAN_DISTILL_ENCODER_T_OUT_1`, `WAN_DISTILL_ONDEVICE_COND`
- AniSora: `WAN_ANISORA_FAST_VAE_ENCODER`, `WAN_ANISORA_ENCODER_T_OUT_1`, `WAN_ANISORA_ONDEVICE_COND`

> `ENCODER_T_OUT_1` MUST accompany `FAST_VAE_ENCODER` (the artifact guard).

---

## 6. How to run (multi-host quad)

Rankfiles / config live in `./config/` here and in `/data_bh/ubuntu/tt-metal/`
(`rankfile_quad4_f2f1`, `rank_bindings_quad4.yaml`, `mesh_graph_descriptor_quad4.textproto`).

**Traced perf (AniSora, 8 steps) — flags must be set on every rank (inner export):**
```bash
cd /data_bh/ubuntu/tt-metal && source python_env/bin/activate
tt-run \
  --rank-binding tests/tt_metal/distributed/config/32x4_quad_bh_galaxy_rank_bindings.yaml \
  --mpi-args "--host OM-F2-GBH01,OM-F2-GBH02,OM-F1-GBH02,OM-F1-GBH01 --rankfile /data_bh/ubuntu/tt-metal/rankfile_quad4_f2f1 --bind-to none --tag-output --merge-stderr-to-stdout" \
  bash -c "cd /data_bh/ubuntu/tt-metal && source python_env/bin/activate && \
    export TT_METAL_HOME=/data_bh/ubuntu/tt-metal PYTHONPATH=/data_bh/ubuntu/tt-metal \
           TT_DIT_CACHE_DIR=/data_bh/videos/dit_cache_wan_distill/ \
           HF_HOME=/data_bh/videos/hf_data_wan_distill/ TT_DIT_ALLOW_HF_DOWNLOAD=1 \
           NUM_STEPS=8 PROMPT_IMAGE=/data_bh/videos/anime_girl.png \
           WAN_ANISORA_FAST_VAE_ENCODER=1 WAN_ANISORA_ENCODER_T_OUT_1=1 WAN_ANISORA_ONDEVICE_COND=1 && \
    pytest -v -s --timeout 100000 \
      models/tt_dit/experimental/tests/test_performance_wan_anisora_i2v.py::test_pipeline_performance \
      -k 'resolution_720p and bh_4x32sp1tp0_ring'"
```
For the **distill** harness: same shape, use `test_performance_wan_distill_i2v.py`
and `WAN_DISTILL_*` flags (no `NUM_STEPS`; it's fixed 4). Helper script in
`scripts/run_anisora_fast_samples.sh`.

> ⚠️ Flags MUST be set inside the per-rank `bash -c` (or via a file each rank
> `source`s). mpirun `-x` forwarding through the tt-run wrapper proved unreliable
> — if flags don't reach ranks you'll see `WanEncoder.forward: T=81` (slow path).

**Reset a wedged fabric** (needed after a crash / hard-kill; ~60s):
```bash
cd /data_bh/ubuntu/tt-metal
MESH_DEVICE=QUAD_BH HOSTS="172.16.104.120,172.16.104.121,172.16.104.122,172.16.104.123" \
TT_METAL_HOME=/data_bh/ubuntu/tt-metal \
bash tests/ttnn/unit_tests/operations/ccl/blackhole_CI/exabox/multi_host/scripts/reset_chips.sh
```

---

## 7. tt-inference-server changes (branch `wan2_distill_new`)

What we added to make the endpoints serve the optimized pipelines. All in
`tt-media-server/tt_model_runners/dit_runners.py` (single file):

- **`TTWan22I2VDistillRunner`** (commit `6403bd0c`): enables `WAN_DISTILL_*` flags
  via `os.environ.setdefault` at `create_pipeline()` (default ON, deployment can
  pin to `0`); bumps `trace_region_size` to **200 MB** on large BH mesh; forces
  **4 steps**, guidance **1.0**.
- **`TTWan22I2VAniSoraRunner`** (commit `ca03ef5c`): same pattern with `WAN_ANISORA_*`
  flags; **200 MB** trace; forces **8 steps**, guidance **3.5** (the client's
  `num_inference_steps` is ignored, like the distill forces 4).

Model selection is by the `model_runner` setting → `runner_map` in
`tt-media-server/tt_model_runners/video_runner.py`:
- `tt-wan2.2-i2v-distill` → `TTWan22I2VDistillRunner`
- `tt-wan2.2-i2v-anisora` → `TTWan22I2VAniSoraRunner`

Multi-host launch uses `tt-run` + a rankfile and the `SP_MESH_4X32` mesh override
(see `video_runner.py` / `config/settings.py`). One built image serves **both**
models — pick the model per deployment via `model_runner` (one model per running
instance). Nothing shared was modified, so base I2V / T2V / Mochi are unaffected.

---

## 8. Endpoints & API

Both on 4×BH Galaxy. API key: `<API_KEY>` (internal — request from the team; do
not commit). Sync mode — the POST response *is* the mp4.

| Endpoint | Model | Steps | Gen time |
|---|---|---|---|
| https://ai-and-wan-2.n.cloud.tenstorrent.com | AniSora V3.2 | 8 | ~9s |
| https://ai-and-wan-3.n.cloud.tenstorrent.com | Wan2.2 Distill | 4 | ~4s |

```bash
# multipart upload form (simplest)
curl -X POST 'https://ai-and-wan-2.n.cloud.tenstorrent.com/v1/videos/generations/i2v/upload' \
  -H "Authorization: Bearer $API_KEY" \
  -F 'prompt=<text>' -F 'image=@anime_girl.png' -F 'seed=42' \
  --output out.mp4
```
JSON form (`/i2v`, base64 `image_prompts:[{image,frame_pos}]`) also works.
`num_inference_steps` only needs to satisfy the schema (≥12); the runner ignores
it (distill=4, AniSora=8).

---

## 9. Gotchas / troubleshooting

- **Slow path (T=81)** → flags didn't reach the ranks; set them per-rank (§6).
- **`mapping_result.success` / "could not fit physical topology"** → fabric
  wedged (often after a crash or hard-kill); run the reset (§6).
- **`trace_buffers_size <= trace_region_size` TT_FATAL** → trace region too small;
  the optimized traces need 200 MB.
- **Stale SHM (`tt_video_in size ... < expected`)** on the server → layout changed;
  `cd tt-media-server && python -m ipc.video_shm_bootstrap down` to reset.
- **Don't hard-kill multi-host jobs casually** — leaves the fabric wedged and the
  next run needs a reset. Kill the mpirun/prterun tree, then reset.
- **Cache dir name is misleading** — `*_wan_distill` is a shared multi-model cache;
  AniSora reads its own `Index-anisora-V3.2` namespace.

---

## 10. Commits (for reference)

**tt-metal** (`tvardhineni/wan2_distill_quad`):
- `339821cd846` distill quad bring-up + CFG fix + perf harness + README
- `8e43f58a0b3` distill fast VAE image-encode (6s → 0.37s; 9.3s → 3.98s)
- `ff84bafb40b` AniSora fast image-encode + quad (4×32) config
- `d0b9a83b46e` cherry-pick CI guard #46770 (don't push `:latest` from branches)

**tt-inference-server** (`wan2_distill_new`):
- `6403bd0c` enable distill fast-image-encode in the server
- `ca03ef5c` enable AniSora fast-image-encode + fixed 8-step

---

## 11. Possible next steps

- Port the fast-encode mixin to the **LoRA** I2V variant (quad row already added).
- Promote the encoder chunking into the base config so base I2V gets the encode win.
- Explore bf8 quantization on denoise (needs PCC check).
- Tighten the perf-harness target metrics now that real numbers are known.
