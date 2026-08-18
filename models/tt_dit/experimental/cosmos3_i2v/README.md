# Cosmos3 on Tenstorrent

NVIDIA Cosmos3 diffusion generation on TT hardware. One unified pipeline covers
image-to-video (I2V), text-to-video (T2V), and text-to-image (T2I); the mode is
picked from the shape of the conditioning you pass, not a flag.

The TT code is a thin layer over the **vendored** NVIDIA pipeline
(`reference/pipeline_cosmos3_omni.py`). We do not modify the denoise loop, the
tokenizer, or the scheduler math — we swap the 64-layer transformer trunk for an
on-device implementation and wrap the entrypoint.

## Quick start

```bash
export TT_DIT_CACHE_DIR=/home/<you>/.tt-dit-cache      # required — see "Cache"

python -m models.tt_dit.experimental.cosmos3_i2v.demo.generate \
  --image /path/to/frame.jpg \
  --prompt '<text or JSON-object prompt>' \
  --frames 189 --height 720 --width 1280 \
  --steps 35 --seed 0 \
  --out out.mp4
```

That's the whole command. Everything load-bearing is now a default:
`--pipeline native-cfg`, `--guidance-scale 6.0`, `--flow-shift 6.0`,
`--vae-{decoder,encoder}-t-chunk-size 4`, and SP-ring SDPA (see below) are all
on by default. Drop `--image` for T2V; drop `--image` and set `--frames 1` for
T2I (writes a PNG).

## The one supported path: `native-cfg`

`--pipeline` has a single choice, `native-cfg`, and here is why. On a full 4×8
mesh the single-mesh `native` trunk runs at sequence-parallel factor 4, which
produces **noise** — it is not a working configuration. `native-cfg` instead:

1. Splits the mesh along its **smaller** axis into two equal submeshes
   (4×8 → dual 2×8, each keeping TP=8, SP=2).
2. Runs the classifier-free-guidance **cond** pass on submesh A and the
   **uncond** pass on submesh B **concurrently** (two threads). This is the
   CFG-parallel win — per-step cost ≈ one trunk pass, not two.
3. On a 1-axis mesh (e.g. a 1×8 LoudBox) it transparently falls back to a single
   submesh — no CFG parallelism, but correct.

The single-mesh `native` builder still exists in the tree because `native-cfg`
*uses* it as its per-submesh builder and 1×8 fallback. It is just no longer
selectable as a top-level pipeline.

### SP-ring SDPA is required

`native-cfg` builds each trunk with `sequence_parallel factor=2`. The ring SDPA
and the token scatter/gather **must** match that sharding, or the denoised
latent is corrupt (clean conditioning frame, pure noise everywhere else). The
sp>1 path always runs the ring SDPA.

## How a run flows

```
run_cosmos3(pipe, prompt, image, num_frames, ...)          # pipelines/cosmos3_mode.py
  │
  ├─ resolve_mode(image, num_frames, enable_sound, action) # shape → ModelMode
  │      image + frames>1 → IMAGE2VIDEO
  │      no image + frames>1 → TEXT2VIDEO
  │      frames==1 → TEXT2IMAGE
  │      (audio / action modes are gated off)
  │
  ├─ load per-mode defaults   # cosmos3_defaults/<mode>/{sample_args,neg_prompts}.json
  │      caller kwargs win over defaults; negative_prompt=None → mode default
  │      merged dict filtered to the pipeline's real __call__ signature
  │
  └─ pipe(**merged)           # reference Cosmos3OmniPipeline.__call__
         │
         ├─ tokenize_prompt   # wrapped by cosmos3_prompt.py:
         │      JSON-object prompt → inject resolution / aspect_ratio /
         │      duration / fps, then serialize (NVIDIA-faithful spec injection).
         │      Free text passes through unchanged. No separate text encoder —
         │      the trunk consumes raw Qwen2 token IDs.
         │
         ├─ prepare_latents   # per-mode vision conditioning tensor:
         │      I2V: frame 0 = your image, kept fixed via the condition mask
         │      T2V/T2I: zeros
         │
         ├─ denoise loop (35 UniPC steps)
         │      each step: cond + uncond trunk passes (parallel across submeshes)
         │      → host CFG combine: uncond + scale·(cond − uncond)
         │      → scheduler.step
         │      (weights freed after the last step to give the VAE DRAM room)
         │
         └─ VAE decode (TT-NN) → postprocess → MP4 / PNG
```

### Prompts

A prompt is either free text or a **JSON object** (a "director's" caption:
`subjects[]`, `scene`, `camera`, `lighting`, …). If it parses as a JSON object,
`cosmos3_prompt.py` overwrites `resolution`/`aspect_ratio` (and `duration`/`fps`
for video) with the real generation specs and re-serializes — the specs are the
source of truth, not whatever those keys held. The negative prompt defaults to
the NVIDIA quality-control string for video modes (empty for T2I) and stays
byte-identical whether or not the positive is JSON.

## Environment

The measured-fastest configuration is the default — no env vars are needed
beyond the cache dir. Baked-in defaults: ring trunk CCLs (`FABRIC_1D_RING` +
`Topology.Ring`, −7% warm step; falls back to Linear under `TT_METAL_WATCHER`,
whose instrumentation overflows the 25600 B ACTIVE_ETH kernel-config buffer
with the ring router on Blackhole), grouped-K/V ring-joint SDPA (8x less ring
KV transport), SDPA HiFi2 + bf16 accumulation (unlocks the ring-joint op's
streaming compute path: 32.2 → 23.8 ms/layer at 720p), and SDPA chunks
q=256/k=512. Ring perturbs bf16 CCL accumulation order, so same-seed runs are
valid but not bit-reproducible across trace re-captures.

| Var | Default | Meaning |
|---|---|---|
| `TT_DIT_CACHE_DIR` | **unset — you must set it** | On-disk tensor cache root. Required: the second trunk loads weights only from cache (it has no state-dict source), so an unset dir fails the build. |

## Cache

Both trunks share `TT_DIT_CACHE_DIR`. On first build for a given
(parallel-config, submesh-shape, dtype) the trunk is built from the HF weights
and **written** to the cache; subsequent builds (including the second submesh in
the same run) load from it. A populated cache makes the build ~40s instead of
minutes.

## Performance notes

- 720p / 189 frames / 35 steps on a BH Galaxy 4×8: warm gen + decode ≈ 196s
  with the default configuration, a short negative prompt, resident weights,
  and per-generation trace re-capture (see `demo/serve.py`). Cold runs pay a
  one-time kernel JIT on the first step.
- The mesh is fully occupied: TP=8 × SP=2 per submesh × 2 submeshes = 32 chips.
  CFG parallelism is not spare capacity — it *is* what fills the second half of
  the mesh. There is no fourth parallel axis to add; the remaining win is moving
  the per-step CFG combine + scheduler step off the host.
- The broker enforces a job timeout; a full 720p run needs `run-bg -t 1800`.

### Fused MM+RS status

`RowParallelLinear(use_fused_mmrs=True)` can route Ring-topology reduce-scatters
through `minimal_matmul_strided_reduce_scatter_async`, but no cosmos3 trunk shape
is enabled in the `fused_mmrs_configs` table for the 10x10 power-clamped grid.
Every candidate passes the op's unit ladder (35-replay trace, two-instance
adjacency, batch protocol) yet corrupts the full traced trunk:

- und stream (M=2720): 9-step latent std 7.3 vs 0.7 — and at 12% of the gen RS
  payload the fusion win is negligible anyway.
- gen `to_out`/`to_add_out` (22144, 1024, 5120): 35-step visual smear (frame std
  46 vs 71 gold), isolated by per-shape trunk bisect; trunk-context-dependent and
  unreproduced at unit level.
- `down_proj` (22144, 3200, 5120): the windowed op's L1-resident MM window
  (512 KB/core, alive across the captured graph) clashes with downstream matmul
  CBs at trace capture. Re-adding it needs DRAM staging or window/CB budget
  coordination with every op captured alongside.

The regression suite for the underlying CCL fixes lives with the MMRS tests in
`tests/nightly/tg/ccl/`.

## Diagnostics (non-default)

- `--cfg-serial-dispatch` — run the two submeshes sequentially instead of in
  parallel threads. Isolates threading bugs from trunk/scheduler bugs.
- `--no-tt-vae` — host PyTorch VAE instead of TT-NN. Bisects TT-VAE regressions
  from trunk/scheduler regressions.
- `--output-type latent` — skip VAE decode; dump the raw post-denoise latent and
  its stats. Use to check whether the trunk (not the VAE) produced garbage.

## Layout

- `pipelines/cosmos3_mode.py` — `ModelMode`, `resolve_mode`, `run_cosmos3`, per-mode defaults.
- `pipelines/cosmos3_prompt.py` — JSON-object prompt spec injection.
- `pipelines/cosmos3_defaults/<mode>/` — per-mode sample args + negative prompt.
- `pipelines/pipeline_cosmos3_native_cfg.py` — dual-submesh CFG-parallel builder (the supported path).
- `pipelines/pipeline_cosmos3_native.py` — single-submesh engine: trunk proxy, TT VAE, UniPC scheduler fixes. Used by native-cfg.
- `reference/pipeline_cosmos3_omni.py` — vendored NVIDIA pipeline. Extend by subclass only; do not edit.
- `demo/generate.py` — one-shot CLI.
- `demo/serve.py` — long-lived interactive server (below).

## Interactive server

`demo/serve.py` builds the pipeline once and serves a single-page UI over HTTP.
It drives all three modes through `run_cosmos3`: pick text2image / text2video /
image2video, upload a conditioning image per request (or fall back to the launch
`--image`), and set prompt / negative / seed / steps / frames / guidance / size
in the form. Jobs are enqueued to one background worker (the device is
single-tenant, so the worker serializes it); the page polls `/status/<id>` for
live progress — phase, `step X/Y`, elapsed — and plays the MP4 (or shows the
PNG) inline when done.

```bash
ttq -H g03blx04 run-bg -t 86400 "cd ~/tt-metal && \
  TT_DIT_CACHE_DIR=~/.tt-dit-cache TT_METAL_HOME=$(pwd) PYTHONPATH=$(pwd) \
  python -m models.tt_dit.experimental.cosmos3_i2v.demo.serve \
    --image ~/car_driving.jpg --port 8080 --mesh-shape 4x8"

ssh -N -L 8080:localhost:8080 g03blx04     # from your Mac
open http://localhost:8080
```

Endpoints: `GET /` (UI), `POST /generate` (multipart → `{job_id}`),
`GET /status/<id>`, `GET /jobs`, `GET /media/<name>` (MP4/PNG, with HTTP Range).
