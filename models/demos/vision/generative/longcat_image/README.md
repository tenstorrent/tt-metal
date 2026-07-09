# LongCat-Image — end-to-end TTNN pipeline

Real, chained TTNN bring-up of **`meituan-longcat/LongCat-Image`** (a diffusers
text-to-image pipeline: Qwen2.5-VL 7B text encoder → LongCat MMDiT denoiser →
Flux AutoencoderKL) on a single Tenstorrent Blackhole (p150a, 32 GB).

This package chains the graduated per-component TTNN stubs in `_stubs/` into the
actual forward pass and compares the final output to the HF reference (Source A).

## Layout

```
longcat_image/
  _stubs/            graduated per-component TTNN ports (build(device, torch_module) -> callable)
  tt/pipeline.py     the ONE shared chained forward (both demo/ and tests/e2e/ import & call it)
  demo/              runnable per-task entrypoints (argparse + __main__)
    demo_text_to_image.py     Call 1: text -> image
    demo_image_edit.py        Call 2: image + text -> image (fires vision tower + VAE encoder)
    demo_server.py            interactive warm-server REPL (see "Warm server" below)
  tests/e2e/         end-to-end PCC + gate tests
    test_text_to_image_e2e.py     Gate 1/2/3 for Call 1
    test_image_edit_e2e.py        Gate 1/2/3 for Call 2
    test_trace_and_host_op.py     Command 3: host_op_selftest + trace_capture_selftest
  tests/pcc/         per-component PCC tests (Source B)
  e2e_plan.json      the planner output (task heads, coverage map, gates)
```

The chained forward lives in `tt/pipeline.py::LongCatImagePipelineTT`. The demo
and the e2e test call the SAME method, so a green test guarantees a working demo.

## Calls (task heads)

### Call 1 — text → image (`LongCatImagePipeline`)
`qwen2_v_l_model` (text encode → `hidden_states[-1]`) → `long_cat_image_transformer2_d_model`
(classifier-free-guidance denoise loop, FlowMatch Euler step + cfg-renorm on device)
→ `autoencoder_k_l` (VAE decode). Golden = the real HF pipeline denoise at the
identical seed / steps / guidance / size / prompt.

### Call 2 — image + text → image (`LongCatImageEditPipeline`)
Adds `autoencoder_k_l` **encode** (input image → latents) and the Qwen2.5-VL
**vision tower** (`qwen2_vision_transformer_pretrained_model` → `qwen2_v_l_vision_block` ×N →
`qwen2_v_l_patch_merger`), then reuses Call 1's DiT denoise + VAE decode. This is
the head that exercises the vision-tower and VAE-encoder graduated modules (which
never fire on the text→image path).

## Gates

- **Gate 1** — every routed graduated stub is real ttnn (no torch host-compute / HF
  orchestration in its hot path). Static scan + `host_op_selftest`.
- **Gate 2** — every graduated module on the call's critical path is INVOKED
  (directly, or subsumed by an invoked graduated container). No module wasted.
- **Gate 3** — final image PCC ≥ 0.95 vs the HF golden.

## Precision notes

The e2e error of this pipeline is dominated by **iterative-diffusion trajectory
divergence**, not per-matmul precision: over many CFG denoise steps the TT and the
independent HF-golden trajectories drift apart, and that drift — not rounding in any
one matmul — is what moves the final PCC. So the DiT runs in **bf16** with **bf8_b
linear weights** (HiFi4, `fp32_dest_acc_en`); dropping from fp32 barely moves PCC
(0.9947 → 0.9922 at the gate) while cutting per-step latency ~2×. The attention
score matmul stays fp32 for softmax stability. (A 16-bit bf16-limb path exists via
`stub.limb=True` but is **off** — it gives no gain here for the same
trajectory-divergence reason.) The VAE uses bf16 conv/group-norm weights with fp32
accumulation and fp32-internal group-norm. The golden is the bf16 HF pipeline at the
identical seed / steps / guidance / size / prompt (fp32 Qwen doesn't fit host RAM;
the TT text encoder clears ~1.0 vs the bf16 reference anyway).

## Run

The Call-1 demo defaults to **512 px / 50 steps / 512 tokens** and writes a PNG. Steps
(50), guidance (4.5) and token budget (512) match the HF reference defaults; the only
difference from HF is **resolution** — the demo uses 512 px where HF defaults to 1024 px
(pass `--size 1024` to match exactly; 1024 fits on a single 32 GB Blackhole — no OOM).
**Use ≥ 512 px** — 256 px is out-of-distribution for this 1024 px-class model and produces
noise. `--compare_golden` adds a slow CPU reference pass (minutes) just to print an accuracy
PCC; omit it for a normal run. (One HF default we do **not** match: `enable_prompt_rewrite`
— HF rewrites the prompt via the encoder's autoregressive `generate()` before encoding;
the TT path skips it, so images correspond to HF with prompt-rewrite off.)

```bash
# Call 1: text -> image  (writes a PNG). Defaults to 512px / 50 steps;
# add --size 1024 to match the HF reference resolution exactly.
./python_env/bin/python -m models.demos.vision.generative.longcat_image.demo.demo_text_to_image \
    --prompt "a photograph of a cat sitting on a red sofa" --out my_image.png
#   add --cq 2            to run the denoise loop under trace + 2 command queues
#   add --compare_golden  to also print e2e PCC vs the HF reference (slow)
#   add --profile         to print per-stage wall-clock timing (text-encode/denoise/vae-decode/total)

# Call 2: image + text -> image
./python_env/bin/python -m models.demos.vision.generative.longcat_image.demo.demo_image_edit \
    --image <path.jpg> --prompt "change the cat to a dog" --compare_golden
#   add --profile         to print per-stage wall-clock timing (edit-encode/vae-encode/denoise/vae-decode/total)

# Warm server: DiT+VAE resident + traced across requests, text encoder on a 2nd chip
# (needs 2 chips; opens a 1x2 MeshDevice). Defaults to 512px/50 steps; add --size 1024
# for HF resolution. See "Warm server" below.
./python_env/bin/python -m models.demos.vision.generative.longcat_image.demo.demo_server \
    --cq 2 --device_id 0 --text_encoder_device_id 1

# e2e gates (on device)
./python_env/bin/python -m pytest models/demos/vision/generative/longcat_image/tests/e2e/ -s

# per-step device latency (trace replay; LONGCAT_PERF_CQ=2 for trace+2CQ).
# NOTE: this harness profiles at a small bounded size (128px / 32 tokens), so its
# ms/step is a relative-optimization figure, NOT the full-resolution per-step cost
# (see the Performance section for the full-model 512/1024 numbers).
LONGCAT_PERF_CQ=1 ./python_env/bin/python -m pytest -s \
    models/demos/vision/generative/longcat_image/tests/e2e/test_text_to_image_perf.py
```

Gate caps (steps / image size / prompt token budget) are small by default for a fast
on-device gate and are applied identically to the TT run and the HF golden; the golden
is cached to disk. Override via `LONGCAT_E2E_{STEPS,SIZE,MAXLEN,GUIDANCE,PROMPT}`.

## Trace & command queues

The **text→image denoise loop is captured as a trace** by default (`_tt_denoise_traced`
in `tt/pipeline.py`): the whole per-step DiT compute — both CFG forwards, the guidance
combine, cfg-renorm, and the FlowMatch-Euler step — is captured once and `execute_trace`d
per step, removing per-op host dispatch. It falls back to eager on the image-edit path or
any trace error. Passing `--cq 2` (or building `LongCatImagePipelineTT(device, pipe,
num_cqs=2)` on a 2-CQ device) additionally runs a **trace + 2CQ** variant
(`_tt_denoise_traced_2cq`) that stages the next step's `temb`/`dt` on command-queue 1 as
DMA while queue 0 runs the trace. Note: CQ1 may only issue DMA, never a program/kernel —
device→device copies stay on CQ0. The 2CQ path is numerically identical to 1CQ (image
PCC 1.0) but ~parity in speed here, since the step is compute-bound with a tiny
prefetchable input.

## Warm server (Phase 0 of the QB2 porting plan)

`demo_text_to_image.py`/`demo_image_edit.py` are one-shot-per-process: every
invocation pays ~44s of stub-build + trace-capture before the first image, on
top of the ~60-70ms/step DiT replay cost. `LongCatImagePipelineTT.warmup()`
does that setup ONCE — it builds the DiT + VAE stubs and captures the DiT's
per-step trace with dummy shape-matched inputs — so later `run_text_to_image()`
calls whose `max_length`/`height`/`width`/`guidance_scale`/`enable_cfg_renorm`/
`cfg_renorm_min` match replay the resident trace instead of rebuilding. A
request that doesn't match falls back transparently to today's cold per-request
path, so correctness never depends on the caller remembering `warmup()`'s exact
arguments — only the throughput win does. Call `close()` on shutdown to release
the resident trace/stubs. Text-to-image (Call 1) only for now; image-edit
(Call 2) can reuse the same mechanism once its image-latents geometry is
validated warm too.

A resident (warm) DiT is ~12.5GB; a full Qwen text-encoder pass is another
~26-28GB (fp32) — measured on real hardware, the two do **not** fit together in
one chip's ~34GB DRAM. So a genuinely warm DiT needs the text encoder on its own
chip: `LongCatImagePipelineTT(..., text_encoder_device=<a second device>)` routes
it there. On its own chip the encoder ALSO stays resident across requests
(`_acquire_text_encoder`): one stub is built once, reused for a request's pos AND
neg branch, and kept alive between requests, so the ~26-28GB fp32 weights upload
**once** (on the first request) instead of build+upload+free per branch per
request. Measured effect: text-encode dropped from ~23s/branch to ~0.78s/branch
after the first request (it was upload/layout-bound, not compute-bound). On a
single shared device this resident mode is disabled (it would collide with the
DiT — the OOM above); there the one-stub-per-request reuse (pos+neg) still applies.
No mesh/CCL machinery is needed for the split — the encoder hands off a plain host
tensor to the denoise stage.

`demo/demo_server.py` puts this together as an interactive REPL. On a QB2
(fabric-connected multi-chip Blackhole) box it opens the two chips as ONE 1x2
`ttnn.MeshDevice` and carves it into two 1x1 submeshes (`create_submesh`), one
per role — chip 0 keeps DiT + VAE resident/warm, chip 1 holds the resident text
encoder. (Opening the two chips as independent `ttnn.open_device()` calls was
observed to hang on QB2 — tt-metal flags opening a subset of fabric-connected
mmio chips one at a time; the mesh open does it coherently.) **Needs two chips.**

```bash
./python_env/bin/python -m models.demos.vision.generative.longcat_image.demo.demo_server \
    --steps 24 --size 512 --max_length 512
#   --device_id / --text_encoder_device_id  pick which chip is which (must differ)
#   --cq 2                                   run the resident DiT trace under trace+2CQ
```

Per-stage timing is always on for the server (`profile=True`); pass `--profile`
to the other two demos, or set `LONGCAT_PROFILE=1`, to get the same
`[longcat-profile] <stage>: <ms>` breakdown from `LongCatImagePipelineTT`.

Measured steady-state on QB2 (1×2 mesh, `--cq 2`), after the first (warmup) request, at
the HF-reference 50 steps: **~47.7 s end-to-end at 512 px** and **~214.9 s at 1024 px**.
The full-model **denoise dominates** (~915 ms/step at 512, ~4.24 s/step at 1024);
text-encode is ~0.77 s/branch (resident, resolution-independent) and VAE decode 0.31 s
(512) / 1.35 s (1024). The one-time warmup (DiT trace capture + VAE warm-decode) and the
first request's encoder-weight upload are paid once, not per request. See the Performance
table below for the full breakdown.

## Performance

**Per-step DiT optimization ladder** — measured by the bounded per-step perf harness
(`test_text_to_image_perf.py`) at its small profiling size (**128 px / 32 tokens**); each
change e2e-PCC-verified (gate 0.95). These are **relative** speedups for the denoise step,
not the full-resolution per-step cost (see the full-model table below):

| Config | ms/step (128px/32tok) | speedup |
| --- | --- | --- |
| fp32 baseline | 125.1 | 1.00× |
| bf16 DiT | 73.1 | 1.71× |
| + attention core-grid 8×8 | 69.9 | 1.79× |
| + bf8_b linear weights | **60.1** | **2.08×** |
| trace + 2CQ | 60.3 | ≈ 1CQ (parity) |

**Full-model end-to-end** — measured on QB2 (1×2 mesh: DiT+VAE resident/warm on chip 0,
text encoder resident on chip 1), warm trace replay, `--cq 2`, steady-state (after the
first request's one-time warmup), at the HF-reference 50 steps:

| Setting | denoise / step | denoise (50) | text-enc ×2 | VAE | end-to-end |
| --- | --- | --- | --- | --- | --- |
| 512×512 / 50 steps | ~915 ms | 45.8 s | ~1.5 s | 0.31 s | **~47.7 s** |
| 1024×1024 / 50 steps (HF default) | ~4.24 s | 211.8 s | ~1.5 s | 1.35 s | **~214.9 s** |

Per-step scales ~4.6× from 512→1024 (image tokens 1024→4096; attention is O(n²) but only
part of the step, so the blend lands near 4.6×, not 16×). The **denoise dominates** (96–99%
of the run) — it is the optimization target. One-time warmup (not per request): DiT trace
capture ~46 s (512) / ~58 s (1024) + VAE warm-decode ~0.5 s (512) / ~15 s (1024); the first
request additionally pays ~23 s for the encoder weight upload. 1024 fits on one 32 GB chip
(no OOM).

## Results

<!-- FINAL_NUMBERS -->
- e2e PCC **0.9931** at the fast 1-step gate (256px); **0.9670** at 512px / 24 steps —
  both above the 0.95 gate. The lower value at more steps is the expected
  trajectory-divergence compounding, not a regression.
- The demo generates a coherent, prompt-accurate image (see the `e2e PCC=...` line
  printed on every run, pass or fail).
- Warm-server generation verified coherent + prompt-accurate at the HF-reference
  settings — **512 px and 1024 px, 50 steps** — across multiple back-to-back requests
  (visually confirmed, not just timings). This exercises the multi-request correctness
  fix: earlier, only the first warm request decoded correctly and every later one came
  out solid black, because the VAE's weights were allocated after the DiT trace was
  captured and got corrupted when the trace re-ran; `warmup()` now warms the VAE before
  capturing the trace (see `tt/pipeline.py`).
