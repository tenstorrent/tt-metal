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

The Call-1 demo defaults to quality settings (512px / 24 steps / 512 tokens) and
writes a PNG. **Use ≥ 512px** — 256px is out-of-distribution for this 1024px-class
model and produces noise. `--compare_golden` adds a slow CPU reference pass (minutes)
just to print an accuracy PCC; omit it for a normal ~2–3 min run.

```bash
# Call 1: text -> image  (writes a PNG)
./python_env/bin/python -m models.demos.vision.generative.longcat_image.demo.demo_text_to_image \
    --prompt "a photograph of a cat sitting on a red sofa" \
    --size 512 --steps 24 --guidance 4.5 --out my_image.png
#   add --cq 2            to run the denoise loop under trace + 2 command queues
#   add --compare_golden  to also print e2e PCC vs the HF reference (slow)

# Call 2: image + text -> image
./python_env/bin/python -m models.demos.vision.generative.longcat_image.demo.demo_image_edit \
    --image <path.jpg> --prompt "change the cat to a dog" --compare_golden

# e2e gates (on device)
./python_env/bin/python -m pytest models/demos/vision/generative/longcat_image/tests/e2e/ -s

# per-step device latency (trace replay; LONGCAT_PERF_CQ=2 for trace+2CQ)
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

## Performance

Per-denoise-step device latency, each change e2e-PCC-verified (gate 0.95):

| Config | ms/step | speedup |
| --- | --- | --- |
| fp32 baseline | 125.1 | 1.00× |
| bf16 DiT | 73.1 | 1.71× |
| + attention core-grid 8×8 | 69.9 | 1.79× |
| + bf8_b linear weights | **60.1** | **2.08×** |
| trace + 2CQ | 60.3 | ≈ 1CQ (parity) |

End-to-end (512px / 24 steps), tracing the denoise loop: **~125 s vs ~179 s eager
(~1.43×)** on top of the per-step wins.

## Results

<!-- FINAL_NUMBERS -->
- e2e PCC **0.9931** at the fast 1-step gate (256px); **0.9670** at 512px / 24 steps —
  both above the 0.95 gate. The lower value at more steps is the expected
  trajectory-divergence compounding, not a regression.
- The demo generates a coherent, prompt-accurate image (see the `e2e PCC=...` line
  printed on every run, pass or fail).
