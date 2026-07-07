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

The DiT runs on device with **16-bit bf16-limb emulated matmuls** (`stub.limb=True`),
not plain bf16/fp32: the Tensix multiplier keeps only ~11 mantissa bits even on
fp32 inputs, and classifier-free guidance amplifies the per-branch noise error by
`guidance_scale`, so the limb path (memory-neutral, ~16-bit) is required to hold
the multi-step CFG latent above the gate. The VAE runs fully fp32 (its group_norm
must be fp32). The golden uses fp32 transformer + VAE (matching the Source-B
per-component PCC methodology) and the bf16 Qwen2.5-VL text encoder (fp32 Qwen
doesn't fit host RAM; the TT text encoder clears ~1.0 vs it anyway).

## Run

```bash
# Call 1 demo (writes a PNG, prints e2e PCC vs golden)
python -m models.demos.vision.generative.longcat_image.demo.demo_text_to_image \
    --size 256 --steps 2 --guidance 4.5 --compare_golden

# Call 2 demo
python -m models.demos.vision.generative.longcat_image.demo.demo_image_edit \
    --image <path.jpg> --prompt "change the cat to a dog" --compare_golden

# e2e gates (on device)
./python_env/bin/python -m pytest models/demos/vision/generative/longcat_image/tests/e2e/ -s

# Command 3: fully-on-device + trace/2CQ checks
./python_env/bin/python -m pytest \
    models/demos/vision/generative/longcat_image/tests/e2e/test_trace_and_host_op.py -s
```

Gate caps (steps / image size / prompt token budget) are small by default for a
fast on-device gate and are applied identically to the TT run and the HF golden;
the golden is cached to disk. Override via `LONGCAT_E2E_{STEPS,SIZE,MAXLEN,GUIDANCE,PROMPT}`.

## Results

<!-- FINAL_NUMBERS -->
See the test output line `e2e PCC=...` (printed on every run, pass or fail).
