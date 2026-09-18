# Stable Diffusion XL Base

Tenstorrent implementation of [Stable Diffusion XL Base 1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0) (SDXL)

### Supported Pipelines

- **Text-to-image** generates image from provided prompt
- **Image-to-image** (img2img) — generate variations from an input image
- **Inpainting** — fill masked region of an image guided by a prompt
- **Base + Refiner** pipeline — run SDXL Base followed by SDXL Refiner for higher quality output

### Supported resolutions:
 - 512x512
 - 1024x1024

### Supported Architectures
- Wormhole N150
- Wormhole N300
- Wormhole LoudBox/QuietBox
- Wormhole Galaxy
- Blackhole p150 (single chip, CI-covered)
- Blackhole Galaxy — validated ad hoc on 1 and 4 chips, see [Blackhole notes](#blackhole-notes)

## Blackhole notes

SDXL has single-chip Blackhole CI coverage (`bh_p150b_civ2`). There is **no multi-chip
Blackhole CI on any SKU** — the only multi-chip SDXL CI leg is Wormhole `wh_n300` (2 chips).
The notes below record an ad hoc validation run on a Blackhole Galaxy (`g11blx01`,
32 chips, 4x UBB trays) so the working configuration isn't rediscovered from scratch.

### Measured results (BH Galaxy, 1024x1024, 50 steps, guidance 5.0, `with_trace`, on-device VAE + encoders)

| Chips | `TT_VISIBLE_DEVICES` | Config | Denoising loop | On-device VAE | Wall clock |
|-------|----------------------|--------|----------------|---------------|-----------|
| 1     | `0`                  | `no_cfg_parallel` | 8.30 s / 1 prompt | 0.27 s | 78 s |
| 4     | `0,1,4,5`            | `no_cfg_parallel` | 8.38 s / 4 prompts | 0.27 s | 77 s |

4 chips denoise 4 prompts in the same wall time 1 chip takes for 1 prompt, i.e. ~linear
**throughput** scaling. It is not a single-image latency win: `determine_tensor_parallel`
asserts TP <= 2, and that "TP" is CFG parallelism (uncond / text on 2 chips joined by one
`all_gather`). Chips beyond that are independent data-parallel replicas.

### Required environment

```bash
export TT_METAL_CORE_GRID_OVERRIDE_TODEPRECATE=10,9   # asserted on any Galaxy; "7,7" on Wormhole
export TT_VISIBLE_DEVICES=0                            # or e.g. 0,1,4,5 for a 4-chip run
```

`TT_METAL_CORE_GRID_OVERRIDE_TODEPRECATE` is hard-asserted for Galaxy cluster types in
`tt/tt_sdxl_pipeline.py`, and `is_galaxy()` is cluster-type based — so it is required even
when only a **single chip** of a Galaxy is visible. (For reference, it yields an `11-10`
compute grid on this box, not `10,9`.)

### Gotchas

- **`pytest`'s timeout is disabled.** This directory's `conftest.py` sets
  `config.option.timeout = 0`, which overrides even a CLI `--timeout`. A hang will run
  forever with nothing to abort it. Wrap runs in an external `timeout`:
  `timeout 2400 pytest ...`
- **Reset the chips after any hung or killed run.** Killing the host process does not clean
  up device state; a wedged chip makes a later, unrelated run hang on its very first device
  op (observed as TRISCs stuck in `ckernel::tensix_sync()` during
  `compute_kernel_hw_startup`). `tt-smi -r <ids>` scoped to the chips you used clears it.
  Avoid bare `tt-smi -r` on a shared Galaxy — it resets all 32 chips.
- **Choose 4-chip groups by topology, not by PCI id.** On this Galaxy, PCI-consecutive
  groups such as `0,1,2,3` are *not* mutually connected. Verified 4-chip blocks:
  `0,1,4,5` / `2,3,6,7` / `8,9,12,13` / `10,11,14,15` / `16,17,20,21` / `18,19,22,23` /
  `24,25,28,29` / `26,27,30,31`. Dump the real adjacency with
  `ttnn._ttnn.cluster.serialize_cluster_descriptor()`.
- **Images land in `output/` relative to the pytest cwd** (normally the repo root), not
  under `demo/`. They are overwritten each run unless `--start-from` is varied.
- **The demos assert nothing about image correctness** — a green run only means no crash.
  Inspect the PNG. For scored runs use `tests/test_sdxl_accuracy.py` (CLIP/FID).
- **A single prompt on N chips wastes N-1 of them.** `demo.py` pads the batch with empty
  prompts and drops them, so 4 chips still yield 1 image. Use
  `tests/test_sdxl_accuracy.py --num-prompts=N` to exercise data parallelism properly.
- **To localize a hang**, `tt-triage` attaches to the live process and reports the stuck op,
  device, core and kernel callstack:
  `./tools/tt-triage.py --run=dump_running_operations` then `--run=dump_callstacks`.

### Not yet validated on Blackhole

- `use_cfg_parallel` (the `all_gather` + `FABRIC_1D` path) on Blackhole multi-chip.
- Blackhole QuietBox 2 (`P300_X2`, 4 chips, native `[2,2]` descriptor).
- 512x512 is explicitly skipped on Blackhole.


## Directory structure
stable_diffusion_xl_base/</br>
├── demo/          # End-to-end demo scripts (text2img, img2img, inpainting, base+refiner)</br>
├── tt/            # TT pipeline implementation</br>
├── vae/           # TT VAE implementation</br>
├── refiner/       # SDXL Refiner (separate UNet model)</br>
├── tests/         # Perf, Accuracy and PCC tests</br>
├── utils/         # accuracy utilities</br>
├── reference/</br>
├── conftest.py</br>
└── README.md


## How to Run

### Text-to-Image SDXL base (demo.py)

Example usage:
```
pytest models/demos/stable_diffusion_xl_base/demo/demo.py \
  -k "device_vae and device_encoders and with_trace and no_cfg_parallel and 1024x1024"
```

### Text-to-Image SDXL base+refiner (demo_base_and_refiner.py)

Example usage:
```
pytest models/demos/stable_diffusion_xl_base/demo/demo_base_and_refiner.py \
  -k "device_vae and device_encoders and with_trace and no_cfg_parallel and 1024x1024"
```
Note: Base + Refiner pipeline loads two separate models — SDXL Base (`stabilityai/stable-diffusion-xl-base-1.0`) and SDXL Refiner (`stabilityai/stable-diffusion-xl-refiner-1.0`), each with their own weights.

### Image-to-Image (demo_img2img.py)

Example usage:
```
pytest models/demos/stable_diffusion_xl_base/demo/demo_img2img.py \
  -k "device_vae and device_encoders and with_trace and no_cfg_parallel and 1024x1024"
```

Img to Img specific Params
- `strength`: How much to transform the input image. `0.0` = identical to input, `1.0` = fully new image
- `input_image`: Path to input image

### Inpainting (demo_inpainting.py)

Example usage:
```
pytest models/demos/stable_diffusion_xl_base/demo/demo_inpainting.py \
  -k "device_vae and device_encoders and with_trace and no_cfg_parallel and 1024x1024"
```

- `strength` - How much to transform the input image. `0.0` = identical to input, `1.0` = fully new image
- `input_image`: Path to input image
- `mask` : Path to input mask

Note: Inpainting uses a separate fine-tuned model (`diffusers/stable-diffusion-xl-1.0-inpainting-0.1`), not the SDXL Base weights.

##  Advanced settings

### Core Grid setting for Galaxy

For Whormhole Galaxy additional ENV variable is needed:
```
TT_METAL_CORE_GRID_OVERRIDE_TODEPRECATE=7,7
```

### CFG Parallel

Runs the UNet simultaneously for the positive and negative prompt on 2 chips, producing a single image with ~40% speedup compared to running both passes sequentially on 1 chip

usage example:
```
pytest models/demos/stable_diffusion_xl_base/demo/demo.py \
-k "device_vae and device_encoders and with_trace and use_cfg_parallel"
```

**Not possible for N150 and Galaxy**

## Performance per component

Device performance measured on Wormhole N150 (single UNet iteration):

| Component | Resolution | Device perf (ms) |
|---|---|---|
| UNet | 1024×1024 | ~190.3 |
| UNet | 512×512 | ~90.5 |
| Refiner UNet | 1024×1024 | ~244.1 |
| Refiner UNet | 512×512 | ~79.8 |
| VAE decode | 1024×1024 | ~663.1 |
| VAE decode | 512×512 | ~171.6 |
| VAE encode | 1024×1024 | ~324.3 |
| VAE encode | 512×512 | ~83.5 |
| CLIP encoder 1 | resolution independent | ~13.1 |
| CLIP encoder 2 | resolution independent | ~63.6 |

## E2E Performance per Architecture (SDXL base, 20 unet iterations)

| Architecture | CFG Parallel | E2E time (s) |
|---|---|---|
| N150 (1 chip) | no | 8.955 |
| N300 (2 chips) | yes | 5.158 |
