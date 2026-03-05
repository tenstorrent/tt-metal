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

### Core Grid setting for Galaxy and Blackhole

For Whormhole Galaxy, BlackHole P150/P300/T3K/Galaxy additional ENV variable is needed:
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

**Not possible for N150, P100/P150 and Galaxy**

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
