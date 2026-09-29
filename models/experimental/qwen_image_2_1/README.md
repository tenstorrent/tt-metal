# Qwen-Image-2.1 on one Blackhole p150

This implementation runs the Qwen-Image-2.1 text encoder, 7B single-stream DiT, and RGBA VAE on one p150 using TTNN. It generates 1024×1024 images with 40 Euler steps by default and supports image-conditioned editing. The default uses Tensix dispatch, bf16 DiT weights, bfp8 text encoder weights, LoFi text-to-image matmuls, HiFi2 editing matmuls and HiFi4 layer norms. Standard SDPA uses approximate exponentiation, matching the original port’s effective numerical policy; streaming text attention retains accurate exponentiation. The denoising loop is captured in one Metal trace.

The checkpoint is [Qwen/Qwen-Image-2.1](https://huggingface.co/Qwen/Qwen-Image-2.1/tree/790c92633540aa0cb11d9abf19eb46d861714758), revision `790c92633540aa0cb11d9abf19eb46d861714758`. It uses the Qwen Research License. This port is Apache-2.0 and was originally published by [Hyunggi Chang](https://github.com/changh95) in [changh95/qwen-image-2.1-p150](https://huggingface.co/changh95/qwen-image-2.1-p150/tree/d8befe24901fbdbf0feeb13964b6f175989f5758). It is a different architecture and checkpoint from the earlier Qwen-Image implementation under `models/tt_dit`.

## Run

From a built tt-metal checkout with an active Python environment:

```bash
export TT_METAL_HOME=$PWD PYTHONPATH=$PWD ARCH_NAME=blackhole
python -m pip install -r models/experimental/qwen_image_2_1/requirements.txt
hf download Qwen/Qwen-Image-2.1 --revision 790c92633540aa0cb11d9abf19eb46d861714758
python -m models.experimental.qwen_image_2_1.demo.demo --out image.png
```

The loader uses only the pinned snapshot. `QWEN_IMAGE_SNAPSHOT` may explicitly name a local checkpoint directory. For editing, add `--image input.png`; repeat the option for more condition images. The output follows the last condition image's aspect ratio at approximately one megapixel. Use `--repeat 2` to include a warm generation.

## Serve

```bash
export QWEN_IMAGE_TRACE_REGION=1073741824
python -m uvicorn models.experimental.qwen_image_2_1.server.app:app --host 127.0.0.1 --port 20000
curl http://127.0.0.1:20000/health
curl http://127.0.0.1:20000/predict -H 'Content-Type: application/json' \
  -d '{"prompt":"White furry llama with black sunglasses, smiling and happy, jumping","seed":42,"num_steps":40}'
```

`/health` reports loading until model initialization and warmup finish. `/info` reports the model configuration. `/predict` accepts `prompt`, `seed`, `num_steps` (1–100), `return_rgba`, and optional `images` containing base64 PNG/JPEG data. It returns base64 PNG data, dimensions and stage timings in milliseconds. The server serializes device work and gives waiting requests with resident prompts priority until its maximum wait time is reached. Shutdown drains queued work before releasing traces and closing the device.

This is an image API. The `/v1/models` endpoint is only a model-catalog readiness endpoint; it does not establish OpenAI-compatible generation support.

## Reference and tests

Reference dependencies are pinned separately from serving dependencies:

```bash
python -m pip install -r models/experimental/qwen_image_2_1/requirements-reference.txt
python -m models.experimental.qwen_image_2_1.reference.make_goldens --out generated/qwen_image_2_1/goldens
```

The reference generator defaults to CUDA with CPU offload. `--device cpu` runs the same reference on the CPU. Store generated tensors outside the source tree. Set `QWEN_IMAGE_GOLDENS` when using a different output directory. Reference artifacts must be generated from the pinned checkpoint before running numerical tests.

Host-only input and server checks:

```bash
pytest models/experimental/qwen_image_2_1/tests/test_config.py \
       models/experimental/qwen_image_2_1/tests/test_server_host.py \
       models/experimental/qwen_image_2_1/tests/test_host_math.py
```

Device tests cover the text encoder, DiT prefix/steps, VAE, complete traced denoising and editing. The source package's original reported accuracy covered one short text prompt and one- and two-image edits. Long prompts and three- or four-image edits require additional validation.

### Publication validation in progress

On one P150b with native base `bdfc59036eea3e988ba0e2374c12ca0c15c6c970`, the exact weekly registry command passed all seven real HTTP tests in 18 minutes 1 second, including cold-cache compilation, request isolation, prompt eviction and both aspect-ratio transitions. Three warmed 40-step requests averaged 22.40 seconds for text and 39.57 seconds for the two-image edit. These measurements cover the fixed examples below, not population image quality.

| Comparison | PCC | Criterion | Result |
| --- | ---: | ---: | --- |
| Text RGB vs published independent CUDA bf16 image | 0.98776 | 0.97 | Pass |
| Two-image edit RGB vs published P150 image | 0.97511 | 0.97 | Regression pass |
| Text final latent vs new independent fp32 Diffusers | 0.99744 | 0.99 | Pass |
| One-image edit final latent vs new independent fp32 Diffusers | 0.93839 | 0.98 | **Fail** |
| One-image edit RGB vs new independent fp32 Diffusers | 0.89603 | 0.90 | **Fail** |

The independent one-image editing failure remains unresolved. Its fp32 reference uses a different trajectory precision from the source package's CUDA bf16 reference; an independent bf16 comparison is pending. Published-P150 agreement does not establish independent editing correctness. The weekly test deliberately labels that check as a regression. The model remains under publication validation.

## Memory and dispatch

Prompt K/V slots and persistent VAE buffers are allocated before denoising traces. Shape changes release traces before resizing buffers and preparing the VAE. Replaying a trace must use the prompt buffers whose addresses it captured.

Tensix dispatch is the supported default for text-to-image and editing. The original package also tested ETH dispatch for text-to-image using unmerged runtime changes; ETH with editing had a reproducible hang. Current-main validation does not claim that optional mode.
