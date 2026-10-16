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

`/health` reports loading until model initialization and warmup finish. `/info` reports the model configuration. `/predict` accepts `prompt`, `seed`, `num_steps` (2–100, defaulting to `QWEN_IMAGE_STEPS`, which defaults to 40), `return_rgba`, and optional `images` containing base64 PNG/JPEG data. The terminal-shift scheduler requires at least two steps; one-step requests, server defaults, and CLI runs are rejected. It returns base64 PNG data, dimensions and stage timings in milliseconds. The server serializes device work and gives waiting requests with resident prompts priority until its maximum wait time is reached. `QWEN_QUEUE_MAX_PENDING` bounds the waiting queue (default 8); excess requests receive HTTP 503. Extreme aspect ratios that round to zero pixels receive HTTP 400 before queue admission. Each VAE convolution retains at most four prepared geometries and releases older device weights and biases. Shutdown drains admitted work before releasing traces and closing the device.

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

### Publication validation

[The P150 weekly run](https://github.com/tenstorrent/tt-metal/actions/runs/36621858813) at source `b9cf1f7edf721d9a053cf2e6f241e8f3930882ad` completed the exact registry command in **17 minutes 17 seconds**, including **66 seconds of cold pinned-checkpoint setup**, cold kernel compilation, all seven real HTTP tests, reporting and cleanup. It covers request isolation, prompt eviction, and portrait/landscape transitions. One warmup followed by three completed 40-step requests gave mean wall times of **21.35 seconds for text** and **36.58 seconds for the two-image edit**. The weekly allowance is 30 minutes. These measurements cover the fixed examples below, not population image quality.

Independent references use the pinned Diffusers revision in `requirements-reference.txt`, checkpoint revision above, seed 42, CPU-generated bf16 initial noise, and 40 Euler steps. The one-image edit uses native CPU bf16 throughout the reference, with VAE convolution weights in channels-last format. A separate real-weight VAE comparison qualified that layout at latent/decoded PCC 0.999994/0.999939 against a predeclared 0.9999 gate; it is not bit-exact. No operators or arithmetic dtypes were substituted.

| Independent full-chain case | Latent PCC | RGB PCC | Gates (latent / RGB) | Result |
| --- | ---: | ---: | --- | --- |
| Text, fp32 reference | 0.997432 | 0.992085 | 0.99 / 0.90 | Pass |
| One-image edit, bf16 reference | 0.999661 | 0.999431 | 0.98 / 0.90 | Pass |
| Two-image edit, fp32 reference | 0.992201 | 0.960086 | 0.98 / 0.90 | Pass |
| One-image edit, fp32 reference | 0.938388 | 0.895834 | 0.98 / 0.90 | **Fail** |

The bf16 one-edit comparison meets the original, unchanged editing gates. The fp32 failure remains: its trajectory differs from the model's bf16 arithmetic, and the independent fp32/bf16 references also diverge. The bf16 scores above use an independent float64 PCC calculation; the original float32 scorer reports 0.999663/0.999716 and reaches the same verdict. All 40 saved bf16 states are finite; seeded noise and step tensors were verified. Device repeats are bit-identical. This establishes agreement for one fixed edit with CPU bf16, not CUDA equivalence or broad task quality.

The weekly HTTP checks separately compare text RGB to the publisher's CUDA artifact (PCC **0.987758**, gate 0.97) and the two-image edit to the publisher's P150 artifact (PCC **0.975105**, gate 0.97). The latter is explicitly a regression check, not an independent correctness oracle. The original author's exact CUDA reference commit and editing tensors were not published.

To generate the one-image bf16 reference with the qualified CPU VAE layout, use the original package's `media/edit_ref_llama.jpg` (SHA256 `acb2dba9fe4966197f18c16f166dcfb9c717b974b6fa08c2004c111d8fc17b37`):

```bash
python -m models.experimental.qwen_image_2_1.reference.make_goldens_edit \
  --device cpu --dtype bfloat16 --vae-memory-format channels_last \
  --image /path/to/edit_ref_llama.jpg --out /path/to/goldens/edit
```

Native bf16 execution on an AVX2-only CPU is slow: the measured full reference took about seven hours. Changing the reference dtype or memory format can alter the generated image and must be recorded with its results.

## Memory and dispatch

Prompt K/V slots and persistent VAE buffers are allocated before denoising traces. Shape changes release traces before resizing buffers and preparing the VAE. Replaying a trace must use the prompt buffers whose addresses it captured.

Tensix dispatch is the supported default for text-to-image and editing. The original package also tested ETH dispatch for text-to-image using unmerged runtime changes; ETH with editing had a reproducible hang. Current-main validation does not claim that optional mode.
