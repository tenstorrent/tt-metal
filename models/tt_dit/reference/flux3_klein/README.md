# FLUX 3 Image Klein: vendored PyTorch reference

The torch implementation of FLUX 3 Image Klein that the tt-metal port (`models/tt_dit/models/transformers/flux3_klein/`,
`models/tt_dit/pipelines/flux3_klein/`) is verified against. It is vendored, not wrapped: the `diffusers` in `python_env`
has no FLUX 3 transformer and the upstream package pins (`vllm`, `triton`, `numpy 2.x`) conflict with `python_env`.
The package imports torch, numpy, Pillow, safetensors and (in `text_encoder.py` / `pipeline.py` only) transformers;
it never imports ttnn. It runs on CPU and on CUDA.

## Provenance

- Upstream: <https://github.com/black-forest-labs/flux-3-image-klein>, commit `e53f4ea`
  ("Merge pull request #3 from black-forest-labs/stephen37/run-elsewhere-links"), directory `src/flux3/`.
- Upstream file names are kept so that `diff -u <upstream>/src/flux3/<f>.py <f>.py` stays small.
- Each vendored module carries the upstream copyright (`© 2026 Black Forest Labs, Inc.`), the Apache-2.0 SPDX tag and
  an "Adapted from ... at e53f4ea" line, as required by Apache-2.0 section 4 for modified files.
- The whole package was passed through the repository's pre-commit (black / isort / autoflake at line length 120),
  which reflows some upstream lines without changing any statement.

## Files and modifications

| file | upstream | modifications beyond the header and formatting |
|---|---|---|
| `dit.py` | `src/flux3/dit.py` | `rmsnorm_rope`: `torch.autocast(device_type="cuda", enabled=False)` -> `device_type=x.device.type`. Nothing else; the SDPA backend choice (`CUDNN_ATTENTION` on CUDA, `MATH` elsewhere) is upstream's. |
| `autoencoder.py` | `src/flux3/autoencoder.py` | none (pure torch; the BFL `ae.safetensors` layout). |
| `sampling.py` | `src/flux3/sampling.py` | `from flux3.dit` -> `from .dit`; the two `@torch.autocast("cuda", dtype=torch.bfloat16)` decorators become `@autocast_bf16`, a wrapper that opens the same bf16 autocast on the device of `packed.data["image"]` (identical on CUDA; on CPU it enables CPU bf16 autocast instead of silently disabling autocast). Note that CPU autocast keeps `layer_norm` and `softmax` in bf16 where CUDA autocast runs them in fp32, so a CPU run differs from a CUDA run by more than the SDPA backend; the GPU goldens are the reference, the CPU run is the interim oracle. |
| `pipeline.py` | `src/flux3/pipeline.py` | imports made relative; the `WEIGHTS = Path(os.environ["WEIGHTS_DIR"])` table and the `QUANT` variants map are removed: `dit_path(weights_dir, model)` takes the weights directory explicitly and resolves bf16 only. `MODELS` (Hub repository, default steps, guidance) is kept verbatim. `Pipeline(device=...)` defaults to CUDA when available, else CPU. `encode_references`, `pack`, `scaled_ids`, the latent helpers and the reference sizing (`reference_canvas`, `prepare_reference`, `prepare_multi_reference`, `default_size`) are untouched. |
| `text_encoder.py` | `src/flux3/text_encoder.py` | `TextEncoder(device=...)` defaults to CUDA when available, else CPU. The layer taps, padding and position handling are untouched; this file is the contract the device text encoder must meet. |
| `checkpoint.py` | `src/flux3/checkpoint.py` | keeps `safetensors_file`, `checkpoint_keys`, `is_flow_map`, `load_checkpoint` / `_load_safetensors` (exact key match, floating tensors cast to `dtype`, integer buffers kept as stored). Dropped: the torch distributed checkpoint reader, `swap_linears` and the FP8 / NVFP4 quantized-linear paths (`fp8.py`, `nvfp4.py` are not vendored). A non-safetensors path raises `FileNotFoundError`; pointing at `variants/fp8r` or `variants/nvfp4` fails the key match with the `.weight_scale` keys listed as unexpected. |
| `bbox.py` | `src/flux3/bbox.py` | none (needed to replay box records and for a future layout mode). |
| `pup.py` | `src/flux3/pup.py` | `from flux3 import bbox` -> `from . import bbox`; the three engine environment defaults (`HF_HUB_OFFLINE`, `VLLM_USE_FLASHINFER_SAMPLER`, `VLLM_USE_DEEP_GEMM`) move from module import into `Upsampler.__init__` so importing the sizing (`reference_size`, `policy_reference`), the constants (`STOP_TOKENS`, `SAMPLING`, `PRESENCE_PENALTY`, `I2I_SCAFFOLD`, `MAX_REFERENCES`) and the prompt scaffolding has no side effect. The vLLM `Upsampler` class and `main()` stay as the documented reference (lazy `vllm` / `transformers` imports; never run here: the device port is `pipelines/flux3_klein/upsampler.py`). |
| `pup_prompts.json` | `src/flux3/pup_prompts.json` | none (byte-equal to the pup weights' `system_prompts.json`; the fallback `pup.py` reads when a checkpoint ships none). |
| `constants.py` | new (Tenstorrent) | the model and sampler constants, each with the upstream file:line it comes from, importable without torch. |
| `examples/` | `examples/**/*.json` | the 32 generation records (`model`, `prompt` as the DiT saw it, `input_prompt`, `boxes`, size, `steps`, `guidance`, `seed`) and the 4 `inputs/*.json` files (two records, `balloons_boxes.json`, `cactus_boxes.json`), same relative layout, copied unchanged apart from a trailing newline the repo's end-of-file-fixer added to 18 of them. The PNGs (57 MB) are not vendored (`check-large-files` 500 KB); the GPU goldens rendered from these records live on the model share. |

Not vendored: `generate.py` (safety checks, upsampler orchestration, EXIF: the upsampler rule of its lines 52-63 is
reproduced in `pipelines/flux3_klein/pipeline_flux3_klein.py::resolve_caption`), `safety.py`, `fp8.py`, `nvfp4.py`,
`scripts/` (CLI, Gradio demo), and every `vllm` / `gradio` dependency.

## Licenses

- Code (this directory): Apache-2.0, `Copyright 2026 Black Forest Labs, Inc.` (upstream `LICENSE`; upstream
  `README.md` "License" section). Tenstorrent's modifications are Apache-2.0 as the rest of tt-metal.
- Weights: `flux-3-image-klein` (distilled) and `flux-3-image-klein-base` are released under the
  [FLUX Kommunity License](https://huggingface.co/black-forest-labs/flux-3-image-klein/blob/main/LICENSE.md); they are
  never checked in and are resolved from `FLUX3_KLEIN_MODEL_PATH` / the Hugging Face cache by the pipeline code.
- Prompt upsampler weights (`flux-3-image-klein-pup`): Apache-2.0.

## Checkpoint layout (bf16 `model.safetensors`)

`DiT(flow_map=True)` has 359 tensors (distilled), `DiT(flow_map=False)` 357 (base); the only difference is
`ref_time_in.{in,out}_layer.weight` (`checkpoint.FLOW_MAP_KEY`). Token order inside a sample, and therefore the order of
the single-stream sequence, is `MODES = ("image", "image_cond", "text")`; `pipeline.pack` takes samples as
`[text, image_cond, image]` triples and lays them out mode-major.

## Running on CPU

```python
import torch
from models.tt_dit.reference.flux3_klein.checkpoint import is_flow_map, load_checkpoint
from models.tt_dit.reference.flux3_klein.dit import DiT
from models.tt_dit.reference.flux3_klein.pipeline import Pipeline, dit_path

root = "/path/to/flux-3-image-klein"  # the Hub snapshot: model.safetensors, ae.safetensors, text_encoder/
with torch.device("meta"):
    dit = DiT(is_flow_map(root))
dit = load_checkpoint(dit, root, "cpu", torch.bfloat16)

# or the whole pipeline (DiT + Qwen3-VL text encoder + autoencoder; ~20 GB of host RAM in bf16/fp32)
pipe = Pipeline({"distilled": root}, f"{root}/text_encoder", f"{root}/ae.safetensors", device="cpu")
image = pipe.generate("distilled", prompt, [], 1024, 1024, steps=16, guidance=None, seed=7)
```
