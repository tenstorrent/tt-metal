# MOSS-Image on Blackhole

Experimental independent text-to-image inference for the official OpenMOSS
[MOSS-Image](https://github.com/OpenMOSS/MOSS-Image) Base and SFT checkpoints.
The learned prompt encoder, diffusion transformer and VAE decoder run on TTNN;
no CUDA runtime or reference activation is required for generation.

Base and SFT share the same architecture and identical prompt encoder, tokenizer,
VAE and scheduler. Their diffusion transformer weights differ. Both variants are
qualified separately. See [DESIGN.md](DESIGN.md) for numerical and placement
choices and [tests/README.md](tests/README.md) for validation commands and asset
requirements. Per-run reports, logs and generated images belong outside the
source checkout; summarize validation evidence in the PR description or attach
it as CI artifacts.

Final independent 50-step image agreement has not met the current 0.99 image-PCC
target for the tested workload. Passing component checks does not establish
final-image accuracy.

## Scope

- Batch one, text-to-image generation.
- Qwen3.5-9B all-token prompt encoding on one chip, including Gated DeltaNet and
  full-attention layers. No language-model head or vision encoder is loaded.
- MOSS diffusion transformer on one Blackhole p150 or two connected p150 cards
  using head/channel tensor parallelism.
- FP32 classifier-free guidance and Euler updates on TT.
- Grouped attention with FP32 scores and softmax, BF16 probabilities and FP32 PV
  accumulation in both the prompt encoder and DiT. This favors accuracy over
  the preceding native-SDPA implementation's runtime.
- FLUX.2 VAE decode on TT; the decoder is replicated on a two-card mesh.
- CPU tokenization, independently seeded initial Gaussian noise, scheduler/RoPE
  constants, module boundary transfers and image/artifact serialization.

The qualification workload uses 1024×1024, 50 steps, seed 42, CFG 3, schedule
shift 5, and the official negative-branch block-22 skip. The runner validates
multiples of 16 at least 256 pixels and 1–1000 requested schedule steps; that
input validation is not a claim that every resolution/step combination has been
qualified. Image editing, image conditioning, batches, training and a service API
are outside this contribution. Prompts use the official chat template and are
not silently expanded.

## Setup

Build the current checkout using the repository's normal source-build instructions
with Python 3.11 or newer and activate its Python environment. Pillow 13 contains
security fixes and no longer supports Python 3.10; use a matching TTNN source
build, such as the repository's Ubuntu 24.04 / Python 3.12 image. Installing this
model into the Ubuntu 22.04 / Python 3.10 image is unsupported. Install the model
additions without replacing TTNN, PyTorch or the repository's pinned packages:

```bash
python -m pip install -r models/experimental/moss_image/requirements.txt
```

The dependency bounds include the current CI stack (Transformers 5.12.1,
Diffusers 0.38.0, Accelerate 1.7.0 and repository PyTorch 2.11). Do not install a
separate TTNN wheel over the source build. The optional CUDA capture environment
uses [requirements-reference.txt](requirements-reference.txt) separately.

## Checkpoints

| Variant | Hugging Face repository | Pinned revision |
|---|---|---|
| Base | `OpenMOSS-Team/MOSS-Image-Base` | `1ceede80badddd6a7812069619edaa5633587192` |
| SFT | `OpenMOSS-Team/MOSS-Image-SFT` | `fc5de0d56fc910cbca6f6625a70f5f582b26deb1` |

Download the fixed revision with `huggingface_hub.snapshot_download`, or supply an
existing local directory containing `transformer/`, `text_encoder/`, `tokenizer/`,
`vae/` and `scheduler/`. `resolve_checkpoint()` supports fixed-revision online
resolution and explicit offline resolution; generation uses local files only.
A complete checkpoint occupies approximately 25.36 GiB. Base plus SFT's differing
transformer costs approximately 35 GiB when shared components are linked. Captures
and outputs belong outside the repository.

Index-declared safetensors shards are authoritative. Additional unindexed files
cannot replace verified weights. Configurations, tensor shapes and required files
are checked before their learned component executes. Component replay verifies
all consumed files against its pinned CUDA manifest. Local checkpoint directories
are identified by their actual hashes, not assumed to match a Hub revision.
No Hugging Face remote Python code is executed.

## Independent generation

Run modules from the repository root. Reserve the selected cards before invoking
the CLI; hardware visibility and physical device IDs are deployment-specific.
Use a connected pair for the two-card case.

```bash
python -m models.experimental.moss_image.demo \
  --checkpoint /path/to/MOSS-Image-Base --variant Base \
  --device-ids 0 --height 1024 --width 1024 --steps 50 --seed 42 \
  --prompt 'the quick brown fox jumps over the lazy dog' \
  --output-dir /path/to/new-output-directory
```

For two cards, use `--device-ids 0 1`. This CLI owns its device/fabric lifetime.
`demo.generate(checkpoint, mesh_device, output_dir, ...)` instead uses an existing
one/two-chip mesh and never opens or closes the caller's mesh or changes fabric
configuration. On two cards it creates a one-chip submesh for prompt encoding.

The new output directory receives `status.json`, conditioning, independently
sampled initial noise, schedule, every denoised latent, first-step predictions,
raw final decoded pixels and `image.png`. The runner refuses to overwrite an
existing output directory and reports nonfinite outputs as failures.

## Accuracy and tests

The tests use the repository's device/mesh fixtures and call Python directly;
no test interpolates environment variables into an OS command. Component replay
and integrated cases live in [tests/](tests/). There is no scheduled CI model
registration for MOSS-Image; CUDA captures are manually generated local
validation assets. Tests report PCC together with relative RMS and maximum
absolute error so gain errors cannot pass solely through correlation.
Asset-dependent tests skip with a provisioning message when the CUDA reference
or checkpoint is unavailable, including under CI. Available references still
undergo digest and configuration checks; invalid assets fail. A skipped case
is not a successful device qualification. See the [test instructions](tests/README.md)
for asset layout and commands.
Attention and CFG/Euler also have capture-free CPU reference tests on one/two
cards. CFG/Euler compares against the vendored MOSS scheduler at early, middle
and final steps; attention compares against dense PyTorch attention. These are
component checks on TT hardware, not full-model CPU inference support.

Component replay intentionally supplies captured CUDA inputs to isolate a stage.
Independent generation starts from raw prompts and independently sampled noise;
reference outputs are read only afterwards. These two kinds of evidence are
reported separately. High component PCC alone does not establish agreement of
the final independent image after 50 denoising steps.

To create fresh native CUDA captures in a separate CUDA environment:

```bash
python -m models.experimental.moss_image.validation.cuda_reference \
  --variant Base --checkpoint /path/to/MOSS-Image-Base \
  --height 1024 --width 1024 --steps 50 --generator-device cpu \
  --output-dir /path/to/new-cuda-capture --local-files-only
```

The upstream reference is pinned to OpenMOSS commit
`c1355e6b1986dfe8fba3c0455783a05899212342`. Its vendored file has formatting
and SPDX attribution changes only; its AST matches the original source.
[Provenance](tt/vendor/PROVENANCE.json) records both hashes. The GDN prefill uses an explicit FP32 SFPU recurrence in
[tt/gated_delta.py](tt/gated_delta.py), replacing the earlier chunked vendor
helpers. Apache-2.0 licenses and upstream attribution are retained.
