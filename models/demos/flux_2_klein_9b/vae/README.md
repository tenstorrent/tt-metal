<!--
SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
SPDX-License-Identifier: Apache-2.0
-->
# FLUX.2 Klein 9B VAE — `AutoencoderKLFlux2` on Tenstorrent (T3K, 1x8)

## What this model is

The image autoencoder of FLUX.2 Klein 9B: a **diffusers** `AutoencoderKLFlux2`
(~84M params), not a `transformers` model. Its `config.json` carries no `model_type` —
it is a diffusers config (`_class_name: AutoencoderKLFlux2`), so the correct registry is
`diffusers.AutoencoderKLFlux2.from_pretrained(<path>)`, never `transformers.AutoModel`.

It maps **image <-> latent** with **8x spatial compression**:

| | shape |
|---|---|
| image | `[1, 3, C, C]`, float32 in `[-1, 1]` (`C = 224`, the resolution the bring-up capture used) |
| latent | `[1, 32, C/8, C/8]` = `[1, 32, 28, 28]` |

Structure: `block_out_channels = [128, 256, 512, 512]`, four `DownEncoderBlock2D` /
four `UpDecoderBlock2D`, `layers_per_block = 2`, `norm_num_groups = 32`, a mid block with
attention on both sides, and 1x1 `quant_conv` / `post_quant_conv`.

There is **no `generate()`** and no autoregressive loop — the reference is a deterministic
forward (`sample_posterior=False` takes the posterior MODE, no RNG). The top-level `bn`
(`BatchNorm2d(128, affine=False)`) is registered on the module but is **not** called by
`encode`/`decode`/`forward` in diffusers 0.38 (patchify/unpatchify live in the FLUX.2
*pipeline*, not the VAE), so it is deliberately absent from the TT chain.

Input/output are built exactly as HF builds them, through
`diffusers.image_processor.VaeImageProcessor` (`vae_scale_factor = 8`):
`preprocess(PIL, height=C, width=C)` in, `postprocess(sample)` out.

**The chained forward pass lives ONLY in `tt/pipeline.py`.** The demos and the tests import
`build_pipeline` from there and call `run_encode` / `run_decode` / `run_reconstruct`.
Nothing in `demo/` or `tests/` re-implements the wiring.

## Layout

```
models/demos/flux_2_klein_9b_vae/
  tt/reference.py            HF loader, processor, captured tensors, the three goldens (Source A only)
  tt/pipeline.py             THE chain: build_pipeline(), run_encode/run_decode/run_reconstruct, trace hooks
  demo/demo_encode.py        Call 1  image  -> latent
  demo/demo_decode.py        Call 2  latent -> image
  demo/demo_reconstruct.py   Call 3  image  -> image   (headline demo)
  tests/e2e/test_e2e_pipeline.py    Gates 1, 2, 3
  tests/e2e/test_trace_contract.py  Command 3 — the trace contract
  e2e_plan.json              the frozen plan this surface implements
```

## The three Calls

| # | Call | Consumes | Produces | HF golden |
|---|---|---|---|---|
| 1 | `encode` | `VaeImageProcessor.preprocess(image, 224, 224)` -> `[1,3,224,224]` fp32 in `[-1,1]` | latent `[1,32,28,28]` — the posterior **mode** (= mean = first 32 channels of `quant_conv(encoder(x))`) | `model.encode(x).latent_dist.mode()` |
| 2 | `decode` | latent `[1,32,28,28]` — by default the captured golden `_captured/decoder/args.pt[0]`, or `--latent` from Call 1's own output | image `[1,3,224,224]` -> `VaeImageProcessor.postprocess` -> PIL | `model.decode(z).sample` |
| 3 | `reconstruct` | same as Call 1 | reconstructed image `[1,3,224,224]` -> PIL | `model(x).sample` (`sample_posterior=False`) |

Call 3 is the model's own forward: `encoder_stack -> quant_conv -> mode -> post_quant_conv ->
decoder_head`. The latent at that joint is the **TT** one — no reference tensor is injected
mid-chain. (Call 2's latent is a *stage input*, which is the head's own input, not a joint.)

Call 3 exists because `encoder_stack` / `decoder_head` are graduated whole-stack ports of the
*same* modules as `encoder` / `decoder`. A single chain cannot invoke both members of an alias
pair without running the same stack twice, so the alias pair gets its own task head rather than
being dropped as wasted work.

## Graduated-stub routing (12 of 12 routed, none wasted)

`GRADUATED` = has a `_stubs/<name>.py.last_good_sharded` snapshot under
`models/tt_dit/pipelines/flux_2_klein_9b_vae/`. All 12 are routed into a real forward path:

| # | Graduated stub | Runs in | Position in the graph |
|---|---|---|---|
| 1 | `encoder` | Call 1 | `encode_stack(x)` — the whole encoder, children replaced by their own stubs |
| 2 | `down_encoder_block2_d` | Call 1 | `encoder.down_blocks[0..3]` (**x4**) |
| 3 | `resnet_block2_d` | Call 1 | `down_blocks[0].resnets[0]` |
| 4 | `downsample2_d` | Call 1 | `down_blocks[0].downsamplers[0]` |
| 5 | `u_net_mid_block2_d` | Call 1 | `encoder.mid_block` |
| 6 | `attention` | Call 1 | `encoder.mid_block.attentions[0]` |
| 7 | `decoder` | Call 2 | `decode_stack(z)` — the whole decoder, children replaced |
| 8 | `self_attention` | Call 2 | `decoder.mid_block.attentions[0]` |
| 9 | `up_decoder_block2_d` | Call 2 | `decoder.up_blocks[0..3]` (**x4**) |
| 10 | `upsample2_d` | Call 2 | `decoder.up_blocks[0].upsamplers[0]` |
| 11 | `encoder_stack` | Call 3 | the whole encode leg of `model(x)` |
| 12 | `decoder_head` | Call 3 | the whole decode leg of `model(x)` |

Alias pairs (two graduated ports of ONE module): `encoder_stack == encoder`,
`decoder_head == decoder`, and `self_attention ~ attention` (`self_attention.py` returns
`attention.py`'s `TtVaeAttention` class built on the decoder mid block's own weights).

**Not routed, and why** — `patch_embed`, `mlp`, `layer` are transformer-template roles
(`llama_conv2d_patch`, SwiGLU MLP, `llama_layernorm`) with no `.last_good_*` snapshot;
`RUN_REPORT.md` places all three `CPU_REUSE`. `AutoencoderKLFlux2` has no patch embedding and
no MLP, and it normalises with GroupNorm, which the graduated stubs implement natively.

Non-graduated ops in the chain (`quant_conv`, `post_quant_conv`, `down_blocks[0].resnets[1]`,
`conv_in`/`conv_out`, the channel slice for the posterior mode) run through
`_stubs/_vae_blocks.py`, which is native ttnn as well.

## Running the demos

```bash
export TT_METAL_HOME=<tt-metal> \
       PYTHONPATH=<tt-metal> \
       ARCH_NAME=wormhole_b0
source <tt-metal>/python_env/bin/activate
cd <tt-metal>

python -m models.demos.flux_2_klein_9b_vae.demo.demo_reconstruct     # headline
python -m models.demos.flux_2_klein_9b_vae.demo.demo_encode
python -m models.demos.flux_2_klein_9b_vae.demo.demo_decode
```

Each demo opens its own `1 x TP` mesh (`ttnn.set_fabric_config(FABRIC_1D)` first, then
`ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, tp), l1_small_size=24576)`), builds the
pipeline through `build_pipeline`, runs the Call, writes the real artifact, prints
`e2e PCC=<v>` against the HF golden on its own line, and prints the invoked graduated modules.

Shared flags: `--image PATH` (default: `reference.load_input_image()`'s built-in),
`--size INT` (default `224`), `--tp INT` (default `8`, or `$TT_HW_PLANNER_SHARD_TP`),
`--output-dir PATH` (default `flux2_vae_demo_out`), `--layers INT` (default `None` = every
layer). `demo_decode` additionally takes `--latent PATH` to consume Call 1's `latent.pt`.

Artifacts written under `--output-dir`:

| demo | artifacts | printed metrics |
|---|---|---|
| `demo_encode` | `input.png`, `latent.pt`, `latent_preview.png` (per-channel-mean, min-max normalised) | `e2e PCC`, max abs err, latent shape/mean/std/min/max |
| `demo_decode` | `tt_decode.png`, `hf_decode.png` | `e2e PCC`, max abs err, PSNR(TT vs HF); PSNR(TT vs original) when `--image` is given |
| `demo_reconstruct` | `input.png`, `tt_reconstruction.png`, `hf_reconstruction.png` | `e2e PCC`, max abs err, PSNR(TT vs HF), PSNR(TT vs original), PSNR(HF vs original) |

The behavioural proof is the round trip: a pipeline that merely passes tensors around cannot
produce a high-PSNR reconstruction of the input image, and a mis-wired joint (say a swapped
`up_block`) shows up as a visibly wrong PNG even when every shape agrees.

## Running the tests

```bash
./python_env/bin/python -m pytest models/demos/flux_2_klein_9b_vae/tests/e2e/test_e2e_pipeline.py -s
./python_env/bin/python -m pytest models/demos/flux_2_klein_9b_vae/tests/e2e/test_trace_contract.py -s
```

`-s` matters: every gate prints its number before it asserts, so the PCC is visible on failure
as well as on pass.

| Test | Device | What it proves |
|---|---|---|
| `test_e2e_pipeline.py::test_gate1_stubs_are_native_ttnn` | no | **Gate 1.** Each live `_stubs/<name>.py` is sha256-identical to its `.last_good_sharded` snapshot; `_runtime_fallbacks.json == {}`; a static scan of the 12 routed stubs + `tt/pipeline.py` + `demo/*.py` finds no forbidden torch compute op and no HF orchestration; no `coverage_step`/`coverage_sweep`/`invoke_all_stubs`/`_touch_all_graduated` exists in the package. |
| `test_e2e_pipeline.py::test_e2e_pipeline` | 1x8 | **Gates 2 + 3.** ONE pipeline build, all three Calls, `invoked_modules()` == the 12 graduated names with `down_encoder_block2_d >= 4` and `up_decoder_block2_d >= 4`, and each Call's PCC >= 0.95. |
| `test_e2e_pipeline.py::test_gate3_chaining_is_real` | 1x8 | The reconstruction is the TT numbers (not bit-identical to `hf_decode(hf_encode(x))`) yet >= 0.95 correlated with them, and perturbing the input moves the output. |
| `test_trace_contract.py::test_pipeline_stages_declared` | no | `PIPELINE_STAGES == ["encode", "decode"]`; the class carries all four hooks per stage; no `decode_prefill`/`decode_step`. |
| `test_trace_contract.py::test_trace_inputs_are_zero_arg` | no | `<stage>_trace_inputs` takes only `self`; `_trace_step`/`_trace_items` need no argument; `_trace_setup` takes exactly `(self, inputs)`. |
| `test_trace_contract.py::test_trace_capture_selftest` | 1x8 + trace | `trace_capture_selftest(mesh_device) is True` — a real capture/execute/release per stage. |
| `test_trace_contract.py::test_host_op_selftest` | 1x8 | `host_op_selftest()["on_device"]` — no host aten op fires in any head. |
| `test_trace_contract.py::test_layers_knob` | 1x8 | `layers=1` builds strictly fewer repeated blocks than `layers=None`, and `layers=None` builds every layer the HF reference has. |

The scan in Gate 1 reads each file's text, drops COMMENT and STRING tokens (so a pattern
merely *named* in a comment or an error message is not a hit), and matches the remaining code.
`tt/reference.py` is Source-A only and is **not** scanned. The only in-file allowlist is
functions whose name starts with `_hf_reference` / `_golden` or ends with `_trace_setup`.

The mesh is a single shared resource — run the on-device gates serially.

## PCC results

Threshold **0.95** for every Call, computed in float32 by
`models.common.utility_functions.comp_pcc` on the FINAL task output.

Measured 2026-08-28 on a T3K (8x wormhole_b0), mesh 1x8, TP=8, 224x224 images, bfloat16
activations with HiFi4 + fp32 dest accumulation. Source: one run of
`tests/e2e/test_e2e_pipeline.py::test_e2e_pipeline` (65.97 s) plus the three demos.

| Call | Metric | Target | Achieved |
|---|---|---|---|
| encode | PCC | >= 0.95 | **0.9855309760071099** |
| encode | max abs err | — | 3.4842178821563720 |
| decode | PCC | >= 0.95 | **0.9986970492997539** |
| decode | max abs err | — | 0.2662031650543213 |
| decode | PSNR (TT vs HF) | — | 36.25 dB |
| reconstruct | PCC | >= 0.95 | **0.9969077014852629** |
| reconstruct | max abs err | — | 0.3154431879520416 |
| reconstruct | PSNR (TT vs HF) | — | 31.71 dB |
| reconstruct | PSNR (TT vs original) | — | 29.87 dB |
| **overall** | `e2e PCC` = min of the three | >= 0.95 | **0.9855309760071099** |

For reference, the HF fp32 reference itself reconstructs the sample image at **35.26 dB** —
that is the ceiling a lossy 8x-compressing VAE imposes, so the TT path gives up 5.4 dB to
bfloat16 and the 8-way channel split, and stays visually indistinguishable.

Two decode numbers exist because the two entry points feed the head different latents, and
both are honest:

* the **test** uses `hf_reference_encode(image)` — a real latent for the sample image — and
  scores 0.99870;
* the **demo** defaults to `_captured/decoder/args.pt`, the bring-up capture. That tensor was
  hooked at the `decoder` submodule, so it is `post_quant_conv`'s *output*, not a latent. Both
  sides still apply `post_quant_conv` to it, so the comparison is like-for-like — it just runs
  the head slightly off its natural input distribution, and scores 0.98872. Pass `--latent` a
  real latent (e.g. `demo_encode`'s `latent.pt`) to see the other number.

Per-Call graduated-stub PCC, for context, comes from the bring-up's own sharded gates:
whole `encoder` 0.99665 and whole `decoder` above its 0.99 target, both at TP=8.

### Gate 2 ledger — measured

`invoked_modules()` after all three Calls, **12/12 graduated modules, none left out**:

```
{'encoder': 1, 'down_encoder_block2_d': 4, 'resnet_block2_d': 1, 'downsample2_d': 1,
 'u_net_mid_block2_d': 1, 'attention': 1, 'decoder': 1, 'self_attention': 1,
 'up_decoder_block2_d': 4, 'upsample2_d': 1, 'encoder_stack': 1, 'decoder_head': 1}
```

The two `x4` counts are the structural check: the encoder has four `down_blocks` and the
decoder four `up_blocks`, so anything less would mean the block ladder was short-circuited.

### Command 3 — measured

```
trace encode: captured=True pcc=1.0
trace decode: captured=True pcc=1.0
[host-op] n_host_ops = 0
[layers] layers=None -> total=32 {'encode_resnets': 10, 'decode_resnets': 14, 'encode_down_blocks': 4, 'decode_up_blocks': 4}
[layers] layers=1    -> total=20 {'encode_resnets': 6,  'decode_resnets': 6,  'encode_down_blocks': 4, 'decode_up_blocks': 4}
```

Both stages capture host-free and replay bit-identically (PCC 1.0), each trace released
before the next is captured. `host_op_selftest()` fires **zero** host aten ops across all
three heads. The `layers` knob measurably moves the built repeated-block count (32 -> 20),
so it is not inert.

### Chaining proof — measured

```
[gate3-chain] reconstruct vs hf_decode(hf_encode(x)): e2e PCC=0.9969077014852629
[gate3-chain] max|recon(x) - recon(0.5x-0.25)| = 0.7724609375
```

Not bit-identical to the golden (so the TT chain is producing its own numbers, not handing
back a reference tensor) and it moves with its input (so nothing is short-circuited).

## Trace contract (Command 3)

```python
PIPELINE_STAGES = ["encode", "decode"]
```

Derived from the diffusers config, which states the phases directly: `down_block_types` is the
compression stack and `up_block_types` the expansion stack. There is no autoregressive phase —
no `generate()`, no KV cache — so no `[prefill, decode]`, no `[vocode]`, and
`decode_prefill`/`decode_step` are deliberately absent.

Per-stage hooks on the pipeline **object**:

| Hook | Contract |
|---|---|
| `<stage>_trace_setup(inputs)` | Pins the stage's variable dim and pre-uploads the padded input plus every shape-dependent constant into persistent device buffers, **outside** the trace. |
| `<stage>_trace_step()` | ONE host-op-free forward at the fixed shape, reading ONLY those persistent buffers. No `from_torch`, no per-call `ttnn.zeros`/`arange` inside the trace. |
| `<stage>_trace_inputs()` | **Zero-arg.** Returns exactly the value `<stage>_trace_setup` takes, assembled from the captured reference tensors: `encode` -> `_captured/encoder/args.pt[0]` `[1,3,224,224]`; `decode` -> `_captured/decoder/args.pt[0]` `[1,32,28,28]`. |
| `<stage>_trace_items()` | **Zero-arg.** Items retired by one `_trace_step`. |

For this model the variable axis is **spatial**, not a sequence: the config bound is
`sample_size = 1024` and the compression factor is 8, so `encode`'s capacity is the image side
`C` and `decode`'s is `C/8`, both pinned at build time (default `C = 224`). Every conv here
zero-pads, so a padded spatial tail is invisible to the convs — but **GroupNorm reduces over
all `H*W` positions**, so a pad WOULD move the statistics of the real region. There is no mask
that makes padded positions free. The honest pin is exact: `VaeImageProcessor.preprocess`
resizes to exactly the pinned capacity before the pipeline ever sees the tensor, so
`real_len == C`. If `_trace_setup` is handed something smaller it zero-pads **and prints** the
fallback plus that warning.

`<stage>_trace_items()` is a params-weighted mean output area over the stage's convs —
`items = sum_c(params_c * H_out_c * W_out_c) / sum_c(params_c)` — so that `2 * params * items`
equals the stage's true MAC count instead of pricing a ~50-conv stack at one item. It is
computed from the HF reference at the pinned capacity, not hardcoded.

Persistent buffers staged outside the trace: the GroupNorm one-hot membership matrices
`[C,G]`/`[G,C]`, all conv weights/biases (device-prepared and cached on the first warm-up call
so `ttnn.conv2d` does no host preparation inside the trace), gamma/beta, and one input buffer
per stage written with `ttnn.copy_host_to_device_tensor`.

### `build_pipeline` and the `layers` knobs

```python
build_pipeline(device, model=None, layers=None,
               encode_layers=None, decode_layers=None, **kwargs) -> TtFluxVaePipeline
```

Constructs and **returns** the resident pipeline object — it never runs it. Demo kwargs are
accepted and ignored for call-signature compatibility. `recommended_trace_region_size()` sizes
the device's trace region (the tests fall back to `23887872` if it cannot be called at
collection time).

* `layers` caps the depth of **every** repeated block; `None` means every layer. `0` is never
  read as "no layers".
* The genuine repeats are the `resnets` *inside* each block (2 per encoder block, 3 per decoder
  block) plus the mid blocks' resnets — **not** `down_blocks`/`up_blocks` themselves, which each
  change channel width and spatial resolution, so removing one would change the output shape and
  the stage could not run. `layers` therefore caps resnets-per-block (floor 1) and leaves the
  block ladder, `conv_in`/`conv_out`, the norms and the attention intact: a capped build still
  runs every *distinct* op the full model runs, just fewer times.
* **Mid-block floor:** each mid block keeps its 2 resnets. `UNetMidBlock2D` runs `resnets[0]`
  then `zip(attentions, resnets[1:])`, so capping it to 1 would make the ATTENTION structurally
  absent — the documented "cap to the smallest depth that keeps every stage able to run" case.
* `encode_layers` / `decode_layers` override per stack, named after the `PIPELINE_STAGES` entry
  that owns each. `None` falls back to `layers`; `layers=None` still means every layer.
* Every repeated block is held as a plain Python list of same-typed elements, and where a
  graduated stub replaces one element **every** element of that list is wrapped in the same
  adapter class, so the list stays homogeneous and discoverable. `pipeline.hf` keeps the HF
  reference reachable as ground truth for how many sections the model has and how deep each is.

Other pipeline surface used by this directory: `invoked_modules()` (the passive Gate 2 ledger),
`reset_invocations()`, `trace_capture_selftest(device=None) -> bool`, and
`host_op_selftest() -> {"on_device", "host_ops", "n_host_ops", "reason"}`.

## Environment

T3K: 8x `wormhole_b0`, topology **TP=8 x DP=1, mesh 1x8** — the topology the bring-up
graduated at (`parallelism_manifest.json`). The stubs keep their `ShardTensorToMesh` weights
and their `all_gather`/`all_reduce`; sharded bodies count as native and are not rewritten to
replication.

```bash
export TT_METAL_HOME=<tt-metal>
export PYTHONPATH=<tt-metal>
export ARCH_NAME=wormhole_b0
source <tt-metal>/python_env/bin/activate
cd <tt-metal>
```

Device params used everywhere here: `l1_small_size = 24576` and
`fabric_config = ttnn.FabricConfig.FABRIC_1D` (plus `trace_region_size` for the trace test).
The fabric **must** be enabled before the mesh is opened or every CCL in the sharded stubs
raises `TT_FATAL ... fabric_context_ != nullptr`. `ttnn.open_mesh_device` in this checkout takes
no `fabric_config` argument, so the demos call `ttnn.set_fabric_config(...)` first, exactly as
the root `conftest.py` fixture does.

Optional overrides: `TT_HW_PLANNER_SHARD_TP` (default `8`) and `TT_HW_PLANNER_SHARD_DP`
(default `1`) select the mesh in the tests; `TT_PLANNER_TEST_SEED` (default `0`) seeds torch.
