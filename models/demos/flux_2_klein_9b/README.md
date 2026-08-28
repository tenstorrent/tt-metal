# FLUX.2-klein-9B

Text-to-image pipeline ([black-forest-labs/FLUX.2-klein-9B](https://huggingface.co/black-forest-labs/FLUX.2-klein-9B)) on Tenstorrent hardware.

A step-distilled flow-matching model in three parts: a **9.08 B MM-DiT** (8 dual-stream + 24 single-stream blocks, 32 heads x 128 head_dim), an **8.19 B Qwen3-8B text encoder** (36 layers, hidden 4096), and a **0.08 B VAE** — 34.71 GB bf16 total. A prompt is encoded, denoised through a FlowMatch (Euler) loop, and decoded through the VAE.

## Status

- **On device:** all three components brought up and wired end-to-end. 40 components graduated with per-component PCC verification, then `emit-e2e` PASSED on all three.
- **Effective placement is 100% on device.** The raw reports read 9/10, 18/18 and 12/15 — see the note below for why the shortfall is a reporting artefact, not host execution.
- **No end-to-end PCC number yet.** The e2e gates passed on placement and harness; `e2e PCC: n/a` on every task. Per-component PCC was verified during bring-up, but the assembled pipeline has not been compared numerically against the HF reference.
- **No perf number yet.** No trace perf test was generated, so there is no measured latency for this pipeline.
- **Next:** an end-to-end PCC gate, then a perf baseline.

### Why the placement counts read below 100%

| component | reported | actually |
|---|---|---|
| text_encoder | 9/10 | `attention` is built and called *inside* `decoder_layer` (`TtAttention` imported, built, invoked), so it runs on the mesh. It is only counted as "not wired" because it is not independently routed as a top-level component. |
| transformer | 18/18 | fully on device, nothing to explain |
| vae | 12/15 | `layer`, `mlp`, `patch_embed` are **not VAE modules**. The reuse registry matched those generic names to LLaMA multimodal modules (`llama_layernorm`, `mlp`, `llama_conv2d_patch`); they are imported by nothing and correctly unused. |

`NEW-fallback=0` on all three: nothing was attempted natively, failed PCC, and retired to host.

## Hardware

- **Board:** Wormhole T3K / LoudBox — 8 chips (4x n300: 4 local PCIe + 4 eth-remote)
- **Mesh:** (2, 4) over `FABRIC_1D`

## Parallelism Strategy

| Component | Parallelism |
|---|---|
| text_encoder (Qwen3-8B, 36 layers) | TP=8 — 7/10 components shard-graduated |
| transformer (MM-DiT, 32 blocks) | TP=8 — 13/18 components shard-graduated |
| vae | TP=8 — 12/12 shard-graduated |

Components that are graduated but not shard-graduated run single-device. On a 9 B model across 8 chips that is the main remaining placement gap.

## Supported

- **Text-to-image**, 1024x1024, **4 denoising steps** (`is_distilled: true`)
- **No classifier-free guidance.** `guidance_scale` is 1.0 and `transformer/config.json` has `guidance_embeds: false`, so there is **one** DiT forward per step, not two. (Do not carry FLUX.2-dev's 4.0 across — it doubles the DiT compute.)
- 128x128 latent -> 2x2 pack -> 4096 tokens; `max_sequence_length` 512
- `text_encoder_out_layers` (9, 18, 27) of 36 -> joint dim 3 x 4096 = 12288

## Prerequisites

The model is **gated**. Accept the FLUX Non-Commercial License on the model page, then:

```bash
hf auth login
```

Verify with a *file* fetch, not `model_info` — metadata on a gated repo is public, so `model_info` returns 200 with no token and gives a false green:

```bash
curl -s -o /dev/null -w '%{http_code}\n' -H "Authorization: Bearer $(hf auth token)" \
  https://huggingface.co/black-forest-labs/FLUX.2-klein-9B/resolve/main/model_index.json
```

`diffusers >= 0.37` is required for the `Flux2*` classes.

## How to Run

All commands from the tt-metal repo root.

```bash
export PYTHONPATH=$PWD TT_METAL_HOME=$PWD
```

Each component has its own demo:

```bash
python models/demos/flux_2_klein_9b/text_encoder/demo/demo.py
python models/demos/flux_2_klein_9b/transformer/demo/demo.py
python models/demos/flux_2_klein_9b/vae/demo/demo.py
```

`TT_FLUX2_KLEIN_TRANSFORMER` / `TT_FLUX2_KLEIN_VAE` override the checkpoint path; by default the components resolve from the HF repo with `subfolder=`.

## Tests

```bash
# per-component PCC vs the HF reference
pytest models/demos/flux_2_klein_9b/text_encoder/tests -s
pytest models/demos/flux_2_klein_9b/transformer/tests -s
pytest models/demos/flux_2_klein_9b/vae/tests -s
```

The `_captured/` goldens these compare against are gitignored (4.1 GB). Regenerate with:

```bash
python -m scripts.tt_hw_planner capture-inputs black-forest-labs/FLUX.2-klein-9B
```

## Accuracy

| Check | Result |
|---|---|
| Per-component PCC, all three components | verified during bring-up (40 components graduated) |
| emit-e2e verdict | PASS on text_encoder, transformer and vae |
| End-to-end PCC vs HF reference | **not yet measured** (`e2e PCC: n/a`) |

## Layout

```
text_encoder/   Qwen3-8B encoder — _stubs/ (native TTNN), tt/, demo/, tests/
transformer/    MM-DiT — flux2_parallel_self_attention, flux2_single_transformer_block,
                flux2_modulation, flux2_swi_g_l_u, flux2_timestep_guidance_embeddings,
                flux2_pos_embed
vae/            encoder/decoder — _stubs/, tt/, demo/, tests/
```

Each component keeps its own README with the per-stage detail.

## Note on the host this was brought up on

The T3K used here runs legacy firmware (CMFW 80.15.0 / ETH FW 6.10.0). Current tt-metal cannot open a mesh on that firmware without a UMD fix — remote (ethernet-attached) chips fail a non-MMIO flush during firmware init. That fix is **not** part of this branch and is tracked separately. On a box with current firmware this is not a concern.
