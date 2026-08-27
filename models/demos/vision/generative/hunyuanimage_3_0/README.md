# HunyuanImage-3.0

Text-to-image pipeline ([tencent/HunyuanImage-3.0](https://huggingface.co/tencent/HunyuanImage-3.0)) on Tenstorrent hardware.

An ~80B-total / ~13B-active Mixture-of-Experts model: a 32-layer transformer (64 routed experts, top-8, plus one shared expert; hidden size 4096) paired with a Conv3D VAE. A prompt is rendered by driving the transformer with a FlowMatch (Euler) denoising loop and classifier-free guidance, then decoding the latent through the VAE. The transformer and the VAE decode both run on-device — the hidden state stays on the mesh across the whole loop.

## Hardware

- **Board:** Blackhole 6U Galaxy
- **Mesh:** (8, 4) — 32 chips over `FABRIC_1D`

## Parallelism Strategy

| Component | Parallelism |
|---|---|
| Transformer (32 layers) | TP=8 + sequence parallel (token-sharded) |
| MoE experts | EP=32 |
| VAE decoder | Replicated (mesh conv3d, distributed GroupNorm) |
| Classifier-free guidance | Batched — cond + uncond as one `bsz=2` forward |

## Supported

- **Text-to-image**, 1024×1024, 50 denoising steps
- Deterministic: `(prompt, seed, steps, size)` reproduces the same image

## Prerequisites

Model weights are gated on HuggingFace. Request access, then log in:

```bash
huggingface-cli login
```

The checkpoint (~157 GB) downloads on first run.

## How to Run

All commands from the tt-metal repo root.

Export the ship configuration first. **`HUNYUAN_SP`, `HUNYUAN_CFG_PARALLEL` and
`HUNYUAN_ONDEVICE_VAE` default to off** — without them a render takes ~84 s instead of ~27 s.
(`HUNYUAN_VAE_WARMUP` and `HUNYUAN_CCL_LINKS` are already at these values by default; they are
listed so the whole configuration is explicit and copy-pasteable.)

```bash
export HUNYUAN_SP=1 HUNYUAN_CFG_PARALLEL=1 HUNYUAN_ONDEVICE_VAE=1 \
       HUNYUAN_VAE_WARMUP=1 HUNYUAN_VAE_AUTOCAST=bf16 HUNYUAN_CCL_LINKS=2
```

### Environment variables

| Variable | Default | Effect |
|---|---|---|
| `HUNYUAN_SP` | `0` | Sequence parallelism — token-shard the sequence across the TP axis (EP 32→8). Part of the ship config. |
| `HUNYUAN_CFG_PARALLEL` | `0` | Run the conditional and unconditional CFG passes as one `bsz=2` forward instead of two sequential ones. Ship config. |
| `HUNYUAN_ONDEVICE_VAE` | off | Decode the VAE on the mesh. Off = host `model.vae.decode` (correct, but ~36 s). Ship config. |
| `HUNYUAN_VAE_WARMUP` | `1` | Compile the VAE ops with a throwaway decode at model setup, so the first real decode is warm. |
| `HUNYUAN_VAE_AUTOCAST` | off | `bf16` runs the on-device VAE in bfloat16. |
| `HUNYUAN_CCL_LINKS` | `2` | Fabric links per collective. Capped at 2 by the available ethernet channels. |
| `HUNYUAN_VAE_CINBLK` / `_COUTBLK` | `128` | Conv3D input/output channel blocking in the VAE decoder. 128/128 is the measured optimum; ≥256 overflows L1. |
| `HUNYUAN_GENIMG_SIZE` | `1024,1024` | Render resolution. |
| `HY3_SINGLE_CHIP` | off | Run fabric-free on one device — for the per-component tests, not for renders. |
| `HUNYUAN_DEMO_DIR` | `$TT_METAL_HOME/generated/hunyuan_demo` | Queue and output directory for the live server. |
| `HUNYUAN_SPARSE_MOE` | off | Sparse top-8 expert dispatch. Numerically correct but far slower on the image path, where all 64 experts are active. |
| `HUNYUAN_STAGE_PROFILE` | off | Print per-layer attention vs MoE timing (adds device syncs; skews absolute numbers). |
| `HUNYUAN_VAE_TIMING` | `0` | Print the per-level VAE decode breakdown. |

### Single image

```bash
python -m models.demos.vision.generative.hunyuanimage_3_0.demo.demo_image3_t2i \
  --prompt "a red panda astronaut, studio lighting" --steps 50 --size 1024x1024 --out panda.png
```

### Transformer prefill only

Runs the 32-layer stack over a prompt without the diffusion loop — useful for bring-up and for
isolating transformer changes from the render path.

```bash
python -m models.demos.vision.generative.hunyuanimage_3_0.demo.demo_image3_prefill
```

### Warm resident server

Builds the 32-layer stack and the conv heads once, then renders prompts warm. Reports a per-render
breakdown and a `WARM_AVG` over all but the first render.

```bash
python -m models.demos.vision.generative.hunyuanimage_3_0.demo.warm_render_server
```

### Live queue-driven server

Tails a JSON-lines queue; append one line per request and the PNG appears in `out/`.

```bash
export HUNYUAN_DEMO_DIR=/tmp/hunyuan_demo
python -m models.demos.vision.generative.hunyuanimage_3_0.demo.demo_live_server

# from another shell:
echo '{"id":"img1","prompt":"a cyberpunk city in the rain","steps":50}' >> $HUNYUAN_DEMO_DIR/queue.jsonl
```

## Tests

```bash
# per-component PCC vs the HF reference — single-chip and TP=8 sharded
pytest models/demos/vision/generative/hunyuanimage_3_0/tests/pcc -s

# full prefill pipeline vs the HF reference forward
pytest models/demos/vision/generative/hunyuanimage_3_0/tests/e2e/test_e2e_prefill.py -s

# whole prompt -> image path
pytest models/demos/vision/generative/hunyuanimage_3_0/tests/e2e/test_image3_t2i_e2e_pcc.py -s

# on-device VAE decode PCC ladder (13 gates)
pytest -o timeout=0 models/demos/vision/generative/hunyuanimage_3_0/tests/e2e/test_vae_decode_pcc.py

# end-to-end render latency — the shipped number
pytest -o timeout=0 models/demos/vision/generative/hunyuanimage_3_0/tests/e2e/test_host_glue_stage3_perf.py -s
```

The rest of the suite is under `tests/e2e/` (decode path, host-glue PCC, trace + 2CQ, mesh health)
and `tests/pcc/`. `HY3_SINGLE_CHIP=1` runs the per-component tests fabric-free on one device; full
renders need the mesh.

## Performance

1024×1024, 50 steps, on one Blackhole 6U Galaxy. Warm figures are the `WARM_AVG` from
`warm_render_server` (first render excluded — it absorbs the one-time kernel compile).

| Stage | Time |
|---|---|
| **End-to-end, warm** | **~27 s/image** |
| — denoising loop | ~25.8 s (~511 ms/step × 50) |
| — VAE decode (on-device) | ~1.0 s |
| End-to-end, cold (first image) | ~170–210 s |

## Accuracy

| Check | PCC | Gate |
|---|---|---|
| Prefill `last_hidden_state` vs HF reference | 0.99977 | ≥ 0.95 |
| Per-component, single-chip (decoder / MoE / gate) | 0.99999 / 0.9996 / 1.0 | ≥ 0.95 |
| Per-component, TP=8 sharded (decoder / MoE / gate) | 0.99999 / 0.9940 / 1.0 | ≥ 0.95 |
| On-device VAE decode vs host `model.vae.decode` | 0.99995 | ≥ 0.99 |
| Image-step velocity vs host reference | 1.0 | ≥ 0.99 |

## Layout

```
_stubs/            native-TTNN transformer blocks: decoder layer, MoE, top-k gate
tt/pipeline.py     shared prefill/decode forward (build, trace, self-tests)
tt/gen_image.py    text→image driver — FlowMatch loop, CFG, velocity head, VAE
tt/host_glue_*.py  native-TTNN patch_embed + final_layer, on-device head glue
tt/vae_decode.py   mesh Conv3D VAE decoder
demo/              single-image demo, warm resident server, live queue server
tests/e2e/         prefill gates, VAE PCC, trace + 2CQ, text→image PCC and latency
tests/pcc/         per-component PCC, single-chip and TP=8 sharded
```
