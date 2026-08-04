# LongCat-Image — end-to-end TTNN pipeline

Real, chained TTNN bring-up of **`meituan-longcat/LongCat-Image`** (a diffusers
text-to-image pipeline: Qwen2.5-VL 7B text encoder → LongCat MMDiT denoiser →
Flux AutoencoderKL) on Tenstorrent Blackhole — single chip (p150a, 32 GB) up to
a 4-chip QB2 mesh (tensor-parallel).

This package chains the graduated per-component TTNN stubs in `_stubs/` into the
actual forward pass and compares the final output to the HF reference.

## Directory structure

```
longcat_image/
├── _stubs/                          graduated per-component TTNN ports (build(device, torch_module) -> callable)
│   ├── qwen2_v_l_*.py                 text encoder + vision tower stubs
│   ├── long_cat_image_*.py            DiT (transformer2d, blocks, timestep embeddings)
│   └── autoencoder_k_l.py, *.py       VAE encoder/decoder stubs
├── tt/
│   └── pipeline.py                  LongCatImagePipelineTT — the ONE shared chained forward
│                                     (demo/ and tests/e2e/ both call it, so a green test == a working demo)
├── demo/                            runnable per-task entrypoints (argparse + __main__)
│   ├── demo_text_to_image.py          Call 1: text -> image
│   ├── demo_image_edit.py             Call 2: image + text -> image
│   ├── demo_4chip.py                  Call 1, tensor-parallel across a 4-chip QB2 mesh (tp=4, all-resident)
│   └── demo_server.py                 2-chip warm-server REPL (superseded by demo_4chip.py; kept for 2-chip boxes)
├── tests/
│   ├── e2e/                         end-to-end correctness + perf tests
│   │   ├── test_text_to_image_e2e.py, test_image_edit_e2e.py   correctness (PCC vs HF golden)
│   │   ├── test_text_to_image_perf.py, test_image_edit_perf.py, test_4chip_perf.py, test_server_perf.py, test_main_perf.py
│   │   └── test_trace_and_host_op.py    host_op_selftest + trace_capture_selftest
│   └── pcc/                         per-component PCC tests, one per graduated stub
└── e2e_plan.json                    planner output (task heads, coverage map)
```

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

Every routed stub is real ttnn (no torch host-compute in the hot path), every graduated
module on a call's critical path is invoked, and the final image PCC is gated at ≥ 0.95
vs the HF golden (`tests/e2e/`).

## Precision

The e2e error of this pipeline is dominated by **iterative-diffusion trajectory
divergence**, not per-matmul precision — over many CFG denoise steps the TT and the
independent HF-golden trajectories drift apart, and that drift is what moves the final
PCC, not rounding in any one matmul. That's why the DiT tolerates aggressive quantization:

| Component | Precision | Notes |
| --- | --- | --- |
| DiT activations | bf16 | |
| DiT linear weights | bf8_b (HiFi4, `fp32_dest_acc_en`) | dropping from fp32 barely moves PCC (0.9947 → 0.9922); ~2× faster |
| DiT attention scores | fp32 | kept for softmax stability |
| DiT bf16-limb path (`stub.limb=True`) | off | no gain here — same trajectory-divergence reason |
| VAE conv / group-norm weights | bf16 | fp32 accumulation, fp32-internal group norm |
| Text encoder (Qwen2.5-VL) | fp32, emulated hi/lo limb | fp32 doesn't fit host RAM directly; TT encoder clears ~1.0 PCC vs the bf16 reference |
| Golden reference | bf16 HF pipeline | identical seed / steps / guidance / size / prompt |

## Run

4-chip tensor-parallel, all-resident warm server (needs a QB2 4-chip mesh). Warms up
once, then prompts for text at a REPL — see Performance below for numbers.

```bash
# 512px (HF steps=50) — 15.4s end-to-end after warmup
./python_env/bin/python -m models.demos.vision.generative.longcat_image.demo.demo_4chip \
    --steps 50 --size 512 --max_length 512 --cq 2

# 1024px (HF reference resolution) — 43.6s end-to-end after warmup
./python_env/bin/python -m models.demos.vision.generative.longcat_image.demo.demo_4chip \
    --steps 50 --size 1024 --max_length 512 --cq 2
```

One HF default we do **not** match: `enable_prompt_rewrite` — HF rewrites the prompt via
the encoder's autoregressive `generate()` before encoding; the TT path skips it, so images
correspond to HF with prompt-rewrite off.

## Performance (tp=4, all-resident, 4 chips)

Measured on QB2 (`sjc2-qb2-9b22`), HF-reference 50 steps, warm trace + 2CQ steady-state
(after the one-time warmup). One 1×4 `FABRIC_1D_RING` mesh, everything **resident** with
**no weight reloads**: the DiT is tensor-parallel tp=4, and the fp32 text encoder is ALSO
tensor-parallel tp=4 (column q/k/v/gate/up, row o/down + all_reduce, one GQA group/chip)
shrinking its ~28 GB to ~7 GB/chip so it co-fits with the ~1.5 GB/chip DiT shard + VAE
(`demo/demo_4chip.py`):

| Setting | text-enc ×2 | denoise / step | denoise (50 steps) | VAE | end-to-end |
| --- | --- | --- | --- | --- | --- |
| 512×512 / 50 steps | ~1.14 s | 279 ms | 14.0 s | 0.31 s | **15.4 s** |
| 1024×1024 / 50 steps | ~1.16 s | 819 ms | 40.9 s | 1.45 s | **43.6 s** |

Text encode is warm/resident (~0.58 s/branch, resolution-independent — no per-request
~28 GB reload). One-time warmup (encoder upload + DiT trace capture + VAE warm): ~90 s
(512) / ~140 s (1024), paid once, not per request. Needs 4 chips.

A follow-up optimization fused the encoder's emulated-fp32 limb matmul (`_emul_linear`,
q/k/v/o projections): its two hi/lo-limb calls that shared weight `wh` are now one stacked
matmul instead of two, halving that weight's DRAM traffic (it was memory-bound at
M=192 tokens ≪ K=N=3584) — exact, no precision change, device_ms 1719.9 → 1672.9 (~2.7%).

### Correctness (PCC)

| Check | PCC | Gate |
| --- | --- | --- |
| DiT tp=4 forward vs. reference | 0.998 | — |
| Text encoder tp=4 `last_hidden_state` vs. reference | 0.9986 | — |
| e2e latent, tp=4 all-resident (512px/50 steps) | 0.987 | ≥ 0.95 |
| e2e pixel, tp=4 all-resident (512px/50 steps) | 0.991 | ≥ 0.95 |
| Encoder `_emul_linear` fuse (exactness check) | 0.994 | — |

Images are visually confirmed coherent and prompt-accurate at both 512px and 1024px
across back-to-back warm requests.
