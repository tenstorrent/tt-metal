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

```bash
# Call 1: text -> image (writes a PNG). Defaults to 512px/50 steps/512 tokens
# (matches HF except resolution — add --size 1024 to match HF's default exactly).
./python_env/bin/python -m models.demos.vision.generative.longcat_image.demo.demo_text_to_image \
    --prompt "a photograph of a cat sitting on a red sofa" --out my_image.png

# Call 2: image + text -> image
./python_env/bin/python -m models.demos.vision.generative.longcat_image.demo.demo_image_edit \
    --image <path.jpg> --prompt "change the cat to a dog"

# Fastest path: 4-chip tensor-parallel, all-resident (needs a QB2 4-chip mesh) — see Performance below.
./python_env/bin/python -m models.demos.vision.generative.longcat_image.demo.demo_4chip \
    --prompt "a photograph of a cat sitting on a red sofa"

# e2e correctness gates (on device)
./python_env/bin/python -m pytest models/demos/vision/generative/longcat_image/tests/e2e/ -s
```

Common flags (all three demos):

| Flag | Effect |
| --- | --- |
| `--size 1024` | match the HF reference resolution exactly (default 512; use ≥ 512 — 256px is out-of-distribution for this 1024px-class model) |
| `--cq 2` | run the denoise loop under trace + 2 command queues |
| `--compare_golden` | also run the slow CPU HF reference and print e2e PCC (minutes; omit for a normal run) |
| `--profile` | print per-stage wall-clock timing (`LONGCAT_PROFILE=1` env var works too) |

One HF default we do **not** match: `enable_prompt_rewrite` — HF rewrites the prompt via
the encoder's autoregressive `generate()` before encoding; the TT path skips it, so images
correspond to HF with prompt-rewrite off.

E2e gate caps (steps / size / token budget) are small by default for a fast on-device
check and applied identically to the TT run and the (disk-cached) HF golden — override via
`LONGCAT_E2E_{STEPS,SIZE,MAXLEN,GUIDANCE,PROMPT}`. For the bounded per-step device-latency
harness (a small 128px/32-token relative-optimization figure, not the full-resolution cost):

```bash
LONGCAT_PERF_CQ=1 ./python_env/bin/python -m pytest -s \
    models/demos/vision/generative/longcat_image/tests/e2e/test_text_to_image_perf.py
```

## Trace & command queues

The denoise loop is captured **once** as a ttnn trace (`_tt_denoise_traced` in
`tt/pipeline.py`) — both CFG forwards, the guidance combine, cfg-renorm, and the
FlowMatch-Euler step — and replayed per step via `execute_trace`, removing per-op host
dispatch. It falls back to eager on the image-edit path or any trace error.

`--cq 2` additionally enables a **trace + 2CQ** variant (`_tt_denoise_traced_2cq`): CQ1
stages the next step's `temb`/`dt` via DMA while CQ0 replays the trace (CQ1 may only issue
DMA, never a program/kernel). Numerically identical to 1CQ (image PCC 1.0); roughly parity
in speed here since the step is compute-bound with a tiny prefetchable input.

`LongCatImagePipelineTT.warmup()` builds the DiT + VAE stubs and captures this trace once
with dummy shape-matched inputs, so later `run_text_to_image()` calls whose
`max_length`/`height`/`width`/`guidance_scale`/`enable_cfg_renorm`/`cfg_renorm_min` match
replay the resident trace instead of rebuilding (a mismatched request falls back
transparently to the cold path — correctness never depends on the caller remembering
`warmup()`'s exact arguments, only the throughput win does). Call `close()` on shutdown.
The VAE must be warmed **before** the trace is captured — warming it after corrupts its
weights when the trace re-runs (fixed in `tt/pipeline.py`; multi-request warm generation
is verified coherent across back-to-back requests at 512px and 1024px).

## Performance

Measured on QB2 (`sjc2-qb2-9b22`), HF-reference 50 steps, warm trace + 2CQ steady-state
(after the one-time warmup), unless noted.

### Optimization timeline — 512×512, baseline → current

| Milestone | ms/step | Speedup vs baseline | Gate PCC (512px/24-step) |
| --- | --- | --- | --- |
| Baseline (bf16 DiT + bf8_b weights, manual attention, single chip) | 915 ms | 1.00× | 0.9670 |
| + FlashAttention-2 (`ttnn.transformer.scaled_dot_product_attention`) | 662 ms | 1.38× | 0.9719 |
| + 8×8 attention core grid | 643 ms | 1.42× | 0.9794 |
| + tensor-parallel **tp=4** (4-chip QB2 mesh, DiT + encoder both tp=4, all-resident) | **279 ms** | **3.28×** | e2e latent/pixel 0.987 / 0.991 |

PCC actually **improved** alongside speed up through the SDPA step — FlashAttention's fp32
online-softmax accumulation is more accurate than the manual QK^T→softmax→P@V path it
replaced. All milestones stayed at/above the 0.95 e2e gate. (The full-130-core grid was
faster still, 561 ms, but over-partitioned the K reduction and broke PCC to 0.9179 —
reverted; 8×8 is the PCC-safe speed point.)

### 1024×1024 (HF default resolution)

| Config | ms/step | denoise (50 steps) | end-to-end |
| --- | --- | --- | --- |
| Single chip (pre-SDPA/grid baseline) | ~4.24 s | 211.8 s | ~214.9 s |
| tp=4, all-resident (4-chip) | **819 ms** | 40.9 s | **43.6 s** |

Per-step scales ~4.6× from 512→1024 in the single-chip case (image tokens 1024→4096;
attention is O(n²) but only part of the step, so it lands near 4.6×, not 16×).

### Current best — 4-chip tensor-parallel, all-resident

One 1×4 `FABRIC_1D_RING` mesh, everything **resident** with **no weight reloads**: the DiT
is tensor-parallel tp=4, and the fp32 text encoder is ALSO tensor-parallel tp=4 (column
q/k/v/gate/up, row o/down + all_reduce, one GQA group/chip) shrinking its ~28 GB to
~7 GB/chip so it co-fits with the ~1.5 GB/chip DiT shard + VAE (`demo/demo_4chip.py`):

| Setting | text-enc ×2 | denoise / step | denoise (50) | VAE | end-to-end |
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
| e2e, fast 1-step gate (256px) | 0.9931 | ≥ 0.95 |
| e2e, 512px / 24 steps (current, SDPA + 8×8 grid) | 0.9794 | ≥ 0.95 |
| DiT tp=4 forward vs single-chip | 0.998 | — |
| Text encoder tp=4 `last_hidden_state` vs single-chip | 0.9986 | — |
| e2e latent, tp=4 all-resident (512px/50 steps) vs single-chip | 0.987 | ≥ 0.95 |
| e2e pixel, tp=4 all-resident (512px/50 steps) vs single-chip | 0.991 | ≥ 0.95 |
| Encoder `_emul_linear` fuse (exactness check) | 0.994 | — |

The demo generates a coherent, prompt-accurate image at every measured milestone (see the
`e2e PCC=...` line printed on every run, pass or fail); images are also visually confirmed
coherent at 512px and 1024px across back-to-back warm requests and across single-chip vs.
tp=4.
