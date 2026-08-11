# tencent/HunyuanImage-3.0 — end-to-end TTNN pipeline (text → image)

A real, on-device TTNN pipeline for `tencent/HunyuanImage-3.0` (`HunyuanImage3ForCausalMM`, `model_type=hunyuan_image_3_moe`) — an ~80B-total / ~13B-active mixed-MLP MoE text→image model. It reproduces the model's real diffusion render — **(prompt) → 1024² image** — by driving the 32-layer MoE transformer with a FlowMatch (Euler) denoising loop, classifier-free guidance (`cfg_factor=2`), a timestep-conditioned velocity head, and VAE decode, entirely on the graduated native TTNN stubs plus native TTNN glue for the leaf conv heads (`patch_embed` / `final_layer`).

The bring-up first graduated the transformer decoder-block as **Call-1** (`hunyuan_image3_transformer_prefill`), PCC-gated against the exact HF forward; the shipped path is the image render built on those same graduated layers. A secondary **incremental-KV transformer decode** path (KV cache + causal single-token attention) is also wired.

## Layout

```
models/demos/vision/generative/hunyuanimage_3_0/
  _stubs/                         the 3 graduated native-TTNN stubs, composed along the real HF nesting
    image3_decoder_layer.py         RMSNorm + GQA attn + 2D-RoPE + qk-norm + SDPA + residuals  (== HunyuanImage3DecoderLayer)
    mo_e.py                         shared + 64 routed SwiGLU experts, EP=32, merged 2D matmuls  (== HunyuanMoE)
    top_k_gate.py                   softmax + top-8 router + l_aux                               (== HunyuanTopKGate)
  tt/
    pipeline.py                     the ONE shared prefill/decode forward (build_pipeline, run_prefill/run_decode,
                                    trace + host-op selftests). Imported by BOTH the prefill demo and the e2e test.
    gen_image.py                    text→image diffusion driver (FlowMatch loop + CFG + velocity head + VAE)
    host_glue_stage3.py             on-device head-glue render — hidden stays on device (PatchEmbedTT + FinalLayerTT)
    host_glue_tt.py                 native-TTNN patch_embed + final_layer (velocity head) ports
  demo/                           demos + resident servers (all import the same tt/ forwards — a green test == a working demo)
  tests/e2e/                      prefill gates, decode, trace+2CQ, text→image PCC + latency, host-glue PCC + perf
  tests/pcc/                      per-component PCC (single-chip + TP=8 sharded)
  UNIFIED_STACK.md                perf-ladder / lever source of truth
  e2e_plan.json                   the planner sketch (Call-1)
```

`tt/gen_image.py` + `tt/host_glue_stage3.py` are the ONE shared image forward, imported by BOTH the demos and the perf tests; `tt/pipeline.py` is the same for the prefill/decode path. A green test therefore guarantees a working demo (no drift).

## Run

All commands from the tt-metal repo root. `HY3_SINGLE_CHIP=1` runs fabric-free on 1 device (per-layer bring-up); full renders need the mesh.

```bash
# e2e gate test: native ttnn (Gate 1) + all stubs invoked (Gate 2) + PCC vs HF golden (Gate 3)
./python_env/bin/python -m pytest \
  models/demos/vision/generative/hunyuanimage_3_0/tests/e2e/test_e2e_prefill.py -s

# per-component PCC (single-chip + TP=8 sharded)
./python_env/bin/python -m pytest models/demos/vision/generative/hunyuanimage_3_0/tests/pcc -s

# trace + 2CQ contract (Command 3): host-free capture/replay PCC + zero host aten ops
./python_env/bin/python -m pytest \
  models/demos/vision/generative/hunyuanimage_3_0/tests/e2e/test_trace_2cq.py -s

# text→image demo (hybrid path)
./python_env/bin/python -m models.demos.vision.generative.hunyuanimage_3_0.demo.demo_image3_t2i \
  --prompt "a red panda astronaut, studio lighting" --steps 50 --size 1024x1024 --out panda.png

# warm resident render server (build stack + conv heads once, then render prompts warm) — ~29.8 s/image
HUNYUAN_SP=1 HUNYUAN_CFG_PARALLEL=1 HUNYUAN_ONDEVICE_VAE=1 HUNYUAN_VAE_WARMUP=1 \
HUNYUAN_VAE_AUTOCAST=bf16 HUNYUAN_CCL_LINKS=2 ./python_env/bin/python -m \
  models.demos.vision.generative.hunyuanimage_3_0.demo.warm_render_server

# live queue-driven server: append to $HUNYUAN_DEMO_DIR/queue.jsonl -> $HUNYUAN_DEMO_DIR/out/<id>.png
export HUNYUAN_DEMO_DIR=/tmp/hunyuan_demo
./python_env/bin/python -m models.demos.vision.generative.hunyuanimage_3_0.demo.demo_live_server
```

## Pipeline (text → image)

`generate_image_ondevice` chains the following on the full `MeshShape(8, 4)` mesh (TP=8 + EP=32, `FABRIC_1D`); hidden never leaves the device:

1. **Setup (once):** HF-tokenize the prompt → prefix/suffix token embeddings; build the 2D image RoPE cos/sin + the block attention mask; upload the static suffix; build `PatchEmbedTT` + `FinalLayerTT`.
2. **FlowMatch (Euler) denoising loop, `cfg_factor=2`, per step:**
   a. `PatchEmbedTT(latent, time_embed(t))` → image tokens on device.
   b. assemble `inputs_embeds` on device via ROW_MAJOR `concat([prefix, img, suffix])` → tilize.
   c. run the **32 graduated decoder layers** (hidden stays on device): per layer RMSNorm → GQA self-attn (fused-QKV, 2D-RoPE, qk-norm, SDPA, o_proj + fused all-reduce) → residual → post-attn RMSNorm → MoE (`top_k_gate` softmax + top-8 → shared + 64 routed SwiGLU experts, EP=32, all-reduce) → residual.
   d. slice the image-position hidden → `FinalLayerTT` → velocity `diffusion_prediction` `[cfg,32,64,64]` (the only per-step download).
   e. FlowMatch scheduler step on the CFG-combined velocity → next latent.
3. **VAE decode** the final latent → 1024² pixels → PNG. Now **on-device** (mesh conv3d + distributed reduce-moments GroupNorm) under `HUNYUAN_ONDEVICE_VAE=1` — 4.0 s warm; host `model.vae.decode` remains the fallback/oracle.

All 3 graduated stubs (`image3_decoder_layer`, `mo_e`, `top_k_gate`) are invoked in the real forward path (**Gate 2: 3/3**).

## Results (32-layer 1024² 50-step render; prefill gate at N=1 layer, seq_len=64)

| metric | value | gate |
|---|---|---|
| Gate 1 (native) — `host_op_selftest` host aten ops | **0** | fully on device |
| Gate 2 (invoked) — graduated stubs on the forward path | image3_decoder_layer / mo_e / top_k_gate | 3/3 |
| Gate 3 (PCC) — prefill `last_hidden_state` vs HF golden | **0.99977** | ≥ 0.95 |
| per-component PCC, single-chip (decoder / mo_e / gate) | 0.99999 / 0.9996 / 1.0 | ≥ 0.95 |
| per-component PCC, TP=8 sharded (decoder / mo_e / gate) | 0.99999 / 0.9940 / 1.0 | ≥ 0.95 |
| Command 3 (trace) — prefill / decode trace PCC | 1.0 / 1.0 | host-free |
| head-glue block PCC (velocity / patch_embed / stage-3 on-device velocity) | 0.99986 / 0.99973 / 0.99989 | ≥ 0.99 |
| text→image path PCC (final-latent / decoded-image, reduced depth) | pass | ≥ 0.95 |
| **E2E render — on-device, warm (current)** | **~29.8 s/image** (loop 25.8 s @ ~511 ms/step + on-device VAE 4.0 s) | perf |
| E2E render — on-device warm (head-glue only, historical) | ~84 s/image | perf |
| E2E render — on-device, cold (one-time step-1 compile) | ~170–196 s/image | perf |
| E2E render — hybrid (pre on-device head-glue) | 351.4 s/image | perf |

Sources: Gate 1/2/3 + Command 3 + per-component PCC — `RUN_REPORT.md` / `e2e_plan.json`; TP=8 sharded PCC + wall-clock numbers — `UNIFIED_STACK.md`; head-glue block PCCs — perf-lever commit history. `TT_HY3_PCC` gate = 0.95, one 6U Blackhole Galaxy.

## Performance

End-to-end **wall-clock s/image** is the shipped metric — per-op Tracy device_ms does not track render time (the diffusion step is ~55% collective/sync-bound, so shaving per-op cost does not move it). The decisive first lever was the **on-device head-glue port**: keeping the hidden state on the mesh (downloading only the small velocity tensor, not the full per-step hidden) cut the per-step from **5548 → 947 ms/step**, taking a render from **hybrid 351.4 s/image → on-device 216.5 s cold / ~84 s warm** (resident server; the ~36 s bf16 VAE decode was the host-side remainder and the next lever). Source: `UNIFIED_STACK.md`.

**Subsequent levers took warm ~84 s → ~29.8 s/image** — all PCC-gated, default-on under the ship env:

- **Sequence parallelism** (`HUNYUAN_SP`, token-shard, EP=32→8): 906 → 623 ms/step.
- **CFG-parallel** (`HUNYUAN_CFG_PARALLEL`): cond+uncond as ONE bsz=2 forward — 645 → 506 ms/step (now default, not just for a resident server).
- **On-device VAE** (`HUNYUAN_ONDEVICE_VAE`) + **distributed reduce-moments GroupNorm** (an O(num_groups) scalar all-reduce instead of a ~1 GB full-spatial gather) + **VAE warm-up** (pre-compiled at model setup): per-image VAE decode **~36 s host → 4.0 s on-device** — closes the "VAE is the host-side remainder" note above.
- **MoE**: bf4_b experts, the 64 per-expert loops folded into 2 batched 2D matmuls, SiLU fused into the SwiGLU multiply. **lm_head** bf8_b; multi-core argmax.

Net: **~511 ms/step loop + 4.0 s on-device VAE = ~29.8 s warm/image** @1024²/50-step (cold first image ~170–196 s is the one-time step-1 kernel compile, amortized). Explored-and-reverted negatives (documented): lower-TP re-shard (OOM), VAE spatial split-reduce (wash), KV all-gather fuse (wash), narrow-`sel` (tile-infeasible at expert_inter/32-block `epd=8`), bf8 CCL cast (regression), FSDP (deadlock).

Ship env (default-on for the ~29.8 s number): `HUNYUAN_SP=1 HUNYUAN_CFG_PARALLEL=1 HUNYUAN_ONDEVICE_VAE=1 HUNYUAN_VAE_WARMUP=1 HUNYUAN_VAE_AUTOCAST=bf16 HUNYUAN_CCL_LINKS=2`. Off by default: `HUNYUAN_SPARSE_MOE` (correct, PCC 1.0, but ~47× slower on the image path). Note: Tencent's `Instruct` / `Instruct-Distil` (8-step) checkpoints are **image-to-image**, not text-to-image — out of scope for this t2i pipeline.

## Determinism

The render is a deterministic feed-forward: a fixed `--seed` seeds `torch.Generator` for the initial latents, so `(prompt, seed, steps, size)` reproduces the same image. The graduated transformer is exact — per-component PCC ≥ 0.9996, prefill e2e 0.99977.

## Trace + 2CQ (Command 3)

`PIPELINE_STAGES = ["prefill", "decode"]`. `build_pipeline(device, model)` returns the resident `HunyuanImage3Pipeline`, exposing per-stage `trace_setup` / `trace_step` / `write_inputs` hooks. `trace_capture_selftest()` captures each stage host-free and PCC-checks the traced output (prefill 1.0, decode 1.0); `host_op_selftest()` runs the forward under the host-op observer and asserts **zero host aten ops** (fully on device). The transformer stages are the trace + 2CQ-validated path; the diffusion loop supports host-free traced replay via `HUNYUAN_T2I_TRACE` (`gen_image`), while the shipped on-device head-glue render currently runs eager (hidden resident on device).
