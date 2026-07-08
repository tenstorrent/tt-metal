# HunyuanVideo-1.5 — real-weight bring-up & video generation on Tenstorrent

This package loads the **real** `tencent/HunyuanVideo-1.5` DiT into the ttnn port and
drives end-to-end video generation. It upgrades the per-component bring-up (which used
random weights) to validated real-weight execution on Blackhole.

## Results (single Blackhole, `--mesh 1,1`)

| What | Result |
|---|---|
| hyvideo→diffusers weight converter | loads 8.33 B params, `missing=0 unexpected=0` |
| ttnn DiT vs diffusers golden, N=2 fp32 | PCC **0.999998** |
| ttnn DiT vs diffusers golden, N=24 fp32 | PCC **0.999913** |
| ttnn DiT, N=54 fp32 | **OOM** (~34 GB DRAM ceiling) |
| ttnn DiT, N=54 **bf16** | PCC **0.967557** (full model, one chip) |
| Full pipeline video (CPU, real weights) | coherent t2v ("cat on grass") |
| TT-driven video (DiT loop on Blackhole) | completes (8 on-device forwards) at tiny res + truncated text |

## Key findings (baked into the code)
- **Patch-embed is 65-channel** (concat-condition: 32 latent + 32 cond + 1 mask), NOT
  the config's nominal `in_channels=32`. `weights.CONFIG` uses `in_channels=65`.
- **t2v** is signaled by **all-zero `image_embeds`** (`is_t2v = torch.all(image_embeds==0)`);
  the byt5 `context_embedder_2` path is **mandatory** in the forward.
- **Token-refiner uses a fused QKV** (`self_attn_qkv`, 6144×2048) → split to `to_q/to_k/to_v`
  in the converter (the only non-rename in `hyvideo_to_diffusers.py`).
- The stubs upload **fp32**; 54 layers fp32 (~33 GB) don't fit one chip's ~34 GB DRAM.
  `coerce_bf16()` (in `tests/e2e/test_real_weight_pcc.py`) monkeypatches ttnn
  `float32→bfloat16` at build+run so full-54 fits (~16.6 GB) — no stub edits needed.

## Downloads (auto-fetched by the code; set `HF_HUB_DISABLE_XET=1` to avoid xet stalls)
- `tencent/HunyuanVideo-1.5` `transformer/720p_t2v/` (~33 GB) — the real DiT weights.
- `hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v` (~50 GB) — full
  diffusers-format pipeline (DiT + VAE `AutoencoderKLHunyuanVideo15` + Qwen2.5-VL text
  encoder + ByT5 + `FlowMatchEulerDiscreteScheduler` + `ClassifierFreeGuidance`).
  NOTE: the diffusers docstring says `...-480p_t2v` but the real id has `-Diffusers-` in it.

## How to run
```bash
# real-weight PCC (full model, bf16, one chip)
HY_N=54 HY_BF16=1 pytest models/demos/hf_eager/hunyuanvideo_1_5/tests/e2e/test_real_weight_pcc.py -s

# TT-driven generation smoke test (one chip: needs truncated text at tiny res)
HY_H=64 HY_W=64 HY_FRAMES=1 HY_STEPS=4 HY_TRUNC=16 HY_OUT=./out \
  pytest models/demos/hf_eager/hunyuanvideo_1_5/tests/e2e/test_stage2b_gen.py -s

# high-quality reference video (pure CPU diffusers pipeline, no ttnn)
python - <<'PY'
import torch; from diffusers import HunyuanVideo15Pipeline
p=HunyuanVideo15Pipeline.from_pretrained("hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v", torch_dtype=torch.bfloat16)
f=p(prompt="A cat walks on the grass, realistic", height=480, width=832, num_frames=13, num_inference_steps=50).frames[0]
PY
```

## RESUME ON QB2 — full-resolution TT video (the next phase)
Full-res TT-driven generation is DRAM-bound on one chip. On QB2 (4× Blackhole, 4×34 GB):
1. **Shard the DiT across the 4-chip mesh** (the `shard`/TP rung skipped in bring-up — bring-up
   used `--mesh 1,1`, TP=1). Re-run bring-up/emit-e2e with `--mesh 2x2` (NOTE: emit-e2e's
   `--mesh` parser only accepts the `NxM` form, not `2,2`), then the `_stubs/*.last_good_sharded`
   bodies do `ShardTensorToMesh` + `all_reduce`. `ttnn.set_fabric_config(FabricConfig.FABRIC_1D)`
   before opening the mesh.
2. In `test_stage2b_gen.py`, drop `HY_TRUNC` (no text truncation) and set real resolution
   (e.g. 480×832, 13 frames, 50 steps). With 4-chip DRAM the weights + full-length text-stream
   activations fit.
3. Alternative if staying single-chip: fp8 weights (~8 GB) or activation streaming / the
   trace+2CQ resident path (`tt/pipeline.py` already exposes the trace hooks).

Everything else (converter, real-weight loader, the CPU pipeline, the ttnn-DiT adapter) is
device-count-agnostic and reused as-is.
