# HunyuanImage-3.0 — status & model details

## Branch
- **Pushed (shareable) — `tenstorrent/tt-metal`:** `nkira/hunyuanimage3-bringup_bh_glx`
  https://github.com/tenstorrent/tt-metal/tree/nkira/hunyuanimage3-bringup_bh_glx
  (native-TTNN model dir + README + `UNIFIED_STACK.md` lever ladder; 31 commits, all authored by Nadim)
- **Working branch (box):** `hunyuan-image3-unified` @ `~/nkira/tt-metal-xtts` — its own `build_Release` + `python_env` (run `./python_env/bin/python`).
  - **Box:** `bh-glx-exp-b04u14` (alias `galaxy-home`) — a Blackhole 6U Galaxy: 32 chips, `MeshShape(8,4)`, FABRIC_1D. Reach via `ssh tt-admin@100.89.226.88`. Fabric is recovered/durable (survives reset).

## Model
`tencent/HunyuanImage-3.0` (`HunyuanImage3ForCausalMM`, `model_type=hunyuan_image_3_moe`) — ~80B-total / ~13B-active MoE **text→image**.
- 32 decoder layers, hidden 4096, 32 Q-heads / 8 KV-heads, head_dim 128, vocab 133120.
- MoE: **64 routed + 1 shared** SwiGLU experts, **top-8** routing; 2D-RoPE, qk-norm, RMSNorm.
- Render = diffusion-in-transformer: FlowMatch (Euler) loop, **50 steps, CFG=2**, timestep-conditioned velocity head, VAE decode → PNG. Weights load from HuggingFace.

## Parallelism (BH Galaxy `MeshShape(8,4)` = 32-chip mesh, FABRIC_1D)
- **Attention / dense = TP=8** — sharded across the length-8 axis, replicated across the 4-axis.
- **MoE = EP=32** — 64 experts sharded **2/chip across all 32 chips** + shard-shared expert. The 4-axis carries **expert parallelism + a 2nd all-reduce**, NOT classic DP (a single image has no batch to data-parallelize).
- Single-chip fabric-free path: `HY3_SINGLE_CHIP=1` (per-layer bring-up / PCC only; full 32-layer stack needs the mesh).

## Current perf
**~58 s/image warm** (SP + on-device VAE, resident server) @1024²/50-step — down from ~67 s (SP + host VAE), ~84 s (default), 216.5 s cold, 351.4 s hybrid.
- **SP** (`HUNYUAN_SP`): diffusion loop token-sharded EP=32→8 → 623 ms/step, −31%, PCC 0.99999 (`1be484e7`).
- **On-device VAE** (`HUNYUAN_ONDEVICE_VAE`): `AutoencoderKLConv3D.decode` ported to the (8,4) mesh (H/W spatial-shard) → **25.2 s** in-render vs 36 s host, real-weight PCC **0.99718** (`148085fb05`→`a40fc8c057`). One-time 19.4 s decoder build cached across images. Next VAE lever = conv3d block-size sweep (blockings are untuned `C_in_block=32`). See `PERF.md`.

## Roofline & target
- Dense-64 (what actually runs): ~65.5 PFLOP/image (MoE ~97% of fwd FLOPs). Sparse top-8: ~10.6 PFLOP. Galaxy peak ~10.6 PFLOP/s bf16 (~3.6× the 3×H100 aggregate).
- **Near-term target ~40-50 s** · **Aggressive ~15-20 s** (dense-MoE @ ~40% MFU) · **Floor ~5-8 s** needs sparse dispatch — DEAD on TT.
- flux2's 5-7 s is a dense ~6-12B DiT — doesn't transfer to an 80B MoE forced dense. Same-model GPU ref: ~137 s on 3×H100 (soft provenance).

## Related port — IGN (`tenstorrent/tt-metal` PR #50968, `ign-christy`)
A parallel Tenstorrent port of the **same model** (HunyuanImage-3.0) — more complete (base T2I + Instruct I2I + Distil + SigLIP2 vision + recaption). Open PR, ready for review.
- **Hardware:** BH **4 chips (2×2), SP×TP + EP** (vs our 32-chip Galaxy TP=8 + EP=32).
- **Base T2I 50-step ~1024² (4,226 tok):** **263 s E2E** = 4.65 s/step denoise + **17.5 s on-device spatial-parallel VAE**.
- **Us vs them:** ~84 s warm on 32 chips — faster in absolute wall-clock, but on 8× the hardware. **Per-chip they're ~1.6-2.5× more efficient** (throughput/chip 3.42 vs our 1.34 img/hr/chip). Our lead is scale; their port is more hardware-efficient.
- **Already implement three items on our lever list** — **on-device VAE (spatial-parallel)**, **spatial-parallel (SP)**, and an **8-step Distil variant** (model-level ~6× denoise cut). Reference implementations to learn from / coordinate on (two HunyuanImage-3.0 ports now live on the same repo).

## Bottleneck (latest Tracy — `HUNYUAN_TRACY_ops_perf.csv`, 224,960 rows)
**CCL-bound.** ReduceScatter (16.9%) + AllGather (13.3%) = **~30% of device-kernel time, ahead of matmul (23%)**. ~55% of wall-clock once cross-chip sync is counted (~3 s device compute vs ~6.4 s wall/step). Full split: CCL 30% · matmul 23% · SDPA 13% · elementwise 12% · layout ~14%.

## Near-term levers (ranked)
1. **On-device VAE decode** — kill the ~36 s host tail (~43% of a warm render). Biggest single win.
2. **AG+MM / RS+MM fusion** — fuse the collectives into the matmuls so they overlap compute. NOT yet tried; directly targets the #1 (CCL) bottleneck. *(Teja's suggestion — top structural lever.)*
3. **1D matmul + block-size sweep** — matmuls currently use ttnn default (2D) program configs; only grid (`MM_FULLGRID`) / dtype / fidelity tuned. No `Matmul1D` / `per_core_M/N/K` / `out_subblock` sweep yet.
4. **Traced on-device render** — the shipped render runs eager; wire trace into the diffusion loop (trace + 2CQ validated on prefill/decode).
5. **CFG-parallel** — batch cond+uncond into one bsz=2 forward (halves per-step collectives). Gated opt-in today.

## Dead ends (don't chase)
- **Sparse top-8 MoE:** ~13-47× slower (datamove-bound). Kept opt-in; dense is default.
- **Per-op device_ms** optimization for the image path: doesn't track render time (measure traced wall-clock).

## How to run (from repo root)
```bash
# perf: full-depth on-device render latency (emits ONDEVICE_E2E_TOTAL_LATENCY_S)
HUNYUAN_VAE_AUTOCAST=bf16 HUNYUAN_CCL_LINKS=2 ./python_env/bin/python -m pytest -o timeout=0 \
  models/demos/vision/generative/hunyuanimage_3_0/tests/e2e/test_host_glue_stage3_perf.py -s
# per-component PCC (single-chip + TP=8 sharded)
./python_env/bin/python -m pytest models/demos/vision/generative/hunyuanimage_3_0/tests/pcc -s
# trace + 2CQ contract
./python_env/bin/python -m pytest models/demos/vision/generative/hunyuanimage_3_0/tests/e2e/test_trace_2cq.py -s
# demo render
./python_env/bin/python -m models.demos.vision.generative.hunyuanimage_3_0.demo.demo_image3_t2i \
  --prompt "..." --steps 50 --size 1024x1024 --out out.png
```
Key env: `HUNYUAN_EP_FULLMESH` (EP=32, default on) · `HUNYUAN_CCL_LINKS` (default 2) · `HUNYUAN_MM_FULLGRID` (default on) · `HUNYUAN_VAE_AUTOCAST=bf16` · `HUNYUAN_CFG_PARALLEL` (opt-in) · `HUNYUAN_SPARSE_MOE` (opt-in, dead) · `HY3_SINGLE_CHIP`.

## Deeper detail
`PERF.md` (this folder) has the perf / roofline / target tables; `OPTIMIZATIONS_BY_DATE.md` (this folder) has the chronological breakdown; in-branch `UNIFIED_STACK.md` has the deepest dated lever ladder.
