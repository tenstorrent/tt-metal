# LTX-2.3 Distilled AV — e2e Optimization Plan

**Goal:** close the gap to FastVideo (hao-ai-lab) on LTX video gen, and beat them on the one
lever they leave on the table. Current: ~10s e2e. FastVideo headline: ~4.55s.

**Status:** living doc. Baseline per-phase numbers + per-op profile (in flight) fill the
`TODO(baseline)` slots. L1-vs-L4 ordering is *gated* on that profile — do not finalize before.

Glossary: **GEMM** = dense matrix multiply (transformer linear layers; runs on the matrix
engine/FPU). **VAE** = variational autoencoder; its *decoder* expands the denoised latent back
to 1080p pixels (3D-conv-heavy). **DiT** = diffusion transformer (the denoiser).

---

## 1. The comparison is NOT apples-to-apples

FastVideo 4.55s, confirmed from their `basic_ltx2_distilled_fast_profile.py`:
**1 × NVIDIA B200, 1088×1920, 121 frames, 7 steps (5 main + 2 refine), NVFP4 DiT, FA4 dense
attention, torch.compile.** Same upstream Lightricks LTX-2.3 distilled checkpoint we use.

Ours: **Blackhole 4×8 galaxy, 1088×1920, 145 frames, 10 steps, bf16/HiFi2, dense ring SDPA,
trace-replay.**

| Their advantage | Portable to us? |
|---|---|
| B200 FP4 tensor cores + FA4 Blackhole kernels | **No — silicon.** We can't match their FPU/FP4 throughput. Decision: don't chase it — use **bfp8_b + LoFi** (preserve precision) where they used NVFP4. bfp4_b only helps if we're bandwidth-bound, which we don't think we are (§4 L1). |
| 121 frames vs our 145 (~17% less work) | **Config** (`NUM_FRAMES`). Product decision, not an optimization. |
| 7 denoise steps vs our 10, NVFP4 quant on DiT linears | **Yes — algorithmic. This is the real headroom.** |

**Honest bottom line:** a real chunk of their 2× is silicon + frame count. Portable algorithmic
delta is well under 2×. The one major technique FastVideo has NOT applied to LTX is **sparse
attention** — that is where we can run LTX *faster than they currently do*.

---

## 2. Confirmed current state (read from source)

LTX was forked from WAN's optimized tt_dit framework and already has, often *extended* past WAN:
ring SDPA, fused QKV / FFN+addcmul, device-resident solver, trace replay, coresident eviction,
per-mesh SDPA chunk tuning, scale_shift baking, rope caching.

Genuine gaps:

| Gap | Evidence | Status |
|---|---|---|
| DiT linear quantization | `attention_ltx.py:187,206` HiFi2 + bf16; no `quant_config.py` | **Missing** |
| FSDP enabled in a preset | wired (`transformer_ltx.py:74,95`) but no preset sets `is_fsdp:True` | **Off** |
| Sparse / sliding-tile attention | `attention_ltx.py:143` dense `SDPAProgramConfig` | **Missing (so is WAN)** |
| 10 vs 7 denoise steps | `pipeline_ltx_distilled.py:24-25` (stage1=7, stage2=3) | **3 extra steps** |

Step detail: stage-1 sigmas `[1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0]`
(7 steps — 4 of them clustered near σ=1), stage-2 `[0.909375, 0.725, 0.421875, 0.0]` (3 steps).

---

## 3. Baseline (TO FILL from Phase A / B profile)

- Clean e2e (uninstrumented, gen#1 steady-state replay): `TODO(baseline) s`
- Per-phase breakdown (gen#1): encode / transformer-prepare / **stage-1 denoise** / upsample /
  **stage-2 denoise** / VAE prepare / VAE decode / audio decode / export — `TODO(baseline)`
- DiT bound class (per-op FPU/SFPU counters + tt-npe NoC/DRAM BW): **GEMM-bound vs attention-bound**
  — `TODO(profile)`. **This resolves the L1-vs-L4 ordering fork.**

---

## 4. Levers, ranked

### Tier 1 — actionable now, no new kernels

**L1. Quantize DiT linears → bfp8_b weights + activations + LoFi compute.**
Decision (locked): **bfp8_b + LoFi on the DiT linears wherever FastVideo used NVFP4. NOT bfp4_b.**
Port `models/tt_dit/pipelines/wan/quant_config.py::all_bf8_lofi()` → new `pipelines/ltx/quant_config.py`.
Hook surface exists: per-layer `.compute_config` on `Linear`/`ColParallelLinear`,
`mm_compute_kernel_config` on `LTXAttention`.

- **Where the speed comes from: HiFi2 → LoFi = 2 matmul passes → 1 (~2× on the GEMM math when
  compute-bound).** bf16 → bfp8_b halves weight bytes (bandwidth/L1 footprint) on top.
- **Why bfp8 not bfp4:** on Tensix, LoFi is 1 pass for *both* — bfp4 does NOT make the matrix
  engine faster, it only halves weight bandwidth again. So if the DiT is compute-bound, bfp4 buys
  ~nothing in throughput; bfp8 is strictly better (same speed, more precision). **Revisit bfp4
  only if tt-npe shows the DiT is weight-bandwidth-bound** (§5).
- **SDPA stays bf16/HiFi2 (fully unquantized) to start** — FastVideo kept attention higher
  precision (NVFP4 on linears only). WAN's `all_bf8_lofi` casts SDPA *input* to bf8 (still HiFi2);
  treat that as a separate later bandwidth-only tweak, not part of the first landing.
- Carve-out (confirmed, already in `all_bf8_lofi`): `self_attn_out` keeps bf16 weights — the fused
  matmul+addcmul kernel needs ternary inputs (residual, gate) to match weight tile format
  (`quant_config.py:120-136`, LTX uses the same fused kernel `attention_ltx.py:337`).
- Port effort: **med** — rewrite `apply_quant_config`'s block-walk (`quant_config.py:228-254`) for
  LTX module names (LTX block structure differs from WAN's `attn1/attn2/ffn`).
- Gate: PCC + visual vs torch oracle.

**L2. Cut denoise 10 → 7 steps** (match their 5+2 sigma schedule, same checkpoint).
- Expected: **~1.43× on DiT** (linear in steps) IF quality holds.
- **Risk: the one lever that can degrade output.** Same upstream checkpoint runs 5+2 at FastVideo
  → strong evidence it's viable. But our extra steps sit in the near-σ=1 region that may do real
  refinement. **Validate hard before claiming.** Effort: low to change, high to validate.

**L3. Enable FSDP in the 4×8 preset.** Fractures FFN+attn weights across SP → frees DRAM/L1.
- Low effort (flag flip + validate). Win is **memory headroom**, only a latency win if it unblocks
  a larger tile / less reshard. Don't oversell.

### Tier 2 — biggest ceiling, needs kernel work

**L4. Block-sparse / sliding-tile attention (STA/VSA analog).** The lever to *beat* them:
FastVideo ships STA/VSA for Hunyuan/Wan but runs **dense FLASH_ATTN for LTX**. At 145 frames @
1080p the DiT is attention-dominated. STA shows up to 3.5× attention-latency / ~2× e2e at
iso-quality on comparable models. 3D-local-window block sparsity maps natively to TT tile compute.
- Expected: **1.4–2.0× on attention** (window-config dependent; training-free mask-search = floor,
  finetune = ceiling).
- Effort: **high** — new TT attention kernel + mask search / finetune. The moonshot, not the quick win.

### Tier 3 — cleanup (~1.05–1.2× each)
- VAE decode: ensure trace-replayed; skip tiling if it fits (FastVideo does).
- Fold RoPE into Q/K norm kernel (`attention_ltx.py:433-470` runs it separate; WAN folds it,
  `attention_wan.py:372-387`) — verify INTERLEAVED-layout correctness first.

### Skip (with reason)
- **Feature caching (TeaCache/FBCache):** no payoff at 5–10 steps; cache hit rate too low to beat
  quality cost. Absent in FastVideo's LTX path too.
- **NVFP4 GEMM / FA4:** Blackhole-incompatible silicon.
- **`exp_ring_sdpa` / `_sdpa_input_dtype` ring-cast (WAN #2/#4):** NOT applicable to our mesh.
  `use_exp_ring_sdpa` gates on `tp_factor==4 and sp_factor==32` (`attention_wan.py:143-144`) — that
  is the **128-chip 4×32** config. Our target is **4×8 = 32-chip galaxy** (sp_factor=8) → never
  fires; we use the regular `ring_joint_scaled_dot_product_attention`. `_sdpa_input_dtype` only
  matters once SDPA inputs are bf8, which L1 defers. Revisit only if we ever target a 4×32 mesh.

---

## 5. The critical fork

L1 (quant) and L4 (sparse attn) attack different bottlenecks. Resolve with the profile:
- High FPU util + FFN/QKV-dominated → **quant (L1) first.**
- SDPA dominates device time → **sparse attention (L4) first.**

The profile also settles the **bfp8-vs-bfp4** question for L1:
- DiT compute/FPU-bound → bfp8 is correct (bfp4 adds no compute throughput).
- DiT weight-bandwidth-bound (tt-npe DRAM BW high, FPU util low) → bfp4 would be faster; revisit.

Per-RISC + FPU/SFPU counters + tt-npe bandwidth (Phase B) settle all of this with numbers.

---

## 6. Sequencing

1. **Baseline + per-op profile** (in flight) → DiT share + GEMM-vs-attention bound.
2. **L1 quant** — fastest ROI, no new kernels. PCC gate.
3. **L2 step cut** — validate quality aggressively; ship if it holds.
4. **L4 sparse attention** — the differentiator; its own scoped effort once Tier-1 is banked.
5. Tier-3 cleanup opportunistically.

---

## 7. Open questions

- [ ] Is 145 frames a hard product requirement, or normalize to 121 for the head-to-head?
- [ ] L1-vs-L4 ordering — pending profile (§5).
- [ ] Does our VAE expose `vae_t_chunk_size` (WAN has it)? Check if VAE decode is a surprise fraction.

---

## Sources
- FastVideo profiling script: `examples/inference/basic/basic_ltx2_distilled_fast_profile.py`
  (5+2 steps, 1088×1920, 121f, `NVFP4Config()`, `FASTVIDEO_ATTENTION_BACKEND=FLASH_ATTN`)
- https://haoailab.com/blogs/fastvideo_realtime_1080p/ , https://haoailab.com/blogs/fastvideo-dreamverse-release/
- STA: https://arxiv.org/html/2502.04507v3 — VSA: https://arxiv.org/abs/2505.13389
- WAN quant template: `models/tt_dit/pipelines/wan/quant_config.py`
- WAN FSDP presets: `models/tt_dit/pipelines/wan/pipeline_wan.py:37-101`
