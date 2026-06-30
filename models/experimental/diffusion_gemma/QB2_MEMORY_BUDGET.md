# DiffusionGemma 26B-A4B on QB2 — memory budget & batch ceiling (#47487)

QB2-only HW enablement for the **causal text backbone** (the Foundation gate the
user scoped: QB2, *not* Galaxy/T3K). This doc sizes the fit and states what is
**verified from code/config** vs **must be measured on device** — every quantity
that cannot be derived statically is marked `MEASURE` rather than guessed (R3).

## Box / mesh

- **QB2 = `bh-qbge-06` = TT-QuietBox 2 (Blackhole)** — a desktop workstation with **4 Blackhole Tensix processors** on **2× `p300` dual-Blackhole PCIe cards** (NOT `p150`, which is a single-chip card), `/dev/tenstorrent/{0..3}`. Product spec: 480 Tensix cores, 720 MB SRAM, **128 GB DDR6 = 32 GB/chip** @ 1024 GB/s; Ryzen 7 9700X + 256 GB DDR5 host.
- In the gemma4 mesh registry QB2 is **`MESH_DEVICE=P150x4 → (1, 4)`**
  (`models/demos/gemma4/demo/text_demo_v2.py:291`; default fallback is also
  `(1,4)`). The model path is **mesh-shape-agnostic**: TP is derived from
  `mesh_device.shape[1]` (`tt/common.py:59-65` → `MeshConfig((1,4), tp=4)`), CCL
  link count is arch-gated (2 on Blackhole, `tt/ccl.py`). **No mesh code edits are
  needed to target QB2** — only `MESH_DEVICE=P150x4` (a tt-metal **mesh-shape launch label** for a 1×4 Blackhole mesh — *not* the card SKU; the cards are `p300`).
- Per-chip DRAM: **8 banks × ~4 GB ≈ 32 GB/chip** (`tech_reports/memory/allocator.md:21`:
  "Blackhole devices have 8 ~4 GB DRAM banks"; telemetry `ENABLED_GDDR=0xff` = 8
  channels). **Measured usable DRAM/chip = 31.87 GiB** (2026-06-24, `ttnn.get_memory_view`)
  — the allocator hands out ~all of the 32 GB/chip (only ~0.13 GiB reserved), consistent
  with 128 GB / 4 chips. (NB: an earlier "~4 GB/chip" estimate misread *per-bank* as
  *per-chip*; the "28-30 GB" estimate is superseded by this 31.87 measurement.)

## 26B-A4B weights (verified dims: hidden 2816, 30 layers, 16/8 heads, head_dim 256,
intermediate 2112, MoE 128 experts top-8, moe_intermediate 704, vocab 262144,
`attention_k_eq_v=True`)

| Component | params | bf16 total | bf8 total | sharding (TP=4) | bf8 / chip |
|---|---|---|---|---|---|
| MoE experts (128 × (gate+up+down) × 30 layers) | ~22.8 B | ~45.6 GB | ~22.8 GB | **sharded** by intermediate (see below) | **~5.7 GB** |
| Dense/shared MLP + attn + norms (30 layers) | ~2.5 B | ~5 GB | ~2.5 GB | TP-sharded | ~0.6 GB |
| Embedding (tied, 262144×2816) | 0.74 B | 1.5 GB | 0.74 GB | replicated/sharded `MEASURE` | ~0.2-0.74 GB |
| **Total (~26 B)** | ~26 B | **~52 GB** | **~26 GB** | | **~6.5-7 GB** |

Expert per-layer footprint: `128 × (2·704·2816 + 704·2816) = ~761 M` params/layer
≈ **764 MB/layer at bf8**, ×30 = **~22.8 GB** total.

### The load-bearing question: are experts SHARDED or REPLICATED across the mesh?

This decides whether 26B-A4B fits QB2, and the code and the gemma4 test **disagree**:

- **Code path (`create_tt_model`)** builds `MeshConfig((1,4), tp=4)` and the expert
  loader (`tt/experts/weights.py:103-136`) then uses
  `mesh_config.column_parallel` / `row_parallel`, which are
  `ShardTensor2dMesh(..., dims=tensor_dim)` (`demos/gemma4/config.py:77-87`) — i.e.
  **experts are TP-sharded** along the intermediate dim → **~5.7 GB/chip**, fits
  comfortably in ~28-30 GB.
- **`test_full_model` guard (`tests/unit/test_model.py:343-344`)** `pytest.skip`s
  any MoE model when `TP<8`, with the comment *"MoE experts are replicated:
  ~764 MB/layer at bf8"* → that reading implies **~22.8 GB/chip** (replicated),
  which would be marginal/OOM at TP=4.

These cannot both be true for the TP=4 path. The replicate branch
(`ReplicateTensorToMesh`) only fires when `tp==1` or `mesh_config is None`
(`weights.py:107-108`), which is **not** the `create_tt_model` path. So the static
evidence favors **sharded → fits**, and the `tp<8` skip looks **conservative/stale
for the sharded path**. **This must be confirmed by measuring per-chip DRAM on
QB2** (see "Device validation" below) — do not promise a number that static
inspection contradicts.

## KV cache @ 256K context (batch 1, replicated per chip)

Layer geometry differs by type (verified `config.json`):
- **Sliding layers (25)**: kv_heads 8, head_dim 256, **bounded** to `sliding_window=1024`.
  KV/layer = `2(k,v) · 8 · 1024 · 256 · 2 B` ≈ **8.4 MB** → ×25 ≈ **210 MB**.
- **Full-attention layers (5)**: `num_global_key_value_heads=2`, `global_head_dim=512`,
  `attention_k_eq_v=True` (V tied to K). **Unbounded** @256K =
  `2 · 2 · 262144 · 512 · 2 B` ≈ **1.07 GB/layer** → ×5 ≈ **5.4 GB** (or ~half with
  K=V reuse `MEASURE`). The demo claims the full-attn KV is **paged/right-sized**
  (`text_demo_v2.py:219-227,251`: page_block_size=64, page_max_num_blocks=4096), so
  the realized footprint is the page-pool size, **not** the naive 5.4 GB —
  `MEASURE` the bounded/paged bytes at 256K.

KV total batch-1 @256K ≈ **0.21 GB (sliding) + paged full-attn (`MEASURE`)**.

## Fit & batch ceiling

- **Backbone fits at batch 1** if experts are sharded: ~6.5-7 GB weights/chip +
  ~0.2 GB sliding KV + paged full-attn KV (`MEASURE`) + prefill scratch/mask
  (`MEASURE`) ≪ ~28-30 GB. Comfortable headroom.
- **Batch ceiling @256K** = `(usable_DRAM − weights/chip − scratch) / per-request KV`.
  Per-request KV scales ~linearly with batch (sliding bounded; full-attn paged). The
  two unknowns (`usable_DRAM`, paged full-attn KV/request) are **device-measured**,
  so the ceiling is computed *after* the measurement sweep — the demo already probes
  batch 1/8/32 (`text_demo_v2.py:329`). **Do not state a numeric ceiling pre-measurement.**
- **If experts turn out replicated** (the test's reading): 26B-A4B does **not** fit
  QB2 at batch 1, and the fix is **Expert Parallelism** — shard the 128 experts
  across the 4 chips (the HF config ships `base_model_ep_plan`,
  `configuration_diffusion_gemma.py:68-77`; gemma4 `tt/` does not wire EP). That is
  the net-new HW-enablement work if the sharded-MLP TP path is insufficient.

## Device validation recipe (run on this box)

```bash
source /home/zni/venvs/tt-diffusion-gemma/bin/activate
export PYTHONPATH=/home/zni/tt-metal TT_METAL_HOME=/home/zni/tt-metal

# 0. Mesh plumbing smoke (no checkpoint): confirms P150x4 + TP=4 + fabric on QB2.
MESH_DEVICE=P150x4 pytest models/demos/gemma4/tests/unit/test_model.py::test_single_layer_model -k "1x4" -v

# 1. 26B-A4B per-component PCC on QB2 (these have blackhole-1x4 thresholds already):
MESH_DEVICE=P150x4 HF_MODEL=<26B path> \
  pytest models/demos/gemma4/tests/unit/test_model.py -k "1x4 and (layer_forward or attention_prefill)" -v

# 2. 12B full causal-backbone-logits PCC on QB2 (dense, no MoE skip) — #47461 stage-1 on QB2:
MESH_DEVICE=P150x4 HF_MODEL=<12B path> \
  pytest models/demos/gemma4/tests/unit/test_model.py::test_full_model -k "1x4" -v

# 3. 26B-A4B full model on QB2: requires bypassing the tp<8 MoE skip; the empirical
#    per-chip DRAM here is the answer to the sharded-vs-replicated question.
```

> ⚠️ This box has a known **erisc teardown re-hang** (active-eth core 29-25) after a
> device run (`plan.md`, Part II). Minimize CreateDevice churn (`use_module_device` / one
> session) and `tt-smi -r` between runs. Local board fw is 19.9.0 (ahead of
> tt-metal's tested 19.5.0); treat the hang as an env quirk.

## Status — measured on QB2 (2026-06-22) ✅ EXPERTS ARE SHARDED → 26B-A4B FITS

The sharded-vs-replicated question is **resolved empirically**:

- **26B-A4B full causal backbone RAN on QB2 (`P150x4`, TP=4) — no OOM, 110 s.** With
  the `test_full_model` `tp<8` MoE skip bypassed (`test_full_model[blackhole-1x4]`),
  the model built (52 GB across 4 chips), prefilled, and produced logits. So experts
  are **TP-sharded** via `MeshConfig.column_parallel`/`row_parallel` (≈5.7 GB/chip),
  **not replicated** — the conservative `tp<8` skip is overly cautious for this fit.
- **Backbone logits PCC vs HF = 0.8665** (prompt "The capital of France is"),
  **above** the established Blackhole `test_full_model[blackhole-1x8]` baseline (0.83).
  The `test_full_model[blackhole-1x4]=0.83` threshold I added passes once `HF_MODEL`
  has basename `gemma-4-26B-A4B-it` (use the symlink `/home/zni/dg_models/gemma-4-26B-A4B-it`
  → snapshot; the bare HF-cache snapshot basename is a hash, so the lookup otherwise
  falls back to the 0.99 default).
- **12B dense backbone on QB2 = 0.9595** (sanity: dense path on this exact HW).
- Board stayed **healthy across R1 + 12B + 26B** (no erisc-29-25 teardown hang this
  session, after the rebuild/reset). Run env: plain bashrc (`PYTHONPATH=$TT_METAL_HOME`,
  `TT_METAL_RUNTIME_ROOT` unset) + venv — the source `_ttnn.so` loads via the venv
  `ttnn-custom.pth`, self-consistent with its kernels via `TT_METAL_HOME`.

Turnkey reproduce (basename-correct path so the 0.83 threshold applies):
```bash
source /home/zni/venvs/tt-diffusion-gemma/bin/activate
export PYTHONPATH=/home/zni/tt-metal TT_METAL_HOME=/home/zni/tt-metal MESH_DEVICE=P150x4
HF_MODEL=/home/zni/dg_models/gemma-4-26B-A4B-it \
  pytest models/demos/gemma4/tests/unit/test_model.py::test_full_model -k "1x4"
# NOTE: requires the test's `if is_moe and tp<8: pytest.skip(...)` guard removed/relaxed
# for QB2 (the skip is conservative — 26B-A4B demonstrably fits at TP=4).
```

### Still pending (not blockers for the QB2 fit conclusion)
- Paged full-attn KV footprint + batch ceiling **at 256K** — the 26B run above was a
  short prompt — **now measured 2026-06-24 (see the definitive section below)**.
- Upstream: relax/remove the `test_full_model` `tp<8` MoE skip for the QB2-fitting case
  (currently conservative), so the QB2 full-model PCC runs without a manual bypass.

## Measured on QB2 — 2026-06-24 (definitive, via `ttnn.get_memory_view`)

Per-chip DRAM measured directly with `ttnn.get_memory_view(mesh_device, BufferType.DRAM)`
(`tests/test_qb2_memory_budget.py`, env-parameterized, one build/process). QB2
`P150x4`, 1×4, TP=4, **bf16 weights**. KV is allocated eagerly at build, so the
weights+KV budget is captured **without a prefill**.

| Quantity | Measured (GiB/chip) | Notes |
|---|---:|---|
| **Usable DRAM** | **31.87** | 8 banks × 3.984; allocator hands out ~all of 32 (only ~0.13 reserved). Resolves the earlier ~28–30 estimate. |
| **Weights (bf16, TP=4-sharded)** | **13.25** | = 52 GB / 4. **Corrects** the bf8-based "~6.5–7 GB/chip" above — the real run loads bf16. |
| **+ paged KV @256K, batch 1** | **17.25** total (Δ **4.0** KV) | 54% of usable; **headroom 14.6**. Resolves the "paged full-attn KV `MEASURE`". |
| **+ paged KV @256K, batch 2** | **19.80** total | ⇒ **+2.55 GiB/chip per extra batch** at 256K. |

**Weights+KV batch ceiling @256K (static)** = `(31.87 − 13.25 − 4.0)/2.55 + 1 ≈ batch 6`.
I.e. the weights + a 256K paged KV cache leave comfortable headroom — the static
fit is **not** the limiter.

### The real limiter is the prefill-activation regime, not weights/KV
The generator forces a **single prefill chunk** (`tt/generator.py:76–85` — Gemma4
ignores the Generator's `chunk_start_idx`, so it rounds `max_prefill_chunk_size`
up to a power of 2 ≥ the prompt and runs one chunk; the code itself warns "very
long contexts (>~64k) can OOM"). A single chunk materializes the full
`[1, L, hidden=2816]` activation (+ per-layer QKV/MoE transients) ∝ L, on top of
the 17.25 GiB weights+KV. So at **batch 1** the practical context ceiling is set
by this single-chunk activation fitting in the ~14.6 GiB headroom, **not** by the
KV cache (which fits to 256K with room to spare).

- **gemma4's own demo validates the fit**: `text_demo_v2.py` notes "single-chunk
  prefill runs at 128k without OOM" — i.e. 128k long-context prefill is known to fit
  in DRAM on this path (consistent with the 14.6 GiB headroom above). 256K-token
  *single-chunk* prefill is the regime the generator's own warning flags as the OOM
  risk; bounded-memory long prefill is the `chunk_start_idx` follow-up.
- **Operational gotchas measured while probing the long-context demo** (not budget
  unknowns — config/infra):
  - A raw single `ttnn_prefill_forward` of L > `sliding_window` is **not** a valid
    probe: it writes the whole sequence into the bounded-sliding KV (1024-token pool)
    and trips `update_cache_device_operation.cpp:106`. Long-context prefill must go
    through the generator's chunked + bounded-sliding path (`operations.py:210–353`).
  - The demo's `trace_region_size=200_000_000` (`text_demo_v2.py:143`) is **too small**
    for long-context prefill traces — a 64k trace needs ~445 MB (`TT_FATAL` at
    `mesh_trace.cpp:78`). Bump `trace_region_size` (it is carved from DRAM, so subtract
    it from the headroom) to run the traced long-context demo.
  - The box intermittently faults (`Bus error` core dump during prefill-trace warmup;
    board fw 19.9.0 ahead of tt-metal's tested 19.5.0 — see the erisc note above);
    `tt-smi -r` clears it.

### Bottom line for #47487 (QB2)
- **Documented budget (measured):** usable **31.87** − weights **13.25** − KV@256K **4.0** ⇒ **14.6 GiB/chip headroom** at 256K batch 1.
- **Current scope = batch=1.** Measured: **batch-1 256K weights+KV fit with margin** — 17.25 GiB/chip used, **14.6 GiB/chip headroom**. **End-to-end 256K prefill *execution* is gated by the single-chunk activation regime / `chunk_start_idx` follow-up** (a gemma4 upstream item: real `chunk_start_idx` for bounded-memory long prefill), **not by weights or KV**.
- **Batch > 1 — forward-looking only (→ #47557), NOT the current scope:** batch-2 256K measured at 19.80 GiB ⇒ ~+2.55 GiB/chip per extra batch ⇒ a *static* KV-bound ceiling ≈6 if batched later (a weights+KV bound, not a validated e2e-generation ceiling).
- **Diffusion-path additions (size these when #47462/#47463 land — NOT in the static budget above):** the denoise forward returns logits for **all 256 canvas positions every step**, so it must **disable gemma4's §2.8 last-tile LM-head slice** (which keeps only the last token). The full-canvas logits `[256, vocab]` are kept on device for sampling: ≈**34 MiB/chip** column-parallel (`vocab/4`) or ≈**137 MiB/chip** if all-gathered to full vocab, **plus** an equal-size softmax/probs buffer for the entropy + Gumbel-max step — recomputed every denoise step. The per-step canvas K/V scratch (#47474 storage class ii) is statically estimated by `memory_budget.py` at **~15 MiB/chip** on QB2 TP=4 bf16 batch=1 (25 sliding layers = ~12.5 MiB, 5 full-attn layers = ~2.5 MiB; current TT path materializes separate K and V tensors even for K=V-tied full layers). Small vs the 14.6 GiB headroom, but count it together with full-canvas logits/probs and the non-causal mask (#47462) against the headroom when sizing the **diffusion-path** batch ceiling (the static weights+KV ceiling above is the AR-backbone bound, not the denoise bound).
