# pi0.5 on a Blackhole Galaxy — Layer Map, Memory Budget, and Deployment Options

Working notes for porting pi0.5 (openpi upstream variant, checkpoint `pi05_libero_upstream`)
to a 32-chip Blackhole Galaxy with SRAM-resident weights. Combines the current pi0.5
TT-NN dtype map (`models/experimental/pi0_5/tt/`) with the pipeline-parallel patterns
exposed by **tt-blaze** (`/home/tt-admin/sdawle/tt-blaze/tt-blaze`).

The doc has three parts:
1. The pi0.5 model footprint we are trying to place (per-stage, per-layer, per-dtype).
2. A short architectural summary of tt-blaze that frames the options.
3. Three concrete deployment options + a detailed evaluation of the user's proposed
   4-stage pipeline.

---

## Part 1 — pi0.5 model footprint

Source checkpoint: `/home/tt-admin/pi05_cache/pi05_libero_upstream/model.safetensors`
(3.617 B params, 7.23 GB raw bf16).

Dtype map referenced below is the current TT-NN one — see
`models/experimental/pi0_5/tt/{ttnn_siglip,ttnn_paligemma,ttnn_gemma,ttnn_suffix,ttnn_common}.py`
and the project memory entry "pi0_5 dtype map". `bf8_b ≈ 1.0625 B/param` (block float w/
shared exponent per 16-elem block); rounded to `1 B/param` for budgeting.

### 1.1 SigLIP-27 vision tower
Width 1152, MLP 4304, 16 heads × 72 head_dim, 27 layers, 256 patches per image, bs=2
(base + wrist cams).

| Tensor / encoder layer  | Shape          | params      | dtype  | bytes      |
|-------------------------|----------------|-------------|--------|------------|
| attn q/k/v/o weight × 4 | [1152,1152]×4  | 5.31 M      | bf16   | 10.6 MB    |
| attn q/k/v/o bias × 4   | [1152]×4       | 5 K         | bf16   | 9 KB       |
| mlp.fc1 weight          | [4304,1152]    | 4.96 M      | bf8_b  | 4.96 MB    |
| mlp.fc1 bias            | [4304]         | 4 K         | bf16   | 8.6 KB     |
| mlp.fc2 weight          | [1152,4304]    | 4.96 M      | bf8_b  | 4.96 MB    |
| mlp.fc2 bias            | [1152]         | 1 K         | bf16   | 2.3 KB     |
| 2× LayerNorm γ,β        | [1152]×4       | 5 K         | bf16   | 9 KB       |
| **per encoder layer**   | —              | **15.2 M**  | mixed  | **~20.7 MB** |
| × 27 layers             | —              | 410 M       | —      | **~558 MB** |
| patch conv W+b          | [1152,3,16,16] | 884 K       | bf16   | 1.77 MB    |
| pos_embed               | [256,1152]     | 295 K       | bf16   | 0.59 MB    |
| final LN + mm_proj      | [1152]+[1152,2048] | 2.36 M  | bf16   | 4.7 MB     |
| **SigLIP total**        | —              | **414.8 M** | —      | **~565 MB** |

If attention weights are also dropped to bf8_b (currently bf16 on the assumption that
QKVO bandwidth doesn't dominate at 1152-wide), per-layer drops to ~15.6 MB, total
**~420 MB**. Worth measuring before committing to a sharding plan.

### 1.2 VLM PaliGemma (Gemma-2B), 18 layers
Width 2048, MLP 16384, 16 Q-heads / 2 KV-heads × 128 head_dim, prefill ~544–968 tokens.

| Tensor / VLM layer | Shape         | params  | dtype  | bytes      |
|--------------------|---------------|---------|--------|------------|
| attn.q.weight      | [2048,2048]   | 4.19 M  | bf8_b  | 4.19 MB    |
| attn.k.weight      | [256,2048]    | 0.52 M  | bf8_b  | 0.52 MB    |
| attn.v.weight      | [256,2048]    | 0.52 M  | bf8_b  | 0.52 MB    |
| attn.o.weight      | [2048,2048]   | 4.19 M  | bf8_b  | 4.19 MB    |
| mlp.gate.weight    | [16384,2048]  | 33.55 M | bf8_b  | 33.55 MB   |
| mlp.up.weight      | [16384,2048]  | 33.55 M | bf8_b  | 33.55 MB   |
| mlp.down.weight    | [2048,16384]  | 33.55 M | bf8_b  | 33.55 MB   |
| 4× RMSNorm γ       | [2048]×4      | 8 K     | bf16   | 16 KB      |
| **per VLM layer**  | —             | **110 M** | bf8_b | **~110 MB** |
| × 18 layers        | —             | 1.98 B  | —      | **~1.98 GB** |
| embed_tokens (= lm_head, tied) | [257152,2048] | 526.6 M | bf8_b | ~527 MB |
| final RMSNorm γ    | [2048]        | 2 K     | bf16   | 4 KB       |
| **VLM total**      | —             | **2.51 B** | —    | **~2.51 GB** |

KV cache (activations, bf8 if the dtype map flips it that way for ≥544-token prefill):
2 KV heads × 128 dim × 968 tokens × 1 B × 18 layers × 2 (K+V) ≈ **8.9 MB total**, or
**~4.5 MB if stored as a single packed K∥V tensor at bf8_b** (matches the number
quoted in the proposed plan).

### 1.3 Action expert (Gemma-300M with adaRMSNorm), 18 layers
Width 1024, MLP 4096, **same 16 Q-heads × 128 head_dim**, 2 KV-heads, ~50 suffix tokens.

| Tensor / expert layer | Shape          | params | dtype  | bytes   |
|-----------------------|----------------|--------|--------|---------|
| attn.q.weight         | [2048,1024]    | 2.10 M | **bf16** | 4.19 MB |
| attn.k.weight         | [256,1024]     | 0.26 M | **bf16** | 0.52 MB |
| attn.v.weight         | [256,1024]     | 0.26 M | **bf16** | 0.52 MB |
| attn.o.weight         | [1024,2048]    | 2.10 M | bf8_b  | 2.10 MB |
| mlp.gate.weight       | [4096,1024]    | 4.19 M | bf8_b  | 4.19 MB |
| mlp.up.weight         | [4096,1024]    | 4.19 M | bf8_b  | 4.19 MB |
| mlp.down.weight       | [1024,4096]    | 4.19 M | bf8_b  | 4.19 MB |
| 4× RMSNorm γ          | [1024]×4       | 4 K    | bf16   | 8 KB    |
| pre_attn_adaln W+b    | [3072,1024]+[3072] | 3.15 M | bf16* | 6.30 MB* |
| pre_ffw_adaln  W+b    | [3072,1024]+[3072] | 3.15 M | bf16* | 6.30 MB* |
| **per expert layer (weights as-is)** | — | **23.6 M** | mixed | **~32.8 MB** |
| × 18 layers                          | — | 425 M | — | **~590 MB** |

`*` adaLN Dense weights can be **precomputed once per inference** because their input
(`adarms_cond = MLP(sincos(t))`) only depends on the integer denoise step. Cache the
modulation outputs `(scale+1, shift, gate)` per (layer, step) in DRAM as bf16: 18 layers
× 10 steps × 3072 × 2 B = **1.1 MB total**. That eliminates **~120 MB** of on-chip weight
storage. The TT-NN model already takes this approach (see `ttnn_pi0_5_model.py` /
`ttnn_suffix.py`).

| Expert weights ex-adaLN Dense | — | ~302 M | mixed | **~108 MB** |
| Expert lm_head [257152,1024]  | — | 263 M  | bf16  | unused — exclude |

### 1.4 Suffix MLP

| Tensor                   | Shape       | params | dtype | bytes   |
|--------------------------|-------------|--------|-------|---------|
| action_in_proj W+b       | [1024,32]   | 33 K   | bf8_b | 33 KB   |
| time_mlp_in W+b          | [1024,1024] | 1.05 M | bf8_b | 1.05 MB |
| time_mlp_out W+b         | [1024,1024] | 1.05 M | bf8_b | 1.05 MB |
| action_out_proj W+b      | [32,1024]   | 33 K   | bf8_b | 33 KB   |
| **Suffix total**         | —           | 2.2 M  | —     | **~2.5 MB** |

### 1.5 Totals to place on-mesh

| Stage                                  | Bytes        | % of total |
|----------------------------------------|--------------|-----------|
| SigLIP-27 encoder + embed + mm_proj    | 565 MB       | 17.0%     |
| VLM embed/lm_head (tied)               | 527 MB       | 15.9%     |
| VLM 18 transformer layers              | 1980 MB      | 59.6%     |
| Expert 18 layers (no adaLN Dense)      | 108 MB       | 3.3%      |
| Suffix MLP                             | 2.5 MB       | 0.1%      |
| Precomputed adaLN modulation (DRAM)    | 1.1 MB       | trace     |
| **Total weights on-mesh**              | **~3.18 GB** | **100%**  |

Per-chip SRAM budget on Blackhole: 1.5 MB × 120 cores = **180 MB**. Realistic budget
after activations / kernel binaries / scratch / KV: **~130 MB / chip**. Across 32 chips:
**~4.2 GB** available for weights → fits with ~1 GB headroom **if** weights are uniformly
sharded; placement irregularity is what makes this nontrivial.

---

## Part 2 — tt-blaze architecture summary

Source: `/home/tt-admin/sdawle/tt-blaze/tt-blaze/{README.md, docs/blaze_101.md,
docs/fused_layer_contract.md, pipeline_manager/architecture.md, pipeline_builder/design.md,
blaze/ops/dense_layer/op.py}`. Distilled below; cite those files for canonical detail.

### 2.1 Placement unit: one fused layer per stage
- The atomic unit is a **FusedOp**: a composition of MicroOps emitted as one kernel
  dispatch, intermediate buffers in L1 circular buffers — no DRAM transit between ops
  inside a fused layer.
- One FusedOp = **one transformer layer = one pipeline stage = one submesh**.
- A loudbox (4×2 = 8 chips) is the canonical submesh; one Galaxy (32 chips) = 4 loudboxes
  → **4 pipeline stages max per Galaxy** under the default contract.
- Within a stage, **tensor parallelism is baked in** (e.g., TP=8 for a dense layer on
  one loudbox) — TP is *not* a separate pipeline dimension.

### 2.2 Inter-stage transport: persistent D2D sockets + 64 B metadata trailer
- Each stage exposes `ReceiverSocket → fused compute → SenderSocket`.
- Activation payload + 64 B trailer share the same backing via
  `cb_from_tensor_overlapped()`; trailer carries `position_id`, `slot_id`, `token_id`,
  + 13 reserved uint32s for per-op metadata.
- Sockets are **persistent across the session** — created once at startup, no per-token
  setup. Stage-local KV is allocated at stage startup; **no D2D KV migration in steady
  state**.
- `CrossDeviceSignal` back-edges from sender → upstream receiver prevent looping
  deadlocks under persistent deployment (`fused_layer_contract.md` §2 item 7).

### 2.3 KV cache: per-stage, no migration
- Each stage owns its own KV slots (one per user) for the layers placed on that stage.
- Token flow is **value-passing**, not state-passing: token `t` enters stage `i`,
  stage `i` writes its own KV at `position[user]`, forwards activation to stage `i+1`.
- This is fine for vanilla decoder-only LLMs. **pi0.5 violates this** — the expert
  layers' joint attention reads VLM-layer K/V values produced during prefill. See §3.3
  below for what to do about it.

### 2.4 Host orchestration: PM ↔ IS split
- `InferenceServer` handles HTTP / tokenization / session state on ms timescale.
- `PipelineManager` is the hot path: a writer thread injecting one token per pipeline
  tick (priority: decode > prefill), and a reader thread reading 64 B `ResultDescriptor`
  packets and routing tokens back into staging FIFOs.
- The framework already provides low-latency H2D / D2H socket abstraction; we don't
  need to roll our own scheduler.

### 2.5 Looping kernel contract
- Every fused kernel must implement `init()` → `for iter: per_loop_iter_setup();
  operator()(); }` → `teardown()`. Per-iter setup re-publishes DRAM-shard pointers to
  L1. This is what lets one kernel run thousands of tokens without re-dispatch
  (`fused_layer_contract.md` §9; `kernel_codegen.py:30–54`).

### 2.6 Reference deployments
- DeepSeek V3 (61 layers + embed + LM head = 63 stages) → 16 Galaxies (504 chips).
- No reference for a vision encoder + dual-tower model yet.

---

## Part 3 — Deployment options for pi0.5 on one Galaxy

Three viable plans, ordered by how strongly they lean on tt-blaze idioms.

### Option A — Pure tensor-parallel TP=32 (no pipeline)
Single logical model replica, every chip holds 1/32 of every weight. All-reduce after
each Q/K/V/O and gate/up/down.

| Per-chip footprint | Bytes |
|--------------------|-------|
| VLM MLP (3 × 33.55 MB × 18) / 32 | 57 MB |
| VLM attn (9.4 MB × 18) / 32 | 5.3 MB |
| SigLIP (565 MB) / 32 | 17.7 MB |
| Expert (108 MB) / 32 | 3.4 MB |
| embed (527 MB) / 32 | 16.5 MB |
| Replicated (LN, biases, suffix) | ~2 MB |
| **Total weights / chip** | **~102 MB** ✓ |

**Pros**: every chip active every layer; simple to reason about; debuggable as a single
replica.
**Cons**: heavy collective traffic (an all-reduce per matmul row-parallel slice);
maps poorly onto tt-blaze (tt-blaze prefers 1 layer per stage, not 1 weight slice per
chip across 32); awkward divisibility for SigLIP intermediate (4304 → pad to 4608),
expert attention (16 Q-heads / 2 KV-heads on 32 chips means head-splitting + 16×
KV replication).

### Option B — tt-blaze canonical: 4 stages × 8 chips/stage (TP-within-stage)
Match tt-blaze's reference layout exactly: one fused layer per stage, 8-chip submesh
per stage, TP=8 inside each stage.

- Stage 0 (8 chips): SigLIP-27 + VLM embed + multi-modal projector
- Stage 1 (8 chips): VLM layers 0–8 (9 layers fused as one stage)
- Stage 2 (8 chips): VLM layers 9–17 (9 layers fused as one stage)
- Stage 3 (8 chips): Expert layers 0–17 + suffix MLP

Per-stage memory (with TP=8 internal):
- Stage 0: 565 MB / 8 = 71 MB/chip — fits.
- Stage 1: 9 × 110 MB / 8 = 124 MB/chip + KV cache slice — tight but fits.
- Stage 2: same as Stage 1 — tight.
- Stage 3: 108 MB / 8 = 13.5 MB/chip + VLM KV the expert reads — easy fit.

**Pros**: cleanest map onto tt-blaze (FusedOp per stage, persistent D2D sockets between
stages, no KV migration needed except expert ← VLM KV).
**Cons**: Stage 1 and Stage 2 each "fuse" 9 transformer layers into one FusedOp emit,
which is larger than the canonical 1-layer-per-stage pattern — codegen and kernel
binary size need to be checked. The denoise loop runs entirely on Stage 3, so Stages
0–2 are idle for the 10 Euler steps of denoise → ~75% of compute time wasted unless
you batch multiple inference requests through the pipeline.

### Option C (user's proposal) — 4-stage extreme pipeline (1 layer per chip)
- **Stage 0 — Orchestrator (host)**: tokenization, image preprocessing, denoise step
  scheduling. Talks to Stage 1 via H2D socket; receives final actions via D2H socket.
- **Stage 1 — Vision encoding (4 chips)**: 3 chips × 9 SigLIP layers each + 1 chip for
  image embedding / mm_projector.
- **Stage 2 — VLM Prefill (18 chips)**: 1 chip per VLM transformer layer.
- **Stage 3 — Denoise (6 chips)**: either 3 chips × 6 expert layers, run as 2 parallel
  copies (denoise-batch=2); or 6 chips × 3 expert layers, single copy.
- Free: 32 − (4+18+6) = **4 chips** unallocated (good — gives slack for embed sharding,
  KV migration buffers, or a duplicate denoise replica).

#### 3.1 Memory check (Option C)

**Vision encoding chips (3× 9-SigLIP-layer chips):**
- 9 layers × 20.7 MB = **186 MB** under current dtype map → **over budget**.
- 9 layers × 15.6 MB (if attn dropped to bf8_b too) = **140 MB** → fits with 40 MB
  headroom for activations.
- Activation [256, 1152] × 1 B = 295 KB; intermediate [256, 4304] = 1.1 MB. Easy.

**Vision embedding chip (1 chip):**
- patch_conv (1.77 MB) + pos_embed (0.59 MB) + final LN + mm_proj (2.4 MB at bf8) ≈
  **5 MB**. Plenty of headroom.
- ⚠️ **Open question**: does this chip also need to host VLM `embed_tokens` (527 MB)?
  That doesn't fit on one chip. Two viable answers:
  - (a) Host pre-looks-up text embeddings and sends them as activations over the H2D
    socket (200 tokens × 2048 × 1 B = 410 KB — trivial). **embed_tokens table lives
    on host, not on chip.** Recommended.
  - (b) Shard embed_tokens by vocab across the 4 vision chips (527/4 = 132 MB/chip) —
    blows the budget once SigLIP layers are also there. Reject.

**Prefill chips (18 chips × 1 VLM layer):**
- 110 MB weights + activation `[968, 2048]` × 1 B = 2 MB + KV write buffer for that
  layer (968 × 256 × 1 B = 250 KB) + scratch (~5 MB) ≈ **118 MB / chip**. Fits with
  ~60 MB headroom. ✓
- Note: the per-layer KV is per-layer, not aggregated — so the "[18, 968, 1, 256]
  ~ 4.5 MB" in the spec is the *global* KV footprint across all 18 prefill chips, not
  per-chip. Each prefill chip stores 250 KB of K + 250 KB of V locally.

**Denoise chips:**
- 6 chips × 3 expert layers: 3 × ~6 MB (weights ex-adaLN) = 18 MB + adaLN precomputed
  (3 × 30 KB ≈ 90 KB) + activation [50, 1024] × 1 B = 50 KB + VLM KV for 3 layers
  (3 × 250 KB × 2 = 1.5 MB) + scratch (5 MB) ≈ **25 MB / chip**. Trivially fits.
- 3 chips × 6 expert layers × 2 parallel pipelines: 6 × 6 MB = 36 MB + adaLN + 3 MB VLM
  KV + scratch ≈ **45 MB / chip**. Also fits.
- **Big headroom on denoise chips** → they could absorb additional copies, a
  speculative-decode-style branch, or batched denoise across multiple inference
  requests.

#### 3.2 Inter-stage transport (Option C, using tt-blaze sockets)

Boundary | Payload | Size | Frequency
---|---|---|---
HOST → Vision chip 0 | image tensors [2, 3, 224, 224] bf16 + lang_tokens [200] u32 | 600 KB | 1 / inference
Vision chip i → chip i+1 | hidden [2, 256, 1152] bf8 | 590 KB | 1 / inference
Vision embed chip → Prefill chip 0 | projected vision tokens [2, 256, 2048] bf8 + lang embed [200, 2048] bf8 + state [1, 2048] bf8 | ~1.4 MB | 1 / inference
Prefill chip i → chip i+1 | activation [968, 2048] bf8 | ~2 MB | 1 / inference
Prefill chip i → matching Denoise chip (KV migration) | K∥V for one layer, [2, 968, 256] bf8 | ~500 KB | **1 / inference** (one-shot)
Denoise chip i → chip i+1 | suffix activation [50, 1024] bf8 | 50 KB | **10× / inference** (1 per Euler step)
Denoise final → HOST | x_t [50, 32] bf16 | 3.2 KB | 1 / inference

Numbers check out — D2D sockets see ~2 MB peak per token boundary at the prefill
boundary and 50 KB during denoise. Well below the BH ethernet budget.

#### 3.3 The KV migration problem — Option C's only architectural risk

tt-blaze's design assumes each stage owns its KV in steady state. **pi0.5 breaks this**:
the expert blocks read VLM K/V during joint attention. Reference code: `forward_expert`
in `models/experimental/pi0_5/reference/torch_paligemma.py` and the parallel TTNN path
in `ttnn_paligemma.py` (`Pi0_5PaliGemmaBackboneTTNN`) — the expert's queries attend to
`concat(VLM_K_prefix, expert_K_suffix)` and same for V, at matching layer indices.

That means: after prefill, each Denoise chip needs the VLM K/V (per layer it owns)
from the corresponding Prefill chip(s). Two ways to do it cleanly:

**3.3.a One-shot D2D migration after prefill (recommended).**
- Layer-paired routing: Denoise chip `d` (which owns expert layers `L..L+2`) opens
  D2D sockets to Prefill chips `L..L+2` and receives one K∥V tensor per layer at
  prefill completion.
- Total bytes: 18 × 500 KB = 9 MB across the mesh, sent in parallel by 18 source
  chips to 6 destinations → ~1–2 ms wall clock at BH ethernet.
- Sockets are persistent; the migration is a single one-shot transfer per inference,
  reused across all 10 denoise steps.
- Equivalent to adding one extra "edge type" in the PipelineGraph: a *one-time
  prefill-to-denoise KV edge*, distinct from the per-token activation edges.

**3.3.b Replicate VLM KV onto all denoise chips (also viable).**
- After prefill, broadcast the full 4.5 MB VLM KV from prefill chips to every denoise
  chip. Per chip: extra 4.5 MB stored locally. Still well under budget.
- Simpler routing (no layer-pairing logic); marginal extra memory. Probably the
  least-risky choice if you can spare the bytes.

**3.3.c (Rejected) — co-locate VLM and expert layer-by-layer.**
- I.e., put VLM layer L and expert layer L on the same chip. Removes KV migration but
  blows the chip's weight budget: 110 MB (VLM) + 6 MB (expert) + adaLN + activations
  ≈ 130 MB just for weights, leaves ~50 MB for everything else. Borderline, and ties
  Stage 2 and Stage 3 together — defeats the clean pipeline structure.

#### 3.4 Latency / utilization considerations (Option C)

Key observation: **the denoise loop runs 10× per inference**, and only the 6 denoise
chips participate. While denoise runs, vision (4 chips) and prefill (18 chips) sit
idle — that's **22 / 32 ≈ 69% of the mesh idle for 10× the per-step latency**.

This is fine for **single-shot inference** (one image-action pair end-to-end) where
latency-per-action dominates. It is bad for **throughput** — you can't pipeline multiple
inference requests through a single Galaxy because the denoise stage becomes the
bottleneck (10× the per-token cost of any other stage).

Mitigations:
- The "2 parallel stages" variant of denoise (3 chips × 6 layers × 2 copies) lets you
  run 2 inference requests interleaved at the denoise stage → ~50% utilization gain.
- The 4 spare chips could host a third denoise replica or a smaller backup VLM
  prefill, smoothing throughput further.
- Or run prefill of inference request N+1 while denoise of inference request N is
  ongoing — classic 1-deep pipeline overlap. Doable if KV migration is async.

For LIBERO sim, where latency-per-action is what we report, **Option C is the right
choice**. For a hypothetical batched evaluation harness or multi-arm policy serving,
the answer changes.

#### 3.5 Where Option C diverges from tt-blaze defaults — items to confirm

| Aspect | tt-blaze default | Option C ask | Risk |
|---|---|---|---|
| Stage submesh size | 8 chips (4×2) | 1, 3, 4, or 18 chips per stage | Need to confirm `PipelineBuilder` supports non-4×2 submeshes (`pipeline_builder/design.md` §1 implies heterogeneous shapes are allowed) |
| Layers per stage | 1 | 9 (vision), 1 (prefill), 3 or 6 (denoise) | FusedOp `emit()` size for 9-layer stage — may need code-gen path or hand-fused kernel |
| KV migration | none (per-stage local) | one-shot edge prefill → denoise | Need a new edge type in PipelineGraph (or use a custom one-shot D2D socket outside the FusedOp loop) |
| Denoise loop | n/a (single-pass decode) | 10-step Euler integrator on Stage 3 only | Stage 3 needs its own internal looping config; Stages 0–2 idle during this window — confirm `LoopingConfig` allows asymmetric per-stage iteration counts |
| Multi-image batch | n/a | SigLIP runs bs=2 (base + wrist) | Existing pi0.5 BS-attention fix (2026-05-18) carries through; preserve the `n_batch,n_seq` reshape gymnastics |

---

## 4. Recommendation

**Adopt Option C as the deployment target, but in two phases:**

**Phase 1** — get it running on one Galaxy without the full tt-blaze stack.
- Use the existing `Pi0_5ModelTTNN` (`models/experimental/pi0_5/tt/ttnn_pi0_5_model.py`)
  with a custom multi-device `mesh_device` that maps to the 4-stage layout.
- Implement KV migration as a one-shot `ttnn.all_gather` or explicit
  `ttnn.from_device(...) → ttnn.to_device(...)` between the prefill device-group and
  the denoise device-group.
- Validate end-to-end LIBERO success rate matches single-device baseline within
  bf8-rounding noise.

**Phase 2** — port to tt-blaze proper.
- Encode each stage as a FusedOp under `blaze/ops/`.
- Convert KV migration to a one-time D2D edge in the PipelineGraph (precedent in tt-
  blaze sockets layer, just used asymmetrically).
- Use `PipelineManager` for the denoise-loop scheduling — its writer thread can drive
  10 ticks per inference request on Stage 3.
- Decode-loopback pattern in PM is a natural fit for the Euler integrator's
  "feed velocity back as next-step state" pattern.

**Open questions to resolve before starting Phase 1:**
1. Is the Vision Encoding "Embedding" chip the SigLIP→VLM mm_projector, the VLM
   `embed_tokens`, or both? (We assumed host-side text embed lookup; if not, plan
   changes.)
2. Should adaLN Dense weights stay on-chip (+120 MB) or be precomputed to DRAM
   activations (~1 MB)? Current TT-NN code already precomputes — keep that.
3. Does the proposed "2 parallel denoise stages" mean 2 batched inference requests
   interleaved, or 2 copies of the same denoise loop for redundancy / spec-decode?
   These have very different implementations.
4. Confirm `PipelineBuilder` supports 1-chip and 3-chip submeshes (not just 4×2). The
   design doc claims runtime topology discovery, but reference code may assume 4×2.

---

## 5. References

- pi0.5 TTNN dtype/compute-config map: project memory `pi0_5 dtype map` and the source
  files it cites.
- pi0.5 model layout: project memory `pi0_5 layout`.
- pi0.5 accuracy levers (esp. `PI0_DENOISE_FP32=1`): project memory `pi0_5 accuracy
  levers` — relevant if denoise stage Euler accumulator drifts on bf16.
- tt-blaze pipeline parallelism: `/home/tt-admin/sdawle/tt-blaze/tt-blaze/docs/blaze_101.md`
- tt-blaze fused layer contract (sockets, metadata trailer, looping kernel):
  `/home/tt-admin/sdawle/tt-blaze/tt-blaze/docs/fused_layer_contract.md`
- tt-blaze pipeline manager (writer/reader threads, decode loopback):
  `/home/tt-admin/sdawle/tt-blaze/tt-blaze/pipeline_manager/architecture.md`
- tt-blaze pipeline builder (mesh topology / submesh shapes):
  `/home/tt-admin/sdawle/tt-blaze/tt-blaze/pipeline_builder/design.md`
- Reference FusedOp emit pattern: `tt-blaze/blaze/ops/dense_layer/op.py`.

---

## 6. Analytical perf model results (date: 2026-06-02)

A first-order analytical perf model lives at `models/experimental/pi0_5/docs/perf_model.py`
(standalone, no tt-metal deps). It models four deployment options on a 32-chip
Blackhole Galaxy with `BH_BF8_TFLOPS = 1490`, `BH_UTIL_FRACTION = 0.5`,
`BH_ETH_GBPS = 100` (link bw), and `ETH_LINK_UTIL = 0.5`. Re-run with
`python perf_model.py` (or override `PI0_PREFILL_SEQ` for the 544-token finetune
variant). Full output is captured in `perf_model_output.txt`.

### 6.1 Comparison table (prefill_seq = 968 tokens)

| Metric | Option A (TP=32) | Option B (4x8) | Option C (heterogeneous) | Option C' (1x2 submeshes) |
|---|---|---|---|---|
| Chips used | 32 / 32 | 32 / 32 | 28 / 32 | **56 / 32 (over)** |
| Vision compute | 79.5 us | 317.8 us | 2.54 ms | 1.27 ms |
| Prefill compute | 160.9 us | 643.8 us | 5.15 ms | 2.58 ms |
| Denoise step | 1.1 us | 4.6 us | 36.7 us | 18.4 us |
| Denoise (x10) | 11.5 us | 45.9 us | 367.4 us | 183.7 us |
| All-reduce overhead | **6.57 ms** | 5.93 ms | 0.0 us | 3.39 ms |
| Inter-stage transfer | 0.0 us | 108.0 us | 833.4 us | 833.4 us |
| KV migration | 0.0 us | 45.0 us | 10.0 us | 10.0 us |
| End-to-end latency | **6.82 ms** | **7.09 ms** | 8.90 ms | 8.26 ms |
| Throughput (inf/s) (slowest-stage-bound) | 146.7 | 3107 | 1180 | 2360 |

Per-chip memory (peak) — L1 budget is 180 MB (modeled as 188.7 MB = 1.5 MB x 120 cores):

| Stage / option | Memory / chip |
|---|---|
| A — pooled TP=32 | 113.7 MB (fits) |
| B — stage 0 (SigLIP + embed, 8 chips) | **148.5 MB** |
| B — stage 1/2 (VLM 9-layer slab, 8 chips) | **136.3 MB** |
| B — stage 3 (expert + suffix, 8 chips) | 26.9 MB |
| C — vision (3 chips, 9 SigLIP layers each) | **156.9 MB** (tight; bf8 attn needed) |
| C — vision-embed (1 chip) | 16.0 MB |
| C — prefill (18 chips, 1 VLM layer each) | 122.5 MB |
| C — denoise (6 chips, 3 expert layers each) | 30.0 MB |
| C' — prefill (TP=2, 36 chips) | 67.2 MB |

### 6.2 Interpretation

**Option A (TP=32)** is the lowest-latency option but only by a hair (6.82 ms vs
7.09 ms for B). The dominant cost is **all-reduce overhead** (6.57 ms, ~96% of
e2e): every Q/K/V/O and gate/up/down emits an N=32 ring all-reduce, and the VLM
prefill all-reduce alone (output `[968, 2048] = 2 MB` bf8 across 32 chips ring)
costs roughly `2 * 31/32 * 2 MB / (100 GB/s * 0.5)` per matmul, summed over 18
layers x 2 outputs. Throughput-per-Galaxy is also worst (~147 inf/s) because
all 32 chips serve a single replica. Memory fits comfortably (113.7 MB / chip),
but every additional matmul makes the collective burden worse — this option
gets *less* efficient as the model scales.

**Option B (4 stages x 8 chips, TP=8 within stage)** is essentially tied with
A on latency but **20x better on throughput** (3107 inf/s) because pipelined
stages let multiple in-flight inferences overlap, gated only by the slowest
stage. The catch is memory: stage 0 (SigLIP-27 + 527 MB embed table) lands at
148.5 MB / chip and stages 1/2 (9 VLM layers each) at 136.3 MB / chip — within
the nominal 180 MB cap, but tight once you add real kernel binaries and dynamic
scratch. The 10-step denoise loop runs entirely on stage 3, idling stages 0-2
for ~46 us per inference; that "idle window" is short in absolute terms here
(denoise is only 0.6% of e2e), but it is the structural cost noted in §3.4.

**Option C (heterogeneous pipeline, 28 chips)** has the **longest single-shot
latency (8.90 ms)** because per-chip compute is no longer parallelized within a
layer — each VLM layer runs on one chip for 286 us, and 18 of those in serial
gives 5.15 ms of prefill. Vision (3 chips x 9 SigLIP layers) costs 2.54 ms.
But it uses only 28 of 32 chips, has **zero all-reduce overhead**, leaves 4
spares for redundant denoise replicas, and the **prefill chip memory (122.5 MB)
sits well inside the L1 budget**. Vision chips at 156.9 MB are the tight spot
and assume bf8 attn (the bf16-attn dtype map pushes them to ~186 MB / 9-layer
chip, blowing the budget — already flagged in §3.1). Throughput in pipelined
steady-state is 1180 inf/s, dominated by the 367 us denoise tail.

**Option C' (uniform 1x2 submeshes)** halves per-layer compute and weight
footprint via TP=2 in every submesh, dropping vision to 1.27 ms and prefill to
2.58 ms (vs. C). But it requires **56 chips total — 1.75 Galaxies** — and the
budget is exhausted partway through prefill: after vision (6) + vision-embed
(2) + prefill (36) = 44 chips, the 12 denoise chips put it at 56. Within one
Galaxy this is **infeasible**; on >=2 Galaxies it gives the best per-stage
utilization of any option. Not recommended for the current single-Galaxy LIBERO
target.

### 6.3 Verdict (analytical)

For single-shot LIBERO latency: **A and B are essentially tied (~7 ms)**, C is
~30% slower, C' is infeasible on one Galaxy. For sustained throughput
(batched policy serving): **B wins decisively** (~3 kHz steady-state), C is
second, A is worst. Option C remains the most natural map onto tt-blaze
(one fused layer per chip, persistent D2D sockets), at the cost of ~2 ms
single-shot latency. If memory pressure on B's stage-0/1/2 chips proves
prohibitive in practice, C is the obvious fallback.

### 6.4 Caveats

- **First-order only.** Single utilization fraction (50%) on peak FLOPS; real
  kernels (especially small matmuls in the expert path) can underperform that
  by 2-3x, and well-tuned dense matmuls can exceed it.
- **All-reduce model is ring-only.** Tree / recursive-doubling on the BH ethernet
  fabric could cut Option A's collective cost noticeably; the 6.57 ms figure is
  an upper bound for a 32-way ring at one link per chip. Multi-link rings would
  scale linearly with `BH_ETH_LINKS_PER_CHIP`.
- **Denoise tail is unmodeled past compute.** The 10 Euler steps assume the
  denoise stage can issue back-to-back with no per-step host orchestration. If
  the host injects each step (current Pi0_5 reference), add ~50-100 us per step.
- **Memory headroom estimates** include 2 MB peak activation (bf8 `[968, 2048]`),
  per-layer KV (~250 KB), and 10 MB realistic scratch. They do **not** include
  kernel binary footprint or DMA descriptors; budget another ~10-20 MB / chip
  for that in a real deployment.
- **All four options assume bf8 weights everywhere.** Mixing in bf16 (e.g. the
  current expert attn QKV in §1.3) inflates weight bytes proportionally — most
  immediately a problem for Option C's vision chips, which only fit if SigLIP
  attn is dropped to bf8.
- **Throughput numbers for B/C/C'** assume the pipeline is deep enough to
  hide stage-traversal latency. For 4-stage B and 28-stage C this is realistic
  after ~2-3 inferences are in flight; below that, the apparent throughput is
  closer to `1 / end-to-end_latency`.
