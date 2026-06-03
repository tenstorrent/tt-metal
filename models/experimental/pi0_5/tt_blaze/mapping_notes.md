# tt-blaze ↔ pi0.5 mapping notes

Concrete notes on the non-obvious decisions made when mapping pi0.5
onto the tt-blaze deployment model. Each section answers a specific
"how did you decide X?" question.

---

## 1. Galaxy → loudbox → stage assignment

**Question**: How does a 32-chip Blackhole Galaxy carve into FusedOp
stages?

**Decision**: One loudbox = 8 chips = one tt-blaze pipeline stage.
4 loudboxes per Galaxy → 4 pipeline stages. TP=8 inside every
stage. Maps to:

| Loudbox | Stage | Content |
|---|---|---|
| A | 0 — Vision | SigLIP-27 + multimodal projector |
| B | 1 — VLM Prefill A | VLM PaliGemma layers 0..8 (9 layers) |
| C | 2 — VLM Prefill B | VLM PaliGemma layers 9..17 (9 layers) |
| D | 3 — Denoise | Expert layers 0..17 + Suffix MLP + 10-step Euler loop |

This is the canonical tt-blaze layout (blaze_101.md §4.4) with one
modification: stages 1 and 2 each host **9 layers** rather than one.
The canonical 1-layer-per-loudbox pattern would need 18 loudboxes
(2.25 Galaxies) for the VLM tower alone.

**Why 9 layers per loudbox is acceptable here**:
- The VLM weight budget per loudbox at TP=8 is ~124 MB/chip
  (`OPTION_B_L1_ASSESSMENT.md`). Fits the 175 MB / chip L1 cap.
- Each layer is its own FusedOp subclass instance but they share the
  loudbox; they chain via L1 CBs internally rather than via D2D sockets
  between layers. See `pipeline.py::_build_vlm_stage_a` for the
  multi-layer chained FusedOp pattern.
- DSv3 packs multiple sparse-MoE layers similarly in some pipelines
  (`blaze/models/deepseek_v3_b1/pipeline.py::create_single_pod_pipeline_configuration`
  groups MoE layers 3..12 across the available stages).

---

## 2. Why not 1 layer per stage like DSv3?

**Question**: tt-blaze's docs emphasize one fused layer per pipeline
stage. Why are we packing 9 VLM layers per stage?

**Trade-off**:
- DSv3 has 61 transformer layers + embed + LM head = 63 stages. Across
  16 Galaxies (504 chips), each loudbox hosts exactly one layer.
- pi0.5 has 27 vision + 18 VLM + 18 expert + 1 suffix = 64 layer-like
  things, but only a **single Galaxy** to spend. 64 stages on 4
  loudboxes means 16 layers per loudbox on average.
- We have two choices:
  - **(a)** Multi-galaxy deployment (4+ Galaxies). Stays canonical
    but multiplies hardware cost ~4×.
  - **(b)** Multi-layer FusedOps. Sacrifices canonical-ness but fits
    on one Galaxy. We choose this.

**Implication for the FusedOp authoring**: each multi-layer stage is
**one chained FusedOp** (like `blaze/ops/dense_layer/op.py` which
chains MLA + CbReconfig + MLPMixed inside one fused emit) — not
multiple FusedOps with sockets between them. The chained pattern uses
L1 CBs between layers, no fabric crossings. This is critical for
latency: D2D socket overhead per crossing is ~1 µs on BH ethernet;
27 vision layers × 1 µs = 27 µs that would dominate the 8.9 ms target.

---

## 3. SigLIP-27 → loudbox A mapping

**Question**: 27 SigLIP layers, 8 chips in the loudbox. How do they pack?

**Choice**: 27 SigLIP layers are CHAINED inside one big FusedOp on
loudbox A, all running at TP=8. Each layer's weights are replicated
across the 8 chips at TP=8 (with the appropriate sharding scheme for
attention head-dim and MLP intermediate-dim). The FusedOp emit walks
27 layer sub-emits in sequence, passing intermediate activations via
L1 CBs.

**Per-chip weight budget on loudbox A**:
- SigLIP layer at TP=8: ~2.4 MB / layer / chip (see layers.py table).
- 27 layers × 2.4 MB = **~65 MB / chip** for SigLIP encoders alone.
- Plus patch_conv (1.77 MB) + pos_embed (0.59 MB) + final LN + mm_proj
  (~5 MB) ≈ **~73 MB / chip**.
- Fits comfortably in the 175 MB L1 cap; leaves ~100 MB headroom for
  activations + kernel scratch + the SigLIP→VLM projection.

This is **strictly better than Option C's 7+7+7+6 packing** which hit
the per-bank L1 cap at the SigLIP MLP shape. TP=8 here shrinks each
chip's matmul shape 8×; the corresponding kernel CB region shrinks
~8× too, so per-bank weights + CBs fit even with all 27 layers
co-resident.

**Alternative considered**: pack 4 SigLIP layers per chip across 7 of
the 8 chips (4×7 = 28, drop one or reduce to ~3.4 layers each). That's
the Option C approach. Rejected here because (a) it keeps DRAM
weights for the rest, defeating the L1 placement plan, and (b) it
introduces multiple D2D crossings inside SigLIP, which is wasted host
latency relative to the chained FusedOp pattern.

---

## 4. KV cache ownership and migration

**Question**: tt-blaze assumes per-stage KV ownership and no migration.
pi0.5 needs the expert layers to read VLM K/V. How?

**Solution**: A one-shot D2D edge type, distinct from the per-token
activation edges.

In standard tt-blaze, each pipeline stage:
- Owns its KV cache (allocated at stage init).
- Writes its KV during prefill (one tick per token).
- Reads its KV during decode (every tick).
- Does NOT share KV with other stages.

pi0.5 violates this: the action expert's joint attention at layer L
reads K and V values that were computed at VLM layer L during prefill.
Two viable implementations:

**(a) One-shot D2D migration edge (recommended in the deployment plan §3.3.a):**
- 18 persistent D2D MeshSockets pre-allocated from stages 1/2 → stage 3.
- After VLM prefill completes (one prefill "tick" through all 18 layers),
  the sockets fire in parallel, copying the 18 K/V tensors from VLM
  stages to denoise stage's `vlm_k_prefix` / `vlm_v_prefix` slots.
- Total payload: 18 × ~500 KB = 9 MB across the mesh, in parallel —
  ~1-2 ms wall clock at BH ethernet (100 GB/s × 0.5 util).
- During the 10 denoise steps, no further migration. The prefix is
  stable; only the expert's own 50-token suffix K/V updates.

**(b) Broadcast VLM KV to all denoise chips:**
- Skip the layer-pairing logic. After prefill, every denoise chip
  receives the full 4.5 MB VLM KV.
- Simpler routing; ~4.5 MB extra L1 per denoise chip (well inside the
  budget).
- Deployment plan §3.3.b notes this is also viable.

**This scaffold chose (a)** because it's the more general pattern —
once tt-blaze has a "one-shot edge" primitive, it's reusable for
other models (encoder-decoder transformers, retrieval-augmented
decoders, etc.).

**Caveat**: tt-blaze's PipelineGraph today only represents per-tick
activation edges. The one-shot KV edge needs either:
- A new edge kind (`KvMigrationEdge`) that the PipelineManager knows
  to fire once per inference rather than per tick, OR
- An "out-of-band" D2D socket built outside the FusedOp's main socket
  path, fired by the orchestrator. The scaffold's
  `pipeline.py::_setup_kv_migration_sockets` sketches the latter.

This is an **open question** for the tt-blaze team.

---

## 5. Denoise loop — asymmetric per-stage iterations

**Question**: PipelineManager injects one token per tick. pi0.5's
denoise loop runs Stage 3 ten times per inference while Stages 0–2
stay idle. How is that modeled?

**Two approaches**:

**(a) Host-driven loop, one "tick" per Euler step.**
- The orchestrator pushes x_step and time(step) into Stage 3 via
  H2D socket once per Euler step (10× per inference).
- Stage 3's FusedOp runs once per push (single iteration).
- Stages 0–2 do nothing during these 10 ticks.
- Pros: matches PipelineManager's existing one-tick-per-token model.
- Cons: 10 host roundtrips per inference. At ~10 µs / roundtrip ≈
  100 µs of host overhead per inference — non-trivial against the
  8.9 ms target.

**(b) Device-driven loop, one tick per inference.**
- Stage 3's FusedOp internally loops 10 times via
  `LoopingConfig(num_iterations=10)`.
- Host pushes x_0 + initial time once; reads x_10 once.
- Pros: latency-optimal, no host-side loop.
- Cons: requires the FusedOp to maintain its own iteration counter
  and to gate the D2H output socket on the final iteration.

This scaffold defaults to **(b)**, with **(a)** as a fallback path
spelled out in `pipeline.py::run_inference`. The FusedOp side of (b)
requires authoring an internal-loop wrapper around the expert tower
+ suffix MLP — sketched in `layers.py::ExpertDecoderLayer` and
`SuffixMlp`.

The key tt-blaze question this raises: **does
`LoopingConfig(num_iterations=10)` work compositionally inside a
PipelineGraph where other stages have `num_iterations=1`?** The
fused_layer_contract.md §9 documents the looping kernel lifecycle but
assumes uniform iteration counts across stages.

If non-uniform looping isn't supported, fall back to (a) and pay the
host overhead.

---

## 6. SigLIP bs=2 — base and wrist cameras

**Question**: pi0.5 processes two camera images (base + wrist) per
inference. How does that interact with TP=8 inside loudbox A?

**Background**: pi0.5's SigLIP runs with batch dim = 2 (base + wrist
cams stacked). The bs=2 has to thread through the BS-attention reshape
quirk currently implemented in
`models/experimental/pi0_5/tt/ttnn_siglip.py` (see commit notes from
2026-05-18 — `QWEN_NLP_CONCAT_HEADS_HEAD_SPLIT` and similar env
flags).

**Mapping decision**: keep bs=2 as the leading dim and reshape
to (bs*seq, hidden) for the matmuls (the existing BS-attention
pattern). The TP=8 head-parallel split is on the head dim, orthogonal
to bs — both cams' tokens go through both chips' attention shards.

**Implication for FusedOp**: the SiglipEncoderLayer.emit body needs to
preserve the (1, 2, 256, 1152) → (1, 512, 1152) reshape that the
ttnn_siglip.py code does. The padding for tile alignment also needs to
match. The dtype map already handles this; the new FusedOp just has
to call the same reshape sub-op (`blaze/ops/retilize` or `gather` —
needs verification).

---

## 7. Embedding table — host or chip?

**Question**: PaliGemma's `embed_tokens` table is [257152, 2048] bf8 ≈
527 MB. Where does it live?

**Decision**: **on host.** Same as Option C's approach
(`PI0_5_GALAXY_DEPLOYMENT_PLAN.md` §3.1 (a)).

Rationale:
- 527 MB sharded across 8 chips of loudbox A = 66 MB / chip. Adds to
  the ~73 MB already on loudbox A for SigLIP + projector → 139 MB /
  chip total. Tight but fits.
- BUT: the lookup itself is trivially small (200 tokens × 2 fetches),
  and pre-fetching on host saves 527 MB of L1 weight pressure on the
  vision chips, where weight-fit is already the gating constraint.
- The host pre-embeds lang_tokens to [200, 2048] bf8 ≈ 410 KB and
  pushes that over H2D alongside the pixel values.

**Implication for FusedOp**: there's no `EmbeddingLayer` FusedOp in
this scaffold. The host handles it.

---

## 8. adaRMS modulation precomputation

**Question**: The expert's adaRMS modulation weights `[3072, 1024]`
(6.3 MB per layer × 18 = 113 MB) would blow the denoise chip budget.
How is this handled?

**Decision**: precompute the modulation outputs `(scale+1, shift,
gate)` per (layer, denoise_step) at startup and cache them in DRAM as
a single tensor `[18 layers, 10 steps, 3072 dim]` bf16 = 1.1 MB total.

Rationale: the adaRMS modulation depends only on the integer denoise
step (and the per-layer modulation weights), not on the activation.
For 10 fixed denoise steps, all 18 × 10 = 180 modulation values are
known at startup.

**Implication for the FusedOp**: ExpertDecoderLayer reads its
modulation values via a `Gather` op against `metadata.position_id`
(which carries the denoise step) — see `layers.py::ExpertDecoderLayer`
step 2 in the emit body. Saves ~120 MB / chip of weight storage; only
costs a tiny DRAM read per layer per step.

This matches what the existing TT-NN model already does
(`ttnn_pi0_5_model.py` / `ttnn_suffix.py`).

---

## 9. Open / deferred questions

Carried forward from the deployment plan §3.5 and the open issues
docs. Calling out the ones specifically relevant to the tt-blaze port:

1. **Non-uniform LoopingConfig.** (See §5 above.) Does tt-blaze
   support 10-iter Stage 3 inside an otherwise 1-iter pipeline? If
   not: fall back to host-driven Euler loop.

2. **One-shot KV migration edge.** (See §4 above.) Is there an idiom
   in tt-blaze for a D2D socket that fires once per inference rather
   than once per token? Or should we manage it manually outside the
   PipelineGraph?

3. **Multi-layer chained FusedOp at depth.** Loudbox A chains 27 SigLIP
   layers in one FusedOp. The DSv3 dense_layer chains 3 internal ops
   (MLA → CbReconfig → MLPMixed). Does the codegen scale to 27×
   chained sub-emits without blowing kernel binary size? Needs a
   probe.

4. **head_dim=72 in SigLIP attention.** Tile-pads to 96 on tt-metal.
   Does tt-blaze's SDPA / GQA family accept padded head dims natively,
   or do we need a custom `siglip_sdpa` micro-op?

5. **bs=2 retilize.** The pi0.5-specific batch-flatten reshape pattern
   needs an equivalent tt-blaze sub-op. Reusing `blaze/ops/retilize`
   may suffice; otherwise author a `bs_reshape` op.

6. **D2H gating on the final denoise step only.** When (b) device-loop
   is chosen, the D2H sender must emit only on iteration 9 of 10.
   `blaze/ops/d2h_sender_socket` doesn't have a conditional-emit
   pattern in any existing op. Needs either a new conditional D2H or
   a host-side filter that discards the first 9 emits.

7. **Vision text-embed bypass.** The host pre-embeds language tokens
   (§7 above). Need to verify the H2D socket on loudbox A is happy
   accepting the embedded `[200, 2048]` bf8 tensor as part of its
   payload alongside the pixel values, rather than the raw token ids.
   Should just be a different page size, but worth flagging.
