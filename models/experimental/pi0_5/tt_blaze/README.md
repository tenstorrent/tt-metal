# pi0.5 on tt-blaze — scaffold

**Status**: design + stub-level scaffolding only. No kernels. No silicon
runs. Aimed at a contributor who needs to know what a tt-blaze port of
pi0.5 would look like, where each piece lives, and what's left to fill
in.

**Companion docs (read first)**:
- `../docs/PI0_5_GALAXY_DEPLOYMENT_PLAN.md` §3 (4-stage deployment).
- `../docs/L1_PLACEMENT_FINDINGS.md` (why this port is even being
  considered — Option C hits a static-CB clash with L1-resident weights
  on chip shapes that tt-blaze's per-stage FusedOp layout would
  side-step).
- `../docs/OPTION_B_L1_ASSESSMENT.md` (the lever tt-blaze gets for
  free: per-stage matmul shape co-designed with L1 weight placement).
- `/home/tt-admin/sdawle/tt-blaze/tt-blaze/docs/{blaze_101,
  fused_layer_contract}.md`, `pipeline_{manager,builder}/*.md`,
  `blaze/ops/dense_layer/op.py` (the canonical FusedOp reference).

---

## 1. What this scaffold is

Five files:

| File | What's in it |
|---|---|
| `README.md` | this doc |
| `layers.py` | Python stubs of the four FusedOps pi0.5 decomposes into |
| `pipeline.py` | `Pi0_5BlazePipeline` driver — sets up sockets, drives denoise loop |
| `mapping_notes.md` | the non-obvious tt-blaze decisions made here |
| `probe_design.md` | what an L1 footprint probe for the tt-blaze pipeline would measure |

The Python imports import `ttnn` and tt-blaze symbols (`FusedOp`,
`Input`, `Output`, `ReceiverSocket`, etc.) but none of the bodies will
actually run today — every `compose()` / `emit()` body is a stubbed
sketch with TODOs marking the kernel work.

The goal is **a contributor can read this in ~30 minutes and know what
class of work the port is.**

---

## 2. The pi0.5 → tt-blaze mapping

pi0.5 (openpi `pi05_libero` variant) decomposes into **four FusedOp
families**, each becoming a pipeline stage on its own loudbox:

```
host (orchestrator)
  │ H2D
  ▼
┌──────────────────────────────────┐
│ Stage 0 — Vision  (loudbox A)   │   SigLIP-27 + multimodal projector
│   FusedOp: SiglipEncoderLayer    │   replicated 27× across the loudbox
│           + ProjectorTail        │   1 token → 256 tokens (per cam, bs=2)
└──────────────────────────────────┘
  │ D2D
  ▼
┌──────────────────────────────────┐
│ Stage 1 — VLM Prefill A (loudbox B) │   VLM PaliGemma layers 0..8
│   FusedOp: VlmDecoderLayer       │   bf8 weights, KV cache stage-local
│           × 9                    │
└──────────────────────────────────┘
  │ D2D
  ▼
┌──────────────────────────────────┐
│ Stage 2 — VLM Prefill B (loudbox C) │   VLM PaliGemma layers 9..17
│   FusedOp: VlmDecoderLayer × 9   │
└──────────────────────────────────┘
  │ D2D (one-shot KV migration on prefill exit; per-step activation on denoise loop)
  ▼
┌──────────────────────────────────┐
│ Stage 3 — Expert Denoise (loudbox D) │   Action expert 18 layers
│   FusedOp: ExpertDecoderLayer × 18 │   + suffix MLP, joint attn with VLM KV
│           + SuffixMlp            │
└──────────────────────────────────┘
  │ D2H (10× per inference, 1 per Euler step)
  ▼
host (collect denoised actions)
```

Loudbox = 8 chips. 4 loudboxes per Galaxy = 32 chips. Matches the
canonical tt-blaze layout exactly: 1 layer-family per loudbox, TP=8
inside, persistent D2D sockets between loudboxes.

---

## 3. Side-by-side with Option B / Option C

| Concern | Option B (today, in-ttnn) | Option C (today, in-ttnn) | tt-blaze (this scaffold) |
|---|---|---|---|
| Placement unit | 4 stages × 8 chips, TP=8 within each | 28 chips heterogeneous (4 vision / 18 prefill / 6 denoise / 4 spare) | 4 loudboxes × 8 chips, TP=8 within each — **same as B** |
| Transport between stages | Host DRAM bounce (`option_b/transport.py`) | Custom `option_c/transport.py` (host-side or D2D where layouts allow) | Persistent `MeshSocket` D2D, created once at startup |
| KV ownership | Stage-local; expert reads cross-stage via `kv_migration.py` after prefill | Stage-local; expert reads via `kv_migration.py` (one-shot post-prefill) | Stage-local; tt-blaze contract says no migration — pi0.5 violates this, **so we add a one-shot D2D edge type** (see mapping_notes §4) |
| Weight residency | DRAM today; L1 path blocked by static-CB clash on the SigLIP/VLM MLP shapes | DRAM today; same clash blocks `layer_paired_l1` and `vision_weights_l1` modes | **L1-resident by construction** — FusedOp compiler lays CBs + weights together, no allocator race |
| Looping | Per-token Python dispatch via `Pi0_5PipelineB.run_inference()` | Per-token Python dispatch via `Pi0_5PipelineC.run_inference()` | `LoopingConfig` — kernels stay resident, host injects one token per tick via PipelineManager writer thread |
| Denoise loop | Host loop over `forward_expert_step` × 10 (~6-10 ms incl. host roundtrip) | Same | Stage 3 loops internally; orchestrator just feeds initial noise + collects after 10 ticks |
| Today's e2e perf | ~220 ms total | ~209 ms total | **target: 8.9 ms** (deployment plan §6.1, Option C analytical) — pulled in by removing per-token host dispatch + DRAM bounce |

---

## 4. What's in this scaffold vs what's left

**Built (in this scaffold):**
- The four FusedOp class skeletons with declared `Input`/`Output`
  sockets, shape/dtype comments matching the dtype map (`SiglipEncoderLayer`,
  `VlmDecoderLayer`, `ExpertDecoderLayer`, `SuffixMlp`).
- A `Pi0_5BlazePipeline` driver that defines the four pipeline stages
  + their socket wiring + the denoise loop scheduling.
- Mapping notes documenting the non-obvious choices (KV migration
  edge, denoise loop count asymmetry, SigLIP bs=2, vision layer
  packing).
- Probe design analogous to the Option B/C L1 footprint probe but
  targeting the tt-blaze tensor placement model.

**Left to fill in:**
- Every `compose()` / `emit()` body. Each is currently a stub plus
  `# TODO(blaze):` comments pointing to the sub-ops that need to be
  invoked (`MLA`, `RmsNorm`, `Mcast`, `DenseSwiglu`, etc.).
- The C++ kernel bodies for any sub-op that doesn't already exist in
  tt-blaze. SigLIP attention (head_dim=72, padded to 96) and the
  expert adaRMS modulation are likely the two that need new
  micro-ops; everything else should reuse existing ones from
  `blaze/ops/`.
- The weight-loading helpers (`get_<stage>_tensors` / `get_<stage>_args`
  pattern in fused_layer_contract.md §5). pi0.5 weights live in HF
  safetensors today — needs an HF→tt-blaze WeightProvider.
- Numerical validation: PCC tests for each FusedOp vs the existing
  torch reference at `models/experimental/pi0_5/reference/`.
- The asymmetric denoise loop config: PipelineManager assumes each
  stage runs once per token. pi0.5's denoise loop runs Stage 3
  ten-times while Stages 0–2 idle. The asymmetric `LoopingConfig` is
  the open question called out in `PI0_5_GALAXY_DEPLOYMENT_PLAN.md`
  §3.5.

---

## 5. How a contributor extends this

The contributor workflow expected by the tt-blaze contract
(`fused_layer_contract.md` §11):

1. **Pick a FusedOp to flesh out.** Start with `SiglipEncoderLayer` —
   smallest weight footprint, runs once per inference (no denoise
   loop interaction), exercises the BS-attention quirk (bs=2 base +
   wrist cams).
2. **Author the `emit()` body.** Compose existing micro-ops from
   `blaze/ops/` (`ReceiverSocket → RmsNorm → Mcast → Matmul → ...→
   SenderSocket → CrossDeviceSignal`). Anything that needs a new
   sub-op gets a new directory under `blaze/ops/<name>/` with a
   `kernels/op.hpp` and a thin Python wrapper.
3. **Write the backed test.** Mirror
   `tt-blaze/tests/blaze/backed/test_dense_layer_backed.py` —
   loopback mode (catches kernel-lifecycle bugs) and host-socket
   mode (catches hangs under persistent looping).
4. **Wire it into `Pi0_5BlazePipeline.build()`** in this directory's
   `pipeline.py`. The driver already knows the per-stage shape and
   which submesh to target.
5. **Run the L1 probe** described in `probe_design.md` to verify the
   per-chip footprint fits the loudbox budget.

The recipe above is intentionally **the same workflow as DeepSeek V3
or Llama 3.1** in tt-blaze. pi0.5 just adds a vision-encoder family
upstream of an LLM-shaped decoder family, plus the denoise loop
quirk.

---

## 6. Open questions for the next session

Listed in detail in `mapping_notes.md` §6 and `probe_design.md` §5.
The two big ones:

1. **Asymmetric per-stage looping.** PipelineManager's writer thread
   injects one token per tick — it doesn't know that Stage 3 needs to
   loop 10× per "token" while Stages 0–2 are idle. Either the
   PipelineManager needs an asymmetric LoopingConfig, or Stage 3's
   FusedOp needs an internal loop count and we treat the whole
   denoise pass as one "tick" from the orchestrator's perspective.
   Default in this scaffold is the latter — see `pipeline.py::DENOISE_STEPS`.

2. **KV migration edge type.** tt-blaze's PipelineGraph today
   represents per-token activation edges. pi0.5 needs a one-shot
   prefill→denoise KV edge. The scaffold sketches this as a separate
   D2D socket built outside the FusedOp's main socket path (see
   `pipeline.py::_setup_kv_migration_sockets`). Needs confirmation
   from the tt-blaze team that this is supported.
