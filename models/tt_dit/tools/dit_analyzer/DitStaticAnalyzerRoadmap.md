# Roadmap: from hand-modelled graphs to real pipeline capture

Companion to [`DitStaticAnalyzerPlan.md`](DitStaticAnalyzerPlan.md), which covers
phases 0–6 of the original design. Phases 1–4 of that plan are prototyped in
this directory and phase 5 was exercised against LTX-2.3 — but only against a
graph **hand-written from the source** (`examples/ltx.py`). This document
inventories everything standing between that and pointing the checker at a real
pipeline, then lays out the work to close it.

## Where we are

| | |
|---|---|
| Analysis | works offline, pure Python: forward availability + backward demand + 6 redundancy rules + proofs |
| Op semantics | 18 specs, 80 call-name aliases |
| Capture | 30 hooks written, **never run on hardware** |
| Graph source | hand-written via `builder.py`, or a trace + hand-declared entry placements |
| Validated on | SD3.5-large block (0 findings), LTX-2.3 block ×2 topologies (6 provable duplicates on Ring, 0 on Linear) |

## Target end state

Two commands, no hand-written model, no `builder.py`:

```bash
# on device: run the real pipeline under capture
ditcheck capture --pipeline models.tt_dit.pipelines.ltx --preset bh_4x8 \
    --frames 121 --height 704 --width 1216 --out ltx_bh4x8.graph.json

# offline: analyze it
ditcheck analyze ltx_bh4x8.graph.json --top 10
```

Everything the analyzer needs — mesh shape, axis roles, entry placements, logical
vs padded shapes, per-device shard offsets, loop structure, call counts — is
derived from the run. The engineer supplies nothing except the pipeline
invocation they already know how to write.

## The one thing that stays manual: op registration

Op semantics cannot be inferred from a call. That is by design (plan §"Key
engineering choices", choice 3), and this roadmap keeps it as the single manual
surface — but changes its ergonomics from *silent degradation* to *explicit,
guided registration*:

1. **Capture records every `ttnn.*` call**, registered or not. Unregistered ops
   become `unregistered` IR nodes with real inputs, outputs, shapes and source
   locations — never invisible edges.
2. **Analysis refuses to guess.** A finding whose proof depends on an
   `unregistered` node is withheld, not downgraded. The report ends with the list
   of ops to register and what each would unlock.
3. **`ditcheck ops --missing <graph>`** prints, per unregistered op: call name,
   arity, observed shapes, call sites, occurrence count, and a copy-paste
   `register(OpSpec(...))` stub with the transfer functions left as `TODO`.
4. **`ditcheck ops --check`** fails in CI when a captured graph contains an op no
   spec covers, so coverage debt is visible rather than accumulating as
   suppressed findings.

Target: registering a straightforward op (pointwise / view-like / axis-mapping)
is a ≤10-line entry; only genuinely new *communication* semantics should need
real thought.

## Blocker inventory

35 items, from the audit of `models/tt_dit` (169 distinct `ttnn.*` call sites).
`P` = phase that closes it.

### A. Capture mechanics

| # | Blocker | P |
|---|---|---|
| 1 | Recorder never run on hardware; `ttnn` doesn't import on a dev laptop | 6 |
| 2 | 78 tensor-math ops unhooked (`mesh_partition`×29, `pad`×20, `copy`×10, `repeat`×9, `lerp`×8, conv2d/3d, `group_norm`, `upsample`, `minimal_matmul_split`, `dit_minimal_matmul_addcmul_fused`, `dit_rms_norm_unary_fused`, `neighbor_pad_async`, `slice_reshard_async`, `send_async`/`recv_async`, `exp_ring_joint_sdpa`, `rotary_embedding_hf`, …) | 8 |
| 3 | `ttnn` trace capture: 9 `begin_trace_capture` sites; `execute_trace` replays with no Python ops | 6 |
| 4 | `Tracer` returns the same tensor objects every call and overwrites post-capture allocations → `id()` identity aliases distinct values | 6 |
| 5 | `ttnn.deallocate` (92 sites) frees buffers whose `id()` can be recycled | 6 |
| 6 | In-place ops (`ttnn.copy`, `multiply_`, `add_`, `StateTensor.update`) break the SSA assumption | 6 |
| 7 | Mesh mappers/composers unhooked (`ShardTensor2dMesh`, `ConcatMesh2dToTensor`, `create_mesh_composer`) — where distribution and readback sets are decided | 7 |

### B. Information a trace doesn't carry

| # | Blocker | P |
|---|---|---|
| 8 | Entry placements must be declared: `.shape` is per-device, nothing says which mesh axis fractures which tensor axis | 7 |
| 9 | Axis roles (sp/tp/cfg) live in `DiTParallelConfig`, not on tensors | 7 |
| 10 | `padded_shape` vs logical shape: matmuls use tile-padded shapes, region math uses logical | 7 |
| 11 | Uneven shards (38→40 heads for tp=4); `RegionSet.shard`'s ceil split may not match ttnn's mapping | 7 |
| 12 | Model-specific weight preprocessing (`_interleave_heads`, `prepare_for_fused_swiglu`, `permute_for_swiglu`) changes logical column order; per-device QKV layout is hardcoded | 7 |
| 13 | Block-float byte accounting: `bfp8_b`/`bfp4_b` exponent overhead missing from `elem_bytes` | 7 |

### C. Semantics coverage

| # | Blocker | P |
|---|---|---|
| 14 | Conv/VAE family missing (conv2d/3d, group norm, upsample, `neighbor_pad_async` halo, `slice_reshard_async`) — the whole VAE stage falls through to `unknown` | 8 |
| 15 | `mesh_partition` unmodelled although it *changes distribution* → silent dist corruption downstream | 8 |
| 16 | Point-to-point comm (`send_async`/`recv_async`) has no participant-group semantics | 8 |
| 17 | Opaque reshapes (VAE's `b,h,w,c → 1,1,h*w,c`) degrade regions to full + taint | 8 |
| 18 | Fused-kernel internal comm is a hand-maintained registry; a new fused kernel silently hides its collective | 8 |
| 19 | Non-DiT parallel configs (`EncoderParallelConfig`, `VaeHWParallelConfig`, `MochiVAEParallelConfig`, `AudioTParallelConfig`) don't map onto the single 2-axis `Dist` | 10 |

### D. Graph structure and scale

| # | Blocker | P |
|---|---|---|
| 20 | Loop rollup: 48 layers capture as 48 copies; `calls` is hand-set today | 9 |
| 21 | Quadratic analysis cost: 1 block (160 nodes) = 0.59 s / 9 MB → a naive 48-layer capture extrapolates to ~20 min / ~20 GB | 9 |
| 22 | Multi-submesh pipelines (118 submesh references; CFG/VAE on separate `MeshDevice`s) vs one `Graph.mesh` | 10 |
| 23 | Multi-host meshes: `distributed_context_get_rank`, socket pairs, fabric config — device IDs span hosts | 10 |
| 24 | Cross-forward state (latents, `StateTensor`, KV caches); `steps` is a hand-supplied multiplier | 10 |
| 25 | Host-side gaps (scheduler/guidance between forwards) split the graph into disconnected pieces | 10 |
| 26 | Report doesn't scale: 22 TP gathers × 48 layers needs finding rollup | 9 |
| 27 | Counter-based node IDs are unstable across captures, so CI can't diff findings | 9 |

### E. Soundness gaps (wrong answers, not missing ones)

| # | Blocker | P |
|---|---|---|
| 28 | Semaphores/barriers unmodelled: a "redundant" collective may exist for synchronisation (`vae_all_gather`'s barrier comment) | 11 |
| 29 | Persistent ping-pong buffers: "reuse the earlier result" is invalid if an intervening collective overwrote that slot | 11 |
| 30 | No memory model: removing a gather changes L1/DRAM residency; a recommendation may not fit | 11 |
| 31 | One capture = one branch (`skip_qk`, `has_gate`, `kv_replicated`, `use_ring_cross`, stage 1/2, LoRA mode, `image_conditioning`, `dynamic_load`) | 12 |
| 32 | Shape-dependent graphs: `video_N`/`audio_N` derive from resolution/duration/fps | 12 |
| 33 | Cost model can't rank at pipeline scale: no link contention, no comm/compute overlap, no per-op fixed cost | 13 |

### F. Workflow

| # | Blocker | P |
|---|---|---|
| 34 | No capture entry point: pipelines have no `--capture-graph` flag or env hook | 13 |
| 35 | No golden baselines for a CI gate | 13 |

## Phases

Estimates are for one engineer, and assume device access for phases 6, 7 and 12.

### Phase 6 — Capture you can trust (2–3 weeks) · closes 1, 3, 4, 5, 6

The foundation: if tensor identity is wrong, every finding is wrong. Nothing
downstream is worth building first.

- Replace `id()`-keyed identity with a **versioned buffer key**: `(buffer
  address, generation)`. Hook `deallocate`, `copy`, `multiply_`, `add_`,
  `StateTensor.update` to bump the generation, which restores SSA by
  construction — an in-place write becomes a new symbol with the mutated value.
- Hook `begin_trace_capture` / `end_trace_capture` / `execute_trace`: record the
  capture pass as a subgraph, treat each `execute_trace` as an invocation of it
  (feeding `calls`), and assert the captured region is entered exactly once.
- Detect and reject unsafe captures loudly: a tensor observed as an input with no
  producer *after* the graph has started is a hard error naming the call site,
  not a silent entry symbol.
- **Acceptance:** on a 2-device mesh, capture `LTXAttention.forward` and one
  `TransformerBlock.forward` and diff the recorded collectives (op, dim, axis,
  shapes, source lines) against the hand-written `examples/` graphs. Where they
  disagree, the capture is right and the example gets fixed.

### Phase 7 — Layout inference (2–3 weeks) · closes 7, 8, 9, 10, 11, 12, 13

Removes the hand-declared `placements=` argument.

- Hook mesh mappers/composers and `mesh_partition` to learn each entry tensor's
  distribution and each readback's device set at the moment it is created.
- Read the pipeline's `DiTParallelConfig` / `EncoderParallelConfig` /
  `VaeHWParallelConfig` for axis roles and factors; record them in `Graph.meta`
  and use them for axis names.
- Record `shape` **and** `padded_shape` per tensor: regions and demand on logical
  extents, byte/cost math on padded.
- Take per-device shard offsets from the mapper rather than assuming an even ceil
  split; represent uneven shards exactly.
- Make the fused-weight layout a declared property of a linear layer
  (`per_device_qkv`, `swiglu_interleave`, …) instead of a hardcoded default.
- Extend the dtype table with block-float exponent overhead.
- **Acceptance:** `trace_to_graph` produces zero entries in
  `meta['assumptions']` for a real LTX capture, and the inferred layouts match
  `examples/ltx.py` where they overlap.

### Phase 8 — Op coverage and the registration workflow (3–4 weeks) · closes 2, 14, 15, 16, 17, 18

- **Auto-hook** instead of an explicit list: wrap the `ttnn`, `ttnn.experimental`
  and `ttnn.transformer` namespaces so every tensor-returning call is recorded;
  keep `HOOKS` only for the ops needing argument extraction.
- Build the `unregistered` node kind and the withhold-don't-guess rule, plus
  `ditcheck ops --missing` (stub generator) and `--check` (CI).
- Declare fused-kernel internal stages as **data**: a table mapping fused op →
  (comm stage, compute stage, axis/dim argument names), so a new fused kernel is
  a table entry rather than a code change and a silent miss.
- Ship Tier-2 specs for the ops that actually appear in DiT forwards:
  `mesh_partition`, `pad`, `copy`, `repeat`, `lerp`, `embedding`,
  `minimal_matmul_split`, `dit_minimal_matmul_addcmul_fused`,
  `dit_rms_norm_unary_fused`, `exp_ring_joint_sdpa`, `rotary_embedding_hf`.
- Conv/VAE family: conv2d/3d, `group_norm`, `upsample`, `neighbor_pad_async`
  (halo exchange — a genuinely new communication shape), `slice_reshard_async`,
  `send_async`/`recv_async`; plus a real spec for rank-changing reshapes so the
  VAE stops tainting.
- **Acceptance:** an LTX and a VAE capture each analyze with zero `unregistered`
  nodes and zero `UNKNOWN_OP` diagnostics.

### Phase 9 — Scale (2 weeks) · closes 20, 21, 26, 27

- Deterministic node IDs from `(op, source location, occurrence index within
  block)`; findings become diffable across captures.
- Block detection by hashing node subsequences (op + loc + shape signature);
  collapse repeats to one instance with `calls`, keeping the first and last
  instances intact so boundary effects aren't hidden.
- Replace per-node full state snapshots with liveness-pruned, copy-on-write
  state; snapshot only at collectives.
- Roll up findings by `(rule, source location)` in the report.
- **Acceptance:** a full 48-layer LTX capture analyzes in <60 s and <2 GB, and
  its findings match the rolled-up single-block result.

### Phase 10 — Multi-mesh, multi-stage, multi-host (2–3 weeks) · closes 19, 22, 23, 24, 25

- `Graph` holds a **set** of meshes; symbols carry a mesh id; collectives
  reference `(mesh, axis)`. Submesh and CFG-parallel pipelines then model
  directly.
- Generalise `Dist` beyond two axes, and map the encoder/VAE parallel configs
  onto it (H/W parallelism is just more axes).
- Stage boundaries (encoder → DiT → VAE) as separate linked graphs; host
  transfers become explicit edges.
- Mark carried state (latents, KV cache, `StateTensor`) as `carried` inputs and
  outputs, so `steps` is derived from the run instead of supplied.
- **Acceptance:** one capture of a full `pipeline_ltx` generation yields a linked
  graph set covering encoder, transformer and VAE, with `steps` inferred.

### Phase 11 — Soundness gates before acting (2 weeks) · closes 28, 29, 30

Until this lands, findings are "look here", not "do this".

- Record persistent-buffer identity and semaphore IDs on every collective;
  suppress a `duplicate_gather` CSE recommendation when the earlier result's
  buffer is reused before the candidate site (and say so).
- Model barrier/sync intent: flag collectives whose removal changes
  synchronisation, and demote them to "needs review" with the reason.
- Track live bytes per device so a recommendation can state whether keeping the
  earlier result actually fits, instead of assuming it does.
- **Acceptance:** each of the 6 LTX Ring findings carries an explicit
  buffer-liveness verdict and a memory-feasibility note.

### Phase 12 — Branch and shape coverage (1–2 weeks) · closes 31, 32

- Capture manifests: record the branch flags and shapes a graph corresponds to
  (topology, `has_audio`, `has_gate`, stage, LoRA mode, resolution, frames).
- `ditcheck matrix <manifest-set>` runs the configured combinations, merges
  findings, and — importantly — **reports which branches were never captured**,
  so unexercised paths are visible rather than assumed clean.
- **Acceptance:** an LTX matrix run covers both topologies × both stages × audio
  on/off and lists the uncovered branches by name.

### Phase 13 — Workflow integration (1–2 weeks) · closes 33, 34, 35

- `--capture-graph` / `TT_DIT_CAPTURE_GRAPH` in the pipeline entry points, via
  one shared helper.
- Golden findings baselines per config; a CI job running `analyze --fail-on
  provable` plus `ops --check` against them.
- Optional time-based ranking: join per-op measured latency from a perf CSV so
  findings can be ordered by microseconds rather than bytes. Explicitly a
  ranking aid, not a latency model — contention and overlap stay out of scope.
- **Acceptance:** a new provable redundancy introduced in a PR fails CI with the
  source line in the failure output.

## Ordering

```
6 (identity/capture)  ──►  7 (layouts)  ──►  8 (op coverage)  ──►  9 (scale)
                                                   │
                                                   ├──►  10 (multi-mesh/stage)
                                                   ├──►  11 (soundness gates)
                                                   └──►  12 (coverage matrix) ──► 13 (CI)
```

6 → 7 → 8 → 9 is the critical path to "runs on a real pipeline end to end"
(~10–12 weeks). 10–13 can proceed in parallel afterwards; 11 gates *acting* on
findings, so it should not be deferred far behind 9.

## Risks

| Risk | Mitigation |
|---|---|
| Buffer-address identity is still ambiguous under aggressive reuse | Cross-check with shape+dtype+producer; on ambiguity emit a hard error, never a guess |
| Auto-hooking every `ttnn` call slows the traced run or perturbs behaviour | Record metadata only (no tensor reads), and gate capture behind the flag; compare a captured vs uncaptured run for equality |
| Op coverage becomes a treadmill as kernels are added | `ops --check` in CI makes debt visible; the fused-op table keeps new kernels to a data change |
| Inferred layouts are subtly wrong and turn into confident wrong findings | Phase 6/7 acceptance is a diff against hand-written graphs; keep `builder.py` examples as regression oracles |
| Findings scale faster than anyone will read them | Rollup by source location plus the existing top-N ranking; the report is a queue, not a dump |

## Out of scope

Automated rewrites (plan §choice 4: proofs before auto-fixes), training/backward
graphs, non-DiT models, and a predictive performance model. The deliverable
remains trustworthy diagnostics with proofs an engineer can check.
