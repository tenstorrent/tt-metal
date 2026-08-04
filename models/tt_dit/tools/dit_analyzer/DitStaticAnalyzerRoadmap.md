# Roadmap: from hand-modelled graphs to real pipelines, without a device

Companion to [`DitStaticAnalyzerPlan.md`](DitStaticAnalyzerPlan.md), which covers
phases 0–6 of the original design. Phases 1–4 are prototyped in this directory
and phase 5 was exercised against LTX-2.3 — but only against a graph
**hand-written from the source** (`examples/ltx.py`). This document inventories
what stands between that and running the checker on a real pipeline, and lays out
the work to close it.

> **Revision 2 — adopts a dry-run front end.** The first version of this roadmap
> assumed on-device graph capture was the way in, and spent its first two phases
> fighting tensor identity under buffer reuse, trace replay and `deallocate`.
> This version makes a **metadata-only `ttnn` shim** the primary front end and
> demotes the device to a *validator*. Seven of the original 35 blockers dissolve
> outright; nine new shim-specific ones replace them. The critical path gets
> shorter and, more importantly, the everyday analysis stops needing hardware.
>
> **The design has been spiked** — see [`spike/FINDINGS.md`](spike/FINDINGS.md).
> The real LTX block runs under a metadata-only ttnn on a laptop and reproduces
> the phase-5 findings byte for byte, so what follows is calibrated against a
> working prototype rather than an estimate.

## Where we are

| | |
|---|---|
| Analysis | works offline, pure Python: forward availability + backward demand + 6 redundancy rules + proofs |
| Op semantics | 18 specs, 80 call-name aliases |
| Capture | 30 monkeypatch hooks written, **never run on hardware** |
| Dry run | spike proven: real `LTXTransformerBlock.forward` under a fake ttnn, 18 ttnn ops, 212 nodes, findings byte-identical to the oracle on both BH configs |
| Graph source | hand-written via `builder.py`, or the spike's dry run; a trace still needs hand-declared placements |
| Validated on | SD3.5-large block (0 findings), LTX-2.3 block ×2 topologies (6 provable duplicates on Ring, 0 on Linear), both reproduced from source by the dry run |

## The design: dry run first, device as validator

Run the real model code against a fake `ttnn` whose `Tensor` carries only
metadata — per-device shape, logical shape, dtype, layout, distribution — and
whose ops compute output metadata and append IR nodes. `LTXTransformerBlock.forward`
is ordinary Python; if `all_gather_minimal_matmul_async` returns a metadata
tensor instead of doing work, the forward pass runs on a laptop and emits the
graph directly. No hardware, no checkpoint download, no trace capture.

The device keeps exactly two jobs, neither of which is in the daily loop:

1. **Per-op conformance** — assert the shim's output shape / layout / distribution
   equals real `ttnn`'s, one op at a time. Cheap, and the only thing that keeps
   the shim honest as ttnn changes.
2. **One whole-block collective log** — a flat record of `(op, dim, cluster axis,
   per-device shapes, source line)` in call order, diffed against the dry run.
   Deliberately *not* a dataflow capture: a flat log needs no tensor identity, so
   none of the aliasing hazards below apply to it.

### Why not capture-primary

Capture observes ground truth, which is a real advantage — but it pays for it
with a permanent hardware dependency on every analysis, one branch and one
resolution per run, and three separate identity hazards (`Tracer` recycling
tensor objects, 92 `deallocate` sites, in-place ops) that must all be solved
before any finding can be trusted. The shim trades that for a *fidelity*
dependency, which is testable per-op in CI rather than per-model. Two
alternatives are non-starters: static AST analysis of the pipelines (too
config-dynamic to be sound), and the GSPMD-style approach of checking a declared
sharding plan against the torch reference — that answers "what collectives
*should* exist" and would not have caught the LTX finding, which lives in the
ttnn implementation, not the math. `TT_METAL_SIMULATOR` (`tt_metal/llrt/rtoptions.cpp`)
runs real kernels far too slowly for whole-pipeline graphs.

## Target end state

```bash
# on a laptop: no device, no weights, no capture
ditcheck dryrun --pipeline models.tt_dit.pipelines.ltx --preset bh_4x8 \
    --frames 121 --height 704 --width 1216 --out ltx_bh4x8.graph.json
ditcheck analyze ltx_bh4x8.graph.json --top 10

# on a device, in CI, occasionally: keep the shim honest
ditcheck conform --ops all_gather_async,minimal_matmul,...   # per-op shape/layout
ditcheck conform --block ltx --against ltx_bh4x8.graph.json  # flat collective log
```

## The one thing that stays manual: op registration

Op semantics cannot be inferred from a call. That is by design (plan §"Key
engineering choices", choice 3) and stays the single manual surface — but the
dry-run design **halves** it: a shim shape rule and an analyzer semantics spec
are the same knowledge, so they become one registration entry.

```python
register(OpSpec(
    name="all_gather", kind=COMM, is_collective=True, preserves_value=True,
    shim=lambda c: c.out(shape=c.in_shape(0), dist=c.in_dist(0).replicated(c.axis)),  # dry run
    apply=_all_gather_apply,    # forward availability
    demand=_all_gather_demand,  # backward necessity
))
```

Ergonomics, unchanged in intent from revision 1:

1. **Nothing is invisible.** An unregistered op still runs under the shim and
   still appears in the IR as an `unregistered` node with real inputs, outputs,
   shapes and source location — it just can't propagate metadata, so the dry run
   stops there with a precise error instead of silently drifting.
2. **Analysis withholds, never guesses.** A finding whose proof depends on an
   `unregistered` node is not emitted, not downgraded. The report lists what to
   register and what each registration unlocks.
3. **`ditcheck ops --missing`** prints, per unregistered op: call name, arity,
   observed shapes, call sites, count, and a copy-paste stub with `shim`,
   `apply` and `demand` left as `TODO`.
4. **`ditcheck ops --check`** fails CI when a graph contains an uncovered op, so
   coverage debt is visible rather than accumulating as suppressed findings.

Target: a pointwise / view-like / axis-mapping op is a ≤10-line entry. Only new
*communication* semantics should need real thought.

## Blocker inventory

The 35 items from the original audit of `models/tt_dit` (169 distinct `ttnn.*`
call sites), plus 9 introduced by the shim (44 was found by the spike). `P` = phase that closes it;
**dissolved** = the dry-run design removes it.

### A. Capture mechanics

| # | Blocker | P |
|---|---|---|
| 1 | Recorder never run on hardware; `ttnn` doesn't import on a dev laptop | 11 |
| 2 | 78 tensor-math ops unhooked (`mesh_partition`×29, `pad`×20, `copy`×10, `repeat`×9, `lerp`×8, conv2d/3d, `group_norm`, `upsample`, `minimal_matmul_split`, `dit_minimal_matmul_addcmul_fused`, `dit_rms_norm_unary_fused`, `neighbor_pad_async`, `slice_reshard_async`, `send_async`/`recv_async`, `exp_ring_joint_sdpa`, `rotary_embedding_hf`, …) | 8 |
| 3 | `ttnn` trace capture: 9 `begin_trace_capture` sites; `execute_trace` replays with no Python ops | **dissolved** — the shim no-ops trace capture; the forward runs in Python every time |
| 4 | `Tracer` returns the same tensor objects every call and overwrites post-capture allocations → `id()` identity aliases distinct values | **dissolved** — the shim mints fresh SSA symbols; no object reuse |
| 5 | `ttnn.deallocate` (92 sites) frees buffers whose `id()` can be recycled | **dissolved** — no buffers exist |
| 6 | In-place ops (`ttnn.copy`, `multiply_`, `add_`, `StateTensor.update`) break the SSA assumption | **dissolved** as a hazard — each becomes an explicit new symbol (spec in P8) |
| 7 | Mesh mappers/composers unhooked (`ShardTensor2dMesh`, `ConcatMesh2dToTensor`, `create_mesh_composer`) | **dissolved** — the shim *is* the mapper, so distribution is known at creation |

### B. Information a trace doesn't carry

| # | Blocker | P |
|---|---|---|
| 8 | Entry placements must be declared: `.shape` is per-device, nothing says which mesh axis fractures which tensor axis | **dissolved** — recorded when the shim creates the tensor |
| 9 | Axis roles (sp/tp/cfg) live in `DiTParallelConfig`, not on tensors | **dissolved** — the dry run reads the config object in-process |
| 10 | `padded_shape` vs logical shape: matmuls use tile-padded shapes, region math uses logical | 7 |
| 11 | Uneven shards (38→40 heads for tp=4); shard split must match ttnn's mapping exactly | 7 |
| 12 | Model-specific weight preprocessing (`_interleave_heads`, `prepare_for_fused_swiglu`, `permute_for_swiglu`) changes logical column order | 7 |
| 13 | Block-float byte accounting: `bfp8_b`/`bfp4_b` exponent overhead missing from `elem_bytes` | 7 |

### C. Semantics coverage

| # | Blocker | P |
|---|---|---|
| 14 | Conv/VAE family missing (conv2d/3d, group norm, upsample, `neighbor_pad_async` halo, `slice_reshard_async`) | 8 |
| 15 | `mesh_partition` unmodelled although it *changes distribution* | 8 |
| 16 | Point-to-point comm (`send_async`/`recv_async`) has no participant-group semantics | 8 |
| 17 | Opaque reshapes (VAE's `b,h,w,c → 1,1,h*w,c`) degrade regions to full + taint | 8 |
| 18 | Fused-kernel internal comm is a hand-maintained registry; a new fused kernel silently hides its collective | 8 |
| 19 | Non-DiT parallel configs (`EncoderParallelConfig`, `VaeHWParallelConfig`, `MochiVAEParallelConfig`, `AudioTParallelConfig`) don't map onto the single 2-axis `Dist` | 10 |

### D. Graph structure and scale

| # | Blocker | P |
|---|---|---|
| 20 | Loop rollup: 48 layers become 48 copies; `calls` is hand-set today | 9 |
| 21 | Quadratic analysis cost: 1 block (160 nodes) = 0.59 s / 9 MB → naive 48-layer graph extrapolates to ~20 min / ~20 GB | 9 |
| 22 | Multi-submesh pipelines (118 submesh references; CFG/VAE on separate `MeshDevice`s) vs one `Graph.mesh` | 10 |
| 23 | Multi-host meshes: `distributed_context_get_rank`, socket pairs, fabric config | 10 |
| 24 | Cross-forward state (latents, `StateTensor`, KV caches); `steps` is hand-supplied | 10 |
| 25 | Host-side gaps (scheduler/guidance between forwards) split the graph | 10 |
| 26 | Report doesn't scale: 22 TP gathers × 48 layers needs finding rollup | 9 |
| 27 | Counter-based node IDs are unstable across runs, so CI can't diff findings | 9 |

### E. Soundness gaps (wrong answers, not missing ones)

| # | Blocker | P |
|---|---|---|
| 28 | Semaphores/barriers unmodelled: a "redundant" collective may exist for synchronisation (`vae_all_gather`'s barrier comment) | 11 |
| 29 | Persistent ping-pong buffers: "reuse the earlier result" is invalid if an intervening collective overwrote that slot | 11 |
| 30 | No memory model: removing a gather changes L1/DRAM residency | 11 |
| 31 | One graph = one branch (`skip_qk`, `has_gate`, `kv_replicated`, `use_ring_cross`, stage 1/2, LoRA mode, `image_conditioning`, `dynamic_load`) | 12 |
| 32 | Shape-dependent graphs: `video_N`/`audio_N` derive from resolution/duration/fps | 12 |
| 33 | Cost model can't rank at pipeline scale: no link contention, no comm/compute overlap, no per-op fixed cost | 13 |

### F. Workflow

| # | Blocker | P |
|---|---|---|
| 34 | No entry point: pipelines have no dry-run / capture flag | 13 |
| 35 | No golden baselines for a CI gate | 13 |

### G. New: shim-specific

| # | Blocker | P |
|---|---|---|
| 36 | **Per-device shape and tile-padding math must be exact.** The graph branches on it — `attention_ltx.py:483` does `need_gather = k_BHNE.shape[2] < _k_cos_pe.shape[2]`. An off-by-a-factor doesn't perturb the graph, it flips a collective in or out of existence | 7 |
| 37 | **`weight._data is None` gates the graph.** `attention_ltx.py:379`: `_compute_gate` returns `None` when the gate weight is unloaded, so a weightless dry run silently loses the exact finding phase 5 reported. Needs `torch.device('meta')` weights so `Parameter._data` is non-None without bytes | 6 |
| 38 | **Checkpoint-derived flags.** `has_gate` comes from state-dict *keys* (`transformer_ltx.py:1090`), as does `cross_attention_adaln`. Needs a key list from a safetensors index without downloading weights | 7 |
| 39 | **Host-value dependence.** `transformer_mochi.py:575` derives `valid_prompt_length` via `encoder_attention_mask.sum(dim=1).max().int().item()`, which drives shapes. Needs representative host inputs or supplied lengths | 12 |
| 40 | **Device and config object stubs.** `MeshDevice.shape/arch/compute_with_storage_grid_size/create_submeshes`, `CoreGrid`/`CoreCoord`/`CoreRangeSet`, `SDPAProgramConfig`, `MemoryConfig`, compute-kernel configs, `SubDevice`, `create_global_semaphore` — and `get_matmul_config`'s assertions must be satisfiable from fake shapes | 6 |
| 41 | **Pipeline construction touches the device before any forward.** `CCLManager.__init__` calls `_init_subdevice()` and `_init_semaphores()` (many `create_global_semaphore`) and `synchronize_device`; pipelines also build persistent buffers and prepare weights at init | 6 |
| 42 | **Shim/ttnn divergence over time.** The shim is a second implementation of ttnn's shape/layout semantics and will rot | 11 |
| 44 | **Source attribution points at shared library code.** Spike findings landed on `layers/linear.py:250` (the AGMM call site inside `ColParallelLinear`) rather than `attention_ltx.py:428` (`to_qkv`). Both are true; only the second is actionable. Record a short caller stack (2-3 tt_dit frames), not one line | 6 |
| 43 | **Readback boundaries end the graph.** Everything after `to_torch(get_device_tensors(...))` (`transformer_ltx.py:1068`, `bwe_ltx.py:85`, `mel_decoder_ltx.py:483`) is host code on fake data; and no kernel-level constraint (program-config asserts, L1 fit, hangs) is visible in a dry run | 10 |

## Phases

Estimates are for one engineer. Only phase 11 needs device access.

### Phase 6 — Dry-run shim core (2–3 weeks) · closes 37, 40, 41, 44; dissolves 3–9

- A `ttnn` stand-in (installed by import shadowing, not by editing model code): metadata
  `Tensor`, `MeshDevice`, mesh mappers/composers, config objects, semaphores,
  subdevices, `synchronize_device`, trace capture as no-ops.
- `from_torch` accepts `torch.device('meta')` tensors, so `Module`/`Parameter`
  loads shapes with zero bytes and `_data` is non-None (37).
- Ops emit IR nodes as a side effect of returning metadata; distribution is
  recorded at creation, so entry placements are known rather than declared.
- A no-device pipeline construction path (or shim coverage of `CCLManager.__init__`
  and persistent-buffer allocation) so a pipeline object can exist on a laptop.
- Record a short caller stack per node, not a single innermost frame (44), so a
  finding names the model call site and the library line under it.
- **Acceptance — met by the spike** for one block: the real forward runs, the
  collectives match `examples/ltx.py` as an identical multiset (31 vs 31 on Ring,
  25 vs 25 on Linear), and the findings are byte-identical (6 provable
  `duplicate_gather`, 128.7 GiB/forward; 0 on Linear). `examples/ltx.py` stays as
  the oracle and the drift regression test (`spike/test_dryrun_matches_oracle.py`).
  What remains for the production version: real torch-meta weights instead of
  `fake_torch`, the `unregistered` node kind, pipeline-level construction, and the
  caller stack.
- **Prerequisites the spike surfaced:** the repo needs Python ≥ 3.10 (PEP 604
  unions in evaluated annotations, `types.NoneType`), and
  `models.common.utility_functions` pulls in numpy *and pytest* for the sake of
  `is_blackhole` — worth splitting upstream.

### Phase 7 — Shape and layout fidelity (2–3 weeks) · closes 10, 11, 12, 13, 36, 38

The load-bearing wall: shapes decide branches (36). The spike's two bugs are
exactly this phase's content — treating `num_heads_per_device=1` (the no-split
default) as a head split, and reusing a fused weight's symbol for a chunked AGMM.
Between them they produced 15 spurious findings next to the 6 real ones.

- Implement tile padding and expose both `shape` and `padded_shape`; regions and
  demand on logical extents, byte/cost math on padded.
- Reproduce ttnn's shard division exactly, including uneven splits, from the
  mapper's own rules rather than a ceil assumption.
- Weight layout as a declared property of a linear layer (`per_device_qkv`,
  `swiglu_interleave`, …), replacing today's hardcoded default; run the real
  `_prepare_torch_state` on meta tensors so preprocessing shapes come for free.
- Checkpoint key lists from a safetensors index for `has_gate`-style flags.
- Block-float byte table.
- A real `chunked_weight` spec: `to_qkv(chunks=3)` consumes a column block of a
  per-device-interleaved weight, which the spike models as separate weights.
- **Acceptance:** for one block, every per-device shape the shim computes matches
  a recorded real run (the phase 11 collective log), and no branch differs.

### Phase 8 — Op coverage, one registration per op (3–4 weeks) · closes 2, 14, 15, 16, 17, 18

- Merge shim shape rule and analyzer semantics into a single `OpSpec`; extend
  `ops --missing` to stub all three functions; build the `unregistered` node kind
  and the withhold-don't-guess rule; add `ops --check` for CI.
- Declare fused-kernel internal stages as **data** (fused op → comm stage, compute
  stage, axis/dim argument names), so a new fused kernel is a table entry rather
  than a silent miss.
- Tier-2 specs for what actually appears in DiT forwards: `mesh_partition`, `pad`,
  `copy`, `repeat`, `lerp`, `embedding`, `minimal_matmul_split`,
  `dit_minimal_matmul_addcmul_fused`, `dit_rms_norm_unary_fused`,
  `exp_ring_joint_sdpa`, `rotary_embedding_hf`.
- Conv/VAE family: conv2d/3d, `group_norm`, `upsample`, `neighbor_pad_async` (a
  genuinely new communication shape — halo exchange), `slice_reshard_async`,
  `send_async`/`recv_async`, plus real rank-changing reshape semantics so the VAE
  stops tainting.
- **Acceptance:** LTX and VAE dry runs complete with zero `unregistered` nodes and
  zero `UNKNOWN_OP` diagnostics.
- **Calibration:** one transformer block touches **18 distinct ttnn ops**, and the
  spike's 53 implemented names covered it with room to spare. The ~100-op estimate
  for the whole surface stands, but the tail is VAE/encoder, not the DiT — so this
  phase can ship DiT-complete early and grow into the VAE.

### Phase 9 — Scale (1–2 weeks) · closes 20, 21, 26, 27

Cheaper than in revision 1: the shim knows the Python call stack and loop index,
so block boundaries and `calls` come from the run instead of being inferred.

- Deterministic node IDs from `(op, source location, occurrence index in block)`.
- Roll repeated blocks up to one instance + `calls`, keeping the first and last
  instances intact so boundary effects aren't hidden.
- Liveness-pruned, copy-on-write state; snapshot only at collectives.
- Roll findings up by `(rule, source location)`, leading with the outermost model
  frame from the caller stack recorded in phase 6 (44).
- **Acceptance:** a full 48-layer dry run analyzes in <60 s and <2 GB, with
  findings matching the rolled-up single-block result.

### Phase 10 — Multi-mesh, multi-stage, multi-host (2–3 weeks) · closes 19, 22, 23, 24, 25, 43

- `Graph` holds a set of meshes; symbols carry a mesh id; collectives reference
  `(mesh, axis)`. Submesh and CFG-parallel pipelines then model directly.
- Generalise `Dist` beyond two axes and map encoder/VAE parallel configs onto it.
- Stage boundaries (encoder → DiT → VAE) as linked graphs; readbacks become
  explicit edges that terminate a graph (43).
- Carried state (latents, KV cache, `StateTensor`) marked `carried`, so `steps` is
  derived rather than supplied.
- **Acceptance:** one dry run of a full `pipeline_ltx` generation yields a linked
  graph set over encoder, transformer and VAE, with `steps` inferred.

### Phase 11 — Conformance and soundness gates (2–3 weeks, needs a device) · closes 1, 28, 29, 30, 42

The device's whole remaining job, plus the gates between "look here" and "do this".
The spike is the argument for the conformance half: two shape bugs invented 15
findings that read exactly as convincingly as the 6 real ones.

- Per-op conformance harness: for each registered op, build inputs on a real
  mesh, run real ttnn, and assert the shim's shape/layout/dist match. Runs in
  device CI; a mismatch is a hard failure naming the op.
- Whole-block collective log (flat, no tensor identity) diffed against the dry
  run: same collectives, same order, same dims/axes/shapes, same source lines.
- Buffer liveness: record persistent-buffer identity and semaphore IDs, and
  suppress a `duplicate_gather` CSE recommendation when the earlier result's slot
  is reused before the candidate site — with the reason stated.
- Barrier/sync intent: flag collectives whose removal changes synchronisation and
  demote them to "needs review".
- Live-bytes tracking so a recommendation can say whether keeping the earlier
  result actually fits.
- **Acceptance:** conformance green for every registered op; the 6 LTX findings
  each carry a buffer-liveness verdict and a memory-feasibility note.

### Phase 12 — Branch and shape coverage (1 week) · closes 31, 32, 39

Nearly free once the dry run works: sweeps are laptop CPU time, not device time.

- Manifests recording the branch flags and shapes a graph corresponds to
  (topology, `has_audio`, `has_gate`, stage, LoRA mode, resolution, frames).
- `ditcheck matrix` runs the combinations, merges findings, and **names the
  branches never exercised**, so unexercised paths are visible rather than
  assumed clean.
- Representative host inputs (or supplied lengths) for value-dependent shapes (39).
- **Acceptance:** an LTX matrix run covers both topologies × both stages × audio
  on/off and lists uncovered branches by name.

### Phase 13 — Workflow integration (1–2 weeks) · closes 33, 34, 35

- `ditcheck dryrun --pipeline …` entry point plus one shared helper; no pipeline
  edits beyond a config hook.
- Golden findings baselines per config; **a device-free CI job** running
  `dryrun` + `analyze --fail-on provable` + `ops --check` on every PR. This is the
  payoff of the shim design: a redundancy check at unit-test cost.
- Optional time-based ranking by joining measured per-op latency from a perf CSV.
  A ranking aid, not a latency model — contention and overlap stay out of scope.
- **Acceptance:** a provable redundancy introduced in a PR fails CI with the
  source line in the output, on a runner with no Tenstorrent hardware.

## Ordering

```
6 (shim core) ──► 7 (shape fidelity) ──► 8 (op coverage) ──► 9 (scale) ──► 13 (device-free CI)
                                              │
                                              ├──► 10 (multi-mesh / stage)
                                              ├──► 11 (conformance + soundness, on device)
                                              └──► 12 (coverage matrix)
```

6 → 7 → 8 is the critical path to "runs on a real pipeline with no hand-written
model" (~8–11 weeks). Phase 11 does not gate producing findings, but it does gate
*trusting* them: until per-op conformance is green, every dry-run finding should
be read as "the shim believes", and phase 7's acceptance criterion depends on one
recorded collective log, so a small amount of device time is needed early.

## Risks

| Risk | Mitigation |
|---|---|
| Shim shape math diverges from ttnn and flips a branch (36) — the worst failure mode, because it produces confident wrong findings, **observed in the spike: 2 bugs, 15 spurious findings** | Per-op conformance in device CI (11); phase 7 acceptance is a per-device-shape diff against a real run; keep the hand-written `examples/` graphs as regression oracles rather than deleting them |
| The shim rots as ttnn adds or changes ops (42) | `ops --check` makes coverage debt visible; conformance failures name the op; fused-kernel behaviour lives in a data table |
| A weightless or input-free dry run silently changes the graph (37, 38, 39) | Meta-tensor weights so `_data` is non-None; checkpoint key lists; the coverage matrix reports which branches were never exercised |
| Import shadowing `ttnn` is fragile or leaks into real runs | Install the shim only under an explicit entry point, and assert `ttnn.__file__` is the shim at graph-emit time |
| Findings scale faster than anyone reads them | Rollup by source location plus top-N ranking; the report is a queue, not a dump |

## Out of scope

Automated rewrites (plan §choice 4: proofs before auto-fixes), training/backward
graphs, non-DiT models, and a predictive performance model. Kernel-level
validation stays with the device — the shim answers "which collectives exist and
which are redundant", not "will this program fit and run".
