# Roadmap: from hand-modelled graphs to real pipelines, without a device

Companion to [`DitStaticAnalyzerPlan.md`](DitStaticAnalyzerPlan.md), which covers
phases 0–6 of the original design. Phases 1–4 are prototyped in this directory and
phase 5 was exercised against LTX-2.3 — originally against a graph **hand-written
from the source** (`examples/ltx.py`), and now against one derived from the source
by the dry run, with the hand-written version kept as the oracle. This document
inventories what stands between that and running the checker on a real *pipeline*,
and lays out the work to close it.

> **Revision 2 — adopts a dry-run front end.** The first version of this roadmap
> assumed on-device graph capture was the way in, and spent its first two phases
> fighting tensor identity under buffer reuse, trace replay and `deallocate`.
> This version makes a **metadata-only `ttnn` shim** the primary front end and
> demotes the device to a *validator*. Seven of the original 35 blockers dissolve
> outright; nine new shim-specific ones replace them. The critical path gets
> shorter and, more importantly, the everyday analysis stops needing hardware.
>
> **Phase 6 is built** — [`dryrun/`](dryrun/) is the front end, driven by
> `ditcheck dryrun`. It grew out of the spike recorded in
> [`spike/FINDINGS.md`](spike/FINDINGS.md), so the phases below are calibrated
> against working code rather than an estimate.

## Where we are

**This section is the project's status of record.** The plan and the README point
here rather than restating it, so there is one place to update and nowhere for a
stale copy to disagree.

> **Scope (decided 2026-08-05): a hand-run tool, not a CI gate.** ditcheck is meant
> to be run by an engineer when they want it, not wired into PR CI. That drops the
> automated CI job and its golden baselines (blocker 35) and demotes stable-ID
> run-to-run diffing (blocker 27), and it **re-orders the remaining phases**: with
> no build to keep green, the bottleneck is no longer automation but *trust* — an
> engineer must believe a finding before spending kernel-engineer time on it. So
> **on-device conformance (phase 11) becomes the most important remaining phase**,
> followed by reach (phase 10, whole pipelines) and the readability/scale part of
> phase 9. See [§ Ordering](#ordering).

| | |
|---|---|
| Analysis | works offline, pure Python: forward availability + backward demand + 6 redundancy rules + proofs |
| Op semantics | 21 analyzer specs, ~100 aliases; the shim's pointwise/passthrough dispatch is generated from one `semantics.GENERIC_OPS` table (one registration per generic op); fused kernels declared in `dryrun/fused.py` |
| Dry run | **built**: `ditcheck dryrun ltx_block --preset bh_4x8`, real forward under a metadata-only ttnn, weights from torch meta tensors, caller stack per node, `--check-oracle` as the drift test |
| Capture | 30 monkeypatch hooks written, **never run on hardware**; demoted to the phase 11 conformance path |
| Graph source | `ditcheck dryrun` from source, or hand-written via `builder.py`; a trace still needs hand-declared placements |
| Validated on | Four blocks/stages **derived from source** by the dry run: LTX-2.3 block ×2 topologies (6 provable duplicates on Ring, 0 on Linear), the SD3.5-large joint block (0 findings), the SD3.5 VAE ResnetBlock (conv/group_norm), and a T5 text-encoder layer (tensor-parallel). The two DiT blocks are diffed against a hand-written oracle; the others checked for zero unregistered ops. `ditcheck link` composes them into multi-stage pipelines (e.g. T5 encoder→DiT). Asserted in `tests/test_dryrun.py` |
| Tests | 26 analyzer + 25 dry-run, no device and no pytest required; plus `conform.py` on a device |
| Conformance | **11a+11b green on the 2×4 Loudbox, plus the first *finding* conformed.** `conform.py` (12/12): distribution/tile-padding for LTX + SD3.5 + matmul/split-qkv/concat output shapes. `conform_collectives.py` (3/3): all-gather (tp & sp) and reduce-scatter through the real `CCLManager`. `conform_block.py`: the **whole SD3.5 block runs on hardware** and its collective log matches the dry run — all-gather(tp)=6, all-gather(sp)=2 (ring-SDPA K/V), reduce-scatter(tp)=2. `conform_encoder.py`: the **`replicated_stage` H3-encoder finding conformed on silicon** — the real encoder built on the 2×4, all 8 device shards read back, SP-row 0 vs SP-row 1 **bit-for-bit identical on every TP column (max\|Δ\|=0)**, the first dry-run finding proven against real hardware. **2026-08-08 — all three findings re-conformed on the 4×8 Blackhole Galaxy over the real Ring fabric**, at production `sp1tp0`: encoder `replicated_stage` **7 of 8** SP rows (TP=4), audio `unused_gather` **7 of 8** SP shards, output-head `participant_shrink` **3 of 4** TP copies unread — every one `max\|Δ\|=0`. Scale changed the counts, not a single verdict. **The whole prod report then followed** (galaxy workstream 2): 23 findings over 15 nodes triaged on the laptop — no shim artifacts, one shared root cause (the packed sequence puts text+audio in SP shard 0) — and the two un-conformed classes proven on device: the DiT text branch (`conform_dit_text.py`, 7 of 8 SP copies redundant) and `overwide_gather` (`conform_overwide.py`, a **poison test** — equality can't prove a region is unread, so the claimed-unread rows were filled with a sentinel and every read row came back `max\|Δ\|=0` while the poisoned rows themselves moved). Remaining: conv2d/group_norm *compute* kernels, soundness gates (11c) |

### Phases

Plan phases 0–5 built the analyzer; roadmap phases 6–12 are about running it on
real code by hand (the CI half of the old phase 13 is now out of scope).

| phase | state | what remains |
|---|---|---|
| plan 0–4 — IR, simulator, demand, rules | **done** | — |
| plan 1 — graph capture | **superseded** | replaced by the dry run; capture survives as the phase 11 validator, still never run on hardware |
| plan 5 — validate on historical wins | **partly** | rediscovers the LTX win from source and passes the SD3.5 precision test; no wider corpus of past AGMM graphs, and none of the human metrics (false-positive rate over a real backlog, time per finding) |
| **roadmap 6 — dry-run front end** | **done** | — |
| **roadmap 7 — shape and layout fidelity** | **done (7a); 7b on 2×4, Ring blocked** | tiling, exact shard division, block-float bytes, checkpoint keys, and the chunk rule all shipped and corroborated against real ttnn on the 2×4 Loudbox via `conform.py`; the 4×8 Ring corroboration needs a 32-chip Galaxy |
| roadmap 8 — op coverage | **core done** | LTX, SD3.5-large **and** the SD3.5 VAE ResnetBlock covered from source (0 unregistered); fused-kernel data table + generic-op one-registration merge landed. First external model scouted (`jonathansu/minimax-h3-textencoder`, a Qwen3-VL text decoder): dry-runs to 0 unregistered after adding `embedding` + GQA `nlp_create_qkv_heads`, then analyses **clean** (13 collectives, all necessary — no redundancy). Note: the first analysis reported 13 false `dead_collective` findings; tracing them found a shim `Tensor.__getitem__` bug (Ellipsis slices landed on the key's positional axis, so RoPE's `x[..., :d//2]` sliced heads instead of head_dim, collapsing attention to empty). Fixed + regression-tested — a concrete case of "the shim believes" catching a wrong result before it reached a human. The remaining tails (LTX-VAE halo/`neighbor_pad`, conv3d, Mochi/Wan tier-2, and the H3 DiT/VAE on Kevin's & Colman's branches) surface as those targets are added and are partly phase-10-gated |
| **roadmap 11 — conformance (needs a device)** | **11a + 11b done on 2×4; first finding conformed** | per-op conformance (`conform.py` 12/12), collective conformance (`conform_collectives.py` 3/3), and the whole SD3.5 block run on hardware with its collective log matching the dry run (`conform_block.py`). **`conform_encoder.py` conforms an actual *finding*:** the `replicated_stage` H3-encoder redundancy — real encoder on the 2×4, all 8 device shards read back, SP rows bit-for-bit identical (max\|Δ\|=0), so the pipeline provably composes and discards a duplicate SP row. "shim believes" → device-proven for this finding. **Now also proven at true scale:** the same three findings re-conformed on a 4×8 Blackhole Galaxy over Ring (2026-08-08) — 7 of 8 SP rows, 7 of 8 audio shards, 3 of 4 TP copies, all `max\|Δ\|=0`; see [`GALAXY_PLAN.md`](GALAXY_PLAN.md) § workstream 1. Remaining: 11c buffer-liveness/barrier/memory soundness gates |
| roadmap 10 — multi-mesh / stage / host | **10a + 10b + 10c-linking done; real encoder stage landed (reach)** | 10a: readback boundaries split a graph into device segments (`Graph.segments()`, blocker 43). 10b: multi-mesh IR — `Graph` mesh registry + per-node/symbol mesh id, analysis resolves each node's mesh (`mesh_of`), submeshes with different device counts model directly (blocker 22); single-mesh byte-identical. 10c: `link_stages` / `ditcheck link` compose independently-dry-run stages into one multi-stage, multi-mesh, segmented pipeline graph and analyze it. **A full encoder→DiT→VAE pipeline now dry-runs from source and analyzes end-to-end:** `ditcheck link enc=t5.json dit=sd35.json vae=vae.json --analyze` → 3 segments, 3 meshes (1×4 T5 + 2×4 DiT + 2×4 VAE), 142 nodes. All three stages are real model code (`t5_encoder_layer`, `sd35_block`, `sd35_vae_resnet`), each 0 unregistered. Getting there: a shim `softmax`; a `diffusers` stub in hostenv (a reference-only dep, not used by shape analysis) so the VAE dry-runs dependency-free; a metadata-torch `no_grad`/`inference_mode` usable as a decorator; and an honesty fix — a batched attention matmul (`q@kᵀ`) breaks the weight-matmul K assumption, so a `K_COVERAGE` mismatch now taints the result (finding → suspicious, never a false provable). **Data-connected linking started** — `link_stages(..., connect=True)` wires stage N's output into stage N+1's matching input across the readback boundary and seeds demand only from the final stage, so backward demand crosses the boundary. This surfaces **cross-stage** redundancy that per-stage analysis can't: demonstrated on synthetic stages where a gather that replicates to every device, feeding a boundary that reads back device 0, is flagged redundant only once linked (necessary standalone). **Applied to the real encoder→DiT** (built on one mesh, as the pipeline does): the TP-only text encoder runs replicated across the SP axis, so its output is identical on every SP row but the handoff consumes one — at 2×4 that is 1 of 2 rows wasted, at 4×8 **7 of 8** (encoder clean standalone; the finding appears only when connected). **Conformed on silicon** (`conform_encoder.py`, 2×4): the real encoder's output read back from all 8 devices is bit-for-bit identical across the SP rows (max\|Δ\|=0 on every TP column) — the redundancy is a physical duplicate, not a shim artifact. This class is **reframed** away from `dead_collective` ("delete the collective", wrong — the live group needs it) to a new **`replicated_stage`** rule (MEDIUM/likely): "replicated across N groups, consumed on M — run it on a submesh that omits the redundant groups." A `dead_collective` that holds on a *strict subset* of a collective's participant groups is now this rule; dead on *all* groups stays a true deletable `dead_collective`. **Faithful boundaries + shape-bridging** — the connect boundary is now a `stage_boundary` op that reads the full logical output from a **minimal covering set** of devices (replicated data read once → over-replication exposed; sharded data read from every shard → no false redundancy), replacing the hard-coded device-0 readback. Confirmed against the real pipeline: the encoder finding matches its own source comment ("replicated across the mesh: read one replica rather than composing all 32 and discarding 31"). And a handoff may now **change shape** — `link_stages([..., (name, graph, in_sym)])` wires an explicit input, and the boundary bridges a host reshape/unpatchify (e.g. DiT→VAE), demanding the whole previous output. Remaining: auto-run the real `pipeline_ltx.generate()` / `pipeline_minimax_h3` in one process (host denoise loop, weights from disk), the real DiT→VAE unpatchify wiring end-to-end, carried state / `steps` (24), LTX-VAE halo ops (14 tail) |
| roadmap 24 — carried state / the denoise loop | **done for step-invariance; the loop is still not unrolled** | `recomputed_stage` flags a collective whose every input traces to values that cannot change between denoise steps, so the branch is redone identically each evaluation — H3's token refiner takes only `prompt_1BLP`, and the pipeline builds the prompt above its `for i, t in enumerate(timesteps)` loop, so it repeats on all 49. Soundness rests on two choices: an undeclared entry reads as *varying* (`step_varying=False` is opt-in, so a graph that declares nothing yields no findings rather than wrong ones), and a stage running less often than its consumer is constant to it (otherwise connecting encoder→DiT hides the redundancy connecting them reveals). Distinct from `invariant_collective`, which asks whether the same *bytes* are re-sent; this asks whether the same *computation* is redone. Production: 5 findings, **+31.5 GiB per generation**, all text-branch. Remaining: genuinely unrolling the loop (a value carried step→step, e.g. a KV/adaln cache, still cannot be reasoned about) |
| roadmap 9 — scale | **rollup done; runtime/memory never became a problem** | `report.rollup_findings` groups findings by (rule, source chain, per-call bytes, verdict) and ranks by *total* impact, so a 50-layer stack reads "×50 occurrences (same call site) = 8.2 GiB across the stack" instead of 50 rows: at production depth **321 findings → 25 distinct**, which is what makes a top-N cut show N distinct problems rather than one repeat N times. Runtime/memory turned out to be a non-issue — real depth (encoder 50, DiT 50) is 5838 nodes and 21.7 s on a laptop, growing linearly. Remaining: the loop/`steps` dimension (blocker 24). Stable-ID run-to-run diffing **dropped with CI** |
| roadmap 12 — branch and shape matrix | not started (optional) | one graph is currently one branch; a manual convenience, not on the spine |
| ~~roadmap 13 — CI job + golden baselines~~ | **out of scope** | dropped by the hand-run decision; only the `--pipeline` entry point and optional cost ranking survive, folded into phase 10 / as an optional aid |

**Blockers: ~23 open and in scope of 44** — 7 dissolved by the dry-run design, 4
closed by phase 6, 6 by phase 7 (10, 11, 12, 13, 36, 38), 3 by phase 8 (2, 17, 18;
14 partly). Open counts by phase: 8→3 (15 `mesh_partition`, 16 p2p, 14 tail),
9→3, 10→6, 11→5, 12→2. **Out of scope under the hand-run decision:** blocker 35
(golden baselines) is dropped, 27 (stable IDs for CI diffing) demotes to optional,
and 34/33 shrink to a `--pipeline` entry point + optional cost ranking. (Blocker 36
is closed for *shape math*: the 2×4 shapes are diffed against real ttnn by
`conform.py`; the 4×8 Ring diff still awaits a Galaxy.)

**The plan's v1 bar is met.** It set the go/no-go at "if it can reliably rediscover
Kevin's finding, and do so early in bring-up, it is already worth the build cost"
(plan § "What 'done' looks like"). The tool now finds it with a proof and a byte
estimate, from the model source, on a laptop, in a few seconds — and reports
nothing on the SD3.5 block where every collective is load-bearing. What the phases
above are still buying is *reach*: one block is not a pipeline, and one
corroborated block is not a trustworthy tool.

Until phase 11's per-op conformance is green, a dry-run finding means *"the shim
believes"*. Today's 6 LTX findings are corroborated by a hand-written oracle built
independently from the same source, which is real evidence — but it is the only
block that has one. Phase 7b took the first bite of the conformance gate:
[`conform.py`](conform.py) builds representative LTX tensors on a real 2×4
Blackhole mesh and diffs each per-device (logical *and* tile-padded) shape against
the shim. Every shape the LTX 2×4 block actually uses matches ttnn exactly; the
one deliberately-uneven probe is refused by ttnn itself under tile layout (its
shards must be uniform, which is why tt_dit pads first — and the shim's predicted
chunk size is the value ttnn *expected* before it refused). This validates the
shape math a wrong-shape finding would exploit, but only on the 2×4 (Linear)
config — the 4×8 Ring finding still needs a 32-chip Galaxy to corroborate.

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
dependency, which is testable per-op rather than per-model. Two
alternatives are non-starters: static AST analysis of the pipelines (too
config-dynamic to be sound), and the GSPMD-style approach of checking a declared
sharding plan against the torch reference — that answers "what collectives
*should* exist" and would not have caught the LTX finding, which lives in the
ttnn implementation, not the math. `TT_METAL_SIMULATOR` (`tt_metal/llrt/rtoptions.cpp`)
runs real kernels far too slowly for whole-pipeline graphs.

## Target end state

```bash
# on a laptop: no device, no weights, no capture. Works today for one block:
ditcheck dryrun ltx_block --preset bh_4x8 --out ltx_bh4x8.graph.json --analyze

# the pipeline-level form is phases 10 and 12:
ditcheck dryrun --pipeline models.tt_dit.pipelines.ltx --preset bh_4x8 \
    --frames 121 --height 704 --width 1216 --out ltx_bh4x8.graph.json
ditcheck analyze ltx_bh4x8.graph.json --top 10

# on a device, occasionally, by hand: keep the shim honest (phase 11)
ditcheck conform --ops all_gather_async,minimal_matmul,...   # per-op shape/layout
ditcheck conform --block ltx --against ltx_bh4x8.graph.json  # flat collective log
```

## The one thing that stays manual: op registration

Op semantics cannot be inferred from a call. That is by design (plan §"Key
engineering choices", choice 3) and stays the single manual surface.

**Where the two halves genuinely are one piece of knowledge, they are now one
registration.** For the generic families (pointwise / passthrough), a call is
declared once in `semantics.GENERIC_OPS` — canonical op + shim shape-rule — and
the shim's dispatch is generated from it (phase 8):

```python
# semantics.py -- one line, one file; the shim reads this table
GENERIC_OPS = { "silu": ("pointwise", "unary", "silu"), "to_layout": ("identity", "passthrough", None), ... }
```

For a **new communication or compute op** the two halves live in two layers on
purpose: the shim emits canonical op names, so the analyzer registry has to stand
alone (a `dump`ed graph is `analyze`d with no shim — the CI path), which means the
analyzer cannot import the shim. So a genuinely new op is an `OpSpec` in
`semantics.py` (`apply` + `demand`) plus a shim rule in `dryrun/ops.py`; the
`ops --missing --stub` generator emits both halves to paste. Revision 1 imagined a
single fused `OpSpec` with a `shim=` callable — that does not fit the layering, so
the realised form is the shared `GENERIC_OPS` table for the common case and a
stub-assisted two-file entry for new semantics.

Ergonomics, unchanged in intent from revision 1; 1, 2 and 4 are built:

1. **Nothing is invisible.** An unregistered op still runs under the shim and
   still appears in the IR as an `unregistered` node with real inputs, outputs,
   shapes and source location. It cannot propagate metadata, so the shim assumes
   its output matches input 0 and records that assumption on the node — the run
   continues deliberately, so one pass lists *every* missing op rather than
   stopping at the first. Nothing built on the assumption is ever reported (2).
2. **Analysis withholds, never guesses.** A finding whose proof depends on an
   `unregistered` node is not emitted, not downgraded. `Report.withheld` carries it
   with the registrations that would unlock it, and the text report prints them as
   a queue. Provenance propagates automatically in `ApplyCtx.define`, so a spec
   author cannot forget to pass it on.
3. **`ditcheck ops --missing`** prints, per unregistered op: call name, arity,
   call sites and count — plus how many findings it blocks. Still to add: the
   copy-paste stub with `shim`, `apply` and `demand` left as `TODO` (phase 8, once
   the two halves are one entry).
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
| 6 | In-place ops (`ttnn.copy`, `multiply_`, `add_`, `StateTensor.update`) break the SSA assumption | **dissolved** as a hazard — the shim repoints the caller's tensor at a fresh symbol, so a write is an ordinary node; done in P6 for `copy`/`*_`, `StateTensor` in P10 |
| 7 | Mesh mappers/composers unhooked (`ShardTensor2dMesh`, `ConcatMesh2dToTensor`, `create_mesh_composer`) | **dissolved** — the shim *is* the mapper, so distribution is known at creation |

### B. Information a trace doesn't carry

| # | Blocker | P |
|---|---|---|
| 8 | Entry placements must be declared: `.shape` is per-device, nothing says which mesh axis fractures which tensor axis | **dissolved** — recorded when the shim creates the tensor |
| 9 | Axis roles (sp/tp/cfg) live in `DiTParallelConfig`, not on tensors | **dissolved** — the dry run reads the config object in-process |
| 10 | `padded_shape` vs logical shape: matmuls use tile-padded shapes, region math uses logical | **closed in 7** — `shape`/`padded_shape` split; cost math on `region.padded_volume` |
| 11 | Uneven shards (38→40 heads for tp=4); shard split must match ttnn's mapping exactly | **closed in 7** — one `shard_chunk_size` reproduces ttnn's `torch.chunk` rule, verified on 2×4 |
| 12 | Model-specific weight preprocessing (`_interleave_heads`, `prepare_for_fused_swiglu`, `permute_for_swiglu`) changes logical column order | **closed in 7 for shape** — shapes validated via `total_shape`+`_check_data`; column *order* not consumed by the analyzer |
| 13 | Block-float byte accounting: `bfp8_b`/`bfp4_b` exponent overhead missing from `elem_bytes` | **closed in 7** — `ir.elem_bytes_for` (1.0625 / 0.5625) |

### C. Semantics coverage

| # | Blocker | P |
|---|---|---|
| 14 | Conv/VAE family missing (conv2d/3d, group norm, upsample, `neighbor_pad_async` halo, `slice_reshard_async`) | **partly closed in 8** — `conv2d` + `group_norm` done (SD3.5 VAE ResnetBlock is clean); conv3d + LTX-VAE halo/`slice_reshard` remain |
| 15 | `mesh_partition` unmodelled although it *changes distribution* | 8 |
| 16 | Point-to-point comm (`send_async`/`recv_async`) has no participant-group semantics | 8 |
| 17 | Opaque reshapes (VAE's `b,h,w,c → 1,1,h*w,c`) degrade regions to full + taint | **closed in 8** — a reshape that only merges/splits leading axes preserves a shard on the kept trailing (channel) axis |
| 18 | Fused-kernel internal comm is a hand-maintained registry; a new fused kernel silently hides its collective | **closed in 8** — `dryrun/fused.py` `FUSED_KERNELS` table drives AGMM/MMRS, is inspectable (`ops --fused`), and `ops --missing` flags fused-looking unregistered ops |
| 19 | Non-DiT parallel configs (`EncoderParallelConfig`, `VaeHWParallelConfig`, `MochiVAEParallelConfig`, `AudioTParallelConfig`) don't map onto the single 2-axis `Dist` | 10 |

### D. Graph structure and scale

| # | Blocker | P |
|---|---|---|
| 20 | Loop rollup: 48 layers become 48 copies; `calls` is hand-set today | 9 |
| 21 | Quadratic analysis cost: 1 block (160 nodes) = 0.59 s / 9 MB → naive 48-layer graph extrapolates to ~20 min / ~20 GB | 9 |
| 22 | Multi-submesh pipelines (118 submesh references; CFG/VAE on separate `MeshDevice`s) vs one `Graph.mesh` | **IR done in 10** — `Graph.meshes` registry + per-node/symbol mesh id + `mesh_of`; analysis resolves each node's mesh. Populating it from real submesh pipelines is 10c |
| 23 | Multi-host meshes: `distributed_context_get_rank`, socket pairs, fabric config | 10 |
| 24 | Cross-forward state (latents, `StateTensor`, KV caches); `steps` is hand-supplied | 10 |
| 25 | Host-side gaps (scheduler/guidance between forwards) split the graph | 10 |
| 26 | Report doesn't scale: 22 TP gathers × 48 layers needs finding rollup | 9 |
| 27 | Counter-based node IDs are unstable across runs, so CI can't diff findings | **optional** — the run-to-run diff motivation went with CI; only matters if a human wants to diff two runs |

### E. Soundness gaps (wrong answers, not missing ones)

| # | Blocker | P |
|---|---|---|
| 28 | Semaphores/barriers unmodelled: a "redundant" collective may exist for synchronisation (`vae_all_gather`'s barrier comment) | 11 |
| 29 | Persistent ping-pong buffers: "reuse the earlier result" is invalid if an intervening collective overwrote that slot | 11 |
| 30 | No memory model: removing a gather changes L1/DRAM residency | 11 |
| 31 | One graph = one branch (`skip_qk`, `has_gate`, `kv_replicated`, `use_ring_cross`, stage 1/2, LoRA mode, `image_conditioning`, `dynamic_load`) | 12 |
| 32 | Shape-dependent graphs: `video_N`/`audio_N` derive from resolution/duration/fps | 12 |
| 33 | Cost model can't rank at pipeline scale: no link contention, no comm/compute overlap, no per-op fixed cost | **optional** — a ranking aid for a human, from a perf CSV; not a latency model |

### F. Workflow

| # | Blocker | P |
|---|---|---|
| 34 | No entry point: pipelines have no dry-run / capture flag | 10 — a `--pipeline` entry point is still wanted for hand runs on whole pipelines |
| 35 | No golden baselines for a CI gate | **out of scope** — hand-run tool, no CI gate to keep green |

### G. New: shim-specific

| # | Blocker | P |
|---|---|---|
| 36 | **Per-device shape and tile-padding math must be exact.** The graph branches on it — `attention_ltx.py:483` does `need_gather = k_BHNE.shape[2] < _k_cos_pe.shape[2]`. An off-by-a-factor doesn't perturb the graph, it flips a collective in or out of existence | **closed in 7 for 2×4** — `conform.py` diffs every shape vs real ttnn; 4×8 Ring diff needs a Galaxy |
| 37 | **`weight._data is None` gates the graph.** `attention_ltx.py:379`: `_compute_gate` returns `None` when the gate weight is unloaded, so a weightless dry run silently loses the exact finding phase 5 reported. Needs `torch.device('meta')` weights so `Parameter._data` is non-None without bytes | **closed in 6** |
| 38 | **Checkpoint-derived flags.** `has_gate` comes from state-dict *keys* (`transformer_ltx.py:1090`), as does `cross_attention_adaln`. Needs a key list from a safetensors index without downloading weights | **closed in 7** — `dryrun/checkpoint.py` (safetensors header / index / declared manifest) |
| 39 | **Host-value dependence.** `transformer_mochi.py:575` derives `valid_prompt_length` via `encoder_attention_mask.sum(dim=1).max().int().item()`, which drives shapes. Needs representative host inputs or supplied lengths | 12 |
| 40 | **Device and config object stubs.** `MeshDevice.shape/arch/compute_with_storage_grid_size/create_submeshes`, `CoreGrid`/`CoreCoord`/`CoreRangeSet`, `SDPAProgramConfig`, `MemoryConfig`, compute-kernel configs, `SubDevice`, `create_global_semaphore` — and `get_matmul_config`'s assertions must be satisfiable from fake shapes | **closed in 6** |
| 41 | **Pipeline construction touches the device before any forward.** `CCLManager.__init__` calls `_init_subdevice()` and `_init_semaphores()` (many `create_global_semaphore`) and `synchronize_device`; pipelines also build persistent buffers and prepare weights at init | **closed in 6** |
| 42 | **Shim/ttnn divergence over time.** The shim is a second implementation of ttnn's shape/layout semantics and will rot | 11 |
| 44 | **Source attribution points at shared library code.** Spike findings landed on `layers/linear.py:250` (the AGMM call site inside `ColParallelLinear`) rather than `attention_ltx.py:428` (`to_qkv`). Both are true; only the second is actionable. Record a short caller stack (2-3 tt_dit frames), not one line | **closed in 6** |
| 43 | **Readback boundaries end the graph.** Everything after `to_torch(get_device_tensors(...))` (`transformer_ltx.py:1068`, `bwe_ltx.py:85`, `mel_decoder_ltx.py:483`) is host code on fake data; and no kernel-level constraint (program-config asserts, L1 fit, hangs) is visible in a dry run | **partly closed in 10** — the shim marks readbacks as boundaries and `Graph.segments()` splits there; using segments to link stages is the rest |

## Phases

Estimates are for one engineer. Only phase 11 needs device access.

### Phase 6 — Dry-run shim core · **done** · closes 37, 40, 41, 44; dissolves 3–9

Built as [`dryrun/`](dryrun/); the spike it grew from is deleted, its findings
kept. What shipped:

- A `ttnn` stand-in installed by import shadowing, not by editing model code:
  metadata `Tensor` (per-device `.shape`, logical shape underneath), `MeshDevice`,
  mesh mappers/composers, config objects, semaphores, subdevices,
  `synchronize_device` and trace capture as no-ops. `install()` refuses to
  displace a real `ttnn`, `uninstall()` restores `sys.modules`, and
  `assert_installed()` guards graph emission (the "import shadowing is fragile"
  risk).
- Ops emit IR nodes as a side effect of returning metadata; distribution is
  recorded at creation, so entry placements are known rather than declared.
- Weights load through the real `Parameter.load_torch_tensor` on
  `torch.empty(..., device='meta')` (37) rather than by assigning `_data`. That
  makes `utils/tensor.from_torch` build the real mesh mapper and
  `Parameter._check_data` compare the shim's per-device shape against tt_dit's own
  `local_shape` on every parameter — a free check on the shard math (36). It
  required `create_mesh_mapper` / `PlacementShard` / `MeshMapperConfig` and
  `ttnn.Layout` to be real objects: a stub mapper reads as "replicated" and
  silently loses a tensor's distribution.
- Real torch is preferred and a metadata-only stand-in is the fallback, so a
  device-free CI job needs no torch install. Whatever was substituted is printed
  with every run. Verified both ways: on an interpreter with torch, loguru,
  safetensors, numpy and pytest the substitution list is empty apart from the
  Python-version hook, the real `models.common.utility_functions` is used, and the
  graph is identical to the fallback run's.
- `CCLManager.__init__` (subdevices, semaphores, persistent buffers) and
  `get_matmul_config` run unmodified against the shim (41, 40).
- A short caller stack per node (44): findings lead with
  `attention_ltx.py:428` and name `layers/linear.py:250` beneath it, with
  `Module.__call__` dispatch frames filtered out.
- The `unregistered` node kind, with provenance propagated automatically in
  `ApplyCtx.define`, so findings downstream of an op with no semantics are
  withheld and reported as registrations to make. `ditcheck ops --missing` /
  `--check` are the user-facing end.
- In-place ops (`copy`, `multiply_`) SSA-ified by repointing the caller's tensor
  at a fresh symbol, which closes 6 for those calls rather than deferring it.
- **Acceptance — met.** For both shipped Blackhole configs, from source: the real
  forward runs (212 nodes / 341 symbols on 4×8, 206 / 343 on 2×4), no
  unregistered ops, no analyzer diagnostics, the collectives match
  `examples/ltx.py` as an identical multiset (31 vs 31 on Ring, 25 vs 25 on
  Linear), and the findings match (6 provable `duplicate_gather`, 128.7
  GiB/forward; 0 on Linear). `examples/ltx.py` stays as the oracle and the drift
  test: `ditcheck dryrun --check-oracle`, asserted by `tests/test_dryrun.py`.
- **Not in phase 6, by design:** pipeline-level construction over submeshes and
  stages is phase 10, and the branch/shape sweep is phase 12. What exists is a
  target registry (`dryrun/targets.py`) holding the little a dry run cannot read
  off the source — mesh shape, axis roles, activation shapes, checkpoint-derived
  flags — which is also what phase 12 sweeps.
- **Prerequisites the spike surfaced, still true:** the repo needs Python ≥ 3.10
  (PEP 604 unions in evaluated annotations, `types.NoneType`) — `hostenv.py`
  works around it with an import hook on older interpreters — and
  `models.common.utility_functions` pulls in numpy *and pytest* for the sake of
  `is_blackhole`, worth splitting upstream. The dry run now imports the real module
  when it can and stubs it otherwise, so this is a soft dependency; two things had
  to be right for that, both of them traps for anything else added to `hostenv.py`:
  the probe must happen *after* the shim is installed (or tt_dit's module-level
  `import ttnn` pulls in real ttnn and `install()` refuses to run), and
  `ttnn.get_arch_name` must be implemented, because `is_blackhole()` reads it and
  the generic stub would answer "not blackhole" for every mesh while the model keys
  chunk sizes and program configs off it.

### Phase 7 — Shape and layout fidelity · **7a done; 7b on 2×4, Ring blocked** · closes 10, 11, 12, 13, 36, 38

The load-bearing wall: shapes decide branches (36). The spike's two bugs are
exactly this phase's content — treating `num_heads_per_device=1` (the no-split
default) as a head split, and reusing a fused weight's symbol for a chunked AGMM.
Between them they produced 15 spurious findings next to the 6 real ones. What
shipped:

- **Tile padding (10).** `shape` and `padded_shape` are distinct; region and
  demand math stay on logical extents, and byte/cost math reads a tile-padded
  volume (`region.padded_volume`) because the fabric moves whole 32×32 tiles.
- **Exact shard division (11).** One canonical `region.shard_chunk_size`
  reproduces ttnn's `xtensor/partition.cpp` chunk rule — `ceil(extent/n)` to the
  leading devices, remainder to the last — used by both the region shard and the
  dry-run per-device shape. The uneven case follows ttnn instead of crashing, and
  the empty-device case that ttnn TT_FATALs on a 2D mesh is refused with that reason.
- **Block-float bytes (13).** `ir.elem_bytes_for` carries the +1/16-byte shared
  exponent overhead: `bfp8_b`=1.0625, `bfp4_b`=0.5625, so a block-float gather is
  no longer undercounted by ~6%.
- **Checkpoint-derived flags (38).** `dryrun/checkpoint.py` reads keys and shapes
  from a metadata-only source — a real `.safetensors` header (no tensor bytes), an
  index JSON, or a declared manifest — and derives `has_gate` /
  `cross_attention_adaln` by tt_dit's own rule, reporting which source it used.
  `apply_gated_attention` is no longer a hardcoded boolean.
- **Weight preprocessing / chunked weights (12).** Established that
  `Parameter.total_shape` + `_check_data` already validate every per-device weight
  shape against tt_dit's own `local_shape`, and that `_prepare_torch_state`'s
  swiglu/interleave reorders are shape-preserving (`prepare_for_fused_swiglu`:
  `[..,2N]→[..,2N]`), so the shapes are already correct and checked. `_weight_chunk`
  now splits fused-weight columns by ttnn's `torch.chunk` rule (ceil, not floor).
  The residual — a fused weight's column *ordering* under `_interleave_heads` — is
  not consumed by the analyzer (it reasons about shape and value identity, not
  column order) and is left to on-device conformance.
- **Acceptance — met on 2×4, blocked on Ring.** [`conform.py`](conform.py) builds
  representative LTX tensors on a real 2×4 Blackhole mesh via the same
  `create_mesh_mapper` path tt_dit uses, reads back each device's shard and diffs
  its logical and tile-padded shape against the shim: `video_act`, `audio_act`,
  the TP-sharded fused `qkv_weight`, `rope_cos` and the tile-padding probe all
  match ttnn exactly; the deliberately-uneven probe is refused by ttnn under tile
  layout with the shim's chunk size as ttnn's own expected value. The 4×8 Ring
  finding — the one with redundancy — needs a 32-chip Galaxy to record, so its
  per-device-shape corroboration stays open.

**Test count: 43 offline (24 analyzer + 19 dry run), no device or pytest needed,
plus `conform.py` on a device.**

### Phase 8 — Op coverage, one registration per op (3–4 weeks) · **in progress** · closes 2, 14, 15, 16, 17, 18

**Landed so far — a second block from source.** The SD3.5-large joint
`TransformerBlock` (`blocks/transformer_block.py` + `blocks/attention.py`) now runs
under the shim as `ditcheck dryrun sd35_block --preset bh_2x4` and matches
`examples/sd35.py` on all four oracle criteria: 0 unregistered ops, 10 collectives
identical to the oracle, 0 findings (every collective load-bearing). Getting there
took exactly the coverage phase 8 is about: a real `split_query_key_value_and_split_heads`
shim (the fused QKV split — LTX uses chunked `to_qkv` and never exercised it),
a new `recorder.emit_multi` for its three outputs, and the `dit_rms_norm_unary_fused`
spec (the q/k RMSNorm, mapped to the local `layernorm` semantics). It also runs the
38→40 padded-head case through the phase-7a shard/shape math on a real block.
Asserted by `tests/test_dryrun.py::test_sd35_block_matches_oracle`.

**Landed — the conv/VAE family, a third block.** `ditcheck dryrun sd35_vae_resnet`
builds the real SD3.5 VAE `ResnetBlock` (`models/vae/vae_sd35.py`) and runs clean:
0 unregistered ops, 2 load-bearing `vae_all_gather`s, 0 findings, **0 diagnostics**.
SD3.5's VAE is single-axis (`VAEParallelConfig.tensor_parallel`), so it maps onto
the one-axis `Dist` and sidesteps the phase-10 multi-mesh work the LTX VAE's
`VaeHWParallelConfig` needs. What it took: a `conv2d` spec (channel-parallel, like a
column-parallel matmul — output channels fractured where the weight's out axis is;
blocker 14), a `group_norm` spec (device-local: whole groups per device, spatial
unsharded → the local `pointwise` semantics), the `ttnn.operations.normalization`
core-grid helper stubs, ROW_MAJOR weight-layout threading through `from_torch`, and
— the analytically real part — **the VAE stops tainting (blocker 17)**: the
`[B,H,W,C] ↔ [B,1,H*W,C]` reshape now preserves a shard on the kept channel axis
instead of degrading to replicated+taint, which is what turned a spurious
`suspicious unused_gather` into a correct clean run. Asserted by
`test_sd35_vae_resnet_is_clean` and `test_reshape_preserves_a_shard_on_a_kept_trailing_axis`.
The conv/group-norm *shapes* are the shim's belief until on-device conformance
(phase 11) — there is no independent VAE oracle yet.

**One registration per op — done for the generic families, bounded by design.**
The duplication that was real — the unary / binary / passthrough ttnn-name lists
appearing in *both* `semantics.py` (analyzer aliases) and `dryrun/ops.py` (shim
dispatch) — is now a single `semantics.GENERIC_OPS` table: each entry declares a
call's canonical op *and* its shim shape-rule once, in the analyzer layer that is
the source of truth, and the shim **generates** its dispatch from it. Adding a
pointwise / view-like op is now one line in one file.

The merge stops there deliberately, and the reason is architectural, not
unfinished work: the shim already emits *canonical* op names, so the analyzer must
own its registry independently — a `dump`ed graph is `analyze`d with no shim in
the process (the phase-13 CI path). So the analyzer cannot import the shim, only
the reverse. Bespoke comm/compute ops (matmul, collectives, conv2d, group_norm,
the fused kernels, `from_torch`, reshape) therefore keep an analyzer spec *and* a
shim rule in their two layers — which matches the plan's intent that "only new
communication semantics should need real thought." The `ops --missing --stub`
generator (phase 8) already emits both halves for a new such op.

**Remaining:**

- ~~Declare fused-kernel internal stages as **data**~~ **done** — `dryrun/fused.py`
  holds a `FUSED_KERNELS` table (call → hidden collective, stage order, chunked,
  epilogue); AGMM and MMRS are now bound to shared builders from that table, the
  set is inspectable via `ditcheck ops --fused`, and `ops --missing` flags an
  unregistered op whose name looks like a collective-hiding kernel (blocker 18).
- Tier-2 specs for what actually appears in DiT forwards: `mesh_partition`, `pad`,
  `copy`, `repeat`, `lerp`, ~~`embedding`~~ **done** (token lookup, hidden-parallel
  like a col-parallel matmul; the id tensor is replicated), `minimal_matmul_split`,
  `dit_minimal_matmul_addcmul_fused`, `dit_rms_norm_unary_fused`,
  `exp_ring_joint_sdpa`, `rotary_embedding_hf`.
- **MiniMax-H3 DiT** (`kevinmi/minimax-h3-t2va`, the t2va video+audio transformer,
  TP **and** sequence-parallel, ring attention) — now dry-runs end to end with **0
  unregistered ops**. Getting there: (a) fixed the reshape-tracking blocker that
  crashed the run — registered `ttnn.slice` (its bounds are on the *local* view, so
  a bound on a sharded axis lifts to logical by the mesh factor; the AdaLN
  modulation slices one param block out of a packed `[param | h]` feature axis), and
  hardened the general `reshape` so a rank reduction can no longer strand a shard on
  a dropped axis; (b) registered `ttnn.mesh_partition` (scatter, the dual of
  all_gather — MiniMax-H3 fractures the assembled packed sequence onto SP with it)
  and `ttnn.cos`/`ttnn.sin`; (c) fixed the `embedding` op to carry the **indices'**
  row-sharding as well as the weight's hidden-sharding (SP-fractured AdaLN indices ×
  TP-fractured table). **Still open (residual blocker 17):** the modulation *fold*
  reshape `[1,1,T, M*P*H] -> [1,1, T*M, P*H]` (a feature sub-axis folded into rows)
  is now **tracked** (`_reshape_local_dist`): ttnn reshapes each device's *local*
  block, so the shard moves onto the output axis carrying its factor (matched by
  trailing-element count) and each device keeps its full shard — not a global
  element permutation. `OPAQUE_RESHAPE` and `LAYOUT_MISMATCH` are now **0**; only the
  expected `K_COVERAGE` taints on batched-attention matmuls remain. With the analysis
  clean, the findings persisted unchanged — confirming they were never caused by the
  reshape/layout gaps. They reflect a **real structural property**: the packed
  sequence is `[text | audio | video]` but only video+audio are outputs, so text
  rows' *output-side* compute (attention output, MLP) is unconsumed (text is still
  needed as K/V), and the SP-replicated token refiner is consumed only where text
  lands. Plausibly real optimizations (skip FFN/output on conditioning-only tokens),
  but they still want device conformance before acting — the "delete the collective"
  fix wording is wrong (it's structural).
- **MiniMax-H3 visual VAE decoder** (`cglagovich/minimax-h3`, `MiniMaxH3ViTDecoder3d`
  — a 36-layer ViT, not a conv stack) — dry-runs with **0 unregistered** after four
  pure name-aliases of families already covered: `rms_norm`/`layer_norm` → the
  `layernorm` emitter, top-level `alt_complex_rotate90` → pointwise (the rule already
  existed under `experimental.*`), `nlp_concat_heads` → `concatenate_heads`
  (merge_heads). No new op. **No findings from the decoder itself:** it is
  replicated-only (0 collectives even on a 2-device mesh) — the VAE's parallelism
  lives in the `data_parallel` / `hw_parallel` wrappers, which is where collective
  findings would surface.
- **MiniMax-H3 conv VAE encoder — the halo frontier** (`MiniMaxH3Encoder3d`,
  H/W-sharded). The first genuinely-*new* semantics in the H3 effort, now implemented:
  - **`neighbor_pad_async` (halo exchange)** — a new spatial collective. Each device
    gains `pad_left`/`pad_right` border rows from its neighbours along each sharded
    axis. Modelled in the grown frame (logical dim += pad_left+pad_right): device `i`
    holds `[i*S, pad_left+(i+1)*S+pad_right]`, so adjacent devices *overlap* by the
    halo width; backward demand maps each device's border to the neighbour that owns
    it. Registered `is_collective`, so a dead/duplicate halo is flaggable.
  - **`conv3d`** — valid 3-D conv (padding pre-applied by the halo/reflect/causal
    pads); spatial-parallel, each device convs its halo'd shard into a clean output
    shard, so the sharded result matches the unsharded one.
  - Plus `reduce_sum` (partial sum over a sharded axis — group-norm stats), `rsqrt`,
    and a **`concat` fix** (pick the richest operand as primary, so a per-device
    auxiliary built from `x.shape` — e.g. the causal zero-frames — can't shrink a
    sharded operand). Two more shim fixes cleared the rest of the forward: `creation`
    now honours the `layout` kwarg (`ttnn.zeros(layout=ROW_MAJOR)` was silently TILE, so
    the causal concat handed a later conv a TILE tensor), and **both** the shim *and* the
    analyzer `concat` inherit the **richest** operand's dist/layout, not input 0's.
  - The encoder now **fully dry-runs — 361 nodes, 0 unregistered** — and analyses:
    **16 collectives necessary, 1 flagged**. Verifying the first result caught a real
    bug: the analyzer concat lost the shard, inventing **10 phantom `unused_gather`
    findings (39.5 MiB that don't exist)** — gone once the concat inherited the sharded
    operand. Then the participant-frame fix below dropped a further 6 false halo
    findings (7 → 1); the surviving one is a genuinely asymmetric downsampler halo.
  - **Halo demand is now exact in 2-D.** `neighbor_pad`'s backward demand splits as a
    *product* over the padded axes, routing the corner (past the border on both H and W)
    to the **diagonal** neighbour that owns it — no device is ever demanded for data
    outside the shard it holds. This makes the `participant_shrink` accounting trustworthy
    on 2-D-haloed sites; the confidence tier stays MEDIUM/likely (participant_shrink is an
    opportunity requiring a code change, not a provable dead collective).
  - **Participant-frame fix.** The sender/needer rules compared `needed` (output frame)
    against `local` (input frame) — fine for shape-preserving collectives, but a halo
    grows the frame by `pad_left`, so a *symmetric* halo (both devices read each other's
    border) read as one-sided and was wrongly flagged. `collect_views` now maps the
    halo's `needed`/`gained` back into the input frame. This dropped the VAE encoder's
    halo findings 7 → 1 (6 were the bug) and is what the **audio T-parallel scout**
    surfaced — it had reported **382 `participant_shrink` / 884 phantom MiB**, essentially
    all of them symmetric halos. Regression-tested (symmetric → necessary, asymmetric →
    shrink). Suite 69 passed.
- **MiniMax-H3 audio VAE decoder** (`cglagovich/minimax-h3`, the shared LTX **BigVGAN
  vocoder** `vocoder_ltx.py`) — dry-runs **fully clean, 3783 nodes, 0 unregistered**.
  Most of it is `conv3d` with `(k,1,1)` kernels (already covered — a payoff from the
  video work), so it needed only: **`ttnn.conv1d`** (depthwise 1-D conv on a `(B,L,1,C)`
  tensor; a `conv2d` with `W=1`, returns ttnn's `(out, out_length, (weight, bias))`
  tuple — the missing tuple was what tripped the vocoder's 3-way unpack) and
  **`snake_beta`** (the BigVGAN activation, a pointwise unary). Both generic — `conv1d`
  benefits LTX audio too (shared vocoder). **T-parallel scouted** (time sharded over a
  mesh axis): it runs (4544 nodes, 4 segments) and surfaced the participant-frame bug
  above, but its findings are **not yet trustworthy** — `conv1d` reuses conv2d's
  channel-parallel spec and so **drops the time-shard** (replicates), which then reads the
  downstream T-halos as redundant. Fixing it means making `conv1d` spatial-parallel
  (preserve the input shard, like `conv3d`), which then collides with the vocoder's
  per-device local-scale reshapes under sharding — a deeper follow-up, not the frame bug.
- **Grouped-query attention** — `nlp_create_qkv_heads` now branches on its call
  shape: `num_kv_heads == 0` (LTX / Ideogram / Wan, `out, _, _ = …`) is the plain
  single-tensor head split; `num_kv_heads > 0` (Qwen3-VL / Gemma / **MiniMax-H3**,
  `q, k, v = …`) is the fused GQA split into q (`heads`) and grouped k, v
  (`kv_heads`). The analyzer's `split_qkv_heads` spec maps per-device columns to
  heads **per output**, so q and the narrower k/v get correct regions on both the
  forward (availability) and backward (demand) passes.
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

### Phase 9 — Scale (1–2 weeks) · closes 20, 21, 26 · 27 optional

Runs *with* phase 10: once whole pipelines produce 48-layer graphs, this keeps
them **readable and tractable for a human** (the CI-diffing motivation for stable
IDs is dropped). Cheaper than in revision 1: the shim knows the Python call stack
and loop index, so block boundaries and `calls` come from the run, not inference.

- Roll repeated blocks up to one instance + `calls`, keeping the first and last
  instances intact so boundary effects aren't hidden (readability).
- Keep runtime/memory tractable: liveness-pruned, copy-on-write state, snapshot
  only at collectives (a naive 48-layer graph extrapolates to ~20 min / ~20 GB).
- Roll findings up by `(rule, source location)`, leading with the outermost model
  frame from the caller stack recorded in phase 6 (44).
- Deterministic node IDs from `(op, source location, occurrence index)` — now
  **optional** (only needed to diff two runs by hand), not a gate.
- **Acceptance:** a full 48-layer dry run analyzes in a reasonable time/footprint,
  with findings matching the rolled-up single-block result.

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

- **11a — per-op conformance harness · done (2×4).** `conform.py` builds inputs on
  the real mesh and diffs the shim's shape/layout/dist against ttnn: 12/12 green —
  distribution/tile-padding for LTX + SD3.5 (incl. the 38→40 padded qkv weight,
  conv and group-norm weights) and the output shapes of `matmul`, the fused-QKV
  split, and `concatenate_heads`. Still to cover: collectives and fused kernels
  (need CCL setup) and conv2d/group_norm compute (program-config setup) — their
  shape math (shard division, tile padding, channel sharding) is already covered
  by the distribution cases, but the kernels themselves aren't run yet.
- **11b — collective conformance + whole-block log · done (2×4).**
  `conform_collectives.py` runs the block's actual collectives through the real
  `CCLManager` on the fabric — all-gather (tp & sp) and reduce-scatter — and their
  per-device output shapes match the shim (3/3). `conform_block.py` then runs the
  **entire SD3.5 block forward on hardware** (random weights, since values don't
  affect which collectives fire) with a collective logger, and diffs the log
  against the dry run: all-gather(tp)=6, all-gather(sp)=2 (the ring-SDPA K/V
  gathers, reconciled via `dryrun/fused.py`), reduce-scatter(tp)=2 — **counts
  match**, and the ordered log carries real per-device shapes and source lines.
  The diff is on (op, mesh-axis) counts plus the ordered log for inspection; a
  strict per-collective shape/source diff (logical↔per-device reconciliation) is a
  refinement. The 4×8 Ring log needs a Galaxy.
- **11d — conform a *finding*, not just an op · done (2×4).** 11a/11b conform that the
  shim's *shapes and collective counts* match ttnn; this conforms that a *reported
  redundancy is physically real*. `conform_encoder.py` builds the real H3 text encoder
  (the structure that fixes the sharding — hidden 5120 / 40 heads / head_dim 128 / 8 KV /
  mrope [16,24,24], TP=4 on axis1; vocab+MLP width shrunk since they don't change the
  distribution) on the 2×4, loads random weights, runs a replicated forward, and reads
  back all 8 device shards. The `replicated_stage` finding claims the output is replicated
  across the SP axis; the harness checks the physical fact it rests on — for each TP column,
  SP-row 0 vs SP-row 1 — and finds them **bit-for-bit identical (max\|Δ\|=0), non-degenerate
  output (std 0.99)**. So the pipeline provably composes both SP rows and discards one; the
  "run the encoder on a submesh" fix now carries device proof. The property is per-token /
  per-layer, so it holds at production seq and depth (the *aggregate* MiB scales; the fact
  doesn't). The 4×8 variant (7 of 8 rows) still needs a Galaxy.
- **11c — soundness gates · parked (needs new capture + a memory model).** Assessed
  and deliberately deferred rather than faked:
  - *Buffer liveness* (suppress a `duplicate_gather` CSE when the earlier result's
    persistent ping-pong slot is reused before the candidate) needs **buffer
    identity** (blocker 29). Modelling the ping-pong allocation in the shim is
    possible, but a soundness verdict built on shim-modelled buffers is itself
    "the shim believes" — the device half that would confirm it (recording real
    buffer/semaphore identity) is the missing capture. Until then the honest
    verdict degrades to "value identity confirmed via SSA; physical slot reuse
    unobserved → needs review", which the global trust banner already says.
  - *Barrier/sync intent* is cleanly implementable (the shim sees `use_barrier`),
    but the block's gathers take the persistent-buffer, no-barrier path, so it does
    **not fire on the LTX/SD3.5 findings** — only the VAE class.
  - *Live-bytes / memory feasibility* needs an L1/DRAM residency model (blocker 30)
    that does not exist; first-order it emits "unknown".
- **Acceptance (partial, honestly):** per-op + collective conformance green on 2×4
  (11a/11b); the buffer-liveness and memory verdicts remain "needs review / not
  modelled", surfaced by the per-graph trust banner rather than a fake per-finding
  gate. Full 11c awaits buffer-identity capture and a memory model.

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

### ~~Phase 13 — Workflow / CI~~ · **mostly out of scope** (hand-run tool)

The device-free **CI job and golden baselines are dropped** — ditcheck is run by an
engineer when they want it, so there is no build to keep green (blocker 35 gone,
27 optional). Two pieces survive as conveniences for hand runs, and are folded
elsewhere rather than kept as a phase of their own:

- **`ditcheck dryrun --pipeline …` entry point** (blocker 34) — point it at a real
  pipeline module instead of a hand-curated block target. Wanted for manual
  whole-pipeline runs, so it rides with phase 10.
- **Optional cost-based ranking** (blocker 33) — join a perf CSV to rank findings
  by measured time, not just bytes. A "look here first" aid, explicitly *not* a
  latency model (contention and comm/compute overlap stay out of scope). Add if and
  when a human wants it.

The old acceptance criterion ("a provable redundancy fails CI on a hardware-free
runner") no longer applies; the equivalent manual check is just
`ditcheck dryrun <target> | analyze --fail-on provable`, run by hand.

## Ordering

Phases 6–8 (build the tool, make it run on real blocks) are done. As a **hand-run
tool**, the remaining order is driven by trust, then reach — not by a critical path
to CI.

```
6 (shim) ══► 7 (shape fidelity) ══► 8 (op coverage) ══►  11 (conformance, on device)  ← the trust bottleneck
   done       done; 2×4 conformed     core done      │
                                                      ├─► 10 (whole pipeline / multi-mesh)  ← reach; unblocks LTX VAE
                                                      │       └─► 9 (rollup + perf, so pipelines stay readable)
                                                      └─► 12 (branch/shape matrix)          ← optional convenience
```

**Why 11 comes first now.** With no CI gate to keep green, the bottleneck is
whether an engineer can *believe* a finding enough to spend kernel-engineer time on
it. Today every dry-run report says "the shim believes"; only the LTX 2×4 *shapes*
are device-corroborated (phase 7b's `conform.py`, a per-tensor check — not yet the
per-op collective-log diff). Phase 11 grows that into real per-op conformance plus
the soundness gates (buffer liveness, barrier intent), which is what converts belief
into confirmation. It needs device time, and the 2×4 Loudbox is enough to start; the
one finding with real redundancy — the 6 LTX 4×8 Ring duplicate gathers — stays
Galaxy-blocked, and until a Galaxy is available that finding reads as "the shim
believes", corroborated only by an independent hand-written oracle.

**Then 10 for reach** (analyze whole `encoder → DiT → VAE` pipelines, not curated
blocks; also the only way to see redundancy that spans stage boundaries, and the
thing that unblocks the LTX VAE), **with 9 alongside it** to roll a 48-layer graph
up to a readable report and keep runtime/memory tractable. **12** (config sweep,
naming unexercised branches) is an optional convenience.

Two cheap checks already push against the top risk, and both are free to keep
running: loading weights through `Parameter.load_torch_tensor` makes tt_dit's own
`local_shape` disagree loudly with the shim's per-device shape, and `--check-oracle`
diffs the derived graph against the hand-written one on every test run.

## Risks

| Risk | Mitigation |
|---|---|
| Shim shape math diverges from ttnn and flips a branch (36) — the worst failure mode, because it produces confident wrong findings, **observed in the spike: 2 bugs, 15 spurious findings** | Per-op conformance on a device (11); phase 7 acceptance is a per-device-shape diff against a real run (`conform.py`, green on 2×4); keep the hand-written `examples/` graphs as regression oracles rather than deleting them |
| The shim rots as ttnn adds or changes ops (42) | `ops --check` makes coverage debt visible; conformance failures name the op; fused-kernel behaviour lives in a data table |
| A weightless or input-free dry run silently changes the graph (37, 38, 39) | Meta-tensor weights so `_data` is non-None (**done in 6**, through the real `Parameter` load path); checkpoint key lists (7); the coverage matrix reports which branches were never exercised (12) |
| Import shadowing `ttnn` is fragile or leaks into real runs | **Done in phase 6:** the shim installs only under `ditcheck dryrun`, `install()` refuses to displace an already-imported real `ttnn`, `uninstall()` restores `sys.modules`, `assert_installed()` runs before any graph is emitted, and the oracle tests run each config in a subprocess |
| Findings scale faster than anyone reads them | Rollup by source location plus top-N ranking; the report is a queue, not a dump |

## Out of scope

Automated rewrites (plan §choice 4: proofs before auto-fixes), training/backward
graphs, non-DiT models, and a predictive performance model. Kernel-level
validation stays with the device — the shim answers "which collectives exist and
which are redundant", not "will this program fit and run".
