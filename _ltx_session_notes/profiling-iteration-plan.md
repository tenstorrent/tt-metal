# Plan: fast profiling iteration on big models via module decomposition

Status: DESIGN ONLY — for an implementing agent. Target repo: `/home/smarton/tt-metal`.
**Scope: general (model-agnostic) method + harness infra** to make profiling a big model's hot
loop take *minutes, not tens of minutes*, by profiling a representative **piece** instead of the
whole model — the way LTX audio decode was broken out from the 22B AV pipeline. LTX is the worked
example; the harness is reusable across tt_dit models (and beyond).

## What this attacks (measured) — it is NOT the compute
Per-iteration wall on LTX today is dominated by **per-process startup**, not compute:
- Weight load (22B for the full AV pipeline).
- **Cold mesh-workload assembly: 496s on 4x8** (cross-chip program-barrier setup; scales with
  chip count AND program count), vs 5.6s on a 1x4 submesh.
- ~10-min video warmup for the full DiT path.
- The actual warm forward is **sub-second**.
So iteration time ≈ load + assembly + warmup. Shrinking *what you build and run* per iteration is
the lever — not making the kernels faster.

## The general method: two orthogonal decompositions
Both reduce the per-iteration workload to a representative piece. Both rest on one primitive:
**build only the target submodule, feed it an input of the correct shape/dtype/layout, profile it.**

### A. Depth decomposition — profile 1 of N repeated layers (the big transformer lever)
A DiT/transformer is N identical blocks on a residual stream. Per-op cost (compute AND the
per-block collective) is the same for block 3 as block 30. So **profile ONE block, multiply by
N.** Build only that block's weights (~1/N of the model), feed a residual-stream tensor of the
real per-block sharded shape, run on the **full mesh**.
- Why full mesh: the block's collectives (RingJointSDPA, AllGather) run over the real sharded
  activation → **the per-op communication cost is real** (see "what does NOT decompose" below).
- Why it's fast anyway: assembly scales with *program count*; one block ≪ N blocks + VAE +
  vocoder, so assembly + load both collapse even though the mesh stays full.

### B. Stage decomposition — profile a pipeline sub-graph via its tensor interface
A pipeline is stages with clean tensor in/out (text-enc → DiT → VAE-decode → vocoder). Profile one
stage standalone by **building only that stage** and feeding a tensor at its input boundary. This
is exactly the LTX audio breakout: `decode_audio` is torch-in/torch-out, so it runs with **no 22B
transformer**, on a **1x4 submesh** (assembly 496s→5.6s). Sub-stages nest: a single vocoder
AMPBlock was profiled by intercepting `resblocks[0]` inside a real `Vocoder.forward`.

## Key simplification: for PERF, you don't need real activations
Device-op performance depends on **shape, dtype, layout, mesh, sharding** — NOT tensor *values*.
So a **synthetic input of the correct spec** profiles identically to a real captured activation.
=> The perf-iteration harness needs an **input spec**, not a capture pass. (Real captured
activations — e.g. `girl_audio_latent.npy` via `LTX_DUMP_AUDIO_LATENT` — are only needed for
*accuracy/PCC* gates, which are a separate concern from perf iteration.)
Caveat: this breaks for **data-dependent control flow / dynamic shapes / MoE routing imbalance**,
where values change the op mix — those pieces need a representative real input (capture).

## What does NOT decompose — be honest about the limits
- **Communication cost does not shrink with the mesh.** A collective's cost depends on the full
  topology and the global tensor; a 1-chip/small-mesh standalone block gives a *wrong* comm
  number (this is the audio T-shard U-shape: fewer chips ≠ proportional cost). So: keep the
  **full mesh** for comm-bound pieces (depth decomposition does — that's its point); only drop to
  a submesh for **comm-free** pieces or when you are deliberately excluding comm.
- **First/last layers may be special** (different seq len, extra norm, embedding) — profile them
  separately, don't fold into the "1 representative block × N".
- **Cross-piece effects** (pipelining, block-to-block overlap) aren't captured — negligible for
  per-op profiling, real for end-to-end latency. Use the full run for the final E2E number.
- Representativeness is a **claim to verify**, not assume: once per model, confirm the
  piece-profile's per-op numbers match the same ops in a full-run profile (within tolerance).

## Infra deliverables (general, reusable)
1. **Module harness utility** — `profile_module(model, submodule_path, input_spec, mesh, warm_reps)`:
   instantiate just `submodule_path` with partial/lazy weight load, place on the chosen mesh, feed
   synthetic (or supplied) inputs matching `input_spec`, run ≥1 warmup + N timed, hand off to the
   profiler. One implementation; per-model config is just (path, spec, mesh).
2. **Input-spec derivation** — capture each target submodule's input shape/dtype/layout/sharding
   from ONE reference run (a hook that records boundary tensor specs), emit a small spec file the
   harness replays. Avoids hand-transcribing shapes; makes synthetic inputs correct.
3. **Optional activation dump** — generalize `LTX_DUMP_AUDIO_LATENT` to dump real boundary tensors
   for the pieces that need real content (accuracy gates, or data-dependent perf).
4. **Mesh-choice helper** — given a piece, pick full mesh (comm present) vs submesh (comm-free);
   warn when a submesh would distort comm.
5. **Compose with the profiler scope-control** (see `tracy-nsight-counters-plan.md` Phase 2.5):
   each piece-profile should ALSO be bounded (N cores/op-grid, per-op NoC aggregate) and
   repeatable. Small workload × small capture scope = fast, complete, repeatable iteration.

## LTX worked examples
- **Audio (stage decomposition): DONE** — `decode_audio` on 1x4 submesh + single-AMPBlock
  capture. Assembly 496s→5.6s; this is the template.
- **DiT/video (depth decomposition): the missing high-value lever.** A single-DiT-block harness
  on the full 4x8 mesh, fed a synthetic residual-stream tensor of the real per-block shape, would
  cut video profiling iteration from ~15–20 min (full 22B + 10-min warmup + 496s assembly) to
  minutes, while keeping the real per-op compute AND collective costs. This is what makes the
  video optimization loop tolerable. Build this first for LTX.

## Relationship to the nsight-counters plan — complementary, not overlapping
- **This doc** shrinks the *workload* (profile a representative piece).
- **`tracy-nsight-counters-plan.md`** makes each *capture* complete (full counters), bounded, and
  repeatable.
- Together: small piece × bounded capture = fast, correct, repeatable iteration. Neither replaces
  the other; the final E2E number still comes from a full-run profile.
- This doc owns **model/runtime decomposition**; the nsight doc owns the **profiler**. Model
  startup cost (load/assembly/warmup) is explicitly out of the nsight doc's scope and in this one.

## Sequencing
1. Module harness utility + input-spec derivation (the reusable core).
2. LTX DiT single-block depth-decomposition harness (the biggest concrete win) + verify its per-op
   numbers match a full-run DiT profile within tolerance (the representativeness gate).
3. Generalize mesh-choice + activation-dump; document the recipe in the `tt:profiler` skill so any
   model can break out a piece.
4. Apply to the next model that needs it (proves model-agnosticism).

## Open questions (resolve, don't guess)
- Can tt_dit submodules be instantiated standalone with partial weight load today, or does
  pipeline construction force loading the whole checkpoint? (Determines harness effort — check
  `LTXDistilledPipeline.create_pipeline` / `audio_only` path, which already does partial build.)
- Does a single DiT block on the full mesh actually reproduce the full-run block's per-op times +
  collective costs within tolerance? (The representativeness gate — must be measured, not assumed.)
- What's the assembly-time floor for a single block on full mesh (fixed per-mesh overhead vs
  per-program)? Determines the real iteration-time win.
