# Multi-step trace batching — one Metal trace for many denoise steps (#47465, path to 100 t/s, lever 10)

## What / why

The landed traced serving loop (`tt/traced_denoise.py::TracedDenoiseController`, verified in
`probe_traced_serving.py`) captures **one Metal trace per denoise step** and replays **N of them
per block** — `ttnn.execute_trace` is called once per step. That removed the ~137 ms/step
host-dispatch tax and cleared 30 t/s (58.3 t/s @12, 33.3 @24 at full 30L, bit-exact). But the
per-block cost still carries a **fixed dispatch term that scales with the step count**:

```
block(K) ≈ 0.275·K + 1.09 s      (K = denoise steps; 58.29 t/s @12, 33.28 @24)
100 t/s  ⇔ block ≤ 256/100 = 2.56 s
```

The ~1.09 s fixed term is the per-replay dispatch of the single-step trace paid `K` times per
block (plus the two per-block `synchronize_device` barriers and per-block refresh). Solving
`0.275·K + 1.09 ≤ 2.56` gives `K ≤ ~5`: with single-step replays, 100 t/s only holds at ~5 steps,
below a quality-safe budget.

**Multi-step trace batching** (`tt/traced_denoise.py::MultiStepTracedDenoiseController`,
`DG_DENOISE_TRACED_MULTISTEP=1`) captures a **window of `G` denoise steps into ONE Metal trace**.
The default `G = max_denoise_steps` captures the **whole fixed-K block in ONE capture + ONE
replay**, so a block does **`ceil(K/G)` replays instead of `K`** — removing `(K − ceil(K/G))`
per-replay dispatch bubbles per block. The per-step compute (the 30-layer sparse-MoE forward +
decision, the `0.275·K` slope) is unchanged; only the fixed dispatch term collapses. That lets
100 t/s hold at a **higher, quality-safe step budget** rather than only at ~5 steps.

`DG_DENOISE_MULTISTEP_GROUP=G` caps the window as a **trace-region memory knob** (below). It is
**opt-in and guarded**, takes precedence over `DG_DENOISE_TRACED` in
`_resolve_default_denoise_block_fn`, and — like the single-step traced path — supports the
**argmax (`gumbel_noise=None`) regime + contiguous cache** only and needs a large
`DG_TRACE_REGION_SIZE`. **No `models/demos/gemma4/` edits** — it composes over the existing
`DenoiseLogitsAdapter` trace-safe hooks and the `denoise_loop.py` step kernel exactly like the
single-step controller (it subclasses it and overrides only `_capture`).

## The equivalence argument (code inspection)

**Claim.** The multi-step trace commits the **byte-identical clean-argmax** that the `K`
single-step replays commit (and that the eager `run_fixed_denoise_steps` commits). Both paths
compute the same deterministic function of the same inputs; multi-step only changes how
`execute_trace` calls are grouped, not any committed decision.

The committed block is a pure function of `(block init canvas, per-step temperature `T[i]`,
per-step noise ids `noise[i]`, the frozen prompt prefix, and the self-conditioning signal chain)`.
Every one of those is identical between the two paths:

1. **Per-step temperature `T[i]`.** Baked exactly as single-step:
   `temperature_at_step(i, max_denoise_steps, t_start, t_end)` folded into the same `logits/T` and
   `token_entropy(..., temperature=T[i])` ops. The multi-step unroll bakes the same constant at
   the same step index.
2. **Per-step noise `noise[i]`.** The same persistent `noise_bufs[i]`, filled from block 0's
   stream at capture and overwritten in place per block by the **same `noise_tokens_fn(step)`
   order and count** as the eager `run_fixed_denoise_steps` — the seeded generator stream is
   untouched, so `noise[i]` matches step-for-step.
3. **Self-conditioning threads step-to-step ON DEVICE.** Both paths carry the self-cond signal in
   the adapter's persistent in-place `signal_buf` (re-zeroed per block by `reset_signal_buffer`, so
   step 0 reads zeros = the eager `condition(None)` = `post_norm(embed)` branch). Step `i` reads
   exactly what step `i−1` wrote. **Within a window** that read-after-write is a normal graph edge
   captured in one trace; **across a window boundary** it is the same in-place buffer carry the
   single-step path already uses between *every* step — an in-order-command-queue RAW dependency,
   verified bit-exact (`CROSSBLOCK_OK`). Baking `signal_buf`'s address before capture is required
   and done.
4. **The canvas threads step-to-step.** Within a window the canvas flows as a pure graph edge
   (step `i+1` consumes step `i`'s `next_canvas` intermediate — the proven trace-safe
   `run_fixed_denoise_steps` shape, which was written to be captured inside
   `begin_trace_capture`/`end_trace_capture`). Across a window boundary the window's end canvas is
   copied into the persistent `canvas_buf`, and the next window reads it — identical to the
   single-step per-step `ttnn.copy(next_canvas, canvas_buf)` carry.
5. **Canvas RoPE.** Refreshed per block OUTSIDE the trace into the constant-shape per-layer-type
   buffers; RoPE cos/sin depend only on absolute position, so this is bit-identical to the
   growing-slice path the eager reference uses. The trace's RoPE tensor addresses/shapes stay fixed
   across blocks.
6. **Per-block reset.** Before each block's replay: fresh init canvas copied into `canvas_buf`,
   `signal_buf` re-zeroed, `noise_bufs` refreshed, canvas RoPE refreshed. No stale cross-block
   state (step 0 never reads the signal buffer's prior content).

The commit is `clean argmax` of the final step; the argmax is stable after convergence, so running
the full fixed budget in one trace commits the same tokens as the same budget in `K` traces. The
only functional change is grouping the `execute_trace` calls.

### Why the earlier "whole-loop trace" divergence does NOT apply

A prior whole-loop trace attempt (`probe_traced_denoise_loop.py`) diverged (60.5% committed-argmax
match) with self-cond ON. That was **root-caused in perf_progress.md session 8 as a probe bug** —
a cross-replay buffer allocated **after** `begin_trace_capture`, overlapping trace scratch and
clobbered on every replay — **not** a self-cond race (ping-pong vs in-place signal is
bit-identical on device; the race was refuted). `MultiStepTracedDenoiseController` follows the
session-8 rule exactly: **every persistent cross-replay buffer — canvas, committed, signal, RoPE,
per-step noise — is allocated BEFORE `begin_trace_capture`.** That discipline is what makes the
multi-step (including whole-block) trace bit-exact, and it is the same discipline that makes the
already-verified single-step controller bit-exact.

## Trace-region memory

Each window trace records `G` steps of commands. The heavy per-step intermediates — the
`[1, 1, C, 262144]` logits (~134 MB bf16 at C=256) plus the 30-layer forward activations, which
dominate the ~168 MB single-step trace region — are **deallocated between steps inside the
capture**, so the window's **peak** intermediate footprint is ~1 step's, not `G`×. A whole-block
window is therefore expected to need roughly **one step's scratch + the K-step command stream**,
i.e. **not** `K×` the single-step trace (which reserves independent per-trace scratch `K` times, so
`K` single-step traces total ~`K`×168 MB → ~2 GB @12, ~4 GB @24). This is a hypothesis to confirm
on device via `bench_multistep_trace.py`; if a whole-block capture overflows the region the bench
prints `RESULT_MULTISTEP_BLOCKED trace_region_size=...`. Use `DG_DENOISE_MULTISTEP_GROUP` (or the
bench `--group`) to shrink the window and bound the per-trace region.

## Verify + bench (device — run only when QB2 is free)

`bench_multistep_trace.py` (DEVICE-OWNERSHIP-gated; not run at authoring time):

- **`--mode verify`** — eager `run_fixed_denoise_steps` vs single-step traced vs multi-step traced
  committed argmax at block-0 offset (twice, for replay determinism) and block-1 offset; all three
  must be byte-identical → `MULTISTEP_BITEXACT`. This is the primary correctness gate.
- **`--mode perf`** — per-block **denoise-replay** latency (no commit; the portion multi-step
  optimizes) for single-step vs multi-step, swept over `--steps-sweep`, with a least-squares
  `denoise(K) = a·K + b` fit for each (`RESULT_MULTISTEP_FIT`). Expect the multi-step **intercept
  `b` to collapse** (`K` replay dispatches → `ceil(K/G)`) while the **slope `a` stays** ~the
  single-step per-step compute.
- **`RESULT_MULTISTEP_PROJECTION`** — the largest step budget `K` at which each loop holds
  `block ≤ 2.56 s` (100 t/s), using `block(K) = denoise(K) + commit`. Pass `--commit-ms` (the
  additive batched-commit constant, unchanged by multi-step) or `--measure-commit` to time one real
  commit (mutates the cache, so it runs last).

```bash
# bit-exactness at 8 steps, whole-block capture, quick (fewer layers):
python bench_multistep_trace.py --mode verify --num-layers 6 --steps 8
# full 30L bench sweep (whole-block multi-step vs single-step) + 100 t/s projection:
python bench_multistep_trace.py --mode all --num-layers 30 --steps-sweep 8,12,16,20,24 --measure-commit
# grouped windows (bound trace-region memory to ~8 steps/trace):
DG_DENOISE_MULTISTEP_GROUP=8 python bench_multistep_trace.py --mode perf --num-layers 30 --steps-sweep 16,24,32
```

The perf phase measures **denoise only** on purpose: multi-step trace batching changes only the
denoise replay, and running two serving loops (single then multi) with commit on one model build
would double-commit into the same KV cache. Commit is an additive per-block constant folded back in
for the 100 t/s projection.

## Gating

Opt-in via `DG_DENOISE_TRACED_MULTISTEP=1` (`DG_DENOISE_MULTISTEP_GROUP=G` optional). Kept opt-in,
not default, for the same reasons as the single-step traced path: it requires the argmax regime, a
contiguous (non-paged) cache + batched commit, and a large `DG_TRACE_REGION_SIZE`; a paged/vLLM
cache, the gumbel regime, or a 0-byte trace region would break a default-on path. The gumbel regime
falls back (the multi-step controller inherits the single-step argmax-regime guard and raises
`NotImplementedError` for a non-None gumbel tensor). `git diff main -- models/demos/gemma4/` stays
empty.
