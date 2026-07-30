# Early halt in the traced denoise loop (#47465)

Status: current — window-1 early halt is intrinsic and unconditional in
`tt/traced_denoise.py::UpfrontTracedDenoiseController` (it reports `halt_window: 1`). The two
early-halt env flags and the grouped-window / frozen-prefix controller variants were deleted; the
module docstring says those variants "do not live in this module" (dead flag names: see
[flag triage](flag_triage_20260728.md)).
Owns: the halt criterion and its device/host split, the halt firing-status contradiction, the
`block(K)` dispatch cost model, the inverted argmax-vs-Gumbel claim, and the recon verdicts absorbed
from the deleted `multistep_trace_batching.md` and `denoise_replay_recovery_plan.md`.
See also: [refuted list](../REFUTED.md), [optimize_perf hub](README.md).

Over the 100-line cap: two open contradictions, four refutations and two cost models with no other
home.

## The criterion

A static Metal trace bakes a fixed op graph, so the variable-length denoise loop **cannot** be traced
whole and still stop early — any halt scheme must not trace the whole loop.

Each traced step reduces the halt condition to two `[1,1,1,1]` device scalars in
`tt/denoise_loop.py`: `mean_entropy` = `sum(entropy)/canvas_len` over the 256 canvas positions
(entropy upcast to fp32 first), and `mismatch` = count of clean-argmax positions changed vs the
previous step (`ne` then `sum`). The **host** reads those two 4-byte scalars and applies the exact
eager rule (`eval_halt`): halt when `mismatch == 0` AND `mean_entropy < entropy_stop_threshold`
(0.005), with a prior step required. No fp threshold decision is baked into the device.
`mismatch == 0` is exactly `torch.equal(argmax, prev)` at `stable_steps_to_halt == 1`, the released
config value; the step cap is K=48.

The device reduction is trustworthy because token ids ≤ 262144 are below 2^24 and therefore exact in
fp32, so `ne`+`sum` reproduces `torch.equal`+count bit-for-bit — measured max mismatch error **0
(exact)**, max mean-entropy error **5e-7** at 6 layers/12 steps and **1.2e-5** at 30 layers/48 steps.

The halt scalars are a **read-only side computation** over the same per-step argmax/entropy the fixed
path already produces; they never touch the canvas thread or the committed argmax, so enabling
halting cannot perturb a commit. `prev_argmax`, `mean_entropy` and `mismatch` are persistent buffers
allocated BEFORE `begin_trace_capture` with their in-trace `ttnn.copy` writes warmed once eagerly —
an instance of the trace-lifetime rule ([hub](README.md)), and what made the traced loop bit-exact.

## Firing status

> **OPEN CONTRADICTION (unexplained):** this file's own 30L eager oracle measured every block running
> the full 48 steps with mean entropy floored at **0.155 / 0.138 / 0.506 nats** (~30–100x above the
> 0.005 threshold), `halted = False` on all three — "early halt never fires under #48291". The
> deleted `denoise_replay_recovery_plan.md` measured halts at steps **[9,17,2] of 48** on the same
> canonical prompt, bit-exact vs the full path, and explicitly withdrew the "never clears 0.005"
> note, attributing the change to the tanh-GELU fix;
> [realized K on 8 GPQA-Diamond prompts](upfront_earlyhalt_gpqa_20260722.md) records **K = 10–43**
> (up-front traced, 8/8 released) and the concat MoE records 8–27 steps. The arms were never re-run
> against each other under one MoE/GELU configuration, so which factor moved the entropy gate is
> **not explained**. The old arm is additionally void as an absolute result: it ran the fast-approx
> GELU and the token-gather denoise MoE that was later deleted.

Diagnostic split that survives the supersession, and why `halted=False` alone tells you nothing: the
**stability** gate does fire (blocks with 14–18 argmax-stable steps) while the **confidence**
(entropy) gate is the one that blocks. A higher-precision *terminal* argmax/entropy re-measures the
same logit distribution and cannot make it more confident — precision is not a lever on the entropy
gate. The telemetry that separates a converged canvas from a collapsed one is in
[device_gumbel_restored.md](../decision_fidelity/device_gumbel_restored.md).

## Noise regime — the claim is inverted

Every pre-2026-07 doc in this tree states that the traced denoise path "supports only argmax
(`gumbel_noise=None`) and raises on a real tensor/descriptor". **That is inverted.** The shipped
up-front traced controller REQUIRES a per-step **materialized Gumbel tensor** and raises
`argmax and chunked/descriptor noise are unsupported`. Any doc still asserting the argmax
prerequisite is provenance, not a constraint.

## Overhead, break-even and cost models

Orchestration overhead is the host sync + 8-byte read + branch a fixed-budget traced path does not
pay: **5.87 ms/step** per-step (48 syncs/block, ~2.3% of the 260 ms device step); **28.1 ms/window**
at K=4 (12 syncs) and **34.7 ms/window** at K=8 (6 syncs). It is tiny because denoise steps are
ALREADY device-serialized (each step's forward depends on the previous step's canvas), so a per-step
sync adds a short host round-trip, not a pipeline stall. Break-even halt step, below which halting
beats the fixed budget: **46.9** per-step, **46.7** at K=4, **47.2** at K=8 — i.e. **any** block that
stops at ≤46 of 48 steps is a net win.

Fixed-48 traced steady block **14.069 s = 18.20 t/s** and fixed-12 traced **4.693 s = 54.55 t/s** at
full 30L, both committing `a9f0d18709b07d1e`. Those two points reconcile as
`block = commit + steps·step_dev` ⇒ **step_dev = 0.260 s/step, commit = 1.57 s**
(`14.069 = 1.57 + 48·0.260`; `4.693 = 1.57 + 12·0.260`).

The dispatch cost model that motivated all trace-shape work: **`block(K) ≈ 0.275·K + 1.09 s`** at
full 30L (58.29 t/s at 12 steps, 33.28 at 24). 100 t/s requires `block ≤ 256/100 = 2.56 s`, which
with single-step replays holds only at `K ≤ ~5` — below any quality-safe step budget. The ~1.09 s
fixed term is the per-replay dispatch of the single-step trace paid K times per block, plus the two
per-block `synchronize_device` barriers and the per-block refresh. The landed single-step traced
serving loop removed a ~137 ms/step host-dispatch tax and cleared 30 t/s bit-exactly.

> **OPEN CONTRADICTION (unexplained):** the denoise per-step cost is **0.260 s/step** by the
> reconciliation above (fixed-48 traced, 14.069 s block, old GELU / token-gather MoE) and **0.42
> s/step** replay-only in the frozen-prefix measurement below (21.5 s fixed-48 block, tanh-GELU, no
> fused MoE-dispatch kernel). The sources attribute the gap to the heavier correct tanh-GELU plus the
> omitted fused kernel — "headroom, not a regression" — but the two were never measured in one
> configuration, so the per-step figure is **not explained**.

## Absorbed recon verdicts

**The recapture regression.** Per-block Metal-trace RECAPTURE dropped steady serving from ~18 to ~3.6
tok/s, while the growing-prefix correctness committed by `ec5b64b4891` (tokens committed in earlier
blocks must be visible to later blocks' cross-attention) had to be kept. The sole remaining recapture
cause was the growing concatenated prefix K/V (dim-2 grows by `canvas_len` per block via
`read_prompt_kv_cache_slice`); canvas RoPE was already trace-fixed by the constant-shape
`canvas_rope_provider`.

**No drop-in paged prefix read exists** — three reasons in [refuted list](../REFUTED.md). What IS
reusable: `ttnn.transformer.chunked_scaled_dot_product_attention` is a real paged prefill SDPA with a
fixed-shape `page_table_tensor` and a trace-safe device-tensor offset (`chunk_start_idx_tensor`,
"update on device, no recompile"), and DiffusionGemma's commit path already writes/reads paged
(`tt/commit_decode.py`). The missing online-softmax merge is exactly what motivated the
[return_lse kernel work](return_lse_kernel_plan.md) and `tt/attention_merge.py`. Design successor:
[paged prefix denoise design](paged_prefix_denoise_design.md).

**Frozen-prefix device measurement** (full 30L, canonical prompt, 3 blocks): fixed-48 gives **11.92
t/s / 21.5 s per block**; frozen + early-halt gives **47.84 t/s / 5.35 s per block** with steps
[9,17,2] halted — ~3.9x and ~16x against the regressed 3.03 t/s. Capture-once confirmed by counters:
2 capture events, 8 frozen-prefix reuses, 0 recapture, per-step cost 1.746 s (capture+replay) → 0.42
s (replay-only). The approach is refuted as an answer (later blocks do not attend earlier blocks'
committed KV); the numbers stand as provenance.

**Correctness watch-item for any prefix scheme:** the reveal / last-page boundary must NEVER expose
uncommitted tokens to the cross-attention.

## Determinism invariants any trace-shape change must preserve

* per-step temperature baked via `temperature_at_step(i, max_denoise_steps, t_start, t_end)`;
* per-step noise from persistent `noise_bufs[i]`, refreshed in the same `noise_tokens_fn(step)` order
  and count so the seeded generator stream is untouched;
* self-conditioning carried in the adapter's persistent in-place `signal_buf`, re-zeroed per block by
  `reset_signal_buffer` so step 0 reads zeros == the eager `condition(None)` == `post_norm(embed)`;
* canvas RoPE refreshed per block OUTSIDE the trace into constant-shape per-layer-type buffers (RoPE
  depends only on absolute position).

## Open items

* ~~The grouped-window (G>1) trace was never gated on device.~~ **It WAS — and it was rejected.**
  On 2026-07-08 (`sweep_at48.py`, 30L / 48 steps / 3 blocks) the bounded G=12 window was bit-exact
  (`committed_sha a9f0d18709b07d1e`) and worth **+0.3%, i.e. noise**, while the whole-block window
  crashed with a `TT_FATAL` buffer-region overflow. At 48 steps the block is compute-bound, so
  batching only removes per-replay host dispatch. See
  [REFUTED.md](../REFUTED.md#trace-commit-and-self-conditioning). This is NOT an open item; the
  trace-region argument below is a first-principles hypothesis that the hardware already answered,
  and the whole-block crash is evidence AGAINST it: the heavy per-step intermediates
  (`[1,1,C,262144]` logits ≈ 134 MB bf16 at C=256, plus the 30-layer forward activations) are
  deallocated between steps inside a capture, so a window's peak *should* be one step's — but the
  whole-block window overflowed anyway. Its harness was deleted with the controller.
* **Trace-region sizing:** 48 single-step traces at 30 layers need roughly **8 GB** of
  `DG_TRACE_REGION_SIZE`; a K-step window scheme needs `ceil(48/K)` window traces.
* **Measurement-design note for any multi-arm denoise benchmark:** measure DENOISE ONLY. Two serving
  loops with commit enabled on one model build double-commit into the same KV cache; commit is an
  additive per-block constant to fold back into a projection.

## Reproduction

env: see [plan](../../plan.md).

```bash
python models/experimental/diffusion_gemma/doc/optimize_perf/probe_halt_gap.py
```

The eager halt-step oracle (30L, 3 blocks, prompt "Explain what a diffusion language model is in one
sentence."). The harness behind the correctness / overhead / break-even numbers above was deleted
with the flags; those numbers are provenance, not re-runnable as written.
