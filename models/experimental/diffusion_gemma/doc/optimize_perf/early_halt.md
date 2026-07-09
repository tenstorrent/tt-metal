# DiffusionGemma — data-dependent early-halt in the traced denoise loop (dg-08 lever 8, #47465)

**Goal.** Recover the eager path's data-dependent early-halt (stop denoising once a block's
canvas has converged) **while keeping the traced path's dispatch savings**, so a converged
block runs fewer than the fixed ≤48 steps. The traced loop is otherwise **fixed-48**
(early-halt was lost when the loop was traced — a static Metal trace fixes the step count);
model-faithful throughput at 48 steps ≈ **17.9–18.2 t/s** (`perf_campaign_worklog.md`; this stage's
own fixed-48 run measured 18.20 t/s — the ~2% spread across runs is timing variance, same committed
tokens `a9f0d18709b07d1e`, and the overhead/break-even below uses a single internally-consistent run).

**Bottom line (honest).** The mechanism is built, correct (bit-identical to the fixed-48
traced path when it does not fire; bit-identical to the eager StableAndConfident reference at
the same halt step when it does), and landed behind a DG-local flag defaulting to the current
fixed-48 behaviour. **But under #48291 it is a no-op on real output**: the confidence
(entropy) gate never clears the 0.005 threshold, so blocks run the full 48 steps — measured,
not assumed (below). Early-halt therefore does **not** beat fixed-48 today; the value here is a
correct, ready mechanism plus a measured break-even for when #48291 is resolved (or a schedule
cut lowers the step budget). This matches `path_to_100tps.md` (d): "100 t/s only exists in the
≤16–20-step early-halt regime — and early-halt cannot fire because of #48291."

---

## The mechanism constraint (why not just trace the loop)

A static Metal trace bakes a fixed op graph, so the whole variable-length denoise loop cannot
be traced and still stop early. The scheme must **not** trace the whole loop. Two evaluated:

- **(A) per-step** — trace ONE denoise step; replay it in a host loop; after each replay read
  a single on-device convergence scalar and branch continue/stop on host.
- **(B) chunked-halt** — trace a fixed K-step window; replay the window; check the halt scalar
  once per window; stop at the first window boundary at-or-after convergence. K trades halt
  granularity against per-window host-orchestration overhead.

Both are implemented in one controller (`tt/traced_denoise.py::EarlyHaltTracedDenoiseController`,
extending the multi-step traced controller): a K-step window capture with `K==1` ⇒ scheme A and
`K>1` ⇒ scheme B. The **only difference between A and B is how often the host syncs+reads+branches**
(every step vs every window); the in-trace per-step compute is identical.

## The on-device halt scalar (no 5-tensor readback)

The retired eager loop read back **5 full `[B,L]` tensors/step** (argmax, entropy, sampled,
accept, canvas) — `bench_loop_readback.py` measured that at **27.76 ms/step**, and killing it
is why the loop was traced. Reintroducing it would defeat the purpose. Instead, every traced
step reduces the halt condition to **one tiny scalar pair on device** (`tt/denoise_loop.py`):

- `mean_entropy` `[1,1,1,1]` — `sum(entropy)/canvas_len` over the 256 canvas positions, matching
  the eager `entropy.mean()` (entropy upcast to fp32 first, as the host path does).
- `mismatch` `[1,1,1,1]` — count of positions whose clean argmax changed vs the previous step
  (`ne` then `sum`); `0` ⟺ `torch.equal(argmax, prev)` — the eager stability gate at the
  released `stable_steps_to_halt == 1`.

The HOST reads these two 4-byte scalars after each step/window and applies the **exact eager
rule** (`eval_halt`): halt when `mismatch == 0` (stable) AND `mean_entropy < entropy_stop_threshold`
(confident), with a prior step required. No fp threshold decision is baked into the device.
The per-step in-trace halt ops are ~sub-ms 256-wide reductions — negligible vs the ~233 ms/step
device compute — so A and B run the same per-step device work as the fixed budget.

Trace-safety: `prev_argmax`, `mean_entropy`, `mismatch` are persistent buffers allocated BEFORE
`begin_trace_capture` and their in-trace `ttnn.copy` writes warmed once eagerly (the session-8
rule that made the traced loop bit-exact). The halt scalars are a **read-only side computation**
over the same per-step `argmax`/`entropy` the fixed-48 path already produces — they never touch
the canvas thread or the committed argmax.

---

## Correctness guards (all green)

Measured on device (`probe_early_halt.py --mode correctness`, 6 layers, N=12, tuned MoE,
`DG_SPARSE_MOE_TUNED=1`). Reduced layers exercise the mechanism (which is layer-count-independent);
the 30L per-step decision faithfulness is separately confirmed against the eager halt-gap below.

| guard | check | result |
|---|---|---|
| **G1 — no-halt ≡ fixed-48** | scheme-A with an impossible threshold (never halts) commits the byte-identical argmax of the fixed-48 traced path | `committed_sha` match **✅** (`3d744378ec43a7e3` both) |
| **G2 — forced-halt ≡ eager (scheme A)** | elevated-threshold scheme-A vs eager `tt_denoise_block` (same rule on host): commit + realized steps + halted flag | `sha` **✅** · `steps` **✅** · `halted` **✅** |
| **G2 — forced-halt ≡ eager (scheme B, K=4)** | same, chunked window | `sha` match **✅** |
| **per-step scalar faithfulness** | device `(mean_entropy, mismatch)` vs eager per-step `(entropy.mean(), argmax_changes)`, all 12 steps | max entropy err **5e-7** · max mismatch err **0 (exact)** |

The exact mismatch agreement (0) is expected: token ids ≤262144 (< 2^24) are exact in fp32, so
the device `ne`+`sum` reproduces `torch.equal`+count bit-for-bit. The 5e-7 entropy agreement is
the device-fp32-mean vs host-fp32-mean summation-order difference — far below any threshold
boundary risk. An actual early-*return* (halt firing mid-block) is demonstrated at 30L below
(the 6-layer reduced model never stabilizes, so it cannot fire even at threshold 100).

---

## The realized halt-step distribution (honest, #48291)

`probe_halt_gap.py` (eager, 30L, tuned MoE, 3 blocks, prompt "Explain what a diffusion language
model is in one sentence.") — the eager `StableAndConfident` oracle:

| block | num_steps | halted | entropy_mean min | steps entropy<0.005 | steps argmax-stable | would_early_halt |
|---|---|---|---|---|---|---|
| 0 | 48 | False | 0.155 | **0** | 0 | False |
| 1 | 48 | False | 0.138 | **0** | 18 | False |
| 2 | 48 | False | 0.506 | **0** | 14 | False |

**Every block runs the full 48 steps. The stability gate DOES fire** (blocks 1–2 have 14–18
steps with zero argmax changes) — it is purely the **confidence (entropy) gate that #48291
blocks**: the bf16/MoE/TP=4 backbone produces broad logit distributions whose mean per-position
entropy floors at **~0.14–0.51 nats, ~30–100× above the 0.005 threshold**. So `confident` is
never true and early-halt never fires. This is a property of the backbone logit *distribution*
(a #48291 fidelity consequence), not of the halt mechanism. A higher-precision *terminal*
argmax/entropy re-measures the same distribution and cannot make it more confident.

Scheme-A with the real 0.005 threshold reproduces this: `denoise_steps_per_block` = full budget,
`halted = False` (see the perf run below) — the mechanism is correct and simply does not trigger.

---

## Overhead + break-even (perf run)

`probe_early_halt.py --mode perf` (full 30L, N=48, tuned MoE, `DG_TRACE_REGION_SIZE=10 GB`,
3 blocks/config, steady = mean(block[1:]); traced Metal capture/replay + synchronized block
timing, ENABLE_TRACY=OFF). All configs commit under `stop_token_ids=[]` (RUN-first).

| config | steps run / block | halted | steady block (s) | t/s | commit sha |
|---|---|---|---|---|---|
| **fixed-48 traced** (baseline) | [48,48,48] | [F,F,F] | 14.069 | **18.20** | `a9f0d18709b07d1e` |
| fixed-12 traced (fit point) | [12,12,12] | [F,F,F] | 4.693 | 54.55 | — |
| **scheme A** no-halt (thr −1e9) | [48,48,48] | [F,F,F] | 14.351 | 17.84 | `a9f0d18709b07d1e` ✅ |
| **scheme A** real (thr 0.005) | [48,48,48] | [F,F,F] | 14.349 | 17.84 | `a9f0d18709b07d1e` ✅ |
| **scheme B K=4** no-halt | [48,48,48] | [F,F,F] | 14.406 | 17.77 | — |
| **scheme B K=8** no-halt | [48,48,48] | [F,F,F] | 14.277 | 17.93 | — |

- **Guard 1 at 30L confirmed byte-identical**: scheme-A no-halt AND scheme-A real-threshold both
  commit the SAME `a9f0d18709b07d1e` as the fixed-48 traced baseline (this is also the established
  `traced_tuned_s48` sha in `perf_campaign_worklog.md`). The halt machinery does not perturb the commit.
- **`ms_per_block` reconciliation**: from the two fixed-budget points,
  `block = commit + steps·step_dev` ⇒ **step_dev = 0.260 s/step, commit = 1.57 s**
  (`14.069 = 1.57 + 48·0.260` ✓; `4.693 = 1.57 + 12·0.260` ✓).
- **Orchestration overhead** (the host sync + 8-byte read + branch the fixed traced path does not pay):
  - scheme A: **5.87 ms/step** (48 syncs/block) — ~2.3% of the 260 ms device step.
  - scheme B: **28.1 ms/window** (K=4, 12 syncs) / **34.7 ms/window** (K=8, 6 syncs).
  The overhead is tiny because the denoise steps are ALREADY device-serialized (each step's forward
  depends on the previous step's canvas), so a per-step sync adds only a short host round-trip, not a
  pipeline stall. Total per-block overhead (~208–337 ms) sits within block-timing noise across A/B/K,
  so the per-sync-count differences are not resolvable here — the takeaway is simply "all ≈ 2%".
- **Break-even halt-step** (below which the scheme beats fixed-48): **A = 46.9, B(K=4) = 46.7,
  B(K=8) = 47.2**. Because the overhead is ~2%, break-even is ≈47 steps — i.e. **any** early halt (any
  block that stops at ≤46 steps) makes the scheme faster than fixed-48. Scheme B(K=8) has the highest
  break-even (fewest syncs), the marginal winner, but the margin is within noise.

**So: when early-halt fires, the win is real and cheap (break-even ≈47/48); the whole cost is the
~2% no-halt overhead you pay on blocks that run to the budget.** Under #48291 every block runs the
budget (below), so today the flag is a ~2% net loss — hence default OFF.

---

## Forced-halt demonstration at 30L (early-return path)

The 6-layer correctness run validated the halt-scalar computation, the host branch, and Guard 1/2
agreement, but the reduced model never stabilizes (argmax changes every step), so the early-*return*
`break` was never taken there. To exercise the actual early-return on real stable-argmax blocks, this
runs `probe_early_halt.py --mode correctness` at **full 30L, N=48**, with an elevated threshold
(`--forced-threshold 100`) so the confidence gate no longer blocks — the halt then fires wherever the
argmax is stable.

Under that threshold the eager oracle (measured, `EAGER_DIAG`) behaves exactly as the halt-gap:
**block 0's argmax never stabilizes** (changes 37,20,16,… — never 0) so it runs the full 48. Block 1
stabilizes early (`argmax_1 == argmax_0` per the halt-gap), so the elevated threshold fires the halt on
block 1. Guard 2 requires scheme A to halt at the SAME step and commit the SAME argmax as eager, and
scheme B(K=4) to halt at the first window boundary at-or-after that step committing the SAME tokens (the
argmax is stable across the early steps, so the window-end commit equals the eager halt-step commit).

Measured (`probe_early_halt.py --mode correctness --num-layers full --forced-threshold 100`, 30L,
N=48, 2 blocks) — the halt FIRES on block 1:

| path | block 0 (unstable) | block 1 (stable) | halted | commit sha vs eager |
|---|---|---|---|---|
| eager `tt_denoise_block` | 48 steps | **2 steps** | [F, **T**] | — |
| scheme A (K=1) | 48 steps | **2 steps** | [F, **T**] | **byte-identical** (Guard 2 sha ✅) |
| scheme B (K=4) | 48 steps | **4 steps** | [F, **T**] | **byte-identical** (Guard 2 sha ✅) |

Guard verdicts (`RESULT_EARLY_HALT`): `guard2_A_eq_eager` = sha ✅ · steps ✅ (`[48,2]==[48,2]`) · halted ✅
(`[F,T]==[F,T]`); `guard2_Bk4_vs_eager` = sha ✅ (scheme B halts at the step-4 window boundary vs eager's
step 2, committing the SAME tokens because the argmax is stable across steps 1–4); `guard1` no-halt ≡
fixed = sha ✅ (`8f015a49e4e31a63`); per-step scalar agreement over all 48 steps = max entropy err
**1.2e-5**, max mismatch err **0 (exact)**.

This exercises the mechanism's early-return end-to-end: a converged block stops before the 48-step cap
and commits the eager reference's tokens, while an unconverged block still runs the full budget and stays
byte-identical to fixed-48. It only requires the confidence gate to be satisfiable — which is exactly what
#48291 blocks on real output.

---

## Chosen scheme + flag

- **Scheme A (per-step, `DG_DENOISE_EARLY_HALT=1`, `DG_DENOISE_EARLY_HALT_WINDOW=1`, default)** is
  the correctness-clean choice: it halts at the EXACT eager step and commits the bit-identical
  argmax (Guard 2). Use it when correctness parity with the eager reference matters.
- **Scheme B (chunked, `DG_DENOISE_EARLY_HALT_WINDOW=K`, K>1)** amortizes the host sync over K
  steps (higher break-even = beats fixed-48 at more steps), at the cost of coarser halt
  granularity: it halts at the first window boundary at-or-after the eager step and commits the
  window-end argmax, which equals the eager commit under argmax-convergence-stability (the same
  property the fixed-48 traced path relies on to match eager).

**Default = OFF** (fixed-48 traced unchanged). Rationale: under #48291 early-halt is a no-op and
the per-step/window host sync only adds orchestration overhead, so enabling it today is a net
loss. The flag is ready to flip once #48291 is resolved (entropy clears 0.005) or a schedule cut
lowers the step budget into the regime where the measured break-even makes A/B win.

Prerequisites (same as the traced paths): argmax (`gumbel_noise=None`) regime, contiguous cache,
warmed program cache, and a large `DG_TRACE_REGION_SIZE` (48 single-step traces @30L ≈ 8 GB;
scheme B K needs `ceil(48/K)` window traces).

## Files

- `tt/denoise_loop.py` — `HaltBuffers`, `compute_halt_scalars`, `write_halt_scalars`,
  `denoise_step_next_canvas_and_halt`, `read_halt_scalars`, `eval_halt` (the on-device halt
  scalar + the eager rule on host).
- `tt/traced_denoise.py` — `EarlyHaltTracedDenoiseController` (schemes A/B), `traced_early_halt_block`,
  `traced_early_halt_enabled`, `early_halt_window`.
- `tt/generate.py` — `_resolve_default_denoise_block_fn` wires `DG_DENOISE_EARLY_HALT` (precedence
  over the fixed-budget traced flags).
- `doc/optimize_perf/probe_early_halt.py` — correctness + overhead + break-even harness.
- `doc/optimize_perf/probe_halt_gap.py` — the eager halt-step oracle (pre-existing).
