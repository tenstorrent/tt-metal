# dg-07 datatype sweep — work log

## 2026-07-08 — bfp8 MoE experts: measured, FAILS decision gate, NOT landed as default

**Goal.** Evaluate the single biggest remaining in-repo speed lever: MoE expert weights
bf16 → `bfloat8_b` (rank-5 lever in `path_to_100tps.md`, "med" risk, fidelity-gated).

**Knob (DG-local, no shared edits).**
- New: `tt/precision_build.py::create_tt_model_dg` — delegates to the shared `create_tt_model`
  when no knob is set; with `DG_EXPERTS_BFP8=1` / `DG_EXPERTS_DTYPE=bfp8` it replicates
  `create_tt_model` and injects `Gemma4Precision({"experts": bfloat8_b})`.
- Wired as the default `create_model_fn` in `checkpoint.py::build_tt_model_from_checkpoint_inputs`
  so text_demo / sweeps / replay / vLLM all honour it.
- `git diff main -- models/demos/gemma4/` empty. CPU sanity: knob resolves None/BFLOAT8_B/BFLOAT16
  correctly.

**Decision agreement (bf16 vs bfp8, deterministic, 16 steps, 30L).** Fixed seeded canvas + fixed
per-step renoise + clean argmax + early-halt off ⇒ only difference is expert dtype.
`decision_agreement.py run x2 + compare`:
- committed clean-argmax agreement **0.227** (bar ≥0.95) — ~77% of committed tokens change.
- step-0 pure-logits argmax agreement 0.949 (~5% flip from bfp8 logits), compounds to 0.227 by step 15.
- mean per-step entropy PCC **0.631** (min 0.036 @step3; bar ≥0.95).
- mean accept/renoise IoU **0.501** (min 0.0; bar ≥0.90).
- Sample text: wash — bfp8 gives an equally-coherent opening sentence then degenerates (same
  #48291 regime as bf16); coherence not destroyed, but not the gate.
- **FAILS all three bars.**

**DRAM.** bf16 13.268 GiB/chip → bfp8 **7.830 GiB/chip** (−5.44 GiB, −41%). 90 `*_BFLOAT8_B`
expert cache files written ⇒ bfp8 genuinely consumed.

**Traced throughput (single-step serving, DG_DENOISE_TRACED, 10 GiB trace region).**
| steps | bf16 t/s | bfp8 t/s | speedup |
|---|---|---|---|
| 48 | 18.18 | 19.83 | +9.1% |
| 24 | 31.49 | 33.99 | +7.9% |
| 12 | 54.58 | 57.84 | +6.0% |
bf16 @48 = 18.18 t/s reproduces the stated 17.9 baseline (harness validated). bfp8 buys ~6–9%
only (step not purely weight-bound; MoE matmul launch/overhead-limited; MoE ~35% of step).

**100 t/s.** ~4.1 steps at bfp8 vs ~3.8 at bf16 — negligible shift, and 4 steps is far below any
quality-acceptable step count. bfp8 does not make 100 t/s reachable at acceptable quality.

**Decision.** bfp8 experts REJECTED on decision fidelity. bf16 experts stay selected; 17.9–18.2
t/s @48 stands. Knob landed OFF-by-default (opt-in) for reuse once #48291 headroom exists.
context_contract.json `datatype_policy` updated to record the sweep outcome (no capacity change —
only expert weight dtype tested, and rejected).

Raw logs: `bhqb:/home/zni/dg-agent-runs/{gate.log,perf.log}`, artifacts
`bhqb:/home/zni/dg-agent-runs/dtsweep/`.
