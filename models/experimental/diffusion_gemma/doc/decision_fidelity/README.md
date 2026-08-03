# DiffusionGemma decision fidelity (#48291)

Status: current for the bf16 floor, the exact-decision probe and the GPQA traps. Its 2026-07-15
verdict ("DECIDED — TT is at the intrinsic bf16 floor, nothing left to fix") was **superseded
2026-07-26**: 94% of the observed canvas collapses turned out to be two TT-specific defects, both
since fixed — see [device Gumbel restored](device_gumbel_restored.md). The floor measurements below
stand; the "therefore it is a ceiling" reading does not.
Owns: the #48291 bf16 chaos-amplification class, the six-field exact diffusion-decision probe, the
bf16-floor self-consistency control and the #48291 gate decision, the localization ledger, and the
GPQA measurement traps.
See also: [refuted list](../REFUTED.md) · [device Gumbel restored](device_gumbel_restored.md) · [degeneracy guard](degenerate_output_fix.md) · [plan](../../plan.md)

Over the 140-line target because every section below is a measurement trap, a refutation pointer or
a reproduction path.

## The #48291 class: bf16 chaos amplification

Block diffusion commits the **clean argmax** with no temperature cushion, and the accept/renoise loop
feeds each step's output back in. A few-ULP per-norm bf16 perturbation therefore compounds through 30
layers × 16–48 steps and flips ~85% of committed tokens — while producing an equally valid paraphrase.
**Bit-exactness is not an achievable gate for this model.** Candidates are ranked on *decision
agreement* instead, and the same class covers batched commit, bfp8 experts and the full-canvas norm
([commit batching](../optimize_perf/commit_batching.md),
[datatype sweep](../datatype_sweep/README.md), [L1 residency](../optimize_perf/l1_residency.md)).

## The bf16-floor self-consistency control (decisive, zero TT kernels)

Run the **same HF model** in fp32 vs bf16 with identical injected noise. 8-step fixed gate schedule,
canonical prompt *"Explain what a diffusion language model is in one sentence."*:

| comparison | seed 0 committed | seed 1 committed |
|---|---|---|
| **HF-fp32 vs HF-bf16** — the intrinsic floor, no TT | **0.86328** (35 differ) | **0.91406** (22 differ) |
| HF-fp32 vs TT — TT against the fp32 *ideal* | 0.86328 (35) | **0.98047** (5) |
| HF-bf16 vs TT — **the gate as specified** | 0.99609 (1) | 0.91406 (22) |

The floor holds at the production 48-step budget with early halt, so it is not an 8-step truncation
artifact: seed 0 fp32 halts at 7 steps / bf16 at 8, committed **0.86719**; seed 1 fp32 at 14 / bf16
at 15, committed **0.91406**. Artifacts `/tmp/dg48291_hf_{fp32,bf16}_prod48_seed{0,1}.pt`.

Across 5 seeds (full 3-way; TT via device replay per seed): floor mean **0.899** (4/5 below 0.95),
fp32-vs-TT mean **0.916**, bf16-vs-TT gate mean **0.935** (4/5 below 0.95). Only seed 4 (0.957)
clears the committed bar, and it still fails the full strict gate. Artifacts
`/tmp/dg48291_floor_5seed.pt`, `/tmp/dg48291_seed{2,3,4}.pt`, `/tmp/dg48291_tanh_seed{0,1}.pt`;
harness `doc/decision_fidelity/floor_5seed.py`.

Direct TT at the production 48-step config (seed 1, canonical prompt, early halt, P150x4;
non-degenerate, 36 content tokens, both halted): TT-vs-fp32 **0.99219** (2 differ), fp32-vs-bf16
0.91406, TT-vs-bf16 0.91016 — **TT tracks the fp32 ideal more closely than the bf16 reference does**.
Artifact `/tmp/dg48291_tt_prod48_seed1_fixed.pt`. *Provenance only as an absolute number:* it was
measured on the token-gather denoise MoE deleted 2026-07-29 for not converging.

**Decision.** The strict gate (`committed_match > 0.95` **and** every active-step entropy PCC `> 0.95`
**and** terminal accept-IoU `> 0.95`) is unreachable by **any** bf16 implementation, the HF reference
included. Recommendation: product-accept the current coherent output; re-spec the gate to fidelity
against the fp32 ideal plus alignment-robust agreement. **The production gate's pass criteria are
deliberately UNCHANGED and still RED, pending owner sign-off** — flipping them is a
correctness-policy change. Two metrics were refuted outright (per-step entropy PCC; positional
`committed_match` / non-EOS agreement) — see the [refuted list](../REFUTED.md#sampling-rng-and-decision-fidelity).

Reproduce the control — needs the DG venv and ~104 GB host RAM for the fp32 model, **no TT device**
(HF runs on CPU); env: see [plan.md](../../plan.md):

```bash
PYTHONPATH=$TT_METAL_HOME python \
  models/experimental/diffusion_gemma/doc/decision_fidelity/measure_bf16_floor.py \
  --stage-artifact /tmp/dg48291_tanh_seed1.pt --checkpoint $DG_CKPT
```

`demo/replay_hf_tt.py --hf-dtype {bfloat16,float32}` (fp32 forbidden under `--stage-gate`)
regenerates the fp32 reference trajectory directly.

## The canonical #48291 replay probe

```bash
# env: see plan.md
python -u models/experimental/diffusion_gemma/demo/replay_hf_tt.py \
  --checkpoint "$DG_CKPT" --hf-checkpoint "$DG_CKPT" --local-files-only \
  --prompt "Explain what a diffusion language model is in one sentence." \
  --seed 0 --max-denoising-steps 8 --noise-mode seeded --stage-gate \
  --output /tmp/dg48291_stage_gate.pt
```

HF and TT receive an identical initial canvas, FP32 Gumbel noise and renoise token ids; SHA-256
hashes are saved in the artifact. `--stage-gate` enforces seeded noise, canvas 256, eight denoise
steps, the full model, P150x4 and no diagnostic HF injection. Post-fix traced seed 0 captures/replays
all eight single-step traces and reproduces the eager committed / non-EOS agreement exactly at
`0.99609375 / 0.9714286`.

## The six-field exact diffusion-decision probe

`verify_selfcond_prechunk_decisions.py` hashes, at **every** step of a full 48-step 256-token
trajectory: clean argmax · sampled token ids · BF16 per-position entropy · entropy-budget accept
mask · accepted/renoised next canvas · the explicit clean **commit candidate** for that step. This is
the instrument for "did this change alter a decision?" — 48/48 exact means byte-identical decisions.

- **Caveat:** intermediate steps are **candidates, not commits.** No KV commit occurs during
  intermediate denoise steps; DiffusionGemma commits the last candidate, and the probe asserts the
  final per-step candidate hash equals the trajectory's actual commit hash.

```bash
# env: see plan.md
python -u models/experimental/diffusion_gemma/doc/optimize_perf/verify_selfcond_prechunk_decisions.py \
  --steps 48 --gumbel-mode chunked --out decisions.json
# then: --compare decisions-control.json decisions-default.json --out decisions-comparison.json
```

## Localization ledger (what the drift is, and is not)

- Worst single-layer output under an exact-input + exact-KV full-canvas ledger is **layer 18**
  (PCC 0.9974325); split branches post-attention 0.9997648, shared FF 0.9983796, routed expert FF
  0.9918051, post-FF 0.9965505.
- Every TT causal-prefill layer run from exact HF input writes K/V at PCC 0.99991–0.99996 — normal
  K/V drift is inherited from hidden drift, not generated in the cache.
- Device, BF16 left-fold, BF16 pairwise and FP32 host TP sums all give branch PCC ≈ 0.9427, so
  summation order is not the cause.
- **TRAP:** injecting HF's exact per-step self-conditioning signal, or HF's exact layer-0 input on
  every step, gives agreement 1.0. Those are **feedback resets from a different trajectory, not TT
  fixes** — injecting the layer-0 input only at step 0 regresses to 0.86328125.
- **Router top-8 study** (device-free proxy, `measure_topk_overlap.py --stage-artifact
  /tmp/dg48291_tanh_seed{0,1}.pt`, artifacts `/tmp/dg48291_topk2_tanh_seed{0,1}.pt`): at step 0 with
  identical input, top-8 index sets share **6.80/8**, weight-mass overlap **0.846**, top-1
  dominant-expert agreement 0.818, rank-0 drop 4.2%, rank-7 drop 36.4%. Flips are **tail-dominated**
  — the dominant expert survives the top-8 ~96% of the time and only ~15% of routing *weight* lands
  on a flipped expert. This is bf16 backbone drift, **not a faulty router**: the topk operator is
  essentially exact where the hidden state is shared (layers 0–4 weight-mass 0.98–0.99, top-1
  0.97–1.00), diverges most at layers 20–27 (0.64–0.72, top-1 0.55–0.66), then partly recovers at
  28–29 (0.85–0.90).
  - **TRAP:** the OVERALL 8-step figures (top-8 shared 5.21/8, weight-mass 0.637) are lower than step
    0 only because the two runs commit different tokens after step 0. The router did not get worse.
- **TRAP:** converged-step entropy absolute error is **not** tiny — the self-consistency control
  reaches max |Δ entropy| ≈ 1.1 (seed 0 step 5) and ≈ 2.8 at transition steps (seed 1 step 2),
  reference against itself in fp32.

## Retained fixes

- All DG MLP paths select `ttnn.GeluVariant.Tanh`, matching checkpoint `gelu_pytorch_tanh` (the
  legacy boolean selected FastLut erf-GELU). Raises canonical seed-0 committed
  `0.9296875 → 0.99609375`; the OFF arm was deleted 2026-07-28 and the selection is unconditional.
  It also raises layer-18 shared FF `0.9983796 → 0.9998859`, routed FF `0.9918051 → 0.9937973`,
  exact-route routed FF `0.9977631 → 0.9997175`, post-FF `0.9965505 → 0.9978347`. An independent
  review clean-passed coverage: direct model, sparse/dense expert, shared MLP, self-conditioning,
  regular/chunked prefill, commit and traced target paths all enter the DG tanh context; plain Gemma4
  stays outside it.
- Sparse-matmul FP32 intermediate circular-buffer sizing follows the actual intermediate format
  (regression PCC 0.999998, max abs error 0.1633).
- Denoise MoE capacity defaults to the canvas length. The old capacity of 32 had per-expert loads of
  156–256 and dropped 838–1711 of 2048 active routes per layer — committed match 0.5625 at capacity
  32 against 0.91796875 zero-drop at 256.
- `tests/trajectory_pcc.py:sound_entropy_step_fidelity` requires the absolute tolerance on **both**
  branches (catching affine/offset errors) and applies PCC only where the reference profile has
  structure. It is a **conditioning fix, not a reachable gate** — at genuine transition-step
  divergence the reference fails it against itself.
- The stage gate stops its entropy minimum at common all-accept and records the raw all-step minimum
  separately.
- **`DG_SPARSE_EXPERT_FP32_FULL_SYNC` — DELETED 2026-08-03, and its number was void twice over.**
  The recorded seed-0 committed gain `0.9296875 → 0.93359375` (+0.4pp, against a seed-1 active-step
  entropy regression) was measured on the token-gather MoE that was deleted 2026-07-29, AND its
  `0.9296875` baseline is the pre-tanh-GeLU figure named at line 139 above, which the tanh-GeLU fix
  moved to `0.99609375` — a +6.6pp correction that swamps the +0.4pp. Nothing in the tree ever set
  the flag. If Blackhole full-DST FP32 experts are ever revisited, it needs a fresh paired run on the
  concat MoE, not this row.

## GPQA measurement traps

1. **Three denominators — extractable / non-empty / all.** The `boxed_choice` extractor's stage 3
   hands a letter to responses that never answered, so the aggregate launders non-answers into
   apparent accuracy. Split by stage, report all three, and never silently drop empties. Worked
   example: a 3072-token thinking run scored 0.35 overall but 0.70 on the traces that *finished*
   ([gpqa_thinking3072](../optimize_perf/gpqa_thinking3072_sub40_20260723.md)).
2. **Check `prefill_block0` count against the question count before recording any score.** A dead
   engine returns HTTP 200 with an empty body and lm_eval still publishes a normal-looking score — a
   GREEN CI run can therefore report a score for a run that died partway through.
3. **A reference score is comparable only at a matched generation budget.** The reference bar
   (70.71% / 70.20%, `gpqa_diamond_cot_zeroshot` + flexible-extract, thinking) was measured at its
   own very large budget; at a matched 5632-token budget on the same 11 questions the ordering
   changes ([serving hub](../vllm_integration/README.md)). Compare budgets before comparing scores.

## Open

- **fp32 MoE backbone precision** is blocked by `ttnn.topk` `TT_FATAL` on FLOAT32 and by fp32 experts
  exceeding the QB2 DRAM budget. It is a separate owned effort on the shared Gemma-4 backbone, and
  DiffusionGemma must not edit `models/demos/gemma4/`.
- **Inherited HARD-RULE gate failure (flagged for owner action):** the branch carries committed
  shared-gemma4 edits — `tt/experts/operations.py` +1 `deallocate` (commit `bf98aaf2e23`, #47464) and
  `tt/model.py` sharded-terminal (commit `a22107f0447`) — that fail
  `check_no_shared_gemma4_edits.sh` against the origin/main merge-base. Related open contradiction:
  [plan §6](../../plan.md#6-open-items-and-contradictions).
- **TRAP for anyone re-running the 48-step device gate:** a first attempt collapsed to all-EOS
  because the run used the default "Once upon a time" prompt rather than the canonical one. HF
  collapsed identically, confirming prompt/invocation rather than the device path.
