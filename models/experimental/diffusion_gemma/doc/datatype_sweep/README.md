# dg-07 datatype sweep — bfp8 MoE experts, rejected

Status: current as a **rejection**; its absolute throughput numbers are provenance only, because the
sweep ran on the token-gather denoise MoE that was later deleted. Marginally over the 100-line cap:
the three-bar gate definition, the measurement traps and the repro commands are never cut for length.
Owns: the bfp8-experts rejection, the three-bar decision-fidelity gate definition, and the
`DG_EXPERTS_BFP8` knob.
See also: [refuted list](../REFUTED.md) · [decision fidelity](../decision_fidelity/README.md) ·
[optimize_perf hub](../optimize_perf/README.md)

**VERDICT: bfp8 MoE experts FAIL the diffusion-decision-fidelity gate. DiffusionGemma keeps bf16
experts.** The DG-local bfp8 knob is landed **off by default** for reuse if #48291 ever creates
fidelity headroom. QB2 `bh-qbge-06` P150x4, mesh `(1,4)`, TP=4, full 30 layers, 2026-07-08;
`git diff main -- models/demos/gemma4/` verified empty.

## The lever and the knob

MoE expert `gate/up/down` weights are ~88.6% of on-device weight DRAM (~11.6 GiB/chip) and a large
fraction of the compute-bound denoise step. DiffusionGemma loads them at bf16 because that is the
gemma4 model-wide default and there is no `precision_overrides.json` entry for `gemma-4-26B-A4B-it`.
It was ranked lever 5 ("med" risk, fidelity-gated) in `../optimize_perf/path_to_100tps.md`.

`tt/precision_build.py::create_tt_model_dg` is wired as the default `create_model_fn` in
`checkpoint.py::build_tt_model_from_checkpoint_inputs`. With no knob it delegates to the shared
`create_tt_model` unchanged; with **`DG_EXPERTS_BFP8=1`** (or `DG_EXPERTS_DTYPE=bfp8`) it replicates
`create_tt_model` and passes `Gemma4Precision({"experts": bfloat8_b})` into `Gemma4Model`. CPU sanity
check: the knob resolves `None` / `BFLOAT8_B` / `BFLOAT16` correctly.

**Only the expert weights change.** Router, attention, shared MLP, embedding, lm_head, KV cache and
the entire decision path (logits softcap, softmax-to-probability, entropy, Gumbel-max argmax,
entropy-budget accept/renoise) keep their existing dtypes; production logits and entropy stay BF16.
Expert cache filenames carry the dtype suffix `_bfp8_dtype_BFLOAT8_B`, so bf16 and bfp8 caches
coexist without invalidating each other.

**Do not confuse `DG_EXPERTS_BFP8` (live, this sweep's knob) with `DG_MOE_EXPERT_BFP8`**, a separate
knob deleted 2026-07-28 as memory-negative as written
([flag triage](../optimize_perf/flag_triage_20260728.md)).

## The measurement contract and the three bars

Accuracy here means the diffusion **DECISION** versus the bf16-experts reference — not top-1/top-5,
not AIME24 teacher forcing. Determinism recipe: fixed seeded initial canvas + fixed per-step renoise
tokens + clean-argmax sampling + early-halt disabled, so a bf16 run and a bfp8 run differ **only** by
the expert weight dtype. Compared with `tests/trajectory_pcc.compare_trajectories`.

| metric | bar | rationale |
|---|---|---|
| committed clean-argmax agreement | >= 0.95 | diffusion commits the CLEAN argmax — no temperature/top-p cushion |
| mean per-step entropy PCC | >= 0.95 | entropy depends on the full distribution incl. small probabilities — the exact bfp8 risk |
| mean accept/renoise IoU | >= 0.90 | small-probability drift flips accept/renoise even when argmax is unchanged |

All three must pass. **WHY THE BAR IS NEUTRALITY:** #48291 already puts the bf16 model at ~50%
argmax vs HF, so a reduced-precision candidate must be near decision-**neutral** against the bf16
reference — there is no headroom to spend. The compounding mechanism that makes this class of
change unmeasurable by bit-exactness: [bf16 chaos amplification](../decision_fidelity/README.md).

## Results

| metric | bfp8 vs bf16 | bar |
|---|---|---|
| committed clean-argmax agreement | **0.227** | >= 0.95 ❌ |
| mean per-step entropy PCC | **0.631** (min 0.036, reached by step 3) | >= 0.95 ❌ |
| mean accept/renoise IoU | **0.501** (min 0.0) | >= 0.90 ❌ |
| step-0 pure-logits argmax agreement | 0.949 | — |
| mean per-step argmax agreement | 0.604 | — |
| mean canvas agreement | 0.906 | — |

bfp8 flips ~5% of positions from expert-logit drift alone at step 0, and that compounds over 16 steps
to 0.227 committed agreement — **~77% of the committed tokens change**.

**MEASUREMENT TRAP: sample text is a WASH, not a save.** bfp8 produces an equally coherent opening
sentence then degenerates into multilingual noise — the same #48291 regime as bf16. Coherence is not
destroyed by bfp8, but coherence is not the gate.

**DRAM:** bf16 experts 13.268 GiB/chip → bfp8 experts **7.830 GiB/chip** = **-5.44 GiB (-41%)**. That
drop, plus 90 `*_dtype_BFLOAT8_B` expert cache files written, is the proof bfp8 was genuinely
consumed rather than silently ignored.

**Traced throughput (provenance only — measured on the since-deleted token-gather MoE path):** at 48
steps bf16 18.18 t/s (14.079 s block) vs bfp8 19.83 t/s (12.907 s) = **+9.1%**; at 24 steps 31.49 vs
33.99 = +7.9%; at 12 steps 54.58 vs 57.84 = +6.0%. The bf16 @48 row reproduced the then-stated 17.9
baseline, which is what makes the delta a real effect rather than a harness artifact. **Why only
6-9%:** the denoise step is not purely weight-bound — the MoE batched matmul is
launch/overhead-limited (~46 GB/s effective), the MoE is ~35% of the step, and the fixed/terminal
overhead is unchanged, so halving the expert bytes only partly speeds it. Supersession: the
17.9–18.2 t/s @48 frame was the baseline at this sweep date; the current selected figure is
**18.844 t/s** (`../optimize_perf/selfcond_logits_l1_e2e.json`, headline in
`../optimize_perf/perf_progress.md`), so the +9.1% is only valid against the deleted path.

**REFUTED — bfp8 as a route to 100 t/s.** Block model fit over s12/s24/s48: bfp8
`block ~= 1.60 + 0.2356*steps`, bf16 `~= 1.56 + 0.2608*steps`. 100 t/s requires block <= 2.56 s, i.e.
~4.1 denoise steps at bfp8 vs ~3.8 at bf16 — and 4 steps is far below any quality-acceptable count
(the model needs >= 16–32 steps for even one coherent sentence).

**PARETO VERDICT:** the bfp8 point is faster (right of bf16) but sits at 0.227 committed argmax /
0.501 accept IoU, below the min-allowed line at **every** step count. There is no Pareto-improving
move; bf16 is the selected point. `doc/context_contract.json` `datatype_policy` records the outcome
with no capacity change, since only the expert weight dtype was tested and it was rejected.

## Reproduction

```bash
# env: see plan.md, plus DG_CKPT=/home/zni/dg_models/diffusiongemma-26B-A4B-it
# NOTE: the original runs also exported DG_SPARSE_MOE / DG_SPARSE_MOE_TUNED / DG_DEDUP_ARGMAX,
# which no longer exist -- the numbers above were taken on that deleted token-gather MoE path.

# decision agreement (run twice, then compare)
python doc/datatype_sweep/decision_agreement.py run \
    --max-denoising-steps 16 --canvas-length 256 --seed 0 --label bf16 --output traj_bf16.pt
DG_EXPERTS_BFP8=1 python doc/datatype_sweep/decision_agreement.py run \
    --max-denoising-steps 16 --canvas-length 256 --seed 0 --label bfp8 --output traj_bfp8.pt
python doc/datatype_sweep/decision_agreement.py compare \
    --ref traj_bf16.pt --cand traj_bfp8.pt --output agreement.json

# traced throughput (run twice: bf16, then with DG_EXPERTS_BFP8=1)
DG_TRACE_REGION_SIZE=10737418240 python doc/datatype_sweep/sweep_dtype.py --steps 48,24,12 --out-dir perf_bf16

# pareto charts
python doc/datatype_sweep/make_pareto.py
```

Artifacts: `sweep_results.json` / `.csv`, `selected_precision_config.json` (shipping bf16 policy plus
the rejected bfp8 candidate), `agreement_bf8_vs_bf16.json`, `pareto_argmax_vs_latency.png`,
`pareto_accept_vs_latency.png`. Raw run logs on bhqb at `/home/zni/dg-agent-runs/{gate.log,perf.log}`
with artifacts under `/home/zni/dg-agent-runs/dtsweep/`.

## Limitations (each strengthens the verdict)

- Agreement was measured at **16 steps** (the 100-t/s-relevant regime). At 48 steps divergence would
  be worse from more compounding, so 16 steps is **generous** to bfp8 and it still fails.
- No HF anchor: the gate is bf16-vs-bfp8 as specified. Because the comparison is deterministic,
  bf16-vs-bf16 would be bit-exact, so all observed drift is unambiguously attributable to bfp8.
- BFP4 was not swept: bfp8 already fails and lower precision would fail worse. Not worth device time
  until #48291 headroom exists.
