# DiffusionGemma dg-07 datatype sweep (#47475 quant / #47465 perf)

**Verdict: bfp8 MoE experts FAIL the diffusion-decision-fidelity gate. DiffusionGemma keeps
bf16 experts. The 17.9–18.2 t/s @48 bf16 baseline stands.** The DG-local bfp8 knob is landed
**off by default** for reuse if #48291 ever creates fidelity headroom.

Hardware: QB2 / `bh-qbge-06` / P150x4, mesh `(1,4)`, TP=4. Date: 2026-07-08. Full 30 layers.

## The lever

MoE expert `gate/up/down` weights are ~88.6% of on-device weight DRAM (~11.6 GiB/chip) and a
large fraction of the compute-bound denoise step. DiffusionGemma loads them at **bf16** (the
gemma4 model-wide default; there is no `precision_overrides.json` entry for
`gemma-4-26B-A4B-it`). Dropping the experts to **`bfloat8_b`** roughly halves the expert DRAM
and speeds the batched-expert matmuls. It is **accuracy-gated** — bfp8 small-probability drift
can flip a diffusion accept/renoise decision, and #48291 already leaves the model with no
fidelity headroom.

## How the knob works (DG-local; no shared-backbone edits)

`models/experimental/diffusion_gemma/tt/precision_build.py::create_tt_model_dg` is wired as the
default `create_model_fn` in `checkpoint.py::build_tt_model_from_checkpoint_inputs`. With no knob
it delegates to the shared `create_tt_model` **unchanged**. With `DG_EXPERTS_BFP8=1` (or
`DG_EXPERTS_DTYPE=bfp8`) it replicates `create_tt_model` (copy-shared-into-DG convention, since
we may not edit the shared constructor) and passes `Gemma4Precision({"experts": bfloat8_b})` into
`Gemma4Model`. **Only the expert weights change**; router, attention, shared MLP, embedding,
lm_head, KV-cache, and the entire decision path (logits softcap, softmax→probability, entropy,
Gumbel-max argmax, entropy-budget accept/renoise) stay bf16/fp32. Expert cache filenames carry
the dtype suffix (`_bfp8_dtype_BFLOAT8_B`), so bf16 and bfp8 caches coexist.

`git diff main -- models/demos/gemma4/` is empty (verified).

## Accuracy = diffusion decisions, not top-1/top-5

The metric is the diffusion **decision** vs the **bf16-experts reference** (NOT AIME24 teacher
forcing). Measured deterministically: a fixed seeded initial canvas + fixed per-step renoise
tokens + clean-argmax sampling + early-halt disabled, so a bf16 run and a bfp8 run differ **only**
by the expert weight dtype. Compared with `tests/trajectory_pcc.compare_trajectories`.

### The bar (and why)

| metric | bar | rationale |
|---|---|---|
| committed clean-argmax agreement vs bf16 | ≥ 0.95 | diffusion commits the CLEAN argmax — no temperature/top-p cushion |
| mean per-step entropy PCC | ≥ 0.95 | entropy depends on the full prob distribution incl. small probs — the exact bfp8 risk |
| mean accept/renoise IoU | ≥ 0.90 | small-prob drift flips accept/renoise even when argmax is unchanged |

Must pass **all three** AND keep generated text no less coherent than bf16. #48291 already puts
the bf16 model at ~50% argmax vs HF, so a reduced-precision candidate must be near
decision-**neutral** vs the bf16 reference — there is no headroom to spend.

## Results

### Decision agreement (bf16 vs bfp8, 16 steps, 30L)

| metric | bfp8 vs bf16 | bar | pass? |
|---|---|---|---|
| committed clean-argmax agreement | **0.227** | ≥0.95 | ❌ |
| mean per-step argmax agreement | 0.604 | — | — |
| step-0 (pure logits) argmax agreement | 0.949 | — | ~5% flip from bfp8 logits alone |
| mean per-step entropy PCC | **0.631** (min 0.036) | ≥0.95 | ❌ |
| mean accept/renoise IoU | **0.501** (min 0.0) | ≥0.90 | ❌ |
| mean canvas agreement | 0.906 | — | — |

bfp8 fails **all three** bars. Step-0 argmax agreement is 0.949 (bfp8 flips ~5% of positions
purely from expert-logit drift); this compounds over the 16-step trajectory to 0.227 committed
agreement — bfp8 changes **~77% of the committed tokens** vs the bf16 reference. Entropy PCC
collapses to 0.036 by step 3, confirming the small-probability drift the bar was designed to
catch.

**Sample text is a wash**, not a save: bfp8 produces an equally-coherent opening sentence then
degenerates into multilingual noise — the same #48291 regime as bf16. Coherence is not destroyed
by bfp8, but the *decision trajectory* diverges massively, and coherence is not the gate.

### DRAM

| config | DRAM used/chip | vs bf16 |
|---|---|---|
| bf16 experts | 13.268 GiB | — |
| bfp8 experts | **7.830 GiB** | **−5.44 GiB (−41%)** |

The 5.44 GiB drop (≈ the predicted expert halving) is the proof bfp8 is genuinely consumed
(also: 90 `*_dtype_BFLOAT8_B` expert cache files written).

### Traced throughput (ranked by traced per-block latency)

| steps | bf16 t/s | bf16 block | bfp8 t/s | bfp8 block | speedup |
|---|---|---|---|---|---|
| 48 | 18.18 | 14.079 s | 19.83 | 12.907 s | **+9.1%** |
| 24 | 31.49 | 8.128 s | 33.99 | 7.532 s | +7.9% |
| 12 | 54.58 | 4.690 s | 57.84 | 4.426 s | +6.0% |

bfp8 buys only ~6–9%. The denoise step is **not** purely weight-bound: the MoE batched matmul is
launch/overhead-limited (~46 GB/s effective, per `path_to_100tps.md`), so halving the expert
bytes only partly speeds it, and the MoE is ~35% of the step while the fixed/terminal overhead is
unchanged.

### 100 t/s reachability

Block model (fit over s12/s24/s48): bfp8 `block ≈ 1.60 + 0.2356·steps`; bf16
`≈ 1.56 + 0.2608·steps`. 100 t/s ⇔ block ≤ 2.56 s ⇔ **~4.1 denoise steps at bfp8** (vs ~3.8 at
bf16). bfp8's ~8% gain shifts the 100-t/s crossover negligibly, and **4 steps is far below any
quality-acceptable step count** (the model needs ≥16–32 steps for even one coherent sentence). So
bfp8 does **not** make 100 t/s reachable at a quality-acceptable step count.

## Pareto interpretation

`pareto_argmax_vs_latency.png` and `pareto_accept_vs_latency.png` plot decision agreement (y, vs
the bf16 reference) against TRACED throughput (x), with the minimum-allowed line and the 100-t/s
target. The bf16 point sits on `agreement = 1.0` (it is the reference / selected point). The bfp8
point sits at 0.227 (committed argmax) / 0.501 (accept IoU) — **below the min-allowed line at
every step count**. bfp8 is faster (right of bf16) but well under the fidelity floor: there is no
Pareto-improving move; the selected point is bf16.

## Commands

```
# decision agreement (run twice, then compare)
python models/experimental/diffusion_gemma/doc/datatype_sweep/decision_agreement.py run \
    --max-denoising-steps 16 --canvas-length 256 --seed 0 --label bf16 --output traj_bf16.pt
DG_EXPERTS_BFP8=1 python .../decision_agreement.py run \
    --max-denoising-steps 16 --canvas-length 256 --seed 0 --label bfp8 --output traj_bfp8.pt
python .../decision_agreement.py compare --ref traj_bf16.pt --cand traj_bfp8.pt --output agreement.json

# traced throughput (run twice: bf16 then DG_EXPERTS_BFP8=1)
DG_TRACE_REGION_SIZE=10737418240 python .../sweep_dtype.py --steps 48,24,12 --out-dir perf_bf16
DG_EXPERTS_BFP8=1 DG_TRACE_REGION_SIZE=10737418240 python .../sweep_dtype.py --steps 48,24,12 --out-dir perf_bfp8

# pareto charts
python models/experimental/diffusion_gemma/doc/datatype_sweep/make_pareto.py
```
(Env prefix for all device runs: `DG_SPARSE_MOE=1 DG_SPARSE_MOE_TUNED=1 DG_DEDUP_ARGMAX=1
DG_CKPT=/home/zni/dg_models/diffusiongemma-26B-A4B-it`.)

## Limitations

- Agreement measured at 16 steps (the 100-t/s-relevant regime). At 48 steps divergence would be
  worse (more compounding), so 16 steps is generous to bfp8 and it still fails.
- No HF anchor: the gate is bf16-vs-bfp8, which is the mission's specified metric. Because the
  comparison is deterministic, bf16-vs-bf16 would be bit-exact (100%); the ONLY perturbation is
  bfp8, so the drift is unambiguously attributable to bfp8. An HF-vs-{bf16,bfp8} anchor is
  optional future strengthening.
- BFP4 was not swept: bfp8 already fails, and BFP4 (lower precision) would fail worse. Not worth
  device time until #48291 headroom exists.

## Artifacts

- `sweep_results.json` / `sweep_results.csv` — full config rows.
- `selected_precision_config.json` — the shipping (bf16) policy + rejected bfp8 candidate.
- `pareto_argmax_vs_latency.png`, `pareto_accept_vs_latency.png` — Pareto charts.
- `decision_agreement.py`, `sweep_dtype.py`, `make_pareto.py` — harnesses.
- `work_log.md` — dated run log.
- Raw run logs on `bhqb`: `/home/zni/dg-agent-runs/{gate.log,perf.log}`; artifacts under
  `/home/zni/dg-agent-runs/dtsweep/`.
