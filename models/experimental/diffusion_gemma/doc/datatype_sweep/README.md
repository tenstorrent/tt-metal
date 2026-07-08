# DiffusionGemma datatype sweep — bf8 experts (dg-07 / #47465 / #47475)

**Verdict: bf8 (bfloat8_b) MoE experts REJECTED — fails the diffusion-decision fidelity gate. Default stays bf16.**

The MoE experts are 88.6% of weight DRAM (11.6 GiB/chip) and ~35% of the compute-bound denoise
step, so bfp8 experts are the single biggest remaining in-repo speed lever. This sweep adds a
DG-local opt-in knob (`DG_EXPERTS_BFP8=1` / `DG_EXPERTS_DTYPE=bfp8`, via `tt/precision_build.py`
+ the `checkpoint.py` builder route — **no shared gemma4 edits**; default with no knob is bf16,
byte-identical to before) and measures the decision-fidelity trade with `decision_agreement.py`.

## Fidelity gate (bf8 vs bf16 experts, 16 steps, decision-aware metric per dg-07)

| metric | bf8 vs bf16 | gate |
|---|---:|---|
| mean Gumbel-max **argmax agreement** | **0.604** | ✗ (want ≥ ~0.95) |
| mean **accept/renoise IoU** | 0.501 | ✗ |
| **committed-token match** | **0.227** | ✗ |
| mean entropy PCC | 0.631 | ✗ |
| mean canvas agreement | 0.906 | — |
| generated text | coherent for ~1 sentence then **degenerates into repetition/garbage** | ✗ |

bf8 experts flip ~40% of the per-step diffusion decisions and only 23% of committed tokens match —
the model's sparse-MoE fidelity is already marginal on Blackhole (#48291, ~0.84 PCC), and bf8 pushes
it over the edge. This is exactly why DG deliberately loads experts in bf16.

## Speed context (bf16 baseline, traced, 30L — for reference)
`traced_tuned` @48 = 18.2 t/s · @24 = 31.5 t/s · @12 = 54.6 t/s (model-faithful requires 48 steps).
Measured bf8-experts speed: @48 = 19.83 t/s (vs bf16 18.18), @24 = 33.99 (vs 31.5) — only **~9% faster**
(the denoise step is not purely weight-bound, so halving expert bytes buys little). So bf8 is doubly
not worth it: it fails the fidelity gate AND gives only ~9%. @48 stays compute-bound; 100 t/s is only
reachable in the short-step regime.

## Conclusion
bf8 experts is the last in-repo speed lever and it fails the fidelity gate. **Model-faithful @48
throughput ceiling ≈ 18 t/s stands.** Reaching 100 t/s at model-faithful quality needs out-of-gate
work: an upstream fused/higher-fidelity sparse-MoE kernel (would also let bf8 pass), or fewer denoise
steps (blocked by #48291), or accepting the short-step (lower-quality) regime where 100 t/s already holds.
