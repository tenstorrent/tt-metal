# GLM-4.7-Flash: one-layer 4-bit expert PCC probe

**Question:** the 30.6B model fits a single 32 GB p150 only if the routed
experts (27.8B of the 30.6B params) are stored at `bfloat4_b`. What does that
cost in accuracy?

**Method:** all 64 routed experts + shared expert + gate of layer 1 (first MoE
layer), real bf16 checkpoint weights, dense-evaluated on one Blackhole chip
(`ttnn.linear`, activations bf16, HiFi2, fp32 dest acc) against a torch fp32
reference. Routing (sigmoid + correction-bias top-4, norm, scale 1.8) computed
host-side in fp32 and held identical across arms, so the arms differ only in
expert weight dtype. Inputs are 512 seeded random-normal tokens: this measures
weight-quantization sensitivity, not activation statistics. Shared expert runs
at `bfloat8_b` in every arm (the planned policy).

Script: `../../probe/expert_bf4_pcc_probe.py` · Raw numbers: `results.json`

## Results

| | experts `bfloat8_b` | experts `bfloat4_b` |
|---|---|---|
| per-expert FFN PCC (min / mean over 64) | 0.999893 / 0.999896 | 0.980913 / 0.981341 |
| routed top-4 sum PCC | 0.999896 | 0.981516 |
| **full MoE block PCC** | **0.999897** | **0.984361** |

Shared expert @ bf8: 0.999898. Host bf4 weight round-trip PCC: 0.9934 min.
Per-expert PCC under bf4 is tightly clustered (0.9809–0.9819): the error is
uniform across experts, so selective mixed-precision experts would not help.

## Read

- `bfloat8_b` experts are accuracy-free but cost ~32 GB of expert weights:
  **does not fit one card**; comfortable on 2–4.
- `bfloat4_b` experts fit one card (~18 GB total weights) at ~0.984 per-MoE-layer
  PCC. That compounds over 46 MoE layers; the DeepSeek/Kimi Galaxy stacks ship
  bf4 routed experts in the same per-layer PCC regime, so it is not
  disqualifying, but the verdict belongs to an end-to-end top-k check
  (the full-model stage's top-5 >= 98% gate), not to this probe.

**Plan implication:** bring the model up as TP over the four p150s with
`bfloat8_b` experts (accuracy-free, trivially fits), and treat single-card
`bfloat4_b` as a follow-on experiment gated on end-to-end top-k accuracy.
