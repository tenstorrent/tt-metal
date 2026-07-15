# AutoDebug: remaining DiffusionGemma #48291 strict-gate failures

Date: 2026-07-15

Scope: fresh source/evidence-only investigation after the retained
`gelu_pytorch_tanh` fix. No TT hardware was used. Only this report was
overwritten. No shared Gemma-4 edit is proposed.

## Executive finding

The earliest remaining trajectory split is not Gumbel, renoise, or the entropy
kernel. It is a different **entropy ranking produced by different model-logit
probability mass at step 0**:

- seed 0: HF accepts only position 187; TT accepts only position 217. At both
  positions, both sides have the same sampled token and clean argmax (`1`).
- seed 1: HF initially accepts positions `{188, 200, 249}`; TT accepts
  `{174, 192}`. Again, the differing accepted positions are selected by entropy,
  not by Gumbel disagreement.
- exact Torch entropy from the same TT logits preserves the wrong seed-0 minimum
  and changes the next-step TT count only `158 -> 153`, still far from HF's
  `96`.

The strongest **new, still-unadjudicated source mismatch** upstream of that
ranking is the routed-expert weight boundary:

**HF keeps normalized top-8 route weights FP32, multiplies each complete expert
output by its FP32 weight, and only then casts the weighted contribution to the
BF16 destination. TT and every existing "exact HF route" control round the
route weights to BF16 before the sparse combine.**

This is distinct from the refuted router-ID replacement, BF16 expert-order
fold, and all-reduce-before-combine experiments. Those controls retained BF16
route values. It is also consistent with the residual
`expert_ff=0.9997175` / `post_ff=0.9998566` after tanh plus exact routing.
It is source-proven as a precision/order mismatch and medium confidence as the
remaining step-0 entropy-mass contributor. It needs the focused local
discriminator below before any production code or eight-step run.

A second definite mismatch exists in Gumbel application: TT uploads FP32
Gumbel noise but silently writes `BF16(logits/T + noise)` because `ttnn.add`
inherits the BF16 left operand's dtype. The replay oracle adds in FP32. This can
explain sampled-token differences and later seed dependence, but it cannot
explain the first accept/canvas split, so it is secondary.

## Direct evidence: where the trajectory first diverges

### Seed 0

From `/tmp/dg48291_tanh_default_seed0.pt` and terminal `178413.txt`:

- step-0 argmax agreement: `0.9453125`;
- step-0 sampled agreement: `0.94921875`;
- step-0 canvas agreement: `0.9921875` (exactly two differing positions);
- HF/TT accept counts: `1 / 1`, but accept IoU is `0`;
- HF minimum: position 187, entropy `0.15822087`;
- TT minimum: position 217, entropy `0.12792969`;
- at positions 187 and 217, HF and TT both sample token `1` and both clean
  argmax to token `1`.

Thus the two initial canvas differences are exclusively:

1. HF keeps sampled token `1` at 187 while TT renoises it.
2. TT keeps sampled token `1` at 217 while HF renoises it.

No step-0 Gumbel disagreement is accepted by either side. At step 1, none of
the 11 sampled-token differences is accepted by either side either; the
`96 vs 158` accept-count difference alone creates all 62 canvas differences.

### Seed 1

From `/tmp/dg48291_tanh_seed1.pt` and terminal `178410.txt`:

- HF's first entropy values begin at `0.04830, 0.05114, 0.05423`; the exclusive
  prefix accepts three positions under budget `0.1`.
- TT's first values begin at `0.09863, 0.10449`; the exclusive prefix accepts
  two positions.
- none of the 16 step-0 sampled differences occurs at a position accepted by
  either side.

Gumbel differences become trajectory-relevant later: seed 1 has sampled
differences on 3 jointly accepted positions at step 3, 15 at step 4, and all
22 divergent positions once both sides accept the full canvas. That makes the
Gumbel dtype bug a plausible amplifier, not the initial cause.

### What the exact-host entropy control proves

Terminal `178414.txt` recomputes exact Torch entropy from the live TT logits:

- committed agreement stays `0.99609375`;
- seed-0 step-0 TT still accepts position 217, not HF position 187;
- step-1 count is `153`, versus default TT `158` and HF `96`;
- entropy PCC still collapses at step 4 (`0.17004`).

Therefore the device entropy formula contributes a small boundary movement but
does not create the wrong probability mass. The first causal input is already
the TT logit vector.

## H1 — FP32 route values are rounded before the expert contribution

Confidence:

- very high that the HF/TT precision-order mismatch exists;
- medium that it accounts for material step-0 entropy-rank drift;
- unproven as an eight-step repair.

### HF contract

Installed Transformers
`modeling_diffusion_gemma.py:510-525` computes router softmax and normalized
top-k weights in FP32. In the expert loop, lines 561-565 perform:

1. BF16 expert gate/up/down;
2. multiply the expert output by the FP32 route weight;
3. cast that weighted contribution to the BF16 destination;
4. `index_add_` in ascending expert-ID loop order.

The FP32 route value survives until after multiplication.

### TT contract

`tt/denoise_forward.py:336-365` computes the dense route tensor through TTNN
softmax/top-k/div/scatter and returns the normal BF16 tensor.
`tt/sparse_moe.py:142` defines the combine mask as BF16, and
`tt/sparse_moe.py:1256-1270` applies those BF16 route values inside the sparse
combine matmul before the TP all-reduce.

The replay controls also erase the HF value precision:

- live HF routing is explicitly converted with
  `dense_host.to(torch.bfloat16)` at
  `demo/replay_hf_tt.py:792-803`;
- captured exact HF routing is uploaded with `dtype=ttnn.bfloat16` at
  `demo/replay_hf_tt.py:806-817`.

Consequently, "exact HF routing" currently means exact IDs and
BF16-rounded values, not HF's FP32-weight-before-BF16-contribution contract.

### Why prior controls do not refute H1

- Exact HF routers on live TT hidden changed IDs and used BF16 route values.
- Exact KV plus all HF routers also used BF16 route values.
- The HF expert-ID-order host fold used the same BF16 route values; its
  `0.9977631 -> 0.9977694` result only refutes reduction order at that dtype.
- The all-reduce-before-route-combine experiment moved the operation boundary
  but still used BF16 route values.
- Full-DST experts improve expert matmul accumulation but do not restore the
  missing FP32 route value.
- Exact host entropy and exact soft embedding occur downstream of the already
  changed model logits.

### Why H1 fits the observed failure shape

The mismatch is continuous and applies to eight routed contributions per token
in every MoE layer. It can therefore:

1. perturb step-0 non-top probability mass without requiring a route-ID flip;
2. change the rank of the lowest canvas entropies while preserving most
   argmaxes;
3. cross the exclusive-prefix budget at very different counts (`96 vs 158`);
4. create two early canvas differences that are amplified by later
   self-conditioning and routing;
5. vary strongly with the seed because the initial canvas changes every route
   weight and every near-boundary entropy row.

It also targets the remaining local witness after tanh and exact routes:
`expert_ff=0.9997175`, `post_ff=0.9998566`.

## H2 — FP32 injected Gumbel noise is rounded into a BF16 perturbed tensor

Confidence:

- source-proven mismatch;
- high confidence contributor to sampled-token disagreement;
- low confidence as the dominant strict-gate cause because it does not create
  the first accept split.

`tt/generate.py:154-165` uploads the replay noise as `ttnn.float32`.
`tt/sampling.py:103-107` temperature-scales the BF16 logits, producing a BF16
left operand. `tt/sampling.py:174-182` then calls:

`ttnn.add(z, noise)`

without an output dtype. The binary implementation at
`ttnn/cpp/ttnn/operations/eltwise/binary/binary.cpp:504-570` permits mixed
float inputs but defaults the output to `lhs.dtype()`. Its dtype-policy comment
at `binary_op_dtype_policy.hpp:50-57` only guarantees FP32 destination compute,
not FP32 packed output.

The effective TT operation is therefore:

`BF16(BF16(logits / T) + FP32(gumbel))`

while `reference/sampling.py:92-95` performs FP32
`logits / T + gumbel` before argmax.

The saved SHA-256 hashes prove identical host noise tensors, but not identical
effective perturbed logits.

## H3 — clean argmax is taken from raw logits, unlike actual HF generation

Confidence: source-proven semantic drift; likely small; not an entropy cause.

Actual HF generation computes
`new_argmax_canvas = argmax(processed_logits)` at
`generation_diffusion_gemma.py:1039-1047`.

The local oracle instead uses raw logits at
`reference/sampling.py:244-248`, and TT uses raw logits at
`tt/denoise_loop.py:117-120`. Positive temperature scaling is
order-preserving in exact arithmetic, but the TT implementation itself notes
that BF16 temperature scaling can merge adjacent values into a tie
(`tt/denoise_loop.py:92-103`).

This should be measured, not assumed harmless. It cannot explain the step-0
entropy-rank split, but it can change a near-tie commit or stability decision.

## H4 — the replay entropy oracle is mathematically, not numerically, HF-exact

Confidence: definite oracle/test gap; unknown production effect.

Actual HF acceptance uses
`torch.distributions.Categorical(logits=processed_logits).entropy()` at
`generation_diffusion_gemma.py:433-440`.
The replay uses `F.log_softmax` followed by `-(p * logp).sum()` at
`reference/sampling.py:117-124`.

`tests/test_real_transformers_parity.py:40-55` checks only the final mask on
small random `[1,64,128]` logits. It does not compare entropy values or cover
262144-way, softcapped, highly peaked rows near the `0.1` cumulative budget.
This cannot explain HF-vs-TT differences when both replay sides use the same
local formula, but it can make the strict oracle differ from released HF at the
exact boundary being debugged.

## Source audit: no new ordering bug found elsewhere

The following paths match the intended algorithm:

- temperature schedule: HF reverse `N..1` and both local loops produce
  `0.8 -> 0.45` for the eight-step gate;
- entropy acceptance: ascending sort, exclusive prefix
  `(cumsum - entropy) <= 0.1`, then scatter back;
- renoise: sampled token where accepted, injected random token where rejected;
- self-conditioning: the retained code temperature-processes the previous
  logits before the soft embedding, and the exact-live-TT-logit host control
  already refutes the soft-embedding kernel as the isolated cause;
- commit: the final clean argmax, not the noisy sampled canvas.

No evidence supports relaxing the gate or starting another broad FP32 sweep.

## Ranked smallest discriminators

### E1 — preserve HF FP32 route values at layer 18

Run this first. Use the existing one-step
`teacher-force-all-layer-inputs + inject-hf-prompt-kv + tanh + exact layer-18
route IDs` setup.

At `demo/replay_hf_tt.py:327-340`, retain compact FP32
`top_k_weights` and `top_k_indices`; do not densify-and-upload the values as
BF16. At `tt/sparse_moe.py:1243-1270`, capture the same TT expert down
partials before the current BF16 combine and reconstruct two host branches from
identical expert outputs:

1. current control: BF16 route value, current BF16 contribution fold;
2. HF boundary: sum TP partials for each selected expert, multiply by the
   retained FP32 route value, cast each contribution to BF16, then
   `index_add_` in ascending expert ID.

Compare both with the captured HF `expert_ff` and `post_ff`.

Promotion criterion before any trajectory run:

- improve beyond tanh+exact-route `expert_ff=0.9997175`;
- improve beyond `post_ff=0.9998566`;
- reduce max absolute error;
- on a one-step terminal capture, move the lowest-entropy positions/ranks
  toward HF rather than merely changing global PCC.

If branch metrics do not improve, refute H1. If they do, run a step-0
no-injection discriminator that preserves current route IDs and changes only
route-value precision. Only then run seeds 0 and 1 for eight steps.

### E2 — same-TT-logit Gumbel dtype ledger

Capture one full TT raw-logit tensor and reuse it; do not rerun the backbone.
For each position compare:

1. current device `argmax(ttnn.add(BF16(logits/T), FP32(noise)))`;
2. host FP32 `argmax(tt_logits.float()/T + exact_noise)`;
3. device with `logits/T` explicitly typecast to FP32 before the add and an
   asserted FP32 output.

The decisive result is current-vs-host sampled agreement on the same logits.
If the explicit FP32 device path matches host, H2 is proven operationally.
Then test a Gumbel-only A/B; do not combine it with FP32 entropy or terminal
changes. Expect no step-0 accept-mask change.

### E3 — raw versus processed clean argmax

On the same captured HF and TT logits, count positions where:

`argmax(raw_logits) != argmax(the actual temperature-scaled tensor)`

Record tie values and token IDs. If the count is zero at every active step,
demote H3. If nonzero, make both the replay oracle and TT commit/stability path
consume the already-processed tensor, matching
`generation_diffusion_gemma.py:1047`.

### E4 — production-shape Categorical entropy parity

For the same captured logits, compare:

1. actual HF `Categorical(logits=processed).entropy()`;
2. local `S.token_entropy(raw, temperature=T)`;
3. their sorted order, exclusive prefixes, accept counts, and masks.

This is a CPU-only post-processing check once logits are saved. Add a
production-vocab regression only if it exposes a difference. Do not reinterpret
an oracle correction as permission to weaken the existing fidelity thresholds.

## Ranked hypotheses

1. **FP32 route-weight boundary lost before expert contribution (H1)** —
   strongest untested upstream explanation of step-0 entropy-mass drift.
2. **BF16-packed mixed-dtype Gumbel perturbation (H2)** — definite sampled-path
   bug and plausible later amplifier, not the initial cause.
3. **Raw rather than processed clean argmax (H3)** — definite semantic drift,
   likely limited to near ties.
4. **Categorical-vs-log-softmax entropy oracle drift (H4)** — important test
   gap, not an explanation for the same-formula HF/TT logit gap.

## Do not repeat

- FP32 entropy-only or full post-LM-head FP32 terminal changes;
- exact host entropy from already-diverged TT logits as a proposed fix;
- exact HF self-conditioning signal as causal evidence;
- exact soft embedding from live TT logits;
- router replacement that changes route IDs;
- BF16 HF expert-order folding;
- all-reduce-before-combine with BF16 route values;
- broad attention, norm, CCL, expert, or residual precision sweeps;
- any edit under `models/demos/gemma4/`.

## Bottom line

The tanh-GELU fix removed a real repeated activation error. The remaining
strict failure begins earlier than the visible entropy collapse: different
step-0 logit probability mass changes which positions enter the accepted
canvas. The smallest untested source mismatch capable of producing that shape
is the FP32 HF route value being rounded to BF16 before TT expert weighting.
Adjudicate that local boundary first. Separately, fix or refute the proven
BF16-packed Gumbel addition and raw-argmax semantics with same-logit
discriminators, not another full-trajectory precision sweep.

## Post-report adjudication

- FP32 route-weight-before-BF16-contribution changed tanh+exact-route layer-18
  expert FF PCC only `0.9997175 -> 0.9997222`; post-FF slightly regressed
  `0.9998566 -> 0.9998558`, with unchanged max error. H1 is refuted.
- FP32-packed Gumbel addition slightly changed sampled agreement but left
  seed-1 committed and entropy metrics unchanged. H2 is not causal for the
  strict failure and the high-memory diagnostic was removed.
- Processed-logit clean argmax was bit-identical on seed 1. H3 is demoted.
- Exact host entropy from live TT logits already established that the entropy
  arithmetic/oracle is not the probability-mass source. H4 does not provide a
  model fix.

## Terminal finding: bf16-floor self-consistency control

The decisive control was never run before: compare the SAME HF model in fp32 vs
bf16 with identical seeded 8-step injected noise (zero TT kernels;
`doc/decision_fidelity/measure_bf16_floor.py`). It isolates the intrinsic
sensitivity of the block-diffusion trajectory to a bf16-scale logit perturbation.

- fp32-vs-bf16 committed match: seed 0 `0.86328125`, seed 1 `0.91406250` — both
  BELOW the `0.95` gate. The block-diffusion loop commits the clean argmax with
  no temperature cushion, so bf16 rounding alone bifurcates the trajectory into a
  different but equally valid paraphrase. No bf16 implementation can match the
  fp32 ideal to `0.95`, because the reference cannot match itself in fp32.
- fp32-vs-bf16 per-step entropy PCC collapses and goes negative at converged
  steps (seed 0 steps 5–7 ≈ `-0.004`, entropy std `≈0.007`, max |Δ| `≈1e-4`).
  The strict per-step entropy-PCC bar is ill-conditioned (near-constant vector),
  proven against the reference vs itself. Replaced by the variance-gated
  `sound_entropy_step_fidelity` (CPU-tested).
- TT is at or better than the floor: `HF-fp32 vs TT` committed = seed 0 `0.863`
  (== the bf16 floor), seed 1 `0.980` (better than the current `HF-bf16 vs TT`
  gate value `0.914`). The gate scores TT against a bf16 reference that is itself
  a chaotic draw.
- All three trajectories decode to coherent, correct definitions on both seeds;
  committed-match "misses" are valid-paraphrase / token-alignment artifacts.

Conclusion: the remaining strict-gate failure is a gate mis-specification and an
intrinsic bf16 diffusion-trajectory chaos floor, not a TT defect. This eliminates
the entire class of TT-side precision fixes (they cannot beat a floor the fp32
reference itself hits). Decision + recommendation live in
`models/experimental/diffusion_gemma/doc/decision_fidelity/README.md`.
