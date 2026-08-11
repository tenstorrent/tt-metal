# Isolated single-op sweep — gemma-4-26B residual `rms_norm`, 79 configurations

**Result: there is no tt-metal bug here, and the veto that cost the run 6,055 µs/model has no numerical
justification.** The op is clean at every core count, every rectangle, every accumulation blocking and every
dynamic range tested — the largest PCC deviation across the whole matrix is **7.3 × 10⁻⁷**, which is
**6,879× too small** to explain the 5.06 × 10⁻³ whole-layer drop it was blamed for.

This closes action 3 of [`PCC-BY-GRID`](ADVCHAL-V3-PCC-BY-GRID.md) and action 4 of
[`OP-BY-OP-VS-V2`](ADVCHAL-V3-OP-BY-OP-VS-V2.md), and **retracts** the candidate defect they proposed.

## What was run

Ran 2026-08-11 on the same host, device idle, `.challenger-STOP` and `.dense-STOP` in place, nothing else
holding `/dev/tenstorrent*`. Scripts: `run-config/norm_grid_sweep.py` and `norm_grid_sweep2.py`, raw output in
`run-config/norm_grid_sweep{,2}-results.json`.

The op is reconstructed exactly as `tt/optimized_decoder.py::_rms_norm` builds it:

| | |
|---|---|
| tensor | `1x1x1x2816` bf16 = **88 tiles wide** (from `shard_advise/sliding_attention/final_ir.mlir`) |
| weight | `1x1x1x2816` bf16, resharded to the same config — **and `None`**, because two of the layer's norms have `operandSegmentSizes <1,0,0>` |
| epsilon | `9.99999997e-7`, from the IR |
| compute config | HiFi4, `math_approx_mode=False`, **`fp32_dest_acc_en=True`**, `packer_l1_acc=False` |
| memory config | `create_sharded_memory_config((32, 2816/C), CoreGrid(x,y), WIDTH, ROW_MAJOR, use_height_and_width_as_shard_shape=True)` |
| program config | `LayerNormShardedMultiCoreProgramConfig(grid, subblock_w, block_h=1, block_w=88/C, inplace=False)` |
| reference | float64 torch `x/√(mean(x²)+ε) · w`, inputs pre-rounded to bf16 |
| device grid | 11 × 10 = 110 cores (Blackhole) |

**The model's grid rule is `(C,1)` for C ≤ 11 and `(11, C//11)` above** — so the rectangles in sweep B are shapes
the model can never emit, which is what makes them worth measuring.

## Sweep A — the divisor ladder, including the rung the run never measured

88 tiles ⇒ the exact divisors are 1, 2, 4, 8, 11, 22, 44, 88, and **all six rungs the run swept are exact
divisors**; 88 was the only one missing.

| cores | grid | `block_w` | `subblock_w` | **PCC** | µs/op |
|---:|---|---:|---:|---:|---:|
| — *(interleaved, the incumbent)* | — | — | — | **0.99999876** | 44.68 |
| 1 | 1×1 | 88 | 4 | **REJECTED by tt-metal** — see §"other findings" | — |
| 2 | 2×1 | 44 | 4 | 0.99999876 | 23.62 |
| 4 | 4×1 | 22 | 2 | 0.99999876 | 24.07 |
| **8** | 8×1 | 11 | 1 | **0.99999876** | 23.65 |
| **11** | 11×1 | 8 | 4 | **0.99999876** | 24.19 |
| 22 | 11×2 | 4 | 4 | 0.99999876 | 31.91 |
| 44 | 11×4 | 2 | 2 | 0.99999876 | 42.04 |
| **88** *(advised; v2 shipped it)* | 11×8 | 1 | 1 | **0.99999876** | **69.15** |

**Every grid returns the identical PCC to eight decimals, and identical to the interleaved incumbent.** The two
grids the run's decision turned on — **11**, which was measured at 0.99457 in the model and vetoed the kind, and
**88**, which v2 shipped and was never tried — are **numerically indistinguishable** from each other and from
doing nothing.

## Sweep B — same core count, different rectangle

The question was whether the *shape* of the grid matters, not just the count.

| cores | model's shape | alternative | PCC identical? | µs (model shape → alt) |
|---:|---|---|:--|---|
| 2 | 2×1 | 1×2 | ✅ | 23.62 → 38.9 |
| 4 | 4×1 | **2×2** | ✅ (Δ ≤ 1.1 × 10⁻⁷ on one input) | 24.07 → **19.71 — fastest of all 79** |
| 8 | 8×1 | 4×2 | ✅ | 23.65 → 23.07 |
| 8 | 8×1 | 2×4 | ✅ | 23.65 → 22.77 |

**Grid shape does not affect correctness.** It affects speed by a few per cent, and the one config that is
consistently fastest — **4 cores as 2×2, 19.71 µs** — is a rectangle the model's `(C,1)`/`(11,h)` rule cannot
produce.

## Sweep C — same grid, forced `subblock_w`

`subblock_w` is chosen by the model as the first of (4, 2, 1) dividing `block_w`, so 8 cores gets 1 while 11 gets
4. That asymmetry was the leading hypothesis for a placement-dependent numerical difference.

| grid | `block_w` | `subblock_w` tested | PCC |
|---|---:|---|---:|
| 11×1 | 8 | 4 *(model)*, 2, 1 | **all 0.99999870** |
| 11×2 | 4 | 4 *(model)*, 2, 1 | **all 0.99999870** |
| 11×4 | 2 | 2 *(model)*, 1 | **all 0.99999861** |

**Accumulation blocking is irrelevant to the result.** Hypothesis dead.

## Sweep E/F — `weight=None` and extreme dynamic range

Two variables sweep A could not see. `weight=None` matters because the **MoE router norm** takes that path
(`_router_weights` calls `self._rms_norm(residual, None)`) and its output feeds `ttnn.topk`.

| weight | input | max/min ratio | interleaved PCC | **worst sharded PCC over all 10 grids** | worst |Δ| |
|---|---|---:|---:|---:|---:|
| weighted | unit | 1 × 10⁴ | 0.99999876 | 0.99999876 | **0** |
| weighted | spiked ×3000 | 1 × 10⁷ | 0.99999976 | 0.99999964 | 1.1 × 10⁻⁷ |
| weighted | log-spread | 1 × 10¹² | 0.99999941 | 0.99999952 *(better)* | 1.1 × 10⁻⁷ |
| **none** | unit | 1 × 10⁴ | **1.00000000** | **1.00000000** | **0 — bit-exact** |
| none | spiked ×3000 | 1 × 10⁷ | 0.99999767 | 0.99999836 *(better)* | 7.3 × 10⁻⁷ |
| none | log-spread | 1 × 10¹² | 0.99999903 | 0.99999892 | 1.1 × 10⁻⁷ |

**60 sharded configurations measured. Minimum PCC anywhere in the matrix: 0.99999836.** With no weight and a
unit-scale input the sharded reduction is **bit-exact** against float64 at every grid. Three of the six rows have
the sharded path *more* accurate than interleaved.

---

# The conclusion, and what it changes

| | |
|---|---|
| whole-layer drop the veto was based on | 0.99962801 → 0.99457296 = **5.055 × 10⁻³** |
| largest PCC deviation this op can produce, anywhere | **7.3 × 10⁻⁷** |
| ratio | **6,879×** |

## 1. The candidate tt-metal defect is retracted

[`PCC-BY-GRID`](ADVCHAL-V3-PCC-BY-GRID.md) §4 proposed that `rms_norm` widened off one core degrades PCC
non-monotonically in the core count, on the strength of three cells clustering at ~0.9946 and v2's 88 cores
passing at 0.9990. **The clustering is real and the explanation was wrong.** All three cells ran the *same*
candidate policy, so they are one observation reported three times, not three independent reproductions — and
the op they blamed is clean at every grid those cells could have used. **No bug to file.**

## 2. The veto has no numerical justification — this is now a measurement, not an argument

`DEVIATIONS` §3.2a established that one PCC sample rejected 17 sliding measurements. This sweep establishes the
stronger claim: **the op under test provably cannot have caused the number that triggered the veto.** So the
veto was not "correct but over-broad" — the attribution behind it was wrong before the scope question even
arises.

## 3. Where the 5 × 10⁻³ actually comes from — ⚠ the follow-up answered this, and not as predicted

**[`PCC-DROP-ISOLATION`](ADVCHAL-V3-PCC-DROP-ISOLATION.md) tested the routing hypothesis below with the real
layer-0 router weights and it is too rare to be the answer: 2 flips in 168 trials, one slot of eight, against an
8th-to-9th logit gap of 1.6–2.2.** It also established that **11 cores and 88 cores are bit-identical** element
for element, and that v2's and v3's differing weight placement is bit-identical too. The actual answer is that
the sliding kind's `oracle_pcc` is a **hardcoded literal** in `build_evidence.py` and its `oracle_passed` is the
expression `kind == "full_attention"`, with **no oracle log committed anywhere in the cell.** Read that file
instead of this section.

The hypothesis is kept below because the mechanism it describes is real (a `topk` over 128 experts does flip under
a 1-ULP perturbation, on ~1 % of tokens) and it is still a reason to gate placement on the op's own output — just
not the explanation for this number.

## 3 (superseded). The routing hypothesis, and why that bar is the wrong gate anyway

The 0.99457 is a **whole-decoder-layer** PCC, and the norm change contributes ≤ 10⁻⁷ of it. The remaining
plausible mechanism is **discontinuity, not rounding**: this is a sparse-MoE layer, the router norm's output
feeds `ttnn.topk`, and a 10⁻⁷ perturbation that lands near a routing boundary **flips an expert selection**,
which changes the layer output by far more than any arithmetic error. That predicts exactly what was seen —
a step from 0.9996 to 0.9946 rather than a gradual slide, and the same value for every rung, because every rung
flips the same selection.

> **A whole-layer PCC bar of 0.995 on a sparse-MoE decoder measures expert-selection agreement, not arithmetic
> accuracy, so it cannot gate a placement change.** Placement changes must be gated on the **op's own** output,
> which is what this sweep does and what `op_under_test` was added to make possible.

This is testable without weights and is the natural next experiment: perturb the router input by 10⁻⁷ and count
how many of the layer's expert selections change.

**And v2's own oracle file confirms the sweep independently.**
`oracle/norm88/pcc_layer0_sliding_attention_shared1.json` records `decode_pcc = 0.99962934` for the **88-core**
configuration, against v3's **untouched incumbent** at **0.99962801** — a difference of **1.3 × 10⁻⁶**. So a
whole-layer measurement of this same re-grid at a different core count shows **no degradation at all**, which is
what the sweep predicts and what the 11-core 0.99457 contradicts.

⚠ v2's oracle also records its scope — `decode_current_pos=32, sequence_length=32, shared_physical_cache=true`,
with `prefill_pcc=0.99880996` reported separately. **v3's sliding oracle records the value and no scope**
(RUN-LOG **P4**). So two explanations survive — a routing flip that 11 triggers and 88 does not, or two oracles
evaluating different positions — and **the artefacts cannot distinguish them.** Either way the veto was
unjustified; which of the two it was decides whether the next run needs a scope assertion or a routing
experiment. **Both are cheap; do both.**

## 4. The advisor's advised grid is the worst legal choice on this op

The advisor advised `l1/width_sharded/1x88`. On this op, in isolation, **88 cores costs 69.15 µs against the
interleaved incumbent's 44.68** — the advice is not merely suboptimal, it is **55 % slower than doing nothing**,
and it is the slowest of all 79 configurations. The cheapest is **4 cores at 2×2, 19.71 µs — 2.3× faster than
the advice and 2.3× faster than the incumbent.**

That is consistent with the advisor's objective having **no latency term** and a `coreCount` tiebreaker that
`NormalizationRules.cpp` overrides with the *input* operand's grid volume: nothing in its ordering can prefer
4 cores to 88. **`getOpRuntime` exists and is never consulted.**

⚠ **Do not read this as "the model should use 4 cores."** These are isolated single-op costs including the
regather to DRAM; in the layer the neighbouring ops share the grid, and the run's end-to-end ladder found 8 cores
best on `full_attention` and 11 on `sliding_attention` with 2 and 4 *worse*. The isolated ranking and the
end-to-end ranking disagree, which is itself the finding: **per-op cost does not compose, so a per-op advisor
cannot be trusted to pick the grid — which is the whole argument for the ladder.**

## 5. Other findings worth keeping

- **1 core width-sharded is infeasible on this shape.** All three `subblock_w` variants fail identically:
  `Statically allocated circular buffers in program N clash with L1 buffers on core range [0-0 - 0-0]. L1 buffer
  allocated at 1032192 and static circular buffer region ends at 1041152.` `block_w=88` on a single core does not
  fit L1. So the "1 core" incumbent in every ladder table is **interleaved**, not 1-core-sharded — the ladder's
  bottom rung is a different memory layout, not a narrower grid. Worth stating in `SKILL.md`, since it means the
  first ladder step changes two things at once.
- **gemma-4-26B's real weights are not on this host** — `/huggingface/hub/models--google--gemma-4-26B-A4B-it`
  is **28 KB, `config.json` only**. Every gemma cell in both corpora reports
  `oracle_weights: "real HuggingFace checkpoint weights"`. That confirms RUN-LOG **P5** as a real defect: the
  provenance string is unverified, and `oracle_weights` should be asserted against the checkpoint's size on disk.
  It does not invalidate the PCC *comparisons* (both sides use the same tensors) but it does invalidate the word
  "real", and it bears directly on the unresolved P3 contradiction.
- **Timing is monotonic in core count above 4** (23.6 → 24.1 → 31.9 → 42.0 → 69.2 µs for 8/11/22/44/88), i.e.
  dominated by the cross-core reduction and the regather, not by the per-core arithmetic. For an op this small,
  more cores is strictly worse in isolation.

# Actions, revised

1. ~~Isolated single-op sweep~~ — **done, this file.**
2. ~~File a tt-metal bug for `rms_norm` by core count~~ — **retracted, no bug.**
3. **`op_under_test` and `oracle_pcc` become CRITICAL per screened candidate** — unchanged and now better
   motivated: the whole-layer number was attributed to an op that cannot have produced it, and nothing in the
   artefacts contradicted that because the field was advisory.
4. **A placement candidate must be gated on the op's own output, not only on the layer's PCC** — new, and it is
   the substantive change this sweep argues for. The layer-level bar stays as a final check.
5. **An oracle verdict binds only the configuration it was measured on** (`DEVIATIONS` §3.2a) — unchanged.
6. **Assert `oracle_weights: real` against the checkpoint on disk** — new, from §5.
7. ~~Router-perturbation experiment~~ — **done**, [`PCC-DROP-ISOLATION`](ADVCHAL-V3-PCC-DROP-ISOLATION.md).
   Routing flips on ~1 % of tokens; not the explanation. The explanation is the hardcoded verdict, and the actions
   move there.
