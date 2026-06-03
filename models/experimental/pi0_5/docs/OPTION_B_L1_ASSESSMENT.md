# Option B (TP=8) with L1-resident weights — analytical assessment

**Date**: 2026-06-03
**Status**: **Validated empirically on hardware (commit `221d9f72d34`).**
The hypothesis below was confirmed: TP=8 shrinks the per-chip matmul CB
region from ~733 KB / bank (Option C SigLIP MLP) to ~110 KB / bank, and
L1-resident weights fit comfortably above the shrunk CB region. Init
succeeds, the forward kernels run the L1 matmuls without the
`validate_circular_buffer_region` collision that blocks Option C.
**Companion**: [L1_PLACEMENT_FINDINGS.md](./L1_PLACEMENT_FINDINGS.md)
(why we're asking this question after the Option C investigation).

---

## TL;DR

**Yes, Option B at TP=8 with L1-resident weights is likely to work
where Option C didn't.** The per-chip weight load is similar (~125 MB
vs ~129 MB), but Option B's TP=8 path runs matmuls at **8× smaller
per-chip output dims**, which **shrinks the kernel's static CB region
by ~7×**. The shrunk CB region leaves enough L1 headroom above it for
the L1-resident weights to land cleanly.

The change is also **much smaller in scope** than Option C — Option B's
TP=8 code path already exists and is production-validated; the only
real change is flipping the `memory_config` default from
`DRAM_MEMORY_CONFIG` to `L1_MEMORY_CONFIG` for weight uploads.

This is, in effect, the same lever as Option C's prefill TP=2 plan
(more TP → smaller per-chip matmul shape → smaller CB region → L1
weights fit). Option B happens to be already TP=8 by construction.

---

## Today's Option B state

Option B's TP=8 path **uses DRAM-resident weights today**, same as
Option C. Look at:

- `tt/option_b/vlm_slice.py:55` — default `memory_config = ttnn.DRAM_MEMORY_CONFIG`
- `tt/option_b/tp_block.py:79` — default `memory_config = ttnn.DRAM_MEMORY_CONFIG`
- `tt/option_b/tp_block.py:223` — all_reduce output to DRAM (CB-clash dodge for the all_reduce kernel)

OPTION_B_STATUS.md L125-132 reports `~125 MB / chip` for stage 1/2
(TP=8 VLM, 9 layers per chip) and `~100 MB / chip` for stage 3 (TP=8
expert, 18 layers per chip). **Those numbers are dominated by DRAM**
holding the weights — L1 only carries transient activations and the
small replicated norm/bias tensors.

E2E perf today: ~220 ms total (Option B bench, sec OPTION_B_VS_C_COMPARISON.md).

## Per-bank arithmetic if we migrate Option B weights to L1

Same approach as Option C: walk each constructed block, move weight
tensors to L1 via `ttnn.to_memory_config(t, L1)` + `ttnn.deallocate(t)`.

### Option B stage 1/2 (VLM, TP=8)

| Component | Per chip (9 layers) |
|---|---|
| VLM layer at bf8 replicated | ~110 MB |
| Sharded TP=8 (per chip = 1/8) | **13.75 MB / layer** |
| Total weights for 9 layers / chip | **~124 MB** |
| Per L1 bank (124 MB / 120 banks) | **~1.03 MB / bank** |

Compare to the threshold:

| Threshold (the gating math) | per-bank |
|---|---|
| L1 bank capacity | 1.43 MB |
| Static CB region for Option C SigLIP MLP (out_block_w=43) | ~0.73 MB |
| Static CB region for Option B TP=8 VLM MLP (out_block_w ≈ 6) | **~0.10 MB (estimated)** |
| Available L1 above CB region (Option B) | **~1.33 MB / bank** |
| Per-bank weight load (Option B TP=8) | 1.03 MB |
| **Fits above CB?** | **YES (1.03 < 1.33)** |

The CB-region estimate `out_block_w ≈ 6 → ~0.10 MB / bank` comes from
the ttnn-API research report: at TP=2 out_block_w drops from 43 (Option
C single-chip) to ~22; at TP=8 it drops to ~6. CB region size scales
roughly linearly with `out_block_w` × `out_block_h`, so a 7× shrink in
width → ~7× shrink in CB region.

**Critical caveat**: this is analytical. The CB region size estimate
is based on the documented relationship between `out_block_w` and CB
allocation — needs hardware measurement to confirm.

### Option B stage 3 (expert, TP=8)

| Component | Per chip (18 layers) |
|---|---|
| Expert layer at bf8 replicated | ~32.8 MB |
| Sharded TP=8 (per chip = 1/8) | **4.1 MB / layer** |
| + replicated adaRMS Dense bf16 | 3 MB / layer (per OPTION_B_STATUS.md) |
| Effective per-layer at TP=8 | ~7.1 MB |
| Total weights for 18 layers / chip | **~128 MB** |
| Per L1 bank | **~1.07 MB / bank** |

Same conclusion as VLM: 1.07 < 1.33 → fits, assuming the CB region
shrinks proportionally with TP=8 shapes.

Note: OPTION_B_STATUS.md reports `~100 MB / chip` measured for stage 3
DRAM. The 128 MB analytical above is slightly higher — likely the
status doc subtracts the modulation Dense or rounds. Either way the
order of magnitude matches.

## Activations during forward (don't blow the budget)

| At inference time on one chip | bytes |
|---|---|
| Prefill input activation `[1, 512, 2048]` bf8 sharded /8 → 256 dim | ~0.13 MB |
| Per-layer KV `(K, V)` sharded TP=8 | ~62 KB / layer |
| 9 layers × 62 KB | ~0.55 MB |
| Replicated all_reduce intermediates (DRAM, doesn't eat L1) | 0 MB L1 |
| Scratch / static CBs (other kernels) | ~5-10 MB |
| **Peak transient L1 during forward** | **~6-11 MB** |

So total L1 occupancy during forward: weights (124 MB) + transient
(~10 MB) ≈ **~134 MB / chip**, well inside the 175 MB cap. Plenty of
headroom (~40 MB).

## What it would take to test

This is **much smaller than Option C's L1 work** because:

1. **No layout changes.** Option B's TP=8 carving on (4,2) submeshes is
   already in production. No 7+7+7+6 redistribution, no sub-mesh
   carving from sub-meshes.
2. **No new helper code.** Just flip 2 default values from
   `DRAM_MEMORY_CONFIG` to `L1_MEMORY_CONFIG`.
3. **No CB-clash workarounds.** If the analysis holds, the CB region is
   small enough that L1 weights fit naturally.

### Concrete change list

| File | Change |
|---|---|
| `tt/option_b/vlm_slice.py:55` | Default `memory_config = ttnn.L1_MEMORY_CONFIG` (or behind a flag — `weights_l1=False` default to stay safe) |
| `tt/option_b/tp_block.py:79` | Same default flip (or flag) |
| `tt/option_b/tp_expert_block.py` | Same pattern for expert weights (analogous line — needs check) |
| (No changes) | mesh_setup, stage_*.py, pipeline.py, transport.py — all unchanged |

### Test sequence

1. **L1 footprint probe for Option B** — modeled on
   `tests/test_option_c_l1_footprint_probe.py` but driving
   `Pi0_5PipelineB` instead. Easy to write (~half the size of Option
   C's probe; no sub-mesh enumeration). Toggle `weights_l1` and
   measure per-chip L1+DRAM.
2. **Existing Option B smoke** (`test_option_b_smoke.py`) — run with
   `weights_l1=True` once the flag is wired. Should pass if the CB
   region analysis is correct.
3. **Existing Option B benchmark** (`test_option_b_benchmark_e2e.py`) —
   compare wall-clock with `weights_l1=False` vs `weights_l1=True` on
   the same shrunk-depth workload. Expected: L1 path is faster
   (smaller wall-clock per matmul because no DRAM read), but the
   shrunk-depth bench may not show much because per-layer compute is
   already tiny.
4. **PCC test** — `tests/test_pcc_option_b_vs_torch.py` (would need
   to be written — Option B doesn't have one yet, only Option C does).
   Mirror `test_pcc_option_c_vs_torch.py` for `Pi0_5PipelineB`.

Total work: **~4-6 hours** if the analysis holds. **If it doesn't
hold**, the probe will fail with the same CB clash and we learn what
the actual per-bank CB region size is for Option B's TP=8 shapes.

## Risks

| Risk | Likelihood | Mitigation |
|---|---|---|
| CB region doesn't shrink as much as estimated → clash returns | medium | Probe is cheap to run; if it fails, the failure address will tell us the actual CB region size for Option B's shapes |
| `in0_block_w` retuning needed at TP=8 for L1-resident pcfg | low | Option B already runs with these shapes — just retest with L1 weights |
| `l1_small_size` interaction (Option B uses None default per `mesh_setup.py:20`) | low | Set to 24576 (the pi0_5 single-device standard) before testing |
| Activation peak during forward exceeds available L1 | low | Math says ~134 MB / chip, well under 175 MB cap; verify with probe |
| All_reduce kernel's CB region clashes with L1 weights | medium | Already DRAM-dodged for the all_reduce **output** (`tp_block.py:223`); the all_reduce **input** is the matmul output which is currently L1 — may need same DRAM bounce around the all_reduce input |

The last risk is the most worth flagging. Today's pattern:
- matmul → L1 output → all_reduce → DRAM output → next matmul (DRAM read)

If matmul weights move to L1, the matmul output is *also* in L1, then
all_reduce reads that L1 input + L1 weights (none), and writes DRAM
output. The all_reduce kernel's static CB region might still collide
with the L1-resident weights of the *previous* matmul if those weights
remain alive across the all_reduce. This is solvable but is the most
likely surprise.

## How this relates to the Option C plan

The TP-within-stage plan for Option C prefill
([OPTION_C_TP_WITHIN_STAGE_PLAN.md](./OPTION_C_TP_WITHIN_STAGE_PLAN.md))
is structurally the same thing as Option B's TP=8 path. The only
difference is the TP factor — Option C prefill is shooting for TP=2
inside a (2,1) col-pair sub-mesh; Option B is at TP=8 inside the (4,2)
submesh.

**If Option B at TP=8 with L1 weights works as the analysis predicts,
the Option C prefill TP=2 plan should also work — same lever, smaller
factor.** The Option B probe is the cleanest first proof of concept
because it doesn't require new code; if it works there it's a strong
signal the Option C plan will work too.

## Recommendation

**Run the Option B L1 probe first.** Three reasons:

1. **Tiny scope.** Flip two defaults, write a probe script, run it.
   Maybe 100 lines of new code total.
2. **Direct test of the per-bank CB-shrinkage hypothesis.** If it
   works → L1-resident weights are viable on Option B today AND
   validates the lever Option C is planning to use. If it fails →
   we get a concrete measurement of how much CB region shrinks at TP=8
   shapes, which calibrates expectations for Option C's TP=2 plan.
3. **Independent of the Option C investigation.** No coupling — can
   be done in parallel with denoise modulation sharding, MLP
   DRAM-bounce, or whatever the next Option C move is.

If the Option B probe succeeds, the natural follow-ups are:

- Apply the same migration pattern to Option C prefill (TP=2). The
  per-bank math is even more favorable (0.46 vs 1.03 MB / bank).
- Pursue denoise lever (modulation sharding or bf8) per the open
  finding in OPTION_C_TP_WITHIN_STAGE_PLAN.md.

If it fails, we still come away with a concrete measurement of per-bank
CB region size at TP=8 — useful data for the kernel-engineering path
or the tt-blaze decision.
