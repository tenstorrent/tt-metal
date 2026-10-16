# r03-b03-a03 result: 0 (accuracy_fail). Fused SiLU-GLU works and is faster on glm/dsv3, but its more precise product raises mean_abs by 1.3-1.5% (gate: 1%)

## What happened vs expected
- The code is r03-b02-a02 (best, 1.5952) plus a plain-SiLU → fused binary SiLU-GLU on PACK. That also
  reverts the parent's all-NoC-0 gate/up reads.
- Host build OK, no JIT error, no hang on the first and only run.
- The five non-SiLU shapes are bit-identical to the baseline (same code path).
- The four SiLU shapes **fail the mean_abs gate even though PCC improved**:

| shape | PCC base → this | mean_abs base → this | Δ mean_abs |
|---|---|---|---|
| dsv3 | 0.9803104 → 0.9804764 | 0.273027 → 0.277167 | +1.52% |
| kimi-k2.7 | 0.9802863 → 0.9804514 | 0.273584 → 0.277733 | +1.52% |
| glm | 0.9802855 → 0.9804510 | 0.232011 → 0.235483 | +1.50% |
| mistral | 0.9802601 → 0.9804191 | 0.149542 → 0.151607 | +1.38% |

- Timing, chip-mean µs, vs r03-b02-a02 (same code apart from this mechanism):

| shape | r03-b02-a02 | this | Δ |
|---|---|---|---|
| glm | 457.5 | 450.6 | **-1.5%** |
| dsv3 | 519.7 | 516.0 | -0.7% |
| mistral | 191.5 | 190.2 | -0.7% |
| kimi-k2.7 | 733.0 | 734.6 | +0.2% |

  The other shapes did not change mechanically: gpt-oss -1.9% and mm3 +0.9% are the usual lottery.
  Geomean over the 9 shapes would have been about 1.0049x r03-b02-a02, i.e. ~1.603. That is inside the
  ±1% noise band.

## Why (profiler evidence)
**The mechanism works as designed.** MATH zones (/tmp/r03b03a03_tl.py on reports/r03-b03-a03):
- RE-ACT is now 0 on every SiLU-shape expert. In r03-b02-a02 it was ~5 µs per expert.
- But DN grew by about the same amount on the DRAM-bound shapes:
  - dsv3 rows y2-3, first expert: DN 240-320 (80 µs) → 233-326 (93 µs).
  - glm: the first DN is +5..+10 µs on most rows. Still, glm's mean core end moved 431 → ~428 and the
    max moved ~8 µs.
- This is the effect r02-b01-a03 described. The removed compute only exposes the down phase's wait for its
  first down-weight blocks. The 5 µs multiply pass had been giving the reader a free prefetch window.
- So on dsv3 / kimi-k2.7 (DRAM-bound) the saving is mostly absorbed. glm and mistral (less DRAM-bound)
  keep a little.

**Accuracy: why mean_abs rose while PCC improved.**
- PCC is scale-invariant and improved by ~1.6e-4 on all four shapes, so the activation is now closer in
  shape to the fp32 reference.
- mean_abs = mean |tt − ref| rose by 1.4-1.5%, uniformly across shapes. That uniformity points to a
  magnitude shift, not noise.
- The old path rounded silu(gate) to bf8 and multiplied it by up in a **LoFi** FPU multiply. LoFi uses
  only the top mantissa bits of one operand, which truncates toward zero. That biased |activated| low by a
  few percent.
- The bf4 weight quantization error (which dominates the error vs the fp32 reference, PCC ~0.98) inflates
  output variance. The old downward bias happened to partly cancel that in mean |error|.
- The exact fp32 product removes the bias, so the abs error grows although correlation improves.
- So the campaign gate (relative mean_abs ≤ +1%) effectively pins the baseline's LoFi-multiply numerics.
  "More precise" is not automatically "passes".

## Classification
repairable failure (not a code bug): the fused op is correct and RE-ACT is gone, but the numerics differ
from the gated baseline. Even if valid, the speed gain is small (~+0.5% geomean, inside noise), because on
the DRAM-bound SiLU shapes the saved time moves into DN's down-weight wait.

## What a child of this node should try next
1. **Don't spend more on the SiLU multiply pass for speed alone.** Even made accuracy-neutral, it is
   worth ≤0.5% geomean (glm -1.5%, mistral/dsv3 -0.7%, kimi-k2.7 0).
   - If you still want it, reproduce the baseline product on the SFPU: round silu to bf8-like precision
     and emulate LoFi's truncated multiply. That is hard to make exact (bf8 shares one exponent per
     16-element row of a face), and the gate is tight (+1%).
   - Simpler: keep `multiply_phase`, but run it per LW subblock inside the last gate/up K-block pipeline
     on MATH (FPU, same LoFi numerics, bit-identical). That hides the pass behind the other subblocks'
     matmuls.
2. **The real limiter on dsv3 / kimi-k2.7 / glm is the down phase's start.** When ACT is removed, DN waits
   for down blocks 0-1. Start the first down-weight DRAM reads during the last gate/up K-blocks: the reader
   is idle there once its last gate/up multicast is out.
   - r02-b02-a03 tried a deeper down-only ring on a lineage without the PACK pipeline and lost 1.3%.
   - Retry only as an explicit "issue down block 0 (and 1) reads right after the last gate/up read", with
     no ring-depth change.
3. Start from r03-b02-a02's code (or this node with `SILU_GLU` disabled: delete the `#define SILU_GLU`
   block in fused_swiglu.cpp; everything else is r03-b02-a02 byte for byte).
   - This node's parent (r03-b03-a02) code is dead: all-NoC-0 gate/up reads cost ~5-10%.
4. Measurement facts gathered here (/tmp/r03b03a03_bw.py, _tl.py, _tx.py):
   - Achieved weight bandwidth on r03-b02-a02: gpt-oss 210, mistral 296, mm3 348, dsv4-flash 356,
     glm 371, kimi-k3 379, dsv3 381, kimi-k2.7 406, dsv4-pro 417 GB/s.
   - mistral uses only 8 of 10 rows: four 2-row groups at pcM 3; rows y10-11 idle.
   - gpt-oss's two 2-row groups with identical work differ by ~21 µs (y8-9 slow), so group speed depends
     on position.
