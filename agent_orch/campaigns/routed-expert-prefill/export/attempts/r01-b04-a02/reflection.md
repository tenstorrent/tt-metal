# r01-b04-a02 result: 0 (accuracy_fail); timing alone would be geomean ~1.004 vs baseline (neutral)

## What happened vs expected
I expected 10-20% from halving the gate/up K-block count, with accuracy unchanged or slightly
better. Neither happened.
- **Accuracy:** all 9 shapes failed the relative gate. mean_abs rose 2-3.5% (DS-V3 0.2730 ->
  0.2818, Kimi-K2.7 +3.2%, GLM +3.1%, V4-flash +3.3%, gpt-oss +2.9%) and PCC dropped ~1-2e-4.
  The kernels ran correctly (no hang, PCC still ~0.980), so this is rounding, not a bug.
- **Timing** (my own baseline_us / us, for information only): DS-V3 1.046, Kimi-K2.7 1.032,
  GLM 1.030, V4-pro 1.004, gpt-oss 1.008, M3 0.994, Mistral 0.988, V4-flash 0.977, Kimi-K3
  0.962. Geomean ~1.004, inside the ±1% noise band. The 14-block shapes gained 3-5%. The 8-block
  shapes and K3 lost, and K3 (14 -> 4 blocks) lost the most.

## Why
- **Accuracy:** fp32_dest_acc_en is false, so inside a K-block the matmul accumulates in the
  16-bit dst, and each K-block boundary hands the partial sum to the packer's L1_ACC. Doubling
  the width doubles the K-span accumulated in bf16 dst (32-tile = 1024-element runs instead of
  512). Widening is therefore a precision reduction in disguise. **in0_block_w_gu is an accuracy
  knob in this op.** Don't widen it, and don't let the L1 fitter pick a wider width than the
  baseline's on any shape. (Narrower than baseline is fine for accuracy.)
- **Timing:** the per-K-block fixed-cost hypothesis is mostly refuted. Halving the number of
  ready/valid rounds, tilize init/uninit and packer drains bought at most ~5%, so per-block
  overhead is not where the ~2x-over-LoFi time goes. K3 and the short-K shapes got slower. The
  likely reason is that bigger blocks mean a longer pipeline fill per expert (the first
  221-330 KB gate+up block must be multicast before compute starts, x 28 experts on K3), and
  only 2 slots of slack per stream. Dropping the M-doubling step also moved K3 off its
  (8, w8) config.
- Together with r01-b03-a01 and r01-b04-a01, this rules out three DM/handshake explanations
  (DRAM read latency, read/multicast serialisation, per-block handshake count). The remaining
  explanations are compute-side work per tile at per_core_M = 1 (bf4 in1 unpack per tile-mult,
  1-row subblocks), per-expert phase transitions, and idle padding rows. The padding-row fix is
  exactly what the row-group nodes exploit (1.44x).

## Classification
flawed idea (accuracy: widening the K-block lengthens bf16-dst accumulation, so it is blocked by
the precision rule; performance: neutral, so the per-block-overhead model is wrong). The
implementation was correct.

## What a child of this node should try next
- Don't build on this node's config. Restore in0_block_w_gu <= the baseline width (16, or 8 for
  K3), because the gate effectively forbids wider K-blocks. A child of this branch should start
  from the root code (this node also reverted the parent's read-ahead in the reader/writer, so
  the dataflow kernels here are root code; only the factory differs).
- Spend the next attempt on the r01-b01-a01 line instead (expert-parallel row groups, 1.44).
  Orthogonal levers there: taller out_subblock_h for per_core_M >= 2 (R=2 groups give
  per_core_M = 3, so the bf4 in1 unpack would be shared across rows). Also overlapping expert
  e+1's gate/up fill with expert e's down phase, since the per-expert pipeline fill is visible
  here: more fill per expert made K3 slower.
- If you want the cheap L1 win this node tried for: per_core_M_max 4 -> 2 is harmless at these
  loads. The freed x-staging L1 could deepen the weight CBs to 3 slots at the SAME width, which
  doesn't change accuracy. Expect a small effect at most, given these results.
