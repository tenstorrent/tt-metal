# Same-input MoE reduction: offline diagnosis

Evidence: `moe_reduction_rows_v2.json` and
`/home/ttuser/qwen38-flash/reference/moe_reduction_rows_v2.pt`.
The parent ran the device probe in about8seconds. This analysis used only saved
CPU tensors. No production changes or device calls were made by the reviewer.

All native repeat, duplicate-row, row-permutation, and32-position controls pass
exactly. The native batched reduction nevertheless differs from the accepted
single-row matmul, including a rank with exactly one owned expert. Consequently,
inter-expert summation association cannot explain every difference. The separate
poolv5 control also found uninterrupted and migrated B2 state/output exact;
that control separates this geometry discrepancy from migration.

## One-owned-expert control

`one_local_rank0`, rank0, contains5120 output values. Comparisons against the
saved native output are:

| CPU product/phase model | Different values | Maximum absolute error |
|---|---:|---:|
| Full product, nearest-even BF16 |943|0.0009765625|
| Four HiFi4 phases, nearest-even BF16 per phase |396|0.0009765625|
| Four HiFi4 phases, nearest-away BF16 per phase |47|0.00048828125|
| Full product, truncation to BF16 |3181|0.001953125|

The nearest-away phase model is exact for all2560 values of row1. Its remaining47
differences occur in row0. Separately rounding each partial product before its
phase addition or rounding only the phase sum makes no difference for this
one-expert control. Changing phase order/tie-rule combinations did not produce
an exact model; this exploratory search is not a hardware specification.

The saved **single-row** path matches the full-product nearest-away model
exactly for all5120 values. Full-product nearest-even differs in190 values.
Thus PyTorch's BF16 cast is not by itself an exact reference for these device
results. The RNE-only emulator in the original JSON must not be interpreted as
proof that native rounding is incorrect.

One residual scalar case, with no inter-expert addition:

```
combined = 1.15625
weight   = 0.050537109375
exact FP32 product             = 0.05843353271484375
full-product BF16 nearest      = 0.058349609375
four-phase nearest-away model  = 0.058349609375
native BF16 reduction          = 0.05859375
```

The phase contributions from the current LLK golden masks are
`[0.05657958984375, 0.0015716552734375, 0.000274658203125,
0.00000762939453125]`. Repeating that exact operand pair through native broadcast
multiply would be a minimal follow-up if hardware rounding semantics need to be
isolated from the reduction wrapper. The discrepancy is a falsifiable scalar
observation; it is **not yet an identified implementation defect**.

## Source boundaries and implications

`MeshMoe._reduce_single` uses the loader's HiFi4/FP32 DEST matmul and emits BF16.
The native batched operation uses HiFi4/BF16 DEST multiplication, then BF16
packer accumulation. `llk_math_eltwise_binary.h` configures four fidelity phases;
the Blackhole `FidelityMasking` golden splits SrcA's low3 and SrcB's low1 BF16
mantissa bits. These are additional rounding boundaries before expert summation.

`reconfigure_packer_l1_acc` in `cpack_common.h` drains PACK before changing its
configuration. The operation's scalar writer supplies weight element0, which
the scalar-broadcast API consumes. Its BF16 row-major scratch is tilized as
BF16; the helper's documented FP32-to-TF32 fast-path narrowing does not apply.
No source-grounded addressing, scalar-selection, or missing packer-config fence
defect was found. Qwen's combine still initializes zeros, so nonowned stale-NaN
behavior is not established by this fixture.

The current emulators omit full hardware FPU phase/rounding semantics. Their
failure to match bitwise cannot establish a kernel bug. The truncation negative
control is substantially worse and does not support a blanket truncation claim.
No accumulation-association repair is justified by these results. Changing
batched decode to FP32 matmul, even while leaving prefill unchanged, would alter
the accepted batched accumulation policy. BF16 pairwise summation would also be
a new arithmetic/performance candidate and would not resolve the single-expert
multiplication discrepancy by itself.

Next evidence should compare the real B1/B2 captures' owned expert values,
router IDs and weights before reduction. Synthetic invariance excludes the
tested row-placement defects; it does not prove this reduction difference is
the earliest cause of real forced-token43 divergence.
