# WH dest-reuse ELWMUL bug repro (after UnpackToDest fp32 read)

Minimal standalone reproducer for a Wormhole-B0-only LLK bug: any compute kernel that
performs an UnpackToDest fp32 read followed by `binary_dest_reuse_tiles<ELWMUL,
DEST_TO_SRCB>` with `fp32_dest_acc_en=true` produces structurally wrong output. The
identical sequence runs correctly on Blackhole.

## Trigger conditions (all required)

1. `fp32_dest_acc_en = true` in compute kernel config.
2. `DstSync::SyncHalf` (the default).
3. At least one CB has `unpack_to_dest_mode = UnpackToDestFp32` AND the kernel
   actually performs an UnpackToDest fp32 read of it (e.g. `copy_tile(cb, 0, dst)`
   where the LLK takes the UnpackToDest path because both src and unpack_dst are
   Float32).
4. After that read, the kernel does `sub_tiles_bcast_cols(primary_cb, bcast_a, ...)`
   followed by `binary_dest_reuse_tiles<ELWMUL, EltwiseBinaryReuseDestType::DEST_TO_SRCB>(
   bcast_b, 0, i)` within the same `tile_regs_acquire/commit` window on a CB
   carrying Float32 data.

**Multi-buffer-index CB aliasing is NOT required.** The test demonstrates this by
running two cases:
- **Case A**: the UnpackToDest cb and the dest-reuse-ELWMUL cb have separate L1
  allocations.
- **Case B**: they share an L1 allocation via the multi-buffer-index pattern.

Both fail with the identical pattern on WH.

## Observed failure pattern on WH

Per 4-tile `SyncHalf` block (`block_size = 4` with fp32 dest acc):

```
Block N (even index): tile 0 correct, tiles 1-3 zero
Block N (odd index):  all 4 tiles scaled by (1 + bcast_b) / bcast_b
```

The over-scaling is exactly the accumulation signature `dst = dst + dst*srcA`
instead of `dst = dst*srcA`. With `bcast_b = 0.7` the ratio is ~2.4286 (= 17/7).

## Files

- [wh_alias_dest_reuse_repro.cpp](wh_alias_dest_reuse_repro.cpp) — host driver.
  Runs Case A then Case B, prints per-tile expected/actual/ratio, classifies tiles
  into "correct", "zero", "over-scaled", and reports pass/fail.
- [kernels/compute/alias_dest_reuse.cpp](kernels/compute/alias_dest_reuse.cpp) —
  compute kernel. Does `copy_tile(cb_alias, 0, 0)` (UnpackToDest fp32 path), then
  per-block: `sub_tiles_bcast_cols(cb_primary, cb_bcast_a, i, 0, i)` for i in 0..3,
  then `binary_dest_reuse_tiles<ELWMUL, DEST_TO_SRCB>(cb_bcast_b, 0, i)` for i in
  0..3, then pack.
- [kernels/dataflow/reader.cpp](kernels/dataflow/reader.cpp) — pushes x / bcast_a
  / bcast_b tiles. Handles both Case A (writes data to alias's own L1) and Case B
  (alias shares L1; reader just pushes the alias semaphore).
- [kernels/dataflow/writer.cpp](kernels/dataflow/writer.cpp) — drains the output
  CB to DRAM.

## Build and run

```
./build_metal.sh -e --enable-fake-kernels-target --build-programming-examples
build_Release/programming_examples/wh_alias_dest_reuse_repro
```

## Expected results

| Arch | Case A | Case B |
|------|--------|--------|
| WH   | FAIL (striped pattern: 5 zero-output tiles, 8 over-scaled tiles) | FAIL (same striped pattern) |
| BH   | PASS (ratio ≈ 0.9989 throughout — TF32 SrcA rounding only) | PASS (same) |

Cases A and B fail identically on WH — **multi-buffer-index aliasing is NOT
required** to trigger the bug. They produce identical correct output on BH
(modulo TF32 SrcA rounding on the dest-reuse mul).

PASS/FAIL is decided purely by structural metrics:
- `num_zero_tiles == 0` (tiles whose output is ~0 but expected is non-trivial)
- `num_overscaled_tiles == 0` (tiles where actual/expected > 2)

Per-element diffs that exceed a small absolute threshold (~1e-3) are still
printed as `mismatch (informational, not part of verdict)` lines so anyone
reading the output can see the magnitude of any precision deviation, but they
do not influence the verdict. The structural metrics are what reliably
distinguish "bug triggered" from "correct output with TF32 rounding" on
architectures where the dest-reuse mul reads its operand through the SrcA
TF32 path.

### Empirical BH output for comparison

On Blackhole, every tile prints a ratio of ~0.9989 (a ~0.1% TF32 rounding loss
because `cb_bcast_b` is read via SrcA in the dest-reuse mul). No zero-output
tiles, no over-scaling. Both cases pass.

On Wormhole-B0, the per-block ratio pattern is:
```
Block N (even index): tile 0 correct, tiles 1-3 zero
Block N (odd index):  all 4 tiles scaled by (1 + bcast_b) / bcast_b ≈ 2.4286
```

## Suspected root cause

The bug appears to be in WH-specific cleanup after `_llk_unpack_A_` runs with
`unpack_to_dest=true`. The function `unpack_to_dest_tile_done` in
[`tt_llk_wormhole_b0/common/inc/cunpack_common.h:920-941`](../../tt-llk/tt_llk_wormhole_b0/common/inc/cunpack_common.h)
includes a WH-only TEN-3868 hardware-bug workaround that performs a dummy
`TT_UNPACR(SrcA, ...)` after every UnpackToDest read. BH's variant of the same
function lacks this workaround. The dummy unpack may leave residual SrcA / cfg
state that corrupts the subsequent dest-reuse ELWMUL path
(`_llk_math_eltwise_binary_with_dest_reuse_` in
[`tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_binary.h:451-523`](../../tt-llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_binary.h))
specifically when both ZEROACC and `move_d2b_fixed_face` are involved.

## Background

Originally found while debugging fp32 + Welford layernorm on the
`fplavec_43673_welford` branch (PCC ≈ 0.6 on
`test_large_layer_norm[dtype=torch.float32-use_welford=True-w=3200-h=2080]`). The
fix in that branch is a kernel-side workaround that avoids `binary_dest_reuse_tiles<
ELWMUL, DEST_TO_SRCB>` and instead packs `(x - mean)` to an intermediate CB then
re-reads it with `mul_tiles_bcast_cols` (the pattern used by `layernorm_welford.cpp`,
which works on WH). This repro strips away the layernorm-specific machinery to give
the LLK team a tight, layernorm-free starting point.
