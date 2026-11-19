# Projection tail fusion: rejected direct BF16 packing and exact-order follow-ups

The parent-owned `v6_projection_reduce_pack_device.json` rejected direct BF16 packing from `fast_reduce_nc`: 42 of 256 strict rounding-boundary lanes differed, maximum absolute delta 0.0078125. The job closed cleanly at 10:16:48. No actual adapter fixtures or timings were reached. The strict gate remains unchanged, and the candidate is not eligible for integration.

Frozen rejected source: `tests/v6_projection_reduce_pack_probe.py`, SHA256 `c61ea132cdfc3201bc916f089b7f6fa1d2faa637a4ffac2c442792e23543fca0`. CPU/source setup: `v6_projection_reduce_pack_cpu_setup.json*`. Parent failure: `v6_projection_reduce_pack_device.json` and `.log`. The source and all failure artifacts remain intact.

The experiment copied the complete `linear` method from snapshot `700c10db90b23fac1d40427c9bbcd82a6924c7704699d54768a0160c2b641e05`. It kept streaming products/cumsum unchanged, then preallocated the native reduction output in the final dtype and placement under the existing `terminal_scan` guard. This would remove a separate typecast and router/out copy, but the failed rounding check disproves numerical equivalence.

## Why the rounding gate matters

`tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_typecast.h`, `calculate_typecast_fp32_to_fp16b`, explicitly computes integer round-to-nearest-even:

```text
rounded_bits = (bits + 0x7fff + ((bits >> 16) & 1)) & 0xffff0000
```

It then stores FP32 with the lower sixteen bits masked before BF16 packing. Direct reduction packing bypasses this SFPU rounding step. The 42 mismatches are consistent with a midpoint tie-rule difference; raw outputs were not retained by the first failed probe, so the exact packer tie rule is not asserted. Regardless of that diagnosis, direct BF16 reduction packing fails the existing strict boundary test.

The next minimal experiment is `v6_projection_cast_placement_probe.py`: retain the same native FP32 reduction and explicit typecast, but write typecast output directly to final DRAM for router/out. Its memo records commands and artifacts. No midpoint exception or tolerance relaxation is used.

## Established native block-reduction order

The factory selects `REDUCE_FP32_IN_DEST` when the input is FP32 and FP32 destination accumulation is enabled, as in this decoder. It explicitly enables direct FP32 unpack. `kernels/reader_reduce_nc.cpp` reads blocks in ascending order, with `inner_tile_size` stride. `kernels/reduce_nc.cpp` copies block0 into `dst0`, then repeats `copy_tile(..., dst1); add_binary_tile(dst0, dst1, dst0)` for blocks1 through N−1. Grouping affects buffering only, not the addition order. There is no reduction tree or TF32 input narrowing on this path.

That exact order could be folded into streaming without changing its mathematical operation sequence: assign each output-column run to one core, scan blocks in ascending order, preserve a separate FP32 destination sum tile, copy block0 unchanged, and use the same native `add_binary_tile` after each later block. Perform those SFPU adds only after the current block's cumsum finishes; reset cumsum carry at the next block. A full destination tile can retain all intermediate rows while only final row31 is eventually emitted, because binary addition is elementwise. A held destination acquisition or explicit spill/restore is required to preserve the sum tile across current per-height destination releases. `copy_dest_values<DataFormat::Float32>` is a candidate exact-copy primitive, but this combined kernel has not been built or verified.

This fusion reduces block parallelism. Router would go from 32 block/column runs to eight output tiles; attention-out from 512 block/column runs to 64 output tiles. Packed attention has more output tiles but changes load balance. Eliminating the native reduction dispatch therefore does not establish a speedup. This remains an untried hand-off item requiring exact partial/final comparisons and measured full-adapter/full-decoder timing.

## Other movement opportunity, not implemented

The remaining singleton coefficient preparation performs DRAM placement, TILE-to-ROW_MAJOR conversion, reshape, pair gather, and conversion back to TILE. A streaming reader could instead construct coefficient column0 directly from the original BF16 TILE activation's row0, using raw uint16 copies and the already-used pair flip (`logical_index ^ 1`). Original TILE row0 offsets are 0..15 and 256..271; coefficient column0 occupies offsets `row*16` for rows0..15 and `512+(row-16)*16` for rows16..31. This preserves BF16 bits, including signed zeros, and could avoid those preparation dispatches and movement. It requires proving the actual decode input placement/layout contract, correct tile addressing, and reader performance. It is an untried follow-up, not an implemented or measured optimization.
