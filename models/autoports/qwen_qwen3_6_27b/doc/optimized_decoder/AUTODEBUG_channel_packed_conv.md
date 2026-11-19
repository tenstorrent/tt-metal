# One native convolution with users packed as channels

Source-only, 2026-11-12. Helper: `candidates/channel_packed_conv.py` (syntax checked; no TT execution). Main decoder unchanged.

## Finding: native arithmetic supports this packing; B32 exchange does not fit

There is no semantic channel-width cap at 10240. The validator requires positive tile-aligned Q/K/V widths, total channels divisible by `channel_chunk_size`, TILE BF16 input `[1,T,W]`, T divisible by 32, matching BF16 taps of logical volume W, and ROW_MAJOR BF16 history `[1,3,W]`. With B32, `(Q,K,V)=(65536,65536,196608)`, W=327680, T=32, chunk=256 satisfy those rules. Factory tile indices and output writer counters are uint32, so Q=65536 channels is not a uint16 boundary. The writer selects Q/K/V by tile position and uses the corresponding width stride; no cross-channel arithmetic occurs.

**However, the proposed B32 `with_carry` exchange variant cannot fit.** `qkv_causal_conv1d_silu_program_factory.cpp:254-257` allocates an exchange-core scratchpad of `2*64 + 2*3*(2*W)` bytes, even on a one-chip mesh. For W=B*10240 this is `128 + 122880*B`: B32 requires **3,932,288 bytes** on one core. Blackhole total L1 is **1,572,864 bytes** (`tt_metal/hw/inc/internal/tt-1xx/blackhole/dev_mem_map.h:33`), before firmware/reservations. This is a hard local-op limit. The generic validator's per-convolution-core L1 check does not include that exchange scratchpad. Enlarging DRAM exchange storage cannot cure it. B8 scratch is 983168 bytes; it still needs actual allocation validation. No largest-feasible B assertion is made.

## Adaptation: one native call without exchange

Use the already-public `ttnn.experimental.kda.qkv_causal_conv1d_silu` (without `_with_carry`), with `actual_start=0` and `predecessor_carry=history_packed`. On one device the first tile reads the supplied history; the predecessor is unused at start zero but must be supplied to satisfy the API. See reader lines 150-168: the first logical segment selects history, and subsequent segment logic selects predecessor. For this one-tile decode no subsequent segment exists.

Update the original history separately with `[old[:,1:3], token]` and copy back to its existing allocation. This preserves native convolution math while avoiding exchange scratch, fabric bookkeeping, carry staging, and the carry unpack. The helper implements this as the default. It also includes the requested exchange variant for smaller B with native final-carry unpack; pass a setup-time buffer `[1,12,B*10240]`, the existing exchange-core semaphores, and actual_end=1. Do not run that branch at B32.

The no-exchange per-core dataflow buffers and reader scratch depend on chunk size 256, not total W; it distributes 40*B independent work items over the grid. B32 gives 1280 work items, serviced sequentially per core, not 1280 simultaneously allocated cores. `qkv_causal_conv1d_silu_distribution` uses ordinary work distribution with no hard work-item limit. Native math remains the same four tap-ordered HiFi4/FP32 accumulator matmuls and precise SILU, with unchanged 256-channel block width. Each user Q/K/V boundary is divisible by 256, preserving channel-block alignment.

## Exact transforms

At setup, each tap `[Q|K|V]` becomes `[Q repeated B | K repeated B | V repeated B]`, TILE BF16 `[1,1,B*C]`; this is shared across decode calls and prepared before capture.

At runtime:

1. Untilize qkv `[B,1,C]`. For each Q/K/V range, slice and reshape `[B,1,width] -> [1,1,B*width]`. Concatenate the three groups, pad time to 32 and tilize to native `[1,32,B*C]`.
2. For each history Q/K/V range, slice `[B,3,width]`, permute to `[3,B,width]`, reshape `[1,3,B*width]`; concatenate along channels. No dtype conversion.
3. Run one native op with expanded Q/K/V widths.
4. Each output `[1,32,B*width]` is untilized, reshaped `[32,B,width]`, permuted `[B,32,width]`, and tilized. This preserves **all** padded output rows exactly as the current native-per-user convention, not just logical row zero.
5. For no-exchange carry, concatenate old history rows 1:3 and the original compact input token, then copy into original history. For exchange carry `[1,3,B*C]`, split group ranges, reshape `[3,B,width]`, permute `[B,3,width]`, concatenate groups, and copy into original history.

## Cost and next measurement

At B32, one BF16 logical token is 655360 bytes, history is 1966080 bytes, and the padded native input/output volume is **20 MiB each** (32*32*10240*2). Packing does not reduce padded native arithmetic volume; it reduces launch count from 32 to one and uses available cores more uniformly. The full output untilize/permute/tilize moves 20 MiB per transformation. This cost could offset the launch win and must be measured, not assumed away.

A later tail-only adaptation can keep only native output row zero, reshape `[1,1,B*width] -> [B,1,width]`, and pad consumer time to32 with zero gates. That preserves valid-token semantics but changes unused padded q/k/v values; test native chunk GDN masking before using it. The helper initially preserves the complete padded outputs to isolate the packing candidate.

Compare B1/B8/B32 against native-per-user conv on real qkv and nonzero histories, exact output/history equality, multiple distinct sequential tokens, full decoder PCC and same-regime traced timings. Inspect whether ROW_MAJOR permute/reshape introduces additional internal layout ops. Watcher and allocation-tracked trace checks remain required. No measured result or final optimization claim exists yet.
