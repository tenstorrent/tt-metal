# r01-b01-a01: apply gamma (x*w) during the all-gather wait so POST is a single x*rsqrt pass

## Motivation
Baseline profile (`reports/baseline_1`, device zones relative to kernel start, chip 0, h7168 / 56 tile-cols per core,
20 workers = one tile-row each + 1 forwarder):
- R_INPUT 0 -> ~8.0 us, PRE + stick push done ~9-10 us.
- W_AGWAIT ends ~12.6 us (forwarder F_FABRIC ~2.4 us). Compute TRISCs are idle from ~9.5 to ~12.6 us.
- POST (x*rsqrt -> fp32 intermediate, then *gamma -> output) runs ~13 -> ~22 us, W_DRAIN tail to ~23-26 us.
POST is HiFi4 / fp32-dest FPU work, ~80-95 ns per tile-op, 2 tile-ops per column tile = ~9 us for 56 tiles.
Same picture at h3584 (28 tiles): compute idle ~6 -> ~9 us during AG, POST ~9.2 -> 14.4 us.
So half of the post-AG critical path is the gamma multiply, which does not depend on the gathered stats at all.

The broadcast weight read is also latency bound (14 blocks x per-block barrier; NCRISC ends ~15.5 us vs R_INPUT
end ~8 us), so the weight would not be resident early enough to move the multiply in front of the AG.

## Mechanism
Reorder RMSNorm as out = (x * gamma) * rsqrt(mean(x^2) + eps):
1. Compute (`dit_rmsnorm_fused_compute.cpp`): new constexpr `pre_ag_weight` (broadcast weight, no bias, no RoPE,
   whole-row norm, resident input, non-block-major, packed-AG path = exactly the campaign config). After PRE pushes the
   transposed stat tile, and BEFORE waiting for the gathered stats, compute `mul_tiles_bcast_rows(input, weight)` for
   the whole row into the (already whole-row, fp32) intermediate_cb. After the AG, P_NRED is unchanged and a single
   pass `mul_tiles_bcast_cols(intermediate, rsqrt)` packs straight to output_cb. Sub-phase 2 is skipped in this mode.
2. Reader (`dit_rmsnorm_fused_reader.cpp`): the broadcast weight read is issued as one deep batch (all face-row reads
   in flight, ONE barrier, one push of the whole row) instead of a per-block barrier, so gamma is resident ~1 us after
   the input row instead of ~7 us after.
Other configs (bias, RoPE, per-token, streaming, block-major, TP=1) keep the old POST order; only the reader's
weight-read batching changes for them (compute waits are cumulative so a whole-row push satisfies them).

## Why this is not a repeat
First node of the campaign; no prior attempts.

## Expected effect and risk
The x*gamma pass (~5.3 us at h7168, ~2.7 us at h3584) overlaps the ~3-3.5 us AG wait; the post-AG path drops from
2 to 1 tile-op per column. Expect ~-3 to -5 us per shape: roughly 1.12-1.2x geomean. Accuracy should be equivalent:
x*gamma is an exact bf16*bf16 product in fp32 dest, stored fp32, then multiplied by rsqrt (tf32 srcA/srcB) like the old
x*rsqrt -> *gamma order. Risks: CB deadlock if weight/intermediate pushes don't match the waits (would show as hang);
compute after the AG becoming DRAM-write bound (drain) so the gain is smaller than the compute saving.
