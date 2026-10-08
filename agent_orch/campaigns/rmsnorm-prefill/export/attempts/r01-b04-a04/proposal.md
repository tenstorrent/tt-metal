# r01-b04-a04: accumulate sum(x^2) in DST across the row (ELWMUL always accumulates) and pack once, instead of one L1-accumulating fp32 pack per input tile; let the stick push preempt the BRISC gamma issue loop

## Motivation
The AG start is gated by the slowest worker's PRE, and PRE trails the input read on every node of every lineage:
- r01-b04-a03 (parent, 1.2075): W_PUSH (PRE + stick) ends 1.3-1.9 µs after R_INPUT at h7168 (R_INPUT 5.9-6.9, W_PUSH
  7.2-8.8) and ~1.4-1.8 µs after at h3584.
- r01-b02-a02 / r01-b02-a03: "PRE compute is now the pre-AG bottleneck", W_PUSH ends ~2.3-2.4 µs after R_INPUT,
  PRE ~125 ns/tile.
PRE (resident path) is, per 4-tile block: 4x mul_tiles(x, x) (HiFi4) into dst 0..3, then 4x `pack_tile<true>` of a
fp32 tile into pre_intermediate_cb[0] with packer L1 accumulation (read-modify-write of a 4 KB tile per input tile).
Unpack (2 x bf16 tile) and HiFi4 ELWMUL (~32 FPU cycles) are cheap; the per-tile fp32 L1-acc pack is the likely
bottleneck. ELWMUL on WH/BH *always* accumulates onto Dst (Dst += SrcA*SrcB, tt-isa-documentation ELWMUL functional
model; the LLK hardcodes acc_to_dest=0 for ELWMUL and HiFi phases rely on the accumulation; DST is zeroed on release).

## Mechanism
1. `kernels/compute/dit_rmsnorm_fused_compute.cpp`, resident (non-streaming) PRE: acquire DST once per row group,
   `mul_tiles(input, input, t, t, 0)` for every tile of the row (cumulative block waits unchanged, DST held across
   them), commit, ONE `pack_tile(0, pre_intermediate_cb)` (no L1 acc), release. Reduce + transpose unchanged.
   Sum is still fp32 and sequential in column order.
2. `kernels/dataflow/dit_rmsnorm_fused_worker_writer.cpp`: with a faster PRE the BRISC gamma issue loop (~4.5 µs at
   h7168, ends 0.3-0.5 µs before PRE at h3584 in the parent) would gate the stick push (BRISC is serial). The stick
   push becomes a lambda; inside the gamma issue loop, if `stats_transposed_local_cb` already has the stat, push the
   first stick immediately (write barrier only; gamma reads stay in flight), and the row loop skips it.
No host/factory change, so no rebuild.

## Why this is not a repeat
No node has touched PRE. Reflections r01-b02-a02 #2, r01-b02-a03 #3, r01-b01-a03 #2, r01-b04-a03 #3 all name it as
untried. The parent's #1 (port onto the column split) is a big cross-lineage merge that other branches' #1 also
point at; this is a structurally different lever (compute pipeline, AG start) and composes with all of them.
The kernel comment says init-hoisting didn't help; this removes the per-tile pack, not inits.

## Expected effect and risk
PRE tail after the input read shrinks from ~1.5-2 µs to ~0.3-0.5 µs, so F_COLLECT / AG / POST / drain all start
~1 µs earlier: ~-0.8 to -1.5 µs per shape, ~+5% geomean. Also less PRE spread across workers.
Risks: if ELWMUL did not accumulate, the stat would be only the last tile's square -> accuracy_fail (clear signal).
fp32 accumulation order is identical (DST fp32 add vs packer fp32 add), so PCC should be unchanged. Hang risk only
if the stick double-push logic is wrong (first_stick_pushed guard).
