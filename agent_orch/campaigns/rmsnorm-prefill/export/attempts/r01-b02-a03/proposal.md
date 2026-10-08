# r01-b02-a03: stack x*gamma under the AG wait (r01-b01-a01) on top of the deep trid input read (r01-b02-a02)

## Motivation
- Best node r01-b01-a01 (1.1089) moved the gamma multiply in front of the all-gather wait so POST is one
  x*rsqrt pass. Its reflection says the wide shapes gained less because gamma landed late: the batched gamma read
  was issued only after the still per-block-barriered input read (R_INPUT ~7.5 µs at h7168), so gamma arrived
  ~10.7-12 µs and the x*gamma pre-pass partly spilled past the AG end.
- Parent r01-b02-a02 (1.0338) made the input read deep (per-block trids): R_INPUT at h7168 ends 4.7-6.7 µs
  (was 7.5-8.5), h3584 2.5-3.8 µs (was 3.7-4.7). Its post-AG path is still the old 2-pass POST (TRISC ends
  21.4-22.1 µs at h7168, ~9 µs after the AG wait ends at 12.4-12.7).
- The two levers are orthogonal: the parent shortens the pre-AG critical path, b01-a01 shortens post-AG. With the
  input landing ~2 µs earlier, the single-barrier gamma batch (issued right after the input row) should land around
  when PRE finishes (~7-9 µs at h7168), so the x*gamma pass can hide fully under the AG wait at all widths.
  Parent reflection item 1 recommends exactly this combination.

## Mechanism
1. Compute (`kernels/compute/dit_rmsnorm_fused_compute.cpp`): apply r01-b01-a01's change verbatim. Constexpr
   `pre_ag_weight` (broadcast weight, no bias/RoPE/per-token/per-head, resident input, non-block-major, packed AG).
   After PRE pushes the stat stick, compute intermediate_cb = x * bcast_row(gamma) (fp32) for the whole row, then
   wait for the gathered stats, and POST is one pass out = intermediate * bcast_col(1/rms). Sub-phase 2 is skipped.
2. Reader (`kernels/dataflow/dit_rmsnorm_fused_reader.cpp`): the broadcast gamma read becomes one batch (all
   2 x num_tile_cols face-row reads in flight, one barrier, one push of the whole row) instead of a per-block barrier,
   keeping this lineage's col_offset page base. It is still issued AFTER the deep input row (not interleaved:
   r01-b01-a02 / r01-b04-a02 showed interleaving tiny gamma reads with the input stream is a large regression).
No factory change. The parent's one-flush-per-row drain stays (compute still pushes block_size per block to
output_cb, which matches its cumulative waits).

## Why this is not a repeat
- r01-b01-a01 has this compute + gamma batch but the per-block input read. r01-b02-a02 has the deep input read but
  the 2-pass POST and per-block gamma read. No node has both.
- r01-b01-a02 / r01-b04-a02 tried deep input + gamma but interleaved the gamma reads into the input stream, which
  delayed the input by 3-8 µs. Here the gamma batch stays strictly after the input row lands.

## Expected effect and risk
Roughly b01-a01's gain plus the parent's read gain, more on wide shapes because gamma is no longer late:
h3584 ~14.5 µs, h4096 ~15.7, h6144 ~20.5, h7168 ~23. Score ~1.13-1.16.
Risks: the output drain (DRAM write contention among 20 writers) is already the tail; compute savings may turn into
writer slack on wide shapes, as b04-a01 saw. Accuracy should match b01-a01 (pcc 0.9999985, max_abs ~0.024).
A CB mismatch would hang; both changes were individually validated, so that is unlikely.
