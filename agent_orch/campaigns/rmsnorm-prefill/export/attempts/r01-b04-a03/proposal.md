# r01-b04-a03: read broadcast gamma on the idle writer (BRISC) at kernel start; reader keeps only the trid-pipelined input

## Motivation
- Parent r01-b04-a02 (0.840) interleaved the 2 x num_tile_cols gamma face-row reads into its trid-pipelined input
  read on NCRISC. The input row then finished ~4-8 µs later (h7168 R_INPUT end 7.6-8.9 -> 15.6-16.8 µs), and the
  whole pipeline shifted by that amount. Its reflection: gamma must leave the input path.
- r01-b01-a02 (0.899) did the same thing with the same result. The early-gamma half did work there: post-AG at
  h7168 went 8.1 -> 5.7 µs because x*gamma fully hid under the AG wait.
- r01-b04-a01 / r01-b01-a01: with gamma read after the input on NCRISC, gamma lands 1.5-8 µs after PRE ends
  (b04-a01 NCRISC end 15.4-16.8 µs vs AG wait end 13.9-14.3 at h7168), so the x*gamma pre-pass waits on cb_weight
  instead of hiding under the AG.
- The worker writer (BRISC) does nothing between filling the scalar CBs (~0 µs) and the stick push (PRE end,
  ~5-9 µs). Both b01-a02's and b04-a02's reflections name "read gamma on BRISC" as the next step. Nobody has tried it.

## Mechanism
1. `kernels/dataflow/dit_rmsnorm_fused_worker_writer.cpp`: new trailing CT args (`writer_reads_weight`, weight_cb,
   weight tiles, weight TensorAccessorArgs) and a common RT arg (weight addr). When set, right after the scalar
   CBs are filled the writer issues every broadcast-gamma face-row read (2 x 32 B per tile) into weight_cb under
   ONE barrier and pushes the whole row. Each worker starts at page `tile_row_start % tiles` and wraps, so the 20
   workers don't all hit the same DRAM bank in lockstep (b04-a02 suspected a same-page hot spot).
2. `kernels/dataflow/dit_rmsnorm_fused_reader.cpp`: new trailing CT arg `weight_from_writer`. When set, the reader
   treats the weight as already pushed, so the trid-pipelined input pass runs with `with_weight=false` (input
   only) and the deferred broadcast-weight read is skipped. Bias, RoPE, etc. are unchanged.
3. `device/dit_fused_distributed_rmsnorm_program_factory.cpp`: enable it for the AG (mux) writer with a broadcast
   [1,H] weight and a resident (non-block-major) POST. Append the CT args, append weight_addr to the writer
   common args, and refresh it in override_runtime_arguments.
Compute is unchanged. It already waits on weight_cb cumulatively, so a single whole-row push satisfies it.

## Why this is not a repeat
- b01-a02 / b04-a02 put the gamma reads on the same NCRISC queue as the input (interleaved), which delayed every
  input push. Here gamma moves to a different RISC and NoC, and the input stream has no gamma reads in it.
- b04-a01 / b01-a01 read gamma after the input on NCRISC (late gamma). Here gamma is issued at t~0 in parallel.
- It also isolates the parent's trid-pipelined input read (lookahead 4), which b04-a02 never measured on its own.
  b02-a02 showed a similar deep read is worth ~0.5-2 µs.

## Expected effect and risk
Input read as in b02-a02 (h7168 ~5.5 µs instead of ~7.5) and gamma resident before PRE ends, so x*gamma hides
under the AG on every shape and post-AG is one pass (~5.5 µs at h7168, ~3 µs at h3584). Expect roughly b01-a01's
times minus 0.5-2 µs, a score of ~1.12-1.18. Watch the drain tail: it may absorb part of the gain on wide shapes.
Risks: wrong CT-arg offset or common-arg index -> garbage gamma (accuracy_fail) or a JIT error. Weight_cb
double-produced -> hang. BRISC's gamma barrier taking longer than PRE would delay the stick push (look at the
W_GAMMA zone end vs W_PUSH start). Host change -> rebuild.
