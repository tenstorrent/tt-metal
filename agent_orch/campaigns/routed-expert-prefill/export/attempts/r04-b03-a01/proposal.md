# r04-b03-a01: measurement node: an in-situ, data-independent gate/up matmul probe at kernel start (perf-counter window on MATH) plus UNPACK-thread wall-clock tallies of every input-CB wait (x, gate, up, activated, down) and the owner's x tilize, to decide whether the ~31 cycles per tile-mult is the matmul pipeline or the compute waiting for data

## Motivation
The campaign's open question since r03-b02-a03 is what sets the ~31 cycles per tile-mult on the pcM-3 windows (mistral,
minimax-m3, dsv3/glm phase A, every DN window, gpt-oss DN). Two readings fit every counter so far:
- (A) intrinsic: the streamed unpack and the FPU serialize on the source-bank hand-off (r04-b01-a01's conclusion: after its
  CFGSHIFTMASK unpack the UNPACK thread is unstalled, yet the unpacker idles ~45% and FPU busy is ~50%).
- (B) data starvation: the compute's UNPACK RISC is spinning in `cb_wait_front` for the reader's next block. A RISC spinning
  on a CB semaphore leaves the Tensix UNPACK queue empty: thread not stalled, unpacker idle, FPU idle, exactly what (A)
  attributes to the bank.

Evidence for (B) that no node has tested directly:
- The LLK team's harness runs the same stock LLK at 17.2 cycles per tile-mult for a 1x8 LoFi block end to end
  (L1_TO_L1, tt-metal #58713 / #58714); math floor 17.1, unpack 16.0, and they overlap there.
- Cycles per mult scale as 1/per_core_M across shapes (31 at pcM 3, 44-47 at pcM 2), i.e. a constant weight-delivery
  rate per core (~6.2 B/cycle), which is what a delivery-bound kernel shows and a compute-bound one does not.
- Every compute-side change that halved unpack work (r03-b01-a04, r03-b02-a04, r03-b03-a04) or removed the unpack stall
  (r04-b01-a01) left cycles per mult unchanged.
- Total DRAM traffic / chip time is 443-475 GB/s on dsv3, kimi-k2.7, glm, dsv4-pro (~90% of 512).

Evidence for (A): gpt-oss DN is also at 30.9 cycles/mult with its DRAM at only ~260 GB/s; delivery chains (not DRAM)
could still starve it, which (B) covers but the counters cannot show.

The two readings send the campaign in opposite directions (op-local LLK work vs data-movement work), and four branches are
currently spending attempts on both. One measurement settles it.

## Mechanism
Profiler-only (PROFILE_KERNEL builds; non-profiler binaries are byte-for-byte the same size as root's):
1. In-situ probe (`re_probe_gu` in `kernels/compute/fused_swiglu.cpp`), all three TRISCs, once per launch right after
   `compute_kernel_hw_startup` and before the counts handshake, where compute is idle anyway (the reader is still fetching
   counts and planning). It is exactly one gate and one up subblock of the gate/up K-loop as `matmul_phase_fused_gu`
   issues them: same `matmul_block_init`, `in0_block_w` calls of 1 x `out_subblock_w` matmul_block with the same in1 index
   stride, gate -> up CB switch, one dst acquire / commit / wait / release each, on whatever L1 the x / gate / up CBs hold
   (no `wait_front`). MATH brackets it with the existing perf-counter window (new re_hw phase 2) and `tensix_sync()` so the
   window closes after the last MVMUL retires. Products are dropped: PACK releases without packing and the release's
   ZEROACC clears the half before any real subblock uses it. Emitted as 4 more `RE-HW` records (indices 16-19:
   ref/FPU, SFPU/mults, UNP0/UNP1, SrcA/SrcB writes).
2. UNPACK wait tally (`re_uw` in `kernels/compute/re_hwprof.hpp`), TRISC0: wall-clock cycles inside each input CB's
   `wait_front` (gate/up: x, gate, up; down: activated, down weights), inside the owner's x tilize (its cb_x_rm wait
   included), and the UNPACK-side length of each gate/up and down phase. 4 `RE-UW` records per core per call.

No schedule, data-movement or numerics change.

## Why this is not a repeat
- r03-b02-a03 / r04-b01-a01 read MATH-side counters over whole windows that mix computing and waiting; they could not
  separate (A) from (B). This node measures the matmul with no data dependency at all, in the real kernel (same LLK,
  compiler flags, dst sync mode, CB formats), and measures the waits directly on the only thread that waits.
- r03-b03-a02 measured the reader/writer (NCRISC/BRISC) waits on a different lineage; nobody has measured the compute's
  waits, which are the ones that cost FPU time.

## Expected effect and risk
- Score: root's ±1% (bit-identical outputs). The probe adds ~2.4-4.4 µs of compute work per launch before the counts
  handshake, where the compute waits for the reader anyway; if it is not fully hidden it costs ≤ ~1% on a shape.
- Reading the result: probe cycles per mult ≈ 17-20 and RE-UW waits ≈ 40-50% of the gate/up window means (B): the next
  nodes should work on delivery, and the per-stream split says which stream (x, weights, activated). Probe ≈ 31 with small
  waits means (A): the bank hand-off is real and the probe becomes a cheap in-situ harness for LLK variants.
- Risks: kernel-config-buffer size (+~1.9 KB of TRISC code in profiler builds; root's largest h7168 program is ~65.9 KB of
  the 70,656 B buffer by the JIT cache sizes), a hang if the probe's dst hand-off were unbalanced across threads (all three
  run the same two acquire/release pairs), profiler buffer space (TRISC1 +4 records, TRISC0 +4 records per launch).
