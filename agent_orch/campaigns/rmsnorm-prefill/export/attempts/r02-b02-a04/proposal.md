# r02-b02-a04: AG release by multicast: the forwarder releases its whole worker group with one go-sem multicast per worker row-segment instead of 20 serial unicast incs, and each worker pre-stages its gathered-stick reads (CB reserve + NoC addresses) before the go wait

## Motivation
Nobody has touched the AG *release* path (forwarder go -> worker has its gathered stats). Every node so far attacked
the read, PRE, the drain or placement. Per-core zones on the best node r02-b02-a03 (`/tmp/r02b02a04/go.py` on
`reports/r02-b02-a03`, medians over measured calls, all 4 chips, µs after the forwarder's F_FABRIC end):

| | first worker go | last worker go | W_DRAIN start - go (stick read) |
|---|---|---|---|
| every shape | 0.17 | 0.52 | 0.65-0.72 |

- **The go fan-out is serial.** The forwarder `group_go_sem.up()`s its 20 workers one after the other. Slot 0 gets go
  0.17 µs after F_FABRIC ends, slot 19 gets it 0.52 µs after. TRISC end tracks go exactly (trisc_end - go is
  constant per shape), so the last-released cores also finish compute ~0.35 µs later.
- **Those late-released cores are the late drainers.** The y=3 row (slots 11-19) ends its drain ~0.3 µs after the
  y=2 row on every shape. r02-b02-a01's reflection saw the same "y=3 ~0.3 µs later" and blamed NoC rows; the
  go order explains it at least as well.
- After go, each worker spends 0.65-0.72 µs before W_DRAIN starts. That is CB reserve + 8 TensorAccessor address
  computations + 8 x 64 B DRAM reads + barrier + push. The address work and reserve can be done while waiting.

## Mechanism
1. **Fork the forwarder into the op directory.** The shared forwarder lives in `dit_fused_norm_common/` (outside
   `allowed_paths`, also used by GroupNorm). Copy it to
   `dit_fused_distributed_rmsnorm/device/kernels/dataflow/dit_rmsnorm_forwarder.cpp` and point the RMSNorm factory at
   the copy. GroupNorm keeps the original. Nothing outside `allowed_paths` is edited.
2. **Multicast go.** The factory splits each forwarder's worker group (contiguous, row-major logical cores) into
   row segments of contiguous logical x. Each segment becomes one NoC rectangle (virtual start/end + core count),
   passed as forwarder RT args (before the fabric-connection args). In a round where the whole group is present,
   the forwarder writes `r+1` into its own (unused) copy of the grid-uniform go semaphore and `set_multicast`s it to
   each rectangle (start/end swapped for NoC1). Workers already `wait_min(r+1)`, so the protocol is unchanged. Partial
   (remainder) rounds keep the unicast inc loop, so a worker with no row that round is never touched. For these
   shapes there are 2 rectangles (11 + 9 workers) instead of 20 incs.
3. **Pre-staged stick read in the worker writer.** Before `go_sem.wait_min`, reserve `stats_transposed_gathered_cb`
   and precompute the ring_size x 2 NoC addresses of this worker's sticks. After go, issue the 8 reads with
   `PrecomposedUnicastEndpoint` and barrier.
4. A compute zone `C_COMB` around the post-AG stat combine (add + transpose_dest + rsqrt). This is instrumentation
   only, for the next node.

Files: `.../device/dit_fused_distributed_rmsnorm_program_factory.cpp`,
`.../device/kernels/dataflow/dit_rmsnorm_fused_worker_writer.cpp`,
`.../device/kernels/dataflow/dit_rmsnorm_forwarder.cpp` (new),
`.../device/kernels/compute/dit_rmsnorm_fused_compute.cpp` (zone only).

## Why this is not a repeat
- No node changed the forwarder or the go/release protocol. Earlier reflections assumed the forwarder was outside
  `allowed_paths`. Its *file* is, but a copy inside the op directory is not.
- Nearest nodes are r02-b02-a01 (drain NoC choice) and r02-b04-a01 (placement). Both moved bytes on links. This
  node moves no bytes: it removes a serial 20-step fan-out and some address arithmetic from the critical path.
- It is orthogonal to the PRE-tail work (a02/a03) and the drain NoC split (a01), which are all kept.

## Expected effect and risk
- Last-released worker gets go ~0.3 µs earlier, and every worker has its sticks ~0.05-0.1 µs earlier. The kernel end
  is the slowest core's drain, and those were the late-released y=3 cores, so expect -0.15 to -0.35 µs per shape:
  ~+1.5-2% at h3584/h4096 and ~+1% at h6144/h7168.
- Risks:
  - Wrong multicast rectangle or num_dests: a hang (ack count never reached), or a go write landing on a
    non-worker core. Rectangles are built only from logical worker cores. Contiguous logical x maps to increasing
    virtual x on BH, and non-Tensix columns in between are skipped by the NoC's broadcast-disable config.
  - NoC1 multicast orientation: start/end are swapped in the kernel when the forwarder's NoC is 1, matching the
    matmul factories' convention.
  - Accuracy: unchanged. Same data and same compute, except the zone.
- How to tell: `go.py`. The "last worker go - F_FABRIC end" column should collapse to the first-worker value (~0.2).
  The y=3 drain ends should move to the y=2 values. A hang would show as `fail_class=hang`.
