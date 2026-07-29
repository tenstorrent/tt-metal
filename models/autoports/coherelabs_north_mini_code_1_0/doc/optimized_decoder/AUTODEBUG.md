# AutoDebug Report: advisor-seed DRAM-sharded hang

## Starting evidence

- Failing command:
  `python models/autoports/coherelabs_north_mini_code_1_0/tests/optimized_decoder_perf.py --implementation optimized --candidate advisor_seed --mode decode --layer 0 --batch 1 --warmups 2 --iterations 10`
- The advisor's down projection uses `in0_block_w=32`. That first failed during JIT with division by zero in
  `reader_bmm_tile_layout_in0_sender_dram_sharded.cpp`. The model-side adaptation to `in0_block_w=8` compiled but the full decoder run hung.
- Triage artifacts:
  `triage/advisor_seed_hang.txt` and `triage/advisor_seed_hang_summary.txt`.
- Triage did not localize the running operation: most low-level reads failed because the installed `tt-triage`/UMD binding expects a different `noc_read` signature. ARC and telemetry remained healthy, and no watcher/assert evidence was captured.
- No hardware commands were run and no implementation files were modified during this investigation.

## Source-derived contracts

For the dense MLP down projection, the physical tiled shape is
`[1,1,32,8192] x [8192,2048]`, hence `Kt=256`, `Nt=64`. The candidate uses:

- activation: L1 width-sharded over 8 cores, each shard 32 rows by 1024 columns (`32` K tiles);
- weight: BFP8, DRAM width-sharded by `_dram_weight_memory_config`;
- output: L1 width-sharded over 64 cores, one N tile per core;
- program: DRAM-sharded multicast, `in0_block_w=8`, `per_core_M=1`, `per_core_N=1`.

Thus the public validation checks pass (`256 % 8 == 0`, activation-shard K tiles `32 % 8 == 0`). In the factory, `num_blocks=32`; with all 8 activation storage cores retained, `num_blocks_per_shard=4`.

## Hypotheses and focused experiments

### H1 (highest priority): overlap removal drops activation sender cores and breaks the multicast semaphore schedule

Evidence:

`create_program_dram_sharded_descriptor` calls `move_common_entries(input_all_storage_cores_vec,
all_worker_cores_vec, storage_worker_common)`, then constructs both `mcast_senders` and
`mcast_receivers` from the *mutated* vectors. The common cores are absent from both sets. Later code attempts to
assign `worker_core_type=2` when a sender is in `storage_worker_common`, but that condition is unreachable because
the common cores were removed before `mcast_senders_coords` was built. Such cores are subsequently classified as
idle. `num_blocks_per_shard` and the sender-coordinate arrays are also computed from the reduced sender vector.

Prediction:

If any of the row-wise activation cores overlap Blackhole's selected DRAM reader cores, the descriptor omits those
activation shards and the remaining receivers/senders disagree about the block/semaphore schedule, producing the
observed device wait. The issue may affect the first DRAM-sharded projection, so the full-run hang cannot presently
be attributed specifically to down projection.

Smallest verify/refute experiment:

1. Add temporary host-only logging immediately after `move_common_entries` for the four sets: original activation
   storage cores, DRAM reader cores, `storage_worker_common`, and final sender/receiver cores. Also print
   `num_blocks`, retained sender count, and `num_blocks_per_shard`.
2. Run a one-op exact-shape matmul isolate for each role in order (`qkv`, `o`, `gate_up`, `down`), one invocation,
   bounded by timeout and with watcher in a separate run.
3. Verified if the first hanging role has non-empty `storage_worker_common` and the common cores receive
   `worker_core_type=0`, or if rebuilding the descriptor so common cores participate as type 2 makes the isolate
   complete. Refuted if the overlap is empty and all expected senders participate.

Candidate fix boundary if verified:

Fix the factory's core-set construction so common cores remain in the multicast sender set and are assigned type 2;
then derive sender count, coordinates, and `num_blocks_per_shard` from the complete activation storage set. Add
fatal validation that `num_blocks` is nonzero and exactly divisible by the number of activation sender shards.
Prefer a model-side non-overlapping activation core set only as a temporary avoidance/control, not as proof the
factory contract is sound.

### H2: the full decoder hangs in an earlier DS projection; the down-projection attribution is stale

Evidence:

The candidate applies the same factory to four operations before/through the MLP. Their exact tiled contracts are:

| role | Kt | in0 block | activation cores | blocks/core if all retained |
|---|---:|---:|---:|---:|
| qkv | 64 | 8 | 8 | 1 |
| o | 128 | 16 | 8 | 1 |
| gate_up | 64 | 2 | 8 | 4 |
| down | 256 | 8 | 8 | 4 |

The earlier `in0_block_w=32` JIT failure identifies a down-kernel compile defect, but after changing it to 8 there is
no usable triage evidence identifying which runtime op waits.

Smallest verify/refute experiment:

Create a test-only exact-shape one-op runner that materializes the same dtype, tile, shard specs, program config, and
compute config, then execute qkv, o, gate_up, and down separately in fresh processes. Use one iteration before any
warmup/trace. The first failing isolate identifies the real boundary; if all pass individually, test the
gate-up/split/SwiGLU/reshard/down chain next.

### H3: integer truncation in `num_blocks_per_shard` permits a deadlock-producing geometry

Evidence:

The factory computes `num_blocks_per_shard = num_blocks / input_all_storage_cores_vec.size()` without checking
remainder or zero. The kernel derives `num_storage_cores = num_blocks / num_blocks_per_shard` and sizes/indexes the
sender-coordinate runtime arrays from that compile-time value. Any sender count that does not divide `num_blocks`
can therefore make the kernel read beyond the runtime coordinate arrays or wait on a sender that does not exist.
Overlap removal in H1 can create exactly this condition even when the original 8-core geometry divides cleanly.

Smallest verify/refute experiment:

In the temporary descriptor logging/isolate, assert before kernel creation:
`sender_count > 0`, `num_blocks >= sender_count`, and `num_blocks % sender_count == 0`; also assert the kernel-derived
`num_storage_cores == sender_coordinate_count`. A host-side assertion failure verifies the unsafe geometry without
dispatching it.

Candidate fix boundary if verified:

Keep the corrected complete sender set from H1 and add these `TT_FATAL` contract checks so unsupported geometries
fail on host rather than deadlock.

### H4: the advisor's `in0_block_w=32` exposes a separate factory/kernel divide-by-zero contract bug

Evidence:

For down projection, `Kt/in0_block_w = 256/32 = 8`, which is legal only when there are no more than eight active
activation sender shards and each receives at least one block. The reported division by zero is consistent with
`num_blocks_per_shard` becoming zero when the factory's effective sender cardinality exceeds `num_blocks`, followed
by kernel compile-time `num_blocks / num_blocks_per_shard`.

Smallest verify/refute experiment:

Use the host descriptor probe from H1 with `in0_block_w=32`, print effective sender count before JIT, and enforce the
H3 assertions. Verified if sender count exceeds eight or computed `num_blocks_per_shard` is zero. If the effective
sender count is exactly eight and the division still occurs, capture the full JIT compile command and compile-time
argument list; the next investigation should target argument ordering/version skew rather than choosing another
block width.

## Recommended experiment order

1. Host-only descriptor core-set and arithmetic assertions (H1/H3/H4).
2. Fresh-process exact-shape DS isolates in execution order (H2).
3. Only after the first failing role is known, test the smallest corrected common-core handling or a non-overlap
   control.
4. Rerun the original command once, then a watcher-clean one-iteration correctness run. Do not start perf warmups
   until the isolate and watcher run complete.

## Current verdict

Still uncertain without a focused hardware experiment, but source evidence makes the descriptor's dropped
sender/reader overlap and unchecked block-to-sender division the leading root-cause family. Merely reducing
`in0_block_w` is not an earned fix: it avoids the compile-time zero for one shape but does not repair the multicast
core-set/semaphore contract and can still hang.
