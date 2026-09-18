# HANDOFF agent S -- Milestone 1: two-stream (two command queues) backward for the sequence-parallel linears

Written 2026-09-18 23:20 on PAUSE. Branch `imichalak/llama-sp/6-sp-fused-matmul-ccl`, all work UNCOMMITTED in the tree
(tt-train only; last `build.sh _ttml` clean at 21:57, nothing edited since). Logs: `generated/spfuse/logs/` (the logs of
runs before the 21:55 scratchpad wipe are gone; their numbers are quoted from the run outputs as reported then, marked *).
Scripts of mine in `generated/spfuse/`: `bench_sp_overlap.sh`, `S_queue.sh`, `S_unit_chain.sh`, `S_summarize.py`,
`S_STATUS.txt` (the status note written after the wipe).

## 1. Design as implemented

**Queues and sub-devices.** The mesh is opened with two hardware command queues (`AutoContext::open_device(...,
num_command_queues=2)`; Python `ttml.open_device_mesh(..., num_command_queues=2)`). `AutoContext::enable_ccl_sub_device(
num_columns, num_rows)` loads a sub-device manager with two sub-devices per chip: sub-device 0 = the compute rectangle
`[0, grid.x - columns) x [0, grid.y - rows)`, sub-device 1 = the CCL region (the bottom `rows` rows or the rightmost
`columns` columns; exactly one non-zero), `local_l1_size 0` (one global allocator), and sets
`MeshDevice::set_compute_with_storage_grid_size_override(compute rectangle)` (cherry-picked plumbing af7b7477e4d) so every
op that sizes itself from `compute_with_storage_grid_size()` stays inside sub-device 0. Queue 0 is the compute queue and
runs on sub-device 0; queue 1 is the CCL queue and runs on sub-device 1. They are never mixed: tt-metal's
`CQOwnerState` makes a sub-device owned by the queue that last launched on it (`take_ownership` TT_FATALs otherwise; the
FSDP work saw a deadlock for a queue-0 collective on the CCL sub-device). `ttnn_fixed::distributed::all_gather /
reduce_scatter` pass `sub_device_id = ccl` when the thread's current command queue (ttnn::core::with_command_queue_id,
cherry-picked 9a0f9ad5749) is queue 1, nullopt (sub-device 0) otherwise. Default layout `rows=1`: a 12-core row fits 2
CCL workers per link at 2 links (AG default_workers: 20/12/4 cores for 4/2/1 workers); a 10-core column fits 1.

**CCLResources.** One pool of rotating global-semaphore sets per command queue (8 sets each of barrier / all-gather (2)
/ reduce-scatter (3) / all-reduce-barrier (2)), allocated on the FULL grid (`AutoContext::full_compute_grid_size()`) so
kernels on the CCL rows find them. The pool is chosen by the thread's current queue id. `close_device` drops
CCLResources (pool count is per open) and the sub-device state.

**Collectives on the CCL queue and buffer lifetime.** Host allocation is immediate and NOT ordered across queues, so:
* Before every CCL-queue collective: a *compute drain* = record an event on queue 0 for sub-device 0
  (`mesh_command_queue(0).enqueue_record_event({compute})`) and `enqueue_wait_for_event` on queue 1. The collective
  therefore starts only after every compute program enqueued so far has finished on this device, which (with the
  op's own barrier semaphore) also makes every buffer the host has freed so far really free on every device.
* After the collective: an event on queue 1 for sub-device 1; queue 0 `enqueue_wait_for_event`s it right before the
  first compute that consumes the output (`SPOverlap::wait`).
* Every tensor the collective reads, writes or stages through (`Collective::buffers`) is kept referenced until TWO
  later collectives have been waited for (`kRetireDepth = 2`, deque `m_retire`), and dropped at device close.
  `reduce_scatter` on queue 1 therefore takes its buffers from the caller: `reduce_scatter_buffers(tensor, dim, axis)`
  allocates `{intermediate, output, penult}` (ring: `reduce_scatter_minimal_async_create_intermediate_buffer`) or
  `{intermediate [2B,1,S,K] tiled, output}` (line) in the op's own layout, and the wrapper TT_FATALs on queue 1 without
  them (the op's self-allocated temporaries would be freed by the host under the running collective) and checks the
  result landed in the persistent output. `all_gather` takes a persistent output (checked the same way).
* Events are device-only (`enqueue_record_event`, not `..._to_host`).

**Deferred weight gradients and the schedule** (`ops/distributed/sp_overlap.{hpp,cpp}`, class `SPOverlap`, process
singleton). In the Composed backward of `sp_column_parallel_linear` / `sp_row_parallel_linear`
(`ops/distributed/sp_linear_ops.cpp`), when `SPOverlap::backward_active()`:
* column (dgrad = reduce_scatter(grad @ W)): `partial = sp_linear_matmul(grad, W)` on q0 -> `issue(RS on q1)`
  -> `drain_one()` (the previous linear's deferred wgrad on q0) -> `defer(own wgrad + bias grad)` -> `wait(RS)` ->
  `x->add_grad(dgrad)`.
* row (dgrad = all_gather(grad) @ W): `issue(AG on q1)` -> `drain_one()` -> `wait(AG)` -> dgrad matmul on q0 ->
  `defer(own wgrad)`.
* In Llama's backward order (w2 row, swiglu, gate_up column, out_proj row, attention, qkv column) each collective hides
  behind the previous linear's weight gradient: w2-AG behind the previous block's qkv wgrad (304 us at B=1), gate_up-RS
  behind w2's (542), out_proj-AG behind gate_up's (1002), qkv-RS behind out_proj's (216; the only slot where the
  collective, 234 ring / 337 line at 4 workers, is longer than the matmul).
* `Tensor::backward` brackets its node loop with `AutoContext::enter/exit_backward` (depth counter) and calls
  `run_backward_end_hooks_if_outermost()` after the loop; SPOverlap's hook drains all deferred work, so everything after
  `backward()` returns (sync_gradients, clipping, optimizer) sees complete gradients. Nested backward (gradient
  checkpointing's recompute, `memory_efficient_runner`) does not drain: deferred wgrads carry to the next block.
  Teardown hooks (`reset_graph`, `close_device`) drop leftovers (an interrupted backward: Synchronize then drop).
* Same ttnn ops, same shapes, same cores, only queue and order differ => results bit-identical to the one-queue run on
  the same (split) grid. The grid split itself changes ttnn::linear's auto block config, so split vs whole grid is
  ULP-level, not bitwise.
* Measurement-only: a NoComm backward takes the same path with the collective replaced by an empty tensor
  (`backward_is_nocomm()` in sp_linear_ops.cpp), to price the machinery alone. Not yet run.

**Switches (names, defaults).**
* C++ `ttnn_fixed::distributed`: `set_sp_linear_impl(impl)` (BOTH sites; clears the backward override),
  `get_sp_linear_impl()` (forward), `set_sp_linear_backward_impl(std::optional<SPLinearImpl>)` (backward only; nullopt =
  follow the forward), `get_sp_linear_backward_impl()`, `SPLinearSite {Forward, Backward}` (passed by the backward
  closures to `all_gather_matmul` / `matmul_reduce_scatter`), `sp_linear_matmul(a, w, transpose_b, bias)`.
  `ttml::ops::distributed::set_sp_overlap_mode(SPOverlapMode::Off|Backward)` / `get_sp_overlap_mode()`; default Off.
  `SPOverlap::backward_active()` = mode Backward && backward impl in {Composed, NoComm} && inside Tensor::backward &&
  2 queues && CCL sub-device present (so a plainly reopened device silently runs one-queue).
* Python (`ttml.ops.distributed`): `set_sp_linear_impl("composed"|"fused"|"nocomm")`, `set_sp_linear_backward_impl(
  "same"|"composed"|"fused"|"nocomm"|None)`, `get_sp_linear_backward_impl()`, `set_sp_overlap("off"|"backward")`,
  `get_sp_overlap()`, `sp_overlap_stats()` -> (deferred, retained); `AutoContext.open_device(shape, ids,
  num_command_queues)`, `.enable_ccl_sub_device(columns, rows)`, `.has_ccl_sub_device()`, `.ccl_sub_device_index()`,
  `.compute_sub_device_index()`, `.full_compute_grid_size()`, `.num_command_queues()`, `.is_backward_in_progress()`;
  `ttml.core.distributed.all_gather(..., persistent_output=None)`, `.reduce_scatter(..., persistent_buffers=None)`,
  `.reduce_scatter_buffers(tensor, dim, cluster_axis)`.
* device_config (train.py): `sp_linear_impl` (fused), `sp_linear_backward_impl` (same), `sp_overlap` off|split|backward
  (off; `split` = CCL region reserved + one queue = the measurement reference; a bare YAML `off` parses as False and is
  mapped back), `sp_ccl_rows` (1), `sp_ccl_columns` (0). train.py opens 2 queues and splits when sp_overlap != off.
* Tests: `TTML_SP_OVERLAP=backward [TTML_SP_CCL=rows=1]` makes conftest's `tp_mesh` open 2 queues + split + set the
  mode for any module; `TTML_SP_TEST_MESH=1x2|1x4_ring|1x4_line` selects test_sp_overlap.py's mesh.

## 2. Files changed / added (all under tt-train/, uncommitted)
Added: `sources/ttml/ops/distributed/sp_overlap.hpp`, `sources/ttml/ops/distributed/sp_overlap.cpp`,
`tests/python/test_sp_overlap.py`.
Modified: `sources/ttml/CMakeLists.txt` (sp_overlap.cpp), `sources/ttml/autograd/auto_context.{hpp,cpp}` (queues, CCL
sub-device, backward depth, backward-end + teardown hooks, close_device resets), `sources/ttml/autograd/tensor.cpp`
(backward bracket + end hooks), `sources/ttml/core/mesh_device.{hpp,cpp}` (num_command_queues),
`sources/ttml/core/distributed/ccl_resources.{hpp,cpp}` (per-queue pools), `sources/ttml/ttnn_fixed/distributed/
ttnn_ops.{hpp,cpp}` (persistent buffers, queue-routed sub-device, reduce_scatter_buffers, per-site policy,
sp_linear_matmul), `sources/ttml/ops/distributed/sp_linear_ops.{hpp,cpp}` (two-stream backward paths, SPLinearSite),
`sources/ttml/nanobind/{nb_autograd,nb_core,nb_ops}.cpp`, `sources/ttml/ttml/_mesh.py`, `sources/ttml/ttml/common/
config.py`, `sources/examples/train/train.py`, `tests/python/conftest.py`, `docs/DISTRIBUTED_TRAINING.md`,
`configs/README.md`. (Also in the tree but not mine: agent F's ttnn edits and the coordinator's NoComm impl.)

## 3. Measurements (Llama-8B tp4 SP, 1x4 galaxy, 6-step runs, s/step = mean of steps 3-5; phases = naive profiler
markers, ms/step, each marker syncs the device). `*` = pre-wipe run, log gone.

| impl (fwd / bwd) | sp_overlap | CCL region | topo | batch | memeff | step s | fwd ms | bwd ms | gradsync | optim | log |
|---|---|---|---|---|---|---|---|---|---|---|---|
| nocomm | off (whole grid) | - | ring | 1 | 0 | 0.533 | 127.8 | 296.3 | 11.8 | 93.9 | coordinator* |
| nocomm | split | rows=1 | ring | 1 | 0 | 0.541 | 136.7 | 293.7 | 12.3 | 93.1 | * |
| composed | off (whole grid) | - | ring | 1 | 0 | 0.603 | 157.5 | 339.4 | 8.1 | 94.4 | coordinator* |
| composed | split | rows=1 | ring | 1 | 0 | 0.609 | 165.2 | 334.5 | 10.2 | 93.7 | * |
| composed | **backward** | rows=1 | ring | 1 | 0 | **0.592** | 165.5 | 315.1 | 12.1 | 93.8 | * |
| fused / composed | backward | rows=1 | ring | 1 | 0 | 0.599 | 173.8 | 313.4 | 12.1 | 93.8 | * |
| fused | off (whole grid) | - | ring | 1 | 0 | 0.599 | 157.0 | 335.4 | 7.7 | 94.4 | coordinator* |
| nocomm | off (whole grid) | - | line | 1 | 0 | 0.534 | 128.1 | 296.4 | 9.6 | 93.9 | coordinator* |
| nocomm | split | rows=1 | line | 1 | 0 | 0.537 | - | - | - | - | * (phases not recorded) |
| composed | off (whole grid) | - | line | 1 | 0 | 0.629 | 171.9 | 351.9 | 6.9 | 94.1 | coordinator* |
| composed | split | rows=1 | line | 1 | 0 | 0.634 | 179.2 | 346.0 | 11.1 | 93.4 | bench_sp_ovl_composed_split_rows1_line_b1 |
| composed | **backward** | rows=1 | line | 1 | 0 | **0.609** | 179.1 | 318.8 | 12.5 | 92.8 | bench_sp_ovl_composed_backward_rows1_line_b1 |
| fused / composed | backward | rows=1 | line | 1 | 0 | 0.605 | 175.7 | 319.0 | 11.7 | 93.2 | bench_sp_ovl_fused-bwdcomposed_backward_rows1_line_b1 |
| fused | off (whole grid) | - | line | 1 | 0 | 0.597 | 157.6 | 334.3 | 6.9 | 94.0 | coordinator* |
| nocomm | off (whole grid) | - | ring | 5 | 1 | 2.397 | 555 | 1743 | - | - | coordinator (DESIGN.md) |
| composed | off (whole grid) | - | ring | 5 | 1 | 2.907 | 674 | 2128 | - | - | coordinator (DESIGN.md) |
| fused | off (whole grid) | - | ring | 5 | 1 | 2.882 | 678 | 2097 | - | - | coordinator (DESIGN.md) |
| composed | split | rows=1 | ring | 5 | 1 | 2.908 | 703.6 | 2096.9 | 8.6 | 93.4 | bench_sp_ovl_composed_split_rows1_ring_b5_memeff |
| composed | **backward** | rows=1 | ring | 5 | 1 | **2.814** | 704.3 | 2002.8 | 7.9 | 93.4 | bench_sp_ovl_composed_backward_rows1_ring_b5_memeff |
| fused / composed | backward | rows=1 | ring | 5 | 1 | 2.963 | 785.1 | 2068.9 | 9.8 | 93.4 | bench_sp_ovl_fused-bwdcomposed_backward_rows1_ring_b5_memeff |
| nocomm / composed / fused | off | - | line | 5 | 1 | 2.397 / 3.074 / 2.907 | | | | | coordinator (DESIGN.md) |

Reading the table (gains as ms of the composed-vs-nocomm gap closed):
* Ring B=1: gap 70 ms. Two-queue backward: -17 ms vs the same grid on one queue (19.4 ms out of the backward phase's
  40.8 ms collective cost, 48%), -11 ms net vs whole-grid composed (the 1-row split costs 6-8 ms: forward +9, backward
  -3). 59 ms of the gap remain: ~29 forward collectives (untouched by M1), ~21 backward, ~8 split.
* Line B=1: gap 95 ms. Two-queue backward: -25 ms vs same grid one queue (27.2 of the backward phase's ~52 ms), -20 ms
  net vs whole-grid composed 0.629; whole-grid fused (0.597) is still 12 ms better at B=1 line because its forward gain
  (32 ms) survives only on the whole grid.
* Ring B=5 memeff: gap 510 ms. Two-queue backward: -94 ms (2.908 -> 2.814; 2.907 whole-grid composed; fused 2.882), the
  split is free at this batch (2.908 vs 2.907). The backward phase still holds the recompute forward's 2 AG + 2 RS per
  block (issued on queue 0, not covered) plus the not-hidden part of the backward collectives.
* Fused forward + two-queue backward: worse than composed everywhere except line B=1 (-3.4 ms fwd): the fused ops carve
  their own 2 CCL rows out of the already reduced 12x9 grid (matmuls on 12x7): ring B=1 +8 ms fwd, ring B=5 memeff
  +81 ms fwd and +66 ms bwd (the recompute forward is fused too).

## 4. Verification status
Passed (logs in generated/spfuse/logs, re-run 22:3x-22:59 after the wipe, identical outcomes to the pre-wipe runs):
* `tests/python/test_sp_overlap.py` on 1x2 line, 2 queues, rows=1: 16 passed (`S_ovl_1x2.log`): column linear B=1,2
  bias/no-bias and row linear B=1,2: overlap-on BITWISE == overlap-off, and overlap-off BITWISE == the all_gather+linear
  / linear+reduce_scatter reference sequence (same split grid); 2-layer Llama (H=256, 8 heads, 4 kv, tp-sharded) every
  parameter grad + logits BITWISE overlap on vs off at B=1 and B=2; 3 consecutive backward passes bitwise (buffer window
  + semaphore rotation); fused forward + composed two-queue backward BITWISE vs the same policy on one queue, logits
  bitwise the all-fused, all grads within 2 ULP of all-fused; TP oracle after `ttml.sync_gradients`: synced grads
  BITWISE overlap on vs off and within 2 ULP of the TP model; switch round trips; config knobs.
* `TTML_SP_OVERLAP=backward tests/python/test_sequence_parallel.py`: 41 passed (`S_suite_overlap.log`) -- the whole
  SP suite (TP oracle, LoRA, optimizer step, validation) under the overlap.
* `TTML_SP_TEST_MESH=1x4_ring` / `1x4_line tests/python/test_sp_overlap.py`: 16 passed each (`S_ovl_1x4_ring.log`,
  `S_ovl_1x4_line.log`). First attempt had the TP-oracle LOGITS at 2.50 ULP (ring) / 2.25 (line) against the 2.0
  limit, identical with overlap on and off: the inherent 4-rank reduce-order spread (the 1x2 suite only ever sums two
  ranks); the limit is 4.0 at tp=4 and the test asserts the overlap's own contract (bitwise vs one queue after sync).
* Training losses: not compared bitwise between overlap on/off runs (the pre-wipe logs are gone; the step logs print the
  loss per step, `grep -E "loss" logs/bench_sp_ovl_*.log` on the re-runs).
Not run: bitwise check of a full training step's losses overlap on vs off (`split` vs `backward` logs, same seed);
NoComm-backward machinery-cost pair; CCL region `columns=1` and `rows=2`; batch 2; line batch 5 memeff (was next in
the queue when paused; the `composed_split_rows1_line_b5_memeff` devrun was cancelled while still waiting for the lock).

## 5. Known problems and open questions
* The split grid changes ttnn::linear's auto block config: split vs whole-grid results are ULP-level, not bitwise. The
  bitwise guarantee is overlap-on vs overlap-off on the same grid (mode `split`).
* Residual backward exposure (~21 ms ring B=1 of 40.8; ~25 line of 52): candidates, unquantified because the device
  profiler runs stall: (a) the 1-row region runs the collectives at 2 workers/link (AG 235 vs 185 us, RS 268 vs 234
  ring) -- `rows=2` restores 4 workers at the cost of a second row; (b) per-collective fixed cost: the one-queue
  composed backward costs 43 ms over nocomm for 128 collectives whose device time sums to ~27 ms, i.e. ~125 us of
  launch/host/barrier cost per collective that the overlap does not remove if the host or the queue-0 dispatcher is
  the bottleneck (the NoComm-backward machinery pair measures the schedule's own share of it); (c) the qkv-RS slot
  (behind out_proj's 216 us wgrad) and the first w2-AG of the pass have nothing long enough to hide behind; (d) each
  queue-0 `wait_for_event` stalls the dispatcher's prefetch of the next program.
* Memory: two retained collective buffer sets (`kRetireDepth = 2`) ~ 2 x (RS input + intermediate + output): at B=5
  line topology up to ~0.5 GB. Batch 5 without recompute already OOMs for every implementation (4.24 of 4.27 GB per
  bank), so the retained window may matter for the line B=5 memeff run (not yet run).
* Fused forward on the split grid is a net loss (see section 3): the fused ops cannot use the CCL sub-device's cores (one
  program = one sub-device), so with the split the forward should stay Composed until M2 pipelines it.
* `CCLResources` was previously never reset on close_device (semaphores of a closed device survived a reopen); it is
  now reset. Multi-module test sessions passed, but this is a behaviour change worth a look.
* `AutoContext::enable_ccl_sub_device` has no inverse (no `disable`): a process keeps the split until close_device.
* Two-queue tracing: `enqueue_record_event` TT_FATALs during trace capture; the overlap is incompatible with a traced
  step (train.py does not trace).
* bfp8 collectives (agent F) are dtype-agnostic here: `reduce_scatter_buffers` follows the input tensor's dtype.

## 6. Next steps (in order) and the commands
Environment: `source /home/imichalak/tenstorrent/tt-metal/generated/spfuse/env.sh`; builds `$SPFUSE/build.sh _ttml`;
device runs only through `$SPFUSE/devrun.sh` (dangerouslyDisableSandbox); `$SPFUSE/S_queue.sh` is idempotent (skips
runs whose log holds 6 steps) and covers everything below except the loss check; `python3 $SPFUSE/S_summarize.py
"$SPFUSE/logs/bench_sp_ovl_*.log"` prints the phase table.
1. Finish batch 5 memeff on the line (3 runs, ~7 min each once the lock is free):
   `for m in split backward; do IMPL=composed OVERLAP=$m CCL=rows=1 BATCH=5 MEMEFF=1 $SPFUSE/bench_sp_overlap.sh line 6; done`
   `IMPL=fused BWD_IMPL=composed OVERLAP=backward CCL=rows=1 BATCH=5 MEMEFF=1 $SPFUSE/bench_sp_overlap.sh line 6`
2. Loss bitwise check of a training step, overlap on vs off:
   `grep -E "loss" $SPFUSE/logs/bench_sp_ovl_composed_split_rows1_ring_b5_memeff.log` vs the `..._backward_...` log
   (same seed, same config; they must be identical line by line).
3. Price the machinery (ring B=1): `IMPL=composed BWD_IMPL=nocomm OVERLAP=split CCL=rows=1 $SPFUSE/bench_sp_overlap.sh
   ring 6` vs `... OVERLAP=backward ...`; the backward-phase difference is the events + staging allocations + deferral
   cost with zero collective bytes.
4. CCL region shape (split vs backward): `CCL=columns=1` and `CCL=rows=2`, ring and line, B=1 (8 runs) -- decides
   rows=1 vs rows=2 (4 workers/link) vs a column; compare against the nocomm-split floor of each shape.
5. Batch 2 (`BATCH=2`, split/backward/fused-bwd, ring+line) to complete the batch rows.
6. Schedule improvements to try (each is a small change in sp_linear_ops.cpp / sp_overlap.cpp, bitwise test =
   `$SPFUSE/S_unit_chain.sh`): (a) under recompute, issue the recompute forward's collectives on the CCL queue too and
   let them drain deferred wgrads (4 wgrads per block would then cover 4 of the 8 backward-phase collectives; today
   the recompute forward's 4 are fully exposed at B=5 memeff); (b) cost-based drain (drain until the drained matmul time
   >= the collective's estimate) instead of drain-one, to fix the qkv-RS slot on the line; (c) `kRetireDepth` 1 if
   line B=5 memeff OOMs.
7. M2 (forward micro-batch pipelining for B > 1): chunk the Composed forward of each SP linear along B, issue
   AG/RS(chunk i+1) on the CCL queue while matmul(chunk i) runs on the compute queue, using the same
   `SPOverlap::issue/wait` mechanics (the compute drain must then be per chunk: record after matmul(chunk i-1), not a
   full drain); compare with the whole-grid fused forward at B=2 and B=5 memeff; the forward's 2 AG + 2 RS per block are
   29 ms (ring) / 42 ms (line) of the step at B=1 and ~120-135 ms at B=5, twice that with recompute.
