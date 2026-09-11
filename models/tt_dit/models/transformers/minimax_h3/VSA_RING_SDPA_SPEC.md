# vsa_ring_sdpa: fused ring all-gather + VSA fine-stage attention

Status: IMPLEMENTED, correct, and a (small) win over the two-op path with the multi-worker forwarder and 2 passes
(section 12). Section 11 records the first (parity) version's breakdown. Companion to `VSA_STREAM_DESIGN.md` (section 13 has
the motivating measurements).

## 1. Problem

The VSA fine stage on the 4x8 galaxy (TP=4 on mesh axis 0, SP=8 on mesh axis 1) runs as two blocking
steps per transformer block:

1. `all_gather_async` of K and V over the SP ring (2 links, ~8.1 ms), so every device holds the full
   sequence's K/V in DRAM; nothing overlaps it because the attention depends on all of it.
2. `vsa_sdpa` (~16.5 ms) streaming that full DRAM K/V once per pass by global block id.

Ring attention overlaps the two: K/V shards are forwarded device to device while the attention consumes
the shards that have already landed, with the online-softmax state carried across shards. Section 13 of
the design doc shows the per-shard compute hides the per-shard comm for these selections (each device's
top-k spreads over all 8 shards, heaviest 29-36 %), and the dense path's `exp_ring_joint_sdpa` proves
the fabric-forward + compute overlap works on this hardware.

## 2. Design in one paragraph

`vsa_ring_sdpa` is ONE program per device (no sub-devices, no extra command queues) with two disjoint
core roles: (a) the forwarder role is the existing `ring_attention_all_gather_async` fused helper
(`ring_attention_all_gather_async_multi_core_with_workers_helper`), which appends 2 x num_links sender
cores that forward K/V shards around the SP ring into the persistent gathered DRAM buffers and signal
the consumer per arriving shard; (b) the compute role is the unchanged `vsa_sdpa` streaming
leader/worker engine, whose LEADER now streams the sequence shard by shard in ring-arrival order,
gating each shard on its arrival signal, and reads each block from the local K/V tensor when the block
is in this device's own shard and from the gathered buffer otherwise. Workers and the compute kernel do
not change: they see the same log of global block ids, only in a different (still deterministic) order.

## 3. Interface

```
ttnn.transformer.vsa_ring_sdpa(
    q, k, v, indices, block_counts,
    persistent_output_buffer_k, persistent_output_buffer_v,   # gathered K/V [1,H,T_local*ring,d]
    *, multi_device_global_semaphore,   # 2 GlobalSemaphores: [backward, forward] (all-gather protocol)
    num_links, cluster_axis, mesh_device, topology,
    ccl_core_grid_offset,               # logical (x, y) where the 2*num_links sender cores start
    subdevice_id=None,
    scale=None, block_size=64, compute_kernel_config=None,
    list_len=0, exempt_ids=[], dense_row_mask=None, coarse_slots_shift=0, coarse_real_per_shard=0,
    dense_row_hint=[])
-> [1, H, S_local, d]
```

Tensors (all DRAM interleaved, TILE unless noted):

| tensor | shape | notes |
|---|---|---|
| q | [1,H,S_local,d] bf16 | this device's query rows (S_local multiple of 64) |
| k, v | [1,H,T_local,d] bf16 | this device's K/V shard, T_local multiple of block_size |
| persistent_output_buffer_k/v | [1,H,T_local*ring_size,d] | the CCL manager's all-gather ping-pong buffers; shard s occupies rows [s*T_local, (s+1)*T_local). The local shard is NOT written into it (fused all-gather semantics); the kernel reads it from k/v. |
| indices | [1,H,S_local/64,W] uint32 ROW_MAJOR | global (padded-per-shard) block ids, exactly as `vsa_sdpa` raw-selection mode |
| block_counts, dense_row_mask | as `vsa_sdpa` | global |

Semantics: identical to `vsa_sdpa(q, all_gather(k), all_gather(v), ...)` up to bf16 rounding order
(the sequence is visited shard-major in ring-arrival order instead of ascending; the same class of
difference as `stream_order`). Deterministic: the order is a fixed function of (device_index, ring_size).

Constraints (validated): streaming kernel only (no v1, no distributed, no stream_order); topology Ring;
ring_size == mesh extent on cluster_axis; grid.y >= 2*num_links (sender column); Blackhole.

## 4. Program structure (per mesh coordinate)

A ring op needs one program per device (device_index, neighbors differ), so the op uses the mesh
workload factory pattern (`create_workload_descriptor` per coord + `override_runtime_arguments` that
re-applies hash-excluded args), like `ring_joint_sdpa` / `exp_ring_joint_sdpa`.

Cores on a grid.x x grid.y compute grid:

- Sender (forwarder) cores: `choose_worker_cores(num_links, 2, ..., ccl_core_grid_offset, COL_MAJOR)`
  = 2*num_links cores in ONE column starting at the offset; default offset (grid.x-1, 0), i.e. the
  last column, rows 0..2*num_links-1 (the same placement the model uses for `ring_joint_sdpa`).
- VSA cores: the rectangle [0, grid.x-1) x [0, grid.y): `core_schedule(i, (grid.x-1)*grid.y, H)`
  groups them per head exactly as today (leader = first core of each group).
  Cost: one column (grid.y cores) reserved for 2*num_links senders. Follow-up: remap core ids to
  reclaim the idle cores of that column.

Kernels pushed in this order (indices matter for the cache-hit patch):
0 reader(workers) 1 writer(workers) 2 reader(leaders) 3 writer(leaders) 4 compute, then the helper
appends 5 AG reader fwd, 6 AG writer fwd, 7 AG reader bwd, 8 AG writer bwd.

## 5. Signaling and shard order

- Two program semaphores (init 0) on the VSA core grid: fused_sem[0] (direction 0 = backward chain)
  and fused_sem[1] (direction 1 = forward chain). Receiver cores = the H LEADER cores only (workers
  never wait). `AllGatherFusedOpSignaler` in MULTI mode: when a shard lands, the AG reader of that
  direction increments the direction's semaphore on every receiver core; the forward-direction AG
  writer additionally pre-signals once for the local slice.
- Expected counts: `get_forward_backward_configuration(ring_size, device_index, Ring)` then swap for
  even device_index (identical to the all-gather helper's own derivation); forward_writes_expected =
  num_targets_forward, backward_writes_expected = num_targets_backward (the `ring_joint` write plan).
- The leader consumes shards with `RingSDPAOpReceiver` (fused_op_receiver.hpp) constructed from the
  rt args `RingSDPAFusedOpSignaler::push_ring_sdpa_fused_op_rt_args` emits (ring_size, ring_index,
  fwd_expected, bwd_expected, sem0, sem1, split flag, split shard, split wait): iteration 0 is the
  local shard (no wait), then it alternates directions, waiting on the direction's semaphore for the
  running count. This is the exact, already-debugged protocol the dense ring uses; nothing is
  re-derived here.
- Split forwarding of the diametric shard is DISABLED in v1 (both the helper and the signaler get
  false): the far shard travels whole in one direction. Correct, slightly unbalanced links; enabling
  it later is a flag flip since the receiver already implements the second-half wait.
- Passes: the semaphore counts are monotonic within a launch and program semaphores are re-initialized
  per launch, so each pass constructs a fresh receiver: pass 0's waits gate on real arrivals, passes
  1+ hit thresholds already satisfied and run at full speed from the (now complete) gathered buffer.
  Overlap therefore covers pass 0's compute; see section 8.
- The AG's own GlobalSemaphores are reset to 0 by the AG reader at kernel end (trace-safe, as in the
  standalone op).

## 6. Kernel changes (all under `#ifdef VSA_RING`; the non-ring build is byte-identical)

Reader, leader half (`vsa_sdpa_stream_reader.cpp`):
- extra CT args: gathered-V TensorAccessorArgs; extra rt args (after the worker coords, before the
  pass row counts): gathered_v_addr, ring_index, blocks_per_shard, then the receiver args.
- per pass: `RingSDPAOpReceiver rx(true, argi_copy)`; for step in [0, ring_size):
  `sigma = rx.get_next_ring_id_and_sync()`; for b in [sigma*bps, (sigma+1)*bps): `stream_block(b)`.
- V address: `b / bps == ring_index ? local_v[head*v_local_stride + (b - ring_index*bps)*vtpb + i]
  : gathered_v[head*v_gath_stride + b*vtpb + i]`.
- `stream_order` is rejected in ring mode (the ring order IS the stream order).

Writer, leader half (`vsa_sdpa_stream_writer.cpp`): extra gathered-K accessor + gathered_k_addr,
ring_index, blocks_per_shard; `fetch_one` selects local vs gathered by `block_id / bps`.

Workers and compute: unchanged. Bitmaps, counts, dense rows, exempt ids all index the global block
space already. The worker's arrival-bin windows are fixed functions of the arrival index, so the
visit partition (and hence bf16 rounding) is deterministic for a given device.

## 7. Host

Files:
- `vsa_sdpa_stream_program_factory.cpp`: the descriptor body is extracted into
  `build_vsa_sdpa_stream_descriptor(attrs, inputs, output, const VsaRingContext*)`; the existing
  stream factory calls it with nullptr (no behavior change). With a ring context it: shrinks the grid,
  compiles the vsa kernels with VSA_RING, appends the gathered accessors/args, creates the two fused
  semaphores, builds the signaler, pushes the vsa kernels, then calls the all-gather helper.
- New op `VsaRingSdpaOperation` (`vsa_ring_sdpa_device_operation.{hpp,cpp}`,
  `vsa_ring_sdpa_program_factory.{hpp,cpp}`): params = `VsaSdpaParams` + `RingAttentionAllGatherAsyncParams`
  + `ccl_core_grid_offset`; inputs = `VsaSdpaInputs` + gathered_k/v. Mesh workload factory;
  `create_workload_descriptor` builds one descriptor per coord (device_index, fwd/bwd coords, expected
  counts, VsaRingContext). Hash: vsa fields + ring_size/num_links/topology/cluster_axis/offset + shapes
  and dtypes; NOT the semaphores.
- `override_runtime_arguments`: buffers are re-bound by the descriptor adapter; the op additionally
  re-applies (a) the vsa reader's raw uint32 addresses (dense mask) as today and (b) the AG
  GlobalSemaphore addresses at the helper's fixed slots (AG reader arg 2, AG writer arg 4; forward
  kernels get semaphore[1], backward get semaphore[0]) -- the hash excludes them, so a cache hit with
  the other ping-pong set would otherwise keep the frozen first address (the bug exp_ring documents).
- Wrapper `vsa_ring_sdpa.{hpp,cpp}` + nanobind `ttnn.transformer.vsa_ring_sdpa`; sources.cmake.

Model (`attention_minimax_h3.py`): when `vsa_config.ring`, skip the two all-gathers and call
`vsa_ring_sdpa` with the CCL manager's ag ping-pong buffers/semaphores for the SP axis, `num_links`,
`topology`, `ccl_core_grid_offset=(grid.x-1, 0)`. `MiniMaxH3VSAConfig.ring: bool = False`.

## 8. Expected performance and its ceiling

Per block today: gather 8.1 ms + vsa 16.5 ms serial. The leader re-streams the sequence once per
pass over resident query rows (3 passes at 15 s, 10 resident rows per consumer, L1-bound), and later
passes reuse the row slots, so only pass 0 can overlap arrivals: hidden ~ min(8.1, 16.5/3) ~ 5.5 ms,
i.e. block 59.8 -> ~54.3 ms (~9 %). Reaching 2 passes (rmax 13, shallower ring) would hide ~8 ms.
The AG helper uses both links (forward/backward chains) so per-step comm ~ 1.0-1.2 ms against
~2.1 ms average per-step compute; the shrunken VSA grid costs ~1 column of workers (measure).

## 9. Verification

1. `tests/ttnn/unit_tests/operations/sdpa/test_vsa_ring_sdpa.py` (galaxy mesh, fabric ring params):
   random sharded q/k/v, raw top-k global indices + exempt + dense rows; per device compare
   `vsa_ring_sdpa` against `vsa_sdpa` on the all-gathered K/V (PCC > 0.999, same math up to rounding
   order) and against the torch block-sparse reference; run twice for bit-exact determinism; run under
   trace capture/replay twice with alternating semaphore sets (cache hit + patch).
2. Standalone perf at 15 s shapes on the mesh: `vsa_ring_sdpa` vs `all_gather_async x2 + vsa_sdpa`.
3. Model: traced block test at 15 s with `ring=True`; e2e perf test mode `ring` vs the 3.03 s/step
   baseline.

## 10. Plan

- S1 host refactor (extract descriptor builder), build, non-mesh vsa tests unchanged.
- S2 new op + mesh factory + wrapper + binding, kernels in "wait for all shards then stream" mode
  (proves fusion, signaling, addressing, cache hits, trace) -- correctness on the mesh.
- S3 shard-by-shard overlap in the leader (+ writer K select); correctness, determinism, trace; measure.
- S4 model integration, block test, e2e.
- S5 follow-ups: split forwarding, 2-pass rmax/depth trade, reclaim idle column cores, Linear topology.

## 11. Implementation status and measurements (2026-09-11)

Shipped: `ttnn.transformer.vsa_ring_sdpa` (op files `vsa_ring_sdpa*` under sdpa/device, shared builder
`build_vsa_sdpa_stream_descriptor` + `vsa_sdpa_stream_descriptor.hpp`, kernel changes under `VSA_RING`),
`MiniMaxH3VSAConfig.ring`, tests `test_vsa_ring_sdpa.py` (6 variants, all pass: 1/2 links, eager + traced
replay with the alternate semaphore set, medium shape with dense rows; bit-exact run to run; PCC 0.9997 vs
`vsa_sdpa` on gathered K/V and vs torch) and `test_vsa_ring_sdpa_perf.py`. The traced 15 s block test passes
with `VSA_RING_BLOCK=1` (PCC 100 % vs the untraced reference).

Deviations from sections 4-5: the senders sit on ROW 0 (`ccl_core_grid_offset=(0,0)`, ROW_MAJOR, the
standalone all-gather's placement) and the VSA grid is rows [1, grid.y); split forwarding is ON (the
receiver implements the second-half wait). Both were changed during triage and kept.

Root cause of the development "hang" (worth remembering): kernel handles equal descriptor push order, but
`collect_kernel_meta` iterates an unordered map, so using its index as a handle patched the WRONG kernels on
every program-cache hit -- the VSA reader's buffer addresses landed in the all-gather writer's NoC-coordinate
args and the ready-signal packets went to core (128, 0); the first call always worked, the second deadlocked.
Both cache-hit patches now use push-order handles (constants in `vsa_sdpa_stream_descriptor.hpp`). Also:
`dense_row_hint` rows are LOCAL q tiles (now validated in every mode); consecutive CCLs must strictly
alternate semaphore sets; background shell jobs on this box die at fabric init (run device tests in the
foreground).

Perf at the 15 s per-device shape (random, shard-decaying selection; slowest device; ms per call):

| variant | ms |
|---|---|
| all_gather_async x2 (K, V) alone | 8.4 |
| vsa_sdpa alone, pre-gathered K/V, 120 cores | 15.3 |
| two-op path (gather x2 + vsa_sdpa) | 23.3 |
| vsa_ring_sdpa, 3 passes (default rmax 10 / depth 18) | 26.3 |
| vsa_ring_sdpa, no overlap (TT_VSA_RING_WAIT_ALL=1) | 29.9 |
| vsa_ring_sdpa, 2 passes (TT_VSA_RMAX=15 TT_VSA_DEPTH=12) | 23.7 |

Reading: inside the fused program the ring_attention all-gather helper (ONE worker per direction per link)
moves K+V in ~12.9 ms, vs 8.4 ms for the standalone multi-worker `all_gather_async` (sweep at the K-shard
shape: 1 worker/link 8.4 ms, 2 workers 5.0, 4 workers 4.2 -- per tensor). The overlap hides 3.6 ms with 3
passes and 6.2 ms with 2 passes (pass 0 is the only pass that can overlap, section 8), which brings the fused
op to parity but not a win. The VSA grid also loses one row (108 vs 120 cores, ~+1.5 ms).

Next (the win): replace the forwarder with the multi-worker all-gather's fusable builder
(`build_all_gather_async_minimal_default_program_artifacts`, Program-based, same OpSignaler protocol) over a
concatenated K/V tensor with 2 workers/direction/link (12 sender cores = row 0), run 2 passes, and reclaim the
idle sender-row cores. Expected: ~max(AG 10, pass0 7.6) + pass1 7.6 ~ 18-19 ms vs 23.3 (~4-5 ms/block).

## 12. Multi-worker forwarder (2026-09-11): the win

Forwarder swapped for the standalone `all_gather_async`'s fusable builder
(`build_all_gather_async_minimal_default_program_artifacts`, Program-based; the ring factory materializes the VSA
descriptor into a Program and adds the all-gather's reader/writer to it). K and V travel as ONE tensor: the model
concatenates them on the head dim (`kv = concat([k, v], 1)`, 0.6 ms at 15 s) and the kernels read V of head h at
head H + h in both the local and the gathered buffer (`v_head_offset`). Sender cores: 2 links x 2 directions x
(workers + 1 MUX); with the default 2 workers/link that is exactly row 0, the VSA grid is rows 1..9 (108 cores).
The per-device ring constants are COMMON runtime args (the gathered address is re-applied on cache hits); the
receiver runs WITHOUT split forwarding because the fused all-gather disables it (`if constexpr (topology == Ring &&
!fuse_op)` in its reader) -- expecting the second-half signal deadlocked the last shard. Ring-mode defaults: 2 passes
(rmax 15, ring depth 14), which also fits the traced 15 s block's L1.

Perf at the 15 s shape (slowest device, ms per call; `test_vsa_ring_sdpa_perf.py`):

| variant | ms |
|---|---|
| concat(k, v) (the fused path's extra op) | 0.6 |
| all_gather_async x2 alone | 8.4 |
| vsa_sdpa alone (120 cores, default 3 passes) | 15.3 |
| two-op path (gather x2 + vsa_sdpa) | 23.3-23.5 |
| vsa_ring_sdpa, 3 passes (10/18), 2 workers | 24.3 |
| vsa_ring_sdpa, 2 passes (15/12) | 22.0 |
| vsa_ring_sdpa, 2 passes (15/14) = default | 21.7-21.8 |
| vsa_ring_sdpa, 2 passes, no overlap (WAIT_ALL) | 28.2 |

Net: ~1.1 ms/block including the concat (~1.7 without). The fused op is comm-bound in pass 0: the in-program
all-gather takes ~11.5 ms (28.2 - ~16.7 of 108-core attention) against ~8 ms of pass-0 compute, so the overlap hides
~6.5 ms and the model is max(11.5, 8) + 8 + ~1.7 overhead. Headroom: (a) fuse the concat away (emit K/V
concatenated upstream), (b) a faster in-program gather: 4 workers/link needs 20 sender cores (spills into row 1 and
currently HANGS -- parked; the fix is an irregular VSA core set that reclaims the rest of row 1), (c) finer overlap
(let pass 1 start on early shards).

Correctness: 6 unit variants pass (1/2 links, eager, traced replay with the alternate semaphore set, medium shape
with dense rows), bit-exact run to run, PCC 0.9997 vs vsa_sdpa on gathered K/V and vs torch; traced 15 s block test
passes with `VSA_RING_BLOCK=1` (PCC 100 %).

Gotchas hit: kernel handles == push order but `collect_kernel_meta` is unordered (never index it as a handle);
`dense_row_hint` rows are local; consecutive CCLs must alternate semaphore sets; K/V must be concatenated PER
DEVICE (per TP shard) -- concatenating over all heads and then sharding hands device 0 only K heads.
