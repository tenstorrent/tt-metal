# GLM-5.2 sparse-MLA top-k / KV all-gather overlap plan

## Objective

Overlap `ttnn.experimental.topk_large_indices` with the SP-axis KVPE-prefix
`ttnn.experimental.high_bw_all_gather` that immediately precedes the main
`ttnn.transformer.sparse_sdpa` in
`models/demos/deepseek_v3_d_p/tt/mla/mla.py`.

The production target is Blackhole Galaxy with a `12 x 10` (120 Tensix) worker
grid per chip, with LoudBox used for production-shaped qualification:

- 80 Tensix cores for the top-k/index branch;
- 40 Tensix cores for the KVPE-prefix all-gather branch;
- no shared Tensix core, circular buffer, or L1 allocation between branches;
- one host command queue, using Fast Dispatch sub-device counters for device-side
  concurrency (the same mechanism already used by the MoE shared-expert/dispatch
  overlap).

The local QB2 development machine exposes an `11 x 10` (110 Tensix) worker grid,
so it uses an 80-core top-k / 30-core gather proxy. QB2 proves functionality,
core isolation, and repeated overlap locally, but it cannot qualify the final
80/40 production performance split. The production split is tested on LoudBox
and signed off on Galaxy.

## Scope and non-goals

In scope:

- GLM-5.2 sparse chunked prefill, specifically a `full` indexer layer that
  computes new indices;
- the `topk_large_indices` operation API and program factory;
- the existing `high_bw_all_gather` sub-device contract and its overlap
  qualification;
- sparse-MLA scheduling, buffer lifetime, teardown, correctness, and production
  performance coverage.

Initially out of scope:

- dense MLA;
- GLM-5.2 `shared` indexer layers, which reuse prior indices and therefore have
  no top-k to overlap (their KV gather remains unchanged);
- `kv_only` layers, decode, non-Blackhole architectures, and SP=1;
- sparse-MLA trace capture, which the model currently rejects before entering
  MLA. The eager implementation must not weaken that guard. Trace support can
  reuse the existing segmented `SubDeviceTraceController` after sparse metadata
  support is implemented separately.

## Current-state findings

1. `TtIndexer.forward()` currently runs the complete indexer, including
   `topk_large_indices` and the TP index redistribution, before `ttMLA.forward()`
   builds MLA Q/KV. The KV cache update and KVPE-prefix gather happen later in
   `_sparse_chunked_attn()`. Disjoint cores alone would not create the desired
   overlap window because full-grid Q/KV programs sit between the two target
   operations.

2. `topk_large_indices` always creates kernels and circular buffers on the full
   `compute_with_storage_grid_size()` and uses that same rectangular grid when
   splitting rows. It has no `subdevice_id` or `sub_core_grids` API.

3. `high_bw_all_gather` already accepts `subdevice_id` and `sub_core_grids`
   (`sub_core_grid` internally), intersects them with the selected sub-device's
   worker cores, and creates its workers/muxes in that intersection. It does
   not, however, validate that the supplied grid is wholly contained by one
   loaded sub-device.

4. `high_bw_all_gather` is already host-enqueued like other TTNN device
   operations and its workers synchronize the collective internally. Its
   program-cache-miss path is not independently asynchronous: the factory
   allocates `ready_sem` and `data_valid_sem`, then calls
   `distributed::Synchronize()` on the selected sub-device before creating the
   workload. The op needs a caller-owned, preinitialized semaphore path so MLA
   can move that allocation/synchronization to model setup. It should not be
   renamed solely to add an `_async` suffix; the existing entry point will gain
   the explicit async-resource contract.

5. The KVPE gather sets `input_batch_index`/`gathered_dim_size`, so it always
   takes the runtime-controlled direct schedule and disables fixed-shape
   bank-owned selection. On Blackhole it starts at two workers/direction (a
   12-core scheduler budget for two links) and selects eight (a 36-core budget)
   only when the worst-case
   output allocation is at least 32 MB/link and the resolved input page is at
   least 2 KB. If eight is preferred, 40 cores admit 36 while a 30-core cap
   falls back to four workers/20 cores. Otherwise both ownership profiles may
   use only the 12-core budget. Actual instantiated cores are topology dependent:
   a two-rank line endpoint creates eight worker plus two live-direction mux
   cores (10 total), while the scheduler reserves mux capacity for both
   directions. The threshold uses worst-case output allocation,
   not the growing active prefix, so the choice is stable after compilation but
   must be measured with the real GLM-5.2 KVPE format/shape before fixing the
   profile.

## Target execution graph

Refactor only the sparse/full-indexer path so the enqueue order is:

```text
default/full-grid manager (serial)
    q_a latent
      -> indexer K write + Q/weights + ring_indexer_score_dsa -> logits
      -> MLA q_stem + kv_stem
      -> update_padded_kv_cache

load sparse-MLA overlap manager
    top-k sub-device (80 cores)       gather sub-device (40 prod / 30 QB2)
    ---------------------------       ----------------------------
    topk_large_indices(logits)   ||   high_bw_all_gather(KV prefix)

clear sparse-MLA overlap manager (full-grid device-side join/reset)
    -> RM->TILE -> TP index all-gather -> TILE->RM (if TP>1)
    -> sparse_sdpa(q, gathered KV, finalized indices)
    -> wkv_b2 + output epilogue
```

The critical enqueue sequence inside the overlap region is deliberately:

1. enqueue local top-k on the 80-core sub-device;
2. enqueue the KVPE-prefix gather on the 40-core production sub-device (30 on
   QB2) immediately;
3. clear the overlap manager; the Fast Dispatch worker-state reset drains both
   sub-device completion counters and restores the default full-grid manager;
4. enqueue the dependent top-k TP redistribution and sparse SDPA on the default
   manager.

This puts the KV gather on its independent dispatch counter while local top-k
is compute-bound, but does not attempt to run the TP-axis index gather
concurrently with the SP-axis KV gather. It preserves the dependency from
`ring_indexer_score_dsa` to top-k and from the KV-cache update to the gather,
and avoids placing any full-grid program inside a two-sub-device region.

## Implementation phases

### Phase 0: Baseline and overlap proof harness

- Add a Blackhole-only operation test alongside
  `tests/ttnn/unit_tests/operations/experimental/test_high_bw_all_gather.py`
  that creates two non-overlapping sub-devices, warms both program caches, then
  enqueues a production-shaped `topk_large_indices` and
  `high_bw_all_gather` on one CQ.
- Warm the exact sub-device/core-grid program hashes used by the overlap test.
  Before Phase 2, record that legacy `high_bw_all_gather` allocates global
  semaphores and synchronizes its selected sub-device on a cache miss; a
  full-grid warmup does not warm the 40/30-core hash. After Phase 2, prove the
  caller-owned-semaphore path performs no operation-internal device sync even
  on its first program creation. Performance overlap is still measured after
  exact-hash warmup so host compilation is not mistaken for device latency.
- Record isolated top-k, isolated gather, and combined device intervals with
  the realtime profiler. Preserve raw start/end timestamps; summed per-program
  duration is insufficient to prove overlap.
- Verify all output values, run with Watcher in the correctness variant, and
  repeat enough iterations to expose stale semaphore state or accidental
  serialization.
- Prove the post-manager full-grid consumer sees both outputs without a host
  `synchronize_device()`. Record the existing mechanism explicitly:
  `clear_loaded_sub_device_manager()` restores and resets worker state and enqueues a
  wait for every sub-device's expected worker completion before the default
  full-grid program runs. Both entry and exit manager switches are full-grid
  drain/reset boundaries; nothing is expected to overlap across them.
- Enumerate every program enqueued while the two-sub-device manager is loaded.
  The production list must contain exactly local `topk_large_indices` on the
  top-k strip and KVPE `high_bw_all_gather` on the gather strip. Add a debug
  assertion that each program's core set is a subset of exactly one owning
  sub-device: tt-metal programs currently support one sub-device only, while a
  full-grid program can geometrically intersect both strips and be accounted to
  only one dispatch counter.
- Capture the baseline core placement and durations before changing
  `topk_large_indices` so the 80-core restriction and overlap benefit can be
  compared against the present full-grid implementation on both 110- and
  120-core devices.

Exit criteria:

- output correctness is unchanged;
- profiler intervals for top-k and gather have a positive intersection;
- the combined interval is less than the sum of isolated intervals after warmup;
- no program in either branch uses a core owned by the other branch.

#### Lightweight-profiler concurrency support

The device profiler must retain one interval per dispatched program even when
independent subdevices complete out of dispatch order. The implementation:

- patches the exact 16-bit runtime ID into each dispatch-s go command instead
  of correlating timestamps through the asynchronous legacy ID FIFO. IDs wrap
  modulo 65,536; dispatch-s permits only one active program per stream, and
  this workload has at most eight streams and fewer than 65,536 commands in
  flight, so two active records on one stream cannot share an ID;
- tracks active runtime ID, start tick, completion count, and end tick
  independently for every dispatch-s worker stream;
- timestamps every completion-counter transition from the dispatch-s compute
  monitor, then selects the transition matching the exact cumulative
  `wait_count` in the next go/flush command. A later flush on another
  subdevice therefore cannot inflate the interval, and programs need not use
  every worker owned by their subdevice;
- queues completed records in dispatch-s L1 and drains them through an
  acknowledged transport, preventing two close completions from overwriting
  the former shared A/B notification slot. The queue holds 128 records; when
  full, dispatch-s drains and stalls for acknowledgment rather than dropping
  or overwriting a record;
- exposes raw `start_timestamp`, `end_timestamp`, and calibrated `frequency`
  through `profile_realtime_program`, in addition to the derived duration.

The op-level regression requires distinct top-k and gather runtime IDs on
every chip, positive raw-tick intersection, and at least 90% of each gather
interval to lie inside top-k. The end-to-end harness computes a device-only
forward span locally on each chip (`latest end - earliest start`) and takes the
maximum span across chips; it never compares absolute clocks between chips or
adds overlapping program durations. Frequency calibration is carried on every
record and applied within its chip before cross-chip durations are compared.
The helper fails on receiver drop accounting, and qualification asserts equal
per-chip counts plus a stable total count across identical samples, so missing
records cannot silently become a shorter span.

### Phase 1: Make `topk_large_indices` sub-device/core-grid aware

Update:

- `ttnn/cpp/ttnn/operations/experimental/topk_large_indices/topk_large_indices.hpp`;
- `topk_large_indices_nanobind.cpp`;
- `device/topk_large_indices_device_operation_types.hpp`;
- `device/topk_large_indices_device_operation.hpp/.cpp`;
- `device/topk_large_indices_program_factory.hpp/.cpp`;
- `tests/ttnn/nightly/unit_tests/operations/experimental/test_topk_large_indices.py`.

API:

```python
ttnn.experimental.topk_large_indices(
    logits,
    k=...,
    valid_length=...,
    subdevice_id=topk_sd_id,
    sub_core_grids=topk_cores,
)
```

Required behavior:

- Resolve available cores as the selected sub-device's Tensix workers,
  intersected with `sub_core_grids` when supplied.
- Require the selected sub-device manager to be loaded at invocation time.
  Reject an empty intersection and reject any supplied core outside the loaded
  sub-device rather than silently falling back to the full device.
- Use the `CoreRangeSet` overload of `split_work_to_cores` both at program
  creation and runtime-argument override. Create kernels/CBs only on that set;
  do not create idle CBs on the gather-owned 40/30 cores.
- Include the resolved structural core set and sub-device selection in the
  program-cache key. Continue excluding `valid_length`, which is runtime-only.
- Preserve the current row-major ordering, output shape/dtype, LLK K snapping,
  sentinel semantics, and cache-hit shape rebinding.
- Add full-grid backward-compatibility tests plus non-origin, rectangular, and
  discontiguous `CoreRangeSet` tests. Assert profiler-reported instantiated cores stay
  inside the requested set and test cache separation between 120-, 110-, and
  80-core programs.
- For non-origin/discontiguous sets, inspect the per-core `(start_row, rows)`
  runtime arguments and prove their traversal order matches the
  `corerange_to_cores(..., row_wise=True)` vector used to accumulate
  `start_row`. Output-only checks can miss a duplicated or swapped row on some
  shapes.
- Document that core-grid selection becomes structural: unlike the current
  deliberately full-grid program, an 80-core cached program cannot later bind
  a different active core subset under the same hash.

### Phase 2: Add an explicit async-resource interface to `high_bw_all_gather`

This phase changes the operation itself. Today the factory owns semaphore
allocation and executes an operation-internal sub-device synchronization on a
program-cache miss. The new path lets a model allocate and initialize those
resources before entering its latency-critical overlap region.

#### Files to change

- `ttnn/cpp/ttnn/operations/experimental/high_bw_all_gather/high_bw_all_gather.hpp`;
- `high_bw_all_gather.cpp`;
- `high_bw_all_gather_nanobind.cpp`;
- `device/high_bw_all_gather_device_operation_types.hpp`;
- `device/high_bw_all_gather_device_operation.hpp/.cpp`;
- `device/high_bw_all_gather_unicast_factory.hpp/.cpp`;
- `ttnn/cpp/ttnn/operations/experimental/high_bw_all_gather/README.md`;
- `tests/ttnn/unit_tests/operations/experimental/test_high_bw_all_gather.py`.

#### Public Python interface

Extend the existing operation rather than adding a second `_async` symbol:

```python
ttnn.experimental.high_bw_all_gather(
    input_tensor,
    dim,
    output_tensor,
    *,
    cluster_axis,
    subdevice_id=None,
    sub_core_grids=None,
    num_links=None,
    input_batch_index=None,
    gathered_dim_size=None,
    ready_semaphore=None,
    data_valid_semaphore=None,
)
```

Contract:

- `ready_semaphore` and `data_valid_semaphore` are optional persistent
  `ttnn.GlobalSemaphore` handles. They must be supplied together or both be
  omitted.
- When omitted, preserve the current legacy behavior exactly: the factory
  allocates both semaphores and performs its one-time selected-sub-device
  synchronization on a program-cache miss.
- When supplied, the caller guarantees they were created and initialized to
  zero before dispatch. The op reuses them and must not allocate a semaphore or
  call `distributed::Synchronize()` in either cache-miss or cache-hit dispatch.
- The two handles must be distinct, belong to the input/output mesh device,
  and cover a `CoreRangeSet` containing every resolved reader/writer worker
  core. The model creates them over the whole 40/30-core gather strip; mux cores
  do not access these semaphores and need no semaphore coverage. The selected
  sub-device manager must be loaded, and `sub_core_grids` must be wholly
  contained in that one sub-device.
- Select the semaphore buffer type exactly as the legacy factory does:
  explicitly request `L1_SMALL` when the device reserves it, otherwise `L1`.
  Validate the supplied handles against that resolved type. Do not rely on
  `ttnn.create_global_semaphore`'s default `L1` argument.
- Mesh-created semaphore handles are required. Validate
  `sem.device() == static_cast<IDevice*>(mesh_device)`; a semaphore created on
  one individual chip is invalid because Fabric writes the same semaphore
  address into neighboring devices.
- Kernels retain the existing self-reset contract: every invocation consumes
  its ready/data-valid counts back to zero. Handles therefore remain reusable
  across serial layers and growing-prefix cache hits. Concurrent gathers may
  not share a pair; supporting multiple simultaneously requires separate pairs.

`high_bw_all_gather` still returns the caller-provided persistent output tensor.
The semaphore handles make dispatch independently asynchronous; they do not
change output readiness semantics. A consumer on another sub-device still
needs the manager-clear/full-grid join described above.

#### C++ operation and cache contract

- Thread optional `GlobalSemaphore` handles through the public function,
  primitive function, `high_bw_all_gather_build_operation_args`, and device
  operation attributes.
- Add a structural `uses_external_semaphores` bit to the program hash. Do not
  hash semaphore addresses: on a cache hit, patch the current handle addresses
  from the current operation attributes into every reader/writer runtime
  argument. Retain handle copies in `shared_variables_t` only for lifetime
  safety; never use the cached copies as the source of pair A/B cache-hit
  addresses.
- Amend `HighBwAllGatherParams`' current comment that says semaphores belong
  only in factory shared variables. Store the optional handles in attributes,
  deliberately exclude their identities/addresses from the custom hash, and
  hash only external-pair presence.
- Split validation by what each layer can observe:
  - `validate_on_program_cache_miss/hit`: both-or-neither presence, distinct
    addresses, mesh identity, and buffer type. Read semaphore core/type metadata
    through `GlobalSemaphore::attribute_values()` because there are no direct
    `cores()` or `buffer_type()` accessors.
  - factory cache miss: loaded sub-device containment, semaphore coverage of
    resolved workers, and scheduler-capacity checks;
  - factory cache hit: recheck current-handle coverage against
    `shared_variables_t.worker_cores` before patching addresses. Scheduler
    capacity is structural and need not be recomputed on a hit.
- In `HighBwAllGatherUnicastFactory::create_mesh_workload`, branch explicitly:

  ```text
  legacy:   allocate ready/data-valid semaphores -> Synchronize(selected SD)
  external: use supplied handles                -> no allocation, no Synchronize
  ```

- Keep `input_batch_index` and `gathered_dim_size` values hash-excluded and
  runtime-patched. Keep semaphore addresses hash-excluded and runtime-patched;
  hash only whether the external-resource path is selected.
- Factor the currently duplicated sub-device/core-grid resolution into one
  helper used for semaphore coverage, scheduler tier selection, and
  `choose_worker_cores`. Replace silent partial intersection with a fatal
  containment check. Confirm every reader, writer, mux, and CB is placed on
  exactly that gather sub-device and every reader/writer is covered by both
  semaphores.
- Permit the legal full-grid case where external handles are supplied without
  explicit `subdevice_id`/`sub_core_grids`; resolve and validate it against the
  default sub-device's complete worker set.
- Update the README with lifetime, zero-initialization, no-concurrent-reuse,
  worker-core-coverage, warmup, failure recovery, and consumer-join
  requirements.

#### Model-side resource creation

Add a `TT_CCL` allocator/cache for one high-bandwidth gather semaphore pair per
`(subdevice_id, CoreRangeSet, buffer_type)` overlap profile:

- create the pair on the 40-core Galaxy/LoudBox gather strip or 30-core QB2
  gather strip during model construction;
- initialize both to zero, then perform one mesh-wide barrier covering every
  device participating on `cluster_axis` before any overlap manager region or
  trace/profiler warmup. Do not substitute a gather-sub-device-only barrier:
  zero initialization must be visible before any neighbor's Fabric atomic
  increment reaches the mesh-uniform address;
- retain the handles for the model lifetime and release them only after all
  cached programs and the overlap sub-device manager are drained/removed;
- pass the pair only to the SP KVPE-prefix gather. The later TP index gather
  remains outside the overlap region and may retain the legacy path initially.
- on an aborted/timed-out iteration, do not reuse the pair with unknown residue.
  The recovery/teardown path must drain the device and call
  `reset_global_semaphore_value(handle, 0)` for both handles before reuse or
  destruction.

Dropping the per-call factory synchronization is safe after that one-time
initialization barrier because the kernels already contain the steady-state
cross-device readiness handshake: each writer Fabric-increments its neighbor's
partner-core ready semaphore only after earlier queue work/output initialization,
and each relay reader waits for and consumes that ready count. The data-valid
path likewise consumes its completion count to zero at the end of a successful
invocation.

#### Op-level tests

Extend `test_high_bw_all_gather.py` with tests independent of sparse MLA:

1. **Backward compatibility:** omitted handles produce bit-exact results for
   the existing row-major/tile, line/ring, full-tensor, selected-batch, and
   partial-prefix cases.
2. **External-resource equivalence:** the same matrix with caller-owned handles
   matches the legacy path and Torch reference.
3. **Interface validation:** reject only-one-handle, identical handles, wrong
   mesh device (including a per-chip rather than mesh-created handle), worker
   cores not covered by the semaphore set, wrong buffer type, unloaded/wrong
   `subdevice_id`, a grid spanning two sub-devices, and a grid too small for any
   legal scheduler tier. Also accept and test external handles over the default
   full-grid sub-device with no explicit sub-device/grid arguments.
4. **No hidden synchronization/allocation:** assert L1-small/L1 memory-view
   allocation is unchanged across an external-path cache-miss call. Prove the
   sync behavior with either a C++ gtest/injectable `Synchronize` counter or a
   behavioral test: a long program already running on the other sub-device is
   not drained before the external gather starts, as shown by device
   timestamps. Retain legacy cache miss as the control.
5. **Program-cache behavior:** alternate external semaphore pair A and pair B
   at the same shape/core profile. Assert one external-path program-cache entry
   is reused and reader/writer runtime arguments receive the current attributes'
   pair addresses rather than cached pair A. Assert legacy and external paths
   do not alias.
6. **Semaphore reuse:** run at least 100 consecutive gathers while alternating
   nonzero `input_batch_index` and increasing/decreasing
   `gathered_dim_size`. Verify every output; successful repeated completion is
   the behavioral zero-residue proof because Python has no semaphore-value read
   API. Add an abort/recovery test that drains, explicitly resets both handles
   to zero, and then completes correctly with the same pair.
7. **Core confinement:** use non-origin 40- and 30-core sub-devices and assert
   every worker/mux/CB/semaphore core belongs to the selected strip; no program
   may span both loaded sub-devices.
8. **Independent progress:** warm exact hashes, enqueue production-shaped top-k
   on the other sub-device, then invoke external-resource gather. Realtime
   profiler start/end timestamps must overlap; repeat under Watcher/stress to
   catch stale counts and deadlocks.
9. **Fabric qualification:** repeat independent-progress and semaphore-reuse
   tests for the line/ring schedules used by Galaxy, LoudBox, and QB2. Confirm
   concurrent top-k DRAM traffic does not starve Fabric progress.
10. **Scheduler budget and confinement:** assert the selected
    workers/direction tier and conservative capacity requirement in the device
    qualification tests and manifests. Validate that the configured 40- and
    30-core strips contain the scheduler budget without extending profiler
    metadata solely to report program placement.

#### 40-core production and 30-core local-proxy tuning gate

Profile these warmed configurations at the real GLM-5.2 KVPE shape:

- unrestricted gather;
- 80/40 production overlap on LoudBox and Galaxy (40 gather-owned cores);
- 80/30 local overlap on QB2 (30 gather-owned cores).

The 30-core QB2 result is a functional/local proxy and must not override the
80/40 production decision. For both BF16 and scaled-FP8 KVPE formats, record
worst-case output bytes/link, resolved input page size, preferred
workers/direction, fallback tier, and scheduler core budget. Expected budgets
are 8 workers/36 cores
on a 40-core strip and 4 workers/20 cores after a 30-core fallback only when
the real shape clears the high-parallelism predicates; otherwise the
runtime-controlled path may use 2 workers with a 12-core budget on either
strip. A line endpoint instantiates fewer muxes than that conservative budget.
If that wastes enough of the 40-core production ownership to lose
end-to-end performance, qualify a scheduler heuristic change separately rather
than assuming an unmeasured worker tier.

### Phase 3: Split index scoring from index selection

Refactor `models/demos/deepseek_v3_d_p/tt/mla/indexer.py` without changing the
public result contract:

- Extract the work through `ring_indexer_score_dsa` into a method such as
  `score(...)->IndexerSelectionState`. The state owns `logits`,
  `topk_valid_length`, and the TP/SP distribution metadata needed later.
- Extract `topk_large_indices` into `select_local(state, subdevice_id,
  sub_core_grids)`.
- Extract the RM->TILE->TP-gather->RM path into
  `finalize_distribution(local_indices)`. Invoke it only after clearing the
  overlap manager, on the restored default grid. Its TP-axis
  `high_bw_all_gather` must not run concurrently with the SP-axis KV gather.
- Keep `TtIndexer.forward()` as a sequential compatibility wrapper around the
  three stages for direct tests and non-overlap callers.
- Make ownership explicit: logits live until local top-k has consumed them;
  local indices live until redistribution completes; TT_CCL-owned persistent
  gather buffers must never be deallocated by a layer.
- Add a confinement assertion/helper used by `select_local`: every program
  enqueued while the overlap manager is loaded must be wholly contained by one
  sub-device. Do not rely on the union of both strips covering the full grid;
  dispatch currently accounts a program to a single intersecting sub-device.

Add stage-level tests comparing the split and compatibility paths exactly for
GLM-5.2 `full` layers, including early prefixes with `0xFFFFFFFF` sentinels,
nonzero compact index-cache slots, TP=1, and TP=4.

### Phase 4: Create and own the production and local sub-device layouts

Add sparse-MLA overlap configurations with explicit `CoreRangeSet`s, not just
counts. Use vertical strips so both profiles keep the same 80-core top-k
geometry:

```text
Galaxy/LoudBox 12x10 production:
  top-k/index branch:  x=[0..7],  y=[0..9]  -> 80 cores
  KV gather branch:    x=[8..11], y=[0..9]  -> 40 cores

QB2 11x10 local proxy:
  top-k/index branch:  x=[0..7],  y=[0..9]  -> 80 cores
  KV gather branch:    x=[8..10], y=[0..9]  -> 30 cores
```

Placement rules:

- Derive and validate against `mesh_device.compute_with_storage_grid_size()`;
  fail closed if the production profile is requested on a different grid.
- Create one two-sub-device manager per mesh/profile through the shared
  `TT_CCL`, not one manager per MLA layer. All serial sparse layers reuse it.
- Create it only when overlap is enabled and at least one local layer is a
  non-`kv_only` `full` indexer layer.
- Create the manager with `local_l1_size=0`, as the existing MoE overlap does.
  Isolation comes from program-local CB/mux allocations on disjoint cores; a
  nonzero sub-device local-L1 reservation shrinks the global allocator and
  prevents safe manager switching while a local allocation is live.
- Add idempotent release to the model teardown path before mesh close. The
  concrete ownership path is: `TT_CCL.release_sparse_mla_overlap_manager()` is
  called once from `TtPrefillTransformer.release_sub_device_managers()` after
  its existing `clear_loaded_sub_device_manager()`. If block-only construction
  can own the manager, also forward block teardown to MLA/TT_CCL. Do not leave a
  registered manager for the mesh destructor, and do not remove a manager while
  it is loaded.
- Audit `TT_CCL` shared scratch keys used by both concurrent branches. Today the
  KVPE gather output, indexer logits/indices, and TP gather outputs use disjoint
  keys, but add construction-time assertions and amend the "layers run
  serially" docstrings to state that no buffer key may be shared across the two
  intra-layer concurrent branches.
- Keep an explicit rollout flag/config parameter. Default it off on unsupported
  hardware/mesh shapes; select 80/40 for the 120-core Galaxy/LoudBox profile and
  80/30 for the 110-core QB2 local profile after Phase 6 passes.
- Add negative tests proving overlap stays disabled for SP=1 and fails closed
  when a named 120/110-core profile is requested on a different worker grid.
  SP=1 uses a `ttnn.slice` prefix path, which is not confined to either overlap
  sub-device.

### Phase 5: Integrate the overlap at sparse-MLA level

Refactor `models/demos/deepseek_v3_d_p/tt/mla/mla.py` so a `full` sparse layer
does the following:

1. Compute `qr` and the indexer score state on the default full-grid manager.
2. Compute `_q_stem()` and `_kv_stem()` while still on the default manager.
3. Move the sparse cache update out of `_sparse_chunked_attn()` into a prepare
   step. Compute the selected slot and rounded `gathered_dim_size` there, but do
   not gather yet.
4. Load the shared sparse-MLA overlap manager.
5. Enqueue local top-k on the 80-core branch.
6. Enqueue `_gather_kvpe_prefix()` on the 40-core production branch (30 on the
   QB2 proxy), passing its `subdevice_id`, `sub_core_grids`,
   `ready_semaphore`, and `data_valid_semaphore` to `high_bw_all_gather`.
7. Clear the manager, which drains/joins both sub-device counters and restores
   the default full-grid worker state.
8. Enqueue the dependent RM->TILE->TP-gather->RM index redistribution, then call
   `_sparse_mla()` on the default grid.

Keep dense, shared-indexer, SP=1, disabled-overlap, and compatibility paths in
their current serial order. Centralize manager load/clear in a small exception-
safe helper so a Python exception cannot leave the mesh on the wrong manager.
Do not insert a host sync or read an output tensor on the host.

The overlap region has a closed allowlist of two device programs. Do not move
`to_layout`, TP `high_bw_all_gather`, `slice`, or any other full-grid/default op
inside it without first adding single-sub-device core control and an explicit
confinement proof.

Preserve these lifetimes across the overlap:

- `logits` until top-k consumption;
- the just-written KV cache and TT_CCL-owned
  `mla_sparse_kv_gather_buffers` entry (referenced by MLA as
  `_sparse_kv_gather_buffer`) until sparse SDPA finishes;
- `tt_q` and finalized indices until sparse SDPA consumes them;
- all TT_CCL persistent buffers across layers.

Update signposts/profiler labels to mark `SPARSE_MLA_OVERLAP_START/END`, local
top-k, KV gather, and index redistribution. This gives both Tracy and realtime
profiler tooling an unambiguous overlap region.

### Phase 6: End-to-end validation and production gate

#### Correctness

- Extend `test_sparse_mla_vs_trace.py` or add a focused neighbor to compare
  overlap off/on for GLM-5.2 `full` layers. Despite the historical filename,
  run sparse eager mode only while the trace guard remains.
- Cover cold, warm, and long prefixes; BF16 and scaled-FP8 KV caches; compact
  nonzero index-cache slots; TP=4; line/ring fabric; and at least two consecutive
  forwards to catch semaphore reuse bugs.
- Run nonzero `input_batch_index` and increasing `gathered_dim_size` across
  cache hits with overlap enabled, not only in the standalone gather test. Both
  values are hash-excluded runtime arguments and are the production growing-
  prefix path.
- Compare top-k index tensors exactly where inputs have deterministic ordering,
  compare selected values for tie-heavy cases, and retain the existing MLA PCC
  threshold for final output.
- Run Watcher and device timeout/deadlock stress before enabling production.

#### Performance

Extend
`models/demos/deepseek_v3_d_p/tests/sparse_mla/test_sparse_mla_perf.py` and the
CCL microbenchmark to report:

- top-k branch interval;
- KV gather interval;
- their intersection and overlap ratio;
- union (critical-path) time versus the serialized sum;
- whole sparse-MLA forward time;
- core profile and actual gather worker tier;
- manager load/clear/reset time per overlap region and accumulated across all
  full-indexer layers/chunks, using
  `models/demos/deepseek_v3_d_p/tests/op_unit_tests/test_sub_device_load_clear_timing.py`
  as the existing measurement pattern.

Measure isolated full-grid top-k, isolated 80-core top-k, 40-core gather,
80/40 overlap on LoudBox and Galaxy, and the 80/30 local proxy on QB2. Use
GLM-5.2 production shapes and fabric configuration. LoudBox is the main
qualification vehicle, but final production approval requires a Galaxy run;
QB2 measurements do not substitute for either.

Production acceptance requires all of:

- exact-correctness and PCC suites pass with overlap enabled and disabled;
- device timestamps prove concurrent execution, not merely adjacent host calls;
- the Galaxy/LoudBox 80/40 overlap union is at least 10% shorter than the warmed
  serialized top-k+gather sum;
- after including two manager switches for every full-indexer layer and chunk,
  whole sparse-MLA and whole-forward latency improve versus overlap disabled in
  cold, warm, and long scenarios; the per-region union metric alone cannot
  approve production;
- no new device timeout, semaphore leak, persistent-buffer leak, or teardown
  failure over a multi-layer/multi-iteration stress run;
- qualification evidence records the chosen firmware, fabric config, core
  split, active worker tier, and baseline rather than baking an unexplained
  duration into the test.

#### LoudBox/Galaxy qualification runbook

After committing the exact code under test, run the checked production driver:

```bash
models/demos/deepseek_v3_d_p/tests/sparse_mla/run_sparse_mla_overlap_qualification.sh \
  loudbox_80_40 "<installed-firmware-bundle>"

# Repeat on Galaxy with:
#   galaxy_80_40 "<installed-firmware-bundle>"
```

The driver requires a clean tracked worktree, executes the correctness and
Tensix-Watcher cases, then runs all six format/scenario serial-overlap pairs.
It rejects skips, stale manifests, wrong device counts, non-schema-v8 output,
wrong commit/firmware provenance, incorrect configured ownership, a branch
win below 10%, or a non-positive whole-forward win. It archives manifests,
per-op CSVs, JUnit files, and a six-case `summary.json` under
`generated/profiler/qualification/`. The commands below document the expanded
manual equivalent and remain useful for rerunning one failed case.

Schema v8 fingerprints the extension actually imported by the pytest process:
its resolved path, independently read ELF build ID, mtime, submodule-aware
tracked C/C++/CMake input watermark, Python executable, and optimization flag.
The driver also reads every device's firmware bundle from sysfs, requires the
records to be present and uniform, and compares that observed version with the
operator-supplied label. An optimized interpreter, shadow-loaded extension,
stale source/build ordering, or firmware disagreement therefore fails rather
than producing qualification evidence.

Every silicon invocation goes through `scripts/run_safe_pytest.sh`, which
serializes device access and applies the normal five-second dispatch-level
hang detection/reset. Watcher coverage sets `TT_METAL_WATCHER=1` and
`TT_METAL_WATCHER_DISABLE_ETH=1` without disabling that dispatch timeout. The
whole-forward threshold is intentionally strict `> 0`: any positive device
span improvement passes, while zero or regression fails. The independently
gated branch must still improve by at least 10%.

The serial compatibility baseline intentionally retains the pre-change
full-grid TopK and therefore records 120 configured TopK cores on these
production systems. The overlap candidate records exactly 80 TopK cores.
This measures the total production change—core restriction plus concurrency—
against the shipping serial path; it is not presented as a scheduler-only
microbenchmark. The branch manifest still reports the individual device
intervals needed to separate placement from overlap when diagnosing a result.

The separate full-model Galaxy gate uses the checked Long driver after the
device-time matrix passes:

```bash
export GLM52_HF_MODEL=/mnt/models/deepseek-prefill-cache/GLM-5.2-FP8
export TT_GLM52_PREFILL_TTNN_CACHE=/mnt/models/deepseek-prefill-cache/glm52_ttnn_cache
export PREFILL_TRACE_DIR=/mnt/models/deepseek-prefill-cache/glm-traces/vllm-glm52-indexer-kcache-55k

models/demos/deepseek_v3_d_p/tests/sparse_mla/run_glm52_sparse_mla_overlap_e2e.sh \
  "<installed-firmware-bundle>"
```

This executes the exact 78-layer, 95-K-preloaded, one-5,120-token-chunk,
ten-iteration GLM-5.2 node in serial--overlap--serial order. Iteration zero of
each process is compilation warmup; the gate compares the candidate median
from iterations 1--9 against the mean of the two bracketing serial medians.
The serial medians must agree within 1%, and the candidate win must exceed the
larger of 0.1% or twice the pooled relative sample standard deviation. Both
limits can be explicitly overridden for an investigation, and the chosen
values are recorded. The test asserts at runtime that serial has zero
overlap-enabled layers and that all 21 eligible full-indexer layers use the
observed 80-core TopK grid and 40-core gather grid in the candidate. It also
counts dispatches: every eligible layer must run the selected branch once per
iteration, for 210 serial or overlap calls per process.

The driver rejects missing model/cache/trace assets, stale source/build
ordering, dirty source, optimized Python assertions, a non-32-device mesh,
hardware skips, missing or duplicate samples, firmware disagreement, excessive
serial drift, and a candidate win that does not clear the noise-aware gate.
It archives all three raw logs, JUnit files, observed runtime configuration,
sample dispersion, ELF build ID, and `summary.json` under
`generated/profiler/qualification/galaxy_80_40_e2e/`.

The full-model number is deliberately labeled
`host_wall_clock_sync_bracketed_whole_transformer_chunk`: it measures the
production end-to-end effect, including submission overhead, and is not a
device-time overlap claim. Raw device timestamps, branch union, manager gaps,
and configured resource-split evidence remain authoritative in the preceding sparse-MLA
qualification matrix. The existing full-model harness is an 8 x 4 Galaxy
test; LoudBox supplies the production-shaped 80/40 sparse-MLA layer gate, while
final full-model production signoff is performed on Galaxy.

The correctness tests carry hardware-gated `qb2_80_30`, `loudbox_80_40`, and
`galaxy_80_40` cases. On LoudBox or Galaxy, run the matching production case;
it executes the serial/cold/warm comparison and both BF16-RM and scaled-FP8
growing-prefix comparisons on the full physical mesh:

```bash
PROFILE=loudbox_80_40  # use galaxy_80_40 on Galaxy
python -m pytest -q \
  models/demos/deepseek_v3_d_p/tests/sparse_mla/test_sparse_mla_cache.py::test_glm52_sparse_mla_overlap_matches_serial \
  models/demos/deepseek_v3_d_p/tests/sparse_mla/test_sparse_mla_cache.py::test_glm52_sparse_mla_overlap_growing_prefix_cache_and_lifetime \
  -k "${PROFILE}" -s
```

Run the op-level concurrent progress and full validation cases with Tensix
Watcher enabled. Ethernet Watcher is disabled because the instrumented fabric
router image does not fit the ACTIVE_ETH kernel-config buffer:

```bash
TT_METAL_WATCHER=1 TT_METAL_WATCHER_DISABLE_ETH=1 python -m pytest -q \
  tests/ttnn/unit_tests/operations/experimental/test_high_bw_all_gather.py::test_high_bw_all_gather_external_semaphore_independent_progress_with_topk \
  tests/ttnn/unit_tests/operations/experimental/test_high_bw_all_gather.py::test_high_bw_all_gather_external_semaphore_validation \
  -s
```

For every `{warm,cold,long} x {kv_bf16,kv_scaled_fp8}` pair, first record the
serial baseline and then gate the matching overlap run. `PROFILE` is
`loudbox_80_40` or `galaxy_80_40`; `FW_VERSION` must identify the installed
firmware bundle. The manifest path is format/profile-specific, so runs cannot
silently consume a baseline from another case:

```bash
PROFILE=loudbox_80_40  # use galaxy_80_40 on Galaxy
SCENARIO=warm
FORMAT=kv_bf16
FW_VERSION="replace-with-installed-firmware-bundle"

DS_PERF_VARIANT=glm_5_2 \
DS_PERF_OVERLAP_PROFILE="${PROFILE}" \
DS_PERF_OVERLAP_ENABLED=0 \
DS_PERF_FIRMWARE_VERSION="${FW_VERSION}" \
python -m pytest -m perf \
  models/demos/deepseek_v3_d_p/tests/sparse_mla/test_sparse_mla_perf.py::test_mla_chunked_perf \
  -k "glm_5_2 and ${SCENARIO} and sparse and ${FORMAT}" -s

BASELINE_NS="$(jq -r '.sparse_mla_overlap.measured_scenario_ns' \
  "generated/profiler/glm_5_2_sparse_${FORMAT}_${PROFILE}_serial_mla_perf/run_manifest_${SCENARIO}.json")"
BRANCH_BASELINE_NS="$(jq -r '.sparse_mla_overlap.measured_branch_serialized_ns' \
  "generated/profiler/glm_5_2_sparse_${FORMAT}_${PROFILE}_serial_mla_perf/run_manifest_${SCENARIO}.json")"

DS_PERF_VARIANT=glm_5_2 \
DS_PERF_OVERLAP_PROFILE="${PROFILE}" \
DS_PERF_OVERLAP_ENABLED=1 \
DS_PERF_OVERLAP_BASELINE_NS="${BASELINE_NS}" \
DS_PERF_OVERLAP_BRANCH_BASELINE_NS="${BRANCH_BASELINE_NS}" \
DS_PERF_OVERLAP_MIN_IMPROVEMENT=0.10 \
DS_PERF_OVERLAP_WHOLE_FORWARD_MIN_IMPROVEMENT=0.0 \
DS_PERF_FIRMWARE_VERSION="${FW_VERSION}" \
python -m pytest -m perf \
  models/demos/deepseek_v3_d_p/tests/sparse_mla/test_sparse_mla_perf.py::test_mla_chunked_perf \
  -k "glm_5_2 and ${SCENARIO} and sparse and ${FORMAT}" -s
```

Archive the six serial/overlap manifest pairs and profiler CSVs for LoudBox,
then repeat the same matrix with `PROFILE=galaxy_80_40`. Production signoff
requires both machines to pass; a skipped hardware marker or QB2 result is not
a substitute.

## Rollout sequence

1. Land the operation-level APIs, core-placement tests, and overlap proof with
   model overlap disabled.
2. Land the split indexer and serial compatibility path; prove no numerical or
   performance regression with overlap disabled.
3. Enable the sparse-MLA overlap behind explicit 120-core and 110-core profile
   flags and run the full correctness/performance matrix.
4. Qualify 80/40 on LoudBox and sign it off on Galaxy; use 80/30 for local QB2
   development only.
5. Enable 80/40 by default only for the qualified GLM-5.2 Galaxy production
   configuration, retaining 80/30 as the QB2 proxy and the serial fallback
   everywhere else.

## Risks and mitigations

- **DRAM contention:** top-k reads long logit rows while gather streams a large
  KV prefix. Core overlap does not guarantee latency overlap. Measure isolated
  and concurrent bandwidth and tune the split from end-to-end union time.
- **Fabric contention from TP index redistribution:** enqueue the KV gather
  immediately after local top-k so it begins while top-k is compute-bound; run
  the TP index redistribution only after the SP gather has joined.
- **Program-cache aliasing:** `high_bw_all_gather` already hashes its sub-device
  and core grid. Add the equivalent structural hash to `topk_large_indices` and
  test alternating 120-, 110-, 80-, 40-, and 30-core calls in one process.
- **L1 collision:** create CBs only on owned cores and keep gather mux cores out
  of every index-branch grid.
- **Premature deallocation:** keep explicit branch state until the post-manager
  join; never infer ownership from Python wrapper identity for persistent
  gather outputs.
- **Manager/trace lifecycle:** share and release one manager per mesh/profile;
  preserve the existing sparse trace rejection until trace metadata support is
  independently complete.
- **Manager-switch overhead:** entry and exit are full-grid worker-state
  drain/resets, and sparse trace capture cannot amortize them today. Gate on
  accumulated whole-forward latency across all full-indexer layers/chunks.
- **False performance win:** use device interval intersection and whole-MLA
  latency, not host dispatch time or summed per-op profiler durations.

## Deliverables checklist

- [x] `topk_large_indices` sub-device/core-grid API and tests
- [x] `high_bw_all_gather` external-semaphore interface and C++ factory path
- [x] `high_bw_all_gather` interface, cache, reuse, validation, and independent-progress tests
- [x] exact-hash warmup and cold-cache serialization coverage
- [x] QB2 isolated device-only 30-versus-40-core gather tier measurements
- [x] split indexer score/select/finalize API with serial wrapper
- [x] shared Galaxy/LoudBox 80/40 and QB2 80/30 sub-device managers with safe teardown
- [x] sparse-MLA overlap integration and serial fallback
- [x] QB2 correctness, growing-prefix lifetime, repeated semaphore, and cache-hit coverage
- [x] QB2 BF16 and scaled-FP8 growing-prefix overlap correctness
- [x] QB2 Tensix Watcher stress for independent top-k/gather progress
- [ ] Watcher/deadlock stress on LoudBox/Galaxy
- [x] concurrent lightweight-profiler record retention and raw device timestamps
- [x] device-only top-k/gather intersection, union, and serialized-branch gate
- [x] device-clock manager-boundary gaps per region and accumulated across cold chunks
- [x] independent matched-mesh whole-forward net-win gate and schema-v8 manifest plumbing
- [ ] passing cold/warm/long 80/40 net-win qualification on LoudBox and Galaxy
- [ ] end-to-end GLM-5.2 LoudBox/Galaxy benchmark
- [x] fail-closed Galaxy GLM-5.2 Long serial-versus-80/40 benchmark driver
- [x] production qualification configuration/manifest recording the selected split and tier
- [x] Claude Opus plan reviews incorporated, with disposition recorded below

### Implemented QB2 integration evidence

The focused GLM-5.2 eager test
`test_glm52_sparse_mla_overlap_matches_serial` runs on the local four-chip
QB2 as a `2 x 2` SP x TP mesh. Every chip still uses its native `11 x 10`
worker grid and the exact 80/30 sub-device profile. It executes one serial
reference, one cold split-grid overlap pass, and one warm repeated overlap pass
with fresh KVPE and compact 21-slot GLM-5.2 index caches. Both overlap passes
match the serial output at PCC 1.0. The repeated pass exercises the cached
80-core top-k and 30-core gather hashes and reuse of the same caller-owned
semaphore pair. Host contract tests additionally assert the closed program
allowlist, join-before-TP-redistribution order, exception recovery, and exact
`high_bw_all_gather` resource arguments.

This is functional QB2 qualification only. 80/40 performance, production-long
prefixes, full-model Watcher stress, production-scale scaled-FP8 performance,
and a passing whole-forward net-win gate remain open for LoudBox/Galaxy Phase 6
runs.

The follow-up integration test
`test_glm52_sparse_mla_overlap_growing_prefix_cache_and_lifetime` runs two
256-token chunks against one 512-token cache in both BF16-RM and scaled-FP8
KVPE formats. It deliberately retains both device outputs until the second
overlap region and its full-grid consumers complete, asserts distinct output
allocations, then reads both back. Serial, cold-overlap, and
warm-cache-overlap outputs match at PCC 1.0 for both the 256-token and
512-token populated prefixes in each format. The second overlap pass adds no
program-cache entries. A separate scaled-FP8 qualification-harness smoke run
also completed the exact 80/30 route and emitted the format-specific manifest;
it is execution evidence, not a production performance result.

Those same two integration tests now collect hardware-gated FABRIC_2D cases
for the full LoudBox `2 x 4` and Galaxy `8 x 4` meshes. Each production case
asserts 80 top-k-owned and 40 gather-owned cores and executes the same PCC and
cache/lifetime checks. On QB2 they are explicitly skipped because only four
devices are available; they are the correctness entry points in the production
runbook above, not locally claimed evidence.

The standalone operation test now proves independent device progress without
depending on profiler delivery: after enqueueing a production-proxy 80-core
top-k followed by a 30-core gather, it synchronizes a host-visible event for
only the gather sub-device. On the local QB2, the gather event completed in
0.314--0.315 ms while the fastest isolated top-k event took 2.331 ms. A
serialized dispatcher could not complete the later gather-scoped event before
the earlier top-k. The same run validates exact top-k and gather outputs; the
adjacent tests cover 100-call semaphore reuse, cache-hit semaphore-address
rebinding, and all interface validation failures.

The completed Phase 2 op suite now also runs caller-owned resources through
all eight combinations of the qualified line/ring schedules and BF16 row,
scaled-FP8 row, BF16 tile, and BF8 tile payloads, with exact decoded output
comparison. The first external-path cache miss leaves both L1 and L1-small
allocator totals unchanged. A separate 100-call test alternates selected batch
and growing/shrinking prefix controls, checks every produced slice, and keeps a
single cache entry. The pair-rebind test covers one miss plus 99 hits, poisons
the old pair to prove current handle addresses are rebound, and resets/reuses
the pair to model exception recovery. Validation covers paired/distinct
handles, L1-small type, worker coverage, loaded sub-device membership, grid
containment, undersized grids, and the implicit default full grid on QB2; the
wrong-parent-mesh case runs when the line is a natural LoudBox/Galaxy submesh.
The Long-shape device qualification calculates and records the selected
workers/direction tier and conservative scheduler budget for both the 30- and
40-core ownership strips, while validation rejects grids below the legal
minimum. Together with the source-level external factory branch, which
contains neither semaphore allocation nor `Synchronize`, the cache-miss
allocator assertion and gather-scoped event are the direct
no-hidden-allocation/no-other-subdevice-drain evidence.

The consolidated local regression on 2026-08-11 passed 12 focused
`high_bw_all_gather` device cases, five restricted-grid `topk_large_indices`
device cases, 17 sparse-MLA host contract cases, and all three QB2 model
integration cases. The model cases are the serial/cold/warm test plus the
BF16-RM and scaled-FP8 growing-prefix tests; every reported comparison was PCC
1.0. This snapshot qualifies the local 80/30 path only and does not close any
80/40 production gate.

The same independent-progress test passes with Tensix Watcher enabled via
`TT_METAL_WATCHER=1 TT_METAL_WATCHER_DISABLE_ETH=1`: the gather-scoped event
completed in 0.595 ms while the isolated 80-core top-k lower bound was 2.445
ms, and Watcher reported no Tensix fault. Ethernet Watcher must be disabled on
the current local QB2 firmware bundle (19.10.0), because enabling Watcher on
the fabric routers makes the ACTIVE_ETH image 29,680 bytes, exceeding its
25,600-byte kernel-config buffer during device setup. This is not an overlap
failure: no operation is dispatched in that configuration. A full
growing-prefix model run under Tensix Watcher later aborts for an undiagnosed
reason in the serial reference `topk_large_indices` call without a Watcher
fault report, before the overlap region. That run gives zero full-model
coverage of the concurrent path. Therefore the op-level concurrent path has
local Watcher coverage, while full-model Watcher/deadlock qualification
remains an explicit LoudBox/Galaxy gate and must both diagnose the serial abort
and use a firmware/build whose instrumented fabric image fits.

`test_sparse_mla_perf.py` now accepts `DS_PERF_OVERLAP_PROFILE` values
`qb2_80_30`, `loudbox_80_40`, and `galaxy_80_40`. The named profile selects the
matched SP x TP mesh, restricts qualification to GLM-5.2 sparse MLA, warms the
exact hashes, and writes a schema-v8 manifest containing the worker grid,
80/30 or 80/40 ownership, KVPE page size, worst-case bytes/link, preferred and
selected workers/direction, conservative gather scheduler budget, per-chip device spans,
top-k/gather durations, intersection and overlap ratio, union versus serialized
sum, device-clock manager-boundary gaps, timing samples, baselines, and both
net-win decisions. `DS_PERF_OVERLAP_ENABLED=0` measures apples-to-apples serial
whole-forward and branch-sum baselines on that same mesh.
`DS_PERF_OVERLAP_BRANCH_BASELINE_NS` enables the default 10% branch-union gate;
`DS_PERF_OVERLAP_BASELINE_NS` separately requires a positive whole-forward win
(or the explicitly configured
`DS_PERF_OVERLAP_WHOLE_FORWARD_MIN_IMPROVEMENT`). The latter no longer
incorrectly applies the branch's 10% threshold to the whole forward.

Manager transition accounting is device-only. For every chip, the manifest
records the raw-device-clock idle gap from the last completed default-manager
program to the earliest branch start, and from the joined branch end to the
first restored-default-manager program. These are conservatively named
manager-boundary gaps: they include dispatch/reset idle visible to the device
and do not claim to isolate Python API time. The older
`test_sub_device_load_clear_timing.py` Tracy/host-bracketed measurement remains
a diagnostic pattern but is not accepted as production latency evidence.

The local BF16 smoke run resolved a 1152-byte KVPE page and selected two
workers/direction (12-core scheduler budget) inside the 30-core ownership strip.
Earlier completion-event measurements are retained only as historical
diagnostics and are not accepted as device-time performance evidence. The
fixed realtime transport now emits both programs on every QB2 chip: top-k is
about 2.246 ms, gather is about 23.7--29.0 us, and the gather interval lies
almost entirely inside top-k. This is the standalone op-test proxy tensor; it
is not the full 128,000-token model gather measured below and its gather time
must not be used to explain the end-to-end speedup.

The exact QB2 Long-shape isolated tier qualification now measures both 30- and
40-core right-edge gather ownership with raw device timestamps. It uses a
129,280-row global prefix (128,000 cached plus the 1,280-token chunk), two SP
ranks, two Fabric links, and seven warmed samples per tier. Both ownership
sizes select two workers/direction, reserve 12 cores for topology-independent
scheduling, and instantiate 10 cores on each endpoint of the two-rank line.

| Cache format | 30-core median | 40-core median | 40 vs 30 | 30-core receive BW | 40-core receive BW |
| --- | ---: | ---: | ---: | ---: | ---: |
| BF16 row-major | 2.588 ms | 2.588 ms | +0.01% | 28.77 GB/s | 28.77 GB/s |
| scaled FP8 row-major | 2.523 ms | 2.522 ms | +0.06% | 18.04 GB/s | 18.05 GB/s |

This closes the local 30-versus-40 tuning question for the current runtime
schedule: merely owning 10 more idle cores provides no material gather gain.
It does not replace the required LoudBox/Galaxy 80/40 end-to-end qualification.

The schema-v6 QB2 Long qualification used seven samples per mode at the
128,000-token box-local prefix (the Galaxy 512,000-token per-chip proxy). It
supersedes the earlier schema-v5 interpretation, which accidentally selected
the short TP index redistribution as the serial gather by choosing the gather
nearest top-k. Schema v6 selects the longest `high_bw_all_gather`, which is the
production sparse-KV prefix gather, and separates the branch and whole-forward
gates. Medians and device-only results were:

| Cache format | Serial forward | 80/30 forward | Forward win | Serial branch sum | Overlap union | Branch win | Manager-boundary gap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BF16 row-major | 17.018 ms | 15.274 ms | 10.24% | 4.789 ms | 2.890 ms | 39.66% | 60.6 us |
| scaled FP8 row-major | 15.517 ms | 13.501 ms | 12.99% | 4.696 ms | 2.889 ms | 38.47% | 60.6 us |

Both formats pass the corrected local gates: their branch unions are more than
10% shorter than the warmed serialized branch sum, and their whole forwards
improve. In every overlap sample and on every chip, the selected KV gather was
fully contained in top-k. Serial branch unions were 4.773--4.790 ms for BF16
and 4.686--4.706 ms for scaled FP8; overlap unions were 2.890--2.934 ms for
BF16 and tightly clustered at 2.889--2.890 ms for scaled FP8. Whole-forward
samples retain the cold first-sample
outlier rather than deleting it: BF16 serial/overlap ranges were
16.886--28.685/14.946--27.083 ms, and scaled FP8 ranges were
15.227--27.601/13.204--25.767 ms.

The manager-boundary figure is the median slowest-chip device-clock gap across
the two transitions, not a host timer. BF16 samples were 60.2--103.0 us;
scaled FP8 samples were 60.4--80.6 us. The raw per-sample list remains in the
manifest so this variability is not hidden. These final manifests were
captured from clean published commit `95d1b93f3a3`; their embedded commit and
branch fields therefore reproduce the implementation under test. These QB2
results qualify the local 80/30 path only. They do not close the
LoudBox/Galaxy 80/40 gates or replace isolated 30-versus-40 gather-tier
measurements.

## Claude Opus review disposition

Reviewed with Claude Code using
`claude --dangerously-skip-permissions --model opus --effort high --print ...`.
Verdict: **APPROVE WITH CHANGES**.

After adding the concrete `high_bw_all_gather` interface, Phase 2 received a
second focused review using
`claude --dangerously-skip-permissions --model opus --effort medium --print ...`.
Its verdict was also **APPROVE WITH CHANGES**.

The concurrent-profiler/device-span update received a third bounded review
using `claude --dangerously-skip-permissions --model opus --effort low
--print ...`. Its verdict was **APPROVE WITH CHANGES**. The required changes
are incorporated above: the op proxy and Long gather are distinguished,
hidden time is reconciled with the net win, seven-sample dispersion and the
provisional FP8 gate are explicit, 16-bit ID wrap and 128-record queue behavior
are specified, missing records and per-chip calibration are covered, and the
full-model Watcher abort is labeled undiagnosed with zero concurrent coverage.
A final bounded re-review with the same Opus command/model returned
**APPROVE**, with no remaining concerns in the corrected profiler and
device-timing sections.

The review's required changes are incorporated in this revision:

- only two single-sub-device-confined programs remain inside the overlap
  manager; index layout conversion and the TP high-bandwidth gather run after
  the full-grid join;
- exact sub-device/core-grid hashes are warmed before overlap measurement;
- manager clear/reset is documented as the existing full-grid device-side
  drain/join, and its accumulated per-layer/chunk cost is included in the
  production gate;
- the manager uses `local_l1_size=0` and has a concrete TT_CCL/transformer
  teardown path;
- top-k work-split traversal, loaded-manager validation, scratch-key
  disjointness, conditional gather tiers, SP=1/unexpected-grid rejection, and
  overlapped growing runtime-argument cases have explicit tests.

The focused operation review's changes are also incorporated:

- external semaphore coverage contains resolved workers and is allocated over
  the whole gather strip; muxes do not require semaphore coverage;
- cache-hit addresses come from current operation attributes, with handle
  copies retained only for lifetime;
- validation responsibilities are split between device-op cache validation and
  factory miss/hit checks using the actual `GlobalSemaphore` accessors;
- mesh-created, mesh-uniform L1-small/L1 handles receive a one-time mesh-wide
  initialization barrier and an explicit failure-reset rule;
- no-sync/no-allocation, pair rebinding, repeated reuse, confinement, default-
  full-grid, and scheduler-tier tests use implementable observability;
- KVPE runtime-control scheduling is gated on real worst-case bytes/link and
  page size instead of assuming a 36/20 scheduler budget from ownership alone.

The implemented Phase 2 diff received a third Claude Opus review using the
same required command/model. Its **APPROVE WITH CHANGES** findings were applied
before the phase commit: the cache-rebind test poisons the old pair, repeated
reuse runs 100 adjacent gathers, registered sub-device managers are removed,
resolved worker grids participate in the program hash, invalid active-manager
sub-device IDs receive a direct diagnostic, and cache-hit coverage checks reuse
the factory's stored worker grid. The remaining independent-progress,
runtime-control recovery, confinement, and scheduler/performance cases stay
open in the checklist because they require the top-k and sparse-MLA integration
phases rather than only the op interface.

The implemented `topk_large_indices` phase received a fourth Claude Opus
review using `claude --dangerously-skip-permissions --model opus --print ...`.
Its verdict was **APPROVE WITH CHANGES**. The requested changes were applied:

- the 163-row discontiguous-grid device test now gives every row a distinct
  top-k index set, so row duplication, omission, or permutation cannot pass;
- production runtime-argument programming uses one canonical
  `(core, start_row, num_rows)` derivation, while the device test directly
  checks row-wise traversal across a discontiguous 80-core set;
- cache hits assert that the resolved grid exactly matches the compiled grid,
  while the cache key uses that resolved grid instead of duplicating entries
  for implicit versus explicit selection of the same subdevice;
- tests cover explicit-grid selection without a subdevice ID, out-of-device
  grids under the default manager, 110/80/30-core cache separation, and
  implicit/explicit full-grid cache equivalence.

Exact logical core coordinates are not exposed by the lightweight realtime
callback (timestamps and kernel sources) or `ProgramAnalysisData` (core count
only). Op-level confinement is therefore pinned structurally by creating all
kernels/CBs on the resolved `CoreRangeSet`, the shared scheduling test, and
restricted-grid device correctness. Exact-coordinate profiler/Tracy evidence
remains an explicit gate in the combined overlap and end-to-end phases, where
both branches and subdevice ownership can be observed together. Two narrower
follow-up Opus invocations were attempted after incorporating the findings but
the Claude CLI produced no output and timed out; no different model was used.
After the sparse-MLA integration commit, one full-branch and one integration-
only review were also invoked with
`claude --dangerously-skip-permissions --model opus ...`. Both remained alive
without producing output and were terminated after bounded waits, so they do
not constitute additional verdicts. The four successful Opus reviews above
remain the recorded plan/operation dispositions; this document does not claim
that the post-integration retries approved the branch.

After adding the independent-progress, growing-prefix, and production-manifest
phases, a further review was invoked with
`claude --dangerously-skip-permissions --model opus --effort medium --print ...`.
It likewise produced no output before the bounded 180-second timeout. No
different Claude model was substituted, and this attempt is not represented as
an approval.

After completing the expanded Phase 2 operation matrix, the same Opus-only
command was invoked once more against the current branch and plan. It produced
no output during a bounded 180-second wait and was terminated, so this attempt
also adds no verdict. The four earlier successful Claude Opus reviews and their
incorporated dispositions remain the review basis for this plan.

The schema-v6 device-time qualification update received a final focused review
using `claude --dangerously-skip-permissions --model opus --effort low --print
...`. The first verdict was **CHANGES REQUIRED** because the critical
serialized branch sum added the maximum top-k and gather durations from
potentially different chips. That would have made the branch baseline
artificially conservative and the 10% gate easier. The implementation now
takes the maximum of each real chip's top-k-plus-gather sum and reports the
minimum real per-chip union reduction ratio. Both Long pairs were rerun after
that correction. A second Opus review returned **APPROVE**, explicitly
confirming the device-only timing source, per-chip interval math, conservative
collapse, independent gates, schema-v6 runbook fields, and updated result
arithmetic.

The production qualification driver received a further Opus-only review cycle
using `claude --dangerously-skip-permissions --model opus --effort low --print
...`. The substantive reviews initially returned **CHANGES REQUIRED** for
fail-open provenance and workload handling: ambient workload overrides, stale
builds/manifests, loose skip/sample checks, unchecked baseline propagation,
and incomplete evidence archival. The driver now pins the production workload,
requires a clean exact commit and newer matching build, tags each manifest with
a unique run ID, checks exact JUnit/sample counts and serial baselines, validates
the runtime-derived gather tier and configured ownership, runs all
silicon commands through the safe runner, and archives mandatory Watcher and
per-call/cold evidence with explicit pass/fail status. The final review returned
**APPROVE** after verifying the worker/mux placement formula directly against
the gather factory and confirming the Watcher, timestamp, and gate plumbing.

The Galaxy full-model Long driver and runtime branch-accounting update received
three high-effort Opus reviews using
`claude --dangerously-skip-permissions --model opus --effort high --print ...`.
The first verdict was **CHANGES REQUIRED** for a fixed-order two-process
comparison, assertion stripping under `PYTHONOPTIMIZE`, hard-coded evidence,
and construction-only profile proof. The driver now brackets the candidate
with two stable serial medians, applies a dispersion-derived win threshold,
reads firmware/build/runtime metadata, and counts every eligible layer's
actual serial or overlap dispatch. The second review verified those fixes but
incorrectly proposed dropping the `/8x4` cache suffix; a focused follow-up
traced `weight_cache_path` through `run_chunked_transformer_updated`, confirmed
that `/8x4` is the exact effective cache, and returned **APPROVE** after the
driver also gained a non-empty flat-tensorbin preflight. No different Claude
model was used.

A resumed high-effort Opus audit of production provenance initially returned
**CHANGES REQUIRED**. It found that deleting firmware-record newlines joined
all device versions into one value, that build watermarks omitted submodules,
and that manifest build fields merely echoed shell-provided values rather than
fingerprinting the imported extension. Both production drivers now read every
firmware record in the parent shell, recurse through tracked submodules, use a
ceiling-second source watermark, and reject optimized Python. The sparse-MLA
manifest contract is schema v8 and independently records the loaded
`ttnn._ttnn` path/build ID, source watermark, interpreter, and optimization
state. A supported QB2 80/30 safe-runner smoke emitted and validated those
fields. The second review empirically exercised the validator against that
manifest and returned **APPROVE**; no different Claude model was used.
