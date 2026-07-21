# Regime-A Fused All-Gather Matmul: Design and Execution Plan

## Purpose

Build a new TTNN operation that fuses an all-gather of `in0` on its last dimension with
`ttnn.experimental.regime_a_matmul` on Blackhole Galaxy.

Each device initially owns a contiguous shard of `in0[M, K_global]` along K and has access to the full
Regime-A weight `in1[K_global, N]` in the existing eight-bank DRAM-sharded layout. Each device must produce
the same matmul result it would produce after gathering the complete `in0`:

```text
out[M, N] = all_gather(in0_shard, dim=-1) @ in1
```

The performance objective is not merely to make the two operations share an API. The fused operation must
stream local and remote K chunks through the matmul so that all-gather time is hidden behind in1 DRAM reads
and matmul wherever the roofline permits it. It must not wait for the complete all-gather before starting
matmul.

The ideal target is:

```text
T_fused ~= max(T_regime_a, T_fabric, T_compute_and_reduction, T_local_fanout)
```

rather than `T_all_gather + T_regime_a`. Report both the absolute result and the distance from this ideal;
do not describe a result as successful solely because it beats an unfused implementation.

Working name: `ttnn.experimental.all_gather_regime_a_matmul_async`. The final public name may be changed
during API review. This must be a new operation. Do not add multi-device conditionals to the single-chip
`regime_a_matmul` kernels or alter its public behavior.

## Working protocol

The work is divided into numbered tasks below. **Complete exactly one task at a time. After every task,
commit the work, give the required status report, and stop for review. Do not begin the next task until the
orchestrator explicitly approves it.** A negative experimental result is a valid task outcome; preserve the
evidence and do not silently replace the requested experiment with a different design.

Every status report must contain:

1. Commit hash and changed files.
2. What was implemented or measured, including any deviations from this document.
3. Exact build/test commands and hardware topology used.
4. Correctness results, including PCC, fresh-program and cached-program behavior, and watcher status where
   applicable.
5. Performance table with all individual relaunches, median, spread, and the relevant A/B baseline.
6. Raw-data and human-readable report paths.
7. Unexpected findings, unresolved risks, and the decision requested from the orchestrator.

Use resumable, checkpoint-per-measurement harnesses. Classify timeout, hang, validation, runtime, PCC, and
success separately. Never overwrite a finalized result artifact with a resumable cache.

## Read these references first

### Regime-A matmul: the compute engine being preserved

Read these in order:

1. `ttnn/cpp/ttnn/operations/experimental/regime_a_matmul/README.md`
2. `ttnn/cpp/ttnn/operations/experimental/regime_a_matmul/device/regime_a_matmul_plan.hpp`
3. `ttnn/cpp/ttnn/operations/experimental/regime_a_matmul/device/regime_a_matmul_config.cpp`
4. `ttnn/cpp/ttnn/operations/experimental/regime_a_matmul/device/regime_a_matmul_program_factory.cpp`
5. The three kernels under `ttnn/cpp/ttnn/operations/experimental/regime_a_matmul/device/kernels/`
6. `tests/ttnn/unit_tests/operations/matmul/test_regime_a_matmul.py`
7. `tests/ttnn/unit_tests/operations/matmul/test_regime_a_matmul_corpus.py`
8. `tools/mm_sweep/REGIME_A_CURRENT_PERF_REPORT.md`
9. `tools/mm_sweep/REGIME_A_FUSION_REPORT.md`

Important invariants to retain:

- `in1` is width-sharded across eight DRAM banks. The bank-adjacent in1 readers and their NoC assignments
  are foundational to bandwidth and must not be casually moved or burdened with fabric work.
- `in0` is currently seeded across eight bank cores and delivered by a topology-optimized local ring.
  Compute waits progressively and starts as soon as the next K block is available; there is no full local
  all-gather barrier.
- `Pk`, `Ns`, and `Sm` partition K, N, and M. `Pk > 1` ends in an on-device reduction. `Ns` consumers reuse
  identical A data; `Sm` consumers require different M rows.
- Balanced tails, PARETO local-ring ordering, `IN1_NEAR` M-split placement, source-lifetime-safe NoC
  ordering, split-K pipelined draining, Picker v3, and the single-chip fusions are accepted behavior.
- Bias, activation, addcmul, and output chunking are applied once at the split-K reduction root. The
  multi-device work must eventually compose with these without moving the epilogue into every K slice.

### Existing fused all-gather minimal matmul: the fabric reference

Read these in order:

1. `ttnn/cpp/ttnn/operations/experimental/ccl/all_gather_minimal_matmul_async/`
2. `device/all_gather_minimal_matmul_async_program_factory.cpp`
3. `device/kernels/dm_in0_sender.cpp`
4. `device/kernels/matmul_dataflow_common.hpp`
5. `device/kernels/dm_in1_sender_out.cpp`
6. `device/kernels/compute.cpp`
7. `models/tt_dit/tests/models/wan2_2/test_all_gather_minimal_matmul_async.py`
8. `tests/nightly/tg/ccl/test_all_gather_minimal_matmul_addcmul_async.py`

Also inspect `strided_all_gather_minimal_matmul_async`; it is useful evidence for mapping nontrivial data
ownership onto fabric workers, but it is not a replacement for understanding the non-strided op.

The existing AGMM establishes these useful patterns:

- Process the local device's K blocks first, then consume remote K blocks as they arrive.
- Publish a block to compute before forwarding it, so compute can overlap the forward.
- Forward A over fabric only during its first N traversal, then reuse it for later N blocks.
- Use global semaphores, persistent buffers, fabric mux channels, per-link worker assignment, bidirectional
  ring traffic, and explicit mux connection/termination protocols.
- The program factory is the authoritative example for constructing `FabricMuxConfig`, reserving mux and
  worker cores, generating compile/runtime connection arguments, and configuring packet routes.

Do not copy its block policy blindly. It was optimized for large square matmuls with much larger payloads,
a rectangular compute grid, and a different bottleneck.

### Fabric and mux references

For a minimal example isolated from matmul, read:

- `tests/tt_metal/tt_fabric/fabric_data_movement/test_basic_fabric_mux.cpp`
- `tests/tt_metal/tt_fabric/fabric_data_movement/kernels/fabric_mux_sender_client.cpp`
- `tests/tt_metal/tt_fabric/fabric_data_movement/kernels/fabric_mux_receiver_client.cpp`
- `tt_metal/fabric/hw/inc/tt_fabric_mux_interface.hpp`
- `tt_metal/fabric/hw/inc/edm_fabric/fabric_connection_manager.hpp`
- `tt_metal/fabric/hw/inc/linear/api.h`

Use the APIs and ordering demonstrated by current repository code. Do not invent a private fabric protocol
when an existing connection manager, mux channel, packet header, flow-control, or teardown primitive already
exists.

## Blackhole Galaxy bring-up and reference commands

Run from the repository root in the machine's normal TT-Metal Python environment. Record the exact commit,
firmware bundle, board identity, visible device count, mesh descriptor, and command line in every report.

Build with tests:

```bash
./build_metal.sh --release --enable-ccache --build-tests
```

For a single 32-chip Blackhole Galaxy, use the repository's 4x8 torus descriptor unless the machine owner
provides a machine-specific descriptor:

```bash
export TT_MESH_GRAPH_DESC_PATH=tt_metal/fabric/mesh_graph_descriptors/single_bh_galaxy_torus_xy_graph_descriptor.textproto
```

Before developing the new op, run these references successfully:

```bash
# Current single-chip Regime-A correctness and cache replay.
pytest -svv tests/ttnn/unit_tests/operations/matmul/test_regime_a_matmul.py --timeout=600

# Existing Blackhole 4x8 fused AGMM, one small correctness case. Expand only after this works.
pytest -svv \
  models/tt_dit/tests/models/wan2_2/test_all_gather_minimal_matmul_async.py::test_linear \
  -k 'bh4x8links2 and unit and fused and check' --timeout=600

# Existing fused epilogue/fabric composition.
pytest -svv tests/nightly/tg/ccl/test_all_gather_minimal_matmul_addcmul_async.py --timeout=600

# Fabric mux unit tests, if this binary is present in the selected build configuration.
./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter='Fabric1DMuxFixture.*'
```

If the machine uses a launcher, rank-binding file, container, or nondefault descriptor, use the local Galaxy
runbook rather than forcing these commands. `tools/scaleout/exabox/run_fabric_tests.sh --help` documents the
standard single-Galaxy `4x8` descriptor and fabric validation flow. Report environmental changes; do not
hide them inside a shell profile.

For device profiling, follow `tools/mm_sweep/regime_a_profile.py`: set `TT_METAL_DEVICE_PROFILER=1`, close
or explicitly flush the device, and retain per-RISC profiler artifacts. A host wall time without kernel spans
is insufficient for overlap claims.

## Proposed algorithm

### Scheduling invariant

The following invariant applies to every phase:

```text
for transport chunks in local-first / distance-aware order:
    obtain local or remote A chunk
    expose its first kb subblock to the local Regime-A delivery path
    begin the matching B[global_k, :] reads
    compute each kb subblock as soon as both A and B are ready
    forward the A chunk to the next device as soon as source lifetime permits
```

There must be no full-device all-gather wait before the first matmul and no wait for an entire transport
chunk if its first compute subblock can safely be exposed.

### Separate transport granularity from compute granularity

Regime-A commonly uses small `kb` values. A compute payload is approximately:

```text
M_tiles * kb * 2048 bytes
```

which can be only 4--32 KiB. Fabric packet, mux, and semaphore costs may dominate if every `kb` becomes an
independent transport operation. Define a transport chunk as `C * kb` blocks. Transfer the coarser chunk but
publish its `kb` subblocks progressively. `C` is an experimental/planner parameter, not a new public tuning
knob until evidence supports it.

For bidirectional rings, initially stripe whole contiguous transport chunks clockwise and counter-clockwise.
Do not split every small compute block into two fragments merely because the square AGMM does so.

### Preserve the local Regime-A path

A remote payload must cross fabric only once per destination device. It must then be distributed on-chip to
the consumers that need it:

- `Ns` groups need identical A and should not cause duplicate fabric traffic.
- `Sm` groups own different M rows and can receive different M slices.
- The existing eight-core local in0 rings should remain the final progressive delivery mechanism unless a
  measured experiment proves a better replacement.

The likely direct-streaming path is therefore:

```text
fabric ingress -> partition A chunk into eight K stripes -> seed local in0 ring slots
               -> local Ns fanout -> progressive compute
```

The precise order of fanout and eight-way seeding is an experiment. Avoid a single ingress worker becoming
a serial bottleneck; use multiple fabric workers or fabric/local-NoC scatter where appropriate.

### Read local A once

The first local DRAM read should feed both local compute and fabric forwarding. A dedicated fabric egress
worker may improve isolation, but it must receive the already-read L1 payload over local NoC rather than
reread A from DRAM. Compare direct ring-core injection against ring-core-to-egress handoff.

Do not put fabric duties on the critical in1 reader RISC. If the in0/reduction/output RISC gains fabric work,
profile whether that contention becomes exposed.

### Partition local K work across `Pk`

The single-chip planner gives each `Pk` band a contiguous global K interval. With device-contiguous K
sharding, that can leave one band rich in local blocks and other bands stalled on fabric.

The multi-device planner should be able to assign explicit global K block IDs or multiple K segments to a
`Pk` band. Experiment with cyclic or balanced assignment so every active `Pk` band receives useful local work
during fabric startup. The in1 reader must read the B rows corresponding to the same global K block order.

This changes scheduling, not mathematical ownership: every global K block is consumed exactly once, and the
existing local split-K reduction still combines the `Pk` partials. Different accumulation order may cease to
be bit-identical; correctness is judged against the BF16/FP32 golden with the established PCC threshold.

### Direct-L1 flow control

Remote L1 capacity is bounded. A direct-streaming implementation requires explicit credits:

- Each ingress owns a bounded number of transport slots.
- A sender may reuse/advance a slot only after the receiver has consumed or safely copied it and any onward
  fabric forwarding no longer reads the source.
- Payload must be ordered before its readiness signal on the same NoC/fabric route.
- Readiness does not prove source-buffer lifetime. Retain the required write flush/barrier before popping or
  overwriting a source slot.
- Drain both payload writes and non-posted semaphore atomics before kernel termination.

Test with more transport chunks than slots so wraparound and backpressure are mandatory. A large L1 depth
that merely makes the race unlikely is not a correctness mechanism.

### Roofline and picker implications

The raw byte ratio is approximately:

```text
remote_A_bytes / in1_DRAM_bytes ~= (M / N) * (D - 1) / D
```

but fabric bandwidth is lower than local DRAM bandwidth. Model at least:

```text
T_fabric / T_in1 ~= (M / N) * (D - 1) / D * (BW_DRAM / BW_fabric_effective)
```

The eventual picker must account for device count, topology, links, workers per link, transport chunk size,
reserved mux/ingress cores, local fanout, and the amount of local K work available before the first remote
stall. Do not modify Picker v3 during early bring-up; use pinned configurations and add a fabric-aware picker
only after the mechanisms have been measured.

### Materialized output policy

Writing the gathered A tensor to DRAM is an intentional first phase and useful correctness/reference path.
It is not assumed to be the final fast path. The production fused path should not materialize gathered A
unless the caller explicitly requests it. If the eventual API returns the gathered tensor or accepts a
persistent output buffer, measure that mode separately because its DRAM writes can change the optimum.

## Numbered execution tasks

### Task 1 — Reproduce both parent operations and freeze baselines

Do not write fusion code yet.

1. Build and run the reference commands above on the target Blackhole Galaxy.
2. Run the current Regime-A production path on the representative shape set below with pinned configs and
   `config=None`; retain host and per-RISC timings.
3. Measure standalone all-gather for the same per-device A shards and the unfused sequence
   `all_gather_async -> regime_a_matmul` using resident inputs and trace capture where supported.
4. Record fabric topology, `D`, link count, workers per link, mux buffers, packet size, and effective fabric
   bandwidth.
5. Produce a baseline table containing `T_mm`, `T_ag`, `T_unfused`, `max(T_mm,T_ag)`, PCC, and all relaunches.

Gate: both parent ops correct, cached replay correct, profiler usable, and at least one D=2 or D=4 and one
full 4x8 reference measurement are reproducible within 5% spread.

**Stop and report Task 1.**

### Task 2 — Scaffold the independent fused op and pure host plan

1. Add a new experimental TTNN op with its own folder, device operation, factory, kernels, nanobind, build
   wiring, and tests. Match Regime-A API style and the existing AGMM fabric arguments.
2. Keep multi-device geometry and transport scheduling in a testable host plan. Represent global K block
   identity explicitly; do not bake contiguous ownership into kernel arithmetic.
3. Reserve fabric mux/worker cores before bank-adjacent reader, M-split slave, and local-ring placement.
4. Initially support BF16, no transpose/batching, tile-aligned K sharding, and no epilogues. Validate all
   constraints clearly.
5. Implement a D=1/no-fabric path that delegates to or is behaviorally identical to production
   `regime_a_matmul`.

Gate: offline plan tests cover D=1/2/4/8, topology, links, K ownership, core collisions, and L1 sizing; D=1
matches the parent op's correctness and has no material performance regression.

**Stop and report Task 2.**

### Task 3 — Phase A: DRAM-staged streaming correctness

Implement the simplest fused dataflow first:

1. The injector reads its local A transport chunk once, makes it available to local compute, and unicasts it
   to the next device.
2. Each receiver writes remote chunks into their global offsets in a persistent DRAM gather buffer, signals
   per-chunk readiness, and forwards them onward.
3. Regime-A workers consume local chunks immediately. For remote chunks, they wait only for that chunk's
   readiness, read it from the DRAM gather buffer, and continue progressively.
4. Start with one direction and D=2. Then add linear and bidirectional ring behavior using existing AGMM
   fabric/mux infrastructure.
5. Include a compile-gated full-gather-before-matmul diagnostic for same-binary overlap A/B. Do not expose it
   publicly.

Gate: random-input PCC >= 0.999 for D=2/4/8, local and remote K blocks, more chunks than buffering slots,
fresh and cached program, and both supported topologies. Device watcher must be clean. Demonstrate with a
timeline or per-RISC evidence that the first matmul begins before the full gather completes.

**Stop and report Task 3.**

### Task 4 — Measure and characterize the DRAM-staged phase

1. Compare staged streaming, staged full-wait diagnostic, unfused AG+MM, standalone MM, and standalone AG.
2. Measure one and two links and sensible worker/mux settings; do not launch an exhaustive search yet.
3. Report first-compute latency, first-remote stall, fabric span, in1 BRISC span, compute span, local fanout
   cost, delivered/effective DRAM BW, and output-write cost.
4. Determine whether DRAM staging is hidden, adds a second critical DRAM read, or perturbs in1 bank BW.
5. Use the evidence to specify the direct-L1 slot size, depth, and expected maximum speedup.

Gate: the implementation is a useful performance reference and the report identifies a measurable reason
to proceed to direct L1. Do not claim direct L1 will win without a ceiling estimate.

**Stop and report Task 4.**

### Task 5 — Phase B: direct remote-L1 streaming with credits

1. Add a direct-L1 path behind an internal diagnostic/config selection; retain staged DRAM for A/B.
2. Receive transport chunks into bounded L1 slots, expose `kb` subblocks progressively, and return credits
   only after local consumption/copy and onward forwarding are source-safe.
3. Preserve compute-before-forward overlap where safe: publish ready data locally, issue the forward, and
   avoid a whole-chunk prewait.
4. Implement slot depths 1/2/4 initially. Test forced wraparound with many more chunks than slots.
5. Validate all payload/readiness/flush/atomic-drain ordering under watcher and repeated stress.

Gate: PCC/cache/watcher/stress clean on D=2 and D=4 before D=8; direct L1 is measured against the exact
same-config staged path. No default selection yet.

**Stop and report Task 5.**

### Task 6 — Select staged DRAM versus direct L1

Run an interleaved, multi-relaunch A/B over the representative corpus and D=2/4/8. Include slot-depth and
credit-stall statistics. Adopt direct L1 only if it has a stable benefit on the intended corpus without a
correctness, hang, watcher, or material control-shape regression. A hybrid policy is acceptable only if a
single explainable predictor, such as required L1 capacity or transport-to-in1 time ratio, separates the
classes.

If direct L1 fails, keep staged DRAM as the functional path, preserve evidence, and ask whether the remaining
performance is sufficient before proceeding.

**Stop and report Task 6.**

### Task 7 — Transport granularity, direction balance, and local-first `Pk` scheduling

Measure these independently before combining them:

1. `C=1/2/4/8` compute blocks per fabric transport chunk, with progressive subblock publication.
2. Whole-chunk bidirectional striping versus the accepted baseline. Do not fragment tiny blocks by default.
3. Current contiguous `Pk` ownership versus cyclic/balanced local-K distribution across `Pk` bands.
4. Local-first versus nearest-remote-distance scheduling after the local work is exhausted.
5. One- versus two-chunk in1 lookahead after remote readiness becomes predictable.

For each lever, hold the other decisions fixed and report payload sizes, packets, semaphores, credits,
fabric utilization, first-remote stall, and per-RISC spans. Then test the best combination.

Gate: adopt only stable gains with no correctness/control regression. The planner must still cover every
global K block exactly once and reduction must remain correct.

**Stop and report Task 7.**

### Task 8 — Receive-once local fanout and fabric/core placement

1. Ensure A crosses fabric once per destination device, not once per `Ns` consumer.
2. Prototype receive-once fanout to `Ns` groups while preserving separate M payloads for `Sm`.
3. Compare direct ring-core endpoints with dedicated ingress/egress workers fed by local NoC. Confirm local A
   is not reread from DRAM.
4. Sweep a bounded set of links, workers per link, mux buffers, ingress count, and reserved-core placements.
5. Re-run bank-adjacent in1 placement, `IN1_NEAR`, and PARETO local-ring route selection after reserving
   fabric resources. Check whether fewer compute cores plus more fabric workers wins.

Gate: no duplicate fabric bytes from `Ns`; no in1 BRISC regression hidden by total wall time; adopted
placement/fanout must win on at least two relevant shapes and remain neutral on controls.

**Stop and report Task 8.**

### Task 9 — Production semantics, tails, fusions, trace, and cache replay

1. Add non-divisible logical K and per-device tails without reading or transmitting wholly padded payloads.
2. Compose bias, activation, addcmul, and output chunking exactly once after local `Pk` reduction, matching
   single-chip dtype and validation rules.
3. Support persistent buffers only where required by the selected path; make gathered-A materialization
   optional rather than an unavoidable side effect.
4. Validate program-cache hashing for topology, D, links, transport policy, chunking, placement, semaphores,
   fusions, and memory layout.
5. Validate trace capture/replay, multiple consecutive invocations, semaphore reset, and mux teardown.

Gate: full correctness matrix passes fresh and cached, all fusion combinations pass, trace replay is stable,
watcher is clean, and D=1 remains unchanged.

**Stop and report Task 9.**

### Task 10 — Fabric-aware picker and final production sweep

Only now add automatic selection. Train or calibrate it from measured data; do not encode unexplained
shape-specific guesses without documenting them.

Sweep:

- D=2/4/8; linear and ring where physically valid.
- One/two links and the bounded accepted worker/mux choices.
- `Mt=1/2/4/8`, including all available LTX/FLUX Regime-A shapes.
- Local-K-rich, early-stall, narrow-N, wide-N, deep-K, shallow-K, `Pk=1/>1`, `Ns=1/>1`, `Sm=1/>1`, and
  balanced-tail cases.
- Unfused, fused staged, fused direct/hybrid, and the ideal lower bound.

Produce a human-readable table containing shape, D/topology, selected Regime-A config, fabric config,
transport policy/chunk/depth, wall time, all relaunches, PCC, `T_mm`, `T_ag`, `T_unfused`, speedup versus
unfused, overlap efficiency, per-RISC spans, fabric utilization, and bottleneck classification.

Gate: zero correctness regressions; no unexplained performance regressions; fused execution approaches
`max(T_mm,T_ag)` on shapes where overlap is physically possible; remaining misses have evidence-backed
explanations. Remove rejected implementation paths after preserving reports/raw data and recovery hashes.

**Stop and report Task 10. Do not declare the op production-ready until the orchestrator reviews the final
table and explicitly accepts it.**

## Representative performance corpus

At minimum include these shapes, using the current single-chip picker/config as a reference and recording
the selected multi-device configuration:

```text
Mt=1:  32x6144x3072, 32x6144x4608, 32x6144x6144, 32x15360x768
Mt=4:  128x2048x512, 128x6144x768, 128x6144x2304, 128x6144x4608
Mt=8:  256x2048x512, 256x2048x1024, 256x6144x768,
       256x6144x1536, 256x6144x4608, 256x15360x768
```

Add the full LTX/FLUX `Mt <= 8` set from `tools/mm_sweep/REGIME_A_CURRENT_PERF_REPORT.md`, plus matched-M
families and tails. Early tasks may use a four-shape subset, but every adoption decision must include:

- a narrow-N fabric-sensitive shape,
- a wide-N in1-bound control,
- a deep-K shape with substantial steady state,
- a shallow-K shape likely to stall early,
- and at least one `Sm > 1` and one `Ns > 1` configuration.

## Decision principles

- Preserve the accepted single-chip dataflow unless an isolated A/B proves a replacement.
- Correctness ordering and source lifetime are independent concerns. Early readiness signals do not permit
  early CB-slot reuse.
- A deeper buffer is not proof of correct flow control.
- Never infer overlap from aggregate wall time alone; retain profiler evidence.
- Do not optimize fabric by sacrificing the bank-adjacent in1 path without showing a net corpus win.
- Do not duplicate A over fabric for N partitions.
- Keep experimental mechanisms internal and cache-hashed. Expose only stable, supported policy through the
  public API.
- When a lever fails its gate, revert its production code, preserve the report/raw evidence and recovery
  hash, and stop rather than expanding the experiment indefinitely.
