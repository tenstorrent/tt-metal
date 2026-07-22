# AGMM Task 2 — op scaffold + pure host plan

Execution-plan reference: `REGIME_A_AGMM_EXECUTION_PLAN.md`, Task 2. Commit basis: merged `de7df778f44` (+ this).

## What was scaffolded
New experimental op `ttnn.experimental.all_gather_regime_a_matmul_async` under
`ttnn/cpp/ttnn/operations/experimental/all_gather_regime_a_matmul_async/` (its own library, links regime_a):

- `all_gather_regime_a_matmul_async.{hpp,cpp}` — public op. Signature matches regime_a API style + the AGMM
  fabric arguments (`cluster_axis`, `topology`, `num_links`, `num_workers_per_link`, `num_buffers_per_channel`,
  `multi_device_global_semaphore`, `barrier_semaphore`, `persistent_output_buffer`). **D is inferred from the
  K-shard ratio** `D = in1.K_global / in0.K_local`.
- `device/all_gather_regime_a_matmul_async_plan.hpp` — **pure, device-free host plan** (the intellectual
  core): given `(M,K,N,D,topology,links,workers,regime_a cfg,C,slots,grid)` it computes, validates, and never
  throws:
  - **explicit global-K-block ownership** per device (a list of global block ids; ownership is not baked into
    kernel arithmetic — Task 7 can swap in cyclic/balanced),
  - **fabric core reservation** (mux = links·(ring?2:1), workers = links·workers) reserved from the grid tail,
    **before** the regime_a compute window `[16,104)`; reports `core_fit` and `core_collision`,
  - **L1 sizing** for ingress transport buffering (`slots · C · kb · Mt · 2048 B`, capped at ⅓ L1),
  - constraint checks (tile-aligned K, `Kt%D==0`, `Kt%kb==0`, whole K-blocks per device).
- `device/..._device_operation_types.hpp` — params/inputs structs (Task-3 scaffold for the streaming prim).
- `..._nanobind.{hpp,cpp}` — binds the op; also exposes a device-free
  `ttnn._ttnn.operations.experimental.all_gather_regime_a_matmul_plan(...)` returning the plan as a dict
  (used by the offline tests).
- `sources.cmake`, `CMakeLists.txt`; wired into `experimental/CMakeLists.txt` (add_subdirectory + umbrella
  link + `$<TARGET_OBJECTS>`) and `experimental_nanobind.cpp`.

## Behavior (v1)
- **D=1** → delegates to `regime_a_matmul` (behaviorally identical; test asserts byte-for-byte equality).
- **D>1** → builds + validates the host plan, then reports that the fabric-streaming path is implemented in
  Task 3. bf16 only, no transpose/batching, tile-aligned K sharding, no epilogues — all validated.

## Deviation from the plan (scoping)
The plan lists "device operation, factory, kernels" as Task-2 scaffold. The **device-operation prim +
program factory + streaming kernels are deferred to Task 3**, where the D>1 device work actually lives; the
`device_operation_types.hpp` + `_plan.hpp` are in place now. Rationale: Task 2's gate is offline plan tests +
D=1 correctness (D>1 does not execute yet), and "delegate to regime_a" for D=1 is explicitly sanctioned.
Deferring avoids a large, non-functional device-op stub. Flagging for orchestrator visibility.

## Tests (`tests/ttnn/unit_tests/operations/matmul/test_all_gather_regime_a_matmul_async.py`) — 13 passed
- Offline plan (device-free), D=1/2/4/8 × {ring, linear}: global-K-block coverage (every block owned exactly
  once, no gaps/dupes), `core_fit` + no collision, `l1_fit`; fabric core counts.
- Invalid/edge: `Kt` not divisible by D; `Kt` not divisible by kb; core over-subscription; L1 over-budget —
  each returns `valid=False` with a specific error.
- D=1 correctness on device: op == `regime_a_matmul` byte-for-byte (PCC 0.999 vs torch golden).

## Gate
Offline plan tests cover D=1/2/4/8, topology, links, K ownership, core collisions, L1 sizing ✅.
D=1 matches the parent op's correctness with no perf regression (it *is* regime_a) ✅.
