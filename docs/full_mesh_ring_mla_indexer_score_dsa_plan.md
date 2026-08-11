# Full-mesh ring plan for `ring_mla` and `ring_indexer_score_dsa`

## Goal

Extend `ttnn.transformer.ring_mla` and `ttnn.experimental.ring_indexer_score_dsa` with the generic full-mesh mode introduced by `high_bw_all_gather`:

- `cluster_axis=0` or `1` keeps the current independent axis-ring behavior.
- Explicit `cluster_axis=None` makes every device in a 2D mesh participate in one direct-neighbor snake ring.
- Communication follows snake order, while tensors, causal positions, persistent buffers, and returned results remain in canonical row-major mesh order.
- Existing axis-ring callers, layouts, cache keys, and performance behavior remain unchanged.

This plan covers the reusable ring infrastructure, the two fused operations, tests, and model-integration prerequisites. It does not propose making every `ring_joint_sdpa` mode full-mesh-capable in the first change.

## Required contract

Full-mesh mode should match the `high_bw_all_gather` contract where it applies:

1. The device mesh is two-dimensional, both dimensions are greater than one, and at least one dimension is even.
2. Fabric2D is enabled.
3. The input tensor uses row-major mesh coordinates.
4. The sequence/gather dimension is sharded across all mesh devices, not sharded on one axis and replicated on the other.
5. The persistent gather buffer is replicated across the complete mesh and has space for `mesh_rows * mesh_cols` local sequence shards.
6. The host selects a row snake when its lane count is even and direct, otherwise a direct column snake. Every edge, including ring closure, must be a direct physical Fabric neighbor with every requested link.
7. The effective ring topology is `Topology::Ring`. If the public `topology` argument is not `Ring` when `cluster_axis=None`, reject it rather than silently changing it.
8. `ring_size` is the complete mesh size. The existing 32-rank limit remains, so an 8x4 Galaxy is supported but larger meshes fail explicitly until masks and fixed-size reader state are widened.
9. The requested `num_links` must be nonzero and supported by both participating physical axes and every selected snake edge. Keep the current public defaults; do not silently increase an existing caller's worker count.

Keep `cluster_axis` a required keyword in Python, but change its type to `Optional[int]`. Callers must deliberately pass `None` to select the new behavior.

## Central design rule: transport rank is not tensor rank

On an axis ring, the same integer currently serves as all of the following:

- the device's position in the communication ring;
- the ID signaled by the all-gather producer;
- the destination shard offset in the gathered buffer;
- the device's global causal/query rank.

That equivalence is false for a snake. For example, the second row of a row snake is traversed in reverse, but its tensor shards must remain in row-major order.

Introduce and consistently name two ranks:

- `transport_rank`: snake position. Use it for neighbors, forward/backward work division, parity, `RingIdSequencer`, and producer/consumer semaphore IDs.
- `tensor_rank`: canonical row-major mesh position. Use it for persistent-buffer offsets, logical K-shard IDs, query causal offsets, KV-pad rotation, block-cyclic ownership, and output composition.

For an axis ring these values remain identical. For a full-mesh ring:

```text
transport_rank --snake mapping--> mesh coordinate --row-major mapping--> tensor_rank
```

Every host and device path must use the same mapping. In particular, program creation and program-cache runtime patching must not derive ranks independently.

## Phase 1: extract reusable mesh-ring infrastructure

The snake implementation and direct-route proof are currently private to `high_bw_all_gather`. Move the reusable pieces into CCL-owned code instead of including an experimental all-gather header from SDPA and indexer code.

Proposed files:

- `ttnn/cpp/ttnn/operations/ccl/shared_with_host/snake_ring.hpp`
  - `Orientation`;
  - coordinate-to-transport-rank conversion;
  - transport-rank-to-coordinate conversion;
  - transport-rank-to-row-major-tensor-rank conversion.
- `ttnn/cpp/ttnn/operations/ccl/common/host/mesh_ring_plan.hpp`
- `ttnn/cpp/ttnn/operations/ccl/common/host/mesh_ring_plan.cpp`
  - full-mesh precondition checks;
  - row/column snake selection;
  - per-edge, per-link direct-neighbor validation;
  - uniform link-capacity validation across active axes;
  - route-plan hashing;
  - per-coordinate transport and tensor rank plus forward/backward coordinates.

Define a hashable host result along these lines:

```cpp
struct MeshRingPlan {
    std::optional<uint32_t> cluster_axis;
    bool full_mesh;
    snake_ring::Orientation orientation;
    uint32_t mesh_rows;
    uint32_t mesh_cols;
    uint32_t ring_size;
    uint32_t num_links;
    ttnn::ccl::Topology topology;
    uint64_t route_plan_hash;
};

struct MeshRingPosition {
    uint32_t transport_rank;
    uint32_t tensor_rank;
    std::optional<MeshCoordinate> forward_coord;
    std::optional<MeshCoordinate> backward_coord;
};
```

The exact split between the two structs may change during implementation, but the mapping and route hash must be single-sourced.

Refactor `high_bw_all_gather` to consume these shared helpers first, without changing its behavior. This provides a regression-tested reference user before the new operations depend on the abstraction. Update the relevant CMake source lists.

## Phase 2: teach the ring-attention all-gather helper about rank mapping

Both target operations use:

`ttnn/cpp/ttnn/operations/experimental/ccl/ring_attention_all_gather_async/device/ring_attention_all_gather_async_multi_core_with_workers_program_factory.*`

Extend the helper with a small structural rank-mapping descriptor: full-mesh flag, snake orientation, mesh rows, and mesh columns. Keep the current axis-ring code path compile-time identical when the flag is false.

Required kernel changes:

1. Continue using `transport_rank` for target counts, direction splitting, relay order, and synchronization.
2. Translate every local or relayed transport shard ID to `tensor_rank` before calculating its output tile/page offset.
3. Keep fused-op semaphore signaling indexed by `transport_rank`; this describes arrival order, not tensor placement.
4. In fused consumers, map the `RingIdSequencer` result to `tensor_rank` before selecting local versus gathered storage and before calculating K/V offsets.
5. Pass mapping parameters as compile-time arguments so axis mode adds no dataflow-core divisions or modulo operations.
6. Include all structural mapping fields in the parent operation's program-cache key.

Add shared host/device tests proving that row and column snakes are bijections and that output slots are row-major for representative shapes such as 2x2, 2x4, 8x4, and 3x2.

## Phase 3: extend `ring_mla`

### API and attributes

Change `cluster_axis` from `uint32_t` to `std::optional<uint32_t>` through:

- `ttnn/cpp/ttnn/operations/transformer/sdpa/sdpa.hpp`
- `ttnn/cpp/ttnn/operations/transformer/sdpa/sdpa.cpp`
- `ttnn/cpp/ttnn/operations/transformer/sdpa/sdpa_nanobind.cpp`
- `ttnn/cpp/ttnn/operations/transformer/sdpa/device/ring_joint_sdpa_device_operation.hpp`
- `ttnn/cpp/ttnn/operations/transformer/sdpa/device/ring_joint_sdpa_device_operation.cpp`
- the ring-joint parameter and all-gather attribute types.

Add the resolved `MeshRingPlan` fields to `RingJointSDPAParams`/`RingAttentionAllGatherAsyncParams` and their attribute hashes. Do not store coordinate-specific values in a mesh-wide attribute; derive `MeshRingPosition` from the shared plan for each workload coordinate.

Limit the initial public enablement to `ring_mla`. Other callers of the shared ring-joint primitive continue passing an integer axis. Reject full-mesh mode with `sliding_window_size`, because the cyclic neighbor-halo path has separate predecessor/wrap assumptions and `ring_mla` does not expose that mode today.

### Validation and shapes

For `cluster_axis=None`:

- set `ring_size = mesh_rows * mesh_cols`;
- validate the shared full-mesh route plan and requested links;
- require Q and local KV sequence placement to span every device in canonical row-major order;
- require the persistent KV buffer to be complete-mesh replicated;
- require its gathered sequence extent to equal `ring_size * local_kv_sequence_extent`;
- validate the existing ring-mask limit before program construction;
- preserve indexed-cache, `kv_actual_isl`, and trace-safe metadata checks.

The returned attention output remains distributed like Q: its sequence rows are sharded over all devices and compose in row-major device order.

### Program-factory changes

Refactor `RingWritePlan` to carry both ranks. Then audit every current use of `device_index` in `ring_joint_sdpa_program_factory.cpp`:

- use `transport_rank` in `get_forward_backward_configuration`, parity swaps, `RingIdSequencer`, all-gather signaling, and the all-gather helper call;
- use shared-plan neighbor coordinates instead of `get_physical_neighbor_from_physical_coord(..., cluster_axis)`;
- map each delivered transport shard to `tensor_rank` before K/V gathered-buffer access and causal comparisons;
- use local `tensor_rank` for Q global-position calculations, `build_kv_pad_q_mapping`, chunk-start derivation, padding masks, and balanced/unbalanced causal work masks;
- update both the host planning implementation and the device-side metadata/KV-pad derivation in `device/kernels/dataflow/ring_joint_kv_pad_derivation.hpp`;
- apply the identical mapping in `apply_ring_joint_scalar_runtime_args` so cache hits cannot retain an axis-ring causal offset;
- pass the rank-mapping descriptor into the ring-attention all-gather helper.

Every remote shard written to the full-mesh gathered KV buffer must use its row-major slot even though arrival order is a snake. The existing local-source optimization may continue reading the local shard directly without copying it into the persistent output. This lets SDPA address generation interpret every consumed global sequence chunk in natural order.

## Phase 4: extend `ring_indexer_score_dsa`

### API and mode restrictions

Change `cluster_axis` to `std::optional<uint32_t>` in:

- `ttnn/cpp/ttnn/operations/experimental/indexer_score/device/indexer_score_device_operation.hpp`
- `ttnn/cpp/ttnn/operations/experimental/indexer_score/device/indexer_score_device_operation.cpp`
- `ttnn/cpp/ttnn/operations/experimental/indexer_score/indexer_score_nanobind.cpp`.

For explicit `cluster_axis=None`:

- all mesh devices are SP ranks;
- `seq_subshard_axis` must be unset because no orthogonal mesh axis remains for TP query sub-sharding;
- `block_cyclic_sp_axis` must be unset because the block-cyclic SP placement spans the complete mesh rather than one named axis;
- `block_cyclic_chunk_local`, when present, means the per-device slab size and resolves to `BlockCyclicLayout{sp = mesh_size, chunk_local = value}`;
- when both block-cyclic arguments are absent, keep the contiguous layout;
- Q, weights, and `k_local` must be sequence-sharded across every device in row-major order; `k` must be complete-mesh replicated.

Axis mode retains the current rule that `block_cyclic_sp_axis == cluster_axis` and retains optional `seq_subshard_axis` support. The asymmetric `block_cyclic_sp_axis=None, block_cyclic_chunk_local=<value>` form is legal only for fused full-mesh mode, so existing classic and axis-ring validation stays strict.

### Attributes and host helpers

Add the resolved mesh-ring fields to `FusedRingConfig` and its program hash. Replace the implicit `sp_axis()`-based fused routing with a helper returning:

- ring size;
- local transport rank;
- local tensor rank;
- forward/backward coordinates;
- transport-shard-to-tensor-shard mapping.

Keep the current `seq_shard_axes={}` meaning of flat row-major query ranking for the full-mesh causal geometry, but do not use its absence to infer the transport route. `FusedRingConfig.full_mesh` is the unambiguous mode flag.

### Program-factory changes

In `ring_indexer_score_dsa_program_factory.cpp`:

- call `ring_writes_for` and seed `RingIdSequencer` with `transport_rank`;
- build `shard_order[tensor_rank(delivered_transport_rank)] = arrival_iteration` so band readiness refers to the canonical K shard stored in the persistent buffer;
- call `device_causal_geometry` with the local `tensor_rank` and `tp_index=0` in full-mesh mode;
- source the local K slab using `tensor_rank` while keeping producer waits/signals in transport-rank space;
- use the shared forward/backward snake coordinates in the all-gather helper;
- pass the mapping descriptor so gathered K is written in canonical order;
- repeat the same tensor-rank causal derivation in the cache-hit runtime-argument override.

Retain the existing `ring_size <= 32` validation. Confirm that fixed arrays in `reader_indexer_score.cpp`, ring-arrival tables, and semaphore counts are exactly sized for the 32-device Galaxy case.

## Phase 5: tests and regressions

### Shared routing tests

- Row-snake and column-snake forward/inverse mapping for several legal mesh shapes.
- Every rank appears exactly once and maps to the expected row-major tensor rank.
- Correct forward/backward neighbors, including closure.
- Rejection of odd-by-odd meshes, non-row-major coordinate lists, non-direct closure, missing link indices, and over-requested links.
- Route hash changes when orientation, dimensions, topology, links, or physical routes change.

### `ring_mla` tests

Extend `tests/nightly/blackhole/sdpa/test_ring_joint_sdpa.py` with:

- 2x2 full-mesh correctness using explicit `cluster_axis=None` and a fully sequence-sharded Q/KV layout;
- 8x4, 32-rank Galaxy correctness, environment-gated like the full-mesh high-bandwidth all-gather test;
- row-major placement of every remotely gathered persistent-KV shard, not only final attention PCC (the optimized local slot may remain direct-sourced);
- chunked prefill with `kv_actual_isl` and a rotated/nonzero start;
- indexed KV-cache selection;
- trace-safe metadata replay and program-cache reuse across changing runtime scalars;
- deterministic repeated execution;
- existing axis-0 and axis-1 regression cases.

Add negative tests for invalid topology, an axis-sharded/orthogonally-replicated input in full-mesh mode, a non-replicated persistent buffer, and ring size greater than 32.

### `ring_indexer_score_dsa` tests

Extend `tests/ttnn/nightly/unit_tests/operations/experimental/test_ring_indexer_score_dsa_4d.py` and the Galaxy/nightly coverage with:

- 2x2 full-mesh contiguous K;
- 2x2 full-mesh block-cyclic K with `block_cyclic_sp_axis=None` and `block_cyclic_chunk_local` set;
- a rotated block-cyclic chunk that exercises straddle geometry;
- indexed multi-slot `k_local`, bounded `kv_len`, and cache-hit scalar patching;
- explicit verification that every remotely gathered K shard is in its row-major slot (the optimized local slot may remain direct-sourced);
- 8x4 full-mesh correctness at the 32-rank limit;
- existing 1x4 and 2x2 SPxTP axis-ring regressions.

Add negative tests for supplying `seq_subshard_axis` or `block_cyclic_sp_axis` in full-mesh mode, invalid tensor placement, invalid topology, unavailable links, and more than 32 ranks.

## Phase 6: model integration prerequisites

Operation support alone does not make the current DeepSeek model use all devices as one SP ring. Today the model assigns one mesh axis to SP and the other to TP; KV is SP-sharded and TP-replicated, while MLA Q heads and several weights are TP-sharded. Passing `cluster_axis=None` to that layout would gather duplicate KV shards and produce incorrect causal ranks.

Add model adoption behind an explicit configuration only after the operation tests pass:

1. Create a full-mesh sequence mapper for Q, indexer weights-by-row, local KV/cache shards, and persistent gather buffers.
2. Change cache writers and block-cyclic cache construction to use `sp = mesh_size` in canonical row-major rank order.
3. Replicate or otherwise redistribute tensors currently TP-sharded on the second axis; account for the resulting memory and compute cost.
4. Allocate/cycle one full-mesh semaphore set rather than independent per-axis sets.
5. Update `chunk_size_global`, logical lengths, RoPE/cache rank derivation, and output composers to use the full mesh size.
6. Preserve the existing SPxTP path as the default until full-model accuracy, memory capacity, and performance are qualified.

This model-layout change is substantial and should be a separate commit from the reusable operation support.

## Suggested implementation sequence

Every numbered step below has a mandatory Claude Opus review gate. Complete the implementation and focused validation for one step, obtain and address the review, and only then proceed to the next step.

1. Extract snake math and host route planning; refactor `high_bw_all_gather` with no functional change. Run the focused checks, then complete the Claude Opus review gate.
2. Add rank mapping to the ring-attention all-gather helper and directly test row-major gather placement. Run the focused checks, then complete the Claude Opus review gate.
3. Implement `ring_mla` full-mesh validation, routing, causal-rank mapping, and cache-hit handling. Run the focused checks, then complete the Claude Opus review gate.
4. Add and pass `ring_mla` 2x2 tests, then Galaxy coverage. Complete the Claude Opus review gate for the implementation and test sufficiency.
5. Implement `ring_indexer_score_dsa` attributes, block-cyclic API semantics, routing, readiness mapping, and cache-hit handling. Run the focused checks, then complete the Claude Opus review gate.
6. Add and pass indexer 2x2 tests, then Galaxy coverage. Complete the Claude Opus review gate for the implementation and test sufficiency.
7. Run all existing axis-ring suites to prove backward compatibility, then complete a Claude Opus regression review of the complete operation change.
8. Add model integration behind a feature/configuration switch and measure it separately. Complete a final Claude Opus review covering model correctness, layouts, memory, performance evidence, and rollout safety.

Keep these as reviewable commits. In particular, land the shared-helper extraction separately so any `high_bw_all_gather` regression is easy to isolate.

## Mandatory Claude Opus review gate

Use Claude CLI with permission prompts disabled and explicitly select Opus for every review. Do not substitute a smaller model if Opus is slow or temporarily busy.

```bash
claude --dangerously-skip-permissions --model opus "Review the current implementation step for the full-mesh ring_mla and ring_indexer_score_dsa plan. Inspect the worktree diff and relevant surrounding code. Look for correctness bugs, axis-ring regressions, transport-rank versus tensor-rank mistakes, route-validation gaps, program-cache miss/hit drift, unsafe device-kernel assumptions, missing validation, and insufficient tests. Report findings in severity order with exact file and line references."
```

For each gate:

1. Give Claude the current step's intent, changed files, relevant test results, and any known limitations in the prompt or follow-up context.
2. Be patient and allow the Opus review to finish. Do not interrupt, restart, or downgrade the model merely because review takes a long time.
3. Treat correctness, hang/deadlock, cache-key, stale-runtime-argument, layout, and regression findings as blocking.
4. Address every blocking finding and document the disposition of non-blocking suggestions.
5. Rebuild or rerun focused tests affected by the fixes.
6. Run another `claude --dangerously-skip-permissions --model opus ...` review after material fixes. Repeat until Opus reports no blocking findings for that step.
7. Record the Claude review command, reviewed commit/worktree state, findings, dispositions, and validation results in the implementation notes or change description before starting the next step.

The final implementation is not done until every step has passed its own Opus review and the complete accumulated diff has received a final Opus review.

## Validation commands

If the implementation branch is rebased and submodule SHAs changed, update them before building:

```bash
git submodule update --init --recursive
```

Build the release configuration with the repository wrapper:

```bash
./build_metal.sh --release
```

Run tests only through the safe pytest wrapper, starting with focused suites:

```bash
scripts/run_safe_pytest.sh tests/ttnn/unit_tests/operations/experimental/test_high_bw_all_gather.py
scripts/run_safe_pytest.sh tests/nightly/blackhole/sdpa/test_ring_joint_sdpa.py
scripts/run_safe_pytest.sh tests/ttnn/nightly/unit_tests/operations/experimental/test_ring_indexer_score_dsa_4d.py
scripts/run_safe_pytest.sh tests/ttnn/nightly/unit_tests/operations/experimental/test_ring_indexer_score_dsa.py
```

Then run the relevant model sparse-MLA tests and hardware-gated Galaxy cases through the same wrapper. Record mesh shape, Fabric configuration, selected snake orientation, resolved link count, route hash, PCC/RMSE, determinism result, and program-cache entry count in the change description.

## Definition of done

- Explicit `cluster_axis=None` works for both operations on legal 2D meshes and produces row-major tensor semantics over a snake transport ring.
- 2x2 and 8x4 accuracy tests pass, including block-cyclic, rotated-chunk, indexed-cache, metadata, and cache-hit cases relevant to each operation.
- Existing integer-axis tests remain unchanged and pass.
- Invalid routes and layouts fail on the host with actionable errors rather than hanging.
- Program-cache keys distinguish all structural route/mapping variants; runtime scalar changes reuse and correctly patch cached programs.
- `high_bw_all_gather` continues to pass after consuming the shared routing implementation.
- The release build succeeds with `./build_metal.sh --release`, and all pytest invocations use `scripts/run_safe_pytest.sh`.
