# Full-mesh ring implementation notes

This log records the required Claude Opus review gates and focused validation for
`docs/full_mesh_ring_mla_indexer_score_dsa_plan.md`.

## Step 1: shared snake math and mesh-ring planning

Worktree base: `0ab7cfa04ed` on `pjosipovic/generic-mesh-high-bw-all-gather`.

Implementation:

- Moved the shared snake mapping into CCL-owned host/device code.
- Added a CCL host route planner and per-coordinate rank/neighbor resolver.
- Refactored `high_bw_all_gather` route selection and its unicast program factory to
  consume the shared helpers.
- Added host gtest coverage for row/column mapping bijections and canonical
  row-major tensor ranks on 2x2, 2x4, 8x4, and 3x2 meshes.

Initial Opus review command:

```bash
claude --dangerously-skip-permissions --model opus "Review Step 1 of docs/full_mesh_ring_mla_indexer_score_dsa_plan.md ..."
```

Claude session: `5b8ba4f3-cb07-4212-b8bb-c62e5ac85b36` (Opus 5, xhigh effort).

Initial review findings and dispositions:

- Blocking: no direct execution evidence for extracted snake math. Addressed with
  `CclHelpers.SnakeRingMappingsAreBijectionsWithRowMajorTensorRanks`.
- Blocking: `get_mesh_ring_position` had no reference consumer and the high-bandwidth
  factory duplicated its mapping. Addressed by constructing the resolved structural
  plan in the factory and using the shared position result.
- Invalid direct-route shapes/axes and invalid full-mesh position geometry were not
  guarded. Addressed with explicit host validation.
- The plan omitted Fabric configuration and both axis topologies from its structural
  cache state. Addressed by storing and hashing those values.
- Position geometry came from both the plan and live mesh. Addressed by consistently
  using the resolved plan dimensions after validating the live tensor/coordinate.
- Degenerate full-mesh rejection was silent. Addressed with an actionable warning.
- Tensor-placement parsing was duplicated. Addressed by exporting and reusing the CCL
  helper.
- Null-device handling was inconsistent. Addressed by returning false from the
  row-major predicate and rejecting it in route construction.
- Standalone interface-header verification was suggested. Both public headers are
  compiled by the focused CCL and high-bandwidth targets; a full release build remains
  part of the final validation gate.
- The host-only plan header was unnecessarily in the JIT kernel file set. Removed;
  only the shared host/device snake header remains there.

Focused validation after fixes:

```text
cmake --build build_Release --target ttnn_op_ccl ttnn_op_experimental_high_bw_all_gather unit_tests_ttnn_ccl -j 8
PASS

build_Release/test/ttnn/unit_tests_ttnn_ccl \
  --gtest_filter=CclHelpers.SnakeRingMappingsAreBijectionsWithRowMajorTensorRanks
PASS (1 test)

scripts/run_safe_pytest.sh \
  tests/ttnn/unit_tests/operations/experimental/test_high_bw_all_gather.py
PASS (8 passed, 14 hardware/environment-gated skips)
```

The available host is an eight-device Blackhole system. The exact 2x2 and 8x4
full-mesh hardware cases remain environment-gated; the pure mapping paths are covered
without hardware by the new gtest.

First re-review result: PASS, no Step 1 blockers. Opus identified seven
non-blocking hardening/coverage suggestions. Before closing the gate, the shared
planner was updated to enforce Fabric2D for full-mesh mode, the reference factory
was changed to use and validate cached structural mesh dimensions, legal snake
closure/adjacency was added to the gtest, public invalid-input behavior was made
consistent, dimension-count naming was clarified, host tensor-rank derivation was
single-sourced through `row_major_index`, and zero-dimension mapping helpers were
made non-dividing.

Validation after the review hardening:

```text
cmake --build build_Release --target ttnn_op_ccl ttnn_op_experimental_high_bw_all_gather unit_tests_ttnn_ccl -j 8
PASS

build_Release/test/ttnn/unit_tests_ttnn_ccl \
  --gtest_filter=CclHelpers.SnakeRingMappingsAreBijectionsWithRowMajorTensorRanks
PASS (1 test)
```

The full-mesh hardware path did not execute on this eight-device host: the 2x2
case requires exactly four physical devices and the 8x4 case requires Galaxy or
simulator enablement. The moved runtime logic remains textually equivalent to the
pre-refactor implementation, while the pure mapping and legal ring edges now have
host coverage.

Final re-review command:

```bash
claude --dangerously-skip-permissions --model opus --print "Final re-review for Step 1 ..."
```

Reviewed state: worktree based on `0ab7cfa04ed`, including all Step 1 files and
the post-review hardening above. Result: **PASS; no blocking Step 1 findings**.

Final non-blocking findings and dispositions:

- Some planner rejection branches still return without their own diagnostic.
  Carry forward to the `ring_mla`/indexer host-validation steps, where invalid
  inputs will receive public actionable errors without changing existing
  high-bandwidth behavior.
- The planner does not yet fold `has_row_major_mesh_coordinates` into full-mesh
  resolution. Carry forward as a blocking prerequisite for Step 3, before
  `ring_mla` becomes the first new planner consumer. The existing high-bandwidth
  caller already enforces this precondition.

Step 1 gate status: **complete**.

## Step 2: ring-attention all-gather rank mapping

Implementation:

- Added `RingAttentionRankMapping` with full-mesh flag, snake orientation, and
  mesh dimensions.
- Added a lightweight shared host/device transport-to-tensor rank function.
- Passed the mapping fields to both directions of the all-gather reader and
  writer as compile-time arguments.
- Kept target scheduling, relay order, parity, and fused semaphore signaling in
  transport-rank space.
- Mapped local and relayed writer output offsets plus reader relay-source offsets
  into canonical tensor-rank space. The axis specialization is compile-time
  identity and does not evaluate snake divisions.
- Added a host gtest for axis identity and full-mesh row-major mapping.
- Canonicalized disabled mapping arguments to `{false, Row, 0, 0}` inside the
  helper so existing axis-ring JIT binaries do not become mesh-shape-specific.
- Added full-mesh boundary validation that the target coordinate, transport
  rank, snake orientation, and forward/backward neighbors all describe the same
  ring, including the even-lane closure requirement.
- Shared the reader/writer fixed compile-time argument counts between host and
  device code and asserted all four host argument lists before appending tensor
  accessors.
- Replaced the mapping wrapper's tautological test with hard-coded row- and
  column-snake permutations.

Focused validation:

```text
cmake --build build_Release --target ttnn_op_experimental_ccl \
  ttnn_op_transformer ttnn_op_experimental_indexer_score unit_tests_ttnn_ccl -j 8
PASS

unit_tests_ttnn_ccl --gtest_filter=<snake and ring-attention mapping tests>
PASS (2 tests)

./build_metal.sh --release
PASS

scripts/run_safe_pytest.sh tests/nightly/blackhole/sdpa/test_ring_joint_sdpa.py \
  -k test_ring_mla_nd_sharded_indexed_kv_cache_accuracy --maxfail=1
PASS (1 passed, PCC 0.999823)

scripts/run_safe_pytest.sh \
  tests/ttnn/nightly/unit_tests/operations/experimental/test_ring_indexer_score_dsa.py \
  -k "test_indexer_score_ring4_fused and not bfp8 and not production and glm5 and contiguous" --maxfail=1
PASS (1 passed)
```

Validation incident dispositions:

- The first ring-MLA attempt loaded the stale pre-install
  `build_Release/lib/_ttnncpp.so` after only targeted builds and correctly failed
  JIT argument-count validation. Running the required release build installed the
  rebuilt module; the identical safe pytest invocation then passed.
- The direct 1x4 QuietBox indexer fixture timed out during Fabric firmware
  initialization before program creation on this eight-device host. The intended
  LoudBox 2x4-to-1x4 fixture initialized successfully and the same indexer/helper
  axis path passed.

Initial Claude Opus review command:

```text
claude --dangerously-skip-permissions --model opus --print \
  "Review Step 2 of docs/full_mesh_ring_mla_indexer_score_dsa_plan.md ..."
```

Reviewed state: worktree based on `0ab7cfa04ed`, including all Step 1 and Step
2 files. Result: **PASS; no blocking Step 2 findings**.

The review's two highest-priority non-blocking findings were addressed before
Step 3: disabled mapping arguments are now structurally canonicalized, and the
helper cross-validates full-mesh transport rank and neighbor geometry. The
recommended compile-argument count guard, hard-coded expected permutations,
transport-rank naming, and parameter contract documentation were also added.
The focused targets compile, both mapping gtests pass, and `git diff --check`
passes after these changes.

Final post-fix Claude Opus review command:

```text
claude --dangerously-skip-permissions --model opus --print \
  "Re-review Step 2 ... after addressing your initial findings ..."
```

Result: **PASS; no blocking Step 2 findings remain; Step 3 may begin**.
The review re-audited all relay/local offsets and all four compile-time argument
lists. Before closing the gate, the four changed-line format findings were
fixed, the current module was rebuilt and installed with
`./build_metal.sh --release`, and the focused safe ring-MLA axis test was rerun:

```text
scripts/run_safe_pytest.sh tests/nightly/blackhole/sdpa/test_ring_joint_sdpa.py \
  -k test_ring_mla_nd_sharded_indexed_kv_cache_accuracy --maxfail=1
PASS (1 passed, PCC 0.999823)
```

Step 3 blocking carry-forwards:

- Cross-check full-mesh mapping dimensions against the live mesh shape before
  enabling the first caller.
- Put the canonicalized mapping and resolved route hash in the ring-MLA parent
  attributes/program-cache key.

Step 2 gate status: **complete**. The full-mesh kernel branch remains
intentionally unreachable until Step 3 wires its first caller; full-mesh runtime
and placement evidence belongs to Step 4.

## Step 3: `ring_mla` full-mesh behavior

Implementation:

- Changed only the public `ring_mla` `cluster_axis` argument to
  `std::optional<uint32_t>`/Python `Optional[int]`; the public ring-joint API
  remains axis-only.
- Resolves `cluster_axis=None` through the shared direct-neighbor mesh-ring
  planner, requires explicit Ring topology, Fabric2D, row-major tensor mesh
  coordinates, complete-mesh sequence sharding, a complete-mesh replicated
  persistent buffer, exact gathered extent, and the existing 32-rank limit.
- Added canonical full-mesh flag/orientation/dimensions/route hash fields to the
  all-gather attributes and program-cache key. Axis mode stores only canonical
  disabled mapping fields, so it does not become mesh-shape-specific.
- Refactored the ring write plan to carry transport rank, tensor rank, and the
  shared planner's forward/backward coordinates. The same plan is rebuilt for
  cache-hit scalar patching.
- Keeps target splitting, fused signaling, sequencer seeds, and neighbor routing
  in transport-rank space. Maps every delivered sequencer rank before causal
  work masks, K/V addresses, local-source selection, Q-pad rotation, and output
  masking.
- Passed the identical compile-time mapping descriptor to reader, writer, and
  compute kernels and updated the shared device-side metadata derivation, so
  trace-safe cache hits use canonical tensor ranks as well.
- Hardened the shared all-gather helper to cross-check mapping dimensions against
  the live mesh and reject non-Fabric2D or singleton-dimension full-mesh calls.

Focused validation:

```text
cmake --build build_Release --target ttnn_op_transformer \
  ttnn_op_experimental_ccl -j 16
PASS

./build_metal.sh --release
PASS

scripts/run_safe_pytest.sh tests/nightly/blackhole/sdpa/test_ring_joint_sdpa.py \
  -k test_ring_mla_nd_sharded_indexed_kv_cache_accuracy --maxfail=1
PASS (1 passed, PCC 0.999823)

git diff --check
PASS
```

Full-mesh runtime correctness and gathered-buffer placement tests are intentionally
the next numbered step.

Claude Opus Step 3 review command:

```text
claude --dangerously-skip-permissions --model opus --print \
  "Review Step 3 ... as a mandatory Opus gate ..."
```

Reviewed state: worktree based on `0ab7cfa04ed`, complete accumulated Step 1-3
diff. Result: **PASS; no blocking Step 3 findings; Step 4 may begin**.

Non-blocking findings and dispositions:

- Full route resolution is repeated on eager dispatch. Carry route-plan
  memoization to model-integration/performance work before decode latency is
  evaluated.
- Make the shared primitive's axis validation message operation-neutral, reject
  representable-but-unsupported standalone full-mesh all-gather attributes, and
  defensively canonicalize the SDPA mapping descriptor. These are hardening
  follow-ups, not runtime-test prerequisites, and remain on the accumulated
  review list.
- Optional V/joint full-mesh placement is not validated. It is unreachable from
  the only enabled public full-mesh entry point (`ring_mla`, latent-V, no joint
  tensors); add it before widening the shared primitive.
- The output data is row-major sequence-sharded like Q, but the existing output
  tensor topology metadata is replicated. Step 4 must validate with an explicit
  row-major composer; derive output topology from Q before model integration.
- Replace the nonzero route-hash sentinel with an explicit resolved flag during
  later attribute cleanup.
- Resolved mapping fields live only in the nested all-gather attributes and are
  hashed transitively by `RingJointSDPAParams`. This deliberate single-source
  deviation from the plan avoids duplicated state.

Step 3 gate status: **complete**.

## Step 4: `ring_mla` full-mesh tests

Coverage added:

- A complete-mesh full-prefill test with explicit `cluster_axis=None`, flat
  row-major sequence sharding, causal PCC, bit-exact repeat execution, and an
  explicit row-major output composer.
- Direct inspection of every destination's persistent gathered-KV buffer. Each
  remote canonical tensor rank must occupy its row-major slot; the optimized
  local slot is asserted to remain unwritten.
- The same test is parameterized for an exact physical 2x2 QuietBox and the
  complete detected mesh. On this eight-device LoudBox, complete means 2x4; on
  Galaxy the unchanged `MESH_CONFIG` makes it the full 4x8, 32-rank ring.
- A full-mesh chunked indexed-cache test with two changing logical lengths,
  an unselected random cache slot, two bit-exact iterations, PCC/RMSE checks,
  and an assertion that the second iteration adds no program-cache entries.

Focused validation on the 2x4 Blackhole host:

```text
scripts/run_safe_pytest.sh \
  'tests/nightly/blackhole/sdpa/test_ring_joint_sdpa.py::test_ring_mla_full_mesh_accuracy_row_major_gather_and_cache_reuse[2x2]' \
  'tests/nightly/blackhole/sdpa/test_ring_joint_sdpa.py::test_ring_mla_full_mesh_accuracy_row_major_gather_and_cache_reuse[complete_mesh]' \
  --maxfail=1 -s
PASS (complete 2x4: PCC 0.999856; exact-2x2 case skipped)

scripts/run_safe_pytest.sh \
  tests/nightly/blackhole/sdpa/test_ring_joint_sdpa.py::test_ring_mla_full_mesh_chunked_indexed_cache_accuracy_and_determinism \
  --maxfail=1 -s
PASS (PCC 0.999872 and 0.999829; bit-exact replay; cache count stable)
```

The first attempt to open a logical 2x2 subset of this physical 2x4 host with
`FABRIC_2D_TORUS_XY` timed out in Fabric firmware initialization, before the
operator ran. The high-bandwidth all-gather reference fixture likewise permits
its 2x2 full-mesh case only when exactly four physical devices are present.
The MLA test now follows that hardware rule rather than treating a non-closed
logical subset as a physical torus.

Initial Claude Opus Step 4 review result: **gate held**. It found an axis-only
test-helper regression plus missing runnable coverage for runtime scalar
patching, metadata derivation, and negative validation. Dispositions:

- Restored separate Q and latent-KV mappers in the axis chunked indexed helper;
  Q remains TP/SP sharded while the one-head latent KV is TP replicated.
- Extended the fixed-shape `kv_actual_isl` helper to full-mesh mode. Three
  rotated starts (`0`, `160`, `320`) share one physical KV/persistent shape,
  exercise cache-hit tensor-rank scalar patching, replay bit-exactly, and pass
  PCC (`0.999901`, `0.999867`, `0.999850`).
- Parameterized indexed and rotation metadata-vs-scalar tests for full-mesh.
  Nonzero slot selection and nonzero rotated-start derivation are bit-exact.
- Added host rejection tests for Linear topology, axis-only sequence placement,
  and a non-replicated persistent buffer. The greater-than-32-rank case remains
  hardware-limited; its host validation was audited by Opus.
- Added a balanced full-mesh case; its causal PCC is `0.999853` and repeat
  execution is bit-exact.
- Made program-cache reuse explicit in the full-prefill test and stopped
  requiring the optional local-slot-elision optimization. Remote placement
  remains checked exactly.

The focused axis regression group requested by Opus passes through the safe
wrapper: chunked indexed cache, `kv_actual_isl` reuse-max, indexed metadata,
and rotation metadata (**4 passed**).

The consolidated current full-mesh group also passes through the safe wrapper
(**10 passed**): balanced/unbalanced full-prefill placement and cache reuse,
chunked indexed cache, rotated `kv_actual_isl` cache patching, three host
rejections, both cache slots through metadata, and three metadata-derived
rotation starts.

Final post-fix Claude Opus Step 4 review result: **PASS; no blocking findings;
Step 5 may begin**. Opus independently re-audited the axis/full mapper split,
rank-dependent cache-hit patching, host/device metadata derivation, negative
failure ordering, balanced semantics, and row-major placement assertion. The
remaining suggestions are non-blocking hardening: add an exact gathered-extent
negative, avoid stale-buffer masking between identical placement runs, document
unused reconstructed planner fields, and fix output topology metadata before
model chaining.

Step 4 gate status: **complete**. Exact physical 2x2 and Galaxy 4x8 executions
remain honest hardware gates for final sign-off; the complete 2x4 run exercises
a non-identity snake permutation and therefore validates the rank mapping.

## Step 5: `ring_indexer_score_dsa` full-mesh implementation

Implementation:

- Changed the required `cluster_axis` keyword to `Optional[int]`. An integer keeps
  the existing axis ring; explicit `None` resolves the shared direct-neighbor
  complete-mesh snake plan and requires Ring topology.
- Added canonical full-mesh flag, orientation, dimensions, and route hash fields
  to `FusedRingConfig` and its program-cache hash.
- Full-mesh validation requires row-major Q/K/weights/K-local coordinates,
  complete-mesh sequence sharding for Q/weights/K-local, a complete-mesh
  replicated persistent K buffer, no TP sub-shard axis, no named block-cyclic SP
  axis, and the existing 32-rank limit.
- Made `block_cyclic_chunk_local` legal without `block_cyclic_sp_axis` only in
  fused complete-mesh mode. It resolves to `sp = mesh_size`; classic and axis-ring
  argument pairing and axis-equality rules are unchanged.
- Split the program factory's rank usage: snake transport rank controls ring
  writes, sequencer seeds, neighbors, and producer signaling; canonical tensor
  rank controls causal geometry, local K ownership, gathered-buffer addresses,
  and readiness-table slots.
- Reconstructed shared snake neighbors from the cached structural plan and passed
  the same rank-mapping descriptor to the ring-attention all-gather helper.
- Extended the reader compile-time arguments so delivered transport IDs are mapped
  before shard readiness and local-versus-gathered K selection. The classic and
  axis-ring factories append canonical disabled mapping fields.
- Kept cache-hit causal scalar patching in canonical tensor-rank space through the
  existing `device_index_for` helper; route/mapping fields are structural and are
  included in the cache key.
- Updated C++ and Python-facing API documentation for the explicit full-mesh and
  block-cyclic contracts.

Focused validation before review:

```text
cmake --build build_Release --target ttnn_op_experimental_indexer_score -j16
PASS

./build_metal.sh --release
PASS

scripts/run_safe_pytest.sh \
  tests/ttnn/nightly/unit_tests/operations/experimental/test_ring_indexer_score_dsa.py \
  -k 'test_indexer_score_ring4_fused and not bfp8 and not production and glm5 and contiguous' \
  --maxfail=1 -s
PASS (1 passed; fused axis-ring output matched the reference)

git diff --check
PASS
```

Full-mesh runtime and negative coverage are intentionally Step 6 and have not yet
been added. Step 5 must pass its Claude Opus implementation gate before that work
begins.

Initial Claude Opus Step 5 review result: **HOLD**. The review found one
blocking correctness issue: full-mesh block-cyclic mode has no named SP axis, so
`device_causal_geometry` selected the approximate flat-linear branch. A rotated
`chunk_start_idx` therefore assigned causal diagonals to the wrong canonical
tensor ranks and could mark the wrong devices as straddling.

Disposition:

- Full-mesh fused block-cyclic mode now selects the rotation-exact SP ownership
  branch using canonical tensor rank. This mirrors the cache writer's rotated
  slab placement and is shared by program creation and cache-hit scalar patching.
- Added a validation invariant that full-mesh `block_cyclic.sp == ring_size`,
  guarded empty/malformed replicate placement metadata, rejected zero links at
  the public entry, asserted 2D coordinates before snake conversion, documented
  `chunk_local == Sq`, and clarified transport/tensor rank log labels.
- Deferred only non-blocking structural/performance suggestions: centralize the
  replicated-topology predicate, memoize route resolution, hoist a currently
  redundant rank-plan check out of the dead `consumers_only` path, and derive
  output topology metadata from Q before model chaining.

Validation after the blocking fix:

```text
cmake --build build_Release --target ttnn_op_experimental_indexer_score -j16
PASS

./build_metal.sh --release
PASS

scripts/run_safe_pytest.sh \
  tests/ttnn/nightly/unit_tests/operations/experimental/test_ring_indexer_score_dsa.py \
  -k 'test_indexer_score_ring4_fused and not bfp8 and not production and glm5 and contiguous' \
  --maxfail=1 -s
PASS (1 passed; 109/109 JIT cache hits)

git diff --check
PASS
```

First post-fix Opus re-review confirmed the rotated ownership blocker was fully
closed on both program creation and cache-hit override, but held the gate on an
axis regression: the new public range check also rejected a singleton SP axis.
Existing 1x4 model layouts intentionally use `cluster_axis=0` with ring size 1.
The indexer check now validates only that the integer axis is in range, restoring
the previous behavior. The release build and identical safe fused axis regression
passed again after this change.

A second Opus confirmation found the same accumulated-diff regression in the
earlier `ring_mla` public validator. Its integer-axis check now likewise validates
only that the axis is in range. This restores the pre-change singleton-axis
behavior consistently for both fused operations while keeping explicit `None` as
the only complete-mesh opt-in.

Validation after restoring both singleton-axis contracts:

```text
./build_metal.sh --release
PASS

scripts/run_safe_pytest.sh tests/nightly/blackhole/sdpa/test_ring_joint_sdpa.py \
  -k test_ring_mla_nd_sharded_indexed_kv_cache_accuracy --maxfail=1 -s
PASS (1 passed, PCC 0.999823)

scripts/run_safe_pytest.sh \
  tests/ttnn/nightly/unit_tests/operations/experimental/test_ring_indexer_score_dsa.py \
  -k 'test_indexer_score_ring4_fused and not bfp8 and not production and glm5 and contiguous' \
  --maxfail=1 -s
PASS (1 passed)

git diff --check
PASS
```

Final Claude Opus Step 5 review result: **PASS; no blocking findings;
Step 6 may begin**. Opus rechecked both restored range-only integer-axis
validators, verified that explicit `None` remains the sole full-mesh opt-in,
confirmed no mapping state leaks into axis JIT variants, and revalidated the
rotation-exact block-cyclic fix plus the transport/tensor rank split.

Step 5 gate status: **complete**.

## Step 6: `ring_indexer_score_dsa` full-mesh tests

Coverage added:

- Complete 2x4 LoudBox contiguous and block-cyclic tests using every physical
  device as one eight-rank snake. Both compare causal scores to an independent
  row-major reference, replay bit-exactly, preserve program-cache entry counts,
  and inspect every remote persistent-K slot for canonical tensor-rank placement.
- The block-cyclic case enters global slab 1 and then adds
  `chunk_local + 32`, forcing a nonzero owner rotation and one boundary rank to
  straddle. This is the runtime regression for the rotation-exact causal-geometry
  bug found by the Step 5 Opus review, including nonzero `boundary_slab`.
- An indexed three-slot, ND-sharded `k_local` test changes selected slot,
  `kv_len`, and causal start on cache hits. It checks PCC, bit-exact replay,
  stable cache entries, slab-rounded transport bounds, and selected-slot remote
  placement before and after shrinking the valid extent.
- Host rejection coverage for non-Ring topology, `seq_subshard_axis`, a named
  `block_cyclic_sp_axis`, zero or unavailable links, axis-only Q placement, and
  a non-replicated persistent K buffer.
- Exact physical 2x2 QuietBox variants for contiguous and rotated block-cyclic
  layouts in the four-device suite. These deliberately skip on larger physical
  systems instead of opening a non-closed torus subset.
- An opt-in complete 8x4 Galaxy case at the fixed 32-rank readiness-table limit,
  gated by the simulator or `TT_METAL_RING_INDEXER_RUN_32_RANK_ACCURACY=1`.

Validation on the complete physical 2x4 Blackhole host:

```text
scripts/run_safe_pytest.sh \
  tests/ttnn/nightly/unit_tests/operations/experimental/test_ring_indexer_score_dsa.py \
  -k full_mesh --maxfail=1 -s
PASS (4 passed, 1 Galaxy-gated skip)

scripts/run_safe_pytest.sh \
  tests/ttnn/nightly/unit_tests/operations/experimental/test_ring_indexer_score_dsa_4d.py \
  -k full_mesh --maxfail=1 -s
PASS (2 exact-physical-2x2 cases skipped on this eight-device host)

./build_metal.sh --release
PASS

python -m py_compile <both changed indexer test files>
PASS

git diff --check
PASS
```

The runnable 2x4 row snake has a non-identity transport-to-tensor permutation,
so the remote-slot assertions exercise the mapping rather than merely an identity
case. Exact physical 2x2 and Galaxy 8x4 remain hardware-gated for final sign-off.

Initial Claude Opus Step 6 review result: **PASS; no blocking findings;
Step 7 may begin**. Before closing the gate, three non-blocking hardening items
were applied: the rotated case now also exercises `boundary_slab > 0`, the
cache-hit shrink explicitly proves slab 2 remains byte-identical while slab 1
changes users, and simulator mode can now reach the 8x4 gate without also setting
`MESH_DEVICE=TG`. The module description was updated for the new scope.

The consolidated main full-mesh group passed again after these refinements
(**4 passed, 1 Galaxy-gated skip**); `py_compile` and `git diff --check` also
pass.

Final post-hardening Claude Opus Step 6 review result: **PASS; no blocking
findings; Step 7 may begin**. Opus independently recomputed the nonzero-slab
rotation/straddle geometry, verified the cache-hit snapshot cannot pass
vacuously, and confirmed the simulator/hardware gate logic. Its final minor
suggestion was applied: the Galaxy test now skips unless exactly 32 devices are
available.

Step 6 gate status: **complete**.

## Step 7: complete axis-ring regression

The existing axis-ring behavior was revalidated after both full-mesh paths and
the shared snake/rank-mapping infrastructure were in place. All test commands
used the repository's safe pytest wrapper.

```text
scripts/run_safe_pytest.sh \
  tests/ttnn/nightly/unit_tests/operations/experimental/test_ring_indexer_score_dsa.py \
  -k 'not full_mesh' --maxfail=1 -s
PASS (19 passed, 5 deselected)

scripts/run_safe_pytest.sh tests/nightly/blackhole/sdpa/test_ring_joint_sdpa.py \
  -k '(test_ring_mla_chunked_nd_sharded_indexed_kv_cache_accuracy_and_determinism or test_ring_mla_chunked_kv_actual_isl_indexed_reuse_max_accuracy_and_determinism or test_ring_mla_metadata_matches_scalar_indexed or test_ring_mla_metadata_matches_scalar_rotation) and not full_mesh' \
  --maxfail=1 -s
PASS (7 passed, 152 deselected)

scripts/run_safe_pytest.sh \
  tests/ttnn/unit_tests/operations/experimental/test_high_bw_all_gather.py \
  --maxfail=1 -s
PASS (8 passed, 14 expected hardware/opt-in skips)

git diff --check
PASS
```

The indexer group covers contiguous and block-cyclic layouts, BF16 and BFP8 K,
production shapes, indexed and ND-sharded bounded gather, program-cache reuse,
runtime KV length, boundary straddling, and rejection of unsupported head
streaming. The MLA group covers chunked indexed-cache reuse/determinism and
bit-exact metadata-versus-scalar equivalence for indexed slots and several
rotation positions. The high-bandwidth all-gather group exercises the shared
route planner over 1D line, 1D ring, 2D, and available torus fabric variants.

Claude Opus completed the accumulated Step 7 regression review after 17 minutes
33 seconds and reported **STEP 7 PASS** with no blocking findings. It traced the
transport-rank/tensor-rank uses in both program factories, axis-mode identity
mapping, route closure validation, cache-key and runtime-override behavior, and
compile-time argument bounds. It recorded low-severity hardening/follow-up items
for duplicated placement validation, validation ordering, sharing the indexer's
32-rank constant, rejecting otherwise-unreachable standalone all-gather
full-mesh attributes, replacing the zero route-hash sentinel, and caching
control-plane route resolution. It also recorded the unavailable exact-2x2 and
Galaxy-32 execution coverage as hardware-gated rather than correctness defects.

Review command:

```text
claude --dangerously-skip-permissions --model opus \
  'Step 7 regression review for the full-mesh ring_mla and ring_indexer_score_dsa implementation ...'
```

Step 7 gate status: **complete**.

## Step 8 scope boundary: model integration

The reusable operation support does not enable full-mesh mode in the DeepSeek
model. As specified in Phase 6 of the plan, model adoption requires a separate
model-layout commit: full-mesh sequence mappers, cache writers, redistribution
of tensors currently TP-sharded on the second axis, full-mesh semaphores,
updated logical-length/rank derivation, and output composition. It must remain
behind an explicit configuration switch and cannot be qualified on this
eight-device host because the required 32-device Galaxy accuracy, memory, and
performance evidence is unavailable. No unqualified model switch is added in
this operation-support change.

Final publication validation: `./build_metal.sh --release` completed
successfully, `git diff --check` passed, and recursive submodule status showed
all submodules at the rebased commits with no local divergence.

The commit hook required the new MLA negative test to use the repository
`expect_error` fixture instead of `pytest.raises`. The focused safe-wrapper run
passed (1 passed, 158 deselected). A patient Claude Opus follow-up reviewed the
fixture implementation, all three exception types and message patterns, and the
runtime cleanup path, then reported **STEP 7 FOLLOW-UP PASS**.

## Step 8: opt-in model integration

Model adoption is implemented behind `PREFILL_FULL_MESH_RING=1`; the default
axis-ring layout and behavior remain unchanged. The option is propagated from
the prefill adapters through transformer/block construction into `ttMLA` and is
rejected for cache migration, which still assumes SP-axis shards with TP
replicas. Full-mesh mode explicitly requires the canonical `sp_axis=0,
tp_axis=1` layout, a legal 2D mesh of at most 32 devices with at least one even
dimension, and fixed chunk-aligned starts.

Dense `ring_mla` integration:

- The model boundary remains SP-by-TP. Q is redistributed with a TP
  all-to-all from TP-head shards to additional sequence shards; KV is partitioned
  across TP. Both are stamped with an explicit row-major
  `Shard(2), Shard(2)` topology before entering the full-mesh ring.
- The persistent MLA cache is sequence-sharded across every physical mesh
  coordinate. `update_padded_kv_cache(cluster_axis=None)` preserves that exact
  topology and derives the cache rank from the declared tensor coordinates.
- After `ring_mla`, the inverse TP all-to-all restores the existing TP-head
  layout before `wkv_b2`, so downstream model boundaries do not change.

Sparse `ring_indexer_score_dsa` integration:

- Sparse MLA attention and its primary KV cache stay on the SP axis. Only the
  indexer's Q, per-row weights, K update, index cache, and fused indexer ring are
  redistributed over the complete mesh.
- Full-mesh indexer mode uses `cluster_axis=None`, no named sequence-subshard or
  block-cyclic SP axis, and a per-device block-cyclic chunk of
  `chunk_size / tp_factor`. Top-k results are gathered back across TP before the
  unchanged sparse-attention path consumes them.
- Full-mesh sequence caches compose in canonical row-major device order; axis
  composition now asserts its existing `sp_axis=0` assumption.

The current model integration deliberately accepts only fixed, chunk-aligned
starts. Although the operations support arbitrary rotated starts, an SP-rotated
model input cannot in general be evenly re-partitioned across TP into the
canonical full-mesh rotated chunks without another rotation-aware exchange.
Both the runtime and MLA (including KV-only scalar calls) fail early instead of
silently producing a different token ownership. The device-metadata path relies
on the runtime's host-side alignment check because its start remains on-device.

Focused model and cache coverage is box-adaptive: 4-device hosts use 2x2,
8-device hosts use 2x4, and 32-device hosts use 8x4. Unsupported or one-dimensional
systems skip instead of constructing an illegal full-mesh ring. Direct cache-op
coverage checks a nonzero rotated start through both scalar and metadata
signatures and verifies that an axis/replicated cache topology is rejected.

Validation on the complete physical 2x4 Blackhole host:

```text
./build_metal.sh --release
PASS

scripts/run_safe_pytest.sh \
  models/demos/deepseek_v3_d_p/tests/test_mla.py::test_mla_chunked_prefill_full_mesh_model_integration
PASS (2 passed: scalar and metadata; output PCC 0.998758/0.998755;
      k_nope PCC 0.999878, k_pe PCC 0.999884)

scripts/run_safe_pytest.sh \
  models/demos/deepseek_v3_d_p/tests/sparse_mla/test_sparse_mla.py::test_sparse_mla_chunked_full_mesh_indexer_model_integration
PASS (1 passed: five 1K chunks; output PCC 0.996701;
      primary cache PCC 0.999909, index cache PCC 0.999881)

scripts/run_safe_pytest.sh \
  models/demos/deepseek_v3_d_p/tests/op_unit_tests/test_deepseek_prefill_update_padded_kv_cache.py \
  -k 'full_mesh_rotated or full_mesh_rejects_axis_topology'
PASS (3 passed: scalar rotated, metadata rotated, invalid topology rejection)

scripts/run_safe_pytest.sh \
  models/demos/deepseek_v3_d_p/tests/test_mla.py::test_mla_chunked_prefill \
  -k 'aligned_min and cpu and 2x4 and fabric2d and dsv3 and scalar'
PASS (1 passed; rotated axis-mode output PCC 0.998219; cache PCC >0.99987)

scripts/run_safe_pytest.sh \
  models/demos/deepseek_v3_d_p/tests/sparse_mla/test_sparse_mla.py::test_sparse_mla_chunked \
  -k 'glm_5_1 and 2x4 and kv_bf16 and c1k'
PASS (1 passed; default sparse output PCC 0.996701)

scripts/run_safe_pytest.sh models/demos/deepseek_v3_d_p/tests/test_sparse_kv_cache_contract.py
PASS (15 passed)

Black 23.10.1, clang-format 19.1.4, compileall, and git diff --check
PASS
```

The model harness is an accuracy/contract test, not a performance benchmark.
On the warm 2x4 run, dense MLA signposts were about 18 ms (scalar) and 7 ms
(metadata) after first-use compilation; sparse steady-state MLA signposts were
about 29-31 ms. The scalar dense first-use signpost was about 226 ms, while a
fresh metadata fixture incurred about 6.5 seconds of JIT work. The 2x4 planner
selects the direct row snake and Blackhole requests two links, but route hashes
and per-model program-cache deltas are not exposed by this harness. No
performance improvement or Galaxy qualification is claimed; the option remains
experimental until a separate 8x4 memory/performance run records those values.

The initial Claude Opus Step 8 review reported two blockers: the sparse test's
anchor selected an illegal 1x4 full mesh on a four-device host, and the changed
Python/C++ files were not fully formatted. It also requested KV-only alignment
validation, explicit axis guards, and direct full-mesh cache-update coverage.
All five items were addressed before the final re-review.

Final Claude Opus Step 8 review result: **PASS; no blocking findings; Step 8
may be committed and force-pushed**. Opus independently checked row-major
ownership through both TP redistributions, RoPE ordering, the full-mesh indexer
parameter combination, cache-hit topology validation and program hashing,
in-place cache topology preservation, default axis equivalence, sparse primary
cache isolation, box-adaptive tests, and the documented qualification boundary.

One non-blocking review finding was folded in before publication: the block's
test/debug `return_kv_cache` path now passes the MLA SP axis and dense-only
full-mesh predicate to `kv_cache_to_host`. Dense full-mesh therefore composes
every canonical shard, while sparse full-mesh indexer mode continues composing
one TP replica of its SP-axis primary cache. A targeted patient Opus follow-up
reviewed dense, sparse, and KV-only control flow and returned **PASS; Step 8
remains safe to commit and force-push**.

Review commands used the required form:

```text
claude --dangerously-skip-permissions --model opus 'Final Step 8 re-review ...'
claude --dangerously-skip-permissions --model opus 'Targeted post-PASS follow-up review ...'
```

Step 8 gate status: **complete**.

## Post-rebase integration gate

The completed branch was rebased again onto `origin/main` commit
`856217be915`. This mainline update overlapped the implementation in two
important places: high-bandwidth all-gather gained runtime-selected input-slot
and gathered-prefix controls, and MLA/indexer gained `active_seq_len`, Kimi K3
output-gate support, and additional high-bandwidth gathers. The conflict
resolution preserves both sets of behavior. The high-bandwidth gather kernels
use the upstream runtime page geometry and fixed worst-case output slots while
appending the full-mesh snake mapping arguments consistently in the host
factory, reader, writer, and shared iterator.

A final cache-key audit found that the upstream custom
`compute_program_hash` did not explicitly include the newly added full-mesh
mode, orientation, resolved link count, or mesh dimensions. Those five stable
structural values are now hashed. Runtime slot and valid-prefix *values* remain
excluded intentionally while their presence is hashed and their bounds are
revalidated on cache hits.

Post-rebase validation on the complete physical 2x4 Blackhole host:

```text
./build_metal.sh --release
PASS

scripts/run_safe_pytest.sh \
  tests/ttnn/unit_tests/operations/experimental/test_high_bw_all_gather.py \
  --maxfail=1 -s
PASS (11 passed, 14 expected hardware/opt-in skips)

scripts/run_safe_pytest.sh \
  models/demos/deepseek_v3_d_p/tests/test_mla.py::test_mla_chunked_prefill_full_mesh_model_integration \
  models/demos/deepseek_v3_d_p/tests/sparse_mla/test_sparse_mla.py::test_sparse_mla_chunked_full_mesh_indexer_model_integration \
  --maxfail=1 -s
PASS (3 passed: dense scalar, dense metadata, sparse; dense output PCC
      0.998773, sparse output PCC 0.996701)

scripts/run_safe_pytest.sh \
  models/demos/deepseek_v3_d_p/tests/op_unit_tests/test_deepseek_prefill_update_padded_kv_cache.py::test_update_padded_kv_cache_full_mesh_rotated \
  models/demos/deepseek_v3_d_p/tests/op_unit_tests/test_deepseek_prefill_update_padded_kv_cache.py::test_update_padded_kv_cache_full_mesh_rejects_axis_topology \
  --maxfail=1 -s
PASS (3 passed)

scripts/run_safe_pytest.sh \
  models/demos/deepseek_v3_d_p/tests/test_mla.py::test_mla_chunked_prefill \
  -k 'aligned_min and cpu and 2x4 and fabric2d and dsv3 and scalar and no_determinism' \
  --maxfail=1 -s
PASS (1 passed; rotated axis-mode output PCC 0.998219)

scripts/run_safe_pytest.sh \
  models/demos/deepseek_v3_d_p/tests/sparse_mla/test_sparse_mla.py::test_sparse_mla_chunked \
  -k 'glm_5_1 and 2x4 and kv_bf16 and c1k' --maxfail=1 -s
PASS (1 passed; default sparse output PCC 0.996701)

scripts/run_safe_pytest.sh \
  models/demos/deepseek_v3_d_p/tests/test_sparse_kv_cache_contract.py \
  --maxfail=1 -s
PASS (15 passed)
```

The required patient post-rebase review used:

```text
claude --dangerously-skip-permissions --model opus \
  'Act as the final post-rebase correctness reviewer ...'
```

Claude Opus reviewed for about fifteen minutes and returned **PASS** with no
blocking correctness findings. It independently traced every merged
compile/runtime argument index, runtime page geometry and cache reuse,
transport-rank versus tensor-rank ownership, direct-neighbor topology proof,
cache-update topology preservation, active-sequence and Kimi output-gate
coexistence, dense/sparse layout transitions, and default axis behavior. It
confirmed that the five-field program-hash correction closes the remaining
structural cache-key gaps.

The principal qualification limitation remains unchanged: real 32-rank 8x4
Galaxy execution is unavailable on this host. The route is host-validated and
the option stays opt-in, but an 8x4 accuracy/memory/performance run under the
appropriate torus fabric remains required before production rollout.

Post-rebase integration gate status: **complete**.
