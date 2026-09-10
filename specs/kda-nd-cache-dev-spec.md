# KDA direct ND-sharded cache implementation plan

Date: 2026-09-10
Work: `tt-metal_tracker-s91`
Design: `specs/kda-nd-cache-design.md`

## Goal

Implement the design's canonical recurrent and convolution ND-DRAM state on latest main, preserving activation behavior and validating real Kimi-K3 accuracy and performance. Cache transport and decode are out of scope.

## Design constraints

- **Required:** exact recurrent `[1,1,128,32]` FP32 tile and convolution `[1,3,64]` BF16 row-major ND shards.
- **Required:** state remains TP-sharded and SP-replicated.
- **Required:** input state is immutable and output state is replacement storage.
- **Required:** no generic TTNN operation changes.
- **Required:** activation output memory configs remain unchanged.
- **Required:** base and implementation measurements use the same latest-main revision, fixture, topology matrix, and timing method.

## Existing implementation

- **Existing:** `ttKDA.allocate_state`, `_validate_forward`, `_convolve_qkv`, and `forward` own the state boundary (`models/demos/deepseek_v3_d_p/tt/kda/kda.py`).
- **Existing:** `_scan_chunks`, `_scan_grouped_chunks`, `_distributed_affine_prefix`, and `_last_group_state` determine recurrent input/final-state placement (`models/demos/deepseek_v3_d_p/tt/kda/recurrence.py`).
- **Existing:** `exchange_convolution_carry` determines SP convolution entry/final carries (`models/demos/deepseek_v3_d_p/tt/kda/convolution.py`).
- **Existing:** recurrent scan currently uses one output memory config for activation and final state and rejects sharded state (`ttnn/cpp/ttnn/operations/experimental/kda/recurrent_chunk_scan/`).
- **Existing:** QKV convolution rejects sharded history (`ttnn/cpp/ttnn/operations/experimental/kda/qkv_causal_conv1d_silu/`).

## Proposed changes

### State memory configuration

Add KDA-local helpers that construct the two exact memory configs from the runtime DRAM grid. `ttKDA` constructs them once, allocates state with them, and validates exact equality. Tests assert physical page/shard geometry.

### Recurrent state

Extend `recurrent_chunk_scan` with an independently keyed final-state memory config while retaining the existing default. Relax only initial-state validation to accept the exact supported ND form. Bind accessors from actual tensor specs so page mapping stays layout-aware.

Thread the canonical config through `KDARecurrence`. Direct scan requests ND final state. Grouped-local `_last_group_state` slices into ND. Distributed prefix emits its final add directly to ND while its entry carries and non-final iterations retain working memory.

### Convolution state

Allow QKV convolution history to be interleaved or exact ND DRAM. Pass local ND history directly. Parameterize SP carry exchange with the canonical output config so its global final slice writes ND; keep partition entry carry in the existing layout consumed by the kernel.

### Tests

- Host tests for exact geometry and forward validation failures.
- Focused recurrent operation tests for ND initial/final state, direct and trace replay/program-cache reuse.
- Focused convolution operation tests for ND history, direct and trace replay/program-cache reuse.
- Layer contract/stateful tests asserting canonical memory config and immutable replacement behavior.
- Existing real-weight acceptance and performance matrix on all required layouts.

## Control and data flow

1. Constructor derives topology-local heads/channels and canonical configs.
2. `allocate_state` creates zeroed ND buffers.
3. `forward` validates exact configs before projections.
4. Convolution reads ND history locally or through SP carry preparation; final projected tail is written to ND.
5. Recurrence consumes ND initial state in direct/grouped/SP flow; its final state is written to ND at the final producing operation.
6. `forward` returns only the activation and canonical replacement state.

Errors are raised at the layer or operation validation boundary before kernel launch. Existing defaults preserve standalone operation callers that do not request ND final state.

## Implementation sequence

1. Capture base correctness/performance evidence at `3f254861838`.
2. Add canonical configs, layer validation, and contract tests.
3. Implement recurrent ND input/output support and focused operation tests.
4. Implement convolution ND input/output flow and focused operation tests.
5. Build and run focused plus full accuracy suites.
6. Run the production performance matrix, compare against base, and investigate material regressions.
7. Commit a self-contained HTML report with exact evidence and residual risks.

Each validated concern is committed independently and pushed to `origin/momcilo/kda-nd-cache`.

## Validation

| Contract | Validation |
| --- | --- |
| Exact physical state | geometry/unit assertions plus device `memory_config`, aligned page size, and shard inspection |
| ND kernel access | focused operation accuracy, trace replay, and program-cache rebind tests |
| Layer semantics | existing CPU-reference output/recurrent/convolution PCC checks |
| Immutability | existing stateful replacement/trace tests with canonical layout assertions |
| Topology | SP1xTP8, SP2xTP4, SP4xTP2 real K3 matrix |
| Performance | synchronized warm trace-wall samples on base and implementation |
| Build validity | Release host build because C++ operation contracts/factories change |

All device tests run through `scripts/run_safe_pytest.sh`; only explicit `SAFE_PYTEST_RESULT: PASS` is accepted.

## Risks and unknowns

- **Unknown:** which latest-main SP primitives accept an ND initial convolution carry as a mixed-layout input; resolve with the smallest KDA-local staging necessary.
- **Risk:** changing recurrent output specs affects program caching; add distinct-layout cache tests.
- **Risk:** device access may be shared; rely exclusively on the serialized safe wrapper and never reset devices manually.
- **Risk:** the real checkpoint may be unavailable; verify the pinned fixture before claiming real-weight coverage.
