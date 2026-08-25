# KDA cache-adapter ablation implementation spec

Date: 2026-08-25
Bead: `tt-metal_tracker-858.3.9.1`
Base: PR7 `479d04f6e38f45b0290eff6bdf5c73abe5e1b4e9`
Status: completed and validated investigation

## Goal

Measure the correctness and performance cost of converting Kimi-K3 prefill state
between PR7's native interleaved-DRAM representation and the physical segment
layout fixed by `specs/k3_disagg_contract.md`.

Required coverage is real Kimi-K3, `B=1`, `T=5120`, on all three eight-device
layouts: `SP1xTP8`, `SP2xTP4`, and `SP4xTP2`.

Decode execution, migration wire transfer, speculative snapshot creation, and
production integration are out of scope. The investigation may probe direct
producer/consumer compatibility, but it must not change the approved cache
contract to make an existing operation pass.

## Design constraints

- **Required.** S transfer segments are `[128,32]` FP32 V-bands, 16,384 bytes.
- **Required.** Convolution transfer segments are `[3,64]` BF16 branch/head/half
  rectangles, 384 bytes.
- **Required.** Every segment is one contiguous DRAM shard, round-robin across
  the runtime DRAM-bank grid, following the existing KV-cache allocation pattern.
- **Required.** Export and import preserve values and logical tensor shapes
  exactly; no dtype conversion is permitted.
- **Required.** Steady-state measurements use preallocated stable outputs and
  exclude allocation and compilation. Cold eager cost is reported separately.
- **Required.** S-only, convolution-only, combined, export, import, and
  direct-layout probes are reported for every topology.
- **Required.** Hardware tests run through `scripts/run_safe_pytest.sh`; a skip is
  not a pass.

## Existing implementation

- **Existing.** `KdaState` contains recurrent and convolution tensors
  (`models/demos/deepseek_v3_d_p/tt/kda/kda.py:87`).
- **Existing.** Recurrent state is `[B,H_local,128,128]`, FP32 tile layout in
  interleaved DRAM; convolution is `[B,3,3*H_local*128]`, BF16 row-major in
  interleaved DRAM (`models/demos/deepseek_v3_d_p/tt/kda/kda.py:239`).
- **Existing.** TP partitions heads while SP ranks hold replicas of the completed
  state (`models/demos/deepseek_v3_d_p/tests/kda/utils.py:149`).
- **Existing.** DeepSeek KV migration allocations use an ND shard equal to one
  wire chunk and `ROUND_ROBIN_1D` DRAM distribution
  (`models/demos/deepseek_v3_d_p/utils/kv_cache_utils.py:1001`).
- **Existing.** `ttnn.to_memory_config` supports interleaved-to-ND and reverse
  conversions, including preallocated outputs
  (`tests/ttnn/unit_tests/base_functionality/test_to_memory_config.py:344`,
  `tests/ttnn/unit_tests/base_functionality/test_to_memory_config.py:2539`).

## Proposed changes

### Adapter module

Add a KDA test-support module that owns exactly two contract memory configs:

| State | Logical per-device shape | ND shard shape |
| --- | --- | --- |
| S | `[1,H_local,128,128]` | `[1,1,128,32]` |
| convolution | `[1,3,3*H_local*128]` | `[1,3,64]` |

No permutation is required. ND sharding gathers each rectangular logical slice
into one physical shard while retaining the source tensor's logical shape:

- S shard enumeration is head-major, then the four V-bands.
- Convolution width is already Q, K, V; within each branch it is head-major and
  each head has two consecutive 64-wide halves.

The module exposes typed functions to allocate contract buffers and to export or
import one `KdaState`. Callers supply preallocated destinations for steady-state
measurement. Each function validates shape, dtype, layout, memory configuration,
and returned destination identity.

### Correctness and performance test

Add one Blackhole-only test module under `models/demos/deepseek_v3_d_p/tests/kda/perf/`.
It will:

1. construct the real-weight `T=5120` layer and run one synchronized forward;
2. check the native output/state against the existing independent CPU oracle;
3. export and import patterned and real states;
4. require exact round-trip equality and the prescribed ND shard specs;
5. time export and import for S-only, convolution-only, and combined paths;
6. probe direct producer/consumer compatibility and record either measured
   latency or the exact unsupported contract/error;
7. print one stable JSON result record per topology.

The timing helper follows the existing layer performance harness: warm once,
capture one trace, execute repeated nonblocking replays, synchronize once per
sample, and report all samples plus min, median, p95, and max. Defaults are 20
sample groups and 100 replays per group, overridable by environment variables.

Cold eager timing covers the first synchronized adapter call into an already
allocated destination. Allocation time is measured independently.

## Data flow

```text
native PR7 state
  |-- S to_memory_config([1,1,128,32] ND DRAM) ----------|
  `-- conv to_memory_config([1,3,64] ND DRAM) -----------+--> contract state
                                                         |
native PR7 state <-- to_memory_config(interleaved DRAM) --'
```

Each SP replica converts locally. Results are reported per synchronized mesh
operation; they are not multiplied by SP size. The report separately states the
unique and physically replicated bytes for each topology.

## Implementation sequence

1. Add pure geometry and memory-config construction with host-only unit checks.
2. Add bidirectional preallocated adapters and patterned round-trip coverage.
3. Integrate real-layer correctness and the three-layout timing matrix.
4. Add direct-layout probes without weakening PR7 operation contracts.
5. Run safe hardware tests, retain raw JSON/profiler logs, and write the results
   report and recommendation.

## Validation

- Host collection proves all three mandatory node IDs exist and geometry yields
  exactly 384 S plus 576 convolution segments globally.
- Patterned state makes every branch, head, half, row, and band distinguishable.
- Export/import is bit-identical for both states.
- Contract buffers have the exact ND shard specs and DRAM buffer type.
- Real state retains the existing layer PCC threshold before timing.
- Trace wall samples and targeted profiler records are captured on Blackhole for
  all three layouts.
- Logs end in `SAFE_PYTEST_RESULT: PASS`; skips and partial matrices fail the
  investigation acceptance criteria.

## Risks and unknowns

- **Unknown.** FP32 tiled ND shards may have physical padding beyond 16,384 bytes;
  the first hardware test must inspect the resulting shard/page geometry.
- **Unknown.** A 384-byte row-major ND shard may be rounded physically. Contract
  registration remains invalid if the addressable payload is not exactly the
  required contiguous span.
- **Unknown.** Multi-device preallocated `to_memory_config` and trace runtime
  argument override may not support these DRAM ND shapes.
- **Unknown.** KDA recurrence and convolution producers may reject ND output or
  input memory configurations. Such rejection is a measured direct-layout result,
  not permission to alter the contract or silently omit the variant.
- **Risk.** Host trace-wall timing of a small adapter can be launch dominated;
  repeated trace replay and device-profiler attribution are both required.
