# KDA direct ND-sharded cache design

Date: 2026-09-10
Work: `tt-metal_tracker-s91`
Base: `origin/main` at `3f254861838`

## Problem

KDA prefill currently exposes recurrent and convolution carries in interleaved DRAM. The disaggregated-decode contract instead migrates recurrent `[128,32]` FP32 bands and convolution `[3,64]` BF16 rectangles. An earlier adapter experiment proved exact conversion, but even its optimized explicit boundary conversion costs up to 29 microseconds per KDA layer and requires duplicate native and contract buffers.

The goal is for `ttKDA.forward` to accept and return the migration-ready ND-sharded representation directly. Accuracy and state semantics must remain unchanged, and performance must be compared with the unmodified base.

Non-goals are changing the cache wire contract, decode execution, transport, non-KDA TTNN operations, checkpoint formats, or KDA activation layouts.

## Current system

- **Observed.** `KdaState` is caller-owned and immutable, but only documents logical shape, dtype, and SP/TP distribution (`models/demos/deepseek_v3_d_p/tt/kda/kda.py`).
- **Observed.** `allocate_state` places recurrent FP32 tiles and convolution BF16 rows in interleaved DRAM (`models/demos/deepseek_v3_d_p/tt/kda/kda.py`).
- **Observed.** production K3 uses grouped recurrence; SP execution computes its global final recurrent carry in `_distributed_affine_prefix`, while local grouped execution extracts the last group state (`models/demos/deepseek_v3_d_p/tt/kda/config.py`, `recurrence.py`).
- **Observed.** convolution state is consumed by `qkv_causal_conv1d_silu`; SP execution first exchanges projected tails and derives per-rank entry carries (`models/demos/deepseek_v3_d_p/tt/kda/kda.py`, `convolution.py`).
- **Observed.** recurrent scan and convolution validators reject sharded persistent inputs, even though their dataflow kernels use layout-aware `TensorAccessor` page access (`ttnn/cpp/ttnn/operations/experimental/kda/`).
- **Observed.** the supplied adapter report records bit-identical round trips for all required layouts and establishes exact shard geometry (`kda-cache-adapter-experiment-report (1).html`).

## Proposed design

### Canonical state layout

`KdaState` has one canonical physical representation on device:

| State | Logical per-device shape | Dtype/layout | ND shard | Distribution |
| --- | --- | --- | --- | --- |
| recurrent | `[1,Hlocal,128,128]` | FP32 tile | `[1,1,128,32]` | DRAM `ROUND_ROBIN_1D` |
| convolution | `[1,3,3*Hlocal*128]` | BF16 row-major | `[1,3,64]` | DRAM `ROUND_ROBIN_1D` |

The shard grid is the device DRAM-bank grid. Shapes and element order do not change. TP partitions heads/channels and SP replicates completed state exactly as today.

### Ownership and flow

```text
caller-owned ND KdaState
  recurrent ----> recurrence entry/prefix/scan ----> ND recurrent replacement
  convolution --> local/SP carry preparation -----> convolution kernel
                     projected global tail --------> ND convolution replacement

ttKDA.forward returns activation + caller-owned ND replacement state
```

The layer owns memory-config construction and validation. Recurrence and convolution code may use temporary L1/interleaved activation buffers where their algorithms require them, but there is no native persistent-state representation and no import/export step at the layer boundary.

### Recurrent boundary

- Direct scan reads the ND initial state through its existing tensor accessor.
- Scan activation output retains its existing memory config; final-state output has an independent state memory config.
- Grouped-local execution writes the selected last group state directly to the canonical ND config.
- Distributed-prefix execution keeps intermediate carry math in its existing working layout and writes the final update directly to ND on the last rank step.

### Convolution boundary

- Local execution passes ND history directly to `qkv_causal_conv1d_silu` and slices the projected final tail directly into ND.
- SP execution may materialize interleaved partition-entry carries required by halo exchange, but reads the caller's ND initial carry as an input and emits the global final tail directly into ND.
- The convolution kernel accepts interleaved or ND-sharded history without changing Q/K/V output layouts.

## Contracts

### Layer state input

Preconditions: batch is one; logical shapes, dtypes, layouts, device placement, exact ND shard shapes, DRAM bank grid, orientation, and round-robin distribution match the layer topology.

Guarantees: the input state is read-only; state values and TP/SP semantics are unchanged; invalid physical state fails before execution with a specific validation error.

### Layer state output

Guarantees: both replacement tensors use the exact canonical ND configs, are independent of input storage, and represent the same logical final state as the CPU reference. No explicit state adapter runs before return.

### KDA kernel inputs

Recurrent and convolution operations accept only the additional ND layouts required by the canonical state. Existing activation contracts and unsupported sharded variants remain rejected.

## Invariants and acceptance criteria

- Exact shard payloads remain 16,384 bytes for recurrent state and 384 bytes for convolution state.
- All three production layouts—SP1xTP8, SP2xTP4, SP4xTP2—return canonical ND state.
- Real Kimi-K3 `B=1,T=5120` output and both final states meet the existing acceptance threshold.
- Existing KDA operation and layer correctness tests pass through `scripts/run_safe_pytest.sh` with no required skips.
- Warm trace-wall layer performance is measured using the same production fixture and repetitions on base and implementation commits.
- No generic copy, collective, model, or non-KDA kernel behavior changes.

## Decisions and trade-offs

The user explicitly authorized autonomous design and implementation, so this spec records decisions without a separate approval pause.

- **Chosen:** make ND layout canonical rather than add a mode. This removes duplicate state representations and keeps one invariant.
- **Chosen:** separate recurrent activation-output and final-state memory configs. Reusing one config cannot express interleaved scan activations plus ND persistent state.
- **Chosen:** preserve temporary working layouts inside grouped/SP algorithms. Forcing every intermediate into the migration layout would expand scope and likely regress compute.
- **Rejected:** retain boundary adapters. They preserve old kernels but contradict the direct-state goal and retain extra buffers/latency.
- **Rejected:** change the transfer segment contract. The earlier experiment established that the approved contract fit current logical ordering exactly.

## Blast radius

- KDA Python: state config construction, validation, recurrence and convolution orchestration, layer tests/perf tests.
- KDA C++ and dataflow kernels: only validators/API/factory addressing needed to read/write the canonical state.
- Public experimental KDA operation signatures may gain a final-state memory-config argument with a backward-compatible default.
- No changes outside `models/demos/deepseek_v3_d_p/**/kda`, `tests/**/experimental/kda`, `ttnn/**/operations/experimental/kda`, specifications, and reports.

## Risks, assumptions, and unknowns

- **Assumed:** `TensorAccessor` maps the canonical ND tile/page IDs used by the existing recurrent reader/writer without address arithmetic changes; focused hardware tests will verify this.
- **Unknown:** mixed-layout concat/slice support in SP convolution may require a KDA-specific staging adjustment.
- **Risk:** program-cache keys must distinguish independent recurrent final-state layouts.
- **Risk:** direct ND writes can change kernel parallelism or DRAM access efficiency; trace-wall comparison is required even if accuracy passes.
- **Risk:** latest-main KDA or infrastructure behavior may differ from the older supplied report; only new base/branch runs are comparative evidence.

## Evidence

- Supplied investigation: `kda-cache-adapter-experiment-report (1).html`
- Layer state flow: `models/demos/deepseek_v3_d_p/tt/kda/kda.py`
- Recurrence flow: `models/demos/deepseek_v3_d_p/tt/kda/recurrence.py`
- Convolution SP flow: `models/demos/deepseek_v3_d_p/tt/kda/convolution.py`
- Recurrent device operation: `ttnn/cpp/ttnn/operations/experimental/kda/recurrent_chunk_scan/`
- Convolution device operation: `ttnn/cpp/ttnn/operations/experimental/kda/qkv_causal_conv1d_silu/`
