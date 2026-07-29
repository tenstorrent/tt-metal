# Kimi Delta Attention: Prefill-Only Design

## Status

Proposed. No source code has been changed.

## Decision summary

Make `KimiDeltaAttention` a prefill-only layer:

```python
def forward(self, hidden_states: ttnn.Tensor) -> ttnn.Tensor:
    ...
```

Remove the public `mode`, `chunk_size`, and `valid_len` arguments, remove the
dedicated `T=1` recurrent/decode implementation, and always execute the existing
chunked prefill algorithm.

Retain the recurrent matrix state and convolution carry. They are mathematical
KDA state, not a decode execution mode, and prefill needs them to compose
sequential prompt segments.

## Evidence and diagnosis

### Two meanings of "recurrent" are currently conflated

The layer has an execution-mode branch and persistent algorithmic state:

- `tt/layer.py:193-229` validates `"recurrent"` versus `"chunk"`.
- `tt/layer.py:354-492` branches between the `T=1` recurrent operation and the
  chunked prefill operation.
- `tt/layer.py:81-85`, `tt/layer.py:125-191`, and the state update at the end of
  `forward` hold matrix/convolution state across calls.

The first is removable. The second is required by prefill:

```text
prompt segment 0 ──> prefill KDA ──> output 0
                         │
                         ├── matrix state ─┐
                         └── conv carry ───┤
                                          v
prompt segment 1 ─────────────────> prefill KDA ──> output 1
```

Removing both would make segmented prefill mathematically different from
running the concatenated prompt.

### The recurrent execution path is isolated

Repository search found:

- `tt/recurrence.py:166-225`: `fused_kda_recurrence`, the only Python caller of
  `ttnn.transformer.kda_recurrent_step`.
- `tt/recurrence.py:23-112`: `composed_kda_recurrence`, with no callers.
- `tests/test_kda_recurrent.py:55`: dedicated operation correctness coverage.
- `tests/perf/test_kda_recurrent_perf.py:42`: dedicated operation performance
  coverage.
- `ttnn/cpp/ttnn/operations/transformer/transformer_nanobind.cpp:81`: public
  binding.
- `ttnn/cpp/ttnn/operations/transformer/sources.cmake:38-40,56` and
  `ttnn/cpp/ttnn/operations/transformer/CMakeLists.txt:22`: build registration.

No other runtime consumer of `kda_recurrent_step` was found in the repository.
All TP, SP, real-weight, and model-performance callers explicitly select
`mode="chunk"`; only `tests/test_ttnn_layer.py` exercises layer decode mode.

### `chunk_size` and `valid_len` are not real layer features

`tt/layer.py:221-224` only accepts `chunk_size=None|32` and only accepts
`valid_len=None|T`. The underlying fused operation enforces 32-token physical
chunks at
`ttnn/cpp/ttnn/operations/transformer/chunk_gated_delta_rule/chunk_gated_delta_rule.cpp:539`.
Keeping these arguments exposes choices that callers cannot make.

### The PyTorch recurrent reference must remain

`reference.py:kda_recurrent_reference` is the mathematical oracle used to
validate chunked prefill, including `tests/test_chunk_kda.py`. It is not a
runtime decode path. Deleting it would remove independent correctness evidence.

## Proposed changes

### 1. Narrow the layer API

In `tt/layer.py`:

- Change `forward(hidden_states, mode=..., chunk_size=..., valid_len=...)` to
  `forward(hidden_states)`.
- Remove `Literal` and the mode-specific validation.
- Always use the current chunk/prefill convolution, tensor layout, KDA, and
  collective branches.
- Preserve the existing state initialization, external-state API, state
  updates, and trace-stable copy behavior.
- Preserve the input-length domain already supported by the chunk path:
  positive `T` for single-device/TP, with the existing SP tile/group
  divisibility requirements.

This treats a one-token input as a one-token prefill executed by the prefill
algorithm; it does not retain or silently dispatch to a decode kernel.

### 2. Make the Python implementation vocabulary prefill-oriented

In `tt/recurrence.py`:

- Rename `chunk_kda_recurrence` to `kda_prefill`.
- Delete unused `composed_kda_recurrence`.
- Delete `fused_kda_recurrence`.

Keep `ttnn.transformer.chunk_kda` unchanged: "chunk" describes the internal
parallel algorithm and is accurate.

Keep names such as `recurrent_state` and `recurrent_state_dtype`: this is the
standard name for the KDA matrix state and changing it to "prefill state" would
misdescribe the mathematics and checkpoint/state interface.

### 3. Remove the dedicated public decode operation

Delete the `kda_recurrent` C++ operation, its nanobind registration, source/build
registration, and its dedicated correctness/performance tests.

This is an intentional API break for
`ttnn.transformer.kda_recurrent_step`. Repository evidence says it has no
remaining consumer after the layer branch is removed.

### 4. Convert tests from mode coverage to prefill contracts

In `tests/test_ttnn_layer.py`:

- Remove the mode parameter from helpers and calls.
- Remove the target-shape decode test.
- Replace prefill/decode continuity with **segmented-prefill continuity**:
  running prompt segments with preserved state must match one concatenated
  prefill within the existing PCC tolerance.
- Adapt external-state/trace-stability coverage to prefill-sized segments.

Update every TP, SP, real-weight, and performance caller to omit
`mode="chunk"`.

Retain `test_chunk_kda.py`, the reference recurrence, and affine-prefix tests;
they validate the prefill algorithm rather than decode support.

### 5. Update the contract and development documentation

Update `API_SPEC.md`:

- one prefill-only `forward` signature;
- no mode contract;
- state remains explicitly part of segmented prefill;
- SP remains prefill-only with its existing divisibility constraints.

Update stale recurrent/decode claims in package-facing documentation. Historical
bring-up-log entries remain historical and should not be rewritten.

## Alternatives considered

### A. Remove only the layer branch

Leave `kda_recurrent_step`, its kernels, bindings, and tests in TTNN.

Rejected: this leaves an unowned public operation and dead subsystem after the
only model consumer disappears. It does not satisfy "get rid of recurrent
path" coherently.

### B. Add `forward_prefill` but retain `forward(..., mode=...)`

Rejected: compatibility machinery has no in-repository consumer and preserves
the ambiguity the change is meant to remove. The package is experimental, so
this is the right point to make the API exact.

### C. Require every prefill length to be a multiple of 32

Rejected as the default: it is a larger behavioral restriction than removing
decode, and the existing chunk implementation supports short/non-tile inputs
outside SP by padding. SP keeps its current explicit divisibility constraints.

## Hardest-to-change decision

Deleting the public `ttnn.transformer.kda_recurrent_step` symbol is the most
expensive decision to reverse because out-of-repository callers may exist even
though repository search found none.

Recommendation: delete it now. Leaving it behind creates a public, tested API
with no owning model and makes "prefill-only" true only at the Python layer.

## Validation plan

Before committing:

1. Static proof:
   - `rg` shows no runtime `mode="recurrent"`, `fused_kda_recurrence`,
     `composed_kda_recurrence`, or `kda_recurrent_step`.
   - Remaining "recurrent" uses are state terminology or reference math.
2. Build because C++ sources/bindings change:
   - `./build_metal.sh`
3. Real-device correctness through the repository safety wrapper:
   - focused chunk-KDA and layer tests;
   - segmented-prefill state-continuity test;
   - TP tests;
   - SP tests, including SP4×TP2 and SP2×TP4;
   - real-weight K3 test when the checkpoint is locally available.
4. Performance smoke:
   - the existing 5K K3 prefill performance test for SP1×TP8, SP2×TP4, and
     SP4×TP2, checking that removing dispatch/API code does not change device-op
     composition or materially regress latency.
5. One small validated commit per concern:
   - layer/Python API and tests;
   - TTNN recurrent-op removal;
   - documentation/call-site cleanup.

Exact commands, PCC values, timings, and unavailable coverage will be reported
with each commit.

## Approval question

Approve the recommended full removal, including deletion of the public
`ttnn.transformer.kda_recurrent_step` operation?
