# Prefill Runtime

`models/common/llm_runtime/prefill` implements the common host-side prefill
pipeline for TTTv2 language models. It plans logical requests, stages TT
inputs, runs eager or traced prefill, postprocesses device outputs, restores
the caller's row order, and releases invocation resources.

This document describes the internals of the prefill package. See the
[common LLM runtime README](../README.md) for model-executor composition,
program and trace registries, paged-KV allocation, warmup, serving lifecycle,
and cleanup order.

## Scope

The package owns:

- immutable prefill request and chunk planning;
- regular-single, regular-batched, and chunked invocation geometry;
- eager program and trace identity construction;
- host and device input staging;
- trace capture inputs, replay refresh, and replay ownership;
- prefill postprocessing and device sampling state;
- streaming synchronized result collection; and
- retryable cleanup of prefill invocation transients.

The package does not choose a model, allocate the paged KV cache, select eager
versus traced execution, own the program or trace registries, or adapt an
inference-server API. Those decisions remain with the model executor and the
common runtime components above this package.

## Composition and construction

A model executor resolves one immutable config and constructs one prefill
runtime per execution lane. `PrefillRuntime` then composes the focused package
collaborators around the exact model, mesh, output reader, and page-table
layout held by that config:

```text
model + mesh + OutputReader + PageTableLayout + static prefill policy
  │
  └─> PrefillRuntimeConfig.resolve(...)
        │
        └─> PrefillRuntime(config)
              │
              ├─> PrefillInputStager(model, mesh)
              │
              ├─> PrefillPostprocessor(config)
              │
              ├─> PrefillResultAssembler(
              │     config,
              │     postprocessor,
              │     transient-release callback)
              │
              ├─> PrefillSequenceRunner(
              │     input stager,
              │     postprocessor,
              │     model-body callbacks,
              │     transient-release callback)
              │
              └─> PrefillTraceLifecycle(
                    input stager,
                    postprocessor,
                    model-body callbacks,
                    transient-release callback)
```

The collaborators share identities rather than constructing replacements:

- the input stager, postprocessor, sequence runner, and trace lifecycle use the
  exact objects composed by `PrefillRuntime`;
- eager and traced paths call the same runtime-owned model-body boundaries;
- eager invocation and trace replay finish through the same postprocessor;
- eager and traced results enter the same streaming result collector; and
- all transient release callbacks return to the runtime's retryable orphan
  owner.

`configure_page_table_layout(...)` is the one intentional late configuration
step. Before allocation or execution, it installs a bounded immutable config
replacement and forwards that exact replacement to config-dependent
collaborators. It does not reconstruct the runtime or its execution helpers.

## Data flow

One public prefill call follows four phases:

```text
host request
  -> plan and classify                  [runtime.py, plan.py, signatures.py,
                                         postprocess.py]
  -> execute each prepared request      [sequence_runner.py or trace.py,
                                         inputs.py, postprocess.py]
  -> collect each invocation result     [result_collector.py]
  -> public logits or sampled tokens
```

`PrefillRuntime` is the package facade. Callers use its public methods rather
than invoking the planner, input stager, sequence runner, postprocessor, or
result collector directly.

| Phase | Primary participants |
| --- | --- |
| Plan and classify | `PrefillRuntime.prepare()` coordinates `_plan_prefill_requests()`, `PrefillPostprocessor.classify_sampling_path()`, and the pure signature builders. |
| Stage inputs | `PrefillInputStager` prepares host values, allocates eager device inputs, or refreshes persistent trace inputs. |
| Execute eager | `PrefillSequenceRunner` stages and runs each `PrefillChunk`, releasing non-final outputs as it advances. |
| Execute traced | `PrefillTraceLifecycle` defines capture, refreshes each replay step, and records replay ownership. |
| Postprocess | `PrefillPostprocessor` turns the final hidden output into logits or device-sampled result values. |
| Collect | `PrefillResultAssembler` synchronizes readback, restores source rows, aggregates outputs, and releases invocation ownership. |

### 1. Plan and classify once

`PrefillRuntime.prepare(...)` validates the request and produces one or more
immutable `PreparedPrefill` values:

```text
tokens + page table + lengths + cached positions + slots
  -> PrefillRequest
       -> one or more PrefillChunk values
  -> sampling path
  -> eager program signatures
  -> optional trace signature
  -> PreparedPrefill
```

The prepared value is the shared input to eager invocation, compilation,
trace capture, and traced dispatch. Those paths do not re-plan the request or
reclassify sampling.

The central immutable values are:

| Value | Meaning |
| --- | --- |
| `PrefillRequest` | Logical rows, physical geometry, page tables, cached offsets, and ordered chunks for one invocation sequence |
| `PrefillChunk` | One model-body step, including its token slice, absolute position, page-table view, and final-token marker |
| `PreparedPrefill` | One request plus its sampling classification and program/trace identities |
| `InvocationResult` | A result value together with the transient resources that remain owned until collection |

### 2. Execute eager or traced prefill

Eager execution runs the complete prepared sequence through
`PrefillSequenceRunner`:

```text
PreparedPrefill
  -> stage one chunk
  -> run the regular or chunked model body
  -> release a non-final chunk output
  -> repeat until the final-token chunk
  -> finish prefill postprocessing
  -> InvocationResult
```

A regular request is the one-step case. A cached or long request can contain
multiple chunks, but it retains one ordered request-level sequence and one
final result.

Traced execution captures a sampling-free hidden body and keeps
program-alias-local postprocessing state separate:

```text
PreparedPrefill
  -> capture plan
       -> persistent model inputs
       -> hidden-body trace
       -> alias-local replay workspace
  -> refresh inputs for each replay step
  -> replay each planned chunk
  -> finish only the final hidden output
  -> InvocationResult
```

Cached offsets and chunk starts are refreshed inputs. A fixed chunk trace can
therefore serve multiple positions through a host-side replay loop without
changing trace identity.

### 3. Postprocess device outputs

`PrefillPostprocessor` owns the boundary between model-body output and the
result that will be read on the host. Its responsibilities include:

- selecting logits, forced-argmax, or top-k behavior;
- creating and refreshing K/P/T device values;
- selecting the logical last-token row or tile;
- allocating sampling and log-probability outputs; and
- preserving which tensors are borrowed, persistent, or invocation-owned.

Postprocessing does not restore public source-row order and does not perform
the final synchronized host read. Those belong to result collection.

### 4. Collect results in source-row order

`PrefillResultAssembler.assemble(...)` consumes prepared invocation results as
a stream. For each result it:

1. completes synchronized device-to-host readback;
2. normalizes logits or sampled tokens;
3. restores the original source-row mapping;
4. records per-row log probabilities when requested; and
5. releases that invocation's owned resources before advancing the stream.

Streaming is a correctness requirement for trace replay. Multiple replays can
reuse one persistent output buffer, so a result must be read before the caller
advances the generator and overwrites it.

## Planning invariants

### Logical rows and physical rows are distinct

`source_rows` preserves caller-visible order. `active_batch_size` records real
rows, while `padded_batch_size` records the physical wave. Compatible active
counts round up to a supported wave in `{1, 2, 4, 8, 16, 32}` without changing
the logical result shape.

Program and trace identities use padded geometry rather than active row count.
Requests with the same physical geometry can therefore share compiled and
captured work.

### Padding rows must not write KV

Regular paged-prefill tables use `-1` as the skip sentinel for unallocated
tails and padding rows. Token padding rows are zero. This makes unused physical
rows inert while retaining one shared padded program identity.

Chunked full-request page tables retain the nonnegative filler required by the
attention path. Fill-only chunk tables use `-1` beyond their mapped prefix.
These table roles are intentionally different.

### Cached and chunk positions stay absolute

Planning preserves the cached-token offset and each chunk's absolute start.
Last-token extraction is derived relative to the current invocation only at
the staging or postprocessing boundary. Cached prefill is not a separate
service or execution mode.

### Planning fallback is not execution fallback

The planner can represent an ineligible batch as sequential single requests.
Once a prepared request selects traced execution, missing program-to-trace
coverage is an error; the traced path does not silently reinvoke eager
execution.

## Identity boundaries

Eager programs and traces have related but different identities:

- `PrefillProgramSignature` includes every material program choice, including
  operation variant, padded invocation geometry, page-table widths, sampling
  path, and any material last-token tile.
- `PrefillTraceSignature` identifies the static hidden body and persistent
  input geometry. Sampling state is deliberately excluded.
- capture-schema and workspace fingerprints allow one hidden trace family to
  keep distinct postprocessing workspaces for its program aliases.

Identity builders are pure. They consume an already-planned request and do not
inspect mutable runtime state.

## Ownership and failures

The runtime borrows the model, mesh device, output reader, and bound paged-KV
context. Eager staging and postprocessing create invocation transients.
Successful invocation transfers those resources in `InvocationResult.owned`
to the result collector.

The sequence runner releases non-final chunk outputs immediately. On a primary
failure it attempts to release everything acquired so far and attaches cleanup
failures without replacing the primary error. Failed releases are retained by
`PrefillRuntime` as retryable tensor-resource orphans and are retried during
runtime cleanup.

Trace-owned persistent inputs and hidden outputs are not treated as eager
invocation transients. `PrefillReplayOwnership` records the boundary between
trace-owned, nested persistent, newly allocated, and replay-local values.

## Module map

| Module | Main responsibility |
| --- | --- |
| `config.py` | Resolve and validate static collaborators, capabilities, and geometry ceilings |
| `plan.py` | Build immutable logical requests, padded waves, page tables, and chunks |
| `signatures.py` | Classify prepared requests and construct pure eager/trace identities |
| `inputs.py` | Prepare host values, stage device inputs, and refresh replay inputs in place |
| `sequence_runner.py` | Run eager chunk sequences with failure-safe ownership transfer |
| `trace.py` | Define capture plans, persistent inputs, replay workspaces, refresh, and replay ownership |
| `postprocess.py` | Classify sampling and transform final hidden outputs into readable result values |
| `result_collector.py` | Read results synchronously, restore source rows, aggregate outputs, and release ownership |
| `sampling_helpers.py` | Provide stateless sampling normalization, slicing, and log-probability helpers |
| `runtime.py` | Expose the stable facade, compose collaborators, call model bodies, and retain cleanup orphans |

The intended dependency direction is from the facade toward focused
collaborators and from execution mechanics toward immutable plan/signature
values. Focused modules must not import the model executor, execution-target
selection, compiler registries, warmup coordination, or serving adapters.

## Tests

The main contract suite is
`models/common/tests/llm_runtime/test_prefill_runtime.py`. It covers planning,
padding, page-table semantics, signatures, eager sequences, trace refresh,
sampling, result collection, and cleanup failures.

`models/common/tests/llm_runtime/test_execution.py` covers the boundary between
prepared prefill values and eager/traced execution, including strict trace
coverage and streaming replay consumption. Warmup-specific prefill coverage is
tested in `models/common/tests/llm_runtime/test_warmup.py`.
