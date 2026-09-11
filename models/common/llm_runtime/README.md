# Common LLM Runtime

`models/common/llm_runtime` is a reusable execution toolkit for TTTv2 language
models. It owns mechanics that are common to autoregressive LLM inference:

- immutable runtime configuration and page-table geometry;
- prefill planning, including prefix caching and chunked prefill;
- decode input staging and output handling;
- eager program compilation and readiness checks;
- trace capture, input refresh, and replay;
- paged KV-cache allocation and ownership;
- warmup coverage;
- blocking and asynchronous output reads;
- data-parallel lane fanout; and
- deterministic cleanup of TT resources.

It deliberately does **not** contain a generic model executor. Shared model
composition lives one layer above at
[`models/common/models/executor.py`](../models/executor.py). A demonstrated
family may add a small policy facade in the model layer, while a model with
distinct orchestration may compose these runtime modules directly.

## Design principles

### The model owns orchestration

The common package supplies focused components, not a single common LLM
executor. A shared model-layer executor, family facade, or direct model-owned
executor decides:

- which tensor model is being executed;
- how model metadata becomes runtime config;
- which resources exist for one execution lane;
- whether eager and traced targets are constructed;
- when physical KV capacity becomes final;
- which execution target handles a call; and
- the cleanup order.

This keeps model-specific decisions out of common runtime mechanics while
allowing deliberate reuse at either the model-composition or individual
runtime-module boundary.

### Resolve static policy at construction

Static decisions are represented by frozen config values:

- `TraceConfig`
- `WarmupConfig`
- `PagedKVCacheConfig`
- `PageTableLayout`
- `PrefillRuntimeConfig`
- `DecodeRuntimeConfig`
- `WarmupCoordinatorConfig`
- `VLLMAdapterConfig`

`resolve(...)` methods validate collaborators and derive static capabilities
once. Runtime objects then read `self.config` instead of repeatedly deriving
policy in forward calls.

The one intentional late configuration step is physical paged-KV capacity.
An inference server may know a maximum at model construction but choose the
physical block count later. In that case, the runtime replaces small immutable
config/layout values before allocation; it does not reconstruct the model or
runtime owners.

### Composition instead of an execution hierarchy

There is one eager execution object:

```text
EagerExecutor
├── PrefillRuntime
├── DecodeRuntime
└── ProgramCompiler
```

When tracing is configured, there is one traced execution object over that
exact eager instance:

```text
TracedExecutor
├── eager_executor ──> exact EagerExecutor above
└── TraceCompiler ───> exact ProgramCompiler above
```

`TracedExecutor` does not inherit from `EagerExecutor`, and it does not choose
an eager fallback. Trace mode selects the operation policy: `none` is eager,
`decode_only` intentionally leaves prefill eager, and `all` selects traced
prefill and decode. Once an operation is selected for tracing, its complete
prepared call must pass coverage preflight; a miss raises `TraceCoverageError`
before device work instead of silently changing that call to eager execution.

### Programs and traces have separate registries

`ProgramCompiler` is the sole registry of eager program signatures and output
contracts. `TraceCompiler` maps an already-compiled program to a separate trace
signature and trace artifact.

Trace artifacts are never folded back into compiled-program records. This
separation makes the lifecycle explicit:

1. compile all required eager programs;
2. register trace capture plans for eligible programs;
3. capture all registered traces;
4. close the compile gate while traces are active;
5. replay through the trace registry; and
6. release trace resources before terminalizing the program registry.

### Plan prefill semantics once

Prefix caching and chunking are represented once in immutable
`PrefillRequest` and `PrefillChunk` values. Planning decides:

- source-row and slot mapping;
- active and padded batch sizes;
- actual and padded sequence lengths;
- cached-token offsets;
- canonical page-table width;
- regular single, regular batched, or chunked execution;
- chunk token slices and absolute chunk positions; and
- which chunk contains the final token.

`PrefillRuntime` consumes that plan. There is no separate cached-prefill
service and no separate chunked-prefill service.

### Pad compatible prefill rows into one physical wave

Batched prefill is an explicit per-model capability, bounded by the model's
configured maximum physical prefill batch and lane capacity. Within each lane,
the planner buckets rows by padded sequence length. An eligible bucket is kept
whole and its active row count is rounded up to the next supported physical
batch size in `{1, 2, 4, 8, 16, 32}`; for example, 15 active rows use one
padded-16 wave rather than being split.

A bucket falls back to sequential single-row planning when batching is disabled
or unsupported, the padded batch exceeds either configured limit, any row has
cached tokens, the padded sequence exceeds the regular prefill chunk size, or
`padded_batch_size * padded_sequence_length` is greater than or equal to
`128 * 1024`. Supplying sampling parameters also selects sequential planning
when the model cannot extract batched prefill outputs. `DISABLE_BATCHED_PREFILL`
is a diagnostic off switch. These are planning choices, not executor fallbacks.

Logical activity remains in `source_rows`, slots, and last-token metadata while
device tensors use padded geometry. Padding token rows are zero. Regular page
tables copy only each active prompt's allocated prefix; active tails and every
padding row remain `-1`, the skip sentinel, so padded rows do not write KV.
Chunked full-request page tables retain their SDPA-safe nonnegative filler,
while fill-only chunk tables use `-1` after the mapped prefix.

Program and trace identities intentionally omit `active_batch_size`; they use
the padded batch, sequence and page geometry, operation variant, and sampling
material. Eager regular batching fills active rows only. Trace capture fills
all physical rows, relying on `-1` page-table rows as no-ops, and replay
refreshes the complete padded inputs. Postprocessing pads extraction indices
separately and returns only active results in original source-row order.

### Make ownership explicit

Borrowed collaborators and owned TT resources are treated differently:

- the runtimes borrow the model, mesh, output reader, and bound KV context;
- `PagedKVCacheManager` exclusively owns physical KV tensors;
- `ProgramCompiler` owns its program registry and any retryable compile-output
  orphan;
- `TraceCompiler` owns trace IDs, persistent trace inputs, and trace outputs;
- `OutputReader` owns pending host destinations and completion events;
- prefill/decode runtimes own invocation transients and external decode leases;
  and
- the model-owned executor is the cleanup root.

Failed TT tensor deallocations are retained as `TensorResourceOrphan` values so
cleanup can retry without double-deallocating resources that were already
released.

## Module map

For the package's internal planning, eager-sequence, trace, postprocessing, and
ownership contracts, see the [prefill runtime README](prefill/README.md).

| Module | Main responsibility |
| --- | --- |
| `config.py` | Trace policy, warmup policy, paged-KV policy, and canonical page-table geometry |
| `prefill/config.py` | Fully resolved static prefill collaborators, capabilities, and geometry ceilings |
| `prefill/plan.py` | Pure host-side construction of `PrefillRequest` and `PrefillChunk` values |
| `prefill/signatures.py` | Prepared requests plus pure eager-program and trace identity construction |
| `prefill/inputs.py` | Host/device input values, staging, in-place replay refresh, and rotary handling |
| `prefill/trace.py` | Trace capture plans, mutable replay state, input refresh, and replay ownership |
| `prefill/postprocess.py` | Sampling classification, K/P/T state, output extraction, and device sampling |
| `prefill/result_collector.py` | Streaming synchronized readback, source-row restoration, and result release |
| `prefill/sequence_runner.py` | Run eager prefill sequence chunks with failure-safe ownership transfer |
| `prefill/runtime.py` | Stable public facade, collaborator composition, model-body calls, and transient cleanup |
| `prefill/sampling_helpers.py` | Stateless prefill sampling parameter and log-probability helpers |
| `decode.py` | Decode preparation, signatures, eager invocation, trace refresh, output leases, and cleanup |
| `execution.py` | `EagerExecutor` and `TracedExecutor` composition |
| `program_compiler.py` | Eager program identity, compilation, output contracts, and compile gate |
| `trace_compiler.py` | Trace registration, capture, refresh decisions, replay, and trace cleanup |
| `warmup.py` | Resolved warmup coverage and the prefill/decode capture barrier |
| `paged_kv_cache.py` | Exclusive physical KV allocation, binding, validation, and release |
| `output_reader.py` | Blocking reads and retained asynchronous read/event lifecycle |
| `lane_group.py` | Data-parallel fanout over already-built lane executors |
| `tensor_resources.py` | Recursive best-effort tensor release and retryable cleanup failures |
| `vllm_adapter.py` | vLLM call normalization and legacy physical KV-shape validation |

## Composition and construction

A concrete model executor normally constructs one lane in this order:

```text
model + model runtime metadata
        │
        ├─> PagedKVCacheManager(model, PagedKVCacheConfig)
        │
        ├─> PageTableLayout.resolve(...)
        │
        ├─> OutputReader(mesh_device)
        │
        ├─> PrefillRuntimeConfig.resolve(...)
        │       └─> PrefillRuntime(config)
        │
        ├─> DecodeRuntimeConfig.resolve(...)
        │       └─> DecodeRuntime(config)
        │
        ├─> ProgramCompiler(mesh_device, bound_cache_context)
        │
        ├─> EagerExecutor(prefill, decode, program_compiler)
        │
        ├─> optional TraceCompiler(program_compiler)
        │       └─> optional TracedExecutor(exact eager, trace_compiler)
        │
        └─> WarmupCoordinatorConfig.resolve(...)
                └─> WarmupCoordinator(config, execution, callbacks)
```

Important identity constraints are checked during construction:

- prefill, decode, output reading, compilation, and tracing use the same mesh;
- prefill and decode share the same model, page-table layout, lane capacity,
  sampling policy, and geometry ceilings;
- `TraceCompiler` composes the same `ProgramCompiler` used by eager execution;
  and
- `TracedExecutor` composes the same `EagerExecutor` exposed by the model
  executor.

## Lifecycle

### 1. Construct against static maximums

The model executor creates `PagedKVCacheConfig` and `PageTableLayout`.

For a direct application, `num_blocks` can already equal `max_num_blocks`.
That layout is final.

For a serving system, `num_blocks` may initially be `None`. The initial layout
uses `max_num_blocks` only as a cheap construction-time ceiling. No physical
KV tensor has been allocated yet.

### 2. Resolve physical KV capacity

If the inference server supplies a physical cache shape:

```text
server cache spec
  -> boundary adapter validates shape, dtype, and layer count
  -> new PagedKVCacheConfig(num_blocks=physical_count)
  -> model executor configures PagedKVCacheManager
  -> PageTableLayout.resolve(physical_count)
  -> prefill/decode/warmup receive bounded immutable layout replacements
```

The replacement may shrink geometry within the original ceilings. It may not
change block size or expand capacity.

### 3. Seal, allocate, and bind

Immediately before physical allocation, the model executor seals runtime
configuration. `PagedKVCacheManager.allocate()` then:

1. derives per-layer K/V shapes from model-owned attention metadata;
2. allocates every K/V tensor;
3. creates an immutable borrowed `PagedKVCacheContext`;
4. binds the cache transactionally through `model.set_kv_cache(cache)`; and
5. returns a borrowed compatibility handle.

The manager remains the owner. Requests may only present the exact returned
handle with unchanged tensor identities.

### 4. Compile and warm

Compilation requires an allocated and bound KV context.

`WarmupCoordinator.warmup_prefill()` and `warmup_decode()` can be called in
either order. Each method:

- validates dynamic warmup hints against static config ceilings;
- materializes device sampling buffers before capture when needed;
- resolves representative inputs through the real planner and lane routing;
- compiles the required eager program signatures;
- registers regular and configured cached/chunked trace plans when requested;
  and
- records completed coverage idempotently.

The compiler and trace registries are authoritative. An immutable coverage
manifest records eager programs, traced source programs, trace signatures, and
aliases; schema and workspace fingerprints keep alias capture independent of
request order. Trace capture begins only after configured prefill and decode
coverage, required programs, and aliases are complete. For Q128 prefill,
configured warmup batch sizes denote physical padded waves; other configured
sequence lengths use single-row coverage by default. In a lane group, capture
is deferred until every lane reports identical readiness, then activated as
one barrier. These barriers prevent early capture from closing the compile gate
before another operation or lane has registered its programs.

### 5. Serve

After trace activation:

- an intentionally eager operation may execute an already-compiled signature;
- no unseen program signature may compile;
- a traced operation must have an explicit program-to-trace association for
  every prepared request before any executes; and
- a selected traced operation never downgrades a coverage miss to eager.

### 6. Read outputs

Blocking output paths read and normalize immediately.

The asynchronous decode path retains ownership:

```text
decode_forward(read_from_device=False)
  -> raw TT output is placed in a DecodeOutputLease
  -> read_decode_output(async_read=True)
  -> OutputReader.submit()
  -> non-blocking host destination + recorded TT event
  -> external scheduler receives (host_value, events)
  -> process_decode_output_host(host_value)
  -> OutputReader.complete()
  -> ttnn.event_synchronize(event)
  -> output normalization + lease release
```

`OutputReader.drain()` and `DecodeRuntime.drain_external_outputs()` retire
anything the external scheduler did not complete.

### 7. Cleanup and terminal state

The model executor should make itself terminal first, then clean up in this
order:

1. outstanding external decode leases;
2. pending output reads;
3. prefill transients;
4. decode transients;
5. trace artifacts and persistent inputs;
6. the program registry;
7. device sampling buffers; and
8. the model-bound paged KV cache.

Cleanup is idempotent where possible. A failed cleanup preserves both the
primary error and any additional cleanup failures.

## Public API call chains

### Eager prefill

The model executor validates the bound cache and sampling capability before
delegating:

```text
model_executor.prefill_forward(execution=eager, ...)
  -> EagerExecutor.prefill_forward(...)
  -> PrefillRuntime.prepare(...)
       -> plan immutable request/chunk values
       -> classify sampling path
       -> derive program and optional trace signatures
  -> PrefillRuntime.invoke(prepared) for each planned request
       -> execute the complete regular or chunked sequence
  -> PrefillRuntime.assemble(...)
       -> read device outputs
       -> restore source-row order
       -> merge log probabilities
       -> release invocation transients
```

Within a lane, compatible rows become one padded wave per padded-sequence
bucket. Ineligible rows become sequential single requests; cached or long
single requests may themselves contain several chunk steps. Slots are carried
within a wave and do not by themselves split it. Batched sampling uses top-k;
the forced-argmax path is restricted to single-row requests.

### Traced prefill

Compilation and capture:

```text
TracedExecutor.compile_prefill(...)
  -> exact eager executor prepares request
  -> eager compiler compiles program
  -> PrefillRuntime.capture_plan(prepared)
  -> TraceCompiler.register_capture_plan(...)
  -> WarmupCoordinator barrier
  -> TraceCompiler.capture_all()
```

Replay:

```text
TracedExecutor.prefill_forward(...)
  -> PrefillRuntime.prepare(...)
  -> preflight every prepared request and program-to-trace association
  -> for each prepared request and each chunk step:
       -> PrefillRuntime.refresh_trace(...)
       -> TraceCompiler.replay(...)
       -> non-blocking ttnn.execute_trace(...)
  -> PrefillRuntime.finish_trace(...) after each request's final step
  -> PrefillRuntime.assemble(...)
```

Regular single, regular batched, cached, and multi-chunk requests can be traced
when their invocation geometry has compiled and captured coverage. Batched
trace identity uses the padded batch size, not the number of active rows.
Cached offsets and chunk starts are refreshed runtime inputs, so one fixed
chunk trace can be replayed repeatedly in a host loop for a long request.
`PrefillRuntime.can_trace()` is a capability query against invocation geometry;
it is not permission to route a selected-operation trace miss to eager.
Preflight covers the complete public call before the first KV write, and exact
association misses raise `TraceCoverageError`.

The capability query intentionally uses only its legacy token/length/start
inputs and therefore cannot classify page-table or sampling details. Serving
correctness does not rely on that approximation: traced execution prepares the
real request and preflights every resulting program-to-trace association.

### Eager decode

```text
model_executor.decode_forward(execution=eager, ...)
  -> EagerExecutor.decode_forward(...)
  -> DecodeRuntime.prepare(...)
       -> validate and normalize page table
       -> classify logits/argmax/top-k
       -> classify device position feedback
       -> record reset and page-table-change state
  -> DecodeRuntime.invoke(...)
       -> stage host/device inputs
       -> run model body and optional device sampling
  -> DecodeRuntime.consume(...)
       -> blocking host read and normalization, or
       -> transfer raw output to an external lease
```

### Traced decode

```text
TracedExecutor.compile_decode(...)
  -> eager compile
  -> DecodeRuntime.capture_plan(...)
  -> TraceCompiler.register_capture_plan(...)

TracedExecutor.decode_forward(...)
  -> DecodeRuntime.prepare(...)
  -> TraceCompiler.replay(...)
       -> refresh full inputs, page table, or only replay-varying fields
       -> non-blocking ttnn.execute_trace(...)
  -> DecodeRuntime.note_submitted(...)
  -> DecodeRuntime.consume(...)
```

Refresh decisions depend on reset, graph switching, page-table changes, and
whether device position feedback is compatible with the current request.

### Program compilation

```text
operation-specific prepared request
  -> operation-specific program signature
  -> ProgramCompiler.key_for(signature)
  -> ProgramCompiler.compile(signature, invoke)
       -> require bound KV context
       -> run once to materialize the TT program
       -> synchronize
       -> retain output contract
       -> release compile output
  -> CompiledProgram
```

Repeated compilation of the same signature is idempotent. Hash identity is
derived from canonical signature material, not object identity.

### Data-parallel execution

`LaneGroupExecutor` presents the same duck-typed execution-target API as one
model executor.

Prefill:

```text
global request rows + global slots
  -> assign rows to lanes by slot
  -> convert slots to lane-local slots
  -> route row-scoped sampling parameters with those rows
  -> plan lane-local padded-sequence buckets and waves
  -> preflight every participating traced lane and prepared request
  -> execute participating lanes only after all preflights pass
  -> restore original source-row order
```

Singleton tensor/list/tuple sampling fields broadcast within each participating
lane; other row-scoped fields follow source rows through lane routing. This
facade aggregates tokens and logits, but data-parallel log-probability
aggregation is not implemented and is rejected explicitly.

Decode:

```text
global fixed-capacity batch
  -> split into contiguous lane batches
  -> execute lanes
  -> aggregate tokens/logits or retain per-lane raw outputs
```

Cache configuration, allocation, and cleanup cover every lane. Decode
compilation covers every fixed-capacity lane, while prefill compilation calls
only lanes participating in the slot-routed request. Warmup coordinates
readiness and trace activation across all lanes. Async output reads run
concurrently per lane and return one combined external contract.

## Main public surfaces

| Surface | Intended caller | Purpose |
| --- | --- | --- |
| `PageTableLayout.resolve` | model executor | Derive shared aligned prefill/decode geometry |
| `PrefillRuntimeConfig.resolve` | model executor | Resolve prefill collaborators and static capabilities |
| `DecodeRuntimeConfig.resolve` | model executor | Resolve decode collaborators and static capabilities |
| `WarmupCoordinatorConfig.resolve` | model executor | Validate cross-runtime identity and build coverage plans |
| `configure_page_table_layout` methods | model executor | Install final physical geometry before allocation/use |
| `EagerExecutor.compile_*` / `*_forward` | model executor or lane | Compile and run eager requests |
| `TracedExecutor.compile_*` / `*_forward` | model executor or lane | Register and replay traced requests |
| `PrefillRuntime.can_trace` | model target and diagnostics | Query request-level prefill trace capability by invocation geometry |
| `ProgramCompiler` | eager/traced composition | Own eager program identity and readiness |
| `TraceCompiler` | traced composition and warmup | Own trace artifacts and replay |
| `PagedKVCacheManager` | model executor | Own physical KV allocation and binding |
| `OutputReader` | prefill/decode runtime | Own device-to-host completion lifecycle |
| `WarmupCoordinator` | model executor | Compile coverage and trigger trace capture barrier |
| `LaneGroupExecutor` | model generator/facade | Present multiple lanes as one execution target |
| `VLLMAdapter` | model-owned vLLM facade | Normalize vLLM calls and validate legacy KV specs |

Callers outside this package should normally use a model-layer executor or
generator instead of invoking `PrefillRuntime`, `DecodeRuntime`,
`ProgramCompiler`, or `TraceCompiler` directly.

## Reusing the toolkit for another model

A new model family should provide:

1. a TTTv2 tensor model with prefill/decode methods and model-owned module
   configs;
2. `iter_executor_named_modules()` for runtime input-contract validation;
3. transactional `set_kv_cache(cache_or_none)`;
4. attention metadata for per-layer KV dtype, local KV heads, head dimension,
   block size, and maximum blocks;
5. model methods used by prefill/decode staging and post-processing;
6. an optional resolved device-sampling module;
7. either the shared model-layer `ModelExecutor`, a justified family policy
   facade, or a concrete model-owned executor that composes this package
   directly; and
8. a server-specific facade that normalizes external calls and selects eager
   or traced execution.

Do not add a generic aggregate executor to this directory. Share model
composition in `models/common/models` when the lifecycle is demonstrated to be
common; otherwise compose the focused modules directly. Keep family/model and
server policy at their respective boundaries.
