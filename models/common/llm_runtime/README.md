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

It deliberately does **not** contain a generic model executor. Each model owns
its concrete executor and cleanup root. For example,
[`Llama3Executor`](../models/llama3_8b/executor.py) composes these runtime
pieces for Llama 3.1 8B.

## Design principles

### The model owns orchestration

The common package supplies focused components, not a single common LLM
executor. A model-owned executor decides:

- which tensor model is being executed;
- how model metadata becomes runtime config;
- which resources exist for one execution lane;
- whether eager and traced targets are constructed;
- when physical KV capacity becomes final;
- which execution target handles a call; and
- the cleanup order.

This keeps model-specific decisions out of common runtime mechanics while
allowing different model families to reuse those mechanics.

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
an eager fallback. The caller selects either target. A serving adapter that
discovers a trace-ineligible request must call the eager target explicitly.

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
- actual and padded sequence lengths;
- cached-token offsets;
- canonical page-table width;
- regular single, regular batched, or chunked execution;
- chunk token slices and absolute chunk positions; and
- which chunk contains the final token.

`PrefillRuntime` consumes that plan. There is no separate cached-prefill
service and no separate chunked-prefill service.

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

| Module | Main responsibility |
| --- | --- |
| `config.py` | Trace policy, warmup policy, paged-KV policy, and canonical page-table geometry |
| `prefill/config.py` | Fully resolved static prefill collaborators, capabilities, and geometry ceilings |
| `prefill/plan.py` | Pure host-side construction of `PrefillRequest` and `PrefillChunk` values |
| `prefill/runtime.py` | Prefill preparation, eager sequence execution, trace hooks, assembly, and transient cleanup |
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
- compiles the required eager program signatures;
- registers trace plans when tracing is requested; and
- records completed coverage idempotently.

Trace capture begins only after the configured prefill and decode coverage is
complete. This shared barrier prevents early capture from closing the compile
gate before the other operation has registered its programs.

### 5. Serve

After trace activation:

- an already-compiled eager signature may still execute eagerly;
- no unseen program signature may compile;
- traced calls must have an explicit program-to-trace association; and
- the serving boundary, not `TracedExecutor`, chooses eager versus traced
  execution.

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

One public prefill call can produce several prepared requests because rows may
have different prompt lengths, slots, cached prefixes, or chunk sequences.

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
  -> TraceCompiler.replay(...)
       -> PrefillRuntime.refresh_trace(...)
       -> non-blocking ttnn.execute_trace(...)
  -> PrefillRuntime.finish_trace(...)
  -> PrefillRuntime.assemble(...)
```

Only regular, uncached requests with configured sequence lengths are trace
eligible. A server adapter should call `PrefillRuntime.can_trace()` through its
model-owned target before selecting traced execution. Cached or multi-chunk
prefill is dispatched to the eager target by that adapter.

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
  -> call each participating lane
  -> restore original source-row order
```

Decode:

```text
global fixed-capacity batch
  -> split into contiguous lane batches
  -> execute lanes
  -> aggregate tokens/logits or retain per-lane raw outputs
```

Cache configuration, allocation, compilation, warmup, and cleanup are
replicated across lanes. Async output reads run concurrently per lane and
return one combined external contract.

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
| `PrefillRuntime.can_trace` | serving dispatch through model target | Classify request-level prefill trace eligibility |
| `ProgramCompiler` | eager/traced composition | Own eager program identity and readiness |
| `TraceCompiler` | traced composition and warmup | Own trace artifacts and replay |
| `PagedKVCacheManager` | model executor | Own physical KV allocation and binding |
| `OutputReader` | prefill/decode runtime | Own device-to-host completion lifecycle |
| `WarmupCoordinator` | model executor | Compile coverage and trigger trace capture barrier |
| `LaneGroupExecutor` | model generator/facade | Present multiple lanes as one execution target |
| `VLLMAdapter` | model-owned vLLM facade | Normalize vLLM calls and validate legacy KV specs |

Callers outside this package should normally use a model-owned executor or
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
7. a concrete model-owned executor that composes this package; and
8. a server-specific facade that normalizes external calls and selects eager
   or traced execution.

Do not solve reuse by adding a generic aggregate executor to this directory.
Share mechanics here; keep model and server policy at their respective
boundaries.
