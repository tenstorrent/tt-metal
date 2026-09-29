# Llama 3.1 8B with TTTv2

This directory contains the model-owned Llama 3.1 8B product path built from
TTTv2 modules and the reusable common LLM runtime.

The path has four layers:

```text
model provider / checkpoint
  -> hf_adaptor.py: provider metadata, tokenizer, and weight conversion
  -> model.py: TTTv2 tensor model assembled from reusable modules
  -> executor.py: thin typed entry point into the Llama family executor
  -> generator.py: vLLM-facing construction, DP composition, and dispatch
```

The most important boundary is between the tensor model and runtime
orchestration:

- TTTv2 `LightweightModule` objects implement tensor computation.
- [`models/common/llm_runtime`](../../llm_runtime/README.md) implements reusable
  execution, tracing, I/O, cache, warmup, and resource mechanics.
- `models/common/models/executor.py::ModelExecutor` composes the common owners.
- `models/common/models/llama3_executor.py::Llama3Executor` supplies the
  Llama-8B sampling and prefill policy as a composition facade.
- `Llama3Generator` adapts the resulting target to vLLM.

## Files

| File | Responsibility |
| --- | --- |
| `hf_adaptor.py` | Load HF config/tokenizer/weights, convert provider naming/layout, compute Llama 3 RoPE values, and create the product model |
| `model.py` | Build and execute the TTTv2 Llama transformer graph |
| `executor.py` | Preserve the model-local typed builder/import surface over `llama3_executor.py` |
| `generator.py` | Construct lanes, optionally compose DP, normalize vLLM calls, and select eager/traced execution |

## End-to-end object graph

For one lane:

```text
Llama3ForCausalLM
├── tokenizer
├── Llama3RuntimeConfig
└── Llama3Transformer1D
    ├── Embedding1D
    ├── RotarySetup1D
    ├── TransformerBlock1D × N
    │   ├── RMSNorm1D
    │   ├── Attention1D
    │   ├── RMSNorm1D
    │   └── MLP1D
    ├── RMSNorm1D
    ├── LMHead1D
    └── optional Sampling1D

Llama3Executor composition facade
└── ModelExecutor
    ├── exact Llama3Transformer1D above
    ├── PagedKVCacheManager
    ├── OutputReader
    ├── PrefillRuntime
    ├── DecodeRuntime
    ├── ProgramCompiler
    ├── EagerExecutor
    ├── optional TraceCompiler
    ├── optional TracedExecutor over the exact EagerExecutor
    └── WarmupCoordinator
```

The vLLM-facing graph is:

```text
Llama3Generator
├── VLLMAdapter
└── target
    ├── Llama3Executor                         when DP = 1
    └── LaneGroupExecutor[Llama3Executor, ...] when DP > 1
```

`Llama3Generator` owns no TT tensors. The lane executors own resources, and a
`LaneGroupExecutor` owns lane/pool lifecycle coordination.

## Building the tensor model

### Provider adaptation

`from_pretrained(...)` in `hf_adaptor.py` is the current Hugging Face provider
entry point. It:

1. resolves the model ID;
2. loads `AutoConfig` and the tokenizer;
3. derives hidden size, heads, KV heads, layers, vocabulary, norm epsilon, and
   context length;
4. computes Llama 3 scaled RoPE cosine/sine tables;
5. loads the HF state dict;
6. splits fused QKV or gate/up weights when necessary;
7. converts Q/K rotary weight layout;
8. maps HF names to the model's Meta-style names;
9. builds `Llama3Transformer1DConfig`;
10. constructs `Llama3Transformer1D`; and
11. returns `Llama3ForCausalLM`, which packages the tensor model, tokenizer,
    generation defaults, and `Llama3RuntimeConfig`.

Provider-facing concerns stop there. Neither `Llama3Executor` nor the common
runtime reads HF config or converts HF weights.

### TTTv2 module composition

`build_llama3_transformer_1d_config(...)` translates Llama architecture and
optimization choices into configs for reusable TTTv2 modules:

- `Embedding1D`
- `RotarySetup1D`
- `RMSNorm1D`
- `Attention1D`
- `MLP1D`
- `LMHead1D`
- optional `Sampling1D`

`Llama3Transformer1D` constructs these modules. Each
`TransformerBlock1D` performs:

```text
attention RMSNorm
  -> Attention1D
  -> residual add
  -> feed-forward RMSNorm
  -> MLP1D
  -> residual add
```

The model exposes two graph entry points:

- `prefill_forward(...)` for one planned regular/batched/chunk invocation; and
- `decode_forward(...)` for one autoregressive step across the fixed lane
  capacity.

It also exposes executor support methods:

- `iter_executor_named_modules()` yields modules whose input contracts must be
  validated during execution;
- `set_kv_cache(cache_or_none)` transactionally binds/unbinds per-layer K/V
  tensors;
- embedding and rotary preparation methods stage model inputs;
- prefill post-processing converts a traced hidden body to logits/sampled
  output; and
- decode output gathering and position increment helpers support runtime
  execution.

## Constructing the vLLM model

The public class entry point is:

```text
Llama3Generator.initialize_vllm_model(...)
  -> Llama3GeneratorConfig
  -> build_llama3_generator(config)
```

`build_llama3_generator(...)` performs the following steps.

### 1. Resolve lane geometry

The global vLLM batch is divided evenly by `tt_data_parallel`. For DP1, the
whole mesh is one lane. For DP2/DP4/DP8, the mesh is split into one submesh per
lane.

Each lane receives:

- one submesh;
- one fixed per-lane batch capacity;
- the same maximum sequence length;
- the same optimization/precision policy; and
- the same trace and device-sampling policy.

### 2. Build one product model per lane

For each submesh:

```text
from_pretrained(...)
  -> Llama3ForCausalLM
  -> Llama3Transformer1D on that submesh
```

The paged-attention block size is 32. `max_num_blocks` is a safe static
construction ceiling derived from maximum sequence length and per-lane batch
capacity.

### 3. Build one model-owned executor per lane

The generator creates `Llama3ExecutorConfig`:

- `TraceConfig(trace_mode)`
- `WarmupConfig()`
- unresolved `PagedKVCacheConfig`
- device-sampling capability

It then calls:

```text
build_llama3_executor(Llama3ForCausalLM, executor_config)
  -> llama3_executor.Llama3Executor facade
  -> ModelExecutor(model, runtime_config, executor_config, Llama policy)
```

The family facade creates native `SamplingState1D` state and resolves the
Llama-8B device-sampling prefill policy. The shared `ModelExecutor` composes
the runtime owners and exposes three execution targets:

- `eager_execution`: always the one `EagerExecutor`;
- `traced_prefill_execution`: the one `TracedExecutor` when prefill tracing is
  configured; and
- `traced_decode_execution`: the same `TracedExecutor` when decode tracing is
  configured.

There is no aggregate executor in `llm_runtime`. The shared composition root
lives in the model layer at `models/common/models/executor.py`.

### 4. Build the vLLM boundary adapter

Model metadata is read from the already-built attention configs:

- layer count;
- KV dtype per layer;
- local KV heads per device; and
- head dimension.

That metadata resolves `VLLMAdapterConfig`. `VLLMAdapter` then owns only static
vLLM normalization/validation policy; it owns no TT resource.

### 5. Compose the target

For DP1, the target is the single `Llama3Executor`.

For DP greater than one:

```text
LaneGroupExecutor(lanes)
  -> one duck-typed global execution target
```

The lane group:

- assigns prefill rows to lanes from their global slots;
- maps global slots to lane-local slots;
- splits decode into contiguous per-lane batches;
- aggregates outputs in global order;
- replicates cache configuration, warmup, and compilation; and
- coordinates concurrent asynchronous output handling and cleanup.

Finally:

```text
Llama3Generator(target, vllm_adapter)
```

is returned to vLLM.

## vLLM lifecycle

### 1. Model construction uses only a maximum KV ceiling

At construction, the generator does not know vLLM's final physical block
count. Each lane therefore has:

```text
PagedKVCacheConfig(
    block_size=32,
    max_num_blocks=construction_ceiling,
    num_blocks=None,
)
```

`Llama3Executor` can still construct prefill, decode, and warmup config against
the maximum. This is cheap TTTv2 reconfiguration: no physical KV tensor is
allocated at this point.

### 2. vLLM resolves physical KV capacity

vLLM calls:

```text
Llama3Generator.allocate_kv_cache(kv_cache_shape, dtype, num_layers)
```

The call chain is:

```text
VLLMAdapter.resolve_legacy_kv_cache_config(...)
  -> validate physical blocks <= maximum
  -> validate local KV heads, block size, head dimension, layer count, dtype
  -> return new PagedKVCacheConfig(num_blocks=physical_blocks)

target.configure_paged_kv_cache(resolved_config)
  -> one executor or every DP lane
  -> PagedKVCacheManager.configure(...)
  -> recompute PageTableLayout for physical capacity
  -> replace PrefillRuntimeConfig layout
  -> replace DecodeRuntimeConfig layout
  -> replace WarmupCoordinatorConfig layout and rebuild coverage plans

target.allocate_kv_cache()
  -> seal runtime geometry
  -> allocate per-layer K/V tensors
  -> bind tensors to Llama3Transformer1D
```

This ordering is important: the physical page-table layout is installed before
allocation, compilation, warmup, or trace capture.

### 3. Warmup and trace capture

vLLM calls `warmup_model_prefill(...)` and `warmup_model_decode(...)`.

Each lane compiles all required program variants. Trace capture waits at the
shared warmup barrier until both configured operation sets are ready. Sampling
buffers are loaded before capture.

For `trace_mode="all"`, prefill and decode traces are separate artifacts over
the same eager program compiler. This means vLLM may still request eager or
traced execution independently on every forward call.

### 4. Prefill dispatch

```text
vLLM
  -> Llama3Generator.prefill_forward(...)
  -> VLLMAdapter.normalize_prefill(...)
       -> bind positional arguments
       -> remove known irrelevant compatibility fields
       -> require explicit Boolean enable_trace
       -> normalize torch dtypes
  -> Llama3Generator._select_prefill_execution(...)
       -> if trace requested, target.can_trace_prefill(...)
       -> cached/chunked/unsupported requests select eager
       -> eligible requests select traced
  -> target.prefill_forward(execution=selected, ...)
       -> Llama3Executor, or LaneGroupExecutor -> each Llama3Executor
  -> selected EagerExecutor or TracedExecutor
  -> PrefillRuntime
  -> Llama3Transformer1D
```

The fallback belongs here, at the vLLM/model boundary. `TracedExecutor` never
silently invokes eager execution.

### 5. Decode dispatch

```text
vLLM
  -> Llama3Generator.decode_forward(...)
  -> VLLMAdapter.normalize_decode(...)
  -> explicit enable_trace selects:
       false -> target.eager_execution
       true  -> target.traced_decode_execution
  -> target.decode_forward(execution=selected, ...)
  -> DecodeRuntime
  -> Llama3Transformer1D
```

Decode trace availability is a static capability. Asking for traced decode
when it was not configured is an error at the vLLM boundary.

### 6. Asynchronous decode output

vLLM can request `read_from_device=False`. The executor returns a raw TT output
under an external lease.

```text
Llama3Generator.read_decode_output(async_read=True)
  -> lane target
  -> DecodeRuntime.read_decode_output(...)
  -> OutputReader.submit(...)
  -> host destination + TT completion events

Llama3Generator.process_decode_output_host(...)
  -> DecodeRuntime.process_decode_output_host(...)
  -> OutputReader.complete(...)
  -> ttnn.event_synchronize(...)
  -> normalize output and release the lease
```

For DP, the lane group performs the per-lane reads concurrently and aggregates
the completed outputs.

### 7. Cleanup

`Llama3Generator.cleanup()` delegates to the target.

One `Llama3Executor` terminalizes and releases:

1. externally leased decode outputs;
2. pending output reads;
3. prefill/decode transients;
4. trace resources;
5. program registry state;
6. sampling buffers; and
7. the bound paged KV cache.

The DP target cleans every lane and then its worker pool. Construction failures
also clean all lanes that were already created.

## Trace-mode behavior

Every vLLM forward call carries an explicit `enable_trace` Boolean.

| Static `trace_mode` | Operation | `enable_trace=False` | `enable_trace=True` |
| --- | --- | --- | --- |
| `none` | prefill or decode | eager | rejected by adapter |
| `decode_only` | prefill | eager | rejected by adapter |
| `decode_only` | decode | eager | traced |
| `all` | decode | eager | traced |
| `all` | eligible regular prefill | eager | traced |
| `all` | cached, chunked, or otherwise trace-ineligible prefill | eager | generator selects eager |

`trace_mode="all"` is the most flexible serving construction because prefill
and decode artifacts are independent. It supports per-call eager/traced
selection without reconstructing the model.

## Applying this pattern to another LLM

The reusable pattern is not “subclass Llama3.” It is:

```text
provider adapter
  -> model-specific TTTv2 graph
  -> shared/family model executor or direct runtime composition
  -> server-specific facade
```

### Model implementation

A new model should build its tensor graph from reusable TTTv2 modules where
possible. The exact module set may differ: another architecture might use a
different attention implementation, normalization, MLP, MoE, positional
encoding, or output head.

The tensor model should expose the runtime contract needed by its executor:

- prefill and decode graph entry points;
- model-owned embedding/input and output-processing helpers;
- module iteration for input-contract validation;
- transactional KV-cache binding;
- per-layer cache metadata; and
- optional device sampling.

### Model execution composition

Use the shared `models/common/models/executor.py::ModelExecutor` when the model
fits its established lifecycle. A demonstrated family may add a small policy
facade such as `llama3_executor.py` or `qwen2_executor.py`.

When a model has genuinely distinct orchestration, its model-local
`executor.py` may instead compose the focused `llm_runtime` modules directly.
Either construction should:

- translate model metadata into resolved common runtime configs;
- construct one exact eager execution composition;
- optionally construct one trace compiler and one traced executor over it;
- own page-layout sealing and late physical-capacity replacement;
- validate that request cache handles belong to its cache manager;
- expose the duck-typed execution target used by a DP group; and
- be the deterministic cleanup root.

Do not add a generic aggregate model executor to `llm_runtime`, and do not
force every model through the shared model-layer executor.

### Server facade

Create a facade for the target serving system. It should own:

- external argument normalization;
- external cache-shape adaptation;
- per-call eager/traced selection;
- request-level trace eligibility fallback;
- server-specific async-output conventions; and
- construction of single-lane or DP targets.

The common prefill/decode/compiler/cache mechanics should not interpret the
server's policy.

## Extensibility dimensions

This architecture separates several dimensions that can evolve independently.

### Other model architectures

Llama, Mistral, Qwen, Gemma, MoE models, and future architectures can share the
runtime mechanics while owning different TTTv2 module graphs and executors.

### Other inference servers

vLLM is one facade. An SGLang integration can build the same model executor and
provide an SGLang-specific adapter for request fields, cache negotiation,
trace selection, and asynchronous output conventions. A direct demo or custom
service can bypass server adapters and call the model-owned executor with an
explicit execution target.

### Other model providers

Hugging Face is currently isolated in `hf_adaptor.py`. Another provider can
supply:

- architecture metadata;
- tokenizer/chat formatting;
- a state-dict reader;
- provider-to-model key and tensor-layout conversion; and
- cache location policy.

That provider adapter should produce the same model product shape:

```text
TTTv2 tensor model + tokenizer + model runtime metadata
```

The Llama executor and common runtime do not need to know whether weights came
from Hugging Face, a native Meta checkpoint, an internal artifact store, or a
preconverted tensor cache.

### Other topologies and execution policies

Mesh topology, tensor parallelism inside modules, data-parallel lane count,
precision/optimization policy, paged-KV capacity, device sampling, warmup
coverage, and trace mode are separate configuration dimensions. A new
combination should normally require new resolved configs and validation, not a
fork of runtime control flow.

## Practical checklist for a new integration

1. Build and validate the provider adapter.
2. Construct the TTTv2 tensor model from module configs.
3. Expose model runtime and KV metadata.
4. Select shared model-layer composition, a justified family policy facade, or
   direct composition from the common runtime.
5. Test direct eager prefill/decode and cleanup.
6. Add program compilation and warmup coverage.
7. Add trace capture/replay without eager fallback inside `TracedExecutor`.
8. Add late physical KV-capacity resolution.
9. Add a server facade that owns normalization and dispatch.
10. Add DP composition through `LaneGroupExecutor` if required.
11. Validate accuracy, deterministic text quality, sustained TPOT, aggregate
    throughput, and cleanup across all supported geometries.
