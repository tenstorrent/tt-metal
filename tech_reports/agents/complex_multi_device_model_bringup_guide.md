# Complex Multi-Device TTNN Model Bringup Guide

This guide is for an expert PyTorch and systems engineer who has not yet worked
with TTNN, Tenstorrent devices, or the tt-metal model stack. It is written to be
usable by a human or by an advanced coding agent. When it says to use
subagents, a human can read that as "ask independent reviewers"; an agent can
literally delegate the check.

The target is any LLM or LLM-like model that must run on a multi-device
Tenstorrent platform. The process should work for a relatively conventional
decoder-only model as well as a model with mode-specific execution, custom
attention, MoE, custom collectives, quantized experts, compressed caches,
multimodal inputs, or other tensor contracts that do not map cleanly to ordinary
transformer blocks.

## Primary Objective

Bring up a model from a PyTorch or HuggingFace implementation into a
multi-device, traced, sharded, optimized TTNN implementation with:

1. Few lines of model-specific code.
2. High performance on the intended Tenstorrent mesh.
3. Easy testability and debuggability at every abstraction boundary.
4. A path for custom ops when decomposition is correct but not fast enough.

Few lines means few lines in the final maintained runtime path. During bringup,
expect extra scaffolds, smoke runners, reference helpers, and narrow adapters.
The discipline is to keep those artifacts named as scaffolding and collapse them
behind semantic ops once their job is done.

The cleanest approach is not to build a general torch-to-TTNN compiler. Build a
model-specific lowering pipeline with explicit semantic contracts, shared helper
code from `models/common`, and a small set of selectable lowerings for each
semantic operation.

## Required Reading

Before writing model code, read these repo areas:

- `tech_reports/ttnn/TTNN-model-bringup.md`
- `tech_reports/LLMs/llms.md`
- `tech_reports/Programming_Mesh_of_Devices/Programming_Mesh_of_Devices_with_TT-NN.md`
- `tech_reports/AdvancedPerformanceOptimizationsForModels/AdvancedPerformanceOptimizationsForModels.md`
- `tech_reports/tensor_sharding/tensor_sharding.md`
- `tech_reports/tensor_layouts/tensor_layouts.md`
- `tech_reports/ttnn/graph-tracing.md`
- `tech_reports/ttnn/operation-tracing.md`
- `tech_reports/ttnn/comparison-mode.md`
- `models/common/README.md`
- `models/common/modules/users_guide.md`
- `models/common/lightweightmodule.py`
- `models/common/modules/lazy_weight.py`
- `models/common/validation_tools.py`
- `models/common/auto_compose.py`
- `models/common/distribute_as.py`
- `models/common/modules/attention/attention_1d.py`
- `models/common/modules/mlp/mlp_1d.py`
- `models/common/modules/tt_ccl.py`

Then read these existing model approaches as examples, not as dependencies:

- `models/tt_transformers`
- `models/demos/gpt_oss`
- `models/demos/deepseek_v3`
- `ttnn/cpp/ttnn/operations/transformer/sdpa`
- `ttnn/cpp/ttnn/operations/transformer/sdpa_decode`
- `ttnn/cpp/ttnn/operations/ccl/all_to_all_dispatch`
- `ttnn/cpp/ttnn/operations/ccl/all_to_all_combine`
- `ttnn/cpp/ttnn/operations/experimental/reduction/deepseek_grouped_gate`
- `ttnn/cpp/ttnn/operations/experimental/deepseek`

For the target model, separately read the authoritative PyTorch or HuggingFace
implementation. Do not use an existing tt-metal model branch as the source of
truth for model semantics. Existing tt-metal model code may be studied for TTNN,
mesh, tracing, common-module, and custom-op patterns only.

Pin exact source revisions before turning formulas into tests:

- Model repository revision, such as an HF model repo commit.
- Transformers, vLLM, Megatron, or vendor reference commit if used.
- Checkpoint config revision.
- Any inference-kernel repository revision.
- Serving mode: prefill, decode, generation, speculative/MTP, embedding, or
  logits-only.

If multiple references disagree, record the disagreement in `source_of_truth.md`
and use subagents to decide which reference matches the target checkpoint and
serving mode. URLs to `main` are orientation commands, not reproducibility
guarantees.

## TTNN Mental Model

TTNN is a Python-facing tensor library that lowers operations to Tenstorrent
device programs. For model bringup, think in terms of these layers:

- Logical model semantics: what the PyTorch graph computes.
- Tensor contracts: shape, dtype, layout, memory location, sharding, mesh
  placement, and ownership after every semantic operation.
- TTNN decomposition: ordinary TTNN ops that implement a semantic operation.
- Custom TTNN ops: fused or special-purpose C++ operations when decomposition
  is correct but slow, memory-heavy, or impossible to express cleanly.
- Runtime plan: preallocated tensors, static trace signatures, program cache,
  two-command-queue synchronization, and host/device data movement.

Tenstorrent performance is usually decided by physical tensor contracts as much
as by arithmetic. A mathematically correct sequence that repeatedly reshards
activations, spills to DRAM, moves weights across the NoC, or breaks trace
capture is not a useful end state.

Key concepts:

- `MeshDevice`: a multi-device mesh presented to TTNN.
- Mesh mappers and composers: explicit host-to-mesh and mesh-to-host tensor
  placement.
- Tensor memory config: DRAM, L1, interleaved, block sharded, height sharded,
  width sharded, or device-specific layouts.
- Program config: kernel/grid/blocking choices for an op.
- CCL: collectives across chips, such as all-gather, reduce-scatter,
  all-to-all dispatch, and all-to-all combine.
- Trace: captured execution for static-shape fast replay.
- Program cache: cached compiled device programs, warmed before stable timing
  and trace capture.

Two current-code caveats matter:

- Do not assume trace means decode-only. Some current stacks support gated
  prefill tracing when the entry point satisfies trace constraints.
- Do not describe a collective such as all-reduce as a universal composition of
  simpler collectives. CCL implementation details are path-specific; inspect the
  current op and mesh path before relying on ordering or intermediate layout.

## Non-Negotiable Principles

### Plan the Whole-Model Topology Early

For large models, do not start with a single-device mental model and bolt on
sharding later. DeepSeek-class models do not fit that way, and many operations
only make sense after deciding tensor parallelism, expert parallelism, sequence
parallelism, cache ownership, and CCL boundaries.

The right compromise is:

1. Use a simple PyTorch or CPU reference to understand semantics.
2. Use simple DRAM-interleaved TTNN decompositions as disposable microscopes for
   suspicious operations.
3. Move to the final multi-device tensor contracts before integrating modules
   into the full model.
4. Optimize program configs and custom ops after the contracts are stable.

Do not treat "functional DRAM-interleaved bringup" as the real implementation
unless the model is small enough that the end state is genuinely close to that.

### Share Weights, Not Runtime Paths

Prefill and decode often need the same canonical weight layout in device DRAM.
That does not mean they must share one forward path. Prefer:

- One converted weight artifact per parameter.
- One canonical physical layout per weight unless there is a measured reason to
  duplicate.
- Separate prefill and decode execution plans when shapes, cache behavior,
  activation sharding, or optimal kernels differ.

The mistake to avoid is coupling unrelated execution logic just because weights
are shared. Weight layout is an ABI; the runtime plan is allowed to be
mode-specific.

### Model Code Should Be a Semantic Plan Plus Lowerings

Keep model-specific code small by representing the model as semantic operations:

- `attention_prefill`
- `attention_decode`
- `mla_prefill`
- `mla_decode`
- `router`
- `expert_dispatch`
- `expert_compute`
- `expert_combine`
- `hyperconnection_pre`
- `hyperconnection_post`
- `lm_head`

Each semantic operation can have multiple lowerings:

- PyTorch reference.
- CPU reference specialized for the model.
- Decomposed TTNN correctness lowering.
- Final-topology TTNN lowering.
- Fused or custom-op TTNN lowering.
- Traced runtime lowering.

Tests should be written against the semantic contract. The selected lowering is
an implementation detail.

### Custom Ops Are First-Class

If an operation is central to model performance or has a tensor contract that is
awkward in ordinary TTNN, plan for a custom op early. Do not wait until the rest
of the model is complete if the custom op determines activation shape, sharding,
or cache layout.

Custom ops fall into three categories:

1. Fusion-preserving ops: same external tensor contract as the decomposed form,
   faster implementation. Example: a grouped gate reduction.
2. Physical-contract-changing ops: same math, different layout, packing,
   preallocation, or intermediate ownership. Example: a gate op that emits a
   padded per-core output format.
3. Logical-contract-changing ops: the operation is a new semantic boundary, not
   just a faster decomposition. Examples: MLA attention or MoE all-to-all
   dispatch/combine.

The third category must be reflected in the model's tensor contracts and
pseudocode specs before implementation.

### Make Pseudocode a Living Artifact

For every nontrivial operation, write optimized pseudocode before TTNN code.
Verify the pseudocode against PyTorch by hand and with subagents. Then implement
the TTNN lowering from the pseudocode, not from a vague memory of the PyTorch
module.

This separates three failure classes:

- Semantic misunderstanding: the pseudocode does not match PyTorch.
- Lowering bug: the TTNN code does not match the pseudocode.
- Stack bug or missing primitive: the TTNN op, custom op, runtime, or compiler
  behaves incorrectly.

Keep the pseudocode in the repo next to the model. Update it when tensor
contracts change.

## Project Artifacts

Create these artifacts before or alongside implementation:

- `MODEL_CARD.md`: model dimensions, modes, supported hardware, checkpoint
  format, expected accuracy, and expected performance.
- `lowering_specs/*.md`: pseudocode and tensor contracts per semantic op.
- `tensor_contracts.json`: machine-readable contracts for shapes, layouts,
  memory configs, mesh placement, and mode-specific variants.
- `weight_manifest.json`: converted weight names, source keys, dtypes, shapes,
  layouts, sharding, content hashes, and quantization metadata.
- `mesh_plan.md`: TP, EP, SP, DP, CCL, cache, and residual-stream ownership.
- `trace_signatures.md`: all traced entry points, static dimensions, persistent
  tensors, preallocated outputs, and cache address assumptions.
- `test_matrix.md`: tests by phase, hardware requirement, expected tolerance,
  and command.
- `perf_log.md`: stable benchmarks, profiler links, compile/cache state, and
  known bottlenecks.

These files are not bureaucracy. They are how a large model remains debuggable
after custom ops, mode-specific traces, and sharded weights enter the system.

## Bringup Phases

### Phase 0: Repo and Hardware Orientation

First, identify the intended hardware:

- Wormhole, Blackhole, or another architecture.
- Device memory size per chip.
- Mesh shape, for example 1x8, 2x4, T3K, or a larger platform.
- Fabric availability and CCL constraints.
- Whether the target mode is prefill, decode, or both.

Then map the relevant repo abstractions:

- Use `models/common` as the only reusable model code dependency unless there is
  a deliberate exception.
- Treat `models/tt_transformers`, `models/demos/gpt_oss`, and
  `models/demos/deepseek_v3` as examples to learn from, not as a base class
  hierarchy to extend.
- Identify existing custom ops that may be reusable or instructive.

Use subagents to separately summarize:

- TTNN tracing and program cache constraints.
- Mesh and CCL programming constraints.
- Current DeepSeek V3 custom op boundaries.
- The target model's unusual PyTorch semantics.

Compare their summaries for contradictions before coding.

### Phase 1: Survey the PyTorch Model

Build a precise module inventory:

- Config fields and derived dimensions.
- Layer count and layer variants.
- Attention type per layer.
- Cache format and cache update semantics.
- MLP, MoE, shared experts, routed experts, and expert quantization.
- Router scoring, bias, normalization, tie-breaking, and token-to-expert
  metadata.
- MTP or auxiliary prediction layers.
- Any compressed, sparse, sliding-window, or hyperconnection mechanism.
- Weight names, checkpoint shards, tied weights, and quantization metadata.

Also create `source_of_truth.md`. It must identify the exact implementation,
commit, files, and formulas used for every red-zone semantic operation. Do not
import conclusions from an existing tt-metal experiment until they have been
anchored back to the authoritative source. If the sources disagree, record both
formulas, checkpoint fields, and the reason for the selected serving source. If
no authoritative source is available for a submodule, record that as a blocker
for final accuracy claims and continue only with provisional bringup tasks.

Create a red-zone inventory for any behavior that is not a plain transformer
block:

- Attention variants: MQA/GQA/MLA, sliding windows, sparse attention, attention
  sinks, local/global alternation, grouped output projections, cross-attention,
  multimodal attention, or nonstandard RoPE.
- Cache variants: paged cache, sliding rings, compressed cache, multiple cache
  streams, query-dependent cache row selection, recurrent state, mutable prefix
  state, or mode-specific cache layout.
- MLP variants: gated activations, SwiGLU clamp, shared experts, routed experts,
  dense-MoE mixtures, residual MLPs, or per-layer MLP schedules.
- Router variants: top-k, top-p, hash routing, correction bias, grouped gating,
  normalization rules, tie-breaking, per-token metadata, or stochastic routing.
- Residual/norm variants: parallel residual streams, hyperconnections, unusual
  RMSNorm behavior, norm placement differences, or mixed-precision norm rules.
- Output variants: tied embeddings, sharded LM head, multimodal heads, MTP,
  speculative heads, classifiers, or logits slicing.
- Quantization variants: FP8, FP4, block scales, activation quantization,
  dequant-in-reference paths, packed weights, or checkpoint-specific scale ABIs.
- Parallelism variants: TP, EP, SP, DP, pipeline parallelism, rank-local
  checkpoints, all-reduce/all-gather/reduce-scatter/all-to-all boundaries.
- Runtime variants: trace-only paths, dynamic shapes, preallocated outputs,
  device-side sampling, host fallbacks, or custom kernels in the source model.

Do not write new optimized TTNN code until every red-zone item has a reference
test and a source-of-truth line in `source_of_truth.md`. It is acceptable to
write smoke scaffolding earlier when the purpose is to expose tensor contracts,
but mark it as provisional in the file docstring and test names.

### Phase 2: Write Lowering Specs

For each red-zone module, create a lowering spec with:

- Source PyTorch file and function.
- Inputs and outputs.
- Shape formulas.
- Dtype rules.
- Layout and memory assumptions.
- Mesh placement.
- Numerical tolerances.
- Pseudocode matching the PyTorch semantics.
- Intended TTNN decomposition.
- Possible fused or custom op replacement.
- Known edge cases.

Use subagents to verify each spec:

- One subagent checks logical equivalence to PyTorch.
- One subagent checks shape and dtype contracts.
- One subagent checks whether the proposed physical contract creates avoidable
  data movement.
- One subagent checks whether the operation should become a custom op.

Require each subagent to state the exact line or formula it is relying on. Treat
"looks right" as a failed review.

### Phase 3: Choose the Whole-Model Mesh Plan

Choose the mesh plan before module integration. Decide:

- Tensor parallelism for dense projections and LM head.
- Expert parallelism for routed experts.
- Sequence parallelism for prefill if sequence length dominates.
- Data parallelism only if it is part of the serving plan.
- Residual stream ownership after every layer.
- Attention cache ownership and update location.
- Router output ownership.
- Expert dispatch and combine CCL boundaries.
- Where all-gather, reduce-scatter, all-to-all, and point-to-point transfers
  occur.
- Which tensors are replicated, sharded, or transient.

Write down two plans if prefill and decode differ. They may share weights while
using different activation and cache contracts.

A useful starting pattern is:

- Decode: prioritize cache locality, low latency, L1-sharded activations, and
  minimal synchronization.
- Prefill: prioritize large matmul throughput, sequence parallelism or chunking,
  and bounded activation memory.
- MoE: route tokens to expert owners early, do expert compute near owned
  weights, and combine only the result needed by the residual stream.

Validate memory feasibility before writing modules. For each weight group and
activation class, estimate:

- Per-chip DRAM.
- Per-core L1 pressure.
- Persistent cache footprint.
- Temporary tensor footprint.
- Collective buffer footprint.

### Phase 4: Convert and Validate Weights

Weight conversion is a compiler pass. Treat it as a first-class component, not
as setup glue.

The converter should:

- Load the source checkpoint.
- Validate model config and inference config.
- Map every source key to a destination semantic key.
- Split dense, expert, cache, and auxiliary weights if useful.
- Apply quantization, packing, transposition, tilization, and sharding.
- Write a manifest with source key, destination key, source shape, destination
  shape, dtype, quantization format, layout, mesh placement, and content hash.
- Reject missing keys, duplicate keys, unknown keys, or unsupported config
  variants.
- Detect incomplete Git LFS pointer files before expensive conversion.

Some staged bringups begin with a high-level converter manifest or a selective
real-checkpoint loader rather than a full physical per-tensor manifest. That is
fine for smoke bringup. Do not replace working converter code just to satisfy the
final manifest shape. Add a per-tensor manifest layer around it, or extend the
existing manifest incrementally, once a tensor enters the maintained TTNN path.

Use `models/common/modules/lazy_weight.py` when it helps delay conversion and
cache converted weights. If correctness depends on source contents rather than
shape alone, add or layer on content hashing in the model-specific converter.

Weight tests should cover:

- Key mapping with tiny synthetic checkpoints.
- Round-trip pack/unpack for quantized formats.
- Dequantization error bounds.
- Shape and dtype validation.
- Per-mesh sharding expectations.
- Manifest stability.

### Phase 5: Bring Up Modules

For each semantic module, follow this ladder:

1. PyTorch or CPU reference test.
2. Optimized pseudocode test against the reference.
3. Decomposed TTNN test on one device if the operation fits and the result is
   useful for debugging.
4. Final-topology TTNN test with conservative memory configs and program
   configs.
5. Final-topology TTNN test with intended sharding, CCL, and cache layout.
6. Fused or custom op test, if needed.
7. Trace-compatible test with preallocated persistent tensors.

For large modules, skip step 3 if it creates a false implementation direction.
It is fine to use a single-device decomposition only as a microscope and throw
it away.

Use `models/common` patterns:

- Use lightweight modules to keep module construction explicit.
- Use lazy weights to avoid eager conversion and to cache physical weights.
- Use validation helpers for PCC and tensor comparisons.
- Use auto-composition helpers where they reduce host/device boilerplate.
- Use distribution helpers to make mesh placement visible in tests.
- Keep static decisions out of `forward`; choose them in construction or config.

The module test must assert the full tensor contract, not only PCC. A passing
PCC with the wrong memory config or mesh placement can still make the full model
unusable.

### Phase 6: Decide Custom Ops Deliberately

Decomposed TTNN is a correctness tool, not always a performance target. Promote
an operation to custom-op work when any of these are true:

- The decomposition creates large intermediate tensors.
- The decomposition requires avoidable activation resharding.
- The operation needs a layout that existing TTNN ops do not naturally emit.
- The operation is a CCL boundary.
- The operation is cache-update-heavy.
- The operation is repeatedly on the critical path.
- Existing op composition prevents trace capture or stable preallocation.

Before requesting or writing a custom op, produce an op contract:

- Semantic name.
- Mathematical definition.
- Inputs, outputs, and in-place behavior.
- Shape formulas.
- Dtypes and quantization.
- Supported layouts.
- Required memory configs.
- Mesh and shard ownership.
- Program config knobs.
- Persistent buffers.
- Trace compatibility.
- Error tolerance and reference implementation.
- Minimal tests and performance target.

Keep the decomposed lowering as a reference unless it is impossible to maintain.
When a custom op changes the physical or logical contract, update the lowering
spec and all downstream contracts before wiring it into the model.

Examples from current code:

- MLA is not ordinary Q/K/V attention. It changes cache and attention output
  contracts and should be modeled as its own semantic op.
- `all_to_all_dispatch` and `all_to_all_combine` are not just faster copies.
  They define MoE routing metadata and ownership boundaries.
- Grouped gate and gate-matmul style ops may preserve math while changing
  physical output format, which affects downstream expert dispatch.

Existing TTNN custom ops usually have this anatomy:

- Public C++ API: `<op>.hpp` and `<op>.cpp`.
- Python binding: `<op>_nanobind.hpp` and `<op>_nanobind.cpp`.
- Device operation contract: `device/<op>_device_operation.hpp`.
- Attribute and tensor args, sometimes split into `*_types.hpp`.
- Program factory: `device/<op>_program_factory.cpp`.
- Device kernels under `device/kernels/`.
- Python or C++ tests that compare against a reference and assert output shape,
  dtype, layout, and memory config.

Good examples to inspect are:

- `ttnn/cpp/ttnn/operations/experimental/reduction/deepseek_grouped_gate`
- `ttnn/cpp/ttnn/operations/experimental/deepseek/moe/moe_gate_mm`
- `ttnn/cpp/ttnn/operations/ccl/all_to_all_dispatch`
- `ttnn/cpp/ttnn/operations/ccl/all_to_all_combine`

When deciding whether to reuse a DeepSeek V3 op for a V4 model:

- Reuse directly only if the semantic and physical tensor contracts match.
- Wrap or adapt if the math matches but shapes, padding, route metadata, or
  memory ownership differ.
- Version or replace the op if downstream tensor contracts change.
- For CCL ops, treat the metadata tensor as part of the ABI. A different
  dispatch metadata format is a different contract even if the collective
  pattern looks similar.
- For router/gate ops, check scoring, bias, grouping, top-k, normalization, and
  hash-routing before assuming compatibility.

### Phase 7: Integrate the Full Model

Integrate in this order:

1. Embedding and first normalization.
2. One layer in prefill with fake or tiny weights.
3. One layer in decode with fake or tiny weights.
4. Layer stack with tiny dimensions.
5. Full-shape single layer.
6. Full-shape selected layers.
7. Full model without trace.
8. Full model with trace.
9. Full model with real weights.

Use deterministic inputs and capture intermediate checkpoints at semantic
boundaries:

- Residual before and after norm.
- Attention input/output.
- Cache before and after update.
- Router scores and selected experts.
- Dispatch metadata.
- Expert outputs.
- Combine outputs.
- MLP output.
- Final logits.

When a mismatch appears, bisect by semantic boundary first, then by TTNN op.
Avoid debugging a full traced model before the untraced semantic boundaries are
known to match.

### Phase 8: Trace and Runtime Bringup

Trace is a runtime contract. It is not a last-minute wrapper.

For every traced entry point, define:

- Static batch size.
- Static sequence length or chunk size.
- Static decode token shape.
- Cache tensor addresses.
- Persistent input and output tensors.
- Preallocated temporary tensors if required.
- Program cache warmup sequence.
- Host synchronization points.
- Two-command-queue event ordering.

Bring up trace in three steps:

1. Untraced run with final tensor contracts.
2. Captured trace replay with the same tensors and inputs.
3. Captured trace replay with changing input contents but stable addresses.

Verify trace and non-trace outputs against the same reference. If trace changes
results, suspect stale tensor addresses, missing synchronization, dynamic
allocation, shape drift, or mutated persistent state before suspecting math.

### Phase 9: Optimize Performance

Only tune after the tensor contracts are stable. Otherwise you may optimize a
layout that the full model cannot use.

Performance work should be organized by bottleneck:

- Dense matmul throughput.
- Attention or MLA throughput.
- Router and top-k overhead.
- Expert dispatch/combine CCL time.
- Expert compute utilization.
- Activation resharding.
- DRAM bandwidth.
- L1 pressure.
- Host overhead.
- Trace replay overhead.

For every optimization, record:

- Baseline command and result.
- Changed config or code.
- Expected effect.
- Measured effect.
- Accuracy impact.
- Whether it changes a tensor contract.

Prefer optimizations that remove movement or intermediates before tuning small
program-config details. A fused op that eliminates a large intermediate usually
beats heroic tuning of a bad decomposition.

### Phase 10: Debugging Discipline

Always classify a failure before fixing it:

- Spec failure: the pseudocode does not match PyTorch.
- Converter failure: the physical weight is wrong.
- Contract failure: shape, dtype, layout, memory config, or mesh placement is
  wrong.
- Decomposition failure: the TTNN sequence does not implement the spec.
- Op failure: a TTNN or custom op misbehaves.
- Runtime failure: trace, cache, synchronization, or allocation changed
  behavior.
- Performance failure: the result is correct but the physical plan is bad.

Use the narrowest available test:

- CPU reference tests for semantics.
- Tiny synthetic TTNN tests for shape and dtype.
- Real-weight module tests for numerical stability.
- Final-topology tests for sharding and CCL.
- Trace tests for runtime.
- Full-model tests for integration.

For hard bugs, use subagents with narrow assignments:

- One subagent reads the reference and explains expected math.
- One subagent inspects tensor contracts and mesh placement.
- One subagent searches for known TTNN op limitations.
- One subagent reviews the custom op or CCL code path.
- One subagent checks whether the test itself is invalid.

Do not let every reviewer inspect everything. Independent narrow reviews catch
different classes of mistakes.

### Phase 11: CI and Test Matrix

Create tests in layers:

- Stage 0: config, key mapping, conversion, quantization, manifests.
- Stage 1: CPU semantic modules.
- Stage 2: TTNN module tests with synthetic weights.
- Stage 3: TTNN module tests with real weights.
- Stage 4: final-topology multi-device tests.
- Stage 5: trace equivalence tests.
- Stage 6: tiny full-model tests.
- Stage 7: real-checkpoint smoke tests.
- Stage 8: performance regression tests.

Every test should state:

- Hardware requirement.
- Runtime expectation.
- Whether it needs real weights.
- Exact tolerance.
- Whether it is deterministic.
- What semantic boundary it protects.

Prefer many small tests with explicit contracts over one expensive end-to-end
test that only says "logits are bad".

## Lessons From Existing Approaches

### `models/tt_transformers`

Useful lessons:

- It demonstrates real multi-device LLM execution.
- It contains valuable patterns for prefill, decode, KV cache handling, tracing,
  and runtime orchestration.
- It shows how much static shape and program-cache discipline matters.

Mistakes to avoid:

- Too much behavior is coupled through broad model config and generator logic.
- Prefill, decode, tracing, sampling, and model structure can become difficult
  to reason about independently.
- The abstraction is large enough that new model bringup inherits incidental
  complexity.

Use it as a reference for runtime realities, not as the default substrate for a
new architecture.

### `models/demos/gpt_oss`

Useful lessons:

- Mesh configuration is explicit and readable.
- TP, EP, and SP are visible concepts.
- The code is closer to a modern bringup shape than older monolithic stacks.

Mistakes to avoid:

- Some reuse of older transformer infrastructure leaks complexity.
- The semantic-op layer is not cleanly separated enough to serve as a general
  lowering pipeline.

Copy the mesh explicitness. Avoid copying inherited coupling.

### `models/demos/deepseek_v3`

Useful lessons:

- The model separates config, conversion, state creation, run config, and
  forward execution more clearly than older examples.
- MLA and MoE forced the code toward custom semantic boundaries.
- Custom CCL and custom gate work show the right instinct: if the model's real
  operation does not fit the generic pattern, make the operation explicit.

Mistakes to avoid:

- There is still substantial model-specific boilerplate.
- Some contracts live implicitly in code rather than as reviewable artifacts.
- Custom op integration can become hard to test if the decomposed and semantic
  references are not preserved.

Use it as evidence that DeepSeek-like models need custom operations and explicit
contracts. Do not treat every local abstraction as the final shape.

## Per-Model Dossier

The guide should not contain the answers for a specific model. It should force
the bringup owner to create a small, reviewable dossier for the target model
before TTNN implementation starts. That dossier is the bridge between "generic
guide" and "Gemma, Llama, Qwen, Mixtral, DeepSeek, or some future architecture."

The first milestone for any model is not "full model runs." It is:

1. Pin the exact checkpoint config and source revisions.
2. Write `source_of_truth.md` with line-anchored formulas from those sources.
3. Write `model_dossier.md` using the tables below.
4. Write `lowering_specs/` for every red-zone semantic operation.
5. Write a tiny PyTorch reference harness with checkpointed intermediate tensors.
6. Write a whole-model mesh and memory plan based on the selected config.
7. Only then, bring up first TTNN modules using `models/common`.

### Source Selection

Start the dossier with:

```text
Target model:
Checkpoint:
Checkpoint config:
Primary source implementation:
Secondary source implementation:
Serving mode:
Out of scope for first milestone:
```

Do not mix defaults from different implementations. A HuggingFace `Config`, a
vendor inference script, a paper pseudocode block, and a converted checkpoint can
all disagree. The selected checkpoint config wins. Secondary references are used
for cross-checking and explanation, not silent substitution.

### Architecture Skeleton

Fill this table from the selected source:

| Area | Source lines | Contract summary | First milestone scope |
| --- | --- | --- | --- |
| Embedding | | | |
| Attention | | | |
| Cache | | | |
| MLP or MoE | | | |
| Norm and residual | | | |
| Output head | | | |
| Optional heads | | | |
| Quantization | | | |
| Distributed reference behavior | | | |

Every row must cite the source lines or formulas. Empty source lines mean the
row is not ready for TTNN code.

### Red-Zone Decisions

For each red-zone item, write:

| Semantic op | Why it is unusual | Reference function | First lowering | Final-risk guess |
| --- | --- | --- | --- | --- |
| | | | torch/decomposed/custom | |

Examples of red-zone items include sparse attention, sliding-window cache,
attention sinks, grouped output projections, MoE routing, MTP, nonstandard RoPE,
parallel residual streams, block quantization, and custom kernels in the source
implementation. The examples are not the checklist; the source code determines
the checklist.

### First-Target Choice

The first TTNN target should be chosen by a written rule, not by habit:

| Candidate | Pros | Cons | Choose when |
| --- | --- | --- | --- |
| Decode only | Small sequence shape, exposes cache and trace issues early | May hide prefill memory and matmul issues | Serving latency is the first goal |
| Prefill only | Exposes large matmul, sequence, and memory pressure early | Can delay cache and trace issues | Throughput or long context is the first goal |
| One layer only | Fastest module correctness path | Can hide stack-level memory and CCL issues | Architecture has many repeated layers |
| One layer per type | Covers per-layer schedules | More test code upfront | Model has alternating or special layers |
| Tiny full model | Exposes integration flow | Can be misleading for full topology | Config can be faithfully shrunk |
| Real-weight slice | Finds converter and numeric issues | Needs checkpoint access | Converter risk is high |

For the first milestone, explicitly state what is excluded. Examples:

- MTP or speculative heads.
- Full-vocab logits.
- Packed quantized device compute.
- Dynamic batching.
- Full prefill length.
- Device-side sampling.
- Production CCL layout.

Exclusion is acceptable only if the omitted feature has its own later semantic
op and tests. Do not omit a feature that changes the layer output being compared
unless the test is explicitly a scaffold.

### Cache And State Contracts

Before TTNN code, fill one row per persistent or mode-dependent state:

| State | Producer | Consumer | Prefill shape/meaning | Decode shape/meaning | Placement candidate | Trace mutability |
| --- | --- | --- | --- | --- | --- | --- |
| KV cache | | | | | | |
| Attention mask or row ids | | | | | | |
| Router metadata | | | | | | |
| Expert dispatch metadata | | | | | | |
| Recurrent/residual state | | | | | | |

Add model-specific rows as needed. If a model has compressed caches, multiple
cache streams, sliding rings, query-dependent row selection, recurrent state, or
prefix state, each one needs a separate row. A combined "cache" row is not
enough.

### Weight Map

Before converter code, fill:

| Group | Source key pattern | Logical shape | Physical TTNN layout | Quantization | First milestone handling |
| --- | --- | --- | --- | --- | --- |
| Embedding | | | | | |
| Attention query | | | | | |
| Attention key/value | | | | | |
| Attention output | | | | | |
| Norms | | | | | |
| MLP or experts | | | | | |
| Router | | | | | |
| Output head | | | | | |
| Optional heads | | | | | |

The source model determines whether these groups split further. For a dense
Gemma-like model, the table may stay small. For a routed or compressed model,
the same table may expand into expert, scale, cache, indexer, and auxiliary
groups. The guide's job is to make the expansion inevitable, not to prefill it.

### Quantization Decision

Before packed-device compute, write:

- Source checkpoint dtype and packing.
- Scale tensor dtype, granularity, and shape.
- Whether the first milestone dequantizes to BF16 for correctness.
- Which final device op is expected to consume the packed format directly.
- Numeric tolerance for dequantized, simulated, and packed-device paths.

If the model is BF16 or FP16 only, record that and move on.

### `models/common` Reuse Decision

For each `models/common` module or helper, decide:

| Helper | Reuse directly | Use as pattern only | Do not use | Reason |
| --- | --- | --- | --- | --- |
| `attention_1d.py` | | | | |
| `mlp_1d.py` | | | | |
| `tt_ccl.py` | | | | |
| `LazyWeight` | | | | |
| validation helpers | | | | |
| distribution helpers | | | | |

Do not force a target architecture into a common module just because the module
exists. A common attention helper is reusable only if its semantic and physical
contracts match the target model. Otherwise use it as a pattern for lifecycle,
tests, and weight handling.

### Custom-Op Candidates

For every operation that looks performance-critical or physically awkward, fill:

| Semantic op | Decomposed reference exists? | Contract-changing? | Existing TTNN op to study | Ops-team question |
| --- | --- | --- | --- | --- |
| | | yes/no | | |

The dossier should distinguish "fuse this for speed" from "this op defines a
new tensor contract." The second case affects upstream and downstream module
design and must be decided before final-topology tests.

### Subagent Dossier Prompts

Use subagents to create and verify the dossier:

- Source verifier: read the selected source implementation and fill the
  architecture skeleton with line-anchored formulas. Do not mention TTNN.
- Contract author: convert the source verifier's output into tensor contracts
  for prefill, decode, cache, weights, and optional heads.
- Topology reviewer: propose a mesh plan from the tensor contracts and memory
  estimates. Identify unavoidable and avoidable data movement.
- Common-module reviewer: inspect `models/common` and state which helpers are
  reusable directly, which are patterns only, and which would be misleading.
- Custom-op reviewer: classify red-zone operations as decomposed, fusion, or
  contract-changing custom ops.
- Test-plan reviewer: turn the dossier into a staged test matrix with fake,
  tiny, real-slice, final-topology, and trace tests.

The subagents should not need model-specific facts from this guide. They should
derive those facts from the selected model sources and write them into the
dossier.

### Generic Confidence Test

To test whether this guide is generic, give a fresh subagent:

1. This guide.
2. One target model source, such as Gemma, Qwen, Llama, Mixtral, DeepSeek V4, or
   a multimodal LLM.
3. The current tt-metal repo.

Ask it not to implement anything. Ask it to produce:

- The first page of `source_of_truth.md`.
- The architecture skeleton table.
- The red-zone inventory.
- The first-target choice and exclusions.
- The cache/state table.
- The weight-map groups.
- A statement of whether it can now create a concrete implementation plan.

The guide passes only if the subagent can create the missing per-model
information itself. If the subagent says it is confident only because the guide
already contains target-model facts, the guide has failed the genericity test.

## Suggested Directory Shape

A clean model directory should look roughly like:

```text
models/demos/<model_name>/
  MODEL_CARD.md
  README.md
  config.py
  mesh_plan.md
  tensor_contracts.json
  trace_signatures.md
  converter.py
  weight_manifest.py
  cpu_reference.py
  lowering_specs/
    attention_prefill.md
    attention_decode.md
    router.md
    moe_dispatch.md
    moe_combine.md
    expert_compute.md
    hyperconnection.md
  semantic_ops.py
  lowerings/
    torch_ref.py
    ttnn_decomposed.py
    ttnn_final_topology.py
    ttnn_custom_ops.py
  runtime/
    state.py
    trace.py
    decode.py
    prefill.py
  tests/
    test_stage0_converter.py
    test_stage1_cpu_reference.py
    test_stage2_lowering_specs.py
    test_stage3_ttnn_modules.py
    test_stage4_mesh_contracts.py
    test_stage5_trace.py
    test_stage6_tiny_model.py
```

This is a guideline, not a required framework. Keep files smaller if the model
does not need every layer.

## Definition of Done

A complex model bringup is ready for optimization when:

- Every red-zone semantic operation has a lowering spec.
- Subagents have reviewed the specs for PyTorch equivalence and tensor-contract
  completeness.
- The converter has manifest tests and rejects unsupported checkpoints.
- Each module has reference, decomposed, and final-topology tests as applicable.
- Custom ops have explicit contracts and reference tests.
- Prefill and decode have separate execution plans if their optimal contracts
  differ.
- Shared weights have one canonical device layout unless duplication is
  deliberately justified.
- The full model runs untraced with final tensor contracts.
- Trace replay matches untraced output for the same static signature.
- Performance work is recorded against stable baselines.

## Subagent Review Prompts

Use prompts like these during bringup.

### Pseudocode Equivalence Review

Read `<source torch file>` and `<lowering spec>`. Do not implement anything.
Verify whether the pseudocode computes the same function as the PyTorch source
for all supported config variants. Report exact mismatches, missing edge cases,
and any assumptions that require tests. Cite source lines or formulas.

### Tensor Contract Review

Read `<lowering spec>`, `<mesh_plan>`, and relevant TTNN module code. Do not
implement anything. Check every input, output, intermediate, cache update, CCL,
and weight for shape, dtype, layout, memory config, sharding, and mesh ownership.
Report any contract that is ambiguous or likely to force avoidable data
movement.

### Custom Op Readiness Review

Read `<lowering spec>` and the decomposed TTNN implementation. Do not implement
anything. Decide whether this operation should remain decomposed, be fused
without changing the external contract, or become a physical/logical custom op.
Report the proposed op contract and the performance risk if it remains
decomposed.

### Trace Readiness Review

Read `<runtime trace plan>` and the TTNN forward path. Do not implement
anything. Verify that shapes, tensor addresses, persistent buffers, program cache
warmup, and synchronization are compatible with trace replay. Report dynamic
allocation or shape drift risks.

### Fresh Bringup Confidence Review

Read this guide, the target model directory, and the relevant tt-metal reports.
Do not implement anything. State whether you could now write a concrete bringup
plan and start implementation. If not, list the missing information, ambiguous
instructions, or repo areas where you would not know how to proceed.
