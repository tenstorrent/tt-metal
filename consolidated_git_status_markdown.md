# Consolidated Git Status Markdown

This generated document combines the 21 Markdown files approved from `git status`.

The source bodies are reproduced verbatim inside each source section. The table of contents, grouping headers, navigation links, and begin/end markers are generated organization only.

## Source Index

### Root

1. [`chunked_prefill_report.md`](#source-01-chunked-prefill-report-md)
### Common Model Plans

2. [`models/common/models/ALWAYS_PAGED_ATTENTION_PLAN.md`](#source-02-models-common-models-always-paged-attention-plan-md)
3. [`models/common/models/CLAUDE.md`](#source-03-models-common-models-claude-md)
4. [`models/common/models/EXECUTOR_TODOS.md`](#source-04-models-common-models-executor-todos-md)
5. [`models/common/models/EXECUTOR_TRACE_SPLIT_PLAN.md`](#source-05-models-common-models-executor-trace-split-plan-md)
6. [`models/common/models/RUN_PERF_BENCHMARK_PLAN.md`](#source-06-models-common-models-run-perf-benchmark-plan-md)
### Llama 3.1 8B Notes

7. [`models/common/models/llama3_8b/DEBUG.md`](#source-07-models-common-models-llama3-8b-debug-md)
8. [`models/common/models/llama3_8b/EXECUTOR_ARCHITECTURE.md`](#source-08-models-common-models-llama3-8b-executor-architecture-md)
9. [`models/common/models/llama3_8b/EXECUTOR_REFACTOR_PLAN.md`](#source-09-models-common-models-llama3-8b-executor-refactor-plan-md)
10. [`models/common/models/llama3_8b/HF_ADAPTOR_PATTERN.md`](#source-10-models-common-models-llama3-8b-hf-adaptor-pattern-md)
11. [`models/common/models/llama3_8b/MODEL_METHOD_SOLUTION.md`](#source-11-models-common-models-llama3-8b-model-method-solution-md)
12. [`models/common/models/llama3_8b/PLAN.md`](#source-12-models-common-models-llama3-8b-plan-md)
13. [`models/common/models/llama3_8b/STATIC_KV_CACHE_PLAN.md`](#source-13-models-common-models-llama3-8b-static-kv-cache-plan-md)
14. [`models/common/models/llama3_8b/Sampling Notes.md`](#source-14-models-common-models-llama3-8b-sampling-notes-md)
15. [`models/common/models/llama3_8b/TODO.md`](#source-15-models-common-models-llama3-8b-todo-md)
16. [`models/common/models/llama3_8b/TRACE_FULL_GRAPH_INVESTIGATION.md`](#source-16-models-common-models-llama3-8b-trace-full-graph-investigation-md)
17. [`models/common/models/llama3_8b/hf_adaptor_refactor_goals.md`](#source-17-models-common-models-llama3-8b-hf-adaptor-refactor-goals-md)
18. [`models/common/models/llama3_8b/perf_diff.md`](#source-18-models-common-models-llama3-8b-perf-diff-md)
19. [`models/common/models/llama3_8b/perf_results.md`](#source-19-models-common-models-llama3-8b-perf-results-md)
20. [`models/common/models/llama3_8b/tttv2_decoupling_goals.md`](#source-20-models-common-models-llama3-8b-tttv2-decoupling-goals-md)
### Common Tests

21. [`models/common/tests/traced_executor_can_trace_plan.md`](#source-21-models-common-tests-traced-executor-can-trace-plan-md)

---

## Working Plan: `Llama3ForCausalLM` Product Boundary

### Product Definition

Treat `Llama3ForCausalLM` as the usable ML model product, not just a wrapper around the accelerator tensor graph.

The product model is:

- architecture and TT tensor graph
- trained weights and weight loading policy
- tokenizer and vocabulary behavior
- special tokens and stop token behavior
- chat template and prompt formatting
- generation and sampling defaults
- KV cache and execution policy
- runtime metadata needed by executors and demos

The inner `Llama3Transformer1D` remains the TTTv2 tensor module. It should own the device-side causal transformer: embeddings, RoPE, transformer blocks, final norm, LM head, sampling module, memory configs, dtypes, sharding, CCL wiring, and cache paths needed for tensor materialization.

The outer `Llama3ForCausalLM` owns the language-model product contract: text in, tokens/logits/text out.

### Proposed Layering

```text
Llama3ForCausalLM
  tokenizer
  generation_config
  runtime_config
  transformer: Llama3Transformer1D
  executor: EagerLlamaExecutor | TracedLlamaExecutor

Llama3Transformer1D
  config: Llama3Transformer1DConfig
  embedding
  rope_setup
  layers
  norm
  lm_head
  sampling
```

The important boundary:

```text
Llama3Transformer1D = compute graph
Llama3ForCausalLM = usable language model
```

### Tokenizer Ownership

The tokenizer is part of the model product in the ML sense. A deployed Llama 3.1-8B model is not just weights plus architecture. It also includes the vocabulary, tokenization rules, special token IDs, stop tokens, chat template, and text decoding behavior.

However, the tokenizer should not be stored inside `Llama3Transformer1DConfig`. That config is currently a device tensor construction config, and it should not grow text I/O concerns.

Put tokenizer ownership on `Llama3ForCausalLM`, with documentation that the tokenizer object must provide:

- `encode(text, add_special_tokens=False) -> list[int]`
- `decode(token_ids, skip_special_tokens=True) -> str`
- access to EOS and stop token IDs
- chat-template application for instruct models, either directly or through a `Llama3ForCausalLM.encode_chat(...)` helper

Do not introduce a `Protocol` yet. Keep the interface documented in prose until there are multiple tokenizer implementations or a stronger static typing need.

The HuggingFace tokenizer should be treated as one backend implementation of this documented tokenizer interface. A future native TTTv2 tokenizer can satisfy the same documented behavior without changing the `Llama3ForCausalLM` product API.

### Runtime And Generation Configs

Split the current runtime bag into smaller concepts.

`Llama3GenerationConfig` should hold text generation and sampling defaults:

- max decode tokens
- temperature
- top-k
- top-p
- stop token IDs
- default on-device sampling mode, if kept as product behavior

`Llama3RuntimeConfig` should hold executor/runtime metadata:

- model name
- model cache path
- mesh cluster shape
- max prefill chunk size
- paged attention config
- trace-supported prefill lengths
- trace eligibility logic such as `can_enable_trace(prefill_seq_len, num_cached_tokens)`

`Llama3Transformer1DConfig` should keep structural and device tensor fields:

- number of layers
- vocab size
- max batch size
- max sequence length
- mesh device
- submodule configs
- memory configs
- activation dtypes
- CCL instance
- tensor weight cache path

### Removing `HFLlama3RuntimeContext`

`HFLlama3RuntimeContext` can go away because it mixes three separate layers:

- HF/text frontend: tokenizer, instruct flag, prompt encoding
- executor runtime: model cache path, cluster shape, max prefill chunk size, paged attention config
- model structure: layer count, head dim, KV head count, max batch size, max sequence length

The replacement should move those fields to their natural owners:

- `tokenizer`, `instruct`, and prompt/chat encoding move to `Llama3ForCausalLM`
- `model_cache_path`, `cluster_shape`, `max_prefill_chunk_size`, and `paged_attention_config` move to `Llama3RuntimeConfig`
- `n_layers`, `max_batch_size`, and `max_seq_len` are read from `model.config`
- `head_dim` and `n_kv_heads` are read from the attention config in `model.config.block_configs`
- `max_context_len` should become explicit product metadata, not an accidental `model_info.__getattr__` lookup

### Loader Shape

`hf_adaptor.py` should become a loader/backend, not the conceptual owner of runtime state.

Desired shape:

```text
HF checkpoint/config/tokenizer
  -> build Llama3Transformer1DConfig
  -> build Llama3Transformer1D
  -> wrap tokenizer behind documented tokenizer behavior
  -> build Llama3RuntimeConfig
  -> return Llama3ForCausalLM
```

The public product constructor should eventually feel like:

```python
llm = Llama3ForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct",
    mesh_device=mesh_device,
    max_batch_size=32,
    max_seq_len=1024,
)

outputs = llm.generate(["What is your favorite condiment?"], max_new_tokens=128)
```

Lower-level users should still be able to bypass text and work with tokens/logits:

```python
logits = llm.forward_tokens(tokens, page_table=page_table, kv_cache=kv_cache)
```

### Migration Notes

Keep the first implementation conservative:

1. Introduce `Llama3ForCausalLM` as the product boundary.
2. Keep `Llama3Transformer1D` unchanged as the inner tensor module.
3. Document the tokenizer interface instead of adding a `Protocol`.
4. Move executor-only metadata out of `HFLlama3RuntimeContext`.
5. Update demos and generators to use the product object where they need text behavior, and `model.config` where they need structural tensor metadata.
6. Remove `HFLlama3RuntimeContext` once all consumers stop depending on its mixed field bag.

Success criteria:

- tokenizer is recognized as part of the model product
- tokenizer is not mixed into the tensor graph config
- HF remains a loading backend, not the product abstraction
- executor runtime metadata is explicit and narrow
- structural model information comes from `Llama3Transformer1DConfig`
- demos stop relying on `model_info.__getattr__` as an implicit metadata escape hatch

---

## Root

<a id="source-01-chunked-prefill-report-md"></a>

### Source 01: `chunked_prefill_report.md`

[Back to Source Index](#source-index) | Previous: none | [Next: `models/common/models/ALWAYS_PAGED_ATTENTION_PLAN.md`](#source-02-models-common-models-always-paged-attention-plan-md)

<!-- BEGIN VERBATIM: chunked_prefill_report.md -->
# Chunked Prefill Report

## Question

Is `chunked_prefill` in `models/tt_transformers/tt/generator.py` really necessary? What problem does it solve? Why does chunking solve that problem?

## Short Answer

`chunked_prefill` is not universally necessary.

It is necessary in this codebase when either:

1. The prompt length exceeds `max_prefill_chunk_size`.
2. Prefix caching is in use (`num_cached_tokens > 0`).

For short prompts with no cached prefix, the normal non-chunked prefill path is used and `chunked_prefill` is not needed.

## Exact Trigger In Code

The decision is made in `models/tt_transformers/tt/generator.py` inside `prefill_forward_single_user_text()`:

```python
seq_len = tokens.shape[-1]
use_chunked_prefill = seq_len > self.model_args[model_id].max_prefill_chunk_size
use_prefix_caching = num_cached_tokens > 0
if use_chunked_prefill or use_prefix_caching:
```

That means:

- Long prompt -> chunked path
- Cached prefix present -> same chunked/page-table path
- Otherwise -> regular prefill path

## What Problem It Solves

### 1. Long-prefill memory / kernel limits

The codebase explicitly treats prefill chunk size as a model/device-specific compatibility limit.

The limit is configured in `models/tt_transformers/tt/model_config.py` via `get_max_prefill_chunk_size()`. For unknown model/device combinations, the code emits this warning:

```python
logger.warning(
    f"Try setting MAX_PREFILL_CHUNK_SIZE to larger powers of 2 up to e.g. 128 for faster performance (if you run out of L1 memory it was too high)"
)
```

This is the clearest code-level statement that larger one-shot prefills can exceed L1-memory capacity, and that `max_prefill_chunk_size` is the knob used to stay within that limit.

The public docs say the same thing more generally in `models/tt_transformers/README.md`:

- chunked prefill is used to support large max context lengths
- some model/device combinations have smaller supported limits due to memory constraints

So the first concrete problem is:

**A full prompt prefill in one shot can exceed the device/kernel working-set limits.**

### 2. Prefix caching changes the attention problem

With prefix caching, the input queries are only for the uncached suffix, but attention must still see the entire prior prefix via KV cache.

The regular prefill path uses standard `scaled_dot_product_attention()` over the current prompt chunk tensors directly.

The cached/chunked path instead:

- writes new K/V into paged KV cache
- uses page tables
- runs `chunked_scaled_dot_product_attention()` with `chunk_start_idx`

This is not just an optimization. It is the mechanism that makes "new query tokens attending over cached prefix + newly written suffix" work correctly with paged KV cache.

So the second concrete problem is:

**Once prefix caching exists, prefill is no longer a simple single dense Q/K/V attention over one contiguous fresh tensor.**

## Why Chunking Solves It

Chunking solves the long-prefill problem by reducing the active working set from "entire prompt at once" to "one chunk at a time".

Mechanically, the chunked path in `generator.py` does this:

1. Split the uncached suffix into chunk-sized token ranges.
2. For each chunk, compute chunk-local inputs and rotary state.
3. Fill that chunk's K/V into paged KV cache using `paged_fill_cache`.
4. Run attention for the chunk using:
   - the global `page_table`
   - the current `chunk_page_table`
   - `chunk_start_idx` so the kernel knows the chunk's absolute position in the full sequence
5. Return the chunk output corresponding to the chunk that contains the requested last token.

This preserves full-sequence semantics while keeping the expensive live tensors bounded by chunk size.

In other words:

- **Semantics** stay global because the cache and page table represent the whole sequence.
- **Working set** stays local because only one chunk is materialized and processed at a time.

## Why `chunk_start_idx` Matters

`chunk_start_idx` is not cosmetic. It tells the chunked SDPA kernel where this chunk lives in the full prompt.

In `ttnn/cpp/ttnn/operations/transformer/sdpa/device/sdpa_program_factory.cpp`:

```cpp
// In chunked prefill mode, the offset of Q in terms of Q chunks
uint32_t chunked_q_chunk_offset = 0;
...
// chunk_start_idx must be a multiple of q_chunk_size
chunked_q_chunk_offset = chunk_start_idx.value() / q_chunk_size;
```

That is how the kernel keeps causal masking and sequence positioning aligned with the full prompt instead of treating each chunk as position 0.

## Why Prefix Caching Reuses The Chunked Path

In `generator.py`, prefix caching is intentionally handled through the same machinery as chunked prefill:

- `use_prefix_caching = num_cached_tokens > 0`
- same page-table requirements
- same chunk/page-table preparation path
- same chunked SDPA path

That makes sense because both features need paged attention semantics:

- long prompt chunking: process long sequences incrementally
- prefix caching: process only the uncached suffix while attending over cached prefix

They are distinct user-visible features, but internally they rely on the same paged-cache/chunked-attention machinery.

## Why Paged Attention Is Required

The chunked path asserts that both `page_table` and `kv_cache` are present:

```python
assert page_table is not None, "page_table must be provided for chunked prefill"
assert kv_cache is not None, "kv_cache must be provided for chunked prefill"
```

That is because the implementation depends on:

- `paged_fill_cache(...)` for writing chunk K/V into the global cache
- `page_table` for mapping logical blocks to physical cache blocks
- chunk-aware SDPA over cached K/V

Without paged attention, chunked prefill as implemented here cannot work.

## Chunk Size Selection

Chunk size is not chosen arbitrarily. `models/tt_transformers/tt/common.py` uses `get_max_prefill_chunk_size()` to choose the largest multiple of 2048 that:

- divides the sequence length
- is less than or equal to the configured max chunk size

This avoids ragged tail chunks and keeps the chunking aligned with kernel expectations.

## Trace Limitation

Prefill tracing is currently not intended for chunked or prefix-cached prefill.

`models/tt_transformers/tt/model_config.py` contains:

```python
# TODO: Support chunked prefill with tracing
# TODO: Support prefix caching with tracing
```

And `can_enable_trace()` only allows prefill tracing when:

- `prefill_seq_len <= max_prefill_chunk_size`
- `num_cached_tokens == 0`

So chunked prefill is primarily a capability/correctness path for long context and cached-prefix handling, not the main traced fast path.

## Final Conclusion

`chunked_prefill` is necessary in this implementation for two real reasons:

1. **Long prompts**: one-shot prefill can exceed practical kernel / L1-memory limits.
2. **Prefix caching**: the uncached suffix must attend over cached K/V using paged attention semantics.

Chunking solves both by:

- processing only a bounded chunk at a time
- storing all prior chunks in paged KV cache
- using `page_table` plus `chunk_start_idx` so each chunk still behaves like part of one global causal sequence

So the right conclusion is:

- **No**, chunked prefill is not always needed.
- **Yes**, it is necessary for long-context support and for prefix-cached prefill in this codebase.

## Key Source Files

- `models/tt_transformers/tt/generator.py`
- `models/tt_transformers/tt/attention.py`
- `models/tt_transformers/tt/common.py`
- `models/tt_transformers/tt/model_config.py`
- `models/tt_transformers/README.md`
- `ttnn/cpp/ttnn/operations/transformer/sdpa/device/sdpa_program_factory.cpp`
<!-- END VERBATIM: chunked_prefill_report.md -->

---

## Common Model Plans

<a id="source-02-models-common-models-always-paged-attention-plan-md"></a>

### Source 02: `models/common/models/ALWAYS_PAGED_ATTENTION_PLAN.md`

[Back to Source Index](#source-index) | [Previous: `chunked_prefill_report.md`](#source-01-chunked-prefill-report-md) | [Next: `models/common/models/CLAUDE.md`](#source-03-models-common-models-claude-md)

<!-- BEGIN VERBATIM: models/common/models/ALWAYS_PAGED_ATTENTION_PLAN.md -->
# Plan: Always-Paged Attention in Executors (DONE)

## Goal

Remove the optional `page_table` code path from `EagerLLMExecutor` and `TracedLLMExecutor`, making paged attention the only supported mode.

## Motivation

- ~15 `if page_table is not None:` conditionals clutter the code
- Two code paths = more bugs, harder to test, harder to maintain
- vLLM and production always use paged attention
- Non-paged path is effectively dead code in real deployments

---

## Phase 1: Add Helper Function

Create `make_contiguous_page_table()` in `executor.py`:

```python
def make_contiguous_page_table(
    batch_size: int,
    max_seq_len: int,
    block_size: int,
) -> torch.Tensor:
    """Create a simple contiguous page table for demos/tests.

    Returns page_table [batch_size, num_blocks] where each user
    gets contiguous blocks: user 0 -> [0,1,2,...], user 1 -> [N,N+1,...], etc.
    """
    num_blocks_per_user = (max_seq_len + block_size - 1) // block_size
    page_table = torch.zeros(batch_size, num_blocks_per_user, dtype=torch.int32)
    for user_id in range(batch_size):
        start_block = user_id * num_blocks_per_user
        page_table[user_id] = torch.arange(start_block, start_block + num_blocks_per_user)
    return page_table
```

**Note:** This helper creates the simplest case (contiguous blocks). For advanced use cases, additional helpers can be added:
- `make_shared_prefix_page_table()` — for prefix caching where users share common prompt blocks
- `make_pooled_page_table()` — for dynamic block allocation from a shared pool (vLLM-style)

### Page Tables and Prefix Caching

**Connection:** Page tables enable prefix caching by allowing multiple users to point to the same physical KV cache blocks.

| Scenario | Page Table Structure |
|----------|---------------------|
| No sharing (contiguous) | User 0: [0,1,2], User 1: [3,4,5] |
| Shared prefix (2 blocks) | User 0: [0,1,2], User 1: [0,1,5] — blocks 0,1 are shared |
| Full prefix cache hit | User 0: [0,1,2], User 1: [0,1,2] — identical, no new compute |

In the executor code, `start_pos` indicates how many tokens are already cached. When `start_pos > 0`, the prefill skips those tokens and only computes new ones, writing to the user's unique blocks while reading from shared prefix blocks.

**Why this matters for always-paged:** Without page tables, there's no mechanism to share KV cache between users. The non-paged path cannot support prefix caching at all — another reason to remove it.

---

## Phase 2: Update Type Signatures

**File:** `executor.py`

| Method | Change |
|--------|--------|
| `_prepare_prefill_device_inputs` | `page_table: torch.Tensor` (remove `None`) |
| `prepare_decode_inputs_host` | `page_table: torch.Tensor` (remove `None`) |
| `prepare_decode_inputs_device` | `page_table: torch.Tensor` (remove `None`) |
| `compile_prefill` | `page_table: torch.Tensor` (remove `\| None`) |
| `compile_decode` | `page_table: torch.Tensor` (remove `\| None`) |
| `prefill_forward` | `page_table: torch.Tensor` (remove `\| None`) |
| `decode_forward` | `page_table: torch.Tensor` (remove `\| None`) |
| `_prefill_single_user` | `page_table: torch.Tensor` (remove implicit None) |
| `_easy_trace_prefill` | same |
| `_capture_and_run_prefill_trace` | same |
| `_capture_decode_trace` | same |

Also update:
- `_compile_prefill_and_decode`
- `run_teacher_forcing`
- `run_perf_benchmark`

---

## Phase 3: Remove Conditionals

Remove these patterns from `executor.py`:

```python
# REMOVE all instances of:
if page_table is not None:
    ...

# REMOVE default None handling:
tt_page_table = None
if page_table is not None:
    tt_page_table = ttnn.from_torch(...)

# REPLACE with unconditional:
tt_page_table = ttnn.from_torch(page_table, ...)
```

**Specific locations (~15 removals):**

| Line(s) | Location | Description |
|---------|----------|-------------|
| 244-252 | `_prepare_prefill_device_inputs` | page_table conditional |
| 255-263 | `_prepare_prefill_device_inputs` | chunk_page_table conditional |
| 303-314 | `prepare_decode_inputs_host` | page_table conditional |
| 497-505 | `prefill_forward` | page_table_user conditional |
| 509-510 | `prefill_forward` | user_id selection based on page_table |
| 545 | `_prefill_single_user` | assert for chunked prefill |
| 1023-1036 | `TracedLLMExecutor.prefill_forward` | page_table_user branching |

---

## Phase 4: Simplify `_prefill_single_user`

**Current code:**
```python
if use_chunked or use_prefix_caching:
    assert page_table is not None and self._kv_cache is not None
```

**After change:**
```python
# page_table is always provided, just assert kv_cache
assert self._kv_cache is not None, "KV cache must be allocated before prefill"
```

Also remove the `user_id=0 if page_table is not None else user_id` ternary — always use `user_id=0` with paged attention.

---

## Phase 5: Update Callers

**Search pattern:**
```bash
grep -r "prefill_forward\|decode_forward\|compile_prefill\|compile_decode" models/
```

**Likely callers to update:**
1. `models/common/models/llama3_8b/` — demo scripts
2. `models/common/tests/` — integration tests
3. Any other model-specific executors

**Migration pattern:**

Each caller that currently passes `page_table=None` must instead:
```python
page_table = make_contiguous_page_table(batch_size, max_seq_len, block_size)
```

---

## Phase 6: Update Helper Functions

- `_get_prefill_user_page_table` — remove None check, always slice
- `_get_prefill_trace_user_page_table` — same
- `_compile_prefill_and_decode` — require `prefill_page_table` (remove `| None`)

---

## Validation Checklist

- [ ] All type hints updated (no `| None` for page_table)
- [ ] All conditionals removed
- [ ] Helper function added and tested
- [ ] Callers updated to use helper or explicit page tables
- [ ] Existing tests pass
- [ ] No dead code paths remain

---

## Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| Breaking existing demos | Search for all callers before changing signatures |
| Performance regression | Contiguous page table has same memory layout as non-paged |
| Missing a conditional | Use `grep "page_table.*None\|None.*page_table"` to find all |

---

## Stakeholder Input Summary

| Stakeholder | Position |
|-------------|----------|
| vLLM Integration | Support — always use paged in production |
| Model Developers | Support with helper function for simple cases |
| Performance Engineers | Support — branching costs more than micro-optimization |
| Memory/Hardware | Support — paged KV cache is the target architecture |
| API/DevEx | Support — one path = fewer bugs, clearer docs |

---

## Next Steps

1. Review this plan
2. Identify all callers with grep search
3. Implement Phase 1 (helper function)
4. Implement Phases 2-4 (executor changes)
5. Implement Phase 5 (update callers)
6. Run tests and validate
<!-- END VERBATIM: models/common/models/ALWAYS_PAGED_ATTENTION_PLAN.md -->

<a id="source-03-models-common-models-claude-md"></a>

### Source 03: `models/common/models/CLAUDE.md`

[Back to Source Index](#source-index) | [Previous: `models/common/models/ALWAYS_PAGED_ATTENTION_PLAN.md`](#source-02-models-common-models-always-paged-attention-plan-md) | [Next: `models/common/models/EXECUTOR_TODOS.md`](#source-04-models-common-models-executor-todos-md)

<!-- BEGIN VERBATIM: models/common/models/CLAUDE.md -->
# TTTv2 Design Proposal

## Problem Analysis

The core issue with TTTv1 (lives in models/tt_transformers/ directory) is the **N×M×P explosion**:
- N models × M platforms × P tests = exponential complexity
- Every change requires validating all combinations
- Adding model #11 requires testing against models 1-10

This problem gets worse with each newly added model
Our motivations for TTTv2 is to address this scaling problem and enalbe scaling to 100+ models

## Goals

Single developer can add a new LLM model without tribal knowledge

TT-Transformers (TTT) code as a library
- Modular
- Composable
- Readable
- Reproducible
- Maintainable and releasable
- Verifiable

TTTv1 is good, first achievement of the goals for 10+ models; now TTTv2 must do better to get to 100+ models. TTTv2 should be a collection of building blocks that models consume, not a framework that controls models.

## Proposed Architecture

### Zen of TTTv2 Architecture

- Library, not Framework
- If-else on static conditions in forward() is bad
- Lazy and Transparent is better than proactive and opaque
- Unit tests are better than end-to-end tests
- Balance between code and codegen

# Instruction for AI agent to refactor tt_transformers modules to TTTv2 style

NOTE: models/common/modules/mlp contains the MLP module, which is the example to follow. `models/common/tests/modules/mlp/` contains the tests for the MLP module.

## Phase 1: Analysis

1. Read the original module (e.g., `models/tt_transformers/tt/attention.py`)
2. Identify all static branching axes in `forward()`:
   - Hardware topology: `is_galaxy` / `TG`
   - Mode: `decode` / `prefill`
   - Model dimensions: `dim`, `n_heads`, etc.
   - Other compile-time-known conditions
3. Draw an execution path graph showing all unique paths through the forward function
4. Identify runtime-dependent logic that MUST stay in forward (e.g., `seq_len`-dependent reshapes)


## Phase 2: Split by Hardware Topology

1. Create 1D (corresponds to 1x1, 1x2, 1x8 mesh_shapes -- non-TG) and 2D (corresponds to 4x8, 8x4 mesh_shapes -- TG) modules in the original file
2. Move the corresponding `if is_galaxy` branch into the 2D module
3. Update `forward()` to be a simple dispatcher

## Phase 3: Extract to Separate Files

1. Create `attention_1d.py` and `attention_2d.py` with:
   - Tightened config classes (remove TG-specific fields)
   - `Attention1D` class with the forward logic
   - Helper functions copied locally (don't import from original)

2. Config class structure:
3. **Test**: Add `test_attention_1d_class_vs_reference` and `test_attention_2d_class_vs_reference` comparing outputs

## Phase 4: Move Static Configs to `__init__`

1. Make separate forward methods
2. Pre-compute all program configs in `__init__`:
3. `Attention1D` and `Attention2D` classes should have `decode_forward` and `prefill_forward` methods
4. Simplify forward methods to use pre-computed configs

## Phase 5: Flatten Forward Functions

1. Create `decode_forward(self, x)` with NO if-else
2. Create `prefill_forward(self, x)` with only runtime logic
3. Inline utility functions (e.g., `tt_all_gather`, `tt_all_reduce`) as mode-specific methods

## Phase 6: Factory Method for Backward Compatibility

Look at `MLP1D.from_model_args` for an example

## Testing Strategy

1. **Unit tests** (no device): Config creation, helper functions
2. **Integration tests** (with device):
   - `test_attention_1d_vs_reference`: Compare `Attention1D` to HuggingFace reference
   - `test_attention_2d_vs_reference`: Compare `Attention2D` to HuggingFace reference
3. **Rejection tests**: Ensure 2D class rejects non-TG devices

## File Structure Convention

```
models/common/modules/
├── attention/
│   ├── attention_1d.py
│   └── attention_2d.py
├── mlp/
│   ├── mlp_1d.py
│   └── mlp_2d.py
models/common/tests/modules/attention/
├── test_attention_1d.py
├── test_attention_2d.py
models/common/tests/modules/mlp/
├── test_mlp_1d.py
├── test_mlp_2d.py
```

# Key Principles to Remember

1. **Never guess configs** - always read the original code to understand what program configs are used
2. **Keep runtime logic minimal** - only `seq_len` checks, input shape checks belong in forward
3. **Use method overriding** - config classes use methods so subclasses can override
4. **Test incrementally** - test after each phase before proceeding
5. **Preserve backward compatibility** - `from_model_args()` factory method bridges old and new
6. **Single device check** - always handle `[1, 1]` mesh as special case (no CCL ops)

# More specific instructions for AI agent to refactor other modules to TTTv2 style

## Key files to reference:
- **Example module**: `models/common/modules/mlp/mlp_1d.py`
- **Example tests**: `models/common/tests/modules/mlp/test_mlp_1d.py`
- **TTTv1 source**: `models/tt_transformers/tt/<module>.py`
- **TTTv1 config**: `models/tt_transformers/tt/model_config.py`
- **CCL functions**: `models/tt_transformers/tt/ccl.py`

## Step 0: Analyze dependencies with trace_dependencies.py
Before starting any refactoring work, run the dependency tracer to understand the module's parameter dependencies:
```bash
python_env/bin/python models/common/modules/trace_dependencies.py --matmul-helper \
    models/tt_transformers/tt/<module>.py \
    models/tt_transformers/tt/model_config.py
```
This will output a hierarchical dependency graph showing what parameters affect the module's behavior.

### How to run trace_dependencies.py:

```bash
# Default: analyze TTTv1 MLP module
python_env/bin/python models/common/modules/trace_dependencies.py

# Analyze a specific module (provide paths to module and its config):
python_env/bin/python models/common/modules/trace_dependencies.py <module_path> <config_path>

# Example: analyze attention module
python_env/bin/python models/common/modules/trace_dependencies.py \
    models/tt_transformers/tt/attention.py \
    models/tt_transformers/tt/model_config.py

# Include detailed matmul helper analysis:
python_env/bin/python models/common/modules/trace_dependencies.py --matmul-helpers

# Output as JSON:
python_env/bin/python models/common/modules/trace_dependencies.py --json

# Write JSON to file:
python_env/bin/python models/common/modules/trace_dependencies.py --json-file deps.json
```

### What the tool outputs:
1. **Config Accesses** - All `model_config["KEY"]` accesses in the module
2. **Attributes Used in Conditions** - `self.*` attributes that control branching
3. **Config Key Dependencies** - What each config key depends on
4. **Root Parameters** - The minimal set of parameters affecting behavior
5. **CCL Function Analysis** - Dependencies in collective communication functions
6. **Matmul Helper Methods** - Analysis of config helper functions (with `--matmul-helpers`)
7. **Complete Parameter Hierarchy** - 6-level dependency graph from hardware to terminal ops

## Step 1: Copy the TTTv1 module to TTTv2 location
```bash
# Create the module directory (do NOT add __init__.py)
mkdir -p models/common/modules/<module_name>/

# Copy the original module
cp models/tt_transformers/tt/<module>.py models/common/modules/<module_name>/<module>.py
```

### NOTES:
- Do NOT create `__init__.py`, `conftest.py`, or other boilerplate files unless explicitly requested. This project has specific file structure conventions.

## Step 2: Identify and bring in required dependencies
From the trace_dependencies.py output:
- **Terminal params**: `dim`, `hidden_dim`, `n_heads`, `cluster_shape`, etc. (from ModelArgs)
- **Config keys**: `DECODE_*_PRG_CONFIG`, `PREFILL_*_PRG_CONFIG`, `*_MEMCFG` (from model_config)
- **Helper functions**: `matmul_config`, `dram_matmul_config`, `create_sharded_memory_config`, etc.
- **CCL functions**: `tt_all_reduce`, `tt_all_gather` (if distributed)

Copy only the required helper functions into the new module file - don't import from original.

## Step 3: Remove model_config and args from __init__ signature
The goal is to make the module self-contained with explicit parameters:
```python
# BEFORE (TTTv1 style):
def __init__(self, mesh_device, args, model_config, layer_num, ...):
    self.model_config = model_config
    self.args = args

# AFTER (TTTv2 style):
# Happy path:
def __init__(self, w1: LazyWeight, w2: LazyWeight, w3: LazyWeight):
    # All config values come from the explicit ModuleConfig dataclass

# Power-user path:
@classmethod
def from_config(cls, config: <module>Config):
    # bypass the __init__ method of the base class for power users who want to customize the config
```
See `MLP1D.__init__` and `MLP1D.from_config` in `mlp_1d.py` for real examples.

## Step 4: Create config dataclass hierarchy
Follow the MLP pattern with sub-config classes, read `models/common/modules/mlp/mlp_1d.py` for reference.

Include `decode_input_memcfg` and `prefill_input_memcfg` fields in the config, and add a `_load_input_device_tensor()` helper function to resolve `LazyWeight` inputs. See `MLP1DConfig` and `_load_input_device_tensor()` in `mlp_1d.py`, or `RMSNorm1DConfig` in `rmsnorm_1d.py` for distributed sharding support.

If a module contains sub-modules, compose the sub-modules into the main module's config dataclass. See `Attention1DConfig` in `attention_1d.py` for an example of how to compose the sub-modules (i.e., `RMSNorm1DConfig`).

## Step 5: Split into 1D and 2D variants
- **1D**: For non-TG topologies (1x1, 1x2, 1x8 mesh shapes)
- **2D**: For TG topologies (4x8, 8x4 mesh shapes)

Create separate files: `<module>_1d.py` and `<module>_2d.py`

## Step 6: Flatten forward() - eliminate static branching
```python
# BEFORE (TTTv1 - branching on static conditions):
def forward(self, x, mode):
    if self.args.is_galaxy:
        # TG path
    else:
        # non-TG path
    if mode == "decode":
        # decode path
    else:
        # prefill path

# AFTER (TTTv2 - separate methods, no branching):
def decode_forward(self, x):
    # Only decode logic, no if-else on mode

def prefill_forward(self, x, seq_len: int):
    # Only prefill logic, seq_len-dependent reshapes are OK
```

Each `*_forward` method should accept `ttnn.Tensor | LazyWeight` and call `_load_input_device_tensor()` at the start to resolve the input. See `decode_forward()` in `mlp_1d.py` or `_decode_local_sharded()` in `rmsnorm_1d.py`.

## Step 7: Implement from_model_args factory method
For backward compatibility with TTTv1 models, see `MLP1D.from_model_args` in `models/common/modules/mlp/mlp_1d.py` for an example implementation.

## Step 8: Create tests
```
models/common/tests/modules/<module_name>/
├── test_<module>_1d.py    # Tests for 1D variant
├── test_<module>_2d.py    # Tests for 2D variant (if applicable)
```

### Test categories:
1. **Unit tests** (no device): Config creation, helper function correctness
2. **Integration tests** (with device): Compare output against HuggingFace reference
3. **Rejection tests**: Ensure 2D class rejects non-TG devices

### Requirements for unit tests:
- Since `forward()` accepts `LazyWeight`, tests can wrap torch input in `LazyWeight` and pass directly to `forward()` - no manual `ttnn.from_torch` conversion needed. This enables optional input caching for faster repeated tests. See `test_mlp_1d.py` or `test_rmsnorm_1d.py` for examples.
- There is pytest fixture -- `ttnn_mesh_device` in `models/common/tests/conftest.py`; see `test_mlp_1d.py` for examples on how to use it.
- two important test cases to include (see `test_mlp_1d.py` for examples):
1) `test_<module>_vs_reference`, where the test cases are collected and parameters are hardcoded as pytest mark parameters; this test focus on testing the `__init__`, `from_config` factory methods, and the `forward` method; the forward method must be run and the output tensor compared to the HuggingFace reference model's output tensor.
2) `test_<module>_vs_reference_from_model_args`, where we just test a single model with small number of parameters to prove backward compatibility; this test focus on testing the `from_model_args` factory method; the forward method must be run and the output tensor compared to the HuggingFace reference model's output tensor.
- Use the `ttnn_mesh_device` pytest fixture from `models/common/tests/conftest.py`.
- When adding collected test cases, hardcode the parameters as pytest mark parameters; do not rely on the csv files (they will be removed); see the below section "NOTES about collecting test cases" for more details.

### Requirements for running tests on hardware:

- Use `python_env/bin/python`
- Use `HF_MODEL` to specify a HF model name for testing `from_model_args` factory method. For example, `HF_MODEL=meta-llama/Llama-3.1-8B-Instruct`
- Use `git submodule update --init --recursive` to update the submodules before running tests.
- Use `./build_metal.sh -c --development && ./create_venv.sh` to build and create the virtual environment after pulling the submodules  and before running tests.
- If wondering "Would you like me to wait for the build to complete and then run the device tests, or would you prefer to run them manually later?", the answer is "wait and then run the device tests".
- As `models/common/tests/setup.cfg` dictates >80% coverage, however, the unit tests aims to achieve >90% coverage.
- to gather coverage metrics, e.g., when running test_mlp_1d.py with pytest, use:
```
"--cov=models.common.modules.mlp.mlp_1d",
"--cov-report=term-missing",
"--cov-config=models/common/tests/setup.cfg"
```
- Use `tt-smi -r` to reset all devices if tests failed due to bad device states; it is a good idea to first `source python_env/bin/activate` to activate the virtual environment before running `tt-smi -r`.
- do NOT skip tests!!! --> unless there is a mismatch in device shape as the example module's tests do.

### Requirements for running tests on simulator:
- Use `/localdev/gwang/scripts/setup_ttsim.sh` to setup the simulator environment. Let me know if you cannot find the script.
- Follow the instructions in the script to use `TT_METAL_SIMULATOR_HOME`, `TT_METAL_SIMULATOR`, `TT_METAL_SLOW_DISPATCH_MODE` to specify the simulator environment.
- Run the tests as you would on hardware, and the simulator should be used automatically -- look for log output like the following to confirm:
```bash
| info     |             UMD | Creating Simulation device (cluster.cpp:222)
```

### Collecting test cases:

#### mlp as an example
in the `models/tt_transformers/tt/mlp.py` file, do at the module level:

```python
collected = set()
if os.path.exists("mlp_1d_performance.csv"):
    with open("mlp_1d_performance.csv", "r") as f:
        reader = csv.reader(f)
        next(reader, None)  # skip header
        for row in reader:
            if row:
                collected.add(",".join(row[1:]))

```

inside the `forward()` of `models/common/modules/mlp/mlp.py` as the first thing, do:
```python
        # "layer_num,cluster_shape,dtype,batch_size,dim,hidden_dim,hf_model_name,seq_len,mode",
        file_exists = os.path.exists("mlp_1d_performance.csv")
        with open("mlp_1d_performance.csv", "a") as f:
            if not file_exists:
                f.write(
                    "layer_num,cluster_shape_x,cluster_shape_y,x_dtype,w1_dtype,w2_dtype,w3_dtype,batch_size,filler,seq_len,dim,hidden_dim,hf_model_name,mode\n"
                )
            entry = f"{self.args.cluster_shape[0]},{self.args.cluster_shape[1]},{x.dtype},{self.w1.dtype},{self.w2.dtype},{self.w3.dtype},{x.shape[0]},{x.shape[1]},{x.shape[2]},{x.shape[3]},{self.args.hidden_dim},{self.args.model_name},{mode}"
            if entry not in collected:
                collected.add(entry)
                f.write(f"{self.layer_num},{entry}\n")
```

### script `run_tttv1_simple_demos.sh` that can run all the simple demos in TTTv1

With the changes done to `models/common/modules/mlp/mlp.py`, the script `run_tttv1_simple_demos.sh` can be updated to run the simple demos in TTTv1.

## Step 9: Run the tests
- When debugging failing tests, prioritize code comparison and analysis BEFORE repeatedly running tests. Compare implementation against working reference code to identify discrepancies.
- Always run tests after implementing changes - don't wait for user to remind you. If a plan says 'run test', execute it before reporting completion.
- After code change, make sure to run linter before reporting completion.

VERY IMPORTANT: make sure the tests do not skip PCC checks for any device shapes for any reason!!! For example:
```
# reduce_scatter sums across devices and each gets 1/N of the result
# We need to sum all devices' local computation to match
# This is complex to replicate exactly, so just check shape + non-zero
```
is a bad example. Do NOT skip PCC checks even when things are complicated!

Two strategies for running the tests are as the following.
### When in active development
Run all the tests on the simulator to quickly iterate on the changes and to decouple from hardware. This should also allow parallel development of multiple modules, multiple ideas, and etc. The main benefit is that the development is not bottlenecked by hardware availability.

### When in verification mode
Run all the tests on hardware to ensure the TTTv2 module is working as expected. If there are any tests that fail, fix them.

## Step 10: Audit the changes
After the tests are created and run through all the collected test cases, do an audit of the changes that were needed to fix all the tests and compare those changes to TTTv1 module implementation in models/tt_transformers/tt/<module>.py. The goal is to check if the TTTv2 module is doing the same thing as the TTTv1 module. If there are any discrepancies, fix them, go to step 9 to run the tests again, go to step 10 to audit the changes again, and repeat until all the tests pass and the audit is successful.

## Step 11: Double-check dependencies
`modules/` is considered the core of the TTTv2 library, so we need to make sure that the dependencies are up to the design goals of TTTv2 -- no explicit dependencies except for TTNN and Python standard library. For example, one key thing to double-check is no explicit dependencies on e.g., `torch`.

Similarly, in the unit tests, we want the reference implementation to be from huggingface and not TTTv1, except for the test functions that covers the backward-compatible APIs (e.g., `from_model_args(...)`)

One situation where it is fine to use torch is when the module uses non-core modules that imports torch. For example, `models/common/tensor_utils.py` imports `torch` and TTTv2 modules can uses tensor_utils.py to construct default configurations. This is not considered an explicit dependency on torch, because the users of the modules could easily override the default configurations with whatever libraries (e.g., other than torch) they want to use.

## Step 12: Double-check tests
- Make sure the tests do NOT skip PCC checks for any device shapes for any reason!!!
- Make sure the tests that uses LazyWeight use `cache_dir = Path(os.getenv("TT_CACHE_PATH", "model_cache/<module_name>"))`
- Make sure there are test cases to cover config fields that accepts more than one types of inputs, e.g., `field : LazyWeight | ttnn.Tensor`; a small test case should be added to cover the `ttnn.Tensor` input.

## Step 13: Double-check the code for more subtle issues
- Make sure LazyWeight pattern is used in the module: 1) create LazyWeight object as part of config resolution; 2) load_device_tensors/load_device_weights as part of module object runs
- Double-check the Config class to make sure the fields are all used and no key fields are missing, e.g., `input_memcfg`; Use Config classes of existing module as examples and the Step 10 audit as guidelines
- Double-check the constructors of the module class to make sure we have both "happy" path and the power user path; use the existing modules as examples
- Double-check to make sure there is `prefill_forward`, `decode_forward`, and `forward functions`.
- Make sure there is no dead code
- Make sure there is no magic constant values
- Make sure there is no dependency on TTTv1 code within TTTv2 modules and tests except for the code in `from_model_args` and its tests.
- Double-check the test cases against the collected test cases to make sure they match
- Double-check the default configs against TTTv1 code to make sure there is no fabricated and dangerous defaults
- Comb through the prefill_forward and decode_forward to find if-else that are switching on static conditions
- Audit the comments and docstrings to make sure they are updated -- matching the code
- Remove all mentions of the CSV files that the collected test cases are from
- Double check for allocated temporary ttnn.tensors that should be deallocated, e.g.,:
```
The pattern `old = x; x = op(x); deallocate(old)` is necessary here because `to_memory_config` and `untilize` both allocate **new** device-side buffers. Without the explicit deallocate, the old buffer stays alive on the device until Python's GC eventually collects the reference — which may never happen in a tight decode loop, causing OOM on the device.
```<!-- END VERBATIM: models/common/models/CLAUDE.md -->

<a id="source-04-models-common-models-executor-todos-md"></a>

### Source 04: `models/common/models/EXECUTOR_TODOS.md`

[Back to Source Index](#source-index) | [Previous: `models/common/models/CLAUDE.md`](#source-03-models-common-models-claude-md) | [Next: `models/common/models/EXECUTOR_TRACE_SPLIT_PLAN.md`](#source-05-models-common-models-executor-trace-split-plan-md)

<!-- BEGIN VERBATIM: models/common/models/EXECUTOR_TODOS.md -->
# Executor.py TODOs

Summary of all TODOs in `executor.py`. Line numbers updated 2026-04-28.

## Completed

| Original | Status | Notes |
|----------|--------|-------|
| Make page_table handling into a composable feature | ✅ DONE | Removed by always-paged attention refactor — page_table is now required |

## Debug/Optional Features

| Line | TODO |
|------|------|
| 169 | Could be a composed optional for debug |
| 1060 | Prefill caching could be refactored into composable feature |

## Code Refactoring

| Line | TODO |
|------|------|
| 205-206 | Could be lazy weight? Use `ttnn.zeros()` directly instead of `as_tensor`? |
| 264 | Rope code could be refactored |
| 362 | Inline the `prepare_decode_inputs_device` function |
| 588 | Refactor chunked/prefix caching logic to be more readable |
| 691 | Move sampling import to the top of the file |
| 705 | Move `get_rot_mats` call closer to where it's used |
| 1179 | Clean up trace capture code |

## Design Violations / Architecture

| Line | TODO |
|------|------|
| 1408 | Question about correctness - compare with TTTv1 |

## Cleanup / Removal

| Line | TODO |
|------|------|
| 416 | Use `TensorSpec.from_tensor()` directly for logits |
| 508 | `output_tensor` is overwritten later - why allocate here? |
| 1018 | Remove unnecessary `sampling_params` param |

## Memory/Trace Management

| Line | TODO |
|------|------|
| 773 | Cannot save many traces in memory - need LRU cache? Warmup traces must persist |
| 1052 | `empty_slots` only used with vLLM - move to `generator.py`? |
| 1094-1096 | Add warning about non-traceable prefill; make whole thing traceable? |
| 1161 | Should assert expected trace exists (look up by prefill_seq_len) |

---

**Total: ~18 TODOs** (was ~30, reduced by refactoring + cleanup)
<!-- END VERBATIM: models/common/models/EXECUTOR_TODOS.md -->

<a id="source-05-models-common-models-executor-trace-split-plan-md"></a>

### Source 05: `models/common/models/EXECUTOR_TRACE_SPLIT_PLAN.md`

[Back to Source Index](#source-index) | [Previous: `models/common/models/EXECUTOR_TODOS.md`](#source-04-models-common-models-executor-todos-md) | [Next: `models/common/models/RUN_PERF_BENCHMARK_PLAN.md`](#source-06-models-common-models-run-perf-benchmark-plan-md)

<!-- BEGIN VERBATIM: models/common/models/EXECUTOR_TRACE_SPLIT_PLAN.md -->
# Executor Trace Split Plan (DONE)

## Goal

Remove all trace-specific behavior from `EagerLLMExecutor`.

After this refactor:

- `EagerLLMExecutor` is fully trace-blind.
- `TracedLLMExecutor` owns trace eligibility, trace input prep, trace capture, replay, and cleanup.
- thin wrappers in `models/common/models/llama3_8b/model.py` reflect that split instead of mirroring the current leak.
- helper loops in `models/common/models/executor.py` stop passing trace flags into eager APIs.

## Current Problems

### 1. `EagerLLMExecutor.prepare_prefill_inputs()` is two APIs hidden behind one boolean

Today `trace_enabled` changes all of this:

- host vs device placement
- raw tokens vs embedded tensor return value
- RoPE slice semantics
- page-table tensor placement

That means the function is not a single coherent API.

The current behavior lives in:

- `models/common/models/executor.py`
  - `EagerLLMExecutor.prepare_prefill_inputs()`

### 2. trace policy leaks into eager public APIs

These eager methods still accept trace controls:

- `EagerLLMExecutor.prefill_forward(..., enable_trace=...)`
- `EagerLLMExecutor.decode_forward(..., enable_trace=...)`
- `EagerLlamaExecutor.prepare_prefill_inputs(..., trace_enabled=...)`
- `EagerLlamaExecutor.prefill_forward(..., enable_trace=...)`
- `EagerLlamaExecutor.decode_forward(..., enable_trace=...)`

That is the wrong abstraction boundary. If the caller wants tracing, it should use a traced executor.

### 3. `TracedLLMExecutor` depends on eager internals for traced prefill prep

These methods call into `_eager.prepare_prefill_inputs(..., trace_enabled=True)`:

- `TracedLLMExecutor._easy_trace_prefill()`
- `TracedLLMExecutor._capture_and_run_prefill_trace()`

That dependency direction is backwards. traced should own traced input prep.

### 4. the common traced executor is already model-specific

`TracedLLMExecutor.prefill_forward()` imports a Llama-specific helper and manually runs:

- `model.norm.prefill_forward(...)`
- `_all_gather_rmsnorm_tensor(...)`
- `model.lm_head.forward(...)`

That means the "generic" traced executor is not actually generic.

## End State

### `EagerLLMExecutor`

`EagerLLMExecutor` should only know how to do eager execution.

It should own:

- eager prefill input preparation
- eager decode input preparation
- eager prefill execution
- eager decode execution
- eager compile warmup
- KV cache allocation / identity checks

It should not own:

- `trace_enabled`
- `enable_trace`
- trace keying
- trace capture
- trace replay
- trace-only host staging
- trace-only page-table sizing rules

### `TracedLLMExecutor`

`TracedLLMExecutor` should own:

- trace eligibility decisions
- trace keys
- capture
- replay
- trace-owned buffers
- eager fallback when tracing is unsupported or disabled
- cleanup of trace resources
- traced prefill host-input preparation

It may still delegate to `_eager` for:

- eager fallback
- shared eager compile / warm kernels
- eager decode helpers where those helpers are truly trace-agnostic

## Concrete API Changes

### Remove trace controls from eager public APIs

Change `EagerLLMExecutor` from:

```python
def prepare_prefill_inputs(
    self,
    tokens,
    start_pos=0,
    page_table=None,
    chunk_page_table=None,
    trace_enabled=False,
    last_token_idx=None,
):
    ...

def prefill_forward(
    self,
    tokens,
    page_table=None,
    kv_cache=None,
    prompt_lens=None,
    empty_slots=None,
    enable_trace=True,
    sampling_params=None,
    start_pos=None,
):
    ...

def decode_forward(
    self,
    tokens,
    start_pos,
    page_table=None,
    kv_cache=None,
    enable_trace=True,
    read_from_device=True,
    sampling_params=None,
):
    ...
```

to:

```python
def _prepare_prefill_device_inputs(
    self,
    tokens,
    start_pos=0,
    page_table=None,
    chunk_page_table=None,
    last_token_idx=None,
):
    ...

def prefill_forward(
    self,
    tokens,
    page_table=None,
    kv_cache=None,
    prompt_lens=None,
    empty_slots=None,
    sampling_params=None,
    start_pos=None,
):
    ...

def decode_forward(
    self,
    tokens,
    start_pos,
    page_table=None,
    kv_cache=None,
    read_from_device=True,
    sampling_params=None,
):
    ...
```

Notes:

- `prepare_prefill_inputs()` should become private on the engine or disappear entirely.
- `EagerLlamaExecutor` should not keep a public `prepare_prefill_inputs()` wrapper once the engine helper becomes private.
- keep `sampling_params` on eager `prefill_forward()` for now. It is not a trace concern, and keeping it avoids coupling this refactor to a separate API simplification decision.

### Add traced-private prefill prep

Add a traced-only helper in `TracedLLMExecutor`, something like:

```python
def _prepare_prefill_trace_inputs_host(
    self,
    tokens,
    start_pos=0,
    page_table=None,
    chunk_page_table=None,
    last_token_idx=None,
):
    ...
```

This helper should own the current traced prefill contract:

- host-side token tensor creation
- traced RoPE slicing rules
- host-side page table / chunk page table tensors
- replay-ready tensor layout for `copy_host_to_device(...)`

It must not live in eager.

### Keep traced trace controls for now

For the first pass, keep `enable_trace` on `TracedLLMExecutor.prefill_forward()` and `decode_forward()`.

Reason:

- `run_perf_benchmark()` currently wants a traced executor that can still run some calls eagerly.
- `run_teacher_forcing()` wants a traced executor for fast tests and eager execution for convenient debugging.
- removing `enable_trace` everywhere at the same time will couple API cleanup to benchmark semantics cleanup.

That can be removed later once the call sites are cleaned up.

## Required Code Moves

### Stage 1: move traced prefill prep out of eager

Implement `TracedLLMExecutor._prepare_prefill_trace_inputs_host()` and switch these methods to use it:

- `TracedLLMExecutor._easy_trace_prefill()`
- `TracedLLMExecutor._capture_and_run_prefill_trace()`

Also clean up `_get_prefill_user_page_table(..., trace_enabled, prefill_seq_len)`:

- collapse the eager path to never pass `trace_enabled=True`
- move the trace-specific page-table sizing into the new traced helper, or split the function in two so the traced branch lives next to other traced code

Do this before changing public signatures.

This is the highest-value first step because it fixes the bad dependency direction without large call-site churn.

### Stage 2: remove trace flags from eager engine

Update `models/common/models/executor.py`:

- remove `trace_enabled` from `EagerLLMExecutor.prepare_prefill_inputs()` or replace the method entirely
- remove `enable_trace` from `EagerLLMExecutor.prefill_forward()`
- remove `enable_trace` from `EagerLLMExecutor.decode_forward()`
- remove eager-only branches that exist only because those flags still exist

### Stage 3: remove trace flags from eager wrappers

Update `models/common/models/llama3_8b/model.py`:

- remove the public `EagerLlamaExecutor.prepare_prefill_inputs()` wrapper entirely
- remove `enable_trace` from `EagerLlamaExecutor.prefill_forward()`
- remove `enable_trace` from `EagerLlamaExecutor.decode_forward()`

The eager wrappers should mirror the eager engine, not the traced engine.

### Stage 4: fix helper loops that assume a shared trace-flag API

Update these helpers in `models/common/models/executor.py`:

- `run_teacher_forcing()`
- `run_perf_benchmark()`

Leave `_compile_prefill_and_decode()` generic for now. It only calls shared `compile_*`
methods today, so it does not need the same cleanup until traced `compile_*`
semantics change in Patch 5.

Current bad pattern:

```python
executor.prefill_forward(..., enable_trace=False)
executor.decode_forward(..., enable_trace=False)
```

For this transition:

1. `run_teacher_forcing()` should remain generic and support both
   `EagerLLMExecutor` and `TracedLLMExecutor`.
2. `run_perf_benchmark()` should remain traced-only for now and use `enable_trace`
   on `TracedLLMExecutor` to benchmark eager-vs-traced decode behavior on the same executor.

Preferred version:

```python
def is_traced_executor(executor) -> bool:
    return hasattr(executor, "trace_id_prefill") and hasattr(executor, "trace_ids_decode")


if is_traced_executor(executor):
    prefill_output = executor.prefill_forward(..., enable_trace=False)
else:
    prefill_output = executor.prefill_forward(...)
```

Do not add a generic `supports_tracing` property just to paper over the split.
Also do not inspect function signatures for `enable_trace`; during the transition,
eager still has that argument until the cleanup lands.

For now, keep this simple:

- traced executors are identified by public trace-facing APIs
- if the helper supports both executor types, branch explicitly and only pass
  trace-only kwargs on the traced path

Do not route `run_perf_benchmark()` through a separate eager executor in this patch.
The point of keeping `enable_trace` on `TracedLLMExecutor` for now is that the traced
executor already owns `_eager`, KV cache state, and eager fallback behavior.

Why this predicate:

- `TracedLLMExecutor` and `TracedLlamaExecutor` both expose trace-facing APIs.
- `EagerLLMExecutor` and `EagerLlamaExecutor` should not expose trace-facing APIs after this cleanup.
- checking trace APIs matches the behavior we care about better than checking class inheritance.

### Stage 5: move model-specific traced prefill tail behind the model

See models/common/models/llama3_8b/MODEL_METHOD_SOLUTION.md for the solution.

Today `TracedLLMExecutor.prefill_forward()` is doing Llama-specific postprocessing itself.

Move that behind the explicit model hook proposed there:

```python
def post_process_prefill_output(self, hidden_states, last_token_idx):
    ...
```

This keeps the common executor from reaching into:

- `model.norm`
- `model.lm_head`
- `_all_gather_rmsnorm_tensor`

## Compile Semantics

### Current behavior

`TracedLLMExecutor.compile_prefill()` and `compile_decode()` do not actually ensure replay-ready traced state.

They currently route through:

- `prefill_forward(..., enable_trace=False)`
- `decode_forward(..., enable_trace=False)`

So compile is really just warmup plus output-spec capture.

### Recommended behavior

After the API split is stabilized:

- eager `compile_*` = warm kernels + capture output spec
- traced `compile_*` = warm kernels + capture trace + capture output spec

Do not change this in the same patch as the eager/traced API split unless necessary.

Reason:

- it changes performance behavior
- it changes first-call timing behavior
- it makes debugging harder if both semantic changes land together

## Prefill Trace Invariants That Must Be Preserved

This is the dangerous part of the refactor.

### 1. traced prefill is keyed by padded prefill seq len

Current key:

- `TracedLLMExecutor._get_prefill_trace_key(tokens)` returns `get_padded_prefill_len(tokens.shape[-1])`

Do not accidentally change trace reuse semantics while splitting helpers.

### 2. traced prefill uses different RoPE slicing semantics than eager

Current eager/traced behavior differs because traced prefill uses:

- `prefill_start = 0`
- `slice_end = max_seq_len`

while eager uses:

- `prefill_start = start_pos`
- `slice_end = min(mat_len, required_end)`

If this is still the intended trace contract, preserve it in the traced helper.

### 3. paged attention sizing differs in traced prefill

`_get_prefill_user_page_table(page_table, kv_cache, prefill_len, trace_enabled, prefill_seq_len)` currently changes page-table slicing when tracing is enabled.

That trace-specific logic should move to traced-owned code or be made explicit in helper naming/signature. This is handled in Stage 1.

Do not leave generic page-table helpers carrying trace policy long term.

### 4. prefix caching is the easiest place to silently break correctness

The dangerous combination is:

- `page_table`
- `start_pos`
- padded `prefill_seq_len`
- `last_token_idx`

If traced prefill prep is "simplified" incorrectly, the code can still run while producing wrong logits.

This is the main correctness risk.

## Proposed Staged Refactor

### Patch 1: internal split only

Goal:

- move traced prefill input prep out of eager
- no public API break yet

Changes:

- add traced-private prefill prep helper
- switch traced prefill capture/replay paths to use it
- leave public eager/traced signatures alone for the moment

Validation:

- existing traced prefill behavior still works
- eager behavior unchanged

### Patch 2: eager API cleanup

Goal:

- make eager public APIs trace-blind

Changes:

- remove eager `trace_enabled`
- remove eager `enable_trace`
- update eager call sites and eager wrapper signatures

Validation:

- eager executor call sites compile without trace knobs
- traced executor still supports existing trace controls

### Patch 3: helper-loop cleanup

Goal:

- stop generic helpers from assuming all executors expose trace flags

Changes:

- update `run_teacher_forcing()`
- update `run_perf_benchmark()`

`_compile_prefill_and_decode()` stays shared in this patch and only needs revisiting
once Patch 5 changes traced `compile_*` semantics.

Validation:

- teacher forcing supports both eager and traced executors without blindly passing
  trace-only kwargs to eager
- perf benchmark remains traced-only and still supports traced decode measurement

### Patch 4: model hook cleanup

Goal:

- remove Llama-specific traced postprocessing from common executor

Changes:

- add explicit `post_process_prefill_output()` model hook
- move norm/all-gather/lm_head path behind the model

Validation:

- common traced executor no longer imports Llama-specific helpers

### Patch 5: compile semantic cleanup

Goal:

- make traced `compile_*` actually capture traces

Changes:

- traced `compile_prefill()` ensures prefill trace exists
- traced `compile_decode()` ensures decode trace exists

Validation:

- compile leaves traced executor replay-ready
- perf numbers are re-baselined if needed

## Specific Files To Touch

Primary:

- `models/common/models/executor.py`
- `models/common/models/llama3_8b/model.py`

Likely call-site cleanup:

- `models/common/models/generator.py`
- `models/common/warmup/warmup_utils.py`
- `models/common/tests/demos/llama3_8b/demo.py`

Tests:

- `models/common/tests/test_executor_parity.py`

Potential legacy/reference reading only:

- `models/tt_transformers/tt/model.py`
- `models/tt_transformers/tt/generator.py`
- `models/tt_cnn/tt/executor.py`

## Test Plan

### Must-have tests after the refactor

1. eager API contract tests

- eager executor no longer accepts trace flags
- eager wrapper no longer accepts trace flags
- eager wrapper no longer exposes public `prepare_prefill_inputs()`

2. traced prefill replay tests

- first traced prefill captures
- second traced prefill replays
- cache key still matches expected padded seq len behavior

3. prefix-cached paged prefill regression test

Cover:

- nonzero `start_pos`
- `page_table` present
- `prompt_lens` present
- `last_token_idx` correctness

This is the highest-risk correctness test.

4. decode fallback tests

- traced executor with `enable_trace=False` still falls back correctly during the transition period

5. teacher forcing compatibility tests

- `run_teacher_forcing()` works with `EagerLLMExecutor`
- `run_teacher_forcing()` works with `TracedLLMExecutor`
- `run_teacher_forcing()` works with `EagerLlamaExecutor`
- `run_teacher_forcing()` works with `TracedLlamaExecutor`
- traced executor detection returns true for `TracedLLMExecutor` and `TracedLlamaExecutor`
- traced executor detection returns false for `EagerLLMExecutor` and `EagerLlamaExecutor`
- traced-only kwargs are only passed on the traced path

6. wrapper / warmup tests

- any wrapper method that forwards old trace kwargs gets updated or removed

### Additional validation

- run the parity tests after each patch
- if traced compile semantics change, explicitly test that `trace_id_prefill[...]` / `trace_ids_decode[...]` are populated by `compile_*`

## Recommended Implementation Order

Do the work in this exact order:

1. add traced-private prefill prep helper
2. switch traced capture/replay paths to it
3. remove eager trace flags
4. update eager wrappers
5. update helper loops
6. move model-specific traced tail behind the model
7. only then revisit traced `compile_*` semantics

## Non-Goals For The First Pass

Do not bundle these into the first patch unless required:

- redesigning trace keys
- changing trace-memory eviction policy
- full genericization of traced postprocessing across all models
- benchmark methodology changes
- removing `enable_trace` from traced executor public APIs

Those are valid follow-ups, but they should not block fixing the eager/traced boundary.

## Short Version

The first real fix is not "delete the flag".

The first real fix is:

1. make traced prefill input prep owned by `TracedLLMExecutor`
2. make eager prefill prep eager-only
3. then remove trace knobs from eager public APIs

If that order is not followed, it is very easy to break traced prefill correctness while making the code look cleaner.
<!-- END VERBATIM: models/common/models/EXECUTOR_TRACE_SPLIT_PLAN.md -->

<a id="source-06-models-common-models-run-perf-benchmark-plan-md"></a>

### Source 06: `models/common/models/RUN_PERF_BENCHMARK_PLAN.md`

[Back to Source Index](#source-index) | [Previous: `models/common/models/EXECUTOR_TRACE_SPLIT_PLAN.md`](#source-05-models-common-models-executor-trace-split-plan-md) | [Next: `models/common/models/llama3_8b/DEBUG.md`](#source-07-models-common-models-llama3-8b-debug-md)

<!-- BEGIN VERBATIM: models/common/models/RUN_PERF_BENCHMARK_PLAN.md -->
# `run_perf_benchmark()` Contract Cleanup Plan

## Goal

Keep `run_perf_benchmark()` as a generic benchmark helper.

That means the function should own only:

- benchmark timing policy
- warmup vs measured execution phases
- decode token loop
- result aggregation

It should not own executor-specific shape surgery such as:

- compile-time batch padding
- fake-user filtering via padded compile batches
- decode-capacity shaping
- mesh-device synchronization hacks via attribute probing

## Problem Statement

Today `run_perf_benchmark()` is presented as generic, but it leaks details from the current executor implementation.

Current leakage:

- it calls `executor.compile(...)`, so it already needs more than `prefill_forward()` and `decode_forward()`
- it pads compile inputs to `max_batch_size` so decode compile succeeds
- it passes `empty_slots=list(range(batch_size))` to avoid executing padded fake users during prefill compile
- it creates decode inputs sized to `max_batch_size`, which is an executor contract, not a benchmarking contract
- it synchronizes by checking `hasattr(executor, "mesh_device")`, which is an implementation detail

This mixing of responsibilities caused the recent bugs:

1. prefill compile executed padded fake users, which produced zero-length paged-fill inputs
2. fixing that in the benchmark helper accidentally broke decode compile batch-shape requirements

The helper is currently compensating for executor internals instead of consuming a clean executor contract.

## Target Contract

`run_perf_benchmark()` should accept semantic active-batch inputs only.

### Required inputs

- `tokens: torch.Tensor`
  - shape: `[active_batch, padded_prompt_width]`
  - host tensor
- `prompt_lens: torch.Tensor`
  - shape: `[active_batch]`
  - required
  - semantic prompt length per active user
- `kv_cache`
  - executor-owned cache allocation passed through to prefill/decode

### Optional inputs

- `page_table: torch.Tensor | None`
- `start_pos: torch.Tensor | None`
  - shape: `[active_batch]`
  - semantic prefix-cache starting position per active user
- `num_decode_tokens: int`
- `enable_trace: bool`
- `sampling_params: SamplingParams | None`

### Explicitly not part of the benchmark-helper contract

- compile batch width
- executor max decode capacity
- how prefill compile is padded
- how decode state tensors are internally shaped

## Target Executor Protocol

If `run_perf_benchmark()` stays generic, then the executor must expose the non-generic behavior.

Minimum benchmark-facing executor protocol:

```python
class BenchmarkExecutor(Protocol):
    def compile(
        self,
        *,
        prefill_tokens: torch.Tensor,
        prefill_page_table: torch.Tensor | None = None,
        kv_cache: list | None = None,
        prompt_lens: torch.Tensor,
        start_pos: torch.Tensor | None = None,
        sampling_params: SamplingParams | None = None,
        validate_configs: bool = False,
    ) -> None: ...

    def prefill_forward(
        self,
        tokens: torch.Tensor,
        *,
        page_table: torch.Tensor | None = None,
        kv_cache: list | None = None,
        prompt_lens: torch.Tensor,
        start_pos: torch.Tensor | None = None,
        enable_trace: bool = False,
        sampling_params: SamplingParams | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor | None]: ...

    def decode_forward(
        self,
        tokens: torch.Tensor,
        start_pos: torch.Tensor,
        *,
        page_table: torch.Tensor | None = None,
        kv_cache: list | None = None,
        enable_trace: bool = True,
        read_from_device: bool = True,
        sampling_params: SamplingParams | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]: ...

    def make_decode_state(
        self,
        *,
        first_token: torch.Tensor,
        prompt_lens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]: ...

    def advance_decode_state(
        self,
        *,
        current_tokens: torch.Tensor,
        current_pos: torch.Tensor,
        next_tokens: torch.Tensor,
        active_batch_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]: ...

    def synchronize(self) -> None: ...
```

Notes:

- `compile()` owns executor-specific padding and compile-shape decisions
- `make_decode_state()` owns creation of executor-native decode tensors
- `advance_decode_state()` owns in-place decode-state updates with any required padding or lane masking
- `synchronize()` replaces `hasattr(executor, "mesh_device")`

## Responsibility Split

### `run_perf_benchmark()` owns

- validating benchmark inputs
- calling compile once before timing
- measuring prefill TTFT
- measuring decode iteration timings
- extracting generated tokens for active users
- returning `PerfBenchmarkResult`

### executor owns

- compile-time shape adaptation
- prefill fake-lane suppression for padded compile batches
- decode tensor capacity and layout
- device synchronization implementation
- any trace-specific shape restrictions

## Proposed API Changes

### 1. Make `prompt_lens` required

Reason:

- it is semantic input, not a convenience override
- fallback to full `tokens.shape[1]` is only valid for rectangular fully-live prompt batches
- mixed-length prompt batches should not silently degrade into incorrect measurements

Target:

```python
def run_perf_benchmark(
    executor: BenchmarkExecutor,
    *,
    tokens: torch.Tensor,
    prompt_lens: torch.Tensor,
    kv_cache: list,
    page_table: torch.Tensor | None = None,
    num_decode_tokens: int = 128,
    start_pos: torch.Tensor | None = None,
    enable_trace: bool = True,
    sampling_params: SamplingParams | None = None,
) -> PerfBenchmarkResult:
```

### 2. Fix `start_pos` type

Current type annotation uses `list[int] | None`.

Target type:

```python
start_pos: torch.Tensor | None
```

Reason:

- executor interfaces already expect `torch.Tensor | None`
- current annotation is simply wrong

### 3. Remove `max_batch_size` from the benchmark-helper API

Reason:

- it is executor policy leakage
- active-batch benchmark code should not know the executor's decode-capacity shape contract

If an executor needs a configured decode width, it should already know that from `model_args`, constructor state, or internal compile policy.

### 4. Replace attribute probing with a real sync API

Current behavior:

```python
if hasattr(executor, "mesh_device"):
    ttnn.synchronize_device(executor.mesh_device)
```

Target:

```python
executor.synchronize()
```

Reason:

- this removes implementation leakage from the helper
- it gives a stable protocol for non-TTNN executors too

## Validation Rules in `run_perf_benchmark()`

The helper should validate semantic inputs early.

Required assertions:

```python
assert tokens.dim() == 2
assert prompt_lens.dim() == 1
assert prompt_lens.shape[0] == tokens.shape[0]
assert torch.all(prompt_lens > 0)
assert torch.all(prompt_lens <= tokens.shape[1])

if start_pos is not None:
    assert start_pos.dim() == 1
    assert start_pos.shape[0] == tokens.shape[0]
    assert torch.all(start_pos >= 0)
    assert torch.all(start_pos <= prompt_lens)
```

Optional page-table validation can remain executor-side because it depends on KV-cache and paging policy details.

## Migration Plan

### Phase 1: Stabilize helper contract

- make `prompt_lens` required
- change `start_pos` annotation to `torch.Tensor | None`
- remove `max_batch_size` from `run_perf_benchmark()`
- add semantic validation asserts
- update docstring to describe active-batch semantic inputs only

### Phase 2: Move executor-specific logic into executors

- move compile padding logic out of `run_perf_benchmark()` and into `executor.compile()`
- make executors internally handle fake compile lanes for prefill warmup
- add `synchronize()` to executor interfaces
- add `make_decode_state()` and `advance_decode_state()` helpers, or equivalent internalized logic

### Phase 3: Update callers

- update llama demo benchmark call sites to pass explicit `prompt_lens`
- remove `max_batch_size` plumbing from benchmark call sites
- keep caller responsibility limited to semantic inputs and cache/page-table setup

### Phase 4: Clean up executor docs

- update `compile()`, `prefill_forward()`, and `decode_forward()` docstrings to separate active-batch semantics from internal compile capacity
- document any executor-private padding assumptions inside executor code, not benchmark helpers

## Non-Goals

- changing the measured TTFT/decode methodology
- redesigning executor tracing in this change
- changing page-table layout semantics
- changing KV-cache allocation policy

## Acceptance Criteria

The cleanup is done when all of the following are true:

- `run_perf_benchmark()` no longer pads compile inputs itself
- `run_perf_benchmark()` no longer accepts `max_batch_size`
- `run_perf_benchmark()` requires `prompt_lens`
- `run_perf_benchmark()` no longer probes `executor.mesh_device`
- executor implementations fully own compile-shape adaptation
- the llama performance benchmark still passes for `batch-1` and `batch-32`
- the previous fake-user prefill bug does not regress
- decode compile still receives the correct executor-native batch shape

## Immediate Tactical Recommendation

Before the full cleanup lands, keep the current tactical fix:

- compile with full executor decode width
- execute prefill compile only for real active users
- skip zero-new-token users in prefill loops

That preserves correctness while the generic-helper refactor is in progress.
<!-- END VERBATIM: models/common/models/RUN_PERF_BENCHMARK_PLAN.md -->

---

## Llama 3.1 8B Notes

<a id="source-07-models-common-models-llama3-8b-debug-md"></a>

### Source 07: `models/common/models/llama3_8b/DEBUG.md`

[Back to Source Index](#source-index) | [Previous: `models/common/models/RUN_PERF_BENCHMARK_PLAN.md`](#source-06-models-common-models-run-perf-benchmark-plan-md) | [Next: `models/common/models/llama3_8b/EXECUTOR_ARCHITECTURE.md`](#source-08-models-common-models-llama3-8b-executor-architecture-md)

<!-- BEGIN VERBATIM: models/common/models/llama3_8b/DEBUG.md -->
- For changes like @models/common/modules/lm_head/lm_head_1d.py:382-385 @models/common/modules/mlp/mlp_1d.py:1024-1025 , the correct way to apply them is to change the configuration of that module when building the model.
- use `tt-smi -r` to reset the device if the device is in an bad state.
- always run the test in background and check the terminal file to see if there are any errors.

# buggy patterns

## deallocate after async ops
```python
        gathered_out = ttnn.experimental.all_gather_async(
            tt_out,
            persistent_output_buffer=None,
            dim=3,
            multi_device_global_semaphore=cfg.tt_ccl.get_and_cycle_ag_semaphore_handles(),
            num_links=cfg.tt_ccl.get_num_links(),
            topology=default_topology(cfg.mesh_device),
            memory_config=tt_out.memory_config(),
            barrier_semaphore=cfg.tt_ccl.get_and_cycle_barrier_semaphore_handle(),
            chunks_per_sync=10,
            num_workers_per_link=2,
            num_buffers_per_channel=2,
        )
        tt_out.deallocate(True)
```
The main concern is that all_gather_async is an asynchronous operation, and you're immediately deallocating the input tensor with tt_out.deallocate(True) ccl.py:380-381 . This could cause issues if:

-     The operation hasn't completed - The input tensor might still be needed by the CCL operation
-     Memory management conflicts - Early deallocation could interfere with the async operation's buffer management
<!-- END VERBATIM: models/common/models/llama3_8b/DEBUG.md -->

<a id="source-08-models-common-models-llama3-8b-executor-architecture-md"></a>

### Source 08: `models/common/models/llama3_8b/EXECUTOR_ARCHITECTURE.md`

[Back to Source Index](#source-index) | [Previous: `models/common/models/llama3_8b/DEBUG.md`](#source-07-models-common-models-llama3-8b-debug-md) | [Next: `models/common/models/llama3_8b/EXECUTOR_REFACTOR_PLAN.md`](#source-09-models-common-models-llama3-8b-executor-refactor-plan-md)

<!-- BEGIN VERBATIM: models/common/models/llama3_8b/EXECUTOR_ARCHITECTURE.md -->
# Executor Architecture Plan

## Overview

This document describes the two-level executor architecture for LLM inference on TT devices. The design uses **composition**: model-specific executors contain a reusable executor engine that handles compile flow, state management, and trace mechanics.

## Goals

1. **Trace capture during compile** — not lazy, but explicit per compile request
2. **Identical public API** — Eager and Traced executors accept the same inputs, produce the same outputs
3. **Debuggable** — swap Traced for Eager with same model to isolate trace bugs from model bugs
4. **Composition over inheritance** — model executors contain `EagerLLMExecutor` / `TracedLLMExecutor`, not inherit from them
5. **Large engine, small model executor** — model-specific code is minimal

## Architecture

### Design Principle: Thick Engine, Thin Model Executor

**Engine owns the implementation** — `prefill_forward()`, `decode_forward()`, KV cache allocation, trace capture/replay, output processing. These are common to all decoder-only LLMs.

**Model executor owns the model** — passes the transformer model to the engine. The engine does the work. Model-specific details come from `model.model_args`.

No new abstractions. The challenge is figuring out what's common to all LLMs (goes in engine) vs model-specific (comes from `model.model_args`).

### What's Common to All Decoder-Only LLMs

| Component | Why it's common |
|-----------|-----------------|
| Prefill flow | tokens → embed → transformer → logits → process output |
| Decode flow | token → embed → transformer → logits → process output |
| Chunked prefill loop | Paged attention pattern, block-based KV cache |
| KV cache allocation | Paged attention with configurable block size |
| Output processing | Untilize, gather across devices, slice vocab |
| Trace capture/replay | ttnn trace API, same pattern for all models |
| Trace key computation | Prefill: padded seq_len. Decode: bool(sampling_params) |
| `warmup_model_prefill()` | Iterate seq_lens, compile each |

### What's Model-Specific (Comes with the Model Object)

The model object is self-contained. It carries everything model-specific:

| Attribute | What it provides |
|-----------|-----------------|
| `model.model_args` | Dimensions, max_seq_len, vocab_size, cluster_shape, etc. |
| `model.embed_prefill()` / `model.embed_decode()` | Model-specific embedding |
| `model.rope_setup` | Model-specific rotation matrices |
| `model.prefill_forward()` / `model.decode_forward()` | The actual transformer forward |
| `model.layers`, `model.norm`, `model.lm_head` | Model architecture |

The engine only needs `(model, mesh_device)`. Everything else comes from the model.

### Two-Level Structure

```
models/common/models/
├── executor.py                              (shared engine + loop policies)
│   ├── EagerLLMExecutor                     (thick: owns prefill/decode implementation)
│   ├── TracedLLMExecutor                    (thick: + trace capture/replay)
│   ├── run_teacher_forcing()                (loop policy function)
│   └── run_perf_benchmark()                 (loop policy function)
│
└── llama3_8b/
    └── model.py                             (model + thin executors)
        ├── Llama3Transformer1D              (the model)
        ├── EagerLlamaExecutor               (thin: passes Llama model to EagerLLMExecutor)
        └── TracedLlamaExecutor              (thin: passes Llama model to TracedLLMExecutor)

models/common/tests/demos/llama3_8b/
└── demo.py                                  (accuracy + performance tests)
```

**Design principles:**
- Model and executors live together in `model.py` — executors are thin wrappers that just wire the model to the engine
- Demos/tests live in `tests/` directory, separate from library code
- Loop policy functions (`run_teacher_forcing`, `run_perf_benchmark`) are in the shared engine file, not model-specific

### Executor Engines (Thick — Own the Implementation)

Engines own `prefill_forward()` and `decode_forward()` directly. Not callbacks.

#### EagerLLMExecutor

```python
class EagerLLMExecutor:
    """Eager executor engine — owns prefill/decode implementation.

    Common LLM operations live here. Model-specific details come from
    the model object (model.model_args).
    """

    def __init__(self, model, mesh_device: ttnn.MeshDevice):
        self.model = model          # Transformer with .prefill_forward(), .decode_forward(), .model_args
        self.mesh_device = mesh_device
        self._kv_cache = None

    @property
    def model_args(self):
        return self.model.model_args

    # === KV Cache (common to all LLMs) ===

    def allocate_kv_cache(self, kv_cache_shape, dtype, num_layers) -> list:
        """Allocate paged KV cache on device. Common pattern for all LLMs."""
        # ... implementation ...
        self._kv_cache = kv_cache
        self.model.set_kv_cache(kv_cache)
        return kv_cache

    # === Compile ===

    def compile_prefill(self, *, tokens: torch.Tensor, page_table=None, **kwargs) -> torch.Tensor:
        """Compile prefill for specific inputs. Returns logits from warmup run."""
        logits = self.prefill_forward(tokens, page_table=page_table, **kwargs)
        ttnn.synchronize_device(self.mesh_device)
        return logits

    def compile_decode(self, *, tokens: torch.Tensor, start_pos: torch.Tensor, **kwargs) -> None:
        """Compile decode for specific inputs. One warmup run, discard output."""
        self.decode_forward(tokens, start_pos, **kwargs)
        ttnn.synchronize_device(self.mesh_device)

    def compile(self, *, prefill_tokens: torch.Tensor, prefill_page_table=None, **kwargs) -> None:
        """Compile prefill + decode. Decode uses argmax of prefill output."""
        logits = self.compile_prefill(tokens=prefill_tokens, page_table=prefill_page_table, **kwargs)
        decode_tokens = torch.argmax(logits[:, -1:, :], dim=-1)
        decode_start_pos = torch.tensor([prefill_tokens.shape[-1]])
        self.compile_decode(tokens=decode_tokens, start_pos=decode_start_pos, **kwargs)

    # === Warmup (iterates seq_lens) ===

    def warmup_model_prefill(self, seq_lens: list[int], make_tokens, make_page_table) -> None:
        """Compile prefill for multiple sequence lengths. Caller provides input factories."""
        for seq_len in seq_lens:
            tokens = make_tokens(seq_len)
            page_table = make_page_table(seq_len)
            self.compile_prefill(tokens=tokens, page_table=page_table)

    # === Forward (engine owns the implementation) ===

    def prefill_forward(self, tokens, page_table=None, kv_cache=None,
                        prompt_lens=None, empty_slots=None, start_pos=None,
                        sampling_params=None) -> torch.Tensor:
        """Per-user prefill loop with chunked prefill + prefix caching.

        This is the actual implementation, not a callback.
        Common to all decoder-only LLMs with paged attention.
        """
        # ... chunked prefill loop, input prep, output processing ...
        # Uses self.model.prefill_forward() for the actual transformer call
        # Uses self.model_args for dimensions, max_seq_len, etc.

    def decode_forward(self, tokens, start_pos, page_table=None, kv_cache=None,
                       sampling_params=None) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Batched decode step.

        This is the actual implementation, not a callback.
        Common to all decoder-only LLMs.
        """
        # ... input prep, model call, output processing ...

    # === Output Processing (common to all LLMs) ===

    def _process_output_prefill(self, tt_out, last_token_idx) -> torch.Tensor:
        """Device→host for prefill. Untilize, gather, slice vocab."""
        # Common pattern for all LLMs
        ...

    def _process_output_decode(self, tt_out, batch_size) -> torch.Tensor:
        """Device→host for decode. Common pattern for all LLMs."""
        ...

    def cleanup(self) -> None:
        pass
```

#### TracedLLMExecutor

```python
class TracedLLMExecutor:
    """Traced executor engine — adds trace capture/replay.

    Inherits all the common LLM logic, adds trace mechanics.
    """

    def __init__(self, model, mesh_device: ttnn.MeshDevice):
        self.model = model
        self.mesh_device = mesh_device
        self._kv_cache = None
        self._traces: dict[Hashable, TraceContext] = {}

    @property
    def model_args(self):
        return self.model.model_args

    # === KV Cache (same as Eager) ===
    def allocate_kv_cache(self, kv_cache_shape, dtype, num_layers) -> list: ...

    # === Compile (adds trace capture) ===

    def compile_prefill(self, *, tokens: torch.Tensor, page_table=None, **kwargs) -> torch.Tensor:
        """Compile prefill — warmup + capture trace. Returns logits from warmup."""
        trace_key = self._get_prefill_trace_key(tokens)
        if trace_key in self._traces:
            return None  # Already compiled, no logits to return
        # Warmup (uses inherited prefill_forward)
        logits = self._run_prefill_eager(tokens, page_table=page_table, **kwargs)
        ttnn.synchronize_device(self.mesh_device)
        # Capture
        self._traces[trace_key] = self._capture_prefill_trace(tokens, page_table, **kwargs)
        return logits

    def compile_decode(self, *, tokens: torch.Tensor, start_pos: torch.Tensor,
                       sampling_params=None, **kwargs) -> None:
        """Compile decode — warmup + capture trace."""
        trace_key = self._get_decode_trace_key(sampling_params)
        if trace_key in self._traces:
            return
        # Warmup
        self._run_decode_eager(tokens, start_pos, sampling_params=sampling_params, **kwargs)
        ttnn.synchronize_device(self.mesh_device)
        # Capture
        self._traces[trace_key] = self._capture_decode_trace(tokens, start_pos, sampling_params, **kwargs)

    def compile(self, *, prefill_tokens: torch.Tensor, prefill_page_table=None, **kwargs) -> None:
        """Compile prefill + decode. Decode uses argmax of prefill output."""
        logits = self.compile_prefill(tokens=prefill_tokens, page_table=prefill_page_table, **kwargs)
        if logits is None:
            return  # Already compiled
        decode_tokens = torch.argmax(logits[:, -1:, :], dim=-1)
        decode_start_pos = torch.tensor([prefill_tokens.shape[-1]])
        self.compile_decode(tokens=decode_tokens, start_pos=decode_start_pos, **kwargs)

    def warmup_model_prefill(self, seq_lens: list[int], make_tokens, make_page_table) -> None:
        """Compile prefill for multiple sequence lengths. Caller provides input factories."""
        for seq_len in seq_lens:
            tokens = make_tokens(seq_len)
            page_table = make_page_table(seq_len)
            self.compile_prefill(tokens=tokens, page_table=page_table)

    # === Forward (replay or fallback) ===

    def prefill_forward(self, tokens, page_table=None, **kwargs) -> torch.Tensor:
        """Run prefill — replay trace if available, else eager."""
        trace_key = self._get_prefill_trace_key(tokens)
        if trace_key in self._traces:
            return self._replay_prefill(self._traces[trace_key], tokens, page_table, **kwargs)
        return self._run_prefill_eager(tokens, page_table, **kwargs)

    def decode_forward(self, tokens, start_pos, sampling_params=None, **kwargs) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Run decode — replay trace if available, else eager."""
        trace_key = self._get_decode_trace_key(sampling_params)
        if trace_key in self._traces:
            return self._replay_decode(self._traces[trace_key], tokens, start_pos, sampling_params, **kwargs)
        return self._run_decode_eager(tokens, start_pos, sampling_params, **kwargs)

    # === Trace Key Computation (common to all LLMs) ===

    def _get_prefill_trace_key(self, tokens) -> Hashable:
        return get_padded_prefill_len(tokens.shape[-1])

    def _get_decode_trace_key(self, sampling_params) -> Hashable:
        return sampling_params is not None

    # === Trace Capture/Replay (common to all LLMs) ===

    def _capture_prefill_trace(self, tokens, page_table, **kwargs) -> TraceContext:
        """Capture prefill trace. Common pattern for all LLMs."""
        ...

    def _capture_decode_trace(self, tokens, start_pos, sampling_params, **kwargs) -> TraceContext:
        """Capture decode trace. Common pattern for all LLMs."""
        ...

    def _replay_prefill(self, ctx, tokens, page_table, **kwargs) -> torch.Tensor:
        """Replay prefill trace. Common pattern for all LLMs."""
        ...

    def _replay_decode(self, ctx, tokens, start_pos, sampling_params, **kwargs) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Replay decode trace. Common pattern for all LLMs."""
        ...

    def cleanup(self) -> None:
        """Release all captured traces."""
        for ctx in self._traces.values():
            ttnn.release_trace(self.mesh_device, ctx.trace_id)
        self._traces.clear()
```

### Model Executors (Thin — Just Pass Model to Engine)

The model executors are thin wrappers. They just pass the model to the engine.

```python
class EagerLlamaExecutor:
    """Thin wrapper: passes Llama model to EagerLLMExecutor."""

    def __init__(self, model: Llama3Transformer1D, mesh_device: ttnn.MeshDevice):
        self._engine = EagerLLMExecutor(model, mesh_device)

    # Delegate everything to engine
    def allocate_kv_cache(self, *args, **kwargs):
        return self._engine.allocate_kv_cache(*args, **kwargs)

    def compile_prefill(self, **kwargs):
        return self._engine.compile_prefill(**kwargs)

    def compile_decode(self, **kwargs):
        self._engine.compile_decode(**kwargs)

    def compile(self, **kwargs):
        self._engine.compile(**kwargs)

    def warmup_model_prefill(self, seq_lens, make_tokens, make_page_table):
        self._engine.warmup_model_prefill(seq_lens, make_tokens, make_page_table)

    def prefill_forward(self, tokens, **kwargs):
        return self._engine.prefill_forward(tokens, **kwargs)

    def decode_forward(self, tokens, start_pos, **kwargs):
        return self._engine.decode_forward(tokens, start_pos, **kwargs)

    def cleanup(self):
        self._engine.cleanup()


class TracedLlamaExecutor:
    """Thin wrapper: passes Llama model to TracedLLMExecutor."""

    def __init__(self, model: Llama3Transformer1D, mesh_device: ttnn.MeshDevice):
        self._engine = TracedLLMExecutor(model, mesh_device)

    # Delegate everything to engine (identical to EagerLlamaExecutor)
    def allocate_kv_cache(self, *args, **kwargs):
        return self._engine.allocate_kv_cache(*args, **kwargs)

    def compile_prefill(self, **kwargs):
        return self._engine.compile_prefill(**kwargs)

    def compile_decode(self, **kwargs):
        self._engine.compile_decode(**kwargs)

    def compile(self, **kwargs):
        self._engine.compile(**kwargs)

    def warmup_model_prefill(self, seq_lens, make_tokens, make_page_table):
        self._engine.warmup_model_prefill(seq_lens, make_tokens, make_page_table)

    def prefill_forward(self, tokens, **kwargs):
        return self._engine.prefill_forward(tokens, **kwargs)

    def decode_forward(self, tokens, start_pos, **kwargs):
        return self._engine.decode_forward(tokens, start_pos, **kwargs)

    def cleanup(self):
        self._engine.cleanup()
```

**Note**: `EagerLlamaExecutor` and `TracedLlamaExecutor` are now nearly identical pure delegation. The only difference is which engine they instantiate. This could be simplified to a single class with an `eager: bool` flag, but keeping them separate preserves the explicit Eager vs Traced distinction for debugging.

### Compile API Summary

| Method | What it does | Who calls it |
|--------|-------------|--------------|
| `compile_prefill(*, tokens, ...)` | Compile prefill, returns logits | Fine-grained control |
| `compile_decode(*, tokens, ...)` | Compile decode | Fine-grained control |
| `compile(*, prefill_tokens, ...)` | Compile both — decode uses argmax(prefill output) | Simple use cases |
| `warmup_model_prefill(seq_lens, make_tokens, make_page_table)` | Compile prefill for multiple seq_lens | Demos, CI, production |

No magic defaults. `compile()` derives decode input from prefill output — mirrors real inference flow.

## Responsibility Split

### Executor Engines Own (Thick — The Implementation)

| Responsibility | EagerLLMExecutor | TracedLLMExecutor |
|----------------|------------------|-------------------|
| `allocate_kv_cache()` | Paged KV cache allocation | Same |
| `compile_prefill()` | Warmup run + sync, returns logits | Same + capture trace |
| `compile_decode()` | Warmup run + sync | Same + capture trace |
| `compile()` | Compile both, decode uses argmax(prefill) | Same |
| `warmup_model_prefill()` | Iterate seq_lens with caller-provided factories | Same |
| `prefill_forward()` | Input prep → model call → output processing | Replay or fallback |
| `decode_forward()` | Input prep → model call → output processing | Replay or fallback |
| `_process_output_prefill()` | Untilize, gather, slice vocab | Same |
| `_process_output_decode()` | Untilize, gather, slice vocab | Same |
| Trace key computation | N/A | `_get_prefill_trace_key()`, `_get_decode_trace_key()` |
| Trace capture/replay | N/A | `_capture_*_trace()`, `_replay_*()` |
| Trace state | None | `_traces` dict |
| `cleanup()` | No-op | Releases all traces |

The engines are **thick** — they own the full implementation of prefill/decode, including input prep, output processing, chunked prefill loop, KV cache allocation, and trace mechanics.

### Model Executors Own (Thin — Just Wire Model to Engine)

| Responsibility | What It Does |
|----------------|--------------|
| Constructor | Create engine with `(model, mesh_device)` |
| All public methods | Pure delegation to `self._engine.*` |

The model executor is just a thin wrapper that passes the model to the engine. All actual logic lives in the engine.

## Debugging Guarantee

```python
model = Llama3Transformer1D(...)

# Same model — only the executor differs
eager = EagerLlamaExecutor(model, mesh_device)
traced = TracedLlamaExecutor(model, mesh_device)

# Same KV cache allocation
eager.allocate_kv_cache(shape, dtype, num_layers)
traced.allocate_kv_cache(shape, dtype, num_layers)

# Same inputs
prefill_tokens = torch.randint(0, vocab_size, (batch, seq_len))
page_table = torch.arange(num_blocks).unsqueeze(0)

# Same compilation (decode_tokens derived from prefill output)
eager.compile(prefill_tokens=prefill_tokens, prefill_page_table=page_table)
traced.compile(prefill_tokens=prefill_tokens, prefill_page_table=page_table)

# These should match
output_eager = eager.prefill_forward(prefill_tokens, page_table=page_table, ...)
output_traced = traced.prefill_forward(prefill_tokens, page_table=page_table, ...)

# If they differ, bug is in trace capture/replay (TracedLLMExecutor)
# If they match but are wrong, bug is in model logic (shared by both executors)
assert torch.allclose(output_eager, output_traced, atol=1e-3)
```

## Trace Key Design

### Prefill Trace Key

`TracedLLMExecutor._get_prefill_trace_key()` returns `get_padded_prefill_len(tokens.shape[-1])` — the padded sequence length. Prefill kernels are compiled per seq_len due to different attention patterns and sharding configs.

### Decode Trace Key

`TracedLLMExecutor._get_decode_trace_key()` returns `sampling_params is not None` — a boolean. Decode is fixed-shape (batch_size tokens), but the graph differs:
- `sampling_on_device=False`: model forward → logits output
- `sampling_on_device=True`: model forward → sampling → token output

The engine owns `_get_prefill_trace_key()` and `_get_decode_trace_key()` directly — these are common to all LLMs.

## Migration Plan

### Phase 1: Done ✅

Created the two-level executor architecture:

1. **Thick engines** in `models/common/models/executor.py`:
   - `EagerLLMExecutor` — owns prefill/decode implementation, KV cache, output processing
   - `TracedLLMExecutor` — adds trace capture/replay
   - `run_teacher_forcing()` — loop policy function for accuracy measurement
   - `run_perf_benchmark()` — loop policy function for performance measurement

2. **Thin model executors** in `models/common/models/llama3_8b/model.py`:
   - `EagerLlamaExecutor` — passes Llama model to `EagerLLMExecutor`
   - `TracedLlamaExecutor` — passes Llama model to `TracedLLMExecutor`
   - Live alongside the model class — no separate executor.py file

3. **Removed `**kwargs`** from public methods to prevent silent typo bugs

### Phase 2: Done ✅

Implemented input/output contracts and validation:

1. **Type hints + shape comments** on all public method parameters and return types:
   ```python
   def prefill_forward(
       self,
       tokens: torch.Tensor,                    # [batch_size, seq_len], int64
       page_table: torch.Tensor | None = None,  # [batch_size, max_blocks], int32
       kv_cache: list[list[ttnn.Tensor]] | None = None,
       prompt_lens: torch.Tensor | None = None,  # [batch_size], int64
       ...
   ) -> torch.Tensor:                            # [batch_size, 1, vocab_size], float32
   ```

2. **Boundary assertions** at public method entry — fail-fast on bad inputs:
   ```python
   assert tokens.dim() == 2, f"tokens must be [batch_size, seq_len], got {tokens.dim()}D"
   ```

3. **Output spec capture** via `TensorSpec` dataclass during compile:
   - `prefill_output_spec` and `decode_output_spec` captured in both `EagerLLMExecutor` and `TracedLLMExecutor`
   - `TensorSpec.from_tensor()` factory captures shape, dtype, layout, device, memory_config

4. **End-to-end config consistency check** via `_validate_module_configs()` context manager:
   - Instruments compile warmup run to check actual vs declared memory configs
   - Integrated into `compile()` for both prefill and decode modes
   - Reports mismatches between actual tensor `memory_config()` and module's declared expectation

5. **Eager == Traced parity test** in `models/common/tests/test_executor_parity.py`:
   - Validates "same inputs, same outputs, regardless of executor"

### Phase 3: Second Model (When Model #2 Arrives)

1. Create model #2 executors that compose (has-a) the shared engines from `models/common/models/executor.py`
2. Each model executor owns its model-specific logic (input prep, output processing, trace capture/replay)
3. Validate that the engine abstraction works across both models

### Later: Structured Contracts

| Contract | When |
|----------|------|
| `PrefillRequest`/`DecodeRequest` dataclasses | When method signatures become unwieldy |

## Design Decisions

### Public API accepts `torch.Tensor`

The public methods (`prefill_forward`, `decode_forward`, `compile_prefill`, `compile_decode`) accept `torch.Tensor` for all caller-provided inputs (`tokens`, `start_pos`, `page_table`, `prompt_lens`).

**Why not host `ttnn.Tensor` (the tt_cnn pattern)?**
- tt_cnn has a single uniform input tensor. LLM executors have multiple heterogeneous inputs (`tokens` as uint32, `page_table` as int32, etc.) — pushing `from_torch()` to the caller leaks device-level dtype/layout knowledge.
- All public inputs are small, scheduler-produced tensors (32-1024 ints). Conversion cost is negligible.
- Primary consumers (vLLM, demos) naturally produce `torch.Tensor`.

**Why not `LazyWeight`/`LazyBuffer`?**
- Those are for persistent state (model weights, KV cache) that's allocated once and reused. Per-step inputs (`tokens`, `start_pos`) change every call — creating a new `LazyBuffer` per step defeats the purpose.

**What about multi-CQ?**
- Multi-CQ pipelining (overlap `copy_host_to_device_tensor` on CQ1 with execution on CQ0) is internal executor mechanics. The model executor owns the `from_torch()` conversion and the pipelined transfer internally. The caller still passes `torch.Tensor`.

**What's internal vs public:**

| Tensor | Crosses public API? | Type |
|--------|-------------------|------|
| `tokens`, `start_pos`, `page_table` | Yes (caller provides) | `torch.Tensor` |
| RoPE matrices, embeddings | No (executor-internal) | `ttnn.Tensor` |
| KV cache | Allocated via `allocate_kv_cache()` | `ttnn.Tensor` |
| Trace input buffers | No (executor-internal) | `ttnn.Tensor` on device |

## Future Considerations

### Multi-CQ Variants (Performance)

A `TracedLLMExecutorMultiCQ` engine variant that uses two command queues (CQ_OPS=0, CQ_IO=1) for overlapped input transfer + execution. The model executor would swap `self._engine = TracedLLMExecutorMultiCQ(mesh_device)` — no change to the public API.

Can be added when decode throughput becomes bottleneck.

## Loop Policies (Functions)

Teacher forcing and performance benchmarking are **loop policies**, not execution modes. They orchestrate prefill/decode calls but don't own execution logic. They're just functions that take an executor.

### Design Principle

```
Executor (owns execution)          Loop Policy (orchestrates calls)
├── EagerLlamaExecutor       →     ├── run_teacher_forcing(executor, ...)
└── TracedLlamaExecutor      →     └── run_perf_benchmark(executor, ...)
```

The same function works with either executor. This enables:
- **Debugging**: Run teacher forcing with Eager to isolate model bugs
- **CI coverage**: Run teacher forcing with Traced to verify trace accuracy
- **Fair benchmarks**: Run perf benchmark with same executor config

### Why Two Separate Functions

These functions measure fundamentally different things and **must not be combined** into a single pass:

- `run_teacher_forcing()` feeds **ground truth tokens** — the model sees optimal inputs at every step
- `run_perf_benchmark()` feeds **model-predicted tokens** — the model sees its own (possibly wrong) outputs

Decode latency measured during teacher forcing is "best-case ground-truth-path latency", not real autoregressive latency. Token values affect embedding lookups and attention patterns, so the numbers are not interchangeable.

If CI needs both accuracy and performance, run them sequentially:

```python
accuracy = run_teacher_forcing(executor, ...)
perf = run_perf_benchmark(executor, ...)
```

### Executor Contract

Any executor passed to a loop policy function must implement:

```python
# Required methods
executor.prefill_forward(tokens, page_table=None, ...) -> torch.Tensor
executor.decode_forward(tokens, start_pos, ...) -> tuple[torch.Tensor, torch.Tensor | None]
executor.mesh_device  # For synchronization
```

### run_teacher_forcing

Purpose: Measure model accuracy via teacher forcing (feed ground truth tokens, measure prediction accuracy).

```python
@dataclass
class TeacherForceResult:
    predicted_tokens: list[int]
    reference_top5: torch.Tensor

    def top1_accuracy(self) -> float: ...
    def top5_accuracy(self) -> float: ...


def run_teacher_forcing(
    executor,
    *,
    prompt_tokens: torch.Tensor,
    reference_tokens: torch.Tensor,
    top5_tokens: torch.Tensor,
    kv_cache: list,
    page_table: torch.Tensor | None = None,
) -> TeacherForceResult:
    """Run teacher-forcing accuracy measurement."""
    # 1. Prefill with prompt
    prefill_output = executor.prefill_forward(prompt_tokens, page_table=page_table, kv_cache=kv_cache, ...)
    first_token = torch.argmax(prefill_output, dim=-1)

    # 2. Decode loop: feed ground truth, record predictions
    predicted = [first_token]
    for step in range(1, num_decode_steps):
        gt_token = reference_tokens[prompt_len + step - 1]
        logits, _ = executor.decode_forward(gt_token, current_pos, page_table=page_table, kv_cache=kv_cache, ...)
        predicted.append(torch.argmax(logits, dim=-1))

    return TeacherForceResult(predicted, top5_tokens)
```

Usage:

```python
# Debug accuracy with Eager
eager = EagerLlamaExecutor(model, mesh_device)
eager.allocate_kv_cache(...)
eager.compile(prefill_tokens=prompt, prefill_page_table=page_table)
result = run_teacher_forcing(eager, prompt_tokens=prompt, reference_tokens=reference, top5_tokens=top5, kv_cache=kv_cache)
print(f"Eager top1: {result.top1_accuracy():.2%}")

# Verify Traced matches
traced = TracedLlamaExecutor(model, mesh_device)
traced.allocate_kv_cache(...)
traced.compile(prefill_tokens=prompt, prefill_page_table=page_table)
result = run_teacher_forcing(traced, prompt_tokens=prompt, reference_tokens=reference, top5_tokens=top5, kv_cache=kv_cache)
print(f"Traced top1: {result.top1_accuracy():.2%}")
```

### run_perf_benchmark

Purpose: Measure inference performance (TTFT, tok/s/u) with explicit control over what is measured.

```python
@dataclass
class PerfBenchmarkResult:
    prefill_time_s: float
    decode_times_s: list[float]
    batch_size: int
    num_decode_tokens: int

    @property
    def ttft_ms(self) -> float:
        """Time to first token (ms) — wall-clock time until first token appears."""
        return self.prefill_time_s * 1000

    @property
    def ttft_per_user_ms(self) -> float:
        """Amortized TTFT per user (ms) — prefill time divided across batch."""
        return self.prefill_time_s / self.batch_size * 1000

    @property
    def tok_s_u(self) -> float:
        """Tokens per second per user (steady-state decode)."""
        return len(self.decode_times_s) / sum(self.decode_times_s)

    def meets_target(self, expected: dict, tolerance: float = 0.05) -> dict[str, bool]:
        return {
            "tok_s_u": self.tok_s_u >= expected["tok_s_u"] * (1 - tolerance),
            "ttft_ms": self.ttft_ms <= expected["ttft_ms"] * (1 + tolerance),
        }


def run_perf_benchmark(
    executor,
    *,
    tokens: torch.Tensor,
    kv_cache: list,
    page_table: torch.Tensor | None = None,
    num_decode_tokens: int = 128,
    sampling_params: SamplingParams | None = None,  # Explicit choice!
    warmup_decode_steps: int = 10,
) -> PerfBenchmarkResult:
    """Run timed prefill + decode loop."""
    batch_size = tokens.shape[0]

    # Timed prefill
    start = time.perf_counter()
    prefill_output = executor.prefill_forward(tokens, page_table=page_table, kv_cache=kv_cache, ...)
    ttnn.synchronize_device(executor.mesh_device)
    prefill_time = time.perf_counter() - start

    # Warmup decode (excluded from timing)
    for _ in range(warmup_decode_steps):
        executor.decode_forward(next_token, current_pos, sampling_params=sampling_params, ...)
        ttnn.synchronize_device(executor.mesh_device)

    # Timed decode
    decode_times = []
    for _ in range(num_decode_tokens):
        start = time.perf_counter()
        output, _ = executor.decode_forward(next_token, current_pos, sampling_params=sampling_params, ...)
        ttnn.synchronize_device(executor.mesh_device)
        decode_times.append(time.perf_counter() - start)
        next_token = ...  # from output

    return PerfBenchmarkResult(prefill_time, decode_times, batch_size, num_decode_tokens)
```

### Sampling Boundary

The `sampling_params` argument explicitly controls what is measured:

| `sampling_params` | Decode Path | What's Measured |
|-------------------|-------------|-----------------|
| `None` | model forward → logits → host argmax | Model inference only |
| `SamplingParams(...)` | model forward → on-device sampling → token | Model + sampling kernel |

This matters because TTTv1 has two materially different decode paths:
- On-device sampling (faster, includes sampling kernel)
- Host logits readback + argmax (slower, excludes sampling kernel)

**Benchmark code must explicitly choose** which path it measures.

```python
# Benchmark WITHOUT device sampling (model-only)
result_logits = run_perf_benchmark(
    traced,
    tokens=tokens, kv_cache=kv_cache, page_table=page_table,
    num_decode_tokens=128,
    sampling_params=None,  # Explicit: no device sampling
)

# Benchmark WITH device sampling (end-to-end)
result_sampling = run_perf_benchmark(
    traced,
    tokens=tokens, kv_cache=kv_cache, page_table=page_table,
    num_decode_tokens=128,
    sampling_params=SamplingParams(temperature=0.0, top_k=1),  # Explicit: device sampling
)

# These numbers are NOT comparable — different things measured
print(f"Model-only tok/s/u: {result_logits.tok_s_u:.1f}")
print(f"End-to-end tok/s/u: {result_sampling.tok_s_u:.1f}")
```

### Current State vs Target State

| Component | Current | Target |
|-----------|---------|--------|
| `TeacherForceExecutor` | Class with `run()` method | `run_teacher_forcing()` function |
| `PerfBenchmarkExecutor` | Class with `run()` method | `run_perf_benchmark()` function |
| Sampling control | Implicit via config | Explicit `sampling_params` argument |
| Executor choice | Often implicit | Explicit Eager vs Traced |

### Migration

1. **Extract** `TeacherForceExecutor.run()` → `run_teacher_forcing()` function
2. **Extract** `PerfBenchmarkExecutor.run()` → `run_perf_benchmark()` function
3. **Keep classes as thin wrappers** for backward compatibility if needed
4. **Update demos** to use functions directly and pass `sampling_params` explicitly
<!-- END VERBATIM: models/common/models/llama3_8b/EXECUTOR_ARCHITECTURE.md -->

<a id="source-09-models-common-models-llama3-8b-executor-refactor-plan-md"></a>

### Source 09: `models/common/models/llama3_8b/EXECUTOR_REFACTOR_PLAN.md`

[Back to Source Index](#source-index) | [Previous: `models/common/models/llama3_8b/EXECUTOR_ARCHITECTURE.md`](#source-08-models-common-models-llama3-8b-executor-architecture-md) | [Next: `models/common/models/llama3_8b/HF_ADAPTOR_PATTERN.md`](#source-10-models-common-models-llama3-8b-hf-adaptor-pattern-md)

<!-- BEGIN VERBATIM: models/common/models/llama3_8b/EXECUTOR_REFACTOR_PLAN.md -->
# Executor Refactor Plan

## Goal

Refactor `models/common/models/llama3_8b/executor.py` to match the `tt_cnn` executor architecture more closely:

- `EagerLlamaExecutor` is the simple, correctness-first reference path
- `TracedLlamaExecutor` is a separate implementation with explicit trace ownership
- both executors expose `compile` APIs
- eager and traced code paths are intentionally separated, even if that means some duplication

This plan optimizes for the simplest code, not maximum helper sharing.

## Design Principles

1. `EagerLlamaExecutor` is mainly for debugging and correctness work.
  - Optimize for readability and semantic trustworthiness.
  - It is acceptable to duplicate logic from traced if that keeps eager simple.
2. `TracedLlamaExecutor` should own all trace-specific concerns.
  - host-side staging
  - persistent trace input buffers
  - trace capture/replay registries
  - replay-specific lifecycle
3. Eager and traced should be separate code paths.
  - Do not force them through shared prep helpers if the shared helper becomes trace-shaped.
  - Small pure helpers are fine.
  - Trace-oriented helper APIs should not live on `EagerLlamaExecutor`.
4. `compile()` must exist on both executors and should compile both prefill and decode in a single call by default.
  - Same method name on both executors
  - Different internal meaning is acceptable
  - `compile_prefill()` and `compile_decode()` remain the mode-specific entry points
  - `compile()` is the convenience lifecycle method for "prepare this executor fully for execution"
  - all three methods are useful for different use cases:
    - `compile()` for full executor bring-up
    - `compile_prefill()` when only prefill needs warming/capture
    - `compile_decode()` when only decode needs warming/capture
5. Match `tt_cnn` architecturally over matching `simple_text_demo.py` structurally.

## Target Shape

### `EagerLlamaExecutor`

Responsibilities:

- allocate KV cache
- prepare eager prefill inputs
- prepare eager decode inputs
- execute eager prefill
- execute eager decode
- optional eager sampling path
- output processing
- correctness-oriented compile warmup

Non-responsibilities:

- no trace input buffer management
- no `copy_host_to_device`
- no trace registry state
- no trace capture logic
- no trace replay logic

Target API:

```python
class EagerLlamaExecutor:
    def __init__(self, model, mesh_device, model_args=None): ...

    def allocate_kv_cache(self, kv_cache_shape, dtype, num_layers): ...

    def compile(self, *, prefill=None, decode=None): ...
    def compile_prefill(self, *, tokens, page_table=None, kv_cache=None, prompt_lens=None, empty_slots=None, start_pos=None, sampling_params=None): ...
    def compile_decode(self, *, tokens, start_pos, page_table=None, kv_cache=None, sampling_params=None): ...

    def prefill_forward(self, tokens, page_table=None, kv_cache=None, prompt_lens=None, empty_slots=None, sampling_params=None, start_pos=None, **kwargs): ...
    def decode_forward(self, tokens, start_pos, page_table=None, kv_cache=None, sampling_params=None, read_from_device=True, **kwargs): ...

    def cleanup(self): ...
```

Notes:

- `compile_prefill()` and `compile_decode()` should just execute representative runs once and discard outputs.
- They are compile/warmup entry points, not trace APIs.
- Eager should return logits by default unless sampling is explicitly requested.
- `compile()` should exist because it matches the `tt_cnn` lifecycle shape and gives a single "prepare everything" entry point.

### `TracedLlamaExecutor`

Responsibilities:

- allocate KV cache
- compile/capture prefill traces
- compile/capture decode traces
- own host trace input prep
- own trace input buffers
- own trace replay
- own trace cleanup

Non-responsibilities:

- should not depend on eager helpers that are trace-agnostic only by accident
- should not leak trace staging APIs into the eager executor surface

Target API:

```python
class TracedLlamaExecutor:
    def __init__(self, model, mesh_device, model_args=None): ...

    def allocate_kv_cache(self, kv_cache_shape, dtype, num_layers): ...

    def compile(self, *, prefill=None, decode=None): ...
    def compile_prefill(self, *, tokens, page_table=None, kv_cache=None, prompt_lens=None, empty_slots=None, start_pos=None, sampling_params=None): ...
    def compile_decode(self, *, tokens, start_pos, page_table=None, kv_cache=None, sampling_params=None): ...

    def prefill_forward(self, tokens, page_table=None, kv_cache=None, prompt_lens=None, empty_slots=None, sampling_params=None, start_pos=None, **kwargs): ...
    def decode_forward(self, tokens, start_pos, page_table=None, kv_cache=None, sampling_params=None, read_from_device=True, **kwargs): ...

    def cleanup(self): ...
```

Notes:

- `compile_prefill()` captures and caches prefill traces keyed by the actual trace key.
- `compile_decode()` captures and caches decode traces keyed by decode mode, especially sampling-on-device vs logits path.
- lazy capture can exist temporarily, but explicit compile should become the intended model.
- `compile()` is the standard "prepare both modes" entry point; the mode-specific compile methods remain available for narrower use cases.

## Teacher Forcing Direction

Current state:

- `TeacherForceExecutor` is a specialized class around eager execution.

Desired direction:

- teacher forcing is a loop policy, not a core execution mode
- move it toward a thin helper/harness over the executor interface rather than a specialized executor taxonomy
- it should work with eager for debugging and traced for CI / replay-path accuracy checks

Proposed end state:

```python
def run_teacher_forcing(
    executor,
    *,
    prompt_tokens,
    reference_tokens,
    top5_tokens,
    kv_cache,
    page_table=None,
    max_batch_size=1,
):
    ...
```

Expected executor contract for this helper:

- `prefill_forward(...)`
- `decode_forward(...)`
- `allocate_kv_cache(...)`

Why:

- easier to reason about than a whole separate executor taxonomy
- keeps eager as the debugging/correctness path
- still allows traced teacher forcing for CI coverage of the traced execution path
- avoids implying teacher forcing is a primitive runtime mode

Transitional note:

- keep `TeacherForceExecutor` short-term if needed for compatibility
- but do not expand its role further
- long-term, prefer a helper that accepts either `EagerLlamaExecutor` or `TracedLlamaExecutor`

## Sampling Boundary

The model already owns `Sampling1D`, so the main architectural question is not where sampling code lives physically, but who decides when to invoke it.

Planned stance:

- eager path: logits-first by default, sampling only when explicitly requested
- traced path: supports both traced decode-with-sampling and traced decode-with-logits
- benchmark code should explicitly choose which one it is measuring

Why this is a legitimate concern:

- In `models/tt_transformers/demo/simple_text_demo.py`, the `batch-1` + `performance` case is configured with non-`None` `sampling_params`, and decode passes `sampling_params=device_sampling_params` into `generator.decode_forward(...)`.
- In that path, `Generator.decode_forward()` sets `sampling_on_device = sampling_params is not None`, so the device run differs depending on whether sampling params are passed.
- In the no-sampling path, TTTv1 falls back to host sampling via `sample_host(...)` after reading logits back.
- We already observed in the TTTv1 vs TTTv2 perf investigation that decode numbers are sensitive to this distinction.

So the concern is not theoretical. TTTv1 already has two materially different decode paths:

- on-device token sampling vs host logits readback + argmax
- traced vs non-traced prefill

## What To Remove From Eager

Remove or keep out of `EagerLlamaExecutor`:

- trace-oriented host input prep helpers
- `copy_host_to_device`
- trace registry state
- trace buffer ownership
- trace replay assumptions
- helpers whose shape is determined by replay buffer layout rather than by eager correctness

## What Is Fine To Duplicate

Duplication is acceptable when it buys clarity in:

- prefill input prep
- decode input prep
- prefill compile path
- decode compile path
- output processing branches

Do not contort eager code to reuse traced infrastructure.

## Compile Semantics

### Eager `compile_*`

Meaning:

- run once to trigger TTNN/kernel compilation
- discard outputs
- synchronize
- no persistent trace state

### Traced `compile_*`

Meaning:

- compile
- capture trace
- allocate/store replay inputs
- make the executor ready for replay

Same method names, intentionally different implementations.

## Suggested Migration Order

### Phase 1: Lock the API

1. Add `compile()`, `compile_prefill()`, and `compile_decode()` to both executors.
2. Keep current lazy behavior as compatibility fallback if needed.
3. Update benchmark/demo code to call compile methods explicitly where appropriate.

### Phase 2: Finish simplifying eager

1. Remove remaining trace-shaped helpers from `EagerLlamaExecutor`.
2. Keep eager prefill/decode prep fully device-oriented.
3. Keep eager readable even if a small amount of logic is duplicated.

### Phase 3: Make traced fully self-owned

1. Move all trace input prep and buffer management into `TracedLlamaExecutor`.
2. Make trace capture explicit in `compile_prefill()` / `compile_decode()`.
3. Make forward methods replay-oriented first, lazy-capture second.

### Phase 4: Teacher forcing cleanup

1. Stop growing `TeacherForceExecutor`.
2. Replace it with a helper/harness over the shared executor contract.
3. Use the same helper with:
   - `EagerLlamaExecutor` for debugging correctness issues
   - `TracedLlamaExecutor` for CI accuracy coverage of the traced path

### Phase 5: Benchmark cleanup

1. Make benchmark code choose explicitly:
  - traced vs eager
  - prefill compile policy
  - device sampling vs host logits
2. Remove ambiguous “same metric name, different measured path” situations.

## Validation Checklist

1. Eager prefill correctness matches current eager behavior.
2. Eager decode correctness matches current eager behavior.
3. Traced prefill replay matches explicit compile capture.
4. Traced decode replay matches explicit compile capture.
5. Teacher forcing still works on eager path.
6. Perf benchmark can run on either executor via the same high-level harness.
7. No eager method requires trace-shaped host tensors or `copy_host_to_device`.

## Non-Goals

- minimizing total line count at all costs
- forcing eager and traced to share the same prep functions
- preserving every current helper name
- making teacher forcing part of low-level decode primitive APIs

## Practical Standard

If a future reviewer can read `EagerLlamaExecutor` without learning anything about trace replay internals, the refactor is going in the right direction.

If a future reviewer can read `TracedLlamaExecutor` without guessing which parts are “just inherited eager behavior”, the refactor is going in the right direction.<!-- END VERBATIM: models/common/models/llama3_8b/EXECUTOR_REFACTOR_PLAN.md -->

<a id="source-10-models-common-models-llama3-8b-hf-adaptor-pattern-md"></a>

### Source 10: `models/common/models/llama3_8b/HF_ADAPTOR_PATTERN.md`

[Back to Source Index](#source-index) | [Previous: `models/common/models/llama3_8b/EXECUTOR_REFACTOR_PLAN.md`](#source-09-models-common-models-llama3-8b-executor-refactor-plan-md) | [Next: `models/common/models/llama3_8b/MODEL_METHOD_SOLUTION.md`](#source-11-models-common-models-llama3-8b-model-method-solution-md)

<!-- BEGIN VERBATIM: models/common/models/llama3_8b/HF_ADAPTOR_PATTERN.md -->
# Llama3 8B HF Adaptor Pattern

## Goal

Use the `llama32_1b` pattern for `llama3_8b`: keep the TTTv2 model graph native and move Hugging Face checkpoint/config adaptation behind a clear adaptor boundary.

This is a design note only. Do not implement this refactor until the code shape is agreed.

## Pattern From `llama32_1b`

`llama32_1b` is already TTTv2-native. It does not use `models.tt_transformers`, and it does not recreate a broad TTTv1 `ModelArgs` surface.

Its pattern is:

| Concern | `llama32_1b` location | Notes |
|---|---|---|
| Native model graph | `model.py` | Owns TTTv2 dataclasses, module construction helpers, executor runtime config, and model forward paths. |
| HF checkpoint loading | `Llama32_1BTransformer1D.from_pretrained(...)` in `model.py` | Loads `AutoConfig` / `AutoModelForCausalLM`, resolves model dimensions, builds TT modules, and attaches executor runtime config. |
| HF tensor/layout conversion | `weight_utils.py` | Converts HF attention, MLP, RMSNorm, embedding, RoPE, and LM-head tensors into TTTv2 layouts. |
| Generator config | `generator.py` | Uses a port-local generator config and calls `Llama32_1BTransformer1D.from_pretrained(...)`; no v1 `ModelArgs`. |
| Demo construction | `demo.py` | Chooses a precision recipe and calls `from_pretrained(...)`; no model-arg adapter layer. |

For `llama3_8b`, we should keep the same conceptual split but move the HF boundary into a dedicated file instead of putting `from_pretrained(...)` and HF tensor helpers directly in `model.py`.

## Proposed `llama3_8b` Shape

Use a lowercase module name to match the repo style:

```text
models/common/models/llama3_8b/
  model.py
  hf_adaptor.py
  generator.py

models/common/tests/demos/llama3_8b/
  demo.py
  demo_utils.py
```

Recommended ownership:

| File | Responsibility |
|---|---|
| `model.py` | Pure TTTv2 model/module graph: configs, transformer block, transformer class, TT module construction, forward paths, executor-facing attributes. |
| `hf_adaptor.py` | Hugging Face boundary: config loading, tokenizer loading, prompt encoding/chat-template primitives, checkpoint loading/conversion, HF-to-TTTv2 weight layout conversion, and a `from_pretrained`-style builder. |
| `models/common/tests/demos/llama3_8b/demo_utils.py` | Local staging point for demo/benchmark input utilities: prompt-file loading, batching, max-prefill clipping, padding to rectangular tensors, and producing `prompt_lens` for executor benchmarks. Shape this so the generic pieces can later move to a shared common demo utility. |
| `generator.py` | Thin serving/vLLM adapter. It should call the HF adaptor or a TTTv2-native constructor, not create or depend on TTTv1 `ModelArgs`. |
| `demo.py` | Demo/test wiring only: chooses optimization mode, batch/sequence settings, and calls the native construction path. |

The key invariant: after the refactor, active `llama3_8b` code should still have no imports from `models.tt_transformers` and no `from_model_args` construction path. Dependency direction should be one-way: `hf_adaptor.py` may import `model.py`, but `model.py` must remain HF/adaptor agnostic.

## Audit Findings From Current `llama3_8b`

The current decoupling split works, but it is more adapter-like than the `llama32_1b` pattern.

### `runtime_args.py`

Current role:

- Replaces the TTTv1 `ModelArgs` object for the active Llama-3.1-8B path.
- Loads HF config/tokenizer/state dict.
- Converts HF checkpoint keys/layouts into the expected TTTv2/Meta-style state dict.
- Owns many model/runtime fields used by `model.py`.
- Provides TTTv1-shaped helper methods such as tensor dtype lookup, math fidelity lookup, memory configs, program configs, cache paths, and CCL tuning.

Assessment:

- This file is the biggest sign that `llama3_8b` is still shaped around replacing old `ModelArgs`.
- Some contents belong in an HF adaptor: HF config loading, tokenizer/chat-template handling, state-dict loading, and HF-to-Meta conversion.
- Some contents belong in native TTTv2 config/build code: program configs, memory configs, precision lookup, topology decisions, cache-path policy.
- The long-term target should be to shrink or delete `runtime_args.py`, not grow it into a new local `ModelArgs`.

### Precision Policy

Previous sidecar role:

- Owns the Llama-3.1-8B precision policy that was formerly obtained through TTTv1 `DecodersPrecision`.
- Provides performance and accuracy recipes, tensor dtype lookup, math fidelity lookup, and the decoder-31 performance override.

Comparison to `llama32_1b`:

- `llama32_1b` keeps precision recipes inline in `model.py` as a frozen dataclass plus module-level constants.
- For `llama3_8b`, the 8B precision policy is more complex and per-layer, but it is still model/runtime policy and belongs in `model.py`.

Applied direction:

- Keep this concept as a TTTv2-owned precision recipe.
- Fold it into `model.py`; precision is runtime/model policy, not an HF concern.

### RoPE Construction

Previous sidecar role:

- Owns local RoPE scaling support and table construction previously sourced from TTTv1.
- Handles `linear` and `llama3` scaling and returns cos/sin tables for `RotarySetup1D`.

Comparison to `llama32_1b`:

- `llama32_1b` builds RoPE tables by calling the HF model's `rotary_emb` module in `weight_utils.build_rope_cos_sin_torch`.
- That makes RoPE table generation part of the HF adaptor boundary rather than a separate local RoPE implementation.

Applied direction:

- Keep RoPE scaling/config interpretation on the HF adaptor side.
- Move the current table construction helpers into `hf_adaptor.py` and delete the standalone `rope.py`.
- `model.py` may consume the generated cos/sin tensors while the HF-derived scaling semantics stay behind the adaptor boundary.

### `input_preprocessing.py`

Current role:

- Holds prompt prefill preprocessing helpers copied out of the old path for the 8B demo/perf workload.
- Converts prompt strings into rectangular prefill token tensors and real per-user prompt lengths for `run_perf_benchmark`.
- Preserves the old TTTv1 perf-demo semantics: encode at natural length, reserve decode-token budget, left-clip overlong prompts, then pad only to the batch maximum prompt length.

Comparison to `llama32_1b`:

- `llama32_1b` has no equivalent sidecar. Its demo is smoke/parity focused and does not carry the old Llama perf-demo prompt prefill helper.
- Other common demos that need this behavior tend to keep similar prompt-tokenization logic close to the demo/benchmark path.
- `llama32_1b`, `llama32_3b`, `llama33_70b`, and `qwen25_72b` carry nearly identical `load_input_prompts(...)` and `tokenize_prompts(...)` helpers directly in their demos.
- Other common demos such as Qwen, Mistral, Phi, and DeepSeek variants carry model-named versions of the same pattern: encode with the HF chat template, reserve decode-token budget, left-clip overlong prompts, pad to the batch maximum length, and return `(tokens, prompt_lens)`.

HF vs demo ownership:

- The HF-related part is limited to tokenizer/chat-template behavior: load/use tokenizer, apply instruct formatting, and expose an `encode_prompt(prompt, instruct=True)` primitive.
- The demo-related part is the actual workload shaping: prompt-file loading, duplicating prompts to batch size, clipping to the perf budget, padding to a rectangular tensor, and returning `prompt_lens` / decode positions.
- Therefore, do not move the whole helper into `hf_adaptor.py`.

Recommendation:

- Split it along the boundary above.
- Put tokenizer construction and prompt encoding/chat-template primitives in `hf_adaptor.py`.
- Put benchmark prompt preprocessing beside the demo in `models/common/tests/demos/llama3_8b/demo_utils.py`, but make it a generic local staging point rather than an 8B-only copy of the old TTTv1 helper.
- Prefer the newer TTTv2 demo API shape: return `(tokens, prompt_lens)`.
- Avoid preserving the old `preprocess_inputs_prefill(...)` return shape unless a caller truly needs it. The current 8B perf path only needs `input_tokens` and real prompt lengths; the old `encoded_prompts`, `prefill_lens`, and `decoding_pos` can collapse into `prompt_lens`.
- Parameterize the tokenizer-specific piece as an `encode_fn` or use an adaptor-owned `encode_prompt(...)`, so the same utility shape can later serve other demos.
- Delete or retire `input_preprocessing.py` once `demo_utils.py` owns the benchmark path.
- If another model needs the exact same benchmark prompt semantics, promote the generic parts from the local demo utility into a shared common demo utility later, for example under `models/common/tests/demos/`.

Potential local API:

```python
def load_input_prompts(path, batch_size, fallback_prompt): ...

def tokenize_prompts_to_batch(
    prompts,
    encode_fn,
    *,
    pad_id,
    max_prefill_len=None,
) -> tuple[torch.Tensor, torch.Tensor]: ...

def preprocess_chat_prompts(
    prompts,
    encode_fn,
    *,
    pad_id,
    max_seq_len,
    reserve_decode_tokens=128,
) -> tuple[torch.Tensor, torch.Tensor]: ...
```

## Suggested Refactor Direction

The cleanest `llama3_8b` target is:

1. Add `hf_adaptor.py`.
2. Move HF-only logic from `runtime_args.py` into `hf_adaptor.py`.
3. Add a `from_pretrained`-style construction path for Llama-3.1-8B that mirrors `llama32_1b`.
4. Keep `model.py` focused on TTTv2 configs/modules and remove reliance on a broad runtime args object.
5. Keep precision as TTTv2-owned policy in `model.py`.
6. Keep RoPE scaling/table construction in `hf_adaptor.py`.
7. Replace `input_preprocessing.py` with a demo-local `demo_utils.py` for benchmark prompt shaping; keep only tokenizer/chat-template primitives in `hf_adaptor.py`.
8. Shape the demo-local utility so its generic prompt-loading/token-batching helpers can later be promoted to a shared common demo utility.

## Success Criteria

- No `models.tt_transformers` imports in active `llama3_8b` model/generator/demo/executor path.
- No `from_model_args` functions or factory calls in active `llama3_8b` path.
- `model.py` no longer needs a broad TTTv1-shaped runtime args replacement.
- HF loading/conversion is isolated in `hf_adaptor.py`.
- `model.py` does not import `hf_adaptor.py`; native runtime config receives normalized model metadata, state-dict loaders, tokenizer/prompt callbacks, and precomputed RoPE tables from the adaptor.
- Existing verified T3K performance parity remains intact.
- Existing top-1/top-5 accuracy remains intact.
<!-- END VERBATIM: models/common/models/llama3_8b/HF_ADAPTOR_PATTERN.md -->

<a id="source-11-models-common-models-llama3-8b-model-method-solution-md"></a>

### Source 11: `models/common/models/llama3_8b/MODEL_METHOD_SOLUTION.md`

[Back to Source Index](#source-index) | [Previous: `models/common/models/llama3_8b/HF_ADAPTOR_PATTERN.md`](#source-10-models-common-models-llama3-8b-hf-adaptor-pattern-md) | [Next: `models/common/models/llama3_8b/PLAN.md`](#source-12-models-common-models-llama3-8b-plan-md)

<!-- BEGIN VERBATIM: models/common/models/llama3_8b/MODEL_METHOD_SOLUTION.md -->
# Investigation: Option B — Model Method for Post-Processing

## Problem Statement

The `TracedLLMExecutor.prefill_forward()` contains Llama-specific code that violates the "thick engine, thin model executor" principle:

```python
from models.common.models.llama3_8b.model import _all_gather_rmsnorm_tensor
# ...
logits = self.model.norm.prefill_forward(ttnn.slice(...))
logits = _all_gather_rmsnorm_tensor(self.model.norm, logits)
logits = self.model.lm_head.forward(logits)
```

## Key Insight

The model **already has this exact logic** in `prefill_forward()` when `get_last_token != -1` (lines 337-349):

```python
# In model.prefill_forward() when get_last_token != -1:
get_last_token_floor = (get_last_token // 32) * 32
x = ttnn.slice(x, (0, 0, get_last_token_floor, 0), (1, 1, get_last_token_floor + 32, x.shape[-1]))
x = self.norm.prefill_forward(x)
x = _all_gather_rmsnorm_tensor(self.norm, x)  # Model-internal!
x = self.lm_head.forward(x)
x = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)
return x
```

The executor is **duplicating** this logic for the traced path. Solution: expose it as a model method.

## Proposed Solution

### 1. Add method to model protocol

Update the protocol comment in `executor.py`:

```python
# Required methods:
#   ...existing methods...
#   - post_process_prefill_output(hidden_states, last_token_idx) -> ttnn.Tensor  # NEW
```

### 2. Implement in Llama model

Add to `llama3_8b/model.py`:

```python
def post_process_prefill_output(
    self,
    hidden_states: ttnn.Tensor,
    last_token_idx: int
) -> ttnn.Tensor:
    """Convert hidden states to final logits for the specified token position.

    This is the post-trace processing for prefill when using get_last_token=-1.

    Args:
        hidden_states: Output from prefill_forward with get_last_token=-1.
                      Shape: [1, 1, seq_len, hidden_dim]
        last_token_idx: Index of the last actual token (not padding).

    Returns:
        Logits for the last token. Shape: [1, 1, 32, vocab_size]
    """
    get_last_token_floor = (last_token_idx // 32) * 32
    x = ttnn.slice(
        hidden_states,
        (0, 0, get_last_token_floor, 0),
        (1, 1, get_last_token_floor + 32, hidden_states.shape[-1]),
    )

    x = self.norm.prefill_forward(x)
    x = _all_gather_rmsnorm_tensor(self.norm, x)

    lm_head_memcfg = self.lm_head.config.input_memcfg
    if lm_head_memcfg is not None and lm_head_memcfg.is_sharded():
        x = ttnn.interleaved_to_sharded(x, lm_head_memcfg)

    x = self.lm_head.forward(x)
    x = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)
    return x
```

### 3. Update executor to use model method

Replace Llama-specific code in `TracedLLMExecutor.prefill_forward()`:

**Before** (lines 1205-1215):
```python
logits = self.model.norm.prefill_forward(
    ttnn.slice(
        logits,
        (0, 0, (last_token_idx // 32) * 32, 0),
        (1, 1, (last_token_idx // 32) * 32 + 32, logits.shape[-1]),
    )
)
logits = _all_gather_rmsnorm_tensor(self.model.norm, logits)
logits = self.model.lm_head.forward(logits)
logits = ttnn.to_memory_config(logits, ttnn.DRAM_MEMORY_CONFIG)
```

**After**:
```python
logits = self.model.post_process_prefill_output(logits, last_token_idx)
```

### 4. Remove the Llama-specific import

Delete from `executor.py`:
```python
from models.common.models.llama3_8b.model import _all_gather_rmsnorm_tensor
```

## Code Changes Summary

| File | Change |
|------|--------|
| `executor.py:1137` | Remove `_all_gather_rmsnorm_tensor` import |
| `executor.py:44-63` | Add `post_process_prefill_output` to protocol |
| `executor.py:1205-1215` | Replace with `self.model.post_process_prefill_output(logits, last_token_idx)` |
| `llama3_8b/model.py` | Add `post_process_prefill_output()` method |

## Comparison: Option A vs Option B

| Aspect | Option A (Full Trace) | Option B (Model Method) |
|--------|----------------------|------------------------|
| **Complexity** | High — requires ttnn API for tensor indices | Low — extract existing code |
| **Performance** | Better — all ops in trace | Same as current — post-processing outside trace |
| **API Blocker** | `ttnn.slice` doesn't accept tensor indices | None |
| **Code Duplication** | Eliminated | Eliminated |
| **Model Changes** | Modify `prefill_forward` signature | Add new method |
| **Engine Changes** | Significant trace redesign | Simple: call model method |
| **Timeline** | Blocked on ttnn API investigation | Can implement now |

## Recommendation

**Implement Option B now** as the practical solution:
- No API blockers
- Minimal code changes
- Eliminates the Llama-specific import
- Establishes the model protocol pattern for future models

**Consider Option A later** if:
- `ttnn` adds tensor-indexed slicing
- Performance profiling shows post-trace overhead is significant

## Implementation Steps

1. Add `post_process_prefill_output()` to `Llama3Transformer1D`
2. Update executor to call the new method
3. Remove `_all_gather_rmsnorm_tensor` import
4. Update protocol comment
5. Test that traced prefill still produces correct output

## Future Model Requirements

When adding a new model (e.g., Mistral), it must implement:
```python
def post_process_prefill_output(self, hidden_states: ttnn.Tensor, last_token_idx: int) -> ttnn.Tensor:
    """Model-specific post-processing for traced prefill output."""
    # Each model implements its own norm + lm_head logic
```

This keeps the engine model-agnostic while allowing model-specific implementations.
<!-- END VERBATIM: models/common/models/llama3_8b/MODEL_METHOD_SOLUTION.md -->

<a id="source-12-models-common-models-llama3-8b-plan-md"></a>

### Source 12: `models/common/models/llama3_8b/PLAN.md`

[Back to Source Index](#source-index) | [Previous: `models/common/models/llama3_8b/MODEL_METHOD_SOLUTION.md`](#source-11-models-common-models-llama3-8b-model-method-solution-md) | [Next: `models/common/models/llama3_8b/STATIC_KV_CACHE_PLAN.md`](#source-13-models-common-models-llama3-8b-static-kv-cache-plan-md)

<!-- BEGIN VERBATIM: models/common/models/llama3_8b/PLAN.md -->
# TTTv2 Llama 3.1-8B: Build Plan

## Overview

Build a complete Llama 3.1-8B model from **purely TTTv2 modules** — no TTTv1 module imports in the forward path. The model lives in `models/common/models/llama3_8b/` and produces:

| File | Purpose |
|------|---------|
| `models/common/models/executor.py` | Shared executor engines (`EagerLLMExecutor`, `TracedLLMExecutor`) + loop policy functions (`run_teacher_forcing`, `run_perf_benchmark`) |
| `models/common/models/llama3_8b/model.py` | TTTv2 Transformer model + thin model executors (`EagerLlamaExecutor`, `TracedLlamaExecutor`) |
| `models/common/models/llama3_8b/generator.py` | `Llama3Generator` — thin vLLM adapter wrapping an executor |
| `models/common/tests/demos/llama3_8b/demo.py` | Teacher-forcing demo with accuracy + performance measurement |

---

## File 1: `model.py` — TTTv2 Transformer

### Design Principles

- **Library, not framework** — each module is explicitly instantiated with typed config
- **No static branching in forward** — separate `decode_forward` / `prefill_forward`
- **No `model_config` dict** — all configs baked into module constructors
- **Direct TTTv2 construction** — the vLLM path should instantiate the real `Llama3Transformer1D`, not route through a TTTv1-style compatibility factory

### Architecture

```
Llama3Transformer1D (1D only — non-TG, covers 1x1, 1x2, 1x8)
├── Embedding1D                     (token lookup)
├── RotarySetup1D                   (cos/sin matrices)
├── TransformerBlock1D × n_layers   (NEW — composing TTTv2 sub-modules)
│   ├── RMSNorm1D  (attention_norm)
│   ├── Attention1D
│   ├── RMSNorm1D  (ff_norm)
│   └── MLP1D
├── RMSNorm1D                       (final norm)
├── LMHead1D                        (output projection)
└── Sampling1D                      (optional on-device sampling)
```

### Classes to implement

#### `TransformerBlock1DConfig` (dataclass)
- Composed from sub-module configs: `Attention1DConfig`, `MLP1DConfig`, `RMSNorm1DConfig` × 2
- No branching axes — Llama 3 doesn't have `pre_ff_norm` / `post_ff_norm`
- Fields: `residual_mem_config` (for decode/prefill), `activation_dtype`

#### `TransformerBlock1D`
- `__init__(attention_norm, attention, ff_norm, feed_forward)` — happy path, takes instantiated sub-modules
- `from_config(config)` — power-user path
- `decode_forward(x, current_pos, rot_mats, page_table, kv_cache)` — no mode branching
- `prefill_forward(x, rot_mats, user_id, page_table, chunk_page_table, chunk_start_idx, kv_cache)` — no mode branching

#### `Llama3Transformer1DConfig` (dataclass)
- `n_layers`, `vocab_size`, `max_batch_size`, `max_seq_len`
- `block_configs: list[TransformerBlock1DConfig]` (per-layer, since precision can differ per layer)
- `embedding_config`, `rope_config`, `norm_config`, `lm_head_config`, `sampling_config`
- `cluster_shape`, `mesh_device`

#### `Llama3Transformer1D`
- `__init__(embedding, rope_setup, layers, norm, lm_head, sampling=None)` — happy path
- `from_config(config)` — power-user path and the primary path used by the generator
- Optional adapter/helper may translate `ModelArgs` into `Llama3Transformer1DConfig`, but the generator should still instantiate the TTTv2 model directly

**Forward methods:**

```python
# kv_cache is passed per-call (not stored on the model). This is mandated by
# vLLM's architecture: TTModelRunner owns the cache and passes it into every
# prefill/decode call. See "KV Cache Design" section below for details.

def decode_forward(self, x, current_pos, rot_mat_idxs, page_table, kv_cache,
                   sampling_on_device=False):
    x_embed = self.embedding.forward(x)
    x_embed = ttnn.unsqueeze_to_4D(x_embed)
    x_embed = ttnn.to_memory_config(x_embed, self.decode_residual_mem_cfg)

    rot_mats = self.rope_setup.decode_forward(rot_mat_idxs)

    for i, layer in enumerate(self.layers):
        x_embed = layer.decode_forward(
            x_embed, current_pos, rot_mats, page_table,
            kv_cache=kv_cache[i] if kv_cache else None,
        )

    x_embed = self.norm.decode_forward(x_embed)
    logits = self.lm_head.forward(x_embed)

    if sampling_on_device and self.sampling:
        tokens, log_probs = self.sampling.decode_forward(logits, ...)
        return tokens, log_probs

    # gather + untilize for host sampling
    if self.num_devices > 1:
        logits = ttnn.experimental.all_gather_async(logits, ...)
    logits = ttnn.untilize(logits, ...)
    return logits, None

def prefill_forward(self, x, rot_mats, user_id, page_table,
                    chunk_page_table, chunk_start_idx, get_last_token, kv_cache):
    x_embed = self.embedding.forward(x)
    x_embed = ttnn.unsqueeze_to_4D(x_embed)

    for i, layer in enumerate(self.layers):
        x_embed = layer.prefill_forward(
            x_embed, rot_mats, user_id, page_table,
            chunk_page_table, chunk_start_idx,
            kv_cache=kv_cache[i] if kv_cache else None,
        )

    if get_last_token != -1:
        x_embed = ttnn.slice(x_embed, ...)

    x_embed = self.norm.prefill_forward(x_embed)
    logits = self.lm_head.forward(x_embed)
    logits = ttnn.to_memory_config(logits, ttnn.DRAM_MEMORY_CONFIG)
    return logits
```

### Key decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Scope | 1D only (no 2D/TG) | Llama 3.1-8B runs on N150/N300/T3K, all non-TG topologies |
| TransformerBlock | New composite, not copied from TTTv1 | TTTv1's `TransformerBlock` has MoE, `pre_ff_norm`, `post_ff_norm`, `DistributedNorm` — none needed for Llama 3 8B |
| Config per layer | Yes | TTTv1 supports per-decoder precision settings (`decoders_optimizations`); must preserve this |
| KV cache | Paged attention (same as TTTv1) | Required for vLLM compatibility |
| `forward()` dispatch | Keep a simple `forward(mode)` that calls `decode_forward` or `prefill_forward` | Convenience for backward compat, but the real work is in the split methods |

### KV Cache Design

**Why per-call?** vLLM's `TTModelRunner` owns the KV cache. It calls `model.allocate_kv_cache()` once during init, stores the result as `self.kv_caches`, and passes it into every `prefill_forward` / `decode_forward` call. The model never stores or manages the cache — this separation lets vLLM swap, resize, or reallocate caches without touching the model.

**Structure:**

```
kv_cache: list[list[ttnn.Tensor]]
         ╰── per layer ──╯
              ╰── [k_cache, v_cache]
```

- One `list[list[ttnn.Tensor]]` per DP rank (for this plan, DP=1 so just one)
- Per layer: `[k_cache_tt, v_cache_tt]` — two ttnn tensors on device
- Shape per tensor: `(max_num_blocks, num_kv_heads, block_size, head_dim)` — e.g. `(1024, 8, 32, 128)` for Llama 3.1-8B with 1024 blocks

**Allocation** (inlined in `generator.py`, not imported):

```python
def allocate_kv_cache(self, kv_cache_shape, dtype, num_layers):
    cache_kv = torch.zeros(kv_cache_shape, dtype=dtype)
    kv_cache = []
    for layer_num in range(num_layers):
        kv_cache_dtype = self._get_kv_cache_dtype(layer_num)
        kv_tt_i = [
            ttnn.as_tensor(
                cache_kv, device=self.mesh_device,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                dtype=kv_cache_dtype,
                cache_file_name=cache_path / f"empty_{kv}cache_paged_attention{kv_cache_shape}",
            )
            for kv in ["k", "v"]
        ]
        kv_cache.append(kv_tt_i)
    return kv_cache
```

**Lifecycle:**

| Phase | Owner | What happens |
|-------|-------|--------------|
| Allocation | `TTModelRunner.initialize_kv_cache()` → `generator.allocate_kv_cache()` | Creates zero-filled ttnn tensors on device per layer |
| Storage | `TTModelRunner.kv_caches` | Stored on the vLLM runner, not the model |
| Per-call | `prefill_forward(kv_cache=...)` / `decode_forward(kv_cache=...)` | Passed through to each layer's attention module |
| Mutation | `Attention1D.decode_forward` / `prefill_forward` | SDPA writes into the cache in-place via page_table indices |
| Deallocation | Implicit (Python GC when runner is destroyed) | No explicit dealloc needed |

### Dependencies (allowed)

- `ttnn` — core
- `models.common.modules.*` — all TTTv2 modules
- `models.common.lightweightmodule.LightweightModule` — base class (no torch dependency)
- `models.common.sampling` — `SamplingParams` (already used by TTTv2 Sampling1D)

### Dependencies (forbidden in forward path)

- `models.tt_transformers.tt.*` — no TTTv1 imports except inside `from_model_args`
- `torch` — no direct torch dependency (except via LazyWeight/tensor_utils indirection)

---

## File 2: Executor + Generator — Composition Pattern

### Design inspiration

`models/tt_cnn/tt/executor.py` demonstrates a clean `compile → execute → cleanup` lifecycle with trace capture as a sibling executor, not a wrapper. The LLM version follows the same pattern but adds prefill/decode dual-mode, paged KV cache, and chunked prefill.

### Two-Level Architecture: Thick Engine, Thin Model Executor

**Engine owns the implementation** — `prefill_forward()`, `decode_forward()`, KV cache allocation, trace capture/replay, output processing. These are common to all decoder-only LLMs.

**Model executor owns the model** — passes the transformer model to the engine. The engine does the work. Model-specific details come from `model.model_args`.

```
models/common/models/
├── executor.py                              (shared engine + loop policies)
│   ├── EagerLLMExecutor                     (thick: owns prefill/decode implementation)
│   ├── TracedLLMExecutor                    (thick: + trace capture/replay)
│   ├── run_teacher_forcing()                (loop policy function)
│   └── run_perf_benchmark()                 (loop policy function)
│
└── llama3_8b/
    └── model.py                             (model + thin executors)
        ├── Llama3Transformer1D              (the model)
        ├── EagerLlamaExecutor               (thin: passes Llama model to EagerLLMExecutor)
        └── TracedLlamaExecutor              (thin: passes Llama model to TracedLLMExecutor)
```

**Design principles:**
- Model and executors live together in `model.py` — executors are thin wrappers that just wire the model to the engine
- Demos/tests live in `tests/` directory, separate from library code
- Loop policy functions (`run_teacher_forcing`, `run_perf_benchmark`) are in the shared engine file, not model-specific

**Key constraints:**
- `TracedLlamaExecutor` is a **sibling** of `EagerLlamaExecutor`, not a decorator around it. Both pass the same model to their respective engines.
- `run_teacher_forcing()` and `run_perf_benchmark()` are **functions**, not classes — they take any executor and orchestrate calls
- `Llama3Generator` takes any executor (typically traced) and adapts its interface to what vLLM's `TTModelRunner` expects

**Debugging guarantee:**
```python
model = Llama3Transformer1D(...)

# Same model — only the executor differs
eager = EagerLlamaExecutor(model, mesh_device)
traced = TracedLlamaExecutor(model, mesh_device)

# Same inputs, same outputs — if they differ, bug is in trace mechanics
output_eager = eager.prefill_forward(tokens, ...)
output_traced = traced.prefill_forward(tokens, ...)
assert torch.allclose(output_eager, output_traced, atol=1e-3)
```

### What we keep from TTTv1 Generator (redistributed to shared engines)

| Capability | TTTv1 lines | New home |
|-----------|-------------|----------|
| Compilation (kernel generation) | ~80 | `compile_prefill()` / `compile_decode()` on `EagerLLMExecutor` / `TracedLLMExecutor` |
| Trace capture/replay | ~200 | `TracedLLMExecutor` (capture in `compile_prefill`/`compile_decode`, replay in `prefill_forward`/`decode_forward`) |
| Trace cleanup | ~10 | `TracedLLMExecutor.cleanup()` |
| Chunked prefill + prefix caching | ~70 | `EagerLLMExecutor` (shared by both engines) |
| Warmup sweeps | ~80 | `TracedLLMExecutor.warmup_model_prefill()` (calls `compile_prefill`/`compile_decode` internally) |
| Input preparation (prefill + decode) | ~90 | `EagerLLMExecutor` |
| Output processing (prefill + decode) | ~60 | `EagerLLMExecutor` |
| On-device sampling integration | ~30 | `EagerLLMExecutor` |
| KV cache allocation | ~35 | `EagerLLMExecutor` |

### What we drop

| TTTv1 code | Lines dropped | Why |
|-----------|--------------|-----|
| Vision / multimodal paths | ~530 | Not Llama 3.1-8B |
| DeepSeek KVDBG | ~15 | Not Llama 3.1-8B |
| Data parallel (`tt_data_parallel > 1`) | ~50 | Non-goal for this PR |
| `generate()`, `chat_completion()`, `text_completion()` | ~142 | Not part of vLLM interface |
| `return_hidden_states` paths | ~25 | Embedding model feature, not needed |
| Duplicate vision trace/decode methods | ~200 | Vision-only variants |

### Shared Executor Engines (Thick — Own the Implementation)

Located in `models/common/models/executor.py`:

```python
class EagerLLMExecutor:
    """Eager executor engine — owns prefill/decode implementation.
    Common LLM operations live here. Model-specific details come from
    the model object (model.model_args)."""

    def __init__(self, model, mesh_device: ttnn.MeshDevice):
        self.model = model
        self.mesh_device = mesh_device
        self._kv_cache = None

    @property
    def model_args(self):
        return self.model.model_args

    # === Forward (engine owns the implementation) ===

    def prefill_forward(self, tokens, page_table=None, kv_cache=None,
                        prompt_lens=None, ...) -> torch.Tensor:
        """Per-user prefill loop with chunked prefill + prefix caching.
        This is the actual implementation, not a callback."""
        ...

    def decode_forward(self, tokens, start_pos, page_table=None,
                       kv_cache=None, ...) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Batched decode step. This is the actual implementation."""
        ...

    # === Compile ===

    def compile_prefill(self, *, tokens: torch.Tensor, ...) -> torch.Tensor:
        """Compile prefill for specific inputs. Returns logits from warmup run."""
        ...

    def compile_decode(self, *, tokens: torch.Tensor, start_pos: torch.Tensor, ...) -> None:
        """Compile decode for specific inputs. One warmup run, discard output."""
        ...

    def compile(self, *, prefill_tokens: torch.Tensor, ...) -> None:
        """Compile prefill + decode. Decode uses argmax of prefill output."""
        ...

    def cleanup(self) -> None:
        pass


class TracedLLMExecutor:
    """Traced executor engine — adds trace capture/replay.
    Inherits all the common LLM logic, adds trace mechanics."""

    def __init__(self, model, mesh_device: ttnn.MeshDevice):
        self.model = model
        self.mesh_device = mesh_device
        self._kv_cache = None
        self._traces: dict[Hashable, TraceContext] = {}

    # Same public API as EagerLLMExecutor, but compile methods capture traces
    # and forward methods replay traces when available

    def cleanup(self) -> None:
        """Release all captured traces."""
        for ctx in self._traces.values():
            ttnn.release_trace(self.mesh_device, ctx.trace_id)
        self._traces.clear()
```

### Model Executors (Thin — Just Pass Model to Engine)

Located in `models/common/models/llama3_8b/model.py` alongside the model:

```python
class EagerLlamaExecutor:
    """Thin wrapper: passes Llama model to EagerLLMExecutor."""

    def __init__(self, model: Llama3Transformer1D, mesh_device: ttnn.MeshDevice):
        self._engine = EagerLLMExecutor(model, mesh_device)

    # Delegate everything to engine
    def allocate_kv_cache(self, *args, **kwargs):
        return self._engine.allocate_kv_cache(*args, **kwargs)

    def prefill_forward(self, tokens, **kwargs):
        return self._engine.prefill_forward(tokens, **kwargs)

    def decode_forward(self, tokens, start_pos, **kwargs):
        return self._engine.decode_forward(tokens, start_pos, **kwargs)

    def cleanup(self):
        self._engine.cleanup()


class TracedLlamaExecutor:
    """Thin wrapper: passes Llama model to TracedLLMExecutor."""

    def __init__(self, model: Llama3Transformer1D, mesh_device: ttnn.MeshDevice):
        self._engine = TracedLLMExecutor(model, mesh_device)

    # Delegate everything to engine (identical to EagerLlamaExecutor)
    ...
```

**Note**: `EagerLlamaExecutor` and `TracedLlamaExecutor` are nearly identical pure delegation. The only difference is which engine they instantiate.

### Loop Policy Functions (Not Classes)

Teacher forcing and performance benchmarking are **loop policies** — they orchestrate prefill/decode calls but don't own execution logic. They're functions that take any executor.

Located in `models/common/models/executor.py`:

```python
def run_teacher_forcing(
    executor,
    *,
    prompt_tokens: torch.Tensor,
    reference_tokens: torch.Tensor,
    top5_tokens: torch.Tensor,
    kv_cache: list,
    page_table: torch.Tensor | None = None,
) -> TeacherForceResult:
    """Run teacher-forcing accuracy measurement.
    Works with either Eager or Traced executor."""
    # 1. Prefill with prompt
    prefill_output = executor.prefill_forward(prompt_tokens, ...)
    # 2. Decode loop: feed ground truth, record predictions
    ...
    return TeacherForceResult(predicted, top5_tokens)


def run_perf_benchmark(
    executor,
    *,
    tokens: torch.Tensor,
    kv_cache: list,
    page_table: torch.Tensor | None = None,
    num_decode_tokens: int = 128,
    sampling_params: SamplingParams | None = None,
    warmup_decode_steps: int = 10,
) -> PerfBenchmarkResult:
    """Run timed prefill + decode loop.
    Works with either Eager or Traced executor."""
    # Timed prefill + warmup decode + timed decode
    ...
    return PerfBenchmarkResult(prefill_time, decode_times, batch_size, num_decode_tokens)
```

**Why functions instead of classes?**
- Same function works with either executor — enables debugging with Eager, CI with Traced
- No state to manage — just orchestration
- Explicit `sampling_params` argument controls what is measured (model-only vs end-to-end)

**Why two separate functions?**
- `run_teacher_forcing()` feeds **ground truth tokens** — measures accuracy
- `run_perf_benchmark()` feeds **model-predicted tokens** — measures realistic performance
- These measure fundamentally different things and **must not be combined**

### `Llama3Generator` — vLLM adapter (thin shim)

```python
class Llama3Generator:
    """vLLM-compatible adapter. Wraps any executor (typically traced).
    Zero trace state, zero warmup state, zero execution logic.
    Just signature adaptation for TTModelRunner."""

    model_capabilities = {"supports_prefix_caching": True}

    def __init__(self, executor):  # EagerLlamaExecutor | TracedLlamaExecutor
        self.executor = executor

    @classmethod
    def initialize_vllm_model(cls, hf_config, mesh_device, max_batch_size,
                              max_seq_len, n_layers=None,
                              optimizations="performance"):
        config = Llama3Transformer1DConfig.from_hf_config(
            hf_config=hf_config, mesh_device=mesh_device,
            max_batch_size=max_batch_size, max_seq_len=max_seq_len,
            n_layers=n_layers, optimizations=optimizations,
            dtype=ttnn.bfloat8_b,
        )
        state_dict = load_llama3_state_dict(config)
        model = Llama3Transformer1D.from_config(config)
        model.load_state_dict(state_dict)
        executor = TracedLlamaExecutor(model, config, mesh_device)
        return cls(executor)

    def prefill_forward(self, *a, **kw):      return self.executor.prefill_forward(*a, **kw)
    def decode_forward(self, *a, **kw):       return self.executor.decode_forward(*a, **kw)
    def allocate_kv_cache(self, *a, **kw):    return self.executor.allocate_kv_cache(*a, **kw)
    def warmup_model_prefill(self, *a, **kw): return self.executor.warmup_model_prefill(*a, **kw)

    @property
    def cache_path(self): return self.executor.config.cache_path
```

### Construction for each use case

```python
# Teacher forcing with Eager (debugging)
model = Llama3Transformer1D.from_config(config)
eager = EagerLlamaExecutor(model, mesh_device)
eager.allocate_kv_cache(...)
eager.compile(prefill_tokens=prompt, prefill_page_table=page_table)
result = run_teacher_forcing(eager, prompt_tokens=prompt, reference_tokens=reference, ...)
print(f"Eager top1: {result.top1_accuracy():.2%}")

# Teacher forcing with Traced (CI accuracy gate)
traced = TracedLlamaExecutor(model, mesh_device)
traced.allocate_kv_cache(...)
traced.compile(prefill_tokens=prompt, prefill_page_table=page_table)
result = run_teacher_forcing(traced, prompt_tokens=prompt, reference_tokens=reference, ...)
print(f"Traced top1: {result.top1_accuracy():.2%}")

# Perf benchmark (with tracing — realistic numbers)
traced = TracedLlamaExecutor(model, mesh_device)
traced.allocate_kv_cache(...)
traced.compile(prefill_tokens=prompt, prefill_page_table=page_table)
result = run_perf_benchmark(traced, tokens=prompt, kv_cache=kv_cache, num_decode_tokens=128)
# result.ttft_ms, result.tok_s_u, result.meets_target(...)

# Perf benchmark (without tracing — for comparison)
result = run_perf_benchmark(EagerLlamaExecutor(model, mesh_device), ...)

# vLLM serving
generator = Llama3Generator.initialize_vllm_model(hf_config, mesh_device, ...)
```

### Key design differences from TTTv1

| Aspect | TTTv1 Generator | TTTv2 Two-Level Architecture |
|--------|----------------|------------------------------|
| Model storage | `self.model: list[Transformer]` (DP list) | `engine.model: Llama3Transformer1D` (single) |
| Config source | `self.model_args: list[ModelArgs]` | `model.model_args` (model carries its own config) |
| Execution logic | Mixed into Generator (~1000 lines) | Shared engines in `executor.py` (~400 lines) |
| Trace state | 6 `defaultdict` fields on Generator | Encapsulated in `TracedLLMExecutor._traces` |
| Model executor | N/A | Thin wrapper (`EagerLlamaExecutor`, `TracedLlamaExecutor`) |
| Warmup | Mixed into Generator | `TracedLLMExecutor.warmup_model_prefill()` |
| Teacher forcing | `enable_trace=False` flag (runtime) | `run_teacher_forcing(executor, ...)` function |
| Perf measurement | Inline in `demo.py` | `run_perf_benchmark(executor, ...)` function |
| vLLM interface | Generator IS the runtime | `Llama3Generator` wraps executor |
| Vision paths | 6+ methods, ~530 lines | None |
| DP chunking | Throughout | None |

### What a new model author implements

For model #12 (e.g., Qwen, DeepSeek):
1. `NewModelTransformer` — the model itself (with `model_args` attribute)
2. `NewModelTransformerConfig` — with `from_hf_config`
3. `EagerNewModelExecutor` — thin wrapper passing model to `EagerLLMExecutor`
4. `TracedNewModelExecutor` — thin wrapper passing model to `TracedLLMExecutor`

What they reuse without change:
- `EagerLLMExecutor` / `TracedLLMExecutor` — shared engines own all execution logic
- `run_teacher_forcing()` — generic loop policy
- `run_perf_benchmark()` — generic loop policy
- `NewModelGenerator` (vLLM adapter) — trivial, ~20 lines

---

## File 3: `demo.py` — Teacher Forcing + Performance

### Design

A standalone pytest script that uses executors directly — no vLLM adapter needed.

### Test configurations

```python
@pytest.mark.parametrize("test_config", [
    "token-accuracy",  # Teacher forcing, top1/top5 vs reference
    "batch-1",         # Single-user latency (tok/s/u, TTFT)
    "batch-32",        # Multi-user throughput (tok/s/u, TTFT)
])
@pytest.mark.parametrize("optimizations", ["performance", "accuracy"])
```

### Flow

```
1. Load HF model name from env (HF_MODEL=meta-llama/Llama-3.1-8B-Instruct)
2. Create mesh device
3. Build Llama3Transformer1DConfig.from_hf_config(...)
4. Build Llama3Transformer1D.from_config(config)

For token-accuracy (Eager — debugging):
  5a. executor = EagerLlamaExecutor(model, mesh_device)
  6a. kv_cache = executor.allocate_kv_cache(...)
  7a. executor.compile(prefill_tokens=prompt, prefill_page_table=page_table)
  8a. result = run_teacher_forcing(executor, prompt_tokens=prompt, reference_tokens=reference, top5_tokens=top5, kv_cache=kv_cache)
  9a. Assert result.top1_accuracy() >= threshold
  10a. Assert result.top5_accuracy() >= threshold

For token-accuracy (Traced — CI gate):
  5b. traced = TracedLlamaExecutor(model, mesh_device)
  6b. kv_cache = traced.allocate_kv_cache(...)
  7b. traced.compile(prefill_tokens=prompt, prefill_page_table=page_table)
  8b. result = run_teacher_forcing(traced, prompt_tokens=prompt, reference_tokens=reference, top5_tokens=top5, kv_cache=kv_cache)
  9b. Assert result.top1_accuracy() >= threshold

For batch-1 / batch-32 (perf):
  5c. traced = TracedLlamaExecutor(model, mesh_device)
  6c. kv_cache = traced.allocate_kv_cache(...)
  7c. traced.compile(prefill_tokens=prompt, prefill_page_table=page_table)
  8c. result = run_perf_benchmark(traced, tokens=prompt, kv_cache=kv_cache, num_decode_tokens=128)
  9c. Assert result.meets_target(expected_metrics)
```

### Accuracy measurement

`run_teacher_forcing()` owns the decode loop with ground-truth injection and returns a result dataclass. No stateful callback needed.

```python
@dataclass
class TeacherForceResult:
    predicted_tokens: list[int]
    reference_top5: torch.Tensor   # shape [num_tokens, 5]

    def top1_accuracy(self) -> float:
        matches = sum(
            1 for i, p in enumerate(self.predicted_tokens)
            if self.reference_top5[i, 0].item() == p
        )
        return matches / len(self.predicted_tokens)

    def top5_accuracy(self) -> float:
        matches = sum(
            1 for i, p in enumerate(self.predicted_tokens)
            if p in self.reference_top5[i, :]
        )
        return matches / len(self.predicted_tokens)
```

Data flow is unidirectional: function orchestrates executor calls → result produced → metrics computed. The function handles ground-truth injection; the caller only sees the final result.

### Performance measurement

`run_perf_benchmark()` owns the timed prefill + decode loop and returns a result dataclass. No inline profiler code in `demo.py`.

```python
@dataclass
class PerfBenchmarkResult:
    # Raw timings
    prefill_time_s: float              # wall time for prefill
    compile_decode_time_s: float       # first decode iteration (compile)
    decode_times_s: list[float]        # per-iteration decode times (excluding compile)
    batch_size: int
    num_decode_tokens: int

    # Derived metrics
    @property
    def ttft_ms(self) -> float:
        return self.prefill_time_s * 1000

    @property
    def tok_s_u(self) -> float:
        """Tokens per second per user (steady-state decode)."""
        return len(self.decode_times_s) / sum(self.decode_times_s)

    @property
    def tok_s(self) -> float:
        """Total throughput."""
        return self.tok_s_u * self.batch_size

    @property
    def decode_latency_mean_ms(self) -> float:
        return (sum(self.decode_times_s) / len(self.decode_times_s)) * 1000

    def meets_target(self, expected: dict, tolerance: float = 0.05) -> dict[str, bool]:
        """Check against expected metrics. Returns {metric: passed}."""
        return {
            "tok_s_u": self.tok_s_u >= expected["tok_s_u"] * (1 - tolerance),
            "ttft_ms": self.ttft_ms <= expected["ttft_ms"] * (1 + tolerance),
        }
```

### Expected metrics (from TTTv1 PERF.md + baselines)

| Config | Device | top1 | top5 | tok/s/u | TTFT (ms) |
|--------|--------|------|------|---------|-----------|
| performance | N150 | ≥90% | ≥97% | ≥28 | ≤110 |
| performance | N300 | ≥90% | ≥97% | ≥44 | ≤70 |
| performance | T3K  | ≥90% | ≥98% | ≥64 | ≤55 |
| accuracy | N150 | ≥96% | ≥100% | ≥25 | ≤140 |
| accuracy | N300 | ≥96% | ≥100% | ≥38 | ≤80 |

### Key detail: Traced teacher forcing now works

With the two-level architecture, `run_teacher_forcing()` works with both `EagerLlamaExecutor` and `TracedLlamaExecutor`. The traced executor replays traces during the decode loop while the function feeds ground-truth tokens. This enables:
- **Eager teacher forcing** — developer debugging (isolate model bugs from trace bugs)
- **Traced teacher forcing** — CI accuracy gate (verify traced execution matches eager)

---

## Implementation Order

### Phase 1: `model.py` — The Transformer
1. Implement `TransformerBlock1DConfig` + `TransformerBlock1D`
   - Compose: `RMSNorm1D` (attn_norm) → `Attention1D` → residual add → `RMSNorm1D` (ff_norm) → `MLP1D` → residual add
   - `decode_forward` and `prefill_forward` — zero static branching
2. Implement `Llama3Transformer1DConfig` + `Llama3Transformer1D`
   - Assemble: `Embedding1D` → `RotarySetup1D` → layers → `RMSNorm1D` → `LMHead1D` → `Sampling1D`
   - Implement input prep methods (`prepare_inputs_prefill`, `prepare_decode_inputs_host`, etc.)
   - Implement output processing (`process_output_prefill`, `process_output_decode`)
   - Implement device-level forward (`ttnn_prefill_forward`, `ttnn_decode_forward`)
3. Implement `Llama3Transformer1DConfig.from_hf_config(...)` — direct config construction from HF config + runtime args
4. Unit test: Config creation, module assembly (no device)

### Phase 2: Shared Executor Engines
1. Implement `EagerLLMExecutor` in `models/common/models/executor.py`:
   - `prefill_forward()`, `decode_forward()` — owns the implementation
   - `allocate_kv_cache()`, `compile_prefill()`, `compile_decode()`, `compile()`
   - Port chunked prefill + prefix caching logic from TTTv1
   - Port input preparation and output processing
2. Implement `TracedLLMExecutor` in `models/common/models/executor.py`:
   - Same public API as `EagerLLMExecutor`
   - Add trace capture in compile methods, replay in forward methods
   - Implement `warmup_model_prefill()` — sweep supported seq lens
   - Implement `cleanup()` — release all traces
3. Integration test: `test_executor_parity.py` — verify Eager == Traced outputs

### Phase 3: Thin Model Executors
1. Implement `EagerLlamaExecutor` in `models/common/models/llama3_8b/model.py`:
   - Pure delegation to `EagerLLMExecutor`
2. Implement `TracedLlamaExecutor` in `models/common/models/llama3_8b/model.py`:
   - Pure delegation to `TracedLLMExecutor`
3. Integration test: Instantiate on device, run single prefill + decode

### Phase 4: Loop Policy Functions + Generator
1. Implement `run_teacher_forcing()` in `models/common/models/executor.py`
2. Implement `run_perf_benchmark()` in `models/common/models/executor.py`
3. Implement `Llama3Generator(executor)` — vLLM adapter shim
4. Implement `Llama3Generator.initialize_vllm_model(...)` — config → model → traced executor → generator

### Phase 5: `demo.py` — Accuracy & Performance
1. Implement `token-accuracy` test using `TeacherForceExecutor`
2. Implement `batch-1` and `batch-32` tests using `PerfBenchmarkExecutor`
3. Validate against expected metrics

### Phase 6: Audit
1. Verify no TTTv1 imports in `model.py` forward path
2. Verify no `torch` imports in `model.py` (except via LazyWeight)
3. Verify no static branching in any forward method
4. Compare `EagerLLMExecutor.prefill_forward`/`decode_forward` against TTTv1 for correctness
5. Compare `TracedLLMExecutor` trace lifecycle against TTTv1 for correctness
6. Check for un-deallocated temporary tensors
7. Verify `Llama3Generator` exposes all methods `TTModelRunner` expects
8. Verify model executors are pure delegation (no logic duplication)
9. Verify that there is no deallocate after async ops
10. Verify that there is no CCL calls at the beginning or the end of the forward functions of any module or model; the goal is to make such CCLs to be out in the "open".
11. Run `test_executor_parity.py` to verify Eager == Traced outputs

---

## Risk Analysis

| Risk | Mitigation |
|------|------------|
| vLLM adapter misses a contract method | Audit every call site in `tt_model_runner.py`; integration test with actual vLLM |
| TTTv2 module configs might not match TTTv1 defaults | Build `Llama3Transformer1DConfig` from HF config and verify each derived field against TTTv1's `ModelArgs` during bring-up |
| Per-layer precision settings from `decoders_optimizations` | Resolve once while building config, store on block/attention configs |
| Traced executor diverges from TTTv1 trace behavior | Port trace code from TTTv1; same TTNN trace API, same capture/replay lifecycle |
| Chunked prefill edge cases | Port chunking logic 1:1 from TTTv1; test with long prompts and cached prefixes |
| Engine and model executor drift apart | Model executors are pure delegation — no logic to drift. Engine owns all execution logic. |
| Eager/Traced output mismatch | `test_executor_parity.py` validates "same inputs, same outputs, regardless of executor" |
| Missing `DistributedNorm` wrapper | `RMSNorm1D` already supports distributed configs for N300/T3K. Verify with multi-device tests |

---

## Non-goals (for this PR)

- 2D / TG support (4x8, 8x4) — future work, needs `Attention2D`, `MLP2D`, `RMSNorm2D`
- Vision / multimodal — Llama 3.2 11B/90B, separate model
- MoE — DeepSeek / Mixtral, separate model
- Prefetcher support — TTTv1 feature, can be added later
- Data parallel — TTTv1 `tt_data_parallel > 1`, can be added later
- `generate()` / `chat_completion()` / `text_completion()` — vLLM handles these; not part of the Generator's responsibility
<!-- END VERBATIM: models/common/models/llama3_8b/PLAN.md -->

<a id="source-13-models-common-models-llama3-8b-static-kv-cache-plan-md"></a>

### Source 13: `models/common/models/llama3_8b/STATIC_KV_CACHE_PLAN.md`

[Back to Source Index](#source-index) | [Previous: `models/common/models/llama3_8b/PLAN.md`](#source-12-models-common-models-llama3-8b-plan-md) | [Next: `models/common/models/llama3_8b/Sampling Notes.md`](#source-14-models-common-models-llama3-8b-sampling-notes-md)

<!-- BEGIN VERBATIM: models/common/models/llama3_8b/STATIC_KV_CACHE_PLAN.md -->
# Static KV Cache Refactor (DONE)

## Problem

`kv_cache` is a static pool — allocated once, never reassigned between calls. The `page_table`
handles all per-request indirection. Yet the current code re-sets `layer.attention.kv_cache`
on every forward call, creating two issues:

1. **Redundant per-forward work**: Three sites re-assign the same kv_cache every call:
   - `Llama3Transformer1D.decode_forward` (model.py:280-281)
   - `Llama3Transformer1D.prefill_forward` (model.py:305-307)
   - `TransformerBlock1D.forward` (model.py:167-168)

2. **`load_device_weights` ordering fragility**: `load_device_weights()` is called lazily
   on the first forward. If someone sets `attention.kv_cache` directly (bypassing config)
   before that first forward, `load_device_weights()` can clobber it. This required an
   `elif not hasattr` guard (attention_1d.py:969) — a workaround for bypassing the config.

## Goal

Make kv_cache flow through `Attention1DConfig.kv_cache` (attention_1d.py:154) like all
other weights. `load_device_weights()` already knows how to resolve it. No direct
`attention.kv_cache` assignment outside of `load_device_weights()`.

This eliminates:
- Per-forward kv_cache plumbing through model forward signatures
- Direct `layer.attention.kv_cache = ...` assignments
- The `elif not hasattr` guard (reverted to plain `else`)

vLLM reference (tenstorrent/vllm `dev` branch, `tt_model_runner.py`):
- Line 286: `self.kv_caches = self.model.allocate_kv_cache(...)` — allocated once
- Line 1395: `"kv_cache": self.kv_caches` — passed verbatim, never reassigned
- `page_table` (via `model_input.block_tables`) does all per-request indirection

## Changes

### 1. `set_kv_cache()` on `Llama3Transformer1D` — set via config (model.py)

Instead of directly setting `layer.attention.kv_cache`, set
`layer.attention.config.kv_cache`. This is the `Attention1DConfig.kv_cache` field
(attention_1d.py:154) that `load_device_weights()` already resolves:

```python
def set_kv_cache(self, kv_cache: list):
    """Bind static kv_cache pool to each attention layer's config.

    Must be called before the first forward (before load_device_weights runs).
    The kv_cache is resolved from config during load_device_weights(), just
    like all other weights.
    """
    assert len(kv_cache) == len(self.layers)
    for i, layer in enumerate(self.layers):
        layer.attention.config.kv_cache = tuple(kv_cache[i])
```

### 2. Call `set_kv_cache()` from executor after allocation (executor.py)

In `EagerLlamaExecutor.allocate_kv_cache`, after building the cache list:

```python
def allocate_kv_cache(self, kv_cache_shape, dtype, num_layers):
    # ... existing allocation code ...
    self._kv_cache = kv_cache
    self.model.set_kv_cache(kv_cache)
    return kv_cache
```

`TracedLlamaExecutor.allocate_kv_cache` delegates to `self._direct.allocate_kv_cache`,
so it inherits this automatically.

### 3. Remove per-forward kv_cache assignment (model.py)

**`TransformerBlock1D.forward`** — remove kv_cache param and assignment:
```python
# BEFORE
def forward(self, x, ..., kv_cache=None):
    if kv_cache is not None:
        self.attention.kv_cache = tuple(kv_cache)

# AFTER
def forward(self, x, ...):
    ...
```

**`Llama3Transformer1D.decode_forward`** — remove kv_cache param and loop assignment:
```python
# BEFORE
def decode_forward(self, x_embed, current_pos, rot_mats, page_table=None, kv_cache=None):
    for i, layer in enumerate(self.layers):
        if kv_cache is not None:
            layer.attention.kv_cache = tuple(kv_cache[i])

# AFTER
def decode_forward(self, x_embed, current_pos, rot_mats, page_table=None):
    for i, layer in enumerate(self.layers):
        ...
```

**`Llama3Transformer1D.prefill_forward`** — same removal.

### 4. Remove kv_cache from executor→model call sites (executor.py)

Drop `kv_cache=kv_cache` from all executor calls into model forward methods.

Affected call sites (all in executor.py):
- `EagerLlamaExecutor._prefill_single_user` → `model.prefill_forward`
- `EagerLlamaExecutor.decode_forward` → `model.decode_forward`
- `TracedLlamaExecutor._capture_and_run_prefill_trace` → `model.prefill_forward`
- `TracedLlamaExecutor._capture_decode_trace` → `model.decode_forward`

### 4a. vLLM compatibility: executor-level `kv_cache` param + identity assertion

**The problem**: vLLM's `TTModelRunner` will continue to pass `kv_cache` on every
`prefill_forward` / `decode_forward` call through the generator:

```
vLLM TTModelRunner
  → generator.prefill_forward(tokens, kv_cache=self.kv_caches, ...)   # *args/**kwargs
    → executor.prefill_forward(tokens, kv_cache=self.kv_caches, ...)
```

After the refactor, the model doesn't use this arg — the executor only uses it for page
table math (`get_block_size(kv_cache)`). If someone passes a *different* kv_cache than
what's bound to the model layers (e.g. after a reallocation that forgot to call
`allocate_kv_cache` again), we'd get silent correctness bugs.

**The solution**: Store the bound kv_cache reference on the executor. Assert identity
on every forward call.

```python
# In EagerLlamaExecutor:
def _assert_kv_cache_identity(self, kv_cache):
    if kv_cache is not None and hasattr(self, "_kv_cache"):
        assert kv_cache is self._kv_cache, (
            "kv_cache passed to forward differs from the allocated cache. "
            "Call allocate_kv_cache() again after reallocating."
        )
```

**Design discussion** — the right group to weigh in:

| Role | Perspective |
|------|-------------|
| **vLLM integration eng** | "Don't break the API. vLLM passes kv_cache on every call. Keep accepting it, but assert identity so drift is caught immediately." |
| **Framework/API designer** | "The executor should *own* the kv_cache reference. Forward methods accept it for backward compat but treat it as a consistency check, not a state setter." |
| **Safety/correctness eng** | "The identity assertion catches reallocation without re-binding. `set_kv_cache` should be idempotent for clean reallocation." |
| **Performance eng** | "Long-term, extract `block_size` at allocation time and drop kv_cache from forward entirely. But that's a follow-up." |

**Consensus**: Keep `kv_cache` in executor-level forward signatures for backward compat.
Assert identity. Generator uses `*args, **kwargs` passthrough — no change needed.

### 5. Revert `elif not hasattr` → plain `else` + assert no dynamic kv_cache (attention_1d.py)

With kv_cache set via **config** (not direct attribute assignment), `load_device_weights()`
resolves it naturally:

```python
# load_device_weights():
if cfg.kv_cache is not None:       # ← set_kv_cache put it here
    keys, values = cfg.kv_cache
    ...
    self.kv_cache = (keys, values)  # resolved from config
else:
    assert not hasattr(self, "kv_cache") or self.kv_cache is None, (
        "kv_cache was set directly on Attention1D instead of via config.kv_cache. "
        "Use model.set_kv_cache() to bind kv_cache through the config."
    )
    self.kv_cache = None            # no cache configured — correct
```

The `elif not hasattr` guard was needed because `set_kv_cache()` was setting
`attention.kv_cache` directly (bypassing config), and `load_device_weights()`
would clobber it with `None`. Now that `set_kv_cache()` writes to `config.kv_cache`,
`load_device_weights()` sees it in the `if` branch and resolves it correctly.

The assert in the `else` branch is a safety net: if someone regresses to the old
pattern of `layer.attention.kv_cache = tuple(...)` directly (bypassing config),
this catches it immediately — `cfg.kv_cache` would be `None` but `self.kv_cache`
would already be set, which is always a bug.

Plain `else` is safe because the only two states are:
- `cfg.kv_cache is not None`: set by `set_kv_cache()` before first forward → resolved
- `cfg.kv_cache is None`: no paged kv_cache → `self.kv_cache = None` is correct

## Execution Order

```
1. Model construction         → Attention1D created, config.kv_cache = None
2. allocate_kv_cache()         → model.set_kv_cache() sets config.kv_cache on each layer
3. First forward               → load_device_weights() resolves config.kv_cache → self.kv_cache
4. All subsequent forwards     → kv_cache already resolved, no per-call work
```

**Invariant**: `allocate_kv_cache()` must be called before the first forward.
This is already the case in both demo tests and vLLM. The identity assertion at
the executor level catches any violation.

## Files Changed

| File | Changes |
|------|---------|
| `models/common/models/llama3_8b/model.py` | `set_kv_cache()` writes to `config.kv_cache`; remove `kv_cache` param from `forward`, `decode_forward`, `prefill_forward` |
| `models/common/models/llama3_8b/executor.py` | Store `self._kv_cache` + call `model.set_kv_cache()` in `allocate_kv_cache`; add identity assertion; drop `kv_cache=` from model-level calls |
| `models/common/modules/attention/attention_1d.py` | Revert `elif not hasattr` → `else` |
| `models/common/models/llama3_8b/generator.py` | **No change** — `*args, **kwargs` passthrough. vLLM keeps passing `kv_cache`, generator relays to executor, executor asserts identity. |

## Testing

Run all N150 tests to verify no regression:

```bash
MESH_DEVICE=N150 HF_MODEL=meta-llama/Llama-3.1-8B-Instruct \
python_env/bin/pytest models/common/models/llama3_8b/demo.py -v --timeout=600 --tb=short
```
<!-- END VERBATIM: models/common/models/llama3_8b/STATIC_KV_CACHE_PLAN.md -->

<a id="source-14-models-common-models-llama3-8b-sampling-notes-md"></a>

### Source 14: `models/common/models/llama3_8b/Sampling Notes.md`

[Back to Source Index](#source-index) | [Previous: `models/common/models/llama3_8b/STATIC_KV_CACHE_PLAN.md`](#source-13-models-common-models-llama3-8b-static-kv-cache-plan-md) | [Next: `models/common/models/llama3_8b/TODO.md`](#source-15-models-common-models-llama3-8b-todo-md)

<!-- BEGIN VERBATIM: models/common/models/llama3_8b/Sampling Notes.md -->
The pipeline:
```
User request (vLLM SamplingParams)
    │
    ▼
tt_model_runner._prepare_model_inputs()    ← converts to TTSamplingParams
    │
    ├── decides: device sampling or host sampling?
    │
    ▼
model.decode_forward(tokens, ..., sampling_params, prompt_tokens, output_tokens)
    │                                                     │
    │  (inside tt-metal model)                            │
    ▼                                                     ▼
SamplingGenerator.reset_sampling_params()    TTPenalties.reset_prompt_tokens()
SamplingGenerator.reset_output_state()       TTPenalties.reset_output_tokens()
    │
    ▼
logits = transformer(tokens)
    │
    ▼
SamplingGenerator.sample(logits)
    ├── TTPenalties.apply(logits)          ← presence/frequency/repetition
    ├── TTSampling(logits)                 ← top-k, top-p, temperature, multinomial
    └── TTPenalties.update_output_tokens() ← track newly sampled token
    │
    ▼
sampled token IDs returned to vLLM
    │
    ▼
tt_model_runner → ModelRunnerOutput → engine

```

---

# new_plan.md
# Plan: Split Sampling into Penalties1D + Sampling1D

## Context

TTTv1's `SamplingGenerator` bundles penalties and sampling into one orchestrator. After design review, we split into **two independent TTTv2 modules**: `Penalties1D` (logit transform) and `Sampling1D` (token selection). They have zero direct coupling — penalties produce logits, sampling consumes logits. This enables independent composition, testing, and reuse per TTTv2 principles.

## Files to Create

```
models/common/modules/
├── lazy_buffer.py        # LazyBuffer — lazy device buffer allocation (mutable sibling of LazyWeight)
models/common/modules/sampling/
├── penalties_1d.py       # Penalties1D — presence/frequency/repetition penalty transforms
├── sampling_1d.py        # Sampling1D — top-k/top-p/temperature + argmax fast path
models/common/tests/modules/sampling/
├── test_penalties_1d.py  # Penalty tests (penalty math, lifecycle, composition)
├── test_sampling_1d.py   # Sampling tests (top-k correctness, argmax, multi-device)
```

No `__init__.py` or `conftest.py` (project convention).

## Reference Files

| Purpose | Path |
|---------|------|
| TTTv2 module pattern | `models/common/modules/mlp/mlp_1d.py` |
| TTTv2 test pattern | `models/common/tests/modules/mlp/test_mlp_1d.py` |
| TTTv1 penalties source | `models/common/sampling/tt_penalties.py` |
| TTTv1 sampling source | `models/common/sampling/tt_sampling.py` |
| TTTv1 orchestrator | `models/common/sampling/generator.py` |
| Base class | `models/common/lightweightmodule.py` |
| LazyWeight (sibling pattern) | `models/common/modules/lazy_weight.py` |
| CCL utilities | `models/common/modules/tt_ccl.py` |
| LogProbsCalculator | `models/common/utils.py` |
| Test fixture | `models/common/tests/conftest.py` (`ttnn_mesh_device`) |

---

## Step 0: Create `lazy_buffer.py`

### Design rationale

TTTv1's `TTPenalties.__init__` (lines 104-115) allocates persistent device buffers via `_alloc_int_buffer()` / `_alloc_bf16_buffer()`, which call `ttnn.from_torch()` with a host source tensor (typically `torch.zeros(...)` or `torch.ones(...)`), dtype, layout, mesh_mapper, and memory_config. This is **exactly the same call pattern as `LazyWeight.get_device_weight()`**. The only thing that makes these buffers different from weights is **post-allocation mutability** (the device data is overwritten in-place via `output_tensor=` during decode) and **no disk caching** (caching a mutable buffer would corrupt state across instances).

`LazyBuffer` captures this by reusing LazyWeight's allocation contract — same fields, same `from_torch()` path — while explicitly omitting the caching/fingerprinting layer. This makes buffer allocation specs declarative, inspectable, and overridable by power users via the config dataclass, following the same "None means auto-resolve" pattern as MLP1DConfig's LazyWeight fields.

### LazyBuffer dataclass

```python
@dataclass
class LazyBuffer:
    """
    Lazy-allocated device buffer for mutable state tensors.

    Mirrors LazyWeight's allocation contract (source + from_torch() parameters) but is
    designed for buffers that are mutated in-place after allocation, NOT immutable model weights.

    Key differences from LazyWeight:
    - No disk caching: The device data is overwritten in-place via output_tensor= during
      decode (e.g., penalty masks, token counts). Caching a mutable buffer would cause
      state corruption if loaded by another instance.
    - No fingerprinting: Without caching, there is no cache to invalidate.
    - _value caching is safe: The ttnn.Tensor *handle* returned by get_device_buffer()
      never changes — only the on-device data changes via output_tensor= writes.
      So "allocate once, return same handle" is correct for mutable buffers.

    The only thing that makes these different from weights is post-allocation mutability
    and no disk caching. If a buffer becomes read-only in a future refactor, it can be
    promoted to a LazyWeight with caching enabled.

    See also: LazyWeight in models/common/modules/lazy_weight.py
    """

    # Source: initial host tensor values (e.g., torch.zeros, torch.ones).
    # Duck-typed — string annotation avoids torch import at module level.
    source: "torch.Tensor"

    # from_torch() parameters — same fields as LazyWeight (minus cache_dir_weight_name, pad_value)
    dtype: ttnn.DataType | None = ttnn.int32
    layout: ttnn.Layout | None = ttnn.TILE_LAYOUT
    device: ttnn.MeshDevice | None = None
    mesh_mapper_config: ttnn.MeshMapperConfig | None = None
    memory_config: ttnn.MemoryConfig | None = None

    # Cached device tensor handle (allocated once, device data mutated in-place)
    _value: ttnn.Tensor | None = field(default=None, repr=False)

    def _build_mesh_mapper(self):
        """Build mesh mapper from config. Shared by get_device_buffer() and update()."""
        if self.mesh_mapper_config is not None:
            return ttnn.create_mesh_mapper(self.device, self.mesh_mapper_config)
        return ttnn.replicate_tensor_to_mesh_mapper(self.device)

    def _from_torch_args(self, source, *, device):
        """
        Build the full from_torch() kwargs. Used by both get_device_buffer() and update()
        to ensure the same dtype/layout/mesh_mapper/memory_config are used consistently.
        Only `device` differs: real device for allocation, None for host-side update.
        """
        return dict(
            dtype=self.dtype,
            layout=self.layout,
            device=device,
            mesh_mapper=self._build_mesh_mapper(),
            memory_config=self.memory_config,
        )

    def get_device_buffer(self) -> ttnn.Tensor:
        """Allocate on first call, return cached handle thereafter."""
        if self._value is not None:
            return self._value

        if self.device is None:
            raise ValueError("device must be set before materializing buffer")
        if self.layout is None:
            raise ValueError("layout must be set before materializing buffer")

        self._value = ttnn.from_torch(
            self.source, **self._from_torch_args(self.source, device=self.device),
        )
        return self._value

    def update(self, new_source: "torch.Tensor") -> None:
        """
        Overwrite the device buffer contents with a new source tensor, without reallocating.

        If the buffer has not yet been materialized (get_device_buffer not called), this
        simply replaces self.source for future materialization.

        If the buffer IS already materialized, this performs an in-place device update
        using the SAME from_torch() args as the original allocation (dtype, layout,
        mesh_mapper, memory_config) but with device=None to create a host tensor:
            host_tt = ttnn.from_torch(new_source, **same_args, device=None)
            ttnn.copy_host_to_device_tensor(host_tt, self._value)

        The ttnn.Tensor handle (self._value) is preserved — no DRAM reallocation.

        This encapsulates the pattern seen in:
        - TTPenalties._copy_host_to_device (tt_penalties.py:157-159)
        - SeedManager.get_new_values (generator.py:382-383)
        """
        self.source = new_source
        if self._value is not None:
            host_tt = ttnn.from_torch(
                new_source, **self._from_torch_args(new_source, device=None),
            )
            ttnn.copy_host_to_device_tensor(host_tt, self._value)

    def is_resolved(self) -> bool:
        """Check if all required fields for materialization are set."""
        return self.device is not None and self.dtype is not None and self.layout is not None
```

### resolve_lazy_buffer() helper

Mirrors `resolve_lazy_weight()` from `lazy_weight.py:346-349`:

```python
def resolve_lazy_buffer(buf: LazyBuffer, **kwargs) -> LazyBuffer:
    """Resolve None fields of `buf` with the given kwargs; do not override non-None fields."""
    to_set = {k: v for k, v in kwargs.items() if getattr(buf, k, None) is None}
    return replace(buf, **to_set)
```

### Unit tests for LazyBuffer

Add to `models/common/tests/modules/sampling/test_penalties_1d.py` (or standalone `test_lazy_buffer.py`):

1. `test_lazy_buffer_defaults` — dtype=int32, layout=TILE, device/mesh_mapper_config/memory_config=None
2. `test_lazy_buffer_is_resolved` — True only when device+dtype+layout are set
3. `test_lazy_buffer_get_device_buffer_allocates` — first call returns ttnn.Tensor, second call returns same handle
4. `test_lazy_buffer_raises_without_device` — ValueError if device is None
5. `test_resolve_lazy_buffer` — fills None fields, preserves non-None
6. `test_lazy_buffer_update_before_materialize` — update() replaces source, get_device_buffer() uses new source
7. `test_lazy_buffer_update_after_materialize` — update() refreshes device data, handle identity unchanged (same `id()`), readback matches new source
8. `test_lazy_buffer_update_preserves_handle` — after update(), the tensor returned by get_device_buffer() is the same Python object as before

---

## Step 1: Create `penalties_1d.py`

### Penalties1DConfig dataclass

```python
@dataclass
class Penalties1DConfig:
    vocab_size: int                                    # Required. Caller pre-pads to be divisible by num_devices.
    mesh_device: ttnn.MeshDevice | None = None         # None → GetDefaultDevice()
    max_batch_size: int = 32                           # TTTv1 hardcode (tt_penalties.py:87)
    sub_core_grids: ttnn.CoreRangeSet | None = None    # From args.sub_core_grids (line 94)

    # --- Persistent buffer specs (LazyBuffer | ttnn.Tensor | None) ---
    # None = auto-filled by _resolve_penalties1d_config() with topology-aware defaults.
    # LazyBuffer = declarative spec, materialized lazily in load_device_buffers().
    # ttnn.Tensor = pre-allocated device tensor, used directly (power user bypass).
    #
    # Sharded vocab buffers: [max_batch_size, vocab_size], int32, TILE, sharded across devices
    prompt_mask: LazyBuffer | ttnn.Tensor | None = None
    output_mask: LazyBuffer | ttnn.Tensor | None = None
    output_counts: LazyBuffer | ttnn.Tensor | None = None
    # Replicated vocab buffers: [max_batch_size, vocab_size], int32, replicated
    output_counts_gathered: LazyBuffer | ttnn.Tensor | None = None
    zeros: LazyBuffer | ttnn.Tensor | None = None
    # Utility buffers
    decode_src: LazyBuffer | ttnn.Tensor | None = None  # [max_batch_size, 1], int32, ROW_MAJOR, ones
    # BF16 penalty param buffers: [max_batch_size, 1], bfloat16, TILE, replicated
    presence_penalties: LazyBuffer | ttnn.Tensor | None = None
    frequency_penalties: LazyBuffer | ttnn.Tensor | None = None
    repetition_penalties: LazyBuffer | ttnn.Tensor | None = None
    inverse_repetition_penalties: LazyBuffer | ttnn.Tensor | None = None

    @staticmethod
    def _buf_resolved(buf) -> bool:
        """A buffer field is resolved if it's a ttnn.Tensor (already on device) or a resolved LazyBuffer."""
        if buf is None:
            return False
        if isinstance(buf, ttnn.Tensor):
            return True  # already materialized — power user path
        return buf.is_resolved()  # LazyBuffer path

    def is_resolved(self) -> bool:
        return (
            self.mesh_device is not None
            and all(self._buf_resolved(getattr(self, f)) for f in (
                "prompt_mask", "output_mask", "output_counts",
                "output_counts_gathered", "zeros", "decode_src",
                "presence_penalties", "frequency_penalties",
                "repetition_penalties", "inverse_repetition_penalties",
            ))
        )
```

NOT config fields (derived internally): `cluster_shape` (from `mesh_device.shape`), `num_devices` (from `max(cluster_shape)`), `shard_dims`/`shard_dims_slice` (from cluster shape comparison at lines 98-103).

### _resolve_penalties1d_config() function

Mirrors `_resolve_mlp1d_config()` pattern: fills None fields with topology-aware defaults, returns `replace(config, **to_set)`, asserts `is_resolved()`.

```python
def _resolve_penalties1d_config(config: Penalties1DConfig) -> Penalties1DConfig:
    """
    Fill None fields in config with topology-aware defaults.
    Power users who set fields explicitly will NOT have them overwritten.
    """
    import torch  # lazy import — only needed for source tensor construction

    to_set: dict = {}

    # Phase 1: Device
    mesh_device = config.mesh_device or ttnn.GetDefaultDevice()
    to_set["mesh_device"] = mesh_device

    # Phase 2: Topology → shard_dims (port from tt_penalties.py:97-103)
    cluster_shape = mesh_device.shape
    num_devices = max(cluster_shape[-1], cluster_shape[-2])
    if cluster_shape[-1] == num_devices:
        shard_dims = (None, 1)     # shard vocab across columns
        shard_dims_slice = (None, 0)
    else:
        shard_dims = (1, None)     # shard vocab across rows
        shard_dims_slice = (0, None)

    B = config.max_batch_size
    V = config.vocab_size

    # Build mesh mapper configs
    shard_mapper = ttnn.MeshMapperConfig(shard_dims=shard_dims, mesh_shape=cluster_shape)
    replicate_mapper = None  # None → replicate_tensor_to_mesh_mapper in LazyBuffer

    # Helper: resolve a single buffer field.
    # - None → create LazyBuffer with defaults
    # - LazyBuffer → fill None fields with defaults via resolve_lazy_buffer()
    # - ttnn.Tensor → pass through unchanged (power user pre-allocated)
    def _resolve_buf(field_val, defaults, source_factory):
        if field_val is None:
            return LazyBuffer(source=source_factory(), **defaults)
        if isinstance(field_val, ttnn.Tensor):
            return field_val  # already on device — no resolution needed
        return resolve_lazy_buffer(field_val, **defaults)

    # Phase 3: Sharded vocab buffers — [B, V], int32, TILE, sharded
    sharded_vocab_defaults = dict(
        dtype=ttnn.int32, layout=ttnn.TILE_LAYOUT, device=mesh_device,
        mesh_mapper_config=shard_mapper, memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    zeros_BV = lambda: torch.zeros(B, V, dtype=torch.int32)
    to_set["prompt_mask"] = _resolve_buf(config.prompt_mask, sharded_vocab_defaults, zeros_BV)
    to_set["output_mask"] = _resolve_buf(config.output_mask, sharded_vocab_defaults, zeros_BV)
    to_set["output_counts"] = _resolve_buf(config.output_counts, sharded_vocab_defaults, zeros_BV)

    # Phase 4: Replicated vocab buffers — [B, V], int32, replicated
    replicated_vocab_defaults = dict(
        dtype=ttnn.int32, layout=ttnn.TILE_LAYOUT, device=mesh_device,
        mesh_mapper_config=replicate_mapper, memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    to_set["output_counts_gathered"] = _resolve_buf(
        config.output_counts_gathered, replicated_vocab_defaults, zeros_BV,
    )

    replicated_vocab_rm_defaults = dict(
        dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh_device,
        mesh_mapper_config=replicate_mapper, memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    to_set["zeros"] = _resolve_buf(config.zeros, replicated_vocab_rm_defaults, zeros_BV)

    # Phase 5: Utility buffers
    decode_src_defaults = dict(
        dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh_device,
        mesh_mapper_config=replicate_mapper, memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    to_set["decode_src"] = _resolve_buf(
        config.decode_src, decode_src_defaults, lambda: torch.ones(B, 1, dtype=torch.int32),
    )

    # Phase 6: BF16 penalty param buffers — [B, 1], bfloat16, TILE, replicated
    bf16_param_defaults = dict(
        dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device,
        mesh_mapper_config=replicate_mapper, memory_config=None,  # bf16 params use default (no explicit DRAM)
    )
    zeros_B1 = lambda: torch.zeros(B, 1, dtype=torch.float32)
    for field_name in ("presence_penalties", "frequency_penalties",
                       "repetition_penalties", "inverse_repetition_penalties"):
        to_set[field_name] = _resolve_buf(getattr(config, field_name), bf16_param_defaults, zeros_B1)

    resolved = replace(config, **to_set)
    assert resolved.is_resolved(), f"Config not fully resolved after _resolve_penalties1d_config"
    return resolved
```

**Phase summary**:

| Phase | What fills | Derived from |
|---|---|---|
| 1 | `mesh_device` | `ttnn.GetDefaultDevice()` fallback |
| 2 | `shard_dims`, `shard_dims_slice` | `mesh_device.shape` topology (port of `tt_penalties.py:97-103`) |
| 3 | `prompt_mask`, `output_mask`, `output_counts` | Sharded vocab LazyBuffers |
| 4 | `output_counts_gathered`, `zeros` | Replicated vocab LazyBuffers |
| 5 | `decode_src` | Utility buffer (ones, ROW_MAJOR) |
| 6 | `presence_penalties`, `frequency_penalties`, `repetition_penalties`, `inverse_repetition_penalties` | BF16 param LazyBuffers |

Note: `shard_dims_slice` and `slice_start`/`slice_end` tensors are NOT in config — they are derived in `load_device_buffers()` since they depend on `num_devices` and `vocab_size` (same as TTTv1 lines 117-139).

### Penalty dataclasses: PenaltyParams + PenaltyAccumulator

Ported from `tt_penalties.py:19-29`, split into two dataclasses by update frequency. Both are caller-managed — NOT owned by the module.

**PenaltyParams** — set once per request, read-only during decode loop. Caller-constructed — no module convenience methods:

```python
@dataclass
class PenaltyParams:
    prompt_mask: ttnn.Tensor                   # [max_batch_size, vocab_per_device], int32, sharded
    presence_penalties: ttnn.Tensor             # [max_batch_size, 1], bfloat16
    frequency_penalties: ttnn.Tensor            # [max_batch_size, 1], bfloat16
    repetition_penalties: ttnn.Tensor           # [max_batch_size, 1], bfloat16
    inverse_repetition_penalties: ttnn.Tensor   # [max_batch_size, 1], bfloat16 (precomputed 1/rep)
```

Construction: caller allocates tensors directly (e.g., via LazyBuffer or their own `ttnn.from_torch()`) and passes them to `PenaltyParams(...)`. The module does NOT provide a `create_penalty_params()` convenience method — all tensors are caller-owned.

**PenaltyAccumulator** — mutated every decode step. Caller-constructed:

```python
@dataclass
class PenaltyAccumulator:
    output_mask: ttnn.Tensor                   # [max_batch_size, vocab_per_device], int32, sharded
    output_counts: ttnn.Tensor                 # [max_batch_size, vocab_per_device], int32, sharded
    output_counts_gathered: ttnn.Tensor        # [max_batch_size, vocab_size], int32, replicated
```

Construction: same pattern — caller allocates and constructs directly. Reset/update are done via `decode_forward` (which applies penalties) and `update_output_tokens` (which accumulates token history).

Key change from TTTv1: `sub_core_grids` removed from both — it's a module property from config, not per-context state. No convenience creation methods — the module is a pure compute pipeline that receives external state.

### Buffer ownership boundary

All penalty buffers are persistent device tensors allocated once and mutated in-place via `output_tensor=` to avoid per-step device memory allocation:

| Update Frequency | Tensors | TTTv2 Owner | Rationale |
|---|---|---|---|
| Every decode step | `output_mask`, `output_counts`, `output_counts_gathered` | `PenaltyAccumulator` | Token history accumulation, mutated between trace executions |
| Once per request | `prompt_mask` | `PenaltyParams` | Set from prompt tokens before decode loop |
| Once per request | `*_penalties`, `inverse_repetition_penalties` | `PenaltyParams` | Set from sampling params before decode loop (precomputed `1/rep` avoids division in hot path) |
| Once at init | `decode_src`, `zeros`, `slice_start`, `slice_end` | Module (`self._*`) | Fixed for model lifetime, depends only on `vocab_size` + `mesh_device` |
| Once at init | `sub_core_grids` | Module (`self.config`) | Model-level config |

**Invariants**:
- `PenaltyParams` holds per-request constants. Set before the decode loop, read-only during it.
- `PenaltyAccumulator` holds per-step accumulators. Mutated by `update_output_tokens()` after each sampled token.
- The module holds everything fixed for the model lifetime.
- All device buffers are allocated once and never re-allocated. Lifecycle methods mutate them in-place.

### Penalties1D class

```python
class Penalties1D(LightweightModule):

    # Construction
    def __init__(self, vocab_size: int, mesh_device: ttnn.MeshDevice | None = None):
    @classmethod
    def from_config(cls, config: Penalties1DConfig) -> "Penalties1D":
    @classmethod
    def from_model_args(cls, mesh_device, args) -> "Penalties1D":  # Rejects Galaxy

    # Device buffers (idempotent, like MLP1D.load_device_weights)
    def load_device_buffers(self):
        # Materializes module-owned buffers only (decode_src, zeros, slice_start, slice_end).
        # PenaltyParams and PenaltyAccumulator tensors are caller-owned — NOT materialized here.
        # Derives: _num_devices, _cluster_shape, _shard_dims_slice, _op_kwargs
        #
        # Example:
        #   self._decode_src = self._materialize(self.config.decode_src)
        #   self._zeros = self._materialize(self.config.zeros)
        #   self._slice_start, self._slice_end = self._build_slice_tensors()

    # Forward — params and accum are caller-constructed, passed as args
    def prefill_forward(self, logits: ttnn.Tensor, params: PenaltyParams,
                        accum: PenaltyAccumulator, prompt_tokens: "torch.Tensor") -> ttnn.Tensor:
        # Sets params.prompt_mask via scatter_add from prompt_tokens (port of init_prompt_penalties).
        # Returns logits UNCHANGED — penalties are not applied during prefill.

    def decode_forward(self, logits: ttnn.Tensor, params: PenaltyParams,
                       accum: PenaltyAccumulator) -> ttnn.Tensor:
        # Applies presence/frequency/repetition penalties to logits (port of apply_penalties).
        # Returns penalized logits.

    def update_output_tokens(self, accum: PenaltyAccumulator, new_tokens: ttnn.Tensor) -> None:
        # Updates accum.output_mask and accum.output_counts via scatter_add.
        # Called AFTER sampling, not inside decode_forward (sampling happens between).
        # Port of TTPenalties.update_output_tokens (tt_penalties.py:247-264).

    def reset_output_tokens(self, accum: PenaltyAccumulator,
                            tokens: "torch.Tensor | None" = None) -> None:
        # Zeros out accum buffers. Optionally re-initializes from provided tokens.
        # Port of TTPenalties.reset_output_tokens (tt_penalties.py:214-245).

    def forward(self, logits, params=None, accum=None) -> ttnn.Tensor:  # Dispatcher

    # Private
    # NOTE: _alloc_int_buffer / _alloc_bf16_buffer → LazyBuffer.get_device_buffer()
    # NOTE: _copy_host_to_device → LazyBuffer.update()
    # NOTE: REMOVED convenience methods (caller constructs PenaltyParams/PenaltyAccumulator directly):
    #   - create_penalty_params → caller allocates tensors via LazyBuffer or ttnn.from_torch
    #   - create_penalty_accumulator → same
    #   - update_penalty_values → caller uses LazyBuffer.update() on params fields
    #   - init_prompt_penalties → absorbed into prefill_forward
    def _materialize(self, buf):  # buf if isinstance(buf, ttnn.Tensor) else buf.get_device_buffer()
    def _build_slice_tensors(self):  # Derives slice_start/slice_end from vocab_size + num_devices (port lines 117-139)
    def _pad_batch_to_max(self, tokens_2d, pad_value) -> "torch.Tensor":
    def _token_bin_counts_and_mask(self, new_tokens, src, counts=None, mask=None, counts_sliced=None):
```

### Key porting decisions

1. **Two caller-constructed dataclasses as forward args**: TTTv1 stores all penalty state as module attributes. TTTv2 splits into `PenaltyParams` (per-request constants) and `PenaltyAccumulator` (per-step state), both passed as arguments to `prefill_forward` and `decode_forward`. No convenience creation methods — callers allocate tensors directly (via LazyBuffer or `ttnn.from_torch`) and construct the dataclasses themselves. This eliminates hidden state, enables multiple concurrent penalty contexts, and makes the module a pure compute pipeline.
2. **LazyBuffer for all persistent buffers**: TTTv1's `_alloc_int_buffer()` / `_alloc_bf16_buffer()` are replaced by `LazyBuffer | ttnn.Tensor | None` fields in `Penalties1DConfig`. Three paths: `None` → auto-resolve with topology-aware defaults; `LazyBuffer` → declarative spec with user overrides; `ttnn.Tensor` → pre-allocated device tensor used directly (power user bypass). `_resolve_penalties1d_config()` fills defaults; `load_device_buffers()` materializes via `_materialize()`.
3. **Penalty math** port from `apply_penalties()` at `tt_penalties.py:32-75`:
   - Presence: `logits -= typecast(output_mask, bf16) * presence` (lines 38-43)
   - Frequency: `logits -= typecast(output_counts, bf16) * frequency` (lines 46-51)
   - Repetition: sign-dependent scaling with `combined_mask = prompt + output` (lines 58-73)
4. **Shape handling**: Reshape to `(-1, original_shape[-1])` before penalties, reshape back after (lines 305-308).
5. **`torch` import is lazy** — only inside `_resolve_penalties1d_config()` and lifecycle methods that construct host tensors. `decode_forward()` is pure ttnn.

---

## Step 2: Create `test_penalties_1d.py`

### Unit tests (no device)
1. `test_config_defaults` — verify defaults (max_batch_size=32, mesh_device=None, sub_core_grids=None)
2. `test_penalty_params_fields` — verify 5 PenaltyParams fields; `test_penalty_accumulator_fields` — verify 3 PenaltyAccumulator fields
3. `test_resolve_param_host_*` — scalar, list, None, truncation cases
4. `test_pad_batch_to_max_*` — padding, truncation, invalid dims → ValueError

### Device tests (parametrized)

```python
@pytest.mark.parametrize("ttnn_mesh_device", [(1,1),(1,2),(1,8)], indirect=True)
@pytest.mark.parametrize("mesh_shape,vocab_size", [
    (1,1)/1024, (1,1)/32000, (1,2)/32000, (1,2)/128256, (1,8)/128256
])
```

5. `test_create_penalty_params` — all 5 tensors allocated with correct shapes; `test_create_penalty_accumulator` — all 3 tensors allocated
6. `test_presence_penalty_math` — known logits + output_mask → PCC vs torch reference
7. `test_frequency_penalty_math` — known logits + counts → PCC vs torch reference
8. `test_repetition_penalty_math` — positive/negative logits, sign-dependent scaling → PCC
9. `test_penalty_full_lifecycle` — construct PenaltyParams/PenaltyAccumulator → prefill_forward (sets prompt_mask) → decode_forward → update_output_tokens
10. `test_penalties_change_argmax` — heavy penalty changes argmax token
11. `test_decode_forward_none_context` — returns logits unchanged
12. `test_from_model_args` — backward compat
13. `test_rejects_galaxy` — ValueError

### PCC verification

Implement `reference_apply_penalties()` in pure torch:
```python
def reference_apply_penalties(logits, prompt_mask, output_mask, output_counts, presence, frequency, repetition):
    logits -= output_mask.float() * presence      # presence
    logits -= output_counts.float() * frequency    # frequency
    combined = ((prompt_mask + output_mask) > 0).float()  # repetition
    scale = torch.where(logits > 0, torch.where(combined.bool(), 1/rep, 1.0), torch.where(combined.bool(), rep, 1.0))
    logits *= scale
    return logits
```

---

## Step 3: Create `sampling_1d.py`

### Sampling1DConfig dataclass

```python
@dataclass
class Sampling1DConfig:
    vocab_size: int                                        # Required. Caller pre-pads.
    mesh_device: ttnn.MeshDevice | None = None             # None → GetDefaultDevice()
    tt_ccl: TT_CCL | None = None                           # None → get_tt_ccl(mesh_device) [if multi-device]
    max_batch_size: int = 32                               # TTTv1 hardcode (tt_sampling.py:95)
    max_top_k: int = 32                                    # TTTv1 default (tt_sampling.py:96)
    sub_core_grids: ttnn.CoreRangeSet | None = None        # From args.sub_core_grids (line 100)
    sub_core_grid_topk: ttnn.CoreRangeSet | None = None    # From args.sub_core_grid_topk (line 101)
    start_core: ttnn.CoreCoord | None = None               # None → CoreCoord(0,0) (line 102)
    num_gather_links: int = 1                              # From GALAXY_NUM_LINKS calc (lines 104-111)
    sampling_memory_config: ttnn.MemoryConfig | None = None # None → DRAM_MEMORY_CONFIG (lines 112-115)
    allow_force_argmax: bool = False                        # From SAMPLING_AG_CONFIG (lines 118-125)
    num_argmax_gather_links: int | None = None             # None → same as num_gather_links
    ag_topology: ttnn.Topology | None = None               # None → Topology.Linear

    # --- Persistent buffer specs (LazyBuffer | ttnn.Tensor | None) ---
    # Same triple-type pattern as Penalties1DConfig.
    # NOTE: k/p/temp are per-call forward args — NOT buffers. They were TTTv1 module state
    # (k_tensor, p_tensor, temp_tensor) but are eliminated in TTTv2.
    #
    # Static index buffers (computed from vocab_size + num_devices, never mutated)
    index_offsets: LazyBuffer | ttnn.Tensor | None = None    # [1,1,32,max_top_k*num_devices], int32, TILE
    local_indices: LazyBuffer | ttnn.Tensor | None = None    # [1,1,32,vocab_per_device], uint16, TILE
    # Seed/ID buffers (seeds mutable via LazyBuffer.update(), user_ids static)
    seeds: LazyBuffer | ttnn.Tensor | None = None            # [32], uint32, ROW_MAJOR
    user_ids: LazyBuffer | ttnn.Tensor | None = None         # [32], uint32, ROW_MAJOR

    def is_resolved(self) -> bool:
        if self.mesh_device is None:
            return False
        if self.mesh_device.get_num_devices() > 1 and self.tt_ccl is None:
            return False
        return True
```

NOT config fields (derived): `cluster_shape`, `multi_step_reduction` (True only for [1,1]), `sampling_all_gather_axis` (dropped — only for 2D/Galaxy, rejected for 1D).

### _resolve_sampling1d_config() function

Fills scalar config fields: `mesh_device` → GetDefaultDevice(), `tt_ccl` → `get_tt_ccl(mesh_device)` if multi-device, `start_core` → CoreCoord(0,0), `sampling_memory_config` → DRAM_MEMORY_CONFIG, `num_argmax_gather_links` → num_gather_links, `ag_topology` → Topology.Linear.

Fills LazyBuffer fields (using same `_resolve_buf` pattern as penalties):
- `index_offsets` → LazyBuffer with source computed from vocab_size/num_devices (port lines 200-216)
- `local_indices` → LazyBuffer with source `arange(0, vocab_per_device)` expanded to batch (port lines 218-229)
- `seeds` → LazyBuffer with source `arange(0, 32)`, uint32, ROW_MAJOR (port lines 168-175)
- `user_ids` → LazyBuffer with source `arange(0, 32)`, uint32, ROW_MAJOR (port lines 176-181)

Asserts `is_resolved()`.

### Sampling1D class

```python
class Sampling1D(LightweightModule):

    # Construction
    def __init__(self, vocab_size: int, mesh_device: ttnn.MeshDevice | None = None):
        # Happy path + _bind_strategy()
    @classmethod
    def from_config(cls, config: Sampling1DConfig) -> "Sampling1D":
        # Power path + _bind_strategy()
    @classmethod
    def from_model_args(cls, mesh_device, tt_ccl, args, model_config=None) -> "Sampling1D":
        # Backward compat. Rejects Galaxy. Extracts GALAXY_NUM_LINKS, SAMPLING_AG_CONFIG, etc.

    # Strategy binding (TTTv2 pattern: eliminate static branching)
    def _bind_strategy(self):
        # Binds self._topk to _topk_single_device (1x1) or _topk_multi_device (1xN)

    # Device buffers (idempotent, like MLP1D.load_device_weights)
    def load_device_buffers(self):
        # Materializes all LazyBuffer fields from resolved config:
        #   self._index_offsets = self._materialize(self.config.index_offsets)
        #   self._local_indices = self._materialize(self.config.local_indices)
        #   self._seeds = self._materialize(self.config.seeds)
        #   self._user_ids = self._materialize(self.config.user_ids)
        #   self._log_probs_calculator = LogProbsCalculator(...)  # non-buffer

    # Forward
    def decode_forward(self, logits, *, k=None, p=None, temp=None,
                       seeds=None, tt_out_tok=None, enable_log_probs=False):
        # k/p/temp all None + allow_force_argmax → _sample_argmax
        # k/p/temp all provided → _sample_topk
        # Otherwise → ValueError
        # Returns: (token_ids, log_probs_or_none)
    def forward(self, logits, **kwargs):  # Dispatcher

    # Private: argmax path — port from tt_sampling.py:310-341
    def _sample_argmax(self, logits, tt_out_tok):
        # all_gather_async (if multi-device) → untilize → argmax → log_probs

    # Private: top-k sampling — port from tt_sampling.py:343-481
    def _sample_topk(self, logits, k, p, temp, tt_out_tok):
        # typecast → self._topk() → offset → seed → ttnn.sampling → log_probs

    # Private: top-k strategies (bound at init time, no if-else in forward)
    def _topk_single_device(self, x_bf16):
        # Split vocab in half → two topk → concat (port lines 346-371)
    def _topk_multi_device(self, x_bf16):
        # Local topk → all_gather (cluster_axis=None for 1D) → gather indices (port lines 372-421)

    # Private: CCL helper
    def _perform_all_gather(self, tensor, dim, cluster_axis, memory_config, num_links, buffer_key=None, dtype=None):
        # Prefers line_all_gather if available, else ttnn.all_gather (port lines 231-259)
```

### Key porting decisions

1. **k/p/temp are per-call parameters, NOT module state**: TTTv1 stores `self.k_tensor` etc. and mutates via `reset_params()`. TTTv2 passes them as arguments to `decode_forward()`. This eliminates 3 mutable buffers entirely (no `k_tensor`, `p_tensor`, `temp_tensor`).
2. **LazyBuffer for all persistent buffers**: TTTv1's `from_torch` calls in `__init__` and `_create_indices_tensors` are replaced by `LazyBuffer | ttnn.Tensor | None` fields in `Sampling1DConfig`. Same triple-type pattern as Penalties1DConfig: `None` → auto-resolve, `LazyBuffer` → custom spec, `ttnn.Tensor` → direct bypass.
3. **Seeds via LazyBuffer.update()**: `seeds` is a `LazyBuffer` in config, materialized in `load_device_buffers()`. SeedManager (at orchestrator level) calls `self.config.seeds.update(new_seeds_tensor)` to refresh per-request. Caller can also pass `seeds=` to `decode_forward()` to override per-call.
4. **Argmax vs sampling is per-call**: TTTv1 tracks `self._force_argmax_sampling`. TTTv2 checks whether k/p/temp are None per call. `allow_force_argmax` config gates availability.
5. **Strategy binding via `_bind_strategy()`**: `self._topk` is bound to either `_topk_single_device` or `_topk_multi_device` at init. No static if-else in forward.
6. **No penalty logic**: Penalties1D runs upstream. Sampling1D never sees penalty tensors.
7. **No trace caching**: Trace management stays in caller/orchestrator (SamplingGenerator or equivalent).
8. **`torch` import is lazy** — only inside `_resolve_sampling1d_config()` and `from_model_args()`.

---

## Step 4: Create `test_sampling_1d.py`

### Unit tests (no device)
1. `test_config_defaults` — max_batch_size=32, max_top_k=32, allow_force_argmax=False, num_gather_links=1
2. `test_config_custom` — explicit config preserves values
3. `test_config_is_resolved` — checks mesh_device and tt_ccl requirements

### Device tests (parametrized)

```python
@pytest.mark.parametrize("ttnn_mesh_device", [(1,1),(1,2),(1,8)], indirect=True)
@pytest.mark.parametrize("mesh_shape,vocab_size", [
    (1,1)/1024, (1,1)/32000, (1,1)/128256,
    (1,2)/32000, (1,2)/128256,
    (1,8)/128256,
])
```

4. `test_topk1_vs_argmax` — k=1, p=0.0, temp=1.0 must match `torch.argmax` on gathered logits. Primary PCC test (exact match, PCC=1.0). Multi-device logits sharded via `ShardTensor2dMesh(dims=(None,None))`.
5. `test_force_argmax` — allow_force_argmax=True, k/p/temp=None → matches torch.argmax
6. `test_topk_distribution` — k=32, p=1.0 → sampled token is within top-32 set
7. `test_error_on_partial_params` — k provided but not p/temp → ValueError
8. `test_from_model_args` — backward compat with mock args, k=1 matches argmax
9. `test_rejects_galaxy` — ValueError

### Helper functions
```python
def make_random_logits(batch_size, vocab_size, num_devices, mesh_device) -> ttnn.Tensor
def make_sampling_params_tt(mesh_device, batch_size=32, k_val=1, p_val=0.0, temp_val=1.0) -> (k, p, temp)
```

---

## Step 5: Composition Verification

Add to `test_sampling_1d.py`:

```python
def test_penalties_sampling_composition(ttnn_mesh_device, mesh_shape, vocab_size):
    penalizer = Penalties1D(vocab_size, ttnn_mesh_device)
    sampler = Sampling1D(vocab_size, ttnn_mesh_device)

    # Caller constructs params and accum directly (no convenience methods)
    params = PenaltyParams(
        prompt_mask=make_zero_buffer(32, vocab_size, ttnn_mesh_device),
        presence_penalties=make_bf16_buffer(32, 1, ttnn_mesh_device, fill=0.0),
        frequency_penalties=make_bf16_buffer(32, 1, ttnn_mesh_device, fill=0.0),
        repetition_penalties=make_bf16_buffer(32, 1, ttnn_mesh_device, fill=1.2),
        inverse_repetition_penalties=make_bf16_buffer(32, 1, ttnn_mesh_device, fill=1/1.2),
    )
    accum = PenaltyAccumulator(
        output_mask=make_zero_buffer(32, vocab_size, ttnn_mesh_device),
        output_counts=make_zero_buffer(32, vocab_size, ttnn_mesh_device),
        output_counts_gathered=make_zero_buffer(32, vocab_size, ttnn_mesh_device),
    )

    logits = make_random_logits(32, vocab_size, max(mesh_shape), ttnn_mesh_device)

    # Prefill: sets prompt_mask, returns logits unchanged
    prompt_tokens = torch.randint(0, vocab_size, (32, 128))
    logits = penalizer.prefill_forward(logits, params, accum, prompt_tokens)

    # Decode step (repeated)
    penalized = penalizer.decode_forward(logits, params, accum)
    k, p, temp = make_sampling_params_tt(ttnn_mesh_device, k_val=1)
    tokens, log_probs = sampler.decode_forward(penalized, k=k, p=p, temp=temp)
    penalizer.update_output_tokens(accum, tokens)

    # Verify: valid token range, no NaN, penalized != original
```

---

## Step 6: Audit (CLAUDE.md Steps 10-13)

### Step 10 — TTTv2 vs TTTv1 correctness audit

| TTTv1 Location | TTTv2 Equivalent | Verify |
|---|---|---|
| `tt_penalties.py:32-75` apply_penalties | `Penalties1D.decode_forward` | Same typecast/deallocate pattern |
| `tt_penalties.py:83-139` __init__ | `Penalties1DConfig` LazyBuffers + `load_device_buffers` | Same shapes, dtypes, shard_dims — now declarative via LazyBuffer |
| `tt_penalties.py:157-159` _copy_host_to_device | `LazyBuffer.update()` | Same from_torch(host) + copy_host_to_device_tensor pattern, now encapsulated |
| `tt_penalties.py:161-170` reset_params | Caller uses `LazyBuffer.update()` directly on PenaltyParams fields | Pad-to-32 logic moves to caller; no module convenience method |
| `tt_penalties.py:194-212` reset_prompt_tokens | `Penalties1D.prefill_forward()` | Same -1 masking, scatter_add — now inside prefill_forward |
| `tt_penalties.py:214-264` reset/update_output | `Penalties1D.reset/update_output_tokens` | Same zero-out + scatter_add (writes to PenaltyAccumulator) |
| `tt_penalties.py:266-289` token_bin_counts | `Penalties1D._token_bin_counts_and_mask` | Same scatter→tilize→add→slice→gt |
| `tt_sampling.py:63-229` __init__+indices | `Sampling1DConfig` LazyBuffers + `load_device_buffers` | Same shapes/dtypes — now declarative via LazyBuffer. k/p/temp buffers eliminated (per-call args) |
| `tt_sampling.py:231-259` _perform_all_gather | `Sampling1D._perform_all_gather` | Same CCL introspection + fallback |
| `tt_sampling.py:310-341` argmax path | `Sampling1D._sample_argmax` | Same all_gather_async→untilize→argmax |
| `tt_sampling.py:343-481` sampling path | `Sampling1D._sample_topk`+`_topk_*` | Same typecast→topk→gather→offset→sample |

### Step 11 — Dependency check
- Module-level: only `ttnn`, `dataclasses`, `typing`, `LightweightModule`, `tt_ccl`, `LazyBuffer`
- NO `import torch` at module level (lazy only — inside `_resolve_penalties1d_config()` and lifecycle methods)
- NO imports from `models.tt_transformers` or `models.common.sampling` except in `from_model_args`

### Step 12 — Test check
- All PCC checks run for every mesh shape — no skips
- Coverage target: >90%
- Test cases cover `(1,1)`, `(1,2)`, `(1,8)` mesh shapes
- Test cases cover small (1024), medium (32000), large (128256) vocab sizes

### Step 13 — Code quality
- No dead code, no magic constants (32 → `cfg.max_batch_size`), no stale comments
- No static if-else in `decode_forward` (strategy binding at init)
- Proper `deallocate()` on all intermediates matching TTTv1 pattern
- No `_alloc_int_buffer` / `_alloc_bf16_buffer` — all buffer allocation via `LazyBuffer.get_device_buffer()`
- No `_copy_host_to_device` — all host→device refresh via `LazyBuffer.update()`
- `_resolve_penalties1d_config()` is idempotent (calling twice produces same result)

---

## Verification Commands

```bash
# Penalties tests with coverage
python_env/bin/python -m pytest models/common/tests/modules/sampling/test_penalties_1d.py -v \
  --cov=models.common.modules.sampling.penalties_1d \
  --cov-report=term-missing \
  --cov-config=models/common/tests/setup.cfg

# Sampling tests with coverage
python_env/bin/python -m pytest models/common/tests/modules/sampling/test_sampling_1d.py -v \
  --cov=models.common.modules.sampling.sampling_1d \
  --cov-report=term-missing \
  --cov-config=models/common/tests/setup.cfg

# Reset devices if needed
source python_env/bin/activate && tt-smi -r
```

## Implementation Order

1. `lazy_buffer.py` → 2. `penalties_1d.py` → 3. `test_penalties_1d.py` (run, includes LazyBuffer tests) → 4. `sampling_1d.py` → 5. `test_sampling_1d.py` (run) → 6. Composition test → 7. Audit

## Test Case Collection Plan
See `models/common/modules/sampling/test_case_collection.md` for the test case collection plan (Phase A-D).

---

# call_chain_report.md
# Call-Chain Report: `tt_penalties.py` & `tt_sampling.py`

> **Generated**: 2026-02-20
> **Source files**:
> - `models/common/sampling/tt_penalties.py` — penalty transforms (presence, frequency, repetition)
> - `models/common/sampling/tt_sampling.py` — token selection (top-k, top-p, temperature, argmax)
> - `models/common/sampling/generator.py` — orchestrator (`SamplingGenerator`)

---

## Architecture Overview

Both modules are **never used independently in production**. They are always
orchestrated by `SamplingGenerator` (`models/common/sampling/generator.py:26`),
which owns one `TTSampling` instance and one `TTPenalties` instance.

---

## Two Pipelines

| Pipeline | Entry Point | Penalties? | Via SamplingGenerator? |
|----------|-------------|------------|------------------------|
| **A** — tt_transformers (production) | `Transformer` → `Generator` | Yes | Yes |
| **B** — llama3_70b_galaxy (production) | `TtTransformer` → `Generator` | Yes | Yes |

---

// NOTE: shouldn't expect to use this in production until end of month
// But in general, 1D should be stable! The exception would some batched prefill in the generator.py
## Pipeline A: tt_transformers (production) -1D 1x1 1x2 1x8

### Model-level classes

- **Model**: `Transformer` in `models/tt_transformers/tt/model.py`
- **Generator**: `Generator` in `models/tt_transformers/tt/generator.py:93`

### Call chain

```
Transformer.__init__()                                          model.py:146
  └─ SamplingGenerator.__init__()                               generator.py:43    [ONCE]
       ├─ TTSampling.__init__(mesh_device, tt_ccl, args)        generator.py:58    [ONCE]
       └─ TTPenalties.__init__(mesh_device, args)               generator.py:59    [ONCE]

Generator.prefill_forward_text() — prefill sampling             tt/generator.py:296 [PER-REQUEST]
  ├─ format_sampling_params(broadcast(...), 32)                 tt/generator.py:414
  ├─ _apply_prefill_sampling_state(model, ...)                  tt/generator.py:418
  │    ├─ sampling_module.reset_sampling_params(sampling_params) tt/generator.py:68
  │    │    ├─ TTSampling.reset_params(k, p, temp, log_probs)   generator.py:110
  │    │    └─ TTPenalties.reset_params(presence, freq, rep)     generator.py:130   [conditional]
  │    ├─ sampling_module.reset_prompt_tokens(prompt_tokens)     tt/generator.py:73
  │    │    └─ TTPenalties.reset_prompt_tokens(prompt_tokens)    generator.py:98
  │    └─ sampling_module.reset_output_state()                   tt/generator.py:74
  │         └─ TTPenalties.reset_output_tokens()                 generator.py:103
  └─ self.model[model_id].sampling.sample(logits)               tt/generator.py:473
       └─ SamplingGenerator.sample()                            generator.py:227
            (see hot-path below)

Generator.decode_forward_text() — param setup                   tt/generator.py:673
  │
  │  ── runs every decode call (when sampling_params is not None) ──  [PER-TOKEN]
  ├─ format_sampling_params(sampling_params_list[i], 32)        tt/generator.py:733
  ├─ sampling_module.reset_sampling_params(formatted_params)    tt/generator.py:738
  │    ├─ TTSampling.reset_params(...)                          generator.py:110
  │    └─ TTPenalties.reset_params(...)                         generator.py:130   [conditional]
  ├─ sampling_module.seed_manager.get_new_values()              tt/generator.py:739
  │
  │  ── only when reset_batch=True ──                                [PER-REQUEST] --> make this into a separate public API that user can call to work around the reshuffling the user ids!
  ├─ sampling_module.reset_prompt_tokens(prompt_chunks[i])      tt/generator.py:741
  │    └─ TTPenalties.reset_prompt_tokens(...)                  generator.py:98
  └─ sampling_module.reset_output_state(output_chunks[i])       tt/generator.py:742
       └─ TTPenalties.reset_output_tokens(tokens)               generator.py:103

Transformer.decode_forward() — decode sampling                  model.py:532       [PER-TOKEN]
  └─ self.sampling.sample(tt_logits, tt_out_tok=x)              model.py:536
       └─ SamplingGenerator.sample()                            generator.py:227
            (see hot-path below)

Generator._decode_forward_trace_text() — trace-based decode     tt/generator.py:869 [PER-TOKEN]
  └─ sampling_module.sample(logits=..., tt_out_tok=...)         tt/generator.py:915
       └─ SamplingGenerator.sample()                            generator.py:227
            (see hot-path below)
```

Prefill_forward_text() has a loop over each user and each user in the batch is prefilled and sampled individually with its own params — the loop processes one user at a time.

---

// NOTE: wait for the testing work done on the existing code
## Pipeline B: llama3_70b_galaxy (production)

### Model-level classes

- **Model**: `TtTransformer` in `models/demos/llama3_70b_galaxy/tt/llama_model.py`
- **Generator**: `Generator` in `models/demos/llama3_70b_galaxy/tt/generator.py:45`

### Call chain

```
TtTransformer.setup_decode()                                    llama_model.py:185
  └─ SamplingGenerator.__init__()                               generator.py:43    [ONCE]
       ├─ TTSampling.__init__(mesh_device, tt_ccl, args)        generator.py:58    [ONCE]
       └─ TTPenalties.__init__(mesh_device, args)               generator.py:59    [ONCE]

Generator.prefill_forward_text()                                galaxy/generator.py:316 [PER-REQUEST]
  ├─ format_sampling_params(sampling_params, max_batch_size)    galaxy/generator.py:316
  ├─ sampling_module.reset_sampling_params(sampling_params)     galaxy/generator.py:349
  │    ├─ TTSampling.reset_params(...)                          generator.py:110
  │    └─ TTPenalties.reset_params(...)                         generator.py:130   [conditional]
  ├─ sampling_module.reset_prompt_tokens(prefill_ids)           galaxy/generator.py:351
  │    └─ TTPenalties.reset_prompt_tokens(...)                  generator.py:98
  ├─ sampling_module.reset_output_state()                       galaxy/generator.py:352
  │    └─ TTPenalties.reset_output_tokens()                     generator.py:103
  └─ sampling_module.sample(tt_logits_batch)                    galaxy/generator.py:355
       └─ SamplingGenerator.sample()                            generator.py:227
            (see hot-path below)

Generator.decode_forward_text() — batch-reset path             galaxy/generator.py:553 [PER-REQUEST]
  ├─ format_sampling_params(sampling_params, max_batch_size)    galaxy/generator.py:555
  ├─ sampling_module.reset_sampling_params(sampling_params)     galaxy/generator.py:558
  │    ├─ TTSampling.reset_params(...)                          generator.py:110
  │    └─ TTPenalties.reset_params(...)                         generator.py:130   [conditional]
  ├─ sampling_module.reset_prompt_tokens(prompt_tokens)         galaxy/generator.py:560
  │    └─ TTPenalties.reset_prompt_tokens(...)                  generator.py:98
  └─ sampling_module.reset_output_state(output_tokens)          galaxy/generator.py:561
       └─ TTPenalties.reset_output_tokens(tokens)               generator.py:103

TtTransformer.decode_forward() — decode sampling                llama_model.py:645  [PER-TOKEN]
  └─ self.sampling.sample(tt_logits[0], tt_out_tok=x)           llama_model.py:645
       └─ SamplingGenerator.sample()                            generator.py:227
            (see hot-path below)

Generator._decode_easy_trace_text() — trace split sampling      galaxy/generator.py:739 [PER-TOKEN]
  └─ self.model.sampling.sample(logits=..., tt_out_tok=...)     galaxy/generator.py:740
       └─ SamplingGenerator.sample()                            generator.py:227
            (see hot-path below)
```

---

## Hot Path: SamplingGenerator.sample() internals

This is the **per-token** path shared by both Pipeline A and Pipeline B.
It is the performance-critical section.

```
SamplingGenerator.sample(logits, tt_out_tok=...)                generator.py:227   [PER-TOKEN]
  │
  ├─ [trace path] ─────────────────────────────────────────────
  │   If trace is already captured:
  │     SamplingGenerator._execute_trace(key)                   generator.py:217
  │       └─ ttnn.execute_trace(...)                            generator.py:224
  │   If trace not yet captured:
  │     SamplingGenerator.capture_trace(logits)                 generator.py:169
  │       └─ _run_sampling() × 2 (compile + capture)           generator.py:187,194
  │
  ├─ [no-trace path] ──────────────────────────────────────────
  │   SamplingGenerator._run_sampling(logits, penalties_on)     generator.py:157
  │     ├─ TTPenalties.apply(logits)                            generator.py:164   [PER-TOKEN, if penalties_on]
  │     │    └─ apply_penalties(logits, PenaltyContext)          tt_penalties.py:32
  │     │         ├─ presence penalty                           tt_penalties.py:38-43
  │     │         ├─ frequency penalty                          tt_penalties.py:46-51
  │     │         └─ repetition penalty                         tt_penalties.py:54-73
  │     └─ TTSampling.forward(logits, tt_out_tok)               generator.py:166   [PER-TOKEN]
  │          ├─ [argmax fast path]                              tt_sampling.py:310
  │          │    ├─ all_gather_async (if multi-device)         tt_sampling.py:316
  │          │    ├─ ttnn.untilize                              tt_sampling.py:330
  │          │    └─ ttnn.argmax                                tt_sampling.py:331
  │          └─ [top-k/p/temp path]                             tt_sampling.py:344
  │               ├─ ttnn.topk (local per-device)              tt_sampling.py:374
  │               ├─ _perform_all_gather (values)               tt_sampling.py:386
  │               ├─ _perform_all_gather (indices)              tt_sampling.py:412
  │               ├─ add device offsets → global indices        tt_sampling.py:438
  │               ├─ ttnn.manual_seed                           tt_sampling.py:454
  │               └─ ttnn.sampling(values, indices, k, p, temp) tt_sampling.py:460
  │
  └─ [post-sampling penalty update] ───────────────────────────
      TTPenalties.update_output_tokens(tt_out)                  generator.py:263   [PER-TOKEN, if penalties_on]
        └─ token_bin_counts_and_mask(new_tokens, src, ...)      tt_penalties.py:266
             ├─ ttnn.scatter_add                                tt_penalties.py:267
             ├─ ttnn.tilize                                     tt_penalties.py:271
             ├─ ttnn.add (accumulate counts)                    tt_penalties.py:275
             ├─ ttnn.slice (shard to local vocab)               tt_penalties.py:278
             └─ ttnn.gt (update binary mask)                    tt_penalties.py:288
```

---

## Per-Function Frequency Summary

### tt_penalties.py

| Function | Position | Frequency | Call Sites (external) |
|----------|----------|-----------|-----------------------|
| `TTPenalties.__init__` | Pipeline start | **Once** (model init) | `generator.py:59` |
| `reset_params(presence, frequency, repetition)` | Request setup | **Per-request** | `generator.py:130` |
| `reset_prompt_tokens(prompt_tokens)` | Post-prefill setup | **Per-request** | `generator.py:98` |
| `reset_output_tokens(tokens=None)` | Post-prefill setup | **Per-request** | `generator.py:103` |
| `apply(tt_logits)` → `apply_penalties()` | Hot path (logit transform) | **Per-token** | `generator.py:164` |
| `update_output_tokens(new_tokens)` → `token_bin_counts_and_mask()` | Hot path (state update) | **Per-token** | `generator.py:263` |
| `token_bin_counts_and_mask(...)` | Internal workhorse | **Per-token** + **per-request** | 3 internal call sites |
| `_alloc_int_buffer(...)` / `_alloc_bf16_buffer()` | Buffer allocation | **Once** (init) + **per-request** | ~10 in `__init__`, 2+ in reset methods |
| `_copy_host_to_device(dst, src)` | H2D transfer | **Per-request** | 4× from `reset_params` |
| `_pad_params(values)` | Param padding | **Per-request** | 3× from `reset_params` |
| `_pad_batch_to_max(tokens_2d, pad_value)` | Batch padding | **Per-request** | from `reset_prompt_tokens`, `reset_output_tokens` |

### tt_sampling.py

| Function | Position | Frequency | Call Sites (external) |
|----------|----------|-----------|-----------------------|
| `TTSampling.__init__` | Pipeline start | **Once** (model init) | `generator.py:58` |
| `_create_indices_tensors()` | Internal (init) | **Once** | 1× from `__init__` |
| `reset_params(k, p, temp, enable_log_probs)` | Request setup | **Per-request** | `generator.py:110` |
| `forward(x, tt_out_tok=None)` | **Hot path** (token selection) | **Per-token** | `generator.py:166` |
| `_perform_all_gather(...)` | Internal (multi-device) | **Per-token** (2× in `forward`) | 2 internal: values + indices |
| `_is_force_argmax_sampling(k, p, temp)` | Fast-path check | **Per-request** | from `reset_params`, read in `sample()` |
| `clamp(value, min, max)` | Param validation | **Per-request** | `generator.py:336` |

### generator.py (SamplingGenerator)

| Function | Position | Frequency | Callers |
|----------|----------|-----------|---------
| `SamplingGenerator.__init__` | Pipeline start | **Once** | `Transformer.__init__` (model.py:146), `TtTransformer.setup_decode` (llama_model.py:185) |
| `reset_sampling_params(sampling_params)` | Param setup | **Per-request** (prefill), **Per-token** (decode†) | `_apply_prefill_sampling_state` (tt/generator.py:68), `Generator.decode_forward_text` (tt/generator.py:738, galaxy/generator.py:558), `Generator.prefill_forward_text` (galaxy/generator.py:349) |
| `reset_prompt_tokens(prompt_tokens)` | Post-prefill setup | **Per-request** | `_apply_prefill_sampling_state` (tt/generator.py:73), `Generator.decode_forward_text` (tt/generator.py:741, galaxy/generator.py:560) |
| `reset_output_state(tokens=None)` | Post-prefill setup | **Per-request** | `_apply_prefill_sampling_state` (tt/generator.py:74), `Generator.decode_forward_text` (tt/generator.py:742, galaxy/generator.py:561) |
| `sample(logits, *, enable_trace, tt_out_tok)` | Token selection | **Per-token** | `Transformer.decode_forward` (model.py:536), `TtTransformer.decode_forward` (llama_model.py:645), `Generator.prefill_forward_text` (tt/generator.py:473, galaxy/generator.py:355), `Generator._decode_forward_trace_text` (tt/generator.py:915), `Generator._decode_easy_trace_text` (galaxy/generator.py:740) |
| `_run_sampling(logits, penalties_on, tt_out_tok)` | Internal dispatch | **Per-token** | from `sample()` and `capture_trace()` |
| `capture_trace(logits, tt_out_tok)` | Trace capture | **Once** per config | from `sample()` on first call |
| `_execute_trace(key)` | Trace replay | **Per-token** | from `sample()` when trace exists |
| `reset_trace()` | Trace invalidation | **Per-config-change** | from `reset_sampling_params()` |
| `format_sampling_params(sampling_params, max_batch_size)` | Param formatting | **Per-request** (prefill), **Per-token** (decode†) | tt/generator.py:414, tt/generator.py:733, galaxy/generator.py:316, galaxy/generator.py:555 |

> **†** In Pipeline A's `decode_forward_text`, `format_sampling_params` → `reset_sampling_params` → `seed_manager.get_new_values()` run on **every decode call** when `sampling_params is not None` (lines 733-739). Only `reset_prompt_tokens` and `reset_output_state` (lines 741-742) are gated by `if reset_batch:`. This means the param-setup path is effectively **per-token** in the decode loop, while the penalty-state-reset path remains **per-request**.

---

## Import Graph

```
models/common/sampling/generator.py
  ├─ imports TTPenalties   from .tt_penalties   (only consumer in production)
  └─ imports TTSampling    from .tt_sampling    (only consumer in production)

models/tt_transformers/tt/model.py
  └─ imports SamplingGenerator from models.common.sampling.generator

models/demos/llama3_70b_galaxy/tt/llama_model.py
  └─ imports SamplingGenerator from models.common.sampling.generator

models/tt_transformers/tt/generator.py
  └─ imports format_sampling_params from models.common.sampling.generator

models/demos/llama3_70b_galaxy/tt/generator.py
  └─ imports format_sampling_params from models.common.sampling.generator

```

---

## Observations for TTTv2 Refactoring

1. **`TTPenalties` has exactly ONE external consumer**: `SamplingGenerator`.
   It is never instantiated or called from anywhere else. This makes it a
   clean extraction target — the TTTv2 `Penalties1D` module only needs to
   satisfy `SamplingGenerator`'s interface.

2. **`token_bin_counts_and_mask`** is the most-called internal function. It
   runs both **per-request** (from `reset_prompt_tokens`/`reset_output_tokens`)
   AND **per-token** (from `update_output_tokens`), making it the single
   most important function to optimize.

3. **The argmax fast path** (`_force_argmax_sampling`) bypasses the entire
   top-k/p/temp pipeline including `_perform_all_gather`. When k=1, p=1.0,
   temp=1.0, it falls through to `ttnn.argmax` directly. This path is
   checked at the `SamplingGenerator` level too (for trace key selection).

4. **Penalty activation is lazy**: `_penalties_active` is only set to `True`
   when any penalty parameter differs from the defaults (presence=0.0,
   frequency=0.0, repetition=1.0). When penalties are inactive, all
   `TTPenalties` per-request and per-token calls are short-circuited in
   `SamplingGenerator` via early returns.

---

# Core & Device Execution Model (Sampling)

> **Generated**: 2026-05-29
> **Sources**:
> - `ttnn/cpp/ttnn/operations/reduction/sampling/device/sampling_program_factory.cpp` — the `ttnn.sampling` device op
> - `models/common/modules/sampling/sampling_1d.py` — TTTv2 module
> - `models/common/sampling/tt_sampling.py` — TTTv1 source

Two orthogonal axes are commonly conflated: **single _device_** (1×1 mesh topology, one chip) vs **single _Tensix core_** (core placement within a chip). Neither the plan above nor the Python modules reason about core-level execution — they only forward `sub_core_grids` / `sub_core_grid_topk` / `start_core` to the ttnn ops. The single-core semantics live entirely in the C++ sampling op.

## Single Tensix core execution: one core per user

`ttnn.sampling` parallelizes over the **batch (user) dimension, not vocab**. The input is `[1, 1, B, K*num_devices]`; the program factory derives one core per row:

```cpp
// sampling_program_factory.cpp:55-65
uint32_t Ht = (input_shape[0] * input_shape[1] * input_shape[2]) / tile_height; // (1·1·B)/32
uint32_t Wt = input_shape[3] / tile_width;
auto num_cores = Ht * tile_height;                                              // == B

CoreRangeSet core_grid = num_cores_to_corerangeset(num_cores, compute_with_storage_grid_size, true);
if (sub_core_grids.has_value()) {
    core_grid = sub_core_grids.value();   // Python-supplied override
}
auto cores = corerange_to_cores(core_grid, num_cores, true);
```

For `B=32`: `Ht=1`, `num_cores=32` → **32 cores, one user per core**. A separate writer+compute kernel is instantiated per core, with the core index `i` (= user index) baked in as a compile-time arg (`sampling_program_factory.cpp:253-321`).

Consequences:
- Each core does the **entire** top-p / temperature / RNG / argmax over that user's `K*num_devices` candidates (e.g. 32·8 = 256 values). Vocab is **not** spread across cores in this op.
- `k`/`p`/`temp` buffers are broadcast to every core (`k_chunk_size = num_cores * 4B`, lines 196-217); each core NOC-reads its own scalar via index `i`. Hence k/p/temp are `[B]`-shaped ROW_MAJOR tensors.

## sub_core_grids / sub_core_grid_topk / start_core

Which physical cores get used is computed host-side and passed as the `sub_core_grids` override above:

```python
# sampling_1d.py:172-179 (load_device_buffers)
self._sampling_sub_core_grids = (
    ttnn.num_cores_to_corerangeset_in_subcoregrids(
        cfg.start_core, cfg.max_batch_size, cfg.sub_core_grids, row_wise=True
    )
    if cfg.sub_core_grids is not None else None
)
```

`num_cores_to_corerangeset_in_subcoregrids(start_core, max_batch_size, sub_core_grids, row_wise=True)` = "take `max_batch_size` cores from `sub_core_grids`, starting at `start_core`, row-wise." That set becomes the `core_grid` override at `sampling_program_factory.cpp:63`. So `start_core` literally decides which physical core is "user 0."

Two grids, placed independently:

| Grid | Consumed by |
|---|---|
| `sub_core_grids` | sampling op (one-core-per-user), plus `typecast`, `untilize`, `manual_seed`, log-probs |
| `sub_core_grid_topk` | only `ttnn.topk` (vocab reduction — different optimal layout) |

**Perf note**: TTTv1 recomputes this corerangeset *inside every* `forward()` (`tt_sampling.py:552-556`); `sampling_1d.py` hoists it to `load_device_buffers()` once (`self._sampling_sub_core_grids`). Same cores, out of the per-token hot path.

## Single device (1×1 mesh) = `multi_step_reduction`

Distinct from single-core — this is mesh topology (one chip), the `[1,1]` special case, bound at init:

```python
# sampling_1d.py:114-121
self._multi_step_reduction = list(cluster_shape) == [1, 1]
self._topk = self._topk_single_device if self._multi_step_reduction else self._topk_multi_device
```

Three single-device specializations:

1. **No cross-device gather.** Multi-device shards vocab across chips → local top-k → all-gather candidates. Single device has the full vocab on one chip, so it **splits the vocab in half and runs top-k twice** (`_topk_single_device`, lines 340-368), emulating a 2-virtual-device reduction. Hence `num_devices_in_mesh = 2` even on one chip (lines 530-533), giving index offsets `{0, V/2}`.
2. **Argmax skips the all-gather** — `_argmax_noop` vs `_argmax_all_gather` (lines 240-262).
3. **TTTv2 bug fix in single-device index width** (lines 559-566): TTTv1 built `tt_indices_tensor` at width `V/2` then split it *again* in `forward` → V/4-wide index halves against V/2-wide logit halves (mismatch). TTTv2 builds it at full width `V` with each half holding `arange(V/2)` twice (`_make_local_indices`), so post-split it's V/2 vs V/2. Only the `[1,1]` path was affected.

## Terminology summary

- **Single _device_** (`[1,1]` mesh, `multi_step_reduction`): split-vocab-twice top-k, no CCL, argmax noop. A Python-level concern, handled by `_bind_strategy()`.
- **Single _Tensix core_**: `ttnn.sampling` assigns one core per user (`num_cores = batch_size`); `sub_core_grids` / `start_core` only pick which physical cores. A C++ op concern — vocab work for a user is entirely on that user's one core (batch-parallel, not vocab-parallel).
<!-- END VERBATIM: models/common/models/llama3_8b/Sampling Notes.md -->

<a id="source-15-models-common-models-llama3-8b-todo-md"></a>

### Source 15: `models/common/models/llama3_8b/TODO.md`

[Back to Source Index](#source-index) | [Previous: `models/common/models/llama3_8b/Sampling Notes.md`](#source-14-models-common-models-llama3-8b-sampling-notes-md) | [Next: `models/common/models/llama3_8b/TRACE_FULL_GRAPH_INVESTIGATION.md`](#source-16-models-common-models-llama3-8b-trace-full-graph-investigation-md)

<!-- BEGIN VERBATIM: models/common/models/llama3_8b/TODO.md -->
- [x] N300
- [x] add TTFT output
- [x] make sure the expected metrics from PERF.md
- [x] print the generated text from the model demo
<!-- - [ ] add checks to make sure fabric config is correctly set # this is not very important as long as we use the correct pytest fixture -->
- [x] debug bad text output.
- [x] make batch=32 in teacher forcing
- [x] T3K
- [x] try to explain the perf difference between TTTv1 and TTTv2 demos
- [ ] refactor the executors
    - [x] capture trace during compile
    - [x] check Sampling in the design doc
    - [x] input and output contracts
- [ ] refactor the generator and the demo code as well --> the goal is to have a standard code for executors, generators, and demos!
  - [x] remove compile() --> reduce API surface area --> users just use compile_prefill() and compile_decode() instead!
  - [ ] other todos in executor.py
  - [ ] refactor the executors further to allow composition of orthogonal features such as paged attention, prefix caching, validate_configs, etc. (search composable feature in executor.py)
  - [ ] solve trace problems: models/common/models/llama3_8b/MODEL_METHOD_SOLUTION.md and models/common/models/llama3_8b/TRACE_FULL_GRAPH_INVESTIGATION.md
  - [ ] refactor demo.py helper functions and reference data loading etc.
  - [ ] review function contracts, e.g., models/common/models/RUN_PERF_BENCHMARK_PLAN.md
- [ ] make sure to use device sampling in TTTv2 demos if TTTv1 also uses device sampling
- [ ] use pure TTTv2 model construction
- [ ] add type hints
- [ ] vLLM integration working on vLLM nightly and model ci (critical paths) --> customer team accepted models --> maybe there are additional pain points!
    - use AGENTS.md to streamline the process
    - add SKILLS.md to TTTv2 as well --> context specific AGENTS.md
    - Codex can create skills from the chat window and Claude may have a different format? --> we should try to make agent-agnostic skills!
- [ ] rethink the kv_cache allocation design
    - vLLM always allocates paged kv_cache
    - demo wants to test accuracy and performance on traced execution in CI --> should use traced executor but it must use paged kv_cache?!
    - demo may allocate kv_cache differently for testing and debugging purposes --> allow that? --> or we should just always use paged kv_cache?
- [ ] enable tracing on prefill
- [ ] get_block_size in executor.py should not need to look at the kv_cache --> block size is a static config?!
- [ ] add device perf with tracy signposts
- [ ] remove the use of from_model_args everywhere --> not user facing --> do not remove until we deprecate TTTv1! --> CI should just assert the configuration object matches expectation!
- [ ] remove imports from TTTv1 completely
- [ ] add a issue to track the TTTv1 model perf degradations -- first one around Jan 2025; second one around Jan 2026 --> we should not let this block TTTv2 work.
- [ ] deepseek over to TTTv2
- [ ] input and output contracts --> across the board --> all the modules
- [ ] better trace management: save trace to a file and load it into device trace region dynamically (through a separate cq)

# Key Inspirations from tt_cnn
| Pattern              | tt_cnn                                                      | Llama executor                              | Gap / Opportunity                                 |
|----------------------|-------------------------------------------------------------|---------------------------------------------|---------------------------------------------------|
| Abstract base class  | `Executor(ABC)` with enforced contract                      | No base class                               | Could standardize interface                       |
| Output schema        | `get_output_schema()` returns (shape, dtype, layout)        | None                                        | Useful for downstream consumers                   |
| Multi-CQ variants    | 4 executor types for different I/O strategies               | Single CQ only                              | Performance opportunity                           |
| Pipelined I/O        | Overlapped input transfer + execution                       | Sequential                                  | Throughput gains                                  |
| Input contract       | Always `host_input` tensor                                  | Mixed `torch`/`ttnn`                        | Cleaner boundary                                  |
| Single execute()     | Uniform interface                                           | `prefill_forward` + `decode_forward`        | LLM-specific complexity                           |

2. Performance Engineer (throughput/latency)
"The multi-CQ patterns in tt_cnn are gold. For Llama decode, MultiCQTracedModelOverlappedInputExecutor would let you overlap token embedding transfer with the previous decode step. The pipelined I/O executor is harder for LLMs because prefill lengths vary, but decode is fixed-shape — perfect for pipelining. I'd add TracedLlamaExecutorMultiCQ."

3. ML/Inference Engineer (model semantics)
"The Llama executor's complexity comes from real requirements: chunked prefill, paged attention, per-request sampling state, prefix caching. tt_cnn has none of that. Don't over-abstract — the prefill_forward/decode_forward split is correct. But I agree the input contract is messy. Define explicit dataclasses: PrefillRequest, DecodeRequest."<!-- END VERBATIM: models/common/models/llama3_8b/TODO.md -->

<a id="source-16-models-common-models-llama3-8b-trace-full-graph-investigation-md"></a>

### Source 16: `models/common/models/llama3_8b/TRACE_FULL_GRAPH_INVESTIGATION.md`

[Back to Source Index](#source-index) | [Previous: `models/common/models/llama3_8b/TODO.md`](#source-15-models-common-models-llama3-8b-todo-md) | [Next: `models/common/models/llama3_8b/hf_adaptor_refactor_goals.md`](#source-17-models-common-models-llama3-8b-hf-adaptor-refactor-goals-md)

<!-- BEGIN VERBATIM: models/common/models/llama3_8b/TRACE_FULL_GRAPH_INVESTIGATION.md -->
# Investigation: Full Graph Trace Capture for Prefill

## Problem Statement

The `TracedLLMExecutor.prefill_forward()` contains Llama-specific code:

```python
from models.common.models.llama3_8b.model import _all_gather_rmsnorm_tensor
# ...
logits = _all_gather_rmsnorm_tensor(self.model.norm, logits)
```

This violates the design principle "thick engine, thin model executor" — the engine should be model-agnostic.

## Root Cause

The trace captures with `get_last_token=-1`, which makes `model.prefill_forward()` return **hidden states** (not final logits). The executor then manually applies norm + all_gather + lm_head **outside** the trace.

### Why `get_last_token=-1`?

The trace output must work for **varying** `last_token_idx` values at replay time. By capturing all positions' hidden states, the executor can slice different positions after replay.

### Current Flow (Traced Prefill)

```
Trace captures:
  embed → layers → hidden_states (all positions)

Outside trace (executor does manually):
  slice(hidden_states, last_token_idx) → norm → all_gather → lm_head → logits
```

## Proposed Solution: Pre-allocate `last_token_idx` Tensor

Make `last_token_idx` a traced input (like tokens, page_table), so the full graph can be captured.

### Required Changes

#### 1. Model: Accept tensor for `get_last_token`

**Current** (`model.py:320`):
```python
def prefill_forward(
    self, x_embed, rot_mats, user_id, page_table,
    chunk_page_table=None, chunk_start_idx=None,
    get_last_token: int = -1,  # <-- Python int
) -> ttnn.Tensor:
```

**Needed**:
```python
def prefill_forward(
    self, x_embed, rot_mats, user_id, page_table,
    chunk_page_table=None, chunk_start_idx=None,
    get_last_token: int | ttnn.Tensor = -1,  # <-- Also accept scalar tensor
) -> ttnn.Tensor:
```

#### 2. Model: Handle tensor in slice operation

**Current** (`model.py:337-339`):
```python
get_last_token_floor = (get_last_token // 32) * 32
x = ttnn.slice(x, (0, 0, get_last_token_floor, 0),
               (1, 1, get_last_token_floor + 32, x.shape[-1]))
```

**Challenge**: `ttnn.slice` expects Python int coordinates, not tensors.

**Options**:
1. Use `ttnn.experimental.tensor_slice` (if it exists) that accepts tensor indices
2. Use a scatter/gather pattern with pre-computed index tensor
3. Compute slice indices on host, pass as separate tensor inputs

#### 3. Executor: Add `last_token_idx` to trace inputs

**In `_capture_and_run_prefill_trace()`**:
```python
# Add to host_inputs
last_token_idx_tt = ttnn.from_torch(
    torch.tensor([last_token_idx], dtype=torch.int32),
    device=self.mesh_device,
    ...
)
# Pass to model.prefill_forward(..., get_last_token=last_token_idx_tt)
```

**In `_easy_trace_prefill()` (replay)**:
```python
# Update last_token_idx tensor along with other inputs
copy_host_to_device(
    host_tensors=(..., last_token_idx_tensor),
    device_tensors=self.trace_inputs_prefill[prefill_seq_len],
)
```

## Key Blocker: `ttnn.slice` API

The main technical blocker is whether `ttnn.slice` can accept tensor-based indices.

### Current `ttnn.slice` behavior

From usage in codebase:
```python
# All usages pass Python ints, not tensors
ttnn.slice(x, (0, 0, start, 0), (1, 1, end, dim))
```

### Investigation needed

1. Check if `ttnn.slice` has a variant accepting tensor indices
2. Check if there's a `ttnn.gather` or `ttnn.index_select` that could work
3. Check if there's an experimental API for dynamic slicing

### Alternative: Pre-compute all slice positions

If dynamic slicing isn't supported, pre-compute a fixed set of slice positions (32, 64, 128, ...) and capture separate traces for each. This is less elegant but avoids the API limitation.

## Expected Outcome

If `ttnn.slice` (or alternative) can accept tensor indices:

```
Trace captures (full graph):
  embed → layers → slice(last_token_idx) → norm → all_gather → lm_head → logits

At replay:
  Update last_token_idx tensor → execute trace → get final logits directly
```

The executor no longer needs `_all_gather_rmsnorm_tensor` or any model-specific code.

## Next Steps

1. Investigate `ttnn` API for dynamic slicing with tensor indices
2. If available: Implement Option A (full graph capture)
3. If not available: Either implement Option B (model method) or use pre-computed slice positions

## Related Code Locations

| File | Lines | Description |
|------|-------|-------------|
| `executor.py` | 1137, 1205-1215 | Llama-specific import and usage |
| `executor.py` | 1269-1309 | `_capture_and_run_prefill_trace()` |
| `executor.py` | 1245-1267 | `_easy_trace_prefill()` |
| `llama3_8b/model.py` | 320, 334-349 | `get_last_token` handling in `prefill_forward()` |
| `llama3_8b/model.py` | 595-618 | `_all_gather_rmsnorm_tensor()` definition |
<!-- END VERBATIM: models/common/models/llama3_8b/TRACE_FULL_GRAPH_INVESTIGATION.md -->

<a id="source-17-models-common-models-llama3-8b-hf-adaptor-refactor-goals-md"></a>

### Source 17: `models/common/models/llama3_8b/hf_adaptor_refactor_goals.md`

[Back to Source Index](#source-index) | [Previous: `models/common/models/llama3_8b/TRACE_FULL_GRAPH_INVESTIGATION.md`](#source-16-models-common-models-llama3-8b-trace-full-graph-investigation-md) | [Next: `models/common/models/llama3_8b/perf_diff.md`](#source-18-models-common-models-llama3-8b-perf-diff-md)

<!-- BEGIN VERBATIM: models/common/models/llama3_8b/hf_adaptor_refactor_goals.md -->
# Llama3 8B HF Adaptor Refactor Goals

## Main Goal

Apply the pattern documented in `HF_ADAPTOR_PATTERN.md` to `llama3_8b` while preserving the currently verified T3K performance and token accuracy.

Success criteria:

- Active `llama3_8b` model/generator/demo path still has no `models.tt_transformers`, `tt_transformers`, or `from_model_args` dependencies.
- HF checkpoint/config/tokenizer adaptation is isolated behind a new `hf_adaptor.py` boundary.
- Demo prompt loading/token batching is owned by `models/common/tests/demos/llama3_8b/demo_utils.py` and returns the newer `(tokens, prompt_lens)` shape.
- `model.py` remains focused on TTTv2 model/module graph construction and forward behavior.
- Performance and accuracy are re-verified sequentially on TT hardware after implementation.

## Agent Goals

### Agent 1: HF Adaptor Boundary Audit

Goal: identify the exact code currently in `runtime_args.py`, `model.py`, `generator.py`, and `demo.py` that should move to `hf_adaptor.py` versus remain model-owned.

Deliverables:

- Proposed `hf_adaptor.py` public API.
- List of functions/classes/fields to move, keep, or delete.
- Risks to performance/accuracy parity from moving HF loading or state-dict conversion.

### Agent 2: Demo Utils Pattern Audit

Goal: compare prompt loading/token batching across TTTv2 demos and specify the demo-local `llama3_8b/demo_utils.py` API that follows the common direction without preserving unnecessary TTTv1 return shapes.

Deliverables:

- Proposed `demo_utils.py` functions and signatures.
- Mapping from current `input_preprocessing.py` return values to the desired `(tokens, prompt_lens)` shape.
- Any edge cases from instruct/chat-template clipping that must be preserved.

### Agent 3: Verification and Regression Risk Audit

Goal: identify the minimum static and TT hardware verification matrix needed after the refactor, and point out code paths most likely to regress performance or top-1/top-5 accuracy.

Deliverables:

- Static checks to run.
- Sequential TT hardware commands to run.
- Specific metrics expected from prior verification.
- Any cheap local parity checks that do not require TT hardware.

## Completion Notes

- Agent audits completed without file edits; implementation followed the recommended boundary split.
- Added `hf_adaptor.py` for HF model resolution, config/tokenizer loading, prompt encoding, HF state-dict conversion, and the `from_pretrained(...)` construction entrypoint.
- Added `models/common/tests/demos/llama3_8b/demo_utils.py` for prompt loading and `(tokens, prompt_lens)` batching; preserved the old instruct clipping semantics in a local parity check.
- Folded the Llama-3.1-8B precision policy into `model.py`.
- Moved RoPE scaling/table construction into `hf_adaptor.py`.
- Inverted the remaining dependency so `model.py` no longer imports `hf_adaptor.py`; `hf_adaptor.py` injects normalized model metadata, callbacks, and precomputed RoPE tables into the native runtime config.
- Retired `input_preprocessing.py` from the active path.
- Static checks, local executor tests, and sequential T3K performance/accuracy runs completed.
<!-- END VERBATIM: models/common/models/llama3_8b/hf_adaptor_refactor_goals.md -->

<a id="source-18-models-common-models-llama3-8b-perf-diff-md"></a>

### Source 18: `models/common/models/llama3_8b/perf_diff.md`

[Back to Source Index](#source-index) | [Previous: `models/common/models/llama3_8b/hf_adaptor_refactor_goals.md`](#source-17-models-common-models-llama3-8b-hf-adaptor-refactor-goals-md) | [Next: `models/common/models/llama3_8b/perf_results.md`](#source-19-models-common-models-llama3-8b-perf-results-md)

<!-- BEGIN VERBATIM: models/common/models/llama3_8b/perf_diff.md -->
# TTTv2 llama3.1-8b-instruct demo

N150:
| Mode        | Batch | Measured tok/s/u | Target tok/s/u | Measured TTFT (ms) | Target TTFT (ms) | Failures |
| ----------- | ----- | ---------------- | -------------- | ------------------ | ---------------- | -------- |
| performance | 1     | 26.4             | 28.3           | 107.6              | 104              | tok/s/u  |
| performance | 32    | 25.2             | 28.3           | 106.7              | 104              | tok/s/u  |
| accuracy    | 1     | 23.9             | 25.2           | 138.4              | 138              | tok/s/u  |
| accuracy    | 32    | 22.8             | 25.2           | 138.2              | 138              | tok/s/u  |

N300:
| Mode        | Batch | Measured tok/s/u | Target tok/s/u | Measured TTFT (ms) | Target TTFT (ms) | Failures         |
| ----------- | ----- | ---------------- | -------------- | ------------------ | ---------------- | ---------------- |
| performance | 1     | 40.6             | 44.2           | 73.4               | 67               | tok/s/u, ttft_ms |
| performance | 32    | 40.5             | 44.2           | 73.9               | 67               | tok/s/u, ttft_ms |
| accuracy    | 1     | 36.0             | 38.8           | 104.2              | 79               | tok/s/u, ttft_ms |
| accuracy    | 32    | 35.9             | 38.8           | 86.7               | 79               | tok/s/u, ttft_ms |

T3K:
| Mode        | Batch | Measured tok/s/u | Target tok/s/u | Measured TTFT (ms) | Target TTFT (ms) | Failures         |
| ----------- | ----- | ---------------: | -------------: | -----------------: | ---------------: | ---------------- |
| performance | 1     | 49.2             | 64.3           | 89.0               | 53               | tok/s/u, ttft_ms |
| performance | 32    | 48.7             | 64.3           | 88.7               | 53               | tok/s/u, ttft_ms |
| accuracy    | 1     | 47.7             | 60.8           | 96.1               | 81               | tok/s/u, ttft_ms |
| accuracy    | 32    | 46.4             | 60.8           | 87.8               | 81               | tok/s/u, ttft_ms |

# TTTv1 llama3.1-8b-instruct demo (simple_text_demo.py)

N300:
| Config        | Batch | Measured `tok/s/u` | Target `tok/s/u` | Measured TTFT (ms) | Target TTFT (ms) | Failures         |
| ------------- | ----- | ------------------ | ---------------- | ------------------ | ---------------- | ---------------- |
| `performance` | 1     | 16.89              | 44.2             | 73.4               | 67               | tok/s/u, ttft_ms |

T3K:
| Config        | Batch | Measured `tok/s/u` | Target `tok/s/u` | Measured TTFT (ms) | Target TTFT (ms) | Failures         |
| ------------- | ----- | ------------------ | ---------------- | ------------------ | ---------------- | ---------------- |
| `performance` | 1     | 39.47              | 64.3             | 89.0               | 53               | tok/s/u, ttft_ms |

## Why the numbers differ

Using the `simple_text_demo.py` `batch-1` + `performance` case as the concrete example:

1. The two demos do not run the same decode workload.
   - `models/tt_transformers/demo/simple_text_demo.py` configures this case with `max_generated_tokens=200`, `stop_at_eos=True`, and `enable_trace=True`.
   - `models/common/models/llama3_8b/demo.py` runs `PerfBenchmarkExecutor.run(..., num_decode_tokens=128, enable_trace=True)`.
   - So TTTv1 averages decode over "until EOS or 200 tokens", while TTTv2 averages over a fixed 128-token window. That alone means `tok/s/u` is not apples-to-apples.

2. The decode sampling path is different.
   - In `simple_text_demo.py`, the batch-1/performance test creates `SamplingParams(...)` and passes `sampling_params=device_sampling_params` into `generator.decode_forward(...)`.
   - If on-device sampling is supported, TTTv1 returns sampled token IDs directly from the device decode path.
   - In TTTv2, `_run_perf_benchmark()` does not pass `sampling_params` into `PerfBenchmarkExecutor.run(...)`.
   - That means `models/common/models/llama3_8b/executor.py` takes the non-sampling path: `model.gather_and_untilize_logits(logits)` followed by host readback and host `argmax`.
   - This adds full-logit gather/readback overhead to every decode iteration in TTTv2.

3. The frontend/runtime stack is different even after recent cleanup.
   - TTTv1 goes through `Generator.prefill_forward_text(...)` / `Generator.decode_forward(...)` in `simple_text_demo.py`.
   - TTTv2 goes directly through `TracedLlamaExecutor` and `PerfBenchmarkExecutor`.
   - So the comparison is not "same kernels, different model code only"; it also includes stack-level overhead differences.

4. The pass/fail targets are different.
   - `simple_text_demo.py` uses its own demo target table. For `Llama-3.1-8B`, the non-CI decode targets are `38` tok/s/u on `N300` and `45` tok/s/u on `T3K`.
   - TTTv2 `demo.py` uses `EXPECTED_METRICS` derived from `PERF.md`, which expects `44.2` tok/s/u on `N300` and `64.3` tok/s/u on `T3K`.
   - So even if the raw measurements were identical, the two demos could still report different failure status.

5. TTFT is also not a pure apples-to-apples comparison.
   - Both demos exclude compile from the reported TTFT.
   - But TTTv1 measures prefill through the `simple_text_demo.py` generator stack, while TTTv2 measures prefill through the direct executor stack in `PerfBenchmarkExecutor`.
   - The metric name is the same, but the surrounding runtime path is not identical.

## Are they actually measuring the same device runs?

No, not today.

If you look past the Python wrapper differences and compare the device call paths for the `simple_text_demo.py` `batch-1` + `performance` case against `models/common/models/llama3_8b/demo.py`, there are two direct mismatches in what runs on device:

1. Prefill is not the same device run.
   - In `simple_text_demo.py`, the batch-1/performance case has `enable_trace=True`, and `generator.prefill_forward_text(...)` is called without overriding it.
   - In TTTv2, `PerfBenchmarkExecutor.run()` builds
     `prefill_kwargs = {..., enable_trace=False, ...}`
     and both the warmup prefill and the timed TTFT prefill use that setting.
   - So TTTv1 is timing traced prefill, while TTTv2 is timing non-traced/direct prefill.
   - That alone is enough to make TTFT a different device measurement.

2. Decode is not the same device run.
   - In `simple_text_demo.py` batch-1/performance, the test config includes
     `sampling_params={"temperature": 0, "top_p": 0.08, "top_k": 32}`. --> todo)) is this happening on host or device?
   - That becomes `device_sampling_params = SamplingParams(...)`, and decode calls
     `generator.decode_forward(..., sampling_params=device_sampling_params, ...)`.
   - Inside `Generator.decode_forward()`, this sets `sampling_on_device=True`, and the actual device call becomes
     `self.model[i].ttnn_decode_forward(..., sampling_on_device=True)`.
   - In TTTv2, `_run_perf_benchmark()` never passes `sampling_params` to `PerfBenchmarkExecutor.run()`.
   - So `TracedLlamaExecutor.decode_forward()` runs with `sampling_on_device=False`, returns logits, then TTTv2 does `gather_and_untilize_logits` plus host-side `argmax`.
   - That means TTTv1 is benchmarking a decode path that returns sampled tokens from the device, while TTTv2 is benchmarking a decode path that materializes full logits and samples on the host.

## What is actually the same?

For the current `batch-1` + `performance` setup, these parts are now broadly aligned:

- same prompt file: `models/tt_transformers/demo/sample_prompts/input_data_questions_prefill_128.json`
- same prompt preprocessing helper: `preprocess_inputs_prefill(...)`
- same paged-attention page-table style
- same traced decode idea
- same single repeat batch

But those common pieces sit around different core measured device runs:

- TTTv1: traced prefill + decode with on-device sampling
- TTTv2: direct prefill + decode returning logits for host argmax

So the honest answer is: the current numbers are not just reporting the same device work through different abstractions. They are timing materially different device execution paths.

## Practical conclusion

Right now, `perf_diff.md` is useful as a "what do the two demos report?" snapshot, but not as a strict like-for-like performance comparison.

If we want a real TTTv1 vs TTTv2 comparison, we should force both paths to use:

- the same prompt file
- the same prompt preprocessing
- the same fixed decode length
- the same stop-at-EOS behavior
- the same sampling path (both on-device sampling or both host argmax)
- the same target table

Until then, the biggest code-level reason to expect TTTv2 decode to differ from `simple_text_demo.py` is that TTTv2 currently benchmarks host-logit decode, while `simple_text_demo.py` can benchmark device-sampled decode.
<!-- END VERBATIM: models/common/models/llama3_8b/perf_diff.md -->

<a id="source-19-models-common-models-llama3-8b-perf-results-md"></a>

### Source 19: `models/common/models/llama3_8b/perf_results.md`

[Back to Source Index](#source-index) | [Previous: `models/common/models/llama3_8b/perf_diff.md`](#source-18-models-common-models-llama3-8b-perf-diff-md) | [Next: `models/common/models/llama3_8b/tttv2_decoupling_goals.md`](#source-20-models-common-models-llama3-8b-tttv2-decoupling-goals-md)

<!-- BEGIN VERBATIM: models/common/models/llama3_8b/perf_results.md -->
we want to optimize llama-3.1-8B implementation in TTTv2 to achieve performance parity with TTTv1 ("new" matches "old") while maintaining the currently top-1 and top-5 accuracies. You can see that the current models/common/models/llama3_8b/model.py is using from_model_args of the modules to construct the model (and the model class itself also has from_model_args). It may be a good idea to first remove the use of from_model_args completely such that the model and its module instances are constructed in the pure TTTv2 way (like other models under models/common/models/.
For this task, write yourself a new goal and spawn agents in parallel -- as many as needed to do it better and faster -- do note that the TT hardware cannot be easily shared so better use it sequentially. Split the work into independent pieces, dispatch them concurrently, and synthesize the results as they return. Give each agent its own dedicated /goal.



# Llama-3.1-8B: old `tt_transformers` vs new `models/common` demo — perf sweep

Sweep run: Thu Jul 2 20:10–20:23, tmux session `old_perf_sweep`, all 12 runs **PASSED** (rc=0).

- Old demo: `models/tt_transformers/demo/simple_text_demo.py` (`python_env/bin/pytest`)
- New demo: `models/common/tests/demos/llama3_8b/demo.py` (actuals from the failing run in terminal 5)
- Env: `HF_MODEL=meta-llama/Llama-3.1-8B-Instruct`, `TT_CACHE_PATH=/localdev/gwang/.cache/meta-llama/Llama-3.1-8B-Instruct`, `HF_HOME=/proj_sw/user_dev/huggingface`
- Mesh mapping: `1x1 -> N150`, `1x2 -> N300`, `1x8 -> T3K` (4× N300 boards = 8 Wormhole chips)
- Old `-k` filters: `batch-1 and <opt>` and `batch-32 and not log-probs and <opt>` (log-probs variant has no new-demo counterpart)
- Per-run logs: `old_perf_sweep_logs/<mesh>-<opt>-<batch>.log`

## Old demo — measured

| mesh | MESH_DEVICE | optimization | batch | TTFT (ms) | tok/s/u | tok/s | result |
|---|---|---|---|---|---|---|---|
| 1x1 | N150 | performance | batch-1 | 177.1 | 9.49 | 9.49 | PASS |
| 1x1 | N150 | performance | batch-32 | 35.1 | 8.81 | 281.9 | PASS |
| 1x1 | N150 | accuracy | batch-1 | 206.8 | 9.11 | 9.11 | PASS |
| 1x1 | N150 | accuracy | batch-32 | 41.1 | 8.49 | 271.5 | PASS |
| 1x2 | N300 | performance | batch-1 | 90.4 | 25.4 | 25.4 | PASS |
| 1x2 | N300 | performance | batch-32 | 25.5 | 22.2 | 711.0 | PASS |
| 1x2 | N300 | accuracy | batch-1 | 96.3 | 23.4 | 23.4 | PASS |
| 1x2 | N300 | accuracy | batch-32 | 28.4 | 20.6 | 660.5 | PASS |
| 1x8 | T3K | performance | batch-1 | 39.9 | 70.3 | 70.3 | PASS |
| 1x8 | T3K | performance | batch-32 | 12.8 | 56.1 | 1794.0 | PASS |
| 1x8 | T3K | accuracy | batch-1 | 41.9 | 64.4 | 64.4 | PASS |
| 1x8 | T3K | accuracy | batch-32 | 13.3 | 52.2 | 1670.6 | PASS |

## tok/s/u: old vs new (actual) vs new target

| case | N150 (1x1) old / new / tgt | N300 (1x2) old / new / tgt | T3K (1x8) old / new / tgt |
|---|---|---|---|
| perf batch-1 | 9.5 / 26.4 / 28.3 | 25.4 / 35.1 / 44.2 | 70.3 / 28.3 / 64.3 |
| perf batch-32 | 8.8 / 25.4 / 28.3 | 22.2 / 33.9 / 44.2 | 56.1 / 28.4 / 64.3 |
| acc batch-1 | 9.1 / 23.8 / 25.2 | 23.4 / 31.2 / 38.8 | 64.4 / 27.0 / 60.8 |
| acc batch-32 | 8.5 / 23.0 / 25.2 | 20.6 / 30.1 / 38.8 | 52.2 / 27.3 / 60.8 |

New-demo targets from `EXPECTED_METRICS` in `models/common/tests/demos/llama3_8b/demo.py`; new actuals from the `FAILED ... tok/s/u X below target Y` lines in terminal 5.

## Headline

- **N150 (1 chip):** new TTTv2 is **faster** than old (26.4 vs 9.5 tok/s/u).
- **T3K (8 chips):** old is **~2.5× faster** than new (70.3 vs 28.3).

Old scales 9.5 -> 70.3 tok/s/u from N150 -> T3K (**~7.4×**, near-linear TP). New goes 26.4 -> 28.3 (**~1.07×** — basically no scaling). Every T3K target misses by >2× while N150/N300 only miss by ~10–25%. The new executor's single-chip decode is healthy, but its tensor-parallel path across 8 devices isn't amortizing — each T3K decode iter (~35ms) is barely faster than N150 (~38ms) despite 8× the compute. That's where to dig.

Caveat: old runs 200 decode tokens in the full demo loop, new runs 128 via `run_perf_benchmark`; both report steady-state decode tok/s/u excluding compile, so the scaling argument holds, but absolute TTFT definitions differ slightly (old = prefill_time/batch; new = measured TTFT).

## 2026-07-06 T3K TTTv2 optimization update

Current TTTv2 results after the construction refactor and T3K decode optimizations:

| case | old TTTv1 tok/s/u | old TTTv1 tok/s | new TTTv2 target tok/s/u | new TTTv2 current tok/s/u | new TTTv2 current tok/s | TTFT (ms) | decode latency (ms) | result |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| T3K perf batch-1 | 70.3 | 70.3 | 64.3 | 88.2 | 88.2 | 36.5 | 11.33 | PASS |
| T3K perf batch-32 | 56.1 | 1794.0 | 64.3 | 85.9 | 2747.5 | 35.5 | 11.65 | PASS |
| T3K accuracy-opt perf batch-1 | 64.4 | 64.4 | 60.8 | 80.9 | 80.9 | 48.0 | 12.35 | PASS |
| T3K accuracy-opt perf batch-32 | 52.2 | 1670.6 | 60.8 | 79.0 | 2527.0 | 37.6 | 12.66 | PASS |
| T3K token accuracy batch-1 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | PASS, top1 96.3%, top5 100.0% |

Verification commands:

```bash
python_env/bin/python -m py_compile models/common/models/executor.py models/common/models/generator.py models/common/models/llama3_8b/model.py models/common/modules/attention/attention_1d.py models/common/modules/mlp/mlp_1d.py models/common/modules/sampling/sampling_1d.py models/common/tests/demos/llama3_8b/demo.py
git diff --check -- models/common/models/executor.py models/common/models/generator.py models/common/models/llama3_8b/model.py models/common/modules/attention/attention_1d.py models/common/modules/mlp/mlp_1d.py models/common/modules/sampling/sampling_1d.py models/common/tests/demos/llama3_8b/demo.py
TT_METAL_HOME=/localdev/gwang/tt-metal-2 TT_CACHE_PATH=/localdev/gwang/.cache/meta-llama/Llama-3.1-8B-Instruct HF_HOME=/proj_sw/user_dev/huggingface HF_MODEL=meta-llama/Llama-3.1-8B-Instruct MESH_DEVICE=T3K python_env/bin/pytest -v 'models/common/tests/demos/llama3_8b/demo.py::test_llama3_8b[performance-1x8-batch-1]'
TT_METAL_HOME=/localdev/gwang/tt-metal-2 TT_CACHE_PATH=/localdev/gwang/.cache/meta-llama/Llama-3.1-8B-Instruct HF_HOME=/proj_sw/user_dev/huggingface HF_MODEL=meta-llama/Llama-3.1-8B-Instruct MESH_DEVICE=T3K python_env/bin/pytest -v 'models/common/tests/demos/llama3_8b/demo.py::test_llama3_8b[performance-1x8-batch-32]'
TT_METAL_HOME=/localdev/gwang/tt-metal-2 TT_CACHE_PATH=/localdev/gwang/.cache/meta-llama/Llama-3.1-8B-Instruct HF_HOME=/proj_sw/user_dev/huggingface HF_MODEL=meta-llama/Llama-3.1-8B-Instruct MESH_DEVICE=T3K python_env/bin/pytest -v 'models/common/tests/demos/llama3_8b/demo.py::test_llama3_8b[accuracy-1x8-batch-1]'
TT_METAL_HOME=/localdev/gwang/tt-metal-2 TT_CACHE_PATH=/localdev/gwang/.cache/meta-llama/Llama-3.1-8B-Instruct HF_HOME=/proj_sw/user_dev/huggingface HF_MODEL=meta-llama/Llama-3.1-8B-Instruct MESH_DEVICE=T3K python_env/bin/pytest -v 'models/common/tests/demos/llama3_8b/demo.py::test_llama3_8b[accuracy-1x8-batch-32]'
TT_METAL_HOME=/localdev/gwang/tt-metal-2 TT_CACHE_PATH=/localdev/gwang/.cache/meta-llama/Llama-3.1-8B-Instruct HF_HOME=/proj_sw/user_dev/huggingface HF_MODEL=meta-llama/Llama-3.1-8B-Instruct MESH_DEVICE=T3K python_env/bin/pytest -v 'models/common/tests/demos/llama3_8b/demo.py::test_llama3_8b[accuracy-1x8-token-accuracy]'
```

### Work done to reach the current numbers, ordered by performance impact

1. Added benchmark-only on-device decode feedback in `TracedLLMExecutor`. After the first traced decode replay, the benchmark keeps token feedback and position increments on device and skips host readback for steady-state iterations. This was the largest throughput improvement, taking T3K batch-1 from about `60.4 tok/s/u` to `88.2 tok/s/u`, and it does not affect teacher-forced accuracy.
2. Fixed the T3K fused attention output projection path. TTTv2 was checking `model_config["USE_FUSED_ALL_GATHER_MATMUL"]`, which was absent in the new pure-config path, so it silently used separate matmul, conversion, and reduce-scatter ops. The new path now follows the old model arg property, `args.use_fused_all_gather_matmul`. This moved T3K batch-1 from about `59.6 tok/s/u` to `60.4 tok/s/u`.
3. Switched the perf path to on-device top-k sampling by default, using `SamplingParams(temperature=0.0, top_k=32, top_p=0.08)`. This made the benchmark path eligible for device-side token feedback and removed avoidable host-side sampling work from the steady-state path.
4. Aligned the T3K short-context perf setup with the old demo: `max_seq_len=1024`, paged attention with `max_num_blocks=1024`, and per-user page allocation derived from `max_num_blocks // max_batch_size`. This made the comparison use the same short-context allocation regime and avoided measuring a heavier setup than the old path.
5. Propagated old TTTv1 communication tuning into the TTTv2 config path: attention decode all-gather-matmul config from `ATTN_AGMM_CONFIG`, MLP decode reduce-scatter config from `MLP_RS_CONFIG`, and sampling all-gather config from `SAMPLING_AG_CONFIG`.
6. Matched LM head compute-kernel fidelity to the old path by using the HiFi2 LM head kernel config.
7. Updated sampling to use the configured sub-core grid for `ttnn.manual_seed`, and made its all-gather cluster-axis handling work for both single-chip and multi-chip meshes. This was primarily a parity and correctness cleanup; the measured perf effect was neutral to slightly negative, but it kept sampling behavior aligned.
8. Added reset/page-table refresh handling so benchmark decode initializes state on the first replay and only refreshes the page table during device-feedback runs when it changes. This was required support code for the highest-impact device-feedback optimization.
9. Removed the Llama-3.1-8B model path's dependency on `from_model_args` construction. The demo and generator now build an explicit TTTv2 config and instantiate `Llama3Transformer1D(config)`.
10. Added an explicit `build_llama3_transformer_1d_config_from_model_args(...)` builder that wires embedding, RoPE, RMSNorm, attention, MLP, LM head, and sampling configs without relying on module factories. This was the enabling refactor that exposed the config parity gaps, but it was not by itself the direct throughput win.
11. Kept the eager executor API compatible with the traced executor by accepting `reset_batch`; eager accuracy ignores it. This had no direct perf impact, but it was needed to keep the accuracy verification path passing after the traced executor API change.

### Experiments that were not kept

- Splitting decode and sampling into separate executor-owned traces, matching the old TTTv1 trace shape, regressed T3K batch-1 to about `55.9 tok/s/u`.
- Rounding `Sampling1DConfig.max_batch_size` to 32 regressed T3K batch-1 to about `54.7 tok/s/u`.
- Changing `topk_global_indices` from `int32` to `uint32` was neutral to negative, about `58.4 tok/s/u`.

The main remaining non-functional cleanup is the repeated matmul warning about `program_config.allowed_worker_cores` being auto-populated. It did not block correctness or perf verification, but it points to program configs that should eventually be normalized before direct matmul use.

## 2026-07-06 TTTv1 dependency-boundary update

This follow-up moved the active TTTv2 model/generator/executor path further away from TTTv1 by removing direct `models.tt_transformers` imports from:

- `models/common/models/llama3_8b/model.py`
- `models/common/models/generator.py`
- `models/common/models/executor.py`

At this point in the history, the active Llama-3.1-8B TTTv2 path no longer imported `models.tt_transformers`. The old temporary adapter `legacy_tttv1_config.py` had been removed, and the TTTv1-derived runtime, precision, RoPE, and prompt preprocessing behavior had been moved into local Llama-3.1-8B files.

The later HF adaptor refactor supersedes that transitional file layout: runtime config and precision policy now live in `model.py`, HF config/tokenizer/state-dict/RoPE adaptation lives in `hf_adaptor.py`, and demo prompt batching lives in `models/common/tests/demos/llama3_8b/demo_utils.py`.

The model builder no longer accepts TTTv1 enum handles through a `LegacyModelArgsBuilderDeps` argument. TTTv2 now owns the Llama-3.1-8B precision policy, including the current decoder-31 performance override, and implements the non-Galaxy decode-only memory/program/norm config methods locally.

Verification after this dependency-boundary change:

| case | TTFT (ms) | tok/s/u | tok/s | decode latency (ms) | result |
|---|---:|---:|---:|---:|---|
| T3K performance batch-1 | 37.0 | 88.2 | 88.2 | 11.34 | PASS |
| T3K performance batch-32 | 36.0 | 85.9 | 2748.6 | 11.64 | PASS |
| T3K accuracy-opt perf batch-1 | 47.8 | 80.9 | 80.9 | 12.36 | PASS |
| T3K accuracy-opt perf batch-32 | 38.8 | 79.0 | 2527.2 | 12.66 | PASS |
| T3K token accuracy batch-1 | n/a | n/a | n/a | n/a | PASS, top1 96.3%, top5 100.0% |

After replacing the temporary TTTv1-backed adapter with `runtime_args.py`, T3K perf initially regressed because the local runtime args did not carry over the old non-power-of-2 sampling-logit padding flag. Restoring that TTTv2-owned equivalent brought batch-1 performance back to parity:

| case | TTFT (ms) | tok/s/u | tok/s | decode latency (ms) | result |
|---|---:|---:|---:|---:|---|
| T3K performance batch-1 | 40.9 | 88.3 | 88.3 | 11.33 | PASS |
| T3K performance batch-32 | 35.3 | 85.9 | 2748.9 | 11.64 | PASS |
| T3K accuracy-opt perf batch-1 | 48.1 | 80.9 | 80.9 | 12.35 | PASS |
| T3K accuracy-opt perf batch-32 | 37.6 | 79.0 | 2526.7 | 12.66 | PASS |
| T3K token accuracy batch-1, accuracy optimizations | n/a | n/a | n/a | n/a | PASS, top1 97.9%, top5 100.0% |

Additional verification for the fully decoupled path:

```bash
python_env/bin/python -m py_compile models/common/models/executor.py models/common/models/generator.py models/common/models/llama3_8b/model.py models/common/models/llama3_8b/hf_adaptor.py models/common/tests/demos/llama3_8b/demo.py models/common/tests/demos/llama3_8b/demo_utils.py
git diff --check -- models/common/models/executor.py models/common/models/generator.py models/common/models/llama3_8b models/common/tests/demos/llama3_8b
rg -n "models\\.tt_transformers|tt_transformers|from_model_args|legacy_tttv1_config|create_legacy_model_args" models/common/models/llama3_8b models/common/models/generator.py models/common/models/executor.py models/common/tests/demos/llama3_8b
TT_METAL_HOME=/localdev/gwang/tt-metal-2 TT_CACHE_PATH=/localdev/gwang/.cache/meta-llama/Llama-3.1-8B-Instruct HF_HOME=/proj_sw/user_dev/huggingface HF_MODEL=meta-llama/Llama-3.1-8B-Instruct MESH_DEVICE=T3K python_env/bin/pytest -v 'models/common/tests/demos/llama3_8b/demo.py::test_llama3_8b[performance-1x8-batch-1]'
TT_METAL_HOME=/localdev/gwang/tt-metal-2 TT_CACHE_PATH=/localdev/gwang/.cache/meta-llama/Llama-3.1-8B-Instruct HF_HOME=/proj_sw/user_dev/huggingface HF_MODEL=meta-llama/Llama-3.1-8B-Instruct MESH_DEVICE=T3K python_env/bin/pytest -v 'models/common/tests/demos/llama3_8b/demo.py::test_llama3_8b[performance-1x8-batch-32]'
TT_METAL_HOME=/localdev/gwang/tt-metal-2 TT_CACHE_PATH=/localdev/gwang/.cache/meta-llama/Llama-3.1-8B-Instruct HF_HOME=/proj_sw/user_dev/huggingface HF_MODEL=meta-llama/Llama-3.1-8B-Instruct MESH_DEVICE=T3K python_env/bin/pytest -v 'models/common/tests/demos/llama3_8b/demo.py::test_llama3_8b[accuracy-1x8-batch-1]'
TT_METAL_HOME=/localdev/gwang/tt-metal-2 TT_CACHE_PATH=/localdev/gwang/.cache/meta-llama/Llama-3.1-8B-Instruct HF_HOME=/proj_sw/user_dev/huggingface HF_MODEL=meta-llama/Llama-3.1-8B-Instruct MESH_DEVICE=T3K python_env/bin/pytest -v 'models/common/tests/demos/llama3_8b/demo.py::test_llama3_8b[accuracy-1x8-batch-32]'
TT_METAL_HOME=/localdev/gwang/tt-metal-2 TT_CACHE_PATH=/localdev/gwang/.cache/meta-llama/Llama-3.1-8B-Instruct HF_HOME=/proj_sw/user_dev/huggingface HF_MODEL=meta-llama/Llama-3.1-8B-Instruct MESH_DEVICE=T3K python_env/bin/pytest -v 'models/common/tests/demos/llama3_8b/demo.py::test_llama3_8b[accuracy-1x8-token-accuracy]'
```
<!-- END VERBATIM: models/common/models/llama3_8b/perf_results.md -->

<a id="source-20-models-common-models-llama3-8b-tttv2-decoupling-goals-md"></a>

### Source 20: `models/common/models/llama3_8b/tttv2_decoupling_goals.md`

[Back to Source Index](#source-index) | [Previous: `models/common/models/llama3_8b/perf_results.md`](#source-19-models-common-models-llama3-8b-perf-results-md) | [Next: `models/common/tests/traced_executor_can_trace_plan.md`](#source-21-models-common-tests-traced-executor-can-trace-plan-md)

<!-- BEGIN VERBATIM: models/common/models/llama3_8b/tttv2_decoupling_goals.md -->
# TTTv2 Llama-3.1-8B Decoupling Goals

## Main Goal

Decouple the TTTv2 Llama-3.1-8B implementation from TTTv1 by removing dependencies on `models.tt_transformers` from the active TTTv2 model/config path, while preserving the current verified T3K performance and token accuracy.

Success criteria:

1. `models/common/models/llama3_8b/model.py` no longer imports or directly depends on `models.tt_transformers`.
2. The Llama-3.1-8B demo/generator path constructs the model from TTTv2-owned config/data only.
3. Any remaining `models.tt_transformers` use in the active demo path is identified, minimized, and documented if not removable in this pass.
4. Current verified behavior is preserved:
   - T3K `performance-1x8-batch-1` remains at or above target.
   - T3K `performance-1x8-batch-32` remains at or above target.
   - T3K `accuracy-1x8-batch-1` remains at or above target.
   - T3K `accuracy-1x8-batch-32` remains at or above target.
   - T3K token accuracy remains at or above the current top-1/top-5 thresholds.
5. Hardware verification is run sequentially only.

## Agent Goals

### Agent 1: Dependency Map

Goal: Map all current imports and runtime dependencies from the TTTv2 Llama-3.1-8B path into `models.tt_transformers`.

Scope:

- Read-only analysis.
- Focus on `models/common/models/llama3_8b/`, `models/common/models/generator.py`, `models/common/models/executor.py`, and `models/common/tests/demos/llama3_8b/demo.py`.
- Identify which dependencies are configuration-only, tokenizer/checkpoint loading, model arg classes, performance constants, or reusable generic utilities.
- Return a prioritized removal plan and file/line references.

### Agent 2: Config Ownership

Goal: Design the smallest TTTv2-owned config replacement for the pieces currently pulled from TTTv1 model config objects.

Scope:

- Read-only analysis unless an isolated helper file is clearly appropriate.
- Focus on what `build_llama3_transformer_1d_config_from_model_args(...)` still consumes from TTTv1-shaped objects.
- Identify exact fields needed for TTTv2 construction, defaults, and where those values should live.
- Return a proposed TTTv2 config schema and migration steps.

### Agent 3: Data/Weights/Tokenizer Boundary

Goal: Determine how the active Llama-3.1-8B demo path loads weights, tokenizer, cache paths, and paged-attention metadata, and identify what still depends on TTTv1 infrastructure.

Scope:

- Read-only analysis.
- Focus on whether `TtModelArgs`, tokenizer creation, state dict loading, and cache paths can be replaced or wrapped by TTTv2-owned code.
- Return risks, required compatibility behavior, and a minimal implementation plan.

### Main Agent Implementation Goal

Goal: While agents analyze independent slices, implement the smallest safe decoupling step locally.

Initial local target:

- Remove direct `models.tt_transformers` imports from `models/common/models/llama3_8b/model.py` if possible.
- Prefer moving legacy adapter logic to demo/generator boundary only if full removal from the active path is not achievable in one pass.
- Keep edits surgical and preserve the already verified perf/accuracy behavior.
<!-- END VERBATIM: models/common/models/llama3_8b/tttv2_decoupling_goals.md -->

---

## Common Tests

<a id="source-21-models-common-tests-traced-executor-can-trace-plan-md"></a>

### Source 21: `models/common/tests/traced_executor_can_trace_plan.md`

[Back to Source Index](#source-index) | [Previous: `models/common/models/llama3_8b/tttv2_decoupling_goals.md`](#source-20-models-common-models-llama3-8b-tttv2-decoupling-goals-md) | Next: none

<!-- BEGIN VERBATIM: models/common/tests/traced_executor_can_trace_plan.md -->
# Traced Executor `can_trace` Plan

## Purpose

This plan focuses on one issue: `can_trace` policy in `TracedLLMExecutor`.

`TracedLLMExecutor` should continue to own the shared code that LLMs have in common:

- prefill batching over users;
- padded prefill length selection;
- page table selection for traced shapes;
- trace capture/replay lifecycle;
- decode trace capture/replay lifecycle;
- trace cache ownership;
- cleanup;
- warmup and benchmark integration.

The goal is not to turn `TracedLLMExecutor` into a generic trace runner. The goal is to make trace eligibility strict and centralized inside `TracedLLMExecutor`.

The existing model post-processing hook should remain:

```python
logits = self.model.post_process_prefill_output(logits, last_token_idx)
```

That pattern is acceptable: the shared LLM executor owns common flow, while the model owns model-specific output conversion.

## Current Problem

The traced prefill loop currently computes `can_trace`, then silently falls back to eager when tracing is unsupported:

```python
can_trace = self.model_args and self.model_args.can_enable_trace(prefill_seq_len, num_cached_tokens)

if can_trace:
    page_table_user = _get_prefill_trace_user_page_table(...)
else:
    page_table_user = _get_prefill_user_page_table(...)

if can_trace:
    logits = self._easy_trace_prefill(...)
    logits = self.model.post_process_prefill_output(logits, last_token_idx)
else:
    logits = self._eager._prefill_single_user(...)
```

That is the architectural issue.

`TracedLLMExecutor` is a traced executor. If it cannot trace a request, it should reject the request with a clear reason. It should not silently run eager code.

Silent eager fallback causes several problems:

- performance benchmarks can report mixed traced/eager behavior;
- trace regressions can be hidden;
- code paths become harder to reason about;
- validation and trace capture scope become tangled;
- `Traced*` APIs no longer mean “traced only.”

## Review Panel

The useful review group for this narrower question is:

- `TracedLLMExecutor owner`: owns shared LLM trace policy and lifecycle.
- `ModelConfig owner`: owns static trace capability fields such as supported sequence lengths.
- `Llama model owner`: validates that shared policy still supports Llama-specific constraints.
- `TTNN trace/runtime expert`: validates that trace-unsafe cases are rejected before capture.
- `Perf/demo owner`: ensures unsupported trace paths fail before timing and do not fall back to eager.
- `Serving/vLLM integration owner`: confirms prefix caching and chunked prefill behavior is explicit.

## Expected Panel Feedback

### TracedLLMExecutor Owner

`TracedLLMExecutor` should own the decision “can this executor trace this request?” because that is part of the traced executor contract.

The decision can consult `model_args`, but callers should not call `model_args.can_enable_trace(...)` directly inside the prefill loop. The traced executor can expose one internal assertion helper:

```python
self._assert_prefill_trace_supported(prefill_seq_len, num_cached_tokens)
```

The error message should include enough detail to debug the unsupported case.

The traced executor should not contain eager fallback branches in traced prefill. If trace is unsupported, raise.

### ModelConfig Owner

`model_args.can_enable_trace(...)` can remain as a static capability query. It should answer whether the model configuration supports a trace shape.

The executor policy should combine:

- model config capability;
- requested padded prefill length;
- number of cached tokens;
- max prefill chunk size;
- max sequence length;
- currently unsupported executor features such as prefix caching or chunked prefill.

If the model config lacks required trace capability information, constructing `TracedLLMExecutor` should fail early.

### Llama Model Owner

Keep `post_process_prefill_output` as a model hook. The current pattern is still a good shared-executor design:

- traced prefill returns hidden states;
- the model hook converts hidden states to logits for the selected token block.

That is a small model-specific hook, not an argument for moving the whole trace flow into `TracedLlamaExecutor`.

### TTNN Trace Runtime Expert

Trace support should be checked before any trace capture starts. Unsupported cases should never reach `begin_trace_capture`.

The trace support result should include a reason because trace failures are expensive to debug. Example reasons:

- no `model_args`;
- no trace-supported sequence lengths;
- padded sequence length unsupported;
- prefix caching unsupported;
- prefill length exceeds max chunk size;
- prefill length exceeds max sequence length.

### Perf/Demo Owner

Performance tests should assert that they use a traced path. If the traced executor cannot trace, fail before timing starts.

Do not allow traced perf tests to silently fall back to eager. That makes TTFT/decode measurements ambiguous.

### Serving/vLLM Integration Owner

Prefix caching and chunked prefill should be explicit trace policy decisions. Today, prefix caching disables trace through `num_cached_tokens == 0`. That should remain visible in the traced executor’s rejection reason.

Future support for prefix-cached tracing can be added by changing the traced executor policy in one place.

## Proposed Design

### 1. Centralize Prefill Trace Policy in `TracedLLMExecutor`

Add a small helper on `TracedLLMExecutor`, or keep the check inline if that reads better after the patch:

```python
def _assert_prefill_trace_supported(self, prefill_seq_len: int, num_cached_tokens: int) -> None:
    if not self.model_args:
        raise RuntimeError("Traced prefill requires model_args")
    if not self.model_args.can_enable_trace(prefill_seq_len, num_cached_tokens):
        raise RuntimeError(
            "Traced prefill is unsupported for "
            f"prefill_seq_len={prefill_seq_len}, num_cached_tokens={num_cached_tokens}"
        )
```

Do not add a result object or custom exception unless a caller actually needs to catch unsupported trace separately. A clear `RuntimeError` is enough for the first patch.

`model_args.can_enable_trace(...)` can keep the existing detailed policy. The architectural boundary is that `TracedLLMExecutor` owns when that check is applied and whether fallback is allowed.

### 2. Replace Eager Fallback With Explicit Rejection

In traced prefill, replace this:

```python
if can_trace:
    ...
else:
    logits = self._eager._prefill_single_user(...)
```

with this:

```python
self._assert_prefill_trace_supported(prefill_seq_len, num_cached_tokens)
```

Then the remaining code can assume trace is supported:

```python
page_table_user = _get_prefill_trace_user_page_table(...)
logits = self._easy_trace_prefill(...)
logits = self.model.post_process_prefill_output(logits, last_token_idx)
```

This keeps `post_process_prefill_output` exactly where it is.

### 3. Keep Eager Behavior Out of `TracedLLMExecutor`

If a caller wants eager behavior, it should instantiate/use `EagerLLMExecutor` or a model-specific eager wrapper.

Do not add flags to traced executors that change them into eager executors.

This aligns with removing `enable_trace` from `Traced*` APIs.

## Validation Scope

This plan does not primarily address module validation, but it interacts with the validation problem.

Once traced prefill no longer has eager fallback inside the traced path, validation can be scoped more clearly:

- validate eager warmup directly;
- avoid wrapping broad compile calls that capture traces;
- keep validation wrappers out of `begin_trace_capture` / `end_trace_capture`;
- eventually remove `suspend_module_input_validation()` from trace capture once validation scope is narrowed.

That is a follow-up. The first change should focus on `can_trace`.

## Migration Plan

### Phase 1: Add the Trace Support Assertion

- Add `_assert_prefill_trace_supported(...)` to `TracedLLMExecutor`, or keep an equivalent inline assertion if that is clearer.
- Use `RuntimeError` with a message that includes `prefill_seq_len` and `num_cached_tokens`.
- Do not add a dataclass, custom exception, or construction-time validation in the first patch.

Success criteria:

- `model_args.can_enable_trace(...)` is called only from the traced prefill support check.
- Unsupported trace cases fail with a clear message.

### Phase 2: Remove Eager Fallback From Traced Prefill

- Replace local `can_trace` branches with a required support check.
- Always use `_get_prefill_trace_user_page_table(...)` in traced prefill.
- Always run `_easy_trace_prefill(...)` for traced prefill.
- Keep `self.model.post_process_prefill_output(...)` after traced prefill.

Success criteria:

- `TracedLLMExecutor.prefill_forward(...)` never calls `_eager._prefill_single_user(...)` as fallback.
- Unsupported requests raise before trace capture.

### Phase 3: Update Tests

Add unit tests for:

- supported prefill trace request passes;
- unsupported sequence length raises;
- prefix caching raises;
- traced prefill no longer calls eager fallback.

Keep existing tests for:

- trace replay copies only mutable inputs;
- validation suspension behavior until validation scope is narrowed.

### Phase 4: Update Call Sites

- Ensure perf/demo paths instantiate traced executors only when they expect traced execution.
- If a path needs eager fallback, route it to `EagerLLMExecutor` before calling forward.
- Remove any remaining caller assumptions that `Traced*` can silently run eager.

## Non-Goals

This plan intentionally does not:

- introduce a full trace-plan abstraction;
- move `post_process_prefill_output` out of the model hook;
- make `TracedLLMExecutor` a generic trace runner detached from LLM semantics;
- redesign page table helpers;
- solve prefix-cached tracing;
- solve module validation scoping in the same patch.

## Recommended First PR

The first PR should be narrow:

- add `_assert_prefill_trace_supported(...)`, or an equivalent inline assertion;
- reject unsupported trace requests instead of falling back to eager;
- keep `post_process_prefill_output` unchanged;
- add focused tests around the new `can_trace` behavior.

This addresses the pointed architectural issue without over-abstracting the shared LLM executor.
<!-- END VERBATIM: models/common/tests/traced_executor_can_trace_plan.md -->

---

## TTTv1-Only Llama3 8B Perf Optimizations Audit

Date: 2026-07-08

Branch: `gongyu/tttv2_llama8b_perf_parity`

Objective: identify all TTTv1-only Llama3 8B performance optimizations found in the current codebase audit that are not yet fully represented in the TTTv2 Llama3 8B path.

Scope:

- TTTv1 path: `models/tt_transformers/demo/simple_text_demo.py` -> `models.tt_transformers.tt.Generator` -> `models.tt_transformers.tt.Transformer`.
- TTTv2 path: `models/common/tests/demos/llama3_8b/demo.py` -> `TracedLlamaExecutor` -> `models.common.models.executor.run_perf_benchmark`.
- No TT hardware runs were used for this audit. This is a source/log/history comparison.

Important correction:

- The earlier shorthand "TTTv1 defaults `allow_force_argmax=True`" is too broad.
- In the current TTTv1 code, non-Galaxy default `SAMPLING_AG_CONFIG` has `allow_force_argmax=False`.
- `allow_force_argmax=True` is model-specific for `Llama-3.1-8B` only when `executed_on_galaxy` is true.

### Finding 1: Galaxy Llama3 8B Model-Specific CCL Bundle Is Not Carried Into TTTv2

Status: confirmed TTTv1-only for Galaxy-class execution.

TTTv1 defines a model-specific Galaxy CCL bundle for `Llama-3.1-8B`:

- `attn_ln_ag`: `num_links=4`, `chunks_per_sync=10`, `num_workers_per_link=1`
- `ffn_ln_ag`: `num_links=4`, `chunks_per_sync=25`, `num_workers_per_link=1`
- `attn_agmm`: `num_links=4`, `chunks_per_sync=1`, `num_workers_per_link=1`
- `mlp_rs`: `num_links=4`, `chunks_per_sync=1`, `num_workers_per_link=1`, `rs_memory_config=L1`
- `sampling_force_argmax`: `allow_force_argmax=True`, `num_links=4`, `chunks_per_sync=10`, `num_workers_per_link=2`, `topology=Ring`

Evidence:

- TTTv1 model-specific CCL config is defined at `models/tt_transformers/tt/model_config.py:1080-1098`.
- It is installed into `ATTN_LN_AG_CONFIG`, `FFN_LN_AG_CONFIG`, `ATTN_AGMM_CONFIG`, `MLP_RS_CONFIG`, and `SAMPLING_AG_CONFIG` only when `executed_on_galaxy` is true at `models/tt_transformers/tt/model_config.py:1100-1116`.
- TTTv1 distributed norm consumes `ATTN_LN_AG_CONFIG` / `FFN_LN_AG_CONFIG` during decode all-gather at `models/tt_transformers/tt/distributed_norm.py:83-102`.
- TTTv1 attention fused AGMM consumes `ATTN_AGMM_CONFIG` at `models/tt_transformers/tt/attention.py:753-770`.
- TTTv1 MLP all-reduce/reduce-scatter path consumes `MLP_RS_CONFIG` at `models/tt_transformers/tt/mlp.py:298-315`.
- TTTv1 sampling config is consumed through `Sampling1D.from_model_args(...)` via `SAMPLING_AG_CONFIG` at `models/common/modules/sampling/sampling_1d.py:659-682`.

TTTv2 gap:

- TTTv2 `make_sampling_config()` builds `Sampling1DConfig` directly and does not pass `allow_force_argmax`, `num_argmax_gather_links`, `argmax_chunks_per_sync`, `argmax_num_workers_per_link`, or topology fields at `models/common/models/llama3_8b/model.py:1417-1429`.
- TTTv2 `_all_gather_rmsnorm_tensor()` uses `tt_ccl.get_num_links()`, `chunks_per_sync=10`, and `num_workers_per_link=2`, with no separate attn/ffn LN AG config at `models/common/models/llama3_8b/model.py:626-638`.
- TTTv2 `model_config["MLP_RS_CONFIG"]` uses default `chunks_per_sync=10`, `num_workers_per_link=2`, `rs_memory_config=DRAM`, and does not set the TTTv1 Galaxy `num_links=4`/L1/chunks=1 bundle at `models/common/models/llama3_8b/model.py:1005-1009`.
- TTTv2 passes those defaults into `MLP1DConfig` at `models/common/models/llama3_8b/model.py:1255-1318`.
- TTTv2 `Attention1DConfig` has fields for `decode_agmm_num_links`, `decode_agmm_chunks_per_sync`, and `decode_agmm_num_workers_per_link`, but the Llama3 8B explicit builder does not pass the TTTv1 model-specific values at `models/common/models/llama3_8b/model.py:1208-1248`.

Impact:

- This is not expected to affect the current N150/N300/T3K parity matrix directly because the TTTv1 4-link bundle is gated on Galaxy execution.
- It is a real TTTv1-only optimization if TTTv2 Llama3 8B is expected to cover Galaxy/TG/P150x4/P150x8-style execution with the same perf tuning.

### Finding 2: Galaxy DP4 1x8 Fused AGMM Enablement Is Not Ported To TTTv2

Status: confirmed TTTv1-only for Galaxy-class execution.

TTTv1 explicitly allows Llama3 8B on Galaxy DP4 routeable 1x8 row submeshes to use the same fused all-gather-matmul path as T3K:

- TTTv1 computes `use_galaxy_dp4_8b_submesh_agmm` for `is_galaxy_cluster`, `base_model_name == "Llama-3.1-8B"`, `num_devices == 8`, and `cluster_shape == (1, 8)` at `models/tt_transformers/tt/model_config.py:696-706`.
- TTTv1 includes that in `_use_fused_all_gather_matmul` at `models/tt_transformers/tt/model_config.py:707-713`.

TTTv2 gap:

- TTTv2 enables fused all-gather-matmul only when `num_devices == 8` and `not is_galaxy_cluster` at `models/common/models/llama3_8b/model.py:761-767`.
- This means TTTv2 carries the T3K fused AGMM path but not the TTTv1 Galaxy DP4 1x8 exception.

Impact:

- No expected impact for N150/N300/T3K.
- Potential perf gap for Galaxy-class TTTv2 Llama3 8B once that device class is in scope.

### Finding 3: TTTv1 Has Longer Traced-Prefill Supported Sequence Lengths

Status: addressed for the TTTv2 Llama3 8B 1D devices; TG remains outside this non-TG TTTv2 port.

TTTv1 supports more traced prefill sequence lengths for Llama3 8B:

- `N150`: `[128, 1024]`
- `N300`: `[128, 1024, 2048, 4096, 8192]`
- `T3K`: `[128, 1024, 2048, 4096, 8192]`
- `TG`: `[128, 1024, 2048, 4096, 8192]`

Evidence:

- TTTv1 model-specific traced prefill lengths are defined at `models/tt_transformers/tt/model_config.py:2430-2450`.
- TTTv1 device-specific `MAX_PREFILL_CHUNK_SIZE` for Llama3 8B is `N150=4K`, `N300=64K`, `T3K=128K`, `TG=128K`, `P150x4=128K` at `models/tt_transformers/tt/model_config.py:2378-2428`.

TTTv2 resolution:

- TTTv2 carries the same max prefill chunk sizes for N150/N300/T3K at `models/common/models/llama3_8b/hf_adaptor.py`.
- TTTv2 now uses the TTTv1 Llama3 8B traced-prefill length table for the supported 1D devices:
  - `N150`: `(128, 1024)`
  - `N300`: `(128, 1024, 2048, 4096, 8192)`
  - `T3K`: `(128, 1024, 2048, 4096, 8192)`
- The selected lengths are still filtered by `max_prefill_chunk_size` and `max_seq_len`.
- TTTv2 `can_enable_trace(...)` continues to reject prefix-cached prefill (`num_cached_tokens != 0`) and any length not in `trace_prefill_supported_seq_lens`.

Impact:

- No impact for the current 128-token perf benchmark.
- Long-prompt traced prefill parity is now enabled for N300/T3K 1D workloads up to 8192, subject to configured max sequence and chunk limits.
- TG still requires a separate TTTv2 TG/Galaxy Llama3 8B path; `models/common/models/llama3_8b/model.py` is explicitly `Llama3Transformer1D` and rejects 32-device meshes.

### Finding 4: Device-Resident Decode Feedback Is Benchmark-Only In TTTv2

Status: confirmed behavior difference; current perf benchmark already adopted this optimization.

TTTv1 uses device-resident sampled-token feedback in the old traced text demo path:

- TTTv1 capture writes sampled output into the decode token feedback buffer at `models/tt_transformers/tt/generator.py:1478-1490`.
- TTTv1 replay avoids full input refresh when on-device sampling is active and only refreshes page table changes at `models/tt_transformers/tt/generator.py:1516-1558`.
- TTTv1 sampling passes `tt_out_tok` into the sampling module at `models/tt_transformers/tt/generator.py:1667-1680`.

TTTv2 state:

- TTTv2 perf benchmark enables `_benchmark_device_decode_feedback` only inside `run_perf_benchmark()` when `sampling_params is not None` at `models/common/models/executor.py:2035-2039`.
- TTTv2 traced decode capture uses sampled token feedback and device-side position increment when that benchmark flag is enabled at `models/common/models/executor.py:1631-1675` and `models/common/models/executor.py:1692-1707`.

Impact:

- This is not missing for the current TTTv2 perf test anymore.
- It remains TTTv1-only outside the TTTv2 perf benchmark path. If TTTv2 text generation or serving paths use `TracedLLMExecutor.decode_forward()` directly without the benchmark flag, they do not automatically get the old TTTv1 device-feedback behavior.

### Finding 5: Host-Logit Gather Galaxy Link Count Differs

Status: confirmed TTTv1-only for Galaxy host-read / no-on-device-sampling decode paths.

TTTv1 host-logit decode gather uses a Galaxy-specific link count:

- TTTv1 sets `cluster_axis = 0 if self.args.is_galaxy else None` and `num_links = 2 if self.args.is_galaxy else 1` for logits all-gather at `models/tt_transformers/tt/model.py:818-835`.

TTTv2 gap:

- TTTv2 `gather_and_untilize_logits()` always uses `num_links=1` and no Galaxy cluster-axis override at `models/common/models/llama3_8b/model.py:585-600`.

Impact:

- No impact for current sampled perf mode when device sampling is active.
- Potential TTTv2 gap for Galaxy-class host-read paths, diagnostics, or teacher-forcing/decode paths that gather logits to host.

### Finding 6: Prefill Sampling Warmup Semantics Are Still Unsettled In TTTv2

Status: open parity candidate, not counted as a confirmed perf optimization yet.

TTTv1 simple text demo uses on-device sampling params for prefill when supported:

- `prefill_sampling_params = device_sampling_params if device_sampling_params is not None else None` at `models/tt_transformers/demo/simple_text_demo.py:1216`.
- TTTv1 passes those params into prefill warmup and measured prefill at `models/tt_transformers/demo/simple_text_demo.py:1221-1243`.

TTTv2 state:

- TTTv2 `run_perf_benchmark()` passes `sampling_params` into measured prefill at `models/common/models/executor.py:2075-2078`.
- But `_compile_prefill_and_decode()` still has an explicit TODO asking whether prefill should run sampling on device like TTTv1, and falls back to host argmax if logits are returned at `models/common/models/executor.py:1834-1838`.

Impact:

- This is not yet proven to be a perf gap in measured TTTv2 results.
- It should remain on the parity audit list because compile/warmup behavior can affect trace capture shape, output ownership, and first-token setup.

### Items Audited And Considered Already Adopted In TTTv2

These TTTv1 mechanisms were checked and are already represented in the current TTTv2 Llama3 8B path:

- Layer-31 performance precision override: TTTv1 JSON at `models/tt_transformers/model_params/Llama-3.1-8B-Instruct/performance_decoder_config.json`; TTTv2 equivalent in `Llama31DecoderPrecision.performance(...)` at `models/common/models/llama3_8b/model.py:75-84`.
- T3K fused attention output all-gather-matmul: TTTv1 at `models/tt_transformers/tt/model_config.py:696-713`; TTTv2 for non-Galaxy 1x8 at `models/common/models/llama3_8b/model.py:761-767` and `models/common/modules/attention/attention_1d.py:1068-1089`.
- Decode MLP DRAM-sharded program configs and residual memory config: TTTv2 carries these through `models/common/models/llama3_8b/model.py:871-945` and `models/common/models/llama3_8b/model.py:1250-1331`.
- LM-head grid and split sizing based on `668 * core_grid.num_cores`: TTTv1 at `models/tt_transformers/tt/model_config.py:2214-2236`; TTTv2 at `models/common/models/llama3_8b/model.py:881-891` and `models/common/models/llama3_8b/model.py:1333-1415`.
- Paged KV cache, 1024-block short perf setup, and 128-token perf prompt are represented in the TTTv2 demo and model construction.
- Non-power-of-2 sampling logit padding for multi-device meshes is carried by TTTv2 at `models/common/models/llama3_8b/hf_adaptor.py:465-470`.

### Current Highest-Priority Follow-Up

For current N150/N300/T3K perf parity, the audit did not find a confirmed TTTv1-only optimization that is both missing in TTTv2 and expected to move the existing 128-token perf benchmark.

For broader TTTv2 parity with TTTv1, the highest-confidence implementation target is to port the model-specific CCL bundle into the explicit TTTv2 Llama3 8B builder, especially:

1. `Sampling1DConfig` force-argmax / argmax AG fields.
2. `Attention1DConfig` fused AGMM link/chunk/worker fields.
3. MLP decode reduce-scatter `num_links`, memory config, chunks, and workers.
4. RMSNorm all-gather per-norm AG settings, if TTTv2 wants exact Galaxy tuning parity.
5. Galaxy DP4 1x8 fused AGMM enablement, if Galaxy-class Llama3 8B is in scope.


# More problems to fix

```
____________________________________________________________________________ test_llama3_8b[performance-1x8-batch-1] _____________________________________________________________________________
models/common/tests/demos/llama3_8b/demo.py:206: in test_llama3_8b
    _run_perf_benchmark(
models/common/tests/demos/llama3_8b/demo.py:400: in _run_perf_benchmark
    assert not failures, f"{case_name}: " + "; ".join(failures)
E   AssertionError: performance/batch-1: ttft_ms 46.3 above target 39.9
E   assert not ['ttft_ms 46.3 above target 39.9']
======================================================================================== warnings summary ========================================================================================
python_env/lib/python3.10/site-packages/pydantic/_internal/_config.py:291
  /localdev/gwang/tt-metal-2/python_env/lib/python3.10/site-packages/pydantic/_internal/_config.py:291: PydanticDeprecatedSince20: Support for class-based `config` is deprecated, use ConfigDict instead. Deprecated in Pydantic V2.0 to be removed in V3.0. See Pydantic V2 Migration Guide at https://errors.pydantic.dev/2.9/migration/
    warnings.warn(DEPRECATION_MESSAGE, DeprecationWarning)

<frozen importlib._bootstrap>:241
  <frozen importlib._bootstrap>:241: DeprecationWarning: builtin type SwigPyPacked has no __module__ attribute

<frozen importlib._bootstrap>:241
  <frozen importlib._bootstrap>:241: DeprecationWarning: builtin type SwigPyObject has no __module__ attribute

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
============================================================================================= PASSES =============================================================================================
-------------------------------------------------- generated xml file: /localdev/gwang/tt-metal-2/generated/test_reports/most_recent_tests.xml ---------------------------------------------------
======================================================================================= slowest durations ========================================================================================
89.17s call     models/common/tests/demos/llama3_8b/demo.py::test_llama3_8b[performance-1x8-batch-1]
65.32s call     models/common/tests/demos/llama3_8b/demo.py::test_llama3_8b[performance-1x2-batch-1]
65.07s call     models/common/tests/demos/llama3_8b/demo.py::test_llama3_8b[performance-1x1-batch-1]
0.93s setup    models/common/tests/demos/llama3_8b/demo.py::test_llama3_8b[performance-1x8-batch-1]
0.57s setup    models/common/tests/demos/llama3_8b/demo.py::test_llama3_8b[performance-1x2-batch-1]
0.51s setup    models/common/tests/demos/llama3_8b/demo.py::test_llama3_8b[performance-1x1-batch-1]
0.01s teardown models/common/tests/demos/llama3_8b/demo.py::test_llama3_8b[performance-1x8-batch-1]
0.00s teardown models/common/tests/demos/llama3_8b/demo.py::test_llama3_8b[performance-1x2-batch-1]
0.00s teardown models/common/tests/demos/llama3_8b/demo.py::test_llama3_8b[performance-1x1-batch-1]
==================================================================================== short test summary info =====================================================================================
PASSED models/common/tests/demos/llama3_8b/demo.py::test_llama3_8b[performance-1x1-batch-1]
PASSED models/common/tests/demos/llama3_8b/demo.py::test_llama3_8b[performance-1x2-batch-1]
FAILED models/common/tests/demos/llama3_8b/demo.py::test_llama3_8b[performance-1x8-batch-1] - AssertionError: performance/batch-1: ttft_ms 46.3 above target 39.9
assert not ['ttft_ms 46.3 above target 39.9']
```
