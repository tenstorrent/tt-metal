# Phi-3.5-mini with TTTv2

This directory is the model-owned TTTv2 path for Microsoft Phi-3.5-mini-instruct on one Blackhole P150.
It started as a copy of `models/common/models/phi4` (same Phi3 architecture) and adds Phi-3.5 specifics:
longrope short-factor RoPE tables (contexts up to 4096 tokens), single-device lanes, and P150 tuning.

It follows the Phi-4 pattern of direct executor construction from
`models/common/llm_runtime`. It is not part of the Llama/Qwen executor
consolidation and does not use `models/common/models/executor.py`.

## Product path

```text
Hugging Face checkpoint
  -> hf_adaptor.py: provider metadata, tokenizer, and weight conversion
  -> model.py: Phi-3.5-mini tensor graph composed from TTTv2 modules
  -> executor.py: direct composition of common runtime owners for one lane
  -> generator.py: vLLM construction, DP composition, and dispatch
```

## Files

| File | Responsibility |
| --- | --- |
| `hf_adaptor.py` | Resolve the Phi-3.5-mini provider configuration/tokenizer and build runtime metadata |
| `weight_utils.py` | Convert and map HF weights into the model-owned layout |
| `model.py` | Build and execute the TTTv2 Phi-3.5-mini transformer graph; precision recipes and P150 configs |
| `attention.py` | `Phi35MiniAttention`: `Attention1D` with interleaved prefill weights, DRAM-sharded decode weights and a model-owned decode path |
| `mlp.py` | `Phi35MiniMLP`: `MLP1D` with the same weight split, a 128-row prefill path with SiLU fused into W1, and a model-owned decode path |
| `executor.py` | Directly compose one execution lane and own its runtime resources |
| `generator.py` | Build lanes, configure `VLLMAdapter`, and expose serving dispatch |

## Tensor-module composition

`model.py` composes:

- `Embedding1D`
- `RotarySetup1D`
- `RMSNorm1D`
- `Attention1D` (through `Phi35MiniAttention`)
- `MLP1D` (through `Phi35MiniMLP`)
- `LMHead1D`
- optional `Sampling1D`
- common TT collective helpers

Phi-specific provider mapping, architecture values, precision, and tuning
remain model-owned.

## P150 performance choices

- Decode matmuls use the Blackhole multi-core DRAM-sharded pipeline (2-3 workers per DRAM bank); the QKV
  output is sharded one head per core so the fused-QK head split takes it without an interleaved copy.
- Decode keeps the residual stream L1 width-sharded on the norms' 32-core grid.
- Rope cos/sin, the rope transformation matrix and the fused-QK head split share one 8-wide core grid.
- Prefill reads DRAM-interleaved copies of the projection weights; the 128-row bucket uses 1D-multicast
  matmuls and a sharded RMSNorm. Decode reads DRAM width-sharded copies (both copies live in DRAM).
- Performance precision: BFP4 + LoFi for W1/W3, BFP8 for W2 (BFP4 W2 scored below the 89% top-1 target),
  BFP8 attention weights, KV cache and LM head.

## Direct executor composition

`Phi35MiniExecutor` directly constructs:

```text
Phi35MiniTransformer
├── PagedKVCacheManager
├── OutputReader
├── PrefillRuntime
├── DecodeRuntime
├── ProgramCompiler
├── EagerExecutor
├── optional TraceCompiler
├── optional TracedExecutor
└── WarmupCoordinator
```

This direct construction path is intentionally retained as an example for new
models whose orchestration does not belong in the shared Llama or Qwen family
composition. It reuses focused runtime modules without adding a protocol,
profile, or executor subclass hierarchy.

The executor owns the physical KV cache, compile/trace artifacts, output
leases, sampling resources, and cleanup. Cleanup is terminal, ordered,
retryable, and idempotent.

## vLLM and data parallelism

`Phi35MiniGenerator` builds one executor per lane. DP1 uses the executor directly;
larger DP configurations use `LaneGroupExecutor`. `VLLMAdapter` performs
server-boundary normalization and KV-cache validation.

## Tests

- `models/common/tests/demos/phi35_mini/demo.py` (`MESH_DEVICE=P150`): token-accuracy, batch-1, batch-32,
  batch-32-ci, eval-32.
