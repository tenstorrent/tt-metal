# GLM-4.7-Flash Hybrid Implementation

A hybrid GLM-4.7-Flash implementation combining **tt-symbiote's modular framework** with **agentic's production optimizations**.

## Architecture

```
glm4_moe_lite_hybrid/
├── core/                       # Framework layer (from tt-symbiote)
│   ├── module.py               # TTNNModule base class with weight lifecycle
│   ├── module_replacement.py   # HuggingFace module replacement utilities
│   ├── config.py               # Glm4MoeLiteHParams model config
│   └── runtime_config.py       # Glm4RuntimeConfig (~30 tunable env vars)
├── modules/                    # TTNN modules (agentic optimizations)
│   ├── attention.py            # HybridGlm4MLA: 3-phase MLA with fused KV branch
│   ├── kvpe_cache.py           # CompressedKVPECache: [blocks,1,block_size,576] BF8
│   ├── moe.py                  # Router + 4 expert paths + shared MoE MLP
│   ├── linear_helpers.py       # 1D prog config, DRAM-sharded matmul, TP helpers
│   ├── layer_weights.py        # Weight conversion: HF -> TT layouts
│   ├── decode_trace.py         # Batch-bucketed decode trace state
│   └── mtp.py                  # Multi-Token Prediction (layer 47)
├── scripts/
│   └── run_hybrid_e2e.py       # End-to-end generation script
├── runner.py                   # HybridGlm4Runner: top-level orchestrator
├── tests/
│   ├── test_hybrid_modules.py  # Unit + integration test suite
│   ├── benchmark_single_device.py   # N150 per-layer benchmark
│   └── benchmark_tp_communication.py # T3K TP reduce benchmark
├── README.md
├── PERF_TESTING_GUIDE.md       # Full performance testing guide
├── BENCHMARK_RESULTS.md        # N150 benchmark results
├── HYBRID_ADVANTAGES.md        # Advantages over both originals
└── TECHNICAL_DEEP_DIVE.md      # CCL, MoE, and TP technical details
```

## Key Optimizations

| Optimization | Source | Expected Impact |
|---|---|---|
| Compressed KVPE cache (576-dim BF8) | agentic | ~2x KV memory reduction |
| Fused KV branch kernel (GLMKVCacheBranch) | agentic | ~30-40% attention latency (batch=1) |
| DRAM-sharded matmuls | agentic | ~20-30% decode latency |
| 1D multicast program config | agentic | ~10-15% decode matmul |
| Sparse MoE (block_size=32) | agentic | Baseline expert perf |
| Fused persistent MoE decode | agentic | ~15-20% MoE latency |
| Decode trace batching | agentic | ~2-3x throughput (batched serving) |
| MTP speculative decoding | agentic | ~1.5-2x effective throughput |
| HF module replacement | tt-symbiote | Drop-in model loading |
| TTNNModule weight lifecycle | tt-symbiote | Clean weight management |
| Distributed RMSNorm | tt-symbiote | TP norm efficiency |
| reduce_scatter_minimal_async | tt-symbiote | ~8% TP linear improvement |

## End-to-End Generation

The primary way to test the hybrid implementation end-to-end:

### Single chip (N150)

```bash
cd /home/ubuntu/agent/agentic/tt-metal

python3 models/demos/glm4_moe_lite_hybrid/scripts/run_hybrid_e2e.py \
  --mesh-cols 1 \
  --prompt "Explain quantum computing in simple terms." \
  --max-new-tokens 32 \
  --kv-cache-dtype bf8
```

### T3K (8 chips)

```bash
python3 models/demos/glm4_moe_lite_hybrid/scripts/run_hybrid_e2e.py \
  --mesh-cols 8 \
  --prompt "Explain quantum computing in simple terms." \
  --max-new-tokens 64 \
  --kv-cache-dtype bf8
```

### T3K with all optimizations

```bash
GLM4_MOE_LITE_DRAM_SHARDED_WEIGHTS=1 \
GLM4_MOE_LITE_FUSED_KV_BRANCH=1 \
GLM4_MOE_LITE_FUSED_MOE=1 \
GLM4_MOE_LITE_TP=1 \
python3 models/demos/glm4_moe_lite_hybrid/scripts/run_hybrid_e2e.py \
  --mesh-cols 8 \
  --prompt "Explain quantum computing in simple terms." \
  --max-new-tokens 64 \
  --kv-cache-dtype bf8
```

### CLI Arguments

| Argument | Default | Description |
|---|---|---|
| `--prompt` | "Explain quantum computing..." | Input text |
| `--max-new-tokens` | 32 | Tokens to generate |
| `--mesh-cols` | 1 | Number of devices (1=N150, 8=T3K) |
| `--kv-cache-dtype` | bf8 | KV cache type: `bf8` (2x savings) or `bf16` |
| `--block-size` | 64 | Paged cache block size |
| `--device-ids` | auto | Specific device IDs or `auto` |

### Output

The script reports:

```
  prefill_s=0.847  decode_tok_s=0.0312  tok_s=32.05

  --- Per-token decode latency (ms) ---
    first token:      45.2 ms
    subsequent:   mean=  31.2  min=  28.9  max=  38.7
```

## Programmatic Usage

### Quick Start

```python
from models.demos.glm4_moe_lite_hybrid.runner import HybridGlm4Runner, HybridRunnerConfig

runner = HybridGlm4Runner.from_pretrained(
    "zai-org/GLM-4.7-Flash",
    device=mesh_device,
    runner_config=HybridRunnerConfig(
        snapshot_dir="/path/to/snapshot",
        enable_moe=True,
    ),
)
runner.load_weights()
runner.init_kvpe_cache(batch_size=1)
runner.init_moe_runtime()
```

### Module Replacement (HuggingFace Integration)

```python
from transformers import AutoModelForCausalLM
from models.demos.glm4_moe_lite_hybrid.core.module_replacement import register_module_replacement
from models.demos.glm4_moe_lite_hybrid.modules.attention import HybridGlm4MLA

model = AutoModelForCausalLM.from_pretrained("zai-org/GLM-4.7-Flash")
replaced = register_module_replacement(model, {
    model.model.layers[0].self_attn.__class__: HybridGlm4MLA,
})
```

## Runtime Configuration

All knobs are controlled via environment variables (parsed once at init):

```bash
# Precision
GLM4_MOE_LITE_MLP_FIDELITY=lofi        # MLP math fidelity
GLM4_MOE_LITE_MLA_FIDELITY=hifi4       # MLA math fidelity

# Memory
GLM4_MOE_LITE_DRAM_SHARDED_WEIGHTS=1   # DRAM-sharded weight storage
GLM4_MOE_LITE_DECODE_L1_ACT=1          # L1 activation memory

# MoE
GLM4_MOE_LITE_MOE_EXPERTS_IMPL=sparse  # sparse|dense_decode|dense_prefill
GLM4_MOE_LITE_FUSED_MOE=1              # Fused persistent MoE decode

# TP
GLM4_MOE_LITE_TP=1                     # Enable tensor parallelism

# Attention
GLM4_MOE_LITE_MLA_SHARD_Q=1            # Q sharding for FlashMLA
GLM4_MOE_LITE_HEAD_PARALLEL_KVB2=1     # Head-parallel kv_b2 path
GLM4_MOE_LITE_FUSED_KV_BRANCH=1        # Fused KV branch kernel
```

## Testing

```bash
# Unit tests (no hardware needed)
python3 -m pytest models/demos/glm4_moe_lite_hybrid/tests/test_hybrid_modules.py \
  -v --noconftest -k "not TTNNIntegration"

# Per-layer benchmark (N150, requires hardware + snapshot)
python3 models/demos/glm4_moe_lite_hybrid/tests/benchmark_single_device.py

# TP communication benchmark (T3K, requires 8 devices)
python3 models/demos/glm4_moe_lite_hybrid/tests/benchmark_tp_communication.py --mesh-cols 8

# End-to-end generation (N150)
python3 models/demos/glm4_moe_lite_hybrid/scripts/run_hybrid_e2e.py \
  --mesh-cols 1 --prompt "Hello" --max-new-tokens 16

# End-to-end generation (T3K)
python3 models/demos/glm4_moe_lite_hybrid/scripts/run_hybrid_e2e.py \
  --mesh-cols 8 --prompt "Hello" --max-new-tokens 64 --kv-cache-dtype bf8
```

## Documentation

| Document | Description |
|---|---|
| [PERF_TESTING_GUIDE.md](PERF_TESTING_GUIDE.md) | Full test guide: 4 tiers, runtime knobs, 3-way T3K comparison, single-chip vs T3K |
| [BENCHMARK_RESULTS.md](BENCHMARK_RESULTS.md) | N150 benchmark results with stage breakdowns |
| [HYBRID_ADVANTAGES.md](HYBRID_ADVANTAGES.md) | Why hybrid is better than both originals |
| [TECHNICAL_DEEP_DIVE.md](TECHNICAL_DEEP_DIVE.md) | CCL ops, MoE dispatch, TP communication deep dive |
