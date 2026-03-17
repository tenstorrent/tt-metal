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
├── runner.py                   # HybridGlm4Runner: top-level orchestrator
├── tests/
│   └── test_hybrid_modules.py  # Unit + integration test suite
└── README.md
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

## Usage

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
pytest models/demos/glm4_moe_lite_hybrid/tests/ -v

# Hardware tests
TT_ENABLE_HW_TESTS=1 pytest models/demos/glm4_moe_lite_hybrid/tests/ -v

# Full model tests (requires snapshot)
TT_ENABLE_HW_TESTS=1 TT_ENABLE_LARGE_MODEL_TESTS=1 \
    pytest models/demos/glm4_moe_lite_hybrid/tests/ -v
```
