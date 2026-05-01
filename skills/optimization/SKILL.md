---
name: performance-optimization
description: Optimize TTNN models for maximum throughput using metal tracing, memory optimization, op fusion, and L1 sharding. Use when optimizing model performance, implementing tracing, fusing ops, or targeting decode latency.
---

# SKILL: Performance Optimization

## Purpose
Optimize TTNN models for maximum throughput using tracing, memory optimization, and op fusion.

## Metal Trace Implementation

### Overview
Tracing records dispatch commands to replay without host overhead. Provides 5-25x speedup for host-bound models.

### Key APIs
```python
# Capture trace
tid = ttnn.begin_trace_capture(device, cq_id=0)
# ... run operations ...
ttnn.end_trace_capture(device, tid, cq_id=0)

# Execute trace (fast replay)
ttnn.execute_trace(device, tid, cq_id=0, blocking=False)
```

### Critical: Warmup Before Trace Capture
**ALL ops must be compiled before trace capture**. Run a warmup pass first:

```python
# 1. WARMUP: Compile all operations
warmup_input = ttnn.from_torch(torch.randn(...), device=device, layout=ttnn.TILE_LAYOUT)
_ = model.forward(warmup_input, kv_caches=kv_caches)  # Include KV cache ops!

# 2. CAPTURE: Record the trace
input_tensor = ttnn.allocate_tensor_on_device(spec, device)
ttnn.copy_host_to_device_tensor(host_input, input_tensor)

tid = ttnn.begin_trace_capture(device, cq_id=0)
output = model.forward(input_tensor, kv_caches=kv_caches)
ttnn.end_trace_capture(device, tid, cq_id=0)

# 3. EXECUTE: Fast replay
for _ in range(100):
    ttnn.copy_host_to_device_tensor(new_host_input, input_tensor)
    ttnn.execute_trace(device, tid, cq_id=0, blocking=False)
```

### Host Tensor Pattern for Trace-Safe Updates
Traces require fixed tensor addresses. Use host tensors + copy:

```python
# Pre-allocate device tensor
device_tensor = ttnn.allocate_tensor_on_device(tensor_spec, device)

# Create host tensor (device=None)
host_tensor = ttnn.from_torch(data, device=None, layout=ttnn.TILE_LAYOUT)

# Copy to fixed device address (trace-safe)
ttnn.copy_host_to_device_tensor(host_tensor, device_tensor, cq_id=0)
```

### Trace with DRAM Input (Recommended)
```python
# Allocate persistent input in DRAM
input_dram = ttnn.allocate_tensor_on_device(spec, device, ttnn.DRAM_MEMORY_CONFIG)

# Warmup
ttnn.copy_host_to_device_tensor(host_tensor, input_dram)
input_l1 = ttnn.to_memory_config(input_dram, ttnn.L1_MEMORY_CONFIG)
output = model(input_l1)

# Capture trace
ttnn.copy_host_to_device_tensor(host_tensor, input_dram)
tid = ttnn.begin_trace_capture(device, cq_id=0)
input_l1 = ttnn.to_memory_config(input_dram, ttnn.L1_MEMORY_CONFIG)
output = model(input_l1)
ttnn.end_trace_capture(device, tid, cq_id=0)

# Execute
for batch in data:
    ttnn.copy_host_to_device_tensor(batch, input_dram)
    ttnn.execute_trace(device, tid, blocking=False)
```

## Vision Tracing (VLM Models)

### Key Patterns
1. **Don't deallocate input tensor** in vision blocks (following tt_transformers)
2. **Don't clone hidden states** - TTNN ops create new tensors
3. **Use ttnn.embedding for gather** instead of CPU indexing

```python
# vision_backbone.py - forward_ttnn for tracing
def forward_ttnn(self, x, positions):
    """Trace-safe forward using ttnn.embedding for gather."""
    # Use ttnn.embedding instead of CPU gather
    selected = ttnn.embedding(positions, x)
    return self.transformer.forward(selected)
```

### Vision Tracing Example
```python
# Capture vision trace
vision_input = ttnn.allocate_tensor_on_device(vision_spec, device)
ttnn.copy_host_to_device_tensor(host_image, vision_input)

tid_vision = ttnn.begin_trace_capture(device, cq_id=0)
vision_output = vision_model.forward_ttnn(vision_input)
ttnn.end_trace_capture(device, tid_vision, cq_id=0)

# Execute (25x faster)
ttnn.execute_trace(device, tid_vision, blocking=False)
```

### CRITICAL: Decode Traces Do NOT Scale to Variable S

Traces are safe ONLY when ALL requests share the same S. For servers processing inputs
of different lengths, the SDPA program config is baked at capture time and gives wrong
results for any S ≠ S_capture.

```python
# SAFE: offline batch inference, all requests padded to fixed S
self._capture_trace(S=max_seq_len)   # all requests use this S

# UNSAFE: server where S varies per request (video length, prompt length)
self._capture_trace(S=first_request_S)  # WRONG for subsequent requests

# FIX for variable-S servers: no trace, use forward_decode_step each call
def decode_forward(self, tokens, start_pos, **kwargs):
    logits = self.model.forward_decode_step(int(tokens[0,0].item()),
                                             int(start_pos[0].item()))
    return logits.squeeze(0).unsqueeze(0)
```

### T3K TP MLP: Use Trace-Safe Async Reduce-Scatter + All-Gather

`ttnn.all_reduce` is NOT replay-safe in decode traces. Use the async RS+AG pattern
from `tt_transformers/tt/ccl.py` for trace-safe MLP all-reduce:

```python
from models.tt_transformers.tt.ccl import tt_all_reduce

# In forward_decode (inside trace):
out = ttnn.linear(x, self.w2, ...)    # row-parallel partial sum
out = tt_all_reduce(out, mesh_device, ...)   # trace-safe async RS+AG
```

Note: `tt_all_reduce(cluster_axis=1)` is a no-op on T3K; this pattern uses
`cluster_axis=0` for the ring, not 1.

### VLM Profiling: ViT AllBroadcast Is the Expected Dominant Bottleneck

When profiling a VLM ViT encoder on T3K with TP8 layout, the all_reduce ops
(one after wo, one after w2 per block) typically dominate device time.

Profile workflow:
```bash
python -m tracy -p -v -r -n vit_encoder \
    -o generated/profiler/block_reports/vit_encoder \
    models/demos/{model}/tests/profile_single_block.py --block vit_encoder
```

Expected pattern: AllBroadcast CCL ops consume >50% of ViT device time.
Root cause: n_blocks × 2 synchronous `ttnn.all_reduce` calls = many CCL barriers.

Key constraints:
- Cannot eliminate with TP (all_reduce required for partial-sum correctness)
- Replicated weights OOM for large n_frames (activations scale as `[n_frames, n_heads_full, S, head_dim]`)
- `num_links=2` hangs on T3K; use `num_links=1` only
- Mitigation: async all_reduce overlapping with next block's compute (complex; future work)

## Op Fusion

### QKV Fusion
Fuse Q, K, V projections into single matmul:

```python
# Instead of 3 separate matmuls:
# q = ttnn.linear(x, wq)
# k = ttnn.linear(x, wk)
# v = ttnn.linear(x, wv)

# Fuse weights
wqkv = torch.cat([wq, wk, wv], dim=0)
self.wqkv = ttnn.from_torch(wqkv, device=device, layout=ttnn.TILE_LAYOUT)

# Single fused matmul
def forward(self, x):
    qkv = ttnn.linear(x, self.wqkv)
    q, k, v = ttnn.split(qkv, 3, dim=-1)
    return q, k, v
```

### Gate-Up Fusion (SwiGLU MLP)
```python
# Fuse gate and up projections
w_gate_up = torch.cat([w_gate, w_up], dim=0)
self.w_gate_up = ttnn.from_torch(w_gate_up, device=device, layout=ttnn.TILE_LAYOUT)

def forward(self, x):
    gate_up = ttnn.linear(x, self.w_gate_up)
    gate, up = ttnn.split(gate_up, 2, dim=-1)
    return ttnn.linear(ttnn.silu(gate) * up, self.w_down)
```

## L1 Memory Optimization

### Target L1 for Hot Paths
- **Decode activations**: Single token, fits in L1
- **Vision blocks**: Fixed sequence length
- **Attention scores**: Keep in L1 during computation

```python
# Decode config - use L1
decode_mem_config = ttnn.L1_MEMORY_CONFIG

# Move to L1 for hot path
x_l1 = ttnn.to_memory_config(x, ttnn.L1_MEMORY_CONFIG)
output = attention(x_l1)  # Fast L1 access
```

### When to Use DRAM
- Large KV caches that don't fit in L1
- Weight storage (accessed once per forward)
- Prefill with long sequences

## Performance Targets

### LLM Decode (Reference: Molmo2-8B on T3K)
| Metric | Target | Notes |
|--------|--------|-------|
| Decode (traced) | ~28ms/token | 35.6 tok/s |
| Decode (no trace) | ~180ms/token | 5.5 tok/s |
| Tracing speedup | 6-7x | |

### Vision (Reference: Molmo2-8B)
| Metric | Target | Notes |
|--------|--------|-------|
| Vision (traced) | ~86ms | |
| Vision (no trace) | ~2150ms | |
| Tracing speedup | 25x | |

### Time to First Token (TTFT)
| Metric | Target |
|--------|--------|
| Prefill compile | ~1400ms (first run only) |
| TTFT (after compile) | ~85ms |

## Common Tracing Pitfalls

### "Writes not supported during trace capture"
**Cause**: `ttnn.embedding` or other write ops after warmup
**Solution**: Include ALL ops in warmup, use unified trace carefully

### Tensor Address Mismatch
**Cause**: Tensor deallocated and reallocated at different address
**Solution**: Use pre-allocated tensors, don't deallocate during trace

### KV Cache Not Compiled
**Cause**: Warmup didn't include `fill_cache` ops
**Solution**: Pass `kv_caches=self.kv_caches` during warmup

## Demo Integration

```python
# Full tracing demo pattern
class Generator:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.trace_id = None

    def setup_trace(self):
        # Warmup
        self.model.forward(warmup_input, kv_caches=self.kv_caches)

        # Capture
        self.input_tensor = ttnn.allocate_tensor_on_device(spec, self.device)
        self.trace_id = ttnn.begin_trace_capture(self.device, cq_id=0)
        self.output_tensor = self.model.forward(self.input_tensor, kv_caches=self.kv_caches)
        ttnn.end_trace_capture(self.device, self.trace_id, cq_id=0)

    def generate(self, tokens):
        for i, token in enumerate(tokens):
            host_input = prepare_input(token)
            ttnn.copy_host_to_device_tensor(host_input, self.input_tensor)
            ttnn.execute_trace(self.device, self.trace_id, blocking=False)
            # Read output...
```

## Optimization Checklist

- [ ] Implement warmup before trace capture
- [ ] Pre-allocate all tensors used in trace
- [ ] Use host tensor + copy pattern for inputs
- [ ] Fuse QKV projections
- [ ] Fuse gate/up projections in MLP
- [ ] Target L1 for decode activations
- [ ] Store weights in DRAM
- [ ] Verify tracing speedup matches targets
- [ ] Compare with similar models in ARCHITECTURE.md
