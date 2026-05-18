---
name: ttnn-implementation
description: Implement model blocks in TTNN achieving PCC greater than 0.99 against PyTorch reference. Use when writing TTNN code, loading weights, implementing attention or MLP in TTNN, or verifying PCC.
---

# SKILL: TTNN Implementation

## Purpose
Implement model blocks in TTNN, achieving PCC > 0.99 against PyTorch reference.

## CRITICAL Prerequisites

**Before implementing ANY TTNN code:**

1. **Verify Reference is COMPLETE and WORKING**
   - ALL components from ARCHITECTURE.md have reference implementations
   - Reference produces correct END OUTPUT (not just runs without errors)
   - For TTS: Reference audio sounds correct when played
   - For LLM: Reference text makes sense

2. **Never implement TTNN against an unverified reference**
   - High PCC against a broken reference is meaningless
   - The reference MUST produce correct functional output first

## Step-by-Step Process

### 1. Directory Structure
```
models/demos/{model_name}/tt/
├── __init__.py
├── model_config.py       # Configuration and memory settings
├── attention.py          # Attention implementation
├── mlp.py                # MLP/FFN implementation
├── model.py              # Full model assembly
└── generator.py          # Inference with tracing support
```

### 2. Weight Loading Pattern
Use `ttnn.from_torch` with proper dtype and layout:

```python
import torch
import ttnn

def load_weights(state_dict, device, dtype=ttnn.bfloat16):
    """Load weights with proper TTNN conversion."""

    # Standard weight loading
    weight = state_dict["layer.weight"]
    weight_ttnn = ttnn.from_torch(
        weight,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,  # Required for matmul
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,  # Store weights in DRAM
    )
    return weight_ttnn

# For fused QKV weights (performance optimization)
def load_fused_qkv(state_dict, device, dtype=ttnn.bfloat16):
    q = state_dict["q_proj.weight"]
    k = state_dict["k_proj.weight"]
    v = state_dict["v_proj.weight"]
    qkv = torch.cat([q, k, v], dim=0)
    return ttnn.from_torch(qkv, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
```

### 3. Mesh Mapper for Weights (MULTI-CHIP)

**CRITICAL — DO NOT default to `ReplicateTensorToMesh` for big weights.** Replicating
a matmul weight on every chip wastes (n_dev − 1)/n_dev of DRAM and silently OOMs at
full layer count. The 4-layer test passes, the 64-layer model fails — discovered too
late. Decide the mesh_mapper at block-design time, not bring-up time.

| Tensor | mesh_mapper | Why |
|--------|-------------|-----|
| **MLP gate/up/down** | `ShardTensor2dMesh(dims=(None,-1) or (0,None))` | 70-90% of model DRAM. MUST be sharded — usually column-parallel gate/up + row-parallel down with `reduce_scatter`+`all_gather`. |
| **Attention QKV (fused)** | `ShardTensor2dMesh(dims=(None,-1))` | Column-parallel on heads → each chip owns `n_heads/n_dev` heads. |
| **Attention output (Wo)** | `ShardTensor2dMesh(dims=(None,-1))` | Row-parallel — followed by `reduce_scatter` on cluster_axis=1. |
| **LM head** | `ShardTensor2dMesh(dims=(None,-1))` | Vocab-parallel — replicated for 32K vocab OK, **MUST shard for >128K vocab** (Qwen3 248K replicated = 2.4 GB/chip just for the head). |
| **Full-attn WO / MLP FF2** | `ShardTensor2dMesh` row-parallel, `reduce_scatter` on the **same `cluster_axis` as the canonical Galaxy port** | The ring topology (4-way vs 8-way) sets a hard floor on per-call CCL latency. Mirroring the canonical port costs nothing at write time and 3-5 ms/layer at audit time. |
| **Norm weights, QK-norm** | `ReplicateTensorToMesh` | Small 1-D tensors (`[dim]`). |
| **Embedding table** | `ReplicateTensorToMesh` | Sparse lookup, OK to replicate if vocab small. |
| **RoPE cos/sin tables** | `ReplicateTensorToMesh` | Small, read by every chip. |
| **KV cache** | Sharded per `n_kv_heads/n_dev` | Grows with sequence — never replicate. |

**Precedent to copy (Galaxy):** `models/demos/llama3_70b_galaxy/tt/llama_mlp.py` uses
`ShardTensor2dMesh` + `bfloat8_b`/`bfloat4_b` quantization + `reduce_scatter`/`all_gather`.
The DeltaNet block in `models/demos/qwen3_6_galaxy/tt/qwen36_deltanet.py` uses
`ShardTensor2dMesh(dims=(0,None))` for the row-parallel split — also follow this.

**Anti-pattern (caused 64-layer OOM):** `models/demos/qwen3_6_galaxy/tt/llama_mlp.py`
and `llama_attention.py` (pre-fix) used `ReplicateTensorToMesh` for every linear.
534 MB/layer/chip × 64 layers = 34 GB/chip > 30 GB budget → DRAM exhaustion at
layer 52 of model loading. Passing PCC at 1-4 layers hid the real defect.

### 3a. Per-Block Memory Budget Check (RUN AT BLOCK-WRITE TIME)

After writing a block's `__init__` and before running its PCC test, print the
weight footprint as a self-check:

```python
def _print_footprint(self, name: str):
    """Sum bytes uploaded by this block per device."""
    import sys
    total = 0
    for attr in dir(self):
        t = getattr(self, attr, None)
        if hasattr(t, "shape") and hasattr(t, "dtype") and hasattr(t, "memory_config"):
            # ttnn Tensor — bytes per device
            n = 1
            for d in t.shape: n *= d
            bpe = {ttnn.bfloat16: 2, ttnn.bfloat8_b: 1.06, ttnn.bfloat4_b: 0.56,
                   ttnn.float32: 4, ttnn.uint32: 4}.get(t.dtype, 2)
            # If replicated: full size. If sharded across N chips: /N.
            replicated = ... # inspect mesh_mapper or set explicitly
            per_dev = n * bpe / (1 if replicated else self.args.num_devices)
            total += per_dev
    print(f"[{name}] per-device weight DRAM: {total/1e6:.0f} MB")
```

**Acceptance rule:** `(per_block_MB × n_layers) < 0.6 × dram_per_chip_MB`. If not,
the block is replicating something it shouldn't. Stop and re-plan before writing
the next block.

### 3b. Sharding Parity Against Canonical Galaxy Reference (RUN BEFORE WRITING THE BLOCK)

**Two references, two roles.** When porting a model to Galaxy, you often have:
- A **v1 implementation** of the same model (correctness-locked; passes PCC > 0.99).
- A **canonical Galaxy port** (e.g. `models/demos/llama3_70b_galaxy/`, `models/demos/olmo_galaxy/`)
  — perf-locked; sharding / CCL choices are tuned for Galaxy ring topology.

**Rule:** the v1 is your *math oracle* (gives you `q_norm`, `rope_dim`, gate placement,
HF key map). The canonical port is your *structural oracle* (gives you cluster_axis,
num_links, output memcfg, head-sharding direction, residual-stream dtype).

**For every block, before writing it**, fill out this 6-row table against both
references. If the row differs between v1 and the Galaxy port, **default to the
Galaxy port** unless the math forces otherwise (and document why in
`MESH_SHARDING_PLAN.md`):

| structural choice            | v1 | canonical Galaxy port | this block uses | reason if differs |
|---|---|---|---|---|
| QKV `cluster_axis`           | ? | ? | ? | |
| QKV head-sharding axis (rows vs cols) | ? | ? | ? | |
| WO (output proj) `cluster_axis` for reduce_scatter | ? | ? | ? | |
| MLP `cluster_axis` (FF1/FF2/FF3) | ? | ? | ? | |
| `num_links` for CCL ops (1 vs 2) | ? | ? | ? | |
| Residual-stream dtype across N layers | ? | ? | ? | |

**Anti-pattern (precedent: V2-4 in qwen3_6_galaxy_v2 bringup):** A `is_<model>`
branch agent was told "mirror v1's working `llama_attention.py`" and inherited
v1's `cluster_axis=1` (4-way ring) for full-attention WO reduce_scatter. The
canonical llama3_70b_galaxy uses `cluster_axis=0` (8-way ring). The 4-way ring
takes ~3.8× longer per call. The miss survived 40+ downstream commits before
tracy revealed it; the fix is a foundational refactor at that point. **Catch
this at table-fill time, not at tracy-audit time.**

### 4. Memory Configuration (Activations)

| Config | Use Case | When to Use |
|--------|----------|-------------|
| `ttnn.DRAM_MEMORY_CONFIG` | Weight storage | Default for large tensors |
| `ttnn.L1_MEMORY_CONFIG` | Activations in hot path | Decode, vision blocks |
| `ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG` | Sharded activations | Large batch, parallelism |

```python
# L1 for decode activations (hot path)
decode_mem_config = ttnn.L1_MEMORY_CONFIG

# Block sharded for distributed compute
sharded_config = ttnn.create_sharded_memory_config(
    shape=(batch_size, hidden_size),
    core_grid=ttnn.CoreGrid(y=8, x=8),
    strategy=ttnn.ShardStrategy.BLOCK,
)
```

### 4. Block Implementation Pattern

```python
from models.common.lightweightmodule import LightweightModule

class TtAttention(LightweightModule):
    def __init__(self, device, state_dict, config, layer_num, dtype=ttnn.bfloat16):
        super().__init__()
        self.device = device
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads

        # Load weights
        layer_prefix = f"model.layers.{layer_num}.self_attn."
        self.wq = ttnn.from_torch(
            state_dict[layer_prefix + "q_proj.weight"],
            dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device
        )
        # ... load k, v, o weights

    def forward(self, x, rot_mat=None, attention_mask=None):
        # QKV projection
        q = ttnn.linear(x, self.wq)
        k = ttnn.linear(x, self.wk)
        v = ttnn.linear(x, self.wv)

        # Apply RoPE
        if rot_mat is not None:
            q = ttnn.experimental.rotary_embedding_llama(q, rot_mat, ...)
            k = ttnn.experimental.rotary_embedding_llama(k, rot_mat, ...)

        # Attention
        attn = ttnn.transformer.scaled_dot_product_attention(q, k, v, is_causal=True)

        # Output projection
        return ttnn.linear(attn, self.wo)
```

### 5. KV-Cache Implementation

```python
class TtAttentionWithKVCache(LightweightModule):
    def __init__(self, device, config, layer_num):
        super().__init__()
        # Pre-allocate KV cache
        self.k_cache = ttnn.zeros(
            (config.max_batch_size, config.n_kv_heads, config.max_seq_len, config.head_dim),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )
        self.v_cache = ttnn.zeros(...)

    def forward(self, x, start_pos, rot_mat):
        # Project Q, K, V
        q, k, v = self.qkv_projection(x)

        # Update cache at current position
        self.k_cache = ttnn.fill_cache(self.k_cache, k, start_pos)
        self.v_cache = ttnn.fill_cache(self.v_cache, v, start_pos)

        # Attend over full cache
        keys = self.k_cache[:, :, :start_pos + 1, :]
        values = self.v_cache[:, :, :start_pos + 1, :]

        return self.attention(q, keys, values)
```

### 5a. Paged KV Cache (REQUIRED for vLLM / tt-inference-server)

vLLM and tt-inference-server require a **paged** KV cache: the cache is allocated as
fixed-size blocks indexed by a `page_table`, allowing arbitrary sequence reorderings.
Without paged attention the model cannot be served behind vLLM.

**Cache layout** (per layer, per device):
```
[max_num_blocks, n_kv_per_dev, block_size, head_dim]
```
- `block_size`: tokens per page block (typical: 64). Must be a multiple of 32.
- `max_num_blocks`: pool size. Sized as `max_batch * ceil(max_seq_len / block_size)`.
- `page_table`: int32 tensor `[B, max_blocks_per_seq]` mapping logical block id → physical block id in the pool.

```python
# Allocation at __init__ — one buffer per layer:
self.k_cache = ttnn.from_torch(
    torch.zeros(max_num_blocks, n_kv_per_dev, block_size, head_dim, dtype=torch.bfloat16),
    device=mesh_device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
    mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(None, 1), mesh_shape=cluster_shape),
)
# Identical for v_cache.
```

**Prefill** uses `paged_fill_cache` (writes user_id's blocks into the pool at indices
specified by the page_table row for that user):

```python
ttnn.experimental.paged_fill_cache(
    self.k_cache, k_tile, page_table_tt, user_id=user_id, batch_offset=0,
)
ttnn.experimental.paged_fill_cache(
    self.v_cache, v_tile, page_table_tt, user_id=user_id, batch_offset=0,
)
```

**Decode** uses `paged_update_cache` (writes ONE token at `current_pos` per user) and
`paged_scaled_dot_product_attention_decode`:

```python
# Update — HEIGHT_SHARDED on a single core, current_pos as int32[B] device tensor
ttnn.experimental.paged_update_cache(
    self.k_cache, k_decode, update_idxs_tensor=cur_pos_tt, page_table=page_table_tt,
    batch_offset=0,
)

# Attend over paged cache
attn = ttnn.experimental.paged_scaled_dot_product_attention_decode(
    q_decode, self.k_cache, self.v_cache,
    cur_pos_tensor=cur_pos_tt,
    page_table=page_table_tt,
    scale=1.0 / math.sqrt(head_dim),
    program_config=ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=(1, 1),  # 1-core for HEIGHT_SHARDED
        q_chunk_size=32, k_chunk_size=block_size,
    ),
    compute_kernel_config=hifi4_kernel,
)
```

**Why both paged_update_cache AND paged_SDPA**: paged_update_cache writes the NEW K/V
into the block pool *in place* (trace-safe). paged_SDPA reads from the pool through the
page_table, so K/V layout in the pool can be sparse / reordered without breaking the
attention pattern. Both ops are required — using regular `update_cache` + `SDPA_decode`
on a paged buffer silently writes/reads the wrong indices.

**Block-size constraints**:
- `block_size % 32 == 0` (TILE_LAYOUT)
- `block_size >= k_chunk_size` in SDPA program config — undersize causes a partial-tile hang.

**Test the paged path with a parity test against the non-paged path** at the same
sequence positions — PCC should be > 0.999 (same math, just different cache layout).
See `models/demos/qwen3_6_galaxy/tests/test_paged_attention.py` for the template.

### 5b. Trace-Safe Forward (REQUIRED for production perf)

`ttnn.begin_trace_capture` / `ttnn.execute_trace` records a graph of device ops and replays
it on subsequent calls — eliminating the ~hundreds of ms/step Python dispatch overhead
for a 64-layer model. Trace gives 10-50× decode speedup. But trace ONLY captures device
ops — host roundtrips inside the captured region crash with:

```
TT_FATAL: Writes are not supported during trace capture. trace id: 0
```

**Rules for a trace-safe forward:**

1. **NO `ttnn.from_torch` inside the captured region.** Every per-step input must live in
   a *preallocated* device buffer. Refresh contents per step with:
   ```python
   ttnn.copy_host_to_device_tensor(host_tensor_cpu, device_buffer_tt)
   ```
   This copies through a pre-existing buffer (trace-safe) instead of allocating a new one.

2. **NO `ttnn.to_torch` inside the captured region.** All host gathers (logits readback,
   sampling) happen OUTSIDE `execute_trace`. The forward returns a `ttnn.Tensor`; the
   caller does `ttnn.to_torch(...)` after `ttnn.execute_trace(...)` returns.

3. **NO per-step `ttnn.from_torch` for masks / position indices.** Either:
   - Build a static device tensor once at `__init__` (e.g. full causal mask) and `ttnn.slice`
     inside the forward (slice IS trace-safe), OR
   - Make `current_pos` itself a preallocated `[B]` int32 device tensor refreshed via
     `copy_host_to_device_tensor`; derive `update_idxs` / `cur_pos_sdpa` on-device.

4. **No host roundtrip for residual reshaping.** Helpers that do `to_torch → torch op →
   from_torch` (e.g. "shard replicated→sharded-across-cols by going through CPU") must be
   replaced by on-device CCL (`ttnn.all_gather`, `ttnn.reduce_scatter`, `ttnn.experimental.all_gather_async`)
   or by keeping the residual stream in its target layout throughout (sharded across cols
   in the Galaxy convention — never gather to replicated mid-layer).

5. **In-place state updates only.** Any persistent state across decode steps (KV cache,
   DeltaNet recurrent state, conv1d input state) must be updated in place in a preallocated
   buffer. Use `paged_update_cache` for KV (5a above); for custom state buffers, prefer
   kernels that accept an output-buffer argument and write into it. `ttnn.assign(dst, src)`
   can also work if the underlying op is trace-safe.

**Trace pattern (lazy capture, keyed by shape):**

```python
class Generator:
    def __init__(self, model, mesh_device):
        self.model = model
        self.mesh_device = mesh_device
        self.trace_id_prefill = {}     # (B, T) -> trace_id
        self.trace_inputs_prefill = {} # (B, T) -> tuple of device buffers
        self.trace_output_prefill = {} # (B, T) -> output tensor handle
        # ... similar for decode

    def prefill_forward(self, input_ids, page_table=None, enable_trace=True):
        B, T = input_ids.shape
        key = (B, T)
        if enable_trace and key not in self.trace_id_prefill:
            # Capture once
            host_inputs = self.model.prepare_prefill_inputs_host(input_ids, page_table)
            device_inputs = self.model.allocate_prefill_input_buffers(B, T)
            self._copy_host_to_device(host_inputs, device_inputs)
            # Compile run (warms caches, compiles kernels)
            _ = self.model.ttnn_prefill_forward(*device_inputs)
            ttnn.synchronize_device(self.mesh_device)
            # Capture run with the SAME device buffers
            self._copy_host_to_device(host_inputs, device_inputs)
            trace_id = ttnn.begin_trace_capture(self.mesh_device, cq_id=0)
            out_tt = self.model.ttnn_prefill_forward(*device_inputs)
            ttnn.end_trace_capture(self.mesh_device, trace_id, cq_id=0)
            self.trace_id_prefill[key] = trace_id
            self.trace_inputs_prefill[key] = device_inputs
            self.trace_output_prefill[key] = out_tt
        # Execute: refresh inputs in place, run trace, gather output AFTER
        host_inputs = self.model.prepare_prefill_inputs_host(input_ids, page_table)
        self._copy_host_to_device(host_inputs, self.trace_inputs_prefill[key])
        ttnn.execute_trace(self.mesh_device, self.trace_id_prefill[key], cq_id=0, blocking=False)
        return ttnn.to_torch(self.trace_output_prefill[key], ...)
```

Precedent: `models/demos/llama3_70b_galaxy/tt/generator.py` — `_capture_trace_prefill`
(L854), `_capture_trace_text` (L1153), `_prefill_forward_trace_text` (L977),
`_decode_forward_trace_text` (L1202). The `copy_host_to_device` helper at the top of
that file is the trace-safe input refresh pattern.

**Decode trace and variable S**: a decode trace captured at S₁ produces wrong output at
S₂ ≠ S₁ because the SDPA program config (core grid, tile sizing) is baked at capture
time. For servers with variable-length sequences either capture one trace per S bucket
or fall back to eager decode. See "Decode Trace Does NOT Scale to Variable S" below.

### 6. Prefill vs Decode Modes

```python
class TtModel(LightweightModule):
    def forward_prefill(self, x, start_pos=0):
        """Prefill mode: process full sequence, populate KV cache."""
        # Use DRAM for larger sequences
        for layer in self.layers:
            x = layer.forward_prefill(x, start_pos)
        return x

    def forward_decode(self, x, start_pos):
        """Decode mode: single token, read from KV cache."""
        # Use L1 for speed (single token)
        for layer in self.layers:
            x = layer.forward_decode(x, start_pos)
        return x
```

### 7. PCC Verification

```python
import torch

def verify_pcc(ttnn_output, reference_output, threshold=0.99):
    """Verify Pearson Correlation Coefficient > threshold."""
    ttnn_torch = ttnn.to_torch(ttnn_output).to(torch.float32)
    ref_torch = reference_output.to(torch.float32)

    # Flatten and compute PCC
    pcc = torch.corrcoef(torch.stack([
        ttnn_torch.flatten(),
        ref_torch.flatten()
    ]))[0, 1].item()

    assert pcc > threshold, f"PCC {pcc:.4f} < {threshold}"
    return pcc
```

### 8. Audio Codec Decoder (for TTS models)

TTS models require an audio decoder to convert codec tokens to waveforms:

```python
class AudioCodecDecoder(LightweightModule):
    def __init__(self, device, state_dict, config, dtype=ttnn.bfloat16):
        super().__init__()
        self.device = device

        # Load codebook embeddings (RVQ)
        # Shape: [num_codebooks, codebook_size, embedding_dim]
        self.codebooks = []
        for i in range(config.num_codebooks):
            codebook = ttnn.from_torch(
                state_dict[f"quantizer.layers.{i}.codebook"],
                dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
            )
            self.codebooks.append(codebook)

        # Load Conv1d weights for upsampling
        # Note: Conv1d weights are [out_channels, in_channels, kernel_size]
        self.conv_weights = ttnn.from_torch(
            state_dict["decoder.conv.weight"],
            dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device
        )

    def forward(self, token_ids):
        """
        Args:
            token_ids: [batch, num_codebooks, seq_len] token indices
        Returns:
            audio: [batch, 1, num_samples] waveform
        """
        # Lookup embeddings from each codebook and sum
        embeddings = None
        for i, codebook in enumerate(self.codebooks):
            ids = token_ids[:, i, :]  # [batch, seq_len]
            emb = ttnn.embedding(ids, codebook)  # [batch, seq_len, embed_dim]
            embeddings = emb if embeddings is None else ttnn.add(embeddings, emb)

        # Pass through conv decoder (upsamples to audio rate)
        audio = self.conv_decode(embeddings)
        return audio
```

**Key considerations for audio decoders:**
- Use `ttnn.embedding` for codebook lookup
- Conv1d operations may need custom handling or fallback to PyTorch
- Upsample ratios can be large (e.g., 1920× for 12.5Hz → 24kHz)

## Common Pitfalls

### Inheriting cluster_axis From v1 Instead of the Canonical Galaxy Port

If the v1 implementation has a `cluster_axis` choice that diverges from the
canonical Galaxy port for the same op shape, **the v1 was probably
correctness-locked at that choice, not perf-locked**. A v1 that passes PCC at
ISL=128 with `cluster_axis=1` on a 4-way ring will pass PCC at any ISL with
`cluster_axis=0` on an 8-way ring too — but the latter has half the per-call
latency on Galaxy. Take cluster_axis from the Galaxy port, not v1.

Symptom (tracy): a single per-op call in your model is 2-4× heavier than the
same op in the reference Galaxy port for the same shape. Before reaching for
fusion or custom kernels, **diff cluster_axis and num_links first**.

### bfloat8_b Numerical Overflow
**Symptom**: PCC drops at specific layers, NaN/Inf values
**Cause**: bfloat8_b precision loss causes overflow in decode mode
**Solution**: Use bfloat16 for weights, or store CPU copies and convert on-demand

```python
# Store CPU copy for decode mode
self.weight_cpu = state_dict["weight"].clone()

def forward_decode(self, x):
    # Convert to bfloat16 on-demand
    weight = ttnn.from_torch(
        self.weight_cpu.to(torch.bfloat16),
        device=self.device,
        layout=ttnn.TILE_LAYOUT
    )
    return ttnn.linear(x, weight)
```

### TILE_LAYOUT Requirements
- All matmul inputs must be in `ttnn.TILE_LAYOUT`
- Tile size is 32x32
- Pad dimensions to multiples of 32

### Sharding Mismatches
- Ensure input/output sharding specs match between ops
- Use `ttnn.to_memory_config()` to reshard when needed

### T3K Multi-Chip: tt_all_reduce is a Silent NO-OP

`tt_all_reduce(cluster_axis=1)` from `models.tt_transformers.tt.ccl` silently does nothing
on T3K [1×8] mesh. PCC passes on single-device but is wrong on T3K for any block using it.

```python
# WRONG — silent no-op on T3K:
from models.tt_transformers.tt.ccl import tt_all_reduce
tt_all_reduce(out, cluster_axis=1)

# CORRECT:
out = ttnn.all_reduce(out, cluster_axis=1, num_links=1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
# Note: num_links=2 hangs on T3K — always use num_links=1
```

### KV Cache Dtype: bfloat16 for Vision Models

`bfloat8_b` KV caches cause logit flips for long sequences (S > 2500) in vision models
due to SDPA precision loss accumulating over many K/V entries.

```python
# Vision models — use bfloat16 for KV cache:
ttnn.as_tensor(cache_kv, dtype=ttnn.bfloat16, ...)   # NOT bfloat8_b

# fill_cache without typecast (k is already bfloat16):
ttnn.fill_cache(keys, k, user_id)   # no ttnn.typecast needed
```

### SDPA Chunk Size: Smaller is Safer for Small n_local_heads

`q_chunk_size` in `SDPAProgramConfig` controls tokens per core per pass.
Larger chunks are NOT always faster — they can overflow L1:

| n_local_heads | Recommended q_chunk_size | Risk of larger |
|--------------|--------------------------|----------------|
| ≤ 2 (TP8 ViT) | 128 | 384+ → L1 overflow → 2× slower |
| 8+ | 256–512 | Generally safe |

Rule: score matrix per chunk = q_chunk × k_chunk × 2 bytes must fit in L1 per core.

### Decode Trace Does NOT Scale to Variable S

If a decode trace is captured at S₁, executing it at S₂ ≠ S₁ gives wrong output.
The SDPA program config (core assignment, tile sizing) is baked at capture time.

**For servers with variable-length inputs**: never use decode traces.
Use `forward_decode_step()` which re-selects the program config each call.

```python
# SAFE for variable-S server:
def decode_forward(self, tokens, start_pos, **kwargs):
    token_id = int(tokens[0, 0].item())
    position = int(start_pos[0].item())
    logits = self.model.forward_decode_step(token_id, position)
    return logits.squeeze(0).unsqueeze(0)
```

### T3K TP8: QKV Column-Parallel Requires Per-Device Head Interleaving

`cat([wq, wk, wv], dim=-1)` with `ShardTensorToMesh(dim=3)` gives each device
the Q slice only — not Q+K+V. Build per-device slices explicitly:

```python
cols = n_local_heads * padded_head_dim
qkv_chunks = [
    torch.cat([wq[:, i*cols:(i+1)*cols], wk[:, i*cols:(i+1)*cols], wv[:, i*cols:(i+1)*cols]], dim=-1)
    for i in range(num_devices)
]
wqkv = torch.cat(qkv_chunks, dim=-1)
```

### QKV Matmul: Increase in0_block_w for DRAM Reuse

Default `in0_block_w=1` loads 1 tile (32 elements) per step, causing excessive DRAM
re-reads. Set `in0_block_w=4` for typical hidden dims (1024–4096):

```python
ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
    in0_block_w=4,   # was 1 — 4 tiles per step improves register reuse
    ...
)
```

## CRITICAL: End-to-End Functional Verification

### PCC Is Necessary But NOT Sufficient

High PCC (> 0.99) at each block is required, but it does NOT guarantee correct output:
- Small errors compound over many layers
- 0.99 PCC per layer × 28 layers = cumulative error
- The final output must also be functionally correct

### Verification Checklist

Before marking TTNN implementation as "DONE":

- [ ] Each block has PCC > 0.99 against reference
- [ ] Full model has PCC > 0.97 against reference (cumulative)
- [ ] **End-to-end output is functionally correct:**
  - For TTS: TTNN audio sounds correct when played
  - For LLM: TTNN generated text makes sense
  - For Vision: TTNN output is visually correct

### For Audio/TTS Models

```python
# Generate audio with TTNN model
ttnn_audio = ttnn_model.generate(input)

# Save and LISTEN
sf.write("/tmp/ttnn_output.wav", ttnn_audio.numpy(), 24000)
print("LISTEN to /tmp/ttnn_output.wav")
print("Compare with /tmp/reference_output.wav")
print("Both should sound similar and be intelligible speech!")

# If TTNN produces noise but reference sounds good:
# 1. DO NOT mark as done
# 2. Debug intermediate tensors to find where PCC drops
# 3. Fix the issue before proceeding
```

### When TTNN Produces Noise/Garbage

If TTNN output is wrong but PCC looks good:
1. Check PCC at EVERY intermediate stage, not just final output
2. Errors may compound - find where PCC first drops significantly
3. Verify the reference itself is correct (produces good output)
4. Don't blame TTNN if the reference is broken

## Output
- `models/demos/{model}/tt/*.py` - TTNN implementations
- All blocks achieving PCC > 0.99 against reference
- **VERIFIED working end-to-end output that matches reference functionally**
