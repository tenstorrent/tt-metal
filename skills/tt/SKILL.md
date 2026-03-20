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

### 3. Memory Configuration

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

#### ⚠️ CRITICAL: Two Modes of PCC Testing

There are two fundamentally different PCC testing modes. **Both are required.**

**Mode A — Isolated (per-block):** each block gets the *reference* hidden state as input.
- Purpose: verifies the block implementation is numerically correct in isolation.
- Passes easily (PCC > 0.99 per block), but **does NOT guarantee correct generation**.

**Mode B — Chained (end-to-end):** TTNN output of layer N feeds into TTNN layer N+1, exactly as happens during inference.
- Purpose: measures accumulated error over all layers.
- **This is the one that matters for audio quality / token correctness.**

**Why Mode A can pass while generation is broken:**
- 28 layers × PCC 0.99 each → compounded chain PCC can fall to ~0.75 or lower
- PCC measures *correlation*, not *magnitude* — a vector can have PCC 1.0000 against reference but max abs error of 160 (scale is off)
- A logit shift of 0.5 across 3072 candidates flips the top predicted token

**Observed example from Qwen3-TTS-12Hz-1.7B:**
```
Layer | Chain PCC | Max Abs Error | Note
    0 |    0.9907 |          0.6  | First layer error
  2-17 |    1.0000 |        160.0  | PCC=1 but scale wrong!
   27 |    0.9967 |        130.0  |
FINAL |    0.9588 |           -   | After final norm reveals scale error
Logit |    0.9694 |           -   | Top token is DIFFERENT from reference
```

#### Chained PCC Test Pattern

Always write a chained test alongside per-block tests:

```python
def test_chained_N_layer_pcc(device, state_dict):
    """Feed TTNN output of layer i into TTNN layer i+1 (chained mode)."""
    # 1. Build the REAL input (e.g. ICL embedding, actual prompt embedding)
    inputs_embeds = build_real_input(state_dict)

    # 2. Run reference chain — decoder_layer() called N times sequentially
    ref_hidden = inputs_embeds.float()
    ref_per_layer = {}
    for i in range(num_layers):
        layer_weights = extract_layer_weights(state_dict, i)
        ref_hidden = reference_decoder_layer(ref_hidden, layer_weights, cos, sin, config)
        ref_per_layer[i] = ref_hidden.clone()

    # 3. Run TTNN chain — each layer receives the TTNN output of the previous layer
    hidden_tt = ttnn.from_torch(inputs_embeds.bfloat16(), device=device,
                                dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                                memory_config=ttnn.DRAM_MEMORY_CONFIG)
    for i, layer in enumerate(ttnn_layers):
        hidden_tt, _ = layer(hidden_tt, cos_tt, sin_tt, trans_mat,
                             attention_mask=None, kv_cache=None,
                             start_pos=0, mode="prefill")
        # Extract for comparison
        ttnn_out = ttnn.to_torch(hidden_tt).squeeze(1).float()[:, :seq_len, :]
        pcc = compute_pcc(ref_per_layer[i], ttnn_out)
        max_err = (ref_per_layer[i] - ttnn_out).abs().max().item()
        print(f"Layer {i:2d}: PCC={pcc:.4f}  MaxAbsErr={max_err:.2f}")

    # 4. Final norm and logit comparison
    final_pcc = compute_pcc(ref_final, ttnn_final)
    logit_pcc = compute_pcc(ref_logits, ttnn_logits)
    print(f"Final PCC: {final_pcc:.4f}  Logit PCC: {logit_pcc:.4f}")

    # 5. Check top predicted token matches
    assert ref_logits.argmax() == ttnn_logits.argmax(), \
        f"Top token mismatch: ref={ref_logits.argmax()} ttnn={ttnn_logits.argmax()}"

    assert final_pcc > 0.99, f"Chained 28-layer PCC {final_pcc:.4f} < 0.99"
```

#### RMSNorm Weight Format for TTNN

When loading RMSNorm weights for `ttnn.rms_norm`, the gamma must be reshaped:

```python
TILE = 32
# Reshape [hidden_size] → [1, 1, hidden_size//32, 32] in ROW_MAJOR_LAYOUT
norm_w = state_dict["model.norm.weight"].to(torch.bfloat16)
norm_w_reshaped = norm_w.view(1, 1, hidden_size // TILE, TILE)
norm_tt = ttnn.as_tensor(norm_w_reshaped, device=device, dtype=ttnn.bfloat16,
                         layout=ttnn.ROW_MAJOR_LAYOUT,
                         memory_config=ttnn.DRAM_MEMORY_CONFIG)
# Then: ttnn.rms_norm(hidden, epsilon=eps, weight=norm_tt)
```

#### PCC Thresholds

| Check | Threshold | Why |
|-------|-----------|-----|
| Per-block isolated | > 0.99 | Confirms block is correct |
| Per-layer chained output | > 0.99 | Confirms errors don't compound |
| Full model final hidden | > 0.99 | Confirms correct output distribution |
| Logit PCC (last pos) | > 0.99 | Confirms correct token prediction |
| Top-1 token match | exact | Confirms greedy decoding correctness |

If chained PCC drops below 0.99 but per-block PCC is fine, the error is in the **RoPE format**, **attention scaling**, **bfloat16 overflow**, or **KV-cache causal masking**.

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
