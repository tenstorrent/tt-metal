---
name: ttnn-debugging
description: Diagnose and fix PCC failures, device hangs, numerical overflow, and other issues during TTNN model bring-up. Use when PCC is below 0.99, device hangs, NaN or Inf values appear, or TTNN output is incorrect.
---

# SKILL: Debugging TTNN Implementations

## Purpose
Diagnose and fix PCC failures, device hangs, and other issues during TTNN bring-up.

## CRITICAL FIRST STEP: Verify Reference Is Correct

**Before debugging TTNN, verify your reference implementation produces correct output!**

### The "Both Are Wrong" Trap

If TTNN output is wrong AND reference output is wrong:
- High PCC between TTNN and reference is MEANINGLESS
- You're measuring agreement between two broken implementations
- The bug might be in the reference, not TTNN

### Verification Steps

1. **Test reference against official package:**
```python
# Run official package (e.g., qwen_tts, transformers)
official_output = official_model.generate(input)

# Run your reference
reference_output = reference_model(input)

# Compare
pcc = compute_pcc(official_output, reference_output)
print(f"Reference vs Official PCC: {pcc}")  # MUST be > 0.99
```

2. **For Audio/TTS: LISTEN to the output:**
```python
# Save reference output as audio
sf.write("/tmp/reference.wav", reference_output.numpy(), 24000)
# LISTEN TO IT! Is it intelligible speech or noise?
```

3. **If reference produces noise/garbage:**
   - DO NOT debug TTNN yet
   - Fix the reference first
   - The reference must match official EXACTLY

## PCC Debugging (< 0.99)

### Layer-by-Layer Debugging Strategy
1. **Isolate the failing layer**: Run each layer independently
2. **Compare intermediate outputs**: Save TTNN and reference outputs at each step
3. **Binary search**: If many layers, bisect to find first failure

```python
def debug_layer_by_layer(model, reference_model, x):
    """Debug by comparing each layer's output."""
    x_ttnn = ttnn.from_torch(x, device=device, layout=ttnn.TILE_LAYOUT)
    x_ref = x.clone()

    for i, (ttnn_layer, ref_layer) in enumerate(zip(model.layers, reference_model.layers)):
        x_ttnn = ttnn_layer(x_ttnn)
        x_ref = ref_layer(x_ref)

        pcc = compute_pcc(ttnn.to_torch(x_ttnn), x_ref)
        print(f"Layer {i}: PCC = {pcc:.6f}")

        if pcc < 0.99:
            print(f"  FAIL at layer {i}")
            # Dump tensors for analysis
            torch.save({'ttnn': ttnn.to_torch(x_ttnn), 'ref': x_ref}, f'debug_layer_{i}.pt')
            break
```

### Common PCC Failure Patterns

| Symptom | Likely Cause | Fix |
|---------|--------------|-----|
| PCC ~0.98-0.99 | Dtype precision | Use bfloat16 instead of bfloat8_b |
| PCC drops at layer N | Overflow accumulation | Check MLP/attention weights at layer N |
| PCC ~0.5 or random | Wrong weight mapping | Verify weight names match HuggingFace |
| PCC ~-1 | Sign flip | Check transpose/reshape operations |
| PCC NaN | Overflow/underflow | Check for Inf values, reduce precision |

### bfloat8_b Numerical Overflow Diagnosis

**Detection**:
```python
def check_for_overflow(tensor):
    """Check if tensor has overflow values."""
    t = ttnn.to_torch(tensor)
    has_inf = torch.isinf(t).any().item()
    has_nan = torch.isnan(t).any().item()
    max_val = t.abs().max().item()
    print(f"Has Inf: {has_inf}, Has NaN: {has_nan}, Max: {max_val}")
    return has_inf or has_nan
```

**Root Cause**: bfloat8_b stores only 8 bits, precision loss accumulates through layers

**Solution**:
```python
# Option 1: Use bfloat16 for all weights
weight = ttnn.from_torch(w, dtype=ttnn.bfloat16, ...)

# Option 2: Store CPU copies, convert on-demand (for decode hot path)
self.weight_cpu = state_dict["weight"].clone()

def forward_decode(self, x):
    weight = ttnn.from_torch(self.weight_cpu.to(torch.bfloat16), device=self.device, ...)
    return ttnn.linear(x, weight)
```

### TILE_LAYOUT Issues
```python
# Check layout before matmul
def safe_linear(x, weight):
    assert x.layout == ttnn.TILE_LAYOUT, f"Input layout: {x.layout}"
    assert weight.layout == ttnn.TILE_LAYOUT, f"Weight layout: {weight.layout}"
    return ttnn.linear(x, weight)
```

### Sharding Spec Mismatches
```python
# Debug sharding
def print_tensor_info(tensor, name="tensor"):
    print(f"{name}:")
    print(f"  Shape: {tensor.shape}")
    print(f"  Layout: {tensor.layout}")
    print(f"  Memory config: {tensor.memory_config()}")
```

## Device Hang Recovery

### T3K Device Reset
```bash
# Environment setup first
export TT_METAL_HOME=$(pwd)
export PYTHONPATH=$(pwd)
source python_env/bin/activate

# Reset device
tt-smi -r
```

### Galaxy Device Reset
```bash
export TT_METAL_HOME=$(pwd)
export PYTHONPATH=$(pwd)
source python_env/bin/activate

# Galaxy-specific reset
tt-smi -glx_reset
```

### Common Hang Causes
1. **Memory exhaustion**: Reduce batch size, use DRAM instead of L1
2. **Infinite loop in kernel**: Check op parameters
3. **Deadlock in multi-chip**: Verify CCL synchronization

## Op-Level Debugging

### Using Unit Tests as Reference
```bash
# Find gold-standard op usage
ls tests/ttnn/unit_tests/operations/

# Run specific op test
pytest tests/ttnn/unit_tests/operations/test_matmul.py -v
```

### Comparing Against Unit Tests
```python
# Check how ops are used in tests
# Example: tests/ttnn/unit_tests/operations/test_scaled_dot_product_attention.py

# Verify your usage matches test patterns
attn = ttnn.transformer.scaled_dot_product_attention(
    q, k, v,
    is_causal=True,
    scale=1.0 / math.sqrt(head_dim),
)
```

## Tensor Inspection Techniques

```python
def inspect_tensor(tensor, name="tensor"):
    """Full tensor inspection."""
    t = ttnn.to_torch(tensor)
    print(f"\n{name}:")
    print(f"  Shape: {t.shape}")
    print(f"  Dtype: {t.dtype}")
    print(f"  Min: {t.min().item():.6f}")
    print(f"  Max: {t.max().item():.6f}")
    print(f"  Mean: {t.mean().item():.6f}")
    print(f"  Std: {t.std().item():.6f}")
    print(f"  Has NaN: {torch.isnan(t).any().item()}")
    print(f"  Has Inf: {torch.isinf(t).any().item()}")
    print(f"  Sample values: {t.flatten()[:5].tolist()}")
```

## Tracing Debugging

### `TT_FATAL: Writes are not supported during trace capture`

**Cause**: a host→device or device→host transfer (`ttnn.from_torch` / `ttnn.to_torch`)
or a non-trace-safe allocation runs INSIDE the captured region. Trace can only record
device-resident ops.

**Diagnosis recipe**: flip trace on with the same forward path and read the stack from
the first `ttnn.begin_trace_capture` frame down. Every per-step `from_torch` /
`to_torch` is a suspect. Common offenders observed in real bring-ups:

| Site | Symptom | Fix |
|------|---------|-----|
| Norm wrappers like `_shard_across_cols` / `_gather_from_cols` that do `to_torch → reshape → from_torch` | Crash on the first decoder layer | Replace with on-device CCL: `ttnn.all_gather(dim, cluster_axis=1)` and keep residual sharded throughout the layer (`llama3_70b_galaxy` pattern) |
| `_embed`: `from_torch(input_ids)` per step | Crash on embedding | Preallocate `input_ids_buf` once; refresh via `ttnn.copy_host_to_device_tensor(host_ids, input_ids_buf)` |
| Logits gather inside forward (`to_torch` at end of `_norm_and_lm_head`) | Crash near LM head | Return `ttnn.Tensor` from forward; do `to_torch` AFTER `ttnn.execute_trace` |
| Decode mask / `cur_pos` / `update_idxs` rebuilt via `from_torch` per call | Crash inside attention | Preallocate `cur_pos` as `[B]` int32 device tensor; derive other indices on-device; build any static mask once at `__init__` and `ttnn.slice` inside forward |
| DeltaNet recurrent state returned as a fresh tensor each step | Trace captures but second `execute_trace` gives wrong output | State must be a `self.state_buffer` preallocated tensor that the kernel writes into in place — same discipline as `paged_update_cache` |

The fix template is in `skills/ttnn` § "Trace-Safe Forward (REQUIRED for production
perf)". Read that section before refactoring.

### `Allocating device buffers is unsafe` Warning

**Cause**: Allocating after `begin_trace_capture` started. Even a stray `ttnn.zeros`
inside the forward will trigger this.

**Solution**: Pre-allocate every tensor before `begin_trace_capture`. Helpers like
`ttnn.zeros(...)` and even `ttnn.linear` with no preallocated output buffer can
allocate. Look for `memory_config=...` args and confirm they reuse existing memory.

### Trace works but second `execute_trace` returns the wrong values

**Cause**: state that should persist across iterations was returned as a fresh tensor
the first time and dangles after that. Trace replays the pointer dependency captured
at `begin_trace_capture` — if the layer returns a *new* DeltaNet state tensor each
call, the trace points at the FIRST iteration's output, not the most recent one.

**Fix**: state-holding layers must hold a `self.state_buffer` (preallocated, persistent
across calls) and update it in place. The KV-cache pattern (`paged_update_cache`) is
the model — copy that discipline for DeltaNet recurrent state and conv1d state.

### `from_torch` works in eager but `copy_host_to_device_tensor` errors with shape mismatch

**Cause**: the preallocated device buffer was created with a different shape, dtype, or
mesh_mapper than the host tensor being copied. `copy_host_to_device_tensor` is strict;
`from_torch` is lenient (it allocates a fresh buffer matching the host tensor).

**Fix**: build the device buffer with `ttnn.from_torch(<reference_host_tensor>, ...)`
ONCE at capture time, then reuse it. The host tensor passed to `copy_host_to_device_tensor`
must match it exactly. Common mismatches: wrong int dtype (`uint32` vs `int32`), wrong
layout (`ROW_MAJOR` vs `TILE`), wrong mesh_mapper (forgot `ReplicateTensorToMesh`).

## Performance Debugging

### Identify Bottlenecks
```python
import time

def profile_layers(model, x):
    """Profile each layer's execution time."""
    for i, layer in enumerate(model.layers):
        ttnn.synchronize_device(device)
        start = time.perf_counter()

        x = layer(x)

        ttnn.synchronize_device(device)
        elapsed = (time.perf_counter() - start) * 1000
        print(f"Layer {i}: {elapsed:.2f}ms")
```

## "Output Is Garbage" Debugging

When the model produces noise/garbage instead of correct output:

### Step 1: Check Component Inventory

```markdown
Is the model COMPLETE? Check ARCHITECTURE.md for:
- [ ] ALL components listed in inventory
- [ ] ALL components have implementations
- [ ] No components marked as "Not implemented"
```

**Common trap for TTS models**: Implementing only the Talker but not the Speech Tokenizer Decoder.

### Step 2: Verify Reference End-to-End

```python
# Run COMPLETE reference pipeline
reference_audio = reference_full_pipeline(text_input)
sf.write("/tmp/reference.wav", reference_audio.numpy(), 24000)
# LISTEN - is it correct?
```

If reference also produces garbage, the bug is in reference, not TTNN.

### Step 3: Compare at Each Stage

```python
# For TTS: Compare at each stage
stages = ["embedding", "talker_out", "code_predictor_out", "decoder_out"]
for stage in stages:
    pcc = compute_pcc(ttnn_outputs[stage], reference_outputs[stage])
    print(f"{stage}: PCC = {pcc:.6f}")
    # First stage with low PCC is where bug is
```

### Step 4: Use Official Package as Ground Truth

If both TTNN and reference are wrong:
```python
# Extract intermediate tensors from OFFICIAL package
# Compare your reference to official at each stage
# Find where your reference diverges from official
```

## T3K / Multi-Chip Bring-Up Bugs

These bugs are silent on single-device tests but appear on T3K [1×8] mesh.

### tt_all_reduce(cluster_axis=1) is a silent NO-OP on T3K

`tt_all_reduce(cluster_axis=1)` silently does nothing on T3K [1,8] mesh.
Symptom: PCC passes on N150 but is ~0.0 or random on T3K for TP blocks.

```python
# WRONG — no-op on T3K:
from models.tt_transformers.tt.ccl import tt_all_reduce
tt_all_reduce(out, cluster_axis=1)

# CORRECT:
out = ttnn.all_reduce(out, cluster_axis=1, num_links=1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
# Note: num_links=2 hangs on T3K — always use num_links=1
```

### RoPE Style Mismatch: Interleaved-Pair vs Concatenated-Halves

Symptom: RoPE-alone PCC ~0.88, text attention PCC ~0.5; first layer to fail.

Check `modeling_{model}.py` for `rotate_half` (HF-style, concatenated-halves →
use `ttnn.experimental.rotary_embedding`) vs `rotate_interleaved` (LLaMA-style →
use `ttnn.experimental.rotary_embedding_llama` + transformation matrix).

HF-style cos/sin format: `cos_hf = cat([cos[:half], cos[:half]], dim=-1)`.

### MLP Gate/Value Order Varies by Model

Symptom: MLP PCC ~0.5.

Check `modeling_{model}.py` for `.chunk(2)`:
- **Standard**: `gate, value = ff.chunk(2, dim=-1)` → `act(gate) * value`
- **Reversed** (e.g. Molmo2): `value, gate = ff.chunk(2, dim=-1)` → `act(gate) * value`

Load accordingly: `w1 = ff_proj[I:]` (gate), `w3 = ff_proj[:I]` (value) for reversed.

### T3K TP8: QKV Head Interleaving Wrong

Simple `cat([wq,wk,wv])` with `ShardTensorToMesh(dim=3)` gives device i only the Q
slice. Symptom: attention PCC ~0.5 even when QKV projection PCC is high.

Build per-device [Q_i ‖ K_i ‖ V_i] slices explicitly (see TTNN skill).

### HEIGHT_SHARDED Decode Head Concat Shape Issues

`nlp_create_qkv_heads_decode` distributes K/V heads across cores; `nlp_concat_heads_decode`
may return unexpected global shape. Symptom: decode output shape wrong, PCC ~0.0.

Workaround: use CPU reference decode (`_generate_cpu_decode`) for initial bringup.

### CPU Decode Weight Reconstruction: Three Common Errors

1. **wqkv de-interleaving**: T3K shards are `[Q₀,K₀,V₀,Q₁,K₁,V₁,...]`; de-interleave
   per device then cat → `[Q_all ‖ K_all ‖ V_all]`.
2. **wo orientation**: TTNN stores `weight.T`; `F.linear` needs `.T` back.
3. **MLP replicated**: `ttnn.get_device_tensors(w1)[0]` only — catting all 8 gives 8× wrong.

### SDPA Partial-Tile Hang for Power-of-2 Sequences

Certain SDPA program configs hang for power-of-2-padded sequences. Use adaptive config:

```python
def sdpa_progcfg(seq_len):
    grid = min(8, max(1, math.ceil(seq_len / 128)))
    return ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=(grid, grid),
        q_chunk_size=min(128, seq_len),
        k_chunk_size=min(128, seq_len),
    )
```

### Decode Trace Wrong Results for Variable S

Trace captured at S₁ gives wrong decode for S₂ ≠ S₁ (SDPA program config baked).
Symptom: correct prefill output, but wrong first decode token (often EOS or wrong letter).

Fix: disable decode traces for servers; use `forward_decode_step()` which re-selects
the program config on every call.

### `ttnn.reshape` view aliasing — `deallocate(True)` on pre-reshape tensor invalidates the view

Symptom: PCC degrades step-by-step in deep stacks (e.g. 16L PCC=0.98, 4L PCC=0.9999).
The per-block PCC is fine in isolation but multi-layer regression diverges. Root cause:
`ttnn.reshape` returns an *aliased view* — not a copy — and the post-reshape tensor
becomes invalid once you call `deallocate(True)` on the pre-reshape tensor it views.

```python
# BUG: view freed under our feet
x = ttnn.reshape(qk_per_head, ttnn.Shape([B, T, n_heads, head_dim]))
qk_per_head.deallocate(True)        # ← invalidates x!
... use x ...                       # corrupted reads, no error raised
```

Fixes (any of):
- Keep the pre-reshape tensor alive until you're fully done with the reshaped view.
- For multi-head loops: collect all per-head views, concat, THEN deallocate the source.
- For the n_heads==1 path: `ttnn.clone()` the view before freeing the source.

Found in `models/demos/qwen3_6_galaxy/tt/llama_attention.py` (T11c fix). The
no-error-raised aspect is the trap — only PCC drift exposes it.

### `fast_reduce_nc` uses padded_shape, not logical_shape (T=1 decode bug)

Symptom: prefill PCC = 0.9999, decode PCC = 0.5 for T=1 batches. Off-by-31 errors
in residual shape, NaN propagation downstream.

`ttnn.fast_reduce_nc` and several other ops operate on the **padded** tile shape, so a
T=1 input stored in a 32-row tile produces a logical T=32 output. The garbage in rows
[1:32] then leaks into subsequent matmuls.

Fix: clip back to the logical shape with `ttnn.slice` after the op:

```python
x_normed = self.distributed_norm(x_sharded)
x_normed = _gather_from_cols(x_normed_sharded, self.mesh_device)
# CLIP back — fast_reduce_nc inside the norm padded T=1 to T=32:
x_normed = ttnn.slice(x_normed, [0, 0, 0], [B, T, H],
                      memory_config=ttnn.DRAM_MEMORY_CONFIG)
```

Watch for this anywhere reductions / norms run on T < 32. Standard `ttnn.reshape`
preserves logical shape, but ops that internally operate on tiles may not.

### Zero-Centered RMSNorm: `(1+w)*norm(x)` vs `w*norm(x)`

HF's `Qwen3NextRMSNorm` uses `output = (1+w) * x * rsqrt(var+eps)` (weight is
*zero-centered*: stored as the deviation from 1.0). Loading the weight as-is and
applying the standard `w*norm(x)` kernel drops PCC catastrophically on early layers
where the residual is small.

Diagnosis: norm-block PCC ≈ 0.5 while every other block is fine; full model output is
near-random.

Fix at weight-load time (no kernel change needed):

```python
w = state_dict[f"{prefix}.weight"].float()
if zero_centered:
    w = w + 1.0   # bake +1 into the stored weight; standard rmsnorm kernel now correct
```

Models that use this convention: Qwen3-Next family (Qwen3.6-27B), some Olmo3 variants.
Check HF `modeling_*.py` for `1 + self.weight` in the RMSNorm forward — that's the tell.

The reverse exists too: `Qwen3NextRMSNormGated` in the same file uses the standard
convention (`w*norm(x)`). Different norms in the SAME model can use different
conventions — read every norm class.

### DistributedNorm Precision (HiFi2 / BF16-accum drops PCC on small activations)

`rms_norm_pre_all_gather` computes sum-of-squares, then `rms_norm_post_all_gather`
applies `rsqrt(var+eps)`. With BF16 accumulation, `rsqrt` of a small variance
(embedding output std ≈ 0.013 → var ≈ 1.7e-4) compounds rounding error catastrophically
on the way to a 5120-dim hidden state.

Symptom: 4L PCC = 0.985 (norm-bounded), block-level norm test passes 0.9999.

Fix: use HiFi4 + fp32 dest accumulation in the compute kernel config:

```python
self.compute_kernel_config = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4,
    math_approx_mode=False,
    fp32_dest_acc_en=True,
    packer_l1_acc=True,
)
```

Apply this to every norm operating on small-magnitude residuals (i.e. the residual
stream norms in a hybrid model). Block-level test gain: 0.99998. 4L gain: 0.985 → 0.9934.

**BUT** — HiFi4 has bigger per-tile CBs. For *replicated* `ttnn.rms_norm` on a wide
dim (5120 on a single core), HiFi4 + fp32_dest_acc CBs don't fit alongside the L1
residual buffer. Symptom:

```
TT_THROW @ program.cpp:1052: Statically allocated circular buffers in program N
clash with L1 buffers on core range [(x=0,y=0)-(x=0,y=0)].
L1 buffer allocated at 1355776 and static circular buffer region ends at 1520128
```

For the replicated case, use **HiFi2 + fp32_dest_acc_en=True** (matches production
`models/common/rmsnorm.py` line 115-120 — Llama-3 8B, 70B all run this way). The
fp32 dest accumulation is what preserves precision; HiFi2 vs HiFi4 only affects
mantissa rounds in the matmul-like inner product, which has negligible impact
on the variance estimate over a 5120-element sum.

Rule of thumb: HiFi4 + fp32_dest_acc on a *sharded* slice (1280-dim per col) → fits
in L1, max precision. HiFi2 + fp32_dest_acc on the *replicated* full dim (5120) →
fits in L1, same effective precision for norms.

### `Circular buffers clash with L1 buffers` — L1 CB-budget overflow

The error above (`program.cpp:1052: Statically allocated circular buffers in
program N clash with L1 buffers on core range ...`) is **not** memory exhaustion in
the usual sense — it's a layout conflict. A static CB region the op wants to put
on core (x, y) overlaps an L1 *buffer* already allocated there (residual stream,
KV cache slice, etc.). Diagnostic info in the message: `L1 buffer allocated at
A`, `static circular buffer region ends at B`. If `A < B`, they overlap.

Causes & fixes:
- **HiFi4 + fp32_dest_acc on a wide unsharded input**: switch to HiFi2 (above).
- **Single-core program config on a multi-core-amenable op**: pass an explicit
  `program_config` with a bigger `compute_with_storage_grid_size`.
- **Op called on an INTERLEAVED L1 input where the L1 buffer is large**: move the
  input to DRAM via `ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)` before the
  op, or shard the input across cores.
- **CB sizing tied to tile-row count × dtype**: drop weight dtype from bfloat16
  to bfloat8_b for the offending op — the weight-side CB shrinks 2×.

The single-core grid in the error (`x=0,y=0 - x=0,y=0`) is the giveaway: ttnn
picked a default single-core layout that doesn't fit. Force multi-core or move
the bulky buffer to DRAM.

### Paged KV Cache PCC Mismatch vs Non-Paged

Symptom: paged-attention test fails parity against non-paged at PCC ≈ 0.5–0.9 (not the
expected 0.999+). Both paths are the same math — divergence is a layout/index bug.

Common causes:
1. **Wrong `block_size` vs `k_chunk_size`**: SDPA's `k_chunk_size` must equal (or
   evenly divide) `block_size`. Mismatch reads garbage from the next block.
2. **`update_idxs_tensor` index calculation wrong**: `paged_update_cache` expects
   absolute positions in the user's logical sequence, not offsets into the current
   block. If you've translated to a block-local offset, the write lands in the wrong
   slot in the pool.
3. **Page table not refreshed before decode** when the user's sequence crosses a block
   boundary — `paged_update_cache` reads `page_table[user_id, block_idx]` for the
   physical block, but if the table still has the old block_idx, the write goes to
   the previous block.
4. **`paged_update_cache` not configured HEIGHT_SHARDED on 1 core**: the default
   program config can shard across cores in a way incompatible with paged layout.
   Use `compute_with_storage_grid_size=(1,1)`.

Test the parity FIRST (small seq, small block_size) with a known-good non-paged
reference; only after PCC ≥ 0.999 against non-paged should you trust the paged
pipeline for production.

---

## VLM Server Accuracy Debugging

When server accuracy is significantly below demo accuracy (≥10 pp gap):

### 1. Compare S values

```python
# Server: log in prefill_forward
logger.info(f"Prefill: S={seq_len}")
# Demo: proc(text=fmt, videos=[frames])["input_ids"].shape[1]
```

If S_server ≠ S_demo: preprocessing path differs — all other symptoms are downstream.

### 2. Frame markers missing from input_ids (most common cause, ~30 pp drop)

vLLM's default PromptReplacement inserts only `image_patch_id` tokens, omitting
`<im_start>/<im_end>` frame markers. Fix: store `video_input_ids` in `_call_hf_processor`
and use it as PromptReplacement content (see tt-inference-server skill).

### 3. token_type_ids missing frame markers

```python
# Wrong — patches only:
tti = (input_ids == IMAGE_PATCH_ID).long()
# Correct — patches AND frame markers:
tti = ((input_ids == IMAGE_PATCH_ID) | (input_ids == IM_START) | (input_ids == IM_END)).long()
```

### 4. Prefill logit comparison (PCC)

```python
model.reset_kv_cache(user_id=0)
logits_server = model.forward_prefill(input_ids=server_ids, pixel_values=server_pv,
                                       token_type_ids=server_tti, user_id=0)
model.reset_kv_cache(user_id=0)
logits_demo = model.forward_prefill(input_ids=demo_ids, pixel_values=demo_pv,
                                     token_type_ids=demo_tti, user_id=0)
pcc = torch.corrcoef(torch.stack([logits_server.float().flatten(),
                                    logits_demo.float().flatten()]))[0, 1].item()
print(f"Prefill PCC: {pcc:.4f}")
# < 0.95 → preprocessing bug; > 0.98 but wrong → decode bug (trace or KV dtype)
```

### 5. bfloat8_b KV cache logit flips (S > 2500)

Borderline token pairs (e.g. 'B' vs 'C') flip at long sequences.
Fix: use bfloat16 KV cache. Symptom: correct prefill but wrong first decode token.

### 6. Decode trace wrong results at variable S

See T3K section above. Fix: disable trace, use `forward_decode_step()`.

---

## Debugging Checklist

- [ ] **Reference produces correct output** (not just runs without errors)
- [ ] **All model components implemented** (check ARCHITECTURE.md inventory)
- [ ] Verify weight names match HuggingFace exactly
- [ ] Check dtypes (bfloat16 vs bfloat8_b)
- [ ] Confirm TILE_LAYOUT for all matmul inputs
- [ ] Verify shapes are padded to tile size (32x32)
- [ ] Compare against similar model in ARCHITECTURE.md
- [ ] Run unit tests for individual ops
- [ ] Check for NaN/Inf values at each layer
- [ ] Verify sharding specs match between ops
