# Chunked Prefill Stability & 65K+ Sequence Length Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix chunked delta rule numerical stability so prefill uses parallel chunked mode, and support sequences up to 65,536+ tokens.

**Architecture:** Fix the decay normalization in `chunk_gated_delta_rule_ttnn` (keep both raw and normalized decay), switch prefill from recurrent to chunk mode, add segmented prefill that processes long sequences in 2048-token segments with state continuity across segments.

**Tech Stack:** Python, TTNN (Tenstorrent), PyTorch (reference), pytest

**Spec:** `docs/superpowers/specs/2026-03-23-qwen35-9b-chunked-prefill-stability-design.md`

---

## File Structure

| File | Role | Action |
|------|------|--------|
| `models/experimental/gated_attention_gated_deltanet/tt/ttnn_delta_rule_ops.py` | Core chunked delta rule algorithm | Modify `chunk_gated_delta_rule_ttnn` |
| `models/demos/blackhole/qwen3_5_9b/tt/qwen35_decoder.py` | Layer dispatch (recurrent vs chunk) | Change prefill mode to "chunk" |
| `models/demos/blackhole/qwen3_5_9b/tt/qwen35_model.py` | Model entry point (prefill, decode) | Add segmented prefill |
| `models/demos/blackhole/qwen3_5_9b/tt/qwen35_gated_deltanet.py` | DeltaNet wrapper (chunk_size config) | Update default chunk_size to 64 |
| `models/experimental/gated_attention_gated_deltanet/tt/ttnn_gated_attention.py` | SDPA program config | Verify/fix 65K KV config |
| `models/demos/blackhole/qwen3_5_9b/tt/qwen35_rope.py` | RoPE precomputation | Already scales with max_seq_len (verified: `__init__` uses `args.max_seq_len` to size tables) |
| `models/demos/blackhole/qwen3_5_9b/tt/model_config.py` | Model configuration | No code change needed — `max_seq_len` is a constructor param passed from `from_pretrained` |
| `models/demos/blackhole/qwen3_5_9b/tests/test_chunked_pcc.py` | PCC validation test | Create new |
| `models/demos/blackhole/qwen3_5_9b/tests/test_seq_len.py` | Seq length sweep test | Add 4096, 8192, 65536 cases |

---

### Task 1: Write PCC Test for Chunked vs Recurrent Delta Rule

**Files:**
- Create: `models/demos/blackhole/qwen3_5_9b/tests/test_chunked_pcc.py`

This test runs BOTH the chunked and recurrent paths on the same input and compares outputs. It must pass AFTER we fix the chunked path (Task 2), so we write it first and expect it to fail.

- [ ] **Step 1: Create the PCC test file**

```python
# models/demos/blackhole/qwen3_5_9b/tests/test_chunked_pcc.py
"""PCC validation: chunked vs recurrent delta rule.

Verifies that the chunked (parallel) delta rule produces output matching
the recurrent (sequential) reference within PCC > 0.99.

Run: pytest models/demos/blackhole/qwen3_5_9b/tests/test_chunked_pcc.py -v -s --timeout=600
"""
import pytest
from loguru import logger
import torch
import ttnn

from models.demos.blackhole.qwen3_5_9b.tt.model_config import Qwen35ModelArgs

CHECKPOINT_DIR = "/localdev/atupe/Qwen3.5-9B-Claude-4.6-Opus-Reasoning-Distilled-v2"


def compute_pcc(a, b):
    """Pearson correlation coefficient between two tensors."""
    a_flat = a.float().flatten()
    b_flat = b.float().flatten()
    a_c = a_flat - a_flat.mean()
    b_c = b_flat - b_flat.mean()
    return ((a_c * b_c).sum() / (a_c.norm() * b_c.norm() + 1e-8)).item()


@pytest.fixture(scope="module")
def device():
    dev = ttnn.open_device(device_id=0)
    dev.enable_program_cache()
    yield dev
    ttnn.close_device(dev)


def _run_single_deltanet_layer_both_modes(device, seq_len, chunk_size=64):
    """Run one DeltaNet layer in both recurrent and chunk mode, return PCC."""
    import glob
    from safetensors import safe_open
    from models.demos.blackhole.qwen3_5_9b.tt.weight_mapping import remap_qwen35_state_dict
    from models.demos.blackhole.qwen3_5_9b.tt.qwen35_gated_deltanet import Qwen35GatedDeltaNet

    args = Qwen35ModelArgs(mesh_device=device, checkpoint_dir=CHECKPOINT_DIR)
    raw = {}
    for path in sorted(glob.glob(f"{CHECKPOINT_DIR}/model.safetensors-*.safetensors")):
        with safe_open(path, framework="pt", device="cpu") as f:
            for key in f.keys():
                raw[key] = f.get_tensor(key)
    sd = remap_qwen35_state_dict(raw)
    del raw

    # Use layer 0 (a DeltaNet layer)
    layer_num = 0
    dn = Qwen35GatedDeltaNet(args, sd, layer_num, device)

    # Create random input of the target seq_len
    x_torch = torch.randn(1, seq_len, 4096, dtype=torch.bfloat16)
    x_ttnn = ttnn.from_torch(x_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    # Run recurrent mode (reference)
    dn.reset_state(batch_size=1)
    x_recurrent = ttnn.clone(x_ttnn)
    out_recurrent = dn.forward(x_recurrent, mode="recurrent")
    out_recurrent_torch = ttnn.to_torch(out_recurrent)
    ttnn.deallocate(out_recurrent)

    # Run chunk mode
    dn.reset_state(batch_size=1)
    x_chunk = ttnn.clone(x_ttnn)
    out_chunk = dn.forward(x_chunk, mode="chunk", chunk_size=chunk_size)
    out_chunk_torch = ttnn.to_torch(out_chunk)
    ttnn.deallocate(out_chunk)
    ttnn.deallocate(x_ttnn)

    pcc = compute_pcc(out_recurrent_torch, out_chunk_torch)
    logger.info(f"seq_len={seq_len}, chunk_size={chunk_size}: PCC={pcc:.6f}")
    logger.info(f"  recurrent range: [{out_recurrent_torch.min():.4f}, {out_recurrent_torch.max():.4f}]")
    logger.info(f"  chunk range:     [{out_chunk_torch.min():.4f}, {out_chunk_torch.max():.4f}]")

    return pcc


@pytest.mark.parametrize("seq_len", [64, 128, 256, 512, 1024, 100, 300, 700])
def test_chunked_vs_recurrent_pcc(seq_len, device):
    """Chunked delta rule must match recurrent reference with PCC > 0.99."""
    pcc = _run_single_deltanet_layer_both_modes(device, seq_len, chunk_size=64)
    assert pcc > 0.99, f"PCC {pcc:.6f} < 0.99 at seq_len={seq_len}"
```

- [ ] **Step 2: Run the test to verify it fails (current chunked mode has precision issues)**

Run: `timeout 300 pytest models/demos/blackhole/qwen3_5_9b/tests/test_chunked_pcc.py -v -s -k "256" --timeout=120 2>&1 | tail -30`

Expected: FAIL with PCC < 0.99 for seq_len=256 (confirming the known precision issue).

If the test hangs: `kill` the process, run `tt-smi -r` to reset the device, then debug.

---

### Task 2: Fix Chunked Delta Rule Numerical Stability

**Files:**
- Modify: `models/experimental/gated_attention_gated_deltanet/tt/ttnn_delta_rule_ops.py` (function `chunk_gated_delta_rule_ttnn`, lines 713-1073)

The fix has three parts: (a) per-chunk decay normalization, (b) clamp before exp, (c) use raw decay for state-interaction sites.

- [ ] **Step 1: Add per-chunk decay normalization and clamping**

In `chunk_gated_delta_rule_ttnn`, after the decay cumsum computation (around line 888), apply normalization. The key changes are:

1. After computing `decay` (cumsum), save `decay_raw` and compute `decay_normalized`:
```python
# AFTER line 890 (decay = reshape(matmul(g_c_3d, triu_ones), [batch, chunk_size]))
decay_raw = decay  # keep un-normalized for state interactions (Sites 2, 3, 5)

# Per-chunk normalization: subtract first element so cumsum starts at 0
decay_offset = decay_raw[:, 0:1]  # [batch, 1]
decay = ttnn.subtract(decay_raw, decay_offset, memory_config=None)  # normalized
```

2. Replace the L_diff/L_mask computation to use clamping:
```python
# REPLACE lines 898-903
decay_col = ttnn.reshape(decay, [batch, chunk_size, 1], memory_config=None)
decay_row = ttnn.reshape(decay, [batch, 1, chunk_size], memory_config=None)
L_diff = ttnn.subtract(decay_col, decay_row, memory_config=None)

# Clamp before exp to prevent overflow/underflow
L_diff_masked = ttnn.multiply(L_diff, tril_mask, memory_config=None)
L_diff_clamped = ttnn.clip(L_diff_masked, min=-20.0, max=0.0)
L_mask = ttnn.multiply(ttnn.exp(L_diff_clamped, memory_config=None), tril_mask, memory_config=None)
```

3. Use `decay_raw` for `decay_exp` (Site 2 - key scaling for state correction):
```python
# REPLACE line 892-896 (decay_exp computation)
# Use raw decay for state-interaction terms
decay_raw_exp = ttnn.reshape(
    ttnn.exp(ttnn.clip(decay_raw, min=-20.0, max=0.0), memory_config=None),
    [batch, chunk_size, 1],
    memory_config=None,
)
```

Then use `decay_raw_exp` instead of `decay_exp` at line 947: `k_beta_decay = ttnn.multiply(k_beta_c, decay_raw_exp, memory_config=None)`

- [ ] **Step 2: Fix the per-chunk loop to use raw decay for Sites 3 and 5**

**IMPORTANT:** The existing line 963 (`decay_3d = ttnn.reshape(decay, ...)`) must remain unchanged — it uses the normalized `decay` and feeds `L_mask_4d` (Site 1, correct). Add a NEW `decay_raw_3d` alongside it:

```python
# Line 963 — KEEP AS-IS (uses normalized decay for L_mask_4d):
decay_3d = ttnn.reshape(decay, [BH, num_chunks, chunk_size], memory_config=None)

# ADD NEW LINE after 963: raw decay for per-chunk Sites 3 and 5
decay_raw_3d = ttnn.reshape(decay_raw, [BH, num_chunks, chunk_size], memory_config=None)
```

In the per-chunk loop (lines 991-1056), REPLACE the `decay_i` assignment at line 997 and update Sites 3, 4, and 5:

```python
# Line 997 — REPLACE with raw decay for Sites 3 and 5:
decay_raw_i = decay_raw_3d[:, i]  # raw (un-normalized) for state interactions
# Note: decay_3d[:, i] is NOT needed — L_mask_4d[:, i] is used directly for Site 1

# Site 3 (line 1013-1018): query scaling uses raw decay
decay_i_exp = ttnn.reshape(
    ttnn.exp(ttnn.clip(decay_raw_i, min=-20.0, max=0.0), memory_config=None),
    [BH, chunk_size, 1],
    memory_config=None,
)
q_decay = ttnn.multiply(q_i, decay_i_exp, memory_config=None)
```

For Site 4 (line 1033, inter-chunk state decay): add clamping for consistency:
```python
# Site 4: clamp dl_i before exp (won't change results but prevents underflow warnings)
dl_i_exp = ttnn.exp(ttnn.clip(dl_i, min=-20.0, max=0.0), memory_config=None)
```

For Site 5 (key scaling for state update, lines 1040-1050): use raw decay for the difference:
```python
# Site 5: decay_diff uses raw decay
decay_diff = ttnn.subtract(
    ttnn.reshape(dl_i, [BH, 1], memory_config=None),
    decay_raw_i,  # raw, not normalized
    memory_config=None,
)
decay_diff_exp = ttnn.exp(ttnn.clip(decay_diff, min=-20.0, max=0.0), memory_config=None)
```

- [ ] **Step 3: Run the PCC test to verify the fix**

Run: `timeout 300 pytest models/demos/blackhole/qwen3_5_9b/tests/test_chunked_pcc.py -v -s -k "256" --timeout=120 2>&1 | tail -30`

Expected: PASS with PCC > 0.99.

If it fails, check:
- Are there NaN values? (indicates unclamped exp overflow)
- Is PCC slightly below 0.99? (try chunk_size=32)
- Does it hang? (kill, `tt-smi -r`, check for L1 OOM)

- [ ] **Step 4: Run the full PCC test suite**

Run: `timeout 600 pytest models/demos/blackhole/qwen3_5_9b/tests/test_chunked_pcc.py -v -s --timeout=120 2>&1 | tail -50`

Expected: All parametrized cases pass (64, 128, 256, 512, 1024, 100, 300, 700) with PCC > 0.99.

---

### Task 3: Update Chunk Size and Switch Prefill to Chunked Mode

**Files:**
- Modify: `models/demos/blackhole/qwen3_5_9b/tt/qwen35_gated_deltanet.py` (line 151)
- Modify: `models/demos/blackhole/qwen3_5_9b/tt/qwen35_decoder.py` (line 79)

- [ ] **Step 1: Change default chunk_size from 128 to 64**

In `models/demos/blackhole/qwen3_5_9b/tt/qwen35_gated_deltanet.py`, line 151:

```python
# Before:
self.prefill_chunk_size = 128
# After:
self.prefill_chunk_size = 64
```

Also update the cached mask creation on line 152 — it already uses `self.prefill_chunk_size`, so just the value change is needed.

- [ ] **Step 2: Switch prefill from recurrent to chunk mode**

In `models/demos/blackhole/qwen3_5_9b/tt/qwen35_decoder.py`, line 79:

```python
# Before:
deltanet_mode = "recurrent"
# After:
deltanet_mode = "chunk"
```

- [ ] **Step 3: Run existing seq_len tests to verify no regression**

Run: `timeout 600 pytest models/demos/blackhole/qwen3_5_9b/tests/test_seq_len.py -v -s -k "seq128 or seq256 or seq512" --timeout=300 2>&1 | tail -50`

Expected: PASS for seq128, seq256, seq512. The model should now use chunked prefill and produce coherent output.

If a test hangs: kill, `tt-smi -r`, debug. Possible issues:
- L1 OOM at seq512 with chunk mode: try DRAM memory config for intermediates
- NaN in logits: check if the clamping bounds need adjustment

---

### Task 4: Add Segmented Prefill for Long Sequences

**Files:**
- Modify: `models/demos/blackhole/qwen3_5_9b/tt/qwen35_model.py`

- [ ] **Step 1: Add `prefill_segmented` method**

Add this method to `Qwen35Model` class, after the existing `prefill` method (after line 123):

```python
def prefill_segmented(self, token_ids, segment_size=2048):
    """Prefill long sequences by processing in segments.

    Each segment runs through all 32 layers. DeltaNet recurrent state and
    conv state carry over between segments automatically (stored as instance
    attributes). Attention KV cache accumulates via concat.

    Args:
        token_ids: [B, T] token IDs, T can be >> segment_size
        segment_size: number of tokens per segment (default 2048)
    """
    B, T = token_ids.shape
    self.reset_state(batch_size=B)

    token_ids_ttnn = ttnn.from_torch(token_ids, dtype=ttnn.uint32, device=self.device)
    x_all = ttnn.embedding(token_ids_ttnn, self.tok_embeddings, layout=ttnn.TILE_LAYOUT)
    # For 65K tokens, x_all is [1, 65536, 4096] = ~512MB — keep in DRAM
    x_all = ttnn.to_memory_config(x_all, ttnn.DRAM_MEMORY_CONFIG)
    ttnn.deallocate(token_ids_ttnn)

    for seg_start in range(0, T, segment_size):
        seg_end = min(seg_start + segment_size, T)

        # Slice embeddings for this segment
        x_seg = x_all[:, seg_start:seg_end, :]
        x_seg = ttnn.to_layout(x_seg, ttnn.TILE_LAYOUT)

        # RoPE with absolute positions
        position_ids = torch.arange(seg_start, seg_end).unsqueeze(0).expand(B, -1)
        cos, sin = self.rope.get_rot_mats(position_ids)

        # Process through all layers
        for layer in self.layers:
            x_seg = layer.forward(x_seg, cos=cos, sin=sin, mode="prefill")

        logger.info(f"Prefill segment [{seg_start}:{seg_end}] done")

    # Free the full embedding tensor
    ttnn.deallocate(x_all)

    # Final norm + LM head on last token only
    x_last = x_seg[:, -1:, :]
    x_last = rms_norm_ttnn(x_last, self.norm_weight, eps=self.norm_eps)
    logits = ttnn.linear(x_last, self.lm_head_weight)

    return logits
```

- [ ] **Step 2: Update `prefill` to dispatch to segmented path for long sequences**

Replace the existing `prefill` method (lines 103-123) with:

```python
def prefill(self, token_ids, segment_size=2048):
    B, T = token_ids.shape

    if T > segment_size:
        return self.prefill_segmented(token_ids, segment_size=segment_size)

    # Original path for short sequences
    self.reset_state(batch_size=B)

    token_ids_ttnn = ttnn.from_torch(token_ids, dtype=ttnn.uint32, device=self.device)
    x = ttnn.embedding(token_ids_ttnn, self.tok_embeddings, layout=ttnn.TILE_LAYOUT)

    position_ids = torch.arange(T).unsqueeze(0).expand(B, -1)
    cos, sin = self.rope.get_rot_mats(position_ids)

    for layer in self.layers:
        x = layer.forward(x, cos=cos, sin=sin, mode="prefill")

    x = rms_norm_ttnn(x, self.norm_weight, eps=self.norm_eps)

    x_last = x[:, -1:, :]
    logits = ttnn.linear(x_last, self.lm_head_weight)

    return logits
```

- [ ] **Step 3: Also update `_prefill_for_trace` to support segmented prefill**

The `_prefill_for_trace` method (lines 196-266) needs the same segmented logic for traced decode at long seq lengths. Replace the core prefill loop (lines 211-216) with:

```python
# Run prefill — segmented for long sequences
for layer in self.layers:
    if layer.is_full_attention:
        layer.attention.use_trace_mode = False
        layer.attention.use_preallocated_cache = False

segment_size = 2048
for seg_start in range(0, T, segment_size):
    seg_end = min(seg_start + segment_size, T)

    x_seg = x[:, seg_start:seg_end, :]
    x_seg = ttnn.to_layout(x_seg, ttnn.TILE_LAYOUT)

    position_ids = torch.arange(seg_start, seg_end).unsqueeze(0).expand(B, -1)
    cos_seg, sin_seg = self.rope.get_rot_mats(position_ids)

    for layer in self.layers:
        x_seg = layer.forward(x_seg, cos=cos_seg, sin=sin_seg, mode="prefill")

x = x_seg  # last segment's output
```

Keep the rest of `_prefill_for_trace` unchanged (final norm, KV cache population, conv state fusion).

- [ ] **Step 4: Test segmented prefill at 4096 tokens**

Run: `timeout 600 pytest models/demos/blackhole/qwen3_5_9b/tests/test_seq_len.py -v -s -k "seq2k" --timeout=300 2>&1 | tail -30`

First verify existing 2K test still works. Then test 4K manually:

```bash
timeout 300 python -c "
import torch, ttnn
from models.demos.blackhole.qwen3_5_9b.tt.qwen35_model import Qwen35Model
device = ttnn.open_device(device_id=0)
device.enable_program_cache()
model = Qwen35Model.from_pretrained(device, '/localdev/atupe/Qwen3.5-9B-Claude-4.6-Opus-Reasoning-Distilled-v2', max_seq_len=8192)
tokens = torch.randint(0, 1000, (1, 4096))
logits = model.prefill(tokens)
print('Logits shape:', logits.shape)
print('NaN:', torch.isnan(ttnn.to_torch(logits)).any().item())
ttnn.close_device(device)
print('SUCCESS: 4096 token prefill works')
" 2>&1 | tail -20
```

If it hangs: kill, `tt-smi -r`.

---

### Task 5: Extend Test Suite for Higher Sequence Lengths

**Files:**
- Modify: `models/demos/blackhole/qwen3_5_9b/tests/test_seq_len.py`

- [ ] **Step 1: Add 4096, 8192, and 65536 parametrized cases**

Note: `max_seq_len` is passed to `Qwen35Model.from_pretrained` which passes it through to `Qwen35ModelArgs`, which passes it to `Qwen35RoPESetup` (sizes the RoPE table) and to attention layers (sizes the KV cache). No code changes needed — the parameter already propagates. We just need to pass larger values from the test.

In `test_seq_len.py`, update the `parametrize` decorator (lines 134-151):

```python
@pytest.mark.parametrize(
    "seq_len, max_seq_len",
    [
        (32, 2048),
        (128, 2048),
        (256, 2048),
        (512, 2048),
        (1024, 2048),
        (2024, 2048),
        (4096, 8192),
        (8192, 16384),
        (65536, 131072),
    ],
    ids=[
        "seq32",
        "seq128",
        "seq256",
        "seq512",
        "seq1k",
        "seq2k",
        "seq4k",
        "seq8k",
        "seq65k",
    ],
)
```

- [ ] **Step 2: Run the 4096 test**

Run: `timeout 600 pytest models/demos/blackhole/qwen3_5_9b/tests/test_seq_len.py -v -s -k "seq4k" --timeout=600 2>&1 | tail -30`

Expected: PASS — segmented prefill processes 4096 tokens in 2 segments of 2048.

If it fails, check:
- L1 OOM: reduce segment_size
- KV cache issue: attention layers may need DRAM memory config at 4K

- [ ] **Step 3: Run the 8192 test**

Run: `timeout 600 pytest models/demos/blackhole/qwen3_5_9b/tests/test_seq_len.py -v -s -k "seq8k" --timeout=600 2>&1 | tail -30`

Expected: PASS — 4 segments of 2048 tokens.

- [ ] **Step 4: Run the 65536 test**

Run: `timeout 1800 pytest models/demos/blackhole/qwen3_5_9b/tests/test_seq_len.py -v -s -k "seq65k" --timeout=1800 2>&1 | tail -30`

Expected: PASS — 32 segments of 2048 tokens. This will be slow (32 segments × 32 layers) but should complete.

If it fails with DRAM OOM:
- KV cache at 65K may be too large alongside weights. Try `max_seq_len=65536` and check memory.
- May need to reduce `segment_size` to 1024 to reduce peak activation memory.

If it hangs: kill, `tt-smi -r`. The most likely culprit is SDPA with 65K KV length — see Task 6.

---

### Task 6: Verify and Fix SDPA at 65K KV Length

**Files:**
- Modify: `models/experimental/gated_attention_gated_deltanet/tt/ttnn_gated_attention.py` (lines 61-89)

This task may not require changes if the existing config works. Run verification first.

- [ ] **Step 1: Smoke test SDPA with large KV length**

```bash
timeout 120 python -c "
import torch, ttnn
device = ttnn.open_device(device_id=0)

# Simulate SDPA at 65K KV length, 2048 query length
# Qwen3.5 attention: 16 Q heads, 4 KV heads, head_dim=256
B, H_q, H_kv, D = 1, 16, 4, 256
KV_LEN = 65536
Q_LEN = 2048

q = ttnn.from_torch(torch.randn(B, H_q, Q_LEN, D, dtype=torch.bfloat16),
                     dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
k = ttnn.from_torch(torch.randn(B, H_kv, KV_LEN, D, dtype=torch.bfloat16),
                     dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device,
                     memory_config=ttnn.DRAM_MEMORY_CONFIG)
v = ttnn.from_torch(torch.randn(B, H_kv, KV_LEN, D, dtype=torch.bfloat16),
                     dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device,
                     memory_config=ttnn.DRAM_MEMORY_CONFIG)

grid = device.compute_with_storage_grid_size()
cfg = ttnn.SDPAProgramConfig(compute_with_storage_grid_size=grid, q_chunk_size=64, k_chunk_size=64, exp_approx_mode=False)
compute_cfg = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=True)

out = ttnn.transformer.scaled_dot_product_attention(q, k, v, is_causal=True, scale=D**-0.5,
                                                      program_config=cfg, compute_kernel_config=compute_cfg)
print('SDPA output shape:', out.shape)
print('NaN:', torch.isnan(ttnn.to_torch(out)).any().item())
ttnn.close_device(device)
print('SUCCESS: SDPA at 65K KV works')
" 2>&1 | tail -10
```

Expected: SUCCESS. If it fails:
- OOM: try larger k_chunk_size (128 or 256)
- Kernel error: may need smaller q_chunk_size
- Update `_get_sdpa_program_config` accordingly

- [ ] **Step 2: If SDPA fails, add a 65K tier to the config**

Only if Step 1 fails. In `ttnn_gated_attention.py`, update `_get_sdpa_program_config`:

```python
def _get_sdpa_program_config(device, seq_len, q_seq_len=None):
    grid_size = device.compute_with_storage_grid_size()
    if q_seq_len is not None and q_seq_len <= 1 and seq_len >= 512:
        q_chunk = 32
        k_chunk = 64
    elif seq_len >= 32768:   # NEW: tier for very long KV
        q_chunk = 64
        k_chunk = 128        # larger k_chunk to reduce overhead
    elif seq_len >= 8192:
        q_chunk = 64
        k_chunk = 64
    # ... rest unchanged
```

- [ ] **Step 3: Re-run 65K test if SDPA config was changed**

Run: `timeout 1800 pytest models/demos/blackhole/qwen3_5_9b/tests/test_seq_len.py -v -s -k "seq65k" --timeout=1800 2>&1 | tail -30`

---

### Task 7: Run Full Regression Test Suite

**Files:** No changes — validation only.

- [ ] **Step 1: Run the complete PCC test**

Run: `timeout 600 pytest models/demos/blackhole/qwen3_5_9b/tests/test_chunked_pcc.py -v -s --timeout=120 2>&1 | tail -30`

Expected: All 8 cases pass with PCC > 0.99.

- [ ] **Step 2: Run the complete seq_len sweep (short lengths)**

Run: `timeout 600 pytest models/demos/blackhole/qwen3_5_9b/tests/test_seq_len.py -v -s -k "seq32 or seq128 or seq256 or seq512 or seq1k or seq2k" --timeout=300 2>&1 | tail -30`

Expected: All 6 existing cases pass (no regression).

- [ ] **Step 3: Run the complete seq_len sweep (long lengths)**

Run: `timeout 1800 pytest models/demos/blackhole/qwen3_5_9b/tests/test_seq_len.py -v -s -k "seq4k or seq8k" --timeout=600 2>&1 | tail -30`

Expected: Both pass.

- [ ] **Step 4: Run the 65K test (final validation)**

Run: `timeout 3600 pytest models/demos/blackhole/qwen3_5_9b/tests/test_seq_len.py -v -s -k "seq65k" --timeout=3600 2>&1 | tail -30`

Expected: PASS — this is the end goal.

- [ ] **Step 5: Run the existing e2e test for decode regression**

Run: `timeout 600 pytest models/demos/blackhole/qwen3_5_9b/tests/test_model_e2e.py -v -s --timeout=300 2>&1 | tail -30`

Expected: PASS — decode path is untouched, should not regress.
