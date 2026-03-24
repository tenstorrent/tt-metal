# Qwen3.5-9B Chunked Prefill Stability & 65K+ Sequence Length Support

**Date:** 2026-03-23
**Status:** Approved
**Goal:** Fix chunked delta rule numerical stability and enable prefill for sequences up to 65,536+ tokens.

## Problem Statement

The Qwen3.5-9B model on Blackhole P150 has two related prefill limitations:

1. **Slow prefill:** DeltaNet layers (24 of 32) use `mode="recurrent"` during prefill, processing tokens sequentially. This is because the chunked parallel path has numerical stability issues at seq_len >= 256.

2. **Sequence length cap at ~3000 tokens:** The QKV projection `[1, T, 4096] x [4096, 8192]` exceeds L1 memory for T > ~3000. No segmentation exists to tile along the sequence dimension.

**Target:** Support seq_len up to 65,536 tokens with chunked (parallel) prefill, PCC > 0.99 against recurrent reference.

## Root Cause: Chunked Delta Rule Precision

In `chunk_gated_delta_rule_ttnn` (ttnn_delta_rule_ops.py, lines 886-903):

```python
decay = g_c @ triu_ones          # cumsum of g values within chunk
L_diff = decay_col - decay_row   # pairwise difference matrix
L_mask = exp(L_diff) * tril_mask # exponentiate differences
```

The `g` values are negative (from `-exp(A_log) * softplus(...)`), so `decay` is a running sum of increasingly negative values. Within a chunk of 128 tokens, cumsum can reach -50 or lower. When computing `L_diff = decay[i] - decay[j]`, float32 rounding on the subtraction of two large-magnitude negative numbers introduces errors amplified by `exp()`. The Neumann series (repeated squaring for R matrix) further amplifies these errors.

## Design

### 1. Fix Chunked Delta Rule Numerical Stability

**File:** `models/experimental/gated_attention_gated_deltanet/tt/ttnn_delta_rule_ops.py`

**Changes to `chunk_gated_delta_rule_ttnn`:**

#### a) Per-chunk decay normalization

After computing `decay = cumsum(g)`, subtract the first element of each chunk:

```python
decay_offset = decay[:, 0:1]  # [batch, 1] — first cumsum value per chunk
decay = decay - decay_offset   # now starts at 0, stays near 0
```

Since `L_diff` only uses differences (`decay[i] - decay[j]`), the L_diff/L_mask computation is mathematically identical — the offset cancels in the subtraction.

#### b) Clamp before exp

Clamp `L_diff_masked` to `[-20, 0]` before exponentiating to prevent overflow/underflow. The choice of -20: `exp(-20) ≈ 2e-9` which is representable in float32 (min positive ≈ 1.4e-45). Since the chunk computation stays in float32, this preserves precision. Values below -20 contribute negligibly to the output.

```python
L_diff_masked = clamp(L_diff * tril_mask, min=-20.0, max=0.0)
L_mask = exp(L_diff_masked) * tril_mask
```

Also clamp `decay_exp` and other exp inputs.

#### c) Complete offset accounting for all five decay usage sites

The normalization changes `decay` from absolute cumsum to chunk-relative cumsum. Here is the complete analysis of every place `decay` or `exp(decay)` appears and whether compensation is needed:

**Site 1: `L_diff = decay_col - decay_row`** (intra-chunk attention mask)
- Offset cancels: `(decay[i] - offset) - (decay[j] - offset) = decay[i] - decay[j]`
- **No compensation needed.** ✓

**Site 2: `decay_exp = exp(decay)` → `k_beta_decay = k_beta_c * decay_exp`** (key scaling for state correction term)
- Before normalization: `decay_exp[t] = exp(cumsum(g)[t])` — absolute decay from chunk start
- After normalization: `decay_exp[t] = exp(cumsum(g)[t] - offset)` — relative decay from position 0 in chunk
- This is used in `k_cumdecay = R @ k_beta_decay` which feeds into `v_prime = k_cumdecay @ S` (correction against incoming state S)
- The incoming state S was scaled by inter-chunk decay. The `decay_exp` here measures how much each key position has decayed *within* the chunk relative to chunk start.
- After normalization, `decay_exp` starts at `exp(0) = 1` for position 0 and decays from there.
- **Compensation needed:** Must multiply `decay_exp` by `exp(decay_offset)` to restore absolute decay. Equivalently, use raw (un-normalized) decay for this term: `decay_exp = exp(decay_raw)` where `decay_raw = decay + decay_offset`.

**Site 3: `decay_i_exp = exp(decay_i)` → `q_decay = q_i * decay_i_exp`** (query scaling for inter-chunk attention)
- This scales queries to measure how much to discount the incoming state S at each position within the chunk.
- Same issue as Site 2: after normalization, the absolute decay information is lost.
- **Compensation needed:** Use `decay_i_exp = exp(decay_i + decay_offset_i)` where `decay_offset_i` is the offset for chunk i. This restores the absolute decay for query scaling.

**Site 4: `decay_last = sum(g_c)` → `exp(decay_last)` for `S = exp(decay_last) * S + state_update`** (inter-chunk state propagation)
- `decay_last` is computed directly from `sum(g_c)` (raw g values), NOT from the normalized `decay`.
- **No compensation needed** — `decay_last` is independent of the normalization. ✓

**Site 5: `decay_diff = decay_last - decay_i` → `k_decay = k_i * exp(decay_diff)`** (key scaling for state update)
- `decay_diff[t] = decay_last - decay[t]` = total chunk decay minus decay at position t
- After normalization: `decay_diff[t] = decay_last_raw - (decay[t] + offset)` ... but `decay_last` is computed from raw g_c sum.
- Actually: `decay_last` (line 966) = `sum(g_c)` which equals the last value of the un-normalized cumsum. And `decay_i` (line 997) = normalized decay for chunk i. So: `decay_diff = sum(g_c) - decay_normalized = sum(g_c) - (decay_raw - offset) = (sum(g_c) - decay_raw) + offset`.
- The un-normalized version would be: `decay_diff_raw = sum(g_c) - decay_raw`. So the normalized version has an extra `+ offset`.
- **Compensation needed:** Subtract `decay_offset` from the `decay_last` used here, OR add `decay_offset` back to `decay_i` before computing the difference. Simplest: `decay_diff = decay_last_for_chunk_i - offset_i - decay_i` which gives the correct raw difference.

**Implementation approach:** Rather than patching each site individually, the cleanest approach is:
1. Compute `decay_raw = cumsum(g)` (as before)
2. Compute `decay_offset = decay_raw[:, 0:1]`
3. Compute `decay_normalized = decay_raw - decay_offset` (used ONLY for L_diff/L_mask)
4. Keep `decay_raw` around for Sites 2, 3, and 5
5. `decay_last` (Site 4) is already independent

This means we store both `decay_normalized` (for the attention mask) and `decay_raw` (for state interactions). The extra memory is just `[batch, chunk_size]` in float32, negligible.

#### d) Reduce default chunk_size

Change default from 128 to 64 to reduce accumulated error within each chunk. This doubles the number of chunks but halves the error accumulation per chunk.

### 2. Switch Prefill to Chunked Mode

**File:** `models/demos/blackhole/qwen3_5_9b/tt/qwen35_decoder.py`

Change line 79:
```python
# Before:
deltanet_mode = "recurrent"
# After:
deltanet_mode = "chunk"
```

This enables parallel processing within chunks during prefill. Decode (T=1) path is unaffected.

### 3. Segmented Prefill for 65K+ Sequences

**File:** `models/demos/blackhole/qwen3_5_9b/tt/qwen35_model.py`

Add `prefill_chunked` method that processes long sequences in segments of `segment_size` tokens (default 2048):

```
prefill(token_ids):
    if T <= segment_size:
        _prefill_single(token_ids)   # existing path
    else:
        prefill_chunked(token_ids, segment_size)
```

**Segmented prefill flow:**
1. Embed all tokens (table lookup, fits in DRAM)
2. For each segment [seg_start, seg_end):
   - Slice embeddings for this segment
   - Compute RoPE for segment positions (absolute positions)
   - Process through all 32 layers:
     - DeltaNet: chunked mode, recurrent state carries over between segments
     - Attention: KV cache accumulates across segments via concat
3. Apply final norm + LM head on last token only

**State continuity:**
- DeltaNet recurrent state `[B, 32, 128, 128]` persists in `self.recurrent_state` between segments — already handled by `Qwen35GatedDeltaNet.forward()` which updates `self.recurrent_state` at line 329.
- DeltaNet conv state `[B, 3, 8192]` persists in `self.conv_state_{q,k,v}` between segments — already handled by `Qwen35GatedDeltaNet.forward()` which updates `self.conv_state_{q,k,v}` at lines 333-335. The first segment starts with `conv_state=None` (zero-padded), and subsequent segments receive the conv state from the previous segment. The `causal_conv1d_ttnn` function already supports `conv_state` input for the FIR path (line 242), and `_causal_conv1d_fir` prepends it correctly (line 168). No changes needed to the conv state mechanism — it works across segments because each `layer.forward()` call stores the updated state as instance attributes.
- Attention KV cache grows via concat across segments. The `Qwen35GatedAttention.forward()` accumulates `past_key`/`past_value` via `ttnn.concat`. At segment 32 of a 65K sequence, the KV cache holds ~64K entries. SDPA program config must handle this — see Section 4.

### 4. Attention Layer Scaling

**Files:** `qwen35_gated_attention.py`, `qwen35_model.py`

- Accept `max_seq_len=65536` in `from_pretrained`
- KV cache at 65K: `[1, 4, 65536, 256] x 2 x 8 layers` = ~1GB in DRAM (fits)
- SDPA streams chunks from DRAM automatically
- SDPA program config: The existing `_get_sdpa_program_config` handles `seq_len >= 8192` with q_chunk=64, k_chunk=64. For 65K KV, this means 1024 k-chunks — should work within SDPA kernel limits but must be verified. If needed, add an explicit tier for seq_len >= 32768 with potentially larger k_chunk.
- Decode SDPA with 65K cache: uses q_chunk=32, k_chunk=64 (1024 k-chunks). Verify no SDPA internal limits are exceeded on Blackhole.
- 4:1 GQA ratio keeps cache manageable (4 KV heads, not 16)

**Peak memory estimate:**
- Weights: ~9GB DRAM
- KV cache (65K, 8 layers): ~1GB DRAM
- DeltaNet recurrent state (24 layers): ~50MB
- Activations per segment (2048 tokens): QKV projection output [1, 2048, 8192] = ~32MB, plus intermediates ~100MB
- Total peak: ~10.5GB — within Blackhole P150 DRAM budget

### 5. RoPE Scaling

**File:** `qwen35_rope.py`

Extend RoPE pre-computation to `max_seq_len` positions (up to 65536). Table size: `[1, 65536, 64]` = ~16MB, negligible. The model's `rope_theta=10,000,000` already supports long sequences.

## Files Changed

| File | Change |
|------|--------|
| `ttnn_delta_rule_ops.py` | Per-chunk normalization, clamp, offset accounting |
| `qwen35_decoder.py` | Switch prefill DeltaNet mode to "chunk" |
| `qwen35_model.py` | Add segmented prefill, increase max_seq_len |
| `qwen35_gated_attention.py` | Verify SDPA configs for long KV |
| `qwen35_rope.py` | Extend RoPE table to max_seq_len |
| `test_seq_len.py` | Add 4096, 8192, 65536 test cases |

## Files NOT Changed

- Decode path (T=1) — completely untouched
- Trace capture infrastructure
- Weight mapping
- MLP
- All existing decode optimizations (fused conv, mega fused, etc.)

## Testing

1. **PCC validation:** Chunked vs recurrent delta rule at seq_len 64, 128, 256, 512, 1024. Target PCC > 0.99. Also test non-multiples of chunk_size (100, 300, 700) to validate the padding path.
2. **Seq len sweep:** Extend test_seq_len.py with 4096, 8192, 65536 cases.
3. **Regression:** Existing cases (32-2024) still pass.
4. **Sanity:** No NaN, non-degenerate output, decode throughput > 1 tok/s.
5. **Cached masks:** Verify that changing chunk_size from 128 to 64 regenerates masks correctly. The `Qwen35GatedDeltaNet.__init__` creates masks at `self.prefill_chunk_size` — this must match the chunk_size used in forward calls.
6. **SDPA smoke test at 65K:** Before the full integration test, run a standalone SDPA test with KV length 65536, q_length 2048, to verify the attention kernel handles 1024 k-chunks without hitting hardware limits on Blackhole.

## Execution Order

1. Fix chunked delta rule numerical stability in `ttnn_delta_rule_ops.py`
2. Validate with PCC test (chunked vs recurrent, PCC > 0.99)
3. Switch prefill to chunked mode in `qwen35_decoder.py`
4. Add segmented prefill in `qwen35_model.py`
5. Extend max_seq_len in attention + RoPE
6. Run full test sweep including 65K

## Risks

- **Chunk size tuning:** 64 may not be the optimal balance between parallelism and stability. May need to test 32 and 128.
- **SDPA at 65K:** The attention layers process 65K KV length during prefill segments. SDPA chunking must handle this without L1 OOM.
- **Memory pressure:** 65K sequence with 1GB KV cache + 9GB weights + activations approaches DRAM limits. May need to verify total memory budget.
