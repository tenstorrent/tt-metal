# Plan: Fix OLMo Prefill PCC

## Problem Statement
- 1L prefill: full-tensor hidden state PCC = 0.999, but **position 6 PCC = 0.942**
- 64L prefill: hidden state PCC = 0.962, magnitude inflation 1.43x
- E2E demo: **incoherent output** (stuck generating same token)
- bfloat16 weights + matmul output: **zero improvement** → NOT a precision issue, likely a BUG

## Key Insight from User
> "precision is not the reason for degradation. There is some bug.
> Also, check the previous stages with high PCC if there is any padding
> because it might be masking some issue"

## Critical Discovery: RoPE Bug (Separate from PCC)
Both reference AND TTNN use **uniform scaling** (`inv_freq / factor`) for ALL frequency dims.
HuggingFace correctly blends between unscaled (high-freq) and scaled (low-freq) via YaRN mask.
This is **wrong** and likely causes incoherent E2E output — but doesn't explain PCC gap
(since both reference and TTNN have the same bug, their outputs should still match).

---

## Phase 1: Fix RoPE (Both Reference and TTNN)

### Step 1.1: Fix reference model RoPE
File: `reference/olmo.py`, line 127
```python
# WRONG (current):
inv_freq = inv_freq_scaled * inv_freq_mask + inv_freq_scaled * (1 - inv_freq_mask)

# CORRECT (matching HuggingFace):
inv_freq = inv_freq * inv_freq_mask + inv_freq_scaled * (1 - inv_freq_mask)
```

### Step 1.2: Fix TTNN RoPE
File: `tt/llama_common.py`, `compute_yarn_inv_freq()`, line 126
```python
# WRONG (current):
inv_freq = inv_freq_scaled

# CORRECT:
inv_freq = inv_freq * inv_freq_mask + inv_freq_scaled * (1 - inv_freq_mask)
```

### Step 1.3: Verify reference matches HuggingFace
Run a quick test: tokenize "What is your favorite condiment?", run BOTH:
- Our fixed reference model
- HuggingFace `AutoModelForCausalLM.from_pretrained("allenai/OLMo-2-0325-32B")`
Compare logits. They should match with PCC > 0.99.

### Step 1.4: After RoPE fix, re-run 1L prefill PCC
The RoPE fix changes BOTH reference and TTNN equally, so PCC might stay the same.
But if one of them was using a different code path for RoPE, PCC could improve.

---

## Phase 2: Per-Position Per-Op PCC (1 Layer) — Find the Bug

The fact that bfloat16 made zero difference means there's a functional bug, not precision.
The "full tensor PCC = 0.999" is inflated by 121 padding positions.
We need to check **real-token PCC at every intermediate stage**.

### Step 2.1: Enhanced per-op test with per-position breakdown
For each captured intermediate in the 1L prefill path, compute:
- **Full tensor PCC** (all 128 positions)
- **Real tokens PCC** (positions 0-6 only)
- **Padding tokens PCC** (positions 7-127 only)
- **Per-position PCC** for positions 0, 1, 6 (worst real), 64, 127 (padding)
- **max_abs_err** and **std ratio** for real vs padding separately

Intermediates to capture (in order):
1. **Embedding output** — is the embedding identical?
2. **QKV matmul output** (before head split) — is the projection identical?
3. **Q after create_heads** — are heads split correctly?
4. **K after create_heads**
5. **V after create_heads**
6. **Q after QK-norm** — is distributed norm identical per-position?
7. **K after QK-norm**
8. **Q after RoPE** — are rotations identical per-position?
9. **K after RoPE**
10. **SDPA output** — is attention computation identical per-position?
11. **WO matmul output** — is projection identical?
12. **WO after all_reduce** — is collective communication correct?
13. **attn_normed** (post-attention norm) — is norm correct?
14. **h_attn** (residual after attention) — where does error appear?
15. **ff_out** (FFN output)
16. **ff_normed** (post-FFN norm)
17. **layer_out** (final hidden state)

### Step 2.2: Identify the error origin
The step where **real-token PCC first drops significantly** (while padding stays high)
reveals the bug location. Possible outcomes:

- If error starts at **QK-norm**: distributed norm bug for specific positions
- If error starts at **RoPE**: rotation angle or format mismatch per position
- If error starts at **SDPA**: attention mask or computation issue
- If error starts at **WO all_reduce**: collective communication bug
- If error starts at **post-norm**: norm computation or residual add bug

---

## Phase 3: Check Padding Contamination at Each Stage

### Step 3.1: Is padding inflating PCC at earlier stages?
Even if "attn_out PCC = 0.999", that might be 121 padding positions at PCC=0.9999
masking 7 real positions at PCC=0.95.

For EACH intermediate captured in Step 2.1:
- Report `PCC_real` and `PCC_pad` separately
- If `PCC_real << PCC_pad`, padding is masking the issue

### Step 3.2: Check reference model handles padding correctly
- The reference creates causal mask but NO padding mask
- TTNN uses `is_causal=True` but NO padding mask
- These should be equivalent... but verify by checking attention weights
  at position 6: does it attend to padding positions 7-127?
  (It shouldn't with a causal mask, but double-check)

### Step 3.3: Check if TTNN distributed ops accidentally include padding
- Distributed QK-norm: does `rms_norm_pre_all_gather` compute stats over
  all 128 positions or per-position? (Should be per-position over hidden dim)
- RMSNorm: is it applied per-position? (Should be — but verify for
  the distributed case where hidden dim is sharded across 8 devices)

---

## Phase 4: Verify RoPE Format Compatibility

### Step 4.1: Compare RoPE output directly
After fixing the frequency computation (Phase 1), verify that the ACTUAL
rotation applied to Q and K produces identical results:

```python
# Extract Q_pre_rope and Q_post_rope from both reference and TTNN
# For EACH position, compare: PCC(ref_Q_post_rope[pos], tt_Q_post_rope[pos])
```

If RoPE PCC is already low per-position, the bug is in the rotation application:
- Reference uses HF format (first half real, second half imag)
- TTNN uses `gather_cos_sin` (duplicated cos/sin) + `rotary_embedding_llama`
- The `transformation_mats["prefill"]` should convert between formats — verify

### Step 4.2: Check cos/sin values directly
```python
# Compare the cos/sin matrices at each position
ref_cos = real(freqs_cis)  # from reference
tt_cos = precompute_freqs_yarn(...)  # from TTNN
# They should be identical if both use the same frequency formula
```

---

## Phase 5: Layer Accumulation Analysis

### Step 5.1: Track per-position PCC across layers
Run prefill with 1, 4, 16, 64 layers. At each count, report:
- Position 0 PCC, Position 6 PCC, Position 64 PCC (padding)
- std ratio (tt/ref) for position 6

This reveals:
- Is the error **additive** (constant per layer) or **multiplicative** (compounding)?
- Does it affect ALL positions or just specific ones?

### Step 5.2: Check if magnitude inflation is systematic
The 64L test showed 43% magnitude inflation. Check:
- At which layer does the inflation start?
- Is it concentrated in attention or FFN?
- Is it per-position or uniform?

---

## Phase 6: Quick Sanity Checks

### Step 6.1: Embedding verification
Compare TTNN embedding output vs reference for the 128-token input.
PCC should be > 0.9999. If not, there's a bug in the embedding layer itself.

### Step 6.2: Single-position test
Create a test with seq_len=1 (just one token, no padding at all).
Run 1L prefill. PCC should be very high (> 0.999). If it's low, the bug
is NOT padding-related.

### Step 6.3: Check norm weight correctness
Verify that `self.ff_norm` and `self.attention_norm` in the decoder load
the correct HuggingFace weights:
- `ff_norm` should have `post_attention_layernorm` weights
- `attention_norm` should have `post_feedforward_layernorm` weights
(These are swapped due to the OLMo weight mapping — verify they're correct)

---

## Execution Order
1. **Phase 1** (RoPE fix) — fixes E2E incoherence, may or may not fix PCC
2. **Phase 6** (sanity checks) — quick tests to narrow down
3. **Phase 2** (per-position per-op) — find the exact bug location
4. **Phase 3** (padding analysis) — check if padding masks issues
5. **Phase 4** (RoPE format) — if Phase 2 points to RoPE
6. **Phase 5** (layer accumulation) — understand compounding after bug fix
