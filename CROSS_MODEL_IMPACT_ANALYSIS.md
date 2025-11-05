# Cross-Model Impact Analysis: Multi-Head Attention Reshape Fix

**Date**: 2025-10-31
**Commit**: 4448e84e9b
**Analysis By**: Code Review Team
**Reviewer Concern**: Client questions about impact on GPT-2 and LLaMA models

---

## Executive Summary

The multi-head attention reshape bug fix in commit 4448e84e9b **affects BERT and GPT-2** models but **does not affect LLaMA/TinyLLaMA** models. The fix corrects a critical token-to-head assignment error that was scrambling attention computations.

**Key Findings**:
- ✅ **BERT**: Fixed and validated (100% test pass rate, PCC > 0.9999)
- ⚠️ **GPT-2**: Same bug existed, now fixed (requires validation testing)
- ✅ **LLaMA/TinyLLaMA**: Uses different code path, unaffected
- ✅ **Weight Loading**: No interaction with LLaMA's `unpermute_proj_rows`

---

## Background: The Bug

### Original Bug (Before Fix)

Location: `tt-train/sources/ttml/ops/multi_head_utils.cpp` - `heads_creation()` function

**Incorrect Reshape Sequence**:
```cpp
// WRONG: Reshapes sequence dimension instead of embedding dimension
[B, 1, S, E] → [B, 1, S*H, E/H] → [B, H, S, E/H]
//              ^^^^^^^^^^^^^^^^^
//              This scrambles token assignments!
```

**Problem**: When reshaping from `[B, 1, S*H, E/H]` to `[B, H, S, E/H]`, the operation incorrectly interprets the `S*H` dimension as sequence length, scrambling which tokens go to which heads.

### The Fix (After)

**Correct Reshape Sequence**:
```cpp
// CORRECT: Split embedding dimension, then transpose
Step 1: [B, 1, S, E] → [B, S, E]           // Remove channel dim
Step 2: [B, S, E] → [B, S, H, E/H]         // Split embedding into heads
Step 3: [B, S, H, E/H] → [B, H, S, E/H]   // Transpose to put heads first
```

**Solution**: Explicitly split the embedding dimension first, ensuring each head receives the correct embedding slice from each token, then transpose to get the desired shape.

---

## Client Questions Answered

### Q1: "does it mean it was broken for gpt models too?"

**Answer**: ✅ **YES** - GPT-2 models had the same bug and are now fixed.

#### Evidence

**Code Path Analysis**:
```
GPT-2 Model (tt-train/sources/ttml/models/gpt2.cpp)
  └─→ GPTBlock (tt-train/sources/ttml/modules/gpt_block.hpp:32)
      └─→ MultiHeadAttention (tt-train/sources/ttml/modules/multi_head_attention.cpp)
          └─→ heads_creation() ← OUR FIX APPLIES HERE
```

**File References**:
- `tt-train/sources/ttml/models/gpt2.cpp:110` - Creates GPTBlock
- `tt-train/sources/ttml/modules/gpt_block.hpp:32` - Uses MultiHeadAttention
- `tt-train/sources/ttml/modules/multi_head_attention.cpp` - Calls `heads_creation()`

#### Impact on GPT-2

**Before Fix**:
- Multi-head attention had scrambled token-to-head assignments
- Each attention head received incorrect token embeddings
- This would cause:
  - Reduced model accuracy
  - Poor attention pattern learning
  - Degraded inference quality

**After Fix**:
- Correct token-to-head mapping restored
- Expected to see **significant improvement** in GPT-2 performance
- May explain existing GPT-2 inference issues mentioned by client

#### Client's Original Concern
> "also inference for gpt2s didn't work great I was going to search for a bug"

**Analysis**: This bug was likely contributing to GPT-2 inference problems. Our fix should improve GPT-2 performance.

#### Recommendation
- ⚠️ **CRITICAL**: Run GPT-2 validation tests before and after fix
- Compare inference outputs with reference implementation
- Expected outcome: **IMPROVEMENT** in accuracy metrics

---

### Q2: "same question about grouped_heads_creation"

**Answer**: ✅ **NO** - LLaMA models use a different code path and are unaffected.

#### Evidence

**Code Path Analysis**:
```
LLaMA Model (tt-train/sources/ttml/models/llama.cpp)
  └─→ LlamaBlock (tt-train/sources/ttml/modules/llama_block.hpp:34)
      └─→ GroupedQueryAttention (tt-train/sources/ttml/modules/grouped_query_attention.cpp)
          └─→ grouped_heads_creation() ← USES DIFFERENT IMPLEMENTATION
```

**File References**:
- `tt-train/sources/ttml/models/llama.cpp` - Uses LlamaBlock
- `tt-train/sources/ttml/modules/llama_block.hpp:34` - Uses GroupedQueryAttention
- `tt-train/sources/ttml/modules/grouped_query_attention.cpp` - Calls `grouped_heads_creation()`

#### Implementation Comparison

**grouped_heads_creation() - LLaMA (UNCHANGED)**:
```cpp
// Location: tt-train/sources/ttml/ops/multi_head_utils.cpp
std::tuple<...> grouped_heads_creation(
    const autograd::TensorPtr& qs,
    const autograd::TensorPtr& kvs,
    uint32_t num_heads,
    uint32_t num_groups) {

    // Uses TTNN built-in function (NOT manual reshapes)
    auto [q, k, v] = ttnn::experimental::nlp_create_qkv_heads(
        qs->get_value(),
        kvs->get_value(),
        /*num_q_heads=*/num_heads,
        /*num_kv_heads=*/num_groups,
        /*transpose_k_heads=*/false,
        /*memory_config=*/std::nullopt,
        /*optional_output_tensors=*/std::nullopt);

    // Returns tensors directly from TTNN
    return {create_tensor(q), create_tensor(k), create_tensor(v)};
}
```

**heads_creation() - BERT/GPT-2 (FIXED BY US)**:
```cpp
// Location: tt-train/sources/ttml/ops/multi_head_utils.cpp
std::tuple<...> heads_creation(
    const autograd::TensorPtr& qkv,
    uint32_t num_heads) {

    // Uses manual reshapes (OUR FIX)
    auto q_no_channel = ttnn::reshape(q_flat, {B, S, E});
    auto q_with_heads = ttnn::reshape(q_no_channel, {B, S, H, E/H});
    auto q = ttnn::transpose(q_with_heads, 1, 2);  // ← OUR FIX

    return {create_tensor(q), create_tensor(k), create_tensor(v)};
}
```

#### Key Differences

| Aspect | heads_creation (BERT/GPT-2) | grouped_heads_creation (LLaMA) |
|--------|----------------------------|--------------------------------|
| **Implementation** | Manual reshape + transpose | TTNN built-in `nlp_create_qkv_heads` |
| **Bug Present** | ✅ YES (before fix) | ❌ NO |
| **Our Fix Applied** | ✅ YES | ❌ NO (not needed) |
| **Affected by Change** | ✅ YES | ❌ NO |

#### Recommendation
- ✅ **LLaMA/TinyLLaMA should be safe** - uses different implementation
- Still recommend validation testing to ensure no regression
- Expected outcome: **IDENTICAL** outputs before and after

---

### Q3: "please take a look how I load kv heads for llama"

**Client Provided Code**:
```cpp
static std::vector<float> unpermute_proj_rows(
    const std::vector<float>& w, int64_t rows, int64_t cols, int64_t n_heads) {
    // Reorder rows within each head: [0..D/2-1, D/2..D-1] → interleave → [0, D/2, 1, D/2+1, ...]
    if (rows % n_heads != 0) {
        throw std::runtime_error(
            fmt::format("unpermute_proj_rows: rows {} not divisible by n_heads {}", rows, n_heads));
    }
    const int64_t D = rows / n_heads;  // rows per head
    if (D % 2 != 0) {
        throw std::runtime_error(fmt::format("unpermute_proj_rows: rows per head {} must be even", D));
    }
    std::vector<float> out(w.size());
    for (int64_t h = 0; h < n_heads; ++h) {
        const int64_t head_row0 = h * D;
        const int64_t half = D / 2;
        for (int64_t i = 0; i < half; ++i) {
            const int64_t src_even = head_row0 + i;
            const int64_t src_odd = head_row0 + half + i;
            const int64_t dst_even = head_row0 + (2 * i);
            const int64_t dst_odd = head_row0 + (2 * i + 1);
            std::memcpy(&out[dst_even * cols], &w[src_even * cols], sizeof(float) * cols);
            std::memcpy(&out[dst_odd * cols], &w[src_odd * cols], sizeof(float) * cols);
        }
    }
    return out;
}
```

**Answer**: ✅ **NO INTERACTION** - Weight unpermuting is independent of our fix.

#### Analysis

**What unpermute_proj_rows Does**:
- **Purpose**: Reorders HuggingFace LLaMA weight rows to match TTML's expected layout
- **Operation**: Interleaves first and second halves of each head's rows
  - Input: `[0, 1, 2, ..., D/2-1, D/2, D/2+1, ..., D-1]`
  - Output: `[0, D/2, 1, D/2+1, 2, D/2+2, ...]`
- **When**: During model loading from safetensors

**Where It's Used**:
```cpp
// Location: tt-train/sources/ttml/models/llama.cpp (load_from_safetensors)

// For Q projection weights
if (info.name == layer_pfx + ".self_attn.q_proj.weight") {
    std::vector<float> src = float_vec;
    if (!meta_style) {
        const int64_t rows = info.shape[0];  // num_heads * head_dim
        const int64_t cols = info.shape[1];  // hidden_size
        const int64_t n_h = static_cast<int64_t>(config.num_heads);
        src = unpermute_proj_rows(src, rows, cols, n_h);  // ← WEIGHT UNPERMUTING
    }
    // ... load into model
}

// Similar for K projection weights
```

#### Timeline Comparison

| Stage | unpermute_proj_rows | Our heads_creation Fix |
|-------|-------------------|----------------------|
| **When** | Model loading (once) | Every forward pass |
| **What** | Weight tensors | Activation tensors |
| **Input** | HuggingFace safetensors | QKV concatenated activations |
| **Output** | Reordered weight matrices | Split Q, K, V heads |
| **Data Flow** | File → Memory → Model params | Model forward → Attention → Output |

#### Independence Proof

**Weight Loading Flow**:
```
1. Load weights from safetensors file
2. Apply unpermute_proj_rows to Q/K weight rows
3. Store in model parameters
   ↓
[Model parameters are now fixed in memory]
```

**Forward Pass Flow**:
```
1. Input activations enter model
2. Linear layers use (already loaded) weights
3. heads_creation splits QKV activations  ← OUR FIX
4. Attention computation proceeds
```

**Conclusion**: These operate on different data (weights vs activations) at different times (loading vs inference). **No interaction possible**.

---

### Q4: "please try to run tinyllama with your changes in inference I had identical outputs"

**Answer**: ✅ **EXPECTED** - TinyLLaMA uses `grouped_heads_creation`, which is unaffected.

#### Why Identical Outputs Are Expected

**TinyLLaMA Architecture**:
- Model: LLaMA architecture variant
- Attention: GroupedQueryAttention (GQA)
- Head Creation: `grouped_heads_creation()` using TTNN built-in
- **Not affected by our fix**

**Code Path** (No Change):
```
TinyLLaMA Inference
  └─→ LlamaBlock
      └─→ GroupedQueryAttention
          └─→ grouped_heads_creation()
              └─→ ttnn::experimental::nlp_create_qkv_heads
                  (TTNN library function - unchanged)
```

#### Validation Test Recommendation

**Purpose**: Confirm no unintended side effects

**Test Command**:
```bash
cd tt-train/sources/examples/llm_inference
./main --config configs/tinyllama.yaml --test-mode
```

**Expected Outcome**:
- ✅ Identical logits (or within floating-point tolerance)
- ✅ Identical generated tokens
- ✅ Same perplexity scores
- ✅ No performance regression

**If Outputs Differ**:
1. Check for unintended changes in `grouped_heads_creation()`
2. Verify TTNN `nlp_create_qkv_heads` behavior hasn't changed
3. Review any other multi-head attention utilities

---

### Q5: "also inference for gpt2s didn't work great I was going to search for a bug"

**Answer**: ⚠️ **LIKELY RELATED** - Our fix may have resolved the GPT-2 inference bug.

#### Bug Symptom Analysis

**Client's Issue**: "GPT-2 inference didn't work great"

**Possible Manifestations**:
- Low perplexity scores
- Incoherent generated text
- Poor attention pattern quality
- Divergence from reference implementations

#### How Our Bug Caused This

**Incorrect Token-to-Head Assignment**:
```
Before Fix (WRONG):
Token 0 embedding → Head 0: wrong slice
Token 1 embedding → Head 0: wrong slice
...
Token 0 embedding → Head 1: wrong slice
Token 1 embedding → Head 1: wrong slice

After Fix (CORRECT):
Token 0 embedding → Head 0: dims [0:64]
Token 0 embedding → Head 1: dims [64:128]
Token 0 embedding → Head 2: dims [128:192]
...
```

**Impact on Training/Inference**:
- Each head learned attention patterns on **wrong input features**
- Pre-trained weights expect correct token-head mapping
- Misalignment causes poor inference quality

#### Validation Strategy

**1. Regression Test (Before/After Fix)**:
```bash
# On branch without fix (04d1cf4f3e)
git checkout 04d1cf4f3e
cd tt-train/build
./examples/llm_inference --config gpt2_small.yaml --test > before.log

# On branch with fix (4448e84e9b)
git checkout 4448e84e9b
cd tt-train/build
./examples/llm_inference --config gpt2_small.yaml --test > after.log

# Compare outputs
diff before.log after.log
```

**2. Reference Comparison**:
```bash
# Compare against HuggingFace GPT-2
python compare_gpt2_outputs.py \
  --ttml-model path/to/model \
  --hf-model gpt2 \
  --prompt "The quick brown fox" \
  --max-length 50
```

**3. Metrics to Track**:
- Perplexity on validation set
- BLEU/ROUGE scores on generation tasks
- Token-level accuracy
- Attention pattern visualization

#### Expected Improvement

**Hypothesis**: Our fix should **improve** GPT-2 inference quality.

**Rationale**:
1. Bug affected all MHA computations
2. Pre-trained GPT-2 weights assume correct head splitting
3. Fixing the bug aligns runtime behavior with training expectations
4. Should see better coherence and accuracy

---

## Blast Radius Summary

### Models Affected by Fix

| Model | Attention Type | Implementation | Bug Present | Fix Applied | Testing Status |
|-------|---------------|----------------|-------------|-------------|----------------|
| **BERT** | MultiHeadAttention | `heads_creation()` manual | ✅ YES | ✅ YES | ✅ **VALIDATED (100%)** |
| **GPT-2** | MultiHeadAttention | `heads_creation()` manual | ✅ YES | ✅ YES | ⚠️ **NEEDS TESTING** |
| **GPT-2 Distributed** | MultiHeadAttention | `heads_creation()` manual | ✅ YES | ✅ YES | ⚠️ **NEEDS TESTING** |

### Models NOT Affected by Fix

| Model | Attention Type | Implementation | Reason |
|-------|---------------|----------------|--------|
| **LLaMA** | GroupedQueryAttention | `grouped_heads_creation()` TTNN | Uses TTNN built-in function |
| **TinyLLaMA** | GroupedQueryAttention | `grouped_heads_creation()` TTNN | Uses TTNN built-in function |
| **LLaMA Distributed** | GroupedQueryAttention | `grouped_heads_creation()` TTNN | Uses TTNN built-in function |

---

## Code Path Mapping

### BERT (Fixed ✅)
```
ttml::models::bert::Transformer
  └─→ ttml::modules::BertBlock (tt-train/sources/ttml/modules/bert_block.cpp)
      └─→ ttml::modules::MultiHeadAttention
          └─→ ttml::ops::heads_creation() ← FIX APPLIED
              ├─→ Manual reshape sequence
              └─→ Fixed: [B,1,S,E] → [B,S,E] → [B,S,H,E/H] → [B,H,S,E/H]
```

### GPT-2 (Fixed ✅, Needs Testing ⚠️)
```
ttml::models::gpt2::Transformer
  └─→ ttml::modules::GPTBlock (tt-train/sources/ttml/modules/gpt_block.cpp)
      └─→ ttml::modules::MultiHeadAttention
          └─→ ttml::ops::heads_creation() ← FIX APPLIED
              ├─→ Manual reshape sequence
              └─→ Fixed: [B,1,S,E] → [B,S,E] → [B,S,H,E/H] → [B,H,S,E/H]
```

### LLaMA/TinyLLaMA (Unaffected ✅)
```
ttml::models::llama::Llama
  └─→ ttml::modules::LlamaBlock (tt-train/sources/ttml/modules/llama_block.cpp)
      └─→ ttml::modules::GroupedQueryAttention
          └─→ ttml::ops::grouped_heads_creation() ← NO CHANGE
              └─→ ttnn::experimental::nlp_create_qkv_heads (TTNN library)
```

---

## Technical Details

### The Bug in Detail

**Root Cause**: Ambiguous reshape semantics when collapsing/expanding multi-dimensional tensors.

**Original Code** (Buggy):
```cpp
// Shape: [B, 1, S, E]
auto q_flat = ttnn::slice(qkv_val, {0,0,0,0}, {B,1,S,E}, {1,1,1,1});

// BUG: This interprets the collapsed dimension as sequence!
auto q_reshaped = ttnn::reshape(q_flat, {B, 1, S*H, E/H});
//                                           ^^^^^^^^^
//                                           Collapses S and H incorrectly

// Now reshaping to [B, H, S, E/H] distributes wrong dimension
auto q = ttnn::reshape(q_reshaped, {B, H, S, E/H});
```

**What Happened**:
- `[B, 1, S, E]` → `[B, 1, S*H, E/H]`: Collapses last two dims
- Memory layout: `[tok0_emb, tok1_emb, ..., tokS_emb]` each with E elements
- Reshape interprets `S*H` as sequence length
- Result: When expanding to `[B, H, S, E/H]`, tokens are distributed incorrectly across heads

**Fixed Code**:
```cpp
// Shape: [B, 1, S, E]
auto q_flat = ttnn::slice(qkv_val, {0,0,0,0}, {B,1,S,E}, {1,1,1,1});

// CORRECT: Remove channel dimension first
auto q_no_channel = ttnn::reshape(q_flat, {B, S, E});

// Split embedding dimension into heads
auto q_with_heads = ttnn::reshape(q_no_channel, {B, S, H, E/H});
//                                                   ^    ^  ^^^^
//                                                   Keep S intact, split E

// Transpose to put heads before sequence
auto q = ttnn::transpose(q_with_heads, 1, 2);  // [B, S, H, E/H] → [B, H, S, E/H]
```

**Why This Works**:
- Explicitly creates `[B, S, H, E/H]` with correct dimension semantics
- Each token (S dimension) has its embedding (E) split across H heads
- Transpose then moves heads to the correct position
- Preserves token-to-head correspondence

### Memory Layout Visualization

**Example**: B=1, S=4 tokens, H=2 heads, E=8 embedding dim

**Before Fix (WRONG)**:
```
Input: [1, 1, 4, 8] - 4 tokens, 8-dim embeddings
       t0: [e0, e1, e2, e3, e4, e5, e6, e7]
       t1: [e0, e1, e2, e3, e4, e5, e6, e7]
       t2: [e0, e1, e2, e3, e4, e5, e6, e7]
       t3: [e0, e1, e2, e3, e4, e5, e6, e7]

After reshape to [1, 1, 8, 4]:  ← WRONG! Treats 8 as sequence length
       Memory is reinterpreted as 8 sequences of 4 elements

After reshape to [1, 2, 4, 4]:  ← Token assignments scrambled!
       Head 0: Gets wrong token slices
       Head 1: Gets wrong token slices
```

**After Fix (CORRECT)**:
```
Input: [1, 1, 4, 8] - 4 tokens, 8-dim embeddings
       t0: [e0, e1, e2, e3, e4, e5, e6, e7]
       t1: [e0, e1, e2, e3, e4, e5, e6, e7]
       t2: [e0, e1, e2, e3, e4, e5, e6, e7]
       t3: [e0, e1, e2, e3, e4, e5, e6, e7]

Remove channel: [1, 4, 8]
       t0: [e0, e1, e2, e3, e4, e5, e6, e7]
       t1: [e0, e1, e2, e3, e4, e5, e6, e7]
       t2: [e0, e1, e2, e3, e4, e5, e6, e7]
       t3: [e0, e1, e2, e3, e4, e5, e6, e7]

Split heads: [1, 4, 2, 4]  ← Each token's embedding split into 2 heads
       t0_h0: [e0, e1, e2, e3]    t0_h1: [e4, e5, e6, e7]
       t1_h0: [e0, e1, e2, e3]    t1_h1: [e4, e5, e6, e7]
       t2_h0: [e0, e1, e2, e3]    t2_h1: [e4, e5, e6, e7]
       t3_h0: [e0, e1, e2, e3]    t3_h1: [e4, e5, e6, e7]

Transpose: [1, 2, 4, 4]  ← Heads first, then tokens
       Head 0: [t0, t1, t2, t3] all with [e0,e1,e2,e3]
       Head 1: [t0, t1, t2, t3] all with [e4,e5,e6,e7]
```

---

## Recommendations

### Immediate Actions

#### 1. GPT-2 Validation (HIGH PRIORITY ⚠️)
- **Owner**: @Roman Furko (covering for original reviewer)
- **Timeline**: Before merge
- **Tasks**:
  - [ ] Run GPT-2 inference tests before/after fix
  - [ ] Compare with HuggingFace reference implementation
  - [ ] Measure perplexity improvement
  - [ ] Validate generated text quality
  - [ ] Document results

#### 2. TinyLLaMA Regression Test (MEDIUM PRIORITY)
- **Owner**: @Roman Furko or team member
- **Timeline**: Before merge
- **Tasks**:
  - [ ] Run TinyLLaMA inference tests
  - [ ] Verify identical outputs (within FP tolerance)
  - [ ] Confirm no performance regression
  - [ ] Document "no change" result

#### 3. LLaMA Model Testing (LOW PRIORITY)
- **Owner**: Team
- **Timeline**: Post-merge validation
- **Tasks**:
  - [ ] Run full LLaMA test suite
  - [ ] Verify no unexpected side effects
  - [ ] Update documentation

### Documentation Updates

#### 1. Update Commit Message
Add section:
```markdown
## Cross-Model Impact

This fix applies to BERT and GPT-2 models using MultiHeadAttention.
LLaMA/TinyLLaMA models use GroupedQueryAttention with a different
implementation and are not affected.

GPT-2 Validation: [Link to test results]
TinyLLaMA Validation: [Link to test results]
```

#### 2. Update PR Description
Add warning:
```markdown
⚠️ **Breaking/Fixing Change for GPT-2**

This fix corrects a critical bug in multi-head attention that affected
BERT and GPT-2 models. GPT-2 inference quality should improve significantly.

Validation required before merge.
```

---

## Conclusion

The multi-head attention reshape fix in commit 4448e84e9b addresses a fundamental bug affecting BERT and GPT-2 models while leaving LLaMA-family models unaffected due to their different implementation path.

**Risk Assessment**:
- **BERT**: ✅ Low risk - fully validated with 100% test pass rate
- **GPT-2**: ⚠️ Medium risk - fix should improve quality but needs validation
- **LLaMA**: ✅ Very low risk - different code path, no changes

**Recommendation**: **PROCEED WITH MERGE** after GPT-2 validation confirms improvement or no regression.

---

## Appendix: File References

### Modified Files in Commit 4448e84e9b
- `tt-train/sources/ttml/ops/multi_head_utils.cpp` - Contains the fix
- `tt-train/sources/ttml/ops/multi_head_utils.hpp` - Function declarations
- `tt-train/sources/ttml/nanobind/nb_util.cpp` - Non-contiguous array fix
- `tt-train/sources/ttml/ops/scaled_dot_product_attention.cpp` - Masking fix

### Related Files (Not Modified)
- `tt-train/sources/ttml/modules/multi_head_attention.cpp` - Calls heads_creation()
- `tt-train/sources/ttml/modules/grouped_query_attention.cpp` - Calls grouped_heads_creation()
- `tt-train/sources/ttml/models/gpt2.cpp` - Uses MultiHeadAttention
- `tt-train/sources/ttml/models/llama.cpp` - Uses GroupedQueryAttention

### Test Files
- `tt-train/tests/model/bert_operator_test.cpp` - BERT validation (100% pass)
- `tt-train/tests/python/test_bert_*.py` - Python validation (100% pass)
- `tt-train/tests/model/nano_gpt_test.cpp` - GPT/LLaMA training tests
- `tt-train/sources/examples/llm_inference/main.cpp` - Inference testing

---

**Generated**: 2025-10-31
**Commit**: 4448e84e9b
**Branch**: ivoitovych/bert-model-for-ttml
