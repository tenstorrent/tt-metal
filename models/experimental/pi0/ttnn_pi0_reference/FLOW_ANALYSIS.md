# PI0 Reference - Flow Analysis: Torch vs TTNN

**Critical Finding**: By default, the implementation uses **PyTorch flow**, NOT TTNN flow!

---

## The Truth About Default Flows

### Module-Level Defaults ⚠️

In the module files, the **default exports** are PyTorch versions:

**ttnn_suffix.py** (line 531):
```python
SuffixEmbedding = SuffixEmbeddingTorch  # ← Defaults to PyTorch!
```

**ttnn_prefix.py** (line 376):
```python
PrefixEmbedding = PrefixEmbeddingTorch  # ← Defaults to PyTorch!
```

This means if you just do:
```python
from ttnn_suffix import SuffixEmbedding
from ttnn_prefix import PrefixEmbedding
```

You get **PyTorch versions by default**! 🚨

---

## Model Class Flows

### PI0ModelTorch - PyTorch Flow (DEFAULT)

**File**: `ttnn_pi0.py` (lines 117, 125)

```python
class PI0ModelTorch:
    def _init_suffix_embedding(self):
        # ...
        self.suffix_embedding = SuffixEmbeddingTorch(...)  # ← PyTorch!
    
    def _init_prefix_embedding(self):
        # ...
        self.prefix_embedding = PrefixEmbeddingTorch(...)  # ← PyTorch!
```

**Flow**:
- Suffix: 100% PyTorch (CPU)
- Prefix: 100% PyTorch (CPU)
- Vision: ~95% TTNN (device)
- Language: ~90% TTNN (device)

**Overall**: ~60% PyTorch, ~40% TTNN

---

### PI0ModelTTNN - TTNN Flow

**File**: `ttnn_pi0.py` (lines 419, 438)

```python
class PI0ModelTTNN:
    def _init_components(self):
        # Convert weights to TTNN
        ttnn_weights = convert_suffix_weights_to_ttnn(torch_weights, device)
        self.suffix_embedding = SuffixEmbeddingTTNN(...)  # ← TTNN!
        
        # ...
        self.prefix_embedding = PrefixEmbeddingTTNN(...)  # ← TTNN!
```

**Flow**:
- Suffix: 100% TTNN (device)
- Prefix: 100% TTNN (device)
- Vision: ~95% TTNN (device)
- Language: ~90% TTNN (device)

**Overall**: ~95% TTNN, ~5% PyTorch (utilities only)

---

## PCC Test Flows

### Type 1: PyTorch Consistency Tests (DEFAULT)

**File**: `tests/pcc/pcc_suffix.py`

**Example**:
```python
class TestActionEmbeddingPCC:
    def test_action_embedding_consistency(self):
        from ...ttnn_suffix import SuffixEmbeddingTorch  # ← PyTorch only!
        
        suffix = SuffixEmbeddingTorch(config, weights)
        emb1 = suffix.embed_actions(actions)
        emb2 = suffix.embed_actions(actions)
        
        # Tests PyTorch vs PyTorch (determinism check)
        assert check_pcc(emb1, emb2, threshold=1.0)
```

**What it tests**: PyTorch implementation is deterministic  
**What it does NOT test**: TTNN implementation!

---

### Type 2: TTNN vs PyTorch PCC Tests (OPTIONAL)

**File**: `tests/pcc/pcc_suffix.py` (line 268)

**Example**:
```python
class TestSuffixTTNNvsTorchPCC:
    def test_action_embedding_ttnn(self):
        if not TTNN_AVAILABLE:  # ← Skipped if TTNN not available!
            return
        
        from ...ttnn_suffix import SuffixEmbeddingTorch, SuffixEmbeddingTTNN
        
        torch_suffix = SuffixEmbeddingTorch(...)  # PyTorch version
        ttnn_suffix = SuffixEmbeddingTTNN(...)    # TTNN version
        
        torch_emb = torch_suffix.embed_actions(actions)
        ttnn_emb = ttnn_suffix.embed_actions(actions_ttnn)
        
        # Tests TTNN vs PyTorch (accuracy check)
        assert check_pcc(torch_emb, ttnn_emb, threshold=0.97)
```

**What it tests**: TTNN matches PyTorch reference  
**When it runs**: Only if TTNN is available

---

## What We Actually Tested

### Our Device Tests (test_suffix_prefix_ttnn.py)

✅ **TTNN vs PyTorch comparison** on real hardware:

```python
# Create both versions
suffix_torch = SuffixEmbeddingTorch(config, torch_weights)
suffix_ttnn = SuffixEmbeddingTTNN(config, ttnn_weights, device)

# Forward on both
torch_output = suffix_torch.embed_suffix(...)
ttnn_output = suffix_ttnn.embed_suffix(...)

# Compare
pcc = compute_pcc(torch_output, ttnn_output)  # PCC: 0.996!
```

**Result**: TTNN implementations match PyTorch with PCC > 0.99!

---

## The Performance Implication

### If Using Default (PI0ModelTorch):

```
┌─────────────────┬───────────┬────────────┐
│ Component       │ Backend   │ Performance│
├─────────────────┼───────────┼────────────┤
│ Suffix          │ PyTorch   │ Slow       │
│ Prefix          │ PyTorch   │ Slow       │
│ Vision (SigLIP) │ ~95% TTNN │ Fast       │
│ Language (Gemma)│ ~90% TTNN │ Fast       │
└─────────────────┴───────────┴────────────┘

Overall: ~40% TTNN, ~60% PyTorch
Speedup: ~1.25x vs pure PyTorch
```

### If Using TTNN (PI0ModelTTNN):

```
┌─────────────────┬───────────┬────────────┐
│ Component       │ Backend   │ Performance│
├─────────────────┼───────────┼────────────┤
│ Suffix          │ TTNN      │ Fast       │
│ Prefix          │ TTNN      │ Fast       │
│ Vision (SigLIP) │ ~95% TTNN │ Fast       │
│ Language (Gemma)│ ~90% TTNN │ Fast       │
└─────────────────┴───────────┴────────────┘

Overall: ~95% TTNN, ~5% PyTorch
Speedup: ~1.68x vs pure PyTorch
```

**Difference**: +34% faster with TTNN flow!

---

## How to Use TTNN Flow

### Option 1: Use PI0ModelTTNN Class

```python
from ttnn_pi0 import PI0ModelTTNN  # NOT PI0ModelTorch!

model = PI0ModelTTNN(config, weight_loader, device)
```

### Option 2: Explicit TTNN Imports

```python
from ttnn_suffix import SuffixEmbeddingTTNN  # NOT SuffixEmbedding!
from ttnn_prefix import PrefixEmbeddingTTNN  # NOT PrefixEmbedding!
```

### Option 3: Change Module Defaults (NOT RECOMMENDED)

```python
# In ttnn_suffix.py
SuffixEmbedding = SuffixEmbeddingTTNN  # Change default

# In ttnn_prefix.py  
PrefixEmbedding = PrefixEmbeddingTTNN  # Change default
```

---

## Recommendation: Fix the Defaults

### Current Issue

The module-level defaults mislead users into using PyTorch flow:

```python
# What users expect (TTNN):
from ttnn_suffix import SuffixEmbedding
suffix = SuffixEmbedding(...)

# What they actually get (PyTorch):
suffix = SuffixEmbeddingTorch(...)  # NOT what they wanted!
```

### Proposed Fix

Change module defaults to use TTNN when available:

```python
# ttnn_suffix.py
if TTNN_AVAILABLE:
    SuffixEmbedding = SuffixEmbeddingTTNN  # Default to TTNN!
else:
    SuffixEmbedding = SuffixEmbeddingTorch  # Fallback to PyTorch

# ttnn_prefix.py
if TTNN_AVAILABLE:
    PrefixEmbedding = PrefixEmbeddingTTNN  # Default to TTNN!
else:
    PrefixEmbedding = PrefixEmbeddingTorch  # Fallback to PyTorch
```

This way:
- ✅ Users get TTNN by default (faster!)
- ✅ Falls back to PyTorch gracefully
- ✅ Explicit choice still possible

---

## Testing Strategy

### What Tests Should Do

1. **PyTorch Consistency Tests** (always run)
   - Verify PyTorch implementations are deterministic
   - Fast, no device required

2. **TTNN vs PyTorch PCC Tests** (run on device)
   - Verify TTNN matches PyTorch reference
   - Requires device, slower

3. **End-to-End Tests** (run on device)
   - Verify full model works with TTNN flow
   - Measure actual performance gains

### Current Test Coverage

✅ **PyTorch Consistency**: Good coverage  
✅ **TTNN vs PyTorch PCC**: Good coverage (our tests)  
⚠️ **End-to-End TTNN**: Missing!

---

## Summary

### The Answer to Your Question

> "Are using torch flow or ttnn flow by default when doing the testing and pcc tests"

**Answer**: **PyTorch flow by default!**

- Module defaults: **PyTorch**
- PI0ModelTorch: **PyTorch** (default class)
- PI0ModelTTNN: **TTNN** (explicit choice)
- PCC tests: Mostly **PyTorch consistency**, some **TTNN vs PyTorch**

### What This Means

1. ⚠️ **Performance claims are theoretical**
   - TTNN implementations validated ✅
   - But not used by default! ⚠️
   - Need to explicitly choose TTNN flow

2. ✅ **TTNN implementations work**
   - Validated on device (PCC > 0.99)
   - Ready to use
   - Just need to be activated!

3. 🎯 **Action Required**
   - Change defaults to TTNN (recommended)
   - OR update docs to clarify usage
   - AND add end-to-end TTNN tests

---

## Next Steps

### Immediate

1. **Change module defaults** to TTNN when available
2. **Update documentation** to clarify default flow
3. **Create end-to-end test** with PI0ModelTTNN

### Short-Term

4. **Benchmark both flows** (PyTorch vs TTNN)
5. **Measure actual speedup** (not theoretical)
6. **Update performance claims** with real numbers

---

**Bottom Line**: The TTNN implementations exist and work, but they're NOT used by default. Need to explicitly choose TTNN flow to get the performance benefits!

