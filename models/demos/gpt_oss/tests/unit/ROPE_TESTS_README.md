# GPT-OSS RoPE Unit Tests

This document explains the RoPE (Rotary Position Embedding) unit tests and the bug fix they helped identify.

## Overview

The `test_rope.py` file contains unit tests that compare our RoPE implementation against the HuggingFace transformers reference implementation. These tests were instrumental in identifying a bug in our Yarn RoPE scaling implementation.

## Test Descriptions

### Test 1: `test_rope_vs_hf_reference`

**Purpose:** Tests the full TTNN-based pipeline that the actual model uses at runtime.

**Code path tested:**
```
create_rope_setup() → RotarySetup → get_rot_mats() → TTNN tensor conversion
```

**What it tests:**
- `models/demos/gpt_oss/tt/model.py` → `create_rope_setup()` function
- `models/tt_transformers/tt/rope.py` → `RotarySetup` class and `get_rot_mats()`
- TTNN conversion: `ttnn.from_torch()` and mesh mapping
- bfloat16 precision handling

### Test 2: `test_rope_pytorch_vs_hf_reference`

**Purpose:** Tests only the PyTorch implementation without any TTNN conversion, allowing isolation of math bugs.

**Code path tested:**
```
rotary_embedding_factory() → YarnRotaryEmbedding → Pure PyTorch tensors
```

**What it tests:**
- `models/tt_transformers/tt/rope.py` → `rotary_embedding_factory()` and `YarnRotaryEmbedding`
- `models/tt_transformers/tt/common.py` → `rope_scaling_model_factory()` and `RopeScalingYarn`
- Pure PyTorch math (float32)

### Test 3: `test_rope_embedding_lookup_multi_user`

**Purpose:** Tests the full embedding lookup mechanism (`get_rot_mats`) for multiple users at different positions.

**What it tests:**
- `models/tt_transformers/tt/rope.py` → `RotarySetup.get_rot_mats()` method
- `ttnn.embedding` lookup operation
- Correct cos/sin values returned for each user's position

**Why it's important:** The main test (`test_rope_vs_hf_reference`) only verifies the precomputed lookup tables match, but doesn't verify the actual embedding lookup works correctly for multiple users at different positions. This test fills that gap by:
1. Creating a RotarySetup with batch_size 1 or 32
2. Creating position indices spread across the sequence (0, 248, 496, ...)
3. Calling `get_rot_mats()` to lookup cos/sin for those positions
4. Comparing each user's returned values against HuggingFace reference

### Test 4: `test_rope_scaling_parameters`

**Purpose:** Verifies that rope scaling parameters are correctly parsed from the HuggingFace config.

## Why Have Both Tests?

| Aspect | Test 1 (TTNN) | Test 2 (PyTorch) |
|--------|---------------|------------------|
| Tests TTNN conversion | ✅ | ❌ |
| Tests bfloat16 precision | ✅ | ❌ (float32) |
| Tests mesh device handling | ✅ | ❌ |
| Faster to run | ❌ | ✅ |
| Isolates PyTorch math bugs | ❌ | ✅ |

**Key insight:** If Test 2 passes but Test 1 fails, the bug is in the TTNN conversion layer. If both fail with identical PCCs, the bug is in the core PyTorch implementation.

## Bug Found and Fixed

### Symptom
- Tests at seq_len=128 and seq_len=1024 passed (PCC > 0.999)
- Tests at seq_len=4096 failed (PCC ≈ 0.979)
- Both Test 1 and Test 2 failed with nearly identical PCC values

### Root Cause
The GPT-OSS model config has `truncate: False` in its `rope_scaling` configuration:
```python
{'beta_fast': 32.0, 'beta_slow': 1.0, 'factor': 32.0,
 'original_max_position_embeddings': 4096, 'rope_type': 'yarn', 'truncate': False}
```

Our `YarnRotaryEmbedding` implementation was missing support for the `truncate` parameter. It always used `math.floor()` and `math.ceil()` on the correction range bounds (equivalent to `truncate=True`), but HuggingFace respects the config value.

### Impact
The difference in the linear ramp mask accumulated over positions:
- With truncation: `low=8, high=18` (integers)
- Without truncation: `low=8.09, high=17.40` (floats)

This caused sign flips in cos/sin values at higher positions (e.g., position 3873), resulting in degraded PCC.

### Fix
Added the `truncate` parameter to:
1. `YarnRotaryEmbedding.__init__()` in `rope.py`
2. `YarnRotaryEmbedding.yarn_find_correction_range()` in `rope.py`
3. `RopeScalingYarn` class in `common.py`

### Results After Fix
All tests now pass with PCC > 0.9999:
- seq_len=4096: PCC improved from ~0.979 to ~0.9999
- seq_len=131072: PCC = 0.9999983

## Running the Tests

```bash
cd /localdev/handrews/tt-metal
source python_env/bin/activate
HF_MODEL=/proj_sw/user_dev/gpt-oss-weights/gpt-oss-20b/ pytest models/demos/gpt_oss/tests/unit/test_rope.py -v
```

## Files Modified

- `models/tt_transformers/tt/rope.py` - Added `truncate` parameter to `YarnRotaryEmbedding`
- `models/tt_transformers/tt/common.py` - Added `truncate` field to `RopeScalingYarn`
