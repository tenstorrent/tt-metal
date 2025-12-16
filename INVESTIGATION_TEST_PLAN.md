# SDXL SSIM Investigation - Test Execution Plan

## Current State (As of Dec 9, 00:30 UTC)

### Code Changes Applied

#### Step 1.1: Debug Logging ✓ COMPLETE
- **File**: `models/experimental/stable_diffusion_xl_base/tests/test_common.py`
- **Changes**: Added 3 logger.info statements at lines 643, 644, 651, 695
- **Status**: Backup at `test_common.py.backup` (current version WITH changes)
- **Purpose**: Verify which path executes when `guidance_rescale_value=0.0`

#### Step 1.2: VAE Host Mode ✓ COMPLETE
- **File**: `sdxl_config.py`
- **Change**: Line 36: `vae_on_device: bool = False` (was True)
- **Status**: APPLIED (to test precision impact)
- **Purpose**: Test if device VAE (bfloat16) precision is the issue
- **Rollback**: Change back to `vae_on_device: bool = True`

#### Step 1.3: Pre-Rescale Revert ○ PREPARED
- **Files**:
  - Pre-rescale `test_common.py` obtained via: `git show 12aadaad7f^:...`
  - Pre-rescale `tt_sdxl_pipeline.py` obtained via: `git show 12aadaad7f^:...`
- **Status**: Files are in working directory (39K test_common.py -> needs restore)
- **Purpose**: Test if commit 12aadaad7f introduced the regression
- **To Execute**:
  ```bash
  # Restore pre-rescale versions
  git show 12aadaad7f^:models/experimental/stable_diffusion_xl_base/tests/test_common.py > models/experimental/stable_diffusion_xl_base/tests/test_common.py
  git show 12aadaad7f^:models/experimental/stable_diffusion_xl_base/tt/tt_sdxl_pipeline.py > models/experimental/stable_diffusion_xl_base/tt/tt_sdxl_pipeline.py
  # Also need to restore config
  sed -i 's/vae_on_device: bool = False/vae_on_device: bool = True/' sdxl_config.py
  ```

---

## Test Execution Sequence

### Phase 1: Pinpoint the Issue

#### Test 1.1+1.2: Current Version + Debug Logging + VAE Host
**When Server is Ready**:
```bash
cd /home/tt-admin/tt-metal

# Ensure correct state
cp test_common.py.backup models/experimental/stable_diffusion_xl_base/tests/test_common.py
cp tt_sdxl_pipeline.py.backup models/experimental/stable_diffusion_xl_base/tt/tt_sdxl_pipeline.py
sed -i 's/vae_on_device: bool = True/vae_on_device: bool = False/' sdxl_config.py

# Run test
/home/tt-admin/tt-metal/python_env/bin/python image_test.py \
  "Photograph of an orange Volcano on a tropical island while someone suntans on a beach with a friendly dinosaur" \
  --compare /home/tt-admin/tt-inference-server/reference_image.jpg \
  --guidance 12.0 \
  --rescale 0.0 \
  --output step_1_1_1_2_test.jpg 2>&1 | tee step_1_1_1_2_test.log

# Extract SSIM score
grep -i "ssim" step_1_1_1_2_test.log
```

**Success Criteria**:
- Log contains: `DEBUG: Executing SIMPLE path (guidance_rescale == 0.0)` (simple path is executed)
- SSIM value reported (compare to threshold 0.9)

**Expected Outcomes**:
- If SSIM >= 0.9: VAE precision was the issue
- If SSIM < 0.9: Issue is elsewhere (trace, tensor handling, or rescale)

---

#### Test 1.3: Pre-Rescale Version (No Guidance Rescale Feature)
**After Test 1.1+1.2**:
```bash
cd /home/tt-admin/tt-metal

# Switch to pre-rescale versions
git show 12aadaad7f^:models/experimental/stable_diffusion_xl_base/tests/test_common.py > models/experimental/stable_diffusion_xl_base/tests/test_common.py
git show 12aadaad7f^:models/experimental/stable_diffusion_xl_base/tt/tt_sdxl_pipeline.py > models/experimental/stable_diffusion_xl_base/tt/tt_sdxl_pipeline.py

# Restore VAE to True
sed -i 's/vae_on_device: bool = False/vae_on_device: bool = True/' sdxl_config.py

# Run test (NO --rescale parameter since old code doesn't support it)
/home/tt-admin/tt-metal/python_env/bin/python image_test.py \
  "Photograph of an orange Volcano on a tropical island while someone suntans on a beach with a friendly dinosaur" \
  --compare /home/tt-admin/tt-inference-server/reference_image.jpg \
  --guidance 12.0 \
  --output step_1_3_revert_test.jpg 2>&1 | tee step_1_3_revert_test.log

# Extract SSIM score
grep -i "ssim" step_1_3_revert_test.log
```

**Success Criteria**:
- SSIM value reported

**Expected Outcomes**:
- If SSIM >= 0.9: Commit 12aadaad7f (rescale) caused regression ← ROOT CAUSE FOUND
- If SSIM < 0.9: Issue predates rescale feature (unlikely)

---

## Phase 1 Decision Matrix

After both tests complete:

| Test 1.1+1.2 Result | Test 1.3 Result | Path Executed | Conclusion | Next Phase |
|-------------------|-----------------|---------------|-----------|-----------|
| SSIM >= 0.9 | N/A | Simple path | VAE precision was causing degradation | Skip Phase 2, restore VAE=True as fix |
| SSIM < 0.9 | SSIM >= 0.9 | Simple path | Rescale feature introduced regression | Phase 2: Fix rescale conditional/logic |
| SSIM < 0.9 | SSIM < 0.9 | Simple path | Issue predates rescale (unlikely) | Widen investigation |
| Full path executes | Any | Full path | Conditional logic bug (rescale=0.0 but full path runs) | Phase 2: Fix conditional check |

---

## Phase 2: Verify Root Cause (Conditional)

### Step 2.1: Verify ttnn_safe_std() Equivalence
**Only if**: Full rescale path is executing or rescale math is suspected

```bash
# Create test script (already prepared in plan)
cat > /tmp/test_std_equiv.py << 'EOF'
# [Test script from plan]
EOF

python /tmp/test_std_equiv.py
```

---

### Step 2.2: Check Trace Capture Consistency
**Only if**: Trace-related issues suspected

```bash
# Disable trace
sed -i 's/capture_trace: bool = True/capture_trace: bool = False/' sdxl_config.py

# Restart server and run test
/home/tt-admin/tt-metal/python_env/bin/python image_test.py \
  "Photograph of an orange Volcano..." \
  --compare /home/tt-admin/tt-inference-server/reference_image.jpg \
  --guidance 12.0 \
  --rescale 0.0 \
  --output step_2_2_notrace_test.jpg 2>&1 | tee step_2_2_notrace_test.log

# Restore trace
sed -i 's/capture_trace: bool = False/capture_trace: bool = True/' sdxl_config.py
```

---

## Rollback Procedures

### Full Rollback to Original State
```bash
cd /home/tt-admin/tt-metal

# Restore all original files
cp /path/to/git_HEAD_versions/* .

# Or manually:
git show HEAD:models/experimental/stable_diffusion_xl_base/tests/test_common.py > models/experimental/stable_diffusion_xl_base/tests/test_common.py
git show HEAD:models/experimental/stable_diffusion_xl_base/tt/tt_sdxl_pipeline.py > models/experimental/stable_diffusion_xl_base/tt/tt_sdxl_pipeline.py
git show HEAD:sdxl_config.py > sdxl_config.py

# Cleanup
rm -f test_common.py.backup tt_sdxl_pipeline.py.backup
rm -f step_1_*.log step_1_*.jpg
rm -f step_2_*.log step_2_*.jpg
```

### Quick Rollback (Just Code)
```bash
# Restore to current HEAD without git
# (If git checkout is unavailable)

# VAE: Restore to True
sed -i 's/vae_on_device: bool = False/vae_on_device: bool = True/' sdxl_config.py

# Remove debug logging from test_common.py
# (Manual edit or use backup)
```

---

## Files and Their Current States

| File | Current State | Backup Location | Git Ref |
|------|---------------|-----------------|---------|
| `test_common.py` | WITH debug logging | `test_common.py.backup` | HEAD (with changes) |
| `tt_sdxl_pipeline.py` | Current (rescale support) | `tt_sdxl_pipeline.py.backup` | HEAD |
| `sdxl_config.py` | VAE = False | N/A | Manually modified |

Pre-rescale versions available via:
- `git show 12aadaad7f^:models/experimental/.../test_common.py`
- `git show 12aadaad7f^:models/experimental/.../tt_sdxl_pipeline.py`

---

## Next Steps When Server is Ready

1. **Verify server health**: `curl http://127.0.0.1:8000/health`
2. **Run Test 1.1+1.2**: Execute combined test (already prepared above)
3. **Analyze logs**: Check for SSIM value and path execution message
4. **Run Test 1.3**: Execute pre-rescale version if needed
5. **Determine root cause**: Use decision matrix above
6. **Execute Phase 2** (if needed based on Phase 1 results)

---

## Investigation Status

- [x] Code changes prepared (Steps 1.1 and 1.2)
- [x] Test scripts written
- [ ] Server available for testing
- [ ] Test 1.1+1.2 executed
- [ ] Test 1.3 executed
- [ ] Phase 1 decision made
- [ ] Phase 2 (if needed) executed
- [ ] Fix implemented
- [ ] Fix verified with 3x SSIM >= 0.9

---

**Created**: 2025-12-09 00:30 UTC
**Status**: Awaiting server availability for test execution
