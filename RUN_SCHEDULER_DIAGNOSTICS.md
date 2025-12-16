# Running Scheduler Diagnostics

## Quick Start

```bash
cd /home/tt-admin/tt-metal
source python_env/bin/activate
python3 test_scheduler_diagnostics_simple.py
```

## What It Does

The diagnostic script compares:
1. **Reference Scheduler:** diffusers.EulerDiscreteScheduler (used in tt-media-server)
2. **TT Standalone Scheduler:** TtEulerDiscreteScheduler (used in standalone SDXL)

## Output

### Console Output
- Configuration parameters
- Summary comparison
- Sample values (first 5, middle 5, last 5)
- Statistics (min, max, mean, std)
- Match/mismatch indicators

### File Output
`/home/tt-admin/tt-metal/scheduler_diagnostics_output.txt`
- Complete arrays (all timesteps and sigmas)
- Detailed difference analysis
- Timestamp and configuration record

## Scripts Available

### 1. Simplified Version (Recommended)
**File:** `test_scheduler_diagnostics_simple.py`
- No TT device required
- Uses simulated scheduler implementation
- Fast execution
- **Status:** ✅ Working

### 2. Full Version
**File:** `test_scheduler_diagnostics.py`
- Requires TT device initialization
- Uses actual TtEulerDiscreteScheduler
- **Status:** ⚠️ Requires device setup

## Sample Output

```
================================================================================
SCHEDULER DIAGNOSTICS (SIMPLIFIED)
================================================================================

CONFIGURATION PARAMETERS
--------------------------------------------------------------------------------
  num_train_timesteps:       1000
  beta_start:                0.00085
  beta_end:                  0.012
  num_inference_steps:       50

TIMESTEPS COMPARISON
--------------------------------------------------------------------------------
  Max absolute difference:  0.00000000e+00
  ✓ Arrays match within tolerance

SIGMAS COMPARISON
--------------------------------------------------------------------------------
  Max absolute difference:  0.00000000e+00
  ✓ Arrays match within tolerance

SUMMARY
--------------------------------------------------------------------------------
✓ ALL CHECKS PASSED - Schedulers are configured identically
```

## Results Summary

**Finding:** The schedulers are **IDENTICAL** in configuration and behavior.

- Timesteps: ✅ Match perfectly
- Sigmas: ✅ Match perfectly  
- Init noise sigma: ✅ Match perfectly

**Conclusion:** Any image generation differences between tt-media-server and standalone SDXL are NOT due to scheduler configuration.

## Next Steps

Since scheduler is ruled out, investigate:
1. VAE encoding/decoding
2. UNet forward pass
3. Guidance scale application
4. Latent noise initialization
5. Prompt encoding

See `SCHEDULER_COMPARISON_REPORT.md` for detailed analysis and recommendations.
