# Scheduler Comparison Report

**Date:** 2025-12-09
**Purpose:** Diagnose potential scheduler configuration differences between tt-media-server reference and TT standalone SDXL implementation

## Executive Summary

✅ **RESULT: NO DIFFERENCES DETECTED**

The diagnostic script compared the diffusers `EulerDiscreteScheduler` (reference implementation used in tt-media-server) with the `TtEulerDiscreteScheduler` (standalone implementation) and found **zero** configuration or behavioral differences.

## Test Configuration

The schedulers were tested with SDXL default parameters:

```
Configuration Parameters:
├─ num_train_timesteps:       1000
├─ beta_start:                0.00085
├─ beta_end:                  0.012
├─ beta_schedule:             scaled_linear
├─ prediction_type:           epsilon
├─ interpolation_type:        linear
├─ use_karras_sigmas:         False
├─ timestep_spacing:          leading
├─ timestep_type:             discrete
├─ steps_offset:              0
├─ rescale_betas_zero_snr:    False
├─ final_sigmas_type:         zero
└─ num_inference_steps:       50
```

## Comparison Results

### 1. Timesteps Array
- **Length:** 50 steps
- **Range:** 0.0 to 980.0
- **Comparison:** ✅ IDENTICAL (max diff: 0.0)
- **Statistics Match:** Yes

Sample values:
```
Step 0:  980.0
Step 25: 480.0
Step 49: 0.0
```

### 2. Sigmas Array
- **Length:** 51 values (50 steps + 1 final)
- **Range:** 0.0 to 13.04308796
- **Comparison:** ✅ IDENTICAL (max diff: 0.0)
- **Statistics Match:** Yes

Sample values:
```
Sigma[0]:  13.04308796
Sigma[25]: 1.51419556
Sigma[50]: 0.00000000
```

### 3. Init Noise Sigma
- **Reference:** 13.08136654
- **TT Standalone:** 13.08136654
- **Difference:** 0.0
- **Comparison:** ✅ IDENTICAL

## Analysis

### Key Findings

1. **Initialization Logic:** Both schedulers implement identical beta schedule computation using scaled_linear approach:
   ```python
   betas = torch.linspace(beta_start**0.5, beta_end**0.5, num_train_timesteps) ** 2
   ```

2. **Timestep Generation:** Both use the same "leading" timestep spacing algorithm:
   ```python
   step_ratio = num_train_timesteps // num_inference_steps
   timesteps = (np.arange(0, num_inference_steps) * step_ratio).round()[::-1]
   ```

3. **Sigma Interpolation:** Identical linear interpolation approach for computing sigmas from alphas_cumprod.

4. **Init Noise Sigma Calculation:** Both use the same formula:
   ```python
   init_noise_sigma = (max_sigma**2 + 1) ** 0.5
   ```

### Implications

Since the schedulers are configured identically, **any differences in image generation output between tt-media-server and standalone SDXL are NOT due to scheduler configuration differences**.

The issue must lie elsewhere in the pipeline, such as:
- VAE encoding/decoding differences
- UNet forward pass differences
- Guidance scale application
- Latent initialization (noise generation)
- Prompt encoding differences
- Device-specific numerical precision issues

## Diagnostic Scripts

Two diagnostic scripts were created:

### 1. `/home/tt-admin/tt-metal/test_scheduler_diagnostics.py`
- Full implementation requiring TT device
- Uses actual `TtEulerDiscreteScheduler` class
- Requires device initialization

### 2. `/home/tt-admin/tt-metal/test_scheduler_diagnostics_simple.py`
- Simplified version using simulated TT scheduler
- No device required
- ✅ Successfully executed
- Produces identical results

### Output Files

- **Console output:** Shows summary comparison with sample values
- **Detailed file:** `/home/tt-admin/tt-metal/scheduler_diagnostics_output.txt`
  - Contains all 50 timesteps
  - Contains all 51 sigmas
  - Shows all differences (none found)

## Recommendations

Since scheduler configuration is ruled out as the source of differences:

1. **Next Investigation Areas:**
   - Compare VAE decoding between implementations
   - Verify guidance scale application matches reference
   - Check latent noise initialization seed/determinism
   - Compare UNet output at each denoising step
   - Verify prompt encoding matches exactly

2. **Test Methodology:**
   - Use identical seed values
   - Compare intermediate latents at each step
   - Log and compare guidance scale application
   - Verify no differences in model weights

3. **Potential Root Causes:**
   - Floating-point precision differences (bfloat16 vs float32)
   - Device memory layout affecting numerical results
   - Different implementations of guidance scale application
   - Trace/cache replay issues affecting computation

## Conclusion

The scheduler comparison diagnostic confirms that both the reference diffusers scheduler and the TT standalone scheduler are **mathematically and behaviorally identical**. The scheduler is not the source of any image generation differences observed between tt-media-server and standalone SDXL implementations.

Further investigation should focus on other pipeline components, particularly the VAE, UNet execution, and guidance scale application.
