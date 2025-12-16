# SDXL Server Log Analysis - November 26, 2025

## Log File Analyzed
`sdxl_server_20251126_193620.log` (4-worker production mode, 19:36-19:54)

## Executive Summary

### What Worked ✅
1. All 4 workers successfully completed warmup (19:40-19:54, ~3.5 min/worker)
2. Server reached "All workers ready. Server is accepting requests" state
3. `generate_input_tensors` executed successfully during warmup for all workers
4. No "str object has no attribute 'type'" errors in THIS log
5. Text encoder compilation successful
6. UNet denoising compilation successful  
7. VAE processing successful

### Critical Issues Found 🔴

#### Issue #1: Segmentation Faults on Shutdown
**Location**: Workers 0, 1, 2 during shutdown cleanup  
**Function**: `MeshDevice::release_mesh_trace()`  
**Error**: Segmentation fault (11) - Address not mapped (0x40)

**Stack Trace Pattern**:
```
libtt_metal.so(_ZNK2tt8tt_metal23SubDeviceManagerTracker29get_active_sub_device_managerEv+0x0)
libtt_metal.so(_ZN2tt8tt_metal11distributed10MeshDevice18release_mesh_traceE...)
_ttnncpp.so(_ZN4ttnn10operations5trace13release_traceE...)
```

**Impact**: Server crashes during graceful shutdown, potentially leaving resources in bad state

#### Issue #2: Multiple Workers in Non-Dev Mode
**Observation**: This log shows 4 workers starting despite earlier runs being in dev mode  
**Concern**: Multiple workers may interfere with each other or cause resource conflicts

### Warnings & Anomalies ⚠️

1. **Inspector RPC Server Failures**
   - `Failed to start Inspector RPC server: Address already in use (127.0.0.1:50051)`
   - Occurs for workers 1, 2, 3
   - May indicate improper cleanup from previous runs

2. **Unexpected Mailbox Values**
   - `Read unexpected run_mailbox value: 0x40 (expected 0x80 or 0x0)`
   - Occurs during device initialization for multiple workers
   - Suggests dispatch kernels from previous run still active

3. **Firmware Version Warning**
   - System has 18.12.0, max supported is 18.10.0
   - May cause undefined behavior

## Comparison: This Log vs Error Log (19:07)

| Aspect | This Log (19:36) | Error Log (19:07) |
|--------|------------------|-------------------|
| **Mode** | Production (4 workers) | Unknown |
| **Warmup** | ✅ All workers successful | ❌ Worker died during warmup |
| **Device Error** | ❌ Not present in warmup | ✅ Present |
| **Shutdown** | ⚠️ Segfaults | N/A - crashed before |
| **Timestamp** | LATER (after fixes?) | EARLIER (before fixes?) |

**Key Question**: Were the device type fixes already applied when this log was generated?

## Proposed Investigation Avenues

### Avenue 1: Verify Fix Application Timeline 🔍 HIGH PRIORITY
**Question**: Which fixes were present when `sdxl_server_20251126_193620.log` was generated?

**Actions**:
1. Check git log/timestamps to see when device fix was applied
2. Correlate log timestamp (19:36 start) with code modification times
3. Determine if this log represents:
   - Pre-fix baseline (would explain no device error)
   - Post-fix validation (proves fix works)

**Expected Outcome**: Clarify whether this log validates our fixes or predates them

### Avenue 2: Trace Cleanup Segfault Investigation 🔍 CRITICAL
**Problem**: Multiple workers crash with segfault in `release_mesh_trace` during shutdown

**Root Cause Hypotheses**:
1. **Hypothesis A**: Trace not properly initialized/captured
   - Workers may be calling `release_mesh_trace` on traces that were never created
   - Check if `capture_trace` flag is properly handled in all code paths

2. **Hypothesis B**: Device already deallocated
   - `close_device()` may be deallocating resources before trace cleanup
   - Order of operations in shutdown sequence may be wrong

3. **Hypothesis C**: SubDeviceManager lifecycle issue
   - `get_active_sub_device_manager` fails because manager already destroyed
   - Suggests improper cleanup ordering

**Investigation Actions**:
1. Review `sdxl_runner.py:161-165` (close_device method)
2. Check `tt_sdxl_pipeline.py` for trace release logic
3. Search for `release_trace` calls and verify they're guarded by existence checks
4. Add logging before trace release to confirm traces exist
5. Review ttnn trace lifecycle documentation

**Potential Fixes**:
- Add null checks before `release_mesh_trace` calls
- Ensure traces are released BEFORE device closure
- Add try-except around trace cleanup
- Verify `capture_trace` config is consistently used

### Avenue 3: Resource Conflict Analysis 🔍 MEDIUM PRIORITY
**Observations**:
- Inspector RPC port conflicts (50051 already in use)
- Unexpected mailbox values (dispatch kernels still running)
- Multiple server instances observed in process list

**Investigation Actions**:
1. Check for orphaned processes from previous runs
2. Verify proper device reset between server restarts
3. Add device cleanup validation before server start
4. Consider adding startup check: fail fast if devices in bad state

**Potential Fixes**:
- Add device health check to launcher script
- Implement forced device reset before initialization
- Add process cleanup to launcher script
- Make Inspector RPC port configurable per worker

### Avenue 4: Dev Mode Configuration Validation 🔍 LOW PRIORITY
**Question**: Why did a --dev run spawn 4 workers instead of 1?

**Actions**:
1. Verify `sdxl_config.py` dev_mode override logic
2. Check environment variable propagation
3. Confirm `--dev` flag properly sets SDXL_DEV_MODE env var
4. Review launcher script's dev mode handling

### Avenue 5: Inference Request Test 🔍 HIGH PRIORITY
**Goal**: Verify the device type fix actually resolves inference errors

**Test Strategy**:
1. **Clean environment**: Kill all servers, reset devices
2. **Start fresh**: Single dev mode server with fixes applied
3. **Verify warmup**: Confirm "All workers ready" message
4. **Test inference**: Run `image_test.py` with simple prompt
5. **Compare behavior**: Check if device type error is gone

**Expected Results**:
- ✅ No "'str' object has no attribute 'type'" error
- ✅ Image generates successfully
- ⚠️ May encounter NEW errors (e.g., trace cleanup segfault if inference uses traces)

### Avenue 6: Concurrent Worker Behavior 🔍 MEDIUM PRIORITY
**Observation**: 4 workers warming up sequentially, each taking ~4-5 minutes

**Questions**:
- Does sequential warmup avoid cache corruption?
- Do workers share compiled kernels from cache?
- Why does each worker take full warmup time (not benefiting from cache)?

**Actions**:
1. Review cache sharing between workers
2. Check if JIT compilation cache is properly utilized
3. Verify cache locking/coordination
4. Consider adding cache prewarming step

## Recommended Action Priority

1. **Immediate**: Test inference with device fix (Avenue 5)
2. **Urgent**: Fix trace cleanup segfaults (Avenue 2)  
3. **Important**: Validate dev mode configuration (Avenue 4)
4. **Medium**: Investigate resource conflicts (Avenue 3)
5. **Nice to have**: Optimize worker warmup (Avenue 6)
6. **Baseline**: Verify fix timeline (Avenue 1)

## Key Observations

### Positive Indicators
- `generate_input_tensors` completed successfully for all workers during warmup
- This was the exact function that failed with device type error in earlier log
- Suggests the device type fix MAY already be working (if applied before this log)

### Concerning Indicators  
- Segfaults during shutdown are unacceptable for production
- Resource conflicts suggest environment cleanup issues
- Lack of full traceback in error log makes debugging harder

## Next Steps Recommendation

**Option A: Validate Current Fixes (Recommended)**
1. Kill all servers
2. Start clean dev mode server (1 worker only)
3. Wait for "ready" message
4. Run single inference test
5. Check if device error is resolved
6. Deal with any NEW issues that appear

**Option B: Deep Dive Trace Cleanup**
1. Add extensive logging to trace lifecycle
2. Modify shutdown sequence to release traces before devices
3. Add safety checks for trace existence
4. Test multi-worker shutdown

**Option C: Comprehensive Testing**
1. Test both dev mode (1 worker) and production (4 workers)
2. Run multiple inference requests
3. Monitor for memory leaks or resource issues
4. Validate shutdown behavior

## Conclusion

The log shows the server CAN complete warmup successfully with 4 workers, but has critical issues:
- Segfault on shutdown (trace cleanup)
- No evidence of inference testing in this log
- Device type fix effectiveness unconfirmed

**Most Valuable Next Action**: Run clean inference test to validate the device type fix addresses the original "'str' object has no attribute 'type'" error.
