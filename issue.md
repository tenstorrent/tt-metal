**Title:** ERISC firmware should zero telemetry structures on initialization

## Is your feature request related to a problem? Please describe

Yes. ERISC fabric telemetry counters in L1 memory contain uninitialized garbage values when tt-telemetry starts reading them.

The root cause is that L1 memory is not zeroed on device power-on or reset, and ERISC firmware does not explicitly initialize telemetry structures before tt-telemetry begins polling them.

This causes:
- Spurious bandwidth calculations at startup (counters appear to have wrapped around)
- Unpredictable behavior after device resets
- Noisy warning logs showing massive garbage deltas
- Reliance on fragile heuristics to detect garbage vs. valid data

Example log output showing uninitialized memory being read:
```
Suspiciously large cycle delta (17287876335105121438) for TX bandwidth on channel 8
```

These values (≈10^18) are clearly garbage from uninitialized memory, not real counter values.

## Describe the solution you'd like

**1. Zero telemetry structures during ERISC initialization:**

Add explicit initialization in the ERISC firmware init routine:
```cpp
// In ERISC init routine
memset(&fabric_telemetry, 0, sizeof(fabric_telemetry));
```

Specifically, zero all fields in:
- `FabricTelemetrySnapshot` structures
- All counter fields: `words_sent`, `packets_sent`, `elapsed_cycles`, `elapsed_active_cycles`
- Static info fields: `mesh_id`, `device_id`, `direction`, `fabric_config`

**2. Add initialization ready flag (optional but recommended):**

Add a bit to `fabric_config` or `supported_stats` indicating telemetry is initialized:
```cpp
constexpr uint8_t TELEMETRY_INITIALIZED = 0x80;  // MSB
fabric_telemetry.static_info.supported_stats |= TELEMETRY_INITIALIZED;
```

This allows telemetry to skip reading until firmware is ready, avoiding race conditions at startup.

## Describe alternatives you've considered

**Current workaround (implemented in tt-telemetry):**

Detect garbage values using heuristics:
- If counter delta > 10^12 cycles (≈14 minutes at 1.2 GHz), assume garbage/reset
- Reset baseline and skip that sample
- Continue from next sample

This works but has limitations:
- Arbitrary threshold that may miss edge cases
- Can't distinguish device reset from legitimately long sampling gaps
- Creates noisy warning logs during normal startup
- More complex telemetry code to work around firmware issue

**Other alternatives considered:**
- Skip first N samples unconditionally - but how many? Race condition still exists
- Check if counters "look" uninitialized (all 0xFF, specific patterns) - unreliable, garbage is random

None of these are as clean or robust as proper firmware initialization.

## Additional context

**Impact of proper initialization:**
- Telemetry works correctly from first sample
- No garbage detection heuristics needed in telemetry code
- Clean logs, predictable behavior
- Simpler, more maintainable code

**Related code locations:**
- Telemetry structures: `tt_metal/api/tt-metalium/experimental/fabric/fabric_telemetry.hpp`
- ERISC firmware: `tt_metal/hw/firmware/src/` (exact path depends on architecture)
- Current workaround: `tt_telemetry/telemetry/ethernet/ethernet_metrics.cpp:545-552`

**Priority:** Medium - telemetry works with current workaround, but proper firmware initialization is the correct long-term solution.
