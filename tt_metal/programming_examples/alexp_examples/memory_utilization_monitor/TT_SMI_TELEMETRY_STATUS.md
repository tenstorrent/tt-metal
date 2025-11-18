# tt-smi Telemetry Status

## Current Status

✅ **Build successful** - `tt-smi` now links against TT-Metal
⚠️ **Telemetry not yet implemented** - Shows "N/A" for temperature/power (same as before)

## What Changed

### Architecture
- **Before**: Standalone tool, read from sysfs
- **Now**: Links against TT::Metalium, can access device APIs

### Build Configuration
```cmake
# tt_smi now links against TT-Metal
target_link_libraries(tt_smi PUBLIC TT::Metalium)
```

### Code Structure
```cpp
// Simplified telemetry placeholder
TelemetryData read_telemetry_from_device(int device_id) {
    TelemetryData data;
    // TODO: Implement using TT-Metal/UMD APIs
    return data;  // Empty for now
}
```

## Why Telemetry Isn't Implemented Yet

The challenge is that **TT-Metal doesn't expose telemetry APIs directly**:

1. ✅ TT-Metal wraps UMD device APIs
2. ❌ But telemetry functions (temperature, power, etc.) aren't exposed through TT-Metal's public API
3. ✅ UMD has `FirmwareInfoProvider` with all the telemetry
4. ❌ But accessing it requires going through internal/private APIs

## Three Options to Implement Telemetry

### Option 1: Access UMD Device Directly (Recommended)

Similar to how `allocation_server_poc` accesses device info:

```cpp
TelemetryData read_telemetry_from_device(int device_id) {
    TelemetryData data;

    try {
        // Create minimal device for telemetry only
        auto device = tt::tt_metal::CreateDeviceMinimal(device_id);

        // Access underlying UMD device (requires accessing private/internal APIs)
        // This is the missing piece - need to find how to get UMD device from IDevice

        // Pseudocode (doesn't compile yet):
        // auto umd_device = device->get_umd_device();  // Not exposed!
        // auto fw_info = umd_device->get_firmware_info_provider();
        // data.asic_temperature = fw_info->get_asic_temperature();
        //... etc

    } catch (...) {
        // Return empty data on error
    }

    return data;
}
```

**Pros:**
- Most direct approach
- All telemetry available
- Works for both local and remote devices

**Cons:**
- Requires accessing TT-Metal internal APIs
- May need changes to TT-Metal to expose UMD device

### Option 2: Use Standalone UMD Tool

Instead of implementing telemetry in `tt-smi`, just call the existing UMD telemetry tool:

```bash
# Use existing tool
/path/to/umd/tools/telemetry -d 0 -d 1 -d 2 -d 3
```

**Pros:**
- Already works
- No code changes needed
- All telemetry features available

**Cons:**
- Not integrated into `tt-smi`
- Separate tool to run
- Different output format

### Option 3: Add Telemetry APIs to TT-Metal

Submit PR to TT-Metal to expose telemetry:

```cpp
// Proposed addition to tt-metalium/host_api.hpp
namespace tt::tt_metal {

    struct DeviceTelemetry {
        double asic_temperature;
        std::optional<double> board_temperature;
        std::optional<uint32_t> aiclk;
        std::optional<uint32_t> tdp;
        // ... etc
    };

    // New API
    DeviceTelemetry GetDeviceTelemetry(IDevice* device);
}
```

**Pros:**
- Clean public API
- Officially supported
- Other tools can use it too

**Cons:**
- Requires PR review/approval
- Takes time
- Needs design discussion

## Recommended Next Steps

### Short-term (Quick Fix)

**Use the standalone UMD telemetry tool:**

```bash
# Terminal 1: Allocation server
./allocation_server_poc

# Terminal 2: Telemetry monitoring
watch -n 1 './path/to/umd/tools/telemetry -d 0 -d 1 -d 2 -d 3'

# Terminal 3: tt-smi for memory + process info
./tt_smi -w
```

This gives you:
- ✅ Full telemetry from UMD tool
- ✅ Memory tracking from tt-smi
- ✅ Process discovery from tt-smi
- ❌ Not integrated (need 2 tools)

### Medium-term (Implement Option 1)

Find how to access UMD device from TT-Metal `IDevice`:

```cpp
// Search in TT-Metal codebase for how to get UMD device
// Likely in device implementation files:
// - tt_metal/impl/device/device.hpp
// - tt_metal/impl/device/device.cpp
```

Once found, implement `read_telemetry_from_device()` properly.

### Long-term (Option 3)

Work with TT-Metal team to add official telemetry APIs.

## Current Workaround

For now, `tt-smi` shows:
- ✅ All devices
- ✅ All processes using devices
- ✅ Memory usage (when server running)
- ⚠️ Temperature/Power: "N/A"

To get telemetry, use the standalone UMD tool:

```bash
# Check where telemetry tool is
find /home/ttuser/aperezvicente -name "telemetry" -type f 2>/dev/null

# Run it
./path/to/telemetry
```

## Files Structure

```
tt_smi.cpp
├── Links against: TT::Metalium
├── Uses: tt::tt_metal::GetNumAvailableDevices()
├── TODO: Implement read_telemetry_from_device()
└── Status: Builds ✅, Telemetry pending ⚠️

allocation_server_poc.cpp
├── Links against: TT::Metalium
├── Uses: tt::tt_metal::CreateDeviceMinimal()
├── Gets: Device info (arch, DRAM size, L1 size)
└── Status: Working ✅

UMD telemetry tool
├── Direct UMD access
├── Full telemetry implementation
└── Status: Working ✅ (standalone)
```

## Summary

**Current state:**
- ✅ `tt-smi` builds successfully with TT-Metal
- ✅ Shows devices, processes, memory usage
- ⚠️ Telemetry not yet implemented
- ⚠️ Temperature/Power show "N/A" (same as sysfs version)

**To get working telemetry:**
1. **Quick**: Use standalone UMD telemetry tool (works now)
2. **Medium**: Implement Option 1 (access UMD device from TT-Metal)
3. **Long**: Propose telemetry APIs to TT-Metal team

**Recommendation:**
Use the standalone UMD telemetry tool for now while we figure out the proper way to access telemetry through TT-Metal's API.
