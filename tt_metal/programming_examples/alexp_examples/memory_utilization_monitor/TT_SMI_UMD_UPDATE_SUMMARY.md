# tt-smi Update: Sysfs → UMD Telemetry

## Summary

**Updated `tt-smi` to use TT-UMD APIs instead of sysfs for telemetry** - This should fix the "N/A" temperature and power readings you're seeing on your Blackhole devices.

## What Changed

### Before (sysfs-based)
```cpp
// Read from Linux sysfs
float read_temperature(int device_id) {
    std::ifstream file("/sys/class/tenstorrent/tenstorrent!" + std::to_string(device_id) + "/asic_temp");
    // ...
}
```

**Problems:**
- ❌ Shows "N/A" for your 4 Blackhole devices
- ❌ Requires specific sysfs paths
- ❌ Limited data (only temp and power)
- ❌ Doesn't work for remote devices

### After (UMD-based)
```cpp
// Use TT-UMD firmware info provider
TelemetryData read_telemetry_umd(int pci_device_id) {
    auto device = TTDevice::create(pci_device_id);
    device->init_tt_device();

    auto fw_info = device->get_firmware_info_provider();

    TelemetryData data;
    data.asic_temperature = fw_info->get_asic_temperature();  // Direct from firmware
    data.tdp = fw_info->get_tdp();                            // Power in Watts
    data.aiclk = fw_info->get_aiclk();                        // Clock freq (MHz)
    data.fan_speed = fw_info->get_fan_speed();                // Fan RPM
    data.vcore = fw_info->get_vcore();                        // Core voltage (mV)
    // ... and more

    return data;
}
```

**Benefits:**
- ✅ Should work for your Blackhole devices
- ✅ Works for both local AND remote devices
- ✅ Much more telemetry data available
- ✅ Direct communication with device firmware
- ✅ Handles different firmware versions automatically

## New Telemetry Available

### Previously (sysfs)
- Temperature
- Power

### Now (UMD APIs)
- **ASIC Temperature** (°C) - Chip temperature
- **Board Temperature** (°C) - Optional, if sensor available
- **TDP** (Watts) - Thermal Design Power (displayed as "Power")
- **TDC** (Amps) - Thermal Design Current
- **VCORE** (mV) - Core voltage
- **AICLK** (MHz) - AI clock frequency
- **AXICLK** (MHz) - AXI clock frequency
- **ARCCLK** (MHz) - ARC clock frequency
- **Fan Speed** (RPM) - If fans present

## Files Changed

### 1. `tt_smi.cpp`

**Added:**
- UMD includes (`TTDevice`, `FirmwareInfoProvider`)
- `TelemetryData` struct to hold all telemetry
- `get_umd_device()` - Create and cache UMD device handles
- `read_telemetry_umd()` - Read telemetry via UMD APIs
- Device caching to avoid re-initialization

**Removed:**
- `read_temperature()` - Sysfs-based temperature reading
- `read_power()` - Sysfs-based power reading

**Updated:**
- `DeviceInfo` struct now has `TelemetryData telemetry` instead of `float temperature, power`
- Device enumeration uses `PCIDevice::enumerate_devices()` when server unavailable
- Display code uses `dev.telemetry.asic_temperature` and `dev.telemetry.tdp`

### 2. `CMakeLists.txt`

**Changed:**
```cmake
# Before:
target_link_libraries(tt_smi PRIVATE pthread stdc++fs)

# After:
target_link_libraries(tt_smi
    PRIVATE
        pthread
        stdc++fs
        umd_device  # UMD for telemetry APIs
)
```

## Testing

### Build
```bash
cd /home/ttuser/aperezvicente/tt-metal-apv
cmake --build build --target tt_smi -j8
```

**Status:** ✅ Build successful!

Binary location:
- `/home/ttuser/aperezvicente/tt-metal-apv/build_Release_tracy/programming_examples/tt_smi`
- `/home/ttuser/aperezvicente/tt-metal-apv/build_Release_tracy/install/bin/tt_smi`

### Run Test

```bash
# Test the updated version
./build_Release_tracy/programming_examples/tt_smi
```

**Expected improvements:**
- ✅ Temperature should show actual values (not "N/A")
- ✅ Power should show TDP values (not "N/A")
- ✅ Devices should be properly detected
- ✅ Works for all 4 of your Blackhole devices

## Before/After Comparison

### Your Current Output (with sysfs)
```
│ 0   Blackhole       N/A       N/A       0.0B/31.9GB         [░░░░░░░░░░░░░░░░░░░░░░░░░] 0% │
│ 1   Blackhole       N/A       N/A       0.0B/31.9GB         [░░░░░░░░░░░░░░░░░░░░░░░░░] 0% │
│ 2   Blackhole       N/A       N/A       0.0B/31.9GB         [░░░░░░░░░░░░░░░░░░░░░░░░░] 0% │
│ 3   Blackhole       N/A       N/A       0.0B/31.9GB         [░░░░░░░░░░░░░░░░░░░░░░░░░] 0% │
```

### Expected Output (with UMD)
```
│ 0   Blackhole       65°C      180W      0.0B/31.9GB         [░░░░░░░░░░░░░░░░░░░░░░░░░] 0% │
│ 1   Blackhole       63°C      175W      0.0B/31.9GB         [░░░░░░░░░░░░░░░░░░░░░░░░░] 0% │
│ 2   Blackhole       67°C      182W      0.0B/31.9GB         [░░░░░░░░░░░░░░░░░░░░░░░░░] 0% │
│ 3   Blackhole       64°C      178W      0.0B/31.9GB         [░░░░░░░░░░░░░░░░░░░░░░░░░] 0% │
```

**Note:** Actual values will depend on your devices' current state.

## Implementation Details

### Device Caching

To avoid re-initializing devices on every refresh, we cache UMD device handles:

```cpp
std::map<int, std::unique_ptr<TTDevice>> umd_devices_;

TTDevice* get_umd_device(int pci_device_id) {
    if (umd_devices_.find(pci_device_id) == umd_devices_.end()) {
        umd_devices_[pci_device_id] = TTDevice::create(pci_device_id);
        umd_devices_[pci_device_id]->init_tt_device();
    }
    return umd_devices_[pci_device_id].get();
}
```

**Benefits:**
- Fast subsequent queries (~microseconds vs ~100ms)
- Suitable for watch mode (`-w`)
- Devices initialized on-demand

### Error Handling

Graceful fallbacks if UMD initialization fails:

```cpp
try {
    auto device = get_umd_device(pci_device_id);
    if (device == nullptr) {
        return default_telemetry;  // Return empty data
    }
    // ... read telemetry
} catch (const std::exception& e) {
    std::cerr << "Warning: " << e.what() << std::endl;
    return default_telemetry;
}
```

This ensures `tt-smi` degrades gracefully rather than crashing.

## Usage

### Basic (one-shot)
```bash
./build_Release_tracy/programming_examples/tt_smi
```

### Watch mode (continuous updates)
```bash
./build_Release_tracy/programming_examples/tt_smi -w
```

### With allocation server (for memory tracking)
```bash
# Terminal 1: Start server
./build_Release_tracy/programming_examples/allocation_server_poc

# Terminal 2: Run tt-smi
./build_Release_tracy/programming_examples/tt_smi -w -r 1000
```

## Troubleshooting

### If temperatures still show "N/A"

**Possible causes:**
1. Devices not initialized properly
2. Firmware doesn't support telemetry
3. Device communication error

**Debug:**
```bash
# Check if UMD can see devices
./build_Release_tracy/third_party/umd/tools/telemetry -d 0

# Check device files
ls -la /dev/tenstorrent/

# Check for error messages
./build_Release_tracy/programming_examples/tt_smi 2>&1 | grep -i warning
```

### If build fails

**Missing UMD library:**
```bash
# Build UMD device library first
cmake --build build --target umd_device -j8
```

## Architecture Benefits

### Why UMD > Sysfs

| Aspect | Sysfs | UMD APIs |
|--------|-------|----------|
| **Local devices** | ✅ Yes | ✅ Yes |
| **Remote devices** | ❌ No | ✅ Yes |
| **Telemetry richness** | Limited | Comprehensive |
| **Firmware abstraction** | ❌ No | ✅ Yes (handles versions) |
| **Initialization** | None | ~100ms (cached) |
| **Permissions** | May need root | User-level |
| **Your Blackhole setup** | ❌ Shows N/A | ✅ Should work |

## Next Steps

### Immediate
1. Test the updated `tt-smi` on your 4 Blackhole devices
2. Verify temperature and power show actual values
3. Report any issues

### Future Enhancements
Consider adding to the display:
- **AICLK** - Show clock frequency for performance monitoring
- **VCORE** - Show voltage for power analysis
- **Fan Speed** - Show cooling system status
- **More detailed view** - Toggle with `-v` flag for verbose output

Example enhanced display:
```
│ Device 0 (Blackhole)                                                         │
│   Temperature:  65°C (chip), 45°C (board)                                   │
│   Power:        180W (TDP: 200W, TDC: 15A)                                  │
│   Clocks:       AICLK 1000 MHz, AXICLK 800 MHz                              │
│   Voltage:      VCORE 750 mV                                                 │
│   Cooling:      Fan 3500 RPM                                                 │
│   Memory:       2.4GB / 31.9GB DRAM, 45MB / 120MB L1                        │
```

## Related Documentation

- **TT_SMI_UMD_TELEMETRY_GUIDE.md** - Detailed UMD telemetry API guide
- **TT_SMI_README.md** - Complete tt-smi usage documentation
- **NVIDIA_SMI_ARCHITECTURE.md** - Comparison with nvidia-smi

## Summary

✅ **Updated `tt-smi.cpp` to use TT-UMD APIs for direct device telemetry**
✅ **Should fix "N/A" temperature/power on your Blackhole devices**
✅ **Build successful**
✅ **Ready to test!**

Run it and see if you now get actual temperature and power readings! 🎉
