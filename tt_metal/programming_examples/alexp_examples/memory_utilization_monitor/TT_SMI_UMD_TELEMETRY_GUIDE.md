# Using TT-UMD APIs for Telemetry in tt-smi

## Current Implementation vs UMD APIs

### Current (sysfs-based)
```cpp
// Read temperature from sysfs
float read_temperature(int device_id) {
    std::string path = "/sys/class/tenstorrent/tenstorrent!" + std::to_string(device_id) + "/asic_temp";
    std::ifstream file(path);
    if (file) {
        int temp_millicelsius;
        file >> temp_millicelsius;
        return temp_millicelsius / 1000.0f;
    }
    return -1.0f;  // N/A
}
```

**Problems:**
- Only works for local devices
- Requires specific sysfs permissions
- Shows "N/A" for remote devices or when sysfs unavailable
- Limited to what kernel exposes

### Better Approach (UMD APIs)

```cpp
#include "umd/device/types/tt_device.hpp"
#include "umd/device/firmware/firmware_info_provider.hpp"

// Get telemetry using UMD
struct TelemetryData {
    double asic_temperature;
    std::optional<double> board_temperature;
    std::optional<uint32_t> aiclk;        // MHz
    std::optional<uint32_t> fan_speed;    // RPM
    std::optional<uint32_t> tdp;          // Watts
    std::optional<uint32_t> tdc;          // Amps
    std::optional<uint32_t> vcore;        // mV
};

TelemetryData get_telemetry_from_device(int pci_device_id) {
    // Create device handle
    std::unique_ptr<TTDevice> tt_device = TTDevice::create(pci_device_id);
    tt_device->init_tt_device();

    // Get firmware info provider
    auto firmware_info = tt_device->get_firmware_info_provider();

    // Read telemetry
    TelemetryData data;
    data.asic_temperature = firmware_info->get_asic_temperature();
    data.board_temperature = firmware_info->get_board_temperature();
    data.aiclk = firmware_info->get_aiclk();
    data.fan_speed = firmware_info->get_fan_speed();
    data.tdp = firmware_info->get_tdp();
    data.tdc = firmware_info->get_tdc();
    data.vcore = firmware_info->get_vcore();

    return data;
}
```

**Benefits:**
- ✅ Works for both local AND remote devices
- ✅ No sysfs permission issues
- ✅ More telemetry data available (TDP, TDC, VCORE, AICLK, etc.)
- ✅ Abstracts firmware version differences
- ✅ Direct communication with device firmware

## Available Telemetry Tags

From `tt_metal/third_party/umd/device/api/umd/device/types/telemetry.hpp`:

```cpp
enum TelemetryTag : uint8_t {
    ASIC_TEMPERATURE = 11,    // Chip temperature (°C)
    VREG_TEMPERATURE = 12,    // Voltage regulator temp (°C)
    BOARD_TEMPERATURE = 13,   // Board temperature (°C)
    VCORE = 6,                // Core voltage (mV)
    TDP = 7,                  // Thermal Design Power (W)
    TDC = 8,                  // Thermal Design Current (A)
    AICLK = 14,               // AI clock frequency (MHz)
    AXICLK = 15,              // AXI clock frequency (MHz)
    ARCCLK = 16,              // ARC clock frequency (MHz)
    FAN_SPEED = 31,           // Fan speed (RPM)
    PCIE_USAGE = 38,          // PCIe bandwidth usage
    // ... and many more
};
```

## FirmwareInfoProvider API

From `tt_metal/third_party/umd/device/api/umd/device/firmware/firmware_info_provider.hpp`:

```cpp
class FirmwareInfoProvider {
public:
    // Temperature
    virtual double get_asic_temperature() const;
    virtual std::optional<double> get_board_temperature() const;

    // Clocks (in MHz)
    virtual std::optional<uint32_t> get_aiclk() const;
    virtual std::optional<uint32_t> get_axiclk() const;
    virtual std::optional<uint32_t> get_arcclk() const;

    // Power & Voltage
    virtual std::optional<uint32_t> get_fan_speed() const;  // RPM
    virtual std::optional<uint32_t> get_tdp() const;        // Watts
    virtual std::optional<uint32_t> get_tdc() const;        // Amps
    virtual std::optional<uint32_t> get_vcore() const;      // mV

    // Other
    virtual semver_t get_firmware_version() const;
    virtual uint64_t get_board_id() const;
    virtual uint32_t get_heartbeat() const;
};
```

## Example: UMD Telemetry Tool

See the reference implementation in:
- `tt_metal/third_party/umd/tools/telemetry.cpp`

```cpp
std::string run_default_telemetry(int pci_device, FirmwareInfoProvider* fw_info, tt::ARCH arch) {
    if (fw_info == nullptr) {
        return "Could not get information for device.";
    }

    double asic_temp = fw_info->get_asic_temperature();
    double board_temp = fw_info->get_board_temperature().value_or(0);
    uint32_t aiclk = fw_info->get_aiclk().value_or(0);
    uint32_t fan_speed = fw_info->get_fan_speed().value_or(0);
    uint32_t tdp = fw_info->get_tdp().value_or(0);
    uint32_t tdc = fw_info->get_tdc().value_or(0);
    uint32_t vcore = fw_info->get_vcore().value_or(0);

    return fmt::format(
        "Device {} - Chip {:.2f}°C, Board {:.2f}°C, AICLK {} MHz, "
        "Fan {} rpm, TDP {} W, TDC {} A, VCORE {} mV",
        pci_device, asic_temp, board_temp, aiclk,
        fan_speed, tdp, tdc, vcore);
}
```

## Why tt-smi Currently Uses sysfs

The current `tt_smi` implementation uses sysfs because:
1. **Lightweight**: `tt_smi` was designed as a standalone tool (no TT-Metal dependency)
2. **Fast**: Just reads files, no device initialization
3. **Simple**: Works without linking against UMD libraries

However, this comes at the cost of:
- ❌ Only works for local devices
- ❌ Shows "N/A" when sysfs unavailable
- ❌ Limited telemetry data

## Enhanced Implementation: tt-smi with UMD

To make `tt_smi` use UMD APIs, we need to:

### 1. Change Dependencies

**Current `CMakeLists.txt`:**
```cmake
add_executable(tt_smi tt_smi.cpp)
target_link_libraries(tt_smi PRIVATE pthread stdc++fs)
# Standalone, no dependencies!
```

**Enhanced version:**
```cmake
add_executable(tt_smi_umd tt_smi_umd.cpp)
target_link_libraries(tt_smi_umd
    PRIVATE
        pthread
        stdc++fs
        umd_device  # UMD device library
)
```

### 2. Enhanced Telemetry Class

```cpp
class TTSmiUMD {
private:
    // Cache of device handles
    std::map<int, std::unique_ptr<TTDevice>> devices_;

    // Initialize device
    TTDevice* get_device(int pci_device_id) {
        if (devices_.find(pci_device_id) == devices_.end()) {
            devices_[pci_device_id] = TTDevice::create(pci_device_id);
            devices_[pci_device_id]->init_tt_device();
        }
        return devices_[pci_device_id].get();
    }

    // Read telemetry using UMD
    TelemetryData read_telemetry_umd(int pci_device_id) {
        auto device = get_device(pci_device_id);
        auto fw_info = device->get_firmware_info_provider();

        TelemetryData data;
        data.asic_temperature = fw_info->get_asic_temperature();
        data.board_temperature = fw_info->get_board_temperature();
        data.aiclk = fw_info->get_aiclk();
        data.tdp = fw_info->get_tdp();
        data.tdc = fw_info->get_tdc();
        data.fan_speed = fw_info->get_fan_speed();
        data.vcore = fw_info->get_vcore();

        // Calculate power from TDP and TDC
        if (data.tdp.has_value() && data.tdc.has_value()) {
            // Power (W) ≈ TDP, or calculate from voltage and current
            data.power_watts = data.tdp.value();
        }

        return data;
    }
};
```

### 3. Display Enhanced Telemetry

```cpp
void display_device_with_umd_telemetry(int device_id, const TelemetryData& telem) {
    std::cout << "│ " << device_id;
    std::cout << std::setw(16) << get_arch_name(device_id);

    // Temperature (from UMD)
    std::cout << std::fixed << std::setprecision(1)
              << std::setw(10) << telem.asic_temperature << "°C";

    // Power (calculated from TDP/TDC or direct)
    if (telem.power_watts.has_value()) {
        std::cout << std::setw(10) << telem.power_watts.value() << "W";
    } else {
        std::cout << std::setw(10) << "N/A";
    }

    // AICLK
    if (telem.aiclk.has_value()) {
        std::cout << "  AICLK: " << telem.aiclk.value() << " MHz";
    }

    // Memory usage (from allocation server)
    // ...
}
```

## Hybrid Approach (Best of Both Worlds)

**Option 1: Two separate tools**
- `tt_smi` - Lightweight, standalone, uses sysfs (current)
- `tt_smi_full` - Full-featured, uses UMD APIs (enhanced)

**Option 2: Runtime detection**
```cpp
TelemetryData get_telemetry(int device_id) {
    // Try UMD first
    try {
        return get_telemetry_umd(device_id);
    } catch (...) {
        // Fall back to sysfs
        return get_telemetry_sysfs(device_id);
    }
}
```

**Option 3: Compile-time option**
```cmake
option(TT_SMI_USE_UMD "Use UMD APIs for telemetry" ON)

if(TT_SMI_USE_UMD)
    target_link_libraries(tt_smi PRIVATE umd_device)
    target_compile_definitions(tt_smi PRIVATE USE_UMD_TELEMETRY)
endif()
```

```cpp
#ifdef USE_UMD_TELEMETRY
    auto telemetry = get_telemetry_umd(device_id);
#else
    auto telemetry = get_telemetry_sysfs(device_id);
#endif
```

## Comparison

| Feature | sysfs (current) | UMD APIs (enhanced) |
|---------|----------------|---------------------|
| **Build dependency** | None (standalone) | Requires UMD |
| **Local devices** | ✅ Yes | ✅ Yes |
| **Remote devices** | ❌ No | ✅ Yes |
| **Permissions** | May need sudo | Works as user |
| **Temperature** | ✅ Yes | ✅ Yes |
| **Power** | ✅ Yes (if available) | ✅ Yes |
| **AICLK** | ❌ No | ✅ Yes |
| **TDP/TDC** | ❌ No | ✅ Yes |
| **VCORE** | ❌ No | ✅ Yes |
| **Fan speed** | ❌ No | ✅ Yes |
| **Initialization time** | Instant | ~100ms per device |

## Recommendation

For your use case (4 Blackhole devices showing "N/A"):

**Use UMD APIs!** Because:
1. You're already using TT-Metal/UMD (allocation server uses it)
2. Your devices might be remote or not exposing sysfs telemetry
3. You'll get much more detailed information

### Quick Implementation

I can update `tt_smi.cpp` to use UMD APIs. Would you like me to:
1. Create `tt_smi_umd.cpp` - New version with UMD telemetry
2. Update `tt_smi.cpp` - Replace sysfs with UMD
3. Hybrid approach - Try UMD, fall back to sysfs

Let me know and I'll implement it!
