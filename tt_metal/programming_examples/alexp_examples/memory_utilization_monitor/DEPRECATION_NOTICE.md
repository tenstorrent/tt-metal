# Deprecation Notice

## tt_smi.cpp - DEPRECATED

**Date**: November 3, 2025

### Status
`tt_smi.cpp` has been **deprecated** and replaced by `tt_smi_umd.cpp`.

### Reason for Deprecation
- `tt_smi_umd` provides superior UMD-based telemetry reading directly from firmware
- Interactive view switching (press `1` for main view, `2` for detailed telemetry)
- Better device caching to avoid conflicts
- Proper telemetry validation with hardware-specific limits
- Active development and maintenance

### Migration Path
Simply use `tt_smi_umd` instead of `tt_smi`:

```bash
# Old (deprecated):
./build/programming_examples/tt_smi

# New (recommended):
./build/programming_examples/tt_smi_umd

# With refresh mode:
./build/programming_examples/tt_smi_umd -r 1000
```

### Features in tt_smi_umd

**View 1: Main View** (Press `1`)
- Device summary table
- Memory breakdown (DRAM, L1, L1_SMALL, TRACE)
- Process list

**View 2: Detailed Telemetry** (Press `2`)
- Complete telemetry for each device:
  - Temperature (ASIC & Board)
  - Power & Current
  - Voltage (mV and V)
  - All clock frequencies (AICLK, AXICLK, ARCCLK)
  - Fan speed

**View 3: Charts** (Coming Soon - Press `3`)
- Memory usage over time
- Telemetry graphs

### Interactive Controls (in `-r` refresh mode)
- `1` - Switch to Main View
- `2` - Switch to Detailed Telemetry View
- `3` - Switch to Charts View (coming soon)
- `q` - Quit

### File Status
- `tt_smi.cpp` → Renamed to `tt_smi.cpp.deprecated`
- Build target removed from CMakeLists.txt
- Use `tt_smi_umd` for all monitoring needs
