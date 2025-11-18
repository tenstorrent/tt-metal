# UMD Custom Modifications for apv-tt-br0-clean

This branch includes custom modifications to the `tt-umd` submodule that are maintained as a patch file rather than a custom fork.

## Modifications

The patch (`umd_custom_modifications.patch`) includes:

1. **Chip Initialization Locking** (`device/chip/local_chip.cpp`)
   - Modified to release the CHIP_IN_USE lock after initialization completes
   - Allows multiple processes to query device info and use the chip after initialization
   - Previously the lock was held for the lifetime of the chip

2. **Blackhole ETH Core Validation** (`device/coordinates/blackhole_coordinate_manager.cpp`)
   - Commented out the strict validation requiring exactly 2 or all ETH cores to be harvested
   - Provides more flexibility for different hardware configurations

## How to Apply the Patch

After cloning this repository and updating submodules, apply the UMD patch:

```bash
# Update all submodules
git submodule update --recursive --init

# Navigate to the UMD submodule
cd tt_metal/third_party/umd

# Apply the custom patch
git apply ../../../umd_custom_modifications.patch

# Return to repository root
cd ../../..
```

## Rebuilding After Updates

If you update the UMD submodule to a newer version, you may need to reapply or update the patch:

```bash
cd tt_metal/third_party/umd

# Try to apply the patch
git apply ../../../umd_custom_modifications.patch

# If conflicts occur, you may need to manually reapply the changes
# Refer to the patch file to see what modifications are needed
```

## Why a Patch Instead of a Fork?

- **Easier maintenance**: No need to maintain a separate fork of tt-umd
- **Transparent changes**: All modifications are visible in a single patch file
- **Upstream compatibility**: Easy to update to newer UMD versions and reapply the patch
- **Collaboration**: Others can easily see and review the custom modifications
