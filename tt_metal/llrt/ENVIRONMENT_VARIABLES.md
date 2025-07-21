# TT-Metal Runtime Environment Variables

This document provides a comprehensive overview of all environment variables read by the TT-Metal runtime system. These variables allow you to configure various aspects of the runtime behavior including debugging, profiling, caching, and hardware-specific settings.

## Core Path Configuration

### `TT_METAL_HOME`
Sets the root directory of the TT-Metal installation. For example the directory as to where the repo was cloned.

### `TT_METAL_KERNEL_PATH`
Specifies the directory path for kernel files. When set, kernels are loaded from this custom location instead of the default path.

### `TT_METAL_CACHE`
Sets the cache directory location. When set, caches are stored in `{value}/tt-metal-cache/`.

## Device Configuration

### `TT_METAL_VISIBLE_DEVICES`
Comma-separated list of device IDs to make visible to the runtime. Allows restricting which devices the application can access.

### `TT_METAL_SIMULATOR`
Enables simulator mode and sets the simulator path. When set, the runtime operates in simulation mode using the specified simulator.

### `ARCH_NAME`
Sets the architecture name. Only necessary during simulation.

## Debug and Development Features

### `TT_METAL_ENABLE_ERISC_IRAM`
Controls whether ERISC IRAM is enabled. Set to '1' to enables.

This variable is particularly important because:

1. **JIT Compilation Impact**: When enabled, it affects how ERISC kernels are compiled and linked. The JIT build system uses different linker scripts based on this setting

2. **Fabric Configuration**: The fabric system automatically enables ERISC IRAM when fabric is configured

3. **Kernel Loading**: When ERISC IRAM is enabled, kernels are loaded differently - they're first placed in L1 memory and then copied to IRAM for execution

4. **Testing Requirements**: Many tests explicitly set this variable to ensure consistent behavior across fabric and non-fabric workloads

The IRAM functionality is architecture-specific (primarily for Wormhole) and provides performance benefits by allowing instruction execution from faster IRAM instead of L1 memory. However, it's automatically disabled when certain debug features like watcher or DPrint are enabled due to IRAM size constraints

### `TT_METAL_KERNEL_MAP`
Enables kernel build mapping for debugging purposes.

### `TT_METAL_NULL_KERNELS`
Enables null kernel mode for testing. When set, kernels are replaced with no-op versions.

### `TT_METAL_KERNELS_EARLY_RETURN`
Makes kernels return early while maintaining same size as normal kernels for testing purposes.

### `TT_METAL_RISCV_DEBUG_INFO`
Controls whether to compile with DWARF debug info in RISC-V binaries. Set to '0' to disable, any other value enables it.

### `TT_METAL_VALIDATE_PROGRAM_BINARIES`
Enables validation of kernel binaries. Set to '1' to enable validation checks.

## Memory Management

### `TT_METAL_CLEAR_L1`
When set to '1', clears L1 memory on device initialization.

### `TT_METAL_CLEAR_DRAM`
When set to '1', clears DRAM memory on device initialization.

### `TT_METAL_ENABLE_HW_CACHE_INVALIDATION`
Enables hardware cache invalidation for Blackhole's L1 data cache.

### `TT_METAL_DISABLE_RELAXED_MEM_ORDERING`
Disables relaxed memory ordering on Blackhole devices.

### `TT_METAL_ENABLE_GATHERING`
Enables instruction gathering in Tensix cores for performance optimization.

## Profiling and Performance Analysis

### `TT_METAL_DEVICE_PROFILER`
When set to '1', enables device profiling (requires TRACY_ENABLE compilation flag).

### `TT_METAL_DEVICE_PROFILER_DISPATCH`
When set to '1', enables profiling of dispatch cores (requires device profiler to be enabled).

### `TT_METAL_PROFILER_SYNC`
When set to '1', enables synchronous profiling mode for more accurate timing.

### `TT_METAL_TRACY_MID_RUN_PUSH`
When set to '1', forces Tracy profiler pushes during execution for real-time profiling.

### `TT_METAL_DEVICE_PROFILER_NOC_EVENTS`
When set to '1', enables NoC (Network-on-Chip) events profiling.

### `TT_METAL_DEVICE_PROFILER_NOC_EVENTS_RPT_PATH`
Sets the report path for NoC events profiling output files.

### `TT_METAL_MEM_PROFILER`
When set to '1', enables memory/buffer usage profiling for tracking memory allocation patterns.

## Dispatch and Execution Control

### `TT_METAL_SLOW_DISPATCH_MODE`
Enables slow dispatch mode when set. This mode provides more debugging capabilities at the cost of performance.

### `TT_METAL_DISPATCH_DATA_COLLECTION`
Enables dispatch data collection for performance analysis and debugging.

### `TT_METAL_GTEST_ETH_DISPATCH`
Sets dispatch core type to Ethernet for testing scenarios.

### `TT_METAL_GTEST_NUM_HW_CQS`
Sets the number of hardware command queues for testing. Must be a valid integer value.

## Watcher Debug System

### `TT_METAL_WATCHER`
Enables watcher with specified interval. Value can be in milliseconds (with 'ms' suffix) or seconds. The watcher monitors device state and can detect hangs and errors.

### `TT_METAL_WATCHER_TEST_MODE`
Enables test mode for watcher, which catches exceptions instead of throwing them.

### `TT_METAL_WATCHER_DUMP_ALL`
Enables dumping all watcher data, including potentially unsafe state information.

### `TT_METAL_WATCHER_APPEND`
Enables append mode for watcher output files instead of overwriting.

### `TT_METAL_WATCHER_NOINLINE`
Disables inlining for watcher functions to reduce binary size.

### `TT_METAL_WATCHER_PHYS_COORDS`
Uses physical coordinates in watcher output instead of logical coordinates.

### `TT_METAL_WATCHER_TEXT_START`
Includes text start information in watcher output for debugging.

### `TT_METAL_WATCHER_SKIP_LOGGING`
Disables watcher logging to reduce overhead.

### `TT_METAL_WATCHER_ENABLE_NOC_SANITIZE_LINKED_TRANSACTION`
Enables NoC sanitization for linked transactions to catch more subtle errors.

### `TT_METAL_WATCHER_DEBUG_DELAY`
Sets debug delay for watcher operations. Requires watcher to be enabled and NoC sanitization to not be disabled.

### `TT_METAL_WATCHER_KEEP_ERRORS`
Keeps errors around for watcher dump testing instead of clearing them.

### Watcher Feature Disable Variables
Multiple variables to disable specific watcher features:
- `TT_METAL_WATCHER_DISABLE_WAYPOINT`
- `TT_METAL_WATCHER_DISABLE_NOC_SANITIZE`
- `TT_METAL_WATCHER_DISABLE_ASSERT`
- `TT_METAL_WATCHER_DISABLE_PAUSE`
- `TT_METAL_WATCHER_DISABLE_RING_BUFFER`
- `TT_METAL_WATCHER_DISABLE_STACK_USAGE`
- `TT_METAL_WATCHER_DISABLE_DISPATCH`

## Inspector Debug System

### `TT_METAL_INSPECTOR`
Enables or disables the inspector system. Set to '0' to disable, any other value enables it.

### `TT_METAL_INSPECTOR_LOG_PATH`
Sets the log path for inspector output. Defaults to `{TT_METAL_HOME}/generated/inspector` if not specified.

### `TT_METAL_INSPECTOR_INITIALIZATION_IS_IMPORTANT`
Controls whether initialization is considered important for inspector. Set to '0' to disable.

### `TT_METAL_INSPECTOR_WARN_ON_WRITE_EXCEPTIONS`
Controls whether to warn on write exceptions in inspector. Set to '0' to disable warnings.

## Runtime Debug Features

The following environment variables control debug features for cores, with support for targeting specific cores, chips, and RISC-V processors.

### DPrint Debug Feature
- `TT_METAL_DPRINT_CORES` - Specifies cores for debug printing
- `TT_METAL_DPRINT_ETH_CORES` - Specifies Ethernet cores for debug printing
- `TT_METAL_DPRINT_CHIPS` - Specifies chips for debug printing
- `TT_METAL_DPRINT_RISCVS` - Specifies RISC-V processors for debug printing
- `TT_METAL_DPRINT_FILE` - Output file for debug printing
- `TT_METAL_DPRINT_ONE_FILE_PER_RISC` - Creates separate files per RISC-V
- `TT_METAL_DPRINT_PREPEND_DEVICE_CORE_RISC` - Prepends device/core/RISC info

### Memory Access Debug Delays
- `TT_METAL_READ_DEBUG_DELAY_*` - Controls read operation debug delays
- `TT_METAL_WRITE_DEBUG_DELAY_*` - Controls write operation debug delays
- `TT_METAL_ATOMIC_DEBUG_DELAY_*` - Controls atomic operation debug delays

### L1 Data Cache Control
- `TT_METAL_DISABLE_L1_DATA_CACHE_*` - Disables L1 data cache for specified targets

## Hardware and Core Management

### `TT_METAL_SKIP_ETH_CORES_WITH_RETRAIN`
Controls whether to skip Ethernet cores with retrain. Set to '0' or '1' to explicitly control behavior.

### `TT_METAL_SKIP_LOADING_FW`
Skips loading firmware when set, useful for testing scenarios.

### `TT_METAL_SKIP_DELETING_BUILT_CACHE`
Skips deletion of built cache when set, preserving compiled kernels between runs.
