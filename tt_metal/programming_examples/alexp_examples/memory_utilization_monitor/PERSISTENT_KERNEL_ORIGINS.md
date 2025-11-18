# Persistent Kernel Origins - Investigation Results

## Summary

The **~204 KB (~0.2 MB) of persistent kernels** that remain in L1 memory after program execution are **system infrastructure kernels** loaded during device initialization. These are **NOT application kernels** and are **expected to remain** until device close.

## Breakdown of the 4 Persistent Kernels (per device)

### 1. **Fabric Kernels (2× 56 KB = 112 KB)**

**Call Stack:**
```
tt::tt_metal::Device::configure_fabric()
  ↓
tt::DevicePool::init_fabric()
  ↓
tt::DevicePool::initialize_active_devices()
  ↓
tt::DevicePool::initialize_fabric_and_dispatch_fw()
```

**Purpose:**
- **Fabric routing kernels** for inter-device communication
- Handle mesh/grid communication between devices
- Enable distributed workloads across multiple devices
- Part of the Fabric subsystem for multi-chip communication

### 2. **Command Queue / Dispatch Kernels (2× 46 KB = 92 KB)**

**Call Stack:**
```
tt::tt_metal::Device::configure_command_queue_programs()
  ↓
tt::tt_metal::Device::init_command_queue_device()
  ↓
tt::DevicePool::initialize_active_devices()
  ↓
tt::DevicePool::initialize_fabric_and_dispatch_fw()
```

**Purpose:**
- **Fast Dispatch infrastructure kernels**
- Prefetch kernel (fetches commands from DRAM)
- Dispatch kernel (executes commands on tensix cores)
- Completion queue kernel (reports completion status)
- Enable the command queue system for fast program execution

## Total Per Device

- **2× 56 KB (Fabric) = 112 KB**
- **2× 46 KB (Dispatch) = 92 KB**
- **Total = 204 KB** ✓

## When Are These Loaded?

These kernels are loaded during **device initialization** at the following point:

```
ttnn::distributed::open_mesh_device()
  ↓
tt::tt_metal::distributed::MeshDevice::create()
  ↓
tt::DevicePool::initialize_fabric_and_dispatch_fw()
  ↓
  ├─ initialize_active_devices()
  │   ├─ init_fabric() → Fabric kernels (2× 56 KB)
  │   └─ init_command_queue_device() → Dispatch kernels (2× 46 KB)
```

This happens **before any user code runs**, as part of the device setup.

## When Are These Unloaded?

These kernels are unloaded when:

### During Normal Operation:
- **Device close** (`ttnn.close_device()`) - explicitly deallocates them
- **Device reset**
- **Process exit** (clean shutdown)

### During Abnormal Operation:
- **Process crash/kill** - the allocation server's cleanup thread will detect the dead process and remove all its kernels (including system kernels) within ~10 seconds

### Automatic Cleanup

The allocation server has a **background cleanup thread** that:
1. Runs every 10 seconds
2. Checks if tracked PIDs are still alive
3. Removes all allocations (buffers, CBs, kernels) from dead processes
4. Updates device statistics accordingly

**This means system kernels WILL be cleaned up automatically** even if the process crashes without calling `close_device()`!

Example cleanup output:
```
⚠️  Detected dead processes, cleaning up orphaned buffers...
   PID 12345 is dead, removing its buffers...
   ✓ Removed 4 kernels (0.199219 MB) from PID 12345
```

## Why Don't They Show Up in `program.impl().deallocate_kernel_buffers()`?

Because they're not associated with any user `Program` object. They are:
- Loaded by the `Device` class directly
- Managed by `DevicePool`
- Part of the system infrastructure, not application code

## Verification

Run with backtrace enabled to see the exact call stacks:

```bash
# Build with backtrace support (already done)
cd /home/ttuser/aperezvicente/tt-metal-apv

# Run any program and capture kernel loads
export TT_ALLOC_TRACKING_ENABLED=1
python3 -c "import ttnn; dev = ttnn.open_device(0); ttnn.close_device(dev)" 2>&1 | grep -A 10 "🔍 KERNEL_LOAD"
```

The first 4 kernel loads will show `configure_fabric` and `configure_command_queue_programs` in their call stacks.

## Conclusion

✅ **The 204 KB of persistent kernels are CORRECT and EXPECTED**

They are:
- ✅ System infrastructure (Fabric + Dispatch)
- ✅ Loaded during device initialization
- ✅ Required for device operation
- ✅ Not a memory leak
- ✅ Not application kernels
- ✅ Will be cleaned up on device close

**This is normal, healthy behavior!**

---

## How to Distinguish System vs Application Kernels

If you want to track only **application kernels** (user program kernels), you could:

1. **Ignore the first 4 kernels per device** (simple approach)
2. **Add a flag to track kernel type** (system vs application) in `track_kernel_load`
3. **Baseline the memory** - take a snapshot after device init, then track deltas
4. **Use kernel IDs** - system kernels have consistent IDs across runs

For monitoring purposes, consider reporting:
- **System Kernels: ~0.2 MB** (fixed baseline)
- **Application Kernels: X MB** (varies with workload)
- **Total: 0.2 + X MB**
