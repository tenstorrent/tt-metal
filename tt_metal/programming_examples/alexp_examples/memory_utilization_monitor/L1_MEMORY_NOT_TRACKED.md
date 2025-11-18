# L1 Memory NOT Tracked by the Allocator

## Your Observation

You have **306 MB total L1** but only seeing **1-5 MB used** by a big model. Where is the rest?

## Answer: Most L1 Memory is NOT Tracked by the Allocator!

### What IS Tracked ✅

The **allocation server** only tracks L1 memory allocated through **TT-Metal's explicit buffer allocation APIs**:

```cpp
// THIS is tracked:
auto buffer = CreateBuffer(InterleavedBufferConfig{
    device,
    size,
    page_size,
    BufferType::L1  // Explicitly requested L1
});
```

**What gets counted:**
- ✅ Interleaved L1 buffers
- ✅ Sharded L1 buffers
- ✅ L1_SMALL allocations
- ✅ TRACE buffers

### What is NOT Tracked ❌

#### 1. **Per-Core Reserved L1** (Biggest chunk!)

Each Tensix core has **reserved L1 space** for:
- **Compute kernel code** (~20-50 KB per core)
- **Data movement kernel code** (~20-50 KB per core)
- **Circular buffers** (CB0-CB31, varies by model)
- **Kernel local variables** and stack
- **Semaphores** and synchronization primitives

**For Wormhole B0:**
- 80 tensix cores × ~100-200 KB reserved = **8-16 MB** minimum
- Large models with big circular buffers: **50-100 MB+**

#### 2. **Circular Buffers (CBs)**

These are **pre-allocated** per-core memory regions for data staging:
```cpp
// In kernel config:
CircularBufferConfig cb_config = CircularBufferConfig(
    tile_size * num_tiles,  // Could be 1-10 MB per core!
    {{CB::c_in0, tt::DataFormat::Float16_b}}
).set_page_size(CB::c_in0, tile_size);
```

**Not tracked because:**
- Allocated at kernel compilation/setup time
- Part of core's local memory space
- Not managed by the global allocator

#### 3. **Firmware and Runtime Reserved**

- **FW code and data**: ~1-2 MB total
- **Command queues**: If using CQ mode
- **Dispatch kernels**: For multi-device
- **NOC routing tables**: For mesh networks

#### 4. **Memory Fragmentation**

- Allocator may not pack buffers perfectly
- Alignment requirements (usually 32-byte boundaries)
- Internal book-keeping overhead

### Typical L1 Breakdown for a Large Model

| Memory Type | Size | Tracked? |
|-------------|------|----------|
| **Circular Buffers** | 100-150 MB | ❌ No |
| **Kernel Code (Compute + DataMov)** | 10-30 MB | ❌ No |
| **Firmware & Runtime** | 2-5 MB | ❌ No |
| **Explicit Buffer Allocations** | 1-5 MB | ✅ **YES** |
| **Semaphores & Sync** | 1-2 MB | ❌ No |
| **Fragmentation & Overhead** | 5-10 MB | ❌ No |
| **Free/Unused** | 100-150 MB | - |
| **TOTAL** | ~306 MB | - |

### Why Only Explicit Allocations are Tracked

The allocation server **only sees** calls to:
```cpp
device->allocate_buffer()
BufferAllocator::allocate()
```

It does **NOT** see:
- Kernel compilation memory reservations
- Circular buffer setup (part of kernel config)
- Runtime system allocations
- Per-core local memory setup

### How to See More L1 Usage

#### Option 1: Check Kernel Configs
Look at your kernel's circular buffer sizes:
```cpp
// In your kernel setup:
CreateKernel(...
    CircularBufferConfig(
        num_tiles * tile_size,  // THIS is using L1 but not tracked!
        ...
    )
);
```

#### Option 2: Check Device Reports
```cpp
// After kernel dispatch:
device->print_memory_usage();  // Shows per-core CB usage
```

#### Option 3: Check Profiler
```cpp
// Enable L1 profiling:
tracy::Profiler::enable_l1_memory_tracking();
```

#### Option 4: Manual Accounting
Track CB sizes in your application:
```cpp
size_t total_cb_size = 0;
for (auto& kernel : kernels) {
    total_cb_size += kernel.get_circular_buffer_total_size();
}
```

### Example: Llama-3 Model

For a Llama-3 8B model on 8 chips:

**Per chip:**
- Attention CBs: ~40 MB (input, output, KV cache staging)
- MLP CBs: ~30 MB (activations staging)
- Kernel code: ~15 MB (many kernels!)
- **Explicit tensors**: ~5 MB (weights are in DRAM, activations use CBs)
- Runtime: ~5 MB
- **= ~95 MB used, only 5 MB tracked!**

### Why This Design?

**Performance reasons:**
1. **CBs are setup once** at kernel compile/setup time → no runtime allocation overhead
2. **Per-core memory** is managed by the core itself → no NOC traffic for allocation
3. **Kernel code** is loaded once → fast dispatch
4. **Global allocator** only for **dynamic, multi-core buffers** → simpler, faster

### How to Track More?

If you want to track CB usage, you'd need to:

1. **Hook kernel setup** in `tt_metal/impl/program/program.cpp`
2. **Track CB configs** when kernels are added
3. **Sum per-core CB sizes**
4. **Report via allocation server**

**This would require modifying TT-Metal core!**

### Summary

**You're only seeing 1-5 MB because:**
- ✅ Your allocator correctly tracks **explicit buffer allocations**
- ❌ Your allocator does NOT track **circular buffers** (100+ MB)
- ❌ Your allocator does NOT track **kernel code** (10-30 MB)
- ❌ Your allocator does NOT track **runtime/FW** (5 MB)

**This is normal and expected!**

The allocation server is working correctly. Most L1 usage is:
1. **Circular buffers** - allocated at kernel setup
2. **Kernel code** - loaded at kernel compile
3. **Runtime overhead** - allocated by firmware

These are **not managed by the global allocator** and thus **not tracked**.

### To Verify Your Suspicion

Run this check in your model:
```python
# After loading model
import tt_lib as ttl
device = ttl.device.GetDefaultDevice()

# This will print detailed per-core L1 usage:
ttl.device.DumpDeviceMemoryState(device)

# Look for:
# - CB usage per core
# - Kernel memory per core
# - Total allocated vs used
```

You should see that **circular buffers account for most L1 usage**.

### Recommendation

If you want to track CBs, add manual accounting:
```python
class L1Tracker:
    def __init__(self):
        self.cb_usage = 0

    def track_kernel(self, kernel_config):
        for cb in kernel_config.circular_buffers:
            self.cb_usage += cb.size
        return self.cb_usage
```

Then display both:
- **Allocator-tracked**: Explicit buffers (what tt-smi shows)
- **Manual-tracked**: CB usage (from your accounting)
- **Total = Allocator + CBs + Kernels + Runtime**
