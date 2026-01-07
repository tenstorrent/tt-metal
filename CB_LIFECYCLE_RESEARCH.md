# Circular Buffer (CB) Lifecycle Research

## Executive Summary

Circular Buffers in TT-Metal have two types:
1. **Program-Local CBs**: Allocated per-program, deallocated when program is destroyed
2. **Global CBs**: Persistent L1 buffers that survive across programs until explicitly destroyed

**Key Finding**: CB memory stays allocated for the entire lifetime of the Program object. With program caching/tracing, this means CBs can remain allocated long after inference completes.

---

## 1. Program-Local Circular Buffers

### Definition
Regular circular buffers allocated as part of a Program. They exist in L1 memory on specified cores.

### Allocation Lifecycle

#### **Step 1: Creation (Python/C++ API)**
```python
# Python API
cb_handle = ttnn.CreateCircularBuffer(
    program=program,
    core_spec=CoreRange(...),
    config=CircularBufferConfig(total_size=2048, ...)
)
```

**What Happens:**
- `CircularBuffer` object is created and added to `program.circular_buffers_`
- Address is **NOT yet assigned** (locally_allocated_address = nullopt)
- CB configuration is stored but memory is not allocated yet

**Code Location:** `tt_metal/impl/program/program.cpp:789-795`

---

#### **Step 2: Allocation (Before First Execution)**
Called automatically before program execution:
```cpp
program.allocate_circular_buffers(device);
```

**What Happens:**
- For each CB in the program:
  - **Compute L1 address** based on core requirements
  - Mark address ranges in `CircularBufferAllocator::l1_regions`
  - Call `GraphTracker::track_allocate_cb()` → reports to AllocationServer
  - **Register program with device** → `device->register_program(this)`
  - Set `locally_allocated_address_` in the CB object

**Important:**
- No actual L1 memory allocation happens here!
- CBs just **reserve address ranges** in the L1 address space
- Physical memory is already part of L1 (always present)
- The "allocation" is really **address range reservation**

**Code Location:** `tt_metal/impl/program/program.cpp:846-975`

**Tracking:**
```cpp
// For each device and CB
GraphTracker::instance().track_allocate_cb(
    circular_buffer->core_ranges(),
    computed_addr,
    circular_buffer->size(),
    circular_buffer->globally_allocated(),  // false for local CBs
    device
);

// This registers the program with the device
device->register_program(this);
```

---

#### **Step 3: Usage (During Program Execution)**
- Device kernels read/write to the L1 addresses
- CB memory stays reserved for the program
- Multiple program executions can reuse the same CBs

---

#### **Step 4: Deallocation (Program Destruction)**

**Trigger:** `Program` object is destroyed (Python `del` or C++ destructor)

```cpp
detail::ProgramImpl::~ProgramImpl() noexcept {
    deallocate_circular_buffers();  // ← This is now implemented!
    deallocate_kernel_buffers();
    Inspector::program_destroyed(this);
}
```

**What Happens in `deallocate_circular_buffers()`:**
```cpp
void detail::ProgramImpl::deallocate_circular_buffers() {
    if (!this->cb_devices_.empty() && !this->circular_buffers_.empty()) {
        // 1. Notify tracking system (for tt-smi monitoring)
        for (const IDevice* idevice : this->cb_devices_) {
            GraphTracker::instance().track_deallocate_cb(idevice);
        }

        // 2. Unregister program from device (removes from active_programs_)
        for (const IDevice* idevice : this->cb_devices_) {
            auto* device = dynamic_cast<Device*>(const_cast<IDevice*>(idevice));
            if (device) {
                device->unregister_program(this);
                // Update SHM stats
                device->get_shm_stats_provider()->update_from_allocator(device, getpid());
            }
        }

        // 3. Clear internal tracking
        this->cb_devices_.clear();
    }
}
```

**Code Location:** `tt_metal/impl/program/program.cpp:1013-1036`

**Important Notes:**
- No actual L1 memory is "freed" - L1 is fixed hardware memory
- The address ranges become available for future CB allocations
- Tracking stats are updated immediately
- `device->unregister_program()` removes the program from `active_programs_` set

---

### Key Lifecycle Implications

#### **⚠️ CB Memory Persists Until Program Destruction**

If your application:
- **Caches programs** for reuse (common for performance)
- **Uses program tracing** (keeps programs alive for fast replay)
- **Holds Program objects** in memory

Then **CB memory stays allocated even after inference completes!**

**Example Scenario:**
```python
# Program 1: Allocates 10 MiB of CBs
program1 = create_my_program()
execute_program(program1)  # CBs allocated: 10 MiB

# Inference completes, but program1 still exists
# CB MEMORY STILL ALLOCATED: 10 MiB

# Program 2: Allocates another 5 MiB
program2 = create_another_program()
execute_program(program2)  # CBs allocated: +5 MiB

# CB MEMORY STILL ALLOCATED: 15 MiB total
# Both programs are alive!

# Only when programs are destroyed:
del program1  # CBs freed: -10 MiB
del program2  # CBs freed: -5 MiB
# NOW CB memory is 0 MiB
```

---

## 2. Global Circular Buffers

### Definition
Persistent L1 buffers that:
- Are allocated independently of programs
- Survive across multiple program executions
- Can be shared between programs
- Backed by actual `Buffer` objects (real memory allocation)

### Key Differences from Local CBs

| Aspect | Program-Local CB | Global CB |
|--------|------------------|-----------|
| **Lifetime** | Program lifetime | Until explicitly destroyed |
| **Memory** | Address reservation only | Actual Buffer allocation |
| **Sharing** | Single program | Multiple programs |
| **Deallocation** | Automatic (with program) | Manual (explicit destroy) |

### Allocation Lifecycle

#### **Step 1: Global CB Creation**
```python
global_cb = ttnn.create_global_circular_buffer(
    device=device,
    sender_receiver_core_mapping=[(sender_core, receiver_cores)],
    size=2048,
    buffer_type=ttnn.BufferType.L1
)
```

**What Happens:**
- Creates a **sharded Buffer** in L1 across specified cores
- Allocates **real L1 memory** via the device allocator
- Sets up sender/receiver configuration
- Buffer persists independently of any program

**Code Location:** `tt_metal/impl/buffers/global_circular_buffer.cpp:60-94`

**Important:** This actually allocates L1 memory from the device allocator!

---

#### **Step 2: Link Global CB to Program**
```python
# Link the global CB to a program's CB index
cb_handle = ttnn.CreateCircularBuffer(
    program=program,
    core_spec=CoreRange(...),
    config=CircularBufferConfig(...),
    global_circular_buffer=global_cb  # ← Link to global CB
)
```

**What Happens:**
- Program's CB uses the address from the global CB's backing buffer
- CB is marked as `globally_allocated() == true`
- When program is allocated, it **tracks** the global CB but doesn't allocate new memory

**Code Location:** `tt_metal/impl/program/program.cpp:798-806`

---

#### **Step 3: Program Deallocation**
When a program using a global CB is destroyed:

```cpp
if (circular_buffer->globally_allocated()) {
    // Global CBs are tracked but NOT deallocated with the program
    // The backing buffer persists
}
```

**The global CB memory stays allocated!**

**Code Location:** `tt_metal/impl/program/program.cpp:915-925`

---

#### **Step 4: Global CB Destruction**
Global CBs must be explicitly destroyed:

```python
# The global CB's backing buffer is deallocated
del global_cb
```

**What Happens:**
- The backing `Buffer` is deallocated
- L1 memory is freed back to the allocator
- All programs using this global CB become invalid

---

### Use Cases for Global CBs

1. **Cross-Program Communication**: Data passing between different programs
2. **Persistent State**: Maintaining state across program executions
3. **Synchronization**: Coordinating between cores across programs
4. **Memory Optimization**: Reusing the same L1 region for multiple programs

---

## 3. Physical Memory vs Address Reservation

### Critical Understanding

**Local CBs do NOT allocate new memory - they reserve addresses!**

- L1 memory is **fixed hardware memory** (~90 MiB per device)
- CBs mark address ranges as "in use"
- No malloc/free happens - just address space management
- Multiple CBs can share the same physical address if on different cores

**Example:**
```
Device L1 Memory (90 MiB total, always exists):
┌─────────────────────────────────────┐
│ Core (0,0): [Address 0x18000-0x1A000] │ ← Program 1 CB
│ Core (0,1): [Address 0x18000-0x1A000] │ ← Program 1 CB (same address, different core!)
│ Core (1,0): [Address 0x20000-0x22000] │ ← Program 2 CB
└─────────────────────────────────────┘
```

**Global CBs ARE different:**
- They actually allocate from the device allocator
- Use the Buffer system (real memory tracking)
- Can be DRAM or L1

---

## 4. Current Implementation in Your Codebase

### What Was Fixed Recently

✅ **`deallocate_circular_buffers()` is now implemented** (was missing before!)

```cpp
void detail::ProgramImpl::deallocate_circular_buffers() {
    if (!this->cb_devices_.empty() && !this->circular_buffers_.empty()) {
        // Track deallocations
        for (const IDevice* idevice : this->cb_devices_) {
            GraphTracker::instance().track_deallocate_cb(idevice);
        }

        // Unregister from devices
        for (const IDevice* idevice : this->cb_devices_) {
            auto* device = dynamic_cast<Device*>(const_cast<IDevice*>(idevice));
            if (device) {
                device->unregister_program(this);
                device->get_shm_stats_provider()->update_from_allocator(device, getpid());
            }
        }

        this->cb_devices_.clear();
    }
}
```

### Per-Device CB Tracking

✅ **`Device::get_total_cb_allocated()` now uses physical tracking:**

```cpp
uint64_t Device::get_total_cb_allocated() const {
    std::lock_guard<std::mutex> lock(active_programs_mutex_);

    // Collect per-core L1 regions from all active programs
    std::map<CoreCoord, std::vector<std::pair<uint64_t, uint64_t>>> device_regions_per_core;

    for (const auto* program : active_programs_) {
        auto program_regions = program->get_cb_l1_regions_per_core(this->id(), num_devices);

        for (const auto& [core, regions] : program_regions) {
            device_regions_per_core[core].insert(...);
        }
    }

    // Merge overlapping regions per core
    // Sum up physical usage
    // Return total
}
```

**This accounts for:**
- ✅ Address reuse when programs are cached
- ✅ Overlapping CB allocations
- ✅ Per-device tracking in mesh setups
- ✅ Physical L1 constraints (~90 MiB limit)

---

## 5. Summary & Implications

### When CB Memory is Allocated
1. **Program-Local CBs**: When `program.allocate_circular_buffers()` is called (before execution)
2. **Global CBs**: When `create_global_circular_buffer()` is called (explicit)

### When CB Memory is Freed
1. **Program-Local CBs**: When Program object is destroyed (`del program` or C++ destructor)
2. **Global CBs**: When global CB object is destroyed (explicit)

### Why CB Usage Stays High
If you observe CB usage remaining high after inference:
- ✅ **Expected if programs are cached/traced** (they hold CB allocations)
- ✅ **Expected if global CBs are used** (persist until explicitly destroyed)
- ⚠️ **Unexpected if all programs are destroyed** (would indicate a leak)

### How to Verify CB Behavior
```python
# Check initial CB usage
print(f"Initial CB: {device.get_total_cb_allocated() / 1024 / 1024} MiB")

# Create and execute program
program = create_program()
execute_program(program)
print(f"After execution: {device.get_total_cb_allocated() / 1024 / 1024} MiB")  # HIGH

# Destroy program
del program
print(f"After deletion: {device.get_total_cb_allocated() / 1024 / 1024} MiB")  # Should drop to 0
```

---

## 6. Mesh Device Considerations

In mesh setups:
- Each physical device has its own L1 memory (~90 MiB per device)
- Programs on mesh devices allocate CBs on **all devices**
- The same logical CoreCoord exists on every physical device
- CB tracking is now **per-device** (correctly accounts for this)

**Example: 8-device mesh**
```
Program allocates CB on cores (0,0)-(7,7), size 1 MiB:
- Device 0: 1 MiB on its cores (0,0)-(7,7)
- Device 1: 1 MiB on its cores (0,0)-(7,7)
- ...
- Device 7: 1 MiB on its cores (0,0)-(7,7)

Total across mesh: 8 MiB
Per device: 1 MiB each (correctly tracked now!)
```
