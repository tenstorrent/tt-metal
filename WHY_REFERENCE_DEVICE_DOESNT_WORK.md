# Why Changing `reference_device()` Doesn't Work

## TL;DR

❌ **Changing `reference_device()` doesn't work** - it's just a query function
✅ **Use `physical_device_ids` reordering** - this is the only way

---

## What You Tried

```cpp
// In tt_metal/distributed/mesh_device.cpp:209
IDevice* MeshDevice::reference_device() const {
    return this->get_devices().at(3);  // Changed from 0 to 3
}
```

**Result**: Device 0 still has high L1 (still the coordinator)

---

## Why It Didn't Work

### `reference_device()` is just a query function!

It's used for:
- Getting device properties (L1 size, DRAM size)
- Checking if devices are initialized
- Validation checks

```cpp
// Examples of reference_device() usage:
uint32_t MeshDevice::l1_size_per_core() const {
    return this->reference_device()->l1_size_per_core();  // Just querying a property!
}

bool MeshDevice::is_initialized() const {
    return this->reference_device()->is_initialized();  // Just checking status!
}
```

**It does NOT control**:
- Which device handles embeddings
- Which device collects AllGather outputs
- Which device has coordination buffers

### The Real Coordinator Logic

The coordinator behavior is **baked into the code** as **"use logical device 0"**:

#### Example 1: Tensor Mappers

**File**: `ttnn/cpp/ttnn/tensor/types.cpp`

```cpp
// ReplicateTensorToMesh: puts tensor on all devices
// BUT: source is always device 0!
std::vector<Tensor> result;
for (int i = 0; i < num_devices; i++) {
    result.push_back(ttnn.replicate(tensor_on_device_0));  // Device 0 is source!
}
```

#### Example 2: Mesh Composers

**File**: `ttnn/cpp/ttnn/tensor/tensor_impl.cpp`

```cpp
// ConcatMesh2dToTensor: gathers tensors from all devices
// BUT: result is stored on device 0!
Tensor output = create_tensor_on_device(devices[0]);  // Device 0 is target!
for (auto& device_tensor : device_tensors) {
    concat_into(output, device_tensor);
}
```

#### Example 3: Model Code

**File**: `models/tt_transformers/tt/model.py:299`

```python
def concat_host_output(self, tt_out):
    # Gets tensors from all devices
    torch_out_tensors = [ttnn.to_torch(x) for x in ttnn.get_device_tensors(tt_out)]

    # Concatenates them (device 0's tensor is first in list!)
    # This is why device 0 needs more memory - it coordinates the gathering
    return torch.cat(torch_out_tensors, dim=...)
```

---

## How Coordinationation Actually Works

```
┌──────────────────────────────────────────────────────────────────┐
│           Coordinator Behavior (Logical Device 0)                 │
├──────────────────────────────────────────────────────────────────┤
│                                                                    │
│  1. Input Embeddings:                                            │
│     - ReplicateTensorToMesh uses device 0 as source              │
│     - Device 0 needs buffer space for original tensor            │
│     → Higher L1/DRAM usage                                        │
│                                                                    │
│  2. Output Gathering:                                            │
│     - AllGather collects results to device 0                     │
│     - Device 0 holds full concatenated output                    │
│     → Higher L1/DRAM usage                                        │
│                                                                    │
│  3. Tensor Materialization:                                      │
│     - ttnn.to_torch() called first on device 0's tensor          │
│     - Other devices' tensors copied to device 0 for concat       │
│     → Higher memory traffic                                       │
│                                                                    │
│  4. Mesh Control:                                                │
│     - Synchronization barriers coordinated from device 0         │
│     - Fabric control messages routed through device 0            │
│     → Additional control structures in L1                         │
│                                                                    │
│  ALL OF THIS assumes LOGICAL device 0!                          │
│                                                                    │
└──────────────────────────────────────────────────────────────────┘
```

---

## The Correct Solution: Physical Device Reordering

### How MeshDevice is Created

```cpp
// tt_metal/distributed/mesh_device.cpp:229-294
std::shared_ptr<MeshDevice> MeshDevice::create(
    const MeshDeviceConfig& config,
    ...
) {
    // Get physical device IDs
    const auto& physical_ids = config.physical_device_ids();

    // Create devices in the order specified by physical_ids
    auto scoped_devices = std::make_shared<ScopedDevices>(
        wrap_to_maybe_remote(physical_ids),  // ← Order matters!
        ...
    );

    // Store in MeshDevice
    auto mesh_device = std::make_shared<MeshDevice>(
        std::move(scoped_devices),
        ...
    );

    // devices[0] is now physical_ids[0]!
    // devices[1] is now physical_ids[1]!
    // etc.
}
```

### The Trick: Reorder Physical IDs

```python
# Default (coordinator = physical device 0):
physical_device_ids = []  # Uses system default: [0, 1, 2, 3, 4, 5, 6, 7]

# Result:
#   Logical 0 = Physical 0 (coordinator)
#   Logical 1 = Physical 1
#   ...

# Custom (coordinator = physical device 3):
physical_device_ids = [3, 0, 1, 2, 4, 5, 6, 7]  # Put 3 first!

# Result:
#   Logical 0 = Physical 3 (coordinator!)  ← This is the trick!
#   Logical 1 = Physical 0
#   Logical 2 = Physical 1
#   Logical 3 = Physical 2
#   Logical 4 = Physical 4
#   ...
```

### Why This Works

```
All the code that does:

    devices[0]->allocate(...)
    devices.at(0)->function()
    reference_device()->query()

Now operates on physical device 3 (because it's at index 0)!

No code changes needed - the reordering happens at creation time.
```

---

## Step-by-Step: Making Physical Device 3 the Coordinator

### Option 1: Environment Variable (with custom conftest)

```bash
export TT_COORDINATOR_DEVICE_ID=3
pytest test_my_model.py -k "batch-32" -v -s
```

The conftest will:
```python
coordinator_id = int(os.environ["TT_COORDINATOR_DEVICE_ID"])  # 3
all_device_ids = ttnn.get_device_ids()  # [0, 1, 2, 3, 4, 5, 6, 7]

# Reorder: coordinator first
physical_device_ids = [coordinator_id] + [d for d in all_device_ids if d != coordinator_id]
# Result: [3, 0, 1, 2, 4, 5, 6, 7]

mesh_device = ttnn.open_mesh_device(
    physical_device_ids=physical_device_ids  # ← Reordered!
)
```

### Option 2: Direct API

```python
mesh_device = ttnn.open_mesh_device(
    mesh_shape=ttnn.MeshShape(1, 8),
    physical_device_ids=[3, 0, 1, 2, 4, 5, 6, 7],  # 3 is first!
    ...
)
```

### Option 3: Modify Your Test

```python
# In your test file
def my_custom_mesh_fixture():
    # Hardcode coordinator selection
    physical_device_ids = [3, 0, 1, 2, 4, 5, 6, 7]

    mesh = ttnn.open_mesh_device(
        mesh_shape=ttnn.MeshShape(1, 8),
        physical_device_ids=physical_device_ids,
        trace_region_size=30000000,
    )

    yield mesh

    ttnn.close_mesh_device(mesh)
```

---

## Verification

### Check in Your Monitor

```bash
# Before (default coordinator = physical device 0):
Device 0: L1 = 3.17 MB  ← High
Device 1: L1 = 73 KB
Device 2: L1 = 73 KB
Device 3: L1 = 73 KB    ← Low
...

# After (custom coordinator = physical device 3):
Device 0: L1 = 73 KB    ← Now low!
Device 1: L1 = 73 KB
Device 2: L1 = 73 KB
Device 3: L1 = 3.17 MB  ← Now high!
...
```

### Quick Test Script

Run: `python test_coordinator_reordering.py`

This will:
1. Create mesh with physical device 3 as logical device 0
2. Create tensors (triggers coordinator behavior)
3. Show you which device has high L1 in your monitor

---

## Summary

| What | Does it Work? | Why |
|------|---------------|-----|
| **Change `reference_device()`** | ❌ No | Just a query function, doesn't control behavior |
| **Modify C++ coordinator logic** | ❌ Too complex | Behavior is distributed across many files |
| **Reorder `physical_device_ids`** | ✅ YES! | Changes which physical device is at index 0 (logical device 0) |

### The Key Insight

**Model code uses LOGICAL device IDs** (always 0-7)
**Your monitor shows PHYSICAL device IDs** (hardware)
**physical_device_ids creates the mapping** between them!

```
physical_device_ids = [3, 0, 1, 2, 4, 5, 6, 7]
                       ↓ Maps to logical IDs ↓
                      [0, 1, 2, 3, 4, 5, 6, 7]
                       ↑
                  Coordinator (always logical 0)
```

So when code does `devices[0]->allocate()`, it operates on physical device 3! 🎯
