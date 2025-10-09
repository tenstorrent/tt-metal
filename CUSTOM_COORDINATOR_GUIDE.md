# Guide: Selecting Custom Coordinator Device in Pytest

## Quick Answer

**YES!** You can select which device acts as coordinator by reordering `physical_device_ids`.

The **first device** in the list always becomes **logical device 0** (the coordinator).

---

## Why Device 0 is Always the Coordinator

**Code**: ```209:209:tt_metal/distributed/mesh_device.cpp```

```cpp
IDevice* MeshDevice::reference_device() const {
    return this->get_devices().at(0);  // ← Always index 0!
}
```

The MeshDevice treats `devices[0]` as the reference/coordinator for:
- Input embedding coordination
- AllGather output collection
- Device mesh control operations
- Higher L1 usage for coordination buffers

---

## Method 1: Environment Variable (Easiest!)

### Step 1: Use the Custom Conftest

Copy `conftest_custom_coordinator.py` to your test directory or merge it into your existing `conftest.py`.

### Step 2: Set Environment Variable

```bash
# Terminal 1: Start allocation monitor
cd tt_metal/programming_examples/alexp_examples/memory_utilization_monitor
./allocation_monitor_client -a

# Terminal 2: Run test with custom coordinator
export TT_COORDINATOR_DEVICE_ID=3  # Use device 3 as coordinator
export TT_ALLOC_TRACKING_ENABLED=1
pytest models/tt_transformers/demo/simple_text_demo.py -k "performance and batch-32"
```

### What Happens:

```
Physical Device IDs: [0, 1, 2, 3, 4, 5, 6, 7]
                      ↓ Reordered ↓
                     [3, 0, 1, 2, 4, 5, 6, 7]
                      ↓ Mapped to logical IDs ↓
Logical Device IDs:  [0, 1, 2, 3, 4, 5, 6, 7]
                      ↑
                  Coordinator!
```

### Expected Monitor Output:

```bash
# With TT_COORDINATOR_DEVICE_ID=3:
Device 3: L1 = 3.17 MB  ← NEW coordinator!
          DRAM = 7.32 GB
          TRACE = 16.05 MB

Devices 0,1,2,4,5,6,7: L1 = 73 KB  ← Workers
                       DRAM = 7.32 GB
                       TRACE = 16.05 MB
```

---

## Method 2: Direct API (For Custom Tests)

```python
import ttnn

# Define which physical device should be coordinator
coordinator_device_id = 5
all_device_ids = ttnn.get_device_ids()  # [0, 1, 2, 3, 4, 5, 6, 7]

# Reorder: coordinator first
physical_device_ids = [coordinator_device_id] + [d for d in all_device_ids if d != coordinator_device_id]
# Result: [5, 0, 1, 2, 3, 4, 6, 7]

# Open mesh with custom ordering
mesh_device = ttnn.open_mesh_device(
    mesh_shape=ttnn.MeshShape(1, 8),
    physical_device_ids=physical_device_ids,  # ← Custom order!
    trace_region_size=30000000,
    num_command_queues=2,
)

# Now physical device 5 is logical device 0 (coordinator)!
```

---

## Method 3: Update Existing Conftest

**File**: `models/tt_transformers/conftest.py`

Add this to your `device_params` fixture:

```python
@pytest.fixture
def device_params(request, galaxy_type):
    params = getattr(request, "param", {}).copy()

    # ... existing code ...

    # NEW: Support custom coordinator
    coordinator_id = os.environ.get("TT_COORDINATOR_DEVICE_ID")
    if coordinator_id is not None:
        coordinator_id = int(coordinator_id)
        all_device_ids = ttnn.get_device_ids()

        # Reorder: coordinator first
        physical_device_ids = [coordinator_id] + [d for d in all_device_ids if d != coordinator_id]
        params["physical_device_ids"] = physical_device_ids

        print(f"Using device {coordinator_id} as coordinator")

    return params
```

Then update your `mesh_device` fixture:

```python
@pytest.fixture(scope="function")
def mesh_device(request, device_params):
    mesh_device_shape = request.param
    mesh_shape = ttnn.MeshShape(*mesh_device_shape) if isinstance(mesh_device_shape, tuple) else None

    # Extract physical_device_ids if provided
    physical_device_ids = device_params.pop("physical_device_ids", [])

    mesh_device = ttnn.open_mesh_device(
        mesh_shape=mesh_shape,
        physical_device_ids=physical_device_ids,  # ← Pass custom ordering
        **device_params,
    )

    yield mesh_device

    ttnn.close_mesh_device(mesh_device)
```

---

## Method 4: Pytest Parameter (Per-Test Control)

```python
@pytest.mark.parametrize(
    "mesh_device,coordinator_id",
    [
        ((1, 8), 0),  # Device 0 as coordinator
        ((1, 8), 3),  # Device 3 as coordinator
        ((1, 8), 7),  # Device 7 as coordinator
    ],
    indirect=["mesh_device"],
)
def test_with_different_coordinators(mesh_device, coordinator_id):
    # Test will run 3 times, each with different coordinator
    print(f"Testing with device {coordinator_id} as coordinator")
    # ... your test code ...
```

---

## Real-World Example

### Test All Devices as Coordinator

```bash
#!/bin/bash
# test_all_coordinators.sh

echo "Testing memory patterns with different coordinators..."

for device_id in {0..7}; do
    echo ""
    echo "=========================================="
    echo "Testing with Device $device_id as coordinator"
    echo "=========================================="

    export TT_COORDINATOR_DEVICE_ID=$device_id
    pytest test_custom_coordinator.py::test_coordinator_memory_pattern -v -s

    # Give monitor time to update
    sleep 2
done

echo ""
echo "All coordinator tests complete!"
```

### Expected Results

You'll see the **high L1 usage (3.17 MB) rotate** to whichever device is coordinator:

```
TT_COORDINATOR_DEVICE_ID=0:
  Device 0: L1 = 3.17 MB  ← Coordinator
  Device 1-7: L1 = 73 KB

TT_COORDINATOR_DEVICE_ID=3:
  Device 3: L1 = 3.17 MB  ← Coordinator
  Device 0,1,2,4-7: L1 = 73 KB

TT_COORDINATOR_DEVICE_ID=7:
  Device 7: L1 = 3.17 MB  ← Coordinator
  Device 0-6: L1 = 73 KB
```

---

## Understanding the Mapping

```
┌────────────────────────────────────────────────────────────────┐
│           Physical to Logical Device Mapping                    │
├────────────────────────────────────────────────────────────────┤
│                                                                  │
│  physical_device_ids = [3, 0, 1, 2, 4, 5, 6, 7]                │
│                         ↓  ↓  ↓  ↓  ↓  ↓  ↓  ↓                 │
│  Logical Device IDs  =  0  1  2  3  4  5  6  7                 │
│                         │                                        │
│                         └─ Coordinator (reference_device())     │
│                                                                  │
│  Model code always sees logical IDs (0-7)                       │
│  Physical hardware receives commands for physical IDs           │
│  Your monitor shows PHYSICAL device IDs                         │
│                                                                  │
└────────────────────────────────────────────────────────────────┘
```

### Key Points:

1. **Model code** uses logical device IDs (always 0-7)
2. **Allocation monitor** shows physical device IDs (hardware)
3. **MeshDevice** handles the translation automatically
4. **Coordinator behavior** follows logical device 0

---

## Why This is Useful

### 1. **Testing Device Health**
Test if a specific physical device has issues:
```bash
# Test each device as coordinator to see if any fails
for i in {0..7}; do
    export TT_COORDINATOR_DEVICE_ID=$i
    pytest test_my_model.py || echo "Device $i FAILED as coordinator!"
done
```

### 2. **Load Balancing**
Distribute coordinator load across devices in long-running tests:
```bash
# Round-robin coordinator selection
export TT_COORDINATOR_DEVICE_ID=$((RANDOM % 8))
pytest test_long_inference.py
```

### 3. **Debugging**
Isolate coordinator-specific issues:
```bash
# Compare memory patterns
export TT_COORDINATOR_DEVICE_ID=0
pytest test_demo.py  # Check device 0 memory

export TT_COORDINATOR_DEVICE_ID=7
pytest test_demo.py  # Check device 7 memory
```

### 4. **Performance Testing**
Check if coordinator location affects performance:
```python
for device_id in range(8):
    os.environ["TT_COORDINATOR_DEVICE_ID"] = str(device_id)
    latency = run_inference_benchmark()
    print(f"Coordinator on device {device_id}: {latency}ms")
```

---

## Quick Reference

| What You Want | How To Do It |
|---------------|--------------|
| **Use device 3 as coordinator** | `export TT_COORDINATOR_DEVICE_ID=3` |
| **Default behavior** | Don't set `TT_COORDINATOR_DEVICE_ID` (device 0 used) |
| **Test all devices** | Loop over `TT_COORDINATOR_DEVICE_ID=0..7` |
| **Programmatic control** | `physical_device_ids=[3, 0, 1, 2, 4, 5, 6, 7]` in `open_mesh_device()` |
| **Check coordinator in monitor** | Look for device with highest L1 (~3MB vs ~73KB) |

---

## Troubleshooting

### Issue: Environment variable not working

**Solution**: Make sure you're using the custom conftest:
```bash
# Check if custom conftest is being used
pytest test_demo.py -v -s 2>&1 | grep "CUSTOM COORDINATOR"
```

### Issue: All devices show same L1

**Solution**: You might not be in decode mode with trace. Check:
```bash
# Should see trace capture
pytest test_demo.py -k "performance" -v -s 2>&1 | grep -i trace
```

### Issue: Invalid device ID error

**Solution**: Check available devices:
```python
import ttnn
print(f"Available devices: {ttnn.get_device_ids()}")
# Use only IDs from this list
```

---

## Summary

✅ **Yes, you can select coordinator!**
✅ **Method**: Reorder `physical_device_ids` (first = coordinator)
✅ **Easiest**: Use `TT_COORDINATOR_DEVICE_ID` environment variable
✅ **Verification**: Watch allocation monitor for high L1 device

The coordinator will **always** be logical device 0, but you control which **physical** device that maps to! 🎯
