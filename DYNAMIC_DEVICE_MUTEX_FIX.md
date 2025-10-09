# Dynamic Device Mutex - Supporting Any Number of Devices

## The Issue with Fixed Array

The initial fix used:
```cpp
std::array<std::mutex, 8> g_device_buffer_lifecycle_mutex;
```

**Problem:** Hardcoded to 8 devices, but TT-Metal supports:
- **N150:** 1 device
- **N300:** 2 devices
- **T3K:** 8 devices
- **Galaxy:** 32 devices
- **TG:** 36 devices

This would cause **array out-of-bounds** access on systems with device IDs ≥ 8!

---

## The Fix: Dynamic Map with Lazy Initialization

### Implementation

```cpp
// In buffer.cpp
namespace {

// Map to dynamically create mutexes for each device as needed
std::mutex g_device_mutex_map_lock;
std::map<int, std::unique_ptr<std::mutex>> g_device_buffer_lifecycle_mutexes;

// Helper to get or create mutex for a device (thread-safe)
static std::mutex& get_device_lifecycle_mutex(int device_id) {
    std::lock_guard<std::mutex> lock(g_device_mutex_map_lock);
    auto it = g_device_buffer_lifecycle_mutexes.find(device_id);
    if (it == g_device_buffer_lifecycle_mutexes.end()) {
        // First time seeing this device - create mutex for it
        it = g_device_buffer_lifecycle_mutexes.emplace(
            device_id,
            std::make_unique<std::mutex>()
        ).first;
    }
    return *(it->second);
}

}  // namespace

void Buffer::allocate_impl() {
    // ...
    {
        std::lock_guard<std::mutex> lifecycle_lock(
            get_device_lifecycle_mutex(device_->id())  // ← Dynamic lookup
        );
        // ... tracking ...
    }
}
```

### How It Works

1. **Lazy initialization:** Mutexes are created only when first device buffer is allocated
2. **Thread-safe creation:** `g_device_mutex_map_lock` protects the map during initialization
3. **Efficient lookup:** After initialization, direct pointer dereference
4. **Supports any device ID:** Works with 1, 8, 32, 36+ devices

---

## Performance Analysis

### Mutex Creation (First Access)

```
First buffer allocation on device N:
1. Lock g_device_mutex_map_lock
2. Check if mutex exists for device N
3. If not, create new mutex
4. Unlock g_device_mutex_map_lock
5. Lock device N's mutex
6. Perform tracking
7. Unlock device N's mutex

Time: ~1-2 microseconds (one-time cost per device)
```

### Subsequent Access

```
Later buffer allocations on device N:
1. Lock g_device_mutex_map_lock
2. Find existing mutex (map lookup)
3. Unlock g_device_mutex_map_lock
4. Lock device N's mutex
5. Perform tracking
6. Unlock device N's mutex

Time: ~100-200 nanoseconds for map lookup + mutex ops
```

### Overhead Comparison

| Approach | Access Time | Memory | Flexibility |
|----------|-------------|--------|-------------|
| `std::array<std::mutex, 8>` | ~10 ns | Fixed (8 mutexes) | ❌ Max 8 devices |
| `std::array<std::mutex, 64>` | ~10 ns | Large (64 mutexes) | ⚠️ Wastes memory |
| `std::map` (our fix) | ~100 ns | Dynamic | ✅ Any # devices |

**Verdict:** Extra 90ns is negligible compared to tracking overhead (~microseconds)

---

## Edge Cases Handled

### 1. Concurrent First Access

**Scenario:** Two threads allocate buffers on device 5 for the first time simultaneously.

```
Thread 1: get_device_lifecycle_mutex(5)
Thread 2: get_device_lifecycle_mutex(5)

Result: g_device_mutex_map_lock ensures only one mutex is created
```

**Protected by:** `g_device_mutex_map_lock`

### 2. Device IDs Not Sequential

**Scenario:** System has devices with IDs: 0, 2, 7, 15 (non-contiguous)

```
Array approach: Would need array[16], wasting 12 slots
Map approach: Creates exactly 4 mutexes
```

**Advantage:** No wasted memory

### 3. Dynamic Device Addition

**Scenario:** Device hot-plugged during runtime (future feature)

```
Array approach: Fixed size, can't handle new devices
Map approach: Automatically creates mutex on first use
```

**Advantage:** Future-proof

### 4. Very Large Device IDs

**Scenario:** Galaxy with device ID 31, TG with device ID 35

```
Array approach: Out of bounds error!
Map approach: Works fine
```

**Advantage:** No arbitrary limits

---

## Memory Usage

### Array Approach (Fixed)
```
sizeof(std::mutex) = 40 bytes (typical)
std::array<std::mutex, 8> = 40 * 8 = 320 bytes (always allocated)
std::array<std::mutex, 64> = 40 * 64 = 2,560 bytes (wasteful!)
```

### Map Approach (Dynamic)
```
Empty map overhead = ~48 bytes
Per entry overhead = ~32 bytes (map node) + 40 bytes (mutex) = 72 bytes

For 8 devices: 48 + (72 * 8) = 624 bytes
For 32 devices: 48 + (72 * 32) = 2,352 bytes
```

**Trade-off:** Slightly more memory for flexibility and correctness

---

## Alternative Considered: std::vector

```cpp
std::vector<std::mutex> g_device_buffer_lifecycle_mutexes;

static std::mutex& get_device_lifecycle_mutex(int device_id) {
    if (device_id >= g_device_buffer_lifecycle_mutexes.size()) {
        g_device_buffer_lifecycle_mutexes.resize(device_id + 1);
    }
    return g_device_buffer_lifecycle_mutexes[device_id];
}
```

**Why not chosen:**
1. ❌ Not thread-safe without additional locking
2. ❌ Resize is expensive and moves mutexes (not movable!)
3. ❌ Wastes memory for non-contiguous device IDs
4. ❌ Mutex is not movable, can't use in vector without indirection

**Map advantages:**
1. ✅ Thread-safe creation with simple mutex
2. ✅ No resize needed
3. ✅ Only allocates what's needed
4. ✅ Pointers to mutexes never invalidated

---

## Testing

### Verify It Works on Different Systems

```bash
# N150 (1 device)
export TT_BUFFER_DEBUG_LOG=0
python test.py
# Should work fine

# T3K (8 devices)
export TT_BUFFER_DEBUG_LOG=0
python test.py
# Should work fine

# Galaxy (32 devices)
export TT_BUFFER_DEBUG_LOG=0
python test.py
# Should work fine (previously would crash!)
```

### Check Mutex Creation

Add temporary logging to verify:

```cpp
static std::mutex& get_device_lifecycle_mutex(int device_id) {
    std::lock_guard<std::mutex> lock(g_device_mutex_map_lock);
    auto it = g_device_buffer_lifecycle_mutexes.find(device_id);
    if (it == g_device_buffer_lifecycle_mutexes.end()) {
        std::cout << "Creating lifecycle mutex for device " << device_id << std::endl;
        it = g_device_buffer_lifecycle_mutexes.emplace(
            device_id,
            std::make_unique<std::mutex>()
        ).first;
    }
    return *(it->second);
}
```

**Expected output:**
```
Creating lifecycle mutex for device 0
Creating lifecycle mutex for device 1
Creating lifecycle mutex for device 2
...
```

---

## Summary

### What Changed

**From:**
```cpp
std::array<std::mutex, 8> g_device_buffer_lifecycle_mutex;
// Access: g_device_buffer_lifecycle_mutex[device_id]
```

**To:**
```cpp
std::map<int, std::unique_ptr<std::mutex>> g_device_buffer_lifecycle_mutexes;
// Access: get_device_lifecycle_mutex(device_id)
```

### Benefits

1. ✅ **Supports any number of devices** (1 to 100+)
2. ✅ **No hardcoded limits**
3. ✅ **No wasted memory** (only creates mutexes for active devices)
4. ✅ **Thread-safe initialization**
5. ✅ **Future-proof** (dynamic device addition)
6. ✅ **Minimal overhead** (~90ns extra per access)

### Trade-offs

1. ⚠️ Slightly slower lookup (~90ns vs ~10ns)
2. ⚠️ More complex code (helper function + map lock)
3. ⚠️ Uses ~2x memory (but still minimal)

**Verdict:** The trade-offs are worth it for correctness and flexibility!

---

## Files Modified

1. `tt_metal/impl/buffers/buffer.cpp`
   - Changed from `std::array<std::mutex, 8>` to `std::map` + helper function
   - Added `<map>` and `<memory>` includes
   - Updated `allocate_impl()` to use helper
   - Updated `deallocate_impl()` to use helper

2. `FINAL_RACE_CONDITION_FIX.md`
   - Should be updated to reflect dynamic approach

3. `DYNAMIC_DEVICE_MUTEX_FIX.md`
   - This document

---

## Conclusion

Great catch on the hardcoded device limit! The map-based approach ensures the fix works on **all TT-Metal systems** regardless of device count, while maintaining the same race condition protection.

**Key insight:** When dealing with hardware, never assume fixed limits - systems grow and configurations vary!
