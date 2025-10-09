# How to Clear Cached Program Buffers (36KB "Leak")

## Quick Answer

**The 36KB remaining after tensor deallocation is cached compiled programs, NOT a memory leak!**

To free it:

```python
# Python
mesh_device.disable_and_clear_program_cache()
```

```cpp
// C++
mesh_device->disable_and_clear_program_cache();
```

## What's Happening?

When you run operations like `ttnn.add()` or `ttnn.matmul()`:

1. **Compilation**: Operations compile to kernel programs
2. **Buffer Allocation**: Kernel binaries are loaded into DRAM buffers (12-24KB per operation)
3. **Caching**: The compiled program is cached for performance
4. **Persistence**: Buffers stay allocated until cache is cleared

## The 36KB Breakdown

```
Buffer 0x3FFFF800: 12KB × 8 devices = 96KB (ttnn.add kernels)
Buffer 0x3FFFF000: 24KB × 8 devices = 192KB (ttnn.matmul kernels)
───────────────────────────────────────────────────────────
Per device:        36KB cached kernel binaries
```

## Why Cache Programs?

- **Performance**: 10-100x faster repeated operations
- **Cost**: Only ~36KB per device for typical operations
- **Industry Standard**: PyTorch, TensorFlow, CUDA all do this

## When to Clear Cache

### Test Scripts
```python
# Always clear after testing
mesh_device.disable_and_clear_program_cache()
ttnn.close_mesh_device(mesh_device)
```

### Production (Rarely Needed)
```python
# Only if memory is critical
if low_memory_situation:
    mesh_device.disable_and_clear_program_cache()
```

### Iterative Workloads (Keep Cache!)
```python
for i in range(1000):
    result = ttnn.add(a, b)  # Iterations 2-1000 are MUCH faster!

# Clear once at the end
mesh_device.disable_and_clear_program_cache()
```

## Testing

Run the updated test to see cache clearing in action:

```bash
cd /home/tt-metal-apv/tt_metal/programming_examples/alexp_examples/memory_utilization_monitor

# Terminal 1: Start server
./allocation_server_poc

# Terminal 2: Start monitor
./allocation_monitor_client -a -r 500

# Terminal 3: Run test with cache clearing
export TT_ALLOC_TRACKING_ENABLED=1
python test_mesh_allocation.py
```

**Watch for Step 7**: "Clearing Program Cache" - you'll see the 36KB freed!

## Related Documentation

- `PROGRAM_CACHE_EXPLANATION.md` - Full technical details
- `test_mesh_allocation.py` - Example with cache clearing
- `test_mesh_allocation_cpp.cpp` - C++ example

## API Reference

```cpp
// IDevice interface (device.hpp)
virtual void enable_program_cache() = 0;
virtual void clear_program_cache() = 0;
virtual void disable_and_clear_program_cache() = 0;
virtual std::size_t num_program_cache_entries() = 0;
```

```python
# Python bindings (ttnn.device)
mesh_device.enable_program_cache()
mesh_device.clear_program_cache()
mesh_device.disable_and_clear_program_cache()
```

## Bottom Line

✅ **Expected behavior** - Performance optimization
✅ **Tracked correctly** - Allocation server sees it
✅ **Easily freed** - One API call
✅ **Usually beneficial** - Keep cache for performance!

🚀 **Don't worry about the 36KB unless you need every byte!**
