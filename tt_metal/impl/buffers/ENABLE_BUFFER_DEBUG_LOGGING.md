# Enable Buffer Debug Logging in TT-Metal

## Quick Start

### Step 1: Apply the Debug Logging Patch

```bash
cd /workspace/tt-metal-apv

# Apply the patch
patch -p1 < tt_metal/impl/buffers/ADD_BUFFER_DEBUG_LOGGING.patch

# Rebuild TT-Metal
cmake --build build -j$(nproc)
```

### Step 2: Run Your Application with Debug Logging

```bash
# Enable buffer debug logging
export TT_BUFFER_DEBUG_LOG=1

# Run your application
python your_test.py

# Debug log will be written to /tmp/tt_buffer_debug.log
```

### Step 3: Analyze the Debug Log

```bash
# View the log
less /tmp/tt_buffer_debug.log

# Count allocations
grep "BUFFER ALLOCATED" /tmp/tt_buffer_debug.log | wc -l

# Count deallocations
grep "BUFFER DEALLOCATED" /tmp/tt_buffer_debug.log | wc -l

# Find leaked buffers (allocated but never deallocated)
# Look for addresses that appear in ALLOCATED but not in DEALLOCATED
```

---

## What the Debug Log Shows

### Buffer Allocation Example:
```
═══════════════════════════════════════════════════════════
BUFFER ALLOCATED
Time: 14:32:15
Device: 0
Address: 0x18862c0 (25718464)
Size: 524288 bytes (512 KB)
Type: DRAM
Buffer*: 0x7f8a4c001230
Owns Data: yes
Hooked: no
Call Stack:
    /path/to/libtensorrent.so: tt::tt_metal::Buffer::allocate_impl()
    /path/to/libtensorrent.so: tt::tt_metal::Buffer::create(...)
    /path/to/libtensorrent.so: ttnn::operations::matmul::create_sharded_tensor(...)
    /path/to/_ttnn.so: PyInit__ttnn
    python3: PyEval_EvalFrameDefault
    python3: PyRun_SimpleFileObject
    python3: Py_RunMain
    python3: main
```

This shows:
- **Exactly where** the buffer was allocated (call stack)
- **What function** triggered it (e.g., `create_sharded_tensor`)
- **Buffer details:** address, size, type
- **Timestamp:** when it was created

### Buffer Deallocation Example:
```
═══════════════════════════════════════════════════════════
BUFFER DEALLOCATED
Time: 14:32:16
Device: 0
Address: 0x18862c0 (25718464)
Size: 524288 bytes (512 KB)
Type: DRAM
Buffer*: 0x7f8a4c001230
Call Stack:
    /path/to/libtensorrent.so: tt::tt_metal::Buffer::deallocate_impl()
    /path/to/libtensorrent.so: tt::tt_metal::Buffer::~Buffer()
    /path/to/libtensorrent.so: std::shared_ptr<Buffer>::~shared_ptr()
    ...
```

---

## Finding Leaked Buffers

### Method 1: Compare Allocations vs Deallocations

```bash
# Extract all buffer addresses that were allocated
grep "BUFFER ALLOCATED" /tmp/tt_buffer_debug.log | grep "Address:" | awk '{print $2}' | sort > /tmp/allocated.txt

# Extract all buffer addresses that were deallocated
grep "BUFFER DEALLOCATED" /tmp/tt_buffer_debug.log | grep "Address:" | awk '{print $2}' | sort > /tmp/deallocated.txt

# Find buffers that were allocated but never deallocated
comm -23 /tmp/allocated.txt /tmp/deallocated.txt > /tmp/leaked_addresses.txt

# Show details of leaked buffers
for addr in $(cat /tmp/leaked_addresses.txt | head -10); do
    echo "=== Leaked Buffer: $addr ==="
    grep -A 15 "Address: $addr" /tmp/tt_buffer_debug.log | grep -A 15 "BUFFER ALLOCATED" | head -20
    echo ""
done
```

### Method 2: Python Analysis Script

Create `/tmp/analyze_buffer_log.py`:

```python
#!/usr/bin/env python3
import re
from collections import defaultdict

def parse_buffer_log(filename):
    with open(filename, 'r') as f:
        content = f.read()

    # Split by separator
    events = content.split('═══════════════════════════════════════════════════════════')

    allocated = {}
    deallocated = set()

    for event in events:
        if 'BUFFER ALLOCATED' in event:
            addr_match = re.search(r'Address: (0x[0-9a-f]+)', event)
            if addr_match:
                addr = addr_match.group(1)
                allocated[addr] = event
        elif 'BUFFER DEALLOCATED' in event:
            addr_match = re.search(r'Address: (0x[0-9a-f]+)', event)
            if addr_match:
                deallocated.add(addr_match.group(1))

    # Find leaks
    leaked = {addr: info for addr, info in allocated.items() if addr not in deallocated}

    print(f"Total allocations: {len(allocated)}")
    print(f"Total deallocations: {len(deallocated)}")
    print(f"Leaked buffers: {len(leaked)}")
    print()

    if leaked:
        print("LEAKED BUFFERS:")
        print("=" * 80)
        for addr, info in list(leaked.items())[:10]:  # Show first 10
            print(f"\nAddress: {addr}")
            # Extract key info
            size_match = re.search(r'Size: (\d+)', info)
            type_match = re.search(r'Type: (\w+)', info)
            stack_match = re.search(r'Call Stack:\n((?:    .*\n)+)', info)

            if size_match:
                print(f"Size: {int(size_match.group(1)) / 1024:.1f} KB")
            if type_match:
                print(f"Type: {type_match.group(1)}")
            if stack_match:
                print("Call stack:")
                print(stack_match.group(1))

if __name__ == '__main__':
    parse_buffer_log('/tmp/tt_buffer_debug.log')
```

Run it:
```bash
python3 /tmp/analyze_buffer_log.py
```

---

## Understanding the Call Stacks

### Example Call Stack for matmul:
```
Call Stack:
    libtensorrent.so: tt::tt_metal::Buffer::allocate_impl()
    libtensorrent.so: tt::tt_metal::Buffer::create(IDevice*, unsigned long, unsigned long, ...)
    libtensorrent.so: ttnn::operations::matmul::create_device_tensor(...)
    libtensorrent.so: ttnn::operations::matmul::MatmulOperation::operator()(...)
    _ttnn.so: pybind11::cpp_function::dispatcher()
```

**Interpretation:**
- Buffer was created by **matmul operation**
- Called from Python via pybind11
- You can trace back to `MatmulOperation::operator()` in the code

### Example Call Stack for circular buffer:
```
Call Stack:
    libtensorrent.so: tt::tt_metal::CircularBuffer::allocate()
    libtensorrent.so: tt::tt_metal::Program::allocate_circular_buffers()
    libtensorrent.so: tt::tt_metal::EnqueueProgram()
```

---

## Debugging Specific Issues

### Issue 1: Buffer Address Reused
```bash
# Find all events for a specific address
grep -A 15 "Address: 0x18862c0" /tmp/tt_buffer_debug.log
```

You'll see:
```
BUFFER ALLOCATED - Size: 524288 (512 KB)
... call stack from matmul ...

BUFFER DEALLOCATED
... call stack from destructor ...

BUFFER ALLOCATED - Size: 98304 (96 KB)  ← Same address, different size!
... call stack from another operation ...
```

### Issue 2: Destructor Never Called

If you see allocations but no corresponding deallocations, check if:
1. **Exception thrown:** Look for exceptions in your Python code
2. **Reference leak:** `shared_ptr` held somewhere
3. **Program exit:** Process killed before cleanup

Add reference counting debug:
```python
import gc
import ttnn

# After your test
print("Forcing garbage collection...")
gc.collect()

# Check for remaining tensor references
print("Remaining tensors:", len([obj for obj in gc.get_objects() if isinstance(obj, ttnn.Tensor)]))
```

---

## Performance Impact

**Note:** This debug logging has performance overhead due to:
- File I/O for every allocation/deallocation
- Stack trace capture (backtrace)
- String formatting

**Use only for debugging, not production!**

To minimize impact:
1. Only enable for specific devices: Modify the patch to check `device_->id() == 0`
2. Only log specific buffer types: Add check for `buffer_type_ == BufferType::L1`
3. Disable after finding the issue

---

## Reverting the Patch

```bash
cd /workspace/tt-metal-apv

# Revert the changes
patch -R -p1 < tt_metal/impl/buffers/ADD_BUFFER_DEBUG_LOGGING.patch

# Rebuild
cmake --build build -j$(nproc)
```

---

## Summary

This debug logging gives you:
1. **Exact call stacks** showing where each buffer originates
2. **Allocation/deallocation timeline** to find leaks
3. **Buffer details** (address, size, type) for correlation with server logs
4. **No server/client modifications** needed

Use this to answer:
- "Where was buffer 0x18862c0 allocated?" → Check call stack in log
- "Why are 87 buffers not deallocated?" → Find addresses with ALLOCATED but no DEALLOCATED
- "What operation created this leaked buffer?" → Look at call stack for that address
