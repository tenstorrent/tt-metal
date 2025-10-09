# Tracing Buffer Origins in TT-Metal

## Your Situation

You have **87 leaked buffers** shown in your server debug.log. You want to know:
1. **Where in TT-Metal code** these buffers originate
2. **Why** they weren't deallocated
3. **How to trace** them without touching the server/client

## Analysis of Your debug.log

From `correlate_leaked_buffers.py` analysis:

### Device 0: 115 Leaked Buffers

**Top Offenders:**

1. **Buffer 0x18b20** (101,152)
   - 176 allocations, only 1 deallocation
   - **Net leak: 175 buffers**
   - Type: L1
   - Sizes: 32KB, 4KB, 12KB, 131KB, 8KB (various)
   - Heavy reuse of same address

2. **Buffer 0x18815e0** (25,695,712)
   - 188 allocations, 71 deallocations
   - **Net leak: 117 buffers**
   - Type: DRAM
   - Sizes: 512B, 64KB, 16KB, 8KB, 34KB (various)
   - Address reused with different sizes

3. **Buffer 0x1885800** (25,712,640)
   - 124 allocations, 41 deallocations
   - **Net leak: 83 buffers**
   - Type: DRAM
   - Sizes: 16KB, 8KB, 512KB, 256KB, 64KB (various)

### Pattern Observed

**All leaked buffers show address reuse:**
- Same address allocated multiple times
- Different sizes each time
- Some deallocations happen, but not all
- This suggests **buffer caching or pooling** in TT-Metal's allocator

---

## Solution: Add Debug Logging to TT-Metal

### Step 1: Apply the Debug Logging Patch

The patch adds call stack capture to buffer allocations/deallocations.

```bash
cd /workspace/tt-metal-apv

# Apply the patch
patch -p1 < tt_metal/impl/buffers/ADD_BUFFER_DEBUG_LOGGING.patch

# Rebuild TT-Metal (this will take a while)
cmake --build build -j$(nproc)
```

### Step 2: Run Your Test with Debug Logging

```bash
# Enable buffer debug logging
export TT_BUFFER_DEBUG_LOG=1

# Run your test
python your_test_script.py

# Debug output goes to /tmp/tt_buffer_debug.log
```

### Step 3: Correlate Leaked Buffers with Call Stacks

Now you can trace the exact origin of each leaked buffer:

```bash
# Find where buffer 0x18b20 was allocated
grep -A 15 "Address: 0x18b20" /tmp/tt_buffer_debug.log | grep -A 15 "BUFFER ALLOCATED"
```

Output will show:
```
═══════════════════════════════════════════════════════════
BUFFER ALLOCATED
Time: 14:32:15
Device: 0
Address: 0x18b20 (101152)
Size: 32768 bytes (32 KB)
Type: L1
Buffer*: 0x7f8a4c001230
Owns Data: yes
Hooked: no
Call Stack:
    libtensorrent.so: tt::tt_metal::Buffer::allocate_impl()
    libtensorrent.so: tt::tt_metal::Buffer::create(IDevice*, ...)
    libtensorrent.so: ttnn::operations::matmul::MatmulOperation::create_sharded_tensor(...)
    libtensorrent.so: ttnn::operations::matmul::MatmulOperation::operator()(...)
    _ttnn.so: pybind11::cpp_function::dispatcher()
    python3: PyEval_EvalFrameDefault
    ...
```

**This tells you:**
- Buffer came from `ttnn::operations::matmul`
- Specifically from `create_sharded_tensor`
- Called via Python's `ttnn.matmul()`

---

## What the Call Stacks Will Reveal

### Example 1: Matmul Temporary Buffers

```
Call Stack:
    Buffer::allocate_impl()
    ttnn::operations::matmul::create_output_tensor()
    ttnn::operations::matmul::MatmulOperation::operator()()
```

**Interpretation:** Buffer created for matmul output, likely held in a cache or not freed properly.

### Example 2: Circular Buffers

```
Call Stack:
    CircularBuffer::allocate()
    Program::allocate_circular_buffers()
    EnqueueProgram()
```

**Interpretation:** Circular buffer allocated for kernel, might be cached in program.

### Example 3: Intermediate Tensors

```
Call Stack:
    Buffer::create()
    ttnn::operations::reshape::ReshapeOperation::operator()()
    ttnn::operations::core::to_device()
```

**Interpretation:** Intermediate buffer from reshape operation.

---

## Common Reasons for Leaked Buffers

Based on the leaked buffer patterns:

### 1. **Tensor Caching**
TT-Metal caches tensors/buffers for performance. If cache isn't cleared:
```python
# Your code probably does:
result = ttnn.matmul(a, b)  # Creates buffers, caches them
# ... more operations ...
# Process exits WITHOUT clearing cache → leak!

# Fix: Explicitly clear caches
ttnn.deallocate(result)
# Or force cleanup:
import gc
gc.collect()
```

### 2. **Program/Kernel Buffer Caching**
Programs cache circular buffers and kernel binaries:
```cpp
// In TT-Metal:
Program* program = CreateKernel(...);  // Allocates buffers
// Buffers stay alive until program destroyed
// If program cached → buffers leak
```

### 3. **Exception Before Cleanup**
```python
try:
    tensor = ttnn.matmul(a, b)  # Allocates
    # Exception here!
    do_something()
except:
    # tensor never deallocated
    pass
```

### 4. **Reference Cycles**
```python
# Python objects holding C++ shared_ptr
class Model:
    def __init__(self):
        self.weights = ttnn.from_torch(...)  # Holds Buffer
        self.cache = {}
        self.cache['self'] = self  # Circular reference!
```

---

## Finding the Leak Source

### Method 1: Count Allocations by Call Stack

```bash
# Extract call stacks for all allocations
grep -A 12 "BUFFER ALLOCATED" /tmp/tt_buffer_debug.log > /tmp/alloc_stacks.txt

# Find most common allocation sites
grep "libtensorrent.so:" /tmp/alloc_stacks.txt | sort | uniq -c | sort -rn | head -20
```

This shows which functions allocate the most buffers.

### Method 2: Track Specific Leaked Buffer

```bash
# For buffer 0x18b20 which leaked 175 times:
echo "=== All events for buffer 0x18b20 ===" > /tmp/buffer_18b20.log
grep -B 2 -A 15 "Address: 0x18b20" /tmp/tt_buffer_debug.log >> /tmp/buffer_18b20.log

# Count allocations vs deallocations
echo "Allocations:"
grep -c "BUFFER ALLOCATED" /tmp/buffer_18b20.log
echo "Deallocations:"
grep -c "BUFFER DEALLOCATED" /tmp/buffer_18b20.log
```

### Method 3: Find Missing Deallocations

```python
# Python script to match alloc/dealloc
import re

with open('/tmp/tt_buffer_debug.log') as f:
    log = f.read()

events = log.split('═══════════════════════')

buffer_ptrs = {}  # Track Buffer* objects

for event in events:
    if 'BUFFER ALLOCATED' in event:
        addr_match = re.search(r'Address: (0x[0-9a-f]+)', event)
        ptr_match = re.search(r'Buffer\*: (0x[0-9a-f]+)', event)
        if addr_match and ptr_match:
            buffer_ptrs[ptr_match.group(1)] = {
                'addr': addr_match.group(1),
                'alloc_stack': event,
                'deallocated': False
            }
    elif 'BUFFER DEALLOCATED' in event:
        ptr_match = re.search(r'Buffer\*: (0x[0-9a-f]+)', event)
        if ptr_match and ptr_match.group(1) in buffer_ptrs:
            buffer_ptrs[ptr_match.group(1)]['deallocated'] = True

# Find leaked Buffer objects
leaked = {ptr: info for ptr, info in buffer_ptrs.items() if not info['deallocated']}

print(f"Leaked Buffer objects: {len(leaked)}")
for ptr, info in list(leaked.items())[:10]:
    print(f"\nBuffer* {ptr} at address {info['addr']}:")
    stack_lines = info['alloc_stack'].split('\n')
    for line in stack_lines:
        if 'libtensorrent.so:' in line or '_ttnn.so:' in line:
            print(f"  {line.strip()}")
```

---

## Expected Findings

After applying the patch and analyzing, you'll likely find:

### Leak Source 1: Cached Program Buffers
```
Call Stack shows:
    Program::allocate_circular_buffers()
    → These buffers cached in program
    → Program never destroyed
    → Fix: Clear program cache
```

### Leak Source 2: Intermediate Operation Buffers
```
Call Stack shows:
    ttnn::reshape() / ttnn::transpose() / etc
    → Temporary buffers created
    → Not freed when operation completes
    → Fix: Enable auto-cleanup or explicit dealloc
```

### Leak Source 3: MeshDevice Buffers
```
Call Stack shows:
    MeshBuffer::create_device_buffers()
    → Device-local buffers
    → MeshDevice destroyed but buffers remain
    → Fix: Already fixed in your codebase patches
```

---

## Next Steps

1. **Apply the patch** (15 minutes)
   ```bash
   cd /workspace/tt-metal-apv
   patch -p1 < tt_metal/impl/buffers/ADD_BUFFER_DEBUG_LOGGING.patch
   cmake --build build -j$(nproc)
   ```

2. **Run with debug logging** (your test runtime)
   ```bash
   export TT_BUFFER_DEBUG_LOG=1
   python your_test.py
   ```

3. **Analyze the results** (10 minutes)
   ```bash
   # See where leaked buffers come from
   python3 tt_metal/programming_examples/alexp_examples/memory_utilization_monitor/correlate_leaked_buffers.py /tmp/tt_buffer_debug.log
   ```

4. **Find the fix** based on call stacks:
   - If from `matmul`: Check tensor lifecycle in your Python code
   - If from `Program`: Clear program cache between iterations
   - If from `MeshDevice`: Your patches should already fix this
   - If from operations: Add explicit cleanup

---

## Files Created

1. **ADD_BUFFER_DEBUG_LOGGING.patch** - Adds call stack tracing to Buffer allocations
2. **ENABLE_BUFFER_DEBUG_LOGGING.md** - Full instructions for using the patch
3. **correlate_leaked_buffers.py** - Analyzes your debug.log to find leaked buffers
4. **TRACING_BUFFER_ORIGINS_SUMMARY.md** - This file

## Key Insight from Your Log

**Buffer address 0x18b20 leaked 175 times with various sizes:**
- 32KB, 4KB, 12KB, 131KB, 8KB allocations
- All L1 buffers
- Address heavily reused (allocated 176 times, freed only once!)

**This suggests:** L1 buffer pooling/caching where buffers are allocated from a pool but never returned to it properly.

The debug logging will show **exactly which operations** are using this pool and not returning buffers.
