# Adding Buffer Origin Tracking - Implementation Plan

## Goal
Track the exact origin (call stack and name) of each buffer allocation to identify what the remaining 381 buffers are.

## Approach

We have two options:

### Option 1: Python-Level Tracking (Easiest)
Intercept buffer allocations at the Python/ttnn level and capture stack traces there.

**Pros:**
- Full Python stack trace available
- Easy to add descriptive names
- No C++ compilation needed

**Cons:**
- Only tracks buffers allocated from Python
- Misses C++-only allocations

### Option 2: C++-Level Tracking (Most Complete)
Add stack trace capture to the C++ allocation tracking in TracyMemoryMonitor.

**Pros:**
- Tracks ALL allocations
- Complete coverage

**Cons:**
- Need C++ backtrace library
- More complex to implement
- Requires recompilation

## Recommended: Hybrid Approach

1. **Python wrapper around ttnn buffer creation**
2. **Capture Python stack trace**
3. **Send to allocation server with buffer name**

## Implementation Steps

### Step 1: Create Python Buffer Tracker Wrapper

```python
# tt_metal/programming_examples/alexp_examples/memory_utilization_monitor/buffer_tracker_wrapper.py

import traceback
import ttnn
from functools import wraps

class BufferOriginTracker:
    """Wrapper to track buffer origins with stack traces"""

    def __init__(self):
        self.enabled = False
        try:
            from allocation_client import AllocationClient, BufferType
            self.client = AllocationClient()
            self.enabled = True
        except:
            pass

    def track_ttnn_buffer(self, buffer, name="", context=""):
        """Track a ttnn buffer with origin info"""
        if not self.enabled:
            return

        # Get stack trace (skip wrapper frames)
        stack = traceback.extract_stack()[:-2]
        trace_str = " -> ".join([
            f"{frame.filename.split('/')[-1]}:{frame.lineno}:{frame.name}"
            for frame in stack[-5:]  # Last 5 frames
        ])

        # Add context if provided
        full_name = f"{context}:{name}" if context else name

        # Report to server
        try:
            device_id = buffer.device().id() if hasattr(buffer, 'device') else 0
            size = buffer.volume() * buffer.element_size()
            buffer_type = self._map_buffer_type(buffer)
            buffer_id = id(buffer)

            # Send with name and stack trace
            # Note: Need to extend AllocationClient.allocate() to accept these
            self.client.allocate_with_context(
                device_id, size, buffer_type, buffer_id,
                buffer_name=full_name,
                stack_trace=trace_str
            )
        except Exception as e:
            print(f"Warning: Failed to track buffer: {e}")

    def _map_buffer_type(self, buffer):
        # Map ttnn buffer type to our enum
        if hasattr(buffer, 'buffer_type'):
            bt = str(buffer.buffer_type())
            if 'DRAM' in bt: return 0
            if 'L1' in bt: return 1
            if 'TRACE' in bt: return 4
        return 0  # Default DRAM

# Global tracker
_tracker = BufferOriginTracker()

def track_buffer(buffer, name="", context=""):
    """Helper function to track a buffer"""
    _tracker.track_ttnn_buffer(buffer, name, context)
```

### Step 2: Use in Model Code

```python
# In generator.py or model code:
from buffer_tracker_wrapper import track_buffer

# When creating buffers:
weights_buffer = ttnn.to_device(weights, device)
track_buffer(weights_buffer, name="layer_0_weights", context="model_init")

kv_cache = ttnn.allocate_kv_cache(...)
track_buffer(kv_cache, name="kv_cache_layer_0", context="cache_init")
```

### Step 3: Simpler Alternative - Use Existing Tracy Integration

Actually, **we already have tracking via TracyMemoryMonitor**! We just need to:
1. Enable verbose output
2. Dump with more details

Let me check what TracyMemoryMonitor already captures...

## Easiest Solution: Enhanced Server Dump

Instead of adding complex tracking, let's **enhance the dump output** to show:
1. Buffer sizes (individual, not just totals)
2. Allocation timestamps
3. Group by size to identify patterns

This will help identify:
- "Many small buffers" → Likely activations
- "Few large buffers" → Likely weights
- "Consistent sizes" → Likely cache entries

### Implementation: Enhanced Dump

```cpp
// In allocation_server_poc.cpp, modify handle_dump_remaining():

void handle_dump_remaining() {
    // ... existing code ...

    // NEW: Show individual buffers grouped by size
    std::cout << "\n📊 Buffer Size Analysis:" << std::endl;

    std::map<uint64_t, int> size_histogram;
    for (const auto& [key, info] : allocations_) {
        // Round to nearest KB for grouping
        uint64_t size_kb = (info.size + 512) / 1024;
        size_histogram[size_kb]++;
    }

    std::cout << "\nMost Common Buffer Sizes:" << std::endl;
    std::vector<std::pair<uint64_t, int>> sorted_sizes(
        size_histogram.begin(), size_histogram.end()
    );
    std::sort(sorted_sizes.begin(), sorted_sizes.end(),
        [](const auto& a, const auto& b) { return a.second > b.second; }
    );

    for (int i = 0; i < std::min(10, (int)sorted_sizes.size()); i++) {
        std::cout << "  " << sorted_sizes[i].first << " KB: "
                  << sorted_sizes[i].second << " buffers" << std::endl;
    }
}
```

This will show patterns like:
```
Most Common Buffer Sizes:
  32 KB: 45 buffers    ← Likely page table entries
  256 KB: 120 buffers  ← Likely activations
  2048 KB: 8 buffers   ← Likely layer weights
```

## What to Do Now

**Quick Win - Add Size Analysis:**
I can add the size histogram to the dump output right now. This will help identify what the buffers are without needing stack traces.

**Full Solution - Stack Traces:**
Requires modifying the message format again and ensuring all clients support it properly.

Which would you prefer?
