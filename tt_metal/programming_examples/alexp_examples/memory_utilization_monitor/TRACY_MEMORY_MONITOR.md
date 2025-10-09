# Tracy Memory Monitor

A real-time memory monitoring system built on Tracy profiling infrastructure that provides:
- ✅ **Real-time queryable memory statistics** per device
- ✅ **Tracy profiler integration** for detailed timeline analysis
- ✅ **Lock-free queries** using atomic counters
- ✅ **Cross-process visibility** when used with Tracy GUI
- ✅ **No separate server required** (embedded in application)

## Architecture

### Key Components

```
┌─────────────────────────────────────────────────────────────┐
│                     Your Application                         │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  Buffer Allocation/Deallocation                       │  │
│  │  (buffer.cpp, graph_tracking.cpp)                     │  │
│  └───────────────┬───────────────────────────────────────┘  │
│                  │                                            │
│                  ▼                                            │
│  ┌───────────────────────────────────────────────────────┐  │
│  │          TracyMemoryMonitor (Singleton)               │  │
│  │  ┌─────────────────┐       ┌─────────────────┐       │  │
│  │  │ Atomic Counters │       │ Active Buffers  │       │  │
│  │  │  (lock-free)    │       │   (mutex)       │       │  │
│  │  └─────────────────┘       └─────────────────┘       │  │
│  └───────┬───────────────────────────┬───────────────────┘  │
│          │                           │                       │
│          │  ┌────────────────────────┘                       │
│          │  │                                                │
└──────────┼──┼────────────────────────────────────────────────┘
           │  │
           │  │  #ifdef TRACY_ENABLE
           │  └────────────────────┐
           │                       ▼
           │              ┌─────────────────────┐
           │              │   Tracy Profiler    │
           │              │  (TracyAllocN/Free) │
           │              └─────────────────────┘
           │                       │
           │                       ▼
           │              ┌─────────────────────┐
           │              │    Tracy GUI        │
           │              │  (memory timeline,  │
           │              │   allocation map)   │
           │              └─────────────────────┘
           │
           ▼  (Real-time query API)
   ┌─────────────────────────┐
   │  Monitor Clients        │
   │  - tracy_memory_        │
   │    monitor_client.cpp   │
   │  - tracy_memory_        │
   │    monitor.py           │
   │  - Your test code       │
   └─────────────────────────┘
```

### Comparison with allocation_server_poc

| Feature | allocation_server_poc | TracyMemoryMonitor |
|---------|----------------------|-------------------|
| **Architecture** | Separate server process + Unix socket | Embedded singleton |
| **IPC Overhead** | Yes (socket communication) | No (direct function calls) |
| **Setup** | Requires running server | Automatic (linked in) |
| **Tracy Integration** | None | Full integration |
| **Query Performance** | ~µs (socket) | ~ns (atomic load) |
| **Cross-Process** | Yes (by design) | Via Tracy GUI only |
| **Memory Timeline** | No | Yes (Tracy GUI) |
| **Allocation Hotspots** | No | Yes (Tracy GUI) |

## Files

### Core Implementation
- `tracy_memory_monitor.hpp` - Main class definition
- `tracy_memory_monitor.cpp` - Implementation
- `graph_tracking.cpp` - Integration point (modified)

### Client Tools
- `tracy_memory_monitor_client.cpp` - C++ real-time monitor
- `tracy_memory_monitor.py` - Python API wrapper

### Documentation
- `TRACY_MEMORY_MONITOR.md` - This file

## Usage

### 1. C++ Direct API

```cpp
#include "tt_metal/impl/profiler/tracy_memory_monitor.hpp"

using namespace tt::tt_metal;

// Query a single device (lock-free)
auto stats = TracyMemoryMonitor::instance().query_device(0);
std::cout << "DRAM: " << stats.dram_allocated.load() << " bytes\n";
std::cout << "L1: " << stats.l1_allocated.load() << " bytes\n";
std::cout << "Active buffers: " << stats.num_buffers.load() << "\n";

// Query all devices
auto all_stats = TracyMemoryMonitor::instance().query_all_devices();
for (int i = 0; i < TracyMemoryMonitor::MAX_DEVICES; i++) {
    if (all_stats[i].get_total_allocated() > 0) {
        std::cout << "Device " << i << " has allocations\n";
    }
}

// Get detailed buffer info (requires lock)
auto buffers = TracyMemoryMonitor::instance().get_active_buffers(0);
for (const auto& buf : buffers) {
    std::cout << "Buffer 0x" << std::hex << buf.buffer_id
              << " size=" << buf.size << "\n";
}
```

### 2. Standalone Monitor Client

Build and run the monitor client:

```bash
# Build (add to your CMakeLists.txt)
# See example_integration.cpp for CMake setup

# Run - Monitor single device
./tracy_memory_monitor_client -d 0

# Run - Monitor multiple devices
./tracy_memory_monitor_client -d 0 -d 1 -d 2

# Run - Monitor all devices
./tracy_memory_monitor_client -a

# Run - Custom refresh rate (500ms)
./tracy_memory_monitor_client -a -r 500

# Run - Single query (no continuous monitoring)
./tracy_memory_monitor_client -s -d 0
```

Example output:
```
═══════════════════════════════════════════════════════════════════════
  Tracy Memory Monitor [Tracy Profiling ENABLED]
═══════════════════════════════════════════════════════════════════════
Time: 14:23:45 | Refresh: 1000ms | Devices: 0, 1, 2

Device 0
───────────────────────────────────────────────────────────────────────
  DRAM:           1.23 GB /      12.00 GB  [█████████░░░░░░░░░░░░░░░░░] 10.3%
  L1:            45.32 MB /      75.00 MB  [████████████████████░░░░░░] 60.4%
  Active Buffers: 42   Total Allocs: 156   Total Frees: 114

Device 1
───────────────────────────────────────────────────────────────────────
  DRAM:           2.45 GB /      12.00 GB  [████████████████░░░░░░░░░░] 20.4%
  L1:            58.67 MB /      75.00 MB  [███████████████████████░░░] 78.2%
  Active Buffers: 38   Total Allocs: 145   Total Frees: 107
```

### 3. Python API (Future)

```python
from tracy_memory_monitor import TracyMemoryMonitor, MemoryMonitorContext

# Simple query
monitor = TracyMemoryMonitor()
stats = monitor.query_device(0)
print(f"DRAM: {stats.dram_allocated} bytes")

# Context manager for measuring memory increase
with MemoryMonitorContext(device_id=0) as ctx:
    run_my_model()

print(f"Memory increase: {ctx.memory_increase} bytes")
print(f"DRAM increase: {ctx.dram_increase} bytes")

# Use in tests
def test_memory_usage():
    monitor = TracyMemoryMonitor()
    before = monitor.query_device(0)

    run_operation()

    after = monitor.query_device(0)
    increase = after.total_allocated - before.total_allocated
    assert increase < MAX_ALLOWED_MEMORY, f"Used too much memory: {increase}"
```

### 4. Integration with Tests

```cpp
TEST_F(MemoryTest, TestAllocationTracking) {
    auto& monitor = TracyMemoryMonitor::instance();

    // Reset statistics
    monitor.reset();

    // Get baseline
    auto before = monitor.query_device(0);

    // Allocate buffers
    auto buffer = Buffer::create(device, 1024*1024, 1024, BufferType::DRAM);

    // Check allocation was tracked
    auto after = monitor.query_device(0);
    EXPECT_EQ(after.num_buffers.load(), before.num_buffers.load() + 1);
    EXPECT_EQ(after.dram_allocated.load(), before.dram_allocated.load() + 1024*1024);

    // Deallocate
    buffer.reset();

    // Check deallocation was tracked
    auto final = monitor.query_device(0);
    EXPECT_EQ(final.num_buffers.load(), before.num_buffers.load());
    EXPECT_EQ(final.dram_allocated.load(), before.dram_allocated.load());
}
```

## Tracy Integration

When compiled with `-DTRACY_ENABLE`:

### 1. Memory Events Sent to Tracy

All allocations/deallocations are automatically sent to Tracy profiler using named memory pools:

```cpp
TracyAllocN(ptr, size, "TT_Dev0_DRAM");   // Device 0 DRAM
TracyAllocN(ptr, size, "TT_Dev0_L1");     // Device 0 L1
TracyAllocN(ptr, size, "TT_Dev1_DRAM");   // Device 1 DRAM
// etc...
```

### 2. Connect Tracy GUI

```bash
# In Terminal 1: Run your application
./your_app

# In Terminal 2: Launch Tracy profiler
tracy

# Tracy GUI will show:
# - Memory timeline per pool
# - Active allocations
# - Allocation/deallocation call stacks
# - Memory map visualization
# - Per-zone memory statistics
```

### 3. View Memory Statistics in Tracy

1. Open **Memory** window in Tracy GUI
2. Select memory pool: `TT_Dev0_DRAM`, `TT_Dev0_L1`, etc.
3. View:
   - Total allocations count
   - Active allocations
   - Memory usage graph
   - Memory span
   - Individual allocations with call stacks

## Performance

### Query Performance

- **Lock-free queries** (`query_device()`): ~10-20ns (atomic load)
- **Detailed queries** (`get_active_buffers()`): ~µs (requires lock)

### Tracking Overhead

- Per allocation: ~100-200ns overhead
  - Atomic increment: ~10ns
  - Map insertion: ~50-100ns
  - Tracy event (if enabled): ~50-100ns

### Memory Overhead

- Per device: ~128 bytes (atomic counters)
- Per active buffer: ~64 bytes (map entry)
- Total for 1000 buffers: ~64KB

## Building

### Add to CMakeLists.txt

```cmake
# Add Tracy memory monitor sources
set(TT_PROFILER_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/tt_metal/impl/profiler/tracy_memory_monitor.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tt_metal/impl/profiler/profiler.cpp
    # ... other profiler sources
)

# Build monitor client tool (optional)
add_executable(tracy_memory_monitor_client
    ${CMAKE_CURRENT_SOURCE_DIR}/tt_metal/programming_examples/alexp_examples/memory_utilization_monitor/tracy_memory_monitor_client.cpp
)

target_link_libraries(tracy_memory_monitor_client
    tt_metal
    profiler
    Tracy::TracyClient  # If using external Tracy
)
```

### Compile Flags

```cmake
# For Tracy integration
if (ENABLE_TRACY)
    add_compile_definitions(TRACY_ENABLE)
endif()
```

## Configuration

### Environment Variables

- `TT_METAL_PROFILER_BUFFER_USAGE_ENABLED=1` - Enable Tracy buffer profiling (if using existing Tracy integration)

### Runtime Options

```cpp
// Reset statistics (useful for tests)
TracyMemoryMonitor::instance().reset();

// Check if Tracy is enabled
if (TracyMemoryMonitor::is_tracy_enabled()) {
    std::cout << "Tracy profiling active\n";
}
```

## Advantages

### vs allocation_server_poc

1. **No separate process** - Embedded in application
2. **Lower latency** - Direct function calls vs socket IPC
3. **Tracy integration** - Memory timeline & visualization
4. **Simpler deployment** - Just link the library
5. **Better performance** - Lock-free queries

### vs Pure Tracy

1. **Real-time queries** - Don't need Tracy GUI running
2. **Programmatic access** - Query from tests/benchmarks
3. **In-process stats** - Immediate feedback
4. **Local state** - Can query without network connection

## Limitations

1. **Single-process** - Each process has its own monitor instance
   - For cross-process, use Tracy GUI or allocation_server_poc
2. **No historical data** - Only current state (unless using Tracy)
3. **Active buffers tracking** - Requires lock (slight overhead)

## Future Enhancements

- [ ] Python bindings via pybind11
- [ ] Histogram of allocation sizes
- [ ] Peak memory tracking
- [ ] Per-thread allocation tracking
- [ ] Memory leak detection utilities
- [ ] Export to Prometheus/metrics systems
- [ ] Integration with system memory profilers

## Troubleshooting

### Monitor shows zero allocations

1. Check that buffers are being allocated through tracked paths
2. Verify `GraphTracker::track_allocate()` is being called
3. Ensure device pointer is not null

### Tracy GUI doesn't show allocations

1. Compile with `-DTRACY_ENABLE`
2. Verify Tracy profiler is connected
3. Check that Tracy server is running

### Build errors

1. Ensure Tracy headers are in include path
2. Link with Tracy library if using external Tracy
3. Check C++17 support (required for `std::atomic`)

## See Also

- `allocation_server_poc.cpp` - Unix socket-based cross-process monitor
- `allocation_monitor_client.cpp` - Client for socket-based monitor
- Tracy manual: `/tt_metal/third_party/tracy/manual/tracy.pdf`
- Buffer implementation: `/tt_metal/impl/buffers/buffer.cpp`
