# Memory Monitoring Architecture - Visual Guide

## Complete System Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      TT Device Memory Monitoring System                     │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                        OPTION 1: In-Process Monitoring                      │
└─────────────────────────────────────────────────────────────────────────────┘

                    ┌──────────────────────────────┐
                    │      Your Process            │
                    │                              │
                    │  ┌─────────────────────┐    │
                    │  │  Your Application   │    │
                    │  │  Code               │    │
                    │  └──────────┬──────────┘    │
                    │             │                │
                    │             │ CreateBuffer() │
                    │             │                │
                    │  ┌──────────▼──────────┐    │
                    │  │  TT-Metal           │    │
                    │  │  IDevice            │    │
                    │  │  └─ Allocator       │    │
                    │  └──────────┬──────────┘    │
                    │             │                │
                    │             │ get_statistics()
                    │             │                │
                    │  ┌──────────▼──────────┐    │
                    │  │  Memory Monitor     │    │
                    │  │  (this process)     │    │
                    │  │                     │    │
                    │  │  Shows: ✓ Own       │    │
                    │  │         ✗ Others    │    │
                    │  └─────────────────────┘    │
                    └──────────────────────────────┘

    ✓ Simple setup                    ✗ Can't see other processes
    ✓ Direct API access               ✗ Limited to single application
    ✓ Low latency


┌─────────────────────────────────────────────────────────────────────────────┐
│                  OPTION 2: Cross-Process Monitoring (Server)                │
└─────────────────────────────────────────────────────────────────────────────┘

┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│  Process 1   │  │  Process 2   │  │  Process 3   │  │  Monitor     │
│  (Python)    │  │  (C++)       │  │  (Python)    │  │  (Any)       │
│              │  │              │  │              │  │              │
│ ┌──────────┐ │  │ ┌──────────┐ │  │ ┌──────────┐ │  │              │
│ │  ttnn    │ │  │ │Your App  │ │  │ │ ttnn     │ │  │              │
│ │  model   │ │  │ │          │ │  │ │ model    │ │  │              │
│ └────┬─────┘ │  │ └────┬─────┘ │  │ └────┬─────┘ │  │              │
│      │       │  │      │       │  │      │       │  │              │
│      │ Alloc │  │      │ Alloc │  │      │ Alloc │  │              │
│      ▼       │  │      ▼       │  │      ▼       │  │              │
│ ┌──────────┐ │  │ ┌──────────┐ │  │ ┌──────────┐ │  │              │
│ │ Client   │ │  │ │ Client   │ │  │ │ Client   │ │  │              │
│ │ Library  │ │  │ │ Library  │ │  │ │ Library  │ │  │              │
│ └────┬─────┘ │  │ └────┬─────┘ │  │ └────┬─────┘ │  │              │
└──────┼───────┘  └──────┼───────┘  └──────┼───────┘  └──────┬───────┘
       │                 │                 │                 │
       │ Unix Socket     │ Unix Socket     │ Unix Socket     │ Unix Socket
       │ Report          │ Report          │ Report          │ Query
       │                 │                 │                 │
       └─────────────────┴─────────────────┴─────────────────┘
                                 │
                                 ▼
                    ┌──────────────────────────┐
                    │  Allocation Server       │
                    │  (daemon process)        │
                    │                          │
                    │  ┌────────────────────┐  │
                    │  │ Allocation         │  │
                    │  │ Registry           │  │
                    │  │                    │  │
                    │  │ Device 0:          │  │
                    │  │   DRAM: 200MB      │  │
                    │  │   L1: 12MB         │  │
                    │  │   Buffers:         │  │
                    │  │   - #1: 100MB (P1) │  │
                    │  │   - #2: 50MB (P2)  │  │
                    │  │   - #3: 50MB (P1)  │  │
                    │  │   ...              │  │
                    │  │                    │  │
                    │  │ Device 1:          │  │
                    │  │   DRAM: 150MB      │  │
                    │  │   ...              │  │
                    │  └────────────────────┘  │
                    └──────────────────────────┘

    ✓ Cross-process visibility         ✓ Per-process attribution
    ✓ System-wide view                 ✓ Historical tracking
    ✓ Multiple monitors                ⚠ Requires daemon
    ✓ Production-ready                 ⚠ Requires instrumentation


┌─────────────────────────────────────────────────────────────────────────────┐
│                      Message Flow: Allocation Example                       │
└─────────────────────────────────────────────────────────────────────────────┘

Time: T0
┌──────────────┐                     ┌──────────────┐
│  Process A   │                     │  Server      │
│  (Python)    │                     │              │
│              │                     │  Allocations:│
│ buffer =     │                     │  (empty)     │
│  CreateBuf() │                     │              │
└──────────────┘                     └──────────────┘

Time: T1
┌──────────────┐                     ┌──────────────┐
│  Process A   │  ──ALLOC_MSG───>   │  Server      │
│              │  (100MB, DRAM)     │              │
│ buffer_id=1  │                     │  Processing  │
└──────────────┘                     └──────────────┘

Time: T2
┌──────────────┐                     ┌──────────────┐
│  Process A   │                     │  Server      │
│              │                     │              │
│ buffer_id=1  │                     │  Allocations:│
│ (using...)   │                     │  #1: 100MB   │
│              │                     │      (PID=A) │
└──────────────┘                     └──────────────┘

Time: T3
┌──────────────┐                     ┌──────────────┐                     ┌──────────────┐
│  Process A   │                     │  Server      │                     │  Monitor     │
│              │                     │              │  <──QUERY_MSG───    │              │
│ buffer_id=1  │                     │  Allocations:│                     │  Querying... │
│              │                     │  #1: 100MB   │  ──RESPONSE_MSG─>   │              │
└──────────────┘                     └──────────────┘                     └──────────────┘

Time: T4
┌──────────────┐                     ┌──────────────┐                     ┌──────────────┐
│  Process A   │                     │  Server      │                     │  Monitor     │
│              │                     │              │                     │              │
│ buffer_id=1  │                     │  Allocations:│                     │  Display:    │
│              │                     │  #1: 100MB   │                     │  DRAM: 100MB │
└──────────────┘                     └──────────────┘                     └──────────────┘

Time: T5
┌──────────────┐                     ┌──────────────┐
│  Process A   │  ──FREE_MSG────>   │  Server      │
│              │  (buffer_id=1)     │              │
│ del buffer   │                     │  Processing  │
└──────────────┘                     └──────────────┘

Time: T6
┌──────────────┐                     ┌──────────────┐                     ┌──────────────┐
│  Process A   │                     │  Server      │                     │  Monitor     │
│              │                     │              │                     │              │
│              │                     │  Allocations:│                     │  Display:    │
│              │                     │  (empty)     │                     │  DRAM: 0MB   │
└──────────────┘                     └──────────────┘                     └──────────────┘


┌─────────────────────────────────────────────────────────────────────────────┐
│                      Message Format (IPC Protocol)                          │
└─────────────────────────────────────────────────────────────────────────────┘

struct AllocMessage {
    ┌─────────────────────┐
    │ type (1 byte)       │  1 = ALLOC, 2 = FREE, 3 = QUERY, 4 = RESPONSE
    ├─────────────────────┤
    │ padding (3 bytes)   │  Alignment
    ├─────────────────────┤
    │ device_id (4 bytes) │  0-7 for device ID
    ├─────────────────────┤
    │ size (8 bytes)      │  Allocation size in bytes
    ├─────────────────────┤
    │ buffer_type (1 byte)│  0=DRAM, 1=L1, 2=L1_SMALL, 3=TRACE
    ├─────────────────────┤
    │ padding (3 bytes)   │  Alignment
    ├─────────────────────┤
    │ process_id (4 bytes)│  PID of requesting process
    ├─────────────────────┤
    │ buffer_id (8 bytes) │  Unique buffer identifier
    ├─────────────────────┤
    │ timestamp (8 bytes) │  Nanoseconds since epoch
    ├─────────────────────┤
    │ Response fields:    │  Only used in RESPONSE messages
    │  - dram_allocated   │  8 bytes
    │  - l1_allocated     │  8 bytes
    │  - l1_small_alloc   │  8 bytes
    │  - trace_allocated  │  8 bytes
    └─────────────────────┘
    Total: 72 bytes
}


┌─────────────────────────────────────────────────────────────────────────────┐
│                   Data Structures: Server Side                              │
└─────────────────────────────────────────────────────────────────────────────┘

class AllocationServer {

    // Registry of all active allocations
    std::unordered_map<buffer_id, BufferInfo> allocations_;

    struct BufferInfo {
        uint64_t buffer_id;       // Unique identifier
        int device_id;            // Which device (0-7)
        uint64_t size;            // Size in bytes
        BufferType type;          // DRAM, L1, etc.
        pid_t owner_pid;          // Which process owns it
        timestamp alloc_time;     // When allocated
    };

    // Per-device statistics (lock-free reads!)
    struct DeviceStats {
        atomic<uint64_t> dram_allocated;
        atomic<uint64_t> l1_allocated;
        atomic<uint64_t> l1_small_allocated;
        atomic<uint64_t> trace_allocated;
        atomic<uint64_t> num_buffers;
    } device_stats_[8];  // One per device

    // Thread model
    - Main thread: accept() new connections
    - Worker threads: One per client connection (detached)
    - No locks needed for statistics reads (atomic!)
    - Mutex only for allocations_ map updates
};


┌─────────────────────────────────────────────────────────────────────────────┐
│                        Deployment Topology                                  │
└─────────────────────────────────────────────────────────────────────────────┘

Single Machine Deployment:
┌─────────────────────────────────────────────────────────────────────────────┐
│                            Machine: ml-server-01                            │
│                                                                             │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐         │
│  │ Training Process │  │ Eval Process     │  │ Inference Server │         │
│  │ (GPU 0)          │  │ (GPU 1)          │  │ (GPU 0,1)        │         │
│  └────────┬─────────┘  └────────┬─────────┘  └────────┬─────────┘         │
│           │                     │                     │                    │
│           └─────────────────────┴─────────────────────┘                    │
│                                 │                                          │
│                    ┌────────────▼────────────┐                             │
│                    │  Allocation Server      │                             │
│                    │  (systemd service)      │                             │
│                    └────────────┬────────────┘                             │
│                                 │                                          │
│                    ┌────────────▼────────────┐                             │
│                    │  Monitor Dashboard      │                             │
│                    │  (Web UI on :8080)      │                             │
│                    └─────────────────────────┘                             │
└─────────────────────────────────────────────────────────────────────────────┘


Multi-Machine Deployment (Future):
┌──────────────────┐       ┌──────────────────┐       ┌──────────────────┐
│  ml-server-01    │       │  ml-server-02    │       │  ml-server-03    │
│                  │       │                  │       │                  │
│  ┌────────────┐  │       │  ┌────────────┐  │       │  ┌────────────┐  │
│  │ Alloc      │  │       │  │ Alloc      │  │       │  │ Alloc      │  │
│  │ Server     │  │       │  │ Server     │  │       │  │ Server     │  │
│  └─────┬──────┘  │       │  └─────┬──────┘  │       │  └─────┬──────┘  │
└────────┼─────────┘       └────────┼─────────┘       └────────┼─────────┘
         │                          │                          │
         │ TCP/gRPC                 │ TCP/gRPC                 │ TCP/gRPC
         └──────────────────────────┴──────────────────────────┘
                                    │
                          ┌─────────▼─────────┐
                          │  Central          │
                          │  Aggregator       │
                          │                   │
                          │  Global Dashboard │
                          └───────────────────┘


┌─────────────────────────────────────────────────────────────────────────────┐
│                    Performance Characteristics                              │
└─────────────────────────────────────────────────────────────────────────────┘

Latency:
  Allocation report:  ────────────────[███]  ~50-100 μs
  Query response:     ─────[█]                ~10-20 μs
  Monitor refresh:    ─────────────────────[████████]  ~1-1000 ms (configurable)

Throughput:
  Server capacity:    50,000 operations/second
  Network bandwidth:  72 bytes/message × 50K/s = ~3.6 MB/s

Scalability:
  Concurrent clients:   128 (default), 1000s possible
  Tracked buffers:      Millions (RAM limited)
  Devices supported:    8 (configurable to 100s)

Memory Overhead:
  Per allocation:       ~100 bytes (registry entry)
  Server base:          ~1 MB
  Client library:       ~50 KB
  Total for 10K allocs: ~10 MB


┌─────────────────────────────────────────────────────────────────────────────┐
│                          Feature Roadmap                                    │
└─────────────────────────────────────────────────────────────────────────────┘

✅ Phase 1: Basic Tracking (DONE)
   - Unix socket IPC
   - Basic allocation/deallocation
   - Real-time queries
   - C++ POC
   - Python client

🚧 Phase 2: Production Features (IN PROGRESS)
   - TT-Metal integration
   - Auto-instrumentation (LD_PRELOAD)
   - Persistent storage (SQLite)
   - Web dashboard

📋 Phase 3: Advanced Features (FUTURE)
   - Distributed tracking (TCP/gRPC)
   - Machine learning (OOM prediction)
   - Integration with observability platforms
   - Auto-scaling triggers
   - Security (authentication, encryption)

🔮 Phase 4: Enterprise (VISION)
   - Multi-tenant support
   - Role-based access control
   - Audit logging
   - SLA monitoring
   - Cost attribution


┌─────────────────────────────────────────────────────────────────────────────┐
│                              Summary                                        │
└─────────────────────────────────────────────────────────────────────────────┘

Two complementary approaches:

1. IN-PROCESS MONITORING
   Best for: Development, single-process apps, quick checks
   Tools: memory_monitor.cpp, memory_monitor_with_test.cpp

2. CROSS-PROCESS TRACKING (ALLOCATION SERVER)
   Best for: Production, multi-process, system-wide visibility
   Tools: allocation_server_poc, allocation_client_demo, allocation_monitor_client

Choose based on your needs, or use both! 🚀
