# TT-Metal Memory Monitoring Integration with tt_telemetry

## Design Document

**Status**: Proposal
**Authors**: Design discussion summary
**Date**: 2025-11-19

---

## 1. Overview

This document describes the integration of TT-Metal memory monitoring capabilities into the `tt_telemetry` system. The goal is to expose per-device memory usage metrics (DRAM, L1, CB, Kernel allocations) alongside hardware telemetry (temperature, power, clocks) via a unified telemetry interface.

---

## 2. Background & Context

### 2.1 Current State

**Existing memory monitoring tools:**
- `allocation_server_poc`: Centralized server tracking all allocations via Unix socket
- `tt_smi_umd`: Interactive monitoring tool (like nvidia-smi)
- `AllocationClient`: Library integrated into TT-Metal allocator for reporting allocations

**Why allocation tracking is required:**
- TT devices have **no hardware register** for memory usage
- UMD firmware telemetry doesn't include memory statistics
- Driver (`/dev/tenstorrent/*`) doesn't expose sysfs memory stats
- Each process allocates independently via hugepages/PCIe BARs
- No central kernel allocator to query

**Therefore**: Userspace allocation tracking is the **only** way to measure memory usage.

### 2.2 Available Metrics

**Metrics requiring allocation tracking:**
- DRAM allocated (bytes, per device)
- L1 allocated (bytes, per device)
- L1_SMALL allocated (bytes, per device)
- TRACE buffer allocated (bytes, per device)
- Circular Buffer (CB) allocated (bytes, per device)
- Kernel code allocated (bytes, per device)
  - Application kernels
  - Fabric kernels
  - Dispatch kernels
- Number of active buffers
- Per-process memory usage

**Metrics available without tracking (direct hardware query):**
- ASIC temperature (°C)
- Board temperature (°C)
- AICLK, AXICLK, ARCCLK frequencies (MHz)
- Fan speed (RPM)
- Power/TDP (Watts)
- Current/TDC (Amps)
- Core voltage (mV)

**Static device information (from UMD SocDescriptor):**
- Total DRAM capacity
- Total L1 capacity
- Architecture type
- DRAM channels count
- L1 size per core
- Grid dimensions

---

## 3. Design Goals

1. **Unified metrics**: Expose memory + hardware telemetry via single `tt_telemetry` interface
2. **Per-process granularity**: Support multiple processes using devices simultaneously
3. **Extensibility**: Architecture should support additional metrics (uptime, kernel cache, perf counters)
4. **Security**: Prevent unauthorized access and socket hijacking
5. **Container-friendly**: Work in containerized deployments
6. **Low overhead**: Minimal impact on application performance
7. **Simple integration**: Minimal changes to existing code

---

## 4. Architecture Decision: Per-Process vs Centralized Server

### 4.1 Option A: Centralized allocation_server (Current)

```
┌─────────────────┐
│ Process A       │──┐
│ (TT-Metal app)  │  │ ALLOC/FREE
└─────────────────┘  │ messages
                     │
┌─────────────────┐  │
│ Process B       │──┤
└─────────────────┘  │
                     ▼
              ┌──────────────────┐
              │ allocation_server │ ← Aggregates totals
              └──────────────────┘
                     │ QUERY
                     ▼
              ┌──────────────────┐
              │  tt_telemetry    │
              └──────────────────┘
```

**Pros:**
- ✅ Single connection for tt_telemetry
- ✅ Automatic cross-process aggregation
- ✅ Dead process cleanup
- ✅ Minimal code in applications

**Cons:**
- ❌ Extra process to manage
- ❌ Not extensible (need new server for each metric type)
- ❌ Doesn't scale to multiple metric types (uptime, kernel cache, perf counters, etc.)
- ❌ If we need additional metrics, we'd need additional servers (uptime_server, kernel_cache_server, etc.)

### 4.2 Option B: Per-Process Metric Endpoints (PROPOSED)

```
┌─────────────────┐
│ Process A       │ ← Exposes: /var/run/tt/metrics_1234.sock
│ (PID 1234)      │    - Memory allocations
│                 │    - Uptime
│ MetricsServer   │    - Active programs
└─────────────────┘    - Kernel cache
                       - Performance counters
┌─────────────────┐
│ Process B       │ ← Exposes: /var/run/tt/metrics_5678.sock
│ (PID 5678)      │
└─────────────────┘

       ↓ Query all processes

┌─────────────────────────┐
│  tt_telemetry           │
│  1. Read KMD PIDs       │ ← /sys/class/tenstorrent/card0/pids
│  2. Connect to sockets  │
│  3. Query all metrics   │
│  4. Aggregate           │
│  5. Export via gRPC     │
└─────────────────────────┘
```

**Pros:**
- ✅ Single endpoint per process for ALL metrics (not just memory)
- ✅ Extensible (add new metrics without new servers)
- ✅ Natural process lifecycle (socket disappears when process exits)
- ✅ Matches observability patterns (Prometheus per-process exporters)
- ✅ KMD provides authoritative PID list

**Cons:**
- ❌ More code in each TT-Metal process (socket server)
- ❌ N socket connections instead of 1
- ❌ tt_telemetry must aggregate across processes

**Decision: Choose Option B (Per-Process Endpoints)** for extensibility and alignment with modern observability practices.

### 4.3 Key Advantage: Multi-Metric Extensibility

**The critical design insight:**

By exposing a per-process metrics socket, we create a **single integration point** for ALL process-specific data that `tt_telemetry` might need—not just memory.

**Today we need:**
- Memory allocations (DRAM, L1, CB, kernels)

**Tomorrow we might need:**
- **Process uptime**: How long has this workload been running?
- **Active programs**: Which kernels are loaded? Cache hit rates?
- **Kernel cache statistics**: L1 ring buffer utilization, eviction counts
- **Performance counters**: Operations executed, bandwidth consumed
- **Resource limits**: Memory quotas, throttling state
- **Application-specific metrics**: Custom counters from user code

**With per-process sockets:**
```cpp
// Adding new metrics is trivial - extend the protocol:
struct MetricsResponse {
    // Memory metrics (existing)
    uint64_t dram_allocated;
    uint64_t l1_allocated;

    // NEW metrics (just add fields)
    uint64_t uptime_seconds;
    uint32_t num_programs_cached;
    uint64_t kernel_cache_bytes;
    uint64_t ops_executed;
    // ... whatever tt_telemetry needs ...
};

// tt_telemetry queries once per process, gets everything
```

**Without per-process sockets (centralized approach):**
- Need `allocation_server` for memory
- Need `uptime_server` for process lifetime tracking
- Need `kernel_cache_server` for L1 ring buffer stats
- Need `performance_server` for counters
- Each server requires: IPC, process discovery, cleanup, integration
- **This doesn't scale**

**Conclusion:** Per-process metrics endpoints are the **correct architecture** for comprehensive telemetry, not just a workaround for memory tracking.

---

## 5. Detailed Design

### 5.1 Process Discovery

**KMD provides PID list per device (added in KMD 2.5.0-rc1, Oct 26):**

```bash
# KMD exposes PIDs via procfs:
/proc/driver/tenstorrent/0/pids → one PID per line
/proc/driver/tenstorrent/1/pids → one PID per line
```

**Example:**
```bash
$ cat /proc/driver/tenstorrent/0/pids
1234
5678
9012

$ cat /proc/driver/tenstorrent/1/pids
5678
9012
```

**Format:**
- One PID per line
- PIDs are processes currently holding an open file descriptor to the device
- Automatically updated as processes open/close devices
- Empty file if no processes using device

**tt_telemetry algorithm:**
```python
for device_id in devices:
    # Read PIDs from KMD
    pids = read_kmd_pids(f"/proc/driver/tenstorrent/{device_id}/pids")

    for pid in pids:
        socket_path = f"/var/run/tt/metrics_{pid}.sock"

        if socket_exists(socket_path):
            metrics = query_process_metrics(socket_path, device_id)
            aggregate(device_id, metrics)
        # If socket missing: process hasn't initialized metrics server yet or crashed
```

**PID discovery implementation:**
```cpp
std::vector<pid_t> read_kmd_pids(int device_id) {
    std::vector<pid_t> pids;

    std::string path = string_format("/proc/driver/tenstorrent/%d/pids", device_id);
    std::ifstream file(path);

    if (!file.is_open()) {
        // KMD doesn't support PID tracking or device doesn't exist
        return pids;
    }

    std::string line;
    while (std::getline(file, line)) {
        if (!line.empty()) {
            pid_t pid = std::stoi(line);
            pids.push_back(pid);
        }
    }

    return pids;
}
```

**Advantages of KMD-based discovery:**
- ✅ Authoritative source (kernel knows which processes have device open)
- ✅ Automatic updates (no manual tracking)
- ✅ Works across all device types
- ✅ No polling `/proc/*/fd` or scanning filesystem
- ✅ PIDs removed automatically when process exits

### 5.2 Communication Protocol

**Unix domain sockets (not TCP):**
- Socket path: `/var/run/tt/metrics_{PID}.sock`
- Protocol: Binary packed struct (extensible)
- Connection: Short-lived (connect, query, disconnect)

**Why Unix sockets over TCP:**
- ✅ No port management/conflicts
- ✅ Filesystem-based discovery
- ✅ Automatic cleanup when process dies
- ✅ Better performance (no network stack)
- ✅ Filesystem permissions for access control
- ✅ Peer credential verification (SO_PEERCRED)

**Message format:**
```cpp
struct MetricsQuery {
    enum Type : uint8_t {
        MEMORY = 1,
        UPTIME = 2,
        KERNELS = 3,
        PERFORMANCE = 4,
        ALL = 255
    };
    Type type;
    int32_t device_id;  // -1 for all devices
};

struct MetricsResponse {
    // Memory metrics
    uint64_t dram_allocated;
    uint64_t l1_allocated;
    uint64_t cb_allocated;
    uint64_t kernel_allocated;

    // Process metrics
    uint64_t uptime_seconds;
    uint32_t num_programs;

    // Performance metrics
    uint64_t ops_executed;
    // ... extensible ...
};
```

### 5.3 TT-Metal Process Implementation

**Each TT-Metal process runs a metrics server:**

```cpp
class ProcessMetricsServer {
public:
    void start() {
        std::string socket_path =
            string_format("/var/run/tt/metrics_%d.sock", getpid());

        // Remove stale socket
        unlink(socket_path.c_str());

        // Create Unix socket
        socket_fd_ = socket(AF_UNIX, SOCK_STREAM, 0);

        struct sockaddr_un addr;
        addr.sun_family = AF_UNIX;
        strncpy(addr.sun_path, socket_path.c_str(), sizeof(addr.sun_path)-1);

        bind(socket_fd_, (struct sockaddr*)&addr, sizeof(addr));

        // Set restrictive permissions (owner-only)
        chmod(socket_path.c_str(), 0600);

        listen(socket_fd_, 5);

        // Start background thread to handle queries
        server_thread_ = std::thread(&ProcessMetricsServer::handle_clients, this);
    }

    void handle_client(int client_fd) {
        // Verify peer credentials
        struct ucred cred;
        socklen_t len = sizeof(cred);
        getsockopt(client_fd, SOL_SOCKET, SO_PEERCRED, &cred, &len);

        // Read query
        MetricsQuery query;
        recv(client_fd, &query, sizeof(query), 0);

        // Build response
        MetricsResponse response = collect_metrics(query);

        // Send response
        send(client_fd, &response, sizeof(response), 0);

        close(client_fd);
    }

private:
    MetricsResponse collect_metrics(const MetricsQuery& query) {
        MetricsResponse resp;

        if (query.type == MetricsQuery::MEMORY || query.type == MetricsQuery::ALL) {
            // Aggregate from allocator
            resp.dram_allocated = get_dram_usage(query.device_id);
            resp.l1_allocated = get_l1_usage(query.device_id);
            // ...
        }

        if (query.type == MetricsQuery::UPTIME || query.type == MetricsQuery::ALL) {
            resp.uptime_seconds = get_process_uptime();
        }

        // ... other metrics ...

        return resp;
    }
};
```

**Integration with allocator:**

```cpp
// In Buffer::allocate_impl():
void track_allocation(int device_id, uint64_t size, BufferType type, uint64_t addr) {
    // Update local counters (thread-safe atomics)
    device_stats_[device_id].dram_allocated += size;

    // Store buffer info for cleanup
    allocations_[addr] = {device_id, size, type};
}

// In Buffer::deallocate():
void track_deallocation(uint64_t addr) {
    auto it = allocations_.find(addr);
    if (it != allocations_.end()) {
        device_stats_[it->device_id].dram_allocated -= it->size;
        allocations_.erase(it);
    }
}
```

### 5.4 tt_telemetry Implementation

```cpp
class TelemetryCollector {
public:
    void collect_device_metrics(int device_id) {
        // 1. Discover processes from KMD
        std::vector<pid_t> pids = read_kmd_pids(device_id);

        // 2. Query each process
        uint64_t total_dram = 0;
        uint64_t total_l1 = 0;

        for (pid_t pid : pids) {
            std::string socket_path =
                string_format("/var/run/tt/metrics_%d.sock", pid);

            if (!socket_exists(socket_path)) {
                continue;  // Process not ready or crashed
            }

            // Verify ownership before connecting
            if (!verify_socket_ownership(socket_path, pid)) {
                log_warning("Socket ownership mismatch for PID %d", pid);
                continue;
            }

            // Connect and query
            auto metrics = query_process_socket(socket_path, device_id);

            // Aggregate
            total_dram += metrics.dram_allocated;
            total_l1 += metrics.l1_allocated;
            // ...
        }

        // 3. Store aggregated metrics
        device_metrics_[device_id].dram_used = total_dram;
        device_metrics_[device_id].l1_used = total_l1;

        // 4. Query hardware telemetry (direct UMD access)
        device_metrics_[device_id].temperature = query_temperature(device_id);
        device_metrics_[device_id].power = query_power(device_id);
        // ...
    }

private:
    bool verify_socket_ownership(const std::string& socket_path, pid_t pid) {
        // Get expected UID from /proc
        struct stat proc_stat;
        std::string proc_path = string_format("/proc/%d", pid);
        if (stat(proc_path.c_str(), &proc_stat) < 0) {
            return false;  // Process doesn't exist
        }

        // Check socket ownership
        struct stat sock_stat;
        if (stat(socket_path.c_str(), &sock_stat) < 0) {
            return false;
        }

        // Verify match (prevents socket hijacking)
        return sock_stat.st_uid == proc_stat.st_uid;
    }
};
```

---

## 6. Security Design

### 6.1 Threat Model

**Threats to mitigate:**
1. **Unauthorized access**: Non-privileged users reading device metrics
2. **Socket hijacking**: Malicious process creating socket with victim's PID
3. **Data poisoning**: Process providing false metrics to tt_telemetry
4. **Information disclosure**: Memory addresses/patterns useful for exploits

**Assumptions:**
- Local machine (not network-exposed)
- Multi-user system with untrusted users
- tt_telemetry runs as privileged service
- TT-Metal processes run as various users

### 6.2 Security Mechanisms

**1. Restrictive socket permissions:**
```cpp
// Socket created with owner-only access
chmod("/var/run/tt/metrics_1234.sock", 0600);  // rw-------

// Only process owner and privileged users can connect
```

**2. Socket ownership verification:**
```cpp
// tt_telemetry verifies socket owner matches /proc/PID owner
// Prevents Process A from creating socket for PID B
if (socket_owner_uid != proc_owner_uid) {
    reject_connection();
}
```

**3. Peer credential verification:**
```cpp
// Server verifies connecting client credentials
struct ucred cred;
getsockopt(client_fd, SOL_SOCKET, SO_PEERCRED, &cred, &len);

// Only allow root or same user
if (cred.uid != 0 && cred.uid != getuid()) {
    close(client_fd);
    return;
}
```

**4. Privileged tt_telemetry:**
```bash
# Grant capability to read all sockets
sudo setcap cap_dac_read_search+ep /usr/bin/tt_telemetry

# Now tt_telemetry can read 0600 sockets from any user
# But users can't read each other's sockets
```

**5. Secure socket directory:**
```bash
# Directory permissions
/var/run/tt/                # drwxr-xr-x root:root (0755)
├── metrics_1234.sock       # srw------- user1:user1 (0600)
├── metrics_5678.sock       # srw------- user2:user2 (0600)

# Directory readable by all (for discovery)
# Sockets readable only by owner + tt_telemetry (via capabilities)
```

### 6.3 Attack Prevention

| Attack | Mitigation |
|--------|-----------|
| User A reads User B's metrics | Socket permissions (0600) prevent access |
| Process A hijacks PID B's socket | Ownership verification rejects mismatched UID |
| Malicious data injection | Peer credentials verify client identity |
| Socket enumeration | Directory is readable (acceptable for discovery) |
| Privilege escalation | tt_telemetry uses minimal capability (CAP_DAC_READ_SEARCH) |

---

## 7. Container Deployment

### 7.1 Socket Path Configuration

**Make socket path configurable:**
```cpp
const char* socket_dir = std::getenv("TT_METRICS_SOCKET_DIR");
if (!socket_dir) {
    socket_dir = "/var/run/tt";  // Default
}
```

### 7.2 Container Architecture

**Option A: Shared volume (recommended)**
```yaml
# docker-compose.yml
version: '3'

services:
  tt_metal_app:
    image: tt_metal:latest
    volumes:
      - tt_metrics:/var/run/tt
    environment:
      - TT_METRICS_SOCKET_DIR=/var/run/tt
    devices:
      - /dev/tenstorrent/0:/dev/tenstorrent/0

  tt_telemetry:
    image: tt_telemetry:latest
    volumes:
      - tt_metrics:/var/run/tt
    environment:
      - TT_METRICS_SOCKET_DIR=/var/run/tt
    cap_add:
      - DAC_READ_SEARCH

volumes:
  tt_metrics:
```

**Option B: Host networking**
```bash
# All containers share host /var/run/tt
docker run --network=host \
           -v /var/run/tt:/var/run/tt \
           tt_metal:latest
```

**Option C: Named volume with host mount**
```bash
# Create directory on host
sudo mkdir -p /var/run/tt

# Mount into containers
docker run -v /var/run/tt:/var/run/tt tt_metal:latest
docker run -v /var/run/tt:/var/run/tt tt_telemetry:latest
```

### 7.3 gRPC over Unix Sockets

**tt_telemetry can expose gRPC via Unix socket:**

```cpp
// Server
grpc::ServerBuilder builder;
builder.AddListeningPort("unix:///var/run/tt/telemetry.sock",
                         grpc::InsecureServerCredentials());

// Client
auto channel = grpc::CreateChannel("unix:///var/run/tt/telemetry.sock",
                                   grpc::InsecureChannelCredentials());
```

**Benefits:**
- No port management in containers
- Can be shared via volume
- Better performance than TCP localhost

---

## 8. Performance Considerations

### 8.1 Overhead Analysis

**Allocation tracking overhead:**
- Socket send: ~1-2 microseconds per allocation/deallocation
- Memory: ~100 bytes per tracked buffer
- Impact: Negligible for typical workloads

**Metrics query overhead:**
- Unix socket roundtrip: ~10 microseconds
- Per-device aggregation: O(1) (atomic counter reads)
- Total for 10 processes: ~100 microseconds
- Polling interval: 1-10 seconds (adjustable)

**Socket buffer capacity:**
- Default: ~212 KB per socket
- Query message: 128 bytes
- Response: 256 bytes
- Capacity: ~500+ queued messages (more than sufficient)

### 8.2 Scalability

**Process scaling:**
- 100 processes: 100 socket connections
- Query time: ~1ms total (parallelizable)
- Acceptable overhead for telemetry polling

**Device scaling:**
- 8 devices × 100 processes = 800 queries
- Can be parallelized per device
- Total time: ~10ms

---

## 9. Implementation Plan

### Phase 1: Per-Process Metrics Server (2-3 weeks)

**Tasks:**
1. ✅ Design metrics protocol (MetricsQuery/MetricsResponse)
2. Implement ProcessMetricsServer class in TT-Metal
3. Integrate with existing allocator tracking
4. Add lifecycle management (start on device open, stop on close)
5. Unit tests for server functionality

**Deliverables:**
- `tt_metal/impl/telemetry/process_metrics_server.hpp`
- `tt_metal/impl/telemetry/process_metrics_server.cpp`
- Integration in device initialization

### Phase 2: tt_telemetry Client (1-2 weeks)

**Tasks:**
1. Implement KMD PID discovery (`/sys/class/tenstorrent/cardN/pids`)
2. Implement Unix socket client
3. Add ownership verification
4. Implement aggregation logic
5. Add hardware telemetry (reuse from tt_smi_umd)
6. Integration tests

**Deliverables:**
- `tt_telemetry/collectors/memory_collector.cpp`
- `tt_telemetry/collectors/hardware_collector.cpp`

### Phase 3: Security & Production Hardening (1 week)

**Tasks:**
1. Add CAP_DAC_READ_SEARCH capability configuration
2. Socket ownership verification
3. Peer credential checking
4. Error handling for stale/missing sockets
5. Security testing

### Phase 4: Container Support (1 week)

**Tasks:**
1. Configurable socket paths via environment
2. Docker compose examples
3. Documentation for container deployment
4. CI/CD integration

### Phase 5: Documentation & Testing (1 week)

**Tasks:**
1. User documentation
2. API documentation
3. Performance benchmarks
4. End-to-end testing
5. Migration guide from allocation_server

---

## 10. Alternatives Considered

### 10.1 Keep Centralized allocation_server

**Why rejected:**
- Not extensible to multiple metric types
- Would need multiple separate servers
- Doesn't scale to diverse telemetry needs

### 10.2 TCP Sockets Instead of Unix

**Why rejected:**
- Port conflict management complexity
- Less secure (network exposure risk)
- Slower than Unix sockets
- No filesystem-based discovery

### 10.3 Shared Memory Instead of Sockets

**Considered:**
```cpp
// Each process writes to: /dev/shm/tt_metrics_<PID>
struct SharedMetrics {
    std::atomic<uint64_t> dram_allocated;
    std::atomic<uint64_t> l1_allocated;
    // ...
};
```

**Pros:**
- Faster than sockets (direct memory access)
- No socket server needed

**Cons:**
- No request/response protocol (polling only)
- Manual memory mapping/unmapping
- Cache coherency concerns
- No extensibility (fixed struct size)
- Harder to version

**Decision:** Sockets provide better flexibility and protocol evolution.

### 10.4 Modify KMD to Track Allocations

**Why rejected:**
- KMD team conservative with changes
- Requires kernel module modifications
- Deployment complexity (kernel version dependencies)
- Limited to DRAM (can't track L1 easily)
- Userspace solution more maintainable

---

## 11. Open Questions

1. **Metric refresh rate**: What polling interval is acceptable? (Default: 5 seconds)
2. **Historical data**: Should tt_telemetry maintain time-series data or just current state?
3. **Protocol versioning**: How to handle protocol evolution when clients/servers upgrade?
4. **Error handling**: How long to retry stale sockets before giving up?
5. **Backward compatibility**: Keep allocation_server for legacy deployments?
6. **Cross-process kernel cache**: Should kernel L1 usage be attributed to specific processes?

---

## 12. Success Criteria

1. ✅ tt_telemetry can query memory usage from multiple processes
2. ✅ Latency < 100ms for full system query (all devices, all processes)
3. ✅ Works in containerized deployments
4. ✅ Secure against unauthorized access and socket hijacking
5. ✅ Extensible to additional metrics without protocol changes
6. ✅ Minimal overhead on application performance (<1% CPU)
7. ✅ Zero downtime when processes start/stop

---

## 13. References

- Current implementation: `tt_metal/programming_examples/alexp_examples/memory_utilization_monitor/`
- UMD telemetry: `tt_smi_umd.cpp:714-781`
- Allocation tracking: `tt_metal/impl/allocator/allocation_client.hpp`
- KMD PID tracking: `/proc/driver/tenstorrent/{N}/pids` (added in KMD 2.5.0-rc1, Oct 26, 2024)
  - Format: One PID per line
  - Automatically updated as processes open/close devices

---

## 14. Appendix: Socket Path Conventions

**Production recommendation:**
- Base directory: `/var/run/tt/` (systemd standard for runtime data)
- Socket naming: `metrics_{PID}.sock`
- Full path: `/var/run/tt/metrics_1234.sock`

**Development override:**
```bash
export TT_METRICS_SOCKET_DIR=/tmp/tt_dev
# Sockets: /tmp/tt_dev/metrics_1234.sock
```

**Container override:**
```yaml
environment:
  - TT_METRICS_SOCKET_DIR=/shared/metrics
volumes:
  - metrics_vol:/shared/metrics
```
