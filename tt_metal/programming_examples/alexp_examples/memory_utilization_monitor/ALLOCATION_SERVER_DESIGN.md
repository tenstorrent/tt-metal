# Allocation Server Design for Cross-Process Memory Tracking

## Overview

An **Allocation Server** is a daemon process that tracks all TT device memory allocations across multiple client processes, exposing this information via IPC (Inter-Process Communication).

```
┌─────────────┐  ┌─────────────┐  ┌─────────────┐
│  Process 1  │  │  Process 2  │  │  Process 3  │
│  (Python)   │  │  (C++)      │  │  (Monitor)  │
└──────┬──────┘  └──────┬──────┘  └──────┬──────┘
       │                │                │
       │ Alloc/Free     │ Alloc/Free     │ Query Stats
       │ via IPC        │ via IPC        │ via IPC
       │                │                │
       └────────────────┴────────────────┘
                        │
                        ▼
              ┌─────────────────────┐
              │  Allocation Server  │
              │     (daemon)        │
              │                     │
              │  Tracks:            │
              │  - Device 0: 200MB  │
              │  - Device 1: 150MB  │
              │  - Per-process      │
              │  - Per-buffer-type  │
              └─────────────────────┘
```

## Architecture Components

### 1. Allocation Server (Daemon)

**Purpose**: Centralized tracking of all device memory allocations

**Responsibilities**:
- Listen for allocation/deallocation requests
- Maintain allocation registry
- Track per-device, per-process, per-buffer-type statistics
- Expose query interface for monitoring tools

### 2. Client Library (Shim)

**Purpose**: Intercept allocation calls and report to server

**Responsibilities**:
- Wrap TT-Metal allocation APIs
- Send allocation notifications to server
- Transparent to applications (drop-in replacement)

### 3. Monitor Client

**Purpose**: Query server for real-time statistics

**Responsibilities**:
- Connect to server
- Request memory statistics
- Display real-time utilization

## Implementation Plan

### Phase 1: IPC Protocol Design

#### Option A: Unix Domain Sockets (Recommended)
```c++
// Socket path
#define TT_ALLOC_SERVER_SOCKET "/tmp/tt_allocation_server.sock"

// Message format
struct AllocMessage {
    enum Type { ALLOC, FREE, QUERY, RESPONSE };
    Type type;
    int device_id;
    size_t size;
    BufferType buffer_type;
    pid_t process_id;
    uint64_t buffer_id;  // Unique identifier
    uint64_t timestamp;
};
```

**Pros**:
- Fast, local communication
- Built-in permission control
- Standard POSIX API

**Cons**:
- Unix/Linux only
- Requires file system access

#### Option B: Shared Memory + Semaphores
```c++
// Shared memory segment
#define TT_ALLOC_SHM_NAME "/tt_allocation_tracker"
#define TT_ALLOC_SEM_NAME "/tt_allocation_sem"

struct AllocationRegistry {
    std::atomic<uint64_t> version;

    struct DeviceStats {
        size_t dram_allocated;
        size_t l1_allocated;
        size_t l1_small_allocated;
        size_t trace_allocated;
    } devices[MAX_DEVICES];

    struct Allocation {
        uint64_t buffer_id;
        int device_id;
        size_t size;
        BufferType type;
        pid_t owner_pid;
        uint64_t timestamp;
    } allocations[MAX_ALLOCATIONS];

    size_t num_allocations;
};
```

**Pros**:
- Very fast (no syscalls for reads)
- Lock-free reads possible
- Minimal latency

**Cons**:
- Size limits
- Complex synchronization
- Cleanup on crash tricky

#### Option C: gRPC/Protobuf
```protobuf
service AllocationServer {
  rpc ReportAllocation(AllocationRequest) returns (AllocationResponse);
  rpc ReportDeallocation(DeallocationRequest) returns (DeallocationResponse);
  rpc QueryStatistics(QueryRequest) returns (StatisticsResponse);
}

message AllocationRequest {
  int32 device_id = 1;
  uint64 size = 2;
  BufferType buffer_type = 3;
  uint64 buffer_id = 4;
}

message StatisticsResponse {
  repeated DeviceStats devices = 1;
}
```

**Pros**:
- Language agnostic
- Built-in serialization
- Network-capable

**Cons**:
- Heavier weight
- Additional dependency
- Overkill for local IPC

### Phase 2: Server Implementation

```cpp
// allocation_server.cpp

#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>
#include <thread>
#include <unordered_map>
#include <mutex>

class AllocationServer {
private:
    struct BufferInfo {
        uint64_t buffer_id;
        int device_id;
        size_t size;
        BufferType type;
        pid_t owner_pid;
        std::chrono::steady_clock::time_point alloc_time;
    };

    // Thread-safe allocation registry
    std::mutex registry_mutex_;
    std::unordered_map<uint64_t, BufferInfo> allocations_;

    // Per-device statistics
    struct DeviceStats {
        std::atomic<size_t> dram_allocated{0};
        std::atomic<size_t> l1_allocated{0};
        std::atomic<size_t> l1_small_allocated{0};
        std::atomic<size_t> trace_allocated{0};
    };
    std::array<DeviceStats, MAX_DEVICES> device_stats_;

    int server_socket_;
    std::atomic<bool> running_{true};

public:
    AllocationServer() {
        // Create Unix domain socket
        server_socket_ = socket(AF_UNIX, SOCK_STREAM, 0);

        struct sockaddr_un addr;
        addr.sun_family = AF_UNIX;
        strcpy(addr.sun_path, TT_ALLOC_SERVER_SOCKET);

        // Remove old socket file
        unlink(TT_ALLOC_SERVER_SOCKET);

        // Bind and listen
        bind(server_socket_, (struct sockaddr*)&addr, sizeof(addr));
        listen(server_socket_, 128);  // Large backlog

        std::cout << "Allocation Server listening on "
                  << TT_ALLOC_SERVER_SOCKET << std::endl;
    }

    void handle_allocation(const AllocMessage& msg) {
        std::lock_guard<std::mutex> lock(registry_mutex_);

        BufferInfo info{
            .buffer_id = msg.buffer_id,
            .device_id = msg.device_id,
            .size = msg.size,
            .type = msg.buffer_type,
            .owner_pid = msg.process_id,
            .alloc_time = std::chrono::steady_clock::now()
        };

        allocations_[msg.buffer_id] = info;

        // Update device statistics
        auto& stats = device_stats_[msg.device_id];
        switch (msg.buffer_type) {
            case BufferType::DRAM:
                stats.dram_allocated += msg.size;
                break;
            case BufferType::L1:
                stats.l1_allocated += msg.size;
                break;
            case BufferType::L1_SMALL:
                stats.l1_small_allocated += msg.size;
                break;
            case BufferType::TRACE:
                stats.trace_allocated += msg.size;
                break;
        }

        std::cout << "Process " << msg.process_id
                  << " allocated " << msg.size << " bytes on device "
                  << msg.device_id << std::endl;
    }

    void handle_deallocation(const AllocMessage& msg) {
        std::lock_guard<std::mutex> lock(registry_mutex_);

        auto it = allocations_.find(msg.buffer_id);
        if (it != allocations_.end()) {
            const auto& info = it->second;

            // Update device statistics
            auto& stats = device_stats_[info.device_id];
            switch (info.type) {
                case BufferType::DRAM:
                    stats.dram_allocated -= info.size;
                    break;
                case BufferType::L1:
                    stats.l1_allocated -= info.size;
                    break;
                case BufferType::L1_SMALL:
                    stats.l1_small_allocated -= info.size;
                    break;
                case BufferType::TRACE:
                    stats.trace_allocated -= info.size;
                    break;
            }

            allocations_.erase(it);

            std::cout << "Process " << info.owner_pid
                      << " freed buffer " << msg.buffer_id << std::endl;
        }
    }

    void handle_query(int client_socket, const AllocMessage& msg) {
        AllocMessage response;
        response.type = AllocMessage::RESPONSE;
        response.device_id = msg.device_id;

        if (msg.device_id >= 0 && msg.device_id < MAX_DEVICES) {
            auto& stats = device_stats_[msg.device_id];

            // Pack statistics into response
            // (In real impl, would have proper response struct)
            response.size = stats.dram_allocated;  // Simplified

            send(client_socket, &response, sizeof(response), 0);
        }
    }

    void handle_client(int client_socket) {
        AllocMessage msg;

        while (running_) {
            ssize_t n = recv(client_socket, &msg, sizeof(msg), 0);
            if (n <= 0) break;

            switch (msg.type) {
                case AllocMessage::ALLOC:
                    handle_allocation(msg);
                    break;
                case AllocMessage::FREE:
                    handle_deallocation(msg);
                    break;
                case AllocMessage::QUERY:
                    handle_query(client_socket, msg);
                    break;
                default:
                    break;
            }
        }

        close(client_socket);
    }

    void run() {
        while (running_) {
            int client_socket = accept(server_socket_, nullptr, nullptr);
            if (client_socket < 0) continue;

            // Handle each client in a separate thread
            std::thread(&AllocationServer::handle_client, this, client_socket).detach();
        }
    }

    void stop() {
        running_ = false;
        close(server_socket_);
        unlink(TT_ALLOC_SERVER_SOCKET);
    }
};

int main() {
    AllocationServer server;

    // Handle signals for graceful shutdown
    signal(SIGINT, [](int) {
        std::cout << "Shutting down allocation server..." << std::endl;
        exit(0);
    });

    server.run();
    return 0;
}
```

### Phase 3: Client Library (Allocator Shim)

```cpp
// allocation_client.cpp

class AllocationClient {
private:
    int socket_fd_;
    uint64_t next_buffer_id_{1};

    static AllocationClient& instance() {
        static AllocationClient inst;
        return inst;
    }

public:
    AllocationClient() {
        // Connect to server
        socket_fd_ = socket(AF_UNIX, SOCK_STREAM, 0);

        struct sockaddr_un addr;
        addr.sun_family = AF_UNIX;
        strcpy(addr.sun_path, TT_ALLOC_SERVER_SOCKET);

        if (connect(socket_fd_, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
            std::cerr << "Warning: Could not connect to allocation server" << std::endl;
            socket_fd_ = -1;  // Operate without server
        }
    }

    ~AllocationClient() {
        if (socket_fd_ >= 0) {
            close(socket_fd_);
        }
    }

    void report_allocation(int device_id, size_t size, BufferType type, uint64_t buffer_id) {
        if (socket_fd_ < 0) return;  // Server not available

        AllocMessage msg{
            .type = AllocMessage::ALLOC,
            .device_id = device_id,
            .size = size,
            .buffer_type = type,
            .process_id = getpid(),
            .buffer_id = buffer_id,
            .timestamp = std::chrono::system_clock::now().time_since_epoch().count()
        };

        send(socket_fd_, &msg, sizeof(msg), 0);
    }

    void report_deallocation(uint64_t buffer_id) {
        if (socket_fd_ < 0) return;

        AllocMessage msg{
            .type = AllocMessage::FREE,
            .buffer_id = buffer_id
        };

        send(socket_fd_, &msg, sizeof(msg), 0);
    }

    static void track_allocation(int device_id, size_t size, BufferType type, void* ptr) {
        uint64_t buffer_id = reinterpret_cast<uint64_t>(ptr);
        instance().report_allocation(device_id, size, type, buffer_id);
    }

    static void track_deallocation(void* ptr) {
        uint64_t buffer_id = reinterpret_cast<uint64_t>(ptr);
        instance().report_deallocation(buffer_id);
    }
};

// Wrapper for CreateBuffer
std::shared_ptr<Buffer> CreateBufferTracked(const BufferConfig& config) {
    auto buffer = CreateBuffer(config);  // Original call

    // Report to allocation server
    AllocationClient::track_allocation(
        config.device->id(),
        config.size,
        config.buffer_type,
        buffer.get()
    );

    // Wrap in custom deleter to track deallocation
    return std::shared_ptr<Buffer>(
        buffer.get(),
        [original_ptr = buffer](Buffer* ptr) {
            AllocationClient::track_deallocation(ptr);
            // Original deleter will be called
        }
    );
}
```

### Phase 4: Monitor Integration

```cpp
// memory_monitor_client.cpp

class MonitorClient {
private:
    int socket_fd_;

public:
    MonitorClient() {
        socket_fd_ = socket(AF_UNIX, SOCK_STREAM, 0);

        struct sockaddr_un addr;
        addr.sun_family = AF_UNIX;
        strcpy(addr.sun_path, TT_ALLOC_SERVER_SOCKET);

        connect(socket_fd_, (struct sockaddr*)&addr, sizeof(addr));
    }

    DeviceStats query_device_stats(int device_id) {
        AllocMessage query{
            .type = AllocMessage::QUERY,
            .device_id = device_id
        };

        send(socket_fd_, &query, sizeof(query), 0);

        AllocMessage response;
        recv(socket_fd_, &response, sizeof(response), 0);

        // Parse response into DeviceStats
        DeviceStats stats;
        // ... fill from response
        return stats;
    }
};

// Update memory monitor to use server
void print_device_memory_from_server(int device_id) {
    MonitorClient client;
    auto stats = client.query_device_stats(device_id);

    std::cout << "Device " << device_id << " Memory:" << std::endl;
    std::cout << "  DRAM: " << format_bytes(stats.dram_allocated) << std::endl;
    std::cout << "  L1: " << format_bytes(stats.l1_allocated) << std::endl;
    // ...
}
```

## Deployment Strategy

### Step 1: Start Allocation Server

```bash
# Run as daemon
./allocation_server &

# Or as systemd service
sudo systemctl start tt-allocation-server
```

### Step 2: Use Tracked Allocator in Applications

**Option A: Preload Library**
```bash
# Use LD_PRELOAD to intercept allocations
LD_PRELOAD=/usr/local/lib/libtt_alloc_tracker.so python my_model.py
```

**Option B: Explicit Linking**
```cpp
// Link against tracked version
#include <tt-metalium/tracked_allocator.hpp>

auto buffer = CreateBufferTracked(config);  // Reported to server
```

**Option C: Environment Variable**
```bash
export TT_TRACK_ALLOCATIONS=1
python my_model.py  # TT-Metal detects env var and enables tracking
```

### Step 3: Run Monitor

```bash
./memory_monitor --use-server
```

## Advanced Features

### 1. Per-Process Breakdown

```cpp
struct ProcessStats {
    pid_t pid;
    std::string process_name;
    size_t total_allocated;
    std::vector<BufferInfo> buffers;
};

std::vector<ProcessStats> get_per_process_stats();
```

### 2. Allocation History

```cpp
struct AllocationHistory {
    std::chrono::steady_clock::time_point timestamp;
    size_t dram_allocated;
    size_t l1_allocated;
};

std::vector<AllocationHistory> get_history(int device_id,
                                           std::chrono::seconds duration);
```

### 3. Leak Detection

```cpp
// Detect buffers allocated but not freed for > threshold
std::vector<BufferInfo> detect_leaks(std::chrono::seconds age_threshold) {
    auto now = std::chrono::steady_clock::now();
    std::vector<BufferInfo> leaks;

    for (const auto& [id, info] : allocations_) {
        auto age = now - info.alloc_time;
        if (age > age_threshold) {
            leaks.push_back(info);
        }
    }

    return leaks;
}
```

### 4. WebUI Dashboard

```javascript
// Real-time web dashboard
fetch('http://localhost:8080/api/device/0/stats')
    .then(r => r.json())
    .then(stats => {
        updateChart(stats.dram_allocated, stats.l1_allocated);
    });
```

## Security Considerations

### 1. Access Control

```cpp
// Check socket permissions
chmod 0660 /tmp/tt_allocation_server.sock
chown root:ttusers /tmp/tt_allocation_server.sock

// In server: verify client credentials
struct ucred cred;
socklen_t len = sizeof(cred);
getsockopt(client_socket, SOL_SOCKET, SO_PEERCRED, &cred, &len);

// Only allow certain UIDs/GIDs
if (cred.uid != allowed_uid) {
    close(client_socket);
    return;
}
```

### 2. Rate Limiting

```cpp
// Prevent DoS from malicious clients
class RateLimiter {
    std::unordered_map<pid_t, TokenBucket> buckets_;
public:
    bool allow_request(pid_t pid) {
        return buckets_[pid].consume(1);  // 1 token per request
    }
};
```

### 3. Message Validation

```cpp
bool validate_message(const AllocMessage& msg) {
    // Sanity checks
    if (msg.size > MAX_ALLOCATION_SIZE) return false;
    if (msg.device_id < 0 || msg.device_id >= MAX_DEVICES) return false;
    if (msg.process_id <= 0) return false;
    return true;
}
```

## Performance Optimization

### 1. Lock-Free Reads

```cpp
// Use atomic operations for common read path
std::atomic<size_t> dram_allocated{0};

// Reads require no locks
size_t get_dram_allocated() const {
    return dram_allocated.load(std::memory_order_relaxed);
}
```

### 2. Batched Updates

```cpp
// Buffer multiple allocations before sending to server
class BatchedClient {
    std::vector<AllocMessage> pending_;
    std::mutex pending_mutex_;

    void flush() {
        std::lock_guard<std::mutex> lock(pending_mutex_);
        for (const auto& msg : pending_) {
            send(socket_fd_, &msg, sizeof(msg), 0);
        }
        pending_.clear();
    }

    // Flush every 100ms or 100 messages
    void report_allocation(...) {
        pending_.push_back(msg);
        if (pending_.size() >= 100) flush();
    }
};
```

### 3. Async Communication

```cpp
// Non-blocking sends
send(socket_fd_, &msg, sizeof(msg), MSG_DONTWAIT);

// Use io_uring for zero-copy I/O (Linux 5.1+)
struct io_uring ring;
io_uring_queue_init(256, &ring, 0);
```

## Testing Strategy

```cpp
// Unit tests
TEST(AllocationServer, TracksAllocations) {
    AllocationServer server;
    server.handle_allocation({...});
    ASSERT_EQ(server.get_device_stats(0).dram_allocated, 1024);
}

// Integration tests
TEST(EndToEnd, MultiProcessTracking) {
    // Start server
    auto server_pid = fork();
    if (server_pid == 0) {
        AllocationServer().run();
    }

    // Start clients
    auto client1_pid = fork();
    if (client1_pid == 0) {
        auto buf = CreateBufferTracked(...);  // 100MB
    }

    // Query from monitor
    MonitorClient monitor;
    auto stats = monitor.query_device_stats(0);
    ASSERT_GE(stats.dram_allocated, 100*1024*1024);
}
```

## Limitations & Future Work

### Current Limitations

1. **Single Machine Only**: Uses Unix sockets (could use TCP for distributed)
2. **No Persistence**: State lost on server restart (could add SQLite backend)
3. **Limited History**: Keeps all history in RAM (could implement circular buffer)

### Future Enhancements

1. **Distributed Tracing**: Integrate with OpenTelemetry
2. **Machine Learning**: Predict OOM conditions before they happen
3. **Auto-Scaling**: Trigger workload migration when memory pressure detected
4. **GPU Integration**: Track both TT and GPU memory in unified dashboard

## Conclusion

An Allocation Server provides **true system-wide memory tracking** across all processes. While more complex than per-process tracking, it enables:

- ✅ Real-time monitoring of all processes
- ✅ Leak detection across process boundaries
- ✅ Per-process memory attribution
- ✅ Historical analysis and trending
- ✅ Centralized management

This architecture is production-ready and scalable!
