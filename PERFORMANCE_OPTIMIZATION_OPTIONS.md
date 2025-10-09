# Buffer Tracking Performance Optimization Options

## Current Implementation
- **Blocking sends** with mutex serialization
- Ensures reliability but may impact performance

## Performance Analysis

### Theoretical Impact
```
Time per send = mutex_acquire + socket_send + mutex_release
              ≈ 100ns + 1-10μs + 100ns
              ≈ 1-10 microseconds per buffer operation
```

### When It Matters
- **High-frequency allocations**: >10K buffers/sec across all devices
- **Tight loops**: Rapid alloc/free cycles
- **MeshDevice operations**: 8 simultaneous buffer operations

### When It Doesn't Matter
- **Initialization**: One-time setup allocations
- **Model loading**: Infrequent, large buffers
- **Background operations**: Non-critical path

## Optimization Options

### Option 1: Increase Socket Buffer Size (RECOMMENDED)
**Impact**: Low effort, significant improvement

```cpp
// In allocation_client.cpp, after socket creation:
int sndbuf_size = 1024 * 1024;  // 1MB send buffer
setsockopt(socket_fd_, SOL_SOCKET, SO_SNDBUF, &sndbuf_size, sizeof(sndbuf_size));

int rcvbuf_size = 1024 * 1024;  // 1MB receive buffer (server side)
setsockopt(socket_fd_, SOL_SOCKET, SO_RCVBUF, &rcvbuf_size, sizeof(rcvbuf_size));
```

**Benefits**:
- Reduces blocking frequency
- Allows burst handling
- Minimal code change

**Trade-off**:
- Uses 2MB memory per process

---

### Option 2: Hybrid Blocking/Non-blocking
**Impact**: Medium effort, good performance

```cpp
// Try non-blocking first
ssize_t sent = send(socket_fd_, &msg, sizeof(msg), MSG_DONTWAIT);
if (sent == sizeof(msg)) {
    return;  // Success, fast path
}

// Fall back to blocking if would block or partial send
if (sent < 0 && errno == EAGAIN) {
    // Socket full, use blocking
    sent = 0;
}

// Blocking retry loop for remainder
while (total_sent < sizeof(msg)) {
    ssize_t n = send(socket_fd_,
                    reinterpret_cast<const char*>(&msg) + total_sent,
                    sizeof(msg) - total_sent,
                    0);
    // ... error handling ...
}
```

**Benefits**:
- Fast path for normal operation
- Reliable fallback for bursts

**Trade-off**:
- More complex logic
- Two system calls in worst case

---

### Option 3: Asynchronous Queue (HIGH PERFORMANCE)
**Impact**: High effort, best performance

Architecture:
```
Application Thread          Background Thread           Server
      │                           │                        │
      ├─ Alloc buffer             │                        │
      ├─ Push to queue ──────────>│                        │
      └─ Continue (no wait)       ├─ Pop from queue        │
                                  ├─ send() ──────────────>│
                                  └─ Continue              │
```

Implementation sketch:
```cpp
class AllocationClient {
private:
    std::queue<AllocMessage> pending_messages_;
    std::mutex queue_mutex_;
    std::condition_variable queue_cv_;
    std::thread sender_thread_;
    std::atomic<bool> running_{true};

    void sender_thread_fn() {
        while (running_) {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            queue_cv_.wait(lock, [this] {
                return !pending_messages_.empty() || !running_;
            });

            if (!running_) break;

            AllocMessage msg = pending_messages_.front();
            pending_messages_.pop();
            lock.unlock();

            // Send without holding lock
            send_message_blocking(msg);
        }
    }

public:
    void send_allocation_message(...) {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        pending_messages_.push(msg);
        queue_cv_.notify_one();  // Wake sender thread
    }
};
```

**Benefits**:
- **Zero blocking** for application threads
- Handles bursts gracefully
- Better CPU utilization

**Trade-offs**:
- Background thread overhead
- Message reordering possible (may need timestamps)
- More complex shutdown logic
- Memory for queue

---

### Option 4: Batching
**Impact**: Medium effort, reduces syscall overhead

```cpp
// Accumulate multiple messages, send as batch
struct MessageBatch {
    AllocMessage messages[32];
    size_t count;
};

void flush_batch(MessageBatch& batch) {
    if (batch.count == 0) return;

    size_t total_size = sizeof(AllocMessage) * batch.count;
    // Send all at once
    send(socket_fd_, batch.messages, total_size, 0);
    batch.count = 0;
}
```

**Benefits**:
- Fewer syscalls
- Better throughput

**Trade-offs**:
- Latency increase (messages delayed)
- Need flush policy (time-based or count-based)
- Complicates error handling

---

### Option 5: Conditional Tracking
**Impact**: Low effort, application-level optimization

```cpp
// Only enable tracking for specific operations
class ScopedTrackingControl {
public:
    ScopedTrackingControl(bool enable) {
        AllocationClient::set_enabled(enable);
    }
    ~ScopedTrackingControl() {
        AllocationClient::set_enabled(true);
    }
};

// In performance-critical sections:
{
    ScopedTrackingControl no_track(false);
    // Rapid buffer operations here - not tracked
}
```

**Benefits**:
- Zero overhead when disabled
- Surgical control

**Trade-offs**:
- Incomplete tracking
- Manual management

---

## Benchmarking

### Create Performance Test

```cpp
// test_allocation_perf.cpp
#include <chrono>
#include <iostream>

void benchmark_allocation(int num_iterations) {
    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < num_iterations; i++) {
        // Allocate buffer (calls tracking)
        auto buffer = Buffer::create(device, size, ...);
        // Deallocate (calls tracking)
        buffer.reset();
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    std::cout << "Total: " << duration.count() << " μs\n";
    std::cout << "Per operation: " << (duration.count() / num_iterations) << " μs\n";
}

int main() {
    std::cout << "=== With Tracking ===\n";
    setenv("TT_ALLOC_TRACKING_ENABLED", "1", 1);
    benchmark_allocation(10000);

    std::cout << "\n=== Without Tracking ===\n";
    setenv("TT_ALLOC_TRACKING_ENABLED", "0", 1);
    benchmark_allocation(10000);
}
```

Run:
```bash
./test_allocation_perf
```

---

## Recommendations

### For Production Use:
1. **Start with Option 1** (larger socket buffer) - easiest, often sufficient
2. **Measure actual impact** with your workload
3. **If < 1% overhead**: Keep current implementation
4. **If > 5% overhead**: Consider Option 3 (async queue)

### For Development/Debug:
- Current blocking implementation is perfect
- Reliability > performance during development

### For High-Performance Inference:
- Option 3 (async queue) or Option 5 (conditional tracking)
- Disable tracking in production, enable only for debugging

---

## Quick Performance Check

Run this to measure overhead:
```bash
cd /workspace/tt-metal-apv

# Compile test program
python3 -c "
import time

# With tracking
import os
os.environ['TT_ALLOC_TRACKING_ENABLED'] = '1'
start = time.time()
# Your workload here
elapsed_with = time.time() - start

# Without tracking
os.environ['TT_ALLOC_TRACKING_ENABLED'] = '0'
start = time.time()
# Same workload here
elapsed_without = time.time() - start

overhead = ((elapsed_with - elapsed_without) / elapsed_without) * 100
print(f'Overhead: {overhead:.2f}%')
"
```

If overhead < 2%, you're fine with current implementation!
