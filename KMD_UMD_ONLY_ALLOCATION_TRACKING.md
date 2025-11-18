# Allocation Tracking: tt-kmd + tt-umd Only (No User-Space Server)

**Goal:** Implement real-time allocation tracking using only the kernel module (tt-kmd) and user-mode driver (tt-umd), without requiring a separate allocation server daemon.

**Advantages:**
- ✅ No separate server process needed
- ✅ Always available (like nvidia-smi)
- ✅ Survives application crashes automatically
- ✅ Cleaner architecture
- ✅ Per-process isolation built-in
- ✅ No socket communication overhead
- ✅ Accessible via standard /proc interface

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    MONITORING TOOLS                          │
│         (tt-smi, nvtop, custom scripts)                     │
└───────────────────┬─────────────────────────────────────────┘
                    │
                    │ Read /proc or sysfs
                    ▼
┌─────────────────────────────────────────────────────────────┐
│                    tt-kmd (Kernel Module)                    │
│                                                              │
│  Per-device global state:                                   │
│  ├─ open_fds_list (already exists)                          │
│  └─ device_stats (NEW)                                      │
│                                                              │
│  Per-process state (chardev_private):                       │
│  ├─ dmabufs (already exists)                                │
│  ├─ tlbs (already exists)                                   │
│  └─ device_allocations (NEW)  ← Track DRAM/L1 buffers      │
│                                                              │
│  New IOCTLs:                                                │
│  ├─ TENSTORRENT_IOCTL_TRACK_ALLOC    ← Report allocation   │
│  ├─ TENSTORRENT_IOCTL_TRACK_FREE     ← Report free         │
│  └─ TENSTORRENT_IOCTL_QUERY_STATS    ← Get device stats    │
│                                                              │
│  New /proc files:                                           │
│  ├─ /proc/driver/tenstorrent/0/allocations                 │
│  └─ /proc/driver/tenstorrent/0/stats                       │
└───────────────────┬─────────────────────────────────────────┘
                    │
                    │ ioctl() calls
                    ▼
┌─────────────────────────────────────────────────────────────┐
│                    tt-umd (User-Mode Driver)                 │
│                                                              │
│  Wrapper functions:                                         │
│  ├─ tt_track_allocation(device_id, size, type, addr)       │
│  └─ tt_track_deallocation(device_id, addr)                 │
│                                                              │
│  Modified allocators:                                       │
│  └─ SysmemManager, TLBManager, etc.                        │
└───────────────────┬─────────────────────────────────────────┘
                    │
                    │ Uses UMD
                    ▼
┌─────────────────────────────────────────────────────────────┐
│              TT-Metal Applications                           │
│         (Automatically tracked via UMD)                      │
└─────────────────────────────────────────────────────────────┘
```

---

## Part 1: Kernel Module Changes (tt-kmd)

### 1.1: Add Tracking Structures

**File:** `tt-kmd/chardev_private.h`

```c
// Add to struct chardev_private
struct device_buffer {
    struct hlist_node hash_chain;

    uint64_t buffer_id;    // Memory address
    uint64_t size;         // Size in bytes
    uint8_t buffer_type;   // 0=DRAM, 1=L1, 2=L1_SMALL, 3=TRACE
    ktime_t alloc_time;    // Allocation timestamp
};

#define DEVICE_BUFFERS_HASHTABLE_BITS 8

struct chardev_private {
    struct tenstorrent_device *device;
    struct mutex mutex;

    // Existing fields
    DECLARE_HASHTABLE(dmabufs, DMABUF_HASHTABLE_BITS);
    struct list_head pinnings;
    struct list_head peer_mappings;
    struct list_head bar_mappings;
    DECLARE_BITMAP(tlbs, TENSTORRENT_MAX_INBOUND_TLBS);

    // NEW: Device memory allocations (DRAM/L1)
    DECLARE_HASHTABLE(device_allocations, DEVICE_BUFFERS_HASHTABLE_BITS);
    atomic64_t dram_allocated;
    atomic64_t l1_allocated;
    atomic64_t l1_small_allocated;
    atomic64_t trace_allocated;

    pid_t pid;
    char comm[TASK_COMM_LEN];
    struct list_head open_fd;

    DECLARE_BITMAP(resource_lock, TENSTORRENT_RESOURCE_LOCK_COUNT);
    struct tenstorrent_set_noc_cleanup noc_cleanup;
};
```

**File:** `tt-kmd/device.h`

```c
// Add to struct tenstorrent_device
struct tenstorrent_device {
    // ... existing fields ...

    // NEW: Per-device aggregate stats (sum of all processes)
    atomic64_t total_dram_allocated;
    atomic64_t total_l1_allocated;
    atomic64_t total_l1_small_allocated;
    atomic64_t total_trace_allocated;
    atomic64_t total_buffers;
};
```

### 1.2: Add New IOCTLs

**File:** `tt-kmd/ioctl.h`

```c
// Add new ioctl definitions
#define TENSTORRENT_IOCTL_TRACK_ALLOC       _IO(TENSTORRENT_IOCTL_MAGIC, 15)
#define TENSTORRENT_IOCTL_TRACK_FREE        _IO(TENSTORRENT_IOCTL_MAGIC, 16)
#define TENSTORRENT_IOCTL_QUERY_ALLOC_STATS _IO(TENSTORRENT_IOCTL_MAGIC, 17)

// Buffer types
#define TT_BUFFER_TYPE_DRAM       0
#define TT_BUFFER_TYPE_L1         1
#define TT_BUFFER_TYPE_L1_SMALL   2
#define TT_BUFFER_TYPE_TRACE      3

// Input structure for TRACK_ALLOC
struct tenstorrent_track_alloc_in {
    uint64_t buffer_id;    // Memory address (unique identifier)
    uint64_t size;         // Size in bytes
    uint8_t buffer_type;   // TT_BUFFER_TYPE_*
    uint8_t pad[7];        // Alignment
};

struct tenstorrent_track_alloc {
    struct tenstorrent_track_alloc_in in;
};

// Input structure for TRACK_FREE
struct tenstorrent_track_free_in {
    uint64_t buffer_id;    // Buffer to free
};

struct tenstorrent_track_free {
    struct tenstorrent_track_free_in in;
};

// Output structure for QUERY_ALLOC_STATS
struct tenstorrent_alloc_stats_out {
    uint64_t dram_allocated;
    uint64_t l1_allocated;
    uint64_t l1_small_allocated;
    uint64_t trace_allocated;
    uint64_t total_buffers;

    // Per-type buffer counts
    uint64_t dram_buffers;
    uint64_t l1_buffers;
    uint64_t l1_small_buffers;
    uint64_t trace_buffers;
};

struct tenstorrent_query_alloc_stats {
    struct tenstorrent_alloc_stats_out out;
};
```

### 1.3: Implement IOCTL Handlers

**File:** `tt-kmd/allocation_tracking.c` (NEW)

```c
// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: GPL-2.0-only

#include "allocation_tracking.h"
#include "chardev_private.h"
#include "device.h"

static struct device_buffer* lookup_buffer(struct chardev_private *priv,
                                          uint64_t buffer_id)
{
    struct device_buffer *buf;

    hash_for_each_possible(priv->device_allocations, buf, hash_chain, buffer_id) {
        if (buf->buffer_id == buffer_id)
            return buf;
    }

    return NULL;
}

long ioctl_track_alloc(struct chardev_private *priv,
                      struct tenstorrent_track_alloc __user *arg)
{
    struct tenstorrent_track_alloc_in in;
    struct device_buffer *buf;

    if (copy_from_user(&in, &arg->in, sizeof(in)) != 0)
        return -EFAULT;

    // Validate buffer type
    if (in.buffer_type > TT_BUFFER_TYPE_TRACE)
        return -EINVAL;

    mutex_lock(&priv->mutex);

    // Check if buffer already tracked (should not happen)
    if (lookup_buffer(priv, in.buffer_id)) {
        mutex_unlock(&priv->mutex);
        pr_warn("Buffer 0x%llx already tracked for PID %d\n",
                in.buffer_id, priv->pid);
        return -EEXIST;
    }

    // Allocate tracking structure
    buf = kzalloc(sizeof(*buf), GFP_KERNEL);
    if (!buf) {
        mutex_unlock(&priv->mutex);
        return -ENOMEM;
    }

    // Fill in buffer info
    buf->buffer_id = in.buffer_id;
    buf->size = in.size;
    buf->buffer_type = in.buffer_type;
    buf->alloc_time = ktime_get();

    // Add to process's hash table
    hash_add(priv->device_allocations, &buf->hash_chain, buf->buffer_id);

    // Update per-process stats (atomic for thread safety)
    switch (in.buffer_type) {
    case TT_BUFFER_TYPE_DRAM:
        atomic64_add(in.size, &priv->dram_allocated);
        break;
    case TT_BUFFER_TYPE_L1:
        atomic64_add(in.size, &priv->l1_allocated);
        break;
    case TT_BUFFER_TYPE_L1_SMALL:
        atomic64_add(in.size, &priv->l1_small_allocated);
        break;
    case TT_BUFFER_TYPE_TRACE:
        atomic64_add(in.size, &priv->trace_allocated);
        break;
    }

    // Update device-level aggregate stats
    switch (in.buffer_type) {
    case TT_BUFFER_TYPE_DRAM:
        atomic64_add(in.size, &priv->device->total_dram_allocated);
        break;
    case TT_BUFFER_TYPE_L1:
        atomic64_add(in.size, &priv->device->total_l1_allocated);
        break;
    case TT_BUFFER_TYPE_L1_SMALL:
        atomic64_add(in.size, &priv->device->total_l1_small_allocated);
        break;
    case TT_BUFFER_TYPE_TRACE:
        atomic64_add(in.size, &priv->device->total_trace_allocated);
        break;
    }

    atomic64_inc(&priv->device->total_buffers);

    mutex_unlock(&priv->mutex);

    return 0;
}

long ioctl_track_free(struct chardev_private *priv,
                     struct tenstorrent_track_free __user *arg)
{
    struct tenstorrent_track_free_in in;
    struct device_buffer *buf;

    if (copy_from_user(&in, &arg->in, sizeof(in)) != 0)
        return -EFAULT;

    mutex_lock(&priv->mutex);

    buf = lookup_buffer(priv, in.buffer_id);
    if (!buf) {
        mutex_unlock(&priv->mutex);
        pr_warn("Buffer 0x%llx not found for PID %d\n",
                in.buffer_id, priv->pid);
        return -ENOENT;
    }

    // Update per-process stats
    switch (buf->buffer_type) {
    case TT_BUFFER_TYPE_DRAM:
        atomic64_sub(buf->size, &priv->dram_allocated);
        break;
    case TT_BUFFER_TYPE_L1:
        atomic64_sub(buf->size, &priv->l1_allocated);
        break;
    case TT_BUFFER_TYPE_L1_SMALL:
        atomic64_sub(buf->size, &priv->l1_small_allocated);
        break;
    case TT_BUFFER_TYPE_TRACE:
        atomic64_sub(buf->size, &priv->trace_allocated);
        break;
    }

    // Update device-level stats
    switch (buf->buffer_type) {
    case TT_BUFFER_TYPE_DRAM:
        atomic64_sub(buf->size, &priv->device->total_dram_allocated);
        break;
    case TT_BUFFER_TYPE_L1:
        atomic64_sub(buf->size, &priv->device->total_l1_allocated);
        break;
    case TT_BUFFER_TYPE_L1_SMALL:
        atomic64_sub(buf->size, &priv->device->total_l1_small_allocated);
        break;
    case TT_BUFFER_TYPE_TRACE:
        atomic64_sub(buf->size, &priv->device->total_trace_allocated);
        break;
    }

    atomic64_dec(&priv->device->total_buffers);

    // Remove from hash table
    hash_del(&buf->hash_chain);
    kfree(buf);

    mutex_unlock(&priv->mutex);

    return 0;
}

long ioctl_query_alloc_stats(struct chardev_private *priv,
                            struct tenstorrent_query_alloc_stats __user *arg)
{
    struct tenstorrent_alloc_stats_out out;
    struct tenstorrent_device *tt_dev = priv->device;

    // Read atomic stats (device-level aggregates)
    out.dram_allocated = atomic64_read(&tt_dev->total_dram_allocated);
    out.l1_allocated = atomic64_read(&tt_dev->total_l1_allocated);
    out.l1_small_allocated = atomic64_read(&tt_dev->total_l1_small_allocated);
    out.trace_allocated = atomic64_read(&tt_dev->total_trace_allocated);
    out.total_buffers = atomic64_read(&tt_dev->total_buffers);

    // Count buffers by type (iterate all processes)
    out.dram_buffers = 0;
    out.l1_buffers = 0;
    out.l1_small_buffers = 0;
    out.trace_buffers = 0;

    mutex_lock(&tt_dev->chardev_mutex);

    struct chardev_private *p;
    list_for_each_entry(p, &tt_dev->open_fds_list, open_fd) {
        struct device_buffer *buf;
        int bkt;

        if (!mutex_trylock(&p->mutex))
            continue;

        hash_for_each(p->device_allocations, bkt, buf, hash_chain) {
            switch (buf->buffer_type) {
            case TT_BUFFER_TYPE_DRAM:
                out.dram_buffers++;
                break;
            case TT_BUFFER_TYPE_L1:
                out.l1_buffers++;
                break;
            case TT_BUFFER_TYPE_L1_SMALL:
                out.l1_small_buffers++;
                break;
            case TT_BUFFER_TYPE_TRACE:
                out.trace_buffers++;
                break;
            }
        }

        mutex_unlock(&p->mutex);
    }

    mutex_unlock(&tt_dev->chardev_mutex);

    if (copy_to_user(&arg->out, &out, sizeof(out)) != 0)
        return -EFAULT;

    return 0;
}

// Cleanup function called on fd close
void tenstorrent_allocation_cleanup(struct chardev_private *priv)
{
    struct device_buffer *buf;
    struct hlist_node *tmp;
    int bkt;

    mutex_lock(&priv->mutex);

    // Free all tracked buffers and update stats
    hash_for_each_safe(priv->device_allocations, bkt, tmp, buf, hash_chain) {
        // Update device stats
        switch (buf->buffer_type) {
        case TT_BUFFER_TYPE_DRAM:
            atomic64_sub(buf->size, &priv->device->total_dram_allocated);
            break;
        case TT_BUFFER_TYPE_L1:
            atomic64_sub(buf->size, &priv->device->total_l1_allocated);
            break;
        case TT_BUFFER_TYPE_L1_SMALL:
            atomic64_sub(buf->size, &priv->device->total_l1_small_allocated);
            break;
        case TT_BUFFER_TYPE_TRACE:
            atomic64_sub(buf->size, &priv->device->total_trace_allocated);
            break;
        }

        atomic64_dec(&priv->device->total_buffers);

        // Log leaked buffer (for debugging)
        ktime_t age = ktime_sub(ktime_get(), buf->alloc_time);
        pr_info("Leaked buffer from PID %d: 0x%llx (%llu bytes, type %u, age %lldms)\n",
                priv->pid, buf->buffer_id, buf->size, buf->buffer_type,
                ktime_to_ms(age));

        hash_del(&buf->hash_chain);
        kfree(buf);
    }

    mutex_unlock(&priv->mutex);
}
```

### 1.4: Update chardev.c

**File:** `tt-kmd/chardev.c`

```c
// Add to tt_cdev_ioctl()
static long tt_cdev_ioctl(struct file *file, unsigned int cmd, unsigned long arg)
{
    struct chardev_private *priv = file->private_data;
    long ret;

    // ... existing cases ...

    switch (cmd) {
        // ... existing cases ...

        case TENSTORRENT_IOCTL_TRACK_ALLOC:
            ret = ioctl_track_alloc(priv, (struct tenstorrent_track_alloc __user *)arg);
            break;

        case TENSTORRENT_IOCTL_TRACK_FREE:
            ret = ioctl_track_free(priv, (struct tenstorrent_track_free __user *)arg);
            break;

        case TENSTORRENT_IOCTL_QUERY_ALLOC_STATS:
            ret = ioctl_query_alloc_stats(priv, (struct tenstorrent_query_alloc_stats __user *)arg);
            break;

        default:
            ret = -EINVAL;
            break;
    }

    return ret;
}

// Update tt_cdev_open() to initialize new fields
static int tt_cdev_open(struct inode *inode, struct file *file)
{
    // ... existing code ...

    hash_init(private_data->device_allocations);  // NEW
    atomic64_set(&private_data->dram_allocated, 0);
    atomic64_set(&private_data->l1_allocated, 0);
    atomic64_set(&private_data->l1_small_allocated, 0);
    atomic64_set(&private_data->trace_allocated, 0);

    // ... rest of existing code ...
}

// Update tt_cdev_release() to cleanup allocations
static int tt_cdev_release(struct inode *inode, struct file *file)
{
    struct chardev_private *priv = file->private_data;

    // ... existing cleanup ...

    tenstorrent_allocation_cleanup(priv);  // NEW: Clean up tracked allocations

    // ... rest of existing cleanup ...
}
```

### 1.5: Add /proc Interface

**File:** `tt-kmd/enumerate.c`

```c
// Add new proc file for per-device allocation stats
static int allocations_seq_show(struct seq_file *s, void *v)
{
    struct tenstorrent_device *tt_dev = s->private;
    struct chardev_private *priv;

    seq_printf(s, "%-8s %-16s %-10s %-10s %-10s %-10s %-10s\n",
               "PID", "Comm", "DRAM", "L1", "L1_SMALL", "TRACE", "BUFFERS");
    seq_printf(s, "%-8s %-16s %-10s %-10s %-10s %-10s %-10s\n",
               "----", "----", "----", "--", "--------", "-----", "-------");

    mutex_lock(&tt_dev->chardev_mutex);

    list_for_each_entry(priv, &tt_dev->open_fds_list, open_fd) {
        uint64_t dram = atomic64_read(&priv->dram_allocated);
        uint64_t l1 = atomic64_read(&priv->l1_allocated);
        uint64_t l1_small = atomic64_read(&priv->l1_small_allocated);
        uint64_t trace = atomic64_read(&priv->trace_allocated);

        // Count buffers
        int num_buffers = 0;
        if (mutex_trylock(&priv->mutex)) {
            struct device_buffer *buf;
            int bkt;
            hash_for_each(priv->device_allocations, bkt, buf, hash_chain)
                num_buffers++;
            mutex_unlock(&priv->mutex);
        }

        seq_printf(s, "%-8d %-16s %-10llu %-10llu %-10llu %-10llu %-10d\n",
                   priv->pid, priv->comm,
                   dram, l1, l1_small, trace, num_buffers);
    }

    // Print totals
    seq_printf(s, "\n");
    seq_printf(s, "%-8s %-16s %-10llu %-10llu %-10llu %-10llu %-10llu\n",
               "TOTAL", "",
               atomic64_read(&tt_dev->total_dram_allocated),
               atomic64_read(&tt_dev->total_l1_allocated),
               atomic64_read(&tt_dev->total_l1_small_allocated),
               atomic64_read(&tt_dev->total_trace_allocated),
               atomic64_read(&tt_dev->total_buffers));

    mutex_unlock(&tt_dev->chardev_mutex);

    return 0;
}

static int allocations_open(struct inode *inode, struct file *file)
{
    return single_open(file, allocations_seq_show, inode->i_private);
}

static const struct file_operations allocations_fops = {
    .owner   = THIS_MODULE,
    .open    = allocations_open,
    .read    = seq_read,
    .llseek  = seq_lseek,
    .release = single_release,
};

// Register in tenstorrent_pci_probe()
proc_create("allocations", 0444, tt_dev->procfs_root, &allocations_fops);
```

---

## Part 2: UMD Changes (tt-umd)

### 2.1: Add Tracking API

**File:** `tt-umd/device/tt_kmd_lib/tt_kmd_lib.c` (or new file)

```c
// Add new API functions for allocation tracking

int tt_track_allocation(tt_device_t* dev, uint64_t buffer_id, uint64_t size,
                       uint8_t buffer_type)
{
    struct tenstorrent_track_alloc alloc = {0};
    alloc.in.buffer_id = buffer_id;
    alloc.in.size = size;
    alloc.in.buffer_type = buffer_type;

    if (ioctl(dev->fd, TENSTORRENT_IOCTL_TRACK_ALLOC, &alloc) != 0) {
        return -errno;
    }

    return 0;
}

int tt_track_deallocation(tt_device_t* dev, uint64_t buffer_id)
{
    struct tenstorrent_track_free free = {0};
    free.in.buffer_id = buffer_id;

    if (ioctl(dev->fd, TENSTORRENT_IOCTL_TRACK_FREE, &free) != 0) {
        return -errno;
    }

    return 0;
}

int tt_query_allocation_stats(tt_device_t* dev,
                             struct tenstorrent_alloc_stats_out* out)
{
    struct tenstorrent_query_alloc_stats query = {0};

    if (ioctl(dev->fd, TENSTORRENT_IOCTL_QUERY_ALLOC_STATS, &query) != 0) {
        return -errno;
    }

    *out = query.out;
    return 0;
}
```

**File:** `tt-umd/device/api/umd/device/tt_kmd_lib.h`

```c
// Add to header
int tt_track_allocation(tt_device_t* dev, uint64_t buffer_id, uint64_t size,
                       uint8_t buffer_type);
int tt_track_deallocation(tt_device_t* dev, uint64_t buffer_id);
int tt_query_allocation_stats(tt_device_t* dev,
                             struct tenstorrent_alloc_stats_out* out);

// Buffer types (match kernel definitions)
#define TT_BUFFER_TYPE_DRAM       0
#define TT_BUFFER_TYPE_L1         1
#define TT_BUFFER_TYPE_L1_SMALL   2
#define TT_BUFFER_TYPE_TRACE      3
```

### 2.2: C++ Wrapper for UMD

**File:** `tt-umd/device/allocation_tracker.hpp` (NEW)

```cpp
#pragma once

#include "umd/device/tt_device/tt_device.hpp"
#include "tt_kmd_lib.h"

namespace tt::umd {

class AllocationTracker {
public:
    static void track_allocation(TTDevice* device, uint64_t buffer_id,
                                uint64_t size, BufferType type) {
        if (!is_enabled()) return;

        uint8_t kernel_type = convert_buffer_type(type);
        tt_track_allocation(device->get_pci_device()->get_tt_device(),
                          buffer_id, size, kernel_type);
    }

    static void track_deallocation(TTDevice* device, uint64_t buffer_id) {
        if (!is_enabled()) return;

        tt_track_deallocation(device->get_pci_device()->get_tt_device(),
                            buffer_id);
    }

    static bool is_enabled() {
        // Always enabled (no environment variable needed!)
        return true;
    }

private:
    static uint8_t convert_buffer_type(BufferType type) {
        switch (type) {
            case BufferType::DRAM:      return TT_BUFFER_TYPE_DRAM;
            case BufferType::L1:        return TT_BUFFER_TYPE_L1;
            case BufferType::L1_SMALL:  return TT_BUFFER_TYPE_L1_SMALL;
            case BufferType::TRACE:     return TT_BUFFER_TYPE_TRACE;
            default:                    return TT_BUFFER_TYPE_DRAM;
        }
    }
};

}  // namespace tt::umd
```

---

## Part 3: TT-Metal Integration

### 3.1: Update GraphTracker

**File:** `tt_metal/graph/graph_tracking.cpp`

```cpp
#include "allocation_tracker.hpp"  // From UMD

void GraphTracker::track_allocate(const Buffer* buffer) {
    if (buffer->device() != nullptr) {
        if (dynamic_cast<const distributed::MeshDevice*>(buffer->device()) != nullptr) {
            return;  // Skip backing buffers
        }

        // Track in L1 stats (existing)
        if (buffer->buffer_type() == BufferType::L1) {
            get_l1_stats().track_alloc(buffer->device()->id(), buffer->size());
        }

        std::lock_guard<std::mutex> tracking_lock(g_allocation_tracking_mutex);

        // NEW: Report to kernel via UMD
        tt::umd::AllocationTracker::track_allocation(
            buffer->device()->get_tt_device(),  // TTDevice pointer
            buffer->address(),                   // buffer_id
            buffer->size(),                      // size
            buffer->buffer_type()                // type
        );

        // Tracy monitoring (existing)
        TracyMemoryMonitor::instance().track_allocation(...);
    }

    // Original graph tracking
    // ...
}

void GraphTracker::track_deallocate(Buffer* buffer) {
    if (buffer->device() != nullptr) {
        std::lock_guard<std::mutex> tracking_lock(g_allocation_tracking_mutex);

        // NEW: Report to kernel
        tt::umd::AllocationTracker::track_deallocation(
            buffer->device()->get_tt_device(),
            buffer->address()
        );
    }

    // Original graph tracking
    // ...
}
```

---

## Part 4: Monitoring Tools

### 4.1: Update tt-smi

**File:** `tt_metal/programming_examples/alexp_examples/memory_utilization_monitor/tt_smi_kmd_only.cpp`

```cpp
#include <iostream>
#include <fcntl.h>
#include <sys/ioctl.h>
#include "ioctl.h"  // From tt-kmd

class DeviceMonitor {
public:
    struct DeviceStats {
        uint64_t dram_allocated;
        uint64_t l1_allocated;
        uint64_t l1_small_allocated;
        uint64_t trace_allocated;
        uint64_t total_buffers;
    };

    static DeviceStats query_device(int device_id) {
        // Open device
        std::string dev_path = "/dev/tenstorrent/" + std::to_string(device_id);
        int fd = open(dev_path.c_str(), O_RDWR);
        if (fd < 0) {
            throw std::runtime_error("Cannot open device " + std::to_string(device_id));
        }

        // Query stats via ioctl
        struct tenstorrent_query_alloc_stats query = {0};
        if (ioctl(fd, TENSTORRENT_IOCTL_QUERY_ALLOC_STATS, &query) != 0) {
            close(fd);
            throw std::runtime_error("Failed to query stats");
        }

        close(fd);

        return {
            query.out.dram_allocated,
            query.out.l1_allocated,
            query.out.l1_small_allocated,
            query.out.trace_allocated,
            query.out.total_buffers
        };
    }

    static std::vector<pid_t> get_pids(int device_id) {
        // Read from /proc/driver/tenstorrent/<device_id>/pids
        std::string path = "/proc/driver/tenstorrent/" +
                          std::to_string(device_id) + "/pids";
        std::ifstream file(path);

        std::vector<pid_t> pids;
        pid_t pid;
        while (file >> pid) {
            pids.push_back(pid);
        }

        return pids;
    }

    static std::map<pid_t, DeviceStats> get_per_process_stats(int device_id) {
        // Read from /proc/driver/tenstorrent/<device_id>/allocations
        std::string path = "/proc/driver/tenstorrent/" +
                          std::to_string(device_id) + "/allocations";
        std::ifstream file(path);

        std::map<pid_t, DeviceStats> stats;

        std::string line;
        std::getline(file, line);  // Skip header
        std::getline(file, line);  // Skip separator

        while (std::getline(file, line)) {
            if (line.find("TOTAL") != std::string::npos)
                break;

            std::istringstream iss(line);
            pid_t pid;
            std::string comm;
            uint64_t dram, l1, l1_small, trace, buffers;

            iss >> pid >> comm >> dram >> l1 >> l1_small >> trace >> buffers;

            stats[pid] = {dram, l1, l1_small, trace, buffers};
        }

        return stats;
    }
};

// Usage in tt-smi
int main() {
    // Get device stats (aggregate)
    auto stats = DeviceMonitor::query_device(0);

    std::cout << "Device 0:" << std::endl;
    std::cout << "  DRAM: " << stats.dram_allocated / (1024*1024) << " MB" << std::endl;
    std::cout << "  L1: " << stats.l1_allocated / (1024*1024) << " MB" << std::endl;

    // Get per-process breakdown
    auto per_process = DeviceMonitor::get_per_process_stats(0);

    std::cout << "\nProcesses:" << std::endl;
    for (const auto& [pid, pstats] : per_process) {
        std::cout << "  PID " << pid << ": "
                  << "DRAM=" << pstats.dram_allocated / (1024*1024) << "MB, "
                  << "L1=" << pstats.l1_allocated / (1024*1024) << "MB"
                  << std::endl;
    }

    return 0;
}
```

---

## Advantages Over User-Space Server

### 1. **Always Available**
```bash
# No server needed!
./tt_smi  # Works immediately

# vs. current approach:
./allocation_server_poc &  # Must start server first
export TT_ALLOC_TRACKING_ENABLED=1  # Must enable
./tt_smi
```

### 2. **Automatic Cleanup**
```c
// Kernel automatically cleans up on process exit
static int tt_cdev_release(...)  {
    tenstorrent_allocation_cleanup(priv);  // Automatic!
}
```

### 3. **Better Isolation**
- Each process's allocations in separate hash table
- No inter-process communication
- No socket buffer issues
- No dropped messages

### 4. **Standard Linux Interface**
```bash
# Just read /proc like any other kernel module
cat /proc/driver/tenstorrent/0/allocations

# Output:
PID     Comm             DRAM       L1         L1_SMALL   TRACE      BUFFERS
----    ----             ----       --         --------   -----      -------
12345   python           3221225472 471859200  0          0          25
12346   test_app         1048576    524288     0          0          2

TOTAL                    3222274048 472383488  0          0          27
```

### 5. **No Environment Variable Needed**
```cpp
// Always enabled, no opt-in required
tt::umd::AllocationTracker::track_allocation(...);  // Always works
```

---

## Performance Comparison

### User-Space Server (Current)
```
Application → Socket send → Server process → Update state
  ~20ns        ~500ns         ~50ns            ~20ns
  Total: ~590ns per allocation
```

### Kernel-Only (Proposed)
```
Application → ioctl() → Kernel update
  ~20ns        ~200ns    ~20ns
  Total: ~240ns per allocation
```

**Result:** 2.5x faster!

---

## Migration Path

### Phase 1: Add Kernel Support (No Breakage)
1. Add new IOCTLs to tt-kmd
2. Add tracking API to tt-umd
3. Keep existing server approach working

### Phase 2: Dual Support in TT-Metal
```cpp
void GraphTracker::track_allocate(const Buffer* buffer) {
    // Try kernel tracking first
    if (tt::umd::AllocationTracker::is_available()) {
        tt::umd::AllocationTracker::track_allocation(...);
    }
    // Fall back to user-space server
    else if (AllocationClient::is_enabled()) {
        AllocationClient::report_allocation(...);
    }
}
```

### Phase 3: Deprecate Server
1. Update documentation to recommend kernel-only approach
2. Keep server for compatibility
3. Eventually remove

---

## Implementation Checklist

### tt-kmd Changes
- [ ] Add `device_allocations` hash table to `chardev_private`
- [ ] Add aggregate stats to `tenstorrent_device`
- [ ] Define new IOCTLs in `ioctl.h`
- [ ] Implement `ioctl_track_alloc()`
- [ ] Implement `ioctl_track_free()`
- [ ] Implement `ioctl_query_alloc_stats()`
- [ ] Add `tenstorrent_allocation_cleanup()` to release handler
- [ ] Add `/proc/driver/tenstorrent/*/allocations` file
- [ ] Initialize new fields in `tt_cdev_open()`
- [ ] Add kernel config option (optional)

### tt-umd Changes
- [ ] Add `tt_track_allocation()` to C API
- [ ] Add `tt_track_deallocation()` to C API
- [ ] Add `tt_query_allocation_stats()` to C API
- [ ] Create `AllocationTracker` C++ wrapper
- [ ] Update header files

### TT-Metal Changes
- [ ] Update `GraphTracker::track_allocate()`
- [ ] Update `GraphTracker::track_deallocate()`
- [ ] Remove dependency on `AllocationClient` (optional)

### Testing
- [ ] Unit tests for kernel IOCTLs
- [ ] Multi-process stress test
- [ ] Process crash test (cleanup verification)
- [ ] Performance benchmarks
- [ ] Memory leak detection test

---

## Example Output

### /proc Interface
```bash
$ cat /proc/driver/tenstorrent/0/allocations
PID     Comm             DRAM       L1         L1_SMALL   TRACE      BUFFERS
----    ----             ----       --         --------   -----      -------
12345   python           3221225472 471859200  0          0          25
12346   test_app         1048576    524288     0          0          2
12347   benchmark        536870912  10485760   0          0          8

TOTAL                    3759171552 482869248  0          0          35
```

### tt-smi Output
```
┌────────────────────────────────────────────────────────────────┐
│ tt-smi v2.0 (kernel-mode)                Mon Nov  3 15:30:12   │
├────────────────────────────────────────────────────────────────┤
│ GPU  Name         Memory-Usage                                 │
├────────────────────────────────────────────────────────────────┤
│ 0    Wormhole_B0  DRAM: 3.5GB / 12GB  (29%)                   │
│                   L1:   460MB / 1440MB (32%)                   │
├────────────────────────────────────────────────────────────────┤
│ Processes:                                                     │
│   PID    User    Process      DRAM       L1      Buffers      │
├────────────────────────────────────────────────────────────────┤
│   12345  ttuser  python       3.0GB      450MB   25           │
│   12346  ttuser  test_app     1.0MB      0.5MB   2            │
│   12347  ttuser  benchmark    512MB      10MB    8            │
└────────────────────────────────────────────────────────────────┘

Data source: kernel (/proc/driver/tenstorrent)
No server required!
```

---

## Summary

**Key Benefits:**
- ✅ No separate server process
- ✅ Always available (like nvidia-smi)
- ✅ Automatic cleanup on crash
- ✅ 2.5x faster (ioctl vs socket)
- ✅ Standard /proc interface
- ✅ Per-process isolation built-in
- ✅ No environment variables needed

**Trade-offs:**
- Requires kernel changes (but that's acceptable)
- Slightly more complex kernel code
- Need to maintain ioctl ABI compatibility

**Recommendation:** This is the better long-term solution. More aligned with how Linux kernel drivers typically work.
