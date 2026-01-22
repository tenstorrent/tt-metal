# Device 2.0 Data Movement API Migration Guide

This guide helps developers migrate from legacy data movement APIs to the new experimental Device 2.0 APIs located in `tt_metal/hw/inc/experimental/`.

> **Note**: These APIs are experimental and subject to change.

## Table of Contents

1. [Overview](#overview)
2. [Header Files](#header-files)
3. [Key Classes](#key-classes)
4. [Migration Patterns](#migration-patterns)
   - [NoC Operations](#noc-operations)
   - [Circular Buffer Operations](#circular-buffer-operations)
   - [Semaphore Operations](#semaphore-operations)
   - [Memory Access](#memory-access)
5. [Complete Migration Examples](#complete-migration-examples)
6. [Troubleshooting](#troubleshooting)

---

## Overview

The Device 2.0 experimental APIs provide a more object-oriented, type-safe interface for data movement operations on Tenstorrent hardware. Key benefits include:

- **Type Safety**: Template-based traits system prevents common errors at compile time
- **Cleaner Abstractions**: Object-oriented wrappers around raw addresses and operations
- **Unified Interface**: Consistent API patterns across different data sources/destinations
- **Better Encapsulation**: State management within class instances

## Header Files

Include the following headers based on your needs:

```cpp
#include "experimental/noc.h"            // Core NoC operations
#include "experimental/circular_buffer.h" // CircularBuffer wrapper
#include "experimental/core_local_mem.h"  // Safe L1 memory pointers
#include "experimental/endpoints.h"       // Unicast/Multicast/AllocatorBank endpoints
#include "experimental/noc_semaphore.h"   // Semaphore synchronization
#include "experimental/tensor.h"          // TensorAccessor traits
#include "experimental/lock.h"            // RAII lock utilities
```

## Key Classes

### `experimental::Noc`

The central class for NoC operations. Wraps a Noc index and provides methods for async reads, writes, multicasts, and barriers.

```cpp
experimental::Noc noc;           // Uses default noc_index
experimental::Noc noc1(1);       // Explicitly use NoC 1
```

### `experimental::CircularBuffer`

Provides circular buffer operations.

```cpp
experimental::CircularBuffer cb(cb_id);
cb.reserve_back(num_pages);
cb.push_back(num_pages);
cb.wait_front(num_pages);
cb.pop_front(num_pages);
```

### `experimental::CoreLocalMem<T>`

Provides a safe zero overhead way to access a given type in L1 memory.

```cpp
experimental::CoreLocalMem<uint32_t> mem(address);
mem[0] = value;              // Array-style access
auto val = *mem;             // Dereference
mem++;                       // Pointer arithmetic
auto addr = mem.get_address();
```

### `experimental::Semaphore`

Provides a semaphore for synchronization.

```cpp
experimental::Semaphore<> sem(semaphore_id);
sem.up(value);               // Local increment
sem.down(value);             // Blocking decrement
sem.wait(value);             // Wait for exact value
sem.wait_min(value);         // Wait for minimum value
```

### Endpoints

Endpoints are used as sources or destinations for the `Noc` interface. Depending on which endpoint is provided, additional arguments can be passed in to `src_args_t` or `dst_args_t` of each `Noc` action to specify additional metadata such as offset or size.

- `experimental::UnicastEndpoint` - For unicast NoC addresses
- `experimental::MulticastEndpoint` - For multicast NoC addresses
- `experimental::AllocatorBank<AllocatorBankType>` - For DRAM/L1 bank addressing

---

## Migration Patterns

### NoC Operations

#### Async Read

**Legacy API:**
```cpp
uint64_t src_noc_addr = get_noc_addr(noc_x, noc_y, src_addr);
noc_async_read(src_noc_addr, dst_l1_addr, size_bytes);
noc_async_read_barrier();
```

**New Experimental API:**
```cpp
experimental::Noc noc;
experimental::UnicastEndpoint src;
experimental::CoreLocalMem<uint32_t> dst(dst_l1_addr);

noc.async_read(
    src,                                    // Source endpoint
    dst,                                    // Destination
    size_bytes,                             // Transfer size
    {.noc_x = x, .noc_y = y, .addr = addr}, // Source args
    {}                                      // Destination args
);
noc.async_read_barrier();
```

#### Async Write

**Legacy API:**
```cpp
uint64_t dst_noc_addr = get_noc_addr(noc_x, noc_y, dst_addr);
noc_async_write(src_l1_addr, dst_noc_addr, size_bytes);
noc_async_write_barrier();
```

**New Experimental API:**
```cpp
experimental::Noc noc;
experimental::CoreLocalMem<uint32_t> src(src_l1_addr);
experimental::UnicastEndpoint dst;

noc.async_write(
    src,                                    // Source
    dst,                                    // Destination
    size_bytes,                             // Transfer size
    {},                                     // Source args
    {.noc_x = x, .noc_y = y, .addr = addr}  // Destination args
);
noc.async_write_barrier();
```

#### Async Write with State (Optimized Repeated Writes)

**Legacy API:**
```cpp
uint64_t dst_noc_addr = get_noc_addr(noc_x, noc_y, base_addr);
noc_async_write_one_packet_set_state(dst_noc_addr, size_bytes);
for (...) {
    noc_async_write_one_packet_with_state(src_l1_addr, dst_offset);
}
```

**New Experimental API:**
```cpp
experimental::Noc noc;
experimental::UnicastEndpoint dst;
experimental::CoreLocalMem<uint32_t> src(src_addr);

noc.set_async_write_state<Noc::ResponseMode::NON_POSTED>(
    dst, size_bytes, {.noc_x = x, .noc_y = y, .addr = base_addr}
);
for (...) {
    noc.async_write_with_state<Noc::ResponseMode::NON_POSTED>(
        src, dst, size_bytes, {.offset_bytes = src_offset}, {.addr = dst_offset}
    );
}
```

#### Multicast Write

**Legacy API:**
```cpp
uint64_t mcast_addr = get_noc_multicast_addr(x_start, y_start, x_end, y_end, l1_addr);
noc_async_write_multicast(src_l1_addr, mcast_addr, size_bytes, num_dests);
```

**New Experimental API:**
```cpp
experimental::Noc noc;
experimental::CoreLocalMem<uint32_t> src(src_l1_addr);
experimental::CircularBuffer cb(cb_id);  // Or any destination with mcast traits

noc.async_write_multicast<Noc::McastMode::EXCLUDE_SRC>(
    src,
    cb,
    size_bytes,
    num_dests,
    {},  // Source args
    {.noc_x_start = x_start, .noc_y_start = y_start,
     .noc_x_end = x_end, .noc_y_end = y_end, .offset_bytes = 0}
);
```

#### Transaction ID Support

**New Experimental API (Transaction IDs):**
```cpp
experimental::Noc noc;
constexpr uint32_t trid = 0;

// Write with transaction ID
noc.async_write<Noc::TxnIdMode::ENABLED>(
    src, dst, size_bytes, src_args, dst_args,
    NOC_UNICAST_WRITE_VC, trid
);

// Barrier on specific transaction ID
noc.async_write_barrier<Noc::BarrierMode::TXN_ID>(trid);
```

---

### Circular Buffer Operations

#### Basic Operations

**Legacy API:**
```cpp
cb_reserve_back(cb_id, num_tiles);
uint32_t write_ptr = get_write_ptr(cb_id);
// ... write data ...
cb_push_back(cb_id, num_tiles);

cb_wait_front(cb_id, num_tiles);
uint32_t read_ptr = get_read_ptr(cb_id);
// ... read data ...
cb_pop_front(cb_id, num_tiles);
```

**New Experimental API:**
```cpp
experimental::CircularBuffer cb(cb_id);

cb.reserve_back(num_tiles);
uint32_t write_ptr = cb.get_write_ptr();
// ... write data ...
cb.push_back(num_tiles);

cb.wait_front(num_tiles);
uint32_t read_ptr = cb.get_read_ptr();
// ... read data ...
cb.pop_front(num_tiles);
```

#### Using CircularBuffer with Noc

**New Experimental API:**
```cpp
experimental::Noc noc;
experimental::CircularBuffer cb(cb_id);
experimental::UnicastEndpoint remote;

// Read into circular buffer
cb.reserve_back(1);
noc.async_read(
    remote,
    cb,  // CircularBuffer as destination
    tile_size,
    {.noc_x = x, .noc_y = y, .addr = addr},
    {.offset_bytes = 0}
);
noc.async_read_barrier();
cb.push_back(1);
```

#### Selecting Read/Write Pointer for Noc Async Read/Write API

Use `experimental::use<>()` to explicitly select which pointer to use:

```cpp
using experimental::CircularBuffer;
using experimental::use;

experimental::CircularBuffer cb(cb_id);

// Use read pointer explicitly
auto cb_read_view = use<CircularBuffer::AddrSelector::READ_PTR>(cb);

// Use write pointer explicitly
auto cb_write_view = use<CircularBuffer::AddrSelector::WRITE_PTR>(cb);
```

These *pointers* can be passed into the NoC async read and write APIs as sources or destinations.

---

### Semaphore Operations

#### Local Semaphore Operations

**Legacy API:**
```cpp
volatile tt_l1_ptr uint32_t* sem_addr =
    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_id));
noc_semaphore_set(sem_addr, 0);
noc_semaphore_wait(sem_addr, 1);
```

**New Experimental API:**
```cpp
experimental::Semaphore<> sem(sem_id);
sem.set(0);
sem.wait(1);
// Or use sem.wait_min(1) for >= comparison
```

#### Remote Semaphore Operations

**Legacy API:**
```cpp
uint64_t remote_sem_addr = get_noc_addr(noc_x, noc_y, get_semaphore(sem_id));
noc_semaphore_inc(remote_sem_addr, 1);
```

**New Experimental API:**
```cpp
experimental::Noc noc;
experimental::Semaphore<> sem(sem_id);
sem.up(noc, noc_x, noc_y, 1);  // Atomic remote increment
```

#### Multicast Semaphore

**Legacy API:**
```cpp
uint64_t mcast_addr = get_noc_multicast_addr(x0, y0, x1, y1, get_semaphore(sem_id));
noc_semaphore_set_multicast(local_sem_addr, mcast_addr, num_dests);
```

**New Experimental API:**
```cpp
experimental::Noc noc;
experimental::Semaphore<> sem(sem_id);
sem.set_multicast<Noc::McastMode::EXCLUDE_SRC>(
    noc, x0, y0, x1, y1, num_dests
);

// Include source in multicast:
sem.set_multicast<Noc::McastMode::INCLUDE_SRC>(
    noc, x0, y0, x1, y1, num_dests
);
```

---

### Memory Access

#### Safe Local Memory Access

**Legacy API:**
```cpp
volatile uint32_t* data = reinterpret_cast<volatile uint32_t*>(address);
uint32_t value = data[0];
data++;
```

**New Experimental API:**
```cpp
experimental::CoreLocalMem<uint32_t> mem(address);
uint32_t value = mem[0];    // Bounds-checked in debug mode
mem++;                       // Type-safe pointer arithmetic
```

#### Struct Access

**New Experimental API:**
```cpp
struct MyStruct {
    uint32_t field1;
    uint64_t field2;
};

experimental::CoreLocalMem<MyStruct> struct_mem(address);
struct_mem->field1 = 42;
struct_mem->field2 = 100;
```

#### Pointer Arithmetic

```cpp
experimental::CoreLocalMem<uint32_t> mem(base_addr);

// Navigate through memory
auto mid = mem + offset;           // Offset by elements
auto diff = mid - mem;             // Difference in elements
auto addr = mid.get_address();     // Get raw address

// Iteration
for (auto ptr = mem; ptr < end; ++ptr) {
    *ptr = value;
}
```

---

## Complete Migration Examples

### Example 1: Tile Read Kernel

**Legacy Kernel:**
```cpp
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    uint32_t src_addr = get_arg_val<uint32_t>(0);
    constexpr uint32_t cb_id = 0;
    constexpr auto tensor_args = TensorAccessorArgs<0>();

    uint32_t tile_size = get_tile_size(cb_id);
    const auto accessor = TensorAccessor(tensor_args, src_addr, tile_size);

    for (uint32_t tile_id = 0; tile_id < num_tiles; tile_id++) {
        cb_reserve_back(cb_id, 1);
        uint32_t l1_write_addr = get_write_ptr(cb_id);
        noc_async_read_tile(tile_id, accessor, l1_write_addr);
        noc_async_read_barrier();
        cb_push_back(cb_id, 1);
    }
}
```

**Migrated Kernel:**
```cpp
#include "api/dataflow/dataflow_api.h"
#include "experimental/noc.h"
#include "experimental/circular_buffer.h"
#include "experimental/tensor.h"

void kernel_main() {
    uint32_t src_addr = get_arg_val<uint32_t>(0);
    constexpr uint32_t cb_id = 0;
    constexpr auto tensor_args = TensorAccessorArgs<0>();

    uint32_t tile_size = get_tile_size(cb_id);
    const auto accessor = TensorAccessor(tensor_args, src_addr, tile_size);

    experimental::Noc noc;
    experimental::CircularBuffer cb(cb_id);

    for (uint32_t tile_id = 0; tile_id < num_tiles; tile_id++) {
        cb.reserve_back(1);
        noc.async_read(
            accessor,
            cb,
            tile_size,
            {.page_id = tile_id},
            {.offset_bytes = 0}
        );
        noc.async_read_barrier();
        cb.push_back(1);
    }
}
```

### Example 2: Core-to-Core Communication

**Legacy Kernel:**
```cpp
void kernel_main() {
    uint32_t src_addr = get_arg_val<uint32_t>(0);
    uint32_t neighbor_x = get_arg_val<uint32_t>(1);
    uint32_t neighbor_y = get_arg_val<uint32_t>(2);
    uint32_t num_bytes = get_arg_val<uint32_t>(3);

    // Write to neighbor
    uint64_t dst_noc_addr = get_noc_addr(neighbor_x, neighbor_y, src_addr);
    noc_async_write(src_addr, dst_noc_addr, num_bytes);
    noc_async_write_barrier();

    // Read from neighbor
    noc_async_read(dst_noc_addr, src_addr, num_bytes);
    noc_async_read_barrier();
}
```

**Migrated Kernel:**
```cpp
#include "experimental/noc.h"
#include "experimental/core_local_mem.h"
#include "experimental/endpoints.h"

void kernel_main() {
    uint32_t src_addr = get_arg_val<uint32_t>(0);
    uint32_t neighbor_x = get_arg_val<uint32_t>(1);
    uint32_t neighbor_y = get_arg_val<uint32_t>(2);
    uint32_t num_bytes = get_arg_val<uint32_t>(3);

    experimental::Noc noc;
    experimental::CoreLocalMem<uint32_t> mem(src_addr);
    experimental::UnicastEndpoint remote;

    // Write to neighbor
    noc.async_write(
        mem,
        remote,
        num_bytes,
        {},
        {.noc_x = neighbor_x, .noc_y = neighbor_y, .addr = src_addr}
    );
    noc.async_write_barrier();

    // Read from neighbor
    noc.async_read(
        remote,
        mem,
        num_bytes,
        {.noc_x = neighbor_x, .noc_y = neighbor_y, .addr = src_addr},
        {}
    );
    noc.async_read_barrier();
}
```

---

## Troubleshooting

### Common Issues

1. **Static assertion failure: "NoC transactions are not supported for this type"**
   - Ensure your type has a `noc_traits_t` specialization
   - Include the appropriate header (e.g., `experimental/tensor.h` for `TensorAccessor`)

2. **"CircularBuffer without mcast range can only be used as L1 source"**
   - CircularBuffers require explicit mcast range for multicast destinations
   - Use `dst_args_mcast_type` with proper coordinates

3. **"CoreLocalMem can only be used as local L1 source/dest"**
   - `CoreLocalMem` represents a buffer in local memory. Therefore, providing it as a destination for a Noc async write or source for Noc async read is invalid.
   - Use `UnicastEndpoint` for remote memory access
