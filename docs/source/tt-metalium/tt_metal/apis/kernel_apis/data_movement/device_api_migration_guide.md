# Device 2.0 Data Movement API Migration Guide

This guide helps developers migrate from legacy data movement APIs to the new Device 2.0 APIs located in `tt_metal/hw/inc/api/`.

## Table of Contents

1. [Overview](#overview)
2. [Header Files](#header-files)
3. [Key Classes](#key-classes)
4. [Migration Patterns](#migration-patterns)
   - [NoC Operations](#noc-operations)
   - [NocOptVals Struct](#nocoptvals-struct)
   - [Circular Buffer Operations](#circular-buffer-operations)
   - [Semaphore Operations](#semaphore-operations)
   - [Memory Access](#memory-access)
   - [Zeroing Memory](#zeroing-memory)
5. [Complete Migration Examples](#complete-migration-examples)
6. [Troubleshooting](#troubleshooting)

---

## Overview

The Device 2.0 APIs provide a more object-oriented, type-safe interface for data movement operations on Tenstorrent hardware. Key benefits include:

- **Type Safety**: Template-based traits system prevents common errors at compile time
- **Cleaner Abstractions**: Object-oriented wrappers around raw addresses and operations
- **Unified Interface**: Consistent API patterns across different data sources/destinations
- **Better Encapsulation**: State management within class instances

## Header Files

Include the following headers based on your needs:

```cpp
#include "api/dataflow/noc.h"        // Core NoC operations
#include "api/dataflow/circular_buffer.h" // CircularBuffer wrapper
#include "api/core_local_mem.h"  // Safe L1 memory pointers
#include "api/dataflow/endpoints.h"       // Unicast/Multicast/AllocatorBank endpoints
#include "api/dataflow/noc_semaphore.h"   // Semaphore synchronization
#include "api/tensor/noc_traits.h"          // TensorAccessor traits
#include "api/lock.h"                    // RAII lock utilities
```

## Key Classes

### `Noc`

The central class for NoC operations. Wraps a Noc index and provides methods for async reads, writes, multicasts, and barriers.

```cpp
Noc noc;           // Uses default noc_index
Noc noc1(1);       // Explicitly use NoC 1
```

### `CircularBuffer`

Provides circular buffer operations.

```cpp
CircularBuffer cb(cb_id);
cb.reserve_back(num_pages);
cb.push_back(num_pages);
cb.wait_front(num_pages);
cb.pop_front(num_pages);
```

### `CoreLocalMem<T>`

Provides a safe zero overhead way to access a given type in L1 memory.

```cpp
CoreLocalMem<uint32_t> mem(address);
mem[0] = value;              // Array-style access
auto val = *mem;             // Dereference
mem++;                       // Pointer arithmetic
auto addr = mem.get_address();
```

### `Semaphore`

Provides a semaphore for synchronization.

```cpp
Semaphore<> sem(semaphore_id);
sem.up(value);               // Local increment
sem.down(value);             // Blocking decrement
sem.wait(value);             // Wait for exact value
sem.wait_min(value);         // Wait for minimum value
```

### Endpoints

Endpoints are used as sources or destinations for the `Noc` interface. Depending on which endpoint is provided, additional arguments can be passed in to `src_args_t` or `dst_args_t` of each `Noc` action to specify additional metadata such as offset or size.

- `UnicastEndpoint` - For unicast NoC addresses
- `MulticastEndpoint` - For multicast NoC addresses
- `AllocatorBank<AllocatorBankType>` - For DRAM/L1 bank addressing

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

**New API:**
```cpp
Noc noc;
UnicastEndpoint src;
CoreLocalMem<uint32_t> dst(dst_l1_addr);

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

**New API:**
```cpp
Noc noc;
CoreLocalMem<uint32_t> src(src_l1_addr);
UnicastEndpoint dst;

noc.async_write(
    src,                                    // Source
    dst,                                    // Destination
    size_bytes,                             // Transfer size
    {},                                     // Source args
    {.noc_x = x, .noc_y = y, .addr = addr}  // Destination args
);
noc.async_write_barrier();
```

#### Async Read with State (Optimized Repeated Reads)

Use `set_async_read_state` to pre-program the source location, page size, (and optionally the transaction id and/or virtual channel) into hardware registers once, then call `async_read_with_state` in a tight loop — the hardware retains state between calls.

**Legacy API:**
```cpp
uint64_t src_noc_addr = get_noc_addr(noc_x, noc_y, base_addr);
noc_async_read_set_state(src_noc_addr, page_size);
for (uint32_t i = 0; i < num_pages; i++) {
    noc_async_read_with_state(src_base + i * page_size, dst_base + i * page_size, page_size);
}
noc_async_read_barrier();
```

**New API (default options):**
```cpp
Noc noc;
UnicastEndpoint src;
CoreLocalMem<uint32_t> dst(dst_l1_addr);

noc.set_async_read_state(
    src, page_size, {.noc_x = x, .noc_y = y, .addr = base_addr}
);
for (uint32_t i = 0; i < num_pages; i++) {
    noc.async_read_with_state(
        src, dst, page_size,
        {.addr = src_base + i * page_size},
        {.addr = dst_base + i * page_size}
    );
}
noc.async_read_barrier();
```

**New API (custom virtual channel via `NocOptVals`):**
```cpp
noc.set_async_read_state<NocOptions::CUSTOM_VC>(
    src, page_size, {.noc_x = x, .noc_y = y, .addr = base_addr},
    NocOptVals{.vc = my_vc}
);
for (uint32_t i = 0; i < num_pages; i++) {
    noc.async_read_with_state<NocOptions::CUSTOM_VC>(
        src, dst, page_size,
        {.addr = src_base + i * page_size},
        {.addr = dst_base + i * page_size},
        NocOptVals{.vc = my_vc}
    );
}
noc.async_read_barrier();
```

**New API (transaction ID via `NocOptVals`):**

```cpp
noc.set_async_read_state<NocOptions::TXN_ID>(
    src, page_size, {.noc_x = x, .noc_y = y, .addr = base_addr},
    NocOptVals{.trid = curr_trid}
);
for (uint32_t i = 0; i < num_pages; i++) {
    noc.async_read_with_state<NocOptions::TXN_ID>(
        src, dst, page_size,
        {.addr = src_base + i * page_size},
        {.addr = dst_base + i * page_size},
        NocOptVals{.trid = curr_trid}
    );
}
noc.async_read_barrier<NocOptions::TXN_ID>({.trid = curr_trid});
```

#### Async Write with State (Optimized Repeated Writes)

Use `set_async_write_state` to program the destination location, page size, and virtual channel into hardware registers once, then call `async_write_with_state` in a tight loop.

**Legacy API:**
```cpp
uint64_t dst_noc_addr = get_noc_addr(noc_x, noc_y, base_addr);
noc_async_write_one_packet_set_state(dst_noc_addr, size_bytes);
for (...) {
    noc_async_write_one_packet_with_state(src_l1_addr, dst_offset);
}
```

**New API (default options):**
```cpp
Noc noc;
UnicastEndpoint dst;
CoreLocalMem<uint32_t> src(src_addr);

noc.set_async_write_state(
    dst, size_bytes, {.noc_x = x, .noc_y = y, .addr = base_addr}
);
for (...) {
    noc.async_write_with_state(
        src, dst, size_bytes, {.offset_bytes = src_offset}, {.addr = dst_offset}
    );
}
```

**New API (custom virtual channel via `NocOptVals`):**
```cpp
noc.set_async_write_state<NocOptions::CUSTOM_VC>(
    dst, size_bytes, {.noc_x = x, .noc_y = y, .addr = base_addr},
    NocOptVals{.vc = my_vc}
);
for (...) {
    noc.async_write_with_state<NocOptions::CUSTOM_VC>(
        src, dst, size_bytes, {.offset_bytes = src_offset}, {.addr = dst_offset},
        NocOptVals{.vc = my_vc}
    );
}
```

#### Multicast Write

**Legacy API:**
```cpp
uint64_t mcast_addr = get_noc_multicast_addr(x_start, y_start, x_end, y_end, l1_addr);
noc_async_write_multicast(src_l1_addr, mcast_addr, size_bytes, num_dests);
```

**New API:**
```cpp
Noc noc;
CoreLocalMem<uint32_t> src(src_l1_addr);
CircularBuffer cb(cb_id);  // Or any destination with mcast traits

noc.async_write_multicast(
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

**New API (Transaction IDs):**
```cpp
Noc noc;
constexpr uint8_t trid = 1;

// Write with transaction ID
noc.async_write<NocOptions::TXN_ID>(
    src, dst, size_bytes, src_args, dst_args,
    NocOptVals{.trid = trid}
);

// Barrier on specific transaction ID
noc.async_write_barrier<NocOptions::TXN_ID>({.trid = trid});
```

---

### NocOptVals Struct

`NocOptVals` is an aggregate struct that carries the runtime values for optional `NocOptions` flags. It is accepted as the last argument of all stateful and trid-aware `Noc` APIs. Fields are only inspected when the matching flag is set in the template `opts` parameter.

```cpp
struct NocOptVals {
    uint32_t vc   = NOC_UNICAST_WRITE_VC;  // used when NocOptions::CUSTOM_VC
    uint32_t trid = 0;                     // used when NocOptions::TXN_ID
};
```

**Usage summary:**

| Scenario | Template `opts` | `NocOptVals` arg |
|---|---|---|
| Default vc, no trid | omit / `NocOptions::DEFAULT` | omit / `{}` |
| Custom vc | `NocOptions::CUSTOM_VC` | `NocOptVals{.vc = v}` |
| Transaction ID barrier/read/write | `NocOptions::TXN_ID` | `NocOptVals{.trid = t}` |
| Custom vc + trid | `NocOptions::TXN_ID \| NocOptions::CUSTOM_VC` | `NocOptVals{.vc = v, .trid = t}` |

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

**New API:**
```cpp
CircularBuffer cb(cb_id);

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

**New API:**
```cpp
Noc noc;
CircularBuffer cb(cb_id);
UnicastEndpoint remote;

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

Use `use<>()` to explicitly select which pointer to use:

```cpp
using CircularBuffer;
using use;

CircularBuffer cb(cb_id);

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

**New API:**
```cpp
Semaphore<> sem(sem_id);
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

**New API:**
```cpp
Noc noc;
Semaphore<> sem(sem_id);
sem.up(noc, noc_x, noc_y, 1);  // Atomic remote increment
```

#### Multicast Semaphore

**Legacy API:**
```cpp
uint64_t mcast_addr = get_noc_multicast_addr(x0, y0, x1, y1, get_semaphore(sem_id));
noc_semaphore_set_multicast(local_sem_addr, mcast_addr, num_dests);
```

**New API:**
```cpp
Noc noc;
Semaphore<> sem(sem_id);
sem.set_multicast<NocOptions::DEFAULT>(
    noc, x0, y0, x1, y1, num_dests
);

// Include source in multicast:
sem.set_multicast<NocOptions::MCAST_INCL_SRC>(
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

**New API:**
```cpp
CoreLocalMem<uint32_t> mem(address);
uint32_t value = mem[0];    // Bounds-checked in debug mode
mem++;                       // Type-safe pointer arithmetic
```

#### Struct Access

**New API:**
```cpp
struct MyStruct {
    uint32_t field1;
    uint64_t field2;
};

CoreLocalMem<MyStruct> struct_mem(address);
struct_mem->field1 = 42;
struct_mem->field2 = 100;
```

#### Pointer Arithmetic

```cpp
CoreLocalMem<uint32_t> mem(base_addr);

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

### Zeroing Memory

#### Zero a local-L1 buffer

**Legacy API** (manual loopback from the hardware zeros region):
```cpp
constexpr uint32_t n = bytes_to_zero / MEM_ZEROS_SIZE;
uint64_t zeros = get_noc_addr(my_x[noc_index], my_y[noc_index], MEM_ZEROS_BASE);
for (uint32_t i = 0; i < n; ++i) {
    noc_async_read(zeros, write_addr + i * MEM_ZEROS_SIZE, MEM_ZEROS_SIZE);
}
noc_async_read_barrier();
```

**New API:**
```cpp
Noc noc;
DataflowBuffer dfb(dfb_id);                // or CircularBuffer
dfb.reserve_back(1);
noc.async_write_zeros(
    dfb,                                   // Destination (DataflowBuffer or CircularBuffer)
    size_bytes,                            // Number of bytes to zero
    {.offset_bytes = 0}                    // Offset within the destination entry (optional; default {})
);
noc.write_zeros_l1_barrier();              // wait for completion before consuming
dfb.push_back(1);
```

`async_write_zeros(dst, size_bytes, {.offset_bytes = off})` zeroes `size_bytes` starting at `dst.get_write_ptr() + off` (`off` defaults to `0`). `dst` must be a `CircularBuffer` or `DataflowBuffer`.

Zeroing a CB/DFB entry is the common case: legacy kernels obtained the zero target's address from a CB (e.g. `get_write_ptr(cb_id)`), so the migrated form zeroes that same CB/DFB entry directly.

#### Zero pages of a DRAM tensor

The DRAM overload streams zeros from a pre-zeroed L1 source, read from its **front (read) pointer**. A CB/DFB is required to source the zeros (the reserved L1 zeros region is reclaimed on Quasar). **Reuse a `DataflowBuffer` (or `CircularBuffer`) the kernel already has** — do not allocate one solely for this if possible, since CB/DFB consumes Quasar tile counter resources. Zero one entry, `push_back`/`wait_front` it to the front before streaming, then `pop_front` to release:
```cpp
Noc noc;
DataflowBuffer dfb(dfb_id);              // a DFB the kernel already owns

// 1. Pre-zero the DFB entry once
dfb.reserve_back(1);
noc.async_write_zeros(dfb, zero_bytes);
noc.write_zeros_l1_barrier();
dfb.push_back(1);

// 2. Stream zeros to DRAM pages from the (front of the) DFB
dfb.wait_front(1);                       // the zeroed entry is now the front
for (uint32_t p = page_start; p < page_end; ++p) {
    noc.async_write_zeros(
        dram_accessor,                   // Destination DRAM tensor accessor
        page_size,                       // Number of bytes to zero (per page)
        {.page_id = p},                  // Destination page args (page_id, optional offset_bytes)
        dfb                              // Pre-zeroed L1 source of zeros
    );
}
noc.write_zeros_dram_barrier();
dfb.pop_front(1);                        // release; the buffer is left as it was
```

**Each call zeroes within a single page:** `offset_bytes + size_bytes` must not exceed the tensor's aligned page size. Zero a multi-page region by looping over `page_id`.

Any buffer the kernel already owns can be reused (i.e. it needn't be one dedicated to zeroing). The pre-zeroed prefix must cover at least `min(page_size, NOC_MAX_BURST_SIZE)` bytes. This overload pairs with `write_zeros_dram_barrier()`.

#### Barriers and the Quasar command-buffer contract

`async_write_zeros` is asynchronous; pair each overload with its barrier:

| Overload | Barrier |
|---|---|
| L1 | `noc.write_zeros_l1_barrier()` |
| DRAM | `noc.write_zeros_dram_barrier()` |

**Quasar contract:** the L1 zero borrows the overlay write command buffer (cmd buffer 0) and switches it to iDMA "zero mode"; it is restored to normal write mode only by `write_zeros_l1_barrier()`. **Do not issue any other NoC write (`noc.async_write` / `noc_async_write`, which also use cmd buffer 0) on the same RISC between `async_write_zeros` and `write_zeros_l1_barrier()`**: such a write would execute in zero mode and silently write zeros instead of its data. The rule is **zero → barrier → reuse**. NoC *reads* use a separate command buffer and are unaffected, so reads may be interleaved. On Wormhole/Blackhole there is no command-buffer borrow (the zero is a read loopback and the barrier is a plain read barrier).

For performance, issue one barrier after a batch of zeros rather than per-call, subject to the no-intervening-write rule above on Quasar.

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
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    uint32_t src_addr = get_arg_val<uint32_t>(0);
    constexpr uint32_t cb_id = 0;
    constexpr auto tensor_args = TensorAccessorArgs<0>();

    uint32_t tile_size = get_tile_size(cb_id);
    const auto accessor = TensorAccessor(tensor_args, src_addr, tile_size);

    Noc noc;
    CircularBuffer cb(cb_id);

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
#include "api/dataflow/noc.h"
#include "api/core_local_mem.h"
#include "api/dataflow/endpoints.h"

void kernel_main() {
    uint32_t src_addr = get_arg_val<uint32_t>(0);
    uint32_t neighbor_x = get_arg_val<uint32_t>(1);
    uint32_t neighbor_y = get_arg_val<uint32_t>(2);
    uint32_t num_bytes = get_arg_val<uint32_t>(3);

    Noc noc;
    CoreLocalMem<uint32_t> mem(src_addr);
    UnicastEndpoint remote;

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
   - Include the appropriate header (e.g., `api/tensor/noc_traits.h` for `TensorAccessor`)

2. **"CircularBuffer without mcast range can only be used as L1 source"**
   - CircularBuffers require explicit mcast range for multicast destinations
   - Use `dst_args_mcast_type` with proper coordinates

3. **"CoreLocalMem can only be used as local L1 source/dest"**
   - `CoreLocalMem` represents a buffer in local memory. Therefore, providing it as a destination for a Noc async write or source for Noc async read is invalid.
   - Use `UnicastEndpoint` for remote memory access
