# CCL Ring Buffer Mismatch

## Description

In Tenstorrent's Collective Communication Library (CCL), operations like all-gather
use ring topologies where data is passed between devices via ethernet data movers (EDMs).
Each EDM channel has a set of buffers, semaphores, and configuration parameters that must
be consistent between the host-side setup and the device-side kernel arguments.

A common class of bugs arises when buffer counts, sizes, or addresses are computed on the
host but passed inconsistently to device kernels — or when sender and receiver sides of a
channel disagree on configuration.

## What to Look For

1. **num_buffers_per_channel mismatch**: The number of buffers per EDM channel is set
   during `EriscDatamoverBuilder` construction and passed as a compile-time argument to
   erisc kernels. Check that the value used in `add_sender_channel()` /
   `add_receiver_channel()` matches what the kernel receives.

2. **Buffer address / semaphore count mismatch**: `local_buffer_addresses` and
   `local_semaphore_addresses` vectors must have the same size. When channels are added
   or iterated, ensure indexing stays consistent.

3. **eth_buffer_size_bytes drift**: Buffer size is computed via
   `EriscDatamoverConfig::compute_buffer_size()` using `num_edm_channels`,
   `num_buffers_per_channel`, and `page_size`. If any of these inputs change after the
   initial computation without recomputing the buffer size, the value becomes stale.

4. **Ring size vs. device count**: `get_topological_dimension()` computes ring size from
   device coordinates. If this value is used inconsistently (e.g., one place uses cluster
   axis size, another uses a hardcoded device count), neighbor lookups will be wrong.

5. **Sender/receiver channel count asymmetry**: In a ring, each device is both a sender
   and receiver. The number of sender channels must match the number of receiver channels
   on the paired device.

## Bad Code Examples

```cpp
// BUG: num_buffers_per_channel is 4 on host but kernel expects 8
auto config = EriscDatamoverConfig(4 /* num_buffers_per_channel */);
// ... later, kernel compile-time args use a different constant
uint32_t ct_args[] = { ..., 8, ... };  // should be 4
```

```cpp
// BUG: buffer addresses has 3 entries but semaphores has 2
std::vector<uint32_t> buf_addrs = {addr0, addr1, addr2};
std::vector<uint32_t> sem_addrs = {sem0, sem1};
// This will cause an out-of-bounds access or silent corruption
builder.add_sender_channel(worker_semaphore, buf_addrs, sem_addrs);
```

```cpp
// BUG: ring_size computed from one axis but neighbor lookup uses another
auto ring_size = cyclic_order.size();  // e.g., 8
// ... but later:
auto neighbor = get_physical_neighbor(..., ClusterAxis::X);  // only 4 devices on X axis
```

## Good Code Examples

```cpp
// GOOD: num_buffers_per_channel is consistent
constexpr uint32_t num_buffers = 4;
auto config = EriscDatamoverConfig(num_buffers);
// Kernel args use the same constant
uint32_t ct_args[] = { ..., num_buffers, ... };
```

```cpp
// GOOD: same size for addresses and semaphores
TT_ASSERT(local_buffer_addresses.size() == local_semaphore_addresses.size());
builder.add_sender_channel(worker_semaphore, local_buffer_addresses, local_semaphore_addresses);
```

```cpp
// GOOD: ring_size derived from the same topology used for neighbor lookup
auto ring_size = get_topological_dimension(devices, axis);
auto neighbor = get_physical_neighbor(coord, axis, ring_size);
```
