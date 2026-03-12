# §7.3 — Kernel Internals

This section covers the internal machinery that runs during every CCL operation: the EDM (Ethernet Data Mover) kernel running on ERISC cores, how Tensix worker kernels feed data into EDM via NOC, program caching behavior, trace mode constraints and benefits, and L1 memory budgeting.

---

## EDM Kernel Deep Dive

The EDM is the firmware running on each ERISC core that manages Ethernet transfers on behalf of CCL worker kernels. Its source lives in `ttnn/cpp/ttnn/operations/ccl/kernels/edm/erisc_datamover.cpp` and the supporting type system in `erisc_async_datamover.hpp`.

### ERISC core independence

ERISC cores run their own RISC-V firmware independently of Tensix cores. When a CCL program is dispatched, the Metal runtime loads the `erisc_datamover.cpp` kernel onto ERISC cores in addition to reader/writer kernels on Tensix cores. The ERISC kernel enters a cooperative multi-tasking event loop that alternates between:

- Checking whether Tensix workers have filled a sender buffer (waiting for worker ack via L1 semaphore).
- Sending filled buffers over Ethernet to the remote device.
- Receiving buffers from Ethernet and notifying local Tensix workers.
- Calling `run_routing()` when idle to yield to fabric routing tasks.

The event loop continues until all channels report `STATE::DONE`.

### ChannelBuffer state machine

Each active ERISC channel is represented by a `ChannelBuffer` object (from `erisc_async_datamover.hpp`). Its `STATE` enum:

```
STATE::DONE
STATE::SENDER_SIGNALING_WORKER       — EDM notifies Tensix: buffer slot is free
STATE::SENDER_WAITING_FOR_WORKER     — EDM waits for Tensix to fill the buffer
STATE::SENDER_READY_FOR_ETH_TRANSFER — Tensix has filled; EDM is ready to send
STATE::SENDER_WAITING_FOR_ETH        — Data sent; EDM waits for remote ACK
STATE::RECEIVER_WAITING_FOR_ETH      — EDM waits for incoming Ethernet data
STATE::RECEIVER_SIGNALING_WORKER     — Data arrived; EDM notifies Tensix
STATE::RECEIVER_WAITING_FOR_WORKER   — EDM waits for Tensix to consume from buffer
```

The state machine for a sender channel:

```
SENDER_SIGNALING_WORKER
      │ (signal Tensix: slot available)
      ▼
SENDER_WAITING_FOR_WORKER
      │ (Tensix fills buffer, increments EDM semaphore)
      ▼
SENDER_READY_FOR_ETH_TRANSFER
      │ (eth_send_bytes)
      ▼
SENDER_WAITING_FOR_ETH
      │ (remote receiver sends first-level ACK)
      ▼
SENDER_SIGNALING_WORKER (loop) or DONE (if all messages sent)
```

And the receiver state machine:

```
RECEIVER_WAITING_FOR_ETH
      │ (eth_wait_for_bytes / check_bytes_arrived)
      ▼
RECEIVER_SIGNALING_WORKER
      │ (increment Tensix worker semaphore via NOC)
      ▼
RECEIVER_WAITING_FOR_WORKER
      │ (Tensix consumes buffer, increments EDM semaphore)
      ▼
RECEIVER_WAITING_FOR_ETH (loop) or DONE (if all messages received)
```

### Credit-based flow control

The EDM uses a credit-based system implemented through L1 semaphores to prevent buffer overrun:

- Each channel has a dedicated L1 semaphore (`local_semaphore_address` in `ChannelBuffer`).
- When the sender EDM signals a free slot, it posts to the Tensix worker's semaphore.
- The Tensix worker waits on that semaphore (via `wait_for_empty_write_slot()` in `WorkerToEdmSender`), then fills the buffer and increments the EDM's local semaphore.
- The EDM polls its local semaphore to detect that the worker has finished writing.

This handshake means the EDM never sends a buffer before Tensix has fully committed the payload, and the Tensix worker never overwrites an in-flight buffer.

```
Sender side:

  EDM L1 semaphore         Tensix worker semaphore
       │                           │
       │ post: slot free           │
       │ ──────────────────────────►
       │                           │ wait_for_empty_write_slot()
       │                           │ (fill buffer via noc_async_write)
       │                           │ noc_semaphore_inc(edm_sem, NEXT_MESSAGE_AVAILABLE)
       ◄──────────────────────────  │
       │ poll: worker done         │
       │ (eth_send_bytes)          │
```

### L1 buffer layout

The ERISC core's L1 is divided into fixed regions (from `EriscDatamoverConfig` in `ccl_host_datastructures.hpp`):

```
ERISC L1 layout:
┌────────────────────────────────────────────┐  usable_l1_base_address
│  Handshake region                          │  handshake_location_size = 16 bytes
│  (first-level ACK dedicated source)        │  + edm_receiver_first_level_ack_source = 16 bytes
│  Handshake padding                         │  3 × 16 bytes
├────────────────────────────────────────────┤
│  Semaphore region                          │  semaphore_size = 32 bytes per channel
│  [chan0_sem][chan1_sem]...[chanN_sem]       │
├────────────────────────────────────────────┤
│  Buffer region                             │
│  [chan0_buf0][chan0_buf1]...[chanN_bufM]    │  eth_buffer_size_bytes per slot
│                                            │  hard limit: < 163,000 bytes
└────────────────────────────────────────────┘  usable_l1_base_address + total_l1_buffer_space
```

The helper methods in `EriscDatamoverConfig` compute base addresses:

```cpp
uint32_t get_edm_handshake_address() const;
uint32_t get_buffers_base_address(std::size_t num_edm_channels) const;
static std::size_t get_buffers_region_start_offset(std::size_t num_edm_channels);
uint32_t compute_buffer_size(std::size_t num_edm_channels,
                             std::size_t num_buffers_per_channel = 1,
                             uint32_t page_size = eth_word_size_bytes);
```

`compute_buffer_size` divides the available buffer region equally among all channels. If you increase `num_links`, you add more channels, which reduces the per-channel buffer size. Smaller per-channel buffers mean more Ethernet packet overhead per byte transferred, so there is a crossover point beyond which more links hurts throughput. In practice this is rarely reached before the physical link count limit.

### num_buffers_per_channel

`EriscDatamoverBuilder` accepts `num_buffers_per_channel` (default 1). With `num_buffers_per_channel=2`, each channel has two buffer slots, enabling a producer-consumer pipeline where the EDM can be loading buffer slot 1 while Tensix is writing buffer slot 0. This reduces the stall between consecutive Ethernet sends. `chip_id` is encoded at `arg[9]` and `num_buffers_per_channel` at `arg[8]` of the compile-time arg vector (see table above).

### Termination modes

The EDM has two termination modes (`EriscDataMoverTerminationMode`):

- `MESSAGE_COUNT_REACHED` (default): the EDM runs until it has processed exactly `num_eth_messages_to_forward` messages per channel. This is the common case for fixed-shape CCL ops.
- `WORKER_INITIATED`: the Tensix worker explicitly sends `TERMINATE_IMMEDIATELY` via `WorkerToEdmSender::close()`. This is used for dynamic-dispatch ops where the message count is not known at compile time.

### Buffer sharing modes

`EriscDataMoverBufferSharingMode` controls whether multiple Tensix workers share a single EDM channel buffer:

- `NOT_SHARED`: each worker has a dedicated channel. Simplest; no round-robin needed.
- `ROUND_ROBIN`: multiple workers take turns accessing one channel. Reduces L1 usage at the cost of serialization between workers.
- `SHARED`: all workers share one channel simultaneously (only valid when workers are synchronized).
- `ROUND_ROBIN_AND_SHARED`: combination mode.

Most CCL ops use `NOT_SHARED`. `ROUND_ROBIN` is used by ops with many workers that would otherwise exhaust the channel count.

### Compile-time arguments to the EDM kernel

The `get_compile_time_args()` method of `EriscDatamoverBuilder` encodes all static configuration:

```cpp
{
    enable_sender ? 1 : 0,          // arg[0]: enable_sender_side
    enable_receiver ? 1 : 0,        // arg[1]: enable_receiver_side
    num_senders,                     // arg[2]
    num_receivers,                   // arg[3]
    buffer_sharing_mode,             // arg[4]: EriscDataMoverBufferSharingMode
    termination_mode,                // arg[5]: EriscDataMoverTerminationMode
    1,                               // arg[6]: reserved
    is_handshake_sender,             // arg[7]
    num_buffers_per_channel,         // arg[8]
    chip_id                          // arg[9]
}
```

Changes to any of these values (e.g., changing `num_links` from 1 to 2) require a full program recompilation because they are `constexpr` values inside the kernel.

### Runtime arguments to the EDM kernel

`get_runtime_args()` produces the per-invocation argument vector:

```
handshake_addr
receiver_channels_offset
  for each receiver channel:
    buffer_base_address
    num_messages_to_forward
    channel_size_bytes
    semaphore_base_address
    worker_semaphore_id
    num_workers
    worker_coords[0..N-1]   (WorkerXY packed as uint32)
sender_channels_offset
  for each sender channel:
    (same layout as receiver)
```

Runtime args can be updated between iterations without recompilation, which is the foundation for efficient program caching.

---

## NOC Transfers

Within a chip, CCL worker kernels move data between DRAM, L1, and EDM buffers using NOC (Network-on-Chip) async transfers.

### Worker → EDM (sender path)

From `worker_edm_utils.hpp`, the `fetch_chunk` function shows the intra-chip read pattern:

```cpp
FORCE_INLINE void fetch_chunk(
    const uint32_t& cb_id, const uint32_t& num_pages,
    const uint32_t& page_size, uint64_t remote_l1_read_addr) {
    cb_reserve_back(cb_id, num_pages);
    uint32_t l1_write_addr = get_write_ptr(cb_id);
    noc_async_read(remote_l1_read_addr, l1_write_addr, page_size * num_pages);
    noc_async_read_barrier();
    cb_push_back(cb_id, num_pages);
}
```

Key points:
- `noc_async_read` issues a non-blocking NOC read request.
- `noc_async_read_barrier()` waits for all outstanding NOC reads to complete before pushing to the CB.
- The Circular Buffer (CB) is the handoff point between the reader kernel (Tensix) and the EDM sender path.

### EDM → Worker (receiver path)

`WorkerToEdmReader::fetch_payload_blocking` (from `worker_edm_adapters.hpp`):

```cpp
FORCE_INLINE void fetch_payload_blocking(
    uint32_t cb_id, uint32_t num_pages, uint32_t page_size, bool last_message) {
    uint64_t buffer_address =
        this->edm_buffer_addr + (this->buffer_index * (this->buffer_size_bytes + sizeof(eth_channel_sync_t)));
    fetch_chunk(cb_id, num_pages, page_size, buffer_address);
    // Signal EDM that we consumed the buffer
    if (!last_message) {
        noc_semaphore_inc(edm_semaphore_addr, NEXT_MESSAGE_AVAILABLE);
    }
    this->buffer_index = (this->buffer_index + 1) % this->num_buffers_per_channel;
}
```

The buffer index advances modulo `num_buffers_per_channel`, cycling through the slots and implementing the pipelined double-buffering when `num_buffers_per_channel > 1`.

### noc_semaphore_inc vs noc_async_write

`noc_semaphore_inc` is a specialized NOC atomic increment to a remote L1 address. It is cheaper than a full `noc_async_write` because:
- It uses the NOC atomic path (smaller message).
- The increment is applied atomically — no read-modify-write race.

All cross-core signaling in CCL uses `noc_semaphore_inc`. The corresponding wait is `noc_semaphore_wait(addr, expected_value)`, which spins in L1 until the value reaches `expected_value`.

### OpSignaler: multi-worker synchronization

When multiple Tensix worker cores need to collectively signal a downstream op (e.g., all workers in the CCL sub-grid signaling the start of a matmul), the `OpSignaler` struct from `worker_sync_utils.hpp` coordinates them through a master-slave pattern:

```
Workers:  [slave_0] [slave_1] [master] [slave_3]
                         ↓
                 slave_sync_master()  →  slave increments master's semaphore
                         ↓
                 master_sync_slaves() →  master notifies op core via noc_semaphore_inc
                                      →  master clears slave semaphores via noc_semaphore_inc
```

The master worker (`curr_worker_is_master`) waits until `num_workers_to_sync - 1` slaves have incremented the master's local semaphore, then signals the fused op core, then resets all slave semaphores. This pattern is used in `FusedOpSignaler::MULTI` mode (see Ch5).

---

## Program Caching

### How caching works

tt-metal caches compiled Metal programs keyed on the `operation_attributes_t` struct of each device operation. The `attribute_names` / `attribute_values()` accessors (present on every device operation struct) produce the cache key. When you call `ttnn.all_gather(...)` a second time with the same attributes, the cached program is used and only runtime arguments are updated.

From the `AllToAllDispatchDeviceOperation::operation_attributes_t` example:

```cpp
struct operation_attributes_t {
    const CoreRangeSet worker_core_range_set;
    const MemoryConfig output_mem_config;
    const std::optional<uint32_t> axis;
    const uint32_t num_links;
    const tt::tt_fabric::Topology topology;
    const AllToAllTransferType impl;
    const uint32_t output_concat_dim;
    static constexpr auto attribute_names = std::forward_as_tuple(
        "worker_core_range_set", "output_mem_config", "axis", "num_links", "topology", "impl", "output_concat_dim");
```

Any field in `attribute_names` that changes between calls causes a cache miss and recompilation.

### What triggers recompilation

| Change | Cache behavior |
|--------|---------------|
| Different `dim` / `axis` | Miss — recompile |
| Different `num_links` | Miss — recompile (compile-time arg to EDM kernel) |
| Different `topology` | Miss — recompile |
| Different `memory_config` | Miss — recompile |
| Different tensor shape | Miss — recompile (affects kernel tiling) |
| Different tensor dtype | Miss — recompile |
| Different tensor device | Hit if same attributes, update runtime args |
| Different `GlobalSemaphore` value | Hit — semaphore address is a runtime arg |
| Different data pointer (same shape) | Hit — buffer address is a runtime arg |

The expensive items are shape changes and `num_links` changes. Both trigger a new kernel compile cycle.

### Keeping shapes constant

For inference loops, keep all tensor shapes constant across iterations. If your model has variable sequence lengths, pad to the maximum length before the CCL op. The overhead of extra padding pages in the CCL is almost always less than a recompile cycle.

If you need multiple CCL configurations (e.g., a prefill shape and a decode shape), create them both in a warmup phase so the compiled programs are cached before the timed inference run.

### validate_on_program_cache_hit

Each device operation implements two validation hooks:
- `validate_on_program_cache_miss` — full validation including shape checks, buffer allocation, etc.
- `validate_on_program_cache_hit` — lightweight validation of only the runtime-changeable aspects (e.g., checking that the tensor's memory config matches).

On a cache hit, the second validator runs and only runtime args are updated (`override_runtime_arguments`), making re-invocation very cheap — typically a few microseconds of host overhead.

---

## Trace Mode

### What trace mode is

Metal trace mode records a sequence of program dispatches into a `CommandTrace` object on the host. The trace is then replayed from device memory, eliminating host-side dispatch overhead and achieving the minimum possible latency between operations.

Trace mode is critical for inference workloads where the model runs in a tight decode loop and host overhead is measurable.

### Persistent output buffers

Async CCL ops in trace mode require **persistent output tensors**: pre-allocated tensors whose L1/DRAM buffers are fixed at the addresses recorded in the trace. If you allocate a new output tensor each iteration, the buffer address changes between the trace capture and replay, causing the replayed trace to write to a stale address.

Pattern for persistent buffers:

```python
# Before trace capture: allocate persistent output
persistent_output = ttnn.allocate_tensor_on_device(
    output_shape, dtype, layout, device,
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
)

# Capture trace
with ttnn.capture_trace(device, trace_id=0):
    result = ttnn.all_gather_async(
        input_tensor,
        dim=1,
        multi_device_global_semaphore=ccl_sem,
        persistent_output_buffer=persistent_output,
        subdevice_id=ccl_sub_id,
    )

# Replay loop
for step in range(num_steps):
    # Update input (same address, new data)
    ttnn.copy_host_to_device_tensor(new_data, input_tensor)
    ttnn.execute_trace(device, trace_id=0)
    ttnn.synchronize_device(device)
    ttnn.reset_global_semaphore_value(ccl_sem, 0)
```

### Constraints in trace mode

For the full set of trace mode constraints with async CCL, see [Ch4 §4.3 — Pattern 5: Traced Overlap](../ch4_async_overlap/overlap_patterns.md#pattern-5-traced-overlap). Constraint 4 (semaphore reset outside the trace) is most frequently violated — reset in the replay loop, not inside the capture block.

### Trace mode and SubDevices

SubDevices interact well with trace mode because each SubDevice's command queue is independent. You can capture a trace that dispatches to SubDevice 0 (CCL) and SubDevice 1 (compute) in interleaved fashion, and the replay will correctly overlap them as recorded.

---

## L1 Memory Budget

### Sources of L1 pressure

For a CCL op on Tensix, L1 is consumed by:

1. **Circular Buffers (CBs):** each worker allocates CBs for the input data, output data, and any intermediate data. CB sizes are set in the program factory's `create_program` step based on tensor geometry.

2. **Kernel argument storage:** runtime args are stored in L1 before being passed to the kernel. Larger `num_links` or more worker coords increase arg size.

3. **GlobalSemaphore allocations:** each semaphore occupies 32 bytes × the number of cores in its `CoreRangeSet`. For a 64-core SubDevice, each semaphore uses 64 × 32 = 2 KB.

4. **Output tensor slices:** if the output is L1-backed (`MemoryConfig` with `L1`), the entire output shard must fit in L1.

### Estimating CB size for AllGather

For an AllGather of an `[N, S, H]` tensor across `D` devices with `num_links=L`, each worker processes a shard of size `S × H / num_workers`. The CB must hold at least one page (tile) of data. A tile is 32 × 32 elements = 1024 elements. For bfloat16 that is 2048 bytes per tile.

The program factory automatically sizes CBs to the page granularity of the tensor. You generally do not need to size CBs manually, but if you are packing multiple operations into the same L1 budget, use:

```
CB_size ≈ num_pages_in_flight × tile_size_bytes
```

where `num_pages_in_flight` is typically 2–4 (double-buffering).

### AllToAllTransferType: FullPacket vs PageByPage

The dispatch device operation (Ch6) selects between two transfer modes based on L1 availability via `detail::get_cb_sizes()`:

```cpp
std::pair<std::array<uint32_t, 6>, std::array<uint32_t, 6>> get_cb_sizes(
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& indices_tensor,
    const ttnn::Tensor& mapping_tensor,
    uint32_t num_links,
    std::optional<uint32_t> axis);
```

- `FullPacket`: the intermediate buffer holds an entire transferred chunk in L1 before writing to output. Highest throughput; requires more L1.
- `PageByPage`: each page is written directly to the output buffer upon receipt. Lower L1 requirement; slightly higher NOC overhead per page.

The automatic selection is conservative — it picks `PageByPage` whenever the estimated `FullPacket` CB would exceed a threshold. If your workload has generous L1, you can force `FullPacket` by ensuring your tensors are small enough that `get_cb_sizes` stays below the threshold. You cannot override this selection from Python; it is an internal decision of the device operation.

### ERISC L1 is separate

ERISC L1 (used by EDM channels) does not share the address space with Tensix L1. `total_l1_buffer_space` in `EriscDatamoverConfig` refers entirely to ERISC-side L1. Increasing `num_links` consumes ERISC L1 but does not reduce the Tensix L1 available for CBs or output tensors.

### Practical L1 budget checklist

Before deploying a CCL-heavy model:

- [ ] Check that output tensors fit in L1 if using `L1` memory config; otherwise use `DRAM` with async CCL for overlap.
- [ ] Estimate GlobalSemaphore cost: `num_sems × num_cores × 32 bytes`.
- [ ] Use `num_buffers_per_channel=2` only when L1 is not the bottleneck — it doubles the EDM CB allocation.
- [ ] For fused ops (Ch5), verify that `intermediate_tensor` and `intermediate_packet_buffer` do not alias each other or the output tensor in L1.
- [ ] Monitor `tt::LogOp` trace logs for "Trimming buffer size" messages, which indicate the per-channel EDM buffer was reduced due to L1 pressure.

---

*Back to [Chapter 7 Index](index.md)*
