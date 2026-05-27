# H2DStreamService

A persistent host-to-device streaming service backed by a fixed device tensor. It pairs an `H2DSocket` per participating mesh coord with a dedicated service core kernel that drains the socket FIFO directly into the backing tensor — no per-call program build, no fast-dispatch queue traffic, no kernel launches on the hot path. One service core kernel per coord is launched at construction and runs for the lifetime of the service.

The socket FIFOs live in PCIe-pinned shared memory. Producers do not need to be the process that constructed the service: another process — typically an inference server driving workload IO — can attach to the exported sockets and stream data into the same backing tensor independently of the process owning the model workload. The owning process sees the data appear in the backing tensor (and, if configured, on worker cores via the metadata multicast and `data_ready_sem`) the same way it would for an in-process write.

## Topology

```
   Producer process(es)
   ────────────────────
   forward_to_tensor / forward_to_tensor_bytes
            │
            ▼
   ┌─────────────────────────────────────────────┐
   │  PCIe-pinned shared-memory FIFO             │  multiple processes can
   │  one socket per participating mesh coord    │  attach to exported sockets
   └────────────────┬────────────────────────────┘
                    │ NoC read (DEVICE_PULL) initiated by service core
                    ▼
   ┌─────────────────────────────────────────────┐
   │  Service core   (one per device, off the    │
   │                  worker grid)               │
   │  ┌───────────────────────────────────────┐  │
   │  │ Persistent service core kernel        │  │
   │  │  • drains FIFO → backing tensor       │  │
   │  │  • multicasts metadata + data-ready   │  │
   │  │  • polls consumed_counter             │  │
   │  └───────────────────────────────────────┘  │
   │  Service-core L1:                           │
   │    socket config + data FIFO                │
   │    termination word                         │
   │    consumed_counter (per coord)             │
   └────────────────┬────────────────────────────┘
                    │ NoC writes
                    ▼
        ┌──────────────────────────┐
        │  Backing tensor (DRAM)   │
        └──────────┬───────────────┘
                   │ NoC reads
                   ▼
   ┌─────────────────────────────────────────────┐
   │  Worker cores    (Config::worker_cores,     │
   │                   uniform across mesh)      │
   │  ┌───────────────────────────────────────┐  │
   │  │ User worker kernel                    │  │
   │  │  • waits on data_ready_sem            │  │
   │  │  • reads backing tensor + metadata    │  │
   │  │  • runs downstream ops                │  │
   │  │  • acks consumed_counter              │  │
   │  └───────────────────────────────────────┘  │
   │  Worker-core L1:                            │
   │    data_ready_sem  (GlobalSemaphore)        │
   │    metadata shard  (HEIGHT_SHARDED L1 buf)  │
   └─────────────────────────────────────────────┘
```

## Data movement model

The service owns one device-resident tensor — the **backing tensor** — whose contents are overwritten on every transfer. The same `Tensor` instance is valid across calls; your consumer op reads from it directly.

Two write paths:
- **Raw bytes** against the un-sharded global shape — the configured `TensorToMesh` mapper splits across the mesh internally.
- **Pre-distributed `Tensor`** — per-coord shards are streamed verbatim, no mapper invocation.

Each service core iteration drains exactly one full tensor's worth of data into the backing tensor. The host call pushes bytes into a FIFO in PCIe-pinned host RAM; the service core kernel reads them over NoC and writes into DRAM.

### Async behavior

- The service workload is dispatched once at construction and stays resident. The fast-dispatch CQ is not touched on the hot path — your worker ops can use the FD CQ freely without contention.
- `forward_to_tensor` is synchronous *from the caller's point of view*: bytes are copied into the FIFO before return, so the source buffer may be reused immediately. It does **not** wait for the device to drain the FIFO.
- Backpressure: if the FIFO is full, the host blocks until the service core drains a slot. No explicit flow control needed.
- `barrier()` waits until the device has ACKed every in-flight host write. Call before reading the backing tensor on the host, or before destruction if writes are still pending.

## Service cores

The service core is a regular Tensix core, but it lives **outside the worker grid** — it's reserved at device init for framework-owned long-running services and is not visible to normal ttnn ops. You never write a kernel for it; the service core kernel is built and dispatched by the service itself.

- L1 on the service core is managed by a separate allocator from the worker-grid `BankManager`. Service-core L1 allocations (socket FIFO, termination word, consumed counter) **do not appear** in your worker-grid L1 budget.
- `get_service_core(coord)` returns a **logical** `CoreCoord`. Your consumer op must convert it to physical NoC coords per device (see [Writing a consumer op](#writing-a-consumer-op-against-the-service)).
- Selection of the service core may differ per device — different devices may pick different physical cores from each device's claimable set.

## Lifecycle

- **Construction** is blocking: builds the device tensor, claims service cores, allocates sockets + buffers, and dispatches the service core workload. Returns once everything is live.
- **Streaming**: the service runs until destroyed. Push as many transfers as you want.
- **Destruction** is blocking and idempotent — it drains in-flight writes, signals termination to the service core kernels, waits for them to exit, and releases all resources. You don't need to do anything manually.

## Worker synchronization

Set `Config::worker_cores` to a `CoreRange` to enable a per-transfer handshake between the service core and your consumer op. The handshake lets your op pick up each transfer as soon as the service core finishes writing it — no host round-trip per iteration. When `worker_cores` is unset, the handshake is disabled entirely (no allocations, no host work).

Three resources are allocated by the service when the handshake is enabled, and exposed to your op through getters:

| Resource | What it is | Where it lives | Getter |
|---|---|---|---|
| `data_ready_sem` | Mesh-wide L1 semaphore on the worker grid. Same L1 address on every (device, worker core). | Worker-core L1 | `get_data_ready_sem_addr()` |
| `consumed_counter` | One `uint32` per coord, on that coord's service core. Workers atomic-inc here to ack. | Service-core L1 | `get_consumed_counter_addr(coord)` |
| `service_core` | Logical core coord of the service core for that coord's device. Workers NoC-write to this core to ack. | — | `get_service_core(coord)` |

`num_workers` is the count of cores in `worker_cores` and must be uniform across the mesh.

Per-iteration protocol:

1. The service core completes one full transfer into the backing tensor.
2. If metadata is enabled, the service core multicasts the metadata payload and barriers before step 3.
3. The service core multicast-increments `data_ready_sem` by 1 across the worker bounding box (`num_dests = num_workers`).
4. Each worker spins on its local `data_ready_sem` until `> 0`, resets it to 0, reads its slice of the backing tensor, and runs downstream operations on it.
5. Each worker atomic-increments `consumed_counter` at the service core's physical NoC coords.
6. The service core polls `consumed_counter` until `(cur - last_consumed) == num_workers` (exact equality). Only then does it proceed to drain the next transfer — the backing tensor is held stable until every worker has signalled consumption, so the next write cannot overwrite data a worker is still reading.

Invariants:

- Exactly one ack per worker per transfer. Skipping the ack hangs the service core.
- `data_ready_sem` is reset to 0 by the worker. No iteration counter or target value flows host-to-worker.
- `Config::worker_cores` must be uniform across the mesh.

Timeline of a single transfer:

```
   Service core                          Worker cores (each)
   ────────────                          ───────────────────

   drain N tensor pages from FIFO
   write pages → backing tensor (DRAM)
        │
        │  (if metadata enabled)
        │  drain 1 trailing page
        │  multicast first metadata_size_bytes → worker L1
        │
        │  multicast-inc data_ready_sem ──────────┐
        │                                         │
        ▼                                         ▼
   poll consumed_counter            wait for *sem > 0
        ▲                           reset *sem = 0
        │                           read backing tensor slice
        │                           read metadata (volatile L1)
        │                           run downstream compute
        │                           atomic-inc consumed_counter ─┐
        │                                                        │
        └──── (cur - last_consumed) == num_workers ──────────────┘
        │
        ▼
   loop to next transfer
```

## Inline metadata

For per-transfer control data — step IDs, attention masks, position vectors, anything small that varies per iteration — set `Config::metadata_size_bytes > 0`. This requires `Config::worker_cores` to be set. Every `forward_to_tensor` call must then supply exactly `metadata_size_bytes` via the `metadata` kwarg.

**How it travels.** The host appends one trailing socket page per transfer, padded internally to the socket page size — you only see the `metadata_size_bytes` you passed. The service core kernel reads that page over PCIe and multicasts the un-padded bytes to a worker-grid L1 buffer at `get_metadata_addr()`. The address is the same on every (device, worker core).

**Ordering guarantee.** The metadata multicast lands **before** the `data_ready_sem` flip. When your op sees data-ready, both the backing tensor (in DRAM) and the metadata (in L1) are valid.

**Layout is your problem.** The service treats metadata as opaque bytes. Producer and consumer must agree on layout out-of-band (a shared struct, a Python dataclass + `tobytes()`, etc.). The kernel reads at `metadata_addr` as a `volatile tt_l1_ptr` pointer of whatever element type the producer wrote.

**Constraint.** `metadata_size_bytes <= socket_page_size`. Increase `scratch_cb_size_bytes` to grow the socket page size if needed (see [Sizing](#sizing)).

### Example: shipping a tensor as metadata

A common pattern is to pack a small control structure — per-step indices, attention masks, RoPE position vectors — as a tensor host-side and ship it inline as metadata. The service treats it as opaque bytes; the worker kernel reinterprets the L1 region as the same element type.

Host (Python):

```python
import torch, ttnn

# 16 int32 control words per transfer.
METADATA_ELEMS = 16
METADATA_DTYPE = torch.int32
metadata_size_bytes = METADATA_ELEMS * METADATA_DTYPE.itemsize

service = ttnn.H2DStreamService(
    mesh_device=mesh_device,
    global_spec=global_spec,
    fifo_size_bytes=fifo_size_bytes,
    scratch_cb_size_bytes=scratch_cb_size_bytes,
    worker_cores=worker_cores,
    metadata_size_bytes=metadata_size_bytes,
)

# Per-iter metadata as a torch tensor, serialized to bytes for the API.
meta = torch.tensor([step, mode, *positions], dtype=METADATA_DTYPE)
assert meta.numel() == METADATA_ELEMS
meta_bytes = meta.contiguous().numpy().tobytes()

service.forward_to_tensor_bytes(data, metadata=meta_bytes)
```

Worker kernel (C++ snippet, given `metadata_l1_addr = service.get_metadata_addr()` plumbed as a CT arg):

```cpp
// Same layout as the host-side torch tensor: 16 int32 control words.
volatile tt_l1_ptr int32_t* meta =
    reinterpret_cast<volatile tt_l1_ptr int32_t*>(metadata_l1_addr);

const int32_t step      = meta[0];
const int32_t mode      = meta[1];
// meta[2..] are positions, etc. — caller-defined layout.

// Snapshot into a worker-owned region BEFORE atomic-incing consumed_counter
// if the value is used past the ack (see Worker-side contract above).
```

The wire encoding is just the tensor's raw byte buffer — there is no framing, no length prefix, no type tag. Producer and consumer must agree on element type, count, and layout out-of-band (a shared header, an op-spec constant, etc.).

## Python API

### Constructor

```python
ttnn.H2DStreamService(
    mesh_device,                # MeshDevice, required
    global_spec,                # TensorSpec, required
    fifo_size_bytes,            # int, required
    scratch_cb_size_bytes,      # int, required
    mapper=None,                # TensorToMesh, optional
    socket_buffer_type=ttnn.BufferType.L1,
    socket_mode=ttnn.H2DMode.DEVICE_PULL,
    worker_cores=None,          # CoreRange, optional
    metadata_size_bytes=0,      # int, optional
)
```

The `mapper` is consumed by move: the underlying `unique_ptr` is transferred to the service, and the Python wrapper is invalidated after the constructor returns.

### Methods

- `forward_to_tensor(host_tensor, metadata=b"")` — stream a pre-distributed `Tensor` whose per-shard spec matches.
- `forward_to_tensor_bytes(data, metadata=b"")` — stream raw bytes (numpy/torch contiguous, `ROW_MAJOR` only).
- `barrier()` — block until the device has ACKed every in-flight write.

### Getters

- `get_backing_tensor() -> Tensor`
- `get_per_shard_spec() -> TensorSpec`
- `get_sockets() -> List[H2DSocket]`
- `get_data_ready_sem_addr() -> int` — raises if `worker_cores` is unset.
- `get_consumed_counter_addr(coord: MeshCoordinate) -> int`
- `get_service_core(coord: MeshCoordinate) -> CoreCoord` — logical.
- `get_metadata_addr() -> int` — raises if `metadata_size_bytes == 0`.

### Per-call contracts

- Bytes path: `len(data) == global_spec.compute_packed_buffer_size_bytes()` and `ROW_MAJOR`.
- Tensor path: per-shard spec must match `get_per_shard_spec()`.
- Both paths: `len(metadata) == Config::metadata_size_bytes` exactly.
- Validation errors raise as `RuntimeError`.

## Writing a consumer op against the service

This is where the service meets your ttnn op. The service has done the IO work; you write the op that consumes from the backing tensor. Standard ttnn-op shape: one `MeshWorkload` with one `Program` per coord, running your consumer kernel on `worker_cores`.

The op stays resident for the lifetime of the service — you launch it once, and it loops over transfers internally. Each loop iteration: wait on `data_ready_sem`, read backing tensor + metadata, run compute, ack `consumed_counter`.

### What to plumb in

Pull these out of the service at op-build time. Decide CT vs RT based on whether the value is uniform across the mesh:

| What | Source | CT / RT | Notes |
|---|---|---|---|
| `data_ready_sem_addr` | `service.get_data_ready_sem_addr()` | **CT** | Uniform across (device, worker core). |
| `metadata_addr` *(if enabled)* | `service.get_metadata_addr()` | **CT** | Uniform across (device, worker core). |
| `backing_tensor` base address | `service.get_backing_tensor().buffer()->address()` | **CT** | Same address across the mesh under replicated allocation. |
| `TensorAccessorArgs` for the backing tensor | from any per-coord device buffer | **CT** | The tensor spec is uniform, so a single set of args works. |
| `consumed_counter_addr` | `service.get_consumed_counter_addr(coord)` | **RT, per coord** | Varies per device. |
| `service_noc_x`, `service_noc_y` | physical NoC coords of `service.get_service_core(coord)` on that device | **RT, per coord** | Varies per device. |
| Per-worker page slice, start/end indices, etc. | your op's design | **RT, per worker core** | The usual ttnn pattern. |

The logical → physical conversion for the service core is per-device and must happen at op-build time, before runtime args are set:

```cpp
auto* device = mesh_device->get_device(coord);
const auto service_logical  = service.get_service_core(coord);
const auto service_physical = device->worker_core_from_logical_core(service_logical);
// Pass service_physical.x, service_physical.y as RT args to every worker on this device.
```

### Consumer kernel pattern

```cpp
// CT args.
constexpr uint32_t data_ready_sem_addr = get_compile_time_arg_val(0);
constexpr uint32_t backing_tensor_addr = get_compile_time_arg_val(1);
// ... metadata_addr, TensorAccessorArgs, etc.

void kernel_main() {
    // RT args (per coord, per worker).
    const uint32_t consumed_counter_addr = get_arg_val<uint32_t>(0);
    const uint32_t service_noc_x         = get_arg_val<uint32_t>(1);
    const uint32_t service_noc_y         = get_arg_val<uint32_t>(2);
    const uint32_t start_page            = get_arg_val<uint32_t>(3);
    const uint32_t end_page              = get_arg_val<uint32_t>(4);

    auto backing = TensorAccessor(acc_args, backing_tensor_addr);

    volatile tt_l1_ptr uint32_t* data_ready =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(data_ready_sem_addr);
    const uint64_t consumed_noc =
        get_noc_addr(service_noc_x, service_noc_y, consumed_counter_addr);

    while (true) {
        // 1. Wait for the service core to signal a fresh transfer. Reset to 0 so the
        //    next iteration starts fresh — no iteration counter flows from host.
        while (*data_ready == 0) { invalidate_l1_cache(); }
        *data_ready = 0;

        // 2. (If metadata is enabled) snapshot it into worker-local L1 BEFORE
        //    the ack. The service core may overwrite metadata_addr the moment it
        //    sees num_workers acks.
        //    e.g. memcpy from metadata_addr into a local scratch region.

        // 3. Read your slice of the backing tensor and run downstream compute.
        for (uint32_t p = start_page; p < end_page; ++p) {
            noc_async_read(backing.get_noc_addr(p), cb_l1, page_size);
            noc_async_read_barrier();
            // ... compute on the page ...
        }

        // 4. Ack exactly once per transfer.
        noc_semaphore_inc(consumed_noc, 1);
        noc_async_atomic_barrier();
    }
}
```

The shape of the consumer kernel mirrors the service core's iteration boundary: one outer loop, one ack per iteration, no host involvement.

### Footguns

- `get_service_core` returns a **logical** `CoreCoord`. Convert via `worker_core_from_logical_core` before using as a NoC address.
- `consumed_counter_addr` and the service core **vary per device**. They must be per-coord runtime args, never compile-time.
- Workers must ack **exactly once** per transfer. Skipping or double-acking hangs the service core — it's polling for exact equality on `num_workers`.
- **Snapshot metadata before acking.** The service core can start the next transfer's multicast the moment it sees the final ack — reading metadata after the ack is racy.
- `Config::worker_cores` must be **uniform across the mesh** (same logical `CoreRange` on every device).
- Don't try to read the backing tensor from the host without calling `barrier()` first — you'll read mid-write.

## Sizing

Supported configurations today:

- `UINT32`, `ROW_MAJOR`, DRAM-interleaved end-to-end.
- The raw-bytes path is `ROW_MAJOR` only.
- `scratch_cb_size_bytes >= tensor_page_size`.

Chunk plan math (determines `socket_page_size` and the metadata budget):

```
pages_per_chunk  = floor(scratch_cb_size_bytes / tensor_page_size)
                   reduced to a divisor of tensor_num_pages
socket_page_size = pages_per_chunk * tensor_page_size
```

The metadata budget is bounded by `socket_page_size`: `metadata_size_bytes <= socket_page_size`. To increase the budget, raise `scratch_cb_size_bytes` so that `pages_per_chunk` grows.

The service core issues `num_pages = 1` per-page writes from the scratch CB; this is intentional and is not batched.

## End-to-end example

```python
import ttnn
import torch

mesh_device = ttnn.open_mesh_device(...)

global_spec = ttnn.TensorSpec(
    shape=[1, 1, 1024, 1024],
    dtype=ttnn.uint32,
    layout=ttnn.ROW_MAJOR_LAYOUT,
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
)

worker_cores = ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 3))

service = ttnn.H2DStreamService(
    mesh_device=mesh_device,
    global_spec=global_spec,
    fifo_size_bytes=4 * 1024 * 1024,
    scratch_cb_size_bytes=64 * 1024,
    mapper=ttnn.replicate_tensor_to_mesh_mapper(mesh_device),
    worker_cores=worker_cores,
    metadata_size_bytes=64,
)

backing_tensor   = service.get_backing_tensor()
data_ready_addr  = service.get_data_ready_sem_addr()
metadata_addr    = service.get_metadata_addr()

# Per-coord runtime args for the consumer op.
per_coord_rt_args = {}
for coord in mesh_device.get_coords():
    service_logical  = service.get_service_core(coord)
    # The consumer kernel passes service_logical through
    # device->worker_core_from_logical_core(...) at op runtime to
    # obtain the physical NoC coords for the ack target.
    per_coord_rt_args[coord] = {
        "consumed_counter_addr": service.get_consumed_counter_addr(coord),
        "service_logical_core":  service_logical,
    }

# Launch the consumer op once; it stays resident and processes each transfer.
launch_consumer_op(mesh_device, backing_tensor, data_ready_addr, metadata_addr, per_coord_rt_args)

# Streaming loop.
for step in range(num_steps):
    host_data = torch.randint(0, 1 << 31, (1, 1, 1024, 1024), dtype=torch.int32)
    metadata  = build_metadata(step)        # exactly 64 bytes
    service.forward_to_tensor_bytes(host_data.numpy().tobytes(), metadata=metadata)

# Optional: barrier before destruction if you need to observe state.
service.barrier()
# Destruction runs: barrier -> signal_termination -> Finish -> wait_done.
del service
```

## Cross-process C++ connector API

The producer does not have to be the process that constructed the service. A separate process — for example a host-side inference scheduler driving IO into a model workload owned by another process — can attach to an exported `H2DStreamService` via shared memory and push bytes into the same backing tensor.

### When to use it

- The owner process holds the `MeshDevice`, builds the workload, and constructs the service. It calls `service.export_descriptor(id)` once.
- The connector process attaches with `H2DStreamService::connect(id)`. It does **not** open a `MeshDevice`, does **not** initialize `MetalContext`, and does **not** need to link MPI or the dispatch stack.
- The connector pushes raw bytes. It does not need access to `Tensor`, `TensorSpec`, or any mapper type — the service's getters return everything in plain scalar units.

### Surface

Three query getters tell the connector what to push:

| Getter | Returns | Use |
|---|---|---|
| `payload_size_bytes()` | `std::size_t` | Size of the buffer to hand to `forward_to_tensor` per call. |
| `metadata_size_bytes()` | `std::size_t` | Size of the trailing metadata buffer per call. Zero ⇒ metadata path disabled; use the single-arg `forward_to_tensor` overload. |
| `get_worker_cores()` | `CoreRange` | The worker grid the owner-side consumer op runs on. `TT_FATAL`s if `worker_cores` was unset at construction. |

Two data-path methods:

- `forward_to_tensor(ttsl::Span<const std::byte> bytes)` — bytes-only; pushes one full tensor's worth of payload.
- `forward_to_tensor(ttsl::Span<const std::byte> bytes, ttsl::Span<const std::byte> metadata)` — bytes + metadata; both sizes must match the getters above exactly.

One sync method:

- `barrier()` — blocks until every socket's `bytes_sent == bytes_acked`. Call before exit, and before any host read of the backing tensor.

### Owner / connector contract

- `service_id` is the string passed to `export_descriptor` and re-used in `connect`. It must agree on both sides; no namespacing or filesystem path is exposed to the caller.
- The descriptor file at `/dev/shm/tt_h2d_stream_service_<id>.bin` is the rendezvous point. `connect` waits for it to appear (timeout default 10s).
- Owner-only getters (`get_backing_tensor`, `get_data_ready_sem_addr`, `get_consumed_counter_addr`, `get_service_core`, `get_metadata_addr`, `export_descriptor`) `TT_FATAL` if called on a connector-side service. The query getters above and the data path do not.
- Lifetime: the connector's `H2DStreamService` destructor marks each socket's SHM connector-state `clean_shutdown=1` and detaches the SHM regions. It does **not** tear down owner-side resources. Crash exits without running the destructor are observable on the owner via `H2DSocket::had_clean_prior_shutdown()`.

### Minimal connector example

Canonical reference for an inference-scheduler producer:

```cpp
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Minimal standalone connector binary for an exported H2DStreamService.
//
// Does NOT open a MeshDevice, NOT initialize MetalContext, NOT link MPI,
// and NEVER touches any Tensor / TensorSpec type. Operates purely on raw
// byte buffers — the service's `payload_size_bytes()` /
// `metadata_size_bytes()` getters tell the connector exactly how big a
// buffer to hand to `forward_to_tensor` per call.
//
// Expects an owner process (the one with the device handle) to have
// already constructed an `H2DStreamService` and called
// `export_descriptor(id)`. The descriptor lives at
// `/dev/shm/tt_h2d_stream_service_<id>.bin`; `H2DStreamService::connect(id)`
// blocks until it appears, then attaches every per-coord H2DSocket via
// shared memory + a process-local UMD Cluster.
//
// Usage:
//   ./minimal_h2d_connector <service_id> <num_iterations> [timeout_ms]

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#include <tt_stl/span.hpp>

#include <ttnn/tensor/socket_services.hpp>

namespace {

// Fill a byte buffer with whatever deterministic pattern the owner-side
// consumer expects. Real connectors will pull from a file / network / model
// output — this is just a placeholder.
void fill_payload(std::vector<std::byte>& buf, uint32_t iter) {
    const uint8_t seed = static_cast<uint8_t>(iter & 0xFF);
    std::memset(buf.data(), seed, buf.size());
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "usage: " << argv[0]
                  << " <service_id> <num_iterations> [timeout_ms]\n";
        return 1;
    }
    const std::string service_id = argv[1];
    const uint32_t num_iterations = static_cast<uint32_t>(std::stoul(argv[2]));
    const uint32_t timeout_ms =
        (argc >= 4) ? static_cast<uint32_t>(std::stoul(argv[3])) : 30'000u;

    // ---- 1. Attach to the exported service. -----------------------------
    // Blocks until the descriptor file appears in /dev/shm/, or throws on
    // timeout. After this returns, every per-coord H2DSocket is connected
    // through shared memory + the process-local UMD Cluster.
    auto service = tt::tt_metal::H2DStreamService::connect(service_id, timeout_ms);

    // ---- 2. Query the bytes-only contract. ------------------------------
    // payload_size_bytes(): bytes to push per `forward_to_tensor` call.
    // metadata_size_bytes(): trailing metadata bytes per call (0 if disabled).
    // No Tensor / TensorSpec types involved.
    const std::size_t payload_bytes = service->payload_size_bytes();
    const std::size_t metadata_bytes = service->metadata_size_bytes();

    std::cout << "[connector] attached. service_id=" << service_id
              << " payload_bytes=" << payload_bytes
              << " metadata_bytes=" << metadata_bytes << std::endl;

    std::vector<std::byte> payload(payload_bytes);
    std::vector<std::byte> metadata(metadata_bytes);

    // ---- 3. Iteration loop. ---------------------------------------------
    for (uint32_t iter = 0; iter < num_iterations; ++iter) {
        // Refresh payload from your data source. Replace with the real I/O.
        fill_payload(payload, iter);

        auto payload_span = ttsl::Span<const std::byte>(payload.data(), payload.size());

        // Synchronous up to the SHM/L1 write — returns as soon as bytes are
        // in the FIFO / pinned host buffer. The FIFO's flow control blocks
        // here naturally if the device kernel hasn't drained the previous
        // push yet.
        if (metadata_bytes == 0) {
            service->forward_to_tensor(payload_span);
        } else {
            auto metadata_span = ttsl::Span<const std::byte>(metadata.data(), metadata.size());
            service->forward_to_tensor(payload_span, metadata_span);
        }

        // (Optional) per-iter sync — uncomment if a downstream consumer
        // needs every iter to be fully drained before the next is pushed.
        // The hot-path default is pipelined (no per-iter barrier).
        //
        // service->barrier();
    }

    // ---- 4. Drain. -------------------------------------------------------
    // Wait until every socket's bytes_sent == bytes_acked before exiting.
    // Without this, process exit could tear down SHM while the device kernel
    // is mid-read (DEVICE_PULL) or mid-fence (HOST_PUSH).
    service->barrier();

    std::cout << "[connector] " << num_iterations << " iters complete, "
              << (static_cast<std::size_t>(num_iterations) * payload_bytes)
              << " B pushed total" << std::endl;

    // ~H2DStreamService runs the connector dtor path: marks each socket's
    // SHM connector-state clean_shutdown=1 and detaches the SHM regions.
    return 0;
}
```

Step-by-step:

1. **Attach.** `H2DStreamService::connect(service_id, timeout_ms)` blocks until the descriptor is visible. After it returns, every per-coord socket is mapped into this process via shared memory and a single shared `umd::Cluster`.
2. **Query the contract.** `payload_size_bytes()` and `metadata_size_bytes()` are the only sizing inputs the connector needs. Allocate two `std::vector<std::byte>` of those sizes once, reuse across iterations.
3. **Push.** Refresh the payload buffer from the scheduler's data source, call `forward_to_tensor`. Backpressure from the FIFO blocks the call when the device hasn't drained the previous push — no flow-control logic in the scheduler.
4. **Drain.** Call `barrier()` once before exit so the SHM regions can be detached cleanly.

### Build

Link only `TTNN::CPP` and `tt_metal`. No MPI, no dispatch fixture. Add to `tests/tt_metal/distributed/multiprocess/CMakeLists.txt` (or wherever the scheduler lives):

```cmake
add_executable(minimal_h2d_connector minimal_h2d_connector.cpp)
target_link_libraries(minimal_h2d_connector PRIVATE TTNN::CPP tt_metal)
target_include_directories(minimal_h2d_connector
    PRIVATE "$<TARGET_PROPERTY:Metalium::Metal,INCLUDE_DIRECTORIES>")
```

### Footguns

- **Owner must call `export_descriptor` before the connector calls `connect`.** Otherwise `connect` waits the full timeout and throws.
- **Payload size must match exactly.** `forward_to_tensor` `TT_FATAL`s on a mismatch — the connector cannot push partial tensors.
- **Metadata is opt-in but enforced.** If the owner constructed the service with `metadata_size_bytes > 0`, the connector must use the two-arg overload on every call with exactly that many bytes; the single-arg overload `TT_FATAL`s.
- **One connector at a time per socket.** Two concurrent connectors attached to the same descriptor will both write into the same FIFO and corrupt each other's flow control. Use distinct service IDs for distinct producers.
- **Always `barrier()` before exit.** Skipping it can leave the device kernel reading from an SHM region the connector has already detached.
