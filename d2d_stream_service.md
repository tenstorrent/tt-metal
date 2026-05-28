# D2DStreamService

A persistent device-to-device streaming service backed by a fixed device tensor on each side and a `MeshSocket` running over `tt-fabric`. The D2D analog of `H2DStreamService`: where H2D drains a PCIe-pinned host FIFO into a single backing tensor, D2D drains a worker-produced backing tensor on a SENDER mesh into a backing tensor on a RECEIVER mesh.

The data path is **purely device-side after construction**. Host involvement is limited to building the pair and tearing it down — no host writes, no host reads, no host barrier API. One persistent kernel per side per participating coord is launched at construction and runs for the lifetime of the service.

## Topology

The full hop sequence for one transfer, from sender-side worker compute to receiver-side worker compute, runs entirely on device:

```
   Sender mesh (one coord shown — replicated 1:1 across the mesh)
   ──────────────────────────────────────────────────────────────────

   ┌─────────────────────────────────────────────┐
   │ Sender worker cores (Config::sender_worker_cores,
   │                      uniform across sender mesh)
   │  ┌───────────────────────────────────────┐  │
   │  │ User sender worker kernel             │  │
   │  │  • writes its slice → backing tensor  │  │
   │  │  • atomic-inc data_ready_counter      │  │
   │  │  • spins on local consumed_sem        │  │
   │  │  • reset *sem = 0; produce next iter  │  │
   │  └───────────────────────────────────────┘  │
   │  Worker-core L1:                            │
   │    consumed_sem  (GlobalSemaphore, mesh-wide)│
   └────────────────┬────────────────────────────┘
                    │ NoC writes (per shard)            ▲ NoC mcast-inc
                    ▼                                   │ (consumed_sem)
        ┌──────────────────────────┐                    │
        │  Sender backing tensor   │                    │
        │      (DRAM)              │                    │
        └──────────┬───────────────┘                    │
                   │ NoC reads (DRAM → scratch CB)      │
                   ▼                                    │
   ┌─────────────────────────────────────────────┐      │
   │ Sender service core (one per device,        │      │
   │                      off the worker grid)   │      │
   │  ┌───────────────────────────────────────┐  │      │
   │  │ Persistent sender kernel (1 RISC)     │  │      │
   │  │  • polls data_ready_counter           │  │      │
   │  │  • per chunk:                         │  │      │
   │  │     – noc_async_read DRAM → CB        │  │      │
   │  │     – fabric-write CB → recv FIFO     │──┼──┐   │
   │  │     – socket_push_pages / notify_recv │  │  │   │
   │  │  • mcast-inc consumed_sem ────────────┼──┼──┼───┘
   │  └───────────────────────────────────────┘  │  │
   │  Service-core L1:                           │  │
   │    sender socket-config                     │  │
   │    scratch CB (chunk staging)               │  │
   │    fabric packet-header CB                  │  │
   │    data_ready_counter (per coord)           │  │
   │    termination word                         │  │
   └─────────────────────────────────────────────┘  │
                                                    │
   ╔════════════════════════════════════════════════╪═══════════════╗
   ║                  tt-fabric  (MeshSocket pair)  │               ║
   ║       sender coord (x, y)  ◄────────────────►  receiver        ║
   ║                                       coord (x, y)             ║
   ╚════════════════════════════════════════════════╪═══════════════╝
                                                    │
   Receiver mesh (one coord shown — 1:1 with sender coord)
   ──────────────────────────────────────────────────────────────────
                                                    ▼
   ┌─────────────────────────────────────────────┐
   │ Receiver service core (one per device,      │
   │                        off the worker grid) │
   │  ┌───────────────────────────────────────┐  │
   │  │ Persistent receiver kernel (1 RISC)   │  │
   │  │  • socket_wait_for_pages              │  │
   │  │  • per chunk:                         │  │
   │  │     – noc_async_read FIFO → CB        │  │
   │  │     – noc_async_write CB → DRAM       │  │
   │  │     – socket_pop_pages / notify_send  │  │
   │  │  • mcast-inc data_ready_sem ──────────┼──┼──┐
   │  │  • poll consumed_counter              │  │  │
   │  └───────────────────────────────────────┘  │  │
   │  Service-core L1:                           │  │
   │    receiver socket-config + data FIFO       │  │
   │    scratch CB (chunk staging)               │  │
   │    consumed_counter (per coord)             │  │
   │    termination word                         │  │
   └────────────────┬────────────────────────────┘  │
                    │ NoC writes                    │
                    ▼                               │
        ┌──────────────────────────┐                │
        │ Receiver backing tensor  │                │
        │      (DRAM)              │                │
        └──────────┬───────────────┘                │
                   │ NoC reads (per shard)          │
                   ▼                                │
   ┌─────────────────────────────────────────────┐  │
   │ Receiver worker cores (Config::receiver_    │  │
   │                        worker_cores)        │  │
   │  ┌───────────────────────────────────────┐  │  │
   │  │ User receiver worker kernel           │◄─┘  │
   │  │  • spins on local data_ready_sem      │     │
   │  │  • reset *sem = 0                     │     │
   │  │  • reads backing tensor + computes    │     │
   │  │  • atomic-inc consumed_counter ───────┼─────┘
   │  └───────────────────────────────────────┘  │
   │  Worker-core L1:                            │
   │    data_ready_sem (GlobalSemaphore, mesh-wide)│
   └─────────────────────────────────────────────┘
```

Two sync resources never leave their side of the fabric — the sender's `data_ready_counter` / `consumed_sem` close a loop entirely on the sender mesh, and the receiver's `data_ready_sem` / `consumed_counter` close a loop entirely on the receiver mesh. The fabric carries data plus the socket's own flow-control credits, nothing else.

## Application: Multi-Galaxy prefill pipeline

D2D is built for staging activations across a chain of Galaxies in a pipelined prefill. Each Galaxy is an 8 × 4 grid of chips and owns one stage of the model. Activations flow Galaxy-to-Galaxy along a coordinate-preserving 1:1 mapping — chip `(x, y)` on stage N forwards to chip `(x, y)` on stage N+1 — so every column streams independently and row synchronization is only required when the model itself runs a CCL on a row.

### Pipeline shape (one coord shown — replicated for every `(x, y) ∈ 8×4 = 32`)

```
       Host                Galaxy 0                Galaxy 1                       Galaxy K-1
                          (stage 0)               (stage 1)                       (stage K-1)
                       ┌────────────┐          ┌────────────┐                  ┌────────────┐
       H2D             │ chip (x,y) │   D2D    │ chip (x,y) │     D2D          │ chip (x,y) │
   ──────────────────► │ inbound    │          │ inbound    │                  │ inbound    │
   H2DStreamService    │            │ ───────► │            │  ───── ... ────► │            │
                       │  workers   │ outbound │  workers   │ outbound         │  workers   │
                       │  (stage 0) │          │  (stage 1) │                  │  (stage   │
                       │            │          │            │                  │   K-1)    │
                       │ outbound   │ ───────► │ outbound   │  ───── ... ────► │ outbound   │
                       └────────────┘   D2D    └────────────┘    D2D           └────────────┘
                                                                                     │
                                                                                     ▼
                                                                              (D2H / next op /
                                                                               result harvest)
```

One `D2DStreamService::create_pair(galaxy_N, galaxy_N+1, cfg)` call per Galaxy boundary. Internally that single pair contains 32 chip-to-chip sockets — one per `(x, y)` coord — built from the mapper's `TensorTopology::mesh_coords()`. Galaxy 0's inbound is **not** D2D — it's an `H2DStreamService` from the host, because there is no upstream Galaxy. The last Galaxy's outbound is whatever the application needs (typically a D2H back to host for result harvest).

### Per-chip view (a chip in any middle stage)

A middle-stage chip is simultaneously the **receiver** of the upstream pair (its "inbound" socket) and the **sender** of the downstream pair (its "outbound" socket). Both pairs run their own persistent service-core kernel; the worker grid in the middle is the model slice.

```
   chip (x, y) on Galaxy N (middle stage)
   ──────────────────────────────────────

       From chip (x, y) on Galaxy N-1
                │
                ▼ (fabric, inbound D2D pair)
   ┌──────────────────────────────────┐
   │ inbound D2DStreamServiceReceiver │
   │  • service-core kernel drains    │
   │    socket FIFO → backing tensor  │
   │  • data_ready_sem on workers     │
   │  • consumed_counter on svc-core  │
   └────────────────┬─────────────────┘
                    │ NoC read (per-shard)
                    ▼
   ┌──────────────────────────────────┐
   │ Worker grid (model slices for    │
   │ stage N)                         │
   │  • spin on inbound data_ready_sem│
   │  • read inbound backing tensor   │
   │  • run model layers; CCLs along  │
   │    the row as needed             │
   │  • write outbound backing tensor │
   │  • atomic-inc inbound's          │
   │    consumed_counter              │
   │  • atomic-inc outbound's         │
   │    data_ready_counter            │
   │  • spin on outbound consumed_sem │
   └────────────────┬─────────────────┘
                    │ NoC writes
                    ▼
   ┌──────────────────────────────────┐
   │ outbound D2DStreamServiceSender  │
   │  • service-core kernel reads     │
   │    backing tensor → fabric write │
   │  • data_ready_counter on svc-core│
   │  • consumed_sem on workers       │
   └────────────────┬─────────────────┘
                    │
                    ▼ (fabric, outbound D2D pair)
       To chip (x, y) on Galaxy N+1
```

A first-stage chip looks identical except the inbound side is `H2DStreamServiceReceiver` (host-fed via PCIe-pinned FIFO) instead of `D2DStreamServiceReceiver`. A last-stage chip looks identical except the outbound side is a D2H / result-harvest path instead of `D2DStreamServiceSender`.

### Why this composition

- **Compute / forwarding overlap.** All three of (inbound service kernel, worker grid, outbound service kernel) on chip `(x, y)` of stage N can be active simultaneously — the inbound kernel is pulling in iter `i+1`, the worker grid is computing iter `i`, and the outbound kernel is shipping iter `i-1` to stage N+1. Per-iter latency is bounded by the slowest of the three; per-iter throughput is decoupled from per-iter latency.
- **No all-gather / scatter at the Galaxy boundary.** Forwarding column-by-column with coord-preserving mapping means the activation slice that chip `(x, y)` of stage N produces is exactly what chip `(x, y)` of stage N+1 needs — no cross-row gather on the sender side, no cross-row scatter on the receiver side. The alternative (forwarding through only the last row of stage N, with a row-wide all-gather to assemble the activation and a corresponding scatter on stage N+1) is what this avoids.
- **Row CCLs stay row-local.** Synchronization between chips on a row only happens when the worker kernels themselves issue a CCL (for model-internal collectives). The D2D forwarding path never reaches across a row — chip `(x, y)`'s outbound socket has no NoC dependency on chip `(x', y)`'s worker grid. CCLs and forwarding are independently scheduled.
- **Flow control is implicit.** The sender's outbound socket and the receiver's inbound socket are the two halves of one `MeshSocket`; the socket's own credits back-pressure the upstream stage when the downstream worker grid hasn't acked the previous iter. Nothing else needs to know about pacing.

## Data movement model

Two backing tensors with identical per-shard spec, one on each mesh. The mapper runs once at `create_pair` time and is shared between the two sides — sender coord `(x, y)` is wired 1:1 to receiver coord `(x, y)`, and `sender_mesh->shape() == receiver_mesh->shape()` is a hard precondition.

Each iteration is one full tensor's worth of data:

1. Sender workers fill their per-coord shard of the sender backing tensor and ack via `data_ready_counter`.
2. The sender service kernel chunks the shard, reads each chunk DRAM → L1 scratch CB, and fabric-writes the chunk into the receiver-side socket FIFO.
3. The receiver service kernel pulls one chunk at a time out of the FIFO, drops it into a scratch CB, and writes to the receiver backing tensor in DRAM.
4. After every chunk of the iteration has landed, the receiver service kernel multicast-increments `data_ready_sem` to the receiver worker grid; receiver workers consume the slice and atomic-inc `consumed_counter`.
5. Once the sender's socket sees `bytes_sent == bytes_acked` for the iteration (handled inside `socket_push_pages` + the receiver's `socket_notify_sender`), the sender kernel multicast-increments `consumed_sem` to the sender worker grid, releasing them to produce the next iter.

The chunk count per iter is `num_socket_pages = tensor_num_pages / pages_per_chunk`, with `pages_per_chunk = floor(scratch_cb_size_bytes / tensor_page_size)` reduced to a divisor of `tensor_num_pages`. Same sizing rule as H2D.

### Async behavior

- Both persistent kernels are dispatched once at `create_pair` and stay resident. The fast-dispatch CQ on each mesh is free for the caller's own workloads.
- The data path is fully decoupled from host — there is no host-side `forward_to_tensor` and **no host `barrier()` API**. The user explicitly removed it: the caller's worker workload `Finish()` (whichever side the caller wants to observe) is the only sync point.
- Backpressure is end-to-end through the socket: if the receiver kernel hasn't drained the previous transfer, `socket_reserve_pages` on the sender will block, which back-pressures the sender service core's main loop, which back-pressures the worker-side `consumed_sem` multicast, which keeps the sender workers from producing.

## Service cores

Same model as H2D on both sides: regular Tensix cores that live **outside the worker grid**, reserved at device init via `ServiceCoreManager`. One service core per participating device per side. L1 on a service core uses a separate allocator from the worker-grid `BankManager`, so the kernel's scratch CB, socket FIFO, packet-header CB, `data_ready_counter` / `consumed_counter` slot, and termination word do **not** appear in the worker-grid L1 budget.

- `get_service_core(coord)` on either handle returns a **logical** `CoreCoord`. Sender-side workers must convert via `worker_core_from_logical_core` before using as a NoC target for `data_ready_counter` atomic-incs; receiver-side workers do the same for `consumed_counter`.
- Selection may differ per device — different devices can pick different physical cores from each device's claimable set. Treat the per-coord getters as the source of truth.

## Lifecycle

- **Construction** (`create_pair`) is blocking. Validates shapes + mapper, runs the mapper to derive the per-shard spec + topology, allocates sender and receiver backing tensors, claims one service core per coord on each side, builds the `SocketConnection` list from `TensorTopology::mesh_coords()`, calls `MeshSocket::create_socket_pair`, allocates the sync resources, and dispatches both persistent kernels non-blocking.
- **Streaming**. Both sides loop until a `~D2DStreamServiceSender` / `~D2DStreamServiceReceiver` signals termination. The two handles can be destroyed independently in any order (each owns its own termination semaphore).
- **Destruction** is blocking and per-handle: flips that side's termination semaphore to 1, runs `Finish()` on that side's mesh CQ to wait for the persistent kernel to exit at its next wait point, then releases service-core allocations, socket endpoint, semaphores, and the backing tensor. There is no host barrier — the kernel exits the moment it observes the termination flag at a polling point.

## Worker synchronization

Both `sender_worker_cores` and `receiver_worker_cores` are required: D2D has no "service-only" mode. The two handshakes are mirror images of one another and run independently — sender workers never see receiver state and vice versa.

| Side | Worker → service | Service → worker |
|---|---|---|
| Sender | `data_ready_counter` (service-core L1, atomic-inc per worker per iter) | `consumed_sem` (worker L1, multicast-inc by service after the transfer is drained over fabric) |
| Receiver | `consumed_counter` (service-core L1, atomic-inc per worker per iter) | `data_ready_sem` (worker L1, multicast-inc by service after the transfer has landed in the backing tensor) |

`num_workers = worker_cores.num_cores()` on each side; values are independent — the sender can run 4 workers and the receiver 16 with no constraint between the two.

### Sender-side timeline (inverted handshake)

```
   Sender workers (each)                  Sender service core
   ────────────────────                   ───────────────────

   write slice → sender                   spin on data_ready_counter
   backing tensor (DRAM)                       ▲
        │                                      │
        │ atomic-inc                           │
        │ data_ready_counter ──────────────────┘
        │                                  (cur - last) == num_workers
        │                                      │
   spin on local consumed_sem                  ▼
        ▲                                  drain N chunks:
        │                                    DRAM → scratch CB → fabric
        │                                  (socket_push_pages each chunk,
        │                                   fabric_socket_notify_receiver)
        │                                      │
        │            mcast-inc consumed_sem    │
        └──────────────────────────────────────┘
   reset *sem = 0
   produce next iter
```

### Receiver-side timeline (H2D-shape handshake)

```
   Receiver service core                   Receiver workers (each)
   ─────────────────────                   ───────────────────────

   socket_wait_for_pages
        │
   drain N chunks:
     FIFO → scratch CB → DRAM
   (socket_pop_pages each chunk,
    fabric_socket_notify_sender)
        │
        │ mcast-inc data_ready_sem ──────────────┐
        │                                        ▼
        ▼                                  wait *sem > 0
   poll consumed_counter                   reset *sem = 0
        ▲                                  read backing tensor slice
        │                                  run downstream compute
        │                                  atomic-inc consumed_counter ─┐
        │                                                               │
        └────── (cur - last) == num_workers ───────────────────────────┘
        │
   loop to next iter
```

Invariants (same shape as H2D, applied per side):

- Exactly one ack per worker per iter on each side. Skipping or double-acking hangs that side's service kernel — it polls for exact equality on `num_workers`.
- `data_ready_sem` / `consumed_sem` are reset to 0 by the worker before producing/consuming the next iter. No iteration counter or target value flows over the fabric.
- `sender_worker_cores` and `receiver_worker_cores` must each be uniform across their respective mesh.

## C++ API

V0 is C++ only. No Python binding has been added.

### Factory

```cpp
namespace tt::tt_metal {

class D2DStreamService {
public:
    static std::pair<
        std::unique_ptr<D2DStreamServiceSender>,
        std::unique_ptr<D2DStreamServiceReceiver>>
    create_pair(
        const std::shared_ptr<distributed::MeshDevice>& sender_mesh,
        const std::shared_ptr<distributed::MeshDevice>& receiver_mesh,
        D2DStreamConfig cfg);
};

}  // namespace tt::tt_metal
```

`create_pair` is the **only** entry point; `D2DStreamService` itself holds no state. The returned handles are non-copyable, non-movable (each holds a persistent `MeshWorkload`, a claimed service-core slot per coord, and a `MeshSocket` endpoint).

### `D2DStreamConfig`

```cpp
struct D2DStreamConfig {
    TensorSpec                                 global_spec;
    std::unique_ptr<ttnn::distributed::TensorToMesh> mapper;
    distributed::SocketMemoryConfig            socket_mem_config;
    CoreRange                                  sender_worker_cores;
    CoreRange                                  receiver_worker_cores;
};
```

| Field | Notes |
|---|---|
| `global_spec` | Logical shape & layout of the un-sharded global tensor. The same per-shard spec is allocated on both sides. |
| `mapper` | Required. Same mapper describes both sides — `create_pair` runs it once to derive the per-shard `TensorSpec` and `TensorTopology`. Ownership is transferred into the service (move). Construct via `ttnn::distributed::create_mesh_mapper`. |
| `socket_mem_config` | Forwarded verbatim to `MeshSocket::create_socket_pair`. Controls `socket_storage_type` (L1 or DRAM for the receiver-side FIFO), `fifo_size`, and any sub-device fields. V0 recommends `L1`. |
| `sender_worker_cores` | Worker grid on the **sender** mesh that produces into the sender backing tensor. Uniform across every participating sender device. |
| `receiver_worker_cores` | Worker grid on the **receiver** mesh that consumes from the receiver backing tensor. Uniform across every participating receiver device. Independent of `sender_worker_cores`. |

### `D2DStreamServiceSender` getters

| Getter | Returns | Use |
|---|---|---|
| `get_backing_tensor()` | `const Tensor&` | Sender-side workers write per-shard slices here. |
| `get_per_shard_spec()` | `const TensorSpec&` | Same as `get_backing_tensor().tensor_spec()`. |
| `get_worker_cores()` | `CoreRange` | Echo of `Config::sender_worker_cores`. |
| `get_service_core(coord)` | `CoreCoord` (logical) | Per-coord; workers must convert via `worker_core_from_logical_core`. |
| `get_data_ready_counter_addr(coord)` | `DeviceAddr` | Service-core L1 slot. Sender workers atomic-inc here once per iter. |
| `get_consumed_sem_addr()` | `DeviceAddr` | Worker-L1 sem. Sender workers spin on the local copy after each produce; service multicast-incs once per drained iter. Same address across (device, worker core). |

### `D2DStreamServiceReceiver` getters

| Getter | Returns | Use |
|---|---|---|
| `get_backing_tensor()` | `const Tensor&` | Receiver-side workers read per-shard slices here. |
| `get_per_shard_spec()` | `const TensorSpec&` | Same as `get_backing_tensor().tensor_spec()`. |
| `get_worker_cores()` | `CoreRange` | Echo of `Config::receiver_worker_cores`. |
| `get_service_core(coord)` | `CoreCoord` (logical) | Per-coord; workers must convert via `worker_core_from_logical_core`. |
| `get_data_ready_sem_addr()` | `DeviceAddr` | Worker-L1 sem. Receiver workers spin on the local copy each iter; service multicast-incs after the transfer has landed. Same address across (device, worker core). |
| `get_consumed_counter_addr(coord)` | `DeviceAddr` | Service-core L1 slot. Receiver workers atomic-inc here once per iter. |

### What V0 does **not** expose

- No `barrier()` on either handle. The user's worker workload's own `Finish()` is the only host sync point.
- No `forward_to_tensor*` on either handle — the data path is fully device-side. The "API" the user actually calls per-iter lives in their own worker kernels (atomic-inc / spin / reset).
- No `export_descriptor` / `connect` — V0 is single-host pair-only.
- No inline metadata. Adding metadata is a follow-up feature that ships a trailing socket page from sender → receiver and multicasts to worker L1 on the receiver side (mirrors H2D's metadata mechanism).

## Writing your worker ops against the service

Two worker ops, one per mesh, each a `MeshWorkload` with one `Program` per coord. The shape mirrors the H2D consumer-op pattern, with the **sender side inverted** (write then ack-counter then spin-on-sem, instead of spin-on-sem then read then ack-counter).

### Sender worker kernel pattern

```cpp
constexpr uint32_t consumed_sem_addr     = get_compile_time_arg_val(0);
constexpr uint32_t backing_tensor_addr   = get_compile_time_arg_val(1);
// ... TensorAccessorArgs etc.

void kernel_main() {
    const uint32_t data_ready_counter_addr = get_arg_val<uint32_t>(0);  // RT, per coord
    const uint32_t service_noc_x           = get_arg_val<uint32_t>(1);  // RT, per coord
    const uint32_t service_noc_y           = get_arg_val<uint32_t>(2);
    const uint32_t start_page              = get_arg_val<uint32_t>(3);
    const uint32_t end_page                = get_arg_val<uint32_t>(4);

    auto backing = TensorAccessor(acc_args, backing_tensor_addr);

    volatile tt_l1_ptr uint32_t* consumed_sem =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(consumed_sem_addr);
    const uint64_t data_ready_noc =
        get_noc_addr(service_noc_x, service_noc_y, data_ready_counter_addr);

    while (true) {
        // 1. Produce this iter's slice into the sender backing tensor.
        for (uint32_t p = start_page; p < end_page; ++p) {
            // ... write page p of the backing tensor ...
        }
        noc_async_write_barrier();

        // 2. Ack into data_ready_counter — service kernel waits for num_workers.
        noc_semaphore_inc(data_ready_noc, 1);
        noc_async_atomic_barrier();

        // 3. Wait for the service to confirm the iter has drained.
        while (*consumed_sem == 0) { invalidate_l1_cache(); }
        *consumed_sem = 0;
    }
}
```

### Receiver worker kernel pattern

Identical in shape to the H2D consumer kernel (see `h2d_stream_service.md`). Wait on `data_ready_sem`, reset, read slice, run downstream compute, atomic-inc `consumed_counter` exactly once.

### What to plumb in (both sides)

Same CT vs RT decision tree as H2D, applied per side:

| What | Source | CT / RT | Notes |
|---|---|---|---|
| `consumed_sem_addr` (sender) / `data_ready_sem_addr` (receiver) | sender / receiver getter | **CT** | Uniform across (device, worker core). |
| Backing tensor base address | `sender.get_backing_tensor().buffer()->address()` / receiver equivalent | **CT** | Uniform across the mesh under replicated allocation. |
| `TensorAccessorArgs` | from any per-coord device buffer | **CT** | The per-shard spec is uniform. |
| `data_ready_counter_addr` (sender) / `consumed_counter_addr` (receiver) | per-coord getter | **RT, per coord** | Varies per device. |
| `service_noc_x`, `service_noc_y` | physical NoC coords of `get_service_core(coord)` on that device | **RT, per coord** | Varies per device. |
| Per-worker page slice etc. | your op's design | **RT, per worker core** | Usual ttnn pattern. |

## Implementation plan

V0 is staged into 10 sequential steps. Each step is small enough to compile cleanly and exercise something testable.

| # | Step | Notes |
|---|---|---|
| 1 | **Public header** `ttnn/api/ttnn/tensor/d2d_stream_service.hpp` | **Done.** Two non-copyable, non-movable handle classes with pimpl; `D2DStreamService::create_pair` factory. |
| 2 | **Skeleton `.cpp`** | Create `ttnn/core/tensor/d2d_stream_service.cpp`. Add to `TTNNCPP_SRCS` in `ttnn/sources.cmake`. Implement `Sender::Impl` / `Receiver::Impl`. `create_pair` runs: (a) validate `sender_mesh->shape() == receiver_mesh->shape()`, (b) validate `cfg.mapper != nullptr`, (c) run the mapper on a zero host tensor → derive per-shard spec + topology, (d) allocate sender + receiver backing tensors via `create_device_tensor`, (e) claim one service core per coord on each side via `ServiceCoreManager`, (f) `TT_FATAL("D2DStreamService: socket build not yet implemented")`. Stub getters so the backing tensor + per-shard spec are queryable. First compile + link target. |
| 3 | **MeshSocket pair construction** | Build the `SocketConnection` list from `TensorTopology::mesh_coords()` (NOT the full mesh range). Each connection: `sender_core = {coord, sender_service_cores[coord]}`, `receiver_core = {coord, receiver_service_cores[coord]}`. Call `MeshSocket::create_socket_pair(sender_mesh, receiver_mesh, base_config)`. Print active cores for sanity. No kernels yet. |
| 4 | **Allocate worker-sync resources** | Sender side: per-coord `data_ready_counter` slot in service-core L1 via `ServiceCoreManager::allocate_l1`; mesh-wide `consumed_sem` `GlobalSemaphore` on `sender_worker_cores`; sender `termination_semaphore` `GlobalSemaphore` on sender service-core ranges (init 0). Receiver side: mirror of H2D — mesh-wide `data_ready_sem` `GlobalSemaphore` on `receiver_worker_cores`; per-coord `consumed_counter` slot in service-core L1; receiver `termination_semaphore` `GlobalSemaphore` (init 0). Wire the getters up; test asserts the addresses are non-zero. |
| 5 | **Persistent receiver kernel + workload** | Single-RISC TENSIX kernel. Receiver `kernel_main()` body is roughly a paste of `persistent_h2d_receiver.cpp`'s main loop with the PCIe-pinned source swapped for a socket FIFO read; see the kernel skeleton below. Dispatch the receiver workload non-blocking. With no sender online yet, the kernel parks on `socket_wait_for_pages_with_termination` — should NOT crash. |
| 6 | **Persistent sender kernel + workload** | Single-RISC TENSIX kernel. First end-to-end transfer. Sender body: open fabric connection at entry, loop on `data_ready_counter`, per chunk read DRAM → scratch CB → fabric-write to receiver socket FIFO, `socket_push_pages` + `fabric_socket_notify_receiver`. Multicast-inc `consumed_sem` after the iter drains. Debug-only path: host-readback receiver backing tensor and assert. |
| 7 | **Receiver-side worker handshake** | Wire in a placeholder receiver worker workload: spin on `data_ready_sem` (mesh-wide GlobalSemaphore on `receiver_worker_cores`), reset, read slice, atomic-inc `consumed_counter` on receiver service core. End-to-end without host readback. |
| 8 | **Sender-side worker handshake** (the new inverted direction) | Placeholder sender worker workload: write iota into sender backing tensor, atomic-inc `data_ready_counter` on sender service core, spin on local `consumed_sem`, reset. This is the only part of the API that has no H2D analog — the H2D model has the host write directly into the FIFO, with no upstream device worker to handshake with. |
| 9 | **Termination + dtor sequence** | Each handle's destructor: `signal_termination()` (flip that side's GlobalSemaphore to 1), `Finish()` on that side's mesh CQ to drain the persistent kernel through its next polling point, release service-core L1 allocations, free GlobalSemaphores, drop MeshSocket endpoint, drop backing tensor. Mirror H2D's dtor sequence minus the `barrier()` call. The two handles can be torn down independently. |
| 10 | **Reuse test** | Run the pair for N iterations with different seeds. If the persistent kernels exited after iter 0, iter ≥1 readback shows iter 0's data and the test fails loudly. |

### Sender kernel skeleton (step 6)

Single RISC. Compile-time args:

```
[0]  socket_config_addr            (sender side)
[1]  termination_semaphore_addr    (service-core L1)
[2]  socket_page_size              (== pages_per_chunk * tensor_page_size)
[3]  num_socket_pages              (chunks per transfer)
[4]  pages_per_chunk
[5]  tensor_page_size              (DRAM page size of sender backing tensor)
[6]  input_tensor_addr             (sender backing tensor base address)
[7]  scratch_cb_index
[8]  fabric_packet_header_cb_index
[9]  fabric_max_payload_size
[10] data_ready_counter_addr       (service-core L1, workers inc here)
[11] consumed_sem_addr             (worker L1, service mcasts here)
[12] num_workers
[13..16] worker_mcast_noc bbox     (x_start, y_start, x_end, y_end)
[17..]   TensorAccessorArgs for sender backing tensor
```

Body (pseudocode):

```cpp
fabric_connection.open();
sender_socket = create_sender_socket_interface(socket_config_addr);
set_sender_socket_page_size(sender_socket, socket_page_size);
downstream_enc = get_downstream_encoding(sender_socket, 0);
fabric_set_unicast_route(data_packet_header_addr, downstream_enc);

uint32_t last_data_ready = 0;
while (!terminated) {
    // 1. Wait for the worker grid.
    while (true) {
        if (*termination_ptr == 1) { terminated = true; break; }
        uint32_t cur = *data_ready_counter_ptr;
        if ((cur - last_data_ready) == num_workers) {
            last_data_ready = cur;
            break;
        }
    }
    if (terminated) break;

    // 2. Drain N chunks to the receiver via fabric.
    for (uint32_t chunk = 0; chunk < num_socket_pages; ++chunk) {
        socket_reserve_pages(sender_socket, 1);
        const uint32_t cb_l1_addr = /* scratch CB write ptr */;
        for (uint32_t i = 0; i < pages_per_chunk; ++i) {
            noc_async_read(
                accessor.get_noc_addr(chunk * pages_per_chunk + i),
                cb_l1_addr + i * tensor_page_size,
                tensor_page_size);
        }
        noc_async_read_barrier();

        const uint64_t dst_addr =
            receiver_noc_addr +
            sender_socket.write_ptr +
            sender_socket.downstream_fifo_addr;
        // ... emit packets from cb_l1_addr -> dst_addr via fabric_connection ...
        socket_push_pages(sender_socket, 1);
        fabric_socket_notify_receiver(
            sender_socket, fabric_connection, socket_packet_header_addr);
    }

    // 3. Multicast-inc consumed_sem on the sender worker grid.
    noc_semaphore_inc_multicast(worker_mcast_addr, /*incr=*/1, num_workers);
}
update_socket_config(sender_socket);
fabric_connection.close();
```

The per-chunk fabric write follows `send_async`'s `sender_writer.cpp::write_data_to_remote_core` — adapt the page emit loop, drop the CB push/pop split.

### Receiver kernel skeleton (step 5)

Single RISC. Compile-time args:

```
[0]  socket_config_addr            (receiver side)
[1]  termination_semaphore_addr
[2]  socket_page_size
[3]  num_socket_pages
[4]  pages_per_chunk
[5]  tensor_page_size
[6]  output_tensor_addr
[7]  scratch_cb_index
[8]  fabric_packet_header_cb_index
[9]  data_ready_sem_addr           (worker L1)
[10] consumed_counter_addr         (service-core L1)
[11] num_workers
[12..15] worker_mcast_noc bbox
[16..]   TensorAccessorArgs for receiver backing tensor
```

Body (pseudocode):

```cpp
receiver_socket = create_receiver_socket_interface(socket_config_addr);
set_receiver_socket_page_size(receiver_socket, socket_page_size);

uint32_t last_consumed = 0;
while (!terminated) {
    // 1. Drain N chunks from the socket FIFO into the backing tensor.
    for (uint32_t chunk = 0; chunk < num_socket_pages; ++chunk) {
        if (!socket_wait_for_pages_with_termination(
                receiver_socket, 1, termination_ptr)) {
            terminated = true;
            break;
        }
        const uint32_t cb_l1_addr = /* scratch CB write ptr */;
        noc_async_read(/* fifo source */, cb_l1_addr, socket_page_size);
        noc_async_read_barrier();

        for (uint32_t i = 0; i < pages_per_chunk; ++i) {
            noc_async_write(
                cb_l1_addr + i * tensor_page_size,
                accessor.get_noc_addr(chunk * pages_per_chunk + i),
                tensor_page_size);
        }
        noc_async_write_barrier();

        socket_pop_pages(receiver_socket, 1);
        fabric_socket_notify_sender(
            receiver_socket, fabric_connection, socket_packet_header_addr);
    }
    if (terminated) break;

    // 2. Multicast-inc data_ready_sem on the receiver worker grid.
    noc_semaphore_inc_multicast(worker_mcast_addr, /*incr=*/1, num_workers);

    // 3. Wait for num_workers acks on consumed_counter.
    while (true) {
        uint32_t cur = *consumed_counter_ptr;
        if ((cur - last_consumed) == num_workers) {
            last_consumed = cur;
            break;
        }
    }
}
update_socket_config(receiver_socket);
```

For V0, prefer `socket_storage_type = L1` (simpler, exercised more by existing `send_async` / `recv_async` tests). DRAM mode (`recv_async_op_program_factory.cpp:145-161`) needs a double-buffered scratch CB pattern.

### File layout

```
ttnn/api/ttnn/tensor/d2d_stream_service.hpp                     (done)
ttnn/core/tensor/d2d_stream_service.cpp                         (step 2)
ttnn/sources.cmake                                              (+ 1 line)

models/demos/deepseek_v3_b1/micro_ops/d2d/kernels/
    persistent_d2d_sender.cpp                                   (step 6)
    persistent_d2d_receiver.cpp                                 (step 5)

tests/ttnn/unit_tests/gtests/tensor/test_d2d_stream_service.cpp (step 10)
```

## Sizing

Same plan as H2D, applied per side. The sender and receiver share `socket_page_size` and `num_socket_pages` by construction:

```
pages_per_chunk  = floor(scratch_cb_size_bytes / tensor_page_size)
                   reduced to a divisor of tensor_num_pages
socket_page_size = pages_per_chunk * tensor_page_size
num_socket_pages = tensor_num_pages / pages_per_chunk
```

V0 ships data only — there is no metadata budget to size.

Supported configurations at V0:

- `UINT32`, `ROW_MAJOR`, DRAM-interleaved backing tensors on both sides.
- One service core per device per side, single RISC.
- `socket_storage_type = L1` recommended (DRAM works but exercises the double-buffered scratch pattern).
- `cfg.sender_worker_cores.num_cores()` and `cfg.receiver_worker_cores.num_cores()` independent.

## Sharp edges

- **Single-RISC fabric writes.** `send_async` / `recv_async` use two RISCs (reader + writer). The single-RISC fabric API is believed to work on TENSIX but should be verified before going deep. Fallback if it doesn't: split into one RISC for the worker handshake + control flow and one RISC for the data-path fabric writes, coupled via a CB. Doubles kernel complexity; only do if forced.
- **Sender backing tensor allocation.** Use the same per-shard spec + topology as the receiver, but allocated on the sender mesh. The per-shard spec is identical on both sides by the symmetric-mapping invariant.
- **`SocketConnection` list assembly.** Iterate `TensorTopology::mesh_coords()`, not `mesh_device->all_coords()`. The mapper may skip coords (e.g. partial-replication topologies), and a stray connection on an unparticipating coord will assert inside `MeshSocket::create_socket_pair`.
- **Fabric link availability.** `send_async_op_program_factory.cpp:81-93` asserts `num_cores <= num_available_links` between two fabric nodes. V0 has one core per device per side, so always 1 link required — fine. Flag if a future change ever asks for multi-core service.
- **Termination during `socket_wait_for_pages`.** `socket_wait_for_pages_with_termination` (from `termination.hpp`) is socket-type-agnostic and works for D2D verbatim.
- **Fabric connection lifecycle.** Open at kernel entry, close at exit. **Never reopen mid-loop.** The persistent kernel holds the fabric connection for its entire lifetime. Verify this is OK at deeper-than-toy scale.
- **`get_service_core` returns logical.** Convert via `worker_core_from_logical_core` before using as a NoC address. Same footgun as H2D, applied on both sides.
- **`data_ready_counter_addr` / `consumed_counter_addr` vary per device.** Must be per-coord runtime args, never compile-time.
- **Workers must ack exactly once per iter.** On either side. Skipping or double-acking hangs that side's service kernel.
- **No host `barrier()` API.** If you need to observe the receiver backing tensor on the host, `Finish()` the receiver worker workload's mesh CQ — the persistent receiver kernel has already written the data by the time the worker workload acknowledges.

## Out of scope for V0

| Feature | Status | Notes |
|---|---|---|
| Cross-process / multi-host pair | Deferred | Uses the **other** `MeshSocket` constructor (`MeshSocket(device, config)` with its own handshake), plus an `export_descriptor` / `connect` flow. Requires adapting the host-side rendezvous. |
| Inline metadata path | Deferred | Single worker core forwarding metadata to a known L1 location on the sender service core, then a trailing socket page mirroring H2D's mechanism on the receiver side. |
| Multi-core / multi-RISC service | Deferred | "no optimizations needed at this point" — verbatim. One service core per device per side, single RISC, period. |
| Host `barrier()` API | **Removed** | Explicit user decision: `Finish()` on the caller's own worker workload is the only host sync needed. |
| Host `forward_to_tensor*` API | **Not applicable** | D2D is purely device-side after `create_pair`. |
