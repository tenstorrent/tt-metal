# Pull-Fabric All-Gather: Worked Example

**Status:** Reference example for the target design.

**Related:** [Quasar Pull Fabric: DFB Transaction IDs](Quasar-Pull-Fabric-DFB.md) ·
[Yukon Star Fabric Design](Yukon-Star-Fabric-Design.md)

**Production reference this mirrors:**
`ttnn/cpp/ttnn/operations/ccl/all_gather/device/all_gather_multicast_factory.cpp`

---

## What it does

All-gather on the **last** dim:

```text
input   (1, 1, M, N)                  interleaved DRAM, one shard per device
output  (1, 1, M, N * num_devices)    L1 sharded, fully replicated

DRAM input shard --[N producer DM threads, TensorAccessor]--> payload DFB
payload DFB      --[1 sender DM, pull Fabric multicast]-----> this device's column band,
                                                              on every device
```

Because the gather is on the last dim, a device's data is a **column band
repeated on every tile row**, not one contiguous region — see "Shapes and the
output page map".

Three files:

Structured as a ttnn device operation, same as the production all-gather:

```text
pull_all_gather/
    pull_all_gather_device_operation_types.hpp   PullAllGatherParams / Inputs
    pull_all_gather_device_operation.cpp         specs, output tensor, validate
    pull_all_gather_factory.cpp                  create_mesh_workload / create_at
    kernels/producer.cpp                         N SPMD DM threads, fills the DFB
    kernels/sender.cpp                           1 DM thread, drives pull Fabric
```



## Ground rules

Route and sizing are queried, never assumed:


| Quantity                  | Source                                                     |
| ------------------------- | ---------------------------------------------------------- |
| Devices on the axis       | `ttnn::ccl::get_topological_dimension(input_tensor, axis)` |
| Per-direction hop counts  | `ttnn::ccl::get_forward_backward_line_mcast_distance(...)` |
| This device's chunk index | `ttnn::ccl::get_linearized_index_from_physical_coord(...)` |
| Max bytes per transfer    | `tt::tt_fabric::get_tt_fabric_max_payload_size_bytes()`    |


Four device-side rules follow from the pull design
([DFB doc §1, §5](Quasar-Pull-Fabric-DFB.md)):


| Rule                                                | Why                                                                                                                                                                                                                             |
| --------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| No NoC TRIDs on this DFB                            | The sender's only NoC transaction is the header write; the payload is pulled later by the remote DMA. A TRID would mean "header delivered", not "payload read".                                                                 |
| Producer uses explicit `reserve_back` / `push_back` | Implicit reserve/push *is* the TRID path — the DFB-destination overload of `Noc::async_read` is `enable_if`'d on `NocOptions::TXN_ID` (`noc.h:766`). It also fixes the transfer at `get_entry_size()`, so it cannot pack tiles. |
| Sender never calls `wait_front` / `pop_front`       | Those live inside `Fabric` / `FabricDataflowBuffer`; DFB sync is implicit in the Fabric API.                                                                                                                                    |
| N producer threads are STRIDED                      | All DFB producers are STRIDED (`kernel_spec.hpp:244`). Thread `p` owns entries `p, p+N, ...`, so filling them from the matching tiles makes the sender's FIFO order equal tile order.                                           |




## Why the route is a per-direction hop pair

The fabric router carries no topology knowledge. It executes a positional
per-hop command stream, terminating only where a hop's field is `WRITE_ONLY`
(`fabric_erisc_router.cpp:701`), and advances by shifting that stream
(`fabric_edm_packet_transmission.hpp:378`). `encode_1d_multicast`
(`fabric_common.h:291`) lays it out as:

```text
hops 0 .. start-2        FORWARD_ONLY       0b10
hops start-1 .. last-1   WRITE_AND_FORWARD  0b11
hop  last                WRITE_ONLY         0b01     last = start + range - 2
```

So one stream long enough to reach every peer of a line would tell the last
chip to forward with no downstream link — the terminator sits past the end of
the chain. One stream cannot both stop at `fwd_hops` and cover the chips behind
the sender, and `LowLatencyRoutingFields` is a bare `value` with no branch
offsets, so it cannot fork.

Hence one route per direction, each with its own packet header — the shape
`PacketHeaderPool::allocate_header_n()` already serves in production, where
`FabricWriter` allocates a header per connection. Hop counts come straight from
the existing host API (`ccl_common.cpp:1849`):

```cpp
auto [fwd_hops, bwd_hops] = ::ttnn::ccl::get_forward_backward_line_mcast_distance(
    num_devices, device_idx, topology, /*static_alternate=*/false);

// Ring, size 8    -> (4, 3) for every index
// Linear, index i -> (size - 1 - i, i)
```

Both directions start at distance 1, so `start_distance` never appears — each
gets its own hop count and terminator, and neither runs off the end.
A zero hop count means that direction contributes no route, which is how a
Linear line endpoint needs no special case — it fills one route, not two.

`M` is **not** `fwd_hops + bwd_hops`. It is the route count: the router issues
one packet per direction and the chain store-and-forwards it, so the sender's
L1 is read once per direction. `M = 2` for a two-direction multicast whatever
the hop counts, and `fwd_hops + bwd_hops = num_devices - 1` is the *peer*
count, which the completion barrier uses as its fan-in.

The routes are **not** separate API calls. One
`async_write_multicast_with_state()` publishes every claimed slot under a
single transaction id, advances the DFB read pointer once, and commits one
transaction of `M` (DFB doc §1). Issuing directions separately would consume
one entry each.

`static_alternate` swaps the pair on even indices — the ring load balancing the
production kernel gets by toggling prebuilt routes.

**2D needs up to four routes, and more per-route state.** A mesh range forks
E/W within its own spine but cannot reach the opposite spine, so a mid-mesh
device needs E-line, W-line, N-rect, S-rect. `mesh/api.h:1305` indexes
`ranges[i]` per header and reads `dst_dev_id` / `dst_mesh_id` from
`connection_manager.get(i)` — per route, not per call. The pull path has no
connection manager, so those move into the route args
(`FabricRouteArgs<..., CHIP_MULTICAST, true>::Route`), and the host populates
them the way the production factory builds its `ranges[]`:

```cpp
// Sketch; this example is 1D and does not exercise it.
// Fill exactly the routes this device needs; num_routes is that count.
uint32_t r = 0;
if (line_hops > 0) {                       // own row
    uint8_t hops[4] = {};
    hops[line_dir] = line_hops;
    route.routes[r++] = {make_range(hops), line_dst_dev, line_dst_mesh};
}
if (rect_spine_hops > 0) {                 // rows above or below, with E/W fan-out
    uint8_t rect[4] = {};
    if (rect_e_hops > 0) rect[rect_e_dir] = rect_e_hops;
    if (rect_w_hops > 0) rect[rect_w_dir] = rect_w_hops;
    rect[rect_spine_dir] = rect_spine_hops;
    route.routes[r++] = {make_range(rect), rect_dst_dev, rect_dst_mesh};
}
route.num_routes = r;   // M is this count: one SWQ per direction
```

`fabric_max_routes<topology>` sizes both the route array and the request set,
so they cannot disagree.

## Shapes and the output page map

Gathering on the last dim means a device's data is **not** one contiguous
region of the output. It is a column band repeated on every tile row — a
stripe, the structure the production factory indexes with
`output_chunks_per_stripe` / `output_page_stripe_jump`:

```text
output tile grid, num_devices = 4, in_tile_cols = 3

 tile row 0 | d0 d0 d0 | d1 d1 d1 | d2 d2 d2 | d3 d3 d3 |
 tile row 1 | d0 d0 d0 | d1 d1 d1 | d2 d2 d2 | d3 d3 d3 |
 tile row 2 | d0 d0 d0 | d1 d1 d1 | d2 d2 d2 | d3 d3 d3 |
              ^^^^^^^^
              one stripe = in_tile_cols contiguous pages
```

With row-major page numbering over the output tile grid this has a closed
form, so no iterator state is needed. For local tile index `t` in
`[0, tiles_per_device)`:

```text
r = t / in_tile_cols                                        // tile row
c = t % in_tile_cols                                        // column within the band
output_page(t) = r * out_tile_cols + device_idx * in_tile_cols + c
```



## Entry sizing and why the output is L1 sharded

A `FWWriteDMADescriptor` is `(src, dest + BAR, size)` — one contiguous run — so
an entry's tiles must be adjacent at the destination. Two consequences:

**Entries cannot cross a stripe.** `tiles_per_entry = min(max_payload / tile_bytes, in_tile_cols)`, requiring `in_tile_cols % tiles_per_entry == 0`. A
narrow gather dim pins entries to one tile whatever the DMA could move
(`N = 32` → `in_tile_cols = 1`).

**Consecutive pages must be consecutive addresses**, which depends on layout:


| Output layout                   | Consecutive pages land                                        | Works?                            |
| ------------------------------- | ------------------------------------------------------------- | --------------------------------- |
| Interleaved                     | different banks (`InterleavedAddrGen` round-robins `page_id`) | **No** — needs an N-chunk scatter |
| Sharded, entry inside one shard | consecutive `bank_page_offset` (`tensor_accessor.h:310`)      | **Yes**                           |


So the output is L1, **height sharded** — whole tile rows per core. Within a
shard `page_offset_within_shard = (r % rows_per_core) * out_tile_cols + c_global`, so a stripe is contiguous and an entry inside it is one descriptor.
The destination core changes per stripe; the accessor resolves that per page.

The input stays interleaved DRAM: the producer reads one tile per NoC read, so
the source side has no contiguity requirement.

## There is no connection to open

The worker establishes nothing. The local DE's send queue is always there, at a
fixed address, shared by every worker on the module (Yukon doc §5, "Tensix → DE
send queue | **1** | Shared"). The destination rides in the packet header
(Yukon doc §1, "Peer / host is routing on the header ... not another worker
connection"). The send path is: wait for a free slot, write the header into it,
bump the DE's occupied counter.

JIT bindings supply the DE coordinates, request-ring base/depth, and credit
addresses (DFB doc §2.3). Relative to an eth-fabric op that removes:


| eth needs                                    | why                            | here                                 |
| -------------------------------------------- | ------------------------------ | ------------------------------------ |
| `build_from_args<BUILD_AND_OPEN_CONNECTION>` | claim an EDM worker slot       | queue is shared and always present   |
| destination `FabricNodeId` per connection    | selects which eth link to open | one DE; destination is in the header |
| teardown / buffer-index semaphores           | persist handshake state        | no handshake                         |
| `close()`                                    | release the slot               | nothing opened                       |


Flow control survives: `free = capacity - (wr_counter - router_free_credit)`
needs only the credit word's static address.

## Host program

A `ttnn` device operation plus a program factory, same split as production:
the framework owns output allocation, caching, and enqueue; the factory owns
per-device program construction. Line-by-line mapping is at the end.

### Operation attributes and inputs

Mirrors `AllGatherParams` / `AllGatherInputs` minus what the pull path does not
need: no per-axis topology array, no `axis_num_links`, no `packet_size`.

```cpp
// pull_all_gather_device_operation_types.hpp
namespace ttnn::operations::ccl {

struct PullAllGatherParams {
    int32_t dim = 0;                       // gather dim; this example requires the last
    MemoryConfig output_mem_config;        // L1, height sharded (see below)
    std::optional<uint32_t> cluster_axis;

    tt::tt_fabric::FabricConfig fabric_config{};
    // Per axis, as AllGatherParams carries them. An inactive axis has
    // num_devices == 1 and Linear topology.
    std::array<tt::tt_fabric::Topology, 2> axis_topology{};
    std::array<uint32_t, 2> axis_num_devices{};
    uint32_t num_devices = 0;
    size_t max_payload_bytes = 0;          // get_tt_fabric_max_payload_size_bytes()

    std::optional<tt::tt_metal::SubDeviceId> subdevice_id;
    std::optional<CoreRangeSet> sub_core_grid;

    // Producer-side and DFB tunables. There is no transaction-ID count here:
    // the counter bank is reserved L1, not an op allocation (see below).
    uint32_t num_producers = 4;
    uint32_t dfb_depth = 8;

    // Peers reached, == fwd_hops + bwd_hops. Used as the barrier fan-in; it is
    // not M, which is the route count.
    uint32_t peer_count() const { return num_devices - 1; }
};

struct PullAllGatherInputs {
    Tensor input_tensor;
    std::optional<Tensor> persistent_output_tensor;
};

}  // namespace ttnn::operations::ccl
```

`num_devices`, `fabric_config`, and `max_payload_bytes` are filled by the
op adapter exactly as `all_gather_device_operation.cpp:295-315` does:

```cpp
args.num_devices       = ::ttnn::ccl::get_topological_dimension(input_tensor, axis);
args.fabric_config     = tt::tt_fabric::get_fabric_config();
for (uint32_t axis = 0; axis < 2; ++axis) {
    args.axis_topology[axis]    = ::ttnn::ccl::get_axis_topology(input_tensor, args.fabric_config, axis);
    args.axis_num_devices[axis] = ::ttnn::ccl::get_topological_dimension(input_tensor, axis);
}
args.max_payload_bytes = tt::tt_fabric::get_tt_fabric_max_payload_size_bytes();
```



### The transaction counters live in reserved L1

The counter bank is **not** allocated per program and its capacity is not an op
parameter. It is a fixed region of the worker's reserved L1, sized once for the
platform in the memory map:

```c
// tt_metal/hw/inc/internal/tt-2xx/quasar/dev_mem_map.h
#define MEM_FABRIC_MAX_TRANSACTION_IDS  <fixed by the memory map>
#define MEM_FABRIC_TXN_COUNTERS_SIZE    (MEM_FABRIC_MAX_TRANSACTION_IDS * 4)
#define MEM_FABRIC_TXN_COUNTERS_BASE    <reserved region base>
```

The op consumes what is there. The builder sub-allocates a disjoint
`txn_id_base` + range to each sender-side DFB out of that region — DFB doc §3:
"One `Fabric` object can send from multiple `FabricDataflowBuffer` instances.
Each payload adapter owns a disjoint transaction-ID range and its own cursors."

Consequences for this op:

- No `kTransactionCounters` scratchpad in the `ProgramSpec`, and no
`scratchpad_bindings` on the sender.
- The sender kernel constructs `FabricDataflowBuffer payload(dfb::payload)`
with no counter accessor.
- In-flight capacity stays `min(DFB occupancy, IDs assigned to this DFB)`, so a
deeper DFB neither needs nor gets more counters.

`FabricTransactionCounterConfig::storage_size_bytes()`
(`tt-metalium/experimental/fabric/fabric_dataflow_buffer.hpp`) becomes a
memory-map sizing helper rather than something an op calls.

### Device operation

```cpp
// pull_all_gather_device_operation.cpp

PullAllGatherDeviceOperation::spec_return_value_t
PullAllGatherDeviceOperation::compute_output_specs(
    const PullAllGatherParams& args, const PullAllGatherInputs& tensor_args) {
    const auto& input_tensor = tensor_args.input_tensor;
    auto shape = input_tensor.logical_shape();
    shape[args.dim] *= args.num_devices;          // (1,1,M,N) -> (1,1,M,N*num_devices)
    return tt::tt_metal::TensorSpec(
        shape,
        tt::tt_metal::TensorLayout(
            input_tensor.dtype(),
            input_tensor.tensor_spec().page_config(),
            args.output_mem_config));             // L1 + NdShardSpec, see validate
}

PullAllGatherDeviceOperation::tensor_return_value_t
PullAllGatherDeviceOperation::create_output_tensors(
    const PullAllGatherParams& args, const PullAllGatherInputs& tensor_args) {
    if (tensor_args.persistent_output_tensor.has_value()) {
        return tensor_args.persistent_output_tensor.value();
    }
    return create_device_tensor(
        compute_output_specs(args, tensor_args), tensor_args.input_tensor.device());
}

void PullAllGatherDeviceOperation::validate_on_program_cache_miss(
    const PullAllGatherParams& args, const PullAllGatherInputs& tensor_args) {
    const auto& input_spec = tensor_args.input_tensor.tensor_spec();
    const auto& shape = input_spec.logical_shape();

    TT_FATAL(args.dim == static_cast<int32_t>(shape.rank()) - 1,
             "This pull all-gather gathers on the last dim only");
    TT_FATAL(input_spec.layout() == tt::tt_metal::Layout::TILE, "TILE layout required");

    // The output must be sharded so that one DMA descriptor covers one entry:
    // a descriptor is (src, dest + BAR, size), one contiguous run. Interleaved
    // round-robins page_id across banks, so consecutive tiles are not adjacent.
    TT_FATAL(args.output_mem_config.buffer_type() == tt::tt_metal::BufferType::L1 &&
             args.output_mem_config.nd_shard_spec().has_value(),
             "Output must be L1 sharded; interleaved would need an N-chunk scatter");

    TT_FATAL(args.dfb_depth % args.num_producers == 0,
             "STRIDED producers must divide the DFB ring");

    // No topology restriction: the forward/backward pair covers Ring and
    // Linear alike, including a Linear interior device.
}

PullAllGatherDeviceOperation::program_factory_t
PullAllGatherDeviceOperation::select_program_factory(
    const PullAllGatherParams&, const PullAllGatherInputs&) {
    return PullAllGatherFactory{};   // one path: multicast pull
}
```

`validate` enforces sharded L1; the shard shape stays the caller's choice, as
in any ttnn op, but it must be height sharded for the stripe-contiguity reason
above.

### Program factory

```cpp
// pull_all_gather_factory.hpp
struct PullAllGatherFactory {
    struct shared_variables_t {
        tt::tt_metal::GlobalSemaphore barrier_sem;
        uint32_t device_idx = 0;          // which column band this device owns
        std::vector<uint32_t> route_args;   // hops[4] + dirs[4]
    };

    using cached_mesh_workload_t =
        ttnn::device_operation::AdaptedCachedMeshWorkload<shared_variables_t>;

    static cached_mesh_workload_t create_mesh_workload(
        const PullAllGatherParams& operation_attributes,
        const ttnn::MeshCoordinateRangeSet& tensor_coords,
        const PullAllGatherInputs& tensor_args,
        Tensor& output_tensor);

    static void override_runtime_arguments(
        cached_mesh_workload_t& cached_workload,
        const PullAllGatherParams& operation_attributes,
        const PullAllGatherInputs& tensor_args,
        Tensor& output_tensor);

private:
    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static cached_program_t create_at(
        const PullAllGatherParams& operation_attributes,
        const ttnn::MeshCoordinate& sender_device_coord,
        const PullAllGatherInputs& tensor_args,
        const Tensor& output_tensor,
        const tt::tt_metal::GlobalSemaphore& barrier_sem);
};
```

`create_mesh_workload` is the production shape unchanged.

```cpp
// pull_all_gather_factory.cpp

PullAllGatherFactory::cached_mesh_workload_t PullAllGatherFactory::create_mesh_workload(
    const PullAllGatherParams& operation_attributes,
    const ttnn::MeshCoordinateRangeSet& tensor_coords,
    const PullAllGatherInputs& tensor_args,
    Tensor& output_tensor) {
    tt::tt_metal::distributed::MeshWorkload workload;
    std::unordered_map<ttnn::MeshCoordinateRange, shared_variables_t> shared_variables;

    auto* mesh_device = tensor_args.input_tensor.device();
    auto subdevice_id =
        operation_attributes.subdevice_id.value_or(mesh_device->get_sub_device_ids().at(0));
    auto available_cores =
        mesh_device->worker_cores(tt::tt_metal::HalProgrammableCoreType::TENSIX, subdevice_id);
    if (operation_attributes.sub_core_grid.has_value()) {
        available_cores = available_cores.intersection(*operation_attributes.sub_core_grid);
    }
    ttsl::SmallVector<tt::tt_metal::SubDeviceId> subdevices = {subdevice_id};

    // Peers must not multicast into an output buffer this device has not
    // allocated yet, so all devices reach the kernel before any data moves.
    const bool l1_small =
        mesh_device->allocator()->get_bank_size(tt::tt_metal::BufferType::L1_SMALL) > 0;
    auto barrier_sem = ttnn::global_semaphore::create_global_semaphore(
        mesh_device,
        available_cores,
        0,
        l1_small ? tt::tt_metal::BufferType::L1_SMALL : tt::tt_metal::BufferType::L1);
    tt::tt_metal::distributed::Synchronize(mesh_device, std::nullopt, subdevices);

    for (const auto& coord : tensor_coords.coords()) {
        auto cached_program =
            create_at(operation_attributes, coord, tensor_args, output_tensor, barrier_sem);
        workload.add_program(ttnn::MeshCoordinateRange(coord), std::move(cached_program.program));
        shared_variables.emplace(
            ttnn::MeshCoordinateRange(coord), std::move(cached_program.shared_variables));
    }

    return cached_mesh_workload_t{std::move(workload), std::move(shared_variables)};
}
```

`create_at` is the per-device body.

```cpp
PullAllGatherFactory::cached_program_t PullAllGatherFactory::create_at(
    const PullAllGatherParams& args,
    const ttnn::MeshCoordinate& sender_device_coord,
    const PullAllGatherInputs& tensor_args,
    const Tensor& output_tensor,
    const tt::tt_metal::GlobalSemaphore& barrier_sem) {
    namespace m2 = tt::tt_metal::experimental;

    const auto& input_tensor = tensor_args.input_tensor;
    auto* mesh_device = input_tensor.device();

    // ---- Sizes, all derived from the tensors and the fabric ----
    const auto& input_spec = input_tensor.tensor_spec();
    const auto& tile = input_spec.tile();
    const uint32_t tile_h = tile.get_height();
    const uint32_t tile_w = tile.get_width();
    const uint32_t tile_bytes = input_spec.compute_page_size_bytes();

    const auto& shape = input_spec.logical_shape();
    const uint32_t rows = shape[shape.rank() - 2];   // M
    const uint32_t cols = shape[shape.rank() - 1];   // N, the gather dim
    TT_FATAL(rows % tile_h == 0 && cols % tile_w == 0, "Shape must be tile aligned");

    const uint32_t tile_rows        = rows / tile_h;
    const uint32_t in_tile_cols     = cols / tile_w;                    // this device's band
    const uint32_t out_tile_cols    = in_tile_cols * args.num_devices;
    const uint32_t tiles_per_device = tile_rows * in_tile_cols;

    // Entry = one DMA transfer, capped by the stripe so it stays contiguous.
    const uint32_t tiles_per_entry = std::max<uint32_t>(
        1, std::min<size_t>(args.max_payload_bytes / tile_bytes, in_tile_cols));
    const uint32_t entry_bytes        = tiles_per_entry * tile_bytes;
    const uint32_t entries_per_device = tiles_per_device / tiles_per_entry;
    TT_FATAL(
        in_tile_cols % tiles_per_entry == 0,
        "An entry would cross a stripe boundary: band is {} tiles, entry is {}",
        in_tile_cols, tiles_per_entry);

    // ---- ProgramSpec ----
    const m2::DFBSpecName        kPayloadDfb{"payload"};
    const m2::KernelSpecName     kProducer{"pull_all_gather_producer"};
    const m2::KernelSpecName     kSender{"pull_all_gather_sender"};
    const m2::TensorParamName    kInputTensor{"input_tensor"};
    const m2::TensorParamName    kOutputTensor{"output_tensor"};
    const m2::ScratchpadSpecName kFabricRequests{"fabric_requests"};
    constexpr m2::NodeCoord      kWorkerNode{0, 0};
    // Two request *sets*: the payload multicast and the completion atomic.
    // Each set holds fabric_max_routes<topology> slots (2 for 1D, 4 for 2D),
    // because a route needs its own packet header (DFB doc §2.1).
    constexpr uint32_t           kNumRequestSets = 2;
    // Must match fabric_max_routes<topology>: 2 for 1D, 4 for 2D.
    constexpr uint32_t           kMaxRoutes = 2;
    // device_idx, num_peers | 2 groups * 10 route words | sem addr, noc x, noc y
    constexpr uint32_t           kNumRouteGroups = 2;
    constexpr uint32_t           kSenderRuntimeArgs = 2 + kNumRouteGroups * 10 + 3;

    m2::KernelSpec producer{
        .unique_id = kProducer,
        .source = "pull_all_gather/kernels/producer.cpp",
        .num_threads = args.num_producers,        // <-- N producers
        .hw_config = m2::DataMovementGen2Config{
            .disable_dfb_implicit_sync_for = {kPayloadDfb},
        },
    };
    producer.dfb_bindings = {m2::ProducerOf(kPayloadDfb, "payload")};
    producer.tensor_bindings = {{
        .tensor_parameter_name = kInputTensor, .accessor_name = "input_tensor"}};
    // Each producer walks its own count and strides the entry index.
    const uint32_t entries_per_producer =
        (entries_per_device + args.num_producers - 1) / args.num_producers;
    producer.compile_time_args = {
        {"entries_per_producer", entries_per_producer},
        {"entries_per_device", entries_per_device},
        {"num_producers", args.num_producers},
        {"tiles_per_entry", tiles_per_entry},
        {"tile_bytes", tile_bytes},
    };

    m2::KernelSpec sender{
        .unique_id = kSender,
        .source = "pull_all_gather/kernels/sender.cpp",
        .num_threads = 1,                          // <-- one sender DM
        .hw_config = m2::DataMovementGen2Config{
            .disable_dfb_implicit_sync_for = {kPayloadDfb},
        },
    };
    sender.dfb_bindings = {m2::ConsumerOf(kPayloadDfb, "payload")};
    // Request slots only. The transaction-counter bank is reserved L1, so it
    // has no scratchpad.
    sender.scratchpad_bindings = {{
        .scratchpad_spec_name = kFabricRequests,
        .accessor_name = "fabric_requests"}};
    sender.tensor_bindings = {{
        .tensor_parameter_name = kOutputTensor, .accessor_name = "output_tensor"}};
    // Hop counts are per-device on a Linear topology, so they are runtime args
    // (see the workload loop); only shape constants are compile time.
    sender.compile_time_args = {
        {"entries_per_device", entries_per_device},
        {"tiles_per_entry", tiles_per_entry},
        {"in_tile_cols", in_tile_cols},
        {"out_tile_cols", out_tile_cols},
    };
    sender.advanced_options.num_runtime_varargs = kSenderRuntimeArgs;

    const m2::ProgramSpec program_spec{
        .name = "pull_all_gather",
        .kernels = {producer, sender},
        .dataflow_buffers = {{
            .unique_id = kPayloadDfb,
            .entry_size = entry_bytes,             // a DMA transfer, not a tensor page
            .num_entries = args.dfb_depth,
            .data_format_metadata = input_spec.data_format(),
        }},
        // The transaction-counter bank is reserved L1, so the only scratchpad
        // is the request ring (packet header + FabricPullMetadata per slot).
        // No fabric teardown / buffer-index semaphores: those persist eth
        // connection-handshake state, and nothing is opened here.
        .scratchpads = {{
            .unique_id = kFabricRequests,
            .size_per_node = kNumRequestSets * kMaxRoutes * tt::align(
                static_cast<uint32_t>(
                    tt::tt_fabric::get_tt_fabric_packet_header_size_bytes() +
                    sizeof(tt::tt_fabric::FabricPullMetadata)),
                16u),
        }},
        .tensor_parameters = {
            {.unique_id = kInputTensor,  .spec = input_spec},
            {.unique_id = kOutputTensor, .spec = output_tensor.tensor_spec()},
        },
        .work_units = {{
            .name = "pull_all_gather_worker",
            .kernels = {kProducer, kSender},
            .target_nodes = kWorkerNode,
        }},
    };

    auto program = m2::MakeProgramFromSpec(*mesh_device, program_spec);

    // ---- Per-device runtime args ----
    // Per-device values only. No fabric connection args: nothing is opened,
    // and the local DE's queue addresses come from the Fabric binding.
    const uint32_t device_idx = ::ttnn::ccl::get_linearized_index_from_physical_coord(
        input_tensor, sender_device_coord, args.cluster_axis);

    // Route scalars: hop counts per direction, plus the physical E/W/N/S slot
    // each one forwards through. This is all_gather_multicast_factory.cpp's
    // derivation at :88-146, unchanged apart from dropping load balancing.
    uint32_t e_hops = 0, w_hops = 0, n_hops = 0, s_hops = 0;
    std::optional<MeshCoordinate> e_coord, w_coord, n_coord, s_coord;

    for (uint32_t axis = 0; axis < 2; ++axis) {
        if (args.axis_num_devices[axis] <= 1) {
            continue;   // inactive axis
        }
        const auto axis_topology = args.axis_topology[axis];

        auto [fwd_hops, bwd_hops] = ::ttnn::ccl::get_forward_backward_line_mcast_distance(
            args.axis_num_devices[axis], sender_device_coord[axis], axis_topology,
            /*static_alternate=*/false);
        auto fwd_coord = ::ttnn::ccl::get_physical_neighbor_from_physical_coord(
            input_tensor, sender_device_coord, 1, axis_topology, axis);
        auto bwd_coord = ::ttnn::ccl::get_physical_neighbor_from_physical_coord(
            input_tensor, sender_device_coord, -1, axis_topology, axis);

        // axis 1 -> (E = fwd, W = bwd); axis 0 -> (S = fwd, N = bwd)
        if (axis == 1) {
            e_hops = fwd_hops; w_hops = bwd_hops;
            e_coord = fwd_coord; w_coord = bwd_coord;
        } else {
            s_hops = fwd_hops; n_hops = bwd_hops;
            s_coord = fwd_coord; n_coord = bwd_coord;
        }
    }
    TT_FATAL(
        e_hops + w_hops + n_hops + s_hops == args.peer_count(),
        "Routes must cover every peer");

    // Physical direction each neighbour sits in; depends on mesh position.
    const auto sender_node = mesh_device->get_fabric_node_id(sender_device_coord);
    auto physical_slot = [&](const std::optional<MeshCoordinate>& neighbor) -> uint32_t {
        if (!neighbor.has_value()) {
            return 0;
        }
        const auto dir = tt::tt_fabric::get_eth_forwarding_direction(
            sender_node, mesh_device->get_fabric_node_id(*neighbor));
        TT_FATAL(
            dir.has_value() &&
                static_cast<uint32_t>(*dir) < tt::tt_fabric::eth_chan_directions::Z,
            "Expected a cardinal E/W/N/S forwarding direction");
        return static_cast<uint32_t>(*dir);
    };

    const std::vector<uint32_t> route_args = {
        e_hops, w_hops, n_hops, s_hops,
        physical_slot(e_coord), physical_slot(w_coord),
        physical_slot(n_coord), physical_slot(s_coord),
    };

    const auto barrier_node = mesh_device->worker_core_from_logical_core(kWorkerNode);

    m2::ProgramRunArgs run_args;
    run_args.kernel_run_args = {{
        .kernel = kSender,
        .advanced_options = {.runtime_varargs = {{kWorkerNode,
            concat({device_idx, args.peer_count()},
                   route_args,
                   {static_cast<uint32_t>(barrier_sem.address()),
                    barrier_node.x, barrier_node.y})}}},
    }};
    run_args.tensor_args = {
        {kInputTensor,  std::cref(input_tensor)},
        {kOutputTensor, std::cref(output_tensor)},
    };
    m2::SetProgramRunArgs(program, run_args);

    return {std::move(program),
            shared_variables_t{barrier_sem, device_idx, std::move(route_args)}};
}
```



On a cache hit the tensors may have moved. Production patches addresses into
cached arg vectors by index; Metal 2.0 has no index to patch, since
`SetProgramRunArgs` is what binds them — so the override re-runs it:

```cpp
void PullAllGatherFactory::override_runtime_arguments(
    cached_mesh_workload_t& cached_workload,
    const PullAllGatherParams& /*args*/,
    const PullAllGatherInputs& tensor_args,
    Tensor& output_tensor) {
    namespace m2 = tt::tt_metal::experimental;

    for (auto& [coordinate_range, program] : cached_workload.workload.get_programs()) {
        const auto& shared_vars = cached_workload.shared_variables.at(coordinate_range);

        m2::ProgramRunArgs run_args;
        run_args.kernel_run_args = {{
            .kernel = m2::KernelSpecName{"pull_all_gather_sender"},
            .advanced_options = {.runtime_varargs = {{m2::NodeCoord{0, 0},
                concat({shared_vars.device_idx, peer_count},
                       shared_vars.route_args,
                       {static_cast<uint32_t>(shared_vars.barrier_sem.address()),
                        barrier_node.x, barrier_node.y})}}},
        }};
        run_args.tensor_args = {
            {m2::TensorParamName{"input_tensor"},  std::cref(tensor_args.input_tensor)},
            {m2::TensorParamName{"output_tensor"}, std::cref(output_tensor)},
        };
        m2::SetProgramRunArgs(program, run_args);
    }
}
```

The producer takes no runtime args — its partition comes from
`get_my_thread_id()` / `get_num_threads()`, so it never appears in the override.

Both kernels disable DFB implicit sync: the producer because it publishes with
explicit `push_back`, the sender because its page lifetime is owned by Fabric
transaction counters. That supersedes the "producer may keep its existing
synchronization mode" note in [DFB doc §10](Quasar-Pull-Fabric-DFB.md), which
holds only when one NoC read fills exactly one entry.

## Delta against today's `fabric_pull.hpp`

This is target design. The kernels follow **DFB doc §2.3 / §2.4** — a
caller-owned `FabricRequestRef` plus typed route args — and do not compile
against the current, still eth-shaped header:


| Today (`fabric_pull.hpp`)                                                           | Design (§2.3)                                                         | Why                                                                  |
| ----------------------------------------------------------------------------------- | --------------------------------------------------------------------- | -------------------------------------------------------------------- |
| `Fabric(const Noc&, RoutingPlaneConnectionManager&)`                                | `Fabric fabric;`                                                      | nothing to connect to                                                |
| `set_async_write_multicast_state(uint8_t connection, uint8_t start, uint8_t range)` | `(FabricRequestRef<topology>, const FabricMcastRouteArgs<topology>&)` | route is one typed value covering every direction                    |
| internal header pool (`ensure_header`)                                              | caller-owned request set                                              | `pull.num_swqs`, `pull.txn_id`, `pull.flags` travel with each header |
| one header per call                                                                 | `FabricPullRequestSet` — one slot per route                           | a 1D stream cannot fork; 2D forks only within one spine              |
| `dst_dev_id` / `dst_mesh_id` from `connection_manager.get(i)`                       | per-route in the route args                                           | there is no connection manager                                       |
| `FabricDataflowBuffer(DFBAccessor, const ScratchpadAccessor&)`                      | `FabricDataflowBuffer(DFBAccessor)`                                   | counter bank is reserved L1                                          |
| `local_writes_barrier()`, `flush()`, `close()`                                      | none of the three                                                     | §2.4's teardown is `payload.finish()` alone                          |


The three teardown methods are tempting to carry over and none belong: §2.5
already flushes the posted local write per entry; `payload.finish()` is
strictly stronger than `flush()` (a counter cannot reach zero until the DE
consumed the request and every SWQ completed); and nothing was opened to close.

Multicast atomic-inc set-state/with-state does not exist in §2.3 at all — it is
[DFB doc §8](Quasar-Pull-Fabric-DFB.md) gap item 6. The barrier below assumes
it in the same request/route shape, plus one requirement §2.3 does not state:
**an update mask with a Flush bit**, matching
`UnicastAtomicIncUpdateMask::Val | UnicastAtomicIncUpdateMask::Flush` in
`kernels/multicast_common.hpp`. Flush is what makes the barrier correct — the
semaphore is on the mirror worker core while the payload went to shard cores,
so the increment is ordered behind the data only because the receiver drains
its preceding NoC writes first.

**Open against Yukon.** On eth the receiving EDM issues those writes and can
drain them. Yukon inserts a hop — DMA → peer RX slot → DE forward → destination
Tensix (Yukon doc §3) — and whether the DE forwards every payload to its
destination core before a following atomic is specified nowhere. Recorded as
Yukon doc §9 item 5.

## kernels/producer.cpp

`num_producers` SPMD threads. Each fills its strided subset of DFB entries,
packing `tiles_per_entry` tiles into each one straight from DRAM through the
`TensorAccessor`.

```cpp
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/kernel_thread_globals.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    constexpr uint32_t entries_per_producer = get_arg(args::entries_per_producer);
    constexpr uint32_t entries_per_device = get_arg(args::entries_per_device);
    constexpr uint32_t num_producers = get_arg(args::num_producers);
    constexpr uint32_t tiles_per_entry = get_arg(args::tiles_per_entry);
    constexpr uint32_t tile_bytes = get_arg(args::tile_bytes);

    DataflowBuffer payload(dfb::payload);
    const auto input_tensor = TensorAccessor(tensor::input_tensor);
    Noc noc;

    ASSERT(payload.get_entry_size() == tiles_per_entry * tile_bytes);

    const uint32_t producer_idx = get_my_thread_id();

    // Count this producer's own entries; stride the entry index.
    // (Same shape as tests/.../test_kernels/dataflow/dfb_producer.cpp.)
    for (uint32_t k = 0; k < entries_per_producer; ++k) {
        // STRIDED: producer i owns entries i, i+P, i+2P, ...
        // Reading the matching input tiles makes the consumer's FIFO order
        // equal tile order.
        const uint32_t entry = k * num_producers + producer_idx;
        // entries_per_device need not be a multiple of num_producers.
        if (entry >= entries_per_device) {
            break;
        }

        payload.reserve_back(1);

        const uint32_t first_tile = entry * tiles_per_entry;
        for (uint32_t t = 0; t < tiles_per_entry; ++t) {
            noc.async_read(
                input_tensor,
                payload,                            // dst_addr = get_write_ptr() + offset_bytes
                tile_bytes,                         // one tile per read
                {.page_id = first_tile + t},
                {.offset_bytes = t * tile_bytes});  // pack into the entry
        }

        noc.async_read_barrier();   // push_back bumps `posted` immediately; reads must have landed
        payload.push_back(1);
    }

    payload.finish();
}
```

Load-bearing: no `NocOptions::TXN_ID` (a tagged read would also bump `posted`
via the implicit-sync ISR, double-counting the entry); the explicit barrier
before `push_back`, which bumps `posted` unconditionally without checking that
any read landed; and `{.offset_bytes = ...}` on the DFB destination, which is
how several tiles land in one entry.

## kernels/sender.cpp

One DM thread. It touches no DFB synchronization — `wait_front`,
`get_read_ptr`, `advance_read_ptr`, `acknowledge_front` all run inside
`Fabric::async_write_impl` and `FabricDataflowBuffer::try_complete_front_txn` —
opens no connection, and computes no route.

```cpp
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/fabric_dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"
#include "tt_metal/fabric/hw/inc/fabric_pull.hpp"
#include "tt_metal/fabric/hw/inc/linear/addrgen_api.h"
#include "tt_metal/fabric/hw/inc/noc_addr.h"

// Reads the route arg block and builds the multicast route args.
//
// Two groups of ten words. Production hands group 0 to the reader (E-line,
// S-rect) and group 1 to the writer (W-line, N-rect); a pull sender owns both,
// so this is multicast_reader.cpp:95-120 run twice. The host has already
// resolved which hop count lands in which slot, so no cardinal direction is
// named here.
//
//   per group: line_hops, rect_e_hops, rect_w_hops, rect_spine_hops,
//              line_dir,  rect_e_dir,  rect_w_dir,  rect_spine_dir,
//              dst_dev_id, dst_mesh_id      (last two unused in 1D)
template <tt::tt_fabric::Topology topology>
FabricMcastRouteArgs<topology> build_mcast_route(std::size_t& arg_idx, bool include_self) {
    constexpr uint32_t kNumGroups = 2;

    FabricMcastRouteArgs<topology> route{};
    uint32_t r = 0;

    // make_fabric_range() already collapses the dimensional difference
    // (multicast_common.hpp:13/19): a MeshMcastRange in 2D, the bare hop count
    // in 1D, where exactly one slot is nonzero. Only the destination fields
    // differ, so that is all the branch covers.
    auto add_route = [&](const uint8_t (&h)[4], uint8_t port,
                         uint16_t dst_dev_id, uint16_t dst_mesh_id)    {
        if constexpr (tt::tt_fabric::is_2D_topology(topology)) {
            route.routes[r++] = {make_fabric_range(h[0], h[1], h[2], h[3]),
                                 dst_dev_id, dst_mesh_id, port};
        } else {
            route.routes[r++] = {make_fabric_range(h[0], h[1], h[2], h[3]), port};
        }
    };

    auto next = [&]() { return static_cast<uint8_t>(get_arg_val<uint32_t>(arg_idx++)); };

    for (uint32_t g = 0; g < kNumGroups; ++g) {
        const uint8_t line_hops       = next();
        const uint8_t rect_e_hops     = next();
        const uint8_t rect_w_hops     = next();
        const uint8_t rect_spine_hops = next();
        const uint8_t line_dir        = next();
        const uint8_t rect_e_dir      = next();
        const uint8_t rect_w_dir      = next();
        const uint8_t rect_spine_dir  = next();
        const uint16_t dst_dev_id  = static_cast<uint16_t>(get_arg_val<uint32_t>(arg_idx++));
        const uint16_t dst_mesh_id = static_cast<uint16_t>(get_arg_val<uint32_t>(arg_idx++));

        if (line_hops > 0) {
            uint8_t h[4] = {};
            h[line_dir] = line_hops;
            add_route(h, line_dir, dst_dev_id, dst_mesh_id);
        }
        // 1D leaves rect_spine_hops at zero, so this never fires there.
        if (rect_spine_hops > 0) {
            uint8_t h[4] = {};
            if (rect_e_hops > 0) { h[rect_e_dir] = rect_e_hops; }
            if (rect_w_hops > 0) { h[rect_w_dir] = rect_w_hops; }
            h[rect_spine_dir] = rect_spine_hops;
            add_route(h, rect_spine_dir, dst_dev_id, dst_mesh_id);
        }
    }

    route.num_routes = r;
    route.include_self = include_self;
    ASSERT(r > 0);
    return route;
}

void kernel_main() {
    constexpr uint32_t entries_per_device = get_arg(args::entries_per_device);
    constexpr uint32_t tiles_per_entry = get_arg(args::tiles_per_entry);
    constexpr uint32_t in_tile_cols = get_arg(args::in_tile_cols);    // N_t, this device's band
    constexpr uint32_t out_tile_cols = get_arg(args::out_tile_cols);  // N_t * num_devices

    // Guaranteed by the host: an entry never crosses a stripe boundary.
    static_assert(in_tile_cols % tiles_per_entry == 0);
    constexpr uint32_t entries_per_stripe = in_tile_cols / tiles_per_entry;

    //   device_idx, num_peers | 2 groups * 10 route words | sem addr, x, y
    std::size_t runtime_arg_index = 0;
    const uint32_t device_idx = get_arg_val<uint32_t>(runtime_arg_index++);
    const uint32_t num_peers = get_arg_val<uint32_t>(runtime_arg_index++);

    // Chip multicast excludes the source chip in every direction. An all-gather
    // needs its own band in its own replica too, hence include_self.
    const auto route = build_mcast_route<topology>(runtime_arg_index, /*include_self=*/true);

    const uint32_t barrier_sem_address = get_arg_val<uint32_t>(runtime_arg_index++);
    const uint8_t barrier_sem_noc_x = static_cast<uint8_t>(get_arg_val<uint32_t>(runtime_arg_index++));
    const uint8_t barrier_sem_noc_y = static_cast<uint8_t>(get_arg_val<uint32_t>(runtime_arg_index++));

    // num_peers is the barrier fan-in, not M. M is the route count: the router
    // issues one packet per direction and the chain forwards it.
    ASSERT(num_peers > 0);

    // One request *set* per distinct packet state: the payload multicast and
    // the header-only completion atomic. A set holds one slot per route, since
    // each route needs its own packet header (DFB doc §2.1).
    using RequestSet =
        FabricPullRequestSet<PACKET_HEADER_TYPE, fabric_max_routes<topology>>;
    Scratchpad<volatile RequestSet> requests(scratch::fabric_requests);
    auto data_request = requests.local_mem() + 0;
    auto barrier_request = requests.local_mem() + 1;

    FabricDataflowBuffer payload(dfb::payload);   // counters come from reserved L1
    const auto output_tensor = TensorAccessor(tensor::output_tensor);
    Noc noc;

    // No connection is opened. JIT bindings initialize the internal
    // WorkerToFabricEdmSender with the local DE's coordinates, request-ring
    // base/depth, and credit addresses (DFB doc §2.4).
    Fabric fabric;

    // Claims one request slot per route, each with its own packet header, and
    // records include_self and the outgoing port in pull (DFB doc §2.4).
    //
    // The local copy reuses the same remote_noc_addr as a posted NoC write on
    // NOC_UNICAST_WRITE_VC + 1, and is not counted in M. That works only
    // because the output is fully replicated with an identical layout on every
    // device, so a page id names the same (core, offset) locally and on every
    // peer. Replicas with differing shard specs would need a separately
    // supplied local address, which the API does not carry.
    fabric.set_async_write_multicast_state(data_request, route);

    // Last-dim gather: this device owns a column band, repeated on every tile
    // row. Column offset of the band is fixed; the row advances per stripe.
    const uint32_t band_offset = device_idx * in_tile_cols;

    for (uint32_t entry = 0; entry < entries_per_device; ++entry) {
        // output_page(t) = r * out_tile_cols + device_idx * in_tile_cols + c
        // with t = entry * tiles_per_entry, so r and c come from the stripe split.
        const uint32_t tile_row = entry / entries_per_stripe;
        const uint32_t col_in_band = (entry % entries_per_stripe) * tiles_per_entry;
        const uint32_t output_page = tile_row * out_tile_cols + band_offset + col_in_band;

        // The entry's tiles are consecutive pages inside one stripe, and the
        // stripe is contiguous inside one shard, so this single address covers
        // the whole payload.
        const uint64_t output_noc_address =
            tt::tt_fabric::addrgen_detail::get_noc_address(output_tensor, output_page, 0);

        // One call, every route. Internally: wait_for_txn_id ->
        // wait_for_next_issue -> get_read_ptr -> prepare_transaction(M) ->
        // publish each claimed slot under the same txn id -> local posted write
        // (include_self) -> advance_read_ptr ONCE -> commit_transaction ->
        // try_complete_front_txn
        fabric.async_write_multicast_with_state(data_request, payload, output_noc_address);
    }

    // Completion atomic. Header-only, so it consumes no transaction ID. This is
    // just another request pushed into the DE send queue, behind every data
    // request; the DE drains that queue in order. Nothing here waits.
    //
    // Flush is load-bearing: the semaphore lives on the mirror worker core,
    // while the payload went to the output tensor's shard cores. Waiting on the
    // semaphore is only meaningful because Flush makes the *receiving* side
    // drain its preceding NoC writes before the increment lands.
    //
    // NOTE: multicast atomic-inc set-state/with-state is DFB doc §8 item 6 --
    // not yet in the design's public Fabric class, and §2.4 specifies no update
    // mask for it. The mask below is a requirement on that API.
    fabric.set_atomic_inc_multicast_state<
        UnicastAtomicIncUpdateMask::Val | UnicastAtomicIncUpdateMask::Flush>(
        barrier_request, route);
    fabric.atomic_inc_multicast_with_state(
        barrier_request,
        safe_get_noc_addr(barrier_sem_noc_x, barrier_sem_noc_y, barrier_sem_address, noc.get_noc_id()),
        /*value=*/1);

    // Teardown, not ordering: drains our transaction counters so no SWQ still
    // references a payload page. Issued after the atomic so the DE can start on
    // it while we spin. The source-local posted writes were already flushed
    // inside each async_write_multicast_with_state (DFB doc §2.6).
    payload.finish();

    // Every peer sends exactly one increment, so the fan-in is the peer count.
    auto* barrier_sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(barrier_sem_address);
    noc_semaphore_wait_min(barrier_sem, num_peers);
    noc_semaphore_inc(
        safe_get_noc_addr(barrier_sem_noc_x, barrier_sem_noc_y, barrier_sem_address),
        static_cast<uint32_t>(-static_cast<int32_t>(num_peers)));

    noc.async_write_barrier();
}
```



The payload size is `payload.get_entry_size()`; the sender never restates it.

`payload.finish()` does **not** order the atomic behind the data, and does not
need to precede it. The atomic is a request enqueued in the DE send queue like
any other; ordering comes from the DE draining that queue in order, exactly as
production relies on ("the sem is sent after all data sends on a particular
link, so it's correctly ordered at the receiver").

`finish()` is teardown: it drains this device's transaction counters so that no
SWQ still references a payload page when the kernel exits. Note the counters
track *terminal source-read completions* (DFB doc §1, §6) — "the DMA is done
reading my L1" — not delivery to the peer's Tensix, which Yukon §3 splits into
a separate RX-slot completion.

The barrier decrements rather than resetting to zero, matching
`multicast_reader.cpp` — increments from other phases must be preserved.

## End-to-end credit path

```text
producer thread p          payload DFB (entry = tiles_per_entry tiles)   sender DM
     |                                                                       |
 reserve_back(1)                                                             |
 async_read(tile kT)   --> entry k, offset 0                                  |
 async_read(tile kT+1) --> entry k, offset tile_bytes                         |
 ...                                                                         |
 async_read_barrier()                                                        |
 push_back(1) --- posted++ -------------------------> wait_front(required_occupancy)
                                                      get_read_ptr()
                                                      prepare_transaction(M = fwd+bwd)
                                                      publish request to DE
                                                      local posted NoC write
                                                      advance_read_ptr()   (no credit)
                                                          |
                  DE -> M = fwd+bwd SWQs -> DMA pulls the entry from worker L1
                  each terminal completion: counters_[txn_id]--
                                                          |
                            counters_[front] == 0 -> acknowledge_front()
 free space opens <---- acked++ <-------------------------------+
```

The producer's credit returns only after every peer DMA has finished reading
that entry. No NoC TRID is involved anywhere on this DFB.

## Mapping to the production factory


| This example                                                                                              | Production                                                                     |
| --------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------ |
| `PullAllGatherDeviceOperation::compute_output_specs` / `create_output_tensors` / `select_program_factory` | `all_gather_device_operation.cpp:89` / `:119` / `:222`                         |
| `PullAllGatherFactory::create_mesh_workload`                                                              | `all_gather_multicast_factory.cpp:16-57`                                       |
| `::create_at` sizing                                                                                      | `create_at:78-150` — route derivation replaced by `num_devices - 1`            |
| `::create_at` run args                                                                                    | `create_at:373-501` — per-link loop collapsed to one node                      |
| `::override_runtime_arguments`                                                                            | `:513-538` — re-runs `SetProgramRunArgs` instead of patching by index          |
| stripe page map                                                                                           | `create_at:291-307` + `next_output_chunk` in both kernels                      |
| producer / sender kernels                                                                                 | `multicast_reader.cpp` read half / `multicast_writer.cpp` + reader's send half |


Not carried over: the per-link partition across `min_num_links` worker cores
(one node here, so no `axis_num_links`); the reader/writer E-S / W-N split
(one DE, nothing to split); and `FabricWriter`'s scatter batching and route
alternation (the pull API has neither — [DFB doc §8](Quasar-Pull-Fabric-DFB.md)).
