# Pull-Fabric All-Gather: Worked Example

**Status:** Reference example for the target design.

**Related:** [Quasar Pull Fabric: DFB Transaction IDs](Quasar-Pull-Fabric-DFB.md)

**Production reference this mirrors:**
`ttnn/cpp/ttnn/operations/ccl/all_gather/device/all_gather_multicast_factory.cpp`

---

## What it does

All-gather on dim **-2**, the row dim:

```text
input   (1, 1, M, N)                  L1 height sharded, ragged last shard OK
output  (1, 1, M * num_devices, N)    L1 height sharded, fully replicated

L1 input shard   --[N producer DM threads, TensorAccessor]--> payload DFB
payload DFB      --[1 sender DM, pull Fabric multicast]-----> this device's shard,
                                                              on every device
```

Because the gather is on the row dim, a device's data is **one contiguous page
range** of the output, `[device_idx * tiles_per_device, ...)`, so the page map
is a base plus a counter — see "Shapes and the output page map".

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
| No NoC TRIDs on the **sender** side                 | The sender's only NoC transaction is the header write; the payload is pulled later by the remote DMA. A TRID would mean "header delivered", not "payload read", so the sender binding disables implicit sync.                   |
| Producer uses implicit TRID sync                    | The DFB-destination overload of `Noc::async_read` is `enable_if`'d on `NocOptions::TXN_ID` (`noc.h:766`); the ISR bumps `posted` when the read lands, so there is no `reserve_back` / `push_back` / `async_read_barrier`.        |
| One read fills one whole entry                      | That overload issues a single `noc_async_read` of `get_entry_size()` bytes from one source address (`dataflow_buffer.inl:579`), so the entry may hold many tiles — but the **source** must be contiguous across them.           |
| Input is L1 sharded, not interleaved                | Interleaved round-robins page *N+1* into another bank, so an `entry_size` read from page *N* would pull the wrong data. Sharding makes the run contiguous, exactly as it does on the output side.                               |
| Sender never calls `wait_front` / `pop_front`       | Those live inside `Fabric` / `FabricDataflowBuffer`; DFB sync is implicit in the Fabric API.                                                                                                                                    |
| N producer threads are STRIDED                      | All DFB producers are STRIDED (`kernel_spec.hpp:244`). Thread `p` owns entries `p, p+N, ...`, so filling them from the matching tiles makes the sender's FIFO order equal tile order.                                           |




## How many routes a topology needs

The fabric router carries no topology knowledge: it walks a positional per-hop
command stream, terminating where a hop is `WRITE_ONLY`
(`fabric_erisc_router.cpp:701`) and advancing by shifting that stream
(`fabric_edm_packet_transmission.hpp:378`). `LowLatencyRoutingFields` is a bare
`value` with no branch offsets, so a stream cannot fork, and one long enough to
reach a whole line would leave its terminator past the end of the chain
(`encode_1d_multicast`, `fabric_common.h:291`). One stream cannot both stop at
`fwd_hops` and cover the chips behind the sender.

So: one route per direction, each with its own packet header. Hop counts come
from the existing host API (`ccl_common.cpp:1849`):

```cpp
auto [fwd_hops, bwd_hops] = ::ttnn::ccl::get_forward_backward_line_mcast_distance(
    num_devices, device_idx, topology, /*static_alternate=*/false);

// Ring, size 8    -> (4, 3) for every index
// Linear, index i -> (size - 1 - i, i)
```

A zero hop count contributes no route, so a Linear endpoint fills one route
instead of two — no special case. `static_alternate` swaps the pair on even
indices, the ring load balancing production gets from prebuilt routes.

`M` is the **route** count, not `fwd_hops + bwd_hops`: the chain
store-and-forwards, so the sender's L1 is read once per direction and `M = 2`
whatever the hop counts. `fwd_hops + bwd_hops = num_devices - 1` is the *peer*
count, the completion barrier's fan-in. All routes go out in one
`async_write_multicast_with_state()` — one transaction id, one read-pointer
advance, one transaction of `M` (DFB doc §1). Issuing directions separately
would consume one DFB entry each.

**2D needs up to four routes.** A mesh range forks E/W within its own spine but
cannot reach the opposite one, so a mid-mesh device needs E-line, W-line,
N-rect, S-rect, and `mesh/api.h:1305` reads `dst_dev_id` / `dst_mesh_id` per
route from `connection_manager.get(i)`. With no connection manager those move
into the route args (`FabricRouteArgs<..., CHIP_MULTICAST, true>::Route`) — the
shape `build_mcast_route()` fills in the sender kernel below.
`fabric_max_routes<topology>` sizes both the route array and the request set, so
they cannot disagree. It lives in `fabric_edm_types.hpp` rather than the pull
header, because the host needs it too — to size the scratchpad and the sender's
runtime-arg count — and `fabric_pull.hpp` is `ARCH_QUASAR`-gated kernel code the
host cannot include.

**All-to-all needs one route, and it is the cheap case.** With every peer a
direct neighbour and nothing forwarding — `Topology::AllToAll` — there is no
per-hop stream to build. (Not `NeighborExchange`: that one is 1D and reaches
only the two adjacent chips, `channel_trimming_report.cpp:45`.) The routing collapses to naming target queues, so one
header carrying a **peer mask** covers every peer and the DE expands it:

```cpp
template <> inline constexpr uint32_t fabric_max_routes<Topology::AllToAll> = 1;

// A route's shape follows from how the topology delivers, so McastRoute is an
// alias keyed on the topology: Linear and Ring resolve to the line shape, Mesh
// and Torus to the rect shape, AllToAll to neither.
struct PeerRoute {
    uint32_t peer_mask;   // no range, no direction: every peer is one hop
};
```

The mask rides in the routing dword itself. `LowLatencyRoutingFieldsT<0>` is a
bare `uint32_t value` holding 16 hops at 2 bits each; with no hops to encode it
carries one bit per peer instead, so 32 peers fit without touching the header
layout. [DFB doc §2.3.1](Quasar-Pull-Fabric-DFB.md) has the encoding and the DE
expansion.

That is the encoding a line cannot use. There, expanding in the DE would mean
synthesising a routing stream per direction — `fabric_set_mcast_route()` reads
`routing_l1_info_t`, computes spine hops and branch offsets, and may fall into
`fabric_set_unicast_route()`'s table walk. Set-state exists to pay that once per
route; moving it into the DE would pay it per send, on the shared resource. A
star has nothing to encode, so the mask is free there and only there.

**What it costs.** `M` stops equalling the header count:

```text
              headers published    source reads owed (M)
line          2                    2
mesh          up to 4              up to 4
all-to-all    1                    num_peers
```

A chain amortises: downstream chips write from what they received and never
re-read the sender's L1. A star has no chain, so every peer's DMA pulls the page
itself. You win DE send-queue slots — one instead of `num_peers`, the shared
resource [DFB doc §8](Quasar-Pull-Fabric-DFB.md) item 4 flags — and win nothing
on source bandwidth. That is why `FabricPullRequestSet` carries
`source_read_completions` separately from `used`, and why
`prepare_transaction()` takes a count rather than deriving one.

### What changes in this op

Everything outside route derivation is untouched — same DFB, same chunk walk,
same page map, same barrier.

| | 1D / 2D | all-to-all |
| --- | --- | --- |
| host derivation | `get_forward_backward_line_mcast_distance()`, four hop counts, four direction slots, per-route anchors | one mask word: which peers to send to |
| runtime args | `num_routes` + `kMaxRoutes * 7` route words | one mask word |
| `build_mcast_route()` | unpacks a block per route | reads the mask |
| `kMaxRoutes` | 2 or 4 | 1 — request scratchpad halves or quarters |
| coverage assert | `(e+w) + (n+s)*(e+w+1) == peer_count()` | `peers_count(mask) == peer_count()` |
| `source_read_completions` | `num_routes` | `peers_count(mask)` |

For an all-gather the mask is simply every peer, so the host derivation reduces
to a constant and `create_at()` loses its whole axis loop.

Still needed before this runs: `to_chip_peer_multicast()` on the packet header,
a per-device node-id-to-target-queue table for the DE, and the expansion of the
mask into one SWQ per set bit, all stamped with the same `transaction_id`. The
mask names fabric nodes, not queues, so that translation is the DE's job. Note
the existing `SparseMulticastRoutingCommandHeader` is *not* it — its bits index
hop distance along a chain, and on a star every peer is at distance 1.

## Shapes and the output page map

Gathering on the row dim means a device's data is one contiguous block of the
output — whole tile rows, in local tile order:

```text
output tile grid, num_devices = 4, tile_cols = 3, tile_rows = 2

 tile row 0 | d0 d0 d0 |
 tile row 1 | d0 d0 d0 |   d0 owns pages [0, 6)
 tile row 2 | d1 d1 d1 |
 tile row 3 | d1 d1 d1 |   d1 owns pages [6, 12)
 tile row 4 | d2 d2 d2 |
 tile row 5 | d2 d2 d2 |
 tile row 6 | d3 d3 d3 |
 tile row 7 | d3 d3 d3 |
```

With row-major page numbering over the output tile grid, the map for local tile
index `t` in `[0, tiles_per_device)` is a base plus a counter — no division, no
iterator state:

```text
output_page(t) = device_idx * tiles_per_device + t
```



## Entry sizing

Both tensors are L1 height sharded, full width `N`, and for the same reason on
each side: a transfer has to be one contiguous run at both ends.

- **Source.** The implicit-TRID read issues one `noc_async_read` of
  `get_entry_size()` bytes from a single address (`dataflow_buffer.inl:579`).
- **Destination.** A `FWWriteDMADescriptor` is `(src, dest + BAR, size)` — one
  contiguous run.

Neither side needs a chunk to start on a tile: `src_args_type` and
`dst_args_type` both carry `offset_bytes` (`noc_traits.h:14-21`), and a height
shard is one contiguous byte range. So nothing is tile-quantised.

**A device owns several shards, and its last one may be ragged.** With
`shard_tile_rows` per shard, a block of `tile_rows` rows is cut into
`ceil(tile_rows / shard_tile_rows)` shards, the last holding whatever remains.
The output is a different tensor with its own cut, so the two sets of shard
boundaries do not generally coincide. A chunk must stay inside one shard on
*each* side, which makes the rule a three-way minimum:

```text
txn_bytes = min(max_payload_bytes,
                bytes to the end of the current input shard,
                bytes to the end of the current output shard)
```

Walking a byte cursor over the device block, both kernels derive the same
sequence from the same recurrence, so they agree by construction — no per-chunk
table has to be passed. The host runs the same walk once to get
`txns_per_device`; the DFB entry size is
`min(max_payload_bytes, in_shard_bytes, out_shard_bytes)`, the largest chunk the
rule can produce. A small shard on either side therefore caps the entry for the
whole run.

For that to work the sequence must not depend on which device is running, since
the producer takes no runtime args and so does not know `device_idx`. That holds
as long as the device block is a whole number of **output** shards, which is the
one constraint worth keeping: the block then starts on an output-shard boundary
and the output side of the minimum is just `cursor % out_shard_bytes`. The
**input** is free — any shard height, ragged last shard — because it is a
per-device tensor and its cut restarts at the block start anyway.

When the two shardings agree and divide evenly this collapses to a uniform
`max_payload_bytes` chunk with one short tail per shard, which is the common
case; the minimum is what keeps the ragged and misaligned cases correct without
constraining the caller's shard specs.

**The short chunks are what split the producer.** The implicit read has no size
argument — it always fetches `get_entry_size()` — so a chunk smaller than the
entry would over-read past the shard, into a different core. Full-size chunks
take the one-read TRID path; anything short takes explicit `reserve_back` /
sized read / `async_read_barrier` / `push_back`. The send side needs no such
split: `PayloadSize` is in the with-state mask and the size is a per-call
argument, so a short chunk is just a smaller `size_bytes`.

## There is no connection to open

The worker establishes nothing. Its send queue to the local DE is always there,
at a fixed address: one queue per worker, shared across that worker's peers
rather than one per connection. The destination rides in the packet header, not
in a per-peer worker connection. The send path is: wait for a free slot, write
the header into it, bump the DE's counter.

JIT bindings supply the DE coordinates, request-ring base/depth, and credit
addresses (DFB doc §2.3). Flow control survives:
`free = capacity - (wr_counter - router_free_credit)` needs only the credit
word's static address.

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
    int32_t dim = 0;                       // gather dim; this example requires -2, the row dim
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
#define MEM_FABRIC_TXN_COUNTERS_STRIDE  (MEM_FABRIC_MAX_TRANSACTION_IDS * 4)
#define MEM_FABRIC_TXN_COUNTERS_BASE(risc_id) \
    (MEM_FABRIC_TXN_COUNTERS_REGION + (risc_id) * MEM_FABRIC_TXN_COUNTERS_STRIDE)
```

Indexed by RISC id, so two sender DMs on one Tensix get separate banks with no
runtime state and nothing to bind. The op consumes what is there.

Consequences for this op:

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
    shape[args.dim] *= args.num_devices;          // (1,1,M,N) -> (1,1,M*num_devices,N)
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

    TT_FATAL(args.dim == static_cast<int32_t>(shape.rank()) - 2,
             "This pull all-gather gathers on the row dim (-2) only");
    TT_FATAL(input_spec.layout() == tt::tt_metal::Layout::TILE, "TILE layout required");

    // Sharded so that one DMA descriptor covers one entry: a descriptor is
    // (src, dest + BAR, size), one contiguous run.
    TT_FATAL(args.output_mem_config.buffer_type() == tt::tt_metal::BufferType::L1 &&
             args.output_mem_config.nd_shard_spec().has_value(),
             "Output must be L1 sharded");

    // The input is read by the implicit-TRID overload, which fetches
    // get_entry_size() bytes from one address, so its pages must be contiguous
    // across an entry -- same requirement the output has, for the same reason.
    TT_FATAL(input_spec.memory_config().buffer_type() == tt::tt_metal::BufferType::L1 &&
                 input_spec.memory_config().nd_shard_spec().has_value(),
             "Input must be L1 sharded; interleaved would put the next page in another bank");

    // Both height sharded: a shard's pages are one contiguous address run only
    // if the shard spans the full width. Nothing further is required of the two
    // shard specs -- they need not match, divide the block, or divide each
    // other, because the chunk rule takes the minimum of both remaining runs.
    // A device's last shard is allowed to be ragged.
    const auto& in_shard_shape = input_spec.memory_config().nd_shard_spec()->shard_shape;
    const auto& out_shard_shape = args.output_mem_config.nd_shard_spec()->shard_shape;
    const uint32_t tile_h = input_spec.tile().get_height();
    TT_FATAL(in_shard_shape.rank() >= 2 && out_shard_shape.rank() >= 2,
             "Shard shapes must have rank >= 2");
    TT_FATAL(in_shard_shape[-1] == shape[-1] && out_shard_shape[-1] == shape[-1],
             "Both must be height sharded: shard width must span the full N");
    TT_FATAL(in_shard_shape[-2] % tile_h == 0 && out_shard_shape[-2] % tile_h == 0,
             "Shard heights must be tile aligned");
    // The one alignment that matters: a device block is a whole number of
    // output shards, so the chunk walk is the same on every device and the
    // producer needs no device_idx. The input shard height is unconstrained --
    // its last shard per device may be ragged.
    TT_FATAL((shape[-2] / tile_h) % (out_shard_shape[-2] / tile_h) == 0,
             "Output shard tile-row count must divide the per-device tile-row count");

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

`validate` enforces sharded L1; the shard height stays the caller's choice, as
in any ttnn op, but the shard must span the full width and its tile-row count
must divide the per-device tile-row count, for the contiguity reason above.

### Program factory

```cpp
// pull_all_gather_factory.hpp
struct PullAllGatherFactory {
    struct shared_variables_t {
        tt::tt_metal::GlobalSemaphore barrier_sem;
        uint32_t device_idx = 0;          // which row block this device owns
        std::vector<uint32_t> route_args;   // num_routes + kMaxRoutes * 7 words
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
    const uint32_t rows = shape[shape.rank() - 2];   // M, the gather dim
    const uint32_t cols = shape[shape.rank() - 1];   // N
    TT_FATAL(rows % tile_h == 0 && cols % tile_w == 0, "Shape must be tile aligned");

    const uint32_t tile_rows        = rows / tile_h;                    // this device's block
    const uint32_t tile_cols        = cols / tile_w;
    const uint32_t tiles_per_device = tile_rows * tile_cols;

    // Both height sharded, so each shard is one contiguous byte range. The two
    // shard specs are independent: only the output's has to divide the block
    // (validate() checked it), and the input's last shard per block may be
    // ragged.
    const auto& in_shard_shape  = input_spec.memory_config().nd_shard_spec()->shard_shape;
    const auto& out_shard_shape = args.output_mem_config.nd_shard_spec()->shard_shape;
    const uint32_t in_shard_tiles  = (in_shard_shape[-2] / tile_h) * tile_cols;
    const uint32_t out_shard_tiles = (out_shard_shape[-2] / tile_h) * tile_cols;
    const uint32_t in_shard_bytes  = in_shard_tiles * tile_bytes;
    const uint32_t out_shard_bytes = out_shard_tiles * tile_bytes;
    const uint32_t block_bytes     = tiles_per_device * tile_bytes;

    // Entry = the largest chunk the rule can produce. Nothing is tile
    // quantised: a chunk is a byte range inside one shard on each side.
    const uint32_t bytes_per_dma_txn =
        std::min<size_t>(args.max_payload_bytes, std::min(in_shard_bytes, out_shard_bytes));

    // The same walk both kernels run, to get the chunk count.
    auto txn_bytes_at = [&](uint32_t cursor) {
        const uint32_t in_end =
            std::min((cursor / in_shard_bytes + 1) * in_shard_bytes, block_bytes);
        const uint32_t out_left = out_shard_bytes - (cursor % out_shard_bytes);
        return std::min(bytes_per_dma_txn, std::min(in_end - cursor, out_left));
    };
    uint32_t txns_per_device = 0;
    for (uint32_t cursor = 0; cursor < block_bytes; cursor += txn_bytes_at(cursor)) {
        ++txns_per_device;
    }

    // ---- ProgramSpec ----
    const m2::DFBSpecName        kPayloadDfb{"payload"};
    const m2::KernelSpecName     kProducer{"pull_all_gather_producer"};
    const m2::KernelSpecName     kSender{"pull_all_gather_sender"};
    const m2::TensorParamName    kInputTensor{"input_tensor"};
    const m2::TensorParamName    kOutputTensor{"output_tensor"};
    const m2::ScratchpadSpecName kFabricRequests{"fabric_requests"};
    constexpr m2::NodeCoord      kWorkerNode{0, 0};
    // How many request *sets*: one per packet state the sender keeps live --
    // the payload multicast and the completion atomic. Two NocSendTypes, two
    // sticky headers; set-state for one would clobber the other. Independent
    // of topology.
    constexpr uint32_t           kNumRequestSets = 2;
    // Slots *within* a set: one per route, since a route needs its own packet
    // header (DFB doc §2.1). 2 for 1D, 4 for 2D, so 4 and 8 slots in total.
    // Same expression the kernel's FabricPullRequestSet uses, so host and
    // kernel cannot disagree on the count.
    constexpr uint32_t           kMaxRoutes = tt::tt_fabric::fabric_max_routes<topology>;
    // One route block per slot, always kMaxRoutes of them so the vararg count
    // is fixed; blocks past num_routes are zero and never read.
    //   h[0..3], port, dst_dev_id, dst_mesh_id
    // A forwarding route is h[0..3] + port + dst_dev_id + dst_mesh_id; a peer
    // route is the mask alone.
    constexpr uint32_t           kRouteWords =
        tt::tt_fabric::is_forwarding_topology(topology) ? 7 : 1;
    // sizeof(FabricPullRequestSet<PACKET_HEADER_TYPE, kMaxRoutes>), which the
    // host cannot write directly: PACKET_HEADER_TYPE is a kernel-side define,
    // so only its size is available here. The helper lives beside the struct in
    // fabric_edm_types.hpp, so the layout is not spelled out twice.
    const uint32_t request_set_bytes = tt::tt_fabric::fabric_pull_request_set_bytes(
        static_cast<uint32_t>(tt::tt_fabric::get_tt_fabric_packet_header_size_bytes()),
        kMaxRoutes);
    // device_idx, num_peers | route block | sem addr, x, y. A forwarding
    // topology's block is num_routes plus the fixed kMaxRoutes route words; a
    // peer topology's is the mask alone, which is already the whole route.
    constexpr uint32_t           kRouteArgWords =
        tt::tt_fabric::is_forwarding_topology(topology) ? 1 + kMaxRoutes * kRouteWords : 1;
    constexpr uint32_t           kSenderRuntimeArgs = 2 + kRouteArgWords + 3;

    m2::KernelSpec producer{
        .unique_id = kProducer,
        .source = "pull_all_gather/kernels/producer.cpp",
        .num_threads = args.num_producers,        // <-- N producers
        // Implicit sync stays ON here: the producer wants the TRID path.
    };
    producer.dfb_bindings = {m2::ProducerOf(kPayloadDfb, "payload")};
    producer.tensor_bindings = {{
        .tensor_parameter_name = kInputTensor, .accessor_name = "input_tensor"}};
    // No per-producer count: each thread walks the whole chunk sequence and
    // acts on the entries it owns, so the stride is all it needs.
    producer.compile_time_args = {
        {"txns_per_device", txns_per_device},
        {"num_producers", args.num_producers},
        {"bytes_per_dma_txn", bytes_per_dma_txn},
        {"in_shard_bytes", in_shard_bytes},
        {"in_shard_tiles", in_shard_tiles},
        {"out_shard_bytes", out_shard_bytes},
        {"block_bytes", block_bytes},
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
        {"txns_per_device", txns_per_device},
        {"bytes_per_dma_txn", bytes_per_dma_txn},
        {"in_shard_bytes", in_shard_bytes},
        {"out_shard_bytes", out_shard_bytes},
        {"out_shard_tiles", out_shard_tiles},
        {"block_bytes", block_bytes},
        {"tiles_per_device", tiles_per_device},
    };
    sender.advanced_options.num_runtime_varargs = kSenderRuntimeArgs;

    const m2::ProgramSpec program_spec{
        .name = "pull_all_gather",
        .kernels = {producer, sender},
        .dataflow_buffers = {{
            .unique_id = kPayloadDfb,
            .entry_size = bytes_per_dma_txn,     // a DMA transfer, not a tensor page
            .num_entries = args.dfb_depth,
            .data_format_metadata = input_spec.data_format(),
        }},
        // The transaction-counter bank is reserved L1, so the only scratchpad
        // is the request ring (packet header + FabricPullMetadata per slot).
        // No fabric teardown / buffer-index semaphores: those persist eth
        // connection-handshake state, and nothing is opened here.
        .scratchpads = {{
            .unique_id = kFabricRequests,
            // sizeof(FabricPullRequestSet<PACKET_HEADER_TYPE, kMaxRoutes>),
            // which the host cannot name: the header type is a kernel-side
            // define and its size is queried. FabricPullRequest is alignas(16),
            // so the set is a padded route array plus the `used` counter --
            // dropping `used` would leave the kernel's `local_mem() + 1` stride
            // pointing past the allocation.
            .size_per_node = kNumRequestSets * request_set_bytes,
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
    std::vector<uint32_t> route_args;
    uint32_t num_routes = 0;

    if constexpr (!tt::tt_fabric::is_forwarding_topology(topology)) {
        // Every peer, named by fabric node id -- the mask means the same set of
        // devices whichever device sends it, which is why it names nodes rather
        // than the DE's queue indices. Our own bit stays clear: chip multicast
        // excludes the source, and our replica is include_self.
        const auto my_node = mesh_device->get_fabric_node_id(sender_device_coord);
        uint32_t peer_mask = 0;
        for (const auto& coord : ttnn::MeshCoordinateRange(mesh_device->shape())) {
            const auto node = mesh_device->get_fabric_node_id(coord);
            if (node == my_node) {
                continue;
            }
            TT_FATAL(
                static_cast<uint32_t>(node.chip_id) < 32,
                "Peer mask holds 32 nodes; node {} does not fit", node.chip_id);
            peer_mask |= (1u << static_cast<uint32_t>(node.chip_id));
        }
        TT_FATAL(
            static_cast<uint32_t>(std::popcount(peer_mask)) == args.peer_count(),
            "Mask must name every peer exactly once");

        route_args = {peer_mask};
        num_routes = 1;
    } else {
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
        // The two lines cover this device's own row; each rect covers `spine` rows
        // beyond it, every one of them (e + w + 1) wide. On 1D both spines are zero
        // and this reduces to e + w.
        TT_FATAL(
            (e_hops + w_hops) + (n_hops + s_hops) * (e_hops + w_hops + 1) == args.peer_count(),
            "Routes must cover every peer exactly once");

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
        // Not the route's final destination -- there is none for a multicast.
        // dst_dev_id / dst_mesh_id become packet_header->dst_start_node_id
        // (tt_fabric_api.h:157), the anchor the E/W/N/S hop counts extend from, so
        // it is the chip the packet enters on: this route's first hop. Per route,
        // not per direction pair -- mesh/api.h:1305 takes it from the connection
        // whose route it is building, and a connection is one eth link.
        auto dst_ids = [&](const std::optional<MeshCoordinate>& neighbor)
            -> std::pair<uint32_t, uint32_t> {
            if (!neighbor.has_value()) {
                return {0, 0};
            }
            const auto node = mesh_device->get_fabric_node_id(*neighbor);
            return {static_cast<uint32_t>(node.chip_id), static_cast<uint32_t>(*node.mesh_id)};
        };

        const uint32_t e_dir = physical_slot(e_coord), w_dir = physical_slot(w_coord);
        const uint32_t n_dir = physical_slot(n_coord), s_dir = physical_slot(s_coord);

        // Up to four routes: the E and W lines along this row, and the N and S
        // rects, each fanning E/W within its own spine. A zero hop count
        // contributes no route, so 1D fills two and a line endpoint fills one.
        auto add_route = [&](uint32_t spine_hops, uint32_t spine_dir, bool fan_out,
                             const std::optional<MeshCoordinate>& first_hop)    {
            if (spine_hops == 0) {
                return;
            }
            uint32_t h[4] = {};
            h[spine_dir] = spine_hops;
            if (fan_out) {
                if (e_hops > 0) { h[e_dir] = e_hops; }
                if (w_hops > 0) { h[w_dir] = w_hops; }
            }
            const auto [dst_dev, dst_mesh] = dst_ids(first_hop);
            route_args.insert(route_args.end(), {h[0], h[1], h[2], h[3], spine_dir, dst_dev, dst_mesh});
            ++num_routes;
        };
        add_route(e_hops, e_dir, /*fan_out=*/false, e_coord);
        add_route(w_hops, w_dir, /*fan_out=*/false, w_coord);
        add_route(s_hops, s_dir, /*fan_out=*/true,  s_coord);
        add_route(n_hops, n_dir, /*fan_out=*/true,  n_coord);

        TT_FATAL(num_routes > 0 && num_routes <= kMaxRoutes,
                 "Need 1..{} routes, derived {}", kMaxRoutes, num_routes);
        route_args.insert(route_args.begin(), num_routes);
        route_args.resize(1 + kMaxRoutes * kRouteWords, 0);   // fixed vararg count
    }

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
    const PullAllGatherParams& args,
    const PullAllGatherInputs& tensor_args,
    Tensor& output_tensor) {
    namespace m2 = tt::tt_metal::experimental;

    auto* mesh_device = tensor_args.input_tensor.device();
    const auto barrier_node = mesh_device->worker_core_from_logical_core(m2::NodeCoord{0, 0});

    for (auto& [coordinate_range, program] : cached_workload.workload.get_programs()) {
        const auto& shared_vars = cached_workload.shared_variables.at(coordinate_range);

        m2::ProgramRunArgs run_args;
        run_args.kernel_run_args = {{
            .kernel = m2::KernelSpecName{"pull_all_gather_sender"},
            .advanced_options = {.runtime_varargs = {{m2::NodeCoord{0, 0},
                concat({shared_vars.device_idx, args.peer_count()},
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

Only the **sender** disables DFB implicit sync, because its page lifetime is
owned by Fabric transaction counters rather than by a NoC read. The producer
keeps it on: one NoC read fills exactly one entry here, which is the condition
[DFB doc §10](Quasar-Pull-Fabric-DFB.md) attaches to "producer may keep its
existing synchronization mode".

## Teardown and the barrier

Three teardown methods are tempting to carry over and none belong:
`async_write_multicast_with_state()` already flushes the posted local write per
entry; the destructor's `finish()` is strictly stronger than `flush()` (a counter cannot
reach zero until the DE consumed the request and every SWQ completed); and
nothing was opened to close.

The barrier uses multicast atomic-inc in the same request/route shape as the
payload send, and passes `flush = true`. Flush is what makes *this* barrier
correct: the semaphore is on the mirror worker core while the payload went to
the shard cores, so the increment is ordered behind the data only if the
receiver drains its preceding NoC writes first. It is a caller argument rather
than a fixed part of the call because it costs the receiver that drain, and an
increment that orders nothing — a counter nobody reads against payload — should
not pay for it.

**Open.** On eth the receiving EDM issues those writes and can drain them. The
pull path inserts a hop — DMA → peer RX slot → DE forward → destination
Tensix — and whether the DE forwards every payload to its destination core
before a following atomic is unspecified.

## The chunk walk

Both kernels share this, so they cut the block identically. It reads the
compile-time constants directly, so in each kernel it sits at file scope just
below them:

```cpp
// Bytes this chunk may carry, starting `cursor` bytes into the device block.
// Three-way minimum: the packet cap, what is left of the current input shard
// (whose last one per block may be ragged), and what is left of the current
// output shard (uniform, because the block is a whole number of them).
constexpr uint32_t txn_bytes_at(uint32_t cursor) {
    const uint32_t in_end =
        std::min((cursor / in_shard_bytes + 1) * in_shard_bytes, block_bytes);
    const uint32_t out_left = out_shard_bytes - (cursor % out_shard_bytes);
    return std::min(bytes_per_dma_txn, std::min(in_end - cursor, out_left));
}
```

## kernels/producer.cpp

`num_producers` SPMD threads. Each walks the whole chunk sequence and fills the
entries it owns from the sharded L1 input — one read per entry for a full
chunk, and the explicit path for a short one.

```cpp
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/kernel_thread_globals.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

constexpr uint32_t txns_per_device = get_arg(args::txns_per_device);
constexpr uint32_t num_producers = get_arg(args::num_producers);
constexpr uint32_t bytes_per_dma_txn = get_arg(args::bytes_per_dma_txn);
constexpr uint32_t in_shard_bytes = get_arg(args::in_shard_bytes);
constexpr uint32_t in_shard_tiles = get_arg(args::in_shard_tiles);
constexpr uint32_t out_shard_bytes = get_arg(args::out_shard_bytes);
constexpr uint32_t block_bytes = get_arg(args::block_bytes);

// ... txn_bytes_at() from the section above ...

void kernel_main() {
    DataflowBuffer payload(dfb::payload);
    const auto input_tensor = TensorAccessor(tensor::input_tensor);
    Noc noc;

    ASSERT(payload.get_entry_size() == bytes_per_dma_txn);

    const uint32_t producer_idx = get_my_thread_id();

    // Walk every chunk; act on the ones this thread owns. STRIDED requires
    // producer i to fill entries i, i+P, i+2P, ... in order, which this does.
    // The skipped iterations are a few arithmetic ops -- cheaper than passing a
    // per-entry table.
    uint32_t cursor = 0;
    for (uint32_t entry = 0; entry < txns_per_device; ++entry) {
        const uint32_t size = txn_bytes_at(cursor);

        if (entry % num_producers == producer_idx) {
            // A chunk is a byte range inside one input shard: page at that
            // shard's start, plus an offset. Nothing is tile-quantised.
            const uint32_t shard = cursor / in_shard_bytes;
            const uint32_t page_id = shard * in_shard_tiles;
            const uint32_t offset_bytes = cursor - shard * in_shard_bytes;

            if (size == bytes_per_dma_txn) {
                // One read fills the whole entry. The overload issues a single
                // noc_async_read of get_entry_size() bytes from this address,
                // and the chunk is contiguous inside one shard. The ISR bumps
                // `posted` when it lands -- no reserve_back, no push_back, no
                // barrier, and the credit is tied to the read completing rather
                // than to the thread reaching a barrier.
                noc.async_read<NocOptions::TXN_ID>(
                    input_tensor,
                    payload,
                    {.page_id = page_id, .offset_bytes = offset_bytes});
            } else {
                // Short chunk. The implicit overload has no size argument -- it
                // would over-read past the shard, into a different core -- so
                // this one goes the explicit way. The read must stay untagged:
                // a tagged one would also bump `posted` through the ISR and
                // double-count against the push_back below.
                payload.reserve_back(1);
                noc.async_read(
                    input_tensor,
                    payload,
                    size,
                    {.page_id = page_id, .offset_bytes = offset_bytes});
                noc.async_read_barrier();
                payload.push_back(1);
            }
        }

        cursor += size;
    }
}
```

Load-bearing: `NocOptions::TXN_ID` selects the DFB-destination overload
(`noc.h:766`), which takes neither a size nor a destination offset — the DFB
supplies both. That is what ties the entry size to the read size, and why only
short chunks need the explicit path.

No `finish()`: the producer holds no transactions, and its credits are the
ordinary DFB kind.

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
// The host has already decided how many routes this device needs and what goes
// in each, so this only unpacks. One block per route:
//
//   h[0], h[1], h[2], h[3], port, dst_dev_id, dst_mesh_id
//
// `h` is indexed by eth_chan_directions, so the host names the cardinal
// directions and the kernel never does. There are always kMaxRoutes blocks; the
// ones past num_routes are zero and are not read.
template <tt::tt_fabric::Topology topology>
FabricMcastRouteArgs<topology> build_mcast_route(std::size_t& arg_idx, bool include_self) {
    FabricMcastRouteArgs<topology> route{};

    if constexpr (!tt::tt_fabric::is_forwarding_topology(topology)) {
        // One word, and it is the whole route: which fabric nodes to deliver
        // to. No hop counts, no directions, no anchors -- every peer is one hop.
        route.routes[0].peer_mask = get_arg_val<uint32_t>(arg_idx++);
        route.num_routes = 1;
        route.include_self = include_self;
        return route;
    }

    const uint32_t num_routes = get_arg_val<uint32_t>(arg_idx++);
    ASSERT(num_routes > 0 && num_routes <= fabric_max_routes<topology>);

    auto next = [&]() { return static_cast<uint8_t>(get_arg_val<uint32_t>(arg_idx++)); };

    for (uint32_t r = 0; r < num_routes; ++r) {
        const uint8_t h0 = next(), h1 = next(), h2 = next(), h3 = next();
        const uint8_t port = next();
        const uint16_t dst_dev_id  = static_cast<uint16_t>(get_arg_val<uint32_t>(arg_idx++));
        const uint16_t dst_mesh_id = static_cast<uint16_t>(get_arg_val<uint32_t>(arg_idx++));

        // make_fabric_range() already collapses the dimensional difference
        // (fabric_pull.hpp): a MeshMcastRange in 2D, the bare hop
        // count in 1D, where exactly one slot is nonzero. Only the destination
        // fields differ, so that is all the branch covers.
        if constexpr (tt::tt_fabric::is_2D_topology(topology)) {
            route.routes[r] = {make_fabric_range(h0, h1, h2, h3), dst_dev_id, dst_mesh_id, port};
        } else {
            route.routes[r] = {make_fabric_range(h0, h1, h2, h3), port};
        }
    }
    // Skip the zeroed tail so the caller's arg_idx lands on the next field.
    arg_idx += (fabric_max_routes<topology> - num_routes) * 7;

    route.num_routes = num_routes;
    route.include_self = include_self;
    return route;
}

constexpr uint32_t txns_per_device = get_arg(args::txns_per_device);
constexpr uint32_t bytes_per_dma_txn = get_arg(args::bytes_per_dma_txn);
constexpr uint32_t in_shard_bytes = get_arg(args::in_shard_bytes);
constexpr uint32_t out_shard_bytes = get_arg(args::out_shard_bytes);
constexpr uint32_t out_shard_tiles = get_arg(args::out_shard_tiles);
constexpr uint32_t tiles_per_device = get_arg(args::tiles_per_device);
constexpr uint32_t block_bytes = get_arg(args::block_bytes);

// The block is a whole number of output shards, which is what makes the chunk
// walk identical on every device.
static_assert(block_bytes % out_shard_bytes == 0);

// ... txn_bytes_at() from the section above ...

void kernel_main() {
    //   device_idx, num_peers | num_routes | kMaxRoutes * 7 route words | sem addr, x, y
    std::size_t runtime_arg_index = 0;
    const uint32_t device_idx = get_arg_val<uint32_t>(runtime_arg_index++);
    const uint32_t num_peers = get_arg_val<uint32_t>(runtime_arg_index++);

    // Chip multicast excludes the source chip in every direction. An all-gather
    // needs its own block in its own replica too, hence include_self.
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
    auto* data_request = requests.local_mem() + 0;
    auto* barrier_request = requests.local_mem() + 1;

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

    // Row-dim gather: this device owns one contiguous page range of the output,
    // in local tile order, so output_page(t) = base_page + t.
    const uint32_t base_page = device_idx * tiles_per_device;

    // Same walk the producer ran, so entry N here is the chunk the producer put
    // in entry N. The cursor is bytes into this device's block.
    uint32_t cursor = 0;
    for (uint32_t entry = 0; entry < txns_per_device; ++entry) {
        const uint32_t payload_bytes = txn_bytes_at(cursor);

        // A chunk is a byte range inside one output shard: page at that shard's
        // start, plus an offset.
        const uint32_t out_shard = cursor / out_shard_bytes;
        const uint32_t output_page = base_page + out_shard * out_shard_tiles;
        const uint32_t offset_bytes = cursor - out_shard * out_shard_bytes;

        // A shard's pages are one contiguous address run, so page + offset is a
        // single address covering the whole chunk. The accessor resolves the
        // destination core, which changes every shard.
        const uint64_t output_noc_address =
            tt::tt_fabric::addrgen_detail::get_noc_address(output_tensor, output_page, offset_bytes);

        // One call, every route. Internally: wait_for_txn_id ->
        // wait_for_next_issue -> get_read_ptr -> prepare_transaction(M) ->
        // publish each claimed slot under the same txn id -> local posted write
        // (include_self) -> commit_transaction, which advances the read pointer
        // ONCE for the whole multicast -> try_complete_front_transaction
        fabric.async_write_multicast_with_state(
            data_request, payload, output_noc_address, payload_bytes);

        cursor += payload_bytes;
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
    fabric.set_atomic_inc_multicast_state(barrier_request, route);
    fabric.atomic_inc_multicast_with_state(
        barrier_request,
        safe_get_noc_addr(barrier_sem_noc_x, barrier_sem_noc_y, barrier_sem_address, noc.get_noc_id()),
        /*value=*/1,
        // The semaphore is on the mirror worker core while the payload went to
        // the shard cores, so without this a peer could see the increment
        // before its data.
        /*flush=*/true);

    // No explicit teardown: ~FabricDataflowBuffer() drains our transaction
    // counters at scope exit, so no SWQ still references a payload page when the
    // kernel returns. The source-local posted writes were already flushed inside
    // each async_write_multicast_with_state (DFB doc §2.6).

    // One increment per peer, plus our own: the route carries include_self, so
    // atomic_inc_multicast_with_state() bumps this device's semaphore locally
    // as well. Fan-in is therefore the peer count plus one.
    const uint32_t barrier_arrivals = num_peers + 1;
    auto* barrier_sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(barrier_sem_address);
    noc_semaphore_wait_min(barrier_sem, barrier_arrivals);
    noc_semaphore_inc(
        safe_get_noc_addr(barrier_sem_noc_x, barrier_sem_noc_y, barrier_sem_address),
        static_cast<uint32_t>(-static_cast<int32_t>(barrier_arrivals)));

    noc.async_write_barrier();
}
```



The payload size is passed per call. `with_state` already patches it —
`UnicastWriteUpdateMask::PayloadSize` is in the mask and the size is an argument
(`fabric_pull.hpp:130-137`) — so a short entry only needs the wrapper to forward
it instead of hardcoding `payload.get_entry_size()`. The source-local
`include_self` write must use the same size.

Teardown does **not** order the atomic behind the data, and does not need to
precede it. The atomic is a request enqueued in the DE send queue like any
other; ordering comes from the DE draining that queue in order, exactly as
production relies on ("the sem is sent after all data sends on a particular
link, so it's correctly ordered at the receiver").

The drain is RAII: `~FabricDataflowBuffer()` calls `finish()`, which spins until
this device's transaction counters are clear so no SWQ still references a
payload page when the kernel exits. Because it spins, the buffer's scope is
where that wait lands — declare it in a narrower scope to drain earlier. Note the counters
track *terminal source-read completions* (DFB doc §1, §6) — "the DMA is done
reading my L1" — not delivery to the peer's Tensix, which is a separate
RX-slot completion.

The barrier decrements rather than resetting to zero, matching
`multicast_reader.cpp` — increments from other phases must be preserved.

## End-to-end credit path

```text
producer thread p          payload DFB (entry = bytes_per_dma_txn bytes)    sender DM
     |                                                                       |
 async_read<TXN_ID>(shard page, offset)     [tail chunk: reserve/read/push]   |
     |    one read, get_entry_size() bytes, contiguous inside one input shard |
     |                                                                       |
 read lands, ISR --- posted++ -----------------------> wait_front(required_occupancy)
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
that entry. The producer side does use a NoC TRID -- that is what bumps
`posted` without an explicit `push_back` -- but the sender side does not: its
only NoC transaction is the header write, and a TRID there would mean "header
delivered", not "payload read".

## Mapping to the production factory


| This example                                                                                              | Production                                                                          |
| --------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------- |
| `PullAllGatherDeviceOperation::compute_output_specs` / `create_output_tensors` / `select_program_factory` | `all_gather_device_operation.cpp:89` / `:119` / `:222`                              |
| `PullAllGatherFactory::create_mesh_workload`                                                              | `all_gather_multicast_factory.cpp:16-57`                                            |
| `::create_at` sizing                                                                                      | `create_at:78-150` — route derivation replaced by `num_devices - 1`                 |
| `::create_at` run args                                                                                    | `create_at:373-501` — per-link loop collapsed to one node                           |
| `::override_runtime_arguments`                                                                            | `:513-538` — re-runs `SetProgramRunArgs` instead of patching by index               |
| block base page `device_idx * tiles_per_device`                                                           | `create_at:291-307` + `next_output_chunk` — the stripe walk a last-dim gather needs |
| producer / sender kernels                                                                                 | `multicast_reader.cpp` read half / `multicast_writer.cpp` + reader's send half      |


Not carried over: the per-link partition across `min_num_links` worker cores
(one node here, so no `axis_num_links`); the reader/writer E-S / W-N split
(one DE, nothing to split); and `FabricWriter`'s scatter batching and route
alternation (the pull API has neither — [DFB doc §8](Quasar-Pull-Fabric-DFB.md)).
