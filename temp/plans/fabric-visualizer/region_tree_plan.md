# Task 2 — per-router region tree (C++, EDM builder)

Parent: `visualizer_plan.md` §"2. Per-router region tree". Design reference: `fabric_debug_infrastructure_design.md` §3.3–3.5.

Goal: the manifest names every byte of router L1 that the builder or HAL owns, plus every overlay
stream register the router uses, so that decode (task 4) is offline slicing of the raw `UNRESERVED`
image and the pre/post stream reads (task 3) know which ids to peek. Same producer as the CT args,
structured output, interned per distinct layout.

---

## 1. What we learned reading the builder (drives every decision below)

### 1.1 The allocation walk is one constructor

`FabricEriscDatamoverConfig::FabricEriscDatamoverConfig(Topology)` (`erisc_datamover_builder.cpp:234-384`)
is a bump allocator from `hal::get_erisc_l1_unreserved_base()`; the tail is handed to
`FabricStaticSizedChannelsAllocator`. Order, sizes, and conditions:

| # | Field on config | Size | Condition |
|---|---|---|---|
| 1 | `perf_telemetry_buffer_address` | 32 | BH always; WH iff `get_enable_fabric_bw_telemetry()` |
| 2 | `code_profiling_buffer_address` | `get_max_code_profiling_timer_types() * sizeof(CodeProfilingTimerResult)` | iff `get_enable_fabric_code_profiling_rx_ch_fwd()` |
| 3 | `datapath_usage_l1_address` / `datapath_usage_buffer_size` | `sizeof(FabricDatapathUsageL1Results<…>)`, then align 16 | iff `get_enable_channel_trimming_capture()` |
| 4 | `handshake_addr` | 16 (`eth_channel_sync_size`) | always — **but see 1.4, the kernel does not use this one** |
| 5 | align 16, then 4 counter blocks: `to_sender_channel_remote_ack_counters_base_addr`, `to_sender_channel_remote_completion_counters_base_addr`, `receiver_channel_remote_ack_counters_base_addr`, `receiver_channel_remote_completion_counters_base_addr` | each `align(4 * num_sender_channels, 16)` = `router_buffer_clear_size_words` (**bytes**, despite the name) | always reserved; *used* only when the credit plan puts that VC on counters |
| 6 | `edm_channel_ack_addr` | 64 (`4 * eth_channel_sync_size`) | always; **never read by the kernel** (no CT/RT arg carries it) → `unused` |
| 7 | `termination_signal_address`, `edm_local_sync_address`, `edm_status_address` | 16 each | always |
| 8 | per sender channel `i < num_sender_channels` (config **max**: `num_max_sender_channels` in 2D, `get_sender_channel_count(false)` in 1D): `sender_channels_buffer_index_address`, `sender_channels_worker_conn_info_base_address` (`sizeof(EDMChannelWorkerLocationInfo)`), `sender_channels_local_flow_control_semaphore_address`, `sender_channels_producer_terminate_connection_address`, `sender_channels_connection_semaphore_address`, `sender_channels_buffer_index_semaphore_address` (`sizeof(SenderChannelProducerCursor)` == 16) | 16 each except conn-info | always, for the max count |
| 9 | per downstream edm `i < get_downstream_edm_count(is_2D)`: `receiver_channels_downstream_teardown_semaphore_address` | 16 | always |
| 10 | `tensix_relay_connection_buffer_index_id`, `edm_local_tensix_sync_address` | 16 each | always |
| 11 | `notify_worker_of_read_counter_update_src_address` | 16 | BH only |
| 12 | align 32 → `available_buffer_memory_regions[0] = [buffer_region_start, unreserved_base + unreserved_size)` | rest | handed to the channel allocator |

The channel allocator (`FabricStaticSizedChannelsAllocator`) exposes per `(vc, channel)`:
`get_sender_channel_base_address`, `get_sender_channel_number_of_slots`, `get_receiver_channel_base_address`,
`get_receiver_channel_number_of_slots`, `get_num_sender_channels(vc)`, `get_num_receiver_channels(vc)`.
Slot stride is `channel_buffer_size_bytes`. It is sized from the **fabric-wide max** shape
(`FabricBuilderContext::max_*_channels_per_vc_`), not the router's actual shape.

### 1.2 Per-router state that changes *enabled* / *writer*, not geometry

Geometry (addresses) is identical for every router built from the same `FabricEriscDatamoverConfig`
(there are at most 1 + 4 configs: `router_config_` and `router_with_mux_config_[direction]`). What varies per router:

- `actual_sender_channels_per_vc_` / `actual_receiver_channels_per_vc_` (`RouterVcShape` from `ComputeMeshRouterBuilder::build`) → which rings / control blocks are *enabled*.
- `is_sender_channel_serviced_[risc][ch]`, `is_receiver_channel_serviced_[risc][ch]` (private; today only surfaced as `IS_*_CHANNEL_i_SERVICED` named CT args) → `writer` (erisc0 / erisc1 / none) on BH 2-ERISC.
- `receiver_channel_to_downstream_adapter->get_downstream_edm_mask_for_vc(vc)` → which `VC{v}_FREE_SLOTS_FROM_DOWNSTREAM_EDGE_{k}` streams are live.
- Credit plan (`StreamAssignment::plan()`, per mesh) and `ENABLE_FIRST_LEVEL_ACK_VC0` (final value computed inside `get_compile_time_args`, with overrides) → whether ack/completion credits are `stream_reg` or `unreserved_l1` counter slices.
- `has_tensix_extension`, `udm_mode`, `is_inter_mesh`, `wait_for_host_signal` → instance facts, plus whether `edm_local_sync` is meaningful (`edm_local_sync_ptr_addr` is 0 in the kernel when `!wait_for_host_signal`; in production it is always set true in `create_kernel`).

So: **layout = geometry + enabled + writer**, interned; **instance = identity + peer + shape + flags**, per router row.

### 1.3 Lifecycle and threading

- `FabricFirmwareInitializer::compile_and_configure_fabric()` runs `dev->compile_fabric()` per device via `detail::async`. Each worker constructs a `FabricBuilder`, runs the phases, compiles, and **destroys all router builders** on return (`create_and_compile_tt_fabric_program`).
- The last point where per-router state is final is `ComputeMeshRouterBuilder::create_kernel` (after `connect_routers`, after `set_wait_for_host_signal(true)`, where `get_compile_time_args(risc_id)` is called per RISC). This is the publish point.
- `FabricBuilderContext` already holds per-chip state written from those workers without a lock: `master_router_chans_` / `num_initialized_routers_` are `std::vector` pre-sized to `num_devices_` and indexed by `chip_id`, with `TT_FATAL` on double-set. The region tree container follows the same pattern. Interning happens later, single-threaded, in the serializer (answers the open question "worker (locked) or at join": **at join**).
- `compile_fabric_only()` (mock) and the `TERMINATE_FABRIC`-only path also compile and would publish; harmless, the manifest is only written in `configure()` under `INIT_FABRIC`.

### 1.4 Two builder quirks the tree must not paper over

1. **Handshake.** The kernel's `HANDSHAKE_ADDR` is `FabricEriscDatamoverBuilder::handshake_address = round_up(unreserved_base, 16)` (= `UNRESERVED` base). `config.handshake_addr` (row 4 above) is a different, later address that nothing reads. On BH (and WH with bw telemetry) the live handshake **overlaps `perf_telemetry`**. Task 1's `router_template.handshake_address` exports the config value — wrong. Fix in this task: export the builder value, keep the config slot as an `unused` region, and allow this one overlap explicitly (`overlaps: ["diagnostics.perf_telemetry"]`) rather than asserting global non-overlap and then special-casing.
2. **`edm_channel_ack_addr`** is a 64-byte hole kept "to match old EDM". `unused`, not a live region.

### 1.5 What the kernel receives, and from where

- Named CT args carry: `TERMINATION_SIGNAL_ADDR`, `EDM_LOCAL_SYNC_PTR_ADDR`, `EDM_LOCAL_TENSIX_SYNC_PTR_ADDR`, `EDM_STATUS_PTR_ADDR`, `HANDSHAKE_ADDR`, `NOTIFY_WORKER_OF_READ_COUNTER_UPDATE_SRC_ADDR`, `LOCAL_SENDER_CH_{i}_CONN_INFO_ADDR`, `PERF_TELEMETRY_BUFFER_ADDR`, `CODE_PROFILING_BUFFER_ADDR`, `RESOURCE_USAGE_CAPTURE_OUTPUT_L1_ADDRESS`, the four `*_COUNTERS_BASE_ADDR`, all stream ids (`SENDER_CHANNEL_{i}_FREE_SLOTS_STREAM_ID`, `TO_SENDER_{i}_PKTS_ACKED_ID`, `TO_SENDER_{i}_PKTS_COMPLETED_ID`, `TO_RECEIVER_{v}_PKTS_SENT_ID`, `VC{v}_FREE_SLOTS_FROM_DOWNSTREAM_EDGE_{k}_STREAM_ID`, `VC2_RECEIVER_FREE_SLOTS_STREAM_ID`, `TENSIX_RELAY_LOCAL_FREE_SLOTS_STREAM_ID`), `VC{0,1,2}_USES_COUNTER_CREDITS`, `ENABLE_FIRST_LEVEL_ACK_VC0`, `IS_*_CHANNEL_{i}_SERVICED`, `NUM_ACTIVE_ERISCS`, `MY_ERISC_ID`.
- Positional CT args carry the channel rings (`emit_channel_allocations_ct_args`).
- Runtime args carry the sender control-block addresses (`sender_channels_connection_semaphore_id[]`, `…buffer_index_semaphore_id[]`, which in the fabric-init path are the config addresses).

This is why the producer below takes the named-arg maps as an input: every region whose address the kernel learns by name is **read from the named arg** and cross-checked against the config field. The tree cannot drift from the kernel because it is built from the same values the kernel compiles against.

---

## 2. Data model

New header/impl: `tt_metal/fabric/builder/fabric_router_debug_layout.hpp/.cpp` (builder dir, next to the allocators). Not part of `FabricRouterDiagnosticBufferMap`.

```cpp
namespace tt::tt_fabric {

enum class DebugRegionBacking : uint8_t { UNRESERVED_L1, FIXED_L1, STREAM_REG };
enum class DebugRegionWriter : uint8_t { NONE, ERISC0, ERISC1, ANY_ERISC, HOST, PEER, WORKER };

struct FabricRouterDebugRegion {
    std::string id;       // dotted path, e.g. "sender.3.control.conn_info", "sender.3.ring", "credits.sender.3.completed"
    std::string parent;   // "" for roots
    DebugRegionBacking backing;
    // L1 backings
    uint32_t address = 0;
    uint32_t size = 0;
    std::optional<uint32_t> count;   // rings / counter arrays
    std::optional<uint32_t> stride;
    // stream_reg backing
    std::optional<uint32_t> stream_id;
    bool allocated = true;  // reserved in the L1 map / assigned a register
    bool enabled = true;    // this router actually uses it (actual shape, credit plan, masks)
    DebugRegionWriter writer = DebugRegionWriter::ANY_ERISC;
    std::string schema;     // decoder key: "u32", "EDMStatus", "TerminationSignal", "EDMChannelWorkerLocationInfo",
                            // "SenderChannelProducerCursor", "packet_ring", "u32_counter_array", "fabric_telemetry",
                            // "routing_l1_info_t", "heartbeat_word", "raw"
    std::vector<std::string> overlaps;  // ids this region is *known* to overlap (handshake vs perf_telemetry)
};

struct FabricRouterDebugLayout {
    std::vector<FabricRouterDebugRegion> regions;  // stable order = emission order
    bool operator==(const FabricRouterDebugLayout&) const = default;
};

struct FabricRouterDebugInstance {
    FabricNodeId local_node; chan_id_t eth_chan;
    FabricNodeId peer_node; chan_id_t peer_eth_chan;  // from ControlPlane::try_get_connected_mesh_chip_chan_ids
    eth_chan_directions direction; bool is_inter_mesh; bool is_dispatch_link;
    uint32_t num_active_eriscs;
    std::array<uint32_t, MAX_NUM_VCS> sender_channels_per_vc, receiver_channels_per_vc;  // actual
    uint32_t worker_sender_channel;      // get_worker_connected_sender_channel()
    CreditTransportPlan credit_plan; bool first_level_ack_vc0;
    uint32_t downstream_edm_mask_vc0, downstream_edm_mask_vc1;
    bool has_tensix_extension, udm_mode;
    FabricRouterDebugLayout layout;
};

// Producer. `named_ct_args_per_risc[r]` is what create_kernel already has in hand for risc r.
FabricRouterDebugInstance build_router_debug_instance(
    const FabricEriscDatamoverBuilder& builder,
    const StreamAssignment& streams,
    const std::vector<std::unordered_map<std::string, uint32_t>>& named_ct_args_per_risc,
    const RouterLocation& location);

}  // namespace tt::tt_fabric
```

Small accessors added to `FabricEriscDatamoverBuilder` (public, const, trivial): `is_sender_channel_serviced(risc, ch)`, `is_receiver_channel_serviced(risc, ch)`, `get_handshake_address()`, `get_actual_sender_channels_per_vc()` / `get_actual_receiver_channels_per_vc()` (return the optional, fall back to config). Everything else needed is already public (`config`, `receiver_channel_to_downstream_adapter`, `is_first_level_ack_enabled`, `has_tensix_extension`, `udm_mode`, `is_inter_mesh`, `get_worker_connected_sender_channel`).

---

## 3. The producer walk (what `build_router_debug_instance` emits)

Emission order is fixed so that the canonical JSON (and therefore `layout_id`) is deterministic. `U = hal.unreserved`.

1. **`lifecycle`** (parent root)
   - `lifecycle.handshake` — `address = named["HANDSHAKE_ADDR"]`, size 16, schema `handshake_info_t`; `overlaps = ["diagnostics.perf_telemetry"]` iff that region is allocated and ranges intersect. Cross-check: `== builder.get_handshake_address()`.
   - `lifecycle.edm_status` — `named["EDM_STATUS_PTR_ADDR"]`, 16, schema `EDMStatus`. Cross-check `config.edm_status_address`.
   - `lifecycle.termination_signal` — `named["TERMINATION_SIGNAL_ADDR"]`, 16, `TerminationSignal`.
   - `lifecycle.local_sync` — `named["EDM_LOCAL_SYNC_PTR_ADDR"]`, 16, `u32`; `enabled = wait_for_host_signal` (kernel reads 0 otherwise).
   - `lifecycle.local_tensix_sync` — `named["EDM_LOCAL_TENSIX_SYNC_PTR_ADDR"]`, 16, `u32`; `enabled = has_tensix_extension || udm_mode`.
   - `lifecycle.heartbeat` — `FIXED_L1`, `FABRIC_KERNEL_HEARTBEAT_ADDR_{WORMHOLE,BLACKHOLE}` by arch, 4, `heartbeat_word`.
2. **`diagnostics`** — rows, not the tree (`perf_telemetry` 32 B, `code_profiling`, `trimming`), each `allocated = addr != 0`; `enabled`: perf ← `named["PERF_TELEMETRY_MODE"] != 0`; profiling ← `named["CODE_PROFILING_ENABLED_TIMERS"] != 0`; trimming ← `named["ENABLE_CHANNEL_TRIMMING_RESOURCE_USAGE_CAPTURE"]`. Sizes from `get_telemetry_and_metadata_buffer_map()`. On BH `perf_telemetry` is allocated even when disabled; that is exactly the allocated≠enabled distinction.
3. **`credits`** (observable-effect grouping; both transports under one parent)
   - `credits.counters.to_sender_ack`, `.to_sender_completion`, `.receiver_ack`, `.receiver_completion` — `UNRESERVED_L1`, base from `named[...COUNTERS_BASE_ADDR]`, `size = router_buffer_clear_size_words`, `count = num_sender_channels (config max)`, `stride = 4`, schema `u32_counter_array`; `enabled = plan.any_vc_uses_counters()`; writer PEER for `to_sender_*`, ANY_ERISC for `receiver_*`. Cross-check config fields.
   - Per flat sender `s` (over config max): `credits.sender.s.acked` and `credits.sender.s.completed` — `STREAM_REG` with `stream_id = named["TO_SENDER_{s}_PKTS_ACKED_ID"]` / `…COMPLETED_ID` when `!= k_unused_stream_id`, else `allocated=false`. `enabled` = channel exists in actual shape ∧ that VC is on registers ∧ (for acked) `named["ENABLE_FIRST_LEVEL_ACK_VC0"]` and VC == 0. When the VC is on counters, emit instead `credits.sender.s.acked` as an `UNRESERVED_L1` slice of the counter array (`address = base + 4*s`, size 4) — same id, different backing. This is the "WH stream-backed vs BH packed L1, expressed as different backing" requirement.
   - Per receiver channel `r` (over `num_used_receiver_channels`): `credits.receiver.r.pkts_sent` — `STREAM_REG`, `named["TO_RECEIVER_{r}_PKTS_SENT_ID"]`.
   - Per VC `v ∈ {0,1}`, edge `k ∈ 1..4`: `credits.downstream.vc{v}.edge{k}.free_slots` — `STREAM_REG`; `enabled = mask_vc(v) bit (k-1)`.
   - `credits.vc2_receiver.free_slots` (31), `credits.tensix_relay.free_slots` (30) — pinned; `enabled` from `named` value `!= k_unused_stream_id`.
4. **`sender.s`** for `s < num_sender_channels` (config max; `enabled = s < Σ actual sender counts`, `writer` from serviced flags across riscs)
   - `sender.s.control.buffer_index` (16, `u32`), `.conn_info` (`sizeof(EDMChannelWorkerLocationInfo)`, schema `EDMChannelWorkerLocationInfo`; address cross-checked against `named["LOCAL_SENDER_CH_{s}_CONN_INFO_ADDR"]`), `.flow_control_sem` (16), `.producer_terminate` (16), `.connection_sem` (16), `.buffer_index_sem` (16, `SenderChannelProducerCursor`).
   - `sender.s.free_slots` — `STREAM_REG`, `named["SENDER_CHANNEL_{s}_FREE_SLOTS_STREAM_ID"]`, writer WORKER/PEER.
   - `sender.s.ring` — `UNRESERVED_L1`, from allocator `(vc, ch)` via `sender_flat_base`, `count = slots`, `stride = channel_buffer_size_bytes`, `size = count*stride`, schema `packet_ring`. Only for channels the allocator sized (`ch < get_num_sender_channels(vc)`); `enabled` per actual shape.
5. **`receiver.r`** for `r < num_used_receiver_channels`: `receiver.r.ring` (allocator), `receiver.r.downstream_teardown_sem` for `r < get_downstream_edm_count(is_2D)`.
6. **`relay`** — `relay.connection_buffer_index` (16), `enabled = udm_mode`.
7. **BH only:** `misc.notify_worker_src` — `named["NOTIFY_WORKER_OF_READ_COUNTER_UPDATE_SRC_ADDR"]`, 16.
8. **`unused`** — `unused.config_handshake` (`config.handshake_addr`, 16), `unused.edm_channel_ack` (`config.edm_channel_ack_addr`, 64).
9. **`hal` siblings (`FIXED_L1`)** — `telemetry` (HAL `FABRIC_TELEMETRY`, schema `fabric_telemetry`), `routing_table` (HAL `ROUTING_TABLE`, schema `routing_l1_info_t`; this is what capture reads for device identity). `go_msg`/`launch` stay in the top-level `hal` block; they are Metal, not fabric.
10. **`padding.*`** — computed last: sort all `UNRESERVED_L1` regions with `allocated`, drop children whose parent already covers them (rings are leaves; control blocks are leaves), and emit one `padding.N` per gap inside `U`. Tail after the last ring is `padding.tail`. The union of allocated + padding must equal `U` exactly — that is the invariant the test checks.

Cross-check policy: every `named[...]` lookup uses a helper that `TT_FATAL`s on a missing key (a missing name is a kernel/host ABI mismatch, the same class of bug `StreamAssignment::named_args` guards). Every address that exists both in `named` and on `config` is compared with `TT_FATAL` — if they disagree the kernel is already running against a different map than the host believes, which is a real builder bug, not a debug-tool problem. Stream ids equal to `k_unused_stream_id` become `allocated=false`, never an error.

`writer` derivation: for `sender.s.*` → set of riscs `r` with `is_sender_channel_serviced(r, s)`; `{0}`→ERISC0, `{1}`→ERISC1, `{0,1}`→ANY_ERISC, `{}`→NONE. Same for `receiver.r.*`. Lifecycle words: ANY_ERISC (WH) / ERISC0 (BH 2-ERISC master). Counters `to_sender_*`: PEER.

---

## 4. Publication into `FabricBuilderContext`

```cpp
// fabric_builder_context.hpp
void publish_router_debug_instances(ChipId chip_id, std::vector<FabricRouterDebugInstance>&& instances);
const std::vector<FabricRouterDebugInstance>& get_router_debug_instances(ChipId chip_id) const;  // TT_FATAL if unpublished
bool has_router_debug_instances(ChipId chip_id) const;

// private, sized to num_devices_ in the ctor like master_router_chans_
std::vector<std::optional<std::vector<FabricRouterDebugInstance>>> router_debug_instances_;
```

Hook: `FabricBuilder::create_kernels()` already loops `routers_` and owns the `KernelCreationContext`. Rather than each `create_kernel` reaching into the context, `create_kernel` returns nothing new; instead add to `FabricRouterBuilder` a pure virtual
`std::optional<FabricRouterDebugInstance> build_debug_instance() const` implemented by `ComputeMeshRouterBuilder` (calls `erisc_builder_->get_compile_time_args(r)` for each risc — cheap, it is the same call `create_kernel` makes — then `build_router_debug_instance`). `FabricBuilder::create_kernels()` collects them **after** the `create_kernel` loop (so `set_wait_for_host_signal(true)` has happened) and publishes once per chip. One writer per chip slot, no lock, `TT_FATAL` on republish (matches `set_num_fabric_initialized_routers`).

Why re-call `get_compile_time_args` instead of caching: caching means threading a return value through `create_kernel`'s virtual signature; the call costs microseconds per router and the duplication is confined to one place. If review prefers zero duplication, `create_kernel` can stash the per-risc named maps on the `ComputeMeshRouterBuilder` and `build_debug_instance` reads them — same shape, decide at review.

Switch-mesh routers (future `SwitchMeshRouterBuilder`) return `nullopt` and get no `layout_id`; the schema allows that (`layout_id` nullable).

---

## 5. Serialization (`fabric_host_utils.cpp`)

- `make_layouts_and_bindings(builder_context, control_plane)` runs once, after all devices published:
  - For each local chip, for each instance: `canonical = json(instance.layout).dump()` (nlohmann preserves our emission order; keys within a region are emitted in a fixed order by our `to_json`). Intern in `std::map<std::string canonical, std::string layout_id>`; `layout_id = "L" + hex(fnv1a64(canonical))` (16 hex chars). `TT_FATAL` if a different canonical string ever maps to an existing id (collision).
  - `manifest["layouts"][layout_id] = { "regions": [...], "router_count": n }`.
  - Router row (`meshes[].chips[].routers[]`) gains `layout_id` and `instance: { peer: endpoint|null, is_inter_mesh, is_dispatch_link, num_active_eriscs, sender_channels_per_vc, receiver_channels_per_vc, worker_sender_channel, credit_plan: {vc0_uses_counters, vc1_uses_counters, vc2_uses_counters}, first_level_ack_vc0, downstream_edm_mask_vc0, downstream_edm_mask_vc1, has_tensix_extension, udm_mode }`.
  - Region JSON: `{ id, parent, backing, address?, size?, count?, stride?, stream_id?, allocated, enabled, writer, schema, overlaps? }` — absent keys omitted, never `null`, so canonical strings stay compact.
- `router_template.handshake_address` switches to the builder value (`get_fabric_router_config()` does not have it; take it from the first published instance's `lifecycle.handshake` or compute `round_up(unreserved_base, 16)` — prefer the former so the template is provably what a kernel got). Add `router_template.unused_config_handshake_address` for the old value so nothing silently changes meaning.
- `lookup` for the serializer: chips are iterated by `FabricNodeId`; instances are keyed by `eth_chan`; join on that. Any `active_fabric_eth_channels` entry without an instance is a `TT_FATAL` (every active router got a kernel).

Schema (`fabric_debug_manifest_schema.json`, still version 1 — no consumer has shipped):
- `layouts`: `additionalProperties: $defs/layout`; `layout = { regions: [$defs/region], router_count }`.
- `region`: enums for `backing` (`unreserved_l1|fixed_l1|stream_reg`) and `writer`; `if backing == stream_reg then required stream_id else required address,size`.
- `router.required += ["layout_id", "instance"]`; `layout_id: string|null`.
- `router_template.required += ["unused_config_handshake_address"]`.

---

## 6. Python capture side (`manifest.py`)

- `FabricManifest.layouts` (dict) and `router_layout(mesh_id, chip_id, eth_chan) -> dict` helper.
- `_validate_header` adds: every local router's `layout_id` is a key of `layouts`; for every layout: `unreserved_l1` regions lie inside `hal.unreserved`; `stream_reg` ids are `< 32`; `count*stride == size` where both present; region `id`s unique; `parent` refers to an existing id or `""`.
- `stream_regs_for_router(...)` → sorted set of enabled `stream_reg` ids: this is what task 3's pre/post bracket peeks. No snapshot change in this task.
- Tests: extend the fixture helpers (`test_manifest.py`, `test_peek.py`) with one minimal layout; add `test_layout_id_must_exist`, `test_unreserved_region_out_of_bounds_rejected`, `test_stream_regs_for_router`.

---

## 7. Tests

**C++ production path** (`test_fabric_debug_manifest.cpp`, both `Fabric1DFixture` and `Fabric2DFixture`, same CI buckets as task 1):
1. Every local router row has a `layout_id` present in `layouts`.
2. For each layout: allocated `unreserved_l1` regions ∪ `padding.*` tile `hal.unreserved` exactly, with the only permitted overlap being `lifecycle.handshake` × `diagnostics.perf_telemetry` and only when the region's `overlaps` says so.
3. Every `stream_reg` region's `stream_id` equals `stream_assignment[mesh][<expected name>]` (name derived from the region id — the test is the second, independent mapping from id → CT-arg name).
4. Ring geometry: `stride == fabric_context.channel_buffer_size_bytes`; sender ring `count` equals `FabricStaticSizedChannelsAllocator::get_sender_channel_number_of_slots(vc, ch)` from the live `builder_context.get_fabric_router_config()`.
5. `enabled` sender count per router equals `Σ instance.sender_channels_per_vc`.
6. `router_template.handshake_address == round_up(hal.unreserved.base, 16)`.
7. Intern sanity: on T3K 1D, the number of distinct layouts is small (≤ 4: edge/interior × direction-dependent shapes); assert `layouts.size() <= routers.size()` and that two routers with identical `instance` shape share an id.

**Builder-level fatal cross-checks** run on every init (they are the "assert in the builder" from the parent plan, cheap and unconditional).

**Manual T3K validation, once, before merge** (script under `/tmp`, not committed): load the manifest, collect all enabled `stream_reg` ids for a WH 1D router, and diff against `fabric_erisc_constants.py`'s `FABRIC_STREAM_GROUPS` ids (0–1 receiver, 2–5 acked, 6–13 completed, 14–21 downstream, 22–29 sender free slots). Every id the dumper labels must appear with the same role. Record the result in the PR description.

**Blackhole**: same tests ride `Fabric*Fixture` on BH; manual dispatch of the BH sanity pipeline before merge as in task 1. Specifically eyeball: `perf_telemetry` allocated+disabled, handshake overlap recorded, `misc.notify_worker_src` present, `writer` split across ERISC0/ERISC1 when 2-ERISC mode is on, counter-backed `credits.sender.s.*` when any VC uses counters.

---

## 8. Slices (each builds, each reviewable on its own)

| # | Slice | Files | Done when |
|---|---|---|---|
| A | Type + producer | new `builder/fabric_router_debug_layout.{hpp,cpp}`; accessors on `FabricEriscDatamoverBuilder`; `sources.cmake` | compiles; producer unit-checks (`TT_FATAL`s) in place; nothing calls it yet |
| B | Publication | `fabric_router_builder.hpp` (virtual), `compute_mesh_router_builder.{hpp,cpp}`, `fabric_builder.cpp`, `fabric_builder_context.{hpp,cpp}` | fabric init on T3K populates one instance per active router; existing manifest test still passes |
| C | Serialize + schema + C++ test | `fabric_host_utils.cpp`, schema JSON, `test_fabric_debug_manifest.cpp` | tests 1–7 pass on T3K 1D and 2D |
| D | Python | `manifest.py`, fixtures, tests, README | `pytest capture/tests -q --noconftest` green |
| E | Validation | `/tmp` script vs dumper; BH CI dispatch | diff recorded; BH run green |

Stop for review after A, after C, and after E.

---

## 9. Risks and how the plan handles them

- **Handshake overlap on BH** — explicit `overlaps`, test allowlist is exactly one pair, `router_template` corrected. Without this the tiling test would fail on the first BH run.
- **Config-max vs actual shape** — control blocks and allocator rings exist for the fabric max; `enabled` carries the actual. Decode must not paint an allocated-but-disabled ring as a live channel; the schema carries the distinction so the viewer can grey it.
- **`k_unused_stream_id` sentinels** — `allocated=false`, not an error; the name set is complete by construction (`StreamAssignment::named_args`).
- **Per-VC flat indexing** — sender flat id = `stream_assignment.sender_flat_base(vc) + ch` (fabric-scoped) which equals `RouterVcShape::flat_sender_id` only when the router's lower-VC counts equal the fabric max. Use the **fabric-scoped** base everywhere (it is what the CT-arg names use); note it in the region id comment.
- **`num_devices_` on TG** (4 + 4 PCIe quirk in the ctor) — index by `chip_id` like the existing vectors; no new assumption.
- **Re-init in one process** — `FabricBuilderContext` is rebuilt with the `FabricContext`; publish `TT_FATAL`s on double-publish for the same chip within one context, matching existing setters.
- **Layout JSON size on Galaxy** — interning bounds it by distinct shapes (single digits); instance blocks are ~15 scalars per router.
- **`get_compile_time_args` re-call** — microseconds; if it ever gains side effects (it mutates `sender_channel_connection_liveness_check_disable_array` only via `build_connection_*`, not here) switch to the stash-in-`create_kernel` variant described in §4.
- **Tensix mux / UDM relay** — erisc side is fully described; the tensix core's own region tree is out of scope (design §3.3: another producer, same inspector). `has_tensix_extension` / `udm_mode` on the instance tell decode not to expect worker traffic on `sender.0`.

## 10. Open questions carried forward (not blocking)

- Whether `instance` should also carry the peer's ring geometry (`remote_receiver_channels_*`) so decode can sanity-check a sender's free-slot count against the remote ring size without opening the peer's manifest row. Cheap to add later; skipped now to keep the row small.
- `writer` for `lifecycle.*` on BH 2-ERISC: today set to ERISC0 by construction (`IS_LOCAL_HANDSHAKE_MASTER` is `risc_id == 0`); confirm against the kernel when the BH run is inspected.
