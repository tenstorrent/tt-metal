# Fabric manifest fields

Every field the manifest records, with the named compile-time argument that feeds it, and the named arguments and defines it leaves out. [FIELD_TABLES.md](FIELD_TABLES.md) explains the field tables, and [FORMAT.md](FORMAT.md) where fields sit in the manifest.

## Fields

### Router fields

Each `erisc<r>` of the `field_sets` block that `routers.<key>.fields` names.

| Key | Kind | Type | Category | Fed as | Description |
| --- | --- | --- | --- | --- | --- |
| `edm_status` | l1 | `EDMStatus` | lifecycle | `EDM_STATUS_PTR_ADDR` | How far the router has come through startup and teardown. |
| `termination_signal` | l1 | `TerminationSignal` | lifecycle | `TERMINATION_SIGNAL_ADDR` | Where the router is told to stop. |
| `local_sync` | l1 | `unsigned int` | lifecycle | `EDM_LOCAL_SYNC_PTR_ADDR` | The count the chip's routers sync through at startup: the others notify the master router, which then notifies them all. |
| `local_tensix_sync` | l1 | `unsigned int` | lifecycle | `EDM_LOCAL_TENSIX_SYNC_PTR_ADDR` | The count the router waits on until its downstream tensix connections are ready. |
| `handshake` | l1 | `handshake_info_t` | lifecycle | `HANDSHAKE_ADDR` | The Ethernet handshake scratch. Each end of the link writes its scratch into the first 16 bytes of the other's. |
| `handshake_sender` | flag |  | lifecycle | `IS_HANDSHAKE_SENDER` | Whether this end writes first: the sender repeats its write until the reply lands in its own scratch, and the receiver waits for the sender's write and then replies once. The two ends of a link differ. |
| `erisc_sync` | stream_reg | `unsigned int` in `remote_src` | lifecycle | `MULTI_RISC_TEARDOWN_SYNC_STREAM_ID` | The register the router's ERISCs sync through at startup and teardown, led by ERISC0. Unused when the router runs one ERISC. |
| `retrain_sync` | stream_reg | `CoordinatedEriscContextSwitchState` in `remote_src` | lifecycle | `ETH_RETRAIN_LINK_SYNC_STREAM_ID` | The register the router's ERISCs sync through to retrain the link, led by the ERISC that context switches. |
| `notify_worker_src` | l1 | `unsigned int` | flow_control | `NOTIFY_WORKER_OF_READ_COUNTER_UPDATE_SRC_ADDR` | Where the router stages the source of the inline write that updates a producer's read counter. Allocated only on Blackhole. |
| `my_direction` | enum | `eth_chan_directions` | kernel_params | `MY_DIRECTION` | The direction the router faces, which the kernel picks its downstream channels and free-slot registers by. The router's key uses ControlPlane's direction. |
| `wait_for_host_signal` | flag |  | kernel_params | `WAIT_FOR_HOST_SIGNAL` | Whether the router takes part in the chip's startup sync through local_sync. Without it, the kernel ignores its local sync arguments. |
| `idle_context_switching` | flag |  | kernel_params | `IDLE_CONTEXT_SWITCHING` | Whether the router context switches only once idle, rather than every interval. |
| `context_switch_interval` | number |  | kernel_params | `SWITCH_INTERVAL` | How many main loop iterations go by between context switches to base firmware; with idle_context_switching, how many in a row without progress. |
| `handshake_context_switch_timeout` | number |  | kernel_params | `DEFAULT_HANDSHAKE_CONTEXT_SWITCH_TIMEOUT` | How many writes the handshake sender makes before it gives base firmware a turn to route, and then starts counting again. |
| `txq_spin_wait_completion_ack` | flag |  | kernel_params | `ETH_TXQ_SPIN_WAIT_RECEIVER_SEND_COMPLETION_ACK` | The same, for a receiver's completion ack. |
| `txq_accept_ahead` | number |  | kernel_params | `DEFAULT_NUM_ETH_TXQ_DATA_PACKET_ACCEPT_AHEAD` | How many packets each Ethernet TX queue accepts before the earlier ones are sent. Written to both queues' ETH_TXQ_DATA_PACKET_ACCEPT_AHEAD register at startup. |
| `txq_seq_update_timeout` | number |  | kernel_params | `ETH_TXQ_SEQ_UPDATE_TIMEOUT` | The Ethernet TX queues' keepalive period while the router runs, in cycles: how long an idle queue waits before it sends its paired RX queue's acknowledgements. |
| `risc_cpu_data_cache` | flag |  | kernel_params | `ENABLE_RISC_CPU_DATA_CACHE` | Whether the kernel treats the RISC's data cache as on, and so invalidates it before reading what other cores write to its L1. |
| `vc0_bubble_flow_control` | flag |  | kernel_params | `ENABLE_DEADLOCK_AVOIDANCE` | Whether VC0 runs bubble flow control for deadlock avoidance, which adds first-level acks: VC0 senders then also get credits back when the peer receives a packet (acked). |
| `injection_min_free_slots` | number |  | flow_control | `BUBBLE_FLOW_CONTROL_INJECTION_SENDER_CHANNEL_MIN_FREE_SLOTS` (with deadlock avoidance on) | With bubble flow control, the free slots an injection channel needs in the peer's VC0 receiver before it sends. |
| `speedy_vc0` | flag |  | kernel_params | `ENABLE_SPEEDY_VC0` | Whether the builder made VC0 a speedy router: a worker-only VC0 whose receiver only delivers to this chip, with no bubble flow control or first-level acks. The kernel doesn't act on this flag: the builder applies the choice by setting VC0's receiver forwarding_disabled and turning vc0_bubble_flow_control off. |
| `perf_telemetry` | l1 | `byte` × 32 | diagnostics | `PERF_TELEMETRY_BUFFER_ADDR` | Where the router records its bandwidth telemetry, when bandwidth telemetry is on. Allocated only then, except on Blackhole, which always allocates it. |
| `code_profiling` | l1 | `CodeProfilingTimerResult` | diagnostics | `CODE_PROFILING_BUFFER_ADDR` | Where the router accumulates each code profiling timer's cycles and captures, one result per timer type. Allocated only with code profiling on. |
| `channel_trimming_capture_enabled` | flag |  | diagnostics | `ENABLE_CHANNEL_TRIMMING_RESOURCE_USAGE_CAPTURE` | Whether the router records which of its channels carry traffic, for a later run to trim the unused ones. |
| `channel_trimming_capture` | l1 | `FabricDatapathUsageL1Results<true, 3, 10>` | diagnostics | `RESOURCE_USAGE_CAPTURE_OUTPUT_L1_ADDRESS` (with the channel trimming capture on) | Where it records them: the packet sizes each sender channel saw, and which sender and receiver channels carried traffic. |
| `local_handshake_master` | flag |  | lifecycle | `IS_LOCAL_HANDSHAKE_MASTER` | Whether this ERISC leads the chip's startup sync (the chip's local_sync). The kernel reads it only with wait_for_host_signal. |
| `handshake_enabled` | flag |  | kernel_params | `ENABLE_ETHERNET_HANDSHAKE` | Whether this ERISC runs the Ethernet handshake with the peer at startup. |
| `context_switch_enabled` | flag |  | kernel_params | `ENABLE_CONTEXT_SWITCH` | Whether this ERISC hands control back to base firmware. |
| `opens_vc0_downstream_connections` | flag |  | lifecycle | `OPEN_DOWNSTREAM_VC0_CONNECTIONS` | Whether this ERISC opens the router's VC0 connections to its downstream routers at startup: set when it runs receiver 0's step. |
| `opens_vc1_downstream_connections` | flag |  | lifecycle | `OPEN_DOWNSTREAM_VC1_CONNECTIONS` | Whether this ERISC opens the router's VC1 connections to its downstream routers at startup: set when it runs receiver 0's step on a router that crosses VC0 traffic over to VC1, or receiver 1's step on a router that services VC1. The kernel reads it only on a router with VC1 senders (ACTUAL_VC1_SENDER_CHANNELS > 0). |
| `teardown_check_iterations` | number |  | kernel_params | `ITERATIONS_BETWEEN_CTX_SWITCH_AND_TEARDOWN_CHECKS` | How many passes over its channel steps the main loop makes between its checks for termination and context switching. |
| `telemetry_stats_mask` | number |  | kernel_params | `FABRIC_TELEMETRY_STATS_MASK` | The FabricTelemetryStatistic bits this ERISC collects; 0 when its telemetry is off. |
| `telemetry_enabled` | flag |  | diagnostics | `ENABLE_FABRIC_TELEMETRY` | Whether this ERISC collects fabric telemetry. telemetry_stats_mask says which statistics. |
| `perf_telemetry_mode` | number |  | diagnostics | `PERF_TELEMETRY_MODE` | The kernel's PerfTelemetryRecorderType: 0 for none, 1 for low-resolution bandwidth. The enum is device-side only, so the value is written as a number. |
| `code_profiling_timers` | number |  | diagnostics | `CODE_PROFILING_ENABLED_TIMERS` | The CodeProfilingTimerType bits the kernel times; 0 when code profiling is off. |
| `sender_txq` | number |  | kernel_params | `SENDER_TXQ_ID` | The Ethernet TX queue the router sends packets on. |
| `receiver_txq` | number |  | kernel_params | `RECEIVER_TXQ_ID` | The Ethernet TX queue the router sends credits back on: 1 in 2-ERISC mode, otherwise the sender's. |
| `intermesh_router` | flag |  | kernel_params | `IS_INTERMESH_ROUTER` | Whether the router's link goes to another mesh. |
| `intermesh_edge` | flag |  | kernel_params | `IS_INTERMESH_ROUTER_ON_EDGE` | Whether the router's channel is one of the chip's inter-mesh-facing channels (ControlPlane::get_intermesh_facing_eth_chans). |
| `intramesh_edge` | flag |  | kernel_params | `IS_INTRAMESH_ROUTER_ON_EDGE` | Whether the router is an intra-mesh router on a chip with inter-mesh links, which checks each packet for an inter-mesh exit. |
| `mesh_x_size` | number |  | kernel_params | `MESH_X_SIZE` (2D only) | The mesh's columns, which bound the router's 2D route table. |
| `mesh_y_size` | number |  | kernel_params | `MESH_Y_SIZE` (2D only) | The mesh's rows, which bound the router's 2D route table. |

### Sender channel fields

Each `erisc<r>` of the `field_sets` block that `routers.<key>.channels.senders.vc<v>.ch<c>.fields` names.

| Key | Kind | Type | Category | Fed as | Description |
| --- | --- | --- | --- | --- | --- |
| `is_injection_channel` | flag |  | flow_control | `SENDER_CH_{}_IS_INJECTION` | Whether the channel injects traffic into the fabric. With deadlock avoidance on, an injection channel uses bubble flow control. |
| `free_slots` | stream_reg | `unsigned int` in `buf_space_available` | flow_control | `SENDER_CHANNEL_{}_FREE_SLOTS_STREAM_ID` | The channel's slots not holding a packet waiting to be sent: the producer takes one per packet it writes, and the router returns it as soon as it sends the packet over Ethernet. The producer gets its credit back later: at the peer receiver's first-level ack, or when the packet's completion group there completes (the peer receiver's completion_group_size). |
| `producer_credit_return_noc` | number |  | flow_control | `SENDER_CH_{}_ACK_NOC_ID` | The NoC the router returns the producer's credits on. |
| `producer_credit_return_cmd_buf` | enum | `NocCmdBuf` | flow_control | `SENDER_CH_{}_ACK_CMD_BUF_ID` | The NoC command buffer it returns them through. |
| `connection` | l1 | `ConnectionValue` | control_info | a runtime argument | The state of the producer's connection: the producer writes open and close requested, and the router writes unused once it has torn the connection down. In the router's runtime args. |
| `conn_info` | l1 | `EDMChannelWorkerLocationInfo` | control_info | `LOCAL_SENDER_CH_{}_CONN_INFO_ADDR` | Where the producer writes where it is, and the router its read counter. |
| `buffer_index_sem` | l1 | `SenderChannelProducerCursor` | control_info | a runtime argument | Where the worker channel's producer keeps its write cursor across connections, for the next producer to pick up. In the producer's runtime args, from the channel's connection spec, and the router's, which zeroes it at startup. Null on the other channels, whose producer keeps none. |
| `static_connection` | flag |  | control_info | `SENDER_CH_{}_WAIT_STATIC_CONNECTION` | Whether a sibling router connects to the channel once, at startup: before its main loop, each ERISC that runs the channel waits for that connection. The builder gives an ERISC that doesn't run it 0. |
| `live_check_skip` | flag |  | control_info | `SENDER_CH_{}_LIVE_CHECK_SKIP` | Whether the router skips the connection liveness check on the channel: it has a static peer, or the builder forces the skip. |

### Receiver channel fields

Each `erisc<r>` of the `field_sets` block that `routers.<key>.channels.receivers.vc<v>.ch<c>.fields` names.

| Key | Kind | Type | Category | Fed as | Description |
| --- | --- | --- | --- | --- | --- |
| `forwarding_disabled` | flag |  | kernel_params | `DISABLE_RX_CH{}_FORWARDING` | Whether the receiver forwards nothing to sibling routers: it only delivers to this chip.  |
| `intermesh_ingress` | flag |  | kernel_params | `IS_RECEIVER_CHANNEL_{}_INTERMESH_INGRESS` (2D only) | Whether the receiver's packets arrive from another mesh: the kernel re-encodes their route from this mesh's route table before deciding where they go. |
| `forward_noc` | number |  | kernel_params | `RX_CH_{}_FWD_NOC_ID` | The NoC the receiver forwards packets to sibling routers on. |
| `forward_data_cmd_buf` | enum | `NocCmdBuf` | kernel_params | `RX_CH_{}_FWD_DATA_CMD_BUF_ID` | The NoC command buffer it writes each forwarded packet through. |
| `forward_sync_cmd_buf` | enum | `NocCmdBuf` | kernel_params | `RX_CH_{}_FWD_SYNC_CMD_BUF_ID` | The NoC command buffer it sends the credit increment that announces a forwarded packet through. |
| `local_write_noc` | number |  | kernel_params | `RX_CH_{}_LOCAL_WRITE_NOC_ID` | The NoC the receiver delivers packets to the local chip on. |
| `local_write_cmd_buf` | enum | `NocCmdBuf` | kernel_params | `RX_CH_{}_LOCAL_WRITE_CMD_BUF_ID` | The NoC command buffer it delivers them through. |
| `pkts_sent` | stream_reg | `unsigned int` in `buf_space_available` | flow_control | `TO_RECEIVER_{}_PKTS_SENT_ID` | The packets the peer's sender has sent the channel that the receiver has not taken yet: the peer adds one per packet it sends over Ethernet, and the receiver takes one when it acks the packet with first-level ack, or else once it has issued the packet's forward or local write. The peer's sender gets the slot back later, through the credit counters: at the ack, or when the slot's completion group completes (completion_group_size). |
| `completion_group_size` | number |  | flow_control | `RX_CH_{}_COMPLETION_GROUP_SIZE` | How many slots the receiver returns credits for (to the upstream sender) at once. It returns a group's credits once no NoC command with the group's transaction ID is in flight. |
| `num_trids` | number |  | kernel_params | `RX_CH_{}_NUM_TRIDS` | # of NoC transaction IDs that a receiver channel uses. Each completion group uses an id, and if more groups exist than ids, trids are shared across groups. |
| `trid_start` | number |  | kernel_params | `RX_CH_{}_TRID_START` | The channel's first transaction ID. The range of trids used for this channel is [trid_start, trid_start + num_trids). |

### Downstream edge fields

Each `erisc<r>` of the `field_sets` block that `routers.<key>.intra_chip_downstream_edges.vc<v>.edge<n>.fields` names.

| Key | Kind | Type | Category | Fed as | Description |
| --- | --- | --- | --- | --- | --- |
| `free_slots` | stream_reg | `unsigned int` in `buf_space_available` | flow_control | `VC{}_FREE_SLOTS_FROM_DOWNSTREAM_EDGE_{}_STREAM_ID` | The router's view of the free slots in the sender channel the edge lands on: the router takes one per packet it forwards, and the sibling returns it once its peer receiver acks the packet with first-level ack, or else once the packet's completion group there completes (that receiver's completion_group_size). |
| `teardown_sem` | l1 | `unsigned int` | control_info | a runtime argument | Where the sibling acks the router closing the edge's connection at teardown. In the router's runtime args. |

## Arguments the manifest leaves out

The named arguments and defines the router kernel is fed that the manifest does not record, each with the reason.

| Named argument | Reason |
| --- | --- |
| `LOCAL_HANDSHAKE_MASTER_ETH_CHAN` | The local sync master's Ethernet channel, from KernelCreationContext. |
| `NUM_LOCAL_EDMS` | The chip's local_sync num_routers, from KernelCreationContext. |
| `EDM_CHANNELS_MASK` | The chip's local_sync router_channels_mask, from KernelCreationContext. |
| `MY_ERISC_ID` | The ERISC's index in the router's eriscs, which the manifest lists in RISC order. |
| `NUM_RECEIVER_CHANNELS` | The router's receiver count. The manifest lists the receivers from the builder's VC shape. |
| `VC{}_WORKER_CONNECTION_SEMAPHORE_ADDR` | The worker channel's connection semaphore, fed for VC0 and VC2 only. The same L1 word as the sender's connection field, which reads it from the router's runtime args. |
| `MAX_NUM_SENDER_CHANNELS` | builder_config::num_max_sender_channels. |
| `MAX_NUM_RECEIVER_CHANNELS` | builder_config::num_max_receiver_channels. |
| `MAX_NUM_VCS` | builder_config::MAX_NUM_VCS. |
| `SKIP_SRC_CH_ID_UPDATE` | Not read: the kernel derives the flag from FABRIC_TENSIX_EXTENSION_MUX_MODE (fabric_erisc_router_ct_args.hpp). |
| `PACKED_DOWNSTREAM_VC{}_SENDER_CHANNEL_IDS` | The edges' landing channels, packed; fed only in 2D with channel trimming capture. Each edge records its own. |
| `VC{}_USES_COUNTER_CREDITS` | Set wherever the VC has senders, and indicates whether a vc credit mechanism is through counters in l1. |
| `TO_SENDER_{}_PKTS_COMPLETED_ID` | A stream register for the channel's credits, which the router doesn't read: it takes every VC's credits through L1 counters (the router's credit_counters). |
| `TO_SENDER_{}_PKTS_ACKED_ID` | A stream register for the channel's credits, which the router doesn't read: it takes every VC's credits through L1 counters (the router's credit_counters). |
| `SENDER_CREDIT_AMORTIZATION_FREQUENCY` | Not read by the router: the kernel only checks it is nonzero on a speedy router (fabric_erisc_router_ct_args.hpp). |
| `RECEIVER_CREDIT_AMORTIZATION_FREQUENCY` | Not read by the router: the kernel only checks it is nonzero on a speedy router (fabric_erisc_router_ct_args.hpp). |
| `RISC_SERVICES_SENDER_CHANNELS` | Not read by the router: an ERISC's roles come from the channels it services (is_sender_role and is_receiver_role in router_common.hpp), which each channel's serviced_by shows. |
| `RISC_SERVICES_RECEIVER_CHANNELS` | Not read by the router: an ERISC's roles come from the channels it services (is_sender_role and is_receiver_role in router_common.hpp), which each channel's serviced_by shows. |
| `REMOTE_WORKER_SENDER_CHANNEL` | Not read by the router: in mux mode it takes the source channel of the packets it receives as 0. |
| `ETH_TXQ_SPIN_WAIT_SEND_NEXT_DATA` | Not read by the router. The host requires 0 (validate_router_config). |
| `ENABLE_INTERRUPTS` | Not read by the router. |
| `VC2_RECEIVER_FREE_SLOTS_STREAM_ID` | Not read by the router. |
| `NUM_DOWNSTREAM_CHANNELS` | Not read by the router. |
| `FUSE_RECEIVER_FLUSH_AND_COMPLETION_PTR` | Not read by the router beyond a static_assert that it is 1 (fabric_erisc_router_ct_args.hpp). |
| `NUM_ETH_PORTS` | Not read by the router. |
| `MY_ETH_CHANNEL` | Not yet described. |
| `NUM_SENDER_CHANNELS` | Not yet described. |
| `NUM_DS_OR_LOCAL_TENSIX_CONNECTIONS` | Not yet described. |
| `VC{}_DOWNSTREAM_EDM_SIZE` | Not yet described. |
| `EDM_NOC_VC` | Not yet described. |
| `FORCE_ALL_PATHS_TO_USE_SAME_NOC` | Not yet described. |
| `UDM_MODE` | Not yet described. |
| `LOCAL_RELAY_NUM_BUFFERS` | Not yet described. |
| `TENSIX_RELAY_LOCAL_FREE_SLOTS_STREAM_ID` | Not yet described. |
| `LINK_SYNC_ROLE` | The streaming profiler's link sync, fed on Blackhole when the profiler is on (link_sync::add_router_compile_args). Not yet described. |
| `LINK_SYNC_L1_ADDR` | The streaming profiler's link sync, fed on Blackhole when the profiler is on (link_sync::add_router_compile_args). Not yet described. |
| `LINK_SYNC_CHECK` | The streaming profiler's link sync, fed on Blackhole when the profiler is on (link_sync::add_router_compile_args). Not yet described. |

| Define | Reason |
| --- | --- |
| `FABRIC_2D` | The same fact as IS_2D_FABRIC: both come from is_2D_routing_enabled. |
| `ROUTING_MODE` | Fabric-wide: CreateEthernetKernel adds it to every fabric kernel (FabricContext::get_fabric_kernel_defines). |
| `UDM_MODE` | Fabric-wide: CreateEthernetKernel adds it to every fabric kernel (FabricContext::get_fabric_kernel_defines). |
| `FABRIC_1D_PKT_HDR_EXTENSION_WORDS` | Fabric-wide: CreateEthernetKernel adds it to every fabric kernel (FabricContext::get_fabric_kernel_defines). |
| `FABRIC_2D_PKT_HDR_ROUTE_BUFFER_SIZE` | Fabric-wide: CreateEthernetKernel adds it to every fabric kernel (FabricContext::get_fabric_kernel_defines). |
| `FABRIC_EXPRESS_ENABLED` | Fabric-wide: CreateEthernetKernel adds it to every fabric kernel (FabricContext::get_fabric_kernel_defines). |
| `FABRIC_EXPRESS_AXIS` | Fabric-wide: CreateEthernetKernel adds it to every fabric kernel (FabricContext::get_fabric_kernel_defines). |
| `FORCE_WATCHER_OFF` | CreateEthernetKernel adds it when watcher is disabled on Ethernet cores. |
