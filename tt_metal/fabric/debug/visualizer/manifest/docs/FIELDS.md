# Fabric manifest fields

Every field the manifest records, with the named compile-time argument that feeds it, and the named arguments and defines it leaves out. [FIELD_TABLES.md](FIELD_TABLES.md) explains the field tables, and [FORMAT.md](FORMAT.md) where fields sit in the manifest.

## Fields

### Router fields

`routers.<key>.fields.erisc<r>`

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
| `vc2_receiver_free_slots` | stream_reg | `unsigned int` in `buf_space_available` | flow_control | `VC2_RECEIVER_FREE_SLOTS_STREAM_ID` | The register reserved for the VC2 receiver's free slots. Null when the router has no VC2. The router kernel declares it but does not read it. |
| `notify_worker_src` | l1 | `unsigned int` | flow_control | `NOTIFY_WORKER_OF_READ_COUNTER_UPDATE_SRC_ADDR` | Where the router stages the source of the inline write that updates a producer's read counter. Allocated only on Blackhole. |
| `my_direction` | enum | `eth_chan_directions` | kernel_params | `MY_DIRECTION` | The direction the router faces, which the kernel picks its downstream channels and free-slot registers by. The router's key uses ControlPlane's direction. |
| `wait_for_host_signal` | flag |  | kernel_params | `WAIT_FOR_HOST_SIGNAL` | Whether the router takes part in the chip's startup sync through local_sync. Without it, the kernel ignores its local sync arguments. |
| `idle_context_switching` | flag |  | kernel_params | `IDLE_CONTEXT_SWITCHING` | Whether the router context switches only once idle, rather than every interval. |
| `context_switch_interval` | number |  | kernel_params | `SWITCH_INTERVAL` | How many main loop iterations go by between context switches to base firmware; with idle_context_switching, how many in a row without progress. |
| `handshake_context_switch_timeout` | number |  | kernel_params | `DEFAULT_HANDSHAKE_CONTEXT_SWITCH_TIMEOUT` | How many writes the handshake sender makes before it gives base firmware a turn to route, and then starts counting again. |
| `txq_spin_wait_send_data` | flag |  | kernel_params | `ETH_TXQ_SPIN_WAIT_SEND_NEXT_DATA` | Whether a data send spins until the Ethernet TX queue is free, instead of skipping the send while it is busy. |
| `txq_spin_wait_completion_ack` | flag |  | kernel_params | `ETH_TXQ_SPIN_WAIT_RECEIVER_SEND_COMPLETION_ACK` | The same, for a receiver's completion ack. |
| `txq_accept_ahead` | number |  | kernel_params | `DEFAULT_NUM_ETH_TXQ_DATA_PACKET_ACCEPT_AHEAD` | How many packets each Ethernet TX queue accepts before the earlier ones are sent. Written to both queues' ETH_TXQ_DATA_PACKET_ACCEPT_AHEAD register at startup. |
| `risc_cpu_data_cache` | flag |  | kernel_params | `ENABLE_RISC_CPU_DATA_CACHE` | Whether the kernel treats the RISC's data cache as on, and so invalidates it before reading what other cores write to its L1. |
| `vc0_bubble_flow_control` | flag |  | kernel_params | `ENABLE_DEADLOCK_AVOIDANCE` | Whether VC0 runs bubble flow control for deadlock avoidance, which adds first-level acks: VC0 senders then also get credits back when the peer receives a packet (acked). |
| `speedy_vc0` | flag |  | kernel_params | `ENABLE_SPEEDY_VC0` | Whether VC0 runs the speedy path: worker-only steps that pass credits in batches, so a credit count can lag the packet count by up to a batch. |
| `sender_credit_amortization` | number |  | kernel_params | `SENDER_CREDIT_AMORTIZATION_FREQUENCY` | Speedy VC0 only: the sender reads completions once this many packets are outstanding, and returns credits to its producer once this many have completed. |
| `receiver_credit_amortization` | number |  | kernel_params | `RECEIVER_CREDIT_AMORTIZATION_FREQUENCY` | Speedy VC0 only: the receiver acks completions to the peer's sender in batches of at least this many packets. |
| `perf_telemetry` | l1 | `byte` × 32 | diagnostics | `PERF_TELEMETRY_BUFFER_ADDR` | Where the router records its bandwidth telemetry, when bandwidth telemetry is on. Allocated only then, except on Blackhole, which always allocates it. |
| `code_profiling` | l1 | `CodeProfilingTimerResult` | diagnostics | `CODE_PROFILING_BUFFER_ADDR` | Where the router accumulates each code profiling timer's cycles and captures, one result per timer type. Allocated only with code profiling on. |
| `channel_trimming_capture_enabled` | flag |  | diagnostics | `ENABLE_CHANNEL_TRIMMING_RESOURCE_USAGE_CAPTURE` | Whether the router records which of its channels carry traffic, for a later run to trim the unused ones. |
| `channel_trimming_capture` | l1 | `FabricDatapathUsageL1Results<true, 3, 10>` | diagnostics | `RESOURCE_USAGE_CAPTURE_OUTPUT_L1_ADDRESS` (with the channel trimming capture on) | Where it records them: the packet sizes each sender channel saw, and which sender and receiver channels carried traffic. |
| `local_handshake_master` | flag |  | lifecycle | `IS_LOCAL_HANDSHAKE_MASTER` | Whether this ERISC leads the chip's startup sync (the chip's local_sync). The kernel reads it only with wait_for_host_signal. |
| `handshake_enabled` | flag |  | kernel_params | `ENABLE_ETHERNET_HANDSHAKE` | Whether this ERISC runs the Ethernet handshake with the peer at startup. |
| `context_switch_enabled` | flag |  | kernel_params | `ENABLE_CONTEXT_SWITCH` | Whether this ERISC hands control back to base firmware. |
| `services_sender_channels` | flag |  | lifecycle | `RISC_SERVICES_SENDER_CHANNELS` | Whether this ERISC has the sender role, so it sets up the sender channels' worker interfaces at startup, and its sender-side TXQ1 state in multi-TXQ mode. On Blackhole with two ERISCs only ERISC0 has it, and otherwise every ERISC does. Channel trimming doesn't change it. |
| `services_receiver_channels` | flag |  | lifecycle | `RISC_SERVICES_RECEIVER_CHANNELS` | Whether this ERISC has the receiver role, so it sets up its TXQ1 state at startup in multi-TXQ mode. On Blackhole with two ERISCs only ERISC1 has it, and otherwise every ERISC does. Channel trimming doesn't change it. |
| `opens_vc0_downstream_connections` | flag |  | lifecycle | `OPEN_DOWNSTREAM_VC0_CONNECTIONS` | Whether this ERISC opens the router's VC0 connections to its downstream routers at startup: set when it runs receiver 0's step. |
| `opens_vc1_downstream_connections` | flag |  | lifecycle | `OPEN_DOWNSTREAM_VC1_CONNECTIONS` | Whether this ERISC opens the router's VC1 connections to its downstream routers at startup: set when it runs receiver 0's step on a router that crosses VC0 traffic over to VC1, or receiver 1's step on a router that services VC1. The kernel reads it only with FABRIC_2D_VC1_ACTIVE. |
| `interrupts_enabled` | flag |  | kernel_params | `ENABLE_INTERRUPTS` | Whether this ERISC is configured with interrupts on. The router kernel declares it but does not read it. |
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
| `remote_worker_sender_channel` | number |  | control_info | `REMOTE_WORKER_SENDER_CHANNEL` | The peer router's worker sender channel. In mux mode the router writes it as the source channel of the packets it receives. 0 when the router has no peer. |

### Sender channel fields

`routers.<key>.channels.senders.vc<v>.ch<c>.fields.erisc<r>`

| Key | Kind | Type | Category | Fed as | Description |
| --- | --- | --- | --- | --- | --- |
| `is_injection_channel` | flag |  | flow_control | `SENDER_CH_{}_IS_INJECTION` | Whether the channel injects traffic into the fabric. With deadlock avoidance on, an injection channel uses bubble flow control. |
| `free_slots` | stream_reg | `unsigned int` in `buf_space_available` | flow_control | `SENDER_CHANNEL_{}_FREE_SLOTS_STREAM_ID` | The channel's slots not holding a packet waiting to be sent: the producer takes one per packet it writes, and the router returns it as soon as it sends the packet over Ethernet. The producer gets its credit back later, at the ack or the completion. |
| `producer_credit_return_noc` | number |  | flow_control | `SENDER_CH_{}_ACK_NOC_ID` | The NoC the router returns the producer's credits on. |
| `producer_credit_return_cmd_buf` | enum | `NocCmdBuf` | flow_control | `SENDER_CH_{}_ACK_CMD_BUF_ID` | The NoC command buffer it returns them through. |
| `connection` | l1 | `ConnectionValue` | control_info | a runtime argument | The state of the producer's connection: the producer writes open and close requested, and the router writes unused once it has torn the connection down. In the router's runtime args. |
| `conn_info` | l1 | `EDMChannelWorkerLocationInfo` | control_info | `LOCAL_SENDER_CH_{}_CONN_INFO_ADDR` | Where the producer writes where it is, and the router its read counter. |
| `buffer_index_sem` | l1 | `SenderChannelProducerCursor` | control_info | a runtime argument | Where the worker channel's producer keeps its write cursor across connections, for the next producer to pick up. In the producer's runtime args, from the channel's connection spec, and the router's, which zeroes it at startup. Null on the other channels, whose producer keeps none. |
| `static_connection` | flag |  | control_info | `SENDER_CH_{}_WAIT_STATIC_CONNECTION` | Whether a sibling router connects to the channel once, at startup: before its main loop, each ERISC that runs the channel waits for that connection. The builder gives an ERISC that doesn't run it 0. |
| `live_check_skip` | flag |  | control_info | `SENDER_CH_{}_LIVE_CHECK_SKIP` | Whether the router skips the connection liveness check on the channel: it has a static peer, or the builder forces the skip. |

### Receiver channel fields

`routers.<key>.channels.receivers.vc<v>.ch<c>.fields.erisc<r>`

| Key | Kind | Type | Category | Fed as | Description |
| --- | --- | --- | --- | --- | --- |
| `forwarding_disabled` | flag |  | kernel_params | `DISABLE_RX_CH{}_FORWARDING` | Whether the receiver's step skips handing its packets on, to siblings or the local chip. Set for the speedy VC0 and VC2 steps, which hand packets on themselves, and when a channel trimming capture saw the channel do neither. |
| `intermesh_ingress` | flag |  | kernel_params | `IS_RECEIVER_CHANNEL_{}_INTERMESH_INGRESS` (2D only) | Whether the receiver's packets arrive from another mesh: the kernel re-encodes their route from this mesh's route table before deciding where they go. |
| `forward_noc` | number |  | kernel_params | `RX_CH_{}_FWD_NOC_ID` | The NoC the receiver forwards packets to sibling routers on. |
| `forward_data_cmd_buf` | enum | `NocCmdBuf` | kernel_params | `RX_CH_{}_FWD_DATA_CMD_BUF_ID` | The NoC command buffer it writes each forwarded packet through. |
| `forward_sync_cmd_buf` | enum | `NocCmdBuf` | kernel_params | `RX_CH_{}_FWD_SYNC_CMD_BUF_ID` | The NoC command buffer it sends the credit increment that announces a forwarded packet through. |
| `local_write_noc` | number |  | kernel_params | `RX_CH_{}_LOCAL_WRITE_NOC_ID` | The NoC the receiver delivers packets to the local chip on. |
| `local_write_cmd_buf` | enum | `NocCmdBuf` | kernel_params | `RX_CH_{}_LOCAL_WRITE_CMD_BUF_ID` | The NoC command buffer it delivers them through. |
| `pkts_sent` | stream_reg | `unsigned int` in `buf_space_available` | flow_control | `TO_RECEIVER_{}_PKTS_SENT_ID` | The packets the peer's sender has sent the channel that the receiver has not taken yet: the peer adds one per packet it sends over Ethernet, and the receiver takes one when it acks the packet with first-level ack, or else once it has forwarded or delivered it. |

### Downstream edge fields

`routers.<key>.intra_chip_downstream_edges.vc<v>.edge<n>.fields.erisc<r>`

| Key | Kind | Type | Category | Fed as | Description |
| --- | --- | --- | --- | --- | --- |
| `free_slots` | stream_reg | `unsigned int` in `buf_space_available` | flow_control | `VC{}_FREE_SLOTS_FROM_DOWNSTREAM_EDGE_{}_STREAM_ID` | The router's view of the free slots in the sender channel the edge lands on: the router takes one per packet it forwards, and the sibling returns it when the packet is acked with first-level ack, or else when it completes. |
| `teardown_sem` | l1 | `unsigned int` | control_info | a runtime argument | Where the sibling acks the router closing the edge's connection at teardown. In the router's runtime args. |

## Arguments the manifest leaves out

The named arguments and defines the router kernel is fed that the manifest does not record, each with the reason.

| Named argument | Reason |
| --- | --- |
| `LOCAL_HANDSHAKE_MASTER_ETH_CHAN` | The chip's local_sync master_eth_chan, from KernelCreationContext. |
| `NUM_LOCAL_EDMS` | The chip's local_sync num_routers, from KernelCreationContext. |
| `EDM_CHANNELS_MASK` | The chip's local_sync router_channels_mask, from KernelCreationContext. |
| `MY_ERISC_ID` | The ERISC's index in the router's eriscs, which the manifest lists in RISC order. |
| `NUM_RECEIVER_CHANNELS` | The router's receiver count. The manifest lists the receivers from the builder's VC shape. |
| `MAX_NUM_SENDER_CHANNELS` | builder_config::num_max_sender_channels. |
| `MAX_NUM_RECEIVER_CHANNELS` | builder_config::num_max_receiver_channels. |
| `MAX_NUM_VCS` | builder_config::MAX_NUM_VCS. |
| `NUM_ETH_PORTS` | The SoC's Ethernet channel count, which sizes the kernel's port_direction_table. |
| `SKIP_SRC_CH_ID_UPDATE` | Not read: the kernel derives the flag from FABRIC_TENSIX_EXTENSION_MUX_MODE (fabric_erisc_router_ct_args.hpp). |
| `PACKED_DOWNSTREAM_VC{}_SENDER_CHANNEL_IDS` | The edges' landing channels, packed; fed only in 2D with channel trimming capture. Each edge records its own. |
| `MY_ETH_CHANNEL` | Not yet described. |
| `NUM_SENDER_CHANNELS` | Not yet described. |
| `NUM_DOWNSTREAM_CHANNELS` | Not yet described. |
| `NUM_DS_OR_LOCAL_TENSIX_CONNECTIONS` | Not yet described. |
| `VC{}_DOWNSTREAM_EDM_SIZE` | Not yet described. |
| `FUSE_RECEIVER_FLUSH_AND_COMPLETION_PTR` | Not yet described. |
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
| `FABRIC_2D_VC1_ACTIVE` | The same fact as ACTUAL_VC1_SENDER_CHANNELS > 0: set when the router has VC1 senders. |
| `ROUTING_MODE` | Fabric-wide: CreateEthernetKernel adds it to every fabric kernel (FabricContext::get_fabric_kernel_defines). |
| `UDM_MODE` | Fabric-wide: CreateEthernetKernel adds it to every fabric kernel (FabricContext::get_fabric_kernel_defines). |
| `FABRIC_1D_PKT_HDR_EXTENSION_WORDS` | Fabric-wide: CreateEthernetKernel adds it to every fabric kernel (FabricContext::get_fabric_kernel_defines). |
| `FABRIC_2D_PKT_HDR_ROUTE_BUFFER_SIZE` | Fabric-wide: CreateEthernetKernel adds it to every fabric kernel (FabricContext::get_fabric_kernel_defines). |
| `FABRIC_EXPRESS_ENABLED` | Fabric-wide: CreateEthernetKernel adds it to every fabric kernel (FabricContext::get_fabric_kernel_defines). |
| `FABRIC_EXPRESS_AXIS` | Fabric-wide: CreateEthernetKernel adds it to every fabric kernel (FabricContext::get_fabric_kernel_defines). |
| `FORCE_WATCHER_OFF` | CreateEthernetKernel adds it when watcher is disabled on Ethernet cores. |
