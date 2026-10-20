# Blackhole fabric throughput and DRISC offload discussion

Discussion notes, 2026-10-08; revised 2026-10-09 after a code and golden review. Checkout inspected: `46b5e2366d6`. The first version cited `ff5ec873e9c`, which is not present in this repository; all line numbers below refer to `46b5e2366d6`.

## Scope and status

Explore architectural ways to increase fabric throughput, starting with cycle budgets and existing benchmark evidence. Focus on Blackhole, a 1.35 GHz AI clock, and hypothetical 200 Gb/s and 400 Gb/s per-link bandwidths. Added latency is acceptable if throughput improves.

This is an analysis record with a candidate prototype direction, not a measured gain or committed implementation. No new device benchmarks were run. Existing goldens provide end-to-end throughput evidence. Per-packet cycle costs below are either derived from goldens or estimated by reading code, and are labelled as such; code-derived estimates are not measurements.

### Revision summary (2026-10-09)

- **Wrong baseline.** The 144-cycle baseline used in the first version comes from the 1D neighbor-exchange *speedy* router path, not the generic 2D VC0/VC1 path this proposal targets. Generic baselines are 186–198 cycles/packet for 1D Linear and about 209–210 for 2D Mesh multicast (aggregate over two flows). See §3.
- **Fused rows contain no flush wait.** The fused write-plus-atomic golden rows use `flush=false`. Their penalty is real but its cause is unidentified. See §3.
- **RX criticality depends on configuration.** Extra RX work (4-chunk scatter) shows up almost 1:1 on 1D ring rows, is absorbed on the 2D VC0-only Mesh multicast rows, and is unmeasured for the target generic VC0+VC1 2D configuration, which is built at `-Os`. See §3.3.
- **Local-delivery cost.** A code-derived estimate puts a local unicast write on the ERISC at about 45–65 cycles, and a stateful push handoff at about 20–30. See §7.1.
- **DRISC speed.** A "~12.6 cycles per DRISC L1 load" figure circulated during the discussion is not a load-latency measurement. There is no evidence that DRISC executes this work slower than ERISC; use α ≈ 1 as the working assumption. See §7.2.
- **Stream-register notification.** DRISC has 16 stream registers, so the existing fabric payload-then-stream-register-credit protocol can notify a DRISC of packets. See §6.2.
- **Push preferred over pull.** Pull lengthens Ethernet RX slot hold time by an estimated 500–700 cycles; push keeps today's retirement rule. See §6.1.
- **Placement.** Pairing with the 8 "free subchannel" DRISCs, whose NIUs are both in stream mode, allows an untagged NoC1 handoff and NoC1 final delivery with the packet's existing addresses. That removes the DRAM-endpoint remap and the producer API change, at the cost of at most 8 helpers per chip. See §6.5.
- **New system-level blockers.** No DRISC reservation mechanism exists and the tensor prefetcher uses all 16 DRISCs; DRISCs are reset on every Metal init and teardown; DRISC programs launch only through slow dispatch; DRISC kernel text is limited to about 11 KB. See §6.10.

## 1. Per-packet budgets

For payload size `P` bytes, bandwidth `B` bytes/s, and frequency `f = 1.35e9` cycles/s:

```text
packet rate             = B / P
target service interval = f * P / B
```

These are steady-state service intervals, not limits on end-to-end packet latency. Multiple packets can be in flight.

| Payload | 200 Gb/s: cycles/packet | 400 Gb/s: cycles/packet |
| --- | ---: | ---: |
| 1 KiB | 55.296 | 27.648 |
| 2 KiB | 110.592 | 55.296 |
| 4 KiB | 221.184 | 110.592 |
| 8 KiB | 442.368 | 221.184 |
| 16 KiB | 884.736 | 442.368 |

200 Gb/s is 25 GB/s, or 18.52 bytes per reference cycle. 400 Gb/s is 50 GB/s, or 37.04 bytes per reference cycle. At 400 Gb/s, 1 KiB requires 48.83 million packets/s; 4 KiB requires 12.21 million packets/s.

These are ideal payload-only budgets, for one link in one direction. Headers, framing, and control traffic reduce useful payload bandwidth. At actual wire saturation, those extra bytes also increase the interval between payload packets; they must not be described as reducing the permitted cycles per packet while retaining the same payload rate.

Both hypothetical rates correspond to real systems. The CCL report gives Blackhole 400 Gb/s with a payload ceiling of about 48 GB/s after framing (`tech_reports/CCLs/CCL_bandwidth.md:20,36`), and the p150 x4 golden reaches 45.5 GB/s for 8704-byte Mesh multicast packets. Blackhole Galaxy links currently run at half rate, 25 GB/s (`CCL_bandwidth.md:201`).

The basic explanation for packet-size sensitivity is that a roughly fixed service cost per packet produces bandwidth proportional to packet size until another resource saturates. A 110–145-cycle interval can approach a 400 Gb/s payload ceiling at approximately 4 KiB while falling well short at 1–2 KiB.

## 2. Current router: facts and hypotheses

Main implementation: [fabric_erisc_router.cpp](tt_metal/fabric/impl/kernels/edm_fabric/fabric_erisc_router.cpp).

- Incoming Ethernet data is placed directly into Ethernet L1 by hardware. The ERISC does not copy payload out of a software-visible RX queue. The transport uses destination addresses and a separate arrival notification. See the [BH Ethernet transport documentation](https://github.com/tenstorrent/tt-isa-documentation/blob/main/BlackholeA0/EthernetTile/EthernetTxRx.md).
- In the supported two-ERISC configuration, E0 services TX and E1 services RX. This is configuration-dependent, not universal. See [erisc_datamover_builder.cpp](tt_metal/fabric/erisc_datamover_builder.cpp).
- TX scans enabled sender channels, checks availability and credits, issues the Ethernet transfer and arrival notification, and services acknowledgements and connection state. Compile-time-disabled channels do not incur the same scan work as enabled but idle channels. A VC0-only 2D Mesh router has 4 sender channels, and up to 9 with VC1 (`fabric_builder_config.hpp:56,68`).
- After issuing the payload send, `send_next_data` spins until the Ethernet transmit queue is free before sending the remote pointer update (`fabric_erisc_router.cpp:534`).
- RX handles arrival, routing/local-delivery decisions, downstream admission, NoC issuance, retirement, and credits. Local delivery includes unicast writes, scatter operations, inline values, atomics, and requested flushes. See [fabric_edm_packet_transmission.hpp](tt_metal/fabric/hw/inc/edm_fabric/fabric_edm_packet_transmission.hpp). The generic RX step processes one packet per receiver-channel visit and sends one Ethernet completion acknowledgement per packet; the speedy paths amortize these.
- Ordinary NoC writes are asynchronous, but command issuance can wait for an available command buffer (`dataflow_api.h:2807`). Some atomic/fused operations explicitly request completion of earlier writes. "Asynchronous" does not mean command issue is free or that the path never waits.
- On Blackhole, every local write also updates the software NoC counters, as a hang workaround (`dataflow_api.h:2809-2813`, issue #28758). Atomics are forced to be non-posted (`noc_nonblocking_api.h:1255`), and inline writes to L1 are emulated as a local store plus a NoC write.
- Ordinary RX buffer retirement checks source consumption, not necessarily destination completion. Despite its name, `WriteTransactionIdTracker::transaction_flushed` calls a transaction-ID **sent** check. See [transaction tracking](tt_metal/fabric/hw/inc/edm_fabric/fabric_erisc_router_transaction_id_tracker.hpp).
- In 2D Mesh without deadlock avoidance, first-level acknowledgement is off (`erisc_datamover_builder.cpp:1075-1079`). A worker-facing sender slot is therefore returned only after the remote RX completion arrives, so the worker credit loop spans a full Ethernet round trip plus RX processing.
- With VC1 active, the router is built at `-Os` instead of `-O3` "to fit in code space" (`erisc_datamover_builder.cpp:1742-1745`). `TT_METAL_FABRIC_OPT_LEVEL` overrides this.

RX performs more control work per packet than TX; §7.1 estimates about 120–160 cycles for a local-delivery packet, plus about 30–45 per forwarding direction, against about 60–90 for TX per packet sent. More work does not by itself make RX the critical stage, however. TX also pays a per-iteration scan over every enabled sender channel, and the worker credit loop can bound throughput independently (§8). Routing uses precomputed actions; it should not be assumed to perform expensive route computation per packet.

TX scanning can matter for sparse traffic. With `N` enabled channels and `A` active channels, an approximate scan contribution per serviced packet is:

```text
C_active + ((N - A) / A) * C_empty + C_loop / A
```

Here an empty visit can include more than checking a flag: each visit invalidates the data cache and reads stream registers, roughly 10–20 cycles by code inspection. A benchmark with every channel active does not isolate this sparse-channel cost.

### Existing instrumentation

- `TT_FABRIC_PROFILE_RX_CH_FWD=1` (`rtoptions.cpp:788-792`) enables the `RECEIVER_CHANNEL_FORWARD` code-profiling timer around decode, admission, forwarding and local issue in the generic RX step (`fabric_erisc_router.cpp:1417-1419,1561`). It excludes the flush/completion stage, the speedy RX path, and TX, so it gives a lower bound on RX busy time per packet.
- `TT_METAL_FABRIC_TELEMETRY` with `stats=BANDWIDTH` (`rtoptions.cpp:781-786`) records separate TX (ERISC0) and RX (ERISC1) active cycles. A loop iteration counts as active if any progress occurred, so it is coarse.
- The device profiler inflates router cycle counts by 11–18% on multicast, ring and mesh tests (`tests/pipeline_reorg/fabric_perf_tests.yaml:24-30`); benchmark with it disabled.

### Related finding: VC2 speedy transaction IDs

`SpeedyReceiverState<2>` starts at `RX_CH_TRID_STARTS[2]`, which is 8 (16 with deadlock avoidance), and flips with `current_write_trid = 1 - current_write_trid` (`fabric_erisc_router_speedy_path.hpp:45,299`). That alternates 8 ↔ 249 as a `uint8_t`. `NOC_PACKET_TAG_TRANSACTION_ID(id)` is an unmasked `id << 10`, so 249 spills into neighbouring tag fields, and the sent check reads `NIU_MST_WRITE_REQS_OUTGOING_ID(249)`, the wrong status register. VC0 is unaffected because its start is 0. This undermines using an "unchanged VC2" path as a baseline and should be confirmed and fixed separately.

## 3. What the current goldens show

Source: [BH p150 x4 fabric golden](tests/tt_metal/tt_fabric/test_infra/golden/golden_bandwidth_summary_blackhole_p150_x4.csv), selecting `num_links = 1`. Configuration: `tests/tt_metal/tt_fabric/test_infra/test_yamls/test_fabric_ubench_at_least_2x2_mesh.yaml` on a 2x2 mesh.

Effective cycles/packet below are `1.35e9 / avg_packets_per_s`. The CSV's `avg_cycles` is the maximum over sender cores of elapsed send time, including warmup and drain; it is not a per-packet instruction count.

| Traffic / local operation | Router path | 2 KiB GB/s | Cycles/packet | 4 KiB GB/s | Cycles/packet |
| --- | --- | ---: | ---: | ---: | ---: |
| Neighbor unicast / write | 1D neighbor exchange, speedy | 19.229 | 143.78 | 38.379 | 144.08 |
| Linear multicast / write | 1D generic | 14.836 | 186.35 | 29.068 | 190.23 |
| Mesh multicast / write | 2D generic, VC0 only, 2 flows/link | 13.148 | 210.29 | 26.455 | 209.02 |
| Full-ring multicast / write | 1D ring, 3 flows/link | 8.922 | 309.87 | 18.055 | 306.27 |
| Neighbor unicast / fused write + atomic | 1D neighbor exchange, speedy | 5.887 | 469.67 | 10.928 | 505.99 |

Corrections and context for these rows:

- **The neighbor rows are the speedy path.** `Topology::NeighborExchange` maps to `FABRIC_1D_NEIGHBOR_EXCHANGE` (`test_tt_fabric.cpp:27`): one worker-only VC0 sender channel, one receiver, no forwarding, no deadlock avoidance. Under those conditions `vc0_speedy_path_enabled` is true (`channel_trimming_import.cpp:171-175`), so the router uses amortized credits and batched transaction-ID completion. These rows do not represent the generic VC0/VC1 receiver.
- **The benchmark producer sends no payload.** In benchmark mode the sender writes headers into each slot during warmup and then issues only stateful credit increments (`tt_fabric_test_kernels_utils.hpp:1074-1117`). The router still transmits and delivers the declared payload. This is the same caveat already noted for mux v2 in §4.
- **The fused rows contain no flush wait.** The test headers set `flush=false` (`tt_fabric_test_kernels_utils.hpp:1280,1298`). The penalty of about 320–360 cycles over plain writes appears on every path (Linear 510–546, Mesh 527–565) and its cause is unidentified. Blackhole forces atomics to be non-posted; if the stall is at the NIU or destination, an offload helper would face the same wall.
- **Mesh and ring rows are link aggregates.** Packet counts imply 2 concurrent flows for Mesh and HalfRing and 3 for FullRing; per-flow intervals are 2–3× larger. All Blackhole rows are bidirectional per link.
- **Freshness.** The Blackhole goldens were last regenerated by `9a4db98b083` (2026-03-17). No CI job comparing against this file was found.

Throughput accounting is in [tt_fabric_test_bandwidth_profiler.cpp](tests/tt_metal/tt_fabric/test_infra/tt_fabric_test_bandwidth_profiler.cpp). The summary uses traffic counts and elapsed cycles; it is not a direct measurement of Ethernet wire utilization or RX execution time.

### 3.1 Packet-size trend

Custom-packet-size rows, `num_links = 1`:

| Size (B) | Linear multicast GB/s | Cycles/packet | Mesh multicast GB/s | Cycles/packet |
| ---: | ---: | ---: | ---: | ---: |
| 2048 | 14.84 | 186 | 13.15 | 210 |
| 3264 | 23.66 | 186 | 20.99 | 210 |
| 4096 | 29.07 | 190 | 26.46 | 209 |
| 5440 | 38.17 | 192 | 35.28 | 208 |
| 7616 | 37.51 | 274 | 42.63 | 241 |
| 8704 | 41.05 | 286 | 45.49 | 258 |
| 15232 | 41.84 | 491 | 36.96 | 556 |

Up to 5440 bytes the interval is constant, so a per-packet cost dominates. From about 7.6 KB, throughput plateaus at 37.5–45.5 GB/s, a byte-rate ceiling near the ~48 GB/s payload limit. The throughput problem is specific to small and medium packets. There are no 1 KiB rows.

### 3.2 Changes since the golden

- `9a4db98b083` changed only the benchmark sender to the credit-only steady state. At 2 KiB, speedy neighbor writes went from 166.10 to 143.78 cycles/packet and speedy scatter from 180.68 to 144.77, while Linear multicast moved from 191.82 to 186.35 and Mesh multicast was unchanged (210.13 → 210.29). The speedy path was bounded by the producer or its credit loop at that time and may still be.
- `4087f83344f` (2026-07-27) raised the 2D local-worker sender depth from 2 to 7 slots on its target topology, improving an 8-device all-gather by about 13%. Its Blackhole P150x8 Mesh ubench A/B was flat for Mesh multicast (within about 0.23%), so worker sender depth was not limiting that test.

### 3.3 Is RX the critical stage? Write versus scatter

The scatter test always uses 4 chunks (`NOC_SCATTER_WRITE_MAX_CHUNKS`), so each packet costs the RX 3 extra local NoC writes. Where RX is the critical stage, that work should show up in the interval:

| Configuration | Write (2K / 4K) | Scatter (2K / 4K) | Δ |
| --- | --- | --- | --- |
| 1D neighbor, speedy | 143.8 / 144.1 | 144.8 / 149.6 | +1 / +5.5 |
| 1D Linear, generic | 186.4 / 190.2 | 192.1 / 191.9 | +6 / +2 |
| 2D Mesh multicast, VC0 only | 210.3 / 209.0 | 208.0 / 208.9 | −2 / 0 |
| 1D HalfRing multicast | 176.0 / 176.2 | 226.4 / 226.6 | +50 / +50 |
| 1D FullRing multicast | 309.9 / 306.3 | 374.2 / 365.1 | +64 / +59 |
| 1D FullRing unicast | 247.3 / 246.5 | 273.2 / 272.0 | +26 / +26 |

- **1D ring:** the extra work lands almost 1:1. Each RX there handles 2–3 flows and both forwards and delivers locally. Local delivery itself on FullRing costs about +60 (multicast versus unicast). RX is at or near the critical stage, and offload should pay off there.
- **Single-flow and 2D Mesh multicast:** the extra work disappears. In the 2D VC0-only Mesh multicast configuration, RX had at least about 50 cycles of slack, so it was busy for at most about 160 of every 210 cycles.
- **Target configuration, generic 2D VC0+VC1: unmeasured.** It is built at `-Os`, the RX serves two receiver channels, TX scans up to 9 sender channels, and larger meshes add transit forwarding in up to three directions. RX being the critical stage there is plausible but not established. §9 lists the measurements that would settle it.
- `SingleSenderMeshUnicastAllDevices` rows use a slower round-robin producer path (write = scatter = 284 cycles/packet) and are producer-bound; they cannot attribute RX cost.

### 3.4 Latency golden

[golden_latency_summary_blackhole_p150_x4.csv](tests/tt_metal/tt_fabric/test_infra/golden/golden_latency_summary_blackhole_p150_x4.csv) gives ping-pong times. For Mesh unicast writes the average round trip is 1432 ns at 16 B and 1813 ns at 4 KiB, or 716–906 ns per hop including worker and NoC overheads. §8 uses these for slot-hold estimates.

## 4. Lessons from mux v1 and v2

Sources:

- [Mux v1](tt_metal/fabric/impl/kernels/tt_fabric_mux.cpp) and [v1 golden](tests/tt_metal/microbenchmarks/ethernet/fabric_mux_bandwidth_golden.csv).
- [Mux v2 forwarder](tt_metal/fabric/impl/kernels/tt_fabric_mux_v2_forwarder.hpp), [manager](tt_metal/fabric/impl/kernels/tt_fabric_mux_v2_manager.hpp), and [BH v2 golden](tests/tt_metal/microbenchmarks/ethernet/fabric_mux_v2_throughput_golden_blackhole.csv).

V1 services forwarding and management on one RISC and checks source consumption before returning packet credits. V2 separates forwarding and management across two RISCs, uses a shared transaction-ID ring, posted stateful payload/header transfers, and cumulative credit updates. The useful architectural lesson is separating issuance from retirement while bounding outstanding work.

For v2, one sender, eight buffers, `r0`:

| Payload | GB/s | Effective cycles/packet |
| --- | ---: | ---: |
| 64 B | 0.775 | 111.49 |
| 1 KiB | 12.401 | 111.47 |
| 2 KiB | 24.606 | 112.37 |
| 4 KiB | 49.292 | 112.18 |
| 4352 B | 51.980 | 113.03 |

The per-packet floor holds from 64 B to 4352 B.

For 4352-byte payloads, its buffer sweep is approximately:

| Buffers/channel | GB/s | Effective cycles/packet |
| --- | ---: | ---: |
| 1 | 10.026 | 586 |
| 2 | 19.545 | 301 |
| 4 | 39.133 | 150 |
| 8 | 51.979 | 113 |
| 16 | 52.055 | 113 |

A rough descriptive fit is `C ≈ max(113, 600 / N)` reference cycles/packet. It illustrates a buffer-reuse limit at shallow depth and a service-rate limit at greater depth; it is not a measured DRISC latency. The same two regimes apply to fabric channels (§8).

Comparability limits matter:

- V2's producer sends only the header while declaring a full payload; the mux still transfers the declared payload to a synthetic drainer. See [v2 benchmark sender](tests/tt_metal/tt_metal/perf_microbenchmark/routing/kernels/tt_fabric_mux_v2_ubench_sender.cpp).
- V1's producer sends the full packet, and its byte accounting includes the header, unlike v2's payload accounting. The architecture-neutral v1 golden is not an established apples-to-apples BH baseline.
- These mux measurements do not include a real Ethernet link or execution of the packet's final local operation.
- The v2 `r0`/`r1` sweep selects the forwarder NoC; it does not mean the forwarder and manager exchange processor responsibilities.

## 5. Rebalancing work before or alongside DRISC offload

Ideas discussed include moving RX retirement/credit work to the TX ERISC, dividing local delivery from transit forwarding, and sharding channels or flows instead of handing off every packet. The current DRISC candidate instead gives the DRISC ownership of local-delivery retirement; Ethernet completion acknowledgements remain an ERISC responsibility. Eager decode on TX only helps if the saved RX work exceeds the extra decoding, metadata transfer, and synchronization cost.

For work `d` moved from a receiver with cost `R` to a sender with cost `T`, with handoff overheads `h0` and `h1`, the new control interval is approximately:

```text
max(T + d + h0, R - d + h1)
```

Shared L1 state is not free. BH documentation gives a minimum L1 load latency of 7 cycles versus 2 cycles for private RAM. See [BH ERISC documentation](https://github.com/tenstorrent/tt-isa-documentation/blob/main/BlackholeA0/EthernetTile/BabyRISCV/README.md). Actual latency depends on access conditions. Prefer private ownership, implicit ring-slot identity, and amortized cumulative notifications where possible. Batching counters does not eliminate per-packet descriptor reads.

### ERISC-only improvements to compare against

An offload must beat a tuned ERISC, not today's. Candidates, with code-derived savings estimates:

- A `[[likely]]` unicast-write branch ahead of the local-delivery switch avoids the jump-table load and indirect jump (about 10–12 cycles).
- Retiring the Blackhole counter-update workaround once inline writes move to stream registers (about 6–8 cycles per local write).
- Keeping command-buffer control and tag state when unchanged (a few cycles).
- Porting the speedy-path techniques (amortized completion credits, batched transaction-ID completion) to the generic receiver where its constraints allow. The gap between the 1D speedy and generic paths (144 versus 186 cycles/packet) is an upper bound on what this can recover in 1D; part of it comes from the speedy path's narrower configuration (one sender channel, no forwarding).
- Processing more than one ready packet per RX step, and batching Ethernet completion acknowledgements.

Together the first three reclaim roughly 15–25 of the 45–65 cycles estimated for a local unicast write (§7.1). Operations with several local writes per packet keep most of the offload advantage.

## 6. DRISC local-delivery schemes

Initial scope: one DRISC per RX ERISC. ERISC retains fabric routing/forwarding; DRISC owns local operation processing. This can move command preparation, issuance, local transaction-ID tracking, and local completion/retirement management, not merely the operation switch. An RX ERISC may service both VC0 and VC1, so one paired DRISC can receive work from both channels even in a 1:1 core mapping.

Target the generic VC0/VC1 receiver path on Blackhole 2D fabric first, behind a default-off host environment flag (working name `TT_METAL_ENABLE_FABRIC_DRISC_LOCAL_DELIVERY`). The host should create DRISC kernels and pass an ERISC compile-time enable only when the flag and supported topology both match. UDM mode already delivers through a local relay and should stay outside this first path. Keep the VC2 speedy path and the separate speedy VC0 path unchanged. The speedy paths also call `execute_chip_unicast_to_local_chip_impl` (`fabric_erisc_router_speedy_path.hpp:268`), so changing that helper globally would unintentionally change them. Select DRISC delivery at the generic receiver's local-delivery call sites or via a compile-time policy passed only from those call sites. A baseline comparison should use the same generic VC configuration with offload disabled and enabled.

| Scheme | ERISC work | DRISC work | Ethernet RX slot retires | Status |
| --- | --- | --- | --- | --- |
| Push full packet | Admit against cached DRISC credits; one stateful NoC write of the packet into a DRISC ring slot; stream-register doorbell | Decode and deliver; return credits | When the push and forwarding writes are sent, as today | First candidate |
| Notify + whole-packet pull | Stream-register doorbell; later poll for read completion | Read packet, decode and deliver; report read completion | After the DRISC read completes and its report returns | Compare only if RX slots are deep |
| Notify + split reads | As pull, plus length if needed | Read header and payload separately; decode after header completes | As pull | Lowest priority |

All three schemes move payload Ethernet L1 → DRISC L1 → destination.

### 6.1 Why push before pull

- **Slot hold time.** Today an RX slot is released once its outgoing writes have been sent. Push keeps that rule. Pull holds the slot until the DRISC's read response has fully arrived and its report has come back: doorbell flight, plus DRISC read latency (about 231 ns ≈ 310 cycles in historical measurements, `tt_metal/tools/profiler/STREAMING_PROFILER_FINDINGS.md:241-251`), plus report flight. That is an estimated 500–700 extra cycles per packet. With `N` receiver slots it adds `(500–700) / N` to the slot-latency floor (§8); for `N = 8` that is 60–90 cycles/packet, more than pull's roughly 10-cycle ERISC saving over push. In 2D without first-level acknowledgement the delay also extends the worker-facing sender loop.
- **Existing protocol.** Push is what router-to-router forwarding already does: payload write, then a stream-register credit increment (`fabric_router_adapter.hpp:256-280`). UDM's relay uses the same pattern for local delivery (`execute_chip_unicast_to_relay`), and `admit_2d_dispatch` already includes relay capacity in admission under `udm_mode` (`fabric_erisc_router.cpp:844`). [Local relay code](tt_metal/fabric/impl/kernels/edm_fabric/fabric_router_relay_extension.cpp) is an architectural analogue for the DRISC side, not a ready-made implementation.
- **No unvalidated access pattern.** Nothing in the repository reads ERISC L1 remotely while Ethernet RX writes it. Push avoids depending on that.
- **ERISC cost.** Push costs the ERISC about 20–30 cycles per packet against 45–65 for today's local unicast write (§7.1). The saving grows with operations that issue several local writes per packet.

### 6.2 Stream-register notification

Blackhole DRISC tiles have 16 stream registers (`DRAM_NOC_NUM_STREAMS`, `noc/noc_overlay_parameters.h:29`), against 32 on Ethernet tiles and 64 on Tensix. Tests exercise them: a DRISC writes, reads and increments its own registers including `SPACE_AVAILABLE_UPDATE` (`tests/tt_metal/tt_metal/api/test_noc.cpp:688`), a Tensix writes a DRISC stream register remotely with a register inline write (`:703`), and a DRISC writes a Tensix stream register.

Protocol:

- **ERISC → DRISC:** push write into a DRISC ring slot, then a stream-register increment. The increment carries a count, so doorbells can be batched.
- **DRISC:** poll its own stream register (a register read, with no data-cache invalidation for the counter), fence before reading the packet header from L1, and decrement the register when the packet is consumed.
- **DRISC → ERISC credits:** an inline write to an ERISC stream register; the ERISC keeps a cached free-slot count, as it does for downstream routers.

Conditions:

1. The doorbell must arrive on an NIU in stream mode. `drisc_mode.h` states that in NOC2AXI mode the bottom 8 GB of the NIU address space maps to GDDR; stream registers at `0xFFB40000` fall inside that window, and inline writes mask the high address bits so they cannot carry the DRISC tag. Pair with free-subchannel DRISCs (§6.5) or send doorbells on NoC0.
2. Payload-then-doorbell ordering on the same NoC and VC is relied on today for Ethernet and Tensix receivers; validate it on a DRISC NIU.
3. The ERISC stream-register budget is tight: 32 registers, ids 0–29 assigned by requirement and 30–31 pinned (`tt_metal/fabric/builder/fabric_stream_assignment.hpp:38-69`). The UDM relay's free-slot id 30 is the natural choice, but it is shared with VC2 sender flow control. With VC2 active, the DRISC credits need another id or the counter-based credit transport.
4. The remote-write test uses NoC0. A register inline write reaching a DRISC over NoC1 has not been tested.

### 6.3 Split-read overlap (pull schemes only)

Use separate transaction IDs for header and payload reads. A header-specific completion check can then allow decode while the payload remains in flight. Existing APIs include [per-ID read barriers](tt_metal/hw/inc/api/dataflow/dataflow_api.h) and [nonblocking per-ID completion polling](tt_metal/hw/inc/api/dataflow/noc.h). A global read barrier would defeat that overlap. ID allocation must also account for other in-flight slots and operations. Switching IDs requires `noc_async_read_set_trid`, which waits for the command buffer, so each switch has a cost.

If header and payload complete at times `tH` and `tP`, measured from the same origin, and decode takes `CD`:

```text
earliest outgoing issue ≈ max(tH + CD, tP)
```

Issuing the payload read before the header arrives requires knowing its length: fixed-size traffic, length in the notification, or a full-slot read. Otherwise extract the length first, issue the payload read, and overlap the remaining processing.

For throughput, overlap across packets may matter more than overlap within one packet. A whole-packet read pipeline can fetch future packets while processing and draining earlier slots, using fewer read commands than split reads. Split reads need a demonstrated benefit against that baseline.

### 6.4 Buffer ownership and ordering

- DRISC owns local-delivery transaction IDs, the local write/atomic sequence, and retirement of its staging slots. Staging storage can be reused once outgoing writes have been sent; destination-completion requirements for flushes/atomics remain separate. VC0 and VC1 need distinct ordered queue/credit state even if they share one DRISC.
- ERISC owns forwarding source transactions and must issue Ethernet completion acknowledgements. With push, the ERISC retires the Ethernet RX slot once the push write and any forwarding writes have been sent, extending today's transaction-ID sent check to the push write. With pull, the slot can be released only after the DRISC has finished reading the packet and all ERISC forwarding consumers have consumed their sources; DRISC read completion must be communicated safely, since notification or read issuance alone is insufficient.
- Flush-bearing local operations need DRISC to wait for the relevant local NoC writes before issuing their atomic or semaphore increment. ERISC forwarding transaction IDs remain for Ethernet RX source ownership; they do not become a local-delivery flush dependency merely because the current ERISC helper polls both kinds of transactions.
- More buffering tolerates latency, but does not fix insufficient CPU, NoC, or L1 throughput.

### 6.5 Placement and NoC choice

Blackhole exposes 24 DRAM-tile endpoints (8 channels × 3 subchannels). Metal excludes the NoC0 view endpoints as syseng-owned, leaving 16 Metal DRISCs (`tt_metal/llrt/metal_soc_descriptor.cpp:68-90`). Firmware sets NIU modes once per boot (`drisc_mode.h`):

- NIU0 is in stream mode on all 16.
- NIU1 is in NOC2AXI mode on the 8 that are a DRAM view's NoC1 endpoint, and in stream mode on the other 8, the "free subchannel" DRISCs (`tests/tt_metal/tt_metal/api/test_dram_kernels.cpp:173-224`).

**Recommended for the prototype:** pair RX ERISCs with free-subchannel DRISCs.

- The ERISC's local NoC is NoC1, and a stream-mode NIU1 accepts both untagged L1 writes and stream-register doorbells, so no tag is needed.
- These DRISCs can initiate on NoC1, so they can deliver using the NoC1 destination addresses already in the packet. The addrgen helper, producers, and DRAM addressing stay unchanged (§6.7).
- Cost: at most 8 helpers per chip, against 14 Ethernet cores on Blackhole, and fewer on DRAM-harvested parts. A comment in the streaming profiler notes that "NOC 1 egress runs ~2x the service interval of NOC 0" for DRISC relays (`tt_metal/impl/streaming_profiler/device_programs.cpp:521`); measure NoC1 delivery throughput from DRAM-column DRISCs before relying on it.

**Alternative:** NoC0 delivery from any of the 16, with NoC0 doorbells or tagged NoC1 writes. That needs the DRAM-endpoint handling in §6.7.

There are four unicast NoC VCs, numbered 0–3 (`NOC_VCS` is 16, but VC4 and above begin the multicast/broadcast range). These are separate from fabric VC0/VC1/VC2. The generic dataflow API defaults to NoC VC1 for writes and read requests; the fabric ERISC builder assigns NoC VC2 or VC3 by link index (`compute_mesh_router_builder.cpp:844-847`). Default ERISC forwarding and local delivery are both on NoC1; a Galaxy placement optimization sends forwarding on NoC0 for one side of some multi-link connections. Read-request VC selection is narrower than the first version stated: in dedicated-NoC mode `noc_async_read` ignores its VC argument (`noc_nonblocking_api.h:562-566`), and choosing a read VC needs `read_set_state<use_vc=true>`, which persists for later reads on that command buffer. Read responses are assigned VCs dynamically.

### 6.6 Transaction IDs and command buffers

Blackhole exposes transaction IDs 0–15 on each NoC NIU (`NOC_MAX_TRANSACTION_ID` is `0xF`). The outstanding-request register is indexed by NoC and ID, so DRISC reads and nonposted writes issued on the same NIU share this 16-ID budget across command buffers; the DRISC and ERISC have separate NIUs and do not need a shared ID allocator. `NOC_MAX_TRANSACTION_ID_COUNT = 255` is the per-ID outstanding-request counter limit, not 255 distinct IDs. The existing ERISC router partitions IDs by receiver channel, with four per channel normally or eight when deadlock avoidance is enabled (`fabric_erisc_router_ct_args.hpp:700`); a DRISC kernel needs its own explicit partition. One ID per staging slot, reused for that slot's outgoing writes, is a simple policy. A flush that covers earlier local writes across slots must wait for the appropriate earlier IDs as well. Commands that never set a tag inherit the last ID set on their command buffer, so ID 0 is implicitly shared.

Each NIU has four command buffers by software convention: 0 for bulk writes, 1 for reads, 2 for small writes, 3 for atomics. On the RX ERISC all four are already in use on NoC1: local writes on 0, forwarding sync on 1, forwarding data on 2, atomics on 3 (`erisc_datamover_builder.cpp:482-489`). The 2D single-live stateful forwarding path keeps state in the data and sync buffers (`fabric_erisc_router.cpp:817-830`). Offloading all generic local delivery frees buffer 0 for the push write, unless another path on that ERISC (such as VC2 speedy local delivery) still uses it. Stateful configuration belongs to a command buffer, so shared buffers must have their state restored. Neither a NoC VC nor a command buffer adds transaction IDs.

### 6.7 DRAM destinations, tags, and inline writes

**DRAM addresses.** Current DRAM delivery is address-driven: the addrgen fabric API calls `get_noc_addr(..., edm_to_local_chip_noc)` with NoC1, selecting `dram_bank_to_noc_xy[1][bank]`, and puts that NoC address in the packet. The ERISC local-write path issues it on NoC1 without remapping. Blackhole worker-L1 virtual coordinates are the same on both NoCs; DRAM endpoints differ by NoC.

- With NoC1 delivery from free-subchannel DRISCs (§6.5), nothing changes.
- With NoC0 delivery, a wrong endpoint is silent corruption, not misrouting: NIU0 is in stream mode on every Metal DRISC, so a NoC0 write addressed to a NoC1 DRAM endpoint lands in that DRISC's L1, not in GDDR.
- The first version proposed changing the shared helper `addrgen_detail::get_noc_address` (`tt_metal/fabric/hw/inc/linear/addrgen_api.h:53`). It has 32 call sites in `mesh/api.h`, is used by the 1D builders, and is called directly from 20 ttnn files, so a change must be a per-kernel compile-time selection and couples producer builds to router mode across the fabric. A DRISC-side NoC1 → NoC0 DRAM-endpoint remap that asserts on unknown coordinates avoids that coupling. Raw-address callers and separately supplied semaphore addresses need the same audit either way.

**DRISC L1 tag (`0x2000000000`).** The tag is required only when the target NIU may be in NOC2AXI mode; stream-mode NIUs accept untagged L1 writes (`drisc_mode.h:25-29,56-62`), and DRAM-sender global circular buffers already write 32-bit DRISC addresses (`tt_metal/impl/buffers/global_circular_buffer.cpp:357-365`). Where a tag is needed:

- `noc_async_write_set_state` masks the high address with `NOC_PCIE_MASK` and drops the tag (`noc_nonblocking_api.h:84,1600`). Use a wide stateful write that programs coordinates and the 64-bit local address separately, as `noc_wwrite_with_state` does, and clear the sticky state afterwards.
- Adding the tag to a `get_noc_addr()` result corrupts the coordinate field, which starts at bit 36.
- Atomics and inline writes also mask the high address and cannot carry the tag.

**Inline writes.** The Blackhole stateful inline-write API supports stream-register targets only and warns that L1 targets can hang. Inline writes to L1 are emulated: they always issue on command buffer 0 regardless of the argument, and the flush variant waits for all outstanding writes on that NoC. On DRISC, the emulation's scratch slot comes from `MEM_L1_INLINE_BASE` = 32 (`noc_nonblocking_api.h:541`, `dev_mem_map.h:108`), which lies inside DRISC's 64-byte reserved region; `MEM_DRISC_L1_INLINE_BASE` is defined but unused. Fix this before the DRISC executes `NOC_UNICAST_INLINE_WRITE` packets.

**Read sizes (pull only).** The stateful per-transaction-ID read API is a one-packet operation, bounded by Blackhole's 16 KiB NoC burst. A non-stateful any-length per-ID read already exists (`Noc::async_read` with `TXN_ID`, `noc.h:203-218`), so 16 KiB-plus-header packets are not blocked, only unavailable on the stateful path.

### 6.8 Early handoff for forward-plus-local packets

The current 2D action-map path admits all selected outputs, calls `forward_2d_dispatch`, then calls `forward_to_local_destination`. For a packet with both forwarding and local delivery, the DRISC handoff can be issued immediately after admission and before downstream copies. Include DRISC capacity in the all-output admission decision so an ERISC does not commit only part of the packet's fanout. With push, all copies are NoC writes from the same RX slot and the order only changes when the DRISC can start; with pull, DRISC reads and ERISC forwarding reads may contend for Ethernet L1 or NoC capacity. Measure either against a handoff issued after forwarding.

In 2D, landing-route edits occur before admission and downstream forwarding leaves the header unmodified, so all copies see a stable packet image. In 1D, forwarding updates routing fields in the RX header and the sparse multicast case advances `chip_idx`/`write_idx`; the existing order (local write before advance) must be preserved, which is another reason to keep 1D out of the first slice. Preserve per-VC local issue order.

### 6.9 Flush ordering for local delivery

ERISC forwarding writes the packet into another local ERISC's sender buffer; it does not perform that packet's final destination write. Today `flush_write_to_noc_pipeline` polls the ERISC's transaction IDs for both local writes and these forwarding copies, because they share the ERISC NoC/tracking machinery. That broad wait does not establish that the fused local write/atomic semantically depends on acknowledgement of forwarding-buffer copies, much less remote Ethernet delivery. The [multicast source-injection test](tests/tt_metal/tt_fabric/fabric_data_movement/kernels/test_mesh_multicast_source_inject_sender.cpp) explicitly describes `flush=true` as ordering a packet's payload before its matching atomic, without ordering different packets or API phases.

For the offload design, DRISC should issue and track local writes, wait for the required local write acknowledgements when the packet requests a flush, and then issue the local atomic or semaphore increment. ERISC independently tracks forwarding source consumption for Ethernet RX slot retirement; it does not need to report forwarding acknowledgement to DRISC for the local flush. Verify the intended scope for standalone flush atomics and earlier local writes during implementation.

Scatter chunk writes currently omit the VC argument and go on NoC VC1, while the scatter semaphore increment goes on the EDM VC (2 or 3) (`fabric_edm_packet_transmission.hpp:237-322`), so an unflushed scatter plus semaphore has no cross-VC ordering today. A DRISC port should preserve or fix this deliberately.

### 6.10 DRISC resources and system integration

**Resources:**

- **L1:** 128 KB. The unreserved region starts at about 0x8600; `DriscL1Arena` keeps a fixed sender-state zone at its base, leaving about 93.5 KiB for a resident kernel, shared with DRAM-sender state (`dev_mem_map.h:322-366`; `tt_metal/impl/buffers/drisc_l1_arena.hpp:52-68`). The §8 example of 1 µs holding time at 400 Gb/s needs about 50 KB of staging.
- **Code space:** firmware and kernel text share a 24 KiB window, and firmware's i-cache flush unrolls 3072 NOPs, about 12 KB (`tt_metal/hw/inc/internal/tt-1xx/risc_common.h:210-219`). That leaves about 11.2 KB for kernel text, plus about 6.8 KB of private data. Overflow is a load-time failure, not a build error. The ERISC RX router text was about 7.4 KB in one cached build, so a local-delivery handler should fit; watcher and debug builds are at risk, as the existing DRISC kernels already show.
- **Clock:** the DRISC wall clock was measured at exactly 1350 MHz, equal to AICLK (`STREAMING_PROFILER_FINDINGS.md:5005`). The core clock is not documented separately.
- **Historical NoC measurements** (harness since removed; DRISC reads from Tensix L1, one NIU): about 231 ns latency, an 86.3 GB/s port (64 B × 1.35 GHz), 34.1 and 45.7 GB/s for 2 KB reads at 8 and 16 outstanding, and 50.7 GB/s for 4 KB reads at 8 outstanding (`STREAMING_PROFILER_FINDINGS.md:241-308`). There is no data for DRISC ↔ Ethernet L1 traffic or for DRISC write bandwidth.

**System integration blockers:**

- **Ownership.** There is no core-level DRISC reservation. The tensor prefetcher requires both DRISCs on every bank, all 16 (`tt_metal/impl/buffers/tensor_prefetcher_manager.cpp:566-571`); the streaming profiler uses up to 8 relays on free-subchannel DRISCs, the same cores §6.5 recommends; `CreateDramKernel` has no ownership check. A fabric helper needs a reservation mechanism plus prefetcher and profiler changes, or mutual exclusion.
- **Lifecycle.** DRISCs are reset unconditionally on every Metal init and teardown (`tt_metal/impl/device/firmware/risc_firmware_initializer.cpp:531,544`), while fabric routers can be launched by an external Fabric Manager and outlive a process. Fabric teardown has no DRISC termination path.
- **Launch.** Fast dispatch cannot launch DRISC programs (`tt_metal/impl/program/dispatch.cpp:134-141`); existing users force slow-dispatch launch. Fabric's direct launch-message path would need to carry DRISC kernels, which nothing exercises yet.
- **Availability.** DRISC programmable cores are enabled only on a sufficiently new firmware bundle (19.12.0 per `STREAMING_PROFILER_FINDINGS.md`), overridable with `TT_METAL_ENABLE_BLACKHOLE_DRAM_PROGRAMMABLE_CORES` (`tt_metal/impl/context/metal_env.cpp:132`).

### Hardware constraints and redirected-read status

DRISC must use an initiating-capable NIU mode. [drisc_mode.h](tt_metal/hw/inc/experimental/drisc_mode.h) documents that stream mode enables initiation but changes that endpoint's inbound routing: it no longer provides the normal direct NoC-to-DRAM path. Firmware chooses the modes once per boot; on Metal-owned DRISCs this leaves no DRAM view's routing affected by NoC0 initiation.

An earlier discussion suggested directing a DRISC-issued read response straight from Ethernet L1 to the final destination. BH's [address-register documentation](https://github.com/tenstorrent/tt-isa-documentation/blob/main/BlackholeA0/NoC/MemoryMap.md#noc_targ_addr-and-noc_ret_addr) describes independent source and response coordinates, but this does not establish a usable, validated transfer mechanism here. Treat redirected reads as unproven and exclude them from the baseline model. The [counter documentation](https://github.com/tenstorrent/tt-isa-documentation/blob/main/BlackholeA0/NoC/Counters.md#read-requests) also shows why ordinary barriers would not work unchanged: initiation increments the issuer's outstanding count, while the response updates counters at its recipient.

### Proposed first design slice

0. **Gate.** Proceed only if the §9 measurements show RX is the critical stage in the target generic VC0+VC1 2D configuration, or if the target is ring-like traffic where §3.3 already shows it.
1. **Validation microtest.** An ERISC (or a Tensix core on NoC1) pushes packets into a free-subchannel DRISC's L1 on NoC1 and rings a stream-register doorbell; the DRISC delivers on NoC1 to Tensix L1 and returns credits to an ERISC stream register. Check doorbell ordering, the NoC1 register write to that NIU, per-packet DRISC service time, and Ethernet-L1 → DRISC → Tensix throughput.
2. **Host.** Behind the default-off flag, reserve one free-subchannel DRISC per active RX ERISC, mutually exclusive with the tensor prefetcher and streaming profiler for the prototype. Launch through the fabric's direct launch path, and add a DRISC termination path to fabric teardown.
3. **Router.** At the generic VC0/VC1 local-delivery call sites only: include cached DRISC credits in `admit_2d_dispatch` (reusing the UDM relay hook), issue the push write on the command buffer freed from local writes, ring the doorbell, and retire the RX slot when the push and forwarding writes have been sent. Choose a DRISC-credit stream id that does not collide with VC2 (§6.2).
4. **DRISC handler, first cut.** Unicast writes to Tensix L1; per-VC rings; one transaction ID per staging slot; staging slots retire after their outgoing writes are sent; batched credits.
5. **Complete the operation set** handled by `execute_chip_unicast_to_local_chip_impl`: inline writes (after fixing the scratch slot), standalone atomics, fused write-plus-atomic, scatter (decide its VC), and sparse multicast. Under the enabled path, all of these go to DRISC in receiver order. Test flush semantics and VC0/VC1 interleaving as each is added.
6. **Measure** against the same configuration with offload disabled and against the tuned-ERISC variants in §5. Sweep DRISC staging depth and doorbell batching.

## 7. DRISC throughput sensitivity model

The first version started from `C0 = 144` reference cycles/packet. That row is the 1D speedy path (§3); generic baselines are 186–198 (1D Linear) and about 210 (2D Mesh multicast, aggregate), and the target configuration is unmeasured. The model also only applies where RX is the critical stage (§3.3). The illustrative tables in §7.3 keep `C0 = 144`; replace it with a measurement from the exact selected path before using the model for a decision.

Define:

- `D`: current local-handling work removed from ERISC.
- `HE`: replacement ERISC handoff and helper-credit work.
- `HD`: additional DRISC staging/coordination work beyond the migrated work.
- `alpha`: DRISC execution time for migrated work relative to ERISC.

```text
CE = C0 - D + HE
CD = alpha * D + HD
Ccontrol = max(CE, CD)
```

All terms are service times expressed in 1.35 GHz reference cycles.

### 7.1 Code-derived per-packet estimates

These are estimates from reading the code, not measurements. `TT_FABRIC_PROFILE_RX_CH_FWD` and an ablation build that skips the local write would measure `D` directly.

ERISC cost of one local unicast write today:

| Step | Code | ~Cycles |
| --- | --- | ---: |
| Packed load of payload size and send type | `fabric_edm_packet_transmission.hpp:345` | 7–8 |
| Sparse check, range check | `:152`, `:173` | 2–3 |
| Jump table: entry load from Ethernet L1, indirect jump | `:177` | 10–12 |
| 64-bit destination NoC address load | `:178` | 8–10 |
| Wait for command buffer ready | `dataflow_api.h:2807` | 5–10 |
| 7 NoC register writes | `noc_nonblocking_api.h:602` onward | 7–14 |
| Software counter updates (Blackhole workaround) | `dataflow_api.h:2809-2813` | 6–8 |
| **D** | | **≈45–65** |

Other estimates:

- **4-chunk scatter:** `D` ≈ 110–140, consistent with the +50 to +64 cycles scatter adds on the RX-critical ring rows.
- **Whole RX step for a local-delivery packet:** about 120–160 (stream-register poll and cache invalidate, header and route-buffer loads, action decode, admission, transaction-ID bookkeeping, `D`, sent check, Ethernet completion acknowledgement), plus about 30–45 per forwarding direction (payload write and credit increment).
- **TX per packet sent:** about 60–90 (free-slot read, header load, Ethernet send, transmit-queue spin, remote pointer update, local credit update, completion processing), plus about 10–20 per enabled sender channel visited in each loop iteration.
- **Push handoff `HE`:** about 20–30 (command-buffer wait, about four stateful register writes, doorbell, amortized credits).
- **Pull handoff `HE`:** about 12–18 (doorbell, amortized read-complete and credit checks).
- **DRISC overhead `HD`:** about 15–25 for push (doorbell poll, fence, slot bookkeeping, batched credit return, staging-slot reuse tracking); about 35–50 for pull (adds read issue, completion wait, and report).
- **DRISC per-packet total (push):** about 60–90 for a plain write and 125–165 for scatter, below all current generic intervals.

These estimates are consistent with the hypothetical inputs used in §7.3.

### 7.2 Relative DRISC speed (alpha)

A figure of "~12.6 cycles per volatile DRISC L1 load" (`STREAMING_PROFILER_FINDINGS.md:335`) was cited during the discussion as evidence that DRISC is slower. It is not a load-latency measurement: an ablation found about 63 extra cycles per core for an "examine" step, which the document divides by five loads although the 63 cycles include the adds and loop overhead, and the NIU was near port saturation at the time. The same document points the other way: a load-plus-store copy loop ran at about 11 cycles per word including loop overhead (`:1114-1116`), and dropping `volatile` from a scan made no difference (`:2893-2902`). The per-read issue floor of about 40 cycles is likewise a property of that harness, not a measured DRISC/ERISC difference.

Use `alpha ≈ 1` as the working assumption until a like-for-like microbenchmark (a dependent load chain and a NoC-write issue loop, run on both cores and measured by ablation, since `get_timestamp()` costs about 26 cycles on DRISC) says otherwise.

### 7.3 Illustrative model (C0 = 144, alpha = 1)

Hypothetical overhead inputs, not calibrated from goldens:

| Scheme | HE | HD |
| --- | ---: | ---: |
| Push full packet | 20 | 20 |
| Whole-packet pull | 10 | 40 |
| Split reads | 10 | 60 |

Resulting control intervals:

| Movable work D | Push full packet | Whole-packet pull | Split reads |
| --- | ---: | ---: | ---: |
| 20 | 144 | 134 | 134 |
| 40 | 124 | 114 | 114 |
| 60 | 104 | 100 | 120 |
| 80 | 100 | 120 | 140 |
| 100 | 120 | 140 | 160 |

This sweeps possible decompositions of a fixed 144-cycle baseline. It does not model increasing operation complexity while holding the remaining work constant, and it omits the slot-hold penalty that §6.1 assigns to pull.

Under these assumptions, offload improves control throughput only if both stages beat the baseline. For `alpha = 1`, this requires `D > HE` and `D + HD < C0`. Moving work can overload DRISC; more offload is not always better.

The balanced point is `D = (C0 + HE - HD) / 2`, if that much work is actually movable. Its interval is `(C0 + HE + HD) / 2`:

| Scheme | Balanced D | Best interval in this model | Control speedup |
| --- | ---: | ---: | ---: |
| Push full packet | 72 | 92 | 1.57× |
| Whole-packet pull | 57 | 97 | 1.48× |
| Split reads | 47 | 107 | 1.35× |

These are conditional mathematical optima, not predicted hardware gains.

For `P` payload bytes, `Bcontrol = 1.35 * P / Ccontrol` GB/s. At `D = 60`, capped at 50 GB/s for the ideal 400 Gb/s comparison:

| Scheme | Cycles/packet | 1 KiB GB/s | 2 KiB GB/s | 4 KiB GB/s |
| --- | ---: | ---: | ---: | ---: |
| Original model | 144 | 9.60 | 19.20 | 38.40 |
| Push full packet | 104 | 13.29 | 26.58 | 50.00 |
| Whole-packet pull | 100 | 13.82 | 27.65 | 50.00 |
| Split reads | 120 | 11.52 | 23.04 | 46.08 |

The 1 KiB baseline is extrapolated. Values at 50 GB/s are capped ideal ceilings, not claims of attainable payload bandwidth after Ethernet overhead. Even a perfect zero-overhead division of 144 cycles between two cores gives 72 cycles/packet, which cannot reach the 55.3-cycle target for 2 KiB at 400 Gb/s or 1 KiB at 200 Gb/s. Those cases need less total per-packet work, more processing parallelism, or evidence that the measured baseline contains avoidable waiting.

### 7.4 Expected gains where RX is critical

Using the 1D ring rows, where §3.3 shows RX at or near the critical stage, and the §7.1 estimates (upper bounds; TX or the credit loop may cap them first):

- **Plain writes** save about 20–35 cycles on the ERISC: HalfRing about 176 → 141–156 cycles/packet (+13–25%), FullRing about 308 → 273–288 (+7–13%).
- **Scatter** moves almost entirely off the ERISC, so its interval approaches the offloaded plain-write interval: about +45–60% on HalfRing and +28–35% on FullRing.
- **Against a tuned ERISC** (§5), plain writes keep only about 10–20 cycles of advantage; multi-write operations keep most of it.
- **Where RX has slack** (single-flow paths, the measured 2D VC0-only Mesh multicast), expect roughly no gain.

## 8. Limits outside the CPU model

If unchanged TX, producer, forwarding, or credit handling imposes interval `CU`, the actual interval cannot be smaller than `CU`. For example, a 120-cycle TX limit caps the nominal 100-cycle pull design at 23.04 GB/s for 2 KiB and 46.08 GB/s for 4 KiB. An unchanged 144-cycle limit erases its throughput gain entirely.

**Credit loops.** By Little's law, a channel with `N` slots, each held for `T_loop` cycles, cannot sustain an interval below `T_loop / N`. The mux v2 buffer sweep (§4) shows this regime at shallow depth. In 2D Mesh without first-level acknowledgement, a worker-facing sender slot is held for a full Ethernet round trip plus remote RX processing; the latency golden puts one hop at about 0.7–0.9 µs. Any scheme that lengthens RX slot hold time, as pull does (§6.1), raises this floor. A slot-depth sweep would show whether a given configuration is in this regime: if the interval scales as `1/N`, it is latency-bound.

All three staging schemes add a second payload-transfer leg:

| Ethernet target | Ethernet → DRISC | DRISC → destination |
| --- | ---: | ---: |
| 200 Gb/s | 25 GB/s | 25 GB/s |
| 400 Gb/s | 50 GB/s | 50 GB/s |

At 400 Gb/s, DRISC L1 must accommodate roughly 50 GB/s of incoming payload writes and 50 GB/s of outgoing payload reads. This is 100 GB/s of aggregate payload movement, not necessarily 100 GB/s on one NoC link. One NIU port is 86.3 GB/s per direction, so one DRISC serving two 400 Gb/s links on one NIU would exceed it. Shared L1/NIU contention and placement still need investigation.

For `N` staging slots with holding time `Lslot` in reference cycles:

```text
Ceffective >= max(CE, CD, CU,
                  1.35 * P / Bingress,
                  1.35 * P / Begress,
                  Lslot / N)
```

Bandwidth terms use GB/s. This is a necessary-resource lower bound on interval, not a sufficient model of shared-resource contention or ordering stalls. Ethernet serialization adds its own bound.

Equivalently, minimum buffering for byte rate `B` and holding time `L` seconds is `N >= ceil(B * L / P)`. If holding time is 1 microsecond at 400 Gb/s, payload occupancy is 50 KB: at least 13 slots at 4 KiB or 49 slots at 1 KiB, before headers, alignment, and headroom. RX and DRISC pools require separate calculations because their holding times differ.

## 9. Revised assessment and next evidence needed

**Assessment:**

- **CPU feasibility: plausible.** With `alpha ≈ 1`, a stream-register doorbell, and push, the ERISC saves about 20–35 cycles per plain write and about 80–120 per 4-chunk scatter, and the DRISC stays below current generic intervals (§7.1).
- **Benefit depends on RX being the critical stage.** That is shown for 1D ring traffic, not shown for the measured 2D VC0-only Mesh multicast (at least about 50 cycles of slack), and unmeasured for the target generic VC0+VC1 2D configuration, where `-Os`, more channels, and transit forwarding make it plausible (§3.3).
- **Preferred design:** push into free-subchannel DRISCs on NoC1, stream-register doorbells and credits, NoC1 final delivery with unchanged packet addresses, and today's RX slot retirement rule (§6.1, §6.2, §6.5).
- **System blockers remain:** no DRISC reservation (the prefetcher uses all 16), DRISC resets on every init and teardown, slow-dispatch-only launch, no teardown path, and a cap of 8 helpers per chip with the NoC1 approach (§6.10).
- **The fused-atomic penalty is unexplained** and may not be addressable by offload (§3).

**Gating measurements, in order:**

1. Re-run the goldens on HEAD for generic 2D, both VC0-only and VC0+VC1, at 2 and 4 KiB.
2. `TT_FABRIC_PROFILE_RX_CH_FWD=1` in the target configuration: compare RX busy cycles per packet with the measured interval, for plain write and scatter.
3. Write versus scatter in the target configuration with multiple flows and forward-plus-local traffic. A +50-cycle scatter penalty means RX is the critical stage.
4. `TT_METAL_FABRIC_OPT_LEVEL=O3` with VC1 enabled, if it fits in code space. A throughput gain means the router CPU is the limit; combine with step 3 to separate RX from TX.
5. `TT_METAL_FABRIC_TELEMETRY` with `stats=BANDWIDTH` for TX versus RX active cycles, as a coarse cross-check.
6. A slot-depth sweep to identify credit-loop-bound configurations.
7. An ablation build that skips the local write but still retires the slot, to measure `D` and the upper bound of any offload.
8. Profile the 320–360-cycle fused-atomic penalty.
9. The §6 validation microtest (NoC1 doorbell to a free-subchannel DRISC, ordering, DRISC service time, Ethernet-L1 → DRISC → Tensix throughput), plus a like-for-like DRISC/ERISC microbenchmark for `alpha`.

**Related findings outside the proposal:**

- VC2 speedy transaction-ID flip bug (§2).
- DRISC inline-write scratch slot inside the reserved region (§6.7).
- Scatter chunk writes and semaphore increment on different NoC VCs (§6.9).

No implementation or performance improvement is established by these notes. They provide the budgets, evidence, and assumptions needed to refine and measure the proposed prototype.
