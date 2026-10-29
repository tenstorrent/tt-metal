# v4 channel trimming: port note

Status (2026-10-29): built and validated (`variants/v4trm` = v4tel + §3; a09u32 jobs 159-163; RESULTS §98): main's
channel-trimming gtests 63/63 on v4trm as on main, two ERISCs and one; every untrimmed config byte-identical to v4tel.
Sources: main at the merge (6d8a103): `fabric_erisc_router.cpp` (`record_sender_channel_usage`,
`record_2d_forwarded_slots`, `run_receiver_channel_step_impl`'s forwarding-off block, the downstream open gates),
`fabric_trimming.hpp`, `fabric_trimming_types.hpp`, `erisc_datamover_builder.cpp` (`apply_channel_trimming_overrides`,
`compute_disable_rx_forwarding`, the open gates), `channel_trimming_import.cpp`; V4_FEATURE_SPEC.md §12, §14.1.

## 1. What trimming asks of a router

Two runs of the same traffic. **Capture**: the router records, in an L1 block the host reads after the run, which
sender channels sent (and the min / max packet size, header included), which receiver channels took a packet, which
NoC send types it wrote locally per VC, and per receiving VC the downstream routers' sender channels it forwarded to.
**Replay**: the host rebuilds every router from the capture:
- sender channels that never sent and receiver channels that never took a packet are not serviced;
- a VC whose receiver neither forwarded nor wrote locally gets `DISABLE_RX_CHx_FORWARDING`;
- a VC0 that only injects or only terminates may drop deadlock avoidance and, worker-only, take the speedy path;
- the downstream connections are opened by the host's gates (`OPEN_DOWNSTREAM_VC{0,1}_CONNECTIONS`), and a trimmed
  sender channel does not wait for its static upstream connection.

## 2. v4 before the port (v4tel)

| Item | Main | v4tel |
| --- | --- | --- |
| Capture: sender channel used, packet size | per send | per send (`sender::send`) |
| Capture: receiver channel took a packet | per admitted packet | per packet (`receive`, `receive_local`) |
| Capture: NoC send types written locally | per local write | per local write (`deliver`) |
| Capture: forwarded-to sender channels | per forward (2D: compact slots through the host's packed ids; 1D: channel 1) | **missing** |
| Replay: unserviced channels | compiled out | compiled out (channel masks) |
| Replay: forwarding off on a generic receiver channel | packet consumed and credited, nothing forwarded or written | **ignored** (forwarded or written as usual) |
| Replay: speedy path from a capture | speedy loop | v4spd's delivery-only receiver (`delivers_only`) |
| Replay: open gates | `OPEN_DOWNSTREAM_VC0/VC1_CONNECTIONS` | receiver 0 serviced; VC1: receiver 1 serviced (**differs on a crossover router whose receiver 1 is trimmed**: its receiver 0 forwards on VC1) |
| Replay: static connection wait of a trimmed sender | `SENDER_CH_i_WAIT_STATIC_CONNECTION` gated | same |

| Replay: an ERISC trimming leaves nothing to service (two ERISCs) | the main loop with every step compiled out | **refused by the bring-up gate** (exactly one role per ERISC) |
| Replay: one ERISC left with one role or none | the main loop | **refused by the gate** (both roles required) |

Without forwarded-to masks a capture says no router forwards; hypothesis (not run): the host would then turn
forwarding off on transit routers and may treat a transit VC0 as terminal-only (deadlock avoidance off, speedy), which
breaks multi-hop replay. The gate rows were found on hardware (job 160: replay scenarios failed to compile, 75 router
builds; every one an ERISC with no serviced channel) and are fixed in item 4 below.

## 3. v4trm

1. `receiver::record_forwarded<CH>(to)` after each forward (`receive`, `sparse_multicast_hop`): 2D maps the
   forward's downstream names to compact slots and those to the downstream routers' sender channels with the host's
   packed table (`PACKED_DOWNSTREAM_VC0/VC1_SENDER_CHANNEL_IDS`, chosen by the channel's downstream VC: VC1's under
   crossover, as main), merged into the receiving channel's row; 1D sets channel 1. Capture builds only (`if
   constexpr`).
2. `Channel::drops` (forwarding off, not the speedy or VC2 delivery-only channel): `receive` consumes the packet
   through `receive_local<CH, false>` (no decode, admission, forward or write; still credited). Main admits before
   dropping; nothing is sent either way.
3. The downstream opens use the host's gates.
4. Trimmed role shapes: on two ERISCs an ERISC with nothing to service runs `run_idle` (router_round.hpp: rounds
   without passes, so the master still runs router commands, and both follow the context switch and termination); one
   ERISC runs `single::run` whatever roles are left (an empty role's pass does nothing); receiver channel 0, and the
   VC1 / VC2 receiver channels, run only when serviced. The gate admits these shapes.

Every config without a capture or a trimmed replay compiles as v4tel (the gates equal v4tel's conditions there;
`drops` is false): checked offline.

## 4. Validation

1. Offline (cov_m, -O3): 1,260 of 1,390 configs build (the 130 others are the `_sc` cases, failing by design, as on
   v4tel); 1,174 images byte-identical to v4tel's, the 86 others the capture configs (bh_2d_trim 28, bh_1d_trim 10,
   bh_2d / bh_1d / bh_ring_allinstr 28 / 10 / 10: the receiver ERISC's images, which now record forwarded-to).
2. Device (`trim_runs.sh`; fabric_unit_tests `ChannelTrimming*`: the capture bitfields, traffic with fixed override
   profiles, and the 11 capture-then-replay scenarios on Mesh and Torus X, which check the captured shape before
   replaying):

   | Router | Two ERISCs | One ERISC |
   | --- | --- | --- |
   | main (kernel v1, flags 0) | 63/63 (job 159) | 63/63 (job 162) |
   | v4trm | 63/63 (job 161; job 160, before item 4: the capture tests passed, replay scenarios failed, their routers refused by the gate; 01-04 seen, the summary was cut and the log overwritten) | 63/63 (job 163) |

   3 T3K-only tests skip on this host. Trimmed shapes the runs built: two ERISCs, an idle ERISC 0 in 66 configs and an
   idle ERISC 1 in 70; one ERISC, 60 with no role, 10 sender-only, 6 receiver-only. On this host main's replay now
   works for multi-hop traffic too (the 11 scenarios; the pre-merge note "1-hop only", job 282, is superseded).
3. Replay configs for offline builds: the v4trm runs ingested as coverage cases `bh_trim_gtest` (316 distinct configs)
   and `bh_trim_gtest_1erisc` (`coverage.py ingest`; the kernel caches `~/.tt-metal-cache-trim-v4trm-{two,one}` must
   stay). v4trm builds 316/316; v4tel 224/316 (the rest: the gate rows above).
4. Not measured: trimmed throughput (exp_2d_trim rows) on v4; perf is parked (V4_PERF_GAPS.md).
