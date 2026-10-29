# v4 telemetry: design note

Status: built and validated (2026-10-29; `variants/v4tel`, job 157; RESULTS §98): the dynamic-telemetry gtest on main and v4
in both ERISC modes, the perf recorder within 5% of measured bandwidth on unicast rows as main's. Call sites are
guarded by `if constexpr` (dead calls alone changed code in some loops).
Last of the user's feature order (UDM, VC2, speedy path, telemetry). Base: the lead,
`variants/v4spd`. Sources: main's router (`update_telemetry`, `update_bw_counters`, `update_bw_cycles`, the perf
recorder calls in the main loop), `telemetry/fabric_bandwidth_telemetry.hpp`, `hostdevcommon/fabric_telemetry_msgs.h`,
V4_FEATURE_SPEC.md §12. Telemetry-on runs are functional checks only; they are never scored (memory
feedback-no-bw-telemetry).

## 1. The two host options

| | Gate | Per packet | Per round (N passes) | Readers |
| --- | --- | --- | --- | --- |
| Perf recorder ("bandwidth telemetry" in test_tt_fabric) | `PERF_TELEMETRY_MODE` = LOW_RESOLUTION_BANDWIDTH (`enable_fabric_bw_telemetry`; test_tt_fabric's perf modes unless `TT_FABRIC_NO_BW_TELEMETRY`) | sender: packets += 1, words += bytes / 16 (truncated) | sender ERISC: a window from the round's start to its end; added to the L1 buffer at `PERF_TELEMETRY_BUFFER_ADDR` (busy cycles, words, packets) when the round sent a packet or a sender channel still holds one | test_tt_fabric's TelemetryManager |
| Dynamic telemetry | `FABRIC_TELEMETRY_ANY_DYNAMIC_STAT` (`TT_METAL_FABRIC_TELEMETRY`; stats mask) | BANDWIDTH: tx on the sender, rx on the receiver: packets += 1, words += (bytes + 3) / 4 | BANDWIDTH: elapsed cycles, and active cycles when the side made progress; ROUTER_STATE; HEARTBEAT_TX (progress, or no sender channel holds a packet); HEARTBEAT_RX (progress, or VC0's packets-received register is 0); all stored into the FabricTelemetry struct | `read_fabric_telemetry`, test_bandwidth_telemetry_validation |

Main keeps the round's recorder and a FabricTelemetry copy as loop locals and passes them into every step.

## 2. v4 design

One header, `router/router_telemetry.hpp`. Everything is compile-time: with both options off no code and no state
(checked offline: every telemetry-off image byte-identical).
- State in one global (`telemetry::counters`, an inline variable only odr-used when an option is on), not in loop
  locals: the loops' register state does not change when an option is on, and nothing is added to the off build.
  It holds the running packet and word counts of each side and the round's start (clock, counts).
- Per packet: `telemetry::count_sent(bytes)` in `send` after a send; `telemetry::count_received(bytes)` in the receive
  paths after admission (forward, deliver, relay, delivery-only). Both empty when off.
- Per round: `telemetry::round_start()` before the passes and a side's `round_end` after `end_round` in each loop
  (sender, receiver, single-ERISC: both sides), out of line. Progress is the side's packet count against its count at
  the round's start. "A sender channel holds a packet" reads the serviced channels' free-slots registers as the pass
  does; "VC0's packets-received register is 0" reads it.
- Init: the perf buffer cleared on the sender role; the FabricTelemetry struct as today (`init_telemetry`).
- Which ERISC writes what, as main: two ERISCs: the sender ERISC tx bandwidth and `erisc[MY_ERISC_ID]` (its state
  and TX heartbeat), the receiver ERISC rx bandwidth and its entry; one ERISC: both.

## 3. Validation

1. Offline: the telemetry configs build (2D / 1D dynamic stats, perf recorder, all instruments; both ERISC modes);
   every other config's image byte-identical to v4spd.
2. Device: test_tt_fabric benchmarks with the perf recorder on (the harness's telemetry bandwidth against its own),
   main's router as the same-job control; `test_bandwidth_telemetry_validation` (dynamic bandwidth telemetry within
   15% of measured) on main and v4; a dynamic-stats read after a run (heartbeats moving, router state RUNNING).
