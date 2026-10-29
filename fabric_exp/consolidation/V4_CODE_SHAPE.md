# v4 code shape after the features (2026-10-29)

User (2026-10-29): with the features in, the code shape may have degraded (lost optimizations, less readable code); take
a closer look and find opportunities. Lead `variants/v4trm`; reference `variants/v4rs` (the lead before the features).

## 1. Generated code: nothing lost on the configs measured before

Offline (cov_m, -O3, `objcopy -O binary` per image): every production config image of v4trm equals v4rs's: 2D, 1D, ring
and torus on two ERISCs (120), every one-ERISC case (60), mux mode (48), and the 22 VC1 receiver images. The only
difference is the 22 VC1 sender images (three stack slots permuted, from the VC2 work); their VC1 rows measure as v4gi's
(158 on 1-hop: census job 167 against the stored post-merge v4gi run, label `mpostvc1_v4gi`; cross-job). Each feature's code is compiled only into the
configs that turn it on (UDM, VC2, speedy, telemetry, trimming capture and replay); express needed none. So the census
gaps (V4_PERF_GAPS.md) are the router's shape, not feature cost.

## 2. Source

Router sources 2,908 -> 3,522 lines (+21%): router_telemetry.hpp 187 (new), router_sender.hpp +123 (VC2),
router_receiver.hpp +118 (VC2, speedy, trimming), router_downstream.hpp +88 (UDM), kernel +43, init +25, round +17,
single +13. Shared files against main (6d8a103): v4 needs +15 lines each in `fabric_erisc_router_ct_args.hpp` and
`fabric_erisc_datamover_channels.hpp`, +67 in the Blackhole `noc_nonblocking_api.h`, +28 in `dataflow_api.h`; the tree's
`fabric_edm_packet_transmission.hpp` carries +261 study lines, mostly DRISC-offload and deferred-atomic helpers v4 does
not call (v4 uses `execute_chip_unicast_to_local_chip_impl` from it).

## 3. Readability and structure (independent review of the v4trm source; "checked" = confirmed in the source here)

[NC] = no codegen change expected (verify with the byte-identity build); [HL] = may change hot-loop code.

| # | Finding | Where | Suggestion | Risk |
| --- | --- | --- | --- | --- |
| 1 | What the receiver does with a packet is split over four mechanisms: `delivers_only` (speedy), `drops` with a hand-written VC2 exception, `receive_local<CH, DELIVERS>`, and VC2's own `deliver_and_complete` | router_receiver.hpp | one `Channel::mode` (Route, DeliverOnly, Drop); DeliverOnly covers speedy VC0 and VC2 | NC |
| 2 | receiver `Channel` has no `vc`; VC2's channel takes VC1's first-level-ack flag (latent: the gate refuses VC1 acks) (checked) | router_receiver.hpp:77 | a `vc` member as sender::Channel has; derive the flags from it | NC |
| 3 | VC2 sender helpers duplicate VC0's: `vc2_connection_event` = `worker_connection_event` minus the preset; `credit_vc2_worker` = `credit_worker`'s non-preset branch with another source word (checked) | router_sender.hpp:344-407 | one `worker_connection_event<PRESET>`, one `credit_worker(source word)` | NC |
| 4 | VC2 sender state uses template tricks: a dummy-width `Vc2Interface`, `NoVc2Worker`, defaulted CH/RX parameters only to avoid instantiating `Channel<VC2>` | router_sender.hpp:137-155, 576-610 | a `WorkerChannel<CH>` record chosen with `conditional_t` | HL on VC2 builds |
| 5 | Five ways to say "feature absent" (`NoChannelState`, `NoVc2ChannelState`, `NoVc2Worker`, `NoRelay`, `NoRelayArgs`), a real unused `Downstreams<0>` stand-in for VC1, `[[no_unique_address]]` on one member and not its twin | router_receiver.hpp, router_single.hpp:37-41, router_init.hpp:31 | one `Absent<Tag>` | NC (HL if a layout changes) |
| 6 | The round epilogue is written twice (sender::run, single::run): `total_sent` twice per round, telemetry, two credit re-arms | router_sender.hpp:670-697, router_single.hpp:86-117 | one out-of-line `finish_round` | HL |
| 7 | Feature facts scattered: `has_vc1/has_vc2/is_*_role` in init, `has_vc2_worker` in sender, `runs_vc*_receiver` in receiver; the receiver re-derives `has_vc2` | several | one place in router_common.hpp | NC |
| 8 | Dead code: `for_each_bit` (no caller), `sender::Channel::far_receiver` (unused; callers pass RX by hand) (both checked) | router_common.hpp:212, router_sender.hpp:68 | remove; use `C::far_receiver` | NC |
| 9 | Stale or wrong comments: the empty-type comment is inverted (distinct empty types are what let `[[no_unique_address]]` members share an address) (checked); `total_sent` and `end_round` lost their comments; "tuned on Wormhole" reads as done; the kernel's 11-line feature sentence | router_receiver.hpp:134, router_sender.hpp:569, router_round.hpp:93, kernel:15-25 | fix | NC |
| 10 | Repeated casts where helpers exist: routing table (7 places, checked), termination signal, telemetry block; `init_telemetry` next to `telemetry::init` | several | helpers in common; rename | NC |
| 11 | Init duplication: `open_relay` repeats `open_downstream`; three counter-zeroing loops in `init_txqs` | router_downstream.hpp, router_init.hpp | share | NC |
| 12 | Wormhole branches the gate cannot admit yet; overlay comments about v2/v3; `rx_batch_completion`'s comment describes v3 | several | label "Wormhole bring-up" or drop; rename the flag | NC |
| 13 | Unguarded capture calls (empty when off) against the stated rule "guard every feature call site" (images are identical, so harmless today) | router_receiver.hpp:231, 281; router_local_delivery.hpp:93 | guard, or note why | NC |
| 14 | `receiver::run` assumes it is not the context-switch master (true: two ERISCs, ERISC 1) | router_receiver.hpp | `static_assert(!is_context_switch_master)` | NC |

## 4. Possible optimizations from the source (hypotheses; none measured; each needs an A/B)

1. Per-round `sent_at_round_start = total_sent(...)` runs even when `IDLE_CONTEXT_SWITCHING` is off (checked; a
   compile-time flag) [HL, per round: small].
2. Deadlock-avoiding routers re-program the credit address on every forward, as main does "for no stated reason" (checked)
   [HL; rings and tori: one command-buffer write per forward; needs a correctness check first].
3. Single-source receiver channels (VC2, speedy VC0) could take the source channel at compile time instead of per packet
   [HL].
4. Trimmed one-ERISC routers with one role make an empty out-of-line pass per pass [HL, one ERISC only].
5. Under mux mode `acknowledge_arrivals` loads each header's source channel though it is always 0 [HL, rare].
