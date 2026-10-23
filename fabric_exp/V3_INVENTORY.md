# Router v3: what goes in

v3 is main's router (`fabric_erisc_router.cpp` at 453ee7bdfa7, 3,492 lines) with only the changes we have decided on,
baked in with no experiment flags and no DRISC offload. This file lists everything the study tree changes relative to
main and marks each item **keep**, **drop** or **decide**. Section numbers refer to `RESULTS.md`.

How the list was built: `git diff main HEAD` over `tt_metal/fabric/` (21 files, about 4,100 added lines).
- The 26 experiment bits come from `fabric_phase_profile_types.hpp`.
- The shared headers and the host builder were read hunk by hunk.
- The v1 router diff was walked hunk by hunk. With every flag off, no delays and the profiler off, it changes no
  behavior. It leaves only dead globals and stores (about 70–80 B) and an equivalent local-delivery wrapper.

## 1. Experiment bits

The best ERISC-only set is 13129088 = bits 7, 8, 10, 12, 14, 19, 22 and 23. Against main it takes 1-hop from 222
to 132 cycles/packet, 3-hop +41%, multicast +18% (§44).

| Bit | What | Section | v3 |
| --- | --- | --- | --- |
| 7 | Lean completions: read a sender channel's completion/ack counter only while it has packets not yet completed | §17, §20 | keep |
| 8 | Read the free-slot stream registers of up to 4 VC0 sender channels together at the top of the loop | §17 | keep |
| 10 | No write-counter update on send (nothing reads it) | §20 | keep |
| 12 | Advance the cached slot address by stride instead of index wrap plus table lookup | §20 | keep |
| 14 | Read the packet size before the TX-queue can-send check. v2 also groups it with the CMD/STATUS reads and returns `false` when the queue is busy | §20, §51, §52 | keep (v2 form) |
| 19 | Skip idle forwarding channels, per channel (router-to-router channels with counter credits only) | §23 | keep |
| 22 | E1 batched completions: 16 transaction IDs on receiver channel 0, completion acks batched up to 4 | §26, §33, §34 | keep |
| 23 | Overlap the post-send TX-queue check with the completion-counter read | §28 | keep |
| 0, 1, 4, 17, 20 | Skip or replace local delivery, push emulations | §22, §24 | drop (benchmark only, break delivery) |
| 2 | Batch upstream credit returns | — | drop |
| 3 | Lazy connection check (every 16th visit) | — | drop |
| 5 | Speedy path: drop the unused TX-queue read | — | decide (speedy path only; not measured in 2D) |
| 6 | Skip the TX-queue busy check before send | — | drop (hangs) |
| 9 | Upstream credit work on E1 | §17 | drop |
| 11 | Early TX-queue read in the send | §20 | drop (superseded by 14 and 23) |
| 13 | Compile-time free-slot stream ID in the send | §20 | drop |
| 15, 16 | E1 grouped register reads | §21 | decide (parked: −5 to −8 on E1-bound paths, costs latency; not in the best set) |
| 18 | Skip all idle forwarding channels with one branch | §23 | drop (superseded by 19) |
| 21 | DRISC offload | §38–§43 | drop (out of scope) |
| 24 | Lean VC0 loop (compile-time bounds, lean channel type) | §32 | drop |
| 25 | `[[likely]]` on the idle skip | §32 | drop (cost +7.8; `[[likely]]` cost again in §49) |
| E1 27 | Stateful NoC1 forwarding on two-ERISC E/W routers (one live direction) | §53 | keep (hard-wired by the user; 3-hop −13, multicast −12) |
| E1 28 (§53) | Compile-time admit stream reads, local-only early return | §53 | keep (hard-wired by the user; N/S decode 36 → 20) |
| E1 variant A | Local delivery takes payload size and send type from one word-40 load | §56 | keep (folded in by the user) |
| E1 26 (§57) | Pre-admission `src_ch_id` load widened to word 40, reused by local delivery | §57 | keep candidate (with the hold) |
| E1 28 (§57) | Hold 1–3 packet completion tails for 4 consecutive idle polls | §57 | keep candidate (small writes −4 to −5; 3-hop 16 B latency +25 ns) |

## 2. v2 structure (§47–§52)

| Item | v3 |
| --- | --- |
| Separate worker step (channel 0, connection check after the send) and forwarding step (no liveness check) | keep |
| `sender_channel_try_send` owns the whole send, including bubble flow control; `sender_channel_process_completions` owns completions and credit return | keep |
| Grouped pre-send loads (size, CMD, STATUS); a busy queue returns `false` | keep |
| Fenced load groups (`std::atomic_signal_fence`) for bits 8, 14 and 23 | decide (relies on GCC behavior, §48) |
| `[[likely]]` early return, semaphore read before the send, spinning on a busy queue | drop (§49, §51) |

## 3. Shared-header changes not behind a flag (they affect the ERISCs today)

| Where | Change | Origin | v3 |
| --- | --- | --- | --- |
| `edm_fabric_utils.hpp` `send_chunk_from_address_with_trid` | Blackhole, non-debug: one command-buffer poll, then `ncrisc_noc_write_with_state<..., skip_cmdbuf_chk>`, instead of the stock call that polls again | §39 | decide (check that the second poll is redundant) |
| `fabric_edm_packet_transmission.hpp` scatter local writes | Chunk writes pass `forward_and_local_write_noc_vc` (main: default VC), matching the other local-write types | §39 | decide |
| `fabric_erisc_datamover_channels.hpp` | Sender channel stores first/last slot address and stride | bit 12 | keep with bit 12 |
| `fabric_router_flow_control.hpp` | `note_packet_sent` / `has_*_pending` / `packets_sent`; `accumulate/flush_completion_credits` | bits 7, 22 | keep with those bits |
| `edm_fabric_flow_control_helpers.hpp`, transaction-ID tracker | `_ct` remote pointer advance, prefetched packet count, `transaction_sent_reg_addr` | bits 24, 15 | drop |
| `fabric_router_adapter.hpp`, `fabric_stream_regs.hpp`, `tt_fabric_api.h` | DRISC-only code, default-preserving template argument | bit 21 | drop |
| `fabric_erisc_router_speedy_path.hpp` | Profiler hooks and bit 5/6 gates only | — | drop |

## 4. Host side and settings

| Item | Today | v3 |
| --- | --- | --- |
| Ethernet keepalive (`ETH_TXQ_LOCAL_SEQ_UPDATE_TIMEOUT`) | Env `TT_FABRIC_EXP_SEQ_UPDATE_TIMEOUT`, default off (firmware 8000). Every study run used 1000; ERISC 0 sets it before the handshake and restores it at teardown | decide (1000 removes a one-way floor above 4114 B frames, §15, §16) |
| Kernel selector | `TT_FABRIC_KERNEL_VERSION=2` | keep; add `=3` |
| Phase profiler, stack paint, delays, microbench | Builder args and env | decide (leave out of v3 and keep v1/v2 as the profiling build, or keep the hooks) |
| Slot caps, W32, receiver amortization env, config logging | Allocator/builder env | drop |
| DRISC pairing, DRISC kernel creation, `fabric_init.cpp` DRISC zeroing | Builder | drop |
| BW telemetry switch | Test harness (`TT_FABRIC_NO_BW_TELEMETRY`), not the router | test harness only |

## 5. Production paths our runs never compiled

Every study run was Blackhole, 2D mesh, two ERISCs, VC0 only, with these all off: deadlock avoidance, first-level
ack, speedy VC0, tensix mux, telemetry, data cache, spin-wait send. v3 has to keep main's code for each of the paths
below, so these production branches stay. Where an optimization applies, it needs at least a validation run there.

| Path | Main has it | Decided changes that touch it |
| --- | --- | --- |
| Wormhole | yes | Bits 14 and 23 and the grouped pre-send loads are Blackhole-only; everything else applies |
| 1D line / ring, neighbor exchange (speedy path) | yes | Speedy path untouched unless bit 5 is kept; the generic 1D path gets the TX changes |
| VC1 / VC2 (2D with VC1 active) | yes | Bit 8 groups VC0 only; bit 22 needs one receiver channel |
| Deadlock avoidance (ring, torus): bubble flow control, 8 transaction IDs | yes | Bit 22 is off with it; v2's try-send carries bubble flow control |
| First-level ack | yes | Bit 7 has an ack-pending check; bit 19 needs it off |
| Stream-register credits (not counter credits) | yes | Bits 7 and 23 fall back to main's reads |
| Single ERISC (`NUM_ACTIVE_ERISCS == 1`) | yes | E0 then does RX too; the yield-on-busy and grouped reads still apply |
| Telemetry on (the harness default in perf mode) | yes | Bit 23 path calls `update_bw_counters` |
| UDM header, tensix mux extension, channel-trimming capture, context switch, link retrain | yes | No decided change; keep main's code |

## 6. Suggested build order

Each step: reduced suite (validation, ERISC-only 4352/mix/mix_trim/latency, plus 2D validation) against main and
against v1 at 13129088. Judge by suite means; layout moves single rows by ±5–10.
1. v3 = main's router, selector `=3`. Must measure as main.
2. Send-side bits 10, 12, 14 (with the v2 grouped pre-send and yield), then 23.
3. Bits 7, 8, then 19 (19 needs 7 and 8).
4. The worker/forwarding step split.
5. Bit 22 (E1).
6. Header items from §3, once decided.
7. Keepalive, once decided.
8. Validation on the paths in §5 the hardware can run: 1D line/ring, neighbor exchange, torus if available, telemetry on.
