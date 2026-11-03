# Send batching on E0: P3, load overlap, P1, and a per-VC design (2026-11-01)

Lead: `variants/v4rxv2i3`. Host bh-lb-120-a09u32. E0 = sender ERISC, E1 = receiver ERISC. Background: V4_BOUNDS.md §1-§3
(bound map, E0 split), §6 (TX-queue probe), §7 (proposals P1-P4). Rules: a bound or a cause is a hypothesis until an
isolating A/B; static counts do not rank changes; readability first ("we shouldn't hurt readability at all").

Facts carry job numbers; hypotheses are labelled. All probes ran in benchmark mode (no payload validation).

## 1. Load overlap: what header loads cost in a batch (jobs 361, 362; microbenchmark)

`tests/tt_metal/tt_fabric/erisc_load_ubench` (RESULTS §17's kernel), new patterns HB_* / HW_* / HP_* / HX_* (kernel,
common header, host table; host target rebuilt alone). ERISC0, fabric off, data cache off, each block one aligned asm
loop; two Ethernet cores (index 0 and 3) agree to 0.01 cycle; every RESULTS §17 / V4_STATE §7 pattern reproduces exactly
in the same runs (the control). `consolidation/sb_ubench_runs.sh`; results/sb_ubench_e{0,3}.csv (job 362; job 361's
first set: sb_ubench_j361_*).

Every new block starts as a sender pass does: the count read (four free-slots stream registers as a group, 6 filler,
all four used). H0..H3 = the payload-size halfword (offset 40) of four ring slots 4448 B apart; DS, DS2 = local-memory
words holding H0's / H1's slot address (a forwarding channel's parked slot); W = seven TX-queue register writes (TX queue
1's staging registers only, never the command register: nothing is sent). Cost = cycles per block minus the same block
without the loads measured; K = independent filler between issue and use.

| pattern (after the count read) | K=0 | K=2 | K=4 | K=6 | K=8 | K=16 | K=48 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| H0, K, use | 2 | 1 | 2 | 2 | 2 | 2 | 2 |
| H0 H1 together, K, use | 2 | 3 | 4 | 4 | 4 | 4 | 4 |
| H0 H1 H2 together, K, use | 2 | 4 | 5 | 5 | 5 | 5 | 5 |
| H0..H3 together, K, use | 2 | 0 | 0 | 6 | 6 | 6 | 6 |
| H0 used at once | 2 | 2 | 3 | 3 | 3 | 3 | 3 |
| H0 at once, H1 at once | 10 | 10 | 11 | 11 | 11 | 11 | 11 |
| H0..H3 each at once | 26 | 26 | 27 | 27 | 27 | 27 | 27 |
| DS -> H0, K, use (parked slot, one channel) | 5 | 4 | 5 | 5 | 6 | 6 | 6 |
| DS DS2 -> H0 H1 together, K, use | 5 | 6 | 7 | 7 | 7 | 7 | 7 |
| DS -> H0 at once, DS2 -> H1 at once | 15 | 15 | 16 | 16 | 16 | 16 | 16 |
| H1, then W, K, use (one-ahead, issued before the writes) | 0 | 2 | 2 | 2 | 2 | 2 | 2 |
| 4 writes, H1, 3 writes, K, use | 5 | 4 | 4 | 5 | 5 | 5 | 5 |
| W, then H1, K, use | 8 | 4 | 4 | 4 | 5 | 5 | 5 |
| DS2 -> H1, then W, K, use | 3 | 5 | 5 | 5 | 5 | 5 | 5 |
| H1 at once, then W | 6 | 3 | 3 | 3 | 3 | 3 | 3 |
| post-send group: TX-queue status (T, Q) + completion counter, K, use | 3 | 4 | 5 | 5 | 5 | 5 | 5 |
| the same + H1 (the next header in the post-send group) | 3 | 1 | 0 | 0 | 6 | 6 | 6 |
| lone L1 load (lw or lhu; H0+40 or §17's word), K, use | 2 | 1 | 2 | 2 | 2 | 2 | 2 |
| the same without the count read before it | 8 | 7 | 7 | 7 | 8 | 8 | 8 |

Facts:
- Header loads issued together overlap like other reads: 2 cost 4, 3 cost 5, 4 cost 6 (K >= 5); used one at a time
  each costs ~8 (the first after the count read only 2-3; then 8 each: 11 for two, 27 for four).
- The first L1 load right after the count read costs 2-3, whatever its form (lw / lhu, offset 40 / 0, either L1
  word); the same load without the count read before it costs 7-8 as in RESULTS §17. Mechanism not isolated
  (hypothesis: the stream group's last response and the L1 read share part of their latency). In the router a send's
  header load is never the first L1 load after the pass's count read (the group itself reads the connection
  semaphores in L1), so the router-relevant figure for a header used at once is taken as ~8 (hypothesis; the PC
  samples' 6.6 at the header's first use on ew4352, V4_BOUNDS §3, are consistent with it).
- The parked-slot chain (forwarding channels: slot address in local memory, then the header) costs 5-6 for one, 7 for
  two issued together, 16 for two one after the other.
- A one-ahead header issued before the TX-queue writes costs 2 (the writes do not wait for it); issued after or among
  them 4-5. Added to the post-send group it costs +1 (6 against 5; 0 at K = 4-6).
- So in a batch the header loads cost about one cycle each when they ride in a group (the count read's, or the
  post-send group), against ~8 each used at once. In today's send the header already rides in the pre-send group with
  the TX-queue status read, which is needed anyway.

The send's own pre-send group (job 365, same kernel, patterns HS_*; both cores identical; results/sb_ubench_e{0,3}.csv),
after the count read, used at once:

| pattern | K=0 | K>=4 |
| --- | ---: | ---: |
| TX-queue status (T discarded, Q) | 2 | 3 |
| header + status | 3 | 4 |
| two sends' pre-send groups (header + status, used; again) | 12 | 13 |

The header adds 1 cycle to the status read it rides with; a second send's group costs 9 (the status read's ~8, used at
once, plus the header's 1). Decision: in a batch, keep each packet's header load in its send's pre-send group (today's
form). Grouping the K headers after the count read or loading the next one ahead could save at most that ~1 cycle per
packet; the ~8 belong to the TX-queue status read, which every send must make (never issue a TX-queue command while it
is busy, ROUTER_MAP obligation 3). Both alternatives would add state (K sizes, or a loop-carried prefetched size) for it.

## 2. P3: index-based slot advance (`variants/v4sbi`)

v4rxv2i3 with the sender's ring positions as ring indices (`Worker::index`, `ForwardingRecord::index`,
`FarRing::next_index`) and, in the receiver's vocabulary, `Channel<CH>::slot_addr(index)` / `next_index(index)` and the
same pair on `FarRingTraits<RX>` (the wrap test against `num_slots`, a small immediate; the address `first_slot_addr +
index * channel_buffer_size`). router_sender.hpp only; final readable form (-7 lines of state, +10 of helpers).

Static (coverage builds `v4sbi_O3` vs `v4rxv2i3_O3`, 2,170 / 2,366 configs build in both, the same 196 fail in both;
sender loop region: instructions / stack loads / stack stores, `txloop.py` in the session scratchpad):

| case | lead loop insns / sp | P3 change |
| --- | --- | --- |
| bh_2d, glx_2d (and telemetry, codeprof, delay) | 509 / 65 | +4 / +1 / +2 |
| bh_2d_vc2 | 665 / 101 | -3 / +2 / -1 |
| bh_1d, glx_1d | 265 / 35 | +4 / +4 / 0 |
| bh_ring, glx_ring | 329 / 44 | -5 / +2 / +1 |
| bh_torus, glx_torus | 651 / 95 | -4 / 0 / +4 |
| bh_vc1 | 898 / 118 | -4 / +3 / +1 |
| glx_express | 810 / 118 | -4..-6 / +1..+2 / +2..+3 |
| one ERISC (whole sender_pass) | | +3..+16 insns, no stack change |

Where it moved on 2D E0 (bh_2d__10043895322277694146, send<0>): the two last-slot reloads are gone (`lw 8(sp)` /
`lw 12(sp)` before each compare), but the worker channel's index now lives at 8(sp) (`lw` before the address, `sw` after
the advance) and the far ring's first-slot address at 12(sp) (`lw` before the destination add), and each address is a
`mul` by the slot stride (kept in a register) plus that base. So per send: still two stack loads, one store more, two
`mul`s, and the source address (stack load, mul, add) now sits in front of the header load. Fewer 32-bit constants live
did not leave the freed registers to the send state (the allocator's choice; not designed for, by rule).

Device (row A/B, jobs 363 and 364: pass a lead / P3 / P1-K2, pass b in reverse order; every pair within 0.1;
`sb_rows_runs.sh`, `sb_rows_report.py a b`), cycles/packet:

| row | lead | P3 `v4sbi` | |
| --- | ---: | ---: | ---: |
| ew16 (E0-bound) | 115.8 | 117.4 | +1.7 |
| bew1024 (E0) | 116.4 | 118.3 | +1.9 |
| ew4352 (E0) | 126.9 | 127.3 | +0.4 |
| ns4352 (E0) | 127.1 | 127.6 | +0.5 |
| hop3 (E0) | 146.2 | 145.0 | -1.2 |
| mcast (E0) | 141.7 | 140.6 | -1.1 |
| bidir | 135.2 | 135.1 | -0.1 |
| fwd / fwdshare (merge, 4352 B) | 125.7 | 125.7 | 0.0 |
| fwdshare16 (merge, 16 B) | 113.0 | 114.1 | +1.1 |
| burstew8 / burstns32 | 129.1 / 139.8 | 129.8 / 140.0 | +0.7 / +0.2 |
| ewscatter, lin1, ring1, torusns (E1 or wire) | 139.7, 124.4, 140.9, 149.2 | 139.8, 124.4, 141.0, 148.8 | within 0.4 |

P3 loses 1.7-1.9 on the E0-bound small writes it was meant to help and gains 1.1-1.2 on 3-hop and multicast: not a
candidate, so no census (it also loses 4.1-4.3 on every VC2-on row and ~3.7 on VC1 1-hop, §3b / §3e). Consistent with the static picture (the reloads moved, they did not go; a hypothesis for the
size of the small-write loss: the source address now waits on a stack load and a `mul` in front of the header load).

## 3. P1 probe: several packets per worker visit (`variants/v4sbk2`, `v4sbk4`; PROBE, not a candidate)

v4rxv2i3 + (router_sender.hpp only): the pass's group read also yields each worker channel's packet count
(`packets_held<CH>`: ring size less the free count); `visit_worker` sends up to `worker_sends_per_visit` (2, or 4 in
v4sbk4) of those packets, one `send` at a time while the far ring has room for each (bubble flow control per packet),
each with its own packets-received update; it stops at a busy TX queue and never waits for packets. Completions,
credits and the per-pass connection check are as on the lead. Applies to both worker channels (VC0's and VC2's), not to
forwarding channels. Header loads stay in each send's pre-send group, so only packets the count read confirmed are
read. Builds every config the lead builds (2,170 / 2,366, `v4sbk2_O3`); 2D sender loop: same instruction count, 12 fewer
stack loads, 1 more store (static; does not rank).

Rows (jobs 363, 364; K2 in both passes, within 0.1; K4 pass b only), cycles/packet:

| row | lead | K2 | | K4 | |
| --- | ---: | ---: | ---: | ---: | ---: |
| ew16 | 115.8 | 112.6 | -3.2 | 112.6 | -3.1 |
| bew1024 | 116.4 | 114.2 | -2.3 | 114.3 | -2.2 |
| hop3 | 146.2 | 142.6 | -3.6 | 142.6 | -3.6 |
| mcast | 141.7 | 139.6 | -2.1 | | |
| burstew8 | 129.1 | 127.7 | -1.4 | 127.6 | -1.5 |
| burstns32 | 139.8 | 140.3 | +0.5 | 140.4 | +0.6 |
| ew4352 | 126.9 | 128.9 | +2.0 | 129.2 | +2.3 |
| ns4352 | 127.1 | 129.2 | +2.1 | 129.5 | +2.4 |
| bidir | 135.2 | 138.0 | +2.8 | 138.1 | +2.9 |
| fwd (merge, 4352 B, sync) | 125.7 | 126.6 | +0.9 | | |
| fwdshare (merge, 4352 B, link) | 125.7 | 126.6 | +0.9 | 128.3 | +2.6 |
| fwdshare16 (merge, 16 B, link) | 113.0 | 111.1 | -1.9 | 115.3 | +2.3 |
| ewscatter, lin1, ring1, torusns | 139.7, 124.4, 140.9, 149.2 | 140.5, 124.4, 140.6, 148.9 | +0.7..-0.3 | 140.6 (ring1) | -0.2 |

Fairness on the merge row (`sb/fwdshare*.yaml` = pcs3/fwd.yaml with sync off: chip [0,0]'s flow forwarded through
[0,1] and [0,1]'s own worker share [0,1]'s east link; each sender's own cycles from the profiler's log lines, 20M
(4352 B) / 40M (16 B) packets each), cycles/packet per sender (the one that finishes first ran under contention
throughout):

| row | lead [0,1] worker / [0,0] forwarded | K2 | K4 |
| --- | --- | --- | --- |
| fwdshare 4352 B | 251.4 / 251.4 | 220.6 / 253.3 | 167.6 / 256.6 |
| fwdshare16 16 B | 226.0 / 226.0 | 174.4 / 222.3 | 141.6 / 230.6 |

The lead splits the link evenly. Batching the worker channel alone gives the worker the larger share: about 57% (4352 B)
and 64% (16 B) at K2, 77% and 83% at K4 (estimate: the forwarded flow's packets while both ran = its total less its solo
tail at the single-flow rate, 126.9 / 112.6). Any batching form must give every channel of a VC the same bound.

Delay perturbation and the bound after batching (job 365; `sb_delay_runs.sh a`, routers `v4bdd` = the lead + the study
hooks and `v4sbk2d` = K2 + the same hooks; at 0 every image equals K2's, 220 / 220 in six cases, `bnd_loadcmp.py`;
short rows `sb/ps_*.yaml`), change in cycles/packet against base (a spin of N costs about N + 12 where its side has no
slack; tx = after each send, rx = after each packet received, sp = after each sender pass):

| row | router | base | tx8 | tx16 | tx32 | rx8 | rx16 | rx32 | sp16 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ew16 | lead | 115.8 | +29.2 | +29.1 | +50.2 | +10.1 | +10.8 | +26.9 | +29.9 |
| ew16 | K2 | 112.6 | +8.0 | +8.3 | +24.5 | +12.7 | +13.4 | +29.9 | +3.5 |
| ew4352 | lead | 126.9 | +15.4 | +15.3 | +38.8 | +3.6 | +4.0 | +17.4 | +16.0 |
| ew4352 | K2 | 128.9 | +3.0 | +5.1 | +10.9 | +4.3 | +4.6 | +16.1 | +1.9 |
| hop3 | lead | 146.2 | +35.1 | +35.1 | +58.4 | +3.9 | +5.0 | +17.4 | +32.5 |
| hop3 | K2 | 142.6 | +8.8 | +26.4 | +41.3 | +7.0 | +7.9 | +21.0 | +24.6 |
| fwd (merge) | lead | 125.7 | +17.7 | +17.8 | +40.9 | +5.5 | +6.1 | +23.4 | +1.6 |
| burstew8 | lead | 129.1 | +16.4 | +16.4 | +36.9 | +4.3 | +4.9 | +14.6 | +16.7 |

(The hook's 8-cycle spin costs what its 16-cycle spin does on every lead row: its loop polls the cycle counter, so 8 is
not a separate point; the 16 -> 32 step is the clean slope.)

Facts (one-side A/B):
- Lead: E0 binds ew16, ew4352, hop3 and burstew8 (tx32 +38.8..+58.4, slope 16 -> 32 of 1.3-1.5); E1 has ~17 slack on
  ew16 (rx32 +26.9) and ~26 on ew4352 (as V4_BOUNDS §1c).
- K2: on ew16 and ew4352 neither ERISC binds any more: tx32 costs 24.5 / 10.9 (E0 slack ~20 / ~33), rx32 29.9 / 16.1
  (E1 slack ~14 / ~28), and a per-pass spin costs 3.5 / 1.9 per packet against 29.9 / 16.0 on the lead. ~~Batching took
  E0 off the critical path~~ (WRONG for ew16, job 369: the K1 control, same code without the second send, runs ew16 at
  the same 112.6; see the K1 table below), and the rows settled at another limit: 112.6 (16 B) and 128.9 (4352 B,
  slower than the lead's E0-bound 126.9).
- hop3 under K2 stays sender-sensitive (tx32 +41.3): only the first hop's worker channel batches; the downstream
  routers forward from forwarding channels, which K2 does not batch.

Where the small-write limit sits: in 801 EW_PureWrite16 runs in results/ (every study router since v1, DRISC offload
builds included) no single-flow E/W 16 B write ran below 112.3 cycles/packet (NS_PureWrite16: 113.7; BurstEW_N1_G0:
113.4); the two-producer forwarded 16 B row of the DRISC builds ran at 61-68 for the link. Hypothesis (window test in
job 369): a single producer's flow of 16 B packets is limited at ~112 by the producer side (the worker kernel's
per-packet injection or its credit window), not by either ERISC; the lead's ew16 (115.8) was 3-4 above it, so batching
could win at most that much there, and K2/K4 did (112.6). Settled by job 369 (below): it is the worker channel's
window (17 slots); the injection-rate alternative is WRONG (with 32 worker slots K2 runs ew16 at 100.1).

Remaining sweep rows (job 369; `v4sbk2d` only, the lead's from job 365):

| row | router | base | tx8 | tx16 | tx32 | rx8 | rx16 | rx32 | sp16 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| burstew8 | K2 | 127.7 | +3.1 | +3.3 | +14.4 | +4.3 | +4.6 | +15.4 | +1.2 |
| fwd (merge) | K2 | 126.6 | +4.6 | +13.5 | +27.3 | +6.2 | +7.0 | +23.5 | +5.7 |

Robustness: no row shows a step on either side beyond the hook's 8-vs-16 quantum (16 -> 32 slopes 0.6-1.5 on both
routers; K2 absorbs small TX spins in its slack, as a non-binding side should). hop3 under K2 is the one irregular
response (tx8 +8.8, tx16 +26.4: the spin there also lands on the forwarding routers, which K2 does not batch).

### 3b. What the gain and the loss are made of (job 369)

K1 control, `variants/v4sbk1` = K2 with `worker_sends_per_visit = 1` (the same code shape: counts from the group read, the
loop; it never sends a second packet; static 2D loop +3 insns, -5 stack loads against the lead), same job as the lead and
K2:

| row | lead | K1 | | K2 | |
| --- | ---: | ---: | ---: | ---: | ---: |
| ew16 | 115.7 | 112.6 | -3.1 | 112.5 | -3.2 |
| hop3 | 146.2 | 143.5 | -2.8 | 142.6 | -3.6 |
| fwdshare16 (merge) | 113.0 | 108.8 | -4.2 | 111.1 | -1.9 |
| ew4352 | 126.8 | 126.9 | +0.1 | 128.8 | +2.0 |
| ns4352 | 127.2 | 127.2 | 0.0 | 129.3 | +2.1 |
| bidir | 135.1 | 135.7 | +0.5 | 138.0 | +2.8 |

Window arms (`TT_FABRIC_EXP_SLOTS_2D`; slots logged: worker / forwarding / receiver; same job):

| row | slots | lead | K2 |
| --- | --- | ---: | ---: |
| ew16 | 17 / 16 / 16 (default) | 115.6 | 112.5 |
| ew16 | 32 / 8 / 16 (deeper worker window) | 113.1 | **100.1** |
| ew16 | 17 / 8 / 32 (deeper far ring) | 115.6 | 112.5 |
| ew4352 | default | 126.8 | 128.9 |
| ew4352 | 32 / 8 / 16 | 125.7 | 125.7 |
| ew4352 | 17 / 8 / 32 | 126.8 | 128.9 |

Facts:
- The ew16 gain of K2/K4 at the default slots is not batching: K1, which never sends a second packet, gets the same
  -3.1 (and -2.8 on hop3, -4.2 on the 16 B merge row, where batching the worker alone then costs back 2.3). The code
  shape of the probe moved these rows. Where (2D E0, bh_2d__10043895322277694146, by source line, `linediff.py` in the
  session scratchpad): in K1 every send's slot advance and packets-received update lost their stack reloads (worker
  send: slot advance 13 insns / 3 stack accesses -> 10 / 0, far advance 10 / 2 -> 6 / 0, register update 7 / 1 -> 6 / 0;
  each forwarding send -1 stack access), paid for in cold code (connection event, upstream credit +3, round loop +1).
  K1's semantics equal the lead's (the worker's "holds a packet" taken as a count != 0 instead of a bit of `pending`),
  so this is the register allocator placing the send path differently: exactly what P3 was after, reached by an
  unrelated edit. Cause = allocation, a hypothesis (static correlation); by rule nothing is designed on it.
- At the default 17 worker slots, a single 16 B flow is window-bound at ~112.5 once E0 is fast enough (K1, K2, K4 all
  sit there; the deeper far ring changes nothing). With 32 worker slots K2 runs ew16 at 100.1, the lead at 113.1 (still
  E0-bound): -12.8 against the lead, of which batching is about half (next table); it only shows where the worker
  window is deeper than the 2D default. K1 / K4 at 32 slots (job 378, same job as the lead and K2; 32 / 8 / 16 logged):

  | row | lead | K1 | K2 | K4 |
  | --- | ---: | ---: | ---: | ---: |
  | ew16 | 112.9 | 106.6 | 100.1 | 99.2 |
  | bew1024 | 113.0 | 106.7 | 100.4 | 99.8 |
  | ew4352 | 125.7 | 125.7 | 125.7 | 125.7 |

  With the window out of the way the two effects separate: the probe's code shape is worth -6.3 (K1, still one send
  per visit), a second send per visit another -6.3..-6.5 (K2 against K1), a third and fourth -0.6..-0.9 (K4).
- The +2 of K2 on the 4352 B 1-hop rows comes with batching (K1 flat) and disappears with 32 worker slots (both 125.7);
  the deeper far ring does not change it. So batching lengthens the worker's credit loop at 4352 B (hypothesis for the
  mechanism; not isolated).

VC2 compiled in (`TT_METAL_ENABLE_FABRIC_VC2=1`, slots 17/14/16 + 2/4 logged; job 369), cycles/packet:

| row | lead | P3 | | K2 | |
| --- | ---: | ---: | ---: | ---: | ---: |
| ew4352 (VC0 traffic) | 134.6 | 138.9 | +4.3 | 133.3 | -1.3 |
| torusns | 166.9 | 171.2 | +4.3 | 167.0 | +0.1 |
| vc02ew4352: VC0 sender (finishes first) | 163.7 | 167.8 | +4.1 | 142.0 | -21.7 |
| vc02ew4352: VC2 sender | 877.2 | 893.9 | +16.7 | 922.5 | +45.3 |
| vc02bew4352: VC2 sender (finishes first) | 987.9 | 1006.6 | +18.6 | 1132.1 | +144.1 |
| vc02bew4352: VC0 sender | 143.3 | 147.4 | +4.1 | 136.9 | -6.4 |

- P3 loses 4.1-4.3 on every VC2-on row (the most register-pressed two-ERISC sender): a second reason it is not a
  candidate.
- K2 batches both worker channels, but VC2's worker has 2 slots (window-bound alone at ~860-990) and rarely holds two
  packets, while VC0's does: VC0 takes a larger share of the shared link (its sender -21.7 / -6.4; VC2's +45 / +144).
  Equal per-channel bounds do not give equal shares when the channels' windows differ.

Tooling check (coordinator's note on offline builds and live-tree swaps, 2026-11-01): the overlay pins a variant's own
mirrored files under tt_metal/hw/inc (dataflow_api.h, both noc_nonblocking_api.h; every v4sb* variant carries the lead's
copies); other hw/inc files come from the live tree. My offline builds ran with no swap active or with only my own jobs'
swaps (v4rxv2i3 family: the same three files, pinned); the v4sbm2 build (20:06:38-20:07:26) ran after job 366 restored
the tree (20:05:51) and before job 367 started (20:09:33). Re-checked: the lead rebuilt now (`sblead_O3`, 312 ELFs in 9
cases) equals the 03:22 baseline `v4rxv2i3_O3` byte for byte, and `v4sbm2` rebuilt now (`sbm2chk_O3`, 248 ELFs) equals
its first build. The static counts and the hook identity (v4sbk2d = v4sbk2) stand.

### 3c. Every channel batched with the same bound (`variants/v4sbm2`; PROBE for the per-VC design's merge question)

v4sbk2 extended to every VC0 channel and VC2's worker: the group read yields every channel's packet count, and
`visit_forwarding` too sends up to 2 of its counted packets per visit (`sends_per_visit = 2` for every channel; VC1's
channels send one). Static 2D loop +72 insns, +9/+9 stack loads/stores (the forwarding visits' sends grow). Builds the
same configs (248 / 248 in 8 cases, VC1+VC2 refused by the lead's gate). Job 373, same job as the lead:

| row | lead | m2 | |
| --- | ---: | ---: | ---: |
| ew16 | 115.7 | 112.9 | -2.8 |
| bew1024 | 116.4 | 114.6 | -1.8 |
| hop3 | 146.2 | 144.2 | -2.0 |
| mcast | 141.7 | 136.3 | -5.4 |
| burstew8 | 129.1 | 128.0 | -1.1 |
| ew4352 / ns4352 | 126.8 / 127.2 | 128.0 / 128.5 | +1.1 / +1.3 |
| bidir | 135.1 | 137.0 | +1.8 |
| ring1 | 140.8 | 140.8 | 0.0 |
| fwdshare (merge 4352 B; link / each sender) | 125.7 / 251.4, 251.4 | 125.7 / 251.4, 251.4 | 0.0 |
| fwdshare16 (merge 16 B; link / each sender) | 113.0 / 226.0, 226.0 | 106.3 / 212.5, 212.5 | -6.7 |

Equal bounds keep the merge fair (both senders 212.5); the 16 B merge (two producers, E0-bound at the merging router)
gains 6.7. Code-shape control m1 (`variants/v4sbm1` = m2 at 1 send per visit), job 381, same job as the lead and m2:

| row | lead | m1 | | m2 | | batching (m2 - m1) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ew16 | 115.7 | 112.9 | -2.9 | 112.9 | -2.8 | 0.0 |
| hop3 | 146.2 | 144.2 | -2.1 | 144.2 | -2.0 | 0.0 |
| mcast | 141.7 | 140.3 | -1.3 | 136.4 | -5.3 | -4.0 |
| fwdshare16 (link; each sender) | 113.0; 226.0 | 110.4; 220.9 | -2.6 | 106.2; 212.4 | -6.8 | -4.2 (fair) |
| ew4352 | 126.8 | 127.0 | +0.2 | 128.0 | +1.1 | +0.9 |
| bidir | 135.1 | 135.8 | +0.6 | 137.0 | +1.9 | +1.2 |

(m2's rows reproduce job 373 within 0.1.)

### 3d. Latency (job 373; exp_latency.yaml, one packet at a time, so no visit ever holds two packets: code shape only)

Median net round trip, ns (1000 samples each):

| case | lead | K2 | m2 |
| --- | ---: | ---: | ---: |
| 1-hop E/W write 16 B / 4096 B | 1344 / 1690 | -6 / -5 | -5 / -6 |
| 1-hop N/S write 16 B / 4096 B | 1336 / 1684 | -2 / -1 | +1 / -2 |
| 1-hop E/W fused atomic 16 B / 4096 B | 1410 / 1687 | -13 / +7 | 0 / +7 |
| 1-hop N/S fused atomic 16 B / 4096 B | 1404 / 1690 | -15 / +1 | -6 / -3 |
| 3-hop E/W write 16 B / 4096 B | 3307 / 4233 | -36 / -2 | +6 / +40 |
| 3-hop E/W fused atomic 16 B / 4096 B | 3377 / 4249 | -39 / -2 | +14 / +28 |

K2 is neutral to better (3-hop 16 B -36..-39 ns); m2's forwarding visits cost the 3-hop 4 KiB round trip +28..+40 ns
(a code-shape effect on every forwarding hop; hypothesis, single run).

### 3e. The VC1 split (job 373; census_vc1_runs.sh pass sba: lead, P3, K2, m2 in one job; slowest sender per test)

| row (cycles/packet) | lead | P3 | K2 | m2 |
| --- | ---: | ---: | ---: | ---: |
| VC1Mesh1HopEW 16 B / 4352 B | 141.9 / 142.0 | 145.7 / 145.8 | 125.9 / 128.3 | 125.9 / 128.6 |
| VC1Mesh1HopNS 16 B / 4352 B | 142.0 / 142.0 | 145.7 / 145.7 | 131.0 / 132.9 | 131.1 / 133.2 |
| VC1BurstEW N8 16 B / 4352 B | 142.2 / 142.3 | 145.9 / 145.9 | 130.0 / 132.9 | 130.0 / 133.4 |
| VC1BurstEW N32 16 B / 4352 B | 142.2 / 142.1 | 145.9 / 145.9 | 127.9 / 131.7 | 128.3 / 131.4 |
| VC1Mesh2HopMerge 16 B / 4352 B | 139.5 / 144.3 | 139.7 / 144.2 | 150.9 / 157.6 | 145.2 / 151.0 |
| VC1Mesh2Hop 16 B / 4352 B | 239.4 / 258.0 | 237.5 / 256.3 | 237.1 / 254.8 | 240.0 / 259.3 |
| VC1MeshMulticast 4352 B | 153.2 | 153.0 | 164.3 | 161.4 |
| suite mean (13 rows) | 159.3 | 161.3 (+1.3%) | 154.2 (-3.2%) | 153.8 (-3.5%) |
| inter-mesh suite mean (8 rows, window-bound) | 336.7 | 335.8 | 334.8 | 335.2 |

- P3 loses ~3.7 on every VC1 1-hop / burst row (VC1 E0 is the most register-pressed sender).
- K2 / m2 take the E0-bound VC1 1-hop rows from ~142 to 126-133 (-9 to -16), the largest leverage seen. Split by the
  controls (job 382, census_vc1_runs.sh pass sbc: lead, K1, m1, K2 in one job; K2 reproduces job 373 within 0.2):

  | row | lead | K1 | m1 | K2 | batching (K2 - K1) |
  | --- | ---: | ---: | ---: | ---: | ---: |
  | VC1Mesh1HopEW 16 B / 4352 B | 142.0 / 142.0 | 136.4 / 136.5 | 136.8 / 136.9 | 125.9 / 128.4 | -10.5 / -8.1 |
  | VC1Mesh1HopNS 16 B / 4352 B | 142.1 / 142.1 | 136.4 / 136.4 | 136.8 / 136.9 | 131.0 / 133.0 | -5.4 / -3.4 |
  | VC1BurstEW N8 / N32, 16 B | 142.2 / 142.1 | 136.6 / 136.6 | 137.1 / 137.0 | 130.0 / 127.9 | -6.6 / -8.7 |
  | VC1Mesh2HopMerge 16 B / 4352 B | 139.4 / 144.2 | 140.4 / 144.0 | 140.3 / 144.1 | 150.7 / 157.6 | +10.3 / +13.6 |
  | VC1MeshMulticast 4352 B | 153.2 | 154.0 | 153.7 | 164.5 | +10.5 |
  | suite mean | 159.3 | 155.7 | 156.0 | 154.3 | |

  So on VC1 1-hop the code shape is worth ~-5.6 and batching proper another -3.4..-10.5; on the VC1 merge and multicast
  batching (not the code shape) costs +10..+14 with the worker alone batched, and +5..+8 with equal bounds (m2, job 373,
  against m1 here).
- The 3-flow VC1 merge (VC1Mesh2HopMerge: three senders into [0,3], the slowest reported) and VC1 multicast get worse
  (+11..+13 with K2, +6..+8 with m2): the merging router's worker batched alone starves the forwarded flows (K2), and
  even with equal per-channel bounds a channel that carries two upstream flows gets one share for both (m2), as it does
  on the lead; why m2 is slower than the lead there is not isolated (hypothesis: longer passes delay the forwarding
  channels' credits upstream).

## 4. Summary

- P3 (index-based slot advance, receiver's vocabulary): not a candidate. ew16 / bew1024 +1.7 / +1.9, VC2-on rows +4.1..+4.3,
  VC1 1-hop +3.7, hop3 / mcast -1.2 / -1.1 (jobs 363, 364, 369, 373). The two per-send stack reloads moved (worker
  index and far-ring base now spilled, two `mul`s on the address path), they did not go.
- Header loads (jobs 361, 362, 365): grouped loads overlap (~1 cycle per extra header); in the send's pre-send group the
  header adds 1 cycle to the TX-queue status read every send makes. Keep the header load where it is; a group or a
  one-ahead prefetch could save at most ~1 cycle per packet and adds state.
- P1 (several packets per visit): batching proper is worth -6.5 per packet on 16 B / 1 KiB single flows once the worker
  window is out of the way (32 worker slots: K2 100.1 vs K1 106.6 vs lead 112.9), -3.4..-10.5 on VC1 1-hop, -4.0 on 2D
  multicast and -4.2 on the 16 B merge with equal per-channel bounds; K4 adds <1 over K2. At today's 2D slots the
  16 B single flow is window-bound (17 worker slots, ~112.5) and batching adds nothing there. It costs +0.9..+2.1 on
  4352 B 1-hop rows and +1.2..+2.3 on bidir (worker credit loop: gone with 32 worker slots), +5..+14 on the VC1 2-hop
  merge and +8..+11 on VC1 multicast (not isolated), and skews link shares toward the channel that batches or holds more
  (worker alone: 57-83% of a merge; VC0 vs VC2's 2-slot worker). Latency: neutral (one packet at a time never batches).
- The probes' code shape alone (K1 / m1: identical semantics to the lead) moved E0-bound rows by -2..-6 (2D small
  writes, hop3, the 16 B merge, VC1 1-hop -5.6): the allocator kept the send path's constants in registers (static, by
  source line). Hypothesis: placement; not a design input by rule.
- The per-VC design proposal (rounds over a VC's channels against one budget) is in the session report, not here
  (design discussion stays in chat); its measured inputs are the tables above.

## 5. Jobs, labels, files

| job | what | labels | device time |
| --- | --- | --- | --- |
| 361, 362 | ubench: send-batch patterns, then load-form diagnostics | results/sb_ubench_j361_*, sb_ubench_j362_* | 1 + 1 min |
| 363 | rows pass a: lead, P3, K2 (16 rows) | sba_* | 16 min |
| 364 | rows pass b: K4, K2, P3, lead (11 rows) | sbb_* | 12 min |
| 365 | ubench pre-send group (HS_*), delay sweep lead (v4bdd) / K2 (v4sbk2d), 5 short rows; stopped on purpose before K2's last two rows (at the run times then seen it would have passed the broker timeout; ps_fwd.yaml was parked so the missing-yaml guard stopped it cleanly with the tree restored; those rows ran in 369) | sb_ubench_e{0,3}, sbda_* | 13 min |
| 369 | sweep remainder (K2: fwd, burstew8); K1 control; window arms w32 / r32; VC2-on rows (lead, P3, K2) | sbda_*, sbc_*, sbw32_*, sbr32_*, sbv2_* | 16 min |
| 373 | m2 rows; latency (lead, K2, m2); VC1 census (lead, P3, K2, m2) | sbm_*, sblat_*, censbavc1[i]_* | 12 min |
| 378 | 32 worker slots: lead, K1, K2, K4 | sbw32b_* | 4 min |
| 381 | m1 control (lead, m1, m2) | sbmc_* | 8 min |
| 382 | VC1 census controls (lead, K1, m1, K2) | censbcvc1[i]_* | 4 min |

New files: variants/v4sbi (P3), v4sbk1 / v4sbk2 / v4sbk4 (P1 probe, worker channels, K = 1 / 2 / 4), v4sbk2d (K2 + the
study hooks), v4sbm1 / v4sbm2 (every channel, bound 1 / 2); consolidation/sb_ubench_runs.sh, sb_rows_runs.sh,
sb_rows_report.py, sb_delay_runs.sh, sb_delay_report.py, sb/*.yaml (fwdshare, fwdshare16, ps_*); microbenchmark patterns
in tests/tt_metal/tt_fabric/erisc_load_ubench (kernels/erisc_load_overlap_common.hpp, kernels/erisc_load_overlap.cpp,
test_erisc_load_overlap.cpp; target rebuilt). Coverage builds (cov_m): v4sbi_O3, v4sbk2_O3 (full), v4sbk1_O3, v4sbk2d_O3,
v4sbm1_O3, v4sbm2_O3, sblead_O3, sbm2chk_O3 (subsets). Caches ~/.tt-metal-cache-sb-*, -sbd-a-*, -cenvc1-v4sb* (24 GB).

## User decisions (2026-11-02)

- P3 (index-based slot advance) is dropped.
- K1's code-shape gain (-3 to -6 on E0-bound rows) is allocator placement, not designed on.
- Batching is parked until the slot decisions (2D worker depth). It is re-tested on the chosen split; if revived,
  the simpler equal per-channel bound (about +8 lines), not per-VC rounds.
- Order: the sender ownership pass first, shaped by the register-budget analysis (receiver agent, 2026-11-02);
  then a fresh look at per-packet send cost on the new shape. Two busy VCs on one P150 link total ~115
  cycles/packet at 256 B-1 KiB, the same as one VC (jobs 389/390), so the sender's per-packet work is the ceiling
  for multi-VC fan-in.
