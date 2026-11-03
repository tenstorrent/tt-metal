# v4 lead bound map, command-buffer audit, TX-queue probe (2026-11-01)

Router: `variants/v4rxv2i3` (the temporary lead). Host bh-lb-120-a09u32. E0 = sender ERISC, E1 = receiver ERISC.
Earlier map: V4_PROFILE.md §1-§6 (v4trm, jobs 172-179). Rules: bound side only from one-side A/Bs; causes are
hypotheses until a variant changing only that piece reproduces them; PC samples split cycles within one ERISC only.

## 0. Instruments

| Instrument | What | Proof / cost |
| --- | --- | --- |
| `variants/v4bdd` | v4rxv2i3 + v4trmd's hooks: TT_FABRIC_EXP_TX_DELAY spin after each send (router_sender.hpp send(), after the packets-received update), TT_FABRIC_EXP_RX_DELAY after each packet handled (router_receiver.hpp handle(), after retire), TT_FABRIC_EXP_DRISC_DELAY per pass (low 16 bits end of sender pass, high 16 bits end of receiver pass) | at 0 every image byte-identical to v4rxv2i3's (all loadable sections, `bnd_loadcmp.py`): -O3 1818 / 1818, -Os 1818 / 1818 (cov_m builds `bddlead_*` = v4rxv2i3 rebuilt, `v4bdd_*`; 2,170 of 2,366 configs build in both, the same 196 fail in both: 130 `*_sc` by design, 66 VC1+VC2 by the lead's gate); the 352 images of the `*_delay` / `*_passdelay` cases build and differ, as intended |
| `bnd_delay_runs.sh GROUP` | per suite, arms (default base, tx32, rx32; ARMS=, SUITES=) in one job and cache; labels `bnd_<suite>_<arm>` | hang capture armed |
| `bnd_report.py`, `bnd_vs_trm.py` | side per row (prof_report.py's rule: a 32-cycle spin costs ~44 with no slack; slack = 44 - c32; "TX"/"RX" when only that side is under 8, "both", "-" neither), and against v4trm's prof_* | |
| `variants/v4bdq` + `bnd_txq_runs.sh`, `bnd_txq_report.py` | TX-queue probe (§6): v4rxv2i3 + router_txq_probe.hpp, compiled out unless TT_FABRIC_TXQ_COUNTERS=1 | attribution only; with the probe off its images equal v4rxv2i3's (220 / 220, 6 cases, `bdqoff_O3`); offline compile with the probe on: 120 / 120 (`bdqon_O3`, -DFABRIC_TXQ_PROBE_ADDR) |
| `bnd_pcs_runs.sh` | PC sampling (pcsv_runs.sh with row yamls from pcs3/ or bnd/) | none (debug bus) |
| `bnd_inline_runs.sh`, `bnd_ethniu.py` | the inline-write check (§5); NIU counters and ERISC PCs of every Ethernet core after a hang | read-only |

## 1. Bound side on the lead, cheap pass (a09u32 job 349, v4bdd, base / tx32 / rx32; rows per side)

| suite | TX (E0) | RX (E1) | both | neither | change from v4trm (V4_PROFILE §2) |
| --- | ---: | ---: | ---: | ---: | --- |
| 2D 4352 B | 5 | 4 | | 1 | Mesh1HopEWBidir write TX -> neither (tx c32 38.0 -> 30.5, slack 13.5) |
| 1D | | 8 | 2 | 6 | none |
| mix | 20 | 9 | | 4 | EW_PureAtomic both -> RX (tx c32 39.4 -> 35.1); NS_MixChunk TX -> neither (41.8 -> 34.5) |
| ring | | 12 | 1 | 1 | Ring3HopEW RX -> both (tx c32 34.0 -> 38.3; rate 173.5 -> 163.7) |
| Y-torus | 4 | 5 | 2 | 1 | TorusY1HopEWBidir write TX -> neither (as 2D's bidir row) |
| VC2 on (pcs3 rows) | ew4352 | ewscatter, torusns_scatter | torusns | | new rows: VC2-on writes are E0-bound (ew4352 tx c32 +49.1, slack 0; torusns both 0) |
| VC2 traffic (pcs3 vc2/vc02 rows) | VC0 sender in vc02ew4352 / vc02bew4352 | VC2 sender in vc02bew4352 | | VC2 alone | new rows; VC2 alone (858.9) is window-bound (2 worker slots, §35 of V4_RECEIVER_SHAPE) |

All changes are threshold effects of a few cycles; no row moved between E0 and E1. The lead's E0 has more slack than
v4trm's on the 1-hop rows (2D write tx c32 45.2 -> 38.7, 1D write 30.3 -> 19.7) at the same rates (127.6 -> 127.0,
124.5 both): on the 2D 1-hop writes the TX side is "TX" by the 8-cycle rule with 5-7 cycles of nominal slack (§1c:
the 16 -> 32 slope settles it). Per-row tables: `python3 bnd_report.py` (results/bnd_*).

### 1b. One ERISC, mux, VC1 (a09u32 job 357, same arms; rows per side)

| suite | TX | RX | both | neither | change from v4trm |
| --- | ---: | ---: | ---: | ---: | --- |
| one ERISC 2D | | 8 | 2 | | none |
| one ERISC 1D | | 7 | 9 | | none |
| one ERISC ring | 2 | 6 | 8 | | none |
| mux | | | 2 | 20 | none (20 rows bound by the mux) |
| VC1 split (vc1_alloc) | 8 | 2 | 1 | 2 | VC1Mesh2HopMerge 4352 both -> RX (tx c32 38.8 -> 32.5) |
| VC1 inter-mesh (vc1_inter, new) | | 2 | | 6 | new: 1_0 (one mesh hop, 208-209) RX; 1_1 / 1_2 / 1_3 (346-414) neither: c32 tx 6-16, rx -5 to +21 |

On one ERISC TX / RX name the half of the pass (send vs receive work) on the critical router's one core.
- VC1 1-hop and burst rows: TX (c32 tx +47 to +49, slack 0; rx +27 to +33): the sender ERISC binds, as on v4trm.
- VC1 inter-mesh 1_1..1_3 rows are bound by neither ERISC's per-packet cost at 32 cycles: a spin on either side moves
  them by about a third to a half of its cost, the signature of a window/round-trip-bound row (each side's spin adds
  to the loop once). Their VC1 windows are small (VC1 forwarding 5, VC1 receiver 4 slots; fabric-slot-defaults).
  A/B (a09u32 job 360, `bnd_vc1win_runs.sh`, v4rxv2i3, vc1_inter.yaml bracketed d / w / d, the two d runs within 0.3;
  slots logged): deeper VC1 windows from the worker's spare (TT_FABRIC_EXP_SLOTS_VC1=13,8,16,7,7 against the study
  row 19,9,16,5,4, both 81 slots):

  | row | 16 B: d -> w | 4352 B: d -> w |
  | --- | --- | --- |
  | VC1Inter 0_0 -> 1_0 | 208.3 -> 211.0 (+2.7) | 209.1 -> 212.7 (+3.6) |
  | VC1Inter 0_0 -> 1_1 | 346.3 -> 294.9 (-51.5) | 397.4 -> 304.5 (-92.7) |
  | VC1Inter 0_0 -> 1_2 | 347.0 -> 295.8 (-51.3) | 395.8 -> 304.4 (-91.4) |
  | VC1Inter 0_0 -> 1_3 | 375.9 -> 298.4 (-77.6) | 413.8 -> 316.2 (-97.7) |

  So the inter-mesh rows with VC1 hops are bound by VC1's windows (5 forwarding / 4 receiver slots), not by either
  ERISC's per-packet cost (fact, A/B). The study's VC1 row (alloc_model.py) is due for a re-derivation on the v4 lead
  (user's call; fabric-slot-defaults: allocation stays offline). VC1 split rows were not run at w (the worker 19 -> 13
  may cost them; 1_0 +3). Hypothesis for §38's slots_due result (+0.4 to +1.7 on exactly these rows, flat elsewhere):
  on a window-bound row a change in the credit loop's timing shows, while fewer E1 instructions cannot; not isolated.

### 1c. The 16-cycle arms (job 357, suites 4352, 1d, ring, torusy; c16 = tx16 - base etc.)

A side with no slack shows a 16-cycle spin at ~16 + 12 = 28 and the 16 -> 32 step at ~16 (slope ~1, or above 1 where
the added time also lengthens a credit round trip, as V4_PROFILE §6 found); a side with slack S shows ~max(0, 28 - S).

| row (write 4352) | base | c16 / c32 TX | slope TX | c16 / c32 RX | slope RX | bound (A/B) |
| --- | ---: | --- | ---: | --- | ---: | --- |
| 2D E/W 1-hop | 127.0 | +17.0 / +38.7 | 1.36 | +4.0 / +17.4 | 0.84 | E0 |
| 2D N/S 1-hop | 127.3 | +15.4 / +37.3 | 1.37 | +6.2 / +22.1 | 0.99 | E0 |
| 2D 3-hop | 146.3 | +35.1 / +58.3 | 1.45 | +5.0 / +17.4 | 0.77 | E0 (compounds along the path) |
| 2D multicast | 141.7 | +28.2 / +50.0 | 1.36 | +4.1 / +17.8 | 0.86 | E0 |
| 1D 1-hop | 124.5 | +0.5 / +19.7 | 1.20 | 0.0 / +14.1 | 0.88 | neither (the wire) |
| 1D 3-hop | 124.5 | +27.1 / +51.5 | 1.53 | +20.3 / +37.6 | 1.08 | both, at the wire's rate |
| ring 1-hop | 140.9 | +9.1 / +29.6 | 1.28 | +27.1 / +45.6 | 1.16 | E1 |
| Y-torus N/S 1-hop | 149.2 | +18.3 / +37.9 | 1.22 | +27.7 / +45.5 | 1.11 | E1, E0 within ~6-10 |
| Y-torus E/W 1-hop | 127.0 | +17.0 / +39.0 | 1.38 | +4.0 / +17.4 | 0.84 | E0 (as 2D E/W) |

Scatter rows (all suites): RX, c16 +21 to +32 (no slack on E1); 3-hop scatter TX (c16 +27). On 2D 1-hop writes the
TX spin costs less than "no slack" at 16 (17 vs 28) yet more than its own cycles from 16 to 32 (slope 1.36): E0 binds,
with the hook's own overhead on the lead smaller than v4trm's ~12 or a few cycles of E0 slack that the first cycles of
spin absorb; the side reading is unaffected.

## 2. Why line -> ring costs +16.4 and 2D N/S -> torus N/S +21.9 (user's question; jobs 349, 357)

Settled on the lead (bound sides by A/B above; per-ERISC times as rate - slack, slack from c16 and c32 with a spin
costing N + 12 where it binds: an estimate, ±4 where the two arms disagree):

| row | rate | bound (A/B) | E0 slack (c16 / c32) | E1 slack (c16 / c32) | E0 time | E1 time |
| --- | ---: | --- | --- | --- | ---: | ---: |
| line 1-hop | 124.5 | neither: the wire | >= 27 / 24.3 | >= 28 / 29.9 | <= ~100 | ~95 |
| 2D N/S 1-hop | 127.3 | E0 | ~0 | 21.8 / 21.9 | 127.3 | ~105.5 |
| ring 1-hop | 140.9 | E1 | 18.9 / 14.4 | 0 | ~122-127 | 140.9 |
| Y-torus N/S 1-hop | 149.2 | E1 (E0 near) | 9.7 / 6.1 | 0 | ~140-143 | 149.2 |

- First-level acks cost E1 about the same on both: ring - line ~ +46, torus - 2D ~ +44 (v4trm: ~52-55; the lead's
  receiver redesign made acks cheaper).
- The visible gaps differ because the rows without acks sit on different floors: line on the wire (124.5, E1 ~95 and
  E0 ~100 both well below), 2D N/S on E0 (127.3). 2D's E1 is ~10 above line's E1 (105.5 vs 95) without acks, hidden
  under both floors; with acks both rows become E1-bound and the 2D receiver's extra ~8-10 shows: torus E1 149.2 vs
  ring E1 140.9 = +8.3. So +21.9 = 16.4 (the ack cost seen against the wire) + 8.3 (2D E1 extra) - 2.8 (2D's E0 floor
  above the wire). The hypothesis as stated holds, with smaller numbers on the lead: torus E0 grows ~13-16 over 2D's
  (not ~27 as on v4trm) and stays 6-10 below the rate ("both" by the 8-cycle rule at 32, E1 alone at 16).
- Caveat on the E1 times: a slack read from an RX spin includes the credit loop's latency, and the two rows have
  different receiver windows (2D 16 slots, 1D 32): part of 2D's ~10 may be the shallower window, not E1 work. The
  E1-bound pair (ring vs torus, both no slack) gives the clean E1 difference: +8.3 (§3).
- Where 2D E1 spends its extra cycles over line E1, and torus E0 over 2D E0: §3.

## 3. Why, per bound class (a09u32 job 358; PC samples)

PC samples of v4rxv2i3 (`bnd_pcs_runs.sh`), -g twins, cycles/packet by the innermost router frame and source line
(session scratchpad tools frames2.py, framediff.py, e0split.py).

Samples split cycles within one ERISC; they do not name the bound (§1-§2 do). A stalled load shows at its first use.
Everything below about why a line costs what it does is a hypothesis.

**E0-bound writes (2D / torus E/W 1-hop 126.9; 16 B-2 KiB 115.8-118.5; forwarded 125.7; 3-hop 146.2; VC2 on 134.6).**
Busiest sender, by phase (e0split.py; "group read" includes any idle passes):

| phase (E0) | ew16 | ew1024 | ew4352 | ns4352 | fwd | hop3 | VC2 on ew4352 | ring1 [E1-bound] | torusns [E1] | lin1 [wire] |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| rate | 115.8 | 116.4 | 126.9 | 127.1 | 125.7 | 146.2 | 134.6 | 140.9 | 149.2 | 124.4 |
| pass group read (free-slots registers, semaphores) + first use | 36.5 | 36.4 | 42.6 | 42.5 | 21.6 | 34.3 | 43.4 | 28.4 | 42.9 | 31.6 |
| connection checks / forwarding-group test | 4.7 | 4.6 | 5.6 | 5.1 | 8.9 | 14.9 | 5.8 | 3.4 | 3.6 | 6.8 |
| send: pre (header size L1 load, TXQ status, stamp) | 12.8 | 14.5 | 14.1 | 14.7 | 17.7 | 21.9 | 16.2 | 15.4 | 14.6 | 14.8 |
| send: data command (4 TXQ writes) | 5.3 | 4.8 | 4.7 | 5.4 | 5.1 | 5.1 | 7.5 | 4.1 | 5.0 | 5.4 |
| send: ring / slot advance | 14.5 | 13.6 | 13.9 | 13.8 | 14.6 | 13.8 | 15.0 | 15.3 | 14.3 | 13.8 |
| send: post-send group (TXQ status, completion / ack counters) | 5.1 | 4.7 | 5.6 | 5.1 | 4.9 | 5.7 | 5.4 | 5.5 | 5.8 | 5.1 |
| send: busy test (+ spin, never taken: §6) | 4.4 | 4.8 | 4.5 | 4.0 | 3.9 | 3.1 | 4.1 | 3.1 | 2.7 | 4.2 |
| send: free counts, packets-received update, tail | 12.8 | 13.2 | 12.1 | 12.7 | 14.1 | 14.7 | 13.2 | 13.7 | 13.2 | 11.3 |
| visit: send gate, completion / ack reads, bookkeeping | 7.3 | 8.1 | 11.1 | 11.7 | 17.9 | 19.2 | 11.0 | 25.2 | 20.4 | 18.0 |
| credits to producers (worker credit, upstream credits) | 7.4 | 6.9 | 6.7 | 6.8 | 13.6 | 9.2 | 5.8 | 22.3 | 21.9 | 6.4 |
| loop / round | 4.3 | 4.4 | 5.2 | 4.4 | 2.9 | 4.4 | 6.4 | 4.5 | 4.8 | 6.0 |

- The send itself is ~55-60 per packet at every size: the four lines worth naming are the payload-size header load
  (its first use, `zext.h`, 6.6 on ew4352), the slot advance (`lw 8(sp)` / `lw 12(sp)`: the far ring's and the
  channel's last-slot constants reloaded from the stack, stalls 3.6 and 1.9, plus the advance arithmetic), the TXQ's
  seven register writes (data 4, packets-received 3; ~1.9 each), and the post-send status read's first use (3.2).
- 16 B -> 4352 B costs E0 +11.1: group read and its first use +4.7, the visit's room/completion lines +2.7, the rest
  spread (framediff); hypothesis: at 4352 B E0 also runs passes with nothing to send while the credit loop turns
  (consistent with the TX spin's slope 1.36 > 1).
- VC2 compiled in and idle, against VC2 off (ew4352, +7.7): VC2's worker visit, connection check and register in the
  group +7.3; the send's own lines +6.4 (TXQ writes +3.5, slot advance +2.9, header read +2.0), the VC0 visit -4.6.
  Hypothesis for the send's growth: register pressure with VC2's state live (spilled values reloaded around the send).

**E1-bound, first-level acks (ring 1-hop 140.9, torus N/S 149.2).** Busiest receiver, by frame (framediff):
line 1-hop (E1 not bound) -> ring 1-hop: acks +45 (acknowledge_arrivals 14.6, add_ack 11.8, ring_add +7.2, the credit
frame +5.8, the TX status read in the group 3.1, credit_slots +2.5) against the completion-group machinery -17.8
(writes_leave, due) +9.0 (release_completions one slot at a time). Ring E1 -> torus E1 (both E1-bound, +8.3):
Routing2D::action +13.5 (router_routing.hpp:88: the two volatile route bytes' first use, 10.5), HeaderWord::of -7.6
(1D's route rides in the header word), handle +2.5 and Route::forwards/delivers +2.0 (router_receiver.hpp:317), add_ack
+1.9, the rest -3.7. So the 2D receiver's extra over 1D is the 2D route read (net ~+8): hypothesis from samples (the
stall moves with the first use).

**E0 on the torus over 2D N/S (+22.1 on E0's samples, E0 not bound there):** the worker credit +15.2 (credit_worker:
its command-buffer ready poll noc_nonblocking_api.h:1731 +4.8, register writes :174 +3.6 and :1743 +3.1) and the visit's
ack lines +7.8 (router_sender.hpp:478-489). Under first-level acks the worker is credited per ack change, about once per
packet, against once per completion group of 4 on the mesh (6.8/packet on 2D N/S vs 21.9 on the torus).

**E1-bound scatter (2D E/W scatter 139.7):** local_write 42.3 (four writes; the command-buffer ready poll inside
ncrisc_noc_set_transaction_id, noc_nonblocking_api.h:1470, 19.1; register writes :174 15.4) and deliver's chunk loop
30.2; handle 11.2 (router_receiver.hpp:317 10.4).

**Neither (1D 1-hop 124.4, the wire):** both ERISCs idle part of each packet (E0 group read 31.6 includes the idle
passes; E1 step/receive/due ~23).

## 4. NoC command-buffer audit (static)

From v4rxv2i3's named args of every built config and the disassembly (audit subagent; its scripts in the session
scratchpad cmdbuf/), checked here where it matters (the buffer-2 and inline-write items).

Facts common to every two-ERISC config: E0 = active ERISC, NOC_INDEX 0, issues only on NoC 0; E1 = subordinate ERISC,
NOC_INDEX 1, only on NoC 1 (RX_CH_*_FWD_NOC_ID = LOCAL_WRITE_NOC_ID = 1; `worker_handshake_noc = noc_index`). Buffers
0 WR, 1 RD, 2 WR_REG, 3 AT. The four buffers exist per NoC per tile and are shared by a tile's two ERISCs: the API
separates them only by NoC in dedicated mode (dataflow_cmd_bufs.h:20-31 gives AERISC0 / AERISC1 the BRISC / NCRISC
sets, which coincide in dedicated mode), so E0's buffer n and E1's buffer n are different hardware buffers on two
ERISCs. One-ERISC routers run on ERISC1 (NOC_INDEX 1), everything on NoC 1, and preset nothing
(`command_buffer_presets = NUM_ACTIVE_ERISCS > 1`).

E0, two ERISCs (2D, 1D, ring, torus, VC1, VC2, express):

| role | buffer | state | per use |
| --- | --- | --- | --- |
| VC0 worker credit | 0 | preset (CTRL posted VC 1, RET_MID, RET_COORD, LEN 4) at connect, after a VC2 connection event, at start_running and after the firmware ran | scratch store + 1 ready poll + 3 writes |
| VC2 worker credit | 2 | stateless, non-posted (noc_semaphore_set_remote) | 1 poll + 6 |
| upstream credit (forwarding channels) | 3 | stateless inline REG write, posted | 1 poll + 7 |
| worker connect (library emulated L1 inline write) / close | 0 / 3 | whole commands | rare |

E1, two ERISCs:

| role | buffer | state | per use |
| --- | --- | --- | --- |
| local write (unicast, scatter chunk, fused write) | 0 | preset once (CTRL non-posted, EDM_NOC_VC: 0x4092) | 1 poll (in the transaction-ID write) + 6 |
| local atomic | 3 | stateless, non-posted (**WRONG**: relies on the firmware's preset return address, below) | 1 poll + 7 |
| inline-write packet to L1 | 0 | **whole command, posted (0x4082), clobbers the preset (§5)** | 2 flush spins + 1 poll + 6 |
| forward data, one live downstream (VC0, E/W 2D, 1D) | 2 | preset (CTRL posted EDM_NOC_VC, RET_MID, RET_COORD) | buffers 2 and 1 ready read as a pair + 5 |
| forward credit, one live downstream | 1 | preset (all fields) | 1 write (2 under deadlock avoidance) |
| forward data, stateless (N/S, edge, intermesh routers; every VC1 forward) | 2 | whole, posted, NoC VC 1 | 1 poll + 7 |
| forward credit, stateless | 3 | whole inline write, non-posted, NoC VC 1 | 1 poll + 7 |
| completion proof | - | NIU WRITE_REQS_OUTGOING_ID(trid) read | 1 read per check |

**WRONG (2026-11-01, job 366, v4rxo0): buffer 3 is not stateless.** The firmware presets its return address
(`noc_init_one`, BH noc_nonblocking_api.h:794-802: RET_ADDR_LO/MID/COORDINATE = the core's atomic return scratch),
and `noc_fast_atomic_increment` never rewrites it in dedicated mode (every atomic is INCR_GET and returns the old
value there). A write issued on buffer 3 (it programs RET_ADDR as its destination) redirects every later atomic's
return. Moving the inline-write emulation onto buffer 3 corrupted data (FusedAtomicInc mismatch). So every E1
buffer carries preset state; "buffer 3 is shareable by stateless users" (interim message, §4 item 1 text) is wrong too.

Stateful E1 configs per case (stateful / stateless): bh_2d 20/8, bh_1d 10/0, bh_ring 10/0, bh_torus 8/4, bh_vc1 0/22
(every router of a 2x2 mesh is an edge or intermesh router), glx_2d 28/20, glx_express 8/20.

Known items:
1. VC0 and VC1 forwards share forward-data buffer 2: **confirmed as code, latent.** RX_CH_0 and RX_CH_1
   FWD_DATA_CMD_BUF_ID are 2 in all 44 bh_vc1 images (host default, erisc_datamover_builder.cpp:519);
   `forwards_are_stateful` does not exclude VC1 fabrics (router_downstream.hpp:79-88; main does, !FABRIC_2D_VC1_ACTIVE).
   No built config has a stateful VC0 router with VC1 (needs an inner E/W router of a multi-mesh fabric with meshes
   larger than 2x2). A VC1 stateless forward would rewrite buffer 2's CTRL (NoC VC EDM_NOC_VC -> 1) and RET_COORD; the
   coordinate would be the same tile (VC0 and VC1 connect the same router pair, compute_mesh_router_builder.cpp:723-783;
   the same E/W live-downstream rule), so the destination stays right and only VC0's NoC VC changes (hypothesis, by code
   read; not the "wrong destination" VC2_LINE_IDEA.md item 1 feared). Fix options (a) / (b) / (c) and their costs: sent
   to main (interim message); the choice is the user's; nothing built.
2. An inline-write packet clobbers buffer 0's preset: **confirmed, and it hangs** (§5).
3. VC0's worker-credit preset fix with VC2: **confirmed complete** for every built config: during the loop E0 issues on
   NoC 0 buffer 0 only VC0 credits and connect writes; restores after a VC0 connect, a VC2 connection event,
   start_running and the firmware (router_sender.hpp:418, 690, 742, 758).
4. VC2 line mode shares VC0's presets: **confirmed as a design constraint** (tt-metal-vc2line, vc2l_p2a): the line VC
   takes VC0's buffers and is stateful on VC0's preset only where its one live downstream equals VC0's, and then every
   forward of both VCs sets the credit address (TARG_LO). Consistent; with VC1 also present, VC1's stateless forward
   would clobber the shared preset as in item 1.
5. Torus worker credit's ~6.6 cycles of ready wait: **partly.** The VC2-related part is gone (VC0 keeps its preset with
   VC2 compiled in). The ready poll per credit remains (torus E0: noc_nonblocking_api.h:1731 6.7 cycles/packet, §3), and
   under first-level acks the credit fires about once per packet. Nothing else issues on that buffer during the loop:
   hypothesis, it is the poll's register-read latency, not a busy buffer.

Still suboptimal or fragile (beyond items 1-2):
- F3 (fragile): E1 never re-presets; that rests on the base firmware not using NoC 1. The ERISC0 firmware's
  `dynamic_noc_init()` runs in every full context switch (erisc.h:38's `#if NOC_MODE == DM_DYNAMIC_NOC` compares two
  enumerators, so it is always true) and writes both NoCs' MID / coordinate fields on all four buffers; today this
  happens only during E0's init, after E1's presets, with values that coincide with E1's presets. A change to either
  breaks E1's presets silently.
- F4 (cost, one ERISC): forward data and local writes share buffer 0 (builder :529) while buffer 1 is unused; two large
  back-to-back writes then wait for each other (V4_STATE §17: 4 KiB ready call 31.8 with one buffer vs 14.5 with two).
  Cheap device check: one-ERISC multicast row with forward data on buffer 1 (one constexpr / host arg).
- F5 (cost): E0's upstream credits are stateless (1 poll + 7 writes) where a router has one forwarding producer (1D,
  ring: fixed destination) and buffer 1 is free on E0; a preset would leave 1-2 writes. Under first-level acks they
  fire per ack.
- F6 (inherited from main): stateless forwards use NoC VC 1 while stateful ones use EDM_NOC_VC; stateless forward
  credits are non-posted on the atomics buffer, so a strict flush also waits for their acks.
- F7 (minor): untagged commands inherit a buffer's last transaction ID (completion proofs conservatively wait for
  them); `worker_credit_preset` lacks the DPRINT/WATCHER guard the other presets have (router_sender.hpp:347).

v4txr2's E0 map (WR: VC0 worker credit preset + library connect; RD unused; WR_REG: VC2 worker credit; AT: upstream
credits, teardown, the interfaces' sync buffer) is NoC 0 on E0, so it is independent of the buffer-2 fix and of VC2
line mode's forwards (NoC 1 on E1). On one ERISC its credits are whole commands on NoC 1 buffers 0 and 2 beside forward
data on 0 (forced) and nothing is preset, so nothing is at risk there either. (Its teardown on AT was not re-read line
by line.)

## 5. Inline-write check (a09u32 job 351; `bnd_inline_runs.sh`)

Code read (§4 item F1): an inline-write packet's local delivery (`execute_chip_unicast_to_local_chip_impl`,
NOC_UNICAST_INLINE_WRITE arm, reached through `deliver_other`) is `noc_inline_dw_write<DEFAULT, posted=true>`, which
on an L1 destination is emulated on write_cmd_buf (0) of E1's local NoC (1): the buffer v4 presets for local writes.
It writes NOC_CTRL 0x4082 (posted) over the preset 0x4092 (non-posted, RESP_MARKED) (disassembly, 2D E/W E1 of
`bddlead_O3` bh_2d__11660148679966767161: d260-d268 vs d3a0-d3a8); nothing restores the preset (the restore in
fabric_edm_packet_transmission.hpp:380-382 is DRISC-only). V4_STATE.md §17's "writes the value the preset holds" is
marked WRONG there.

Device (fabric_unit_tests, one fixture = one fabric instance per run: the unicast inline-write test, then the fused
write + flushed atomic-inc test; main's router first as the control, then v4rxv2i3):

| router | 1D: InlineWrite, then FusedAtomicInc | 2D: InlineWrite, then FusedAtomicInc |
| --- | --- | --- |
| main (kernel v1, flags 0) | OK, OK | OK, OK |
| v4rxv2i3 | OK, **hang** (300 s) | not run (the job stops at a hang) |

The hang (results/bndi_v4rxv2i3_1d_hangcap/, bndi_v4rxv2i3_1d_ethniu.json): chip 2 eth 4, E1, 10 packets pending on
rx0; E1's PC fixed at 0xd8c0 (16 of 16 reads) = flush_write_to_noc_pipeline (fabric_edm_packet_transmission.hpp:116,
`while (!ncrisc_noc_nonposted_writes_flushed(noc))`) <- local_atomic_inc <- deliver<0>'s fused arm; the loop compares
NOC1 NIU_MST_WR_ACK_RECEIVED with the software noc_nonposted_writes_acked[1]. Hardware: WR_ACK_RECEIVED =
NONPOSTED_WR_REQ_SENT = NONPOSTED_WR_REQ_STARTED = 41,831,566 and every REQS_OUTSTANDING_ID = 0, so nothing
non-posted is in flight while E1 waits for more acks: writes counted non-posted went out posted. (The software count
is in ERISC local memory, 0xFFB00014, outside the capture.) The device was held degraded and reset through the broker.

So: a real hang on two-ERISC routers after any L1 inline-write packet followed by a counted local write and then a
flushed atomic (meshes and lines without deadlock avoidance) or another L1 inline write's flush. Not covered by the
study's validation (no test_tt_fabric yaml sends inline writes). Fix (not built; user's call): re-run the local-write
preset after the inline-write arm (rare path).

## 6. TX-queue probe: can E0 batch its ack / completion checks? (a09u32 job 359)

`bnd_txq_runs.sh t2`, `bnd_txq_report.py t2`.

v4bdq on 12 rows (E/W 1-hop writes of 16, 256, 1024, 2048, 4352 B; forwarded; N/S; 3-hop; ring 1-hop; torus N/S;
line 1-hop; E/W scatter). Busiest sender ERISC per row; sampled = every 61st send (the probe's wait perturbs it).

| row | probe rate (lead) | sends | post-send status read saw the queue busy | sampled: data command until free | one poll on the idle queue | register command until free | waits 0-15 / 16-31 / 32-47 cycles |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| E/W 16 B | 128.8 (115.8) | 93.1 M | 0.0 % | 15.1 | 15.1 | 15.1 | 87 / 13 / 0 % |
| E/W 256 B | 128.8 (115.8) | 93.1 M | 0.0 % | 15.1 | 15.1 | 15.1 | 87 / 13 / 0 |
| E/W 1024 B | 128.7 (116.4) | 93.1 M | 0.0 % | 15.1 | 15.1 | 15.2 | 87 / 13 / 0 |
| E/W 2048 B | 128.6 (118.5) | 93.1 M | 0.0 % | 15.1 | 15.1 | 15.2 | 87 / 13 / 0 |
| E/W 4352 B | 131.5 (126.9) | 85.0 M | 0.0 % | 15.1 | 15.1 | 15.1 | 87 / 13 / 0 |
| forwarded 4352 B | 125.8 (125.7) | 171.4 M | 0.0 % | 15.1 | 15.1 | 15.1 | 91 / 9 / 0 |
| N/S 4352 B | 132.0 (127.1) | 85.0 M | 0.0 % | 15.1 | 15.1 | 15.1 | 87 / 13 / 0 |
| 3-hop | 150.7 (146.2) | 75.0 M | 0.0 % | 15.1 | 15.1 | 15.1 | 87-89 / 11-13 / 0 |
| ring 1-hop | 140.8 (140.9) | 50.0 M | 0.0 % | 16.7 | 15.1 | 15.2 | 69 / 23 / 8 |
| torus N/S | 149.6 (149.2) | 50.0 M | 0.0 % | 16.7 | 15.1 | 15.2 | 67 / 25 / 8 |
| line 1-hop | 124.4 (124.4) | 83.7 M | 0.0 % | 15.1 | 15.1 | 15.1 | 93 / 7 / 0 |
| E/W scatter | 141.7 (139.7) | 73.5 M | 0.0 % | 15.1 | 15.1 | 15.1 | 87 / 13 / 0 |

Facts:
- The TX queue's busy time after a data command is below one status poll at every packet size (16 B to 4352 B): the
  first poll (~15 cycles, a CMD read then STATUS) already reads it free; data - idle poll = 0 (1.6 on ring / torus,
  where 8 % of samples need a second poll). The packets-received register command is the same. The wire time does not
  show in CMD_ONGOING (the queue accepts the command into its accept-ahead window).
- In the router's own flow the post-send status read never saw the queue busy: 0 of 1.28 billion sends (all E0s); PC samples
  put 0 cycles in the spin (router_sender.hpp:314). No pre-send yields either.
- So the second case holds: E0's post-send reads are not hidden under any TX-queue wait; on E0-bound rows they sit on
  the critical path. Per packet (§3): post-send group ~5 + the status read's first use ~4.5 + free counts, register
  update and tail ~12-13.

Decision (with the numbers): batching the ack / completion checks cannot pay beyond the 1-2 cycle band. The status
read in the post-send group is mandatory (never issue a TX-queue command while it is busy, ROUTER_MAP obligation 3;
removing a check hung the router, RESULTS §11); the completion (and ack) counter loads ride in its group and share its
latency. What batching would remove per send on E0-bound rows: one L1 load (two with acks), one subtract, the visit's
completions test: ~2-4 cycles/packet of work by instruction count (estimate; the status read's stall stays). Against
that, credits would reach workers and upstream routers up to N-1 sends later, on windows of 2 (VC2), 4-5 (VC1) and
2-slot bubble rings. The batching probe was therefore not built (budget better spent on §7's levers); it can be run on
request.

## 7. send() and the sender pass: proposals (for discussion; nothing built)

Where E0's time goes on E0-bound rows (§3, ew16 115.8): per-pass work ~53 (group read and its first use ~36, checks
~5, visit ~7, loop ~4), the send ~55 (pre ~13, data command ~5, slot advance ~14, post-send ~10, free counts / register
update / tail ~13), credits ~7. Each proposal is a hypothesis with its cheapest isolating A/B.

P1. Several packets per worker (or forwarding) visit. When the channel holds k > 1 packets and the far ring has room for
them (bubble flow control respected per packet), send up to K in the visit (K bounded, e.g. min(available, 4), so a
busy channel cannot starve the others for more than K sends). Each extra packet costs the send (~45-55: TX status poll
~15, header size, data command, slot advance) instead of a whole pass (~115). The per-pass work (group read, connection
checks: still every pass, rule kept) is amortized over k. Expected: E0-bound small and medium writes (115-127) move
toward E1's time (~95-105, §2), where E1 binds; E1-bound and wire-bound rows unchanged. Burst rows: event-driven (sends
what is there now, never waits for more). A/B: a probe variant with K = 2 on the worker channel only, rows ew16,
bew1024, ew4352, burst rows, VC1 1-hop. Touches the sender state-model agent's visit structure and register map
(v4txr*): the visit then carries `available` (from the free-slots register) as a loop-carried value.

P2. With P1, one packets-received update (+k) and one free-slots update (+k) per burst instead of per packet: saves 3
TX-queue register writes, one busy test and one stream-register write per extra packet (~10-12). Cost: the far
receiver sees packet 1 of a burst after packet k's data (~45 cycles per extra packet of added latency). Variant: keep
the per-packet update and amortize only the pass (P1 alone). A/B: P1 vs P1+P2 on the same rows plus latency rows.

P3. The slot advance (~14 per send): two compare-and-wrap advances whose 32-bit last-slot constants are reloaded from
the stack every send on 2D E0 (`lw 8(sp)`, `lw 12(sp)`; their first uses stall 3.6 and 1.9). A form with fewer 32-bit
constants live across the send (ring index against num_slots as a small immediate, address = base + index * stride,
or the far ring's next slot taken from a per-slot table) would remove two loads; whether GCC then keeps or spills the
rest is not something to design on (rule: no compiler-fragile tricks). Belongs to the state-model agent's register map;
A/B on 2D 1-hop and VC1 1-hop (the most register-pressed E0).

P4. Worker credit under first-level acks (ring / torus E0: ~22 per packet, a credit per ack change, about one per
packet; mesh 6.8 per completion group of 4). These rows are E1-bound today (torus E0 within 6-10 of binding), so it pays
only if E1 gets faster; v4txr2's preset-on-own-buffer credit already cuts its command (6 -> 3 writes).

Not proposed: batching completion / ack checks (§6); dropping the post-send status read (obligation 3).
Open questions for the user: K's bound for P1 (a fixed small K, or a quarter of the channel's slots as the
credit-hold rule of feedback-robust-over-peak); whether P2's added first-packet latency is acceptable.

## 8. Jobs, labels and files

| job | what | labels | device time |
| --- | --- | --- | --- |
| 349 | bound sweep, two ERISCs + VC2 rows + VC2 traffic rows, base / tx32 / rx32 | bnd_{4352,1d,mix,ring,torusy}_*, bnd_vc2_*, bnd_v2t_* | 12 min |
| 350 | inline check, failed to start (gtest filter form) | - | 0 |
| 351 | inline-write check: main 1D/2D pass, v4rxv2i3 1D hangs (device reset via broker) | bndi_*, bndi_v4rxv2i3_1d_hangcap/, _ethniu.json | 7 min |
| 357 | tx16 / rx16 on 4352, 1d, ring, torusy; one ERISC, mux; VC1 split and inter | bnd_*_tx16/rx16, bnd_{4352_1e,1d_1e,rse,mux}_*, bndvc1[i]_* | 11 min |
| 358 | probe t1 (empty: probe did not set pkts) + PC sampling of v4rxv2i3 (2D 10 rows, 1D, ring, torus, 2D VC2) | bndq_t1_* (void), bndp_{b2d,b1d,bring,btor,b2dv2}_* | 10 min |
| 359 | TX-queue probe t2 | bndq_t2_* | 4 min |
| 360 | VC1 inter-mesh window A/B | bndw_* | 1 min |

New files: variants/v4bdd (hook variant), variants/v4bdq (TX-queue probe, attribution only), consolidation/
bnd_delay_runs.sh, bnd_report.py, bnd_vs_trm.py, bnd_loadcmp.py, bnd_txq_runs.sh, bnd_txq_report.py, bnd_pcs_runs.sh,
bnd_inline_runs.sh, bnd_ethniu.py, bnd_vc1win_runs.sh, bnd/bew{256,1024,2048}.yaml. Coverage builds in cov_m: v4bdd_O3,
v4bdd_Os, bddlead_O3, bddlead_Os, bdqon_O3, bdqoff_O3. Caches ~/.tt-metal-cache-{bnd-*,bndi-*,bndp-*,bndq-*,bndw}.
Existing note edited: V4_STATE.md §17 (WRONG marker, §5 above).
