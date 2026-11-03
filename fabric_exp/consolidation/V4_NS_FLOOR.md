# The Galaxy N/S floor: why N/S-involving rows run ~172-187 cycles/packet (2026-11-02)

Status: attribution study (slot study agent). Lead `variants/v4rxv2i3` (Galaxy trees: `variants/v4bdd` = the lead plus
delay hooks, byte-identical at 0). Nothing proposed here is built for adoption; the two variants below are diagnostic
single changes. Rules: bound side only from one-side A/Bs; causes hypotheses until a single-change variant reproduces
them; PC samples split cycles within one ERISC only.

## 0. What existed

- RESULTS §98 "Galaxy (200G links): main against v3q" (Slurm job 155240, v3q): "7-hop N/S (column 0 down the Galaxy)
  runs at ~181 against 3-hop E/W's ~138. Not yet explained (candidates: a longer credit round trip on some N/S links, or
  one slower link in the column); main hides it at 222."
- RESULTS §98 E1 per-packet table (v3 era): "The 1-hop decode is 15 against 2D's 20 (E/W) and 27 (N/S)."
- V4_BOUNDS.md §4 (command-buffer audit): E1 forwards are stateful (preset command buffers, "buffers 2 and 1 ready read
  as a pair + 5" register ops) only on routers with one live downstream; stateless ("1 poll + 7" for the data, "1 poll +
  7" for the credit) on N/S, edge and intermesh routers; stateful / stateless E1 configs: bh_2d 20 / 8, glx_2d 28 / 20.
- SLOT_STUDY.md P3 (job 167464): the floor, ~178-187 at 1 KiB on every N/S-involving Galaxy 2D row, flat over 13-16
  forwarding and 17-26 worker slots; P150's turn row 145. Below 16 forwarding slots the 7-hop row is window-bound.
- Nothing on NoC distance between Ethernet cores. Measured here (mock arg dumps, `TT_FABRIC_EXP_DUMP_ARGS`, with
  Blackhole's Ethernet NoC x = 1,16,2,15,3,14,4,13,5,12,6,11,7,10 for channels 0-13): on the Galaxy the N- and S-facing
  routers of a chip are 1 NoC hop apart (channels 2/3 and 4/5: x 2/15 and 3/14), E/W 3 hops (x 1/16 and 4/13); on the
  P150 E/W 2-9 hops and the turn row's N -> E turn 10-11 hops. NoC distance does not favour the rows that run faster.

## 1. Bound side (Galaxy job 167648, one-side spins of 16 / 32 cycles per packet, v4bdd)

A spin of N on a side with no slack costs ~N + 12 (V4_BOUNDS §1's reading). 2D rows (sls_2d_glx_4k / _2k.yaml):

| row (1 KiB; 2 KiB alike) | 4352 B slots 17,16,16: base / c_tx32 / c_rx32 | side | 2 KiB slots 41,32,32: base / c_tx32 / c_rx32 | side |
| --- | --- | --- | --- | --- |
| 1-hop E/W | 122.9 / +43.8 / +21.1 | E0 | 109.6 / +54.1 / +29.4 | E0 |
| 3-hop E/W | 143.4 / +61.0 / +20.4 | E0 | 140.7 / +61.3 / +14.2 | E0 |
| 3-hop N/S | 182.3 / +14.8 / +43.4 | E1 | 172.5 / +21.8 / +44.1 | E1 |
| 7-hop N/S | 186.0 / +11.0 / +40.8 | E1 (+ window) | 173.1 / +21.1 / +44.0 | E1 |
| 10-hop corner | 186.7 / +17.7 / +40.3 | E1 (+ window) | 173.2 / +27.9 / +43.8 | E1 |
| turn (1 N/S + 3 E/W) | 178.1 / +25.9 / +42.8 | E1 | 170.5 / +30.7 / +44.3 | E1 |
| column merge (7 flows) | 166.2 / +21.4 / +42.2 | E1 | 158.4 / +25.9 / +44.4 | E1 |

- Every N/S-involving row is bound by E1 (the receiving ERISC of a forwarding router): a receive-side spin costs the
  full ~44, a send-side spin a fraction. E/W rows and 1-hop rows are bound by E0. Settled (one-side A/B).
- The floor does not move with payload (1 KiB = 2 KiB within 0.4 on every E1-bound row): a per-packet cost, not bytes.
- It moves with the forwarding depth only through the window: at 16 forwarding slots (4352 B slots) the 7-hop / corner
  rows sit 13 above the E1 floor (their N/S forwarding loop ~2,970 / 16 = 186); at 24 slots 2 above, at 32 on it (172-173;
  2 KiB slots, SLOT_STUDY.md 2 KiB section). So "the N/S floor" = E1's per-packet cost on N/S routers, ~172, plus a
  window remnant at 16 forwarding slots.

## 2. Why: two single-change A/Bs (diagnostic variants, not for adoption)

- `variants/v4bddsl` = v4bdd with `forwards_are_stateful()` returning false: every forward takes the stateless path N/S
  routers take. Images: every stateful E1 image differs, nothing else (bh_2d 20, bh_1d 10, glx_2d 28, glx_1d 12 of 196,
  `bnd_loadcmp.py`, cov_m `v4bddsl_O3`).
- `variants/v4bddml` = v4bdd with every 2D router's downstream entries all live (`live_downstreams()` without the E/W
  exclusion): E/W routers inside a mesh get the N/S routers' multi-downstream bookkeeping and, with it, stateless
  forwards (packets still never turn onto N/S there; the action bits select E/W only). bh_2d 16, glx_2d 24 images differ.

E/W 3-hop and turn rows (cycles/packet; c_rx32 / c_tx32 = cost of a 32-cycle spin):

| row | system | lead | stateless (sl) | stateless + all downstreams live (ml) | Galaxy N/S rows (lead) |
| --- | --- | --- | --- | --- | --- |
| 3-hop E/W, 1 KiB | P150 (jobs 412-413, 4 / 2 runs) | 144.7 E0 (rx +17.7) | 155.5 E1 (rx +43.1) | 174.5 E1 (rx +42.3) | |
| 3-hop E/W, 4352 B | P150 | 145.6 E0 | 157.6 E1 | 175.2 E1 | |
| turn, 1 KiB | P150 | 144.7 E0 | 161.3 E1 | 167.3 E1 | |
| 1D 3-hop, 1 KiB | P150 | 118.0 (wire / both) | 155.6 E1 (rx +44.0) | (1D: unchanged) | |
| 3-hop E/W, 1 KiB, 2 KiB slots | Galaxy (job 167673) | 140.2 E0 | 147.5 E1 (rx +44.3) | 164.6 E1 (rx +44.0) | 172.5-173.3 E1 |
| N/S rows, 1 KiB, 2 KiB slots | Galaxy | 172.5-173.3 | 172.5-173.3 (unchanged) | 172.5-173.3 (unchanged) | |

- Settled (single-change A/Bs, both systems): an E/W forwarder that takes the N/S routers' stateless forward becomes
  E1-bound; E1's per-packet time rises by ~37 (P150: lead E1 ~118 from its slack -> 155.5; Galaxy ~110 -> 147.5); taking
  their multi-downstream bookkeeping as well adds ~17-19 more (P150 155.5 -> 174.5; Galaxy 147.5 -> 164.6). The N/S rows
  do not move in either variant (control: their routers already work that way). Together the two account for the
  Galaxy floor to within ~8 cycles (164.6 for an N/S-shaped E/W forwarder against 172.5 for the N/S rows).
- The remaining ~8 on the Galaxy: hypothesis, the N/S routers' route decode (they read both route bytes as volatile
  loads, E/W routers one; RESULTS §98's 27 vs 20 on v3). Not isolated: making an E/W router read both bytes is not a
  safe single change (the row byte at its row may name an E/W direction), so no diagnostic was built.
- Why the P150 never shows it: a 2-row mesh has no N/S forwarding, and its turn routers have one live downstream (the
  P150 dump: chip 4's N router has NUM_DOWNSTREAM_SENDERS_VC0 1, so its forwards are stateful), while the Galaxy's turn
  router (chip 4, N) has 2 downstream senders and 3 entries (stateless). Mesh size changes the router type, not the
  link.

Where E1's extra cycles land (P150, PC samples of the forwarding E1 on the 3-hop E/W row, job 414, `bnd_pcs_runs.sh`,
`pcs_lines.py`; cycles/packet = share x rate; within one ERISC only, a stalled load shows at its first use: hypotheses):

| line (innermost) | lead (146.2) | stateless (157.4) | + multi-downstream (175.0) |
| --- | ---: | ---: | ---: |
| NOC_CMD_BUF_WRITE_REG (command-buffer register writes) | 10.9 | 20.5 | 21.0 |
| stateless credit: noc_fast_default_write_dw_inline (lines 1171-1185) | - | ~18 | ~15 |
| noc_async_write_one_packet_with_trid_impl / ncrisc_noc_fast_write | (write_with_state 5.2) | 7.4 + 4.6 | 8.4 + 3.9 |
| Downstreams::any_used_up (router_downstream.hpp:253: admission test over the live downstreams `to` needs) | - | - | 17.2 |
| HeaderWord::bits (the route word) | - | - | 7.9 |
| forward_each (dispatch over live downstreams) | - | - | 3.9 |

- Hypothesis (stateless path, ~+37): the whole NoC command per forward (address, length, coordinates, control: ~+10 in
  register writes) and a stateless credit write (an inline NoC write with its own command-buffer poll and ~7 register
  writes, ~18) where the preset forward does 5 writes and one credit-register write.
- Hypothesis (multi-downstream, ~+17-19): with one live downstream, admission (`any_used_up`) does not depend on the
  packet; with several it needs the packet's route word, so the header load's latency lands on the admission test
  (17.2 at line 253 right after the route word's 7.9), plus the per-downstream dispatch. **WRONG for the route word
  (V4_ROUTE_PRELOAD.md §8.2, jobs 457-459):** with the route preloaded a pass earlier, the route's samples fell ~4
  cycles/packet but the admission's did not (16.6 -> 18.4). They are spread over the admission loop's instructions,
  so they read as executed bookkeeping (hypothesis), not the route load's wait.

## 3. What a fix could be (proposals only; readability first; nothing built)

1. A preset for the straight-through downstream of an N/S router: keep the stateful (preset) forward and credit for the
   downstream that continues along the column, the stateless path only for turns (one per packet's route, rare on long
   N/S legs). Expected (estimate from the A/B, removed work only): up to ~37 E1 cycles/packet on N/S straight forwards,
   which would move the N/S rows from E1 (~172) to E0's bound (~140, as the E/W 3-hop). Readability cost: the forward
   picks a path per downstream name (compile-time preset for one name), one more branch in `forward<D>`; the VC0/VC1
   command-buffer-2 sharing (V4_BOUNDS §4 item 1) must be solved first or alongside, as N/S routers forward VC1 too.
2. Admission without the route: test the credits of every live downstream (or of the straight one) before the header
   arrives, or read the route word in the receiver's pass-start load group, so the header load overlaps. Expected
   (hypothesis): part of the ~17. Readability cost: small. **Designed in V4_ROUTE_PRELOAD.md (2026-11-02):** the route
   is preloaded at the end of the previous step, not in the pass-start group (that group cannot order a header load
   behind the count, V4_ROUTE_PRELOAD §2).
3. The decode (~8, hypothesis): only after 1-2 and an isolating A/B.
Order: (1) is the bulk; each needs its own A/B on the Galaxy N/S rows (and E/W controls), and the receiver ownership
work (v4rxo*, V4_RECEIVER_OWNERSHIP.md) owns the command-buffer layout, so (1) belongs there.

**(1) as planned (partial presets on forwards) is designed in V4_NS_PRESETS.md (2026-11-02; offline drafts v4rxnsa /
v4rxnsb, nothing run).** It replaces the straight-through preset above: every forward on a multi-downstream router keeps
its invariant fields preset and writes only source, destination, coordinate and length (data, buffer 2) and register,
coordinate and go (credit, buffer 1), whichever downstream it picks. **Measured (Galaxy job 169615, V4_NS_PRESETS §8.9):
at 2 KiB slots the N/S rows fall from 172.9-173.7 to 136.8-140.2 (3-hop N/S -36.1, 7-hop -36.1, corner -33.5, turn
-31.0), the 3-hop E/W row's level; built as v4rxnsf2.**

## 4. Files and jobs

- Variants: `variants/v4bdd` (instrument), `variants/v4bddsl`, `variants/v4bddml` (diagnostic, not for adoption).
- Galaxy: jobs 167648 (arms, 2 KiB study), 167673 (sl / ml on the Galaxy); trees `/data/aagarwal/frs-glx/trees/
  sls_bdd`, `sls_bddsl`, `sls_bddml` (hard-linked from `sls_lead`, each differing in its variant's files only; checked).
  Results `fabric_exp/results/glx167648/`, `glx167673/`; tables `consolidation/sls2k_report.py RESULTS arms`.
- P150: jobs 412 / 413 (`sls_sl_runs.sh`, labels `sls[abcd]sl_*`), 414 (PC samples, labels `bndp_sls{lead,sl,ml}_hop3`).
