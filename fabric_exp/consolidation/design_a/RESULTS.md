# Design A: static results (2026-10-31)

Variant: `fabric_exp/variants/v4rxda` (full design, S1-S4). Stage snapshots (variant dirs) in `design_a/stages/`: `s1`
(shape), `s2` (+ one-slot release), `s3` (+ first-level-ack intake), `s4m` (probe: the proposal's fallback placement),
`pa`, `pg` (attribution probes, not designs). Expectations written before any build: `EXPECTATIONS.md`.

Builds (coverage, -O3, `cov_m`): `v4rxdas1_O3` (S1), `v4rxdas2_O3`, `v4rxdas3_O3`, `v4rxda_O3` (full); `v4rxdaq_O3`
(early S1 compile check, 16 cases), `v4rxdam_O3` (fallback probe, 7 cases). Debug twins of the four profiled logs:
`/home/aagarwal/.tt-metal-v3r-builds/design_a/tw_<build>_<tv|tso|hr|js1e>`; their receiver-loop listings with source
lines and inline chains: `/home/aagarwal/.tt-metal-v3r-builds/design_a/listings/<cfg>_<build>.lst`. Per-case
comparisons: `.../design_a/rxcmp_s1.txt`, `rxcmp_full.txt`, `textcmp_*.txt`.

Tools (this dir): `rxcmp.py` (per ELF: same / layout / rename / differ, and the receiver loop function's instruction and
sp-access deltas, whole function and its loop region), `cfgs.py`, `rxlist.py` (listing with source lines and inline
chains), `srcdiff.py` (instruction-count difference per source position between two listings), `spills.py`,
`covcheck.py`, `twins.sh`.

Caveat on the sp counts: they count sp-relative loads and stores. The full design's kept-header table is on the stack
but GCC addresses it through base registers it computes once (`addi s10,sp,96`), so its 2 loads per pass and 2 stores
per ack do not show in the sp counts; every sp access the counts show beyond v4rx1's is a spill or a reload.

## 1. State (as built)

| object | members | placement (DWARF, `placement.py`) |
| --- | --- | --- |
| `c0`, `c1`, `c2` (`CursorOf<CH>` = `Present<runs_channel<CH>(), Cursor, CH>`) | `position` {available, slot, index, ack_slot}, `counts` {uncompleted, idle_passes} | split (registers / spill slots) where the channel runs; `-` (absent) elsewhere |
| `frame_due` | bool | split |
| `tables` (`Tables` {ch0, ch1, ch2: `ChannelTable<CH>`}) | `sources` (`SlotRecords<N>`), `kept` (`std::array<KeptHeader, N>`, channels with first-level acks) | frame (stack); zero-initialized at entry (a `memset` call) |
| `credits` (`Credits`) | counts (memory), block addresses (constants) | frame |
| `me`, `n`, `routing_table` | | `-` (optimized; `me` lives on the stack in Y-torus builds, see §4) |
| `round` | | frame |
| one ERISC: `single::ReceiverState` | {c0, c1, c2, tables, credits, frame_due} | memory (record behind `receiver_pass`'s pointer) |

Stage 1 had v4rx1's placement: `cursor0` split, VC1's/VC2's cursor with its table in a `ChannelState` record inside
`tables`. `placement.py v4rx1_O3 v4rxdas1_O3` (bh_torus, bh_2d_vc2, bh_vc1, bh_ring): same placement on 72 of 72
images. Full: `cursor0` -> `c0` (split); `c1` split on bh_vc1 and bh_2d_vc2, absent on ring/torus; `tables`, `credits`,
`round` frame on every image.

Loop-carried registers seen in the listings (two ERISCs): ring/torus (full) index, uncompleted, available, ack_slot,
slot, frame_due, TX status, pass count, plus constants (stream base, buffer size, credit block bases, kept-table bases
on rx1tso/rx1hr); `idle_passes` and `ack_slot` of non-acking channels are gone, as expected.

## 2. Coverage, bytes, size, placement

- Coverage: S1, S2, S3 and full each build 2,086 of 2,216 configs; the 130 failures are the `*_sc` configs (counter-credit
  `static_assert`, as every v4 variant). All 1,666 configs `v4rx1_O3` built build (`covcheck.py`).
- E0 (sender) images: byte-identical to v4rx1 on every two-ERISC config, S1 and full.
- Bytes (`textcmp.py`, 1,666 shared images): S1 995 identical / 671 differ; full 967 / 699. With normalization
  (`rxcmp.py`): S1 995 same, 8 layout, 74 register renaming, 589 differ; full 967 same, 68 renaming, 631 differ.
- Size (`size_report.py v4rx1 v4rxdas1 v4rxda`, -O3): VC1 E0 + E1 23,264 / 23,264 / **23,156** (limit 26,624; 3,468
  left). Largest one-ERISC binary 14,160 / 14,140 / 14,216. Least free stack E0 1,604 in all; E1 3,864 / 3,864 /
  **3,712** (Galaxy express one-ERISC; the kept table). Deepest E1 1,056 -> 1,184. Ring E1 free 4,496 -> 4,256, torus
  4,432 -> 4,288.

### Receiver loop function against v4rx1 (rxcmp; "fn" whole function, "loop" its loop region; dI/dS x images)

| case | stage 1 | full |
| --- | --- | --- |
| bh_2d (28 E1) | identical | identical |
| bh_1d, glx_1d | -2/-2 | -2/-2 |
| bh_ring, glx_ring | -2/-2 | fn +63/-2, loop +66/+6 |
| bh_torus (12) | -1/0 x4, 0 x4 (E/W), +17/0 x4 (N/S) | fn +34/-4 x4, +51/+6 x4, +53/+7 x4 (loop +33/-2, +50/+7, +50/+9) |
| glx_express, _line | -1/0 | fn +31/+2 .. +62/+6 (loop +30/+4 .. +61/+9) |
| bh_2d_vc2 (28) | -10/-1 x12, -3/-1 x16 | fn -18/-1 x8, -11/-3 x4, +4/-7 x8, +5/-7 x8 (loop -14/+3, -9/0, +6/-4, +7/-4) |
| bh_vc1 (22 E1) | 6 identical, 6 renaming, -10/0 x4, -1/0 x4 | 6 identical, fn -34/-9 x4, -31/-14 x4, -27/-14 x2, -14/-10 x2, -6/-11 x2 |
| bh_2d_1erisc, bh_2d_mux | +2/0 | +2/0 |
| bh_torus_1erisc | +3/0 x8, +6/+2 x4 | +21/0 |
| bh_ring_1erisc | +3/-2 | +6/-2 |
| bh_2d_vc2_1erisc | -8/0, -5/0 | -16/0, -13/0 |
| telemetry, trim, allinstr 2D | identical or renaming | identical or renaming |

By stage (rxcmp between consecutive stages): S2 bh_2d_vc2 -6..+2 / -4, bh_vc1 -4..-2 / -4, ring -1, torus -2..+3,
one-ERISC -6..-8; S3 ring +66 / loop sp +8, torus +17..+53 / -4..+7, torus one-ERISC +21..+24; S4 bh_2d_vc2 -11..+13
/ +3..-3 (loop +8..+1), bh_vc1 -28..-5 / -11..-6. Ring and torus images do not change at S4 (one channel).

Profiled twins (fn insns / fn sp / loop sp): rx1tso 566/40/22 -> S1 547/40/22 -> S2 545 -> S3 588/34/16 -> full
588/34/16; rx1tv 963/64/38 -> 959/63/37 -> 913/59/33 -> 893/64/43 -> 891/52/34; rx1hr 517/28/11 -> 515/26/9 -> 514 ->
580/26/17 -> 580/26/17; rx1js1e (one ERISC, chosen N/S config 1841379155958340204) 622/14 -> 643/14.

### What makes the differences (srcdiff.py on the listings)
- S1 bh_torus N/S +17: GCC keeps one more inline copy of `deliver`'s atomic-increment arm (NoC command-register writes
  +7, `noc_fast_atomic_increment` +10); the rest is the same work under new names (`receive` -> `handle`,
  `send_credit_frame` moved). Off the write and forward paths.
- S1 bh_2d_vc2 -10: one fewer duplicated copy of the credit accumulation (-6) and the header decode (-2). 1D -2/-2: the
  routing word (the struct copy kept on the stack for the sparse-multicast call is gone), as v4rx1r1.
- S2: VC2's and VC1's 4-slot releases lose the two `sw zero, idle_passes` stores and the word load + shift; rx1tv's
  VC0 release drops from the general form (`minu` x2, `credit_slots`' run check with its `mul`) to 20 instructions
  (`release_completions` block 96 -> 47 static instructions over both channels).
- S3 ring +66: about +40 is GCC duplicating `deliver`'s atomic-increment arm again (NoC command writes +23, atomic
  increment +17), +7 three copies of the header-word decode (one per way into `handle`: no ack, ack of a later packet,
  ack of this packet), +12 the one ack and the step's kept loads and select against the ack loop.
- S4: VC2's (VC1's) cursor in registers: its step's stack loads and stores go; elsewhere new spills (bh_2d_vc2: loop
  sp +1..+8).

## 3. Expectations against results

| expectation (EXPECTATIONS.md) | result | held? / why |
| --- | --- | --- |
| E0 identical everywhere | identical (both stages) | held |
| S1 bh_2d identical/layout, 0..+5 on N/S | identical on all 28 | held |
| S1 bh_1d -2/-2 | -2/-2 | held |
| S1 bh_ring -3/-4 +-3 | -2/-2 | held |
| S1 bh_torus 0..+5 | +17 on the 4 N/S configs | **not held**: GCC duplicates an atomic-increment block (above); not on the hot paths |
| S1 bh_2d_vc2 identical/layout +-5 | -10/-1, -3/-1 | held but larger: fewer duplicated blocks |
| S1 bh_vc1 intermesh identical, edge -8..+5 | 6 identical + 6 renaming, edge -10 and -1 | held (edge -10 just past the range) |
| S1 one ERISC 0..+5 / +-5 | +2, +3, +6 | held (torus +6 at the edge) |
| S1 placement same | same 72/72 | held |
| Full bh_2d = S1 | identical | held |
| Full bh_1d = S1 | -2/-2 | held |
| Full bh_ring -10..+15 insns, loop sp +2..+6 | +63 insns, loop sp +6 | **not held** (insns): block duplication (+40) and the decode copies; the sp +6 is not the kept table (not sp-addressed) but new spills (NoC status base, scatter-path TX status and trid) |
| Full bh_torus -5..+20, loop sp +2..+6 | +34..+53, loop sp -2..+9 | **not held**: as the ring, plus `me` reloads and the scatter encoding spill (§4) |
| Full bh_2d_vc2 -35..0 insns, fn sp -9..+1 | -18..+5, fn sp -7..-1 | held (+4/+5 on 16 N/S configs just past) |
| Full bh_vc1 -40..-5, sp -15..-6 | -34..-6, -14..-9 | held |
| Full bh_2d_1erisc = S1, single::run +1 store | +2 (= S1) | held |
| Full bh_torus_1erisc -10..+20, sp -5..+2 | +21, 0 | held at the edge |
| rx1tso -5..+20, loop sp 24-28 | +22, loop sp 16 | insns at the edge; sp **not as reasoned** (kept table not sp-addressed; v4rx1's 4 caller-saved spills around the out-of-line delivery call became 2) but new per-packet stack loads appeared (§4) |
| rx1hr -10..+15, loop sp 13-17 | +63, loop sp 17 | insns **not held** (duplication); sp held |
| rx1tv -40..-10, loop sp 20-34 | -72, loop sp 34 | insns better than expected (S2 -46, S3 -20, S4 -2); sp held at the edge |
| rx1js1e -10..+20 | +21 | at the edge |
| Size: VC1 +-200 B, E1 free -256 at most | -108 B; E1 free -152 | held |
| Coverage all of v4rx1's | all | held |
| Listing 1: group's three reads back to back, no mask; count first used at its branch | rx1tso/tv/hr: `lw 1188(stream)`, `lw 4(txq)`, `lw 8(txq)` adjacent; count tested by `slli 21; beqz` after the kept loads | held |
| Listing 1: TX status held to the frame in a callee-saved register (tso, hr); likely spilled after its read on tv | tso `s2`, hr `s8`, first use `slli ..,15` at the frame; tv: `t6`, first use at the frame, saved only around the out-of-line delivery call and the scatter path's command-buffer wait | held; tv better than feared (no store after the group read) |
| Listing 2: kept loads before the count branch | `sh3add; lw; sh3add; lw` between the group and `beqz` on all three | held |
| Listing 3: header loads back to back, no `mul`, first use at the kept stores | hr `lw 40(ack)`, `lw 44(ack)` adjacent, the kept `sw`s after 4 other instructions; tso/tv word `lw` between the two route-byte `lbu`s; no `mul` in any ring/torus loop | held, except: on tso the route-byte addresses wait on `me.x`/`me.y` reloaded from the stack just before (`lw 16(sp)`, `lw 12(sp)`) - not expected |
| Listing 4: select by branch/`mv`, no stack | branch + `mv`s | held; GCC also makes three copies of the decode |
| Listing 5: no store right after a load on the per-packet path except kept/NoC stores | write and forward paths: held. Scatter path on rx1tso and rx1tv: the chunk encoding is stored to the stack right after its `lbu` (`sw ..,24(sp)` at router_local_delivery.hpp:102) and reloaded per chunk | **not held** on scatter (v4rx2's mechanism) |
| Listing 6: one-slot release, no `minu`/`mul`/`of_run` | 20 instructions, `lbu` of the source | held; but on rx1tso/rx1hr the NoC status base is reloaded from the stack in it (`lw 8(sp)` / `lw 4(sp)`), new |
| Listing 7: frame | raw status bit test; fresh read only if busy; four TXQ writes | held |
| One ERISC: state loads before the header loads | `lw ack_slot(record)` -> `add me` -> `lbu` route bytes; kept via the record pointer | held |

## 4. The S3 register cost (static finding, mechanism stated, hardware not run)

S3 (first-level-ack intake) is where the new stack traffic on rings and tori comes from (`spills.py`, profiled twins,
v4rx1 / S1 / S2 / S3 / full): rx1tso: scatter-encoding spill 0/0/0/1/1, `me` reloads 0/0/0/2/2, NoC-status-base reloads
0/0/0/1/1; rx1hr: status base 0/0/0/1/1; rx1tv: encoding spill 0/0/0/1/1. Per pass on rx1tso's write path that is
about four more stack loads (`me.x`, `me.y` at each ack, the status base at each release, the downstream pointer
`lw s8,28(sp)` at each pass start); v4rx1's write path had none. On the scatter path the encoding is stored right after
its load (v4rx2 sampled 4.4 cycles/packet at that store on rx1tv).

Static probes (twins only, not designs) to find which part of S3 costs the registers: `pg` (kept table at file scope
instead of run()'s frame) and `pa` (no `ack_slot`: the acked header's address from its index, with a `mul`). Neither
removes the encoding spill or the `me` reloads on rx1tso (pa / pg drop only the status-base reload). So it is not the
table's placement and not the ack pointer alone: S3 as a whole keeps more values live (ack_slot, the two kept values
loaded before the ack, the incoming word/route pair) and GCC's allocation tips; no source-level design change without
steering the compiler was found. Hypothesis (static, not isolated on hardware): this costs Y-torus rows a few cycles
per packet, more on scatter.

Fallback probe `s4m` (the proposal's fallback: `keep_in_memory` on every cursor at the end of its step on routers with
more than one receiver channel): rx1tv loop 34 -> 47 sp accesses (the encoding spill goes, VC0's cursor now makes a
store/load round trip every step); bh_vc1 fn +7..+54 insns / +28..+30 sp against the full design. As specified it
puts VC0's cursor in memory too; v4rx2m's narrower form (VC2's cursor only) is the one that kept VC0 lean.

## 5. Deviations from the proposal

1. `Channel<CH>::vc` is `vc_of_receiver_channel(CH)` (the existing ct_args helper; the proposal's formula, already
   there). `runs_channel<CH>()` is computed from a per-VC table `vc_received` (VC1's from `FABRIC_2D_VC1_SERVICED`,
   VC2's from `FABRIC_2D_VC2_SERVICED`, as v4rx1's flags), `NUM_RECEIVER_CHANNELS` and `is_receiver_channel_serviced`,
   without instantiating `Channel<CH>`.
2. The forwarding-disabled flags: the variant's ct_args array is now `is_vc_forwarding_disabled` (size `MAX_NUM_VCS`,
   indexed by VC); the `Channel` contract `static_assert` indexes it by `vc`.
3. `send_credit_frame(credits, group_txq)` moved from router_completions.hpp to router_receiver.hpp (it reads
   `any_channel_acks`); `acking<CH>()` (runs and acks) feeds `any_channel_acks` and `arrival_register<CH>()`.
4. `acknowledge` computes the kept index first (for the `ASSERT` on `ack_slot`), keeps v4rx1's cache invalidation at
   the ack, and stores the whole `KeptHeader` (route 0 on a channel that only delivers) rather than the route only
   where the channel routes.
5. `retire<CH>(c, sources, source)` folds v4rx1's `slot_retired` + `retire`; `note_received<CH>` names the capture and
   telemetry count; `handle` computes the transaction ID and size after admission, as v4rx1.
6. The kept table is value-initialized (`kept{}`): step reads the next slot's record before knowing a packet was acked
   (the value is unused then, but must not be indeterminate); GCC emits a `memset` call at `run()` entry.
7. Stage 1 kept VC1's and VC2's cursor with its table in a `ChannelState` record (v4rx1's placement); S4 removes it.
8. The design's `registers_empty` heartbeat change was not adopted (heartbeat on receiver channel 0, as v4rx1).
9. No change was made for performance: the S3 spills (§4) have no design-level fix found that does not steer GCC.

## 6. Open issues; rows expected to move on hardware (hypotheses)

- Y-torus, VC2 on (TorusY1HopNS write and scatter, rx1tv): S2's one-slot release removes the general-form VC0 release
  (§17's hypothesis for the VC2-on gain: -9.8 measured for the end block that carried the same one-slot credit), S4
  removes VC2's per-pass state traffic; expect a gain on write (v4rx1js/rg/r1 reached 192.7-194.2 from 205.1); on
  scatter the encoding spill gives part of it back.
- Y-torus, VC2 off (rx1tso): write about equal (r1 157.5 vs 158.8) or slightly worse from the four new per-packet stack
  loads; scatter likely worse (r1 203.6 vs 200.8, and the encoding spill).
- Ring 1-hop and half-ring (rx1hr): S3 as v4rx1r1 without its end block plus one status-base reload per release:
  within about +-3 of v4rx1 (r1 152.4 / 184.3 vs 150.7 / 182.4).
- 2D mesh and Galaxy 2D: identical images, no change. 1D line: -2 instructions, at the wire, no change.
- VC1 split / inter-mesh and 2D VC2 suites: S4 as v4rx2 (-0.4 / -1.0 %, VC2 suites -0.3 to -0.7 %), plus S2.
- One ERISC Y-torus: the ack's header loads wait behind `ack_slot` and `me` loads from the record (as v4rx1rg, 267.9
  vs v4rx1 271.4 / js 264.5).
- Static size grows on ring/torus E1 (+260 B ring, +160 B torus) from GCC's block duplication; E1 stack -152 B worst.

## 7. What a reviewer should read (readability)

router_receiver.hpp: the VC tables and `runs_channel`, `Channel<CH>`, `Position`/`Cursor`/`CursorOf`/`start`,
`KeptHeader`/`ChannelTable`/`Tables`, `retire`, `acknowledge`, `downstreams_for`, `handle`, `receive`, `step`,
`arrival_register`, `send_credit_frame`, `pass`, `run`. router_routing.hpp: `Routing2D::read` (landing and exits
inside), `Routing1D::read`, `Route`, `before_forward`. router_completions.hpp: `release_completions` (the T == 1
branch) and `SlotRecords::of`. router_single.hpp: `ReceiverState`, `receiver_pass`. fabric_erisc_router_v4.cpp:
`kernel_main`'s receiver branch (downstreams by VC). fabric_erisc_router_ct_args.hpp: `is_vc_forwarding_disabled`.
