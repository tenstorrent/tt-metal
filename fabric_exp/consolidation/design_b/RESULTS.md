# Design B: implementation and static evaluation (2026-10-31)

Variant: `fabric_exp/variants/v4rxdb` (base v4rx1). Offline only; nothing ran on a device. Expectations were written
first (`EXPECTATIONS.md`, unedited since). All numbers -O3, coverage builds under
`/home/aagarwal/.tt-metal-v3r-builds/cov_m/builds/`, debug twins under `/home/aagarwal/.tt-metal-v3r-builds/design_b/twins/`.

| build | source | what |
| --- | --- | --- |
| `v4rxdbs1_O3` | `design_b/stage1/` | stage 1: the shape only (v4rx1's ack loop, completions and placement) |
| `v4rxdb_O3` | `variants/v4rxdb` | **full design** (final) |
| `v4rxdbr2_O3` | (final without the DownstreamSet `Present` change) | same receiver loop as the final (identical/layout on every image) |
| `v4rxdbt2_O3` | (the proposal as written, with `keep_in_memory` after forwards; 8 cases) | evidence for deviation 1 |
| twins `s2a_rx1tv` | `design_b/stage2a/` | probe (attribution only): full design with VC1's/VC2's cursors back in `Tables` |

Scripts (this directory): `loopstat.py` (receiver loop function per image: instructions; sp accesses split into
callee-saved saves, zero-init stores and **body**; identical / layout / regs / differ), `listing.py` (loop function with
source lines and inline chain), `fndiff.py` (instructions by innermost inline function, A vs B), `hunks.py --regs --net`
(order-independent net instruction difference with source lines), `placement2.py` (placement.py with per-build exact
function names and a rename map).

Coverage: every config v4rx1_O3 built (1,666) builds in both stages; 2,086/2,216 overall, the 130 failures are the
`*_sc` cases (stream-register credits), as every v4 variant.

## 1. Final state

| object | members | placement |
| --- | --- | --- |
| `c0`, `c1`, `c2` (`CursorOf<CH>` = `Cursor<CH>` or `Absent<CH>`) | available, slot, index, uncompleted; `ack_slot` (first-level acks only); `idle_passes` (groups > 1 only) | loop locals, left to the compiler: DWARF "split" for every present cursor (c1 = VC1's, or VC2's on routers without VC1) |
| `t` (`Tables`) | `records0..2` (`Records<CH>`: `source` byte per slot, `kept[N]` {word, route} with first-level acks), `credits` (`CreditCounts`: completions[], acks[]) | frame object (memory by design: indexed at run time) |
| `frame_due` | bool | split |
| `ready` | arrival bit per acking channel + `txq_free`; constant 0 without first-level acks | pass-local value (GCC keeps the TXQ bit in one register after the arrival branch) |
| `me` (`Coordinates`, 4 B by value) | y, x, mesh_id | GCC: bytes in registers, spilled on Y-torus E1 (reloaded per ack); argument registers on one ERISC |
| `ds` (`DownstreamSet&`, kernel_main's) | `vc0`; `vc1` only on VC1 routers (`Present`) | memory (reference); the pointer is reloaded per pass on Y-torus E1 |
| constants | ring bounds, stream ids, credit block addresses (`CreditCounts`), `txq_free` | compile time |
| one ERISC: `single::ReceiverState` | {c0, c1, c2, tables, frame_due} | memory (single::run's frame); `receiver_pass(ReceiverState&, DownstreamSet&, Coordinates)` |

Placement (`placement2.py v4rx1_O3 v4rxdb_O3`, rename map cursor0->c0, tables->t): every E0 image the same; on E1 the
only change is the new `c1` "split" where a second channel runs (bh_vc1, bh_2d_vc2) and "-" (empty) elsewhere; c0
split, t frame, frame_due split, round frame, as before. Stage 1: the same kinds as v4rx1 on every image (credits now
inside `t`, a frame object either way).

## 2. Stage 1 against v4rx1 (`v4rxdbs1_O3`)

E0: .text byte-identical on all 634 two-ERISC E0 images. E1 and one-ERISC: every loop function differs in bytes
(classes: 0 identical, 0 layout, 48 regs (trimmed one-ERISC images with no receiver channel), rest differ).

| case | (d_insns, d_body) x images |
| --- | --- |
| bh_2d | (-8,+4)x16 (0,0)x4 (+13,+6)x8 |
| bh_1d | (+3,-2)x10 |
| bh_ring | (+7,-2)x10 |
| bh_torus | (0,0)x8 (+10,-6)x4 |
| bh_2d_vc2 | (-44,0)x4 (-27,0)x8 (-15,0)x16 |
| bh_vc1 | (-23..-11, 0..+2)x14 (+10,-1)x2 (+12,0)x6 |
| bh_neighbor | (-31,0)x4 |
| bh_2d_1erisc / bh_torus_1erisc | (+4/+5, 0) / (-1/+1, 0) |
| bh_2d_udm_gtest | (+15..+42, 0) |
| glx_express(_line) | (-27,+2) (0,0) (+5,+6) |

Why they differ (net instruction diffs, `hunks.py --net`):
- **GCC tail duplication / cross-jumping of the delivery arms**: the atomic-increment command block (about 13-30
  instructions) is duplicated in one build and shared in the other (bh_neighbor -31: the atomic-increment arm now
  jumps to a shared copy; bh_2d +13 / bh_torus +10: one more copy). The source of `deliver` is unchanged; what changed
  is its continuation (`act` returns into `retire` on every arm).
- **`Tables t{}` zeroed by a `memset` call** instead of 12-18 inline zero stores (prologue; bh_2d_vc2 -18 stores).
- **Credit block addresses as constants** (`CreditCounts`): `lui`+`addi` per credit site instead of loading the credit
  sender's pointer members (one ERISC: +4..+5 instructions, -3 loads), and two loop invariants saved around the
  per-round calls (bh_2d +4 body: cold, once per round or around `deliver_other`).
- **1D routing word**: the routing-fields stack copy is gone (-2 body per packet), as v4rx1r1 measured.
- **UDM one ERISC**: the forward path is tail-duplicated (relay / no-relay continuation) and a shrink-wrapped `s0`
  save appears on the two-downstream forward path (no body access on the per-packet path).
- **Stage-1 shim** (`cursor1`, `cursor2` in `Tables`, two `Absent<1>`): one extra zero-init store.
No per-packet spill was added in any class (every new body access is around a call or once per round). Stage-1 size:
VC1 E0+E1 23,208 (v4rx1 23,264); least free E1 3,704 (v4rx1 3,864: kernel_main's frame +176 B for the unused VC1
stand-in in `DownstreamSet`; fixed in the final, deviation 2).

## 3. Full design against v4rx1 (`v4rxdb_O3`)

E0: .text byte-identical on all 634 two-ERISC E0 images (the receiver program is not instantiated on E0).

| case | (d_insns, d_body) x images | against stage 1 |
| --- | --- | --- |
| bh_2d, glx_2d, glx_express_mesh | (-14,0)x16 (-1,0)x4 (+12,+6)x8 | (-6,-4)x16 (-1,0)x12 |
| bh_1d, glx_1d | (+2,-2) | (-1,0) |
| bh_ring, glx_ring | (+77,+3) | (+70,+5) |
| bh_torus | (+54,+11)x4 (+56,+9)x4 (+57,+10)x4 | (+46,+15)x4 (+54,+11)x4 (+57,+10)x4 |
| glx_express(_line) | (+23,+6) (+35,+6) (+52,+11) (+55,+10) | (+18,0) (+52,+11) (+55,+10) (+62,+4) |
| bh_2d_vc2 | (-48,-12)x8 (-11,+5)x4 (+1,+1)x8 (+2,+1)x8 | (-21,-12)x8 (+16,+1)x8 (+17,+1)x8 (+33,+5)x4 |
| bh_vc1 (VC1 serviced) | (-46..-14, -21..-14)x14 | (-28..-1, -23..-16)x14 |
| bh_vc1 (VC1 not serviced) | (+10,-1)x2 (+12,0)x6 | (0,0)x8 |
| bh_neighbor | (-32,0) | (-1,0) |
| bh_2d_1erisc / bh_2d_vc2_1erisc | (+4/+5, 0) / (-9..+3, 0) | (0,0) / (-8,0) |
| bh_torus_1erisc / glx_express_1erisc | (+5/+8, 0) / (+4..+9, 0) | (+4/+9, 0) / (+4..+10, 0) |
| bh_ring_1erisc | (+10,-2) | (-2,0) |
| bh_2d_udm_gtest | (+15..+42, 0) | (0,0) |

`mul` in the receiver loop: first-level-ack routers 1 -> 0 (the ack's header address); groups > 1 keep
credit_slots' each-slot identity (2D: 2). `minu` in the VC2 Y-torus loop 2 -> 0 (the T==1 form).

Where the ring/torus instructions come from (`fndiff.py`, ring twin rx1hr, v4rx1 517 -> 594, +77): the ack loop, the
general-form release, `PassRegisters`, the credit sender and v4rx1's `receive` body go (-75); `take_in`, `next_slot`,
the kept loads, the `ready` reduction, the T==1 release, the frame and the `act`/`receive` split come in (+105): the
receiver logic is +30 net. The other +47 is delivery code GCC duplicates: two more copies of the atomic-increment
block (+30), of the scatter local-write `set_transaction_id` sequence (+10), and small tails. On Y-torus N/S coverage configs the body accesses are spills on the two-downstream forward path
and the delivery payload address (see §5).

Size (`size_report.py v4rx1 v4rxdbs1 v4rxdb`): VC1 E0 + E1 23,168 B (v4rx1 23,264; limit 26,624, 3,456 left); E0
unchanged everywhere (largest 12,444, least free 1,604); largest one-ERISC binary 14,040 (v4rx1 14,160); least free E1
3,752 (v4rx1 3,864; deepest E1 1,168: Galaxy express one ERISC, the kept ring in `ReceiverState`); first-level-ack
two-ERISC E1 frames grow by the kept ring (bh_ring 4,496 -> 4,272 free).

## 4. Debug twins of the profiled builds (receiver loop function)

| build | rx1tv (Y-torus N/S + VC2, E1 1960938547760765240) | rx1tso (Y-torus N/S, 16246661628456555625) | rx1hr (ring, 16383455491875178035) | rx1js1e (one ERISC, 11904114896705350928) |
| --- | --- | --- | --- | --- |
| v4rx1 | 963 insns, body 39, mul 3, minu 2 | 566, 24, 1, 0 | 517, 11, 1, 0 | 622, 6, 1, 0 |
| stage 1 | 947, 41, 3, 2 | 547, 24, 1, 0 | 524, 9, 1, 0 | 621, 6, 1, 0 |
| stage 2a probe (VC2 cursor in memory) | 893, 46, 0, 0 | | | |
| **full** | **895, 38, 0, 0** | **591, 21, 0, 0** | **594, 14, 0, 0** | **630, 6, 0, 0** |

Per-pass/per-packet body accesses in the full design:
- rx1tv: `ds` reloaded once per pass; `me.y`/`me.x` reloaded per ack (v4rx1 reloaded them per packet); a reload of a
  stack word holding 1 for `frame_due = true` (up to 3 per pass; GCC rematerializes `true` from the stack instead of
  `li`); scatter path: the chunk encoding stored to the stack after its header load and reloaded 4 times. VC2's step
  has no stack access (v4rx1: 19 sp-relative accesses to VC2's in-memory cursor and records, plus zero stores to its dead `idle_passes`). The stage-2a probe has the same
  VC0-path reloads (ds, me) without the frame_due one: scalarizing VC2's cursor adds no VC0 per-packet spill here
  (v4rx2's delivery-coordinate spills do not appear).
- rx1tso: `me.y`/`me.x` reloaded per ack (v4rx1 kept them in t0/t2: new); the transaction-ID status base reloaded per
  completion (new); scatter path: **the chunk encoding stored to the stack 4 instructions after its header load**
  (`lbu a5,39(s7)` ... `sw a5,24(sp)`) and reloaded 3 times - v4rx2's loss mechanism on this row (v4rx1 kept it in a
  register); trid spills on the scatter path as in v4rx1.
- rx1hr: the transaction-ID status base reloaded per completion; scatter-path spills (cold for this row). No per-packet
  spill of the route word (v4rx1 stored the routing fields to the stack per packet).
- rx1js1e: none beyond v4rx1's (6, the same).

## 5. Expectations against results

| expectation (EXPECTATIONS.md) | result | held? why |
| --- | --- | --- |
| every E0 image .text identical, both stages | 634/634 identical in both | held |
| stage 1, 2D mesh classes 0 +-2, 0 | (-8,+4)x16 (0,0)x4 (+13,+6)x8 | **not held**: GCC tail duplication of the atomic-increment block flips with the new `act` continuation (+-13); +4/+6 body are two loop invariants saved around the per-round calls (cold). Layout-class effects, larger than predicted |
| stage 1, 1D -2, -2 | (+3,-2) | body held (routing word); +5 insns = one zero-init store and branch layout |
| stage 1, ring -2..+2, -2..0 / torus 0 +-2, 0 | (+7,-2) / (0,0)x8 (+10,-6)x4 | body held; ring +7 and torus +10 are tail-duplication flips; torus -6 is a cold `deliver_other` save/restore gone |
| stage 1, bh_2d_vc2 0 +-2, 0 | (-44..-15, 0) | **not held**: `Tables t{}` zeroed by `memset` (-18 stores) plus block sharing; body unchanged as predicted |
| stage 1, VC1 -3..+3, -2..0 | (-23..+12, 0..+2) | **not held** on instructions (memset, layout); body roughly held |
| stage 1, neighbor 0, 0 | (-31, 0) | **not held**: the atomic-increment arm shares a block instead of a duplicate copy |
| stage 1, one ERISC -4..+2, 0 | 2D (+4/+5, 0), ring (+12,-2), 1D (+9,-2), UDM (+15..+42, 0) | body held; instructions higher: credit addresses are `lui`+`addi` instead of a pointer load (+1 per site), UDM forward path tail-duplicated |
| stage 1 byte classes "layout/regs" on 2D/torus/ring/VC2/neighbor | all "differ" | **not held**: the memset and tail-duplication changes alter the instruction sequence, not only offsets |
| full, 2D mesh 0..+3, 0 | (-14,0)x16 (-1,0)x4 (+12,+6)x8 | instructions not held (layout, as stage 1); `keep_in_memory` dropped (deviation 1), so the full design equals stage 1 there within (-6,-4) |
| full, 1D -2, -2 (+0..+2) | (+2,-2) | held on body |
| full, ring +5..+45, -3..+2; `mul` 1 -> 0 | (+77,+3); mul 1 -> 0 | **not held** on size: the receiver logic is +30 net (inside the range), GCC duplicates delivery blocks for +47; body +3 (trid base reload per completion; scatter spills) |
| full, torus +5..+45, -3..+3 | (+54..+57, +9..+11) | **not held**: register pressure on the forward path (two-downstream forwards spill slot/size), the payload address spilled; one more loop-carried value (`ack_slot`) and GCC hoisting both `kept` bases (`sp+N`, `sp+N+4`) into callee-saved registers on these configs |
| full, bh_2d_vc2 like v4rx2 minus 2..6 | (-48,-12)x8 (-11,+5)x4 (+1/+2,+1)x16 (v4rx2: (-26,+8)x8 (-13,+4)x4 (+1,0)x16) | held, better than v4rx2 on 8 images: VC2's T==1 form drops its dead `idle_passes` stores; no `keep_in_memory` |
| full, VC1 -38..-3, -9..0 | (-46..-14, -21..-14) | held (better: one `ds` pointer, VC1's cursor scalarized without new spills) |
| full, neighbor 0..+2 | (-32, 0) | not held (stage-1 block sharing) |
| full, one ERISC FLA -5..+20, -4..+2 | torus (+5/+8, 0), express (+4..+9, 0), ring (+10,-2) | held |
| full, bh_2d_vc2_1erisc -8..0, 0 | (-9..+3, 0) | held (VC2's T==1 form) |
| full: 13 s-regs as now; ack_slot fits; me.x reload per packet as v4rx1 | 13 s-regs; on Y-torus E1 `me.y`/`me.x` reloaded per ack (rx1tso: new; v4rx1 kept them in caller-saved registers) and `ds` per pass (rx1tv) | partly held: everything fits only by spilling `me`/`ds`/the trid base, not the cursor |
| full, Y-torus + VC2: VC2's step writes 3-4 s-regs, VC0 path spills 2-6 values, body above v4rx1 | body 38 vs 39; VC0 per-pass reloads ds, me, frame_due's `true`; scatter encoding spilled | partly held: body not above v4rx1 (VC2's memory accesses gone), VC0 path reloads as predicted, no delivery-coordinate spill |
| `ready` stored to the stack right after the group (50/50) | never stored; after the arrival branch GCC keeps only the TXQ bit in one register (s1/t5) | held (one register) |
| placement: stage 1 same kinds; full c1/c2 split, nothing else moves | as predicted | held |
| size: VC1 E0+E1 at or below v4rx1; least free E1 3,700-3,760 | 23,168 (-96); 3,752 | held |
| hot: no `mul` for a slot/header address | 0 in all four twins (v4rx1: 3/1/1/1) | held |
| hot: TX CMD, STATUS, count back to back; status first used right after the group; no raw store | all four | held |
| hot: kept loads before the arrival test | all four (rx1tso/rx1hr through two hoisted base registers) | held |
| hot: header word + route loads back to back, then ack_slot advance (no `mul`), kept stores after it, src from `word >> 24` | all four; on Y-torus E1 the route-byte loads wait for the `me` reloads (loads from the stack) | held except the `me` reloads |
| hot: receive has no header loads for the route | all four | held |
| hot: T==1 release lean (no `minu`, no run test) | rx1tv minu 2 -> 0, mul 3 -> 0 | held |
| hot: frame tests the `ready` bit in a register | all four | held |
| ring: routing word to `kept` without a stack copy; before_forward shift + store | yes | held |
| one ERISC: `me` in argument registers; header loads wait only on `ack_slot`'s load | yes (the isra clone passes y and x pre-extracted) | held |

## 6. Deviations from the proposal

1. **No `keep_in_memory(downstreams.cursor)` after forwards.** The `DownstreamSet` is kernel_main's object reached
   through a reference, so GCC must keep it in memory anyway: the hint guarantees nothing new. Its only effect is on
   GCC's scheduling and allocation, and it is large and of both signs (`v4rxdbt2_O3` -> `v4rxdbr2_O3`, removing it:
   bh_vc1 (-60,-39)x6 ... (-6,-6)x8, bh_ring (-10,-6), bh_1d (-15,-3), bh_torus (+21..+30, +6..+14), bh_2d_vc2
   (-3..+33, -2..+10)). A hint whose effect is purely heuristic is banned by standing rule 2, so it is dropped. Note:
   with it the coverage torus configs were smaller; without it they spill on the forward path (§5).
2. **`DownstreamSet::vc1` exists only on VC1 routers** (`Present<has_vc1, ...>`), not as an unused stand-in: with the
   stand-in, kernel_main's frame grew on every non-VC1 router (E1 free stack -32 to -176 B; stage 1 least free E1
   3,704). Receiver loops unchanged (identical or layout on every image); VC1 E1 free stack 4,188 -> 4,108 (one copy
   of VC1's `Downstreams` through `present()`).
3. **`Channel<CH>` is two templates**: the primary (`serviced = false`, `first_level_ack = false`) for a channel the
   router does not run, and the specialization `Channel<CH, true>` with every trait. The proposal guards the
   static_asserts by `serviced`; a single struct would also need guarded array indexing (`RECEIVER_NUM_BUFFERS_ARRAY` and
   `RECEIVER_TO_ENTRY_IDX` have NUM_RECEIVER_CHANNELS entries, and `Ids` needs `num_slots` to form its type), so the
   unserviced form carries only the two facts the generic code asks about.
4. **A `NoRouting` policy** for channels that only deliver (the proposal's "Routing = none"): `read` returns 0,
   `decide` returns a Route that delivers and never forwards. `receive` and `act` then have one body; the forward block
   and the admission of forwards stay under `if constexpr (C::routes)` (VC2's `downstream_vc` is 2, which no
   `DownstreamSet` member answers). `via_relay` is admitted whenever the route delivers (one condition for routes and
   delivers-only channels).
5. **Code-profiling timer**: v4rx1's `NamedProfiler` lines kept as they are (`CodeProfilingTimer` is already a type
   name in fabric_code_profiling.hpp).
6. Minor: `HeaderWord::load` removed (no user left); `SlotRecords::of(index)` added for the T==1 credit;
   `CreditCounts` takes zero-initialised arrays as members (no constructor); `Tables t{}` value-initialises everything
   (GCC zeroes it with one `memset` call); the forwarding flags are `is_vc_receiver_forwarding_disabled[vc]` in the
   variant's ct_args (the per-channel array removed: only the v4 router used it there); `channel0_register_empty`
   reads `to_receiver_packets_sent_streams[0]` directly (receiver channel 0's register, as main; no `Channel<0>` needed
   where channel 0 is trimmed); `take_in` keeps v4rx1r1's watcher-only `ASSERT` on `ack_slot`.

## 7. Open issues

- **Register pressure on first-level-ack E1 loops.** The design carries one more value than v4rx1 (`ack_slot`) and
  GCC sometimes hoists both kept-record bases (`sp+N` for `word`, `sp+N+4` for `route`) into callee-saved registers
  (rx1tso, ring; rx1tv uses `sp` directly). The result on rx1tso is per-ack `me` reloads and the scatter chunk encoding
  stored right after its header load (v4rx2's measured loss mechanism, 4.4 cycles at that store there). The coverage
  Y-torus configs also spill the forward path. Not fixable by source placement within the rules; a design that
  carries less (or keeps one value fewer per pass) would be needed. The stage-2a probe shows VC2's scalarized cursor is
  not the cause on rx1tv.
- Static deltas on every class are dominated by GCC's tail-duplication choices for the delivery arms (+-13..+-31) that
  flip with benign edits; they are layout, not design. Stability under benign edits was not measured (no
  `jitter_edits.py` run) and -O2/-Os were not built.
- GCC rematerialises `frame_due = true` from a stack word on rx1tv (a load where `li` would do): an allocator
  artifact.

## 8. Hardware outlook (hypotheses until measured)

- Likely wins: Y-torus + VC2 rows (TorusY1HopNS write/scatter with VC2 on): the T==1 completion form (the source of
  v4rx1e's -9.6/-9.8, §17) and the one-ack scheme (v4rx1r1 -2.6 on top) without the end block; VC2's step no longer
  touches the stack. One-ERISC Y-torus: `me` by value removes the `me` loads v4rx1rg's ack waited on (§15). VC1 split /
  inter-mesh rows: 14-23 fewer stack accesses per loop (v4rx2 measured -0.4% / -1.0% with fewer).
- Risks: VC2-off first-level-ack rows. TorusY1HopNS scatter (rx1tso): the chunk-encoding spill (+2 to +4 expected;
  v4rx1r1 measured +2.8 there); ring 1-hop and half-ring: per-completion trid-base reload and the larger loop
  (v4rx1r1: 152.6 vs 150.7). 2D mesh, 1D: no per-packet change; within noise expected.
- Attribution builds ready: stage 1 (shape only), stage-2a probe (VC2 cursor in memory) for the VC2 rows.

## 9. What to read

`router_receiver.hpp`: `Channel` (both forms), `Cursor` / `Records` / `Tables` / `DownstreamSet`, `take_in`, `act`,
`retire`, `receive`, `release_completions`, `read_pass_registers`, `step`, `pass`, `run`. `router_routing.hpp`:
`Routing2D::read`/`decide` (route word), `Routing1D`, `NoRouting`. `router_completions.hpp`: `CreditCounts`,
`try_send_frame`. `router_single.hpp`: `ReceiverState`, `receiver_pass`.
