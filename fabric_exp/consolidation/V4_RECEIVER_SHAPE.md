# v4 receiver structure cleanup: design (2026-10-30)

User: the receiver's pass reads as if it was written for VC0 and the other channels were bolted on; a readability pass
like the sender's (V4_SENDER_SHAPE.md). Base: v4pl6 (the lead). Same staging and acceptance as the sender's.

## 1. What reads badly (v4pl6 `router_receiver.hpp`)

1. Three spellings of one step. VC0's is written out inside `pass` (the first-level-ack group read, the timer,
   `receive`, `release_completions`); VC1's goes through `receive_and_complete`, VC2's through `deliver_and_complete`.
   All three are "at most one packet, then its completions".
2. State spelled per VC: VC0 as three objects (`Vc0Position`, `Vc0Counts`, `Vc0Records`, Present tags 0/10/20), VC1
   and VC2 as `ChannelState` records. `pass` takes 10 parameters; `single::ReceiverState` repeats the list.
3. First-level acks are a VC0 special case in `pass`, while `Channel<CH>::first_level_ack` exists for every channel and
   `receive` honours it: a VC1 channel with acks would never be acked (the host never sets it: an unchecked invariant).
4. Channel kinds scattered: "delivers only" (VC0 on speedy routers) and "drops" are traits; VC2 is recognised by
   `CH == VC2_RECEIVER_CHANNEL` inside the `drops` formula and by calling `deliver_and_complete`; `receive` hands two
   kinds to `receive_local`, VC2 calls `receive_local` directly.
5. Placement is implicit: VC0 is split into three objects so its counters stay registers (GCC does not split an
   aggregate holding a runtime-indexed array), VC1 is one record so its counters stay out of registers. Both rest on
   compiler behaviour; `keep_in_memory` (V4_PLACEMENT.md) can say it.
6. The code-profiling timer (main's `RECEIVER_CHANNEL_FORWARD`, host option `enable_rx_ch_fwd`, read by the test
   infra) wraps VC0's receive only; main times every receiver channel's step (`run_receiver_channel_step_impl`).

## 2. Decisions (user, 2026-10-30)

- First-level acks per channel: `channel_step<CH>` acks under `if constexpr (Channel<CH>::first_level_ack)`. The
  pass's grouped register read becomes: the packets-received count of every channel with acks, and the TX queue's
  status when any channel has acks, as one group. With acks on VC0 only (every config today) that is today's read.
- The timer is generic: each channel's receive inside it, as main. It compiles away unless the host enables it.
- Staging and acceptance as below (the sender's).
- No "drops" kind (user, 2026-10-30: the router is a dumb forwarder; a contradictory capture is the builder's to
  reject). The host turns a channel's forwarding off (`DISABLE_RX_CHx_FORWARDING`) where the channel delivers every
  packet (VC2's, VC0's on a speedy router) and where a channel-trimming capture saw nothing forwarded or written on its
  VC; with a consistent capture such a channel also consumed no packet, so trimming does not service it. Main's router
  consumed and credited packets on a serviced channel with forwarding off; that combination only comes from a capture
  saying a channel consumed packets but did nothing with them. Now: the builder `TT_FATAL`s on it (capture failure,
  `erisc_datamover_builder.cpp`, after the `DISABLE_RX_CHx_FORWARDING` args), and the router `static_assert`s the
  contract in `Channel<CH>` (not serviced, or delivers, or forwarding on). None of the 1,610 coverage configs has the
  combination.

## 3. Target shape

- Channel kind from `Channel<CH>` traits: routes, or delivers here (VC2's channel, `is_vc2`; VC0 on speedy routers).
- One `receive<CH>` dispatching on the kind under `if constexpr`.
- One `channel_step<CH>`: ack (channels with first-level acks), the timed receive, completions; it sets `frame_due`
  when the credit counts changed.
- State by placement class, one rule for every channel, as the sender's (user, 2026-10-30: the sender treats no VC
  specially, nor should the receiver; its channels all have one role):
  - each channel's `Cursor` {`Position`, `CompletionCounts`}, its per-packet scalars, and `frame_due`: left to the
    compiler (register candidates), as the sender's worker state and far rings;
  - each channel's `SlotRecords` (indexed by slot at run time) and the credit counts: in memory.
  Today VC0's cursor is split out so it can stay in registers and VC1's and VC2's are kept in a record by reference
  (v4p: VC1's counters as loop locals cost VC0 +8 cycles/packet, before placement control); step 1 keeps that layout,
  step 2 applies the one rule.
- `pass(cursor0, frame_due, tables, credits, downstreams0, downstreams1, me)`: the pass's register group,
  `channel_step` for each present channel in order, the credit frame. The single-ERISC router keeps the same objects.

## 4. Steps

- **Step 1 - nothing moves.** The reshaping (kinds, one `receive`, one `channel_step`, generic acks and timer, the
  state grouped by placement class with VC1's and VC2's still one record each). Accept: every image byte-identical to
  v4pl6 except code-profiling builds (the timer now on every channel), or a normalized per-function diff showing layout
  only; DWARF placement kinds unchanged (`placement.py`); validation 13/13. No perf runs when images are identical.
- **Step 2 - one placement rule.** Every channel's cursor left to the compiler, every channel's records in memory (VC1's
  and VC2's state split into cursor and records, the cursors no longer behind a reference). Images change only on
  routers with more than one receiver channel (VC1, VC2). Accept: placement as intended (`placement.py`); A/B on the VC1
  split and inter-mesh rows and the VC2 census suites (`sh3_vc2_runs.sh`), then the census; no suite regressing beyond
  noise. A loss there (register pressure on multi-channel routers) comes back with its mechanism before any fallback.

## 5. Step 1 done: `v4rx1` (2026-10-30; jobs 274-276)

v4rx1 = v4pl6 with the shape of §3, placement as today: VC0's cursor (`Vc0Cursor`) and `frame_due` loop locals of their
own; `Tables` {VC0's slot records, VC1's and VC2's `ChannelState` records}; the credit sender (`Credits`) an object of
its own; `PassRegisters` (each acking channel's count and the TX queue's status, one fenced group); `channel_step` and
one `receive`; kinds routes / delivers; the contract `static_assert`. Builder: the `TT_FATAL` of §2.

Two shapes were tried first and lost placement (offline, normalized per-function diffs against v4pl6):
- the credit sender inside `Tables`: GCC keeps the whole aggregate in memory (its records are indexed at run time), so
  the sender's constant block addresses were loaded from the stack at each use (2D receiver loop +9 stack accesses);
- VC0's cursor and `frame_due` in one `Counters` struct: not scalarized as the separate locals are (+4 stack accesses);
- `channel_step` returning a `changed` flag (`changed = ack; changed |= release; frame_due |= changed`): one more
  value live across the receive, two spill pairs on torus N/S; `frame_due` updated in place instead.
So grouping state into structs by placement class works only where the struct is in memory anyway.

Result (coverage 1,666/1,666, -O3; `normall.py`: byte-identical / layout-only / register renaming / different):
- every two-ERISC image byte-identical, layout-only or register-renamed (2D, 1D, ring, torus, Galaxy, mux, trim,
  telemetry); VC1 and VC2 one byte store/load pair at a new offset in the channel record (layout);
  `bh_vc1_allinstr` +73 to +77 instructions (the code-profiling timer now also on VC1's receive, as main);
- single-ERISC images: -2 to -8 instructions (1D, 2D, ring, torus), +1 on 12 Galaxy express configs, +15 on
  NeighborExchange (a duplicated atomic-increment block).
- Validation 13/13 (job 274). Single-ERISC A/B against v4pl6, same job, 2 repetitions each within 0.1 (job 275):
  2D 1-hop 198.4 / 198.4, 1D 1-hop 180.4 / 180.4, ring 1-hop 250.0 / 250.0, torus N/S 271.4 / 271.4,
  NeighborExchange write 246.8 / 246.2.
- The "drops" removal and the contract `static_assert` (applied after): all 1,666 images byte-identical to the
  step-1 images. Validation with the builder's `TT_FATAL` (job 276): 13/13 (both channel-trimming gtest suites, 63
  tests each, go through it with real captures; it never fires).
- **Step 1 accepted.**

## 6. Step 2 in progress: `v4rx2` (2026-10-30; jobs 277-282)

v4rx2 = v4rx1 with one placement rule for every channel: `cursor0`, `cursor1`, `cursor2` loop locals of their own
(`cursor_start<RUNS, TAG, CH>()`), `Tables` {records0, records1, records2} (`ChannelRecords<CH>`), `pass(cursor0,
cursor1, cursor2, frame_due, tables, credits, ...)`. Only VC1 and VC2 images change; their cursors are now scalarized.

- Validation 13/13 (job 277). VC1 split -0.4%, inter-mesh -1.0%. VC2 on (jobs 279, 281; both passes agree): 4352
  -0.3%, mix -0.7%, burst 0, 4352 one-ERISC 0, **torus Y +0.7%**: TorusY1HopNS scatter 244.0 -> 253.4, NSBidir
  scatter +9.4, NS write +2.5, Bidir write +2.8.
- PC samples, TorusY1HopNS scatter with VC2 on (job 282: v4rx1, v4rx2, v4rx1 again; `torusns_scatter.yaml`): 244.1 /
  253.5 / 244.1 cycles/packet. E0's loadable images are identical in both variants (20/20) and every E1 image differs
  (28/28), so the loss comes from the receiver ERISC's change alone. Busiest E1 (chip 5 channel 11), cycles/packet:
  stack loads and stores 5.1 / 12.7 (control 5.4). New spills on VC0's path in v4rx2: the TX queue status read in
  `read_pass_registers` (stored to the stack right after the register read: 2.1 at the store, 1.0 at the reload);
  the scatter chunk encoding (store right after its header load: 4.4; reload 1.2, then 2.6 at its first use); the
  NoC coordinate of the delivery writes (1.2 + 1.1). v4rx1 kept VC0's `available` on the stack (-2.7 in v4rx2).
  Callee-saved registers written by VC2's step (receiver channel 1 here): none in v4rx1, four in v4rx2 (s0, s4, s7,
  s10). VC2 carries no traffic in this row.
- Hypothesis: VC2's scalarized cursor holds four callee-saved registers across the whole pass, and VC0's delivery
  path spills for them. **NOT CONFIRMED (job 283):** the A/B below removed the spills but not the loss.
- Probe `v4rx2m` (not a design) = v4rx2 + `keep_in_memory(cursor2)` before and after VC2's step. Offline (the sampled
  E1 build): VC2's step writes no callee-saved register; VC0's stack accesses in the loop 18 -> 6 (v4rx1 7). Device
  (job 283, same job, VC2 on), cycles/packet v4rx1 / v4rx2 / v4rx2m: TorusY1HopNS scatter 244.1 / 253.5 / 253.4,
  TorusY1HopNS write 205.1 / 207.7 / 214.7. E0 images identical in all three. Sampled stack-access cycles on E1
  (scatter) 5.3 / 11.6 / 6.5: the delivery spills are gone, but v4rx2m computes `txq_free` right after the TX queue
  status read in `read_pass_registers` (6.2 at its first use, 9.5 on that line in all) and stores the bool to the
  stack. v4rx1 kept the raw status in a callee-saved register until the frame send, which hid the read's latency
  behind the whole VC0 step; v4rx2 stored it to the stack right after the read (2.1 + a reload).
- What it shows: these rows' rates follow where GCC places the first use of the pass's long-latency register reads
  (the TX queue status under first-level acks; the scatter encoding's header load), which no source line fixes today.

## 7. Frame first: `v4rx1f` (2026-10-31; job 284) - rejected

v4rx1f = v4rx1 with the credit frame at the start of the pass, right after the register group: the group reads the TX
queue's CMD and STATUS first, then the acking channels' packets-received registers raw (masked after the frame);
`txq_takes_frame` (the group's status with first-level acks, a read at the frame without); `send_credit_frame` only
flushes; `send_due_frame` at each round's end (waits for the queue). Coverage: every config v4rx1 built builds (the
stream-register-credit `_sc` cases added since fail the counter-credit static_assert, as v4rx1). Sampled Y-torus E1:
the group's three reads back to back, but the raw count is stored to the stack right after its read and reloaded after
the frame.

Device (row-first A/B, same job, two repetitions agreeing within 0.3), cycles/packet v4rx1 / v4rx1f:
- VC2 on: TorusY1HopNS write 205.1 / 209.2, scatter 244.1 / 248.1;
- VC2 off: TorusY1HopNS write 158.8 / 162.8, scatter 200.8 / 204.7; 2D 1-hop write 127.2 / 127.2, 2D scatter
  141.9 / 142.2;
- one ERISC: 2D 1-hop write 198.4 / 203.6, TorusY1HopNS write 271.4 / 281.1.
Expected +1 to +2 on first-level-ack rows (chat, 2026-10-30): **WRONG**, +4.

Hypothesis (not isolated): moving the frame delays the credits in time. On one ERISC the frame due at the end of a
receiver pass now waits for the sender's half-pass (the 2D 1-hop write row has no first-level acks, so no register
group, and still loses 5.2). On two ERISCs it waits for the next pass's group (about 8 cycles): rows without
first-level acks (no group) are unchanged; Y-torus N/S rows (receive-bound with no slack, V4_PROFILE §5: a cycle of
pass time costs 1.2 to 2.1 per packet through the credit round trip) lose 4. Not separated from the group's own cost.

## 8. Acks first: `v4rx1k` (2026-10-31; job 285)

v4rx1k = v4rx1 where first-level acks run (ring and torus wrap receivers): the register group (TX queue CMD and
STATUS first, then the acking channels' packets-received registers, raw, each into a scalar of its own between the
fences), then every acking channel's acks, then the frame (this pass's acks, the last pass's completions; the group's
status, no second read), then the steps (receive, release); `send_due_frame` at each round's end. Builds without
first-level acks are byte-identical to v4rx1 (coverage: every config v4rx1 built). Size equal (V4_SIZE.md). A first
build put the status into a `PassRegisters` record between the fences: GCC stored it to the stack right after its
read, before the next read issued, so the two reads ran in series; with scalars the reads issue back to back and the
status is first used at the frame test, after the ack loop.

Device (row-first A/B, same job, two repetitions agreeing within 0.1), cycles/packet v4rx1 / v4rx1k: ring 1-hop
150.7 / 150.6; TorusY1HopNS write 158.8 / 159.9, scatter 200.8 / 201.1; with VC2 205.1 / 205.7, 244.1 / 244.9; one
ERISC TorusY1HopNS write 271.4 / 274.8.

What it shows: acks leaving about a packet's handling earlier gain nothing on these rows, so ack timing does not bind
them (single A/B). The small losses fit completions going out a pass later (after the next pass's group and acks;
on one ERISC after the sender's half-pass): hypothesis, not isolated.
**WRONG (job 286, §9: a pass of completion lag costs 5.5 to 9.2):** Weak (2026-10-31): in time the completions leave only the group and the acks later (~20-30 cycles); a rough round-trip
budget (wire ~117, handling ~150-200, NIU flush >= 68, frame and flight, sender poll) is well inside the window (10
usable slots of 11 x ~160 cycles), where a credit lag costs nothing (toy model: scratchpad credit_loop.py). Competing
hypothesis: receiver work on passes where nothing new arrived but a frame is due, where no ack work covers the TX
status read. Isolating A/B: v4rx1 with every completion credit held one extra pass (same work, ~one pass of lag).

## 9. Completion lag probe (2026-10-31; job 286) and the half-ring receiver's stalls

Probe `v4rx1h0` / `v4rx1h1` (not a design) = v4rx1 + every proven completion group held one pass before it is credited
(`CompletionCounts::held`, credited at the next release; held groups flushed at each round's end), switched by a
volatile setting read once before the loop: both settings build byte-identical code (2,086 of 2,086 images), h0
credits at once, h1 one pass late. Device (row-first A/B, same job, two repetitions agreeing within 0.1), cycles/packet
v4rx1 / h0 / h1: 2D scatter 141.9 / 147.1 / 152.7; ring 1-hop 150.7 / 154.8 / 160.5; TorusY1HopNS write 158.8 / 163.3
/ 168.8, scatter 200.8 / 221.8 / 227.7; with VC2 205.1 / 226.0 / 234.5, 244.1 / 266.5 / 275.7.
- A pass of completion lag (h1 - h0, same code) costs 5.5 to 5.9 cycles/packet without VC2, 8.5 and 9.2 with VC2: a
  credit lag is not pipelined away on these rows (about 0.03 to 0.04 cycles/packet per cycle of lag, taking a pass as
  150 to 250 cycles). At that slope v4rx1k's ~20-30 cycles of completion delay predicts ~0.6-1.2: its +0.3 to +1.1 fits
  (not isolated). Why the window does not absorb it: open (§8's budget is WRONG).
- The probe's own code (h0 - v4rx1): +4 to +22 (the hold path and the round-end flush in the receiver loop).

Half-ring multicast (v4rx1, ring, first-level acks; the receiver forwards and delivers; pcsv rx1hr, 182.4 cycles/packet),
busiest E1 (chip 6 channel 7), stalls at first use, cycles/packet: the ack loop's `src_ch_id` byte 6.6 (plus 2.9 at the
`mul` computing each acked header's address); the handled packet's HeaderWord 6.0 (the 1D route field loads with it);
the packets-received register 5.5; the completion check (transaction ID status) 6.6; the NoC command buffer 7.1 (the
delivery's and forward's header loads land behind it). HeaderWord (offset 40) holds `src_ch_id` (byte 43).

## 10. Transaction-ID sent latency (2026-10-31; job 289; `TridSentLatency`, results/trid_sent_latency.csv)

Microbenchmark (tests/tt_metal/tt_fabric/erisc_load_ubench, kernels/trid_sent_latency.cpp): ERISC1 / NOC1 writes to a
Tensix core's L1 as the receiver's local writes (buffer 0 preset, transaction ID per write, VC 3), then polls the ID's
outgoing-write count (the router's proof) until it reads 0; 256 repetitions per case, each from an idle NIU. One poll
costs 5 cycles; a poll loop iteration ~11, so times are quantized to ~11. Cycles from the last command's go to the
poll that read 0, p50 (p10-p99 within one poll step):
- idle NIU: 1 x 64 B 34, 1 x 1 KiB 46, 1 x 2 KiB 57, 1 x 4352 B 101, 2 x 2176 B 56 (issue 119), 4 x 1088 B 33 (issue 195);
- behind one 4352 B write of another ID to another core: 46, 57, 79, 123; scatter unchanged (its own issue covers it).
Issue of one write (ID call with its ready read, then the command): 56 cycles.

Window for a check before the frame (pass k checking k-1, release-before-frame): from k-1's last go, the rest of its
step (retire ~10-13, loop and VC2's step) plus pass k's group (9-11) and acks (32-33): roughly 55-80 cycles (v4rx1
profiles, rx1tv and rx1hr; an estimate). So it would usually pass for writes up to 2 KiB and for scatter, and usually
fail for a 4352 B write (101); v4rx1's check after `receive` k has about a pass (180-245).

## 11. One ack per pass with a header group: `v4rx1j` (D1) and `v4rx1je` (D2) (2026-10-31; jobs 290-292)

v4rx1j = v4rx1 where VC0 has first-level acks: the pass's register group reads the packets-received register raw; then
a `fence r,r` and one group of header loads (the oldest unacked packet's `src_ch_id`, the handled packet's header word and
routing fields, raw, decoded right after the group); one ack per pass (`acknowledge_one`); the packet handled
(`handle`, the routing body split out of `receive`); at the pass's end `release_and_frame`: VC0's completion (groups of
one) with its proof and the TX queue's status read as one group, then the frame. v4rx1je = v4rx1j + the early-if-ready
completion and frame (the group also reads the TX status and the oldest uncredited slot's proof; credited and framed
before the packet when its writes have left). Coverage: every config v4rx1 built builds. Builds without first-level acks:
byte-identical except +4/+5 instructions of scheduling in some 2D, VC1 and Galaxy receivers (the `handle` split; with
the body inline they are identical). Ring and torus receivers: v4rx1j -4 to +9 instructions, v4rx1je +48 to +78.

Device (row-first A/B, same job, two repetitions), cycles/packet v4rx1 / v4rx1j / v4rx1je:
- VC2 off: TorusY1HopNS write 158.8 / 166.8 / 194.4, scatter 200.8 / 212.5 / 223.7; ring 1-hop 150.7 / 157.3 / 189.1;
  half-ring 182.4 / 190.0 / 221.0; 2D 1-hop write 127.2 / 127.2 / 127.2, 2D scatter 141.9 / 141.9 / 142.0;
- VC2 on: TorusY1HopNS write 205.1 / 194.2 / 210.8, scatter 244.1 / 234.2 / 249.7;
- one ERISC: TorusY1HopNS write 271.4 / 274.3 / 297.4.
v4rx1je loses everywhere: rejected.

Where v4rx1j's cycles moved (PC samples, busiest E1 by section, v4rx1 / v4rx1j; jobs 282 and 292):
- half-ring (182.4 / 190.0): acks 33.5 / 15.8, pass register group 11.0 / 0.5, header group with its fence - / 29.6,
  completion + frame 40.1 / 62.6, the packet's handling and the rest 97.9 / 81.4. The ack side and the header loads
  together save ~15; the end-of-pass block costs 22.5 more: 52 sampled instructions against 34, among them the
  `OneSlotDue` result kept on the stack (stored as a word and a byte, reloaded), the `due || frame_due` test, and the
  TX status's two reads (the transaction-ID check's stall is the same, 7.3 / 7.6);
- Y-torus scatter with VC2 (244.3 / 234.3): acks 32.4 / 16.1, group 8.7 / 1.9, header group - / 34.5, completion +
  frame 68.7 / 69.7, the rest 134.4 / 112.1.
Hypothesis (not isolated): the header group saves what was expected; the end-of-pass block's extra instructions
cancel it where VC0 is the only channel. Next: that block as scalars (no struct across the fences).

## 12. The end-of-pass block as scalars: `v4rx1js` (2026-10-31; jobs 293-294)

v4rx1js = v4rx1j with `release_and_frame` computing the due slot as scalars (`waiting`, `slot`, `due`), no record across
the fences (`OneSlotDue` and its helpers gone). Sampled half-ring receiver: the block's stack accesses 6 -> 0 (GCC
duplicates it on the handled / not-handled paths: 64 static instructions, one copy per pass). Coverage: every config
v4rx1 built builds; only first-level-ack receivers (ring, torus, express) differ from v4rx1j. Size equal (V4_SIZE.md).

Device (row-first A/B, same job, two repetitions), cycles/packet v4rx1 / v4rx1j / v4rx1js:
- VC2 off: ring 1-hop 150.7 / 157.3 / 148.2; TorusY1HopNS write 158.8 / 166.7 / 157.7, scatter 200.8 / 212.5 / 202.9;
  half-ring 182.4 / 189.0 / 181.8 (v4rx1js repetitions 180.1 / 183.5); 2D 1-hop write 127.2 / 127.2 / 127.2, 2D
  scatter 141.9 / 142.0 / 142.0;
- VC2 on: TorusY1HopNS write 205.1 / 194.2 / 193.2, scatter 244.1 / 234.2 / 233.2;
- one ERISC: TorusY1HopNS write 271.4 / 274.3 / 264.6.
The block's record was v4rx1j's loss (v4rx1js recovers 7 to 10 on every first-level-ack row). Expected from §11's
section split (chat): "D1 would net around -10 to -15 on the half-ring": **WRONG**, -0.6 (noisy); the ack-side saving
seen in v4rx1j's samples did not carry over in full. Gains: VC2 on -11 to -12, one ERISC -6.8, ring 1-hop -2.5, torus
write -1.1; loss: torus scatter without VC2 +2.1.

## 13. Where v4rx1js's cycles went (2026-10-31; job 295, PC samples, VC2 off, same E1 per row)

Rates reproduce: TorusY1HopNS scatter 200.8 / 202.9, half-ring 182.4 / 180.8 (v4rx1 / v4rx1js). By section,
cycles/packet, v4rx1 -> v4rx1js (scatter chip 5 channel 11; half-ring chip 5 channel 7):
- acks 30.9 -> 14.0 / 33.6 -> 12.5; pass register group 9.2 -> 1.2 / 11.6 -> 1.0; decode, loop and the rest 30.2 ->
  13.4 / 27.7 -> 14.8: together -41.7 / -44.6;
- the new header group with its fence +33.9 / +34.4;
- completion + frame 42.4 -> 51.4 / 39.5 -> 49.5 (+9.1 / +10.0);
- delivery, forward, retire unchanged.
In the header group: the pass's count read, then the acked header's address (`mul`, 7.0 at the `add` that uses it),
the `fence r,r` (4.2 at the load after it), then the group's loads (one latency, 6 on the first use). The count read,
the `mul` and the header loads still run one after another; v4rx1 had the count, the ack's header and the handled
packet's header in series, so the group overlaps one load latency, not two. The end block's extra ~10: the TX status's
two reads and the `due || frame_due` test (the transaction-ID check's stall is the same, 7.5 / 8.0 against 7.6).

## 14. One-insert header ring: `v4rx1rg` (2026-10-31; jobs 296-298)

v4rx1rg = v4rx1js where VC0 has first-level acks: no header group, no fence, no raw-load helpers. Each step reads the
next packet's ring record (`acked_headers<CH>[p.index]`: the header word as it is, `HeaderWord::bits`, and the routing
fields; local memory) first; then, if a packet arrived, the oldest unacked packet's header word and routing fields from
`Position::ack_slot` (a pointer advanced while they load, no `mul`), stored in its ring record (one insert per pass),
and the ack from its word; the packet handled takes the ring record, or this pass's when it was the one acked. End block
as v4rx1js. `HeaderWord::load` = `of(bits(...))`. Coverage: every config v4rx1 built builds; `Position::ack_slot`
changes the layout of in-memory cursors (VC1, VC2, one ERISC). E1 stack -152 B (V4_SIZE.md).

Device (row-first A/B, same job, two repetitions), cycles/packet v4rx1 / v4rx1js / v4rx1rg:
- VC2 off: TorusY1HopNS write 158.8 / 157.7 / 157.5, scatter 200.8 / 202.9 / 203.6; ring 1-hop 150.7 / 148.2 / 154.0;
  half-ring 182.4 / 180.0 / 185.6 (v4rx1rg repetitions 187.0 / 184.3); 2D 127.2 / 127.2 / 127.2, 142.0 / 141.8 / 141.9;
- VC2 on: TorusY1HopNS write 205.1 / 193.2 / 192.7, scatter 244.1 / 233.2 / 232.9;
- one ERISC: TorusY1HopNS write 271.4 / 264.5 / 267.9, 2D 1-hop write 198.4 / 198.4 / 198.4.

Ring 1-hop (PC samples, job 298, same E1, v4rx1js / v4rx1rg 148.3 / 154.0): the ack and header work 43.0 / 48.3; in
v4rx1rg the 1D routing fields (`LowLatencyRoutingFields`, a packed struct with an extension-word array) loaded at the ack
are stored to the stack right after the load (5.95 at that store) and reloaded twice (for the ring record and for the
handling). The 2D builds keep them in registers (listing). On the Y-torus rows the ring equals v4rx1js.

## 15. 1D routing as one word: `v4rx1r1` (2026-10-31; jobs 299-301)

v4rx1r1 = v4rx1rg with `Routing1D::Fields` the routing word (`routing_fields.value`): `decide` masks this hop's field;
`before_forward` shifts it out and, only in builds with extension words when the word empties, refills from the
header's own extension words (the shared update's no-refill branch rewrote unchanged words; the router no longer
reads them). Coverage: every config builds; only 1D receivers (line, ring) differ from v4rx1rg. Sampled ring 1-hop
receiver: the ack step's stack accesses 12 -> 8, the routing word stored to the ring record without a stack copy.

Device (row-first A/B, same job, two repetitions), cycles/packet v4rx1 / v4rx1js / v4rx1rg / v4rx1r1: ring 1-hop 150.7 /
148.2 / 154.0 / 152.6; half-ring 182.4 / 182.8 / 185.7 / 184.3 (v4rx1r1 repetitions 185.7 / 182.9); 1D line 1-hop and
3-hop 124.4 / 124.5 in all (at the wire); TorusY1HopNS write 158.8 / 157.7 / 157.5 / 157.5, scatter 200.8 / 203.0 /
203.6 / 203.6.

PC samples (same E1): ring 1-hop, v4rx1js / v4rx1r1 (148.3 / 152.4): the ack step costs the same in both (the count's
first use, then one L1 header latency, 7.1 at the ring store); the difference is spread over completion + frame (+2.5)
and retire (+1.5). One ERISC Y-torus, v4rx1js / v4rx1rg (264.6 / 268.1): the ack step 61.5 / 67.2; its route-byte
loads wait behind loads of the ack pointer and `me` from the in-memory state (8.5 at the route byte's first use).
Hypotheses (not isolated): on two ERISCs, the ring's two values carried across the loop (the ack pointer and the ring
base, held in s9) cost elsewhere; on one ERISC, the chain of state loads before the header loads.

## 16. Isolating the end block: `v4rx1e` (2026-10-31; job 302)

Probe v4rx1e (attribution only) = v4rx1 + the end block alone: the pass-start group without the TX status; VC0 (first-
level acks) releases at the pass's end in `release_and_frame` (v4rx1js's scalar version: TX status and completion proof
in one group, then the frame); v4rx1's ack loop and receive unchanged. Compile-checked on the Y-torus and ring configs
only. Device (same job, two repetitions agreeing within 0.1), cycles/packet v4rx1 / v4rx1e / v4rx1r1:
- VC2 on: TorusY1HopNS write 205.1 / 195.3 / 192.7, scatter 244.1 / 234.5 / 232.9;
- VC2 off: ring 1-hop 150.7 / 157.8 / 152.4, TorusY1HopNS scatter 200.8 / 204.2 / 203.6.
Attribution (A/B): the end block alone gives -9.8 / -9.6 with VC2 on and costs +7.1 (ring 1-hop) / +3.4 (torus scatter)
without VC2; the ack-side change of v4rx1r1 on top of it gives -2.6 / -1.6 with VC2 on, -5.4 / -0.6 without. Why the
same block helps with VC2 and costs without: not known (no profile of v4rx1e yet).

## 17. Why the end block helps with VC2 and costs without (2026-10-31; job 303, PC samples, TorusY1HopNS scatter)

Rates reproduce (v4rx1e 234.5 VC2 on, 204.2 off). Completion + frame on the busiest E1 (chip 5 channel 11), cycles/packet
v4rx1 -> v4rx1e: VC2 on 68.7 -> 56.3 (60 -> 48 sampled instructions); VC2 off 42.4 -> 50.3 (34 -> 43). The other
sections move by +-5 at most.
- VC2 on: v4rx1's `release_completions` for VC0 (groups of one under first-level acks) is compiled in its general form
  there: the group's end and size from `minu` (`to_end`, `n = min(waiting, to_end)`), `credit_slots`'s run check
  (`each_slot` from a `mul`), and the one-slot credit through the run path. In the VC2-off build GCC folds the same code
  to the one-slot case (34 instructions with the frame). v4rx1e's `release_and_frame` passes one slot explicitly
  (`credit_slots(records, slot, 1, credits)`), so it is lean in both builds. That is the VC2-on gain, not the TX status.
- VC2 off: v4rx1e adds the `due || frame_due` early-exit test (4 instructions: sltu, or, zext.b, beqz), the TX status's
  two reads and their address (3), and layout (a `mv` and a `j`): +9 instructions, +8 cycles; the transaction-ID
  check's stall is the same (6.9 / 8.0).
Hypothesis to A/B: a one-slot specialization of `release_completions` (`if constexpr (T == 1)`) gives v4rx1 the VC2-on
gain with no VC2-off cost, independent of where the TX status is read.
**Measured (§19):** the VC2-on part holds (-20); "no VC2-off cost" understated it: it gains 5 to 6.5 there too
(**WRONG** as "neutral without VC2", where v4rx1's folded code looked as lean).

## 18. Two redesigns, implemented and measured (2026-10-31; jobs 304-305)

Designs A (`v4rxda`, the receiver around channel features, sender-style pass) and B (`v4rxdb`, main-style generic
step, pack expansion over channels) were proposed, implemented and statically evaluated by two agents
(fabric_exp/consolidation/design_a, design_b: EXPECTATIONS.md written before any build, RESULTS.md after). A's stages:
s1 the shape, s2 + the one-slot (T == 1) release, s3 + the first-level-ack intake (one ack per pass, `ack_slot`, kept
headers), full + one placement rule (VC1/VC2 cursors as locals). `v4rxdas2` = A's stage 2 (no intake). Both agents
found the same static problem: the intake pushes ring and torus receivers past the register limit (scatter chunk
encoding spilled, `me` reloaded per ack, the NoC status base reloaded per completion check).

Device (row-first A/B, same jobs, two repetitions agreeing within 0.1 except the half-ring), cycles/packet,
v4rx1 / A stage 2 / A full / B full:
- VC2 off: TorusY1HopNS write 158.8 / 149.2 / 150.4 / 155.1, scatter 200.8 / 190.8 / 195.4 / 200.1; ring 1-hop 150.7 /
  140.8 / 137.6 / 141.7; half-ring 182.4 / 172.9 / 172.4 / 176.8; 2D 1-hop write 127.2 / 127.2 / 127.2 / 127.1, 2D
  scatter 141.9 / 142.0 / 141.8 / 142.1;
- VC2 on: TorusY1HopNS write 205.1 / 182.9 / 182.8 / 184.4, scatter 244.1 / 222.2 / 225.0 / 230.5;
- one ERISC: TorusY1HopNS write 271.4 / 262.0 / 260.9 / 273.1.
A's stage 2 gains 9.4 to 22.2 on every first-level-ack row and is flat on 2D; the intake on top of it (A full) is
mixed (ring 1-hop -3.2, torus scatter +4.6). Which of stage 2's two parts (the shape, the one-slot release) gives the
gain is not isolated: stage 1 has not been run. Validation (13 tests) has not been run on any of these.

## 19. Attribution of A's stage 2 (2026-10-31; jobs 306-307)

A's stage 1 (`v4rxdas1`, the shape: features in `Channel<CH>`, one step/receive/handle, the routing words, the frame
helper, the VC-indexed forwarding flags) and stage 2 (`v4rxdas2` = stage 1 + the explicit one-slot release; the two
differ only in router_completions.hpp). Row-first A/B, same jobs, two repetitions agreeing within 0.1, cycles/packet
v4rx1 / stage 1 / stage 2:
- VC2 off: TorusY1HopNS write 158.8 / 155.8 / 149.3, scatter 200.8 / 197.0 / 190.8; ring 1-hop 150.7 / 145.7 / 140.8;
  half-ring 182.4 / 178.2 / 172.9; 2D scatter 141.9 / 141.9 / 142.0;
- VC2 on: TorusY1HopNS write 205.1 / 203.3 / 182.9, scatter 244.1 / 242.3 / 222.2;
- one ERISC: TorusY1HopNS write 271.4 / 270.3 / 262.0.
Attribution (A/B): the shape -1.1 to -5.0 (VC2 off -3.0 to -5.0, VC2 on -1.8, one ERISC -1.1); the one-slot release
-4.9 to -6.5 without VC2, -20.1 / -20.4 with VC2, -8.3 on one ERISC. 2D unchanged. Why the explicit release gains
without VC2 too (v4rx1's code there looked folded, §17): not profiled.

## 20. Temporary lead: `v4rxdas2` (2026-10-31, user)

Design A's stage 2 (the feature-based shape + the explicit one-slot release, v4rx1's ack loop and placement) is the
temporary receiver lead, on the row A/Bs of §18-§19 (no full census: the user judged the pre-screen enough; mux and
UDM rows are bound by the mux, as the mux v1/v2 comparisons show). Next (user): a fresh-context analysis of the
first-level-ack intake (kept headers), placement stability, a completion rethink (readability; cost that does not grow
with VCs; no held completions on rings under bubble flow control) and grouped arrival-register reads with or without
first-level acks, readability as the anchor and no regression from stage 2.
Validation (job 308, variant_val_runs.sh, 13 runs, data-checked, hang capture armed): 13/13, failed none. Two ERISCs:
mesh 96, 1D 143, ring 19, Y-torus 12, mux 19, VC2 10, NeighborExchange 7 tests, channel-trimming gtests 63/63; one
ERISC: mesh 96, ring 19, VC2 10, NeighborExchange 7, trimming 63/63. The VC1 split (census_vc1_runs.sh) was not run.
VC1 split (census_vc1_runs.sh, jobs 309-310, two passes in swapped router order agreeing within 0.2): data-checked
`vc1_alloc` 25/25 per rank and `vc1_inter` 8/8 per rank, validation passed, both routers. Cycles/packet v4rx1 ->
v4rxdas2 (mean of passes): 1-hop E/W and N/S and the burst rows -0.1 to 0; multicast 157.8 -> 159.2 (+1.4); 2-hop
246.9 -> 248.9 (+2.0) at 16 B, 264.0 -> 265.8 (+1.7) at 4352 B; 2-hop merge 146.2 -> 147.6 (+1.4), 150.0 -> 151.3
(+1.3); split suite mean +0.35%; `vc1_inter` mean +0.02%. The multi-hop VC1 rows regress, reproducibly; not attributed
(the shape or the one-slot release; VC1's 4-slot rings take the one-slot path).

## 21. VC2-on rows run another slot split (2026-10-31; CPU-only mock, TT_FABRIC_EXP_LOG_SLOTS=1)

Found by the slot-allocation review (code read), confirmed by mock-cluster runs of test_tt_fabric with the device runs'
host library and settings (kernel 2, flags 13129088), Y-torus and 2D mesh, VC2 on and off (scratchpad `vc2slots/`):
- VC2 off: VC0 sender slots {17, 16, 16, 16}, receiver {16} (the study's 2D/torus split).
- VC2 on: VC0 sender slots {43, 7, 7, 7}, receiver {11}; VC2 sender {2}, receiver {4}. Main's VC0+VC2 tier row for
  Blackhole {7, 11, 0, 0, 2, 4} plus the spare slots to VC0's worker; the study's slot switches refuse VC2
  (fabric_static_sized_channels_allocator.cpp, `TT_FATAL(!vc1_present && !vc2_present ...)`, and the VC1 default
  requires no VC2).
Consequences:
- Every VC2-on row so far (§12-§19, jobs 302-307, 314) ran VC0 with an 11-slot receiver ring (not a power of 2, so
  `ring_add` is a compare; groups of 1 under first-level acks, 2 on the mesh) and 7-slot forwarding channels. A/Bs
  between routers at VC2 on stay valid (same split in the same job); a VC2-on vs VC2-off difference measures the split
  and VC2's code together, so "what VC2 costs" has not been measured.
- §17's finding that GCC folds VC0's one-slot release in the VC2-off build but not in the VC2-on build fits the ring
  depth (16, a power of 2, vs 11), not VC2's presence: hypothesis, open.

## 22. Design C's variants on the device (2026-10-31; jobs 312-315)

Design C (offline analysis agent, `design_c/RESULTS.md`): `v4rxc1` (the handled packet's header and route loads as one
fenced group before the acks), `v4rxc2b` (every channel's packets-received register in the pass-start group, the TX
status only where a channel acks), `v4rxc6` (c1 + c2b), `v4rxc4` (`keep_in_memory` on the VC1/VC2 cursors), `v4rxc5`
(c3 + catch-up of single-slot releases). `v4rxc3` (one completion skeleton) is byte-identical to `v4rxdas2` on all
2,086 coverage images (checked independently: loadable bytes), so it was not run. Row A/Bs, two repetitions agreeing
within 0.1 (row_ab_runs.sh, tags rxck, rxck2, rxck1e, rxckv2; the lead in every job); VC1 split census_vc1_runs.sh
passes rxc / rxcb in swapped order, data-checked 24/24 runs. Cycles/packet, the lead and the change against it:

| row | lead | c1 | c2b | c6 | c4 | c5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ring 1-hop | 140.8 | +6.9 | 0 | +6.9 | 0 | +11.7 |
| half-ring | 172.9 | +5.1 | 0 | +5.1 | 0 | +12.4 |
| TorusY1HopNS write | 149.2 | +3.7 | 0 | +3.7 | 0 | +11.4 |
| TorusY1HopNS scatter | 190.9 | +1.6 | 0 | +1.6 | 0 | +8.9 |
| 2D E/W write | 127.2 | 0 | -0.3 | -0.3 | 0 | 0 |
| 2D E/W scatter | 142.0 | 0 | -2.4 | -2.4 | 0 | 0 |
| one ERISC TorusY1HopNS | 262.0 | -1.6 | 0 | -1.6 | 0 | +3.1 |
| VC2 on: TorusY1HopNS write | 182.9 | -0.2 | -0.7 | -0.2 | 0 | +7.3 |
| VC2 on: TorusY1HopNS scatter | 222.2 | -1.2 | -4.2 | -3.7 | 0 | +7.5 |
| VC2 on: 2D N/S write | 164.2 | -0.1 | -0.2 | -0.2 | 0 | 0 |
| VC2 on: 2D E/W scatter | 190.6 | 0 | -7.0 | -6.9 | +1.5 | +2.8 |
| VC1: 1-hop, burst rows | 140.7-140.9 | 0 | 0 to +0.1 | 0 to +0.2 | 0 | 0 to +3.4 |
| VC1: multicast | 155.2 | -1.0 | -6.0 | -8.0 | -0.1 | +1.3 |
| VC1: 2-hop 16 B / 4352 B | 248.9 / 265.8 | -0.9 / -0.5 | -8.2 / -6.3 | -8.9 / -6.4 | 0 | +3.2 / +3.7 |
| VC1: 2-hop merge 16 B / 4352 B | 147.6 / 151.2 | -1.0 / -0.8 | -7.3 / -7.1 | -10.1 / -8.1 | 0 | +1.1 / +1.3 |
| VC1 split suite mean | | -0.19% | -1.44% | -1.74% | -0.01% | +0.72% |
| `vc1_inter` mean | | -0.35% | -0.28% | -0.66% | 0.00% | +0.15% |

(VC2-on rows at VC0 43/7/11 slots, §21. The lead's VC1 multicast measured 159.2 in jobs 309-310 and 155.2 here: compare
VC1 multicast within one job only.)
- c2b gains on every scatter row with more than one active receiver channel or VC (2D, VC2 on, VC1 multi-hop) and is
  byte-identical on the first-level-ack routers without VC2. On the VC1 multi-hop rows it gains more than §20's
  regression of the lead against v4rx1 (+1.3 to +2.0; v4rx1 not in this job). Mechanism (VC2/VC1 idle poll's stall
  gone, design C Q4): hypothesis.
- c1 loses on every first-level-ack row with two ERISCs and VC2 off (+1.6 to +6.9) and gains a little elsewhere (one
  ERISC, VC2 on, VC1); design C expected -3 to -6 (WRONG, marked there). Why: not known (no profile).
- c5 loses everywhere it changes code (+7 to +12 on first-level-ack rows). Hypothesis: the catch-up's extra due test
  and proof read per pass cost more than any lag it removes on these rows. Steady-state lag: still unmeasured.
- c4 neutral except 2D VC2-on scatter (+1.5).

## 23. Where `v4rxc1` loses (2026-10-31; job 316, PC samples, E1 of the busiest receiver)

Rates reproduce: ring 1-hop 140.8 / 147.7, TorusY1HopNS 149.3 / 152.9 (lead / c1). E0 images are identical, so the
loss comes through E1's change; the split below is E1's (samples cannot name the bound). Cycles/packet by section,
lead -> c1, ring 1-hop and TorusY1HopNS (design_tools/pcs_insns.py output):
- receive/decode ("rest"): -7.9 / -9.3. The header word's first-use stall (`srli` of the header word, 8.3 on the
  ring) is gone, as intended.
- acks: +8.8 / +6.4. The ack loop's ring arithmetic right after the stream-register store (`add`, `andi`) stalls
  +4.0 more than the lead's: hypothesis, the store to the packets-received register waits for the header group's
  loads, so their latency is paid at the ack's first memory operation instead of being hidden. Also the
  `available != 0` path now jumps out of line and back (+2.4) and the credit block address is rebuilt (+2.5).
- completion + frame: +9.8 / +7.8. Same instructions, but the constants the lead keeps in registers across the loop
  (the transaction-ID status base, the credit block, the frame source) are rebuilt with `lui` + `addi` at each use:
  hot `lui` samples 3.7 -> 8.2 (ring), 3.0 -> 8.1 (torus). Hypothesis: the header word and route fields live across the
  ack loop take the registers those constants held.
- delivery -2.8 / -1.7, retire -0.8 / -0.2.
So the overlap design C counted on is not realized: the header latency moves to the ack's register store, and the
longer live ranges cost about 7-9 cycles/packet of rebuilt constants. Both mechanisms are hypotheses from samples; an
isolating A/B would be c1 with the header loads placed after the ack's register store (inside the ack step).

## 24. Lead candidate `v4rxc7` = c3 + c2b (2026-10-31, user)

v4rxdas2 + design C's completion skeleton (`v4rxc3`'s router_completions.hpp) + one rule for the pass's register group
(`v4rxc2b`'s router_receiver.hpp) + one comment at `ChannelState` (VC1's/VC2's cursors stay in memory by GCC's choice,
not by contract; design C Q2). c1, c4, c5 and c6 are not taken (§22, §23).
Expected before the build: every coverage image byte-identical to v4rxc2b's (c3 is byte-identical to the lead; the
comment changes no code), so jobs 312-315's v4rxc2b numbers and its VC1-split data checks apply unchanged; size as
v4rxc2b (VC1 E0+E1 23,376). Then validation 13/13 (variant_val_runs.sh) before adopting.
Built (coverage -O3, 2,086 of 2,216 configs; the 130 `*_sc` fail by design, as every v4 build): byte-identical to
v4rxc2b on all 2,086 (loadable bytes), as expected.
Validation (job 317, variant_val_runs.sh v4rxc7, data-checked, hang capture armed): 13/13, failed none. Two ERISCs:
mesh 96, 1D 143, ring 19, Y-torus 12, mux 19, VC2 10, NeighborExchange 7, trimming gtests 63/63; one ERISC: mesh 96,
ring 19, VC2 10, NeighborExchange 7, trimming 63/63. VC1 split: v4rxc2b's images, data-checked 24/24 (job 315).
`v4rxc7` is the temporary receiver lead from here (replacing v4rxdas2; row A/Bs of §22, no full census).

## 25. Study slot rows for VC1/VC2 fabrics (2026-11-01, user: fixed rows at 4352 B, a general rule later)

Host change (tt_metal/fabric/builder/fabric_static_sized_channels_allocator.cpp, the study's slot block): one table of
rows {VC0 worker, VC0 forwarding, VC0 receiver, VC1 forwarding, VC1 receiver, VC2 worker, VC2 receiver} replaces the
VC1-only default. Study router (kernel 2), Blackhole, the measured shapes, applied only when the row's design worker
fits (the default 4352 B packet: 81 slots; a larger packet keeps the tier table instead of failing):
- VC0+VC1 19, 9, 16, 5, 4 (unchanged); VC0+VC2 17, 14, 16 + 2, 4 (new; was the tier's 43, 7, 11 + 2, 4, §21);
  VC0+VC1+VC2 19, 8, 16, 4, 4 + 2, 4 (new; not measured). VC0-only fabrics unchanged (2D 17/16/16, 1D 26/25/32).
- Switches: TT_FABRIC_EXP_SLOTS_VC2=w,f0,r0,w2,r2; TT_FABRIC_EXP_SLOTS_VC1 takes 7 values with VC2;
  TT_FABRIC_EXP_TIER_SLOTS=1 restores the tier table.
Checked on CPU-only mocks (TT_FABRIC_EXP_LOG_SLOTS=1, scratchpad slotmock/): every shape as listed, the switches, one
ERISC, and 14 KB packets falling back to the tier (VC0+VC2 11/2/4 + 2/2; VC1 7/2/4 + 2/2) without the old VC1
default's capacity failure. VC2-on rows from here use 17/14/16 + 2/4: not comparable with jobs 302-315's VC2 rows.

## 26. VC2 on/off at the same VC0 split (2026-11-01; job 318, v4rxc7, slots logged per run)

row_ab_runs.sh tags vcoff16, vcoff14, vc2on, vc2tier; two repetitions agreeing within 0.2. Cycles/packet:

| row | VC2 off, 17/16/16 | VC2 off, 17/14/16 | VC2 on, 17/14/16 + 2/4 | VC2 on, tier 43/7/11 + 2/4 |
| --- | ---: | ---: | ---: | ---: |
| 2D E/W write 4352 | 126.9 | 126.9 | 152.6 | 163.9 |
| 2D E/W scatter | 139.6 | 139.6 | 153.7 | 183.6 |
| 2D N/S write 4352 | 127.3 | 127.3 | 152.5 | 164.0 |
| TorusY1HopNS write | 149.2 | 149.2 | 181.5 | 182.2 |
| TorusY1HopNS scatter | 190.9 | 190.8 | 208.8 | 218.0 |

- VC0 forwarding 16 -> 14: no effect on these rows (within 0.1).
- VC2 compiled in and idle, same VC0 split: +25.6 / +25.2 on 2D writes, +14.1 on 2D scatter, +32.3 on torus write,
  +17.9 on torus scatter. Which ERISC pays it and for what: not known (no A/B yet).
- The old tier split, against the new one (both VC2 on): +11.3 on 2D writes, +29.9 on 2D scatter, +9.2 on torus
  scatter, +0.7 on torus write. So §21's caveat held for 2D and scatter rows; on torus write the old gap was VC2's code.

## 27. Where idle VC2's cycles go on each ERISC (2026-11-01; job 319, PC samples, v4rxc7, VC0 17/14/16 both)

Rates reproduce (2D E/W write 126.9 -> 152.4, TorusY1HopNS 149.1 -> 181.5). Busiest E0 and E1, cycles/packet VC2 off
-> on, by VC (design_tools/vcsplit.py: VC2's code by its template channel, E0 visit_worker<4>, E1 step<1>).
Samples add up to the row's rate on both ERISCs, so this splits cycles within each ERISC; it does not name the bound.
- 2D E/W write (+25.5). E0: VC2's worker visit (idle; its connection check, its free-slots test) +20.1; VC0's
  worker credit as a whole inline-write command +8.1; VC0's send +7.5; the rest -10.1. E1: VC2's receiver step (its
  register read in the pass group, poll, completions) +20.4; VC0's step +2.4; frame and loop +2.8.
- TorusY1HopNS write (+32.4). E0: VC0's worker credit +27.1 (mostly the command buffer's ready wait, 6.6, and the
  whole command's ~20 instructions), VC2's visit +20.8, VC0's send +7.6, the loop -15.3. E1: VC2's step +14.3, VC0's
  step +12.5.
Hypotheses for the A/B (each probe valid only without VC2 traffic): (1) E0's idle VC2 visit; (2) VC0's worker credit
losing its preset command buffer with VC2 present (V4_VC2.md: VC2's credit shares the write command buffer); (3) E1's
idle VC2 step.

## 28. VC1 + VC2 together: `v4rxc7v` (2026-11-01, user: a test)

The limitation was not v4's: v4's VC2 gate admitted only routers without VC1 (V4_VC2.md: "what this host can run and
check"), while the host builds VC1 + VC2 on any multi-mesh fabric with VC2 on (fabric_builder_context.cpp). Main's own
router cannot build it: two ERISCs fail on its stack limit ("stack usage is 2176 bytes [-Werror=stack-usage=]"), one
ERISC on "VC2 receiver channel not serviced" (job 320's capture). v4rxc7v = v4rxc7 with `!has_vc1` dropped from
`vc2_config_supported` (one line). Captured with itself in the v2 slot (job 321; coverage cases bh_vc1_vc2,
bh_vc1_vc2_1erisc): every router config JIT-built (90 two-ERISC, 48 one-ERISC images). Coverage -O3: 2,152 of 2,282
configs (the 130 `*_sc` fail by design); the 66 new VC1+VC2 configs build; the other 2,086 images byte-identical to
v4rxc7. Size: VC1+VC2 E0 13,792 + E1 12,284 = 26,076 B against the 26,624 B kernel config buffer (548 B left, before
runtime args); one ERISC 21,128 B; least free stack E0 1,476, E1 3,408. Slots: 19/8/16/4/4 + 2/4 (§25).

## 29. Idle VC2's cost is E0's: probe A/B (2026-11-01; job 322, slots 17/14/16 + 2/4 logged in every run)

Probes on v4rxc7, each lean on one ERISC only and valid only without VC2 traffic (each changes only its ERISC's VC2
images, offline check): v4rxv2p1 E0 neither reads nor visits VC2's worker channel; v4rxv2p2 VC0's worker credit keeps
its preset command buffer with VC2 compiled in; v4rxv2p3 E1 neither reads nor steps VC2's receiver channel. Two
repetitions agreeing within 0.3; cycles/packet:

| row | VC2 off | VC2 on | p1 (E0 no VC2 visit) | p2 (VC0 credit preset) | p3 (E1 no VC2 step) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2D E/W write 4352 | 126.9 | 152.7 | 129.3 (-23.4) | 150.7 (-2.0) | 155.7 (+3.0) |
| TorusY1HopNS write | 149.2 | 181.5 | 167.2 (-14.3) | 175.9 (-5.6) | 182.9 (+1.4) |

- Bound (A/B): E0 on both rows. E0's idle VC2 visit is 91% of the 2D gap (23.4 of 25.8) and 44% of the torus gap;
  VC0's worker credit losing its preset is 2.0 / 5.6; removing E1's VC2 step does not help (+1.4 to +3.0, unexplained).
  §27's hypotheses: (1) holds, (2) holds and is small, (3) does not bind.
- Torus: about 12 of the 32.3 remain after p1 and p2 (if additive; not run combined).
- Inside the idle visit (job 319 samples, 2D, 20.1 cycles/packet): the connection check is most of it: VC2's
  `connection_semaphore` pointer loaded from the stack (`lw a5,12(sp)`, 3.2), the semaphore through it (2.7), and the
  compare stalled on that load (`beq`, router_sender.hpp:463, 6.9); the free-slots test ~2.6. Hypothesis, from samples.
Device (job 323, vc1vc2_runs.sh a, v4rxc7v, slots logged 19/8/16/4/4 + VC2 2/4): every run passed, no hang.
Two ERISCs: vc1_alloc 25/25 per rank (data-checked), vc1_inter 8/8 per rank; one ERISC: vc1_alloc 25/25 per rank;
VC2 traffic in the two-mesh setup (test_fabric_vc2_at_least_2x2_mesh.yaml): 10/10 per rank on two ERISCs and on one.
So the two-ERISC VC1+VC2 program (26,076 B) loads. Rates (cross-job, against job 315's v4rxc2b = v4rxc7 images at
19/9/16/5/4 without VC2; not an isolating comparison: VC2's idle cost, the credit preset and the slot change mix):
VC1 1-hop and burst 140.9 -> 181.8, multicast 149.2 -> 176.0, 2-hop 240.6 / 259.6 -> 287.9 / 311.6, merge
140.2 / 144.1 -> 156.5 / 166.2.

## 30. One visit for the worker channels; the connection semaphore at a compile-time address (2026-11-01, user)

From §29 (E0's idle VC2 visit binds; its connection check's pointer load, semaphore load and stalled compare):
- `v4rxc7w` = v4rxc7 with `visit_workers`: VC0's and VC2's worker channels in one visit (send; one fenced load group
  of both workers' counters and connection semaphores, scalars only; settle each). VC2's visit moves from the end of the
  pass to right after VC0's (before the forwarding visits). The connection is still checked every pass.
- `v4rxc7c` = v4rxc7 with each worker's connection semaphore read at a compile-time address (`Channel<CH>::
  connection_semaphore`, new named args SENDER_CH_<i>_CONNECTION_SEMAPHORE_ADDR from the host builder; `Worker` loses
  its pointer; a watcher-build ASSERT checks the runtime arg matches). Offline builds of older captures get the arg from
  offline_build.py (connection info + 96 B, the host's layout).
- `v4rxc7wc` = both.
Expected before the builds: v4rxc7w byte-identical to v4rxc7 on every image without a VC2 worker (same operations in the
same order) and E1 images everywhere; **WRONG (offline, below): +2 instructions on the sender loop of most configs without VC2
(2D, 1D, VC1; torus and ring identical), and +156 / +33 stack accesses on 2D+VC2's.** E0/one-ERISC VC2 images changed. v4rxc7c and wc: E1 images identical; E0 and
one-ERISC images change on every config (VC0's pointer becomes a constant; one fewer loop-carried value).
Device (hypothesis, removed work: none for w, one pointer load per worker per pass for c): VC2 on, 2D E/W write
152.7 -> 143 to 152 (w: VC2's semaphore stall overlapped with VC0's, at most its ~10 stall cycles), c: -0 to -3;
VC2 off flat (c: within +-1).
Offline (coverage -O3, 2,086 configs each; the 66 VC1+VC2 configs fail as for v4rxc7, whose gate excludes them).
Sender loop (two ERISCs: sender::run's loop region; rxcmp.py on the sender function), instructions / stack accesses:
- v4rxc7w vs v4rxc7: 2D, 1D, VC1, Galaxy 2D +2 / 0; torus, ring identical; 2D+VC2 +156 / +33 (fn 801 -> 955
  instructions, 182 -> 214 sp accesses). By inline frame (2D+VC2 E0): worker_settle<0> +96 / +15 against
  visit_worker<0>'s -30 / -9, worker_reads<0> +33 / +11, worker_reads<4> +36 / +12. Hypothesis: the two workers'
  booleans and loaded values across the group do not fit the registers (loaded values stored to the stack, the
  footgun) and GCC duplicates the settle code per path.
- v4rxc7c vs v4rxc7 (compile-time semaphore address alone): 2D+VC2 -14 / -22, VC1 -4 / -6, 2D +2 / -2, 1D +4 / 0,
  torus 0 / -2; E1 images identical.
- v4rxc7wc vs v4rxc7c: as w vs the lead (2D+VC2 +155 / +33).
So the joint visit, as written, costs the VC2 sender loop more than it could overlap; the compile-time address removes
VC2's pointer load and 22 of its loop's stack accesses. Not yet run on the device.
Device (job 324, row_ab_runs.sh tags jv2on / jv2off, slots 17/14/16 (+ 2/4) logged; two repetitions within 0.4):

| row | VC2 off (all four) | VC2 on: v4rxc7 | v4rxc7c | v4rxc7w | v4rxc7wc |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2D E/W write 4352 | 126.9-127.0 | 152.7 | 147.9 (-4.8) | 151.2 (-1.5) | 144.9 (-7.8) |
| TorusY1HopNS write | 149.2 | 181.5 | 180.6 (-0.9) | 193.0 (+11.5) | 183.2 (+1.7) |

The compile-time address recovers 4.8 / 0.9 of idle VC2's 25.8 / 32.3; the joint visit as written loses on torus
(+11.5 alone, +2.6 on top of the address) and gains 3.0 on 2D on top of the address. Idle VC2 still costs 18-34
cycles/packet. Next (user): a design agent on idle VC2's E0 cost with readability as the anchor and VC2's connection
checked every pass; meanwhile an A/B of checking it once per round (main's behavior), to bound what the check costs.

## 31. What VC2's per-pass connection check costs: per-round probe (2026-11-01; job 325, VC2 on, slots logged)

`v4rxv2pr` (probe, attribution only) = v4rxc7c with VC2's worker connection read and compared once per round (after
the round's passes, main's behavior) instead of in its visit; images without VC2 identical to v4rxc7c; VC2 sender loop
-2 instructions, +3 stack accesses. Same job: v4rxc7, v4rxc7c, v4rxv2pr, v4rxv2p1 (E0 never visits VC2, the floor);
two repetitions within 0.2. Cycles/packet:

| row | VC2 off (job 324) | v4rxc7 | v4rxc7c | v4rxv2pr (per round) | v4rxv2p1 (no VC2 visit) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2D E/W write 4352 | 126.9 | 152.7 | 148.0 | 134.6 | 129.3 |
| TorusY1HopNS write | 149.2 | 181.5 | 180.6 | 166.9 | 167.2 |

- VC2's per-pass connection check (with the compile-time address) costs 13.4 (2D) / 13.7 (torus) cycles/packet on E0
  (A/B: one change, E0 only). The rest of VC2's idle visit: about 5 on 2D, nothing measurable on torus.
- Torus with no VC2 visit still trails VC2 off by ~18 (VC0's worker credit without its preset, 5.6 in §29; the rest
  not attributed).
- Per round breaks the user's rule (every worker connection checked every pass); this is a bound, not a candidate.

## 32. Idle VC2: design agent's candidates (2026-11-01; offline), and a conflict with V4_STATE.md §12

**Conflict (flagged; being settled by job 327's VC1 split):** v4rxc7c / i1 / i3 read the worker connection semaphore at
a compile-time address. V4_STATE.md §12 (job 231, isolating A/B v4pl3n -> v4pl3p) found that the compile-time address
cost VC1 1-hop and burst rows ~2.2 cycles/packet (likely its per-pass `lui`, hypothesis) and the runtime pointer was
kept for that. §30 did not check this before reintroducing it.

Candidates (variants from v4rxc7c, only router_sender.hpp differs; full coverage 2,086 of 2,282 as c7c; E1 images
identical to c7c):
- `v4rxv2i1`: the worker channels' connection semaphores read in the pass's free-slots load group
  (`channels_with_packets<MASK, WORKERS>(semaphores)`), then `check_worker_connections` (each worker, every pass, out of
  line on a change), before the visits. Sender loop vs c7c (instructions / stack accesses): 2D+VC2 -16 / +12 (GCC now
  spills far.vc0 next_slot: +1 load +1 store per VC0 send), 2D -7 / -3, torus -2 / 0, VC1 -6 / -2, 1D -6 / -2, ring +3 /
  +2. Expected (hypothesis, if the group hides the load): VC2 on near v4rxv2pr (134-137 / 166-169).
- `v4rxv2i3`: i1 + VC2's worker credit off the write command buffer (`noc_semaphore_set_remote`, non-posted) so VC0's
  credit keeps its preset (VC2's connection event also uses that buffer, so the preset is restored after it). Sender
  loop vs c7c: 2D+VC2 +15 / +18 (cold restore path +13 / +6), 2D -5 / -2, torus 0 / +1, VC1 -4 / -1. Expected i3 - i1 ~
  p2 (-2.0 / -5.6).
- Rejected offline: `v4rxv2i2` (one connected bit-word for both workers): +30 / +2 vs i1. Not built: VC2's sender state
  in memory (targets the ~5 on 2D only; ranked last).
Device: jobs 326 (VC2 on: c7c, pr, i1, i3; VC2 off: c7, c7c, i1, i3 on 2D write, torus write, 1D 1-hop, ring 1-hop)
and 327 (VC1 split, c7, c7c, i1, i3, two passes in swapped order).
Device (job 326, tags i1v2on / i1v2off; two repetitions within 0.1). Cycles/packet:

| row | VC2 on: v4rxc7c | v4rxv2i1 | v4rxv2i3 | v4rxv2pr (per round) | VC2 off: v4rxc7 | v4rxc7c | i1 | i3 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2D E/W write 4352 | 147.9 | 136.3 | 134.6 | 134.6 | 126.9 | 126.9 | 126.9 | 126.8 |
| TorusY1HopNS write | 180.6 | 171.6 | 166.9 | 166.9 | 149.2 | 149.2 | 149.2 | 149.2 |
| 1D 1-hop (lin1) | | | | | 124.4 | 124.4 | 124.4 | 124.4 |
| ring 1-hop | | | | | 140.8 | 140.8 | 140.9 | 140.9 |

- i1 (semaphores in the pass group, checked every pass): -11.6 / -9.0 against c7c; within 1.7 of the per-round bound
  on 2D. So most of the per-pass check's cost was the load's exposed latency, and the group hides it (A/B).
- i3 (i1 + VC0's credit keeps its preset): equal to the per-round bound on both rows; i3 - i1 = -1.7 / -4.7 (p2's
  -2.0 / -5.6 in §29). Idle VC2 now costs 7.7 (2D) / 17.7 (torus) against VC2 off, down from 25.8 / 32.3.
- VC2 off: all flat. VC1 split (the §12 conflict): job 327.
VC1 split (job 327, census_vc1_runs.sh ci1 / ci1b in swapped router order, data-checked 8/8 logs, passes within 0.7),
delta against v4rxc7, cycles/packet: v4rxc7c VC1 1-hop and burst rows +2.1 to +2.4, 2-hop +0.5 / +0.8, suite mean
+0.90%, inter-mesh +0.11%; v4rxv2i1 1-hop/burst +1.2 to +1.4, 2-hop -1.2 / -1.4, merge -1.1 / 0, suite +0.43%, inter
-0.24%; v4rxv2i3 1-hop/burst +1.1 to +1.3, 2-hop -1.3 / -1.5, suite +0.35%, inter -0.23%. So V4_STATE.md §12 holds
(the compile-time address costs VC1 1-hop ~2.2 on c7c's shape); i1/i3's grouping halves it to ~1.2 but does not remove
it. Mechanism (§12's hypothesis, VC0's semaphore address rebuilt with a `lui` every pass on the idle-worker path): not
re-checked here.

## 33. Where VC1 1-hop's +1.1 lands (2026-11-01; job 328, PC samples, VC1 split, PcsVc1EW4352)

vc1_pcs_runs.sh, tags c7v1 / c7cv1 / i3v1. Rates reproduce: v4rxc7 140.9, v4rxc7c 143.1, v4rxv2i3 142.0. The busiest
E0 (chip 1 chan 5) is a VC0 worker sending every pass. c7cv1's samples are not usable: about 60% of its E0 samples sit
in a startup/teardown loop (tt_fabric_utils.h:86; E1 in router_init.hpp:66), so its split is contaminated.
E0, lead -> i3, cycles/packet by inline frame (design_tools/byframe.py): visit_worker<0> 16.1 -> 8.2 (the lead's
connection compare stalled on the semaphore load, `beq` router_sender.hpp:463, 5.9, is gone); the pass's group grows:
holds_packet<1> 5.9 -> 10.1 (a new 5.2 on its `snez`), read_connection_semaphores<1> 0 -> 2.7, check_connection<0>
0 -> 1.3, holds_packet<4> +1.4; net +1.3. Hypothesis: on a row whose worker sends every pass, the group's values are
used right after the group, so the semaphore's L1 load added to it lengthens the group's wait instead of overlapping
work: the stall moved (from the visit's compare to the group's first uses) rather than being hidden. On the VC2-on rows
the same grouping recovered 11.6 / 9.0 (§32). Also from these samples: the lead's per-pass VC0 connection check costs
~6 cycles/packet of stall on this row.

## 34. `v4rxv2i3` is the temporary lead (2026-11-01, user)

Validation (job 329, variant_val_runs.sh v4rxv2i3, data-checked, hang capture armed): 13/13, failed none. Two ERISCs:
mesh 96, 1D 143, ring 19, Y-torus 12, mux 19, VC2 10, NeighborExchange 7, trimming gtests 63/63; one ERISC: mesh 96,
ring 19, VC2 10, NeighborExchange 7, trimming 63/63. VC1 split data-checked in job 327. It needs the host's
SENDER_CH_<i>_CONNECTION_SEMAPHORE_ADDR named args (erisc_datamover_builder.cpp; the host library has them).
Numbers: §32 (VC2 on 134.6 / 166.9, VC2 off flat), §32-§33 (VC1 1-hop +1.1 to +1.3, a moved stall). Next: the sender
readability design (agent) and the receiver beautification proposal (agent), both from v4rxv2i3; the non-blocking flush
design is parked in FLUSH_DEFER_DESIGN.md for the feature pass after.

## 35. VC2 traffic rows (2026-11-01; job 333, vc2rows_runs.sh c, slots logged)

New rows (fabric_exp/pcs3): vc2ew4352.yaml (VC2 alone, chip 0 -> its E neighbour, 4352 B, benchmark mode) and
vc02ew4352.yaml (VC0 + VC2 senders on chip 0 to the same neighbour, one link). Benchmark mode with a VC2 sender needed a
worker-API fix: `fabric_detail::update_credits_and_slots` was declared only for VC0's `WorkerToFabricEdmSender`; now
templated on the sender type (edm_fabric_worker_adapters.hpp). Splits: d = study row (VC0 17/14/16, VC2 2/4); v8 =
TT_FABRIC_EXP_SLOTS_VC2=17,10,16,8,10 (VC0 17/10/16, VC2 8/10). Cycles/packet, two repetitions within 0.1:

| row | v4rxc7 d | v4rxc7 v8 | v4rxv2i3 d | v4rxv2i3 v8 |
| --- | ---: | ---: | ---: | ---: |
| VC0 alone (ew4352, VC2 on) | 152.7 | 152.4 | 134.6 | 134.8 |
| VC2 alone (vc2ew4352) | 881.8 | 243.4 | 858.1 | 240.6 |

- VC2's own rate is set by its worker window: 2 slots ~880, 8 slots ~240 (still about twice VC0's; by the window
  model it needs ~16-18 worker slots to reach VC0's 1-hop rate). Placement effects on VC2 traffic cannot show until VC2
  has that depth.
- v4rxv2i3 does not slow VC2 traffic (its VC2 credit is now a non-posted `noc_semaphore_set_remote`): -23.7 at 2 slots,
  -2.8 at 8.
- VC0 + VC2 on one link: not measured. The bandwidth profiler throws when two cores send on the same (device,
  direction, link) (tt_fabric_test_bandwidth_profiler.cpp:350-357).

## 36. VC0 + VC2 on one link (2026-11-01; job 340, vc2rows_runs.sh e, vc2rows_report.py)

Test infra: the bandwidth profiler no longer throws on two senders sharing a (device, direction, link); the link gets
the longest sender's cycles and each sender core's packets and cycles are logged with its VC
(tt_fabric_test_bandwidth_profiler.cpp). The concurrent rows run with `sync: false`: with sync, the end-of-run syncs give
every sender the slowest one's cycles (job 338, killed). vc02ew4352 (4M each): VC0 finishes first, so VC0's rate is
contended; vc02bew4352 (VC0 40M, VC2 2M): VC2 finishes first, so VC2's rate is contended. Two repetitions within 0.5.
Cycles/packet (alone: job 333):

| sender | v4rxc7 d (VC2 2/4) | v4rxc7 v8 (VC2 8/10) | v4rxv2i3 d | v4rxv2i3 v8 |
| --- | ---: | ---: | ---: | ---: |
| VC0 alone | 152.7 | 152.4 | 134.6 | 134.8 |
| VC0, VC2 active | 175.7 | 217.3 | 163.6 | 190.7 |
| VC2 alone | 881.8 | 243.4 | 858.1 | 240.6 |
| VC2, VC0 active | 1043.0 | 307.2 | 989.5 | 368.7 |
| both, combined (estimate from the two rows) | 150.4 | 127.3 | 140.4 | 125.7 |

- With both VCs busy each slows (v4rxv2i3 v8: VC0 +56, VC2 +128) but the link carries more in total (~126 against
  VC0 alone's 135): E0 sends from both worker channels per pass.
- v4rxv2i3 shifts the share toward VC0 against v4rxc7 at v8 (VC0 190.7 vs 217.3, VC2 368.7 vs 307.2; combined about
  equal). Mechanism not measured (candidates: VC2's credit now a non-posted write; the connection checks moved to the
  pass's start). VC2 alone is unchanged (240.6 vs 243.4), so it shows only under contention.
- VC2's depth sets its rate alone and under contention; placement questions (sender stage 2) should be judged on
  these rows with VC2 at a realistic depth.

## 37. Sender restructure stage 1 (2026-11-01; design agent; validation job 346)

`v4tx1` (shape only, from v4rxv2i3): `Vc<V>` / `Channel<CH>` features (worker and lead roles, per-VC
`reads_at_pass_start`), one generic `Each<>` container for register channels, far rings, records and connections,
`pass()` = group read, `check_connections<worker_channels>`, `visit_vc<0..2>`; `return_slots<CH>` by role; placement in
one place (today's rule). Changed: router_sender.hpp, router_single.hpp (sender half), fabric_erisc_router_v4.cpp
(sender init). `v4tx1b` = v4tx1 + every worker's credit source its own connection-info read counter (VC2 already did;
the shared scratch word is gone).
Static (-O3): sender loop +0 / +0 on every config (txcmp); 878 images byte-identical (every E1); the rest commuted
compare operands, renamed registers, one per-round load one slot later, one init store dropped; E0 -4 B. Torus (Y) +
VC2, the case design flagged as missing (coverage cases bh_torus_vc2[_1erisc], captured with v4rxv2i3 in the v2 slot,
job 345: 80 + 40 configs): v4tx1 renames registers only on E0, E1 identical; one ERISC equal counts; v4tx1b equal
counts. Validation of v4tx1b (covers v4tx1's code): 13/13 (job 346). Stage 2 (placement) waits for the user (host-set
masks) and the VC2 traffic rows (§35-§36).

## 38. `slots_due` A/B (2026-11-01; jobs 347-348; v4rxsd = v4rxbt17 + slots_due only, against v4rxv2i3)

Static (receiver loop vs v4rxv2i3): 1D -11/-12 instructions, 2D -8 to -14, VC1 -3 to -16, ring rename only, 2D+VC2
mixed. Device, cycles/packet, two reps each (row_ab_report.py sdk / sdkv2; census_report.py vc1 vc1i, passes sd and sdb
in reversed router order):

| rows | v4rxv2i3 | v4rxsd |
| --- | --- | --- |
| ew4352 / ns4352 / lin1 / lin3 / ring1 / torusns | 126.9 / 127.2 / 124.4 / 124.4 / 140.9 / 149.2 | 126.8 / 127.3 / 124.4 / 124.4 / 140.9 / 149.2 |
| ewscatter | 139.7 | 140.1 (both reps) |
| VC2 on: ew4352 / torusns | 134.6 / 166.9 | 134.6 / 167.0 |
| VC1 1-hop and bursts (8 rows) | 142.0-142.2 | equal within 0.1 |
| VC1Mesh2HopMerge 16 / 4352 | 139.5 / 144.3 | 139.0 / 143.8 |
| VC1Mesh2Hop 16 / 4352 | 239.5 / 258.1 | 239.7 / 258.5 |
| VC1Inter 1_0 / 1_1 / 1_2 / 1_3 (16 B) | 208.3 / 346.1 / 346.8 / 375.6 | 209.0 / 347.8 / 347.6 / 376.0 |
| VC1Inter 1_0 / 1_1 / 1_2 / 1_3 (4352 B) | 209.1 / 397.1 / 395.7 / 414.0 | 209.9 / 397.8 / 396.0 / 414.0 |
| suite means vc1 / vc1i | 159.3 / 336.6 | 159.3 / 337.3 |

Every delta repeats in both passes (pass spread <= 0.3). The instruction savings do not show as rate: flat on the
E0-bound rows (V4_PROFILE §2 on v4trm; not re-measured on this lead), within +-0.5 on scatter and the VC1 2-hop rows,
and +0.4 to +1.7 on the inter-mesh rows. Why the inter-mesh rows lose is not known (no isolating A/B); a moved stall
or layout effect is a hypothesis only. Adoption is a readability call, within the 1-2 cycle tolerance.
