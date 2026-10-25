# Router v3l: a lean, specialized E0

v3l redesigns the program ERISC 0 runs in two-ERISC mode (the sending ERISC, "E0") and leaves everything else as v3a.
E1 compiles to v3a's instruction stream on every buildable config at -O3, -O2 and -Os (also under both probes), so a
hardware difference against v3a is E0's. Variant: `fabric_exp/variants/v3l/` (v3a's layout; only
`fabric_erisc_router_v3.cpp` differs from v3a).

## 1. The design in a page

**Scope.** `e0::runs` selects the new program: `MY_ERISC_ID == 0`, two ERISCs, no receiver channel on this ERISC,
Blackhole, multi-TXQ (counter credits), no speedy path, no VC2, no data cache, no spin-wait sends, at most four VC0
senders, channel 0 is the only worker-type channel and every other visited channel is router-to-router, and the
program visits exactly the channels the generic loop visits (checked at compile time). In the logged configs that is
all 168 non-speedy two-ERISC E0s (2D, 2D VC1, 1D line, 1D ring). Everything else keeps v3a's code: speedy E0, E1,
single-ERISC, and the production paths no logged config has (VC2, express/fifth VC0 sender, stream-register
credits, Wormhole, data cache).

**Structure.** `kernel_main`'s init is v3a's. After it, E0 calls `e0::run_main_loop`, a `noinline` function that owns
its state. Router commands and the outer loop (termination, telemetry, heartbeat, idle context switch) are v3a's E0
code, around a 32-pass inner loop. One pass:

1. VC0's free-slots registers, channels 0-3, as one fenced group (v3a's `read_grouped_free_slots`).
2. Worker channel 0, inline and specialized: send if it has a packet and the far ring has room; take completions;
   return credits to the worker; poll the connection semaphore, calling `worker_connection_event` (`noinline`) only
   when the semaphore differs from the value it holds while nothing changes.
3. VC0's forwarding channels: **one test** (the AND of their free-slots values against the shared power-of-two slot
   count, or an OR of masked differences otherwise, plus one in-flight word) says whether any has work. Only then is
   an unsent mask computed and each channel with work visited, lowest first, by `visit_forwarding_channel`, one
   `noinline` body that every forwarding channel of the VC shares and that takes the channel index at run time.
4. VC1, when serviced: its 3-4 free-slots registers as a second fenced group, then the same test and the same body
   (instantiated for VC1's far ring).

**State.**

| Held across passes | Where |
| --- | --- |
| Per VC: far-ring free count, far slot pointer | locals of the loop (passed to and returned from that VC's forwarding visit by value) |
| Worker: slot, sent, completed (and acked under first-level ack), expected semaphore value, semaphore pointer | locals of the loop |
| Forwarding channels with packets in flight (one bit each), "send attempted this round" | `ForwardingState` in the loop's frame |
| Per forwarding channel: slot, sent, completed, acked, first/last slot, upstream credit address | `ForwardingState::channels[ch]`, indexed by channel at run time; read only by that channel's visit |
| Far-ring base/last/stride, worker slot bounds, stream-register and counter addresses | compile-time constants (`FarRing<RX>`, `ChannelSlots<CH>`, `free_slots_update_addr`, `completion_counter_addr`) |

v3a's ~58 held values (ROUTER_MAP §4) are ~23 here in the 2D -O3 build: ~10 in registers, ~13 in stack slots,
of which the hot pass reads two (the semaphore pointer and the in-flight word) when idle.

**Per-pass and per-packet work against v3a** (2D E/W E0, three forwarding channels idle):

| | v3a | v3l |
| --- | --- | --- |
| Idle forwarding channels, per pass | 3 bit tests + 3 sent/processed compares, 11 instructions, 4 taken branches (~16 cycles, §73) | one AND of the three values, one in-flight word, no taken branch |
| Worker completion test, per pass | sent and processed from the stack | both in registers |
| Worker credit counter | a separate read counter, kept equal to processed (stack load and store per credit) | the completed count itself (they are always equal) |
| Worker connection poll, per pass | `!established` test, then pointer from stack, L1 load, compare with close-request | pointer from stack, L1 load, compare with the expected value (one compare covers open and close) |
| `did_something` | global byte store per send attempt | `ForwardingState::attempted` byte store, same points |
| Far-ring and slot geometry | struct fields on the stack (base, last, stride ×5) | compile-time constants; GCC still hoists four of them into stack slots in the 2D -O3 build (see risks) |
| Forwarding channel with work | inline per channel; slot, bounds, counts mostly on the stack | out-of-line call: same send, completion and credit code; record loads instead of stack loads; plus the call (~12 instructions with arguments and two dead stores of the returned far ring) |
| VC1 channels, per pass | 3 ungrouped free-slots reads used at once, 3 completion tests | one 3-read group, one test |
| Contiguous hot loop (2D E/W, -O3) | 226 instructions, 33 stack loads | 92 instructions, 8 stack loads (2 on the idle path) |

**Removed:** the per-channel copies of the forwarding send (one shared body per VC instead); the separate worker read
counter; the per-pass idle tests of idle forwarding channels (one test instead); VC1's ungrouped per-channel reads;
the static-connection wait's per-channel inlined poll loops (one `noinline` poll loop, `e0::wait_for_connect_request`:
3,124 → 756 B on VC1 E0).

**The brief's directions.**
- *One test per pass for the forwarding channels, bookkeeping only on a channel with work*: adopted. The in-flight
  bit is the only per-channel bookkeeping outside a visit, and only a visit changes it.
- *Derive constants instead of holding them*: adopted for state (nothing compile-time-known is a field any more;
  stream and counter addresses are one base plus the channel index). GCC still materializes some of these constants
  once and spills them (risks).
- *Shorten the worker visit's load chains by design*: partly. The worker's per-pass compare values (sent, completed,
  expected) are in registers, and the one compare per pass replaces two. In the 2D -O3 build the worker slot and the
  semaphore pointer are still stack slots, so the payload-size and teardown-flag loads each still wait behind a stack
  load, as in v3a (in the 1D build the pointer is in a register). Removing the semaphore chain by construction needs
  a host change (proposal below). Grouping the semaphore load into the pre-send group was not adopted: §49 tried it in
  the post-send group with a mixed result, and §17's two-L1-read groups have irregular costs; it is the first
  candidate for iteration 2.
- *Per-channel code off the hot path so code size does not scale with channels*: adopted. Forwarding visits,
  connection transitions and the static-connection wait are out of line; the VC1 E0 fits at -O3.
- *Few live values, so edits, the hook and opt levels move it little*: the hot loop already contains calls
  (forwarding visits, connection events), so the hook's call adds no new constraint. With the hook the hot loop
  differs from the base only in register names plus the hook's own compare (§4).

**Ledger ideas.** Shared runtime forwarding step (LOST-CONTEXT, §35/§36/§81): used, in the form the ledger asks for:
idle tests are register-only (no per-channel stack loads, s80's loss), the body is a function rather than a loop the
compiler can peel or hoist into, and the worker's counts stay in registers. A pure inline channel loop was tried first
(offline): GCC hoisted the body's constants into the pass loop and spilled the worker's sent, slot and semaphore
pointer, and the hook moved more worker state to the stack. A `noinline` service handling all channels of a VC in one
call was also tried: its internal loop hoisted ~12 constants and saved all 12 callee-saved registers per call (§36's
"each entry spills"). One call per channel with work, with no loop inside the body, keeps the body's prologue to a
stack adjustment. Smaller sender-channel state (bit 24/s77t/s78l): used for E0 (compile-time bounds), without
relying on scalarization: forwarding records are memory by construction (runtime index), the worker's state is a
handful of scalars. Forwarding group gate (MIXED, §37): used; the forwarding-router cost is now a call instead of a
count update. Cold connection helpers (LOST-CONTEXT, §79): used; the transition is a call whose result replaces the
expected value, and the hot path is not duplicated. Loop function boundary (§78): the E0 loop is its own function
that owns its state; the speedy path, which lost 6-10 out of line, keeps v3a's inline loop. Summary doorbell, batched
pointer updates, multi-packet visits: not used (protocol or producer changes).

**Hardware ordering.** v3a's fences: the grouped free-slots read is v3a's function, and `send_packet`'s pre-send group
(payload size, TXQ CMD, STATUS) and post-send group (CMD, STATUS, completion counter) are
`sender_channel_try_send`'s, fence for fence. VC1's group uses the same fenced pattern. No new primitive: these
groups have hardware parity with v1's asm (§48) and the brief allows them.

## 2. Semantic equivalence with v3a

- **Channels and order.** `e0::shape_supported` checks at compile time that the program visits exactly the channels
  `run_fabric_edm_main_loop` visits, on the same VC; otherwise the build uses the generic loop. Within a pass:
  worker, then VC0 forwarding channels in increasing index, then VC1's, as v3a.
- **Worker send** (`send_packet`) is `sender_channel_try_send`'s overlapped BH form line for line: same fences, same
  register reads and writes in the same order, `src_ch_id` stamp, busy-queue yield before any state change, far
  pointer and slot advance (same values: v3a's runtime first/last/stride are initialized from the same allocation
  constants and `channel_buffer_size`), free count −1, free-slots register +1, spin, packets-received update, sent +1,
  completion delta.
- **Completions and credits.** Same conditions: the post-send delta on a send, otherwise an L1 read only when
  sent ≠ completed; far free count and completed += delta; credit. The worker credit value is v3a's read counter, which
  starts at 0 and is incremented by exactly the deltas `completed` is (acks under first-level ack), so it always equals
  `completed` (`acked`); the NoC write, its flags, source address and destination are v3a's
  `notify_worker_of_read_counter_update`, and the disconnected path writes the same location-info word. Forwarding
  credits are v3a's `notify_persistent_connection_of_free_space` call with the same cached address. First-level ack
  and bubble flow control follow v3a per channel (compile-time masks).
- **The forwarding test.** v3a returns early from a forwarding visit when every slot is free and sent = processed;
  any other visit does the same work in both. v3l skips a channel under the same condition: the in-flight bit is
  recomputed at the end of every visit as sent ≠ completed (or ≠ acked), and only a visit changes those counts. Free
  counts never exceed the slot count, so the power-of-two bit test and the AND across channels are exact.
- **Connection protocol.** v3a reads the semaphore once per pass (`!established` short-circuits, or the teardown
  test). v3l reads it once per pass and calls the transition only when it differs from `unused` (no worker) or `open`
  (connected); the transition re-runs v3a's `check_worker_connections` logic on that value (establish on open or
  close request, tear down on close request, nothing otherwise) through the same interface methods, after setting the
  interface's read counter to the credits returned. Values the generic loop ignores also do nothing here.
- **Idle context switch.** `attempted` is set where v3a sets `did_something` (after the pre-send group, before the
  busy test) and reset per outer iteration.
- **Init.** Identical except `e0::wait_for_static_connections`: same channels, same spin condition, same context
  switch every 100,000 polls, same address caching; one shared poll loop instead of one per channel.
- **Telemetry.** BW and perf telemetry and channel-trimming capture are called at v3a's points. No logged config has
  them on; they were compile-checked offline (§4).

## 3. Proposals (no host change made)

- **Worker connection semaphore address as a compile-time arg.** It is fixed per router but arrives as a runtime arg,
  so the per-pass teardown check is pointer-then-flag. As a CT arg the check is one L1 load from a constant address,
  by construction.
- **Smaller E0 init.** E0 constructs the downstream adapters it never opens (E1 repeats the same writes in the same
  init phase). Dropping them would add size margin for VC1 at -O3; left as v3a for now.

## 4. Offline results

Builds (`offline_build.py`, -Werror): **-O3 536/536, -O2 536/536, -Os 500/524** buildable. The -Os database has 535
configs; 11 have only a link command and 24 compile commands are corrupted in the logs (interleaved MPI output), so
v3a fails on the same 24 (identical list) and the tool cannot build the other 11 for any variant.

`codegen_diff.py` against v3a (same opt level):

| Level | E0 changed | E0 instructions med/max | E1 changed | Hot E0 (E/W) |
| --- | --- | --- | --- | --- |
| -O3 | 168/208 (the 40 speedy E0s are identical) | −138 / −901 | **0/328** | 2523 → 2409 |
| -O2 | 168/208 | −124 / −535 | **0/328** | 2210 → 2085 |
| -Os | 151/191 | −138 / −459 | **0/309** | 2130 → 1962 |

E1 is also identical to v3a's under both probes (hook 0/328, statefield 0/328, against v3a's probe builds).

**VC1 size** (largest E0 + largest E1 loadable bytes over the VC1 configs; limit 26,624):

| | -O3 | -O2 | -Os |
| --- | ---: | ---: | ---: |
| v3a | 16,996 + 12,708 = 29,704 | 13,920 + 12,232 = 26,152 | 12,340 + 11,660 = 24,000 |
| v3l | 13,388 + 12,708 = **26,096** (fits, 528 B spare) | 11,732 + 12,232 = **23,964** | 10,480 + 11,660 = 22,140 |

VC1 E0 at -O3: `kernel_main` (init) 6,416 B, `run_main_loop` 2,228, two forwarding visits 544 each,
`wait_for_connect_request` 756, `worker_connection_event` 364.

**Stack** (E0, deepest `kernel_main` + `run_main_loop` + callee frame, over all E0 configs): -O3 1,424 B (v3a 1,408),
-O2 1,520 (v3a 1,632), -Os 1,488 (v3a 1,504). Median -O3 656 (v3a 768).

**Hot loop** (contiguous pass loop: instructions / stack loads; "+ hook" is the hook probe at -O3):

| E0 config | v3a -O3 | v3a -O2 | v3a -Os | v3a + hook | v3l -O3 | v3l -O2 | v3l -Os | v3l + hook |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2D E/W (EAST, hot) | 226/33 | 226/33 | 172/31 | 247/26 | 92/8 | 92/8 | 141/13 | 95/10 |
| 2D WEST | 230/33 | 229/32 | 172/31 | 251/26 | 92/8 | 92/8 | 141/13 | 95/10 |
| 2D VC1 | 510/81 | 512/82 | — | 563/64 | 132/9 | 132/9 | — | 135/11 |
| 1D line | 114/16 | 118/16 | 140/19 | 127/14 | 74/8 | 74/8 | 116/10 | 74/8 |
| 1D ring | 159/22 | 159/22 | 209/30 | 180/21 | 79/8 | 79/8 | 121/11 | 77/9 |

(v3l's stack loads include two reloads on the connection-event path, which only runs when the semaphore changes. In
the idle pass the 2D E0 reads two stack slots: the semaphore pointer and the in-flight word.)

-O3 and -O2 give the same v3l hot loop (at most 2 instructions differ, registers and stack offsets ignored). -Os lays the
worker send inline in the loop body (-O3/-O2 out of line), so its count includes the send. The VC1 -Os config is one
of the unbuildable ones.

**Probes** (`perturb.py v3l ... hook statefield`, -O3, all 536):
- hook: E0 +22 median / +41 max instructions (v3a: +44 / +75); hot E0 +5 instructions (v3a +57). On the 2D and VC1
  E0s the hot loop with the hook is the base loop's instruction sequence (registers and stack offsets aside) plus the
  hook's compare, whose two operands come from the stack. On the 1D E0 the hook costs its compare only (operands in
  registers; one `lui` leaves the loop). The ring E0 loop moves a little more (14 of 79 instructions: the hook's
  operands and two spill slots).
- statefield (adds a field to `ForwardingChannel`, 32 → 36 B): E0 +0 / +2 instructions; the hot loop is unchanged
  on the 2D, 1D and ring E0s (3 instructions differ on VC1).
- Markers: one DECL, two PRELOOP/LOOP pairs (v3a's loop for the other builds, e0's loop), STATE in
  `ForwardingChannel` (and v3a's in the shared sender channel struct).

**Compile coverage beyond the logged configs** (`patched_build.py` in the session scratchpad: named args overridden on
2D, VC1, 1D and ring E0 configs): BW telemetry with all stats (-O3, -O2, -Os), perf telemetry (low-resolution
bandwidth), both together, and channel-trimming capture all build.

Scratch (tools, builds, disassembly): `/tmp/claude-4004/-home-aagarwal-tt-metal/76f59f51-9416-4ea6-898e-740976fca7f2/scratchpad/v3l/`
(`final_O3|O2|Os`, `final_hook|statefield`, `base/` = v3a at all three levels; `loopstat.py`, `loopdiff.py`,
`size.py`, `stackdepth.py`, `patched_build.py`).

## 5. Open risks

1. **Forwarding routers pay a call per forwarding visit with work.** Beyond the send and completion work, a visit
   costs ~25 instructions (the 9-instruction unsent mask, channel selection, argument setup and call, record address,
   in-flight update, two dead stores), against ~4 per channel test in v3a's inline steps. 3-hop middle routers (E1-bound, E0
   ~12 behind) and multicast (both ERISCs bound) are where this shows. The idle worker visit on those routers is
   cheaper than v3a's, which offsets part of it; how much is a hardware question.
2. **Worker slot and semaphore pointer are still stack slots in the 2D -O3 build**, and GCC hoists four
   compile-time constants of the worker send (far last slot, stride, worker last slot, packets-received address) into
   stack slots: ~7 stack loads per worker send, close to v3a's 8. This follows from the calls in the loop (values
   live across a call need callee-saved registers). It is stable across -O3/-O2 and the hook, but not cheaper.
3. **One stack load per pass for the in-flight word**, and two dead stores per forwarding call (GCC homes the
   returned far-ring struct).
4. **-O3 VC1 fits with 528 B spare.** A larger E1 from another design could push the pair over; smaller E0 init is the
   lever (§3).
5. **GCC lays the worker send out of line at -O3/-O2** (two taken branches per sending pass; -Os puts it inline).
   No hints by rule.
6. **Paths without hardware coverage**: telemetry-on and trimming capture are compile-checked only (no logged
   config has them). The ring E0 (first-level ack, bubble flow control) changes in v3l and is covered by the 1D suite
   and 1D sanity.

## 6. Levers for iteration 2

- **Semaphore read in the pre-send group** (payload size, CMD, STATUS, semaphore: four reads) on sending passes, a
  lone read otherwise: removes one serialized load group from every sending pass, at the cost of a group with two L1
  reads (§17) on the send's critical path.
- **CT arg for the semaphore address** (§3): removes the pointer load by construction.
- **If forwarding rows regress:** visit VC0's forwarding channels inline (VC1's stay out of line). Costs the VC1 -O3
  fit (about +1 KB on VC1 E0) unless E0's init shrinks (§3).
- **Telemetry plumbing** costs nothing when off: `e0::Telemetry` is an empty struct and every use is `if constexpr`.

## 7. v3l2: inline specialized forwarding (after job 222)

Hardware on v3l (job 222, -O3): worker rows best so far (E/W 16 B 121.6 → 117.2, N/S 16 B 127.2 → 124.3, 1-hop E/W 4352
132.3 → 129.6, flushmix −1.2), forwarding rows much worse (forwarded 16 B 111.6 → 130.9, 3-hop 146.8 → 172.3, multicast
140.2 → 181.5, linear 3-hop +4.4). Variant: `fabric_exp/variants/v3l2/` (only `fabric_erisc_router_v3.cpp` differs
from v3a).

**What changed from v3l.**
- **Worker path:** unchanged (inline, specialized, same state, same per-pass work).
- **Forwarding:** the one combined test per VC stays (AND of the forwarding channels' free-slots values plus the
  in-flight word; no taken branch when idle). When it fails, each forwarding channel is tested on its own in channel
  order (its all-free bit, its in-flight bit) and a channel with work runs code specialized for it at compile time,
  inlined in the pass: slot bounds, free-slots and counter addresses and `src_ch_id` are constants; slot and counts
  come from the channel's record. No runtime channel index, no mask/ctz loop, no call per visit; the far ring stays
  in the loop's locals.
- **Out of line, off the pass's path:** the upstream-router credit write (one call per completion frame, 0.2-0.5 per
  packet), the TX-queue busy spin (never taken in practice), worker connection open/close and the static-connection
  wait (as v3l).
- **E0 init sets up the sender side only.** `kernel_main` skips, on this ERISC only, the receiver-side setup:
  downstream router adapters (VC0 and VC1, with their teardown-semaphore zeroing and local free-slots register init)
  and the receiver channel buffers (local and remote header zeroing). E0 never opens or uses them. E1 receives the
  same compile-time args (checked: identical positional CT args and channel counts for every E0/E1 pair of config
  types) and the same runtime args (`get_runtime_args()` is per router, not per RISC), and performs exactly these
  writes in the same init phase, before the next ERISC sync, so the state at that sync is unchanged. E1's code is
  untouched (`if constexpr` on `e0::runs`). Contract for a future E1 design: the receiving ERISC keeps doing its
  receiver-side init.

**Forwarding cost per forwarded packet** (2D E/W E0 `10733501891889981293` at -O3, a pass in which one forwarding
channel sends and nothing else happens; instruction counts from the disassembly):

| | v3a | v3l | v3l2 |
| --- | ---: | ---: | ---: |
| Group read + idle worker visit + connection poll | ~24 | ~18 | ~18 |
| Forwarding tests before the send | ch1, ch2 idle tests + ch3 test: ~8 | one test 6, unsent mask 10, channel select 9, call setup | one test 4, per-channel tests ~9 |
| Send + completion check + in-flight bookkeeping | ~48 (inline) | ~68 in the callee (entry/exit, record pointer, in-flight word in memory) | ~60 (inline) |
| After the visit | — | return, 2 dead stores, far ring copied back: ~7 | — |
| Total, and calls | ~80, no call | ~120, `jal` + `ret` | ~93, no call |

So v3l2 removes ~27 instructions and a call/return pair per forwarded packet from v3l (more on multicast, which in
v3a averages ~3 forwarding visits per packet, §72). It is ~13 instructions above v3a on a forwarding pass: the one
test before the per-channel tests, the in-flight bit tests and the in-flight update, against the cheaper idle worker
visit. On source routers (forwarding idle) the one test replaces v3a's ~16-cycle idle tests (§73).

**Builds (-Werror, all logged configs):** -O3 536/536, -O2 536/536, -Os 534/536. The two -Os failures are logged
commands the tool cannot parse (`<command-line>: ISO C++11 requires whitespace after the macro name`), and v3a fails
on the same two.

| codegen_diff vs v3a | E0 changed | E0 instructions med/max | E1 changed | Hot E0 (E/W) |
| --- | --- | --- | --- | --- |
| -O3 | 168/208 | −94 / −962 | **0/328** | 2523 → 2486 |
| -O2 | 168/208 | −51 / −390 | **0/328** | 2210 → 2212 |
| -Os | 167/207 | −167 / −506 | **0/327** | 2130 → 1973 |

E1 is identical to v3a's under both probes too (hook 0/328, statefield 0/328).

**VC1 E0 + E1** (largest of each over the VC1 configs, loadable bytes; limit 26,624; -Os over the 47 buildable):

| | -O3 | -O2 | -Os |
| --- | ---: | ---: | ---: |
| v3a | 16,996 + 12,708 = 29,704 | 13,920 + 12,232 = 26,152 | 12,344 + 11,660 = 24,004 |
| v3l | 13,388 + 12,708 = 26,096 | 11,732 + 12,232 = 23,964 | 10,484 + 11,660 = 22,144 |
| v3l2 | 13,128 + 12,708 = **25,836** (788 B spare) | 12,200 + 12,232 = **24,432** | 10,196 + 11,660 = **21,856** |

Where the bytes went on the largest VC1 E0 at -O3: six inline forwarding visits (~440 B each) replace v3l's two
out-of-line bodies (544 B each); `kernel_main` drops from 6,416 to 4,520 B without the receiver-side init. Outlining
the outer loop's pause and idle-context-switch hand-offs was tried and made E0 188 B larger (the inline copies share
code), so they stay inline.

**Stack:** deepest E0 (`kernel_main` + main loop) 800 B at -O3, 864 at -O2, 912 at -Os, against v3a's 1,408 / 1,632 /
1,504: E0 no longer builds the downstream adapters and receiver channel tables on its stack.

**Hot loop** (contiguous pass loop, instructions / stack loads; the loop now contains the inline forwarding visits):

| E0 config | -O3 | -O2 | -Os | -O3 + hook |
| --- | --- | --- | --- | --- |
| 2D E/W | 150/17 | 148/16 | 104/12 | 152/18 |
| 2D VC1 | 251/26 | 249/24 | 124/17 | 254/28 |
| 1D line | 66/7 | 66/7 | 193/20 | 66/9 |
| 1D ring | 77/8 | 78/8 | 182/23 | 77/9 |

-O2 against -O3: 2-10 instructions differ (registers and stack offsets ignored). Hook against base: the hook's
compare plus 3-16 instructions (two constants rematerialized instead of hoisted, spill-slot changes). Statefield:
no change on any lean E0 (the forwarding records are scalarized now; see risks).

**What the worker path pays for inline forwarding** (2D -O3): GCC now spends callee-saved registers on the forwarding
channels' completed counts and the in-flight word, and keeps the worker's sent count, slot and the far-ring pointer
in stack slots. A worker send reads 9 stack slots (v3l 7, v3a 8); the idle pass reads two (worker sent count and the
semaphore pointer; v3l: semaphore pointer and in-flight word). Expect worker rows somewhat behind v3l and still ahead
of v3a.

**Compile coverage:** BW + perf telemetry (-O3, -Os) and trimming capture build on 2D, VC1, 1D and ring E0s.

**Risks (v3l2).**
1. Forwarding passes are ~13 instructions above v3a's (above), plus whatever the layout does: GCC places the
   per-channel tests as several taken branches.
2. Worker path allocation is shared with the inline forwarding code (above); this is GCC's choice, not the source's,
   and it is what v3l avoided by keeping forwarding out of line.
3. Init ownership: E0 relies on E1 for the receiver-side writes (§7). True for v3a's E1 and any E1 that keeps its
   init; write it into the E1 design's contract.
4. VC1 -O3 margin 788 B.

## 8. The -Os loss is not in E0's worker path

Hardware (job 223): v3l E/W 16 B 117.2 / 120.4 / 146.1 at -O3 / -O2 / -Os, v3a 121.6 / 123.1 / 146.9; 1-hop E/W 4352
v3l 146.3 at -Os, v3a 147.2 (§82). v3l's and v3a's E0 code are very different, yet at -Os they land within a cycle of
each other on every E0-bound-at--O3 row, while their E1 code is identical. That points at E1 (or a shared floor)
binding at -Os, not at E0.

**E0 hot loop, -O3 against -Os** (v3l, config `10733501891889981293`, one pass in which the worker sends and nothing
else happens):

| | -O3 | -Os |
| --- | --- | --- |
| Instructions on the path | ~76 | ~71 |
| Stack loads on the path | 9 (worker slot ×2, far last, stride ×2, worker last, packets-received address, semaphore pointer, in-flight word) | 5 (worker last, packets-received address, semaphore pointer, expected value, in-flight word) |
| Taken branches | 3 (send path laid out of line, jump back, back-edge) | 1-2 (send inline, back-edge) |
| Calls on the path | none | none (the busy-spin call is on a never-taken path) |
| Inlining | everything inline | everything inline (`send_packet`, `eth_send_packet_bytes_unsafe`, gate lambdas) |
| Alignment | none either way: this toolchain emits no loop or label padding at -O3 (`.align 2`, 0 `nop`s in the ELF) | none |
| Constant materialization | far last, stride, worker last, packets-received address hoisted and spilled | rematerialized with `lui`/`addi`, worker slot in a register |

Only the outer loop changes in a way -Os is known for (`mul` by 144 for the launch-message index instead of
`sh3add`+`slli`), once per 32 passes. So the E0 worker path is the same or shorter at -Os, and the ~29 cycles per
packet come from elsewhere. The same holds for -O2: v3l's -O2 hot loop is the -O3 loop (2 instructions differ), yet
E/W 16 B is +3.2.

**Decisive test, ready to run** (no tree change; `p0c_runs.sh`-style ELF injection into an -O3 cache, since the JIT
hash does not include the opt level): mixed sets in
`/tmp/claude-4004/-home-aagarwal-tt-metal/76f59f51-9416-4ea6-898e-740976fca7f2/scratchpad/v3l/inject/`:
- `v3l_e0Os_e1O3/`, `v3l_e0O3_e1Os/` (v3l: E0 ELFs from one level, all `subordinate_active_erisc` ELFs from the other);
- `v3l2_e0Os_e1O3/`, `v3l2_e0O3_e1Os/` (the same for v3l2).
If E0 at -Os with E1 at -O3 runs the E/W 16 B row at ~117 and the reverse at ~146, the -Os loss is E1's, and the fix
belongs to the E1 design (v3l's E0 already has a level-independent hot path: same work at -O3, -O2 and -Os, no
calls, no lost inlining). I have not changed E1 (brief: E1 identical to v3a).

**Result (job 226, labels `p5inj_`, `p5e0os_`, `p5e1os_`):** confirmed. v3l with E0 at -Os and E1 at -O3: E/W 16 B
114.3 (all -O3: 117.2), forwarded 16 B 125.3 (130.9), 3-hop 166.5 (172.3), 1-hop E/W 4352 129.0 (129.6), mix mean
−2.0. E0 at -O3 with E1 at -Os is identical to all -Os (E/W 16 B 146.1, scatter 199.4). v3l's E0 is not -Os-sensitive
(slightly faster at -Os, as the shorter -Os path above suggests); the whole -Os penalty is E1's. Injection checks: no
router recompiles, all injected ELFs still in the cache after the runs.

## 9. VC1 hang of v3l2 + v3e1b (job 242, RESULTS §92): offline audit of E0's side

**Init ownership: complete in the combination.** For every VC1 and crossover config:
- **Same args on both ERISCs.** All 96 VC1 ELFs (E0 and E1, crossover and VC1-serviced) carry one positional compile-time argument vector: the channel allocations and downstream slot counts [4, 2, 0]. Runtime args are per router.
- **Each write v3l2's E0 dropped is done by v3e1b's E1,** with the same values, between the first and second ERISC syncs. That is the same phase as v3a, where both ERISCs did them.
  - Teardown semaphores: per connected downstream, zeroed by `e1_init_downstreams`, VC0 by dense index and VC1 at `NUM_DOWNSTREAM_SENDERS_VC0 + dense`.
  - Downstream local-copy registers: `init_ptr_val(vc0_local_free_slots_stream_id(compact), DOWNSTREAM_SENDER_NUM_BUFFERS_VC0)` and the VC1 equivalent. These are the values kernel_main's adapters use, crossover included.
  - Receiver buffer headers: zeroed by `local_receiver_channels.init`.
- **The one write neither ERISC does any more is still covered.** `remote_receiver_channels.init` (zeroing headers at the far-ring addresses, in local L1) was dropped by both: v3l2 counted it as receiver side, v3e1b as sender side. Its entries (397,824 × 8 slots, 433,408 × 4) are exactly the local receiver entries, with the same slot counts, stride and header size. E1's `local_receiver_channels.init` writes the same words.
- **Sender-side init dropped by v3e1b's E1 is done by v3l2's E0:** `edm_read_counter = 0` for every sender channel, sender header zeroing, worker interfaces.
- **The union of writes is v3a's.** With each write done once, v3a's old race between E1's `edm_read_counter = 0` and E0's later `= num_slots` on forwarding channels is gone.

**The binaries are not the problem.** The hung run's 48 E0 ELFs are text-identical to the passing v3l2 run's (`.tt-metal-cache-p1vc1-v3l2-auto`) and to my offline -Os build. Its 48 E1 ELFs are identical to the passing v3e1b run's (`-v3e1b-auto`). The hang comes from the pairing, through behaviour or timing, not from either program alone.

**v3l2's E0 VC1 path against v3a.**
- **Same sequence as v3a.**
  - Same send, completion, credit and far-ring arithmetic. Same channel order.
  - It sends only with far-ring room. The only waits are v3a's: the TX-queue busy spin after a data command, and the NoC command-buffer ready waits in credit writes.
  - No E0 wait depends on E1: the context-switch handshake is v3a's, and v3e1b's E1 has no in-pass wait on E0.
- **One predicate is not v3a's: the idle test for VC1 forwarding channels.**
  - v3a reads `(reg & 0x1FFFF) != 2`. v3l2 uses a bit test, `reg & 2` (likewise channel 1 on the ring E0: bit test where v3a compares).
  - They agree whenever the free-slots register is in [0, 2], the invariant §2 relies on.
  - If an upstream ever over-fills a 2-slot VC1 channel, the register wraps to 0x1FFFF or 0x1FFFE. Both have bit 1 set, so v3l2 treats the channel as idle and never serves it again: a hang. v3a's compare still sees a packet and keeps sending (wrongly, but without stopping).
  - v3u2 and v3u3 use the same bit test (`tx_idle_bit`), and v3u2 passed VC1. So this is a robustness hazard and a candidate, not a demonstrated cause; no over-fill source was identified offline.
- **Proposed fix (built as v3l2b, §10).** Use v3a's exact idle predicates: the bit test only where v3a uses it (VC0 forwarding with `skip_idle_per_channel && skip_idle_bit_test`), v3a's masked compare everywhere else, and the same form in the one combined test. v3l2 then behaves as v3a for every register value. The change is a few lines and costs nothing on VC0 configs.
- **Confirming on a reproduction:** read the stuck routers' VC1 sender free-slots registers (channels 4–6) and E1's VC1 local copies. A value above 2 means an overrun upstream; a value of 0–2 rules this mechanism out.

## 10. v3l2b (v3a's exact idle predicates) and v3l2be1b (v3l2b's E0 + v3e1b's E1)

**v3l2b is v3l2 with v3a's idle predicates and nothing else.** It lives in `fabric_exp/variants/v3l2b/`: 65 lines of the router .cpp change, and the headers are v3a's.
- `all_slots_free<CH>` now applies the generic loop's test for each channel:
  - **Grouped forwarding channels that v3a idle-skips:** calls `channel_all_slots_free` itself. That is the bit test `raw & N` when `skip_idle_bit_test`, else `(raw & 0x7FF) == N`.
  - **Other grouped channels:** `(raw & 0x7FF) == N`. The worker's test was already v3a's and is unchanged.
  - **Channels outside the grouped read (VC1):** `(raw & 0x1FFFF) == N`. This is `get_ptr_val`'s mask, `(1 << REMOTE_DEST_WORDS_FREE_WIDTH) - 1`.
- **The combined test uses the same predicate.** A VC's forwarding channels are always tested alike (`static_assert(tested_alike(CHANNELS))`): VC0's are all in the grouped read with one idle-skip decision, and VC1's are all outside it.
  - Bit-test form: the AND of the values with N, as in v3l2.
  - Compare form: `((raw_a ^ N_a) | (raw_b ^ N_b) | ...) & mask == 0`, which holds exactly when each `(raw & mask) == N` holds.
  - A first version OR-ed the per-channel `(raw & mask) ^ N` instead. It cost 68 B more on VC1 and 2 more instructions in the idle VC1 test, so it was replaced.
- **Effect by config,** from a .text comparison against v3l2 of every ELF at each level:

| E0 configs | vs v3l2 | What changes |
|---|---|---|
| 2D VC0-only (40), 1D line (80), VC1 built but unserviced (16) | byte-identical at -O3, -O2, -Os | nothing: v3a uses the bit test here too |
| 1D ring with deadlock avoidance (40) | changed | channel 1 now tests `(raw & 0x7FF) == 16`. v3a: first-level ack means no idle skip, so it compares. Idle pass +1 instruction (`xori; slli 21; bnez` against `andi; beqz`). Worker path: same instructions, two pairs reordered (-O3/-O2), identical at -Os. |
| 2D VC1-serviced (32) | changed | VC1 channels now test `(raw & 0x1FFFF) == 2`. See the next list. |

- **VC1-serviced configs in detail:**
  - The idle VC1 test is now 15 instructions with 1 branch; v3l2's was 12 with 2. Each per-channel test costs +1 instruction.
  - GCC keeps 0x1FFFF in a callee-saved register across the loop; v3a keeps it on the stack. As a knock-on, VC0 channel 2's in-flight completion check reloads two values from the stack: +2 loads, only on passes where channel 2 has packets in flight, at -O3 and -O2, none at -Os.
  - The worker path, the VC0 combined test and VC0 channels 1 and 3 are unchanged. The VC1 send blocks now fall through where v3l2 had some taken; this is layout only.
- **E1 is unchanged:** 0 of 328 differ from v3a at every level, as with v3l2.
- **Builds:** -O3 536/536, -O2 536/536, -Os 534/536. The 2 -Os failures are the malformed logged commands; v3a fails the same 2.
- **VC1 size, E0+E1** (max E0 + max E1 over the 48 VC1 configs): -O3 13,200 + 12,708 = 25,908 B (v3l2: 25,836; limit 26,624, so 716 B left); -O2 24,548 B; -Os 21,916 B.
- **Stability probes (-O3):** statefield leaves the main loop unchanged on the hot configs. Hook costs what it costs on v3l2 (one compare plus 1–2 loads). Outside the ring and VC1 configs, the probe builds are byte-identical to v3l2's.

**VC1 completion and credit return match v3a, confirmed by disassembly.** Checked on config 10243667986353281238, E0, -O3. v3a is inline in kernel_main; v3l2b is in `visit_forwarding_channel` and `credit_upstream_router`. Channel 4 is shown; channels 5 and 6 are the same.
- **After a send (both):**
  - The TXQ CMD dummy read, the STATUS read and the completion counter read (0x15BC0 = base + 4·4) are issued as one group.
  - Then `free_slots--`, local free-slots +1 (`sw 64 → 0xffb4e438`), the TXQ busy check, and the packets-sent register write (DEST 0xffb45438, DATA 64, CMD 4).
  - Then `sent++` and `c = counter − processed`.
- **No send (both):** if `sent != processed`, one L1 read gives `c = counter − processed`.
- **When c ≠ 0 (both):** `free_slots += c`, `processed = counter`, then one posted NoC inline write to the upstream address cached at init:
  - wait for the command buffer to be ready (0xffb21840);
  - AT_DATA = `c << 6`, CTRL = 0x208a;
  - ADDR LO, MID (`& 0x1000000f`), HI (`(hi >> 4) & 0xffffff`);
  - BE = `0xf << lo`, CMD = 1;
  - `noc_posted_writes_num_issued++`.
  
  v3l2b issues the same register writes in the same order with the same values. The only difference is that they sit in a noinline function called with the cached address and c.
- **Amount and timing are v3a's.** The credit equals the completion delta seen at that visit. There is one write per visit with a nonzero delta, issued in that visit after the free count is updated. There is no batching and no deferral, and the NoC and command buffer are the same.
- **Timing differences, unchanged since v3l2:**
  - The three VC1 free-slots registers are read as one group at the start of VC1's part of the pass; v3a reads each at its channel's visit.
  - A VC1 channel with nothing unsent and nothing in flight is not visited. v3a's visit does nothing in that case: no send, and `has_completions_pending` is false.
- **Conclusion:** the credit path does not explain the VC1 hang. The predicate was the one divergence from v3a found on E0's VC1 path, and v3l2b removes it.

**v3l2be1b = v3l2b's E0 + v3e1b's E1, an exact composition.**
- **Recipe:** `git merge-file -p v3l2b/fabric_erisc_router_v3.cpp v3a/fabric_erisc_router_v3.cpp v3e1b/fabric_erisc_router_v3.cpp`. It merges cleanly with no conflicts. The headers are v3a's, which are also v3l2b's and v3e1b's.
- **Cross-checks on the recipe:**
  - The same recipe on v3l2 reproduces v3l2e1b byte for byte.
  - v3l2be1b − v3l2e1b equals v3l2b − v3l2, line for line.
- **Per-config check:** compared PT_LOAD segments (address, sizes and bytes), keyed by MY_ERISC_ID, at -O3, -O2 and -Os (536, 536 and 534 configs). Every ERISC0 ELF equals v3l2b's and every ERISC1 ELF equals v3e1b's.
  - Against v3a: 168 ERISC0 ELFs differ (v3l2b's) and 128 ERISC1 ELFs differ (v3e1b's); all others are identical.
- **VC1 size, E0+E1:** -O3 13,200 + 12,140 = 25,340 B (1,284 B left); -O2 24,304 B; -Os 21,340 B.
- **Reading a hardware result:**
  - The hang is timing-dependent, so one passing run settles nothing. If v3l2be1b passes repeatedly where v3l2e1b hung, §9's over-fill mechanism becomes the likely cause; the register read in §9 confirms it.
  - If v3l2be1b still hangs, E0's VC1 path behaves as v3a's for every register value. The cause is then in the E0/E1 pairing's timing or below the router code.

## 11. v3l3: forwarding passes at v3a's length (and v3l3b)

Context: v3l2be1b is the v3 base (RESULTS §93). Its E0 forwarding passes run ~+7.5 per forwarded 16 B packet over v3a's E0, and +3.5 / +6 on 3-hop and multicast. Variants: `fabric_exp/variants/v3l3/` and `fabric_exp/variants/v3l3b/`, each built from v3l2be1b.
- Each diff is 68 lines of `fabric_erisc_router_v3.cpp`, all inside `namespace e0` and its header comment; headers are v3a's. v3l3b differs from v3l3 in one template argument and two comments.
- `_strict` and `_hook` variant dirs were made with prep_round.sh (at `--jobs 16`). Their -O3 builds are in `scratchpad/prep/build_v3l3{,_hook}` and `build_v3l3b{,_hook}`.

**How the passes were measured.** A small RV32 emulator runs E0's main loop from the ELF and counts each pass's instructions, taken branches and loads. It is `scratchpad/v3l/emu.py`, with scoring in `scratchpad/v3l/l3/ev.py` and `l3/table.py`.
- **Device model:**
  - free-slots registers are scripted: a busy channel always holds a packet;
  - TXQ and NoC command buffers are never busy;
  - each channel's completion counter catches up every second pass, so credit frames run at 0.5 per packet (hardware: 0.22 worker, 0.44 forwarded).
- **v3a** runs from `kernel_main`, with its init waits skipped or forced.
- **Score** = instructions + 2 × taken branches + loads. It is only a proxy for cycles, but it matches the hardware on the FWD row: v3l2be1b scores 240 per two-packet pass, i.e. 120 per packet, against 119.4 measured (E0-bound).

**What changed, and why each change removes work.**
1. **No calls on the pass's path.** `credit_upstream_router` (the upstream credit write) and `wait_for_sender_txq_idle` are inline again, as in the generic loop.
   - In v3l2 the credit write was a call on 0.2–0.5 of forwarded packets. Every value live across it then had to sit in a callee-saved register, or be saved and reloaded around the call. The forwarding send paid stack reloads and rematerialized constants for it.
   - Stack loads per two-send pass fall from 21 to 8. The completion pass loses the call and its saves (208 → 181 instructions).
   - VC1 code gets smaller, not larger, at -O3.
2. **The idle-context-switch flag is no longer written on every send.**
   - Before: `attempted = true` on every send attempt.
   - Now: a yielded attempt (TX queue busy) sets it, on the cold path. A round that sent anything is detected from the send counts, comparing worker_sent + Σ channel.sent at the round's start and end.
   - The decision is the same as before, because every attempt either yields or sends. That is 2 instructions and 1 stack store less per send.
3. **No combined test on VC0 in v3l3.** Each forwarding channel is tested on its own (free-slots value, in-flight bit), as in the generic loop.
   - On a pass that forwards, the combined test always fails: 4 instructions and a taken branch for nothing. On VC0 that is the common case on every router that is not a source or a sink.
   - VC1 keeps the test (template parameter `COMBINED_TEST`): its channels are idle on most passes.
   - **v3l3b** keeps it on VC0 too. That makes it the other shape to measure: shorter worker-only and idle passes, longer forwarding passes.

Unchanged: the worker path's structure, the exact idle predicates (§10), E0 init ownership, the per-pass connection check, the in-flight word, and the order of every register write.

**Same device behaviour (emulator check).** Every device-register write (TXQ commands, stream-register updates, credit NoC writes) was logged in order over 60 passes.
- **v3l3 and v3l3b against v3l2be1b:** identical sequences in all 13 scenarios: 2D (worker, ch1, ch2+3 and all-four mixes), 1D line, and the largest VC1 E0 with VC0 and VC1 traffic. This holds for both variants at -O3, -O2 and -Os, against v3l2be1b's -O3 build, whose own -O2 and -Os builds give the same sequences.
- **v3l3 against v3a (2D, -O3):** the same sequence except the worker credit's target address. That address comes from connection data, which the emulator zeroes differently for the two programs.

**Per pass, 2D E/W E0 (`10733501891889981293`)**, as instructions / taken branches / loads, with the score in brackets. FWD rows send on the worker channel and channel 1. Channel 3 (N/S routers) gives the same picture: v3l3 202 / 135 against v3a 192 / 126.

| -O3 | v3a | v3l2be1b | v3l3 | v3l3b |
| --- | --- | --- | --- | --- |
| worker + forwarded (FWD) | 142/7.4/40 [197] | 178/11.6/40 [240] | 154/9.0/27 [200] | 160/9.0/29 [207] |
| forwarded only (3-hop, multicast) | 89/8.7/24 [131] | 113/9.9/23 [155] | 101/7.3/17 [133] | 105/7.3/18 [137] |
| forwarded on every 2nd pass | 70/7.6/19 [104] | 87/8.2/18 [121] | 78/6.7/15 [106] | 82/6.7/15 [110] |
| worker only (E/W, N/S rows) | 88/5.8/25 [125] | 91/5.8/24 [126] | 84/7.8/16 [116] | 80/5.8/18 [110] |
| idle | 35/7.0/9 [59] | 26/4.0/7 [41] | 31/6.1/7 [49] | 25/4.1/7 [39] |

| score at -O2 / -Os | v3a | v3l2be1b | v3l3 | v3l3b |
| --- | --- | --- | --- | --- |
| FWD | 198 / n/a | 236 / 230 | 204 / 209 | 205 / 214 |
| forwarded only | 132 / n/a | 154 / 163 | 138 / 142 | 141 / 147 |
| worker only | 125 / n/a | 124 / 115 | 120 / 122 | 104 / 106 |
| idle | 59 / n/a | 41 / 48 | 53 / 56 | 39 / 40 |

(v3a's -Os build does not get through init in the emulator.)

**The forwarding send pass, segment by segment.** -O3, forwarded only: one forwarded packet, no completions.

| Segment | v3a | v3l2be1b | v3l3 |
| --- | ---: | ---: | ---: |
| grouped free-slots read | 8 | 6 | 5 |
| idle worker: packet and completion tests | 10 | 8 | 7 |
| connection check | 5 | 3 | 2 |
| forwarding tests before the send | 3 | 5 (combined + own) | 2 |
| channel 1 send | 42 | 58 | 56 |
| after the send (completion test, in-flight bit) | 1 | 3 | 5 |
| channels 2 and 3 idle tests | 6 | 8 | 8 |
| back-edge | 2 | 2 | 2 |
| total (instructions / taken / loads) | 75 / 9 / 21 | 93 / 9 / 19 | 86 / 7 / 15 |

- **The send itself is still ~14 instructions above v3a's.**
  - GCC rematerializes constants that v3a happens to hold in registers or stack slots: the TXQ base, command values, stride and slot bounds.
  - The forwarding channel's counts, slot and the in-flight word live in stack slots.
  - v3a's send loads ~6 more values from the stack, so the two sends are close in the proxy.
  - Changing this means steering allocation, which is out of bounds.
- **Completion passes gain the most.** With credits, forwarded only: v3a 101 / 8 / 27, v3l2be1b 129 / 10 / 26, v3l3 114 / 7 / 19.
- **1D line E0** (`10744374302904868627`, -O3):
  - FWD: v3l3 143 / 7.0 / 23 against v3l2be1b 152 / 8.5 / 28;
  - worker only: 72 / 5.8 / 12 against 81 / 5.8 / 18;
  - forwarded only: unchanged, 91.

**Tried and dropped** (same harness; each moved results only by GCC allocation and layout noise of ±5–10 in the proxy, or lost):
- in-flight bit set on send and cleared on catch-up;
- in-flight bit updated only on transitions;
- per-channel outstanding counts in place of the in-flight word: idle tests then load per channel;
- dropping the in-flight word for v3a's `sent != completed` test;
- forwarding visits ahead of the worker visit: forwarded only −3, but worker +7 and FWD +3;
- testing the in-flight word first in the combined test: GCC merges the two tests either way;
- skipping the combined test for single-channel VCs: GCC already merges it on 1D;
- inlining the connection event: no change, so it stays out of line.

**Builds** (-Werror, all logged configs; `--jobs 16`):
- -O3 536/536, -O2 536/536, -Os 534/536 for both variants. The two -Os failures are the malformed logged commands; v3a fails the same two.
- **E1 is identical to v3e1b on every config,** keyed by MY_ERISC_ID with PT_LOAD images compared: 208 ERISC1 configs at -O3 and -O2, 207 at -Os. The same holds under the hook probe: 328 subordinate ELFs, against both v3l2be1b_hook and v3e1b_hook.
- **E0:** the 168 configs where `e0::runs` changed; the other 160 E0 ELFs are identical to v3l2be1b.
- **VC1 E0 + E1** (largest of each over the VC1 configs; limit 26,624 B):

| | -O3 | -O2 | -Os |
| --- | --- | --- | --- |
| v3l2be1b | 13,200 + 12,140 = 25,340 | 24,304 | 21,340 |
| v3l3 | 13,064 + 12,140 = **25,204** (1,420 B spare) | 24,120 | 21,860 |
| v3l3b | 13,120 + 12,140 = 25,260 | 24,184 | 21,876 |

- **Stack:** deepest E0 is 880 B at -O3, 944 at -O2 and 928 at -Os (v3l2be1b 800 / 864 / 912; v3a 1,408 / 1,632 / 1,504).
- **Probes (-O3):**
  - Statefield leaves the 2D passes identical, and changes 1D and VC1 passes by ≤1 instruction.
  - Under the hook, v3l3 scores FWD 216, forwarded only 141 and worker 129, against v3l2be1b_hook's 247 / 161 / 133. v3l3b_hook scores 219 / 144 / 119.
- **Compile coverage:** BW + perf telemetry (-O3, -Os) and trimming capture build for both variants on 2D, VC1, 1D and ring E0s.

**Risks and what to watch on hardware.**
1. **The proxy is a proxy.** It ranks v3l3 at v3a's length on forwarding passes: FWD 200 against 197, forwarded only 133 against 131.
   - Expect the FWD rows to become E1-bound at about 112.
   - Expect 3-hop and multicast within a cycle or two of v3e1b's 143.4 / 137.5.
   - If they stay at v3l2be1b's level, the remaining cost is in the send itself (allocation, above), not in the pass structure.
2. **Worker rows on v3l3:** the three ungated channel tests cost +2 taken branches per worker-only pass.
   - The proxy still has v3l3 ahead of v3l2be1b at -O3 and -O2 (116 / 120 against 126 / 124), but behind it at -Os (122 against 115).
   - Watch E/W and N/S 16 B at -Os: if E1 stops binding there, v3l3 could give back 1–3.
   - v3l3b keeps the gate and is ahead on worker rows at every level (110 / 104 / 106).
3. **VC1 routers:** v3l3's worker-only pass is about +3 in the proxy against v3l2be1b. VC1 traffic is −3, and with forwarded VC0 traffic −25 to −30.
4. **Idle routers:** v3l3's idle pass is 29 instructions against 24. Its only effect is how soon an idle E0 makes its next pass; the context switch decision is unchanged.
5. **Merging with v3e1c:** the diff is confined to `namespace e0` and the header comment above it, so the three-way merge over v3l2be1b should be mechanical.
