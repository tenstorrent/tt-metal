# What changed against the old router, and why the new router is faster (draft for review)

Source for the report's "What changed" tab. build_page.py parses the theme, the chapters (### headings, each with an
intro) and their entries, and the tried-and-dropped table.

## Theme

**A packet costs its time on the wire plus the router's bookkeeping, and a link runs at whichever is slower.**
Bookkeeping is everything the router does around the transfer: finding the packet, checking for room downstream,
programming the data movement, returning credits, polling for the next thing to do. At 400G a 4,352 B packet spends
about 125 cycles on the wire. The old router spent about 220 cycles of bookkeeping per packet on the 2D mesh, whatever
the packet size, so its links waited on the router. The new router does the same work in 120-130 cycles and runs the
link at about 98% of the wire.

**Where the old router's cycles went.** The old router was measured cycle by cycle before anything was designed: phase timers,
a fixed delay on one core at a time, and PC samples read over the debug bus while traffic ran. Most of the 220 cycles
were not the transfer itself:
- **Load stalls.** The Ethernet core runs in order and stalls on a load at the load's first use. A register or L1 read
  costs 6-10 cycles; a read from the core's local data memory, where the stack lives, about 2. The old router issues its reads
  one at a time, each used at once, so it pays nearly every stall in full.
- **Polling.** Every pass visits every sender channel, busy or idle, at about 27 cycles a visit: more than a third of
  a 2D packet's time.
- **Redundant work.** Every NoC command is programmed field by field, though most fields never change for a channel,
  and its command buffer is polled before each command. Values the build already knows are reloaded at run time.
- **Expensive credits.** Every packet returns its own credit frame, each with a TX-queue check and a remote write, and
  a flushed atomic waits on the transaction IDs one at a time. Credit handling alone took 75 cycles per packet on
  the old router's sending core and 97 on its receiving core.
- **Register pressure.** The main loop, both roles' channel steps and the per-round housekeeping are all inlined into
  one function. The hot state does not fit the registers, so the compiler spills it to the stack, and where it spills
  changes with unrelated edits.

**What the new router does instead.** It was designed around those costs, along four axes:
- **Work with the hardware.**
  - Issue the reads a pass needs together, so their stalls overlap.
  - Keep the NoC's command buffers programmed and write only what changes.
  - Test idle channels with one bit.
  - Send credits as counts, merged per pass and per group.
- **Keep the state minimal and in the right place.**
  - Each core runs only its own role.
  - Per-channel facts are compile-time constants.
  - The state every pass touches lives in registers, colder state in the core's fast local memory, by design.
- **Work with the compiler.** The source decides the generated code through its structure, not the register
  allocator's luck:
  - a hot loop of its own, with rare work out of line;
  - fences that keep each read group together;
  - a placement primitive for state meant to stay in memory.

  The goal is speed and codegen that stays put when other code changes.
- **Size buffers and flow control from measurements.**
  - Slot depths come from the credit round trip each kind of channel needs.
  - Transaction IDs follow each channel's slots.
  - Ring flow control keeps the ring moving.

The theme throughout: more useful work per packet, and fewer cycles polling, waiting or redoing work.

**How the work was done.**
- **Measure first.** The tools:
  - per-phase timers;
  - one-core delay injection, to find which core sets a row's rate;
  - PC sampling over the debug bus, which leaves the rate unchanged;
  - microbenchmarks of loads and register reads;
  - a bare-link probe and a frame model for the Ethernet ceiling.
- **Keep a change only on a same-job A/B against the build before it, across every suite.** That covers the P150x8
  (400G) and the BH Galaxy (200G), and more than 500 router variants. Estimates chose what to try, never what to keep.
- **Check codegen, not only speed.** Every configuration was rebuilt offline at -O3, -O2 and -Os and compared image by
  image, so a cleanup meant to leave code unchanged provably does.
- **Robust over peak.** Designs with timing cliffs were dropped, and delay-perturbation sweeps checked for cliffs. A 1-2
  cycle loss was accepted for a smoother design, and a design that hung was ruled out.

## 1. What changed, by chapter

### Hardware: loads and polling

The core can have up to four reads in flight. Four reads issued together cost about as much as one; a lone read used
at once always costs its full latency, wherever it sits. The new loops issue the reads a pass needs together and use
them late. They skip reads whose answer is already known, and they never wait where they could serve another channel.

**1. Reads issued as groups.**
- Changed: the old router reads each value where it first needs it. The new router reads in groups:
  - the sender reads every channel's free-slot register at once at the top of a pass;
  - after the data command, the send reads the TX-queue status and its credit counters as one group;
  - the receiver reads its packets-received and TX-queue registers together;
  - a forward keeps the downstream credit counts it has seen, and re-reads them as one group only when a count it
    needs runs out.
- Why: the stalls of a group overlap, so a pass pays about one read's wait for up to four reads.

**2. Idle channels cost one bit test.**
- Changed: the old router visits every sender channel on every pass. The new sender keeps a mask of channels that hold a packet
  or have packets in flight, built from the pass's grouped reads, and visits only those. A VC's forwarding channels
  are skipped with a single test when none has work.
- Why: on the old router an idle visit is two dependent reads, each used at once. Here it is one bit in a register.

**3. The header read once, early.**
- Changed: the header word (size, send type, source channel) is loaded once as a word and unpacked later. The 2D route
  decode issues both route-byte loads back to back; the old router loads the X byte only after it has tested the Y byte.
- Why: the header's latency overlaps other work, and the decode's two loads overlap instead of queuing behind each
  other.

**4. No read or wait whose answer is known.**
- Changed:
  - completion counters are read only while packets are outstanding;
  - a send that finds the TX queue busy goes back to the loop instead of spinning;
  - a single completed slot on a ring is checked once and otherwise left to a later pass.
- Why: the core spends that time serving other channels. The TX-queue status read itself stays: it orders the command
  writes, and dropping it hung the router.

### Hardware: the NoC

A NoC command is a set of command-buffer registers written one by one, after a poll that the buffer is ready. The old router
programs most commands whole each time. The new router treats the command buffers as state it owns: each buffer
carries one kind of command, its fixed fields are written once, and each packet writes only what changes. All of it
goes through the old router's stateful NoC API, extended where needed, with no raw register sequences.

**5. Command buffers with one role each, preset once.**
- Changed: a table gives each command buffer one kind of command: local write, forward data, forward credit, worker
  credit, atomics. Forwards run on the link's NoC virtual channel, with posted credits. The old router presets forwards only on
  2D routers with one live direction. The new router presets:
  - its local writes;
  - every forward, fully where all forwarding VCs share one destination, otherwise partially (per forward only the
    source, destination, coordinate and length change);
  - the worker credit, whole.
- Why: a forward drops from 53 instructions and 14 register writes to about 24 and 9 with a partial preset, or 6 with
  a full one; a worker credit becomes one store and one send request. The form is chosen per router shape at compile
  time.

**6. Ready checks paired, or skipped where already known.**
- Changed: a forward reads the data and credit buffers' ready bits in one read, then issues both commands without
  their own polls. The old router's API polls before every command, and its stock stateful write polls twice.
- Why: one poll in place of two or three serial ones, each used at once.

**7. A flushed atomic waits on one counter.**
- Changed: before a flushed atomic increment, the old router waits on the channel's transaction IDs one by one. On Blackhole
  without deadlock avoidance, the new router waits once on the NoC's count of acknowledged writes, which covers every
  earlier write. Earlier atomics are not waited on.
- Why: one counter test instead of a poll per transaction ID.

### Hardware: Ethernet and credits

Every packet needs credits back: the receiver tells the sender that a slot is free, and the sender tells its producer.
The old router sends a credit frame per packet. The new router makes each credit event cheap rather than holding credits back.
Holding them was tried several times; on rings it can stall the whole ring.

**8. Credits as counts, one frame per pass.**
- Changed: credits travel as counts in L1 counters on every VC and in every mode. The old router uses stream registers on
  single-core routers. Each pass merges all its acks and completions into one frame.
- Why: one frame per pass instead of one per credit event, and a counter is cheaper to read and update than a stream
  register.

**9. Completions credited in groups, proved by one read.**
- Changed:
  - The old router credits each slot. The new router credits aligned groups of 2 or 4 slots, sized by the ring depth. Rings and
    tori keep groups of 1, since their packets are acknowledged one by one.
  - Every NoC command that reads a slot carries its group's transaction ID, so one NIU status read proves the group's
    writes have left.
  - A group waits for that proof, bounded, rather than slipping to a later pass. A partial group goes out after a few
    idle passes.
- Why: one status read, one credit frame and one producer credit per group instead of per slot, on both cores.

**10. Ethernet keepalive interval.**
- Changed: the old router leaves the firmware's TX-queue keepalive at 8,000 cycles. The new router's host sets 1,000 on
  Blackhole while a router runs.
- Why: on one-way traffic, sequence acknowledgements come back sooner. Otherwise frames above 4,114 B are floored near
  187 cycles per packet whatever the router does. It helps the old router too. On wire-bound 200G rows it costs about
  0.4%.

### Keeping the state small

A value in a register is free to reuse. A spilled value costs a store and a reload, and the reload stalls if it is
used at once. In a loop shaped like the old router's, the sender's per-pass state lived in a large struct on the stack, and
7-15% of the sending core's time went to that memory traffic. The new router decides where each piece of state lives,
and keeps the state small enough that the hot part fits in registers.

**11. One program per core.**
- Changed: the old router runs one loop on both Ethernet cores of a link, with sender or receiver channels switched on per core.
  The new router gives the sending core a sender program and the receiving core a receiver program. A single-core
  router runs both passes in turn.
- Why: each loop carries only its own role's state, so more of it stays in registers. Each program's stack and code
  shrink: the receiver's frame halved, and the sender's deepest stack went from 1,408 to 800 B.

**12. Per-channel facts are compile-time constants.**
- Changed: each channel's code is generated from its compile-time description: its stream registers, slot range,
  command buffers and downstreams, and the form of its forward (full, partial or no preset). The old router keeps several of
  these in run-time tables and structs.
- Why: constants fold into immediates. The hot path then has no table loads and no chains of dependent address loads.
  Run-time channel tables were tried five ways and lost every time.

**13. Hot state in registers, colder state in local memory by design.**
- Changed: state is split by owner: the sender channel, the VC, the command buffers, the receiver channel's progress.
  Each owner keeps only what it needs.
  - The VC every pass serves keeps its channel state in loop locals, which the compiler holds in registers.
  - A later VC's state, which only some passes touch, is kept in memory on purpose (#16).
  - The far receiver's free slots and next slot are packed into one word.
  - Grouped reads land in scalars, not records.
- Why: the registers go to the state every pass uses. The rest is read from 2-cycle local memory, and only on the
  passes that need it. Spills no longer land right after long-latency reads, where each spill store waits for its
  load.

### Working with the compiler

The old router compiles as one function: the main loop, both roles' channel steps and the per-round housekeeping are
force-inlined into `kernel_main`. Its hot loop competes for registers with code that rarely runs. The compiler spills
under that pressure, and how it lays out the loop depends on everything around it, so an unrelated edit can move a row
by several cycles. The old router's initialization carries four `asm volatile("nop")`, added to undo a performance regression
that a postcode change caused through code layout (#35346). The new router is written so its structure, not chance,
decides the generated code: speed first, and codegen that stays put. It uses documented compiler semantics only: no
branch hints, no register pinning, no tuning to one compiler version.

**14. A hot loop of its own, rare work out of line.**
- Changed:
  - Each core's loop is a function of its own whose body is only the per-pass work.
  - Per-round work (termination, heartbeat, context switch, router commands) is in `noinline` functions, as are rare
    paths (connection events, teardown, sparse multicast, routing exits).
  - The control paths carry GCC's `cold` attribute.
- Why:
  - The loop's register allocation and layout depend only on its own body, so the hot loop gets the registers.
  - Edits to rare code no longer re-lay out the loop: ordinary edits move suite means by under half a cycle.
  - The -Os penalty the old router pays mostly disappears.

**15. Fences keep read groups together.**
- Changed: each read group sits between `begin_read_group()` and `end_read_group()`, compiler barriers
  (`std::atomic_signal_fence`) that emit no instruction.
- Why: `volatile` keeps the reads in order, but the compiler may still schedule their uses between them and spread
  them out, so each read stalls alone. No memory access crosses a fence, so the loads issue back to back and their
  first uses come after the group. This lets plain C++ say "issue these together", where the new router's first
  builds needed inline assembly.

**16. Colder state stays in memory on purpose.**
- Changed: `keep_in_memory(obj)`, i.e. `asm volatile("" : "+m"(obj))`, tells the compiler, by GCC's documented
  semantics, that the object's value is in memory at that point. The value is stored before and read back after. It is
  applied to state that only some passes touch.
- Why:
  - Left alone, the register allocator decides what to spill, differently with each edit and optimization level, and
    tends to spill hot values right after long-latency reads.
  - With the primitive, the registers go to the every-pass state, and the cold state is read from 2-cycle local memory
    only when needed.
  - It applies to the whole object, so hot and cold state live in separate objects.
  - Memory-held state is reloaded around every fence, so the send path keeps its state in registers.

**17. Smaller code: -O3 everywhere.**
- Changed: initialization and rare work are out of line, and cold code has no per-channel copies.
- Why:
  - The new router's VC1 build is about 22 KB at -O3. The old router's is 30 KB, which does not fit the 26 KB kernel buffer, so
    the old router ships VC1 at -Os.
  - Every configuration now gets -O3's unrolling, scheduling and frequency-aware register allocation.

**18. Source shapes that keep codegen predictable.**
- Changed:
  - no closures in the loop;
  - no branch hints;
  - flags combined with shifts, not ternaries;
  - fields copied once and passed by value;
  - telemetry and debug calls behind `if constexpr`, so builds without them are byte-identical to builds that never had
    them.
- Why: written the other way, each of these changed what the compiler kept in registers or how it laid out the loop.
  A captured lambda, for one, became a stack object of pointers that every channel test loaded through.

### Buffers and flow control

A channel of W slots can move at most W packets per credit round trip, so slot counts set floors no code change
removes. Rings add a correctness constraint: a ring must never fill so that nothing can move.

**19. Slot depths from the credit loop each channel needs.**
- Changed:
  - The old router takes its slot counts from a tier table; for VC1 it gives 51, 4, 8, 2 and 4 slots.
  - The new host allocator gives each kind of channel a full-rate depth: the slots it needs to cover its credit round
    trip at the router's rate.
  - It splits the buffer in stages: receivers first, then senders, then the leftovers by a fixed order.
  - Sender channels need no power of 2; receivers stay powers of 2.
- Why: no channel is held back by its window where the buffer has room, and the split is the same on every router of
  a fabric.

**20. Transaction IDs follow each receiver channel's slots.**
- Changed: the host gives each receiver channel transaction IDs from its slot count and completion groups, within the
  NoC's 16. The old router gives every receiver channel a fixed 4 (8 on rings).
- Why: every completion group gets its own ID where the IDs allow (#9). Where they don't, groups share an ID, which
  can only delay a proof, never make it wrong.

**21. Bubble flow control sized to the ring.**
- Changed: on rings and tori an injecting channel sends only while the far receiver has max(2, R − 16) free slots. The old router
  uses 2, whatever the depth. A completion on a ring is never held back.
- Why: injected packets never fill the ring, which would stall its forwarded traffic (ring jam). A held completion on
  a ring is a held bubble slot.

### Features

**22. Configurations the old router cannot build.**
- Changed: VC1 and VC2 together, and VC2 on a single-core router. The old router fails stack usage on the first and a
  static_assert on the second. Every CI multi-mesh configuration fits at -O3.
- Why: the smaller per-core programs (#11, #17) leave the stack and code room.

## 2. Tried and dropped

| Tried | What it taught |
| --- | --- |
| Offloading local delivery to the chip's DRISC cores | Handing a packet to another core costs about what delivering it costs, and adds a hop of latency. The Ethernet cores' own work was the bigger lever. |
| Holding credits back to send fewer frames (several forms) | On deadlock-avoiding rings a held completion is a held bubble slot, and the ring advanced one hop per timeout. Make each credit cheaper; never hold it. |
| A credit per slot everywhere, with no groups | Simpler, but each credit costs both cores. Groups stay wherever packets are not acknowledged one by one. |
| Run-time channel tables and shared visit code (five forms) | A table access is a chain of dependent loads, and a shared call forces state into callee-saved registers. Per-channel compile-time code wins. |
| Plain volatile loads for the read groups | The compiler put the uses between the loads and the overlap was lost; hence the fences. |
| Parking the send path's state in memory | The fences make memory-held state reload around each one. The send path keeps its state in registers. |
| Every receiver cursor as a register-held local | The compiler then spilled right after long-latency reads, where each spill waits for its load. Placement has to be chosen, not left to the allocator. |
| A smaller channel struct | Moved register allocation the wrong way, three times. A smaller type alone does not win. |
| Ready checks grouped with the header loads | The wait moved ahead of decode and delayed the header's first use, since loads complete in order. A group holds only reads the pass needs at that point. |
| Preloading the next packet's route | A byte load's zero-extension is its first use, so the stall stayed where it was, and the carried values spilled. |
| Spinning on a busy TX queue; dropping its check | The queue is almost never busy, so yielding costs nothing. But its status read orders the command writes, and dropping it hung the router. |
| Stream-register credits on a single-core router | A stream-register read and decrement cost more than an L1 counter. |
| Credits attributed by ring position | Cheaper on the receiving core, dearer on the sender under mixed traffic. Also a lesson in method: the bound came from profiles, and an A/B overturned it. |
| Several sends per channel visit | Real gains once the worker window is deep, none at default depths, and losses on merged links. Parked. |
| Branch hints | They turned branchless checks into taken branches. |
| A lambda in the sender loop | The compiler kept its captures as a stack object of pointers. Static counts fell while the hot path got slower. |
| A stateless credit write in place of the preset one | It hung the Galaxy. The stateful form is required, not only faster. |

## 3. Summary for readers outside the fabric team

Glossary:
- A **router** is the small program on the two little processors (ERISCs) that drive each Ethernet link between chips.
- A **packet** is one message between chips.
- A **credit** is the receiver's note back that a buffer slot is free again.

- **Faster on most traffic.** On the 8-chip test system, with both processors of each link in use, the PR's router
  moves 1.35 to 1.54 times as many packets per second as the current one across the 2D, 1D, ring and torus suites,
  measured side by side.
  - The study build it came from reached 2.1 to 2.5 times on the two-mesh (VC1) suites, which have not been rerun on
    the PR's router.
  - With one processor per link the gain is 1.07 to 1.24 times; through the mux it is about even.
  - Bursty traffic with long pauses is about even, because the pauses set its pace (CLAIMS A1).
  - Part of the gain on two suites comes from settings that changed with it: the keepalive interval, the buffer slot
    counts and the compiler setting.
- **At the cable's limit where the cable is the bottleneck.**
  - On the 400G 1D line, large packets over one hop run as fast as the link allows (99.7% of it at 4,352 B); on the 2D
    mesh the same test reaches 97.7%.
  - On the 200G Galaxy at 4,352 B, both routers reach the link limit on 1-hop, 3-hop, ring and 1D writes (and on the
    1-hop rows at 4,096 B), so those sizes do not separate them.
  - On the XY torus only the new router reaches it (99.5% against 93-94%), and multicast stays near 76% on both.
- **How:** each packet costs its time on the cable plus the router's bookkeeping, and the old router's bookkeeping took longer
  than the cable. The new router cuts the bookkeeping:
  - it waits on several memory reads at once instead of one after another;
  - it sets up the chip's data mover once and changes only what differs per packet;
  - it sends "slot free" notes in groups and in combined messages;
  - it keeps what it uses every packet in the processor's registers;
  - each processor on a link does one job (one sends, one receives);
  - the program is laid out so the compiler produces the same fast code even as other parts change.
- **Smaller and more capable.** The program is smaller, so every configuration builds with the compiler's fastest
  setting. The old router cannot do that for setups that join several groups of chips. The new router also runs
  traffic-class combinations the old router cannot build.
- **Not everywhere ahead:**
  - One single-processor configuration still trails on the PR's router: neighbor exchange at 4,352 B, 2.8% more time
    per packet.
  - On the 200G Galaxy, neighbor exchange at 3,072-4,352 B trails the old router by 0.6-1.5%; a likely cause is being tested.
  - Full-ring broadcast of the largest packets on the Galaxy, which trailed on the study build, now runs about 5% faster
    than the old router (254.1 against 267.7 cycles per packet).
  - Decode collective ops (ttnn traces) range from 0.6% slower to 13.3% faster on an earlier build, mostly compared
    across separate runs (CLAIMS A5).
