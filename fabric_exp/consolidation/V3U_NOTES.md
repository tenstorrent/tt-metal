# Router v3u: one runtime table for every sender channel on ERISC 0

v3u is v3a with ERISC 0 (the sending ERISC, `MY_ERISC_ID == 0`) rewritten as one program. Every sender channel E0 steps
(the worker channel and every forwarding channel, VC0 and VC1) is one 32 B entry of a table in local memory. One runtime
step body serves every entry. Each pass reads all the free-slot registers as one group and builds a work mask from them
and from a pending-completion mask. It then visits only the channels in that mask. E1 is untouched: its code is
identical to v3a's on all 328 E1 configs at -O3, -O2 and -Os.

Offline result: all 536 configs build at all three levels. With VC1, E0 + E1 fit the 26,624 B kernel config buffer at
-O3 (25,496 B; v3a 29,704 B). A VC1 E0 shrinks by 4.2 KB at -O3. The E0 program is about the same size and shape at
-O3, -O2 and -Os, and there is one send body per VC at every level.

Files: `fabric_exp/variants/v3u/`. Only `fabric_erisc_router_v3.cpp` differs from v3a. The new section is "ERISC 0 transmit
program" just above `kernel_main`, and `kernel_main` gains one `if constexpr`. The shared headers are v3a's, unchanged.

## 1. Design

**Who runs it.** The program runs when `use_uniform_tx_program` holds:
- two ERISCs, E0, multi-TXQ, and no receiver channel on this ERISC;
- not speedy and no VC2;
- Blackhole, no spin-wait TX queue and no data cache;
- every stepped channel uses counter credits;
- the worker channel, if there is one, is channel 0.

Every other configuration keeps v3a's templated loop (`run_fabric_edm_main_loop`) and v3a's code. That covers
single-ERISC, speedy, VC2, stream-register credits, Wormhole and UDM/mux, which force single-ERISC. Of the logged
configs, 168 of 208 E0s use it: 1D 40, ring 40, 2D 40 and 2D VC1 48. The 40 speedy E0s are byte-identical to v3a.

**Structure.**
- `kernel_main` is unchanged up to the static-connection wait. For covered configs it then calls
  `run_uniform_tx_program` (`noinline`), which runs the following, then returns to the usual teardown:
  - table init;
  - static-connection wait;
  - last ERISC sync, `INITIALIZATION_COMPLETE`;
  - the RUN/PAUSE outer loop;
  - the 32-pass inner loop.
- The program owns its state. It makes local copies of the ch0 worker interface and of its own table and far rings, so
  the loop reaches nothing in `kernel_main`'s frame through pointers.
- The static-connection wait is one out-of-line helper called once per channel that has an upstream router
  (`tx_wait_for_upstream_router`). It is the same wait as v3a's and returns the upstream credit address into the table.

**State.**

| State | Where | Compile time or run time |
| --- | --- | --- |
| `TxChannelState` per channel: next slot, first slot, last slot, packets sent, completions processed, acks processed, upstream credit NoC address (8 B) | `std::array` in the program's frame, indexed by compact channel. Indexed at run time, so it stays in memory: there is nothing for SRA to split | Run time. The cursor and counts change. The slot bounds are filled once from compile-time alloc entries. The credit address is known only once the upstream router connects |
| `TxFarReceiver<VC>`: next far slot, free far slots | locals (one pair per VC) | Run time. Base, slot count, stride, last slot and the packets-received register are compile-time members (`eth_remote_channel_allocs`, `REMOTE_RECEIVER_NUM_BUFFERS_ARRAY`, `channel_buffer_size`). v3a held them in a struct and loaded 3 of them from the stack per send |
| `pending`: bit per channel with packets sent and not yet fully completed (and acked, with first-level ack) | one local word | Run time |
| Worker connection flag and worker interface copy | locals | Run time |
| Stepped set, per-VC masks, slot counts, free-slot register addresses, credit-counter addresses (`base + 4·ch`), free-slot update register (`VC base + (ch − first) · 0x1000`, contiguity is asserted at the top of the file), worker channel, injection mask, first-level ack per VC | `constexpr` | Compile time. They are fixed by the host's compile-time args. Making them run time would only add loads |

**One pass.**
1. Read the free-slot registers of all stepped channels as one group (`tx_issue_loads`, four per asm block, blocks back
   to back: 4 loads on 2D, 7 on VC1).
2. Idle bits: a one-bit test for power-of-two slot counts, otherwise `(v & 0x7FF) == slots`. Then
   `unsent = idle ^ stepped_mask`, so a channel costs 3–4 ALU operations and no branch.
3. Issue the worker's connection-semaphore load.
4. `work = unsent | pending`. Per VC: `while (w) { ch = ctz(w); w &= w − 1; visit(ch); }`, so channels are visited in
   ascending order, at most one packet each.
5. Worker check, every pass: `!connected || sem == CLOSE_REQUEST` takes the open or teardown path on the value loaded in
   step 3.

**Visit (`tx_visit_channel<VC>`).** The same per-packet work as v3a's `sender_channel_try_send` plus
`sender_channel_process_completions`, with the channel index as a run-time value:
- the far-space test (two slots for an injection channel under bubble flow control);
- v3a's fenced pre-send group (payload size, BH-55 CMD, STATUS); a busy queue yields;
- `did_something`, the `src_ch_id` stamp, the data command, the far and slot cursor advance;
- v3a's fenced overlap group (CMD, STATUS, completion counter), free-slot +1, the packets-received register update;
- sent++ and `pending |= bit`;
- completions: far slots freed, credits returned upstream (to the worker's absolute read counter via the unchanged
  `send_credits_to_upstream_workers`, or +n to the upstream router's register), and `pending` cleared once
  processed == sent;
- with first-level ack, completions only free far slots and acks return credits.

The worker arm in the credit return is selected by `ch == 0`, only in VC0's instantiation.

**Code shape (checked in the disassembly, not tuned by it).**
- One send body per VC at -O3, -O2 and -Os: one `src_ch_id` stamp and one `ctz` per VC.
- The program function, in instructions, at -O3 / -O2 / -Os:

  | Config | -O3 | -O2 | -Os |
  | --- | ---: | ---: | ---: |
  | Hot 2D E/W E0 | 566 | 537 | 512 |
  | 1D | 537 | 503 | 477 |
  | Ring | 555 | 529 | 493 |
  | 2D VC1 (two VCs) | 794 | 773 | 718 |

- Idle pass on the hot E0 at -O3: 31 instructions (v3a: 34). VC1 idle pass: about 45 instructions for 7 channels.
- A sending pass on the hot E0 is about 100 instructions, about the same as v3a.
- At -O3 and -O2 GCC places the send block out of line (two taken jumps per send). At -Os it is inline.
- Around the visit loop GCC stores and reloads 3 values at -O3 (loop counter, semaphore value, semaphore pointer) and 2
  at -O2.

## 2. Per-pass and per-packet work vs v3a

| | v3a (hot 2D E/W E0, ch0 sending, ch1–3 idle) | v3u |
| --- | --- | --- |
| Free-slot registers | VC0 ch0–3 grouped (fenced). ch4 and the VC1 channels each read at their visit and used at once | Every stepped channel in one group (asm primitive), 7 on VC1 |
| Idle forwarding channel | Visited every pass: a bit test plus `sent != processed` (registers or stack), branches, ~5 cycles each (§73). VC1 and ch4 channels: a full visit (register read used at once, compare, pending check) | Not visited: 3 ALU ops in the mask, no branch |
| Idle worker channel (no packet, nothing in flight) | Visited: packet test, completion test (2 stack loads), connection check | Not visited. Connection check only |
| Worker connection check | After ch0's step: stack load of the semaphore pointer, then the L1 load, used at once (~13 / pass, §73) | Same check every pass. The semaphore load is issued before the visits and tested after them, so it overlaps the visits on any pass that visits a channel |
| Dispatch | None (unrolled) | Per visited channel: `ctz`, clear lowest bit, index scaling (~9 ALU ops) |
| Send: slot cursor and bounds | Struct fields on the stack (cursor, last, first) | Table entry at `sp + 32·ch` (the same three loads) |
| Send: far ring geometry | 3 stack loads (last, stride, base on wrap) | Compile-time immediates |
| Send: free-slot +1 and credit-counter addresses | Compile-time per channel | Shift and add from `ch` |
| Send: bookkeeping | sent++ | sent++ plus `pending |= bit` (one `or`) |
| Completion frame | far += n, processed += n, credit | Same, plus `processed == sent` and clear the bit |
| Telemetry, perf telemetry, trimming capture | Per send / per outer iteration | Same calls (compile-checked with telemetry on, §7) |

## 3. What was removed

- The per-channel inlined copies of the sender step: 4 on 2D, 7 on VC1 (VC1 steps were not `FORCE_INLINE`, but GCC
  inlined them anyway). They become one body per VC.
- The per-pass idle-channel branches. The ungrouped per-visit register reads of ch4 and of VC1 channels. VC1's
  idle-visit bookkeeping.
- The stack loads of the far ring geometry.
- The VC1 static-connection wait, which was 3,124 B of inlined per-channel spin loops. It is now one 828 B helper.
- E0's instance of `run_fabric_edm_main_loop`, with its receiver-side locals.
- E0's largest stack frame: `kernel_main` + program is below v3a's `kernel_main` on every config. Max over the 168:
  1,392 vs 1,408 B at -O3, 1,472 vs 1,632 B at -O2, 1,392 vs 1,504 B at -Os.

## 4. Ledger ideas used, and why they fit this shape

- **Shared runtime forwarding step** (old bit 27, s80; LOST-CONTEXT). This is that idea, applied to every channel.
  RESULTS gives four causes for the earlier losses; this shape addresses each:
  - §81, peeling into 3 copies at fixed stack offsets: the dispatch is a `ctz` loop over a run-time mask, which cannot
    be peeled into per-channel copies. Checked: one send body per VC at all three levels.
  - §81, idle visits waiting on two stack loads: idle channels are never visited. The test is register-only: the
    grouped value plus the pending word.
  - §81, the worker's counts stayed on the stack while forwarding registers were freed: the worker is a table entry
    like the others. Nothing depends on which counts SRA puts in registers.
  - §36, "each entry spills 6 registers": there is no call boundary. The visit is inlined once per VC. The hook-probe
    build moves the loop-carried state but keeps the shape.
- **Loop function boundaries** (§47, §78). The E0 program is its own function that owns its state, which is what §78
  asked for. Edits to `kernel_main`'s init code cannot reach the loop's allocation.
- **Worker visit latency** (§73, OPEN). The semaphore load is moved ahead of the visits. The check itself stays in every
  pass (rule). This is not §49's variant: the load is not in the bit-23 group, it is issued before the pre-send group.
- **Load groups** (§46/§48). A documented asm primitive for the grouped read (section 6). The send keeps v3a's fences.
- **Forwarding gate** (bits 18/26, MIXED) and **summary doorbell** (OPEN). Both are subsumed. The work mask is a
  per-channel gate with no gate branch and no count, and the grouped read already gives the ready set without producer
  changes.
- **0x7FF mask** (IN) and **bit-19 bit test** (IN), generalised per channel. Power-of-two slot counts use the bit test,
  others the masked compare.
- **Not used**: branch hints, batched pointer updates, several packets per visit, first-level ack for the worker. They
  are protocol or hint changes, outside this design.

## 5. Semantic equivalence (vs v3a, which is s79mc minus the hints)

- **Same channels, same order.** `tx_steps_channel` reproduces v3a's stepping list exactly:
  - ch0 and ch1;
  - ch2–4 in 2D (ch4 when VC0 has five channels);
  - VC1 +0..+3 in 2D under `FABRIC_2D_VC1_SERVICED`.

  Configs where v3a would step a VC1 index with VC0's far ring are excluded. The `ctz` order is ascending, the same as
  v3a's unrolled order. At most one packet per channel per pass. Far slots are shared per VC in the same order.
- **Skipped visits do nothing in v3a.** A channel with no unsent packet and `sent == processed` (and
  `acks == sent` with first-level ack) does nothing observable in v3a:
  - `try_send` returns before any side effect;
  - the completion read is skipped (`has_completions_pending` / `has_acks_pending`);
  - credits are returned only on a non-zero delta.

  `pending` is exactly that predicate. It starts at 0 with every count, is set on every send, and is cleared only when
  a completion (or ack) delta brings processed (and acked) to sent. The channel cannot catch up in its own send visit:
  the counter is read before the packets-received update is issued.
- **Unsent detection.** v3a tests `(v & 0x7FF) != slots`. v3u uses the same test, or bit log2(slots) for power-of-two
  counts. These are equal because the free count never exceeds the slot count; v3a's bit-19 path relies on the same
  fact. VC1 and ch4 values are masked with 0x7FF instead of `(1 << 17) − 1`. That is equivalent because no slot count
  exceeds 0x7FF (static_assert).
- **Send.** The visit is v3a's try_send sequence, statement for statement:
  - the same two fenced groups;
  - the same ordering of `did_something`, the `src_ch_id` stamp, the busy check, the data command, free-slot +1, the
    spin and the packets-received update;
  - the same far-space and bubble rule (`BUBBLE_FLOW_CONTROL_INJECTION_SENDER_CHANNEL_MIN_FREE_SLOTS`).

  Addresses come from the same compile-time sources: the alloc entries that `local_sender_channels.init` and
  `remote_receiver_channels.init` use, the stream-id table and the counter bases.
- **Completions and credits.** Same order: completions (free far slots, credit unless first-level ack), then acks. The
  worker arm calls v3a's `send_credits_to_upstream_workers<…, false>` on a copy of the ch0 interface. The router arm is
  the same `noc_inline_dw_write<REG, posted>` as `notify_persistent_connection_of_free_space_impl`, to the address v3a
  caches in `wait_for_static_connection_to_ready`. That address is computed by the same `get_noc_addr` from the same
  location info, after the same spin, with context switches every 100k polls and the termination check, in channel
  order.
- **Worker connection.** It is checked every pass, after the visits, as v3a checks it after ch0's step. The semaphore
  value is read earlier in the pass, and v3a's open path re-reads it, so a request can be seen at most one pass later.
  This is safe because the states are monotone until the router acts:
  - OPEN stays until the worker closes;
  - CLOSE stays until the router writes UNUSED;
  - the router's UNUSED write precedes the next pass's read in program order.

  The open and teardown bodies are v3a's (`establish_worker_connection`, `teardown_worker_connection<true>`).
  Credits in a visit use the connection state from the start of the pass, as in v3a.
- **Outer loop.** It is a copy of v3a's E0 path:
  - termination and router-command checks, `did_something`, the heartbeat, the idle context switch;
  - BW and perf telemetry, the perf buffer clear at loop entry;
  - RUN/PAUSE with the coordinated context switch.
- **Init and teardown.** v3a's init side effects are kept in `kernel_main` and in the program. Teardown is the same
  call.

## 6. Hardware ordering: `tx_issue_loads`

`tx_issue_loads<A0..A3>(v0..v3)` is one inline-asm block. It emits the `lui` upper parts first, then 1–4 `lw` of the
given compile-time addresses, back to back and in order.

Contract:
- the loads issue on consecutive instructions, so their latencies overlap (§17);
- no instruction that uses a result can be placed between them;
- the block reads and writes nothing else.

Larger groups are consecutive blocks with nothing between them in the source.

It is used for the grouped free-slot read only. That group is new: it is 7 loads on VC1, and its consumers are ALU mask
operations that a scheduler could legally interleave between plain volatile loads (§46). The addresses are immediates,
so the block holds no address registers across the loop.

The send keeps v3a's `atomic_signal_fence` groups unchanged, so the per-packet path stays the code that hardware has
validated. Both are "v3a's fences or one documented primitive", as the brief allows.

## 7. Builds (offline_build.py, `-Werror`)

| Level | Configs built | E0 changed vs v3a | E1 changed vs v3a | Hot E0 (`10733501891889981293`) |
| --- | --- | --- | --- | --- |
| -O3 | 536/536 | 168/208 (median −272 instructions) | 0/328 | 2,274 instructions (−249), largest frame 768 (−528) |
| -O2 | 536/536 | 168/208 (median −274) | 0/328 | |
| -Os | 536/536 (see note) | 168/208 (median −236) | 0/328 | |

- **The -Os note.** The logged -Os commands are garbled for 35 VC1 configs: interleaved MPI log lines; 11 have no
  compile+link pair and 24 fail to compile, for v3a too. So -Os is built from the -O2 commands with `--set-opt Os`.
  On the 500 configs whose -Os commands are intact, this gives the same `.text` as the direct -Os build, for both v3a
  and v3u (500/500 identical). The -Os baseline is v3a built the same way.
- **By category at -O3.** 2D E0 −248 to −1,053 instructions. 1D −17 to +52 and ring −23 to +45 (two channels, so
  nothing to fold).
- **Compile check of paths the logs do not enable.** `ENABLE_FABRIC_TELEMETRY=1`, `FABRIC_TELEMETRY_STATS_MASK=15` and
  `PERF_TELEMETRY_MODE=1` (CT args overridden in the generated header) build on the 2D, VC1, ring and 1D E0s.
  Channel-trimming capture cannot be enabled that way: v3a fails the same way.

## 8. Code size

Largest E0 + largest E1 over the 48 + 48 VC1 kernels, in loadable bytes (`readelf -lW`, sum of LOAD filesz):

| | -O3 | -O2 | -Os |
| --- | ---: | ---: | ---: |
| v3a | 16,996 + 12,708 = 29,704 (does not fit) | 13,920 + 12,232 = 26,152 | 12,344 + 11,660 = 24,004 |
| v3u | 12,788 + 12,708 = **25,496 (fits, 1,128 B spare)** | 11,156 + 12,232 = 23,388 | 10,068 + 11,660 = 21,728 |

E0 loadable bytes, median / max:

| | VC0-only E0s, -O3 | VC0-only E0s, -Os | VC1 E0s, -O3 |
| --- | --- | --- | --- |
| v3a | 8,752 / 12,588 | 7,260 / 9,140 | 15,112 / 16,996 |
| v3u | 8,956 / 10,812 | 6,752 / 8,156 | 11,044 / 12,788 |

The largest VC1 E0 is mostly `kernel_main` init: 6.2 KB on a local-handshake master. The program is 3.2 KB.

## 9. Probes (perturb.py on v3u: one more PRELOOP/LOOP pair, in the E0 program; STATE in `TxChannelState`)

| Probe, -O3 | v3u: E0 / E1 configs changed | Hot E0 | v3a for comparison |
| --- | --- | --- | --- |
| `hook` | 208/208 (+30 median) / 328/328 | +38 instructions, 721 differ; program 566 → 601 | +57, 1,274 differ |
| `statefield` | 208/208 (+13) / 120/328 (single-ERISC) | +17 instructions, 570 differ, frame unchanged | +0, 66 differ |

Both build at -O3.
- **`hook`.** Same structure. `worker_connected` moves to the stack, and the visit constants are rematerialised per
  busy pass instead of spilled.
- **`statefield`.** The table entry grows from 32 to 40 B, so indexing becomes ×40 (shift-add) instead of `<< 5`. This
  is the one probe where v3u moves more than v3a: a run-time-indexed table makes its stride part of the code. Padding
  the entry to a power of two (`alignas`) would hide it, but it forces stack realignment. It was left as is.

## 10. Open risks and what to look at on hardware

1. **E0-bound small writes (E/W and N/S, 16 B; v3a 121.6 / 127.2).**
   - What should help: idle channels are nearly free and the semaphore latency is hidden.
   - What could cost: the dispatch (~9 ALU ops); the out-of-line send at -O3/-O2 (two taken jumps); three
     stores/reloads per busy pass at -O3; a slot cursor reached through `ctz` → index → load instead of a fixed stack slot
     (2–3 cycles on the size-load chain).
   - Net: unknown; this is the row to watch.
2. **-Os.** E0's program is nearly the same code at -Os as at -O3, so E0-bound rows should lose much less than v3a's
   +19 to +28 at -Os. E1-bound rows keep E1's -Os cost, since E1 is v3a.
3. **Ring E0** (first-level ack and bubble flow control) goes through new code: the ack path and the 2-slot rule in the
   runtime body. Run 1D sanity (Linear/Ring) with RELAXED and STRICT flushes.
4. **VC1.** Multi-mesh sanity at -O3 now that it fits, and at -Os (the builder's choice).
5. **Worker open/teardown.** Detection can be one pass later than v3a. Validation exercises connect and close per test.
   Watch for teardown hangs.
6. **Telemetry-on configs** (production default for BW telemetry) only compile-checked.
7. **Untouched by design.** E0 `kernel_main` still builds E1-only objects at init (downstream adapters), as v3a does.
   That is the next size lever if VC1 ever needs more room at -O3.

## 11. Candidates for a second iteration (after measurement)

- If E/W small writes regress: hide the slot-cursor → header-size chain by issuing the entry's cursor load with the
  grouped read for the lowest work bit. Alternatively, keep the per-VC far state and the hot entry's cursor in the visit
  loop's locals across a pass.
- One step body for both VCs (far state per VC in a small array), if VC1 code size matters more than the VC0 path.
- A gate on the VC1 loop when only VC0 has work (already one `andi`/`beqz`, so probably nothing to gain).

## Reproducing

Scratch: `/tmp/claude-4004/-home-aagarwal-tt-metal/76f59f51-9416-4ea6-898e-740976fca7f2/scratchpad/v3u/`.

| What | Path or command |
| --- | --- |
| Builds of v3u | `v2_O3`, `v2_O2`, `v2_Os` |
| Probe builds | `v2hook_O3`, `v2sf_O3` |
| v3a baselines | `base_O3`, `base_O2`, `base_Os2` (the -O2 commands at -Os) |
| All three levels | `build_all.sh <variant> <name>` |
| Builds with overridden CT args | `tools/patched_build.py` |
| Size, stack and code shape | `tools/size.py` (VC1 E0+E1), `tools/e0size.py`, `tools/frames.py`, `tools/dis.py`, `tools/loop.py` |
| -Os from the -O2 commands | `offline_build.py build --variant fabric_exp/variants/v3u --out DIR --opt O2 --set-opt Os` |


---

# Round 1 on hardware, and v3u2

## 12. Where v3u's ~30 cycles/packet go

**Hardware** (job 218, -O3, `p2u` against v3a = `p1nohint`). Validation passes. E1-bound rows are unchanged (E1 is
identical to v3a). Every E0-bound row lands near 152 at every packet size:
- E/W 16 B 121.6 → 153.8, N/S 16 B 127.2 → 153.0;
- forwarded 16 B 111.6 → 141.0;
- 1-hop E/W 4352 132.3 → 151.7, 3-hop 146.8 → 155.0, multicast 140.2 → 167.9.

Jobs 220/221:
- -O2 152.3 and -Os 154.3 on E/W 16 B (v3a: 123.1 and 146.9);
- the hook probe +18 (v3a +4.5);
- VC1 routers 6–15 faster than v3a, and they run at -O3.

**Method.** `tools/walk.py` follows one path through the -O3 disassembly of the hot E/W E0
(`10733501891889981293`), given which branches are taken, and counts instructions, taken branches and loads.

The worker pass is: the worker sends, no completion frame, connected, ch1–3 idle.

| Sending pass (worker) | Instructions | Taken | Loads (stack) | E/W 16 B |
| --- | ---: | ---: | ---: | ---: |
| v3a | 74 | 6 | 21 (10) | 121.6 |
| v3u | 102 | 5 | 21 (6) | 153.8 |
| v3l | 73 | 6 | 20 (9) | 117.2 |

**Budget of v3u against v3a** (one sending pass is ~1 pass/packet; per-cycle costs from §17). The listed items add up to
+28 instructions; the per-item counts are approximate.

| # | Item, biggest first | Extra instructions | ≈ cycles |
| ---: | --- | ---: | ---: |
| 1 | **Run-time channel index in the send.** The entry address `sp + 32·ch` is re-formed 4 times; the completion-counter pointer is `base + 4·ch`; the free-slot update register is `base + 0x1000·ch`. v3a has fixed stack offsets and constant addresses | +11 | +11 |
| 2 | **Visit loop nested inside the pass.** On every pass with work, GCC rematerialises 5 constants on entry and stores 3 live values (loop counter, semaphore value, semaphore pointer), then reloads them on exit | +11 | +11 |
| 3 | **Pre-send group serialised.** The scheduler placed the size's `zext.h` between the CMD and STATUS reads of the fenced group, so STATUS issues only after the L1 size load returns | +1 | +6..7 |
| 4 | **Dispatch** (`ctz`, unsent bit, far test, channel bit, clear) instead of the templated ch0 test | +6 | +6 |
| 5 | **Post-send window lost.** A hoisted, spilled 32-bit constant (the free-slot register base) is reloaded right after the CMD/STATUS/counter group, so it waits for the group (§17: +6..+14). v3a fills that window with ALU work | +1 | +5 |
| 6 | Pending-bit `or`, jump back from the out-of-line send | +2 | +2 |
| 7 | **Saved:** idle forwarding channels are a branch-free mask, against v3a's tests (~10 instructions, 4 taken branches, a stack load) | −1 | −4 |
| 8 | **Saved:** semaphore. v3u's early lone L1 load replaces v3a's stack→L1 chain. A lone load is not hidden by ALU work (§17), so this saves only the stack load | −2 | −2 |
| | **Total** | **+28** | **≈ +34** (measured +32 E/W, +26 N/S) |

The forwarding pass in v3u has the same shape: dispatch → indexed send. The FWD row (+29) pays items 1–6 too. Multicast
(+28) pays items 1 and 4 once per visited channel.

## 13. v3u2

`fabric_exp/variants/v3u2/`. Only `fabric_erisc_router_v3.cpp` differs from v3a. The design keeps v3u's state and
masks, and changes the shape of the pass and of the visits:

1. **The worker channel goes through a compile-time-index instance of the step body, inline in the pass.**
   - Its table entry (fixed stack offsets), counter, stream register and credit path are constants.
   - It is visited when it has an unsent packet or packets in flight.
   - Why the worker gets its own instance: there is one per router; it is the only channel with a connection
     protocol; and every E0-bound flow that starts on the chip sends through it.
2. **The connection check is one compare (v3l's form).** The semaphore is compared with the value it holds while
   nothing changes (unused, or open), and `tx_worker_connection_event` makes `check_worker_connections`' transitions on
   that value.
3. **Every send's first load group also reads the connection semaphore**
   (`tx_load_payload_size_txq_status_and_semaphore`: payload size, CMD, STATUS, semaphore; one asm block, four reads,
   §17's limit).
   - The per-pass check uses that value, so a pass that sends has no separate L1 read for it.
   - A pass that does not send reads the semaphore at the check.
   - The value is at most one pass old, as in v3u, which was validated (job 218).
4. **One test for all forwarding channels.** It is the AND of their free-slot values when their slot counts are an
   equal power of two, plus the pending mask. Only when a forwarding channel has work does the pass compute their bits
   and visit them.
5. **Flat forwarding rounds.**
   - The first forwarding channel with work is visited in the same iteration as the pass; the others take one iteration
     each, lowest channel first.
   - There is no loop nested in an iteration (item 2).
   - A pass is still v3a's pass: each channel with work gets one visit, in v3a's order.
6. **A per-VC forwarding working copy (the answer to item 1 for forwarding).**
   - The VC's forwarding channels share one copy of the state a send uses (cursor, bounds, sent / completed / acked
     counts), at fixed addresses. The visit runs on the copy.
   - The table entry is touched only when the visited channel changes (4 stores back, 6 loads in) and for the upstream
     credit address.
7. **The load groups are ordered by construction (items 3 and 5).**
   - The pre-send group is one asm block, so `zext.h` cannot land between the reads.
   - The send reads the channel's entry fields first and writes the new cursor and sent count once the TX queue accepts
     the packet.
   - The free-slot increment addresses its stream as register block + register index, so no 32-bit constant is hoisted
     and spilled.
   - The post-send group stays fenced plain loads. There the scheduler fills the window with the increment and register
     arithmetic, which it cannot do around an asm block (an asm version produced a `mv` of STATUS right after the
     group).

**Paths on the hot E/W E0, -O3** (`walk.py`):

| | Instructions | Taken | Loads (stack) |
| --- | ---: | ---: | ---: |
| v3u2 worker sending pass | **74** | 6 | 19 (8), no separate semaphore load |
| v3a worker sending pass | 74 | 6 | 21 (10), including the stack→L1 semaphore chain |
| v3l worker sending pass | 73 | 6 | 20 (9), lone semaphore L1 load |
| v3u2 forwarding pass (worker idle, one forwarding channel sends) | 101 | 10 | 18 (7) |
| v3a forwarding pass | 76 | 6 | 21 (10) |
| v3l forwarding pass (caller 49 + out-of-line visit 77) | ~126 | ~11 | ~23 |
| v3u2 idle pass | ~27 | | |
| v3a idle pass | 34 | | |

**Forwarding cost accounting** (the open question: can a run-time-indexed forwarding visit cost what v3a's inline
visits cost?). The v3u2 forwarding pass against v3a's, piece by piece:

| Piece | v3a | v3u2 | Δ |
| --- | ---: | ---: | ---: |
| Loop control and grouped read | 10 | 12 | +2 |
| Idle worker channel (test, completions-pending test) | 9 (2 stack loads used at once) | 7 | −2 |
| Connection check | 5 (stack load → L1 load used at once, ~10 cycles) | 1 (value from the last send's group, no load) | −4 |
| Forwarding selection: v3a's ch1/ch2 idle tests; v3u2's combined test, per-channel bits, round setup | 6 | 21 | **+15** |
| Channel dispatch: v3a's ch3 test; v3u2's `ctz`, clear, unsent bit, far test, working-copy check, bit, counter pointer | 3 | 13 | **+10** |
| The send itself | 43 | 47 | +4 |
| **Total** | 76 | 101 | +25 |

What the table shows:
- **The send is at parity once it runs on a fixed working copy.** 47 instructions against v3a's 43; v3u's indexed send
  was ~60, and v3l's out-of-line one 77. The +4 is the semaphore read added to the group, two stride
  rematerialisations and a register move.
- **The run-time index is now paid once per visit in the dispatch** (counter pointer, channel bit), and once per channel
  change in the swap. A visit with a swap is 77 instructions from `ctz` to the completion test, against 60 without and
  46 for v3a's ch3 test plus send. Multicast with several forwarding channels interleaving pays the swap per visit.
- **The remaining gap is selection: ~34 instructions to find the channel against v3a's ~9.** v3a's tests are unrolled at
  2–4 instructions per channel with the channel known at compile time; that is cheap for 3 channels. The run-time
  selection costs about the same for 3 channels or 7, which is why VC1 routers already gained in v3u.
- **In cycles, v3u2 drops two stalls that v3a pays on the same pass:** the semaphore chain (~10) and the idle worker's
  stack loads (~3). With 4 more taken branches, the estimate for the forwarding E0 pass is v3a +10 to +15 cycles:
  - FWD 16 B ≈ 120–125 (v3a 111.6, v3l 130.9, v3u 141.0), if the forwarding E0 binds;
  - 3-hop and multicast between v3a and v3u.

  These are estimates; only hardware says.
- **Parity on forwarding rows would need selection close to v3a's**, about −15 instructions. The options, none of them
  free:
  - **(a) Bounded compile-time instances for VC0's forwarding channels** (≤4 by the 2D topology), with VC1 at run time.
    Parity by construction, but VC0 code grows by ~0.5 KB per instance. VC1 at -O3 (now 508 B spare) would then need
    E0's init trimmed: E0 builds downstream adapters it never opens.
  - **(b) Skip the combined test on passes that turn out to have forwarding work.** Not knowable before the test; it
    trades −5 on forwarding passes for +6 on worker passes.
  - **(c) A host-side summary of which forwarding channels have an upstream router** (`WAIT_STATIC_CONNECTION` already
    says it). This would shrink the selection masks to live channels. It conflicts with the brief's rule that generic
    targets assume every serviced channel may be live, so it is not used.

**Tried and dropped** (offline, by path counts):

| Variant | Worker pass | Forwarding pass | Why dropped |
| --- | ---: | ---: | --- |
| Worker state as scalar locals | 85 | 107 | Register pressure moved into the forwarding send |
| No working copy (indexed send) | 72 | 107 | Forwarding send ~62 |
| Working copy also holding the counter/stream addresses and the bit | 79 | 106 | Constants spilled |
| Per-field arrays for the table | — | — | Stride spilled on the worker path |

None of these changes the design's semantics. They differ only in what GCC kept in registers. The chosen version is the
one with the worker pass at v3l's level and the shortest forwarding pass among those.

**Semantic equivalence (additions to section 5).**
- **Pass order is v3a's:** grouped read, worker visit, connection check, then forwarding channels in increasing
  channel order, one visit each per pass.
- **Outer-loop cadence.** Forwarding visits after the first in a pass take extra iterations, so outer-loop checks
  (termination, commands, heartbeat, idle context switch) run every 32 iterations. That is more often than every 32
  passes under multi-channel load, and the same when idle.
- **Stale round state is safe.** Carried round state can only understate what is unsent, because only E0 consumes
  packets.
- **The working copy holds the authoritative state of exactly one forwarding channel per VC.** That channel's table
  entry is not read while it is held: only the swap reads entries, and only for a different channel. The fields the
  copy holds are the only ones a visit changes.
- **The one-compare connection check makes `check_worker_connections`' transitions on a value read in this pass or the
  previous one.** OPEN and CLOSE are stable until the router acts, and the router's UNUSED write precedes every later
  read.
- **The worker's credits use the connection state as of the start of its visit** (`expected == open`), as in v3a.

**Builds** (-Werror; -Os from the -O2 commands, as before).

| Level | Built | E0 changed vs v3a (median / max instructions) | E1 changed vs v3a |
| --- | --- | --- | --- |
| -O3 | 536/536 | 168/208 (−100 / −898) | 0/328 |
| -O2 | 536/536 | 168/208 (−113 / −523) | 0/328 |
| -Os | 536/536 | 168/208 (−124 / −417) | 0/328 |

- **Speedy E0s:** the 40 speedy E0s are identical to v3a.
- **By category at -O3:**
  - 2D E0s: −75 to −898 instructions;
  - 1D: +78 to +140; ring: +88 to +164 (the working copy and swap on two-channel routers).
- **Program size (hot / VC1 / ring / 1D), instructions:** 739 / 949 / 674 / 625 at -O3, 627 / 841 / 563 / 530 at -Os.
- **Send bodies:** 2 on VC0 configs, 3 on VC1, at every level.

**Code size.** Largest E0 + largest E1 over the VC1 kernels, loadable bytes:

| Level | v3u2 |
| --- | --- |
| -O3 | 13,408 + 12,708 = **26,116 (fits, 508 B spare)** |
| -O2 | 11,780 + 12,232 = 24,012 |
| -Os | 10,652 + 11,660 = 22,312 |

**Stack.** E0 `kernel_main` + program, max over the 168 configs: 1,456 / 1,552 / 1,488 B at -O3 / -O2 / -Os. v3a is
1,408 / 1,632 / 1,504 B.

**Probes (-O3).**
- **`hook`:** E0 +73 median / +94 max instructions (v3a +44 / +75); E1 as v3a.
  - Hot worker pass 74 → 81 instructions: the hook's compare plus one stack reload. The pass keeps its shape, and no
    load lands inside a group.
  - Hardware will say whether the cost is nearer v3a's +4.5 or v3u's +18.
- **`statefield`:** E0 +0 instructions (59 differ, offsets only).

**Risks.**
1. **Forwarding rows** (estimates above):
   - FWD and 3-hop should improve a lot on v3u but stay above v3a.
   - Multicast pays a swap whenever consecutive forwarding visits change channel.
2. **VC1 fits at -O3 with 508 B.** Any E1 growth, or option (a), needs E0's init trimmed first.
3. **The semaphore read in the pre-send group** puts a second L1 read in that group. §17 says four reads overlap; §49's
   different placement was mixed. Watch the E/W 16 B row against v3l's 117.2.
4. **More iterations per pass under multi-channel load** means more frequent outer checks: a small fixed cost per 32
   iterations.
5. **The 1D and ring E0s grew** (working copy and swap on two-channel routers). Their rows were credit-loop or E1 bound
   in v3a; the 1D suite will show it.

Scratch:
- builds `scratchpad/v3u/f2_O3`, `f2_O2`, `f2_Os`, probes `f2hook_O3`, `f2sf_O3`;
- final source snapshot `snap/u6_final.cpp`, earlier variants `snap/u3.cpp`, `u4.cpp`, `u5.cpp`;
- path tool `tools/walk.py`; listings `f2_hot.txt`, `v3a_hot.txt`, `v3u_hot.txt`, `v3l_hot.txt`, `v3l_fwd.txt`.

---

# Round 2 on hardware, and v3u3

## 14. v3u3

`fabric_exp/variants/v3u3/`. Only `fabric_erisc_router_v3.cpp` differs from v3a, and from v3u2 in two places. Pass
order, masks, the worker's compile-time instance, the connection check, the load-group primitives and the flat rounds
are v3u2's.

The two points from v3u2's hardware run:
- **Multicast +29 (also at -O2 and -Os).** The working-copy swap is paid whenever consecutive forwarding visits change
  channel.
- **The worker row moved with the level** (E/W 16 B 116.8 at -O3, 121.2 at -O2). v3u2 was chosen among variants that
  differed only in what GCC kept in registers, and that choice did not hold at -O2.

### 14.1 What changed

1. **The grouped free-slot read accounts each value into three scalars as it is read.** These are the worker's idle
   bit, the forwarding idle bits, and the AND of the forwarding values (`tx_account_free_slots`, called in
   `tx_read_free_slots` after the `tx_issue_loads` blocks, which still issue back to back).
   - v3u2 collected the values in a `std::array` (`TxFreeSlots`) and folded it with lambdas (`tx_idle_bits`,
     `tx_forwarding_and`). Whether that array became registers depended on GCC splitting the aggregate.
     - At -O2 the hot E/W E0 stored the four values to the stack right after the group and reloaded three of them.
     - The VC1 E0 did this at both levels (seven stores).
   - There is no aggregate now, so nothing depends on that split.
2. **Forwarding state is read and written in place in the table: no working copy, no swap.**
   - A forwarding visit reads its channel's entry through the run-time index (`channels[ch]`) and writes the new
     cursor and counts back to it. The worker's visit uses the same table at fixed offsets, as before.
   - The entry is the only copy of a channel's state. A visit has no branch, load or store that depends on which
     channel the previous visit served, so alternating channels costs nothing beyond the visits themselves.
   - Removed: `TxForwardingWorkingCopy`, `tx_working_copy_for`, and the second state parameter of `tx_visit_channel`
     and `tx_return_credits`.

### 14.2 -O3 against -O2 on the hot paths

`walk.py` paths. "Differ" is a line diff of the two paths with register names normalized; stack offsets are kept.

| Path | v3u2 -O3 | v3u2 -O2 | differ | v3u3 -O3 | v3u3 -O2 | differ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| E/W E0 worker sending pass | 74 / 6 / 19 (8) | 79 / 5 / 20 (9) | 19 | **72 / 5 / 17 (6)** | 72 / 5 / 17 (6) | **0** |
| E/W E0 forwarding pass (worker idle, one forwarding send) | 101 / 10 / 18 (7) | 107 / 10 / 19 (8) | 31 | **109 / 10 / 19 (4)** | 109 / 10 / 19 (4) | **0** |
| E/W E0 forwarding visit (`ctz` to completion test) | 60 same channel, 77 changed | | | **71 either way** | 71 | **0** |
| E/W E0 idle pass | ~27 | | | 26 / 5 / 5 (0) | 26 / 5 / 5 (0) | 0 |
| VC1 E0 worker sending pass | 103 / 6 / 26 (12) | 104 / 6 / 27 (13) | 22 | **88 / 6 / 20 (6)** | 88 / 6 / 20 (6) | **0** |

Cells are instructions / taken branches / loads (of which stack-relative).

- **v3u2's -O2 difference is exactly the aggregate.**
  - The worker pass's 19 differing lines are the four stores after the group, three reloads before the forwarding
    test, and the constants that moved as a result.
  - The VC1 E0 stores seven values at both levels, so its worker pass is 103 even at -O3.
- **v3u3's worker, forwarding and idle passes are line-for-line the same at -O3 and -O2.** The VC1 E0's worker pass
  is the same too.
- **The `hook` probe shows the same property at both levels** (worker pass 79 at -O3 and -O2, one stack offset differs).
- **Nothing was given up at -O3.** The worker pass is 72, against v3u2's 74 and v3l's 73.
  - Of the two changes, the scalar reader alone already made v3u2's paths level-stable: with the working copy kept,
    the worker pass is 78 / 78 and the forwarding pass 104 / 104.
  - Without the working copy the worker pass drops to 72 at both levels, with 6 stack loads instead of 10.
- **-Os has its own shape**, as for every design so far (busy-wait and wrap blocks placed differently). Its rows were
  E1-bound in v3u2.

### 14.3 Forwarding: what a visit costs now

From `ctz` to the completion test on the hot E/W E0 at -O3 (the same at -O2):

| | v3a (ch3 test + send) | v3u2, same channel as last visit | v3u2, channel changed | v3u3, any order |
| --- | ---: | ---: | ---: | ---: |
| Instructions | 46 | 60 | 77 | **71** |
| Loads | | 13 | 18 | 15 |
| Stores | | 10 | 15 | 13 |
| Extra work when the channel changes | none | none | swap: +17 instructions, +5 loads, +5 stores | **none** |

**Why the same-channel visit is 11 longer than v3u2's.** In v3u2 the working copy was a local struct, so GCC kept the
held channel's cursor and sent count in registers. A swap moved them between those registers and the table; v3a's
per-channel inline state works the same way. Keeping the state in the table puts it in memory:

| Item | Δ vs v3u2 same channel |
| --- | ---: |
| Entry address `sp + 64 + 32·ch`, formed once for the cursor/bound loads and again for the counts (GCC recomputes it rather than keep it live) | +7 |
| Cursor and sent count loaded from the entry and stored back after the send, not held in registers | +4 (2 loads in the entry group, 2 stores) |
| Payload size kept on the stack across the TX-queue busy test | +2 |
| No working-copy compare | −1 |
| Other scheduling | −1 |
| **Total** | **+11** |

**What it means per row** (estimates; hardware decides):
- **Multicast, 3-hop, and anything that interleaves forwarding channels.**
  - Every changed-channel visit loses the swap: +17 instructions, +5 loads, +5 stores.
  - It pays the +11 above instead: net −6 instructions, −3 loads and −2 stores per alternating visit.
  - I expect multicast to recover part of v3u2's +29, not all of it.
- **Remaining gap to v3a on forwarding visits: +25 instructions per visit (71 against 46).** Alternation no longer
  changes it.
  - Dispatch (`ctz`, clear, unsent bit, far-space test, channel bit, counter address): about +10.
  - Entry addressing: +7.
  - State in memory: +4.
  - Payload spill: +2.
  - The rest is scheduling.
  - v3a pays none of this because each channel is its own inline copy with constant addresses. That is the
    channel-count growth the brief rules out.
- **FWD (one upstream channel, so v3u2 never swapped).** The forwarding pass is 109 instructions against v3u2's 101 at
  -O3, and against 107 at -O2, where v3u2 measured 117.3. It has 4 stack loads against 7, so expect FWD about where
  v3u2 was at -O2, a few cycles above v3u2's -O3 112.8.

**Tried and dropped this round** (offline, by path counts):

| Variant | Worker pass | Forwarding pass | Visit | Why dropped |
| --- | ---: | ---: | ---: | --- |
| Scalar reader, working copy kept | 78 | 104 | 63 same / 79 changed | Keeps the swap; worker pass 6 longer than without the copy |
| One array per state field (structure of arrays) | — | — | 76 | Each array is indexed separately (two `sh2add`s, more spills); worse than one entry |
| Compile-time index when a VC has one forwarding channel (1D, ring) | = | = | = | Hot and VC1 E0s unchanged; 1D −1 / ring −18 instructions at -O3, 0 / +5 at -O2. Not worth a second dispatch form |

### 14.4 Builds, size, stack, probes

**Builds** (-Werror; -Os from the -O2 commands, as before): 536/536 at each level.

| Level | E0 changed vs v3a (median / max instructions) | E1 changed vs v3a |
| --- | --- | --- |
| -O3 | 168/208 (−128 / −969) | 0/328 |
| -O2 | 168/208 (−128 / −598) | 0/328 |
| -Os | 168/208 (−141 / −502) | 0/328 |

- **Speedy E0s:** the 40 speedy E0s are identical to v3a.
- **By category at -O3**, instructions vs v3a:
  - 2D: −104 to −356;
  - 2D VC1: −677 to −969;
  - 1D: +80 to +162 (v3u2 +78 to +140);
  - ring: +120 to +205 (v3u2 +88 to +164).
  - At -Os every category except speedy (identical) is below v3a.
- **Program size (`run_uniform_tx_program`, hot / VC1 / ring / 1D), instructions:**
  - -O3: 710 / 878 / 715 / 647 (v3u2 739 / 949 / 674 / 625);
  - -O2: 683 / 854 / 678 / 615;
  - -Os: 607 / 779 / 549 / 515.
- **Send bodies:** 2 on VC0 configs, 3 on VC1, at every level. Code size still does not grow with channel count.

**Code size.** Largest E0 + largest E1 over the VC1 kernels, loadable bytes:

| Level | v3u3 | v3u2 | v3a |
| --- | --- | --- | --- |
| -O3 | 13,124 + 12,708 = **25,832 (fits, 792 B spare)** | 26,116 | 29,704 (does not fit) |
| -O2 | 11,480 + 12,232 = **23,712** | 24,012 | 26,152 |
| -Os | 10,312 + 11,660 = **21,972** | 22,312 | |

**Stack.** E0 `kernel_main` + program, max over the 168 configs:
- v3u3: 1,392 / 1,472 / 1,408 B at -O3 / -O2 / -Os;
- v3u2: 1,456 / 1,552 / 1,488 B;
- v3a: 1,408 / 1,632 / 1,504 B.

**Probes.**
- **`hook`** (-O3): E0 +58 median / +82 max instructions (v3u2 +73 / +94, v3a +44 / +75); E1 as v3a.
  - Hot worker pass 72 → 79: the hook's compare and branch, and three constants rematerialized.
  - At -O2: +53 / +83, worker pass 79, the same lines as -O3 except one stack offset.
- **`statefield`** (-O3): E0 +5 / +6 instructions on 128 of 208 configs (v3u2 +0).
  - The entry grows from 32 to 40 B, so the run-time index is scaled by 40 (`sh2add` + `sh3add`) instead of a shift.
  - The worker pass is unchanged (72, identical). The forwarding pass is still 109; 19 lines differ, all index
    scaling and offsets.

### 14.5 Risks

1. **FWD rows** (one upstream channel) pay +11 instructions, +2 loads and +3 stores per visit against v3u2's held copy. The estimate is about
   v3u2's -O2 level at both levels.
2. **Multicast should improve, not close.** About +25 instructions per forwarding visit against v3a remain, all from
   run-time indexing; this round removed only the alternation cost.
3. **The entry stride is now on the forwarding path.** Entry sizes of 2^k or {3, 5, 9}·2^k bytes scale with one or two
   Zba instructions (32, 40, 48, 64 B). A field that made it 56 B would cost more per address.
4. **The scalar reader and the in-place state are structural.** Neither relies on GCC splitting an aggregate or keeping
   a copy in registers, and the hot paths are the same at -O3 and -O2. -Os keeps its own shape.
5. **Unchanged from v3u2:** the semaphore read in the pre-send group, the 32-iteration outer cadence under multi-channel
   load, and 1D / ring E0 growth at -O3. The 1D / ring growth is now from in-place state rather than the copy, and it
   was credit-loop / E1 bound in v3a.

Scratch:
- builds `scratchpad/v3u/g3_O3`, `g3_O2`, `g3_Os`, probes `g3hook_O3`, `g3hook_O2`, `g3sf_O3`;
- source snapshot `snap/g3_indexed.cpp`;
- path listings `var/u3_O3.txt`, `var/u3_O2.txt` (hot E0), `var/g3vc1_O3.txt`, `var/g3vc1_O2.txt` (VC1 E0), `var/f2_O2.txt`,
  `var/f2vc1_O3.txt`, `var/f2vc1_O2.txt`;
- walked paths `var/p_*.txt`;
- path diff `tools/pathdiff.py A B` (two `walk.py -v` outputs, registers normalized);
- category deltas `tools/cat_delta.py BASE NEW...`;
- dropped experiments `var/v1` (working copy kept), `var/v4` (per-field arrays).
