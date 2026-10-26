# E0 audit: the lead's sender_erisc against what E1 learned (2026-10-26)

The lead is `fabric_exp/variants/v3v6` (RESULTS §98). Its E0 (`sender_erisc` in
`fabric_erisc_router_v3_sender_erisc.hpp`) is v3tk's, which is v3sb's, which is v3l3b's E0 renamed (round 11). Every
Phase 1 improvement since then was on E1. This note checks E0 for the classes of problem that were found and fixed on
E1, and proposes candidates. Offline only: no device runs, no tree edits. Hardware decides; everything here that is not
an offline check result is marked as inferred.

## Summary

Findings, ranked by expected value:
1. **A freed far slot is used one pass late** (§2.3). When a worker packet waits for room in the far receiver ring, the
   pass tests for room before it reads the completion counter, so the room a credit frame frees is used on the next pass.
   That adds the rest of the pass plus the next pass's head, about 30–60 cycles, to every credit frame's first send. It
   is not a race but a fixed pass of latency whose size depends on how long the rest of the pass is. **Candidate:** the
   worker visit takes completions before its send decision when it waits for room (v3yc on v3v6, v3yb on v3y).
   Offline: the credit-frame-to-send lag goes from 1 pass to 0 on all five hot E0 configs at all levels. **Inferred:**
   up to ~6–7 cycles/packet on rows bound by an 8-slot far ring (the VC0 rows of VC1 routers, worker channel 51 slots),
   up to ~2–3 on 16-slot 2D rings, nothing where the worker's own window or E1 binds.
2. **E0's loop function still holds its once-per-round and rare work** (§1). It is already insulated from cold code
   (coldmul, init, templates, os never DIFF), but an edit to the per-round code itself re-lays out the whole pass loop:
   one more termination test changes 600–1,600 lines of `run_main_loop` and moves pass paths by up to +14 in the proxy
   score. That is §97's P1 failure mode on E0. **Candidate:** v3y, per-round and rare work out of line as on E1.
   Offline: the same edits leave v3y's loop "moved" (identical code). Hardware effect unknown: the pass code is
   re-allocated (2D worker send +5 score at -O3, VC1 worker send −6 with 11.7 → 3.7 stack loads per pass).
3. **The VC0 combined forwarding test** (§6, v3v6l3): round 11's v3l3 A/B, blocked by a VC1 hang that turned out to be
   the test race. E0 PT_LOAD-identical to v3l3's. Proxy: forwarded passes −3 to −8, worker-only and idle passes +6 to
   +16.
4. **Register access** (§4): E0's NoC programming already goes through the API (`noc_inline_dw_write`), but its TX-queue
   and stream-register accesses were hand-written. v3y uses the existing helpers; PT_LOAD-identical to v3v6 on every
   config at every level (verified with an API-only control), so it needs no hardware run.
5. **No E1-style timing cliff on E0** (§2): every condition E0 samples is re-sampled on the next pass, and its passes
   are shortest when credits bind. The perturbation sweep (a) shifted E0's credit arrivals by 8–32 cycles with smooth
   results (RESULTS §98). The one deliberate cliff is the idle context switch, counted in rounds rather than time
   (§2.8; proposal only).
6. **Credits** (§3): E0 holds nothing. Every visit that sees new completions credits the worker or the upstream router
   in that visit. The holds that matter for bursty traffic are E1's (v3x).

Candidates (variant dirs, E0 only; E1 PT_LOAD-identical to v3v6's on every config):

| Dir | E0 change | Built on |
| --- | --- | --- |
| `v3v6l3` | VC0's combined forwarding test dropped (round 11's v3l3) | v3v6 |
| `v3y` | Once-per-round and rare work out of line; TX-queue and stream registers through the provided helpers | v3v6 |
| `v3yb` | v3y + the worker visit takes completions first when it waits for far room | v3y |
| `v3yc` | v3v6 + the helpers + the same worker change, without v3y's restructuring | v3v6 |

v3yc − v3v6 measures the worker change (the helpers are codegen-neutral); v3yb − v3yc measures the restructuring with
it in; v3y − v3v6 measures the restructuring alone. Run list: §10.

## 0. Method

- **Hot E0 configs:** 2D E/W `10733501891889981293` (channels {17,16,16,16}, far ring 16), 2D N/S `8414232161770541884`,
  1D line `10744374302904868627` ({16,16}, far ring 32), 1D ring `16671252331734229980` (first-level ack, deadlock
  avoidance), and the largest VC1 E0 `1526198486225049198` (VC0 {51,4,4,4} with an 8-slot far ring; VC1 {2,2,2} with a
  4-slot far ring). Built at -O3, -O2 and -Os (VC1 at -Os from its -O2 command).
- **Code comparison:** `scratch_tools/v3t/fncmp.py --fn sender_erisc` (same / moved / DIFF per function).
- **Per-pass counts:** an RV32 emulator runs `sender_erisc::run_main_loop` from the ELF
  (`scratch_tools/e0audit/emu2.py`, extended from v3l's `emu.py`; `paths.py` makes the tables). Device model: a busy
  channel's free-slots register always shows one packet; completion (and, on the ring, first-level ack) counters catch up
  every second pass, i.e. 0.5 credit frames per packet; TX queue and NoC command buffers never busy. 64 measured passes =
  two rounds, so per-round work is amortized at 1/32. Reported as instructions / taken branches / loads per pass, and the
  V3L §11 score = instructions + 2 × taken + loads. The "wait" scenario lets counters catch up only every K passes
  (K = 20, 40 on 1D, 12 on VC1), so the far ring fills, and reports how many passes after a credit frame lands the next
  send happens. The score is a proxy that matched one E0-bound row (V3L §11); it says where code moved, not cycles.
- **Probes** (`scratch_tools/e0audit/probes.py`), two edits a contributor might make to E0's per-round or rare code:
  - `term`: the round test also stops on `GRACEFULLY_TERMINATE` (one more load and compare per round, in
    `continue_running_main_run_loop`);
  - `ctxsw`: a debug word written when the idle context switch fires.

## 1. Structure

**What v3v6's `run_main_loop` holds.** It is `noinline` and owns its state (v3l, §86). Besides the pass loop it holds
the router-command switch, the per-round tests and the rare paths. Disassembly, -O3, 2D E/W config, 818 instructions at
0x7ab0–0x8778:
- 0x7c00–0x7c2c, per round: the termination test (`lw a5,268(zero)`, the launch mailbox's read pointer, `sh3add` +
  `slli` for the 144-byte entries, `lbu 379(a5)`, the termination word) and the router-command test (`lw a5,848(…)`).
- 0x7c30–0x7cd4, rare: PAUSE as context-switch master (stream-30 scratch writes, two spin loops, `jal
  execute_pause_command`).
- 0x7d84–0x7de4, per round: s10 and s11 saved to the frame each round and restored at 0x816c, and the round's sent-count
  snapshot (`lw a5,148(sp); sw a5,52(sp)`, three channels).
- 0x7de8–0x80f8: the pass (back edge `bnez a6,7de8`).
- 0x80fc–0x82b4, per round: heartbeat (`andi a5,s4,63`), the yield flag (`lbu a5,220(sp)`), the sent-count sums,
  `li a5,32; bltu a5,t2,8668` (SWITCH_INTERVAL).
- 0x8668–0x8774, rare: the idle context switch as master, inline: scratch handshake spins, the wall clock
  (0xffb121f0/f8), a `jalr` into the firmware's link-status check (debounced by a stored timestamp).

**Cold edits do not reach it** (5 hot configs × 3 levels, against v3v6's own build):

| Edit | v3v6 E0 `run_main_loop` |
| --- | --- |
| coldmul, init, os | moved (identical code), or same at -Os for init and os |
| templates, reorder | same |
| hb (heartbeat every 128 rounds) | DIFF, 2 lines: the `andi` immediate |
| statefield (a field in its own channel record) | DIFF (its own state) |

So E0 does not show E1's cold-code sensitivity (§97's P1 cells changed under coldmul, templates and os).

**Edits to the per-round code do reach it.** The probes:

| Probe | v3v6 | v3y / v3yb |
| --- | --- | --- |
| `term` | DIFF everywhere, a whole-function re-layout: W 818 → 832 instructions with 1,208 differing lines at -O3; 1D 746, ring 661, VC1 1,525 lines; -O2 724–1,593; -Os 295–1,025 | `run_main_loop` moved; only `next_round` changes (19 → 20 instructions) |
| `ctxsw` | DIFF, 13–19 lines (local) | moved; only `end_round` changes |

The `term` re-layout moves the pass paths (proxy score, -O3): 1D worker send 97 → 105, ring worker send 104 → 113,
ring worker + forwarded 202 → 216, 2D worker send 110 → 109 (2 more instructions, 4 fewer loads). On v3y the same edit
moves no pass path by more than 0.1 (the amortized call). v3yc keeps v3v6's structure, and its probes DIFF like
v3v6's.

**v3y.** As on E1 (v3t, §97): `termination_requested()`, `next_round()` (termination + pending router command, per
round), `end_round()` (heartbeat and the idle context switch with its handshake, per round) and `run_router_command()`
(PAUSE) are small `noinline` functions; `run_main_loop` keeps the pass loop, the round's sent-count snapshot and the
yield flag (they read the loop's state), and telemetry, which is compile-time off in every logged config (as on E1). The
worker connection check stays in every pass; only its open/close is out of line (`worker_connection_event`, unchanged).
Sizes, -O3 2D: `run_main_loop` 818 → 656 instructions, plus 11 + 19 + 107 + 43 in the helpers (VC1: 1,264 → 1,086).
The PAUSE switch keeps v3v6's semantics: the command is read once, RUN runs rounds, PAUSE runs the handshake, anything
else asserts.

What the restructuring does to the pass code is an allocation outcome, not a design property: at -O3 the 2D worker
send rematerializes the slot stride and bounds (`lui`/`addi`) where v3v6 reloaded them from stack slots (+7
instructions, −3 loads), and on the VC1 E0 the worker slot and semaphore pointer move from stack slots into registers
(11.7 → 3.7 stack loads per worker pass). §8 has the per-path table.

## 2. Timing cliffs

What E0 samples once per pass, and whether a small shift moves work by a pass or more:

| # | Condition | Sampled | Step possible? | Size | Robust form |
| --- | --- | --- | --- | --- | --- |
| 1 | Worker packet arrival (free-slots register) | grouped read at the top of the pass | yes, ≤ 1 pass | idle pass 20–27 instructions (−O3 2D: 25 / 4 taken / 7 loads, ~35–45 cycles); sending pass ~80 | inherent to polling; short passes keep it small. Nothing to change |
| 2 | Completion counters (far E1 over Ethernet) | in the post-send group on a sending pass, otherwise a plain read while packets are in flight | yes, ≤ 1 pass | as 1 | self-limiting: when credits bind, E0 has nothing to send and runs idle passes, so it polls fastest exactly then. Sweep (a) of §98 shifted these arrivals by 8–32 cycles: smooth slopes, no step (v3tk / v3v6, E0 = the lead's) |
| 3 | Far-ring room for a waiting packet | the room test comes before the completion read in the same pass | not a race: always one pass late | rest of the pass + next pass's head: ~20 instructions plus the next pass's register and stack loads on 2D (~30–45 cycles), ~33 plus two read groups on VC1 (~45–60) | take completions before the send decision when waiting for room (§2.3) |
| 4 | TX queue busy before a send | pre-send group | yields to the next pass | 1 pass; never seen busy (§11) | none needed |
| 5 | TX queue busy after a send | post-send group | inline spin | hardware | none |
| 6 | Forwarding channels (free slots, in-flight bit) | grouped read, per-VC test | ≤ 1 pass | as 1 | as 1 |
| 7 | Worker connection semaphore | every pass (rule) | ≤ 1 pass | as 1 | rule |
| 8 | Idle context switch | per round | **yes, a threshold** | see §2.8 | proposal |
| 9 | Round boundary | every 32 passes | a packet arriving then waits for the per-round work | ~70 instructions (v3v6 idle round-boundary pass 97 vs 23) | same in v3y (calls) |

### 2.3 The waiting packet (finding 1)

In v3v6's worker visit (`sender_erisc.hpp`, the `has_worker` block) the send condition is `far_has_room && has_unsent`,
and the completion read comes after it (`if (!sent_packet && sent != completed)`). With the far ring full and a packet
waiting, the pass that sees a credit frame takes it, credits the worker and ends; the send happens on the next pass.
Emulator, 2D -O3, wait scenario (instructions/sends per pass), v3v6: `27/0 27/0 27/0 55/0 64/1 65/1 …` — the
55-instruction pass takes the frame and credits the worker without sending. v3yc: `28/0 28/0 28/0 28/0 92/1 61/1 …` —
the frame's pass also sends (v3yb: `32/0 … 101/1 68/1`).

| Hot config | v3v6 | v3v6l3 | v3y | v3yb | v3yc |
| --- | ---: | ---: | ---: | ---: | ---: |
| passes from a credit frame to the send, every config and level | 1 | 1 | 1 | 0 | 0 |

**When it binds** (inferred): the far ring is the binding window only if the channel has more packets to send than the
ring has room, i.e. its own window is deeper than the far ring or several flows share the ring. The builder gives the
worker channel the spare slots: 51 slots against an 8-slot far ring on VC1 routers, 17 against 16 on 2D, 16 against 32 on
1D (never binds there for one flow). Forwarding channels have at most the far ring's depth in every layout (16/16,
4/8, 2/4), so one forwarded flow waits on its own channel's credits. Per packet, when the far ring binds, every slot's
loop is shorter by the saving: about 30–45/16 ≈ 2–3 cycles on 2D and 45–60/8 ≈ 6–7 on VC1 routers' VC0 rows, as upper
bounds. The VC1-router 1-hop rows (199–214 cycles/packet, against E0 passes that score ~130–145) look bound by their
8-slot far ring, so they are where to look.

**The change** (v3yb, v3yc; worker visit only):

```cpp
if (has_unsent_packet) {
    // Waiting for far room: take the completions first (see the top of this file).
    if (!far_has_room<VC0_RECEIVER_CHANNEL, 1u>(far_vc0, 0) && worker_sent != worker_completed) {
        const int32_t freed = new_completions(0, worker_completed);
        if (freed) {
            far_vc0.free_slots += freed;
            worker_completed += freed;
            if constexpr (!worker_first_level_ack) {
                credit_worker(worker, worker_completed, connected);
            }
        }
    }
    if (far_has_room<VC0_RECEIVER_CHANNEL, 1u>(far_vc0, 0)) {
        sent_packet = send_packet<...>(...);   // unchanged; its post-send group reads the counter again
    }
}
if (!sent_packet && worker_sent != worker_completed) { completions = new_completions(0, worker_completed); }
// unchanged from here: take `completions`, credit, first-level acks, connection check
```

- Same protocol: the same credits as absolute counts, the same packets. The worker's credit for the early completions
  goes before the send (so the worker's refill is not delayed by the send); a second credit write follows only if the
  post-send read finds more. First-level ack (ring) keeps crediting on acks; the early read only frees far room.
- A waiting pass with no new frame reads the counter twice (early, then late): ~4 instructions and one L1 read on a
  pass that has nothing to send.
- Forms tried and not used (offline): the same in every forwarding visit (`mk_v3yb_allch.py`, `mk_v3yb_formC.py`):
  -Os forwarded passes +8 instructions and +3 taken branches (score +14), VC1 E0 +1.2 KB (E0 + E1 26,304 B at -O3, 320 B
  under the limit). A flat condition instead of the nested one (`mk_v3yb` history): at -Os every pass, idle included,
  paid one more taken branch (+4 to +6 score); the nested form keeps idle and forwarded passes at v3v6's.

### 2.8 The idle context switch (deliberate cliff, proposal only)

Every logged E0 has `ENABLE_CONTEXT_SWITCH = 1`, `IDLE_CONTEXT_SWITCHING = 1` and `SWITCH_INTERVAL = 32`, so after 34
rounds without a send attempt E0 runs the coordinated context switch: a scratch-register handshake with E1 (which
answers at its next round end and then waits) and the firmware's link-status check, debounced to about once per 10⁹
cycles. A packet arriving during the handshake waits for it. The threshold is a count of passes, about 34 × 32 idle
passes ≈ 35k–50k cycles at today's idle pass, so it moves with E0's idle-pass length (v3v6l3's idle pass is 6–16 score
longer). It is the generic loop's and main's behavior; a cycle threshold would make it a function of time. Not built:
it changes the protocol's timing and the host-set meaning of SWITCH_INTERVAL. Burst gaps in `exp_burst.yaml` (500 and
2,000 cycles) are far below it.

## 3. Credits toward the worker and upstream routers

- `credit_worker`: in every pass that sees new completions (or acks, with first-level ack), one `noc_inline_dw_write` of
  the absolute count to the worker's L1, or to the connection info while disconnected. No batching, no hold.
- `credit_upstream_router`: in every forwarding visit with new completions, `+n` to the upstream router's free-slots
  register. No batching, no hold.
- The far E1 sends credit frames for up to T slots (v3v6: 4; v3x: 1 below 8 slots, else min(4, slots/4)); E0 passes each
  frame on within the pass that sees it. So E0 adds at most one pass to any credit and holds nothing relative to any
  window: rule R2 holds on E0 by construction, and bursty or short senders see only E1's T.
- `enable_read_counter_update_noc_flush` is false: the worker credit write does not wait for an ack.

## 4. Register access and the NoC API

| Access in E0's loop | v3v6 | v3y (and v3yb, v3yc) |
| --- | --- | --- |
| Worker and upstream credits | `noc_inline_dw_write` (API) | unchanged |
| Data command | `internal_::eth_send_packet_bytes_unsafe` (API) | unchanged |
| TX-queue busy spin | `internal_::eth_txq_is_busy` (API) | unchanged |
| TX-queue CMD and STATUS reads in the pre- and post-send groups | `txq_regs[ETH_TXQ_CMD >> 2]`, `txq_regs[ETH_TXQ_STATUS >> 2]` on a hand-made pointer | `eth_txq_reg_read(sender_txq_id, ETH_TXQ_CMD / ETH_TXQ_STATUS)` |
| Far packets-received +1 | three hand-written TX-queue stores (DEST_ADDR, REMOTE_REG_DATA, CMD = START_REG) | `remote_update_ptr_val<packets_received_stream, sender_txq_id>(1)` (`eth_write_remote_reg_no_txq_check`) |
| Local free-slots +1 | a store to a computed `STREAM_REG_ADDR` (VC base + offset, contiguity static_assert) | `increment_local_update_ptr_val<free_slots_stream<CH>>(1)`; the per-channel stream id is a compile-time constant, so the contiguity machinery is gone |
| Grouped free-slots reads (shared `.cpp`, also the generic loop's) | a raw pointer to `STREAM_REG_ADDR(…SPACE_AVAILABLE…)` | `NOC_STREAM_READ_REG(...)`, unmasked (`get_ptr_val` would mask at once, which the grouped read must not) |
| Context-switch handshake | `read/write_stream_scratch_register` (API) | unchanged |
| Completion and ack counters, semaphore, heartbeat, termination word | L1 words | unchanged |

The grouped TX-queue status read cannot use `eth_txq_is_busy`, which reads and tests at once; `eth_txq_reg_read` is the
existing accessor, and the dummy CMD read keeps the BH-55 workaround. No API addition is needed for E0 (E1's count
accessors, Phase 1 part E1-20, are E1's). **Check:** v3v6 with only these changes (`mk_v3y.py --api-only` +
`mk_cpp.py`) is PT_LOAD-identical to v3v6 on every config: E0 208/208, 208/208, 207/207 and 48/48 VC1 at -Os; E1
328/328, 328/328, 327/327, 48/48.

Compiler-assumption items left in E0, not changed here (V3_REVIEW b):
- Two fences in `send_packet` order no reads (after the `src_ch_id` stamp, after the free-slot bookkeeping): they pin
  code position. Proposal: measure without them.
- Bit tests written as sign tests: a rewrite re-allocates the pass (V3_REVIEW). Readability only; left with the anchors
  `derive_edits.py bittest` uses.
- The in-flight word is a member of the loop's `ForwardingState`; GCC keeps it in the frame, so a pass with idle
  forwarding channels does one stack load (`lw a5,216(sp)` at 0x7ee0). Allocation; not addressable by rule.

## 5. Old LOST-CONTEXT and OPEN ideas that target E0

| Idea (ledger part 1) | Today | Re-test? |
| --- | --- | --- |
| Worker visit latency (§73): the semaphore and payload-size loads each behind a stack load | Residency of the worker slot and semaphore pointer in the pass (emulator traces): v3v6 has both in registers on 2D and 1D at -O3/-O2; the pointer is a stack slot at -Os everywhere; on the VC1 E0 at -O3/-O2 both are stack slots (`lw a7,20(sp); lhu a1,40(a7)` and `lw a3,4(sp); lw a1,0(a3)`). v3y: both registers at -O3/-O2 on every config, the pointer still a stack slot at -Os on 2D and VC1 | Allocation outcomes either way. The structural fix is the semaphore address as a compile-time arg (host proposal, E0-16) |
| Smaller sender-channel state (bit 24, s77t, s78l) | In by construction (v3l) | no |
| Shared runtime forwarding step (s80) | Lost three times on VC0 in Phase 1; won on VC1 (v3u: −6 to −15) | **VC1 routers only**: one inline visit per channel costs the VC1 E0 ~440 B each; worth a VC1-only design if VC1 rows matter |
| `noinline` main loop, cold connection helpers | In | the per-round part is v3y |
| Forwarding group gate (bit 18, MIXED) | v3v6 keeps the per-VC combined test | v3v6l3 (§6) |
| Early semaphore read in the post-send group (§49, MIXED) | not needed where the pointer is a register | no |
| Drain several packets per visit (OPEN) | E0 sends at most one packet per channel per pass; on the VC1 E0 a worker send pass scores ~130–145 | Worth a design for E0-bound small-packet rows (it removes per-pass work per packet); the TX queue must be idle before a second data command (post-send busy test) |
| Batched pointer updates, summary doorbell (OPEN) | protocol and producer changes | not this round |

## 6. v3v6l3

v3v6 with round 11's v3l3 E0 change: VC0's forwarding visits run without the combined "any forwarding work" test (the
`SKIP_IF_ALL_IDLE` argument of VC0's `visit_forwarding_channels` is false; VC1 keeps its test), plus v3l3's comments. 16
changed lines. Its E0 is PT_LOAD-identical to v3l3's on all 208 E0 configs at -O3 (v3l3's own dir, built today), and
its E1 to v3v6's everywhere, so it is exactly "round 11's v3l3 E0 + today's E1". It changes the 2D (40), VC1 (48) and
1D ring (40) E0s; 1D line E0s are byte-identical (one forwarding channel). Round 11 measured v3l3 ≈ v3l3b at -O3 (FWD
16 B 112.3 vs 112.5, E/W 16 B 115.6 vs 116.0); its only blocker was the third VC1 hang, which RESULTS §98 traced to the
test race.

## 7. Checks (offline)

| Check | v3v6l3 | v3y | v3yb | v3yc |
| --- | --- | --- | --- | --- |
| Builds, every logged config (-Werror), -O3 / -O2 / -Os | 536 / 536 / 534 | 536 / 536 / 534 | 536 / 536 / 534 | 536 / 536 / 534 |
| VC1 at -Os from the -O2 commands | 96/96 | 96/96 | 96/96 | 96/96 |
| E1 PT_LOAD vs v3v6 (-O3, -O2, -Os, VC1 -Os) | 328, 328, 327, 48 identical | same | same | same |
| E0 configs that change vs v3v6 (-O3) | 128 (2D, VC1, ring) | 168 (all sender_erisc E0s; the 40 speedy E0s identical) | 168 | 168 |
| Ordinary edits on `run_main_loop` (coldmul, init, templates, os; 5 hot configs × 3 levels) | never DIFF: coldmul moved; init moved (same at -Os); templates same; os moved (same at -Os) | same pattern, never DIFF | same | same |
| `hb` edit | DIFF (2 lines) | same | same | DIFF (2 lines) |
| Per-round probes `term`, `ctxsw` | DIFF (as v3v6) | moved | moved | DIFF (as v3v6) |
| VC1 E0 + E1, -O3 (limit 26,624 B) | 13,064 + 11,996 = 25,060 | 13,136 + 11,996 = 25,132 | 13,268 + 11,996 = 25,264 | 13,264 + 11,996 = 25,260 |
| VC1 E0 + E1, -O2 / -Os | 23,980 / 21,624 | 23,984 / 21,792 | 24,092 / 21,920 | 24,176 / 21,756 |
| Coverage (`coverage.py build --opt O3`, 41 cases, 1,395 configs) | 1,259/1,395, identical to v3tk case by case | same | same | same |
| VC2 (`bh_2d_vc2`) at -O3 / -O2 / -Os | 56/56 each | 56/56 each | 56/56 each | 56/56 each |
| Raw register access in E0 loop code | as v3v6 (§4) | none | none | none |

(v3v6: VC1 13,120 + 11,996 = 25,116 / 24,044 / 21,640.) The only coverage failures are the inherited ones:
`bh_1d_nofuse`, `bh_2d_nofuse`, `bh_vc1_nofuse`, `wh_1d_nofuse` (0/n) and `wh_vc1` 15/20. The `heartbeat` edit of `derive_edits.py` (not one of the four) asserts v3v6's heartbeat layout and needs a
small update before it can run on v3y/v3yb.

## 8. Per-pass counts (method: §0)

Instructions / taken branches / loads per pass, and the score in brackets. 2D E/W and N/S E0s give identical numbers.

**-O3**

| config | path | v3v6 | v3v6l3 | v3y | v3yb | v3yc |
| --- | --- | --- | --- | --- | --- | --- |
| 2D | worker send | 80/5.8/18 [110] | 84/7.8/17 [116] | 88/5.9/16 [115] | 84/3.9/16 [108] | 76/3.8/19 [103] |
| 2D | forwarded send | 105/7.3/18 [137] | 102/7.3/18 [134] | 108/7.4/19 [142] | 105/8.4/19 [141] | 101/8.3/19 [136] |
| 2D | worker + forwarded | 160/9.0/30 [208] | 155/9.0/28 [200] | 168/9.1/28 [215] | 165/7.1/28 [208] | 157/7.0/32 [202] |
| 2D | idle | 25/4.1/7 [40] | 31/6.1/7 [50] | 27/4.2/6 [42] | 24/5.2/6 [41] | 20/5.1/7 [37] |
| 1D | worker send | 73/5.8/12 [97] | 73/5.8/12 [97] | 80/5.9/12 [104] | 78/4.9/13 [101] | 70/4.8/12 [92] |
| 1D | forwarded send | 91/5.2/15 [117] | 91/5.2/15 [117] | 91/5.3/15 [117] | 88/4.8/16 [114] | 88/4.7/15 [113] |
| 1D | worker + forwarded | 144/6.9/23 [181] | 144/6.9/23 [181] | 150/7.0/23 [187] | 148/3.5/25 [181] | 140/3.4/23 [170] |
| 1D | idle | 21/4.1/4 [33] | 21/4.1/4 [33] | 21/4.2/4 [33] | 18/6.2/4 [35] | 18/6.1/4 [34] |
| ring | worker send | 78/6.3/14 [104] | 77/7.3/14 [105] | 84/6.4/15 [112] | 80/4.4/15 [105] | 74/4.3/14 [96] |
| ring | forwarded send | 104/7.2/18 [136] | 100/5.2/18 [128] | 105/7.3/18 [138] | 101/8.3/18 [136] | 99/8.2/18 [133] |
| ring | worker + forwarded | 158/8.4/26 [202] | 154/6.4/26 [194] | 165/8.5/29 [211] | 162/6.5/29 [204] | 154/6.4/26 [194] |
| ring | idle | 24/5.1/4 [39] | 23/6.1/4 [40] | 24/5.2/4 [38] | 20/6.2/4 [36] | 19/6.1/4 [36] |
| VC1 | worker send | 103/6.9/27 [143] | 109/8.9/27 [153] | 105/7.0/19 [137] | 103/5.0/20 [132] | 100/4.9/27 [136] |
| VC1 | forwarded send (VC0) | 120/8.6/24 [161] | 117/8.6/24 [158] | 123/8.7/22 [162] | 121/9.7/23 [163] | 117/9.6/24 [160] |
| VC1 | VC1 channel send | 128/9.6/25 [172] | 134/11.6/25 [182] | 131/9.7/23 [174] | 129/10.7/24 [175] | 125/10.6/25 [171] |
| VC1 | idle | 41/5.1/12 [63] | 47/7.1/12 [73] | 44/5.2/10 [64] | 42/6.2/11 [65] | 38/6.1/12 [62] |

**-Os**

| config | path | v3v6 | v3v6l3 | v3y | v3yb | v3yc |
| --- | --- | --- | --- | --- | --- | --- |
| 2D | worker send | 80/4.8/17 [107] | 88/7.8/19 [123] | 83/4.9/15 [108] | 85/6.9/15 [114] | 79/5.8/17 [108] |
| 2D | forwarded send | 104/11.3/20 [148] | 102/10.3/20 [143] | 106/11.4/20 [149] | 109/12.4/20 [154] | 105/11.3/20 [148] |
| 2D | worker + forwarded | 160/12.1/30 [214] | 156/11.1/30 [209] | 164/12.2/28 [216] | 166/14.1/28 [222] | 159/13.1/30 [215] |
| 2D | idle | 25/4.0/7 [40] | 33/7.0/9 [56] | 25/4.2/7 [41] | 28/5.2/7 [45] | 25/4.0/7 [40] |
| 1D | worker send | 72/5.8/14 [98] | 72/5.8/14 [98] | 75/5.8/12 [99] | 75/6.8/12 [101] | 71/6.8/14 [99] |
| 1D | forwarded send | 87/8.2/16 [120] | 87/8.2/16 [120] | 87/8.3/15 [119] | 87/8.3/15 [119] | 87/8.2/16 [120] |
| 1D | idle | 18/5.0/5 [33] | 18/5.0/5 [33] | 18/5.2/4 [32] | 18/5.2/4 [32] | 18/5.0/5 [34] |
| ring | worker send | 78/6.2/15 [105] | 78/6.2/15 [105] | 78/6.3/13 [104] | 80/7.3/13 [108] | 80/7.2/15 [109] |
| ring | forwarded send | 101/10.8/20 [142] | 95/10.8/19 [135] | 101/10.8/18 [141] | 101/10.8/18 [141] | 101/10.8/20 [142] |
| ring | idle | 20/6.0/5 [38] | 20/6.0/5 [38] | 20/6.2/4 [36] | 20/6.2/4 [36] | 20/6.0/5 [38] |
| VC1 | worker send | 97/5.8/21 [130] | 105/8.8/23 [146] | 101/5.9/19 [132] | 100/6.9/19 [133] | 96/6.8/21 [131] |
| VC1 | forwarded send (VC0) | 121/12.8/24 [171] | 118/11.8/24 [166] | 122/12.9/24 [172] | 122/12.9/24 [172] | 121/12.8/24 [171] |
| VC1 | VC1 channel send | 131/12.1/25 [180] | 139/14.6/27 [195] | 132/11.7/25 [180] | 132/11.7/25 [180] | 131/12.1/25 [180] |
| VC1 | idle | 41/5.0/11 [62] | 49/8.0/13 [78] | 41/5.2/11 [63] | 42/5.2/11 [63] | 41/5.0/11 [62] |

-O2 (scores; same scenarios): 2D worker send 105 / 121 / 115 / 108 / 103, forwarded 141 / 138 / 146 / 145 / 140, idle
40 / 54 / 42 / 41 / 37; VC1 worker send 144 / 158 / 138 / 132 / 137, idle 63 / 77 / 64 / 65 / 62 (v3v6 / v3v6l3 / v3y /
v3yb / v3yc). The full table is `scratchpad/e0audit/paths_table.md` (session scratch, not durable).

Reading:
- **v3yc** is at or below v3v6 on every -O3 and -O2 path but the 1D idle pass (+1) (2D worker send −7, VC1 −7, 1D
  worker + forwarded −11), and within +1 at -Os except the ring's worker paths (+4). The gains at -O3 are layout (fewer taken branches on the send path
  with the nested form), not removed work.
- **v3y** moves the pass code both ways by allocation: -O3 2D +5 on worker send and forwarded, 1D/ring worker +7/+8,
  VC1 worker −6 (stack loads 11.7 → 3.7). -Os within ±3. These are the kind of moves the `term` probe makes on v3v6
  itself (+8 to +14), i.e. inside today's allocation spread.
- **v3yb** = v3y plus the worker change: at -O3 and -O2 at or below v3y on most paths; at -Os +5 to +8 on every 2D
  path (an extra taken branch in v3y's allocation), up to +5 on 1D and ring passes with a worker send, ±1 elsewhere.
- **v3v6l3**: forwarded passes −3 to −8, worker-only and idle passes +6 to +16 (three ungated channel tests), as V3L §11
  predicted; 1D line unchanged.

## 9. Risks

- **The proxy is a proxy.** Rankings by score are not cycles (RESULTS §82); the pass-code moves of v3y are allocation and
  may come out either way on hardware, within the jitter band if §96/§98 hold for E0 too.
- **The worker change's gain is inferred** from the loop structure, not measured: it exists only where the far ring is
  the binding window with a worker packet waiting. If the worker's own window binds (worker refill latency), the branch
  is never taken and the change costs only its test.
- **v3yb at -Os** pays +5 to +8 on the 2D passes; v3yc does not. If v3yb loses at -Os, that is the
  first suspect, not the worker change.
- **Semantics** (v3yb, v3yc): credits are absolute counts and stay monotonic; a pass can now make two worker credit
  writes (early, and after the send if the post-send read finds more). A yielded send after an early read does no late
  read that pass (the counter was read already). Bubble flow control uses the same `far_has_room` predicate in both
  tests. First-level ack credits on acks as before.
- **v3y's helpers are plain `noinline` functions** in a header compiled for every build (as E1's `end_round`): coverage
  builds all 41 cases (Wormhole, single-ERISC, VC1, telemetry, UDM, mux) like v3tk.
- **v3v6l3**: worker-only rows may lose 1–3 (proxy +6 to +16 per pass on worker and idle passes); VC1 routers too.
- Not checked: the `hook` probe on the candidates (make the `_hook` dirs with `prep_round.sh`).

## 10. Hardware run list

Each as the full protocol: validation (RELAXED and STRICT), the suites at -O3 / -O2 / -Os, VC1 at every level with the
capture armed (the test-race fix, or the capture, for the 4 KiB all-to-all), and `_hook` / `_strict` dirs from
`prep_round.sh`.

1. **v3yc** (the worker change alone; the helpers are codegen-neutral).
   - Rows: the VC1 split's 1-hop E/W and N/S writes, 16 B and 4352 B (v3v6 ~199–214; the 8-slot far ring is where the
     change should show, up to ~6–7 cycles/packet); Mesh1HopEW/NS write, EW/NS_PureWrite4352 (16-slot ring, up to
     ~2–3, or nothing if the worker's window binds); FWD and mixed rows where worker and forwarded traffic share a far
     ring; MixSize4.
   - E0-bound rows at -Os (E/W and N/S 16 B writes): the worker-send pass +1 taken branch.
   - The burst benchmark (`exp_burst.yaml`, `exp_burst_vc1.yaml`): the first send after each credit frame moves a pass
     earlier. The latency suite should not move (a lone packet never waits for room).
   - Optional: perturbation sweep (a) on v3yc to confirm a smooth slope.
2. **v3v6l3** (round 11's cheap A/B): FWD 16 B, Mesh3HopEW, multicast, forwarded fused (expect ≤ v3v6); E/W and N/S 16 B,
   1-hop 4352 and the VC1-router worker rows (expect ≥ v3v6 by ~1–3). VC1 34/34 at every level.
3. **v3yb** (restructuring + worker change): v3yb − v3yc is the restructuring. Watch -Os 2D worker rows (§9) and VC1
   worker rows (stack loads 11.7 → 3.7 per pass at -O3/-O2).
4. **v3y** only if v3yb − v3yc needs confirming without the worker change. An insulation check on hardware, if wanted:
   the `term` probe (`probes.py`) on v3v6 and on v3y in one jitter-style job; v3v6's rows may move, v3y's should not.
