# Ring-jam fix: design (2026-11-03, for the user's review; no build, no runs)

Base: the final router, `fabric_exp/variants/v4txa11`. Data: FULLRING_MCAST_GAP.md §8.14-8.19 (the injection-throttle
probe `variants/v4rxg3it` on v4rxg3), A11's Galaxy sweep and breakdown (jobs 170094-170096). Everything not measured is
marked hypothesis. Nothing here is built.

## 1. The problem, measured

The probe (block VC0 injection while the in-ring channel's backlog >= T, unless the in-ring channel sent K packets
since the last injection) removed the Galaxy slow mode:
- 8-ring full-ring multicast at 4352 B: 371 -> 254 (B and C; main 263.8);
- 4-ring: 339 -> 255;
- half ring: 332 -> 254;
- torus: 270 -> 260 (B).

At 400G (P150), where there is no slow mode, it cost:
- (a) a fixed E0 cost while the gate never engages: +4..7 cycles on the XY-torus 1-hop write rows, in every arm. E0
  `.text` grew +88..108 B on the torus and +32 B on the 1D ring, where the 1-hop rows did not move;
- (b) over-throttling on lighter patterns:
  - ring all-to-all multicast write: +21.5 / +20.1 on B, +15.1 / +7.8 on C;
  - 2x4 XY-torus multicast write: +15..22 on A and B.

A11 keeps the slow mode: Galaxy 8-ring 4352 B is 371.4, unchanged from v4rxg3 (jobs 170094/170095).

## 2. Why the probe cost what it did (code reading, P150 run images of lb2 170034)

The probe's actual run images: base (switch off) and arm A from `/data/aagarwal/lb2/jobs/170034/caches/`.
- Pairing is by named args, with EXP_DRISC_DELAY normalized.
- `sender::run` is compared instruction by instruction, with addr2line inline chains (scratchpad `jamfix/run_lines.py`).
- Spills come from `loopvars.py --headroom`.

**XY-torus E0** (2x4, E router; worker ch0 and turns ch2 / ch3 gated, in-ring ch1):
- Code size: `.text` +88 B; `run` 756 -> 778 instructions; loop 651 -> 673.
- Code under the throttle helpers' inline frames:
  - `injection_allowed`, inlined three times (worker and both turns), 4-5 instructions each
    (`seqz` / `sltiu` / `or` / `beqz`);
  - `count_sent_for_throttle`: three resets (`li s7,0`) and one saturating increment (`bgeu` / `li` / `add`).
- **The loop carries 14 registers instead of 13.** The count lives in s7 for the whole loop, and the backlog flag in
  s9.
- **headroom fails:** three hot values of the worker's send move to the stack, stored at `send` (router_sender.hpp:329):
  `workers.vc0.slot`, `send<0>:header` and `send<0>:src`. Stack accesses in the loop: 95 -> 98.
- Elsewhere the loop moved by a few instructions per line (visit_forwarding, `run`'s pass back-edge +4).

**1D ring E0** (worker ch0 gated, in-ring ch1):
- Code size: `.text` +32 B; `run` 412 -> 420 instructions.
- One gate, one reset, one increment.
- Carried registers also 13 -> 14; one hot value spilled (`far.vc0.next_slot`, at `send` line 328).

**Reading (hypothesis: static counts, not cycles).** On a pass where only the worker holds a packet (the 1-hop write
rows), the probe adds per pass:
- one gate evaluation (~5 instructions);
- one count reset;
- the spill traffic around the worker's send (three stores, three reloads).

Together that is roughly 10-15 executed instructions where the gate is open. The measured +4..7 cycles on the
E0-bound torus 1-hop rows fits that.

**Not the cause:**
- an extra register read: the backlog came from the pass's group read already loaded;
- the in-ring channel's own code.

On the 1D ring the 1-hop rows did not move (140.9 in every arm) despite a similar spill. Hypothesis: those rows are not
E0-bound, so the added E0 work hides. Only an A/B on that row can say.

**Design rule taken from this:** the fix must carry no loop-wide state in registers (no new carried register). Its
fast-path test uses only values the pass already holds, with immediates.

## 3. The mechanism, read from v4txa11's code (hypothesis; both readings point to the same fix)

From `router_sender.hpp`:
- `visit_channels`: each pass, a VC's worker channel is visited first, then its forwarding channels, lowest channel
  first. Each visit sends at most one packet.
- `send()` issues the data and, while the TX queue is busy, spins until it is idle before posting the
  packets-received increment. So a send holds E0 for about the packet's wire time.
- Admission (`remote_free_slots_to_send`): an injection channel needs 2 free remote slots (bubble), the in-ring channel
  needs 1.

With both holding packets and at least 2 free remote slots, each pass sends one injection packet and one in-ring
packet. The link is shared 1:1 by channel:
- 1D ring and torus E/W routers: worker : in-ring = 1:1;
- torus N/S routers: worker plus two turns, all visited before in-ring ch3 = 3:1.

The link of an N-ring in full-ring multicast carries N-1 flows, and the in-ring channel aggregates N-2 of them. The
worker gets 1/2 of the link instead of 1/(N-1); the flows from far away finish late, and the row (the slowest sender)
is slow.

The bubble rule overrides the 1:1 only when the remote ring has fewer than 2 free slots, that is, when the far
receiver (E1) is the constraint. That is H9's regime split (§8.7).

**Fits the data:**
- **H9:** there is a slow mode only when E1 serves faster than the link. The RX-delay sweep (§8.10) made E1 the
  constraint and the slow mode went away. Main's E1 is slower than the 200G wire.
- **K dependence:** K = 2 fixes the 4-ring (3 flows per link: worker share 1/3). K = 6 fixes the 8-ring (7 flows) but
  2 does not.
- **T independence on 1D:** in the slow mode the in-ring channel always holds a backlog.
- **A11's point-5 breakdown** (Galaxy full ring at 4352 B, busiest router), within E0 only: 285 of 367 cycles per
  packet in Ethernet send, 19 in polls and room checks. E0 is mostly sending, not waiting for far credits. (A
  within-ERISC split, not a bound claim.)

It is not separated by an A/B from §8.13's bubble-jam reading. Both say the same thing: when the far ring is not the
constraint, nothing shares the link by flow.

**Consequence for a trigger on in-ring far-credit refusals** (the candidate (i) in the request): such refusals happen
when the far ring is the constraint. That is the E1-bound regime (P150 at 400G), where there is no jam, and it should
be rare in the Galaxy jam per the E0 split above. Section 5.3 develops this.

## 4. Pieces common to the candidates

- **Contest:** an injection channel and VC0's in-ring channel both hold packets this pass.
  - From the pass's `holding` mask (already in a register) and immediate masks: no load, no new state.
  - The injection group is the probe's: the worker, plus on a torus the turn channels (those the host marks injection,
    `sender_channel_is_traffic_injection_channel_all_`).
  - The in-ring channel is the VC0 forwarding channel not marked injection (one per router on a ring or torus).
- **Link-bound test:** the remote receiver channel has at least R free slots, with R = K + 2 (room for the in-ring
  channel's K sends plus the injection's bubble).
  - The free count is the low half of the loop-local `RemoteReceiverChannelProgress` word, already in a register (the
    send test computes it).
  - Hypothesis: in the jam the far ring is mostly free (E1 drains faster than the link fills it); at 400G it is
    mostly near full. Section 6, step 1 checks this before any fix is built.
- **K from the host, per router:** the ring length along the router's axis minus 2, the upstream flows a link carries
  in a full-ring pattern.
  - 1D ring: the ring's size. Torus: the X ring's for E/W routers, the Y ring's for N/S routers.
  - The host's control plane knows this; it becomes one named argument (for example `VC0_RING_SHARE`), 0 on line axes
    and non-ring VCs, which compiles the feature out.
  - Galaxy: 4-ring 2, 8-ring 6; torus X 2, Y 6. P150: 8-ring 6; the 2x4 / 4x2 torus ring axis 2, the line axis 0.
  - The probe used K = 6 on every router: twice the 4-long X axis's share on the Galaxy torus. Hypothesis: part of why
    the torus gained only 270 -> 260.
- **Bound by backlog:** the share given to the in-ring channel never exceeds what it holds (min(K, backlog)), so the
  share adapts to lighter patterns without a pattern-dependent K.
- **Scope:** VC0 on deadlock-avoidance (ring / torus) routers only. Mesh, line and non-ring VCs compile it out
  (`if constexpr` on the host's argument; off-builds byte-identical, as v4rxg3it's).

## 5. Candidates

### 5.1 C1: contest gate (the probe, made cheap)

Once per pass, before the worker's visit:
- if there is a contest and the far ring is link-bound, injection may send only when `owed == 0`;
- an injection send sets `owed = min(K, in-ring backlog)`;
- each contest pass in which the injection is held decrements `owed`. The in-ring channel can send then: room >= R > 1.

`owed` lives in memory (the SenderChannelStates block), not in a loop register. It is touched only on contest passes.
Torus turns are held with the worker during `owed > 0`.

**Static cost vs v4txa11** (estimates; step 0 below measures them):

| path | added |
| --- | --- |
| worker-ready pass, no contest | about 2 instructions (`andi holding, in_ring_bit` + branch), no load, no carried register |
| turn-channel visit holding a packet (torus) | about 2 instructions each |
| contest pass | 1 load (`owed`), 1 store, about 4-6 instructions |
| in-ring path | 0 |

`.text`: estimate +30..60 B on torus E0, +20..30 B on 1D E0.

**Rows:** Section 5.4.

**Risks:**
- It blocks injection, like the probe. If the link-bound test misfires at 400G, it over-throttles again; P150 safety
  rests on the R test (hypothesis).
- The `owed` decrement on a held pass is approximate: if the in-ring send yields on a busy TX queue, the debt still
  drops. Bounded, but a small share error.
- The worker's wait per injection is bounded by K contest passes. Admission is unchanged (injection still needs 2,
  in-ring 1), so no new deadlock: whenever the injection could send, the in-ring channel can too.

### 5.2 C2: in-ring burst (stateless weighted round robin), recommended

In the in-ring channel's visit, after a successful send:
- if there is a contest and the far ring is link-bound, the channel sends again, up to min(K, backlog) packets this
  pass, each needing room;
- the backlog check re-reads its free-slots register, one register read per extra packet.

The worker and the turns still send one packet each per pass and are never held. The share becomes
injection 1 : in-ring K per pass.
- The worker is visited before the forwarding channels today, so the order stays.
- E/W torus routers: worker, then in-ring (ch1, up to K), then the turns. N/S routers: worker, turns, then in-ring
  (ch3, up to K).

No state across passes, no counter, no new register.

**Static cost vs v4txa11** (estimates):

| path | added |
| --- | --- |
| worker path | 0 instructions |
| in-ring send, no contest | about 2 instructions after each send (`andi holding, injection_bits` + branch) |
| burst iteration | 1 free-slots register read, 2 compares, the loop branch; the send sequence is reused (a loop back, not a copy) |

`.text`: estimate +40..80 B on torus E0, +30..50 B on 1D E0.

**Rows:** Section 5.4.

**Risks:**
- **Longer passes during a contest:** up to K sends back to back (K x wire time, ~1.5k cycles at 4352 B on 200G for
  K = 6). The per-pass worker connection check, the other VCs' service and the completions of other channels wait
  that long; the check still happens every pass. The user decides whether that latency bound is acceptable (memory:
  keep the per-pass connection check; worker-side latency adds to op cost).
- **Code on the hottest forwarding path:** the burst loop could raise register pressure on the in-ring send. The
  headroom check (step 0) must pass on 1D and torus E0.
- **Upstream credits and completions** for the in-ring channel are taken once per burst instead of per packet.
- **No starvation** (the worker sends every pass) and **no admission change** (no new deadlock).

### 5.3 C3: trigger on in-ring far-credit refusals (candidate (i)): not recommended

Engage the floor after the in-ring channel was refused for lack of remote room (a flag set on the existing refusal
path), instead of on any backlog.

Cost would be near zero: one store on a cold path, one test at the injection.

But by Section 3 it fires when the far ring is the constraint:
- the E1-bound regime (P150 at 400G, Galaxy at <= 3072 B), where there is no jam;
- and it would rarely fire in the Galaxy jam, where E0 is mostly sending (point-5 split).

Expected (hypothesis): it throttles the systems that do not need it and misses the one that does. Step 1 below would
show this too (refusal counts by regime) without building it.

### 5.4 Expected effect on rows (hypothesis), C1 and C2

**Galaxy, should fix** (about the probe's B):
- 8-ring 4352 B (~254) and 3584 B (~226);
- 4-ring 4352 B (~255; K = 2);
- half-ring multicast 4352 B (~254);
- XY torus full ring 4352 B: at least the probe's 259.6, and possibly lower with per-axis K (low confidence).

**Galaxy, should leave alone:**
- 3072 B and below: E1 is the constraint, so no link-bound contest;
- the 1-hop rows: no in-ring traffic, no contest;
- mesh and line suites: compiled out.

**P150, should leave every row within noise:**
- ring, half, unidirectional and all-to-all multicast; 1-hop; XY torus on both layouts.
- The far ring is near full (E1-bound), so the link-bound test fails; the backlog bound limits any burst.
- The probe's cost rows specifically:
  - torus 1-hop writes: C2 adds nothing on the worker path; C1 about 2 instructions per pass (maybe <= 1 cycle);
  - ring all-to-all and 2x4 torus multicast: no contests pass the R test.
- If the R test does not separate the regimes (step 1), both candidates over-throttle like the probe and the design
  needs a different discriminator. The narrow fallback: the host enables the feature only where E1 outruns the link
  (per link speed), which the generic-over-narrow rule disfavors.

## 6. Cheapest isolating A/Bs, in order (proposed, not queued)

0. **No device:** both candidates as study switches on v4txa11.
   - Off-builds must be byte-identical (frg_elfcmp on Galaxy and P150 mock builds, including both torus layouts).
   - `loopvars.py --headroom` must pass on 1D-ring and torus E0 (no new hot spill: the probe's failure).
   - Count the fast-path instructions by inline chain.
   - A candidate that spills is redesigned before any run.
1. **Trigger check, telemetry only:** count contest passes with remote free >= R, contest passes with free < 2, and
   in-ring refusals for room. Behavior is unchanged.
   - Rows: Galaxy 8-ring 4352 B (jam) and P150 ring all-to-all multicast 3072 B (no jam), one job each.
   - Prediction: Galaxy contests mostly have room >= R and few refusals; P150 contests mostly have room < 2 and many
     refusals.
   - This one A/B decides the trigger (R test vs (i)) before either fix is built.
2. **Performance:** base, C1, C2, two passes in reversed order.
   - Galaxy, one job (~25 min): 8-ring 3072 / 3584 / 4352, 4-ring 4352, half ring 3072 / 4352, torus 3072 / 4352,
     ring 1-hop.
   - P150, one lb2 job, validation first: ring all-to-all / half / unidirectional multicast write + scatter, 1-hop,
     XY torus on both layouts (the probe's cost rows).
   - C2's pass-length effect would show on the multi-VC and latency rows, so add one VC0+VC2 matched row.

## 7. Recommendation

**C2, the in-ring burst**, with:
- K = the ring length along the router's axis minus 2, from the host;
- the burst bounded by min(K, backlog) and by remote room;
- the contest and link-bound tests from values the pass already holds (`holding`, the remote word).

Why C2: it never holds an injection channel, so there is no starvation and no blocked worker on lighter patterns. It
adds nothing to the worker path (the probe's cost rows), keeps no state across passes, and its share adapts to the
backlog.

C1 is the fallback if:
- C2's burst fails the headroom check on the forwarding path; or
- the user rejects the longer passes during a contest (the per-pass connection-check latency bound).

Both depend on step 1 confirming that remote room separates the jam (Galaxy) from the E1-bound regime (P150). If it
does not, neither should be built as designed.

**Decisions for the user:**
1. C2's latency bound during a contest (up to K packets per pass before the next connection check).
2. A new host argument for the ring share per router.
3. Whether the telemetry trigger check (step 1) runs before any fix is built (proposed: yes).

## 8. User decisions (2026-11-03) and step 1: telemetry (written before any run)

**Decisions:**
- C2 (in-ring burst) is the primary design; C1 is the comparison arm.
- K is a cap, not a target: in a contested pass the in-ring channel sends min(K, backlog). The backlog comes from the
  pass's existing snapshot. The burst never waits for packets and stops at the first send without a free far slot.
- A host argument gives each router its K: ring length along its axis minus 2, per axis on the torus.
- The telemetry check runs first. If the counters do not show the room test separating the Galaxy jam from the P150
  cases, stop and report: no fallback without the user.

**Build `variants/v4txa11j`** = v4txa11 plus counters. No behaviour change. Compiled out when off.
- **Switch:** the host's phase profile at level 1 (`TT_FABRIC_PHASE_PROFILE=1`). The host then allocates the
  per-ERISC FabricPhaseProfile block (`PHASE_PROFILE_BUFFER_ADDR`), and test_tt_fabric reads it before and after each
  test into `TT_FABRIC_PHASE_PROFILE_CSV`. No host change.
- **Where counted:** VC0 of a deadlock-avoiding router's E0, in a global (local memory, not loop registers, as
  router_telemetry.hpp's counts). The counts are written to the block once per round (after_round).
- **Fields** (the CSV columns keep the phase profile's names):

  | CSV column | holds |
  | --- | --- |
  | iters | passes |
  | pkts | VC0 sends |
  | busy_ticks | contests: an injection channel and the in-ring channel both hold a packet at the pass's read |
  | blocked_ticks | contests with < 2 free far slots |
  | idle_ticks | passes with the in-ring channel holding a packet |
  | gap_ticks | in-ring sends |
  | events[0] / [1] | in-ring / injection far-credit refusals |
  | events[2] | injection sends |
  | phase_count[i] | contests with i free far slots (11 = 11+) |
  | phase_ticks[i] | contests with an in-ring backlog of i (one extra free-slots register read per contest) |
  | info | in-ring channel, injection channel mask, far ring slots, "jam" |

- The room test is evaluated offline per router from the histogram, with R = K + 2:

  | router | K | R |
  | --- | ---: | ---: |
  | Galaxy 8-ring (N/S) | 6 | 8 |
  | Galaxy 4-ring (E/W) | 2 | 4 |
  | Galaxy torus X | 2 | 4 |
  | Galaxy torus Y | 6 | 8 |
  | P150 8-ring | 6 | 8 |

- **Cost when on:** a few local-memory increments per pass, one register read per contest, one block write per round.
  The host's block allocation also shifts the L1 layout, so slots are logged in both arms. Measured as off vs on on the
  same rows, two passes.

**Rows, two passes (off, on / on, off):**
- Galaxy: `glx_frg_nsrx` (8-ring 3072 / 3584 / 4352 B; 3072 B is the router-bound control). Extras for the per-axis
  test: `glx_frg_ew` (4-ring 4352) and `glx_frg_torus` (3072 / 4352). Validation first: `glx_frg_val` on the on-arm.
- P150 (lb2): `p150_it_ring.yaml` (ring 1-hop, unidirectional / half / all-to-all multicast, write and scatter,
  3072 / 4352 B). These are the proven rows, with no flow-control full-ring row. Validation first: `val_ring_p150` on
  both arms.

**Predictions (hypothesis, from §3):**
1. **Galaxy 8-ring 4352 B (and 3584 B), the jam:**
   - contests on most passes the in-ring channel holds a packet;
   - room >= 8 in at least 70% of contests;
   - room < 2 in at most 10%;
   - in-ring refusals below 0.1 per in-ring send;
   - in-ring backlog at contests mostly >= 4.
2. **Galaxy 8-ring 3072 B (control, router-bound) and every P150 multicast row (E1-bound):**
   - room >= 8 in at most 30% of contests;
   - in-ring refusals at 0.3 or more per in-ring send.
3. **Galaxy 4-ring 4352 B:** room >= 4 in at least 70% of contests.
4. **Telemetry cost:** on vs off within 2% on every row (if more, the counts still stand, but the cost is reported).
   **WRONG** (Galaxy 171203): v4txa11j's on-arm ran the 8-ring at 491.9 / 613.0 / 607.4 against 191.9 / 306.2 /
   374.6 (3072 / 3584 / 4352 B). Per-pass counting perturbs the regime and biases the room counts toward "room
   plentiful", so they do not stand. Replaced by round sampling (v4txa11k, §8.2).

**Decision rule, fixed now:** the room test separates if
- at least 70% of the contests on the Galaxy jam rows (8-ring 3584 / 4352 B, 8-ring routers) have room >= R, and
- at most 30% do on the P150 multicast rows (3072 / 4352 B).

Otherwise: stop and report.

### 8.1 v4txa11j built (2026-11-03; mock compiles only, nothing run)

- **Variant:** `variants/v4txa11j` (v4txa11 plus the counters in `router_sender.hpp`, and two ct args reading
  `PHASE_PROFILE_MODE` / `PHASE_PROFILE_BUFFER_ADDR`). Galaxy tree `trees/v4txa11j_1103` (base_1102). The counts sit
  behind template accessors, so nothing is instantiated without the block.
- **Off = v4txa11, byte-identical:** Galaxy (frg_nsrx, frg_torus, check_2d) 408 / 408; P150 (p150_it_ring,
  torusxy_2x4) 1056 / 1056 router ELFs.
- **On (`TT_FABRIC_PHASE_PROFILE=1`):** every mock row reaches launch. No slot argument changes.
- **On, E0 `.text`:**

  | router | Galaxy | P150 |
  | --- | --- | --- |
  | 1D ring | +940 to +1012 B | +512 to +1432 B |
  | torus | +1072 to +1148 B | +644 to +1572 B |
  | 2D mesh (counters compiled out) | -128 to +76 B | |

  Most of the ring and torus growth is the per-round block write. The 2D-mesh change comes from the host's L1 shift
  (addresses move).
- **On, `loopvars --headroom` (off vs on):** passes on both. No hot value moves to the stack.

  | E0 | carried registers | loop instructions (incl. the per-round write) |
  | --- | --- | --- |
  | 1D ring | 14 -> 14 | 317 -> 560 |
  | torus | 13 -> 15 | 642 -> 918 |

  Per pass: about 7 instructions per 64-bit local-memory count, a few count updates per pass. Per contest: one
  register read and two histogram updates. The run cost is measured off vs on. **WRONG as a cost estimate:** E0 makes
  several passes per packet, so this per-pass work cost hundreds of cycles per packet (§8, prediction 4).
- **Analysis tool:** `consolidation/jam_report.py galaxy|p150 CSV...` gives, per test and axis: contests, share with
  room >= R, share with room < 2, refusals per in-ring send, median backlog. Tested on a synthetic CSV; the CSV's
  direction column is RoutingDirection (0 N, 1 E, 2 S, 3 W).
- **Harness:**
  - Galaxy: `frs_glx_frgap.sbatch` configs `a11j` (off) and `a11jam` (on; the CSV per label in the job's results).
    Stub dry run: 14 / 14 steps. Real-tree preflight passes. The dry run found and fixed the new-tree copy, which only
    recognised `v4*` config names.
  - P150: `consolidation/frg_jam_p150_runs.sh` (val on both arms, then passes).
