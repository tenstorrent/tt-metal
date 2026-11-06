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
packet. (**WRONG** as the jam rows' mechanism, Galaxy 171300, §8.3: at least 2 free far slots in only 4-5% of their
contests.) The link is shared 1:1 by channel:
- 1D ring and torus E/W routers: worker : in-ring = 1:1;
- torus N/S routers: worker plus two turns, all visited before in-ring ch3 = 3:1.

The link of an N-ring in full-ring multicast carries N-1 flows, and the in-ring channel aggregates N-2 of them. The
worker gets 1/2 of the link instead of 1/(N-1); the flows from far away finish late, and the row (the slowest sender)
is slow.

The bubble rule overrides the 1:1 only when the remote ring has fewer than 2 free slots, that is, when the far
receiver (E1) is the constraint. (**WRONG** as an equation, §8.3: a full far ring can come from E1 blocked on its
output, and the jam rows have < 2 free far slots in 95-96% of contests.) That is H9's regime split (§8.7).

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
    mostly near full. Section 6, step 1 checks this before any fix is built. **WRONG** (Galaxy 171300, §8.3): in the
    jam rows the sender sees 0 or 1 free far slots in 95-96% of contests.
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

**WRONG** (round samples, §8.5): in-ring refusals, as a share of in-ring-ready samples, are 59-78% in the Galaxy jam
rows and 11-32% on P150 and the Galaxy 3072 B control.

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
1. **Galaxy 8-ring 4352 B (and 3584 B), the jam:** **WRONG** (171300, §8.3): room >= 8 in 0%, room < 2 in 95-96%,
   in-ring ready with no far slot 72-78%; backlog 11+ and contests on every in-ring-ready sample, as predicted.
   - contests on most passes the in-ring channel holds a packet;
   - room >= 8 in at least 70% of contests;
   - room < 2 in at most 10%;
   - in-ring refusals below 0.1 per in-ring send;
   - in-ring backlog at contests mostly >= 4.
2. **Galaxy 8-ring 3072 B (control, router-bound) and every P150 multicast row (E1-bound):**
   - room >= 8 in at most 30% of contests;
   - in-ring refusals at 0.3 or more per in-ring send.
3. **Galaxy 4-ring 4352 B:** room >= 4 in at least 70% of contests. **WRONG** (171300, §8.3): 20%.
4. **Telemetry cost:** on vs off within 2% on every row (if more, the counts still stand, but the cost is reported).
   **WRONG** (Galaxy 171203): v4txa11j's on-arm ran the 8-ring at 491.9 / 613.0 / 607.4 against 191.9 / 306.2 /
   374.6 (3072 / 3584 / 4352 B). Per-pass counting perturbs the regime, so the counts do not stand. Replaced by
   round sampling (v4txa11k, §8.2). (The guess written here first, that the perturbation biases the counts toward
   "room plentiful", is also **WRONG**: the perturbed counts show room < 2 in 94-95% of 8-ring contests, §8.2.)

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
- **On, `loopvars --headroom` (off vs on):** passes on both. No hot value moves to the stack. (One hand-picked pair
  per shape. Whether each pair was the same router image was not checked; see §8.2's pairing note.)

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

### 8.2 v4txa11j's run (perturbed reference) and v4txa11k (round samples) (2026-11-03)

**v4txa11j on Galaxy 171203 (a06u02, `results/glx171203`): perturbed, reference only.** The off arm is v4txa11's
images. Cycles per packet, passes a / b:

| row | off | on (per-pass counts) |
| --- | --- | --- |
| 8-ring 3072 B | 191.9 / 191.9 | 491.9 / 490.2 |
| 8-ring 3584 B | 306.2 / 305.6 | 613.0 / 612.8 |
| 8-ring 4352 B | 374.6 / 376.2 | 607.4 / 604.6 |
| 4-ring 4352 B | 339.2 / 324.2 | 320.2 / 319.8 |
| torus 3072 B | 215.2 / 215.2 | 466.9 / 468.0 |
| torus 4352 B | 261.5 / 261.5 | 594.9 / 595.0 |

Its counts (`jam_report.py galaxy`, both passes within 2 points):

| row | axis | room >= R | room < 2 | in-ring refusals per in-ring send | median in-ring backlog |
| --- | --- | ---: | ---: | ---: | ---: |
| 8-ring 3072 B | Y, R 8 | 0% | 94% | 1.72 | 11+ |
| 8-ring 3584 B | Y, R 8 | 0% | 95% | 2.55 | 11+ |
| 8-ring 4352 B | Y, R 8 | 0% | 95% | 2.50 | 11+ |
| 4-ring 4352 B | X, R 4 | 26% / 24% | 63% / 64% | 0.49 / 0.54 | 11+ |
| torus 3072 B | X, R 4 / Y, R 8 | 6% / 0% | 55% / 93% | 0.11 / 1.26 | 9 / 11+ |
| torus 4352 B | X, R 4 / Y, R 8 | 0% / 0% | 64% / 94% | 0.06 / 2.02 | 1 / 11+ |
| validation rows | both | 100% | 0% | 0 | 1 |

- Not evaluated against the rule: the 8-ring and torus rows ran 1.6-2.5x slower with the counts on, so these counts
  describe another regime. Reading (hypothesis): with every E0 slowed, the in-ring channels fill (backlog 11+), the
  far E1 cannot forward into them, and the far ring fills too.
- The 4-ring row did not slow down with the counts on (320 vs 324-339). Its counts show room >= 4 in 24-26% of
  contests, against prediction 3's 70%. Whether that stands without the per-pass counts is what v4txa11k measures.

**v4txa11k** = v4txa11 plus round samples in `router_sender.hpp` (namespace `jam`), with the same switch, ct args and
block as v4txa11j. Galaxy tree `trees/v4txa11k_1103` (base_1102).
- **Nothing in pass().** The round loop in `run()` gets one compare per pass (`n == sample_at`). Before the pass at
  that index it calls `take_sample(room)`, out of line. The sample reads VC0's in-ring and injection channels'
  free-slots registers and takes the room from the sender's own far-slot count (the loop copy's
  `RemoteReceiverChannelProgress`). After the round, one LDM increment. Every 16 rounds `flush()`, out of line, writes
  the block. The VC0 send counts come from the channels' existing `progress.sent` counters at flush time.
- **Fields:** as v4txa11j, except:
  - iters = samples, wall = rounds;
  - tx_blk_credit = in-ring-ready samples with no free far slot;
  - tx_blk_txq = injection-ready samples with < 2 free far slots;
  - rx_blk_trid = injection-ready samples;
  - info tag "jsk".
  `jam_report.py` reads both tags. It reports, per test and axis: samples, contests, the contest share of
  in-ring-ready samples, room >= R, room < 2, the in-ring and injection refusal shares of their ready samples, the
  median backlog, and the room histogram.
- **Same statistic as the rule's:** v4txa11j counted every pass, so its shares were per pass. The samples are uniform
  over pass positions, so they estimate the same per-pass shares. Each router gets one sample per round. Estimate, not measured: thousands of samples per router
  over a test's traffic. The CSV's samples column gives the real count.

**Why not sample at the round boundary (the coordinator's check).** I cannot argue that it is uncorrelated with the
contest state:
- **The boundary is a pause.** Between pass 31 and pass 0, E0 runs after_round and before_round (router-command and
  teardown checks, total_sent, the credit presets). That is tens of cycles with no send and no visit. Meanwhile the
  channel registers keep filling: the upstream E1 forwards into the in-ring channel, and the worker writes. So pass 0
  sees the "holding" state, and with it which samples are contests, later than any other pass does.
- **Phase lock.** A round is 32 passes. A steady pattern of passes whose period divides 32 would put every round's
  pass 0 at the same phase. Examples: send / refuse alternating with period 2, or a 4- or 8-pass credit cycle. A fixed
  index then samples one phase only.

So v4txa11k samples at a rotating index: round r samples before pass (13 r + 5) mod 32.
- 13 is odd, so over 32 rounds every index is sampled once. For any period dividing 32, every phase is sampled
  equally.
- The room value is the sender's own count, which changes only in visits. The sample's few cycles before its pass do
  not change what it reads, and it reads the channel registers first.
- Left open: traffic whose period tracks 13 passes per round could still alias. Not expected, not excluded.

**Build checks (mock, 2026-11-03; scratch builds, pairs by class and .text shift):**
- **Off = v4txa11, byte-identical:** Galaxy (frg_nsrx, frg_torus, check_2d) 408 / 408; P150 (p150_it_ring,
  torusxy_2x4) 1056 / 1056 router ELFs. These are the same cases as v4txa11j's check.
- **On (`TT_FABRIC_PHASE_PROFILE=1`):** every mock row reaches launch (Galaxy 3 / 3, P150 12 / 12). The samples are
  compiled into every ring and torus E0, and the 2D mesh has none.
- **On, `loopvars --headroom` (off vs on, every distinct image: Galaxy 37, P150 22):** all pass, and no hot value
  moves to the stack.

  | E0 | `.text` | carried registers | loop instructions | sp accesses in the loop |
  | --- | --- | --- | --- | --- |
  | 1D ring (Galaxy, P150) | +796 B | 14 -> 17 | 317 -> 339 | 34 -> 37 |
  | torus (Galaxy, P150) | +868 B | 13 -> 17 | 642 -> 671 | 77 -> 87 |
  | 2D mesh (no samples) | +8 B (host L1 shift) | 13 -> 13 | 499 -> 499 | 52 -> 52 |

  The added sp accesses are saves and restores around the out-of-line sample call (the tool does not class them as
  hot).
- **Earlier shapes:** two inlined shapes (with and without a lambda for the pass) showed hot values on the stack.
  Those checks used pairs that were not matched: the first image per class, where a class holds router images
  36-460 B apart. So their failures are not established. Only the out-of-line shape was checked on exact pairs, and
  it is the one built.

**Prediction 4, replaced (hypothesis):** on vs off within 2% on every row: one compare per pass, plus once-per-round
and once-per-16-rounds out-of-line work. Measured off vs on in the same job, two passes, as before. The rows, steps
and decision rule are unchanged (§8).

### 8.3 v4txa11k on Galaxy 171300 (a03u08): the room test does not separate (2026-11-03)

The job ran 14 / 14 steps (validation 10 / 10 on the on-arm). Results are in `results/glx171300`. The first
submission, 171298, landed on a02u02 while another user's out-of-Slurm job was running there. It stopped at its
busy-node check before touching the device.

**Cost (cycles per packet, off / on in the same pass; pass a = off then on, pass b = on then off):**

| row | pass a | pass b | on vs off |
| --- | --- | --- | --- |
| 8-ring 3072 B | 191.9 / 192.0 | 191.9 / 192.0 | +0.05% |
| 8-ring 3584 B | 305.7 / 331.2 | 306.8 / 332.9 | +8.3% / +8.5% |
| 8-ring 4352 B | 378.7 / 390.4 | 374.0 / 391.5 | +3.1% / +4.7% |
| 4-ring 4352 B | 339.2 / 339.3 | 324.2 / 324.0 | 0.0% / -0.1% |
| torus 3072 B | 215.2 / 215.7 | 215.2 / 215.7 | +0.2% |
| torus 4352 B | 261.6 / 265.1 | 261.4 / 266.1 | +1.3% / +1.8% |

- Prediction 4 (within 2% on every row) is **WRONG** for the 8-ring at 3584 and 4352 B.
- Confound: the on-arm's host block allocation shifts L1, and the 1D ring's worker channel drops from 26 to 25 slots
  (logged). The torus slots are unchanged. Which of the instrument and the slot loss costs what is not separated.
- The 4-ring row moves by pass position (339 in pass a, 324 in pass b), the same in both arms and in 171203. On vs
  off is compared within a pass.

**Counts (`jam_report.py galaxy`; the two passes agree within 1 point; room histogram from pass a, buckets 0..11+):**

| row | axis, R | contests | room >= R | room < 2 | in-ring ready, no far slot | injection ready, < 2 far slots | median in-ring backlog | room histogram |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 8-ring 3072 B (control) | Y, 8 | 308k | 0% | 88% | 29% | 88% | 2-3 | 89527 180633 36258 1409 16 1 3 3 0 2 0 23 |
| 8-ring 3584 B (jam) | Y, 8 | 739k | 0% | 95% | 72% | 95% | 11+ | 533845 166973 34621 3080 242 33 119 4 3 2 1 14 |
| 8-ring 4352 B (jam) | Y, 8 | 920k | 0% | 96% | 78% | 96% | 11+ | 716697 164583 33237 4393 415 23 19 7 20 6 34 204 |
| 4-ring 4352 B | X, 4 | 155k | 20% | 76% | 59% | 75% | 11+ | 93486 23218 5089 1148 1255 674 1492 615 1309 593 1699 23946 |
| torus 3072 B | X, 4 / Y, 8 | 81k / 298k | 0% / 0% | 63% / 88% | 10% / 28% | 62% / 88% | 1 / 2 | |
| torus 4352 B | X, 4 / Y, 8 | 101k / 381k | 0% / 0% | 80% / 91% | 37% / 46% | 78% / 91% | 1 / 2 | |

- Far ring: 32 slots on the 1D ring, 16 on the torus. Contests are 99-100% of in-ring-ready samples on every ring
  and torus row: whenever the in-ring channel holds a packet, an injection channel does too.

**Decision rule (§8): does not separate.**
- Its first condition needs room >= 8 in at least 70% of the contests on the Galaxy jam rows. The rows show 0% (3584
  and 4352 B, both passes).
- The rule needs both conditions, so the P150 half (lb2 171279) cannot change the verdict.
- Per the user's decision: stop and report. No C2 / C1 build, no fallback without the user.
- The counts come from the on-arm, which ran 3-8% slower on the jam rows. Two other measurements show the same
  far-ring-full picture:
  - the 4-ring row, where the on-arm cost nothing: room < 2 in 76%, room >= 4 in 20%;
  - v4txa11j's 1.6-2.5x-perturbed run (§8.2): 95% room < 2 on the 8-ring.

**What this says about §3 / §4 (measured state, not a bound claim):**
- §4's hypothesis that "in the jam the far ring is mostly free (E1 drains faster than the link fills it)" is
  **WRONG** for v4txa11 on Galaxy 200G. In the jam rows the sender sees 0 or 1 free far slots in 95-96% of contests.
  That is the bubble check's own view of the far ring.
- §3's mechanism needs at least 2 free far slots in a contest: each pass then sends one injection packet and one
  in-ring packet, and the worker gets half the link. In the jam rows that holds in 4-5% of contests. In 95-96% the
  bubble rule refuses the injection channel, and in 72-78% of in-ring-ready samples the in-ring channel has no far
  slot either. So §3's mechanism is **WRONG** as the description of these rows. Its gloss "fewer than 2 free slots,
  that is, when the far receiver (E1) is the constraint" is also **WRONG** as an equation. A full far ring can come
  from E1 blocked on its output (the next E0's in-ring channel, backlog 11+), not from E1's service time.
- No conflict with H9's RX-delay A/B (FULLRING_MCAST_GAP.md §8.10: the slow mode needs E1 faster than the link).
  These counts say nothing about which core bounds the row.
- What does differ between the jam rows and the 3072 B control is the in-ring channel:
  - backlog 11+ vs 2-3;
  - in-ring ready with no free far slot: 72-78% vs 29%.

  Both buffers of a hop (E1's receiver ring and the next E0's in-ring channel) are full in the jam rows. That fits
  the paper's full-ring state (FULLRING_MCAST_GAP.md §8.13) with the queue in both buffers, not in the forwarding
  channel alone. This is a reading, not tested.
- The send ratio (injection / in-ring sends: 0.17 on the 8-ring, 0.50 on the 4-ring) is fixed by the pattern over a
  whole test. It says nothing about the share during contests.

### 8.4 In-ring state as the separator: P150 predictions and decision rules (written before lb2 171279 lands)

The coordinator's pointer (2026-11-03): in 171300 the in-ring channel's state separates the Galaxy jam rows from the
3072 B control. That is close to C3 (a trigger on the in-ring channel's far-credit refusals, §5.3) and to a backlog
trigger. Both were set aside by reasoning ("it fires when E1 is the constraint, on P150"), not by data. lb2 171279
(P150, v4txa11k, both arms, two passes) gives that data. No build and no runs beyond 171279.

**The two separators (round samples, as v4txa11k records them):**
- **B, backlog:** the share of contests in which the in-ring channel holds at least 8 packets. It has 25 slots on the
  1D ring (Galaxy and P150; 26 / 25 / far 32 logged on both). The `phase_ticks` histogram gives it.
- **S, in-ring starvation:** the share of in-ring-ready samples with no free far slot. A visit then refuses the
  in-ring channel, which is C3's event. Columns `tx_blk_credit / idle`.

**Galaxy 171300, measured** (both passes within 1 point; `jam_report.py` now prints backlog >= 4 / >= 8 / 11+ and
the backlog histogram):

| row | B: backlog >= 8 | backlog >= 4 | median backlog | S: in-ring ready, no far slot |
| --- | ---: | ---: | ---: | ---: |
| 8-ring 3584 B (jam) | 100% / 99% | 100% | 11+ | 72% |
| 8-ring 4352 B (jam) | 100% | 100% | 11+ | 78% |
| 4-ring 4352 B (jam: the probe took it 339 -> 255) | 94% | 96% | 11+ | 59% |
| 8-ring 3072 B (control, router-bound) | 3% | 32% | 2-3 | 29% |
| torus 3072 B, X / Y | 0% / 1% | 0% / 14% | 1 / 2 | 10% / 28% |
| torus 4352 B, X / Y | 0% / 3% | 0% / 19-20% | 1 / 2 | 37% / 46-47% |

- B has a wide gap on Galaxy: 94-100% vs 3%.
- S has a narrow one: 59-78% vs 29%. The 4-ring jam row (59%) sits near the torus Y at 4352 B (47%), which the
  probe barely moved (270 -> 260).

**P150 predictions per row (hypothesis; written before any P150 count):**

The basis:
- H9 (its A/B: the RX-delay sweep, FULLRING_MCAST_GAP.md §8.10) puts these rows in the E1-bound regime. At 400G the
  wire is about 88 / 124 cycles per packet at 3072 / 4352 B. The rows run at 141-265, so v4's E1 is slower than the
  wire, more so than in the Galaxy 3072 B control (E1 ~192 against wire 176).
- With the far ring the constraint, the bubble rule gives the lone free far slot to the in-ring channel: it needs 1,
  injection needs 2. So the in-ring channel drains as fast as slots free, and its backlog stays low, as in the
  control. **WRONG** (171410, §8.5): 8 of 12 P150 multicast rows exceed 30% backlog >= 8 in at least one pass (up to
  100%).
- The in-ring channel still waits for far slots, so S can be as high as the control's 29% or higher. That holds
  most on scatter, where E1 is slowest.
- Probe evidence (FULLRING_MCAST_GAP.md §8.18): on RingMulticast write, the backlog >= 1 gate cost +20 and the
  backlog >= 4 gate +8..15. So backlog >= 4 is not rare there. HalfRingMulticast write moved +3..4.5 at T 1 and 0 at
  T 4. Scatter rows and UnidirRingMulticast barely moved.

| P150 row (both sizes) | median backlog | backlog >= 4 | B: backlog >= 8 | S: in-ring ready, no far slot | room < 2 / room >= 8 |
| --- | ---: | ---: | ---: | ---: | --- |
| UnidirRingMulticast write | 1-3 | <= 35% | <= 10% | 20-50% | >= 75% / <= 5% |
| UnidirRingMulticast scatter | 1-2 | <= 25% | <= 5% | 25-60% | >= 80% / <= 5% |
| HalfRingMulticast write | 1-2 | <= 20% | <= 5% | 15-45% | >= 70% / <= 5% |
| HalfRingMulticast scatter | 1-2 | <= 20% | <= 5% | 20-55% | >= 75% / <= 5% |
| RingMulticast write | 2-4 | 20-50% | <= 15% | 20-50% | >= 70% / <= 5% |
| RingMulticast scatter | 1-2 | <= 25% | <= 5% | 25-60% | >= 80% / <= 5% |
| Ring1HopEW write / scatter | no in-ring traffic: in-ring ready in < 1% of samples, contests ~0 (excluded) | | | | |

- Room histogram: mode at 0 or 1 free far slots (32), little above 4.
- **Separating thresholds, predicted:**
  - B separates every P150 multicast row from the Galaxy jam rows at backlog >= 8: <= 15% predicted against
    94-100% measured.
  - S separates them from the 8-ring jam rows (72-78%) only if every P150 row stays below about 65%. It separates
    them from the 4-ring jam row (59%) only if every P150 row stays below about 50%. The predicted ranges reach 60%
    on scatter, so S is predicted to separate weakly or not at all.

**Decision rules, fixed now.** T = 8 and the S bounds were chosen from the Galaxy counts already seen; the P150 counts
are the test.
- **Rule B (backlog >= 8) separates if all three hold:**
  1. at least 70% of contests on the Galaxy jam rows (8-ring 3584 / 4352 B, 4-ring 4352 B): measured 94-100%, met;
  2. at most 30% on the Galaxy 8-ring 3072 B control: measured 3%, met;
  3. at most 30% on every P150 multicast row (Unidir / Half / Ring multicast x write / scatter x 3072 / 4352 B: 12
     rows), in each pass. Ring1HopEW is excluded (no contests).
- **Rule S (in-ring ready, no far slot) separates if all three hold:**
  1. at least 55% of in-ring-ready samples on the same Galaxy jam rows: measured 59-78%, met;
  2. at most 40% on the 3072 B control: measured 29%, met;
  3. at most 40% on every P150 multicast row, in each pass.
- **Perturbation check:** a P150 row whose on-arm runs more than 5% slower than its off-arm in the same pass is
  flagged. Its counts are reported, but the rule is not scored on it without the user.
- **What a verdict means:**
  - A rule that separates shows the regimes differ in that state. It does not make a trigger on it free: the probe's
    backlog >= 1 gate fired on P150 and cost +20 on RingMulticast write. A trigger's cost and effect are their own
    A/B.
  - Either way: report, and no build without the user.

### 8.5 P150 lb2 171410 (bh-lb-120-a07u28): rule S separates, rule B does not (2026-11-03)

**Run.**
- Submitted with `lb2_run.sh -n any` (one job, first approved LB free). It landed on **bh-lb-120-a07u28** and ran
  23:05-23:07 UTC with rc 0, no hang. The link check passed (58 / 58 baseline cores UP / PASS / 400G).
- Validation `val_ring_p150` passed 19 / 19 in both arms. `p150_it_ring` passed 16 / 16 per arm and pass.
- Results: `results/frgk{v,a,b}_{off,on}_ring*`, manifest `results/lb2/171410_a07u28.tsv`. Both arms ran in this one
  job on a07u28.
- Earlier submissions, none of which reached the device: 171279 failed the stale-.git check; 171327, 171366 and
  171394 were cancelled while pending.
- Slots: the on-arm's worker channel is 25 instead of 26 (host L1 shift), as on Galaxy.

**Cost (cycles per packet, off -> on, mean of passes a / b; `frg_tables.py p150jam results frgk`).** No row is
flagged (> 5%):
- RingMulticast write 3072 B: 141.9 -> 147.9 (+4.2%);
- RingMulticast write 4352 B: 150.8 -> 153.6 (+1.9%);
- every other row: within 0.2%.

**Counts (`jam_report.py p150`; R = 8; pass a / pass b):**

| P150 row | B: backlog >= 8 | S: in-ring ready, no far slot | median backlog | room < 2 | room >= 8 | contests / in-ring ready |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| UnidirRingMulticast write 3072 B | 98% / 98% | 11% / 11% | 11+ | 99% | 0% | 39% |
| UnidirRingMulticast write 4352 B | 78% / 79% | 15% / 16% | 11+ | 93% | 0% | 41% |
| UnidirRingMulticast scatter 3072 B | 88% / 88% | 22% / 21% | 11+ | 100% | 0% | 46% |
| UnidirRingMulticast scatter 4352 B | 81% / 90% | 28% / 28% | 11+ | 100% | 0% | 50% |
| HalfRingMulticast write 3072 B | 16% / 16% | 15% / 16% | 2 | 77% | 0% | 88% |
| HalfRingMulticast write 4352 B | 46% / 48% | 15% / 15% | 6-7 | 74-75% | 0% | 88-89% |
| HalfRingMulticast scatter 3072 B | 0% / 0% | 18% / 18% | 1 | 82-83% | 0% | 87% |
| HalfRingMulticast scatter 4352 B | 0% / 0% | 18% / 18% | 1 | 81% | 0% | 89-90% |
| RingMulticast write 3072 B | 100% / 100% | 20% / 20% | 11+ | 68-69% | 0% | 89% |
| RingMulticast write 4352 B | 82% / 82% | 22% / 22% | 11+ | 69% | 0% | 85% |
| RingMulticast scatter 3072 B | 30% / 29% | 25% / 25% | 1 | 83-84% | 0% | 100% |
| RingMulticast scatter 4352 B | 33% / 29% | 32% / 32% | 1 | 81% | 0% | 100% |
| Ring1HopEW (4 rows) | no contests (in-ring ready in 30-51 samples) | | | | | |

- Room histograms (pass a, free far slots 0..11+ out of 32) peak at 1 on every multicast row. Examples:
  - Unidir write 3072 B: 942 2415 29 0 ...;
  - Ring write 4352 B: 2073 3492 2124 311 70 ...;
  - Half write 4352 B: 10045 35214 14243 1339 ....
- Backlog histograms of the backlogged rows pile up at 11+:
  - Unidir write 3072 B: 3093 of 3391 contests;
  - Ring write 3072 B: 8162 of 8193.

**Verdicts (rules fixed in §8.4):**
- **Rule B (backlog >= 8): does NOT separate.** 8 of the 12 P150 multicast rows exceed 30% in at least one pass.
  - 7 do in both passes: Unidir all four (78-98%), Ring write both (82-100%), and Half write 4352 B (46-48%). Six of
    them reach the Galaxy jam rows' level.
  - RingMulticast scatter 4352 B does in pass a (33%).
- **Rule S (in-ring ready with no free far slot): separates.**
  - Galaxy jam rows: 59-78% (>= 55%: met).
  - Galaxy 3072 B control: 29% (<= 40%: met).
  - Every P150 multicast row in both passes: 11-32% (<= 40%: met).
  - Margin: the highest P150 row (RingMulticast scatter 4352 B, 32%) against the lowest jam row (4-ring, 59%).
  - Not in the rule: Galaxy torus Y at 4352 B sits between them at 46-47%. The probe moved that row only 270 -> 260.

**Predictions (§8.4), checked:**
- Backlog: **WRONG** on every row except HalfRingMulticast scatter (0%, as predicted) and Half write 3072 B (16%;
  predicted <= 5%, also wrong). The hypothesis "with the far ring the constraint, the in-ring channel drains as fast
  as slots free and its backlog stays low" is **WRONG** on P150. Its in-ring channel is as backlogged as in the
  Galaxy jam, yet it finds a free far slot in 68-89% of its ready samples.
- S: inside or below the predicted ranges (11-32% against 15-60%). The verdict prediction "S separates weakly or not
  at all" is **WRONG**: it separates with a 27-point margin.
- Room: room >= 8 in 0% (predicted <= 5%: right). Room < 2 in 68-100% (predicted >= 70-80%: RingMulticast write's
  68-69% is just under).

**What this says (state counts; no bound or cause claim):**
- Both systems hold a large in-ring backlog on their busiest rows. What differs is whether the in-ring channel, when
  it holds a packet, finds the far ring with no slot at all:
  - Galaxy jam: 59-78% of in-ring-ready samples;
  - P150 and the Galaxy control: 11-32%.
  - On P150 the room histogram's mode is 1: the in-ring channel's one-slot admission is usually met, and injection's
    two-slot admission is not.
- §5.3's reason for rejecting C3 was "it fires when the far ring is the constraint (P150 at 400G, Galaxy at <= 3072
  B) ... and it would rarely fire in the Galaxy jam". That reason is **WRONG** by these counts. In-ring refusals are
  2-7x more frequent (as a share of in-ring-ready samples) in the Galaxy jam than on P150 or the 3072 B control.
- They still happen on P150 (11-32%). So a trigger that engages on any single refusal would still fire there. A
  refusal-based trigger would need a rate (for example, refusals per in-ring-ready pass over a window) with its
  threshold between 32% and 59%. That is design, for the user; whether such a trigger removes the jam, and what it
  costs on P150, are its own A/Bs.
- No build. Per the user's rule: report, and the next step is the user's.

## 9. Bubble-threshold sweep (user go, 2026-11-03): an A/B of the injection threshold T

**Variant `variants/v4txa11t`** (README there):
- An injection channel on a deadlock-avoiding VC sends only with T free far slots (`remote_free_slots_to_send`). The
  injection channels are the worker, plus the turns on a torus.
- T comes from `TT_FABRIC_EXP_DRISC_DELAY`. Unset means 2, v4txa11's rule. The in-ring channel's 1 is unchanged. One
  T applies to every router in an arm.
- With the switch off, mock builds are byte-identical to v4txa11: Galaxy 408 / 408, P150 1056 / 1056. T 6 reaches
  launch on every mock row.
- No predictions: a plain sweep (user). Only throughput was expected to change.

### 9.1 Galaxy 171672 (bh-glx-110-a03u08, 23:56-00:23 UTC; `results/glx171672`; `bt_report.py glx`)

**Run.** One job, tree `v4txa11t_1103` (base_1102), configs a11t2..a11t6, `VAL_STOPS_ARM=1`.
- The check (T2) passed 1 / 1. glx_frg_val passed 10 / 10 on T3, T4, T5 and T6. All 45 steps exited 0: no hang, no
  timeout.
- Pass a ran suite by suite with arms T2 -> T6; pass b with arms T6 -> T2.
- The first submission, 171630, landed on a01u14, whose /tmp was full. Its T2 check segfaulted in MPI init ("No
  space left on device"), before the device ran anything.
- Slots, logged by every run and the same in every arm:
  - 1D ring: {26, 25} / receiver {32};
  - torus: {17, 16, 16, 16} / {16};
  - five more lines of {16, 0} / {32} (1D) or {16, 0, 0, 0} / {16} (torus) per run.

**Cycles per packet**, pass a / pass b. % is each arm's mean against T2's mean. `*` marks passes that differ by more
than 1%.

| row | T2 | T3 | T4 | T5 | T6 | main (cross-job) |
| --- | --- | --- | --- | --- | --- | ---: |
| 8-ring 3072 B | 191.9 / 191.9 | 191.6 / 191.6 (-0.2%) | 191.6 / 191.7 (-0.1%) | 191.5 / 191.5 (-0.2%) | 191.6 / 191.5 (-0.2%) | 261-267 |
| 8-ring 3584 B | 306.6 / 306.2 | 293.5 / 295.6 (-3.9%) | 287.6 / 286.5 (-6.3%) | 233.9 / 234.3 (-23.6%) | 222.4 / 226.5* (-26.7%) | 261.3 |
| 8-ring 4352 B | 375.2 / 374.9 | 367.0 / 368.6 (-1.9%) | 368.3 / 366.3 (-2.1%) | 365.1 / 363.3 (-2.9%) | 362.1 / 361.0 (-3.6%) | 263.8 |
| 4-ring 4352 B | 324.4 / 324.5 | 336.8 / 336.8 (+3.8%) | 336.8 / 337.0 (+3.8%) | 336.9 / 336.9 (+3.8%) | 340.2 / 336.9 (+4.3%) | 342.7 |
| torus 3072 B | 215.2 / 215.2 | 214.4 / 214.4 (-0.4%) | 214.4 / 214.4 (-0.4%) | 214.4 / 214.4 (-0.4%) | 214.4 / 214.4 (-0.4%) | |
| torus 4352 B | 261.5 / 261.7 | 260.5 / 260.5 (-0.4%) | 260.6 / 260.6 (-0.4%) | 260.5 / 260.5 (-0.4%) | 260.6 / 260.7 (-0.4%) | |
| half ring 3072 B | 186.3 / 186.2 | 186.1 / 186.1 (-0.1%) | 186.1 / 186.1 (-0.1%) | 186.1 / 186.2 (-0.1%) | 186.2 / 186.2 (-0.1%) | |
| half ring 4352 B | 332.1 / 331.1 | 313.2 / 318.0* (-4.8%) | 320.0 / 324.7* (-2.8%) | 328.2 / 329.5 (-0.8%) | 325.8 / 332.5* (-0.7%) | |
| ring 1-hop 3072 / 4352 B | 177.0 / 249.5 | 177.0 / 249.5 | 177.0 / 249.5 | 177.0 / 249.5 | 177.0 / 249.5 | |

Main's numbers are from other jobs: FULLRING_MCAST_GAP.md A/B 1 and A/B 8, Galaxy 169776. The 8-ring row there is the
full ring (A/B 8); 3072 B is 261.3-267.3 across its rows.

**What the numbers show:**
- 8-ring 3584 B falls step by step with T. At T 5-6 (222-234) it is below main's 261.3.
- 8-ring 4352 B moves 2-4% (375 -> 361), far from main's 263.8.
- 4-ring 4352 B rises about 4% at every T >= 3.
- Half ring 4352 B gains most at T 3 (-4.8%) and less at higher T. Three of its arms have passes more than 1% apart.
- Torus rows: -0.4% at every T >= 3. The 3072 B rows and 1-hop rows do not move beyond 0.4%.

### 9.2 P150 lb2 171629 (bh-lb-120-a07u28, 00:41-00:52 UTC; `results/frgbt*`; `bt_report.py p150 results`)

**Run.** One job via `lb2_run.sh -n any`; all arms ran in it, on a07u28.
- **Host build:** stage src-20261104-001304 / full-20261104-001304. That is the dyn-slots host library (off by
  default), from the re-stage after broker 648. Do not compare these numbers with P150 runs from before that
  re-stage.
- **Validation** (T3-T6; val_ring_p150, plus val_torus_xy_p150 on 2x4 and 4x2): 19 / 19, 12 / 12 and 12 / 12 in every
  arm.
- **Benchmarks:** all 30 runs exited 0 (ring 16 tests, each torus 80), with no hang. The link check after the job
  passed (58 / 58).
- **Slots, logged and the same in every arm and pass:**
  - ring: {26, 25} / {32};
  - torus 2x4 and 4x2: {17, 16, 16, 16} / {16};
  - plus five lines of {16, 0...} per run.

**Ring rows** (cycles per packet, pass a / pass b; % against T2's mean; no pass pair differs by more than 1%):

| row | T2 | T3 | T4 | T5 | T6 |
| --- | --- | --- | --- | --- | --- |
| RingMulticast write 3072 B | 142.1 / 142.0 | 141.2 / 141.2 (-0.6%) | 141.1 / 141.1 (-0.6%) | 141.1 / 141.1 (-0.6%) | 141.1 / 141.1 (-0.6%) |
| RingMulticast write 4352 B | 150.9 / 150.9 | 146.4 / 146.5 (-2.9%) | 150.4 / 150.4 (-0.3%) | 146.5 / 146.5 (-2.9%) | 146.5 / 146.6 (-2.9%) |
| HalfRingMulticast write 4352 B | 172.9 / 172.9 | 172.1 / 172.2 (-0.5%) | 172.0 / 172.0 (-0.5%) | 172.1 / 172.1 (-0.5%) | 172.2 / 172.2 (-0.4%) |
| UnidirRingMulticast write 3072 / 4352 B | 198.8 / 219.4 | 199.0 / 219.4 | 199.0 / 219.3 | 199.0 / 219.3 | 199.0 / 219.3 |
| every scatter row, Half write 3072 B, Ring1HopEW | | within +-0.2% | within +-0.2% | within +-0.2% | within +-0.2% |

**XY torus rows** (5 sizes, 256-4352 B):
- **2x4:** every row is within +-0.4% of T2 at every T, except TorusXYMulticast write 4352 B at T6: 128.6 / 128.3
  against 126.8 / 126.8 (+1.3%).
- **4x2 unicast rows:** within +-0.2%.
- **4x2 TorusXY4x2_Multicast (write and scatter):** the noisy rows. 25 of their 50 arm cells, T2's included, have
  passes more than 1% apart. Their means move -1.7% to +3.9%. The largest moves are on write at 4352 B: T5 +3.0% and
  T6 +3.9% (192.8 / 195.6*), against T2 190.6 / 183.1*.

**What the numbers show (both systems):**
- The large changes are on the Galaxy 8-ring:
  - 3584 B: -24% at T5 and -27% at T6, below main;
  - 4352 B: -2% to -4%.
- Galaxy 4-ring 4352 B is +3.8-4.3% at every T >= 3.
- Galaxy half ring 4352 B is -0.7% to -4.8%, with passes apart.
- On P150, every ring row and every 2x4 torus row moves 3% or less. The largest move is RingMulticast write 4352 B at
  -2.9%, though T4 there is -0.3%.
- The 4x2 multicast rows move within their own pass-to-pass noise.
- Per-axis T choices and follow-ups go to the coordinator and the user (nothing here goes beyond this variant).

### 9.3 Slot depth x injection threshold, Galaxy 171994 (bh-glx-110-a08u02, SC16, 01:34-03:05 UTC; `results/glx171994`)

**The question (user):** does slot depth on the ring close the 4352 B jam, or a larger injection threshold? The slot
rule under test is worker + receiver <= forwarding.

**Setup.**
- Variant v4txa11t, T range widened to 2..16. The switch-off images are still byte-identical to v4txa11: Galaxy
  408 / 408, P150 1056 / 1056.
- Tree v4txa11t_1103. Slots set with TT_FABRIC_EXP_SLOTS_1D (worker, forwarding, receiver), the same counts at every
  size; every receiver a power of 2; every row sums to 83 slots at 4352 B.

| arm | slots (W / F / R) | about the arm |
| --- | --- | --- |
| A | 26 / 25 / 32 | today |
| B | 42 / 25 / 16 | the smaller receiver alone |
| C | 16 / 51 / 16 | user rule |
| D | 8 / 59 / 16 | user rule, small worker |
| E | 4 / 63 / 16 | the worker as a tight credit window |
| F | 8 / 43 / 32 | user rule, today's receiver |
| N | none (defaults) | reference |

- Configs (arm + T): N2, A2, B2, C2, D2, E2, F2; C6, D6, E6, F6; A8, A12, A16. T 2 is the switch off.
- Rows: glx_frg_nsrx (8-ring 3072 / 3584 / 4352 B), glx_frg_ew (4-ring 4352 B), glx_frg_short4k (half ring and ring
  1-hop at 4352 B).
- Order: the 2D check, then glx_frg_val on all 14 configs (VAL_STOPS_ARM=1), then pass a over the configs in the order
  above, then pass b reversed. Suite by suite.

**Run.**
- All 99 steps exited 0: check 1 / 1; glx_frg_val 10 / 10 on all 14 configs; no hang, no timeout, no ring-mapping
  skip.
- Slots, logged by every run, exactly as asked. The 1D ring routers got:
  - A, A8, A12, A16, N: {26, 25} / {32};
  - B: {42, 25} / {16};
  - C: {16, 51} / {16};
  - D: {8, 59} / {16};
  - E: {4, 63} / {16};
  - F: {8, 43} / {32}.

  Each run also logged five lines of {16, 0} / {32}.
- Cap: the study had 4 Galaxy allocations when it started.

**Cycles per packet, mean of passes a / b** (% against A2's mean; `*` where the passes differ by more than 1%, with
both passes shown):

| config | 8-ring 3072 | 8-ring 3584 | 8-ring 4352 | 4-ring 4352 | half ring 4352 | ring 1-hop 4352 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A2 (today) | 191.9 | 305.4 / 308.8* | 377.0 | 324.2 | 331.5 | 249.5 |
| N2 (defaults) | 191.9 (0.0%) | 306.1 (-0.3%) | 377.6 (+0.2%) | 324.1 (0.0%) | 331.3 (0.0%) | 249.5 |
| B2 42/25/16 | 192.2 (+0.2%) | 209.1 (-31.9%) | 364.3 / 254.4* (-17.9%) | 255.1 (-21.3%) | 253.7 (-23.4%) | 249.4 |
| C2 16/51/16 | 191.1 (-0.4%) | 209.0 (-32.0%) | 254.3 (-32.5%) | 255.1 (-21.3%) | 253.7 (-23.4%) | 249.6 |
| D2 8/59/16 | 190.9 (-0.5%) | 209.0 (-32.0%) | 254.3 (-32.5%) | 255.1 (-21.3%) | 255.1 (-23.0%) | 249.6 |
| E2 4/63/16 | 191.0 (-0.5%) | 209.0 (-32.0%) | 254.3 (-32.5%) | 255.1 (-21.3%) | 263.3 (-20.5%) | **437.1 (+75.2%)** |
| F2 8/43/32 | 190.9 (-0.5%) | 296.3 (-3.5%) | 375.1 (-0.5%) | 273.9 (-15.5%) | 282.1 (-14.9%) | 249.6 |
| C6 | 196.1 (+2.2%) | 211.5 (-31.1%) | 254.3 (-32.5%) | 255.2 (-21.3%) | 253.8 (-23.4%) | 249.6 |
| D6 | 196.7 (+2.5%) | 212.6 (-30.8%) | 254.3 (-32.5%) | 255.2 (-21.3%) | 257.2 (-22.4%) | 249.6 |
| E6 | 200.4 (+4.4%) | 215.1 (-30.0%) | 254.3 (-32.5%) | 255.2 (-21.3%) | 276.8 (-16.5%) | **437.3 (+75.2%)** |
| F6 | 190.6 (-0.7%) | 209.0 (-31.9%) | 304.4 (-19.2%) | 274.4 (-15.4%) | 282.2 (-14.9%) | 249.6 |
| A8 | 191.6 (-0.2%) | 208.9 (-32.0%) | 359.7 (-4.6%) | 337.0 (+3.9%) | 331.3 (0.0%) | 249.5 |
| A12 | 191.6 (-0.2%) | 209.1 (-31.9%) | 327.3 / 330.7* (-12.7%) | 336.7 (+3.9%) | 299.7 / 305.2* (-8.7%) | 249.5 |
| A16 | 191.6 (-0.2%) | 209.0 (-31.9%) | 254.2 (-32.6%) | 255.1 (-21.3%) | 253.8 (-23.4%) | 249.5 |
| main (cross-job) | 261-267 | 261.3 | 263.8 | 342.7 | - | 248.6 |

Main's numbers come from other jobs: FULLRING_MCAST_GAP.md A/B 1 / A/B 8 (169776); the 1-hop is the A11 sweep
(170094 / 170095).

**What the numbers show:**
- **8-ring 4352 B.** At 254.2-254.3, 32.5% below today and below main's 263.8:
  - with a 16-slot receiver at T 2 or T 6: C, D and E;
  - with B in pass b only (pass a 364.3);
  - with today's slots at T 16.
  It stays at 360-376 with a 32-slot receiver at T 2-8 (A, F, A8). In between: A12 327-331, F6 303-305.
- **8-ring 3584 B.** 209 in every config except A2, N2 (305-309) and F2 (296), and 211-215 for C6, D6, E6.
- **4-ring 4352 B.** 255.1-255.2 for B, C, D, E (T 2 and 6) and A16; 274 for F2 and F6; 337 for A8 and A12. Main is
  342.7.
- **Half ring 4352 B.** 253.7-255 for B, C, D (T 2), C6 and A16; 257-277 for D6, E2 and E6; 282 for F; 300-305 for A12;
  331 for A8 and today.
- **8-ring 3072 B.** Within 0.7% everywhere except T 6 with a 16-slot receiver (C6 +2.2%, D6 +2.5%, E6 +4.4%).
- **Ring 1-hop 4352 B.** 437 (+75%) with the 4-slot worker (E2, E6), 249.4-249.6 in every other config.
- **Pass differences over 1%:** A2 8-ring 3584, B2 8-ring 4352 (364.3 / 254.4), A12 8-ring 4352, A12 half ring.
- The rule decision (slots, T, per axis) goes to the user. This section stops at the numbers.

### 9.4 Costing the two candidate rules: A16 (today's slots, T = R/2 = 16) and C2 (16/51/16 at T 2)

**Galaxy 172155** (bh-glx-110-a08u02, SC16, 03:13-03:33 UTC; `results/glx172155`). Two more passes of A16 and C2,
with A2 as the in-job reference, plus the torus rows at T 2 against T = R/2 = 8 (torus receivers are 16 slots).
- **Run:** 26 / 26 steps exited 0. Check 1 / 1; glx_frg_val 10 / 10 on C2, A16 and N8; no hang.
- **Slots as asked:**
  - A2 and A16: {26, 25} / {32};
  - C2: {16, 51} / {16};
  - torus (N2, N8): {17, 16, 16, 16} / {16}.

| row | A2 (a / b) | C2 (a / b) | A16 (a / b) |
| --- | --- | --- | --- |
| 8-ring 3072 B | 191.9 / 191.9 | 191.0 / 191.0 (-0.4%) | 191.6 / 191.6 (-0.2%) |
| 8-ring 3584 B | 305.6 / 308.6 | 209.0 / 209.0 (-32.0%) | 209.1 / 209.1 (-31.9%) |
| 8-ring 4352 B | 377.0 / 376.0 | 254.3 / 254.4 (-32.4%) | 254.2 / 254.2 (-32.5%) |
| 4-ring 4352 B | 324.1 / 324.5 | 255.1 / 255.1 (-21.3%) | 255.1 / 255.1 (-21.3%) |
| half ring 4352 B | 331.8 / 331.2 | 253.7 / 253.7 (-23.5%) | 253.8 / 253.7 (-23.4%) |
| ring 1-hop 4352 B | 249.5 / 249.5 | 249.6 / 249.6 (0.0%) | 249.5 / 249.5 (0.0%) |

| torus row | T 2 (N2) | T 8 (N8) |
| --- | --- | --- |
| 3072 B | 215.2 / 215.2 | 223.7 / 223.7 (+4.0%) |
| 4352 B | 262.1 / 261.6 | 264.3 / 264.3 (+1.0%) |

- **The slow mode on these rows:** 8-ring 4352 B at ~360-378, as A2 and B2's pass a (364) sat.
  - A16 and C2 stayed out of it in all four passes so far, in two jobs (171994 and 172155): 254.2-254.4.
  - Every other row matched between the two jobs within 0.2.
  - No A16 or C2 pass differed by more than 1%.
- **Torus at T 8:** slower than T 2 on both torus rows (+4.0%, +1.0%).

**P150 broker job 673** (bh-lb-120-a09u32, 03:47-03:53 UTC; `results/frgbu*`; `bt_report.py p150u results`).
- **Why the broker:** the LB route failed three times. In lb2 172149, 172157 and 172175, a07u28's chip 6 eth (7,1)
  stayed ('2','10','400') through every reset; 172175 used the new poll, with 3 resets of 90 s each. Nothing ran.
- **Host build:** a09u32's live build_Release, the dyn-slots host library rebuilt by broker 641 / 648 (off by
  default). All arms ran in this one job.
- **Run:** exit 0, runtime 364 s; no hang; post-job health gate OK.
  - Validation: A16 and C2 on val_ring_p150, 19 / 19 each; T8 on val_torus_xy_p150, 12 / 12 on 2x4 and 12 / 12 on 4x2.
  - Benchmarks: all 14 runs passed (ring 16 tests each, torus 80 each).
- **Slots, logged:**
  - A2 and A16 ring: {26, 25} / {32};
  - C2 ring: {16, 51} / {16};
  - torus (A2, T8): {17, 16, 16, 16} / {16}.

**Ring rows: A16 and C2 against A2** (pass a / b; no pass pair more than 1% apart):

| row | A2 | A16 | C2 |
| --- | --- | --- | --- |
| Ring1HopEW scatter 3072 B | 179.6 / 179.7 | 179.7 / 179.7 (0.0%) | 180.1 / 180.1 (+0.2%) |
| Ring1HopEW scatter 4352 B | 180.0 / 180.0 | 180.1 / 180.1 (0.0%) | 180.4 / 180.4 (+0.2%) |
| Ring1HopEW write 3072 / 4352 B | 141.1 / 140.9 | +0.0 / +0.1% | +0.3 / +0.3% |
| RingMulticast write 3072 / 4352 B | 142.0 / 150.9 | -0.6 / -0.2% | 0.0 / -0.1% |
| RingMulticast scatter, both sizes | 180.1 / 180.5 | 0.0% | +0.2% |
| HalfRingMulticast write / scatter | 171.9-225.4 | -0.2 to +0.1% | 0.0 to +0.4% |
| UnidirRingMulticast write / scatter | 198.9-265.2 | -0.1 to 0.0% | -0.3 to -0.1% |

**XY-torus rows: T 8 (= R/2) against A2 (T 2):**
- Every write row is slower at T 8, more so at larger sizes:
  - 1-hop write: +2.0-2.6% at 2048 B, +4% at 3072 B, +6.8-6.9% at 4352 B, on both 2x4 and 4x2;
  - 1-hop bidirectional write: +3.9%, +8.4-8.8%, +13.8-14.4%;
  - 2x4 3-hop EW write: +3.4%, +7.0%, +9.2%;
  - 2x4 TorusXYMulticast write: +2.1% at 256 B up to +10.1% at 4352 B;
  - 4x2 Multicast write: +2.9% at 256 B up to +14.3% at 4352 B;
  - ColMcast and RowMcast write rows: 0.0%.
- Scatter rows are within 0.2%, except 4x2 Multicast scatter (-1.1 to +3.1%, with passes apart in both arms).

**What the numbers show:**
- On P150's ring rows, A16 and C2 cost at most 0.6% on any row. Ring 1-hop scatter is +0.0% (A16) and +0.2% (C2).
- T = R/2 = 8 on the torus costs P150 2-14% on every write row. On Galaxy it is also slower than T 2 (+4.0% at 3072 B,
  +1.0% at 4352 B; §9.4 above).
- The rule decision goes to the user.

### 9.5 Fraction or cap: T = R/2 against R - T ≈ 14-16 (user go, 2026-11-04)

**The question.** Is the rule a fraction of the receiver (T = R/2) or an absolute occupancy cap (R - T ≈ 14-16 packets
allowed in the far receiver before injection is blocked)? The two coincide only at R = 32, so the arms vary R:

| arm | R (receiver) | slots (W / F / R) | T | T/R | R - T |
| --- | ---: | --- | ---: | ---: | ---: |
| C2 | 16 | 16 / 51 / 16 | 2 | 0.125 | 14 |
| C8 | 16 | 16 / 51 / 16 | 8 | 0.500 | 8 |
| A2 (today) | 32 | 26 / 25 / 32 | 2 | 0.062 | 30 |
| A16 | 32 | 26 / 25 / 32 | 16 | 0.500 | 16 |
| G2 | 64 | 18 / 17 / 64 | 2 | 0.031 | 62 |
| G32 | 64 | 18 / 17 / 64 | 32 | 0.500 | 32 |
| G48 | 64 | 18 / 17 / 64 | 48 | 0.750 | 16 |

**Build.**
- v4txa11t's T range is widened to 2..62, with a static_assert in `send<CH>` that T is below the far receiver's slots.
  The switch-off images are still byte-identical to v4txa11 (fresh reference builds): Galaxy 408 / 408, P150
  1056 / 1056.
- **R 64 fits only with a smaller router max packet.** test_tt_fabric sizes the router's buffers by
  `fabric_setup.max_packet_size`, not by the packet size. At the default (4352 B), 18 / 17 / 64 is refused by the
  host: "18 worker + 81 other slots, capacity 83".
  - So the R 64 arms run on rows with a 3584 B router max packet: `glx_frg_mps3584.yaml`, with 8-ring 3072 / 3584 B,
    4-ring 3584 B and half ring 3584 B.
  - No 4-ring or half-ring row at 3584 B existed; these are the existing rows at that size.
  - Every arm runs them, so R32/T2 is the reference there too.
- **Validation is ring-only.** T applies to every router, and the torus receivers (16 at 4096 B, 32 at 3584 B) are not
  above T 16-48. So every arm validates on the Ring rows of glx_frg_val (`glx_frg_val_ring.yaml`, 4096 B;
  `glx_frg_val_ring_mps3584.yaml` for G).
  - Note: A16 (T 16) on a 16-slot torus receiver, which ran in 172155, is now rejected by the assert.
- Mock: every arm reaches launch with its slots logged, on Galaxy and P150. T >= R fails to build (T 32 with R 32:
  no launch).

**Galaxy 172449** (bh-glx-110-a09u14, SC16, 05:14-05:40 UTC; `results/glx172449`).
- All 46 steps exited 0: check 1 / 1; ring-only validation 4 / 4 on all 7 arms; no hang.
- Slots, logged by every run, as asked:
  - C2 and C8: {16, 51} / {16};
  - A2 and A16: {26, 25} / {32};
  - G2, G32 and G48: {18, 17} / {64}.
- Order: pass a, then pass b reversed.

Cycles per packet: pass a / pass b (one number where the two passes agree to 0.1) and % against A2. `*` marks
passes more than 1% apart. **slow** marks a pass at A2's slow-mode level on a jam row.

At the default 4352 B router max packet (C and A arms):

| row | A2 (R-T 30) | C2 (R-T 14) | A16 (R-T 16) | C8 (R-T 8) |
| --- | --- | --- | --- | --- |
| 8-ring 3072 B | 191.9 / 191.9 | 191.1 / 191.0 (-0.4%) | 191.6 / 191.6 (-0.2%) | 205.4 / 205.2 (+7.0%) |
| 8-ring 3584 B | 307.1 / 305.4 slow | 209.0 / 209.0 (-31.8%) | 209.1 / 209.1 (-31.7%) | 218.3 / 218.3 (-28.7%) |
| 8-ring 4352 B | 373.3 / 374.8 slow | 254.3 / 254.3 (-32.0%) | 254.2 / 254.2 (-32.0%) | 257.9 / 257.9 (-31.0%) |
| 4-ring 4352 B | 324.3 / 324.4 slow | 255.1 / 255.1 (-21.4%) | 255.1 / 255.1 (-21.3%) | 256.4 / 256.4 (-21.0%) |
| half ring 4352 B | 332.7 / 331.2 slow | 253.7 / 253.7 (-23.6%) | 253.8 / 253.8 (-23.6%) | 254.3 / 254.4 (-23.4%) |
| ring 1-hop 4352 B | 249.5 / 249.5 | 249.6 (0.0%) | 249.5 (0.0%) | 249.6 (0.0%) |

At a 3584 B router max packet (all arms):

| row | A2 (R-T 30) | C2 (14) | A16 (16) | C8 (8) | G2 (62) | G32 (32) | G48 (16) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 8-ring 3072 B | 193.7 / 193.7 | 192.8 (-0.4%) | 193.3 / 193.4 (-0.2%) | 207.1 / 206.9 (+6.9%) | 196.9 / 196.8 (+1.6%) | 195.5 (+0.9%) | 195.3 (+0.8%) |
| 8-ring 3584 B | 301.6 / 297.9* slow | 209.0 (-30.3%) | 209.1 (-30.3%) | 218.2 (-27.2%) | 302.9 / 301.5 slow (+0.8%) | 282.1 / 279.8 (-6.3%) | 217.5 (-27.4%) |
| 4-ring 3584 B | 275.8 / 275.9 slow | 209.7 (-24.0%) | 209.7 (-24.0%) | 223.1 (-19.1%) | 270.3 / 270.4 slow (-2.0%) | 279.6 / 279.7 slow (+1.4%) | 209.6 / 209.5 (-24.0%) |
| half ring 3584 B | 267.0 / 267.6 slow | 208.5 (-22.0%) | 208.5 (-22.0%) | 220.9 / 220.8 (-17.4%) | 271.7 / 270.8 slow (+1.5%) | 279.5 / 267.9* slow (+2.4%) | 208.4 (-22.0%) |

**What the Galaxy numbers show:**
- **R - T = 14 or 16** (C2, A16, G48): 208-210 at 3584 B and 254 at 4352 B on every jam row, whatever R is.
- **R - T = 8** (C8): also out of the slow level, but 3-7% above the 14-16 arms on the 3584 B rows, and +7% on 8-ring
  3072 B.
- **R - T = 30-62** (A2, G2): at the slow level on every jam row.
- **R - T = 32** (G32, T/R 0.5): 8-ring 3584 B at 280-282 (-6%); the 4-ring and half ring at the slow level.
- **T/R = 0.5** gives 209 (A16, R 32), 218 (C8, R 16) and 280-282 (G32, R 64) on the 8-ring 3584 B row.
- Ring 1-hop does not move in any arm.
- P150 cost is broker 710, below. The rule choice goes to the user.

**P150 cost, broker job 710** (bh-lb-120-a09u32, 05:58-06:05 UTC; `results/frgbv*`; `bt_report.py p150v results`).
- **Run:** exit 0, runtime 419 s; no hang; post-job health gate OK. Host build: a09u32's live build_Release, the
  dyn-slots host library (off by default). All arms ran in this one job.
- **Validation:** val_ring_p150, and its 3584 B max-packet copy for the ring35 rows, 19 / 19 for every arm and row
  kind.
- **Benchmarks:** 22 / 22 runs passed.
- **Rows:**
  - ring = p150_it_ring (3072 / 4352 B, default 4352 B router max packet);
  - ring35 = p150_it_ring_mps3584 (3072 / 3584 B, 3584 B router max packet).
- **Slots logged:**
  - C2 and C8: {16, 51} / {16};
  - G: {18, 17} / {64};
  - A2 and A16 on ring: {26, 25} / {32}.
  - **A2 and A16 on ring35: {34, 34} / {32}.** The P150 runner sets no slot switch for the A arms, so at the 3584 B max
    packet they get the host's default allocation (R still 32). On Galaxy the A arms set 26 / 25 / 32 explicitly.
    The P150 ring35 reference is therefore R 32 with 34-slot sender channels.
- No pass pair differs by more than 1%; no row is in a slow mode (P150 has none).

% against A2, per arm (R - T in brackets):

| row | A16 (16) | C2 (14) | C8 (8) | G2 (62) | G32 (32) | G48 (16) |
| --- | --- | --- | --- | --- | --- | --- |
| Ring1HopEW write 3072 / 4352 B (ring) | 0.0 / +0.1% | +0.4 / +0.3% | **+5.7 / +8.8%** | - | - | - |
| Ring1HopEW write 3072 / 3584 B (ring35) | 0.0 / 0.0% | +0.3 / +0.3% | **+4.8 / +7.1%** | -0.2 / -0.1% | -0.1 / -0.1% | 0.0 / -0.1% |
| Ring1HopEW scatter, every size | 0.0% | +0.2 to +0.3% | +0.2 to +0.3% | -0.1% | 0.0% | +0.1% |
| RingMulticast write (ring 3072 / 4352; ring35 3072 / 3584) | -0.6 / -0.3%; 0.0 / 0.0% | -0.1 / 0.0%; +0.2 / +0.2% | **+5.6 / +3.7%; +5.9 / +7.8%** | -0.1 / -0.1% | -0.1 / -0.1% | -0.1 / -0.1% |
| HalfRingMulticast write (ring; ring35) | +0.1 / -0.2%; +0.1 / -0.1% | +0.4 / 0.0%; +0.5 / +0.3% | +1.9 / +3.1%; +1.9 / +2.4% | +0.7 / +0.5% | +0.2 / 0.0% | +0.3 / +0.1% |
| UnidirRingMulticast write (ring; ring35) | 0.0 / 0.0%; +0.1 / -0.1% | -0.3 / -0.2%; -0.1 / -0.1% | -0.3 / -2.4%; 0.0 / -0.1% | +0.8 / +0.8% | +0.9 / +0.8% | +0.9 / +0.8% |
| every other scatter row | within 0.2% | within 0.3% | within 0.4% | within 0.5% | within 0.4% | within 0.4% |

**What the numbers show (both systems):**
- On the Galaxy jam rows, the arms with **R - T = 14-16** (C2, A16, G48) are at the fast level at R 16, 32 and 64.
  - R - T = 8 (C8) is fast but 3-7% slower at 3584 B and +7% on 8-ring 3072 B.
  - R - T = 30-62 (A2, G2) is at the slow level.
  - R - T = 32 (G32, the only R 64 arm at T/R = 0.5) is slow on the 4-ring and half ring and -6% on the 8-ring.
- On P150:
  - C2, A16 and the three R 64 arms cost at most 0.9% on any row.
  - C8 (R 16, T 8; R - T 8) costs +1.9 to +8.8% on the write rows, with ring 1-hop write the largest (+7.1 / +8.8%).
- The rule choice goes to the user.

## 10. Rule implemented (2026-11-04): T = max(2, R - 16) on the mainline port

**Decision (user, 2026-11-04):** on deadlock-avoiding routers an injection channel sends only with T = max(2, R - 16)
free slots in the far VC0 receiver of R slots; forwarding channels keep 1. 16-slot receivers (tori) keep T 2; 32-slot
rings get 16; 64-slot rings would get 48. The static_assert allows T <= R (user, 2026-11-04), so a 2-slot far receiver
still builds with T 2, as before.

**Commit `ac86bf0f116`** on branch `aagarwal/fabric-router-mainline-bfc`, off 4db4f45790b (not pushed). Worktree
`/home/aagarwal/tt-metal-mainline-bfc`.
- **Host** (`fabric_builder_config.hpp`, `erisc_datamover_builder.cpp`):
  - `builder_config::bubble_flow_control_injection_max_far_receiver_occupancy = 16`, the most packets the far VC0
    receiver may hold when an injection channel sends into it;
  - `bubble_flow_control_injection_min_free_slots(R)` = max(2, R - 16);
  - each router's builder takes R from its own far VC0 receiver (the slots of the deadlock-avoidance check already
    there), so a per-router R works later;
  - the new named arg `BUBBLE_FLOW_CONTROL_INJECTION_SENDER_CHANNEL_MIN_FREE_SLOTS` is passed only with deadlock
    avoidance. A `log_debug` line names T and R per router; Release builds compile it out.
- **Router** (`fabric_erisc_router_ct_args.hpp`, `router_sender.hpp`):
  - the constant reads the arg (0 without deadlock avoidance, where no channel uses it);
  - `SenderChannel<CH>::remote_free_slots_to_send` uses it for injection channels, still one immediate in the same
    `has_room` compare;
  - a static_assert in `send<CH>` keeps an injection channel's T in [2, far receiver slots].
- **Unchanged:** the mux kernels' own space check (2), the host's `bubble_flow_control_protected_receiver_min_slots`
  (2). In Tensix mux mode on a ring the router flags no injection channel (the mux injects), so the new rule does not
  reach it: those images are unchanged.
- The cleanup branch also touches `router_sender.hpp`, in other hunks (the cold `wait_for_connect_request`).
- **Open item: ring + Tensix mux (coordinator, 2026-11-04: record it, do not change it).** In mux mode the mux kernel
  gates injection with its own 2-slot check, and the router flags no injection channel. So the occupancy cap does not
  apply on mux rings, which keep today's rule. Whether mux rings jam the same way is not measured.

### 10.1 Offline gates (before any device run)

Two mock captures of 49 coverage configs (the mainline cov.py case set, plus the P150 X and Y tori and the Galaxy Y
torus), host and router both rebuilt for each: 4db4f45790b against the branch. Builds were paired by their inputs with
the new arg removed. Tools and outputs: `fabric_exp/consolidation/bfc/gate/` (`gate_full.txt`, `insn_summary.txt`,
`line_compare.txt`, one ring image pair).

| what | result |
| --- | --- |
| router images, all 49 configs | **5,345 identical, 170 differ**; plus 23 builds the multi-rank logs caught on one side only, paired through the JIT cache: identical |
| the 170 | all rings with 32-slot receivers, T 16: P150 ring E0 (36), P150 ring one ERISC E1 (36), Galaxy ring E0 (60), Galaxy 1x32 ring E0 (38) |
| each of the 170 | exactly one instruction, `srli 0x11 -> srli 0x14` at `router_sender.hpp:643` (`visit_worker`'s `has_room`): the compiler tests free >= 2 as `(word << 16) >> 17 != 0`, free >= 16 as `>> 20` |
| code size | +0 bytes on every differing image (and identical elsewhere) |
| line_compare (line numbers mapped across the moved lines) | `sender::run` 381 -> 381, `single::sender_pass` 322 -> 322, `single::receiver_pass` 532 -> 532 instructions; 0 changed lines in each |
| T 2 configs (identical images) | P150 XY, X and Y tori, one-ERISC torus, Y torus with VC2, Galaxy XY / Y tori, express (both ERISC modes), express E/W, per-mesh torus (VC1), trimming gtests |
| no deadlock avoidance (identical, no arg) | 1D lines, 2D meshes, neighbour exchange, mux 1D / 2D, VC2, VC1 and VC1 + VC2 splits, UDM, telemetry, code profiling, trimming, multi-mesh and pass-through |
| not comparable | streaming profiler (bh_2d_streaming, 2D mesh, no arg): the profiler numbers its translation units per run, so E0 / E1 swapped TU ids 10 / 11 between the captures; the only differences are 3 `lui` immediates per image in `kernel_profiler_streaming.hpp:202` |

- Mock: every ring and torus config reaches `Running Test` and `validate_results`. T per router, from each build's
  generated named args:
  - P150 ring T16 x72; P150 torus T2 x96;
  - Galaxy ring T16 x120; Galaxy 1x32 ring T16 x76;
  - Galaxy torus T2 x120; express T2 x216; per-mesh torus T2 x96;
  - ring + mux T16 x36 (passed, unused).
- **Base trees for the device runs:** `/home/aagarwal/tt-metal-new` and the validation worktree have moved to
  2fc06852f48, and the regress /data tree `new_4db4f45` was removed (about 08:00). So the base is my own build of
  4db4f45790b: worktree `/home/aagarwal/tt-metal-mainline-bfcbase`, built like the branch. Its host library lacks the
  new arg's name; the branch's has it.
  - Galaxy runs it from `/data/aagarwal/regress/trees/bfcbase`; P150 runs it from the worktree.
  - The /data tree, through a symlink farm, had matched the 4db4f45790b capture (204 / 204 images) before it was
    removed. It is not used.

### 10.2 Device runs (commands for the coordinator; nothing run yet)

- **P150 (a09u32 broker; workspace /home/aagarwal, timeout_sec 1500):**
  - `bash /home/aagarwal/tt-metal/fabric_exp/consolidation/bfc/bfc_p150_runs.sh val`: the branch, data-checked. Rows:
    mesh sanity, val1d_m1, val_ring_p150, val_torus_y_p150, val_torus_xy_p150 on the 2x4 and the 4x2.
  - `bash /home/aagarwal/tt-metal/fabric_exp/consolidation/bfc/bfc_p150_runs.sh perf`: base against branch, pass a
    (base, bfc), then pass b (bfc, base). Rows: p150_it_ring, torusxy_2x4_sizes, torusxy_4x2_sizes.
  - Base = `/home/aagarwal/tt-metal-mainline-bfcbase` (4db4f45790b). Branch = `/home/aagarwal/tt-metal-mainline-bfc`.
  - Queued: broker 758 (val), 2026-11-04 07:59 UTC.
  - Results in `fabric_exp/consolidation/bfc/results/`. Telemetry off; host-default opt level and slots.
- **Galaxy:** `fabric_exp/galaxy/frs_glx_bfc.sbatch`, after `bfc_stage_glx.sh` puts the branch on /data
  (`/data/aagarwal/regress/trees/bfc`).
  - Per-job symlink farms, no tree copies.
  - Kernel caches on the node's /tmp (LOCAL_CACHE=1, as dyn_glx.sbatch). At exit only the router builds (named args
    and ELFs) go to `$J/router_builds_<tree>/`. /data is short of inodes.
  - Steps: check on each tree; glx_frg_val on the branch, then the base; then frg_nsrx, frg_ew and frg_short4k,
    pass a (base, bfc) and pass b reversed.
  - Preflight and a stub dry run (farms, source tarballs, stop at the stubbed reset) passed. `sbatch --test-only`
    passed (bh_sc36_6).
  - Staged: `trees/bfc` (ac86bf0f116) and `trees/bfcbase` (4db4f45790b), about 48.6k files each, no python_env. The
    capture tool runs with `trees/main_d2bd12f/python_env`.
  - **Submitted: Slurm 172951**, 2026-11-04 09:00 UTC, under the one-time cap of 8 (6 study Galaxy allocations at
    submission); /data had 5.5M free inodes; standing excludes.
    - Cancelled while pending (coordinator) and resubmitted with bh_sc36_2 allowed (the user's one-time 120-c
      exception, `GLX_SC36_2_EXCEPTION=1`).
    - 172981 landed on bh-glx-120-c08u20 and stopped before the reset: another user's process was there (a
      tt-device-mcp server).
    - **172983** runs on bh-glx-120-c08u08 (from 09:13 UTC), with bh-glx-120-c08u20 added to the excludes.

### 10.3 Results

**Galaxy 172983** (bh-glx-120-c08u08, partition bh_sc36_2, 09:13-09:23 UTC; `bfc/results/glx172983`;
`bfc/bfc_report.py glx`).
- **Run:** all 16 steps exited 0; no hang, no timeout.
  - The 2D check: 1 / 1 on each tree.
  - glx_frg_val (Ring and XY-torus full-ring multicast and all-to-all, unicast write and fused atomic inc, 4096 B):
    10 / 10 on the branch and 10 / 10 on the base.
- **T per router** (the branch's 408 saved router builds):
  - 120 at T 16 (the ring);
  - 120 at T 2 (the validation's XY torus, 16-slot receivers);
  - the rest without the arg (the 2D check, no deadlock avoidance).
- **Node class:** a 120-c Galaxy, not the 110-a nodes of §9. The two trees ran in one job on one node; numbers from
  other nodes are not comparable to these.

Cycles per packet, pass a / pass b (one number where the passes agree to 0.1); `*` where a tree's two passes differ by
more than 1%:

| row | base 4db4f45790b | branch ac86bf0f116 | branch vs base |
| --- | --- | --- | ---: |
| 8-ring 3072 B (control) | 195.6 | 194.7 / 194.8 | -0.5% |
| 8-ring 3584 B | 310.1 / 312.6 | 228.5 / 228.4 | -26.6% |
| 8-ring 4352 B | 375.1 / 372.2 | 269.6 / 269.3 | -27.9% |
| 4-ring 4352 B | 323.7 / 339.9* | 254.9 / 255.0 | -23.2% |
| half ring 4352 B | 331.5 / 332.1 | 253.6 | -23.6% |
| ring 1-hop 4352 B | 249.5 | 249.5 | 0.0% |

**What the numbers show:**
- The branch is out of the slow level on every jam row, in both passes:
  - 4-ring and half ring at 253.6-255.0, as the R - T 14-16 arms of §9.5 (255.1, 253.7);
  - 8-ring 3584 B at 228.5 and 4352 B at 269.5. The §9.5 arms ran at 209 / 254 on 110-a nodes. This node's base is
    also not §9's (3072 B control 195.6 against 191.9 there), so the two are not compared.
    - Later, Galaxy 176994 (2026-11-05, one 110-a node, one job; SLOT_ALLOCATOR.md §12.15): the port with main's
      tier rows (16,16,32, T 16) ran 228.0 / 269.4, within 0.3% of this branch, and with the 1D leftover order's
      26,25,32 (T 16) ran 209.0 / 254.1, within 0.5% of §9.5's A16. On that node the slot rows alone make a gap of this
      size. No jam-level row in either arm.
- The base is at the slow level on every jam row (310-375; 4-ring 324 / 340, passes 5% apart).
- The 3072 B control and ring 1-hop do not move.

**P150, broker 758 (validation) and 759 (perf)** (bh-lb-120-a09u32, 09:52-10:05 UTC; `bfc/results/`;
`bfc/bfc_report.py p150`, table in `bfc/results/p150_report.txt`).
- **Validation (branch, data-checked), 758:** exit 0; no hang. Every row passed:
  - mesh sanity 96 / 96, val1d_m1 143 / 143, val_ring_p150 19 / 19;
  - Y torus 12 / 12, XY torus 2x4 12 / 12 and 4x2 12 / 12.
- **Perf, 759:** exit 0, all 12 runs passed (ring 16 tests, each torus 80); post-job health gate OK. The order was pass
  a base then branch, pass b branch then base.
- **Ring rows** (p150_it_ring, the 32-slot ring receivers; the branch at T 16): every row within -0.6% to +0.1% of the
  base. No pass pair differs by more than 1%.
  - RingMulticast write 3072 B: 142.3 -> 141.5 (-0.6%);
  - RingMulticast write 4352 B: 151.4 -> 151.0 (-0.3%);
  - HalfRingMulticast write / scatter: 0.0 to -0.2%;
  - UnidirRingMulticast: -0.2 to 0.0%;
  - Ring1HopEW: 0.0 to +0.1%.
- **XY torus rows** (the images are byte-identical, T 2):
  - 2x4: every row within +-0.2%;
  - 4x2: every row within +-0.1%, except TorusXY4x2_Multicast, which moved -1.2% to +0.8% with passes more than 1%
    apart in both trees (its known pass-to-pass noise, §9.2).

**What the device runs show:**
- The rule fixes the jam rows on Galaxy: -23% to -28% on the 8-ring 3584 / 4352 B, 4-ring and half-ring rows. The
  3072 B control and ring 1-hop are unchanged.
- The data-checked validation passes on both systems: Galaxy full-ring flow control 10 / 10; P150 mesh, 1D, ring, Y and
  XY tori.
- P150 cost: none beyond 0.6% on any ring row; the torus rows are unchanged (identical images).
- Open: ring + Tensix mux keeps the 2-slot mux check (above); the cap does not reach it.
