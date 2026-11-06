# Needs-first slot allocator: proposal (2026-11-04, for discussion; nothing built)

Builds on SLOT_ALLOCATOR.md (step 1, the ratio allocator) and its §12 slot-need sweeps. Labels: measured (job or
section), hypothesis.

## 1. The idea

Each channel kind has a **need in slots**: the depth past which more slots stop helping (the knee, §12). The allocator:
1. gives every channel its need;
2. when slots are left over, gives them to the one kind not yet shown to saturate (VC0 forwarding);
3. when the budget is short, gives every channel the same fraction of its need.

It has no VC ratios, kind weights, receiver weight or floors, and it takes no perf numbers and no link speed. Bytes appear
once: N = floor(channel buffer bytes / slot size).

## 2. Why this shape (the data)

- **The need is a slot count, not a byte share.** The 2D worker's knee is 17-18 slots for small packets at both 3072 B
  and 4352 B slots, with the same cycles at 16 under both (D2, lb2 172838, §12.2). Measured for the 2D worker only;
  receiver and forwarding have not been checked this way.
- **The need does not depend on the system for router-bound traffic.** The P150 and the Galaxy give the same knees for
  256-2048 B packets (G1, §12.4), so link speed is not an input. Wire-bound packets need fewer slots (Galaxy 4352 B
  packets: worker 9-12, receiver 5-8). A table of router-bound needs covers the worst case; large packets simply do not
  use all of it.
- **Below the knee, rate is proportional to depth.** Cycles/packet x depth is nearly constant per kind (§12.2-12.5).
  Hypothesis (credit loop, not isolated): a channel at a fraction f of its need runs its rows at about f of full rate.
  This is the basis for the equal-fraction squeeze in rule R4.
- **Mesh size:** on the Galaxy, the 4x4 VC1 split's cycles x depth products are close to the 2x2's (G2, §12.5), so this
  data shows no mesh-size input.

## 3. The needs (named constants, one per VC x kind x 1D/2D)

| channel | need | evidence |
| --- | ---: | --- |
| VC0 worker, 2D | 20 | knee 17-20 (P150 and Galaxy), D2 17-20 |
| VC0 worker, 1D | 24 | knee 21-24 (P150) |
| VC0 receiver, 2D | 16 | knee 9-16 (P150 and Galaxy) |
| VC0 receiver, 1D | 32 | 16 is 3% slower than 32 at small packets (P150) |
| VC0 forwarding, 2D | 16 | **provisional**: still improving at the sweep's top (15 beats 12 by 13% on 3-hop) |
| VC0 forwarding, 1D | 24 | **provisional**: 24 beats 16 by 6%; nothing measured past 24 |
| VC1 forwarding | 10 | P150 C2: 8 -> 10 is 3%, 10 -> 12 is 1.2%. Galaxy C2 (Slurm 172999, 2x2 and 4x4 splits): 10 -> 12 gains 1.1-1.6%, other inter rows settle at 9-10 (§12.6) |
| VC1 receiver | 8 | knee 5-8 (P150 C2; Galaxy C2 on both splits) |
| VC2 worker / receiver | 16 / 16 | **unmeasured**: the 3 KiB measured pick (§3) |

Each constant gets a one-line comment stating its reason, in the same style as
`bubble_flow_control_injection_max_far_receiver_occupancy`.

## 4. Rules

- **R1.** N = floor(channel buffer bytes / slot size).
- **R2.** Each channel's need comes from section 3 (by VC, kind, and whether the router is 1D or 2D).
- **R3. Total need <= N:** every channel gets its need. The leftover goes to the VC0 forwarding channels, one slot each
  in turn, up to 127 per channel. Forwarding is the only kind not yet shown to saturate.
- **R4. Total need > N:** every channel gets the same fraction N / total need of its need, rounded by largest
  remainder, with a minimum of 2.
- **R5. Ring/torus VC0 receiver:** a power of 2, the one nearest its share; the senders absorb the difference. A
  non-power-of-2 receiver costs about 8 cycles/packet on rings because of the wrap compare (§9, lb2 171579). Mesh and
  line receivers take any count; the cost there is 0-1.6 cycles/packet (§9).
- **R6.** Every count is logged with its reason (need, or the fraction applied). The explicit study switches still win.

Dropped from step 1: VC ratios, kind weights, the receiver weight, power-of-2 rounding on meshes, the 1/5 receiver
floor, `worker_cap_slots` (the need is the cap), and the long-dim forwarding bump.
Kept: slots not bytes, the minimum of 2, logging, explicit overrides, and step 2 (per router) unchanged in shape.

## 5. What it gives (sketch arithmetic: scratchpad needs/needs_first.py)

| config | 4352 B (N 81 / 83 for 1D) | today at 4352 B | 3072 B (N 115) | measured best at 3072 B (§3) |
| --- | --- | --- | --- | --- |
| 2D | 19,16/16/15,15 | 17,16,16 | 20,27/26/26,16 | 20,21,32 |
| 2D ring / torus | 19,16/15/15,16 | 17,16,16 | 20,27/26/26,16 | - |
| 1D | 24,27,32 | 26,25,32 | 24,61,32 | - |
| VC1 split | 13,11/11/10,10 + 7,5 | 19,9,16,5,4 | 19,15,15 + 10/9/9,8 | 20,15,16,9,7 |
| VC2 on | 14,12/11/11,11 + 11,11 | 17,14,16 + 2,4 | 19,16,16 + 16,16 | 20,14,16 + 16,16 |
| VC1 + VC2 | 11,9,8 + 5,4 + 8,8 | 19,8,16,4,4 + 2,4 | 15,12,12 + 8/7/7,6 + 12,12 | - |

Rows read: VC0 worker, forwarding (per channel when they differ), receiver; then VC1 forwarding, receiver; then VC2
worker, receiver.

At 15,232 B (N 23), 2D gives 6,5/4/4,4. That's close to main's tier 7,4,4, the balanced choice in the F squeeze data
(§12.1).

## 6. Where it may be worse (to settle by A/B, link total for mixed rows)

- **VC1 split at 4352 B.** Total need is 122 against N 81, so every channel gets 66%. That leaves VC0's receiver at 10
  and its worker at 13. Today's measured row 19,9,16,5,4 protects VC0's worker and receiver and squeezes forwarding.
  The equal fraction rests on the rate-proportional hypothesis. Today's row may win VC0 1-hop rows and lose VC1 rows.
  At 3072 B the proposal lands within a slot of the measured best.
- **VC2 at 4352 B.** VC2 gets 11 / 11 against today's 2 / 4, and VC0 drops to 14,12,11. Whether that is right depends
  on what VC2 carries, and its need is unmeasured.
- **Leftover to forwarding** (2D 3072 B: 27/26/26) may buy nothing past forwarding's real knee, which is not yet known.
- **Need independent of slot size:** shown for the 2D worker only.

## 7. Decisions for the user

1. **Squeeze rule (R4):** equal fraction of need (proposed), receivers first, or VC0 first. At 4352 B on the VC1 split
   they give 13,11/11/10,10 + 7,5 (equal), 12,9,16 + 6,8 (receivers first), and equal again for VC0 first, because
   VC0's need alone exceeds N minus VC1's minimums.
2. **VC2's need while unmeasured:** 16 / 16 (proposed), or VC2's minimum unless a build asks for more.
3. **Leftover (R3):** to VC0 forwarding (proposed), or spread over all channels by need.
4. **Where the needs live:** named constants in the builder config (proposed), one per VC x kind x 1D/2D.

## 8. Data still needed (small)

- **VC0 forwarding knee:** P150 at 3072 B slots, 2D forwarding up to about 26 (worker 20, receiver 16), and 1D
  forwarding past 24. One LB job of about 20 min.
- ~~Galaxy VC1 knee~~: done (Galaxy C2, §12.6). Same knee in slots as the P150 on both splits; cycles x depth runs
  10-20% longer on the Galaxy.
- **Slot-size independence for receiver and forwarding:** a D2-style pair at 3072 B and 4352 B.
- **Then the A/B:** needs-first rows against today's, on the standard P150 and Galaxy sets, mixed rows scored by link
  total.

## 9. Interplay

- **Ring-jam rule T = max(2, R - 16):** with a ring receiver of 16, T is 2 and the cap never binds. A 1D ring receiver
  of 32 gives T = 16, as gated today.
- **TRIDs:** receivers settle at 16 (2D VC0), 32 (1D VC0) and 8 (VC1). That narrows the completion-group cases for the
  pending TRID decision.
