# Completion credits: the problem, stepped back

Status: analysis for discussion (2026-10-28). Data: RESULTS §98 (jobs 093-105). Nothing here is decided.

## 1. What a completion credit is for

Every channel is a closed credit loop. A producer (the worker, or the router that forwards into a forwarding
channel) may have at most W packets it has not had credited back, W being the channel's slots:

```
producer ──▶ sender channel (W slots, E0) ──link──▶ far ring (R slots, E1) ──NoC writes──▶ next hop / local core
   ▲                                                    │
   └── credit write (E0) ◀── E0 reads count ◀──link── frame (E1) ◀── proof ◀── due (batching policy)
```

The far ring is a second loop inside the first: E0 may have at most R packets un-completed on the far ring.

A loop with window W and round trip RT carries at most W / RT packets per cycle (Little's law). So for each row
there are two regimes:
- **service-bound:** W / RT is above what E0, E1 or the link can do. Credit timing doesn't matter; per-packet cost
  does, including the cost of each credit.
- **window-bound:** W / RT is the limit. Every cycle added to RT costs throughput; per-credit cost matters less.

E1 cannot tell which regime a producer is in. E0 can, for its own channels.

## 2. What the round trip is made of

For the oldest slot of a group, from the producer writing it to the producer seeing it credited:

Measured (RESULTS §98, job 108; 1-hop, v4gi; the timing hooks add +2 to +8 cycles/packet, so a little high):

| term | kind | cycles |
| --- | --- | ---: |
| queueing in the worker channel ahead of the send | load | 2-3 packets, ~300-400 |
| link out + link back + E0's look at the count | mechanism | 820-950 (16 B), 970-1,090 (4352 B) |
| E1: at the head -> retired (decode, writes issued) | mechanism | 85-91 |
| **E1: retired -> credited (batching wait and proof)** | policy | 325-370 |
| E1: credited -> frame sent | mechanism | 50-57 |
| E0's credit write to the worker | mechanism | 27 |

E0's send -> completion seen is 1,290-1,580; the worker's whole loop ~1,800, against 17 slots x ~120 = ~2,040. The
link is ~2/3 of the round trip and no router change touches it; batching is the largest router term (~25%).
The mean round trip is shorter in v4gi than in v4vc on every row, yet v4gi is slower on some burst rows: that
difference is about which credits are late while the worker is blocked, not about the mean.

## 3. The windows we actually have

| config | worker W | forwarding W | far ring R | T |
| --- | ---: | ---: | ---: | ---: |
| 2D, VC0 only | 17 | 16 (x3) | 16 | 4 |
| 1D (Linear, Ring) | 26 | 25 | 32 | 4 (rings: 1) |
| VC1 builds, VC0 | 51 | 4 (x3) | 8 | 2 |
| VC1 builds, VC1 | (none) | 2 (x3) | 4 | 1 |

Where they come from (`fabric_static_sized_channels_allocator.cpp`): the first row of a per-arch table that fits,
one depth for every channel of a VC, then every spare slot to the worker channel. For VC1 on Blackhole the first row
is {4, 8, 2, 4}: 34 of ~81 slots, and the worker gets the other 47 (51 in all). So in VC1 builds the windows that
carry forwarded traffic, and the far rings, are tiny, and the one window nobody shares is the largest. This is a
table choice, not a capacity limit.

## 4. What the measurements say

1. **The window binds 1-hop worker rows in 2D, for both designs.** Worker 17 -> 32 slots (jobs 104-105): steady
   16 B 120.2 -> 118.1; bursts N8 G500 16 B ~131 -> 123, N12 G500 16 B 131 -> 118, N32 G500 4352 B 135 -> 126.
2. **The design difference only exists while the window binds,** and is small next to (1). v4gi - v4vc at 17 slots:
   N8 +3.2 / +3.5, N12 +2.0 / +2.3, N6 +0.7, N32 -2.6 to +4.7, steady -0.3 to -1.0. At 32 slots: within 0.1 except
   two rows (+0.5, +1.9).
3. **v4vc's edge was partly the test suite's.** exp_burst.yaml's bursts (1, 2, 4, 8, 32) are all multiples of
   T = 4, so v4vc's batches, which restart at a gap, end exactly at the next one. At N = 6 v4vc leaves partial batches
   too (8-10% of its credits are 2 slots) and the difference is +0.7.
4. **Per-credit cost is real, so batching is worth something.** E0 pays 27 cycles per worker credit. No batching at
   all (E1-38): means +1.9 to +7.2, steady small writes ~133. The first batching (bit 22): E0-bound rows -12 to -19.
   Releasing partial groups on the first idle pass (v4gidg1): steady 16 B +1.7 to +4.4 (0.33-0.37 credits/packet).
5. **Batching also costs window.** Every group's oldest slot waits ~T packet times. On a 17-slot window, T = 4 is
   about a quarter of it. On VC1's 8-slot ring, T = 2 is a quarter. On 4-slot rings, holding any credit cost +30 to
   +45 (E1-27).

## 5. The problem, restated

- **Primary:** some windows are too small for their round trip: the 2D worker channel at 1-hop near link rate, and
  every forwarding channel and far ring in VC1 builds. While a window binds, any term of RT costs throughput, whatever
  the completion design.
- **Secondary:** the batching policy (T and the gap hold) is part of RT, and how much it adds depends on the traffic
  pattern (gaps vs group boundaries), which is where the v4gi / v4vc differences come from.
- **The tension:** batching saves per-credit work (E0 above all) and spends window. Its right size depends on the
  regime, and the side that decides (E1) can't see the regime.

## 6. Levers

| lever | what it changes | for | against / unknowns |
| --- | --- | --- | --- |
| A. allocation | W and R per channel within the fixed ~81 slots | VC1 only: its table leaves 47 slots on the worker while forwarding gets 4 / 2 and the rings 8 / 4, so a rebalance moves window to where it binds | zero-sum: 2D VC0-only uses all 81 slots (17 + 3 x 16 + 16), so a wider worker means narrower forwarding (32 -> 11, very likely slower multi-hop: 16 already binds 3-hop in 1D). Not a 2D lever. Growing windows without L1 would need slots sized to the packet (a 16 B packet takes a 4,448 B slot), a major change |
| B. T relative to the window | how much of the window batching takes | T is set from the ring today, but the binding window may be the producer's (worker 17); decision 4 (T = 1 below 16) is this lever | more credits per packet: E0's 27 cycles each on service-bound rows |
| C. the gap hold | the 4-idle-pass wait for a partial group | v4gidg1 halved N8 x 16 B's gap | v4gidg1 costs steady rows +2 to +4; not a fix alone |
| D. where batching is decided | E1 credits sooner; E0 decides when to write the producer's credit | E0 knows the producer's window state | more frames (E1, reverse link); moves policy into E0's loop, which is the bottleneck on many rows |
| E. cheaper credits | the 27 cycles per worker credit | less reason to batch | the worker credit protocol (scratch word + 4-byte NoC write on Blackhole) |
| F. group alignment (v4gi) | fixed-position vs restart-at-gap groups | the v4gi-only part of the difference | pattern-dependent (N8: +3.5, N6: +0.7); E1 state for restart-at-gap |

## 7. Suggested order

With L1 fixed (user, 2026-10-28), the window is not the lever in 2D: the round trip is. Window-bound throughput is
W / RT, so every cycle RT loses is the gain.

1. **Measure RT's breakdown** (section 2) first, so the terms aren't guesses: timestamps of sampled packets at E0
   send, E0 seeing the completion, E0's credit write, and on E1 head, retire, credit and frame (intervals within one
   chip only).
2. **Then pick the terms worth attacking** (B, C, D, E, F act on different ones).
3. **VC1: redistribute** (A). Done as a sweep (RESULTS §98, jobs 109-110; `TT_FABRIC_EXP_SLOTS_VC1`, inter-mesh
   benchmark rows now possible): VC0 receiver 8 -> 16 takes 26-43 off every 1-hop row; VC0 forwarding 4 -> 8 -> 12
   takes the 2-hop row 482 -> 259 -> 215; VC1 forwarding 2 -> 4 -> 6 takes inter-mesh flows with VC1 hops 780-970 ->
   400-500 -> 310-410; the worker needs ~19. The rule (how to split the rest between VC0 and VC1 forwarding) is open.
   Lever B measured on the way (v4git: T from the smallest sender channel): helps only while forwarding channels are
   under ~3 groups, costs ~9.5 on 1-hop rows and 16-22 on forwarded rows otherwise; so size the channels, keep T.
4. **Re-measure the v4gi / v4vc difference** after (2) and (3).

## 8. Open questions

- What RT does a forwarded packet see, and how much of the forwarding window does T take at each hop?
- Does the 2D worker need 17 at all on multi-hop routers, or should the split depend on the router's position?
- Is per-credit cost on E0 mostly the credit write (27) or the visit around it? (PC profile: worker visit +2.6 /
  credit +1.7 per packet on the N8 row.)
