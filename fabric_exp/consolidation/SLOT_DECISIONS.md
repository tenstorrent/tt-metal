# Slot study: findings and open decisions (2026-11-02, parked for the user)

Details, tables and job numbers: SLOT_STUDY.md (sections P1-P4, "2 KiB slots on a 200G Galaxy", "3-3.5 KiB slots on
the 200G Galaxy"); the N/S floor: V4_NS_FLOOR.md. Router: the lead v4rxv2i3 (-O3). No builder or allocator change was
made by the study.

## Findings (measured)

1. **P150 (400G), 4352 B slots, VC1 split.**
   - The VC0:VC1 weight is effectively binary: either VC1 is filled to its path bound, or VC0 comes first.
   - Today's row 19,9,16,5,4 is the model's 3:2 pick (VC0 2-hop 258, VC1 one-hop 396).
   - 18,7,16,7,5 gives VC1 -22% (308) for VC0 2-hop +25% (323).
   - Today's VC1 rows are bound by the VC1 forwarding window (5), not the ring.
   - Lead fit (190 points): mean error 1.1%.
2. **Galaxy (200G) at 4352 B.**
   - VC0 rows sit on the wire (252) once the worker is ~10, ring 8 and forwarding 11.
   - The VC1 path is router-bound (~296).
   - At 4352 B, 81 slots can't give VC0 2-hop at the wire and deep VC1 windows together.
3. **Mesh size (Galaxy 8x4 vs P150 2x4).**
   - N/S forwarding loops grow with mesh rows (~2,550 at 3 hops, ~2,950 at 7, vs ~2,100 E/W).
   - Long N/S rows are forwarding-window-bound below 16 slots.
   - The 2D rule picks 20,15,16 (vs today's 17,16,16); that trade is ~3x sharper on the Galaxy.
4. **2 KiB slots on the Galaxy (169 slots).**
   - The windows go away: VC1 split 32,18,32,12,10 puts both VCs at their bounds; VC2 16/16 runs at 144 alone.
   - 1-hop reaches the wire at 2048 B (framing -4%).
   - Forwarding paths become router-bound: 3-hop -15%, N/S -31%, inter-mesh with VC1 hops -49%.
   - "~2.5K" in RESULTS §98 was 1D saturation (2D ~2.3 KiB).
5. **3072 B slots on the Galaxy (115 slots): the best single size measured for 200G.**
   - Every VC0 path reaches the wire (-0.9% vs 4352 B), because a 179-cycle wire time covers every VC0 router cost
     (110-173).
   - VC0 and VC1 windows both fit at their bounds, so no weight trade is needed. VC2 16/16 reaches the wire, and VC0 +
     VC2 split the link ~50/50.
   - The cost: paths whose own per-packet cost exceeds 179 cycles. Inter-mesh direct -15% (~208), VC0 hop +
     inter-mesh -18% (~240), inter-mesh + VC1 hops -26% (~285-303).
   - 3584 B (99 slots) brings inter-mesh direct to the wire but forces a worker-or-forwarding trade.
   - Measured allocations at 3072 B: 2D 20,21,32; VC1 24,15,16,8,6; VC2 16/16.
6. **The builder's slot defaults at non-4352 sizes are poor.** It keeps the 4352 B windows and gives every extra slot
   to the worker. Any other slot size needs the measured allocations, not the builder's default.
7. **Models.** sls_model.py / sls3k_model.py predict within ~1-1.5% mean (max ~6%) when predictions are written before
   the runs.

## Open decisions (recommendations in brackets)

1. **200G (Galaxy) default slot size.** [3072 B with the measured allocations as the host's default rows; keep 4352 B
   as an option for large-VC1-transfer-dominated deployments.]
2. **VC1 policy on 400G (P150, 4352 B).** [Keep today's VC0-first row as the default; add a host setting for the
   VC1-first row.] On 200G at 3 KiB the question disappears.
3. **The allocator design note** (mesh-size and packet-size aware). [Start once wanted; inputs are measured for link
   speed × packet size and for mesh rows. The exit-chip loop waits on P4 (meshes larger than 2x2), which needs the
   command-buffer-2 fix from the receiver pass.]
4. **P4** (VC1 on 4x4 / 4x2 Galaxy splits) is prepared and not run. It waits for the buffer-2 fix.
5. **Other open items from the study:**
   - a burst-aware worker step (bursts of 8 need 19 worker slots at 4352 B on P150);
   - the model's max() underestimating paths where two windows are close;
   - per-link loop differences on the Galaxy.

## Tooling to fix regardless

alloc_model.py's row map and fabric_exp/vc1/vc1_mix.yaml assume the pre-merge placement ([0,0] now lands on [1,0]).
sls_vc1_all.yaml has the corrected mix rows.
