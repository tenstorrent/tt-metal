# Idea, parked: VC2 as a forwarding "line" VC (1D routing on a 2D fabric)

Status: idea captured for later (2026-11-01, user). Not started; nothing built. Come back to it (or hand it to an agent)
after the current receiver/VC2-cost experiments. The detailed exploration below is a design agent's (offline, read-only)
report, condensed; its numbers are estimates unless marked measured.

## The idea (user)

Today VC2 is a neighbour-exchange-only VC on 2D fabrics: one worker channel per router (E0), one receiver channel (E1)
that only delivers to its own chip (`DISABLE_RX_CH2_FORWARDING`), counter credits, 2/4 slots (V4_VC2.md).

Make VC2 a forwarding VC that routes like 1D (lines along a mesh axis, rings on torus axes), next to VC0's 2D routing:
- both VC0 and VC2 take a worker, so different kinds of traffic route through the fabric at once (e.g. ring/line CCLs on
  one axis while 2D traffic that turns runs on VC0);
- VC2 keeps the 2D packet header (same slot size, same header code); a VC2 route only goes straight along one axis, so a
  VC2 receiver forwards only to the router on the opposite side of the chip (one downstream, not three);
- E0 needs a VC2 forwarding sender channel;
- inter-mesh: both VC0 and VC2 fan into VC1, so inter-mesh routing stays 2D; VC0 and VC2 may block each other at VC1
  (accepted trade-off, as long as there is no deadlock);
- motivation: 200G Galaxy use cases saturate at smaller packets, which give more slots than the ~81 per ERISC at 4352 B.

## Decisions and constraints from the user (2026-11-01)

1. VC2's mode is a host config flag: neighbour exchange only (today's behavior and code) or same-dimension forwarding;
   both as compile-time features of the VC2 channels, so routers in neighbour mode keep today's images.
2. Growing the ERISC kernel config buffer (`MEM_ERISC_KERNEL_CONFIG_SIZE`, 26 KiB on Blackhole) is acceptable if the
   feature is useful. Do not chase code-size cuts for this.
3. Command-buffer / NoC-state strategy for several forwarding paths in the same dimension (VC0 and VC2 forwards to the
   same opposite router) can be derived on the host or by static logic in the router; not a concern.
4. The user disputes the code-size estimates below (e.g. VC2 rings bring their own copy of the first-level-ack code);
   treat them as unverified until a build measures them.

## What the exploration found (design agent, 2026-11-01; labels: code = read from code, meas = measured, est = estimate)

**Feasibility.** Mostly composed from what v4 has: per-VC far rings, the forwarding-group visit, per-channel features
(`Channel<CH>`: routes, acks, lands...), `Downstreams<VC>`. VC2 line mode = VC0's 1D-ring machinery (bubble flow
control + first-level acks) on a third VC whose downstream set is the straight direction.

**Semantics.**
- A unicast to a destination on the same row/column is already a straight 2D route (code:
  fabric_2d_route_interface.h:151-191; router_routing.hpp decode, Y first). Same header, same decode.
- Receiver: keep `Routing2D`; `forward_to = action & live` with `live = 1 << opposite(my_direction)` gives one
  downstream and one admission test. A turn bit would be masked (packet retired without forward): add a watcher ASSERT.
- Multicast: straight branches only; proposal: VC2 line setters with an explicit direction
  (`fabric_set_line_unicast_route(hdr, dir, hops)`, `fabric_set_line_mcast_route(hdr, dir, start, range)`).
- The 2D header carries ~46 B more per packet than 1D on the wire (meas, RESULTS ~6700).

**Deadlock argument (est from code).**
- VC0 unchanged. VC2 lines on a mesh: one direction, ends in delivery: acyclic. VC2 rings on a torus axis: bubble flow
  control (worker = injection, needs 2 free far slots; forwarding needs 1) + first-level acks on VC2, as VC0's rings;
  VC2 receiver ring >= 2 slots (main's tier rows give VC2 receiver depth 1 in some rows).
- VC dependency graph VC0 -> VC1 <- VC2 is acyclic: VC1 receivers forward only into VC1; landing only rewrites the route;
  nothing drains from VC1 into VC0 or VC2 (router_receiver.hpp:99-103, 294-304; router_routing.hpp:49-74).
  Pre-existing caveats: mesh pass-through "NOT guaranteed deadlock-free" (fabric_builder_context.cpp ~383); VC1 has no
  bubble flow control on torus-shaped meshes.
- No shared non-buffer resource couples VCs in v4: E1 admission is non-blocking before any side effect; E0 sends only
  with per-VC far credit; the only spins wait on the NoC/Ethernet (TXQ, command buffer ready), not on fabric credits.
  So VC0 and VC2 only delay each other at VC1 (head-of-line), no cycle.

**Device changes (v4).**
- Mode flag drives existing args: `ACTUAL_VC2_SENDER_CHANNELS` 1 -> 2 (worker + forwarding), `DISABLE_RX_CH2_FORWARDING`
  1 -> 0, new `ENABLE_FIRST_LEVEL_ACK_VC2` (rings), VC2 downstream args; `delivers_only[2]` and `first_level_ack_on[2]`
  (router_receiver.hpp:72-73) follow them; new members `Present`/`Absent` so neighbour-mode images stay identical.
- E0: `vc2_forwarding` mask into `all_forwarding`; one more forwarding-group visit in `pass()` (idle cost: one register
  read + test, est +3-6 cycles/packet; A/B it, or fold into the top group).
- E1: `DownstreamTraits<2>` (router_downstream.hpp:35-51 handles VC0/VC1 only); hold `Downstreams<2>` in VC2's
  in-memory `ChannelState` (E1's loop is at its register limit); VC2 runtime downstream block; teardown already waits
  on VC2's trids.
- One-ERISC routers compose the same passes (router_single.hpp).

**Host / API changes (est).** VC2 turn set = {opposite}; producer slots; adapter and named args
(`NUM_DOWNSTREAM_SENDERS_VC2`, `VC2_DOWNSTREAM_EDM_SIZE`, `OPEN_DOWNSTREAM_VC2_CONNECTIONS`,
`DOWNSTREAM_SENDER_NUM_BUFFERS_VC2`); +2 stream registers (about 20-22 of 30); bubble flow control on VC2
(`bubble_flow_control_enabled_on_vc(2)`, injection flags, receiver min-slot check); allocation rows with a VC2
forwarding depth (and a packet-size-aware rule); VC2 worker API: validate same-axis destinations, line setters;
trimming capture tables for VC2; VC1 fan-in (`setup_downstream_vc_connection(upstream=2, downstream=1)`, ingress flag
for receiver 2, VC2 exits on edge routers); test infra already writes full 2D routes for `vc_id: 2`, needs multi-hop
line / ring / mixed VC0+VC2 yamls. Express / Z-boundary fabrics: the flat sender table would overflow (5+4+2 = 11 > 10):
exclude at first.

**Code size (est, disputed by the user, see decision 4).** Measured today (-O3, E0+E1): 2D+VC2 19,436 B; VC1+VC2
26,076 B (job 323: it loads; 548 B under 26,624 before runtime args). Agent's estimates: 2D+VC2 line ~21.5 KB, torus +
VC2 line with acks ~22.5 KB, VC0+VC1+VC2 line ~28.1-28.4 KB (needs ~29 KiB). Growing the buffer: one constant
(dev_mem_map.h:233) + HAL (UNRESERVED base) + fabric channel space; at 4352 B, 2D keeps 81 slots with +3 KiB, 1D 83 -> 82.

**Slots (est, window model, ALLOCATION_NOTE.md / alloc_model.py fits from v4us: re-fit first).** At 4352 B on 400G
P150 two full-rate VCs do not fit 81 slots (trade by weight, as VC1). On 200G Galaxy (wire ~248 cycles/packet at 4352 B)
or with ~2-2.3 KB packets (151-169 slots) they fit. Needs the packet-size-aware allocation rule (deferred, §25 of
V4_RECEIVER_SHAPE.md).

**Use cases (repo grep).** No op uses VC2 today (only test_fabric_vc2_at_least_2x2_mesh.yaml, one hop). VC2's
introducing commit (f3520c84d37): "additional/parallel data planes ... future work could enable VC2 with other
topologies". Value: isolation (ring/line CCLs on one axis concurrent with 2D traffic that turns: MoE dispatch/combine,
deepseek dispatch) and new multi-hop line ops (line multicast, barriers) instead of per-hop worker relays; not more link
bandwidth at wire-bound sizes.

**Staged plan.**
0. Cut idle VC2's E0 cost (prerequisite for small-packet saturation): in progress (V4_RECEIVER_SHAPE.md §29-§30).
1. (a) Line VC2 on meshes: offline coverage (neighbour-mode images identical, new line configs build, size), CPU mock
   capture for host wiring; device: data-checked line unicast 1-3 hops, line multicast, mixed VC0 2D + VC2 line, both
   ERISC modes, hang capture; perf: VC0 rows with VC2 off/neighbour/line (lean-ERISC probes per side), VC2 line rows.
2. (b) VC2 rings on tori: Y-torus on P150x8 (full-ring multicast, wrap unicast, bidirectional) with VC0 torus traffic;
   stress at small VC2 rings; perturbation sweeps; then Galaxy torus via Slurm (dry run first).
3. (c) VC2 -> VC1 fan-in: size first; then the VC1 split with a VC2 line crossing the boundary, data-checked, capture
   armed.

**Open questions (for the user, when this resumes).** Line mode on both axes or one chosen axis; VC2 inter-mesh
semantics (exits only on the line, or drop stage c); buffer target (~29 KiB); express + VC2 line in scope; target packet
size for 200G (~2.3 KB?) and whether the packet-size slot rule comes with this work; dedicated line setters with an
explicit ring direction.

## Related problems found along the way (code reads, not device-checked)

1. VC0 and VC1 forwards share forward data command buffer 2 (erisc_datamover_builder.cpp:519); v4's VC0 forwards are
   stateful on E/W routers inside a mesh (router_downstream.hpp:79-88, not excluded on VC1 fabrics, unlike main), so on
   multi-mesh fabrics with meshes larger than 2x2 a VC1 stateless forward would overwrite VC0's preset destination. The
   2x2 VC1 split harness has no such router. Fix independently of this idea.
2. Transaction IDs: with VC1 or VC2 and deadlock avoidance, 8 IDs per channel at `ch * 8`, so a third channel lands on
   IDs 16-23 (past the 4-bit ID).
3. VC2 worker API indexes by fabric-max flat position, the kernel by per-router compact `VC2_LOCAL_CHANNEL_START`; they
   differ on Z-boundary fabrics (fabric_vc2_connection.cpp:161-164 vs ct_args:99).

## Extension: several 1D line VCs for native worker fan-in (user, 2026-11-01; queued after the VC2 line bring-up)

The idea (user): allow 1D routing on VC2 too, and more generally up to 3-4 line VCs on a 1D (line or ring) fabric,
each with its own worker channel. Why: many ops have a fan-in of more than one worker writing into the fabric through
the same router. Today a router takes one worker connection per VC, so even a fan-in of 2 needs a mux (a Tensix core
per link that matmuls could use, plus an extra hop). With N line VCs a router takes N worker connections natively. The
router shape barely changes: the VCs are not special; the host says how many line VCs the fabric has (up to a maximum
found by experiment; 3 should be reachable, since 2D already runs 3 VCs), and the router follows the host's per-VC
features (the ownership pass, V4_RECEIVER_OWNERSHIP.md / V4_SENDER_OWNERSHIP.md). Every pass already checks each
worker connection in one group.

Questions to settle first (main session, not yet checked):
- Per-VC resources on 1D (one worker sender, one forwarding sender, one receiver per line VC):
  - slots: 82 at 4352 B with the 29 KiB buffer, split N ways; per-VC windows shrink. That is fine while the workers
    share the link, but limits a lone busy VC; the slot study's window model applies;
  - transaction IDs: 16 per NoC on E1, today 8 per receiver channel with deadlock avoidance; with N receivers each
    gets fewer, which changes completion-group proofs (related problem 2 above);
  - stream registers per ERISC, the flat sender-channel table (max 10 positions), credit blocks, code size.
- Per-pass cost of each extra worker channel on E0: its free-slots read and connection check join the group, but a
  group overlaps only up to 4 reads in flight (RESULTS §17). Idle VC2 cost ~7.7 cycles/packet on 2D (§29-§30); 1D
  1-hop has ~14 of E0 slack at the wire. Channel trimming removes unused VCs in deployment.
- Rings: bubble flow control + first-level acks per VC (VC2 rings exist since phase 2); line VCs never cross over, so
  no inter-VC dependency.
- Host/API: the number of line VCs as a fabric setting; the worker connection API per VC (VC2 has its own today);
  slot, stream, trid allocation per VC count; on a 1D fabric VC1 has no inter-mesh role and could be a line VC too.
- Alternative to compare: several worker channels on one VC (fan-in into one far ring, as forwarding channels merge
  today). That costs less (no extra receiver, forwarding channel or trid range), but gives no isolation between flows
  and changes the one-worker-per-VC rule.

Addition (user, 2026-11-01): on 1D fabrics, VC1 need not be the inter-mesh VC; it can be one more line VC. The user
expects the router shape to be neutral to this, with the change on the host side. Main session's check, from code
reads, of what must hold for that:
- Per-VC facts come from the host (the ownership passes: serviced, first-level acks, forwarding, crossover,
  intermesh ingress per receiver channel). The host fixes VC1's first-level ack at 0 today ("must agree across every
  VC1 link"); a 1D VC1 ring needs it per VC.
- Worker role comes from the host arg SENDER_CH_<i>_IS_WORKER_CHANNEL (sender option (a)), so a VC1 worker channel
  needs no kernel change.
- Crossover is a VC1-named fact (FABRIC_2D_VC0_CROSSOVER_TO_VC1, VC2_CROSSOVER_TO_VC1). Generalize it to a per-VC
  "forwards on VC" target when this work starts.
- The kernel's supported-VC-combination gate (VC1+VC2 / line-mode gates) moves to the host; the kernel keeps
  consistency static_asserts only.
- Audit remaining VC-number assumptions: sender channel counts per VC, the flat sender table, trid ranges (ch * 8),
  telemetry indices, slot rows.

Census and first perf (VC2 line agent, 2026-11-01; /home/aagarwal/tt-metal-vc2line/fabric_exp/consolidation/
VC2_LINE_1D_VCS.md and VC2_LINE_BRINGUP.md "Perf: idle and full VCs"; Galaxy jobs 167481/167482):
- N = 3 line VCs fit (code ~24 KB of ~29.2 KB on two ERISCs; 4 transaction IDs per receiver channel; 12 of 30
  stream registers; 6 of 10 sender-table positions). N = 4 overflows on code (estimate) and needs code past the
  three-VC structure.
- An idle line VC costs a busy VC0 1-hop flow +17 to +20 cycles/packet at 256 B-1 KiB (two ERISCs; one ERISC +53);
  multi-hop and multicast +46 to +65 at 4352 B. The wire hides it only on 1-hop at 4352 B.
- Several busy VCs on one link: at 4352 B they reach the wire. At 256 B-1 KiB the total is +9% over one VC (+27% with
  doubled windows), far below the wire. So line VCs buy native fan-in, not router capacity (router per-pass work caps
  small sizes: hypothesis). VC1's 4/4 windows bind VC0 flows into VC1 (-29% to -39% doubled).
