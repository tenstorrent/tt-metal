# TT-Fabric V1 Link Reliability on Blackhole

Status: Initial design discussion
Scope owner: TT-Fabric builder and ERISC router kernels
Control-plane changes: Deliberately deferred

# Document organization

This document separates two kinds of content:

- **Stable design requirements and invariants** describe the intended behavior
  independently of the current builder classes, kernel layout, channel numbers,
  and stream-register assignments. Incoming implementation refactors must
  preserve these properties.
- **Current implementation mapping** records how the stable design could map
  onto the code that exists today. These sections are evidence and an initial
  implementation candidate, not architectural requirements.

The words **must**, **must not**, and **required** are used only for stable
design requirements unless a section is explicitly marked non-normative.

# Part I - Stable V1 Requirements

## 1. Purpose

Blackhole can have up to two Ethernet links between a pair of neighboring
devices. A device can have up to ten links in total, corresponding to at most
five neighbors.

Today, fabric initialization expects the required links to be alive. If a link
does not come up during initialization or cluster reset, the practical recovery
is to reset again until all required links are healthy.

The v1 reliability goal is:

- allow fabric initialization to continue when exactly one of the two links
  between a device pair is down;
- preserve the two logical routing-plane endpoints presented to fabric clients;
- carry the failed plane's traffic over the surviving physical link;
- make the repair transparent to fabric users and worker kernels;
- accept degraded bandwidth and latency;
- preserve the routing and flow-control semantics of every active logical
  transport VC;
- keep ERISC L1, code-size, and stack overhead small.

The proposed direction reuses the resource budget that would otherwise belong
to disabled worker-facing VC2, but it is not a VC remapping. A correct repair
needs:

1. a logical endpoint for the failed routing plane;
2. a specialized NoC-to-private-carrier-to-NoC path;
3. preservation of the original logical sender identity;
4. end-to-end logical ACK/completion transport;
5. builder support for separating logical planes from physical links.

In this document, "reuse VC2" means reuse VC2's place in the static channel/L1
resource budget. It does not mean reuse worker-facing VC2 behavior, APIs, or
routing semantics.

## 2. V1 scope

### In scope

- Blackhole only.
- Static repair selected during fabric build after link health is known.
- Exactly two physical links between the affected device pair.
- Exactly one of those two links is down.
- The surviving link is healthy in both directions.
- The failed object is an Ethernet link, not a device or NoC failure.
- The feature applies to every direction and router type. It is not specific to
  N/E/S/W, Z, intra-mesh, or intermesh routers.
- The failed-plane logical endpoint remains on its original Ethernet tile for
  the initial design.
- Link training failure does not remove ERISC programmability or access to the
  failed tile's L1.
- Fabric clients continue to see both logical routing planes even though only
  one physical link is carrying traffic.
- Existing fabric clients continue to select and connect to the same logical
  routing plane.
- Every active logical transport VC retains its existing semantics, regardless
  of whether that VC is currently client-visible or internal to fabric.
- VC2 storage and channel resources may be reused as a private reliability
  carrier.
- Reliability mode and user-visible VC2 mode are mutually exclusive.
- Performance degradation is allowed, but starvation and avoidable
  head-of-line blocking are not.

### Enablement and feature composition

- Reliability is opt-in through a dedicated environment-controlled runtime
  option and is disabled by default.
- The option is resolved before fabric build so the builder can emit
  reliability-specific kernel compile-time configuration.
- When reliability is enabled, worker-facing VC2 channels, connection APIs, and
  neighbor-exchange semantics are not built.
- The VC2 channel/L1 allocation is instead assigned to the private reliability
  carrier.
- Reliability and ordinary user-visible VC2 are mutually exclusive in v1.
- Sharing one VC2 receiver between worker and reliability traffic is excluded
  from v1.
- Adding a fourth VC or private carrier lane alongside worker VC2 is also
  excluded from v1.
- Coexistence may be reconsidered only after the mutually exclusive baseline is
  working and its resource usage is measured.
- This environment option selects an initialization-time build variant. It is
  not runtime failover or a device-side dynamic mode switch.

### Baseline scheduling policy

- The first implementation preserves the current static channel-service
  structure and cadence, replacing the worker VC2 service with private
  reliability-carrier service.
- The logical-credit bridge receives bounded service independently of data.
- The baseline does not promise equal bandwidth between the healthy and
  repaired logical planes and does not introduce weighted scheduling.
- Lack of a bandwidth guarantee does not permit deadlock or starvation: ready
  repaired traffic and its credits must continue to make progress.
- Scheduling call order, service ratios, and weighting are optimization work
  to consider only after the baseline data and credit paths are functionally
  validated and characterized.

### Out of scope for this discussion

- Control-plane discovery and representation of failed links.
- Runtime failover after workloads have started.
- Repair when both links to a neighbor are down.
- Repair around a failed device.
- Bandwidth guarantees while repaired.
- More than one fallback hop.
- Sharing reliability mode with fabric mux or UDM mode.
- Worker-facing VC2 coexistence or a fourth reliability VC.

### Resolved architectural requirements

- Direction and router type are parameters, not feature gates.
- The original failed-link Ethernet tile remains programmable and its L1
  remains usable.
- The failed-plane sender/receiver functionality remains on that original tile
  in v1.
- Clients must continue to observe two logical routing planes.
- Reliability logic must preserve semantics for the set of active logical
  transport VCs and must not hardcode the current client-visible VC set.

### Open architecture decision

The control plane and fabric builder must jointly provide two logical routing
planes while mapping the failed plane onto the healthy physical carrier. The
ownership and exact representation of that logical-to-physical mapping still
need to be defined.

## 3. Terminology

- **Failed plane / P0**: the logical routing plane whose physical Ethernet link
  is down.
- **Carrier plane / P1**: the logical plane whose physical link remains healthy.
- **Failed-link endpoint / F**: the original Ethernet-tile router endpoint for
  P0. It retains P0's sender/receiver VC semantics but does not use the failed
  Ethernet link.
- **Carrier router / H**: the normal router on P1. Its active logical VCs
  continue to carry P1 traffic. A private carrier channel also carries repaired
  P0 traffic.
- **Logical credits**: P0 ACK/completion state exchanged end-to-end between the
  two failed-link endpoints.
- **Transit credits**: local NoC and healthy-link private-carrier credits that
  protect the carrier buffers.

### Meaning of neighbor exchange for the reliability carrier

"Neighbor exchange" describes only the routing capability of the private
reliability carrier:

```text
local carrier TX
  -> one healthy Ethernet hop to the same neighboring device
  -> remote carrier RX
  -> drain into the remote failed-link endpoint's original RX VC
```

The carrier does not route the packet beyond that neighboring device and does
not execute the packet's final local NoC command. Its remote endpoint is the
failed-link router, not the workload destination.

This does not restrict the repaired workload packet to neighbor-only traffic.
Once deposited in the original failed-plane RX channel, the packet resumes the
ordinary receiver flow exactly as if it had arrived over the failed link. It
may then be delivered locally or forwarded for additional logical fabric hops.

# Part II - Current Implementation Context (Non-Normative)

The following two sections describe the repository at the time of this design
discussion. Names, channel geometry, and build phases may change without
changing the stable design.

## 4. Current fabric build flow

The relevant host flow is:

```text
ControlPlane
  - maps logical topology to physical devices and Ethernet channels
  - assigns N/E/S/W/Z direction to live channels
  - chooses routing-plane counts
  - generates and writes routing tables

create_and_compile_tt_fabric_program()
  FabricBuilder::discover_channels()
  FabricBuilder::create_routers()
  FabricBuilder::connect_routers()
  FabricBuilder::compile_ancillary_kernels()
  FabricBuilder::create_kernels()
  Program::compile()
```

The current builder makes these important assumptions:

- `discover_channels()` receives only active channels from the control plane.
- `create_routers()` creates one router per active physical Ethernet channel.
- parallel physical links are independent routing planes;
- local router-to-router connections pair channels by the same vector index;
- there is no cross-talk between routing planes;
- a down channel is omitted rather than represented by a logical router.

The current relaxed reliability behavior reduces routing-plane counts and trims
extra channels. It does not preserve a missing logical plane and tunnel that
plane over a surviving link.

V1 repair therefore needs an explicit build-time distinction between:

```text
logical router endpoint  !=  physical Ethernet carrier
```

## 5. Current router and VC behavior

At the time of writing, VC0 is the client-visible transport VC. Other VCs can
still carry internal fabric semantics. Reliability therefore operates on the
configured set of active logical transport VCs rather than hardcoding VC0 or
any current client-visible set.

### VC0

- One receiver ring.
- Four sender channels on a normal 2D mesh router:
  - one worker injection channel;
  - three forwarded directions.
- Used for intra-mesh traffic.
- Injection and turn channels use bubble flow control on ring/torus axes.
- First-level ACK is enabled when deadlock avoidance requires it.

### VC1

- One receiver ring.
- Usually three sender channels, or four when a Z connection is present.
- Used to preserve intermesh traffic isolation.
- Does not use the VC0 bubble-flow-control protocol.

### VC2 today

- Blackhole and 2D only.
- Enabled only when mux/UDM are disabled.
- One sender channel and one receiver channel on a non-Z mesh router.
- Worker/neighbor-exchange semantics.
- The receiver never forwards.
- Uses the speedy sender/receiver path.
- Uses stream 30 for sender flow control and stream 31 for receiver flow
  control; both IDs already have other compile-time uses.
- Has no ordinary per-source ACK/completion stream allocation.
- Is not covered together with 2D torus, bubble flow control, intermesh VC1,
  and two physical links by existing hardware tests.

The reliability design should therefore reuse VC2's allocated channel and
buffer space, but replace its behavior at compile time. Worker-facing VC2 is
compiled out while reliability is enabled. The private carrier happens to be a
one-hop neighbor exchange as well, but its drain-to-failed-endpoint behavior is
distinct from existing VC2 local-delivery semantics.

# Part III - Stable Functional Design

## 6. Topology reference

### Healthy two-plane pair

```text
                    logical/physical plane P0
      Device A  [ router A.F ] ==================== [ router B.F ]  Device B

                    logical/physical plane P1
                [ router A.H ] ==================== [ router B.H ]
```

### Repaired pair

```text
                                  FAILED ETH LINK
                 logical P0       - - - - - - -
      Device A  [ failed A.F ]                       [ failed B.F ]  Device B
                    |   ^                                 ^   |
                 NoC|   |NoC                           NoC|   |NoC
                    v   |                                 |   v
                +-----------+                         +-----------+
      logical P1| carrier   |==== healthy ETH link ===| carrier   |logical P1
                | A.H       |  active VCs: normal P1  | B.H       |
                |           |  private carrier: P0    |           |
                +-----------+                         +-----------+

      P0 data A -> B:
        A.F logical TX VC
          -> NoC
          -> A.H private reliability TX
          -> healthy Ethernet link
          -> B.H private reliability RX
          -> NoC demux
          -> B.F original logical RX VC

      P0 data B -> A is the exact reverse path.
```

The carrier's normal logical-VC path and repaired private transit path coexist
on the healthy physical link. They do not share logical receiver rings.

### Data and credit paths in one direction

```text
DATA
  A.F failed-link sender
      --NoC--> A.H private carrier sender
      --ETH--> B.H private carrier receiver
      --NoC--> B.F failed-link receiver

LOGICAL ACK/COMPLETION
  B.F failed-link receiver
      --NoC--> B.H credit staging
      --ETH TXQ1--> A.H credit staging
      --NoC--> A.F failed-link sender credit counters

TRANSIT COMPLETION
  B.H returns transit completion to A.H only after the
  B.H -> B.F NoC write has flushed.
```

## 7. Core correctness invariant

Before A.F places a P0 packet into the shared carrier, it must reserve space in
the destination logical receiver ring on B.F.

This gives the following invariant:

> Every packet admitted to the private reliability carrier already owns a slot
> in its final failed-link endpoint VC receiver ring.

Consequences:

- B.H never waits for B.F receiver space while holding a reliability-carrier
  packet.
- Multiple logical VCs may share the physical carrier without sharing their
  logical credit pools.
- A blocked logical VC cannot consume another logical VC's destination slots.
- Any logical VC that requires bubble admission remains governed by its final
  P0 receiver ring, not by aggregate carrier space.
- The carrier FIFO can add latency and head-of-line delay, but it does not add a
  cyclic wait on logical destination capacity.

This invariant is what makes one shared physical transit ring potentially safe.
Without it, aggregate carrier flow control collapses VC isolation and is not
deadlock-safe.

The source failed-link endpoint's logical admission state and destination
carrier's deposit state must also remain synchronized:

- both initialize each logical VC to the same receiver-ring slot;
- the source reserves and advances once for every admitted packet;
- the destination advances once for every deposited packet;
- the healthy carrier preserves packet order;
- reset or rebuild reinitializes both ends before traffic is enabled.

The source does not need to send a destination address with every packet if
these pointers are initialized together and advance deterministically. A lost,
duplicated, or skipped carrier packet would desynchronize them; v1 therefore
assumes the selected carrier link itself is reliable after initialization.

## 8. Device-side reliability handshake

Reliability bring-up is a device-side protocol. The host still clears and
launches the configured router programs as usual, but it does not coordinate
the reliability handshake.

The failed-link endpoint cannot run the ordinary Ethernet handshake on its
untrained link. Instead, bring-up establishes the following readiness chain:

```text
A.F <--local NoC--> A.H <--healthy Ethernet--> B.H <--local NoC--> B.F
```

This chain is sufficient to prove that both failed endpoints and both carrier
bridge roles are initialized. The failed endpoints do not need to exchange an
end-to-end zero-credit snapshot.

### 8.1 Per-tile initialization

Before advertising readiness, every participating Ethernet tile must:

1. initialize both local ERISCs and their role-specific state;
2. initialize all reliability staging, counter, pointer, mailbox, and
   notification words;
3. initialize logical processed counters and publication sequences to zero;
4. complete the ordinary same-tile ERISC0/ERISC1 synchronization.

The same-tile ERISC synchronization is necessary but not sufficient: it cannot
synchronize the failed and carrier Ethernet tiles. Those tiles use the local
NoC handshake below.

### 8.2 Local carrier-to-failed-endpoint handshake

The carrier is the designated initiator. After its local bridge state is ready,
it repeatedly NoC-writes a `LOCAL_CARRIER_READY` request to the failed endpoint.

The failed endpoint initializes its receive word before waiting. It responds
with `FAILED_ENDPOINT_READY_ACK` only after all of its sender, receiver, credit,
and mailbox state is initialized. The carrier does not declare the local pair
ready until it observes this response.

The request is retried, matching the robustness property of the ordinary
Ethernet handshake: a request that arrives while the failed endpoint is still
initializing may be overwritten, but the next request will be observed.

This exchange runs independently on both neighboring devices.

### 8.3 Carrier-to-carrier reliability handshake

After each carrier has synchronized with its local failed endpoint, the two
carriers exchange reliability-ready state over the healthy Ethernet link.
Neither carrier may acknowledge peer reliability readiness until:

- its ordinary physical Ethernet handshake on the healthy link is complete;
- its local carrier-to-failed-endpoint handshake is complete;
- ERISC0's inbound credit-forwarding state is initialized;
- ERISC1's outbound staging and TXQ1 publication state are initialized.

The initial implementation should keep this reliability-ready exchange
separate from the ordinary physical-link handshake. This avoids changing the
meaning of the existing handshake and can directly validate the TXQ used by
the logical-credit bridge. Reusing the physical handshake is valid only if
entry into it is gated on the same local reliability-ready conditions.

### 8.4 Final local release

After the carrier-to-carrier reliability handshake completes, each carrier
NoC-writes `RELIABILITY_PATH_READY` to its local failed endpoint. This final
release tells the failed endpoint that the remote carrier and remote failed
endpoint are also ready.

The repaired data and logical-credit paths may enter normal service only after
this release. No participant clears or reinitializes reliability state after
advertising readiness.

The resulting startup sequence is:

```text
per-tile initialization and local ERISC synchronization
    -> local failed-endpoint/carrier NoC handshake
    -> carrier/carrier reliability handshake
    -> carrier-to-local-failed-endpoint final release
    -> normal routing service
```

There is no reliability-specific runtime reset, epoch, independent endpoint
restart, or `FAULT` recovery state in v1. Existing fabric termination and
teardown behavior remains responsible for ending the program.

## 9. Functional data path

### 9.1 Initialization contract

Before router programs are built, fabric setup must have a symmetric repair
description containing at least:

```text
local device
remote neighbor
direction
failed logical plane id and failed eth channel
carrier logical plane id and healthy eth channel
remote failed/carrier channel pairing
set of active logical transport VCs
```

Whether the control plane owns this complete object or the builder derives part
of it remains open. The stable requirement is that both sides agree on the
logical planes, physical carrier, active VCs, and endpoint addresses before
traffic is enabled.

For each repaired pair, fabric setup must:

1. create a normal carrier router on the healthy channel;
2. keep the failed-plane logical endpoint active on its original Ethernet
   tile;
3. preserve both planes in the client-visible logical router registry;
4. connect the failed-link endpoint to the same local logical graph that the
   healthy physical implementation of that plane would have used;
5. connect the failed-link TX scheduler to one private carrier sender through
   a local NoC path;
6. connect the remote private carrier receiver to every active logical
   receiver VC on the remote failed-link endpoint;
7. configure a bidirectional logical-credit bridge;
8. expose the failed-link endpoint's ordinary worker connection information
   for P0;
9. expose the carrier router's ordinary worker connection information for P1.

Logical connection pairing must use explicit logical plane identity. It must
not infer logical identity only from the number or ordering of surviving
physical links.

### 9.2 Failed-link sender behavior

The failed-link sender retains the ordinary fair scheduling policy over its
logical sender channels.

For a candidate logical sender channel:

1. check that the sender channel contains a packet;
2. check the destination logical VC receiver tracker:
   - ordinary flow control requires at least one free logical slot;
   - any VC/channel requiring bubble flow control applies its ordinary
     reservation threshold;
3. check that the local NoC path feeding the private carrier has space;
4. update the packet's logical routing state exactly as the failed physical
   sender would have updated it for this hop;
5. attach sufficient internal reliability metadata to recover the destination
   logical VC and original source-channel identity;
6. NoC-write the packet into the private carrier sender ring;
7. decrement the destination logical receiver-slot count;
8. release the failed-link sender source slot only after the NoC copy is safe;
9. continue processing logical ACK/completion counters as in the ordinary
   sender loop.

The carrier sender must not update the logical routing state a second time.

### 9.3 Carrier sender behavior

The private carrier sender has:

- one statically connected upstream producer;
- one healthy Ethernet receiver ring;
- one transit credit domain.

Its upstream is the failed-link endpoint rather than a worker. It transmits the
packet and reliability metadata unchanged.

### 9.4 Carrier receiver behavior

The private carrier receiver:

1. receives a carrier packet;
2. reads the destination logical VC and original source-channel identity from
   internal reliability metadata;
3. map that identity to the destination failed-link receiver VC;
4. select the next slot in that failed-link endpoint VC receiver ring;
5. NoC-write the complete packet into the selected slot;
6. signal the failed-link receiver's packet-arrival counter;
7. advance the per-logical-VC failed-link receiver write pointer;
8. return the transit completion only after the NoC write has flushed.

This path is a small static demux, not a fabric routing stage. The destination
failed-link core, receiver addresses, and notification addresses are fixed
during initialization.

The carrier's routing responsibility ends after this NoC deposit. All remaining
logical routing is performed by the failed-link receiver using the original
failed-plane VC state.

### 9.5 Failed-link receiver behavior

After the packet is deposited in its original VC receiver ring, the normal
receiver logic can:

- decode the existing logical routing state;
- deliver locally and/or forward to the same logical plane;
- preserve the semantics of the selected logical VC;
- generate the ordinary per-source ACK and completion state.

The receiver data-path logic can remain mostly unchanged. Its credit backend
changes from direct Ethernet responses to the NoC/ETH/NoC logical-credit
bridge.

## 10. Reliability metadata

The carrier packet must contain enough internal metadata to:

- select the destination logical receiver VC;
- return ACK/completion state to the original logical sender channel;
- validate that the packet belongs to the configured repaired plane if needed.

This metadata is internal to fabric. It must not change the client API or
client-visible routing semantics. The design does not require a specific packet
field or encoding.

If the sender-channel identity uniquely determines the logical VC for a given
build, carrying only that identity is sufficient. Otherwise the metadata must
carry both values explicitly.

### Current implementation candidate: `src_ch_id` (non-normative)

In the current packet layout, a compact candidate is:

```text
src_ch_id = original flat sender-channel ID on the failed logical router
```

The current maximum is ten sender channels, so the existing byte is sufficient.
A build-time lookup can map that flat source ID to the destination logical VC.
The packet then arrives at the failed-link receiver with the same `src_ch_id`
that it would have carried across the failed Ethernet link.

This can avoid increasing the packet header:

- the ordinary sender path uses `src_ch_id` to route credits;
- the current speedy VC2 receiver uses a compile-time producer sender index for
  its own transit completion;
- the specialized reliability VC2 path can therefore preserve `src_ch_id` for
  logical identity.

This mapping is valid only while those current layout properties hold. A future
implementation may use a different field or explicit reliability metadata
without changing the stable design.

## 11. Two-level flow control

### 11.1 Logical end-to-end flow control

The failed-link sender tracks destination receiver capacity independently for
every active logical transport VC.

Logical completion:

- frees a slot in the destination logical receiver ring;
- updates the failed-link sender's remote-receiver free-slot count.

Logical first-level ACK, when enabled:

- returns upstream producer credit early;
- preserves the ordinary bubble-flow-control behavior of any VC/channel that
  requires it.

These credits must be independent per original sender channel. An aggregate
transit completion is not a replacement.

### 11.2 Transit flow control

Transit flow control protects:

1. the NoC connection from failed-link sender to carrier sender;
2. the carrier sender/receiver ring across the healthy Ethernet link;
3. the NoC copy from carrier receiver to failed-link receiver.

Transit completion may be returned once the packet is safely deposited in the
pre-reserved failed-link receiver slot. It must not wait for that receiver to
forward the packet through the logical fabric.

This decoupling is important:

- logical completion protects network deadlock invariants;
- transit completion keeps the healthy carrier pipeline moving.

## 12. Logical-credit transport options

Any implementation of logical-credit transport must:

- make progress without reverse data traffic;
- preserve the credit semantics of every active logical VC;
- identify the original logical sender channel;
- receive bounded service relative to carrier data;
- avoid consuming an unbounded amount of ERISC L1 or stack;
- remain independent of the current direction and router type.

### Current candidate A: Occupied cumulative-snapshot bridge (non-normative)

Recommended starting point for the current Blackhole implementation.

For one data direction, logical credits follow:

```text
B.F failed-link receiver
  -> NoC
B.H carrier ERISC1
  -> healthy-link credit TXQ
A.H carrier ERISC0
  -> NoC
A.F failed-link sender
```

The reverse data direction has an independent instance of the same protocol.

#### Snapshot layout

One directional snapshot contains:

```text
ACK[N]
completion[N]
alignment padding
publish_seq       // final word
```

where `N` is the number of logical sender channels on the failed endpoint.
`publish_seq` is a runtime publication generation, not an initialization epoch.
It starts at zero and advances once for every published cumulative snapshot.

The final-word placement is required: ordered NoC and Ethernet copies make the
new sequence visible only after the preceding cumulative words.

#### Failed-receiver publication mailbox

The failed-link receiver maintains:

- cumulative ACK and completion words in its local publication snapshot;
- a 32-bit `mailbox_occupied` word, initially zero;
- a dirty indication for counter changes not yet published;
- the next `publish_seq`.

ACK and completion generation always updates the local cumulative words and
marks the snapshot dirty. When dirty and `mailbox_occupied == 0`, the failed
receiver:

1. sets `mailbox_occupied = 1`;
2. advances `publish_seq`;
3. NoC-copies the full snapshot to its local carrier's outbound staging;
4. waits until NoC has consumed the local source;
5. leaves subsequent counter changes accumulated locally until the mailbox is
   released.

The failed endpoint must not overwrite carrier outbound staging while the
mailbox is occupied.

#### Destination-carrier TXQ publication

Carrier ERISC1 detects a snapshot when outbound staging contains a
`publish_seq` different from its last transmitted sequence. It requests
reliability service from the credit-TXQ arbiter.

When granted, ERISC1 sends the full snapshot over the healthy link and waits
until the TXQ has finished consuming the source block. It then records the
transmitted sequence and NoC-writes `mailbox_occupied = 0` to the failed
receiver.

If logical credits changed while occupied, the failed receiver immediately
publishes the newest cumulative state after observing the clear. Intermediate
states may be coalesced.

#### Source-carrier forwarding

Carrier ERISC0 polls the inbound snapshot's `publish_seq`. When it differs from
the last forwarded sequence, ERISC0:

1. captures the inbound sequence value;
2. NoC-copies the ACK and completion words to the failed sender's counter view;
3. NoC-writes the captured sequence to the failed sender as the final commit;
4. records the captured sequence as forwarded.

The inbound Ethernet block may be overwritten by a newer snapshot during this
NoC copy. This is safe because every copied counter word is cumulative and
monotonic. The importer commits the older captured sequence; the newer inbound
sequence remains different and therefore triggers another copy.

#### Failed-sender consumption

The failed sender consumes a newly committed sequence by unsigned subtraction
from its processed ACK and completion values. It processes completion before
ACK, matching the ordinary sender behavior.

Completion deltas restore logical receiver slots and return producer credit
only for completion-only channels. ACK deltas return producer credit only for
channels with first-level ACK enabled.

The consumer must bound-check:

- completion deltas against sent-but-not-completed logical packets and the
  final logical receiver-ring depth;
- ACK deltas against admitted-but-not-ACKed packets.

#### Credit-TXQ arbitration

Ordinary and reliability credit snapshots share the credit TXQ through a
one-bit, work-conserving arbiter. If both classes are pending, the arbiter
services the selected class and flips the bit. If only one is pending, it is
serviced immediately and the next preference is assigned to the other class.

The occupied mailbox may delay another reliability publication, but cumulative
counters continue to advance locally. Bounded TXQ service guarantees eventual
mailbox release.

Properties:

- credits do not depend on reverse data traffic;
- cumulative 32-bit counters tolerate duplicate snapshots and wrap using
  unsigned subtraction, matching the existing Blackhole counter approach;
- no additional stream-register IDs are required;
- credit traffic remains separate from data traffic;
- counter blocks can be copied in bulk;
- skipped intermediate snapshots are safe because the newest snapshot is
  cumulative;
- `publish_seq` replaces per-word last-sent and last-forwarded mirror arrays;
- the occupied mailbox keeps the carrier's TXQ source stable.

Approximate raw storage for the current channel geometry:

```text
N = failed-link sender channels in the current implementation
  = 4 VC0 + 3 VC1 = 7 for ordinary non-Z mesh
  = 4 VC0 + 4 VC1 = 8 when a Z-facing sender is present

X = align16(2 * N * 4 + 4-byte publish_seq)

N=7:
  X = align16(60) = 64 bytes

N=8:
  X = align16(68) = 80 bytes

healthy carrier reliability staging:
  outbound X + inbound X
  = 128 bytes for N=7
  = 160 bytes for N=8

additional carrier state:
  last transmitted sequence + last forwarded sequence
  = 8 bytes

failed endpoint:
  publication snapshot X + inbound counter view X
  + mailbox/dirty/sequence scalar state
```

The failed endpoint should attempt to repurpose its original direct-link
logical-credit storage. The healthy carrier staging is an additional
reliability allocation. Scalar state should not be represented as per-channel
stack arrays.

### Candidate B: Dedicated reliability credit packets

The failed-link receiver injects compact router-generated control packets that
are consumed by the source failed-link sender.

Advantages:

- explicit protocol;
- naturally extensible beyond one failed plane.

Costs and risks:

- packet construction and parsing code on ERISC;
- packet buffering and arbitration;
- a control packet must not wait behind the data whose progress depends on
  that credit;
- requires reserved capacity or strict priority;
- more code and likely more L1 than a counter bridge.

This option should not use the ordinary data queue without a reserved-credit
escape path.

### Candidate C: Piggyback credits on reverse data

Not sufficient as the only mechanism. Reverse data may not exist, so the
forward direction could stop permanently while credits wait to be piggybacked.
Piggybacking may be a later optimization over an independent base mechanism.

## 13. Deadlock and starvation argument

The proposal is conditionally deadlock-safe only if all of the following hold:

1. A logical destination slot is reserved before entering the carrier.
2. Bubble admission for any applicable logical VC uses that VC's final receiver
   count.
3. The carrier receiver never waits for an unreserved failed-link receiver
   slot.
4. Transit completion is returned after the carrier-to-endpoint NoC copy
   flushes.
5. Logical credits have an independent, non-data-dependent transport.
6. Credit transport receives bounded service relative to data.
7. The failed-link sender schedules logical sender channels fairly.
8. Logical routing state is advanced exactly once for the repaired Ethernet
   hop.

Under these conditions, the shared carrier is a bounded transport segment
between already-reserved logical buffers. It adds serialization but does not
merge the logical VC credit pools.

If condition 1 is removed, a carrier head packet can wait for a full logical
receiver while holding capacity needed by another VC. That collapses VC
isolation and can participate in topology-level deadlock.

# Part IV - Current Implementation Mapping (Non-Normative)

The following sections map the stable design onto the repository as it exists
today. Parallel builder and kernel refactors may change these classes and
mechanisms without changing the required behavior.

## 14. Candidate builder changes

Expected builder-level work:

1. Introduce a reliability-plan input type supplied by the control plane.
2. Maintain a logical-router registry keyed by direction and logical plane.
3. Maintain a separate physical-carrier registry keyed by live Ethernet
   channel.
4. Instantiate a failed-plane endpoint builder even though its physical link is
   not live.
5. Instantiate reliability VC2 only on a carrier selected for a failed peer
   plane.
6. Make reliability VC2 and user-visible VC2 mutually exclusive.
7. Wire failed-link TX to carrier VC2 TX through one static downstream adapter.
8. Wire carrier VC2 RX to every active failed-link receiver VC.
9. Wire credit staging, publication sequences, mailbox-occupied words, and
   ordered NoC commit addresses in both directions.
10. Preserve ordinary local connections for the failed logical plane.
11. Return the original failed-link endpoint's worker connection data when a
    client asks for the failed logical plane.
12. Validate symmetric repair plans on both devices before compiling kernels.
13. Wire the local failed-endpoint/carrier handshake and final-release words.
14. Wire the carrier-to-carrier reliability handshake independently from the
    failed physical link's ordinary handshake.

The current `min(chans_dir1.size(), chans_dir2.size())` and vector-index pairing
cannot represent this. Pairing must use explicit logical plane IDs.

This mapping must be generalized across all direction and router variants.
Current special cases such as a Z router having no ordinary VC2 receiver are
properties of user-facing VC2, not limitations on the private reliability
carrier.

## 15. Candidate kernel variants

Two compile-time-specialized variants are preferable to adding runtime branches
to every router.

### Failed-link endpoint

- Runs on the original failed-link Ethernet tile.
- Services all active logical sender and receiver VCs.
- Replaces Ethernet send with one NoC adapter to the carrier VC2 sender.
- Receives packets through NoC writes into its ordinary receiver rings.
- Replaces direct Ethernet credit return with NoC writes to carrier credit
  staging.
- Omits the ordinary Ethernet handshake on the failed link.
- Participates in the local device-side reliability handshake and waits for
  the carrier's final `RELIABILITY_PATH_READY` release.

Link training failure does not remove ERISC or L1 programming capability. Any
current software classification that omits the down channel or chooses a
particular RISC assignment is an implementation issue to adapt, not an
architectural availability constraint.

### Healthy carrier endpoint

- Keeps ordinary behavior for every active logical VC on its own plane.
- Replaces user VC2 with a private reliability carrier.
- Uses the existing one-sender VC2 TX shape.
- Uses a specialized VC2 RX demux-to-failed-endpoint path.
- Bridges cumulative logical credits.
- Initiates the local failed-endpoint handshake and participates in the
  healthy-link carrier reliability handshake.
- Does not run generic VC2 local-delivery semantics.

## 16. Feasibility assessment

### Parts that fit the current architecture

- One healthy link can physically carry its own VCs plus a separate VC2 ring.
- Existing allocator tables already provision VC0+VC1+VC2 combinations.
- VC2 already has a one-producer/one-receiver transport shape.
- Existing router-to-router NoC adapters provide a model for static local
  forwarding and transit flow control.
- Blackhole two-ERISC mode already uses cumulative L1 credit counters and a
  separate Ethernet TXQ for credit traffic.
- `src_ch_id` can potentially preserve logical sender identity without a packet
  header expansion.
- Build-time specialization avoids runtime topology lookup.

### Gaps that require real design/implementation

- Down channels currently do not become fabric routers.
- VC2 RX forwarding is hard-disabled.
- VC2's current speedy RX performs final local delivery, not demux relay.
- VC2 has no per-logical-channel ACK/completion streams.
- Stream IDs 0 through 31 are already assigned or overlaid.
- Current builder pairings assume one physical router per logical plane and no
  plane crossing.
- The sender path must track NoC-copy flush before reusing its source slot.
- The carrier receiver needs per-logical-VC failed-endpoint write pointers.
- Credit relay must have bounded service and a compact implementation.
- Device-side handshake words and startup ordering must be integrated without
  depending on the failed physical link or host-coordinated handshaking.
- Existing VC2 tests do not cover torus + BFC + VC1/intermesh + two links.

### Assessment of the original proposal

The high-level direction is sound:

- use VC2 resources only when reliability is active;
- multiplex failed logical VCs over one healthy-link carrier;
- demultiplex into the original receiver VCs;
- preserve end-to-end logical flow control;
- keep the failed logical sender's round-robin scheduling.

The following refinements are required:

- preserve the original sender-channel identity, not only the VC ID;
- reserve the destination logical receiver slot before entering VC2;
- use a specialized reliability VC2 handler rather than generic VC2
  forwarding;
- transport logical ACK and completion state independently from data;
- explicitly represent the failed logical plane even when its physical link is
  absent;
- apply the same role-based design to every direction and router type.

## 17. Approaches that are insufficient by themselves

- **Only use existing carrier credits.** This merges all failed logical VCs
  into one aggregate credit pool and loses per-VC flow-control semantics.
- **Only stamp the VC ID.** ACK/completion return also needs the original source
  sender-channel ID.
- **Enable generic VC2 forwarding.** This brings in the full 2D forwarding
  switch, does not solve logical credits, and increases code/stack pressure.
- **Return logical credit when the carrier frees a slot.** Carrier completion
  says only that transit storage is free; it does not say the final logical
  receiver slot is free.
- **Depend only on reverse data for credits.** Unidirectional workloads can
  deadlock.
- **Let carrier RX wait for failed-link RX space.** This allows one logical VC
  to block the shared carrier while another VC needs it.

## 18. Failure and validation behavior

Fabric build should fail clearly if:

- both links to the neighbor are down;
- the repair plan is asymmetric across the device pair;
- the failed-link endpoint program or L1 layout cannot be initialized;
- the healthy carrier cannot allocate reliability VC2 resources;
- reliability is requested together with user VC2, mux, or UDM;
- logical channel geometry differs across the repaired pair;
- either side cannot preserve both logical routing planes.

# Part V - Validation and Repository References

## 19. Validation plan

No tests or builds were run while drafting this document.

Suggested implementation validation order:

1. Builder graph unit tests:
   - healthy two-plane baseline;
   - plane 0 down and plane 1 carrier;
   - plane 1 down and plane 0 carrier;
   - symmetric mappings on both devices;
   - worker connections remain keyed by logical plane.
2. Packet identity tests:
   - every active logical source sender channel maps to the correct receiver VC;
   - internal reliability metadata preserves the original source identity;
   - logical routing state advances exactly once.
3. Flow-control tests:
   - fill each logical receiver independently;
   - verify one logical VC cannot consume another VC's credits;
   - verify ACK and completion wraparound using cumulative counters;
   - verify unidirectional data makes progress;
   - verify counter updates accumulated while the mailbox is occupied are
     included in the next publication;
   - verify skipped intermediate publication sequences preserve all credits.
4. Baseline progress tests:
   - sustained traffic on every simultaneously active logical VC;
   - verify a blocked logical channel does not starve other ready channels;
   - verify repaired traffic continues to progress alongside healthy-plane
     traffic;
   - verify credit traffic receives bounded service under saturated data;
   - do not require equal healthy/repaired bandwidth in this phase.
5. Direction and router-role coverage:
   - every N/E/S/W direction;
   - Z and other intermesh router roles;
   - mesh and torus routing;
   - active VCs with and without bubble-flow-control requirements.
6. Resource checks:
   - ERISC code size;
   - stack high-water mark;
   - L1 layout;
   - stream-register aliasing;
   - NoC command-buffer and transaction-ID usage.
7. Device-side handshake tests:
   - delay failed-endpoint initialization and verify repeated local requests
     survive initialization-time overwrites;
   - verify neither carrier acknowledges peer reliability readiness before its
     local failed endpoint is ready;
   - verify final local release occurs only after both carriers are ready;
   - verify the first runtime credit publication starts from sequence one
     without an end-to-end zero snapshot.
8. Post-baseline scheduling characterization:
   - measure healthy-plane and repaired-plane packet and byte throughput;
   - record carrier queue occupancy and data/credit service gaps;
   - measure collective completion skew across repaired and healthy rows or
     columns;
   - use those measurements to decide whether call reordering, repeated
     service, or lightweight weighting is justified.

## 20. Implementation anchors

Host build flow:

- `tt_metal/fabric/fabric_init.cpp`
  - `create_and_compile_tt_fabric_program`
- `tt_metal/fabric/fabric_builder.cpp`
  - `discover_channels`
  - `create_routers`
  - `get_router_connection_pairs`
  - `connect_routers`
- `tt_metal/fabric/compute_mesh_router_builder.cpp`
  - `build`
  - `establish_connections_to_router`
  - `compute_sender_channel_injection_flags_for_vc`
- `tt_metal/fabric/fabric_router_channel_mapping.cpp`
- `tt_metal/fabric/builder/router_connection_mapping.cpp`

ERISC configuration and allocation:

- `tt_metal/fabric/erisc_datamover_builder.cpp`
  - Blackhole two-ERISC split
  - compile-time arguments
  - `setup_downstream_vc_connection`
- `tt_metal/fabric/erisc_datamover_builder.hpp`
  - `StreamRegAssignments`
- `tt_metal/fabric/builder/fabric_static_sized_channels_allocator.cpp`

Router kernel:

- `tt_metal/fabric/impl/kernels/edm_fabric/fabric_erisc_router.cpp`
  - `send_next_data`
  - `run_sender_channel_step_impl`
  - `run_receiver_channel_step_impl`
  - VC2 service loop
- `tt_metal/fabric/hw/inc/edm_fabric/fabric_erisc_router_speedy_path.hpp`
- `tt_metal/fabric/hw/inc/edm_fabric/fabric_router_flow_control.hpp`
- `tt_metal/fabric/hw/inc/edm_fabric/fabric_router_adapter.hpp`
- `tt_metal/fabric/fabric_edm_packet_header.hpp`

Current VC2 API and coverage:

- `tt_metal/fabric/fabric_vc2_connection.cpp`
- `tests/tt_metal/tt_fabric/test_infra/test_yamls/test_fabric_vc2_at_least_2x2_mesh.yaml`

## 21. Recommended next decisions

Before converting this into an implementation plan:

1. Define the control-plane-to-builder ownership and representation of the
   two-logical-plane repair mapping.
2. Agree that a destination logical slot is reserved before carrier admission.
3. Confirm the occupied cumulative-snapshot bridge's TXQ source-consumption and
   destination write-ordering requirements on Blackhole.
4. Select the internal reliability metadata representation.
5. Define the builder-facing reliability-plan contract, including all
   device-side handshake and bridge addresses.
6. Measure the carrier and failed-link endpoint kernel resource budgets before
   fixing the exact counter layout.
