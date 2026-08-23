# Multicast families, exact receiver groups, and chain forwarding

Status: agreed implementation plan

This plan extends the host and device `mcast_pipe` helpers so one semantic
multicast stream can describe several exact receiver groups. It also makes an
irregular group selectable between exact multi-rectangle multicast and
row-major chain forwarding. The first production proof is GroupNorm; Conv3D is
the proof for chain forwarding.

The execution tracker is [`tracker.md`](tracker.md).

## Goals

- Represent one semantic stream as an `McastFamily` containing independent
  `McastGroup`s.
- Let one sender `send()` cover any number of exact, non-overlapping multicast
  rectangles while each receiver still calls `receive()` once.
- Hide rectangle decomposition, group selection, coordinate conversion,
  semaphore ownership, and row-major chain topology behind the helper.
- Preserve dense multicast as the preferred transport, even when chain
  forwarding is enabled.
- Cover both payload and signal protocols with exact multi-rectangle multicast
  and chain forwarding.
- Preserve ordinary `Mcast1D`, `Mcast2D`, and `McastArgs` usage through
  convenience wrappers and compatible serialization.

## Host model

### `McastFamily`

An `McastFamily` is one semantic multicast stream, such as `in0`, `in1`,
weights, statistics, or a control release. It owns the protocol configuration
shared by all its groups:

- semaphore allocation and protocol kind;
- handshake and fixed-versus-rotating sender policy;
- logical-to-physical coordinate conversion and NoC selection;
- group validation and per-core group selection;
- rectangle decomposition or chain topology generation;
- the existing `McastArgs` compile-time/runtime serialization contract.

A core selects its group through runtime arguments in the same way that a core
selects its row or column in `Mcast1D` today. A family produces all arguments
needed to construct the appropriate sender or receiver pipe. Callers do not
construct per-rectangle pipes or chain links themselves.

### `McastGroup`

Each group contains:

- an exact, potentially non-dense `CoreRangeSet receiver_set`;
- a fixed sender or rotating sender schedule;
- `use_chain_forwarding`, explicit at host construction and `false` by
  default.

`receiver_set` remains a static property of the group. If the active sender for
round `r` belongs to that set, the cores that receive in that round are:

```text
active_receivers(r) = receiver_set - {active_sender(r)}
```

Thus a rotating sender can be a receiver in other rounds without changing the
meaning of the receiver set.

The group footprint is:

```text
footprint = receiver_set union sender_schedule
```

Group footprints within one family must be disjoint. Sender/receiver overlap
inside one group is valid and expected. Including sender schedules in the
cross-group check prevents a sender in one group from accidentally
participating in another group's wire or semaphore protocol.

All groups in a family must be wire-compatible. At minimum, they share protocol
kind, semaphore behavior, handshake policy, sender mode, and a compatible
rotation span. Invalid families fail during host construction rather than
creating per-core ABI ambiguity.

### Convenience forms

- `Mcast1D` constructs a family with one group per logical row or column.
- `Mcast2D` constructs a family containing one group.

Ordinary dense uses retain their current `McastArgs` and kernel shape. The
wrappers should remain the normal concise API when the caller does not need an
explicit family.

## Transport selection

Transport is selected per group and round from the exact active receiver set:

1. An empty/local-only destination is a no-op or local completion as required
   by the existing protocol.
2. A dense rectangle always uses one hardware multicast, regardless of
   `use_chain_forwarding`.
3. An irregular set with `use_chain_forwarding == false` uses its exact
   non-overlapping rectangle decomposition. One logical send covers every
   rectangle.
4. An irregular set with `use_chain_forwarding == true` uses a row-major chain.

The rectangle decomposition is allowed to produce any required number of
rectangles; it is not capped at the three needed by GroupNorm. It must cover the
active receiver set exactly, with no duplicate destinations and no bounding-box
holes.

Chain order is never a user argument. The helper derives it deterministically
in logical row-major order `(y, x)`, places the active sender at the head, and
orders all remaining group cores row-major. Logical ordering is established
before physical-coordinate conversion so topology does not depend on NoC
orientation or Blackhole virtualization.

Rotating hardware multicast, including multi-rectangle multicast, is supported.
Rotating irregular chains are rejected in the first implementation because a
different sender would require a different chain head and per-round neighbor
topology.

## Device API and protocol

### Payload API

Payload receivers always supply the forwarding buffer and transfer size:

```cpp
receiver.receive(uint32_t dst_l1, uint32_t size_bytes, uint32_t round = 0);
```

The no-argument payload `receive()` is removed. Existing receivers already know
or derive both values from the destination circular-buffer write pointer and
tile/page count. Making them explicit is what allows a middle chain node to
forward the bytes it just received without exposing chain topology to the
caller.

The sender API remains:

```cpp
sender.send(uint32_t src_l1, uint32_t dst_l1, uint32_t size_bytes,
            uint32_t round = 0);
```

The exact final spelling should follow the current helper's established round
and source-lifetime conventions; the semantic change is the mandatory receiver
destination and size.

### Multi-rectangle multicast

For one logical payload send, the sender:

1. waits for receiver readiness once for the whole active group;
2. issues the payload to every exact, non-overlapping rectangle;
3. performs the required source completion/fence once after all rectangles.

Each destination core executes one `receive(dst_l1, size_bytes, round)`. The
wire must therefore carry the rectangle list/count and group selection needed
by the sender without requiring one pipe or one call per rectangle.

Signal-only operations follow the same transport decision through
`send_signal(value)` and `receive_signal()`. `receive_signal()` remains
no-argument.

### Chain forwarding

For an irregular chain:

- the head waits for its successor and injects the payload;
- a middle node receives into the supplied `dst_l1`, waits for its successor,
  forwards exactly `size_bytes` from that destination, and only then returns;
- the tail receives and returns without forwarding.

This is a per-hop readiness handshake. Phase one requires handshake-enabled
chain payloads rather than introducing an unproven no-handshake lifetime
contract.

Signals use the same topology. A middle node relays the same Flag value or one
Counter event before returning. The helper, not the operation, owns predecessor
and successor coordinates and relay behavior.

## Serialization and compatibility

`McastFamily` owns serialization through the existing `McastArgs` abstraction.
The extended wire needs enough information to identify a core's group and to
construct its selected transport, including a variable rectangle list or chain
neighbors. The implementation should preserve existing field ordering and
fixed-width ordinary forms where possible, and use the helper's opaque
compile-time/runtime block boundaries for extensions.

Compatibility requirements:

- Existing dense `Mcast1D` and `Mcast2D` call sites should not need semantic
  changes.
- Payload receiver call sites undergo one intentional mechanical migration to
  pass destination L1 and size.
- `send_signal()` and `receive_signal()` retain their public shape.
- Runtime operation arguments after a helper block must continue from the
  helper-reported next offset rather than assuming a fixed new width.

## Implementation sequence

The host hierarchy and multi-rectangle device support form one vertical slice;
they are implemented and tested together before changing an operation.

### Stage 1 — family/group model plus multi-rectangle transport

1. Add `McastFamily` and `McastGroup`, validation, per-core group lookup, exact
   rectangle decomposition, and the `Mcast1D`/`Mcast2D` convenience builders.
2. Extend `McastArgs` and sender-pipe construction to serialize/select a group
   and carry an arbitrary rectangle decomposition.
3. Change payload receivers to require `dst_l1` and `size_bytes`; migrate all
   existing payload receiver calls mechanically.
4. Implement one-send/many-rectangles behavior for payloads and for Flag and
   Counter signals.
5. Build host code and pass focused host and device helper tests as one gate.

### Stage 2 — GroupNorm proof

Replace the three GroupNorm `Mcast2D` wires with one family whose exact groups
use `use_chain_forwarding == false`.

- Replace chained `McastArgs`, three sender pipes, and three sends with one
  family, one selected sender pipe, and one send.
- Let the helper decompose the exact receiver set into up to three rectangles;
  do not create fake singleton rectangles to fill absent regions.
- Receive into the global-statistics destination with the exact payload size.
- Preserve the existing early/manual readiness ACK that protects remote
  Welford-stat reads before distribution; it is a separate dependency, not
  redundant helper readiness.
- Cover both legacy and Welford routes.

Do not begin chain forwarding until focused helper tests, the GroupNorm POC,
and the full GroupNorm unit and nightly suites pass.

### Stage 3 — chain transport

Add row-major topology derivation, payload forwarding, and signal forwarding to
the family/pipe implementation. Reject rotating irregular chains explicitly.
Validate chain behavior independently before integrating Conv3D.

### Stage 4 — Conv3D proof

Represent Conv3D weight sharing as one family over exact work groups with
`use_chain_forwarding == true`.

- Dense groups continue to use hardware multicast.
- Irregular groups use helper-owned row-major chains.
- Remove operation-owned predecessor/successor arguments and separate Chain and
  Mcast roles where the family replaces them.
- Do not launch passive bounding-box participants merely to make a dense
  rectangle.
- Receivers pass the weight circular-buffer write pointer and weight-block byte
  count to `receive()`.

Validate one dense and one irregular parameter first, then the POC and complete
Conv3D unit and nightly suites.

### Stage 5 — combined closeout

Run the complete helper regressions, GroupNorm and Conv3D suites sequentially,
check source audits and fresh JIT coverage, compare performance to the existing
routes, and update the public helper documentation and migration records.

## Validation matrix

### Host tests

- group footprint disjointness, including all scheduled senders;
- valid sender/receiver overlap inside one group;
- fixed and rotating schedule compatibility and rotation-span failures;
- exact 1, 2, 3, and N-rectangle decomposition;
- deterministic logical row-major chain topology;
- sender inside and outside the receiver set;
- runtime offsets, both NoCs, and Blackhole coordinate virtualization;
- `Mcast1D` and `Mcast2D` compatibility.

### Device helper tests

- payload and Flag/Counter signal traffic over 1, 2, 3, and N rectangles;
- concurrent disjoint groups with different rectangle counts;
- rotating multi-rectangle multicast;
- one readiness phase and one source-completion phase per logical send;
- dynamic destination pointer and size across repeated sends;
- source lifetime and aliasing cases.

### Chain tests

- two-core and multi-hop payload chains;
- middle-node forward-before-return behavior;
- repeated/pipelined sends with changing destination and size;
- Flag value relay and Counter event relay;
- dense fallback while the chain flag is enabled;
- explicit rejection of rotating irregular chains.

### Operation tests

Run device tests sequentially through `scripts/run_safe_pytest.sh` after
activating `/localdev/sjovic/tt-metal/python_env/bin/activate`. For each
operation, first run one representative parametrization to expose compilation
or protocol errors, then expand to:

- `tests/ttnn/unit_tests/operations/fused/test_group_norm.py`;
- `tests/ttnn/nightly/unit_tests/operations/fused/test_group_norm.py`;
- `tests/ttnn/unit_tests/operations/conv/test_conv3d.py`;
- `tests/ttnn/nightly/unit_tests/operations/conv/test_conv3d.py`.

Host changes require `./build_metal.sh`; kernel-only changes do not.

## Non-goals for the first implementation

- User-specified chain order.
- Rotating irregular chain topology.
- Overlapping group footprints inside one family.
- Bounding-box multicast to inactive holes.
- More than one receiver call for one semantic send.
- Moving GroupNorm or Conv3D synchronization that is independent of multicast
  transport into the helper.

## Completion criteria

The work is complete when the family/group abstraction and both transports are
covered by host and device tests, GroupNorm uses one exact multi-rectangle
family, Conv3D uses dense multicast or row-major chain forwarding per exact
group, all four operation suites pass, and no operation-owned rectangle or
chain topology remains in the migrated paths.
