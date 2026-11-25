# fabric_connections_spec invariant tests

Local invariants of `FabricConnectionsSpec` (`fabric_connections_spec.hpp`).
Rules that need the kernels that bind a spec, or every spec of the Program, are structural and live in
`../program_spec/`.

## Listed invariants

`FabricConnectionsSpec` as declared in `fabric_connections_spec.hpp`, with every field and only its invariants.

```cpp
struct FabricConnectionsSpec {
    FabricConnectionsSpecName unique_id;

    // Invariant:
    // - Must not be empty.
    // - Must not name a target more than once.
    // - Fabric1DTargets: each direction is N, E, S, W or Z, and the fabric is in a 1D mode.
    // - Fabric1DTargets: on each device that the Program runs on, each direction has a fabric neighbour.
    // - Fabric2DTargets: the fabric is in a 2D mode.
    // - On each device, the targets resolve to different directions.
    std::variant<Fabric1DTargets, Fabric2DTargets> targets;

    // Invariant:
    // - Must not be empty.
    // - On each device, each node's link exists in each resolved direction.
    Table<NodeCoord, uint32_t> links;
};
```

## Known gaps

- Fabric 2D: two peers whose first hops share a direction are not tested here.
