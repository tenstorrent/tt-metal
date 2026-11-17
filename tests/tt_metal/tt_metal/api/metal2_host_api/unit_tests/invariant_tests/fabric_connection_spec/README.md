# fabric_connection_spec invariant tests

Local invariants of `FabricConnectionSpec` (`fabric_connection_spec.hpp`). The rules that need the binding kernels
(every spec is bound, every target node runs exactly one binding kernel, each (src, dst, link) is used once) are
structural and are listed in `../program_spec/`. The binding invariants are listed in `../advanced_options/`.

## Listed invariants

`FabricConnectionSpec` as declared in `fabric_connection_spec.hpp`, with every field and only its invariants.

```cpp
struct FabricConnectionTarget {
    tt::tt_fabric::FabricNodeId dst_fabric_node_id;
    uint32_t link_idx = 0;
};

struct FabricConnectionSpec {
    FabricConnectionSpecName unique_id;

    tt::tt_fabric::FabricNodeId src_fabric_node_id;

    // Invariant: each node has at most one target.
    Table<NodeCoord, Group<FabricConnectionTarget>> targets;
};
```

Not checked here: a target's `dst_fabric_node_id` and `link_idx` must name a forwarding link of
`src_fabric_node_id`. The fabric control plane checks this when the Program is built.
