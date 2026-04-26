# Sub-Context API

When `tt-run` launches an MPI job with a **rank-bindings mapping**, it splits
`MPI_COMM_WORLD` into sub-contexts (via `MPI_Comm_split`).  Each sub-context
gets its own fabric topology, visible devices, and mesh graph descriptor while
still being part of the same MPI job.

All sub-context metadata and translation helpers live on the
`DistributedContext` interface (`distributed_context.hpp`).

For the full design rationale and architecture, see
[example_dual_rankbindings_one_psd.md](../../../models/demos/deepseek_v3_b1/docs/example_dual_rankbindings_one_psd.md).

---

## Launching with tt-run

### 1. Create rank-binding overlay YAMLs

Each sub-context gets its own rank-binding file describing mesh assignments,
visible devices, and (optionally) a mesh graph descriptor:

```yaml
# prefill_rank_bindings.yaml
rank_bindings:
  - rank: 0
    mesh_id: 0
    mesh_host_rank: 0
    env_overrides:
      TT_VISIBLE_DEVICES: 11,13,17,23,24,26,27,29
  - rank: 1
    mesh_id: 0
    mesh_host_rank: 1
    env_overrides:
      TT_VISIBLE_DEVICES: 1,5,9,12,16,19,20,22
mesh_graph_desc_path: "path/to/single_4x4_mesh.textproto"
```

### 2. Create a rank-bindings mapping YAML

This maps each sub-context id to its overlay file:

```yaml
# rank_bindings_mapping.yaml
subcontext_id_to_rank_bindings:
  0: prefill_rank_bindings.yaml
  1: decode_rank_bindings.yaml
```

`tt-run` loads each overlay in ascending sub-context id order and merges them
into one MPI launch with contiguous global rank blocks (sub-context 0 gets
world ranks `0..n₀-1`, sub-context 1 gets `n₀..n₀+n₁-1`, etc.).

### 3. Launch

```bash
tt-run \
  --rank-bindings-mapping rank_bindings_mapping.yaml \
  ./my_binary
```

With a mock cluster (for CPU-only testing):

```bash
tt-run \
  --mock-cluster-rank-binding mock_cluster_desc_mapping.yaml \
  --rank-bindings-mapping rank_bindings_mapping.yaml \
  --mpi-args "--allow-run-as-root --oversubscribe" \
  ./build/test/tt_metal/distributed/distributed_unit_tests \
  --gtest_filter="MpiSubContext.*"
```

### Environment variables set by tt-run

`tt-run` sets these per-rank environment variables before `MPI_Init`.
The `DistributedContext` API parses them automatically at `create()` time.

| Variable | Description |
|---|---|
| `TT_RUN_SUBCONTEXT_ID` | This rank's sub-context id (the mapping key) |
| `TT_RUN_SUBCONTEXT_SIZE` | Number of ranks in this sub-context |
| `TT_RUN_SUBCONTEXT_COUNT` | Total number of sub-contexts |
| `TT_RUN_SUBCONTEXT_SIZES` | Comma-separated sizes of all sub-contexts |
| `TT_MESH_GRAPH_DESC_PATH` | Mesh graph descriptor for this rank's sub-context (if specified) |

---

## C++ API

### Getting communicator handles

```cpp
#include <tt-metalium/distributed_context.hpp>

using namespace tt::tt_metal::distributed::multihost;

// The sub-context communicator (post-split).
// rank() and size() are local to this sub-context.
const auto& ctx = DistributedContext::get_current_world();

// The full, unsplit MPI_COMM_WORLD communicator.
// rank() and size() are global across the entire MPI job.
auto world = DistributedContext::get_world_context();
```

### Querying sub-context layout

```cpp
// Which sub-context does this process belong to? (nullopt if not split)
std::optional<SubcontextId> my_id = world->subcontext_id();

// How many sub-contexts exist?
int count = world->subcontext_count();

// Size of a specific sub-context.
Size sz = world->subcontext_size(SubcontextId{0});

// All sizes as a span, indexed by sub-context id.
tt::stl::Span<const int> sizes = world->subcontext_sizes();
```

### Rank translation

```cpp
// Local rank 0 in sub-context 1 → world rank.
Rank world_rank = world->local_to_world_rank(SubcontextId{1}, Rank{0});
```

### Intra-context communication (within a sub-context)

Use `get_current_world()`. Ranks are local (0-based within the sub-context):

```cpp
const auto& ctx = DistributedContext::get_current_world();
ctx->send(buffer, Rank{1}, Tag{10});   // send to local rank 1
ctx->recv(buffer, Rank{0}, Tag{10});   // recv from local rank 0
```

### Inter-context communication (across sub-contexts)

Use `get_world_context()` with translated world ranks:

```cpp
auto world = DistributedContext::get_world_context();

// Send to local rank 0 in sub-context 1, using its world rank.
Rank dest = world->local_to_world_rank(SubcontextId{1}, Rank{0});
world->send(buffer, dest, Tag{42});

// Receive from local rank 1 in sub-context 0, using its world rank.
Rank src = world->local_to_world_rank(SubcontextId{0}, Rank{1});
world->recv(buffer, src, Tag{42});
```

### API reference

| Method | Returns | Description |
|---|---|---|
| `get_current_world()` | `const ContextPtr&` | Sub-context communicator (post-split) |
| `get_world_context()` | `ContextPtr` | Unsplit `MPI_COMM_WORLD` handle |
| `subcontext_id()` | `std::optional<SubcontextId>` | This process's sub-context, or `nullopt` |
| `subcontext_count()` | `int` | Number of sub-contexts (1 if unsplit) |
| `subcontext_size(id)` | `Size` | Number of ranks in the given sub-context |
| `subcontext_sizes()` | `Span<const int>` | All sub-context sizes |
| `local_to_world_rank(id, rank)` | `Rank` | Translate (sub-context, local rank) → world rank |
