# Illustration: rank-bindings mapping, one PSD, sub-context at mesh open

**Status**

- **Design** (MPI sub-context split, `inter_context_*` API, merged launch): described below; most of that C++ / `tt-run` work is **not merged yet**.
- **In-tree today (mock / CPU):** two overlay rank-binding YAMLs (optionally **different** `mesh_graph_desc_path` per overlay), a **`subcontext_id_to_rank_bindings` mapping** file, **`tt-run --rank-bindings-mapping`**, and a **mock cluster mapping** (`rank_to_cluster_mock_cluster_desc`, keyed by **MPI world rank**) under `tests/tt_metal/distributed/config/` and `tests/tt_metal/tt_fabric/custom_mock_cluster_descriptors/`. See **Target CLI** and fabric CPU-only workflow for the BH sub-context test line (4×4 vs dual-2×4+intermesh).
- **Not in-tree:** `tests/tt_metal/multihost/single_host_mp_tests/test_sub_context.cpp` and the `inter_context_*` helpers are **plan-only** until implemented; the code blocks below are the target shape.

---

## What you want (one sentence)

Launch **one** MPI job where **PhysicalSystemDescriptor (PSD)** is built **once** for the machine, a **rank-bindings mapping** (like the mock-cluster mapping file, but keyed by **sub-context id**) names **one rank-binding overlay YAML per sub-context**, all on that **same** PSD, and each process—knowing **which host it is on**—opens a **MeshDevice** using the binding / sub-context that owns that rank.

**Fabric:** Rank-binding YAML does **not** carry `FabricConfig` or router settings. **`tt-run` only tags** which overlay a rank belongs to (e.g. `TT_RUN_SUBCONTEXT_ID`). Fabric is configured in C++ via `MetalContext::set_fabric_config()` before opening the mesh device, keyed by `TT_RUN_SUBCONTEXT_ID`.

**Rank semantics:** There is **one** rank in the programming model: **`distributed_context_get_rank()`** (and matching **`rank:`** in the overlay YAML for that process). It is always **`0 … size-1` within that sub-context**, never a linear index across the whole job. **Size** is **`distributed_context_get_size()`** for that same context — equal to the number of ranks in that overlay. Achieve this by initializing the distributed stack on an **MPI sub-communicator** per sub-context (or equivalent), so the old "world-linear" rank is not what the app sees.

---

## Target CLI: `--rank-bindings-mapping` (sub-context id → overlay path)

One `tt-run` invocation passes **`--rank-bindings-mapping`** pointing at a small YAML. The mapping mirrors **`--mock-cluster-rank-binding`**: **keys are indices** and **values are paths**. Here keys are **sub-context id** (`0`, `1`, …); values are **rank-binding overlay YAMLs** for that sub-context.

**Example mapping** (planned path; not required to exist in-tree until someone adds it for CI):

```yaml
# Same spirit as rank_to_cluster_mock_cluster_desc, but sub_context_id -> rank_binding overlay.
# Merge order for global MPI ranks: ascending sub-context id (0, then 1, …); within each file, YAML row order.
subcontext_id_to_rank_bindings:
  0: tests/tt_metal/distributed/config/mock_galaxy_single_host_subcontext_a_rank_bindings.yaml
  1: tests/tt_metal/distributed/config/mock_galaxy_single_host_subcontext_b_rank_bindings.yaml
```

```bash
tt-run \
  --mock-cluster-rank-binding tests/tt_metal/tt_fabric/custom_mock_cluster_descriptors/mock_galaxy_quad_2x4_four_rank_cluster_desc_mapping.yaml \
  --rank-bindings-mapping tests/tt_metal/distributed/config/mock_galaxy_single_host_subcontext_rank_bindings_mapping.yaml \
  --mpi-args "--allow-run-as-root --oversubscribe" \
  ./build/test/tt_metal/distributed/distributed_unit_tests \
  --gtest_filter="MpiSubContext.SingleGalaxySplitContext"
```

The launcher **loads each overlay from the mapping**, assigns **`TT_RUN_SUBCONTEXT_ID`** from the map key, and builds one merged launch table (global rank order = sorted sub-context ids, contiguous ranks per overlay as today). Each overlay may set its own **`mesh_graph_desc_path`**; **`tt-run`** copies that onto each merged rank binding and sets **`TT_MESH_GRAPH_DESC_PATH`** per process accordingly.

- **Each** referenced YAML describes how **context** ranks map to meshes **on the same physical cluster** (same PSD).
- **One** rankfile / one `mpirun` still launches every process, but **after** `init_distributed_context` each process sees **rank and size only for its sub-context** (same integers as `rank:` / count of rows in that overlay).
- **Fabric differs by sub-context in C++:** e.g. sub-context `0` uses `FABRIC_2D`; sub-context `1` uses `FABRIC_2D_PUSH`—both on the **same** PSD. The rank-binding files stay free of fabric fields.

---

## Rank-binding YAML (mesh / devices only — no fabric)

**Checked-in overlays (mock UBB Black Hole galaxy, different mesh graphs per sub-context):**

- `tests/tt_metal/distributed/config/mock_galaxy_single_host_subcontext_a_rank_bindings.yaml` — prefill: **one** logical **4×4** mesh (`bh_galaxy_single_4x4_mesh.textproto`); local ranks **0, 1** share **mesh_id 0** with **mesh_host_rank** **0** / **1**.
- `tests/tt_metal/distributed/config/mock_galaxy_single_host_subcontext_b_rank_bindings.yaml` — decode: **two** 2×4 meshes + inter-mesh link (`bh_galaxy_dual_2x4_intermesh.textproto`); local ranks **0, 1** → **mesh_id** **0** / **1** (unchanged dual-2×4 shape).

**Rank-bindings mapping:** one YAML with **`subcontext_id_to_rank_bindings`** pointing at the two files above (see **Target CLI**). This is analogous to **`rank_to_cluster_mock_cluster_desc`** in the mock-cluster mapping: cluster mapping is **world rank → cluster desc**; rank-bindings mapping is **sub-context id → rank-binding overlay**.

**Mock cluster descriptor mapping (per MPI rank → `bh_6u_cluster_desc.yaml`):**

- `tests/tt_metal/tt_fabric/custom_mock_cluster_descriptors/mock_galaxy_quad_2x4_four_rank_cluster_desc_mapping.yaml`

---

## Mock bring-up (CPU only, today)

This validates **Metal fabric config**, **MPI sub-context** (`TT_RUN_SUBCONTEXT_ID` + `MPI_Comm_split`), **DistributedContext** send/recv, and **inter-context** `MPI_Send`/`MPI_Recv` on the mock BH galaxy with four MPI ranks (prefill overlay: single 4×4 mesh graph; decode overlay: dual 2×4 + intermesh). It does **not** exercise the fabric control plane or `inter_context_*` helpers (those remain plan-only where applicable).

**Test:** `MpiSubContext.SingleGalaxySplitContext` in `tests/tt_metal/distributed/test_mpi_subcontext.cpp`.

**Run (matches fabric CPU-only workflow):**

```bash
tt-run \
  --mock-cluster-rank-binding tests/tt_metal/tt_fabric/custom_mock_cluster_descriptors/mock_galaxy_quad_2x4_four_rank_cluster_desc_mapping.yaml \
  --rank-bindings-mapping tests/tt_metal/distributed/config/mock_galaxy_single_host_subcontext_rank_bindings_mapping.yaml \
  --mpi-args "--allow-run-as-root --oversubscribe" \
  ./build/test/tt_metal/distributed/distributed_unit_tests \
  --gtest_filter="MpiSubContext.SingleGalaxySplitContext"
```

---

## Hypothetical generic pods (illustrative only)

For a **different** cluster, overlays might look like this (paths are placeholders):

**`configs/pod_a_rank_bindings.yaml`**

```yaml
mesh_graph_desc_path: "path/to/mesh_a.textproto"
rank_bindings:
  - rank: 0
    mesh_id: 0
    mesh_host_rank: 0
    env_overrides:
      TT_VISIBLE_DEVICES: "0,1,2,3"
  - rank: 1
    mesh_id: 0
    mesh_host_rank: 1
    env_overrides:
      TT_VISIBLE_DEVICES: "4,5,6,7"
```

**`configs/pod_b_rank_bindings.yaml`**

```yaml
mesh_graph_desc_path: "path/to/mesh_b.textproto"
rank_bindings:
  - rank: 0
    mesh_id: 0
    mesh_host_rank: 0
    env_overrides: { ... }
  # ... more local ranks for B ...
```

**`configs/pods_rank_bindings_mapping.yaml`** (same pattern as mock cluster mapping; keys are sub-context ids)

```yaml
subcontext_id_to_rank_bindings:
  0: configs/pod_a_rank_bindings.yaml
  1: configs/pod_b_rank_bindings.yaml
```

---

## What tt-run would do (conceptual)

1. **Instantiate PSD once**
   Load or discover the cluster once → **one** `PhysicalSystemDescriptor` (or one canonical cluster description) for the run.

2. **Load `--rank-bindings-mapping`**
   Parse **`subcontext_id_to_rank_bindings`**. For each key in **ascending numeric order**, load the referenced rank-binding YAML. Validate that all overlays are **consistent with the same PSD** (hosts, ASICs, no conflicting claims on the same hardware unless explicitly allowed).

3. **Merge into one launch table**
   The rankfile still orders **processes** for `mpirun`, but the **binding row** for each process is chosen so that after **sub-context MPI + TTNN init**, **`distributed_context_get_rank()` equals that row's `rank:`** (and size equals the number of rows in that overlay). For each process record:
   - which **sub-context** it belongs to — the **mapping key** for that overlay (`0`, `1`, …); see env below,
   - `mesh_id`, `mesh_host_rank`, `TT_MESH_GRAPH_DESC_PATH`, `TT_VISIBLE_DEVICES`, mock desc path, etc., **from the row whose `rank:` matches that process's context rank**.

   **No fabric keys** are read from YAML; fabric is configured in C++ downstream.

4. **Set env on each spawned process** (illustrative):

   | Variable | Meaning |
   |----------|--------|
   | `TT_RUN_SUBCONTEXT_ID` | Same as the key in **`subcontext_id_to_rank_bindings`** for that process's overlay — which YAML defines mesh env |
   | `TT_RUN_SUBCONTEXT_SIZE` | Number of ranks in this sub-context |
   | `TT_MESH_ID`, … | From the overlay row where **`rank:` == `distributed_context_get_rank()`** after init |

   **Metal note:** Heterogeneous `FabricConfig` across ranks is only valid if the runtime allows it for your topology (same PSD, different logical fabric mode per sub-context). If not, validation should fail at launch or document the supported matrix.

5. **Single `mpirun`**
   Same as now; only the **pre-launch** configuration is richer.

---

## Context rank (default)

**By default, `distributed_context_get_rank()` and `distributed_context_get_size()` are the only rank and size** the application uses. They match **`rank:`** and the number of **`rank_bindings`** entries **in that process's overlay YAML** — not a merged "world" index across A and B.

Implementation expectation:

1. **`MPI_Comm_split`** by sub-context (e.g. color = `TT_RUN_SUBCONTEXT_ID` or derived from launch metadata).
2. **`init_distributed_context`** (or equivalent) on that **sub-communicator**, not on `MPI_COMM_WORLD`.
3. **`tt-run`** sets **`TT_MESH_*`** / device env from the row with **`rank:` == `distributed_context_get_rank()`** after that init.

Cross-job coordination (if ever needed) is a **separate** path (e.g. **`get_world_context()`** / `MPI_COMM_WORLD` or a side channel); **pipeline and mesh code use context rank only** for intra-context work. Planned **`DistributedContext`** accessors for **world rank** and **sub-context id** are summarized under **Inter-context communication** → **`DistributedContext`: intra- vs inter-context API parity**.

---

## Merged rankfile (global MPI rank vs YAML `rank:`)

Open MPI's **rankfile** always uses **`MPI_COMM_WORLD` rank**: one line per **global** process index `0 … N_total-1` (`rank K=hostname slot=…`). That is **not** the same number as **`rank:`** inside `pod_a` / `pod_b` YAML, which is **context rank** within that overlay only (each pod's YAML still has `0, 1, …` starting over for pod B).

**Rule:** You do **not** duplicate rankfile lines for "pod A rank 0" and "pod B rank 0". You have **one** line per **global** `K`. The launcher keeps a **merge map**:

```text
global_rank  ->  (subcontext_id, yaml_rank / context_rank)
     0       ->  (0, 0)
     1       ->  (0, 1)
     2       ->  (1, 0)
 ...
     7       ->  (1, 5)
```

`tt-run` uses that map to set **`TT_RUN_SUBCONTEXT_ID`** and mesh env from the **correct** YAML row (`row.rank == yaml_rank`).

### How to build `merged_rankfile.txt`

1. **Fix an ordering of sub-contexts.** With **`subcontext_id_to_rank_bindings`**, use **ascending sub-context id** (`0`, then `1`, …). Let `n_A` = number of `rank_bindings` in the overlay for id `0`, `n_B` = for id `1` (names are illustrative). Then `N_total = n_A + n_B`.

2. **Assign contiguous global ranks per pod** (recommended so `MPI_Comm_split` orders subcomm ranks like YAML):
   - Pod A occupies **global** `0 … n_A-1`.
   - Pod B occupies **global** `n_A … n_A+n_B-1`.

3. **Renumber standalone rankfiles:**
   - If you already have `rankfile_A` with lines `rank 0=…`, `rank 1=…` (from an A-only run), those can stay as **global** `0` and `1` when A is first.
   - If you have `rankfile_B` with `rank 0=…` … `rank 5=…` (B-only), **add offset `n_A`** to every rank index:
     - `rank 0=host_b0 …` → `rank n_A+0=host_b0 …`
     - … through `rank n_A+5=host_b5 …`

4. **Concatenate** into one file (or merge and sort by global rank). Each **global** `K` appears **once**; host/slot must match where that process should actually run on the PSD.

**Example** (`n_A=2`, `n_B=6`): globals `0–1` are pod A (unchanged from A-only); globals `2–7` are pod B's local `0–5` with offset `+2`.

```text
rank 0=host-a0 slot=0
rank 1=host-a1 slot=0
rank 2=host-b0 slot=0
rank 3=host-b1 slot=0
rank 4=host-b2 slot=0
rank 5=host-b3 slot=0
rank 6=host-b4 slot=0
rank 7=host-b5 slot=0
```

**`--map-by rankfile:file=merged_rankfile.txt`** (plus your usual MCA args) makes **`OMPI_COMM_WORLD_RANK`** equal that **`K`**. After **`MPI_Comm_split`**, **`MPI_Comm_rank(subcomm)`** is `0…n_A-1` or `0…n_B-1` in ascending **`K`** within each color — which matches **`rank:`** in each YAML if each pod's YAML rows are ordered `0,1,2,…` in the same order as increasing **`K`** within that pod.

### What must match

| Artifact | Namespace |
|----------|-----------|
| Rankfile `rank K=` | **Global** `MPI_COMM_WORLD` rank only |
| YAML `rank:` | **Context** rank within that pod / sub-context |
| Merge map | `K` → which YAML row (subcontext + `rank:`) |

### Pitfalls

- **Overlapping global ranks** if you forget the offset on pod B's rankfile.
- **Non-contiguous global blocks** (interleaving A/B world ranks) — avoid unless you fully control `MPI_Comm_split` **key** so subcomm rank still aligns with YAML `rank:`.
- **Host/slot** lines must reflect **real** placement on the shared PSD; both standalone rankfiles must be consistent with **one** physical layout.

---

## Inter-context communication (C++ MPI host-to-host only)

**No sockets, no Ethernet, no device-to-device.** All inter-context and intra-context communication uses the MPI distributed context `send`/`recv` (host-to-host).

### Intra-context: unchanged from today

`get_current_world()` returns the sub-communicator after auto-split. All existing code using `context->send(buf, Rank{1}, tag)` still uses sub-context-local ranks. `context->barrier()` is a sub-context barrier. **No API changes.**

### Inter-context: new C++ API

For communication **between** sub-contexts, we store the unsplit world communicator and provide explicit free functions.

#### Auto-split in `MPIContext::create`

**File:** `tt_metal/distributed/multihost/mpi_distributed_context.cpp`

Read `TT_RUN_SUBCONTEXT_ID` from the environment. If set, perform `MPI_Comm_split` automatically. Store the **unsplit** world communicator for inter-context use.

```cpp
void MPIContext::create(int argc, char** argv) {
    init_env(argc, argv);
    world_context_ = std::make_shared<MPIContext>(MPI_COMM_WORLD)->duplicate();

    const char* sub_id_str = std::getenv("TT_RUN_SUBCONTEXT_ID");
    if (sub_id_str) {
        int color = std::stoi(sub_id_str);
        int world_rank = std::dynamic_pointer_cast<MPIContext>(world_context_)->rank_;
        current_world_ = world_context_->split(Color(color), Key(world_rank));
    } else {
        current_world_ = world_context_;
    }
}
```

#### New static field + accessor

**New static field** in `MPIContext` (alongside `current_world_`):

```cpp
inline static ContextPtr world_context_;  // unsplit COMM_WORLD dup
```

**New static accessor** on `DistributedContext`:

```cpp
static const ContextPtr& get_world_context();
```

Returns the pre-split world context. When no sub-context split happened, returns the same as `get_current_world()`.

#### `DistributedContext`: intra- vs inter-context API parity (planned)

**Goal:** Anything you need for **intra-context** messaging should have a clear analogue for **inter-context** (world) messaging, without re-reading `TT_RUN_SUBCONTEXT_*` from the environment in application code. **`DistributedContextId id()`** (instance identity for splits/dups) stays unrelated to **sub-context id** (launcher overlay index).

**Two handles, same virtual interface**

Today, **`DistributedContext::get_current_world()`** returns the communicator used for normal TTNN / pipeline work after an optional **`MPI_Comm_split`**. Its **`rank()`**, **`size()`**, **`send` / `recv` / `barrier` / …** all use **sub-context-local** `Rank` / `Size` (aligned with YAML **`rank:`**).

**Planned:** **`DistributedContext::get_world_context()`** returns the **unsplit** world communicator (duplicate of `MPI_COMM_WORLD`). On that handle, the **same** methods mean **world** space:

| Operation | Intra-context (`get_current_world()`) | Inter-context / world (`get_world_context()`) |
|-----------|--------------------------------------|-----------------------------------------------|
| Rank | `rank()` — **local** context rank | `rank()` — **`MPI_COMM_WORLD` rank** (same index as rankfile `K`) |
| Size | `size()` — ranks in this sub-context | `size()` — **total** MPI processes across all sub-contexts |
| Point-to-point | `send` / `recv` / `isend` / `irecv` — `Rank` is **local** | Same methods — `Rank` must be a **world** rank (use translation below) |
| Barrier | `barrier()` — within sub-context only | `barrier()` — **global** (all ranks in `MPI_COMM_WORLD`) |
| Collectives | `broadcast`, `all_gather`, … — **local** team | Same — **world** team (use only when semantics match the job) |

So **raw MPI parity** is indirect but complete: the communicator behind **`get_world_context()`** is the one whose rank matches **`MPI_COMM_WORLD`**; you can pass its underlying `MPI_Comm` to **`MPI_Send` / `MPI_Recv`** if you bypass the virtual API, and **`get_world_context()->rank()`** matches **`MPI_Comm_rank(MPI_COMM_WORLD, …)`**.

**Sub-context metadata (planned additions on `DistributedContext`)**

These describe **which overlay** this process belongs to and how **local ranks map to world ranks**. They should be populated once at **`MPIContext::create`** from launcher env / merged config (same source as **`TT_RUN_SUBCONTEXT_ID`** today), and exposed as **getters** so **`inter_context_*`** and user code do not parse getenv.

| Planned accessor | Meaning |
|------------------|--------|
| **`std::optional<int> subcontext_id()`** | Launcher sub-context key (`0`, `1`, …). **`std::nullopt`** if the job is a normal single-context run (no overlay split). |
| **`int subcontext_count()`** | Number of sub-contexts in the merged job (`N` in **`subcontext_id_to_rank_bindings`**). `1` in a non-split job. |
| **`Size subcontext_size(int subcontext_id)`** | Number of MPI ranks in that sub-context (length of that overlay’s **`rank_bindings`**). Valid for `0 … subcontext_count()-1`. |
| **`std::span<const int> subcontext_sizes()`** (or fixed small buffer) | Read-only view of all **`subcontext_size(i)`** in id order — used to build world-rank translation without N separate calls. |

**Translation (world rank for a logical peer)**

| Planned | Meaning |
|---------|--------|
| **`Rank local_to_world_rank(int subcontext_id, Rank local_rank)`** | With contiguous global blocks in ascending sub-context id (**merge order** in this doc), returns the **world** `Rank` for **`(subcontext_id, local_rank)`**. Implemented from **`subcontext_sizes()`** prefix sums. |

**Optional thin wrappers**

The free functions **`inter_context_send` / `inter_context_recv` / `inter_context_barrier`** remain convenience: internally **`get_world_context()`**, **`local_to_world_rank`**, then **`send` / `recv` / `barrier`** on the world handle (or equivalent **`MPI_*`**).

**C / Python parity (if/when exposed)**

Mirror the same split: **current** context getters stay **local**; add **world** rank/size and **subcontext_id** (and optionally **subcontext_sizes**) next to existing **`distributed_context_*`** entry points so bindings do not duplicate env parsing.

#### Inter-context free functions

**File:** New header `tt_metal/distributed/multihost/inter_context.hpp`

```cpp
namespace tt::tt_metal::distributed::multihost {

Rank local_to_world_rank(int subcontext_id, Rank local_rank);

void inter_context_send(
    ttsl::Span<std::byte> buffer,
    int target_subcontext_id,
    Rank target_local_rank,
    Tag tag);

void inter_context_recv(
    ttsl::Span<std::byte> buffer,
    int source_subcontext_id,
    Rank source_local_rank,
    Tag tag);

void inter_context_barrier();

}  // namespace
```

#### Implementation logic

**Cached launch metadata:** At init, record **`subcontext_count`**, per-id **`subcontext_size`**, and this process’s **`subcontext_id`** from the same source **`tt-run` uses** (env and/or merged YAML). Implement **`subcontext_sizes()`** and **`local_to_world_rank`** from that cache — do not require each rank to know only its own **`TT_RUN_SUBCONTEXT_SIZE`** when translating arbitrary peers.

**`local_to_world_rank`:** Sub-context `0` ranks map to world `[0, n_0)`, sub-context `1` to `[n_0, n_0+n_1)`, etc., with **`n_i = subcontext_size(i)`**.

**`inter_context_send` / `inter_context_recv`:** Obtain **`get_world_context()`**, compute **`dst` / `src = local_to_world_rank(target_or_source_subcontext_id, local_rank)`**, call **`world_ctx->send` / `recv`** with that **`Rank`**, or call **`MPI_Send` / `MPI_Recv`** on the same communicator with the same integer rank.

---

## Full C++ test: pipeline across 4 ranks, 2 sub-contexts (planned)

**Target file:** `tests/tt_metal/multihost/single_host_mp_tests/test_sub_context.cpp` (**not added yet** — pseudocode for the eventual MPI + `inter_context_*` test).

4 MPI processes. Sub-context 0 has ranks 0,1. Sub-context 1 has ranks 0,1. Data flows in a line through all four:

```text
ctx0:rank0 --(intra)--> ctx0:rank1 --(inter)--> ctx1:rank0 --(intra)--> ctx1:rank1
   produce                 relay                    relay                 verify
```

Each step is a blocking MPI `send`/`recv` on the host. Intra-context transfers use `get_current_world()` with sub-context-local ranks. The cross-context transfer uses `inter_context_send`/`inter_context_recv` with the world context.

```cpp
#include <gtest/gtest.h>

#include <cstdlib>
#include <cstring>
#include <vector>

#include "tt-metalium/distributed_context.hpp"
#include "tt_metal/distributed/multihost/inter_context.hpp"

using namespace tt::tt_metal::distributed;
using namespace tt::tt_metal::distributed::multihost;

namespace {

ttsl::Span<std::byte> as_byte_span(std::vector<int>& v) {
    return {reinterpret_cast<std::byte*>(v.data()), v.size() * sizeof(int)};
}

ttsl::Span<const std::byte> as_const_byte_span(const std::vector<int>& v) {
    return {reinterpret_cast<const std::byte*>(v.data()), v.size() * sizeof(int)};
}

}  // namespace

// 4 processes, 2 per sub-context.  Data flows as a pipeline:
//
//   ctx0:rank0 --> ctx0:rank1 --> ctx1:rank0 --> ctx1:rank1
//     produce       relay          relay          verify
//
// Intra-context legs use get_current_world()->send/recv with LOCAL ranks (0, 1).
// The cross-context leg uses inter_context_send/recv.
TEST(SubContextTest, FullPipeline) {
    int sub_id = std::stoi(std::getenv("TT_RUN_SUBCONTEXT_ID"));
    const auto& ctx = DistributedContext::get_current_world();

    ASSERT_EQ(*ctx->size(), 2) << "Each sub-context must have exactly 2 ranks";
    int local_rank = *ctx->rank();

    const std::vector<int> expected = {10, 20, 30, 40, 50};
    const size_t n = expected.size();

    if (sub_id == 0 && local_rank == 0) {
        // ---- ctx0:rank0  (produce) ----
        // Send to ctx0:rank1 via intra-context (local rank 1).
        auto payload = expected;
        ctx->send(as_byte_span(payload), Rank{1}, Tag{0});

    } else if (sub_id == 0 && local_rank == 1) {
        // ---- ctx0:rank1  (relay: intra-recv then inter-send) ----
        std::vector<int> buf(n, 0);
        ctx->recv(as_byte_span(buf), Rank{0}, Tag{0});
        EXPECT_EQ(buf, expected);

        // Forward across sub-contexts to ctx1:rank0.
        inter_context_send(as_byte_span(buf), /*target_subcontext_id=*/1, Rank{0}, Tag{1});

    } else if (sub_id == 1 && local_rank == 0) {
        // ---- ctx1:rank0  (relay: inter-recv then intra-send) ----
        std::vector<int> buf(n, 0);
        inter_context_recv(as_byte_span(buf), /*source_subcontext_id=*/0, Rank{1}, Tag{1});
        EXPECT_EQ(buf, expected);

        // Forward within sub-context to ctx1:rank1.
        ctx->send(as_byte_span(buf), Rank{1}, Tag{2});

    } else if (sub_id == 1 && local_rank == 1) {
        // ---- ctx1:rank1  (verify) ----
        std::vector<int> buf(n, 0);
        ctx->recv(as_byte_span(buf), Rank{0}, Tag{2});
        EXPECT_EQ(buf, expected);
    }

    // Sync everyone before teardown.
    inter_context_barrier();
}

// Verify that ctx->barrier() only synchronizes within a sub-context,
// while inter_context_barrier() synchronizes across both.
TEST(SubContextTest, BarrierScoping) {
    const auto& ctx = DistributedContext::get_current_world();

    // Intra-context barrier — only the 2 local ranks participate.
    ctx->barrier();

    // World barrier — all 4 ranks participate.
    inter_context_barrier();
}
```

### How to run

```bash
mpirun -np 2 -x TT_RUN_SUBCONTEXT_ID=0 -x TT_RUN_SUBCONTEXT_SIZE=2 ./test_sub_context : \
       -np 2 -x TT_RUN_SUBCONTEXT_ID=1 -x TT_RUN_SUBCONTEXT_SIZE=2 ./test_sub_context
```

Or using Open MPI's multi-app syntax with `appfile.txt`:

```
-np 2 -x TT_RUN_SUBCONTEXT_ID=0 -x TT_RUN_SUBCONTEXT_SIZE=2 ./test_sub_context
-np 2 -x TT_RUN_SUBCONTEXT_ID=1 -x TT_RUN_SUBCONTEXT_SIZE=2 ./test_sub_context
```

---

## End-to-end example: disaggregated prefill + decode

This test shows the full picture: two sub-contexts each open a `MeshDevice` with **different fabric configurations**, run compute on-device, and transfer results between sub-contexts via MPI host-to-host.

```text
  Sub-context 0 (prefill, FABRIC_2D)            Sub-context 1 (decode, FABRIC_2D_PUSH)

  rank 0: matmul  --(intra)--> rank 1  --(inter)--> rank 0: eltwise  --(intra)--> rank 1: verify
           compute KV              relay KV             consume KV                  final check
```

4 MPI processes. Each sub-context has ranks 0 and 1. Prefill uses `FABRIC_2D`, decode uses `FABRIC_2D_PUSH`. Data flows as a pipeline through intra-context and inter-context MPI transfers.

```cpp
#include <gtest/gtest.h>

#include <cstdlib>
#include <cstring>
#include <vector>

#include <tt-metalium/distributed.hpp>
#include <tt-metalium/distributed_context.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/system_mesh.hpp>

#include "tt_metal/impl/context/metal_context.hpp"
#include "tt_metal/distributed/multihost/inter_context.hpp"

using namespace tt::tt_metal;
using namespace tt::tt_metal::distributed;
using namespace tt::tt_metal::distributed::multihost;

namespace {

constexpr size_t kKvSize = 1024;

ttsl::Span<std::byte> as_byte_span(std::vector<float>& v) {
    return {reinterpret_cast<std::byte*>(v.data()), v.size() * sizeof(float)};
}

// ---------------------------------------------------------------------------
// Prefill sub-context (FABRIC_2D)
//   rank 0: compute KV cache, send to rank 1 (intra)
//   rank 1: receive KV cache, forward to decode rank 0 (inter)
// ---------------------------------------------------------------------------
void run_prefill() {
    MetalContext::instance().set_fabric_config(
        tt_fabric::FabricConfig::FABRIC_2D,
        tt_fabric::FabricReliabilityMode::STRICT_SYSTEM_HEALTH_SETUP_MODE);
    MetalContext::instance().initialize_fabric_config();

    auto mesh_shape = SystemMesh::instance().shape();
    auto mesh_device = MeshDevice::create(MeshDeviceConfig(mesh_shape));

    const auto& ctx = DistributedContext::get_current_world();
    int local_rank = *ctx->rank();

    std::vector<float> kv_cache(kKvSize, 0.0f);

    if (local_rank == 0) {
        for (size_t i = 0; i < kKvSize; i++) {
            kv_cache[i] = static_cast<float>(i) * 0.01f;
        }
        ctx->send(as_byte_span(kv_cache), Rank{1}, Tag{10});

    } else {
        ctx->recv(as_byte_span(kv_cache), Rank{0}, Tag{10});
        inter_context_send(as_byte_span(kv_cache), /*target_subcontext_id=*/1, Rank{0}, Tag{20});
    }

    mesh_device->close();
}

// ---------------------------------------------------------------------------
// Decode sub-context (FABRIC_2D_PUSH)
//   rank 0: receive KV cache from prefill rank 1 (inter), process, send to rank 1 (intra)
//   rank 1: receive processed result, verify end-to-end correctness
// ---------------------------------------------------------------------------
void run_decode() {
    MetalContext::instance().set_fabric_config(
        tt_fabric::FabricConfig::FABRIC_2D_PUSH,
        tt_fabric::FabricReliabilityMode::STRICT_SYSTEM_HEALTH_SETUP_MODE);
    MetalContext::instance().initialize_fabric_config();

    auto mesh_shape = SystemMesh::instance().shape();
    auto mesh_device = MeshDevice::create(MeshDeviceConfig(mesh_shape));

    const auto& ctx = DistributedContext::get_current_world();
    int local_rank = *ctx->rank();

    std::vector<float> kv_cache(kKvSize, 0.0f);

    if (local_rank == 0) {
        inter_context_recv(as_byte_span(kv_cache), /*source_subcontext_id=*/0, Rank{1}, Tag{20});

        for (auto& v : kv_cache) {
            v += 1.0f;
        }
        ctx->send(as_byte_span(kv_cache), Rank{1}, Tag{30});

    } else {
        ctx->recv(as_byte_span(kv_cache), Rank{0}, Tag{30});

        for (size_t i = 0; i < kKvSize; i++) {
            float expected = static_cast<float>(i) * 0.01f + 1.0f;
            EXPECT_NEAR(kv_cache[i], expected, 1e-5f) << "Mismatch at index " << i;
        }
    }

    mesh_device->close();
}

}  // namespace

// 4 processes, 2 per sub-context.
// Sub-context 0 runs run_prefill() with FABRIC_2D.
// Sub-context 1 runs run_decode()  with FABRIC_2D_PUSH.
//
// Data flow:
//   prefill:rank0 --(intra)--> prefill:rank1 --(inter)--> decode:rank0 --(intra)--> decode:rank1
//      compute KV                 relay KV                  consume KV                verify
TEST(SubContextTest, PrefillDecodeDisaggregated) {
    int sub_id = std::stoi(std::getenv("TT_RUN_SUBCONTEXT_ID"));
    const auto& ctx = DistributedContext::get_current_world();
    ASSERT_EQ(*ctx->size(), 2);

    if (sub_id == 0) {
        run_prefill();
    } else {
        run_decode();
    }

    inter_context_barrier();
}
```

**What this demonstrates:**

1. **Two separate functions** -- `run_prefill()` and `run_decode()` are independent modules, each handling their own sub-context logic. The test dispatches to one based on `TT_RUN_SUBCONTEXT_ID`.
2. **Different fabric configs per sub-context** -- prefill opens with `FABRIC_2D`, decode opens with `FABRIC_2D_PUSH`, both on the same PSD. `MetalContext::instance().set_fabric_config()` is called in C++ before `MeshDevice::create()`.
3. **Intra-context transfer** -- prefill rank 0 sends to prefill rank 1 using `ctx->send(buf, Rank{1}, tag)` with local ranks 0 and 1. Same pattern in decode.
4. **Inter-context transfer** -- prefill rank 1 sends KV cache to decode rank 0 using `inter_context_send()` / `inter_context_recv()` over the unsplit world context.
5. **End-to-end verification** -- decode rank 1 checks that the data made it through the entire pipeline with the expected values.

---

## tt-run changes

**File:** `ttnn/ttnn/distributed/ttrun.py`

1. Add **`--rank-bindings-mapping`** (`Path`): YAML containing **`subcontext_id_to_rank_bindings`**, each value a path to a rank-binding overlay (same schema as today's single `--rank-binding` file).
2. Add **`parse_rank_bindings_mapping(mapping_path)`** (name illustrative):
   - Load the mapping; sort keys numerically (`0`, `1`, …).
   - For each key `id`, load that overlay; `n_id = len(rank_bindings)`.
   - Build one merged `TTRunConfig` with `sum n_id` rows: sub-contexts are concatenated in key order; **global** MPI ranks are contiguous blocks `0..n_0-1`, `n_0..n_0+n_1-1`, …
   - Tag each row with **`subcontext_id`** = map key and **`subcontext_size`** = `n_id` for that overlay.
3. In `get_rank_environment`, when overlay metadata exists, inject:
   - `TT_RUN_SUBCONTEXT_ID` = string form of the map key
   - `TT_RUN_SUBCONTEXT_SIZE` = `str(n_id)` for that overlay
   - Per-row mesh env from the overlay row whose **`rank:`** matches context rank
4. **Interaction with `--rank-binding`:** either accept **only** the mapping for multi-overlay jobs, or define explicit precedence; single-overlay runs can keep one `--rank-binding` with no mapping.

---

## Files: landed vs planned

**In repository today (mock / docs / CI hook):**

| File | Purpose |
|------|--------|
| `tests/tt_metal/distributed/config/mock_galaxy_single_host_subcontext_a_rank_bindings.yaml` | Sub-context **0**: **4×4** mesh graph, two ranks on mesh_id **0** |
| `tests/tt_metal/distributed/config/mock_galaxy_single_host_subcontext_b_rank_bindings.yaml` | Sub-context **1**: dual **2×4** + intermesh, mesh_id **0** / **1** |
| `tests/tt_metal/distributed/config/mock_galaxy_single_host_subcontext_rank_bindings_mapping.yaml` | `subcontext_id_to_rank_bindings` (overlays may use different `mesh_graph_desc_path`) |
| `tests/tt_metal/tt_fabric/custom_mesh_descriptors/bh_galaxy_single_4x4_mesh.textproto` | One **4×4** BH mesh (prefill / “big mesh” overlay) |
| `tests/tt_metal/tt_fabric/custom_mesh_descriptors/bh_galaxy_dual_2x4_intermesh.textproto` | Two **2×4** BH meshes + one inter-mesh connection (decode overlay) |
| `tests/tt_metal/tt_fabric/custom_mock_cluster_descriptors/mock_galaxy_quad_2x4_four_rank_cluster_desc_mapping.yaml` | Mock cluster mapping → `bh_6u_cluster_desc.yaml` (ranks 0–3) |
| `tests/tt_metal/distributed/test_mpi_subcontext.cpp` | `MpiSubContext.SingleGalaxySplitContext` |
| `.github/workflows/fabric-cpu-only-tests-impl.yaml` | Fabric CPU-only workflow (`--rank-bindings-mapping` for quad-2×4 sub-context test) |

**Planned (not necessarily present yet):**

| File | Action |
|------|--------|
| `ttnn/ttnn/distributed/ttrun.py` | **`--rank-bindings-mapping` landed**; C++ `get_world_context()` / `inter_context_*` still planned |
| `tt_metal/distributed/multihost/mpi_distributed_context.cpp` | Auto-split in `create()`, store `world_context_` |
| `tt_metal/distributed/multihost/mpi_distributed_context.hpp` | `world_context_` static field |
| `tt_metal/api/tt-metalium/distributed_context.hpp` | `get_world_context()`; **`subcontext_id()`**, **`subcontext_count()`**, **`subcontext_size(id)`**, **`subcontext_sizes()`**; **`local_to_world_rank(subcontext_id, local_rank)`** (names illustrative) |
| `tt_metal/distributed/multihost/distributed_context.cpp` | Dispatch new static/instance methods; cache merged launch metadata at init |
| `tt_metal/distributed/multihost/inter_context.hpp` / `.cpp` | `inter_context_send` / `recv` / `barrier` |
| `tests/tt_metal/multihost/single_host_mp_tests/test_sub_context.cpp` | Full pipeline + prefill/decode style tests |

---

## Tiny picture

```text
+------------------------------------------+
|     One PSD (one physical cluster)       |
+------------------------------------------+
                      |
        rank_bindings_mapping.yaml
        subcontext_id_to_rank_bindings
           0 -> overlay_a.yaml
           1 -> overlay_b.yaml
        (mesh / devices only per file)
                      |
        +-------------------------------+
        | one mpirun launches all procs |
        | sub-context 0: rank 0, 1      |
        | sub-context 1: rank 0, 1      |
        +-------------------------------+
                      |
        MPIContext::create reads TT_RUN_SUBCONTEXT_ID -> auto MPI_Comm_split;
        rank == YAML "rank:" field; C++ sets FabricConfig; open_mesh_device
```
