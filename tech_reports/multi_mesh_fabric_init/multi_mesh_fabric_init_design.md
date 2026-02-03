# Multi-Mesh Multi-Host Fabric Initialization Design

## 1. Executive Summary

This document outlines the design and transition plan for the Multi-Mesh Multi-Host Fabric initialization flow. The goal is to move from a manual, hardcoded rank-binding approach to a fully automated, Fabric Manager (FM) driven topology resolution and initialization system.

The transition involves introducing a global Fabric Manager that maintains the physical cluster graph, creating a Topology Mapper to resolve logical Mesh Graph Descriptors (MGD) to physical resources, and updating `tt-run` and the Metal runtime to utilize these services.

The design is explicitly tied to the following efforts:

*   **Multi-mesh topology solver** (PR #36174): [https://github.com/tenstorrent/tt-metal/pull/36174](https://github.com/tenstorrent/tt-metal/pull/36174)
*   **Auto assign Mesh Rank Bindings from Rankfile** (Issue #31613): [https://github.com/tenstorrent/tt-metal/issues/31613](https://github.com/tenstorrent/tt-metal/issues/31613)
*   **Auto assign Mesh ID via Topology Mapper** (Issue #31745): [https://github.com/tenstorrent/tt-metal/issues/31745](https://github.com/tenstorrent/tt-metal/issues/31745)
*   **Standardize file for Physical System Groupings** (Issue #35708): [https://github.com/tenstorrent/tt-metal/issues/35708](https://github.com/tenstorrent/tt-metal/issues/35708)

---

## 2. Physical Groupings and Descriptors

To enable the Fabric Manager to understand the cluster hierarchy and make intelligent placement decisions, we introduce standardized descriptors for physical system groupings.

### System Hierarchy
*   **Chip:** Single Tenstorrent device.
*   **Board/Shelf:** Group of chips (e.g., Galaxy, Nebula).
*   **Host:** Server node controlling one or more boards.
*   **Pod:** Collection of connected hosts/boards.
*   **Cluster:** The entire set of available resources.

### Descriptors
We define a schema for describing these resources.

**Mesh Descriptor Example:**
```protobuf
mesh_descriptors {
  name: "M0"
  arch: WORMHOLE_B0
  device_topology {
    dims: [ 8, 16 ]
    dim_types: [ LINE, RING ]
  }
  host_topology   { dims: [1, 4] }
  channels {
    count: 4
    policy: STRICT
  }
}
```

**Graph Descriptor Example:**
```protobuf
graph_descriptors {
  name: "G0"
  type: "POD"
  instances { mesh { mesh_descriptor: "M0" mesh_id: 0 } }
  instances { mesh { mesh_descriptor: "M0" mesh_id: 1 } }
  connections {
    nodes { mesh { mesh_descriptor: "M0" mesh_id: 0 } }
    nodes { mesh { mesh_descriptor: "M0" mesh_id: 1 } }
    channels { count: 2 }
  }
}
```

---

## 3. Current Flow (Pre-PR 36174: Manual Rank Bindings)

Currently, the user is responsible for manually defining the mapping between MPI ranks and physical devices via a rank binding file. There is no global validation of inter-mesh connectivity by a central manager before runtime.

### 3.1 Job Submission & Launch
The user requests nodes via SLURM and then uses `tt-run` with a manually crafted rank binding file.

**Step 1: Allocation**
```bash
# Example: Allocating 4 nodes for a job
salloc --partition=galaxy --nodes=4
```

**Step 2: Configuration (Rank Bindings)**
The user must create a `rank_bindings.yaml` that maps each MPI rank to a `mesh_id`, `mesh_host_rank`, and the specific `TT_VISIBLE_DEVICES`.

*Example `rank_bindings.yaml`:*
```yaml
rank_bindings:
  - rank: 0
    mesh_id: 0
    mesh_host_rank: 0
    env_overrides:
      TT_VISIBLE_DEVICES: "0,1,2,3,4,5,6,7"

  - rank: 1
    mesh_id: 1
    mesh_host_rank: 0
    env_overrides:
      TT_VISIBLE_DEVICES: "0,1,2,3,4,5,6,7"

mesh_graph_desc_path: "path/to/my_mesh_graph_descriptor.textproto"
```

**Step 3: Execution**
The user runs `tt-run`, passing the rank binding file and any necessary MPI arguments (like a rankfile if specific host placement is needed).

```bash
# Example launch script
RANK_BINDING=rank_bindings.yaml
RANKFILE=hosts_rankfile.txt
HOSTS="host0,host1,host2,host3"

MPI_ARGS="--host ${HOSTS} --rankfile ${RANKFILE} --oversubscribe --mca btl self,tcp --mca btl_tcp_if_exclude docker0,lo,tailscale0 --bind-to none --tag-output"

tt-run \
  --rank-binding "${RANK_BINDING}" \
  --mpi-args "${MPI_ARGS}" \
  ./bh_blitz_decode
```

### 3.2 Flow Diagram

![Current Flow Diagram](diagram1_current_flow.png)

### 3.3 Explanation
1.  **Allocation:** User requests nodes via SLURM (`salloc` or `srun`).
2.  **Configuration:** User creates a `rank_bindings.yaml` that maps MPI ranks to specific `mesh_id`s and sets `TT_VISIBLE_DEVICES`.
3.  **Execution:** User runs `tt-run`.
    *   `tt-run --rank-binding rank_bindings.yaml ./build/test/tt_metal/my_app`
4.  **Initialization:** `tt-run` launches processes. Each process initializes the Metal backend. Rank bindings determine the pipeline.

---

## 4. After PR 36174: Hierarchical Multi‑Mesh Mapping in CP

PR 36174 adds a **multi‑mesh topology mapping utility** inside TT‑Metal. This doesn’t change the end‑to‑end flow yet, but it **changes what CP can do** when it sees an MGD. This is the intermediate "Transition" step before full FM integration.

### 4.1 New Mapping Model

New data structures are introduced:
*   **LogicalMultiMeshGraph**: Mesh‑level adjacency graph (logical meshes and their connections).
*   **PhysicalMultiMeshGraph**: Mesh‑level adjacency graph for physical meshes and their ASIC connectivity.

The main utility:
```cpp
MultiMeshMapping map_multi_mesh_to_physical(
    const LogicalMultiMeshGraph& logical,
    const PhysicalMultiMeshGraph& physical,
    const TopologyMappingConfig& config);
```

Internally, it performs:
1.  **Inter‑mesh mapping**: Map logical meshes → physical meshes using adjacency and **inter‑mesh validation** (STRICT/RELAXED).
2.  **Intra‑mesh mapping**: Map logical fabric nodes → ASIC IDs within each mesh.

The solver is wired into `TopologyMapper::map_multi_mesh_to_physical()`. It is effectively the engine we will rely on for **Phase 1 and Phase 2** inside FM and CP.

### 4.2 Temporary Constraint

PR 36174 currently enforces a **1:1 mesh ID mapping** (physical mesh ID = logical mesh ID), as a temporary constraint until the **rank bindings file is removed** (Issues 31613/31745).

*   **Current/Transition Impact:** Users still provide rank bindings manually.
*   **Improvement:** The Control Plane (CP) now has the logic to validate that the physical connectivity matches the logical MGD, rather than blindly trusting the rank bindings.

---

## 5. Phase 1 Flow (Transitional: FM-Assisted)

In this phase, the Fabric Manager (FM) is introduced to automate the topology mapping using the logic from PR 36174. FM resolves the logical MGD to the physical graph and generates the rank bindings. `tt-run` still consumes these bindings, maintaining compatibility with the current launch mechanism.

### 5.1 Automated Binding Generation
Instead of writing `rank_bindings.yaml` manually, the user submits their intent (MGD) to SLURM/FM, which produces the bindings.

**Step 1: Allocation & Binding Generation**
```bash
# User submits job with MGD
srun --partition=galaxy --comment="mgd=my_graph.proto" ...
```
Behind the scenes, the SLURM plugin calls FM. FM finds a valid placement and writes a `generated_rank_bindings.yaml`.

*Example Generated `rank_bindings.yaml`:*
```yaml
# AUTO-GENERATED BY FABRIC MANAGER
rank_bindings:
  - rank: 0
    mesh_id: 0         # Mapped from Logical M0 -> Physical M7
    mesh_host_rank: 0
    env_overrides:
      TT_VISIBLE_DEVICES: "0,1,2,3,4,5,6,7"

  - rank: 1
    mesh_id: 1         # Mapped from Logical M1 -> Physical M8
    mesh_host_rank: 0
    env_overrides:
      TT_VISIBLE_DEVICES: "0,1,2,3,4,5,6,7"

mesh_graph_desc_path: "path/to/my_mesh_graph_descriptor.textproto"
```

**Step 2: Execution**
The user (or the wrapper script) uses the generated file but **must still provide all MPI launch arguments** (rankfile, host list, etc.).

**Important:** The `tt-run` CLI interface is **unchanged** in Phase 1. The only difference is that the input files (`rank_bindings.yaml`, `rankfile.txt`) are generated rather than handwritten.

```bash
# Retrieve artifacts generated by FM/SLURM
RANK_BINDING="/data/slurm/$JOBID/generated_rank_bindings.yaml"
RANKFILE="/data/slurm/$JOBID/rankfile.txt"
HOSTS=$(scontrol show hostnames $SLURM_JOB_NODELIST | paste -sd,)

MPI_ARGS="--host ${HOSTS} --rankfile ${RANKFILE} --oversubscribe --bind-to none"

tt-run \
  --rank-binding "${RANK_BINDING}" \
  --mpi-args "${MPI_ARGS}" \
  ./bh_blitz_decode
```

### 5.2 Flow Diagram

![Phase 1 Flow Diagram](diagram2_phase1_flow.png)

### 5.3 Explanation
1.  **FM Initialization:** FM runs as a persistent daemon and discovers the whole cluster's physical graph.
2.  **Allocation Request:** User submits a job with a Mesh Graph Descriptor (MGD).
3.  **Topology Resolution:** SLURM plugin queries FM. FM uses the **Topology Mapper** (from PR 36174) to find a valid placement for the MGD.
4.  **Binding Generation:** FM returns the allocation. This allocation is used to auto-generate a `rank_bindings.yaml` (removing manual error).
5.  **Execution:** `tt-run` is invoked with the generated bindings.
6.  **Runtime:** Metal initializes. The Control Plane uses the MGD and bindings. It performs validation using the PR 36174 mapper logic locally, while inter-mesh connections are dictated by the valid rank bindings provided by FM.

---

## 6. Phase 2 Flow (Ideal: Fully Automated)

In the final state, FM handles the entire fabric initialization. `tt-run` connects to FM directly. Rank bindings and device visibility are handled transparently. The user experience is simplified to requesting resources and running the application.

### 6.1 Job Submission & Launch
The workflow is simplified to just the essential inputs: topology (MGD) and application binary.

**Step 1: Allocation & Placement**
```bash
# Request resources for the topology
srun --partition galaxy --comment="mgd=my_graph.proto" --nodes=4 ...
```
1.  **Placement:** SLURM consults FM to find a valid placement for the requested topology (MGD).
2.  **Fabric Initialization:** Once placement is determined, FM performs the full fabric initialization *before* the job starts. This uses the MGD and the internal visibility map. FM may trigger an internal MPI process ("under the hood") running Control Plane logic to physically program the routing tables and binaries onto the devices.

**Step 2: Execution**
No rank bindings file is needed.
```bash
tt-run --fabric-manager ./bh_blitz_decode
```

### 6.2 Flow Diagram

![Phase 2 Flow Diagram](diagram3_phase2_flow.png)

### 6.3 Explanation

#### 1. Allocation & Placement (SLURM + Topology Mapper)
The user initiates the job using SLURM.
```bash
# Request resources for the topology
srun --partition galaxy --comment="mgd=my_graph.proto" --nodes=4 ...
```
SLURM consults the Fabric Manager. The FM uses the **Topology Mapper** to find a valid placement for the requested MGD on the physical cluster. **Note:** The user does not create any rank binding files; the configuration is implicit.

#### 2. Fabric Initialization (Fabric Manager)
Once the allocation is determined, but *before* the user's application runs, the Fabric Manager takes control.
*   **Binding Generation:** FM internally generates the rank bindings and determines the correct `TT_VISIBLE_DEVICES` for each rank based on the topology mapping.
*   **Device Programming:** FM initializes the fabric on the allocated chips. It compiles and uploads router binaries and configures routing tables. This step may utilize a "Control Plane" process running via MPI "under the hood" to handle the low-level device interactions.
*   **Readiness:** The fabric is now fully configured and ready for traffic.

#### 3. Execution (User Application)
The user launches the application.
```bash
tt-run --fabric-manager ./bh_blitz_decode
```

When the application starts, it seamlessly picks up the configuration from Step 2:
1.  **Context Init:** The Control Plane (CP) inside the user app connects to FM using the session ID.
2.  **Handshake:** CP sends its **MPI Rank** to FM.
3.  **Resolution:** FM returns the pre-calculated device mapping (ASIC IDs) from Step 2.
4.  **Visibility:** The CP internally configures `TT_VISIBLE_DEVICES` (or equivalent) so the process only sees its assigned devices.
5.  **Proceed:** Since the fabric was initialized in Step 2, the CP simply verifies the state and the application proceeds.
