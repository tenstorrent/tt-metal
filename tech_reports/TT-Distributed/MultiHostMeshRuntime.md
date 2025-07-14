# TT-Distributed: Multi-Host Runtime

Date: April 22, 2025
Last updated: July 10, 2025

**Goal and Scope:**

This proposal defines how multi-device native TT-Metalium concepts (e.g., `MeshDevice`, `MeshWorkload`, `MeshBuffer`) can be extended to become natively **multi-host** aware for **Single Program, Multiple Data (SPMD)** execution models.

**The specific focus of this document is:**

*   Managing a **single, large, uniform logical 2D/3D mesh** (potentially a torus) distributed across a potentially large number of hosts.
*   Presenting this distributed system to the user as a **single global view** for both the underlying TT-fabric connectivity and for workload definition (`MeshWorkload`, `MeshBuffer`).
*   **Abstracting away the multi-host implementation details** (like mesh partitioning and process coordination) from the user's application code, which interacts primarily with the global mesh interface.
*   Targeting workloads amenable to SPMD execution, particularly those leveraging **Tensor Parallelism (TP)** and **Data Parallelism (DP)** across the mesh.
*   Providing a clear path to **scale up** these TP/DP workloads from single-host systems (e.g., an 8x4 mesh on one Galaxy host) to large multi-host configurations (e.g., spanning 16 Galaxies or more), enabling models like Llama to scale from ~70B to 2T+ parameters and sequence lengths to 10M tokens primarily via TP, for which the uniform mesh design and SPMD execution model is a great fit.

**Out of Scope for this Document:**

*   **Multi-Mesh Scenarios:** Managing multiple distinct logical meshes is not covered here.
*   **MPMD Execution Models:** Multiple Program, Multiple Data execution, where different ranks might run different host programs, is not addressed by this SPMD design.
*   **Pipeline/Model Parallelism:** Workloads requiring more complex graph-level parallelism (like pipeline parallelism or general model parallelism) often necessitate MPMD execution models and potentially multi-mesh capabilities. These topics will be handled in a separate proposal/document.

This proposal aims to extend the single-host, multi-device programming model to multi-host for SPMD workloads, maintaining a unified global view of the mesh and fabric while hiding the distribution complexity.

## Table of Contents

- [Design Philosophy & Rationale](#design-philosophy--rationale)
  - [Core Principle: Global Definition vs. Local Execution](#core-principle-global-definition-vs-local-execution)
  - [SPMD Execution Model](#spmd-execution-model)
  - [Coordination vs. Data Movement](#coordination-vs-data-movement)
  - [Global View and Local Ownership](#global-view-and-local-ownership)
  - [Underlying Fabric Assumption](#underlying-fabric-assumption)
  - [Determinism](#determinism)
- [Proposed Design Multi-Host, Multi-Device (SPMD / Multiple Lockstep Controllers)](#proposed-design-multi-host-multi-device-spmd--multiple-lockstep-controllers)
  - [Visualization](#visualization-4-host-example-16x8-mesh--8x4-sub-meshes)
  - [Comparison with Other Architectures](#comparison-with-other-architectures)
    - [1. Single-Host, Single-Device](#1-single-host-single-device)
    - [2. Single-Host, Multi-Device (e.g., Galaxy)](#2-single-host-multi-device-eg-galaxy)
    - [3. Multi-Host, Multi-Device (Single Controller, Multiple Executors)](#3-multi-host-multi-device-single-controller-multiple-executors)
- [Host Coordination Dependency](#host-coordination-dependency)

## Design Philosophy & Rationale

![image](https://github.com/user-attachments/assets/40aba6fa-b170-49ef-91a4-6cff1e4a81ea)

### Core Principle: Global Definition vs. Local Execution

The core design principle is to **separate the global, host-symmetric definition of work from the local, host-specific execution.** All participating host processes first agree on *what* needs to be done across the entire logical mesh before each host determines *how* to execute its portion of that work on the devices it locally manages.

### SPMD Execution Model

We adopt a Single Program, Multiple Data (SPMD) execution model. Crucially, this means the user's application itself runs as **multiple processes** (typically one MPI rank per host), a shift from typical single-process host applications. These processes execute the same program code.

This model requires **lockstep behavior** during the workload definition phase: the sequence of operations defining the global view (like `MeshBuffer` allocation and `MeshWorkload` creation) must be identical across all participating processes.

### Coordination vs. Data Movement

A host-to-host coordination mechanism is needed to maintain the SPMD lockstep and perform optional validation checks. As a proof-of-concept, **MPI** can be used for this coordination layer.

It is critical to understand that the coordination layer is **not** intended for bulk data movement between host processes or between hosts and devices. All substantial data movement, including device-to-device transfers and collective communications across the logical mesh, is assumed to occur over the **underlying TT-fabric**.

### Global View and Local Ownership

*   **Global View (All Processes):** Users interact with a virtual `MeshDevice` representing the entire logical mesh. `MeshBuffer` allocation and `MeshWorkload` creation are performed identically and deterministically across all participating processes. This ensures every process has a consistent, global view of the resources and the intended computation, simplifying workload design and debugging.
*   **Local Ownership (Per Process):** Each host process (rank) physically manages a specific sub-mesh region of the overall logical mesh. Internally, the `MeshDevice` on a given host owns and manages a collection of `Device` objects representing the mesh nodes within its assigned sub-mesh. Each local `Device` has its own `CommandQueue`.
*   **Local Dispatch Abstraction:** The host-local `MeshCommandQueue` serves as the interface for submitting the globally defined `MeshWorkload`. Internally, it filters and dispatches the relevant commands from the global workload to the appropriate local `CommandQueue`s managed by that host process.

### Underlying Fabric Assumption

This proposal assumes that the devices comprising the logical mesh are connected via a **single, unified fabric**. Direct device-to-device communication within the mesh occurs over this fabric without host intervention.

### Determinism

**Crucially, Determinism is Required:** As mandated by the SPMD model for the definition phase, both the runtime itself *and* the user's application code (including workload generation functions) **must be deterministic**. This means using deterministic algorithms and data structures (e.g., avoiding hash maps with non-deterministic iteration order if the order affects workload generation). If any host process diverges due to non-determinism, the system's behavior becomes undefined.

Python bindings (e.g., pybind11) require additional care regarding determinism:

1.  **Python Garbage Collection (GC) and Resource Deallocation:**
    *   **Problem:** Standard Python GC timing is non-deterministic across different processes (ranks). If C++ resource deallocation (like freeing a `MeshBuffer`) directly modifies allocator state (which is necessary for correct subsequent allocations) and is tied *only* to the Python object's destruction (e.g., via `__del__`), this critical state change will occur at different times on different ranks, leading to divergence.
    *   **Solution:** The C++ resource lifetime management **must be decoupled** from Python's GC timing. The Python bindings **must provide explicit, deterministic mechanisms** for resource deallocation. Users **must** use these explicit mechanisms. The strongly recommended approach is to use **context managers (`with` statement)** for resources like `MeshBuffer`. This ensures deallocation happens deterministically upon exiting the `with` block scope, synchronized across all ranks.
        ```python
        # Example of deterministic deallocation with context manager
        with device.allocate_buffer(...) as buf:
            # ... use buf ...
        # <-- buf is deterministically deallocated here on all ranks
        ```
    *   An alternative might be an explicit `buffer.free()` method, but it must be called at the exact same logical point in the code by all ranks.

2.  **Python Hash Randomization:**
    *   **Problem:** Since Python 3.3, dictionary and set iteration order is randomized by default across different process invocations. If user logic iterates over these collections and the order affects workload generation (e.g., determines the order of operations), the generated `MeshWorkload` will differ between ranks, breaking the SPMD model.
    *   **Recommendation:** User code **must not** rely on unordered collection iteration if it impacts workload definition. Either use ordered collections (e.g., `collections.OrderedDict`, lists) or ensure the logic is insensitive to iteration order. Alternatively, setting the `PYTHONHASHSEED=0` environment variable for all ranks can enforce deterministic iteration, but makes the application reliant on this external setting.

3.  **Other Sources:** Standard pitfalls like rank-dependent branching logic that affects workload definition, uncoordinated I/O, or multi-threading races within a rank's definition phase must also be avoided.

**Layered Architecture per Host:** The separation between global definition and local execution can be visualized as a layered stack on each host process:

```
+----------------------------------------------------+
|          Global View (Identical on all Hosts)      |
|----------------------------------------------------|
| - MeshDevice (Full Logical Mesh)                   |
| - MeshBuffer (Global Resource Spec)                |
| - MeshWorkload (Global Computation Spec)           |
+----------------------- | --------------------------+
                         |
                         | Submission Interface
                         V
+----------------------- - --------------------------+
|                MeshCQ (Global Input)               |
|----------------------------------------------------|
|          Local Dispatch Logic (Host-Specific)      |
|            (Maps Global Work to Local CQs)         |
|----------------------- | --------------------------+
                         |
                         V Local CQs
+----------------------- - --------------------------+
|          Local View (Host-Specific Subset)         |
|----------------------------------------------------|
| - Set of local Device objects                      |
| - Associated local CommandQueue per Device         |
+----------------------------------------------------+
```

This architecture ensures that all host processes agree on the *what* (the global `MeshWorkload` submitted to `MeshCommandQueue`) before diverging into the *how* (the host-specific dispatch to local device `CommandQueue`s).

**Single-Rank Debugging:** A significant advantage of the proposed design is that because the user's application code interacts primarily with the global view objects (`MeshDevice`, `MeshBuffer`, `MeshWorkload`) and this view is identical across all ranks/processes up to the `MeshCommandQueue` submission point, much of the application logic can often be debugged effectively by running with a single rank/process (`mpirun -np 1 ...`). This drastically simplifies the debugging process, especially for systems with a large number of hosts.

In essence, the user defines *what* computation should happen on the *entire mesh* (global view), and the runtime handles *how* to distribute and execute that computation on the *local devices* managed by each host process.

## **Proposed Design:** Multi-Host, Multi-Device (SPMD / Multiple Lockstep Controllers)

```ascii
      +---------------------------------------------------+
      |                Host Coordination                  |
      |               (All-to-All/Global)                 |
      +---|-------------------|---------------------|-----+
          ^                   ^                     ^
          |                   |                     |
          V                   V                     V
+-------------------+  +-------------------+  +-------------------+
| Host 1 (Rank 0)   |  | Host 2 (Rank 1)   |  | Host N (Rank N-1) |
| Process (SPMD)    |  | Process (SPMD)    |  | Process (SPMD)    |
|-------------------|  |-------------------|  |-------------------|
| User Code (Same)  |  | User Code (Same)  |  | User Code (Same)  |
| MeshDevice(Global)|  | MeshDevice(Global)|  | MeshDevice(Global)|
| MeshWorkload(Same)|  | MeshWorkload(Same)|  | MeshWorkload(Same)|
|     |             |  |     |             |  |     |             |
|   MeshCQ          |  |   MeshCQ          |  |   MeshCQ          |
|     | (Dispatch)  |  |     | (Dispatch)  |  |     | (Dispatch)  |
|     V   Local     |  |     V   Local     |  |     V   Local     |
|   +-------+       |  |   +-------+       |  |   +-------+       |
|   | DevCQ | ...   |  |   | DevCQ | ...   |  |   | DevCQ | ...   |
|   +-------+       |  |   +-------+       |  |   +-------+       |
+-------------------+  +-------------------+  +-------------------+
```

### Visualization (4-Host Example: 16x8 Mesh / 8x4 Sub-Meshes)

Imagine a 16x8 logical mesh divided among 4 hosts (ranks 0-3), where each host manages an 8x4 sub-mesh:

```ascii
+------------------------------------------------------------+
|            Logical Mesh (16x8 Global)                      |
|                                                            |
|       +---------------------+----------------------+       |
|       | Host 0 (Rank 0)     | Host 1 (Rank 1)      |       |    Global View (All Hosts):
|       | Owns 8x4 Sub-Mesh   | Owns 8x4 Sub-Mesh    |       |    - MeshDevice (16x8)
|       | Global Coords:      | Global Coords:       |       |    - MeshBuffer (Global)
|       | x=[0..8), y=[0..4)  | x=[8..16), y=[0..4)  |       |    - MeshWorkload (Global)
|       | (Local Devs+CQs)    | (Local Devs+CQs)     |       |    - MeshCommandQueue (Submit Global)
|       +---------------------+----------------------+       |
|       | Host 2 (Rank 2)     | Host 3 (Rank 3)      |       |
|       | Owns 8x4 Sub-Mesh   | Owns 8x4 Sub-Mesh    |       |    Local Ownership (per Host):
|       | Global Coords:      | Global Coords:       |       |    - 8x4 set of Devices
|       | x=[0..8), y=[4..8)  | x=[8..16), y=[4..8)  |       |    - 32 DeviceCQs
|       | (Local Devs+CQs)    | (Local Devs+CQs)     |       |    - dispatch_pending
|       +---------------------+----------------------+       |
|                                                            |
+------------------------------------------------------------+

- All Hosts (Ranks 0-3) create the same GLOBAL specifications
  for Buffers & Workloads relative to the 16x8 Logical Mesh.

- Each Host only DISPATCHES commands to the 32 LOCAL Devices
  (and their DeviceCQs) within its owned 8x4 Sub-Mesh region.

```

### Comparison with Other Architectures

To better understand the proposed design (SPMD / Multiple Lockstep Controllers), it's helpful to contrast it with other common system architectures:

#### 1. Single-Host, Single-Device

*   **Execution:** User code runs as a single process.
*   **Scope:** `MeshDevice` represents a 1x1 mesh. The `MeshCommandQueue` maps directly (1:1) to the single underlying device `CommandQueue`.
*   **Relevance:** Simplest baseline, not applicable to multi-device or multi-host scenarios.

```ascii
+-------------+
| Host        |
| Process (1) |
|-------------|
| User Code   |
| MeshDevice  |
|  (1x1)      |
|   |         |
| MeshCQ      |
|   | (1:1)   |
|   V         |
| DeviceCQ    |
+-------------+
```

#### 2. Single-Host, Multi-Device (e.g., Galaxy)

*   **Execution:** User code runs as a single process.
*   **Scope:** `MeshDevice` represents the multi-device mesh (e.g., 8x4). `MeshBuffer` and `MeshWorkload` are defined globally for this mesh.
*   **Dispatch:** The single `MeshCommandQueue` receives the global workload and internally dispatches relevant commands to the multiple `CommandQueue`s corresponding to the devices managed by the host. This dispatch might be multi-threaded for efficiency.
*   **Relevance:** Represents the current standard for multi-device systems *within* a single host. The proposed multi-host design leverages similar concepts for the *local* dispatch part on each host.

```ascii
+--------------------+
| Host               |
| Process (1)        |
|--------------------|
| User Code          |
| MeshDevice (NxM)   |
|      |             |
|    MeshCQ          |
|      | (Dispatch)  |
|  +---V---+---V---+ |
|  | DevCQ | DevCQ | |
|  +-------+-------+ |
|    ... (NxM) ...   |
+--------------------+
```

#### 3. Multi-Host, Multi-Device (Single Controller, Multiple Executors)

This is a common alternative approach to managing multi-host systems:

*   **Execution:** User code runs as a **single process** on a designated controller host (or potentially a separate dedicated host).
*   **Scope:** The Controller process holds the global view (`MeshDevice`, `MeshWorkload`).
*   **Dispatch:** The Controller process serializes commands derived from the `MeshWorkload` and sends them over the network (e.g., using RPC or a message queue) to Executor processes running on the other hosts. Each Executor process manages its local sub-mesh devices and has a form of "Remote Mesh CQ" that receives and executes commands from the Controller.
*   **Key Contrast:** Unlike the proposed SPMD model where *every* host process runs the *same* user code up to the `MeshCommandQueue` submission, here only the Controller runs the main user logic. Executors are passive receivers of commands.

```ascii
    +----------------------------+
    | Controller Host            |
    | Process (1)                |
    |----------------------------|
    | User Code                  |
    | MeshDevice (Global)        |
    | MeshWorkload               |
    | (Serialize Cmds)           |
    |                            |
    | Network Dispatch           |
    +----|-----------------|-----+
         |                 |
         |     (Cmds)      |
         |                 |
         V                 V
+------------------+  +------------------+
| Executor Host 1  |  | Executor Host N  |
| Process (1)      |  | Process (1)      |
|------------------|  |------------------|
| Executor Logic   |  | Executor Logic   |
| RemoteMeshCQ     |  | RemoteMeshCQ     |
|   | (Dispatch)   |  |   | (Dispatch)   |
|   V   Local      |  |   V   Local      |
| +-------+        |  | +-------+        |
| | DevCQ | ...    |  | | DevCQ | ...    |
| +-------+        |  | +-------+        |
+------------------+  +------------------+
```

*   **Potential Pros:**
    *   **Avoids User Code Divergence:** A major advantage. Since user application logic runs only on the Controller, the risk of divergence between hosts due to non-deterministic user code (e.g., unordered map iteration impacting command generation) is eliminated. This contrasts with the SPMD model's strict requirement for deterministic user code on all ranks.
    *   **Familiar User Model:** User code remains single-process, which might be simpler for developers accustomed to single-host programming paradigms.
    *   **Centralized Host State:** If complex host-side global state needs to be managed or calculated *during* workload generation, doing so in a single Controller process can be simpler than coordinating it across multiple SPMD processes.
*   **Potential Cons:**
    *   **Serialization Overhead:** Requires defining, maintaining, and executing serialization/deserialization protocols for commands sent between Controller and Executors, adding complexity and potential performance cost.
    *   **Controller Bottleneck:** The single Controller's computation and network dispatch capabilities can become a performance bottleneck, limiting throughput and scalability, especially with many hosts or high-frequency command submission.
    *   **Executor Complexity:** Requires implementing non-trivial Executor processes capable of receiving, deserializing, and executing commands, plus managing local resources.
    *   **Debugging Distributed System:** Debugging involves understanding and potentially coordinating logs/state across the Controller, the network communication layer, and multiple Executor processes, presenting different challenges than debugging SPMD divergence.

In summary, the Single Controller model offers robustness against user-code-induced host divergence and presents a familiar single-process programming model, but potentially at the cost of serialization overhead, controller bottlenecks, and the complexity of building and debugging the distributed Controller/Executor system. The proposed SPMD / "Multiple Lockstep Controllers" design trades the determinism burden on user code for potentially better scalability (avoiding the controller bottleneck) and reduced serialization overhead by replicating the workload definition phase across hosts.

## Host Coordination Dependency

Any multi-host system requires a mechanism for communication and coordination between the host processes. In the SPMD model proposed here, this is needed for debugging, validation, and host-level barriers or collective operations. In the Single Controller model, it's needed for the Controller to dispatch commands to Executors (e.g., via RPC, ZeroMQ, etc.).

As a proof-of-concept, **MPI** can be used for this coordination layer, primarily due to its ubiquity in HPC environments and its straightforward primitives for collective operations.

However, relying strictly on MPI introduces a potentially heavy dependency that might not be desirable or necessary for all users, especially those not running in traditional HPC environments or those only targeting single-host configurations.

The multi-host runtime should therefore be flexible:

*   **Optional MPI:** The requirement for MPI is gated by a build flag, removing the dependency entirely for users only building/running single-host configurations.
*   **Abstract Interface:** An abstract C++ interface (e.g., `DistributedContext`) should specify the necessary coordination primitives (like `barrier`, `allreduce`, `bcast`, potentially basic point-to-point messaging if needed for other models).
*   **Pluggable Implementations:** The core runtime would use the `DistributedContext` interface. We could then provide:
    *   A default implementation based on MPI (shipped with the package for ease of use).
    *   Potentially other implementations (e.g., using ZeroMQ, gRPC, or other messaging libraries).
    *   Allow users (consumers) to provide their own custom implementation if they have specific infrastructure or performance requirements.

This approach allows users to choose the coordination mechanism that best fits their environment while keeping the core runtime logic agnostic to the specific underlying library.
