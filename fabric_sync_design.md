# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

# Fabric Phase Synchronization for Fused Ops

## 1. Objective

Fused ops compose multiple CCL (collective communication) operations into a single kernel timeline. When these CCL phases connect to overlapping fabric routers, explicit synchronization is required to prevent undefined behavior.

This document specifies a general-purpose fabric phase synchronization primitive applicable to any fused op with multiple CCL phases sharing overlapping fabric router endpoints. The `attention_block` kernel is used as the reference integration example.

## 2. Problem Statement

**The fabric router supports only one active client connection at a time.** If a second client attempts to `open()` a connection to a fabric router before the previous client has fully `close()`d its connection, the behavior is undefined.

In the `attention_block` kernel, three CCL phases connect to overlapping fabric routers:

| Phase | Op | Core Set | RISCs Using Fabric |
|---|---|---|---|
| `ccl_broadcast` | `Broadcast::Op` | `is_input_core` (1 core) | BRISC |
| `sdpa_reduce_forwarder` | `SdpaReduceForwarder::Op` | `is_sdpa_forwarder_core` (2 cores) | BRISC + NCRISC |
| `ccl_all_reduce_sender` | `AllReduceSender::Op` | `is_ccl_sender_core` (1 core) | BRISC |

Each op explicitly and synchronously disconnects from its fabric router before returning (`connections[i].close()`, `fabric_connection.close()`, `close_connections(fabric_connection)`). The required synchronization guarantee is:

> **A consumer phase must not call `open()` until the producer phase's op has fully returned on all participating cores and RISCs.**

## 3. Design Overview

### 3.1 Two-Level Synchronization

- **Inter-phase sync**: Ordering between phases. Always 1:1 between designated master cores. Uses `FABRIC_SYNC_SEM` with exact state values via `noc_semaphore_set_remote` and `noc_semaphore_wait`.

- **Intra-phase sync**: Coordination within a phase's endpoints (multiple cores and/or multiple RISCs). Handles fan-out (master notifies all peers) and fan-in (all peers signal completion to master). Uses `INTRA_FANOUT_SEM` with `noc_semaphore_set_remote` for fan-out and `INTRA_FANIN_SEM` with `noc_semaphore_inc` for fan-in, with `noc_semaphore_wait_min` for threshold checks on both.

### 3.2 Uniform Peer Model

All endpoints in a phase — whether on a different core or a different RISC on the same core — are treated identically. The host provides NOC coordinates for every peer endpoint. The kernel loops over them uniformly.

### 3.3 Three Global Semaphores

| Semaphore | Purpose | Operations |
|---|---|---|
| `FABRIC_SYNC_SEM` | Inter-phase ordering between masters | `set`/`set_remote` (exact state values), `wait` (exact equality) |
| `INTRA_FANOUT_SEM` | Master → peers start signal (fan-out) | `set_remote` (epoch-correct value), `wait_min` (>= threshold) |
| `INTRA_FANIN_SEM` | Peers → master completion signal (fan-in) | `inc`, `wait_min` (>= num_peers), `set(0)` reset by master |

**Why split fan-out and fan-in into separate semaphores?**

Using a single `INTRA_SYNC_SEM` for both fan-out and fan-in creates two bugs when the master and a peer share the same L1 address (i.e., same-core BRISC/NCRISC):

1. **Same-core clobber**: The master sets `INTRA_SYNC_SEM = 1` locally to unblock the same-core peer. The peer immediately runs, reaches `signal()`, and `inc`s the shared address to `2`. But the master's fan-out loop then issues `set_remote(1)` to that same NOC address — overwriting `2` back to `1`, erasing the peer's completion `inc`. The master then hangs at `wait_min(1 + num_peers)` forever.

2. **Stale fan-out value**: After Iteration 0, a different-core peer's `INTRA_SYNC_SEM` stays at `1`. In Iteration 1, the peer's `wait_min(1)` passes immediately — bypassing the master's fan-out barrier entirely and opening a fabric connection before the previous phase has finished.

Splitting into `INTRA_FANOUT_SEM` and `INTRA_FANIN_SEM` eliminates both bugs cleanly:
- The master writes to `INTRA_FANOUT_SEM` (fan-out) and reads from `INTRA_FANIN_SEM` (fan-in). These are different L1 addresses — a same-core peer's `inc` to `INTRA_FANIN_SEM` can never be overwritten by the master's `set_remote` to `INTRA_FANOUT_SEM`.
- `INTRA_FANOUT_SEM` uses epoch-doubled state values (same scheme as `FABRIC_SYNC_SEM`), so a stale value from Iteration N can never satisfy a `wait_min` in Iteration N+1. No explicit reset of `INTRA_FANOUT_SEM` is needed.
- `INTRA_FANIN_SEM` is reset to `0` by the master after `wait_min(num_peers)`, as before.

### 3.4 Multi-Iteration Safety

**`FABRIC_SYNC_SEM` and `INTRA_FANOUT_SEM` — epoch-doubled state values with in-place toggle:**

The naive scheme (`expected_val = phase_index`) breaks on the second iteration: the semaphore still holds the value written in Iteration N, so the same `wait` passes immediately in Iteration N+1 before the preceding phase has run at all.

The fix is **epoch-doubled state values**: state values cycle over `2 × num_phases` unique values, alternating between two epochs. No two consecutive iterations share any `expected_val`, so a stale value from Iteration N can never satisfy the wait in Iteration N+1. No explicit reset of either semaphore is needed.

Only the epoch-0 `base_expected` value per phase is stored in the RTAs (1 word instead of 4). The kernel toggles `expected_val` in-place at the start of each iteration in `reset()`:

```cpp
// In reset(), for each phase ctx:
ctx.expected_val = (ctx.expected_val < NUM_PHASES)
                 ? ctx.expected_val + NUM_PHASES   // epoch 0 → epoch 1
                 : ctx.expected_val - NUM_PHASES;  // epoch 1 → epoch 0
```

`sync_val` is derived on the fly in `signal()` with a single branch:

```cpp
constexpr uint32_t MAX_STATE = 2 * NUM_PHASES;
uint32_t sync_val = (ctx.expected_val + 1 == MAX_STATE) ? 0 : ctx.expected_val + 1;
```

Both `FABRIC_SYNC_SEM` and `INTRA_FANOUT_SEM` use the same `base_expected` per phase (they share the same epoch-cycling scheme). The `sync_val` derivation is identical for both.

**`INTRA_FANIN_SEM`:** The master resets `INTRA_FANIN_SEM` to `0` after `wait_min(num_peers)` in `signal()`. Peers do not reset it. The master's reset happens before the inter-phase `set_remote`, which is before the next phase's master fans out — so no next-iteration peer can `inc` before the reset completes.

**`ctx_idx`**: Reset to `0` via `reset()` at the start of each iteration.

## 4. Host-Side API: `FabricSyncGraph`

The host builds a linear phase chain, then compiles it into **common named compile-time args** (per RISC type, same value for all cores) and **common runtime args** (per RISC type, same flat list for all cores). There are no per-core compile-time or runtime args for the sync primitive. The kernel self-identifies its role using `my_logical_x_` / `my_logical_y_` and `is_brisc` / `is_ncrisc`, which are firmware-set globals and constexpr flags already present in all kernels via `kernel_utils.hpp` and `kernel_op_api.hpp`.

`FabricSyncGraph` owns semaphore creation. The op does not need to know how many semaphores are required or which cores they should span — the graph computes this internally from the phases that have been added.

### 4.1 Graph Construction

```python
# RISC type constants
BRISC  = 0
NCRISC = 1

class FabricSyncGraph:
    """
    Builds a linear chain of fabric phases and compiles synchronization
    metadata into common compile-time args and common runtime args.
    Owns the three global semaphores required by the protocol.
    """
    def __init__(self):
        self.phases            = []
        self.fabric_sync_sem   = None
        self.intra_fanout_sem  = None
        self.intra_fanin_sem   = None

    def add_phase(self, name, producer_cores, riscs, master_risc):
        """
        Appends a phase to the linear chain. Phases are ordered by insertion order.
        The inter-phase signal target is automatically derived from the next phase's
        producer_cores, so there is no need to specify consumer_cores explicitly.

        Args:
            name:           Human-readable phase name (e.g. "ccl_broadcast").
            producer_cores: CoreRangeSet of cores running this phase's CCL op.
            riscs:          List of RISC types that open fabric connections in this phase.
                            e.g. [BRISC], [NCRISC], or [BRISC, NCRISC].
            master_risc:    Which RISC on the master core drives the inter-phase sync.
                            Must be one of the entries in `riscs`.
        """
        self.phases.append({
            "name":           name,
            "producer_cores": producer_cores,
            "riscs":          riscs,
            "master_risc":    master_risc,
        })

    def get_sem_core_set(self):
        """
        Returns the union of all producer_cores across all phases.
        All three global semaphores must be allocated over this core set.
        """
        result = CoreRangeSet()
        for phase in self.phases:
            result |= phase["producer_cores"]
        return result

    def create_semaphores(self, mesh_device):
        """
        Allocates FABRIC_SYNC_SEM, INTRA_FANOUT_SEM, and INTRA_FANIN_SEM as global
        semaphores over the union of all participating cores. Must be called before
        compile(). The semaphore handles are stored internally and referenced by compile().
        """
        core_set = self.get_sem_core_set()
        self.fabric_sync_sem  = ttnn.create_global_semaphore(mesh_device, core_set, 0)
        self.intra_fanout_sem = ttnn.create_global_semaphore(mesh_device, core_set, 0)
        self.intra_fanin_sem  = ttnn.create_global_semaphore(mesh_device, core_set, 0)

    def compile(self):
        """
        Processes the phase chain and produces common compile-time and runtime args.
        create_semaphores() must be called before compile().

        Returns a SyncConfig object providing:
            - get_compile_time_args(risc)  -> list of (name, value) named CTA pairs
            - get_runtime_args(risc)       -> flat list of uint32 values
        Both outputs are identical for every core of the given RISC type.
        """
        ...
```

### 4.2 Compile Step Internals

For each phase `P` with index `i` (0-based):

1. **State value assignment** (epoch-doubled, single stored value): Only the epoch-0 `base_expected = i` is stored in the RTAs. The kernel toggles it in-place in `reset()` each iteration. `sync_val` is derived on the fly in `signal()` as `(base_expected + 1) % (2 * N)`, with the single wrap case handled by a branch. Both `FABRIC_SYNC_SEM` and `INTRA_FANOUT_SEM` use the same `base_expected` per phase.

2. **Master nomination**: The first core in `producer_cores` (CoreRangeSet iteration order) is the master core. `master_risc` (passed to `add_phase`) is the master RISC on that core.

3. **Peer enumeration**: Every `(core, risc)` pair in `producer_cores × riscs` except `(master_core, master_risc)` is a peer. For a 2-core phase with `riscs=[BRISC, NCRISC]` and `master_risc=BRISC`:
   - Master: `(core_0, BRISC)`
   - Peers: `(core_0, NCRISC)`, `(core_1, BRISC)`, `(core_1, NCRISC)` — 3 peers total

4. **Inter-phase target**: Always the master core of the next phase's `producer_cores` (or phase 0 for the last phase). Derived automatically during `compile()` — no caller input needed.

5. **Intra-phase fan-in wait target** (for the master in `signal()`): `num_peers`. Each peer `inc`s `INTRA_FANIN_SEM` by 1; the master waits until the value reaches `num_peers` to confirm all peers have completed. (Unlike the old single-semaphore design, no base `set(1)` is needed — fan-out and fan-in are on separate semaphores.)

### 4.3 Common Compile-Time Args (Named, Per RISC Type)

Two named CTAs size the kernel-side arrays. They are set identically for every core of a given RISC type:

| Arg name | Value | Purpose |
|---|---|---|
| `fabric_sync_num_phases` | Number of phases | Size the per-phase context array |
| `fabric_sync_max_peers` | Max `num_peers` across all phases | Size the intra-peer NOC addr array |

All other sync metadata (state values, master coordinates, peer NOC coords, inter-phase target NOC coords) is carried as **common runtime args**, parsed once during `init()` before the iteration loop.

### 4.4 Common Runtime Args Layout (Per RISC Type)

The same flat list is passed to every core of a given RISC type. Peer NOC coords are variable-length per phase (exactly `num_intra_peers` pairs, no padding). `MAX_PEERS` only sizes the kernel-side struct array — it has no effect on the wire format. The inter-phase target is a single fixed pair since the chain is always linear.

```
[fabric_sync_sem_addr]      -- L1 address of FABRIC_SYNC_SEM
[intra_fanout_sem_addr]     -- L1 address of INTRA_FANOUT_SEM
[intra_fanin_sem_addr]      -- L1 address of INTRA_FANIN_SEM

Per phase (repeated fabric_sync_num_phases times):
  [master_logical_x]        -- logical X of master core (for role self-identification)
  [master_logical_y]        -- logical Y of master core (for role self-identification)
  [master_risc]             -- 0=BRISC, 1=NCRISC
  [master_noc_x]            -- NOC X of master core (for peer fan-in inc)
  [master_noc_y]            -- NOC Y of master core (for peer fan-in inc)
  [base_expected]           -- epoch-0 FABRIC_SYNC_SEM/INTRA_FANOUT_SEM expected value (= phase index i)
  [num_intra_peers]         -- number of peers (0 for single-endpoint phases)
  [peer_0_noc_x, peer_0_noc_y]    -- only present if num_intra_peers > 0; no padding
  [peer_1_noc_x, peer_1_noc_y]    --   ...
  ...
  [target_noc_x, target_noc_y]    -- NOC coords of next phase's master (inter-phase signal)
```

Non-participating cores (BRISC/NCRISC) parse the same args during `init()` but never call `wait()` / `signal()`. TRISC skips `init()` entirely via `if constexpr (is_trisc)`. If arg parsing overhead is a concern for non-participants, an `is_core_sync_participant` named CTA can gate the `init()` call via `if constexpr`.

### 4.5 Host Pseudocode: `compile()`

```python
def compile(self):
    num_phases = len(self.phases)

    common_rtas = [
        self.fabric_sync_sem.address(),
        self.intra_fanout_sem.address(),
        self.intra_fanin_sem.address(),
    ]

    max_peers = 0
    for i, phase in enumerate(self.phases):
        master_core = first_core(phase["producer_cores"])  # logical coords
        master_noc  = logical_to_noc(master_core)
        peers       = enumerate_peers(phase)               # list of (core, risc) pairs
        next_phase  = self.phases[(i + 1) % num_phases]
        target_noc  = logical_to_noc(first_core(next_phase["producer_cores"]))
        max_peers   = max(max_peers, len(peers))

        common_rtas += [
            master_core.x, master_core.y,
            phase["master_risc"],
            master_noc.x, master_noc.y,
            i,             # base_expected (epoch-0 value); kernel toggles in-place each iteration
            len(peers),
        ]
        # Peer NOC coords: variable-length, no padding
        for (peer_core, _) in peers:
            common_rtas += [logical_to_noc(peer_core).x, logical_to_noc(peer_core).y]

        # Inter-phase target: always exactly one for a linear chain
        common_rtas += [target_noc.x, target_noc.y]

    named_ctas = [
        ("fabric_sync_num_phases", num_phases),
        ("fabric_sync_max_peers",  max_peers),
    ]
    return SyncConfig(named_ctas=named_ctas, common_rtas=common_rtas)
```

## 5. Kernel-Side API: `FabricSync`

A templated struct that parses sync args once before the iteration loop and provides `wait()` / `signal()` / `reset()` methods.

Role determination (`is_master`) is done at runtime in `init()` by comparing `my_logical_x_` / `my_logical_y_` (firmware globals from `kernel_utils.hpp`) and `MY_RISC_ID` (derived from `is_brisc` / `is_ncrisc` constexpr flags in `kernel_op_api.hpp`) against the master coordinates in the common RTAs.

`init()` is guarded by `if constexpr (is_trisc)` — TRISC does not have access to `get_common_arg_val` and never participates in fabric sync.

### 5.1 Struct Definition

```cpp
#include "dataflow_api.h"
#include "unified_kernels/kernel_utils.hpp"   // my_logical_x_, my_logical_y_
#include "unified_kernels/kernel_op_api.hpp"  // is_brisc, is_ncrisc, is_trisc

constexpr uint32_t RISC_BRISC  = 0;
constexpr uint32_t RISC_NCRISC = 1;
constexpr uint32_t MY_RISC_ID  = is_brisc ? RISC_BRISC : RISC_NCRISC;

template <uint32_t NUM_PHASES, uint32_t MAX_PEERS>
struct FabricSync {

    struct PhaseCtx {
        bool     is_master;
        // Epoch-0 base value stored; toggled in-place by reset() each iteration.
        // sync_val is derived on the fly: (expected_val + 1) with wrap at 2*NUM_PHASES.
        uint32_t expected_val;
        uint32_t num_intra_peers;
        // Fan-in wait target = num_intra_peers: each peer incs INTRA_FANIN_SEM by 1.
        uint32_t intra_wait_target;
        uint64_t intra_peer_fanout_noc_addrs[MAX_PEERS]; // INTRA_FANOUT_SEM on each peer (master only)
        uint64_t master_fanin_noc_addr;                  // INTRA_FANIN_SEM on master  (peer only)
        uint64_t inter_target_fab_noc_addr;              // FABRIC_SYNC_SEM on next-phase master
    };

    PhaseCtx ctxs[NUM_PHASES];
    uint32_t fab_sem_addr;
    uint32_t fanout_sem_addr;
    uint32_t fanin_sem_addr;
    uint32_t ctx_idx = 0;

    void init(uint32_t& arg_idx) {
        if constexpr (is_trisc)        return;
        if constexpr (NUM_PHASES == 0) return;

        fab_sem_addr    = get_common_arg_val<uint32_t>(arg_idx++);
        fanout_sem_addr = get_common_arg_val<uint32_t>(arg_idx++);
        fanin_sem_addr  = get_common_arg_val<uint32_t>(arg_idx++);

        for (uint32_t p = 0; p < NUM_PHASES; p++) {
            auto& ctx = ctxs[p];

            uint32_t master_lx    = get_common_arg_val<uint32_t>(arg_idx++);
            uint32_t master_ly    = get_common_arg_val<uint32_t>(arg_idx++);
            uint32_t master_risc  = get_common_arg_val<uint32_t>(arg_idx++);
            uint32_t master_noc_x = get_common_arg_val<uint32_t>(arg_idx++);
            uint32_t master_noc_y = get_common_arg_val<uint32_t>(arg_idx++);

            ctx.is_master   = (my_logical_x_ == master_lx)
                           && (my_logical_y_ == master_ly)
                           && (MY_RISC_ID    == master_risc);
            ctx.expected_val = get_common_arg_val<uint32_t>(arg_idx++);  // base_expected (epoch 0)

            ctx.master_fanin_noc_addr = get_noc_addr(master_noc_x, master_noc_y, fanin_sem_addr);

            // Peer NOC coords: variable-length, no padding in the arg stream.
            ctx.num_intra_peers = get_common_arg_val<uint32_t>(arg_idx++);
            for (uint32_t i = 0; i < ctx.num_intra_peers; i++) {
                uint32_t px = get_common_arg_val<uint32_t>(arg_idx++);
                uint32_t py = get_common_arg_val<uint32_t>(arg_idx++);
                ctx.intra_peer_fanout_noc_addrs[i] = get_noc_addr(px, py, fanout_sem_addr);
            }
            ctx.intra_wait_target = ctx.num_intra_peers;

            uint32_t tx = get_common_arg_val<uint32_t>(arg_idx++);
            uint32_t ty = get_common_arg_val<uint32_t>(arg_idx++);
            ctx.inter_target_fab_noc_addr = get_noc_addr(tx, ty, fab_sem_addr);
        }
    }

    // Must be called once at the start of each iteration (before any wait()/signal() pairs).
    // Resets ctx_idx and toggles all per-phase expected_val between epoch 0 and epoch 1.
    void reset() {
        if constexpr (NUM_PHASES == 0) return;
        ctx_idx = 0;
        for (uint32_t p = 0; p < NUM_PHASES; p++) {
            ctxs[p].expected_val = (ctxs[p].expected_val < NUM_PHASES)
                                 ? ctxs[p].expected_val + NUM_PHASES   // epoch 0 → epoch 1
                                 : ctxs[p].expected_val - NUM_PHASES;  // epoch 1 → epoch 0
        }
    }

    void wait() {
        if constexpr (NUM_PHASES == 0) return;
        auto& ctx = ctxs[ctx_idx];

        if (ctx.is_master) {
            // Inter-phase: block until previous phase master signals the epoch-correct value
            volatile tt_l1_ptr uint32_t* fab_sem =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(fab_sem_addr);
            noc_semaphore_wait(fab_sem, ctx.expected_val);

            if (ctx.num_intra_peers > 0) {
                // Intra-phase fan-out: broadcast epoch-correct value to all peers via INTRA_FANOUT_SEM.
                // No write barrier needed — the master proceeds to its own fabric work immediately.
                // The fan-in wait_min in signal() confirms all peers received and acted on it.
                constexpr uint32_t MAX_STATE = 2 * NUM_PHASES;
                uint32_t fanout_val = (ctx.expected_val + 1 == MAX_STATE) ? 0 : ctx.expected_val + 1;
                for (uint32_t i = 0; i < ctx.num_intra_peers; i++) {
                    noc_semaphore_set_remote(fanout_val, ctx.intra_peer_fanout_noc_addrs[i]);
                }
            }
        } else {
            // Peer: block until master fans out the epoch-correct value
            constexpr uint32_t MAX_STATE = 2 * NUM_PHASES;
            uint32_t fanout_expected = (ctx.expected_val + 1 == MAX_STATE) ? 0 : ctx.expected_val + 1;
            volatile tt_l1_ptr uint32_t* fanout_sem =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(fanout_sem_addr);
            noc_semaphore_wait_min(fanout_sem, fanout_expected);
        }
    }

    void signal() {
        if constexpr (NUM_PHASES == 0) return;
        auto& ctx = ctxs[ctx_idx];

        if (ctx.is_master) {
            if (ctx.num_intra_peers > 0) {
                // Wait for all peer fan-in incs to INTRA_FANIN_SEM
                volatile tt_l1_ptr uint32_t* fanin_sem =
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(fanin_sem_addr);
                noc_semaphore_wait_min(fanin_sem, ctx.intra_wait_target);
                noc_semaphore_set(fanin_sem, 0);  // master resets INTRA_FANIN_SEM for next iteration
            }
            // Derive sync_val and signal next phase master via FABRIC_SYNC_SEM
            constexpr uint32_t MAX_STATE = 2 * NUM_PHASES;
            uint32_t sync_val = (ctx.expected_val + 1 == MAX_STATE) ? 0 : ctx.expected_val + 1;
            volatile tt_l1_ptr uint32_t* fab_sem =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(fab_sem_addr);
            noc_semaphore_set(fab_sem, sync_val);
            noc_semaphore_set_remote(fab_sem_addr, ctx.inter_target_fab_noc_addr);
            noc_async_write_barrier();
        } else {
            // Peer: fan-in to master's INTRA_FANIN_SEM. Peers do NOT touch INTRA_FANOUT_SEM.
            noc_semaphore_inc(ctx.master_fanin_noc_addr, 1);
        }

        ctx_idx++;
    }
};
```

### 5.2 Auto-Advancing Phase Index

`ctx_idx` increments on every `signal()` call. Since `wait()` / `signal()` pairs are only invoked on cores that participate in a given phase (gated by `if constexpr (Core::is_X_core)`), `ctx_idx` is always at the correct context for that core. For the current `attention_block`, every participating core is in exactly one phase (`NUM_PHASES = 1`). Non-participating cores have `NUM_PHASES = 0` and all methods compile to no-ops.

`reset()` resets `ctx_idx = 0` and toggles all per-phase `expected_val`s between epoch 0 and epoch 1 ranges (`[0, N)` ↔ `[N, 2N)`). It must be called once at the start of each iteration before any `wait()` / `signal()` pair.

## 6. Detailed Protocol

### 6.1 Consumer Wait (Inter-Phase + Intra-Phase Fan-Out)

```
Master (is_master == true):
  1. noc_semaphore_wait(FABRIC_SYNC_SEM, expected_val)          -- block until previous phase signals
  2. for each peer: noc_semaphore_set_remote(INTRA_FANOUT_SEM,  -- fan-out epoch-correct value to all peers
                        fanout_val = (expected_val+1) % 2N)
     (no write barrier — master proceeds to its own fabric work; fan-in in signal() confirms delivery)

Every peer (is_master == false):
  1. noc_semaphore_wait_min(INTRA_FANOUT_SEM, fanout_val)       -- block until master fans out
```

Step 2 is skipped when `num_intra_peers == 0` (single-endpoint phase).

### 6.2 Producer Signal (Intra-Phase Fan-In + Inter-Phase)

```
Every peer (is_master == false):
  1. noc_semaphore_inc(master's INTRA_FANIN_SEM addr, 1)        -- fan-in completion to master

Master (is_master == true):
  1. noc_semaphore_wait_min(INTRA_FANIN_SEM, num_peers)         -- wait for all peer incs
  2. noc_semaphore_set(INTRA_FANIN_SEM, 0)                      -- master resets for next iteration
  3. noc_semaphore_set(FABRIC_SYNC_SEM, sync_val)               -- set epoch-correct inter-phase state
  4. noc_semaphore_set_remote(FABRIC_SYNC_SEM, target_addr)     -- signal next phase master
  5. noc_async_write_barrier()
```

Steps 1–2 are skipped on the master when `num_intra_peers == 0`.

### 6.3 Ordering Guarantee

- **Inter-phase**: The master's `wait(FABRIC_SYNC_SEM, expected_val)` blocks until the preceding phase's master fires `set_remote(sync_val)`. Epoch-doubling guarantees `expected_val` is unique across consecutive iterations.
- **No write barrier after fan-out**: The master proceeds to its own fabric work immediately after issuing `set_remote`s to `INTRA_FANOUT_SEM`. The NOC guarantees delivery; the fan-in `wait_min` in `signal()` confirms all peers have received and acted on it before the next phase is unblocked.
- **Fan-in completeness**: No peer can call `signal()` (issue its `inc` to `INTRA_FANIN_SEM`) until after it has passed `wait_min(fanout_val)`, executed the CCL op, and called `close()` on its fabric connection. The master's `wait_min(num_peers)` therefore guarantees all peers have closed their fabric connections.
- **Same-core safety**: `INTRA_FANOUT_SEM` and `INTRA_FANIN_SEM` are at different L1 addresses. The master's `set_remote` to a peer's `INTRA_FANOUT_SEM` address can never collide with that peer's `inc` to the master's `INTRA_FANIN_SEM` address — even when master and peer share a core (BRISC/NCRISC).
- **`noc_semaphore_inc` is commutative**, so arrival order of peer completions does not matter.
- **`INTRA_FANOUT_SEM` stale-value safety**: Epoch-doubled `fanout_val` ensures a stale value from Iteration N (`fanout_val = v`) never satisfies the `wait_min(v')` in Iteration N+1 (where `v' ≠ v` since epoch flipped). No explicit reset of `INTRA_FANOUT_SEM` is needed.

### 6.4 Multi-Iteration Safety Summary

| What | Mechanism | Owner |
|---|---|---|
| `FABRIC_SYNC_SEM` stale-value safety | Epoch-doubled `expected_val` (in-place toggle in `reset()`) | `reset()` |
| `INTRA_FANOUT_SEM` stale-value safety | Same epoch-doubled value (`fanout_val = sync_val`) | `reset()` |
| `INTRA_FANIN_SEM` reset | `noc_semaphore_set(fanin_sem, 0)` after `wait_min` | master in `signal()` |
| `ctx_idx` reset | `ctx_idx = 0` | `reset()` |
| epoch advance | in-place `expected_val` toggle across all phases | `reset()` |

## 7. Integration Example: `attention_block`

### 7.1 Phase Topology

```
ccl_broadcast ──→ sdpa_reduce_forwarder ──→ ccl_all_reduce_sender ──→ (reset to ccl_broadcast)
  (1 core)          (2 cores × 2 RISCs)        (1 core)
  BRISC only         BRISC + NCRISC             BRISC only
```

| Phase | `base_expected` | Master | Peers |
|---|---|---|---|
| `ccl_broadcast` | 0 | `bcast_core` BRISC | none |
| `sdpa_reduce_forwarder` | 1 | `fwd_0` BRISC | `fwd_0` NCRISC, `fwd_1` BRISC, `fwd_1` NCRISC |
| `ccl_all_reduce_sender` | 2 | `sender_core` BRISC | none |

State values cycle over `2 × 3 = 6` unique values (`[0,1,2]` in even iterations, `[3,4,5]` in odd). `FABRIC_SYNC_SEM` and `INTRA_FANOUT_SEM` are initialized to `0`. `INTRA_FANIN_SEM` is initialized to `0`.

### 7.2 Compile-Time Args

Common named CTAs, same for all cores of each RISC type:

| RISC | `fabric_sync_num_phases` | `fabric_sync_max_peers` |
|---|---|---|
| BRISC | 1 | 3 |
| NCRISC | 1 | 3 |

All BRISC and NCRISC cores receive these values. Cores not in any phase also receive them but never call `wait()` / `signal()`.

### 7.3 Common Runtime Args (for `attention_block`)

Both BRISC and NCRISC receive the same list. Peer slots carry exactly `num_intra_peers` pairs — no padding. `MAX_PEERS = 3` only sizes the kernel struct array.

```
[fab_sem_addr, fanout_sem_addr, fanin_sem_addr]

-- Phase 0: ccl_broadcast (0 peers) --
[bcast_lx, bcast_ly, BRISC=0, bcast_noc_x, bcast_noc_y,
 base_expected=0,
 num_peers=0,
 fwd0_noc_x, fwd0_noc_y]              <- inter-phase target: sdpa_reduce_forwarder master

-- Phase 1: sdpa_reduce_forwarder (3 peers) --
[fwd0_lx, fwd0_ly, BRISC=0, fwd0_noc_x, fwd0_noc_y,
 base_expected=1,
 num_peers=3,
   fwd0_noc_x, fwd0_noc_y,            <- peer: fwd_0 NCRISC (same NOC coords, different sem addr)
   fwd1_noc_x, fwd1_noc_y,            <- peer: fwd_1 BRISC
   fwd1_noc_x, fwd1_noc_y,            <- peer: fwd_1 NCRISC
 sender_noc_x, sender_noc_y]          <- inter-phase target: ccl_all_reduce_sender master

-- Phase 2: ccl_all_reduce_sender (0 peers) --
[sender_lx, sender_ly, BRISC=0, sender_noc_x, sender_noc_y,
 base_expected=2,
 num_peers=0,
 bcast_noc_x, bcast_noc_y]            <- inter-phase target: ccl_broadcast master
```

> **Note on Phase 1 same-core peers**: `fwd_0 NCRISC` is on the same core as the master (`fwd_0 BRISC`). They share NOC coordinates but have separate `INTRA_FANOUT_SEM` and `INTRA_FANIN_SEM` L1 addresses (global semaphores are at the same L1 address on all cores). The master's `set_remote` writes to `fwd_0`'s `INTRA_FANOUT_SEM`; the NCRISC's `inc` goes to the master's `INTRA_FANIN_SEM`. These are different addresses — no clobber is possible.

### 7.4 Host-Side Setup (in `op.py`)

```python
# Build the phase graph — FabricSyncGraph owns semaphore creation
graph = FabricSyncGraph()
graph.add_phase(
    name="ccl_broadcast",
    producer_cores=input_cores,
    riscs=[BRISC],
    master_risc=BRISC,
)
graph.add_phase(
    name="sdpa_reduce_forwarder",
    producer_cores=forwarder_cores,     # 2 cores: fwd_0, fwd_1
    riscs=[BRISC, NCRISC],
    master_risc=BRISC,                  # fwd_0 BRISC is the phase master
)
graph.add_phase(
    name="ccl_all_reduce_sender",
    producer_cores=sender_cores,
    riscs=[BRISC],
    master_risc=BRISC,
)

# Semaphore creation: graph determines participating cores internally
graph.create_semaphores(mesh_device)

sync_config = graph.compile()

# Append to existing per-RISC arg lists
# TRISC does not receive sync args (init() is a no-op for TRISC)
brisc_named_compile_time_args  += sync_config.get_compile_time_args(BRISC)
ncrisc_named_compile_time_args += sync_config.get_compile_time_args(NCRISC)
brisc_common_runtime_args      += sync_config.get_runtime_args(BRISC)
ncrisc_common_runtime_args     += sync_config.get_runtime_args(NCRISC)

# Expose the semaphores for inclusion in the program's semaphore list
semaphore_list += graph.get_semaphores()  # returns [fabric_sync_sem, intra_fanout_sem, intra_fanin_sem]
```

### 7.5 Kernel-Side Integration (in `attention_block_kernel.cpp`)

```cpp
// Common named CTAs (same for all cores; TRISC receives them but init() is a no-op)
constexpr uint32_t SYNC_NUM_PHASES = get_named_compile_time_arg_val("fabric_sync_num_phases");
constexpr uint32_t SYNC_MAX_PEERS  = get_named_compile_time_arg_val("fabric_sync_max_peers");

using SyncType = FabricSync<SYNC_NUM_PHASES, SYNC_MAX_PEERS>;
SyncType fabric_sync;

// Parse once before the iteration loop (TRISC: compiles to no-op)
uint32_t sync_arg_idx = FABRIC_SYNC_COMMON_RTA_BASE;  // offset after preceding common args
fabric_sync.init(sync_arg_idx);

// ---- inside mla_body() — called once per iteration ----

fabric_sync.reset();  // reset ctx_idx to 0 at the start of each iteration

// Phase 0: CCL Broadcast
if constexpr (Core::is_input_core) {
    fabric_sync.wait();
    if constexpr (!Core::skip_ccl) {
        deepseek_b1_ops::Broadcast::Op<BcastCTArgs, Core::is_input_core> bcast;
        bcast(bcast_args);
    }
    fabric_sync.signal();
}

// ... non-fabric ops (RMSNorm, Matmul, Gather, MLA, SDPA) ...

// Phase 1: SDPA Reduce Forwarder
if constexpr (Core::is_sdpa_forwarder_core) {
    fabric_sync.wait();
    deepseek_b1_ops::SdpaReduceForwarder::Op<SdpaReduceForwarderCTArgs> sdpa_reduce_forwarder;
    sdpa_reduce_forwarder(sdpa_reduce_forwarder_args);
    fabric_sync.signal();
}

// ... non-fabric ops (Matmul4, Gather2, Mcast3, Matmul5, Gather3) ...

// Phase 2: CCL All-Reduce Sender
if constexpr (Core::is_ccl_sender_core) {
    fabric_sync.wait();
    deepseek_b1_ops::AllReduceSender::Op<CCLSenderReaderCTArgs, DummyWriterCTArgs> ccl_sender_reader;
    ccl_sender_reader(ccl_sender_args);
    deepseek_b1_ops::AllReduceSender::Op<DummyReaderCTArgs, CCLSenderWriterCTArgs> ccl_sender_writer;
    ccl_sender_writer(ccl_sender_args);
    fabric_sync.signal();
}
```

### 7.6 Execution Trace: `sdpa_reduce_forwarder` Phase

This is the most complex phase: 2 cores × 2 RISCs = 4 endpoints. Master is `fwd_0 BRISC`. `base_expected = 1`.

**Epoch 0 state values** (`expected_val=1`, `fanout_val = sync_val = 2`):

**`wait()` — fan-out:**

| Step | fwd_0 BRISC (master) | fwd_0 NCRISC (peer) | fwd_1 BRISC (peer) | fwd_1 NCRISC (peer) |
|---|---|---|---|---|
| 1 | `wait(FAB_SEM, 1)` | `wait_min(FANOUT, 2)` | `wait_min(FANOUT, 2)` | `wait_min(FANOUT, 2)` |
| 2 | unblocks; `set_remote(FANOUT=2)` → fwd_0, fwd_1 BRISC, fwd_1 NCRISC | — | — | — |
| 3 | — | unblocks, proceeds | unblocks, proceeds | unblocks, proceeds |

All 4 endpoints now open fabric connections and run the forwarder concurrently.

**`signal()` — fan-in (signals `sync_val=2` to `ccl_all_reduce_sender`):**

| Step | fwd_0 BRISC (master) | fwd_0 NCRISC (peer) | fwd_1 BRISC (peer) | fwd_1 NCRISC (peer) |
|---|---|---|---|---|
| 1 | — | `inc(fwd_0 FANIN, 1)` → FANIN=1 | `inc(fwd_0 FANIN, 1)` → FANIN=2 | `inc(fwd_0 FANIN, 1)` → FANIN=3 |
| 2 | `wait_min(FANIN, 3)` [num_peers=3] | done | done | done |
| 3 | `set(FANIN, 0)` — master resets | — | — | — |
| 4 | `set(FAB_SEM, 2)` | — | — | — |
| 5 | `set_remote(FAB_SEM=2)` → sender core | — | — | — |
| 6 | `write_barrier()` | — | — | — |

**Epoch 1 state values** (after `reset()` toggles): `expected_val=4`, `fanout_val=sync_val=5`. Same structure, different values.

**Completion guarantee chain:**

```
fwd_0 NCRISC close() → inc(fwd_0 FANIN) ──┐
fwd_1 BRISC  close() → inc(fwd_0 FANIN) ──┤
fwd_1 NCRISC close() → inc(fwd_0 FANIN) ──┤
fwd_0 BRISC  close() ─────────────────────→ wait_min(FANIN, 3) → set_remote(FAB_SEM=2) → sender
```

After step 6, the sender core unblocks. All 4 fabric connections are guaranteed closed.

## 8. Handling `skip_ccl` (Single-Device Mode)

When `skip_ccl` is true, no fabric connections are opened, but the sync chain still executes. `wait()` and `signal()` are placed outside the `skip_ccl` conditional:

```cpp
if constexpr (Core::is_input_core) {
    fabric_sync.wait();
    if constexpr (!Core::skip_ccl) {
        // ... Broadcast::Op ...
    }
    fabric_sync.signal();
}
```

The host generates valid sync args regardless of `skip_ccl`. In single-device mode the sync is effectively a no-op: `FABRIC_SYNC_SEM` is already at `expected_val`, so `wait()` passes immediately and `signal()` advances the state. The graph topology remains static.
