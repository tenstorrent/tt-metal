# A.2.6 Evaluation: tt_fabric::Reconfigure() for minimal-diff reroute

**Branch**: `nsexton/0-racecondition-hunt`
**Worktree**: `/workspace/group/worktrees/racecondition-main/`

---

## 1. Does the problem described in A.2.6 exist?

**Yes.** When `reliability_mode`, `num_planes`, or `udm_mode` change, the code destroys
and fully reconstructs the ControlPlane — including topology discovery, mesh graph
generation, topology mapping/constraint solving, and routing table generation — even
though these parameters only affect a small subset of that work.

### The code path

When `set_fabric_config()` is called with the same `FabricConfig` enum but different
sub-parameters (reliability_mode, udm_mode, etc.), the flow is:

**`metal_env.cpp:240-316`** — `MetalEnvImpl::set_fabric_config()`:
```
force_reinit_ = true;                          // line 248 — always
...
if (control_plane_ != nullptr) {               // line 311
    system_mesh_.reset();                      // line 313 — destroy SystemMesh
    this->initialize_control_plane_impl();     // line 314 — full rebuild
}
```

**`metal_env.cpp:554`** — `initialize_control_plane_impl()` calls `construct_control_plane()`:
```
control_plane_ = std::make_unique<tt::tt_fabric::ControlPlane>(...);  // line 601-615
```

**`control_plane.cpp:685-706`** — Every ControlPlane constructor calls:
```
init_control_plane_auto_discovery();   // or init_control_plane() for descriptor-based
initialize_fabric_context();
```

**`control_plane.cpp:585-684`** — `init_control_plane_auto_discovery()` performs:
1. `run_physical_system_discovery()` — UMD-level hardware interrogation (line 606)
2. `TopologyMapper::generate_mesh_graph_from_physical_system_descriptor()` — mesh graph generation (line 610-614)
3. `TopologyMapper` construction — constraint solving + topology mapping (line 637-646)
4. `RoutingTableGenerator` construction (line 672)
5. `initialize_distributed_contexts()` (line 675)
6. `generate_intermesh_connectivity()` (line 676)

Steps 1, 3, 4, 5, 6 are **completely independent** of `reliability_mode`, `udm_mode`,
or `num_planes`. Only step 2 uses `reliability_mode`, and only to set a single boolean:

**`mesh_graph.cpp:967-968`**:
```cpp
bool is_relaxed = (reliability_mode == FabricReliabilityMode::RELAXED_SYSTEM_HEALTH_SETUP_MODE);
mesh_graph.intra_mesh_relaxed_policy_[mesh_id] = is_relaxed;
```

The `udm_mode` is stored in the ControlPlane and only read later by `FabricContext`
construction (`fabric_context.cpp:207,376`), not during topology mapping at all.

### Where the waste happens

The actual callers that trigger this full-rebuild path:

1. **`set_custom_fabric_topology()`** (`metal_env.cpp:663`) — calls `set_fabric_config()` with same fabric_config_ but hardcoded `STRICT_SYSTEM_HEALTH_SETUP_MODE`. Full rebuild.
2. **`set_default_fabric_topology()`** (`metal_env.cpp:676`) — same pattern, plus resets control_plane_ to null first.
3. **`DeviceManager::initialize()`** (`device_manager.cpp:291`) — calls `set_fabric_config()` with `STRICT_SYSTEM_HEALTH_SETUP_MODE` even when fabric config hasn't changed.

---

## 2. Comparison with nsexton/0-batch-t3k-ttnn-unit

**The situation is the same.** The `set_fabric_config()` function in racecondition-hunt
is structurally identical to the batch branch for this specific flow. The racecondition
branch has added significant teardown safety (GAP-78 use_count check, FIX AB extension
for teardown timeout propagation, FIX BA lazy-init guard, etc.) but none of those changes
touch the "config changed, rebuild everything" path at lines 311-314.

Key differences in the racecondition branch that are **adjacent but do not affect A.2.6**:
- `teardown_fabric_config()` has extensive ERISC force-reset logic (FIX AI, FIX AB, F5a) — but this only runs when transitioning to DISABLED, not during same-config sub-parameter changes.
- `check_use_count_zero()` GAP-78 guard — protects against concurrent teardown, orthogonal to the rebuild path.
- `control_plane_mutex_` locking — serializes access but doesn't reduce rebuild cost.

**Verdict: same problem, same severity.**

---

## 3. What would implementing this recommendation look like?

The recommendation is to add a `Reconfigure(new_config)` path that avoids full
ControlPlane destruction + reconstruction. Concretely:

### Option A: ControlPlane::reconfigure() method

Add to `ControlPlane`:
```
void reconfigure(FabricReliabilityMode mode, FabricUDMMode udm, uint8_t num_planes);
```

This method would:
1. Update `fabric_reliability_mode_` in-place
2. Update `fabric_udm_mode_` in-place
3. Update `mesh_graph_->intra_mesh_relaxed_policy_[mesh_id]` based on the new reliability mode
4. If `num_planes` changed: re-run `configure_routing_tables_for_fabric_ethernet_channels()` only
5. If `udm_mode` changed: reconstruct `fabric_context_` only (not the whole ControlPlane)
6. Skip: physical_system_descriptor_, topology_mapper_, routing_table_generator_, distributed contexts

### Option B: Diff-based detection in set_fabric_config()

In `MetalEnvImpl::set_fabric_config()` (lines 260-316), before the `if (control_plane_ != nullptr)` block at line 311, add diff detection:

```
bool topology_changed = (fabric_config != this->fabric_config_);
bool mode_only_changed = !topology_changed && (
    reliability_mode != this->fabric_reliability_mode_ ||
    fabric_udm_mode != this->fabric_udm_mode_);

if (control_plane_ != nullptr) {
    if (topology_changed) {
        // full rebuild (current code)
    } else if (mode_only_changed) {
        control_plane_->reconfigure(reliability_mode, fabric_udm_mode, ...);
    }
}
```

### Affected files:
- `tt_metal/impl/context/metal_env.cpp` — set_fabric_config() diff logic
- `tt_metal/fabric/control_plane.cpp` — new reconfigure() method
- `tt_metal/api/tt-metalium/experimental/fabric/control_plane.hpp` — declare reconfigure()
- `tt_metal/fabric/mesh_graph.cpp` — need setter for `intra_mesh_relaxed_policy_`
- `tt_metal/api/tt-metalium/experimental/fabric/mesh_graph.hpp` — declare setter

### Savings estimate:
- Avoids `run_physical_system_discovery()` — this is a full UMD hardware walk
- Avoids `TopologyMapper` construction — constraint solving with potential timeout (60-120s budget)
- Avoids `RoutingTableGenerator` reconstruction
- Avoids file I/O (YAML export of mappings)
- For the `set_custom_fabric_topology()` / `set_default_fabric_topology()` callers, the saving is large: they call `set_fabric_config()` with the _same_ fabric_config but always `STRICT_SYSTEM_HEALTH_SETUP_MODE`.

---

## 4. Interactions with race-condition fixes

Several race-condition fixes in this branch are **relevant** to implementing A.2.6:

### a) GAP-78: use_count teardown ordering (`metal_env.cpp:130-161`)
**Impact: Positive.** The `check_use_count_zero()` guard ensures no dispatch threads are active during fabric config changes. This means a `reconfigure()` path would have the same safety guarantees — no concurrent L1 writes to worry about. A reconfigure() would inherit this protection naturally.

### b) `control_plane_mutex_` serialization (`metal_env.cpp:542,550,646`)
**Impact: Must be respected.** The mutex protects lazy init of `control_plane_`. A `reconfigure()` call must hold this mutex. Currently `set_fabric_config()` does NOT hold it (the lock is only in `get_control_plane()` and `initialize_control_plane()`), so adding a `reconfigure()` call inside `set_fabric_config()` would need to acquire the lock — or the existing code already has a latent bug where concurrent `get_control_plane()` could race with `set_fabric_config()` destroying the ControlPlane.

### c) FIX BA: lazy re-init guard (`metal_env.cpp:495-500`)
**Impact: Simplifies things.** FIX BA prevents `get_control_plane()` from accidentally re-initializing during teardown. With a `reconfigure()` path, we would never set `control_plane_` to null for mode-only changes, so FIX BA's guard becomes a non-issue for this path.

### d) `force_reinit_ = true` at line 248
**Impact: Must be addressed.** Currently `set_fabric_config()` unconditionally sets `force_reinit_ = true`. With a `reconfigure()` path, we'd want to distinguish: if only sub-parameters changed and we successfully reconfigured in-place, `force_reinit_` should NOT be set (the MetalContext doesn't need a full close+reopen). This requires changing line 248 to be conditional.

### e) `set_custom_fabric_topology()` and `set_default_fabric_topology()`
**Impact: These are the highest-value targets.** Both explicitly destroy `control_plane_` before calling `set_fabric_config()`, which forces a full rebuild. With a reconfigure path, these could skip the destroy when only the reliability mode is changing. However, `set_default_fabric_topology()` also clears `logical_mesh_chip_id_to_physical_chip_id_mapping_` and resets `custom_mesh_graph_desc_path_`, which means it's doing more than just a mode change — it's genuinely resetting topology. A reconfigure() path wouldn't help there unless we split the "reset topology" from "update mode" concerns.

### Summary of interactions

The race-condition fixes are **orthogonal** to A.2.6 — they protect teardown/init ordering but don't make the full-rebuild path faster or slower. Implementing A.2.6 would:
- Benefit from GAP-78 (safe concurrent access guarantee)
- Need to respect `control_plane_mutex_` (add locking or verify caller holds it)
- Need to conditionally set `force_reinit_` (avoid unnecessary MetalContext restart)
- Work cleanly alongside FIX BA (no null control_plane_ for mode-only changes)

**Recommendation**: A.2.6 is applicable and independently implementable. It does not conflict with any race-condition fix. The `control_plane_mutex_` interaction deserves attention but is a straightforward fix (acquire lock around reconfigure call). Priority is moderate — the saving is real but the callers that trigger mode-only changes are relatively infrequent (test fixture SetUp/TearDown, not hot paths).
