# Handoff — Inter-Mesh Connection Contention Findings

Owner/handoff-from: debugging session, 2026-07
Related design: [`intermesh_port_pairing_redesign.md`](./intermesh_port_pairing_redesign.md)
Tracking: [#49629](https://github.com/tenstorrent/tt-metal/issues/49629) (SC16 64-stage z-link strand), [#49628](https://github.com/tenstorrent/tt-metal/issues/49628) (Z contention epic)

---

## TL;DR

The SC16 64-stage superpod ring fatals with "boundary `62↔63` resolved ZERO routers." We
traced it end to end. **The PSD/cable and the topology mapper are fine.** The strand is a
**greedy port-exhaustion in `propose_port_descriptors_for_exit_nodes`**: the torus wrap-around
`M63↔M0` (spread over all 8 edge chips) is assigned NESW ports first and drains the shared
chips (2–5) that `M63↔M62` needs. `#49629`'s "Z-link the mapper doesn't assign" hypothesis is
**wrong** — nothing is dropped in pairing; one side just never proposes.

## Reproduce

```bash
tt-run --mesh-graph-descriptor models/demos/deepseek_v3_b1/scaleout_configs/blitz_decode_mesh_graph_descriptor_superpod.textproto \
  --mock-cluster-rank-binding tt_metal/third_party/tt-cluster-descriptors/superclusters/blackhole/SC16_32x4_revC_subtorus_aisleD/SC16_32x4_revC_subtorus_aisleD_mapping.yaml \
  --mpi-args "--allow-run-as-root --oversubscribe" --force-rediscovery \
  ./build/test/tt_metal/tt_fabric/fabric_unit_tests \
  --gtest_filter="ControlPlaneFixture.TestGalaxyLayoutCheck:ControlPlaneFixture.TestBlitzDecodePipelineBuilder"
```
Fatal: `control_plane.cpp: num_resolved_between(src,dst) > 0 ... between mesh 63 and mesh 62`.
Run artifacts (rank bindings, rankfile, mock mapping, MGD) copied to
`/data/rsong/twohost_test/sc16_64stage_run/`.

## Debug tooling

Set `TT_INTERMESH_DEBUG=1`. Committed diagnostics (commit `102bd9e`) log per-boundary
drop tallies + a per-chip Z-lane owner table in `pair_logical_intermesh_ports`. During this
investigation additional (uncommitted, exploratory) logs in `propose`/`generate` printed
per-neighbor proposal counts, PSD cable dumps, and per-chip logical-port inventories — re-add
as needed.

## Investigation chain (what was ruled out, in order)

1. **Not a determinism bug.** Earlier work made the mapping host-independent (sorted
   host/neighbor/exit-node/boundary orders). Placement md5 is byte-identical across hosts.
2. **Not `pair_logical` contention.** With `TT_INTERMESH_DEBUG`, across all 64 boundaries:
   `z_mismatch` drops = 0, `chip_z_taken` drops = 0. The reconcile drops nothing.
3. **The stranded boundaries are one-sided.** `M6->M7`, `M42->M43`, `M62->M63` each have
   `src_props` or `dst_props` = **0** — one mesh proposes no ports for the boundary, so there
   is nothing to hash-match.
4. **The PSD *has* the cable, both ways.** For `M63↔M62` and `M63↔M0`:
   `get_connecting_exit_nodes` returns 8 cables each direction, `connection_requested=true`,
   the neighbor is in `get_host_neighbors`. So gather sees the cables.
5. **`propose` returns 0 on one side despite 8 cables.** `M63->M62` → PROPOSED 0 (while
   `M62->M63` → 8). The loss is inside `propose_port_descriptors_for_exit_nodes`.
6. **Root cause: greedy per-neighbor exhaustion on shared chips.** `assigned_port_ids`
   accumulates across neighbors; the wrap `M63->M0` is processed first and takes the ports.

## The concrete data (mesh 63)

PSD physical cables:

| boundary | cables | src chips | channels | port_type |
|---|---|---|---|---|
| M63↔M0 (wrap) | 16 | **all 8** (chip0–7) | ch8/ch9 | QSFP_DD |
| M63↔M62 (hop) | 8 | chip2–5 | ch4/ch5 | TRACE |

They overlap on chips 2–5. Each shared chip exposes only **2 NESW ports + 2 Z ports**
(`RoutingDirection` Z = enum 4). What each boundary took:

| chip | logical ports | M0 took | M62 took | free |
|---|---|---|---|---|
| 2 | `W:ch2 W:ch3` `Z:ch4 Z:ch5` | `W:ch2 W:ch3` | (none) | `Z:ch4 Z:ch5` |
| 3 | `E:ch2 E:ch3` `Z:ch6 Z:ch7` | `E:ch2 E:ch3` | (none) | `Z:ch6 Z:ch7` |
| 4 | `W:ch4 W:ch5` `Z:ch8 Z:ch9` | `W:ch4 W:ch5` | (none) | `Z:ch8 Z:ch9` |
| 5 | `E:ch4 E:ch5` `Z:ch10 Z:ch11` | `E:ch4 E:ch5` | (none) | `Z:ch10 Z:ch11` |

So: `should_assign_z_direction(63↔0) = false` → M0 (though physically on the Z lane ch8/ch9)
is put on **NESW**, grabbing each shared chip's only 2 NESW ports. M62 needs NESW, finds only
the Z ports free — **no Z was ever assigned; the Z ports sit unused**.

## Two distinct issues

1. **Allocation is unfair (the fatal).** Greedy per-neighbor fill lets the wrap drain shared
   chips before the hop gets a turn → M62 resolves zero. Fixed by the round-robin, both-sides
   allocator in the redesign doc.
2. **The `assign_z` classifier under-marks (the *correct* placement).**
   `should_assign_z_direction(63↔0)` returns `false` even though the wrap rides the Z lane
   (ch8/ch9, QSFP). If it returned `true`, the wrap would take Z and free the NESW for the
   hop — the clean assignment. This is a **separate fix** in the classifier, independent of
   the allocator rework. Worth investigating how `should_assign_z_direction` decides (the MGD
   has no explicit `assign_z` field; it is derived).

## Sweep context

All 10 SC20 subtorus-aisleD sweep mappings (`SC20_sweep_01..10`) also fatal at
`num_resolved==0`, each on a different boundary (28↔27, 4↔5, 3↔4, 59↔58, 32↔33, …). Same
class of failure — subtorus carvings create boundaries whose shared chips get drained. The
sweep data lives in tt-cluster-descriptors PR #14
(`riddy21/sc36-revab-aisled-sweep-mappings`).

## Pointers

- Design/fix: [`intermesh_port_pairing_redesign.md`](./intermesh_port_pairing_redesign.md)
- Determinism work + goldens: tt-metal PR #49447 (`riddy21/fatal-intermesh-routing-validation`)
- Sweep mappings + per-rank descriptors: tt-cluster-descriptors PR #14
- Run artifacts: `/data/rsong/twohost_test/sc16_64stage_run/`
