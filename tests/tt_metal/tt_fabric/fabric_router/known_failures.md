# Known failures: fabric CPU-only PGD / topology-mapper tests

Recorded 2026-09-14 on `rsong/sat-joint-placement` at `9cb505271c8`. Causes 1 to 4 below all
reproduce with the working-tree changes stashed and rebuilt — same tests, same assertions, same line
numbers — so none of them come from the host-split contract work in the working tree. They are
grouped by root cause rather than by test, because 10 of the 13 come from two causes.

Runner: `tests/scripts/multihost/run_fabric_cpu_only_unit_tests.sh --group phys-grouping`.

| Command | Result |
| --- | --- |
| `--gtest_filter=PhysicalGroupingDescriptorSP4Tests*` | 7 passed, 3 failed |
| `--gtest_filter=TopologyMapperUtilsTest.BuildPhysicalMultiMeshGraph_WithPGDAndPSD_Sp4Glx*` | 5 passed, 4 failed |
| `--gtest_filter=...SingleBHGalaxy_*` (+ `VectorOverload_ThreadsPinnings_MatchesSingular`) | 3 passed, 5 failed |
| `--gtest_filter=AdjacencyGuidedPlacement.*` | 15 passed, 1 failed |

Everything else in the group passes: DualT3k PGD, `T3kTopologyMapperCustomMapping` (12),
`T3kMeshGraphTest` (3), N300, P100, `DualGalaxyBigMeshTest`, BHQB4x4 (4),
`ClosetBox3PodTTSwitchHostnameAPIs`, `Pinning*` (3), `ClosetBoxSuperpod*PolicyTest` (2).

## Cause 1: the MGD fallback outranks a placeable PGD grouping

`get_valid_groupings_for_mgd` encodes priority as the order of the committed vector — PGD variants
first, the MGD's own topology appended last — and says so at the commit site:

> Priority is encoded in this vector's order: PGD first, MGD last. Placement walks that list, so
> PGD is preferred and this is only the fallback.

SAT joint placement does not walk that list. Every variant with a non-empty adjacency graph goes
into the per-mesh candidate pool as an equal, which is the open `TODO(preferred-meshes)` on
`MeshCandidatePool`'s constructor in `tt_metal/fabric/physical_grouping_descriptor_matching.cpp`:

> Rework variant ranking for the master solve. Grouping priority from `get_valid_groupings_for_mgd`
> is not a simple boolean (PGD vs MGD vs torus variant vs footprint quality); encode it as a proper
> ranking/objective once the placement model is settled.

So the master solve is free to seat the fallback, and on these machines it does. The logs show the
PGD variants committed and enumerated as live candidates, and the fallback seated anyway:

```
Physical groupings: Mesh graph descriptor 'M0': 4 topology match(es), committed: 4x8_Mesh_flat (MESH), 4x8_Mesh_flat_torus_x (TORUSX), ...
Physical groupings: Mesh graph descriptor 'M0': 4 topology match(es), also offering Mesh graph descriptor: M0 (MESH)
SAT joint placement initial enumeration: M0 (global mesh 0): 21 candidate(s)
Adjacency-guided placement complete (SAT joint placement): 1 mesh(es) seated with PGD grouping(s):
  M0 (global mesh 0): M0 (MESH)          <-- the fallback, not 4x8_Mesh_flat
```

The fallback is the mesh descriptor's own topology, so seating it has three consequences, and each
failing test trips over one of them: it carries no `mesh_node_to_asic_position`, so no pinning is
produced; its footprint is whatever region the solver liked rather than a PGD-shaped block, so
per-node degree is unbounded by the PGD layout; and a region with no boundary has no exit nodes.

| Test | Assertion | What the fallback did |
| --- | --- | --- |
| `AdjacencyGuidedPlacement.PlaceableMgdFallbackDoesNotOutrankPgd` | `test_physical_grouping_descriptor.cpp:5513`, `:5515`, `:5526` — no pinning; footprint `{101,102}` not the pinned `{100,101}` | Direct test of the proposition, and the smallest repro: one 1x2 mesh on a 4-chip line, both variants enumerated (`1x2_Mesh_OnePair_flat[1 found]`, `M0[6 found]`), fallback seated |
| `...SingleBHGalaxy_N300` | `test_topology_mapper_utils.cpp:5035`, `:5037` — 2 vs 4 nodes, degree 2 vs 4 | Seated the MGD's own 1x2 instead of the PGD's 2x2 half-tray |
| `...SingleBHGalaxy_Custom1x17` | `:5087` — 17 vs 32 nodes | Seated the MGD's own 1x17 instead of the PGD's 4x8 block |
| `...SingleBHGalaxy_PgdPinnings` | `:5136` — `mesh_pgd_pinnings_` empty | No chip -> slot pinning on the fallback |
| `...VectorOverload_ThreadsPinnings_MatchesSingular` | `:5221` — `mesh_pgd_pinnings_` empty | Same |
| `...SingleBHGalaxy_RankBoundFastPath_AssignsPgdPinnings` | `:5285` — `mesh_pgd_pinnings_` empty | Same |
| `...Sp4Glx_SingleBHGalaxy` | `:4134` — 0 exit nodes | Region has no boundary the exit-node pass recognises |
| `...Sp4Glx_Quad32x4BhGalaxyTorus` | `:4193` — 0 exit nodes | Same |
| `...Sp4Glx_Galaxy1x32` | `:4924` — 0 exit nodes; `:4937` — degree 10 vs <= 8 | Same, plus a footprint denser than a 1x32 line |

Fix direction: give the master solve the ranking the TODO asks for, so a placeable PGD grouping is
seated ahead of the fallback. The pending preferred-variant pass (mark each pool's top-ranked
variant by `effective_torus_variant_priority`, one budgeted solve with the rest assumed false,
retract and fall back) is the version of this that `main` had before the per-mesh pool rewrite.
Restoring it should close all nine at once; `PlaceableMgdFallbackDoesNotOutrankPgd` is the test to
drive it from, since it isolates the proposition on a four-chip line.

## Cause 2: a test counts committed groupings without counting the fallback

`PhysicalGroupingDescriptorSP4Tests.GetValidGroupingsForMGD_SingleGalaxy4x8`, assertion at
`test_physical_grouping_descriptor.cpp:2103`: 3 committed groupings where it expects 2. The test
comment states its premise — "A 4x8 (32-ASIC) mesh matches both the MESH and a torus variant of the
4x8_Mesh grouping -> 2 matches" — and the log confirms both of those, plus a third entry:

```
... committed: 4x8_Mesh_flat (MESH), 4x8_Mesh_flat_torus_x (TORUSX)
... also offering Mesh graph descriptor: M0 (MESH)
```

The third is the MGD fallback, which is offered whenever it embeds even though a PGD grouping
already committed. The expectation predates that. This is a stale test rather than a code bug: the
fix is to expect 3, or to assert on the two PGD variants by name and let the fallback be.

## Cause 3: no embedding for a multi-galaxy mesh on the SC16 mock

`PhysicalGroupingDescriptorSP4Tests.GetValidGroupingsForMGD_8x16Mesh` (128 chips, quad-galaxy MGD)
and `GetValidGroupingsForMGD_DualGalaxy8x8` (64 chips). Both abort out of the `require_placement`
guard in `get_valid_groupings_for_mgd`:

```
TT_FATAL: Physical groupings: Mesh graph descriptor 'M0': no PGD grouping and no MGD grouping could be placed on the PSD (4 topology match(es))
```

The abort is the guard working as intended — phase 1 asks for a placement and there is none. The
cause is upstream of it, and it is not the host contract: the host-split constraint accepts all four
variants of both shapes.

```
build_flat_adjacency_map_from_psd: 512 ASIC node(s), 3840 local eth link(s), 304 cross-host eth link(s), 76 distinct cross-host ASIC pair(s)
PGD host split '8x16_Mesh_flat' ACCEPTED: 4 declared host group(s) sized [32,32,32,32] each required onto a single one of 16 PSD host partition(s)
DIAG enumerate '8x16_Mesh_flat': NO-EMBEDDING (128 target node(s) on 512 asic(s), validation_mode=0, unique_shapes=true): no new mapping found (all solutions exhausted or excluded)
```

Every variant, and the MGD fallback too, is refused by the topology solver itself: the 8x16 (and
8x8) lattice does not embed in this machine's graph under STRICT validation (`validation_mode=0`).
The machine is 16 hosts x 32 ASICs, and the four galaxies are joined by only 76 distinct cross-host
ASIC pairs, so a contiguous mesh crossing galaxy boundaries has far fewer seam edges available than
its shape needs.

Open question to settle before fixing: whether the `SC16_32x4_revAB_aisleD` mock is meant to wire
the galaxies into a contiguous 8x16 lattice and does not (a mock/PSD gap), or whether these two
tests should be asking for the shape on a different mock, or under RELAXED seam validation. The
other SP4 tests in the same command pass, including the ones that stay inside a single galaxy, which
points at the inter-galaxy wiring rather than at the matcher.

## Cause 4: placement does not pack a pipeline into the fewest hosts

`...Sp4Glx_Blitz2x4_32Stage`, assertion at `test_topology_mapper_utils.cpp:4710`: the mapped
32-stage pipeline spans 10 hosts where the test expects exactly 8. Unlike cause 1, placement did the
right thing at the grouping level — all 32 meshes seated on the PGD variant
`4x2_Mesh_horizontal_flat_torus_x (TORUSX)`. Thirty-two 4x2 meshes are 256 chips, which is 8 hosts
of 32 exactly, so the expectation is that the pipeline packs perfectly; the solver instead spreads
it across 10, leaving two hosts partly used.

Fix direction: host count is a tie-break in the placement objective ("prefer heavier placements,
then single-host, then earlier solver enumeration"), not a packing objective across meshes. Getting
a perfect pack needs the number of hosts spanned by the whole solution to be minimised, not each
mesh's own host count.

## Already tracked in code

New in the working tree, so outside the baseline above. These fail by design and carry their own
explanation at the test, listed here only so the set is complete:

- `PhysicalGroupingDescriptorTests.GetValidGroupingsForMGD_PgdHostsThatContradictThePsdAreRejected`
  — FIXME: nothing validates a PGD's declared host level against the PSD's hosts.
- `PhysicalGroupingDescriptorTests.GetValidGroupingsForMGD_QuadrantSplitPsdMatchingQuadrantMgdCommits`
  — TODO: the matcher never reads the PGD host level, so an MGD split is stamped onto whichever
  orientation the isomorphism returned.
- `PhysicalGroupingDescriptorTestsHostSplit.LengthwiseSplitOnWidthwiseSplitHostsIsTurnedToFitOnASquareMesh`
  — the same gap seen as a rotation: only one MGD<->PGD match comes back, and the rotation that
  would seat the declared split on the hosts is never considered.
