# Handoff: Phased cluster validation & discovery by `instance_path` hierarchy

You are picking up hardware validation of a phased cluster-bring-up feature. All code is written, builds,
and passes offline unit tests; **the distributed/multi-host behavior has NOT been run on real hardware** —
that's your job. This brief has everything to build, run, and know what to watch for.

## Goal

Bring a multi-host cluster up **tier by tier, by hardware hierarchy**, using the `instance_path` field in the
Factory System Descriptor (FSD). Each host's `instance_path` is its hierarchy path (e.g.
`bh_galaxy_sp_0/bh_galaxy_node_0/node_0`). Links whose endpoints share a longer `instance_path` prefix are
"closer"; we validate/retrain/discover the closest tier first, then widen outward.

## Branches (checkout)

- **`agupta/fsd-query-instance-path`** — foundation PR: the `FsdQuery` read-only query layer over the FSD proto
  + cabling instance-filter logging. No runtime behavior change on its own.
- **`agupta/phased-cluster-validation`** — **check this one out to test.** Stacked on the foundation; contains
  the phased validation/retrain loop, phased discovery, and the `generate_hostfile` PoC.

```
git fetch origin
git checkout agupta/phased-cluster-validation
```

## What's implemented (files / entry points)

- `tools/scaleout/factory_system_descriptor/query.{hpp,cpp}` — `FsdQuery`:
  `longest_common_prefix`, `hierarchy_depth` (LCP length = tier), cached `max_hierarchy_depth` /
  `hierarchy_tiers_deepest_first`, `get_instance_path`, `hierarchy_partition(depth)` (groups hosts by depth-D
  prefix = the per-hierarchy-node subgroups).
- `tools/scaleout/validation/utils/cluster_validation_utils.{hpp,cpp}`:
  - `filter_topology_by_tier(...)` — slice the missing-connections topology to one tier (LCP == depth).
  - `rediscover_by_hierarchy_subgroups(...)` — one collective `split` by hierarchy-node color → each subgroup
    runs its own `run_physical_system_discovery` → merge (subgroup-local PSD).
  - `phased_bring_up_tier(...)` — per tier: discover subgroups, validate/retrain each subgroup's own tier links
    to convergence. Globally lockstepped via an `all_gather` convergence vote; ends with a world `barrier`.
- `tools/scaleout/validation/run_cluster_validation.cpp` (`main`) — runs phased bring-up tier-by-tier (deepest
  first), then **one whole-system discovery + authoritative validation** with the real `--hard-fail` semantics.
- `tools/scaleout/src/generate_hostfile.cpp` — standalone PoC: FSD → MPI hostfile ordered by `instance_path`
  (rank i = i-th host in hierarchy order). Not yet wired into `ttrun.py`.

## Build

```
./build_metal.sh --build-tests
```
(Use `build_metal.sh`, NOT a bare `cmake --target` — the lib goes stale otherwise.)

## Test

### 1. Offline unit tests (no hardware) — should pass as-is
```
./build_Release/test/tools/scaleout/test_factory_system_descriptor --gtest_filter='FsdQuery.*'   # 12 pass
./build_Release/test/tools/scaleout/test_instance_filter                                          # 16 pass
```

### 2. generate_hostfile (no hardware) — sanity of the ordering
```
./build_Release/tools/scaleout/generate_hostfile \
  --factory-descriptor-path <some FSD .textproto with instance_path> --output /tmp/hostfile
```
Expect: hosts printed as `rank N -> hostname [/seg/seg/...]`, ordered so each hierarchy subtree is a
contiguous rank range.

### 3. THE HARDWARE TEST — phased validation/discovery on a real multi-host cluster
Run `run_cluster_validation` under `mpirun` with a real FSD (or cabling+deployment descriptors). Requires a
system where `supports_ethernet_link_retraining()` is true (WH_B0, or BH with eth FW >= 1.9.0).
Verify in the logs:
- `Starting Tier <N> (depth <d>)` / `Ending Tier <N> (depth <d>)` bookends, deepest tier first.
- Each tier converges (its subgroups' links come up) before the next tier begins.
- A final whole-system authoritative validation pass at the end; overall pass/fail matches a known-good run.
- On a healthy cluster, the phased result should match the pre-existing whole-system flow's result.

## KNOWN LIMITATIONS — read before running on hardware

- **Unbounded hardware poll loops (can hang the run).** The reused reset path polls HW with **no timeout**:
  `tools/scaleout/validation/utils/ethernet_link_api.cpp:49` (`while (reset_status[0])`) and the
  `eth_mailbox_ready` loop at `:89` (Blackhole). `run_physical_system_discovery`'s cross-host gather/barriers
  are also untimed. **If a run hangs, suspect an unresponsive eth core / a slow or absent host** — a single bad
  core stalls the shared collectives (everyone blocks at the next `all_gather`/barrier). Bounding these
  (timeouts + breakout) is a deferred reliability pass; we intentionally left the tried-and-tested paths as-is.
- **FSD must have `instance_path` populated.** If absent, every connection collapses to one tier and phasing
  degenerates to the old whole-system behavior (no error, no warning).
- The phased-discovery collective code (`split`/subgroup-discover/`all_gather`/`reset`/`barrier`) has only been
  compile- and unit-verified. This hardware run is its first real exercise.

## Design decisions already made (context, don't re-litigate)

- **Per-subgroup validation (option B):** each subgroup validates its OWN tier links against the FSD. Works
  because `validate_fsd_against_gsd` (`tools/scaleout/factory_system_descriptor/utils.cpp:465`) only flags a
  connection missing when BOTH endpoints are discovered — so a subgroup-local PSD is naturally scoped to
  intra-subgroup links. No global PSD is assembled during phasing (there is no public PSD gather primitive).
- **Per-hierarchy-node partition:** at depth D, each distinct depth-D `instance_path` prefix is one subgroup.
- **Phase barrier:** each tier ends with a world `barrier()` (defensive; the per-iteration `all_gather` already
  locksteps ranks).

## What to report back

- Does phased bring-up converge tier-by-tier on healthy hardware, and does the final authoritative pass match a
  known-good whole-system run?
- Any hangs (and where — which tier/subgroup, which host/core)? That directly informs whether the deferred
  timeout/breakout reliability work needs to be prioritized.
- Any correctness gaps in the subgroup `split`/discover/merge or the `reset_ethernet_links` composition across
  subgroups.

Deeper design rationale (feasibility, the Phase-A reliability gap, tree gather-reduce) is in
`tools/scaleout/validation/PHASED_DISCOVERY_DESIGN.md` if present locally (it may be uncommitted).
