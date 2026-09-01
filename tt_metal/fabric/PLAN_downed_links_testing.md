# Test plan: FSD-backed topology map + downed links

**Status:** Test plan
**Umbrella:** [tenstorrent/tt-metal#52859](https://github.com/tenstorrent/tt-metal/issues/52859)
**Implementation plan:** [`PLAN_fsd_solve_and_downed_links.md`](PLAN_fsd_solve_and_downed_links.md)
**Contract:** [`README_downed_links_contract.md`](README_downed_links_contract.md)

> **Decisions locked in §7.1–§7.6, §7.8, and §7.11.** Intermesh downed is **not** empty. Cross-host intermesh is
> visible. `is_link_healthy` is live-PSD presence and **works on mock**. FSD is golden (extras OK;
> host mismatch and **>10%** missing are errors — log the fraction). **STRICT + FSD: do not fatal
> on down** — document in LinkHealth (including intermesh) and **ignore `is_link_healthy` / ETH-up
> for abort**. Filter failure fails **every rank with the same message**. **Downgraded-plane holes
> go to `get_unused_downed_links()`** and do not count as `planes_lost`. **Mapper keys on
> `(host, tray, loc)`, not AsicID** — FSD placement matches live/tt-run. See README §3.1–§3.3.

**Contents**

1. [Layers and targets](#1-layers-and-targets)
2. [PSD diff (offline)](#2-psd-diff-offline)
3. [`LinkHealth`](#3-linkhealth)
4. [ControlPlane — single host](#4-controlplane--single-host)
5. [ControlPlane — multi host](#5-controlplane--multi-host)
6. [E2E testing](#6-e2e-testing)
7. [Risk-driven cases](#7-risk-driven-cases) — walk these one at a time
8. [Open questions](#8-open-questions)
9. [Not tested in v1](#9-not-tested-in-v1)
10. [Tests for the leftover review findings](#10-tests-for-the-leftover-review-findings-implementation-plan-10) — pairs with implementation plan §10

---

## 1. Layers and targets

| Layer | Needs HW | File | Target / list |
|-------|----------|------|---------------|
| PSD diff | no | `physical_discovery/test_physical_system_descriptor_diff.cpp` | `test_physical_discovery` / `UNIT_TESTS_PHYSICAL_DISCOVERY_SRC` |
| FSD→PSD + host filter | no | `fabric_router/test_physical_descriptor_builder.cpp` (extend) | `fabric_unit_tests` / `UNIT_TESTS_FABRIC_SRC` |
| `LinkHealth` | mock or HW (see §8) | `fabric_router/test_link_health.cpp` | `fabric_unit_tests` |
| ControlPlane single-host | mock + FSD | `fabric_router/test_fsd_psd_e2e.cpp` | `fabric_unit_tests` |
| ControlPlane multi-host | mock + FSD, ≥2 ranks | `fabric_router/test_multi_host.cpp` (extend) | `fabric_unit_tests` |
| E2E FSD vs mock PSD | mock | `fabric_router/test_fsd_psd_e2e.cpp` | `fabric_unit_tests` |
| Phase 1 (`generate_rank_bindings`) | yes | QuietBox scale-out sanity | `tests/scale_out/4x_bh_quietbox` |

`test_physical_descriptor_builder.cpp` is the model for offline work: it builds FSD protos in-code and links
`TT::ScaleoutTools`, no device. `TopologyMapperTest` is the model for mock/device work (`run_physical_system_discovery`
in `SetUp`).

Reusable FSD fixtures:

- Small: `tests/scale_out/4x_bh_quietbox/factory_system_descriptors/factory_system_descriptor_4x_bh_quietbox.textproto`
  (paired with `global_system_descriptors/4x_bh_quietbox_physical_desc.yaml`).
- Supercluster: FSDs live in `tt_metal/third_party/tt-cluster-descriptors` next to the matching
  `*_cluster_desc/` + mapping YAML. Do **not** copy the 40k–80k-line files into `tests/`. See §6.

RTOptions has a setter (`set_factory_system_descriptor_path`), so a test can drive the FSD path without the
`TT_METAL_FACTORY_SYSTEM_DESCRIPTOR_PATH` env var. Restore it in `TearDown`.

---

## 2. PSD diff (offline)

Build two descriptors in-code, no device.

- Identical → `matches()`; all six vectors empty.
- Drop one cable from candidate → both directed endpoints in `missing_links`, nothing in `extra_links`.
- Swap arguments → the same cable now in `extra_links`.
- Drop a whole ASIC → `missing_asics` plus every cable it terminated.
- Add a cable to candidate → `extra_links` only.
- Drop one and add another → both populated, no cross-contamination.
- Same endpoints, different `port_type` → `mismatched_links`, **not** missing+extra (join key excludes `port_type`).
- Same ASIC, different tray → `mismatched_asics`.
- One cable is two directed records → exactly one entry per end, not two per end.
- Output vectors sorted; two runs byte-identical.
- Disjoint AsicID spaces with the **same** `(host, tray, loc)` on both sides → still `matches()` for
  those chips. Join is positional (implementation plan §5.5). Disjoint positions → missing+extra.

## 3. `LinkHealth`

Fixture per `test_topology_mapper.cpp`. Mutate the live PSD copy, keep the mapper fixed.

- Identical expected/live → `get_downed_links()` empty, `fsd_rerouting_active() == false`.
- Drop an intra cable → one record per direction, `logical_resolved`, `scope == IntraMesh`.
- Intra `src_direction` / `dst_direction` both match `intra_mesh_connectivity` and are typically opposites
  (`E`/`W`, `N`/`S`).
- Extras in live (present in live, absent in expected) never appear in `get_downed_links()`.
- Unmapped ASIC (in FSD, never placed by the solve) → `logical_resolved == false`, `scope == Unknown`,
  `is_intramesh()` and `is_intermesh()` both false, present in per-host / per-ASIC queries, absent from
  per-node / per-direction / per-mesh-pair queries.
- Physical fields (`host`, `tray`, `loc`, `medium`) come from the **expected** PSD, not live.
- Index pointers: every `const LinkInfo*` points inside `downed_`; `downed_.data()` unchanged after indexing.
- After plane downgrade: intra holes on that `(node, dir)` move to `get_unused_downed_links()`; they must
  not appear in `get_downed_links()` or `planes_lost` (§7.11 / implementation §3.3).
- `refresh()` twice is idempotent; `refresh(nullptr, &other_live)` rebinds.
- Not default-constructible, not copyable, not movable (static asserts).

Dropping a whole **ASIC** from the live PSD is a class-level unit test only — `LinkHealth` takes any two
PSDs, so the behaviour is worth pinning. It is **not** a reachable ControlPlane state: §7.7 fails init
before `LinkHealth` is built. Do not turn it into a ControlPlane expectation.

### 3.1 Intermesh downed is **not** empty (required)

An FSD-expected intermesh cable missing from the live PSD **is a downed link**. It goes into `downed_`
with `scope == InterMesh` and is returned by `get_downed_intermesh_links()`. Do not filter it out
against post-pairing MeshGraph (that table is assigned-up only and would empty the set).

Setup: drop one FSD-expected **intermesh** cable from the live PSD, keep the mapper.

| Assertion | Why |
|-----------|-----|
| `get_downed_intermesh_links()` has one record per direction | the set is not empty |
| `get_downed_links(LinkScope::InterMesh)` and `get_downed_intermesh_eth_chans(node)` agree | same records |
| `src_direction` / `dst_direction` are `NONE` if pairing never assigned that cable a port | MeshGraph has no per-cable direction for a hole |
| physical fields from the **expected** PSD | DC-bringup identity |
| `get_downed_links_between_hosts(src_host, dst_host)` contains it when the two ends are on different hosts | §7.2 — cross-host is not invisible |
| `is_link_healthy(node, chan) == false` (and the AsicID overload agrees) | presence check; see §7.3 |

Works on **mock cluster descriptors**: `is_link_healthy` is FSD-expected vs live-PSD presence
(`live_present_`). It does **not** call `cluster.is_ethernet_link_up`. Mock discovery builds a live
PSD from the YAML; drop the cable there (or omit it from the mock desc) and the API returns `false`.
The ETH-up path is `confirm_local_downed_links` only — and with an FSD that path must not fatal
(§7.5). So a mock E2E can assert `is_link_healthy` without silicon.

STRICT + FSD: init must **not** `TT_FATAL` on this missing cable. The record in LinkHealth *is* the
result. Do not consult `is_link_healthy` / ETH-up to decide whether to abort. See §7.5.

### 3.2 `is_link_healthy` works on mock cluster descriptors

`is_link_healthy` is FSD-expected vs live-PSD presence (`live_present_`). It does **not** call
`cluster.is_ethernet_link_up` (`tt_cluster.cpp:1435` → `ethernet_core_has_active_ethernet_link` on
the YAML). Mock discovery already builds a live PSD from the cluster-desc YAML, so this API works
without silicon.

| Setup | Expected |
|-------|----------|
| FSD + mock, cable present in live PSD | `true` (node, AsicID, and host/tray/loc overloads agree) |
| FSD + mock, cable dropped/omitted from live PSD (intra or intermesh) | `false` |
| FSD + mock, `(node, chan)` never in the FSD | `std::out_of_range` |
| No FSD | `true` (no throw) |

Assert this in the SC20 mock E2E (`test_fsd_psd_e2e.cpp`), not only on silicon. STRICT + FSD must
still **ignore** this return value for abort — see §7.5.

## 4. ControlPlane — single host

Uses an FSD. Launch: one MPI rank, a single-galaxy mock cluster desc (the `*_SINGLE_GALAXY_CLUSTER_DESC`
paths already in `run_fabric_cpu_only_unit_tests.sh`) plus that cluster's FSD. Hostname join (§6.3) must
exist before these can init. **STRICT + FSD must not fatal** on downed links (§7.5) — run these in
STRICT as well as RELAXED. `make_control_plane` as in `test_multi_host.cpp`.

- **No FSD (one case, same binary):** walk the whole contract surface. Defaults, no throw / segfault.
  `link_health_ == nullptr`, `get_link_health() == nullptr`, `locally_unhealthy_` empty.
- **FSD + mock, healthy pair:** alias succeeds; init completes; mapper is on the FSD PSD; `get_downed_links()`
  empty; `fsd_rerouting_active() == false`. Placement is identical to a solve on a PSD whose AsicIDs were
  rewritten to UMD-like values (§7.8). A missing live chip at ControlPlane **fatals** (§7.7); it does
  **not** fatal in `generate_rank_bindings`.
- **FSD is a superset of this one host:** downed links only for the allocated host; `fsd_rerouting_active()`
  still false if that host is healthy. Without the filter this reports the rest of the aisle.
- `LinkHealth` constructed only after `generate_intermesh_connectivity()` (§7.1). A dropped intermesh
  cable **is** in `get_downed_intermesh_links()` / `get_downed_links()` — the set is not empty.
- Drop one intra cable from the live PSD (mutate after discovery, or via `refresh`): one record per
  direction, `scope == IntraMesh`, `src_direction` / `dst_direction` from MeshGraph. `LinkInfo` host is
  the **FSD** hostname (`bh-glx-110-c01u02`), not the YAML basename. After `configure_routing_tables`
  (plane min + trim + merge): if that direction **downgraded**, the record is in
  `get_unused_downed_links()` and `planes_lost == 0`; if not, it stays in `get_downed_intramesh_links()`
  (§7.11).
- Local ETH down → `locally_unhealthy_`. Remote chip → not in `locally_unhealthy_`, still in
  `get_downed_links()`.
- **STRICT + FSD:** init must not `TT_FATAL` on a downed intra or intermesh cable. The record in
  LinkHealth is the result. Do not consult `is_link_healthy` / ETH-up to abort (§7.5).
- **FSD golden / PSD extras:** a cable present only in the live PSD is OK — not downed, not an error
  (§7.4). Hosts that do not match the FSD, or more than 10% of FSD-expected connections down → error
  (log + fail).

## 5. ControlPlane — multi host

Uses an FSD. Launch: `tt-run --mock-cluster-rank-binding <SC20 mapping>` + the matching FSD. Every
assertion on every rank (Finding A: a throw on a subset deadlocks MPI).

- Alias table is identical on every rank (same FSD file, same mapping). All-reduce a checksum of the
  sorted `(mock_key → fsd_hostname)` pairs before ingest — this is the check that turns §7.6 into an
  error instead of a hang.
- Filter uses `alias.values()`; retained host set equals the mapping's host set.
- `get_downed_links()` identical on every rank, remotes included. Order-sensitive checksum, MIN/MAX
  all-reduce, same helper shape as `expect_intermesh_resolved_pairs_consistent_across_ranks`.
- `fsd_rerouting_active()` agrees on every rank.
- Drop one **cross-host intermesh** cable: every rank sees it in `get_downed_intermesh_links()` **and**
  in `get_downed_links_between_hosts(a, b)`. Cross-host is not invisible (§7.2). STRICT + FSD: no
  `TT_FATAL` (§7.5).
- `locally_unhealthy_` is a strict subset of `get_downed_links()` and disjoint across ranks.
- Healthy exact-match: empty downed set, flag false.
- Drop one intra cable on one rank's live view: every rank sees that record; only the owning rank has
  it in `locally_unhealthy_`.
- Force the alias / filter to miss on exactly one rank: **every rank throws the same `what()`**
  (`agree_or_throw_fsd_host_filter`, §7.6). No hang.

---

## 6. E2E testing

Real hall/aisle FSDs + the mock clusters fabric already runs in CI. §2–§3 stay small and synthetic.
§4–§5 are the ControlPlane cases; this section is the assets, the hostname join, and the FSD-vs-PSD
diff that those cases depend on.

**Assets:** [tenstorrent/tt-cluster-descriptors#18](https://github.com/tenstorrent/tt-cluster-descriptors/pull/18)
(`79cb691`). Each BH supercluster dir has `<cluster>_factory_system_descriptor.textproto` next to
`*_cluster_desc/` and the mapping YAML. The matching live PSD is mock-discovered from those UMD YAMLs
(`tt-run --mock-cluster-rank-binding`). QuietBox is the one pair with a checked-in PSD
(`4x_bh_quietbox_physical_desc.yaml`). Do **not** copy the 40k–80k-line FSDs into `tests/`.

Submodule pin: `tt_metal/third_party/tt-cluster-descriptors` @ `79cb691`. Re-pin to `main` when #18 merges.

Launch like the existing SC20 / SC36 mapper tests in `tests/scripts/multihost/run_fabric_cpu_only_unit_tests.sh`,
plus `TT_METAL_FACTORY_SYSTEM_DESCRIPTOR_PATH` (or `rtoptions.set_factory_system_descriptor_path` in `SetUp`).

### 6.1 Pair table

| Cluster | FSD vs cluster host set | Use |
|---------|-------------------------|-----|
| `SC20_32x4_revAB_aisleC` | exact | **start here** — 1-rank slice + full 20-rank ControlPlane |
| `SC20_32x4_revC_subtorus_aisleC` | exact | same, revC |
| `SC16_32x4_revC_aisleC` | FSD superset (aisle C) | filter + ControlPlane |
| `SC16_32x4_revAB_aisleD` | FSD superset (aisle D) | filter + ControlPlane |
| `SC16_32x4_revC_subtorus_aisleD` | FSD superset (aisle D) | filter |
| `SC36_32x4_revAB_subtorus_aisleC` | FSD superset (aisle C) | filter |
| `SP3_16x8_revAB_aisleC` | FSD superset (aisle C) | filter; 1-pod mapping already in the cpu-only script |
| `SC36_32x4_revC_subtorus_aisleD` | near-miss (~1 host) | **error-case fixture** (implementation plan §5.3 case 2) |
| `SC24_32x4_revC_subtorus_virtu` | no FSD (`rNNuNN` names) | skip |
| `superclusters/wormhole/wh_closetbox` | no FSD | skip — do not run FSD+ControlPlane here |

### 6.2 The names do not match — what the code actually does

Three different strings name the same machine:

| Side | Produced by | Example |
|------|-------------|---------|
| FSD `hosts[].hostname` | cabling-descriptor author | `bh-glx-110-c01u02` |
| Cluster-desc **filename** | tt-cluster-descriptors convention | `SC20_32x4_revAB_aisleC_cluster_desc_bh-glx-c01u02_rank_0.yaml` (no hall `110`) |
| Live PSD host key | `get_local_discovery_hostname()` | that YAML **basename**, including `.yaml` |

`get_local_discovery_hostname()` (`physical_system_discovery.cpp:60`) is the whole reason. When
`TT_METAL_MOCK_CLUSTER_DESC_PATH` is set it returns `std::filesystem::path(...).filename()`, not a
hostname. `tt-run --mock-cluster-rank-binding` sets that env var **per rank** to that rank's YAML, so
each rank's live key is a different basename and `resolve_hostname_uniqueness()` usually reports unique
— no `_<rank>` suffix on BH supercluster mappings.

The packed hostname is **exactly** the FSD `hosts[].hostname` (`bh-glx-110-c01u02`).
Hall is part of that name; we do not drop it. The mock YAML **filename** is not a
hostname — it just happens to contain a different token (`bh-glx-c01u02`, SC36
`bh-glx-120-d10u20`). That is file naming in tt-cluster-descriptors, not a
canonicalization we apply. `PhysicalNodeId.host[]` is the FSD string
([`PLAN_physical_node_id.md`](PLAN_physical_node_id.md) §8).

When names do collide (comment at `run_local_discovery` ~654: 64-rank superpod reusing 16 mock files),
the live key becomes `basename + "_" + rank` (`physical_system_descriptor.cpp:455`,
`all_hostnames_unique_ == false`).

ClosetBox is a different pattern (`closet_box_cluster_desc_metal-wh-09.yaml`) and has no FSD.
`TopologyMapperTest.ClosetBox3PodTTSwitchHostnameAPIs` asserts on those basename keys. **Do not change
`get_local_discovery_hostname()`.** Every existing mock test keys on the basename.

The FSD proto already has the structured location (`factory_system_descriptor.proto` `Host`: `hall`,
`aisle`, `rack`, `shelf_u`). `#18` matched clusters by the aisle+rack+shelf token (`c01u02`). That is
the join, not the raw string.

**Do not rewrite FSD files or cluster-desc filenames.** One name space is produced at ingest, for the
FSD path only.

### 6.3 Hostname consolidation (ingest remap)

New helpers, next to the mock host alias — not in discovery, not in the builder's default path.

```
aisle_token_from_fsd_host(h):
    // prefer structured fields, not the hostname string
    // aisle="C", rack=1, shelf_u=2  →  "c01u02"
    lower(aisle) + pad(rack, 2) + "u" + pad(shelf_u, 2)

aisle_token_from_mock_key(k):
    strip trailing "_<rank>" if !psd.get_all_hostnames_unique()
    last regex [a-z][0-9]{2}u[0-9]{2} in the basename, before _rank_ or .yaml
    // SC20_..._bh-glx-c01u02_rank_0.yaml  →  c01u02
    // no match (closetbox, virtu rNNuNN) → nullopt

build_mock_to_fsd_host_alias(live_psd, fsd):
    for each live host key:
        tok = aisle_token_from_mock_key(key)
        find the unique FSD Host with aisle_token_from_fsd_host(H) == tok
        0 matches  → fatal, list the live key + a few FSD names   // wrong FSD / no token
        >1 match   → fatal, ambiguous join
    require the map is injective both ways
    // values() are FSD hosts[].hostname — PhysicalNodeId.host[] and the filter both use these
```

Prefer `ClusterDescriptor::get_host_id()`
([`PLAN_umd_cluster_descriptor_hostname.md`](PLAN_umd_cluster_descriptor_hostname.md)) —
the descriptor's `host_id`, which today holds the exact FSD name, so no filename parse.
The aisle-token alias is only a fallback while YAMLs lack the field, and the one-time
script that **writes** the field. See
[`PLAN_physical_node_id.md`](PLAN_physical_node_id.md) §8.

**Where it runs** — ControlPlane init, after live discovery, before filter:

1. `run_physical_system_discovery()` — live keys stay YAML basenames. Existing mock tests unchanged.
2. If FSD path is set **and** mock env is set: `alias = build_mock_to_fsd_host_alias(live, fsd)`.
3. Filter the FSD with **`alias.values()`** (real FSD hostnames), not `live.get_all_hostnames()`.
   `filter_factory_descriptor` matches `hosts[].hostname`; passing YAML basenames is a zero-overlap fatal.
4. Join `(host, tray, loc)` with **`host` compared by token** (or by walking `alias`), not by raw
   string. Do **not** rewrite FSD AsicIDs to live UMD ids before the mapper (§5.5 / §7.8). Copy
   `host_to_rank` from live onto the FSD PSD at provision (ranks only).
5. Live PSD keys are **not** rewritten. `my_host_name()` stays the basename. `LinkInfo` physical host
   comes from the expected (FSD) PSD, so it is the FSD hostname. Queries in these tests use FSD names
   (from `LinkInfo` or from the FSD).

Silicon (no mock): skip the alias. Filter is `live.get_all_hostnames()` after the implementation-plan
§5.2 FQDN/lowercase canonicalization. Position join uses that canonical string.

**Fatal cases for the alias** (these replace implementation-plan §5.3 case 10):

| Condition | Action |
|-----------|--------|
| Mock + FSD, a live key has no token | Fatal: "cannot join mock host to FSD" + the key. ClosetBox+FSD dies here, which is correct. |
| Token matches no FSD host | Fatal, same as live host absent from FSD |
| Token matches two FSD hosts | Fatal, ambiguous |
| Two live keys share a token | Fatal, injective |

Mock + FSD with a working alias **inits**. That is what makes §4 / §5 possible.

**Unit-test the extractor offline** (no device, in `test_physical_descriptor_builder.cpp`):

- FSD structured fields `C,1,2` → `c01u02`; hostname-only fallback `bh-glx-110-c01u02` → `c01u02`.
- Mock basename `..._bh-glx-c01u02_rank_0.yaml` → `c01u02`; hall number must not be required.
- `...yaml_3` uniqueness suffix still yields `c01u02`.
- ClosetBox basename → `nullopt`.
- Two FSD hosts with the same token → alias fatal.
- Alias values are FSD hostnames; feeding them to `filter_factory_descriptor` retains exactly the
  mapped hosts.

### 6.4 FSD-PSD vs mock PSD (no ControlPlane)

File: `test_fsd_psd_e2e.cpp`. Data-driven over the pair table. This slice can land as soon as the alias
helpers exist; it does not wait on `LinkHealth`.

1. Mapping YAML → live keys → tokens → FSD hostnames via §6.3.
2. Filter + `build_physical_descriptor_from_file`.
3. **Exact-match:** retained host count equals mapping size.
4. **Superset:** FSD larger than mapping; after filter, counts match; boundary cables gone (§7.10).
5. Every retained host has ≥1 ASIC; every ASIC has ≥1 neighbor (`IntegrationBuildFromQuietboxFsd`).
6. `tt-run --mock-cluster-rank-binding` → `run_physical_system_discovery` → position-join via token →
   `diff_physical_system_descriptors` (no AsicID overlay):
   - exact-match: `missing_asics` / `missing_links` empty, or list known mock holes in the test.
   - `extra_*` on the mock side is allowed and is not a downed link.
7. Near-miss `SC36_32x4_revC_subtorus_aisleD`: filter names the missing host.
8. QuietBox: FSD vs `4x_bh_quietbox_physical_desc.yaml` — no `tt-run`, names already match.

### 6.5 Not in this E2E v1

- Changing `get_local_discovery_hostname()` or ClosetBox hostname APIs.
- Rewriting FSD textprotos or cluster-desc filenames so the strings coincide.
- Dumping a golden PSD textproto next to each FSD.
- ControlPlane + FSD on ClosetBox / virtu (no FSD, no token).
- Silicon ControlPlane + FSD — same assertions as §4 / §5, but the alias is skipped; later hardware pass.

---

## 7. Risk-driven cases

These target specific holes in the current design. Each is `risk → test → expected`. Where the expected
result is "undecided", the design question must be closed before the test is written. Walk these one at
a time.

### 7.1 Intermesh downed set is **not** empty

An FSD-expected intermesh cable missing from live **must** appear in `get_downed_intermesh_links()` (and
`get_downed_links()` / `get_downed_links(LinkScope::InterMesh)`). Do not filter those misses against
post-pairing MeshGraph — that table is assigned-up only and would empty the set.

- Drop one intermesh cable from live. The record is in `downed_`, `scope == InterMesh`.
- Run on a ControlPlane that actually paired (SC20 E2E), not only a mapper that never called
  `generate_intermesh_connectivity()`.
- Construction order still matters: `LinkHealth` after pairing. Runtime guard if the MGD requested
  intermesh and `inter_mesh_connectivity_` is still empty.

Full assertion table: §3.1.

### 7.2 Cross-host cable is not always invisible

Cross-host is **intra** when both ends share a mesh (same-mesh, different hosts) and **intermesh** when
they do not. Intra cross-host already lands in `downed_` from the PSD diff. Intermesh cross-host now
does too (§7.1).

- Drop a same-mesh cross-host intra cable → `get_downed_links_between_hosts(a, b)` contains it,
  `scope == IntraMesh`.
- Drop a different-mesh cross-host cable → same query contains it, `scope == InterMesh`.
- A healthy host pair still returns empty. The query is not "always empty for intermesh".

### 7.3 `is_link_healthy` — keep the API test; drop the old "reports healthy" risk

The old hole (health derived as "expected and not in `downed_`") is closed by answering from
`live_present_`. Do **not** keep a risk that missing intermesh reports healthy. Do keep a test that
the API works:

- Expected and present → `true`. Expected and missing (intra or intermesh) → `false`. Never in the
  FSD → `out_of_range`. No FSD → `true`.
- **Works on mock cluster descriptors.** `is_link_healthy` is presence (FSD-expected vs live PSD). It
  does not call `cluster.is_ethernet_link_up` (`tt_cluster.cpp:1435`, which on mock just reads
  `ethernet_core_has_active_ethernet_link` from the YAML). Mock discovery builds the live PSD from
  that YAML; drop the cable in the live PSD (or omit it from the mock desc) and the API returns
  `false`. Assert this in the SC20 mock E2E, not only on silicon.
- STRICT + FSD: **do not use `is_link_healthy` (or ETH-up) to fatal.** Record the downed intermesh in
  LinkHealth and continue. See §7.5.

### 7.4 FSD is golden; extras in the PSD are OK; incompatible hosts and >10% down are not

The FSD is the expected graph. The live PSD may have **more** cables than the FSD — those are
`extra_links`, ignored by `LinkHealth`, not an error.

| Condition | Action |
|-----------|--------|
| Live has a cable the FSD does not | OK. Log at info. Not downed. |
| Hosts do not match (zero overlap / wrong FSD) | **Error.** Fatal. Same as implementation-plan §5.3 cases 2 and 4. |
| More than **10%** of FSD-expected connections are missing from live | **Error.** Log the fraction (`downed / expected`) and fail init. Exact lines: README §3.3. |
| At most 10% missing, hosts match | Init continues. Records are in LinkHealth. |

Tests: extras-only fixture does not fail; host-mismatch fixture fatals; a fixture with >10% of FSD
edges dropped logs the error and does not complete init; a 1-cable drop on a large FSD (well under
10%) inits and the record is in `get_downed_links()`.

### 7.5 STRICT + FSD: do **not** fatal on down — document intermesh in LinkHealth

**This is the load-bearing ControlPlane rule for FSD. Document it in the README, the
implementation plan, and the logs.** Without it, default STRICT dies at
`configure_routing_tables_for_fabric_ethernet_channels()` (`control_plane.cpp:1156` and `:1170`)
before `LinkHealth` exists, and the feature never runs.

When `has_factory_system_descriptor_path()` is true:

- Do **not** `TT_FATAL` on missing connections or short link counts in STRICT
  (`configure_routing_tables_for_fabric_ethernet_channels`).
- Skip the same class of check in `validate_mesh_connections`, channel trim,
  `generate_intermesh_connectivity` (assigned < requested), and
  `validate_requested_intermesh_connections` (strict `resolved == requested`) if they fire on FSD holes.
- Do **not** `TT_FATAL` in `confirm_local_downed_links` on ETH-up-but-missing.
- **Ignore `is_link_healthy` for abort.** Do not consult `is_link_healthy` or
  `is_ethernet_link_up` to decide whether to fatal. Health stays a presence diagnostic
  (and works on mock — §3.2 / §7.3). The abort path is closed.
- **Do** put the missing cables — **including intermesh** — into LinkHealth (`downed_`,
  `get_downed_intermesh_links()`). That record *is* the STRICT outcome. Pairing still
  assigns live / ETH-up only; the hole is documented here, not by crashing.
- Log each skipped STRICT fatal at **warning** (README §3.3):
  `STRICT + FSD: skipping fatal on missing/short connection … FSD in use — recording in LinkHealth, not fatal.`

Without an FSD, STRICT is unchanged (fatal on missing connections).

| Test | Expected |
|------|----------|
| FSD + one down **intra**, STRICT | init completes; record in `get_downed_intramesh_links()`; warning log; no fatal |
| FSD + one down **intermesh**, STRICT | init completes; record in `get_downed_intermesh_links()`; warning log; no fatal |
| FSD + down intermesh, STRICT, `is_link_healthy == false` | still inits — **do not** fatal on that false |
| No FSD + missing connection, STRICT | still `TT_FATAL` (today's behaviour) |

### 7.6 Per-rank divergence when the FSD host filter fails

Any per-rank fallback here diverges the topology: one rank maps on live while the others map on the FSD, so
`fsd_rerouting_active()` disagrees and any collective gated on it deadlocks. A local throw on the failing
rank hangs everyone else at the next MPI collective.

**Rule:** catch the local filter failure, all-reduce, then **every rank throws the same message**.
No per-rank fallback to live. Planned helper name: `agree_or_throw_fsd_host_filter`.
Implementation plan §5.3 case 9. Do not implement until asked.

Same `what()` on every rank (only all-reduced scalars):

```
FSD host filter is not identical on every rank (local_ok_min={}, host_checksum min={} max={},
fsd_fingerprint min={} max={}). Every rank fails together with this message.
No per-rank fallback to live mapping.
```

| Planned test | Setup | Expected |
|--------------|-------|----------|
| All-ok | every rank `local_ok=true`, same checksums | no throw |
| One-rank failure | rank 1 `local_ok=false`, others true | **every** rank throws; `what()` byte-identical; `local_ok_min=0` |
| Checksum disagreement | rank 0 checksum 1, others 2 | **every** rank throws; `what()` byte-identical; `host_checksum min=1 max=2` |
| Real ingest miss | force alias/filter miss on exactly one rank (wrong FSD path) | same string on every rank; no hang |

Do **not** `TT_FATAL` locally before the all-reduce — that hangs the other ranks. Launch on ≥2 ranks
(dual-T3K `tt-run`). Write the tests when implementing; not landed yet.

### 7.7 A whole FSD chip absent from live: **fail early**

Implementation plan §7.4: at **ControlPlane provision**, a chip in the filtered FSD with no live
counterpart throws **before** the `TopologyMapper` is constructed. `LinkHealth` is not involved. A missing
*cable* is §7.1; a missing *chip* is a wrong allocation.

This check does **not** run in `generate_rank_bindings`: placement must succeed when UMD discovery failed
(implementation plan §5.5). Assert that path inits / maps with the chip still in the FSD and absent from
any live PSD.

This is a **negative** test section: almost everything here asserts that something does *not* happen.

- FSD contains a chip absent from live → init **throws**. The message names **every** absent chip with
  `(host, tray, loc)`, not just the first — one pulled board is several ASICs.
- **The throw precedes the mapper.** Assert it is the absent-chip error and not a
  `verify_topology_mapping` fatal, a `validate_mesh_connections` fatal, or the >10%-downed error. Matching
  on the message is the point of this assertion: if the error arrives from `TopologyMapper` instead, the
  check landed in the wrong place, and every fixture with a missing chip becomes unusable.
- **`LinkHealth` never exists.** No downed records are produced for that chip's cables. There is nothing to
  query, so there is nothing to assert about `get_downed_links()` — assert instead that the throw happened
  during init and no ControlPlane is left constructed.
- **`verify_topology_mapping` is untouched.** All three checks stay fatal; no test may assert that Check 1
  or Check 3 tolerates anything. A fixture with an absent chip must never reach them.
- **No ChipId placeholder exists.** On a healthy FSD fixture, every mapped node resolves to a real ChipId
  and no two `FabricNodeId`s share one. ChipId 0 belongs solely to the real chip 0. This is the regression
  guard for reintroducing a sentinel.
- **Reliability mode is irrelevant.** STRICT and relaxed both throw. §7.5's "do not fatal on down" applies
  to cables only — assert that a missing chip still fails under relaxed policy, or §7.5 will get
  over-applied to this case.
- **Negative control (the load-bearing one).** An FSD with **no** absent chips inits cleanly, including a
  fixture with several *downed cables*. Otherwise a check that flags every chip as absent would pass every
  assertion above while making the whole feature unusable.
- **Multi-host: every rank throws the same message.** The check reads the filtered FSD and the live PSD,
  both rank-identical under global discovery, so a chip missing on rank 1 must fail rank 0 too, with
  identical `what()`. A single-rank abort is an MPI deadlock (§7.6).
- **`run_global_discovery` false** (§10.6): the live PSD is local-only, so a naive check sees every remote
  chip as absent and fatals on a healthy system. Assert the check is scoped to hosts the live PSD
  enumerates. This is the most likely way to ship a false fatal.

### 7.8 Mapper identity is `(host, tray, loc)`, not AsicID

Implementation plan §5.5. FSD synthesizes AsicIDs `1..N`; live discovery uses UMD unique ids. If the
solver keys on AsicID, those two graphs pick different equally-valid placements. Overlay-before-solve
is the wrong fix: it makes placement wait on UMD (place → recover → provision cannot).

- **Load-bearing.** Same FSD, two PSDs that differ **only** in AsicID values (builder `1..N` vs an
  injected UMD-like map on the same `(host, tray, loc)` graph). `TopologyMapper` produces the **same**
  `FabricNodeId` for every position. If this fails, FM and `tt-run` will disagree in production.
- `generate_rank_bindings` with an FSD path does **not** call `run_psd_discovery` and does **not**
  rewrite AsicIDs. Its placement equals ControlPlane's on the same FSD when live is healthy.
- A host missing from live: `generate_rank_bindings` still places it; ControlPlane fatals (§7.7).
- Diff of those two PSDs: `matches()` on cables (positional join). No overlay step.
- `#54752` already instantiates the solver on the position tuple with no library change — keep those
  tests; they are the mechanical proof the mapper change is legal.
- Duplicate `(host, tray, loc)` → fatal.
- Documented, not blocking: lexicographic host order (`host10` before `host2`). Both FSD and live use
  the same canonical string, so they still agree. Do not sort by AsicID to "fix" it.

### 7.9 Dangling references across `refresh()`

`ControlPlane::get_downed_links()` returns a reference into `link_health_->downed_`, and every index holds
`const LinkInfo*` into the same vector. `refresh_connectivity_diff()` invalidates all of it.

- Hold the reference, call `refresh_connectivity_diff()`, observe under ASAN.
- Expected: documented invalidation, and no internal caller holds across a refresh. `locally_unhealthy_` holds
  copies, so it is safe — assert that it stays valid.

### 7.10 Host filter joins on a key that is often not a hostname

The FSD is filtered to the allocated hosts by hostname, but the live PSD's host keys are the mock descriptor
basename in mock mode, `hostname_<rank>` when hostnames collide, and a raw `gethostname()` (short or FQDN)
otherwise. A naive intersection matches nothing and the failure looks like "wrong FSD". Implementation
plan §5.2 / §5.3; mock join is §6.3.

Offline, in `test_physical_descriptor_builder.cpp`:

- Canonical match across FQDN vs short name, case differences, and a `_<rank>` suffix.
- Two live hosts canonicalizing to the same string → fatal (ambiguous join, would attach one machine's
  cables to another).
- Duplicate hostnames inside the FSD among the retained hosts → fatal. `filter_factory_descriptor` keeps
  every matching index today, so one requested name silently pulls in two machines.
- Requesting a host absent from the FSD reports **all** missing names, not the first.
- Zero overlap → its own error message, distinct from "some hosts missing".
- Retained hosts spanning more than one connected component → detected.
- **Boundary cables:** a cable from a retained host to a filtered-out host must end up in `extra_links` and
  never in `missing_links`. If this regresses, every job reports the whole pod boundary as downed.
- **`host_to_rank`:** build an FSD whose host order is reversed relative to MPI rank order, filter, copy
  ranks from live (not AsicIDs), and assert ranks come from live. Filtering densely renumbers and the
  builder assigns rank = FSD index, so without the copy every rank is wrong.

On device / mock E2E (§6.3):

- Mock + FSD **with** a working aisle-token alias → init. Mock + FSD **without** a token (ClosetBox
  basename) → clean fatal, not a zero-overlap error.
- Multi-host: every rank derives the same alias table; assert with an all-reduce (this is the check that
  turns §7.6 into an error instead of a hang).

### 7.11 `planes_lost` + unused downed links on downgraded routing planes

**Decision.** A PSD has no direction, and a downed channel has no live plane index — do **not** call
`get_eth_chan_direction` / `get_routing_plane_id`. After fabric **downgrades** routing planes (row/col
min + trim), ignore those holes for rerouting / `planes_lost`. Still document them in LinkHealth:
`get_unused_downed_links()`. This is the **only** unused set — intermesh holes are active downed links
(§7.1). Implementation plan §3.3.

```
downgraded(node, dir)  ⇔  live_planes < expected_planes
planes_lost(node, dir) = count of *active* downed_ records on (node, src_direction)
                       = 0 when that direction was downgraded
```

`live_planes` comes from the trimmed+merged ETH-chan map (after
`collect_and_merge_router_port_directions_from_all_hosts`). `expected_planes` comes from MeshGraph
golden counts. Classify is a second pass after plane init — not at `LinkHealth` ctor.

| Test | Expected |
|------|----------|
| Direction downgraded (live_planes < expected); one FSD intra cable missing | record in `get_unused_downed_links()`, **not** in `get_downed_links()` / `get_downed_intramesh_links()`; `planes_lost == 0`; `fsd_rerouting_active()` false if that was the only hole |
| Direction **not** downgraded; one intra cable missing | record stays in `downed_`; `planes_lost == 1` |
| Intermesh missing (no plane downgrade for that scope) | still in `get_downed_intermesh_links()` — unused set is intra / downgraded-plane only |
| Trimmed extra **live** channel | not downed, not unused |
| After classify, every rank | `unused_downed_` identical (classify after merge) |
| `is_link_healthy` on an unused-plane hole | still `false` (presence). Do not use it to abort. |
| No `configure_routing_tables` / CUSTOM | no classify; intra holes stay in `downed_` |

Do not invent a `routing_plane` on `LinkInfo`. Split is by `(node, dir)` downgrade, not per-cable plane.

---

## 8. Open questions that block writing these tests

1. **`LinkHealth` cannot be tested without hardware today.** `TopologyMapper` requires `tt::Cluster` and a
   `DistributedContext`, so `test_link_health.cpp` is a device test, not a unit test. Options: run it under
   `TT_METAL_MOCK_CLUSTER_DESC_PATH`, or split the diff-to-records logic into a free function that takes
   (expected PSD, live PSD, asic→node resolver) and unit-test that offline. The second option makes most of §3
   runnable with no device and is worth doing before the tests are written.
2. Whether `confirm_local_downed_links` should be part of `construct_link_health_after_intermesh()`. The
   implementation plan's snippet calls it there; the code currently in `control_plane.cpp` does not.
3. Hostname join is specified in §6.3. STRICT + FSD is specified in §7.5. Filter agreement is specified
   in §7.6. Unused-plane holes / `planes_lost` is specified in §7.11. Mapper identity is specified in
   §7.8. §7.1–§7.8 and §7.11 are closed.
4. **The §10 findings are not designed yet.** Implementation plan §10 records them; §10 here records the
   tests. Three of them (`planes_lost(NONE)`, whether `refresh_connectivity_diff()` keeps existing,
   behaviour when `run_global_discovery` is false) are decisions the test cannot invent.
5. **No log-capture fixture exists.** §10.7 and §10.8 assert on warning counts and wording. Either add one
   or expose test-only counters.

## 9. Not tested in v1

- Degraded-but-present links (`EthernetMetrics` classification) — presence only, by contract.
- Reroute behaviour on top of this API.
- Acting on `extra_links` (they are allowed; we only assert they are not treated as downed — §7.4).
- A second pairing pass over FSD-expected (including down) intermesh cables.
- **Per-boundary intermesh shortfall** (requested vs. resolved as a count) — still deferred; v1 reports
  the missing **cables** in `get_downed_intermesh_links()` instead (implementation plan §4.1).
- Joining pairing hashes into `LinkHealth` — explicitly rejected (implementation plan §4.2). Downed
  intermesh is FSD-vs-live; pairing stays in ControlPlane.
- Adding channel ids to `RouterEdge`.

---

## 10. Tests for the leftover review findings (implementation plan §10)

Implementation-plan §10 records the findings from the plan review that are **not** yet designed. Each one
still needs a test, and for most of them the test is what pins the decision. Numbering follows §10 so the
two documents stay joinable. Same `risk → test → expected` form as §7.

**Two prerequisites before any of this is writable.** Several cases below assert on *log output* (warning
fired once, warning names a host, no duplicate warning). There is no log-capture fixture in the fabric
tests today — add one, or convert those assertions to counters exposed for test only. And §8.1 applies
here too: anything that needs a `TopologyMapper` needs a device or a mock cluster descriptor.

### 10.1 #8 The silent all-clear: guard on the FSD PSD, not the RTOption

The worst failure this feature can have is `has_factory_descriptor() == true` with an empty downed set for
the wrong reason. It reads exactly like a healthy cluster.

- **Universal invariant, assert it in every ControlPlane fixture** (§4, §5, §6, and each §7 case):
  `has_factory_descriptor() == (get_link_health() != nullptr)`. Cheap, and it catches the guard
  disagreeing with reality no matter which path produced the disagreement.
- **The load-bearing case.** Build a fixture where the FSD contains **one cable the mock live PSD does
  not** — a known, deliberate miss. `get_downed_links()` must contain it. An **empty** set here is the
  signature of the bug: it means the mapper was built on the live PSD, so `expected == live` and the diff
  is vacuous. Without a known-miss fixture, no ControlPlane test can distinguish "healthy" from
  "comparing the live PSD against itself".
- Assert the mapper's PSD is the FSD whenever `link_health_ != nullptr` — directly if a getter exists, or
  indirectly via the known-miss fixture above.
- Distinguish the three states explicitly, since two of them have empty downed sets: no FSD
  (`has_factory_descriptor()` false, `get_link_health()` null); FSD + clean live (true, non-null, empty);
  FSD + degraded (true, non-null, non-empty).
- **Unknown hostname in the filter** (`build_physical_descriptor_from_file` throws): assert init fails
  with a message naming the host, on **every** rank with the same `what()` (§7.6), and that it does not
  instead land in a state where `link_health_` is null but `has_factory_descriptor()` is true.

### 10.2 #9 Unmapped ASICs must resolve to `nullopt`, not fatal

`is_mapped == false` means the MGD solve never placed this ASIC — an FSD that describes more chips than the
mesh graph does. (It no longer means "absent from live": §7.7 fails early on that.) `refresh()` still walks
these ASICs, so the resolver must tolerate them.

- Two fixtures, both must return `nullopt`: (a) an AsicID absent from `asic_id_to_mapping_` entirely,
  (b) an AsicID present but with `is_mapped == false`. (b) is the one the existing throwing getter fatals
  on separately (`topology_mapper.cpp:181`), so a `find` that only handles (a) still aborts refresh.
- The throwing `get_fabric_node_id_from_asic_id` must **still fatal** in both cases. Assert it — the fix
  is a new accessor, not a relaxation of the old one.
- Integration: an FSD containing an unplaced ASIC that also has a downed cable → `refresh()` completes,
  the record has `logical_resolved == false` and `scope == Unknown`, and it appears in the physical façade
  only. §3 already covers the record's shape; what is new here is that **refresh does not abort**.

### 10.3 #10 `planes_lost` is intra-only — prove it, don't infer it

It currently comes out right only because intermesh records carry `src_direction == NONE`.

- Drop **only** an intermesh cable. `planes_lost(exit_node, dir)` is 0 for every real direction, while
  `get_downed_intermesh_links()` is non-empty. Assert those together: a bare 0 must never be the only
  signal, and there is no `intermesh_planes_lost` query.
- **`planes_lost(node, RoutingDirection::NONE)`** — decide and assert. `NONE` is a valid enum value, so an
  implementation that keys a map on direction will happily return the intermesh count from a query
  documented as intra-only. Either return 0 or throw; do not leave it implementation-defined.
- Mixed node: one intra cable down in direction `E` **and** one intermesh cable down on the same node →
  `planes_lost(node, E) == 1` exactly. The intermesh record must not contaminate it.
- Interaction with §7.11: unused-plane holes are already excluded from `planes_lost`. Assert the two
  exclusions compose — an intermesh miss and an unused-plane intra miss on the same node still give
  `planes_lost == 0`.

### 10.4 #11 `refresh_connectivity_diff()` cannot change its answer

It re-diffs the same two stored descriptors. The test depends on which way the design goes, so write the
test for the decision:

- **If it stays a no-op:** call it after init, assert the downed set is byte-identical (same size, same
  records, and per §7.9 the same `downed_.data()` if no reallocation is expected), and assert the README
  no longer calls it "recompute". A test that just calls it and passes is worse than no test — it implies
  the call does something.
- **If it gains a fresh-PSD overload:** mutate a copy of the live PSD, pass it, and assert the downed set
  changes accordingly, every index is rebuilt, and no `const LinkInfo*` from before the call is
  dereferenced afterward (§7.9).
- Either way: assert `locally_unhealthy_` is recomputed and not stale or double-appended after the call —
  `confirm_local_downed_links()` runs inside it.

### 10.5 #12 `missing_links` shape and the duplicate-destination merge

`AsicTopology` is `unordered_map<AsicID, vector<pair<AsicID, vector<EthConnection>>>>`, so **one source
can list the same destination in more than one entry**. A diff that compares entry-by-entry reports false
misses. All of this is offline in `test_physical_system_descriptor_diff.cpp`.

- Golden has source A listing B twice (chan 0 in one entry, chan 1 in another); candidate has the same two
  channels merged into a **single** entry for B. Expected: `matches()`, empty `missing_links`. This is the
  false-miss case and it is the reason this test exists.
- Same split golden, candidate missing only chan 1 → exactly **one** missing record, for chan 1. Not two,
  not zero.
- Both directions of the same cable split across entries → still one record per endpoint (§2 already
  asserts one-per-endpoint; here the input shape is adversarial).
- `missing_links` is **flat** — no host key. A cross-host cable appears once per endpoint, not once per
  host. Assert the type is `AsicTopology`, not `unordered_map<host, AsicTopology>`, so the plan comment
  and the code agree.
- Flattening is keyed on `AsicID` alone: build a delta whose two hosts each contribute edges, and assert
  no entry is dropped or double-counted by the merge.

### 10.6 Global completeness is conditional on `run_global_discovery`

The rank-equality assertions in §5 depend on it. If it is false the PSD is local-only and those
tests would fail for a reason that has nothing to do with `LinkHealth`.

- With global discovery **off**: assert the documented behaviour — either `link_health_` is not constructed
  or `get_downed_links()` is explicitly local-only — and that init does not silently claim global results.
- Every rank-equality test states this dependency in a comment and skips (not fails) when the precondition
  does not hold. A confusing failure here will be misread as a `LinkHealth` bug.

### 10.7 #4 residual — one warning per shortfall, not two

Under relaxed policy, `validate_requested_intermesh_connections` **already** warns on
`requested_channels > resolved`; §7.5 adds a "skip + warning" for the FSD case. Both can fire for the same
boundary.

- Relaxed policy + FSD + an intermesh shortfall: the shortfall is reported **once**. Two warnings for one
  boundary is a log-noise regression, and on a large system it is the difference between a readable init
  log and an unusable one.
- Strict policy + FSD: the fatal is skipped (§7.5) and the warning fires; strict policy **without** FSD
  still fatals (`control_plane.cpp`, `validate_requested_intermesh_connections`). Cite by function name —
  the line numbers in that file drift.

### 10.8 #5 residual — a peer host dropping out must not produce one warning per cable

`remove_unresolved_nodes` / `erase_one_sided_connections` erase **every** edge to a host that failed to
join discovery, and each of those is then "FSD-expected, missing from live, local ETH up".

- Mock live PSD with one peer host's ASICs removed entirely; FSD intact. Init completes, and the warning
  count is **O(hosts), not O(cables)** — assert an upper bound, e.g. one per peer host, rather than the
  exact text.
- The message names the absent peer host and does not say the descriptor and cluster "disagree": the
  descriptor is right and the peer is simply absent. Wrong wording here sends whoever is debugging a
  bringup to the wrong system.
- Every cable to that host is still in `get_downed_links()` — suppressing the *logs* must not suppress the
  *records*.
