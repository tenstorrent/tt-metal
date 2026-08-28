# Plan: stable physical node id `(host, tray, loc)` as a packed POD

**Status:** Implementation plan — **separate change** from FSD / downed-links
**Why split:** Fabric Manager and `generate_rank_bindings` need a stable mapper identity whether or not UMD discovery has run. That is independent of `LinkHealth`. Land this first; the FSD-backed downed-links plan ([`PLAN_fsd_solve_and_downed_links.md`](PLAN_fsd_solve_and_downed_links.md) §5.5) then consumes it.
**Related:** [tt-metal#54752](https://github.com/tenstorrent/tt-metal/pull/54752) (solver is already generic over `GlobalNode`). Production uses a POD, not a `std::tuple` / `std::string`.

---

## 1. Problem

The FSD does not encode ASIC ids. `physical_descriptor_builder` synthesizes `1..N` in `(host_id, tray, loc)` file order. Live discovery uses UMD `chip_unique_id` as `AsicID`. The topology solver erases nodes to dense indices **in iteration order**, which selects among equally valid mappings. Different id spaces → different legal placements. Overlaying UMD ids onto the FSD graph before the solve makes placement wait on discovery, which the place → recover → provision flow cannot do.

We want one value, computed from physical position, that FSD-built and live-discovered descriptors both produce for the same chip.

---

## 2. Decision

`PhysicalNodeId` is a **POD**: fixed-size canonical host bytes + tray + loc. No host hash. Same canonical host string + same tray + same loc ⇒ same id on every path. Fully reversible.

```
PhysicalNodeId = {
    host[kPhysicalHostNameLen],  // NUL-padded canonical hostname, not a hash
    tray,                        // 16 bits
    loc                          // 16 bits
}
```

Do **not** pack the host as `hash32` into a `uint64`. After tray and loc take 32 bits, only four characters would fit; `bh-glx-110-c01u02` is 18. A 128-bit pack is still only 12 host chars. The hash was the only way to stay in 64 bits, and it is the conversion this plan drops.

Do **not** use FSD `host_id` (0..N-1 in file order): live discovery has no FSD index.

Do **not** use `std::string` inside the node (heap, allocator-sensitive compare). Do **not** use `std::array` — the host is a fixed buffer we never pass around on its own; a C array is the pack.

UMD unique ids stay on `ASICDescriptor::umd_unique_id` / `Cluster::get_unique_chip_ids()` — they are not mapper identity.

**Mock + FSD is the load-bearing case.** The name that goes into `host[]` is the **exact** FSD / OS hostname. Mock obtains it from `ClusterDescriptor::get_hostname()` ([`PLAN_umd_cluster_descriptor_hostname.md`](PLAN_umd_cluster_descriptor_hostname.md)), not from the YAML filename. Until that UMD field is filled on the assets, §8.3 is the fallback. If mock and FSD pack different `host[]` bytes, the solve diverges again.

---

## 3. Encoding

```cpp
// tt_metal/api/tt-metalium/experimental/fabric/physical_node_id.hpp
namespace tt::tt_metal {

// POSIX HOST_NAME_MAX-class. First DNS label after canonicalization.
// Today's FSD names: "bh-glx-110-c01u02" (18), "sjc1-tt-qb-01" (13).
inline constexpr std::size_t kPhysicalHostNameLen = 64;

struct PhysicalNodeId {
    char host[kPhysicalHostNameLen]{};  // NUL-padded C buffer, not std::array / std::string
    TrayID tray{0};
    ASICLocation loc{0};

    friend bool operator==(const PhysicalNodeId&, const PhysicalNodeId&) = default;
    friend auto operator<=>(const PhysicalNodeId&, const PhysicalNodeId&) = default;
};

// Lowercase, strip a trailing "_<rank>" when hosts are not unique, take the first DNS label.
std::string canonical_host_for_node_id(std::string_view host, bool hosts_unique = true);

PhysicalNodeId make_physical_node_id(
    std::string_view host,
    TrayID tray,
    ASICLocation loc,
    bool hosts_unique = true);

// host is the NUL-trimmed canonical string stored in the id.
struct PhysicalNodeFields {
    std::string host;
    TrayID tray{0};
    ASICLocation loc{0};
};
PhysicalNodeFields decode_physical_node_id(PhysicalNodeId id);

}  // namespace tt::tt_metal
```

`make_physical_node_id`:

1. `canonical = canonical_host_for_node_id(host, hosts_unique)`.
2. **Fatal** if `canonical.empty()` or `canonical.size() >= kPhysicalHostNameLen` (do not truncate).
3. **Fatal** if `tray` or `loc` does not fit in 16 bits.
4. Write `canonical` into `host[]`, leftover bytes `'\0'`. Write tray and loc.

`PhysicalNodeId{}` (all zeros) is unset. `make_*` must never return it (empty host already fatals).

C++20 defaulted `==` / `<=>` compare a C array member element-wise, so the struct stays a POD with no handwritten compare. Do not pass `id.host` as a decaying `char*` — keep the buffer inside the id; callers who need a string use `decode_physical_node_id` or `string_view{id.host}`.

Provide `std::hash<PhysicalNodeId>` that hashes the **whole POD bytes**. That is a container hash only — it is not identity. Do **not** use `std::hash<std::string>` on the host as the node id.

`make_physical_node_id` always canonicalizes. Callers must not pass a mix of FQDN and short names for the same machine — the FSD host-filter canonicalization (downed-links plan §5.2) is the same function. Put `canonical_host_for_node_id` here so there is one implementation.

**Collision:** two different triples cannot produce the same id unless they share the same canonical host bytes, tray, and loc. Duplicate `(host, tray, loc)` in a descriptor is already fatal. There is no hash-collision case.

**Sort order:** `operator<=>` is host bytes, then tray, then loc. Lexicographic hosts (`host10` before `host2`) are a locality-heuristic issue, **not** a stability issue: FSD and live pack the same string, so they still agree. Do not sort by AsicID to "fix" it. Rack order, if wanted later, is `(aisle, rack, u, tray, loc)` from the FSD.

---

## 4. Where it is used

One utility, three producers, mapper as the consumer. Do not invent a second encoding.

| Site | Today | After |
|------|--------|--------|
| `physical_descriptor_builder` | `next_id++` from 1 | `unique_id` / graph key = `make_physical_node_id(fsd_hostname, tray, loc)` |
| `physical_system_discovery` | `AsicID{umd_chip_unique_id}` as graph key | graph key = `make_physical_node_id(host_for_node_id(...), tray, loc)`; keep UMD id in `umd_unique_id` |
| Mock cluster YAML | basename as live host key; `chip_id << 32` as unique id | **host for the id is the FSD hostname** (§8), never the YAML basename |
| `TopologyMapper` adjacency / solver `GlobalNode` | `AsicID` from the descriptor | `PhysicalNodeId` |
| `Cluster::get_unique_chip_ids()` | UMD ids | **unchanged** |

`PhysicalSystemDescriptor::get_asic_id(hostname, tray, loc)` already exists. Implement it as `make_physical_node_id` + lookup, so there is one construction path.

`host_for_node_id` is defined in §8. On silicon with no mock, it is `canonical_host_for_node_id(live_key)`. On mock + FSD it is the aliased FSD hostname.

---

## 5. Type: new strong type vs reuse `AsicID`

**New `PhysicalNodeId` struct.** `AsicID` is already the live UMD unique id in discovery, serialization, exit-node tables, and `verify_topology_mapping` against `cluster.get_unique_chip_ids()`. Overloading it to mean "position pack" silently breaks Check 1/3 (packed id ≠ UMD id). A struct also **cannot** be stuffed into `AsicID` — the earlier "store the uint64 in AsicID for a first slice" path is gone.

Mapper-facing maps become `PhysicalNodeId`. `MappedChipInfo` holds both:

- `physical_node_id` — solver identity, packed position
- `asic_id` — UMD unique id when known (`nullopt` during FSD-only placement)

Do **not** instantiate the solver on a `tuple<string, TrayID, ASICLocation>` in production. #54752 proves it is legal; the POD is the production node.

---

## 6. File changes

**New:** `tt_metal/api/tt-metalium/experimental/fabric/physical_node_id.hpp` + `tt_metal/fabric/physical_node_id.cpp` (canonicalization + `make` / `decode` / hash). Add the header to `TT_METAL_PUBLIC_API`.

**`physical_descriptor_builder.cpp`:** drop `next_id++`. Every ASIC's graph key is `make_physical_node_id(hostname_of(host_id), TrayID{tray}, ASICLocation{loc})`. Duplicate `(host, tray, loc)` is fatal while filling `key_to_unique_id`.

**`physical_system_discovery.cpp`:** when creating `ASICDescriptor`, set the graph key from position (`host_for_node_id`, §8) and `umd_unique_id` from `chip_unique_ids`. Cross-host gather still carries UMD ids on the wire if they do today — translate to packed ids on ingest using the peer's **resolved** host + tray + loc from the payload, not by hashing the UMD id.

**`topology_mapper.cpp`:** build `PhysicalAdjacencyMap` / `hostname_to_asics` / `asic_positions` from packed ids (host/tray/loc are on the id). `verify_topology_mapping` compares **UMD** ids to `cluster.get_unique_chip_ids()`, never the packed id. ChipId backfill: match `umd_unique_id`, not packed id.

**Tests:** `tests/tt_metal/tt_fabric/fabric_router/test_physical_node_id.cpp` (offline).

No Mesh Graph Descriptor change. No FSD protobuf change. FSD still has no ASIC ids.

---

## 7. Tests

- Same `(host, tray, loc)` → same id, including FQDN vs short name vs case after canonicalization.
- `_rank` suffix stripped iff `hosts_unique == false`.
- Different loc, tray, or host → different id.
- `decode` restores the canonical host string, tray, and loc.
- Host that does not fit in `kPhysicalHostNameLen - 1` → fatal (do not truncate).
- Tray or loc `> 0xffff` → fatal.
- Empty / unset id is all zeros; `make_*` never returns it.
- Golden vector: one fixed host/tray/loc → exact `host[]` bytes + tray + loc.
- **Stability (the load-bearing one):** build one graph from FSD and one from a live-style descriptor with UMD-like `umd_unique_id`s; adjacency maps keyed by `PhysicalNodeId` are equal. `TopologyMapper` (or the solver on those maps) assigns the same `FabricNodeId` per position.
- **Mock + FSD (the other load-bearing one):** descriptor with `hostname: bh-glx-110-c01u02` packs the same id as the FSD builder. Filename-only (no field) must **not** equal that id. After the UMD field is filled, SC36 rank 15 packs `bh-glx-110-d10u20`, not the `120` in the filename.
- Builder unit test: QuietBox / a tiny in-memory FSD, two ASICs, ids equal `make_physical_node_id` of their FSD hostnames, not `1` and `2`.
- Discovery: graph key ≠ `umd_unique_id` on silicon when UMD ids are large; `get_asic_id(host, tray, loc)` returns the packed id.

Existing mapper tests that assert one specific mapping may need a re-baseline (iteration order follows packed ids). That is expected; it is the same cost #54752 recorded for tuple nodes.

---

## 8. Mock hostnames come from the UMD cluster descriptor

If mock and FSD pack different `host[]` bytes, the mapper sees two graphs and we are back to the AsicID-order bug. The hostname we pack is **exactly** FSD `hosts[].hostname` / silicon `gethostname()` (`bh-glx-110-c01u02`). Hall stays.

**Better source — UMD field, not the filename.** A cluster descriptor is one host. Add `hostname:` to that YAML and query it:

[`PLAN_umd_cluster_descriptor_hostname.md`](PLAN_umd_cluster_descriptor_hostname.md)

```yaml
hostname: bh-glx-110-c01u02    # exact FSD / OS name
arch:
  ...
```

```
desc.get_hostname()  →  optional<string>   // UMD
get_local_discovery_hostname(desc):
    if desc.get_hostname():  return *that
    if mock env set:         return filename basename   // ClosetBox / unfilled
    return get_host_name()
```

Land the UMD PR first (optional field — old YAMLs still work). Metal prefers `get_hostname()` and falls back to the basename when it is absent, so existing tests stay green.

**TODO:** add `hostname:` to **all** cluster descriptors in tt-cluster-descriptors ([UMD plan §7](PLAN_umd_cluster_descriptor_hostname.md)). Incremental fill is fine. Do not parse a name out of the filename at runtime. Do not rename the YAML files.

### 8.1 What each string is (today, before the UMD field is filled)

| Side | What it is | Example |
|------|------------|---------|
| **Hostname (what we pack)** | FSD `hosts[].hostname` | `bh-glx-110-c01u02` |
| Mock YAML **filename** | asset-repo file name, **not** a hostname | `SC20_…_bh-glx-c01u02_rank_0.yaml` |
| Live PSD key today | `get_local_discovery_hostname()` = basename | that filename |
| Silicon | `gethostname()` | `bh-glx-110-c01u02` |

QuietBox already matches (`sjc1-tt-qb-01`). BH mock YAMLs have no hostname field yet.

### 8.2 Filling the UMD field (asset script, once)

Not a runtime join. For each BH YAML next to an FSD:

1. Filename token: last `[a-z][0-9]{2}u[0-9]{2}` before `_rank_` / `.yaml` → `c01u02`.
2. Unique FSD `Host` with matching `aisle` / `rack` / `shelf_u`.
3. Write that host's `hostname` into the YAML (`bh-glx-110-c01u02`).

SC36 file `bh-glx-120-d10u20` vs FSD `bh-glx-110-d10u20` is why the filename is not the name: the field gets the FSD string.

### 8.3 Fallback until every FSD-paired YAML has the field

If `get_hostname()` is empty and mock + FSD are both on, use the aisle-token alias (testing plan §6.3) so `host[]` is still the FSD hostname. Delete the alias once the descriptor pin is complete. ClosetBox / no-FSD keeps the basename fallback.

```
host_for_node_id(desc, fsd):
    if desc.get_hostname():
        return canonical(*desc.get_hostname())
    if mock and fsd:
        return alias[basename]     // FSD hostname; temporary
    return canonical(live_key)
```

| Path | String that goes into `host[]` |
|------|--------------------------------|
| FSD builder | `hosts[].hostname` after canonicalization |
| Silicon | canonical `gethostname()` / `desc.get_hostname()` (same string after UMD stamp) |
| Mock + field set | `desc.get_hostname()` — must equal the FSD name |
| Mock, field empty, + FSD | alias fallback |
| Mock, no FSD | basename (ClosetBox) |

### 8.4 What does not change

- ClosetBox tests until/unless those YAMLs get a hostname field.
- FSD files. Cluster-desc **filenames**.
- `LinkInfo` physical host still comes from the expected (FSD) PSD.

---

## 9. Non-goals

- Changing UMD `chip_unique_id` or Cluster ChipId
- Hash-joining pairing / `ExitNodeConnection` (rejected for `LinkHealth`)
- Downed-link reporting (separate plan)
- Sorting hosts into rack order
- Making `PhysicalNodeId` a protobuf field of the FSD
- Packing the host as `hash32` or stuffing the POD into `AsicID`
- Treating the mock filename token (`bh-glx-120-d10u20`) as a hostname

---

## 10. Sequence

1. UMD hostname field ([`PLAN_umd_cluster_descriptor_hostname.md`](PLAN_umd_cluster_descriptor_hostname.md)) — separate PR.
2. Utility + tests (no callers). Mock golden uses `desc.get_hostname()`, not the filename.
3. Builder switches off `1..N`. Offline test: FSD-only ids are the POD of the FSD hostname.
4. Mapper re-keys adjacency through the utility. FSD vs live agree when live uses `host_for_node_id` (§8.3).
5. Discovery writes packed graph keys and keeps UMD on `umd_unique_id`.
6. **TODO:** fill `hostname:` on **all** tt-cluster-descriptors YAMLs (UMD plan §7); delete the aisle-token fallback when FSD-paired files are done.
7. Downed-links / FSD ControlPlane work then diffs and maps without overlay.

Slices 2–4 are the PhysicalNodeId PR series. Slice 1 can land first or in parallel. Slice 6 is assets.
