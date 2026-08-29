# Plan: hostname on the UMD cluster descriptor

**Status:** Implementation plan — **UMD change**, consumed by PhysicalNodeId / FSD mock
**Repos:** [tt-umd](https://github.com/tenstorrent/tt-umd) (API + YAML), then [tt-cluster-descriptors](https://github.com/tenstorrent/tt-cluster-descriptors) (fill the field), then tt-metal (read it). Land UMD first.
**Consumers:** [`PLAN_physical_node_id.md`](PLAN_physical_node_id.md) §8 (problem / solution / checklist), [`PLAN_downed_links_testing.md`](PLAN_downed_links_testing.md) §6.3.

---

## 0. Problem, then solution

### Problem

A cluster descriptor is one host's chips. It has **no hostname field**. Metal mock discovery therefore keys on the YAML **filename** (`get_local_discovery_hostname` = basename of `TT_METAL_MOCK_CLUSTER_DESC_PATH`). That string is not the FSD / OS hostname.

| Side | String | Example |
|------|--------|---------|
| FSD / silicon (what we must pack) | real hostname | `bh-glx-110-c01u02` |
| Mock YAML filename | asset name, not a hostname | `SC20_…_bh-glx-c01u02_rank_0.yaml` (hall dropped) |
| SC36 filename token | hall **wrong** | file `bh-glx-120-d10u20` vs FSD `bh-glx-110-d10u20` |

If mock and FSD pack different `host[]` bytes, `PhysicalNodeId` graphs disagree and the solver sees two topologies. Parsing a name out of the filename cannot work (SC20 omits hall, SC36 disagrees). Do **not** parse `bh-glx-c01u02` at runtime.

### Solution

Add optional top-level `hostname:` to the UMD cluster descriptor and query it. Value is the **exact** FSD / OS name. Metal prefers `desc.get_hostname()`, falls back to today's basename if absent. Live discovery stamps `gethostname()`. Field is optional forever — old YAMLs still load; old UMD ignores the unknown key; existing mock tests stay green.

```yaml
hostname: bh-glx-110-c01u02
arch:
  ...
```

**TODO:** write that field on **every** YAML in tt-cluster-descriptors (~230 files). Does not block this UMD PR. Full checklist: [`PLAN_physical_node_id.md`](PLAN_physical_node_id.md) §8.3.

---

## 1. Model

One `ClusterDescriptor` YAML is **one host**. Chips, trays, and `asic_locations` are on that host. Cross-host ETH is `ethernet_connections_to_remote_devices` keyed by the peer's **unique chip id**, not by hostname.

So hostname is a **descriptor-level** field, not per-chip:

```yaml
hostname: bh-glx-110-c01u02
arch:
  0: blackhole
  ...
```

That value is the same string as FSD `hosts[].hostname` and as silicon `gethostname()` (first label / short name as the machine is configured). Hall stays (`110`). We do not invent a shorter token.

| Path | Who writes `hostname` | Who reads it |
|------|------------------------|--------------|
| Live silicon | `TopologyDiscovery::fill_cluster_descriptor_info` stamps `gethostname()` | metal discovery, `serialize()` |
| Mock YAML | author / asset fill (exact FSD name) | `create_from_yaml` → `get_hostname()` |
| Recapture | `serialize_to_file` writes whatever is on the object | next mock YAML gets the live name for free |

Absent field = old descriptor. That is the compatibility path, not a temporary hack.

### 1.1 Backward compatibility (locked)

| Combination | What happens |
|-------------|--------------|
| New UMD + old YAML (no `hostname:`) | `get_hostname() == nullopt`. Parse succeeds. |
| Old UMD + new YAML (`hostname:` present) | Parser only reads known keys; the new field is ignored. File still loads. |
| New metal + old YAML | `get_local_discovery_hostname` uses filename basename / `gethostname()` — **today's behavior**. |
| New metal + new YAML | uses `get_hostname()`. |
| Old metal + new YAML | still keys on the filename. ClosetBox / existing mapper tests unchanged. |

Do **not** put `hostname` in the schema `required` list. Do **not** fatal in UMD or metal because the field is missing. Do **not** rewrite filenames. `serialize()` omits the key when unset so existing golden dumps do not change unless discovery stamped a name.

Adding the field to assets is therefore safe to land in any order relative to metal.

---

## 2. UMD API

```cpp
// umd/device/cluster_descriptor.hpp
class ClusterDescriptor {
public:
    // Empty when the YAML omitted the key and discovery did not stamp one.
    const std::optional<std::string>& get_hostname() const;

    // Reject empty, whitespace, path separators, length >= 64.
    void set_hostname(std::string hostname);

    // ...
private:
    std::optional<std::string> hostname_;
};
```

**Query from metal** (already has the descriptor):

```cpp
auto* desc = cluster.get_cluster_desc();            // llrt Cluster
// or the ClusterDescriptor& already passed into run_local_discovery
if (const auto& host = desc->get_hostname(); host.has_value()) {
    // this is the name
}
```

Do **not** add a second metal wrapper type. `tt::umd::Cluster::get_cluster_description()->get_hostname()` is the query. Nanobind: bind `get_hostname` / `set_hostname` next to the other `ClusterDescriptor` getters (`py_api_topology_discovery.cpp`).

`create_mock_cluster(...)` stays hostname-less (`nullopt`) unless a new optional argument is passed. Simulation / unit mocks do not need a galaxy name.

---

## 3. YAML

**Schema** (`docs/yaml_schemas/cluster_descriptor.yaml`): add optional top-level

```yaml
hostname:
  type: string
  minLength: 1
  maxLength: 63
  description: >
    OS / FSD hostname of the machine this descriptor describes.
    One descriptor is one host. Not a filename and not a unique chip id.
  pattern: "^[A-Za-z0-9]([A-Za-z0-9.-]*[A-Za-z0-9])?$"
```

`additionalProperties` stays `false` — the key must be listed. Not in `required`.

**Parse** (`create_from_yaml_content`): if `yaml["hostname"]` is defined, `set_hostname(as<string>)`. Missing → leave `nullopt`. Present but invalid → throw (same as a bad `arch` key).

**Serialize** (`serialize()`): if `hostname_` is set, emit it **first** in the map (readable, stable). Omit the key when unset so old golden YAMLs do not grow a dummy field.

**Round-trip:** `create_from_yaml` → `serialize` → `create_from_yaml_content` preserves the name.

---

## 4. Copy / constrain / discover

| Site | Behavior |
|------|----------|
| `create_from_yaml` / `_content` | load optional key |
| `serialize` / `serialize_to_file` | write if set |
| `create_constrained_cluster_descriptor` | copy `hostname_` (same host, fewer chips) |
| `apply_chip_id_remapping` | copy `hostname_` onto the remapped descriptor |
| `TopologyDiscovery::fill_cluster_descriptor_info` | `set_hostname(os_hostname())` after the chip tables are filled. Use POSIX `gethostname`. Store the raw result (no FQDN strip in UMD). Metal `canonical_host_for_node_id` still does case / first-label / `_rank`. |
| `create_mock_cluster` | leave unset unless an optional hostname argument is added |

`os_hostname()` in UMD: small helper next to the descriptor (or reuse whatever UMD already uses for logs). Do not pull metal `get_host_name()`.

---

## 5. Validation (UMD)

If the key is present or `set_hostname` is called:

- non-empty after trim
- no space / tab / `/` / `\`
- `size() < 64` (fits `PhysicalNodeId::host` including NUL)
- no leading / trailing `.`

Do **not** require it to match the YAML filename. The filename stays an asset-repo convention; the field is the name.

Do **not** try to be unique across ranks. One descriptor = one host. Metal's `resolve_hostname_uniqueness()` still suffixes `_<rank>` when two ranks report the same string (legacy 16-file superpod reuse). That suffix is a PSD merge key, not something UMD writes back into the YAML.

---

## 6. Metal consumption (after the UMD bump)

`run_local_discovery` already has `ClusterDescriptor& cluster_desc`. Change `get_local_discovery_hostname()` to take it:

```
get_local_discovery_hostname(cluster_desc):
    if cluster_desc.get_hostname() has a value:
        return *that
    if TT_METAL_MOCK_CLUSTER_DESC_PATH is set:
        return path.filename()          // legacy ClosetBox / unfilled YAMLs
    return get_host_name()              // silicon, descriptor not yet stamped (should not happen after UMD)
```

`PhysicalNodeId` packs `canonical_host_for_node_id` of that string. Mock + FSD then use `bh-glx-110-c01u02` on both sides with **no** aisle-token alias.

Keep the filename fallback for any YAML that still lacks the field (the §7 TODO). ClosetBox and every existing mock test stay on the basename until that file is filled. Do not change those tests in the UMD PR.

Metal does **not** open the YAML itself to peek at `hostname:`. Always go through `ClusterDescriptor::get_hostname()`.

---

## 7. TODO: add `hostname:` to all tt-cluster-descriptors YAMLs

**TODO:** Write `hostname:` on **every** cluster descriptor in [tt-cluster-descriptors](https://github.com/tenstorrent/tt-cluster-descriptors) (`superclusters/`, `wormhole/`, `blackhole/`, ClosetBox, virtu, T3K, dual-host, … — all ~230 files). Not only FSD-paired BH. Separate repo PR. **Not** required for the UMD PR to merge. Because the field is optional (§1.1), partial fill is fine: filled files use the real name, unfilled files keep the basename fallback.

Do not rename files. Do not edit FSD textprotos.

**Value to write** — exact OS / FSD hostname of that machine, hall included:

| Descriptor family | How to pick the name |
|-------------------|----------------------|
| BH supercluster next to an FSD | FSD `hosts[].hostname` via aisle/rack/u (below) |
| QuietBox | `sjc1-tt-qb-01` etc. — already on the PSD |
| ClosetBox | the real host in the filename token (`metal-wh-09`), not the whole basename |
| Virtu / `bg-ale22` | the real host series name for that YAML |
| Wormhole / T3K / dual-host / others | OS hostname of the machine the capture came from; if unknown, leave unset until someone recaptures (`serialize_to_file` will stamp it on silicon) |

**FSD-paired fill script** (asset one-shot, not runtime):

1. From the YAML filename, take the last `[a-z][0-9]{2}u[0-9]{2}` before `_rank_` / `.yaml` (`c01u02`, `d10u20`).
2. In the sibling `*_factory_system_descriptor.textproto`, find the unique `Host` with `aisle` / `rack` / `shelf_u` equal to that token.
3. Write that host's `hostname` field into the YAML.

That join is how we **populate** the field. After the field exists, metal never runs the join. Filename hall mismatches (SC36 file `bh-glx-120-d10u20` vs FSD `bh-glx-110-d10u20`) do not matter: the YAML field is the FSD string.

After a file has the field, metal never joins on the filename. The aisle-token alias is only for YAMLs that still lack the field.

---

## 8. Tests (UMD, offline)

File: `tests/baremetal/test_cluster_descriptor_hostname.cpp` (or cases on `test_cluster_descriptor_offline.cpp`).

- YAML **with** `hostname: bh-glx-110-c01u02` → `get_hostname() == that string`.
- YAML **without** the key → `get_hostname() == nullopt`. **Back-compat:** every existing offline descriptor in the UMD tree still parses (no `hostname` required).
- `serialize()` omits the key when unset; includes it when set; parse(serialize(d)) == d.
- `set_hostname("")` / whitespace / 64+ chars / `foo/bar` throw.
- `create_constrained_cluster_descriptor` copies the hostname.
- Nanobind: `ClusterDescriptor.create_from_yaml_content(...).get_hostname()`.
- Hardware (optional, `TopologyDiscovery`): after `discover`, `get_hostname()` equals `gethostname()` on that machine.

No metal test in the UMD PR.

---

## 9. File list (UMD)

- `device/api/umd/device/cluster_descriptor.hpp` — getter, setter, `hostname_`
- `device/cluster_descriptor.cpp` — parse, serialize, constrain, remap, `create_mock_cluster` unchanged
- `device/topology/topology_discovery.cpp` — stamp in `fill_cluster_descriptor_info`
- `docs/yaml_schemas/cluster_descriptor.yaml` — optional `hostname`
- `nanobind/py_api_topology_discovery.cpp` — bind get/set
- `tests/baremetal/test_cluster_descriptor_hostname.cpp`

No change to chip unique ids, ETH tables, or mock filename conventions.

---

## 10. Sequence

1. **UMD PR** — optional field + query + live stamp + tests. Old YAMLs still load (back-compat).
2. Bump `tt_metal/third_party/umd` in tt-metal.
3. **tt-metal** — `get_local_discovery_hostname(cluster_desc)` prefers `get_hostname()`, basename fallback if absent. Existing tests keep passing.
4. **TODO / tt-cluster-descriptors** — write `hostname:` on **all** cluster descriptor YAMLs (§7). Can land incrementally; pin when a batch is ready.
5. Drop the aisle-token fallback once FSD-paired BH files are filled.

Slice 1 is reviewable with no metal or asset changes. Slice 4 does not gate slice 1 or 3.

---

## 11. Non-goals

- Per-chip hostname
- Hostname on `ethernet_connections_to_remote_devices`
- Renaming cluster-desc files to match FSD
- FQDN canonicalization inside UMD
- Requiring the field on every existing YAML in the UMD PR (the §7 TODO is a follow-up)
- Removing the metal basename fallback before the §7 TODO is done
- Breaking old UMD or old metal by adding the key to assets
