# Plan: `host_id` on the UMD cluster descriptor

**Status:** Implementation plan — **UMD change**, consumed by PhysicalNodeId / FSD mock
**Repos:** [tt-umd](https://github.com/tenstorrent/tt-umd) (API + YAML), then [tt-cluster-descriptors](https://github.com/tenstorrent/tt-cluster-descriptors) (fill the field), then tt-metal (read it). Land UMD first.
**Consumers:** [`PLAN_physical_node_id.md`](PLAN_physical_node_id.md) §8 (problem / solution / checklist), [`PLAN_downed_links_testing.md`](PLAN_downed_links_testing.md) §6.3.
**Note on the filename:** this file keeps its `_hostname` name so the links from the other plans do not rot. The field is `host_id`, and the consumer plans above reference it by that name.
**Name collision to watch:** the FSD builder already has a `host_id` meaning *host index 0..N-1 in file order* ([`PLAN_physical_node_id.md`](PLAN_physical_node_id.md) §2). That is a different thing. Where both are in scope, write "FSD host index" and "UMD `host_id`".

---

## 0. Problem, then solution

### Problem

A cluster descriptor is one host's chips. It has **no field naming that host**. Metal mock discovery therefore keys on the YAML **filename** (`get_local_discovery_hostname` = basename of `TT_METAL_MOCK_CLUSTER_DESC_PATH`). That string is not the FSD / OS hostname.

| Side | String | Example |
|------|--------|---------|
| FSD / silicon (what we must pack) | real hostname | `bh-glx-110-c01u02` |
| Mock YAML filename | asset name, not a hostname | `SC20_…_bh-glx-c01u02_rank_0.yaml` (hall dropped) |
| SC36 filename token | hall **wrong** | file `bh-glx-120-d10u20` vs FSD `bh-glx-110-d10u20` |
| Container / VM `gethostname()` | runtime-generated, unrelated to the machine | `7f3a91c2b4de`, `pod-fabric-worker-3` |

If mock and FSD pack different `host_id[]` bytes, `PhysicalNodeId` graphs disagree and the solver sees two topologies. Parsing a name out of the filename cannot work (SC20 omits hall, SC36 disagrees). Do **not** parse `bh-glx-c01u02` at runtime.

The container / VM row is the second half of the problem. Even on live silicon, `gethostname()` is only the right answer when the process runs on bare metal. In a container it returns the container id; in a VM it returns whatever the guest was named. Both are stable strings that are *not* the identity of the accelerator group, so a hostname-only design breaks exactly where we are heading.

### Solution

Add an optional top-level **`host_id:`** to the UMD cluster descriptor and query it.

**Semantics:** `host_id` is *a unique string identifying a group of TT accelerators connected to a common host / controller / root complex.* It is an **identifier**, not a name of a machine. For the time being its value **is** the bare-metal hostname, because that is what the FSD and every current consumer key on — but the meaning of the field is the accelerator group, and the value scheme is free to change later (§11).

UMD fills it, in order:

1. `TT_HOST_ID` environment variable, if set and non-empty.
2. otherwise POSIX `gethostname()`.

The env var is what makes the field usable in containers and VMs: the operator (or the launcher) sets `TT_HOST_ID` to the id of the physical accelerator group, and UMD stamps that instead of the meaningless in-guest hostname.

Metal prefers `desc.get_host_id()`, falls back to today's basename if absent. Field is optional forever — old YAMLs still load; old UMD ignores the unknown key; existing mock tests stay green.

```yaml
host_id: bh-glx-110-c01u02
arch:
  ...
```

**TODO:** write that field on **every** YAML in tt-cluster-descriptors (~230 files). Does not block this UMD PR. Full checklist: [`PLAN_physical_node_id.md`](PLAN_physical_node_id.md) §8.3.

---

## 1. Model

One `ClusterDescriptor` YAML is **one host** — one accelerator group. Chips, trays, and `asic_locations` are on that host. Cross-host ETH is `ethernet_connections_to_remote_devices` keyed by the peer's **unique chip id**, not by host.

So `host_id` is a **descriptor-level** field, not per-chip:

```yaml
host_id: bh-glx-110-c01u02
arch:
  0: blackhole
  ...
```

Today that value is the same string as FSD `hosts[].hostname` and as bare-metal `gethostname()` (first label / short name as the machine is configured). Hall stays (`110`). We do not invent a shorter token.

| Path | Who writes `host_id` | Who reads it |
|------|------------------------|--------------|
| Live silicon | `TopologyDiscovery::fill_cluster_descriptor_info` stamps `TT_HOST_ID` else `gethostname()` | metal discovery, `serialize()` |
| Mock YAML | author / asset fill (exact FSD name) | `create_from_yaml` → `get_host_id()` |
| Recapture | `serialize_to_file` writes whatever is on the object | next mock YAML gets the live id for free |
| Container / VM | operator sets `TT_HOST_ID`; UMD stamps it verbatim | same as live silicon |

Absent field = old descriptor. That is the compatibility path, not a temporary hack.

### 1.1 Why a new name and not `hostname`

Calling the field `hostname` would freeze the wrong contract into the YAML, the schema, the Python bindings, and ~230 asset files, and every one of those is expensive to rename later. The field's job is *identify the accelerator group*; "it currently happens to be the hostname" is the fill rule (§4), not the definition. Naming it `host_id` lets the value scheme change without another schema migration.

Do **not** add both `hostname` and `host_id`. Do **not** alias one to the other. There is one field.

### 1.2 Backward compatibility (locked)

| Combination | What happens |
|-------------|--------------|
| New UMD + old YAML (no `host_id:`) | `get_host_id() == nullopt`. Parse succeeds. |
| Old UMD + new YAML (`host_id:` present) | Parser only reads known keys; the new field is ignored. File still loads. |
| New metal + old YAML | `get_local_discovery_hostname` uses filename basename / `gethostname()` — **today's behavior**. |
| New metal + new YAML | uses `get_host_id()`. |
| Old metal + new YAML | still keys on the filename. ClosetBox / existing mapper tests unchanged. |
| `TT_HOST_ID` unset | `gethostname()`, i.e. exactly today's effective value on bare metal. |

Do **not** put `host_id` in the schema `required` list. Do **not** fatal in UMD or metal because the field is missing. Do **not** rewrite filenames. `serialize()` omits the key when unset so existing golden dumps do not change unless discovery stamped an id.

Adding the field to assets is therefore safe to land in any order relative to metal.

---

## 2. UMD API

```cpp
// umd/device/cluster_descriptor.hpp
class ClusterDescriptor {
public:
    // Unique id of the group of TT accelerators attached to a common
    // host / controller / root complex. Currently the bare-metal hostname
    // (or $TT_HOST_ID); semantically the accelerator group, not a machine name.
    // Empty when the YAML omitted the key and discovery did not stamp one.
    const std::optional<std::string>& get_host_id() const;

    // Reject empty, whitespace, path separators, length >= 64.
    void set_host_id(std::string host_id);

    // ...
private:
    std::optional<std::string> host_id_;
};
```

**Query from metal** (already has the descriptor):

```cpp
auto* desc = cluster.get_cluster_desc();            // llrt Cluster
// or the ClusterDescriptor& already passed into run_local_discovery
if (const auto& id = desc->get_host_id(); id.has_value()) {
    // this is the accelerator-group id
}
```

Do **not** add a second metal wrapper type. `tt::umd::Cluster::get_cluster_description()->get_host_id()` is the query. Nanobind: bind `get_host_id` / `set_host_id` next to the other `ClusterDescriptor` getters (`py_api_topology_discovery.cpp`).

`create_mock_cluster(...)` stays id-less (`nullopt`) unless a new optional argument is passed. Simulation / unit mocks do not need a galaxy id, and they must **not** pick one up from `TT_HOST_ID` (§4.1).

---

## 3. YAML

**Schema** (`docs/yaml_schemas/cluster_descriptor.yaml`): add optional top-level

```yaml
host_id:
  type: string
  minLength: 1
  maxLength: 63
  description: >
    Unique id of the group of TT accelerators attached to a common
    host / controller / root complex. One descriptor is one such group.
    Currently the bare-metal hostname of that machine (this is what the FSD
    and the fabric topology solver key on), or the value of $TT_HOST_ID when
    the process runs in a container or VM. Not a filename and not a unique
    chip id.
  pattern: "^[A-Za-z0-9]([A-Za-z0-9._-]*[A-Za-z0-9])?$"
```

`additionalProperties` stays `false` — the key must be listed. Not in `required`.

The pattern is deliberately a hostname-shaped charset even though the field is not defined as a hostname: today's values *are* hostnames and must join against FSD `hosts[].hostname`. Widening the charset (`:`, `/`, structured ids) is a later, deliberate change that goes together with the FSD gaining its own `host_id` (§11) — not something to pre-authorize here.

**Parse** (`create_from_yaml_content`): if `yaml["host_id"]` is defined, `set_host_id(as<string>)`. Missing → leave `nullopt`. Present but invalid → throw (same as a bad `arch` key).

**Serialize** (`serialize()`): if `host_id_` is set, emit it **first** in the map (readable, stable). Omit the key when unset so old golden YAMLs do not grow a dummy field.

**Round-trip:** `create_from_yaml` → `serialize` → `create_from_yaml_content` preserves the id.

---

## 4. Copy / constrain / discover

| Site | Behavior |
|------|----------|
| `create_from_yaml` / `_content` | load optional key |
| `serialize` / `serialize_to_file` | write if set |
| `create_constrained_cluster_descriptor` | copy `host_id_` (same host, fewer chips) |
| `apply_chip_id_remapping` | copy `host_id_` onto the remapped descriptor |
| `TopologyDiscovery::fill_cluster_descriptor_info` | `set_host_id(local_host_id())` after the chip tables are filled |
| `create_mock_cluster` | leave unset unless an optional `host_id` argument is added |

### 4.1 `local_host_id()` — the fill rule

Small helper next to the descriptor in UMD:

```
local_host_id():
    if getenv("TT_HOST_ID") is set and non-empty after trim:
        validate it (§5); fatal on invalid
        return it
    return gethostname()          // POSIX, raw result, no FQDN strip in UMD
```

- **Empty / whitespace-only `TT_HOST_ID` counts as unset** — an exported-but-empty var is a launcher accident, not a request for an empty id.
- **Invalid `TT_HOST_ID` is fatal, not a fallback to `gethostname()`.** Someone set it on purpose; silently substituting a container hostname yields a wrong-but-plausible topology, which is the exact failure this field exists to prevent. Fail loudly.
- Store the raw `gethostname()` result. Metal `canonical_host_for_node_id` still does case / first-label / `_rank`.
- Do not pull metal `get_host_name()` into UMD.

This is the **only** place the env var is read. In particular:

- `create_from_yaml` does **not** apply `TT_HOST_ID` when the YAML lacks the key. A mock YAML describes *some other* machine; the env var describes *this* one. Substituting would relabel every mock descriptor as the local host and collapse a multi-host mock into one node.
- `create_mock_cluster` does **not** read it either.

Log the resolved id and its source (`env` / `gethostname`) once at discovery. That one line is what an operator will ask for first when a containerized run maps to the wrong FSD host.

---

## 5. Validation (UMD)

If the key is present, `set_host_id` is called, or `TT_HOST_ID` is set:

- non-empty after trim
- no space / tab / `/` / `\`
- `size() < 64` (fits `PhysicalNodeId::host_id` including NUL)
- no leading / trailing `.`
- matches the §3 pattern

Do **not** require it to match the YAML filename. The filename stays an asset-repo convention; the field is the id.

Do **not** try to enforce uniqueness across ranks from inside UMD. One descriptor = one accelerator group. Metal's `resolve_hostname_uniqueness()` still suffixes `_<rank>` when two ranks report the same string (legacy 16-file superpod reuse). That suffix is a PSD merge key, not something UMD writes back into the YAML. (When ranks share a `host_id` because a container fleet forgot to set `TT_HOST_ID` distinctly, that suffix is also the symptom to look for.)

---

## 6. Metal consumption (after the UMD bump)

`run_local_discovery` already has `ClusterDescriptor& cluster_desc`. Change `get_local_discovery_hostname()` to take it:

```
get_local_discovery_hostname(cluster_desc):
    if cluster_desc.get_host_id() has a value:
        return *that
    if TT_METAL_MOCK_CLUSTER_DESC_PATH is set:
        return path.filename()          // legacy ClosetBox / unfilled YAMLs
    return get_host_name()              // silicon, descriptor not yet stamped (should not happen after UMD)
```

`PhysicalNodeId` packs `canonical_host_for_node_id` of that string. Mock + FSD then use `bh-glx-110-c01u02` on both sides with **no** aisle-token alias.

Metal may rename that function to `get_local_host_id()` in the metal PR — the string it returns is now an accelerator-group id, and every metal-side `hostname` identifier is on the §11 list. That rename is cosmetic and does not gate anything.

Keep the filename fallback for any YAML that still lacks the field (the §7 TODO). ClosetBox and every existing mock test stay on the basename until that file is filled. Do not change those tests in the UMD PR.

Metal does **not** open the YAML itself to peek at `host_id:`, and metal does **not** read `TT_HOST_ID`. Always go through `ClusterDescriptor::get_host_id()` — one reader of the env var, in UMD (§4.1).

---

## 7. TODO: add `host_id:` to all tt-cluster-descriptors YAMLs

**TODO:** Write `host_id:` on **every** cluster descriptor in [tt-cluster-descriptors](https://github.com/tenstorrent/tt-cluster-descriptors) (`superclusters/`, `wormhole/`, `blackhole/`, ClosetBox, virtu, T3K, dual-host, … — all ~230 files). Not only FSD-paired BH. Separate repo PR. **Not** required for the UMD PR to merge. Because the field is optional (§1.2), partial fill is fine: filled files use the real id, unfilled files keep the basename fallback.

Do not rename files. Do not edit FSD textprotos.

**Value to write** — exact OS / FSD hostname of that machine, hall included (the current value scheme, §0):

| Descriptor family | How to pick the value |
|-------------------|----------------------|
| BH supercluster next to an FSD | FSD `hosts[].hostname` via aisle/rack/u (below) |
| QuietBox | `sjc1-tt-qb-01` etc. — already on the PSD |
| ClosetBox | the real host in the filename token (`metal-wh-09`), not the whole basename |
| Virtu / `bg-ale22` | the real host series name for that YAML |
| Wormhole / T3K / dual-host / others | OS hostname of the machine the capture came from; if unknown, leave unset until someone recaptures (`serialize_to_file` will stamp it on silicon) |

**FSD-paired fill script** (asset one-shot, not runtime):

1. From the YAML filename, take the last `[a-z][0-9]{2}u[0-9]{2}` before `_rank_` / `.yaml` (`c01u02`, `d10u20`).
2. In the sibling `*_factory_system_descriptor.textproto`, find the unique `Host` with `aisle` / `rack` / `shelf_u` equal to that token.
3. Write that host's `hostname` field into the YAML as `host_id`.

That join is how we **populate** the field. After the field exists, metal never runs the join. Filename hall mismatches (SC36 file `bh-glx-120-d10u20` vs FSD `bh-glx-110-d10u20`) do not matter: the YAML field is the FSD string.

After a file has the field, metal never joins on the filename. The aisle-token alias is only for YAMLs that still lack the field.

---

## 8. Tests (UMD, offline)

File: `tests/baremetal/test_cluster_descriptor_host_id.cpp` (or cases on `test_cluster_descriptor_offline.cpp`).

- YAML **with** `host_id: bh-glx-110-c01u02` → `get_host_id() == that string`.
- YAML **without** the key → `get_host_id() == nullopt`. **Back-compat:** every existing offline descriptor in the UMD tree still parses (no `host_id` required).
- `serialize()` omits the key when unset; includes it when set; parse(serialize(d)) == d.
- `set_host_id("")` / whitespace / 64+ chars / `foo/bar` throw.
- `create_constrained_cluster_descriptor` and `apply_chip_id_remapping` copy the id.
- `local_host_id()` with `TT_HOST_ID=tt-vm-host-7` → returns it; unset → returns `gethostname()`; `TT_HOST_ID=""` / `"   "` → treated as unset; `TT_HOST_ID="a/b"` or 64+ chars → fatal (**not** a silent fallback).
- **Env var does not leak into parsing:** with `TT_HOST_ID` set, a YAML *without* the key still yields `nullopt`, and a YAML *with* `host_id: other-host` still yields `other-host`.
- Nanobind: `ClusterDescriptor.create_from_yaml_content(...).get_host_id()`.
- Hardware (optional, `TopologyDiscovery`): after `discover`, `get_host_id()` equals `gethostname()` with no env var, and equals `TT_HOST_ID` when it is set.

No metal test in the UMD PR.

---

## 9. File list (UMD)

- `device/api/umd/device/cluster_descriptor.hpp` — getter, setter, `host_id_`
- `device/cluster_descriptor.cpp` — parse, serialize, constrain, remap, `create_mock_cluster` unchanged
- `device/topology/topology_discovery.cpp` — stamp in `fill_cluster_descriptor_info`
- `device/…/local_host_id.{hpp,cpp}` (or the existing small-utils home) — `TT_HOST_ID` → `gethostname()` fallback + validation + the one log line
- `docs/yaml_schemas/cluster_descriptor.yaml` — optional `host_id`
- `nanobind/py_api_topology_discovery.cpp` — bind get/set
- `tests/baremetal/test_cluster_descriptor_host_id.cpp`
- UMD docs / README env-var table — document `TT_HOST_ID` (containers and VMs need it; bare metal does not)

No change to chip unique ids, ETH tables, or mock filename conventions.

---

## 10. Sequence

1. **UMD PR** — optional `host_id` field + query + `TT_HOST_ID`/`gethostname()` stamp + tests. Old YAMLs still load (back-compat).
2. Bump `tt_metal/third_party/umd` in tt-metal.
3. **tt-metal** — `get_local_discovery_hostname(cluster_desc)` prefers `get_host_id()`, basename fallback if absent. Existing tests keep passing.
4. **TODO / tt-cluster-descriptors** — write `host_id:` on **all** cluster descriptor YAMLs (§7). Can land incrementally; pin when a batch is ready.
5. Drop the aisle-token fallback once FSD-paired BH files are filled.
6. Container / VM enablement: set `TT_HOST_ID` in the launchers and images that run fabric workloads (no code change; §11 tracks the rest).

Slice 1 is reviewable with no metal or asset changes. Slice 4 does not gate slice 1 or 3.

---

## 11. Direction of travel: off hostnames for ASIC addressing

`host_id` is the first step, not the whole move. The end state is that **no** component derives accelerator identity from a machine name; they all carry an opaque `host_id` that some higher layer assigns. Getting there is out of scope for this plan, and each item below needs its own change:

| Component | Current hostname assumption | What it needs |
|-----------|------------------------------|---------------|
| FSD (`tt-cluster-descriptors`) | `hosts[].hostname` is the join key | its own `host_id` field, populated the same way, hostname kept for humans |
| `PhysicalSystemDescriptor` / `ASICDescriptor` | `host_name` field, `HostName` type, `get_all_hostnames()`, `my_host_name()` | rename to host id; the value already flows from `get_host_id()` after slice 3. `TopologyMapper` deliberately does **not** do this rename ([`PLAN_physical_node_id.md`](PLAN_physical_node_id.md) §6.1) |
| `PhysicalNodeId` | `host_id[64]` NUL-padded, `canonical_host_for_node_id` | member already renamed; keep the buffer, retire the DNS-label canonicalization once ids are not hostnames |
| Metal discovery | `get_host_name()`, filename basename, `resolve_hostname_uniqueness()` `_<rank>` suffix | id from UMD; uniqueness enforced by whoever assigns ids |
| Rank bindings / launchers | map MPI ranks by hostname; `TopologyMappingConfig::hostname_to_asics` | map by `host_id` |
| Distributed bring-up / logs / dashboards | print hostnames | print both while both exist |

Until those land, `host_id` **must** stay hostname-valued (§3 pattern, §7 fill rule) — the FSD join and the mock/FSD `PhysicalNodeId` agreement both depend on it. Changing the value scheme before the FSD carries `host_id` would re-break the exact thing this plan fixes.

---

## 12. Non-goals

- Per-chip `host_id`
- `host_id` on `ethernet_connections_to_remote_devices`
- Renaming cluster-desc files to match FSD
- FQDN canonicalization inside UMD
- Requiring the field on every existing YAML in the UMD PR (the §7 TODO is a follow-up)
- Removing the metal basename fallback before the §7 TODO is done
- Breaking old UMD or old metal by adding the key to assets
- A second `hostname:` field, or an alias between `hostname` and `host_id`
- Reading `TT_HOST_ID` anywhere except `local_host_id()` (not in metal, not in YAML parsing, not in `create_mock_cluster`)
- Changing the value scheme away from hostnames in this plan (§11 is the follow-up, and it starts in the FSD)
- Renaming metal's hostname-typed APIs / PSD fields as part of the UMD PR
