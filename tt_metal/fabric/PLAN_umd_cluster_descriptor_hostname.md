# Plan: `cluster_id` on the UMD cluster descriptor

**Status:** Implementation plan — **UMD change**, consumed by PhysicalNodeId / FSD mock
**Repos:** [tt-umd](https://github.com/tenstorrent/tt-umd) (API + YAML), then [tt-cluster-descriptors](https://github.com/tenstorrent/tt-cluster-descriptors) (fill the field), then tt-metal (read it). Land UMD first.
**Consumers:** [`PLAN_physical_node_id.md`](PLAN_physical_node_id.md) §8 (problem / solution / checklist), [`PLAN_downed_links_testing.md`](PLAN_downed_links_testing.md) §6.3.
**Note on the filename:** this file keeps its `_hostname` name so the links from the other plans do not rot. The field is `cluster_id`, and the consumer plans above reference it by that name.
**Nearby name to keep apart:** the FSD builder has a `host_id` meaning *host index 0..N-1 in file order* ([`PLAN_physical_node_id.md`](PLAN_physical_node_id.md) §2). Naming this field `cluster_id` is what keeps the two from sharing a name, but the FSD index is still spelled `host_id`, so write "FSD host index" where both are in scope.

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

If mock and FSD pack different `cluster_id[]` bytes, `PhysicalNodeId` graphs disagree and the solver sees two topologies. Parsing a name out of the filename cannot work (SC20 omits hall, SC36 disagrees). Do **not** parse `bh-glx-c01u02` at runtime. Reading the id off the descriptor is what replaces that parsing; whether a given asset's id also matches its FSD is a question about that asset, not about the mechanism (§7).

The container / VM row is the second half of the problem. Even on live silicon, `gethostname()` is only the right answer when the process runs on bare metal. In a container it returns the container id; in a VM it returns whatever the guest was named. Both are stable strings that are *not* the identity of the accelerator group, so a hostname-only design breaks exactly where we are heading.

### Solution

Add an optional top-level **`cluster_id:`** to the UMD cluster descriptor and query it.

**Semantics:** `cluster_id` is *a unique string identifying a group of TT accelerators connected to a common host / controller / root complex.* It is an **identifier**, not a name of a machine. For the time being its value **is** the bare-metal hostname, because that is what the FSD and every current consumer key on — but the meaning of the field is the accelerator group, and the value scheme is free to change later (§11).

UMD fills it, in order:

1. `TopologyDiscoveryOptions::cluster_id`, when the caller supplied one.
2. otherwise POSIX `gethostname()`.

The discovery option is what makes the field usable in containers and VMs: the caller passes the id of the physical accelerator group, and UMD stamps that instead of the meaningless in-guest hostname. UMD deliberately reads no environment variable for this — container and VM plumbing belongs to whoever launches the process, and one string on the options struct is enough to carry it down. Metal is that caller, so metal is where the env var lives (§6.1).

Metal prefers `desc.get_cluster_id()`, falls back to today's basename if absent. Field is optional forever — old YAMLs still load; old UMD ignores the unknown key; existing mock tests stay green.

```yaml
cluster_id: bh-glx-110-c01u02
arch:
  ...
```

**TODO:** write that field on **every** YAML in tt-cluster-descriptors (~230 files). Does not block this UMD PR. Full checklist: [`PLAN_physical_node_id.md`](PLAN_physical_node_id.md) §8.3.

---

## 1. Model

One `ClusterDescriptor` YAML is **one host** — one accelerator group. Chips, trays, and `asic_locations` are on that host. Cross-host ETH is `ethernet_connections_to_remote_devices` keyed by the peer's **unique chip id**, not by host.

So `cluster_id` is a **descriptor-level** field, not per-chip:

```yaml
cluster_id: bh-glx-110-c01u02
arch:
  0: blackhole
  ...
```

Today that value is the same string as FSD `hosts[].hostname` and as bare-metal `gethostname()` (first label / short name as the machine is configured). Hall stays (`110`). We do not invent a shorter token.

| Path | Who writes `cluster_id` | Who reads it |
|------|------------------------|--------------|
| Live silicon | `TopologyDiscovery::fill_cluster_descriptor_info` stamps the supplied id else `gethostname()` | metal discovery, `serialize()` |
| Mock YAML | author / asset fill (exact FSD name) | `create_from_yaml` → `get_cluster_id()` |
| Recapture | `serialize_to_file` writes whatever is on the object | next mock YAML gets the live id for free |
| Container / VM | caller passes `TopologyDiscoveryOptions::cluster_id`; UMD stamps it verbatim | same as live silicon |

Absent field = old descriptor. That is the compatibility path, not a temporary hack.

### 1.1 Why a new name and not `hostname`

Calling the field `hostname` would freeze the wrong contract into the YAML, the schema, the Python bindings, and ~230 asset files, and every one of those is expensive to rename later. The field's job is *identify the accelerator group*; "it currently happens to be the hostname" is the fill rule (§4), not the definition. Naming it `cluster_id` lets the value scheme change without another schema migration.

Do **not** add both `hostname` and `cluster_id`. Do **not** alias one to the other. There is one field.

### 1.2 Backward compatibility (locked)

| Combination | What happens |
|-------------|--------------|
| New UMD + old YAML (no `cluster_id:`) | `get_cluster_id() == nullopt`. Parse succeeds. |
| Old UMD + new YAML (`cluster_id:` present) | Parser only reads known keys; the new field is ignored. File still loads. |
| New metal + old YAML | `get_local_discovery_hostname` uses filename basename / `gethostname()` — **today's behavior**. |
| New metal + new YAML | uses `get_cluster_id()`. |
| Old metal + new YAML | still keys on the filename. ClosetBox / existing mapper tests unchanged. |
| No id supplied to discovery | `gethostname()`, i.e. exactly today's effective value on bare metal. |

Do **not** put `cluster_id` in the schema `required` list. Do **not** fatal in UMD or metal because the field is missing. Do **not** rewrite filenames. `serialize()` omits the key when unset so existing golden dumps do not change unless discovery stamped an id.

Adding the field to assets is therefore safe to land in any order relative to metal.

---

## 2. UMD API

```cpp
// umd/device/cluster_descriptor.hpp
class ClusterDescriptor {
public:
    // Unique id of the group of TT accelerators attached to a common
    // host / controller / root complex. Currently the bare-metal hostname;
    // semantically the accelerator group, not a machine name.
    // Empty when the YAML omitted the key and discovery did not stamp one.
    const std::optional<std::string>& get_cluster_id() const;

    // ...
private:
    std::optional<std::string> cluster_id_;
};
```

There is **no** setter. A cluster id is fixed when the descriptor is built — parsed from the YAML or stamped by discovery — and never changes over the descriptor's lifetime. Both entry points validate (§5), so every id on a descriptor is a legal one, and there is no third way to put a value there.

**Supply the id to discovery** through the options struct, which `ClusterOptions` already carries for the caller:

```cpp
TopologyDiscoveryOptions options;
options.cluster_id = "bh-glx-110-c01u02";   // omit on bare metal; UMD uses gethostname()
auto cluster_desc = TopologyDiscovery::discover(options).first;
```

**Query from metal** (already has the descriptor):

```cpp
auto* desc = cluster.get_cluster_desc();            // llrt Cluster
// or the ClusterDescriptor& already passed into run_local_discovery
if (const auto& id = desc->get_cluster_id(); id.has_value()) {
    // this is the accelerator-group id
}
```

Do **not** add a second metal wrapper type. `tt::umd::Cluster::get_cluster_description()->get_cluster_id()` is the query. Nanobind: bind `get_cluster_id` next to the other `ClusterDescriptor` getters, and `cluster_id` on `TopologyDiscoveryOptions` (`py_api_topology_discovery.cpp`).

`create_mock_cluster(...)` stays id-less (`nullopt`) unless a new optional argument is passed. Simulation / unit mocks do not need a galaxy id, and nothing ambient can give them one (§4.1).

---

## 3. YAML

**Schema** (`docs/yaml_schemas/cluster_descriptor.yaml`): add optional top-level

```yaml
cluster_id:
  type: string
  minLength: 1
  maxLength: 128
  description: >
    Unique id of the group of TT accelerators attached to a common
    host / controller / root complex. One descriptor is one such group.
    Currently the bare-metal hostname of that machine (this is what the FSD
    and the fabric topology solver key on). Not a filename and not a unique
    chip id.
  pattern: "^[A-Za-z0-9._-]+$"
```

`additionalProperties` stays `false` — the key must be listed. Not in `required`.

The pattern is deliberately a hostname-shaped charset even though the field is not defined as a hostname: today's values *are* hostnames and must join against FSD `hosts[].hostname`. Widening the charset (`:`, `/`, structured ids) is a later, deliberate change that goes together with the FSD gaining its own `cluster_id` (§11) — not something to pre-authorize here.

**Parse** (`create_from_yaml_content`): if `yaml["cluster_id"]` is defined, validate `as<string>` (§5) and store it. Missing → leave `nullopt`. Present but invalid → throw (same as a bad `arch` key).

**Serialize** (`serialize()`): if `cluster_id_` is set, emit it **first** in the map (readable, stable). Omit the key when unset so old golden YAMLs do not grow a dummy field.

**Round-trip:** `create_from_yaml` → `serialize` → `create_from_yaml_content` preserves the id.

---

## 4. Copy / constrain / discover

| Site | Behavior |
|------|----------|
| `create_from_yaml` / `_content` | load optional key |
| `serialize` / `serialize_to_file` | write if set |
| `create_constrained_cluster_descriptor` | copy `cluster_id_` (same host, fewer chips) |
| `apply_chip_id_remapping` | copy `cluster_id_` onto the remapped descriptor |
| `TopologyDiscovery::fill_cluster_descriptor_info` | stamp `resolve_cluster_id(options.cluster_id)` after the chip tables are filled |
| `create_mock_cluster` | leave unset unless an optional `cluster_id` argument is added |

### 4.1 `resolve_cluster_id()` — the fill rule

Small helper next to the descriptor in UMD, taking the caller's option:

```
resolve_cluster_id(supplied):
    if supplied has a value:
        validate it (§5); fatal on invalid
        return it
    return gethostname()          // POSIX, raw result, no FQDN strip in UMD
```

- **An illegal supplied id is fatal, not a fallback to `gethostname()`.** The caller passed it on purpose; silently substituting a container hostname yields a wrong-but-plausible topology, which is the exact failure this field exists to prevent. Fail loudly.
- **An unusable OS hostname only warns** and leaves the field unset (§1.2 back-compat), because nobody asked for it. Throwing there would break discovery on hosts that work today.
- Store the raw `gethostname()` result. Metal `canonical_cluster_id_for_node_id` still does case / first-label / `_rank`.
- Do not pull metal `get_host_name()` into UMD.

The id reaches UMD only through that one option. In particular:

- `create_from_yaml` does **not** apply the option when the YAML lacks the key. A mock YAML describes *some other* machine; the option describes *this* one. Substituting would relabel every mock descriptor as the local host and collapse a multi-host mock into one node.
- `create_mock_cluster` does **not** see it either.
- **UMD reads no environment variable.** There is no `TT_CLUSTER_ID`. Whoever launches the process knows where the value comes from — a scheduler, an orchestrator label, an env var of their own — and passes it in. For fabric that launcher path runs through metal (§6.1).

Log the resolved id and its source (`supplied` / `gethostname`) once at discovery. That one line is what an operator will ask for first when a containerized run maps to the wrong FSD host.

---

## 5. Validation (UMD)

Both paths that can put a value on the descriptor — the YAML key and the discovery option — validate, so every cluster id on a descriptor is a legal one:

- non-empty
- `size() <= 128` (fills `PhysicalNodeId::cluster_id`, which is NUL-padded rather than NUL-terminated, so a full-length id needs no room for a terminator)
- every character in `[A-Za-z0-9._-]`, i.e. the §3 pattern

Do **not** require it to match the YAML filename. The filename stays an asset-repo convention; the field is the id.

Do **not** try to enforce uniqueness across ranks from inside UMD. One descriptor = one accelerator group. Metal's `resolve_hostname_uniqueness()` still suffixes `_<rank>` when two ranks report the same string (legacy 16-file superpod reuse). That suffix is a PSD merge key, not something UMD writes back into the YAML. (When ranks share a `cluster_id` because a container fleet supplied the same one to every rank, that suffix is also the symptom to look for.)

---

## 6. Metal consumption (after the UMD bump)

`run_local_discovery` already has `ClusterDescriptor& cluster_desc`. Change `get_local_discovery_hostname()` to take it:

```
get_local_discovery_hostname(cluster_desc):
    if cluster_desc.get_cluster_id() has a value:
        return *that
    if TT_METAL_MOCK_CLUSTER_DESC_PATH is set:
        return path.filename()          // legacy ClosetBox / unfilled YAMLs
    return get_host_name()              // silicon, descriptor not yet stamped (should not happen after UMD)
```

`PhysicalNodeId` packs `canonical_cluster_id_for_node_id` of that string, with **no** aisle-token alias. Mock and FSD agree wherever the descriptor's `cluster_id` and the FSD hostname are the same string, which is a property of each asset rather than something the field enforces (§7).

Metal may rename that function to `get_local_cluster_id()` in the metal PR — the string it returns is now an accelerator-group id, and every metal-side `hostname` identifier is on the §11 list. That rename is cosmetic and does not gate anything.

Keep the filename fallback for any YAML that still lacks the field (the §7 TODO). ClosetBox and every existing mock test stay on the basename until that file is filled. Do not change those tests in the UMD PR.

Metal does **not** open the YAML itself to peek at `cluster_id:`. Always go through `ClusterDescriptor::get_cluster_id()` — one reader of the field, whatever put the value there.

### 6.1 Supplying the id: `TT_METAL_CLUSTER_ID`

UMD reads no environment variable (§4.1), so the container / VM half of the problem (§0) lands on its caller. For fabric that caller is metal: `Cluster::open_driver` is what constructs `tt::umd::Cluster` for silicon, and `ClusterOptions` already embeds `TopologyDiscoveryOptions`.

Metal therefore takes a `TT_METAL_CLUSTER_ID` env var through the usual `rtoptions` registry and passes it straight down:

```cpp
device_driver = std::make_unique<tt::umd::Cluster>(tt::umd::ClusterOptions{
    .num_host_mem_ch_per_mmio_device = std::nullopt,
    .topology_discovery_options = {.cluster_id = rtoptions_.get_cluster_id()},
});
```

Unset on bare metal, where UMD's `gethostname()` is already right. Set by the launcher in a container or a VM, where it is the only way the run can name the accelerator group it actually owns.

That is the **one** place metal reads the variable. Discovery still reads only `ClusterDescriptor::get_cluster_id()`; it does not consult the environment to second-guess what UMD stamped. Metal does not validate the value either — UMD throws on an illegal one (§5), and a second copy of the charset rule in metal would be one more thing to keep in sync.

The one thing metal does decide is that **exported-but-empty counts as unset**, matching how `rtoptions` already reads `TT_METAL_MOCK_CLUSTER_DESC_PATH`. An empty id is illegal, so passing it through would throw; but `export TT_METAL_CLUSTER_ID=$SOME_UNSET_VAR` is a launcher accident rather than a request for an empty cluster id, and failing discovery over it would break bare-metal runs that the OS hostname would have served correctly.

The variable is `TT_METAL_CLUSTER_ID` and not `TT_CLUSTER_ID` because it is metal's knob, sitting in metal's `EnvVarID` registry next to `TT_METAL_MOCK_CLUSTER_DESC_PATH` and the rest. A non-metal application that drives UMD directly sets `TopologyDiscoveryOptions::cluster_id` itself and never sees this variable.

---

## 7. TODO: add `cluster_id:` to all tt-cluster-descriptors YAMLs

**TODO:** Write `cluster_id:` on **every** cluster descriptor in [tt-cluster-descriptors](https://github.com/tenstorrent/tt-cluster-descriptors) (`superclusters/`, `wormhole/`, `blackhole/`, ClosetBox, virtu, T3K, dual-host, … — all ~230 files). Not only FSD-paired BH. Separate repo PR. **Not** required for the UMD PR to merge. Because the field is optional (§1.2), partial fill is fine: filled files use the real id, unfilled files keep the basename fallback.

Do not rename files. Do not edit FSD textprotos.

**Value to write** — the id the captured machine reports for itself. Where that machine has an FSD, its `hosts[].hostname` is the best source, hall included (the current value scheme, §0). Note that the assets as shipped do not all follow that rule: see the caveat under the fill script below.

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
3. Write that host's `hostname` field into the YAML as `cluster_id`.

That join is how we **populate** the field. After the field exists, metal never runs the join.

**The shipped assets do not all carry the FSD string.** In many descriptors `cluster_id` is the filename token instead -- `bh-glx-d03u02` where the FSD calls that machine `bh-glx-110-d03u02`, and `bh-glx-120-d05u20` where the FSD says `bh-glx-110-d09u20`. Do not treat that as a defect to repair: these captures are regression assets, they record the id the machine reported when it was captured, and they are not obliged to track whatever the FSD says today. It does mean the mock/FSD agreement this section once claimed is a property of individual assets rather than a guarantee of the field. What the field does guarantee is that a mock run keys on a string the capture carries about itself, the same kind of string UMD stamps on live silicon, rather than on an asset filename that exists only in the mock path.

After a file has the field, metal never joins on the filename. The aisle-token alias is only for YAMLs that still lack the field.

---

## 8. Tests (UMD, offline)

File: `tests/baremetal/test_cluster_descriptor_cluster_id.cpp` (or cases on `test_cluster_descriptor_offline.cpp`).

- YAML **with** `cluster_id: bh-glx-110-c01u02` → `get_cluster_id() == that string`.
- YAML **without** the key → `get_cluster_id() == nullopt`. **Back-compat:** every existing offline descriptor in the UMD tree still parses (no `cluster_id` required).
- `serialize()` omits the key when unset; includes it when set; parse(serialize(d)) == d.
- A YAML carrying `cluster_id: ""` / `foo/bar` / 129+ chars throws; exactly 128 chars is accepted.
- `create_constrained_cluster_descriptor` and `apply_chip_id_remapping` copy the id.
- `resolve_cluster_id("tt-vm-host-7")` → returns it; `resolve_cluster_id(nullopt)` → returns `gethostname()`; `resolve_cluster_id("a/b")` or 129+ chars → fatal (**not** a silent fallback).
- **The option does not leak into parsing:** it is not an input to `create_from_yaml`, so a YAML *without* the key still yields `nullopt` and a YAML *with* `cluster_id: other-host` still yields `other-host`, whatever the caller would pass to discovery.
- Nanobind: `ClusterDescriptor.create_from_yaml_content(...).get_cluster_id()` and `TopologyDiscoveryOptions.cluster_id`.
- Hardware (optional, `TopologyDiscovery`): after `discover`, `get_cluster_id()` equals `gethostname()` with the option unset, and equals the supplied id when it is set.

No metal test in the UMD PR.

---

## 9. File list (UMD)

- `device/api/umd/device/cluster_descriptor.hpp` — getter, `cluster_id_` (no setter)
- `device/api/umd/device/topology/topology_discovery_options.hpp` — `std::optional<std::string> cluster_id`
- `device/cluster_descriptor.cpp` — parse, serialize, constrain, remap, `create_mock_cluster` unchanged
- `device/topology/topology_discovery.cpp` — stamp in `fill_cluster_descriptor_info`
- `device/common/utils.hpp` — `resolve_cluster_id()` + validation + the one log line
- `docs/yaml_schemas/cluster_descriptor.yaml` — optional `cluster_id`
- `nanobind/py_api_topology_discovery.cpp` — bind the getter and the option
- `tests/baremetal/test_cluster_descriptor_cluster_id.cpp`
- `docs/CLUSTER_ID.md` — what the id is, where the value comes from, and when a caller has to supply one

No change to chip unique ids, ETH tables, or mock filename conventions.

---

## 10. Sequence

1. **UMD PR** — optional `cluster_id` field + query + `TopologyDiscoveryOptions::cluster_id`/`gethostname()` stamp + tests. Old YAMLs still load (back-compat).
2. Bump `tt_metal/third_party/umd` in tt-metal.
3. **tt-metal** — `get_local_discovery_hostname(cluster_desc)` prefers `get_cluster_id()`, basename fallback if absent. Existing tests keep passing.
4. **tt-metal** — `TT_METAL_CLUSTER_ID` through `rtoptions` into `ClusterOptions::topology_discovery_options` (§6.1), so a containerized run can name its accelerator group.
5. **TODO / tt-cluster-descriptors** — write `cluster_id:` on **all** cluster descriptor YAMLs (§7). Can land incrementally; pin when a batch is ready.
6. Drop the aisle-token fallback once FSD-paired BH files are filled.
7. Container / VM enablement: set `TT_METAL_CLUSTER_ID` in the launchers and images that run fabric workloads (no further code change; §11 tracks the rest).

Slice 1 is reviewable with no metal or asset changes. Slices 3 and 4 are independent of each other. Slice 5 does not gate slice 1 or 3.

---

## 11. Direction of travel: off hostnames for ASIC addressing

`cluster_id` is the first step, not the whole move. The end state is that **no** component derives accelerator identity from a machine name; they all carry an opaque `cluster_id` that some higher layer assigns. Getting there is out of scope for this plan, and each item below needs its own change:

| Component | Current hostname assumption | What it needs |
|-----------|------------------------------|---------------|
| FSD (`tt-cluster-descriptors`) | `hosts[].hostname` is the join key | its own `cluster_id` field, populated the same way, hostname kept for humans |
| `PhysicalSystemDescriptor` / `ASICDescriptor` | `host_name` field, `HostName` type, `get_all_hostnames()`, `my_host_name()` | rename to cluster id; the value already flows from `get_cluster_id()` after slice 3. `TopologyMapper` deliberately does **not** do this rename ([`PLAN_physical_node_id.md`](PLAN_physical_node_id.md) §6.1) |
| `PhysicalNodeId` | `cluster_id[128]` NUL-padded, `canonical_cluster_id_for_node_id` | member already renamed; keep the buffer, retire the DNS-label canonicalization once ids are not hostnames |
| Metal discovery | `get_host_name()`, filename basename, `resolve_hostname_uniqueness()` `_<rank>` suffix | id from UMD; uniqueness enforced by whoever assigns ids |
| Rank bindings / launchers | map MPI ranks by hostname; `TopologyMappingConfig::hostname_to_asics` | map by `cluster_id` |
| Distributed bring-up / logs / dashboards | print hostnames | print both while both exist |

Until those land, `cluster_id` **must** stay hostname-valued (§3 pattern, §7 fill rule) — the FSD join and the mock/FSD `PhysicalNodeId` agreement both depend on it. Changing the value scheme before the FSD carries `cluster_id` would re-break the exact thing this plan fixes.

---

## 12. Non-goals

- Per-chip `cluster_id`
- `cluster_id` on `ethernet_connections_to_remote_devices`
- Renaming cluster-desc files to match FSD
- FQDN canonicalization inside UMD
- Requiring the field on every existing YAML in the UMD PR (the §7 TODO is a follow-up)
- Removing the metal basename fallback before the §7 TODO is done
- Breaking old UMD or old metal by adding the key to assets
- A second `hostname:` field, or an alias between `hostname` and `cluster_id`
- Any environment variable inside UMD; and in metal, reading `TT_METAL_CLUSTER_ID` anywhere except the one `open_driver` site that fills `ClusterOptions` (§6.1) — not in discovery, not in YAML parsing, not in `create_mock_cluster`
- A `set_cluster_id()` on `ClusterDescriptor` — the id is fixed when the descriptor is built (§2)
- Changing the value scheme away from hostnames in this plan (§11 is the follow-up, and it starts in the FSD)
- Renaming metal's hostname-typed APIs / PSD fields as part of the UMD PR
