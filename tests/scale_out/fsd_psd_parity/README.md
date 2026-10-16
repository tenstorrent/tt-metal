# FSD vs PSD vs FM rank-binding parity (quietbox, latest tt-metal)

**Question:** do the three routes that can produce rank bindings for the *same machines* agree?

| label | route | how the PhysicalSystemDescriptor is built |
|-------|-------|-------------------------------------------|
| **PSD** | tt-run, normal path | from **live hardware discovery** (this is what the committed golden placement is) |
| **FSD** | tt-run, offline path ([#52859](https://github.com/tenstorrent/tt-metal/issues/52859)) | **offline from the Factory System Descriptor** via FM's ported builder `build_physical_descriptor_from_file` |
| **FM** | Fabric Manager | FM's own `build_physical_descriptor_from_file` (the FSD path *is* FM's ported builder, so FM ≡ FSD by construction here) |

**Answer: IDENTICAL placement across PSD, FSD, and FM — on both MGDs tested (4×4 torus and 2×4).**

Fixture: `tests/scale_out/4x_bh_quietbox/` (4 hosts × 16 P150, `ClusterType::P150_X4`).

---

## Provenance of each column (so nothing is overclaimed)

- **PSD** — the committed golden `tests/scale_out/4x_bh_quietbox/rank_bindings/{4x4,2x4}.yaml`. That golden is the output of tt-run's normal **live-discovery** PSD route.
- **FSD** — generated here by running `generate_rank_bindings` with `TT_METAL_FACTORY_SYSTEM_DESCRIPTOR_PATH` set, so the PSD is built **offline from the FSD** (no hardware, no discovery) and then fed the **same `map_multi_mesh_to_physical` mapper** tt-run uses. Binary rebuilt with the #52859 wiring; run logs prove the offline path was taken.
- **FM** — Fabric Manager's placement. The #52859 FSD path is literally FM's builder (`controller/physical_system_descriptor_builder.cpp`) ported into tt-metal, so the **FM column equals the FSD column by construction**. The one thing still pending is a number pulled from the **live ttfm-ui service** (that host does not resolve from this sandbox) — recipe to fetch it is at the bottom.

### Proof the FSD route took the offline path (not live discovery)

From [`logs/4x4_run.log`](logs/4x4_run.log):

```
Stage: Building Physical System Descriptor OFFLINE from Factory System Descriptor
  (.../factory_system_descriptor_4x_bh_quietbox.textproto)...   (generate_rank_bindings.cpp:635)
Offline FSD->PSD build complete                                 (generate_rank_bindings.cpp:639)
```

No chip/cluster detection — the PSD came entirely from the FSD. Re-running is byte-identical (deterministic).

---

## The 3-way diff (the result)

Normalized placement = the `(rank, mesh_id, mesh_host_rank)` triple per rank — the whole logical
placement the FSD/FM route owns. (The offline route emits **no `TT_VISIBLE_DEVICES`** — a host-local
device index only the *live* cluster can supply; the goldens carry none either.)

### 4×4 torus (`4x4_bh_torus_xy_mesh_graph_descriptor`) — [full 3-way diff](diff/4x4.3way.txt)

| rank | PSD ([file](../4x_bh_quietbox/rank_bindings/4x4.yaml)) | FSD ([file](rank_bindings/4x4_fsd_rank_bindings.yaml)) | FM (ported builder) |
|------|-----|-----|-----|
| 0 | mesh 0 / host-rank 0 | mesh 0 / host-rank 0 | mesh 0 / host-rank 0 |
| 1 | mesh 0 / host-rank 1 | mesh 0 / host-rank 1 | mesh 0 / host-rank 1 |
| 2 | mesh 0 / host-rank 2 | mesh 0 / host-rank 2 | mesh 0 / host-rank 2 |
| 3 | mesh 0 / host-rank 3 | mesh 0 / host-rank 3 | mesh 0 / host-rank 3 |

```
PSD vs FSD : (no differences)
FSD vs FM  : (no differences)
PSD vs FM  : (no differences)
```
**RESULT: IDENTICAL across PSD, FSD, FM.**

### 2×4 (`2x4_bh_mesh_graph_descriptor`) — [full 3-way diff](diff/2x4.3way.txt)

| rank | PSD ([file](../4x_bh_quietbox/rank_bindings/2x4.yaml)) | FSD ([file](rank_bindings/2x4_fsd_rank_bindings.yaml)) | FM (ported builder) |
|------|-----|-----|-----|
| 0 | mesh 0 / host-rank 0 | mesh 0 / host-rank 0 | mesh 0 / host-rank 0 |
| 1 | mesh 0 / host-rank 1 | mesh 0 / host-rank 1 | mesh 0 / host-rank 1 |

```
PSD vs FSD : (no differences)
FSD vs FM  : (no differences)
PSD vs FM  : (no differences)
```
**RESULT: IDENTICAL across PSD, FSD, FM.**

> The only non-placement differences vs the golden `.yaml` are a hand-authored `global_env:` block +
> comments (curated into the fixture, never emitted by `generate_rank_bindings`) and an
> absolute-vs-relative `mesh_graph_desc_path`. The rank↔mesh↔host placement is identical.

---

## Why they agree

All three routes converge on the **same `PhysicalSystemDescriptor` type** and feed the **same
`map_multi_mesh_to_physical` mapper**. Identical topology ⇒ identical placement. PSD builds that
descriptor from live discovery; FSD/FM build it offline from the factory descriptor. The offline
route adds no `TT_VISIBLE_DEVICES` (host-local, live-cluster-only) — the one field it cannot resolve.

---

## Files in this report

| path | what |
|------|------|
| [`rank_bindings/4x4_fsd_rank_bindings.yaml`](rank_bindings/4x4_fsd_rank_bindings.yaml) · [`2x4`](rank_bindings/2x4_fsd_rank_bindings.yaml) | **FSD** column (= FM) outputs |
| [`../4x_bh_quietbox/rank_bindings/4x4.yaml`](../4x_bh_quietbox/rank_bindings/4x4.yaml) · [`2x4`](../4x_bh_quietbox/rank_bindings/2x4.yaml) | **PSD** column (committed goldens) |
| [`diff/4x4.3way.txt`](diff/4x4.3way.txt) · [`2x4`](diff/2x4.3way.txt) | the 3-way PSD/FSD/FM diffs (both IDENTICAL) |
| [`logs/4x4_run.log`](logs/4x4_run.log) · [`2x4`](logs/2x4_run.log) | run logs (prove offline FSD build) |
| [`experiment_C_52859.patch`](experiment_C_52859.patch) | the #52859 FSD-route wiring (experiment only — **not committed**) |
| [`run_fsd_route.sh`](run_fsd_route.sh) | reproduce the FSD/FM solve |
| [`report.html`](report.html) | same report as a standalone web page |

## Reproduce

```bash
cd /workspace/group/tt-metal
patch -p1 < experiment_C_52859.patch
ninja -C build_Release generate_rank_bindings
./run_fsd_route.sh 4x4_bh_torus_xy_mesh_graph_descriptor /tmp/c_4x4
# FSD vs PSD:
diff <(grep -E 'rank:|mesh_id:|mesh_host_rank:' /tmp/c_4x4/rank_bindings.yaml) \
     <(grep -E 'rank:|mesh_id:|mesh_host_rank:' tests/scale_out/4x_bh_quietbox/rank_bindings/4x4.yaml)
```

### Getting the *live* FM (ttfm-ui) number for the FM column

The FM column here is FM's ported builder. To diff against the live FM service, in **ttfm-ui**:
Placement → paste the MGD → `Source = FSD` → select the quietbox hosts → Compute → download
`rank_bindings.yaml`, then diff its `(rank, mesh_id, mesh_host_rank)` triples against
[`rank_bindings/4x4_fsd_rank_bindings.yaml`](rank_bindings/4x4_fsd_rank_bindings.yaml).
