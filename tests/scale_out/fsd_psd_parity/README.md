# Experiment C — FSD-route rank bindings == golden (the FM route, proven locally)

**Question:** does solving rank bindings from a **Factory System Descriptor (FSD)** — the route
**Fabric Manager (FM)** uses — produce the *same* rank bindings as the committed golden placement
for the same machines, on **latest tt-metal**?

**Answer: yes — byte-identical placement on both MGDs tested (4x4 and 2x4).**

---

## What was actually run

`tt-run`'s Phase-1 tool `generate_rank_bindings` normally builds a **PhysicalSystemDescriptor (PSD)**
from **live hardware discovery** (or a mock cluster descriptor). FM instead builds the PSD **offline
from the FSD** (the desired/as-built topology) and feeds it to the topology mapper.

For this experiment `generate_rank_bindings` was wired (issue
[#52859](https://github.com/tenstorrent/tt-metal/issues/52859)) to take the **FM path**: when
`TT_METAL_FACTORY_SYSTEM_DESCRIPTOR_PATH` is set it builds the PSD with FM's **own ported builder**
`build_physical_descriptor_from_file` (ported from tt-fabric-manager
`controller/physical_system_descriptor_builder.cpp`) — **no hardware, no live discovery** — and then
runs the **same `map_multi_mesh_to_physical` mapper tt-run uses**. So this is the FM (FSD) route,
computed locally and compared against the committed goldens.

Fixture: `tests/scale_out/4x_bh_quietbox/` (4 hosts × 16 P150, built as `ClusterType::P150_X4`).

### Proof the offline FSD path was taken (not a live-discovery fallback)

From [`logs/4x4_run.log`](logs/4x4_run.log):

```
Stage: Building Physical System Descriptor OFFLINE from Factory System Descriptor
  (.../factory_system_descriptor_4x_bh_quietbox.textproto)...   (generate_rank_bindings.cpp)
Offline FSD->PSD build complete
```

No `run_physical_system_discovery`, no cluster/chip detection — the PSD came entirely from the FSD.
Re-running produced a **byte-identical** `rank_bindings.yaml` (deterministic).

---

## The diff (this is the result)

Normalized placement = the `(rank, mesh_id, mesh_host_rank)` triple per rank, which is the whole
logical placement the FSD route owns. (The FSD route emits **no `TT_VISIBLE_DEVICES`** — that is a
host-local device index only the *live* cluster can supply, so the FSD route is logical placement
only. The committed goldens likewise carry no `TT_VISIBLE_DEVICES`.)

### 4×4 torus (`4x4_bh_torus_xy_mesh_graph_descriptor`) — [full diff](diff/4x4.diff.txt)

| source | rank 0 | rank 1 | rank 2 | rank 3 |
|--------|--------|--------|--------|--------|
| **FSD route** ([file](rank_bindings/4x4_fsd_rank_bindings.yaml)) | mesh 0 / host-rank 0 | mesh 0 / host-rank 1 | mesh 0 / host-rank 2 | mesh 0 / host-rank 3 |
| **golden** ([file](../4x_bh_quietbox/rank_bindings/4x4.yaml)) | mesh 0 / host-rank 0 | mesh 0 / host-rank 1 | mesh 0 / host-rank 2 | mesh 0 / host-rank 3 |

```diff
--- diff/4x4_golden.norm
+++ diff/4x4_fsd.norm
(no differences)
```
**RESULT: IDENTICAL placement.**

### 2×4 (`2x4_bh_mesh_graph_descriptor`) — [full diff](diff/2x4.diff.txt)

| source | rank 0 | rank 1 |
|--------|--------|--------|
| **FSD route** ([file](rank_bindings/2x4_fsd_rank_bindings.yaml)) | mesh 0 / host-rank 0 | mesh 0 / host-rank 1 |
| **golden** ([file](../4x_bh_quietbox/rank_bindings/2x4.yaml)) | mesh 0 / host-rank 0 | mesh 0 / host-rank 1 |

```diff
--- diff/2x4_golden.norm
+++ diff/2x4_fsd.norm
(no differences)
```
**RESULT: IDENTICAL placement.**

> The only non-placement differences vs the golden `.yaml` files are a hand-authored `global_env:`
> block and comments (curated into the fixture, never emitted by `generate_rank_bindings`) and an
> absolute-vs-relative `mesh_graph_desc_path`. The rank↔mesh↔host placement is identical.

---

## Files in this report

| path | what |
|------|------|
| [`rank_bindings/4x4_fsd_rank_bindings.yaml`](rank_bindings/4x4_fsd_rank_bindings.yaml) | FSD-route output, 4×4 torus |
| [`rank_bindings/2x4_fsd_rank_bindings.yaml`](rank_bindings/2x4_fsd_rank_bindings.yaml) | FSD-route output, 2×4 |
| [`logs/4x4_run.log`](logs/4x4_run.log) / [`2x4_run.log`](logs/2x4_run.log) | run logs (prove offline FSD build) |
| [`../4x_bh_quietbox/rank_bindings/4x4.yaml`](../4x_bh_quietbox/rank_bindings/4x4.yaml) / [`2x4_golden.yaml`](../4x_bh_quietbox/rank_bindings/2x4.yaml) | committed goldens under test |
| [`diff/4x4.diff.txt`](diff/4x4.diff.txt) / [`diff/2x4.diff.txt`](diff/2x4.diff.txt) | the diffs (both IDENTICAL) |
| [`experiment_C_52859.patch`](experiment_C_52859.patch) | the #52859 FSD-route wiring (experiment only — **not committed**) |
| [`run_fsd_route.sh`](run_fsd_route.sh) | reproduce the FSD-route solve |
| [`report.html`](report.html) | same report as a standalone web page |

## Reproduce

```bash
cd /workspace/group/tt-metal
patch -p1 < /workspace/group/fsd_psd_parity/experiment_C/experiment_C_52859.patch
ninja -C build_Release generate_rank_bindings
/workspace/group/fsd_psd_parity/experiment_C/run_fsd_route.sh 4x4_bh_torus_xy_mesh_graph_descriptor /tmp/c_4x4
diff <(grep -E 'rank:|mesh_id:|mesh_host_rank:' /tmp/c_4x4/rank_bindings.yaml) \
     <(grep -E 'rank:|mesh_id:|mesh_host_rank:' tests/scale_out/4x_bh_quietbox/rank_bindings/4x4.yaml)
```

## How this fits the bigger parity question

- **A (done earlier):** PSD route with vs without `TT_METAL_FACTORY_SYSTEM_DESCRIPTOR_PATH` set →
  byte-identical (on stock `main` the env var is inert; FSD-driven mapping is gated behind #52859).
- **B (done earlier):** a deliberately broken mock cluster descriptor is caught cleanly on the bad
  rank (`yaml-cpp` parse error / "Invalid or missing 'boards' node.") — no silent mis-placement; the
  whole MPI job fails rather than placing wrongly.
- **C (this report):** the **FSD route == golden**, exactly. Both the PSD route and the FSD route
  converge on the **same `PhysicalSystemDescriptor` type** and feed the **same topology mapper**, so
  identical topology ⇒ identical placement. The FSD route adds no `TT_VISIBLE_DEVICES` (host-local,
  live-cluster-only) — that is the one field the offline route cannot and does not resolve.

### Getting the *real* FM (ttfm-ui) numbers to diff against

This experiment runs FM's own ported builder as the FM stand-in. To diff against the live FM service,
in **ttfm-ui**: Placement → paste the MGD → `Source = FSD` → select the quietbox hosts → Compute →
download `rank_bindings.yaml`, then `diff` its `(rank, mesh_id, mesh_host_rank)` triples against
[`rank_bindings/4x4_fsd_rank_bindings.yaml`](rank_bindings/4x4_fsd_rank_bindings.yaml).
