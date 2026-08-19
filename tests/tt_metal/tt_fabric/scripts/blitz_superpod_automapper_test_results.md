# Blitz Superpod Mapping Determinism — Test Results

Recorded: 2026-06-05

## Overview

Mapping determinism testing: end-to-end validation that the Blitz superpod automapper produces a stable ASIC→fabric mapping under input perturbations (mock cluster shuffles, hardware launch-host rotation, and `--hosts` order changes).

| Layer | Test | Status |
|-------|------|--------|
| GTest | `MultiHost.TestBlitzSuperpodAutoMapperControlPlaneInit` | Added |
| Mock / CPU CI | `run_blitz_superpod_automapper_tests.py` (canonical + 5 variations) | Wired in `run_fabric_cpu_only_unit_tests.sh` |
| Real hardware | 20-host SP4 GLX superpod, 64-mesh MGD | Partial run (see below) |

---

## Mock / CPU CI (expected in `run_fabric_cpu_only_unit_tests.sh`)

**Command:**
```bash
TT_METAL_SLOW_DISPATCH_MODE=1 python_env/bin/python3 \
  tests/scripts/multihost/run_blitz_superpod_automapper_tests.py \
  --mock-cluster-rank-binding tests/tt_metal/tt_fabric/custom_mock_cluster_descriptors/sp4_glx_cluster_desc_mapping.yaml \
  --mesh-graph-descriptor tests/tt_metal/tt_fabric/custom_mesh_descriptors/fabric_cpu_only_blitz_superpod_mesh_graph_descriptor.textproto \
  --num-variations 5 --seed 42 \
  --golden tests/tt_metal/tt_fabric/golden_mapping_files/TestBlitzSuperpodAutoMapperControlPlaneInit.yaml
```

**What is exercised per run:**

| Run | Mock rank binding | MGD | mesh_id permutation |
|-----|-------------------|-----|---------------------|
| Canonical | Original SP4 GLX mapping | Canonical superpod MGD | None |
| Variation 0–4 | Shuffled (seed 42) | Relabeled descriptors + permuted mesh_ids | Yes (topology-invariant) |

**GTest checks (canonical only):**
- `MPI rank == mesh_id` for all 64 ranks
- Generated mapping matches golden YAML (`hostname`, `tray_id`, `asic_location`, `mesh_id`, `chip_id`, `asic_id`)

**Python checks (all runs):**
- Exact YAML compare vs golden (or canonical artifact when `--golden` omitted)
- Structured log: `generated/blitz_superpod_automapper/automapper_test.log`

---

## Real Hardware — SP4 GLX Superpod (20 hosts, 64 meshes)

**Cluster:** 20 physical hosts (`bh-glx-c01u02` … `bh-glx-c10u08`)
**Script host:** `bh-glx-c05u08`
**MGD:** `models/demos/deepseek_v3_b1/scaleout_configs/blitz_decode_mesh_graph_descriptor_superpod.textproto`
**Command:**
```bash
python_env/bin/python3 tests/scripts/multihost/run_blitz_superpod_automapper_tests.py \
  --hosts bh-glx-c01u02,...,bh-glx-c10u08 \
  --mesh-graph-descriptor models/demos/deepseek_v3_b1/scaleout_configs/blitz_decode_mesh_graph_descriptor_superpod.textproto \
  --tcp-interface ens5f0np0 --num-variations 10 --seed 42
```

### Results summary

| Run | Launch host | Remote SSH | Exit | Mapping compare |
|-----|-------------|------------|------|-----------------|
| canonical | bh-glx-c05u08 (local) | no | 0 | (reference captured) |
| variation_0 | bh-glx-c01u08 | yes | 0 | PASS |
| variation_1 | bh-glx-c02u02 | yes | 0 | PASS |
| variation_2 | bh-glx-c02u08 | yes | 0 | PASS |
| variation_3 | bh-glx-c03u02 | yes | 0 | PASS |
| variation_4 | bh-glx-c03u08 | yes | 0 | PASS |
| variation_5 | bh-glx-c04u02 | yes | 0 | PASS |
| variation_6 | bh-glx-c04u08 | yes | 0 | PASS |
| variation_7 | bh-glx-c05u02 | — | — | **In progress / interrupted** |
| variation_8–9 | — | — | — | Not run |

**Mapping invariance:** All completed variation artifacts are byte-identical to the canonical mapping (MD5 `d86ba39b6e360594af7d36bc34aca9c8`). This is the expected pass condition — physical ASIC placement must not change across perturbations.

**Artifacts saved under:** `generated/blitz_superpod_hardware_variations/{canonical,variation_N}/`

**Full structured log:** `generated/blitz_superpod_automapper/automapper_test.log`

### Notes on hardware variation design

- **Launch host** rotates across cluster nodes (variations SSH tt-run to a peer host).
- **`--hosts` order** is shuffled per variation in the logged command; tt-run Phase 1 fingerprints hosts in **sorted** order, so MPI rank bindings / rankfile are stable across permutations of the same host set. The primary hardware perturbation under test is **launch-host rotation**.
- GTest golden compare is skipped on real silicon (`TT_METAL_BLITZ_SUPERPOD_VARIATION=1` for variations); Python strict compare is used instead.

---

## Compare fields (canonical and variations)

All of the following must match the reference for each `fabric_node_id` key:

- `hostname`
- `tray_id` / `asic_location`
- `mesh_id` / `chip_id`
- `asic_id`
