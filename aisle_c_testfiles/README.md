# Aisle C — disaggregated prefill / decode (host-to-host)

This directory holds **Aisle C** hostnames (`bh-glx-c…`) for the Blitz-style decode pipeline rank layout, plus a **host-to-host** rank-bindings mapping: sub-context **0** uses `blitz_decode_pipeline_rank_binding_superpod.yaml` (decode-scale mesh graph), sub-context **1** uses `single_bh_galaxy_one_rank_rank_bindings.yaml` (single BH Galaxy mesh).

## SP5 mock cluster mapping (MPI world rank → PSD)

**`sp5_cluster_desc_mapping.yaml`** matches the lab layout: **20** ranks (**0–19**) map to **`bh-glx-c01u02` … `bh-glx-c10u08`** with per-rank YAML under **`sp5_cluster_descs/`** (paths are relative to **`TT_METAL_HOME`**). Use this with **`--mock-cluster-rank-binding`** when driving tests that expect one physical cluster descriptor per MPI rank (same content as the repo-root `sp5_cluster_desc_mapping.yaml` if present).

The **Blitz superpod** rank file **`blitz_decode_pipeline_rank_file_superpod`** matches **`generated/ttrun/.../phase2_mock_mapping.yaml`** (and **`sp5_cluster_descs/`**): MPI ranks **0–63** use **16** GLx hosts (**`c01`–`c04`** and **`c07`–`c10`**, four ranks per host), then rank **64** is the single-Galaxy prefill placeholder.

## What to fill in `runme.sh`

`runme.sh` runs **`example_disaggregated_prefill_decode_cross_context`** with **65** mock ranks: **64** decode stages (sub-context **0**, **FABRIC_2D**) and **1** prefill process (sub-context **1**, **FABRIC_1D**). Cross-sub-context KV goes **only** from prefill rank **0** to decode rank **0** (see source: **`tt_metal/programming_examples/distributed/5_disaggregated_prefill_decode_cross_context/example_disaggregated_prefill_decode_cross_context.cpp`**).

Config:

- **`mock_bh_6u_65_rank_prefill_decode_cluster_desc_mapping.yaml`** — per-rank PSD mock (65× `bh_6u_cluster_desc`).
- **`disaggregated_prefill_decode_1_prefill_64_decode_rank_bindings_mapping.yaml`** — single-rank prefill overlay + **64-rank** Blitz superpod decode overlay (`disaggregated_prefill_decode_blitz_superpod_64_decode_host_rank_bindings.yaml`).

Adjust **`HOSTSP`**, **`HOSTS`**, and **rank 64** in **`blitz_decode_pipeline_rank_file_superpod`** if your prefill host or machine list differs.

| Item | What it is |
|------|------------|
| **`TT_METAL_HOME`** | Export before running (e.g. `export TT_METAL_HOME="$PWD"` from the repo root). |
| **`HOSTSP`** | Comma-separated hosts **with slot counts** (e.g. `:4` per GLx), used by **`tt-run`** when mapping by rankfile / `--host`. |
| **`HOSTS`** | Comma-separated hostnames **without** slot counts, used by plain **`mpirun`** / **`mpirun-ulfm`** (GLx reset and cluster validation in `runme.sh`). |
| **`--tcp-interface ens5f0np0`** | Passed on every **`tt-run`** line in `runme.sh` so MPI TCP uses that interface (matches **`mpirun-ulfm`**’s **`btl_tcp_if_include`**). |
| **Rank 64** | In **`blitz_decode_pipeline_rank_file_superpod`**, must match the prefill / single-Galaxy host in **`HOSTSP`**. |

## Cluster prep in `runme.sh` (before `tt-run`)

`runme.sh` runs these on **`HOSTS`** (see the script for the exact list):

1. **`mpirun --host $HOSTS tt-smi -glx_reset`** — GLx reset on all listed hosts.
2. **`mpirun-ulfm --host $HOSTS --mca btl_tcp_if_include ens5f0np0 --tag-output ./build/tools/scaleout/run_cluster_validation ...`** — cabling/deployment descriptors under **`/data/scaleout_configs/SP5/`**, **`--send-traffic`**, **`--num-iterations 5`**.

Comment those lines out if you only want the example / fabric **`tt-run`** without reset or validation. Ensure **`run_cluster_validation`** is built and that **`/data/scaleout_configs/SP5/*.textproto`** exist on the runners.

**Build:** `cmake --build build --target example_disaggregated_prefill_decode_cross_context` (with **`ENABLE_DISTRIBUTED`**). For the optional fabric test, build **`fabric_unit_tests`**.

**GTest (2+2 quad mock, different layout):** `distributed_unit_tests --gtest_filter='MpiSubContext.*'` still uses **`disaggregated_prefill_decode_host_2_host_rank_bindings_mapping.yaml`** and **`mock_galaxy_quad_2x4_four_rank_cluster_desc_mapping.yaml`** — not the 1+65 example above.

Example **`HOSTSP`** (matches `runme.sh`; last entry is the rank-64 / prefill host with **`:1`**):

```bash
export HOSTSP=bh-glx-c01u02:4,bh-glx-c01u08:4,bh-glx-c02u02:4,bh-glx-c02u08:4,bh-glx-c03u02:4,bh-glx-c03u08:4,bh-glx-c04u02:4,bh-glx-c04u08:4,bh-glx-c07u02:4,bh-glx-c07u08:4,bh-glx-c08u02:4,bh-glx-c08u08:4,bh-glx-c09u02:4,bh-glx-c09u08:4,bh-glx-c10u02:4,bh-glx-c10u08:4,bh-glx-c05u02:1
```

Example **`HOSTS`** (matches `runme.sh`; no slot suffixes):

```bash
export HOSTS=bh-glx-c01u08,bh-glx-c02u02,bh-glx-c02u08,bh-glx-c03u02,bh-glx-c03u08,bh-glx-c04u02,bh-glx-c04u08,bh-glx-c05u02,bh-glx-c05u08,bh-glx-c06u02,bh-glx-c06u08,bh-glx-c07u02,bh-glx-c07u08,bh-glx-c08u02,bh-glx-c08u08,bh-glx-c09u02,bh-glx-c09u08,bh-glx-c10u02,bh-glx-c10u08
```

## Rank file vs mock cluster descriptor

- **Hardware / multi-host runs** use the **MPI rank file** `blitz_decode_pipeline_rank_file_superpod` with `--map-by rankfile:file=…` and **`--rank-bindings-mapping disaggregated_prefill_decode_host_2_host_rank_bindings_mapping.yaml`** (or the aisle-specific mapping).
  Do **not** pass `--mock-cluster-rank-binding` for that flow; placement comes from the rank file + hosts.

- **`sp5_cluster_desc_mapping.yaml`** is a **per-rank mock PSD cluster YAML** mapping for **20** Aisle C GLx hosts (ranks **0–19**). Use it for mock / bring-up where each MPI rank needs the captured PSD for that machine.

## Run

```bash
export TT_METAL_HOME="$PWD"
# Build example_disaggregated_prefill_decode_cross_context (and run_cluster_validation if using prep steps) first.
# Edit HOSTSP, HOSTS, and rank 64 if needed.
./aisle_c_testfiles/runme.sh
```

All **`tt-run`** invocations in `runme.sh` use **`--tcp-interface ens5f0np0`**.

## Other aisles

- **Aisle A** (`bh-glx-120-a…`): **`aisle_a_testfiles/README.md`**
- **Aisle D** (`bh-glx-d…`): **`aisle_d_testfiles/README.md`**
