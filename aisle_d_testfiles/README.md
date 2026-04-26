# Aisle D — disaggregated prefill / decode (host-to-host)

This directory holds **Aisle D** hostnames (`bh-glx-d…`) for the Blitz-style decode pipeline rank layout, plus a **host-to-host** rank-bindings mapping: sub-context **0** uses `blitz_decode_pipeline_rank_binding_superpod.yaml` (decode-scale mesh graph), sub-context **1** uses `single_bh_galaxy_one_rank_rank_bindings.yaml` (single BH Galaxy mesh).

## What to fill in `runme.sh`

`runme.sh` runs **`example_disaggregated_prefill_decode_cross_context`** with **65** mock ranks: **1** prefill process (sub-context 0, **FABRIC_1D**) and **64** decode stages (sub-context 1, **FABRIC_2D**). Cross-sub-context KV goes **only** from prefill rank **0** to decode rank **0** (see source: **`tt_metal/programming_examples/distributed/5_disaggregated_prefill_decode_cross_context/example_disaggregated_prefill_decode_cross_context.cpp`**).

Config:

- **`mock_bh_6u_65_rank_prefill_decode_cluster_desc_mapping.yaml`** — per-rank PSD mock (65× `bh_6u_cluster_desc`).
- **`disaggregated_prefill_decode_1_prefill_64_decode_rank_bindings_mapping.yaml`** — single-rank prefill overlay + **64-rank** Blitz superpod decode overlay (`disaggregated_prefill_decode_blitz_superpod_64_decode_host_rank_bindings.yaml`).

Replace **`<YOUR_BH_GALAXY_HOST>`** in **`HOSTSP`** and on **rank 64** in the rank file when using the **optional** hardware rankfile line at the bottom of `runme.sh`.

| Item | What it is |
|------|------------|
| **`TT_METAL_HOME`** | Export before running (e.g. `export TT_METAL_HOME="$PWD"` from the repo root). |
| **`HOSTSP`** | Pre-filled with 16 GLx hosts (`:4` each) + **`<YOUR_BH_GALAXY_HOST>:1`**. Used by the **optional** `fabric_unit_tests` command (commented in `runme.sh`). |
| **Rank 64** | In **`blitz_decode_pipeline_rank_file_superpod`**, set **`rank 64=<same-as-HOSTSP-galaxy> slot=0-31`**. |

**`HOSTS` vs `HOSTSP`:** the optional hardware flow uses **`HOSTSP`** (host **with slot counts**).

**Build:** `cmake --build build --target example_disaggregated_prefill_decode_cross_context` (with **`ENABLE_DISTRIBUTED`**). For the optional fabric test, build **`fabric_unit_tests`**.

**GTest (2+2 quad mock, different layout):** `distributed_unit_tests --gtest_filter='MpiSubContext.*'` still uses **`disaggregated_prefill_decode_host_2_host_rank_bindings_mapping.yaml`** and **`mock_galaxy_quad_2x4_four_rank_cluster_desc_mapping.yaml`** — not the 1+65 example above.

Example **`HOSTSP`** (same as in `runme.sh`; replace `<YOUR_BH_GALAXY_HOST>`):

```bash
export HOSTSP=bh-glx-d03u08:4,bh-glx-d03u02:4,bh-glx-d04u02:4,bh-glx-d04u08:4,bh-glx-d06u08:4,bh-glx-d06u02:4,bh-glx-d05u08:4,bh-glx-d05u02:4,bh-glx-d07u02:4,bh-glx-d07u08:4,bh-glx-d08u02:4,bh-glx-d08u08:4,bh-glx-d09u02:4,bh-glx-d09u08:4,bh-glx-d10u02:4,bh-glx-d10u08:4,<YOUR_BH_GALAXY_HOST>:1
```

## Rank file vs mock cluster descriptor

- **Hardware / multi-host runs** use the **MPI rank file** `blitz_decode_pipeline_rank_file_superpod` with `--map-by rankfile:file=…` and **`--rank-bindings-mapping disaggregated_prefill_decode_host_2_host_rank_bindings_mapping.yaml`** (or the aisle-specific mapping).
  Do **not** pass `--mock-cluster-rank-binding` for that flow; placement comes from the rank file + hosts.

- **`tests/tt_metal/tt_fabric/custom_mock_cluster_descriptors/sp4_glx_cluster_desc_mapping_blitz_decode_rank_order.yaml`** is a **per-rank mock PSD cluster YAML** mapping used for **mock / CPU-side** bring-up (e.g. CI). It is **not** a substitute for the rank file when mapping real hosts with OpenMPI.

## Run

```bash
export TT_METAL_HOME="$PWD"
# Build example_disaggregated_prefill_decode_cross_context first; edit HOSTSP/rank 64 if using optional fabric line
./aisle_d_testfiles/runme.sh
```

## Aisle A

**Aisle A** (`bh-glx-120-a…`): **`aisle_a_testfiles/runme.sh`** and **`aisle_a_testfiles/README.md`**.
