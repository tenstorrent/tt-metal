# Aisle A — disaggregated prefill / decode (host-to-host)

This directory holds **Aisle A** hostnames (`bh-glx-120-a…`) for the Blitz-style rank file and host-to-host mappings, plus the same **`disaggregated_prefill_decode_host_2_host_rank_bindings_mapping.yaml`** overlay as the other aisles (Blitz decode + single BH Galaxy rank).

## What to fill in `runme.sh`

`runme.sh` runs **`example_disaggregated_prefill_decode_cross_context`** with **65** mock ranks: **64** decode stages (sub-context **0**, **FABRIC_2D**) and **1** prefill process (sub-context **1**, **FABRIC_1D**). Cross-sub-context KV goes **only** from prefill rank **0** to decode rank **0** (see **`tt_metal/programming_examples/distributed/5_disaggregated_prefill_decode_cross_context/example_disaggregated_prefill_decode_cross_context.cpp`**).

Config (same as repo defaults for the example):

- **`mock_bh_6u_65_rank_prefill_decode_cluster_desc_mapping.yaml`**
- **`disaggregated_prefill_decode_1_prefill_64_decode_rank_bindings_mapping.yaml`**

Replace **`<YOUR_BH_GALAXY_HOST>`** in **`HOSTSP`** and on **rank 64** in **`blitz_decode_pipeline_rank_file_superpod`** when using the **optional** hardware rankfile block at the bottom of `runme.sh`.

| Item | What it is |
|------|------------|
| **`TT_METAL_HOME`** | Export before running (e.g. `export TT_METAL_HOME="$PWD"` from the repo root). |
| **`HOSTSP`** | Pre-filled with 16 Aisle A GLx hosts (`:4` each) + **`<YOUR_BH_GALAXY_HOST>:1`**. Used by the **optional** `fabric_unit_tests` command (commented in `runme.sh`). |
| **Rank 64** | In **`blitz_decode_pipeline_rank_file_superpod`**, set **`rank 64=<same-as-HOSTSP-galaxy> slot=0-31`**. |

**Build:** `cmake --build build --target example_disaggregated_prefill_decode_cross_context` (with **`ENABLE_DISTRIBUTED`**).

Example **`HOSTSP`** (same as in `runme.sh`; replace `<YOUR_BH_GALAXY_HOST>`):

```bash
export HOSTSP=bh-glx-120-a02u08:4,bh-glx-120-a02u02:4,bh-glx-120-a03u02:4,bh-glx-120-a03u08:4,bh-glx-120-a04u02:4,bh-glx-120-a04u08:4,bh-glx-120-a05u02:4,bh-glx-120-a05u08:4,bh-glx-120-a06u02:4,bh-glx-120-a06u08:4,bh-glx-120-a07u02:4,bh-glx-120-a07u08:4,bh-glx-120-a08u02:4,bh-glx-120-a08u08:4,bh-glx-120-a09u02:4,bh-glx-120-a09u08:4,<YOUR_BH_GALAXY_HOST>:1
```

## Run

```bash
export TT_METAL_HOME="$PWD"
./aisle_a_testfiles/runme.sh
```

## Aisle D

**Aisle D** (`bh-glx-d…`): **`aisle_d_testfiles/README.md`** and **`aisle_d_testfiles/runme.sh`**.

## Aisle C

**Aisle C** (`bh-glx-c…`): **`aisle_c_testfiles/README.md`** and **`aisle_c_testfiles/runme.sh`**.
