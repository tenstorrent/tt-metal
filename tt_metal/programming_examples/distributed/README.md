# Distributed Programming Examples

This directory contains examples of distributed programming model using the TT-Metalium API.

They are intended to be simple demonstrations for distributed program dispatch, distributed memory management, and end-to-end distributed program execution.

Users familiar with the single-device TT-Metal programming model will find the distributed programming model to be a natural extension.

## Multihost, logs, Inspector, and triage

When running with **MPI** (multiple host processes), logs and debug tools must stay aligned with **which rank** owns which devices.

### Rank-scoped log paths

If `TT_METAL_LOGS_PATH` is set to a shared directory (for example on NFS), the runtime may write under:

`<TT_METAL_LOGS_PATH>/<hostname>_rank_<N>/generated/inspector`

**tt-triage** resolves this layout automatically when standard launcher variables are set (`OMPI_COMM_WORLD_RANK`, `PMI_RANK`, `SLURM_PROCID`, `PMIX_RANK`, or `TT_MESH_HOST_RANK`). Prefer Open MPI / PMI rank variables when both they and `TT_MESH_HOST_RANK` are present, since the mesh host rank may not be unique per world rank.

### Inspector RPC port

The effective Inspector gRPC port is **base port + MPI world rank** (default base **50051**). Override with `TT_METAL_INSPECTOR_RPC_SERVER_ADDRESS` if needed; ensure `base + max_rank ≤ 65535`.

### tt-run (TTNN)

For Python distributed launches, **tt-run** (`ttnn.distributed.ttrun`) rank-scopes `TT_METAL_LOGS_PATH` and `TT_METAL_JIT_SCRATCH`, keeps `TT_METAL_CACHE` shared by default, strips CI/runner noise from the MPI launcher environment, and matches the Inspector port rule above when using the default RPC port. See `tt-run --help`.

"Rank-scoping" means tt-run rewrites the path to include `<hostname>_rank_<N>`, so each rank writes to its own sub-directory. For example, if `TT_METAL_LOGS_PATH=/shared/logs`, rank 3 on host `node1` writes to `/shared/logs/node1_rank_3/generated/inspector/`.

### Further reading

Authoritative detail: [``triage.rst``](../../../docs/source/tt-metalium/tools/triage.rst) (Sphinx: **Tools → tt-triage**).
