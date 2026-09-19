# Gemma4 prefill test flows

Both tests use real TT hardware, allocate six KV slots, and prefill slot 0 with the exact token prefix from the GPU capture. KV stays resident on the device during validation; only logs, address metadata, and PCC reports are written to disk.

## Mock migration

`mock` exports the migration address table and checks its addresses without transferring KV between slots. Model execution and GPU comparison are real.

```mermaid
flowchart TD
    R["Runner loads model<br/>Allocates six KV slots"] --> T["Exports KV address table<br/>and device map"]
    T --> P["Producer sends exact GPU-capture tokens<br/>to slot 0"]
    P --> F["TT runs prefill chunk by chunk<br/>Writes KV into slot 0"]
    F --> A["Producer drains device layer acknowledgments"]
    A --> S["Producer sends shutdown sentinel"]
    S --> H["Runner's test hook runs<br/>Mesh and KV remain alive"]

    H --> Q["TTNN reads slot 0's full prefix<br/>Host gathers and reorders shards"]
    Q --> C["Compare all heads and 60 layers<br/>against GPU KV using PCC"]
    G["GPU KV safetensors"] --> C

    Q --> V["Check sampled migration-table addresses<br/>UMD bytes must match gathered TT values"]
    T -.-> V

    C --> D["Pass checks, then close mesh"]
    V --> D
```

## Loopback migration

`loopback` copies KV from slot 0 to slot 5 through the migration endpoint's internal sender and receiver workers on the same machine. It verifies destination bytes before running the GPU comparison once on the source slot.

```mermaid
flowchart TD
    E["Start migration endpoint<br/>Internal sender and receiver workers"] --> R["Runner loads model<br/>Allocates six KV slots"]
    R --> T["Publish address table and device map"]
    T --> W["Workers load table and connect<br/>Report WORKER_READY"]
    W --> P["Migration driver sends GPU-capture tokens<br/>TT prefills slot 0"]
    P --> A["Driver drains device layer acknowledgments"]
    A --> M["Driver requests migration<br/>slot 0 → slot 5"]

    M --> X["Sender reads slot 0 DRAM<br/>using migration-table addresses"]
    X --> N["MPI transport<br/>between local workers"]
    N --> Y["Receiver writes slot 5 DRAM<br/>using migration-table addresses"]
    Y --> K["Workers report migration complete"]

    K --> B["Driver reads source and destination via UMD<br/>Checks every applicable block is byte-identical"]
    B --> S["Driver sends shutdown sentinel"]
    S --> H["Runner's test hook, before mesh closes:<br/>Full-prefix GPU PCC for slot 0<br/>plus sampled table-address checks"]
    G["GPU KV safetensors"] --> H
    H --> D["Pass checks, then close mesh"]
```

## What each check proves

| Check | Coverage |
| --- | --- |
| GPU PCC | All applicable heads and all 60 layers over the complete requested prefix in slot 0; threshold 0.91 |
| Address-table samples | Each head/layer and each CP rank's first and last populated 32-token block; table-based UMD reads must exactly match TTNN-gathered values |
| Loopback byte equality | Every applicable 32-token block over the requested prefix; slot 5 must exactly match slot 0 |

TTNN readback slices the populated prefix, untilizes a temporary copy to BF16 row-major, reads through the owning mesh command queue, and gathers/reorders host shards with PyTorch. The live BFP8 caches remain unchanged. UMD reads use the physical addresses described by the exported migration table and device map.

See [test commands and setup](PREFILL_MIGRATION.md) and [PCC performance](PCC_PERFORMANCE.md).
