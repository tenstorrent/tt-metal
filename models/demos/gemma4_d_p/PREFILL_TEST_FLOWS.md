# Gemma4 prefill test flows

## Run the tests

Both tests require a Blackhole 8×4 mesh, a [tt-metal source build](../../../INSTALLING.md#source) with its Python environment, the Gemma4-31B-it checkpoint, and the [GPU capture](PREFILL_MIGRATION.md#gpu-reference). The checkpoint and capture are separate from the repository; provision them before running. The commands below use the canonical model paths.

From the tt-metal repository root, in both terminals for loopback:

```bash
source python_env/bin/activate
export PYTHONPATH="$PWD" TT_METAL_HOME="$PWD" OMP_NUM_THREADS=16
export \
    HF_MODEL=google/gemma-4-31B-it \
    HF_HOME=/mnt/models/huggingface \
    TT_CACHE_PATH=/mnt/models/huggingface/tt_cache/gemma4_d_p/google--gemma-4-31B-it \
    HF_HUB_OFFLINE=1
export PREFILL_TTNN_CACHE="$TT_CACHE_PATH"
```

The default GPU reference is already in validation channel order. See [Prepared GPU reference](PREFILL_MIGRATION.md#prepared-gpu-reference) to create another copy.

### Mock, 16K

The test starts the runner and producer automatically. It does not require tt-llm-engine.

```bash
pytest 'models/demos/gemma4_d_p/tests/test_prefill_migration.py::test_prefill_migration[mock-16k]' -sv
```

### Loopback, 16K

The test starts the model runner and migration driver. Build the external migration dependency, then start its endpoint separately and keep it running until the test finishes.

**Loopback environment — both terminals:**

Use OpenMPI 5 with ULFM and PRRTE, provided by tt-metal's [dependency installer](../../../install_dependencies.sh). These commands use its default installation path. The tt-llm-engine checkout lives alongside tt-metal; set `TT_LLM_ENGINE_DIR` to another location if needed.

```bash
export TT_LLM_ENGINE_DIR="$TT_METAL_HOME/../tt-llm-engine"
export MIGRATION_BUILD_DIR="$TT_LLM_ENGINE_DIR/disaggregation/migration/build_RelWithDebInfo"
export PREFILL_MIGRATION_CLIENT_DIR="$MIGRATION_BUILD_DIR/python"
export MPI_HOME=/opt/openmpi-v5.0.7-ulfm
export PATH="$MPI_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$MPI_HOME/lib:$TT_METAL_HOME/build/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

**One-time setup — clone and build tt-llm-engine:**

This requires GitHub access to `tenstorrent/tt-llm-engine`, the tt-metal build toolchain, and Protobuf and libibverbs development packages. On Ubuntu, from the tt-metal root with `python_env` active:

```bash
sudo apt-get install protobuf-compiler libprotobuf-dev libibverbs-dev
git clone git@github.com:tenstorrent/tt-llm-engine.git "$TT_LLM_ENGINE_DIR"
(
    cd "$TT_LLM_ENGINE_DIR"
    export TT_METAL_DIR="$TT_METAL_HOME" TT_METAL_BUILD_DIR="$TT_METAL_HOME/build"
    export CC=clang-20 CXX=clang++-20
    ./build_migration_layer.sh --build-type=RelWithDebInfo --no-device-tests \
        --targets='migration_endpoint migration_worker _migration_client' --jobs=8
)
```

The build uses this tt-metal checkout and its existing libraries. `--no-device-tests` skips the dependency's own tests; the migration worker still supports real hardware. It produces the endpoint and worker under `$MIGRATION_BUILD_DIR/bin` and the Python client under `$MIGRATION_BUILD_DIR/python`.

**Terminal 1 — start the endpoint:**

Set `MIGRATION_NETWORK_INTERFACE` to this host's network interface; `eth0` below is an example. This single-host test uses TCP transport.

```bash
export MIGRATION_NETWORK_INTERFACE=eth0
export OMPI_MCA_pml=ob1 OMPI_MCA_btl=self,tcp
export OMPI_MCA_btl_tcp_if_include="$MIGRATION_NETWORK_INTERFACE"
export PRTE_MCA_oob_tcp_if_include="$MIGRATION_NETWORK_INTERFACE"
export PMIX_MCA_ptl_tcp_if_include="$MIGRATION_NETWORK_INTERFACE"
export MIGRATION_DEVICE_BACKEND=umd

"$MIGRATION_BUILD_DIR/bin/migration_endpoint" \
    --endpoint-id 1 \
    --cmd-queue /mig_ep1_cmd --table-queue /mig_ep1_table --response-queue /mig_ep1_resp \
    --worker-bin "$MIGRATION_BUILD_DIR/bin/migration_worker" \
    --worker-hosts "$(hostname)" 2>&1 | tee /tmp/gemma4-loopback-endpoint.log
```

**Terminal 2 — wait for endpoint readiness, then run the test:**

```bash
python "$MIGRATION_BUILD_DIR/../_migration_endpoint_driver.py" \
    --client-dir "$PREFILL_MIGRATION_CLIENT_DIR" \
    --cmd-queue /mig_ep1_cmd --table-queue /mig_ep1_table --resp-queue /mig_ep1_resp \
    --mode ready --alive-timeout 60 && \
GEMMA4_TEST_LOOPBACK=1 pytest \
    'models/demos/gemma4_d_p/tests/test_prefill_migration.py::test_prefill_migration[loopback-16k]' -sv
```

The readiness command prints `ALIVE` before pytest starts. The runner subsequently publishes its table and device map, then waits for `WORKER_READY`. After the test finishes, stop the endpoint with Ctrl-C in terminal 1.

Replace `16k` with `8k`, `128k`, or `256k` to select another context. Both tests start and stop their model runner; loopback leaves the external migration endpoint running.

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
