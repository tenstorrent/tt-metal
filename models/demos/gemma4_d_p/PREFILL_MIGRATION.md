# Gemma4 prefill migration tests

These tests implement the two gates in the [shared migration guide](../common/prefill/docs/PREFILL_MIGRATION_TESTING.md). Both use the production Gemma4-31B-it configuration: one Blackhole 8×4 mesh, 60 layers, batch 1, 8192-token chunks, and six slots with 256K capacity.

| Gate | Driver | Checks |
| --- | --- | --- |
| 1: `mock` | Shared `prefill_producer` | Six distinct prompts; source KV read through the exported table and device map, compared with independent HF goldens |
| 2: `loopback` | Shared `migration_driver` | Three distinct source prompts; source golden PCC, real migration `0→5, 1→3, 2→4`, destination byte equality and golden PCC |

All 60 layers and all applicable heads are checked. The 36 table configurations describe four global packed heads, sixteen sliding K heads, and sixteen sliding V heads. Unused layer/config combinations are excluded from the driver's verification plan. Missing chunks, missing local devices, malformed goldens, and nonfinite KV fail verification.

The tests launch the runner and driver in separate processes. The runner publishes the table and map; numerical validation runs in the driver. Neither test adds validation to the serving loop. Logs and the exported table/map are retained in pytest's temporary directory.

## Setup and golden traces

Use the environment setup in [Prefill service](PREFILL_SERVICE.md). Generate six independent reference traces from real text. The exporter runs the HF model on CPU, captures post-RoPE K and normalized V before sliding-window eviction, and writes the shared producer's `metadata.json` and `kv_cache/layer_N.safetensors` format. It does not use TT hardware.

The full 31B checkpoint and reference KV require substantial host RAM and disk space. Run this preparation separately from hardware testing; it is not part of test collection.

```bash
export GEMMA4_GOLDEN_ROOT=/mnt/models/huggingface/gemma4_migration_goldens
mkdir -p /tmp/gemma4_prefill_text
for book in 135 2600 1184 996 1023 1399; do
    curl -fL "https://www.gutenberg.org/cache/epub/$book/pg$book.txt" \
        -o "/tmp/gemma4_prefill_text/pg$book.txt"
    python -m models.demos.gemma4_d_p.scripts.generate_golden_kv_cache \
        --text "/tmp/gemma4_prefill_text/pg$book.txt" \
        --tokens 16384 --out "$GEMMA4_GOLDEN_ROOT/$book"
done
export GEMMA4_MIGRATION_TRACES="$GEMMA4_GOLDEN_ROOT/135,$GEMMA4_GOLDEN_ROOT/2600,$GEMMA4_GOLDEN_ROOT/1184,$GEMMA4_GOLDEN_ROOT/996,$GEMMA4_GOLDEN_ROOT/1023,$GEMMA4_GOLDEN_ROOT/1399"
```

Output directories must be new. The default 16384-token traces exercise two prefill chunks across every CP row. This verifies the populated prefix, not every byte of the 256K capacity. Use `--tokens 262144` to check full-context migration. Trace lengths must be at least 8192, at most 262144, and divisible by 32 so the byte verifier covers the complete populated range. Distinct prompts are required to detect crossed slots.

## Gate 1: mock migration

No migration endpoint or tt-llm-engine is required. The test enables `PREFILL_MOCK_MIGRATION=1`, exports the table and device map, and runs shared producer golden PCC with threshold 0.93.

```bash
pytest 'models/demos/gemma4_d_p/tests/test_prefill_migration.py::test_prefill_migration[mock]' \
    -sv --basetemp=/tmp/gemma4-migration-mock
```

## Gate 2: loopback migration

Build/provision the tt-llm-engine migration endpoint and workers against this tt-metal checkout as described in the shared guide. Start the endpoint on the same host before running pytest:

```bash
# In the tt-llm-engine checkout:
cd disaggregation/migration
./launch_migration_endpoints.sh --name_server_host "$(hostname)" \
    --prefill_hosts "$(hostname)" --prefill_endpoint_id 1
```

In the tt-metal terminal, set the client directory to the one containing `_migration_client*.so`:

```bash
export PREFILL_MIGRATION_CLIENT_DIR=/path/to/tt-llm-engine/disaggregation/migration/build_RelWithDebInfo/python
export GEMMA4_TEST_LOOPBACK=1
pytest 'models/demos/gemma4_d_p/tests/test_prefill_migration.py::test_prefill_migration[loopback]' \
    -sv --basetemp=/tmp/gemma4-migration-loopback
```

The endpoint's default queues are `/mig_ep1_cmd`, `/mig_ep1_table`, and `/mig_ep1_resp`. The corresponding `PREFILL_MIGRATION_*_QUEUE` variables can override them. The test enables real migration, requires the worker-ready handshake, and invokes the shared driver with `--verify-migration both`. Source slots 0–2 and destination slots 3–5 are disjoint. The driver shuts the runner down only after verification. The test does not start or stop the external endpoint.

Gate 1 skips unless `GEMMA4_MIGRATION_TRACES` is set. Gate 2 additionally requires `GEMMA4_TEST_LOOPBACK=1`. These gates are single-host loopback tests; they do not validate a decode endpoint's layout.

## Host-only checks

```bash
pytest models/demos/gemma4_d_p/tests/unit/test_prefill_migration.py -q
```

These checks use host table objects and mocked DRAM reads. They cover cache-stage descriptors, protobuf round trips, all 36 config mappings, detection of corrupted destination bytes, packed global/sliding golden decoding, and reference capture before sliding-window eviction. They do not prove real transport or hardware accuracy.
