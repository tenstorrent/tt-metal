# Gemma4 prefill GPU-trace and migration tests

The tests use Gemma4-31B-it on one Blackhole 8×4 mesh: 60 layers, batch 1, 8192-token prefill chunks, and six allocated KV slots with 256K capacity. Each test sends one request to slot 0 and compares it once against the GPU trace.

| Context | Tokens | Prefill chunks |
| --- | ---: | ---: |
| `8k` | 8192 | 1 |
| `16k` | 16384 | 2 |
| `128k` | 131072 | 16 |
| `256k` | 262144 | 32 |

## GPU reference

Use the environment setup in [Prefill service](PREFILL_SERVICE.md). The adapter defaults to this staged HF/SDPA GPU capture:

```text
/mnt/models/huggingface/gpu_traces/gemma4_d_p/hf-gemma4-31b-36db66e9-262144tok
```

Set `PREFILL_TRACE_DIR` to override it. The capture contains one Gutenberg *Les Misérables* prompt, its exact token IDs in `metadata.json`, and all 60 layers of KV. The producer replays the requested token prefix without retokenizing or repeating it.

The `chunked_group_a_v1` reference stores `kv_cache/layer_N/rows_START_END.safetensors`, with key `kv_post_transform_layer_N`. Each row contains K followed by V, flattened in head order. Sliding rows have width 8192 (16 heads × 256 channels × K/V); global rows have width 4096 (4 heads × 512 channels × K/V). K is captured after normalization and RoPE; V after normalization, before sliding-window eviction.

The validator reconstructs Gemma4's packed global KV and sliding K channel order. The hardware test slices the selected slot and populated prefix on the owning mesh, converts the temporary copy to BF16 row-major on-device, reads it through the TTNN command queue, joins the host shards with PyTorch, and restores chunk-major CP token order. The live BFP8 cache is unchanged. It also checks the first and last populated 32-token block on each CP rank against the exported migration table for every head and layer. These address checks are samples; the GPU comparison covers the full requested prefix. It compares all applicable heads in all 60 layers and reports separate minima for global rotary K, global V, sliding K, and sliding V. Each PCC flattens one head over the entire requested context. Prefill runs chunk-first; validation starts after all chunks finish and runs layer-first. Missing rows, invalid addresses, malformed tensors, and nonfinite values fail.

## PCC definition and threshold

Each score is Pearson correlation between a TT cache head and its GPU counterpart, flattened over the entire requested token prefix and the compared channels. Global rotary K and V are scored separately. There are 1680 scores per context.

- `layer_minima`: the lowest head score for each cache type in the current layer. It can increase between layers.
- `running_min_pcc`: the lowest score seen across all layers checked so far. It cannot increase.
- Final minimum: the lowest of all 1680 scores. Each of the four reported cache minima also spans every applicable layer and head.

For example, layer minima of `0.98, 0.94, 0.96` produce running minima of `0.98, 0.94, 0.94`. PCC is correlation, not the percentage of matching values.

The regression threshold is **0.91**, calibrated on this capture and the current model precision. The 8K, 16K, and 128K hardware tests passed; all 1680 scores per context matched the original full UMD readback exactly:

| Context | Minimum PCC | Global rotary K | Global V | Sliding K | Sliding V |
| --- | ---: | ---: | ---: | ---: | ---: |
| 8K | 0.920747 | 0.951169 | 0.948299 | 0.945258 | 0.920747 |
| 16K | 0.929830 | 0.954082 | 0.952419 | 0.950691 | 0.929830 |
| 128K | 0.934371 | 0.950708 | 0.954783 | 0.948936 | 0.934371 |
| 256K | Not calibrated | — | — | — | — |

The lowest measured score is sliding V, layer 39, head 11 at 8K. The threshold leaves about 0.0107 absolute PCC below that value; the initial 0.93 candidate rejected the 8K and 16K baselines. This is a regression floor for one captured prompt, not a general model-quality guarantee. The 256K case remains available but its calibration run was canceled. Longer prefixes produce different correlation statistics, so their minima need not decrease.

## Gate 1: GPU-trace comparison

The test starts the service with `PREFILL_MOCK_MIGRATION=1`, which exports the address table and device map without requiring a migration endpoint. The shared producer sends tokens followed by a shutdown sentinel. The runner synchronizes each chunk on the device before processing the next message. The test intercepts the end of the service request loop to read and compare KV before the mesh closes. The service keeps KV resident throughout validation. TT KV is read directly into host memory; only address metadata, logs, and PCC reports are written to disk.

```bash
# All four contexts:
pytest models/demos/gemma4_d_p/tests/test_prefill_migration.py \
    -k mock -sv --basetemp=/tmp/gemma4-gpu-test

# One context:
pytest 'models/demos/gemma4_d_p/tests/test_prefill_migration.py::test_prefill_migration[mock-128k]' -sv
```

The test uses `GPU_PCC_THRESHOLD` in `tests/test_prefill_migration.py`. Each case starts a fresh runner process and closes its mesh after validation. It retains `producer.log`, the table, the device map, and `gemma4_slot0.json` with every layer/head PCC and comparison timings. Runner output is saved in `runner.log`. The test sets `PREFILL_PRODUCER_CHECK_PCC=0` because the owning process performs the GPU comparison; the runner synchronizes device completion before leaving its request loop.

To use an already-running service, set matching service/table/map paths and invoke the shared producer. This external-process path reads the entire prefix through the migration table using slower UMD MMIO reads:

```bash
export PREFILL_MODEL=gemma4_d_p
export PREFILL_SP=8 PREFILL_TP=4 PREFILL_NUM_LAYERS=60
export PREFILL_MAX_SEQ_LEN=262144 PREFILL_CHUNK_SIZE=8192
export PREFILL_NUM_USERS=1 PREFILL_PRODUCER_MAX_REQUESTS=1
export PREFILL_PRODUCER_CHECK_PCC=1
export PREFILL_PRODUCER_CHUNKS=16  # 128K; use 1, 2, 16, or 32
export PREFILL_PCC_SUMMARY_DIR=/tmp/gemma4-gpu-pcc
export PREFILL_STANDALONE_CHUNKED_PCC=0.91
python -m models.demos.common.prefill.runners.prefill_producer
```

The producer defaults to keeping the service alive. Set `PREFILL_SEND_SHUTDOWN=1` to stop it after comparison. Do not set `PREFILL_PRODUCER_SLOT_TRACES` or `PREFILL_PCC_GOLDEN_LEN` for these prefix comparisons: the former selects the entire prompt length, and the latter caps verification.

## Gate 2: loopback migration

Provision the tt-llm-engine migration endpoint and workers against this checkout as described in the [shared migration guide](../common/prefill/docs/PREFILL_MIGRATION_TESTING.md), then start the endpoint:

```bash
# In the tt-llm-engine checkout:
cd disaggregation/migration
./launch_migration_endpoints.sh --name_server_host "$(hostname)" \
    --prefill_hosts "$(hostname)" --prefill_endpoint_id 1
```

In the tt-metal terminal:

```bash
export PREFILL_MIGRATION_CLIENT_DIR=/path/to/tt-llm-engine/disaggregation/migration/build_RelWithDebInfo/python
export GEMMA4_TEST_LOOPBACK=1
pytest models/demos/gemma4_d_p/tests/test_prefill_migration.py \
    -k loopback -sv --basetemp=/tmp/gemma4-migration-loopback
```

The shared migration driver migrates `0→5` and checks destination bytes against source bytes using `--verify-migration dst-bytes`. The owning test then compares source slot 0 against the GPU trace once through TTNN. It requires the worker-ready handshake. The test does not start or stop the external endpoint. Default queues are `/mig_ep1_cmd`, `/mig_ep1_table`, and `/mig_ep1_resp`; `PREFILL_MIGRATION_*_QUEUE` can override them.

This covers one prompt and one source slot at four context lengths. It does not validate distinct prompts across all six slots or a decode endpoint's layout. Loopback cases skip unless `GEMMA4_TEST_LOOPBACK=1`; both gates skip when the GPU trace is absent.

For measured data volumes, readback throughput, and comparison costs, see [PCC performance](PCC_PERFORMANCE.md).

## Host-only checks

```bash
pytest models/demos/gemma4_d_p/tests/unit/test_prefill_migration.py -q
```

These cover cache-stage descriptors, protobuf round trips, all 36 configurations, corrupt destination detection, GPU shard decoding and coverage, batched BFP8 readback order, CP chunk-order restoration, and nonfinite PCC rejection.
