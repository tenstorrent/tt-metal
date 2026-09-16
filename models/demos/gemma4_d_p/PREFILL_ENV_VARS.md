# Gemma4 prefill environment variables

This reference covers the Gemma4 service, its producer and model, the shared runner and migration helpers, and the documented launch setup. See [Prefill service](PREFILL_SERVICE.md) for commands.

## Defaults and overrides

Defaults below apply when launching `python -m models.demos.gemma4_d_p.tt.runners.prefill_runner`, which loads [manifest.json](tt/runners/manifest.json).

- Exported environment variables override manifest values. The manifest fills only unset variables; remaining defaults come from the adapter and shared runner.
- Set variables before starting Python. The runner resolves most settings during import.
- **Fixed** means the variable is read from the environment, but the supported configuration requires the listed value. An incompatible override is rejected.
- **Unused** means it has no effect on this model's serving path. It does **not** need to be exported.
- Required fixed values are supplied by defaults; they do not need manual exports either.
- Use `0` and `1` for `PREFILL_*` Boolean flags.

For example, the shared runner reads `PREFILL_MAX_SEQ_LEN` with a fallback of `CHUNK_SIZE * 11`. The Gemma4 manifest supplies `262144` before that read. Exporting `PREFILL_MAX_SEQ_LEN=131072` overrides the manifest, but Gemma4 validation rejects it.

## Model, shape, and weight-cache settings

| Environment variable | Default for the Gemma4 service | Use and override behavior |
|---|---|---|
| `PREFILL_MANIFEST` | `models/demos/gemma4_d_p/tt/runners/manifest.json` | Runner configuration file. The Gemma4 producer does not read it. |
| `PREFILL_MODEL` | `gemma4_d_p` | Selects the adapter. Changing it selects another model rather than reconfiguring Gemma4. |
| `PREFILL_HF_MODEL` | Falls back to `HF_MODEL`, then `google/gemma-4-31B-it` | Checkpoint ID or local directory. Configuration must match Gemma4-31B-it. Read by runner and producer. |
| `PREFILL_TTNN_CACHE` | Empty string | Optional cache-root override. Empty/unset falls back to `TT_CACHE_PATH`, then the model's cache-path resolver. Runner only. |
| `PREFILL_SP` | `8` | **Fixed:** eight context-parallel rows. |
| `PREFILL_TP` | `4` | **Fixed:** four tensor-parallel columns. |
| `PREFILL_NUM_LAYERS` | `60` | **Fixed:** all 60 layers on one pipeline rank. |
| `PREFILL_PP_LAYER_COUNTS` | Unset; even split resolves to one rank with 60 layers | May explicitly be `60`. Multiple pipeline ranks are unsupported. |
| `PREFILL_CHUNK_SIZE` | `8192` | **Fixed:** physical tokens per chunk, including padding. |
| `PREFILL_MAX_SEQ_LEN` | `262144` | **Fixed:** capacity per KV slot. Shorter prompts use the same capacity. |
| `PREFILL_NUM_USERS` | `6` | Configurable from `1` to `6`. Allocates that many KV slots; it does not change batch size. Also supplies the producer's default `--slots`. |
| `PREFILL_USE_TRACE` | `1` | **Fixed:** the service requires trace replay. |
| `PREFILL_KV_ONLY_LAST_LAYER` | `1` | **Required by the shared runner:** `0` is rejected with tracing. Gemma4 does not consume this field or expose a logits head. |
| `PREFILL_TP_SHARD_KV` | `0` | **Fixed:** the shared runner rejects enabling this optional cache mode for Gemma4. |

Weight-cache root precedence is `PREFILL_TTNN_CACHE` → `TT_CACHE_PATH` → local checkpoint directory, or `$HF_HOME/tt_cache/<model ID with / replaced by -->` for a repository ID. For that last fallback, the model uses `~/.cache/huggingface` when `HF_HOME` is unset. The resolved tensor-cache directory is `<root>/tensor_cache_bf16_mesh8x4`.

The shared startup dump prints `PREFILL_HF_MODEL` or the adapter default; it does not account for the `HF_MODEL` fallback. The actual adapter and producer do use that fallback. An empty printed `PREFILL_TTNN_CACHE` likewise does not mean that weight caching is disabled.

## Service transport, tracing, and timing

| Environment variable | Default for the Gemma4 service | Use and override behavior |
|---|---|---|
| `PREFILL_H2D_SERVICE_ID` | `gemma4_prefill` | Input-service identity and acknowledgment-channel suffix. Runner and producer must agree. |
| `PREFILL_FABRIC_MODE` | `1d` | Fabric configuration at mesh open. Shared runner also parses `2d`, `1d_ring`, `2d_torus_x`, `2d_torus_y`, and `2d_torus_xy`; Gemma4 service validation was run with `1d`. |
| `PREFILL_TRACE_REGION_SIZE` | `268435456` bytes (256 MiB), from the manifest | Device memory reserved for the trace. Must fit the captured graph. |
| `PREFILL_LAYER_ACK_D2H` | `1` | `1` emits device records after each layer. `0` emits host callbacks for all layers after the complete chunk has synchronized. Hardware service validation used `1`. |
| `PREFILL_LAYER_ACK_FIFO_BYTES` | `4096` bytes | D2H acknowledgment FIFO capacity. Used when `PREFILL_LAYER_ACK_D2H=1`. |
| `PREFILL_LAYER_COMPLETION_RING` | `/tt_prefill_layer_completion_ring` | Shared-memory ring-name base. The runner appends `_0` for its single rank. |
| `PREFILL_MASTER_RANK` | `0` | Must remain `0` for the supported single-rank service. Owns the producer-facing acknowledgment channel. |
| `PREFILL_LAYER_COMPLETION_PUSH_TIMEOUT_S` | `30.0` seconds | Host callback queue backpressure timeout. Used only when `PREFILL_LAYER_ACK_D2H=0`. |
| `PREFILL_SYNC_PER_CHUNK` | `0` | Enables shared-runner synchronization and compute timing. The Gemma4 runtime already synchronizes each chunk regardless of this flag. |
| `PREFILL_TIMING_DIR` | Empty string | With `PREFILL_SYNC_PER_CHUNK=1`, append timings to `<directory>/rank0.csv`. The directory must already exist. |

## Shared settings unused by Gemma4

None of these requires an export.

| Environment variable | Default | Why it is unused |
|---|---|---|
| `PREFILL_CAPACITY_FACTOR` | `8` | Passed in shared run parameters, but the dense Gemma4 runtime does not use MoE capacity settings. |
| `PREFILL_GATE_FALLBACK_MODE` | `DEVICE_FP32` | Passed as the gate-mode name, but Gemma4 has no MoE gate. |
| `PREFILL_OVERLAP_SHARED_EXPERT` | `1` | Passed in shared run parameters; Gemma4 has no shared expert. |
| `PREFILL_PP_D2D_FIFO_BYTES` | `256` bytes | Only used for activation sockets between pipeline ranks. Gemma4 uses one rank. |
| `PREFILL_TRACE_DIR` | Empty string | Printed by the shared runner; unused by the Gemma4 runtime and producer. It does not select the device trace or producer text. |
| `PREFILL_DFLASH` | `0` | Cannot enable DFlash: the adapter's `supports_dflash` is false. |
| `DFLASH_HF_MODEL` | Unset | DFlash checkpoint setting; Gemma4 has no supported drafter. |
| `PREFILL_ALLOW_UNTESTED_TP_SHARD_TRACE` | `0` | Cannot enable the unsupported TP cache mode; the adapter capability check rejects it first. |

## Migration settings

The runtime exposes KV-table construction for mock export. Full migration is not wired: `PREFILL_ENABLE_MIGRATION=1` fails because the runtime lacks `kv_migration_stages`/`kv_migration_base_address`. Mock export was not exercised by the six-slot service test.

| Environment variable | Default | Use and override behavior |
|---|---|---|
| `PREFILL_ENABLE_MIGRATION` | `0` | Keep `0`; the full worker-integration path is unsupported. |
| `PREFILL_MOCK_MIGRATION` | `0` | With full migration disabled, `1` builds and exports the KV table and device map without a migration worker. |
| `PREFILL_MIGRATION_TABLE_PATH` | `/tmp/prefill_kv_chunk_table_gemma4_prefill.pb` | Mock-export destination. The default filename follows `PREFILL_H2D_SERVICE_ID`. |
| `PREFILL_MIGRATION_DEVICE_MAP_PATH` | `/tmp/prefill_kv_device_map.json` for mock export | Mock-export device-map destination. The shared full-migration file-export path instead defaults to `/tmp/prefill_device_map.txt`, but that path is unsupported here. |
| `PREFILL_MIGRATION_WAIT_READY_MS` | `120000` milliseconds | Unused by mock export. Full-migration worker-ready timeout. |
| `PREFILL_MIGRATION_EXPORT_TO_FILE` | `0` | Full-migration option; does not bypass Gemma4's missing migration hooks. Not required for mock export. |
| `PREFILL_MIGRATION_CMD_QUEUE` | `/prefill_mig_cmd_1` | Unused here; full-migration command queue. |
| `PREFILL_MIGRATION_TABLE_QUEUE` | `/prefill_mig_tbl_1` | Unused here; full-migration table queue. |
| `PREFILL_MIGRATION_RESP_QUEUE` | `/prefill_mig_rsp_1` | Unused here; full-migration response queue. |
| `PREFILL_MIGRATION_CLIENT_DIR` | Unset | Unused here; full-migration Python extension search directory. |
| `PREFILL_MIGRATION_ATTACH_WAIT_S` | `0` seconds, meaning unlimited wait | Unused here; full-migration client attachment timeout. |

## Launch environment and checkpoint fallbacks

These variables support imports, kernel resources, and checkpoint/cache discovery. They are not unused model switches. Machine-specific paths in the launch recipe are explicit settings, not built-in defaults.

| Environment variable | Default when not exported | Setting in the launch recipe / purpose |
|---|---|---|
| `PYTHONPATH` | No value supplied by this package | Set to the repository root for repository imports. |
| `TT_METAL_HOME` | No value supplied by this package | Set to the repository root for Metal runtime/kernel resources. |
| `HF_MODEL` | Unset; the service adapter falls back to `google/gemma-4-31B-it` | Recipe sets `google/gemma-4-31B-it`. Used by runner and producer unless `PREFILL_HF_MODEL` overrides it. The canonical demo reads `HF_MODEL` directly. |
| `HF_HOME` | No value supplied by this package; model cache resolver falls back to `~/.cache/huggingface` | Recipe sets `/mnt/models/huggingface`. Used for Hugging Face storage and the fallback TT cache root. |
| `TT_CACHE_PATH` | Unset | Recipe sets `/mnt/models/huggingface/tt_cache/gemma4_d_p/google--gemma-4-31B-it`. Used unless `PREFILL_TTNN_CACHE` supplies the root. |
| `HF_HUB_OFFLINE` | No value supplied by this package | Recipe sets `1` to load Hugging Face artifacts offline. Does not prevent the producer from downloading Gutenberg text. |

## Model operation settings

These existing model controls are read by the service's model implementation. The six-slot service test used their defaults.

| Environment variable | Default | Use and override behavior |
|---|---|---|
| `GEMMA4_CCL_ASYNC` | `0` | Selects synchronous collectives. `1` selects asynchronous reduce-scatter/all-gather operations. |
| `GEMMA4_CCL_TOPOLOGY` | Empty string, resolving to Ring | Async collective topology. `linear`, `line`, or `l` selects Linear; other values select Ring. Sync collectives do not consume this setting. |
| `GEMMA4_CCL_CHUNKS_PER_SYNC` | `10` | Async collective packet grouping; clamped to at least `1`. Unused by sync collectives. |
| `GEMMA4_CCL_NUM_WORKERS` | `2` | Async workers per link; clamped to at least `1`. Unused by sync collectives. |
| `GEMMA4_CCL_NUM_BUFFERS` | `2` | Async buffers per channel; clamped to at least `1`. Unused by sync collectives. |
| `GEMMA4_CCL_PERSISTENT_BUF` | `1` | Reuses async reduce-scatter buffers. Unused by sync collectives. |
| `GEMMA4_PREFILL_L1_ACT` | `0` | Uses DRAM for short-lived attention activations; `1` selects L1. |

## Demo-only environment variables

These are read by `demo/text_demo_prefill.py`, not by the service runtime. Setting them does not override the corresponding `PREFILL_*` service settings.

| Environment variable | Demo default | Service behavior |
|---|---|---|
| `GEMMA4_MAX_SEQ_LEN` | Test context length; `262144` in the canonical run | Unused. The service uses `PREFILL_MAX_SEQ_LEN`. |
| `GEMMA4_PREFILL_TRACE_REGION_SIZE` | `256000000` bytes | Unused. The service uses `PREFILL_TRACE_REGION_SIZE`. |
| `GEMMA4_PREFILL_LOAD_FULL_WEIGHTS` | `0` | Unused. The service does not read this demo flag; its model builder uses the normal cache-completion check. |

## Producer arguments

The Gemma4 producer reads `PREFILL_HF_MODEL`/`HF_MODEL` for its tokenizer, `PREFILL_H2D_SERVICE_ID` for the default service ID, and `PREFILL_NUM_USERS` for the default slot count. It does not load the runner manifest. Hugging Face setup variables also apply to tokenizer loading. Its remaining controls are CLI arguments, not the common producer's `PREFILL_PRODUCER_*` variables.

| CLI argument | Default | Behavior |
|---|---|---|
| `--service-id` | `PREFILL_H2D_SERVICE_ID`, else `gemma4_prefill` | Must match the runner. CLI overrides the environment default. |
| `--slots` | `PREFILL_NUM_USERS`, else `6` | Sends to slots `0` through `slots - 1`; must fit the runner's allocated slots. Accepts `1`–`6`. |
| `--tokens` | `262144` | Actual prompt length per slot, from `1` to `262144`. Does not resize the runner's KV cache. |
| `--text` | Six Gutenberg books, IDs `135`, `2600`, `1184`, `996`, `1023`, `1399` | Repeat once per slot to use local UTF-8 files. |
| `--text-cache` | `/tmp/gemma4_prefill_text` | Downloaded-text cache directory. |
| `--timeout` | `1200` seconds | Connection and per-chunk acknowledgment timeout. `PREFILL_H2D_CONNECT_TIMEOUT` is not read by this producer. |
| `--keep-serving` | Off | Omit the shutdown sentinel after requests finish. |
| `--results` | Unset | Optional JSON file for per-slot token counts and chunk timings. |

## Derived values and fixed Python settings

These are not environment variables and cannot be overridden by exporting their displayed names.

| Variable / displayed value | Value | Source or meaning |
|---|---|---|
| `resolved weight_cache_path` | `<resolved root>/tensor_cache_bf16_mesh8x4` | Derived from the cache-root precedence above. |
| `DFLASH_ENABLED` | `False` | Derived from adapter capability and DFlash environment settings. Adapter capability is false. |
| `GLOBAL_MESH_SHAPE` | `(8, 4)` | Derived from `PREFILL_SP` and `PREFILL_TP`; validated by the adapter. |
| `max_batch_size` | `1` | Fixed in runtime model construction; independent of the number of KV slots. |
| `num_links` | `2` on Blackhole | Selected by model collective setup. No Gemma4 service environment override. |
| `sp_axis`, `tp_axis` | `0`, `1` | Fixed mesh axes. |
| `first_layer_idx`, `is_first_rank`, `is_last_rank` | `0`, `True`, `True` | Required single-rank layer ownership. |
| `FABRIC_PAYLOAD_SIZE` | `8192` bytes | Adapter's router packet payload setting. |
| `l1_small_size` | `0` | Adapter's small-L1 reservation. |
| `METADATA_SIZE_BYTES` | `12` bytes | Three 32-bit words: slot ID, actual start, and actual end. |
| `SYNC_WORKER_CORES` | Core `(0, 0)` | Shared runner's socket service worker core. |

## Sources

- [Gemma4 manifest](tt/runners/manifest.json), [adapter and validation](tt/runners/adapter.py), [runtime](tt/runners/runtime.py), and [producer](tt/runners/prefill_producer.py).
- [Shared runner](../common/prefill/runners/prefill_runner.py), [mesh and layer-split utilities](../common/prefill/runners/runner_utils.py), and [migration helpers](../common/prefill/runners/migration.py).
- [Model cache resolution](tt/model_config.py), [collective operations](tt/ccl.py), [attention operation settings](tt/attention/operations.py), and [canonical demo](demo/text_demo_prefill.py).
