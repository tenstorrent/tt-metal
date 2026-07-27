# Prefill KV-migration — manifest-driven runs

The same gates as [PREFILL_MIGRATION_TESTING.md](PREFILL_MIGRATION_TESTING.md), driven by YAML manifests
and `prefill_producer` instead of long `env` lines and the C++ `prefill_scheduler_driver`.

Assumes the model is already integrated — adapter registered, golden trace staged, weight cache populated
(see [ADDING_A_PREFILL_MODEL.md](ADDING_A_PREFILL_MODEL.md)).


| Gate                       | What it exercises                                             | Needs                    |
| -------------------------- | ------------------------------------------------------------- | ------------------------ |
| **0 — standalone PCC**     | prefill writes correct KV (precondition for everything)       | tt-metal tree only       |
| **1 — mock migration**     | the KV-chunk **address table** is correct, read device-lessly | tt-metal tree only       |
| **2 — loopback migration** | the real DRAM→transport→DRAM copy + migrated-KV accuracy      | + tt-llm-engine binaries |


---



## 0. The three config files


| File                        | Location                          | Owns                               |
| --------------------------- | --------------------------------- | ---------------------------------- |
| model manifest (`.json`)    | your model package                | `PREFILL_MODEL` + model-wide knobs |
| rank binding (`.yaml`)      | `runners/topology_configuration/` | mesh topology + runner env         |
| producer manifest (`.yaml`) | `runners/producer_manifests/`     | workload + migration scenario      |


Rank bindings and producer manifests are shared and model-agnostic; the only model-specific file is the
one in your own package. Swapping models means swapping the model manifest, not editing the other two.

Precedence differs by process. **Runner** (under `tt-run`): `global_env` > model manifest > code default —
a shell-exported `PREFILL_*` is *not* an override, since `tt-run` only forwards `TT_/ARCH_/WH_/TTNN_/ DEEPSEEK_/MESH_`. **Producer** (a normal process): CLI flag > exported env > manifest `env:` block >
manifest typed block.

### Shared setup (every terminal)

```bash
cd <tt-metal>
export TT_METAL_HOME="$PWD" PYTHONPATH="$PWD"
source python_env/bin/activate
export HOST=$(hostname)
export RUN=./models/demos/common/prefill/runners/run_pipeline_prefill.sh
export BINDING=models/demos/common/prefill/runners/topology_configuration/<binding>.yaml
export MANIFEST=models/demos/common/prefill/runners/producer_manifests/<manifest>.yaml
```

Constraints as before: `MAX_SEQ_LEN % CHUNK_SIZE == 0` and `CHUNK_SIZE % (SP*32) == 0`.

---



## 1. Model manifest

```json
// models/demos/my_model/tt/runners/manifests/my_model.json
{ "env": { "PREFILL_MODEL": "my_model", "PREFILL_GATE_FALLBACK_MODE": "DEVICE_FP32" } }
```



## 2. Rank binding

Copy `topology_configuration/pipeline_prefill_request_1rank.yaml` and add what migration needs:

```yaml
rank_bindings:
  - {rank: 0, mesh_id: 0, mesh_host_rank: 0, env_overrides: {}}
mesh_graph_desc_path: tt_metal/fabric/mesh_graph_descriptors/single_bh_galaxy_torus_x_graph_descriptor.textproto
global_env:
  PREFILL_MANIFEST: "models/demos/my_model/tt/runners/manifests/my_model.json"
  PREFILL_NUM_LAYERS: "<L>"        # the runner defaults to 61 — pin the real depth
  PREFILL_MAX_SEQ_LEN: "<S>"
  PREFILL_FABRIC_MODE: "2d"
  PREFILL_H2D_SERVICE_ID: "my_prefill"
  PREFILL_NUM_USERS: "4"           # must cover every dst slot: src 0,1 -> dst 2,3

  PREFILL_ENABLE_MIGRATION: "1"
  PREFILL_MIGRATION_WAIT_READY_MS: "120000"
  PREFILL_MIGRATION_TABLE_PATH: "/tmp/prefill_kv_chunk_table.pb"
  PREFILL_MIGRATION_CMD_QUEUE: "/mig_ep1_cmd"      # endpoint default is /mig_ep<id>_{cmd,table,resp}
  PREFILL_MIGRATION_TABLE_QUEUE: "/mig_ep1_table"
  PREFILL_MIGRATION_RESP_QUEUE: "/mig_ep1_resp"
  PREFILL_MIGRATION_CLIENT_DIR: "<tt-llm-engine>/disaggregation/migration/build_RelWithDebInfo/python"

  PREFILL_VALIDATE_MIGRATION: "1"
  PREFILL_MIGRATE_PAIRWISE: "1"                    # dst == src; omit for burst (vs golden)
  PREFILL_STANDALONE_CHUNKED_NCHUNKS: "22"         # = num_users x chunks the producer pushes
  MIGRATION_DONE_FILE: "/tmp/migration_done.sentinel"
  PREFILL_MIGRATE_WAIT_S: "1200"
```



## 3. Producer manifest

Start from `producer_manifests/prefill_producer_manifest.example.yaml`, which lists every field with its
env var and default. A migration scenario needs little:

```yaml
transport:
  sp: 8
  tp: 4
  h2d_service_id: my_prefill      # [MATCH RUNNER]
  connect_timeout_s: 60

workload:
  num_users: 2                    # src slots 0,1
  chunks: "11"
  max_requests: 2                 # one resident request per slot
  interleave: round_robin
  check_pcc: false                # the RUNNER validates migrated KV on-device

migration:
  issue: true                     # attach the client and migrate after prefill
  dest_endpoint_id: 1             # == our own id => loopback
  dst_slot_offset: 2              # dst = src + 2  (0->2, 1->3)
  timeout_ms: 3600000
  done_file: /tmp/migration_done.sentinel                # [MATCH RUNNER]
  table_path: /tmp/prefill_kv_chunk_table.pb             # [MATCH RUNNER]
  cmd_queue: /mig_ep1_cmd                                # [MATCH ENDPOINT]
  table_queue: /mig_ep1_table
  resp_queue: /mig_ep1_resp
  client_dir: <tt-llm-engine>/disaggregation/migration/build_RelWithDebInfo/python
```

The shared producer manifests carry **no** `model:` **block** — that is what keeps them model-agnostic. Export
the four model/transport vars so they match the runner, or add a `model:` block for a self-contained file:

```bash
export PREFILL_MODEL=my_model PREFILL_NUM_LAYERS=<L> \
       PREFILL_MAX_SEQ_LEN=<S> PREFILL_CHUNK_SIZE=<C>
```

Unset, the producer falls back to the adapter defaults (61 layers), which will not match a deeper runner —
the ack count is `num_layers × chunks`, so the drain hangs.

For an arbitrary mapping instead of a uniform offset, replace `dst_slot_offset` with `pairs` (equivalent
CLI: `--migrations "0:3,1:2"`). src and dst sets must be disjoint in loopback; duplicate dsts are rejected.

```yaml
  pairs:
    - {src: 0, dst: 3}
    - {src: 1, dst: 2}
```

---



## Gate 0 — Standalone KV PCC (precondition; no migration)

```bash
PREFILL_MODEL=my_model PREFILL_STANDALONE=1 PREFILL_STANDALONE_PCC=1 \
  python -m models.demos.common.prefill.runners.prefill_runner
```

**Expect:** `[kv-pcc]` per-layer lines, min ≥ `PREFILL_STANDALONE_CHUNKED_PCC`. The exact wording is your
model's — the validator lives in your package.

---



## Gate 1 — Mock migration + producer read-back (table addresses; no endpoint)

The runner serializes the KV-chunk table + device map and nothing else; the producer reads each chunk
device-lessly via `read_dram_umd` and PCCs vs golden. Isolates "is `build_kv_chunk_table` correct?".

In the binding, replace the migration block with:

```yaml
  PREFILL_MOCK_MIGRATION: "1"
  PREFILL_MIGRATION_TABLE_PATH: "/tmp/prefill_kv_chunk_table.pb"
  PREFILL_MIGRATION_DEVICE_MAP_PATH: "/tmp/prefill_kv_device_map.json"
  PREFILL_NUM_USERS: "2"
```

In the producer manifest: `check_pcc: true`, `migration: {issue: false}`.

```bash
# Terminal 1 — runner (stays open across producer runs)
$RUN $BINDING $HOST:1

# Terminal 2 — producer
python -m models.demos.common.prefill.runners.prefill_producer --manifest $MANIFEST
```

**Expect:** producer `[producer] KV cache PCC PASSED`.

---



## Gate 2 — Loopback migration (endpoint + runner + producer)

The real DRAM→transport→DRAM copy. Loopback means `dest_endpoint_id` equals the endpoint's own id: src and
dst slots share one table, routed through the endpoint's internal A→B worker pair. Three processes, one
host — the shmem queues and UMD access are host-local, so all three must co-locate.

**Launch order: A → B (wait for** `WORKER_READY`**) → C.** Between runs, clear stale state:

```bash
pkill -f migration_endpoint ; pkill prte
rm -f /dev/shm/tt_h2d_* /dev/shm/tt_prefill_layer_acks_* /tmp/migration_done.sentinel*
```

```bash
# ---- Terminal A — migration endpoint ----
cd <tt-llm-engine>/disaggregation/migration
./launch_migration_endpoints.sh --name_server_host $HOST \
    --prefill_hosts $HOST --prefill_endpoint_id 1

ls /dev/shm/mig_ep1_*      # all three queues must exist BEFORE starting B

# ---- Terminal B — runner (wait for WORKER_READY before starting C) ----
cd $TT_METAL_HOME && $RUN $BINDING $HOST:1

# ---- Terminal C — producer ----
cd $TT_METAL_HOME
python -m models.demos.common.prefill.runners.prefill_producer --manifest $MANIFEST
```

The producer prefills, drains the acks, migrates each pair, then writes the DONE sentinel; the runner is
polling for it and validates on-device.

**2a — pairwise** (`PREFILL_MIGRATE_PAIRWISE=1`): each dst asserted bit-equal to its src, golden-free and
length-agnostic. Also catches cross-talk between concurrent migrations.
**Expect (terminal B):** `[kv-migrate-validate] AFTER pairwise src=0 dst=2 min_pcc=…` per pair, then
`ALL <N> pair(s) dst==src PASSED` (≥ `PREFILL_MIGRATE_PAIRWISE_PCC`, default 0.99).

**2b — burst** (omit `PREFILL_MIGRATE_PAIRWISE`): src and dst both PCC'd vs the same golden.
**Expect (terminal B):** `BEFORE src_slot=0 …` + `AFTER dst_slot=2 …`, then `ALL <N> migrated pair(s) PASSED`.

The two are either/or per run — run twice for both signals.

---



## What each gate needs from the runtime

Validation is driven by the optional runtime hooks in [ADDING_A_PREFILL_MODEL.md](ADDING_A_PREFILL_MODEL.md)
§2. Implement only what the gates you want require.


| Gate               | Hook                   | Signature requirement                                                       |
| ------------------ | ---------------------- | --------------------------------------------------------------------------- |
| 0                  | `kv_cache_pcc_check`   | accepts `trace_dir`, `first_layer_idx`                                      |
| 1                  | `build_kv_chunk_table` | serializes the block-cyclic layout; issues no comms                         |
| 2a pairwise        | `read_slot_kv`         | returns bare host tensors, `[num_layers, heads(or 1), seq_cache, head_dim]` |
| 2b burst           | `kv_cache_pcc_check`   | **also** accepts `real_len=`                                                |
| 2a + golden anchor | `kv_cache_pcc_check`   | **also** accepts `pt_path_override=`                                        |


`runners/validation.py` calls these by keyword, so a runtime missing `real_len` or `pt_path_override`
raises `TypeError` before any PCC runs — it looks like a crash, not a validation failure.

---



## Values that must agree across the three processes


| Value        | Runner `global_env`                        | Producer manifest                  | Endpoint                           |
| ------------ | ------------------------------------------ | ---------------------------------- | ---------------------------------- |
| H2D socket   | `PREFILL_H2D_SERVICE_ID`                   | `transport.h2d_service_id`         | —                                  |
| layer depth  | `PREFILL_NUM_LAYERS`                       | same var, exported                 | —                                  |
| chunk size   | `PREFILL_CHUNK_SIZE`                       | same var, exported                 | —                                  |
| mesh         | `PREFILL_SP` / `PREFILL_TP`                | `transport.sp` / `.tp`             | —                                  |
| KV table     | `PREFILL_MIGRATION_TABLE_PATH`             | `migration.table_path`             | —                                  |
| sentinel     | `MIGRATION_DONE_FILE`                      | `migration.done_file`              | —                                  |
| queues       | `PREFILL_MIGRATION_{CMD,TABLE,RESP}_QUEUE` | `migration.{cmd,table,resp}_queue` | `--prefill-{cmd,table,resp}-queue` |
| client `.so` | `PREFILL_MIGRATION_CLIENT_DIR`             | `migration.client_dir`             | —                                  |
| chunk budget | `PREFILL_STANDALONE_CHUNKED_NCHUNKS`       | `num_users` × `chunks`             | —                                  |
| slot count   | `PREFILL_NUM_USERS`                        | ≥ max dst slot + 1                 | —                                  |
