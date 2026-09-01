# MiniMax-M3 prefill KV-migration — test command sheet

How to validate M3's KV-cache migration on a **single Blackhole galaxy**, no decode side — single-rank
(SP=8 × TP=4) and 2-stage intragalaxy pipeline (two Z-linked 4×4 sub-meshes, 30 layers per stage; gates
P1/P2 below). This is the M3 instance of `models/demos/common/prefill/docs/PREFILL_MIGRATION_TESTING.md`
— that document explains the mechanism and the config layering; this one is the copy-pasteable version
with M3's paths, layer count, and manifests filled in.

M3 stores three cache tensors (`k`, `v`, `index_k`). `k`/`v` are TP-head-sharded (column `c` holds head
`c`); `index_k` is replicated across the TP columns. The migration worker treats a chunk's device group as
*replicas*, so the KV chunk table encodes **one config per (tensor, head-shard)** — 9 configs
(`k_h0..3`, `v_h0..3`, `index_k`) — with single-member device groups for K/V and a replica group for
`index_k` (`tt/runners/kv_chunk_table.py`). The migration layer needs no C++ change.

| Gate | What it proves | Needs |
|------|----------------|-------|
| **0 — standalone PCC** | Prefill writes correct K / V / index_k vs golden | tt-metal tree only |
| **1 — mock migration** | The 9-config address table is correct, read device-lessly | tt-metal tree only |
| **2 — loopback migration** | The real DRAM → DCN → DRAM copy, and migrated-KV accuracy | + tt-llm-engine migration layer |
| **P0 — merged-table unit test** | The multi-stage merge's address math, device-free | tt-metal tree only, no device |
| **P1 — pipeline mock migration** | The MERGED 60-layer table over 2 stages, read device-lessly | tt-metal tree only |
| **P2 — pipeline loopback migration** | The real copy driven off the merged table | + tt-llm-engine migration layer |

(The former dependency on `nzhao/non-mla-migration-fixes` landed as #52113.)

---

## Config files

| File | Owns |
|------|------|
| `manifests/minimax_m3.json` | model: `PREFILL_MODEL`, 60 layers, `M3_INDEX_CACHE_BF16` |
| `manifests/m3_binding_mock_migration_1rank.yaml` | Gate 1 runner: 1-rank topology + mock-migration env |
| `manifests/m3_binding_loopback_migration_1rank.yaml` | Gate 2 runner: 1-rank topology + real-migration env |
| `manifests/m3_producer_mock_migration.yaml` | Gate 1 producer: 2 slots × 2 chunks, golden PCC |
| `manifests/m3_producer_loopback_migration.yaml` | Gate 2 driver: prefill 0,1 → migrate to 2,3 |
| `manifests/m3_binding_mock_migration_intragalaxy_2rank.yaml` | Gate P1 runner: 2-stage intragalaxy + merged mock |
| `manifests/m3_binding_loopback_migration_intragalaxy_2rank.yaml` | Gate P2 runner: 2-stage intragalaxy + real migration |
| `manifests/m3_producer_mock_migration_2rank.yaml` | Gate P1 producer (sp=4, merged table paths) |
| `manifests/m3_producer_loopback_migration_2rank.yaml` | Gate P2 driver (sp=4, merged table paths) |

The rank bindings and producer manifests in `models/demos/common/prefill/runners/{topology_configuration,producer_manifests}/`
are model-agnostic by design, so M3's versions live here instead. Everything M3-specific is in these
files; the gate commands below take no per-run env.

Every `PREFILL_*` the runner needs must live in the binding: it runs under `tt-run`, which forwards only
`TT_`, `ARCH_`, `WH_`, `TTNN_`, `DEEPSEEK_` and `MESH_` prefixed shell variables
(`ENV_PASSTHROUGH_PREFIXES` in `ttnn/ttnn/distributed/ttrun.py`). The producer and the migration driver are
ordinary processes, so for them exported env wins over the manifest.

---

## 0. Shared setup (every terminal)

```bash
cd /data/philei/tt-metal
export TT_METAL_HOME="$PWD" PYTHONPATH="$PWD"
source python_env/bin/activate
export HOST=$(hostname -s)

export RUN=./models/demos/common/prefill/runners/run_pipeline_prefill.sh
export M3=models/demos/minimax_m3/tt/runners/manifests
export ENGINE=/data/philei/tt-metal/tt-llm-engine
export MIG="$ENGINE/disaggregation/migration/build_RelWithDebInfo"

# Golden trace (block-sparse MSA, separate_k_v — valid at full length). The manifests here are wired for
# longbook_10240 (10240 tok -> 2 chunks). For another length, change PREFILL_MAX_SEQ_LEN + PREFILL_TRACE_DIR
# in the binding and workload.chunks + model.max_seq_len in the producer manifest.
#   longbook_5120  (5120 tok  -> 1 chunk)
#   longbook_10240 (10240 tok -> 2 chunks)   <- what the manifests use
#   longbook_56320 (55218 tok -> 11 chunks)  <- full length
export GOLDEN=/data/philei/models/minimax-m3-prefill-cache/golden/longbook_10240
```

The checkpoint (`/mnt/models/MiniMaxAI/MiniMax-M3-ref/`, config + tilized weight cache) comes from the
adapter default; override with `PREFILL_HF_MODEL` / `TT_CACHE_PATH` in the binding's `global_env`, not the
shell. Shape constraints: `MAX_SEQ_LEN % CHUNK_SIZE == 0` and `CHUNK_SIZE % (SP*32) == 0`; 5120 satisfies
both at SP=8.

---

## Gate 0 — standalone KV PCC (precondition, no migration)

The runner's standalone mode was removed (#52213), so the bare-process precondition is the M3 harness
test — same golden trace, per-layer KV PCC, no socket/second process:

```bash
env PREFILL_CHUNKED=1 PREFILL_CHUNK_SIZE=5120 \
    PREFILL_TRACE_DIR=$GOLDEN \
    python3 models/demos/minimax_m3/tests/galaxy_prefill_kv_pcc.py
```

**Expect:** per-layer PCC lines and a min across 60 layers for K / V / index_k at or above
`PREFILL_STANDALONE_CHUNKED_PCC` (default `0.88`).

---

## Gate 1 — mock migration + producer read-back (table addresses only)

The cheapest isolated check of `build_kv_chunk_table`: the runner serializes the 9-config table and the
fabric-node→ASIC device map, and the producer reads each chunk **device-lessly** (`read_dram_umd`, the same
UMD path the migration worker uses) and PCCs it against golden. No endpoint, no worker, no MPI.

```bash
# ---- Terminal 1 — runner (stays up across producer runs) ----
$RUN $M3/m3_binding_mock_migration_1rank.yaml $HOST:1

# ---- Terminal 2 — producer ----
python -m models.demos.common.prefill.runners.prefill_producer \
  --manifest $M3/m3_producer_mock_migration.yaml
```

**Expect:** runner `[mock-migration] KV chunk table -> /tmp/m3_kv_chunk_table.pb, device map -> …`;
producer `[producer] layer acks 240/240`, per-layer `K=… V=… index_k=…`, then
`[producer] slot N M3 KV PCC over [0,10240) across 60 layers` and `[producer] KV cache PCC PASSED`
(threshold `PREFILL_STANDALONE_CHUNKED_PCC`, producer default `0.93` — different from the runner's `0.88`
default for the same variable).

`index_k` is PCC'd for every layer whose golden carries `index_k_cache_layer_<L>` (dense layers carry none
and are skipped); the read-back picks the bf16 or bf8 decoder from the config's chunk size, so
`M3_INDEX_CACHE_BF16` must match on both sides.

`prefill_producer` here, deliberately: this gate migrates nothing.

The runner's request loop is unbounded, so it stays up after the producer exits — re-run the producer as
often as you like, then stop it with Ctrl-C, or set `PREFILL_SEND_SHUTDOWN: "1"` in the producer manifest's
`env:` block for a graceful drain-and-exit on the last push.

---

## Gate 2 — full loopback migration (endpoint + runner + driver)

Three processes on one host — the shm queues and UMD access are host-local, so they must co-locate. Loopback
means the destination endpoint id equals the endpoint's own, routed through its internal A→B worker pair.

**One endpoint launch per runner launch.** A worker terminates on a second `SET_TABLE`
(`control_thread.cpp:1368`), and the runner publishes a table at startup, so a second runner against a live
endpoint kills both workers.

```bash
# ---- Between runs: stop terminal A (Ctrl-C), then clear ----
pkill -f migration_endpoint ; pkill prun ; pkill prted ; pkill prte
rm -f /dev/shm/mig_ep1_* /dev/shm/ep_1_[ab]_* \
      /dev/shm/tt_h2d_* /dev/shm/tt_d2h_* /dev/shm/tt_prefill_layer_acks_* \
      /tmp/m3_kv_chunk_table.pb /tmp/m3_kv_device_map.json /tmp/m3_migration_done.sentinel

# ---- Terminal A — migration endpoint (leave running) ----
cd /data/philei/tt-metal && ./m3_gate2_endpoint.sh

# ---- Gate check — must print READY before terminal B ----
cd /data/philei/tt-metal && ./m3_gate2_check.sh

# ---- Terminal B — runner ----
$RUN $M3/m3_binding_loopback_migration_1rank.yaml $HOST:1

# ---- Terminal C — prefill + migrate (only after B prints WORKER_READY + LayerAck ready) ----
export PREFILL_MIGRATION_CLIENT_DIR="$MIG/python"
python -m models.demos.common.prefill.runners.migration_driver \
  --manifest $M3/m3_producer_loopback_migration.yaml
```

`m3_gate2_endpoint.sh` wraps `launch_migration_endpoints.sh` with two things this cluster requires: the
worker's `TT_METAL_HOME` (the engine's submodule, the tree it was built against), and a Slurm-detached
`prte` — the endpoint's two workers need 2 launcher slots on the host, and Slurm advertises a whole galaxy
as `CPUTot=1`, so `prte` must take its pool from `--host <node>:2` rather than from the allocation. It
preflights a 2-rank placement before launching. The workers then outlive the allocation; the `pkill` line
above is the teardown.

The `ep_1_*` queue names are fixed per endpoint id, so a leftover pair from a killed run collides with the
next endpoint's — `launch_migration_endpoints.sh` clears only the outward `/mig_ep1_*` set. The KV table
must go too: the driver waits only for that file to *exist*.

Terminal B before starting C (the table build is 691200 entries set one Python call at a time — expect
minutes of silence first):

```
[migration] delivered 32 local device-map entries -> /ep_1_a_cmd     (and again for /ep_1_b_cmd)
[migration] KV chunk address table serialized to /tmp/m3_kv_chunk_table.pb (configs=9, entries=691200)
[migration] WORKER_READY: table=/tmp/m3_kv_chunk_table.pb
[migration] LayerAck channel ready at /tt_prefill_layer_acks_m3_prefill
```

**Every verdict is logged by the DRIVER (terminal C), not the runner** — the runner-side
`[kv-migrate-validate]` / `PREFILL_MIGRATE_PAIRWISE` validation was removed with the runner self-test
(#52214). The driver's two read-backs (device-less, over UMD via the published table + device map):

### 2a — destination bytes (`--verify-migration=dst-bytes`, the default)

Each dst is asserted byte-equal to its src across all 9 configs — golden-free, length-agnostic, and it
catches cross-talk between concurrent migrations.

### 2b — destination golden (`--verify-migration=dst-golden`, or `both`)

The dst slots are PCC'd against the golden trace (`workload.check_pcc: true` covers the SOURCE slots the
same way). `dst-golden` needs the migrated range to span the full golden length, or it PCCs unwritten
memory — the driver guards this and says so.

---

## Gates P0/P1/P2 — 2-stage intragalaxy pipeline (still one galaxy)

The galaxy is carved into two Z-linked 4×4 sub-meshes (rank 0 = layers 0–29, rank 1 = 30–59; the
TT_VISIBLE_DEVICES splits come from `topology_configuration/pipeline_prefill_request_intragalaxy_2rank.yaml`).
Migration-wise everything is the same flow at 2 ranks: both ranks join the per-cache stage-layout
all-gathers (`kv_migration_stages` reports one `KvCacheStage` per cache — k, v, index_k), rank 0
builds ONE merged 60-layer 9-config table (each stage's chunks at ITS cache's base address, global layer
indices) and publishes it. Both ranks are on this host, so the producer/driver read-backs cover all 60
layers once they merge the two rank-scoped device maps.

Because num_ranks>1, the table path must be on shared storage — the bindings use
`/data/philei/tmp/m3_kv_chunk_table_pp.pb` (`mkdir -p /data/philei/tmp` once). The weight cache for the
4×4 mesh (`tensor_cache_bfp8_MeshShape(4,4)`) populates slowly on first use.

### P0 — device-free merged-table unit test (no hardware)

```bash
pytest models/demos/minimax_m3/tests/test_kv_chunk_table_merge.py -q
```

### P1 — pipeline mock migration (merged table + producer read-back, no endpoint)

```bash
# ---- Terminal 1 — 2-rank runner ----
$RUN $M3/m3_binding_mock_migration_intragalaxy_2rank.yaml $HOST:2

# ---- Terminal 2 — producer (after rank 0 logs the merged table) ----
python -m models.demos.common.prefill.runners.prefill_producer \
  --manifest $M3/m3_producer_mock_migration_2rank.yaml
```

**Expect:** rank 0 `[mock-migration] merged KV chunk table -> /data/philei/tmp/m3_kv_chunk_table_pp.pb`
(configs=9, layers=60); each rank `local device map -> /tmp/m3_kv_device_map_r<rank>.json`; producer
merges both maps (`merged 2 device maps: 32 chips total`), then per-layer `K=… V=… index_k=…` across
`60/60 local layers` and `KV cache PCC PASSED`.

### P2 — pipeline loopback migration (endpoint + 2-rank runner + driver)

Same three-terminal flow as Gate 2 — the endpoint launch (`./m3_gate2_endpoint.sh`), the readiness check
(`./m3_gate2_check.sh`), and the teardown are unchanged (one host, endpoint id 1). Add the pipeline
artifacts to the between-runs cleanup:

```bash
rm -f /tmp/m3_kv_device_map_r*.json /data/philei/tmp/m3_kv_chunk_table_pp.pb
```

```bash
# ---- Terminal B — 2-rank runner ----
$RUN $M3/m3_binding_loopback_migration_intragalaxy_2rank.yaml $HOST:2

# ---- Terminal C — prefill + migrate (after B prints WORKER_READY + LayerAck ready) ----
export PREFILL_MIGRATION_CLIENT_DIR="$MIG/python"
python -m models.demos.common.prefill.runners.migration_driver \
  --manifest $M3/m3_producer_loopback_migration_2rank.yaml
```

**Expect:** both ranks deliver their device maps to the co-located workers; rank 0
`[migration] merged KV chunk table …` then `WORKER_READY`; driver `MIGRATE slot 0->2 / 1->3 complete`,
source `check_pcc` over 60 layers, and `--verify-migration=dst-bytes` byte-equal across all 9 configs.
Run once more with `--verify-migration=dst-golden` for the golden-anchored destination check.

The merged table has 2× the single-rank entry count spread over both stages; the one-`table.set()`-per-
entry Python build still takes minutes of silence in terminal B before `WORKER_READY`.

---

## tt-llm-engine — build and configure

Only the **migration layer** is needed (`migration_endpoint`, `migration_worker`, `_migration_client`);
`build-full` / `prefill_scheduler_driver` are not, since terminal C is a python process. It requires a
tt-metal build carrying the `internal/disaggregation` headers.

Build against the engine's **bundled tt-metal submodule**, which is what the CMake defaults assume. The pin
(`3675a4e3d63`) carries those headers, and both the header and `kv_chunk_address_table.proto` are
byte-identical to `/data/philei/tt-metal`'s, so the table the python runner writes and the table the worker
parses agree either way.

```bash
# 1. Build the bundled submodule. Creates tt-metal/build_Release and a `build` symlink to it. Python
#    bindings are unused by the migration layer, so skipping them cuts the largest part of the build.
cd $ENGINE/tt-metal
./build_metal.sh -c --without-python-bindings

# 2. Migration layer. The old build dir must go — its cache points at another metal tree, and mixing
#    objects compiled against two trees is not safe. No TT_METAL_* env: the defaults resolve TT_METAL_DIR
#    to $ENGINE/tt-metal and TT_METAL_BUILD_DIR to its `build` symlink.
cd $ENGINE
rm -rf disaggregation/migration/build_RelWithDebInfo
env -u TT_METAL_DIR -u TT_METAL_BUILD_DIR \
  ./build_migration_layer.sh --build-type RelWithDebInfo --jobs $(nproc)
```

Check two lines of the cmake output:

```
-- protobuf: using tt-metal build (…)
-- migration_worker: device DRAM support enabled (pure-UMD; libtt_metal NOT linked, detected at
   …/tt-llm-engine/tt-metal/build/tt_metal/libtt_metal.so)
```

Device support is a *presence probe* for `libtt_metal.so` in the metal build dir, and step 1's failures are
swallowed by `build_migration_layer.sh`'s `|| true` — skip or fail step 1 and you silently get a synthetic
worker (SimulatedDram) that migrates nothing real. `readelf -d $MIG/bin/migration_worker | grep RUNPATH`
should resolve into `$ENGINE/tt-metal/build/…`.

Outputs land in `$MIG/bin/{migration_endpoint,migration_worker}` and `$MIG/python/_migration_client*.so`; the
binding's `PREFILL_MIGRATION_CLIENT_DIR` and the terminal-C export both point at that `python/` dir.

Configuration notes:

* **Transport.** `disaggregation/migration/.roce_env` is present, so the launcher defaults to RoCEv2 and
  hard-fails if the RoCE device or the UCX/PRRTE build is unusable. Single-host loopback does not need it;
  `m3_gate2_endpoint.sh` passes `--tcp-transport`. To use RoCE instead, run `./setup_roce_mpi.sh` first.
* **Queue names.** `--prefill_endpoint_id 1` gives `/mig_ep1_{cmd,table,resp}`. The runner's own defaults are
  the older `/prefill_mig_{cmd,tbl,rsp}_1`, so the binding sets `PREFILL_MIGRATION_{CMD,TABLE,RESP}_QUEUE`
  explicitly. Those are the master-only SET_TABLE / WORKER_READY channel; the device map goes to the
  workers' own `/dev/shm/ep_1_{a,b}_*` queues, which the runner discovers by globbing.

---

## Values that must agree across the three processes

| Value | M3 | Runner (`global_env`) | Producer / driver manifest |
|-------|-----|----------------------|----------------------------|
| model | `minimax_m3` | via `PREFILL_MANIFEST` | `model.variant` |
| layers | 60 | via `PREFILL_MANIFEST` | `model.num_layers` |
| chunk size | 5120 | `PREFILL_CHUNK_SIZE` | `model.chunk_size` |
| cache length | 10240 | `PREFILL_MAX_SEQ_LEN` | `model.max_seq_len` |
| mesh | 8 × 4 | `PREFILL_SP` / `PREFILL_TP` | `transport.sp` / `.tp` |
| H2D socket | `m3_prefill` | `PREFILL_H2D_SERVICE_ID` | `transport.h2d_service_id` |
| index dtype | bf16 | `M3_INDEX_CACHE_BF16` | `env.M3_INDEX_CACHE_BF16` |
| KV table | `/tmp/m3_kv_chunk_table.pb` | `PREFILL_MIGRATION_TABLE_PATH` | `migration.table_path` |
| sentinel | `/tmp/m3_migration_done.sentinel` | `MIGRATION_DONE_FILE` | `migration.done_file` |
| queues | `/mig_ep1_*` | `PREFILL_MIGRATION_{CMD,TABLE,RESP}_QUEUE` | `migration.{cmd,table,resp}_queue` |
| slot count | 4 (0,1 → 2,3) | `PREFILL_NUM_USERS` | `workload.num_users` = 2 src |
| chunk budget | 4 | `PREFILL_STANDALONE_CHUNKED_NCHUNKS` | `num_users` × `chunks` |

`PREFILL_STANDALONE_CHUNKED_NCHUNKS` is the one worth double-checking: the driver never sends a shutdown
push, so the runner uses that count to know prefill is done before it polls the sentinel. Too low and it
exits mid-prefill; too high and it blocks on chunks that never come — no PCC either way.

---

## Notes / gotchas

- **`num_key_value_heads` must equal `TP`** (head `h` → column `h`); the table builder asserts it. M3 = 4 = TP.
- **Config-id order is the src↔dst contract.** Loopback is self-consistent (one table is both source and
  destination). A real prefill→decode run needs the decode endpoint to publish its configs in the same order
  (`k_h0..N-1`, `v_h0..N-1`, `index_k`). Not exercised here.
- **The JSON device map is written on BOTH paths now.** The real-migration branch delivers the map to the
  workers over the client AND writes the sidecar, so producer-side `check_pcc` and the driver's
  `--verify-migration` read-backs work in Gate 2 too. With num_ranks>1 each rank writes a rank-scoped
  sidecar (`<stem>_r<rank>.json`); readers glob and merge them — clear stale `_r*.json` between runs whose
  topology changed.
- **Table build is ~9× the DeepSeek entry count** (9 configs, 691200 entries). It runs once at startup;
  slowness there is Python `table.set()` overhead, not a correctness problem.
- `_migration_client not importable` → `PREFILL_MIGRATION_CLIENT_DIR` must point at `$MIG/python`, in the
  binding for the runner **and** exported for the driver.
